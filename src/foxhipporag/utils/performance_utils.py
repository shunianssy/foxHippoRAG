"""
性能优化模块

包含多种性能优化组件：
1. PPR结果缓存
2. 嵌入向量缓存
3. 批量操作优化
4. 内存管理优化
"""

import hashlib
import json
import os
import sqlite3
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PPRCache:
    """PPR结果缓存
    
    缓存个性化PageRank计算结果，避免重复计算
    """
    
    def __init__(self, cache_dir: str, max_memory_size: int = 100):
        """
        初始化PPR缓存
        
        Args:
            cache_dir: 缓存目录
            max_memory_size: 内存中最大缓存条目数
        """
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_file = os.path.join(cache_dir, "ppr_cache.sqlite")
        self.memory_cache: OrderedDict = OrderedDict()
        self.max_memory_size = max_memory_size
        self.lock = threading.RLock()
        
        self.hits = 0
        self.misses = 0
        
        self._init_db()
    
    def _init_db(self):
        """初始化数据库"""
        with self.lock:
            conn = sqlite3.connect(self.cache_file)
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS ppr_cache (
                    query_hash TEXT PRIMARY KEY,
                    doc_ids TEXT,
                    doc_scores TEXT,
                    graph_hash TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            c.execute("CREATE INDEX IF NOT EXISTS idx_graph_hash ON ppr_cache(graph_hash)")
            conn.commit()
            conn.close()
    
    def _compute_hash(self, reset_prob: np.ndarray, damping: float, graph_hash: str) -> str:
        """计算缓存键的哈希值"""
        content = f"{graph_hash}:{damping}:{np.array2string(reset_prob, precision=6)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, reset_prob: np.ndarray, damping: float, graph_hash: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        获取缓存的PPR结果
        
        Args:
            reset_prob: 重置概率数组
            damping: 阻尼因子
            graph_hash: 图的哈希值
            
        Returns:
            缓存的(doc_ids, doc_scores)或None
        """
        cache_key = self._compute_hash(reset_prob, damping, graph_hash)
        
        with self.lock:
            if cache_key in self.memory_cache:
                self.memory_cache.move_to_end(cache_key)
                self.hits += 1
                return self.memory_cache[cache_key]
        
        with self.lock:
            conn = sqlite3.connect(self.cache_file)
            c = conn.cursor()
            c.execute(
                "SELECT doc_ids, doc_scores FROM ppr_cache WHERE query_hash = ? AND graph_hash = ?",
                (cache_key, graph_hash)
            )
            row = c.fetchone()
            conn.close()
            
            if row:
                doc_ids = np.array(json.loads(row[0]))
                doc_scores = np.array(json.loads(row[1]))
                
                with self.lock:
                    if len(self.memory_cache) >= self.max_memory_size:
                        self.memory_cache.popitem(last=False)
                    self.memory_cache[cache_key] = (doc_ids, doc_scores)
                
                self.hits += 1
                return doc_ids, doc_scores
        
        self.misses += 1
        return None
    
    def put(self, reset_prob: np.ndarray, damping: float, graph_hash: str,
            doc_ids: np.ndarray, doc_scores: np.ndarray) -> None:
        """
        存储PPR结果到缓存
        
        Args:
            reset_prob: 重置概率数组
            damping: 阻尼因子
            graph_hash: 图的哈希值
            doc_ids: 文档ID数组
            doc_scores: 文档得分数组
        """
        cache_key = self._compute_hash(reset_prob, damping, graph_hash)
        
        with self.lock:
            if len(self.memory_cache) >= self.max_memory_size:
                self.memory_cache.popitem(last=False)
            self.memory_cache[cache_key] = (doc_ids.copy(), doc_scores.copy())
        
        with self.lock:
            conn = sqlite3.connect(self.cache_file)
            c = conn.cursor()
            c.execute(
                """INSERT OR REPLACE INTO ppr_cache 
                   (query_hash, doc_ids, doc_scores, graph_hash) 
                   VALUES (?, ?, ?, ?)""",
                (cache_key, json.dumps(doc_ids.tolist()), json.dumps(doc_scores.tolist()), graph_hash)
            )
            conn.commit()
            conn.close()
    
    def invalidate(self, graph_hash: str = None):
        """
        使缓存失效
        
        Args:
            graph_hash: 如果提供，只使该图的缓存失效
        """
        with self.lock:
            self.memory_cache.clear()
        
        with self.lock:
            conn = sqlite3.connect(self.cache_file)
            c = conn.cursor()
            if graph_hash:
                c.execute("DELETE FROM ppr_cache WHERE graph_hash = ?", (graph_hash,))
            else:
                c.execute("DELETE FROM ppr_cache")
            conn.commit()
            conn.close()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0
            return {
                'memory_size': len(self.memory_cache),
                'max_memory_size': self.max_memory_size,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate
            }


class EmbeddingCache:
    """嵌入向量缓存
    
    缓存文本的嵌入向量，避免重复计算
    """
    
    def __init__(self, cache_dir: str, max_memory_size: int = 1000):
        """
        初始化嵌入缓存
        
        Args:
            cache_dir: 缓存目录
            max_memory_size: 内存中最大缓存条目数
        """
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_file = os.path.join(cache_dir, "embedding_cache.sqlite")
        self.memory_cache: OrderedDict = OrderedDict()
        self.max_memory_size = max_memory_size
        self.lock = threading.RLock()
        
        self.hits = 0
        self.misses = 0
        
        self._init_db()
    
    def _init_db(self):
        """初始化数据库"""
        with self.lock:
            conn = sqlite3.connect(self.cache_file)
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    text_hash TEXT PRIMARY KEY,
                    embedding TEXT,
                    model_name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            c.execute("CREATE INDEX IF NOT EXISTS idx_model_name ON embedding_cache(model_name)")
            conn.commit()
            conn.close()
    
    def _compute_hash(self, text: str, model_name: str) -> str:
        """计算文本哈希"""
        content = f"{model_name}:{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, text: str, model_name: str) -> Optional[np.ndarray]:
        """获取缓存的嵌入向量"""
        cache_key = self._compute_hash(text, model_name)
        
        with self.lock:
            if cache_key in self.memory_cache:
                self.memory_cache.move_to_end(cache_key)
                self.hits += 1
                return self.memory_cache[cache_key]
        
        with self.lock:
            conn = sqlite3.connect(self.cache_file)
            c = conn.cursor()
            c.execute(
                "SELECT embedding FROM embedding_cache WHERE text_hash = ? AND model_name = ?",
                (cache_key, model_name)
            )
            row = c.fetchone()
            conn.close()
            
            if row:
                embedding = np.array(json.loads(row[0]))
                
                with self.lock:
                    if len(self.memory_cache) >= self.max_memory_size:
                        self.memory_cache.popitem(last=False)
                    self.memory_cache[cache_key] = embedding
                
                self.hits += 1
                return embedding
        
        self.misses += 1
        return None
    
    def get_batch(self, texts: List[str], model_name: str) -> Dict[int, np.ndarray]:
        """
        批量获取缓存的嵌入向量
        
        Args:
            texts: 文本列表
            model_name: 模型名称
            
        Returns:
            索引到嵌入向量的映射（只包含缓存命中的）
        """
        results = {}
        uncached_indices = []
        
        with self.lock:
            for i, text in enumerate(texts):
                cache_key = self._compute_hash(text, model_name)
                if cache_key in self.memory_cache:
                    self.memory_cache.move_to_end(cache_key)
                    results[i] = self.memory_cache[cache_key]
                    self.hits += 1
                else:
                    uncached_indices.append(i)
        
        if uncached_indices:
            uncached_texts = [texts[i] for i in uncached_indices]
            cache_keys = [self._compute_hash(t, model_name) for t in uncached_texts]
            
            with self.lock:
                conn = sqlite3.connect(self.cache_file)
                c = conn.cursor()
                placeholders = ','.join(['?' for _ in cache_keys])
                c.execute(
                    f"SELECT text_hash, embedding FROM embedding_cache WHERE text_hash IN ({placeholders}) AND model_name = ?",
                    cache_keys + [model_name]
                )
                rows = c.fetchall()
                conn.close()
                
                db_results = {row[0]: np.array(json.loads(row[1])) for row in rows}
                
                for i, cache_key in zip(uncached_indices, cache_keys):
                    if cache_key in db_results:
                        embedding = db_results[cache_key]
                        results[i] = embedding
                        
                        with self.lock:
                            if len(self.memory_cache) >= self.max_memory_size:
                                self.memory_cache.popitem(last=False)
                            self.memory_cache[cache_key] = embedding
                        
                        self.hits += 1
                    else:
                        self.misses += 1
        
        return results
    
    def put(self, text: str, model_name: str, embedding: np.ndarray) -> None:
        """存储嵌入向量到缓存"""
        cache_key = self._compute_hash(text, model_name)
        
        with self.lock:
            if len(self.memory_cache) >= self.max_memory_size:
                self.memory_cache.popitem(last=False)
            self.memory_cache[cache_key] = embedding.copy()
        
        with self.lock:
            conn = sqlite3.connect(self.cache_file)
            c = conn.cursor()
            c.execute(
                "INSERT OR REPLACE INTO embedding_cache (text_hash, embedding, model_name) VALUES (?, ?, ?)",
                (cache_key, json.dumps(embedding.tolist()), model_name)
            )
            conn.commit()
            conn.close()
    
    def put_batch(self, texts: List[str], model_name: str, embeddings: np.ndarray) -> None:
        """批量存储嵌入向量"""
        with self.lock:
            conn = sqlite3.connect(self.cache_file)
            c = conn.cursor()
            
            for text, embedding in zip(texts, embeddings):
                cache_key = self._compute_hash(text, model_name)
                
                if len(self.memory_cache) >= self.max_memory_size:
                    self.memory_cache.popitem(last=False)
                self.memory_cache[cache_key] = embedding.copy()
                
                c.execute(
                    "INSERT OR REPLACE INTO embedding_cache (text_hash, embedding, model_name) VALUES (?, ?, ?)",
                    (cache_key, json.dumps(embedding.tolist()), model_name)
                )
            
            conn.commit()
            conn.close()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0
            return {
                'memory_size': len(self.memory_cache),
                'max_memory_size': self.max_memory_size,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate
            }


class BatchProcessor:
    """批量处理器
    
    优化批量操作的性能
    """
    
    def __init__(self, batch_size: int = 32, max_workers: int = 8):
        """
        初始化批量处理器
        
        Args:
            batch_size: 批处理大小
            max_workers: 最大工作线程数
        """
        self.batch_size = batch_size
        self.max_workers = max_workers
    
    def process_in_batches(self, items: List[Any], process_fn, desc: str = "Processing") -> List[Any]:
        """
        分批处理项目
        
        Args:
            items: 待处理的项目列表
            process_fn: 处理函数
            desc: 进度条描述
            
        Returns:
            处理结果列表
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm
        
        if not items:
            return []
        
        if len(items) <= self.batch_size:
            return [process_fn(item) for item in tqdm(items, desc=desc)]
        
        batches = [items[i:i + self.batch_size] for i in range(0, len(items), self.batch_size)]
        
        results = [None] * len(items)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_batch = {
                executor.submit(self._process_batch, batch, process_fn, start_idx): (batch, start_idx)
                for start_idx, batch in enumerate(batches)
            }
            
            for future in tqdm(as_completed(future_to_batch), total=len(future_to_batch), desc=desc):
                batch, start_idx = future_to_batch[future]
                try:
                    batch_results = future.result()
                    for i, result in enumerate(batch_results):
                        results[start_idx * self.batch_size + i] = result
                except Exception as e:
                    logger.error(f"Batch processing failed: {e}")
        
        return results
    
    def _process_batch(self, batch: List[Any], process_fn, start_idx: int) -> List[Any]:
        """处理单个批次"""
        return [process_fn(item) for item in batch]


class MemoryManager:
    """内存管理器
    
    监控和管理内存使用
    """
    
    def __init__(self, warning_threshold: float = 0.8, critical_threshold: float = 0.9):
        """
        初始化内存管理器
        
        Args:
            warning_threshold: 警告阈值（内存使用比例）
            critical_threshold: 临界阈值（内存使用比例）
        """
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self._callbacks = []
    
    def add_cleanup_callback(self, callback):
        """添加清理回调函数"""
        self._callbacks.append(callback)
    
    def check_memory(self) -> Tuple[float, str]:
        """
        检查内存使用情况
        
        Returns:
            (内存使用比例, 状态)
        """
        try:
            import psutil
            memory = psutil.virtual_memory()
            usage = memory.percent / 100.0
            
            if usage >= self.critical_threshold:
                status = "critical"
                self._trigger_cleanup()
            elif usage >= self.warning_threshold:
                status = "warning"
            else:
                status = "normal"
            
            return usage, status
        except ImportError:
            return 0.0, "unknown"
    
    def _trigger_cleanup(self):
        """触发清理操作"""
        logger.warning("Memory usage critical, triggering cleanup...")
        for callback in self._callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Cleanup callback failed: {e}")
    
    def get_memory_info(self) -> Dict[str, Any]:
        """获取内存信息"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return {
                'total': memory.total,
                'available': memory.available,
                'used': memory.used,
                'percent': memory.percent,
                'status': self.check_memory()[1]
            }
        except ImportError:
            return {'status': 'psutil not available'}


class PerformanceMonitor:
    """性能监控器
    
    监控和记录性能指标
    """
    
    def __init__(self):
        """初始化性能监控器"""
        self.metrics = {}
        self.lock = threading.Lock()
    
    def record(self, name: str, value: float):
        """记录性能指标"""
        with self.lock:
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append({
                'value': value,
                'timestamp': time.time()
            })
    
    def record_time(self, name: str):
        """记录时间的上下文管理器"""
        class Timer:
            def __init__(self, monitor, metric_name):
                self.monitor = monitor
                self.metric_name = metric_name
                self.start_time = None
            
            def __enter__(self):
                self.start_time = time.time()
                return self
            
            def __exit__(self, *args):
                elapsed = time.time() - self.start_time
                self.monitor.record(self.metric_name, elapsed)
        
        return Timer(self, name)
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """获取指标统计信息"""
        with self.lock:
            if name not in self.metrics:
                return {}
            
            values = [m['value'] for m in self.metrics[name]]
            if not values:
                return {}
            
            return {
                'count': len(values),
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'total': np.sum(values)
            }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """获取所有指标的统计信息"""
        with self.lock:
            return {name: self.get_stats(name) for name in self.metrics}
    
    def clear(self):
        """清除所有指标"""
        with self.lock:
            self.metrics.clear()


global_performance_monitor = PerformanceMonitor()
