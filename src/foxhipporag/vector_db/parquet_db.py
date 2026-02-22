"""
Parquet 向量数据库后端

基于 Parquet 文件格式的本地向量存储实现。
这是默认的后端，无需额外依赖。
"""

import os
import threading
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from copy import deepcopy

import numpy as np
import pandas as pd

from .base import BaseVectorDB

logger = logging.getLogger(__name__)


class ParquetVectorDB(BaseVectorDB):
    """
    基于 Parquet 文件的向量存储
    
    特性：
    - 使用 Parquet 格式存储，高效的列式存储
    - 支持线程安全的并发访问
    - 支持类级别缓存
    - 支持增量保存
    """
    
    # 类级别缓存，避免重复加载同一文件
    _cache: Dict[str, Dict[str, Any]] = {}
    _cache_lock = threading.Lock()
    
    def __init__(
        self,
        embedding_dim: int,
        namespace: str = "default",
        db_path: str = "./vector_db",
        auto_save_threshold: int = 100,
        enable_cache: bool = True,
        **kwargs
    ):
        """
        初始化 Parquet 向量存储
        
        Args:
            embedding_dim: 嵌入向量维度
            namespace: 命名空间，用于数据隔离
            db_path: 数据库存储路径
            auto_save_threshold: 自动保存的阈值（修改次数）
            enable_cache: 是否启用缓存
            **kwargs: 额外参数
        """
        super().__init__(embedding_dim, namespace, **kwargs)
        
        self.db_path = db_path
        self.auto_save_threshold = auto_save_threshold
        self.enable_cache = enable_cache
        
        # 线程安全锁
        self._lock = threading.RLock()
        
        # 增量保存计数器
        self._pending_changes = 0
        
        # 数据存储
        self._ids: List[str] = []
        self._vectors: List[np.ndarray] = []
        self._metadatas: List[Dict[str, Any]] = []
        
        # 索引
        self._id_to_idx: Dict[str, int] = {}
        
        # 向量矩阵缓存
        self._matrix_cache: Optional[np.ndarray] = None
        self._cache_dirty: bool = True
        
        # 确保目录存在
        if not os.path.exists(db_path):
            logger.info(f"创建向量数据库目录: {db_path}")
            os.makedirs(db_path, exist_ok=True)
        
        # 构建文件路径
        self._filepath = os.path.join(db_path, f"vdb_{namespace}.parquet")
        
        # 加载现有数据
        self._load()
        
        self._initialized = True
    
    def _load(self) -> bool:
        """
        从文件加载数据
        
        Returns:
            bool: 加载是否成功
        """
        with self._lock:
            cache_key = self._filepath
            
            # 尝试从缓存加载
            if self.enable_cache:
                with ParquetVectorDB._cache_lock:
                    if cache_key in ParquetVectorDB._cache:
                        cached = ParquetVectorDB._cache[cache_key]
                        self._ids = cached['ids'].copy()
                        self._vectors = [v.copy() for v in cached['vectors']]
                        self._metadatas = deepcopy(cached['metadatas'])
                        logger.info(f"从缓存加载了 {len(self._ids)} 条记录")
                        self._rebuild_index()
                        return True
            
            # 从文件加载
            if os.path.exists(self._filepath):
                try:
                    df = pd.read_parquet(self._filepath)
                    self._ids = df['id'].tolist()
                    self._vectors = [np.array(v, dtype=np.float32) for v in df['vector'].tolist()]
                    self._metadatas = df['metadata'].tolist() if 'metadata' in df.columns else [{} for _ in self._ids]
                    
                    # 更新缓存
                    if self.enable_cache:
                        with ParquetVectorDB._cache_lock:
                            ParquetVectorDB._cache[cache_key] = {
                                'ids': self._ids.copy(),
                                'vectors': [v.copy() for v in self._vectors],
                                'metadatas': deepcopy(self._metadatas)
                            }
                    
                    logger.info(f"从 {self._filepath} 加载了 {len(self._ids)} 条记录")
                    self._rebuild_index()
                    return True
                    
                except Exception as e:
                    logger.error(f"加载数据失败: {e}")
                    self._ids, self._vectors, self._metadatas = [], [], []
                    self._rebuild_index()
                    return False
            else:
                # 文件不存在，初始化为空
                self._ids, self._vectors, self._metadatas = [], [], []
                self._rebuild_index()
                return True
    
    def _save(self) -> bool:
        """
        保存数据到文件
        
        Returns:
            bool: 保存是否成功
        """
        with self._lock:
            start_time = time.time()
            
            try:
                df = pd.DataFrame({
                    'id': self._ids,
                    'vector': self._vectors,
                    'metadata': self._metadatas
                })
                df.to_parquet(self._filepath, index=False)
                
                # 更新缓存
                if self.enable_cache:
                    cache_key = self._filepath
                    with ParquetVectorDB._cache_lock:
                        ParquetVectorDB._cache[cache_key] = {
                            'ids': self._ids.copy(),
                            'vectors': [v.copy() for v in self._vectors],
                            'metadatas': deepcopy(self._metadatas)
                        }
                
                self._pending_changes = 0
                self._cache_dirty = True
                
                save_time = time.time() - start_time
                logger.info(f"保存 {len(self._ids)} 条记录到 {self._filepath}，耗时 {save_time:.2f}s")
                return True
                
            except Exception as e:
                logger.error(f"保存数据失败: {e}")
                return False
    
    def _rebuild_index(self):
        """重建ID索引"""
        self._id_to_idx = {id_: idx for idx, id_ in enumerate(self._ids)}
        self._cache_dirty = True
    
    def insert(
        self,
        ids: List[str],
        vectors: List[np.ndarray],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> bool:
        """
        插入向量数据
        
        注意：如果ID已存在，将跳过该记录
        """
        with self._lock:
            # 过滤已存在的ID
            new_items = [
                (id_, vec, meta if metadatas else {})
                for id_, vec, meta in zip(
                    ids, 
                    vectors, 
                    metadatas or [{}] * len(ids)
                )
                if id_ not in self._id_to_idx
            ]
            
            if not new_items:
                logger.info("所有记录已存在，跳过插入")
                return True
            
            # 添加新记录
            for id_, vec, meta in new_items:
                self._ids.append(id_)
                self._vectors.append(np.array(vec, dtype=np.float32))
                self._metadatas.append(meta)
            
            self._rebuild_index()
            self._pending_changes += len(new_items)
            
            logger.info(f"插入了 {len(new_items)} 条新记录")
            return self._save()
    
    def upsert(
        self,
        ids: List[str],
        vectors: List[np.ndarray],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> bool:
        """
        插入或更新向量数据
        """
        with self._lock:
            metadatas = metadatas or [{}] * len(ids)
            
            for id_, vec, meta in zip(ids, vectors, metadatas):
                if id_ in self._id_to_idx:
                    # 更新现有记录
                    idx = self._id_to_idx[id_]
                    self._vectors[idx] = np.array(vec, dtype=np.float32)
                    self._metadatas[idx] = meta
                else:
                    # 插入新记录
                    self._ids.append(id_)
                    self._vectors.append(np.array(vec, dtype=np.float32))
                    self._metadatas.append(meta)
            
            self._rebuild_index()
            self._pending_changes += len(ids)
            
            logger.info(f"更新/插入了 {len(ids)} 条记录")
            return self._save()
    
    def delete(self, ids: List[str], **kwargs) -> bool:
        """
        删除指定ID的向量
        """
        with self._lock:
            # 找到要删除的索引
            indices_to_delete = [
                self._id_to_idx[id_] 
                for id_ in ids 
                if id_ in self._id_to_idx
            ]
            
            if not indices_to_delete:
                logger.info("没有找到要删除的记录")
                return True
            
            # 从大到小排序，避免删除时索引变化
            indices_to_delete.sort(reverse=True)
            
            for idx in indices_to_delete:
                self._ids.pop(idx)
                self._vectors.pop(idx)
                self._metadatas.pop(idx)
            
            self._rebuild_index()
            self._pending_changes += len(indices_to_delete)
            
            logger.info(f"删除了 {len(indices_to_delete)} 条记录")
            return self._save()
    
    def get_vectors(
        self,
        ids: List[str],
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        根据ID获取向量
        """
        with self._lock:
            return {
                id_: self._vectors[self._id_to_idx[id_]]
                for id_ in ids
                if id_ in self._id_to_idx
            }
    
    def get_metadatas(
        self,
        ids: List[str],
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """
        根据ID获取元数据
        """
        with self._lock:
            return {
                id_: deepcopy(self._metadatas[self._id_to_idx[id_]])
                for id_ in ids
                if id_ in self._id_to_idx
            }
    
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        向量相似度搜索（使用余弦相似度）
        """
        with self._lock:
            if not self._ids:
                return []
            
            # 获取向量矩阵
            matrix = self._get_matrix()
            
            # 归一化查询向量
            query = np.array(query_vector, dtype=np.float32)
            query_norm = np.linalg.norm(query)
            if query_norm > 0:
                query = query / query_norm
            
            # 计算相似度
            similarities = np.dot(matrix, query)
            
            # 应用元数据过滤
            if filter_dict:
                valid_indices = []
                for idx, meta in enumerate(self._metadatas):
                    if all(meta.get(k) == v for k, v in filter_dict.items()):
                        valid_indices.append(idx)
                
                if valid_indices:
                    filtered_similarities = [(idx, similarities[idx]) for idx in valid_indices]
                    filtered_similarities.sort(key=lambda x: x[1], reverse=True)
                    top_results = filtered_similarities[:top_k]
                else:
                    return []
            else:
                # 获取top_k索引
                top_indices = np.argsort(similarities)[::-1][:top_k]
                top_results = [(idx, similarities[idx]) for idx in top_indices]
            
            # 构建结果
            results = []
            for idx, score in top_results:
                results.append((
                    self._ids[idx],
                    float(score),
                    deepcopy(self._metadatas[idx])
                ))
            
            return results
    
    def batch_search(
        self,
        query_vectors: List[np.ndarray],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[List[Tuple[str, float, Dict[str, Any]]]]:
        """
        批量向量相似度搜索
        """
        results = []
        for query in query_vectors:
            results.append(self.search(query, top_k, filter_dict, **kwargs))
        return results
    
    def get_all_ids(self, **kwargs) -> List[str]:
        """获取所有向量ID"""
        with self._lock:
            return self._ids.copy()
    
    def count(self, **kwargs) -> int:
        """获取向量总数"""
        with self._lock:
            return len(self._ids)
    
    def clear(self, **kwargs) -> bool:
        """清空所有数据"""
        with self._lock:
            self._ids = []
            self._vectors = []
            self._metadatas = []
            self._rebuild_index()
            
            # 清除缓存
            if self.enable_cache:
                with ParquetVectorDB._cache_lock:
                    if self._filepath in ParquetVectorDB._cache:
                        del ParquetVectorDB._cache[self._filepath]
            
            return self._save()
    
    def save(self, path: Optional[str] = None, **kwargs) -> bool:
        """保存数据到指定路径"""
        if path:
            old_path = self._filepath
            self._filepath = path
            result = self._save()
            self._filepath = old_path
            return result
        return self._save()
    
    def load(self, path: Optional[str] = None, **kwargs) -> bool:
        """从指定路径加载数据"""
        if path:
            old_path = self._filepath
            self._filepath = path
            result = self._load()
            self._filepath = old_path
            return result
        return self._load()
    
    def _get_matrix(self) -> np.ndarray:
        """获取向量矩阵（带缓存）"""
        if self._matrix_cache is None or self._cache_dirty:
            if self._vectors:
                self._matrix_cache = np.vstack(self._vectors)
                # 归一化
                norms = np.linalg.norm(self._matrix_cache, axis=1, keepdims=True)
                norms[norms == 0] = 1  # 避免除零
                self._matrix_cache = self._matrix_cache / norms
            else:
                self._matrix_cache = np.array([]).reshape(0, self.embedding_dim)
            self._cache_dirty = False
        return self._matrix_cache
    
    def clear_cache(self):
        """清除缓存"""
        with ParquetVectorDB._cache_lock:
            if self._filepath in ParquetVectorDB._cache:
                del ParquetVectorDB._cache[self._filepath]
        self._matrix_cache = None
        self._cache_dirty = True
