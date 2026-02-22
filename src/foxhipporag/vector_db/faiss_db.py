"""
FAISS 向量数据库后端

基于 Facebook AI Similarity Search (FAISS) 的本地向量存储实现。
FAISS 提供高效的向量检索能力，特别适合大规模向量搜索。
"""

import os
import json
import threading
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from copy import deepcopy

import numpy as np

from .base import BaseVectorDB

logger = logging.getLogger(__name__)

# FAISS 可选导入
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS 未安装。请运行: pip install faiss-cpu 或 pip install faiss-gpu")


class FAISSVectorDB(BaseVectorDB):
    """
    基于 FAISS 的向量存储
    
    特性：
    - 高效的向量索引和检索
    - 支持 CPU 和 GPU
    - 支持多种索引类型（Flat, IVF, HNSW 等）
    - 持久化到本地文件
    """
    
    def __init__(
        self,
        embedding_dim: int,
        namespace: str = "default",
        db_path: str = "./vector_db",
        index_type: str = "Flat",
        nlist: int = 100,
        nprobe: int = 10,
        use_gpu: bool = False,
        metric_type: str = "cosine",
        **kwargs
    ):
        """
        初始化 FAISS 向量存储
        
        Args:
            embedding_dim: 嵌入向量维度
            namespace: 命名空间，用于数据隔离
            db_path: 数据库存储路径
            index_type: 索引类型 ('Flat', 'IVF', 'IVFFlat', 'IVFPQ', 'HNSW')
            nlist: IVF 索引的聚类中心数量
            nprobe: 搜索时探测的聚类数量
            use_gpu: 是否使用 GPU
            metric_type: 距离度量类型 ('cosine', 'l2', 'ip')
            **kwargs: 额外参数
        """
        if not FAISS_AVAILABLE:
            raise ImportError(
                "FAISS 未安装。请运行: pip install faiss-cpu 或 pip install faiss-gpu"
            )
        
        super().__init__(embedding_dim, namespace, **kwargs)
        
        self.db_path = db_path
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.use_gpu = use_gpu
        self.metric_type = metric_type
        
        # 线程安全锁
        self._lock = threading.RLock()
        
        # 数据存储
        self._ids: List[str] = []
        self._metadatas: List[Dict[str, Any]] = []
        self._id_to_idx: Dict[str, int] = {}
        
        # FAISS 索引
        self._index: Optional[faiss.Index] = None
        
        # 确保目录存在
        if not os.path.exists(db_path):
            logger.info(f"创建向量数据库目录: {db_path}")
            os.makedirs(db_path, exist_ok=True)
        
        # 构建文件路径
        self._index_path = os.path.join(db_path, f"faiss_{namespace}.index")
        self._meta_path = os.path.join(db_path, f"faiss_{namespace}_meta.json")
        
        # 初始化或加载索引
        self._init_index()
        
        self._initialized = True
    
    def _get_metric(self):
        """获取 FAISS 度量类型"""
        if self.metric_type == "cosine" or self.metric_type == "ip":
            return faiss.METRIC_INNER_PRODUCT
        else:
            return faiss.METRIC_L2
    
    def _init_index(self):
        """初始化 FAISS 索引"""
        # 尝试加载现有索引
        if os.path.exists(self._index_path) and os.path.exists(self._meta_path):
            self._load()
        else:
            # 创建新索引
            self._create_index()
    
    def _create_index(self):
        """创建新的 FAISS 索引"""
        d = self.embedding_dim
        metric = self._get_metric()
        
        if self.index_type == "Flat":
            if metric == faiss.METRIC_INNER_PRODUCT:
                self._index = faiss.IndexFlatIP(d)
            else:
                self._index = faiss.IndexFlatL2(d)
        
        elif self.index_type in ["IVF", "IVFFlat"]:
            # IVF 需要训练，先用 Flat 作为量化器
            quantizer = faiss.IndexFlatL2(d) if metric == faiss.METRIC_L2 else faiss.IndexFlatIP(d)
            self._index = faiss.IndexIVFFlat(quantizer, d, self.nlist, metric)
            self._index.nprobe = self.nprobe
            self._index.is_trained = False  # 需要训练
        
        elif self.index_type == "IVFPQ":
            quantizer = faiss.IndexFlatL2(d) if metric == faiss.METRIC_L2 else faiss.IndexFlatIP(d)
            # PQ 参数：m=8 表示将向量分成8个子向量
            m = min(8, d // 4) if d >= 8 else 1
            self._index = faiss.IndexIVFPQ(quantizer, d, self.nlist, m, 8)
            self._index.nprobe = self.nprobe
            self._index.is_trained = False
        
        else:
            # 默认使用 Flat
            logger.warning(f"未知的索引类型 {self.index_type}，使用 Flat")
            if metric == faiss.METRIC_INNER_PRODUCT:
                self._index = faiss.IndexFlatIP(d)
            else:
                self._index = faiss.IndexFlatL2(d)
        
        # GPU 支持
        if self.use_gpu and hasattr(faiss, 'StandardGpuResources'):
            try:
                res = faiss.StandardGpuResources()
                self._index = faiss.index_cpu_to_gpu(res, 0, self._index)
                logger.info("FAISS 索引已移至 GPU")
            except Exception as e:
                logger.warning(f"无法使用 GPU: {e}")
        
        self._ids = []
        self._metadatas = []
        self._id_to_idx = {}
        
        logger.info(f"创建了新的 FAISS 索引: type={self.index_type}, dim={d}")
    
    def _train_index(self, vectors: np.ndarray):
        """训练 IVF 索引（如果需要）"""
        if hasattr(self._index, 'is_trained') and not self._index.is_trained:
            if len(vectors) >= self.nlist:
                logger.info(f"训练 FAISS 索引，使用 {len(vectors)} 个向量...")
                self._index.train(vectors)
                logger.info("索引训练完成")
            else:
                logger.warning(
                    f"向量数量 ({len(vectors)}) 少于聚类数 ({self.nlist})，"
                    f"跳过训练，将使用 Flat 索引"
                )
    
    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """归一化向量（用于余弦相似度）"""
        if self.metric_type == "cosine":
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1  # 避免除零
            return vectors / norms
        return vectors
    
    def insert(
        self,
        ids: List[str],
        vectors: List[np.ndarray],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> bool:
        """
        插入向量数据
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
            
            # 准备向量
            new_vectors = np.vstack([item[1] for item in new_items]).astype(np.float32)
            new_vectors = self._normalize_vectors(new_vectors)
            
            # 训练索引（如果需要）
            if hasattr(self._index, 'is_trained') and not self._index.is_trained:
                self._train_index(new_vectors)
            
            # 添加向量到索引
            self._index.add(new_vectors)
            
            # 更新元数据
            for id_, _, meta in new_items:
                self._ids.append(id_)
                self._metadatas.append(meta)
            
            self._id_to_idx = {id_: idx for idx, id_ in enumerate(self._ids)}
            
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
        
        注意：FAISS 不支持直接更新，需要重建索引
        """
        with self._lock:
            # 对于 FAISS，upsert 操作比较复杂
            # 简化处理：只插入新记录，忽略已存在的
            return self.insert(ids, vectors, metadatas, **kwargs)
    
    def delete(self, ids: List[str], **kwargs) -> bool:
        """
        删除指定ID的向量
        
        注意：FAISS 不支持直接删除，需要重建索引
        """
        with self._lock:
            # 找到要保留的索引
            ids_to_keep = set(self._ids) - set(ids)
            
            if len(ids_to_keep) == len(self._ids):
                logger.info("没有找到要删除的记录")
                return True
            
            # 重建索引
            keep_indices = [self._id_to_idx[id_] for id_ in ids_to_keep]
            
            # 获取保留的向量
            if keep_indices:
                # 从原索引中提取向量
                old_vectors = self._index.reconstruct_n(0, self._index.ntotal)
                keep_vectors = old_vectors[keep_indices]
                keep_metas = [self._metadatas[i] for i in keep_indices]
                keep_ids = [self._ids[i] for i in keep_indices]
            else:
                keep_vectors = np.array([]).reshape(0, self.embedding_dim).astype(np.float32)
                keep_metas = []
                keep_ids = []
            
            # 重建索引
            self._create_index()
            
            if len(keep_ids) > 0:
                if hasattr(self._index, 'is_trained') and not self._index.is_trained:
                    self._train_index(keep_vectors)
                self._index.add(keep_vectors)
            
            self._ids = keep_ids
            self._metadatas = keep_metas
            self._id_to_idx = {id_: idx for idx, id_ in enumerate(self._ids)}
            
            logger.info(f"删除了 {len(ids) - len(ids_to_keep)} 条记录")
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
            result = {}
            for id_ in ids:
                if id_ in self._id_to_idx:
                    idx = self._id_to_idx[id_]
                    vec = self._index.reconstruct(idx)
                    result[id_] = vec
            return result
    
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
        向量相似度搜索
        """
        with self._lock:
            if not self._ids:
                return []
            
            # 准备查询向量
            query = np.array(query_vector, dtype=np.float32).reshape(1, -1)
            query = self._normalize_vectors(query)
            
            # 搜索
            k = min(top_k * 2, len(self._ids))  # 搜索更多结果用于过滤
            distances, indices = self._index.search(query, k)
            
            # 构建结果
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0:  # FAISS 返回 -1 表示无效结果
                    continue
                
                id_ = self._ids[idx]
                meta = self._metadatas[idx]
                
                # 应用元数据过滤
                if filter_dict:
                    if not all(meta.get(k) == v for k, v in filter_dict.items()):
                        continue
                
                # 转换距离为相似度分数
                if self.metric_type == "cosine" or self.metric_type == "ip":
                    score = float(dist)  # 内积即为相似度
                else:
                    score = float(1.0 / (1.0 + dist))  # L2 转换为相似度
                
                results.append((id_, score, deepcopy(meta)))
                
                if len(results) >= top_k:
                    break
            
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
        with self._lock:
            if not self._ids:
                return [[] for _ in query_vectors]
            
            # 准备查询向量
            queries = np.vstack(query_vectors).astype(np.float32)
            queries = self._normalize_vectors(queries)
            
            # 批量搜索
            k = min(top_k * 2, len(self._ids))
            distances, indices = self._index.search(queries, k)
            
            # 构建结果
            all_results = []
            for i in range(len(query_vectors)):
                results = []
                for dist, idx in zip(distances[i], indices[i]):
                    if idx < 0:
                        continue
                    
                    id_ = self._ids[idx]
                    meta = self._metadatas[idx]
                    
                    if filter_dict:
                        if not all(meta.get(k) == v for k, v in filter_dict.items()):
                            continue
                    
                    if self.metric_type == "cosine" or self.metric_type == "ip":
                        score = float(dist)
                    else:
                        score = float(1.0 / (1.0 + dist))
                    
                    results.append((id_, score, deepcopy(meta)))
                    
                    if len(results) >= top_k:
                        break
                
                all_results.append(results)
            
            return all_results
    
    def get_all_ids(self, **kwargs) -> List[str]:
        """获取所有向量ID"""
        with self._lock:
            return self._ids.copy()
    
    def count(self, **kwargs) -> int:
        """获取向量总数"""
        with self._lock:
            return self._index.ntotal if self._index else 0
    
    def clear(self, **kwargs) -> bool:
        """清空所有数据"""
        with self._lock:
            self._create_index()
            
            # 删除文件
            for path in [self._index_path, self._meta_path]:
                if os.path.exists(path):
                    os.remove(path)
            
            return True
    
    def save(self, path: Optional[str] = None, **kwargs) -> bool:
        """保存索引到磁盘"""
        with self._lock:
            try:
                index_path = path or self._index_path
                meta_path = index_path.replace('.index', '_meta.json')
                
                # 如果使用 GPU，先转回 CPU
                index_to_save = self._index
                if self.use_gpu and hasattr(faiss, 'index_gpu_to_cpu'):
                    index_to_save = faiss.index_gpu_to_cpu(self._index)
                
                # 保存索引
                faiss.write_index(index_to_save, index_path)
                
                # 保存元数据
                meta_data = {
                    'ids': self._ids,
                    'metadatas': self._metadatas,
                    'embedding_dim': self.embedding_dim,
                    'index_type': self.index_type,
                    'metric_type': self.metric_type
                }
                with open(meta_path, 'w', encoding='utf-8') as f:
                    json.dump(meta_data, f, ensure_ascii=False, indent=2)
                
                logger.info(f"保存了 {len(self._ids)} 条记录到 {index_path}")
                return True
                
            except Exception as e:
                logger.error(f"保存索引失败: {e}")
                return False
    
    def load(self, path: Optional[str] = None, **kwargs) -> bool:
        """从磁盘加载索引"""
        with self._lock:
            try:
                index_path = path or self._index_path
                meta_path = index_path.replace('.index', '_meta.json')
                
                if not os.path.exists(index_path) or not os.path.exists(meta_path):
                    logger.warning(f"索引文件不存在: {index_path}")
                    return False
                
                # 加载索引
                self._index = faiss.read_index(index_path)
                
                # 如果使用 GPU，移至 GPU
                if self.use_gpu and hasattr(faiss, 'StandardGpuResources'):
                    try:
                        res = faiss.StandardGpuResources()
                        self._index = faiss.index_cpu_to_gpu(res, 0, self._index)
                    except Exception as e:
                        logger.warning(f"无法使用 GPU: {e}")
                
                # 加载元数据
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta_data = json.load(f)
                
                self._ids = meta_data['ids']
                self._metadatas = meta_data['metadatas']
                self._id_to_idx = {id_: idx for idx, id_ in enumerate(self._ids)}
                
                logger.info(f"从 {index_path} 加载了 {len(self._ids)} 条记录")
                return True
                
            except Exception as e:
                logger.error(f"加载索引失败: {e}")
                return False
