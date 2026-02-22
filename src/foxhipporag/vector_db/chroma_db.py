"""
Chroma 向量数据库后端

基于 Chroma (ChromaDB) 的嵌入式向量存储实现。
Chroma 是一个轻量级、开源的向量数据库，适合本地开发和中小规模应用。
"""

import logging
import threading
from typing import List, Dict, Any, Optional, Tuple
from copy import deepcopy

import numpy as np

from .base import BaseVectorDB

logger = logging.getLogger(__name__)

# Chroma 可选导入
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logger.warning("Chroma 支持未安装。请运行: pip install chromadb>=0.4.0")


class ChromaVectorDB(BaseVectorDB):
    """
    基于 Chroma 的向量存储
    
    特性：
    - 轻量级嵌入式数据库
    - 支持本地持久化
    - 支持元数据过滤
    - 简单易用，无需额外服务
    """
    
    def __init__(
        self,
        embedding_dim: int,
        namespace: str = "default",
        db_path: str = "./chroma_db",
        collection_name: Optional[str] = None,
        distance_metric: str = "cosine",
        persistent: bool = True,
        **kwargs
    ):
        """
        初始化 Chroma 向量存储
        
        Args:
            embedding_dim: 嵌入向量维度
            namespace: 命名空间，用于数据隔离
            db_path: 数据库存储路径
            collection_name: 集合名称，默认使用 namespace
            distance_metric: 距离度量类型 ('cosine', 'l2', 'ip')
            persistent: 是否持久化存储
            **kwargs: 额外参数
        """
        if not CHROMA_AVAILABLE:
            raise ImportError(
                "Chroma 支持未安装。请运行: pip install chromadb>=0.4.0"
            )
        
        super().__init__(embedding_dim, namespace, **kwargs)
        
        self.db_path = db_path
        self.collection_name = collection_name or f"foxhipporag_{namespace}"
        self.distance_metric = distance_metric
        self.persistent = persistent
        
        # 线程安全锁
        self._lock = threading.RLock()
        
        # Chroma 客户端和集合
        self._client: Optional[chromadb.Client] = None
        self._collection: Optional[chromadb.Collection] = None
        
        # 初始化客户端和集合
        self._init_client()
        
        self._initialized = True
    
    def _init_client(self):
        """初始化 Chroma 客户端和集合"""
        try:
            if self.persistent:
                # 持久化模式
                self._client = chromadb.PersistentClient(
                    path=self.db_path,
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
            else:
                # 内存模式
                self._client = chromadb.Client(
                    Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
            
            # 获取或创建集合
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": self.distance_metric,
                    "embedding_dim": self.embedding_dim
                }
            )
            
            logger.info(
                f"初始化 Chroma 集合: {self.collection_name}, "
                f"现有记录数: {self._collection.count()}"
            )
            
        except Exception as e:
            logger.error(f"初始化 Chroma 失败: {e}")
            raise
    
    def insert(
        self,
        ids: List[str],
        vectors: List[np.ndarray],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> bool:
        """
        插入向量数据
        
        注意：Chroma 不支持检查重复ID，需要先查询
        """
        with self._lock:
            try:
                # 检查已存在的ID
                existing = self._collection.get(ids=ids)
                existing_ids = set(existing["ids"]) if existing["ids"] else set()
                
                # 过滤新记录
                new_items = [
                    (id_, vec, meta if metadatas else {})
                    for id_, vec, meta in zip(
                        ids, 
                        vectors, 
                        metadatas or [{}] * len(ids)
                    )
                    if id_ not in existing_ids
                ]
                
                if not new_items:
                    logger.info("所有记录已存在，跳过插入")
                    return True
                
                # 准备数据
                insert_ids = [item[0] for item in new_items]
                insert_vectors = [item[1].tolist() for item in new_items]
                insert_metadatas = [item[2] for item in new_items]
                
                # 插入数据
                self._collection.add(
                    ids=insert_ids,
                    embeddings=insert_vectors,
                    metadatas=insert_metadatas
                )
                
                logger.info(f"插入了 {len(new_items)} 条新记录")
                return True
                
            except Exception as e:
                logger.error(f"插入数据失败: {e}")
                return False
    
    def upsert(
        self,
        ids: List[str],
        vectors: List[np.ndarray],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> bool:
        """
        插入或更新向量数据
        
        Chroma 支持 upsert 操作（从 0.4.0 版本开始）
        """
        with self._lock:
            try:
                # 准备数据
                upsert_ids = list(ids)
                upsert_vectors = [v.tolist() for v in vectors]
                upsert_metadatas = list(metadatas) if metadatas else [{}] * len(ids)
                
                # 使用 upsert
                self._collection.upsert(
                    ids=upsert_ids,
                    embeddings=upsert_vectors,
                    metadatas=upsert_metadatas
                )
                
                logger.info(f"更新/插入了 {len(ids)} 条记录")
                return True
                
            except Exception as e:
                logger.error(f"更新数据失败: {e}")
                return False
    
    def delete(self, ids: List[str], **kwargs) -> bool:
        """
        删除指定ID的向量
        """
        with self._lock:
            try:
                self._collection.delete(ids=ids)
                logger.info(f"删除了 {len(ids)} 条记录")
                return True
                
            except Exception as e:
                logger.error(f"删除数据失败: {e}")
                return False
    
    def get_vectors(
        self,
        ids: List[str],
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        根据ID获取向量
        """
        with self._lock:
            try:
                if not ids:
                    return {}
                
                results = self._collection.get(
                    ids=ids,
                    include=["embeddings"]
                )
                
                return {
                    id_: np.array(emb, dtype=np.float32)
                    for id_, emb in zip(results["ids"], results["embeddings"] or [])
                }
                
            except Exception as e:
                logger.error(f"获取向量失败: {e}")
                return {}
    
    def get_metadatas(
        self,
        ids: List[str],
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """
        根据ID获取元数据
        """
        with self._lock:
            try:
                if not ids:
                    return {}
                
                results = self._collection.get(
                    ids=ids,
                    include=["metadatas"]
                )
                
                return {
                    id_: deepcopy(meta)
                    for id_, meta in zip(results["ids"], results["metadatas"] or [])
                }
                
            except Exception as e:
                logger.error(f"获取元数据失败: {e}")
                return {}
    
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
            try:
                # 构建过滤条件
                where_filter = None
                if filter_dict:
                    where_filter = {"$and": [
                        {k: v} for k, v in filter_dict.items()
                    ]} if len(filter_dict) > 1 else filter_dict
                
                # 执行搜索
                results = self._collection.query(
                    query_embeddings=[query_vector.tolist()],
                    n_results=top_k,
                    where=where_filter,
                    include=["distances", "metadatas"]
                )
                
                # 构建结果
                output = []
                if results["ids"] and results["ids"][0]:
                    for i, id_ in enumerate(results["ids"][0]):
                        distance = results["distances"][0][i] if results["distances"] else 0
                        meta = results["metadatas"][0][i] if results["metadatas"] else {}
                        
                        # 转换距离为相似度分数
                        if self.distance_metric == "cosine":
                            # Chroma 返回的 cosine 距离是 1 - similarity
                            score = 1.0 - distance
                        elif self.distance_metric == "l2":
                            score = 1.0 / (1.0 + distance)
                        else:  # ip (inner product)
                            score = distance
                        
                        output.append((id_, float(score), deepcopy(meta)))
                
                return output
                
            except Exception as e:
                logger.error(f"搜索失败: {e}")
                return []
    
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
            try:
                # 构建过滤条件
                where_filter = None
                if filter_dict:
                    where_filter = {"$and": [
                        {k: v} for k, v in filter_dict.items()
                    ]} if len(filter_dict) > 1 else filter_dict
                
                # 执行批量搜索
                results = self._collection.query(
                    query_embeddings=[v.tolist() for v in query_vectors],
                    n_results=top_k,
                    where=where_filter,
                    include=["distances", "metadatas"]
                )
                
                # 构建结果
                all_results = []
                if results["ids"]:
                    for i, ids in enumerate(results["ids"]):
                        query_results = []
                        for j, id_ in enumerate(ids):
                            distance = results["distances"][i][j] if results["distances"] else 0
                            meta = results["metadatas"][i][j] if results["metadatas"] else {}
                            
                            if self.distance_metric == "cosine":
                                score = 1.0 - distance
                            elif self.distance_metric == "l2":
                                score = 1.0 / (1.0 + distance)
                            else:
                                score = distance
                            
                            query_results.append((id_, float(score), deepcopy(meta)))
                        
                        all_results.append(query_results)
                else:
                    all_results = [[] for _ in query_vectors]
                
                return all_results
                
            except Exception as e:
                logger.error(f"批量搜索失败: {e}")
                return [[] for _ in query_vectors]
    
    def get_all_ids(self, **kwargs) -> List[str]:
        """获取所有向量ID"""
        with self._lock:
            try:
                # Chroma 需要使用 get 方法获取所有数据
                results = self._collection.get(include=[])
                return results["ids"] or []
                
            except Exception as e:
                logger.error(f"获取所有ID失败: {e}")
                return []
    
    def count(self, **kwargs) -> int:
        """获取向量总数"""
        with self._lock:
            try:
                return self._collection.count()
            except Exception as e:
                logger.error(f"获取计数失败: {e}")
                return 0
    
    def clear(self, **kwargs) -> bool:
        """清空所有数据"""
        with self._lock:
            try:
                # 删除集合并重新创建
                self._client.delete_collection(self.collection_name)
                self._collection = self._client.create_collection(
                    name=self.collection_name,
                    metadata={
                        "hnsw:space": self.distance_metric,
                        "embedding_dim": self.embedding_dim
                    }
                )
                
                logger.info(f"清空了集合: {self.collection_name}")
                return True
                
            except Exception as e:
                logger.error(f"清空数据失败: {e}")
                return False
    
    def save(self, path: Optional[str] = None, **kwargs) -> bool:
        """
        保存数据（Chroma 自动持久化）
        
        对于持久化模式的 Chroma，数据自动保存
        """
        if self.persistent:
            logger.info("Chroma 已自动持久化数据")
            return True
        else:
            logger.warning("内存模式下数据无法持久化")
            return False
    
    def load(self, path: Optional[str] = None, **kwargs) -> bool:
        """
        加载数据（Chroma 自动加载）
        
        对于持久化模式的 Chroma，数据自动加载
        """
        if self.persistent:
            logger.info("Chroma 已自动加载数据")
            return True
        else:
            logger.warning("内存模式下无法加载数据")
            return False
