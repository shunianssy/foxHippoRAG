"""
Milvus 向量数据库后端

基于 Milvus 的分布式向量存储实现。
Milvus 是一个开源的向量数据库，支持大规模向量检索。
"""

import logging
import threading
import time
from typing import List, Dict, Any, Optional, Tuple
from copy import deepcopy

import numpy as np

from .base import BaseVectorDB

logger = logging.getLogger(__name__)

# Milvus 可选导入
try:
    from pymilvus import (
        connections,
        Collection,
        CollectionSchema,
        FieldSchema,
        DataType,
        utility,
        MilvusException
    )
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    logger.warning("Milvus 支持未安装。请运行: pip install pymilvus>=2.3.0")


class MilvusVectorDB(BaseVectorDB):
    """
    基于 Milvus 的向量存储
    
    特性：
    - 分布式架构，支持大规模向量
    - 支持多种索引类型
    - 支持元数据过滤
    - 高可用和可扩展
    """
    
    def __init__(
        self,
        embedding_dim: int,
        namespace: str = "default",
        host: str = "localhost",
        port: int = 19530,
        user: str = "",
        password: str = "",
        db_name: str = "default",
        collection_name: Optional[str] = None,
        index_type: str = "IVF_FLAT",
        metric_type: str = "COSINE",
        nlist: int = 1024,
        nprobe: int = 16,
        consistency_level: str = "Bounded",
        **kwargs
    ):
        """
        初始化 Milvus 向量存储
        
        Args:
            embedding_dim: 嵌入向量维度
            namespace: 命名空间，用于数据隔离
            host: Milvus 服务器地址
            port: Milvus 服务器端口
            user: 用户名
            password: 密码
            db_name: 数据库名称
            collection_name: 集合名称，默认使用 namespace
            index_type: 索引类型 ('FLAT', 'IVF_FLAT', 'IVF_PQ', 'HNSW', 'ANNOY')
            metric_type: 距离度量类型 ('L2', 'IP', 'COSINE')
            nlist: IVF 索引的聚类中心数量
            nprobe: 搜索时探测的聚类数量
            consistency_level: 一致性级别 ('Strong', 'Bounded', 'Session', 'Eventually')
            **kwargs: 额外参数
        """
        if not MILVUS_AVAILABLE:
            raise ImportError(
                "Milvus 支持未安装。请运行: pip install pymilvus>=2.3.0"
            )
        
        super().__init__(embedding_dim, namespace, **kwargs)
        
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.db_name = db_name
        self.collection_name = collection_name or f"foxhipporag_{namespace}"
        self.index_type = index_type
        self.metric_type = metric_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.consistency_level = consistency_level
        
        # 线程安全锁
        self._lock = threading.RLock()
        
        # 集合对象
        self._collection: Optional[Collection] = None
        self._connected = False
        
        # 连接并初始化
        self._connect()
        self._init_collection()
        
        self._initialized = True
    
    def _connect(self):
        """连接到 Milvus 服务器"""
        try:
            # 检查是否已连接
            alias = f"alias_{self.namespace}_{id(self)}"
            
            try:
                connections.connect(
                    alias=alias,
                    host=self.host,
                    port=self.port,
                    user=self.user,
                    password=self.password,
                    db_name=self.db_name
                )
                self._alias = alias
                self._connected = True
                logger.info(f"成功连接到 Milvus: {self.host}:{self.port}")
            except MilvusException as e:
                if "already connected" in str(e).lower():
                    self._connected = True
                    logger.info("已连接到 Milvus")
                else:
                    raise
                    
        except Exception as e:
            logger.error(f"连接 Milvus 失败: {e}")
            raise
    
    def _init_collection(self):
        """初始化集合"""
        with self._lock:
            try:
                # 检查集合是否存在
                if utility.has_collection(self.collection_name):
                    self._collection = Collection(self.collection_name)
                    self._collection.load()
                    logger.info(f"加载现有集合: {self.collection_name}")
                else:
                    # 创建新集合
                    self._create_collection()
                    
            except Exception as e:
                logger.error(f"初始化集合失败: {e}")
                raise
    
    def _create_collection(self):
        """创建新集合"""
        # 定义字段
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=256, is_primary=True, auto_id=False),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
        ]
        
        # 创建 schema
        schema = CollectionSchema(
            fields=fields,
            description=f"foxHippoRAG vector collection for {self.namespace}"
        )
        
        # 创建集合
        self._collection = Collection(
            name=self.collection_name,
            schema=schema,
            consistency_level=self.consistency_level
        )
        
        # 创建索引
        index_params = {
            "metric_type": self.metric_type,
            "index_type": self.index_type,
            "params": {"nlist": self.nlist}
        }
        self._collection.create_index(field_name="vector", index_params=index_params)
        
        # 加载集合
        self._collection.load()
        
        logger.info(f"创建新集合: {self.collection_name}")
    
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
            try:
                # 检查已存在的ID
                existing_ids = set()
                if self._collection.num_entities > 0:
                    expr = f"id in {ids}"
                    try:
                        results = self._collection.query(expr=expr, output_fields=["id"])
                        existing_ids = {r["id"] for r in results}
                    except Exception:
                        pass
                
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
                insert_contents = [item[2].get("content", "") for item in new_items]
                
                # 插入数据
                self._collection.insert([insert_ids, insert_vectors, insert_contents])
                self._collection.flush()
                
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
        
        注意：Milvus 2.3+ 支持 upsert 操作
        """
        with self._lock:
            try:
                # 准备数据
                upsert_ids = list(ids)
                upsert_vectors = [v.tolist() for v in vectors]
                upsert_contents = [
                    (meta if metadatas else {}).get("content", "") 
                    for meta in (metadatas or [{}] * len(ids))
                ]
                
                # 使用 upsert
                self._collection.upsert([upsert_ids, upsert_vectors, upsert_contents])
                self._collection.flush()
                
                logger.info(f"更新/插入了 {len(ids)} 条记录")
                return True
                
            except Exception as e:
                logger.error(f"更新数据失败: {e}")
                # 如果 upsert 不支持，尝试先删除再插入
                try:
                    self.delete(ids)
                    return self.insert(ids, vectors, metadatas, **kwargs)
                except Exception as e2:
                    logger.error(f"备用插入也失败: {e2}")
                    return False
    
    def delete(self, ids: List[str], **kwargs) -> bool:
        """
        删除指定ID的向量
        """
        with self._lock:
            try:
                expr = f"id in {ids}"
                self._collection.delete(expr)
                self._collection.flush()
                
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
                
                expr = f"id in {ids}"
                results = self._collection.query(
                    expr=expr,
                    output_fields=["id", "vector"]
                )
                
                return {
                    r["id"]: np.array(r["vector"], dtype=np.float32)
                    for r in results
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
                
                expr = f"id in {ids}"
                results = self._collection.query(
                    expr=expr,
                    output_fields=["id", "content"]
                )
                
                return {
                    r["id"]: {"content": r.get("content", "")}
                    for r in results
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
                # 准备搜索参数
                search_params = {
                    "metric_type": self.metric_type,
                    "params": {"nprobe": self.nprobe}
                }
                
                # 准备查询向量
                query = [query_vector.tolist()]
                
                # 构建过滤表达式
                expr = None
                if filter_dict:
                    conditions = [f'{k} == "{v}"' for k, v in filter_dict.items()]
                    expr = " and ".join(conditions)
                
                # 执行搜索
                results = self._collection.search(
                    data=query,
                    anns_field="vector",
                    param=search_params,
                    limit=top_k,
                    expr=expr,
                    output_fields=["id", "content"]
                )
                
                # 构建结果
                output = []
                for hits in results:
                    for hit in hits:
                        output.append((
                            hit.id,
                            hit.score,
                            {"content": hit.entity.get("content", "")}
                        ))
                
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
                # 准备搜索参数
                search_params = {
                    "metric_type": self.metric_type,
                    "params": {"nprobe": self.nprobe}
                }
                
                # 准备查询向量
                queries = [v.tolist() for v in query_vectors]
                
                # 构建过滤表达式
                expr = None
                if filter_dict:
                    conditions = [f'{k} == "{v}"' for k, v in filter_dict.items()]
                    expr = " and ".join(conditions)
                
                # 执行搜索
                results = self._collection.search(
                    data=queries,
                    anns_field="vector",
                    param=search_params,
                    limit=top_k,
                    expr=expr,
                    output_fields=["id", "content"]
                )
                
                # 构建结果
                all_results = []
                for hits in results:
                    query_results = []
                    for hit in hits:
                        query_results.append((
                            hit.id,
                            hit.score,
                            {"content": hit.entity.get("content", "")}
                        ))
                    all_results.append(query_results)
                
                return all_results
                
            except Exception as e:
                logger.error(f"批量搜索失败: {e}")
                return [[] for _ in query_vectors]
    
    def get_all_ids(self, **kwargs) -> List[str]:
        """获取所有向量ID"""
        with self._lock:
            try:
                # 使用迭代查询获取所有ID
                all_ids = []
                batch_size = 1000
                
                results = self._collection.query(
                    expr="",
                    output_fields=["id"],
                    limit=batch_size
                )
                
                all_ids = [r["id"] for r in results]
                
                return all_ids
                
            except Exception as e:
                logger.error(f"获取所有ID失败: {e}")
                return []
    
    def count(self, **kwargs) -> int:
        """获取向量总数"""
        with self._lock:
            try:
                self._collection.flush()
                return self._collection.num_entities
            except Exception as e:
                logger.error(f"获取计数失败: {e}")
                return 0
    
    def clear(self, **kwargs) -> bool:
        """清空所有数据"""
        with self._lock:
            try:
                # 删除集合并重新创建
                utility.drop_collection(self.collection_name)
                self._create_collection()
                
                logger.info(f"清空了集合: {self.collection_name}")
                return True
                
            except Exception as e:
                logger.error(f"清空数据失败: {e}")
                return False
    
    def save(self, path: Optional[str] = None, **kwargs) -> bool:
        """
        保存数据（Milvus 自动持久化）
        
        对于 Milvus，数据自动持久化到存储，此方法主要用于触发 flush
        """
        with self._lock:
            try:
                self._collection.flush()
                logger.info("数据已刷新到 Milvus")
                return True
            except Exception as e:
                logger.error(f"刷新数据失败: {e}")
                return False
    
    def load(self, path: Optional[str] = None, **kwargs) -> bool:
        """
        加载数据（Milvus 自动加载）
        
        对于 Milvus，数据自动从存储加载，此方法主要用于确保集合已加载
        """
        with self._lock:
            try:
                self._collection.load()
                logger.info("集合已加载到内存")
                return True
            except Exception as e:
                logger.error(f"加载数据失败: {e}")
                return False
    
    def __del__(self):
        """析构函数，断开连接"""
        try:
            if hasattr(self, '_alias') and self._connected:
                connections.disconnect(self._alias)
        except Exception:
            pass
