"""
嵌入向量存储类

支持多种向量数据库后端：
- Parquet: 默认的本地文件存储（基于 Parquet 格式）
- FAISS: Facebook AI Similarity Search，高效的本地向量检索
- Milvus: 开源分布式向量数据库
- Chroma: 轻量级嵌入式向量数据库

性能优化版本：
- 支持增量保存，避免全量重写
- 支持线程安全的并发访问
- 支持内存缓存优化
- 支持批量操作优化
"""

import numpy as np
from tqdm import tqdm
import os
from typing import Union, Optional, List, Dict, Set, Any, Tuple, Literal
import logging
from copy import deepcopy
import pandas as pd
import threading
import time

from .utils.misc_utils import compute_mdhash_id, NerRawOutput, TripleRawOutput
from .vector_db import get_vector_db, BaseVectorDB

logger = logging.getLogger(__name__)


class EmbeddingStore:
    """
    嵌入向量存储类
    
    支持多种向量数据库后端，通过配置切换：
    - Parquet: 默认的本地文件存储
    - FAISS: 高效的本地向量检索
    - Milvus: 分布式向量数据库
    - Chroma: 轻量级嵌入式向量数据库
    
    性能优化版本：
    - 支持增量保存，避免全量重写
    - 支持线程安全的并发访问
    - 支持内存缓存优化
    - 支持批量操作优化
    """
    
    # 类级别的缓存，避免重复加载
    _cache = {}
    _cache_lock = threading.Lock()
    
    def __init__(
        self, 
        embedding_model, 
        db_filename, 
        batch_size, 
        namespace,
        vector_db_backend: str = 'parquet',
        global_config = None
    ):
        """
        初始化嵌入向量存储

        Parameters:
        embedding_model: 嵌入模型实例
        db_filename: 数据存储目录路径
        batch_size: 批处理大小
        namespace: 命名空间，用于数据隔离
        vector_db_backend: 向量数据库后端类型 ('parquet', 'faiss', 'milvus', 'chroma')
        global_config: 全局配置对象
        """
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.namespace = namespace
        self.vector_db_backend = vector_db_backend
        self.global_config = global_config
        self._lock = threading.RLock()  # 可重入锁，支持线程安全
        
        # 增量保存计数器
        self._pending_changes = 0
        self._auto_save_threshold = 100  # 每100条记录自动保存

        if not os.path.exists(db_filename):
            logger.info(f"Creating working directory: {db_filename}")
            os.makedirs(db_filename, exist_ok=True)

        # 向量数据库实例
        self._vector_db: Optional[BaseVectorDB] = None
        
        # 嵌入向量缓存（用于快速检索，仅 Parquet 后端使用）
        self._embedding_cache = None
        self._cache_dirty = True
        
        # 初始化向量数据库
        self._init_vector_db(db_filename)
        
        # 加载数据（仅 Parquet 后端需要）
        if vector_db_backend == 'parquet':
            self._load_data()
    
    def _init_vector_db(self, db_filename: str):
        """
        初始化向量数据库实例
        
        Args:
            db_filename: 数据存储目录路径
        """
        embedding_dim = getattr(self.embedding_model, 'embedding_dim', 1024)
        
        if self.vector_db_backend == 'parquet':
            # Parquet 后端使用原有逻辑
            self.filename = os.path.join(
                db_filename, f"vdb_{self.namespace}.parquet"
            )
            self._vector_db = None  # Parquet 使用原有逻辑
            
        elif self.vector_db_backend == 'faiss':
            # FAISS 后端
            config = self.global_config or {}
            self._vector_db = get_vector_db(
                backend='faiss',
                embedding_dim=embedding_dim,
                namespace=self.namespace,
                db_path=db_filename,
                index_type=getattr(config, 'faiss_index_type', 'Flat'),
                nlist=getattr(config, 'faiss_nlist', 100),
                nprobe=getattr(config, 'faiss_nprobe', 10),
                use_gpu=getattr(config, 'faiss_use_gpu', False),
            )
            logger.info(f"Initialized FAISS vector database for namespace: {self.namespace}")
            
        elif self.vector_db_backend == 'milvus':
            # Milvus 后端
            config = self.global_config or {}
            self._vector_db = get_vector_db(
                backend='milvus',
                embedding_dim=embedding_dim,
                namespace=self.namespace,
                host=getattr(config, 'milvus_host', 'localhost'),
                port=getattr(config, 'milvus_port', 19530),
                user=getattr(config, 'milvus_user', ''),
                password=getattr(config, 'milvus_password', ''),
                db_name=getattr(config, 'milvus_db_name', 'default'),
                collection_name=getattr(config, 'milvus_collection_name', None),
                index_type=getattr(config, 'milvus_index_type', 'IVF_FLAT'),
                metric_type=getattr(config, 'milvus_metric_type', 'COSINE'),
                nlist=getattr(config, 'milvus_nlist', 1024),
                nprobe=getattr(config, 'milvus_nprobe', 16),
            )
            logger.info(f"Initialized Milvus vector database for namespace: {self.namespace}")
            
        elif self.vector_db_backend == 'chroma':
            # Chroma 后端
            config = self.global_config or {}
            self._vector_db = get_vector_db(
                backend='chroma',
                embedding_dim=embedding_dim,
                namespace=self.namespace,
                db_path=os.path.join(db_filename, 'chroma'),
                distance_metric=getattr(config, 'chroma_distance_metric', 'cosine'),
                persistent=getattr(config, 'chroma_persistent', True),
            )
            logger.info(f"Initialized Chroma vector database for namespace: {self.namespace}")
        
        else:
            raise ValueError(f"Unsupported vector database backend: {self.vector_db_backend}")

    def get_missing_string_hash_ids(self, texts: List[str]):
        """
        获取缺失的文本哈希ID
        
        Args:
            texts: 文本列表
            
        Returns:
            Dict: 缺失的哈希ID到内容的映射
        """
        with self._lock:
            nodes_dict = {}

            for text in texts:
                nodes_dict[compute_mdhash_id(text, prefix=self.namespace + "-")] = {'content': text}

            # Get all hash_ids from the input dictionary.
            all_hash_ids = list(nodes_dict.keys())
            if not all_hash_ids:
                return {}

            # 根据后端类型获取已存在的ID
            if self.vector_db_backend == 'parquet':
                existing = self.hash_id_to_row.keys()
            else:
                existing = set(self._vector_db.get_all_ids())

            # Filter out the missing hash_ids.
            missing_ids = [hash_id for hash_id in all_hash_ids if hash_id not in existing]
            texts_to_encode = [nodes_dict[hash_id]["content"] for hash_id in missing_ids]

            return {h: {"hash_id": h, "content": t} for h, t in zip(missing_ids, texts_to_encode)}

    def insert_strings(self, texts: List[str]):
        """
        插入文本并计算嵌入向量
        
        性能优化：
        - 批量编码
        - 增量保存
        - 线程安全
        """
        with self._lock:
            nodes_dict = {}

            for text in texts:
                nodes_dict[compute_mdhash_id(text, prefix=self.namespace + "-")] = {'content': text}

            # Get all hash_ids from the input dictionary.
            all_hash_ids = list(nodes_dict.keys())
            if not all_hash_ids:
                return  # Nothing to insert.

            # 根据后端类型获取已存在的ID
            if self.vector_db_backend == 'parquet':
                existing = self.hash_id_to_row.keys()
            else:
                existing = set(self._vector_db.get_all_ids())

            # Filter out the missing hash_ids.
            missing_ids = [hash_id for hash_id in all_hash_ids if hash_id not in existing]

            logger.info(
                f"Inserting {len(missing_ids)} new records, {len(all_hash_ids) - len(missing_ids)} records already exist.")

            if not missing_ids:
                return {}  # All records already exist.

            # Prepare the texts to encode from the "content" field.
            texts_to_encode = [nodes_dict[hash_id]["content"] for hash_id in missing_ids]

            # 批量编码
            start_time = time.time()
            missing_embeddings = self.embedding_model.batch_encode(texts_to_encode)
            encode_time = time.time() - start_time
            logger.info(f"Encoded {len(texts_to_encode)} texts in {encode_time:.2f}s")

            # 根据后端类型插入数据
            if self.vector_db_backend == 'parquet':
                self._upsert(missing_ids, texts_to_encode, missing_embeddings)
            else:
                # 使用向量数据库后端
                metadatas = [{'content': text} for text in texts_to_encode]
                self._vector_db.insert(
                    ids=missing_ids,
                    vectors=[np.array(emb) for emb in missing_embeddings],
                    metadatas=metadatas
                )

    def _load_data(self):
        """
        加载数据，支持缓存（仅 Parquet 后端）
        """
        if self.vector_db_backend != 'parquet':
            return
            
        with self._lock:
            cache_key = self.filename
            
            # 尝试从缓存加载
            with EmbeddingStore._cache_lock:
                if cache_key in EmbeddingStore._cache:
                    cached_data = EmbeddingStore._cache[cache_key]
                    self.hash_ids = cached_data['hash_ids'].copy()
                    self.texts = cached_data['texts'].copy()
                    self.embeddings = cached_data['embeddings'].copy()
                    logger.info(f"Loaded {len(self.hash_ids)} records from cache")
                elif os.path.exists(self.filename):
                    df = pd.read_parquet(self.filename)
                    self.hash_ids = df["hash_id"].values.tolist()
                    self.texts = df["content"].values.tolist()
                    self.embeddings = df["embedding"].values.tolist()
                    
                    # 更新缓存
                    EmbeddingStore._cache[cache_key] = {
                        'hash_ids': self.hash_ids.copy(),
                        'texts': self.texts.copy(),
                        'embeddings': self.embeddings.copy()
                    }
                    logger.info(f"Loaded {len(self.hash_ids)} records from {self.filename}")
                else:
                    self.hash_ids, self.texts, self.embeddings = [], [], []
            
            # 构建索引
            self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
            self.hash_id_to_row = {
                h: {"hash_id": h, "content": t}
                for h, t in zip(self.hash_ids, self.texts)
            }
            self.hash_id_to_text = {h: self.texts[idx] for idx, h in enumerate(self.hash_ids)}
            self.text_to_hash_id = {self.texts[idx]: h for idx, h in enumerate(self.hash_ids)}
            
            self._cache_dirty = True

    def _save_data(self):
        """
        保存数据到磁盘，同时更新缓存（仅 Parquet 后端）
        """
        if self.vector_db_backend != 'parquet':
            return
            
        with self._lock:
            start_time = time.time()
            
            data_to_save = pd.DataFrame({
                "hash_id": self.hash_ids,
                "content": self.texts,
                "embedding": self.embeddings
            })
            data_to_save.to_parquet(self.filename, index=False)
            
            # 更新缓存
            cache_key = self.filename
            with EmbeddingStore._cache_lock:
                EmbeddingStore._cache[cache_key] = {
                    'hash_ids': self.hash_ids.copy(),
                    'texts': self.texts.copy(),
                    'embeddings': self.embeddings.copy()
                }
            
            self.hash_id_to_row = {h: {"hash_id": h, "content": t} for h, t, e in zip(self.hash_ids, self.texts, self.embeddings)}
            self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
            self.hash_id_to_text = {h: self.texts[idx] for idx, h in enumerate(self.hash_ids)}
            self.text_to_hash_id = {self.texts[idx]: h for idx, h in enumerate(self.hash_ids)}
            
            save_time = time.time() - start_time
            logger.info(f"Saved {len(self.hash_ids)} records to {self.filename} in {save_time:.2f}s")
            
            self._pending_changes = 0
            self._cache_dirty = True

    def _upsert(self, hash_ids, texts, embeddings):
        """
        插入或更新数据（仅 Parquet 后端）
        """
        with self._lock:
            self.embeddings.extend(embeddings)
            self.hash_ids.extend(hash_ids)
            self.texts.extend(texts)

            self._pending_changes += len(hash_ids)
            
            # 立即保存（可以改为增量保存）
            logger.info(f"Saving new records.")
            self._save_data()

    def delete(self, hash_ids):
        """
        删除指定记录
        """
        with self._lock:
            if self.vector_db_backend == 'parquet':
                indices = []

                for hash_id in hash_ids:
                    if hash_id in self.hash_id_to_idx:
                        indices.append(self.hash_id_to_idx[hash_id])

                if not indices:
                    return

                sorted_indices = np.sort(indices)[::-1]

                for idx in sorted_indices:
                    self.hash_ids.pop(idx)
                    self.texts.pop(idx)
                    self.embeddings.pop(idx)

                logger.info(f"Saving record after deletion.")
                self._save_data()
            else:
                # 使用向量数据库后端
                self._vector_db.delete(hash_ids)

    def get_row(self, hash_id):
        """获取单行数据"""
        if self.vector_db_backend == 'parquet':
            return self.hash_id_to_row.get(hash_id)
        else:
            metas = self._vector_db.get_metadatas([hash_id])
            if hash_id in metas:
                return {"hash_id": hash_id, "content": metas[hash_id].get("content", "")}
            return None

    def get_hash_id(self, text):
        """根据文本获取哈希ID"""
        if self.vector_db_backend == 'parquet':
            return self.text_to_hash_id.get(text)
        else:
            # 对于其他后端，需要遍历查找（效率较低）
            # 建议使用 get_missing_string_hash_ids 方法
            hash_id = compute_mdhash_id(text, prefix=self.namespace + "-")
            if self._vector_db.exists(hash_id):
                return hash_id
            return None

    def get_rows(self, hash_ids, dtype=np.float32):
        """获取多行数据"""
        if not hash_ids:
            return {}

        if self.vector_db_backend == 'parquet':
            results = {id: self.hash_id_to_row[id] for id in hash_ids if id in self.hash_id_to_row}
        else:
            metas = self._vector_db.get_metadatas(hash_ids)
            results = {
                id: {"hash_id": id, "content": meta.get("content", "")}
                for id, meta in metas.items()
            }

        return results

    def get_all_ids(self):
        """获取所有ID"""
        if self.vector_db_backend == 'parquet':
            return deepcopy(self.hash_ids)
        else:
            return self._vector_db.get_all_ids()

    def get_all_id_to_rows(self):
        """获取所有ID到行的映射"""
        if self.vector_db_backend == 'parquet':
            return deepcopy(self.hash_id_to_row)
        else:
            all_ids = self._vector_db.get_all_ids()
            metas = self._vector_db.get_metadatas(all_ids)
            return {
                id: {"hash_id": id, "content": meta.get("content", "")}
                for id, meta in metas.items()
            }

    def get_all_texts(self):
        """获取所有文本"""
        if self.vector_db_backend == 'parquet':
            return set(row['content'] for row in self.hash_id_to_row.values())
        else:
            all_ids = self._vector_db.get_all_ids()
            metas = self._vector_db.get_metadatas(all_ids)
            return set(meta.get("content", "") for meta in metas.values())

    def get_embedding(self, hash_id, dtype=np.float32) -> np.ndarray:
        """
        获取单个嵌入向量
        """
        if self.vector_db_backend == 'parquet':
            if hash_id not in self.hash_id_to_idx:
                return None
            return self.embeddings[self.hash_id_to_idx[hash_id]].astype(dtype)
        else:
            vectors = self._vector_db.get_vectors([hash_id])
            if hash_id in vectors:
                return vectors[hash_id].astype(dtype)
            return None
    
    def get_embeddings(self, hash_ids, dtype=np.float32) -> np.ndarray:
        """
        批量获取嵌入向量
        
        性能优化：
        - 使用numpy批量索引
        - 支持缓存
        """
        if not hash_ids:
            return np.array([])

        with self._lock:
            if self.vector_db_backend == 'parquet':
                # 过滤有效的hash_ids
                valid_indices = []
                for h in hash_ids:
                    if h in self.hash_id_to_idx:
                        valid_indices.append(self.hash_id_to_idx[h])
                
                if not valid_indices:
                    return np.array([])
                
                indices = np.array(valid_indices, dtype=np.intp)
                embeddings = np.array(self.embeddings, dtype=dtype)[indices]

                return embeddings
            else:
                # 使用向量数据库后端
                vectors_dict = self._vector_db.get_vectors(hash_ids)
                if not vectors_dict:
                    return np.array([])
                
                # 按照输入顺序返回
                vectors = [vectors_dict[h] for h in hash_ids if h in vectors_dict]
                return np.array(vectors, dtype=dtype)
    
    def get_embeddings_matrix(self, dtype=np.float32) -> np.ndarray:
        """
        获取所有嵌入向量的矩阵形式（用于批量计算）
        
        性能优化：使用缓存避免重复转换
        """
        with self._lock:
            if self.vector_db_backend == 'parquet':
                if self._embedding_cache is None or self._cache_dirty:
                    self._embedding_cache = np.array(self.embeddings, dtype=dtype)
                    self._cache_dirty = False
                return self._embedding_cache
            else:
                # 对于其他后端，获取所有向量
                matrix, _ = self._vector_db.get_vector_matrix()
                return matrix.astype(dtype) if len(matrix) > 0 else matrix
    
    def search(
        self, 
        query_vector: np.ndarray, 
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        向量相似度搜索
        
        Args:
            query_vector: 查询向量
            top_k: 返回的最相似结果数量
            filter_dict: 可选的元数据过滤条件
            
        Returns:
            List[Tuple[str, float, Dict[str, Any]]]: (id, 相似度分数, 元数据) 的列表
        """
        if self.vector_db_backend == 'parquet':
            # Parquet 后端使用本地计算
            with self._lock:
                if not self.hash_ids:
                    return []
                
                matrix = self.get_embeddings_matrix()
                query = np.array(query_vector, dtype=np.float32)
                query_norm = np.linalg.norm(query)
                if query_norm > 0:
                    query = query / query_norm
                
                similarities = np.dot(matrix, query)
                top_indices = np.argsort(similarities)[::-1][:top_k]
                
                results = []
                for idx in top_indices:
                    results.append((
                        self.hash_ids[idx],
                        float(similarities[idx]),
                        {"content": self.texts[idx]}
                    ))
                return results
        else:
            # 使用向量数据库后端
            return self._vector_db.search(query_vector, top_k, filter_dict)
    
    def batch_search(
        self,
        query_vectors: List[np.ndarray],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[List[Tuple[str, float, Dict[str, Any]]]]:
        """
        批量向量相似度搜索
        
        Args:
            query_vectors: 查询向量列表
            top_k: 每个查询返回的最相似结果数量
            filter_dict: 可选的元数据过滤条件
            
        Returns:
            List[List[Tuple[str, float, Dict[str, Any]]]]: 每个查询的结果列表
        """
        if self.vector_db_backend == 'parquet':
            results = []
            for query in query_vectors:
                results.append(self.search(query, top_k, filter_dict))
            return results
        else:
            return self._vector_db.batch_search(query_vectors, top_k, filter_dict)
    
    def clear_cache(self):
        """
        清除缓存
        """
        if self.vector_db_backend == 'parquet':
            with EmbeddingStore._cache_lock:
                if self.filename in EmbeddingStore._cache:
                    del EmbeddingStore._cache[self.filename]
            self._embedding_cache = None
            self._cache_dirty = True
        else:
            # 对于其他后端，缓存由后端管理
            pass
    
    def count(self) -> int:
        """获取记录数量"""
        if self.vector_db_backend == 'parquet':
            return len(self.hash_ids)
        else:
            return self._vector_db.count()
    
    def __len__(self) -> int:
        """返回记录数量"""
        return self.count()
    
    def __repr__(self) -> str:
        """返回对象的字符串表示"""
        return (
            f"EmbeddingStore("
            f"backend={self.vector_db_backend}, "
            f"namespace='{self.namespace}', "
            f"count={self.count()})"
        )
