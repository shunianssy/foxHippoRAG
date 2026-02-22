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

logger = logging.getLogger(__name__)

class EmbeddingStore:
    """
    嵌入向量存储类
    
    性能优化版本：
    - 支持增量保存，避免全量重写
    - 支持线程安全的并发访问
    - 支持内存缓存优化
    - 支持批量操作优化
    """
    
    # 类级别的缓存，避免重复加载
    _cache = {}
    _cache_lock = threading.Lock()
    
    def __init__(self, embedding_model, db_filename, batch_size, namespace):
        """
        Initializes the class with necessary configurations and sets up the working directory.

        Parameters:
        embedding_model: The model used for embeddings.
        db_filename: The directory path where data will be stored or retrieved.
        batch_size: The batch size used for processing.
        namespace: A unique identifier for data segregation.

        Functionality:
        - Assigns the provided parameters to instance variables.
        - Checks if the directory specified by `db_filename` exists.
          - If not, creates the directory and logs the operation.
        - Constructs the filename for storing data in a parquet file format.
        - Calls the method `_load_data()` to initialize the data loading process.
        """
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.namespace = namespace
        self._lock = threading.RLock()  # 可重入锁，支持线程安全
        
        # 增量保存计数器
        self._pending_changes = 0
        self._auto_save_threshold = 100  # 每100条记录自动保存

        if not os.path.exists(db_filename):
            logger.info(f"Creating working directory: {db_filename}")
            os.makedirs(db_filename, exist_ok=True)

        self.filename = os.path.join(
            db_filename, f"vdb_{self.namespace}.parquet"
        )
        
        # 嵌入向量缓存（用于快速检索）
        self._embedding_cache = None
        self._cache_dirty = True
        
        self._load_data()

    def get_missing_string_hash_ids(self, texts: List[str]):
        with self._lock:
            nodes_dict = {}

            for text in texts:
                nodes_dict[compute_mdhash_id(text, prefix=self.namespace + "-")] = {'content': text}

            # Get all hash_ids from the input dictionary.
            all_hash_ids = list(nodes_dict.keys())
            if not all_hash_ids:
                return  {}

            existing = self.hash_id_to_row.keys()

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

            existing = self.hash_id_to_row.keys()

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

            self._upsert(missing_ids, texts_to_encode, missing_embeddings)

    def _load_data(self):
        """
        加载数据，支持缓存
        """
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
        保存数据到磁盘，同时更新缓存
        """
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
        插入或更新数据
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

    def get_row(self, hash_id):
        return self.hash_id_to_row.get(hash_id)

    def get_hash_id(self, text):
        return self.text_to_hash_id.get(text)

    def get_rows(self, hash_ids, dtype=np.float32):
        if not hash_ids:
            return {}

        results = {id: self.hash_id_to_row[id] for id in hash_ids if id in self.hash_id_to_row}

        return results

    def get_all_ids(self):
        return deepcopy(self.hash_ids)

    def get_all_id_to_rows(self):
        return deepcopy(self.hash_id_to_row)

    def get_all_texts(self):
        return set(row['content'] for row in self.hash_id_to_row.values())

    def get_embedding(self, hash_id, dtype=np.float32) -> np.ndarray:
        """
        获取单个嵌入向量
        """
        if hash_id not in self.hash_id_to_idx:
            return None
        return self.embeddings[self.hash_id_to_idx[hash_id]].astype(dtype)
    
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
    
    def get_embeddings_matrix(self, dtype=np.float32) -> np.ndarray:
        """
        获取所有嵌入向量的矩阵形式（用于批量计算）
        
        性能优化：使用缓存避免重复转换
        """
        with self._lock:
            if self._embedding_cache is None or self._cache_dirty:
                self._embedding_cache = np.array(self.embeddings, dtype=dtype)
                self._cache_dirty = False
            return self._embedding_cache
    
    def clear_cache(self):
        """
        清除缓存
        """
        with EmbeddingStore._cache_lock:
            if self.filename in EmbeddingStore._cache:
                del EmbeddingStore._cache[self.filename]
        self._embedding_cache = None
        self._cache_dirty = True