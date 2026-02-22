"""
向量数据库抽象基类

定义了所有向量数据库后端必须实现的接口。
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BaseVectorDB(ABC):
    """
    向量数据库抽象基类
    
    所有向量数据库后端必须继承此类并实现所有抽象方法。
    提供统一的接口用于：
    - 向量插入和更新
    - 向量检索（KNN）
    - 向量删除
    - 元数据管理
    """
    
    def __init__(
        self,
        embedding_dim: int,
        namespace: str = "default",
        **kwargs
    ):
        """
        初始化向量数据库
        
        Args:
            embedding_dim: 嵌入向量维度
            namespace: 命名空间，用于隔离不同类型的数据
            **kwargs: 子类特定的参数
        """
        self.embedding_dim = embedding_dim
        self.namespace = namespace
        self._initialized = False
        
    @abstractmethod
    def insert(
        self,
        ids: List[str],
        vectors: List[np.ndarray],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> bool:
        """
        插入向量数据
        
        Args:
            ids: 唯一标识符列表
            vectors: 嵌入向量列表
            metadatas: 可选的元数据列表
            **kwargs: 额外参数
            
        Returns:
            bool: 插入是否成功
        """
        pass
    
    @abstractmethod
    def upsert(
        self,
        ids: List[str],
        vectors: List[np.ndarray],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> bool:
        """
        插入或更新向量数据
        
        Args:
            ids: 唯一标识符列表
            vectors: 嵌入向量列表
            metadatas: 可选的元数据列表
            **kwargs: 额外参数
            
        Returns:
            bool: 操作是否成功
        """
        pass
    
    @abstractmethod
    def delete(self, ids: List[str], **kwargs) -> bool:
        """
        删除指定ID的向量
        
        Args:
            ids: 要删除的ID列表
            **kwargs: 额外参数
            
        Returns:
            bool: 删除是否成功
        """
        pass
    
    @abstractmethod
    def get_vectors(
        self,
        ids: List[str],
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        根据ID获取向量
        
        Args:
            ids: ID列表
            **kwargs: 额外参数
            
        Returns:
            Dict[str, np.ndarray]: ID到向量的映射
        """
        pass
    
    @abstractmethod
    def get_metadatas(
        self,
        ids: List[str],
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """
        根据ID获取元数据
        
        Args:
            ids: ID列表
            **kwargs: 额外参数
            
        Returns:
            Dict[str, Dict[str, Any]]: ID到元数据的映射
        """
        pass
    
    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        向量相似度搜索
        
        Args:
            query_vector: 查询向量
            top_k: 返回的最相似结果数量
            filter_dict: 可选的元数据过滤条件
            **kwargs: 额外参数
            
        Returns:
            List[Tuple[str, float, Dict[str, Any]]]: 
                (id, 相似度分数, 元数据) 的列表
        """
        pass
    
    @abstractmethod
    def batch_search(
        self,
        query_vectors: List[np.ndarray],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[List[Tuple[str, float, Dict[str, Any]]]]:
        """
        批量向量相似度搜索
        
        Args:
            query_vectors: 查询向量列表
            top_k: 每个查询返回的最相似结果数量
            filter_dict: 可选的元数据过滤条件
            **kwargs: 额外参数
            
        Returns:
            List[List[Tuple[str, float, Dict[str, Any]]]]: 
                每个查询的结果列表
        """
        pass
    
    @abstractmethod
    def get_all_ids(self, **kwargs) -> List[str]:
        """
        获取所有向量ID
        
        Args:
            **kwargs: 额外参数
            
        Returns:
            List[str]: 所有ID列表
        """
        pass
    
    @abstractmethod
    def count(self, **kwargs) -> int:
        """
        获取向量总数
        
        Args:
            **kwargs: 额外参数
            
        Returns:
            int: 向量总数
        """
        pass
    
    @abstractmethod
    def clear(self, **kwargs) -> bool:
        """
        清空所有数据
        
        Args:
            **kwargs: 额外参数
            
        Returns:
            bool: 清空是否成功
        """
        pass
    
    @abstractmethod
    def save(self, path: Optional[str] = None, **kwargs) -> bool:
        """
        保存数据到磁盘
        
        Args:
            path: 保存路径，如果为None则使用默认路径
            **kwargs: 额外参数
            
        Returns:
            bool: 保存是否成功
        """
        pass
    
    @abstractmethod
    def load(self, path: Optional[str] = None, **kwargs) -> bool:
        """
        从磁盘加载数据
        
        Args:
            path: 加载路径，如果为None则使用默认路径
            **kwargs: 额外参数
            
        Returns:
            bool: 加载是否成功
        """
        pass
    
    def exists(self, id: str, **kwargs) -> bool:
        """
        检查指定ID是否存在
        
        Args:
            id: 要检查的ID
            **kwargs: 额外参数
            
        Returns:
            bool: ID是否存在
        """
        vectors = self.get_vectors([id], **kwargs)
        return id in vectors
    
    def batch_exists(self, ids: List[str], **kwargs) -> Dict[str, bool]:
        """
        批量检查ID是否存在
        
        Args:
            ids: 要检查的ID列表
            **kwargs: 额外参数
            
        Returns:
            Dict[str, bool]: ID到存在状态的映射
        """
        vectors = self.get_vectors(ids, **kwargs)
        return {id: id in vectors for id in ids}
    
    def get_vector_matrix(self, **kwargs) -> Tuple[np.ndarray, List[str]]:
        """
        获取所有向量组成的矩阵
        
        Args:
            **kwargs: 额外参数
            
        Returns:
            Tuple[np.ndarray, List[str]]: (向量矩阵, ID列表)
        """
        ids = self.get_all_ids(**kwargs)
        if not ids:
            return np.array([]), []
        
        vectors_dict = self.get_vectors(ids, **kwargs)
        vectors = [vectors_dict[id] for id in ids if id in vectors_dict]
        
        if not vectors:
            return np.array([]), []
        
        return np.vstack(vectors), ids
    
    def __len__(self) -> int:
        """返回向量数量"""
        return self.count()
    
    def __repr__(self) -> str:
        """返回对象的字符串表示"""
        return (
            f"{self.__class__.__name__}("
            f"embedding_dim={self.embedding_dim}, "
            f"namespace='{self.namespace}', "
            f"count={self.count()})"
        )
