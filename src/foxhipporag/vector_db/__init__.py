"""
向量数据库集成模块

支持的向量数据库后端：
- Parquet: 默认的本地文件存储（基于 Parquet 格式）
- FAISS: Facebook AI Similarity Search，高效的本地向量检索
- Milvus: 开源分布式向量数据库
- Chroma: 轻量级嵌入式向量数据库
"""

from .base import BaseVectorDB
from .parquet_db import ParquetVectorDB
from .faiss_db import FAISSVectorDB

# 可选导入，仅在安装了相应依赖时可用
try:
    from .milvus_db import MilvusVectorDB
except ImportError:
    MilvusVectorDB = None

try:
    from .chroma_db import ChromaVectorDB
except ImportError:
    ChromaVectorDB = None


def get_vector_db(backend: str, **kwargs) -> BaseVectorDB:
    """
    工厂函数：根据配置获取向量数据库实例
    
    Args:
        backend: 向量数据库后端类型 ('parquet', 'faiss', 'milvus', 'chroma')
        **kwargs: 传递给具体实现的参数
        
    Returns:
        BaseVectorDB: 向量数据库实例
        
    Raises:
        ValueError: 不支持的后端类型
        ImportError: 未安装必要的依赖
    """
    backend = backend.lower()
    
    if backend == 'parquet':
        return ParquetVectorDB(**kwargs)
    elif backend == 'faiss':
        return FAISSVectorDB(**kwargs)
    elif backend == 'milvus':
        if MilvusVectorDB is None:
            raise ImportError(
                "Milvus 支持未安装。请运行: pip install pymilvus>=2.3.0"
            )
        return MilvusVectorDB(**kwargs)
    elif backend == 'chroma':
        if ChromaVectorDB is None:
            raise ImportError(
                "Chroma 支持未安装。请运行: pip install chromadb>=0.4.0"
            )
        return ChromaVectorDB(**kwargs)
    else:
        raise ValueError(
            f"不支持的向量数据库后端: {backend}。"
            f"支持的后端: parquet, faiss, milvus, chroma"
        )


__all__ = [
    'BaseVectorDB',
    'ParquetVectorDB', 
    'FAISSVectorDB',
    'MilvusVectorDB',
    'ChromaVectorDB',
    'get_vector_db',
]
