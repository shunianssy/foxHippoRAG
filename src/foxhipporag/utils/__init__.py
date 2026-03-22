"""
foxHippoRAG 工具模块

提供各种辅助工具和优化函数。
"""

from .c_utils import (
    is_c_extension_available,
    get_backend_info,
    cosine_similarity,
    cosine_similarity_batch,
    top_k_indices,
    top_k_indices_2d,
    l2_normalize,
    min_max_normalize,
    batch_l2_normalize,
    knn_search,
    knn_search_batch,
    retrieve_knn,
    fuse_scores,
    multiplicative_fuse,
)

__all__ = [
    # C扩展可用性检查
    "is_c_extension_available",
    "get_backend_info",
    # 相似度计算
    "cosine_similarity",
    "cosine_similarity_batch",
    # Top-K选择
    "top_k_indices",
    "top_k_indices_2d",
    # 归一化
    "l2_normalize",
    "min_max_normalize",
    "batch_l2_normalize",
    # KNN检索
    "knn_search",
    "knn_search_batch",
    "retrieve_knn",
    # 分数融合
    "fuse_scores",
    "multiplicative_fuse",
]
