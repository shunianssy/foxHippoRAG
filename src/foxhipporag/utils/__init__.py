"""
foxHippoRAG 工具模块

提供各种辅助工具和优化函数。
"""

from .rust_utils import (
    is_rust_available,
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
    count_entities,
    count_facts,
    compute_content_richness,
    batch_count_entities,
    batch_count_facts,
    compute_density_score,
    batch_compute_density,
    compute_evidence_score,
    batch_evidence_scores,
    fuse_scores,
    multiplicative_fuse,
)

__all__ = [
    # Rust 可用性检查
    "is_rust_available",
    "get_backend_info",
    # 相似度计算
    "cosine_similarity",
    "cosine_similarity_batch",
    # Top-K 选择
    "top_k_indices",
    "top_k_indices_2d",
    # 归一化
    "l2_normalize",
    "min_max_normalize",
    "batch_l2_normalize",
    # KNN 检索
    "knn_search",
    "knn_search_batch",
    "retrieve_knn",
    # 文本特征
    "count_entities",
    "count_facts",
    "compute_content_richness",
    "batch_count_entities",
    "batch_count_facts",
    # 段落密度
    "compute_density_score",
    "batch_compute_density",
    "compute_evidence_score",
    "batch_evidence_scores",
    # 分数融合
    "fuse_scores",
    "multiplicative_fuse",
]
