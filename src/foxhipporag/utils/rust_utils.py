"""
Rust 加速工具模块

该模块提供高性能的 Rust 实现，用于加速 foxHippoRAG 的核心计算。

模块结构：
- similarity: 余弦相似度计算
- topk: Top-K 选择算法
- normalize: 向量归一化
- knn: KNN 检索
- text: 文本特征计算
- density: 段落密度评估

使用方法：
    from foxhipporag.utils.rust_utils import (
        cosine_similarity,
        top_k_indices,
        knn_search,
        batch_compute_density,
    )

性能优势：
- 余弦相似度：2-5x 加速
- Top-K 选择：3-10x 加速
- KNN 检索：2-4x 加速
- 文本特征计算：5-20x 加速
"""

import logging
import os
from typing import List, Dict, Tuple, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# 尝试导入 Rust 扩展
RUST_AVAILABLE = False
try:
    import foxhipporag_rust
    RUST_AVAILABLE = True
    logger.info("foxhipporag_rust Rust 扩展已启用")
except ImportError:
    logger.debug("foxhipporag_rust 未安装，使用 Python 回退实现")


def is_rust_available() -> bool:
    """检查 Rust 扩展是否可用"""
    return RUST_AVAILABLE


def get_backend_info() -> Dict:
    """获取后端信息"""
    info = {
        "rust_available": RUST_AVAILABLE,
        "backend": "rust" if RUST_AVAILABLE else "python",
    }
    
    if RUST_AVAILABLE:
        try:
            info["rust_version"] = foxhipporag_rust.__version__
            info["rust_info"] = foxhipporag_rust.get_rust_info()
            info["num_threads"] = foxhipporag_rust.get_num_threads()
        except Exception as e:
            info["error"] = str(e)
    
    return info


# ============================================================
# 余弦相似度计算
# ============================================================

def cosine_similarity(
    query: np.ndarray,
    matrix: np.ndarray,
    use_rust: Optional[bool] = None
) -> np.ndarray:
    """
    计算单个查询向量与矩阵中所有向量的余弦相似度
    
    Args:
        query: 查询向量 (dim,)
        matrix: 向量矩阵 (n, dim)
        use_rust: 是否使用 Rust 加速（None 表示自动检测）
        
    Returns:
        余弦相似度数组 (n,)
    """
    if use_rust is None:
        use_rust = RUST_AVAILABLE
    
    query = np.ascontiguousarray(query, dtype=np.float32)
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    
    if use_rust:
        try:
            return foxhipporag_rust.cosine_similarity(query, matrix)
        except Exception as e:
            logger.warning(f"Rust cosine_similarity 失败，回退到 NumPy: {e}")
    
    # NumPy 回退实现
    query_norm = np.linalg.norm(query)
    if query_norm < 1e-10:
        return np.zeros(matrix.shape[0], dtype=np.float32)
    
    matrix_norms = np.linalg.norm(matrix, axis=1)
    matrix_norms = np.maximum(matrix_norms, 1e-10)
    
    return np.dot(matrix, query) / (matrix_norms * query_norm)


def cosine_similarity_batch(
    queries: np.ndarray,
    keys: np.ndarray,
    use_rust: Optional[bool] = None
) -> np.ndarray:
    """
    批量计算查询向量矩阵与键向量矩阵的余弦相似度
    
    Args:
        queries: 查询向量矩阵 (m, dim)
        keys: 键向量矩阵 (n, dim)
        use_rust: 是否使用 Rust 加速
        
    Returns:
        相似度矩阵 (m, n)
    """
    if use_rust is None:
        use_rust = RUST_AVAILABLE
    
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    keys = np.ascontiguousarray(keys, dtype=np.float32)
    
    if use_rust:
        try:
            return foxhipporag_rust.cosine_similarity_batch(queries, keys)
        except Exception as e:
            logger.warning(f"Rust cosine_similarity_batch 失败，回退到 NumPy: {e}")
    
    # NumPy 回退实现
    query_norms = np.linalg.norm(queries, axis=1, keepdims=True)
    query_norms = np.maximum(query_norms, 1e-10)
    queries_normalized = queries / query_norms
    
    key_norms = np.linalg.norm(keys, axis=1, keepdims=True)
    key_norms = np.maximum(key_norms, 1e-10)
    keys_normalized = keys / key_norms
    
    return np.dot(queries_normalized, keys_normalized.T)


# ============================================================
# Top-K 选择
# ============================================================

def top_k_indices(
    scores: np.ndarray,
    k: int,
    use_rust: Optional[bool] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从一维数组中选择 Top-K 元素
    
    Args:
        scores: 得分数组 (n,)
        k: 要选择的数量
        use_rust: 是否使用 Rust 加速
        
    Returns:
        (indices, scores): Top-K 索引和得分
    """
    if use_rust is None:
        use_rust = RUST_AVAILABLE
    
    scores = np.ascontiguousarray(scores, dtype=np.float32)
    k = min(k, len(scores))
    
    if use_rust:
        try:
            return foxhipporag_rust.top_k_indices(scores, k)
        except Exception as e:
            logger.warning(f"Rust top_k_indices 失败，回退到 NumPy: {e}")
    
    # NumPy 回退实现
    indices = np.argsort(scores)[::-1][:k]
    return indices.astype(np.int64), scores[indices]


def top_k_indices_2d(
    scores: np.ndarray,
    k: int,
    use_rust: Optional[bool] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从二维矩阵的每一行选择 Top-K 元素
    
    Args:
        scores: 得分矩阵 (m, n)
        k: 每行要选择的数量
        use_rust: 是否使用 Rust 加速
        
    Returns:
        (indices, scores): Top-K 索引和得分矩阵 (m, k)
    """
    if use_rust is None:
        use_rust = RUST_AVAILABLE
    
    scores = np.ascontiguousarray(scores, dtype=np.float32)
    k = min(k, scores.shape[1])
    
    if use_rust:
        try:
            return foxhipporag_rust.top_k_indices_2d(scores, k)
        except Exception as e:
            logger.warning(f"Rust top_k_indices_2d 失败，回退到 NumPy: {e}")
    
    # NumPy 回退实现
    indices = np.argsort(scores, axis=1)[:, ::-1][:, :k]
    top_scores = np.take_along_axis(scores, indices, axis=1)
    return indices.astype(np.int64), top_scores


# ============================================================
# 归一化
# ============================================================

def l2_normalize(
    vector: np.ndarray,
    use_rust: Optional[bool] = None
) -> np.ndarray:
    """
    L2 归一化向量
    
    Args:
        vector: 输入向量
        use_rust: 是否使用 Rust 加速
        
    Returns:
        归一化后的向量
    """
    if use_rust is None:
        use_rust = RUST_AVAILABLE
    
    vector = np.ascontiguousarray(vector, dtype=np.float32)
    
    if use_rust:
        try:
            return foxhipporag_rust.l2_normalize(vector)
        except Exception as e:
            logger.warning(f"Rust l2_normalize 失败，回退到 NumPy: {e}")
    
    # NumPy 回退实现
    norm = np.linalg.norm(vector)
    if norm < 1e-10:
        return np.zeros_like(vector)
    return vector / norm


def min_max_normalize(
    values: np.ndarray,
    use_rust: Optional[bool] = None
) -> np.ndarray:
    """
    Min-Max 归一化
    
    Args:
        values: 输入数组
        use_rust: 是否使用 Rust 加速
        
    Returns:
        归一化后的数组
    """
    if use_rust is None:
        use_rust = RUST_AVAILABLE
    
    values = np.ascontiguousarray(values, dtype=np.float32)
    
    if use_rust:
        try:
            return foxhipporag_rust.min_max_normalize(values)
        except Exception as e:
            logger.warning(f"Rust min_max_normalize 失败，回退到 NumPy: {e}")
    
    # NumPy 回退实现
    min_val = np.min(values)
    max_val = np.max(values)
    range_val = max_val - min_val
    
    if range_val < 1e-10:
        return np.ones_like(values)
    
    return (values - min_val) / range_val


def batch_l2_normalize(
    matrix: np.ndarray,
    use_rust: Optional[bool] = None
) -> np.ndarray:
    """
    批量 L2 归一化二维矩阵（每行独立归一化）
    
    Args:
        matrix: 输入矩阵 (m, n)
        use_rust: 是否使用 Rust 加速
        
    Returns:
        归一化后的矩阵 (m, n)
    """
    if use_rust is None:
        use_rust = RUST_AVAILABLE
    
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    
    if use_rust:
        try:
            return foxhipporag_rust.batch_l2_normalize(matrix)
        except Exception as e:
            logger.warning(f"Rust batch_l2_normalize 失败，回退到 NumPy: {e}")
    
    # NumPy 回退实现
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return matrix / norms


# ============================================================
# KNN 检索
# ============================================================

def knn_search(
    query: np.ndarray,
    index_vectors: np.ndarray,
    k: int = 10,
    use_rust: Optional[bool] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    KNN 检索
    
    Args:
        query: 查询向量 (dim,)
        index_vectors: 索引向量矩阵 (n, dim)
        k: 返回的最近邻数量
        use_rust: 是否使用 Rust 加速
        
    Returns:
        (indices, scores): 最近邻索引和相似度得分
    """
    if use_rust is None:
        use_rust = RUST_AVAILABLE
    
    query = np.ascontiguousarray(query, dtype=np.float32)
    index_vectors = np.ascontiguousarray(index_vectors, dtype=np.float32)
    
    # 确保查询向量是 2D
    if query.ndim == 1:
        query = query.reshape(1, -1)
    
    if use_rust:
        try:
            return foxhipporag_rust.knn_search(query, index_vectors, k)
        except Exception as e:
            logger.warning(f"Rust knn_search 失败，回退到 NumPy: {e}")
    
    # NumPy 回退实现
    similarities = cosine_similarity(query.flatten(), index_vectors, use_rust=False)
    return top_k_indices(similarities, k, use_rust=False)


def knn_search_batch(
    queries: np.ndarray,
    index_vectors: np.ndarray,
    k: int = 10,
    use_rust: Optional[bool] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    批量 KNN 检索
    
    Args:
        queries: 查询向量矩阵 (m, dim)
        index_vectors: 索引向量矩阵 (n, dim)
        k: 每个查询返回的最近邻数量
        use_rust: 是否使用 Rust 加速
        
    Returns:
        (indices, scores): Top-K 索引和得分矩阵 (m, k)
    """
    if use_rust is None:
        use_rust = RUST_AVAILABLE
    
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    index_vectors = np.ascontiguousarray(index_vectors, dtype=np.float32)
    
    if use_rust:
        try:
            return foxhipporag_rust.knn_search_batch(queries, index_vectors, k)
        except Exception as e:
            logger.warning(f"Rust knn_search_batch 失败，回退到 NumPy: {e}")
    
    # NumPy 回退实现
    similarity_matrix = cosine_similarity_batch(queries, index_vectors, use_rust=False)
    return top_k_indices_2d(similarity_matrix, k, use_rust=False)


def retrieve_knn(
    query_ids: List[str],
    key_ids: List[str],
    query_vecs: np.ndarray,
    key_vecs: np.ndarray,
    k: int = 2047,
    query_batch_size: int = 1000,
    key_batch_size: int = 10000,
    use_rust: Optional[bool] = None
) -> Dict:
    """
    完整的 KNN 检索接口
    
    Args:
        query_ids: 查询向量 ID 列表
        key_ids: 键向量 ID 列表
        query_vecs: 查询向量矩阵 (m, dim)
        key_vecs: 键向量矩阵 (n, dim)
        k: 返回的最近邻数量
        query_batch_size: 查询批处理大小
        key_batch_size: 键批处理大小
        use_rust: 是否使用 Rust 加速
        
    Returns:
        Dict: 查询 ID 到 (key_ids, scores) 的映射
    """
    if use_rust is None:
        use_rust = RUST_AVAILABLE
    
    query_vecs = np.ascontiguousarray(query_vecs, dtype=np.float32)
    key_vecs = np.ascontiguousarray(key_vecs, dtype=np.float32)
    
    if use_rust:
        try:
            return foxhipporag_rust.retrieve_knn(
                query_ids, key_ids, query_vecs, key_vecs,
                k, query_batch_size, key_batch_size
            )
        except Exception as e:
            logger.warning(f"Rust retrieve_knn 失败，回退到 Python: {e}")
    
    # Python 回退实现（使用现有的 embed_utils）
    from .embed_utils import retrieve_knn as py_retrieve_knn
    return py_retrieve_knn(
        query_ids, key_ids, query_vecs, key_vecs,
        k, query_batch_size, key_batch_size
    )


# ============================================================
# 文本特征计算
# ============================================================

def count_entities(text: str, use_rust: Optional[bool] = None) -> int:
    """
    计算文本中的实体数量
    
    Args:
        text: 输入文本
        use_rust: 是否使用 Rust 加速
        
    Returns:
        实体数量
    """
    if use_rust is None:
        use_rust = RUST_AVAILABLE
    
    if use_rust:
        try:
            return foxhipporag_rust.count_entities(text)
        except Exception as e:
            logger.warning(f"Rust count_entities 失败，回退到 Python: {e}")
    
    # Python 回退实现
    import re
    patterns = [
        re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'),
        re.compile(r'\b\d+(?:\.\d+)?(?:\s*(?:年|月|日|时|分|秒|km|m|kg|元|美元|亿|万))?\b'),
        re.compile(r'\b[A-Z]{2,}(?:\s+[A-Z]{2,})*\b'),
    ]
    
    count = 0
    for pattern in patterns:
        count += len(pattern.findall(text))
    return count


def count_facts(text: str, use_rust: Optional[bool] = None) -> int:
    """
    估算文本中的事实/关系数量
    
    Args:
        text: 输入文本
        use_rust: 是否使用 Rust 加速
        
    Returns:
        估算的事实数量
    """
    if use_rust is None:
        use_rust = RUST_AVAILABLE
    
    if use_rust:
        try:
            return foxhipporag_rust.count_facts(text)
        except Exception as e:
            logger.warning(f"Rust count_facts 失败，回退到 Python: {e}")
    
    # Python 回退实现
    import re
    cn_patterns = ['是', '有', '位于', '属于', '包含', '包括', '成立于', '创建于',
                   '发明', '发现', '生产', '制造', '设计', '开发', '建立', '设立']
    en_patterns = [r'\bis\b', r'\bare\b', r'\bwas\b', r'\bwere\b',
                   r'\blocated\b', r'\bbelongs\b', r'\bcontains\b', r'\bincludes\b',
                   r'\bfounded\b', r'\bcreated\b', r'\binvented\b', r'\bdiscovered\b',
                   r'\bproduced\b', r'\bmanufactured\b', r'\bdesigned\b', r'\bdeveloped\b']
    
    count = 0
    for pattern in cn_patterns:
        count += len(re.findall(pattern, text))
    for pattern in en_patterns:
        count += len(re.findall(pattern, text, re.IGNORECASE))
    return count


def compute_content_richness(text: str, use_rust: Optional[bool] = None) -> float:
    """
    计算内容丰富度
    
    Args:
        text: 输入文本
        use_rust: 是否使用 Rust 加速
        
    Returns:
        内容丰富度得分 (0-1)
    """
    if use_rust is None:
        use_rust = RUST_AVAILABLE
    
    if use_rust:
        try:
            return foxhipporag_rust.compute_content_richness(text)
        except Exception as e:
            logger.warning(f"Rust compute_content_richness 失败，回退到 Python: {e}")
    
    # Python 回退实现
    import re
    stopwords = {
        '的', '了', '是', '在', '有', '和', '与', '或', '等', '这', '那',
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
        'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
        'into', 'through', 'during', 'before', 'after', 'above', 'below',
    }
    
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return 0.0
    
    non_stopword_count = sum(1 for w in words if w not in stopwords)
    non_stopword_ratio = non_stopword_count / len(words)
    
    unique_words = set(words)
    vocabulary_diversity = len(unique_words) / len(words)
    
    return min(1.0, 0.6 * non_stopword_ratio + 0.4 * vocabulary_diversity)


def batch_count_entities(texts: List[str], use_rust: Optional[bool] = None) -> List[int]:
    """
    批量计算实体数量
    
    Args:
        texts: 文本列表
        use_rust: 是否使用 Rust 加速
        
    Returns:
        实体数量列表
    """
    if use_rust is None:
        use_rust = RUST_AVAILABLE
    
    if use_rust:
        try:
            return foxhipporag_rust.batch_count_entities(texts)
        except Exception as e:
            logger.warning(f"Rust batch_count_entities 失败，回退到 Python: {e}")
    
    return [count_entities(t, use_rust=False) for t in texts]


def batch_count_facts(texts: List[str], use_rust: Optional[bool] = None) -> List[int]:
    """
    批量计算事实数量
    
    Args:
        texts: 文本列表
        use_rust: 是否使用 Rust 加速
        
    Returns:
        事实数量列表
    """
    if use_rust is None:
        use_rust = RUST_AVAILABLE
    
    if use_rust:
        try:
            return foxhipporag_rust.batch_count_facts(texts)
        except Exception as e:
            logger.warning(f"Rust batch_count_facts 失败，回退到 Python: {e}")
    
    return [count_facts(t, use_rust=False) for t in texts]


# ============================================================
# 段落密度评估
# ============================================================

def compute_density_score(
    text: str,
    entity_weight: float = 0.3,
    fact_weight: float = 0.3,
    length_weight: float = 0.2,
    richness_weight: float = 0.2,
    use_rust: Optional[bool] = None
) -> float:
    """
    计算单个段落的密度得分
    
    Args:
        text: 段落文本
        entity_weight: 实体数量权重
        fact_weight: 事实数量权重
        length_weight: 文本长度权重
        richness_weight: 内容丰富度权重
        use_rust: 是否使用 Rust 加速
        
    Returns:
        密度得分 (0-1)
    """
    if use_rust is None:
        use_rust = RUST_AVAILABLE
    
    if use_rust:
        try:
            return foxhipporag_rust.compute_density_score(
                text, entity_weight, fact_weight, length_weight, richness_weight
            )
        except Exception as e:
            logger.warning(f"Rust compute_density_score 失败，回退到 Python: {e}")
    
    # Python 回退实现
    entity_count = count_entities(text, use_rust=False)
    fact_count = count_facts(text, use_rust=False)
    text_length_score = _compute_text_length_score(text)
    content_richness = compute_content_richness(text, use_rust=False)
    
    normalized_entity_score = min(1.0, np.log1p(entity_count) / 3.0)
    normalized_fact_score = min(1.0, np.log1p(fact_count) / 2.5)
    
    density_score = (
        entity_weight * normalized_entity_score +
        fact_weight * normalized_fact_score +
        length_weight * text_length_score +
        richness_weight * content_richness
    )
    
    return density_score


def _compute_text_length_score(text: str) -> float:
    """计算文本长度得分"""
    length = len(text)
    optimal_length = 350
    max_length = 1000
    
    if length == 0:
        return 0.0
    elif length <= optimal_length:
        return length / optimal_length
    else:
        return max(0.5, 1.0 - (length - optimal_length) / (max_length - optimal_length) * 0.5)


def batch_compute_density(
    texts: List[str],
    entity_weight: float = 0.3,
    fact_weight: float = 0.3,
    length_weight: float = 0.2,
    richness_weight: float = 0.2,
    use_rust: Optional[bool] = None
) -> np.ndarray:
    """
    批量计算段落密度得分
    
    Args:
        texts: 段落文本列表
        entity_weight: 实体数量权重
        fact_weight: 事实数量权重
        length_weight: 文本长度权重
        richness_weight: 内容丰富度权重
        use_rust: 是否使用 Rust 加速
        
    Returns:
        密度得分数组
    """
    if use_rust is None:
        use_rust = RUST_AVAILABLE
    
    if use_rust:
        try:
            return foxhipporag_rust.batch_compute_density(
                texts, entity_weight, fact_weight, length_weight, richness_weight
            )
        except Exception as e:
            logger.warning(f"Rust batch_compute_density 失败，回退到 Python: {e}")
    
    # Python 回退实现
    scores = [
        compute_density_score(t, entity_weight, fact_weight, length_weight, richness_weight, use_rust=False)
        for t in texts
    ]
    return np.array(scores, dtype=np.float32)


def compute_evidence_score(
    semantic_score: float,
    density_score: float,
    query_type: str = 'general',
    use_rust: Optional[bool] = None
) -> float:
    """
    计算证据质量得分
    
    Args:
        semantic_score: 语义相似度得分 (0-1)
        density_score: 信息密度得分 (0-1)
        query_type: 查询类型 ('general', 'factual', 'exploratory')
        use_rust: 是否使用 Rust 加速
        
    Returns:
        证据质量得分 (0-1)
    """
    if use_rust is None:
        use_rust = RUST_AVAILABLE
    
    if use_rust:
        try:
            return foxhipporag_rust.compute_evidence_score(semantic_score, density_score, query_type)
        except Exception as e:
            logger.warning(f"Rust compute_evidence_score 失败，回退到 Python: {e}")
    
    # Python 回退实现
    if query_type == 'factual':
        semantic_weight, density_weight = 0.7, 0.3
    elif query_type == 'exploratory':
        semantic_weight, density_weight = 0.4, 0.6
    else:
        semantic_weight, density_weight = 0.5, 0.5
    
    multiplicative_score = semantic_score * density_score
    weighted_score = semantic_weight * semantic_score + density_weight * density_score
    
    return 0.3 * multiplicative_score + 0.7 * weighted_score


def batch_evidence_scores(
    semantic_scores: np.ndarray,
    density_scores: np.ndarray,
    query_type: str = 'general',
    use_rust: Optional[bool] = None
) -> np.ndarray:
    """
    批量计算证据质量得分
    
    Args:
        semantic_scores: 语义相似度得分数组 (n,)
        density_scores: 信息密度得分数组 (n,)
        query_type: 查询类型
        use_rust: 是否使用 Rust 加速
        
    Returns:
        证据质量得分数组 (n,)
    """
    if use_rust is None:
        use_rust = RUST_AVAILABLE
    
    semantic_scores = np.ascontiguousarray(semantic_scores, dtype=np.float32)
    density_scores = np.ascontiguousarray(density_scores, dtype=np.float32)
    
    if use_rust:
        try:
            return foxhipporag_rust.batch_evidence_scores(semantic_scores, density_scores, query_type)
        except Exception as e:
            logger.warning(f"Rust batch_evidence_scores 失败，回退到 NumPy: {e}")
    
    # NumPy 回退实现
    if query_type == 'factual':
        semantic_weight, density_weight = 0.7, 0.3
    elif query_type == 'exploratory':
        semantic_weight, density_weight = 0.4, 0.6
    else:
        semantic_weight, density_weight = 0.5, 0.5
    
    multiplicative_score = semantic_scores * density_scores
    weighted_score = semantic_weight * semantic_scores + density_weight * density_scores
    
    return 0.3 * multiplicative_score + 0.7 * weighted_score


# ============================================================
# 分数融合
# ============================================================

def fuse_scores(
    scores1: np.ndarray,
    scores2: np.ndarray,
    weight1: float,
    weight2: float,
    use_rust: Optional[bool] = None
) -> np.ndarray:
    """
    融合两组分数（加权平均）
    
    Args:
        scores1: 第一组分数 (n,)
        scores2: 第二组分数 (n,)
        weight1: 第一组权重
        weight2: 第二组权重
        use_rust: 是否使用 Rust 加速
        
    Returns:
        融合后的分数 (n,)
    """
    if use_rust is None:
        use_rust = RUST_AVAILABLE
    
    scores1 = np.ascontiguousarray(scores1, dtype=np.float32)
    scores2 = np.ascontiguousarray(scores2, dtype=np.float32)
    
    if use_rust:
        try:
            return foxhipporag_rust.fuse_scores(scores1, scores2, weight1, weight2)
        except Exception as e:
            logger.warning(f"Rust fuse_scores 失败，回退到 NumPy: {e}")
    
    return weight1 * scores1 + weight2 * scores2


def multiplicative_fuse(
    scores1: np.ndarray,
    scores2: np.ndarray,
    alpha: float = 0.5,
    use_rust: Optional[bool] = None
) -> np.ndarray:
    """
    乘法融合两组分数
    
    Args:
        scores1: 第一组分数 (n,)
        scores2: 第二组分数 (n,)
        alpha: 加权融合的权重
        use_rust: 是否使用 Rust 加速
        
    Returns:
        融合后的分数 (n,)
    """
    if use_rust is None:
        use_rust = RUST_AVAILABLE
    
    scores1 = np.ascontiguousarray(scores1, dtype=np.float32)
    scores2 = np.ascontiguousarray(scores2, dtype=np.float32)
    
    if use_rust:
        try:
            return foxhipporag_rust.multiplicative_fuse(scores1, scores2, alpha)
        except Exception as e:
            logger.warning(f"Rust multiplicative_fuse 失败，回退到 NumPy: {e}")
    
    multiplicative = scores1 * scores2
    weighted = alpha * scores1 + (1.0 - alpha) * scores2
    return 0.3 * multiplicative + 0.7 * weighted
