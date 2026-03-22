"""
C扩展工具模块

该模块提供高性能的C实现，用于加速foxHippoRAG的核心计算。

模块结构：
- similarity: 余弦相似度计算
- topk: Top-K选择算法
- normalize: 向量归一化
- knn: KNN检索
- fusion: 分数融合

使用方法：
    from foxhipporag.utils.c_utils import (
        cosine_similarity,
        top_k_indices,
        knn_search,
        fuse_scores,
    )

性能优势：
- 余弦相似度：2-5x 加速
- Top-K选择：3-10x 加速
- KNN检索：2-4x 加速
"""

import logging
import os
from typing import List, Dict, Tuple, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# 尝试导入C扩展
C_EXTENSION_AVAILABLE = False
try:
    import foxhipporag_cext
    C_EXTENSION_AVAILABLE = True
    logger.info("foxhipporag_cext C扩展已启用")
except ImportError:
    logger.debug("foxhipporag_cext未安装，使用Python/NumPy回退实现")


def is_c_extension_available() -> bool:
    """检查C扩展是否可用"""
    return C_EXTENSION_AVAILABLE


def get_backend_info() -> Dict:
    """获取后端信息"""
    info = {
        "c_extension_available": C_EXTENSION_AVAILABLE,
        "backend": "c" if C_EXTENSION_AVAILABLE else "numpy",
    }
    
    if C_EXTENSION_AVAILABLE:
        try:
            info["c_version"] = foxhipporag_cext.get_version()
            info["simd_support"] = foxhipporag_cext.has_simd_support()
        except Exception as e:
            info["error"] = str(e)
    
    return info


# ============================================================
# 余弦相似度计算
# ============================================================

def cosine_similarity(
    query: np.ndarray,
    matrix: np.ndarray,
    use_c: Optional[bool] = None
) -> np.ndarray:
    """
    计算单个查询向量与矩阵中所有向量的余弦相似度
    
    Args:
        query: 查询向量 (dim,)
        matrix: 向量矩阵 (n, dim)
        use_c: 是否使用C加速（None表示自动检测）
        
    Returns:
        余弦相似度数组 (n,)
    """
    if use_c is None:
        use_c = C_EXTENSION_AVAILABLE
    
    query = np.ascontiguousarray(query, dtype=np.float32)
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    
    if use_c:
        try:
            return foxhipporag_cext.cosine_similarity(query, matrix)
        except Exception as e:
            logger.warning(f"C cosine_similarity失败，回退到NumPy: {e}")
    
    # NumPy回退实现
    query_norm = np.linalg.norm(query)
    if query_norm < 1e-10:
        return np.zeros(matrix.shape[0], dtype=np.float32)
    
    matrix_norms = np.linalg.norm(matrix, axis=1)
    matrix_norms = np.maximum(matrix_norms, 1e-10)
    
    return np.dot(matrix, query) / (matrix_norms * query_norm)


def cosine_similarity_batch(
    queries: np.ndarray,
    keys: np.ndarray,
    use_c: Optional[bool] = None
) -> np.ndarray:
    """
    批量计算查询向量矩阵与键向量矩阵的余弦相似度
    
    Args:
        queries: 查询向量矩阵 (m, dim)
        keys: 键向量矩阵 (n, dim)
        use_c: 是否使用C加速
        
    Returns:
        相似度矩阵 (m, n)
    """
    if use_c is None:
        use_c = C_EXTENSION_AVAILABLE
    
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    keys = np.ascontiguousarray(keys, dtype=np.float32)
    
    if use_c:
        try:
            return foxhipporag_cext.cosine_similarity_batch(queries, keys)
        except Exception as e:
            logger.warning(f"C cosine_similarity_batch失败，回退到NumPy: {e}")
    
    # NumPy回退实现
    query_norms = np.linalg.norm(queries, axis=1, keepdims=True)
    query_norms = np.maximum(query_norms, 1e-10)
    queries_normalized = queries / query_norms
    
    key_norms = np.linalg.norm(keys, axis=1, keepdims=True)
    key_norms = np.maximum(key_norms, 1e-10)
    keys_normalized = keys / key_norms
    
    return np.dot(queries_normalized, keys_normalized.T)


# ============================================================
# Top-K选择
# ============================================================

def top_k_indices(
    scores: np.ndarray,
    k: int,
    use_c: Optional[bool] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从一维数组中选择Top-K元素
    
    Args:
        scores: 得分数组 (n,)
        k: 要选择的数量
        use_c: 是否使用C加速
        
    Returns:
        (indices, scores): Top-K索引和得分
    """
    if use_c is None:
        use_c = C_EXTENSION_AVAILABLE
    
    scores = np.ascontiguousarray(scores, dtype=np.float32)
    k = min(k, len(scores))
    
    if use_c:
        try:
            return foxhipporag_cext.top_k_indices(scores, k)
        except Exception as e:
            logger.warning(f"C top_k_indices失败，回退到NumPy: {e}")
    
    # NumPy回退实现
    indices = np.argsort(scores)[::-1][:k]
    return indices.astype(np.int64), scores[indices]


def top_k_indices_2d(
    scores: np.ndarray,
    k: int,
    use_c: Optional[bool] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从二维矩阵的每一行选择Top-K元素
    
    Args:
        scores: 得分矩阵 (m, n)
        k: 每行要选择的数量
        use_c: 是否使用C加速
        
    Returns:
        (indices, scores): Top-K索引和得分矩阵 (m, k)
    """
    if use_c is None:
        use_c = C_EXTENSION_AVAILABLE
    
    scores = np.ascontiguousarray(scores, dtype=np.float32)
    k = min(k, scores.shape[1])
    
    if use_c:
        try:
            return foxhipporag_cext.top_k_indices_2d(scores, k)
        except Exception as e:
            logger.warning(f"C top_k_indices_2d失败，回退到NumPy: {e}")
    
    # NumPy回退实现
    indices = np.argsort(scores, axis=1)[:, ::-1][:, :k]
    top_scores = np.take_along_axis(scores, indices, axis=1)
    return indices.astype(np.int64), top_scores


# ============================================================
# 归一化
# ============================================================

def l2_normalize(
    vector: np.ndarray,
    use_c: Optional[bool] = None
) -> np.ndarray:
    """
    L2归一化向量
    
    Args:
        vector: 输入向量
        use_c: 是否使用C加速
        
    Returns:
        归一化后的向量
    """
    if use_c is None:
        use_c = C_EXTENSION_AVAILABLE
    
    vector = np.ascontiguousarray(vector, dtype=np.float32)
    
    if use_c:
        try:
            return foxhipporag_cext.l2_normalize(vector)
        except Exception as e:
            logger.warning(f"C l2_normalize失败，回退到NumPy: {e}")
    
    # NumPy回退实现
    norm = np.linalg.norm(vector)
    if norm < 1e-10:
        return np.zeros_like(vector)
    return vector / norm


def min_max_normalize(
    values: np.ndarray,
    use_c: Optional[bool] = None
) -> np.ndarray:
    """
    Min-Max归一化
    
    Args:
        values: 输入数组
        use_c: 是否使用C加速
        
    Returns:
        归一化后的数组
    """
    if use_c is None:
        use_c = C_EXTENSION_AVAILABLE
    
    values = np.ascontiguousarray(values, dtype=np.float32)
    
    if use_c:
        try:
            return foxhipporag_cext.min_max_normalize(values)
        except Exception as e:
            logger.warning(f"C min_max_normalize失败，回退到NumPy: {e}")
    
    # NumPy回退实现
    min_val = np.min(values)
    max_val = np.max(values)
    range_val = max_val - min_val
    
    if range_val < 1e-10:
        return np.ones_like(values)
    
    return (values - min_val) / range_val


def batch_l2_normalize(
    matrix: np.ndarray,
    use_c: Optional[bool] = None
) -> np.ndarray:
    """
    批量L2归一化二维矩阵（每行独立归一化）
    
    Args:
        matrix: 输入矩阵 (m, n)
        use_c: 是否使用C加速
        
    Returns:
        归一化后的矩阵 (m, n)
    """
    if use_c is None:
        use_c = C_EXTENSION_AVAILABLE
    
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    
    if use_c:
        try:
            return foxhipporag_cext.l2_normalize(matrix)
        except Exception as e:
            logger.warning(f"C batch_l2_normalize失败，回退到NumPy: {e}")
    
    # NumPy回退实现
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return matrix / norms


# ============================================================
# KNN检索
# ============================================================

def knn_search(
    query: np.ndarray,
    index_vectors: np.ndarray,
    k: int = 10,
    use_c: Optional[bool] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    KNN检索
    
    Args:
        query: 查询向量 (dim,)
        index_vectors: 索引向量矩阵 (n, dim)
        k: 返回的最近邻数量
        use_c: 是否使用C加速
        
    Returns:
        (indices, scores): 最近邻索引和相似度得分
    """
    if use_c is None:
        use_c = C_EXTENSION_AVAILABLE
    
    query = np.ascontiguousarray(query, dtype=np.float32)
    index_vectors = np.ascontiguousarray(index_vectors, dtype=np.float32)
    
    # 确保查询向量是1D
    if query.ndim == 2:
        query = query.flatten()
    
    if use_c:
        try:
            return foxhipporag_cext.knn_search(query, index_vectors, k)
        except Exception as e:
            logger.warning(f"C knn_search失败，回退到NumPy: {e}")
    
    # NumPy回退实现
    similarities = cosine_similarity(query, index_vectors, use_c=False)
    return top_k_indices(similarities, k, use_c=False)


def knn_search_batch(
    queries: np.ndarray,
    index_vectors: np.ndarray,
    k: int = 10,
    use_c: Optional[bool] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    批量KNN检索
    
    Args:
        queries: 查询向量矩阵 (m, dim)
        index_vectors: 索引向量矩阵 (n, dim)
        k: 每个查询返回的最近邻数量
        use_c: 是否使用C加速
        
    Returns:
        (indices, scores): Top-K索引和得分矩阵 (m, k)
    """
    if use_c is None:
        use_c = C_EXTENSION_AVAILABLE
    
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    index_vectors = np.ascontiguousarray(index_vectors, dtype=np.float32)
    
    if use_c:
        try:
            return foxhipporag_cext.knn_search_batch(queries, index_vectors, k)
        except Exception as e:
            logger.warning(f"C knn_search_batch失败，回退到NumPy: {e}")
    
    # NumPy回退实现
    similarity_matrix = cosine_similarity_batch(queries, index_vectors, use_c=False)
    return top_k_indices_2d(similarity_matrix, k, use_c=False)


def retrieve_knn(
    query_ids: List[str],
    key_ids: List[str],
    query_vecs: np.ndarray,
    key_vecs: np.ndarray,
    k: int = 2047,
    query_batch_size: int = 1000,
    key_batch_size: int = 10000,
    use_c: Optional[bool] = None
) -> Dict:
    """
    完整的KNN检索接口
    
    Args:
        query_ids: 查询向量ID列表
        key_ids: 键向量ID列表
        query_vecs: 查询向量矩阵 (m, dim)
        key_vecs: 键向量矩阵 (n, dim)
        k: 返回的最近邻数量
        query_batch_size: 查询批处理大小
        key_batch_size: 键批处理大小
        use_c: 是否使用C加速
        
    Returns:
        Dict: 查询ID到(key_ids, scores)的映射
    """
    if use_c is None:
        use_c = C_EXTENSION_AVAILABLE
    
    query_vecs = np.ascontiguousarray(query_vecs, dtype=np.float32)
    key_vecs = np.ascontiguousarray(key_vecs, dtype=np.float32)
    
    # 使用NumPy实现
    m = len(query_ids)
    n = len(key_ids)
    
    # 计算相似度
    similarities = cosine_similarity_batch(query_vecs, key_vecs, use_c=use_c)
    
    # 选择Top-K
    indices, scores = top_k_indices_2d(similarities, k, use_c=use_c)
    
    # 构建结果字典
    result = {}
    for i, query_id in enumerate(query_ids):
        result[query_id] = (
            [key_ids[idx] for idx in indices[i]],
            scores[i].tolist()
        )
    
    return result


# ============================================================
# 分数融合
# ============================================================

def fuse_scores(
    scores1: np.ndarray,
    scores2: np.ndarray,
    weight1: float,
    weight2: float,
    use_c: Optional[bool] = None
) -> np.ndarray:
    """
    融合两组分数（加权平均）
    
    Args:
        scores1: 第一组分数 (n,)
        scores2: 第二组分数 (n,)
        weight1: 第一组权重
        weight2: 第二组权重
        use_c: 是否使用C加速
        
    Returns:
        融合后的分数 (n,)
    """
    if use_c is None:
        use_c = C_EXTENSION_AVAILABLE
    
    scores1 = np.ascontiguousarray(scores1, dtype=np.float32)
    scores2 = np.ascontiguousarray(scores2, dtype=np.float32)
    
    if use_c:
        try:
            return foxhipporag_cext.fuse_scores(scores1, scores2, weight1, weight2)
        except Exception as e:
            logger.warning(f"C fuse_scores失败，回退到NumPy: {e}")
    
    return weight1 * scores1 + weight2 * scores2


def multiplicative_fuse(
    scores1: np.ndarray,
    scores2: np.ndarray,
    alpha: float = 0.5,
    use_c: Optional[bool] = None
) -> np.ndarray:
    """
    乘法融合两组分数
    
    Args:
        scores1: 第一组分数 (n,)
        scores2: 第二组分数 (n,)
        alpha: 加权融合的权重
        use_c: 是否使用C加速
        
    Returns:
        融合后的分数 (n,)
    """
    if use_c is None:
        use_c = C_EXTENSION_AVAILABLE
    
    scores1 = np.ascontiguousarray(scores1, dtype=np.float32)
    scores2 = np.ascontiguousarray(scores2, dtype=np.float32)
    
    if use_c:
        try:
            return foxhipporag_cext.multiplicative_fuse(scores1, scores2, alpha)
        except Exception as e:
            logger.warning(f"C multiplicative_fuse失败，回退到NumPy: {e}")
    
    multiplicative = scores1 * scores2
    weighted = alpha * scores1 + (1.0 - alpha) * scores2
    return 0.3 * multiplicative + 0.7 * weighted


# ============================================================
# 性能测试
# ============================================================

def benchmark(size: int = 10000, dim: int = 512) -> dict:
    """
    性能基准测试
    
    Args:
        size: 向量数量
        dim: 向量维度
        
    Returns:
        性能比较结果
    """
    import time
    
    np.random.seed(42)
    matrix = np.random.randn(size, dim).astype(np.float32)
    query = np.random.randn(dim).astype(np.float32)
    
    results = {}
    
    # 余弦相似度测试
    start = time.time()
    for _ in range(5):
        _ = cosine_similarity(query, matrix, use_c=False)
    numpy_time = time.time() - start
    
    if C_EXTENSION_AVAILABLE:
        start = time.time()
        for _ in range(5):
            _ = cosine_similarity(query, matrix, use_c=True)
        c_time = time.time() - start
    else:
        c_time = 0
    
    results['cosine_similarity'] = {
        'numpy_time': numpy_time,
        'c_time': c_time,
        'speedup': numpy_time / c_time if c_time > 0 else 0
    }
    
    # Top-K测试
    scores = np.random.randn(size).astype(np.float32)
    
    start = time.time()
    for _ in range(50):
        _ = top_k_indices(scores, 10, use_c=False)
    numpy_time = time.time() - start
    
    if C_EXTENSION_AVAILABLE:
        start = time.time()
        for _ in range(50):
            _ = top_k_indices(scores, 10, use_c=True)
        c_time = time.time() - start
    else:
        c_time = 0
    
    results['top_k'] = {
        'numpy_time': numpy_time,
        'c_time': c_time,
        'speedup': numpy_time / c_time if c_time > 0 else 0
    }
    
    return results


if __name__ == '__main__':
    print("C扩展工具模块")
    print("=" * 50)
    
    info = get_backend_info()
    print(f"C扩展可用: {info['c_extension_available']}")
    print(f"后端: {info['backend']}")
    
    if info['c_extension_available']:
        print(f"C版本: {info.get('c_version', 'unknown')}")
        print(f"SIMD支持: {info.get('simd_support', False)}")
        
        # 运行基准测试
        print("\n运行性能基准测试...")
        results = benchmark(size=10000, dim=512)
        
        print("\n性能比较结果:")
        for test_name, data in results.items():
            print(f"\n{test_name}:")
            print(f"  NumPy时间: {data['numpy_time']:.4f}s")
            print(f"  C时间: {data['c_time']:.4f}s")
            print(f"  加速比: {data['speedup']:.2f}x")
