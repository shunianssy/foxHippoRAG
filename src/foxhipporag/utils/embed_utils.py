"""
嵌入向量检索工具模块

提供高效的向量相似度检索功能，支持：
1. KNN（K近邻）检索
2. 批量检索优化
3. Numba JIT 加速（可选）
4. PyTorch GPU 加速

性能优化：
- 支持 Numba JIT 编译加速（CPU 并行）
- 支持 GPU 加速（通过 PyTorch）
- 批量处理优化
"""

from typing import List, Dict, Optional
import logging
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

# 尝试导入 PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.debug("PyTorch 不可用，将使用 NumPy 实现")

# 尝试导入 Numba 优化
try:
    from .numba_utils import (
        numba_cosine_similarity,
        numba_top_k_indices,
        numba_l2_normalize,
        is_numba_available
    )
    USE_NUMBA = is_numba_available()
    if USE_NUMBA:
        logger.info("使用 Numba 优化的向量检索")
except ImportError:
    USE_NUMBA = False
    logger.debug("Numba 不可用，使用标准实现")


def retrieve_knn(
    query_ids: List[str],
    key_ids: List[str],
    query_vecs,
    key_vecs,
    k: int = 2047,
    query_batch_size: int = 1000,
    key_batch_size: int = 10000,
    use_numba: Optional[bool] = None,
    use_torch: Optional[bool] = None
) -> Dict:
    """
    检索每个查询向量的 Top-K 最近邻
    
    性能优化版本，支持：
    1. Numba JIT 加速（CPU 并行计算）
    2. PyTorch GPU 加速
    3. 批量处理优化
    
    Args:
        query_ids: 查询向量 ID 列表
        key_ids: 键向量 ID 列表
        query_vecs: 查询向量数组
        key_vecs: 键向量数组
        k: 返回的最近邻数量
        query_batch_size: 查询批处理大小
        key_batch_size: 键批处理大小
        use_numba: 是否使用 Numba 加速（None 表示自动检测）
        use_torch: 是否使用 PyTorch GPU 加速（None 表示自动检测）
        
    Returns:
        Dict: 查询 ID 到 (key_ids, scores) 的映射
    """
    # 确定使用哪种实现
    if use_numba is None:
        use_numba = USE_NUMBA
    if use_torch is None:
        use_torch = TORCH_AVAILABLE and torch.cuda.is_available()
    
    # 如果有 GPU 且 PyTorch 可用，优先使用 GPU
    if use_torch:
        return _retrieve_knn_torch(
            query_ids, key_ids, query_vecs, key_vecs,
            k, query_batch_size, key_batch_size
        )
    
    # 如果 Numba 可用，使用 Numba 优化版本
    if use_numba:
        return _retrieve_knn_numba(
            query_ids, key_ids, query_vecs, key_vecs,
            k, query_batch_size, key_batch_size
        )
    
    # 否则使用纯 NumPy 实现
    return _retrieve_knn_numpy(
        query_ids, key_ids, query_vecs, key_vecs,
        k, query_batch_size, key_batch_size
    )


def _retrieve_knn_torch(
    query_ids: List[str],
    key_ids: List[str],
    query_vecs,
    key_vecs,
    k: int = 2047,
    query_batch_size: int = 1000,
    key_batch_size: int = 10000
) -> Dict:
    """
    使用 PyTorch GPU 加速的 KNN 检索
    
    这是原始实现，使用 PyTorch 进行 GPU 加速计算
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if len(key_vecs) == 0:
        return {}

    query_vecs = torch.tensor(query_vecs, dtype=torch.float32)
    query_vecs = torch.nn.functional.normalize(query_vecs, dim=1)

    key_vecs = torch.tensor(key_vecs, dtype=torch.float32)
    key_vecs = torch.nn.functional.normalize(key_vecs, dim=1)

    results = {}

    def get_batches(vecs, batch_size):
        for i in range(0, len(vecs), batch_size):
            yield vecs[i:i + batch_size], i

    for query_batch, query_batch_start_idx in tqdm(
            get_batches(vecs=query_vecs, batch_size=query_batch_size),
            total=(len(query_vecs) + query_batch_size - 1) // query_batch_size,  # Calculate total batches
            desc="KNN for Queries"
    ):
        query_batch = query_batch.clone().detach()
        query_batch = query_batch.to(device)

        batch_topk_sim_scores = []
        batch_topk_indices = []

        offset_keys = 0

        for key_batch, key_batch_start_idx in get_batches(vecs=key_vecs, batch_size=key_batch_size):
            key_batch = key_batch.to(device)
            actual_key_batch_size = key_batch.size(0)

            similarity = torch.mm(query_batch, key_batch.T)

            topk_sim_scores, topk_indices = torch.topk(similarity, min(k, actual_key_batch_size), dim=1, largest=True,
                                                       sorted=True)

            topk_indices += offset_keys

            batch_topk_sim_scores.append(topk_sim_scores)
            batch_topk_indices.append(topk_indices)

            del similarity
            key_batch = key_batch.cpu()
            torch.cuda.empty_cache()

            offset_keys += actual_key_batch_size
        # end for each kb batch

        batch_topk_sim_scores = torch.cat(batch_topk_sim_scores, dim=1)
        batch_topk_indices = torch.cat(batch_topk_indices, dim=1)

        final_topk_sim_scores, final_topk_indices = torch.topk(batch_topk_sim_scores,
                                                               min(k, batch_topk_sim_scores.size(1)), dim=1,
                                                               largest=True, sorted=True)
        final_topk_indices = final_topk_indices.cpu()
        final_topk_sim_scores = final_topk_sim_scores.cpu()

        for i in range(final_topk_indices.size(0)):
            query_relative_idx = query_batch_start_idx + i
            query_idx = query_ids[query_relative_idx]

            final_topk_indices_i = final_topk_indices[i]
            final_topk_sim_scores_i = final_topk_sim_scores[i]

            query_to_topk_key_relative_ids = batch_topk_indices[i][final_topk_indices_i]
            query_to_topk_key_ids = [key_ids[idx] for idx in query_to_topk_key_relative_ids.cpu().numpy()]
            results[query_idx] = (query_to_topk_key_ids, final_topk_sim_scores_i.numpy().tolist())

        query_batch = query_batch.cpu()
        torch.cuda.empty_cache()
    # end for each query batch

    return results


def _retrieve_knn_numba(
    query_ids: List[str],
    key_ids: List[str],
    query_vecs,
    key_vecs,
    k: int = 2047,
    query_batch_size: int = 1000,
    key_batch_size: int = 10000
) -> Dict:
    """
    使用 Numba JIT 加速的 KNN 检索
    
    使用 Numba 进行 CPU 并行计算优化
    """
    if len(key_vecs) == 0:
        return {}
    
    # 转换为 NumPy 数组
    query_vecs = np.ascontiguousarray(query_vecs, dtype=np.float32)
    key_vecs = np.ascontiguousarray(key_vecs, dtype=np.float32)
    
    # L2 归一化
    query_vecs = numba_l2_normalize(query_vecs)
    key_vecs = numba_l2_normalize(key_vecs)
    
    results = {}
    n_queries = len(query_ids)
    n_keys = len(key_ids)
    k = min(k, n_keys)
    
    # 批量处理查询
    for query_start in tqdm(range(0, n_queries, query_batch_size), desc="KNN for Queries (Numba)"):
        query_end = min(query_start + query_batch_size, n_queries)
        query_batch = query_vecs[query_start:query_end]
        
        # 批量处理键
        all_scores = []
        
        for key_start in range(0, n_keys, key_batch_size):
            key_end = min(key_start + key_batch_size, n_keys)
            key_batch = key_vecs[key_start:key_end]
            
            # 使用 Numba 优化的余弦相似度计算
            batch_scores = numba_cosine_similarity(query_batch, key_batch)
            all_scores.append(batch_scores)
        
        # 合并所有批次的得分
        combined_scores = np.hstack(all_scores)
        
        # 使用 Numba 优化的 Top-K 选择
        top_k_indices, top_k_scores = numba_top_k_indices(combined_scores, k)
        
        # 构建结果
        for i, q_idx in enumerate(range(query_start, query_end)):
            query_id = query_ids[q_idx]
            
            # 获取实际的 key IDs
            actual_indices = top_k_indices[i]
            actual_scores = top_k_scores[i]
            
            query_to_topk_key_ids = [key_ids[idx] for idx in actual_indices]
            results[query_id] = (query_to_topk_key_ids, actual_scores.tolist())
    
    return results


def _retrieve_knn_numpy(
    query_ids: List[str],
    key_ids: List[str],
    query_vecs,
    key_vecs,
    k: int = 2047,
    query_batch_size: int = 1000,
    key_batch_size: int = 10000
) -> Dict:
    """
    使用纯 NumPy 的 KNN 检索（回退实现）
    
    当 PyTorch 和 Numba 都不可用时使用
    """
    if len(key_vecs) == 0:
        return {}
    
    # 转换为 NumPy 数组
    query_vecs = np.ascontiguousarray(query_vecs, dtype=np.float32)
    key_vecs = np.ascontiguousarray(key_vecs, dtype=np.float32)
    
    # L2 归一化
    query_norms = np.linalg.norm(query_vecs, axis=1, keepdims=True)
    query_norms = np.maximum(query_norms, 1e-10)
    query_vecs = query_vecs / query_norms
    
    key_norms = np.linalg.norm(key_vecs, axis=1, keepdims=True)
    key_norms = np.maximum(key_norms, 1e-10)
    key_vecs = key_vecs / key_norms
    
    results = {}
    n_queries = len(query_ids)
    n_keys = len(key_ids)
    k = min(k, n_keys)
    
    # 批量处理查询
    for query_start in tqdm(range(0, n_queries, query_batch_size), desc="KNN for Queries (NumPy)"):
        query_end = min(query_start + query_batch_size, n_queries)
        query_batch = query_vecs[query_start:query_end]
        
        # 批量处理键
        all_scores = []
        
        for key_start in range(0, n_keys, key_batch_size):
            key_end = min(key_start + key_batch_size, n_keys)
            key_batch = key_vecs[key_start:key_end]
            
            # 计算相似度（点积，因为已归一化）
            batch_scores = np.dot(query_batch, key_batch.T)
            all_scores.append(batch_scores)
        
        # 合并所有批次的得分
        combined_scores = np.hstack(all_scores)
        
        # Top-K 选择
        top_k_indices = np.argsort(combined_scores, axis=1)[:, ::-1][:, :k]
        top_k_scores = np.take_along_axis(combined_scores, top_k_indices, axis=1)
        
        # 构建结果
        for i, q_idx in enumerate(range(query_start, query_end)):
            query_id = query_ids[q_idx]
            
            actual_indices = top_k_indices[i]
            actual_scores = top_k_scores[i]
            
            query_to_topk_key_ids = [key_ids[idx] for idx in actual_indices]
            results[query_id] = (query_to_topk_key_ids, actual_scores.tolist())
    
    return results


def batch_cosine_similarity(
    query_vecs: np.ndarray,
    key_vecs: np.ndarray,
    use_numba: Optional[bool] = None
) -> np.ndarray:
    """
    批量计算余弦相似度
    
    Args:
        query_vecs: 查询向量矩阵 (m, dim)
        key_vecs: 键向量矩阵 (n, dim)
        use_numba: 是否使用 Numba 加速
        
    Returns:
        相似度矩阵 (m, n)
    """
    if use_numba is None:
        use_numba = USE_NUMBA
    
    if use_numba:
        return numba_cosine_similarity(query_vecs, key_vecs)
    
    # NumPy 实现
    query_vecs = np.asarray(query_vecs, dtype=np.float32)
    key_vecs = np.asarray(key_vecs, dtype=np.float32)
    
    # L2 归一化
    query_norms = np.linalg.norm(query_vecs, axis=1, keepdims=True)
    query_norms = np.maximum(query_norms, 1e-10)
    query_normalized = query_vecs / query_norms
    
    key_norms = np.linalg.norm(key_vecs, axis=1, keepdims=True)
    key_norms = np.maximum(key_norms, 1e-10)
    key_normalized = key_vecs / key_norms
    
    # 点积即为余弦相似度
    return np.dot(query_normalized, key_normalized.T)