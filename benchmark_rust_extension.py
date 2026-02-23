"""
Rust 扩展性能基准测试

对比 Rust 和 Python 实现的性能差异。
"""

import numpy as np
import time
from typing import Callable, Dict, List, Tuple
import sys

# 导入 Rust 工具模块
from foxhipporag.utils.rust_utils import (
    is_rust_available,
    get_backend_info,
    cosine_similarity as rust_cosine_similarity,
    cosine_similarity_batch as rust_cosine_similarity_batch,
    top_k_indices as rust_top_k_indices,
    top_k_indices_2d as rust_top_k_indices_2d,
    l2_normalize as rust_l2_normalize,
    min_max_normalize as rust_min_max_normalize,
    batch_l2_normalize as rust_batch_l2_normalize,
    knn_search as rust_knn_search,
    knn_search_batch as rust_knn_search_batch,
    batch_compute_density as rust_batch_compute_density,
    batch_count_entities as rust_batch_count_entities,
)

# 导入 Numba 工具模块
try:
    from foxhipporag.utils.numba_utils import (
        numba_cosine_similarity,
        numba_top_k_indices,
        numba_l2_normalize,
        numba_min_max_normalize,
        is_numba_available,
    )
    NUMBA_AVAILABLE = is_numba_available()
except ImportError:
    NUMBA_AVAILABLE = False


def benchmark_function(
    func: Callable,
    args: tuple,
    iterations: int = 5,
    warmup: int = 1,
    name: str = ""
) -> Dict:
    """
    基准测试函数
    
    Args:
        func: 要测试的函数
        args: 函数参数
        iterations: 迭代次数
        warmup: 预热次数
        name: 测试名称
        
    Returns:
        测试结果字典
    """
    # 预热
    for _ in range(warmup):
        try:
            func(*args)
        except Exception as e:
            return {"error": str(e), "name": name}
    
    # 正式测试
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        try:
            result = func(*args)
        except Exception as e:
            return {"error": str(e), "name": name}
        end = time.perf_counter()
        times.append(end - start)
    
    return {
        "name": name,
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
        "iterations": iterations,
    }


def numpy_cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """NumPy 余弦相似度实现"""
    query_norm = np.linalg.norm(query)
    if query_norm < 1e-10:
        return np.zeros(matrix.shape[0], dtype=np.float32)
    
    matrix_norms = np.linalg.norm(matrix, axis=1)
    matrix_norms = np.maximum(matrix_norms, 1e-10)
    
    return np.dot(matrix, query) / (matrix_norms * query_norm)


def numpy_cosine_similarity_batch(queries: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """NumPy 批量余弦相似度实现"""
    query_norms = np.linalg.norm(queries, axis=1, keepdims=True)
    query_norms = np.maximum(query_norms, 1e-10)
    queries_normalized = queries / query_norms
    
    key_norms = np.linalg.norm(keys, axis=1, keepdims=True)
    key_norms = np.maximum(key_norms, 1e-10)
    keys_normalized = keys / key_norms
    
    return np.dot(queries_normalized, keys_normalized.T)


def numpy_top_k_indices(scores: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """NumPy Top-K 实现"""
    indices = np.argsort(scores)[::-1][:k]
    return indices.astype(np.int64), scores[indices]


def numpy_top_k_indices_2d(scores: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """NumPy 二维 Top-K 实现"""
    indices = np.argsort(scores, axis=1)[:, ::-1][:, :k]
    top_scores = np.take_along_axis(scores, indices, axis=1)
    return indices.astype(np.int64), top_scores


def numpy_l2_normalize(vector: np.ndarray) -> np.ndarray:
    """NumPy L2 归一化"""
    norm = np.linalg.norm(vector)
    if norm < 1e-10:
        return np.zeros_like(vector)
    return vector / norm


def numpy_min_max_normalize(values: np.ndarray) -> np.ndarray:
    """NumPy Min-Max 归一化"""
    min_val = np.min(values)
    max_val = np.max(values)
    range_val = max_val - min_val
    
    if range_val < 1e-10:
        return np.ones_like(values)
    
    return (values - min_val) / range_val


def numpy_batch_l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """NumPy 批量 L2 归一化"""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return matrix / norms


def run_cosine_similarity_benchmark() -> List[Dict]:
    """运行余弦相似度基准测试"""
    results = []
    
    sizes = [
        (1000, 128),
        (5000, 256),
        (10000, 512),
    ]
    
    for size, dim in sizes:
        print(f"\n{'='*60}")
        print(f"余弦相似度测试: {size} 向量, 维度 {dim}")
        print(f"{'='*60}")
        
        query = np.random.randn(dim).astype(np.float32)
        matrix = np.random.randn(size, dim).astype(np.float32)
        
        # NumPy 基准
        numpy_result = benchmark_function(
            numpy_cosine_similarity,
            (query, matrix),
            iterations=10,
            name=f"NumPy ({size}x{dim})"
        )
        results.append(numpy_result)
        print(f"NumPy: {numpy_result['mean']*1000:.2f}ms ± {numpy_result['std']*1000:.2f}ms")
        
        # Numba 测试
        if NUMBA_AVAILABLE:
            numba_result = benchmark_function(
                numba_cosine_similarity,
                (query, matrix),
                iterations=10,
                warmup=2,
                name=f"Numba ({size}x{dim})"
            )
            results.append(numba_result)
            if "error" not in numba_result:
                speedup = numpy_result['mean'] / numba_result['mean']
                print(f"Numba: {numba_result['mean']*1000:.2f}ms (加速 {speedup:.2f}x)")
        
        # Rust 测试
        if is_rust_available():
            rust_result = benchmark_function(
                rust_cosine_similarity,
                (query, matrix),
                iterations=10,
                warmup=2,
                name=f"Rust ({size}x{dim})"
            )
            results.append(rust_result)
            if "error" not in rust_result:
                speedup = numpy_result['mean'] / rust_result['mean']
                print(f"Rust:  {rust_result['mean']*1000:.2f}ms (加速 {speedup:.2f}x)")
    
    return results


def run_topk_benchmark() -> List[Dict]:
    """运行 Top-K 基准测试"""
    results = []
    
    sizes = [10000, 50000, 100000]
    k = 100
    
    for size in sizes:
        print(f"\n{'='*60}")
        print(f"Top-K 测试: {size} 元素, k={k}")
        print(f"{'='*60}")
        
        scores = np.random.randn(size).astype(np.float32)
        
        # NumPy 基准
        numpy_result = benchmark_function(
            numpy_top_k_indices,
            (scores, k),
            iterations=20,
            name=f"NumPy TopK ({size})"
        )
        results.append(numpy_result)
        print(f"NumPy: {numpy_result['mean']*1000:.2f}ms ± {numpy_result['std']*1000:.2f}ms")
        
        # Numba 测试
        if NUMBA_AVAILABLE:
            numba_result = benchmark_function(
                numba_top_k_indices,
                (scores, k),
                iterations=20,
                warmup=2,
                name=f"Numba TopK ({size})"
            )
            results.append(numba_result)
            if "error" not in numba_result:
                speedup = numpy_result['mean'] / numba_result['mean']
                print(f"Numba: {numba_result['mean']*1000:.2f}ms (加速 {speedup:.2f}x)")
        
        # Rust 测试
        if is_rust_available():
            rust_result = benchmark_function(
                rust_top_k_indices,
                (scores, k),
                iterations=20,
                warmup=2,
                name=f"Rust TopK ({size})"
            )
            results.append(rust_result)
            if "error" not in rust_result:
                speedup = numpy_result['mean'] / rust_result['mean']
                print(f"Rust:  {rust_result['mean']*1000:.2f}ms (加速 {speedup:.2f}x)")
    
    return results


def run_knn_benchmark() -> List[Dict]:
    """运行 KNN 基准测试"""
    results = []
    
    configs = [
        (100, 5000, 128, 10),
        (500, 10000, 256, 20),
    ]
    
    for n_queries, n_index, dim, k in configs:
        print(f"\n{'='*60}")
        print(f"KNN 测试: {n_queries} 查询, {n_index} 索引, 维度 {dim}, k={k}")
        print(f"{'='*60}")
        
        queries = np.random.randn(n_queries, dim).astype(np.float32)
        index_vectors = np.random.randn(n_index, dim).astype(np.float32)
        
        # NumPy 实现
        def numpy_knn():
            all_indices = []
            all_scores = []
            for i in range(n_queries):
                sims = numpy_cosine_similarity(queries[i], index_vectors)
                idx, sc = numpy_top_k_indices(sims, k)
                all_indices.append(idx)
                all_scores.append(sc)
            return all_indices, all_scores
        
        numpy_result = benchmark_function(
            numpy_knn,
            (),
            iterations=3,
            name=f"NumPy KNN ({n_queries}x{n_index})"
        )
        results.append(numpy_result)
        print(f"NumPy: {numpy_result['mean']:.2f}s ± {numpy_result['std']:.2f}s")
        
        # Rust 测试
        if is_rust_available():
            rust_result = benchmark_function(
                rust_knn_search_batch,
                (queries, index_vectors, k),
                iterations=3,
                warmup=1,
                name=f"Rust KNN ({n_queries}x{n_index})"
            )
            results.append(rust_result)
            if "error" not in rust_result:
                speedup = numpy_result['mean'] / rust_result['mean']
                print(f"Rust:  {rust_result['mean']:.2f}s (加速 {speedup:.2f}x)")
    
    return results


def run_density_benchmark() -> List[Dict]:
    """运行密度计算基准测试"""
    results = []
    
    # 生成测试文本
    sample_text = (
        "John Smith works at Google in New York. The company was founded in 1998 "
        "and has over 100,000 employees worldwide. Google is a subsidiary of Alphabet Inc. "
        "and is headquartered in Mountain View, California. The company is known for its "
        "search engine, cloud computing services, and various software products."
    )
    
    text_counts = [100, 500, 1000]
    
    for count in text_counts:
        print(f"\n{'='*60}")
        print(f"密度计算测试: {count} 文本")
        print(f"{'='*60}")
        
        texts = [sample_text] * count
        
        # Python 实现
        def python_density():
            import re
            results = []
            for text in texts:
                entity_count = len(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text))
                results.append(entity_count)
            return results
        
        python_result = benchmark_function(
            python_density,
            (),
            iterations=5,
            name=f"Python Density ({count})"
        )
        results.append(python_result)
        print(f"Python: {python_result['mean']*1000:.2f}ms ± {python_result['std']*1000:.2f}ms")
        
        # Rust 测试
        if is_rust_available():
            rust_result = benchmark_function(
                rust_batch_compute_density,
                (texts,),
                iterations=5,
                warmup=1,
                name=f"Rust Density ({count})"
            )
            results.append(rust_result)
            if "error" not in rust_result:
                speedup = python_result['mean'] / rust_result['mean']
                print(f"Rust:   {rust_result['mean']*1000:.2f}ms (加速 {speedup:.2f}x)")
    
    return results


def main():
    """运行所有基准测试"""
    print("=" * 60)
    print("foxHippoRAG Rust 扩展性能基准测试")
    print("=" * 60)
    
    # 显示环境信息
    info = get_backend_info()
    print(f"\n后端信息:")
    print(f"  Rust 可用: {info['rust_available']}")
    print(f"  Numba 可用: {NUMBA_AVAILABLE}")
    if info['rust_available']:
        print(f"  Rust 版本: {info.get('rust_version', 'unknown')}")
        print(f"  并行线程: {info.get('num_threads', 'unknown')}")
    
    all_results = []
    
    # 运行各项测试
    all_results.extend(run_cosine_similarity_benchmark())
    all_results.extend(run_topk_benchmark())
    all_results.extend(run_knn_benchmark())
    all_results.extend(run_density_benchmark())
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("性能测试汇总")
    print("=" * 60)
    
    # 计算总体加速比
    rust_speedups = []
    numba_speedups = []
    
    for i, result in enumerate(all_results):
        if "error" in result:
            continue
        if "Rust" in result['name']:
            # 找到对应的 NumPy 基准
            for j in range(i - 1, -1, -1):
                if "NumPy" in all_results[j]['name'] or "Python" in all_results[j]['name']:
                    if "error" not in all_results[j]:
                        speedup = all_results[j]['mean'] / result['mean']
                        rust_speedups.append(speedup)
                        break
        elif "Numba" in result['name']:
            for j in range(i - 1, -1, -1):
                if "NumPy" in all_results[j]['name']:
                    if "error" not in all_results[j]:
                        speedup = all_results[j]['mean'] / result['mean']
                        numba_speedups.append(speedup)
                        break
    
    if rust_speedups:
        print(f"\nRust 平均加速比: {np.mean(rust_speedups):.2f}x")
        print(f"Rust 最大加速比: {np.max(rust_speedups):.2f}x")
    
    if numba_speedups:
        print(f"\nNumba 平均加速比: {np.mean(numba_speedups):.2f}x")
        print(f"Numba 最大加速比: {np.max(numba_speedups):.2f}x")


if __name__ == "__main__":
    main()
