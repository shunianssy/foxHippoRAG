"""
Rust 扩展测试模块

测试 foxhipporag_rust 的功能正确性和性能。
"""

import numpy as np
import pytest
import time
from typing import Tuple

# 导入 Rust 工具模块
from foxhipporag.utils.rust_utils import (
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


class TestRustAvailability:
    """测试 Rust 扩展可用性"""
    
    def test_is_rust_available(self):
        """测试 Rust 可用性检查"""
        result = is_rust_available()
        assert isinstance(result, bool)
        print(f"Rust 可用: {result}")
    
    def test_get_backend_info(self):
        """测试后端信息获取"""
        info = get_backend_info()
        assert "rust_available" in info
        assert "backend" in info
        print(f"后端信息: {info}")


class TestCosineSimilarity:
    """测试余弦相似度计算"""
    
    def test_cosine_similarity_basic(self):
        """测试基本余弦相似度计算"""
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        matrix = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0],
        ], dtype=np.float32)
        
        result = cosine_similarity(query, matrix)
        
        assert len(result) == 4
        assert np.isclose(result[0], 1.0, atol=1e-5)  # [1,0,0] vs [1,0,0] = 1.0
        assert np.isclose(result[1], 0.0, atol=1e-5)  # [1,0,0] vs [0,1,0] = 0.0
        assert np.isclose(result[2], 0.0, atol=1e-5)  # [1,0,0] vs [0,0,1] = 0.0
        assert np.isclose(result[3], 0.7071, atol=1e-3)  # [1,0,0] vs [0.5,0.5,0] ≈ 0.707
    
    def test_cosine_similarity_batch(self):
        """测试批量余弦相似度计算"""
        queries = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        
        keys = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0],
        ], dtype=np.float32)
        
        result = cosine_similarity_batch(queries, keys)
        
        assert result.shape == (2, 3)
        assert np.isclose(result[0, 0], 1.0, atol=1e-5)
        assert np.isclose(result[1, 1], 1.0, atol=1e-5)
    
    def test_cosine_similarity_zero_vector(self):
        """测试零向量处理"""
        query = np.zeros(3, dtype=np.float32)
        matrix = np.random.randn(10, 3).astype(np.float32)
        
        result = cosine_similarity(query, matrix)
        
        assert np.allclose(result, 0.0, atol=1e-5)


class TestTopK:
    """测试 Top-K 选择"""
    
    def test_top_k_indices_basic(self):
        """测试基本 Top-K 选择"""
        scores = np.array([0.1, 0.5, 0.3, 0.9, 0.2], dtype=np.float32)
        
        indices, top_scores = top_k_indices(scores, 3)
        
        assert len(indices) == 3
        assert len(top_scores) == 3
        assert indices[0] == 3  # 最高分 0.9
        assert indices[1] == 1  # 第二高分 0.5
        assert indices[2] == 2  # 第三高分 0.3
    
    def test_top_k_indices_2d(self):
        """测试二维矩阵 Top-K 选择"""
        scores = np.array([
            [0.1, 0.5, 0.3],
            [0.9, 0.2, 0.7],
        ], dtype=np.float32)
        
        indices, top_scores = top_k_indices_2d(scores, 2)
        
        assert indices.shape == (2, 2)
        assert indices[0, 0] == 1  # 第一行最高分索引
        assert indices[1, 0] == 0  # 第二行最高分索引
    
    def test_top_k_larger_than_array(self):
        """测试 K 大于数组长度"""
        scores = np.array([0.1, 0.5, 0.3], dtype=np.float32)
        
        indices, top_scores = top_k_indices(scores, 10)
        
        assert len(indices) == 3


class TestNormalization:
    """测试归一化函数"""
    
    def test_l2_normalize(self):
        """测试 L2 归一化"""
        vector = np.array([3.0, 4.0], dtype=np.float32)
        
        result = l2_normalize(vector)
        
        expected = np.array([0.6, 0.8], dtype=np.float32)
        assert np.allclose(result, expected, atol=1e-5)
    
    def test_min_max_normalize(self):
        """测试 Min-Max 归一化"""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        
        result = min_max_normalize(values)
        
        expected = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
        assert np.allclose(result, expected, atol=1e-5)
    
    def test_batch_l2_normalize(self):
        """测试批量 L2 归一化"""
        matrix = np.array([
            [3.0, 4.0],
            [6.0, 8.0],
        ], dtype=np.float32)
        
        result = batch_l2_normalize(matrix)
        
        expected = np.array([
            [0.6, 0.8],
            [0.6, 0.8],
        ], dtype=np.float32)
        assert np.allclose(result, expected, atol=1e-5)


class TestKNN:
    """测试 KNN 检索"""
    
    def test_knn_search_basic(self):
        """测试基本 KNN 检索"""
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        index_vectors = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        
        indices, scores = knn_search(query, index_vectors, k=2)
        
        assert len(indices) == 2
        assert indices[0] == 0  # 最相似的是 [1,0,0]
        assert scores[0] > scores[1]
    
    def test_knn_search_batch(self):
        """测试批量 KNN 检索"""
        queries = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        
        index_vectors = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0],
        ], dtype=np.float32)
        
        indices, scores = knn_search_batch(queries, index_vectors, k=2)
        
        assert indices.shape == (2, 2)
    
    def test_retrieve_knn(self):
        """测试完整 KNN 检索接口"""
        query_ids = ["q1", "q2"]
        key_ids = ["k1", "k2", "k3"]
        
        query_vecs = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ], dtype=np.float32)
        
        key_vecs = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ], dtype=np.float32)
        
        results = retrieve_knn(query_ids, key_ids, query_vecs, key_vecs, k=2)
        
        assert "q1" in results
        assert "q2" in results
        assert len(results["q1"][0]) == 2


class TestTextFeatures:
    """测试文本特征计算"""
    
    def test_count_entities(self):
        """测试实体计数"""
        text = "John Smith works at Google in New York. The company was founded in 1998."
        
        count = count_entities(text)
        
        assert count >= 3  # John Smith, Google, New York, 1998
    
    def test_count_facts(self):
        """测试事实计数"""
        text = "Google is located in Mountain View. It was founded by Larry Page."
        
        count = count_facts(text)
        
        assert count >= 2  # "is", "founded"
    
    def test_compute_content_richness(self):
        """测试内容丰富度计算"""
        text = "The quick brown fox jumps over the lazy dog."
        
        richness = compute_content_richness(text)
        
        assert 0.0 <= richness <= 1.0
    
    def test_batch_count_entities(self):
        """测试批量实体计数"""
        texts = [
            "John Smith works at Google.",
            "Mary Jane lives in Paris.",
        ]
        
        counts = batch_count_entities(texts)
        
        assert len(counts) == 2
        assert all(c >= 1 for c in counts)


class TestDensity:
    """测试段落密度评估"""
    
    def test_compute_density_score(self):
        """测试密度得分计算"""
        text = "John Smith works at Google in New York. The company was founded in 1998 and has over 100,000 employees."
        
        score = compute_density_score(text)
        
        assert 0.0 <= score <= 1.0
        print(f"密度得分: {score:.4f}")
    
    def test_batch_compute_density(self):
        """测试批量密度计算"""
        texts = [
            "John Smith works at Google.",
            "This is a simple sentence.",
            "The Eiffel Tower, located in Paris, France, was designed by Gustave Eiffel and completed in 1889.",
        ]
        
        scores = batch_compute_density(texts)
        
        assert len(scores) == 3
        assert all(0.0 <= s <= 1.0 for s in scores)
    
    def test_compute_evidence_score(self):
        """测试证据质量得分"""
        semantic_score = 0.8
        density_score = 0.6
        
        result = compute_evidence_score(semantic_score, density_score, "general")
        
        assert 0.0 <= result <= 1.0
    
    def test_batch_evidence_scores(self):
        """测试批量证据质量得分"""
        semantic_scores = np.array([0.8, 0.6, 0.9], dtype=np.float32)
        density_scores = np.array([0.7, 0.5, 0.8], dtype=np.float32)
        
        results = batch_evidence_scores(semantic_scores, density_scores, "factual")
        
        assert len(results) == 3


class TestScoreFusion:
    """测试分数融合"""
    
    def test_fuse_scores(self):
        """测试加权融合"""
        scores1 = np.array([0.5, 0.6, 0.7], dtype=np.float32)
        scores2 = np.array([0.8, 0.4, 0.9], dtype=np.float32)
        
        result = fuse_scores(scores1, scores2, 0.6, 0.4)
        
        expected = 0.6 * scores1 + 0.4 * scores2
        assert np.allclose(result, expected, atol=1e-5)
    
    def test_multiplicative_fuse(self):
        """测试乘法融合"""
        scores1 = np.array([0.5, 0.6, 0.7], dtype=np.float32)
        scores2 = np.array([0.8, 0.4, 0.9], dtype=np.float32)
        
        result = multiplicative_fuse(scores1, scores2, 0.5)
        
        assert len(result) == 3
        assert all(0.0 <= r <= 1.0 for r in result)


class TestPerformance:
    """性能测试"""
    
    def test_cosine_similarity_performance(self):
        """测试余弦相似度性能"""
        size = 10000
        dim = 512
        
        query = np.random.randn(dim).astype(np.float32)
        matrix = np.random.randn(size, dim).astype(np.float32)
        
        # 预热
        _ = cosine_similarity(query, matrix[:100])
        
        # 测试
        start = time.time()
        for _ in range(5):
            _ = cosine_similarity(query, matrix)
        elapsed = time.time() - start
        
        print(f"\n余弦相似度 ({size}x{dim}): {elapsed/5:.4f}s per iteration")
    
    def test_knn_search_performance(self):
        """测试 KNN 检索性能"""
        n_queries = 100
        n_index = 10000
        dim = 128
        k = 10
        
        queries = np.random.randn(n_queries, dim).astype(np.float32)
        index_vectors = np.random.randn(n_index, dim).astype(np.float32)
        
        # 预热
        _ = knn_search(queries[0], index_vectors[:100], k=5)
        
        # 测试
        start = time.time()
        for i in range(n_queries):
            _ = knn_search(queries[i], index_vectors, k=k)
        elapsed = time.time() - start
        
        print(f"\nKNN 检索 ({n_queries} queries, {n_index} index, k={k}): {elapsed:.4f}s")
    
    def test_batch_compute_density_performance(self):
        """测试批量密度计算性能"""
        texts = [
            "John Smith works at Google in New York. The company was founded in 1998 and has over 100,000 employees worldwide. "
            "Google is a subsidiary of Alphabet Inc. and is headquartered in Mountain View, California. "
            "The company is known for its search engine, cloud computing services, and various software products."
            for _ in range(1000)
        ]
        
        # 预热
        _ = batch_compute_density(texts[:10])
        
        # 测试
        start = time.time()
        scores = batch_compute_density(texts)
        elapsed = time.time() - start
        
        print(f"\n批量密度计算 ({len(texts)} texts): {elapsed:.4f}s")
        assert len(scores) == len(texts)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
