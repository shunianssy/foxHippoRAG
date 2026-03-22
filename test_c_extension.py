"""
C扩展测试模块

测试C扩展实现的正确性和性能
"""

import unittest
import numpy as np
import time

try:
    from foxhipporag.utils.c_utils import (
        cosine_similarity,
        cosine_similarity_batch,
        top_k_indices,
        top_k_indices_2d,
        l2_normalize,
        min_max_normalize,
        batch_l2_normalize,
        knn_search,
        knn_search_batch,
        fuse_scores,
        multiplicative_fuse,
        is_c_extension_available,
        get_backend_info,
    )
    C_UTILS_AVAILABLE = True
except ImportError:
    C_UTILS_AVAILABLE = False


@unittest.skipIf(not C_UTILS_AVAILABLE, "c_utils模块不可用")
class TestCosineSimilarity(unittest.TestCase):
    """测试余弦相似度计算"""
    
    def setUp(self):
        """设置测试数据"""
        np.random.seed(42)
        self.dim = 128
        self.n = 1000
        self.query = np.random.randn(self.dim).astype(np.float32)
        self.matrix = np.random.randn(self.n, self.dim).astype(np.float32)
    
    def test_cosine_similarity_single(self):
        """测试单个查询向量的余弦相似度"""
        # 使用C扩展计算
        result_c = cosine_similarity(self.query, self.matrix, use_c=True)
        
        # 使用NumPy计算（作为参考）
        query_norm = np.linalg.norm(self.query)
        matrix_norms = np.linalg.norm(self.matrix, axis=1)
        expected = np.dot(self.matrix, self.query) / (matrix_norms * query_norm)
        
        # 验证结果
        self.assertEqual(len(result_c), self.n)
        np.testing.assert_allclose(result_c, expected, rtol=1e-5, atol=1e-5)
    
    def test_cosine_similarity_batch(self):
        """测试批量余弦相似度计算"""
        m = 100
        queries = np.random.randn(m, self.dim).astype(np.float32)
        
        # 使用C扩展计算
        result_c = cosine_similarity_batch(queries, self.matrix, use_c=True)
        
        # 验证形状
        self.assertEqual(result_c.shape, (m, self.n))
        
        # 验证结果正确性（抽样检查）
        for i in range(5):
            single_result = cosine_similarity(queries[i], self.matrix, use_c=True)
            np.testing.assert_allclose(result_c[i], single_result, rtol=1e-5)
    
    def test_zero_vector(self):
        """测试零向量的处理"""
        zero_query = np.zeros(self.dim, dtype=np.float32)
        result = cosine_similarity(zero_query, self.matrix, use_c=True)
        
        # 零向量应该返回全零
        np.testing.assert_allclose(result, np.zeros(self.n), atol=1e-7)


@unittest.skipIf(not C_UTILS_AVAILABLE, "c_utils模块不可用")
class TestTopK(unittest.TestCase):
    """测试Top-K选择算法"""
    
    def setUp(self):
        """设置测试数据"""
        np.random.seed(42)
        self.n = 1000
        self.scores = np.random.randn(self.n).astype(np.float32)
    
    def test_top_k_indices_1d(self):
        """测试一维Top-K选择"""
        k = 10
        indices, top_scores = top_k_indices(self.scores, k, use_c=True)
        
        # 验证结果
        self.assertEqual(len(indices), k)
        self.assertEqual(len(top_scores), k)
        
        # 验证是否是Top-K
        expected_indices = np.argsort(self.scores)[::-1][:k]
        expected_scores = self.scores[expected_indices]
        
        np.testing.assert_array_equal(indices, expected_indices)
        np.testing.assert_allclose(top_scores, expected_scores, rtol=1e-5)
    
    def test_top_k_indices_2d(self):
        """测试二维Top-K选择"""
        m = 100
        n = 500
        k = 20
        scores_2d = np.random.randn(m, n).astype(np.float32)
        
        indices, top_scores = top_k_indices_2d(scores_2d, k, use_c=True)
        
        # 验证形状
        self.assertEqual(indices.shape, (m, k))
        self.assertEqual(top_scores.shape, (m, k))
        
        # 验证结果正确性（抽样检查）
        for i in range(5):
            row_indices, row_scores = top_k_indices(scores_2d[i], k, use_c=True)
            np.testing.assert_array_equal(indices[i], row_indices)
            np.testing.assert_allclose(top_scores[i], row_scores, rtol=1e-5)
    
    def test_top_k_larger_than_n(self):
        """测试k大于数组长度的情况"""
        k = self.n + 100
        indices, top_scores = top_k_indices(self.scores, k, use_c=True)
        
        # 应该返回所有元素
        self.assertEqual(len(indices), self.n)


@unittest.skipIf(not C_UTILS_AVAILABLE, "c_utils模块不可用")
class TestNormalization(unittest.TestCase):
    """测试归一化函数"""
    
    def setUp(self):
        """设置测试数据"""
        np.random.seed(42)
        self.dim = 128
        self.n = 100
    
    def test_l2_normalize(self):
        """测试L2归一化"""
        vector = np.random.randn(self.dim).astype(np.float32)
        normalized = l2_normalize(vector, use_c=True)
        
        # 验证范数为1
        norm = np.linalg.norm(normalized)
        self.assertAlmostEqual(norm, 1.0, places=5)
        
        # 验证方向不变
        original_norm = np.linalg.norm(vector)
        expected = vector / original_norm
        np.testing.assert_allclose(normalized, expected, rtol=1e-5)
    
    def test_batch_l2_normalize(self):
        """测试批量L2归一化"""
        matrix = np.random.randn(self.n, self.dim).astype(np.float32)
        normalized = batch_l2_normalize(matrix, use_c=True)
        
        # 验证每行的范数为1
        norms = np.linalg.norm(normalized, axis=1)
        np.testing.assert_allclose(norms, np.ones(self.n), rtol=1e-5)
    
    def test_min_max_normalize(self):
        """测试Min-Max归一化"""
        values = np.random.randn(self.n).astype(np.float32)
        normalized = min_max_normalize(values, use_c=True)
        
        # 验证范围在[0, 1]
        self.assertTrue(np.all(normalized >= 0))
        self.assertTrue(np.all(normalized <= 1))
        
        # 验证最小值为0，最大值为1
        self.assertAlmostEqual(np.min(normalized), 0.0, places=5)
        self.assertAlmostEqual(np.max(normalized), 1.0, places=5)


@unittest.skipIf(not C_UTILS_AVAILABLE, "c_utils模块不可用")
class TestKNN(unittest.TestCase):
    """测试KNN检索"""
    
    def setUp(self):
        """设置测试数据"""
        np.random.seed(42)
        self.dim = 64
        self.n = 500
        self.index_vectors = np.random.randn(self.n, self.dim).astype(np.float32)
    
    def test_knn_search(self):
        """测试单个KNN检索"""
        query = np.random.randn(self.dim).astype(np.float32)
        k = 10
        
        indices, scores = knn_search(query, self.index_vectors, k, use_c=True)
        
        # 验证结果
        self.assertEqual(len(indices), k)
        self.assertEqual(len(scores), k)
        
        # 验证分数是降序排列
        for i in range(len(scores) - 1):
            self.assertGreaterEqual(scores[i], scores[i + 1])
    
    def test_knn_search_batch(self):
        """测试批量KNN检索"""
        m = 50
        queries = np.random.randn(m, self.dim).astype(np.float32)
        k = 20
        
        indices, scores = knn_search_batch(queries, self.index_vectors, k, use_c=True)
        
        # 验证形状
        self.assertEqual(indices.shape, (m, k))
        self.assertEqual(scores.shape, (m, k))
        
        # 验证每行的分数是降序排列
        for i in range(m):
            for j in range(k - 1):
                self.assertGreaterEqual(scores[i, j], scores[i, j + 1])


@unittest.skipIf(not C_UTILS_AVAILABLE, "c_utils模块不可用")
class TestScoreFusion(unittest.TestCase):
    """测试分数融合"""
    
    def setUp(self):
        """设置测试数据"""
        np.random.seed(42)
        self.n = 100
        self.scores1 = np.random.rand(self.n).astype(np.float32)
        self.scores2 = np.random.rand(self.n).astype(np.float32)
    
    def test_fuse_scores(self):
        """测试加权融合"""
        weight1, weight2 = 0.6, 0.4
        result = fuse_scores(self.scores1, self.scores2, weight1, weight2, use_c=True)
        
        # 验证结果
        expected = weight1 * self.scores1 + weight2 * self.scores2
        np.testing.assert_allclose(result, expected, rtol=1e-5)
    
    def test_multiplicative_fuse(self):
        """测试乘法融合"""
        alpha = 0.5
        result = multiplicative_fuse(self.scores1, self.scores2, alpha, use_c=True)
        
        # 验证结果
        multiplicative = self.scores1 * self.scores2
        weighted = alpha * self.scores1 + (1.0 - alpha) * self.scores2
        expected = 0.3 * multiplicative + 0.7 * weighted
        np.testing.assert_allclose(result, expected, rtol=1e-5)


@unittest.skipIf(not C_UTILS_AVAILABLE, "c_utils模块不可用")
class TestPerformance(unittest.TestCase):
    """性能测试"""
    
    def test_performance_comparison(self):
        """比较C扩展和NumPy的性能"""
        np.random.seed(42)
        size = 5000
        dim = 256
        
        matrix = np.random.randn(size, dim).astype(np.float32)
        query = np.random.randn(dim).astype(np.float32)
        
        # NumPy性能
        start = time.time()
        for _ in range(10):
            _ = cosine_similarity(query, matrix, use_c=False)
        numpy_time = time.time() - start
        
        # C扩展性能
        if is_c_extension_available():
            start = time.time()
            for _ in range(10):
                _ = cosine_similarity(query, matrix, use_c=True)
            c_time = time.time() - start
            
            # C扩展应该更快（或至少相当）
            print(f"\n性能比较: NumPy={numpy_time:.4f}s, C={c_time:.4f}s, 加速比={numpy_time/c_time:.2f}x")
        else:
            print("\nC扩展不可用，跳过性能比较")


class TestBackendInfo(unittest.TestCase):
    """测试后端信息"""
    
    def test_backend_info(self):
        """测试获取后端信息"""
        info = get_backend_info()
        
        self.assertIn('c_extension_available', info)
        self.assertIn('backend', info)
        
        print(f"\n后端信息: {info}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
