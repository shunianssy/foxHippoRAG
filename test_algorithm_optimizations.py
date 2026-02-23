"""
算法优化模块的测试文件

测试所有优化算法的功能和性能
"""

import sys
import os
import numpy as np
import logging
import time

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.foxhipporag.utils.algorithm_optimizations import (
    OptimizedVectorRetrieval,
    OptimizedGraphPPR,
    AdaptiveRetrievalFusion,
    QueryOptimizer,
    BatchOptimizer,
    AlgorithmOptimizationPipeline
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_vector_retrieval():
    """测试优化的向量检索算法"""
    logger.info("=" * 60)
    logger.info("测试优化的向量检索算法")
    logger.info("=" * 60)
    
    np.random.seed(42)
    n_samples = 1000
    n_dim = 128
    
    key_ids = [f"key_{i}" for i in range(n_samples)]
    key_vectors = np.random.randn(n_samples, n_dim).astype(np.float32)
    key_vectors = key_vectors / np.linalg.norm(key_vectors, axis=1, keepdims=True)
    
    n_queries = 10
    query_ids = [f"query_{i}" for i in range(n_queries)]
    query_vectors = np.random.randn(n_queries, n_dim).astype(np.float32)
    query_vectors = query_vectors / np.linalg.norm(query_vectors, axis=1, keepdims=True)
    
    # 测试暴力检索
    logger.info("\n1. 测试暴力检索:")
    flat_retriever = OptimizedVectorRetrieval(index_type='flat')
    flat_retriever.build_index(key_ids, key_vectors)
    
    start_time = time.time()
    flat_results = flat_retriever.retrieve(query_ids, query_vectors, k=10)
    flat_time = time.time() - start_time
    logger.info(f"   暴力检索耗时: {flat_time:.4f}s")
    
    # 测试分层检索
    logger.info("\n2. 测试分层检索:")
    hierarchical_retriever = OptimizedVectorRetrieval(
        index_type='hierarchical',
        coarse_k=100,
        fine_k=10
    )
    hierarchical_retriever.build_index(key_ids, key_vectors)
    
    start_time = time.time()
    hierarchical_results = hierarchical_retriever.retrieve(query_ids, query_vectors, k=10)
    hierarchical_time = time.time() - start_time
    logger.info(f"   分层检索耗时: {hierarchical_time:.4f}s")
    logger.info(f"   加速比: {flat_time / hierarchical_time:.2f}x")
    
    # 测试LSH检索
    logger.info("\n3. 测试LSH近似检索:")
    lsh_retriever = OptimizedVectorRetrieval(
        index_type='lsh',
        use_lsh=True,
        num_hash_tables=8,
        num_hash_bits=16
    )
    lsh_retriever.build_index(key_ids, key_vectors)
    
    start_time = time.time()
    lsh_results = lsh_retriever.retrieve(query_ids, query_vectors, k=10)
    lsh_time = time.time() - start_time
    logger.info(f"   LSH检索耗时: {lsh_time:.4f}s")
    logger.info(f"   加速比: {flat_time / lsh_time:.2f}x")
    
    # 验证结果一致性
    logger.info("\n4. 验证结果一致性:")
    match_count = 0
    for q_id in query_ids:
        flat_docs = flat_results[q_id][0]
        hierarchical_docs = hierarchical_results[q_id][0]
        intersection = set(flat_docs) & set(hierarchical_docs)
        match_count += len(intersection)
    
    avg_match = match_count / (n_queries * 10)
    logger.info(f"   分层检索与暴力检索的平均重叠率: {avg_match:.2%}")
    
    return True


def test_graph_ppr():
    """测试优化的图PPR算法"""
    logger.info("\n" + "=" * 60)
    logger.info("测试优化的图PPR算法")
    logger.info("=" * 60)
    
    np.random.seed(42)
    n_nodes = 500
    n_edges = 2000
    
    node_names = [f"node_{i}" for i in range(n_nodes)]
    edges = []
    
    for _ in range(n_edges):
        s_idx = np.random.randint(0, n_nodes)
        t_idx = np.random.randint(0, n_nodes)
        if s_idx != t_idx:
            weight = np.random.uniform(0.01, 1.0)
            edges.append((node_names[s_idx], node_names[t_idx], weight))
    
    # 测试近似PPR
    logger.info("\n1. 测试近似PPR:")
    approx_ppr = OptimizedGraphPPR(
        use_approximate=True,
        max_iterations=50,
        tolerance=1e-4,
        prune_threshold=0.01
    )
    approx_ppr.build_graph(node_names, edges)
    
    n_queries = 5
    reset_probs_list = []
    for _ in range(n_queries):
        reset_prob = np.zeros(n_nodes)
        seed_idx = np.random.randint(0, n_nodes)
        reset_prob[seed_idx] = 1.0
        reset_probs_list.append(reset_prob)
    
    start_time = time.time()
    approx_results = approx_ppr.compute_ppr_batch(reset_probs_list, damping=0.5)
    approx_time = time.time() - start_time
    logger.info(f"   近似PPR耗时: {approx_time:.4f}s (平均每个查询: {approx_time/n_queries:.4f}s)")
    
    # 测试精确PPR（仅用于小图）
    logger.info("\n2. 测试图剪枝效果:")
    logger.info(f"   原始边数: {n_edges}")
    logger.info(f"   剪枝后边数: {len(edges) - len(approx_ppr.pruned_edges)}")
    logger.info(f"   剪枝比例: {len(approx_ppr.pruned_edges) / n_edges:.2%}")
    
    # 验证PPR结果
    logger.info("\n3. 验证PPR结果:")
    for i, (sorted_indices, scores) in enumerate(approx_results):
        top_score = scores[0]
        logger.info(f"   查询 {i+1}: 最高得分 = {top_score:.6f}, 前3个节点 = {[node_names[idx] for idx in sorted_indices[:3]]}")
    
    return True


def test_retrieval_fusion():
    """测试自适应检索融合"""
    logger.info("\n" + "=" * 60)
    logger.info("测试自适应检索融合")
    logger.info("=" * 60)
    
    fusion = AdaptiveRetrievalFusion(
        min_confidence=0.5,
        max_confidence=0.95,
        enable_early_termination=True,
        fusion_method='weighted'
    )
    
    # 测试1: DPR结果置信度较高
    logger.info("\n1. 测试高置信度DPR结果:")
    dpr_ids_1 = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    dpr_scores_1 = [0.95, 0.85, 0.75, 0.65, 0.55]
    graph_ids_1 = ["doc3", "doc1", "doc6", "doc7", "doc8"]
    graph_scores_1 = [0.80, 0.70, 0.60, 0.50, 0.40]
    
    fused_ids_1, fused_scores_1 = fusion.fuse(
        (dpr_ids_1, dpr_scores_1),
        (graph_ids_1, graph_scores_1),
        query="test query 1"
    )
    logger.info(f"   融合后的前3个文档: {fused_ids_1[:3]}")
    logger.info(f"   DPR置信度: {fusion._calculate_confidence(dpr_scores_1):.3f}")
    logger.info(f"   图检索置信度: {fusion._calculate_confidence(graph_scores_1):.3f}")
    
    # 测试2: 不同融合方法
    logger.info("\n2. 测试不同融合方法:")
    for method in ['weighted', 'reciprocal_rank', 'borda']:
        fusion_method = AdaptiveRetrievalFusion(fusion_method=method)
        fused_ids, _ = fusion_method.fuse(
            (dpr_ids_1, dpr_scores_1),
            (graph_ids_1, graph_scores_1)
        )
        logger.info(f"   {method} 方法: 前3个 = {fused_ids[:3]}")
    
    # 测试3: 早期终止
    logger.info("\n3. 测试早期终止:")
    high_conf_dpr_ids = ["docA", "docB", "docC"]
    high_conf_dpr_scores = [0.99, 0.50, 0.40]
    low_conf_graph_ids = ["docX", "docY", "docZ"]
    low_conf_graph_scores = [0.60, 0.55, 0.50]
    
    fused_ids_early, _ = fusion.fuse(
        (high_conf_dpr_ids, high_conf_dpr_scores),
        (low_conf_graph_ids, low_conf_graph_scores)
    )
    logger.info(f"   早期终止后选择: {fused_ids_early[:3]}")
    logger.info(f"   (直接选择DPR结果，因为置信度很高)")
    
    return True


def test_query_optimizer():
    """测试查询优化器"""
    logger.info("\n" + "=" * 60)
    logger.info("测试查询优化器")
    logger.info("=" * 60)
    
    optimizer = QueryOptimizer(
        enable_rewriting=True,
        enable_expansion=True,
        enable_decomposition=False
    )
    
    test_queries = [
        "What is the capital of France?",
        "How to make delicious chocolate chip cookies and cake?",
        "Who wrote the famous novel 1984 and where was the author born?"
    ]
    
    for i, query in enumerate(test_queries):
        logger.info(f"\n查询 {i+1}: {query}")
        result = optimizer.optimize(query)
        logger.info(f"   优化后的查询: {result['optimized_queries']}")
        logger.info(f"   来自缓存: {result['from_cache']}")
        
        # 测试缓存
        cached_result = optimizer.optimize(query)
        logger.info(f"   第二次查询来自缓存: {cached_result['from_cache']}")
    
    return True


def test_batch_optimizer():
    """测试批量优化器"""
    logger.info("\n" + "=" * 60)
    logger.info("测试批量优化器")
    logger.info("=" * 60)
    
    batch_optimizer = BatchOptimizer(
        initial_batch_size=10,
        max_batch_size=100,
        min_batch_size=5,
        max_workers=4
    )
    
    def process_fn(items):
        """模拟处理函数"""
        time.sleep(0.01)
        return [x * 2 for x in items]
    
    items = list(range(100))
    
    logger.info("\n1. 测试批量处理:")
    start_time = time.time()
    results = batch_optimizer.process_batch(items, process_fn, desc="测试处理")
    batch_time = time.time() - start_time
    
    logger.info(f"   批量处理耗时: {batch_time:.4f}s")
    logger.info(f"   结果验证: {all(results[i] == items[i] * 2 for i in range(10))}")
    logger.info(f"   当前批处理大小: {batch_optimizer.batch_size}")
    
    return True


def test_pipeline():
    """测试完整的优化管道"""
    logger.info("\n" + "=" * 60)
    logger.info("测试完整的算法优化管道")
    logger.info("=" * 60)
    
    config = {
        'index_type': 'hierarchical',
        'coarse_k': 50,
        'fine_k': 10,
        'use_approximate_ppr': True,
        'ppr_max_iterations': 30,
        'fusion_method': 'weighted',
        'enable_query_rewriting': True,
        'initial_batch_size': 32
    }
    
    pipeline = AlgorithmOptimizationPipeline(config)
    
    logger.info("\n管道组件:")
    for name in ['vector_retrieval', 'graph_ppr', 'retrieval_fusion', 'query_optimizer', 'batch_optimizer']:
        optimizer = pipeline.get_optimizer(name)
        logger.info(f"   ✓ {name}")
    
    return True


def run_all_tests():
    """运行所有测试"""
    logger.info("=" * 60)
    logger.info("开始算法优化模块测试")
    logger.info("=" * 60)
    
    tests = [
        ("向量检索优化", test_vector_retrieval),
        ("图PPR优化", test_graph_ppr),
        ("检索融合优化", test_retrieval_fusion),
        ("查询优化", test_query_optimizer),
        ("批量优化", test_batch_optimizer),
        ("完整管道", test_pipeline)
    ]
    
    results = []
    for test_name, test_fn in tests:
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"运行测试: {test_name}")
            logger.info(f"{'='*60}")
            success = test_fn()
            results.append((test_name, success))
            logger.info(f"✓ 测试 {test_name} 通过")
        except Exception as e:
            logger.error(f"✗ 测试 {test_name} 失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            results.append((test_name, False))
    
    logger.info("\n" + "=" * 60)
    logger.info("测试总结")
    logger.info("=" * 60)
    
    for test_name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        logger.info(f"   {test_name}: {status}")
    
    success_count = sum(1 for _, s in results if s)
    logger.info(f"\n总计: {success_count}/{len(results)} 测试通过")
    
    return success_count == len(results)


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
