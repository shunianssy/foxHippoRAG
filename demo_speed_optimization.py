"""
优化版foxHippoRAG使用示例

展示如何使用优化版本实现高性能的知识库操作
"""

import os
import time
import logging
import multiprocessing
from dotenv import load_dotenv

load_dotenv()

if __name__ == '__main__':
    multiprocessing.freeze_support()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('SpeedDemo')


def demo_fast_mode():
    """演示快速模式"""
    from src.foxhipporag.foxHippoRAG_optimized import OptimizedfoxHippoRAG
    from src.foxhipporag.utils.config_utils import BaseConfig
    
    logger.info("=" * 60)
    logger.info("优化版foxHippoRAG - 快速模式演示")
    logger.info("=" * 60)
    
    # 配置
    config = BaseConfig()
    config.save_dir = "outputs/demo_fast"
    config.llm_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    config.llm_base_url = os.getenv("OPENAI_API_BASE_URL")
    config.embedding_model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    config.embedding_base_url = os.getenv("EMBEDDING_BASE_URL")
    config.embedding_api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
    config.openie_mode = "online"
    config.force_index_from_scratch = True
    
    # 启用快速模式
    config.use_fast_index = True      # 快速索引：跳过OpenIE，直接使用DPR
    config.use_fast_retrieve = True   # 快速检索：直接使用DPR，跳过图搜索
    config.preload_embeddings = True  # 预加载嵌入向量到内存
    config.enable_query_cache = True  # 启用查询缓存
    
    # 初始化
    logger.info("初始化优化版foxHippoRAG...")
    init_start = time.time()
    hippo = OptimizedfoxHippoRAG(global_config=config)
    init_time = time.time() - init_start
    logger.info(f"初始化完成，耗时: {init_time:.2f}s")
    
    # 准备测试文档
    test_docs = [
        "张三是一名软件工程师，他在北京工作，喜欢打篮球。",
        "李四是一名医生，他在上海工作，喜欢阅读。",
        "王五是一名教师，他在广州工作，喜欢旅游。",
        "赵六是一名律师，他在深圳工作，喜欢游泳。",
        "钱七是一名设计师，他在杭州工作，喜欢画画。",
        "孙八是一名会计，他在成都工作，喜欢唱歌。",
        "周九是一名销售，他在武汉工作，喜欢跑步。",
        "吴十是一名产品经理，他在南京工作，喜欢摄影。",
        "郑十一是一名架构师，他在西安工作，喜欢下棋。",
        "王十二是一名测试工程师，他在重庆工作，喜欢游戏。",
    ]
    
    # 索引文档
    logger.info(f"\n索引 {len(test_docs)} 条文档...")
    index_start = time.time()
    hippo.index(docs=test_docs)
    index_time = time.time() - index_start
    logger.info(f"索引完成，耗时: {index_time:.2f}s")
    logger.info(f"平均每条文档: {index_time/len(test_docs)*1000:.2f}ms")
    
    # 检索测试
    test_queries = [
        "谁在北京工作？",
        "谁是医生？",
        "谁喜欢旅游？",
        "谁是产品经理？",
        "谁在杭州工作？",
    ]
    
    logger.info(f"\n检索测试（{len(test_queries)} 个查询）...")
    retrieve_start = time.time()
    results = hippo.retrieve(queries=test_queries, num_to_retrieve=3)
    retrieve_time = time.time() - retrieve_start
    logger.info(f"检索完成，耗时: {retrieve_time:.2f}s")
    logger.info(f"平均每个查询: {retrieve_time/len(test_queries)*1000:.2f}ms")
    
    # 显示结果
    logger.info("\n检索结果:")
    for i, result in enumerate(results):
        logger.info(f"\n查询 {i+1}: {result.question}")
        if result.docs:
            for j, doc in enumerate(result.docs[:2]):
                logger.info(f"  结果 {j+1}: {doc}")
                if result.doc_scores and j < len(result.doc_scores):
                    logger.info(f"  得分: {result.doc_scores[j]:.4f}")
    
    # 性能总结
    logger.info("\n" + "=" * 60)
    logger.info("性能总结:")
    logger.info(f"  初始化: {init_time:.2f}s")
    logger.info(f"  索引 {len(test_docs)} 条文档: {index_time:.2f}s")
    logger.info(f"  检索 {len(test_queries)} 个查询: {retrieve_time:.2f}s")
    logger.info(f"  总耗时: {init_time + index_time + retrieve_time:.2f}s")
    logger.info("=" * 60)


def demo_standard_mode():
    """演示标准模式（带OpenIE）"""
    from src.foxhipporag.foxHippoRAG_optimized import OptimizedfoxHippoRAG
    from src.foxhipporag.utils.config_utils import BaseConfig
    
    logger.info("\n" + "=" * 60)
    logger.info("优化版foxHippoRAG - 标准模式演示（带知识图谱）")
    logger.info("=" * 60)
    
    # 配置
    config = BaseConfig()
    config.save_dir = "outputs/demo_standard"
    config.llm_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    config.llm_base_url = os.getenv("OPENAI_API_BASE_URL")
    config.embedding_model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    config.embedding_base_url = os.getenv("EMBEDDING_BASE_URL")
    config.embedding_api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
    config.openie_mode = "online"
    config.force_index_from_scratch = True
    
    # 不启用快速模式（使用完整的知识图谱）
    config.use_fast_index = False
    config.use_fast_retrieve = False
    config.preload_embeddings = True
    
    # 初始化
    logger.info("初始化优化版foxHippoRAG（标准模式）...")
    init_start = time.time()
    hippo = OptimizedfoxHippoRAG(global_config=config)
    init_time = time.time() - init_start
    logger.info(f"初始化完成，耗时: {init_time:.2f}s")
    
    # 准备测试文档
    test_docs = [
        "张三是一名软件工程师，他在北京工作，喜欢打篮球。",
        "李四是一名医生，他在上海工作，喜欢阅读。",
        "王五是一名教师，他在广州工作，喜欢旅游。",
    ]
    
    # 索引文档
    logger.info(f"\n索引 {len(test_docs)} 条文档（带OpenIE）...")
    index_start = time.time()
    hippo.index(docs=test_docs)
    index_time = time.time() - index_start
    logger.info(f"索引完成，耗时: {index_time:.2f}s")
    
    # 检索测试
    test_queries = ["谁在北京工作？", "谁是医生？"]
    
    logger.info(f"\n检索测试（{len(test_queries)} 个查询）...")
    retrieve_start = time.time()
    results = hippo.retrieve(queries=test_queries, num_to_retrieve=3)
    retrieve_time = time.time() - retrieve_start
    logger.info(f"检索完成，耗时: {retrieve_time:.2f}s")
    
    # 显示结果
    logger.info("\n检索结果:")
    for i, result in enumerate(results):
        logger.info(f"\n查询 {i+1}: {result.question}")
        if result.docs:
            for j, doc in enumerate(result.docs[:2]):
                logger.info(f"  结果 {j+1}: {doc}")
    
    # 显示图信息
    try:
        graph = hippo.graph
        logger.info(f"\n知识图谱信息:")
        logger.info(f"  节点数量: {graph.vcount()}")
        logger.info(f"  边数量: {graph.ecount()}")
    except Exception as e:
        logger.warning(f"无法获取图信息: {e}")


def demo_comparison():
    """演示快速模式和标准模式的对比"""
    logger.info("\n" + "=" * 60)
    logger.info("快速模式 vs 标准模式 性能对比")
    logger.info("=" * 60)
    
    from src.foxhipporag.foxHippoRAG_optimized import OptimizedfoxHippoRAG
    from src.foxhipporag.utils.config_utils import BaseConfig
    
    # 准备测试文档
    test_docs = [
        f"文档{i}: 用户{i % 10}喜欢{i % 5}号产品，他在城市{i % 8}工作。"
        for i in range(50)
    ]
    
    test_queries = [
        "谁喜欢3号产品？",
        "谁在城市5工作？",
        "用户2的偏好是什么？",
    ]
    
    # 测试快速模式
    logger.info("\n--- 快速模式 ---")
    config_fast = BaseConfig()
    config_fast.save_dir = "outputs/demo_comparison_fast"
    config_fast.llm_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    config_fast.llm_base_url = os.getenv("OPENAI_API_BASE_URL")
    config_fast.embedding_model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    config_fast.embedding_base_url = os.getenv("EMBEDDING_BASE_URL")
    config_fast.embedding_api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
    config_fast.openie_mode = "online"
    config_fast.force_index_from_scratch = True
    config_fast.use_fast_index = True
    config_fast.use_fast_retrieve = True
    
    hippo_fast = OptimizedfoxHippoRAG(global_config=config_fast)
    
    start = time.time()
    hippo_fast.index(docs=test_docs)
    fast_index_time = time.time() - start
    
    start = time.time()
    hippo_fast.retrieve(queries=test_queries, num_to_retrieve=5)
    fast_retrieve_time = time.time() - start
    
    logger.info(f"索引耗时: {fast_index_time:.2f}s")
    logger.info(f"检索耗时: {fast_retrieve_time:.2f}s")
    
    # 测试标准模式（使用较少文档）
    logger.info("\n--- 标准模式 ---")
    config_std = BaseConfig()
    config_std.save_dir = "outputs/demo_comparison_std"
    config_std.llm_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    config_std.llm_base_url = os.getenv("OPENAI_API_BASE_URL")
    config_std.embedding_model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    config_std.embedding_base_url = os.getenv("EMBEDDING_BASE_URL")
    config_std.embedding_api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
    config_std.openie_mode = "online"
    config_std.force_index_from_scratch = True
    config_std.use_fast_index = False
    config_std.use_fast_retrieve = False
    
    hippo_std = OptimizedfoxHippoRAG(global_config=config_std)
    
    # 标准模式使用较少文档
    small_docs = test_docs[:10]
    
    start = time.time()
    hippo_std.index(docs=small_docs)
    std_index_time = time.time() - start
    
    start = time.time()
    hippo_std.retrieve(queries=test_queries, num_to_retrieve=5)
    std_retrieve_time = time.time() - start
    
    logger.info(f"索引耗时（10条文档）: {std_index_time:.2f}s")
    logger.info(f"检索耗时: {std_retrieve_time:.2f}s")
    
    # 对比总结
    logger.info("\n" + "=" * 60)
    logger.info("性能对比总结:")
    logger.info(f"  快速模式索引 {len(test_docs)} 条文档: {fast_index_time:.2f}s")
    logger.info(f"  标准模式索引 {len(small_docs)} 条文档: {std_index_time:.2f}s")
    logger.info(f"  快速模式检索: {fast_retrieve_time:.2f}s")
    logger.info(f"  标准模式检索: {std_retrieve_time:.2f}s")
    
    # 估算加速比
    if std_index_time > 0 and fast_index_time > 0:
        estimated_speedup = (std_index_time / len(small_docs)) / (fast_index_time / len(test_docs))
        logger.info(f"  估算索引加速比: {estimated_speedup:.1f}x")
    
    logger.info("=" * 60)


if __name__ == "__main__":
    # 运行演示
    demo_fast_mode()
    # demo_standard_mode()  # 取消注释以运行标准模式演示
    # demo_comparison()     # 取消注释以运行对比演示
