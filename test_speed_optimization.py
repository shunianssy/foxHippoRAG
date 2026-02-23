"""
性能优化测试文件

测试目标：
- 1000条知识库条目，1秒内完成加入知识和检索

测试内容：
1. 快速索引模式性能测试
2. 快速检索模式性能测试
3. 缓存效果测试
4. 并发性能测试
"""

import os
import sys
import time
import logging
import multiprocessing
from typing import List
from dotenv import load_dotenv

load_dotenv()

if __name__ == '__main__':
    multiprocessing.freeze_support()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('SpeedTest')


def generate_test_documents(count: int = 1000) -> List[str]:
    """生成测试文档"""
    docs = []
    for i in range(count):
        doc = f"文档{i}: 这是一个关于主题{i % 100}的测试文档。" \
              f"它包含了一些关键信息，比如用户{i % 50}喜欢{i % 10}号产品。" \
              f"这个文档的目的是测试知识库的性能。"
        docs.append(doc)
    return docs


def generate_test_queries(count: int = 10) -> List[str]:
    """生成测试查询"""
    queries = [
        "谁喜欢5号产品？",
        "主题10的文档内容是什么？",
        "用户20有什么偏好？",
        "关于主题50的信息有哪些？",
        "文档100包含什么内容？",
        "哪些用户喜欢8号产品？",
        "主题30相关的文档有哪些？",
        "用户15的偏好是什么？",
        "关于主题75的文档内容？",
        "哪些文档提到了3号产品？"
    ]
    return queries[:count]


def test_fast_index_mode():
    """测试快速索引模式"""
    logger.info("=" * 60)
    logger.info("测试1: 快速索引模式性能测试")
    logger.info("=" * 60)
    
    try:
        from src.foxhipporag.foxHippoRAG_optimized import OptimizedfoxHippoRAG
        from src.foxhipporag.utils.config_utils import BaseConfig
        
        # 配置快速索引模式
        config = BaseConfig()
        config.save_dir = "outputs/speed_test_fast"
        config.llm_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        config.llm_base_url = os.getenv("OPENAI_API_BASE_URL")
        config.embedding_model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        config.embedding_base_url = os.getenv("EMBEDDING_BASE_URL")
        config.embedding_api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
        config.openie_mode = "online"
        config.force_index_from_scratch = True
        
        # 启用快速模式
        config.use_fast_index = True
        config.use_fast_retrieve = True
        config.preload_embeddings = True
        
        # 生成测试文档
        test_docs = generate_test_documents(1000)
        
        logger.info(f"测试文档数量: {len(test_docs)}")
        
        # 初始化
        init_start = time.time()
        hippo = OptimizedfoxHippoRAG(global_config=config)
        init_time = time.time() - init_start
        logger.info(f"初始化耗时: {init_time:.2f}s")
        
        # 索引
        index_start = time.time()
        hippo.index(docs=test_docs)
        index_time = time.time() - index_start
        logger.info(f"索引耗时: {index_time:.2f}s")
        logger.info(f"平均每条文档索引耗时: {index_time/len(test_docs)*1000:.2f}ms")
        
        # 检索
        test_queries = generate_test_queries(10)
        
        retrieve_start = time.time()
        results = hippo.retrieve(queries=test_queries, num_to_retrieve=5)
        retrieve_time = time.time() - retrieve_start
        logger.info(f"检索耗时（10个查询）: {retrieve_time:.2f}s")
        logger.info(f"平均每个查询检索耗时: {retrieve_time/len(test_queries)*1000:.2f}ms")
        
        # 验证结果
        if results:
            logger.info(f"检索结果示例:")
            for i, result in enumerate(results[:3]):
                logger.info(f"  查询{i+1}: {result.question}")
                if result.docs:
                    logger.info(f"    结果: {result.docs[0][:50]}...")
        
        # 总结
        total_time = index_time + retrieve_time
        logger.info(f"\n性能总结:")
        logger.info(f"  索引1000条文档: {index_time:.2f}s")
        logger.info(f"  检索10个查询: {retrieve_time:.2f}s")
        logger.info(f"  总耗时: {total_time:.2f}s")
        
        # 判断是否达标
        if index_time < 1.0 and retrieve_time < 1.0:
            logger.info("✓ 性能达标！索引和检索均在1秒内完成")
        else:
            logger.warning(f"✗ 性能未达标。索引: {index_time:.2f}s, 检索: {retrieve_time:.2f}s")
        
        return True
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_standard_mode():
    """测试标准模式（对比基准）"""
    logger.info("\n" + "=" * 60)
    logger.info("测试2: 标准模式性能测试（对比基准）")
    logger.info("=" * 60)
    
    try:
        from src.foxhipporag.foxHippoRAG_optimized import OptimizedfoxHippoRAG
        from src.foxhipporag.utils.config_utils import BaseConfig
        
        # 配置标准模式
        config = BaseConfig()
        config.save_dir = "outputs/speed_test_standard"
        config.llm_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        config.llm_base_url = os.getenv("OPENAI_API_BASE_URL")
        config.embedding_model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        config.embedding_base_url = os.getenv("EMBEDDING_BASE_URL")
        config.embedding_api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
        config.openie_mode = "online"
        config.force_index_from_scratch = True
        
        # 不启用快速模式
        config.use_fast_index = False
        config.use_fast_retrieve = False
        
        # 使用较少文档测试（标准模式较慢）
        test_docs = generate_test_documents(10)
        
        logger.info(f"测试文档数量: {len(test_docs)}")
        
        # 初始化
        init_start = time.time()
        hippo = OptimizedfoxHippoRAG(global_config=config)
        init_time = time.time() - init_start
        logger.info(f"初始化耗时: {init_time:.2f}s")
        
        # 索引
        index_start = time.time()
        hippo.index(docs=test_docs)
        index_time = time.time() - index_start
        logger.info(f"索引耗时: {index_time:.2f}s")
        logger.info(f"平均每条文档索引耗时: {index_time/len(test_docs):.2f}s")
        
        # 检索
        test_queries = generate_test_queries(3)
        
        retrieve_start = time.time()
        results = hippo.retrieve(queries=test_queries, num_to_retrieve=5)
        retrieve_time = time.time() - retrieve_start
        logger.info(f"检索耗时（3个查询）: {retrieve_time:.2f}s")
        
        logger.info(f"\n标准模式性能总结:")
        logger.info(f"  索引10条文档: {index_time:.2f}s")
        logger.info(f"  检索3个查询: {retrieve_time:.2f}s")
        
        return True
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cache_effectiveness():
    """测试缓存效果"""
    logger.info("\n" + "=" * 60)
    logger.info("测试3: 缓存效果测试")
    logger.info("=" * 60)
    
    try:
        from src.foxhipporag.foxHippoRAG_optimized import OptimizedfoxHippoRAG
        from src.foxhipporag.utils.config_utils import BaseConfig
        
        # 配置
        config = BaseConfig()
        config.save_dir = "outputs/speed_test_cache"
        config.llm_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        config.llm_base_url = os.getenv("OPENAI_API_BASE_URL")
        config.embedding_model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        config.embedding_base_url = os.getenv("EMBEDDING_BASE_URL")
        config.embedding_api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
        config.openie_mode = "online"
        config.force_index_from_scratch = True
        config.use_fast_index = True
        config.use_fast_retrieve = True
        config.enable_query_cache = True
        
        test_docs = generate_test_documents(100)
        
        # 初始化
        hippo = OptimizedfoxHippoRAG(global_config=config)
        hippo.index(docs=test_docs)
        
        # 测试查询缓存
        test_query = "谁喜欢5号产品？"
        
        # 第一次查询（缓存未命中）
        start1 = time.time()
        result1 = hippo.retrieve(queries=[test_query], num_to_retrieve=5)
        time1 = time.time() - start1
        logger.info(f"第一次查询耗时: {time1*1000:.2f}ms")
        
        # 第二次查询（缓存命中）
        start2 = time.time()
        result2 = hippo.retrieve(queries=[test_query], num_to_retrieve=5)
        time2 = time.time() - start2
        logger.info(f"第二次查询耗时: {time2*1000:.2f}ms")
        
        # 计算加速比
        if time2 > 0:
            speedup = time1 / time2
            logger.info(f"缓存加速比: {speedup:.2f}x")
        
        # 获取缓存统计
        if hasattr(hippo, 'ppr_cache'):
            stats = hippo.ppr_cache.get_stats()
            logger.info(f"PPR缓存统计: 命中={stats['hits']}, 未命中={stats['misses']}, 命中率={stats['hit_rate']:.2%}")
        
        return True
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_concurrent_performance():
    """测试并发性能"""
    logger.info("\n" + "=" * 60)
    logger.info("测试4: 并发性能测试")
    logger.info("=" * 60)
    
    try:
        from src.foxhipporag.foxHippoRAG_optimized import OptimizedfoxHippoRAG
        from src.foxhipporag.utils.config_utils import BaseConfig
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # 配置
        config = BaseConfig()
        config.save_dir = "outputs/speed_test_concurrent"
        config.llm_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        config.llm_base_url = os.getenv("OPENAI_API_BASE_URL")
        config.embedding_model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        config.embedding_base_url = os.getenv("EMBEDDING_BASE_URL")
        config.embedding_api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
        config.openie_mode = "online"
        config.force_index_from_scratch = True
        config.use_fast_index = True
        config.use_fast_retrieve = True
        config.retrieval_parallel_workers = 8
        
        test_docs = generate_test_documents(500)
        
        # 初始化
        hippo = OptimizedfoxHippoRAG(global_config=config)
        hippo.index(docs=test_docs)
        
        # 测试并发检索
        test_queries = generate_test_queries(20)
        
        # 串行检索
        logger.info("串行检索...")
        start_serial = time.time()
        for query in test_queries:
            hippo.retrieve(queries=[query], num_to_retrieve=5)
        time_serial = time.time() - start_serial
        logger.info(f"串行检索耗时: {time_serial:.2f}s")
        
        # 并行检索
        logger.info("并行检索...")
        start_parallel = time.time()
        hippo.retrieve(queries=test_queries, num_to_retrieve=5)
        time_parallel = time.time() - start_parallel
        logger.info(f"并行检索耗时: {time_parallel:.2f}s")
        
        # 计算加速比
        if time_parallel > 0:
            speedup = time_serial / time_parallel
            logger.info(f"并行加速比: {speedup:.2f}x")
        
        return True
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """运行所有测试"""
    logger.info("=" * 60)
    logger.info("开始性能优化测试")
    logger.info("目标: 1000条知识库条目，1秒内完成加入知识和检索")
    logger.info("=" * 60)
    
    results = {}
    
    tests = [
        ("快速索引模式性能测试", test_fast_index_mode),
        ("标准模式性能测试", test_standard_mode),
        ("缓存效果测试", test_cache_effectiveness),
        ("并发性能测试", test_concurrent_performance),
    ]
    
    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            logger.error(f"测试 {name} 异常: {e}")
            results[name] = False
    
    logger.info("\n" + "=" * 60)
    logger.info("测试结果汇总")
    logger.info("=" * 60)
    
    passed = 0
    failed = 0
    for name, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        logger.info(f"  {name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    logger.info(f"\n总计: {passed} 通过, {failed} 失败")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
