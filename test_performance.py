"""
性能优化测试文件

测试内容：
1. 基本功能测试
2. 缓存效果测试
3. 并行处理测试
4. 批量操作测试
5. 性能对比测试
"""

import os
import sys
import time
import json
import logging
import multiprocessing
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

if __name__ == '__main__':
    multiprocessing.freeze_support()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('PerformanceTest')


def test_basic_functionality():
    """测试基本功能"""
    logger.info("=" * 50)
    logger.info("测试1: 基本功能测试")
    logger.info("=" * 50)
    
    try:
        from src.foxhipporag import foxHippoRAG
        from src.foxhipporag.utils.config_utils import BaseConfig
        
        config = BaseConfig()
        config.save_dir = "outputs/test_performance"
        config.llm_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        config.llm_base_url = os.getenv("OPENAI_API_BASE_URL")
        config.embedding_model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        config.embedding_base_url = os.getenv("EMBEDDING_BASE_URL")
        config.embedding_api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
        config.openie_mode = "online"
        config.force_index_from_scratch = False
        
        hippo = foxHippoRAG(global_config=config)
        
        test_docs = [
            "张三是一名软件工程师，他在北京工作。",
            "李四是一名医生，他在上海工作。",
            "王五是一名教师，他在广州工作。",
        ]
        
        logger.info("索引测试文档...")
        start_time = time.time()
        hippo.index(docs=test_docs)
        index_time = time.time() - start_time
        logger.info(f"索引完成，耗时: {index_time:.2f}s")
        
        test_queries = [
            "谁在北京工作？",
            "谁是医生？",
            "王五的职业是什么？"
        ]
        
        logger.info("测试检索...")
        for query in test_queries:
            start_time = time.time()
            results = hippo.retrieve(queries=[query], num_to_retrieve=2)
            retrieve_time = time.time() - start_time
            logger.info(f"查询: {query}")
            logger.info(f"  耗时: {retrieve_time:.2f}s")
            if results:
                logger.info(f"  结果: {results[0].docs[:1]}")
        
        logger.info("基本功能测试通过！")
        return True
        
    except Exception as e:
        logger.error(f"基本功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cache_effectiveness():
    """测试缓存效果"""
    logger.info("\n" + "=" * 50)
    logger.info("测试2: 缓存效果测试")
    logger.info("=" * 50)
    
    try:
        from src.foxhipporag.utils.performance_utils import (
            PPRCache, EmbeddingCache, LRUCache
        )
        import numpy as np
        
        temp_dir = "outputs/test_performance/cache_test"
        os.makedirs(temp_dir, exist_ok=True)
        
        logger.info("测试LRU缓存...")
        lru_cache = LRUCache(max_size=100)
        
        for i in range(150):
            lru_cache.put(f"key_{i}", f"value_{i}")
        
        stats = lru_cache.get_stats()
        logger.info(f"LRU缓存统计: 大小={stats['size']}, 最大={stats['max_size']}")
        assert stats['size'] == 100, "LRU缓存大小应该为100"
        
        for i in range(50):
            lru_cache.get(f"key_{i}")
        
        stats = lru_cache.get_stats()
        logger.info(f"LRU缓存命中统计: 命中={stats['hits']}, 未命中={stats['misses']}, 命中率={stats['hit_rate']:.2%}")
        
        logger.info("测试PPR缓存...")
        ppr_cache = PPRCache(temp_dir, max_memory_size=50)
        
        test_prob = np.random.rand(100)
        test_prob = test_prob / test_prob.sum()
        test_doc_ids = np.arange(100)
        test_doc_scores = np.random.rand(100)
        
        ppr_cache.put(test_prob, 0.5, "test_graph", test_doc_ids, test_doc_scores)
        
        cached = ppr_cache.get(test_prob, 0.5, "test_graph")
        assert cached is not None, "PPR缓存应该命中"
        
        stats = ppr_cache.get_stats()
        logger.info(f"PPR缓存统计: 命中={stats['hits']}, 未命中={stats['misses']}, 命中率={stats['hit_rate']:.2%}")
        
        logger.info("测试嵌入缓存...")
        emb_cache = EmbeddingCache(temp_dir, max_memory_size=100)
        
        test_texts = [f"test text {i}" for i in range(10)]
        test_embeddings = np.random.rand(10, 128)
        
        emb_cache.put_batch(test_texts, "test_model", test_embeddings)
        
        cached_emb = emb_cache.get("test text 0", "test_model")
        assert cached_emb is not None, "嵌入缓存应该命中"
        
        stats = emb_cache.get_stats()
        logger.info(f"嵌入缓存统计: 命中={stats['hits']}, 未命中={stats['misses']}, 命中率={stats['hit_rate']:.2%}")
        
        logger.info("缓存效果测试通过！")
        return True
        
    except Exception as e:
        logger.error(f"缓存效果测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_parallel_processing():
    """测试并行处理"""
    logger.info("\n" + "=" * 50)
    logger.info("测试3: 并行处理测试")
    logger.info("=" * 50)
    
    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time
        
        def cpu_intensive_task(n: int) -> int:
            """模拟CPU密集型任务"""
            result = 0
            for i in range(n):
                result += i * i
            return result
        
        tasks = [100000] * 8
        
        logger.info("串行处理...")
        start_time = time.time()
        serial_results = [cpu_intensive_task(n) for n in tasks]
        serial_time = time.time() - start_time
        logger.info(f"串行处理耗时: {serial_time:.2f}s")
        
        logger.info("并行处理（4线程）...")
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(cpu_intensive_task, n) for n in tasks]
            parallel_results = [f.result() for f in as_completed(futures)]
        parallel_time = time.time() - start_time
        logger.info(f"并行处理耗时: {parallel_time:.2f}s")
        
        speedup = serial_time / parallel_time if parallel_time > 0 else 0
        logger.info(f"加速比: {speedup:.2f}x")
        
        assert len(serial_results) == len(parallel_results), "结果数量应该一致"
        
        logger.info("并行处理测试通过！")
        return True
        
    except Exception as e:
        logger.error(f"并行处理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_operations():
    """测试批量操作"""
    logger.info("\n" + "=" * 50)
    logger.info("测试4: 批量操作测试")
    logger.info("=" * 50)
    
    try:
        from src.foxhipporag.utils.performance_utils import BatchProcessor
        
        def process_item(item: int) -> int:
            """处理单个项目"""
            time.sleep(0.01)
            return item * 2
        
        processor = BatchProcessor(batch_size=10, max_workers=4)
        items = list(range(50))
        
        logger.info("批量处理50个项目...")
        start_time = time.time()
        results = processor.process_in_batches(items, process_item, "批量处理")
        batch_time = time.time() - start_time
        logger.info(f"批量处理耗时: {batch_time:.2f}s")
        
        assert len(results) == 50, "结果数量应该为50"
        assert all(r == i * 2 for i, r in enumerate(results)), "结果应该正确"
        
        logger.info("批量操作测试通过！")
        return True
        
    except Exception as e:
        logger.error(f"批量操作测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_comparison():
    """性能对比测试"""
    logger.info("\n" + "=" * 50)
    logger.info("测试5: 性能对比测试")
    logger.info("=" * 50)
    
    try:
        from src.foxhipporag.utils.performance_utils import (
            PerformanceMonitor, global_performance_monitor
        )
        
        monitor = PerformanceMonitor()
        
        for i in range(10):
            with monitor.record_time("test_operation"):
                time.sleep(0.01 * (i % 3 + 1))
        
        stats = monitor.get_stats("test_operation")
        logger.info(f"性能统计:")
        logger.info(f"  调用次数: {stats['count']}")
        logger.info(f"  平均耗时: {stats['mean']:.4f}s")
        logger.info(f"  标准差: {stats['std']:.4f}s")
        logger.info(f"  最小耗时: {stats['min']:.4f}s")
        logger.info(f"  最大耗时: {stats['max']:.4f}s")
        logger.info(f"  总耗时: {stats['total']:.4f}s")
        
        assert stats['count'] == 10, "调用次数应该为10"
        
        logger.info("性能对比测试通过！")
        return True
        
    except Exception as e:
        logger.error(f"性能对比测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_management():
    """测试内存管理"""
    logger.info("\n" + "=" * 50)
    logger.info("测试6: 内存管理测试")
    logger.info("=" * 50)
    
    try:
        from src.foxhipporag.utils.performance_utils import MemoryManager
        
        memory_manager = MemoryManager(warning_threshold=0.8, critical_threshold=0.9)
        
        cleanup_called = []
        def cleanup_callback():
            cleanup_called.append(True)
            logger.info("清理回调被调用")
        
        memory_manager.add_cleanup_callback(cleanup_callback)
        
        usage, status = memory_manager.check_memory()
        logger.info(f"内存使用: {usage:.1%}, 状态: {status}")
        
        memory_info = memory_manager.get_memory_info()
        logger.info(f"内存信息: {memory_info}")
        
        logger.info("内存管理测试通过！")
        return True
        
    except Exception as e:
        logger.error(f"内存管理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """运行所有测试"""
    logger.info("开始性能优化测试...")
    logger.info("=" * 60)
    
    results = {}
    
    tests = [
        ("基本功能测试", test_basic_functionality),
        ("缓存效果测试", test_cache_effectiveness),
        ("并行处理测试", test_parallel_processing),
        ("批量操作测试", test_batch_operations),
        ("性能对比测试", test_performance_comparison),
        ("内存管理测试", test_memory_management),
    ]
    
    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            logger.error(f"测试 {name} 异常: {e}")
            results[name] = False
    
    logger.info("\n" + "=" * 60)
    logger.info("测试结果汇总:")
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
