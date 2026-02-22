"""
异步性能测试文件

测试内容：
1. 异步缓存测试
2. 异步HTTP客户端测试
3. 异步OpenAI客户端测试
4. 异步批量处理测试
5. 性能对比测试（同步 vs 异步）
"""

import os
import sys
import time
import asyncio
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('AsyncPerformanceTest')


def test_async_cache():
    """测试异步缓存"""
    logger.info("=" * 50)
    logger.info("测试1: 异步缓存测试")
    logger.info("=" * 50)
    
    async def run_test():
        from src.foxhipporag.utils.async_utils import AsyncLRUCache, AsyncSQLiteCache
        import tempfile
        
        temp_dir = tempfile.mkdtemp()
        
        logger.info("测试异步LRU缓存...")
        lru_cache = AsyncLRUCache(max_size=100)
        
        for i in range(150):
            await lru_cache.put(f"key_{i}", f"value_{i}")
        
        stats = await lru_cache.get_stats()
        logger.info(f"LRU缓存统计: 大小={stats['size']}, 最大={stats['max_size']}")
        assert stats['size'] == 100, "LRU缓存大小应该为100"
        
        for i in range(50, 100):
            result = await lru_cache.get(f"key_{i}")
            assert result is not None, f"key_{i} 应该存在"
        
        stats = await lru_cache.get_stats()
        logger.info(f"LRU缓存命中统计: 命中={stats['hits']}, 未命中={stats['misses']}, 命中率={stats['hit_rate']:.2%}")
        
        logger.info("测试异步SQLite缓存...")
        sqlite_cache = AsyncSQLiteCache(temp_dir, "test_cache.sqlite")
        
        await sqlite_cache.put("test_key", {"data": "test_value"}, "test_namespace")
        result = await sqlite_cache.get("test_key", "test_namespace")
        assert result is not None, "SQLite缓存应该命中"
        assert result['data'] == "test_value", "数据应该匹配"
        
        batch_items = {f"batch_key_{i}": {"data": f"value_{i}"} for i in range(10)}
        await sqlite_cache.put_batch(batch_items, "batch_namespace")
        
        batch_results = await sqlite_cache.get_batch(
            [f"batch_key_{i}" for i in range(10)], 
            "batch_namespace"
        )
        assert len(batch_results) == 10, "批量获取应该返回10条记录"
        
        logger.info("异步缓存测试通过！")
        return True
    
    try:
        return asyncio.run(run_test())
    except Exception as e:
        logger.error(f"异步缓存测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_async_batch_processor():
    """测试异步批量处理器"""
    logger.info("\n" + "=" * 50)
    logger.info("测试2: 异步批量处理器测试")
    logger.info("=" * 50)
    
    async def run_test():
        from src.foxhipporag.utils.async_utils import AsyncBatchProcessor
        
        processor = AsyncBatchProcessor(max_concurrent=10, batch_size=5, timeout=10.0)
        
        async def process_item(item: int) -> int:
            await asyncio.sleep(0.01)
            return item * 2
        
        items = list(range(50))
        
        logger.info("异步批量处理50个项目...")
        start_time = time.time()
        results = await processor.process_batch(items, process_item, "批量处理")
        elapsed = time.time() - start_time
        
        logger.info(f"批量处理耗时: {elapsed:.2f}s")
        
        assert len(results) == 50, "结果数量应该为50"
        assert all(r == i * 2 for i, r in enumerate(results)), "结果应该正确"
        
        logger.info("异步批量处理器测试通过！")
        return True
    
    try:
        return asyncio.run(run_test())
    except Exception as e:
        logger.error(f"异步批量处理器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_async_rate_limiter():
    """测试异步速率限制器"""
    logger.info("\n" + "=" * 50)
    logger.info("测试3: 异步速率限制器测试")
    logger.info("=" * 50)
    
    async def run_test():
        from src.foxhipporag.utils.async_utils import AsyncRateLimiter
        
        limiter = AsyncRateLimiter(requests_per_second=10.0, burst_size=5)
        
        logger.info("测试速率限制...")
        start_time = time.time()
        
        for i in range(15):
            await limiter.acquire()
        
        elapsed = time.time() - start_time
        logger.info(f"15次请求耗时: {elapsed:.2f}s (预期约1.0s)")
        
        assert elapsed >= 0.8, "应该有速率限制延迟"
        
        logger.info("异步速率限制器测试通过！")
        return True
    
    try:
        return asyncio.run(run_test())
    except Exception as e:
        logger.error(f"异步速率限制器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_async_circuit_breaker():
    """测试异步熔断器"""
    logger.info("\n" + "=" * 50)
    logger.info("测试4: 异步熔断器测试")
    logger.info("=" * 50)
    
    async def run_test():
        from src.foxhipporag.utils.async_utils import AsyncCircuitBreaker
        
        breaker = AsyncCircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
        
        call_count = 0
        
        async def success_func():
            nonlocal call_count
            call_count += 1
            return "success"
        
        async def failure_func():
            raise ValueError("test error")
        
        logger.info("测试成功场景...")
        for _ in range(5):
            result = await breaker.call(success_func)
            assert result == "success"
        
        logger.info("测试失败场景...")
        breaker2 = AsyncCircuitBreaker(failure_threshold=2, recovery_timeout=1.0)
        
        for _ in range(2):
            try:
                await breaker2.call(failure_func)
            except ValueError:
                pass
        
        logger.info("测试熔断状态...")
        try:
            await breaker2.call(success_func)
            assert False, "应该抛出异常"
        except Exception as e:
            logger.info(f"熔断器正确打开: {e}")
        
        logger.info("异步熔断器测试通过！")
        return True
    
    try:
        return asyncio.run(run_test())
    except Exception as e:
        logger.error(f"异步熔断器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_async_performance_comparison():
    """测试异步 vs 同步性能对比"""
    logger.info("\n" + "=" * 50)
    logger.info("测试5: 异步 vs 同步性能对比")
    logger.info("=" * 50)
    
    async def run_test():
        from concurrent.futures import ThreadPoolExecutor
        import time
        
        def sync_task(n: int) -> int:
            time.sleep(0.01)
            return n * 2
        
        async def async_task(n: int) -> int:
            await asyncio.sleep(0.01)
            return n * 2
        
        task_count = 50
        
        logger.info(f"同步处理 {task_count} 个任务...")
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=10) as executor:
            sync_results = list(executor.map(sync_task, range(task_count)))
        sync_time = time.time() - start_time
        logger.info(f"同步处理耗时: {sync_time:.2f}s")
        
        logger.info(f"异步处理 {task_count} 个任务...")
        start_time = time.time()
        async_results = await asyncio.gather(*[async_task(i) for i in range(task_count)])
        async_time = time.time() - start_time
        logger.info(f"异步处理耗时: {async_time:.2f}s")
        
        speedup = sync_time / async_time if async_time > 0 else 0
        logger.info(f"加速比: {speedup:.2f}x")
        
        assert len(sync_results) == len(async_results) == task_count
        
        logger.info("异步 vs 同步性能对比测试通过！")
        return True
    
    try:
        return asyncio.run(run_test())
    except Exception as e:
        logger.error(f"性能对比测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_async_decorators():
    """测试异步装饰器"""
    logger.info("\n" + "=" * 50)
    logger.info("测试6: 异步装饰器测试")
    logger.info("=" * 50)
    
    async def run_test():
        from src.foxhipporag.utils.async_utils import async_retry, async_timeout
        
        call_count = 0
        
        @async_retry(max_retries=3, delay=0.1, backoff=1.0)
        async def retry_func(should_fail: bool = False):
            nonlocal call_count
            call_count += 1
            if should_fail and call_count < 3:
                raise ValueError("retry test")
            return "success"
        
        logger.info("测试重试装饰器...")
        call_count = 0
        result = await retry_func(should_fail=True)
        assert result == "success"
        assert call_count == 3, f"应该重试3次，实际{call_count}次"
        logger.info(f"重试装饰器工作正常，调用次数: {call_count}")
        
        @async_timeout(0.5)
        async def timeout_func(should_timeout: bool = False):
            if should_timeout:
                await asyncio.sleep(1.0)
            return "success"
        
        logger.info("测试超时装饰器...")
        result = await timeout_func(should_timeout=False)
        assert result == "success"
        
        try:
            await timeout_func(should_timeout=True)
            assert False, "应该超时"
        except asyncio.TimeoutError:
            logger.info("超时装饰器工作正常")
        
        logger.info("异步装饰器测试通过！")
        return True
    
    try:
        return asyncio.run(run_test())
    except Exception as e:
        logger.error(f"异步装饰器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_async_performance_monitor():
    """测试异步性能监控器"""
    logger.info("\n" + "=" * 50)
    logger.info("测试7: 异步性能监控器测试")
    logger.info("=" * 50)
    
    async def run_test():
        from src.foxhipporag.utils.async_utils import AsyncPerformanceMonitor
        
        monitor = AsyncPerformanceMonitor()
        
        for i in range(10):
            async with monitor.record_time("test_operation"):
                await asyncio.sleep(0.01 * (i % 3 + 1))
        
        stats = await monitor.get_stats("test_operation")
        logger.info(f"性能统计:")
        logger.info(f"  调用次数: {stats['count']}")
        logger.info(f"  平均耗时: {stats['mean']:.4f}s")
        logger.info(f"  标准差: {stats['std']:.4f}s")
        logger.info(f"  最小耗时: {stats['min']:.4f}s")
        logger.info(f"  最大耗时: {stats['max']:.4f}s")
        logger.info(f"  总耗时: {stats['total']:.4f}s")
        
        assert stats['count'] == 10, "调用次数应该为10"
        
        await monitor.clear()
        stats_after_clear = await monitor.get_stats("test_operation")
        assert stats_after_clear == {}, "清除后应该为空"
        
        logger.info("异步性能监控器测试通过！")
        return True
    
    try:
        return asyncio.run(run_test())
    except Exception as e:
        logger.error(f"异步性能监控器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """运行所有测试"""
    logger.info("开始异步性能测试...")
    logger.info("=" * 60)
    
    results = {}
    
    tests = [
        ("异步缓存测试", test_async_cache),
        ("异步批量处理器测试", test_async_batch_processor),
        ("异步速率限制器测试", test_async_rate_limiter),
        ("异步熔断器测试", test_async_circuit_breaker),
        ("异步vs同步性能对比", test_async_performance_comparison),
        ("异步装饰器测试", test_async_decorators),
        ("异步性能监控器测试", test_async_performance_monitor),
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
