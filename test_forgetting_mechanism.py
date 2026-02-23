"""测试遗忘机制和异步组件"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_forgetting_mechanism():
    """测试遗忘机制"""
    print("=" * 60)
    print("测试遗忘机制")
    print("=" * 60)
    
    from src.foxhipporag.utils.forgetting_mechanism import (
        AsyncForgettingMechanism,
        ForgettingConfig,
        MemoryState
    )
    
    config = ForgettingConfig(
        compression_threshold=0.3,
        forgetting_threshold=0.1,
        max_nodes=100,
        max_edges=200,
    )
    
    mechanism = AsyncForgettingMechanism(config)
    
    # 添加节点
    print("\n添加测试节点...")
    await mechanism.async_add_node("node1", "张三是一名工程师", metadata={"type": "person"})
    await mechanism.async_add_node("node2", "李四是一名医生", metadata={"type": "person"})
    await mechanism.async_add_node("node3", "王五是一名教师", metadata={"type": "person"})
    
    # 添加边
    print("添加测试边...")
    await mechanism.async_add_edge("张三", "工程师", "职业是")
    await mechanism.async_add_edge("李四", "医生", "职业是")
    
    # 访问节点
    print("\n访问节点...")
    node = await mechanism.async_access_node("node1")
    print(f"访问 node1: {node.content}, 访问次数: {node.access_count}")
    
    # 搜索节点
    print("\n搜索节点...")
    results = await mechanism.async_search_nodes(query_text="工程师", top_k=5)
    print(f"搜索结果: {[(n.content, s) for n, s in results]}")
    
    # 获取统计
    stats = mechanism.get_stats()
    print(f"\n统计信息:")
    print(f"  总节点数: {stats['total_nodes']}")
    print(f"  总边数: {stats['total_edges']}")
    print(f"  活跃节点: {stats['active_nodes']}")
    print(f"  遗忘系数: {stats['forgetting_coefficient']:.4f}")
    
    print("\n✓ 遗忘机制测试通过")
    return True


async def test_async_utils():
    """测试异步工具"""
    print("\n" + "=" * 60)
    print("测试异步工具")
    print("=" * 60)
    
    from src.foxhipporag.utils.async_utils import (
        AsyncLRUCache,
        AsyncSQLiteCache,
        AsyncRateLimiter,
        AsyncCircuitBreaker,
    )
    
    # 测试 LRU 缓存
    print("\n测试 LRU 缓存...")
    cache = AsyncLRUCache(max_size=10)
    await cache.put("key1", {"data": "value1"})
    result = await cache.get("key1")
    print(f"缓存结果: {result}")
    
    stats = await cache.get_stats()
    print(f"缓存统计: {stats}")
    
    # 测试速率限制器
    print("\n测试速率限制器...")
    limiter = AsyncRateLimiter(requests_per_second=10, burst_size=5)
    await limiter.acquire()
    print("速率限制器获取成功")
    
    # 测试熔断器
    print("\n测试熔断器...")
    breaker = AsyncCircuitBreaker(failure_threshold=3, recovery_timeout=5.0)
    
    async def test_func():
        return "success"
    
    result = await breaker.call(test_func)
    print(f"熔断器调用结果: {result}")
    
    print("\n✓ 异步工具测试通过")
    return True


async def main():
    """主测试函数"""
    try:
        result1 = await test_forgetting_mechanism()
        result2 = await test_async_utils()
        
        if result1 and result2:
            print("\n" + "=" * 60)
            print("所有测试通过！")
            print("=" * 60)
            return 0
        else:
            print("\n部分测试失败")
            return 1
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
