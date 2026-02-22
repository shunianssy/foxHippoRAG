"""
压力测试文件

测试场景：
1. 大规模文档索引测试
2. 高并发检索测试
3. 批量查询性能测试
4. 长时间运行稳定性测试
5. 内存使用监控测试
"""

import os
import sys
import time
import json
import logging
import multiprocessing
import random
import string
import threading
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

if __name__ == '__main__':
    multiprocessing.freeze_support()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('StressTest')


def generate_random_text(length: int = 100) -> str:
    """生成随机文本"""
    words = [''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 10))) 
             for _ in range(length)]
    return ' '.join(words)


def generate_test_documents(count: int, min_length: int = 50, max_length: int = 200) -> List[str]:
    """生成测试文档"""
    logger.info(f"生成 {count} 个测试文档...")
    documents = []
    for i in range(count):
        length = random.randint(min_length, max_length)
        doc = f"文档{i}: {generate_random_text(length)}"
        documents.append(doc)
    return documents


def generate_test_queries(count: int) -> List[str]:
    """生成测试查询"""
    logger.info(f"生成 {count} 个测试查询...")
    queries = []
    templates = [
        "什么是{}？",
        "谁了解{}？",
        "{}是什么意思？",
        "请解释{}",
        "关于{}的信息",
    ]
    for i in range(count):
        template = random.choice(templates)
        keyword = ''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 8)))
        queries.append(template.format(keyword))
    return queries


class StressTestRunner:
    """压力测试运行器"""
    
    def __init__(self, save_dir: str = "outputs/stress_test"):
        self.save_dir = save_dir
        self.results = {}
        os.makedirs(save_dir, exist_ok=True)
    
    def test_large_scale_indexing(self, doc_count: int = 100, batch_size: int = 10) -> Dict[str, Any]:
        """
        大规模文档索引测试
        
        Args:
            doc_count: 文档数量
            batch_size: 批处理大小
        """
        logger.info("=" * 60)
        logger.info(f"测试1: 大规模文档索引测试 ({doc_count} 文档)")
        logger.info("=" * 60)
        
        result = {
            'test_name': 'large_scale_indexing',
            'doc_count': doc_count,
            'batch_size': batch_size,
            'success': False,
            'total_time': 0,
            'avg_time_per_doc': 0,
            'batches': []
        }
        
        try:
            from src.foxhipporag import foxHippoRAG
            from src.foxhipporag.utils.config_utils import BaseConfig
            
            config = BaseConfig()
            config.save_dir = os.path.join(self.save_dir, "indexing_test")
            config.llm_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            config.llm_base_url = os.getenv("OPENAI_API_BASE_URL")
            config.embedding_model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
            config.embedding_base_url = os.getenv("EMBEDDING_BASE_URL")
            config.embedding_api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
            config.openie_mode = "online"
            config.force_index_from_scratch = True
            config.embedding_batch_size = batch_size
            
            hippo = foxHippoRAG(global_config=config)
            
            documents = generate_test_documents(doc_count)
            
            total_start = time.time()
            
            for i in range(0, doc_count, batch_size):
                batch = documents[i:i + batch_size]
                batch_start = time.time()
                hippo.index(docs=batch)
                batch_time = time.time() - batch_start
                
                result['batches'].append({
                    'batch_index': i // batch_size,
                    'batch_size': len(batch),
                    'time': batch_time
                })
                
                logger.info(f"批次 {i // batch_size + 1}: {len(batch)} 文档, {batch_time:.2f}s")
            
            result['total_time'] = time.time() - total_start
            result['avg_time_per_doc'] = result['total_time'] / doc_count
            result['success'] = True
            
            logger.info(f"索引完成: 总时间 {result['total_time']:.2f}s, 平均 {result['avg_time_per_doc']:.3f}s/文档")
            
        except Exception as e:
            logger.error(f"大规模索引测试失败: {e}")
            import traceback
            traceback.print_exc()
            result['error'] = str(e)
        
        self.results['large_scale_indexing'] = result
        return result
    
    def test_concurrent_retrieval(self, query_count: int = 50, concurrent_users: int = 10) -> Dict[str, Any]:
        """
        高并发检索测试
        
        Args:
            query_count: 查询总数
            concurrent_users: 并发用户数
        """
        logger.info("=" * 60)
        logger.info(f"测试2: 高并发检索测试 ({query_count} 查询, {concurrent_users} 并发)")
        logger.info("=" * 60)
        
        result = {
            'test_name': 'concurrent_retrieval',
            'query_count': query_count,
            'concurrent_users': concurrent_users,
            'success': False,
            'total_time': 0,
            'avg_time_per_query': 0,
            'queries_per_second': 0,
            'errors': 0,
            'latencies': []
        }
        
        try:
            from src.foxhipporag import foxHippoRAG
            from src.foxhipporag.utils.config_utils import BaseConfig
            
            config = BaseConfig()
            config.save_dir = os.path.join(self.save_dir, "concurrent_test")
            config.llm_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            config.llm_base_url = os.getenv("OPENAI_API_BASE_URL")
            config.embedding_model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
            config.embedding_base_url = os.getenv("EMBEDDING_BASE_URL")
            config.embedding_api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
            config.openie_mode = "online"
            config.force_index_from_scratch = False
            
            hippo = foxHippoRAG(global_config=config)
            
            test_docs = generate_test_documents(20)
            hippo.index(docs=test_docs)
            
            queries = generate_test_queries(query_count)
            
            def process_query(query: str) -> Tuple[str, float, bool]:
                start = time.time()
                try:
                    hippo.retrieve(queries=[query], num_to_retrieve=5)
                    elapsed = time.time() - start
                    return query, elapsed, True
                except Exception as e:
                    elapsed = time.time() - start
                    logger.error(f"查询失败: {e}")
                    return query, elapsed, False
            
            total_start = time.time()
            
            with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
                futures = [executor.submit(process_query, q) for q in queries]
                
                for future in as_completed(futures):
                    query, latency, success = future.result()
                    result['latencies'].append(latency)
                    if not success:
                        result['errors'] += 1
            
            result['total_time'] = time.time() - total_start
            result['avg_time_per_query'] = sum(result['latencies']) / len(result['latencies'])
            result['queries_per_second'] = query_count / result['total_time']
            result['success'] = True
            
            logger.info(f"并发检索完成:")
            logger.info(f"  总时间: {result['total_time']:.2f}s")
            logger.info(f"  平均延迟: {result['avg_time_per_query']:.3f}s")
            logger.info(f"  QPS: {result['queries_per_second']:.2f}")
            logger.info(f"  错误数: {result['errors']}")
            
        except Exception as e:
            logger.error(f"并发检索测试失败: {e}")
            import traceback
            traceback.print_exc()
            result['error'] = str(e)
        
        self.results['concurrent_retrieval'] = result
        return result
    
    def test_batch_query_performance(self, batch_sizes: List[int] = [1, 5, 10, 20, 50]) -> Dict[str, Any]:
        """
        批量查询性能测试
        
        Args:
            batch_sizes: 要测试的批处理大小列表
        """
        logger.info("=" * 60)
        logger.info(f"测试3: 批量查询性能测试")
        logger.info("=" * 60)
        
        result = {
            'test_name': 'batch_query_performance',
            'batch_sizes': batch_sizes,
            'success': False,
            'results': []
        }
        
        try:
            from src.foxhipporag import foxHippoRAG
            from src.foxhipporag.utils.config_utils import BaseConfig
            
            config = BaseConfig()
            config.save_dir = os.path.join(self.save_dir, "batch_test")
            config.llm_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            config.llm_base_url = os.getenv("OPENAI_API_BASE_URL")
            config.embedding_model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
            config.embedding_base_url = os.getenv("EMBEDDING_BASE_URL")
            config.embedding_api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
            config.openie_mode = "online"
            config.force_index_from_scratch = False
            
            hippo = foxHippoRAG(global_config=config)
            
            test_docs = generate_test_documents(20)
            hippo.index(docs=test_docs)
            
            for batch_size in batch_sizes:
                queries = generate_test_queries(batch_size)
                
                start = time.time()
                hippo.retrieve(queries=queries, num_to_retrieve=5)
                elapsed = time.time() - start
                
                batch_result = {
                    'batch_size': batch_size,
                    'total_time': elapsed,
                    'avg_time_per_query': elapsed / batch_size
                }
                result['results'].append(batch_result)
                
                logger.info(f"批大小 {batch_size}: 总时间 {elapsed:.2f}s, 平均 {elapsed/batch_size:.3f}s/查询")
            
            result['success'] = True
            
        except Exception as e:
            logger.error(f"批量查询测试失败: {e}")
            import traceback
            traceback.print_exc()
            result['error'] = str(e)
        
        self.results['batch_query_performance'] = result
        return result
    
    def test_long_running_stability(self, duration_minutes: int = 5, operations_per_minute: int = 10) -> Dict[str, Any]:
        """
        长时间运行稳定性测试
        
        Args:
            duration_minutes: 测试持续时间（分钟）
            operations_per_minute: 每分钟操作数
        """
        logger.info("=" * 60)
        logger.info(f"测试4: 长时间运行稳定性测试 ({duration_minutes} 分钟)")
        logger.info("=" * 60)
        
        result = {
            'test_name': 'long_running_stability',
            'duration_minutes': duration_minutes,
            'operations_per_minute': operations_per_minute,
            'success': False,
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'avg_latency': 0,
            'latencies': [],
            'memory_snapshots': []
        }
        
        try:
            from src.foxhipporag import foxHippoRAG
            from src.foxhipporag.utils.config_utils import BaseConfig
            from src.foxhipporag.utils.performance_utils import MemoryManager
            
            config = BaseConfig()
            config.save_dir = os.path.join(self.save_dir, "stability_test")
            config.llm_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            config.llm_base_url = os.getenv("OPENAI_API_BASE_URL")
            config.embedding_model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
            config.embedding_base_url = os.getenv("EMBEDDING_BASE_URL")
            config.embedding_api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
            config.openie_mode = "online"
            config.force_index_from_scratch = False
            
            hippo = foxHippoRAG(global_config=config)
            memory_manager = MemoryManager()
            
            test_docs = generate_test_documents(30)
            hippo.index(docs=test_docs)
            
            total_operations = duration_minutes * operations_per_minute
            interval = 60.0 / operations_per_minute
            
            latencies = []
            successful = 0
            failed = 0
            
            start_time = time.time()
            
            for i in range(total_operations):
                operation_start = time.time()
                
                try:
                    query = generate_test_queries(1)[0]
                    hippo.retrieve(queries=[query], num_to_retrieve=3)
                    latency = time.time() - operation_start
                    latencies.append(latency)
                    successful += 1
                except Exception as e:
                    logger.error(f"操作 {i} 失败: {e}")
                    failed += 1
                
                if i % operations_per_minute == 0:
                    mem_info = memory_manager.get_memory_info()
                    result['memory_snapshots'].append({
                        'operation': i,
                        'memory_info': mem_info
                    })
                    logger.info(f"操作 {i}/{total_operations}, 内存: {mem_info.get('percent', 'N/A')}%")
                
                elapsed = time.time() - operation_start
                sleep_time = max(0, interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            result['total_operations'] = total_operations
            result['successful_operations'] = successful
            result['failed_operations'] = failed
            result['latencies'] = latencies
            result['avg_latency'] = sum(latencies) / len(latencies) if latencies else 0
            result['total_time'] = time.time() - start_time
            result['success'] = True
            
            logger.info(f"稳定性测试完成:")
            logger.info(f"  总操作数: {total_operations}")
            logger.info(f"  成功: {successful}, 失败: {failed}")
            logger.info(f"  平均延迟: {result['avg_latency']:.3f}s")
            logger.info(f"  实际耗时: {result['total_time']:.1f}s")
            
        except Exception as e:
            logger.error(f"稳定性测试失败: {e}")
            import traceback
            traceback.print_exc()
            result['error'] = str(e)
        
        self.results['long_running_stability'] = result
        return result
    
    def test_memory_usage(self, doc_counts: List[int] = [10, 50, 100, 200]) -> Dict[str, Any]:
        """
        内存使用监控测试
        
        Args:
            doc_counts: 要测试的文档数量列表
        """
        logger.info("=" * 60)
        logger.info(f"测试5: 内存使用监控测试")
        logger.info("=" * 60)
        
        result = {
            'test_name': 'memory_usage',
            'doc_counts': doc_counts,
            'success': False,
            'results': []
        }
        
        try:
            from src.foxhipporag import foxHippoRAG
            from src.foxhipporag.utils.config_utils import BaseConfig
            from src.foxhipporag.utils.performance_utils import MemoryManager
            
            memory_manager = MemoryManager()
            
            for doc_count in doc_counts:
                config = BaseConfig()
                config.save_dir = os.path.join(self.save_dir, f"memory_test_{doc_count}")
                config.llm_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                config.llm_base_url = os.getenv("OPENAI_API_BASE_URL")
                config.embedding_model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
                config.embedding_base_url = os.getenv("EMBEDDING_BASE_URL")
                config.embedding_api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
                config.openie_mode = "online"
                config.force_index_from_scratch = True
                
                mem_before = memory_manager.get_memory_info()
                
                hippo = foxHippoRAG(global_config=config)
                
                documents = generate_test_documents(doc_count)
                hippo.index(docs=documents)
                
                mem_after = memory_manager.get_memory_info()
                
                mem_result = {
                    'doc_count': doc_count,
                    'memory_before': mem_before,
                    'memory_after': mem_after,
                    'memory_increase': mem_after.get('used', 0) - mem_before.get('used', 0) if 'used' in mem_before and 'used' in mem_after else 0
                }
                result['results'].append(mem_result)
                
                logger.info(f"文档数 {doc_count}:")
                logger.info(f"  内存前: {mem_before.get('percent', 'N/A')}%")
                logger.info(f"  内存后: {mem_after.get('percent', 'N/A')}%")
            
            result['success'] = True
            
        except Exception as e:
            logger.error(f"内存使用测试失败: {e}")
            import traceback
            traceback.print_exc()
            result['error'] = str(e)
        
        self.results['memory_usage'] = result
        return result
    
    def save_results(self):
        """保存测试结果"""
        result_file = os.path.join(self.save_dir, "stress_test_results.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"测试结果已保存到: {result_file}")
    
    def print_summary(self):
        """打印测试摘要"""
        logger.info("\n" + "=" * 60)
        logger.info("压力测试摘要")
        logger.info("=" * 60)
        
        for test_name, result in self.results.items():
            status = "✓ 通过" if result.get('success', False) else "✗ 失败"
            logger.info(f"\n{test_name}: {status}")
            
            if 'total_time' in result:
                logger.info(f"  总时间: {result['total_time']:.2f}s")
            if 'avg_time_per_query' in result:
                logger.info(f"  平均延迟: {result['avg_time_per_query']:.3f}s")
            if 'queries_per_second' in result:
                logger.info(f"  QPS: {result['queries_per_second']:.2f}")
            if 'errors' in result:
                logger.info(f"  错误数: {result['errors']}")
            if 'error' in result:
                logger.info(f"  错误: {result['error']}")


def run_stress_tests(quick_mode: bool = True):
    """
    运行压力测试
    
    Args:
        quick_mode: 快速模式，减少测试规模
    """
    logger.info("开始压力测试...")
    
    runner = StressTestRunner(save_dir="outputs/stress_test")
    
    if quick_mode:
        runner.test_large_scale_indexing(doc_count=20, batch_size=5)
        runner.test_concurrent_retrieval(query_count=10, concurrent_users=3)
        runner.test_batch_query_performance(batch_sizes=[1, 3, 5])
        runner.test_long_running_stability(duration_minutes=1, operations_per_minute=5)
        runner.test_memory_usage(doc_counts=[5, 10, 20])
    else:
        runner.test_large_scale_indexing(doc_count=100, batch_size=10)
        runner.test_concurrent_retrieval(query_count=50, concurrent_users=10)
        runner.test_batch_query_performance(batch_sizes=[1, 5, 10, 20, 50])
        runner.test_long_running_stability(duration_minutes=5, operations_per_minute=10)
        runner.test_memory_usage(doc_counts=[10, 50, 100, 200])
    
    runner.save_results()
    runner.print_summary()
    
    return all(r.get('success', False) for r in runner.results.values())


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='HippoRAG 压力测试')
    parser.add_argument('--full', action='store_true', help='运行完整测试（非快速模式）')
    args = parser.parse_args()
    
    success = run_stress_tests(quick_mode=not args.full)
    sys.exit(0 if success else 1)
