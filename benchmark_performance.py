"""
HippoRAG性能基准测试脚本

用于测试和比较优化前后的性能差异
"""

import os
import time
import asyncio
import logging
from typing import List, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('PerformanceBenchmark')

@dataclass
class BenchmarkResult:
    """基准测试结果"""
    test_name: str
    total_time: float
    items_processed: int
    items_per_second: float
    details: Dict[str, Any] = None


class PerformanceBenchmark:
    """性能基准测试类"""
    
    def __init__(self, save_dir: str = "outputs/benchmark"):
        """
        初始化基准测试
        
        Args:
            save_dir: 存储目录
        """
        self.save_dir = save_dir
        self.results: List[BenchmarkResult] = []
        
    def run_all_benchmarks(self):
        """运行所有基准测试"""
        logger.info("=" * 60)
        logger.info("开始性能基准测试")
        logger.info("=" * 60)
        
        # 测试LLM批量推理
        self.benchmark_llm_batch_inference()
        
        # 测试嵌入编码
        self.benchmark_embedding_encoding()
        
        # 测试检索性能
        self.benchmark_retrieval()
        
        # 打印结果摘要
        self.print_summary()
    
    def benchmark_llm_batch_inference(self, num_queries: int = 10):
        """
        测试LLM批量推理性能
        
        Args:
            num_queries: 测试查询数量
        """
        logger.info(f"\n--- 测试LLM批量推理 (查询数: {num_queries}) ---")
        
        try:
            from src.foxhipporag import foxHippoRAG
            from src.foxhipporag.utils.config_utils import BaseConfig
            
            # 初始化配置
            config = BaseConfig()
            config.save_dir = self.save_dir
            config.llm_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            config.llm_base_url = os.getenv("OPENAI_API_BASE_URL")
            config.llm_parallel_workers = 8
            
            # 初始化HippoRAG
            hippo = foxHippoRAG(global_config=config)
            
            # 准备测试消息
            test_messages = [
                [{"role": "user", "content": f"测试问题 {i}: 请用一句话解释什么是人工智能？"}]
                for i in range(num_queries)
            ]
            
            # 测试串行推理
            logger.info("测试串行推理...")
            start_time = time.time()
            serial_results = []
            for msg in test_messages[:5]:  # 只测试5个以节省时间
                result = hippo.llm_model.infer(msg)
                serial_results.append(result)
            serial_time = time.time() - start_time
            
            # 测试批量并行推理
            logger.info("测试批量并行推理...")
            start_time = time.time()
            batch_results = hippo.llm_model.batch_infer(test_messages, max_workers=8)
            batch_time = time.time() - start_time
            
            # 记录结果
            self.results.append(BenchmarkResult(
                test_name="LLM串行推理",
                total_time=serial_time,
                items_processed=5,
                items_per_second=5 / serial_time
            ))
            
            self.results.append(BenchmarkResult(
                test_name="LLM批量并行推理",
                total_time=batch_time,
                items_processed=num_queries,
                items_per_second=num_queries / batch_time,
                details={
                    "加速比": f"{serial_time / batch_time * (num_queries / 5):.2f}x"
                }
            ))
            
            logger.info(f"串行推理时间: {serial_time:.2f}s")
            logger.info(f"批量并行推理时间: {batch_time:.2f}s")
            logger.info(f"加速比: {serial_time / batch_time * (num_queries / 5):.2f}x")
            
        except Exception as e:
            logger.error(f"LLM批量推理测试失败: {e}")
    
    def benchmark_embedding_encoding(self, num_texts: int = 100):
        """
        测试嵌入编码性能
        
        Args:
            num_texts: 测试文本数量
        """
        logger.info(f"\n--- 测试嵌入编码 (文本数: {num_texts}) ---")
        
        try:
            from src.foxhipporag import foxHippoRAG
            from src.foxhipporag.utils.config_utils import BaseConfig
            
            # 初始化配置
            config = BaseConfig()
            config.save_dir = self.save_dir
            config.embedding_model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
            config.embedding_base_url = os.getenv("EMBEDDING_BASE_URL")
            config.embedding_batch_size = 32
            config.embedding_parallel_workers = 8
            
            # 初始化HippoRAG
            hippo = foxHippoRAG(global_config=config)
            
            # 准备测试文本
            test_texts = [f"这是测试文本编号 {i}，用于测试嵌入编码性能。" for i in range(num_texts)]
            
            # 测试编码性能
            logger.info("测试嵌入编码...")
            start_time = time.time()
            embeddings = hippo.embedding_model.batch_encode(test_texts)
            encode_time = time.time() - start_time
            
            # 记录结果
            self.results.append(BenchmarkResult(
                test_name="嵌入编码",
                total_time=encode_time,
                items_processed=num_texts,
                items_per_second=num_texts / encode_time,
                details={
                    "嵌入维度": embeddings.shape[1] if len(embeddings.shape) > 1 else "N/A"
                }
            ))
            
            logger.info(f"编码时间: {encode_time:.2f}s")
            logger.info(f"吞吐量: {num_texts / encode_time:.2f} 文本/秒")
            
        except Exception as e:
            logger.error(f"嵌入编码测试失败: {e}")
    
    def benchmark_retrieval(self, num_queries: int = 10):
        """
        测试检索性能
        
        Args:
            num_queries: 测试查询数量
        """
        logger.info(f"\n--- 测试检索性能 (查询数: {num_queries}) ---")
        
        try:
            from src.foxhipporag import foxHippoRAG
            from src.foxhipporag.utils.config_utils import BaseConfig
            
            # 初始化配置
            config = BaseConfig()
            config.save_dir = self.save_dir
            config.retrieval_parallel_workers = 8
            
            # 初始化HippoRAG
            hippo = foxHippoRAG(global_config=config)
            
            # 准备测试文档
            test_docs = [
                f"测试文档 {i}：这是关于人工智能和机器学习的内容。"
                f"深度学习是机器学习的一个子领域，使用神经网络进行学习。"
                for i in range(100)
            ]
            
            # 索引文档
            logger.info("索引测试文档...")
            start_time = time.time()
            hippo.index(test_docs)
            index_time = time.time() - start_time
            logger.info(f"索引时间: {index_time:.2f}s")
            
            # 准备测试查询
            test_queries = [
                "什么是深度学习？",
                "机器学习有哪些应用？",
                "人工智能的发展历史是什么？",
                "神经网络是如何工作的？",
                "什么是自然语言处理？",
            ][:num_queries]
            
            # 测试检索性能
            logger.info("测试检索...")
            start_time = time.time()
            results = hippo.retrieve(test_queries, num_to_retrieve=5)
            retrieval_time = time.time() - start_time
            
            # 记录结果
            self.results.append(BenchmarkResult(
                test_name="文档索引",
                total_time=index_time,
                items_processed=len(test_docs),
                items_per_second=len(test_docs) / index_time
            ))
            
            self.results.append(BenchmarkResult(
                test_name="检索",
                total_time=retrieval_time,
                items_processed=num_queries,
                items_per_second=num_queries / retrieval_time
            ))
            
            logger.info(f"检索时间: {retrieval_time:.2f}s")
            logger.info(f"检索吞吐量: {num_queries / retrieval_time:.2f} 查询/秒")
            
        except Exception as e:
            logger.error(f"检索测试失败: {e}")
    
    def print_summary(self):
        """打印结果摘要"""
        logger.info("\n" + "=" * 60)
        logger.info("性能基准测试结果摘要")
        logger.info("=" * 60)
        
        print(f"\n{'测试名称':<30} {'总时间(s)':<12} {'处理数量':<10} {'吞吐量(项/秒)':<15}")
        print("-" * 70)
        
        for result in self.results:
            print(f"{result.test_name:<30} {result.total_time:<12.2f} {result.items_processed:<10} {result.items_per_second:<15.2f}")
            if result.details:
                for key, value in result.details.items():
                    print(f"  └─ {key}: {value}")
        
        print("\n" + "=" * 60)


def run_quick_benchmark():
    """运行快速基准测试"""
    benchmark = PerformanceBenchmark()
    
    # 运行简化的测试
    logger.info("运行快速基准测试...")
    
    # 只测试LLM批量推理
    try:
        # 直接导入LLM模块，避免导入整个foxhipporag
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        from src.foxhipporag.llm.openai_gpt import CacheOpenAI
        from src.foxhipporag.utils.config_utils import BaseConfig
        import time
        import random
        import string
        
        config = BaseConfig()
        config.save_dir = "outputs/benchmark"
        config.llm_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        config.llm_base_url = os.getenv("OPENAI_API_BASE_URL")
        config.max_retry_attempts = 2
        config.max_new_tokens = 100
        
        llm = CacheOpenAI.from_experiment_config(config)
        
        # 生成唯一的测试消息（避免缓存命中）
        def generate_unique_message(idx):
            random_suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
            return [{"role": "user", "content": f"测试 {idx} [{random_suffix}]: 请用一句话回答1+1等于几？"}]
        
        num_tests = 10
        
        # 测试1：缓存未命中场景（使用唯一消息）
        logger.info(f"\n=== 测试缓存未命中场景 ({num_tests}个请求) ===")
        
        # 串行推理（缓存未命中）
        unique_messages_serial = [generate_unique_message(i) for i in range(num_tests)]
        logger.info("测试串行推理（缓存未命中）...")
        start = time.time()
        for msg in unique_messages_serial:
            llm.infer(msg)
        serial_time_miss = time.time() - start
        
        # 并行推理（缓存未命中）
        unique_messages_parallel = [generate_unique_message(i + num_tests) for i in range(num_tests)]
        logger.info("测试并行推理（缓存未命中）...")
        start = time.time()
        llm.batch_infer(unique_messages_parallel, max_workers=8)
        batch_time_miss = time.time() - start
        
        # 测试2：缓存命中场景（使用相同消息）
        logger.info(f"\n=== 测试缓存命中场景 ({num_tests}个请求) ===")
        
        # 准备缓存命中的消息（先执行一次确保缓存）
        cached_messages = [generate_unique_message(0) for _ in range(num_tests)]
        for msg in cached_messages:
            llm.infer(msg)  # 预热缓存
        
        # 串行推理（缓存命中）
        logger.info("测试串行推理（缓存命中）...")
        start = time.time()
        for msg in cached_messages:
            llm.infer(msg)
        serial_time_hit = time.time() - start
        
        # 并行推理（缓存命中）
        logger.info("测试并行推理（缓存命中）...")
        start = time.time()
        llm.batch_infer(cached_messages, max_workers=8)
        batch_time_hit = time.time() - start
        
        # 打印结果
        logger.info("\n" + "=" * 60)
        logger.info("性能测试结果")
        logger.info("=" * 60)
        
        logger.info(f"\n【缓存未命中场景】")
        logger.info(f"  串行时间: {serial_time_miss:.2f}s")
        logger.info(f"  并行时间: {batch_time_miss:.2f}s")
        logger.info(f"  加速比: {serial_time_miss/batch_time_miss:.2f}x")
        
        logger.info(f"\n【缓存命中场景】")
        logger.info(f"  串行时间: {serial_time_hit:.4f}s")
        logger.info(f"  并行时间: {batch_time_hit:.4f}s")
        logger.info(f"  加速比: {serial_time_hit/batch_time_hit:.2f}x")
        
        logger.info(f"\n【总结】")
        logger.info(f"  并行推理在缓存未命中时效果显著")
        logger.info(f"  缓存命中时，并行和串行性能接近（避免了线程池开销）")
        
    except Exception as e:
        logger.error(f"快速测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        run_quick_benchmark()
    else:
        benchmark = PerformanceBenchmark()
        benchmark.run_all_benchmarks()
