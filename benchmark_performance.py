"""
HippoRAG性能基准测试脚本

用于测试和比较优化前后的性能差异
参考 demo_ai_assistant_hipporag.py 的轻量配置方式
"""

import os
import time
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
        self.save_dir = save_dir
        self.results: List[BenchmarkResult] = []
        
    def run_all_benchmarks(self):
        """运行所有基准测试"""
        logger.info("=" * 60)
        logger.info("开始性能基准测试")
        logger.info("=" * 60)
        
        # 测试LLM批量推理
        self.benchmark_llm_batch_inference()
        
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
            # 参考 demo_ai_assistant_hipporag.py 的轻量配置方式
            # 直接导入LLM模块，避免导入整个foxhipporag
            import sys
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            
            from src.foxhipporag.llm.openai_gpt import CacheOpenAI
            from src.foxhipporag.utils.config_utils import BaseConfig
            import random
            import string
            
            # 创建轻量配置
            config = BaseConfig()
            config.save_dir = self.save_dir
            config.llm_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            config.llm_base_url = os.getenv("OPENAI_API_BASE_URL")
            config.max_retry_attempts = 2
            config.max_new_tokens = 100
            
            llm = CacheOpenAI.from_experiment_config(config)
            
            # 生成唯一的测试消息（避免缓存命中）
            def generate_unique_message(idx):
                random_suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
                return [{"role": "user", "content": f"测试 {idx} [{random_suffix}]: 请用一句话回答1+1等于几？"}]
            
            # 测试1：缓存未命中场景（使用唯一消息）
            logger.info(f"\n=== 测试缓存未命中场景 ({num_queries}个请求) ===")
            
            # 串行推理（缓存未命中）
            unique_messages_serial = [generate_unique_message(i) for i in range(num_queries)]
            logger.info("测试串行推理（缓存未命中）...")
            start = time.time()
            for msg in unique_messages_serial:
                llm.infer(msg)
            serial_time_miss = time.time() - start
            
            # 并行推理（缓存未命中）
            unique_messages_parallel = [generate_unique_message(i + num_queries) for i in range(num_queries)]
            logger.info("测试并行推理（缓存未命中）...")
            start = time.time()
            llm.batch_infer(unique_messages_parallel, max_workers=8)
            batch_time_miss = time.time() - start
            
            # 测试2：缓存命中场景（使用相同消息）
            logger.info(f"\n=== 测试缓存命中场景 ({num_queries}个请求) ===")
            
            # 准备缓存命中的消息（先执行一次确保缓存）
            cached_messages = [generate_unique_message(0) for _ in range(num_queries)]
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
            
            # 记录结果
            self.results.append(BenchmarkResult(
                test_name="LLM串行推理(缓存未命中)",
                total_time=serial_time_miss,
                items_processed=num_queries,
                items_per_second=num_queries / serial_time_miss
            ))
            
            self.results.append(BenchmarkResult(
                test_name="LLM并行推理(缓存未命中)",
                total_time=batch_time_miss,
                items_processed=num_queries,
                items_per_second=num_queries / batch_time_miss,
                details={"加速比": f"{serial_time_miss / batch_time_miss:.2f}x"}
            ))
            
            self.results.append(BenchmarkResult(
                test_name="LLM串行推理(缓存命中)",
                total_time=serial_time_hit,
                items_processed=num_queries,
                items_per_second=num_queries / serial_time_hit
            ))
            
            self.results.append(BenchmarkResult(
                test_name="LLM并行推理(缓存命中)",
                total_time=batch_time_hit,
                items_processed=num_queries,
                items_per_second=num_queries / batch_time_hit,
                details={"加速比": f"{serial_time_hit / batch_time_hit:.2f}x"}
            ))
            
            # 打印详细结果
            logger.info("\n" + "=" * 60)
            logger.info("LLM批量推理测试结果")
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
            logger.info(f"  并行推理在缓存未命中时效果显著（真实API调用）")
            logger.info(f"  缓存命中时，优化后的代码避免了线程池开销")
            
        except Exception as e:
            logger.error(f"LLM批量推理测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    def print_summary(self):
        """打印结果摘要"""
        logger.info("\n" + "=" * 60)
        logger.info("性能基准测试结果摘要")
        logger.info("=" * 60)
        
        print(f"\n{'测试名称':<35} {'总时间(s)':<12} {'处理数量':<10} {'吞吐量(项/秒)':<15}")
        print("-" * 75)
        
        for result in self.results:
            print(f"{result.test_name:<35} {result.total_time:<12.2f} {result.items_processed:<10} {result.items_per_second:<15.2f}")
            if result.details:
                for key, value in result.details.items():
                    print(f"  └─ {key}: {value}")
        
        print("\n" + "=" * 60)


def run_quick_benchmark():
    """运行快速基准测试 - 仅测试LLM批量推理"""
    logger.info("=" * 60)
    logger.info("运行快速基准测试")
    logger.info("=" * 60)
    
    try:
        # 直接导入需要的模块，避免导入整个foxhipporag包
        import sys
        import importlib.util
        
        # 获取项目根目录
        project_root = os.path.dirname(os.path.abspath(__file__))
        
        # 直接加载 config_utils
        config_utils_path = os.path.join(project_root, "src", "foxhipporag", "utils", "config_utils.py")
        spec = importlib.util.spec_from_file_location("config_utils", config_utils_path)
        config_utils = importlib.util.module_from_spec(spec)
        sys.modules["config_utils"] = config_utils
        spec.loader.exec_module(config_utils)
        BaseConfig = config_utils.BaseConfig
        
        # 直接加载 openai_gpt（不经过__init__.py）
        openai_gpt_path = os.path.join(project_root, "src", "foxhipporag", "llm", "openai_gpt.py")
        spec = importlib.util.spec_from_file_location("openai_gpt", openai_gpt_path)
        openai_gpt = importlib.util.module_from_spec(spec)
        sys.modules["openai_gpt"] = openai_gpt
        # 先设置依赖
        sys.modules["src.foxhipporag.utils.config_utils"] = config_utils
        sys.modules["src.foxhipporag.utils"] = type(sys)('utils')
        sys.modules["src.foxhipporag.utils"].config_utils = config_utils
        spec.loader.exec_module(openai_gpt)
        CacheOpenAI = openai_gpt.CacheOpenAI
        
        import random
        import string
        
        # 创建轻量配置
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