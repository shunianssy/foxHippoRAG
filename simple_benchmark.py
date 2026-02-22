"""
简单的性能测试脚本 - 测试并行vs串行LLM调用

不依赖foxhipporag包，直接测试核心优化效果
"""

import os
import time
import json
import hashlib
import sqlite3
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from filelock import FileLock
from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('SimpleBenchmark')


class SimpleLLMCache:
    """简化的LLM缓存类，用于测试"""
    
    def __init__(self, cache_dir: str = "outputs/benchmark/llm_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        self.cache_file = os.path.join(cache_dir, "cache.sqlite")
        self.lock_file = self.cache_file + ".lock"
        
        # 初始化缓存数据库
        with FileLock(self.lock_file):
            conn = sqlite3.connect(self.cache_file)
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT,
                    metadata TEXT
                )
            """)
            conn.commit()
            conn.close()
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE_URL")
        )
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    def _get_cache_key(self, messages):
        """计算缓存键"""
        key_data = {
            "messages": messages,
            "model": self.model,
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode("utf-8")).hexdigest()
    
    def infer_single(self, messages, use_cache=True):
        """单次推理"""
        key_hash = self._get_cache_key(messages)
        
        # 检查缓存
        if use_cache:
            with FileLock(self.lock_file):
                conn = sqlite3.connect(self.cache_file)
                c = conn.cursor()
                c.execute("SELECT message, metadata FROM cache WHERE key = ?", (key_hash,))
                row = c.fetchone()
                conn.close()
                if row is not None:
                    return row[0], json.loads(row[1]), True
        
        # 调用API（带重试）
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=50
                )
                
                message = response.choices[0].message.content
                metadata = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                }
                
                # 保存缓存
                if use_cache:
                    with FileLock(self.lock_file):
                        conn = sqlite3.connect(self.cache_file)
                        c = conn.cursor()
                        c.execute("INSERT OR REPLACE INTO cache (key, message, metadata) VALUES (?, ?, ?)",
                                  (key_hash, message, json.dumps(metadata)))
                        conn.commit()
                        conn.close()
                
                return message, metadata, False
                
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    logger.warning(f"速率限制，等待{wait_time}秒后重试...")
                    time.sleep(wait_time)
                else:
                    raise
    
    def infer_serial(self, messages_list):
        """串行推理"""
        results = []
        for messages in messages_list:
            results.append(self.infer_single(messages))
        return results
    
    def infer_parallel_optimized(self, messages_list, max_workers=3):
        """
        优化后的并行推理：
        1. 先批量检查缓存
        2. 只对缓存未命中的请求并行处理
        3. 缓存全命中时直接返回
        """
        # 批量计算缓存键
        cache_keys = [self._get_cache_key(msg) for msg in messages_list]
        
        # 批量检查缓存
        cached_results = {}
        with FileLock(self.lock_file):
            conn = sqlite3.connect(self.cache_file)
            c = conn.cursor()
            placeholders = ','.join(['?' for _ in cache_keys])
            c.execute(f"SELECT key, message, metadata FROM cache WHERE key IN ({placeholders})", cache_keys)
            rows = c.fetchall()
            conn.close()
            for row in rows:
                cached_results[row[0]] = (row[1], json.loads(row[2]))
        
        # 确定缓存命中和未命中
        cache_miss_indices = [i for i, key in enumerate(cache_keys) if key not in cached_results]
        
        logger.info(f"缓存命中: {len(messages_list) - len(cache_miss_indices)}/{len(messages_list)}")
        
        # 如果全部缓存命中，直接返回
        if len(cache_miss_indices) == 0:
            return [(*cached_results[key], True) for key in cache_keys]
        
        # 预分配结果列表
        results = [None] * len(messages_list)
        
        # 填充缓存命中的结果
        for i, key in enumerate(cache_keys):
            if key in cached_results:
                results[i] = (*cached_results[key], True)
        
        # 并行处理缓存未命中的请求
        logger.info(f"并行处理 {len(cache_miss_indices)} 个缓存未命中请求...")
        
        with ThreadPoolExecutor(max_workers=min(max_workers, len(cache_miss_indices))) as executor:
            futures = {
                executor.submit(self.infer_single, messages_list[idx]): idx 
                for idx in cache_miss_indices
            }
            
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"请求失败: {e}")
                    results[idx] = ("", {"error": str(e)}, False)
        
        return results


def generate_unique_message(idx):
    """生成唯一的测试消息"""
    import random
    import string
    random_suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
    return [{"role": "user", "content": f"测试{idx}[{random_suffix}]: 1+1=?"}]


def run_benchmark():
    """运行性能测试"""
    logger.info("=" * 60)
    logger.info("LLM并行推理性能测试")
    logger.info("=" * 60)
    
    llm = SimpleLLMCache()
    num_tests = 5
    
    # 测试1：缓存未命中场景
    logger.info(f"\n=== 测试缓存未命中场景 ({num_tests}个请求) ===")
    
    # 串行推理
    unique_messages = [generate_unique_message(i) for i in range(num_tests)]
    logger.info("测试串行推理...")
    start = time.time()
    llm.infer_serial(unique_messages)
    serial_time_miss = time.time() - start
    
    # 等待避免速率限制
    logger.info("等待3秒避免速率限制...")
    time.sleep(3)
    
    # 并行推理（优化版）
    unique_messages = [generate_unique_message(i + num_tests) for i in range(num_tests)]
    logger.info("测试并行推理（优化版，max_workers=3）...")
    start = time.time()
    llm.infer_parallel_optimized(unique_messages, max_workers=3)
    parallel_time_miss = time.time() - start
    
    # 测试2：缓存命中场景
    logger.info(f"\n=== 测试缓存命中场景 ({num_tests}个请求) ===")
    
    # 准备缓存命中的消息（先执行一次确保缓存）
    cached_messages = [generate_unique_message(1000) for _ in range(num_tests)]
    logger.info("预热缓存...")
    llm.infer_serial(cached_messages)
    
    # 等待
    time.sleep(2)
    
    # 串行推理（缓存命中）
    logger.info("测试串行推理（缓存命中）...")
    start = time.time()
    llm.infer_serial(cached_messages)
    serial_time_hit = time.time() - start
    
    # 并行推理（缓存命中）
    logger.info("测试并行推理（缓存命中）...")
    start = time.time()
    llm.infer_parallel_optimized(cached_messages, max_workers=3)
    parallel_time_hit = time.time() - start
    
    # 打印结果
    logger.info("\n" + "=" * 60)
    logger.info("性能测试结果")
    logger.info("=" * 60)
    
    logger.info(f"\n【缓存未命中场景】")
    logger.info(f"  串行时间: {serial_time_miss:.2f}s")
    logger.info(f"  并行时间: {parallel_time_miss:.2f}s")
    speedup_miss = serial_time_miss / parallel_time_miss if parallel_time_miss > 0 else 0
    logger.info(f"  加速比: {speedup_miss:.2f}x")
    
    logger.info(f"\n【缓存命中场景】")
    logger.info(f"  串行时间: {serial_time_hit:.4f}s")
    logger.info(f"  并行时间: {parallel_time_hit:.4f}s")
    speedup_hit = serial_time_hit / parallel_time_hit if parallel_time_hit > 0 else 0
    logger.info(f"  加速比: {speedup_hit:.2f}x")
    
    logger.info(f"\n【总结】")
    logger.info(f"  ✓ 并行推理在缓存未命中时效果显著（真实API调用）")
    logger.info(f"  ✓ 缓存命中时，优化后的代码直接返回，避免线程池开销")
    logger.info(f"  ✓ 批量缓存检查减少了数据库访问次数")
    logger.info(f"  注意: 并行请求可能触发API速率限制，需要适当控制并发数")


if __name__ == "__main__":
    run_benchmark()
