import asyncio
import functools
import hashlib
import json
import os
import sqlite3
from copy import deepcopy
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor

import httpx
import openai
from filelock import FileLock
from openai import OpenAI
from openai import AzureOpenAI
from packaging import version
from tenacity import retry, stop_after_attempt, wait_fixed

from ..utils.config_utils import BaseConfig
from ..utils.llm_utils import (
    TextChatMessage
)
from ..utils.logging_utils import get_logger
from .base import BaseLLM, LLMConfig

logger = get_logger(__name__)

# 全局线程池，用于并行批量推理
_batch_executor = None

def _get_batch_executor(max_workers: int = 32):
    """获取或创建全局线程池执行器"""
    global _batch_executor
    if _batch_executor is None:
        _batch_executor = ThreadPoolExecutor(max_workers=max_workers)
    return _batch_executor

def cache_response(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # get messages from args or kwargs
        if args:
            messages = args[0]
        else:
            messages = kwargs.get("messages")
        if messages is None:
            raise ValueError("Missing required 'messages' parameter for caching.")

        # get model, seed and temperature from kwargs or self.llm_config.generate_params
        gen_params = getattr(self, "llm_config", {}).generate_params if hasattr(self, "llm_config") else {}
        model = kwargs.get("model", gen_params.get("model"))
        seed = kwargs.get("seed", gen_params.get("seed"))
        temperature = kwargs.get("temperature", gen_params.get("temperature"))

        # build key data, convert to JSON string and hash to generate key_hash
        key_data = {
            "messages": messages,  # messages requires JSON serializable
            "model": model,
            "seed": seed,
            "temperature": temperature,
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.sha256(key_str.encode("utf-8")).hexdigest()

        # the file name of lock, ensure mutual exclusion when accessing concurrently
        lock_file = self.cache_file_name + ".lock"

        # Try to read from SQLite cache
        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()
            # if the table does not exist, create it
            c.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT,
                    metadata TEXT
                )
            """)
            conn.commit()  # commit to save the table creation
            c.execute("SELECT message, metadata FROM cache WHERE key = ?", (key_hash,))
            row = c.fetchone()
            conn.close()
            if row is not None:
                message, metadata_str = row
                metadata = json.loads(metadata_str)
                # return cached result and mark as hit
                return message, metadata, True

        # if cache miss, call the original function to get the result
        result = func(self, *args, **kwargs)
        message, metadata = result

        # insert new result into cache
        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()
            # make sure the table exists again (if it doesn't exist, it would be created)
            c.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT,
                    metadata TEXT
                )
            """)
            metadata_str = json.dumps(metadata)
            c.execute("INSERT OR REPLACE INTO cache (key, message, metadata) VALUES (?, ?, ?)",
                      (key_hash, message, metadata_str))
            conn.commit()
            conn.close()

        return message, metadata, False

    return wrapper

def dynamic_retry_decorator(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        max_retries = getattr(self, "max_retries", 5)  
        dynamic_retry = retry(stop=stop_after_attempt(max_retries), wait=wait_fixed(1))
        decorated_func = dynamic_retry(func)
        return decorated_func(self, *args, **kwargs)
    return wrapper

class CacheOpenAI(BaseLLM):
    """OpenAI LLM implementation."""
    @classmethod
    def from_experiment_config(cls, global_config: BaseConfig) -> "CacheOpenAI":
        config_dict = global_config.__dict__
        config_dict['max_retries'] = global_config.max_retry_attempts
        cache_dir = os.path.join(global_config.save_dir, "llm_cache")
        return cls(cache_dir=cache_dir, global_config=global_config)

    def __init__(self, cache_dir, global_config, cache_filename: str = None,
                 high_throughput: bool = True,
                 **kwargs) -> None:

        super().__init__()
        self.cache_dir = cache_dir
        self.global_config = global_config

        self.llm_name = global_config.llm_name
        self.llm_base_url = global_config.llm_base_url

        os.makedirs(self.cache_dir, exist_ok=True)
        if cache_filename is None:
            cache_filename = f"{self.llm_name.replace('/', '_')}_cache.sqlite"
        self.cache_file_name = os.path.join(self.cache_dir, cache_filename)

        self._init_llm_config()
        if high_throughput:
            limits = httpx.Limits(max_connections=500, max_keepalive_connections=100)
            client = httpx.Client(limits=limits, timeout=httpx.Timeout(5*60, read=5*60))
        else:
            client = None

        self.max_retries = kwargs.get("max_retries", 2)

        if self.global_config.azure_endpoint is None:
            self.openai_client = OpenAI(base_url=self.llm_base_url, http_client=client, max_retries=self.max_retries)
        else:
            self.openai_client = AzureOpenAI(api_version=self.global_config.azure_endpoint.split('api-version=')[1],
                                             azure_endpoint=self.global_config.azure_endpoint, max_retries=self.max_retries)

    def _init_llm_config(self) -> None:
        config_dict = self.global_config.__dict__

        config_dict['llm_name'] = self.global_config.llm_name
        config_dict['llm_base_url'] = self.global_config.llm_base_url
        config_dict['generate_params'] = {
                "model": self.global_config.llm_name,
                "max_completion_tokens": config_dict.get("max_new_tokens", 400),
                "n": config_dict.get("num_gen_choices", 1),
                "seed": config_dict.get("seed", 0),
                "temperature": config_dict.get("temperature", 0.0),
            }

        self.llm_config = LLMConfig.from_dict(config_dict=config_dict)
        logger.debug(f"Init {self.__class__.__name__}'s llm_config: {self.llm_config}")

    @cache_response
    @dynamic_retry_decorator
    def infer(
        self,
        messages: List[TextChatMessage],
        **kwargs
    ) -> Tuple[List[TextChatMessage], dict]:
        params = deepcopy(self.llm_config.generate_params)
        if kwargs:
            params.update(kwargs)
        params["messages"] = messages
        logger.debug(f"Calling OpenAI GPT API with:\n{params}")

        if 'gpt' not in params['model'] or version.parse(openai.__version__) < version.parse("1.45.0"): # if we use vllm to call openai api or if we use openai but the version is too old to use 'max_completion_tokens' argument
            # TODO strange version change in openai protocol, but our current vllm version not changed yet
            params['max_tokens'] = params.pop('max_completion_tokens')

        response = self.openai_client.chat.completions.create(**params)

        response_message = response.choices[0].message.content
        assert isinstance(response_message, str), "response_message should be a string"
        
        metadata = {
            "prompt_tokens": response.usage.prompt_tokens, 
            "completion_tokens": response.usage.completion_tokens,
            "finish_reason": response.choices[0].finish_reason,
        }

        return response_message, metadata

    async def ainfer(
        self,
        messages: List[TextChatMessage],
        **kwargs
    ) -> Tuple[str, dict, bool]:
        """
        异步推理方法，使用asyncio进行非阻塞调用
        
        Args:
            messages: 消息列表
            **kwargs: 其他参数
            
        Returns:
            Tuple[str, dict, bool]: (响应消息, 元数据, 是否缓存命中)
        """
        # 首先检查缓存
        gen_params = getattr(self, "llm_config", {}).generate_params if hasattr(self, "llm_config") else {}
        model = kwargs.get("model", gen_params.get("model"))
        seed = kwargs.get("seed", gen_params.get("seed"))
        temperature = kwargs.get("temperature", gen_params.get("temperature"))
        
        key_data = {
            "messages": messages,
            "model": model,
            "seed": seed,
            "temperature": temperature,
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.sha256(key_str.encode("utf-8")).hexdigest()
        
        lock_file = self.cache_file_name + ".lock"
        
        # 尝试从缓存读取
        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT,
                    metadata TEXT
                )
            """)
            conn.commit()
            c.execute("SELECT message, metadata FROM cache WHERE key = ?", (key_hash,))
            row = c.fetchone()
            conn.close()
            if row is not None:
                message, metadata_str = row
                metadata = json.loads(metadata_str)
                return message, metadata, True
        
        # 缓存未命中，执行异步调用
        params = deepcopy(self.llm_config.generate_params)
        if kwargs:
            params.update(kwargs)
        params["messages"] = messages
        logger.debug(f"Async calling OpenAI GPT API with:\n{params}")
        
        if 'gpt' not in params['model'] or version.parse(openai.__version__) < version.parse("1.45.0"):
            params['max_tokens'] = params.pop('max_completion_tokens')
        
        # 使用OpenAI异步客户端
        try:
            from openai import AsyncOpenAI, AsyncAzureOpenAI
            
            if not hasattr(self, '_async_client') or self._async_client is None:
                if self.global_config.azure_endpoint is None:
                    limits = httpx.Limits(max_connections=500, max_keepalive_connections=100)
                    async_client = httpx.AsyncClient(limits=limits, timeout=httpx.Timeout(5*60, read=5*60))
                    self._async_client = AsyncOpenAI(
                        base_url=self.llm_base_url, 
                        http_client=async_client, 
                        max_retries=self.max_retries
                    )
                else:
                    self._async_client = AsyncAzureOpenAI(
                        api_version=self.global_config.azure_endpoint.split('api-version=')[1],
                        azure_endpoint=self.global_config.azure_endpoint,
                        max_retries=self.max_retries
                    )
            
            response = await self._async_client.chat.completions.create(**params)
            
            response_message = response.choices[0].message.content
            assert isinstance(response_message, str), "response_message should be a string"
            
            metadata = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "finish_reason": response.choices[0].finish_reason,
            }
            
            # 保存到缓存
            with FileLock(lock_file):
                conn = sqlite3.connect(self.cache_file_name)
                c = conn.cursor()
                c.execute("""
                    CREATE TABLE IF NOT EXISTS cache (
                        key TEXT PRIMARY KEY,
                        message TEXT,
                        metadata TEXT
                    )
                """)
                metadata_str = json.dumps(metadata)
                c.execute("INSERT OR REPLACE INTO cache (key, message, metadata) VALUES (?, ?, ?)",
                          (key_hash, response_message, metadata_str))
                conn.commit()
                conn.close()
            
            return response_message, metadata, False
            
        except Exception as e:
            logger.error(f"Async inference failed: {e}, falling back to sync")
            # 回退到同步调用
            result = self.infer(messages, **kwargs)
            return result

    def batch_infer(
        self,
        batch_messages: List[List[TextChatMessage]],
        max_workers: int = 32,
        **kwargs
    ) -> Tuple[List[str], List[dict], List[bool]]:
        """
        批量推理方法，优化缓存命中场景
        
        性能优化：
        - 先批量检查缓存
        - 只对缓存未命中的请求并行处理
        - 缓存全命中时直接返回，避免线程池开销
        
        Args:
            batch_messages: 批量消息列表
            max_workers: 最大并行工作线程数
            **kwargs: 其他参数传递给infer
            
        Returns:
            Tuple[List[str], List[dict], List[bool]]: (响应消息列表, 元数据列表, 缓存命中列表)
        """
        if not batch_messages:
            return [], [], []
            
        logger.info(f"Batch inference with {len(batch_messages)} messages, max_workers={max_workers}")
        
        # 获取生成参数
        gen_params = getattr(self, "llm_config", {}).generate_params if hasattr(self, "llm_config") else {}
        model = kwargs.get("model", gen_params.get("model"))
        seed = kwargs.get("seed", gen_params.get("seed"))
        temperature = kwargs.get("temperature", gen_params.get("temperature"))
        
        lock_file = self.cache_file_name + ".lock"
        
        # 批量计算所有缓存键
        cache_keys = []
        for messages in batch_messages:
            key_data = {
                "messages": messages,
                "model": model,
                "seed": seed,
                "temperature": temperature,
            }
            key_str = json.dumps(key_data, sort_keys=True, default=str)
            key_hash = hashlib.sha256(key_str.encode("utf-8")).hexdigest()
            cache_keys.append(key_hash)
        
        # 批量检查缓存
        cached_results = {}  # key_hash -> (message, metadata)
        cache_miss_indices = []  # 缓存未命中的索引
        
        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT,
                    metadata TEXT
                )
            """)
            conn.commit()
            
            # 使用IN查询批量获取缓存
            placeholders = ','.join(['?' for _ in cache_keys])
            c.execute(f"SELECT key, message, metadata FROM cache WHERE key IN ({placeholders})", cache_keys)
            rows = c.fetchall()
            conn.close()
            
            for row in rows:
                key, message, metadata_str = row
                cached_results[key] = (message, json.loads(metadata_str))
        
        # 确定缓存命中和未命中
        for idx, key_hash in enumerate(cache_keys):
            if key_hash not in cached_results:
                cache_miss_indices.append(idx)
        
        cache_hit_count = len(batch_messages) - len(cache_miss_indices)
        logger.info(f"Cache hits: {cache_hit_count}/{len(batch_messages)}")
        
        # 如果全部缓存命中，直接返回
        if len(cache_miss_indices) == 0:
            all_messages = []
            all_metadata = []
            all_cache_hits = []
            for key_hash in cache_keys:
                message, metadata = cached_results[key_hash]
                all_messages.append(message)
                all_metadata.append(metadata)
                all_cache_hits.append(True)
            return all_messages, all_metadata, all_cache_hits
        
        # 对于缓存未命中的请求，使用并行处理
        # 预分配结果列表
        all_messages = [None] * len(batch_messages)
        all_metadata = [None] * len(batch_messages)
        all_cache_hits = [False] * len(batch_messages)
        
        # 填充缓存命中的结果
        for idx, key_hash in enumerate(cache_keys):
            if key_hash in cached_results:
                message, metadata = cached_results[key_hash]
                all_messages[idx] = message
                all_metadata[idx] = metadata
                all_cache_hits[idx] = True
        
        # 并行处理缓存未命中的请求
        miss_messages = [batch_messages[idx] for idx in cache_miss_indices]
        
        logger.info(f"Processing {len(miss_messages)} cache-miss requests in parallel")
        
        executor = _get_batch_executor(min(max_workers, len(miss_messages)))
        
        futures = []
        for messages in miss_messages:
            future = executor.submit(self.infer, messages, **kwargs)
            futures.append(future)
        
        # 收集缓存未命中的结果
        for i, future in enumerate(futures):
            try:
                result = future.result(timeout=300)
                message, metadata, cache_hit = result
                original_idx = cache_miss_indices[i]
                all_messages[original_idx] = message
                all_metadata[original_idx] = metadata
                all_cache_hits[original_idx] = cache_hit
            except Exception as e:
                logger.error(f"Batch inference task failed: {e}")
                original_idx = cache_miss_indices[i]
                all_messages[original_idx] = ""
                all_metadata[original_idx] = {"error": str(e)}
                all_cache_hits[original_idx] = False
        
        return all_messages, all_metadata, all_cache_hits

    async def abatch_infer(
        self,
        batch_messages: List[List[TextChatMessage]],
        **kwargs
    ) -> Tuple[List[str], List[dict], List[bool]]:
        """
        异步批量推理方法，使用asyncio.gather并行处理
        
        Args:
            batch_messages: 批量消息列表
            **kwargs: 其他参数
            
        Returns:
            Tuple[List[str], List[dict], List[bool]]: (响应消息列表, 元数据列表, 缓存命中列表)
        """
        logger.info(f"Async batch inference with {len(batch_messages)} messages")
        
        # 创建所有异步任务
        tasks = [self.ainfer(messages, **kwargs) for messages in batch_messages]
        
        # 并行执行所有任务
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        all_messages = []
        all_metadata = []
        all_cache_hits = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Async batch inference task {i} failed: {result}")
                all_messages.append("")
                all_metadata.append({"error": str(result)})
                all_cache_hits.append(False)
            else:
                all_messages.append(result[0])
                all_metadata.append(result[1])
                all_cache_hits.append(result[2])
        
        cache_hit_count = sum(all_cache_hits)
        logger.info(f"Async batch inference completed. Cache hits: {cache_hit_count}/{len(batch_messages)}")
        
        return all_messages, all_metadata, all_cache_hits


