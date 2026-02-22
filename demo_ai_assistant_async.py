"""
AI管家演示 - 异步高性能版

这个演示展示了如何使用异步I/O优化构建高性能AI管家：
1. 完全异步I/O操作
2. 多级缓存优化
3. 并发控制和速率限制
4. 批量处理优化
5. 熔断保护和智能重试

性能优化：
- 使用asyncio实现真正的非阻塞I/O
- 连接池复用减少连接开销
- 信号量控制并发数避免过载
- 批量操作减少网络往返
- 智能重试机制提高可靠性
"""

import os
import sys
import time
import json
import hashlib
import logging
import multiprocessing
import asyncio
from typing import List, Dict, Tuple, Any, Optional
from dotenv import load_dotenv

load_dotenv()

if __name__ == '__main__':
    multiprocessing.freeze_support()

from src.foxhipporag.utils.async_utils import (
    AsyncOpenAIClient,
    AsyncLRUCache,
    AsyncSQLiteCache,
    AsyncBatchProcessor,
    AsyncRateLimiter,
    AsyncCircuitBreaker,
    async_retry,
    async_timeout,
    gather_with_concurrency,
    global_async_monitor
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('AsyncAIAssistant')


class AsyncAIAssistant:
    """异步AI管家类
    
    性能优化：
    1. 完全异步I/O操作
    2. 多级缓存（内存LRU + SQLite持久化）
    3. 并发控制和速率限制
    4. 批量处理优化
    5. 熔断保护和智能重试
    6. 懒加载初始化（快速启动）
    """
    
    def __init__(
        self, 
        save_dir: str = "outputs/ai_assistant_async",
        max_concurrent: int = 50,
        rate_limit: float = 30.0
    ):
        """初始化异步AI管家（轻量级初始化，延迟加载重型组件）
        
        Args:
            save_dir: 存储目录
            max_concurrent: 最大并发数
            rate_limit: 每秒请求数限制
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_API_BASE_URL")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        
        if not self.api_key:
            raise ValueError("请在.env文件中设置OPENAI_API_KEY")
        
        # 配置参数（延迟创建组件）
        self._max_concurrent = max_concurrent
        self._rate_limit = rate_limit
        
        # 延迟加载的组件
        self._client: Optional[AsyncOpenAIClient] = None
        self._memory_cache: Optional[AsyncLRUCache] = None
        self._persistent_cache: Optional[AsyncSQLiteCache] = None
        self._batch_processor: Optional[AsyncBatchProcessor] = None
        self._rate_limiter: Optional[AsyncRateLimiter] = None
        self._circuit_breaker: Optional[AsyncCircuitBreaker] = None
        
        self._init_lock = asyncio.Lock()
        
        self.conversation_history: List[Tuple[str, str]] = []
        
        self.system_prompt = """你是一个智能AI管家。你的任务是：

1. **信息提取**：从用户输入中提取关键信息
2. **智能记忆**：记住用户告诉你的重要信息
3. **主动回忆**：在回答时，主动参考之前对话中用户告诉你的信息
4. **个性化回应**：根据用户的历史信息提供定制化的回答

请用友好、自然的语气与用户交流。"""
        
        self._knowledge_base: Dict[str, Any] = {}
    
    async def _ensure_initialized(self):
        """确保组件已初始化（懒加载）"""
        if self._client is not None:
            return
        
        async with self._init_lock:
            if self._client is not None:
                return
            
            logger.info("正在初始化异步AI管家组件...")
            start_time = time.time()
            
            self._client = AsyncOpenAIClient(
                api_key=self.api_key,
                base_url=self.base_url,
                cache_dir=os.path.join(self.save_dir, "cache")
            )
            
            self._memory_cache = AsyncLRUCache(max_size=500)
            self._persistent_cache = AsyncSQLiteCache(
                os.path.join(self.save_dir, "cache"),
                "assistant_cache.sqlite"
            )
            
            self._batch_processor = AsyncBatchProcessor(
                max_concurrent=self._max_concurrent,
                batch_size=10,
                timeout=120.0
            )
            
            self._rate_limiter = AsyncRateLimiter(
                requests_per_second=self._rate_limit,
                burst_size=int(self._rate_limit * 2)
            )
            
            self._circuit_breaker = AsyncCircuitBreaker(
                failure_threshold=5,
                recovery_timeout=30.0
            )
            
            elapsed = time.time() - start_time
            logger.info(f"异步AI管家组件初始化完成！耗时: {elapsed:.2f}s")
    
    @property
    def client(self) -> AsyncOpenAIClient:
        """获取客户端（需要先调用 _ensure_initialized）"""
        if self._client is None:
            raise RuntimeError("组件未初始化，请先调用 await _ensure_initialized()")
        return self._client
    
    @property
    def memory_cache(self) -> AsyncLRUCache:
        """获取内存缓存"""
        if self._memory_cache is None:
            raise RuntimeError("组件未初始化")
        return self._memory_cache
    
    @property
    def persistent_cache(self) -> AsyncSQLiteCache:
        """获取持久化缓存"""
        if self._persistent_cache is None:
            raise RuntimeError("组件未初始化")
        return self._persistent_cache
    
    @property
    def batch_processor(self) -> AsyncBatchProcessor:
        """获取批量处理器"""
        if self._batch_processor is None:
            raise RuntimeError("组件未初始化")
        return self._batch_processor
    
    @property
    def rate_limiter(self) -> AsyncRateLimiter:
        """获取速率限制器"""
        if self._rate_limiter is None:
            raise RuntimeError("组件未初始化")
        return self._rate_limiter
    
    @property
    def circuit_breaker(self) -> AsyncCircuitBreaker:
        """获取熔断器"""
        if self._circuit_breaker is None:
            raise RuntimeError("组件未初始化")
        return self._circuit_breaker
    
    def _get_cache_key(self, text: str, prefix: str = "") -> str:
        """生成缓存键"""
        content = f"{prefix}:{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    @async_retry(max_retries=3, delay=1.0, backoff=2.0)
    @async_timeout(60.0)
    async def chat_completion(
        self, 
        messages: List[Dict],
        temperature: float = 0.7
    ) -> Tuple[str, Dict[str, Any]]:
        """异步聊天补全"""
        await self._ensure_initialized()  # 确保组件已初始化
        async with global_async_monitor.record_time("chat_completion"):
            await self.rate_limiter.acquire()
            
            cache_key = self._get_cache_key(
                json.dumps(messages, sort_keys=True),
                f"chat_{self.model}_{temperature}"
            )
            
            cached = await self.memory_cache.get(cache_key)
            if cached is not None:
                logger.debug("内存缓存命中")
                return cached['content'], cached['metadata']
            
            cached = await self.persistent_cache.get(cache_key, "chat")
            if cached is not None:
                logger.debug("持久化缓存命中")
                await self.memory_cache.put(cache_key, cached)
                return cached['content'], cached['metadata']
            
            content, metadata, _ = await self.circuit_breaker.call(
                self.client.chat_completion,
                messages=messages,
                model=self.model,
                temperature=temperature
            )
            
            cache_value = {'content': content, 'metadata': metadata}
            await self.memory_cache.put(cache_key, cache_value)
            await self.persistent_cache.put(cache_key, cache_value, "chat")
            
            return content, metadata
    
    async def chat_completion_batch(
        self,
        batch_messages: List[List[Dict]],
        temperature: float = 0.7,
        max_concurrent: int = 20
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """异步批量聊天补全"""
        async with global_async_monitor.record_time("chat_completion_batch"):
            tasks = [
                self.chat_completion(messages, temperature)
                for messages in batch_messages
            ]
            return await gather_with_concurrency(max_concurrent, *tasks)
    
    @async_retry(max_retries=3, delay=1.0, backoff=2.0)
    @async_timeout(30.0)
    async def get_embedding(self, text: str) -> Any:
        """异步获取嵌入向量"""
        await self._ensure_initialized()  # 确保组件已初始化
        async with global_async_monitor.record_time("embedding"):
            await self.rate_limiter.acquire()
            
            cache_key = self._get_cache_key(text, f"emb_{self.embedding_model}")
            
            cached = await self.memory_cache.get(cache_key)
            if cached is not None:
                import numpy as np
                return np.array(cached)
            
            embeddings, _ = await self.client.embedding(
                [text],
                model=self.embedding_model
            )
            
            await self.memory_cache.put(cache_key, embeddings[0].tolist())
            
            return embeddings[0]
    
    async def get_embedding_batch(
        self,
        texts: List[str],
        batch_size: int = 100
    ) -> Any:
        """异步批量获取嵌入向量"""
        import numpy as np
        
        async with global_async_monitor.record_time("embedding_batch"):
            results = []
            uncached_texts = []
            uncached_indices = []
            
            for i, text in enumerate(texts):
                cache_key = self._get_cache_key(text, f"emb_{self.embedding_model}")
                cached = await self.memory_cache.get(cache_key)
                if cached is not None:
                    results.append((i, np.array(cached)))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
            
            if uncached_texts:
                for i in range(0, len(uncached_texts), batch_size):
                    batch = uncached_texts[i:i + batch_size]
                    embeddings, _ = await self.client.embedding(
                        batch,
                        model=self.embedding_model
                    )
                    
                    for j, (text, emb) in enumerate(zip(batch, embeddings)):
                        original_idx = uncached_indices[i + j]
                        results.append((original_idx, emb))
                        
                        cache_key = self._get_cache_key(text, f"emb_{self.embedding_model}")
                        await self.memory_cache.put(cache_key, emb.tolist())
            
            results.sort(key=lambda x: x[0])
            return np.array([r[1] for r in results])
    
    async def extract_info(self, user_input: str) -> List[Dict[str, str]]:
        """异步提取信息"""
        async with global_async_monitor.record_time("extract_info"):
            cache_key = self._get_cache_key(user_input, "extract")
            
            cached = await self.memory_cache.get(cache_key)
            if cached is not None:
                return cached
            
            cached = await self.persistent_cache.get(cache_key, "extract")
            if cached is not None:
                await self.memory_cache.put(cache_key, cached)
                return cached
            
            extraction_prompt = f"""请从以下用户输入中提取关键信息，以JSON格式返回。

用户输入：{user_input}

提取规则：
1. 只提取有价值的信息，忽略问候语和一般性问题
2. 格式为JSON数组，每个元素是一个三元组：{{"subject": "主体", "predicate": "关系", "object": "客体"}}
3. 主体通常是"用户"或具体的人名/物名
4. 谓词描述关系（如"喜欢"、"名字是"、"职业是"等）
5. 客体是具体值

示例：
输入："我叫小明，喜欢打篮球"
输出：[{{"subject": "用户", "predicate": "名字是", "object": "小明"}}, {{"subject": "用户", "predicate": "喜欢", "object": "打篮球"}}]

输入："你好"
输出：[]

请直接返回JSON数组，不要有其他内容："""

            messages = [
                {"role": "system", "content": "你是一个信息提取助手，只返回JSON格式的数据。"},
                {"role": "user", "content": extraction_prompt}
            ]
            
            response, _ = await self.chat_completion(messages, temperature=0.1)
            
            result_text = response.strip()
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
            
            try:
                extracted_info = json.loads(result_text)
            except json.JSONDecodeError:
                extracted_info = []
            
            if isinstance(extracted_info, list):
                for info in extracted_info:
                    if isinstance(info, dict):
                        subject = info.get('subject', '')
                        predicate = info.get('predicate', '')
                        obj = info.get('object', '')
                        if subject and predicate and obj:
                            key = f"{subject}_{predicate}"
                            self._knowledge_base[key] = obj
                            logger.info(f"存储信息: {subject}{predicate}{obj}")
            
            await self.memory_cache.put(cache_key, extracted_info)
            await self.persistent_cache.put(cache_key, extracted_info, "extract")
            
            return extracted_info if isinstance(extracted_info, list) else []
    
    async def extract_info_batch(
        self,
        user_inputs: List[str],
        max_concurrent: int = 10
    ) -> List[List[Dict[str, str]]]:
        """异步批量提取信息"""
        tasks = [self.extract_info(inp) for inp in user_inputs]
        return await gather_with_concurrency(max_concurrent, *tasks)
    
    async def retrieve_relevant_info(
        self,
        query: str,
        top_k: int = 5
    ) -> List[str]:
        """异步检索相关信息"""
        async with global_async_monitor.record_time("retrieve"):
            cache_key = self._get_cache_key(query, f"retrieve_{top_k}")
            
            cached = await self.memory_cache.get(cache_key)
            if cached is not None:
                return cached
            
            cached = await self.persistent_cache.get(cache_key, "retrieve")
            if cached is not None:
                await self.memory_cache.put(cache_key, cached)
                return cached
            
            query_embedding = await self.get_embedding(query)
            
            results = []
            for key, value in self._knowledge_base.items():
                subject, predicate = key.rsplit('_', 1) if '_' in key else (key, '')
                doc = f"{subject}{predicate}{value}。"
                results.append(doc)
            
            if len(results) > top_k:
                results = results[:top_k]
            
            await self.memory_cache.put(cache_key, results)
            await self.persistent_cache.put(cache_key, results, "retrieve")
            
            return results
    
    async def retrieve_batch(
        self,
        queries: List[str],
        top_k: int = 5,
        max_concurrent: int = 20
    ) -> List[List[str]]:
        """异步批量检索"""
        tasks = [self.retrieve_relevant_info(q, top_k) for q in queries]
        return await gather_with_concurrency(max_concurrent, *tasks)
    
    async def process_user_input(self, user_input: str) -> str:
        """异步处理用户输入"""
        await self._ensure_initialized()  # 确保组件已初始化
        async with global_async_monitor.record_time("process_input"):
            start_time = time.time()
            
            extract_task = asyncio.create_task(self.extract_info(user_input))
            retrieve_task = asyncio.create_task(self.retrieve_relevant_info(user_input, 3))
            
            extracted_info, relevant_docs = await asyncio.gather(
                extract_task, retrieve_task
            )
            
            messages = [
                {"role": "system", "content": self.system_prompt}
            ]
            
            if relevant_docs:
                context = "\n".join([f"- {doc}" for doc in relevant_docs])
                messages.append({
                    "role": "assistant",
                    "content": f"让我回忆一下...\n根据我的记忆：\n{context}"
                })
            
            messages.append({"role": "user", "content": user_input})
            
            response, _ = await self.chat_completion(messages)
            
            self.conversation_history.append((user_input, response))
            if len(self.conversation_history) > 50:
                self.conversation_history = self.conversation_history[-50:]
            
            elapsed = time.time() - start_time
            logger.info(f"处理完成，耗时: {elapsed:.2f}s")
            
            return response
    
    async def process_batch(
        self,
        user_inputs: List[str],
        max_concurrent: int = 10
    ) -> List[str]:
        """异步批量处理用户输入"""
        async with global_async_monitor.record_time("process_batch"):
            start_time = time.time()
            logger.info(f"批量处理 {len(user_inputs)} 条输入")
            
            await self.extract_info_batch(user_inputs, max_concurrent)
            
            all_relevant_docs = await self.retrieve_batch(user_inputs, 3, max_concurrent)
            
            batch_messages = []
            for user_input, relevant_docs in zip(user_inputs, all_relevant_docs):
                messages = [
                    {"role": "system", "content": self.system_prompt}
                ]
                
                if relevant_docs:
                    context = "\n".join([f"- {doc}" for doc in relevant_docs])
                    messages.append({
                        "role": "assistant",
                        "content": f"让我回忆一下...\n根据我的记忆：\n{context}"
                    })
                
                messages.append({"role": "user", "content": user_input})
                batch_messages.append(messages)
            
            responses = await self.chat_completion_batch(
                batch_messages,
                max_concurrent=max_concurrent
            )
            
            response_texts = [r[0] for r in responses]
            
            for user_input, response in zip(user_inputs, response_texts):
                self.conversation_history.append((user_input, response))
            
            if len(self.conversation_history) > 200:
                self.conversation_history = self.conversation_history[-200:]
            
            elapsed = time.time() - start_time
            logger.info(f"批量处理完成，耗时: {elapsed:.2f}s，平均: {elapsed/len(user_inputs):.2f}s/条")
            
            return response_texts
    
    async def show_knowledge(self):
        """显示知识库内容"""
        await self._ensure_initialized()
        print("\n=== 知识库内容 ===")
        print(f"\n【存储的知识】")
        for key, value in list(self._knowledge_base.items())[:20]:
            subject, predicate = key.rsplit('_', 1) if '_' in key else (key, '')
            print(f"  {subject} {predicate} {value}")
        
        print(f"\n【缓存统计】")
        stats = await self.memory_cache.get_stats()
        print(f"  内存缓存: {stats['size']}/{stats['max_size']} 条")
        print(f"  命中率: {stats['hit_rate']:.2%}")
        
        print(f"\n【性能统计】")
        all_stats = await global_async_monitor.get_all_stats()
        for name, stat in all_stats.items():
            print(f"  {name}:")
            print(f"    调用次数: {stat.get('count', 0)}")
            print(f"    平均耗时: {stat.get('mean', 0):.3f}s")
            print(f"    总耗时: {stat.get('total', 0):.2f}s")
        
        print(f"\n【对话历史】")
        if self.conversation_history:
            for user, assistant in self.conversation_history[-5:]:
                print(f"  用户: {user}")
                print(f"  助手: {assistant[:50]}...")
                print()
        else:
            print("  暂无对话记录")
    
    async def clear_cache(self):
        """清除所有缓存"""
        if self._memory_cache is not None:
            await self._memory_cache.clear()
            logger.info("内存缓存已清除")
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        if self._memory_cache is not None:
            cache_stats = await self._memory_cache.get_stats()
        else:
            cache_stats = {'size': 0, 'max_size': 0, 'hit_rate': 0}
        
        perf_stats = await global_async_monitor.get_all_stats()
        
        return {
            'cache': cache_stats,
            'performance': perf_stats,
            'knowledge_base_size': len(self._knowledge_base),
            'conversation_history_size': len(self.conversation_history)
        }
    
    async def close(self):
        """关闭客户端"""
        if self._client is not None:
            await self._client.close()
            logger.info("异步客户端已关闭")


async def async_main():
    """异步主函数"""
    print("AI管家演示（异步高性能版 - 快速启动）")
    print()
    
    start_time = time.time()
    print("正在创建AI管家实例...")
    try:
        assistant = AsyncAIAssistant()
        elapsed = time.time() - start_time
        print(f"AI管家实例创建完成！启动耗时: {elapsed:.2f}s")
        print("（组件将在首次使用时延迟加载）")
    except Exception as e:
        print(f"创建失败: {e}")
        return
    
    print("\n=== 开始对话 ===")
    print("命令: '退出' - 结束对话 | '知识' - 显示知识库 | '统计' - 显示统计 | '清缓存' - 清除缓存")
    
    while True:
        try:
            user_input = input("\n用户: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == '退出':
                print("AI管家: 再见！")
                await assistant.close()
                break
            
            if user_input == '知识':
                await assistant.show_knowledge()
                continue
            
            if user_input == '统计':
                stats = await assistant.get_stats()
                print(f"\n统计信息:")
                print(f"  知识库大小: {stats['knowledge_base_size']}")
                print(f"  对话历史: {stats['conversation_history_size']}")
                print(f"  缓存命中率: {stats['cache']['hit_rate']:.2%}")
                continue
            
            if user_input == '清缓存':
                await assistant.clear_cache()
                print("缓存已清除")
                continue
            
            response = await assistant.process_user_input(user_input)
            print(f"AI管家: {response}")
            
        except KeyboardInterrupt:
            print("\n\nAI管家: 再见！")
            await assistant.close()
            break
        except Exception as e:
            print(f"发生错误: {e}")
            logger.error(f"主循环错误: {e}")


def main():
    """主函数入口"""
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\n程序已退出")


if __name__ == "__main__":
    main()
