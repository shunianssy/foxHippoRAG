"""
AI管家演示 - 异步高性能版

这个演示展示了如何使用异步I/O优化构建高性能AI管家：
1. 完全异步I/O操作
2. 多级缓存优化
3. 并发控制和速率限制
4. 批量处理优化
5. 熔断保护和智能重试
6. 拟人脑遗忘机制

性能优化：
- 使用asyncio实现真正的非阻塞I/O
- 连接池复用减少连接开销
- 信号量控制并发数避免过载
- 批量操作减少网络往返
- 智能重试机制提高可靠性

拟人脑遗忘机制：
- 对使用少的记忆片段进行压缩，减少空间占用和token使用
- 设计模糊化阈值，当记忆片段的使用频率低于阈值时，会被压缩
- 当记忆发生遗忘时，相应节点/边不会立即被删除，而是被标记为"遗忘"
- 当某次记忆构建再次检索到该节点时，节点会被重新激活并被赋予更高的权重
- 使用LRU序列对记忆节点和边进行管理，定期剪枝
- 设计遗忘系数，随图谱的膨胀而增大，避免图谱膨胀导致的检索时间退化
- 设计保护阈值，当图谱的节点数超过阈值时，会自动触发已遗忘节点的释放
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
from src.foxhipporag.utils.forgetting_mechanism import (
    AsyncForgettingMechanism,
    ForgettingConfig,
    MemoryState,
    MemoryNode
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
    7. 拟人脑遗忘机制（自动管理记忆生命周期）
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
        
        # 嵌入模型配置：支持单独的嵌入 API
        self.embedding_api_key = os.getenv("EMBEDDING_API_KEY") or self.api_key
        self.embedding_base_url = os.getenv("EMBEDDING_BASE_URL") or self.base_url
        # 使用 stepfun 支持的嵌入模型，如果 stepfun 不支持，需要配置单独的嵌入 API
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
        
        # 拟人脑遗忘机制
        self._forgetting_config = ForgettingConfig(
            compression_threshold=0.3,      # 使用频率低于30%时压缩
            forgetting_threshold=0.1,       # 使用频率低于10%时遗忘
            max_nodes=10000,                # 最大节点数
            max_edges=50000,                # 最大边数
            protected_threshold=0.8,        # 达到80%容量时触发清理
            pruning_interval=3600.0,        # 每小时剪枝一次
        )
        self._forgetting_mechanism: Optional[AsyncForgettingMechanism] = None
        
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
                cache_dir=os.path.join(self.save_dir, "cache"),
                embedding_api_key=self.embedding_api_key,
                embedding_base_url=self.embedding_base_url,
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
            
            # 初始化遗忘机制
            self._forgetting_mechanism = AsyncForgettingMechanism(self._forgetting_config)
            
            # 从持久化存储加载遗忘机制数据
            await self._load_forgetting_mechanism()
            
            elapsed = time.time() - start_time
            logger.info(f"异步AI管家组件初始化完成！耗时: {elapsed:.2f}s")
    
    async def _load_forgetting_mechanism(self):
        """从持久化存储加载遗忘机制数据"""
        forgetting_file = os.path.join(self.save_dir, "forgetting_mechanism.json")
        if os.path.exists(forgetting_file):
            try:
                with open(forgetting_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self._forgetting_mechanism = AsyncForgettingMechanism.from_dict(
                    data, self._forgetting_config
                )
                logger.info(f"从 {forgetting_file} 加载遗忘机制数据: {len(self._forgetting_mechanism.nodes)} 节点")
            except Exception as e:
                logger.error(f"加载遗忘机制数据失败: {e}")
    
    async def _save_forgetting_mechanism(self):
        """保存遗忘机制数据到持久化存储"""
        if self._forgetting_mechanism is None:
            return
        forgetting_file = os.path.join(self.save_dir, "forgetting_mechanism.json")
        try:
            data = self._forgetting_mechanism.to_dict()
            with open(forgetting_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"遗忘机制数据已保存到 {forgetting_file}")
        except Exception as e:
            logger.error(f"保存遗忘机制数据失败: {e}")
    
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
    
    @property
    def forgetting_mechanism(self) -> AsyncForgettingMechanism:
        """获取遗忘机制"""
        if self._forgetting_mechanism is None:
            raise RuntimeError("组件未初始化")
        return self._forgetting_mechanism
    
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
            
            # 获取已知的人名列表（从遗忘机制中提取）
            known_names = set()
            if self._forgetting_mechanism is not None:
                for node in self._forgetting_mechanism.nodes.values():
                    if node.metadata:
                        subject = node.metadata.get('subject', '')
                        obj = node.metadata.get('object', '')
                        if subject and subject != '用户':
                            known_names.add(subject)
                        if obj:
                            known_names.add(obj)
            
            known_names_str = "、".join(known_names) if known_names else "暂无"
            
            extraction_prompt = f"""请从以下用户输入中提取关键信息，以JSON格式返回。

用户输入：{user_input}

已知的人名：{known_names_str}

提取规则：
1. 提取所有有价值的信息，包括关于用户自己或其他人（如朋友、家人）的信息
2. 格式为JSON数组，每个元素是一个三元组：{{"subject": "主体", "predicate": "关系", "object": "客体"}}
3. 主体可以是"用户"或具体的人名（如"小熙"、"小明"等）
4. 谓词描述关系（如"喜欢"、"名字是"、"职业是"、"年龄是"、"想表白"等）
5. 客体是具体值，**必须明确指向具体人名或事物，不能使用代词（如"她"、"他"、"它"）**
6. **重要**：如果用户使用了代词（她/他/它），必须根据上下文替换为具体的人名或事物名称

示例：
输入："小明：喜欢打篮球"
输出：[{{"subject": "小明", "predicate": "喜欢", "object": "打篮球"}}]
我叫
输入："小熙喜欢爬山"
输出：[{{"subject": "小熙", "predicate": "喜欢", "object": "爬山"}}]

输入："我17岁了"
输出：[{{"subject": "用户", "predicate": "年龄是", "object": "17岁"}}]

输入："我想跟她表白"（已知小熙是之前提到的人）
输出：[{{"subject": "用户", "predicate": "想表白", "object": "小熙"}}]

输入："他喜欢吃苹果"（已知小明是之前提到的人）
输出：[{{"subject": "小明", "predicate": "喜欢", "object": "吃苹果"}}]

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
                            
                            # 添加到遗忘机制
                            node_id = hashlib.md5(f"{subject}_{predicate}_{obj}".encode()).hexdigest()
                            try:
                                await self.forgetting_mechanism.async_add_node(
                                    node_id=node_id,
                                    content=f"{subject}{predicate}{obj}",
                                    metadata={"subject": subject, "predicate": predicate, "object": obj}
                                )
                                logger.info(f"已添加节点到遗忘机制: {node_id[:8]}...")
                            except Exception as e:
                                logger.error(f"添加节点到遗忘机制失败: {e}")
                            
                            # 添加边关系
                            subject_node = hashlib.md5(subject.encode()).hexdigest()
                            object_node = hashlib.md5(obj.encode()).hexdigest()
                            try:
                                await self.forgetting_mechanism.async_add_edge(
                                    source_id=subject_node,
                                    target_id=object_node,
                                    relation=predicate
                                )
                            except Exception as e:
                                logger.error(f"添加边到遗忘机制失败: {e}")
            
            # 保存遗忘机制数据到持久化存储
            if extracted_info:
                await self._save_forgetting_mechanism()
            
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
        top_k: int = 5,
        use_cache: bool = True
    ) -> List[str]:
        """异步检索相关信息（带遗忘机制）
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            use_cache: 是否使用缓存（新增信息后应禁用缓存）
        """
        async with global_async_monitor.record_time("retrieve"):
            cache_key = self._get_cache_key(query, f"retrieve_{top_k}")
            
            if use_cache:
                cached = await self.memory_cache.get(cache_key)
                if cached is not None:
                    return cached
                
                cached = await self.persistent_cache.get(cache_key, "retrieve")
                if cached is not None:
                    await self.memory_cache.put(cache_key, cached)
                    return cached
            
            query_embedding = await self.get_embedding(query)
            
            # 使用遗忘机制搜索
            if self._forgetting_mechanism is not None:
                # 先检查遗忘机制中有多少节点
                stats = self._forgetting_mechanism.get_stats()
                logger.info(f"遗忘机制状态: 总节点={stats['total_nodes']}, 活跃={stats['active_nodes']}")
                
                search_results = await self.forgetting_mechanism.async_search_nodes(
                    query_embedding=query_embedding,
                    query_text=query,
                    top_k=top_k,
                    include_forgotten=False  # 不包含已遗忘的节点
                )
                
                logger.info(f"搜索返回 {len(search_results)} 条结果")
                
                # 访问找到的节点，更新其状态
                results = []
                for node, score in search_results:
                    await self.forgetting_mechanism.async_access_node(node.node_id)
                    results.append(node.content)
                    logger.info(f"检索到节点: {node.content}, 得分: {score:.2f}")
                    
                    # 如果节点被重新激活，记录日志
                    if node.state == MemoryState.REACTIVATED:
                        logger.info(f"记忆被重新激活: {node.content[:50]}...")
                
                logger.info(f"遗忘机制检索到 {len(results)} 条结果")
            else:
                # 回退到传统检索
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
            
            # 先提取信息（保存到知识库）
            extracted_info = await self.extract_info(user_input)
            
            # 再检索相关信息（禁用缓存以确保获取最新信息）
            relevant_docs = await self.retrieve_relevant_info(user_input, 3, use_cache=False)
            
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
        
        # 遗忘机制统计
        if self._forgetting_mechanism is not None:
            forgetting_stats = self._forgetting_mechanism.get_stats()
            print(f"\n【遗忘机制统计】")
            print(f"  总节点数: {forgetting_stats['total_nodes']}")
            print(f"  总边数: {forgetting_stats['total_edges']}")
            print(f"  活跃节点: {forgetting_stats['active_nodes']}")
            print(f"  压缩节点: {forgetting_stats['compressed_nodes']}")
            print(f"  遗忘节点: {forgetting_stats['forgotten_nodes']}")
            print(f"  重新激活节点: {forgetting_stats['reactivated_nodes']}")
            print(f"  遗忘系数: {forgetting_stats['forgetting_coefficient']:.4f}")
        
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
    
    async def trigger_pruning(self):
        """手动触发剪枝"""
        if self._forgetting_mechanism is not None:
            logger.info("手动触发剪枝...")
            await self._forgetting_mechanism.prune()
            stats = self._forgetting_mechanism.get_stats()
            logger.info(f"剪枝完成: 活跃={stats['active_nodes']}, 压缩={stats['compressed_nodes']}, 遗忘={stats['forgotten_nodes']}")
        else:
            logger.warning("遗忘机制未初始化")
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        if self._memory_cache is not None:
            cache_stats = await self._memory_cache.get_stats()
        else:
            cache_stats = {'size': 0, 'max_size': 0, 'hit_rate': 0}
        
        perf_stats = await global_async_monitor.get_all_stats()
        
        forgetting_stats = {}
        if self._forgetting_mechanism is not None:
            forgetting_stats = self._forgetting_mechanism.get_stats()
        
        return {
            'cache': cache_stats,
            'performance': perf_stats,
            'knowledge_base_size': len(self._knowledge_base),
            'conversation_history_size': len(self.conversation_history),
            'forgetting': forgetting_stats,
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
    print("命令: '退出' - 结束对话 | '知识' - 显示知识库 | '统计' - 显示统计 | '清缓存' - 清除缓存 | '剪枝' - 触发遗忘机制剪枝")
    
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
            
            if user_input == '剪枝':
                await assistant.trigger_pruning()
                print("剪枝操作已完成")
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
