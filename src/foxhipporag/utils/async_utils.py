"""
异步I/O优化模块

提供全面的异步I/O优化，包括：
1. 异步LLM调用（带连接池和重试）
2. 异步嵌入编码
3. 异步缓存操作
4. 异步批量处理
5. 异步检索操作

性能优化策略：
- 使用asyncio实现真正的非阻塞I/O
- 连接池复用减少连接开销
- 信号量控制并发数避免过载
- 批量操作减少网络往返
- 智能重试机制提高可靠性
"""

import asyncio
import hashlib
import json
import os
import sqlite3
import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Callable, TypeVar, ParamSpec, TYPE_CHECKING, Awaitable
from dataclasses import dataclass
from functools import wraps
from collections import OrderedDict

# 延迟导入重型库以提高启动速度
_aiohttp = None
_numpy = None

def _get_aiohttp():
    """延迟导入 aiohttp"""
    global _aiohttp
    if _aiohttp is None:
        import aiohttp
        _aiohttp = aiohttp
    return _aiohttp

def _get_numpy():
    """延迟导入 numpy"""
    global _numpy
    if _numpy is None:
        import numpy
        _numpy = numpy
    return _numpy

# 类型检查时导入（不影响运行时性能）
if TYPE_CHECKING:
    import aiohttp

logger = logging.getLogger(__name__)

P = ParamSpec('P')
T = TypeVar('T')


@dataclass
class AsyncConfig:
    """异步配置"""
    max_concurrent_requests: int = 100
    max_connections_per_host: int = 50
    connection_timeout: float = 30.0
    read_timeout: float = 300.0
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_backoff: float = 2.0
    cache_enabled: bool = True
    cache_max_memory_size: int = 1000


class AsyncLRUCache:
    """异步线程安全的LRU缓存"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: OrderedDict = OrderedDict()
        self._lock = asyncio.Lock()
        self._stats = {'hits': 0, 'misses': 0}
    
    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._stats['hits'] += 1
                return self._cache[key]
            self._stats['misses'] += 1
            return None
    
    async def put(self, key: str, value: Any) -> None:
        async with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self.max_size:
                    self._cache.popitem(last=False)
            self._cache[key] = value
    
    async def delete(self, key: str) -> bool:
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    async def clear(self) -> None:
        async with self._lock:
            self._cache.clear()
            self._stats = {'hits': 0, 'misses': 0}
    
    async def get_stats(self) -> Dict[str, Any]:
        async with self._lock:
            total = self._stats['hits'] + self._stats['misses']
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'hit_rate': self._stats['hits'] / total if total > 0 else 0
            }


class AsyncSQLiteCache:
    """异步SQLite缓存（使用线程池执行）- 懒加载优化"""
    
    def __init__(self, cache_dir: str, cache_name: str = "async_cache.sqlite"):
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_file = os.path.join(cache_dir, cache_name)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._initialized = False  # 懒加载标记
        self._init_lock = asyncio.Lock()
    
    async def _ensure_initialized(self):
        """确保数据库已初始化（懒加载）"""
        if self._initialized:
            return
        
        async with self._init_lock:
            if self._initialized:
                return
            
            loop = self._get_loop()
            
            def _sync_init():
                conn = sqlite3.connect(self.cache_file)
                c = conn.cursor()
                c.execute("""
                    CREATE TABLE IF NOT EXISTS async_cache (
                        key TEXT PRIMARY KEY,
                        value TEXT,
                        namespace TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP
                    )
                """)
                c.execute("CREATE INDEX IF NOT EXISTS idx_namespace ON async_cache(namespace)")
                c.execute("CREATE INDEX IF NOT EXISTS idx_expires ON async_cache(expires_at)")
                conn.commit()
                conn.close()
            
            await loop.run_in_executor(None, _sync_init)
            self._initialized = True
    
    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """获取事件循环"""
        if self._loop is None:
            self._loop = asyncio.get_event_loop()
        return self._loop
    
    async def get(self, key: str, namespace: str = "default") -> Optional[Any]:
        """异步获取缓存"""
        await self._ensure_initialized()  # 懒加载初始化
        loop = self._get_loop()
        
        def _sync_get():
            conn = sqlite3.connect(self.cache_file)
            c = conn.cursor()
            c.execute(
                "SELECT value FROM async_cache WHERE key = ? AND namespace = ?",
                (key, namespace)
            )
            row = c.fetchone()
            conn.close()
            return row[0] if row else None
        
        result = await loop.run_in_executor(None, _sync_get)
        return json.loads(result) if result else None
    
    async def put(self, key: str, value: Any, namespace: str = "default", 
                  ttl_seconds: Optional[int] = None) -> None:
        """异步存储缓存"""
        await self._ensure_initialized()  # 懒加载初始化
        loop = self._get_loop()
        
        def _sync_put():
            conn = sqlite3.connect(self.cache_file)
            c = conn.cursor()
            expires_at = None
            if ttl_seconds:
                expires_at = f"datetime('now', '+{ttl_seconds} seconds')"
            c.execute(
                """INSERT OR REPLACE INTO async_cache (key, value, namespace, expires_at) 
                   VALUES (?, ?, ?, """ + (expires_at if expires_at else "NULL") + """)""",
                (key, json.dumps(value, ensure_ascii=False), namespace)
            )
            conn.commit()
            conn.close()
        
        await loop.run_in_executor(None, _sync_put)
    
    async def get_batch(self, keys: List[str], namespace: str = "default") -> Dict[str, Any]:
        """异步批量获取缓存"""
        loop = self._get_loop()
        
        def _sync_batch_get():
            conn = sqlite3.connect(self.cache_file)
            c = conn.cursor()
            placeholders = ','.join(['?' for _ in keys])
            c.execute(
                f"SELECT key, value FROM async_cache WHERE key IN ({placeholders}) AND namespace = ?",
                keys + [namespace]
            )
            rows = c.fetchall()
            conn.close()
            return {row[0]: json.loads(row[1]) for row in rows}
        
        return await loop.run_in_executor(None, _sync_batch_get)
    
    async def put_batch(self, items: Dict[str, Any], namespace: str = "default",
                        ttl_seconds: Optional[int] = None) -> None:
        """异步批量存储缓存"""
        loop = self._get_loop()
        
        def _sync_batch_put():
            conn = sqlite3.connect(self.cache_file)
            c = conn.cursor()
            for key, value in items.items():
                c.execute(
                    "INSERT OR REPLACE INTO async_cache (key, value, namespace) VALUES (?, ?, ?)",
                    (key, json.dumps(value, ensure_ascii=False), namespace)
                )
            conn.commit()
            conn.close()
        
        await loop.run_in_executor(None, _sync_batch_put)
    
    async def delete(self, key: str, namespace: str = "default") -> bool:
        """异步删除缓存"""
        loop = self._get_loop()
        
        def _sync_delete():
            conn = sqlite3.connect(self.cache_file)
            c = conn.cursor()
            c.execute(
                "DELETE FROM async_cache WHERE key = ? AND namespace = ?",
                (key, namespace)
            )
            deleted = c.rowcount > 0
            conn.commit()
            conn.close()
            return deleted
        
        return await loop.run_in_executor(None, _sync_delete)
    
    async def clear_expired(self) -> int:
        """清理过期缓存"""
        loop = self._get_loop()
        
        def _sync_clear():
            conn = sqlite3.connect(self.cache_file)
            c = conn.cursor()
            c.execute("DELETE FROM async_cache WHERE expires_at IS NOT NULL AND expires_at < datetime('now')")
            deleted = c.rowcount
            conn.commit()
            conn.close()
            return deleted
        
        return await loop.run_in_executor(None, _sync_clear)


class AsyncHTTPClient:
    """异步HTTP客户端（带连接池和重试）"""
    
    def __init__(self, config: AsyncConfig = None):
        if config is None:
            config = AsyncConfig()
        self.config = config
        self._semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        self._session: Optional[Any] = None  # aiohttp.ClientSession
    
    async def get_session(self) -> Any:
        """获取或创建会话（延迟加载 aiohttp）"""
        aiohttp = _get_aiohttp()
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(
                total=self.config.read_timeout,
                connect=self.config.connection_timeout
            )
            connector = aiohttp.TCPConnector(
                limit=self.config.max_connections_per_host,
                limit_per_host=self.config.max_connections_per_host,
                enable_cleanup_closed=True
            )
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector
            )
        return self._session
    
    async def close(self):
        """关闭会话"""
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None
    
    async def request(
        self, 
        method: str, 
        url: str, 
        **kwargs
    ) -> Tuple[Any, Dict[str, Any]]:
        """发送请求（带重试和限流）"""
        aiohttp = _get_aiohttp()
        async with self._semaphore:
            session = await self.get_session()
            
            last_error = None
            for attempt in range(self.config.max_retries):
                try:
                    async with session.request(method, url, **kwargs) as response:
                        if response.status == 200:
                            data = await response.json()
                            return data, {'status': response.status}
                        elif response.status in [429, 503]:
                            retry_after = float(response.headers.get('Retry-After', self.config.retry_delay))
                            await asyncio.sleep(retry_after * (self.config.retry_backoff ** attempt))
                            continue
                        else:
                            text = await response.text()
                            raise aiohttp.ClientResponseError(
                                request_info=response.request_info,
                                history=response.history,
                                status=response.status,
                                message=text
                            )
                except aiohttp.ClientError as e:
                    last_error = e
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(self.config.retry_delay * (self.config.retry_backoff ** attempt))
                    continue
                except asyncio.TimeoutError as e:
                    last_error = e
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(self.config.retry_delay * (self.config.retry_backoff ** attempt))
                    continue
            
            raise last_error or Exception(f"Request failed after {self.config.max_retries} retries")
    
    async def get(self, url: str, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        return await self.request('GET', url, **kwargs)
    
    async def post(self, url: str, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        return await self.request('POST', url, **kwargs)
    
    async def request_batch(
        self, 
        requests: List[Dict[str, Any]]
    ) -> List[Tuple[Any, Dict[str, Any]]]:
        """批量发送请求"""
        tasks = [
            self.request(req['method'], req['url'], **req.get('kwargs', {}))
            for req in requests
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)


class AsyncOpenAIClient:
    """异步OpenAI客户端"""
    
    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        config: AsyncConfig = None,
        cache_dir: str = "outputs/async_cache",
        # 支持单独的嵌入 API 配置
        embedding_api_key: Optional[str] = None,
        embedding_base_url: Optional[str] = None,
    ):
        self.api_key = api_key
        self.base_url = base_url or "https://api.openai.com/v1"
        self.config = config or AsyncConfig()
        self.http_client = AsyncHTTPClient(self.config)
        self.memory_cache = AsyncLRUCache(max_size=self.config.cache_max_memory_size)
        self.persistent_cache = AsyncSQLiteCache(cache_dir, "openai_async_cache.sqlite")
        
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # 嵌入 API 配置
        self.embedding_api_key = embedding_api_key or api_key
        self.embedding_base_url = embedding_base_url or self.base_url
        self.embedding_headers = {
            "Authorization": f"Bearer {self.embedding_api_key}",
            "Content-Type": "application/json"
        }
    
    def _compute_cache_key(self, messages: List[Dict], model: str, **kwargs) -> str:
        """计算缓存键"""
        content = json.dumps({
            'messages': messages,
            'model': model,
            **{k: v for k, v in kwargs.items() if k in ['temperature', 'max_tokens', 'seed']}
        }, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()
    
    async def chat_completion(
        self,
        messages: List[Dict],
        model: str = "gpt-4o-mini",
        **kwargs
    ) -> Tuple[str, Dict[str, Any], bool]:
        """异步聊天补全"""
        cache_key = self._compute_cache_key(messages, model, **kwargs)
        
        if self.config.cache_enabled:
            cached = await self.memory_cache.get(cache_key)
            if cached is not None:
                return cached['content'], cached['metadata'], True
            
            cached = await self.persistent_cache.get(cache_key, "chat_completion")
            if cached is not None:
                await self.memory_cache.put(cache_key, cached)
                return cached['content'], cached['metadata'], True
        
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            **kwargs
        }
        
        data, resp_meta = await self.http_client.post(
            url,
            headers=self.headers,
            json=payload
        )
        
        content = data['choices'][0]['message']['content']
        metadata = {
            'prompt_tokens': data['usage']['prompt_tokens'],
            'completion_tokens': data['usage']['completion_tokens'],
            'model': data['model'],
            'finish_reason': data['choices'][0]['finish_reason']
        }
        
        if self.config.cache_enabled:
            cache_value = {'content': content, 'metadata': metadata}
            await self.memory_cache.put(cache_key, cache_value)
            await self.persistent_cache.put(cache_key, cache_value, "chat_completion")
        
        return content, metadata, False
    
    async def chat_completion_batch(
        self,
        batch_messages: List[List[Dict]],
        model: str = "gpt-4o-mini",
        **kwargs
    ) -> List[Tuple[str, Dict[str, Any], bool]]:
        """异步批量聊天补全"""
        tasks = [
            self.chat_completion(messages, model, **kwargs)
            for messages in batch_messages
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def embedding(
        self,
        texts: List[str],
        model: str = "text-embedding-3-small"
    ) -> Tuple[Any, Dict[str, Any]]:
        """异步嵌入编码"""
        np = _get_numpy()
        cache_key = hashlib.md5(json.dumps({'texts': texts, 'model': model}).encode()).hexdigest()
        
        if self.config.cache_enabled:
            cached = await self.memory_cache.get(cache_key)
            if cached is not None:
                return np.array(cached['embeddings']), cached['metadata']
        
        # 使用单独的嵌入 API 配置
        url = f"{self.embedding_base_url}/embeddings"
        payload = {
            "model": model,
            "input": [t.replace("\n", " ") for t in texts]
        }
        
        data, resp_meta = await self.http_client.post(
            url,
            headers=self.embedding_headers,
            json=payload
        )
        
        embeddings = np.array([item['embedding'] for item in data['data']])
        metadata = {
            'model': data['model'],
            'total_tokens': data['usage']['total_tokens']
        }
        
        if self.config.cache_enabled:
            await self.memory_cache.put(cache_key, {
                'embeddings': embeddings.tolist(),
                'metadata': metadata
            })
        
        return embeddings, metadata
    
    async def embedding_batch(
        self,
        all_texts: List[List[str]],
        model: str = "text-embedding-3-small",
        batch_size: int = 100
    ) -> List[Tuple[Any, Dict[str, Any]]]:
        """异步批量嵌入编码"""
        tasks = []
        for texts in all_texts:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                tasks.append(self.embedding(batch, model))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
    
    async def close(self):
        """关闭客户端"""
        await self.http_client.close()


class AsyncBatchProcessor:
    """异步批量处理器"""
    
    def __init__(
        self,
        max_concurrent: int = 50,
        batch_size: int = 10,
        timeout: float = 300.0
    ):
        self.max_concurrent = max_concurrent
        self.batch_size = batch_size
        self.timeout = timeout
        self._semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_batch(
        self,
        items: List[Any],
        process_fn: Callable[[Any], Awaitable[T]],
        desc: str = "Processing"
    ) -> List[T]:
        """异步批量处理"""
        results = [None] * len(items)
        
        async def process_with_semaphore(idx: int, item: Any):
            async with self._semaphore:
                try:
                    result = await asyncio.wait_for(
                        process_fn(item),
                        timeout=self.timeout
                    )
                    return idx, result, None
                except Exception as e:
                    return idx, None, e
        
        tasks = [
            process_with_semaphore(i, item)
            for i, item in enumerate(items)
        ]
        
        completed = 0
        for coro in asyncio.as_completed(tasks):
            idx, result, error = await coro
            completed += 1
            if error:
                logger.error(f"Batch processing error at index {idx}: {error}")
                results[idx] = None
            else:
                results[idx] = result
            
            if completed % self.batch_size == 0:
                logger.info(f"{desc}: {completed}/{len(items)} completed")
        
        return results
    
    async def process_in_chunks(
        self,
        items: List[Any],
        process_fn: Callable[[List[Any]], Awaitable[List[T]]],
        chunk_size: int = None
    ) -> List[T]:
        """分块批量处理"""
        if chunk_size is None:
            chunk_size = self.batch_size
        
        chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
        
        results = []
        for i, chunk in enumerate(chunks):
            try:
                chunk_results = await asyncio.wait_for(
                    process_fn(chunk),
                    timeout=self.timeout
                )
                results.extend(chunk_results)
                logger.info(f"Processed chunk {i + 1}/{len(chunks)}")
            except Exception as e:
                logger.error(f"Error processing chunk {i}: {e}")
                results.extend([None] * len(chunk))
        
        return results


class AsyncRateLimiter:
    """异步速率限制器"""
    
    def __init__(
        self,
        requests_per_second: float = 10.0,
        burst_size: int = 20
    ):
        self.requests_per_second = requests_per_second
        self.burst_size = burst_size
        self._tokens = burst_size
        self._last_update = time.monotonic()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1):
        """获取令牌"""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_update
            self._tokens = min(
                self.burst_size,
                self._tokens + elapsed * self.requests_per_second
            )
            self._last_update = now
            
            if self._tokens < tokens:
                wait_time = (tokens - self._tokens) / self.requests_per_second
                await asyncio.sleep(wait_time)
                self._tokens = 0
            else:
                self._tokens -= tokens


class AsyncCircuitBreaker:
    """异步熔断器"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._state = 'closed'  # closed, open, half_open
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """调用函数（带熔断保护）"""
        async with self._lock:
            if self._state == 'open':
                if time.monotonic() - self._last_failure_time > self.recovery_timeout:
                    self._state = 'half_open'
                else:
                    raise Exception("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs)
            async with self._lock:
                if self._state == 'half_open':
                    self._state = 'closed'
                self._failure_count = 0
            return result
        except self.expected_exception:
            async with self._lock:
                self._failure_count += 1
                self._last_failure_time = time.monotonic()
                if self._failure_count >= self.failure_threshold:
                    self._state = 'open'
            raise


def async_retry(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """异步重试装饰器"""
    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = delay * (backoff ** attempt)
                        logger.warning(f"Retry {attempt + 1}/{max_retries} after {wait_time}s: {e}")
                        await asyncio.sleep(wait_time)
            raise last_exception
        return wrapper
    return decorator


def async_timeout(seconds: float):
    """异步超时装饰器"""
    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
        return wrapper
    return decorator


async def run_in_threadpool(func: Callable[..., T], *args, **kwargs) -> T:
    """在线程池中运行同步函数"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


async def gather_with_concurrency(n: int, *tasks, return_exceptions: bool = False):
    """限制并发数的gather"""
    semaphore = asyncio.Semaphore(n)
    
    async def sem_task(task):
        async with semaphore:
            return await task
    
    return await asyncio.gather(
        *[sem_task(task) for task in tasks],
        return_exceptions=return_exceptions
    )


class AsyncPerformanceMonitor:
    """异步性能监控器"""
    
    def __init__(self):
        self.metrics: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = asyncio.Lock()
    
    async def record(self, name: str, value: float, metadata: Dict[str, Any] = None):
        """记录指标"""
        async with self._lock:
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append({
                'value': value,
                'timestamp': time.time(),
                'metadata': metadata or {}
            })
    
    def record_time(self, name: str):
        """记录时间的上下文管理器"""
        class AsyncTimer:
            def __init__(self, monitor, metric_name):
                self.monitor = monitor
                self.metric_name = metric_name
                self.start_time = None
            
            async def __aenter__(self):
                self.start_time = time.time()
                return self
            
            async def __aexit__(self, *args):
                elapsed = time.time() - self.start_time
                await self.monitor.record(self.metric_name, elapsed)
        
        return AsyncTimer(self, name)
    
    async def get_stats(self, name: str) -> Dict[str, float]:
        """获取统计信息"""
        async with self._lock:
            if name not in self.metrics:
                return {}
            
            values = [m['value'] for m in self.metrics[name]]
            if not values:
                return {}
            
            # 使用内置函数计算统计信息，避免导入 numpy
            count = len(values)
            mean = sum(values) / count
            variance = sum((x - mean) ** 2 for x in values) / count
            std = variance ** 0.5
            
            return {
                'count': count,
                'mean': mean,
                'std': std,
                'min': min(values),
                'max': max(values),
                'total': sum(values)
            }
    
    async def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """获取所有统计信息"""
        async with self._lock:
            result = {}
            for name in self.metrics:
                result[name] = await self.get_stats(name)
            return result
    
    async def clear(self):
        """清除所有指标"""
        async with self._lock:
            self.metrics.clear()


global_async_monitor = AsyncPerformanceMonitor()
