"""
异步OpenIE模块

提供异步的信息抽取功能，包括：
1. 异步NER（命名实体识别）
2. 异步三元组抽取
3. 异步批量处理
4. 智能重试和错误处理

性能优化：
- 使用asyncio实现真正的非阻塞I/O
- 并发控制和速率限制
- 智能缓存减少API调用
- 批量处理优化
"""

import asyncio
import json
import re
import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from tqdm.asyncio import tqdm_asyncio

from ..utils.async_utils import (
    AsyncOpenAIClient, 
    AsyncBatchProcessor,
    AsyncRateLimiter,
    AsyncCircuitBreaker,
    async_retry,
    async_timeout,
    global_async_monitor
)
from ..utils.misc_utils import TripleRawOutput, NerRawOutput
from ..utils.llm_utils import fix_broken_generated_json, filter_invalid_triples
from ..prompts import PromptTemplateManager

logger = logging.getLogger(__name__)


@dataclass
class AsyncOpenIEConfig:
    """异步OpenIE配置"""
    max_concurrent_ner: int = 20
    max_concurrent_triples: int = 20
    ner_timeout: float = 60.0
    triple_timeout: float = 60.0
    max_retries: int = 3
    cache_enabled: bool = True
    rate_limit_rps: float = 50.0


class AsyncOpenIE:
    """异步OpenIE处理器"""
    
    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
        config: AsyncOpenIEConfig = None,
        cache_dir: str = "outputs/async_cache"
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.config = config or AsyncOpenIEConfig()
        
        self.client = AsyncOpenAIClient(
            api_key=api_key,
            base_url=base_url,
            cache_dir=cache_dir
        )
        
        self.prompt_template_manager = PromptTemplateManager(
            role_mapping={"system": "system", "user": "user", "assistant": "assistant"}
        )
        
        self.ner_processor = AsyncBatchProcessor(
            max_concurrent=self.config.max_concurrent_ner,
            timeout=self.config.ner_timeout
        )
        
        self.triple_processor = AsyncBatchProcessor(
            max_concurrent=self.config.max_concurrent_triples,
            timeout=self.config.triple_timeout
        )
        
        self.rate_limiter = AsyncRateLimiter(
            requests_per_second=self.config.rate_limit_rps,
            burst_size=int(self.config.rate_limit_rps * 2)
        )
        
        self.circuit_breaker = AsyncCircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30.0
        )
    
    def _extract_ner_from_response(self, response: str) -> List[str]:
        """从响应中提取命名实体"""
        pattern = r'\{[^{}]*"named_entities"\s*:\s*\[[^\]]*\][^{}]*\}'
        match = re.search(pattern, response, re.DOTALL)
        if match is None:
            return []
        try:
            return eval(match.group())["named_entities"]
        except Exception:
            return []
    
    def _extract_triples_from_response(self, response: str) -> List[Tuple]:
        """从响应中提取三元组"""
        pattern = r'\{[^{}]*"triples"\s*:\s*\[[^\]]*\][^{}]*\}'
        match = re.search(pattern, response, re.DOTALL)
        if match is None:
            return []
        try:
            return eval(match.group())["triples"]
        except Exception:
            return []
    
    @async_retry(max_retries=3, delay=1.0, backoff=2.0)
    @async_timeout(60.0)
    async def ner(self, chunk_key: str, passage: str) -> NerRawOutput:
        """异步NER"""
        async with global_async_monitor.record_time("async_ner"):
            await self.rate_limiter.acquire()
            
            ner_input_message = self.prompt_template_manager.render(
                name='ner', 
                passage=passage
            )
            
            try:
                raw_response, metadata, cache_hit = await self.circuit_breaker.call(
                    self.client.chat_completion,
                    messages=ner_input_message,
                    model=self.model
                )
                
                metadata['cache_hit'] = cache_hit
                
                if metadata.get('finish_reason') == 'length':
                    real_response = fix_broken_generated_json(raw_response)
                else:
                    real_response = raw_response
                
                extracted_entities = self._extract_ner_from_response(real_response)
                unique_entities = list(dict.fromkeys(extracted_entities))
                
                return NerRawOutput(
                    chunk_id=chunk_key,
                    response=raw_response,
                    unique_entities=unique_entities,
                    metadata=metadata
                )
                
            except Exception as e:
                logger.warning(f"NER failed for chunk {chunk_key}: {e}")
                return NerRawOutput(
                    chunk_id=chunk_key,
                    response="",
                    unique_entities=[],
                    metadata={'error': str(e)}
                )
    
    @async_retry(max_retries=3, delay=1.0, backoff=2.0)
    @async_timeout(60.0)
    async def triple_extraction(
        self, 
        chunk_key: str, 
        passage: str, 
        named_entities: List[str]
    ) -> TripleRawOutput:
        """异步三元组抽取"""
        async with global_async_monitor.record_time("async_triple_extraction"):
            await self.rate_limiter.acquire()
            
            messages = self.prompt_template_manager.render(
                name='triple_extraction',
                passage=passage,
                named_entity_json=json.dumps({"named_entities": named_entities})
            )
            
            try:
                raw_response, metadata, cache_hit = await self.circuit_breaker.call(
                    self.client.chat_completion,
                    messages=messages,
                    model=self.model
                )
                
                metadata['cache_hit'] = cache_hit
                
                if metadata.get('finish_reason') == 'length':
                    real_response = fix_broken_generated_json(raw_response)
                else:
                    real_response = raw_response
                
                extracted_triples = self._extract_triples_from_response(real_response)
                triplets = filter_invalid_triples(triples=extracted_triples)
                
                return TripleRawOutput(
                    chunk_id=chunk_key,
                    response=raw_response,
                    metadata=metadata,
                    triples=triplets
                )
                
            except Exception as e:
                logger.warning(f"Triple extraction failed for chunk {chunk_key}: {e}")
                return TripleRawOutput(
                    chunk_id=chunk_key,
                    response="",
                    metadata={'error': str(e)},
                    triples=[]
                )
    
    async def openie(self, chunk_key: str, passage: str) -> Dict[str, Any]:
        """异步OpenIE（NER + 三元组抽取）"""
        ner_output = await self.ner(chunk_key=chunk_key, passage=passage)
        triple_output = await self.triple_extraction(
            chunk_key=chunk_key,
            passage=passage,
            named_entities=ner_output.unique_entities
        )
        return {"ner": ner_output, "triplets": triple_output}
    
    async def batch_ner(
        self, 
        chunk_passages: Dict[str, str],
        show_progress: bool = True
    ) -> Dict[str, NerRawOutput]:
        """异步批量NER"""
        async def process_item(item: Tuple[str, str]) -> NerRawOutput:
            chunk_key, passage = item
            return await self.ner(chunk_key, passage)
        
        items = list(chunk_passages.items())
        
        if show_progress:
            results = []
            for coro in tqdm_asyncio.as_completed(
                [process_item(item) for item in items],
                total=len(items),
                desc="Async NER"
            ):
                result = await coro
                results.append(result)
        else:
            results = await self.ner_processor.process_batch(
                items,
                process_item,
                desc="Async NER"
            )
        
        return {res.chunk_id: res for res in results if res is not None}
    
    async def batch_triple_extraction(
        self,
        chunk_passages: Dict[str, str],
        ner_results: Dict[str, NerRawOutput],
        show_progress: bool = True
    ) -> Dict[str, TripleRawOutput]:
        """异步批量三元组抽取"""
        async def process_item(item: Tuple[str, NerRawOutput]) -> TripleRawOutput:
            chunk_key, ner_result = item
            return await self.triple_extraction(
                chunk_key=chunk_key,
                passage=chunk_passages[chunk_key],
                named_entities=ner_result.unique_entities
            )
        
        items = list(ner_results.items())
        
        if show_progress:
            results = []
            for coro in tqdm_asyncio.as_completed(
                [process_item(item) for item in items],
                total=len(items),
                desc="Async Triple Extraction"
            ):
                result = await coro
                results.append(result)
        else:
            results = await self.triple_processor.process_batch(
                items,
                process_item,
                desc="Async Triple Extraction"
            )
        
        return {res.chunk_id: res for res in results if res is not None}
    
    async def batch_openie(
        self,
        chunks: Dict[str, Dict[str, Any]],
        show_progress: bool = True
    ) -> Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
        """
        异步批量OpenIE
        
        Args:
            chunks: chunk_id -> {content: str, ...} 的映射
            show_progress: 是否显示进度条
            
        Returns:
            (ner_results_dict, triple_results_dict)
        """
        chunk_passages = {k: v["content"] for k, v in chunks.items()}
        
        logger.info(f"Starting async batch OpenIE for {len(chunks)} chunks")
        
        ner_results = await self.batch_ner(chunk_passages, show_progress)
        
        triple_results = await self.batch_triple_extraction(
            chunk_passages, ner_results, show_progress
        )
        
        logger.info(f"Async batch OpenIE completed: {len(ner_results)} NER, {len(triple_results)} triples")
        
        return ner_results, triple_results
    
    async def close(self):
        """关闭客户端"""
        await self.client.close()
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return await global_async_monitor.get_all_stats()


class AsyncOpenIEWithFallback:
    """带回退机制的异步OpenIE"""
    
    def __init__(
        self,
        async_openie: AsyncOpenIE,
        sync_openie: Any
    ):
        self.async_openie = async_openie
        self.sync_openie = sync_openie
    
    async def ner(self, chunk_key: str, passage: str) -> NerRawOutput:
        """异步NER（带回退）"""
        try:
            return await self.async_openie.ner(chunk_key, passage)
        except Exception as e:
            logger.warning(f"Async NER failed, falling back to sync: {e}")
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self.sync_openie.ner,
                chunk_key,
                passage
            )
    
    async def triple_extraction(
        self,
        chunk_key: str,
        passage: str,
        named_entities: List[str]
    ) -> TripleRawOutput:
        """异步三元组抽取（带回退）"""
        try:
            return await self.async_openie.triple_extraction(
                chunk_key, passage, named_entities
            )
        except Exception as e:
            logger.warning(f"Async triple extraction failed, falling back to sync: {e}")
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self.sync_openie.triple_extraction,
                chunk_key,
                passage,
                named_entities
            )
    
    async def batch_openie(
        self,
        chunks: Dict[str, Dict[str, Any]],
        show_progress: bool = True
    ) -> Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
        """异步批量OpenIE（带回退）"""
        try:
            return await self.async_openie.batch_openie(chunks, show_progress)
        except Exception as e:
            logger.warning(f"Async batch OpenIE failed, falling back to sync: {e}")
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self.sync_openie.batch_openie,
                chunks
            )
    
    async def close(self):
        """关闭客户端"""
        await self.async_openie.close()
