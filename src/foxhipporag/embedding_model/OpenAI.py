from copy import deepcopy
from typing import List, Optional
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from tqdm import tqdm
from openai import OpenAI
from openai import AzureOpenAI

from ..utils.config_utils import BaseConfig
from ..utils.logging_utils import get_logger
from .base import BaseEmbeddingModel, EmbeddingConfig

logger = get_logger(__name__)

class OpenAIEmbeddingModel(BaseEmbeddingModel):
    """
    OpenAI嵌入模型实现
    
    性能优化版本：
    - 支持并行批量编码
    - 支持自动重试
    - 支持错误处理
    """

    def __init__(self, global_config: Optional[BaseConfig] = None, embedding_model_name: Optional[str] = None) -> None:
        super().__init__(global_config=global_config)

        if embedding_model_name is not None:
            self.embedding_model_name = embedding_model_name
            logger.debug(
                f"Overriding {self.__class__.__name__}'s embedding_model_name with: {self.embedding_model_name}")

        self._init_embedding_config()

        # 并行工作线程数
        self.max_workers = getattr(global_config, 'embedding_parallel_workers', 8) if global_config else 8

        # Initializing the embedding model
        logger.debug(
            f"Initializing {self.__class__.__name__}'s embedding model with params: {self.embedding_config.model_init_params}")

        if self.global_config.azure_embedding_endpoint is None:
            # 使用embedding_api_key，如果没有则使用环境变量OPENAI_API_KEY
            api_key = self.global_config.embedding_api_key or os.getenv("OPENAI_API_KEY")
            self.client = OpenAI(
                api_key=api_key,
                base_url=self.global_config.embedding_base_url
            )
        else:
            self.client = AzureOpenAI(api_version=self.global_config.azure_embedding_endpoint.split('api-version=')[1],
                                      azure_endpoint=self.global_config.azure_embedding_endpoint)


    def _init_embedding_config(self) -> None:
        """
        Extract embedding model-specific parameters to init the EmbeddingConfig.

        Returns:
            None
        """

        config_dict = {
            "embedding_model_name": self.embedding_model_name,
            "norm": self.global_config.embedding_return_as_normalized,
            # "max_seq_length": self.global_config.embedding_max_seq_len,
            "model_init_params": {
                # "model_name_or_path": self.embedding_model_name2mode_name_or_path[self.embedding_model_name],
                "pretrained_model_name_or_path": self.embedding_model_name,
                "trust_remote_code": True,
                # "torch_dtype": "auto",
                'device_map': "auto",  # added this line to use multiple GPUs
                # **kwargs
            },
            "encode_params": {
                "max_length": self.global_config.embedding_max_seq_len,  # 32768 from official example,
                "instruction": "",
                "batch_size": self.global_config.embedding_batch_size,
                "num_workers": 32
            },
        }

        self.embedding_config = EmbeddingConfig.from_dict(config_dict=config_dict)
        logger.debug(f"Init {self.__class__.__name__}'s embedding_config: {self.embedding_config}")

    def encode(self, texts: List[str], max_retries: int = 3, instruction: str = "") -> np.ndarray:
        """
        编码文本为嵌入向量，支持自动重试
        
        Args:
            texts: 文本列表
            max_retries: 最大重试次数
            instruction: 编码指令（OpenAI 官方 API 不支持 instruction，此参数仅为接口一致性）
            
        Returns:
            嵌入向量数组
            
        Note:
            OpenAI embedding API 不支持 instruction 前缀。
            根据 OpenAI 官方文档，直接传入原始文本即可。
            参考: https://platform.openai.com/docs/guides/embeddings
        """
        # OpenAI embedding API 不支持 instruction，直接使用原始文本
        # 这与其他模型（如 NVEmbedV2、GritLM）的行为不同
        texts = [t.replace("\n", " ") for t in texts]
        texts = [t if t != '' else ' ' for t in texts]
        
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(input=texts, model=self.embedding_model_name)
                results = np.array([v.embedding for v in response.data])
                return results
            except Exception as e:
                logger.warning(f"Encoding attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"All {max_retries} encoding attempts failed")
                    raise
                # 等待后重试
                import time
                time.sleep(1 * (attempt + 1))
        
        return np.array([])

    def batch_encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """
        批量编码文本为嵌入向量
        
        性能优化：
        - 使用线程池并行处理
        - 支持自动重试
        - 显示进度条
        
        Args:
            texts: 文本列表
            **kwargs: 其他参数
            
        Returns:
            嵌入向量数组
        """
        if isinstance(texts, str): 
            texts = [texts]
            
        if len(texts) == 0:
            return np.array([])

        params = deepcopy(self.embedding_config.encode_params)
        if kwargs: 
            params.update(kwargs)

        # OpenAI embedding API 不支持 instruction 前缀
        # 忽略 instruction 参数，直接使用原始文本
        # 这与 NVEmbedV2、GritLM 等模型的行为不同
        instruction = params.pop("instruction", "")
        if instruction:
            logger.debug(f"OpenAI embedding API does not support instruction prefix, ignoring: {instruction[:50]}...")

        logger.debug(f"Calling {self.__class__.__name__} with {len(texts)} texts")

        batch_size = params.pop("batch_size", 16)

        if len(texts) <= batch_size:
            results = self.encode(texts)
        elif len(texts) <= batch_size * 4:
            # 小批量，串行处理
            pbar = tqdm(total=len(texts), desc="Batch Encoding")
            results = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                try:
                    results.append(self.encode(batch))
                except Exception as e:
                    logger.error(f"Encoding batch failed: {e}")
                    # 用零向量填充失败的批次
                    results.append(np.zeros((len(batch), 1536)))  # OpenAI默认维度
                pbar.update(len(batch))
            pbar.close()
            results = np.concatenate(results)
        else:
            # 大批量，并行处理
            logger.info(f"Parallel encoding {len(texts)} texts with {self.max_workers} workers")
            
            # 创建批次
            batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
            
            results = [None] * len(batches)
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_idx = {
                    executor.submit(self.encode, batch): idx 
                    for idx, batch in enumerate(batches)
                }
                
                for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="Parallel Encoding"):
                    idx = future_to_idx[future]
                    try:
                        results[idx] = future.result(timeout=60)
                    except Exception as e:
                        logger.error(f"Parallel encoding batch {idx} failed: {e}")
                        # 用零向量填充失败的批次
                        results[idx] = np.zeros((len(batches[idx]), 1536))
            
            results = np.concatenate(results)

        if isinstance(results, torch.Tensor):
            results = results.cpu()
            results = results.numpy()
        if self.embedding_config.norm:
            results = (results.T / np.linalg.norm(results, axis=1)).T

        return results
