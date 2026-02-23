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
        texts = [t.replace("\n", " ") for t in texts]
        texts = [t if t != '' else ' ' for t in texts]
        
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(input=texts, model=self.embedding_model_name)
                results = np.array([v.embedding for v in response.data])
                # 记录实际的嵌入维度
                self._embedding_dim = results.shape[1] if len(results.shape) > 1 else 1536
                return results
            except Exception as e:
                logger.warning(f"Encoding attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"All {max_retries} encoding attempts failed")
                    raise
                import time
                time.sleep(1 * (attempt + 1))
        
        return np.array([])
    
    @property
    def embedding_dim(self) -> int:
        """获取嵌入向量维度"""
        if not hasattr(self, '_embedding_dim'):
            # 根据模型名称推断维度
            if 'text-embedding-3-small' in self.embedding_model_name:
                self._embedding_dim = 1536
            elif 'text-embedding-3-large' in self.embedding_model_name:
                self._embedding_dim = 3072
            elif 'text-embedding-ada-002' in self.embedding_model_name:
                self._embedding_dim = 1536
            else:
                self._embedding_dim = 1536  # 默认维度
        return self._embedding_dim

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
                    batch_result = self.encode(batch)
                    results.append(batch_result)
                except Exception as e:
                    logger.error(f"Encoding batch failed: {e}")
                    # 用零向量填充失败的批次（使用正确的维度）
                    dim = self.embedding_dim
                    results.append(np.zeros((len(batch), dim)))
                pbar.update(len(batch))
            pbar.close()
            results = np.concatenate(results)
        else:
            # 大批量，并行处理
            logger.info(f"Parallel encoding {len(texts)} texts with {self.max_workers} workers")
            
            batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
            
            results = [None] * len(batches)
            first_batch_dim = None  # 记录第一个成功批次的维度
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_idx = {
                    executor.submit(self.encode, batch): idx 
                    for idx, batch in enumerate(batches)
                }
                
                for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="Parallel Encoding"):
                    idx = future_to_idx[future]
                    try:
                        batch_result = future.result(timeout=60)
                        # 检查维度一致性
                        if first_batch_dim is None:
                            first_batch_dim = batch_result.shape[1] if len(batch_result.shape) > 1 else batch_result.shape[0]
                            self._embedding_dim = first_batch_dim
                        elif batch_result.shape[1] != first_batch_dim:
                            logger.warning(f"Batch {idx} has different dimension {batch_result.shape[1]} vs expected {first_batch_dim}, padding/truncating")
                            # 调整维度
                            if batch_result.shape[1] < first_batch_dim:
                                # 填充零
                                padding = np.zeros((batch_result.shape[0], first_batch_dim - batch_result.shape[1]))
                                batch_result = np.concatenate([batch_result, padding], axis=1)
                            else:
                                # 截断
                                batch_result = batch_result[:, :first_batch_dim]
                        results[idx] = batch_result
                    except Exception as e:
                        logger.error(f"Parallel encoding batch {idx} failed: {e}")
                        # 用零向量填充失败的批次（使用正确的维度）
                        dim = self.embedding_dim
                        results[idx] = np.zeros((len(batches[idx]), dim))
            
            results = np.concatenate(results)

        if isinstance(results, torch.Tensor):
            results = results.cpu()
            results = results.numpy()
        if self.embedding_config.norm:
            results = (results.T / np.linalg.norm(results, axis=1)).T

        return results
