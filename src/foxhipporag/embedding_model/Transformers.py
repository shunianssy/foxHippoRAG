from typing import List
import logging

import torch
import numpy as np
from tqdm import tqdm

from .base import BaseEmbeddingModel
from ..utils.config_utils import BaseConfig
from ..prompts.linking import get_query_instruction
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class TransformersEmbeddingModel(BaseEmbeddingModel):
    """
    To select this implementation you can initialise HippoRAG with:
        embedding_model_name starts with "Transformers/"
    
    性能优化版本：
    - 支持动态批处理大小
    - 支持GPU内存优化
    - 支持多GPU并行编码
    """
    def __init__(self, global_config:BaseConfig, embedding_model_name:str) -> None:
        super().__init__(global_config=global_config)

        self.model_id = embedding_model_name[len("Transformers/"):]
        self.embedding_type = 'float'
        
        # 从配置获取批处理大小，支持动态调整
        self.batch_size = getattr(global_config, 'embedding_batch_size', 64)
        self.max_batch_size = getattr(global_config, 'embedding_max_batch_size', 256)
        
        # 检测设备
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        logger.info(f"Initializing embedding model {self.model_id} on {self.device}")
        if self.num_gpus > 1:
            logger.info(f"Multiple GPUs detected: {self.num_gpus}")

        self.model = SentenceTransformer(self.model_id, device=self.device)
        
        # 如果有多个GPU，使用DataParallel
        if self.num_gpus > 1:
            self.model = torch.nn.DataParallel(self.model)
            logger.info(f"Using DataParallel across {self.num_gpus} GPUs")

        self.search_query_instr = set([
            get_query_instruction('query_to_fact'),
            get_query_instruction('query_to_passage')
        ])

    def encode(self, texts: List[str], show_progress_bar: bool = False) -> np.ndarray:
        """
        编码文本为嵌入向量
        
        Args:
            texts: 文本列表
            show_progress_bar: 是否显示进度条
            
        Returns:
            嵌入向量数组
        """
        try:
            # 使用SentenceTransformer的批量编码
            response = self.model.encode(
                texts, 
                batch_size=self.batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=True
            )
        except Exception as err:
            logger.error(f"Encoding error: {err}")
            raise Exception(f"An error occurred: {err}")
        return np.array(response)

    def batch_encode(self, texts: List[str], instruction: str = None, norm: bool = False, **kwargs) -> np.ndarray:
        """
        批量编码文本为嵌入向量，支持指令和归一化
        
        性能优化：
        - 动态调整批处理大小
        - 支持GPU内存优化
        - 显示进度条
        
        Args:
            texts: 文本列表
            instruction: 编码指令（用于查询编码）
            norm: 是否归一化
            **kwargs: 其他参数
            
        Returns:
            嵌入向量数组
        """
        if len(texts) == 0:
            return np.array([])
            
        # 对于小批量，直接编码
        if len(texts) <= self.batch_size:
            result = self.encode(texts, show_progress_bar=False)
            if norm:
                result = result / np.linalg.norm(result, axis=1, keepdims=True)
            return result
        
        # 动态调整批处理大小以优化GPU内存使用
        optimal_batch_size = self._get_optimal_batch_size(len(texts))
        
        logger.info(f"Batch encoding {len(texts)} texts with batch_size={optimal_batch_size}")
        
        # 使用更高效的大批量处理
        results = []
        batch_indexes = list(range(0, len(texts), optimal_batch_size))
        
        for i in tqdm(batch_indexes, desc="Batch Encoding", unit="batch"):
            batch_texts = texts[i:i + optimal_batch_size]
            batch_result = self.encode(batch_texts, show_progress_bar=False)
            results.append(batch_result)
        
        result = np.concatenate(results)
        
        # 归一化
        if norm:
            result = result / np.linalg.norm(result, axis=1, keepdims=True)
            
        return result
    
    def _get_optimal_batch_size(self, num_texts: int) -> int:
        """
        根据文本数量和GPU内存动态计算最优批处理大小
        
        Args:
            num_texts: 文本数量
            
        Returns:
            最优批处理大小
        """
        # 基础批处理大小
        base_batch_size = self.batch_size
        
        # 如果文本数量很大，尝试增大批处理大小
        if num_texts > 10000:
            # 大规模数据，使用更大的批次
            return min(self.max_batch_size, base_batch_size * 2)
        elif num_texts > 1000:
            # 中等规模数据
            return min(base_batch_size * 2, self.max_batch_size)
        else:
            # 小规模数据，使用默认批次
            return base_batch_size
    
    def encode_queries(self, queries: List[str], instruction: str = None) -> np.ndarray:
        """
        专门用于查询编码的方法，支持指令
        
        Args:
            queries: 查询列表
            instruction: 编码指令
            
        Returns:
            查询嵌入向量
        """
        if instruction:
            # 添加指令前缀
            queries = [f"{instruction}{q}" for q in queries]
        
        return self.batch_encode(queries, norm=True)
