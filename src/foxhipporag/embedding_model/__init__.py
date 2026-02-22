from .Contriever import ContrieverModel
from .base import EmbeddingConfig, BaseEmbeddingModel
from .GritLM import GritLMEmbeddingModel
from .NVEmbedV2 import NVEmbedV2EmbeddingModel
from .OpenAI import OpenAIEmbeddingModel
from .Cohere import CohereEmbeddingModel
from .Transformers import TransformersEmbeddingModel
from .VLLM import VLLMEmbeddingModel

from ..utils.logging_utils import get_logger

__all__ = [
    "ContrieverModel",
    "EmbeddingConfig",
    "BaseEmbeddingModel",
    "GritLMEmbeddingModel",
    "NVEmbedV2EmbeddingModel",
    "OpenAIEmbeddingModel",
    "CohereEmbeddingModel",
    "TransformersEmbeddingModel",
    "VLLMEmbeddingModel",
]

logger = get_logger(__name__)


def _get_embedding_model_class(embedding_model_name: str = "nvidia/NV-Embed-v2"):
    if "GritLM" in embedding_model_name:
        return GritLMEmbeddingModel
    elif "NV-Embed-v2" in embedding_model_name:
        return NVEmbedV2EmbeddingModel
    elif "contriever" in embedding_model_name:
        return ContrieverModel
    elif "text-embedding" in embedding_model_name:
        return OpenAIEmbeddingModel
    elif "cohere" in embedding_model_name:
        return CohereEmbeddingModel
    elif embedding_model_name.startswith("Transformers/"):
        return TransformersEmbeddingModel
    elif embedding_model_name.startswith("VLLM/"):
        return VLLMEmbeddingModel
    # 支持更多OpenAI兼容的嵌入模型
    elif "bge" in embedding_model_name.lower():
        return OpenAIEmbeddingModel
    elif "embedding" in embedding_model_name.lower():
        return OpenAIEmbeddingModel
    # 默认使用OpenAI兼容模式（大多数API都兼容）
    else:
        logger.warning(f"Unknown embedding model name: {embedding_model_name}, using OpenAI compatible mode")
        return OpenAIEmbeddingModel
