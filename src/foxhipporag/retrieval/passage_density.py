"""
段落信息密度评估器模块

该模块实现了段落信息密度评估和证据质量评分功能，
用于解决 HippoRAG 中三重相似性权重过高导致检索证据不足段落的问题。

核心功能：
1. 信息密度评估：评估段落的信息丰富程度
2. 证据质量评分：结合语义相似度和信息密度
3. 动态权重融合：根据检索场景自适应调整权重

性能优化：
- 使用 Numba JIT 加速数值计算
- 批量处理优化
- 缓存机制
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import re
import logging

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

# 尝试导入 Numba 优化
try:
    from ..utils.numba_utils import (
        numba_min_max_normalize,
        is_numba_available
    )
    USE_NUMBA = is_numba_available()
    if USE_NUMBA:
        logger.info("PassageDensityEvaluator 使用 Numba 加速")
except ImportError:
    USE_NUMBA = False
    logger.debug("Numba 不可用，PassageDensityEvaluator 使用标准实现")


@dataclass
class PassageDensityConfig:
    """段落信息密度评估配置"""
    # 信息密度权重配置
    entity_count_weight: float = 0.3  # 实体数量权重
    fact_count_weight: float = 0.3    # 事实数量权重
    text_length_weight: float = 0.2   # 文本长度权重
    content_richness_weight: float = 0.2  # 内容丰富度权重
    
    # 动态权重融合配置
    min_passage_weight: float = 0.1   # 最小段落权重
    max_passage_weight: float = 0.5   # 最大段落权重
    default_passage_weight: float = 0.2  # 默认段落权重
    
    # 证据质量阈值
    low_density_threshold: float = 0.3   # 低密度阈值
    high_density_threshold: float = 0.7  # 高密度阈值
    
    # 混合检索配置
    enable_adaptive_weight: bool = True  # 是否启用自适应权重
    dpr_fallback_threshold: float = 0.4  # DPR 回退阈值


class PassageDensityEvaluator:
    """
    段落信息密度评估器
    
    评估段落的信息密度，用于识别证据充足的段落。
    信息密度高的段落通常包含：
    - 更多的命名实体
    - 更多的关系/事实
    - 更丰富的语义内容
    - 更完整的上下文信息
    """
    
    def __init__(self, config: Optional[PassageDensityConfig] = None):
        """
        初始化段落信息密度评估器
        
        Args:
            config: 段落密度评估配置，如果为 None 则使用默认配置
        """
        self.config = config or PassageDensityConfig()
        
        # 预编译正则表达式以提高性能
        self._entity_patterns = [
            re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'),  # 人名/地名
            re.compile(r'\b\d+(?:\.\d+)?(?:\s*(?:年|月|日|时|分|秒|km|m|kg|元|美元|亿|万))?\b'),  # 数字+单位
            re.compile(r'\b[A-Z]{2,}(?:\s+[A-Z]{2,})*\b'),  # 缩写
        ]
        
        # 停用词集合（用于计算内容丰富度）
        self._stopwords = set([
            '的', '了', '是', '在', '有', '和', '与', '或', '等', '这', '那',
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
            'into', 'through', 'during', 'before', 'after', 'above', 'below',
        ])
        
        logger.info("PassageDensityEvaluator initialized with config: %s", self.config)
    
    def compute_entity_count(self, text: str) -> int:
        """
        计算文本中的实体数量
        
        Args:
            text: 输入文本
            
        Returns:
            实体数量
        """
        count = 0
        for pattern in self._entity_patterns:
            matches = pattern.findall(text)
            count += len(matches)
        return count
    
    def compute_fact_count(self, text: str) -> int:
        """
        估算文本中的事实/关系数量
        
        通过检测动词和关系词来估算事实数量
        
        Args:
            text: 输入文本
            
        Returns:
            估算的事实数量
        """
        # 中文关系词
        cn_relation_patterns = [
            r'是', r'有', r'位于', r'属于', r'包含', r'包括', r'成立于', r'创建于',
            r'发明', r'发现', r'生产', r'制造', r'设计', r'开发', r'建立', r'设立',
        ]
        
        # 英文关系词
        en_relation_patterns = [
            r'\bis\b', r'\bare\b', r'\bwas\b', r'\bwere\b',
            r'\blocated\b', r'\bbelongs\b', r'\bcontains\b', r'\bincludes\b',
            r'\bfounded\b', r'\bcreated\b', r'\binvented\b', r'\bdiscovered\b',
            r'\bproduced\b', r'\bmanufactured\b', r'\bdesigned\b', r'\bdeveloped\b',
        ]
        
        count = 0
        for pattern in cn_relation_patterns + en_relation_patterns:
            count += len(re.findall(pattern, text, re.IGNORECASE))
        
        return count
    
    def compute_text_length_score(self, text: str) -> float:
        """
        计算文本长度得分（归一化）
        
        Args:
            text: 输入文本
            
        Returns:
            归一化的文本长度得分 (0-1)
        """
        # 使用对数函数进行归一化，避免过长的文本得分过高
        length = len(text)
        # 假设理想段落长度为 200-500 字符
        optimal_length = 350
        max_length = 1000
        
        if length <= 0:
            return 0.0
        elif length <= optimal_length:
            return length / optimal_length
        else:
            # 超过最优长度后，得分逐渐降低
            return max(0.5, 1.0 - (length - optimal_length) / (max_length - optimal_length) * 0.5)
    
    def compute_content_richness(self, text: str) -> float:
        """
        计算内容丰富度
        
        基于词汇多样性和非停用词比例
        
        Args:
            text: 输入文本
            
        Returns:
            内容丰富度得分 (0-1)
        """
        # 分词（简单按空格和标点分割）
        words = re.findall(r'\b\w+\b', text.lower())
        
        if len(words) == 0:
            return 0.0
        
        # 计算非停用词比例
        non_stopword_count = sum(1 for w in words if w not in self._stopwords)
        non_stopword_ratio = non_stopword_count / len(words)
        
        # 计算词汇多样性（唯一词/总词数）
        unique_words = set(words)
        vocabulary_diversity = len(unique_words) / len(words) if len(words) > 0 else 0
        
        # 综合得分
        richness = 0.6 * non_stopword_ratio + 0.4 * vocabulary_diversity
        
        return min(1.0, richness)
    
    def evaluate_passage_density(self, text: str) -> Dict[str, float]:
        """
        评估段落的信息密度
        
        Args:
            text: 段落文本
            
        Returns:
            包含各项密度指标的字典
        """
        entity_count = self.compute_entity_count(text)
        fact_count = self.compute_fact_count(text)
        text_length_score = self.compute_text_length_score(text)
        content_richness = self.compute_content_richness(text)
        
        # 归一化实体和事实数量（使用对数函数）
        normalized_entity_score = min(1.0, np.log1p(entity_count) / 3.0)  # log(x+1)/3
        normalized_fact_score = min(1.0, np.log1p(fact_count) / 2.5)
        
        # 计算综合信息密度得分
        density_score = (
            self.config.entity_count_weight * normalized_entity_score +
            self.config.fact_count_weight * normalized_fact_score +
            self.config.text_length_weight * text_length_score +
            self.config.content_richness_weight * content_richness
        )
        
        return {
            'entity_count': entity_count,
            'fact_count': fact_count,
            'text_length_score': text_length_score,
            'content_richness': content_richness,
            'normalized_entity_score': normalized_entity_score,
            'normalized_fact_score': normalized_fact_score,
            'density_score': density_score
        }
    
    def batch_evaluate_density(self, texts: List[str]) -> np.ndarray:
        """
        批量评估段落信息密度
        
        性能优化版本：
        - 使用 Numba 加速数值计算
        - 批量处理减少函数调用开销
        
        Args:
            texts: 段落文本列表
            
        Returns:
            密度得分数组
        """
        n = len(texts)
        
        # 预分配数组
        entity_counts = np.zeros(n, dtype=np.float32)
        fact_counts = np.zeros(n, dtype=np.float32)
        text_length_scores = np.zeros(n, dtype=np.float32)
        content_richness_scores = np.zeros(n, dtype=np.float32)
        
        # 批量计算各项指标
        for i, text in enumerate(texts):
            entity_counts[i] = self.compute_entity_count(text)
            fact_counts[i] = self.compute_fact_count(text)
            text_length_scores[i] = self.compute_text_length_score(text)
            content_richness_scores[i] = self.compute_content_richness(text)
        
        # 归一化实体和事实数量（使用对数函数）
        # 使用 Numba 优化或 NumPy 实现
        normalized_entity_scores = np.minimum(1.0, np.log1p(entity_counts) / 3.0)
        normalized_fact_scores = np.minimum(1.0, np.log1p(fact_counts) / 2.5)
        
        # 计算综合信息密度得分
        density_scores = (
            self.config.entity_count_weight * normalized_entity_scores +
            self.config.fact_count_weight * normalized_fact_scores +
            self.config.text_length_weight * text_length_scores +
            self.config.content_richness_weight * content_richness_scores
        )
        
        return density_scores.astype(np.float32)


class EvidenceQualityScorer:
    """
    证据质量评分器
    
    结合语义相似度和信息密度，评估段落的证据质量。
    用于在检索时优先选择证据充足的段落。
    """
    
    def __init__(self, density_evaluator: Optional[PassageDensityEvaluator] = None,
                 config: Optional[PassageDensityConfig] = None):
        """
        初始化证据质量评分器
        
        Args:
            density_evaluator: 段落密度评估器
            config: 配置
        """
        self.density_evaluator = density_evaluator or PassageDensityEvaluator(config)
        self.config = config or PassageDensityConfig()
        
        # 缓存已计算的密度得分
        self._density_cache: Dict[str, float] = {}
        
        logger.info("EvidenceQualityScorer initialized")
    
    def compute_evidence_score(self, 
                               semantic_score: float,
                               density_score: float,
                               query_type: str = 'general') -> float:
        """
        计算证据质量得分
        
        结合语义相似度和信息密度，根据查询类型调整权重
        
        Args:
            semantic_score: 语义相似度得分 (0-1)
            density_score: 信息密度得分 (0-1)
            query_type: 查询类型 ('general', 'factual', 'exploratory')
            
        Returns:
            证据质量得分 (0-1)
        """
        # 根据查询类型调整权重
        if query_type == 'factual':
            # 事实型查询：更重视语义相似度
            semantic_weight = 0.7
            density_weight = 0.3
        elif query_type == 'exploratory':
            # 探索型查询：更重视信息密度
            semantic_weight = 0.4
            density_weight = 0.6
        else:
            # 一般查询：平衡权重
            semantic_weight = 0.5
            density_weight = 0.5
        
        # 使用乘法融合，确保两者都高时得分才高
        multiplicative_score = semantic_score * density_score
        
        # 使用加权平均
        weighted_score = semantic_weight * semantic_score + density_weight * density_score
        
        # 最终得分是两种方式的加权平均
        final_score = 0.3 * multiplicative_score + 0.7 * weighted_score
        
        return final_score
    
    def rank_by_evidence_quality(self,
                                 passage_ids: List[str],
                                 passage_texts: List[str],
                                 semantic_scores: np.ndarray,
                                 query_type: str = 'general') -> Tuple[np.ndarray, np.ndarray]:
        """
        根据证据质量对段落进行排序
        
        Args:
            passage_ids: 段落ID列表
            passage_texts: 段落文本列表
            semantic_scores: 语义相似度得分
            query_type: 查询类型
            
        Returns:
            排序后的段落索引和得分
        """
        n = len(passage_texts)
        evidence_scores = np.zeros(n)
        
        for i, (pid, text) in enumerate(zip(passage_ids, passage_texts)):
            # 检查缓存
            if pid in self._density_cache:
                density_score = self._density_cache[pid]
            else:
                density_result = self.density_evaluator.evaluate_passage_density(text)
                density_score = density_result['density_score']
                self._density_cache[pid] = density_score
            
            # 计算证据质量得分
            evidence_scores[i] = self.compute_evidence_score(
                semantic_scores[i], 
                density_score,
                query_type
            )
        
        # 排序
        sorted_indices = np.argsort(evidence_scores)[::-1]
        sorted_scores = evidence_scores[sorted_indices]
        
        return sorted_indices, sorted_scores
    
    def clear_cache(self):
        """清空密度缓存"""
        self._density_cache.clear()
        logger.debug("Density cache cleared")


class AdaptiveWeightFusion:
    """
    自适应权重融合器
    
    根据检索场景动态调整 DPR 和图谱检索的权重，
    解决三重相似性权重过高的问题。
    """
    
    def __init__(self, config: Optional[PassageDensityConfig] = None):
        """
        初始化自适应权重融合器
        
        Args:
            config: 配置
        """
        self.config = config or PassageDensityConfig()
        
        logger.info("AdaptiveWeightFusion initialized")
    
    def compute_adaptive_passage_weight(self,
                                        avg_fact_score: float,
                                        avg_density_score: float,
                                        query_complexity: float = 0.5) -> float:
        """
        计算自适应段落权重
        
        根据事实得分、密度得分和查询复杂度动态调整段落权重
        
        Args:
            avg_fact_score: 平均事实得分
            avg_density_score: 平均密度得分
            query_complexity: 查询复杂度 (0-1)
            
        Returns:
            自适应段落权重
        """
        if not self.config.enable_adaptive_weight:
            return self.config.default_passage_weight
        
        # 基础权重
        base_weight = self.config.default_passage_weight
        
        # 根据事实得分调整：事实得分高时，降低段落权重（更依赖图谱）
        # 事实得分低时，提高段落权重（更依赖 DPR）
        fact_adjustment = (0.5 - avg_fact_score) * 0.3
        
        # 根据密度得分调整：密度低时，提高段落权重
        # 因为低密度段落通过图谱传播效果不好
        density_adjustment = (0.5 - avg_density_score) * 0.2
        
        # 根据查询复杂度调整：复杂查询更依赖图谱，简单查询更依赖 DPR
        complexity_adjustment = (0.5 - query_complexity) * 0.1
        
        # 计算最终权重
        final_weight = base_weight + fact_adjustment + density_adjustment + complexity_adjustment
        
        # 限制在配置范围内
        final_weight = max(self.config.min_passage_weight, 
                          min(self.config.max_passage_weight, final_weight))
        
        return final_weight
    
    def fuse_retrieval_results(self,
                               dpr_scores: np.ndarray,
                               graph_scores: np.ndarray,
                               passage_weight: float = None) -> np.ndarray:
        """
        融合 DPR 和图谱检索结果
        
        Args:
            dpr_scores: DPR 检索得分
            graph_scores: 图谱检索得分
            passage_weight: 段落权重（如果为 None，使用默认值）
            
        Returns:
            融合后的得分
        """
        if passage_weight is None:
            passage_weight = self.config.default_passage_weight
        
        graph_weight = 1.0 - passage_weight
        
        # 归一化得分
        dpr_normalized = self._normalize_scores(dpr_scores)
        graph_normalized = self._normalize_scores(graph_scores)
        
        # 加权融合
        fused_scores = passage_weight * dpr_normalized + graph_weight * graph_normalized
        
        return fused_scores
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        归一化得分（Min-Max 归一化）
        
        使用 Numba 优化版本（如果可用）
        
        Args:
            scores: 原始得分
            
        Returns:
            归一化后的得分
        """
        if USE_NUMBA:
            try:
                return numba_min_max_normalize(scores)
            except Exception as e:
                logger.debug(f"Numba 归一化失败，回退到 NumPy: {e}")
        
        min_val = np.min(scores)
        max_val = np.max(scores)
        
        if max_val - min_val < 1e-10:
            return np.ones_like(scores)
        
        return (scores - min_val) / (max_val - min_val)
    
    def should_fallback_to_dpr(self,
                               top_facts_count: int,
                               avg_fact_score: float,
                               graph_coverage: float) -> bool:
        """
        判断是否应该回退到纯 DPR 检索
        
        当图谱检索效果不佳时，回退到 DPR
        
        Args:
            top_facts_count: Top-K 事实数量
            avg_fact_score: 平均事实得分
            graph_coverage: 图谱覆盖率
            
        Returns:
            是否回退到 DPR
        """
        # 条件1：几乎没有找到相关事实
        if top_facts_count == 0:
            logger.debug("Fallback to DPR: no relevant facts found")
            return True
        
        # 条件2：事实得分过低
        if avg_fact_score < self.config.dpr_fallback_threshold:
            logger.debug(f"Fallback to DPR: low fact score ({avg_fact_score:.3f})")
            return True
        
        # 条件3：图谱覆盖率过低
        if graph_coverage < 0.1:
            logger.debug(f"Fallback to DPR: low graph coverage ({graph_coverage:.3f})")
            return True
        
        return False


def create_density_evaluator(config: Optional[PassageDensityConfig] = None) -> PassageDensityEvaluator:
    """
    创建段落密度评估器的工厂函数
    
    Args:
        config: 配置
        
    Returns:
        PassageDensityEvaluator 实例
    """
    return PassageDensityEvaluator(config)


def create_evidence_scorer(config: Optional[PassageDensityConfig] = None) -> EvidenceQualityScorer:
    """
    创建证据质量评分器的工厂函数
    
    Args:
        config: 配置
        
    Returns:
        EvidenceQualityScorer 实例
    """
    return EvidenceQualityScorer(config=config)


def create_weight_fusion(config: Optional[PassageDensityConfig] = None) -> AdaptiveWeightFusion:
    """
    创建自适应权重融合器的工厂函数
    
    Args:
        config: 配置
        
    Returns:
        AdaptiveWeightFusion 实例
    """
    return AdaptiveWeightFusion(config)
