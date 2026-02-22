"""
检索增强模块

该模块包含检索相关的增强功能：
- 段落信息密度评估
- 证据质量评分
- 自适应权重融合
"""

from .passage_density import (
    PassageDensityConfig,
    PassageDensityEvaluator,
    EvidenceQualityScorer,
    AdaptiveWeightFusion,
    create_density_evaluator,
    create_evidence_scorer,
    create_weight_fusion,
)

__all__ = [
    'PassageDensityConfig',
    'PassageDensityEvaluator',
    'EvidenceQualityScorer',
    'AdaptiveWeightFusion',
    'create_density_evaluator',
    'create_evidence_scorer',
    'create_weight_fusion',
]
