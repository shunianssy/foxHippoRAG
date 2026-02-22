"""
简单测试脚本：验证段落密度评估功能
"""
import sys
sys.path.insert(0, 'src')

from foxhipporag.retrieval.passage_density import (
    PassageDensityEvaluator,
    EvidenceQualityScorer,
    AdaptiveWeightFusion,
    PassageDensityConfig,
)

def test_density_evaluator():
    """测试段落密度评估器"""
    print("=" * 60)
    print("测试段落密度评估器")
    print("=" * 60)
    
    evaluator = PassageDensityEvaluator()
    
    # 高密度段落（证据充足）
    high_density_text = """
    北京大学创建于1898年，是中国近代第一所国立综合性大学。
    学校位于北京市海淀区，占地面积约274公顷，拥有教职工约7000人。
    北京大学设有理学部、信息与工程科学部、人文学部等6个学部，
    开设130多个本科专业，涵盖文、理、工、医等多个学科领域。
    """
    
    # 低密度段落（仅列出实体名称）
    low_density_text = "北京大学、清华大学、复旦大学、上海交通大学。"
    
    print("\n高密度段落测试:")
    result1 = evaluator.evaluate_passage_density(high_density_text)
    print(f"  实体数量: {result1['entity_count']}")
    print(f"  事实数量: {result1['fact_count']}")
    print(f"  文本长度得分: {result1['text_length_score']:.3f}")
    print(f"  内容丰富度: {result1['content_richness']:.3f}")
    print(f"  综合密度得分: {result1['density_score']:.3f}")
    
    print("\n低密度段落测试:")
    result2 = evaluator.evaluate_passage_density(low_density_text)
    print(f"  实体数量: {result2['entity_count']}")
    print(f"  事实数量: {result2['fact_count']}")
    print(f"  文本长度得分: {result2['text_length_score']:.3f}")
    print(f"  内容丰富度: {result2['content_richness']:.3f}")
    print(f"  综合密度得分: {result2['density_score']:.3f}")
    
    print(f"\n验证: 高密度得分 ({result1['density_score']:.3f}) > 低密度得分 ({result2['density_score']:.3f}): {result1['density_score'] > result2['density_score']}")
    
    return result1['density_score'] > result2['density_score']


def test_evidence_scorer():
    """测试证据质量评分器"""
    print("\n" + "=" * 60)
    print("测试证据质量评分器")
    print("=" * 60)
    
    scorer = EvidenceQualityScorer()
    
    # 测试不同组合
    print("\n证据质量得分测试:")
    
    # 高语义 + 高密度
    score1 = scorer.compute_evidence_score(0.9, 0.8, 'general')
    print(f"  高语义(0.9) + 高密度(0.8): {score1:.3f}")
    
    # 高语义 + 低密度
    score2 = scorer.compute_evidence_score(0.9, 0.2, 'general')
    print(f"  高语义(0.9) + 低密度(0.2): {score2:.3f}")
    
    # 低语义 + 高密度
    score3 = scorer.compute_evidence_score(0.3, 0.8, 'general')
    print(f"  低语义(0.3) + 高密度(0.8): {score3:.3f}")
    
    print(f"\n验证: 高语义+高密度得分最高: {score1 > score2 and score1 > score3}")
    
    return score1 > score2 and score1 > score3


def test_adaptive_weight():
    """测试自适应权重融合"""
    print("\n" + "=" * 60)
    print("测试自适应权重融合")
    print("=" * 60)
    
    fusion = AdaptiveWeightFusion()
    
    print("\n自适应段落权重测试:")
    
    # 高事实得分 + 高密度
    weight1 = fusion.compute_adaptive_passage_weight(0.8, 0.8, 0.5)
    print(f"  高事实(0.8) + 高密度(0.8): {weight1:.3f}")
    
    # 低事实得分 + 低密度
    weight2 = fusion.compute_adaptive_passage_weight(0.3, 0.3, 0.5)
    print(f"  低事实(0.3) + 低密度(0.3): {weight2:.3f}")
    
    print(f"\n验证: 低质量时权重更高(更依赖DPR): {weight2 > weight1}")
    
    print("\nDPR回退测试:")
    # 应该回退的情况
    fallback1 = fusion.should_fallback_to_dpr(0, 0.0, 0.0)
    print(f"  无事实: {fallback1} (应该为 True)")
    
    fallback2 = fusion.should_fallback_to_dpr(5, 0.2, 0.5)
    print(f"  低事实得分: {fallback2} (应该为 True)")
    
    # 不应该回退的情况
    fallback3 = fusion.should_fallback_to_dpr(5, 0.7, 0.8)
    print(f"  正常情况: {fallback3} (应该为 False)")
    
    return fallback1 and fallback2 and not fallback3


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("HippoRAG 证据质量评估功能测试")
    print("解决三重相似性权重过高导致检索证据不足段落的问题")
    print("=" * 60)
    
    results = []
    
    # 运行测试
    results.append(("密度评估器", test_density_evaluator()))
    results.append(("证据评分器", test_evidence_scorer()))
    results.append(("自适应权重", test_adaptive_weight()))
    
    # 输出结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("所有测试通过！功能正常工作。")
    else:
        print("部分测试失败，请检查实现。")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
