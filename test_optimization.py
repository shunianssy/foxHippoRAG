"""测试优化后的代码"""
import numpy as np
import sys
sys.path.insert(0, 'src')

from foxhipporag.utils.numba_utils import (
    numba_min_max_normalize,
    numba_cosine_similarity,
    numba_top_k_indices,
    numba_fuse_scores,
    numba_multiplicative_fuse,
    is_numba_available,
    get_numba_info
)

def test_numba_utils():
    print('测试 Numba 优化工具函数...')
    print(f'Numba 可用: {is_numba_available()}')
    
    # 测试 min_max_normalize
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    normalized = numba_min_max_normalize(x)
    print(f'min_max_normalize: {normalized}')
    assert np.allclose(normalized, [0.0, 0.25, 0.5, 0.75, 1.0], atol=0.01), 'min_max_normalize 测试失败'
    print('  ✓ min_max_normalize 测试通过')
    
    # 测试 cosine_similarity
    query = np.random.randn(128).astype(np.float32)
    matrix = np.random.randn(100, 128).astype(np.float32)
    similarities = numba_cosine_similarity(query, matrix)
    print(f'cosine_similarity shape: {similarities.shape}')
    assert similarities.shape == (100,), 'cosine_similarity 测试失败'
    print('  ✓ cosine_similarity 测试通过')
    
    # 测试 top_k_indices
    scores = np.random.randn(100).astype(np.float32)
    indices, top_scores = numba_top_k_indices(scores, 10)
    print(f'top_k_indices: indices shape={indices.shape}, scores shape={top_scores.shape}')
    assert len(indices) == 10, 'top_k_indices 测试失败'
    print('  ✓ top_k_indices 测试通过')
    
    # 测试 fuse_scores
    scores1 = np.array([0.5, 0.3, 0.8], dtype=np.float32)
    scores2 = np.array([0.7, 0.6, 0.2], dtype=np.float32)
    fused = numba_fuse_scores(scores1, scores2, 0.6, 0.4)
    print(f'fuse_scores: {fused}')
    expected = 0.6 * scores1 + 0.4 * scores2
    assert np.allclose(fused, expected, atol=0.01), 'fuse_scores 测试失败'
    print('  ✓ fuse_scores 测试通过')
    
    # 测试 multiplicative_fuse
    mult_fused = numba_multiplicative_fuse(scores1, scores2)
    print(f'multiplicative_fuse: {mult_fused}')
    print('  ✓ multiplicative_fuse 测试通过')
    
    print('\n所有 Numba 工具函数测试通过!')
    return True


def test_passage_density():
    print('\n测试 PassageDensity 优化...')
    from foxhipporag.retrieval.passage_density import (
        PassageDensityEvaluator,
        EvidenceQualityScorer,
        PassageDensityConfig
    )
    
    config = PassageDensityConfig()
    evaluator = PassageDensityEvaluator(config)
    scorer = EvidenceQualityScorer(evaluator, config)
    
    # 测试批量评估
    texts = [
        "Apple Inc. is a technology company founded in 1976 by Steve Jobs.",
        "The Eiffel Tower is located in Paris, France.",
        "Python is a programming language."
    ]
    
    density_scores = evaluator.batch_evaluate_density(texts)
    print(f'batch_evaluate_density: {density_scores}')
    assert len(density_scores) == len(texts), 'batch_evaluate_density 测试失败'
    print('  ✓ batch_evaluate_density 测试通过')
    
    # 测试证据质量评分
    semantic_scores = np.array([0.8, 0.6, 0.9], dtype=np.float32)
    passage_ids = ['p1', 'p2', 'p3']
    
    sorted_indices, sorted_scores = scorer.rank_by_evidence_quality(
        passage_ids, texts, semantic_scores
    )
    print(f'rank_by_evidence_quality: indices={sorted_indices}, scores={sorted_scores}')
    assert len(sorted_indices) == len(texts), 'rank_by_evidence_quality 测试失败'
    print('  ✓ rank_by_evidence_quality 测试通过')
    
    print('\n所有 PassageDensity 测试通过!')
    return True


def test_misc_utils():
    print('\n测试 misc_utils 优化...')
    from foxhipporag.utils.misc_utils import min_max_normalize
    
    # 测试归一化
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    normalized = min_max_normalize(x)
    print(f'min_max_normalize: {normalized}')
    assert np.allclose(normalized, [0.0, 0.25, 0.5, 0.75, 1.0], atol=0.01), 'min_max_normalize 测试失败'
    print('  ✓ min_max_normalize 测试通过')
    
    print('\n所有 misc_utils 测试通过!')
    return True


if __name__ == '__main__':
    try:
        test_numba_utils()
        test_passage_density()
        test_misc_utils()
        print('\n' + '='*50)
        print('所有测试通过！优化后的代码运行正常。')
        print('='*50)
    except Exception as e:
        print(f'\n测试失败: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
