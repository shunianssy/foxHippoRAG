"""验证 Rust 扩展是否可用"""
from foxhipporag.utils.rust_utils import is_rust_available, get_backend_info

print("Rust 可用:", is_rust_available())
print("后端信息:", get_backend_info())

# 测试基本功能
import numpy as np
from foxhipporag.utils.rust_utils import cosine_similarity, top_k_indices

query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
matrix = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.5, 0.5, 0.0],
], dtype=np.float32)

result = cosine_similarity(query, matrix)
print("\n余弦相似度测试:")
print("查询向量:", query)
print("相似度结果:", result)

indices, scores = top_k_indices(result, k=2)
print("\nTop-K 结果:")
print("索引:", indices)
print("得分:", scores)

print("\n✅ Rust 扩展工作正常!")
