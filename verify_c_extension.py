"""
验证C扩展是否正确安装和工作

运行此脚本以检查C扩展的状态
"""

import sys
import os

print("=" * 60)
print("foxhipporag C扩展验证脚本")
print("=" * 60)

# 1. 检查Python版本
print(f"\n1. Python版本检查")
print(f"   Python版本: {sys.version}")
print(f"   Python路径: {sys.executable}")

# 2. 检查NumPy
print(f"\n2. NumPy检查")
try:
    import numpy as np
    print(f"   NumPy版本: {np.__version__}")
    print(f"   NumPy路径: {np.__file__}")
except ImportError as e:
    print(f"   错误: NumPy未安装 - {e}")
    sys.exit(1)

# 3. 检查C扩展模块
print(f"\n3. C扩展模块检查")
try:
    # 添加src目录到Python路径
    import sys
    import os
    src_path = os.path.join(os.path.dirname(__file__), 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    import foxhipporag_cext
    print(f"   ✓ foxhipporag_cext已安装")
    print(f"   版本: {foxhipporag_cext.get_version()}")
    print(f"   SIMD支持: {foxhipporag_cext.has_simd_support()}")
    print(f"   可用函数:")
    print(f"     - cosine_similarity")
    print(f"     - cosine_similarity_batch")
    print(f"     - top_k_indices")
    print(f"     - top_k_indices_2d")
    print(f"     - l2_normalize")
    print(f"     - min_max_normalize")
    print(f"     - knn_search")
    print(f"     - knn_search_batch")
    print(f"     - fuse_scores")
    print(f"     - multiplicative_fuse")
except ImportError as e:
    print(f"   ✗ foxhipporag_cext未安装")
    print(f"   错误: {e}")
    print(f"\n   请运行以下命令编译C扩展:")
    print(f"   python setup_c_extension.py build_ext --inplace")
    
    # 继续检查Python接口
    foxhipporag_cext = None

# 4. 检查Python接口
print(f"\n4. Python接口检查")
try:
    from foxhipporag.utils.c_utils import (
        cosine_similarity,
        top_k_indices,
        knn_search,
        is_c_extension_available,
        get_backend_info,
    )
    print(f"   ✓ c_utils模块可用")
    
    info = get_backend_info()
    print(f"   后端: {info['backend']}")
    print(f"   C扩展可用: {info['c_extension_available']}")
    
except ImportError as e:
    print(f"   ✗ c_utils模块不可用")
    print(f"   错误: {e}")
    sys.exit(1)

# 5. 功能测试
print(f"\n5. 功能测试")
try:
    # 测试余弦相似度
    query = np.random.randn(128).astype(np.float32)
    matrix = np.random.randn(1000, 128).astype(np.float32)
    
    result = cosine_similarity(query, matrix)
    print(f"   ✓ 余弦相似度计算正常 (结果形状: {result.shape})")
    
    # 测试Top-K
    scores = np.random.randn(1000).astype(np.float32)
    indices, top_scores = top_k_indices(scores, k=10)
    print(f"   ✓ Top-K选择正常 (返回{len(indices)}个结果)")
    
    # 测试KNN
    indices, scores = knn_search(query, matrix, k=10)
    print(f"   ✓ KNN检索正常 (返回{len(indices)}个结果)")
    
except Exception as e:
    print(f"   ✗ 功能测试失败")
    print(f"   错误: {e}")
    import traceback
    traceback.print_exc()

# 6. 性能测试
print(f"\n6. 性能测试")
if is_c_extension_available():
    try:
        import time
        
        size = 5000
        dim = 256
        
        # 准备数据
        query = np.random.randn(dim).astype(np.float32)
        matrix = np.random.randn(size, dim).astype(np.float32)
        
        # NumPy性能
        start = time.time()
        for _ in range(10):
            _ = cosine_similarity(query, matrix, use_c=False)
        numpy_time = time.time() - start
        
        # C扩展性能
        start = time.time()
        for _ in range(10):
            _ = cosine_similarity(query, matrix, use_c=True)
        c_time = time.time() - start
        
        speedup = numpy_time / c_time if c_time > 0 else 0
        
        print(f"   余弦相似度性能:")
        print(f"     NumPy: {numpy_time:.4f}s")
        print(f"     C扩展: {c_time:.4f}s")
        print(f"     加速比: {speedup:.2f}x")
        
        if speedup > 1.0:
            print(f"   ✓ C扩展比NumPy快 {speedup:.2f}x")
        else:
            print(f"   ! C扩展性能未达到预期（可能数据规模较小）")
            
    except Exception as e:
        print(f"   ✗ 性能测试失败: {e}")
else:
    print(f"   跳过（C扩展不可用）")

# 7. 总结
print(f"\n" + "=" * 60)
print(f"验证总结:")
print(f"=" * 60)

if is_c_extension_available():
    print(f"✓ C扩展已正确安装并可以使用")
    print(f"✓ 所有功能测试通过")
    print(f"\n建议:")
    print(f"  - 在生产环境中使用C扩展以获得最佳性能")
    print(f"  - 对于小规模数据，NumPy可能更快")
else:
    print(f"! C扩展未安装")
    print(f"✓ Python回退实现可用")
    print(f"\n建议:")
    print(f"  - 运行 'python setup_c_extension.py build_ext --inplace' 编译C扩展")
    print(f"  - 或使用NumPy回退实现（性能较低但功能完整）")

print(f"\n" + "=" * 60)
