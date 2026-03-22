# foxhipporag C扩展

高性能C语言扩展，用于加速foxHippoRAG的核心计算。

## 功能特性

- **余弦相似度计算**: 单向量和批量计算
- **Top-K选择算法**: 快速选择和排序
- **向量归一化**: L2归一化和Min-Max归一化
- **KNN检索**: 高效的最近邻搜索
- **分数融合**: 加权融合和乘法融合

## 性能优势

- 余弦相似度：2-5x 加速
- Top-K选择：3-10x 加速
- KNN检索：2-4x 加速
- 使用OpenMP并行优化

## 编译安装

### 前置要求

- Python 3.10+
- NumPy 1.20+
- C编译器（Windows: MSVC, Linux: GCC, macOS: Clang）

### 编译步骤

```bash
# 方法1：使用提供的构建脚本
python setup_c_extension.py build_ext --inplace

# 方法2：安装为包
pip install -e ".[cext]"
```

### Windows编译

Windows系统需要安装Visual Studio Build Tools，包含MSVC编译器。

```powershell
# 确保已安装Visual Studio Build Tools
# 然后运行
python setup_c_extension.py build_ext --inplace
```

### Linux/macOS编译

```bash
# 确保已安装gcc或clang
# 对于Linux，安装OpenMP支持
sudo apt-get install libomp-dev  # Ubuntu/Debian
sudo yum install libomp-devel     # CentOS/RHEL

# 编译
python setup_c_extension.py build_ext --inplace
```

## 使用方法

### 基本使用

```python
import numpy as np
from foxhipporag.utils.c_utils import (
    cosine_similarity,
    top_k_indices,
    knn_search,
    is_c_extension_available,
)

# 检查C扩展是否可用
print(f"C扩展可用: {is_c_extension_available()}")

# 准备数据
query = np.random.randn(128).astype(np.float32)
matrix = np.random.randn(10000, 128).astype(np.float32)

# 计算余弦相似度
similarities = cosine_similarity(query, matrix)

# Top-K选择
indices, scores = top_k_indices(similarities, k=10)

# KNN检索
indices, scores = knn_search(query, matrix, k=10)
```

### 批量操作

```python
# 批量余弦相似度
queries = np.random.randn(100, 128).astype(np.float32)
keys = np.random.randn(10000, 128).astype(np.float32)
similarity_matrix = cosine_similarity_batch(queries, keys)

# 批量KNN检索
indices, scores = knn_search_batch(queries, keys, k=10)
```

### 归一化

```python
# L2归一化
vector = np.random.randn(128).astype(np.float32)
normalized = l2_normalize(vector)

# 批量L2归一化
matrix = np.random.randn(1000, 128).astype(np.float32)
normalized_matrix = batch_l2_normalize(matrix)

# Min-Max归一化
values = np.random.randn(1000).astype(np.float32)
normalized_values = min_max_normalize(values)
```

### 分数融合

```python
# 加权融合
scores1 = np.random.rand(1000).astype(np.float32)
scores2 = np.random.rand(1000).astype(np.float32)
fused = fuse_scores(scores1, scores2, weight1=0.6, weight2=0.4)

# 乘法融合
fused = multiplicative_fuse(scores1, scores2, alpha=0.5)
```

## 回退机制

如果C扩展不可用，所有函数会自动回退到NumPy实现，确保代码兼容性。

```python
# 强制使用NumPy
result = cosine_similarity(query, matrix, use_c=False)

# 自动选择（优先使用C扩展）
result = cosine_similarity(query, matrix, use_c=None)
```

## 性能测试

```python
from foxhipporag.utils.c_utils import benchmark

# 运行基准测试
results = benchmark(size=10000, dim=512)

for test_name, data in results.items():
    print(f"{test_name}:")
    print(f"  NumPy时间: {data['numpy_time']:.4f}s")
    print(f"  C时间: {data['c_time']:.4f}s")
    print(f"  加速比: {data['speedup']:.2f}x")
```

## 项目结构

```
src/foxhipporag/
├── c_extension/
│   ├── foxhipporag_cext.h          # C头文件
│   ├── foxhipporag_cext.c          # C实现
│   └── foxhipporag_cext_module.c   # Python绑定
└── utils/
    └── c_utils.py                   # Python接口
```

## 技术细节

### OpenMP并行

C扩展使用OpenMP进行并行计算，在多核CPU上可以获得更好的性能。

### SIMD优化

代码中预留了SIMD优化的接口，可以在支持的CPU上获得额外加速。

### 内存管理

- 所有函数都接受NumPy数组作为输入
- 自动处理内存对齐和连续性
- 避免不必要的内存拷贝

## 故障排除

### 编译错误

1. **找不到numpy/arrayobject.h**
   ```bash
   pip install numpy --upgrade
   ```

2. **Windows下找不到编译器**
   - 安装Visual Studio Build Tools
   - 确保选择"Desktop development with C++"

3. **Linux下OpenMP错误**
   ```bash
   sudo apt-get install libomp-dev
   ```

### 运行时错误

1. **ImportError: cannot import name 'foxhipporag_cext'**
   - 确保已成功编译
   - 检查Python路径是否包含编译输出目录

2. **性能没有提升**
   - 确保OpenMP已启用
   - 检查数据规模是否足够大（小数据可能NumPy更快）

## 许可证

GNU Affero General Public License v3.0
