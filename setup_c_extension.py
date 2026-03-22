"""
foxhipporag C扩展构建脚本

编译方式：
    python setup_c_extension.py build_ext --inplace
"""

import os
import sys
import numpy as np
from setuptools import setup, Extension

# 检测操作系统
is_windows = sys.platform == 'win32'
is_linux = sys.platform.startswith('linux')
is_macos = sys.platform == 'darwin'

# 编译器标志
if is_windows:
    # Windows (MSVC)
    extra_compile_args = ['/O2', '/openmp']
    extra_link_args = []
elif is_macos:
    # macOS (clang)
    extra_compile_args = ['-O3', '-ffast-math']
    extra_link_args = []
    # macOS 默认不支持 OpenMP，需要额外配置
else:
    # Linux (gcc)
    extra_compile_args = ['-O3', '-ffast-math', '-fopenmp']
    extra_link_args = ['-fopenmp']

# 定义扩展模块
extensions = [
    Extension(
        'foxhipporag_cext',
        sources=[
            'src/foxhipporag/c_extension/foxhipporag_cext.c',
            'src/foxhipporag/c_extension/foxhipporag_cext_module.c',
        ],
        include_dirs=[
            np.get_include(),
            'src/foxhipporag/c_extension',
        ],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        define_macros=[
            ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION'),
        ],
    )
]

setup(
    name='foxhipporag_cext',
    version='1.0.0',
    description='foxhipporag C extension for high-performance numerical computations',
    ext_modules=extensions,
    install_requires=['numpy>=1.20.0'],
    python_requires='>=3.10',
)
