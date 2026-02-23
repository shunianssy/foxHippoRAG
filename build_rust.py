#!/usr/bin/env python
"""
构建脚本：编译 Rust 扩展

使用方法：
    python build_rust.py           # 开发模式构建
    python build_rust.py --release # 发布模式构建
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def build_rust(release: bool = False):
    """构建 Rust 扩展"""
    rust_dir = Path(__file__).parent / "foxhipporag_rust"
    
    if not rust_dir.exists():
        print(f"错误：Rust 项目目录不存在: {rust_dir}")
        sys.exit(1)
    
    os.chdir(rust_dir)
    
    # 使用 maturin 构建
    cmd = [sys.executable, "-m", "maturin", "build"]
    
    if release:
        cmd.append("--release")
    
    print(f"执行命令: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("构建失败:")
        print(result.stderr)
        sys.exit(1)
    
    print("构建成功!")
    print(result.stdout)


def develop():
    """开发模式安装（可编辑）"""
    rust_dir = Path(__file__).parent / "foxhipporag_rust"
    os.chdir(rust_dir)
    
    cmd = [sys.executable, "-m", "maturin", "develop"]
    
    print(f"执行命令: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("开发安装失败:")
        print(result.stderr)
        sys.exit(1)
    
    print("开发安装成功!")
    print(result.stdout)


def main():
    parser = argparse.ArgumentParser(description="构建 foxhipporag_rust Rust 扩展")
    parser.add_argument("--release", action="store_true", help="发布模式构建")
    parser.add_argument("--develop", action="store_true", help="开发模式安装")
    
    args = parser.parse_args()
    
    if args.develop:
        develop()
    else:
        build_rust(args.release)


if __name__ == "__main__":
    main()
