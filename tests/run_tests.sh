#!/bin/bash
# VeraRAG 测试脚本

set -e

echo "=========================================="
echo "  VeraRAG 测试套件"
echo "=========================================="

# 运行单元测试
echo ""
echo "[1/3] 运行单元测试..."
python -m pytest tests/ -v --tb=short

# 运行演示
echo ""
echo "[2/3] 运行功能演示..."
python demo.py

# 代码检查
echo ""
echo "[3/3] 代码检查..."
if command -v flake8 &> /dev/null; then
    echo "运行 flake8..."
    flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics
else
    echo "flake8 未安装，跳过代码检查"
fi

echo ""
echo "=========================================="
echo "  所有测试完成！"
echo "=========================================="
