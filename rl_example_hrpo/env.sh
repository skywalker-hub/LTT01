#!/bin/bash
# ============================================
# HRPO 项目统一缓存路径设置
# 将 HuggingFace 模型和数据集缓存到 autodl-tmp
# ============================================

# 设置 HuggingFace 缓存路径
export HF_HOME=/root/autodl-tmp/huggingface
export HF_DATASETS_CACHE=/root/autodl-tmp/huggingface/datasets
export TRANSFORMERS_CACHE=/root/autodl-tmp/huggingface/models

# 自动创建目录，避免路径不存在报错
mkdir -p $HF_HOME $HF_DATASETS_CACHE $TRANSFORMERS_CACHE

echo ">>> HuggingFace 模型和数据缓存路径已设置到: $HF_HOME"
echo ">>> 数据集缓存: $HF_DATASETS_CACHE"
echo ">>> 模型缓存: $TRANSFORMERS_CACHE"
