"""
分析 token_gate_matrix 的训练结果
查看不同 token 的门控值有什么规律
"""

import torch
import numpy as np
from transformers import AutoTokenizer
import os 

# ============ 配置区域 ============
# 修改为你的 checkpoint 路径
CHECKPOINT_PATH = "./test0116.2.5.0/Qwen2.5-1.5B-Instruct-gsm8k-group4-lora32-lr0.01-init-2-rmin0.981-temp0.5/checkpoint-934"
MODEL_NAME = "/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct"  # 或 "Qwen/Qwen2.5-1.5B-Instruct"
INIT_VALUE = -2.0  # 初始化值（改为 -2.0 了）
# =================================


def load_gate_matrix(checkpoint_path):
    """加载 token_gate_matrix 权重"""
    # 尝试不同的文件名
    possible_files = [
        "adapter_model.bin",
        "adapter_model.safetensors",
        "pytorch_model.bin",
        "model.safetensors",
    ]
    
    for filename in possible_files:
        filepath = os.path.join(checkpoint_path, filename)
        if os.path.exists(filepath):
            print(f"Loading file: {filepath}")
            if filename.endswith(".safetensors"):
                from safetensors.torch import load_file
                state_dict = load_file(filepath)
            else:
                state_dict = torch.load(filepath, map_location="cpu")
            break
    else:
        raise FileNotFoundError(f"Model file not found in {checkpoint_path}")
    
    # 查找 token_gate_matrix
    # 注意：PEFT 会创建两个版本：
    #   - original_module.weight (frozen, 不训练)
    #   - modules_to_save.default.weight (真正训练的)
    # 我们需要读取 modules_to_save.default.weight！
    
    gate_weight = None
    gate_keys = []
    for key, value in state_dict.items():
        if "token_gate_matrix" in key and "weight" in key:
            gate_keys.append((key, value))
            print(f"Found key: {key}, shape: {value.shape}")
    
    if not gate_keys:
        print("Available keys:")
        for key in state_dict.keys():
            print(f"  {key}")
        raise KeyError("token_gate_matrix not found")
    
    # 优先选择 modules_to_save.default.weight（真正训练的权重）
    for key, value in gate_keys:
        if "modules_to_save" in key:
            print(f"OK: Using trained weights: {key}")
            gate_weight = value
            break
    
    # 如果没有 modules_to_save，则使用找到的第一个（可能是非 PEFT 保存）
    if gate_weight is None:
        key, value = gate_keys[0]
        print(f"WARN: Using weights (no modules_to_save): {key}")
        gate_weight = value
    
    return gate_weight


def analyze_gate_matrix(gate_weight, tokenizer, init_value=-3.0):
    """分析门控矩阵"""
    vocab_size, hidden_size = gate_weight.shape
    print(f"\n{'='*60}")
    print(f"Matrix shape: {gate_weight.shape}")
    print(f"vocab_size: {vocab_size}, hidden_size: {hidden_size}")
    print(f"{'='*60}")
    
    # 1. 基本统计
    print(f"\n[1. Overall stats]")
    print(f"  Mean: {gate_weight.mean().item():.6f}")
    print(f"  Std: {gate_weight.std().item():.6f}")
    print(f"  Min: {gate_weight.min().item():.6f}")
    print(f"  Max: {gate_weight.max().item():.6f}")
    
    # 2. 计算每行（每个 token）的统计
    row_mean = gate_weight.mean(dim=1)  # 每个 token 的门控均值
    row_std = gate_weight.std(dim=1)    # 每个 token 的门控标准差
    row_sigmoid_mean = torch.sigmoid(gate_weight).mean(dim=1)  # sigmoid 后均值
    
    # 3. 计算与初始值的偏离程度
    delta_from_init = (gate_weight - init_value).abs().mean(dim=1)  # 每个 token 偏离初始值的程度
    
    # 4. 找出变化最大的 token
    print(f"\n[2. Top 20 most changed tokens] (largest deviation from init {init_value})")
    top_changed = delta_from_init.topk(100)
    print(f"{'Token ID':>10} | {'Token':>20} | {'Deviation':>10} | {'Mean':>10} | {'SigmoidMean':>12}")
    print("-" * 70)
    for idx, delta in zip(top_changed.indices, top_changed.values):
        token_id = idx.item()
        try:
            token_str = tokenizer.decode([token_id]).replace('\n', '\\n')
        except:
            token_str = "<UNK>"
        mean_val = row_mean[token_id].item()
        sigmoid_val = row_sigmoid_mean[token_id].item()
        print(f"{token_id:>10} | {token_str:>20} | {delta.item():>10.6f} | {mean_val:>10.6f} | {sigmoid_val:>12.6f}")
    
    # 5. 找出门控最开放的 token（sigmoid 最大）
    print(f"\n[3. Top 20 most open tokens] (largest sigmoid mean)")
    top_open = row_sigmoid_mean.topk(20)
    print(f"{'Token ID':>10} | {'Token':>20} | {'SigmoidMean':>12} | {'RawMean':>10}")
    print("-" * 60)
    for idx, val in zip(top_open.indices, top_open.values):
        token_id = idx.item()
        try:
            token_str = tokenizer.decode([token_id]).replace('\n', '\\n')
        except:
            token_str = "<UNK>"
        mean_val = row_mean[token_id].item()
        print(f"{token_id:>10} | {token_str:>20} | {val.item():>12.6f} | {mean_val:>10.6f}")
    
    # 6. 找出门控最关闭的 token（sigmoid 最小）
    print(f"\n[4. Top 20 most closed tokens] (smallest sigmoid mean)")
    bottom_closed = row_sigmoid_mean.topk(20, largest=False)
    print(f"{'Token ID':>10} | {'Token':>20} | {'SigmoidMean':>12} | {'RawMean':>10}")
    print("-" * 60)
    for idx, val in zip(bottom_closed.indices, bottom_closed.values):
        token_id = idx.item()
        try:
            token_str = tokenizer.decode([token_id]).replace('\n', '\\n')
        except:
            token_str = "<UNK>"
        mean_val = row_mean[token_id].item()
        print(f"{token_id:>10} | {token_str:>20} | {val.item():>12.6f} | {mean_val:>10.6f}")
    
    # 7. 分析特定类型的 token
    print(f"\n[5. Specific token type analysis]")
    
    # 数字 token
    digit_tokens = []
    for i in range(10):
        tokens = tokenizer.encode(str(i), add_special_tokens=False)
        digit_tokens.extend(tokens)
    digit_tokens = list(set(digit_tokens))
    if digit_tokens:
        digit_sigmoid = row_sigmoid_mean[digit_tokens].mean().item()
        digit_mean = row_mean[digit_tokens].mean().item()
        print(f"  Digits (0-9): sigmoid_mean={digit_sigmoid:.6f}, raw_mean={digit_mean:.6f}")
    
    # 运算符 token
    operators = ['+', '-', '*', '/', '=', '(', ')', '<', '>', '.']
    op_tokens = []
    for op in operators:
        tokens = tokenizer.encode(op, add_special_tokens=False)
        op_tokens.extend(tokens)
    op_tokens = list(set(op_tokens))
    if op_tokens:
        op_sigmoid = row_sigmoid_mean[op_tokens].mean().item()
        op_mean = row_mean[op_tokens].mean().item()
        print(f"  Operators (+-*/= etc.): sigmoid_mean={op_sigmoid:.6f}, raw_mean={op_mean:.6f}")
    
    # 常见数学词汇
    math_words = ['answer', 'total', 'sum', 'result', 'equal', 'plus', 'minus', 'times']
    math_tokens = []
    for word in math_words:
        tokens = tokenizer.encode(word, add_special_tokens=False)
        math_tokens.extend(tokens)
    math_tokens = list(set(math_tokens))
    if math_tokens:
        math_sigmoid = row_sigmoid_mean[math_tokens].mean().item()
        math_mean = row_mean[math_tokens].mean().item()
        print(f"  Math words: sigmoid_mean={math_sigmoid:.6f}, raw_mean={math_mean:.6f}")
    
    # 8. 统计有多少 token 发生了显著变化
    threshold = 0.01  # 偏离阈值
    changed_count = (delta_from_init > threshold).sum().item()
    print(f"\n[6. Change statistics]")
    print(f"  Tokens with deviation > {threshold}: {changed_count} / {vocab_size} ({100*changed_count/vocab_size:.2f}%)")
    
    # 9. 输出变化最大的 token 的每个维度的数值和 sigmoid 值
    most_changed_idx = delta_from_init.argmax().item()
    try:
        most_changed_token_str = tokenizer.decode([most_changed_idx]).replace('\n', '\\n')
    except:
        most_changed_token_str = "<UNK>"
    most_changed_raw = gate_weight[most_changed_idx]
    most_changed_sigmoid = torch.sigmoid(most_changed_raw)
    print(f"\n[8. Most changed token - all dimensions]")
    print(f"  Token: '{most_changed_token_str}' (id={most_changed_idx}), deviation={delta_from_init[most_changed_idx].item():.6f}")
    print(f"  Total dims: {hidden_size}")
    print(f"  {'Dim':>6} | {'Raw':>12} | {'Sigmoid':>12}")
    print(f"  {'-'*36}")
    for dim_idx in range(hidden_size):
        raw_val = most_changed_raw[dim_idx].item()
        sig_val = most_changed_sigmoid[dim_idx].item()
        print(f"  {dim_idx:>6} | {raw_val:>12.6f} | {sig_val:>12.6f}")
    
    # 10. 打印指定 token 的详细门控向量
    print(f"\n[9. Detailed gate vectors for selected tokens (first 20 dims)]")
    sample_tokens = ['0', '1', '2', '+', '-', '=', 'the', 'answer', '\n']
    for token_str in sample_tokens:
        tokens = tokenizer.encode(token_str, add_special_tokens=False)
        if tokens:
            token_id = tokens[0]
            gate_vec = gate_weight[token_id, :20]
            sigmoid_vec = torch.sigmoid(gate_vec)
            print(f"  '{token_str}' (id={token_id}):")
            print(f"    raw (first 10 dims): {[f'{v:.4f}' for v in gate_vec[:10].tolist()]}")
            print(f"    sigmoid (first 10 dims): {[f'{v:.4f}' for v in sigmoid_vec[:10].tolist()]}")
    
    return {
        'row_mean': row_mean,
        'row_std': row_std,
        'row_sigmoid_mean': row_sigmoid_mean,
        'delta_from_init': delta_from_init,
    }


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    print("Loading token_gate_matrix...")
    gate_weight = load_gate_matrix(CHECKPOINT_PATH)
    
    print("Analyzing gate matrix...")
    results = analyze_gate_matrix(gate_weight, tokenizer, init_value=INIT_VALUE)
    
    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
