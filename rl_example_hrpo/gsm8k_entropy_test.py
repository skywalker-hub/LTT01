import unsloth
from unsloth import FastLanguageModel

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime
from transformers import GenerationConfig

from utils import *


# ====== 在此处修改测试问题 ======
QUESTION = "Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glasses. How much does he need to pay for them?"
# ================================


def compute_entropy(logits: torch.Tensor) -> float:
    """
    计算单步 logits 的信息熵 H(p) = -sum(p * log(p))，单位: nats。
    logits: shape (vocab_size,) 或 (1, vocab_size)
    """
    logits = logits.float()  # 确保精度
    if logits.dim() == 2:
        logits = logits.squeeze(0)
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-12)  # 避免 log(0)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy.item()


def run_entropy_test(
    model_path: str,
    adapter_path: str,
    temperature: float,
    is_inference: bool,
    question: str = QUESTION,
):
    # ---- 1. 加载模型 ----
    print("=" * 60)
    print("加载模型...")
    print(f"  基础模型: {model_path}")
    print(f"  Adapter:  {adapter_path}")
    print(f"  Temperature: {temperature}")
    print(f"  Greedy (is_inference): {is_inference}")
    print("=" * 60)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=1024,
        load_in_4bit=False,
        fast_inference=False,
    )
    model.answer_start = ANSWER_START
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model.load_adapter(adapter_path)
    model = FastLanguageModel.for_inference(model)

    # ---- 1.5 从 checkpoint 文件直接加载 token_gate_matrix ----
    import os as _os
    gate_weight = None
    for filename in ["adapter_model.safetensors", "adapter_model.bin"]:
        filepath = _os.path.join(adapter_path, filename)
        if _os.path.exists(filepath):
            if filename.endswith(".safetensors"):
                from safetensors.torch import load_file
                state_dict = load_file(filepath)
            else:
                state_dict = torch.load(filepath, map_location="cpu")
            for key, value in state_dict.items():
                if "token_gate_matrix" in key and "weight" in key:
                    gate_weight = value
                    print(f"从 {filename} 加载 token_gate_matrix: {key}, shape={gate_weight.shape}")
                    break
            del state_dict
            break
    if gate_weight is None:
        raise RuntimeError("未在 checkpoint 中找到 token_gate_matrix 权重")
    row_sigmoid_mean = torch.sigmoid(gate_weight).mean(dim=1)  # (vocab_size,)

    # ---- 2. 构造 Prompt ----
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question.strip()},
    ]
    formatted_prompt = tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True,
    )

    prompt_inputs = tokenizer(
        [formatted_prompt],
        return_tensors="pt",
        padding=True,
        padding_side="left",
        add_special_tokens=False,
    )
    prompt_ids = prompt_inputs["input_ids"].to(model.device)
    prompt_mask = prompt_inputs["attention_mask"].to(model.device)
    prompt_length = prompt_ids.size(1)

    # ---- 3. 生成回复，同时收集每步 logits ----
    # output_scores=True 让 generate() 在每步前向传播时记录 logits 并通过返回值返回
    print("\n正在生成回复...")
    with torch.no_grad():
        outputs = model.generate(
            prompt_ids,
            attention_mask=prompt_mask,
            generation_config=GenerationConfig(
                do_sample=True,
                temperature=temperature,
                max_new_tokens=512,
                output_scores=True,
                return_dict_in_generate=True,
            ),
            processing_class=tokenizer,
            is_inference=is_inference,
            return_thinking_embeds=True,
        )

    # ---- 4. 解码文本 ----
    generated_ids = outputs.sequences[0][prompt_length:]
    response_text = tokenizer.decode(generated_ids)
    response_text = response_text.split(
        tokenizer.special_tokens_map["eos_token"]
    )[0]

    extracted = extract_from_response(response_text)
    generated_answer = process_gsm8k_answer(extracted)

    print("\n" + "=" * 60)
    print("【问题】")
    print(question)
    print("\n【模型回复】")
    print(response_text)
    print(f"\n【提取的答案】{generated_answer}")
    print("=" * 60)

    # ---- 5. 从 scores 逐步计算信息熵 ----
    # outputs.scores 是一个 tuple，每个元素 shape (1, vocab_size)，对应每步的 logits
    # 注意：这些 logits 已经是 temperature 缩放前的原始值，generate 内部会再做 temperature
    # 因此这里手动除以 temperature 以反映实际采样时的概率分布
    scores = outputs.scores
    num_steps = len(scores)
    entropies = []
    gate_values = []
    tokens_text = []
    # hidden_ratio 向量统计 (per-step, per-dimension)
    hr_mean_values = []   # 每步 hidden_ratio 向量的均值
    hr_std_values = []    # 每步 hidden_ratio 向量的标准差
    hr_min_values = []    # 每步 hidden_ratio 向量的最小值
    hr_max_values = []    # 每步 hidden_ratio 向量的最大值

    # a_t_vectors shape: (batch, num_gen_steps, hidden_size) — 不含 prefill 首步
    # hidden_ratio = sqrt(1 - a_t²) 逐维度计算
    has_a_t = hasattr(outputs, "a_t_vectors") and outputs.a_t_vectors is not None
    if has_a_t:
        a_t_gen = outputs.a_t_vectors[0].cpu().float()  # (num_gen_steps, hidden_size)
        hr_vectors = torch.sqrt(1 - a_t_gen ** 2)       # (num_gen_steps, hidden_size)
        print(f"已获取 a_t 完整向量，shape: {a_t_gen.shape} → hidden_ratio 向量 shape: {hr_vectors.shape}")
    else:
        hr_vectors = None

    for step_idx in range(num_steps):
        logits = scores[step_idx] / temperature  # 应用 temperature
        entropy = compute_entropy(logits)
        entropies.append(entropy)

        token_id = generated_ids[step_idx].item()
        token_str = tokenizer.decode([token_id])
        tokens_text.append(token_str)
        gate_values.append(row_sigmoid_mean[token_id].item())

        if hr_vectors is not None and step_idx < hr_vectors.shape[0]:
            hr_vec = hr_vectors[step_idx]  # (hidden_size,)
            hr_mean_values.append(hr_vec.mean().item())
            hr_std_values.append(hr_vec.std().item())
            hr_min_values.append(hr_vec.min().item())
            hr_max_values.append(hr_vec.max().item())
        else:
            hr_mean_values.append(float("nan"))
            hr_std_values.append(float("nan"))
            hr_min_values.append(float("nan"))
            hr_max_values.append(float("nan"))

    # 打印所有步骤的熵、门控值和 hidden_ratio 向量统计
    print(f"\n共生成 {num_steps} 个 token")
    print("-" * 115)
    print(f"{'Step':>5}  {'Entropy':>10}  {'GateSigm':>10}  {'HR_mean':>9}  {'HR_std':>9}  {'HR_min':>9}  {'HR_max':>9}  Token")
    print("-" * 115)
    for step_idx in range(num_steps):
        token_repr = repr(tokens_text[step_idx])
        if not np.isnan(hr_mean_values[step_idx]):
            hr_str = f"{hr_mean_values[step_idx]:>9.5f}  {hr_std_values[step_idx]:>9.5f}  {hr_min_values[step_idx]:>9.5f}  {hr_max_values[step_idx]:>9.5f}"
        else:
            hr_str = f"{'N/A':>9}  {'N/A':>9}  {'N/A':>9}  {'N/A':>9}"
        print(f"{step_idx + 1:>5}  {entropies[step_idx]:>10.4f}  {gate_values[step_idx]:>10.6f}  {hr_str}  {token_repr}")
    print("-" * 115)

    # 统计摘要
    ent_array = np.array(entropies)
    gate_array = np.array(gate_values)
    hr_mean_array = np.array(hr_mean_values)
    print(f"\n信息熵统计:")
    print(f"  平均值: {ent_array.mean():.4f}")
    print(f"  标准差: {ent_array.std():.4f}")
    print(f"  最小值: {ent_array.min():.4f} (step {ent_array.argmin() + 1})")
    print(f"  最大值: {ent_array.max():.4f} (step {ent_array.argmax() + 1})")
    print(f"\nToken Gate Sigmoid 统计:")
    print(f"  平均值: {gate_array.mean():.6f}")
    print(f"  最小值: {gate_array.min():.6f} (step {gate_array.argmin() + 1})")
    print(f"  最大值: {gate_array.max():.6f} (step {gate_array.argmax() + 1})")
    if has_a_t:
        valid_mask = ~np.isnan(hr_mean_array)
        valid_hr_mean = hr_mean_array[valid_mask]
        if len(valid_hr_mean) > 0:
            print(f"\nHidden Ratio 向量统计 (= sqrt(1 - a_t²), per dimension):")
            print(f"  各步 HR_mean 的均值: {valid_hr_mean.mean():.6f}")
            print(f"  各步 HR_mean 的标准差: {valid_hr_mean.std():.6f}")
            print(f"  HR_mean 最小步: {valid_hr_mean.min():.6f} (step {np.nanargmin(hr_mean_array) + 1})")
            print(f"  HR_mean 最大步: {valid_hr_mean.max():.6f} (step {np.nanargmax(hr_mean_array) + 1})")
            all_hr_flat = hr_vectors[valid_mask[:hr_vectors.shape[0]]].numpy()
            print(f"  全维度全步骤统计: mean={all_hr_flat.mean():.6f}, std={all_hr_flat.std():.6f}, "
                  f"min={all_hr_flat.min():.6f}, max={all_hr_flat.max():.6f}")

    # 打印熵最高的 Top-20 步骤
    top_k = min(20, num_steps)
    top_indices = np.argsort(ent_array)[::-1][:top_k]
    print(f"\n熵最高的 Top-{top_k} 步骤:")
    print("-" * 105)
    print(f"{'Rank':>4}  {'Step':>5}  {'Entropy':>10}  {'GateSigm':>10}  {'HR_mean':>9}  {'HR_std':>9}  Token")
    print("-" * 105)
    for rank, idx in enumerate(top_indices):
        token_repr = repr(tokens_text[idx])
        if not np.isnan(hr_mean_values[idx]):
            hr_str = f"{hr_mean_values[idx]:>9.5f}  {hr_std_values[idx]:>9.5f}"
        else:
            hr_str = f"{'N/A':>9}  {'N/A':>9}"
        print(f"{rank + 1:>4}  {idx + 1:>5}  {entropies[idx]:>10.4f}  {gate_values[idx]:>10.6f}  {hr_str}  {token_repr}")
    print("-" * 105)

    # 打印 Hidden Ratio(mean) 最高的 Top-20 步骤（隐藏思维占比最大的步骤）
    if has_a_t:
        valid_mask = ~np.isnan(hr_mean_array)
        if valid_mask.sum() > 0:
            top_hr_k = min(20, int(valid_mask.sum()))
            sorted_ratio_indices = np.argsort(np.where(valid_mask, hr_mean_array, -np.inf))[::-1][:top_hr_k]
            print(f"\nHidden Ratio(mean) 最高的 Top-{top_hr_k} 步骤:")
            print("-" * 115)
            print(f"{'Rank':>4}  {'Step':>5}  {'HR_mean':>9}  {'HR_std':>9}  {'HR_min':>9}  {'HR_max':>9}  {'Entropy':>10}  Token")
            print("-" * 115)
            for rank, idx in enumerate(sorted_ratio_indices):
                token_repr = repr(tokens_text[idx])
                print(f"{rank + 1:>4}  {idx + 1:>5}  {hr_mean_values[idx]:>9.5f}  {hr_std_values[idx]:>9.5f}  "
                      f"{hr_min_values[idx]:>9.5f}  {hr_max_values[idx]:>9.5f}  {entropies[idx]:>10.4f}  {token_repr}")
            print("-" * 115)

    # ---- 6. 绘制折线图（双 Y 轴：Entropy + Gate Sigmoid + Hidden Ratio）----
    fig, ax1 = plt.subplots(figsize=(14, 5))
    steps = np.arange(1, num_steps + 1)

    # 左 Y 轴：Entropy
    color_entropy = "steelblue"
    ax1.plot(steps, entropies, linewidth=0.8, color=color_entropy, alpha=0.9, label="Entropy")
    ax1.set_xlabel("Generation Step", fontsize=12)
    ax1.set_ylabel("Entropy (nats)", fontsize=12, color=color_entropy)
    ax1.tick_params(axis="y", labelcolor=color_entropy)

    # 右 Y 轴：Gate Sigmoid Mean + Hidden Ratio（共享 0~1 范围）
    ax2 = ax1.twinx()
    color_gate = "darkorange"
    ax2.plot(steps, gate_values, linewidth=0.8, color=color_gate, alpha=0.7, label="Gate Sigmoid")
    if has_a_t:
        color_ratio = "forestgreen"
        ax2.plot(steps, hr_mean_values, linewidth=0.8, color=color_ratio, alpha=0.7, label="Hidden Ratio (mean)")
    ax2.set_ylabel("Gate Sigmoid / Hidden Ratio", fontsize=12, color=color_gate)
    ax2.tick_params(axis="y", labelcolor=color_gate)

    # 标注 #### 答案标记位置
    answer_marker = ANSWER_START
    full_gen_text = ""
    answer_step = None
    for idx, t in enumerate(tokens_text):
        full_gen_text += t
        if answer_marker in full_gen_text and answer_step is None:
            answer_step = idx + 1  # 1-indexed

    if answer_step is not None:
        ax1.axvline(x=answer_step, color="red", linestyle="--", linewidth=1.0, alpha=0.7)
        ax1.text(
            answer_step, ax1.get_ylim()[1] * 0.95,
            f" {answer_marker}",
            color="red", fontsize=9, va="top",
        )

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=10)

    ax1.set_title("Token-level Entropy, Gate Sigmoid & Hidden Ratio during Generation", fontsize=14)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()

    # 保存图片
    save_dir = adapter_path if os.path.isdir(adapter_path) else os.path.dirname(adapter_path)
    if not save_dir:
        save_dir = "."
    plot_path = os.path.join(save_dir, "entropy_plot.png")
    fig.savefig(plot_path, dpi=150)
    print(f"\n折线图已保存至: {plot_path}")
    plt.close(fig)

    # ---- 6.5 Hidden Ratio(mean) 与 Entropy 关联性分析 ----
    ratio_array = hr_mean_array  # 用于关联性分析的是每步 hidden_ratio 向量的均值
    if has_a_t:
        valid_mask = ~np.isnan(ratio_array)
        r_valid = ratio_array[valid_mask]
        e_valid = ent_array[valid_mask]

        if len(r_valid) >= 5:
            # --- (a) 相关系数 ---
            pearson_r, pearson_p = stats.pearsonr(r_valid, e_valid)
            spearman_r, spearman_p = stats.spearmanr(r_valid, e_valid)

            print("\n" + "=" * 60)
            print("【Hidden Ratio(mean) ↔ Entropy 关联性分析】")
            print("=" * 60)
            print(f"  Pearson  r = {pearson_r:+.4f}  (p = {pearson_p:.2e})")
            print(f"  Spearman ρ = {spearman_r:+.4f}  (p = {spearman_p:.2e})")
            if abs(pearson_r) < 0.2:
                strength = "极弱/无"
            elif abs(pearson_r) < 0.4:
                strength = "弱"
            elif abs(pearson_r) < 0.6:
                strength = "中等"
            elif abs(pearson_r) < 0.8:
                strength = "强"
            else:
                strength = "极强"
            direction = "负" if pearson_r < 0 else "正"
            print(f"  → {strength}{direction}相关")

            # --- (b) Top-N 重叠分析 ---
            overlap_k = min(20, len(r_valid))
            top_entropy_set = set(np.argsort(ent_array)[::-1][:overlap_k])
            high_ratio_set = set(np.argsort(np.where(valid_mask, ratio_array, -np.inf))[::-1][:overlap_k])
            overlap = top_entropy_set & high_ratio_set
            print(f"\n  Top-{overlap_k} 重叠分析:")
            print(f"    熵最高 {overlap_k} 步 ∩ HR_mean最高 {overlap_k} 步 = {len(overlap)} 步重叠")
            print(f"    重叠率: {len(overlap)/overlap_k*100:.1f}%")
            if overlap:
                overlap_sorted = sorted(overlap, key=lambda i: ratio_array[i], reverse=True)
                print(f"    重叠步骤 (按 HR_mean 降序):")
                for idx in overlap_sorted:
                    print(f"      Step {idx+1:>4}: HR_mean={hr_mean_values[idx]:.6f}, HR_std={hr_std_values[idx]:.6f}, "
                          f"Entropy={entropies[idx]:.4f}, Token={repr(tokens_text[idx])}")

            # --- (c) 分箱统计 ---
            n_bins = 5
            bin_edges = np.linspace(r_valid.min(), r_valid.max() + 1e-9, n_bins + 1)
            print(f"\n  分箱统计 ({n_bins} 等宽区间):")
            print(f"  {'HR_mean 区间':>25}  {'样本数':>6}  {'平均Entropy':>12}  {'Entropy标准差':>13}")
            print("  " + "-" * 62)
            bin_mean_entropy = []
            bin_centers = []
            for b in range(n_bins):
                mask_bin = (r_valid >= bin_edges[b]) & (r_valid < bin_edges[b + 1])
                cnt = mask_bin.sum()
                if cnt > 0:
                    mean_e = e_valid[mask_bin].mean()
                    std_e = e_valid[mask_bin].std()
                    bin_mean_entropy.append(mean_e)
                    bin_centers.append((bin_edges[b] + bin_edges[b + 1]) / 2)
                else:
                    mean_e = std_e = float("nan")
                label = f"[{bin_edges[b]:.4f}, {bin_edges[b+1]:.4f})"
                print(f"  {label:>25}  {cnt:>6}  {mean_e:>12.4f}  {std_e:>13.4f}")

            # --- (d) 绘制关联性图 (2x1 子图) ---
            fig_corr, (ax_scatter, ax_bin) = plt.subplots(1, 2, figsize=(14, 5))

            # 左图：散点图 + 回归线
            ax_scatter.scatter(r_valid, e_valid, s=10, alpha=0.5, color="steelblue", edgecolors="none")
            slope, intercept = np.polyfit(r_valid, e_valid, 1)
            x_fit = np.linspace(r_valid.min(), r_valid.max(), 100)
            ax_scatter.plot(x_fit, slope * x_fit + intercept, color="red", linewidth=1.5,
                            label=f"y={slope:.2f}x+{intercept:.2f}")
            ax_scatter.set_xlabel("Hidden Ratio (mean over dims)", fontsize=12)
            ax_scatter.set_ylabel("Entropy (nats)", fontsize=12)
            ax_scatter.set_title(f"Scatter: Pearson r={pearson_r:+.3f}, Spearman ρ={spearman_r:+.3f}", fontsize=11)
            ax_scatter.legend(fontsize=10)
            ax_scatter.grid(True, alpha=0.3)

            # 右图：分箱柱状图
            if bin_centers:
                bar_width = (bin_edges[1] - bin_edges[0]) * 0.7
                ax_bin.bar(bin_centers, bin_mean_entropy, width=bar_width,
                           color="steelblue", alpha=0.7, edgecolor="white")
                ax_bin.set_xlabel("Hidden Ratio mean (bin center)", fontsize=12)
                ax_bin.set_ylabel("Mean Entropy (nats)", fontsize=12)
                ax_bin.set_title("Binned: Mean Entropy per Hidden Ratio Range", fontsize=11)
                ax_bin.grid(True, alpha=0.3, axis="y")

            fig_corr.tight_layout()
            corr_path = os.path.join(save_dir, "hidden_ratio_entropy_correlation.png")
            fig_corr.savefig(corr_path, dpi=150)
            print(f"\n关联性分析图已保存至: {corr_path}")
            plt.close(fig_corr)

            correlation_stats = {
                "pearson_r": float(pearson_r),
                "pearson_p": float(pearson_p),
                "spearman_r": float(spearman_r),
                "spearman_p": float(spearman_p),
                "top_overlap_k": overlap_k,
                "top_overlap_count": len(overlap),
                "top_overlap_steps": sorted([int(i + 1) for i in overlap]),
            }
        else:
            correlation_stats = None
    else:
        correlation_stats = None

    # ---- 7. 保存熵数据为 JSON ----
    entropy_data = {
        "question": question,
        "response": response_text,
        "generated_answer": generated_answer,
        "temperature": temperature,
        "adapter_path": adapter_path,
        "timestamp": datetime.now().isoformat(),
        "num_steps": num_steps,
        "entropy_mean": float(ent_array.mean()),
        "entropy_std": float(ent_array.std()),
        "gate_sigmoid_mean": float(gate_array.mean()),
        "entropies": [float(e) for e in entropies],
        "gate_values": [float(g) for g in gate_values],
        "hidden_ratio_mean": [float(r) if not np.isnan(r) else None for r in hr_mean_values],
        "hidden_ratio_std": [float(r) if not np.isnan(r) else None for r in hr_std_values],
        "hidden_ratio_min": [float(r) if not np.isnan(r) else None for r in hr_min_values],
        "hidden_ratio_max": [float(r) if not np.isnan(r) else None for r in hr_max_values],
        "correlation_stats": correlation_stats,
        "tokens": tokens_text,
    }
    json_path = os.path.join(save_dir, "entropy_data.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(entropy_data, f, indent=2, ensure_ascii=False)
    print(f"熵数据已保存至: {json_path}")

    return entropies, tokens_text, response_text


if __name__ == "__main__":
    # ====== 在此处手动修改参数，直接运行即可调试 ======
    checkpoint_path = "/root/autodl-tmp/HRPO/test0116.2.5.0/Qwen2.5-1.5B-Instruct-gsm8k-group4-lora32-lr0.01-init-2-rmin0.981-temp0.5/checkpoint-934"  # 修改为你的 adapter 路径
    temperature = 0.9
    is_inference = True   # True = greedy, False = sampling
    # ================================================

    # 本地模型路径映射（与 eval_gsm8k.py 一致）
    local_model_paths = {
        "Qwen2.5-1.5B-Instruct": "/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct",
        "Qwen2.5-3B-Instruct": "/root/autodl-tmp/models/Qwen2.5-3B-Instruct",
    }
    base_model = None
    base_models = ["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct"]
    for model in base_models:
        model_name = model.split("/")[-1]
        if model_name in checkpoint_path:
            base_model = local_model_paths.get(model_name, model)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Base model: {base_model}")
    print(f"Temperature: {temperature}")

    run_entropy_test(
        model_path=base_model,
        adapter_path=checkpoint_path,
        temperature=temperature,
        is_inference=is_inference,
        question=QUESTION,
    )
