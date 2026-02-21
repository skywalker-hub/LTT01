"""
LT-Tuning 单问题推理测试脚本。
用法: python test.py
修改下方 QUESTION 变量即可测试不同问题。
"""

import torch
from transformers import AutoTokenizer
from model import LT_Tuning_Model
from eval.utils import extract_answer_from_output, apply_chat_template_if_needed

# ==================== 可修改配置区 ====================

# 要测试的问题（修改这里即可）
QUESTION = "Carlos is planting a lemon tree. The tree will cost $90 to plant. Each year it will grow 7 lemons, which he can sell for $1.5 each. It costs $3 a year to water and feed the tree. How many years will it take before he starts earning money on the lemon tree?" 

# 训练好的模型路径（DeepSpeed/HF 保存的 checkpoint 目录）
MODEL_PATH = "models/example"

# 基础模型路径（仅当 MODEL_PATH 下没有 config.json 时需要，用于加载模型结构）
BASE_MODEL_PATH = "../Llama-3.2-1B"

# 推理参数
MAX_NEW_TOKENS = 1024
STAGE_MODE = "soft_fusion"      # 评测使用的阶段模式: common / hidden_state / soft_fusion
FUSION_ALPHA = 0.6
FUSION_TOP_P = 0.9
FUSION_TEMPERATURE = 1.0
USE_THINKING_TOKEN = True       # True: 允许模型生成 <thinking> 进行潜在推理; False: 禁用
THINKING_TOKEN = "<thinking>"
HIDDEN_STATE_LAYER = -1

# =====================================================


def load_model(device: torch.device):
    """加载训练好的 LT_Tuning_Model 和 tokenizer。"""
    # 先加载 tokenizer 以获取正确的 token ID
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if THINKING_TOKEN not in tokenizer.get_vocab():
        raise ValueError(f"Thinking token '{THINKING_TOKEN}' not found in tokenizer at {MODEL_PATH}. "
                         "The model may not have been trained with this token.")
    thinking_token_id = tokenizer.convert_tokens_to_ids(THINKING_TOKEN)

    model_kwargs = {
        "thinking_token_id": thinking_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "base_model_name_or_path": BASE_MODEL_PATH,
        "device": str(device),
        "stage_mode": STAGE_MODE,
        "fusion_alpha": FUSION_ALPHA,
        "fusion_top_p": FUSION_TOP_P,
        "fusion_temperature": FUSION_TEMPERATURE,
        "hidden_state_layer_index": HIDDEN_STATE_LAYER,
    }

    model = LT_Tuning_Model.from_pretrained(
        model_path=MODEL_PATH,
        **model_kwargs,
    )
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()

    print(f"Model loaded | stage_mode={STAGE_MODE} | thinking_token_id={thinking_token_id}")
    print(f"  dtype={next(model.parameters()).dtype}, device={device}")

    return model, tokenizer, thinking_token_id


def run_inference(model, tokenizer, question: str, device: torch.device):
    """对单个问题进行推理并返回完整输出和提取的答案。"""
    message = [{"role": "user", "content": question}]
    input_text = apply_chat_template_if_needed(tokenizer, message)
    input_ids = tokenizer.encode(input_text, add_special_tokens=False, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            without_thinking_token=(not USE_THINKING_TOKEN),
        )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
    generated_part = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=False)
    answer = extract_answer_from_output(full_output)

    return full_output, generated_part, answer


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    model, tokenizer, _ = load_model(device)

    print("=" * 70)
    print(f"Question: {QUESTION}")
    print("=" * 70)

    full_output, generated_part, answer = run_inference(model, tokenizer, QUESTION, device)

    print(f"\n--- Model Generated ---\n{generated_part}")
    print(f"\n--- Extracted Answer ---\n{answer}")
    print("=" * 70)


if __name__ == "__main__":
    main()
