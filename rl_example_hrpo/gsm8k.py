import os
os.environ["PYDEVD_USE_FRAME_EVAL"] = "NO"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


import os
import subprocess

# ---------------------------------------------------------------------------
# Bootstrap environment *before* importing datasets / huggingface_hub.
#
# In PyCharm remote debug, the Python process often does NOT inherit:
#   source env.sh
#   export HF_ENDPOINT=https://hf-mirror.com
#
# Also, huggingface_hub reads HF_ENDPOINT at import time (cached constants),
# so setting os.environ["HF_ENDPOINT"] later may not affect requests.
# ---------------------------------------------------------------------------

def _source_env_sh_into_process(env_sh_path: str) -> None:
    """Load `export VAR=...` from env.sh into current os.environ (best-effort)."""
    if not os.path.exists(env_sh_path):
        return
    cmd = f'set -a && source "{env_sh_path}" >/dev/null 2>&1 && env -0'
    out = subprocess.check_output(["bash", "-lc", cmd])
    for item in out.split(b"\x00"):
        if not item:
            continue
        k, _, v = item.partition(b"=")
        if k:
            os.environ[k.decode("utf-8", errors="ignore")] = v.decode("utf-8", errors="ignore")


# 1) Emulate: source env.sh (in current Python process)
_source_env_sh_into_process(os.path.join(os.path.dirname(__file__), "env.sh"))

# 2) Emulate: export HF_ENDPOINT=https://hf-mirror.com
# Prefer externally provided value; otherwise default to mirror.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# Some environments/tools use this name; harmless if unused.
os.environ.setdefault("HF_HUB_ENDPOINT", os.environ["HF_ENDPOINT"])

# 3) Pin caches so debug + CLI share the same dataset cache
_cache_root = os.environ.get("HF_CACHE_DIR", "/root/autodl-tmp/hf_cache")
os.environ.setdefault("HF_HOME", _cache_root)
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(_cache_root, "datasets"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(_cache_root, "transformers"))


import unsloth
from unsloth import is_bfloat16_supported
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

import os
from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset, Dataset
from patch import patch_trainer_optimizer
from utils import *

os.environ["WANDB_PROJECT"] = "latent-reasoning"


def preprocess_gsm8k(split="train", chunk_size=1000) -> Dataset:
    dataset = load_dataset('openai/gsm8k', 'main')[split]
    return dataset.map(process_gsm8k, batched=True, 
                       batch_size=chunk_size, load_from_cache_file=False)


def main(args):
    exp_name = (f"./experiments/{args.model_name.split('/')[-1]}-gsm8k-group{args.group_size}"
                f"-lora{args.lora_rank}-rmin{args.residual_r_min}-temp{args.temperature}")
    if os.path.exists(exp_name) and len(os.listdir(exp_name)) > 0:
        print(f"Experiment {exp_name} already exists. Exiting...")
        exit()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_name,
        max_seq_length = args.max_prompt_length + args.max_completion_length,
        load_in_4bit = False,
        load_in_8bit = False,
        fast_inference = False,
    )
    model.answer_start = ANSWER_START

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r = args.lora_rank,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        modules_to_save = [
            "thinking_residual_gate_r",
            "thinking_residual_gate_i",
            "thinking_residual_Lambda",
            "thinking_residual_head",  # 新增: 隐状态变换头
            "token_gate_matrix",  # 新增: Token 门控矩阵
        ], 
        lora_alpha = args.lora_rank * 2,
        use_gradient_checkpointing = "unsloth",
        random_state = args.seed,
    )
    model.model.model.thinking_residual_Lambda.reset_lambda_parameters(
        r_min = args.residual_r_min, r_max = args.residual_r_max,
    )
    
    # ============ 【真正生效的初始化】 ============
    # 注意：模型定义中的初始化会被 post_init() 覆盖，PEFT 包装后需要初始化 modules_to_save.default
    # 这里才是真正决定训练初始值的地方！
    import torch.nn as nn
    
    # 获取真正可训练的权重（PEFT 包装后是 modules_to_save.default.weight）
    head_module = model.model.model.thinking_residual_head
    gate_module = model.model.model.token_gate_matrix
    
    if hasattr(head_module, 'modules_to_save'):
        head_trainable_weight = head_module.modules_to_save.default.weight
    else:
        head_trainable_weight = head_module.weight
    
    if hasattr(gate_module, 'modules_to_save'):
        gate_trainable_weight = gate_module.modules_to_save.default.weight
    else:
        gate_trainable_weight = gate_module.weight
    
    # ★★★ 真正生效的初始化 ★★★
    nn.init.zeros_(head_trainable_weight)           # thinking_residual_head: 初始化为 0
    nn.init.constant_(gate_trainable_weight, -1.0)  # token_gate_matrix: 初始化为 -3, sigmoid(-3)≈0.047
    # ★★★ 修改上面的值来改变初始化 ★★★

    training_args = GRPOConfig(
        use_vllm = False,
        learning_rate = args.lr,
        beta = args.beta,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = args.weight_decay,
        warmup_ratio = args.warmup_ratio,
        lr_scheduler_type = args.lr_scheduler_type,
        optim = args.optimizer,
        max_grad_norm = args.max_grad_norm,
        logging_steps = 1,
        bf16 = is_bfloat16_supported(),
        fp16 = not is_bfloat16_supported(),
        temperature = args.temperature,
        num_generations = args.group_size,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        per_device_train_batch_size = args.per_device_train_batch_size,
        max_prompt_length = args.max_prompt_length,
        max_completion_length = args.max_completion_length,
        num_train_epochs = 1,
        save_steps = 250,
        save_total_limit = 3,
        report_to = "wandb",
        output_dir = exp_name,
    )

    dataset = preprocess_gsm8k('train', chunk_size=500)
    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            get_reward_func(process_gsm8k_answer),
        ],
        args = training_args,
        train_dataset = dataset,
    )
    patch_trainer_optimizer(
        trainer,
        args.lr_residual_gate,
        args.lr_residual_Lambda,
        args.lr_residual_head,  # 新增: 隐状态变换头的学习率
        args.lr_token_gate_matrix,  # 新增: Token 门控矩阵的学习率
    )
    trainer.train()


if __name__ == "__main__":
    args = type("Args", (), {
        "lora_rank": 32,
        "lr": 5e-6,
        "beta": 0.005,
        "residual_r_min": 0.981,
        "residual_r_max": 0.999,
        "lr_residual_gate": 1e-4,
        "lr_residual_Lambda": 1e-3,
        "lr_residual_head": 1e-4,  # 新增: 隐状态变换头的学习率
        "lr_token_gate_matrix": 1e-2,  # 新增: Token 门控矩阵的学习率 (提高以克服bfloat16精度问题)
        "weight_decay": 0.1,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "cosine",
        "optimizer": "paged_adamw_8bit",
        "max_grad_norm": 0.1,
        "group_size": 2,
        "temperature": 0.5,
        "gradient_accumulation_steps": 4,
        "per_device_train_batch_size": 8,
        "max_prompt_length": 1024,
        "max_completion_length": 1024,
        "model_name": "/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct",
        "seed": 42,
    })()

    # "Qwen/Qwen2.5-1.5B-Instruct"
    # "Qwen/Qwen2.5-3B-Instruct"
    # "meta-llama/Llama-3.2-1B-Instruct"
    # "meta-llama/Llama-3.2-3B-Instruct"

    main(args)

    ###日志：本代码仅对HRPO的h做了替换，加入了一个线性头训练。
