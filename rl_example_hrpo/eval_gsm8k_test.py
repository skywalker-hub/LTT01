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
from unsloth import FastLanguageModel

import json
import torch
from datetime import datetime
from datasets import load_dataset
from datasets import DownloadConfig
from transformers import GenerationConfig
from tqdm import tqdm

from utils import *


def evaluate_model(
    model_path: str,
    adapter_path: str,
    temperature: float,
    is_inference: bool,
    batch_size: int = 4,
    num_samples: int = None,
    save_results: bool = True,
):
    # ---- Make debug + CLI behave the same (HuggingFace env & cache) ----
    # PyCharm remote debug often doesn't inherit `source env.sh` / proxy vars,
    # and may use a different $HOME thus a different HF cache.
    # We fix this by pinning cache dirs and using HF_ENDPOINT explicitly.
    HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ["HF_ENDPOINT"] = HF_ENDPOINT
    # Pin caches to a stable directory (override via HF_CACHE_DIR if needed)
    _cache_root = os.environ.get("HF_CACHE_DIR", "/root/autodl-tmp/hf_cache")
    os.environ.setdefault("HF_HOME", _cache_root)
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(_cache_root, "datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(_cache_root, "transformers"))

    def _load_gsm8k_test():
        """
        Prefer local cache (works offline / in restricted debug env).
        If missing, fall back to online download via HF_ENDPOINT.
        """
        cache_dir = os.environ.get("HF_DATASETS_CACHE")
        # 1) Offline / cache-only attempt
        try:
            return load_dataset(
                "openai/gsm8k",
                "main",
                cache_dir=cache_dir,
                download_config=DownloadConfig(local_files_only=True),
            )["test"]
        except Exception as e_offline:
            print(f"[info] GSM8K not found in local cache ({cache_dir}) or cache-only load failed: {type(e_offline).__name__}: {e_offline}")
        # 2) Online attempt (mirror)
        return load_dataset(
            "openai/gsm8k",
            "main",
            cache_dir=cache_dir,
            download_config=DownloadConfig(local_files_only=False),
        )["test"]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = 1024,
        load_in_4bit = False,
        fast_inference = False,
    )
    model.answer_start = ANSWER_START
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model.load_adapter(adapter_path)
    model = FastLanguageModel.for_inference(model)

    dataset = _load_gsm8k_test()
    if num_samples and len(dataset) > num_samples:
        dataset = dataset.shuffle(seed=42).select(range(num_samples))
    total_samples = len(dataset)
    print(f"Loaded {total_samples} samples")

    results = []
    correct = 0
    total = 0

    progress_bar = tqdm(
        total=total_samples,
        desc="Processing samples",
        unit="examples",
        dynamic_ncols=True,
    )
    progress_bar.set_postfix({'acc': '0.00%', 'correct': '0'})

    # Process samples in batches
    for i in range(0, total_samples, batch_size):
        batch_data = dataset[i:i + batch_size]
        current_batch_size = len(batch_data['question'])

        # Prepare prompts using the same format as training
        prompts = [
            [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': q.strip()},
            ]
            for q in batch_data['question']
        ]

        # Convert chat prompts to the required format
        formatted_prompts = [
            tokenizer.apply_chat_template(
                p,
                tokenize=False,
                add_generation_prompt=True
            )
            for p in prompts
        ]

        prompt_inputs = tokenizer(
            formatted_prompts, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        prompt_ids = prompt_ids.to(model.device)
        prompt_mask = prompt_mask.to(model.device)
        prompt_length = prompt_ids.size(1)

        # Generate responses
        #####主断点1:在这里打断点
        outputs = model.generate(
            prompt_ids, attention_mask=prompt_mask, 
            generation_config=GenerationConfig(
                do_sample=True,  # for temperature, top-k, etc.
                temperature=temperature,
                max_new_tokens=512,
                output_hidden_states=True,
            ),
            processing_class=tokenizer,
            is_inference=is_inference,
        )

        # Process each generated response
        for j, output in enumerate(outputs):
            response = tokenizer.decode(output[prompt_length:])
            response = response.split(
                tokenizer.special_tokens_map['eos_token']
            )[0]

            # Extract the generated answer using XML tags
            extracted = extract_from_response(response)
            generated_answer = process_gsm8k_answer(extracted)
            true_answer = extract_hash_answer(batch_data['answer'][j])
            true_answer = process_gsm8k_answer(true_answer)
            print(generated_answer, true_answer, generated_answer == true_answer)

            # Store the result
            result = {
                'question': batch_data['question'][j],
                'true_answer': true_answer,
                'generated_answer': generated_answer,
                'full_response': response,
                'correct': generated_answer == true_answer
            }
            results.append(result)

            if generated_answer == true_answer:
                correct += 1
            total += 1

        progress_bar.update(current_batch_size)
        progress_bar.set_postfix({
            'acc': f'{(correct/total)*100:.2f}%',
            'correct': f'{correct}/{total}',
        })

    progress_bar.close()
    accuracy = correct / total if total > 0 else 0
    metrics = {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'model_path': adapter_path,
        'timestamp': datetime.now().isoformat()
    }

    if save_results:
        save_path = adapter_path + "/eval_results.json"
        with open(save_path, 'w') as f:
            json.dump({'metrics': metrics, 'results': results}, f, indent=2)
        print(f"\nResults saved to {save_path}")

    return metrics


if __name__ == "__main__":
    # 直接写死为你启动命令中的配置
    greedy = False
    batch_size = 2
    checkpoint_path = "/root/autodl-tmp/HRPO/test0116.2.0.4/Qwen2.5-1.5B-Instruct-gsm8k-group4-lora32-lr0.01-init-2-rmin0.981-temp0.5/checkpoint-934"

    base_model = None
    ### 本地模型路径映射
    local_model_paths = {
        "Qwen2.5-1.5B-Instruct": "/root/autodl-tmp/models/Qwen2.5-1.5B-Instruct",
        "Qwen2.5-3B-Instruct": "/root/autodl-tmp/models/Qwen2.5-3B-Instruct"
    }
    base_models = ["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct"]
    for model in base_models:
        model_name = model.split('/')[-1]
        if model_name in checkpoint_path:
            # 使用本地路径
            base_model = local_model_paths.get(model_name, model)
    #temperature = float(checkpoint_path.split('-temp')[-1].split('/')[0])
    temperature = 0.9
    print(checkpoint_path, base_model, temperature)

    if 'eval_results.json' not in os.listdir(checkpoint_path):
        print(f"Starting GSM8k evaluation on {checkpoint_path}")
        metrics = evaluate_model(
            model_path=base_model,
            adapter_path=checkpoint_path,
            temperature=temperature,
            is_inference=greedy,
            batch_size=batch_size,
            num_samples=None,
            save_results=True,
        )