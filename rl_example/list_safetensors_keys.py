#!/usr/bin/env python3
"""从 adapter_model.safetensors 读取并打印所有 key 列表。"""

from safetensors.torch import load_file

path = "./test0116.2.0.4/Qwen2.5-1.5B-Instruct-gsm8k-group4-lora32-lr0.01-init-2-rmin0.981-temp0.5/checkpoint-934/adapter_model.safetensors"  # 改成你的实际路径
state_dict = load_file(path)

print("Total keys:", len(state_dict))
print("=" * 60)

for k in state_dict.keys():
    print(k)
