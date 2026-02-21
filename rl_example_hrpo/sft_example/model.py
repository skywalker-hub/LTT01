# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from collections import namedtuple
from typing import Optional
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.models.gpt2 import GPT2LMHeadModel
import gc
from safetensors.torch import load_file
## 模型输出的命名元组，包含: 损失、输入嵌入、logits、KV缓存、最后一层隐藏状态、注意力权重
Outputs = namedtuple(
    "Outputs", ["loss", "inputs_embeds", "logits", "past_key_values", "last_hidden_state", "attentions"]
)
MAX_THINKING_AUTOREGRESS = 32


class LT_Tuning_Model(nn.Module):
    """
    潜在思维调优（Latent Thoughts Tuning）模型。
    该模型包装了一个基础因果语言模型，通过三阶段课程学习实现潜在推理：
      - 阶段0 (common): 显式CoT推理预热，标准SFT训练（对应论文4.1节）
      - 阶段1 (hidden_state): 动态潜在标记生成，用隐藏状态替代<thinking>的嵌入（对应论文4.2节）
      - 阶段2 (soft_fusion): 上下文-预测融合，融合隐藏状态与预测分布的嵌入（对应论文4.3节）
    """

    def __init__(
        self,
        base_causallm,
        thinking_token_id,
        eos_token_id,
        use_thinking_mlp: bool = False,
        mlp_hidden_dim: Optional[int] = None,
        mlp_activation: str = "gelu",
        hidden_state_layer_index: int = -1,       # 论文中的第I层，用于提取隐藏状态 h_{t-1,I}
        stage_mode: str = "explicit",              # 当前阶段模式: explicit/common, hidden_state, soft_fusion
        fusion_alpha: float = 0.6,                 # 论文公式(7)中的 α，平衡隐藏状态与预测嵌入
        fusion_top_p: float = 0.9,                 # 论文公式(6)中 Top-p 过滤的阈值
        fusion_temperature: float = 1.0,           # 论文公式(6)中 logits 的温度缩放参数
        **kwargs,
    ):
        super().__init__()
        ## 基础因果语言模型（如 Llama-3.2-1B）
        self.base_causallm = base_causallm
        # 获取嵌入层和语言模型头，兼容 GPT2 和 Llama 架构
        if isinstance(self.base_causallm, GPT2LMHeadModel):
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
            self.lm_head = self.base_causallm.lm_head
        else:
            self.embedding = self.base_causallm.get_input_embeddings()
            self.lm_head = self.base_causallm.lm_head
        ## <thinking> 标记的 token ID，用于标识潜在推理位置
        self.thinking_token_id = thinking_token_id
        self.eos_token_id = eos_token_id
        self.gen_forward_cnt = 0

        self.use_thinking_mlp = use_thinking_mlp
        self.mlp_hidden_dim = mlp_hidden_dim
        self.mlp_activation = mlp_activation
        ## 论文中的第 I 层索引，用于提取 h_{t-1,I}（默认 -1 表示最后一层）
        self.hidden_state_layer_index = hidden_state_layer_index
        
        ## 当前训练阶段模式，对应论文三阶段课程学习
        self.stage_mode = stage_mode
        ## 论文公式(7)中的 α 系数，用 register_buffer 避免设备不匹配问题
        self.register_buffer("fusion_alpha", torch.tensor(fusion_alpha))
        ## 论文中 Top-p 过滤阈值和温度缩放参数
        self.fusion_top_p = fusion_top_p
        self.fusion_temperature = fusion_temperature

        ## 可选的 MLP 变换层，用于对 <thinking> 嵌入进行非线性变换
        if self.use_thinking_mlp:
            input_dim = self.embedding.embedding_dim
            hidden_dim = self.mlp_hidden_dim or input_dim
            activation_module = self._get_activation(self.mlp_activation)
            self.thinking_mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                activation_module,
                nn.Linear(hidden_dim, input_dim),
                nn.LayerNorm(input_dim),
            )
        else:
            self.thinking_mlp = None

    @property
    def config(self):
        """Expose base model's config for HuggingFace Trainer/DeepSpeed compatibility."""
        return self.base_causallm.config

    @property
    def device(self):
        """Return the device of the model."""
        return next(self.parameters()).device

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs
    ):
        """
        Load LT_Tuning_Model model from pretrained weights.
        
        Supports two formats:
        1. Standard HuggingFace format: has config.json, weights are base_causallm's
        2. LT_Tuning_Model checkpoint format: single model.safetensors with LT_Tuning_Model structure
        """
        print("Using from_pretrained scheme for LTF models...")
        print(f"Loading model from: {model_path}")
        print(f"Target device: {device}")
        
        config_path = os.path.join(model_path, "config.json")
        single_safetensor = os.path.join(model_path, "model.safetensors")
        
        # Detect format: if config.json exists, it's HuggingFace format
        # If only model.safetensors exists without config.json, it's LT_Tuning_Model checkpoint format
        has_config = os.path.exists(config_path)
        has_single_safetensor = os.path.exists(single_safetensor)
        
        # Check if it's LT_Tuning_Model checkpoint format (no config but has model.safetensors)
        is_LT_Tuning_Model_checkpoint = has_single_safetensor and not has_config
        
        if is_LT_Tuning_Model_checkpoint:
            print("Detected LT_Tuning_Model checkpoint format (no config.json, has model.safetensors)")
            return cls._load_from_LT_Tuning_Model_checkpoint(model_path, device, **kwargs)
        else:
            print("Detected HuggingFace format (has config.json)")
            return cls._load_from_hf_format(model_path, device, **kwargs)
    
    @classmethod
    def _load_from_hf_format(
        cls,
        model_path: str,
        device: str,
        **kwargs
    ):
        """Load from standard HuggingFace format (base_causallm weights)."""
        # Get attention implementation from kwargs (default: flash_attention_2)
        attn_implementation = kwargs.pop("attn_implementation", "flash_attention_2")
        
        base_causallm = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation=attn_implementation,
            trust_remote_code=True,
        )
        print(f"Base model loaded: {type(base_causallm).__name__} (attn: {attn_implementation})")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        t_id = kwargs.get("thinking_token_id")
        if t_id is None:
            if "<thinking>" in tokenizer.get_vocab():
                t_id = tokenizer.convert_tokens_to_ids("<thinking>")
            else:
                t_id = tokenizer.unk_token_id
                
        if "thinking_token_id" in kwargs:
            model = cls(
                base_causallm=base_causallm,
                **kwargs
            )
        else:
            model = cls(
                base_causallm=base_causallm,
                thinking_token_id=t_id,
                **kwargs
            )
        custom_weights_path = os.path.join(model_path, "LT_Tuning_Model_custom_weights.pt")
        if os.path.exists(custom_weights_path):
            print(f"Loading custom LT_Tuning_Model weights from {custom_weights_path}")
            custom_state = torch.load(custom_weights_path, map_location=device)
            model.load_state_dict(custom_state, strict=False)
        
        print(f"Successfully loaded model (HF format) from {model_path}.")
        model.eval()
        
        return model
    
    @classmethod
    def _load_from_LT_Tuning_Model_checkpoint(
        cls,
        model_path: str,
        device: str,
        **kwargs
    ):
        """Load from LT_Tuning_Model checkpoint format (full LT_Tuning_Model structure in model.safetensors)."""
        base_model_path = kwargs.pop("base_model_name_or_path", None)

        print(f"Loading base model from: {base_model_path}")
        
        # Get attention implementation from kwargs (default: flash_attention_2)
        attn_implementation = kwargs.pop("attn_implementation", "flash_attention_2")
        base_causallm = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation=attn_implementation,
            trust_remote_code=True,
        )
        print(f"Base model structure loaded from: {base_model_path} (attn: {attn_implementation})")
        
        checkpoint_file = os.path.join(model_path, "model.safetensors")
        checkpoint_state = load_file(checkpoint_file)
        checkpoint_vocab_size = None
        for key in ["base_causallm.model.embed_tokens.weight", "embedding.weight"]:
            if key in checkpoint_state:
                checkpoint_vocab_size = checkpoint_state[key].shape[0]
                break
        
        if checkpoint_vocab_size is None:
            raise RuntimeError(f"Cannot determine vocab size from checkpoint at {model_path}")
        
        print(f"Checkpoint vocab size: {checkpoint_vocab_size}")

        tokenizer_path = model_path if os.path.exists(os.path.join(model_path, "tokenizer.json")) else base_model_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        print(f"Tokenizer vocab size: {len(tokenizer)}")
        
        thinking_token = kwargs.pop("thinking_token", "<thinking>")
        use_unk_for_thinking = kwargs.pop("use_unk_for_thinking", False)
        
        if use_unk_for_thinking:
            if tokenizer.unk_token_id is None:
                raise ValueError("Tokenizer has no unk_token but use_unk_for_thinking=True")
            t_id = tokenizer.unk_token_id
            print(f"Using UNK token as thinking token: {t_id}")
        else:
            if thinking_token in tokenizer.get_vocab():
                t_id = tokenizer.convert_tokens_to_ids(thinking_token)
                print(f"Thinking token '{thinking_token}' already in vocab at index {t_id}")
            else:
                newly_added = tokenizer.add_tokens([thinking_token])
                t_id = tokenizer.convert_tokens_to_ids(thinking_token)
                print(f"Added thinking token '{thinking_token}' at index {t_id} (newly_added={newly_added})")
        
        current_vocab_size = base_causallm.get_input_embeddings().num_embeddings
        if current_vocab_size != checkpoint_vocab_size:
            print(f"Resizing embeddings from {current_vocab_size} to {checkpoint_vocab_size} to match checkpoint")
            base_causallm.resize_token_embeddings(checkpoint_vocab_size)
            
            if len(tokenizer) != checkpoint_vocab_size:
                print(f"Warning: tokenizer size {len(tokenizer)} != checkpoint vocab size {checkpoint_vocab_size}")
                if len(tokenizer) < checkpoint_vocab_size:
                    num_to_add = checkpoint_vocab_size - len(tokenizer)
                    print(f"Adding {num_to_add} placeholder tokens to match checkpoint")
                    tokenizer.add_tokens([f"<placeholder_{i}>" for i in range(num_to_add)])
        
        t_id = kwargs.get("thinking_token_id", t_id)
        
        if "thinking_token_id" in kwargs:
            model = cls(
                base_causallm=base_causallm,
                **kwargs
            )
        else:
            model = cls(
                base_causallm=base_causallm,
                thinking_token_id=t_id,
                **kwargs
            )
        
        print("Model skeleton initialized, loading weights from checkpoint...")
        
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint_state, strict=False)
        if missing_keys:
            print(f"Missing keys (expected for non-trained params): {len(missing_keys)}")
            if len(missing_keys) > 0:
                print(f"Sample missing keys: {missing_keys[:5]}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
        model = model.to(device)
        if model.thinking_mlp is not None:
            model.thinking_mlp = model.thinking_mlp.to(device)
            print(f"Moved thinking_mlp to device: {device}")
        
        print(f"Successfully loaded model (LT_Tuning_Model checkpoint) from {model_path}.")
        model.eval()
        
        return model

    def update_stage_config(self, stage_mode: str, fusion_alpha: float = None, 
                           fusion_top_p: float = None, fusion_temperature: float = None):
        """Update stage-specific configuration dynamically."""
        self.stage_mode = stage_mode
        if fusion_alpha is not None:
            self.fusion_alpha = torch.tensor(fusion_alpha, device=self.fusion_alpha.device, dtype=self.fusion_alpha.dtype)
        if fusion_top_p is not None:
            self.fusion_top_p = fusion_top_p
        if fusion_temperature is not None:
            self.fusion_temperature = fusion_temperature

    def _get_activation(self, name: str) -> nn.Module:
        name = name.lower()
        if name == "relu":
            return nn.ReLU()
        if name == "gelu":
            return nn.GELU()
        if name == "tanh":
            return nn.Tanh()
        raise ValueError(f"Unsupported activation for thinking MLP: {name}")

    def _apply_transform(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """对隐藏状态应用可选的 MLP 变换（论文中未强调但代码支持的可选扩展）"""
        if self.use_thinking_mlp and self.thinking_mlp is not None:
            return self.thinking_mlp(hidden_state)
        return hidden_state
    
    def _soft_fusion_embedding(self, hidden_state: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """
        【论文4.3节 - 上下文-预测融合】
        将隐藏状态 h_{t-1,I} 与预测分布加权嵌入 e_pred 进行融合。
        对应论文公式(6)和(7):
          e_pred = Σ_{w∈V} P̂(w) · E(w)          ... 公式(6)
          e_fusion = α · h_{t-1,I} + (1-α) · e_pred  ... 公式(7)
        """
        ## 论文公式(6)的温度缩放: 对 logits 进行温度缩放以聚焦高置信度预测
        scaled_logits = logits / self.fusion_temperature
        
        ## 屏蔽 <thinking> 标记，防止其参与预测分布（论文4.3节要求）
        masked_logits = scaled_logits.clone()
        masked_logits[self.thinking_token_id] = float('-inf')
        
        ## 计算 softmax 概率分布 P(w)
        probs = torch.softmax(masked_logits, dim=-1)
        
        ## 论文公式(6): 应用 Top-p 过滤，只保留累积概率不超过 top_p 的高置信度标记
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
        
        ## 找到 Top-p 的截断位置
        cutoff_mask = cumsum_probs <= self.fusion_top_p
        if cutoff_mask.sum() == 0:  # 至少保留一个标记
            cutoff_mask[0] = True
        
        ## Top-p 过滤后重新归一化概率，得到 P̂(w)
        filtered_probs = torch.zeros_like(probs)
        filtered_probs[sorted_indices[cutoff_mask]] = sorted_probs[cutoff_mask]
        filtered_probs = filtered_probs / filtered_probs.sum()
        
        ## 论文公式(6): 计算预测分量 e_pred = Σ P̂(w) · E(w)
        ## 将过滤后的概率分布投影到嵌入流形上，得到加权嵌入向量
        token_embed_component = filtered_probs @ self.embedding.weight
        
        ## 论文公式(7): 上下文-预测融合 e_fusion = α · h_{t-1,I} + (1-α) · e_pred
        ## α 平衡隐藏状态（上下文历史）与预测嵌入（模型预测分布）
        alpha = self.fusion_alpha.item()
        fused_embed = alpha * hidden_state + (1 - alpha) * token_embed_component
        
        return fused_embed
    
    def get_fusion_stats(self, hidden_state: torch.Tensor, logits: torch.Tensor, tokenizer=None):
        """Get statistics about fusion for debugging/monitoring (call with no_grad)."""
        with torch.no_grad():
            scaled_logits = logits / self.fusion_temperature
            masked_logits = scaled_logits.clone()
            masked_logits[self.thinking_token_id] = float('-inf')
            probs = torch.softmax(masked_logits, dim=-1)
            
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
            cutoff_mask = cumsum_probs <= self.fusion_top_p
            if cutoff_mask.sum() == 0:
                cutoff_mask[0] = True
            
            stats = {
                'top_token_prob': sorted_probs[0].item(),
                'num_tokens_in_top_p': cutoff_mask.sum().item(),
                'fusion_alpha': self.fusion_alpha.item(),
                'entropy': -(probs * torch.log(probs + 1e-10)).sum().item(),
            }
            
            if tokenizer is not None:
                top_5_tokens = [(tokenizer.decode([idx.item()]), prob.item()) 
                               for idx, prob in zip(sorted_indices[:5], sorted_probs[:5])]
                stats['top_5_tokens'] = top_5_tokens
            
            return stats

    def _select_hidden_state(self, hidden_states) -> torch.Tensor:
        """从模型各层隐藏状态中选取第 I 层的隐藏状态 h_{t-1,I}（论文4.2节和4.3节）"""
        index = self.hidden_state_layer_index
        if index is None:
            index = -1
        if index < 0:
            index = len(hidden_states) + index
        if index < 0 or index >= len(hidden_states):
            raise IndexError(
                f"hidden_state_layer_index {self.hidden_state_layer_index} out of range for hidden_states of length {len(hidden_states)}"
            )
        return hidden_states[index]

    def forward(
        self,
        input_ids,
        attention_mask = None,
        labels = None,
        position_ids = None,
        return_probing_info = False,
        output_attentions = False,
        **kwargs,
    ):
        """
        【核心前向传播】
        将序列按 <thinking> 标记位置切分为多个片段(segment)，依次处理。
        在每个 <thinking> 位置，根据当前阶段模式(stage_mode)，
        用前一位置的隐藏状态(或融合嵌入)来替代 <thinking> 的输入嵌入。
        这是论文方法的核心实现——让 <thinking> 作为不可语言化的潜在推理步骤。
        """
        batch_size, seq_len = input_ids.shape

        embed_dim = self.embedding.embedding_dim
        
        all_attentions = [] if output_attentions else None

        ## 【步骤1】找出 batch 中每个样本的所有 <thinking> 标记位置
        thinking_positions_batch = []
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=input_ids.device)
        if position_ids is None:
            position_ids = torch.arange(
                0, seq_len, dtype=torch.long, device=input_ids.device
            ).unsqueeze(0).expand(batch_size, -1)
        attention_mask_bool = attention_mask.bool()
        for batch_idx in range(batch_size):
            thinking_mask = (
                (input_ids[batch_idx] == self.thinking_token_id)
                & attention_mask_bool[batch_idx]
            )
            positions = torch.nonzero(thinking_mask, as_tuple=False).squeeze(-1)
            thinking_positions_batch.append(positions.tolist())

        ## 【步骤2】确定切分边界：在每个 <thinking> 位置处切分序列
        ## 这样可以先处理 <thinking> 之前的文本标记，获取隐藏状态后再生成 <thinking> 的嵌入
        boundary_positions = set()
        for positions in thinking_positions_batch:
            for pos in positions:
                if 0 < pos < seq_len:
                    boundary_positions.add(pos)
        boundary_positions = sorted(boundary_positions)
        boundary_positions.append(seq_len)
        current_start = 0
        kv_cache = None
        outputs = None
        logits_segments = []
        segment_embeds_accum = []
        ## 记录每个 batch 样本中尚未处理的 <thinking> 位置
        thinking_sets = [set(lst) for lst in thinking_positions_batch]
        ## 存储已生成的 <thinking> 替代嵌入，key=(batch_idx, position)
        thinking_replacements = {}

        selected_hidden_state = None

        ## 【步骤3】按片段依次处理序列，在 <thinking> 边界处生成潜在嵌入
        for boundary_end in boundary_positions:
            if boundary_end <= current_start:
                continue

            segment_length = boundary_end - current_start
            if segment_length <= 0:
                current_start = boundary_end
                continue

            segment_slice = slice(current_start, boundary_end)
            segment_ids = input_ids[:, segment_slice]

            ## 为当前片段构建嵌入矩阵
            segment_embeds = torch.zeros(
                batch_size,
                segment_length,
                embed_dim,
                device=segment_ids.device,
                dtype=self.embedding.weight.dtype,
            )

            ## 逐位置构建嵌入：普通标记用嵌入查表，<thinking> 用之前生成的替代嵌入
            for offset in range(segment_length):
                absolute_pos = current_start + offset
                token_column = segment_ids[:, offset]
                thinking_mask = token_column == self.thinking_token_id

                ## 普通文本标记：直接查嵌入表
                if (~thinking_mask).any():
                    non_thinking_ids = token_column[~thinking_mask]
                    non_thinking_embeds = self.embedding(non_thinking_ids)
                    segment_embeds[~thinking_mask, offset, :] = non_thinking_embeds

                ## <thinking> 标记：使用之前计算好的替代嵌入（来自上一片段的隐藏状态）
                if thinking_mask.any():
                    thinking_indices = torch.nonzero(thinking_mask, as_tuple=False).flatten()
                    for batch_idx in thinking_indices.tolist():
                        replacement = thinking_replacements.get((batch_idx, absolute_pos))
                        if replacement is None:
                            raise RuntimeError(
                                "Missing replacement embedding for thinking token at position"
                                f" {absolute_pos} in batch {batch_idx}."
                            )
                        segment_embeds[batch_idx, offset, :] = replacement
                        del thinking_replacements[(batch_idx, absolute_pos)]

            ## 将当前片段送入基础模型，使用 KV 缓存实现增量式前向传播
            if kv_cache is None:
                outputs = self.base_causallm(
                    inputs_embeds=segment_embeds,
                    attention_mask=attention_mask[:, segment_slice],
                    position_ids=position_ids[:, segment_slice],
                    output_hidden_states=True,   # 需要输出各层隐藏状态以提取 h_{t-1,I}
                    output_attentions=output_attentions,
                    use_cache=True,
                )
            else:
                outputs = self.base_causallm(
                    inputs_embeds=segment_embeds,
                    attention_mask=attention_mask[:, :boundary_end],
                    position_ids=position_ids[:, segment_slice],
                    past_key_values=kv_cache,
                    output_hidden_states=True,
                    output_attentions=output_attentions,
                    use_cache=True,
                )
            
            if output_attentions and outputs.attentions is not None:
                all_attentions.append(outputs.attentions)

            logits_segments.append(outputs.logits)
            segment_embeds_accum.append(segment_embeds)
            ## 从模型输出中提取第 I 层的隐藏状态 h_{t-1,I}（论文4.2/4.3节）
            hidden_states = self._select_hidden_state(outputs.hidden_states)
            selected_hidden_state = hidden_states
            kv_cache = outputs.past_key_values

            ## 【关键步骤】如果下一个位置是 <thinking>，为其生成替代嵌入
            ## 这是论文方法的核心：根据 stage_mode 决定 <thinking> 标记的输入嵌入
            if boundary_end < seq_len:
                batch_indices_to_update = [
                    batch_idx
                    for batch_idx in range(batch_size)
                    if boundary_end in thinking_sets[batch_idx]
                ]

                if batch_indices_to_update and segment_length > 0:
                    ## 取当前片段最后一个位置的隐藏状态，即位置 t-1 的 h_{t-1,I}
                    hidden_idx = segment_length - 1
                    for batch_idx in batch_indices_to_update:
                        prev_hidden = hidden_states[batch_idx, hidden_idx, :]
                        
                        ## ===== 根据训练阶段选择不同的嵌入生成策略 =====
                        if self.stage_mode == "hidden_state":
                            ## 【阶段1 - 论文4.2节】动态潜在标记生成
                            ## 直接使用位置 t-1 处第 I 层的隐藏状态 h_{t-1,I} 作为 <thinking> 的输入嵌入
                            ## "确保潜在推理仅保留用于不确定步骤"
                            replacement = self._apply_transform(prev_hidden)
                        elif self.stage_mode == "soft_fusion":
                            ## 【阶段2 - 论文4.3节】上下文-预测融合
                            ## 先获取位置 t-1 的 logit 分布 l_{t-1}
                            prev_logits = outputs.logits[batch_idx, hidden_idx, :]
                            ## 调用 _soft_fusion_embedding 执行融合：
                            ## e_fusion = α · h_{t-1,I} + (1-α) · e_pred（公式7）
                            replacement = self._soft_fusion_embedding(prev_hidden, prev_logits)
                            replacement = self._apply_transform(replacement)
                        else:
                            ## 【阶段0 - 论文4.1节】显式推理预热（common 模式）
                            ## 此阶段通常不插入 <thinking> 标记，若意外出现则用隐藏状态替代
                            replacement = self._apply_transform(prev_hidden)
                        
                        ## 将生成的替代嵌入存入字典，供下一片段处理 <thinking> 位置时使用
                        thinking_replacements[(batch_idx, boundary_end)] = replacement
                        thinking_sets[batch_idx].remove(boundary_end)

            current_start = boundary_end

        if outputs is None:
            raise RuntimeError("No forward pass executed in LT_Tuning_Model model.")

        ## 【步骤4】拼接所有片段的 logits 和嵌入
        logits = torch.cat(logits_segments, dim=1)
        inputs_embeds = torch.cat(segment_embeds_accum, dim=1)
        self.gen_forward_cnt += len(logits_segments)
        del logits_segments
        del segment_embeds_accum
        gc.collect()
        torch.cuda.empty_cache()
        ## 【步骤5】计算损失 - 对应论文公式(4): L_CoT = -Σ_t log p_θ(y_t | x, y_{<t})
        ## 使用标准的自回归交叉熵损失，shift 使 logits[t] 预测 labels[t+1]
        ## 注意：<thinking> 位置的 labels 是正常的下一个文本标记，
        ## 这意味着模型需要通过潜在嵌入来预测后续的显式标记
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
        else:
            loss = None

        if selected_hidden_state is None:
            raise RuntimeError("Failed to select hidden state during forward pass in LT_Tuning_Model.")
        merged_attentions = None
        if output_attentions and all_attentions:
            num_layers = len(all_attentions[0])
            merged_attentions = []
            
            for layer_idx in range(num_layers):
                # Get attention from all segments for this layer
                layer_attentions = [seg_attn[layer_idx] for seg_attn in all_attentions]
                
                # Build full attention matrix by padding and concatenating
                batch_size, num_heads = layer_attentions[0].shape[:2]
                
                # Calculate total sequence length from the last segment's key dimension
                total_seq_len = layer_attentions[-1].shape[-1]
                
                # Create full attention matrix
                full_attention = torch.zeros(
                    batch_size, num_heads, total_seq_len, total_seq_len,
                    dtype=layer_attentions[0].dtype,
                    device=layer_attentions[0].device
                )
                
                # Fill in attention values from each segment
                query_start = 0
                for seg_attn in layer_attentions:
                    seg_query_len = seg_attn.shape[2]
                    seg_key_len = seg_attn.shape[3]
                    
                    # The segment attends to positions [0, seg_key_len)
                    # Query positions are [query_start, query_start + seg_query_len)
                    full_attention[:, :, query_start:query_start + seg_query_len, :seg_key_len] = seg_attn
                    query_start += seg_query_len
                
                merged_attentions.append(full_attention)
            
            merged_attentions = tuple(merged_attentions)

        # Prepare output with optional probing info
        output = Outputs(
            loss=loss,
            inputs_embeds=inputs_embeds,
            logits=logits,
            past_key_values=outputs.past_key_values,
            last_hidden_state=selected_hidden_state,
            attentions=merged_attentions,
        )
        
        if return_probing_info:
            # Directly extract logits from final logits tensor
            # Assume batch_size = 1 for probing
            probing_info = {
                'thinking_positions': thinking_positions_batch[0],
                'logits_before_thinking': [],  # Logits at position before <thinking>
                'logits_at_thinking': [],   # Logits at <thinking> position after transformation
            }
            
            for thinking_pos in thinking_positions_batch[0]:
                if thinking_pos > 0 and thinking_pos < logits.shape[1]:
                    # Logits at position before <thinking>
                    probing_info['logits_before_thinking'].append({
                        'thinking_position': thinking_pos,
                        'logits': logits[0, thinking_pos - 1].clone().detach()
                    })
                    # Logits at <thinking> position (after latent forward)
                    probing_info['logits_at_thinking'].append({
                        'thinking_position': thinking_pos,
                        'logits': logits[0, thinking_pos].clone().detach()
                    })
            
            return output, probing_info
        
        return output
    def train(self, mode: bool = True):
        self.base_causallm.train(mode)
        if self.thinking_mlp is not None:
            self.thinking_mlp.train(mode)
        return super().train(mode)

    def eval(self):
        return self.train(False)

    def generate(
        self,
        input_ids,
        attention_mask,
        max_new_tokens=16,
        output_embedding=False,
        synced_gpus=False,
        without_thinking_token=False,     # 若为 True，则禁止模型生成 <thinking> 标记
        **kwargs
    ):
        """
        自回归生成方法。
        关键参数 without_thinking_token:
          - True: 将 <thinking> 的 logit 设为 -inf，禁止生成 <thinking>（评测时使用）
          - False: 允许模型自主决定是否生成 <thinking> 进行潜在推理
        当模型生成 <thinking> 时，会根据 stage_mode 为其构建嵌入（同 forward 逻辑）。
        """

        self.gen_forward_cnt = 0

        assert input_ids.shape[0] == 1, "only support batch_size == 1 now"
        tokens = input_ids[0].detach().tolist()

        ## 先对输入序列做一次完整的前向传播，获取 KV 缓存和隐藏状态
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids, device=input_ids.device),
            labels=None,
            position_ids=torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            ).reshape(1, -1),
        )
        inputs_embeds = outputs.inputs_embeds
        past_key_values = outputs.past_key_values
        selected_hidden_state = outputs.last_hidden_state

        ## 如果禁用 thinking，将 <thinking> 标记的 logit 设为负无穷，阻止其被选中
        if without_thinking_token:
            outputs.logits[0, -1, self.thinking_token_id] = float('-inf')

        ## 贪心解码：选择概率最大的 token
        next_token = torch.argmax(outputs.logits[0, -1]).item()
        tokens.append(next_token)
        ## 如果生成了 <thinking> 标记，需要根据阶段模式构建其嵌入
        if next_token == self.thinking_token_id:
            prev_hidden = selected_hidden_state[:, -1, :]
            if self.stage_mode == "soft_fusion":
                ## 阶段2: 上下文-预测融合
                prev_logits = outputs.logits[0, -1, :]
                fused = self._soft_fusion_embedding(prev_hidden, prev_logits)
                new_token_embed = self._apply_transform(fused).unsqueeze(1)
            else:
                ## 阶段1: 直接使用隐藏状态作为嵌入
                new_token_embed = self._apply_transform(prev_hidden).unsqueeze(1)
        else:
            ## 普通文本标记：直接查嵌入表
            new_token_embed = self.embedding(
                torch.tensor(next_token, device=input_ids.device)
            ).view(1, 1, -1)

        new_inputs_embeds = torch.cat((inputs_embeds, new_token_embed), dim=1)

        ## 自回归循环生成后续标记
        for _ in range(max_new_tokens - 1):
            outputs = self.base_causallm(
                inputs_embeds=new_token_embed,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
            )
            self.gen_forward_cnt += 1
            if without_thinking_token:
                outputs.logits[0, -1, self.thinking_token_id] = float('-inf')
            next_token = torch.argmax(outputs.logits[0, -1]).item()
            if next_token == self.eos_token_id:
                break
            tokens.append(next_token)
            past_key_values = outputs.past_key_values
            selected_hidden_state = self._select_hidden_state(outputs.hidden_states)
            ## 同理：如果生成 <thinking>，根据阶段模式构建嵌入
            if next_token == self.thinking_token_id:
                prev_hidden = selected_hidden_state[:, -1, :]
                if self.stage_mode == "soft_fusion":
                    prev_logits = outputs.logits[0, -1, :]
                    fused = self._soft_fusion_embedding(prev_hidden, prev_logits)
                    new_token_embed = self._apply_transform(fused).unsqueeze(1)
                else:
                    new_token_embed = self._apply_transform(prev_hidden).unsqueeze(1)
            else:
                new_token_embed = self.embedding(
                    torch.tensor(next_token, device=input_ids.device)
                ).view(1, 1, -1)

            new_inputs_embeds = torch.cat((new_inputs_embeds, new_token_embed), dim=1)

        if synced_gpus:
            # in FSDP, the number of forward pass need to be the same across devices
            while self.gen_forward_cnt < max_new_tokens + MAX_THINKING_AUTOREGRESS:
                self.gen_forward_cnt += 1
                _ = self.base_causallm(
                    inputs_embeds=new_token_embed,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

        generated_tokens = torch.tensor(
            tokens, dtype=input_ids.dtype, device=input_ids.device
        ).view(1, -1)

        if output_embedding:
            # for analysis purpose
            return generated_tokens, new_inputs_embeds

        return generated_tokens
