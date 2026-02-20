from functools import partial
import json
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import os
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from tqdm import tqdm
from utils import apply_chat_template_if_needed, print_list_in_json

def is_distributed():
    try:
        import torch.distributed as dist
        return dist.is_initialized()
    except:
        return False

def get_rank():
    if is_distributed():
        import torch.distributed as dist
        return dist.get_rank()
    return 0
class ThinkingTokenStrategy(ABC):
    """
    <thinking> 标记插入策略的抽象基类。
    论文4.2节描述了基于置信度的动态标记插入机制，
    本基类定义了所有策略共享的接口和辅助方法。
    子类包括:
      - RandomThinkingStrategy: 随机插入（基线方法）
      - ArithmeticThinkingStrategy: 在数字/运算符位置插入
      - ConfidenceThinkingStrategy: 基于模型预测置信度插入（对应论文公式5）
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        thinking_token_id: int,
        tokens_per_stage: Union[float, int] = 1,     # 每阶段插入的标记数量
        insertion_prob: float = 1.0,                  # 在候选位置实际插入的概率
        secondary_insertion_prob: float = 0.0,        # 连续插入第二个 <thinking> 的概率
        seed: Optional[int] = None,
    ) -> None:

        self.tokenizer = tokenizer
        self.thinking_token_id = thinking_token_id
        self.tokens_per_stage = tokens_per_stage
        self.insertion_prob = insertion_prob
        self.secondary_insertion_prob = secondary_insertion_prob
        self._seed = seed
        self.generator = random.Random(seed) if seed is not None else random.Random()
        self.requires_serial_processing = False

    @abstractmethod
    def _candidate_indices(
        self, sample: Dict[str, Any]
    ) -> List[int]:
        """返回候选插入位置的索引列表（0-based）。"""

    def _build_rng(
        self, sample: Dict[str, Any], scheduled_stage: int = 1
    ) -> random.Random:

        if self._seed is None:
            return self.generator

        sample_idx = int(sample.get("idx", 0))
        offset = 23
        derived_seed = (
            1000003 * sample_idx
            + 9176 * scheduled_stage
            + offset
            + self._seed
        ) & 0xFFFFFFFF
        return random.Random(derived_seed)

    def _select_indices(
        self,
        candidates: List[int],
        scheduled_stage: int = 1,
        rng: Optional[random.Random] = None,
    ) -> List[int]:

        if scheduled_stage <= 0 or not candidates:
            return []
        if self.tokens_per_stage < 1:
            target_n = int(len(candidates) * self.tokens_per_stage)
        else:
            target_n = int(self.tokens_per_stage * scheduled_stage)
        if target_n <= 0:
            return []

        target_n = min(target_n, len(candidates))
        if target_n < len(candidates):
            return sorted(rng.sample(candidates, target_n))
        return sorted(candidates[:target_n])

    def apply(
        self,
        sample: Dict[str, Any],
        scheduled_stage: int = 1,
        candidate_indices: Optional[List[int]] = None,
    ) -> Tuple[List[int], List[int]]:
        """
        在候选位置插入 <thinking> 标记，构建混合序列（文本标记与潜在标记交替）。
        对应论文4.2节的数据构建过程。
        返回: (插入 <thinking> 后的 token 序列, 插入位置列表)
        """

        response = sample.get("response_tokenized", [])
        input_tokens = sample.get("question_tokenized", [])
        if len(response) == 0:
            return response, []
        rng = self._build_rng(sample, scheduled_stage)
        ## 使用预计算的候选索引（ConfidenceThinkingStrategy 批量处理时传入）
        if candidate_indices is not None:
            candidates = candidate_indices
        else:
            candidates = self._candidate_indices(sample=sample)

        selected = self._select_indices(
            candidates, scheduled_stage, rng=rng
        )
        if not selected:
            return sample.get("full_tokenized", []), []

        ## 在选定的候选位置逐一插入 <thinking> 标记
        inserted_positions: List[int] = []
        offset = 0
        updated_input_ids = list(sample.get("full_tokenized", []))
        for idx in selected:
            ## 以 insertion_prob 的概率决定是否在此位置实际插入
            if rng.random() > self.insertion_prob:
                continue
            if idx < len(input_tokens):
                print("********************************Warning: Thinking token inserted in input tokens region.********************************")
            
            insert_at = idx + offset

            if insert_at < 0:
                insert_at = 0

            ## 插入一个 <thinking> 标记
            updated_input_ids.insert(insert_at, self.thinking_token_id)
            inserted_positions.append(insert_at)
            offset += 1

            ## 以 secondary_insertion_prob 的概率在同一位置连续插入第二个 <thinking>
            ## 这允许模型在特别困难的位置进行更多步的潜在推理
            if (
                self.secondary_insertion_prob > 0.0
                and rng.random() < self.secondary_insertion_prob
            ):
                insert_at += 1
                updated_input_ids.insert(insert_at, self.thinking_token_id)
                inserted_positions.append(insert_at)
                offset += 1

        return updated_input_ids, inserted_positions

class RandomThinkingStrategy(ThinkingTokenStrategy):

    def _candidate_indices(
        self, flat_steps: Sequence[int], sample: Dict[str, Any]
    ) -> List[int]:

        # Every token can be a potential insertion point (except the first one)
        return list(range(1, len(flat_steps)))


class ArithmeticThinkingStrategy(ThinkingTokenStrategy):

    OPERATORS = set("+-=*/%()")

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        thinking_token_id: int,
        tokens_per_stage: int = 1,
        insertion_prob: float = 1.0,
        secondary_insertion_prob: float = 0.0,
        seed: Optional[int] = None,
        operator_regex: Optional[str] = None,
    ) -> None:

        super().__init__(
            tokenizer,
            thinking_token_id,
            tokens_per_stage=tokens_per_stage,
            insertion_prob=insertion_prob,
            secondary_insertion_prob=secondary_insertion_prob,
            seed=seed,
        )

        self._operator_regex = re.compile(operator_regex) if operator_regex else None

    def _is_numeric_or_operator(self, text: str) -> bool:

        stripped = text.strip()
        if not stripped:
            return False

        if self._operator_regex and self._operator_regex.search(stripped):
            return True

        if any(char in self.OPERATORS for char in stripped):
            return True

        return stripped.replace(".", "", 1).isdigit()

    def _candidate_indices(
        self, sample: Dict[str, Any]
    ) -> List[int]:
        candidates: List[int] = []
        input_length = len(sample.get("question_tokenized", []))
        full_length = len(sample.get("full_tokenized", []))
        for idx in range(input_length, full_length):
            token_id = sample["full_tokenized"][idx]
            token_text = self.tokenizer.decode([token_id])
            if self._is_numeric_or_operator(token_text):
                candidates.append(idx)
            if token_text.strip() == "###":
                candidates.append(idx+1)
        return candidates


class ConfidenceThinkingStrategy(ThinkingTokenStrategy):
    """
    【论文4.2节 - 基于置信度的数据构建】
    对应论文公式(5):
      若 p_θ(y_t | y_{<t}) < τ，则在位置 t 插入 <thinking> 占位符
    该策略通过模型前向传播计算每个位置的预测置信度，
    在低置信度位置（模型不确定的地方）插入 <thinking> 标记。
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        thinking_token_id: int,
        model: Optional[torch.nn.Module],         # 用于计算置信度的模型实例
        tokens_per_stage: Union[float, int] = 1,
        insertion_prob: float = 1.0,
        secondary_insertion_prob: float = 0.0,
        seed: Optional[int] = None,
        probability_threshold: float = 0.05,       # 论文公式(5)中的阈值 τ
        max_sequence_length: int = 2048,
    ) -> None:

        super().__init__(
            tokenizer,
            thinking_token_id,
            tokens_per_stage=tokens_per_stage,
            insertion_prob=insertion_prob,
            secondary_insertion_prob=secondary_insertion_prob,
            seed=seed,
        )

        if model is None:
            raise ValueError(
                "ConfidenceThinkingStrategy requires a model instance for probability estimation."
            )

        self.model = model
        self.model_device = next(model.parameters()).device
        ## 论文公式(5)中的置信度阈值 τ
        self.threshold = float(max(min(probability_threshold, 1.0), 0.0))
        self.max_sequence_length = max_sequence_length
        ## 需要串行处理，因为依赖 GPU 推理计算置信度
        self.requires_serial_processing = True

    def batch_candidate_indices(
        self, samples: List[Dict[str, Any]]
    ) -> List[List[int]]:
        """
        【论文公式(5)的核心实现 - 批量计算候选插入位置】
        对一批样本进行前向传播，计算每个位置的预测概率 p_θ(y_t | y_{<t})，
        找出概率低于阈值 τ 的位置作为 <thinking> 插入候选位置。
        """
        
        if not samples:
            return []
        
        ## 构建 batch 输入：将问题和完整回答（推理链+答案）拼接为对话格式
        batch_messages = [
            [
                {"role": "user", "content": sample["question"]},
                {"role": "assistant", "content": sample["reasoning_chain"] + "\n### " + sample["answer"]}
            ] for sample in samples
        ]
        batch_input_ids = [
            self.tokenizer.encode(
                apply_chat_template_if_needed(self.tokenizer, message),
                add_special_tokens=False,
            )
            for message in batch_messages
        ]
        batch_input_ids = [torch.tensor(ids, dtype=torch.long) for ids in batch_input_ids]
        padded_sequences = torch.nn.utils.rnn.pad_sequence(
            batch_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id, padding_side="right"
        )
        batch_input = {
            "input_ids": padded_sequences.to(self.model_device),
            "attention_mask": (padded_sequences != self.tokenizer.pad_token_id).long().to(self.model_device),
        }
        
        ## 使用基础模型进行前向传播，获取每个位置的 logits
        with torch.no_grad():
            self.model.eval()
            outputs = self.model(**batch_input)
            logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)
        
        ## 逐样本处理，计算每个位置的预测置信度
        all_candidates = [[] for _ in range(len(samples))]
        for idx, sample in enumerate(samples):
            ## 计算问题部分的长度，<thinking> 只插入在回答(response)区域
            user_msg = [{"role": "user", "content": sample["question"]}]
            user_text = apply_chat_template_if_needed(self.tokenizer, user_msg)
            question_tokens = self.tokenizer.encode(user_text, add_special_tokens=False)
            question_len = len(question_tokens)
            ## 计算每个位置对真实下一个标记的预测概率 p_θ(y_t | y_{<t})
            next_tokens = batch_input_ids[idx][1:].to(self.model_device)
            log_probs = F.log_softmax(logits[idx, : len(batch_input_ids[idx]), :], dim=-1)
            token_log_probs = log_probs.gather(1, next_tokens.unsqueeze(-1)).squeeze(-1)
            token_probs = token_log_probs.exp().tolist()
            ## 论文公式(5): 若 p_θ(y_t | y_{<t}) < τ，则标记为低置信度位置
            ## 只在回答区域（question_len 之后）检查，不在问题区域插入
            for pos in range(question_len, len(batch_input_ids[idx])):
                if token_probs[pos - 1] < self.threshold:
                    all_candidates[idx].append(pos)

        return all_candidates

    def _candidate_indices(
        self, sample: Dict[str, Any]
    ) -> List[int]:
        """Find candidate indices for a single sample by calling batch processing with batch size 1."""
        results = self.batch_candidate_indices([sample])
        return results[0] if results else []


def build_thinking_strategy(
    configs: Any,
    tokenizer: PreTrainedTokenizerBase,
    thinking_token_id: int,
    model: Optional[torch.nn.Module] = None,
) -> ThinkingTokenStrategy:
    """
    根据配置构建 <thinking> 标记插入策略。
    论文默认使用 'confidence'（置信度策略），对应论文4.2节的公式(5)。
    """

    strategy_name = getattr(configs, "thinking_strategy", "random").lower()
    tokens_per_stage = getattr(
        configs,
        "tokens_per_stage",
        getattr(configs, "c_thought", 1),
    )
    insertion_prob = getattr(configs, "thinking_insertion_prob", 1.0)
    secondary_prob = getattr(configs, "thinking_secondary_insertion_prob", 0.0)
    seed = getattr(configs, "seed", None)

    def _resolve_stage_value(value, default):
        if isinstance(value, (list, tuple)):
            return value[0] if value else default
        return value

    if strategy_name == "arithmetic":
        operator_regex = getattr(configs, "thinking_operator_regex", None)
        return ArithmeticThinkingStrategy(
            tokenizer,
            thinking_token_id,
            tokens_per_stage=tokens_per_stage,
            insertion_prob=insertion_prob,
            secondary_insertion_prob=secondary_prob,
            seed=seed,
            operator_regex=operator_regex,
        )

    if strategy_name in {"confidence","reinforce", "policy"}:
        prob_threshold = _resolve_stage_value(
            getattr(configs, "reinforce_prob_threshold", 0.05), 0.05
        )
        max_eval_len = int(
            _resolve_stage_value(
                getattr(configs, "reinforce_max_eval_length", 2048), 2048
            )
        )
        return ConfidenceThinkingStrategy(
            tokenizer,
            thinking_token_id,
            model=model,
            tokens_per_stage=tokens_per_stage,
            insertion_prob=insertion_prob,
            secondary_insertion_prob=secondary_prob,
            seed=seed,
            probability_threshold=prob_threshold,
            max_sequence_length=max_eval_len,
        )

    if strategy_name != "random":
        raise ValueError(f"Unsupported thinking strategy: {strategy_name}")

    return RandomThinkingStrategy(
        tokenizer,
        thinking_token_id,
        tokens_per_stage=tokens_per_stage,
        insertion_prob=insertion_prob,
        secondary_insertion_prob=secondary_prob,
        seed=seed,
    )


def get_dataset(path, dataset_name="gsm8k", max_size=1000000000):
    """Load raw dataset without tokenization.
    
    Tokenization will be done later in get_question_latent_dataset or 
    get_cot_latent_dataset to allow for different tokenization strategies.
    """
    if path.endswith(".jsonl"):
        data = [json.loads(line) for line in open(path)][:max_size]
    else:
        data = json.load(open(path))[:max_size]
    def extract_reasoning_chain_from_answer(answer: str) -> str:
        """Extract reasoning chain from the answer field."""
        return answer.split("####")[0].strip()

    def extract_answer_from_answer_field(answer: str) -> str:
        """Extract final answer from the answer field."""
        return answer.split("####")[-1].strip()
    
    if dataset_name == "gsm8k":
        data = [
            {
                "idx": idx,
                "question": d["question"],
                "reasoning_chain": "\n".join(d["steps"]) if "steps" in d else extract_reasoning_chain_from_answer(d["answer"]),
                "answer": d["answer"] if d["answer"].startswith("###") else extract_answer_from_answer_field(d["answer"]),
            } 
             for idx, d in enumerate(data)
        ]
    else:
        raise ValueError(f"Unsupported Dataset Now: {dataset_name}")
    
    keys = data[0].keys()
    dataset = Dataset.from_dict({k: [d[k] for d in data] for k in keys})
    
    return dataset


@dataclass
class MyCollator:

    tokenizer: PreTrainedTokenizerBase
    thinking_id: Optional[int] = None
    label_pad_token_id: Optional[int] = -100

    def __call__(self, features, return_tensors=None):

        assert self.tokenizer.padding_side == "right"

        """
        Pad the batch like this
        E.g.,
        
        xxxxxxxxxx<thinking><thinking>xxxxx--
        -----xxxxx<thinking>xxxxxxxx-------
        ---xxxxxxx<thinking><thinking>xxxxxxx


        ("x" is word token, "-" is pad token)
        """
        if self.thinking_id is not None:
            earliest_thinking = [
                feature["input_ids"].index(self.thinking_id)
                for feature in features
                if self.thinking_id in feature["input_ids"]
            ]

            if (
                len(earliest_thinking) > 0
            ):  # if there are continuous thoughts in the sequence
                latest_earliest_thinking = max(earliest_thinking)
                for feature in features:
                    if self.thinking_id in feature["input_ids"]:
                        n_tok_pad = latest_earliest_thinking - feature["input_ids"].index(
                            self.thinking_id
                        )
                    else:
                        n_tok_pad = 0
                    feature["position_ids"] = [0] * n_tok_pad + list(
                        range(len(feature["input_ids"]))
                    )
                    feature["input_ids"] = [
                        self.tokenizer.pad_token_id
                    ] * n_tok_pad + feature["input_ids"]
                    if "labels" in feature:
                        feature["labels"] = [
                            self.label_pad_token_id
                        ] * n_tok_pad + feature["labels"]
                    feature["attention_mask"] = [
                        0
                    ] * n_tok_pad + feature["attention_mask"]

        return_tensors = "pt"

        label_name = "label" if "label" in features[0].keys() else "labels"

        non_label_position_features = [
            {
                k: v
                for k, v in feature.items()
                if k != label_name and k != "position_ids" and k != "thinking_positions"
            }
            for feature in features
        ]

        # run through tokenizer without labels to ensure no side effects
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            non_label_position_features,
            padding=True,
            pad_to_multiple_of=None,
            return_tensors=return_tensors,
        )

        labels = (
            [feature[label_name] for feature in features]
            if label_name in features[0].keys()
            else None
        )
        if labels is not None and all(label is None for label in labels):
            labels = None
        position_ids = (
            [feature["position_ids"] for feature in features]
            if "position_ids" in features[0].keys()
            else None
        )
        # we have to pad the labels and position_ids manually as we cannot rely on `tokenizer.pad`

        if labels is not None:
            max_label_length = max(len(l) for l in labels)

            batch["labels"] = [
                label + [self.label_pad_token_id] * (max_label_length - len(label))
                for label in labels
            ]
            batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)

        if position_ids is not None:
            max_pos_length = max(len(l) for l in position_ids)

            batch["position_ids"] = [
                position_id + [0] * (max_pos_length - len(position_id))
                for position_id in position_ids
            ]
            batch["position_ids"] = torch.tensor(
                batch["position_ids"], dtype=torch.int64
            )

        return batch


def get_question_dataset(
    stage_type: str,
    base_dataset_valid,
    tokenizer: PreTrainedTokenizerBase,
):
    """Generate question dataset with tokenization."""

    def process_dataset(sample):
        # Apply chat template if available, otherwise use plain text
        message = [
            {
                "role": "user",
                "content": sample["question"],
            }
        ]

        question = apply_chat_template_if_needed(
            tokenizer, message
        )
        question_tokenized = tokenizer.encode(question, add_special_tokens=False)

        return {
            "input_ids": question_tokenized,
            "idx": sample["idx"],
            "attention_mask": [1] * len(question_tokenized),
            "position_ids": list(range(len(question_tokenized))),
        }

    return base_dataset_valid.map(
        process_dataset, remove_columns=list(base_dataset_valid.features), num_proc=32
    )


def get_cot_latent_dataset(
    stage_type: str,
    base_dataset,
    configs,
    strategy: ThinkingTokenStrategy,
    tokenizer: PreTrainedTokenizerBase,
    shuffle: bool = False,
    return_text: bool = False,
    debug_num: Optional[int] = None,
):
    """
    构建 CoT + 潜在标记的训练数据集。
    对应论文4.2节的"基于置信度的数据构建"：
      - 阶段0 (common): 仅使用显式 CoT 数据（不插入 <thinking>）
      - 阶段1/2: 使用 strategy 在低置信度位置插入 <thinking>，
        构建混合序列（文本标记与潜在标记交替出现）
    """
    thinking_answer_prob = getattr(configs, "thinking_answer_prob", 0.0)
    thinking_answer_extra_prob = getattr(
        configs, "thinking_answer_extra_prob", 0.0
    )
    if isinstance(thinking_answer_prob, list):
        thinking_answer_prob = thinking_answer_prob[1]
        thinking_answer_extra_prob = thinking_answer_extra_prob[1]
        print(f"Using first element of thinking_answer_prob list: {thinking_answer_prob}")
        print(f"Using first element of thinking_answer_extra_prob list: {thinking_answer_extra_prob}")

    if debug_num is not None:
        base_dataset = base_dataset.select(range(debug_num))

    def process_dataset(sample, candidate_indices=None, strategy=None):
        """处理单个样本：构建 CoT 格式文本，分词后根据策略插入 <thinking> 标记"""
        ## 将推理链格式化为分步格式
        reasoning_list = sample['reasoning_chain'].split('\n')
        full_response = ""
        for idx in range(len(reasoning_list)):
            full_response += f'## Step {idx + 1}: {reasoning_list[idx]}\n'

        full_response += f"The final answer is:\n### {sample['answer']}"
        
        ## 构建对话格式: [用户问题, 助手回答(推理链+答案)]
        messages = [
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": full_response}
        ]
        
        ## 应用聊天模板并分词
        formatted_text = apply_chat_template_if_needed(tokenizer, messages)
        full_tokenized = tokenizer.encode(formatted_text, add_special_tokens=False)
        
        ## 计算用户问题部分的 token 长度，用于确定 labels 中哪些位置设为 -100（不计算损失）
        user_messages = [{"role": "user", "content": sample["question"]}]
        user_formatted = apply_chat_template_if_needed(tokenizer, user_messages)
        user_tokenized = tokenizer.encode(user_formatted, add_special_tokens=False)
        question_length = len(user_tokenized)

        question_tokenized = full_tokenized[:question_length]

        tokenized_sample = {
            'question': sample["question"],
            'reasoning_chain': sample['reasoning_chain'],
            'answer': sample['answer'],
            "question_tokenized": question_tokenized,
            "response_tokenized": full_tokenized[question_length:],
            "full_tokenized": full_tokenized
        }
        
        question_tokens = list(tokenized_sample["question_tokenized"])
        response_tokens = list(tokenized_sample["response_tokenized"])
        thinking_positions: List[int] = []
        
        ## 【关键步骤】根据策略在低置信度位置插入 <thinking> 标记
        ## 阶段0: strategy=None，不插入任何 <thinking>（纯 CoT 训练）
        ## 阶段1/2: 使用 ConfidenceThinkingStrategy 在 p_θ(y_t|y_{<t}) < τ 的位置插入
        if strategy is not None:
            decorated_input_ids, inserted_positions = strategy.apply(tokenized_sample, candidate_indices=candidate_indices)
            thinking_positions = inserted_positions
        else:
            decorated_input_ids = full_tokenized
        ## 添加结束标记
        tokens = decorated_input_ids + [tokenizer.eos_token_id]

        ## 构建 labels：问题部分设为 -100（不计算损失），只对回答部分计算损失
        ## 注意：<thinking> 位置的 label 是其后的正常文本标记，
        ## 这迫使模型在潜在空间中推理以正确预测下一个显式标记
        labels = (
            [-100] * len(question_tokens)
            + tokens[len(question_tokens):]
        )

        if return_text:
            label_tokens = [tok for tok in labels if tok != -100]
            return {
                "idx": sample["idx"],
                "input_text": tokenizer.decode(
                    tokens, skip_special_tokens=False
                ),
                "label_text": tokenizer.decode(
                    label_tokens, skip_special_tokens=False
                ),
            }
        dataset_save_path = getattr(configs, "dataset_save_path", None)
        if dataset_save_path is not None and get_rank() == 0:
            os.makedirs(dataset_save_path, exist_ok=True)
            stage_suffix = getattr(
                configs, "current_stage_label", stage_type
            )
            if configs.save_dataset or sample["idx"] < 50:
                with open(
                    f"{dataset_save_path}/{configs.name}_{stage_suffix}_dataset.jsonl",
                    "a",
                ) as f:
                    json_record = {
                        "idx": sample["idx"],
                        "input_text": tokenizer.decode(
                            tokens, skip_special_tokens=False
                        ),
                        "label_text": tokenizer.decode(
                            [tok for tok in labels if tok != -100], skip_special_tokens=False
                        )
                    }
                    f.write(json.dumps(json_record) + "\n")
        return {
            "input_ids": tokens,
            "labels": labels,
            "attention_mask": [1] * len(tokens),
            "idx": sample["idx"],
            "position_ids": list(range(len(tokens))),
            "thinking_positions": thinking_positions if thinking_positions else None,
        }

    ## 阶段0 (common)：不需要 GPU 推理，可并行处理；阶段1/2 使用置信度策略需要串行 GPU 推理
    if configs.current_stage_mode == "common" and stage_type != "pause" and strategy is None:
        map_num_proc = 32
        process_dataset = partial(process_dataset, strategy=None)
    elif strategy is not None and strategy.requires_serial_processing:
        ## ConfidenceThinkingStrategy 需要 GPU 推理，只能串行处理
        map_num_proc = None
        process_dataset = partial(process_dataset, strategy=strategy)
    else:
        map_num_proc = 32
        process_dataset = partial(process_dataset, strategy=strategy)
        
    print(f"Starting dataset processing for stage: {stage_type} with map_num_proc={map_num_proc}")
    if map_num_proc is not None:
        ## 并行模式：阶段0 或不需要 GPU 的策略，使用 HuggingFace datasets 的 map 并行处理
        print("Using map function for dataset processing.")
        if get_rank() == 0:
            processed_dataset = base_dataset.map(
                process_dataset,
                remove_columns=list(base_dataset.features),
                num_proc=map_num_proc,
            )
            if shuffle:
                processed_dataset = processed_dataset.shuffle()
            processed_dataset = [processed_dataset]
        else:
            processed_dataset = [None]
        if is_distributed():
            import torch.distributed as dist
            dist.broadcast_object_list(processed_dataset, src=0)
        dataset = processed_dataset[0]
    else:
        ## 串行模式：阶段1/2 使用 ConfidenceThinkingStrategy，
        ## 需要批量前向传播计算置信度，然后根据结果插入 <thinking>
        batch_size = configs.batch_size_eval if hasattr(configs, "batch_size_eval") else 8
        batch_start = 0
        print(f"Processing dataset in batches of size {batch_size} for stage: {stage_type}")
        processed_dataset = []
        for batch_start in tqdm(
            range(0, len(base_dataset), batch_size),
            desc=f"Processing dataset for stage: {stage_type}",
        ):
            batch_end = min(batch_start + batch_size, len(base_dataset))
            batch_samples = [
                base_dataset[i] for i in range(batch_start, batch_end)
            ]
            ## 批量计算每个样本中低置信度位置（论文公式5）
            candidate_indices = strategy.batch_candidate_indices(batch_samples)
            ## 在低置信度位置插入 <thinking> 标记
            for i, sample in enumerate(batch_samples):
                processed_sample = process_dataset(
                    sample, candidate_indices=candidate_indices[i]
                )
                processed_dataset.append(processed_sample)
        dataset = processed_dataset
    print(f"✓ Dataset processing completed for stage: {stage_type}")
    print(f"Dataset size: {len(dataset)} samples")
    return dataset