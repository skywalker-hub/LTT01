<div align="center">

# Latent Thoughts Tuning

### Bridging Context and Reasoning with Fused Information in Latent Tokens

[![arXiv](https://img.shields.io/badge/arXiv-2602.10229-b31b1b.svg)](https://arxiv.org/abs/2602.10229)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-ee4c2c.svg)](https://pytorch.org/)

[Paper](https://arxiv.org/abs/2602.10229) | [Code](https://github.com/NeosKnight233/Latent-Thoughts-Tuning)

</div>

---

## :sparkles: Overview

**LT-Tuning** is a post-training framework that enables LLMs to reason in continuous latent space without external assistant models. Instead of verbalizing every intermediate step as text tokens (explicit CoT), our method allows models to dynamically interleave text and latent `<thinking>` tokens through **confidence-driven insertion** and **Context-Prediction Fusion**.

<p align="center">
  <img src="plots/fig1.pdf" width="70%" alt="Comparison of reasoning paradigms"/>
</p>
<!-- Replace with actual figure path (e.g., assets/fig1.png) -->

### :dart: Key Contributions

- **Context-Prediction Fusion** — Constructs latent tokens by fusing contextual hidden states with predictive semantic guidance from the vocabulary embedding space, mitigating feature collapse.
- **Confidence-Driven Dynamic Switching** — Adaptively decides when to engage latent reasoning vs. explicit text generation based on prediction confidence.
- **Three-Stage Curriculum Learning** — Progressively transitions from explicit CoT to latent reasoning for stable optimization.

---

## :chart_with_upwards_trend: Results

LT-Tuning achieves state-of-the-art performance across all model scales (1B / 3B / 8B) on mathematical reasoning benchmarks, trained on GSM8K and evaluated on four test sets.

### Main Results

| Model | Method | GSM8K-NL | ASDiv-Aug | MultiArith | SVAMP | Avg |
|:------|:-------|:--------:|:---------:|:----------:|:-----:|:---:|
| **Llama-3.2-1B** | Explicit CoT | 14.9 | 44.8 | 37.8 | 22.3 | 29.9 |
| | Coconut | 14.6 | 42.5 | 22.8 | 21.0 | 25.2 |
| | SoftCoT | 14.9 | **54.1** | 38.9 | 25.0 | 33.2 |
| | **LT-Tuning (Ours)** | **15.8** | 53.9 | **51.7** | 24.3 | **36.4** |
| **Llama-3.2-3B** | Explicit CoT | 29.5 | 69.8 | 57.2 | 45.7 | 50.5 |
| | Coconut | 31.8 | 61.9 | 63.3 | 44.0 | 50.3 |
| | **LT-Tuning (Ours)** | **32.1** | 67.2 | **64.4** | **45.7** | **52.4** |
| **Llama-3.1-8B** | Explicit CoT | 49.5 | 69.6 | 78.3 | 49.3 | 61.7 |
| | Soft-Thinking | 53.1 | 74.9 | 85.0 | 51.0 | 66.0 |
| | Coconut | 32.7 | 38.8 | 51.7 | 43.0 | 41.5 |
| | **LT-Tuning + Adapter (Ours)** | **58.5** | 70.7 | **96.1** | **55.7** | **70.3** |

> LT-Tuning achieves up to **+4.3%** average improvement over the strongest baseline. Notably, Coconut degrades severely at 8B scale due to feature collapse, while LT-Tuning exhibits robust scaling.

### Feature Collapse Mitigation

<p align="center">
  <img src="plots/pca_3d_positions_grid.pdf" width="80%" alt="PCA visualization of latent token embeddings"/>
</p>
<!-- Replace with actual figure path (e.g., assets/pca_visualization.png) -->

PCA visualization of latent token embeddings on Llama-3.1-8B. **Coconut** (green) collapses after 2 steps; **LT-Tuning** (red) maintains semantic diversity across all reasoning steps.

---

## :building_construction: Method

<p align="center">
  <img src="plots/method_final.pdf" width="95%" alt="LT-Tuning Framework"/>
</p>
<!-- Replace with actual figure path (e.g., assets/method.png) -->

LT-Tuning uses a three-stage curriculum:

| Stage | Name | Description |
|:-----:|:-----|:------------|
| 1 | **Explicit CoT Warm-up** | Standard SFT on Chain-of-Thought data to build reasoning foundations |
| 2 | **Dynamic Latent Generation** | Confidence-driven `<thinking>` token insertion; hidden states as initial latent embeddings |
| 3 | **Context-Prediction Fusion** | Fuses contextual hidden states with probability-weighted vocabulary embeddings: `e_fusion = α · h_ctx + (1-α) · e_pred` |

---

## :rocket: Getting Started

### Prerequisites

- Python >= 3.9
- PyTorch >= 2.7
- CUDA 12.x with 4x NVIDIA A100 80GB (or equivalent)

### Installation

```bash
git clone https://github.com/NeosKnight233/Latent-Thoughts-Tuning.git
cd Latent-Thoughts-Tuning
pip install -r requirements.txt
```

### :file_folder: Data Preparation

Prepare JSONL training data with the following format:

```json
{"question": "...", "answer": "42", "reasoning_chain": "Step 1: ... Step 2: ..."}
```

Place your data files in the `data/` directory and update paths in the config file.

### :gear: Configuration

All training hyperparameters are managed via a single YAML config file. See [`configs/example_config.yaml`](configs/example_config.yaml) for a full example.

Key parameters:

```yaml
# Model
model_name_or_path: meta-llama/Llama-3.2-1B

# Three-stage curriculum
stage_epochs: [1, 2, 7]          # epochs per stage
stage_modes: [common, hidden_state, soft_fusion]

# Confidence-driven insertion
thinking_strategy: confidence
reinforce_prob_threshold: [0.0, 0.3, 0.2]

# Context-Prediction Fusion
fusion_alpha: [0.5, 0.5, 0.6]   # weight for hidden state component
fusion_top_p: 0.9
fusion_temperature: 1.0
```

### :weight_lifting: Training

Launch multi-GPU training with DeepSpeed:

```bash
# Edit configs/example_config.yaml to set your model, data paths, and hyperparameters
bash scripts/train.sh
```

The training script (`run.py`) handles all three stages automatically via the `StageManager`. Stage transitions, dataset regeneration, and model config updates happen through callbacks — no manual intervention is needed.

<details>
<summary><b>Custom launch command</b></summary>

```bash
deepspeed --num_gpus 4 run.py configs/your_config.yaml
```

</details>

### :test_tube: Evaluation

Evaluate a trained model on all benchmarks (GSM8K-NL, ASDiv-Aug, MultiArith, SVAMP):

```bash
bash scripts/eval_LT_Tuning.sh
```

<details>
<summary><b>Custom evaluation</b></summary>

```bash
# Evaluate on specific datasets
torchrun --nproc_per_node=4 eval/eval_LT_Tuning.py configs/your_config.yaml \
    --datasets gsm8k asdiv multiarith svamp
```

</details>

---

## :open_file_folder: Project Structure

```
Latent-Thoughts-Tuning/
├── run.py                  # Main training entry point
├── model.py                # LT_Tuning_Model with fusion mechanism
├── dataset.py              # Data processing & thinking strategies
├── utils.py                # StageManager, Config, utilities
├── configs/
│   ├── example_config.yaml # Full training config template
│   └── ds_config_zero2.json# DeepSpeed ZeRO-2 config
├── scripts/
│   ├── train.sh            # Training launch script
│   └── eval_LT_Tuning.sh  # Evaluation launch script
├── eval/
│   ├── eval_LT_Tuning.py  # Multi-dataset evaluation
│   ├── dataset.py          # Benchmark data loading
│   └── utils.py            # Answer extraction & matching
└── data/                   # Training & evaluation data
```

---

## :page_facing_up: Citation

If you find this work useful, please cite our paper:

```bibtex
@article{liu2026latent,
  title={Latent Thoughts Tuning: Bridging Context and Reasoning with Fused Information in Latent Tokens},
  author={Liu, Weihao and Min, Dehai and Cheng, Lu},
  journal={arXiv preprint arXiv:2602.10229},
  year={2026}
}
```

## :balance_scale: License

This project is licensed under the [MIT License](LICENSE).
