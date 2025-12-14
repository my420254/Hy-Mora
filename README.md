
# Hy-Mora: Hybrid Pooling and Memory-Augmented LoRA for Class-Imbalanced Sentiment Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B-ee4c2c.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-blue)](https://huggingface.co/)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA%2FDoRA-green)](https://github.com/huggingface/peft)

This repository contains the official PyTorch implementation of the paper **"Hy-Mora: Hybrid Pooling and Memory-Augmented LoRA for Class-Imbalanced Sentiment Analysis"**.

## 📖 Abstract

**Hy-Mora** is a novel Parameter-Efficient Fine-Tuning (PEFT) framework specifically designed to mitigate the "Representation Collapse" problem in long-tailed sentiment analysis. While Low-Rank Adaptation (LoRA) is efficient, it often struggles with minority classes in highly imbalanced scenarios.

Hy-Mora addresses this via two synergistic mechanisms:
1.  **Adaptive Hierarchical Smart Pooling (AHSP):** Replaces the standard `[CLS]` token with a dynamic fusion of attentive, contextual, and maximum-salience features, capturing richer semantic signals.
2.  **Tail-aware Prototype Memory Bank (TPMB):** Incorporates a non-parametric memory module that stores class-specific prototypes. It utilizes a contrastive auxiliary loss to enforce **Margin Expansion** in the feature space, explicitly pushing minority class representations away from majority clusters.

## 📂 Repository Structure

The codebase is structured for reproducibility across multiple datasets and settings.

```text
Hy-Mora/
├── exp_smp2020_comparison.py   # Main Experiment: SMP2020-EWECT (Chinese, 6 classes)
├── exp_sst5_comparison.py      # Main Experiment: SST-5 (English, 5 classes)
├── exp_tweeteval_comparison.py # Main Experiment: TweetEval (English, 3 classes)
├── exp_llm_Qwen_zero-shot.py   # LLM Baseline: Qwen-2.5-7B (Zero-Shot)
├── exp_llm_Qwen_three-shot.py  # LLM Baseline: Qwen-2.5-7B (3-Shot In-Context Learning)
├── exp_llm_Llama_zero-shot.py  # LLM Baseline: Llama-3.1-8B (Zero-Shot)
├── exp_llm_Llama_three-shot.py # LLM Baseline: Llama-3.1-8B (3-Shot In-Context Learning)
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
└── README.md                   # Project documentation
````

## 🛠️ Installation

We recommend using a virtual environment (Conda or venv). The code requires PyTorch and Hugging Face libraries.

```bash
# Clone the repository
git clone [https://github.com/YourUsername/Hy-Mora.git](https://github.com/YourUsername/Hy-Mora.git)
cd Hy-Mora

# Install dependencies
pip install -r requirements.txt

**Key Dependencies:**

  * `torch>=2.6.0`
  * `transformers>=4.53.3`
  * `peft>=0.16.0`
  * `bitsandbytes` (Required for LLM quantization)

## 🚀 Usage & Reproduction

### 1\. Main Fine-Tuning Experiments (Hy-Mora vs. Baselines)

These scripts replicate the comprehensive benchmark comparison (Table 2 in the paper). They automatically run:

  * **Methods:** Full Fine-tuning, LoRA (Vanilla/Balanced), LoRA+Focal, LoRA+LDAM, DoRA, and **Hy-Mora (Ours)**.
  * **Ablation Studies:** Testing components without Memory Bank or without Smart Pooling.
  * **Sensitivity Analysis:** Experiments on Temperature and Loss Weights (saved to separate CSVs).

Run the script corresponding to the dataset you wish to evaluate:

```bash
# Run experiments on SMP2020 (Chinese)
python exp_smp2020_comparison.py

# Run experiments on SST-5 (English)
python exp_sst5_comparison.py

# Run experiments on TweetEval (English)
python exp_tweeteval_comparison.py
```

**Output:**

  * Results are saved to CSV files (e.g., `smp2020_results_IPM.csv`).
  * A markdown summary report (e.g., `*_Summary.md`) is automatically generated after execution.
  * Visualization data (.npz) and bad-case analysis (.csv) are saved in the `viz_data_*` directories.

### 2\. Large Language Model (LLM) Baselines

To compare the performance of Hy-Mora against SOTA LLMs using In-Context Learning (Zero-Shot & Few-Shot), run the following scripts. These scripts use 4-bit quantization via `bitsandbytes` for memory efficiency.

**Qwen-2.5-7B-Instruct:**

```bash
python exp_llm_Qwen_zero-shot.py   # Zero-Shot
python exp_llm_Qwen_three-shot.py  # 3-Shot
```

**Llama-3.1-8B-Instruct:**

```bash
python exp_llm_Llama_zero-shot.py  # Zero-Shot
python exp_llm_Llama_three-shot.py # 3-Shot
```

## 📊 Datasets

The scripts automatically download and cache datasets via the Hugging Face `datasets` library.

| Dataset | Language | Classes | Imbalance Info | Base Model |
| :--- | :--- | :--- | :--- | :--- |
| **SMP2020-EWECT** | Chinese | 6 | Tail Classes: `Fear(1)`, `Surprise(5)` | `hfl/chinese-macbert-base` |
| **SST-5** | English | 5 | Tail Classes: `Very Negative(0)`, `Negative(1)` | `roberta-base` |
| **TweetEval** | English | 3 | Tail Classes: `Negative(0)`, `Positive(2)` | `roberta-base` |

## 📝 Methodology Summary

The `UnifiedModel` class in our code implements the following architecture:

1.  **Encoder:** Uses a pre-trained Transformer (RoBERTa/MacBERT) with LoRA/DoRA adapters injected into Query/Key/Value projections.
2.  **Hierarchical Smart Pooling (HSP):**
      * Extracts `hidden_states` from the last layer.
      * Computes an attention-weighted sum (Attentive).
      * Computes a global mean (Contextual).
      * Computes global max-pooling (Salient).
      * Fuses these views using a learnable gating mechanism.
3.  **Memory Bank:**
      * Maintains a queue of prototype feature vectors for each class.
      * During training, computes a similarity-based contrastive loss to ensure minority class samples maintain distance from majority prototypes.

## 📜 Citation

If you find this code or our paper useful for your research, please cite:

```bibtex
@article{Zhang2025HyMora,
  title={Hy-Mora: Hybrid Pooling and Memory-Augmented LoRA for Class-Imbalanced Sentiment Analysis},
  author={Zhang, Mengyang},
  journal={arXiv preprint},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

Copyright (c) 2025 Mengyang Zhang.

