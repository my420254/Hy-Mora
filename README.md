
# Hy-Mora: Hybrid Pooling and Memory-Augmented LoRA for Class-Imbalanced Sentiment Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PEFT](https://img.shields.io/badge/PEFT-State%20of%20the%20Art-green)](https://github.com/huggingface/peft)

This repository contains the official implementation of the paper **"Hy-Mora: Hybrid Pooling and Memory-Augmented LoRA for Class-Imbalanced Sentiment Analysis"**.

## 🌟 Overview

**Hy-Mora** is a parameter-efficient fine-tuning (PEFT) framework designed to address the "Representation Collapse" problem in long-tailed sentiment analysis tasks. While Large Language Models (LLMs) are powerful, they often exhibit severe majority-class bias in few-shot settings. Hy-Mora solves this via two core mechanisms:

1.  **Adaptive Hierarchical State Pooling (AHSP):** Aggregates multi-granular features (Salient, Contextual, Attentive) to mitigate the information bottleneck of the single `[CLS]` token.
2.  **Tail-aware Prototype Memory Bank (TPMB):** A dynamic, non-parametric memory reservoir that enforces "Margin Expansion" in the feature space, explicitly pushing minority classes away from majority clusters.

## 📂 Repository Structure

The code is organized to be self-contained and easy to reproduce.

| File | Description |
| :--- | :--- |
| `exp_smp2020_comparison.ipynb` | Training & Evaluation on the **SMP2020-EWECT** (Chinese) dataset. Includes Hy-Mora vs. DoRA/LoRA comparisons. |
| `exp_sst5_comparison.ipynb` | Training & Evaluation on the **SST-5** (English) dataset. |
| `exp_tweeteval_comparison.ipynb` | Training & Evaluation on the **TweetEval** (English) dataset. |
| `exp_llm_comparison.py` | Script to run **Zero-Shot & 3-Shot In-Context Learning** benchmarks using **Qwen-2.5-7B** and **Llama-3.1-8B**. |
| `requirements.txt` | List of dependencies required to run the experiments. |

## 🚀 Quick Start

### 1. Installation

Clone this repository and install the required packages:

git clone [https://github.com/YourUsername/Hy-Mora.git](https://github.com/YourUsername/Hy-Mora.git)
cd Hy-Mora
pip install -r requirements.txt


### 2\. Reproducing Main Results (Fine-tuning)

To reproduce the main comparative results (Table 2 in the paper) for each dataset, simply run the corresponding Jupyter Notebook:

  * **SMP2020:** Open and run `exp_smp2020_comparison.ipynb`
  * **SST-5:** Open and run `exp_sst5_comparison.ipynb`
  * **TweetEval:** Open and run `exp_tweeteval_comparison.ipynb`

Each notebook contains the full pipeline: Data Loading -\> Model Initialization (Hy-Mora) -\> Training -\> Evaluation -\> Visualization.

### 3\. Reproducing LLM Baselines

To reproduce the comparison with Large Language Models (Table 3 in the paper), run the python script:


python exp_llm_comparison.py


This script will:

1.  Load Qwen-2.5-7B and Llama-3.1-8B (via `unsloth` for efficiency).
2.  Perform evaluation on strict validation sets under both **Zero-Shot** and **3-Shot** settings.
3.  Save the results to `exp_llm_comparison_results.csv`.

## 📊 Datasets

This repository relies on public benchmarks available via the [Hugging Face Datasets](https://huggingface.co/docs/datasets/index) library.

**You do NOT need to manually download these files.** The provided scripts will automatically download, cache, and process the data upon the first run.

### Data Sources

We use the following official repositories as our data sources:

| Dataset | Hugging Face ID | Description |
| :--- | :--- | :--- |
| **SMP2020-EWECT** | `Um1neko/smp2020` | Chinese Social Media Emotion Classification (6 classes). |
| **SST-5** | `SetFit/sst5` | Stanford Sentiment Treebank, fine-grained (5 classes). |
| **TweetEval** | `tweet_eval` (sentiment) | Twitter Sentiment Analysis (3 classes). |

### Data Pre-processing & Splitting

To reproduce the **Class-Imbalanced** settings described in the paper ("Config A" and "Config B"), our scripts apply **Stratified Sampling** on the fly:

  * **Training Sets:** We construct long-tailed distributions by down-sampling specific classes to match the imbalance ratios (e.g., 20:1, 30:1) detailed in **Table 1** of the paper.
  * **Validation Sets:** We use strictly balanced validation sets (80-100 samples per class) to ensure unbiased evaluation metrics.

*Note: See the `get_validation_set` and data loading functions in the notebooks for the exact splitting logic.*

## 📝 Citation

If you find this code or our paper useful for your research, please cite:

```bibtex
@article{Zhang2025HyMora,
  title={Hy-Mora: Hybrid Pooling and Memory-Augmented LoRA for Class-Imbalanced Sentiment Analysis},
  author={Zhang, Mengyang and [Co-author Name] and [Co-author Name]},
  journal={arXiv preprint},
  year={2025}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.




