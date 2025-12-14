
````markdown
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

| File                             | Description                                                                                                     |
| :------------------------------- | :-------------------------------------------------------------------------------------------------------------- |
| `exp_smp2020_comparison.ipynb`   | Training & Evaluation on the **SMP2020-EWECT** (Chinese) dataset. Includes Hy-Mora vs. DoRA/LoRA comparisons.   |
| `exp_sst5_comparison.ipynb`      | Training & Evaluation on the **SST-5** (English) dataset.                                                       |
| `exp_tweeteval_comparison.ipynb` | Training & Evaluation on the **TweetEval** (English) dataset.                                                   |
| `exp_llm_comparison.py`          | Script to run **Zero-Shot & 3-Shot In-Context Learning** benchmarks using **Qwen-2.5-7B** and **Llama-3.1-8B**. |
| `requirements.txt`               | List of dependencies required to run the experiments.                                                           |

## 🚀 Quick Start

### 1. Installation

Clone this repository and install the required packages:

```bash
git clone [https://github.com/YourUsername/Hy-Mora.git](https://github.com/YourUsername/Hy-Mora.git)
cd Hy-Mora
pip install -r requirements.txt
````

### 2\. Reproducing Main Results (Fine-tuning)

To reproduce the main comparative results (Table 2 in the paper) for each dataset, simply run the corresponding Jupyter Notebook:

  * **SMP2020:** Open and run `exp_smp2020_comparison.ipynb`
  * **SST-5:** Open and run `exp_sst5_comparison.ipynb`
  * **TweetEval:** Open and run `exp_tweeteval_comparison.ipynb`

Each notebook contains the full pipeline: Data Loading -\> Model Initialization (Hy-Mora) -\> Training -\> Evaluation -\> Visualization.

### 3\. Reproducing LLM Baselines

To reproduce the comparison with Large Language Models (Table 3 in the paper), run the python script:

```bash
python exp_llm_comparison.py
```

This script will:

1.  Load Qwen-2.5-7B and Llama-3.1-8B (via `unsloth` for efficiency).
2.  Perform evaluation on strict validation sets under both **Zero-Shot** and **3-Shot** settings.
3.  Save the results to `exp_llm_comparison_results.csv`.

## 📊 Datasets

We use standard benchmarks available via Hugging Face Datasets. No manual download is required; the scripts handle data loading automatically.

  * **SMP2020-EWECT:** Chinese social media emotion classification (6 classes).
  * **SST-5:** Stanford Sentiment Treebank (5 classes).
  * **TweetEval:** Twitter sentiment analysis (3 classes).

*Note: For imbalanced learning, we employ stratified sampling to create the "Low Resource" (Config A) and "Medium Resource" (Config B) splits described in the paper.*

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

```


