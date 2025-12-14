# ==============================================================================
# 🏆 UNIFIED LLM BENCHMARK SUITE: Zero-Shot & 3-Shot
# Files: exp_llm_comparison.py
# Models: Qwen/Qwen2.5-7B-Instruct, unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit
# Logic: Unified validation set + Standard Chat Template + Regular Expression Parsing
# ==============================================================================

import os
import gc
import re
import torch
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# 抑制 HuggingFace 的一些冗余警告
import warnings
warnings.filterwarnings("ignore")

# =========================== ⚙️ 全局配置 ===========================
SEED = 42
BATCH_SIZE = 1

# 定义实验任务列表
# 结构: (Model_Path, Model_Short_Name)
MODELS_TO_TEST = [
    ("Qwen/Qwen2.5-7B-Instruct", "Qwen-2.5-7B"),
    ("unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit", "Llama-3.1-8B")
]

DATASETS = ["SMP2020", "SST-5", "TweetEval"]

# =========================== 🛠️ 工具函数 ===========================

def set_seed(seed=SEED):
    """固定所有随机种子，确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_validation_set(dataset_name):
    """
    统一验证集加载逻辑 (绝对公平)
    保持了原始代码中的数据处理和采样逻辑
    """
    set_seed(SEED) # 确保每次采样一致
    
    if dataset_name == "SMP2020":
        # print(f"📚 Loading {dataset_name}...")
        ds = load_dataset("Um1neko/smp2020", split="train")
        df = pd.DataFrame(ds)
        if "content" in df.columns: df = df.rename(columns={"content": "text"})
        df = df.dropna(subset=["text", "label"])
        df["label"] = df["label"].astype(int)
        val_count = 80 
        
    elif dataset_name == "SST-5":
        # print(f"📚 Loading {dataset_name}...")
        ds = load_dataset("SetFit/sst5", split="train")
        df = pd.DataFrame(ds)
        if "sentence" in df.columns: df = df.rename(columns={"sentence": "text"})
        if "label_text" in df.columns: df = df.drop(columns=["label_text"])
        df = df[["text", "label"]].dropna()
        df["label"] = df["label"].astype(int)
        val_count = 100
        
    elif dataset_name == "TweetEval":
        # print(f"📚 Loading {dataset_name}...")
        ds = load_dataset("tweet_eval", "sentiment", split="train")
        df = pd.DataFrame(ds)
        df = df[["text", "label"]].dropna()
        df["label"] = df["label"].astype(int)
        val_count = 100
    
    # Stratified Split & Sampling
    try:
        _, val_pool = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)
    except ValueError:
        # Fallback if dataset is too small for stratification
        val_pool = df.sample(frac=0.2, random_state=SEED)

    num_labels = df['label'].nunique()
    sampled_dfs = []
    
    for label in range(num_labels):
        class_df = val_pool[val_pool['label'] == label]
        if len(class_df) > 0:
            n_samples = min(len(class_df), val_count)
            sampled_dfs.append(class_df.sample(n=n_samples, random_state=SEED))
    
    final_df = pd.concat(sampled_dfs).sample(frac=1, random_state=SEED).reset_index(drop=True)
    # print(f"✅ Validation Set ({dataset_name}): {len(final_df)} samples.")
    return final_df

def get_prompt_content(dataset_name, text, model_short_name, shot_mode):
    """
    统一 Prompt 构造工厂
    根据 模型(Qwen/Llama) 和 模式(Zero/3-Shot) 返回对应的 Prompt
    """
    # ------------------- 3-Shot Prompts (Unified) -------------------
    # 根据提供的代码，3-Shot 的 Prompt 在 Qwen 和 Llama 上使用了相同的模板
    if shot_mode == "3-shot":
        if dataset_name == "SMP2020":
            return f"""任务：判断文本的情感类别。
参考示例：
文本: "今天天气真不错，心情好极了！" -> 类别: 2
文本: "这服务态度太差了，气死我了。" -> 类别: 0
文本: "这就是一本普通的书。" -> 类别: 3

请对以下文本分类：
文本: "{text}"
类别选项: 0:愤怒, 1:恐惧, 2:高兴, 3:中性, 4:悲伤, 5:惊奇
请仅输出一个数字 ID (0-5)。
答案:"""
        elif dataset_name == "SST-5":
            return f"""Task: Classify the sentiment.
Examples:
Text: "An absolute masterpiece, thrilling from start to finish." -> Class: 4
Text: "Boring, predictable, and a waste of time." -> Class: 0
Text: "It's a movie that exists." -> Class: 2

Classify this:
Text: "{text}"
Options: 0:Very Negative, 1:Negative, 2:Neutral, 3:Positive, 4:Very Positive
Return ONLY the numeric ID (0-4).
Answer:"""
        elif dataset_name == "TweetEval":
            return f"""Task: Classify tweet sentiment.
Examples:
Text: "Can't wait for the concert tonight! #excited" -> Class: 2
Text: "My flight got cancelled again. Ugh." -> Class: 0
Text: "Just had lunch." -> Class: 1

Classify this:
Text: "{text}"
Options: 0:Negative, 1:Neutral, 2:Positive
Return ONLY the numeric ID (0-2).
Answer:"""

    # ------------------- Zero-Shot Prompts (Model Specific) -------------------
    elif shot_mode == "zero-shot":
        # Qwen Zero-Shot (SMP2020 使用中文指令)
        if "Qwen" in model_short_name:
            if dataset_name == "SMP2020":
                return f"""分析这句话的情感。
文本: "{text}"
选项:
0: 愤怒
1: 恐惧
2: 高兴
3: 中性
4: 悲伤
5: 惊奇
请只回答一个数字 ID (0-5)。不要解释。
答案:"""
            elif dataset_name == "SST-5":
                return f"""Classify the sentiment.
Text: "{text}"
Options:
0: Very Negative
1: Negative
2: Neutral
3: Positive
4: Very Positive
Return ONLY the numeric ID (0-4). Do not explain.
Answer:"""
            elif dataset_name == "TweetEval":
                return f"""Classify the sentiment.
Text: "{text}"
Options:
0: Negative
1: Neutral
2: Positive
Return ONLY the numeric ID (0-2). Do not explain.
Answer:"""
        
        # Llama Zero-Shot (SMP2020 使用英文指令以提高稳定性)
        elif "Llama" in model_short_name:
            if dataset_name == "SMP2020":
                return f"""Analyze the sentiment of the following Chinese text.
Text: "{text}"
Options:
0: 愤怒 (Angry)
1: 恐惧 (Fear)
2: 高兴 (Happy)
3: 中性 (Neutral)
4: 悲伤 (Sad)
5: 惊奇 (Surprise)
Return ONLY the numeric ID (0-5). Do not explain.
Answer:"""
            elif dataset_name == "SST-5":
                return f"""Classify the sentiment of the text.
Text: "{text}"
Options:
0: Very Negative
1: Negative
2: Neutral
3: Positive
4: Very Positive
Return ONLY the numeric ID (0-4). Do not explain.
Answer:"""
            elif dataset_name == "TweetEval":
                return f"""Classify the sentiment of the tweet.
Text: "{text}"
Options:
0: Negative
1: Neutral
2: Positive
Return ONLY the numeric ID (0-2). Do not explain.
Answer:"""
    
    return ""

def parse_prediction(response, dataset_name, model_short_name):
    """
    统一解析逻辑
    """
    try:
        # 正则提取第一个数字
        match = re.search(r'\d', response)
        if match:
            return int(match.group())
        else:
            # 兜底策略 (根据数据集分布猜测大类)
            if dataset_name == "SMP2020": return 3 # 中性/Majority
            if dataset_name == "SST-5": return 2   # Neutral
            return 1 # TweetEval Neutral
    except:
        return 1

# =========================== 🚀 核心推理循环 ===========================

def run_inference_for_model(model_path, model_short_name):
    """
    加载一个模型，并运行它所有的 Zero-shot 和 3-shot 任务
    """
    print(f"\n\n{'='*20} 🤖 Loading Model: {model_short_name} {'='*20}")
    
    # 1. 4-bit 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    # 2. 加载 Tokenizer & Model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            
    except Exception as e:
        print(f"❌ Failed to load {model_short_name}: {e}")
        return []

    model_results = []
    
    # 3. 遍历两种模式: Zero-shot 和 3-shot
    modes = ["zero-shot", "3-shot"]
    
    for mode in modes:
        print(f"\n➡️  Mode: {mode.upper()}")
        
        # System Prompt 设置
        if mode == "3-shot":
            sys_msg = "You are a helpful sentiment analysis assistant. Follow the examples provided."
        else:
            sys_msg = "You are a helpful sentiment analysis assistant. You output only numeric class IDs."

        for ds_name in DATASETS:
            val_df = get_validation_set(ds_name)
            preds = []
            labels = val_df['label'].tolist()
            
            # 使用 tqdm 显示进度
            iterator = tqdm(val_df['text'], desc=f"   Running {ds_name}", leave=False)
            
            for text in iterator:
                # 获取 Prompt
                content = get_prompt_content(ds_name, text, model_short_name, mode)
                
                messages = [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": content}
                ]
                
                # Apply Chat Template
                text_input = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                
                inputs = tokenizer([text_input], return_tensors="pt").to(model.device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs.input_ids,
                        max_new_tokens=5,     # 限制输出长度
                        do_sample=False,      # 贪婪解码
                        temperature=0.0,
                        pad_token_id=tokenizer.pad_token_id
                    )
                
                # Decode Response
                response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
                
                # Parse
                pred = parse_prediction(response, ds_name, model_short_name)
                preds.append(pred)
            
            # Calculate Metrics
            macro_f1 = f1_score(labels, preds, average="macro")
            acc = accuracy_score(labels, preds)
            
            print(f"   ✅ {ds_name}: Macro-F1 = {macro_f1:.4f}, Acc = {acc:.4f}")
            
            model_results.append({
                "Model": model_short_name,
                "Mode": mode,
                "Dataset": ds_name,
                "Macro-F1": macro_f1,
                "Accuracy": acc
            })

    # 4. 清理显存 (Crucial for running multiple models in one script)
    print(f"🗑️  Unloading {model_short_name}...")
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    return model_results

# =========================== 🏁 主程序入口 ===========================

if __name__ == "__main__":
    all_results = []
    
    # 依次运行每个模型
    for model_path, model_short_name in MODELS_TO_TEST:
        results = run_inference_for_model(model_path, model_short_name)
        all_results.extend(results)
    
    # 输出并保存最终表格
    final_df = pd.DataFrame(all_results)
    
    print("\n\n" + "="*50)
    print("🏆 FINAL BENCHMARK RESULTS SUMMARY")
    print("="*50)
    print(final_df.to_string(index=False))
    
    # 保存为 CSV
    final_df.to_csv("exp_llm_comparison_results.csv", index=False)
    print("\n📄 Results saved to 'exp_llm_comparison_results.csv'")