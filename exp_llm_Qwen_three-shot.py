import torch
import random
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

CURRENT_MODEL = "Qwen/Qwen2.5-7B-Instruct" 


SEED = 42

print(f"🤖 初始化模型 (3-Shot 模式): {CURRENT_MODEL} ...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(CURRENT_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    CURRENT_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id

def get_validation_set(dataset_name):
    np.random.seed(SEED)
    random.seed(SEED)
    if dataset_name == "SMP2020":
        ds = load_dataset("Um1neko/smp2020", split="train")
        df = pd.DataFrame(ds)
        if "content" in df.columns: df = df.rename(columns={"content": "text"})
        df = df.dropna(subset=["text", "label"])
        df["label"] = df["label"].astype(int)
        val_count = 80 
    elif dataset_name == "SST-5":
        ds = load_dataset("SetFit/sst5", split="train")
        df = pd.DataFrame(ds)
        if "sentence" in df.columns: df = df.rename(columns={"sentence": "text"})
        if "label_text" in df.columns: df = df.drop(columns=["label_text"])
        df = df[["text", "label"]].dropna()
        df["label"] = df["label"].astype(int)
        val_count = 100
    elif dataset_name == "TweetEval":
        ds = load_dataset("tweet_eval", "sentiment", split="train")
        df = pd.DataFrame(ds)
        df = df[["text", "label"]].dropna()
        df["label"] = df["label"].astype(int)
        val_count = 100
    _, val_pool = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)
    num_labels = df['label'].nunique()
    sampled_dfs = []
    for label in range(num_labels):
        class_df = val_pool[val_pool['label'] == label]
        n_samples = min(len(class_df), val_count)
        sampled_dfs.append(class_df.sample(n=n_samples, random_state=SEED))
    return pd.concat(sampled_dfs).sample(frac=1, random_state=SEED).reset_index(drop=True)

def get_3shot_prompt(dataset_name, text):
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

def benchmark_dataset(dataset_name):
    print(f"\n🚀 Running {dataset_name} (3-Shot)...")
    val_df = get_validation_set(dataset_name)
    preds = []
    labels = val_df['label'].tolist()
    
    for text in tqdm(val_df['text']):
        content = get_3shot_prompt(dataset_name, text)
        messages = [
            {"role": "system", "content": "You are a helpful sentiment analysis assistant. Follow the examples provided."},
            {"role": "user", "content": content}
        ]
        
        text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text_input], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=5, 
                do_sample=False, 
                temperature=0.0,
                pad_token_id=tokenizer.pad_token_id
            )
            
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        match = re.search(r'\d', response)
        if match:
            pred = int(match.group())
        else:
            pred = 2 if dataset_name == "SST-5" else (3 if dataset_name == "SMP2020" else 1)
        preds.append(pred)
        
    f1 = f1_score(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    return f1, acc

results = []
datasets = ["SMP2020", "SST-5", "TweetEval"]

for ds in datasets:
    f1, acc = benchmark_dataset(ds)
    print(f"✅ {ds}: F1={f1:.4f}, Acc={acc:.4f}")
    results.append({"Dataset": ds, "Model": CURRENT_MODEL + " (3-Shot)", "Macro-F1": f1, "Accuracy": acc})

final_df = pd.DataFrame(results)
print("\n=== 3-Shot Benchmark Results ===")
print(final_df.to_string(index=False))
