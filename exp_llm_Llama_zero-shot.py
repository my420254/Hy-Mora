
import torch, random, gc, re
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


MODEL_ID = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
SEED = 42

print(f"🤖 Loading LLM: {MODEL_ID}...")


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

def get_validation_set(dataset_name):
    np.random.seed(SEED)
    random.seed(SEED)
    
    if dataset_name == "SMP2020":
        print("📚 Loading SMP2020...")
        ds = load_dataset("Um1neko/smp2020")
        df = pd.DataFrame(ds["train"])
        if "content" in df.columns: df = df.rename(columns={"content": "text"})
        df = df.dropna(subset=["text", "label"])
        df["label"] = df["label"].astype(int)
        val_count = 80  # SMP
        
    elif dataset_name == "SST-5":
        print("📚 Loading SST-5...")
        ds = load_dataset("SetFit/sst5")
        df = pd.DataFrame(ds["train"])
        if "sentence" in df.columns: df = df.rename(columns={"sentence": "text"})
        if "label_text" in df.columns: df = df.drop(columns=["label_text"])
        df = df[["text", "label"]].dropna()
        df["label"] = df["label"].astype(int)
        val_count = 100 # SST
        
    elif dataset_name == "TweetEval":
        print("📚 Loading TweetEval...")
        ds = load_dataset("tweet_eval", "sentiment")
        df = pd.DataFrame(ds["train"])
        df = df[["text", "label"]].dropna()
        df["label"] = df["label"].astype(int)
        val_count = 100 # TweetEval
        
    _, val_pool = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)
    
    num_labels = df['label'].nunique()
    val_balanced_config = {k: val_count for k in range(num_labels)}
    
    sampled_dfs = []
    for label, count in val_balanced_config.items():
        class_df = val_pool[val_pool['label'] == label]
        n_samples = min(len(class_df), count)
        sampled_dfs.append(class_df.sample(n=n_samples, random_state=SEED))
    
    val_df = pd.concat(sampled_dfs).sample(frac=1, random_state=SEED).reset_index(drop=True)
    print(f"✅ Validation Set ({dataset_name}): {len(val_df)} samples (Balanced).")
    return val_df

def get_prompt(dataset_name, text):
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

def parse_prediction(response, dataset_name):
    try:
        match = re.search(r'\d', response)
        if match:
            return int(match.group())
        else:
            if dataset_name == "SMP2020": return 3
            if dataset_name == "SST-5": return 2
            return 1
    except:
        return 1

def run_benchmark(model, tokenizer, datasets):
    results = []
    
    for ds_name in datasets:
        val_df = get_validation_set(ds_name)
        preds = []
        labels = val_df['label'].tolist()
        
        print(f"🚀 Running Llama-3 Zero-Shot Inference on {ds_name}...")
        
        for text in tqdm(val_df['text']):
            prompt = get_prompt(ds_name, text)
            messages = [{"role": "user", "content": prompt}]
            
            text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([text_input], return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_new_tokens=5,     
                    do_sample=False,      
                    temperature=0.0,
                    pad_token_id=tokenizer.eos_token_id
                )
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            pred = parse_prediction(response, ds_name)
            preds.append(pred)
            
        macro_f1 = f1_score(labels, preds, average="macro")
        acc = accuracy_score(labels, preds)
        
        print(f"🏆 {ds_name} (Llama-3) Result: Macro-F1 = {macro_f1:.4f}, Acc = {acc:.4f}")
        results.append({"Dataset": ds_name, "Method": "Llama-3-8B-ZeroShot", "Macro-F1": macro_f1, "Accuracy": acc})

    return pd.DataFrame(results)


datasets_to_test = ["SMP2020", "SST-5", "TweetEval"] 


df_llama_results = run_benchmark(model, tokenizer, datasets_to_test)

print("\n📊 FINAL LLAMA-3 BENCHMARK RESULTS:")
print(df_llama_results.to_string(index=False))

df_llama_results.to_csv("llama3_benchmark_results.csv", index=False)
