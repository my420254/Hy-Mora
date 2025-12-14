import os, shutil, gc, torch, warnings, random, time, json
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer, AutoModel, Trainer, TrainingArguments,
    DataCollatorWithPadding, EarlyStoppingCallback,
    AutoModelForSequenceClassification
)
from peft import LoraConfig, get_peft_model, TaskType
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# --- Environment ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== 1. Global Config (SST-5) ====================
MODEL_NAME = "roberta-base"
NUM_LABELS = 5
MAX_LENGTH = 128
EPOCHS = 20
BATCH_SIZE = 64  # <--- [已修复] 补上了 BATCH_SIZE
RANDOM_SEEDS = [45, 123, 789, 2024, 1001]
FULL_LR = 2e-5        
PEFT_LR = 3e-4        

# Dataset Specific Configs
CONFIGS = {
    2300: {
        "train": {4: 1200, 3: 600, 2: 300, 1: 150, 0: 50}, 
        "eval_steps": 10, "memory_size": 1200, "temperature": 0.5, "loss_weight": 0.2, 
        "warmup_steps": 30, "tail_weight": 1.0, "lr_scale": 0.9, "grad_acc": 1,
        "fusion_init": -1.8, "smoothing": 0.1, "clamp_weights": True    
    },
    1150: {
        "train": {4: 600, 3: 300, 2: 150, 1: 80, 0: 20},
        "eval_steps": 10, "memory_size": 1200, "temperature": 0.15, "loss_weight": 0.1,   
        "warmup_steps": 20, "tail_weight": 2.0, "lr_scale": 1.0, "grad_acc": 2,              
        "fusion_init": 0.3, "smoothing": 0.05, "clamp_weights": True
    }
}
TAIL_CLASSES = [0, 1] 

# === [FINAL EXPERIMENT LIST: 10 Experiments] ===
EXPERIMENTS = [
    # --- Group 1: LoRA Series ---
    {"name": "LoRA-Vanilla",        "method": "peft",    "loss_type": "original",  "use_class_weight": False, "peft_type": "lora", "hsp": False, "memory_bank": False}, 
    {"name": "LoRA-Balanced",       "method": "peft",    "loss_type": "original",  "use_class_weight": True,  "peft_type": "lora", "hsp": False, "memory_bank": False}, 
    {"name": "LoRA-Ablation-NoMem", "method": "peft",    "loss_type": "original",  "use_class_weight": True,  "peft_type": "lora", "hsp": True,  "memory_bank": False}, 
    {"name": "LoRA-Ablation-NoHSP", "method": "peft",    "loss_type": "original",  "use_class_weight": True,  "peft_type": "lora", "hsp": False, "memory_bank": True},  
    {"name": "LoRA-Ours",           "method": "peft",    "loss_type": "original",  "use_class_weight": True,  "peft_type": "lora", "hsp": True,  "memory_bank": True},  
    
    # --- Group 2: Strong Baseline ---
    {"name": "DoRA-Balanced",       "method": "peft",    "loss_type": "original",  "use_class_weight": True,  "peft_type": "dora", "hsp": False, "memory_bank": False}, 
    
    # --- Group 3: Baselines ---
    {"name": "LoRA-Focal",          "method": "peft",    "loss_type": "focal",     "use_class_weight": True,  "peft_type": "lora", "hsp": False, "memory_bank": False}, 
    {"name": "LoRA-LDAM",           "method": "peft",    "loss_type": "ldam",      "use_class_weight": True,  "peft_type": "lora", "hsp": False, "memory_bank": False}, 
    {"name": "LoRA-LogitAdj",       "method": "peft",    "loss_type": "logit_adj", "use_class_weight": True,  "peft_type": "lora", "hsp": False, "memory_bank": False}, 
    {"name": "Full-FineTuning",     "method": "full_ft", "loss_type": "original",  "use_class_weight": True,  "peft_type": None,   "hsp": False, "memory_bank": False}, 
]

SENS_TEMPS = [0.05, 0.1, 0.3, 0.5]  
SENS_LOSS_WEIGHTS = [0.01, 0.05, 0.1, 0.2]

# File Paths
MAIN_RESULTS_FILE = "sst5_results_Final_Full.csv"
SENSITIVITY_FILE = "sst5_sensitivity_Final_Full.csv"
IMG_DATA_DIR = "./viz_data_sst5"
os.makedirs(IMG_DATA_DIR, exist_ok=True)

# ==================== Helper & Classes ====================
def save_experiment_full_data(trainer, model, tokenizer, output_dir, file_prefix):
    with open(f"{output_dir}/{file_prefix}_history.json", "w") as f: json.dump(trainer.state.log_history, f)
    dataloader = trainer.get_eval_dataloader()
    feats, labs, preds, logits_list, inputs_txt = [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask", "labels"]}
            if hasattr(model, "encoder") and hasattr(model.encoder, "forward"): out_enc = model.encoder(inputs["input_ids"], inputs["attention_mask"]); feat = out_enc.last_hidden_state[:, 0, :]
            elif hasattr(model, "model") and hasattr(model.model, "base_model"): feat = model.model.base_model(inputs["input_ids"], inputs["attention_mask"]).last_hidden_state[:, 0, :]
            else: feat = torch.zeros(inputs["input_ids"].size(0), 768)
            out = model(inputs["input_ids"], inputs["attention_mask"]); logits = out["logits"] if isinstance(out, dict) else out.logits; p = torch.argmax(logits, dim=-1)
            feats.append(feat.cpu().numpy()); labs.append(inputs["labels"].cpu().numpy()); preds.append(p.cpu().numpy()); logits_list.append(logits.cpu().numpy())
            decoded = tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True); inputs_txt.extend(decoded)
    np.savez(f"{output_dir}/{file_prefix}_viz.npz", feats=np.vstack(feats), labels=np.concatenate(labs), preds=np.concatenate(preds), logits=np.vstack(logits_list))
    df_cases = pd.DataFrame({"text": inputs_txt, "true_label": np.concatenate(labs), "pred_label": np.concatenate(preds)})
    df_cases.to_csv(f"{output_dir}/{file_prefix}_cases.csv", index=False)

def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children(): result += [f"{name}.{n}" for n in get_parameter_names(child, forbidden_layer_types) if not isinstance(child, tuple(forbidden_layer_types))]
    result += list(model._parameters.keys())
    return result

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__(); self.alpha = alpha; self.gamma = gamma; self.reduction = reduction
    def forward(self, logits, labels):
        ce_loss = F.cross_entropy(logits, labels, reduction='none', weight=self.alpha); pt = torch.exp(-ce_loss); focal_loss = ((1 - pt) ** self.gamma * ce_loss)
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, s=30):
        super().__init__(); m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list)); m_list = m_list * (max_m / np.max(m_list)); self.m_list = torch.tensor(m_list, dtype=torch.float32); self.s = s
    def forward(self, logits, labels):
        if self.m_list.device != logits.device: self.m_list = self.m_list.to(logits.device)
        batch_m = self.m_list[labels]; logits_m = logits - batch_m.unsqueeze(1) * torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), 1)
        return F.cross_entropy(self.s * logits_m, labels)

class LogitAdjustmentLoss(nn.Module):
    def __init__(self, cls_num_list, tau=1.0):
        super().__init__(); cls_probs = np.array(cls_num_list) / np.sum(cls_num_list); self.logit_adj = torch.log(torch.tensor(cls_probs, dtype=torch.float32) ** tau + 1e-12)
    def forward(self, logits, labels):
        if self.logit_adj.device != logits.device: self.logit_adj = self.logit_adj.to(logits.device)
        adjusted_logits = logits + self.logit_adj.unsqueeze(0); return F.cross_entropy(adjusted_logits, labels)

class MemoryBank(nn.Module):
    def __init__(self, feature_dim=128, num_classes=5, memory_size=600, temperature=0.3, tail_classes=[0, 1], tail_weight=1.3, warmup_steps=10, min_samples=5):
        super().__init__(); self.feature_dim = feature_dim; self.num_classes = num_classes; self.temperature = temperature
        self.tail_classes = tail_classes; self.tail_weight = tail_weight; self.warmup_steps = warmup_steps; self.min_samples = min_samples; self.current_step = 0
        capacity = memory_size // num_classes
        for c in range(num_classes): self.register_buffer(f'memory_bank_{c}', torch.randn(capacity, feature_dim))
        self.register_buffer('bank_ptrs', torch.zeros(num_classes, dtype=torch.long)); self.register_buffer('bank_sizes', torch.zeros(num_classes, dtype=torch.long))
    def get_memory_bank(self, class_id): return getattr(self, f'memory_bank_{class_id}')
    def set_memory_bank(self, class_id, data, start_idx, end_idx): getattr(self, f'memory_bank_{class_id}')[start_idx:end_idx] = data
    @torch.no_grad()
    def update_memory_bank(self, features, labels):
        if self.current_step < self.warmup_steps: return
        features = F.normalize(features.detach().clone(), dim=1); labels = labels.detach().clone()
        for c in range(self.num_classes):
            mask = (labels == c); 
            if not mask.any(): continue
            feats_c = features[mask].clone(); n = feats_c.size(0); bank = self.get_memory_bank(c); ptr = self.bank_ptrs[c].item(); cap = bank.size(0)
            if ptr + n <= cap: self.set_memory_bank(c, feats_c, ptr, ptr + n); self.bank_ptrs[c] = (ptr + n) % cap
            else: rem = cap - ptr; self.set_memory_bank(c, feats_c[:rem], ptr, cap); self.set_memory_bank(c, feats_c[rem:], 0, n - rem); self.bank_ptrs[c] = n - rem
            self.bank_sizes[c] = min(self.bank_sizes[c] + n, cap)
    def forward(self, features, labels):
        self.current_step += 1; 
        if self.current_step <= self.warmup_steps: return torch.tensor(0.0, device=features.device, requires_grad=True)
        features_norm = F.normalize(features, dim=1); total_loss = 0.0; valid = 0
        for i in range(features.size(0)):
            feat = features_norm[i]; label = labels[i].item(); pos = self.get_memory_bank(label)[:self.bank_sizes[label]].detach().clone()
            if pos.size(0) < self.min_samples: continue
            negs = [self.get_memory_bank(c)[:self.bank_sizes[c]].detach().clone() for c in range(self.num_classes) if c != label and self.bank_sizes[c] >= self.min_samples]
            if not negs: continue
            neg_feats = torch.cat(negs, dim=0); logits = torch.cat([torch.matmul(feat.unsqueeze(0), pos.t()) / self.temperature, torch.matmul(feat.unsqueeze(0), neg_feats.t()) / self.temperature], dim=1)
            total_loss += (self.tail_weight if label in self.tail_classes else 1.0) * F.cross_entropy(logits, torch.zeros(1, dtype=torch.long, device=features.device)); valid += 1
        return total_loss / valid if valid > 0 else torch.tensor(0.0, device=features.device, requires_grad=True)

class HierarchicalSmartPooling(nn.Module):
    def __init__(self, hs, dr=0.1):
        super().__init__(); self.attn = nn.Sequential(nn.Linear(hs, hs), nn.Tanh(), nn.Linear(hs, 1), nn.Softmax(dim=1)); self.fusion = nn.Sequential(nn.Linear(hs*3, hs*2), nn.LayerNorm(hs*2), nn.GELU(), nn.Dropout(dr), nn.Linear(hs*2, hs))
    def forward(self, x, m):
        w = self.attn(x).masked_fill(m.unsqueeze(-1)==0, -1e9); w = F.softmax(w, dim=1)
        return self.fusion(torch.cat([torch.sum(x*w, 1), torch.sum(x*m.unsqueeze(-1), 1)/m.sum(1, keepdim=True).clamp(min=1e-9), x.masked_fill(m.unsqueeze(-1)==0, -1e9).max(1)[0]], dim=1))

class UnifiedModel(nn.Module):
    def __init__(self, cfg):
        super().__init__(); self.cfg = cfg; self.is_peft = (cfg["method"] == "peft")
        if not self.is_peft: self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS); self.config = self.model.config
        else:
            peft_type = cfg.get("peft_type", "lora"); target_modules = ["query", "key", "value"]
            use_dora = True if peft_type == "dora" else False
            peft_config = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=16, lora_alpha=32, lora_dropout=0.1, target_modules=target_modules, use_dora=use_dora)
            self.encoder = get_peft_model(AutoModel.from_pretrained(MODEL_NAME), peft_config); self.config = self.encoder.config; self.config.num_labels = NUM_LABELS; hs = self.encoder.config.hidden_size
            self.classifier_base = nn.Linear(hs, NUM_LABELS)
            if cfg["hsp"]: self.hsp_module = HierarchicalSmartPooling(hs); self.classifier_hsp = nn.Linear(hs, NUM_LABELS); nn.init.constant_(self.classifier_hsp.weight, 0); nn.init.constant_(self.classifier_hsp.bias, 0); self.fusion_weight = nn.Parameter(torch.tensor([cfg.get("fusion_init", 0.1)]))
            else: self.hsp_module = None
            if cfg["memory_bank"]: self.projector = nn.Sequential(nn.Linear(hs, hs), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hs, 128))
            else: self.projector = None
    def forward(self, input_ids, attention_mask, labels=None):
        if not self.is_peft: return {"loss": None, "logits": self.model(input_ids, attention_mask, labels=labels).logits, "proj_features": None}
        hidden = self.encoder(input_ids, attention_mask).last_hidden_state; cls_feat = hidden[:, 0, :]; logits = self.classifier_base(cls_feat)
        if self.hsp_module: logits = logits + torch.sigmoid(self.fusion_weight) * self.classifier_hsp(self.hsp_module(hidden, attention_mask))
        return {"loss": None, "logits": logits, "proj_features": self.projector(cls_feat) if self.projector else None}

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def get_custom_dataset(df, config, seed):
    sampled = [df[df['label'] == l].sample(n=min(len(df[df['label'] == l]), c), random_state=seed) for l, c in config.items()]
    return pd.concat(sampled).sample(frac=1, random_state=seed).reset_index(drop=True)

def compute_metrics(eval_pred):
    logits = eval_pred.predictions; preds = np.argmax(logits, axis=-1); labels = eval_pred.label_ids
    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    recalls = [report[str(i)]['recall'] for i in range(NUM_LABELS)]
    try: probs = F.softmax(torch.tensor(logits), dim=-1).numpy(); auc = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
    except: auc = 0.0
    metrics = {"macro_f1": f1_score(labels, preds, average="macro"), "weighted_f1": f1_score(labels, preds, average="weighted"), "accuracy": accuracy_score(labels, preds), "balanced_acc": np.mean(recalls), "g_mean": np.prod(recalls) ** (1/NUM_LABELS), "auc": auc}
    for i in range(NUM_LABELS): metrics[f"f1_class_{i}"] = report[str(i)]['f1-score']
    return metrics

def append_to_csv(filename, row_dict):
    df = pd.DataFrame([row_dict]); df.to_csv(filename, mode='a', header=not os.path.exists(filename), index=False)

class UnifiedTrainer(Trainer):
    def __init__(self, loss_type, class_weights, cls_num_list, memory_loss, loss_weight, is_peft, smoothing, use_class_weight=True, **kwargs):
        super().__init__(**kwargs); self.loss_type = loss_type; self.use_class_weight = use_class_weight; self.class_weights = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
        self.cls_num_list = cls_num_list; self.memory_loss_module = memory_loss; self.aux_loss_weight = loss_weight; self.is_peft = is_peft; self.label_smoothing = smoothing; self.current_epoch = 0
        if loss_type == "ldam": self.ldam_loss = LDAMLoss(cls_num_list, max_m=0.5, s=30)
        elif loss_type == "logit_adj": self.logit_adj_loss = LogitAdjustmentLoss(cls_num_list, tau=1.0)
        elif loss_type == "focal": alpha = self.class_weights if self.use_class_weight else None; self.focal_loss = FocalLoss(alpha=alpha, gamma=2.0)
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels"); outputs = model(inputs["input_ids"], inputs["attention_mask"], labels); logits = outputs["logits"]
        weight_to_use = None
        if self.use_class_weight and self.class_weights is not None:
             if self.class_weights.device != logits.device: self.class_weights = self.class_weights.to(logits.device)
             weight_to_use = self.class_weights
        if self.loss_type == "original":
            loss_fct = nn.CrossEntropyLoss(weight=weight_to_use, label_smoothing=self.label_smoothing); total_loss = loss_fct(logits.view(-1, NUM_LABELS), labels.view(-1))
            if self.is_peft and self.memory_loss_module is not None and outputs.get("proj_features") is not None:
                proj_features = outputs["proj_features"]; loss_mb = self.memory_loss_module(proj_features, labels); total_loss += self.aux_loss_weight * loss_mb
                with torch.no_grad(): self.memory_loss_module.update_memory_bank(proj_features, labels)
        elif self.loss_type == "focal":
            if hasattr(self.focal_loss, 'alpha') and self.focal_loss.alpha is not None:
                 if self.focal_loss.alpha.device != logits.device: self.focal_loss.alpha = self.focal_loss.alpha.to(logits.device)
            total_loss = self.focal_loss(logits, labels)
        elif self.loss_type == "ldam":
            if self.current_epoch < int(EPOCHS * 0.5): total_loss = self.ldam_loss(logits, labels)
            else: loss_fct = nn.CrossEntropyLoss(weight=weight_to_use); total_loss = loss_fct(logits.view(-1, NUM_LABELS), labels.view(-1))
        elif self.loss_type == "logit_adj": total_loss = self.logit_adj_loss(logits, labels)
        return (total_loss, SequenceClassifierOutput(loss=total_loss, logits=logits)) if return_outputs else total_loss
    def create_optimizer(self):
        if self.optimizer is None:
            decay_parameters = get_parameter_names(self.model, [nn.LayerNorm]); decay_parameters = [name for name in decay_parameters if "bias" not in name]; optimizer_grouped_parameters = []
            for n, p in self.model.named_parameters():
                if not p.requires_grad: continue
                if "fusion_weight" in n: optimizer_grouped_parameters.append({"params": [p], "weight_decay": 0.0, "lr": self.args.learning_rate * 5})
                else: optimizer_grouped_parameters.append({"params": [p], "weight_decay": self.args.weight_decay if n in decay_parameters else 0.0, "lr": self.args.learning_rate})
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args); self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer
    def on_epoch_begin(self, args, state, control, **kwargs): self.current_epoch = state.epoch

# ==================== 4. 数据 & 实验 A ====================
print(">>> Loading SST-5 Dataset...")
try: dataset_raw = load_dataset("SetFit/sst5")
except: dataset_raw = load_dataset("SetFit/sst5") 
full_df = pd.DataFrame(dataset_raw["train"]).dropna(subset=["text", "label"])
full_df["label"] = full_df["label"].astype(int)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
def tokenize(ex): return tokenizer(ex["text"], truncation=True, max_length=MAX_LENGTH)
if os.path.exists(MAIN_RESULTS_FILE): os.remove(MAIN_RESULTS_FILE)
if os.path.exists(SENSITIVITY_FILE): os.remove(SENSITIVITY_FILE)

print(f"\n{'='*80}\nPART A: MAIN + SOTA EXPERIMENTS\n{'='*80}")
for N_SAMPLES in [2300, 1150]: 
    cfg = CONFIGS[N_SAMPLES]
    train_pool, val_pool = train_test_split(full_df, test_size=0.2, stratify=full_df["label"], random_state=42)
    val_df = get_custom_dataset(val_pool, {k: 80 for k in range(NUM_LABELS)}, 42)
    val_ds = Dataset.from_pandas(val_df).map(tokenize, batched=True).select_columns(["input_ids", "attention_mask", "label"])

    for exp in EXPERIMENTS:
        safe_name = exp['name'].replace('/', '_').replace('+', '_plus')
        for seed_idx, SEED in enumerate(RANDOM_SEEDS):
            print(f"\n[Part A] N={N_SAMPLES} | {exp['name']} | Seed={SEED}")
            set_seed(SEED)
            train_df = get_custom_dataset(train_pool, cfg["train"], SEED)
            cw = compute_class_weight('balanced', classes=np.unique(train_df['label']), y=train_df['label'])
            class_weights_np = cw
            if cfg['clamp_weights']: class_weights_np = torch.tensor(cw, dtype=torch.float).clamp(max=10.0).numpy()
            cls_num_list = [len(train_df[train_df['label'] == i]) for i in range(NUM_LABELS)]
            train_ds = Dataset.from_pandas(train_df).map(tokenize, batched=True).select_columns(["input_ids", "attention_mask", "label"])
            
            current_cfg = exp.copy(); current_cfg["fusion_init"] = cfg["fusion_init"]
            model = UnifiedModel(current_cfg).to(device)
            lr = FULL_LR if exp["method"] == "full_ft" else PEFT_LR * cfg["lr_scale"]
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            output_dir_path = f"./tmp_sst5_{N_SAMPLES}_{safe_name}_{SEED}"

            trainer = UnifiedTrainer(
                loss_type=exp["loss_type"],
                class_weights=class_weights_np, 
                cls_num_list=cls_num_list,
                memory_loss=MemoryBank(128, NUM_LABELS, cfg["memory_size"], cfg["temperature"], TAIL_CLASSES, cfg["tail_weight"], cfg["warmup_steps"], 5).to(device) if exp["memory_bank"] else None,
                loss_weight=cfg["loss_weight"], 
                is_peft=(exp["method"] == "peft"), 
                model=model,
                use_class_weight=exp.get("use_class_weight", True),
                args=TrainingArguments(output_dir=output_dir_path, num_train_epochs=EPOCHS, per_device_train_batch_size=BATCH_SIZE, gradient_accumulation_steps=cfg["grad_acc"], learning_rate=lr, warmup_ratio=0.1, weight_decay=0.01, eval_strategy="steps", eval_steps=cfg["eval_steps"], save_steps=cfg["eval_steps"], save_total_limit=1, load_best_model_at_end=True, metric_for_best_model="macro_f1", fp16=True, report_to="none", logging_steps=5),
                train_dataset=train_ds, eval_dataset=val_ds, tokenizer=tokenizer, data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=compute_metrics, callbacks=[EarlyStoppingCallback(early_stopping_patience=8)], 
                smoothing=cfg["smoothing"]
            )
            
            torch.cuda.reset_peak_memory_stats(); start_time = time.time(); trainer.train()
            train_runtime = time.time() - start_time; peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024; res = trainer.evaluate()
            start_inf = time.time(); _ = trainer.predict(val_ds); inf_time = time.time() - start_inf
            
            row = { "Dataset": "SST-5", "N": N_SAMPLES, "Method": exp['name'], "Seed": SEED, "Macro-F1": res['eval_macro_f1'], "Weighted-F1": res['eval_weighted_f1'], "Accuracy": res['eval_accuracy'], "Balanced_Acc": res['eval_balanced_acc'], "G-Mean": res['eval_g_mean'], "AUC": res['eval_auc'], "Train_Time_Sec": train_runtime, "Inference_Time_Sec": inf_time, "Peak_Memory_MB": peak_memory, "Params_M": num_params / 1e6 }
            for i in range(NUM_LABELS): row[f"F1_Class_{i}"] = res[f"eval_f1_class_{i}"]
            append_to_csv(MAIN_RESULTS_FILE, row)
            file_prefix = f"{safe_name}_N{N_SAMPLES}_seed{SEED}"
            save_experiment_full_data(trainer, model, tokenizer, IMG_DATA_DIR, file_prefix)
            del model, trainer; torch.cuda.empty_cache(); gc.collect(); shutil.rmtree(output_dir_path, ignore_errors=True)

# ==================== 6. 实验 B: 敏感性分析 (LoRA Only) ====================
print(f"\n{'='*80}\nPART B: SENSITIVITY EXPERIMENTS (LoRA-Ours Only)\n{'='*80}")
cfg_sens = CONFIGS[2300]
train_pool, val_pool = train_test_split(full_df, test_size=0.2, stratify=full_df["label"], random_state=42)
val_df = get_custom_dataset(val_pool, {k: 80 for k in range(NUM_LABELS)}, 42)
val_ds = Dataset.from_pandas(val_df).map(tokenize, batched=True).select_columns(["input_ids", "attention_mask", "label"])

# --- Temperature ---
for temp in SENS_TEMPS:
    for SEED in RANDOM_SEEDS:
        print(f"\n[Sensitivity-LoRA] Type=Temperature | Value={temp} | Seed={SEED}")
        set_seed(SEED)
        train_df = get_custom_dataset(train_pool, cfg_sens["train"], SEED)
        cw = compute_class_weight('balanced', classes=np.unique(train_df['label']), y=train_df['label'])
        class_weights_np = torch.tensor(cw, dtype=torch.float).clamp(max=10.0).numpy()
        train_ds = Dataset.from_pandas(train_df).map(tokenize, batched=True).select_columns(["input_ids", "attention_mask", "label"])
        
        model = UnifiedModel({"name": "LoRA-Ours", "method": "peft", "loss_type": "original", "peft_type": "lora", "hsp": True, "memory_bank": True, "fusion_init": cfg_sens["fusion_init"]}).to(device)
        trainer = UnifiedTrainer(
            loss_type="original",
            class_weights=class_weights_np, cls_num_list=[],
            memory_loss=MemoryBank(128, NUM_LABELS, cfg_sens["memory_size"], temp, TAIL_CLASSES, cfg_sens["tail_weight"], cfg_sens["warmup_steps"], 5).to(device),
            loss_weight=cfg_sens["loss_weight"], is_peft=True, model=model, use_class_weight=True,
            args=TrainingArguments(output_dir=f"./tmp_sens_T{temp}_S{SEED}", num_train_epochs=EPOCHS, per_device_train_batch_size=BATCH_SIZE, gradient_accumulation_steps=cfg_sens["grad_acc"], learning_rate=PEFT_LR * cfg_sens["lr_scale"], warmup_ratio=0.1, weight_decay=0.01, eval_strategy="steps", eval_steps=cfg_sens["eval_steps"], save_steps=cfg_sens["eval_steps"], save_total_limit=1, load_best_model_at_end=True, metric_for_best_model="macro_f1", fp16=True, report_to="none", logging_steps=5),
            train_dataset=train_ds, eval_dataset=val_ds, tokenizer=tokenizer, data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=compute_metrics, callbacks=[EarlyStoppingCallback(early_stopping_patience=8)], smoothing=cfg_sens["smoothing"]
        )
        trainer.train(); res = trainer.evaluate()
        row = {"Type": "Temperature", "Value": temp, "Seed": SEED, "Macro_F1": res['eval_macro_f1'], "Weighted-F1": res['eval_weighted_f1'], "Accuracy": res['eval_accuracy'], "G-Mean": res['eval_g_mean'], "AUC": res['eval_auc']}
        append_to_csv(SENSITIVITY_FILE, row)
        save_experiment_full_data(trainer, model, tokenizer, IMG_DATA_DIR, f"Sens_Temp_{temp}_Seed_{SEED}")
        del model, trainer; torch.cuda.empty_cache(); gc.collect(); shutil.rmtree(f"./tmp_sens_T{temp}_S{SEED}", ignore_errors=True)

# --- Loss Weight ---
for lw in SENS_LOSS_WEIGHTS:
    for SEED in RANDOM_SEEDS:
        print(f"\n[Sensitivity-LoRA] Type=LossWeight | Value={lw} | Seed={SEED}")
        set_seed(SEED)
        train_df = get_custom_dataset(train_pool, cfg_sens["train"], SEED)
        cw = compute_class_weight('balanced', classes=np.unique(train_df['label']), y=train_df['label'])
        class_weights_np = torch.tensor(cw, dtype=torch.float).clamp(max=10.0).numpy()
        train_ds = Dataset.from_pandas(train_df).map(tokenize, batched=True).select_columns(["input_ids", "attention_mask", "label"])
        
        model = UnifiedModel({"name": "LoRA-Ours", "method": "peft", "loss_type": "original", "peft_type": "lora", "hsp": True, "memory_bank": True, "fusion_init": cfg_sens["fusion_init"]}).to(device)
        trainer = UnifiedTrainer(
            loss_type="original",
            class_weights=class_weights_np, cls_num_list=[],
            memory_loss=MemoryBank(128, NUM_LABELS, cfg_sens["memory_size"], cfg_sens["temperature"], TAIL_CLASSES, cfg_sens["tail_weight"], cfg_sens["warmup_steps"], 5).to(device),
            loss_weight=lw, is_peft=True, model=model, use_class_weight=True,
            args=TrainingArguments(output_dir=f"./tmp_sens_LW{lw}_S{SEED}", num_train_epochs=EPOCHS, per_device_train_batch_size=BATCH_SIZE, gradient_accumulation_steps=cfg_sens["grad_acc"], learning_rate=PEFT_LR * cfg_sens["lr_scale"], warmup_ratio=0.1, weight_decay=0.01, eval_strategy="steps", eval_steps=cfg_sens["eval_steps"], save_steps=cfg_sens["eval_steps"], save_total_limit=1, load_best_model_at_end=True, metric_for_best_model="macro_f1", fp16=True, report_to="none", logging_steps=5),
            train_dataset=train_ds, eval_dataset=val_ds, tokenizer=tokenizer, data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=compute_metrics, callbacks=[EarlyStoppingCallback(early_stopping_patience=8)], smoothing=cfg_sens["smoothing"]
        )
        trainer.train(); res = trainer.evaluate()
        row = {"Type": "LossWeight", "Value": lw, "Seed": SEED, "Macro_F1": res['eval_macro_f1'], "Weighted-F1": res['eval_weighted_f1'], "Accuracy": res['eval_accuracy'], "G-Mean": res['eval_g_mean'], "AUC": res['eval_auc']}
        append_to_csv(SENSITIVITY_FILE, row)
        save_experiment_full_data(trainer, model, tokenizer, IMG_DATA_DIR, f"Sens_LossWeight_{lw}_Seed_{SEED}")
        del model, trainer; torch.cuda.empty_cache(); gc.collect(); shutil.rmtree(f"./tmp_sens_LW{lw}_S{SEED}", ignore_errors=True)

print(f"\n{'='*80}\nSST-5 DONE.\n{'='*80}")

# ==================== 7. [ENHANCED] Final Auto-Summary Report ====================
def generate_final_summary(csv_path, tail_classes):
    import os
    import pandas as pd
    
    if not os.path.exists(csv_path):
        print(f"!!! Error: Results file {csv_path} not found.")
        return

    print(f"\n{'='*80}\n>>> GENERATING FINAL SUMMARY REPORT (ALL METRICS)...\n{'='*80}")
    try: df = pd.read_csv(csv_path)
    except: return

    if df.empty: return

    # 1. Calc Tail F1
    tail_cols = [f"F1_Class_{c}" for c in tail_classes]
    available_tail = [c for c in tail_cols if c in df.columns]
    if available_tail:
        df["Tail_F1"] = df[available_tail].mean(axis=1)
    
    # 2. Define ALL Metrics to Summarize
    target_metrics = [
        "Macro-F1", "Weighted-F1", "Accuracy", "Balanced_Acc", 
        "G-Mean", "AUC", "Tail_F1",
        "Train_Time_Sec", "Inference_Time_Sec", "Peak_Memory_MB", "Params_M"
    ]
    
    # Filter only existing metrics in the CSV
    metrics = [m for m in target_metrics if m in df.columns]
    
    summary_rows = []
    grouped = df.groupby(["N", "Method"])

    for (n_val, method), group in grouped:
        row = {"N": n_val, "Method": method}
        
        # Best Seed
        best_idx = group["Macro-F1"].idxmax()
        row["Best (Seed/F1)"] = f"Seed {int(group.loc[best_idx, 'Seed'])}: {group.loc[best_idx, 'Macro-F1']:.4f}"

        # Mean +/- Std
        for m in metrics:
            vals = group[m].dropna().tolist()
            if vals:
                mean_val = np.mean(vals)
                std_val = np.std(vals, ddof=1)
                row[f"{m} (Mean±Std)"] = f"{mean_val:.4f} ± {std_val:.4f}"
                if m == "Macro-F1":
                    row[f"{m} Raw"] = str(vals)
        
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    # Sort
    summary_df = summary_df.sort_values(by=["N", "Method"])
    
    # Print
    try: print("\n" + summary_df.to_markdown(index=False))
    except: print(summary_df)

    # Save Markdown
    out_file = csv_path.replace(".csv", "_Summary.md")
    with open(out_file, "w") as f:
        f.write(f"# Final Experiment Summary\n")
        f.write(f"Generated Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        try: f.write(summary_df.to_markdown(index=False))
        except: f.write(str(summary_df))
    print(f"\n>>> Full Summary Saved to: {out_file}")

if __name__ == "__main__":
    generate_final_summary(MAIN_RESULTS_FILE, TAIL_CLASSES)