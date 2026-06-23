import pandas as pd
from datasets import Dataset
import torch
from trl import SFTTrainer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
import os
import sys

# --- Setup logging ke file + console (tee) ---
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

os.makedirs("checkpoints_gemma/logs", exist_ok=True)
log_file = open("checkpoints_gemma/logs/training.log", "w")
sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)

# --- Konfigurasi GPU ---
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["NCCL_P2P_DISABLE"] = "1"

# --- Load dataset ---
df = pd.read_csv("ESC1.csv")
df = df.sort_values(["conv_id", "turn_id"]).reset_index(drop=True)
print(f"Dataset loaded: {df.shape[0]} rows, {df['conv_id'].nunique()} conversations")

# Prompt template
strategy_prompt = """<s>[INST] {history}

Seeker: {seeker_input}

Tugas Anda adalah memprediksi strategy terbaik yang harus digunakan supporter.
[/INST] {strategy} </s>"""

response_prompt = """<s>[INST] {history}

Seeker: {seeker_input}

Gunakan strategy berikut dalam jawabanmu: {strategy}
[/INST] {supporter_response} </s>"""

# Membuat prompts

def build_prompts_multitask(df: pd.DataFrame):
    prompts = []
    for conv_id, conv_df in df.groupby("conv_id"):
        conv_df = conv_df.sort_values("turn_id").reset_index(drop=True)

        for idx, row in conv_df.iterrows():
            if row["speaker"].strip().lower() != "supporter":
                continue

            seeker_input = None
            for j in range(idx - 1, -1, -1):
                if conv_df.loc[j, "speaker"].strip().lower() == "seeker":
                    seeker_input = str(conv_df.loc[j, "content"])
                    seeker_idx = j
                    break
            if seeker_input is None:
                continue

            history_list = []
            for k in range(seeker_idx):
                sp = conv_df.loc[k, "speaker"].strip().capitalize()
                content = str(conv_df.loc[k, "content"]).strip()
                history_list.append(f"{sp}: {content}")
            history = "\n".join(history_list)

            supporter_response = str(row["content"]).strip()
            strat_val = row.get("strategy", None)
            if pd.isna(strat_val) or (isinstance(strat_val, str) and strat_val.strip() == ""):
                strategy = "General Support"
            else:
                strategy = str(strat_val).strip()

            s_prompt = strategy_prompt.format(history=history, seeker_input=seeker_input, strategy=strategy)
            r_prompt = response_prompt.format(history=history, seeker_input=seeker_input, strategy=strategy, supporter_response=supporter_response)

            prompts.append(s_prompt)
            prompts.append(r_prompt)

    return prompts

all_prompts = build_prompts_multitask(df)
print(f"Total multitask prompts: {len(all_prompts)}")

if len(all_prompts) == 0:
    raise ValueError("Tidak ditemukan prompt training, periksa struktur dataset.")

dataset = Dataset.from_dict({"text": all_prompts}).shuffle(seed=42)

# --- Load Gemma model for FULL finetuning ---
# Gunakan model Gemma yang open-access (misalnya gemma-2b)
BASE_MODEL = "google/gemma-2b"

use_bf16 = torch.cuda.is_bf16_supported()
print(f"Loading base model {BASE_MODEL} (for full fine-tuning)...")

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

try:
    model.gradient_checkpointing_enable()
except Exception:
    pass

# --- TrainingArguments ---
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    warmup_steps=50,
    max_steps=30000,
    learning_rate=1e-5,
    fp16=not use_bf16,
    bf16=use_bf16,
    logging_dir="checkpoints_gemma/logs",
    logging_strategy="steps",
    logging_steps=20,
    optim="adamw_torch",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="checkpoints_gemma",
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=2,
    report_to="none",
)

# SFTTrainer di TRL terbaru tidak menerima max_seq_length langsung di __init__.
# Atur panjang input melalui TrainingArguments atau gunakan dataset yang sudah dipotong.
def formatting_prompts_func(example):
    return example["text"]

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    formatting_func=formatting_prompts_func,
    args=training_args,
)

print("Starting full fine-tuning with Gemma...")
trainer.train()
print("Training finished.")

trainer.save_model("fine_tuned_gemma")
tokenizer.save_pretrained("fine_tuned_gemma")
print("Model and tokenizer saved to fine_tuned_gemma")