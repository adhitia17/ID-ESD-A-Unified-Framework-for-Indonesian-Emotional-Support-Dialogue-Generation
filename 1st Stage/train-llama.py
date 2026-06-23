# train_grok_multitask.py
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
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

os.makedirs("checkpoints-llama/logs", exist_ok=True)
log_file = open("checkpoints-llama/logs/training.log", "w")
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

def build_prompts_multitask(df: pd.DataFrame):
    prompts = []
    for conv_id, conv_df in df.groupby("conv_id"):
        conv_df = conv_df.sort_values("turn_id").reset_index(drop=True)

        for idx, row in conv_df.iterrows():
            if row["speaker"].strip().lower() != "supporter":
                continue

            # cari seeker terakhir sebelum turn ini
            seeker_input = None
            for j in range(idx - 1, -1, -1):
                if conv_df.loc[j, "speaker"].strip().lower() == "seeker":
                    seeker_input = str(conv_df.loc[j, "content"])
                    seeker_idx = j
                    break
            if seeker_input is None:
                continue

            # history sebelum seeker_input
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

            # Buat dua prompt: prediksi strategy + response
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

# --- Load model ---
max_seq_length = 2048
dtype = None
load_in_4bit = True

print("Loading base model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)



print("Applying LoRA...")
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# --- TrainingArguments ---
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    max_steps=20000,
    learning_rate=2e-4,
    #fp16=not torch.cuda.is_bf16_supported(),
    #bf16=torch.cuda.is_bf16_supported(),
    logging_dir="checkpoints-llama/logs",
    logging_strategy="steps",   # pastikan log ditulis
    logging_steps=10,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="checkpoints-llama",
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,
    eval_strategy="no",
    report_to="none",
)

# --- Trainer ---
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_kwargs={"skip_prepare_dataset": True},
    args=training_args,
)

print("Starting multitask training...")
trainer.train()
print("Training finished.")

trainer.save_model("fine_tuned_llama")
tokenizer.save_pretrained("fine_tuned_llama")
print("Model and tokenizer saved to fine_tuned_llama")
