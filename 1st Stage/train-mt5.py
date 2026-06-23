import os
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# --- Konfigurasi GPU ---
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["NCCL_P2P_DISABLE"] = "1"

# --- Load dataset ---
df = pd.read_csv("ESC1.csv")
df = df.sort_values(["conv_id", "turn_id"]).reset_index(drop=True)

# --- Buat prompt multitask ---
def build_prompts(df: pd.DataFrame):
    inputs, outputs = [], []
    for conv_id, conv_df in df.groupby("conv_id"):
        conv_df = conv_df.sort_values("turn_id").reset_index(drop=True)
        for idx, row in conv_df.iterrows():
            if row["speaker"].lower() != "supporter":
                continue

            seeker_input = None
            for j in range(idx - 1, -1, -1):
                if conv_df.loc[j, "speaker"].lower() == "seeker":
                    seeker_input = conv_df.loc[j, "content"]
                    break
            if seeker_input is None:
                continue

            strategy = row.get("strategy", "General Support")
            supporter_response = row["content"]

            # Task A: Strategy classification
            inputs.append(f"Seeker: {seeker_input}\nTugas: Prediksi strategi")
            outputs.append(strategy)

            # Task B: Response generation
            inputs.append(f"Seeker: {seeker_input}\nStrategy: {strategy}\nTugas: Buat respons")
            outputs.append(supporter_response)
    return inputs, outputs

inputs, outputs = build_prompts(df)
dataset = Dataset.from_dict({"input_text": inputs, "target_text": outputs}).train_test_split(test_size=0.1, seed=42)

# --- Tokenizer & Model ---
MODEL_NAME = "google/mt5-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess(batch):
    model_inputs = tokenizer(batch["input_text"], max_length=512, truncation=True)
    labels = tokenizer(batch["target_text"], max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_ds = dataset.map(preprocess, batched=True, remove_columns=["input_text", "target_text"])

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# --- TrainingArguments ---
training_args = Seq2SeqTrainingArguments(
    output_dir="checkpoints-mt5",
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    num_train_epochs=6,
    learning_rate=3e-5,
    weight_decay=0.01,
    logging_dir="checkpoints-mt5/logs",
    logging_steps=100,
    seed=42,
    report_to="none",
    bf16=True,   # A100 support bf16
)

# --- Trainer ---
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# --- Train ---
print("Starting multitask training with mT5-base on GPU A100...")
trainer.train()
print("Training finished.")

# --- Save ---
trainer.save_model("fine_tuned_mt5")
tokenizer.save_pretrained("fine_tuned_mt5")
print("Model saved to fine_tuned_mt5")
