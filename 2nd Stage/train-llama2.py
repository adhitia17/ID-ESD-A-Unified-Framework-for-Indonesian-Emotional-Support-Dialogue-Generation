# train-llama2.py – END-TO-END (emotion + strategy + response)
# CONFIG MATCH: train-llama.py (H100, FP8, Flash Attn 2, Unsloth)
# LOG: progress.jsonl (loss, grad_norm, lr, epoch)

import json
import logging
import os
import sys
import re
from datasets import Dataset
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from transformers.trainer_callback import TrainerCallback

# ========================================
# 1. LOGGING (SAMA SEPERTI train-llama.py + JSONL)
# ========================================
os.makedirs("4llama/checkpoints-llama/logs", exist_ok=True)

# JSONL progress logger
progress_logger = logging.getLogger("progress")
progress_handler = logging.FileHandler("4llama/checkpoints-llama/logs/progress.jsonl", encoding='utf-8')
progress_handler.setFormatter(logging.Formatter('%(message)s'))
progress_logger.addHandler(progress_handler)
progress_logger.setLevel(logging.INFO)

# Standard logger (sama seperti train-llama.py)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler("4llama/checkpoints-llama/logs/training.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)
logger = logging.getLogger(__name__)

# ========================================
# 2. GPU & DATA (SESUAI train-llama2.py)
# ========================================
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

DATA_PATH = "augmenting/final_data/dataset_esc_augmenting_final_mapped.json"
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)
logger.info(f"Loaded {len(data)} conversations")

# ========================================
# 3. VALID EMOSI & STRATEGI (SESUAI LIST BARU)
# ========================================
VALID_EMOTIONS = {
    "malu", "sedih", "takut", "rindu", "ragu", "lega", "bahagia", "optimis",
    "marah", "jijik", "bosan", "bangga", "bingung", "cemas", "depresi",
    "terharu", "penasaran", "harapan", "kesepian", "netral"
}

VALID_STRATEGIES = {
    "bertanya", "merefleksikan perasaan", "memberikan penegasan dan penyemangat",
    "memberikan saran", "mengulang atau memparafrasekan", "memberikan informasi",
    "berbagi pengalaman pribadi"
}

# ========================================
# 4. END-TO-END PROMPT (SESUAI TUJUAN)
# ========================================
END_TO_END_PROMPT = """<s>[INST] {history}

Seeker: {seeker_input}
[/INST] (emotion: {seeker_emotion}) (strategy: {strategy}) {supporter_response} </s>"""

# ========================================
# 5. BUILD PROMPTS (AMAN DARI NoneType)
# ========================================
def build_end_to_end_prompts(data):
    prompts = []
    for conv_idx, conv in enumerate(data):
        dialog = conv.get("dialog", [])
        if not isinstance(dialog, list) or len(dialog) < 2:
            continue

        history = []
        i = 0
        while i < len(dialog) - 1:
            curr = dialog[i]
            next_t = dialog[i + 1]

            # Validasi speaker
            curr_speaker = str(curr.get("speaker", "") or "").strip().lower()
            next_speaker = str(next_t.get("speaker", "") or "").strip().lower()

            if curr_speaker != "seeker" or next_speaker != "supporter":
                i += 1
                continue

            s_input = str(curr.get("content", "") or "").strip()
            supp_resp = str(next_t.get("content", "") or "").strip()
            if not s_input or not supp_resp:
                i += 1
                continue

            # Emotion
            raw_emo = curr.get("emotion")
            s_emo = "netral"
            if isinstance(raw_emo, str) and raw_emo.strip():
                s_emo = raw_emo.strip().lower()
            s_emo = s_emo if s_emo in VALID_EMOTIONS else "netral"

            # Strategy
            raw_strategy = next_t.get("strategy")
            strategy = "bertanya"
            if isinstance(raw_strategy, str) and raw_strategy.strip():
                strategy = raw_strategy.strip().lower()
            strategy = strategy if strategy in VALID_STRATEGIES else "bertanya"

            # Build
            h = "\n".join(history) if history else ""
            prompt = END_TO_END_PROMPT.format(
                history=h,
                seeker_input=s_input,
                seeker_emotion=s_emo,
                strategy=strategy,
                supporter_response=supp_resp
            )
            prompts.append(prompt)

            history.append(f"Seeker: {s_input}")
            history.append(f"Supporter: {supp_resp}")
            i += 1

    logger.info(f"Generated {len(prompts)} end-to-end prompts")
    return prompts

prompts = build_end_to_end_prompts(data)
if len(prompts) == 0:
    logger.error("NO PROMPTS! Check dataset.")
    sys.exit(1)

dataset = Dataset.from_dict({"text": prompts}).shuffle(seed=42)

# ========================================
# 6. MODEL (SESUAI train-llama.py: NO dtype, Flash Attn 2 di from_pretrained)
# ========================================
logger.info("Loading model with Flash Attention 2 (H100)...")
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
    attn_implementation="flash_attention_2",  # SESUAI train-llama
)

model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# ========================================
# 7. LOG CALLBACK (progress.jsonl)
# ========================================
class ProgressCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            entry = {"step": state.global_step}
            if state.epoch:
                entry["epoch"] = round(state.epoch, 2)
            if "loss" in logs:
                entry["loss"] = round(logs["loss"], 6)
            if "grad_norm" in logs:
                entry["grad_norm"] = round(logs["grad_norm"], 6)
            if "learning_rate" in logs:
                entry["learning_rate"] = logs["learning_rate"]
            progress_logger.info(json.dumps(entry, ensure_ascii=False))

# ========================================
# 8. TRAINING ARGS (MATCH train-llama.py, FP8 aktif)
# ========================================
training_args = TrainingArguments(
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    warmup_steps=100,        # SESUAI train-llama
    max_steps=3000,          # Dikontrol (bukan 10000)
    learning_rate=1e-4,
    #fp8=True,                # SESUAI train-llama
    optim="adamw_torch_fused",
    logging_steps=50,        # SESUAI train-llama
    save_steps=500,
    save_total_limit=3,
    output_dir="4llama/checkpoints-llama",
    logging_dir="4llama/checkpoints-llama/logs",
    report_to="none",
    dataloader_num_workers=4,
    torch_compile=True,
    remove_unused_columns=False,
    gradient_checkpointing=True,
    log_level="info",
    logging_strategy="steps",
)

# ========================================
# 9. TRAIN
# ========================================
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    args=training_args,
    packing=False,
    callbacks=[ProgressCallback()],
)

logger.info("Starting H100-optimized END-TO-END training...")
trainer.train()

# ========================================
# 10. SAVE
# ========================================
final_path = "4llama/fine_tuned_llama_end2end"
trainer.save_model(final_path)
tokenizer.save_pretrained(final_path)
logger.info(f"Training selesai! Model disimpan: {final_path}")