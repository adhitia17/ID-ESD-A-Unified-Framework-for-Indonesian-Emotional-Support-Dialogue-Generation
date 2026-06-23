import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import re
from tqdm import tqdm

# --- Konfigurasi GPU ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["NCCL_P2P_DISABLE"] = "1"

# --- Fungsi bantu: membersihkan output ---
def clean_output(text: str):
    """Hilangkan markup dan whitespace berlebih"""
    text = re.sub(r"</s>|<pad>|<extra_id_\d+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --- Load model fine-tuned mT5 ---
MODEL_DIR = "fine_tuned_mt5"
print(f"🔹 Loading model from {MODEL_DIR} ...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to("cuda")
model.eval()

# --- Load eval subset ---
EVAL_FILE = "eval_subset.csv"
print(f"📄 Loading evaluation subset from: {EVAL_FILE}")

df = pd.read_csv(EVAL_FILE)
df = df.sort_values(["conv_id", "turn_id"]).reset_index(drop=True)
conv_ids = df["conv_id"].unique().tolist()
print(f"✅ Total conversations for evaluation: {len(conv_ids)}")

output_rows = []

# --- Loop untuk setiap percakapan ---
for conv_id in tqdm(conv_ids, desc="Running mT5 inference"):
    conv_df = df[df["conv_id"] == conv_id].reset_index(drop=True)

    # Simpan semua utterance asli dulu
    for _, row in conv_df.iterrows():
        output_rows.append({
            "conv_id": row["conv_id"],
            "turn_id": row["turn_id"],
            "speaker": row["speaker"],
            "content": row["content"],
            "strategy": row.get("strategy", "")
        })

    # Ambil utterance seeker terakhir untuk dijawab
    seekers = conv_df[conv_df["speaker"].str.lower() == "seeker"]
    if len(seekers) == 0:
        continue
    last_seeker = seekers.iloc[-1]
    seeker_input = str(last_seeker["content"]).strip()

    # Buat konteks percakapan
    history_list = []
    for i in range(last_seeker["turn_id"]):
        sp = conv_df.loc[i, "speaker"].capitalize()
        content = str(conv_df.loc[i, "content"]).strip()
        history_list.append(f"{sp}: {content}")
    history = "\n".join(history_list)

    # --- Task A: Prediksi strategy ---
    strategy_prompt = f"Seeker: {seeker_input}\nTugas: Prediksi strategi"
    inputs = tokenizer(strategy_prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        strategy_output = model.generate(
            **inputs,
            max_new_tokens=64,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    pred_strategy = clean_output(tokenizer.decode(strategy_output[0], skip_special_tokens=False))

    # --- Task B: Generate supporter response ---
    response_prompt = f"Seeker: {seeker_input}\nStrategy: {pred_strategy}\nTugas: Buat respons"
    inputs = tokenizer(response_prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        response_output = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    generated_response = clean_output(tokenizer.decode(response_output[0], skip_special_tokens=False))

    # Tambahkan hasil prediksi
    next_turn_id = conv_df["turn_id"].max() + 1
    output_rows.append({
        "conv_id": conv_id,
        "turn_id": next_turn_id,
        "speaker": "supporter",
        "content": generated_response,
        "strategy": pred_strategy
    })

# --- Simpan hasil ke CSV ---
output_df = pd.DataFrame(output_rows)
output_path = "results_mt5_eval.csv"
output_df.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"\n✅ Hasil inferensi disimpan ke: {output_path}")
print(output_df.tail(5))
