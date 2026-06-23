import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import re
from tqdm import tqdm

# --- Konfigurasi GPU ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["NCCL_P2P_DISABLE"] = "1"

# --- Fungsi bantu: bersihkan output model ---
def clean_output(text: str):
    """Hilangkan markup dan ambil hanya hasil setelah [/INST]."""
    if "[/INST]" in text:
        text = text.split("[/INST]")[-1]
    text = re.sub(r"</s>|<s>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --- Load model hasil fine-tuning ---
MODEL_DIR = "fine_tuned_gemma"  # ganti ke model lain jika perlu
print(f"🔹 Loading model from: {MODEL_DIR}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --- Load eval subset dataset ---
EVAL_FILE = "eval_subset.csv"
print(f"📄 Loading evaluation subset from: {EVAL_FILE}")

df = pd.read_csv(EVAL_FILE)
df = df.sort_values(["conv_id", "turn_id"]).reset_index(drop=True)

conv_ids = df["conv_id"].unique().tolist()
print(f"✅ Total conversations for evaluation: {len(conv_ids)}")

output_rows = []

# --- Loop semua percakapan di eval_subset ---
for conv_id in tqdm(conv_ids, desc="Running evaluation inference"):
    conv_df = df[df["conv_id"] == conv_id].reset_index(drop=True)

    # Masukkan semua utterance asli dulu ke hasil akhir
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

    # Buat history percakapan sebelumnya
    history_list = []
    for i in range(last_seeker["turn_id"]):
        sp = conv_df.loc[i, "speaker"].capitalize()
        content = str(conv_df.loc[i, "content"]).strip()
        history_list.append(f"{sp}: {content}")
    history = "\n".join(history_list)

    # --- Prediksi strategy ---
    strategy_prompt = f"""<s>[INST] {history}

Seeker: {seeker_input}

Tugas Anda adalah memprediksi strategy terbaik yang harus digunakan supporter.
[/INST]"""
    inputs = tokenizer(strategy_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        strategy_output = model.generate(
            **inputs,
            max_new_tokens=64,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
    raw_strategy = tokenizer.decode(strategy_output[0], skip_special_tokens=False)
    pred_strategy = clean_output(raw_strategy)

    # --- Prediksi respons supporter ---
    response_prompt = f"""<s>[INST] {history}

Seeker: {seeker_input}

Gunakan strategy berikut dalam jawabanmu: {pred_strategy}
[/INST]"""
    inputs = tokenizer(response_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        response_output = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
    raw_response = tokenizer.decode(response_output[0], skip_special_tokens=False)
    cleaned_response = clean_output(raw_response)

    # Tambahkan hasil prediksi bersih ke output
    next_turn_id = conv_df["turn_id"].max() + 1
    output_rows.append({
        "conv_id": conv_id,
        "turn_id": next_turn_id,
        "speaker": "supporter",
        "content": cleaned_response,
        "strategy": pred_strategy
    })

# --- Simpan hasil ke CSV ---
output_df = pd.DataFrame(output_rows)
output_path = f"results_gemma_eval.csv"
output_df.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"\n✅ Hasil evaluasi disimpan ke: {output_path}")
print(output_df.tail(5))
