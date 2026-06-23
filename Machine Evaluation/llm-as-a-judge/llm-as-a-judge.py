import os
import time
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from llm_client import query_llm  # gunakan klien kamu sendiri

# ==========================
# KONFIGURASI
# ==========================
EVAL_FILE = "eval_subset.csv"
PRED_FILES = {
    "gemma": "results_gemma_eval.csv",
    "llama": "results_llama_eval.csv",
    "mt5":   "results_mt5_eval.csv",
}
RAW_OUTPUT = "llm_judge_raw.csv"
SUMMARY_OUTPUT = "llm_judge_summary.csv"

MAX_SAMPLES = 150
MAX_TOKENS = 512
TEMPERATURE = 0.0

CRITERIA = [
    ("empathy", "Empathy / Supportiveness (1-5)"),
    ("helpfulness", "Helpfulness / Usefulness (1-5)"),
    ("relevance", "Relevance / On-topic (1-5)"),
    ("fluency", "Fluency / Grammaticality (1-5)"),
    ("strategy_adherence", "Strategy adherence (1-5)"),
]

# ==========================
# FUNGSI BANTU
# ==========================
def build_prompt(history, seeker, reference, candidate, expected_strategy):
    return f"""
Anda adalah evaluator objektif.
Nilailah *respons kandidat* (supporter) terhadap percakapan berikut.

Riwayat percakapan:
{history}

Seeker: {seeker}

Respons referensi (supporter asli):
{reference}

Respons kandidat (yang akan dievaluasi):
{candidate}

Strategi yang diharapkan: {expected_strategy}

Tugas Anda: Beri penilaian numerik (1-5, 5 terbaik) dan alasan singkat.
Kriteria:
- empathy: sejauh mana jawaban menunjukkan empati & dukungan emosional
- helpfulness: seberapa membantu dan relevan nasihatnya
- relevance: apakah tetap fokus pada topik
- fluency: kelancaran dan tata bahasa
- strategy_adherence: kesesuaian dengan strategi yang diharapkan

Kembalikan **hanya dalam format JSON berikut**:
{{
  "scores": {{
    "empathy": <1-5>,
    "helpfulness": <1-5>,
    "relevance": <1-5>,
    "fluency": <1-5>,
    "strategy_adherence": <1-5>
  }},
  "justifications": {{
    "empathy": "<alasan singkat>",
    "helpfulness": "<alasan singkat>",
    "relevance": "<alasan singkat>",
    "fluency": "<alasan singkat>",
    "strategy_adherence": "<alasan singkat>"
  }}
}}
"""

def parse_json_safe(text):
    import re
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        cleaned = match.group(0).replace("'", "\"").strip()
        try:
            return json.loads(cleaned)
        except Exception:
            return None

# ==========================
# LOAD DATA
# ==========================
print("📂 Loading data...")
ref_df = pd.read_csv(EVAL_FILE).sort_values(["conv_id", "turn_id"]).reset_index(drop=True)
conv_ids = ref_df["conv_id"].unique().tolist()[:MAX_SAMPLES]

preds = {}
for name, path in PRED_FILES.items():
    if os.path.exists(path):
        preds[name] = pd.read_csv(path).sort_values(["conv_id", "turn_id"]).reset_index(drop=True)
    else:
        print(f"⚠️ File not found: {path}")

if not preds:
    raise ValueError("Tidak ada file prediksi yang ditemukan!")

# ==========================
# PROSES PENILAIAN
# ==========================
results = []

for conv_id in tqdm(conv_ids, desc="Evaluating conversations"):
    conv_df = ref_df[ref_df["conv_id"] == conv_id].reset_index(drop=True)
    seeker_rows = conv_df[conv_df["speaker"].str.lower() == "seeker"]
    if seeker_rows.empty:
        continue
    last_seeker_idx = seeker_rows.index[-1]
    history_df = conv_df.loc[:last_seeker_idx - 1]
    history_text = "\n".join(f"{r.speaker.capitalize()}: {r.content}" for _, r in history_df.iterrows())
    seeker_text = conv_df.loc[last_seeker_idx, "content"]

    next_rows = conv_df.loc[last_seeker_idx + 1:]
    next_supporters = next_rows[next_rows["speaker"].str.lower() == "supporter"]
    ref_candidate = next_supporters.iloc[0]["content"] if not next_supporters.empty else ""
    expected_strategy = next_supporters.iloc[0].get("strategy", "") if not next_supporters.empty else ""

    for model_name, pdf in preds.items():
        model_conv = pdf[pdf["conv_id"] == conv_id]
        if model_conv.empty:
            continue
        model_supporters = model_conv[model_conv["speaker"].str.lower() == "supporter"]
        candidate = model_supporters.iloc[-1]["content"] if not model_supporters.empty else model_conv.iloc[-1]["content"]

        prompt = build_prompt(history_text, seeker_text, ref_candidate, candidate, expected_strategy)
        messages = [{"role": "system", "content": "You are an impartial evaluator."},
                    {"role": "user", "content": prompt}]

        try:
            response = query_llm(messages, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
        except Exception as e:
            print(f"❌ API error ({model_name}, conv {conv_id}): {e}")
            continue

        parsed = parse_json_safe(response)
        if not parsed:
            print(f"⚠️ Gagal parse JSON (model {model_name}, conv {conv_id})")
            continue

        scores = parsed.get("scores", {})
        justs = parsed.get("justifications", {})
        result = {"conv_id": conv_id, "model": model_name}
        for key, _ in CRITERIA:
            result[f"score_{key}"] = scores.get(key, np.nan)
            result[f"just_{key}"] = justs.get(key, "")
        results.append(result)

        time.sleep(0.3)  # hindari rate limit

# ==========================
# SIMPAN HASIL
# ==========================
raw_df = pd.DataFrame(results)
raw_df.to_csv(RAW_OUTPUT, index=False, encoding="utf-8-sig")
print(f"✅ Disimpan: {RAW_OUTPUT}")

summary = []
for model in raw_df["model"].unique():
    mdf = raw_df[raw_df["model"] == model]
    row = {"model": model, "samples": len(mdf)}
    for key, _ in CRITERIA:
        row[f"avg_{key}"] = mdf[f"score_{key}"].astype(float).mean()
    summary.append(row)

summary_df = pd.DataFrame(summary)
summary_df.to_csv(SUMMARY_OUTPUT, index=False, encoding="utf-8-sig")

print("\n📊 Ringkasan Hasil:")
print(summary_df.round(3).to_string(index=False))
