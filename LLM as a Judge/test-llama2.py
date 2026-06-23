# test-llama2-eval.py
# BLEU & ROUGE disimpan per conv_id + rata-rata di summary

import os
import time
import json
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from unsloth import FastLanguageModel
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
import nltk
import torch
from llm_client import query_llm
from sklearn.metrics import accuracy_score

# ==========================
# FIX NLTK
# ==========================
nltk_data_path = os.path.join(os.environ["VIRTUAL_ENV"], "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.insert(0, nltk_data_path)
nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
nltk.download('punkt_tab', download_dir=nltk_data_path, quiet=True)

# ==========================
# KONFIGURASI
# ==========================
EVAL_FILE = "eval_subset2.csv"
MODEL_DIR = "/mnt/data-hps/adhitiae/vscode/ESC/4llama/checkpoints-llama/checkpoint-2500"
PRED_MODEL_NAME = "fine_tuned_llama"

RAW_OUTPUT = "llm_judge_raw.csv"
SUMMARY_OUTPUT = "llm_judge_summary.csv"
INFERENCE_OUTPUT = "inference_results.csv"

MAX_SAMPLES = 150
MAX_TOKENS = 512
TEMPERATURE = 0.0
RATE_LIMIT_DELAY = 0.3

CRITERIA = [
    ("empathy", "Empathy / Supportiveness (1-5)"),
    ("helpfulness", "Helpfulness / Usefulness (1-5)"),
    ("relevance", "Relevance / On-topic (1-5)"),
    ("fluency", "Fluency / Grammaticality (1-5)"),
    ("strategy_adherence", "Strategy Adherence (1-5)"),
]

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["NCCL_P2P_DISABLE"] = "1"

# ==========================
# LOAD MODEL
# ==========================
print(f"Loading model: {MODEL_DIR}")
model, tokenizer = FastLanguageModel.from_pretrained(
    MODEL_DIR, max_seq_length=2048, load_in_4bit=True, dtype=None, device_map="auto"
)
FastLanguageModel.for_inference(model)

print("Warming up...")
dummy = tokenizer("<s>[INST] Halo [/INST]", return_tensors="pt")
dummy = {k: v.to(model.device) for k, v in dummy.items()}
with torch.no_grad():
    _ = model.generate(**dummy, max_new_tokens=1)
print("Model siap!\n")

# ==========================
# LOAD DATA
# ==========================
ref_df = pd.read_csv(EVAL_FILE).sort_values(["conv_id", "turn_id"]).reset_index(drop=True)
conv_ids = ref_df["conv_id"].unique().tolist()[:MAX_SAMPLES]
print(f"Evaluating {len(conv_ids)} convs\n")

# ==========================
# PROMPT & PARSER
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
- strategy_adherence: seberapa sesuai respons dengan strategi yang diharapkan ({expected_strategy})

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
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match: return None
    try: return json.loads(match.group(0))
    except: 
        try: return json.loads(match.group(0).replace("'", "\""))
        except: return None

# ==========================
# INFERENCE LOOP
# ==========================
inference_results = []
judge_results = []

true_emotions = []
pred_emotions = []
true_strategies = []
pred_strategies = []

bleu_scores = []
rouge1_scores = []
rougeL_scores = []

valid_convs = 0

for conv_id in tqdm(conv_ids, desc="Evaluating"):
    conv = ref_df[ref_df["conv_id"] == conv_id].reset_index(drop=True)

    # --- LAST SEEKER dengan EMOTION ---
    seeker_rows = conv[
        (conv["speaker"].str.lower() == "seeker") &
        (conv["emotion"].notna()) &
        (conv["emotion"].str.strip() != "")
    ]
    if seeker_rows.empty: 
        continue
    last_seeker = seeker_rows.iloc[-1]
    seeker_input = str(last_seeker["content"]).strip()
    true_emotion = str(last_seeker["emotion"]).strip().lower()

    # --- LAST SUPPORTER dengan STRATEGY ---
    supporter_rows = conv[
        (conv["speaker"].str.lower() == "supporter") &
        (conv["strategy"].notna()) &
        (conv["strategy"].str.strip() != "")
    ]
    if supporter_rows.empty: 
        continue
    last_supporter = supporter_rows.iloc[-1]
    ref_response = str(last_supporter["content"]).strip()
    true_strategy = str(last_supporter["strategy"]).strip().lower()

    valid_convs += 1

    # --- History ---
    history_df = conv[conv["turn_id"] < last_seeker["turn_id"]]
    history = "\n".join(f"{r['speaker'].capitalize()}: {r['content']}" for _, r in history_df.iterrows())

    # --- Generate ---
    prompt = f"<s>[INST] {history}\n\nSeeker: {seeker_input} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048-200)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    raw = tokenizer.decode(out[0], skip_special_tokens=False)

    # --- Parse ---
    text = raw.split("[/INST]")[-1].split("</s>")[0].strip()
    text = re.sub(r"\s+", " ", text)
    emo = re.search(r'\(emotion:\s*([^)]+)\)', text)
    strat = re.search(r'\(strategy:\s*([^)]+)\)', text)
    clean = re.sub(r'\(emotion:[^)]+\)\s*', '', text)
    clean = re.sub(r'\(strategy:[^)]+\)\s*', '', clean).strip()

    pred_emo = emo.group(1).strip().lower() if emo else "netral"
    pred_strat = strat.group(1).strip().lower() if strat else "bertanya"
    pred_resp = clean

    # --- BLEU & ROUGE ---
    bleu_val = rouge1_val = rougeL_val = np.nan
    if ref_response:
        try:
            ref_tok = word_tokenize(ref_response)
            pred_tok = word_tokenize(pred_resp)
            bleu_val = sentence_bleu([ref_tok], pred_tok, smoothing_function=SmoothingFunction().method4)
            bleu_scores.append(bleu_val)

            scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
            r = scorer.score(ref_response, pred_resp)
            rouge1_val = r["rouge1"].fmeasure
            rougeL_val = r["rougeL"].fmeasure
            rouge1_scores.append(rouge1_val)
            rougeL_scores.append(rougeL_val)
        except Exception as e:
            print(f"BLEU/ROUGE error (conv {conv_id}): {e}")

    # --- Akurasi ---
    true_emotions.append(true_emotion)
    pred_emotions.append(pred_emo)
    true_strategies.append(true_strategy)
    pred_strategies.append(pred_strat)

    # --- Simpan semua turn ---
    for _, row in conv.iterrows():
        inference_results.append({
            "conv_id": row["conv_id"], "turn_id": row["turn_id"], "speaker": row["speaker"],
            "content": row["content"], "emotion": row.get("emotion", ""), "strategy": row.get("strategy", ""),
            "pred_emotion": "", "pred_strategy": "", "pred_response": "",
            "bleu": "", "rouge1": "", "rougeL": ""
        })

    # --- Simpan prediksi + BLEU/ROUGE ---
    next_id = conv["turn_id"].max() + 1
    inference_results.append({
        "conv_id": conv_id, "turn_id": next_id, "speaker": "supporter",
        "content": ref_response, "emotion": "", "strategy": true_strategy,
        "pred_emotion": pred_emo, "pred_strategy": pred_strat, "pred_response": pred_resp,
        "bleu": round(bleu_val, 4) if not np.isnan(bleu_val) else "",
        "rouge1": round(rouge1_val, 4) if not np.isnan(rouge1_val) else "",
        "rougeL": round(rougeL_val, 4) if not np.isnan(rougeL_val) else ""
    })

    # --- LLM Judge ---
    prompt = build_prompt(history, seeker_input, ref_response, pred_resp, true_strategy)
    messages = [{"role": "system", "content": "You are an impartial evaluator."},
                {"role": "user", "content": prompt}]

    try:
        response = query_llm(messages, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
    except Exception as e:
        print(f"API error (conv {conv_id}): {e}")
        continue

    parsed = parse_json_safe(response)
    if not parsed:
        print(f"Parse error (conv {conv_id})")
        continue

    scores = parsed.get("scores", {})
    justs = parsed.get("justifications", {})
    result = {
        "conv_id": conv_id, "model": PRED_MODEL_NAME,
        "bleu": round(bleu_val, 4) if not np.isnan(bleu_val) else np.nan,
        "rouge1": round(rouge1_val, 4) if not np.isnan(rouge1_val) else np.nan,
        "rougeL": round(rougeL_val, 4) if not np.isnan(rougeL_val) else np.nan
    }
    for key, _ in CRITERIA:
        result[f"score_{key}"] = scores.get(key, np.nan)
        result[f"just_{key}"] = justs.get(key, "")
    judge_results.append(result)

    time.sleep(RATE_LIMIT_DELAY)

# ==========================
# SIMPAN HASIL
# ==========================
# 1. Inference + BLEU/ROUGE
pd.DataFrame(inference_results).to_csv(INFERENCE_OUTPUT, index=False, encoding="utf-8-sig")
print(f"\nInference + BLEU/ROUGE: {INFERENCE_OUTPUT}")

# 2. Metrics
emotion_acc = accuracy_score(true_emotions, pred_emotions) if true_emotions else 0
strategy_acc = accuracy_score(true_strategies, pred_strategies) if true_strategies else 0
avg_bleu = np.mean(bleu_scores) if bleu_scores else 0
avg_rouge1 = np.mean(rouge1_scores) if rouge1_scores else 0
avg_rougeL = np.mean(rougeL_scores) if rougeL_scores else 0

print(f"Valid conversations: {valid_convs}")
print(f"Emotion Adherence: {emotion_acc:.4f}")
print(f"Strategy Accuracy: {strategy_acc:.4f}")
print(f"BLEU: {avg_bleu:.4f}")
print(f"ROUGE-1: {avg_rouge1:.4f}")
print(f"ROUGE-L: {avg_rougeL:.4f}")

# 3. Raw Judge + BLEU/ROUGE
raw_df = pd.DataFrame(judge_results)
raw_df.to_csv(RAW_OUTPUT, index=False, encoding="utf-8-sig")
print(f"Raw judge + metrics: {RAW_OUTPUT}")

# 4. Summary
summary = []
for model in raw_df["model"].unique():
    mdf = raw_df[raw_df["model"] == model]
    row = {
        "model": model,
        "samples": len(mdf),
        "avg_emotion_adherence": emotion_acc,
        "avg_strategy_accuracy": strategy_acc,
        "avg_bleu": avg_bleu,
        "avg_rouge1": avg_rouge1,
        "avg_rougeL": avg_rougeL
    }
    for key, _ in CRITERIA:
        row[f"avg_{key}"] = mdf[f"score_{key}"].astype(float).mean()
    summary.append(row)

summary_df = pd.DataFrame(summary)
summary_df.to_csv(SUMMARY_OUTPUT, index=False, encoding="utf-8-sig")
print(f"\nSummary + BLEU/ROUGE: {SUMMARY_OUTPUT}")
print("\nRINGKASAN HASIL:")
print(summary_df[[
    "model", "samples",
    "avg_emotion_adherence", "avg_strategy_accuracy",
    "avg_bleu", "avg_rouge1", "avg_rougeL",
    "avg_empathy", "avg_helpfulness", "avg_relevance",
    "avg_fluency", "avg_strategy_adherence"
]].round(4).to_string(index=False))