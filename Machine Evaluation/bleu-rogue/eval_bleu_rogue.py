import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from tqdm import tqdm
import numpy as np
import os

# ==========================
# KONFIGURASI
# ==========================
REFERENCE_FILE = "eval_subset.csv"
PREDICTION_FILES = [
    "results_gemma_eval.csv",
    "results_llama_eval.csv",
    "results_mt5_eval.csv",
]
OUTPUT_FILE = "evaluation_scores.csv"

# ==========================
# LOAD DATA REFERENSI
# ==========================
print(f"📄 Loading reference from {REFERENCE_FILE}")
ref_df = pd.read_csv(REFERENCE_FILE)
ref_df["conv_id"] = ref_df["conv_id"].astype(str)
ref_df = ref_df[ref_df["speaker"].str.lower() == "supporter"].reset_index(drop=True)
ref_last = ref_df.groupby("conv_id").tail(1).reset_index(drop=True)

# ==========================
# EVALUASI SETIAP MODEL
# ==========================
smooth_fn = SmoothingFunction().method1
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

results_list = []

for pred_file in PREDICTION_FILES:
    if not os.path.exists(pred_file):
        print(f"⚠️ File not found: {pred_file}, skipping.")
        continue

    model_name = os.path.splitext(os.path.basename(pred_file))[0]
    print(f"\n🚀 Evaluating model: {model_name}")

    pred_df = pd.read_csv(pred_file)
    pred_df["conv_id"] = pred_df["conv_id"].astype(str)
    pred_df = pred_df[pred_df["speaker"].str.lower() == "supporter"].reset_index(drop=True)
    pred_last = pred_df.groupby("conv_id").tail(1).reset_index(drop=True)

    merged = pd.merge(ref_last, pred_last, on="conv_id", suffixes=("_ref", "_pred"))
    print(f"✅ Total paired conversations: {len(merged)}")

    bleu_scores, rouge1_scores, rouge2_scores, rougeL_scores = [], [], [], []

    for _, row in tqdm(merged.iterrows(), total=len(merged), desc=f"Evaluating {model_name}"):
        ref_text = str(row["content_ref"]).strip()
        pred_text = str(row["content_pred"]).strip()

        # BLEU
        ref_tokens = [ref_text.split()]
        pred_tokens = pred_text.split()
        bleu = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smooth_fn)
        bleu_scores.append(bleu)

        # ROUGE
        rouge = scorer.score(ref_text, pred_text)
        rouge1_scores.append(rouge["rouge1"].fmeasure)
        rouge2_scores.append(rouge["rouge2"].fmeasure)
        rougeL_scores.append(rouge["rougeL"].fmeasure)

    results = {
        "model": model_name,
        "samples": len(merged),
        "BLEU": np.mean(bleu_scores),
        "ROUGE-1": np.mean(rouge1_scores),
        "ROUGE-2": np.mean(rouge2_scores),
        "ROUGE-L": np.mean(rougeL_scores),
    }
    results_list.append(results)

# ==========================
# SIMPAN & CETAK HASIL
# ==========================
if results_list:
    result_df = pd.DataFrame(results_list)
    result_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("\n📊 Final Evaluation Summary:")
    print(result_df.round(4).to_string(index=False))
    print(f"\n✅ Results saved to {OUTPUT_FILE}")
else:
    print("❌ No results generated.")
