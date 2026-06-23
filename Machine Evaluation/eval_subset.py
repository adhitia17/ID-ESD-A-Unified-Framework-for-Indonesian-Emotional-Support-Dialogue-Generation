import pandas as pd
import random
from sklearn.model_selection import train_test_split

# --- Load dataset utama ---
df = pd.read_csv("ESC1.csv")
df = df.sort_values(["conv_id", "turn_id"]).reset_index(drop=True)

# Pastikan hanya baris dengan speaker = supporter dan ada strategi
supporter_df = df[df["speaker"].str.lower() == "supporter"].dropna(subset=["strategy"])
supporter_df["strategy"] = supporter_df["strategy"].str.strip()

print(f"Total supporter turns dengan strategy: {len(supporter_df)}")

# --- Ambil 1 turn per conv_id (yang terakhir dari supporter) ---
last_supporter = supporter_df.groupby("conv_id").tail(1).reset_index(drop=True)

# --- Stratified sampling berdasarkan strategy ---
# Hitung distribusi
strategy_counts = last_supporter["strategy"].value_counts()
print("\nDistribusi awal strategy:")
print(strategy_counts)

# Tentukan total target percakapan evaluasi
TARGET_CONV = 150

# Hitung proporsi sampling berdasarkan distribusi strategi
total = strategy_counts.sum()
subset_list = []

for strategy, count in strategy_counts.items():
    prop = count / total
    n_sample = max(3, int(TARGET_CONV * prop))  # minimal 3 per strategi
    strat_df = last_supporter[last_supporter["strategy"] == strategy]
    
    if len(strat_df) > n_sample:
        strat_sample = strat_df.sample(n=n_sample, random_state=42)
    else:
        strat_sample = strat_df
    subset_list.append(strat_sample)

subset_df = pd.concat(subset_list).sort_values("conv_id").reset_index(drop=True)

# --- Ambil seluruh percakapan (bukan hanya satu turn) berdasarkan conv_id hasil sampling ---
eval_convs = df[df["conv_id"].isin(subset_df["conv_id"].unique())]
eval_convs = eval_convs.sort_values(["conv_id", "turn_id"]).reset_index(drop=True)

# --- Simpan hasil ---
output_path = "eval_subset.csv"
eval_convs.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"\n✅ eval_subset.csv berhasil dibuat: {len(eval_convs['conv_id'].unique())} conversations")
print(f"Total baris: {len(eval_convs)}")
print(f"Disimpan ke: {output_path}")
