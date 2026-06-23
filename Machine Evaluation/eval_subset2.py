# eval_subset2.py
import pandas as pd

# Load dataset
csv_path = "/mnt/data-hps/adhitiae/vscode/ESC/augmenting/final_data/dataset_esc_augmenting_final_mapped.csv"
df = pd.read_csv(csv_path)
df = df.sort_values(["conv_id", "turn_id"]).reset_index(drop=True)

print(f"Dataset loaded: {len(df)} rows, {df['conv_id'].nunique()} convs")

# Filter supporter turns dengan strategy (untuk stratified sampling)
supporter_df = df[(df["speaker"].str.lower() == "supporter") & df["strategy"].notna() & (df["strategy"].str.strip() != "")].copy()
supporter_df["strategy"] = supporter_df["strategy"].str.strip()

print(f"Supporter turns dengan strategy: {len(supporter_df)}")

# Ambil 1 turn terakhir per conv_id
last_supporter = supporter_df.groupby("conv_id").tail(1).reset_index(drop=True)

# Stratified sampling (TARGET_CONV = 150)
TARGET_CONV = 150
strategy_counts = last_supporter["strategy"].value_counts()
print("\nDistribusi strategy:")
print(strategy_counts)

subset_list = []
total = strategy_counts.sum()
for strategy, count in strategy_counts.items():
    prop = count / total
    n_sample = max(3, int(TARGET_CONV * prop))  # Min 3 per strategy
    strat_df = last_supporter[last_supporter["strategy"] == strategy]
    if len(strat_df) >= n_sample:
        sample = strat_df.sample(n=n_sample, random_state=42)
    else:
        sample = strat_df
    subset_list.append(sample)

subset_df = pd.concat(subset_list).drop_duplicates().reset_index(drop=True)
selected_convs = subset_df["conv_id"].unique()

# Ambil seluruh percakapan dari selected convs
eval_df = df[df["conv_id"].isin(selected_convs)].sort_values(["conv_id", "turn_id"]).reset_index(drop=True)

# Simpan
output_path = "eval_subset2.csv"
eval_df.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"\n✅ Subset evaluasi: {len(selected_convs)} convs, {len(eval_df)} rows")
print(f"Disimpan ke: {output_path}")