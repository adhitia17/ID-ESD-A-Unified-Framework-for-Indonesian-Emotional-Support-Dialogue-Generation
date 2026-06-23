import json
from collections import Counter
import numpy as np
from tqdm import tqdm

# ==== Load dataset ====
input_file = ".translate/ESConv_translated.json"
output_file = ".translate/dataset_statistics.json"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# ==== Inisialisasi counter ====
experience_counter = Counter()
emotion_counter = Counter()
problem_counter = Counter()
strategy_counter = Counter()

# ==== Variabel untuk menghitung panjang dan turns ====
num_dialogues = len(data)
total_turns = 0
utterance_lengths = []
dialogue_lengths = []
utterance_lengths_seeker = []
utterance_lengths_supporter = []
dialogue_lengths_seeker = []
dialogue_lengths_supporter = []

# ==== Loop utama ====
for item in tqdm(data, desc="Analisis dataset"):
    # Count kategori
    experience_counter[item.get("experience_type", "").strip()] += 1
    emotion_counter[item.get("emotion_type", "").strip()] += 1
    problem_counter[item.get("problem_type", "").strip()] += 1

    # Dialog
    dialog = item.get("dialog", [])
    total_turns += len(dialog)

    dialogue_texts = [turn.get("content", "").strip() for turn in dialog if "content" in turn]
    dialogue_length = sum(len(t.split()) for t in dialogue_texts)
    dialogue_lengths.append(dialogue_length)

    seeker_texts = [turn["content"].strip() for turn in dialog if turn.get("speaker") == "seeker" and "content" in turn]
    supporter_texts = [turn["content"].strip() for turn in dialog if turn.get("speaker") == "supporter" and "content" in turn]

    dialogue_lengths_seeker.append(sum(len(t.split()) for t in seeker_texts))
    dialogue_lengths_supporter.append(sum(len(t.split()) for t in supporter_texts))

    # Hitung panjang utterance
    for turn in dialog:
        text = turn.get("content", "").strip()
        if not text:
            continue
        utter_len = len(text.split())
        utterance_lengths.append(utter_len)

        if turn.get("speaker") == "seeker":
            utterance_lengths_seeker.append(utter_len)
        elif turn.get("speaker") == "supporter":
            utterance_lengths_supporter.append(utter_len)

        # Hitung strategi (annotation)
        strategy = turn.get("annotation", {}).get("strategy")
        if strategy:
            strategy_counter[strategy.strip()] += 1

# ==== Statistik agregat ====
avg_turn_per_dialogue = total_turns / num_dialogues
avg_length_per_dialogue = float(np.mean(dialogue_lengths))
avg_length_per_utterance = float(np.mean(utterance_lengths))

avg_length_per_dialogue_seeker = float(np.mean(dialogue_lengths_seeker))
avg_length_per_utterance_seeker = float(np.mean(utterance_lengths_seeker))
avg_length_per_dialogue_supporter = float(np.mean(dialogue_lengths_supporter))
avg_length_per_utterance_supporter = float(np.mean(utterance_lengths_supporter))

# ==== Buat struktur hasil ====
stats = {
    "num_dialogues": num_dialogues,
    "total_turns": total_turns,
    "num_utterances": len(utterance_lengths),
    "averages": {
        "avg_turn_per_dialogue": avg_turn_per_dialogue,
        "avg_length_per_dialogue": avg_length_per_dialogue,
        "avg_length_per_utterance": avg_length_per_utterance,
        "avg_length_per_dialogue_seeker": avg_length_per_dialogue_seeker,
        "avg_length_per_utterance_seeker": avg_length_per_utterance_seeker,
        "avg_length_per_dialogue_supporter": avg_length_per_dialogue_supporter,
        "avg_length_per_utterance_supporter": avg_length_per_utterance_supporter
    },
    "composition": {
        "experience_type": dict(experience_counter),
        "emotion_type": dict(emotion_counter),
        "problem_type": dict(problem_counter),
        "strategy": dict(strategy_counter)
    }
}

# ==== Simpan ke JSON ====
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

print(f"✅ Statistik dataset selesai dihitung dan disimpan ke: {output_file}")
