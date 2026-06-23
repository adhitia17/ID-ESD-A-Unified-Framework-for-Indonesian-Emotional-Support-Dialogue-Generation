# generate_large_dataset_open_emotion_v2.py
from llm_client import query_llm
from user_generator import generate_user_query
from emotion_mapping import EMOTION_MAPPING
import json
import os
import csv
import re

# ==============================================================
# KONFIGURASI
# ==============================================================
TARGET = 20000  # jumlah percakapan valid yang ingin dihasilkan
BATCH_SIZE = 10
OUTPUT_DIR = "augmenting/dataset_esc"
os.makedirs(OUTPUT_DIR, exist_ok=True)

dataset_rows = []
file_counter = 1
generated_users = set()

# Untuk laporan summary conv_id -> problem
summary_rows = []

# ==============================================================
# STRATEGI
# ==============================================================
STRATEGIES = [
    "bertanya",
    "mengulang atau memparafrasekan",
    "merefleksikan perasaan",
    "berbagi pengalaman pribadi",
    "memberikan penegasan dan penyemangat",
    "memberikan saran",
    "memberikan informasi",
    "lainnya"
]

# ==============================================================
# FUNGSI PEMBANTU
# ==============================================================
def clean_text(text):
    """Membersihkan hasil LLM agar mudah di-parse."""
    if not text:
        return ""
    text = text.replace("```json", "").replace("```", "")
    text = re.sub(r"(?i)berikut( ini)?( versi| percakapan)?.*?:", "", text)
    return text.strip()

def normalize_emotion(raw_emotion):
    """Normalisasi emosi menggunakan EMOTION_MAPPING (substring longest match)."""
    if not raw_emotion:
        return ""
    e = raw_emotion.strip().lower()
    e = re.sub(r'^[\'"]|[\'"]$', '', e).strip()
    best_match = None
    best_key_len = 0
    for k, v in EMOTION_MAPPING.items():
        if k in e:
            if len(k) > best_key_len:
                best_match = v
                best_key_len = len(k)
    if best_match:
        return best_match
    return e

def parse_to_conversation(response_text):
    """Konversi teks menjadi list of {role, content}."""
    response_text = clean_text(response_text)
    try:
        parsed = json.loads(response_text)
        if isinstance(parsed, list):
            return [{"role": d.get("role", "").lower(), "content": d.get("content", "").strip()} for d in parsed]
    except Exception:
        pass

    # fallback parsing manual
    lines = [l.strip() for l in response_text.split("\n") if l.strip()]
    conv = []
    for line in lines:
        match = re.match(r"(?i)^(seeker|supporter)\s*[:\-]\s*(.*)$", line)
        if match:
            role = match.group(1).lower()
            content = match.group(2).strip()
            conv.append({"role": role, "content": content})
    return conv

def classify_emotion(text):
    """Gunakan LLM untuk menentukan emosi natural (open-label)."""
    if not text:
        return ""
    prompt = (
        "Kamu adalah pengklasifikasi emosi dalam Bahasa Indonesia. "
        "Diberikan satu kalimat dari pengguna (seeker), sebutkan **emosi utama** "
        "yang paling menggambarkan perasaan orang tersebut. "
        "Jawab hanya dengan satu kata atau frasa pendek, misalnya: sedih, frustrasi, kecewa, cemas, marah, lelah, bingung, takut, malu, lega, dll.\n\n"
        f"Kalimat: \"{text}\"\n\n"
        "Jawab hanya dengan satu kata atau frasa pendek (tanpa penjelasan)."
    )
    try:
        result = query_llm(
            [
                {"role": "system", "content": "Anda adalah pengklasifikasi emosi yang akurat dan ringkas."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=16
        )
        return normalize_emotion(result)
    except Exception:
        return ""

def classify_strategy(supporter_text, seeker_context=""):
    """Gunakan LLM untuk menentukan strategi supporter."""
    if not supporter_text:
        return ""
    prompt = (
        "Tentukan strategi dukungan emosional yang digunakan oleh kalimat berikut dari 'supporter'. "
        "Pilih **satu saja** dari daftar berikut:\n"
        "bertanya, mengulang atau memparafrasekan, merefleksikan perasaan, berbagi pengalaman pribadi, "
        "memberikan penegasan dan penyemangat, memberikan saran, memberikan informasi, lainnya.\n\n"
        f"Konteks seeker (opsional): \"{seeker_context}\"\n"
        f"Respon supporter: \"{supporter_text}\"\n\n"
        "Jawab hanya dengan satu frasa dari daftar di atas."
    )
    try:
        result = query_llm(
            [{"role": "system", "content": "Anda adalah pengklasifikasi strategi dukungan emosional."},
             {"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=32
        )
        res = result.strip().lower()
        for s in STRATEGIES:
            if s in res:
                return s
    except Exception:
        return ""
    return ""

def write_batch_csv(rows, filename):
    """Tulis batch rows ke CSV dengan urutan kolom yang ditentukan."""
    with open(filename, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "conv_id",
                "turn_id",
                "speaker",
                "content",
                "problem",
                "emotion",
                "strategy"
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

def write_summary_report(summary_rows, out_path):
    """Tulis laporan summary conv_id,problem (unique per conv_id)."""
    seen = set()
    unique = []
    for r in summary_rows:
        if r["conv_id"] not in seen:
            unique.append(r)
            seen.add(r["conv_id"])
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["conv_id", "problem"])
        writer.writeheader()
        writer.writerows(unique)

# ==============================================================
# VALIDATION HELPERS
# ==============================================================
def clean_turn_content(raw):
    """Bersihkan content turn: strip, collapse whitespace, hapus control chars."""
    if raw is None:
        return ""
    s = raw.strip()
    # hilangkan control characters
    s = re.sub(r"[\x00-\x1f\x7f]", " ", s)
    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s

def is_valid_content(s):
    """Valid jika panjang minimal 2 dan ada setidaknya satu alnum (huruf/angka)."""
    if not s:
        return False
    if len(s) < 2:
        return False
    if not re.search(r"[0-9A-Za-z\u00C0-\u024F\u1E00-\u1EFF\u0100-\u017F\u00E0-\u00FF\u0E00-\u0E7F\u0E00-\u0E7F\u0E00-\u0E7F\u0E00-\u0E7F\u0E00-\u0E7F\u0E00-\u0E7F\u0E00-\u0E7F\u0E00-\u0E7F\u0E00-\u0E7F\u0E00-\u0E7F\u0E00-\u0E7F\u0E00-\u0E7F\u0E00-\u0E7F\u0E00-\u0E7F\u0E00-\u0E7F]", s):
        # cek presence of letters (termasuk unicode bahasa Indonesia)
        return False
    return True

# ==============================================================
# LOOP UTAMA (DENGAN FIX PENOMORAN BERURUT)
# ==============================================================
saved_conv_id = 0  # menghitung percakapan valid

for attempt_id in range(1, TARGET * 3):  # memberi ruang untuk percobaan gagal
    if saved_conv_id >= TARGET:
        break

    user_seed, problem, _ = generate_user_query(generated_users)
    generated_users.add(user_seed)

    system_prompt = (
        "Kamu adalah Emotional Supporter yang empatik dan berbicara santai seperti teman dekat. "
        "Buat percakapan antara 'seeker' (curhat) dan 'supporter' (menanggapi) dalam Bahasa Indonesia yang natural dan tidak formal. "
        "Jangan sertakan tag [Strategy:]. "
        "Buat antara 20 sampai 30 giliran percakapan bergantian dimulai dari seeker:\n"
        f"'{user_seed}'.\n\n"
        "Tampilkan hasil dalam format JSON list berisi objek {\"role\": \"seeker\"/\"supporter\", \"content\": \"...\"}. "
        "Jangan tambahkan penjelasan lain apa pun."
    )

    try:
        response = query_llm(
            [{"role": "system", "content": system_prompt}],
            temperature=0.85,
            max_tokens=3200
        )
        conversation = parse_to_conversation(response)

        # basic validation: non-empty and length 20-30
        if not conversation or len(conversation) < 20 or len(conversation) > 30:
            print(f"⚠️ Percakapan (percobaan {attempt_id}) dilewati (turn={len(conversation) if conversation else 0}).")
            continue

        # bersihkan dan validasi setiap turn
        cleaned_conv = []
        invalid_flag = False
        for i, turn in enumerate(conversation):
            role = (turn.get("role") or "").lower()
            raw_content = turn.get("content", "")
            content = clean_turn_content(raw_content)

            # validasi content tidak kosong dan cukup panjang
            if not is_valid_content(content):
                print(f"⚠️ Percakapan (percobaan {attempt_id}) dilewati: ditemukan turn kosong/invalid di index {i} (role={role}).")
                invalid_flag = True
                break

            cleaned_conv.append({"role": role, "content": content})

        if invalid_flag:
            continue

        # validasi urutan role: harus mulai dengan seeker dan bergantian
        expected = "seeker"
        ok_roles = True
        for i, t in enumerate(cleaned_conv):
            if t["role"] not in ("seeker", "supporter"):
                ok_roles = False
                print(f"⚠️ Percakapan (percobaan {attempt_id}) dilewati: role tidak valid di index {i}: {t['role']}")
                break
            if t["role"] != expected:
                ok_roles = False
                print(f"⚠️ Percakapan (percobaan {attempt_id}) dilewati: urutan role tidak sesuai di index {i} (diharapkan {expected}, dapat {t['role']}).")
                break
            expected = "supporter" if expected == "seeker" else "seeker"

        if not ok_roles:
            continue

        # semua valid -> simpan percakapan
        saved_conv_id += 1
        summary_rows.append({"conv_id": saved_conv_id, "problem": problem})
        print(f"✅ Percakapan {saved_conv_id} (problem: {problem}) — valid dan disimpan.")

        for turn_id, turn in enumerate(cleaned_conv):
            role = turn["role"]
            content = turn["content"]

            # pastikan seeker tidak punya strategy; supporter tidak punya emotion
            if role == "seeker":
                emotion = classify_emotion(content)
                strategy = ""  # wajib kosong untuk seeker
            else:
                seeker_context = ""
                if turn_id > 0 and cleaned_conv[turn_id - 1]["role"] == "seeker":
                    seeker_context = cleaned_conv[turn_id - 1]["content"]
                strategy = classify_strategy(content, seeker_context)
                emotion = ""  # wajib kosong untuk supporter

            dataset_rows.append({
                "conv_id": saved_conv_id,
                "turn_id": turn_id,
                "speaker": role,
                "content": content,
                "problem": problem,
                "emotion": emotion,
                "strategy": strategy
            })

        # simpan batch setiap BATCH_SIZE
        if saved_conv_id % BATCH_SIZE == 0:
            filename = os.path.join(OUTPUT_DIR, f"esc_conversation_part{file_counter}.csv")
            write_batch_csv(dataset_rows, filename)
            print(f"📂 Disimpan: {filename} ({len(dataset_rows)} baris)")
            # juga simpan/refresh summary report sementara
            summary_path = os.path.join(OUTPUT_DIR, "summary_report.csv")
            write_summary_report(summary_rows, summary_path)
            print(f"📄 Laporan ringkasan disimpan: {summary_path} (conv_id, problem)")
            dataset_rows = []
            file_counter += 1

        if saved_conv_id % 10 == 0:
            print(f"🔹 Total percakapan valid sejauh ini: {saved_conv_id}")

    except Exception as e:
        print(f"❌ Error pada percobaan {attempt_id}: {e}")
        continue

# Simpan sisa terakhir
if dataset_rows:
    filename = os.path.join(OUTPUT_DIR, f"esc_conversation_part{file_counter}.csv")
    write_batch_csv(dataset_rows, filename)
    print(f"📂 Disimpan: {filename} ({len(dataset_rows)} baris)")

# Simpan summary akhir
summary_path = os.path.join(OUTPUT_DIR, "summary_report.csv")
write_summary_report(summary_rows, summary_path)
print(f"📄 Laporan ringkasan akhir disimpan: {summary_path} (conv_id, problem)")

print(f"\n✅ Total percakapan valid: {saved_conv_id}")
