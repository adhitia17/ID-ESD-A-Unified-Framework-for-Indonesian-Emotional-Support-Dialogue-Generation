import os
import re
import json
import csv
import random
import time

from llm_client import query_llm
from user_generator_short import generate_user_query, classify_conversation, greetings
from emotion_mapping import EMOTION_MAPPING

# ==============================================================
# KONFIGURASI
# ==============================================================
TARGET = 20000  # jumlah percakapan valid yang ingin dihasilkan
BATCH_SIZE = 10
OUTPUT_DIR = "augmenting/dataset_esc_short"
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
    """
    Normalisasi emosi menggunakan EMOTION_MAPPING (substring longest match).
    Jika tidak cocok, kembalikan hasil LLM yang sudah dipangkas.
    """
    if not raw_emotion:
        return ""
    e = raw_emotion.strip().lower()
    e = re.sub(r'^[\'"]|[\'"]$', '', e).strip()

    best_match = None
    best_key_len = 0
    for k, v in EMOTION_MAPPING.items():
        if k in e:
            if len(k) > best_key_len:
                best_key_len = len(k)
                best_match = v
    if best_match:
        return best_match
    # jika LLM memberikan label yang sudah sama dengan mapping values, pertahankan
    if e in set(EMOTION_MAPPING.values()):
        return e
    # fallback: return raw trimmed token (single word if possible)
    e_simple = e.split()[0] if " " in e else e
    return e_simple

def parse_to_conversation(response_text):
    """Konversi teks menjadi list of {role, content}."""
    response_text = clean_text(response_text)
    # coba parse JSON langsung
    try:
        parsed = json.loads(response_text)
        if isinstance(parsed, list):
            conv = []
            for d in parsed:
                role = (d.get("role") or "").strip().lower()
                content = (d.get("content") or "").strip()
                conv.append({"role": role, "content": content})
            return conv
    except Exception:
        pass

    # fallback parsing manual per baris "seeker: ..." atau "supporter - ..."
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
    # berikan daftar contoh emosi untuk mendorong variasi
    examples = list({v for v in EMOTION_MAPPING.values()})
    prompt = (
        "Kamu adalah pengklasifikasi emosi sederhana dalam Bahasa Indonesia.\n"
        "Diberikan satu kalimat dari pengguna (seeker), sebutkan EMOSI UTAMA "
        "yang paling menggambarkan perasaan orang tersebut. "
        "Pilih satu kata atau frasa pendek dari contoh berikut (atau sinonim yang relevan): "
        f"{examples}\n\n"
        f"Kalimat: \"{text}\"\n\n"
        "Jawab hanya dengan satu kata atau frasa pendek (tanpa penjelasan). Jika ini hanya sapaan/netral, jawab 'netral'."
    )
    try:
        result = query_llm(
            [
                {"role": "system", "content": "Anda adalah pengklasifikasi emosi yang akurat dan ringkas."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=12
        )
        return normalize_emotion(result)
    except Exception:
        return ""

def classify_strategy(supporter_text, seeker_context=""):
    """Gunakan LLM untuk menentukan strategi supporter."""
    if not supporter_text:
        return ""
    prompt = (
        "Tentukan strategi dukungan emosional yang digunakan oleh respon 'supporter'. "
        "Pilih SATU dari daftar ini (jawab hanya dengan frasa yang ada):\n"
        "bertanya, mengulang atau memparafrasekan, merefleksikan perasaan, berbagi pengalaman pribadi, "
        "memberikan penegasan dan penyemangat, memberikan saran, memberikan informasi, lainnya.\n\n"
        f"Konteks seeker (opsional): \"{seeker_context}\"\n"
        f"Respon supporter: \"{supporter_text}\"\n\n"
        "Jawab hanya dengan satu frasa persis dari daftar di atas."
    )
    try:
        result = query_llm(
            [{"role": "system", "content": "Anda adalah pengklasifikasi strategi dukungan emosional."},
             {"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=24
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
    s = re.sub(r"[\x00-\x1f\x7f]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def is_valid_content(s):
    """Valid jika panjang minimal 2 dan ada setidaknya satu alnum (huruf/angka)."""
    if not s:
        return False
    if len(s) < 2:
        return False
    if not re.search(r"[0-9A-Za-z\u00C0-\u024F\u1E00-\u1EFF\u0100-\u017F\u00E0-\u00FF\u0E00-\u0E7F]", s):
        return False
    return True

def is_greeting_text(text):
    """Deteksi apakah text adalah sapaan singkat (mengabaikan tanda baca dan casing)."""
    if not text:
        return False
    t = text.lower().strip().strip('"\'' )
    t = re.sub(r"[^\w\s]", "", t)
    return t in [g.lower() for g in greetings]

# ==============================================================
# LOOP UTAMA (DENGAN ALUR SAPAAN AWAL, KLASIFIKASI SETELAH PERCAPAKAN)
# ==============================================================
saved_conv_id = 0  # menghitung percakapan valid
start_time = time.time()

for attempt_id in range(1, TARGET * 3):  # memberi ruang untuk percobaan gagal
    if saved_conv_id >= TARGET:
        break

    user_seed, _, _ = generate_user_query(generated_users)
    generated_users.add(user_seed)

    system_prompt = (
        "Kamu adalah Emotional Supporter yang empatik dan berbicara santai seperti teman dekat. "
        "Buat percakapan antara 'seeker' (curhat) dan 'supporter' (menanggapi) dalam Bahasa Indonesia yang natural dan tidak formal. "
        "Buat percakapan 'seeker' cukup singkat saja (maksimal 2 kalimat) namun 'supporter' tetap menjawab dengan baik. "
        "Jangan sertakan tag [Strategy:]. "
        "Buat antara 10 sampai 20 giliran percakapan bergantian dimulai dari 'seeker' yang diawali dengan kata sapaan seperti hai, halo, selamat pagi, selamat siang, selamat sore, selamat malam, atau lainnya:\n"
        f"'{user_seed}'.\n\n"
        "Tampilkan hasil dalam format JSON list berisi objek {\"role\": \"seeker\"/\"supporter\", \"content\": \"...\"}. "
        "Jangan tambahkan penjelasan lain apa pun."
    )

    try:
        response = query_llm(
            [{"role": "system", "content": system_prompt}],
            temperature=0.8,
            max_tokens=3200
        )
        conversation = parse_to_conversation(response)

        # basic validation: non-empty dan panjang percakapan 10-20
        if not conversation or len(conversation) < 10 or len(conversation) > 20:
            # <<< LOGGING DITAMBAHKAN
            print(f"⚠️ Percakapan (percobaan {attempt_id}) dilewati (turn={len(conversation) if conversation else 0}).")
            continue

        # bersihkan dan validasi setiap turn
        cleaned_conv = []
        invalid_flag = False
        for i, turn in enumerate(conversation):
            role = (turn.get("role") or "").lower()
            raw_content = turn.get("content", "")
            content = clean_turn_content(raw_content)

            if not is_valid_content(content):
                invalid_flag = True
                break

            cleaned_conv.append({"role": role, "content": content})

        if invalid_flag:
            # <<< LOGGING DITAMBAHKAN
            print(f"⚠️ Percakapan (percobaan {attempt_id}) dilewati (invalid content).")
            continue

        # validasi urutan role: harus mulai dengan seeker dan bergantian
        expected = "seeker"
        ok_roles = True
        for i, t in enumerate(cleaned_conv):
            if t["role"] not in ("seeker", "supporter"):
                ok_roles = False
                break
            if t["role"] != expected:
                ok_roles = False
                break
            expected = "supporter" if expected == "seeker" else "seeker"

        if not ok_roles:
            # <<< LOGGING DITAMBAHKAN
            print(f"⚠️ Percakapan (percobaan {attempt_id}) dilewati (bad roles).")
            continue

        # Setelah seluruh percakapan bersih, klasifikasikan problem (level percakapan)
        full_text = "\n".join([t["content"] for t in cleaned_conv])
        problem_pred, conversation_emotion = classify_conversation(full_text)

        # semua valid -> simpan percakapan
        saved_conv_id += 1
        summary_rows.append({"conv_id": saved_conv_id, "problem": problem_pred})
        
        # <<< LOGGING DITAMBAHKAN
        print(f"✅ Percakapan {saved_conv_id} ({problem_pred}) — valid dan disimpan.")

        for turn_id, turn in enumerate(cleaned_conv):
            role = turn["role"]
            content = turn["content"]

            if role == "seeker":
                # jika turn pertama hanya sapaan, beri label emosi 'netral'
                if turn_id == 0 and is_greeting_text(content):
                    emotion = "netral"
                else:
                    # klasifikasikan emosi per-turn untuk variasi; fallback ke conversation_emotion atau 'netral'
                    e = classify_emotion(content)
                    emotion = e if e else (conversation_emotion if conversation_emotion else "netral")
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
                "problem": problem_pred,
                "emotion": emotion,
                "strategy": strategy
            })

        # simpan batch setiap BATCH_SIZE
        if saved_conv_id % BATCH_SIZE == 0:
            filename = os.path.join(OUTPUT_DIR, f"esc_conversation_part{file_counter}.csv")
            write_batch_csv(dataset_rows, filename)
            # <<< LOGGING DITAMBAHKAN
            print(f"📂 Disimpan: {filename} ({len(dataset_rows)} baris)")
            
            # refresh summary
            summary_path = os.path.join(OUTPUT_DIR, "summary_report.csv")
            write_summary_report(summary_rows, summary_path)
            # <<< LOGGING DITAMBAHKAN
            print(f"📄 Laporan ringkasan disimpan: {summary_path} (conv_id, problem)")
            
            dataset_rows = []
            file_counter += 1

        # <<< LOGGING DITAMBAHKAN (SESUAI FILE 1)
        if saved_conv_id % 10 == 0:
            print(f"🔹 Total percakapan valid sejauh ini: {saved_conv_id}")

        # safety / rate control sedikit
        if attempt_id % 20 == 0:
            time.sleep(0.5)

    except Exception as e:
        # <<< LOGGING DIPERJELAS
        print(f"❌ Error pada percobaan {attempt_id}: {e}")
        continue

# Simpan sisa terakhir
if dataset_rows:
    filename = os.path.join(OUTPUT_DIR, f"esc_conversation_part{file_counter}.csv")
    write_batch_csv(dataset_rows, filename)
    # <<< LOGGING DITAMBAHKAN
    print(f"📂 Disimpan: {filename} ({len(dataset_rows)} baris)")

# Simpan summary akhir
summary_path = os.path.join(OUTPUT_DIR, "summary_report.csv")
write_summary_report(summary_rows, summary_path)
# <<< LOGGING DITAMBAHKAN
print(f"📄 Laporan ringkasan disimpan: {summary_path} (conv_id, problem)")

print(f"\n✅ Total percakapan valid: {saved_conv_id}")