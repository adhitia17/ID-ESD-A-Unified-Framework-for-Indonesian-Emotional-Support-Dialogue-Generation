# ...existing code...
import random
from llm_client import query_llm
from difflib import SequenceMatcher
from emotion_mapping import EMOTION_MAPPING
import json
# ...existing code...

# ======================
# Kategori masalah (problem) dalam Bahasa Indonesia
# ======================
problems = [
    "depresi yang sedang berlangsung",
    "krisis pekerjaan",
    "putus dengan pasangan",
    "masalah dengan teman",
    "tekanan akademik"
]

# Bangun daftar emosi yang lebih variatif dari EMOTION_MAPPING
_emotion_keys = list(EMOTION_MAPPING.keys())
# tambahkan beberapa emosi umum/varian jika perlu
_extra_emotions = ["bingung", "lega", "rindu", "kesepian", "terharu", "frustrasi", "kecewa"]
emotions = list(dict.fromkeys(_emotion_keys + _extra_emotions))  # hapus duplikat sambil mempertahankan urutan

# ======================
# Ganti template: sekarang hanya sapaan sederhana
# ======================
greetings = [
    "hai",
    "halo",
    "hi",
    "selamat pagi",
    "selamat siang",
    "selamat sore",
    "selamat malam"
]

# ======================
# Generate kalimat seed awal: hanya sapaan
# ======================
def generate_greeting_query():
    return random.choice(greetings)

# ======================
# Parafrase sapaan agar variasi alami (opsional)
# ======================
def paraphrase_user_query(base_query, n_variations=3):
    system_msg = (
        "Anda adalah konselor empatik yang memparafrasekan ucapan pengguna dalam Bahasa Indonesia yang natural, hangat, "
        "dan terasa seperti percakapan sehari-hari antara teman. Hindari kalimat kaku atau formal. "
        "Untuk input yang berupa sapaan singkat, kembalikan beberapa varian sapaan yang natural."
    )
    user_msg = f"Tolong buat {n_variations} versi berbeda dari kalimat ini tanpa mengubah maknanya: '{base_query}'"

    try:
        response = query_llm(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.9,
            max_tokens=128
        )
        # ekstrak baris - hapus bullet jika ada
        variations = [line.strip("-• ").strip() for line in response.split("\n") if line.strip()]
        # pastikan base_query termasuk untuk fallback
        if base_query not in variations:
            variations.insert(0, base_query)
        return variations
    except Exception:
        return [base_query]

# ======================
# Cek duplikasi kalimat
# ======================
def is_duplicate(new_query, existing_queries, threshold=0.8):
    if new_query in existing_queries:
        return True
    for q in existing_queries:
        if SequenceMatcher(None, new_query, q).ratio() >= threshold:
            return True
    return False

# ======================
# Fungsi utama generate query -> hanya sapaan, problem/emotion dikembalikan None
# ======================
def generate_user_query(existing_queries=None):
    """
    Menghasilkan satu seed user (sapaan saja).
    existing_queries: set atau list yang berisi seed sebelumnya untuk menghindari duplikasi.
    Mengembalikan tuple: (kalimat, problem_choice, emotion_choice)
    problem_choice dan emotion_choice akan None (akan diklasifikasi setelah percakapan).
    """
    if existing_queries is None:
        existing_queries = set()
    if not isinstance(existing_queries, set):
        existing_queries = set(existing_queries)

    base_query = generate_greeting_query()
    variations = paraphrase_user_query(base_query, n_variations=3)

    for candidate in variations:
        if not is_duplicate(candidate, existing_queries):
            return candidate, None, None

    # jika semua variasi duplikat, kembalikan base_query (meskipun duplikat)
    return base_query, None, None

# ======================
# Klasifikasi problem dan emotion setelah percakapan selesai
# ======================
def classify_conversation(conversation_text):
    """
    Klasifikasikan problem (dari daftar 'problems') dan emotion (dari 'emotions') berdasarkan seluruh teks percakapan.
    Mengembalikan tuple: (problem_choice, emotion_choice)
    Menggunakan LLM, dengan fallback random jika gagal.
    """
    system_msg = (
        "Anda adalah asisten yang mengklasifikasikan percakapan pengguna dalam Bahasa Indonesia. "
        "Dari teks percakapan ini, pilih SAKIT satu masalah yang paling relevan dari daftar berikut dan satu emosi "
        "yang paling relevan dari daftar emosi. Kembalikan output dalam format JSON: {\"problem\": \"...\", \"emotion\": \"...\"}."
    )
    user_msg = (
        f"Daftar masalah: {problems}\nDaftar emosi: {emotions}\n\nTeks percakapan:\n{conversation_text}\n\n"
        "Berikan JSON yang valid dengan kunci 'problem' dan 'emotion'."
    )

    try:
        response = query_llm(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.0,
            max_tokens=150
        )
        # coba parse JSON dari respon
        parsed = json.loads(response.strip())
        problem_choice = parsed.get("problem")
        emotion_choice = parsed.get("emotion")
        # validasi terhadap daftar kita
        if problem_choice not in problems:
            problem_choice = random.choice(problems)
        if emotion_choice not in emotions:
            emotion_choice = random.choice(emotions)
        return problem_choice, emotion_choice
    except Exception:
        # fallback sederhana
        return random.choice(problems), random.choice(emotions)

# ...existing code...
if __name__ == "__main__":
    # contoh: buat beberapa sapaan sebagai percakapan, lalu klasifikasikan setelah selesai
    seen = set()
    convo = []
    for i in range(5):
        q, p, e = generate_user_query(seen)
        print(i+1, q)
        convo.append(q)
        seen.add(q)

    full_text = "\n".join(convo)
    problem_pred, emotion_pred = classify_conversation(full_text)
    print("=== Setelah percakapan selesai ===")
    print("Klasifikasi problem:", problem_pred)
    print("Klasifikasi emotion:", emotion_pred)
# ...existing code...