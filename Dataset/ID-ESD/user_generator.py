import random
from llm_client import query_llm
from difflib import SequenceMatcher
from emotion_mapping import EMOTION_MAPPING

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
# Template kalimat pembuka (natural, tidak formal)
# ======================
templates = [
    "Akhir-akhir ini aku merasa {emotion} karena {problem}.",
    "Aku lagi ngerasa {emotion} banget gara-gara {problem}.",
    "Aku gak tahu kenapa, tapi {problem} bikin aku terus merasa {emotion}.",
    "Rasanya berat banget menghadapi {problem}, aku sering banget ngerasa {emotion}.",
    "Aku cuma pengen cerita, aku lagi ngerasa {emotion} karena {problem}.",
    "Aku capek banget sama {problem}, rasanya penuh dengan {emotion}.",
    "Belakangan ini aku sering ngerasa {emotion}, mungkin karena {problem}.",
    "Aku gak tahu harus gimana, {problem} bikin aku ngerasa {emotion} terus."
]

# ======================
# Generate kalimat seed awal
# ======================
def generate_base_query():
    problem_choice = random.choice(problems)
    emotion_choice = random.choice(emotions)
    template = random.choice(templates)
    # jika emotion_choice berupa frase panjang, ambil versi singkat untuk naturalness
    return template.format(problem=problem_choice, emotion=emotion_choice), problem_choice, emotion_choice

# ======================
# Parafrase seed agar variasi alami
# ======================
def paraphrase_user_query(base_query, n_variations=3):
    system_msg = (
        "Anda adalah konselor empatik yang memparafrasekan ucapan pengguna dalam Bahasa Indonesia yang natural, hangat, "
        "dan terasa seperti percakapan sehari-hari antara teman. Hindari kalimat kaku atau formal."
    )
    user_msg = f"Tolong buat {n_variations} versi berbeda dari kalimat ini tanpa mengubah maknanya: '{base_query}'"

    try:
        response = query_llm(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.9,
            max_tokens=256
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
# Fungsi utama generate query
# ======================
def generate_user_query(existing_queries=None):
    """
    Menghasilkan satu seed user (kalimat pembuka), bersama dengan problem dan emotion yang digunakan.
    existing_queries: set atau list yang berisi seed sebelumnya untuk menghindari duplikasi.
    Mengembalikan tuple: (kalimat, problem_choice, emotion_choice)
    """
    if existing_queries is None:
        existing_queries = set()
    # Pastikan tipe set untuk operasi membership cepat
    if not isinstance(existing_queries, set):
        existing_queries = set(existing_queries)

    base_query, problem_choice, emotion_choice = generate_base_query()
    variations = paraphrase_user_query(base_query, n_variations=3)

    for candidate in variations:
        if not is_duplicate(candidate, existing_queries):
            return candidate, problem_choice, emotion_choice

    # jika semua variasi duplikat, kembalikan base_query (meskipun duplikat) untuk menghindari loop tak berujung
    return base_query, problem_choice, emotion_choice


if __name__ == "__main__":
    # contoh penggunaan cepat
    seen = set()
    for i in range(10):
        q, p, e = generate_user_query(seen)
        print(i+1, q, "| problem=", p, "| emotion=", e)
        seen.add(q)
