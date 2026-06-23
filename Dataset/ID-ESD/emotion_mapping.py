# emotion_mapping.py
# Peta emosi yang luas — memetakan variasi frasa/istilah emosi ke label normalisasi.
# Tambahkan atau sesuaikan entri ini sesuai kebutuhan dataset/analisis Anda.

EMOTION_MAPPING = {
    # MARAH / FRUSTRASI
    "frustrasi": "marah",
    "frustasi": "marah",
    "kesal": "marah",
    "jengkel": "marah",
    "geram": "marah",
    "sebal": "marah",
    "marah": "marah",
    "malesin": "marah",

    # SEDIH / KECEWA
    "kecewa": "sedih",
    "sedih": "sedih",
    "patah hati": "sedih",
    "hancur": "sedih",
    "putus asa": "depresi",
    "terpuruk": "depresi",
    "menangis": "sedih",

    # DEPRESI / LELAH EMOSIONAL
    "depresi": "depresi",
    "lelah": "depresi",
    "capek": "depresi",
    "kehilangan semangat": "depresi",
    "tak berdaya": "depresi",

    # CEMAS / GELISAH
    "cemas": "cemas",
    "khawatir": "cemas",
    "gelisah": "cemas",
    "gugup": "cemas",
    "was-was": "cemas",
    "waswas": "cemas",

    # TAKUT / PANIK
    "takut": "takut",
    "panik": "takut",
    "ketakutan": "takut",
    "ngeri": "takut",

    # JIJIK / MUAK
    "jijik": "jijik",
    "muak": "jijik",
    "muak banget": "jijik",
    "ilfeel": "jijik",

    # MALU / MENYESAL
    "malu": "malu",
    "sungkan": "malu",
    "menyesal": "malu",
    "malu banget": "malu",

    # BAHAGIA / LEG A / TERHARU / RINDU / KESEPIAN / BINGUNG
    "bahagia": "bahagia",
    "senang": "bahagia",
    "lega": "lega",
    "lega banget": "lega",
    "terharu": "terharu",
    "haru": "terharu",
    "rindu": "rindu",
    "kangen": "rindu",
    "kesepian": "kesepian",
    "sepi": "kesepian",
    "bingung": "bingung",
    "galau": "bingung",

    # EMOSI CAMPURAN / LAINNYA (map ke label ringkas atau biarkan apa adanya)
    "campur aduk": "bingung",
    "mixed": "bingung",
    "overwhelmed": "depresi",
    "tertekan": "depresi",
    "terbebani": "depresi",
    "emosi campur": "bingung",
    "sensitif": "sedih",
    "sensasi tidak enak": "jijik",

    # varian bahasa/slang
    "btw sedih": "sedih",
    "bete": "marah",
    "bad mood": "marah",
    "down": "sedih",
    "not okay": "depresi",

    # fallback phrase examples
    "bingung banget": "bingung",
    "gak semangat": "depresi",
    "gak kuat": "depresi"
}

# Anda bisa memperbesar dict ini dengan menambahkan entri baru atau regular expression
# Jika ingin pemetaan berbasis regex/keyword yang lebih kompleks, nanti kita ubah fungsi normalize_emotion.
