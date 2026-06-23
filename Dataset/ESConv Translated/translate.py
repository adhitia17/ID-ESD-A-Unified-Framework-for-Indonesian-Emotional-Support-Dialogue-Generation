from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import json
import os
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "8"
os.environ["NCCL_P2P_DISABLE"] = "1"

# ==== Fungsi baca JSON fleksibel & toleran ====
def load_json_auto(file_path):
    with open(file_path, "r", encoding="utf-8-sig") as f:  # utf-8-sig akan buang BOM
        text = f.read().strip()
        
        # Jika format JSON array → langsung load
        if text.startswith("["):
            return json.loads(text)
        
        # Jika setiap baris adalah objek JSON (format JSONL)
        if "\n" in text and all(line.strip().startswith("{") for line in text.splitlines() if line.strip()):
            lines = [json.loads(line) for line in text.splitlines() if line.strip()]
            return lines
        
        # Jika beberapa objek JSON terpisah tanpa koma/array
        if text.count("}\n{") > 0:
            text = "[" + text.replace("}\n{", "},\n{") + "]"
            return json.loads(text)
        
        # Jika masih gagal → coba baca manual per objek
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            print(f"⚠️ Gagal parse JSON langsung: {e}")
            # Coba ekstrak setiap blok { ... } manual
            objs = []
            buff = ""
            depth = 0
            for ch in text:
                if ch == "{":
                    depth += 1
                if depth > 0:
                    buff += ch
                if ch == "}":
                    depth -= 1
                    if depth == 0 and buff.strip():
                        try:
                            objs.append(json.loads(buff))
                        except Exception:
                            pass
                        buff = ""
            if objs:
                print(f"✅ Loaded {len(objs)} JSON objects secara manual.")
                return objs
            raise e

# ==== Load model NLLB ====
model_name = "facebook/nllb-200-1.3B"  # atau facebook/nllb-200-distilled-600M jika GPU kecil
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

src_lang = "eng_Latn"
tgt_lang = "ind_Latn"

# ==== Fungsi translate ====
def translate_text(text, tokenizer, model, src_lang, tgt_lang, max_length=256):
    if not text or not isinstance(text, str):
        return text
    
    # Tentukan ID token bahasa tujuan
    tgt_lang_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )
    with torch.no_grad():
        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tgt_lang_id,
            max_length=max_length
        )
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]


# ==== Load data ====
input_file = ".translate/ESConv.json"
output_file = ".translate/ESConv_translated.json"

data = load_json_auto(input_file)

# ==== Translate ====
translated_data = []
for item in tqdm(data, desc="Translating..."):
    translated_item = item.copy()
    
    for key in ["experience_type", "emotion_type", "problem_type", "situation"]:
        if key in item:
            translated_item[key] = translate_text(item[key], tokenizer, model, src_lang, tgt_lang)
    
    if "dialog" in item:
        for turn in item["dialog"]:
            if "content" in turn:
                turn["content"] = translate_text(turn["content"], tokenizer, model, src_lang, tgt_lang)
            if "annotation" in turn and "strategy" in turn["annotation"]:
                turn["annotation"]["strategy"] = translate_text(turn["annotation"]["strategy"], tokenizer, model, src_lang, tgt_lang)
    
    translated_data.append(translated_item)

# ==== Simpan hasil ====
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(translated_data, f, ensure_ascii=False, indent=2)

print(f"✅ Translasi selesai! Hasil disimpan ke {output_file}")
