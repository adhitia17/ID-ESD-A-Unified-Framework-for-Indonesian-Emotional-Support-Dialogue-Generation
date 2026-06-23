import requests
import json

API_URL = "https://mkp-api.fptcloud.com/chat/completions"
API_KEY = "sk-K-BOY2FM0rmYxnaae8lzyQ"

def query_llm(messages, model="Llama-3.3-70B-Instruct", temperature=0.1, max_tokens=512):
    """
    Kirim request ke LLM API.
    messages: list of dict [{role: "system"/"user"/"assistant", content: "..."}]
    return: string jawaban dari LLM
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    response = requests.post(API_URL, headers=headers, data=json.dumps(payload))

    if response.status_code != 200:
        raise Exception(f"Error {response.status_code}: {response.text}")

    data = response.json()
    return data["choices"][0]["message"]["content"]
