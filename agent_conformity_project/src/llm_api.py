import time
import requests

from src.config import API_URL, MODEL_NAME, TEMPERATURE, MAX_TOKENS


def call_llm(messages):
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }

    for _ in range(3):
        try:
            r = requests.post(API_URL, json=payload, timeout=120)
            if r.status_code == 200:
                data = r.json()
                return data["choices"][0]["message"]["content"]
        except Exception:
            time.sleep(2)

    return ""