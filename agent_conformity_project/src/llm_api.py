import time
import requests

from src.config import API_URL, MODEL_NAME, TEMPERATURE, MAX_TOKENS


def post_chat_completion(payload):
    payload = {
        "model": MODEL_NAME,
        **payload,
    }

    for _ in range(3):
        try:
            r = requests.post(API_URL, json=payload, timeout=120)
            if r.status_code == 200:
                return r.json()
        except Exception:
            time.sleep(2)

    return {}


def call_llm(messages):
    data = post_chat_completion(
        {
            "messages": messages,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
        }
    )
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        return ""
