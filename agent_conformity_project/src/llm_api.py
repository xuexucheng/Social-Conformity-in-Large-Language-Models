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


def call_llm_with_logprobs(messages, max_tokens=1, temperature=0.0, top_logprobs=20):
    data = post_chat_completion(
        {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "logprobs": True,
            "top_logprobs": top_logprobs,
        }
    )
    if not data:
        return {"text": "", "top_logprobs": [], "raw": {}}

    try:
        choice = data["choices"][0]
        text = choice["message"]["content"]
        content_logprobs = choice.get("logprobs", {}).get("content", []) or []
        top = content_logprobs[0]["top_logprobs"] if content_logprobs else []
        return {
            "text": text,
            "top_logprobs": top,
            "content_logprobs": content_logprobs,
            "raw": data,
        }
    except Exception:
        return {"text": "", "top_logprobs": [], "content_logprobs": [], "raw": data}
