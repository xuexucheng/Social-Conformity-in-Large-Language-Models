import time
import requests

from src.config import API_URL, MODEL_NAME, TEMPERATURE, MAX_TOKENS


def _payload_logprob_flags(payload):
    return (
        f"payload_has_logprobs={'logprobs' in payload}; "
        f"payload_has_top_logprobs={'top_logprobs' in payload}"
    )


def _response_text(response):
    if response is None:
        return ""
    return (response.text or "")[:1000]


def _api_error(message, payload, response=None, exception=None):
    status_code = response.status_code if response is not None else None
    response_preview = _response_text(response)
    details = [
        message,
        f"API_URL={API_URL}",
        f"MODEL_NAME={MODEL_NAME}",
        f"status_code={status_code}",
        f"response_text_first_1000={response_preview!r}",
        _payload_logprob_flags(payload),
    ]
    if exception is not None:
        details.append(f"exception={type(exception).__name__}: {exception}")
    return RuntimeError("\n".join(details))


def post_chat_completion(payload):
    payload = {
        "model": MODEL_NAME,
        **payload,
    }

    last_response = None
    last_exception = None
    for attempt in range(1, 4):
        try:
            r = requests.post(API_URL, json=payload, timeout=120)
            last_response = r
            if r.status_code != 200:
                if attempt < 3:
                    time.sleep(2)
                    continue
                raise _api_error("Chat completion API returned non-200 status.", payload, response=r)
            try:
                return r.json(), r
            except ValueError as exc:
                raise _api_error("Chat completion API returned invalid JSON.", payload, response=r, exception=exc)
        except requests.RequestException as exc:
            last_exception = exc
            if attempt < 3:
                time.sleep(2)
                continue
            raise _api_error("Chat completion API request failed.", payload, response=last_response, exception=exc)
        except RuntimeError:
            raise
        except Exception as exc:
            last_exception = exc
            if attempt < 3:
                time.sleep(2)
                continue
            raise _api_error("Unexpected chat completion API failure.", payload, response=last_response, exception=exc)

    if last_exception is not None:
        raise _api_error("Chat completion API request failed.", payload, response=last_response, exception=last_exception)
    raise _api_error("Chat completion API failed without a response.", payload, response=last_response)


def _first_choice(data, payload, response=None):
    choices = data.get("choices")
    if not choices:
        raise _api_error("Chat completion response has no choices.", payload, response=response)
    return choices[0]


def _message_content(choice, payload, response=None):
    message = choice.get("message")
    if not isinstance(message, dict) or "content" not in message:
        raise _api_error(
            "Chat completion response choice is missing message/content.",
            payload,
            response=response,
        )
    content = message.get("content")
    if content is None or not str(content).strip():
        raise _api_error(
            "Chat completion response message/content is empty.",
            payload,
            response=response,
        )
    return content


def call_llm(messages):
    payload = {
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    data, response = post_chat_completion(payload)
    full_payload = {"model": MODEL_NAME, **payload}
    choice = _first_choice(data, full_payload, response=response)
    return _message_content(choice, full_payload, response=response)


def call_llm_with_logprobs(messages, max_tokens=1, temperature=0.0, top_logprobs=20):
    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "logprobs": True,
        "top_logprobs": top_logprobs,
    }
    data, response = post_chat_completion(payload)
    full_payload = {"model": MODEL_NAME, **payload}
    choice = _first_choice(data, full_payload, response=response)
    text = _message_content(choice, full_payload, response=response)

    logprobs = choice.get("logprobs") or {}
    content_logprobs = logprobs.get("content", []) or []
    top = content_logprobs[0].get("top_logprobs", []) if content_logprobs else []
    return {
        "text": text,
        "top_logprobs": top,
        "content_logprobs": content_logprobs,
        "raw": data,
    }
