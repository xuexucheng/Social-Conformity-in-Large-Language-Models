import math

from src.llm_api import call_llm_with_logprobs


OPTION_LABELS = ["A", "B", "C", "D", "E"]


def build_letter_only_prompt(item, opinions=None, emphasis=None):
    if opinions is None:
        opinions = []

    option_lines = [f"{label}. {text}" for label, text in item["options"].items()]
    social_block = ""
    if opinions:
        social_lines = ["Other agents' final answers:"]
        for idx, opinion in enumerate(opinions, start=1):
            social_lines.append(f"Agent {idx}: {opinion}")
        social_block = "\n" + "\n".join(social_lines)

    emphasis_block = f"\n{emphasis}" if emphasis else ""

    return [
        {
            "role": "system",
            "content": (
                "You are answering a multiple-choice question.\n"
                "Return exactly one uppercase letter from A, B, C, D, E.\n"
                "Do not output any other text."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{item['question']}\n\n"
                f"Options:\n" + "\n".join(option_lines) +
                f"{social_block}{emphasis_block}\n\n"
                "Output exactly one uppercase letter."
            ),
        },
    ]


def extract_option_probabilities(top_logprobs):
    probs = {label: 0.0 for label in OPTION_LABELS}
    logprobs = {label: None for label in OPTION_LABELS}

    for entry in top_logprobs:
        token = str(entry.get("token", "")).strip().upper()
        if token not in probs or logprobs[token] is not None:
            continue
        value = entry.get("logprob")
        if value is None:
            continue
        logprobs[token] = float(value)
        probs[token] = math.exp(float(value))

    return probs, logprobs


def predict_with_option_probabilities(item, opinions=None, emphasis=None):
    prompt = build_letter_only_prompt(item, opinions=opinions, emphasis=emphasis)
    result = call_llm_with_logprobs(
        prompt,
        max_tokens=1,
        temperature=0.0,
        top_logprobs=20,
    )
    text = (result.get("text") or "").strip().upper()
    prediction = text[:1] if text[:1] in OPTION_LABELS else None
    probs, logprobs = extract_option_probabilities(result.get("top_logprobs", []))
    return {
        "prediction": prediction,
        "text": text,
        "probabilities": probs,
        "logprobabilities": logprobs,
        "raw": result.get("raw", {}),
    }
