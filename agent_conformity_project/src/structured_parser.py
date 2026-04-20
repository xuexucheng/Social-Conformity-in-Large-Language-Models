import json
import re


JSON_BLOCK_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_object(text, default=None):
    if default is None:
        default = {}

    if not text:
        return dict(default)

    stripped = text.strip()
    try:
        data = json.loads(stripped)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    match = JSON_BLOCK_PATTERN.search(stripped)
    if not match:
        return dict(default)

    try:
        data = json.loads(match.group(0))
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        return dict(default)

    return dict(default)


def normalize_score(value, default=0.0):
    try:
        score = float(value)
    except (TypeError, ValueError):
        return float(default)
    return max(0.0, min(1.0, score))


def normalize_confidence(value, default=0):
    try:
        confidence = int(round(float(value)))
    except (TypeError, ValueError):
        return int(default)
    return max(0, min(100, confidence))
