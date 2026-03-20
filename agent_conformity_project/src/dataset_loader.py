import json

from src.config import DATA_PATH


def load_dataset():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data