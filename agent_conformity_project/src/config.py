import os


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/v1/chat/completions")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")

RAW_DATA_DIR = os.getenv("RAW_DATA_DIR", "data/raw")
DATA_SPLIT = os.getenv("DATA_SPLIT", "validation")
DATA_PATH = os.getenv("DATA_PATH", "data/dataset.json")
RESULT_DIR = os.getenv("RESULT_DIR", "results")

N_SAMPLES = int(os.getenv("N_SAMPLES", "500"))
N_ATTACK_AGENTS = int(os.getenv("N_ATTACK_AGENTS", "5"))
SEED = int(os.getenv("SEED", "42"))

MAX_TOKENS = int(os.getenv("MAX_TOKENS", "128"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))

USE_CONFIDENCE = os.getenv("USE_CONFIDENCE", "1") == "1"


def _sanitize_name(name: str) -> str:
    name = name.strip().lower()
    for ch in ["/", "\\", " ", ":", "*", "?", "\"", "<", ">", "|"]:
        name = name.replace(ch, "-")
    while "--" in name:
        name = name.replace("--", "-")
    return name.strip("-")


MODEL_TAG = _sanitize_name(os.path.basename(MODEL_NAME))
RUN_NAME = os.getenv("RUN_NAME", f"{MODEL_TAG}_n{N_SAMPLES}")
RUNS_DIR = os.path.join(RESULT_DIR, "runs")
RUN_DIR = os.path.join(RUNS_DIR, RUN_NAME)