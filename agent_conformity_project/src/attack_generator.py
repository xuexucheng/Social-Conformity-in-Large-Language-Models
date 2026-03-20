from src.config import N_ATTACK_AGENTS


def generate_attack_opinions(item, level: str):
    distractor = item["distractor"]

    if level == "L1":
        n_wrong = 1
    elif level == "L2":
        n_wrong = max(1, N_ATTACK_AGENTS // 2)
    elif level == "L3":
        n_wrong = N_ATTACK_AGENTS
    elif level == "L4":
        n_wrong = N_ATTACK_AGENTS
    elif level == "L5":
        n_wrong = N_ATTACK_AGENTS
    else:
        n_wrong = 0

    opinions = [distractor for _ in range(n_wrong)]
    return opinions


ATTACK_LEVELS = ["L1", "L2", "L3", "L4", "L5"]