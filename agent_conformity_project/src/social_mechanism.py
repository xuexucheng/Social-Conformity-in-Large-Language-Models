from src.config import N_ATTACK_AGENTS


def build_mechanism_conditions(item):
    distractor = item["distractor"]
    n_agents = max(1, N_ATTACK_AGENTS)
    return [
        {
            "dimension": "input_mode_comparison",
            "condition_name": f"target_consensus_n{n_agents}",
            "opinions": [distractor] * n_agents,
            "attack_level": f"N{n_agents}",
        }
    ]
