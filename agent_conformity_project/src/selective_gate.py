import math

import numpy as np
from sklearn.linear_model import SGDClassifier


FEATURE_NAMES = [
    "initial_probability",
    "initial_entropy",
    "top1_top2_margin",
    "candidate_probability",
    "candidate_entropy",
    "delta_probability",
    "delta_entropy",
    "peer_consistency_score",
    "evidence_diversity_score",
    "evidence_independence_score",
    "pressure_language_score",
    "social_pressure_risk",
    "candidate_revision_strength",
    "baseline_error_score",
    "best_option_margin",
    "beneficial_revision_score",
    "candidate_evidence_strength",
    "correction_explanation_quality",
    "harmful_conformity_risk",
    "peer_pressure_index",
    "logic_echo_strength",
    "counterevidence_score",
    "first_switch_step_norm",
    "num_switches",
    "confidence_drop_slope",
    "entropy_rise_slope",
    "late_revision_signal",
    "pattern_unanimous",
    "pattern_diverse",
    "pattern_devils_advocate",
    "candidate_changed",
    "baseline_is_uncertain",
]


def _sigmoid(value):
    return 1.0 / (1.0 + math.exp(-value))


class SelectiveGate:
    def __init__(self):
        self.model = SGDClassifier(loss="log_loss", penalty="l2", alpha=0.0005, random_state=42)
        self.is_fitted = False
        self.training_examples = 0

    def vectorize(self, features):
        return np.array([float(features.get(name, 0.0)) for name in FEATURE_NAMES], dtype=float)

    def heuristic_accept_probability(self, features):
        score = 0.0
        score += 2.8 * float(features.get("beneficial_revision_score", 0.0))
        score += 1.8 * float(features.get("candidate_evidence_strength", 0.0))
        score += 1.3 * float(features.get("correction_explanation_quality", 0.0))
        score += 1.4 * float(features.get("baseline_error_score", 0.0))
        score += 1.0 * float(features.get("best_option_margin", 0.0))
        score += 0.8 * float(features.get("evidence_independence_score", 0.0))
        score += 0.6 * float(features.get("evidence_diversity_score", 0.0))
        score += 0.5 * float(features.get("candidate_changed", 0.0))
        score -= 2.6 * float(features.get("harmful_conformity_risk", 0.0))
        score -= 1.6 * float(features.get("peer_pressure_index", 0.0))
        score -= 1.2 * float(features.get("counterevidence_score", 0.0))
        score -= 1.0 * float(features.get("logic_echo_strength", 0.0))
        score -= 0.9 * float(features.get("pressure_language_score", 0.0))
        score -= 0.6 * float(features.get("pattern_unanimous", 0.0))
        score += 0.5 * float(features.get("baseline_is_uncertain", 0.0))
        score += 0.4 * float(features.get("late_revision_signal", 0.0))
        score += 0.3 * float(features.get("delta_probability", 0.0))
        return _sigmoid(score - 0.35)

    def predict(self, features):
        heuristic_accept = self.heuristic_accept_probability(features)
        if not self.is_fitted:
            accept_prob = heuristic_accept
        else:
            vector = self.vectorize(features).reshape(1, -1)
            learned_accept = float(self.model.predict_proba(vector)[0][1])
            mix = min(0.75, 0.2 + 0.05 * self.training_examples)
            accept_prob = (1.0 - mix) * heuristic_accept + mix * learned_accept

        reject_prob = 1.0 - accept_prob
        strict_prob = max(0.0, 1.0 - abs(accept_prob - 0.5) * 2.0)
        decision = "ACCEPT_REVISION" if accept_prob >= max(reject_prob, strict_prob) else "REJECT_REVISION"
        if strict_prob > 0.55 and abs(accept_prob - reject_prob) < 0.18:
            decision = "STRICT_VERIFY"

        return {
            "accept_revision_probability": round(accept_prob, 6),
            "reject_revision_probability": round(reject_prob, 6),
            "need_strict_verification_probability": round(strict_prob, 6),
            "decision": decision,
            "training_examples": self.training_examples,
        }

    def update(self, features, label):
        vector = self.vectorize(features).reshape(1, -1)
        target = np.array([int(label)], dtype=int)
        if not self.is_fitted:
            self.model.partial_fit(vector, target, classes=np.array([0, 1], dtype=int))
            self.is_fitted = True
        else:
            self.model.partial_fit(vector, target)
        self.training_examples += 1
