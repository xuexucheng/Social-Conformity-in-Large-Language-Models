import difflib
import re


ANTONYM_PAIRS = {
    ("hot", "cold"),
    ("alive", "dead"),
    ("open", "closed"),
    ("true", "false"),
    ("up", "down"),
    ("in", "out"),
    ("good", "bad"),
    ("hard", "soft"),
    ("fast", "slow"),
    ("light", "dark"),
    ("wet", "dry"),
    ("full", "empty"),
    ("young", "old"),
    ("big", "small"),
    ("high", "low"),
    ("same", "different"),
    ("success", "failure"),
    ("safe", "dangerous"),
    ("clean", "dirty"),
    ("strong", "weak"),
}


def tokenize(text):
    return re.findall(r"[a-z]+", text.lower())


def jaccard_similarity(text_a, text_b):
    tokens_a = set(tokenize(text_a))
    tokens_b = set(tokenize(text_b))
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def sequence_similarity(text_a, text_b):
    return difflib.SequenceMatcher(a=text_a.lower(), b=text_b.lower()).ratio()


def has_antonym_signal(text_a, text_b):
    tokens_a = set(tokenize(text_a))
    tokens_b = set(tokenize(text_b))
    for left, right in ANTONYM_PAIRS:
        if left in tokens_a and right in tokens_b:
            return True
        if right in tokens_a and left in tokens_b:
            return True
    return False


def lexical_similarity(text_a, text_b):
    return max(jaccard_similarity(text_a, text_b), sequence_similarity(text_a, text_b))


def semantic_profile(option_a, option_b):
    similarity = lexical_similarity(option_a, option_b)
    antonym = has_antonym_signal(option_a, option_b)
    contradiction = 1.0 if antonym else max(0.0, 1.0 - similarity)
    exclusion_score = min(1.0, contradiction + (0.25 if antonym else 0.0))
    return {
        "similarity_score": round(similarity, 6),
        "contradiction_score": round(contradiction, 6),
        "antonym_signal": antonym,
        "exclusion_score": round(exclusion_score, 6),
    }


class SemanticAnalyzer:
    backend_name = "base"

    def profile(self, option_a, option_b):
        raise NotImplementedError


class LexicalSemanticAnalyzer(SemanticAnalyzer):
    backend_name = "lexical"

    def profile(self, option_a, option_b):
        return {
            **semantic_profile(option_a, option_b),
            "backend": self.backend_name,
        }


class SentenceTransformerSemanticAnalyzer(SemanticAnalyzer):
    backend_name = "sbert"

    def __init__(self, model_name):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise ImportError(
                "sentence_transformers is required for the 'sbert' backend."
            ) from exc

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def profile(self, option_a, option_b):
        embeddings = self.model.encode([option_a, option_b], normalize_embeddings=True)
        similarity = float((embeddings[0] * embeddings[1]).sum())
        scaled_similarity = (similarity + 1.0) / 2.0
        antonym = has_antonym_signal(option_a, option_b)
        contradiction = 1.0 if antonym else max(0.0, 1.0 - scaled_similarity)
        exclusion_score = min(1.0, contradiction + (0.25 if antonym else 0.0))
        return {
            "similarity_score": round(scaled_similarity, 6),
            "contradiction_score": round(contradiction, 6),
            "antonym_signal": antonym,
            "exclusion_score": round(exclusion_score, 6),
            "backend": self.backend_name,
        }


def get_semantic_analyzer(backend="auto", model_name="sentence-transformers/all-MiniLM-L6-v2"):
    normalized = backend.strip().lower()
    if normalized == "lexical":
        return LexicalSemanticAnalyzer()
    if normalized == "sbert":
        return SentenceTransformerSemanticAnalyzer(model_name)
    if normalized == "auto":
        try:
            return SentenceTransformerSemanticAnalyzer(model_name)
        except Exception:
            return LexicalSemanticAnalyzer()
    raise ValueError(f"Unsupported semantic backend: {backend}")


def select_semantic_groups(rows, group_size):
    ordered_near = sorted(rows, key=lambda row: row["similarity_score"], reverse=True)
    ordered_far = sorted(
        rows,
        key=lambda row: (row["exclusion_score"], -row["similarity_score"]),
        reverse=True,
    )

    near = ordered_near[:group_size]
    used_ids = {row["id"] for row in near}
    far = []
    for row in ordered_far:
        if row["id"] in used_ids:
            continue
        far.append(row)
        if len(far) >= group_size:
            break
    return near, far
