import re

OPTION_LABELS = ("A", "B", "C", "D", "E")
ANSWER_PAT = re.compile(r"ANSWER:\s*([A-E])\b", re.I)
CONF_PAT = re.compile(r"CONFIDENCE:\s*(\d{1,3})", re.I)
CODE_FENCE_RE = re.compile(r"```(?:[\w.+-]+)?|```", re.I)
WHITESPACE_RE = re.compile(r"\s+")
STANDALONE_LINE_PAT = re.compile(
    r'^[\s>\-*`_~"\']*[\(\[]?\s*([A-E])\s*[\)\]]?[\s\.,:;!?*_~"`\'-]*$',
    re.I,
)
EXPLICIT_PATTERNS = [
    (
        "json_answer_key",
        re.compile(
            r'"(?:answer|final_answer|choice)"\s*:\s*"?(?:OPTION\s+)?([A-E])"?',
            re.I,
        ),
    ),
    (
        "final_answer_phrase",
        re.compile(r"\bFINAL\s+ANSWER\s*(?:IS|:)?\s*[\(\[]?\s*([A-E])\s*[\)\]]?", re.I),
    ),
    (
        "correct_answer_phrase",
        re.compile(r"\bTHE\s+CORRECT\s+ANSWER\s+IS\s*[\(\[]?\s*([A-E])\s*[\)\]]?", re.I),
    ),
    (
        "my_answer_phrase",
        re.compile(r"\bMY\s+ANSWER\s+IS\s*[\(\[]?\s*([A-E])\s*[\)\]]?", re.I),
    ),
    (
        "answer_phrase",
        re.compile(r"\bANSWER\s*(?:IS|:)?\s*[\(\[]?\s*([A-E])\s*[\)\]]?", re.I),
    ),
    (
        "choice_phrase",
        re.compile(r"\bCHOICE\s*(?:IS|:)?\s*[\(\[]?\s*([A-E])\s*[\)\]]?", re.I),
    ),
    (
        "option_phrase",
        re.compile(r"\bOPTION\s+([A-E])\b", re.I),
    ),
    (
        "i_choose_phrase",
        re.compile(r"\bI\s+CHOOSE\s*[\(\[]?\s*([A-E])\s*[\)\]]?", re.I),
    ),
    (
        "zh_answer_phrase",
        re.compile(r"\u7B54\u6848\u662F\s*[\(\[]?\s*([A-E])\s*[\)\]]?", re.I),
    ),
    (
        "zh_choose_phrase",
        re.compile(r"\u6211\u9009\s*[\(\[]?\s*([A-E])\s*[\)\]]?", re.I),
    ),
    (
        "zh_select_phrase",
        re.compile(r"\u6211\u9009\u62E9\s*[\(\[]?\s*([A-E])\s*[\)\]]?", re.I),
    ),
    (
        "zh_selection_phrase",
        re.compile(r"\u9009\u62E9\s*[\(\[]?\s*([A-E])\s*[\)\]]?", re.I),
    ),
]
TRAILING_SENTENCE_SPLIT_RE = re.compile(r"(?<=[\.\!\?。！？])\s+")
FALLBACK_TOKEN_RE = re.compile(r"(?<![A-Z0-9])([A-E])(?![A-Z0-9])", re.I)


def _clean_text(text):
    if text is None:
        return ""
    cleaned = str(text).replace("\r\n", "\n").replace("\r", "\n").replace("\ufeff", "")
    cleaned = CODE_FENCE_RE.sub("", cleaned)
    return cleaned.strip()


def _normalize_text(text):
    cleaned = _clean_text(text)
    cleaned = cleaned.replace("**", "").replace("__", "").replace("*", "").replace("_", "")
    cleaned = cleaned.replace("`", "").replace("~", "")
    return cleaned


def _collapse_whitespace(text):
    return WHITESPACE_RE.sub(" ", text or "").strip()


def _normalized_option_text(text):
    normalized = _normalize_text(text).lower()
    normalized = re.sub(r"[\"'“”‘’\(\)\[\]\{\}\.,:;!?/\\-]+", " ", normalized)
    return _collapse_whitespace(normalized)


def _candidate_segments(text):
    normalized = _normalize_text(text)
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    sentences = [part.strip() for part in TRAILING_SENTENCE_SPLIT_RE.split(normalized) if part.strip()]
    candidates = []

    def add(segment):
        segment = segment.strip()
        if not segment:
            return
        if segment not in candidates:
            candidates.append(segment)

    for line in reversed(lines[-5:]):
        add(line)
    for size in range(min(5, len(lines)), 1, -1):
        add("\n".join(lines[-size:]))
    for sentence in reversed(sentences[-3:]):
        add(sentence)
    add(normalized)
    return candidates


def _structured_result(answer=None, parse_method=None, matched_text=None, matched_span=None, normalized_text=""):
    return {
        "answer": answer,
        "parse_status": "success" if answer else "failed",
        "parse_method": parse_method,
        "matched_text": matched_text,
        "matched_span": matched_span,
        "normalized_text": normalized_text,
    }


def _match_explicit_patterns(segment, normalized_text):
    for method, pattern in EXPLICIT_PATTERNS:
        match = pattern.search(segment)
        if match:
            return _structured_result(
                answer=match.group(1).upper(),
                parse_method=method,
                matched_text=match.group(0),
                matched_span=match.span(),
                normalized_text=normalized_text,
            )
    return None


def _match_standalone_line(segment, normalized_text):
    match = STANDALONE_LINE_PAT.match(segment)
    if not match:
        return None
    return _structured_result(
        answer=match.group(1).upper(),
        parse_method="standalone_line",
        matched_text=match.group(0),
        matched_span=match.span(),
        normalized_text=normalized_text,
    )


def _match_option_text(segment, options, normalized_text):
    if not isinstance(options, dict) or not options:
        return None

    normalized_segment = _normalized_option_text(segment)
    matches = []
    for label, option_text in options.items():
        option_label = str(label).strip().upper()
        if option_label not in OPTION_LABELS:
            continue
        candidate = _normalized_option_text(option_text)
        if len(candidate) < 4:
            continue
        if candidate and candidate in normalized_segment:
            matches.append((option_label, option_text))

    if len(matches) != 1:
        return None

    answer, matched_text = matches[0]
    return _structured_result(
        answer=answer,
        parse_method="option_text_match",
        matched_text=str(matched_text),
        matched_span=None,
        normalized_text=normalized_text,
    )


def _match_fallback_token(segment, normalized_text):
    condensed = _collapse_whitespace(segment)
    if len(condensed) > 24:
        return None

    tokens = [match.group(1).upper() for match in FALLBACK_TOKEN_RE.finditer(condensed)]
    unique_tokens = list(dict.fromkeys(tokens))
    if len(unique_tokens) != 1 or len(tokens) != 1:
        return None

    match = FALLBACK_TOKEN_RE.search(condensed)
    return _structured_result(
        answer=unique_tokens[0],
        parse_method="fallback_single_token",
        matched_text=match.group(0) if match else condensed,
        matched_span=match.span() if match else None,
        normalized_text=normalized_text,
    )


def parse_answer_details(text, options=None):
    normalized_text = _normalize_text(text)
    if not normalized_text:
        return _structured_result(normalized_text=normalized_text)

    for segment in _candidate_segments(normalized_text):
        result = _match_explicit_patterns(segment, normalized_text)
        if result:
            return result

    for segment in _candidate_segments(normalized_text):
        result = _match_standalone_line(segment, normalized_text)
        if result:
            return result

    for segment in _candidate_segments(normalized_text):
        result = _match_option_text(segment, options, normalized_text)
        if result:
            return result

    for segment in _candidate_segments(normalized_text):
        result = _match_fallback_token(segment, normalized_text)
        if result:
            return result

    return _structured_result(normalized_text=normalized_text)


def parse_answer(text, options=None):
    return parse_answer_details(text, options=options).get("answer")


def parse_confidence(text):
    if text is None:
        return None

    m = CONF_PAT.search(text)
    if not m:
        return None

    val = int(m.group(1))
    val = max(0, min(100, val))
    return val
