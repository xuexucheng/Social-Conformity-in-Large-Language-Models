import re

ANSWER_PAT = re.compile(r"ANSWER:\s*([A-E])\b", re.I)
CONF_PAT = re.compile(r"CONFIDENCE:\s*(\d{1,3})", re.I)
STANDALONE_ANSWER_PAT = re.compile(r"^[\s\(\[]*([A-E])[\s\)\].,:;!?]*$", re.I)


def parse_answer(text):
    if text is None:
        return None

    upper_text = text.upper()

    # 如果模型输出了多个选项，如 B/C/D/E，则视为无效
    answer_line_match = re.search(r"ANSWER:\s*(.+)", upper_text)
    if answer_line_match:
        answer_line = answer_line_match.group(1).strip()
        if "/" in answer_line or "," in answer_line:
            return None

    m = ANSWER_PAT.search(upper_text)
    if m:
        return m.group(1).upper()

    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    for line in reversed(lines):
        m = STANDALONE_ANSWER_PAT.match(line)
        if m:
            return m.group(1).upper()
    return None


def parse_confidence(text):
    if text is None:
        return None

    m = CONF_PAT.search(text)
    if not m:
        return None

    val = int(m.group(1))
    val = max(0, min(100, val))
    return val
