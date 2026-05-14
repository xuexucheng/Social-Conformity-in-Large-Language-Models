import argparse
import gzip
import json
import os
import sys
import tarfile
from collections import Counter


FINAL_FIELD_ALIASES = ("final_answer", "attack_prediction", "prediction")
RAW_OUTPUT_ALIASES = ("raw_output", "raw_attack_output", "text")
OPTION_LOGPROBS_ALIASES = ("option_logprobs", "attack_option_logprobs")
ITEM_ID_ALIASES = ("item_id", "id")


def _is_empty_value(value):
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) == 0
    return False


def _first_present(record, keys):
    for key in keys:
        if key in record:
            return key, record.get(key)
    return None, None


def _line_preview(line, width=500):
    text = line.rstrip("\n\r")
    return text[:width], text[-width:] if len(text) > width else text


def _new_stats(source_name):
    return {
        "source": source_name,
        "total_lines": 0,
        "empty_lines": 0,
        "json_ok": 0,
        "json_failed": 0,
        "bad_lines": [],
        "is_error_true": 0,
        "final_answer_null": 0,
        "final_answer_empty": 0,
        "final_answer_valid": 0,
        "analysis_final_null": 0,
        "analysis_final_empty": 0,
        "analysis_final_valid": 0,
        "raw_output_empty": 0,
        "raw_output_nonempty": 0,
        "analysis_raw_output_empty": 0,
        "analysis_raw_output_nonempty": 0,
        "option_logprobs_empty": 0,
        "option_logprobs_nonempty": 0,
        "analysis_option_logprobs_empty": 0,
        "analysis_option_logprobs_nonempty": 0,
        "item_id_missing": 0,
        "item_id_duplicates": 0,
        "duplicate_item_ids": [],
        "unique_item_ids": 0,
        "effective_samples": 0,
        "ineffective_rows": [],
        "experiment_modes": Counter(),
    }


def check_jsonl_lines(source_name, lines):
    stats = _new_stats(source_name)
    item_ids = []

    for line_number, line in enumerate(lines, start=1):
        stats["total_lines"] += 1
        if not line.strip():
            stats["empty_lines"] += 1
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            first, last = _line_preview(line)
            stats["json_failed"] += 1
            stats["bad_lines"].append(
                {
                    "line_number": line_number,
                    "error": str(exc),
                    "first_500": first,
                    "last_500": last,
                    "length": len(line.rstrip("\n\r")),
                }
            )
            continue

        stats["json_ok"] += 1
        if record.get("is_error") is True:
            stats["is_error_true"] += 1

        final_answer = record.get("final_answer")
        if final_answer is None:
            stats["final_answer_null"] += 1
        elif isinstance(final_answer, str) and not final_answer.strip():
            stats["final_answer_empty"] += 1
        else:
            stats["final_answer_valid"] += 1

        _, analysis_final = _first_present(record, FINAL_FIELD_ALIASES)
        if analysis_final is None:
            stats["analysis_final_null"] += 1
        elif isinstance(analysis_final, str) and not analysis_final.strip():
            stats["analysis_final_empty"] += 1
        else:
            stats["analysis_final_valid"] += 1

        raw_output = record.get("raw_output")
        if _is_empty_value(raw_output):
            stats["raw_output_empty"] += 1
        else:
            stats["raw_output_nonempty"] += 1

        _, analysis_raw_output = _first_present(record, RAW_OUTPUT_ALIASES)
        if _is_empty_value(analysis_raw_output):
            stats["analysis_raw_output_empty"] += 1
        else:
            stats["analysis_raw_output_nonempty"] += 1

        option_logprobs = record.get("option_logprobs")
        if _is_empty_value(option_logprobs):
            stats["option_logprobs_empty"] += 1
        else:
            stats["option_logprobs_nonempty"] += 1

        _, analysis_option_logprobs = _first_present(record, OPTION_LOGPROBS_ALIASES)
        if _is_empty_value(analysis_option_logprobs):
            stats["analysis_option_logprobs_empty"] += 1
        else:
            stats["analysis_option_logprobs_nonempty"] += 1

        item_key, item_id = _first_present(record, ITEM_ID_ALIASES)
        if item_key is None or _is_empty_value(item_id):
            stats["item_id_missing"] += 1
        else:
            item_ids.append(str(item_id))

        mode = record.get("experiment_mode")
        if mode:
            stats["experiment_modes"][str(mode)] += 1

        if record.get("is_error") is not True and not _is_empty_value(analysis_final):
            stats["effective_samples"] += 1
        else:
            stats["ineffective_rows"].append(
                {
                    "line_number": line_number,
                    "item_id": item_id,
                    "is_error": record.get("is_error"),
                    "analysis_final": analysis_final,
                }
            )

    item_counts = Counter(item_ids)
    duplicate_ids = sorted(item_id for item_id, count in item_counts.items() if count > 1)
    stats["unique_item_ids"] = len(item_counts)
    stats["item_id_duplicates"] = sum(item_counts[item_id] - 1 for item_id in duplicate_ids)
    stats["duplicate_item_ids"] = duplicate_ids
    return stats


def iter_directory_jsonl(path):
    for root, _, filenames in os.walk(path):
        for filename in sorted(filenames):
            if filename.startswith("exp") and filename.endswith(".jsonl"):
                full_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_path, path)
                with open(full_path, "r", encoding="utf-8") as handle:
                    yield rel_path, handle


def iter_tar_jsonl(path):
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            filename = os.path.basename(member.name)
            if not member.isfile() or not filename.startswith("exp") or not filename.endswith(".jsonl"):
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            with extracted:
                text_stream = (line.decode("utf-8", errors="replace") for line in extracted)
                yield member.name, text_stream


def iter_gzip_jsonl(path):
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        yield os.path.basename(path), handle


def check_path(path):
    if os.path.isdir(path):
        iterator = iter_directory_jsonl(path)
    elif tarfile.is_tarfile(path):
        iterator = iter_tar_jsonl(path)
    elif path.endswith(".gz"):
        iterator = iter_gzip_jsonl(path)
    else:
        raise ValueError(f"Unsupported input path: {path}")

    results = []
    for source_name, lines in iterator:
        results.append(check_jsonl_lines(source_name, lines))
    return results


def print_stats(stats):
    print(f"## {stats['source']}")
    print(f"total_lines: {stats['total_lines']}")
    print(f"empty_lines: {stats['empty_lines']}")
    print(f"json_ok: {stats['json_ok']}")
    print(f"json_failed: {stats['json_failed']}")
    print(f"is_error_true: {stats['is_error_true']}")
    print(
        "final_answer exact null/empty/valid: "
        f"{stats['final_answer_null']}/{stats['final_answer_empty']}/{stats['final_answer_valid']}"
    )
    print(
        "analysis final alias null/empty/valid "
        f"({', '.join(FINAL_FIELD_ALIASES)}): "
        f"{stats['analysis_final_null']}/{stats['analysis_final_empty']}/{stats['analysis_final_valid']}"
    )
    print(
        "raw_output exact empty/nonempty: "
        f"{stats['raw_output_empty']}/{stats['raw_output_nonempty']}"
    )
    print(
        "analysis raw output alias empty/nonempty "
        f"({', '.join(RAW_OUTPUT_ALIASES)}): "
        f"{stats['analysis_raw_output_empty']}/{stats['analysis_raw_output_nonempty']}"
    )
    print(
        "option_logprobs exact empty/nonempty: "
        f"{stats['option_logprobs_empty']}/{stats['option_logprobs_nonempty']}"
    )
    print(
        "analysis option logprobs alias empty/nonempty "
        f"({', '.join(OPTION_LOGPROBS_ALIASES)}): "
        f"{stats['analysis_option_logprobs_empty']}/{stats['analysis_option_logprobs_nonempty']}"
    )
    print(f"item_id missing: {stats['item_id_missing']}")
    print(f"unique_item_ids: {stats['unique_item_ids']}")
    print(f"duplicate item_id extra rows: {stats['item_id_duplicates']}")
    print(f"effective_samples: {stats['effective_samples']}")
    if stats["ineffective_rows"]:
        print("ineffective_rows:")
        for row in stats["ineffective_rows"][:20]:
            print(
                f"  line {row['line_number']}: item_id={row['item_id']!r}, "
                f"is_error={row['is_error']!r}, analysis_final={row['analysis_final']!r}"
            )
        if len(stats["ineffective_rows"]) > 20:
            print(f"  ... {len(stats['ineffective_rows']) - 20} more")
    if stats["experiment_modes"]:
        modes = ", ".join(f"{mode}={count}" for mode, count in sorted(stats["experiment_modes"].items()))
        print(f"experiment_modes: {modes}")
    if stats["duplicate_item_ids"]:
        print("duplicate_item_ids: " + ", ".join(stats["duplicate_item_ids"][:20]))
        if len(stats["duplicate_item_ids"]) > 20:
            print(f"duplicate_item_ids_truncated: {len(stats['duplicate_item_ids']) - 20} more")
    for bad_line in stats["bad_lines"]:
        print(f"- bad line {bad_line['line_number']}")
        print(f"  error: {bad_line['error']}")
        print(f"  length: {bad_line['length']}")
        print(f"  first_500: {bad_line['first_500']!r}")
        print(f"  last_500: {bad_line['last_500']!r}")
    print()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Check JSONL integrity for experiment result directories or tar.gz archives."
    )
    parser.add_argument("paths", nargs="+", help="Directory, .jsonl.gz, or .tar.gz path to inspect.")
    args = parser.parse_args(argv)

    any_missing = False
    for path in args.paths:
        print(f"# Input: {path}")
        if not os.path.exists(path):
            print(f"ERROR: path does not exist: {path}", file=sys.stderr)
            any_missing = True
            continue
        results = check_path(path)
        if not results:
            print("No exp*.jsonl files found.")
        for stats in results:
            print_stats(stats)
    return 1 if any_missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
