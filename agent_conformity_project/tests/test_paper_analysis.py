#!/usr/bin/env python3
"""Dependency-free tests for paper-facing paired and repeated-subset analysis."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "agent_conformity_project" / "analysis"
sys.path.insert(0, str(ANALYSIS))

import paired_protocol_analysis as paired  # noqa: E402
import repeated_subset_analysis as repeated  # noqa: E402


def _row(item_id: int, final: str | None, initial: str = "A", question: str | None = None):
    return {
        "id": f"item-{item_id}",
        "question": question or f"Question {item_id}",
        "options": {"A": "gold", "B": "target", "C": "other"},
        "correct_answer": "A",
        "distractor": "B",
        "initial_prediction": initial,
        "attack_prediction": final,
        "is_error": final is None,
    }


def _write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _fixture(temp: Path, n: int = 10) -> Path:
    paths = {}
    finals = {
        "exp1": ["A"] * n,
        "exp2": ["B"] + ["A"] * (n - 1),
        "exp3": ["A"] * n,
        "exp4": ["A"] * n,
    }
    for condition, values in finals.items():
        path = temp / f"{condition}.jsonl"
        _write_jsonl(path, [_row(index, value) for index, value in enumerate(values)])
        paths[condition] = path.name
    manifest = {
        "comparisons": [["exp1", "exp2"], ["exp2", "exp3"]],
        "cells": [
            {
                "dataset": "Synthetic",
                "model": "stub",
                "conditions": paths,
            }
        ],
    }
    manifest_path = temp / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_normalize_legacy_schema():
    record = paired.normalize_row(_row(1, "B"))
    assert record["id"] == "item-1"
    assert record["valid"] is True
    assert record["initial"] == "A"
    assert record["final"] == "B"


def test_common_valid_pair_and_direction():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest = paired.load_manifest(_fixture(root))
        payload = paired.analyze_manifest(manifest, bootstrap_reps=200, bootstrap_seed=7)
        row = next(
            item
            for item in payload["comparisons"]
            if item["condition1"] == "exp1"
            and item["condition2"] == "exp2"
            and item["metric"] == "harmful_conformity"
        )
        assert row["paired_n"] == 10
        assert row["rate1"] == 0.0
        assert row["rate2"] == 0.1
        assert row["delta"] == 0.1
        assert row["mcnemar_b"] == 0 and row["mcnemar_c"] == 1


def test_invalid_rows_are_excluded_from_pair():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        one = root / "one.jsonl"
        two = root / "two.jsonl"
        _write_jsonl(one, [_row(0, "A"), _row(1, "A")])
        _write_jsonl(two, [_row(0, None), _row(1, "B")])
        records1 = paired.load_condition(one)
        records2 = paired.load_condition(two)
        ids = paired.paired_ids(records1, records2, lambda _row: True)
        assert ids == ["item-1"]


def test_metadata_drift_fails_closed():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        one = root / "one.jsonl"
        two = root / "two.jsonl"
        _write_jsonl(one, [_row(0, "A")])
        _write_jsonl(two, [_row(0, "A", question="different")])
        try:
            paired.audit_pair("one", paired.load_condition(one), "two", paired.load_condition(two))
        except ValueError as exc:
            assert "metadata drift" in str(exc)
        else:
            raise AssertionError("metadata drift must fail closed")


def test_repeated_subset_is_deterministic_and_aligned():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest = paired.load_manifest(_fixture(root, n=10))
        first = repeated.analyze_repeated_subsets(
            manifest, sample_size=5, repetitions=4, seed_start=100
        )
        second = repeated.analyze_repeated_subsets(
            manifest, sample_size=5, repetitions=4, seed_start=100
        )
        assert first == second
        assert first["sampling"]["frame_sizes"] == {"Synthetic": 10}
        assert all(len(sample["sample_ids"]) == 5 for sample in first["samples"])


def _run_all() -> bool:
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    passed = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
            passed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}")
    print(f"{passed}/{len(tests)} tests passed")
    return passed == len(tests)


if __name__ == "__main__":
    raise SystemExit(0 if _run_all() else 1)
