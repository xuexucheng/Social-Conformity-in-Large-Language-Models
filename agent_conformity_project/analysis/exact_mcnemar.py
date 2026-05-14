#!/usr/bin/env python3
"""Compute an exact two-sided McNemar test from paired discordant counts.

The test uses the exact binomial formulation with p=0.5 on the discordant
pairs. It is dependency-free and intended for paper table sanity checks.
"""

from __future__ import annotations

import argparse
import math


def binom_pmf(k: int, n: int) -> float:
    return math.comb(n, k) * (0.5**n)


def exact_mcnemar_pvalue(b: int, c: int) -> float:
    """Return the exact two-sided McNemar p-value for discordant counts b,c."""
    if b < 0 or c < 0:
        raise ValueError("Discordant counts must be non-negative.")

    n = b + c
    if n == 0:
        return 1.0

    tail = sum(binom_pmf(k, n) for k in range(0, min(b, c) + 1))
    return min(1.0, 2.0 * tail)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Exact two-sided McNemar test. Use b for pairs where condition 1 "
            "succeeds and condition 2 fails, and c for the reverse."
        )
    )
    parser.add_argument("--b", type=int, required=True)
    parser.add_argument("--c", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pvalue = exact_mcnemar_pvalue(args.b, args.c)
    print(f"b={args.b}")
    print(f"c={args.c}")
    print(f"discordant={args.b + args.c}")
    print(f"exact_two_sided_p={pvalue:.10g}")


if __name__ == "__main__":
    main()
