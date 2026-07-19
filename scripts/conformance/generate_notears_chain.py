#!/usr/bin/env python3
"""Generate frozen NOTEARS linear-SEM chain fixture (x0 → x1 → x2).

Deterministic LCG-style noise; no upstream oracle. Re-run only to refresh the
frozen CSV when the SCM recipe in expected.json changes.
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "conformance" / "discovery" / "notears_chain"
N = 800


def noise(i: int, a: float) -> float:
    return ((i * a) % 1.0) - 0.5


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(N):
        e0 = noise(i, 0.137)
        e1 = noise(i, 0.271)
        e2 = noise(i, 0.419)
        x0 = e0
        x1 = 0.8 * x0 + e1
        x2 = 0.8 * x1 + e2
        rows.append((x0, x1, x2))
    path = OUT / "data.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x0", "x1", "x2"])
        w.writerows(rows)
    print(f"wrote {path} n={N}")


if __name__ == "__main__":
    main()
