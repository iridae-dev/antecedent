#!/usr/bin/env python3
"""Regenerate frontdoor.csv (deterministic).

SCM: U ~ N(0,1) unobserved; T = 3 + U + 0.5 e (uncentred); M = 0.4 T + 0.5 e;
Y = 5 M + U + 0.5 e. The mediated effect is 0.4 * 5 = 2, and neither path
coefficient alone (0.4, 5) nor the confounded Y ~ T slope is near it.

    python3 conformance/estimate/noisy_estimators/generate_frontdoor.py
    python3 conformance/estimate/noisy_estimators/generate_frontdoor.py --check
"""

import random
import sys
from pathlib import Path

import numpy as np

N = 800
SEED = 7


def render() -> str:
    rng = random.Random(SEED)
    rows = ["t,y,m"]
    for _ in range(N):
        u = rng.gauss(0.0, 1.0)
        t = 3.0 + u + 0.5 * rng.gauss(0.0, 1.0)
        m = 0.4 * t + 0.5 * rng.gauss(0.0, 1.0)
        y = 5.0 * m + u + 0.5 * rng.gauss(0.0, 1.0)
        rows.append(f"{t!r},{y!r},{m!r}")
    return "\n".join(rows) + "\n"


def _columns(text: str) -> np.ndarray:
    """Parse ``t,y,m`` rows. The header is not a numeric pin."""
    lines = [line for line in text.splitlines() if line]
    if not lines or lines[0] != "t,y,m":
        raise SystemExit("frontdoor.csv header drifted")
    return np.asarray(
        [[float(part) for part in line.split(",")] for line in lines[1:]],
        dtype=np.float64,
    )


def main() -> None:
    path = Path(__file__).with_name("frontdoor.csv")
    text = render()
    if "--check" in sys.argv:
        # rtol=0 and atol=0: a finite committed row must match the generator.
        stored = _columns(path.read_text(encoding="utf-8"))
        fresh = _columns(text)
        if stored.shape != fresh.shape or not np.all(np.isclose(stored, fresh, rtol=0.0, atol=0.0)):
            raise SystemExit("frontdoor.csv is stale; rerun without --check")
        print("frontdoor.csv matches")
        return
    path.write_text(text, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
