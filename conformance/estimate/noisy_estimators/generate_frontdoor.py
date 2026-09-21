#!/usr/bin/env python3
"""Regenerate frontdoor.csv (standard library only; deterministic).

SCM: U ~ N(0,1) unobserved; T = 3 + U + 0.5 e (uncentred); M = 0.4 T + 0.5 e;
Y = 5 M + U + 0.5 e. The mediated effect is 0.4 * 5 = 2, and neither path
coefficient alone (0.4, 5) nor the confounded Y ~ T slope is near it.

    python3 conformance/estimate/noisy_estimators/generate_frontdoor.py
"""

import random
from pathlib import Path

N = 800
SEED = 7


def main() -> None:
    rng = random.Random(SEED)
    rows = ["t,y,m"]
    for _ in range(N):
        u = rng.gauss(0.0, 1.0)
        t = 3.0 + u + 0.5 * rng.gauss(0.0, 1.0)
        m = 0.4 * t + 0.5 * rng.gauss(0.0, 1.0)
        y = 5.0 * m + u + 0.5 * rng.gauss(0.0, 1.0)
        rows.append(f"{t!r},{y!r},{m!r}")
    Path(__file__).with_name("frontdoor.csv").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
