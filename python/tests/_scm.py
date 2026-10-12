"""Confounded SCM shared by tests.

Z → T, Z → Y, T → Y, structural ATE 2. The noise, columns, and edge list are
the same at every call site. ``n`` and ``seed`` are not: each caller passes the
sample size and seed it already used. This generator has no defaults so those
cannot collapse into one.
"""

from __future__ import annotations

import math
import random

import numpy as np


def confounded_scm(n: int, seed: int) -> tuple[dict[str, np.ndarray], list[tuple[str, str]]]:
    rng = random.Random(seed)
    z = np.empty(n, dtype=np.float64)
    t = np.empty(n, dtype=np.float64)
    y = np.empty(n, dtype=np.float64)
    for i in range(n):
        zi = rng.gauss(0.0, 1.0)
        p = 1.0 / (1.0 + math.exp(-(-0.4 + 0.9 * zi)))
        ti = 1.0 if rng.random() < p else 0.0
        yi = 2.0 * ti + zi + rng.gauss(0.0, 0.4)
        z[i] = zi
        t[i] = ti
        y[i] = yi
    return {"t": t, "y": y, "z": z}, [("z", "t"), ("z", "y"), ("t", "y")]
