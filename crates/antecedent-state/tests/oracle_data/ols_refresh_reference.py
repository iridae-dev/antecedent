"""Least-squares refresh sequences against exact rational arithmetic.

Each scenario is a list of rows (id, x, y) and a sequence of operations on the retained rows:
append one row, append a window, evict a row, change a count weight, mask or unmask a row. After
every operation the reference recomputes, from the live rows alone (unmasked, count > 0, each
counted `count` times), the sample size, the least-squares coefficients solving X'X b = X'y and
the residual variance SSE / (n - p), all in `Fraction` arithmetic on the binary64 inputs.

    python3 ols_refresh_reference.py > ols_refresh_reference.json
"""

import json
from fractions import Fraction as F


def solve(a, b):
    n = len(a)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = next((r for r in range(col, n) if m[r][col] != 0), None)
        if piv is None:
            return None
        m[col], m[piv] = m[piv], m[col]
        p = m[col][col]
        m[col] = [v / p for v in m[col]]
        for r in range(n):
            if r != col and m[r][col] != 0:
                f = m[r][col]
                m[r] = [v - f * w for v, w in zip(m[r], m[col])]
    return [m[i][n] for i in range(n)]


def fit(p, live):
    """live: list of (x, y, count). Returns (n, beta, residual variance) or n only if rank deficient."""
    n = sum(c for _, _, c in live)
    xtx = [[sum(F(x[i]) * F(x[j]) * c for x, _, c in live) for j in range(p)] for i in range(p)]
    xty = [sum(F(x[i]) * F(y) * c for x, y, c in live) for i in range(p)]
    beta = solve(xtx, xty)
    if beta is None:
        return {"n": n, "beta": None, "residual_variance": None}
    if n <= p:
        return {"n": n, "beta": [float(b) for b in beta], "residual_variance": None}
    sse = sum(c * (F(y) - sum(b * F(v) for b, v in zip(beta, x))) ** 2 for x, y, c in live)
    return {"n": n, "beta": [float(b) for b in beta], "residual_variance": float(sse / (n - p))}


def scenario(name, p, rows, ops, beta_tolerance=1e-8, variance_tolerance=1e-6, check_fresh=True):
    state = {}
    checkpoints = []
    for op in ops:
        kind = op[0]
        if kind in ("append", "window"):
            for rid in op[1]:
                state[rid] = {"x": rows[rid][0], "y": rows[rid][1], "count": 1, "masked": False}
        elif kind == "evict":
            del state[op[1]]
        elif kind == "count":
            state[op[1]]["count"] = op[2]
        elif kind == "mask":
            state[op[1]]["masked"] = op[2]
        live = [(s["x"], s["y"], s["count"]) for s in state.values() if not s["masked"] and s["count"] > 0]
        checkpoints.append(fit(p, live))
    return {
        "name": name,
        "ncols": p,
        "rows": [{"x": x, "y": y} for x, y in rows],
        "ops": [list(op) for op in ops],
        "expected": checkpoints,
        "beta_tolerance": beta_tolerance,
        "variance_tolerance": variance_tolerance,
        "check_fresh": check_fresh,
    }


def lcg(seed):
    s = seed
    while True:
        s = (s * 6364136223846793005 + 1442695040888963407) % 2**64
        yield (s >> 33) % 2001 / 1000 - 1  # in [-1, 1]


def make_year_rows(count, seed, base_year, response_offset, slope):
    g = lcg(seed)
    rows = []
    for t in range(count):
        year = base_year + t
        rows.append(([1.0, float(year)], response_offset + slope * (year - base_year) + next(g)))
    return rows


def make_scaled_rows(count, seed, scales, normalized=True):
    """normalized: each column's effect is (k + 1) per unit of its own scale, so the response
    stays of order one; otherwise it is (k + 1) per unit of the column, so the scaled column
    with 1e6 carries a signal 1e13 times the noise variance."""
    g = lcg(seed)
    rows = []
    for _ in range(count):
        us = [next(g) for _ in scales]
        xs = [u * s for u, s in zip(us, scales)]
        signal = sum((k + 1) * u for k, u in enumerate(us)) if normalized else sum((k + 1) * x for k, x in enumerate(xs))
        rows.append(([1.0] + xs, 0.5 + signal + next(g)))
    return rows


ops_common = lambda n: (
    [("append", [0]), ("append", [1]), ("append", [2]), ("window", list(range(3, n - 2)))]
    + [("window", [n - 2, n - 1]), ("evict", 2), ("count", 4, 3), ("mask", 5, True), ("count", 0, 0), ("mask", 5, False), ("evict", 7)]
)

SCENARIOS = [
    scenario("year_regressor_2020_response_1e8", 2, make_year_rows(14, 1, 2020, 1e8, 3.0), ops_common(14)),
    scenario("year_regressor_1e6_response_1e9", 2, make_year_rows(14, 4, 1_000_000, 1e9, -0.5), ops_common(14), beta_tolerance=1e-6, variance_tolerance=1e-6, check_fresh=False),
    # response ulp (1.2e-4) is as large as the unit-scale noise: the one-pass centered scatter
    # keeps about four digits of the slope, which the tolerance records
    scenario("year_regressor_1e6_response_1e12", 2, make_year_rows(14, 2, 1_000_000, 1e12, -0.5), ops_common(14), beta_tolerance=2e-3, variance_tolerance=1e-3, check_fresh=False),
    scenario("three_columns_over_twelve_decades", 4, make_scaled_rows(14, 3, [1e-6, 1.0, 1e6]), ops_common(14)),
    # R^2 = 1 - 1e-13: the residual sum of squares from summary statistics loses about thirteen
    # of sixteen digits, so only the coefficients and the sample size are held to a tight bound
    scenario("three_columns_near_perfect_fit", 4, make_scaled_rows(14, 5, [1e-6, 1.0, 1e6], normalized=False), ops_common(14), beta_tolerance=1e-6, variance_tolerance=2e-2, check_fresh=False),
]
print(json.dumps({"scenarios": SCENARIOS}))
