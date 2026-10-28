# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Exact reference for the leave-one-out adjustment-covariate sensitivity check.

  uv run --script graph_refute_reference.py > graph_refute_reference.json

Written from the stated definition, with no product code. On the rows complete for the full
adjustment set (all of T, Y and every Z), the full fit gives ate0 (coefficient of T in the exact
least-squares fit of Y on (1, T, Z...)); dropping covariate j gives ate_j on the SAME rows;
  delta_j = |ate_j - ate0| / max(|ate0|, 1e-3 * SD(Y)/SD(T))      (SDs: n-1, complete-case rows)
  worst   = the largest delta_j, the later covariate winning an exact tie;
  passed  = worst < threshold;  refuted = ate of the worst drop;  replicates = |Z|.
Fractions throughout; the SD scale uses 60-digit sqrt only for the floor.

Repository-authored (MIT OR Apache-2.0).
"""
import json
from fractions import Fraction as F

import mpmath as mp

mp.mp.dps = 60


def lcg(seed):
    state = seed
    while True:
        state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        yield F(state >> 40, 1 << 24)


def d4(x):
    return F(round(x * 10**4), 10**4)


def d64(x):
    return F(round(x * 64), 64)


def case_rows(name, n, seed, nz, kind, missing=()):
    g = lcg(seed)
    t, y, zs = [], [], [[] for _ in range(nz)]
    for _ in range(n):
        u = [next(g) for _ in range(nz + 2)]
        dec = d64 if kind == "exact_null" else d4
        z = [dec((u[k] - F(1, 2)) * 4) for k in range(nz)]
        if kind == "confounded":
            ti = dec(z[0] * 2 + (u[nz] - F(1, 2)))
            yi = dec(3 * z[0] + F(1, 2) * ti + (u[nz + 1] - F(1, 2)) / 2)
        elif kind == "exact_null":
            ti = dec(z[0] / 2 + (u[nz] - F(1, 2)) * 2)
            yi = 2 * z[0] + 3 * z[1]
        else:
            ti = dec((u[nz] - F(1, 2)) * 3 + z[0] / 4)
            yi = dec(1 + F(3, 5) * ti + z[0] / 2 - z[-1] / 3 + (u[nz + 1] - F(1, 2)))
        t.append(ti)
        y.append(yi)
        for k in range(nz):
            zs[k].append(z[k])
    valid = [i not in missing for i in range(n)]
    return {"name": name, "t": [float(v) for v in t], "y": [float(v) for v in y],
            "z": [[float(v) for v in c] for c in zs], "z_invalid_rows": sorted(missing)}


CASES = [
    (case_rows("balanced", 40, 31, 3, "mixed"), [0.5, 0.6, 0.05]),
    (case_rows("confounded", 40, 32, 2, "confounded"), [0.5, 5.0]),
    (case_rows("exact_null_effect", 32, 33, 2, "exact_null"), [0.5, 1e6]),
    (case_rows("missing_in_last_covariate", 36, 34, 2, "mixed", missing=(3, 7, 11, 20)), [0.5, 0.01]),
    (case_rows("single_covariate", 30, 35, 1, "mixed"), [0.5, 0.01]),
]


def solve(a, b):
    n = len(a)
    m = [row[:] + [bb] for row, bb in zip(a, b)]
    for c in range(n):
        piv = next((r for r in range(c, n) if m[r][c] != 0), None)
        if piv is None:
            return None
        m[c], m[piv] = m[piv], m[c]
        inv = 1 / m[c][c]
        m[c] = [v * inv for v in m[c]]
        for r in range(n):
            if r != c and m[r][c] != 0:
                f = m[r][c]
                m[r] = [v - f * w for v, w in zip(m[r], m[c])]
    return [m[r][n] for r in range(n)]


def ate(t, y, zs, rows):
    cols = [[F(1)] * len(rows), [F(t[i]) for i in rows]] + [[F(z[i]) for i in rows] for z in zs]
    yy = [F(y[i]) for i in rows]
    p = len(cols)
    xtx = [[sum(a * b for a, b in zip(cols[i], cols[j])) for j in range(p)] for i in range(p)]
    xty = [sum(a * b for a, b in zip(cols[i], yy)) for i in range(p)]
    return solve(xtx, xty)[1]


def sd(v):
    m = sum(v) / len(v)
    var = sum((a - m) ** 2 for a in v) / (len(v) - 1)
    return mp.sqrt(mp.mpf(var.numerator) / var.denominator)


out = []
for case, thresholds in CASES:
    t, y, zs = case["t"], case["y"], case["z"]
    n = len(t)
    bad = set(case["z_invalid_rows"])
    rows = [i for i in range(n) if i not in bad]
    ate0 = ate(t, y, zs, rows)
    floor = mp.mpf("1e-3") * sd([F(y[i]) for i in rows]) / sd([F(t[i]) for i in rows])
    drops = []
    for j in range(len(zs)):
        a = ate(t, y, [z for k, z in enumerate(zs) if k != j], rows)
        a0 = mp.mpf(ate0.numerator) / ate0.denominator
        aj = mp.mpf(a.numerator) / a.denominator
        delta = abs(aj - a0) / max(abs(a0), floor)
        drops.append({"ate": float(aj), "delta": float(delta)})
    worst = max(range(len(drops)), key=lambda j: (drops[j]["delta"], j))
    runs = []
    for thr in thresholds:
        runs.append({"threshold": thr, "worst": worst, "comparison": drops[worst]["delta"],
                     "refuted": drops[worst]["ate"], "passed": bool(drops[worst]["delta"] < thr)})
    out.append({"case": case, "ate0": float(ate0), "drops": drops, "runs": runs})
print(json.dumps(out))
