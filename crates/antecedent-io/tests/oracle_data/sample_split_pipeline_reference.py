# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath"]
# ///
"""End-to-end sample-split effect on frozen data: the split by definition, then exact least squares.

Regenerate with:  uv run --script sample_split_pipeline_reference.py > sample_split_pipeline_reference.json

The seeded split is reproduced from its definition: Fisher-Yates over the eligible rows driven by
splitmix64 from the seed (state += 0x9E3779B97F4A7C15, then the two multiply-xorshift rounds), the
first round(m * fraction) positions of the shuffled order go to discovery, the rest are analysis
rows in their original order.  Discovery is not reproduced here; the data come from structural
models in which the treatment's parents form a collider, so the equivalence class orients every
edge at the treatment and the expected adjustment set is the set of true parents.  For that
adjustment set the script solves y ~ 1 + x + parents over the rationals on the analysis rows (the
logarithm of every column first for a log-linear study), and evaluates the Student quantile at 60
digits; the log-scale coefficient is mapped to a mean ratio exp(shift * b), the interval to the
ordered images of its endpoints and the standard error to ratio * |shift| * se.
Repository-authored (MIT OR Apache-2.0).
"""
import json
import math
import random
from fractions import Fraction as F

from mpmath import mp, mpf, betainc, findroot, sqrt as msqrt

mp.dps = 60
M64 = (1 << 64) - 1


def splitmix(state):
    state = (state + 0x9E3779B97F4A7C15) & M64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return state, z ^ (z >> 31)


def split(m, seed, fraction):
    order = list(range(m))
    state = seed
    for i in range(m - 1, 0, -1):
        state, r = splitmix(state)
        j = r % (i + 1)
        order[i], order[j] = order[j], order[i]
    take = int(math.floor(m * fraction + 0.5))
    take = min(max(take, 1), m - 1)
    d = sorted(order[:take])
    ds = set(d)
    return d, [r for r in range(m) if r not in ds]


def solve(A, b):
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        piv = next(r for r in range(c, n) if M[r][c] != 0)
        M[c], M[piv] = M[piv], M[c]
        d = M[c][c]
        M[c] = [v / d for v in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [v - f * w for v, w in zip(M[r], M[c])]
    return [M[i][n] for i in range(n)]


def t_cdf(t, df):
    t, df = mpf(t), mpf(df)
    tail = betainc(df / 2, mpf(1) / 2, 0, df / (df + t * t), regularized=True) / 2
    return 1 - tail if t > 0 else tail


def t_ppf(p, df):
    return findroot(lambda t: t_cdf(t, df) - mpf(p), mpf(2), tol=mpf(10) ** -50, maxsteps=200)


def ols(y, x, zs, level=0.95):
    n = len(y)
    cols = [[F(1)] * n, [F(v) for v in x]] + [[F(v) for v in z] for z in zs]
    k = len(cols)
    G = [[sum(a * b for a, b in zip(cols[i], cols[j])) for j in range(k)] for i in range(k)]
    mom = [sum(a * F(b) for a, b in zip(cols[i], y)) for i in range(k)]
    beta = solve(G, mom)
    e1 = solve(G, [F(int(i == 1)) for i in range(k)])
    rss = sum((F(y[r]) - sum(beta[c] * cols[c][r] for c in range(k))) ** 2 for r in range(n))
    df = n - k
    var = rss / df * e1[1]
    se = msqrt(mpf(var.numerator) / mpf(var.denominator))
    pt = mpf(beta[1].numerator) / mpf(beta[1].denominator)
    q = t_ppf(1 - (1 - mpf(level)) / 2, df)
    return {"point": float(pt), "se": float(se), "df": df, "lower": float(pt - q * se), "upper": float(pt + q * se), "rows": n}


def reported_ratio(b, shift):
    r = math.exp(shift * b["point"])
    lo, hi = math.exp(shift * b["lower"]), math.exp(shift * b["upper"])
    return {"point": r, "lower": min(lo, hi), "upper": max(lo, hi), "se": r * abs(shift) * b["se"]}


def sem(rng, n, spec):
    cols = [[] for _ in spec]
    for _ in range(n):
        row = []
        for coefs, noise in spec:
            row.append(sum(c * row[i] for i, c in coefs) + noise * rng.gauss(0, 1))
        for j, v in enumerate(row):
            cols[j].append(v)
    return cols


def main():
    out = {}
    # five variables: v0, v1 -> v2 (treatment) -> v3 (outcome) <- v0 ; v3 -> v4
    rng = random.Random(77)
    spec5 = [([], 1.0), ([], 1.0), ([(0, 0.9), (1, 0.9)], 0.6), ([(2, 0.8), (0, 0.5)], 0.6), ([(3, 0.7)], 0.6)]
    cols = sem(rng, 800, spec5)
    d, a = split(800, 31, 0.5)
    ana = lambda c: [c[r] for r in a]
    ref = ols(ana(cols[3]), ana(cols[2]), [ana(cols[0]), ana(cols[1])])
    out["effect"] = {"columns": cols, "treatment": 2, "outcome": 3, "split_seed": 31, "discovery_seed": 5,
                     "expected_adjustment": [0, 1], "analysis_rows": len(a), "discovery_rows": len(d), "reference": ref,
                     "analysis_row_ids": a}
    # ten variables: v0, v1 -> v2 (treatment) -> v3 (outcome); v3 -> v4 -> v9; v5 - v6 and v7 - v8 unrelated
    rng = random.Random(78)
    spec10 = [([], 1.0), ([], 1.0), ([(0, 0.7), (1, 0.7)], 0.6), ([(2, 0.8)], 0.6), ([(3, 0.7)], 0.6),
              ([], 1.0), ([(5, 0.8)], 0.6), ([], 1.0), ([(7, 0.6)], 0.8), ([(4, 0.5)], 0.7)]
    lin = sem(rng, 1000, spec10)
    log_cols = [[math.exp(0.3 * v) for v in c] for c in lin]
    studies = []
    for version, fraction in [(1, 0.5), (2, 0.75)]:
        d, a = split(1000, 19, fraction)
        pick = lambda c: [c[r] for r in a]
        b_lin = ols(pick(lin[3]), pick(lin[2]), [pick(lin[0]), pick(lin[1])])
        logs = [[math.log(v) for v in c] for c in log_cols]
        b_log = ols(pick(logs[3]), pick(logs[2]), [pick(logs[0]), pick(logs[1])])
        studies.append({"version": version, "discovery_fraction": fraction, "discovery_rows": len(d), "analysis_rows": len(a),
                        "linear": b_lin,
                        "log_linear": {"coefficient": b_log, "shifts": {str(s): reported_ratio(b_log, s) for s in (0.5, -0.75)}}})
    out["parent_set"] = {"linear_columns": lin, "log_columns": log_cols, "treatment": 2, "outcome": 3, "split_seed": 19,
                         "discovery_seed": 6, "max_degree": 4, "expected_parents": [0, 1], "studies": studies}
    print(json.dumps(out))


main()
