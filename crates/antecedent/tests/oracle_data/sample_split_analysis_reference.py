# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath"]
# ///
"""Row split, weight carrying, Kish size, weighted HC3 effect and Student interval, exactly.

Regenerate with:  uv run --script sample_split_analysis_reference.py > sample_split_analysis_reference.json

Table: 70 rows of three searched columns (some cells missing, encoded null), a fourth column that is
not searched, an analysis mask that drops some rows and sampling weights with exact zeros and a
spread of two orders of magnitude.  Eligible rows are those the mask keeps and that are complete in
every searched column; a seeded split shuffles the eligible rows (Fisher-Yates driven by splitmix64
from the seed) and gives round(m * fraction) of them to discovery; an explicit rule names them.
Each part carries its rows' weights unnormalized and the Kish size (sum w)^2 / sum w^2.  The effect
of the treatment on the outcome adjusting for one covariate is the weighted least-squares
coefficient over the analysis rows of positive weight, weights normalized to mean one, with the
HC3 sandwich bread^-1 (sum psi psi') bread^-1, psi_i = w_i e_i x_i / (1 - h_i),
h_i = w_i x_i' bread^-1 x_i and bread = X' W X; the interval is effect +/- t * se with the Student
quantile at (Kish size of the rows used) - 3 degrees of freedom (n - 3 unweighted).  Everything is
Fraction arithmetic except the square root and the quantile (60 digits).
Repository-authored (MIT OR Apache-2.0).
"""
import json
import math
import random
from fractions import Fraction as F

from mpmath import mp, mpf, betainc, findroot, sqrt as msqrt

mp.dps = 60
M64 = (1 << 64) - 1
LEVELS = [0.9, 0.95, 0.99]


def splitmix(state):
    state = (state + 0x9E3779B97F4A7C15) & M64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return state, z ^ (z >> 31)


def seeded(eligible, seed, fraction):
    m = len(eligible)
    order = list(range(m))
    state = seed
    for i in range(m - 1, 0, -1):
        state, r = splitmix(state)
        j = r % (i + 1)
        order[i], order[j] = order[j], order[i]
    take = min(max(int(math.floor(m * fraction + 0.5)), 1), m - 1)
    d = sorted(eligible[p] for p in order[:take])
    ds = set(d)
    return d, [r for r in eligible if r not in ds]


def kish(ws):
    s = sum(F(w) for w in ws)
    q = sum(F(w) ** 2 for w in ws)
    return float(s * s / q)


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


def inv(A):
    n = len(A)
    cols = [solve(A, [F(int(i == j)) for i in range(n)]) for j in range(n)]
    return [[cols[j][i] for j in range(n)] for i in range(n)]


def t_cdf(t, df):
    t, df = mpf(t), mpf(df)
    tail = betainc(df / 2, mpf(1) / 2, 0, df / (df + t * t), regularized=True) / 2
    return 1 - tail if t > 0 else tail


def t_ppf(p, df):
    return findroot(lambda t: t_cdf(t, df) - mpf(p), mpf(2), tol=mpf(10) ** -50, maxsteps=200)


def hc3_effect(x, y, z, w):
    """Weighted least squares of y on 1, x, z with HC3; rows with zero weight are dropped."""
    keep = [i for i in range(len(y)) if w[i] > 0]
    n = len(keep)
    wsum = sum(F(w[i]) for i in keep)
    scale = F(n) / wsum
    rows = [[F(1), F(x[i]), F(z[i])] for i in keep]
    ww = [F(w[i]) * scale for i in keep]
    yy = [F(y[i]) for i in keep]
    k = 3
    A = [[sum(ww[r] * rows[r][a] * rows[r][b] for r in range(n)) for b in range(k)] for a in range(k)]
    mom = [sum(ww[r] * rows[r][a] * yy[r] for r in range(n)) for a in range(k)]
    bread = inv(A)
    beta = [sum(bread[a][b] * mom[b] for b in range(k)) for a in range(k)]
    meat = [[F(0)] * k for _ in range(k)]
    for r in range(n):
        e = yy[r] - sum(beta[a] * rows[r][a] for a in range(k))
        h = ww[r] * sum(rows[r][a] * bread[a][b] * rows[r][b] for a in range(k) for b in range(k))
        psi = [ww[r] * e * rows[r][a] / (1 - h) for a in range(k)]
        for a in range(k):
            for b in range(k):
                meat[a][b] += psi[a] * psi[b]
    var = sum(bread[1][a] * meat[a][b] * bread[b][1] for a in range(k) for b in range(k))
    se = msqrt(mpf(var.numerator) / mpf(var.denominator))
    ess = kish([w[i] for i in keep])
    return beta[1], se, n, len(y) - n, ess, float(wsum)


def interval_rows(point, se, info, coefs=3):
    df = info - coefs
    out = []
    for lvl in LEVELS:
        q = t_ppf(1 - (1 - mpf(lvl)) / 2, df)
        pt = mpf(point.numerator) / mpf(point.denominator)
        out.append({"level": lvl, "lower": float(pt - q * se), "upper": float(pt + q * se), "df": df})
    return out


def main():
    rng = random.Random(2024)
    n = 70
    v0 = [rng.gauss(0, 1) for _ in range(n)]
    v1 = [rng.gauss(0, 1) for _ in range(n)]
    v2 = [0.8 * v0[i] + 0.5 * v1[i] + 0.7 * rng.gauss(0, 1) for i in range(n)]
    v3 = [rng.gauss(0, 1) for _ in range(n)]
    cols = [v0, v1, v2, v3]
    missing = [(3, 0), (11, 1), (40, 2), (55, 0)]  # searched cells made missing
    for r, c in missing:
        cols[c][r] = None
    unsearched_missing = (20, 3)
    cols[3][20] = None  # not searched: must travel with its row
    mask = [True] * n
    for r in (5, 17, 33, 60):
        mask[r] = False
    mask[11] = False  # masked and incomplete? 11 is incomplete: the mask wins
    weights = [math.exp(rng.gauss(0, 0.9)) * 0.1 for _ in range(n)]
    for r in (8, 9, 30, 31, 44, 66):
        weights[r] = 0.0
    eligible = [r for r in range(n) if mask[r] and all(cols[c][r] is not None for c in range(3))]
    masked_out = [r for r in range(n) if not mask[r]]
    incomplete = [r for r in range(n) if mask[r] and any(cols[c][r] is None for c in range(3))]
    d, a = seeded(eligible, 9, 0.5)
    explicit = eligible[::3]
    ed = sorted(explicit)
    ea = [r for r in eligible if r not in set(ed)]
    split = {
        "seed": 9, "fraction": 0.5, "eligible": eligible, "masked_out": masked_out, "incomplete": incomplete,
        "seeded": {"discovery": d, "analysis": a,
                   "discovery_ess": kish([weights[r] for r in d]), "analysis_ess": kish([weights[r] for r in a])},
        "explicit": {"discovery": ed, "analysis": ea,
                     "discovery_ess": kish([weights[r] for r in ed]), "analysis_ess": kish([weights[r] for r in ea])},
    }
    # weighted effect on the analysis rows of the seeded split
    pick = lambda c, rows: [cols[c][r] for r in rows]
    beta, se, used, zeros, ess, wsum = hc3_effect(pick(0, a), pick(2, a), pick(1, a), [weights[r] for r in a])
    weighted = {"point": float(beta), "se": float(se), "rows_used": used, "zero_weight_rows": zeros, "ess": ess,
                "weight_sum": wsum, "rows": len(a), "intervals": interval_rows(beta, se, ess)}
    # unweighted: the complete unmasked rows only, as their own table
    sub = eligible
    ucols = [[cols[c][r] for r in sub] for c in range(3)]
    ud, ua = seeded(list(range(len(sub))), 9, 0.5)
    ones = [1.0] * len(ua)
    ub, use, uused, _, _, _ = hc3_effect([ucols[0][r] for r in ua], [ucols[2][r] for r in ua], [ucols[1][r] for r in ua], ones)
    unweighted = {"columns": ucols, "analysis_ids": ua, "point": float(ub), "se": float(use), "rows": uused,
                  "intervals": interval_rows(ub, use, uused)}
    print(json.dumps({"columns": cols, "mask": mask, "weights": weights, "split": split, "weighted": weighted, "unweighted": unweighted}))


main()
