# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath"]
# ///
"""Mechanism-change detection on a linear-Gaussian chain: residuals, statistics and the BH family.

Regenerate with:  uv run --script mechanism_change_detection_reference.py > mechanism_change_detection_reference.json

The graph is x -> y and x -> z.  For a target with the parent x the baseline residual of row r is
the held-out residual of the least-squares line fitted on the baseline rows of the other folds
(fold = row mod k, k = 5 for at least 30 baseline rows), and the comparison residual of a row is
its residual from the line fitted on all baseline rows.  Exact rational least squares
(fractions.Fraction); statistics at 50 digits (mpmath) from the definitions:
  mean_diff       |mean a - mean b| and the two-sided Welch t p-value (Satterthwaite df);
  likelihood      n ln v0 - n1 ln v1 - n2 ln v2 with MLE variances (denominator n), p = exp(-stat/2)
                  (chi-square with 2 df) when both segments have at least 30 rows;
  change_point    the same statistic on the concatenated series split at the baseline length;
  kernel          the biased RBF MMD^2 V-statistic exp(-(x-y)^2 / (2 h^2)) with h the median of the
                  pooled pairwise absolute differences (element n(n-1)/4 of the sorted list, lower
                  index convention floor(len/2)).
Benjamini-Hochberg adjusted p-values over the two targets are formed from the reference raw
p-values (mean_diff).  Repository-authored (MIT OR Apache-2.0); mpmath is BSD-3-Clause.
"""
import json
from fractions import Fraction as F

from mpmath import mp, mpf, log, exp, sqrt, betainc

mp.dps = 50
_state = 0x452821E638D01377


def u01():
    global _state
    _state = (_state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
    return (_state >> 11) / float(1 << 53)


def line(xs, ys):
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    return (sy - slope * sx) / n, slope


def residual_sets(x, y, n_base, folds):
    xs = [F(v) for v in x]
    ys = [F(v) for v in y]
    base = [None] * n_base
    for f in range(folds):
        train = [r for r in range(n_base) if r % folds != f]
        a, b = line([xs[r] for r in train], [ys[r] for r in train])
        for r in range(n_base):
            if r % folds == f:
                base[r] = ys[r] - (a + b * xs[r])
    a, b = line(xs[:n_base], ys[:n_base])
    comp = [ys[r] - (a + b * xs[r]) for r in range(n_base, len(x))]
    return base, comp


def mean(v):
    return sum(v) / len(v)


def welch(a, b):
    ma, mb = mean(a), mean(b)
    va = sum((v - ma) ** 2 for v in a) / (len(a) - 1)
    vb = sum((v - mb) ** 2 for v in b) / (len(b) - 1)
    qa, qb = va / len(a), vb / len(b)
    t = abs(ma - mb) / sqrt(mpf(qa.numerator) / qa.denominator + mpf(qb.numerator) / qb.denominator)
    df = (qa + qb) ** 2 / (qa * qa / (len(a) - 1) + qb * qb / (len(b) - 1))
    dfm = mpf(df.numerator) / df.denominator
    p = betainc(dfm / 2, mpf(1) / 2, 0, dfm / (dfm + t * t), regularized=True)
    return float(abs(ma - mb)), float(p)


def lr_stat(a, b):
    n1, n2 = len(a), len(b)
    n = n1 + n2
    allv = a + b
    m0 = mean(allv)
    v0 = sum((v - m0) ** 2 for v in allv) / n
    v1 = sum((v - mean(a)) ** 2 for v in a) / n1
    v2 = sum((v - mean(b)) ** 2 for v in b) / n2
    L = lambda q: log(mpf(q.numerator) / q.denominator)
    return n * L(v0) - n1 * L(v1) - n2 * L(v2)


def mmd2(a, b):
    pooled = [mpf(v.numerator) / v.denominator for v in a + b]
    diffs = sorted(abs(pooled[i] - pooled[j]) for i in range(len(pooled)) for j in range(i + 1, len(pooled)))
    h = diffs[len(diffs) // 2]
    k = lambda x, y: exp(-((x - y) / h) ** 2 / 2)
    A, B = pooled[: len(a)], pooled[len(a):]
    kxx = sum(k(x, y) for x in A for y in A) / len(A) ** 2
    kyy = sum(k(x, y) for x in B for y in B) / len(B) ** 2
    kxy = sum(k(x, y) for x in A for y in B) / (len(A) * len(B))
    return float(kxx + kyy - 2 * kxy), float(h)


def bh(ps):
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj = [None] * m
    running = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        running = min(running, ps[i] * m / rank)
        adj[i] = running
    return adj


def build(n_base, n_comp, shift_y, shift_z):
    n = n_base + n_comp
    x = [4.0 * (u01() - 0.5) for _ in range(n)]
    noise = lambda: u01() - 0.5
    y = [1.0 + 2.0 * x[r] + noise() + (shift_y if r >= n_base else 0.0) for r in range(n)]
    z = [-0.5 + 0.7 * x[r] + noise() + (shift_z if r >= n_base else 0.0) for r in range(n)]
    return x, y, z


def case(name, n_base, n_comp, shift_y, shift_z):
    x, y, z = build(n_base, n_comp, shift_y, shift_z)
    folds = 5 if n_base >= 30 else 3
    out = {"name": name, "n_base": n_base, "n_comp": n_comp, "x": x, "y": y, "z": z, "targets": {}}
    raw = []
    for key, series in (("y", y), ("z", z)):
        base, comp = residual_sets(x, series, n_base, folds)
        stat_md, p_md = welch(base, comp)
        lr = lr_stat(base, comp)
        mm, h = mmd2(base, comp)
        raw.append(p_md)
        out["targets"][key] = {
            "mean_diff": {"statistic": stat_md, "p_value": p_md},
            "likelihood_ratio": {"statistic": float(lr), "p_value": float(exp(-lr / 2)) if min(n_base, n_comp) >= 30 else None},
            "change_point": {"statistic": float(lr), "p_value": float(exp(-lr / 2)) if min(n_base, n_comp) >= 30 else None},
            "kernel": {"statistic": mm, "bandwidth": h},
        }
    out["mean_diff_adjusted"] = bh(raw)
    return out


cases = [
    case("shifted_intercept", 40, 40, 0.8, 0.0),
    case("both_unchanged", 40, 40, 0.0, 0.0),
    case("small_population", 24, 14, 0.9, 0.0),
]
print(json.dumps({"cases": cases}, indent=1))
