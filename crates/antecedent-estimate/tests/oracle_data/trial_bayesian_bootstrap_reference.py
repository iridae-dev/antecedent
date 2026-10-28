# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Exact reference for the Bayesian-bootstrap trial-to-target draw.

Regenerate with:  uv run --script trial_bayesian_bootstrap_reference.py > trial_bayesian_bootstrap_reference.json

For trial rows with score psi_i = sign_i (1 - s_i) / s_i * y_i / arm_i (arm_i = e_i for treated rows
and 1 - e_i for controls), m trial rows and n_t target rows, a draw is
  D = (m / n_t) * sum_i pi_i psi_i,    pi ~ Dirichlet(1, ..., 1).
With a_i = m psi_i / n_t distinct, the law of sum pi_i a_i is the simplex spline of the a_i: its
distribution function is
  F(x) = 1 - sum_i (a_i - x)_+^(m-1) / prod_{j != i} (a_i - a_j)
(the Hermite-Genocchi form), its mean is mean(a) and its variance is sum (a_i - mean a)^2 / (m (m+1)).
In the block form every trial row of a contiguous block of `block` table rows shares one
Exp(1) weight; when every block holds the same number c of trial rows the normalized weights are
Dirichlet over the blocks and a_b = (m / (c n_t)) S_b with S_b the block sum of scores.
When the blocks hold unequal numbers c_b of trial rows the draw is the ratio
(m / n_t) sum_b g_b S_b / sum_b g_b c_b of independent Exp(1) weights g_b; with y = x n_t / m,
P(draw <= x) = 1 - P(sum_b g_b (S_b - y c_b) > 0) and, for distinct nonzero a_b = S_b - y c_b,
P(sum_b a_b g_b > 0) = sum_{b: a_b > 0} prod_{k != b} a_b / (a_b - a_k).
The formulas are cross-checked against 400,000 exponential-gap simulations inside this script.
Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import json
import random

import mpmath as mp

mp.mp.dps = 40


def cdf(a, x):
    m = len(a)
    total = mp.mpf(0)
    for i, ai in enumerate(a):
        if ai > x:
            denom = mp.mpf(1)
            for j, aj in enumerate(a):
                if j != i:
                    denom *= ai - aj
            total += (ai - x) ** (m - 1) / denom
    return 1 - total


def moments(a):
    m = len(a)
    mean = sum(a) / m
    var = sum((v - mean) ** 2 for v in a) / (m * (m + 1))
    return mean, var


def check(a, x):
    rng = random.Random(5)
    m = len(a)
    hits = 0
    n = 200000
    af = [float(v) for v in a]
    for _ in range(n):
        g = [rng.expovariate(1.0) for _ in range(m)]
        t = sum(g)
        if sum(gi * ai for gi, ai in zip(g, af)) / t <= float(x):
            hits += 1
    p = float(cdf(a, x))
    assert abs(hits / n - p) < 5 * (p * (1 - p) / n) ** 0.5 + 1e-9, (hits / n, p)


def ratio_cdf(sums, counts, m, nt, x):
    y = mp.mpf(x) * nt / m
    a = [sb - y * cb for sb, cb in zip(sums, counts)]
    tail = mp.mpf(0)
    for i, ai in enumerate(a):
        if ai > 0:
            term = mp.mpf(1)
            for k, ak in enumerate(a):
                if k != i:
                    term *= ai / (ai - ak)
            tail += term
    return 1 - tail


def check_ratio(sums, counts, m, nt, x):
    rng = random.Random(9)
    n = 200000
    hits = 0
    for _ in range(n):
        g = [rng.expovariate(1.0) for _ in sums]
        draw = (m / nt) * sum(gi * float(s) for gi, s in zip(g, sums)) / sum(gi * c for gi, c in zip(g, counts))
        if draw <= float(x):
            hits += 1
    p = float(ratio_cdf(sums, counts, m, nt, x))
    assert abs(hits / n - p) < 5 * (p * (1 - p) / n) ** 0.5 + 1e-9, (hits / n, p)


def build_ratio(name, outcome, treatment, trial, sel, prob, block, sums, counts, m, nt):
    pts = [(m / nt) * min(float(s) / c for s, c in zip(sums, counts)),
           (m / nt) * max(float(s) / c for s, c in zip(sums, counts))]
    lo, hi = mp.mpf(pts[0]), mp.mpf(pts[1])
    xs = [lo + (hi - lo) * mp.mpf(k) / 10 for k in range(1, 10)]
    for x in (xs[1], xs[4], xs[7]):
        check_ratio(sums, counts, m, nt, x)
    return {
        "name": name, "outcome": outcome, "treatment": treatment, "trial": trial,
        "selection": sel, "propensity": prob, "block": block,
        "mean": None, "variance": None,
        "min": mp.nstr(lo, 30), "max": mp.nstr(hi, 30),
        "cdf": [{"x": mp.nstr(x, 30), "p": mp.nstr(ratio_cdf(sums, counts, m, nt, x), 30)} for x in xs],
    }


def build(name, outcome, treatment, trial, sel, prob, block, a_values):
    mean, var = moments(a_values)
    lo, hi = min(a_values), max(a_values)
    xs = [lo + (hi - lo) * mp.mpf(k) / 10 for k in range(1, 10)]
    for x in (xs[1], xs[4], xs[7]):
        check(a_values, x)
    return {
        "name": name, "outcome": outcome, "treatment": treatment, "trial": trial,
        "selection": sel, "propensity": prob, "block": block,
        "mean": mp.nstr(mean, 30), "variance": mp.nstr(var, 30),
        "min": mp.nstr(lo, 30), "max": mp.nstr(hi, 30),
        "cdf": [{"x": mp.nstr(x, 30), "p": mp.nstr(cdf(a_values, x), 30)} for x in xs],
    }


def scores(outcome, treatment, trial, sel, prob):
    out = []
    for y, t, s_, s, e in zip(outcome, treatment, trial, sel, prob):
        if not s_:
            continue
        arm = e if t else 1 - e
        sign = 1 if t else -1
        out.append(sign * (1 - mp.mpf(s)) / mp.mpf(s) * mp.mpf(y) / mp.mpf(arm))
    return out


def main():
    # Free draw over eight trial rows: four treated, four control, six target rows.
    outcome = [2.1, 0.4, 3.3, -0.6, 1.2, 2.7, 0.9, -1.4, 0, 0, 0, 0, 0, 0]
    treatment = [True, False, True, False, True, False, True, False] + [False] * 6
    trial = [True] * 8 + [False] * 6
    sel = [0.35, 0.5, 0.42, 0.3, 0.6, 0.45, 0.38, 0.55] + [0.5] * 6
    prob = [0.4, 0.55, 0.6, 0.45, 0.5, 0.35, 0.65, 0.5] + [0.5] * 6
    psi = scores(outcome, treatment, trial, sel, prob)
    m, nt = len(psi), 6
    a = [mp.mpf(m) * p / nt for p in psi]
    free = build("free", outcome, treatment, trial, sel, prob, None, a)

    # Block form: four blocks of three consecutive table rows, two trial rows and one target row
    # each, so the normalised block weights are Dirichlet over four blocks.
    outcome_b = [1.5, -0.5, 0, 2.4, 0.2, 0, 3.1, 1.0, 0, -1.2, 0.7, 0]
    treatment_b = [True, False, False, True, False, False, True, False, False, False, True, False]
    trial_b = [True, True, False] * 4
    sel_b = [0.4, 0.5, 0.5, 0.35, 0.45, 0.5, 0.55, 0.3, 0.5, 0.6, 0.4, 0.5]
    prob_b = [0.5, 0.45, 0.5, 0.6, 0.4, 0.5, 0.35, 0.55, 0.5, 0.5, 0.65, 0.5]
    psi_b = scores(outcome_b, treatment_b, trial_b, sel_b, prob_b)
    m_b, c, nt_b = len(psi_b), 2, 4
    sums = [psi_b[2 * k] + psi_b[2 * k + 1] for k in range(4)]
    a_b = [mp.mpf(m_b) / (c * nt_b) * s for s in sums]
    blocked = build("blocked", outcome_b, treatment_b, trial_b, sel_b, prob_b, 3, a_b)

    # Unequal trial counts per block (1, 2, 3, 1 trial rows; 2, 1, 0, 2 target rows).
    outcome_u = [1.5, 0, 0, 2.4, -0.5, 0, 0.2, 3.1, 1.0, 0.9, 0, 0]
    treatment_u = [True, False, False, True, False, False, False, True, False, True, False, False]
    trial_u = [True, False, False, True, True, False, True, True, True, True, False, False]
    sel_u = [0.4, 0.5, 0.5, 0.35, 0.45, 0.5, 0.55, 0.3, 0.6, 0.4, 0.5, 0.5]
    prob_u = [0.5, 0.5, 0.5, 0.6, 0.4, 0.5, 0.35, 0.55, 0.45, 0.65, 0.5, 0.5]
    psi_u = scores(outcome_u, treatment_u, trial_u, sel_u, prob_u)
    counts_u = [1, 2, 3, 1]
    sums_u, at = [], 0
    for c in counts_u:
        sums_u.append(sum(psi_u[at:at + c]))
        at += c
    unequal = build_ratio("blocked_unequal", outcome_u, treatment_u, trial_u, sel_u, prob_u, 3,
                          sums_u, counts_u, len(psi_u), 5)
    print(json.dumps([free, blocked, unequal]))


main()
