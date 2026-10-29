# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Exact reference for the weighted, clustered and serial trial-to-target IPW contrast.

Regenerate with:  uv run --script transport_design_reference.py > transport_design_reference.json

Everything is rational arithmetic on dyadic inputs (outcomes, probabilities and weights are
multiples of 1/1024, so they are exact binary64 values), so the point estimate and the variance
are exact fractions; the standard error is their square root to 40 digits.

For rows i with trial flag S_i, treatment A_i, outcome y_i, selection probability s_i, treatment
probability e_i and sampling weight w_i (rows of weight zero are deleted):
  psi_i = S_i * (+1 if A_i else -1) * (1 - s_i) / s_i * y_i / (e_i if A_i else 1 - e_i)
  T     = sum_i w_i (1 - S_i)                       (target weight)
  ipw   = sum_i w_i psi_i / T
  u_i   = w_i psi_i for a trial row, -ipw w_i for a target row   (linearized influence, sum 0)
  independent:  Var = sum u_i^2 / T^2
  clusters:     Var = G/(G-1) sum_g (sum_{i in g} u_i)^2 / T^2
  series (lag L): Var = (sum u_t^2 + 2 sum_{k=1}^{min(L, n-1)} (1 - k/(L+1)) sum_t u_t u_{t+k}) / T^2
Bayesian bootstrap: each group (a trial row, a cluster, or a block of consecutive table rows)
gets an independent Exp(1) weight E_g shared by its trial rows, and a draw is
  D = (W_trial / T) * sum_g E_g A_g / sum_g E_g B_g,  A_g = sum_{i in g} w_i psi_i, B_g = sum_{i in g} w_i,
so with k = W_trial / T and y = x / k,  P(D <= x) = 1 - sum_{g: a_g > 0} prod_{h != g} a_g / (a_g - a_h)
for a_g = A_g - y B_g (the survival function of a weighted sum of independent exponentials).
The distribution function is cross-checked against 200,000 simulated draws inside this script.
Repository-authored (MIT OR Apache-2.0).
"""
import json
import random
from decimal import Decimal, getcontext
from fractions import Fraction

getcontext().prec = 60


def dec(value: Fraction) -> str:
    return str(Decimal(value.numerator) / Decimal(value.denominator))[:48]


def sqrt_str(value: Fraction) -> str:
    return str(Decimal(value.numerator).sqrt() / Decimal(value.denominator).sqrt())[:48]


def fr(x) -> Fraction:
    return Fraction(x)


def dyadic(rng, lo, hi, den=8):
    return Fraction(rng.randint(lo * den, hi * den), den)


def make_rows(seed, n, n_trial_pattern, weights=None):
    rng = random.Random(seed)
    trial = [bool(p) for p in n_trial_pattern]
    treatment = [rng.random() < 0.5 for _ in range(n)]
    # Both arms among the trial rows.
    trial_rows = [i for i in range(n) if trial[i]]
    treatment[trial_rows[0]] = True
    treatment[trial_rows[1]] = False
    outcome = [dyadic(rng, -3, 5) for _ in range(n)]
    selection = [rng.choice([Fraction(1, 8), Fraction(1, 4), Fraction(1, 2), Fraction(3, 4)]) for _ in range(n)]
    propensity = [rng.choice([Fraction(1, 4), Fraction(1, 2), Fraction(3, 4)]) for _ in range(n)]
    return {
        "trial": trial, "treatment": treatment, "outcome": outcome,
        "selection": selection, "propensity": propensity,
        "weights": weights if weights is None else [fr(w) for w in weights],
    }


def psi(rows, i):
    if not rows["trial"][i]:
        return Fraction(0)
    a = rows["treatment"][i]
    arm = rows["propensity"][i] if a else 1 - rows["propensity"][i]
    sign = 1 if a else -1
    s = rows["selection"][i]
    return sign * (1 - s) / s * rows["outcome"][i] / arm


def active(rows):
    n = len(rows["trial"])
    w = rows["weights"] or [Fraction(1)] * n
    return [i for i in range(n) if w[i] > 0], w


def estimate(rows, dependence):
    act, w = active(rows)
    target = sum(w[i] for i in act if not rows["trial"][i])
    ipw = sum(w[i] * psi(rows, i) for i in act if rows["trial"][i]) / target
    u = {i: (w[i] * psi(rows, i) if rows["trial"][i] else -ipw * w[i]) for i in act}
    assert sum(u.values()) == 0
    kind = dependence["kind"]
    if kind == "independent":
        var = sum(v * v for v in u.values()) / target**2
    elif kind == "clusters":
        sums = {}
        for i in act:
            sums[dependence["labels"][i]] = sums.get(dependence["labels"][i], 0) + u[i]
        g = len(sums)
        var = Fraction(g, g - 1) * sum(v * v for v in sums.values()) / target**2
    else:
        # Deleted rows are inert inside the series: they stay in place with zero influence.
        n = len(rows["trial"])
        series = [u.get(i, Fraction(0)) for i in range(n)]
        lag = dependence["lag"]
        var = sum(v * v for v in series)
        for k in range(1, min(lag, n - 1) + 1):
            var += 2 * (1 - Fraction(k, lag + 1)) * sum(series[t] * series[t + k] for t in range(n - k))
        var = max(var, Fraction(0)) / target**2
    return ipw, var


def ratio_cdf(a_sums, b_sums, k, x):
    y = x / k
    a = [aa - y * bb for aa, bb in zip(a_sums, b_sums)]
    tail = Fraction(0)
    for i, ai in enumerate(a):
        if ai > 0:
            term = Fraction(1)
            for j, aj in enumerate(a):
                if j != i:
                    term *= ai / (ai - aj)
            tail += term
    return 1 - tail


def check_ratio(a_sums, b_sums, k, x, seed):
    rng = random.Random(seed)
    n = 200000
    hits = 0
    af, bf, kf, xf = [float(v) for v in a_sums], [float(v) for v in b_sums], float(k), float(x)
    for _ in range(n):
        e = [rng.expovariate(1.0) for _ in af]
        draw = kf * sum(ei * ai for ei, ai in zip(e, af)) / sum(ei * bi for ei, bi in zip(e, bf))
        if draw <= xf:
            hits += 1
    p = float(ratio_cdf(a_sums, b_sums, k, x))
    assert abs(hits / n - p) < 5 * (p * (1 - p) / n) ** 0.5 + 1e-9, (hits / n, p)


def bayes_case(name, rows, grouping, seed):
    """grouping: 'rows', ('clusters', labels) or ('blocks', length)."""
    act, w = active(rows)
    trial_rows = [i for i in act if rows["trial"][i]]
    target = sum(w[i] for i in act if not rows["trial"][i])
    w_trial = sum(w[i] for i in trial_rows)
    groups = {}
    for i in trial_rows:
        if grouping == "rows":
            key = i
        elif grouping[0] == "clusters":
            key = grouping[1][i]
        else:
            key = i // grouping[1]
        groups.setdefault(key, []).append(i)
    keys = sorted(groups)
    a_sums = [sum(w[i] * psi(rows, i) for i in groups[g]) for g in keys]
    b_sums = [sum(w[i] for i in groups[g]) for g in keys]
    k = w_trial / target
    ratios = [a / b for a, b in zip(a_sums, b_sums)]
    lo, hi = k * min(ratios), k * max(ratios)
    xs = []
    for j in range(1, 10):
        x = lo + (hi - lo) * Fraction(j, 10)
        # A tie in a_g makes the partial-fraction form singular; nudge x off it.
        y = x / k
        while len({aa - y * bb for aa, bb in zip(a_sums, b_sums)}) < len(keys):
            x += Fraction(1, 10**9)
            y = x / k
        xs.append(x)
    for j in (1, 4, 7):
        check_ratio(a_sums, b_sums, k, xs[j], seed + j)
    return {
        "name": name,
        "grouping": "rows" if grouping == "rows" else grouping[0],
        "labels": grouping[1] if grouping != "rows" and grouping[0] == "clusters" else None,
        "block": grouping[1] if grouping != "rows" and grouping[0] == "blocks" else None,
        "rows": rows_json(rows),
        "min": dec(lo), "max": dec(hi),
        "cdf": [{"x": dec(x), "p": dec(ratio_cdf(a_sums, b_sums, k, x))} for x in xs],
    }


def rows_json(rows):
    return {
        "trial": rows["trial"], "treatment": rows["treatment"],
        "outcome": [float(v) for v in rows["outcome"]],
        "selection": [float(v) for v in rows["selection"]],
        "propensity": [float(v) for v in rows["propensity"]],
        "weights": None if rows["weights"] is None else [float(v) for v in rows["weights"]],
    }


def estimate_case(name, rows, dependence):
    ipw, var = estimate(rows, dependence)
    return {
        "name": name, "rows": rows_json(rows), "dependence": dependence,
        "ipw": dec(ipw), "variance": dec(var), "se": sqrt_str(var),
    }


def main():
    cases = []
    pattern14 = [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0]
    w14 = [Fraction(v) for v in [1, 2, Fraction(1, 2), 4, Fraction(5, 2), 1, 2, 1, Fraction(1, 2), 4, 1, 2, Fraction(5, 2), 1]]
    cases.append(estimate_case("independent_unweighted", make_rows(1, 14, pattern14), {"kind": "independent"}))
    cases.append(estimate_case("independent_weighted", make_rows(2, 14, pattern14, w14), {"kind": "independent"}))
    extreme = [Fraction(1, 1024), 1, 1024, 4096, 1, 1, Fraction(1, 1024), 1024, 1, 4096, 1, 1024, 1, 1]
    cases.append(estimate_case("independent_extreme_weights", make_rows(3, 14, pattern14, extreme), {"kind": "independent"}))
    zero = [1, 0, Fraction(1, 2), 4, 0, 1, 2, 1, Fraction(1, 2), 0, 1, 2, Fraction(5, 2), 1]
    zrows = make_rows(4, 14, pattern14, zero)
    # A deleted row may carry a value that would be refused if it were read.
    zrows["selection"][1] = Fraction(0)
    zrows["propensity"][4] = Fraction(1)
    cases.append(estimate_case("independent_zero_weight_rows_are_deleted", zrows, {"kind": "independent"}))
    # Clusters: unbalanced sizes 1,1,2,3,5,4 and non-contiguous labels, with trial rows in several.
    labels16 = [7, 3, 3, 11, 11, 11, 2, 2, 2, 2, 2, 40, 40, 40, 40, 9][:16]
    pattern16 = [1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0]
    cases.append(estimate_case("clusters_unbalanced_unweighted", make_rows(5, 16, pattern16), {"kind": "clusters", "labels": labels16}))
    w16 = [Fraction(v) for v in [1, 2, Fraction(1, 2), 4, Fraction(5, 2), 1, 2, 1, Fraction(1, 2), 4, 1, 2, Fraction(5, 2), 1, 3, 1]]
    cases.append(estimate_case("clusters_unbalanced_weighted", make_rows(6, 16, pattern16, w16), {"kind": "clusters", "labels": labels16}))
    singles = list(range(16))
    cases.append(estimate_case("clusters_one_row_each", make_rows(5, 16, pattern16), {"kind": "clusters", "labels": singles}))
    # A tiny design: four rows, two clusters.
    cases.append(estimate_case("clusters_tiny", make_rows(7, 4, [1, 1, 0, 0]), {"kind": "clusters", "labels": [1, 2, 1, 2]}))
    cases.append(estimate_case("series_lag2_unweighted", make_rows(8, 14, pattern14), {"kind": "series", "lag": 2}))
    cases.append(estimate_case("series_lag3_weighted", make_rows(9, 14, pattern14, w14), {"kind": "series", "lag": 3}))
    cases.append(estimate_case("series_lag_beyond_the_rows", make_rows(10, 5, [1, 1, 0, 1, 0]), {"kind": "series", "lag": 9}))
    cases.append(estimate_case("series_lag0_is_independent", make_rows(11, 14, pattern14, w14), {"kind": "series", "lag": 0}))

    bayes = []
    bayes.append(bayes_case("bayes_rows_weighted", make_rows(12, 12, [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
                                                          [1, 2, Fraction(1, 2), 4, 1, Fraction(5, 2), 1, 2, 1, Fraction(1, 2), 3, 1]),
                            "rows", 100))
    bayes.append(bayes_case("bayes_clusters_weighted", make_rows(13, 16, pattern16, w16), ("clusters", labels16), 200))
    bayes.append(bayes_case("bayes_clusters_unweighted", make_rows(14, 16, pattern16), ("clusters", labels16), 300))
    bayes.append(bayes_case("bayes_blocks_weighted", make_rows(15, 14, pattern14, w14), ("blocks", 3), 400))
    json.dump({"estimate": cases, "bayes": bayes}, fp=__import__("sys").stdout, indent=1)
    print()


main()
