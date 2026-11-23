# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Exact multinomial-resampling covariance of several plug-in estimators over one row sample.

Regenerate with:  uv run --script shared_scenario_covariance_reference.py > shared_scenario_covariance_reference.json

Each design fixes integer row data and a list of estimators defined from the data (a linear score
mean, a ratio of two linear sums that is undefined when its denominator is zero, and a stratified
contrast of arm means that is undefined when a represented stratum lacks an arm).  Every count
vector (c_1..c_n) of n draws with replacement from n rows is enumerated with its exact probability
n!/(prod c_i!)/n^n.  A count vector on which any estimator is undefined is dropped for all of them,
its mass is reported and the surviving mass is renormalized.  The reported mean, covariance (no
bias correction: the weights are the exact probabilities) and the plug-in sampling variance of each
covariance entry, kappa/n with kappa = sum_r w_r (c_r - m)^2 for c_r = (t_ri - mu_i)(t_rj - mu_j)
and m = sum_r w_r c_r, are exact Fractions, written as the nearest binary64 and as num/den text.
A second block checks the weighted moment routine on fixed replicate matrices with the R/(R-1)
correction.  Python standard library only.  Repository-authored (MIT OR Apache-2.0).
"""
import itertools
import json
from fractions import Fraction as F
from math import factorial


def compositions(n, total):
    if n == 1:
        yield (total,)
        return
    for c in range(total + 1):
        for rest in compositions(n - 1, total - c):
            yield (c,) + rest


def est_linear(data, counts, sc):
    n = len(counts)
    return F(sum(c * a for c, a in zip(counts, data[sc["key"]])), n)


def est_ratio(data, counts, sc):
    num = sum(c * x for c, x in zip(counts, data["x"]))
    den = sum(c * z for c, z in zip(counts, data["z"]))
    return None if den == 0 else F(num, den)


def est_strat(data, counts, sc):
    n = len(counts)
    total = F(0)
    for s in sorted(set(data["s"])):
        n_s = sum(c for c, si in zip(counts, data["s"]) if si == s)
        if n_s == 0:
            continue
        arms = {}
        for c, si, t, y in zip(counts, data["s"], data["t"], data["y"]):
            if si == s and c:
                arms.setdefault(t, [0, 0])
                arms[t][0] += c * y
                arms[t][1] += c
        if 0 not in arms or 1 not in arms:
            return None
        total += F(n_s, n) * (F(arms[1][0], arms[1][1]) - F(arms[0][0], arms[0][1]))
    return total


ESTIMATORS = {"linear": est_linear, "ratio": est_ratio, "strat": est_strat}


def cov_and_var(values, weights, correction, n_rows):
    k = len(values[0])
    mass = sum(weights)
    w = [x / mass for x in weights]
    mu = [sum(wr * v[i] for wr, v in zip(w, values)) for i in range(k)]
    cov = [[F(0)] * k for _ in range(k)]
    var = [[F(0)] * k for _ in range(k)]
    for i in range(k):
        for j in range(k):
            prods = [(v[i] - mu[i]) * (v[j] - mu[j]) for v in values]
            m = sum(wr * c for wr, c in zip(w, prods))
            kappa = sum(wr * (c - m) ** 2 for wr, c in zip(w, prods))
            cov[i][j] = correction * m
            var[i][j] = correction * correction * kappa / n_rows
    return mu, cov, var


def out(x):
    return {"value": float(x), "exact": f"{x.numerator}/{x.denominator}"}


def design(name, data, scenarios):
    n = len(next(iter(data.values())))
    denom = F(n) ** n
    kept_values, kept_weights, kept_counts = [], [], []
    failed_mass = F(0)
    failed = 0
    total = 0
    for counts in compositions(n, n):
        total += 1
        p = F(factorial(n), 1)
        for c in counts:
            p /= factorial(c)
        p /= denom
        row = [ESTIMATORS[s["kind"]](data, counts, s) for s in scenarios]
        if any(v is None for v in row):
            failed_mass += p
            failed += 1
        else:
            kept_values.append(row)
            kept_weights.append(p)
    mu, cov, var = cov_and_var(kept_values, kept_weights, F(1), n)
    k = len(scenarios)
    return {
        "name": name,
        "data": data,
        "scenarios": scenarios,
        "n": n,
        "count_vectors": total,
        "failed_vectors": failed,
        "failed_mass": out(failed_mass),
        "means": [out(m) for m in mu],
        "covariance": [[out(cov[i][j]) for j in range(k)] for i in range(k)],
        "entry_sampling_variance": [[out(var[i][j]) for j in range(k)] for i in range(k)],
    }


def moment_case(name, values, weights, correction, n_rows):
    mu, cov, var = cov_and_var([[F(x) for x in v] for v in values], [F(w) for w in weights],
                               F(correction), n_rows)
    k = len(values[0])
    return {
        "name": name,
        "replicates": values,
        "weights": weights,
        "correction": float(F(correction)),
        "n_rows": n_rows,
        "covariance": [[out(cov[i][j]) for j in range(k)] for i in range(k)],
        "entry_sampling_variance": [[out(var[i][j]) for j in range(k)] for i in range(k)],
    }


designs = [
    # ties in the linear scores, a ratio that fails when only row 2 (z = 0) is drawn
    design("ties_and_ratio_failure", {"a": [1, 1, 2, 0], "b": [2, 0, 2, 1], "x": [1, 2, 3, 1], "z": [1, 0, 2, 1]},
           [{"kind": "linear", "key": "a"}, {"kind": "linear", "key": "b"}, {"kind": "ratio"}]),
    # two rows: one estimator constant (zero variance), a ratio failing on (0,2)
    design("two_rows_constant", {"a": [0, 1], "b": [3, 3], "x": [1, 2], "z": [1, 0]},
           [{"kind": "linear", "key": "a"}, {"kind": "linear", "key": "b"}, {"kind": "ratio"}]),
    # exactly collinear pair (b = 3a + 1) plus an extreme-valued third score
    design("collinear_and_extreme", {"a": [0, 1, 1, 4, 0], "b": [1, 4, 4, 13, 1], "e": [1000, 0, 0, 0, 1]},
           [{"kind": "linear", "key": "a"}, {"kind": "linear", "key": "b"}, {"kind": "linear", "key": "e"}]),
    # unbalanced strata: stratified contrast fails when a represented stratum loses an arm
    design("stratified_unbalanced", {"a": [1, 0, 2, 2, 5, 0], "s": [0, 0, 0, 1, 1, 1], "t": [0, 1, 1, 0, 0, 1],
                                     "y": [1, 3, 2, 0, 4, 9]},
           [{"kind": "strat"}, {"kind": "linear", "key": "a"}]),
]
moment_cases = [
    moment_case("bootstrap_correction_five_replicates",
                [[1, 2], [3, 1], [2, 2], [5, 0], [0, 4]], [1, 1, 1, 1, 1], F(5, 4), 6),
    moment_case("unequal_weights_zero_weight_replicate",
                [[1, 0, 2], [2, 2, 2], [0, 1, 5], [9, 9, 9]], [1, 3, 2, 0], 1, 4),
]
print(json.dumps({"designs": designs, "moment_cases": moment_cases}, separators=(",", ":")))
