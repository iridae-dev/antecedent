# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent reference for the Bayesian conditional-independence fixture.

Everything is computed from the stored columns, with exact rational linear algebra where the
quantity is rational and double-precision special functions (log, lgamma, sqrt) elsewhere. No
Antecedent code, no NumPy and no SciPy.

* residual correlation: residuals of columns 0 and 1 on [1 | Z] from the normal equations solved
  over Fractions; the statistic is |Pearson correlation of the residuals|.
* g-prior Bayes factor (g = n): log BF10 = -1/2 ln(1+g) - 1/2 (n-1-|Z|) ln(1 - g r^2 / (1+g)),
  with r the partial correlation; posterior independence = 1 / (1 + BF10) for equal prior mass.
  These are written to the fixture's `g_prior` blocks by `--write` and checked by `--check`.
* posterior-predictive tail probability: a replicate with the null mean has residual vector M e,
  e ~ N(0, s^2 I), M the residual maker of [1 | Z], so the replicate residual correlation with
  M x is distributed as a cosine in n-q dimensions: r^2 ~ Beta(1/2, (n-q-1)/2), q = 1+|Z|. The
  exact tail P(|r| >= observed) is the regularised incomplete beta (Lentz continued fraction). A
  seeded simulation of the same replicates is a second route.
* Normal-inverse-gamma marginal likelihoods: standardised columns (mean 0, variance 1 with
  denominator n), coefficient prior precision 0.01 on every coefficient including the intercept,
  inverse-gamma(0.001, 0.001); log p(y) = -n/2 ln(2 pi) + 1/2 ln|L0| - 1/2 ln|Ln| + a0 ln b0
  - an ln bn + lgamma(an) - lgamma(a0). Checked against the stored values; the posterior-mean and
  precision entries are not rechecked.

    uv run --frozen --script conformance/discovery/bayesian_ci/reference.py --check
    uv run --frozen --script conformance/discovery/bayesian_ci/reference.py --self-test
    uv run --frozen --script conformance/discovery/bayesian_ci/reference.py --write   (g_prior blocks)

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent


def solve(a: list[list[Fraction]], b: list[list[Fraction]]) -> list[list[Fraction]]:
    n = len(a)
    m = [row[:] + rhs[:] for row, rhs in zip(a, b)]
    for c in range(n):
        pivot = next(r for r in range(c, n) if m[r][c] != 0)
        m[c], m[pivot] = m[pivot], m[c]
        p = m[c][c]
        m[c] = [v / p for v in m[c]]
        for r in range(n):
            if r != c and m[r][c] != 0:
                f = m[r][c]
                m[r] = [v - f * w for v, w in zip(m[r], m[c])]
    return [row[n:] for row in m]


def determinant(a: list[list[Fraction]]) -> Fraction:
    n = len(a)
    m = [row[:] for row in a]
    det = Fraction(1)
    for c in range(n):
        pivot = next((r for r in range(c, n) if m[r][c] != 0), None)
        if pivot is None:
            return Fraction(0)
        if pivot != c:
            m[c], m[pivot] = m[pivot], m[c]
            det = -det
        det *= m[c][c]
        for r in range(c + 1, n):
            f = m[r][c] / m[c][c]
            m[r] = [v - f * w for v, w in zip(m[r], m[c])]
    return det


def columns_of(case: dict) -> list[list[Fraction]]:
    return [[Fraction(v) for v in col] for col in case["columns"]]


def residuals(design: list[list[Fraction]], targets: list[list[Fraction]]) -> list[list[Fraction]]:
    n, q = len(design), len(design[0])
    gram = [[sum((design[r][i] * design[r][j] for r in range(n)), Fraction(0)) for j in range(q)] for i in range(q)]
    rhs = [[sum((design[r][i] * t[r] for r in range(n)), Fraction(0)) for t in targets] for i in range(q)]
    beta = solve(gram, rhs)
    return [
        [t[r] - sum((beta[j][k] * design[r][j] for j in range(q)), Fraction(0)) for r in range(n)]
        for k, t in enumerate(targets)
    ]


def partial_correlation_squared(case: dict) -> tuple[Fraction, int, int]:
    cols = columns_of(case)
    n = len(cols[0])
    z = case["z_indices"]
    design = [[Fraction(1)] + [cols[j][r] for j in z] for r in range(n)]
    rx, ry = residuals(design, [cols[0], cols[1]])
    sxy = sum((a * b for a, b in zip(rx, ry)), Fraction(0))
    sxx = sum((a * a for a in rx), Fraction(0))
    syy = sum((b * b for b in ry), Fraction(0))
    return sxy * sxy / (sxx * syy), n, 1 + len(z)


def betacf(a: float, b: float, x: float) -> float:
    tiny = 1e-300
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 2000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((a + m2 - 1.0) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1.0))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def reg_inc_beta(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * betacf(a, b, x) / a
    return 1.0 - front * betacf(b, a, 1.0 - x) / b


def tail_probability(r2: Fraction, n: int, q: int) -> float:
    """P(r^2_rep >= r2) for r^2_rep ~ Beta(1/2, (n-q-1)/2)."""
    b = (n - q - 1) / 2.0
    x = float(r2)
    # P(R2 >= x) = I_{1-x}(b, 1/2)
    return reg_inc_beta(b, 0.5, 1.0 - x)


def simulated_tail(case: dict, draws: int, seed: int) -> float:
    cols = columns_of(case)
    n = len(cols[0])
    z = case["z_indices"]
    design = [[Fraction(1)] + [cols[j][r] for j in z] for r in range(n)]
    rng = random.Random(seed)
    # one residual-maker application per draw: residualise the noise on the same design
    r2_obs, _, _ = partial_correlation_squared(case)
    rx = [float(v) for v in residuals(design, [cols[0]])[0]]
    q = len(design[0])
    # Gram inverse in floats for the hat matrix
    gram = [[float(sum((design[r][i] * design[r][j] for r in range(n)), Fraction(0))) for j in range(q)] for i in range(q)]
    inv = [[float(v) for v in row] for row in solve(
        [[Fraction(v) for v in row] for row in gram],
        [[Fraction(int(i == j)) for j in range(q)] for i in range(q)])]
    des = [[float(v) for v in row] for row in design]
    hits = 0
    sxx = sum(v * v for v in rx)
    for _ in range(draws):
        e = [rng.gauss(0.0, 1.0) for _ in range(n)]
        xe = [sum(des[r][i] * e[r] for r in range(n)) for i in range(q)]
        coef = [sum(inv[i][j] * xe[j] for j in range(q)) for i in range(q)]
        res = [e[r] - sum(coef[i] * des[r][i] for i in range(q)) for r in range(n)]
        sxy = sum(a * b for a, b in zip(rx, res))
        syy = sum(b * b for b in res)
        if sxy * sxy / (sxx * syy) >= float(r2_obs):
            hits += 1
    return hits / draws


def g_prior(case: dict) -> dict:
    r2, n, q = partial_correlation_squared(case)
    g = Fraction(n)
    d = n - q
    one_minus = 1 - g / (1 + g) * r2
    log_bf = -0.5 * math.log(1 + n) - 0.5 * d * math.log(float(one_minus))
    return {
        "partial_correlation_abs": math.sqrt(float(r2)),
        "log_bf10": log_bf,
        "posterior_independence": 1.0 / (1.0 + math.exp(log_bf)),
    }


def standardised(case: dict) -> list[list[Fraction]]:
    out = []
    for col in columns_of(case):
        n = len(col)
        mean = sum(col, Fraction(0)) / n
        sd = Fraction(math.sqrt(float(sum(((v - mean) ** 2 for v in col), Fraction(0)) / n)))
        out.append([(v - mean) / sd for v in col])
    return out


def log_marginal(y: list[Fraction], regressors: list[list[Fraction]], model: dict) -> float:
    n = len(y)
    a0, b0 = Fraction(str(model["alpha0"])), Fraction(str(model["beta0"]))
    lam0 = Fraction(str(model["coefficient_prior_precision"]))
    design = [[Fraction(1)] + [col[r] for col in regressors] for r in range(n)]
    q = len(design[0])
    lam = [
        [sum((design[r][i] * design[r][j] for r in range(n)), Fraction(0)) + (lam0 if i == j else 0) for j in range(q)]
        for i in range(q)
    ]
    xty = [[sum((design[r][i] * y[r] for r in range(n)), Fraction(0))] for i in range(q)]
    mu = [row[0] for row in solve(lam, xty)]
    yty = sum((v * v for v in y), Fraction(0))
    bn = b0 + (yty - sum((m * t[0] for m, t in zip(mu, xty)), Fraction(0))) / 2
    an = float(a0) + n / 2
    logdet0 = q * math.log(float(lam0))
    logdetn = math.log(float(determinant(lam)))
    return (
        -n / 2 * math.log(2 * math.pi)
        + 0.5 * logdet0
        - 0.5 * logdetn
        + float(a0) * math.log(float(b0))
        - an * math.log(float(bn))
        + math.lgamma(an)
        - math.lgamma(float(a0))
    ), an, float(bn)


def nig(case: dict, model: dict) -> dict:
    std = standardised(case)
    x, y = std[0], std[1]
    z = [std[j] for j in case["z_indices"]]
    lm0, an, bn0 = log_marginal(y, z, model)
    lm1, _, bn1 = log_marginal(y, [x, *z], model)
    bf = lm1 - lm0
    return {
        "log_marginal_m0": lm0,
        "log_marginal_m1": lm1,
        "log_bf10": bf,
        "posterior_dependence": 1.0 / (1.0 + math.exp(-bf)),
        "alpha_n": an,
        "beta_n_m0": bn0,
        "beta_n_m1": bn1,
    }


def check(document: dict) -> list[str]:
    problems = []
    tol = document["tolerances"]
    for case in document["cases"]:
        name, ref = case["name"], case["reference"]
        pp = ref["posterior_predictive"]
        r2, n, q = partial_correlation_squared(case)
        if abs(math.sqrt(float(r2)) - pp["observed_abs_residual_correlation"]) > tol["ppc_statistic_atol"]:
            problems.append(f"{name}: observed statistic {pp['observed_abs_residual_correlation']} vs {math.sqrt(float(r2))}")
        exact = tail_probability(r2, n, q)
        band = max(tol["ppc_absolute_floor"], tol["ppc_mcse_multiplier"] * pp["mcse"])
        if abs(exact - pp["tail_probability"]) > min(band, 0.02):
            problems.append(f"{name}: tail {pp['tail_probability']} vs exact {exact}")
        sim = simulated_tail(case, 3000, 20260801)
        if abs(sim - exact) > 0.03:
            problems.append(f"{name}: simulated tail {sim} vs exact {exact}")
        g = g_prior(case)
        stored = case.get("g_prior")
        if stored is None:
            problems.append(f"{name}: g_prior block missing (run --write)")
        else:
            for key, value in g.items():
                if abs(stored[key] - value) > 1e-12 * max(1.0, abs(value)):
                    problems.append(f"{name}: g_prior {key} {stored[key]} vs {value}")
        nig_values = nig(case, document["model"])
        for key, stored_value in (
            ("log_marginal_m0", ref["log_marginal_m0"]),
            ("log_marginal_m1", ref["log_marginal_m1"]),
            ("log_bf10", ref["log_bf10"]),
            ("alpha_n", ref["m0"]["alpha_n"]),
            ("beta_n_m0", ref["m0"]["beta_n"]),
            ("beta_n_m1", ref["m1"]["beta_n"]),
        ):
            if abs(nig_values[key] - stored_value) > 1e-8 * max(1.0, abs(stored_value)):
                problems.append(f"{name}: NIG {key} {stored_value} vs {nig_values[key]}")
        if abs(nig_values["posterior_dependence"] - ref["posterior_dependence"]) > 1e-9:
            problems.append(f"{name}: NIG dependence {ref['posterior_dependence']} vs {nig_values['posterior_dependence']}")
    return problems


def self_test(document: dict) -> list[str]:
    failures = []
    for mutate, label in (
        (lambda d: d["cases"][1]["reference"]["posterior_predictive"].__setitem__("tail_probability", 0.5), "tail"),
        (lambda d: d["cases"][0]["reference"].__setitem__("log_bf10", 0.0), "NIG BF"),
        (lambda d: d["cases"][3]["g_prior"].__setitem__("log_bf10", 0.0), "g-prior BF"),
        (lambda d: d["cases"][2]["reference"]["posterior_predictive"].__setitem__("observed_abs_residual_correlation", 0.5), "statistic"),
    ):
        wrong = copy.deepcopy(document)
        mutate(wrong)
        if not check(wrong):
            failures.append(f"changed {label} accepted")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    path = HERE / "expected.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    if args.write:
        for case in document["cases"]:
            case["g_prior"] = g_prior(case)
        path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("written")
        return 0
    problems = self_test(document) if args.self_test else check(document)
    for problem in problems:
        print("FAIL", problem)
    if not problems:
        print("ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
