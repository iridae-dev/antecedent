# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the conjugate count posteriors and the empirical-Bayes discount.

Beta-Binomial: prior Beta(a, b) and k successes in n trials give the posterior Beta(a + k, b + n - k),
mean (a + k)/(a + b + n), and the marginal likelihood of the count C(n, k) B(a + k, b + n - k) / B(a, b).
Exceedance: a success is an outcome strictly above the threshold.
Gamma-Poisson: prior Gamma(shape a, rate b) and counts y_i over unit exposures give the posterior
Gamma(a + sum y, b + n), mean (a + sum y)/(b + n), and the marginal likelihood
prod 1/y_i! * b^a / Gamma(a) * Gamma(a + sum y) / (b + n)^(a + sum y).
Empirical Bayes: the power prior at discount alpha is Beta(a + alpha k0, b + alpha (n0 - k0)); the score
of alpha is the log marginal likelihood of the target count under it and the selected discount is the
grid point with the largest score (the smallest on a tie).

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/bayesian/conjugate_count_posteriors/reference.py --check

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def ln_beta(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def ln_choose(n, k):
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def beta_log_ml(a, b, k, n):
    return ln_choose(n, k) + ln_beta(a + k, b + n - k) - ln_beta(a, b)


def gamma_log_ml(a, b, counts):
    s, n = sum(counts), len(counts)
    kernel = -sum(math.lgamma(y + 1) for y in counts)
    return (
        a * math.log(b)
        - math.lgamma(a)
        + math.lgamma(a + s)
        - (a + s) * math.log(b + n)
        + kernel
    )


def close(got, want, tol, label, problems):
    if not math.isclose(got, want, rel_tol=0.0, abs_tol=tol):
        problems.append(f"{label}: derived {got!r} != fixture {want!r} (tol {tol})")


def check(document):
    problems = []
    tol = document["tol"]
    b = document["beta_binomial_posterior"]
    a0, b0 = b["prior"]
    close(a0 + b["successes"], b["expected_alpha"], tol, "beta alpha", problems)
    close(
        b0 + b["trials"] - b["successes"],
        b["expected_beta"],
        tol,
        "beta beta",
        problems,
    )
    close(
        b["expected_alpha"] / (b["expected_alpha"] + b["expected_beta"]),
        b["expected_mean"],
        tol,
        "beta mean",
        problems,
    )
    close(
        beta_log_ml(a0, b0, b["successes"], b["trials"]),
        b["expected_log_marginal_likelihood"],
        tol,
        "beta log ml",
        problems,
    )

    e = document["exceedance"]
    successes = sum(1 for y in e["outcomes"] if y > e["threshold"])
    close(successes, e["expected_successes"], tol, "exceedance successes", problems)
    close(
        e["prior"][0] + successes,
        e["expected_alpha"],
        tol,
        "exceedance alpha",
        problems,
    )
    close(
        e["prior"][1] + len(e["outcomes"]) - successes,
        e["expected_beta"],
        tol,
        "exceedance beta",
        problems,
    )

    g = document["gamma_poisson_posterior"]
    a, r = g["prior"]
    close(a + sum(g["counts"]), g["expected_shape"], tol, "gamma shape", problems)
    close(r + len(g["counts"]), g["expected_rate"], tol, "gamma rate", problems)
    close(
        g["expected_shape"] / g["expected_rate"],
        g["expected_mean"],
        tol,
        "gamma mean",
        problems,
    )
    close(
        gamma_log_ml(a, r, g["counts"]),
        g["expected_log_marginal_likelihood"],
        tol,
        "gamma log ml",
        problems,
    )

    eb = document["empirical_bayes_beta"]
    base_a, base_b = eb["base"]
    k0, n0 = eb["source"]
    k, n = eb["target"]
    scores = [
        beta_log_ml(base_a + al * k0, base_b + al * (n0 - k0), k, n)
        for al in eb["grid"]
    ]
    for i, (got, want) in enumerate(zip(scores, eb["expected_scores"])):
        close(got, want, tol, f"eb score {i}", problems)
    best = 0
    for i in range(1, len(scores)):
        if scores[i] > scores[best] + 1e-12 * max(1.0, abs(scores[best])):
            best = i
    close(
        eb["grid"][best],
        eb["expected_selected_alpha"],
        tol,
        "eb selected alpha",
        problems,
    )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.parse_args()
    document = json.loads((HERE / "expected.json").read_text(encoding="utf-8"))
    problems = check(document)
    for problem in problems:
        print("FAIL", problem)
    if not problems:
        print("ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
