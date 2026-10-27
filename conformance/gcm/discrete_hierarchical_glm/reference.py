# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.1.3", "scipy==1.14.1", "pandas==2.2.3", "statsmodels==0.14.4"]
# ///
"""Retained regeneration of the discrete and hierarchical-GLM reference coefficients.

The data are the fixture's declared deterministic construction (binary parent, outcome equal
to the parent except on rows with index divisible by 8).  Three quantities are produced:

* the unpenalized logit coefficients, by the public ``statsmodels`` ``Logit`` API;
* the empirical-Bayes ridge weight lambda = sigma^2 / tau^2 from linear-probability OLS
  moments (sigma^2 = SSR/(n-p), tau^2 = max(mean slope^2 - sigma^2/n, 1e-8), clamped to
  [1e-4, 1e3]), the documented rule of the product, recomputed here from ``statsmodels`` OLS;
* the ridge logit solution minimizing  -loglik + (lambda/2) * slope^2  (intercept unpenalized),
  by a Newton iteration written in this script on the analytic score and Hessian.

It imports no Antecedent code.  The lambda rule and the penalty scaling are the product's
specification, matched here, so agreement establishes the arithmetic and the logistic solve,
not that the rule is a good empirical-Bayes estimator.

    uv run --frozen --script conformance/gcm/discrete_hierarchical_glm/reference.py --check
    uv run --frozen --script conformance/gcm/discrete_hierarchical_glm/reference.py --self-test

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import statsmodels.api as sm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from reference_compare import values_close  # noqa: E402

ATOL = 1e-9


def data(n: int):
    x = np.array([0.0 if i < n // 2 else 1.0 for i in range(n)])
    y = np.array([1.0 - p if i % 8 == 0 else p for i, p in enumerate(x)])
    return sm.add_constant(x), y


def eb_lambda(design, y) -> float:
    n = len(y)
    fit = sm.OLS(y, design).fit()
    slope = fit.params[1:]
    sigma2 = fit.ssr / (n - design.shape[1])
    tau2 = max(float(np.mean(slope**2)) - sigma2 / n, 1e-8)
    return float(min(max(sigma2 / tau2, 1e-4), 1e3))


def ridge_logit(design, y, lam: float) -> list[float]:
    beta = np.zeros(design.shape[1])
    penalty = np.diag([0.0] + [lam] * (design.shape[1] - 1))
    for _ in range(100):
        p = 1.0 / (1.0 + np.exp(-(design @ beta)))
        score = design.T @ (y - p) - penalty @ beta
        hessian = design.T @ (design * (p * (1 - p))[:, None]) + penalty
        step = np.linalg.solve(hessian, score)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-15:
            break
    return [float(b) for b in beta]


def outputs(document: dict) -> dict:
    design, y = data(document["data"]["n"])
    lam = eb_lambda(design, y)
    return {
        "unpenalized_logit_coefficients": [
            float(b) for b in sm.Logit(y, design).fit(disp=0, tol=1e-14, maxiter=200).params
        ],
        "eb_lambda": lam,
        "ridge_logit_coefficients": ridge_logit(design, y, lam),
    }


def check(document: dict) -> list[str]:
    problems = []
    if document["acceptance"]["coefficient_atol"] != 2e-07:
        problems.append("product coefficient tolerance changed from the reviewed bound")
    if document["data"]["n"] != 120:
        problems.append("declared sample size changed")
    for field, actual in outputs(document).items():
        if not values_close(document["reference"][field], actual, rtol=0.0, atol=ATOL):
            problems.append(f"{field} differs from the regenerated reference")
    return problems


def self_test(document: dict) -> list[str]:
    failures = []
    wrong = copy.deepcopy(document)
    wrong["reference"]["ridge_logit_coefficients"][1] += 1e-6
    if not check(wrong):
        failures.append("shifted ridge slope accepted")
    wrong = copy.deepcopy(document)
    wrong["reference"]["eb_lambda"] *= 1.01
    if not check(wrong):
        failures.append("changed lambda accepted")
    wrong = copy.deepcopy(document)
    wrong["acceptance"]["coefficient_atol"] = 1e-3
    if not check(wrong):
        failures.append("widened tolerance accepted")
    wrong = copy.deepcopy(document)
    wrong["data"]["n"] = 80
    if not check(wrong):
        failures.append("changed data accepted")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    document = json.loads((HERE / "expected.json").read_text(encoding="utf-8"))
    problems = self_test(document) if args.self_test else check(document)
    for p in problems:
        print("FAIL", p)
    if not problems:
        print("ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
