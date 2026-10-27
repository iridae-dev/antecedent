# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.1.3", "scipy==1.14.1", "pandas==2.2.3", "statsmodels==0.14.4"]
# ///
"""Retained black-box regeneration of the frozen OLS/WLS reference outputs.

Reads only the fixture's declared design/response/weight inputs. No Antecedent
algorithm, test helper, or upstream implementation source is imported. The
comparison helper compares documents and computes no reference values.

uv run --frozen --script conformance/stats/ols_wls/reference.py --check
SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import math
import sys
from pathlib import Path

import numpy as np
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from reference_compare import values_close

PINS = {"numpy": "2.1.3", "scipy": "1.14.1", "pandas": "2.2.3", "statsmodels": "0.14.4"}


def outputs(case: dict) -> dict:
    x = np.asarray(case["x_rowmajor"], dtype=float)
    y = np.asarray(case["y"], dtype=float)
    weights = case["weights"]
    model = (
        sm.OLS(y, x) if weights is None else sm.WLS(y, x, weights=np.asarray(weights))
    )
    fit = model.fit()
    return {
        "coefficients": fit.params.tolist(),
        "fitted": fit.fittedvalues.tolist(),
        "residuals": fit.resid.tolist(),
        "rank": int(model.rank),
        "rss": float(fit.resid @ fit.resid),
        "weighted_rss": None if weights is None else float(fit.ssr),
    }


def check(document: dict) -> list[str]:
    problems = []
    # A corrupted fixture cannot widen its own regeneration acceptance bounds.
    if document.get("atol") != 1e-9 or document.get("rtol") != 1e-9:
        problems.append("fixture tolerance changed from reviewed 1e-9 bounds")
    for case in document["cases"]:
        for field, actual in outputs(case).items():
            if not values_close(case.get(field), actual, rtol=1e-9, atol=1e-9):
                problems.append(
                    f"{case['name']}.{field} differs from pinned statsmodels"
                )
    if not document["cases"]:
        problems.append("empty oracle case selection")
    return problems


def check_covariances(document: dict) -> list[str]:
    x, y = np.asarray(document["x_rowmajor"]), np.asarray(document["y"])
    fit = sm.OLS(y, x).fit()
    problems = []
    expected_names = {
        "homoskedastic",
        "hc0",
        "hc1",
        "hc2",
        "hc3",
        "cluster",
        "newey_west_lag_0",
        "newey_west_lag_1",
        "newey_west_lag_3",
        "newey_west_lag_20",
        "newey_west_lag_20_capped",
    }
    if set(document["covariances"]) != expected_names:
        problems.append("sandwich covariance scenario grid changed")
    if not values_close(
        document["residuals"], fit.resid.tolist(), rtol=1e-9, atol=1e-9
    ):
        problems.append("sandwich residual inputs differ from fitted OLS residuals")
    for name, expected in document["covariances"].items():
        if name == "homoskedastic":
            actual = fit.cov_params()
        elif name.startswith("hc"):
            actual = fit.get_robustcov_results(cov_type=name.upper()).cov_params()
        elif name == "cluster":
            actual = fit.get_robustcov_results(
                cov_type="cluster",
                groups=document["groups"],
                use_correction=True,
                df_correction=False,
            ).cov_params()
        elif name == "newey_west_lag_20_capped":
            actual = fit.get_robustcov_results(
                cov_type="HAC", maxlags=min(20, len(y) - 1), use_correction=False
            ).cov_params()
        elif name.startswith("newey_west_lag_"):
            actual = fit.get_robustcov_results(
                cov_type="HAC",
                maxlags=int(name.rsplit("_", 1)[1]),
                use_correction=False,
            ).cov_params()
        else:
            continue
        if not values_close(expected, actual.tolist(), rtol=1e-9, atol=1e-9):
            problems.append(f"sandwich {name} differs from pinned statsmodels")
    return problems


def check_glms(document: dict) -> list[str]:
    families = {
        "gaussian_identity": sm.families.Gaussian(),
        "binomial_logit": sm.families.Binomial(),
        "binomial_probit": sm.families.Binomial(link=sm.families.links.Probit()),
        "poisson_log": sm.families.Poisson(),
        "negative_binomial_alpha_0_7": sm.families.NegativeBinomial(alpha=0.7),
    }
    problems = []
    if {c["name"] for c in document["cases"]} != families.keys():
        problems.append("GLM family scenario grid changed")
    for case in document["cases"]:
        if case["name"] not in families:
            continue
        model = sm.GLM(
            np.asarray(case["y"]),
            np.asarray(case["x_rowmajor"]),
            family=families[case["name"]],
        )
        fit = model.fit(tol=1e-12, maxiter=500)
        actual = {
            "coefficients": fit.params.tolist(),
            "covariance": fit.cov_params().tolist(),
            "deviance": float(fit.deviance),
            "fitted_mean": fit.fittedvalues.tolist(),
            "log_likelihood": float(fit.llf),
            "score": model.score(fit.params).tolist(),
        }
        for field, value in actual.items():
            rtol, atol = (
                (5e-7, 5e-7) if field in {"coefficients", "covariance"} else (0.0, 2e-6)
            )
            if not values_close(case[field], value, rtol=rtol, atol=atol):
                problems.append(
                    f"{case['name']}.{field} differs from pinned statsmodels"
                )
    return problems


def build_grid() -> dict:
    cases = []
    for n in (12, 32):
        for design in ("continuous", "categorical"):
            x = [
                [
                    1.0,
                    math.sin(0.7 * i) if design == "continuous" else float(i % 3 == 1),
                    (i - n / 2) / n if design == "continuous" else float(i % 3 == 2),
                ]
                for i in range(n)
            ]
            y = [
                1.2
                + 0.8 * row[1]
                - 0.3 * row[2]
                + (0.05 + 0.02 * (i % 5)) * math.cos(1.3 * i)
                for i, row in enumerate(x)
            ]
            for weighting in ("unweighted", "unequal", "zero_rows"):
                weights = (
                    None
                    if weighting == "unweighted"
                    else [
                        0.0
                        if weighting == "zero_rows" and i % 7 == 0
                        else 0.2 + (i % 5)
                        for i in range(n)
                    ]
                )
                case = {
                    "name": f"{design}_{weighting}_n{n}",
                    "x_rowmajor": x,
                    "y": y,
                    "weights": weights,
                }
                case.update(outputs(case))
                if weights is None:
                    case["covariance_hc3"] = (
                        sm.OLS(y, x)
                        .fit()
                        .get_robustcov_results(cov_type="HC3")
                        .cov_params()
                        .tolist()
                    )
                cases.append(case)
    return {
        "atol": 1e-9,
        "rtol": 1e-9,
        "cases": cases,
        "oracle": {
            "kind": "external_package",
            "project": "statsmodels",
            "version": "0.14.4",
            "generator": "conformance/stats/ols_wls/reference.py",
            "limitations": "Fixed designs; HC3 only for unweighted fits. No repeated-sampling coverage claim.",
        },
        "reference": {
            "project": "statsmodels",
            "version": "0.14.4",
            "available": True,
            "command": "uv run --frozen --script conformance/stats/ols_wls/reference.py --check --fixture ols_wls_grid",
            "generator_retained": True,
            "environment_lock": "conformance/stats/ols_wls/reference.py.lock",
        },
    }


def check_grid(document: dict) -> list[str]:
    return (
        []
        if values_close(document, build_grid(), rtol=1e-9, atol=1e-9)
        else [
            "OLS/WLS grid differs from regenerated independent design/output/uncertainty"
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check", action="store_true")
    modes.add_argument("--self-test", action="store_true")
    modes.add_argument(
        "--write-grid",
        action="store_true",
        help="generate the independent OLS/WLS scenario grid",
    )
    modes.add_argument(
        "--write-capped-lag",
        action="store_true",
        help="add a comparator with matched effective HAC lag",
    )
    parser.add_argument(
        "--fixture",
        choices=("ols_wls", "ols_wls_grid", "sandwich_covariance", "glm_irls"),
        default="ols_wls",
    )
    args = parser.parse_args()
    for package, pin in PINS.items():
        if importlib.metadata.version(package) != pin:
            raise SystemExit(
                f"{package} must be {pin}; use the frozen script environment"
            )
    if args.write_grid:
        target = Path(__file__).parent.parent / "ols_wls_grid/expected.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(build_grid(), indent=2) + "\n", encoding="utf-8")
        return 0
    if args.write_capped_lag:
        target = Path(__file__).parent.parent / "sandwich_covariance/expected.json"
        document = json.loads(target.read_text(encoding="utf-8"))
        fit = sm.OLS(document["y"], document["x_rowmajor"]).fit()
        document["covariances"]["newey_west_lag_20_capped"] = (
            fit.get_robustcov_results(
                cov_type="HAC",
                maxlags=min(20, len(document["y"]) - 1),
                use_correction=False,
            )
            .cov_params()
            .tolist()
        )
        document["conventions"]["oversized_lag"] = (
            "Antecedent caps the lag and Bartlett denominator at n-1; statsmodels uses the requested "
            "bandwidth. Compare Antecedent lag=20 against statsmodels maxlags=min(20,n-1). "
            "The raw statsmodels lag=20 output is retained to assert the intentional difference."
        )
        target.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        return 0
    document = json.loads(
        (Path(__file__).parent.parent / args.fixture / "expected.json").read_text(
            encoding="utf-8"
        )
    )
    checker = {
        "ols_wls": check,
        "ols_wls_grid": check_grid,
        "sandwich_covariance": check_covariances,
        "glm_irls": check_glms,
    }[args.fixture]
    problems = checker(document)
    if args.self_test and not problems:
        mutations = []
        changed = copy.deepcopy(document)
        if args.fixture == "sandwich_covariance":
            changed["covariances"]["hc1"][0][0] += 0.01
        else:
            changed["cases"][0]["coefficients"][0] += 0.01
        mutations.append(changed)
        if args.fixture == "ols_wls":
            changed = copy.deepcopy(document)
            changed["atol"] = 1.0
            mutations.append(changed)
        changed = copy.deepcopy(document)
        changed["covariances" if args.fixture == "sandwich_covariance" else "cases"] = (
            [] if args.fixture != "sandwich_covariance" else {}
        )
        mutations.append(changed)
        for index, mutated in enumerate(mutations):
            if not checker(mutated):
                problems.append(f"fault control {index} passed incorrectly")
    for problem in problems:
        print(f"FAIL: {problem}")
    if not problems:
        cases = document.get("cases", document.get("covariances", []))
        print(f"{args.fixture}: {len(cases)} pinned black-box cases reproduced")
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
