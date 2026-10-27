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
from scipy.interpolate import BSpline
from statsmodels.stats.multitest import multipletests

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


def check_descriptive(document: dict) -> list[str]:
    problems = []
    if document["atol"] != 1e-12 or document["rtol"] != 1e-12:
        problems.append("descriptive acceptance bounds changed")
    if {c["name"] for c in document["cases"]} != {
        "ordinary",
        "translated",
        "constant",
        "two_values",
        "singleton",
    } or len(document["cases"]) != 5:
        problems.append("descriptive scenario grid changed")
    for case in document["cases"]:
        values = np.asarray(case["values"])
        actual = {
            "mean": float(np.mean(values)),
            "population_variance": float(np.var(values)),
            "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        }
        if not values_close(
            {key: case[key] for key in actual}, actual, atol=1e-12, rtol=1e-12
        ):
            problems.append(f"descriptive {case['name']} differs from NumPy")
    return problems


def check_gam_basis(document: dict) -> list[str]:
    """Only basis matrices are external evidence; fitted outputs are regression pins."""
    problems = []
    if document["atol"] != 2e-9 or document["rtol"] != 2e-9:
        problems.append("GAM basis acceptance bounds changed")
    if {c["name"] for c in document["cases"]} != {
        "single_smooth",
        "high_penalty",
        "two_smooths",
    } or len(document["cases"]) != 3:
        problems.append("GAM basis scenario grid changed")
    for case in document["cases"]:
        matrices = []
        for spec in case["specs"]:
            degree = len(spec["knots"]) - spec["n_basis"] - 1
            if degree != 3:
                problems.append(f"{case['name']}: unsupported basis degree")
                continue
            basis = BSpline.design_matrix(
                case["x_colmajor"][spec["raw_col"]], spec["knots"], degree
            ).toarray()
            matrices.append(basis.tolist())
        if not values_close(
            case["basis_matrices_rowmajor"], matrices, atol=2e-9, rtol=2e-9
        ):
            problems.append(f"{case['name']}: basis differs from SciPy")
    return problems


def closure_outputs(fixture: str, document: dict) -> dict:
    """Use public fits on inputs specified by the fixture, never product helpers."""
    if fixture == "discovery/multiplicity":
        methods = {
            "benjamini_hochberg": "fdr_bh",
            "benjamini_yekutieli": "fdr_by",
            "bonferroni": "bonferroni",
            "holm": "holm",
        }
        result = {}
        for case in document["cases"]:
            outputs = {}
            for name, method in methods.items():
                # Upstream rejects zero-width input; this case is a local empty
                # contract, explicitly not an external algorithm comparison.
                reject, adjusted = (
                    ([], [])
                    if not case["p_values"]
                    else multipletests(case["p_values"], alpha=0.05, method=method)[:2]
                )
                outputs[name] = {
                    "adjusted": list(map(float, adjusted)),
                    "reject_at_0_05": list(map(bool, reject)),
                }
            result[case["name"]] = outputs
        routing = document["discovery_routing"]
        routed = {}
        for exclude in (True, False):
            indices = [
                i
                for i, lag in enumerate(routing["source_lags"])
                if not exclude or lag != 0
            ]
            reject, adjusted = multipletests(
                [routing["p_values"][i] for i in indices], alpha=0.05, method="fdr_bh"
            )[:2]
            by_link = [None] * len(routing["p_values"])
            retained = [
                i
                for i, lag in enumerate(routing["source_lags"])
                if exclude and lag == 0 and routing["p_values"][i] <= 0.05
            ]
            for i, q, keep in zip(indices, adjusted, reject, strict=True):
                by_link[i] = float(q)
                if keep:
                    retained.append(i)
            routed[str(exclude).lower()] = {
                "adjusted_by_link": by_link,
                "retained_indices": sorted(retained),
            }
        return {"cases": result, "routing": routed}
    if fixture == "estimate/conditional_effects":
        i = np.arange(document["data"]["n"])
        t, w = i % 2, i % 7 - 3
        y = (
            1.2
            + 1.8 * t
            - 0.4 * w
            + 0.65 * t * w
            + 0.25 * np.sin(0.31 * i)
            + 0.1 * np.cos(0.17 * i)
        )
        fit = sm.OLS(y, np.column_stack([np.ones(len(i)), t, w, t * w])).fit()
        contrast = np.array([0, 1, 0, np.mean(w)])
        return {
            "coefficients": fit.params.tolist(),
            "modifier_mean": float(np.mean(w)),
            "ate_at_modifier_mean": float(contrast @ fit.params),
            "analytic_se": float(np.sqrt(contrast @ fit.cov_params() @ contrast)),
        }
    if fixture == "estimate/glm_adjustment_grid":
        rows = [
            (c["treatment"], c["z"], y)
            for c in document["data"]["cells"]
            for y, n in [(1, c["successes"]), (0, c["failures"])]
            for _ in range(n)
        ]
        t, z, y = np.asarray(rows).T
        fit = sm.GLM(
            y, np.column_stack([np.ones(len(y)), t, z]), family=sm.families.Binomial()
        ).fit(tol=1e-12, maxiter=500)
        x0 = np.column_stack([np.ones(len(y)), np.zeros(len(y)), z])
        x1 = x0.copy()
        x1[:, 1] = 1
        return {
            "coefficients_intercept_t_z": fit.params.tolist(),
            "all_observed_ate": float(np.mean(fit.predict(x1) - fit.predict(x0))),
        }
    if fixture == "estimate/temporal_mediation_grid":
        i = np.arange(document["data"]["n"])
        t = np.sin(0.071 * i) + 0.35 * np.cos(0.137 * i)
        # Lagging drops row zero; it never wraps the final observation around.
        lag = t[:-1]
        m = 0.8 * lag + 0.12 * np.sin(0.43 * i[1:])
        y = 0.25 * lag + 0.55 * m + 0.09 * np.cos(0.29 * i[1:])
        first = sm.OLS(m, sm.add_constant(lag)).fit()
        total = sm.OLS(y, sm.add_constant(lag)).fit()
        direct = sm.OLS(y, np.column_stack([np.ones(len(lag)), lag, m])).fit()
        a, b = first.params[1], direct.params[2]
        return {
            "a": float(a),
            "b": float(b),
            "total": float(total.params[1]),
            "direct": float(direct.params[1]),
            "mediated": float(a * b),
            "se_total": float(total.bse[1]),
            "se_direct": float(direct.bse[1]),
            "se_mediated_sobel": float(
                np.sqrt(b * b * first.bse[1] ** 2 + a * a * direct.bse[2] ** 2)
            ),
        }
    raise ValueError(f"unsupported closure fixture {fixture}")


def check_closure(fixture: str, document: dict) -> list[str]:
    if fixture == "discovery/multiplicity":
        expected = {
            "cases": {c["name"]: c["outputs"] for c in document["cases"]},
            "routing": document["discovery_routing"]["exclude_contemporaneous"],
        }
        valid_names = {"ordinary", "ties_and_boundaries", "permuted", "single", "empty"}
        tolerance, rtol = 1e-15, 1e-13
        valid = {c["name"] for c in document["cases"]} == valid_names and document[
            "tolerance"
        ] == {
            "atol": tolerance,
            "rtol": rtol,
        }
        valid = valid and len(document["cases"]) == len(valid_names)
        valid = (
            valid
            and document["discovery_routing"]["alpha"] == 0.05
            and document["discovery_routing"]["method"] == "benjamini_hochberg"
        )
    else:
        expected = document["reference"]
        tolerance, rtol = (
            (
                2e-7
                if fixture.endswith("glm_adjustment_grid")
                else 3e-9
                if fixture.endswith("temporal_mediation_grid")
                else 2e-9
            ),
            0,
        )
        if fixture == "estimate/conditional_effects":
            inputs = {
                "n": 240,
                "treatment": "i mod 2",
                "modifier": "(i mod 7)-3",
                "outcome": "1.2 + 1.8*T - 0.4*W + 0.65*T*W + 0.25*sin(0.31*i) + 0.1*cos(0.17*i)",
            }
            acceptance = {"atol": 2e-9}
        elif fixture == "estimate/temporal_mediation_grid":
            inputs = {
                "n": 320,
                "treatment": "sin(0.071*i)+0.35*cos(0.137*i)",
                "mediator": "0.8*T[i-1]+0.12*sin(0.43*i)",
                "outcome": "0.25*T[i-1]+0.55*M[i]+0.09*cos(0.29*i)",
            }
            acceptance = {"atol": 3e-9}
        else:
            inputs = {
                "columns": ["treatment", "outcome", "z"],
                "cells": [
                    {"z": z, "treatment": t, "successes": s, "failures": 100 - s}
                    for z, t, s in [(0, 0, 20), (0, 1, 50), (1, 0, 40), (1, 1, 70)]
                ],
            }
            acceptance = {"coefficient_atol": 2e-7, "ate_atol": 2e-7}
        valid = document["data"] == inputs and document["acceptance"] == acceptance
    if not valid:
        return [f"{fixture}: scenario or acceptance bounds differ from reviewed inputs"]
    actual = closure_outputs(fixture, document)
    return (
        []
        if valid and values_close(expected, actual, atol=tolerance, rtol=rtol)
        else [
            f"{fixture}: scenario or reference output differs from pinned public fits"
        ]
    )


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
    # A reference-review receipt describes the comparison; it is not a generated
    # numerical output. All other fixture inputs, outputs and metadata must match.
    payload = {
        **document,
        "oracle": {
            key: value
            for key, value in document["oracle"].items()
            if key != "reference_review"
        },
    }
    return (
        []
        if values_close(payload, build_grid(), rtol=1e-9, atol=1e-9)
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
        choices=(
            "ols_wls",
            "ols_wls_grid",
            "sandwich_covariance",
            "glm_irls",
            "descriptive",
            "gam",
            "discovery/multiplicity",
            "estimate/conditional_effects",
            "estimate/glm_adjustment_grid",
            "estimate/temporal_mediation_grid",
        ),
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
        (
            (
                ROOT / "conformance"
                if "/" in args.fixture
                else Path(__file__).parent.parent
            )
            / args.fixture
            / "expected.json"
        ).read_text(encoding="utf-8")
    )
    checker = (
        (lambda doc: check_closure(args.fixture, doc))
        if "/" in args.fixture
        else {
            "ols_wls": check,
            "ols_wls_grid": check_grid,
            "sandwich_covariance": check_covariances,
            "glm_irls": check_glms,
            "descriptive": check_descriptive,
            "gam": check_gam_basis,
        }[args.fixture]
    )
    problems = checker(document)
    if args.self_test and not problems:
        mutations = []
        changed = copy.deepcopy(document)
        if args.fixture == "descriptive":
            changed["cases"][0]["mean"] += 0.01
        elif args.fixture == "gam":
            changed["cases"][0]["basis_matrices_rowmajor"][0][0][0] += 0.01
        elif args.fixture == "discovery/multiplicity":
            changed["cases"][0]["outputs"]["holm"]["adjusted"][0] += 0.01
        elif "/" in args.fixture:
            field = next(iter(changed["reference"]))
            if isinstance(changed["reference"][field], list):
                changed["reference"][field][0] += 0.01
            else:
                changed["reference"][field] += 0.01
        elif args.fixture == "sandwich_covariance":
            changed["covariances"]["hc1"][0][0] += 0.01
        else:
            changed["cases"][0]["coefficients"][0] += 0.01
        mutations.append(changed)
        if "/" in args.fixture:
            changed = copy.deepcopy(document)
            if args.fixture == "discovery/multiplicity":
                changed["tolerance"]["atol"] = 1.0
            else:
                changed["acceptance"][next(iter(changed["acceptance"]))] = 1.0
            mutations.append(changed)
            changed = copy.deepcopy(document)
            if args.fixture == "discovery/multiplicity":
                changed["cases"] = []
            else:
                changed["data"]["n"] = 1
            mutations.append(changed)
        if args.fixture == "ols_wls":
            changed = copy.deepcopy(document)
            changed["atol"] = 1.0
            mutations.append(changed)
        if "/" not in args.fixture:
            changed = copy.deepcopy(document)
            changed[
                "covariances" if args.fixture == "sandwich_covariance" else "cases"
            ] = [] if args.fixture != "sandwich_covariance" else {}
            mutations.append(changed)
        for index, mutated in enumerate(mutations):
            if not checker(mutated):
                problems.append(f"fault control {index} passed incorrectly")
    for problem in problems:
        print(f"FAIL: {problem}")
    if not problems:
        cases = document.get(
            "cases", document.get("covariances", document.get("reference", []))
        )
        label = (
            "output fields"
            if "/" in args.fixture and args.fixture != "discovery/multiplicity"
            else "cases"
        )
        print(f"{args.fixture}: {len(cases)} pinned black-box {label} reproduced")
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
