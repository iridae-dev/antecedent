# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["arviz==0.20.0", "numpy==2.1.3", "scipy==1.14.1", "pandas==2.2.3", "xarray==2024.11.0", "matplotlib==3.9.2"]
# ///
"""Reproduce rank diagnostics through public ArviZ APIs on frozen chain inputs.

uv run --frozen --script conformance/validate/bayesian_checks/reference.py --check
No Antecedent sampler, algorithm or test helper is imported or executed.
SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import sys
from decimal import Decimal, localcontext
from itertools import accumulate, takewhile
from pathlib import Path

import arviz as az
import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from reference_compare import values_close


def outputs(case: dict) -> dict:
    samples = np.asarray(case["samples_chain_major"], dtype=float).reshape(
        case["chains"], case["draws"]
    )
    return {
        "rhat_rank": float(az.rhat(samples, method="rank")),
        "ess_bulk": float(az.ess(samples, method="bulk")),
        "ess_tail": float(az.ess(samples, method="tail")),
    }


def primary_outputs(case: dict) -> dict:
    """Independent 60-digit evaluation of the paper's full-pair IPS/IMS equations.

    SciPy supplies ranks/normal scores; Decimal evaluates all covariance sums.
    This is a separate reference for the explicit unaveraged full-pair variant,
    not an assertion that ArviZ uses the same finite-lag convention.
    """
    raw = np.asarray(case["samples_chain_major"]).reshape(case["chains"], case["draws"])
    raw = raw[:, : 2 * (case["draws"] // 2)].reshape(2 * case["chains"], -1)

    def normal_scores(x):
        return ndtri(
            (rankdata(x, method="average").reshape(x.shape) - 0.375) / (x.size + 0.25)
        )

    def moments(x):
        rows = [[Decimal.from_float(float(v)) for v in row] for row in x]
        m, n = len(rows), len(rows[0])
        means = [sum(row) / n for row in rows]
        centered = [
            [v - mean for v in row] for row, mean in zip(rows, means, strict=True)
        ]
        within = sum(sum(v * v for v in row) / (n - 1) for row in centered) / m
        grand = sum(means) / m
        between = n * sum((mean - grand) ** 2 for mean in means) / (m - 1)
        variance = within * Decimal(n - 1) / n + between / n
        return centered, within, variance

    def ess(x):
        centered, within, variance = moments(x)
        m, n = len(centered), len(centered[0])
        if variance <= 0:
            return 0.0  # Explicit fail-closed constant-indicator policy.
        correlations = [Decimal(1)]
        for lag in range(1, n):
            covariance = (
                sum(
                    sum(a * b for a, b in zip(row[:-lag], row[lag:], strict=True)) / n
                    for row in centered
                )
                / m
            )
            correlations.append(1 - (within - covariance) / variance)
        positive = takewhile(
            lambda pair: pair > 0,
            (a + b for a, b in zip(correlations[::2], correlations[1::2])),
        )
        monotone = list(accumulate(positive, min))
        if not monotone:
            return float(m * n)
        tau = max(-1 + 2 * sum(monotone), 1 / Decimal(m * n).log10())
        return float(Decimal(m * n) / tau)

    with localcontext() as context:
        context.prec = 60
        scores = normal_scores(raw)
        folded_scores = normal_scores(np.abs(raw - np.median(raw)))
        ratios = []
        for values in (scores, folded_scores):
            _, within, variance = moments(values)
            ratios.append(float((variance / within).sqrt()))
        quantiles = np.quantile(raw, [0.05, 0.95])
        return {
            "rhat_rank": max(ratios),
            "ess_bulk": ess(scores),
            "ess_tail": min(ess(raw <= quantile) for quantile in quantiles),
        }


def independent_cases(document: dict) -> list[dict]:
    cases = [
        {key: case[key] for key in ("name", "chains", "draws", "samples_chain_major")}
        for case in document["scenarios"]
    ]
    # The 19-valued cycling background has a persistent rare upper-tail block.
    # Its 95% quantile is tied, exposing >=q versus the paper's <=q indicator.
    cases.append(
        {
            "name": "tied_upper_tail",
            "chains": 4,
            "draws": 128,
            "samples_chain_major": [
                float(19 if 20 * c <= i < 20 * c + 6 else i % 19)
                for c in range(4)
                for i in range(128)
            ],
        }
    )
    cases.append(
        {
            "name": "odd_draws_drop_last",
            "chains": 3,
            "draws": 129,
            "samples_chain_major": [
                float(np.sin(0.19 * i + 0.6 * c) + 0.4 * np.cos(0.07 * i + c))
                for c in range(3)
                for i in range(129)
            ],
        }
    )
    return cases


def build_primary_fixture(document: dict) -> dict:
    cases = independent_cases(document)
    for case in cases:
        case["expected"] = primary_outputs(case)
        # Retain the public comparator's actual output, even where it differs.
        case["arviz_raw"] = outputs(case)
        case["comparison"] = (
            "Distinct finite-lag/odd-split conventions; agreement is scenario dependent. Compare the full-pair primary target separately."
        )
    return {
        "schema_version": 1,
        "oracle": {
            "kind": "independent_reimplementation",
            "generator": "conformance/validate/bayesian_checks/reference.py",
            "method": "Vehtari 2021 sections 3.1-4.3; full initial positive/monotone pair sums including the final complete pair, without odd/even truncation averaging; 60-digit Decimal covariance arithmetic and public SciPy normal scores.",
            "limitations": "Marginal scalar diagnostics on fixed chains. Drop the last draw for odd lengths. Constant tail indicators fail closed with zero ESS. No convergence proof or posterior/calibration license.",
            "command": "uv run --frozen --script conformance/validate/bayesian_checks/reference.py --check-ips",
            "environment_lock": "conformance/validate/bayesian_checks/reference.py.lock",
        },
        "tolerance": {"rhat_atol": 1e-6, "ess_atol": 1e-6},
        "scenarios": cases,
    }


def check_primary(document: dict, source: dict) -> list[str]:
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
        if values_close(payload, build_primary_fixture(source), atol=5e-10, rtol=0)
        else [
            "independent IPS fixture differs from regenerated inputs, primary-equation outputs or scope"
        ]
    )


def check(document: dict) -> list[str]:
    problems = []
    cases = document["scenarios"]
    if len(cases) != 3 or {c["name"] for c in cases} != {
        "iid",
        "shifted_chain",
        "autocorrelated",
    }:
        return ["rank diagnostics scenario grid changed or empty"]
    if document["oracle"]["version"] != "0.20.0":
        problems.append("rank diagnostics comparator pin changed")
    for case in cases:
        if (
            case["chains"] != 4
            or case["draws"] != 128
            or len(case["samples_chain_major"]) != 512
        ):
            problems.append(f"{case['name']}: chain/draw shape changed")
            continue
        if not values_close(case["arviz"], outputs(case), atol=5e-10, rtol=0):
            problems.append(f"{case['name']}: diagnostics differ from pinned ArviZ")
    # Regeneration tolerance is independent of the product's broader ESS budget.
    if document.get("regeneration_tolerance", {"atol": 5e-10, "rtol": 0}) != {
        "atol": 5e-10,
        "rtol": 0,
    }:
        problems.append("rank diagnostics regeneration tolerance changed")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--write-ips", action="store_true")
    mode.add_argument("--check-ips", action="store_true")
    mode.add_argument("--self-test-ips", action="store_true")
    args = parser.parse_args()
    for package, pin in {
        "arviz": "0.20.0",
        "numpy": "2.1.3",
        "scipy": "1.14.1",
        "pandas": "2.2.3",
        "xarray": "2024.11.0",
        "matplotlib": "3.9.2",
    }.items():
        if importlib.metadata.version(package) != pin:
            raise SystemExit(
                f"{package} must be {pin}; use the frozen script environment"
            )
    document = json.loads(
        Path(__file__).with_name("diagnostics_oracle.json").read_text(encoding="utf-8")
    )
    if args.write_ips:
        Path(__file__).with_name("independent_ips.json").write_text(
            json.dumps(build_primary_fixture(document), indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            "Generated independent full-pair diagnostics for five scenarios; raw ArviZ outputs retained"
        )
        return 0
    if args.check_ips or args.self_test_ips:
        independent = json.loads(
            Path(__file__).with_name("independent_ips.json").read_text(encoding="utf-8")
        )
        problems = check_primary(independent, document)
        if args.self_test_ips and not problems:
            mutations = []
            changed = copy.deepcopy(independent)
            changed["scenarios"][0]["expected"]["ess_bulk"] += 1.0
            mutations.append(changed)
            changed = copy.deepcopy(independent)
            changed["scenarios"][0]["samples_chain_major"][0] += 0.5
            mutations.append(changed)
            changed = copy.deepcopy(independent)
            changed["scenarios"] = []
            mutations.append(changed)
            changed = copy.deepcopy(independent)
            changed["tolerance"]["ess_atol"] = 1.0
            mutations.append(changed)
            changed = copy.deepcopy(independent)
            changed["oracle"]["kind"] = "regression_pin"
            mutations.append(changed)
            for index, changed in enumerate(mutations):
                if not check_primary(changed, document):
                    problems.append(f"IPS fault control {index} passed incorrectly")
        for problem in problems:
            print(f"FAIL: {problem}")
        if not problems:
            print("Five high-precision primary-equation scenarios reproduced")
        return int(bool(problems))
    problems = check(document)
    if args.self_test and not problems:
        mutations = []
        changed = copy.deepcopy(document)
        changed["scenarios"][0]["arviz"]["rhat_rank"] += 0.1
        mutations.append(changed)
        changed = copy.deepcopy(document)
        changed["scenarios"] = []
        mutations.append(changed)
        changed = copy.deepcopy(document)
        changed["scenarios"][0]["chains"] = 2
        mutations.append(changed)
        changed = copy.deepcopy(document)
        changed["regeneration_tolerance"] = {"atol": 1.0, "rtol": 0}
        mutations.append(changed)
        for index, changed in enumerate(mutations):
            if not check(changed):
                problems.append(f"fault control {index} passed incorrectly")
    for problem in problems:
        print(f"FAIL: {problem}")
    if not problems:
        print("ArviZ: all three frozen chain scenarios and nine diagnostics reproduced")
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
