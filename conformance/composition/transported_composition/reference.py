# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Exact moments of a composed transported claim.

A transported claim has three pieces: source sampling S, target sampling T and the transport
functional F. Composed as a weighted sum c_S S + c_T T + c_F F, the pieces are dependent when
evidence is shared, so the composed moments need their joint law. Two joint laws give them:

* a supplied covariance Sigma of the pieces and their means m:
      mean = c . m          variance = c' Sigma c           (every cross term kept);
* the aligned draws of one joint law (row i holds one realization of all three pieces):
      mean = (1/n) sum_i r_i   variance = (1/n) sum_i (r_i - mean)^2,  r_i = c . (S_i, T_i, F_i)
  (population variance of the aligned row combinations).

Both are evaluated in exact rational arithmetic from decimal literals and rounded to float once.
Independence between the pieces is never assumed, so a transported composition declared
independent, or with unknown dependence, has no value: the product refuses it. No Antecedent
code is imported.

    uv run --frozen --script conformance/composition/transported_composition/reference.py --check
    uv run --frozen --script conformance/composition/transported_composition/reference.py --write

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED = HERE / "expected.json"
RTOL = 1e-12


def frac(x) -> Fraction:
    return Fraction(str(x))


def covariance_route(case: dict) -> dict:
    c = [frac(v) for v in case["coefficients"]]
    m = [frac(v) for v in case["means"]]
    s = [frac(v) for v in case["covariance"]]
    k = len(c)
    return {
        "mean": float(sum(ci * mi for ci, mi in zip(c, m))),
        "variance": float(sum(c[i] * s[i * k + j] * c[j] for i in range(k) for j in range(k))),
    }


def aligned_route(case: dict) -> dict:
    c = [frac(v) for v in case["coefficients"]]
    columns = [[frac(v) for v in column] for column in case["pieces"]]
    n = len(columns[0])
    rows = [sum(ci * column[i] for ci, column in zip(c, columns)) for i in range(n)]
    mean = sum(rows) / n
    variance = sum((r - mean) ** 2 for r in rows) / n
    return {"mean": float(mean), "variance": float(variance)}


def close(a: float, b: float) -> bool:
    return abs(a - b) <= RTOL * max(1.0, abs(a), abs(b))


def check(document: dict) -> list[str]:
    problems = []
    for key, solve in (("covariance_route", covariance_route), ("aligned_route", aligned_route)):
        case = document[key]
        for name, value in solve(case).items():
            if not close(value, case["expected"][name]):
                problems.append(f"{key}: {name} {case['expected'][name]} vs {value}")
    return problems


def self_test(document: dict) -> list[str]:
    failures = []
    for key in ("covariance_route", "aligned_route"):
        for name in ("mean", "variance"):
            wrong = copy.deepcopy(document)
            wrong[key]["expected"][name] += 1e-6
            if not check(wrong):
                failures.append(f"{key}: a perturbed {name} was accepted")
    # Dropping the cross terms (assuming independence) must change the variance.
    case = document["covariance_route"]
    k = len(case["coefficients"])
    diagonal = copy.deepcopy(case)
    diagonal["covariance"] = [
        v if i // k == i % k else "0" for i, v in enumerate(case["covariance"])
    ]
    if close(covariance_route(diagonal)["variance"], case["expected"]["variance"]):
        failures.append("the covariance route does not depend on the cross terms")
    return failures


def gram(rows: list[list[str]], ridge: str) -> list[str]:
    n = len(rows[0])
    out = []
    for i in range(n):
        for j in range(n):
            v = sum(frac(r[i]) * frac(r[j]) for r in rows) + (frac(ridge) if i == j else 0)
            out.append(format(Decimal(v.numerator) / Decimal(v.denominator), "f"))
    return out


def build() -> dict:
    covariance = {
        "coefficients": ["1", "2", "-0.5"],
        "means": ["1.5", "-0.25", "3"],
        "covariance": gram(
            [["0.5", "0.25", "-0.5"], ["0.25", "-0.5", "0.5"], ["0", "0.5", "0.25"]], "0.125"
        ),
    }
    aligned = {
        "coefficients": ["1", "1", "1"],
        "pieces": [
            ["1", "2", "3", "4", "5", "6", "7", "8"],
            ["2", "1", "4", "3", "6", "5", "8", "7"],
            ["0", "0", "1", "2", "-1", "0", "2", "1"],
        ],
    }
    aligned["coefficients"] = ["1", "0.5", "-2"]
    covariance["expected"] = covariance_route(covariance)
    aligned["expected"] = aligned_route(aligned)
    return {
        "schema_version": 2,
        "scope": "Moments of a transported claim composed from its source-sampling, target-sampling and transport-functional pieces under a supplied covariance and under the aligned draws of one joint law; independence and unknown dependence are refused.",
        "tolerance": {"relative": RTOL},
        "refused_without_dependence": ["unknown", "declared_independent"],
        "covariance_route": covariance,
        "aligned_route": aligned,
        "oracle": {
            "kind": "closed_form",
            "project": "exact rational arithmetic of linear-combination moments",
            "method": "mean c.m and variance c'Sigma c from a supplied covariance; population mean and variance of the aligned row combinations from one joint law",
            "generator": "conformance/composition/transported_composition/reference.py",
            "command": "uv run --frozen --script conformance/composition/transported_composition/reference.py --check",
            "environment_lock": "conformance/composition/transported_composition/reference.py.lock",
            "generator_retained": True,
            "limitations": "Hand-constructed three-piece covariance and eight aligned rows; point moments only, no interval or coverage claim.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true")
    group.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.write:
        fresh = build()
        if EXPECTED.exists():
            review = json.loads(EXPECTED.read_text(encoding="utf-8")).get("oracle", {}).get("reference_review")
            if review is not None:
                fresh["oracle"]["reference_review"] = review
        EXPECTED.write_text(json.dumps(fresh, indent=2) + "\n", encoding="utf-8")
        return 0
    document = json.loads(EXPECTED.read_text(encoding="utf-8"))
    problems = check(document) + self_test(document)
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        return 1
    print("covariance and aligned routes agree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
