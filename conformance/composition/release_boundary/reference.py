# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Exact rational reference for the numeric chain of the release-boundary fixture.

Recomputes, from the declared scenario inputs only, the misclassification-corrected
prevalence, the inverse-variance and covariance-aware (generalized least squares) pools,
the row-overlap counts, the composed finite-model risks, the attributable excess risk, the
aligned product-lattice mean and the decision utilities before and after the study revision.
Everything is exact `fractions.Fraction` arithmetic; no Antecedent code, no estimator.

The refusal codes, mutation, conversion and variant tables in the fixture are product
contract strings, not numerical propositions; this script does not derive or compare them.

    uv run --frozen --script conformance/composition/release_boundary/reference.py --check
    uv run --frozen --script conformance/composition/release_boundary/reference.py --self-test

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from fractions import Fraction as F
from pathlib import Path

HERE = Path(__file__).resolve().parent


def fr(x) -> F:
    return F(str(x))


def pool(items: list[tuple[F, F]], covariance: F = F(0)) -> tuple[F, F]:
    """GLS pool of estimates with a 2x2 covariance (diagonal unless two items)."""
    if len(items) == 1:
        return items[0]
    if covariance == 0:
        weights = [1 / v for _, v in items]
        total = sum(weights)
        return sum(w * e for w, (e, _) in zip(weights, items)) / total, 1 / total
    (e1, v1), (e2, v2) = items
    det = v1 * v2 - covariance**2
    inverse = [[v2 / det, -covariance / det], [-covariance / det, v1 / det]]
    row = [inverse[0][0] + inverse[1][0], inverse[0][1] + inverse[1][1]]
    total = sum(row)
    return (row[0] * e1 + row[1] * e2) / total, 1 / total


def rows(span) -> set[int]:
    return set(range(span[0], span[1]))


def compute(document: dict) -> dict:
    sc = document["scenario"]
    m = sc["measurement"]
    se, sp = fr(m["sensitivity"]), fr(m["specificity"])
    p_star = F(m["events"], m["total"])
    youden = se + sp - 1
    prevalence = (p_star + sp - 1) / youden
    base = fr(sc["baseline_risk"])
    studies = sc["studies"]
    a, b = studies[0], studies[1]
    effect, variance = pool([(fr(a["estimate"]), fr(a["variance"])), (fr(b["estimate"]), fr(b["variance"]))])
    slots = sum(len(rows(s["rows"])) for s in (a, b))
    distinct = len(rows(a["rows"]) | rows(b["rows"]))

    def risks(d: F) -> dict:
        exposed = base + d
        mean = (1 - prevalence) * base + prevalence * exposed
        return {"exposed": exposed, "mean": mean, "excess": mean - base}

    def utility(excess: F) -> F:
        return fr(sc["decision"]["benefit_per_unit_risk"]) * excess - fr(sc["decision"]["programme_cost"])

    now = risks(effect)
    c = sc["overlapping_study"]
    overlap, overlap_variance = pool(
        [(fr(a["estimate"]), fr(a["variance"])), (fr(c["estimate"]), fr(c["variance"]))],
        fr(c["covariance"]),
    )
    rev = sc["revision"]
    rev_effect, rev_variance = pool(
        [(fr(a["estimate"]), fr(a["variance"])), (fr(rev["estimate"]), fr(rev["variance"]))]
    )
    after = risks(rev_effect)
    draws = sc["marginal_draws"]
    lattice = [fr(p) * fr(d) for p in draws["prevalence"] for d in draws["effect"]]
    values = {
        "prevalence": prevalence,
        "naive_prevalence": p_star,
        "pooled_effect": effect,
        "pooled_variance": variance,
        "declared_slots": slots,
        "distinct_observations": distinct,
        "outcome_risk_exposed": now["exposed"],
        "outcome_risk_unexposed": base,
        "outcome_risk": now["mean"],
        "excess_risk": now["excess"],
        "act": utility(now["excess"]),
        "hold": F(0),
        "selected": "act" if utility(now["excess"]) > 0 else "hold",
        "aligned_product_mean": sum(lattice) / len(lattice),
        "overlap_pooled_estimate": overlap,
        "overlap_pooled_variance": overlap_variance,
        "overlap_declared_slots": len(rows(a["rows"])) + len(rows(c["rows"])),
        "overlap_distinct_observations": len(rows(a["rows"]) | rows(c["rows"])),
    }
    revised = {
        "pooled_effect": rev_effect,
        "pooled_variance": rev_variance,
        "outcome_risk_exposed": after["exposed"],
        "outcome_risk": after["mean"],
        "excess_risk": after["excess"],
        "act": utility(after["excess"]),
        "hold": F(0),
        "selected": "act" if utility(after["excess"]) > 0 else "hold",
    }
    return {"values": values, "revised_values": revised}


def check(document: dict) -> list[str]:
    problems = []
    got = compute(document)
    tol = document["tolerance"]["exact"]
    for section in ("values", "revised_values"):
        for key, reference in got[section].items():
            fixture = document[section].get(key)
            if fixture is None:
                problems.append(f"{section}.{key} missing from fixture")
            elif isinstance(reference, str):
                if fixture != reference:
                    problems.append(f"{section}.{key}: {fixture} != {reference}")
            elif abs(float(fr(fixture) - reference)) > tol:
                problems.append(f"{section}.{key}: fixture {fixture} reference {float(reference)}")
        for key in document[section]:
            if key not in got[section]:
                problems.append(f"{section}.{key} has no reference value")
    return problems


def self_test(document: dict) -> list[str]:
    failures = []
    for section, key in (
        ("values", "prevalence"),
        ("values", "overlap_pooled_variance"),
        ("revised_values", "act"),
    ):
        wrong = copy.deepcopy(document)
        wrong[section][key] += 1e-6
        if not check(wrong):
            failures.append(f"shifted {section}.{key} accepted")
    wrong = copy.deepcopy(document)
    wrong["scenario"]["measurement"]["events"] = 411
    if not check(wrong):
        failures.append("changed event count accepted")
    wrong = copy.deepcopy(document)
    wrong["values"]["selected"] = "hold"
    if not check(wrong):
        failures.append("changed selected action accepted")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    document = json.loads((HERE / "expected.json").read_text(encoding="utf-8"))
    problems = self_test(document) if args.self_test else check(document)
    for problem in problems:
        print("FAIL", problem)
    if not problems:
        print("ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
