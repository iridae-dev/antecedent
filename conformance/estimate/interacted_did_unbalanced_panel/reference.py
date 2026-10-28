# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.1.3"]
# ///
"""Independent reference for the interacted (Sun-Abraham) staggered DiD on an unbalanced panel.

Known data-generating process (cohorts 0, 3, 5 of 40 units each, periods 1..6):

    y_it = alpha_i + lambda_t + tau[g, t - g] 1[g > 0, t >= g] + e_it,   e_it ~ N(0, 0.05^2)

with unit effects alpha_i ~ N(0, 1), period effects lambda_t = 0.5 t + 0.3 (t mod 2), and
known cohort-specific effects tau[3, 0..3] = 1.0, 2.0, 3.0, 3.5 and tau[5, 0..1] = 2.0, 2.5.

The panel is unbalanced by an observation process that depends on cohort only (attrition
ignorable given cohort): in periods 1..5 each unit is observed with probability 0.8 (0.7 in
cohort 5), topped up with its earliest missing periods so that every unit has at least
three observed rows; period 6 is observed only for treated cohorts (probability 0.8), so no
comparison unit exists at period 6 and the treated rows at period 6 have no support.

The estimand is the cohort-by-period coefficient of the saturated regression

    y_it = a_i + l_t + sum_{g, t != g - 1} delta[g, t] 1[cohort_i = g, period = t] + e_it

on the observed unit-periods that have a comparison unit at their period. The reference
builds the full dummy design (a dummy per unit, period and cohort-period cell) and solves it
with numpy.linalg.lstsq, after checking that the design has full column rank. It then
aggregates with the cohort sizes: the simple average over post-adoption cells and the
pre-adoption leads by event time. Nothing here uses a within transformation or the
production solver.

Run: ``uv run --frozen --script conformance/estimate/interacted_did_unbalanced_panel/reference.py``
(add ``--write`` to regenerate expected.json, ``--check`` to compare).

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
COHORTS = (0, 3, 5)
UNITS_PER_COHORT = 40
PERIODS = (1, 2, 3, 4, 5, 6)
TAU = {3: [1.0, 2.0, 3.0, 3.5], 5: [2.0, 2.5]}
NOISE_SD = 0.05


def lam(period: int) -> float:
    return 0.5 * period + 0.3 * (period % 2)


def truth(cohort: int, period: int) -> float:
    if cohort == 0 or period < cohort:
        return 0.0
    return TAU[cohort][period - cohort]


def simulate() -> dict[str, list]:
    rng = np.random.default_rng(20261027)
    unit, period, cohort, outcome = [], [], [], []
    for g in COHORTS:
        keep_probability = 0.7 if g == 5 else 0.8
        for i in range(UNITS_PER_COHORT):
            label = f"u{g}_{i:02d}"
            alpha = rng.normal()
            seen = [t for t in range(1, 6) if rng.uniform() < keep_probability]
            for t in range(1, 6):
                if len(seen) >= 3:
                    break
                if t not in seen:
                    seen.append(t)
            if g > 0 and rng.uniform() < 0.8:
                seen.append(6)
            for t in sorted(seen):
                unit.append(label)
                period.append(t)
                cohort.append(g)
                outcome.append(alpha + lam(t) + truth(g, t) + NOISE_SD * rng.normal())
    return {"unit": unit, "period": period, "cohort": cohort, "outcome": outcome}


def saturated_regression(rows: dict[str, list]) -> dict:
    unit = np.array(rows["unit"])
    period = np.array(rows["period"])
    cohort = np.array(rows["cohort"])
    y = np.array(rows["outcome"])
    controls_at = {t for t in PERIODS if np.any((cohort == 0) & (period == t))}
    supported = np.array([c == 0 or t in controls_at for c, t in zip(cohort, period)])
    dropped_rows = {int(g): int(np.sum((cohort == g) & ~supported)) for g in COHORTS if g > 0}
    unit, period, cohort, y = unit[supported], period[supported], cohort[supported], y[supported]
    labels = sorted(set(unit))
    columns = [(unit == label).astype(float) for label in labels]
    names = [("unit", label) for label in labels]
    for t in sorted(set(period))[1:]:
        columns.append((period == t).astype(float))
        names.append(("period", t))
    cells = []
    for g in COHORTS[1:]:
        for t in sorted(set(period)):
            if t == g - 1:
                continue
            columns.append(((cohort == g) & (period == t)).astype(float))
            names.append(("cell", (g, int(t))))
            cells.append((g, int(t)))
    design = np.column_stack(columns)
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise SystemExit("the saturated design is rank deficient")
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    delta = {name[1]: float(b) for name, b in zip(names, beta) if name[0] == "cell"}
    sizes = {g: UNITS_PER_COHORT for g in COHORTS if g > 0}
    post = [(g, t) for (g, t) in cells if t >= g]
    simple = sum(sizes[g] * delta[(g, t)] for g, t in post) / sum(sizes[g] for g, _ in post)
    leads = {}
    for event in sorted({t - g for (g, t) in cells if t < g}):
        members = [(g, t) for (g, t) in cells if t - g == event]
        leads[str(event)] = sum(sizes[g] * delta[(g, t)] for g, t in members) / sum(
            sizes[g] for g, _ in members
        )
    treated = {
        f"{g}_{t}": int(len({u for u, c, p in zip(unit, cohort, period) if c == g and p == t}))
        for (g, t) in cells
    }
    return {
        "cells": [
            {
                "cohort": g,
                "period": t,
                "estimate": delta[(g, t)],
                "truth": truth(g, t),
                "treated_units": treated[f"{g}_{t}"],
                "comparison_units": int(np.sum((cohort == 0) & (period == t))),
            }
            for (g, t) in cells
        ],
        "simple": simple,
        "simple_truth": sum(sizes[g] * truth(g, t) for g, t in post)
        / sum(sizes[g] for g, _ in post),
        "leads_by_event": leads,
        "dropped_rows_by_cohort": {str(g): n for g, n in dropped_rows.items()},
        "periods_without_comparison": [t for t in PERIODS if t not in controls_at],
    }


ORACLE = {
    "kind": "independent_reimplementation",
    "method": "full-dummy saturated cohort-by-period regression on the observed unit-periods by numpy lstsq",
    "independence_basis": "Simulated unbalanced panel with known cohort-by-event-time effects; the reference builds one dummy per unit, period and cohort-period cell and solves it with numpy.linalg.lstsq after a full-column-rank check, using no within transformation, union-find support screening or production solver. The retained generator and the executing assertion were read for the review recorded in reference_review.",
    "generator": "conformance/estimate/interacted_did_unbalanced_panel/reference.py",
    "generator_retained": True,
    "environment_lock": "conformance/estimate/interacted_did_unbalanced_panel/reference.py.lock",
    "command": "uv run --frozen --script conformance/estimate/interacted_did_unbalanced_panel/reference.py --check",
    "asserted_outputs": [
        "reference.cells[*].estimate, treated_units and comparison_units for post-adoption cells",
        "reference.leads_by_event",
        "reference.simple and reference.simple_truth (0.1 band)",
        "reference.dropped_rows_by_cohort",
        "reference.periods_without_comparison",
    ],
    "context_only_outputs": ["reference.cells[*].truth for pre-adoption cells", "rows (inputs)"],
    "historical_exposure": "Historical author exposure is unrecorded and remains unknown.",
}


REVIEW_OWNED = ("reference_review", "executing_assertion")


def committed_review() -> dict:
    """The reviewed-artifact record and the assertion path live in the committed fixture."""
    path = HERE / "expected.json"
    if not path.is_file():
        return {}
    oracle = json.loads(path.read_text(encoding="utf-8")).get("oracle", {})
    return {k: oracle[k] for k in REVIEW_OWNED if k in oracle}


def build() -> dict:
    rows = simulate()
    oracle = {**ORACLE, **committed_review()}
    return {
        "oracle": oracle,
        "schema_version": 1,
        "case": "interacted_did_unbalanced_panel",
        "rows": rows,
        "reference": saturated_regression(rows),
    }


def close(left: object, right: object, atol: float = 1e-9) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(close(left[k], right[k], atol) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(close(a, b, atol) for a, b in zip(left, right))
    if isinstance(left, float) or isinstance(right, float):
        return abs(left - right) <= atol
    return left == right


def main() -> None:
    expected = build()
    path = HERE / "expected.json"
    text = json.dumps(expected, indent=1) + "\n"
    if "--write" in sys.argv:
        path.write_text(text, encoding="utf-8", newline="\n")
    if "--check" in sys.argv:
        frozen = json.loads(path.read_text(encoding="utf-8"))
        if not close(frozen, json.loads(text)):
            raise SystemExit("expected.json is stale; rerun with --write")
    reference = expected["reference"]
    print(json.dumps({k: reference[k] for k in ("simple", "simple_truth", "leads_by_event")}, indent=2))
    print("max |estimate - truth| over supported cells:",
          max(abs(c["estimate"] - c["truth"]) for c in reference["cells"]))


if __name__ == "__main__":
    main()
