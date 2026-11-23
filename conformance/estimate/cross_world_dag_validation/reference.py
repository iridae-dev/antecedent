# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the cross-world refuter fixture from the structural equations alone.

Counterfactual: the outcome equation y = 1 + 2 t + z moves every unit's outcome by its treatment
coefficient times the contrast, so the mean unit effect of setting t to 1.5 against 0 is 2 * 1.5.

Nested natural direct effect: the mediator is m = 0.8 t + e_m and the outcome is
y = 0.3 t + 1.2 m + u. When the outcome error u carries a coupling c with e_m (u = s (c a + sqrt(1-c^2) b),
e_m = s_m a, a and b unit-variance and uncorrelated), the regression of y on (t, m) reads the
direct coefficient 0.3 - kappa * 0.8 with kappa = cov(u, e_m) / var(e_m): the sequential-ignorability
read of the direct effect, which differs from the structural 0.3 exactly when c is not zero. Rational
arithmetic throughout; standard library only; imports no Antecedent code.

    uv run --frozen --script conformance/estimate/cross_world_dag_validation/reference.py --check
    uv run --frozen --script conformance/estimate/cross_world_dag_validation/reference.py --self-test

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
F = Fraction
FIXTURE = HERE / "expected.json"


def frac(text):
    return F(str(text))



def close(got, want, label, problems):
    if not math.isclose(float(got), float(want), rel_tol=0.0, abs_tol=1e-12):
        problems.append(f"{label}: derived {float(got)!r} != fixture {want!r}")


def check(document):
    problems = []
    cf = document["counterfactual"]
    slope = frac(cf["structural_equation"]["coefficients"]["t"])
    contrast = frac(cf["active_level"]) - frac(cf["control_level"])
    close(slope * contrast, cf["mean_unit_effect"], "counterfactual.mean_unit_effect", problems)
    close(F(0) * contrast, cf["null_slope_mean_unit_effect"], "counterfactual.null_slope_mean_unit_effect", problems)
    nested = document["nested"]
    direct = frac(nested["outcome_equation"]["t"])
    through_m = frac(nested["outcome_equation"]["m"])
    a_tm = frac(nested["mediator_equation"]["t"])
    s_m = frac(nested["mediator_error_scale"])
    s_u = frac(nested["outcome_error_scale"])
    coupling = frac(nested["error_coupling"])
    close(direct, nested["structural_direct_effect"], "nested.structural_direct_effect", problems)
    close(direct + through_m * a_tm, nested["total_effect"], "nested.total_effect", problems)
    kappa = (s_u * coupling * s_m) / (s_m * s_m)
    close(direct - kappa * a_tm, nested["shared_exogenous_read_under_coupling"], "nested.shared_exogenous_read_under_coupling", problems)
    return problems


MUTATIONS = [
    ("counterfactual effect changed", lambda d: d["counterfactual"].__setitem__("mean_unit_effect", 2.0)),
    ("coupling read changed", lambda d: d["nested"].__setitem__("shared_exogenous_read_under_coupling", 0.3)),
    ("total effect changed", lambda d: d["nested"].__setitem__("total_effect", 0.3)),
]


def self_test(document):
    failures = []
    if check(document):
        failures.append("the unmodified fixture is rejected")
    for label, mutate in MUTATIONS:
        wrong = copy.deepcopy(document)
        mutate(wrong)
        if not check(wrong):
            failures.append(f"accepted: {label}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    problems = self_test(document) if args.self_test else check(document)
    for problem in problems:
        print("FAIL", problem)
    if not problems:
        print("ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
