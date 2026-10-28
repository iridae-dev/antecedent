# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the mediation effects behind the staged_static_kinds fixture: with m = 2a + e and y = 3a + 4m + noise, moving a from 0.2 to 0.8 gives natural direct 3*0.6 = 1.8, natural indirect 4*2*0.6 = 4.8 and total 6.6, evaluated by substituting into the structural equations at several disturbance values.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/estimate/staged_static_kinds/reference.py --check
    uv run --frozen --script conformance/estimate/staged_static_kinds/reference.py --self-test

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


def close(got, want, tol, label, problems):
    if got is None or want is None or not math.isclose(float(got), want, rel_tol=0.0, abs_tol=tol):
        problems.append(f"{label}: derived {got!r} != fixture {want!r} (tol {tol})")


F_M = lambda a, e: 2 * a + e
F_Y = lambda a, m: 3 * a + 4 * m

def derived(control, active):
    out = None
    for e in (Fraction(-3), Fraction(0, 1), Fraction(1, 7), Fraction(5, 3)):
        m_c, m_a = F_M(control, e), F_M(active, e)
        nde = F_Y(active, m_c) - F_Y(control, m_c)
        nie = F_Y(control, m_a) - F_Y(control, m_c)
        total = F_Y(active, m_a) - F_Y(control, m_c)
        here = (nde, nie, total)
        if out is not None and here != out:
            raise SystemExit("effects depend on the disturbance")
        out = here
    return out

def check(document):
    problems = []
    control, active = Fraction(str(document["control"])), Fraction(str(document["active"]))
    nde, nie, total = derived(control, active)
    close(nde, document["direct"], 1e-12, "direct", problems)
    close(nie, document["indirect"], 1e-12, "indirect", problems)
    close(total, document["total"], 1e-12, "total", problems)
    close(total, document["counterfactual_mean"], 1e-12, "counterfactual_mean", problems)
    if not 0 < document["tolerance"] < min(float(nde), float(nie)):
        problems.append("tolerance is not a positive band narrower than the smallest effect")
    return problems

MUTATIONS = [
    ("direct shifted by 0.5", lambda d: d.__setitem__("direct", d["direct"] + 0.5)),
    ("indirect swapped with direct", lambda d: d.__setitem__("indirect", d["direct"])),
    ("counterfactual mean shifted", lambda d: d.__setitem__("counterfactual_mean", 6.0)),
    ("active level moved", lambda d: d.__setitem__("active", 0.9)),
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
    document = json.loads((HERE / "expected.json").read_text(encoding="utf-8"))
    problems = self_test(document) if args.self_test else check(document)
    for problem in problems:
        print("FAIL", problem)
    if not problems:
        print("ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
