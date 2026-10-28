# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the front-door functional fixture: the front-door adjustment sum_m P(m | t) sum_t' P(t') P(y | t', m) evaluated with rational arithmetic on the fully enumerated binary table gives the stated average effect 0.282.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/estimate/admg_frontdoor_functional/reference.py --check
    uv run --frozen --script conformance/estimate/admg_frontdoor_functional/reference.py --self-test

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


def close(got, want, tol, label, problems):
    if got is None or want is None or not math.isclose(float(got), float(want), rel_tol=0.0, abs_tol=tol):
        problems.append(f"{label}: derived {float(got) if got is not None else got!r} != fixture {want!r} (tol {tol})")


def same(got, want, label, problems):
    if got != want:
        problems.append(f"{label}: derived {got!r} != fixture {want!r}")

def check(document):
    problems = []
    n = {(int(c["t"]), int(c["m"]), int(c["y"])): F(c["count"]) for c in document["contingency_table"]}
    total = sum(n.values())
    def pt(t): return sum(v for (a, _, _), v in n.items() if a == t) / total
    def pm_t(m, t): return sum(v for (a, b, _), v in n.items() if a == t and b == m) / sum(v for (a, _, _), v in n.items() if a == t)
    def py(t, m): return n[(t, m, 1)] / (n[(t, m, 0)] + n[(t, m, 1)])
    def mean(t): return sum(pm_t(m, t) * sum(pt(tt) * py(tt, m) for tt in (0, 1)) for m in (0, 1))
    ate = mean(int(document["query"]["active_level"])) - mean(int(document["query"]["control_level"]))
    close(ate, document["frequentist"]["expected_ate"], document["frequentist"]["absolute_tolerance"], "expected_ate", problems)
    if ate != F(282, 1000):
        problems.append(f"exact front-door effect {ate} != 141/500")
    return problems


MUTATIONS = [
    ("effect changed", lambda d: d["frequentist"].__setitem__("expected_ate", 0.3)),
    ("count changed", lambda d: d["contingency_table"][7].__setitem__("count", 200)),
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
