# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Exact enumeration reference for the asserted static recalculation pins.

Covers the three pins the executing static-recalculation test reads from `expected.json`:

* `frontdoor_discrete_contrast`: the front-door adjustment
  sum_m [P(m|t=1) - P(m|t=0)] sum_t' P(t') E[Y|m,t'] evaluated in exact rational arithmetic
  on the retained 3-column contingency table of the front-door functional fixture;
* `verma_do_means` and `verma_contrast`: E[X4 | do(X2=x)] for the latent-variable binary
  SCM  U ~ Bern(.3), X1 ~ Bern(.5), X2 ~ Bern(.1 + .2 X1 + .4 U), X3 ~ Bern(.25 + .5 X2),
  X4 ~ Bern(.15 + .35 X3 + .3 U), by enumeration over U with the structural equations (the
  truth is the SCM's own intervention, not a Verma-functional evaluation of data).  The
  script also checks that the 16000-row population table the test builds from this SCM has
  integer cell counts, so no rounding enters the data.

The remaining fields (`iv_effect`, `rd_jump`, `rd_hc1_variance`, `frontdoor_linear_effect`,
`frontdoor_hc0_variance`) are not read from this file by any executing assertion; the
design test hard-codes its own literals.  They are not derived or compared here.

    uv run --frozen --script conformance/recalculation/checked_static_design/reference.py --check
    uv run --frozen --script conformance/recalculation/checked_static_design/reference.py --self-test

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
from fractions import Fraction as F
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TABLE = ROOT / "conformance" / "estimate" / "admg_frontdoor_functional" / "expected.json"
ATOL = 1e-12


def frontdoor_contrast() -> F:
    table = json.loads(TABLE.read_text(encoding="utf-8"))["contingency_table"]
    n = {(int(c["t"]), int(c["m"]), int(c["y"])): F(c["count"]) for c in table}
    total = sum(n.values())
    cell = lambda t=None, m=None, y=None: sum(
        v for (a, b, c), v in n.items() if (t is None or a == t) and (m is None or b == m) and (y is None or c == y)
    )

    def do(t):
        out = F(0)
        for m in (0, 1):
            p_m = cell(t=t, m=m) / cell(t=t)
            inner = sum(
                cell(t=tp) / total * (cell(t=tp, m=m, y=1) / cell(t=tp, m=m)) for tp in (0, 1)
            )
            out += p_m * inner
        return out

    return do(1) - do(0)


def verma():
    def p(bit, prob):
        return prob if bit else 1 - prob

    means = []
    for x2 in (0, 1):
        total = F(0)
        for u, x3, x4 in itertools.product((0, 1), repeat=3):
            pr = (
                p(u, F(3, 10))
                * p(x3, F(1, 4) + F(1, 2) * x2)
                * p(x4, F(3, 20) + F(7, 20) * x3 + F(3, 10) * u)
            )
            total += pr * x4
        means.append(total)
    integral = True
    for u, x1, x2, x3, x4 in itertools.product((0, 1), repeat=5):
        pr = (
            p(u, F(3, 10))
            * p(x1, F(1, 2))
            * p(x2, F(1, 10) + F(1, 5) * x1 + F(2, 5) * u)
            * p(x3, F(1, 4) + F(1, 2) * x2)
            * p(x4, F(3, 20) + F(7, 20) * x3 + F(3, 10) * u)
        )
        integral &= (pr * 16000).denominator == 1
    return means, integral


def check(document: dict) -> list[str]:
    problems = []
    contrast = frontdoor_contrast()
    if abs(float(contrast) - document["frontdoor_discrete_contrast"]) > ATOL:
        problems.append(f"frontdoor_discrete_contrast {document['frontdoor_discrete_contrast']} vs {float(contrast)}")
    means, integral = verma()
    if not integral:
        problems.append("16000-row Verma population cells are not integers")
    for i, mean in enumerate(means):
        if abs(float(mean) - document["verma_do_means"][i]) > ATOL:
            problems.append(f"verma_do_means[{i}] {document['verma_do_means'][i]} vs {float(mean)}")
    if abs(float(means[1] - means[0]) - document["verma_contrast"]) > ATOL:
        problems.append("verma_contrast differs from the do-mean difference")
    return problems


def self_test(document: dict) -> list[str]:
    failures = []
    wrong = copy.deepcopy(document)
    wrong["frontdoor_discrete_contrast"] += 1e-6
    if not check(wrong):
        failures.append("shifted front-door contrast accepted")
    wrong = copy.deepcopy(document)
    wrong["verma_do_means"][0] += 1e-6
    if not check(wrong):
        failures.append("shifted Verma do-mean accepted")
    wrong = copy.deepcopy(document)
    wrong["verma_contrast"] = 0.5025 - 0.3275 + 0.01
    if not check(wrong):
        failures.append("shifted Verma contrast accepted")
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
