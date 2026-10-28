# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the finite-discrete Bayesian transport posterior means. Under
do(X=1) the target P(Y=1) is the conditional cell ratio of the two X=1 cells. The empirical
support Bayesian bootstrap gives those cells Dirichlet parameters equal to their counts; the
binary state-space provider adds one to every cell. A Dirichlet component ratio has mean
a_y1 / (a_y0 + a_y1). Exact rational arithmetic.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/estimate/bayesian_structural_transport/reference.py --check

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def close(got, want, tol, label, problems):
    if not math.isclose(got, want, rel_tol=0.0, abs_tol=tol):
        problems.append(f"{label}: derived {got!r} != fixture {want!r} (tol {tol})")


from fractions import Fraction as F


def check(document):
    problems = []
    n00, n01, n10, n11 = document["counts_x0y0_x0y1_x1y0_x1y1"]
    empirical = F(n11, n10 + n11)
    state_space = F(n11 + 1, (n10 + 1) + (n11 + 1))
    close(float(empirical), document["empirical_support_p_y1_given_do_x1"], 1e-15, "empirical support", problems)
    close(float(state_space), document["state_space_p_y1_given_do_x1"], 1e-15, "state space", problems)
    return problems



def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.parse_args()
    document = json.loads((HERE / "expected.json").read_text(encoding="utf-8"))
    problems = check(document)
    for problem in problems:
        print("FAIL", problem)
    if not problems:
        print("ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
