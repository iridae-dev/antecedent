# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the temporal observation fixture: for Y_s = 5 + 2 T_{s-1} + 3 T_{s-2} + U_s with zero-mean uniform treatment and noise, a pulse at -1 gives E[Y_0] = 5 + 2 d and E[Y_1] = 5 + 3 d on the stated dose grid.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/response/temporal_observation/reference.py --check
    uv run --frozen --script conformance/response/temporal_observation/reference.py --self-test

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
    if got is None or want is None or not math.isclose(got, want, rel_tol=0.0, abs_tol=tol):
        problems.append(f"{label}: derived {got!r} != fixture {want!r} (tol {tol})")


def simpson(f, a=-10.0, b=10.0, n=4000):
    h = (b - a) / n
    total = f(a) + f(b)
    for i in range(1, n):
        total += f(a + i * h) * (4 if i % 2 else 2)
    return total * h / 3


def phi(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def Phi(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def check(document):
    problems = []
    # Y_s = 5 + 2 T_{s-1} + 3 T_{s-2} + U_s with T_s ~ Uniform(-1, 1) and U_s mean zero: a pulse at -1 sets T_{-1} = d
    # and leaves E[T] = 0 for the other lags, so horizon 1 gives 5 + 2 d and horizon 2 gives 5 + 3 d.
    coefficient = {1: 2.0, 2: 3.0}
    level = lambda d, h: 5.0 + coefficient[h] * d
    surface = [level(d, h) for d in document["grid"] for h in document["horizons"]]
    for i, (got, want) in enumerate(zip(surface, document["surface"]["mean"])):
        close(got, want, 1e-12, f"surface.mean[{i}]", problems)
    path = [level(0.5, h) for h in document["horizons"]]
    for i, (got, want) in enumerate(zip(path, document["intervention_set_0_5"])):
        close(got, want, 1e-12, f"intervention_set_0_5[{i}]", problems)
    if not 0 < document["naive_gap_min"] < document["atol"]:
        problems.append("naive_gap_min must be positive and below the point tolerance")
    return problems


MUTATIONS = [
    ("surface cell shifted", lambda d: d["surface"]["mean"].__setitem__(5, 6.0)),
    ("set path shifted", lambda d: d["intervention_set_0_5"].__setitem__(0, 6.5)),
    ("grid changed", lambda d: d.__setitem__("grid", [-1.0, 0.0, 1.0])),
]


def self_test(document):
    failures = []
    if check(document):
        failures.append("the unmodified fixture is rejected")
    for label, mutate in MUTATIONS:
        wrong = copy.deepcopy(document)
        mutate(wrong)
        if not check(wrong):
            failures.append(f"accepted: {{label}}")
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
