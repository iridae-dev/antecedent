# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the observed-data temporal response truths: for Y_t = 2 + 1.5 X_{t-1} + 0.75 X_{t-2} + noise a pulse d at lag 1 gives horizon-one mean 2 + 1.5 d + 0.75 mean(X) and horizon-two mean 2 + 0.75 d + 1.5 mean(X), and the sequence X_{-2} = 1, X_{-1} = 2 gives 5.75.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/bayesian/temporal_observed_response/reference.py --check
    uv run --frozen --script conformance/bayesian/temporal_observed_response/reference.py --self-test

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
    a, b, c = (F(str(document[k])) for k in ("intercept", "lag1_coefficient", "lag2_coefficient"))
    levels = [F(str(v)) for v in document["sequence_levels"]]     # X_{-2}, X_{-1}
    close(a + b * levels[1] + c * levels[0], document["sequence_mean"], 1e-12, "sequence_mean", problems)
    # population standardization: X ~ N(0,1), so mean(X) = 0; a pulse at -1 enters lag 1 at horizon 1 and lag 2 at horizon 2
    horizon_means = {}
    for horizon in document["horizons"]:
        for d in document["doses"]:
            d = F(str(d))
            horizon_means[(horizon, d)] = a + (b if horizon == 1 else c) * d
    expected = {(1, 0): 2, (1, 1): F(7, 2), (2, 0): 2, (2, 1): F(11, 4)}
    for key, want in expected.items():
        close(horizon_means[(key[0], F(key[1]))], want, 1e-12, f"horizon mean {key}", problems)
    if not document["mean_atol"] > 0 or not document["sequence_atol"] > 0:
        problems.append("tolerances must be positive")
    return problems


MUTATIONS = [
    ("sequence mean changed", lambda d: d.__setitem__("sequence_mean", 5.5)),
    ("lag-1 coefficient changed", lambda d: d.__setitem__("lag1_coefficient", 1.0)),
    ("sequence levels swapped", lambda d: d.__setitem__("sequence_levels", [2.0, 1.0])),
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
