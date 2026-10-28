# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the posterior temporal response truth: the sample moments mean_z and mean_w of the 48-row law and the interventional levels 1.4 x + 0.6 mean_z + mean_w at doses 0 and 1.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/estimate/temporal_class_posterior_response_truth/reference.py --check
    uv run --frozen --script conformance/estimate/temporal_class_posterior_response_truth/reference.py --self-test

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
    n = document["n"]
    phase = [0.37 * i for i in range(n)]
    z = [math.sin(p) for p in phase]
    w = [0.1 * math.sin(1.7 * p) for p in phase]
    rows = range(1, n)
    mz = sum(F(z[i - 1]) for i in rows) / len(rows)
    mw = sum(F(w[i]) for i in rows) / len(rows)
    close(mz, document["moments"]["mean_z"], 1e-12, "mean_z", problems)
    close(mw, document["moments"]["mean_w"], 1e-12, "mean_w", problems)
    level = lambda x: F(14, 10) * x + F(6, 10) * mz + mw
    h = document["horizon_1"]["level"]
    close(level(0), h["curve"][0], 1e-12, "curve dose 0", problems)
    close(level(1), h["curve"][1], 1e-12, "curve dose 1", problems)
    close(level(1), h["set"][0], 1e-12, "set dose 1", problems)
    close(sum(F(str(x)) for x in document["posterior_weights"]), 1, 1e-12, "weights sum to one", problems)
    return problems


MUTATIONS = [
    ("mean_z changed", lambda d: d["moments"].__setitem__("mean_z", 0.07)),
    ("dose-0 level changed", lambda d: d["horizon_1"]["level"]["curve"].__setitem__(0, 0.0)),
    ("set level changed", lambda d: d["horizon_1"]["level"]["set"].__setitem__(0, 1.5)),
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
