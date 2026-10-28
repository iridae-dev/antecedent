# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the temporal-class response truth: the structural moments of the 300-row law, the z-adjusting completion's interventional level 1 + 2x + 0.6 mean_z + mean_w and the omitted-variable slope of the completion that does not adjust for z, evaluated with plain arithmetic over the sample rows.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/estimate/temporal_class_response_truth/reference.py --check
    uv run --frozen --script conformance/estimate/temporal_class_response_truth/reference.py --self-test

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
    r = [math.sin(0.29 * i) for i in range(n)]
    z = [0.4 * r[i] + math.cos(0.13 * i) + 0.3 * math.sin(1.7 * i) for i in range(n)]
    t = [0.3 + 0.5 * r[i] + 0.2 * z[i] + 0.4 * math.cos(2.3 * i) for i in range(n)]
    w = [0.1 * math.sin(0.77 * i) for i in range(n)]
    rows = range(1, n)
    mean = lambda v: sum(F(x) for x in v) / len(v)
    mt, mz, mw = mean([t[i - 1] for i in rows]), mean([z[i - 1] for i in rows]), mean([w[i] for i in rows])
    cov = lambda a, b: sum((F(x) - mean(a)) * (F(y) - mean(b)) for x, y in zip(a, b)) / len(a)
    ta, za, wa = [t[i - 1] for i in rows], [z[i - 1] for i in rows], [w[i] for i in rows]
    var_t, cov_tz, cov_tw = cov(ta, ta), cov(ta, za), cov(ta, wa)
    m = document["moments"]
    for key, got in (("mean_t", mt), ("mean_z", mz), ("mean_w", mw), ("var_t", var_t), ("cov_tz", cov_tz), ("cov_tw", cov_tw)):
        close(got, m[key], 1e-10, f"moments.{key}", problems)
    slope = 2 + (F("0.6") * cov_tz + cov_tw) / var_t
    mean_y = 1 + 2 * mt + F("0.6") * mz + mw
    close(slope, m["slope_z_as_mediator"], 1e-10, "slope_z_as_mediator", problems)
    close(mean_y, m["mean_y"], 1e-10, "mean_y", problems)
    level = lambda x: 1 + 2 * x + F("0.6") * mz + mw
    adj, med = document["horizon_1"]["adjusting_z"], document["horizon_1"]["z_as_mediator"]
    close(level(0), adj["curve"][0], 1e-10, "adjusting curve dose 0", problems)
    close(level(1), adj["curve"][1], 1e-10, "adjusting curve dose 1", problems)
    close(level(1), adj["set"][0], 1e-10, "adjusting set dose 1", problems)
    close(level(mt + F(1, 2)), adj["shift"][0], 1e-10, "adjusting shift", problems)
    assoc = lambda x: mean_y + slope * (F(x) - mt)
    close(assoc(0), med["curve"][0], 1e-10, "mediator curve dose 0", problems)
    close(assoc(1), med["curve"][1], 1e-10, "mediator curve dose 1", problems)
    close(assoc(1), med["set"][0], 1e-10, "mediator set dose 1", problems)
    close(assoc(mt + F(1, 2)), med["shift"][0], 1e-10, "mediator shift", problems)
    return problems


MUTATIONS = [
    ("slope changed", lambda d: d["moments"].__setitem__("slope_z_as_mediator", 2.0)),
    ("adjusting level changed", lambda d: d["horizon_1"]["adjusting_z"]["set"].__setitem__(0, 3.0)),
    ("mediator shift changed", lambda d: d["horizon_1"]["z_as_mediator"]["shift"].__setitem__(0, 2.9)),
    ("sample size changed", lambda d: d.__setitem__("n", 301)),
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
