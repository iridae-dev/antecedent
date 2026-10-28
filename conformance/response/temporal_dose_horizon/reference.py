# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the dose-by-horizon surface: for Y_s = 1 + 2 T_{s-1} + 3 T_{s-2} with zero-mean exogenous treatment, a pulse at -1 sets T_{-1} = d, so horizon 1 gives 1 + 2 d and horizon 2 gives 1 + 3 d; soft, additive-shift and two-step sequence interventions follow by substituting into the same equation.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/response/temporal_dose_horizon/reference.py --check
    uv run --frozen --script conformance/response/temporal_dose_horizon/reference.py --self-test

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
    contract = document["contract"]
    doses, horizons = contract["dose_grid"], contract["horizons"]
    # Structural equations Y_s = 1 + 2 T_{s-1} + 3 T_{s-2}, exogenous T with zero mean over the lag-aligned
    # windows (period-four sequence 0, 1, 0, -1). Pulse at -1 sets T_{-1} = d.
    coefficient = {1: 2.0, 2: 3.0}   # horizon h evaluates Y_{h-1}: T_{-1} enters with lag h
    level = lambda d, h: 1.0 + coefficient[h] * d
    surface = [level(d, h) for d in doses for h in horizons]   # dose-major
    for i, (got, want) in enumerate(zip(surface, contract["surface"]["mean"])):
        close(got, want, 1e-12, f"surface.mean[{i}]", problems)
    pairs = [[d, float(h)] for d in doses for h in horizons]
    if pairs != contract["surface"]["grid_pairs"]:
        problems.append("dose-major grid pairs differ")
    # Interventions at dose 1.
    paths = contract["intervention_paths"]
    set_1 = [level(1.0, h) for h in horizons]
    soft_additive = [1.0 + 2.0 * (0.0 + 1.0) + 3.0 * 0.0, 1.0 + 2.0 * 0.0 + 3.0 * (0.0 + 1.0)]  # T replaced by T + 1, E[T] = 0
    two_step = [1.0 + 2.0 * 1.0 + 3.0 * 1.0, 1.0 + 2.0 * 0.0 + 3.0 * 1.0]   # T_{-2} = T_{-1} = 1; Y_1 uses E[T_0] = 0
    for name, got in (("set_1", set_1), ("soft_constant_1", set_1), ("soft_additive_shift_1", soft_additive),
                      ("sequence_two_step_set_1", two_step), ("sequence_two_step_soft_constant_1", two_step)):
        for i, (g, w) in enumerate(zip(got, paths[name])):
            close(g, w, 1e-12, f"intervention_paths.{name}[{i}]", problems)
    horizon_1 = level(1.0, 1) - level(0.0, 1)
    close(horizon_1, contract["pulse_effect_projection"]["contrast"], 1e-12, "pulse effect contrast", problems)
    close(horizon_1, contract["sustained_effect_projection"]["contrast"], 1e-12, "sustained effect contrast", problems)
    return problems


MUTATIONS = [
    ("surface cell shifted", lambda d: d["contract"]["surface"]["mean"].__setitem__(3, 4.5)),
    ("two-step path collapsed to the last step", lambda d: d["contract"]["intervention_paths"]["sequence_two_step_set_1"].__setitem__(0, 3.0)),
    ("pulse contrast shifted", lambda d: d["contract"]["pulse_effect_projection"].__setitem__("contrast", 3.0)),
    ("dose grid changed", lambda d: d["contract"].__setitem__("dose_grid", [0.0, 2.0])),
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
