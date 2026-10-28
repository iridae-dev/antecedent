# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the trial-to-target fixture: inverse-odds transport weights (1 - s)/s on trial rows, the weighted IPW contrast divided by the number of target rows, the augmented estimator with the outcome-regression plug-in on target rows plus the weighted residual correction, and Kish effective sample sizes (sum w)^2 / sum w^2.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/response/trial_transport/reference.py --check
    uv run --frozen --script conformance/response/trial_transport/reference.py --self-test

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

def equations(inputs):
    y, t, s = inputs["outcome"], inputs["treatment"], inputs["trial"]
    sel, prop, mu0, mu1 = inputs["selection_probability"], inputs["treatment_probability"], inputs["mu0"], inputs["mu1"]
    n_target = sum(1 for in_trial in s if not in_trial)
    trial = [i for i, in_trial in enumerate(s) if in_trial]
    # Inverse-odds transport weights for trial rows: P(target | x) / P(trial | x) = (1 - s) / s.
    w = {i: (1 - sel[i]) / sel[i] for i in trial}
    tw = {i: (1 / prop[i] if t[i] else 1 / (1 - prop[i])) for i in trial}
    ipw = sum(w[i] * ((y[i] / prop[i]) if t[i] else -(y[i] / (1 - prop[i]))) for i in trial) / n_target
    plug_in = sum(mu1[i] - mu0[i] for i, in_trial in enumerate(s) if not in_trial)
    correction = sum(
        w[i] * (((y[i] - mu1[i]) / prop[i]) if t[i] else -((y[i] - mu0[i]) / (1 - prop[i])))
        for i in trial
    )
    aipw = (plug_in + correction) / n_target
    ess = lambda weights: sum(weights) ** 2 / sum(x * x for x in weights)
    return ipw, aipw, ess(list(w.values())), ess(list(tw.values()))


def check(document):
    problems = []
    tol = document["tolerance"]["atol"]
    ipw, aipw, sel_ess, treat_ess = equations(document["inputs"])
    want = document["expected"]
    close(ipw, want["ipw"], tol, "ipw", problems)
    close(aipw, want["aipw"], tol, "aipw", problems)
    close(sel_ess, want["selection_effective_sample_size"], tol, "selection ess", problems)
    close(treat_ess, want["treatment_effective_sample_size"], tol, "treatment ess", problems)
    return problems


MUTATIONS = [
    ("ipw shifted", lambda d: d["expected"].__setitem__("ipw", 2.000001)),
    ("aipw shifted", lambda d: d["expected"].__setitem__("aipw", 1.999999)),
    ("selection ess shifted", lambda d: d["expected"].__setitem__("selection_effective_sample_size", 1.5)),
    ("outcome changed so the estimate moves", lambda d: d["inputs"]["outcome"].__setitem__(1, 5.0)),
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
