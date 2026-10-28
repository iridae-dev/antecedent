# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the observation-primitive fixture: the selected-outcome AIPW pseudo-value mu + R/p (Y - mu), the product-limit inverse-censoring weight 1/G(t-) (conditional on the entry time G(t-)/G(entry) under delayed entry, floored), and the Gaussian left-, right-, interval-censored and truncated log-likelihood terms from the normal CDF and density.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/response/observation_primitives/reference.py --check
    uv run --frozen --script conformance/response/observation_primitives/reference.py --self-test

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

def aipw_pseudo(block):
    """Selected-outcome AIPW pseudo-value: mu + R/p * (Y - mu), with the residual taken as 0 when R = 0."""
    out = []
    for y, r, p, mu in zip(block["observed"], block["indicator"], block["probabilities"], block["outcome_regression"]):
        residual = 0.0 if r == 0.0 or y is None else y - mu
        out.append(mu + r / p * residual)
    return out


def censoring_survival(time, event, entry, at, inclusive, use_entry):
    """Product-limit survival of the censoring distribution (censoring = event 0) up to `at`."""
    value = 1.0
    for u in sorted({t for t, e in zip(time, event) if e == 0.0}):
        if u > at or (u == at and not inclusive):
            continue
        risk = sum(1 for t, s in zip(time, entry) if t >= u and (s <= u if use_entry else True))
        died = sum(1 for t, e in zip(time, event) if t == u and e == 0.0)
        value *= 1.0 - died / risk
    return value


def ipcw(block, use_entry):
    time, event, entry, floor = block["time"], block["event"], block["entry"], block["survival_floor"]
    out = []
    for t, e, s in zip(time, event, entry):
        g = censoring_survival(time, event, entry, t, False, use_entry)
        if use_entry:
            g /= censoring_survival(time, event, entry, s, True, use_entry)
        out.append(0.0 if e == 0.0 else 1.0 / max(g, floor))
    return out


def log_likelihood(block):
    sigma = block["sigma"]
    total = 0.0
    for mean, obs in zip(block["means"], block["observations"]):
        kind = obs["kind"]
        if kind == "left_censored":
            total += math.log(Phi((obs["bound"] - mean) / sigma))
        elif kind == "right_censored":
            total += math.log(Phi(-(obs["bound"] - mean) / sigma))
        elif kind == "interval_censored":
            total += math.log(Phi((obs["upper"] - mean) / sigma) - Phi((obs["lower"] - mean) / sigma))
        elif kind == "truncated":
            mass = Phi((obs["upper"] - mean) / sigma) - Phi((obs["lower"] - mean) / sigma)
            total += math.log(phi((obs["value"] - mean) / sigma) / sigma) - math.log(mass)
        else:
            raise ValueError(kind)
    return total


def check(document):
    problems = []
    tol = document["tolerance"]["atol"]
    selected = document["selected_outcome"]
    for i, (got, want) in enumerate(zip(aipw_pseudo(selected), selected["expected_aipw_pseudo_values"])):
        close(got, want, tol, f"aipw pseudo-value {i}", problems)
    km = document["kaplan_meier_ipcw"]
    for key, use_entry in (("expected_without_entry", False), ("expected_with_entry", True)):
        for i, (got, want) in enumerate(zip(ipcw(km, use_entry), km[key])):
            close(got, want, tol, f"ipcw {key}[{i}]", problems)
    gauss = document["gaussian_observation_likelihood"]
    close(log_likelihood(gauss), gauss["expected_log_likelihood"], tol, "gaussian log-likelihood", problems)
    return problems


def _shift(path, index=None, delta=1e-6):
    def mutate(document):
        node = document
        for key in path[:-1]:
            node = node[key]
        if index is None:
            node[path[-1]] += delta
        else:
            node[path[-1]][index] += delta
    return mutate


MUTATIONS = [
    ("aipw pseudo-value shifted", _shift(["selected_outcome", "expected_aipw_pseudo_values"], 0)),
    ("ipcw weight without entry shifted", _shift(["kaplan_meier_ipcw", "expected_without_entry"], 1)),
    ("ipcw weight with entry replaced by the no-entry weight", lambda d: d["kaplan_meier_ipcw"]["expected_with_entry"].__setitem__(1, 1.5)),
    ("log-likelihood shifted", _shift(["gaussian_observation_likelihood", "expected_log_likelihood"])),
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
