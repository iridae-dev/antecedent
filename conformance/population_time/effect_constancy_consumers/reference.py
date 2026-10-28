# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the effect-constancy fixture: the variance of a difference under the declared covariance, the two-sided normal tail erfc(|z|/sqrt 2), negative absolute-distance prior scores, and the common leaders of an affine policy with baseline utility 0 and treatment utility equal to the effect.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/population_time/effect_constancy_consumers/reference.py --check
    uv run --frozen --script conformance/population_time/effect_constancy_consumers/reference.py --self-test

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
    t = document["transport"]
    v1, c12, c21, v2 = t["covariance"]
    if c12 != c21:
        problems.append("covariance is not symmetric")
    diff = t["effects"][0] - t["effects"][1]
    var = v1 + v2 - 2.0 * c12
    close(var, 0.25, 1e-15, "difference variance", problems)
    close(math.sqrt(var), t["difference_standard_error"], 1e-12, "difference standard error", problems)
    close(diff, t["difference"], 1e-12, "difference", problems)
    z = diff / math.sqrt(var)
    p = math.erfc(abs(z) / math.sqrt(2.0))
    close(p, t["holm_p_single_contrast"], 1e-12, "two-sided p", problems)   # Holm with one contrast leaves p unchanged
    prior = document["prior"]
    scores = [-abs(e - prior["target_effect"]) for e in prior["source_effects"]]
    for i, (got, want) in enumerate(zip(scores, prior["scores"])):
        close(got, want, 1e-12, f"prior score {i}", problems)
    names = ["far", "near"]   # source order in the executing test: far (effect 1.0), near (effect 2.0)
    ranked = [names[i] for i in sorted(range(2), key=lambda i: -scores[i])]
    if ranked != prior["compatible_rank"]:
        problems.append(f"compatible rank {ranked} != fixture {prior['compatible_rank']}")

    def common_leaders(effects):
        leaders = None
        for effect in effects:   # baseline utility 0, treat utility = effect
            utility = {"baseline": 0.0, "treat": effect}
            best = max(utility.values())
            top = {a for a, u in utility.items() if u == best}
            leaders = top if leaders is None else leaders & top
        return sorted(leaders)

    policy = document["policy"]
    for key, effects in (("disagreement", "disagreement_effects"), ("tie", "tie_effects"), ("same", "same_effects")):
        got = common_leaders(policy[effects])
        if got != sorted(policy[f"{key}_common_leaders"]):
            problems.append(f"{key} common leaders {got} != fixture {policy[key + '_common_leaders']}")
    return problems


MUTATIONS = [
    ("p value shifted", lambda d: d["transport"].__setitem__("holm_p_single_contrast", 0.05)),
    ("covariance changed", lambda d: d["transport"].__setitem__("covariance", [0.25, 0.0, 0.0, 0.25])),
    ("score shifted", lambda d: d["prior"]["scores"].__setitem__(0, -1.0)),
    ("rank reversed", lambda d: d["prior"].__setitem__("compatible_rank", ["far", "near"])),
    ("tie leaders collapsed", lambda d: d["policy"].__setitem__("tie_common_leaders", ["treat"])),
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
