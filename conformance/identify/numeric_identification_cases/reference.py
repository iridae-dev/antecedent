# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the numeric identification cases: each binary structural model is enumerated exactly over rationals, the interventional mean is computed by truncated factorization (cutting the treatment's own mechanism), and the identifying formula of each case is evaluated on the observational joint to confirm it equals that truth.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/identify/numeric_identification_cases/reference.py --check
    uv run --frozen --script conformance/identify/numeric_identification_cases/reference.py --self-test

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

from itertools import product


def bern(p, v):
    return p if v else 1 - p


def marginal(joint, names, keep):
    out = {}
    for cell, p in joint.items():
        key = tuple(cell[names.index(k)] for k in keep)
        out[key] = out.get(key, 0) + p
    return out


def prob(joint, names, **fixed):
    return sum(p for cell, p in joint.items() if all(cell[names.index(k)] == v for k, v in fixed.items()))


def case_observed_parent(spec):
    names = ["Z", "X", "Y"]
    pz = F("0.35")
    px = lambda z: F("0.2") + F("0.45") * z
    py = lambda x, z: F("0.1") + F("0.4") * x + F("0.25") * z
    joint = {}
    for z, x, y in product((0, 1), repeat=3):
        joint[(z, x, y)] = bern(pz, z) * bern(px(z), x) * bern(py(x, z), y)
    # truncated factorization: drop P(x | z)
    do = lambda x: sum(bern(pz, z) * py(x, z) for z in (0, 1))
    ate = do(1) - do(0)
    close(ate, spec["expected_ate"], 1e-12, "expected_ate", problems_ref)
    # the parent-free graph: P(y | x) is the functional
    pf = (F("0.15") + F("0.5")) - F("0.15")
    close(pf, spec["parent_free_expected_ate"], 1e-12, "parent_free_expected_ate", problems_ref)
    observed = prob(joint, names, X=1, Y=1) / prob(joint, names, X=1) - prob(joint, names, X=0, Y=1) / prob(joint, names, X=0)
    if abs(observed - ate) <= F("0.01"):
        problems_ref.append(f"the observed contrast {float(observed)} does not differ from the adjusted effect {float(ate)}")
    if spec["functional_must_differ_from_parent_free"] and ate == pf:
        problems_ref.append("adjusted and parent-free functionals coincide")


def case_napkin(spec):
    names = ["U1", "U2", "W", "Z", "X", "Y"]
    p1, p2 = F("0.35"), F("0.7")
    pw = lambda u1, u2: F("0.2") + F("0.3") * u1 + F("0.25") * u2
    pz = lambda w: F("0.3") + F("0.45") * w
    px = lambda z, u1: F("0.15") + F("0.5") * z + F("0.2") * u1
    py = lambda x, u2: F("0.1") + F("0.35") * x + F("0.4") * u2
    joint = {}
    for u1, u2, w, z, x, y in product((0, 1), repeat=6):
        joint[(u1, u2, w, z, x, y)] = bern(p1, u1) * bern(p2, u2) * bern(pw(u1, u2), w) * bern(pz(w), z) * bern(px(z, u1), x) * bern(py(x, u2), y)
    do = lambda x: sum(bern(p1, u1) * bern(p2, u2) * bern(pw(u1, u2), w) * bern(pz(w), z) * py(x, u2)
                       for u1, u2, w, z in product((0, 1), repeat=4))
    got = [do(0), do(1)]
    for g, want in zip(got, spec["expected_do_y1"]):
        close(g, want, 1e-12, "expected_do_y1", problems_ref)
    close(got[1] - got[0], spec["expected_ate"], 1e-12, "expected_ate", problems_ref)
    obs = marginal(joint, names, ["W", "Z", "X", "Y"])
    on = ["W", "Z", "X", "Y"]
    for z in (0, 1):
        for x in (0, 1):
            num = sum(prob(obs, on, W=w, Z=z, X=x, Y=1) / prob(obs, on, W=w, Z=z) * prob(obs, on, W=w) for w in (0, 1))
            den = sum(prob(obs, on, W=w, Z=z, X=x) / prob(obs, on, W=w, Z=z) * prob(obs, on, W=w) for w in (0, 1))
            close(num / den, got[x], 1e-12, f"napkin ratio at z={z}, x={x}", problems_ref)


def case_discriminating(spec):
    names = ["A", "L1", "L2", "Q", "C", "T", "Y"]
    pa, p1, p2 = F("0.4"), F("0.4"), F("0.55")
    pq = lambda a, l1: F("0.15") + F("0.35") * a + F("0.25") * l1
    pc = lambda l1, l2: F("0.2") + F("0.3") * l1 + F("0.25") * l2
    pt = lambda l2: F("0.25") + F("0.4") * l2
    py = lambda q, c, t: F("0.05") + F("0.2") * q + F("0.25") * c + F("0.35") * t
    joint = {}
    for a, l1, l2, q, c, t, y in product((0, 1), repeat=7):
        joint[(a, l1, l2, q, c, t, y)] = bern(pa, a) * bern(p1, l1) * bern(p2, l2) * bern(pq(a, l1), q) * bern(pc(l1, l2), c) * bern(pt(l2), t) * bern(py(q, c, t), y)
    do = lambda t: sum(bern(pa, a) * bern(p1, l1) * bern(p2, l2) * bern(pq(a, l1), q) * bern(pc(l1, l2), c) * py(q, c, t)
                       for a, l1, l2, q, c in product((0, 1), repeat=5))
    got = [do(0), do(1)]
    for g, want in zip(got, spec["expected_do_y1"]):
        close(g, want, 1e-12, "expected_do_y1", problems_ref)
    close(got[1] - got[0], spec["expected_ate"], 1e-12, "expected_ate", problems_ref)
    obs = marginal(joint, names, ["Q", "C", "T", "Y"])
    on = ["Q", "C", "T", "Y"]
    for t in (0, 1):
        adj = sum(prob(obs, on, Q=q, C=c) * prob(obs, on, Q=q, C=c, T=t, Y=1) / prob(obs, on, Q=q, C=c, T=t)
                  for q, c in product((0, 1), repeat=2))
        close(adj, got[t], 1e-12, f"adjustment over {{Q,C}} at t={t}", problems_ref)


problems_ref = []


def check(document):
    problems_ref.clear()
    by_id = {c["id"]: c for c in document["cases"]}
    case_observed_parent(by_id["treatment_with_observed_parent"])
    case_napkin(by_id["napkin"])
    case_discriminating(by_id["discriminating_path_visibility"])
    return list(problems_ref)


def _case(d, name):
    return next(c for c in d["cases"] if c["id"] == name)


MUTATIONS = [
    ("observed-parent effect changed", lambda d: _case(d, "treatment_with_observed_parent").__setitem__("expected_ate", 0.5)),
    ("napkin do value changed", lambda d: _case(d, "napkin")["expected_do_y1"].__setitem__(1, 0.7)),
    ("discriminating-path value changed", lambda d: _case(d, "discriminating_path_visibility")["expected_do_y1"].__setitem__(0, 0.25)),
    ("napkin effect changed", lambda d: _case(d, "napkin").__setitem__("expected_ate", 0.4)),
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
