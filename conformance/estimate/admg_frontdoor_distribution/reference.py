# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the binary front-door distribution fixture: the contingency table is the exact joint of the stated latent-variable SCM at 800 rows, and P(Y=1 | do(T=t)) follows by averaging the outcome equation over the cut treatment mechanism, then agrees with the front-door formula applied to the table alone.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/estimate/admg_frontdoor_distribution/reference.py --check
    uv run --frozen --script conformance/estimate/admg_frontdoor_distribution/reference.py --self-test

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

def joint():
    pu = {0: F(1, 2), 1: F(1, 2)}
    pt = {0: F(1, 5), 1: F(4, 5)}
    pm = {0: F(1, 4), 1: F(3, 4)}
    out = {}
    for u in (0, 1):
        for t in (0, 1):
            for m in (0, 1):
                py1 = F(1, 10) + F(1, 2) * m + F(3, 10) * u
                for y in (0, 1):
                    p = pu[u] * (pt[u] if t else 1 - pt[u]) * (pm[t] if m else 1 - pm[t]) * (py1 if y else 1 - py1)
                    out[(t, m, y)] = out.get((t, m, y), 0) + p
    return out


def check(document):
    problems = []
    scm = document["scm"]
    same(scm["p_u1"], 0.5, "p_u1", problems)
    same((scm["p_t1_given_u"]["0"], scm["p_t1_given_u"]["1"]), (0.2, 0.8), "p_t1_given_u", problems)
    same((scm["p_m1_given_t"]["0"], scm["p_m1_given_t"]["1"]), (0.25, 0.75), "p_m1_given_t", problems)
    same(tuple(scm["p_y1_given_m_u"].values()), (0.1, 0.5, 0.3), "p_y1_given_m_u", problems)
    p = joint()
    rows = document["rows"]
    for cell in document["contingency_table"]:
        key = (int(cell["t"]), int(cell["m"]), int(cell["y"]))
        if p[key] * rows != cell["count"]:
            problems.append(f"count {key}: derived {p[key] * rows} != fixture {cell['count']}")
    if sum(c["count"] for c in document["contingency_table"]) != rows:
        problems.append("contingency counts do not sum to rows")
    # interventional truth: cut U -> T, average the outcome equation over U
    for t in (0, 1):
        pm1 = F(3, 4) if t else F(1, 4)
        truth = sum((pm1 if m else 1 - pm1) * (F(1, 4) + F(1, 2) * m) for m in (0, 1))
        want = next(r["p_y1"] for r in document["truth"] if int(r["t"]) == t)
        close(truth, want, 1e-15, f"truth t={t}", problems)
        # front-door formula from the table only
        n = {k: F(c["count"]) for c in document["contingency_table"] for k in [(int(c["t"]), int(c["m"]), int(c["y"]))]}
        total = sum(n.values())
        def pt_(tt): return sum(v for (a, _, _), v in n.items() if a == tt) / total
        def pm_t(m, tt): return sum(v for (a, b, _), v in n.items() if a == tt and b == m) / sum(v for (a, _, _), v in n.items() if a == tt)
        def py_tm(tt, m): return n[(tt, m, 1)] / (n[(tt, m, 0)] + n[(tt, m, 1)])
        fd = sum(pm_t(m, t) * sum(pt_(tt) * py_tm(tt, m) for tt in (0, 1)) for m in (0, 1))
        close(fd, want, 1e-15, f"front-door from table t={t}", problems)
    return problems


MUTATIONS = [
    ("one cell count changed", lambda d: d["contingency_table"][0].__setitem__("count", 250)),
    ("treated truth changed", lambda d: d["truth"][1].__setitem__("p_y1", 0.65)),
    ("outcome coefficient changed", lambda d: d["scm"]["p_y1_given_m_u"].__setitem__("m", 0.4)),
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
