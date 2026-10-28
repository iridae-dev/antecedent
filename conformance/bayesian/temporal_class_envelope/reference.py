# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the Bayesian temporal class-envelope fixture: on the noise-free law y = 1 + 2 t_{i-1} + 0.5 z_{i-1} the z-adjusting completion recovers 2, the completion without z has slope 2 + 0.5 Cov(z,t)/Var(t) on the 800 rows, and the class-prior mixture 0.7*2 + 0.3*slope; the multi-step window effects follow from the lag coefficients.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/bayesian/temporal_class_envelope/reference.py --check
    uv run --frozen --script conformance/bayesian/temporal_class_envelope/reference.py --self-test

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

def solve(a, b):
    """Exact Gauss-Jordan solve of a x = b over Fractions."""
    n = len(a)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = next(r for r in range(col, n) if m[r][col] != 0)
        m[col], m[piv] = m[piv], m[col]
        p = m[col][col]
        m[col] = [v / p for v in m[col]]
        for r in range(n):
            if r != col and m[r][col] != 0:
                f = m[r][col]
                m[r] = [v - f * w for v, w in zip(m[r], m[col])]
    return [m[i][n] for i in range(n)]


def ols(columns, y):
    """Least squares of y on the given columns (include a constant column yourself), exact on float inputs."""
    cols = [[F(v) for v in c] for c in columns]
    yy = [F(v) for v in y]
    xtx = [[sum(a * b for a, b in zip(ci, cj)) for cj in cols] for ci in cols]
    xty = [sum(a * b for a, b in zip(ci, yy)) for ci in cols]
    return solve(xtx, xty)


def check(document):
    problems = []
    n = document["n"]
    z = [i % 2 for i in range(n)]
    t = [0.3 + 0.4 * z[i] + 0.05 * math.sin(0.017 * i) for i in range(n)]
    y = [None] + [1 + 2 * t[i - 1] + 0.5 * z[i - 1] for i in range(1, n)]
    rows = range(1, n)
    one = [1] * (n - 1)
    adjusted = ols([one, [t[i - 1] for i in rows], [z[i - 1] for i in rows]], [y[i] for i in rows])[1]
    unadjusted = ols([one, [t[i - 1] for i in rows]], [y[i] for i in rows])[1]
    a = document["assertions"]["with_class_prior"]
    byc = a["pulse_effect_by_completion"]
    close(adjusted, byc["z_causes_t_adjusts_z"], 1e-9, "z_causes_t_adjusts_z", problems)
    close(unadjusted, byc["t_causes_z_no_adjustment"], 5e-5, "t_causes_z_no_adjustment", problems)
    prior = document["class_prior_ordered"]     # order binds 0.3 to the no-adjustment completion, 0.7 to the adjusting one
    mixture = F(str(prior[1])) * adjusted + F(str(prior[0])) * unadjusted
    close(mixture, a["pulse_effect"], a["pulse_effect_tolerance"], "pulse_effect", problems)
    close(mixture, a["pulse_effect"], 5e-5, "pulse_effect to four decimals", problems)
    pag = document["temporal_pag_mixed_identification"]
    close(1 - F(str(pag["identified_mass"])), pag["unidentified_mass"], 1e-12, "pag unidentified mass", problems)
    close(sum(F(str(p)) for p in pag["class_prior_ordered"]), 1, 1e-12, "pag prior sums to one", problems)
    close(pag["sustained_single_step_effect"], pag["pulse_effect"], 1e-12, "single-step sustained equals pulse", problems)
    close(F(2), pag["pulse_effect"], 1e-12, "pag pulse coefficient", problems)
    ms = document["multi_step"]
    close(F(2), ms["last_step_effect"], 1e-12, "last step coefficient", problems)
    close(F(2) + F(3), ms["full_window_effect"], 1e-12, "full window = sum of lag coefficients", problems)
    return problems


MUTATIONS = [
    ("mixture changed", lambda d: d["assertions"]["with_class_prior"].__setitem__("pulse_effect", 2.6)),
    ("biased completion changed", lambda d: d["assertions"]["with_class_prior"]["pulse_effect_by_completion"].__setitem__("t_causes_z_no_adjustment", 3.0)),
    ("full window changed", lambda d: d["multi_step"].__setitem__("full_window_effect", 2.0)),
    ("prior order swapped", lambda d: d.__setitem__("class_prior_ordered", [0.7, 0.3])),
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
