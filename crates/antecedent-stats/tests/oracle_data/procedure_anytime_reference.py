# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Reference for adaptive_inference.procedure_anytime_exclusion.

Regenerate with:  uv run --offline --script procedure_anytime_reference.py > procedure_anytime_reference.json

Per cell with n observed units and mean m, the bound at miscoverage a = alpha / cells and tuning
rho is  [max(m - r, 0), min(m + r, 1)]  with the normal-mixture radius
r = (1/2) sqrt((n + rho) ln((n + rho) / (rho a^2))) / n  (sub-Gaussian scale 1/2 for outcomes in
[0, 1]), evaluated at 60 digits (decimal).  A cell without observations has the whole unit
interval.  The exclusion probability of a truth is the exact probability, over the path-enumerated
rational law of the procedure (procedure_exact.py), of the terminal states at which some observed
cell's bound misses its truth.  The smallest distance of any bound end to its truth is recorded
and must exceed 1e-9, so no comparison is decided by rounding.

Repository-authored (MIT OR Apache-2.0); standard library only.
"""
import json
import os
import sys
from decimal import Decimal, getcontext

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import procedure_exact as pe  # noqa: E402

getcontext().prec = 60
F = pe.F


def radius(n, alpha, rho):
    n, rho, alpha = Decimal(n), Decimal(rho), Decimal(alpha)
    total = n + rho
    return Decimal("0.5") * (total * (total / (rho * alpha * alpha)).ln()).sqrt() / n


def bounds(counts, cells, alpha, rho):
    out = []
    a = Decimal(alpha) / Decimal(cells)
    for c in range(cells):
        s, f = counts[2 * c], counts[2 * c + 1]
        n = s + f
        if n == 0:
            out.append((None, Decimal(0), Decimal(1)))
            continue
        m = Decimal(s) / Decimal(n)
        r = radius(n, a, rho)
        out.append((m, max(m - r, Decimal(0)), min(m + r, Decimal(1))))
    return out


smallest = Decimal(1)
exclusions = []


def exclusion(name, decl, truth, alpha, rho):
    global smallest
    cells = decl["arms"]
    atoms = pe.enumerate_law(decl, ("fixed", truth))
    t = [Decimal(pe.frac(x).numerator) / Decimal(pe.frac(x).denominator) for x in truth]
    miss = F(0)
    for (look, stopped, obs, dec), p in atoms.items():
        out = bounds(obs, cells, alpha, rho)
        excluded = False
        for c, (m, lo, hi) in enumerate(out):
            if m is None:
                continue
            smallest = min(smallest, abs(lo - t[c]) if lo > 0 else Decimal(1), abs(hi - t[c]) if hi < 1 else Decimal(1))
            if lo > t[c] or hi < t[c]:
                excluded = True
        if excluded:
            miss += p
    exclusions.append(
        {
            "name": name,
            "declaration": decl,
            "truth": truth,
            "alpha": alpha,
            "tuning": rho,
            "exclusion_probability": float(miss),
            "terminal_states": len(atoms),
        }
    )


FIXED2 = {"kind": "fixed", "probabilities": ["1/2", "1/2"]}
THOMPSON = {"kind": "thompson", "prior": [1, 1]}


def decl(arms, assignment, enrol, stopping, **extra):
    d = {"arms": arms, "assignment": assignment, "enrolment": enrol, "stopping": stopping, "selection": {"kind": "none"}}
    d.update(extra)
    return d


exclusion("fixed_two_arms_loose_level", decl(2, FIXED2, [2, 2, 2], {"kind": "none"}), ["1/4", "3/4"], "0.9", 5)
exclusion("thompson_early_stop_by_probability_best", decl(2, THOMPSON, [1] * 7, {"kind": "best", "threshold": "73/100", "prior": [1, 1]}), ["1/5", "4/5"], "0.8", 2)
exclusion("three_arm_ucb_resource_cap", decl(3, {"kind": "ucb", "exploration": "2"}, [1] * 6, {"kind": "cap", "max_units": 5}), ["1/4", "1/2", "3/4"], "0.7", 3)
exclusion("lagged_epsilon_greedy_with_loss", decl(2, {"kind": "epsilon_greedy", "epsilon": "1/5", "prior": [1, 1]}, [1] * 5, {"kind": "none"}, lag=1, retain=["3/4", "3/4"]), ["1/2", "1/2"], "0.6", 4)
exclusion("truth_on_the_boundary_of_the_support", decl(2, FIXED2, [2, 2, 2], {"kind": "none"}), ["0", "1"], "0.9", 1)
exclusion("exposure_failure_delivered_attribution", decl(2, FIXED2, [2, 2], {"kind": "none"}, exposure={"failure": ["0", "1/4"], "fallback": 0, "attribution": "delivered"}), ["1/3", "2/3"], "0.9", 3)

reports = []
for counts, alpha, rho in (
    ([0, 0, 0, 0], "0.05", 10),
    ([3, 1, 0, 0], "0.05", 10),
    ([30, 10, 5, 25], "0.2", 50),
    ([1, 0, 0, 1], "0.99", 1),
    ([400, 600, 250, 250], "0.01", 1000),
    ([2, 2, 2, 2], "0.5", 0.5),
):
    out = bounds(counts, 2, alpha, rho)
    reports.append(
        {
            "counts": counts,
            "alpha": alpha,
            "tuning": rho,
            "alpha_per_cell": float(Decimal(alpha) / 2),
            "cells": [
                {"mean": None if m is None else float(m), "lower": float(lo), "upper": float(hi), "units": counts[2 * c] + counts[2 * c + 1]}
                for c, (m, lo, hi) in enumerate(out)
            ],
        }
    )
assert smallest > Decimal("1e-9"), smallest
print(json.dumps({"exclusions": exclusions, "reports": reports, "smallest_gap": float(smallest)}, indent=1))
