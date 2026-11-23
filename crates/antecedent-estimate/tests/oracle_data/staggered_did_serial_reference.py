# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath"]
# ///
"""Cohort-by-period difference-in-differences contrasts on serially dependent panels, exactly.

Regenerate with:  uv run --script staggered_did_serial_reference.py > staggered_did_serial_reference.json

For an adoption cohort g > 0 with baseline g - 1 and a period t, the contrast is the mean change
(y_t - y_{g-1}) of the cohort's units minus the mean change of the never-treated units (cohort
0), over the units observed at both periods.  Its standard error is the cluster sandwich of the
influence of the difference of two means: a treated unit contributes (d - mean_T)/n_T and a
never-treated unit -(d - mean_C)/n_C to its cluster's score, and
Var = G/(G-1) * sum over the G clusters present of score^2.  Everything is Fraction arithmetic
on the exact binary64 inputs; the standard error is sqrt at 60 digits.  `post` lists the
group-time cells (t >= g).  An unbalanced case drops unit-periods; its contrasts use only the
units observed at both periods.  Values are written as nearest binary64.
Repository-authored (MIT OR Apache-2.0).
"""
import json
import random
from fractions import Fraction as F

from mpmath import mp, mpf, sqrt

mp.dps = 60
rng = random.Random(20261029)


def contrasts(rows, post_only):
    """rows: (y, unit, period, cohort, cluster) -> list of dict, ordered by cohort then period."""
    units = {}
    for y, u, t, g, c in rows:
        units.setdefault(u, (g, c, {}))[2][t] = F(y)
    periods = sorted({t for *_, t, _, _ in [(0, 0, r[2], 0, 0) for r in rows]})
    out = []
    for g in sorted({v[0] for v in units.values() if v[0] > 0}):
        base = g - 1
        for t in periods:
            if t == base or (post_only and t < g):
                continue
            def sel(cohort):
                return [(v[1], v[2][t] - v[2][base]) for v in units.values()
                        if v[0] == cohort and t in v[2] and base in v[2]]
            ctrl, trt = sel(0), sel(g)
            mt = sum(d for _, d in trt) / len(trt)
            mc = sum(d for _, d in ctrl) / len(ctrl)
            score = {}
            for c, d in trt:
                score[c] = score.get(c, 0) + (d - mt) / len(trt)
            for c, d in ctrl:
                score[c] = score.get(c, 0) - (d - mc) / len(ctrl)
            G = len(score)
            var = F(G, G - 1) * sum(s * s for s in score.values())
            se = sqrt(mpf(var.numerator) / mpf(var.denominator))
            out.append(dict(cohort=g, period=t, event_time=t - g, effect=float(mt - mc),
                            treated=len(trt), comparison=len(ctrl), se=float(se), clusters=G))
    return out


def panel(cohort_sizes, periods, cluster_of, rho, drop=0.0):
    """Every unit's errors follow a stationary AR(1) in the period (coefficient rho) around a
    unit level, so outcomes within a unit are strongly serially dependent."""
    rows, uid = [], 0
    for g, n in cohort_sizes:
        for _ in range(n):
            u = f"u{uid}"
            uid += 1
            level = rng.uniform(-1, 1)
            slope = rng.uniform(-0.5, 0.5)
            e = rng.gauss(0, 1.0)
            for t in periods:
                e = rho * e + rng.gauss(0, 0.5)
                y = level + slope * t + (g > 0 and t >= g) * (1 + 0.3 * (t - g)) + e
                if drop and rng.random() < drop and t not in (g - 1, g):
                    continue
                rows.append((y, u, t, g, cluster_of(uid - 1, g)))
    return rows


cases = []
specs = [
    ("ar1_rho_0_9_unit_clusters", [(0, 8), (3, 6), (4, 6)], range(1, 8), lambda i, g: f"c{i}", 0.9, 0.0),
    ("ar1_rho_0_6_paired_clusters", [(0, 8), (2, 6), (4, 6)], range(1, 7), lambda i, g: f"c{i // 2}", 0.6, 0.0),
    ("ar1_rho_0_9_unbalanced", [(0, 10), (3, 7), (5, 6)], range(1, 8), lambda i, g: f"c{i}", 0.9, 0.2),
]
for name, sizes, periods, cl, rho, drop in specs:
    rows = panel(sizes, list(periods), cl, rho, drop)
    case = dict(name=name, rows=[list(r) for r in rows], available_case=drop > 0, rho=rho)
    case["event"] = contrasts(rows, False)
    if drop == 0:
        case["group_time"] = contrasts(rows, True)
    cases.append(case)
print(json.dumps(dict(cases=cases)))
