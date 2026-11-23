# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Misclassification-corrected temporal design under weights, missing outcomes and dependence.

Regenerate with:  uv run --script temporal_design_measurement_reference.py > temporal_design_measurement_reference.json

A case is a unit-period panel (three equally likely arms, a recorded binary outcome that may be
missing, optional analysis weights) and a declaration (lag, missing-data policy, serial
dependence).  Everything below is written from the definitions in fractions.Fraction.

Pairs.  A decision row t is paired with the same unit's row `lag` periods later.  complete_case
keeps a pair when both rows observe the outcome, available_case when the later row does,
balanced_panel when the unit observes the outcome in every period it has rows.  A pair whose
decision row has weight zero is removed.
Estimate.  With z_ia = 1[A_i = a] y_i / p_a and pair weights s_i, the arm mean is
mu_a = sum s z_a / sum s (the weighted Horvitz-Thompson ratio) and a coordinate is the linear
combination sum_a c_a mu_a (an arm mean: c = e_a; a contrast: the difference of arm means).
Sampling variance.  The linearized residual of a pair is s_i (sum_a c_a z_ia - theta).  Clustered by
unit: G/(G-1) sum_g (sum_{i in g} r_i)^2 / (sum s)^2.  Serial: residuals are summed over the pairs
that decide in the same period, then the Bartlett long-run variance with L = min(max(floor(4
(T/100)^(2/9)), 1), T - 1) is scaled by T/(T-1)/(sum s)^2; periods enter by label.
Correction.  Rogan-Gladen with the validation counts (900/100 positives, 850/150 negatives): an
arm mean is (theta + sp - 1) / (se + sp - 1), a contrast theta / (se + sp - 1), and the variance
adds the delta-method binomial terms of se and sp.
Values are written as the nearest binary64.
Repository-authored (MIT OR Apache-2.0); standard library only.
"""
import json
from fractions import Fraction as F
from math import isqrt

SE, SP = F(9, 10), F(85, 100)
VAR_SE = SE * (1 - SE) / 1000
VAR_SP = SP * (1 - SP) / 1000
YOUDEN = SE + SP - 1
COORDS = [
    ("mean:a0", (1, 0, 0), True),
    ("mean:a1", (0, 1, 0), True),
    ("mean:a2", (0, 0, 1), True),
    ("contrast:a1-a0", (-1, 1, 0), False),
    ("contrast:a2-a0", (-1, 0, 1), False),
]


def fsqrt(q, digits=60):
    q = max(q, F(0))
    scaled = q * 10 ** (2 * digits)
    return F(isqrt(scaled.numerator // scaled.denominator), 10**digits)


def panel(units, periods, drop_rows=()):
    rows = []
    for u in range(units):
        for t in range(periods):
            if (u, t) in drop_rows:
                continue
            arm = (u + t) % 3
            y = 1 if (u * 7) % 10 < 3 + arm else 0
            rows.append({"unit": u, "period": t, "arm": arm, "y": y})
    return rows


def estimate(rows, lag, policy, serial, weights):
    index = {(r["unit"], r["period"]): i for i, r in enumerate(rows)}
    observed = [r["y"] is not None for r in rows]
    unit_complete = {}
    for r, seen in zip(rows, observed):
        unit_complete[r["unit"]] = unit_complete.get(r["unit"], True) and seen
    pairs, dropped, zero = [], [], 0
    all_pairs = 0
    for i, r in enumerate(rows):
        j = index.get((r["unit"], r["period"] + lag))
        if j is None:
            continue
        all_pairs += 1
        if policy is None:
            keep = True
        elif policy == "complete_case":
            keep = observed[i] and observed[j]
        elif policy == "available_case":
            keep = observed[j]
        else:
            keep = unit_complete[r["unit"]]
        if not keep:
            reason = "missing_value" if not (observed[i] and observed[j]) else "unit_incomplete"
            dropped.append((i, reason))
            continue
        s = F(weights[i]) if weights else F(1)
        if s == 0:
            zero += 1
            continue
        pairs.append((i, j, s))
    total = sum(s for _, _, s in pairs)
    out = {}
    for name, c, mean in COORDS:
        z = []
        for i, j, s in pairs:
            arm = rows[i]["arm"]
            z.append(sum(F(c[a]) * (3 * F(rows[j]["y"]) if arm == a else 0) for a in range(3)))
        theta = sum(s * zi for (_, _, s), zi in zip(pairs, z)) / total
        resid = [s * (zi - theta) for (_, _, s), zi in zip(pairs, z)]
        if serial:
            by_period = {}
            for (i, _, _), r_ in zip(pairs, resid):
                by_period[rows[i]["period"]] = by_period.get(rows[i]["period"], F(0)) + r_
            labels = sorted(by_period)
            T = len(labels)
            L = min(max(int(4 * (T / 100) ** (2 / 9)), 1), T - 1)
            meat = sum(by_period[p] ** 2 for p in labels)
            for lag_ in range(1, L + 1):
                w = 1 - F(lag_, L + 1)
                meat += 2 * w * sum(by_period[p] * by_period[p - lag_] for p in labels if p - lag_ in by_period)
            var = meat * F(T, T - 1) / total**2
        else:
            per_unit = {}
            for (i, _, _), r_ in zip(pairs, resid):
                per_unit[rows[i]["unit"]] = per_unit.get(rows[i]["unit"], F(0)) + r_
            G = len(per_unit)
            var = sum(v * v for v in per_unit.values()) * F(G, G - 1) / total**2
        corrected = (theta + SP - 1) / YOUDEN if mean else theta / YOUDEN
        if mean:
            d_sp, d_se = (SE - theta) / YOUDEN**2, -(theta + SP - 1) / YOUDEN**2
        else:
            d_sp = d_se = -theta / YOUDEN**2
        full = var / YOUDEN**2 + d_se**2 * VAR_SE + d_sp**2 * VAR_SP
        out[name] = [float(corrected), float(fsqrt(full))]
    kept = len(pairs) + zero
    units = len(unit_complete)
    return {
        "coordinates": out,
        "pairs_formed": all_pairs,
        "pairs_kept": kept,
        "pairs_used": len(pairs),
        "zero_weight": zero,
        "dropped": [[i, reason] for i, reason in dropped],
        "units": units,
        "units_incomplete": sum(1 for v in unit_complete.values() if not v),
        "weight_sum": float(total),
        "kish": float(total**2 / sum(s * s for _, _, s in pairs)),
    }


def case(name, rows, lag=0, policy=None, serial=False, weights=None):
    out = estimate(rows, lag, policy, serial, weights)
    out.update(
        name=name, lag=lag, policy=policy, serial=serial, weights=weights,
        unit=[r["unit"] for r in rows], period=[r["period"] for r in rows],
        arm=[r["arm"] for r in rows], y=[r["y"] for r in rows],
    )
    return out


base = panel(60, 6)
# Unbalanced clusters: units 50..59 lose their last two periods.
ragged = panel(60, 6, {(u, t) for u in range(50, 60) for t in (4, 5)})


def with_missing(rows, cells):
    rows = [dict(r) for r in rows]
    for i, r in enumerate(rows):
        if (r["unit"], r["period"]) in cells:
            r["y"] = None
    return rows


holes = {(3, 2), (3, 3), (10, 0), (17, 5), (22, 1), (22, 4), (41, 2), (55, 1)}
missing = with_missing(ragged, holes)

# Weights: a repeating pattern, an extreme weight, and zero-weight rows.
def weight_pattern(rows):
    w = []
    for i, r in enumerate(rows):
        s = 1.0 + (i % 4)
        if i % 37 == 5:
            s = 1000.0
        if i % 53 == 7:
            s = 0.0
        w.append(s)
    return w


cases = [
    case("weights_varying_extreme_and_zero", base, weights=weight_pattern(base)),
    case("weights_unbalanced_clusters", ragged, weights=weight_pattern(ragged)),
    case("weights_scaled", base, weights=[1000.0 * w for w in weight_pattern(base)]),
    case("complete_case_lag_one", missing, lag=1, policy="complete_case"),
    case("available_case_lag_one", missing, lag=1, policy="available_case"),
    case("balanced_panel", missing, lag=0, policy="balanced_panel"),
    case("complete_case_with_weights", missing, lag=0, policy="complete_case", weights=weight_pattern(missing)),
    case("serial_dependence", ragged, serial=True),
    case("serial_dependence_weighted", ragged, serial=True, weights=weight_pattern(ragged)),
    case("serial_dependence_available_case", missing, lag=1, policy="available_case", serial=True),
    case("tiny", panel(5, 3), lag=0),
]
print(json.dumps({"cases": cases}, indent=1))
