# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Independent bootstrap-t reference for the dependent unit interval of the finite two-step sequence.

Regenerate in two stages (the replicate ids come from the product's keyed identity words, so
they are inputs of this reference, not outputs of it; everything downstream of them is derived
here):

  python3 temporal_dependent_interval_reference.py inputs > inputs.json
  (the ignored dump test in temporal_dependent_interval_reference.rs reads inputs.json, runs the
   product on every scenario and writes stream.json with each scenario's replicate ids and
   points)
  uv run --script temporal_dependent_interval_reference.py final inputs.json stream.json \
      > temporal_dependent_interval_reference.json

What is derived here, without the product:

* the panels (units of complete binary two-step histories with one shared, unequal multiplicity
  per cell and unit-specific outcomes in eighths) and their unit scores: the observed-state,
  fixed-state and target-law responses
      sum_l P(l | s0, a1) E[y | s0, a1, l, a2],  P(l | s0, a1) = n(s0, a1, l) / n(s0, a1),
  evaluated on one unit as exact fractions;
* the pooled estimate on the whole panel and on every resample, from the histories themselves, so
  that the mean-of-unit-scores identity is checked and not assumed;
* the unit draws: splitmix64 from mix64(replicate_id + gamma); draw i of a resample of n units is
  the unit with index floor(out_i * n / 2^64);
* the original standard error sqrt(s^2 / n) of the unit scores (s^2 with n - 1), each resample's
  own standard error from its selected scores, a resample of exactly zero variance dropped
  (exact fractions decide this, not floating point), the pivots (theta* - theta) / se*, the
  type-7 quantiles of the sorted pivots at (1 - level) / 2 and 1 - (1 - level) / 2, and the
  equal-tailed bootstrap-t endpoints [theta - q_high se, theta - q_low se], at 50 digits.

Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import json
import sys
from fractions import Fraction as F

from mpmath import mp, mpf, sqrt

mp.dps = 50
MASK = (1 << 64) - 1
GAMMA = 0x9E3779B97F4A7C15


def mix64(z):
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return z ^ (z >> 31)


def draws(replicate_id, n):
    state = mix64((replicate_id + GAMMA) & MASK)
    out = []
    for _ in range(n):
        state = (state + GAMMA) & MASK
        out.append(min((mix64(state) * n) >> 64, n - 1))
    return out


CELLS = [(s0, a1, l, a2) for s0 in range(2) for a1 in range(2) for l in range(2) for a2 in range(2)]


def multiplicity(panel, cell):
    s0, a1, l, a2 = cell
    if panel == "A":
        return 1 + ((3 * s0 + 5 * a1 + 7 * l + 11 * a2) % 3)
    return 1 + ((s0 + 2 * a1 + l + a2) % 2)


def make_unit(panel, unit_id, rng_state):
    rows, t = [], 0
    for cell in CELLS:
        for _ in range(multiplicity(panel, cell)):
            rng_state = (rng_state * 6364136223846793005 + 1442695040888963407) & MASK
            y = F((rng_state >> 33) % 17, 8)
            rows.append([t, *cell, float(y)])
            t += 1
    return {"unit_id": unit_id, "rows": rows}, rng_state


def make_panels():
    panels = {}
    state = 12345
    units = []
    for i in range(24):
        unit, state = make_unit("A", i, state)
        units.append(unit)
    panels["A"] = {"snapshot": "bootstrap-t-panel-a", "units": units}
    # Panel B: four units, units 100 and 101 hold identical outcomes (tied scores), so a resample
    # that draws only those two has exactly zero variance.
    state = 777
    units = []
    for i in range(4):
        unit, state = make_unit("B", 100 + i, state)
        units.append(unit)
    twin = json.loads(json.dumps(units[0]))
    twin["unit_id"] = 101
    units[1] = twin
    panels["B"] = {"snapshot": "bootstrap-t-panel-b", "units": units}
    return panels


SCENARIOS = [
    {
        "name": "observed_state_a",
        "panel": "A",
        "estimator": {"kind": "observed", "sequence": [1, 0]},
        "config": {"replicates": 200, "seed": 31, "level": 0.95, "min_units": 20, "max_failed_fraction": 0.05},
    },
    {
        "name": "fixed_state_a",
        "panel": "A",
        "estimator": {"kind": "fixed", "sequence": [0, 1], "s0": 1},
        "config": {"replicates": 120, "seed": 32, "level": 0.9, "min_units": 20, "max_failed_fraction": 0.05},
    },
    {
        "name": "marginalized_a",
        "panel": "A",
        "estimator": {"kind": "marginalized", "sequence": [1, 1], "law": [[0, 0.25], [1, 0.75]]},
        "config": {"replicates": 80, "seed": 33, "level": 0.8, "min_units": 20, "max_failed_fraction": 0.05},
    },
    {
        "name": "tied_scores_b",
        "panel": "B",
        "estimator": {"kind": "observed", "sequence": [0, 0]},
        "config": {"replicates": 200, "seed": 34, "level": 0.95, "min_units": 3, "max_failed_fraction": 0.2},
    },
]


def estimate(histories, estimator):
    """Pooled response from raw histories (s0, a1, l, a2, y), straight from the definition."""
    seq = estimator["sequence"]

    def response(s0):
        followed = [h for h in histories if h[0] == s0 and h[1] == seq[0]]
        total = len(followed)
        out = F(0)
        for l in (0, 1):
            at_l = [h for h in followed if h[2] == l]
            if not at_l:
                continue
            took = [h[4] for h in at_l if h[3] == seq[1]]
            out += F(len(at_l), total) * sum((F(y) for y in took), F(0)) / len(took)
        return out

    kind = estimator["kind"]
    if kind == "fixed":
        return response(estimator["s0"])
    if kind == "observed":
        n = len(histories)
        return sum((F(sum(1 for h in histories if h[0] == s0), n) * response(s0) for s0 in (0, 1)), F(0))
    return sum((F(mass) * response(s0) for s0, mass in estimator["law"]), F(0))


def histories_of(unit):
    return [tuple(r[1:]) for r in unit["rows"]]


def variance(values):
    n = len(values)
    mean = sum(values, F(0)) / n
    return mean, sum(((v - mean) ** 2 for v in values), F(0)) / (n - 1)


def type7(sorted_values, p):
    h = mpf(p) * (len(sorted_values) - 1)
    lo = int(mp.floor(h))
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (h - lo) * (sorted_values[hi] - sorted_values[lo])


def mp_of(fraction):
    return mpf(fraction.numerator) / mpf(fraction.denominator)


def solve(panel, scenario, stream):
    units = panel["units"]
    n = len(units)
    estimator, config = scenario["estimator"], scenario["config"]
    scores = [estimate(histories_of(u), estimator) for u in units]
    point = estimate([h for u in units for h in histories_of(u)], estimator)
    mean, var = variance(scores)
    assert mean == point, "pooled estimate is not the mean of the unit scores"
    standard_error = sqrt(mp_of(var) / n)
    pivots, errors, failed = [], [], 0
    for rid, product_point in zip(stream["replicate_ids"], stream["points"]):
        picks = draws(rid, n)
        chosen = [h for k in picks for h in histories_of(units[k])]
        theta = estimate(chosen, estimator)
        assert theta == sum((scores[k] for k in picks), F(0)) / n, "resample is not the mean of selected scores"
        assert product_point is not None and abs(float(theta) - product_point) < 1e-12, (
            "regenerated unit draw does not reproduce the product's replicate point"
        )
        _, resample_var = variance([scores[k] for k in picks])
        if resample_var == 0:
            failed += 1
            errors.append(None)
            pivots.append(None)
            continue
        se = sqrt(mp_of(resample_var) / n)
        errors.append(se)
        pivots.append(mp_of(theta - point) / se)
    kept = sorted(p for p in pivots if p is not None)
    tail = (1 - mpf(config["level"])) / 2
    low, high = type7(kept, tail), type7(kept, 1 - tail)
    text = lambda v: None if v is None else mp.nstr(v, 40)
    return {
        "point": float(point),
        "unit_scores": [float(s) for s in scores],
        "standard_error": text(standard_error),
        "replicate_standard_errors": [text(e) for e in errors],
        "pivots": [text(p) for p in pivots],
        "failed": failed,
        "quantile_low": text(low),
        "quantile_high": text(high),
        "lower": text(mp_of(point) - high * standard_error),
        "upper": text(mp_of(point) - low * standard_error),
    }


def main():
    mode = sys.argv[1]
    panels = make_panels()
    if mode == "inputs":
        print(json.dumps({"panels": panels, "scenarios": SCENARIOS}))
        return
    stream = json.load(open(sys.argv[3]))
    out = {"panels": panels, "scenarios": []}
    for scenario in SCENARIOS:
        record = dict(scenario)
        record["stream"] = stream[scenario["name"]]
        record["expected"] = solve(panels[scenario["panel"]], scenario, stream[scenario["name"]])
        out["scenarios"].append(record)
    print(json.dumps(out))


main()
