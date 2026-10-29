# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Exact rational reference for the two-lag Bayesian temporal DAG fixtures.

Regenerate with:  uv run --offline --script bayesian_temporal_two_lag_reference.py > bayesian_temporal_two_lag_reference.json
Verify with:      python3 bayesian_temporal_two_lag_reference.py --check

The fixture is y_i = s * (2 t_{i-1} + 4 t_{i-2} + 0.12 e_i) with t, e independent standard normals
and s the data scale (1 and 1.2).  The rows are the exact doubles the test generates, read from
bayesian_temporal_two_lag_rows.json and taken as exact Fractions.  The horizon-1 pulse query fits
y_i on [1, t_{i-1}] over i = 1..n-1 and the horizon-2 query fits y_i on [1, t_{i-2}] over
i = 2..n-1; the other lag is an independent omitted regressor, so each fit is the marginal slope
    slope = Sxy / Sxx,   Sxx = sum (x - mean x)^2,   Sxy = sum (x - mean x)(y - mean y),
and its sampling error around the analytic coefficient (2 s and 4 s) is the omitted lag plus the
noise projected on x.  The analytic standard error is
    se = sqrt(rss / (m - 2) / Sxx)
of that regression.  Slopes and errors are written as 40-digit decimal strings.
Repository-authored (MIT OR Apache-2.0); standard library only.
"""
import json
import os
import sys
from decimal import Decimal, getcontext
from fractions import Fraction as F

getcontext().prec = 60
HERE = os.path.dirname(os.path.abspath(__file__))


def dec(x):
    x = F(x)
    return str(Decimal(x.numerator) / Decimal(x.denominator))[:48]


def fit(x, y):
    m = len(x)
    mx, my = sum(x) / m, sum(y) / m
    sxx = sum((a - mx) ** 2 for a in x)
    slope = sum((a - mx) * (b - my) for a, b in zip(x, y)) / sxx
    rss = sum((b - my - slope * (a - mx)) ** 2 for a, b in zip(x, y))
    return slope, rss / (m - 2) / sxx


def build():
    rows = json.load(open(os.path.join(HERE, "bayesian_temporal_two_lag_rows.json")))
    t = [F(v) for v in rows["t"]]
    n = len(t)
    out = {}
    for name, scale in (("y_scale_1", 1), ("y_scale_1_2", F(6, 5))):
        y = [F(v) for v in rows[name]]
        for h in (1, 2):
            idx = range(h, n)
            slope, var = fit([t[i - h] for i in idx], [y[i] for i in idx])
            truth = scale * (2 if h == 1 else 4)
            # Normalise to a float truth for the decimal comparison: 1.2 * 2 = 2.4 exactly.
            out[f"{name}.horizon_{h}"] = {
                "rows": len(idx),
                "marginal_slope": dec(slope),
                "analytic_coefficient": dec(truth),
                "sampling_error": dec(slope - truth),
                "standard_error": f"{float(var) ** 0.5:.12f}",
                "error_in_standard_errors": f"{float(slope - truth) / float(var) ** 0.5:.6f}",
            }
    return out


if __name__ == "__main__":
    result = build()
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if "--check" in sys.argv:
        stored = open(os.path.join(HERE, "bayesian_temporal_two_lag_reference.json")).read()
        if stored != text:
            sys.exit("bayesian_temporal_two_lag_reference.json is stale; regenerate it")
        print("ok")
    else:
        sys.stdout.write(text)
