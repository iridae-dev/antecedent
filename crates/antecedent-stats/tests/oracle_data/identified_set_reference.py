# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Imbens-Manski critical value and finite-set intervals, in 40-digit arithmetic.

Regenerate with:  uv run --script identified_set_reference.py > identified_set_reference.json

Critical value: for a level L and standardised width r >= 0, c >= 0 solves
Phi(c + r) - Phi(-c) = L; r = 0 gives the two-sided c = z_{(1+L)/2}, found here by root-finding on
the normal distribution function.
Influence interval: completions g = 1..G have points t_g and score columns phi_ig on n shared rows.
With centred scores, sigma_g = sqrt(sum_i phi_ig^2 / (n (n - 1))) and the width standard error is
the same quantity for the difference of the highest and lowest completion's scores.  The width
t_max - t_min is retained when it exceeds sqrt(max(ln max(n, 3), 1)) times that standard error;
then r = width / max_g sigma_g, else r = 0.  The interval is
[min_g (t_g - c sigma_g), max_g (t_g + c sigma_g)], and the endpoint standard errors are
(t_min - lower) / c and (upper - t_max) / c.
Replicate interval: sigma_g is the sample standard deviation of completion g's replicate draws
times a scale (replicates with a non-finite refit are dropped for every completion), the width
standard error is the scaled sample standard deviation of the per-replicate (max - min), and the
rest is as above.
Repository-authored (MIT OR Apache-2.0).
"""
import json
import mpmath as mp

mp.mp.dps = 40


def critical(level, r):
    level = mp.mpf(level)
    r = mp.mpf(r)
    return mp.findroot(lambda c: mp.ncdf(c + r) - mp.ncdf(-c) - level, mp.mpf("1.5"))


def sample_sd(values):
    n = len(values)
    mean = sum(values) / n
    return mp.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))


def interval(points, sds, width_se, rows, level):
    points = [mp.mpf(p) for p in points]
    lo_p, hi_p = min(points), max(points)
    width = hi_p - lo_p
    kappa = max(mp.sqrt(mp.log(max(rows, 3))), 1)
    retained = len(points) > 1 and width > kappa * width_se and width > 0
    scale = max(sds)
    r = width / scale if retained and scale > 0 else mp.mpf(0)
    c = critical(level, r)
    lower = min(p - c * s for p, s in zip(points, sds))
    upper = max(p + c * s for p, s in zip(points, sds))
    return dict(level=level, critical=float(c), width_retained=bool(retained), lower=float(lower),
                upper=float(upper), lower_se=float((lo_p - lower) / c), upper_se=float((upper - hi_p) / c),
                bound_lower=float(lo_p), bound_upper=float(hi_p))


def influence_case(name, points, n, offsets, level, mix=1.0):
    scores = []
    for g in range(len(points)):
        own = [mp.mpf((((i * (5 + 2 * g) + 3 * g) % 23) - 11) / 11.0) * (1 + g) for i in range(n)]
        common = [mp.mpf((((i * 3 + 1) % 17) - 8) / 8.0) for i in range(n)]
        col = [mix * o + (1 - mix) * c + offsets[g] for o, c in zip(own, common)]
        scores.append([float(v) for v in col])
    cent = []
    for col in scores:
        m = sum(mp.mpf(v) for v in col) / n
        cent.append([mp.mpf(v) - m for v in col])
    sds = [mp.sqrt(sum(v * v for v in col) / (n * (n - 1))) for col in cent]
    hi = max(range(len(points)), key=lambda g: points[g])
    lo = min(range(len(points)), key=lambda g: points[g])
    width_se = mp.sqrt(sum((a - b) ** 2 for a, b in zip(cent[hi], cent[lo])) / (n * (n - 1)))
    return dict(name=name, points=points, scores=scores, rows=n, **interval(points, sds, width_se, n, level))


def replicate_case(name, points, replicates, scale, rows, level, bad_rows):
    draws = []
    for b in range(replicates):
        draws.append([points[g] + (((b * (7 + 3 * g) + 2 * g) % 29) - 14) / 14.0 * (0.4 + 0.3 * g) for g in range(len(points))])
    for b in bad_rows:
        draws[b][0] = float("nan")
    kept = [[mp.mpf(v) for v in d] for d in draws if all(v == v for v in d)]
    sds = [sample_sd([d[g] for d in kept]) * mp.mpf(scale) for g in range(len(points))]
    width_se = sample_sd([max(d) - min(d) for d in kept]) * mp.mpf(scale)
    out = interval(points, sds, width_se, rows, level)
    return dict(name=name, points=points, draws=[[None if v != v else v for v in d] for d in draws],
                scale=scale, rows=rows, kept=len(kept), **out)


crit = [dict(level=lv, r=r, c=float(critical(lv, r))) for lv in (0.8, 0.9, 0.95, 0.99) for r in (0.0, 0.3, 1.0, 2.5, 8.0)]
cases = dict(
    critical=crit,
    influence=[
        influence_case("separated_completions", [0.0, 1.5, 3.4], 80, [0.2, -0.1, 0.4], 0.95),
        influence_case("correlated_moderate_width", [0.0, 0.07], 80, [0.0, 0.0], 0.95, 0.02),
        influence_case("nearly_equal_completions", [1.0, 1.01, 0.995], 60, [0.0, 0.1, -0.2], 0.9),
        influence_case("single_completion", [2.0], 50, [0.3], 0.95),
    ],
    replicates=[
        replicate_case("shared_resamples", [0.0, 2.0, 4.5], 200, 1.15, 300, 0.95, []),
        replicate_case("moderate_width_resamples", [0.0, 0.9], 200, 1.0, 300, 0.95, []),
        replicate_case("dropped_non_finite_replicates", [0.0, 0.2, 5.0], 150, 1.0, 120, 0.9, [3, 17]),
    ],
)
print(json.dumps(cases, separators=(",", ":")))
