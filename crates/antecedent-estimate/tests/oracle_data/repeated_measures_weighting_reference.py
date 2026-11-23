# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Reference for adaptive_inference.repeated_measures_weighting.

Regenerate with:  uv run --offline --script repeated_measures_weighting_reference.py > repeated_measures_weighting_reference.json

Subjects contribute several measurements each.  The route reduces a subject to the mean of its
measurements and weights the subject score by the number of measurements (a unit multiplier).  From
the raw measurements this script builds, at 60 digits (decimal) and from the definition:

    ybar_i  = (sum of the subject's measurements) / m_i,
    m_i(a)  = (strength * mean + sum of earlier subject means attributed to a) / (strength + earlier count),
    Gamma_i(a) = m_i(a) + 1[A_i = a] (ybar_i - m_i(a)) / e_i(a),
    h_i     = m_i-count * (1, or (sum over the estimand's arms of 1 / e_i(arm))^(-1/2)),
    estimate = sum h Gamma / sum h,  variance = sum h^2 (Gamma - estimate)^2 / (sum h)^2,
    effective units = (sum h)^2 / sum h^2,  largest share = max h / sum h.

Designs: unequal subject sizes (1 to 1e6), two and three arms, adaptive assignment rows as small as
1/1024, arm-mean and contrast estimands, uniform and variance-stabilizing weights, a prior mean far
from the data.  Repository-authored (MIT OR Apache-2.0); standard library only.
"""
import json
import random
from decimal import Decimal, getcontext
from fractions import Fraction as F

getcontext().prec = 60


def dec(x):
    f = F(x)
    return Decimal(f.numerator) / Decimal(f.denominator)


def run(arms, assigned, rows, measurements, estimand, scheme, strength, mean, floor):
    sums, counts = [Decimal(0)] * arms, [0] * arms
    values, weights = [], []
    for a_i, row, ys in zip(assigned, rows, measurements):
        size = len(ys)
        ybar = sum(dec(y) for y in ys) / Decimal(size)
        scores = []
        for a in range(arms):
            model = (dec(strength) * dec(mean) + sums[a]) / (dec(strength) + counts[a])
            scores.append(model + ((ybar - model) / dec(row[a]) if a == a_i else Decimal(0)))
        sums[a_i] += ybar
        counts[a_i] += 1
        if estimand[0] == "arm":
            members, v = [estimand[1]], scores[estimand[1]]
        else:
            members, v = [estimand[1], estimand[2]], scores[estimand[1]] - scores[estimand[2]]
        base = Decimal(1) if scheme == "uniform" else sum(1 / dec(row[a]) for a in members) ** Decimal("-0.5")
        values.append(v)
        weights.append(base * size)
    total = sum(weights)
    est = sum(h * v for h, v in zip(weights, values)) / total
    var = sum((h * (v - est)) ** 2 for h, v in zip(weights, values)) / total**2
    sq = sum(h * h for h in weights)
    return {
        "estimate": float(est),
        "variance": float(var),
        "weight_sum": float(total),
        "effective_units": float(total**2 / sq),
        "max_weight_share": float(max(weights) / total),
    }


rng = random.Random(20241028)
cases = []


def case(name, arms, n, estimand, scheme, strength, mean, sizes, tiny=False):
    assigned, rows, measurements = [], [], []
    for i in range(n):
        raw = [F(rng.randint(1, 1023), 1024) if tiny and rng.random() < 0.3 else F(rng.randint(1, 9), 10) for _ in range(arms)]
        if tiny and i % 3 == 0:
            raw[0] = F(1, 1024)
        total = sum(raw)
        row = [r / total for r in raw]
        a = rng.choices(range(arms), weights=[float(r) for r in row])[0]
        size = sizes[i % len(sizes)]
        ys = [F(rng.randint(-40, 60), 7) for _ in range(size)]
        assigned.append(a)
        rows.append(row)
        measurements.append(ys)
    floor = F(1, 4096)
    out = run(arms, assigned, rows, measurements, estimand, scheme, strength, mean, floor)
    cases.append(
        {
            "name": name,
            "arms": arms,
            "assigned": assigned,
            "probabilities": [[f"{x.numerator}/{x.denominator}" for x in row] for row in rows],
            "measurements": [[f"{y.numerator}/{y.denominator}" for y in ys] for ys in measurements],
            "estimand": estimand,
            "scheme": scheme,
            "strength": strength,
            "mean": f"{F(mean).numerator}/{F(mean).denominator}",
            "floor": "1/4096",
            "expected": out,
        }
    )


case("two_arm_mean_uniform_unequal_sizes", 2, 9, ["arm", 1], "uniform", 2, F(1, 2), [1, 5, 2, 12, 3])
case("two_arm_contrast_variance_stabilizing", 2, 10, ["contrast", 1, 0], "variance_stabilizing", 1, F(0), [4, 1, 9, 2])
case("three_arm_contrast_small_probabilities", 3, 12, ["contrast", 2, 0], "variance_stabilizing", 3, F(10), [1, 2, 3, 20], True)
case("arm_mean_with_prior_far_from_data", 3, 8, ["arm", 1], "uniform", 100, F(1000), [2, 3], True)
case("single_dominant_subject", 2, 6, ["arm", 0], "uniform", 1, F(1, 3), [400, 1, 1, 1, 1, 1])
print(json.dumps({"cases": cases}, indent=1))
