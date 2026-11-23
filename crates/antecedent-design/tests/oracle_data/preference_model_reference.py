"""Exact rational reference for the linear form of value models.

Regenerate with:  python3 preference_model_reference.py > preference_model_reference.json

Each model is written directly as a closed-form function f(x, theta) of the attributes x and
the preference parameters theta, with fractions.Fraction throughout. The linear form is
recovered by evaluation at unit parameter vectors, not by walking an expression:
  fixed(x)    = f(x, 0)
  term_j(x)   = f(x, e_j) - f(x, 0)
  value(x, t) = f(x, t)
which is exact for a value affine in theta. A check confirms value = fixed + sum_j t_j term_j
in exact arithmetic. Inputs are dyadic rationals with small denominators, so every input and
every reported number is exactly the f64 the Rust test receives.

Models:
  rates        w_benefit * benefit - w_harm * harm - cost
  curve        sum_i s_i * (clamp(dose, b_i, b_{i+1}) - b_i) - w_cost * cost,
               breakpoints 0, 1, 5/2, 4
  excess       w_benefit * (benefit + 1/2) - w_wait * max(wait - 3, 0)
               - max(1 - benefit, 0) / 4
  interaction  w_joint * benefit * adherence + w_both * min(benefit, adherence)
               - w_joint * adherence + 2 * w_both

Repository-authored (MIT OR Apache-2.0); standard library only.
"""
import json
import random
from fractions import Fraction as F


def clamp(x, low, high):
    return min(max(x, low), high)


def rates(x, t):
    return t[0] * x[0] - t[1] * x[1] - x[2]


BREAKS = [F(0), F(1), F(5, 2), F(4)]


def curve(x, t):
    total = F(0)
    for i in range(3):
        total += t[i] * (clamp(x[0], BREAKS[i], BREAKS[i + 1]) - BREAKS[i])
    return total - t[3] * x[1]


def excess(x, t):
    return t[0] * (x[0] + F(1, 2)) - t[1] * max(x[1] - 3, F(0)) - max(1 - x[0], F(0)) / 4


def interaction(x, t):
    return t[0] * x[0] * x[1] + t[1] * min(x[0], x[1]) - t[0] * x[1] + 2 * t[1]


MODELS = [
    ("rates", rates, 3, 2),
    ("curve", curve, 2, 4),
    ("excess", excess, 2, 2),
    ("interaction", interaction, 2, 2),
]


def exact_float(value):
    as_float = float(value)
    assert F(as_float) == value, value
    return as_float


def main():
    rng = random.Random(20261123)
    out = []
    for name, f, n_attributes, n_parameters in MODELS:
        points = []
        for _ in range(8):
            x = [F(rng.randint(-48, 48), 8) for _ in range(n_attributes)]
            t = [F(rng.randint(-48, 48), 16) for _ in range(n_parameters)]
            zero = [F(0)] * n_parameters
            fixed = f(x, zero)
            terms = []
            for j in range(n_parameters):
                unit = [F(1) if i == j else F(0) for i in range(n_parameters)]
                terms.append(f(x, unit) - fixed)
            value = f(x, t)
            assert value == fixed + sum(tj * term for tj, term in zip(t, terms))
            points.append(
                {
                    "attributes": [exact_float(v) for v in x],
                    "parameters": [exact_float(v) for v in t],
                    "fixed": exact_float(fixed),
                    "terms": [exact_float(v) for v in terms],
                    "value": exact_float(value),
                }
            )
        out.append({"name": name, "points": points})
    print(json.dumps({"models": out}, indent=1))


if __name__ == "__main__":
    main()
