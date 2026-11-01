"""Independent two-cell logistic EB and penalized-score solution for the n=120 fixture.

The saturated two-group logit has closed-form pilot coefficients and slope variance
1/I0 + 1/I1. A separate two-variable Newton solve finds the penalized root.
This script uses only Python's standard library and imports no Antecedent code.
"""
import json
import math
from pathlib import Path

n = 120
outcomes = [(0 if i < n // 2 else 1) for i in range(n)]
y = [1 - x if i % 8 == 0 else x for i, x in enumerate(outcomes)]
n0 = outcomes.count(0)
n1 = n - n0
s0 = sum(v for x, v in zip(outcomes, y) if x == 0)
s1 = sum(v for x, v in zip(outcomes, y) if x == 1)
p0, p1 = s0 / n0, s1 / n1
pilot0 = math.log(p0 / (1 - p0))
pilot1 = math.log(p1 / (1 - p1)) - pilot0
information0 = n0 * p0 * (1 - p0)
information1 = n1 * p1 * (1 - p1)
slope_variance = 1 / information0 + 1 / information1
strength = min(max(1 / max(pilot1 * pilot1 - slope_variance, 1e-8), 1e-4), 1e3)
a = b = 0.0
for _ in range(100):
    q0 = 1 / (1 + math.exp(-a))
    q1 = 1 / (1 + math.exp(-a - b))
    w0, w1 = n0 * q0 * (1 - q0), n1 * q1 * (1 - q1)
    g0 = s0 - n0 * q0 + s1 - n1 * q1
    g1 = s1 - n1 * q1 - strength * b
    determinant = (w0 + w1) * (w1 + strength) - w1 * w1
    da = ((w1 + strength) * g0 - w1 * g1) / determinant
    db = ((w0 + w1) * g1 - w1 * g0) / determinant
    a += da
    b += db
    if max(abs(da), abs(db)) < 1e-14:
        break
else:
    raise RuntimeError("penalized score did not converge")
assert max(abs(g0), abs(g1)) < 1e-11
path = Path(__file__).with_suffix('.json')
path.write_text(json.dumps({'n': n, 'strength': strength, 'coefficients': [a, b]}, indent=2) + '\n')
