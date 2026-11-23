"""Independent two-dimensional quadrature reference for hierarchical choice EVSI."""

import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad_vec
from scipy.special import gammaln

COUNTS = [(3, 1), (1, 3), (2, 2), (4, 0), (0, 4), (3, 1), (2, 2)]
MU = math.log(5.0)


def likelihood(p, v, a, b):
    k = math.exp(v)
    return math.exp(
        gammaln(k)
        - gammaln(k + a + b)
        + gammaln(k * p + a)
        - gammaln(k * p)
        + gammaln(k * (1 - p) + b)
        - gammaln(k * (1 - p))
    )


def integrand(p, v):
    if p <= 0 or p >= 1:
        return np.zeros(12)
    base = math.exp(-0.5 * (v - MU) ** 2) / math.sqrt(2 * math.pi)
    for a, b in COUNTS:
        base *= likelihood(p, v, a, b)
    out = [base, base * p]
    for n in (1, 2):
        for a in range(n + 1):
            weight = base * math.comb(n, a) * likelihood(p, v, a, n - a)
            out.extend((weight, weight * p))
    return np.array(out)


def over_p(p):
    return quad_vec(
        lambda v: integrand(p, v), MU - 9, MU + 9, epsabs=1e-12, epsrel=1e-9
    )[0]


result = quad_vec(over_p, 1e-10, 1 - 1e-10, epsabs=1e-12, epsrel=1e-9)[0]
z, first = result[:2]
now = max(first, z - first) / z
values = []
i = 2
for n in (1, 2):
    future = 0.0
    mass = 0.0
    for _ in range(n + 1):
        weight, weighted_p = result[i : i + 2]
        i += 2
        mass += weight
        future += max(weighted_p, weight - weighted_p)
    assert abs(mass / z - 1) < 1e-6
    values.append(float(future / z - now))
Path(__file__).with_suffix(".json").write_text(
    json.dumps({"evsi": values}, indent=2) + "\n"
)
