# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Reference values for the numerical statistics of the target-native refuters.

Regenerate with:  uv run --script target_native_numerics_reference.py > target_native_numerics_reference.json

Written from the stated definitions:

  joint Mahalanobis   D2 = (e - m)' S^-1 (e - m) with m the replicate mean and S the unbiased replicate
                      covariance, in exact rational arithmetic (Gauss-Jordan over Fraction);
                      p = P(chi2_k > D2) from the regularised upper incomplete gamma function at 50 digits.
  finite difference   D(h) = (f(x+h) - f(x-h)) / (2h), D(h/2) = (f(x+h/2) - f(x-h/2)) / h,
                      richardson = (4 D(h/2) - D(h)) / 3, truncation = |D(h/2) - D(h)| / 3,
                      rounding = 8 eps max|f| / h, scale = tol (1 + |richardson|) + 4 truncation + rounding,
                      standardized = |reported - richardson| / scale, eps = 2^-52 (the double epsilon).
  local linearity     slope = (f(x+h) - f(x-h)) / (2h), curvature = (f(x+h) - 2 f(x) + f(x-h)) / h^2,
                      nonlinearity = |curvature| h / (2 max(|slope|, floor)).

Function values are doubles written as their exact values, and the response in the test is a lookup of
those values, so the comparison isolates the statistic.  Repository-authored (MIT OR Apache-2.0).
"""
import json
import random
from fractions import Fraction as F

import mpmath as mp

mp.mp.dps = 50
rng = random.Random(20261028)
EPS = F(1, 2 ** 52)


def solve(a, b):
    n = len(a)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        piv = next(r for r in range(c, n) if m[r][c] != 0)
        m[c], m[piv] = m[piv], m[c]
        inv = 1 / m[c][c]
        m[c] = [v * inv for v in m[c]]
        for r in range(n):
            if r != c and m[r][c] != 0:
                f = m[r][c]
                m[r] = [x - f * y for x, y in zip(m[r], m[c])]
    return [m[i][n] for i in range(n)]


def mahalanobis(expected, reps, k):
    r = len(reps)
    mean = [sum(F(row[j]) for row in reps) / r for j in range(k)]
    cov = [[sum((F(row[i]) - mean[i]) * (F(row[j]) - mean[j]) for row in reps) / (r - 1) for j in range(k)] for i in range(k)]
    d = [F(e) - m for e, m in zip(expected, mean)]
    y = solve(cov, d)
    return sum(a * b for a, b in zip(d, y))


def chi2_sf(x, k):
    x = mp.mpf(x.numerator) / x.denominator
    return mp.gammainc(mp.mpf(k) / 2, x / 2, mp.inf, regularized=True)


joint = []
for i in range(24):
    k = rng.randint(1, 5)
    r = rng.randint(k + 2, k + 40)
    mix = [[rng.gauss(0, 1) for _ in range(k)] for _ in range(k)]
    centre = [rng.gauss(0, 3) for _ in range(k)]
    axis = [rng.choice([0.01, 1.0, 100.0]) for _ in range(k)]
    reps = []
    for _ in range(r):
        z = [rng.gauss(0, 1) for _ in range(k)]
        reps.append([centre[j] + sum(mix[j][l] * z[l] for l in range(k)) * axis[j] for j in range(k)])
    expected = [centre[j] + rng.gauss(0, 2) for j in range(k)]
    d2 = mahalanobis(expected, reps, k)
    joint.append({"k": k, "expected": expected, "replicates": [v for row in reps for v in row],
                  "d2": mp.nstr(mp.mpf(d2.numerator) / d2.denominator, 30), "p": mp.nstr(chi2_sf(d2, k), 30)})
# a large common offset and a tiny replicate spread
reps = [[1e6 + 1e-3 * rng.gauss(0, 1), -1e6 + 2e-3 * rng.gauss(0, 1)] for _ in range(30)]
e = [1e6 + 2e-3, -1e6 - 1e-3]
d2 = mahalanobis(e, reps, 2)
joint.append({"k": 2, "expected": e, "replicates": [v for row in reps for v in row],
              "d2": mp.nstr(mp.mpf(d2.numerator) / d2.denominator, 30), "p": mp.nstr(chi2_sf(d2, 2), 30)})

fd = []
for i in range(30):
    x = rng.uniform(-2, 2)
    step = rng.choice([1e-1, 1e-2, 1e-3, 0.5])
    tol = rng.choice([1e-6, 1e-3, 1e-1])
    scale_f = rng.choice([1.0, 1e6])
    f = lambda t: scale_f * (mp.sin(t) + 0.3 * t ** 3)
    xs = {"p": x + step, "m": x - step, "p2": x + step / 2, "m2": x - step / 2}
    vals = {key: float(f(mp.mpf(v))) for key, v in xs.items()}
    reported = float(scale_f * (mp.cos(x) + 0.9 * x * x)) * rng.choice([1.0, 1.0, 1.0001, 1.1])
    fp, fm, fp2, fm2 = (F(vals[k_]) for k_ in ("p", "m", "p2", "m2"))
    h = F(step)
    dh = (fp - fm) / (2 * h)
    dh2 = (fp2 - fm2) / h
    rich = (4 * dh2 - dh) / 3
    trunc = abs(dh2 - dh) / 3
    mag = max(abs(fp), abs(fm), abs(fp2), abs(fm2))
    rounding = 8 * EPS * mag / h
    scale = F(tol) * (1 + abs(rich)) + 4 * trunc + rounding
    std = abs(F(reported) - rich) / scale
    fd.append({"x": x, "step": step, "tolerance": tol, "reported": reported, "values": vals,
               "richardson": float(rich), "truncation": float(trunc), "standardized": float(std)})

ll = []
for i in range(30):
    x = rng.uniform(-2, 2)
    step = rng.choice([0.5, 0.1, 1e-2])
    floor = rng.choice([1e-12, 1e-3, 1.0])
    f = lambda t: mp.exp(1.5 * t) + 0.1 * t
    xs = {"c": x, "p": x + step, "m": x - step}
    vals = {key: float(f(mp.mpf(v))) for key, v in xs.items()}
    f0, fp, fm = (F(vals[k_]) for k_ in ("c", "p", "m"))
    h = F(step)
    slope = (fp - fm) / (2 * h)
    curv = (fp - 2 * f0 + fm) / (h * h)
    nl = abs(curv) * h / (2 * max(abs(slope), F(floor)))
    ll.append({"x": x, "step": step, "floor": floor, "values": vals,
               "slope": float(slope), "curvature": float(curv), "nonlinearity": float(nl)})
print(json.dumps({"joint": joint, "finite_difference": fd, "local_linearity": ll}, indent=1))
