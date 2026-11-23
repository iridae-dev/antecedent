# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""High-precision reference values for the special-function implementations.

Regenerate with:  uv run --script special_functions_reference.py > special_functions_reference.json

Every argument is an exact binary64 value; every expected value is computed from the
mathematical definition (mpmath erfc, ncdf, betainc, gammainc, loggamma) at 60-80 digits and
rounded once to binary64.  Quantiles are solved by bisection on the high-precision CDF, not by
inverting an approximation.  This script and its frozen output are repository-authored
(MIT OR Apache-2.0); mpmath is used only here, never at product run time.
"""
import json
import math
import sys

import mpmath as mp

mp.mp.dps = 50
out = {"erf": [], "erfc": [], "norm_cdf": [], "norm_sf": [], "norm_inv": [], "log_norm_cdf": [],
       "student_t_sf": [], "student_t_ppf": [], "gamma_q": [], "ln_gamma": []}


def f(v):
    return float(v)


def nxt(x, up):
    return math.nextafter(x, math.inf if up else -math.inf)


# erf / erfc / normal CDF and survival: approximation-region boundaries and the whole tail.
xs = [0.0, 1e-300, 1e-10, 0.01, 0.2, 0.46875, nxt(0.46875, False), nxt(0.46875, True), 0.5, 0.9,
      1.5, 2.5, 3.5, 3.999999, 4.0, nxt(4.0, False), nxt(4.0, True), 4.5, 6.0, 9.0, 13.0, 18.0,
      24.0, 26.0, 26.5, 26.6, 27.0, 27.2]
for x in xs:
    for s in (1.0, -1.0):
        v = s * x
        out["erf"].append([v, f(mp.erf(mp.mpf(v)))])
        out["erfc"].append([v, f(mp.erfc(mp.mpf(v)))])
zs = [0.0, 1e-5, 0.3, 1.0, 1.6, 1.959963984540054, 2.9, 3.7, 5.0, 5.65685424949238, 5.656854249492381,
      7.0, 8.2, 12.0, 20.0, 30.0, 36.0, 37.5, 38.0, 38.4]
for z in zs:
    for s in (1.0, -1.0):
        v = s * z
        out["norm_cdf"].append([v, f(mp.ncdf(mp.mpf(v)))])
        out["norm_sf"].append([v, f(1 - mp.ncdf(mp.mpf(v)) if v < 0 else mp.ncdf(-mp.mpf(v)))])
for v in [-40.0, -37.0, -20.0, -8.0, nxt(-8.0, True), nxt(-8.0, False), -7.9, -2.0, -0.5, 0.0, 0.5, 4.0, 9.0, 15.0]:
    with mp.workdps(400):  # ln(1 - 4e-51) needs far more digits than the working precision
        out["log_norm_cdf"].append([v, f(mp.log(mp.ncdf(mp.mpf(v))))])


def ncdf_lower(x):
    return mp.ncdf(x)


def norm_inv_ref(p):
    p = mp.mpf(p)
    if p > mp.mpf("0.5"):
        return -norm_inv_ref_low(1 - p)
    return norm_inv_ref_low(p)


def norm_inv_ref_low(p):
    # solve log Phi(x) = log p by bisection in mp (p <= 1/2)
    lo, hi = mp.mpf(-40), mp.mpf(0)
    lp = mp.log(p)
    for _ in range(300):
        mid = (lo + hi) / 2
        if mp.log(mp.ncdf(mid)) < lp:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


ps = [1e-300, 1e-200, 1e-100, 1e-30, 1e-16, 1e-9, 1e-5, 0.001, 0.02, 0.02424, 0.02425, nxt(0.02425, True),
      0.024251, 0.1, 0.25, 0.4999, 0.5, 0.6, 0.9, 0.97574, 0.97575, 0.975751, 0.99, 0.999999,
      1 - 1e-9, 1 - 1e-12, 1 - 1e-15, nxt(1.0, False)]
for p in ps:
    out["norm_inv"].append([p, f(norm_inv_ref(p))])


def t_tail_by_quadrature(t, df):
    # P(T > t), t >= 0, by tanh-sinh quadrature of the density; used only where the
    # hypergeometric evaluation of the incomplete beta does not converge (very large df).
    logc = mp.loggamma((df + 1) / 2) - mp.loggamma(df / 2) - mp.log(mp.pi * df) / 2

    def pdf(s):
        return mp.exp(logc - (df + 1) / 2 * mp.log1p(s * s / df))

    scale = mp.sqrt(df / (df + 1)) if df > 1 else mp.mpf(1)
    pts = [t] + [t + scale * k for k in (0.5, 1, 2, 4, 8, 16, 32, 64)] + [t * k for k in (1.5, 2, 4, 10, 100, 10**4)] + [mp.inf]
    pts = sorted(set(pts), key=lambda v: (v == mp.inf, v))
    return mp.quad(pdf, pts)


def t_sf(t, df):
    t, df = mp.mpf(t), mp.mpf(df)
    a = abs(t)
    x = df / (df + a * a)
    try:
        tail = mp.betainc(df / 2, mp.mpf(1) / 2, 0, x, regularized=True) / 2
    except (mp.libmp.libhyper.NoConvergence, ValueError):
        tail = t_tail_by_quadrature(a, df)
    return tail if t >= 0 else 1 - tail


for df in [0.05, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0, 100.0, 1e3, 1e4, 1e5, 1e6, 9.9e6, 1e8, 1e12]:
    for t in [0.0, 1e-9, 0.1, 0.7, 1.0, 2.0, 3.5, 8.0, 40.0, 1e3, 1e5, -0.3, -2.0, -9.0]:
        out["student_t_sf"].append([t, df, f(t_sf(t, df))])


def t_pdf(t, df):
    logc = mp.loggamma((df + 1) / 2) - mp.loggamma(df / 2) - mp.log(mp.pi * df) / 2
    return mp.exp(logc - (df + 1) / 2 * mp.log1p(t * t / df))


def t_ppf(p, df):
    p, df = mp.mpf(p), mp.mpf(df)
    sign = -1 if p < mp.mpf("0.5") else 1
    target = 1 - p if p > mp.mpf("0.5") else p
    # geometric bracket, then bisection to a 1% bracket, then Newton on sf(t) - target
    lo, hi = mp.mpf(0), mp.mpf(1)
    while t_sf(hi, df) > target:
        lo, hi = hi, hi * 4
    for _ in range(40):
        mid = (lo + hi) / 2
        if t_sf(mid, df) > target:
            lo = mid
        else:
            hi = mid
        if hi - lo < (lo + hi) * mp.mpf("0.0005"):
            break
    t = (lo + hi) / 2
    for _ in range(12):
        t = t + (t_sf(t, df) - target) / t_pdf(t, df)
    return sign * t


for df in [0.3, 1.0, 2.0, 4.0, 9.0, 30.0, 500.0, 1e5, 9e6, 2e7, 1e9]:
    for p in [1e-10, 1e-4, 0.01, 0.1, 0.4, 0.5000001, 0.9, 0.975, 0.999, 1 - 1e-9]:
        out["student_t_ppf"].append([p, df, f(t_ppf(p, df))])

for a in [0.5, 1.0, 2.5, 10.0, 50.0, 100.0, 500.0, 1e3, 1e4, 1e5, 1e6]:
    for r in [0.2, 0.8, 0.95, 1.0, 1.0 + 1.0 / a, 1.01, 1.05, 1.2, 1.5, 3.0, 10.0]:
        x = a * r
        out["gamma_q"].append([a, x, f(mp.gammainc(mp.mpf(a), mp.mpf(x), mp.inf, regularized=True))])
    for dev in [1.0, 3.0, 6.0, 10.0]:  # x = a + dev*sqrt(2a), the sqrt(a) transition region
        x = a + dev * math.sqrt(2 * a)
        out["gamma_q"].append([a, x, f(mp.gammainc(mp.mpf(a), mp.mpf(x), mp.inf, regularized=True))])

for z in [1e-5, 0.01, 0.3, 0.4999, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 5.0, 10.5, 50.0, 171.0, 1e3, 1e5, 1e10,
          -0.5, -1.5, -2.5, -10.3]:
    out["ln_gamma"].append([z, f(mp.log(abs(mp.gamma(mp.mpf(z)))))])

json.dump(out, sys.stdout, indent=0, allow_nan=False)
print()
