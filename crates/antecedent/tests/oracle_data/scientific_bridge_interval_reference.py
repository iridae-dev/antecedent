# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath"]
# ///
"""Interval of a bridged value: Student-t prediction interval, affine transform, delta method.

Regenerate with:  uv run --script scientific_bridge_interval_reference.py > scientific_bridge_interval_reference.txt

Exact rational sums (fractions.Fraction on the binary64 inputs) give the least-squares line of the
validation trials and its residual variance s^2 = (Syy - Sxy^2/Sxx)/(k-2); the quantiles are
evaluated at 50 digits (normal: sqrt(2) erfinv(2p - 1); Student t: root of the t distribution
function written with the Gauss hypergeometric function).
  trial-level   value = a + b x0,  sd^2 = s^2 (1 + 1/k + (x0 - xbar)^2/Sxx) + b^2 se^2,
                interval value -+ t_{k-2}(1 - (1-level)/2) sd
  affine        value = a + b x0, interval value -+ z |b| se
  correction    inversion p = (q + Sp - 1)/J with Var(p) = (se^2 + p^2 Var(Se) + (1-p)^2 Var(Sp))/J^2;
                forward q = Se p + (1-Sp)(1-p) with Var(q) = J^2 se^2 + p^2 Var(Se) + (1-p)^2 Var(Sp);
                the interval is value -+ z sd clipped to [0, 1]
Values are written as nearest binary64.  Repository-authored (MIT OR Apache-2.0).
"""
from fractions import Fraction as F
from mpmath import mp, mpf, erfinv, sqrt, hyp2f1, gamma, pi, findroot

mp.dps = 50


def zq(level):
    p = 1 - (1 - mpf(level)) / 2
    return sqrt(2) * erfinv(2 * p - 1)


def tq(level, df):
    p = 1 - (1 - mpf(level)) / 2
    nu = mpf(df)

    def cdf(t):
        return mpf(1) / 2 + t * gamma((nu + 1) / 2) / (sqrt(nu * pi) * gamma(nu / 2)) * hyp2f1(
            mpf(1) / 2, (nu + 1) / 2, mpf(3) / 2, -t * t / nu)

    lo, hi = mpf(0), mpf(1)
    while cdf(hi) < p:
        hi *= 2
    for _ in range(220):
        mid = (lo + hi) / 2
        if cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def tofrac(x):
    return F(x)


def trial(name, trials, x0, se, level):
    k = len(trials)
    xs = [F(t[0]) for t in trials]
    ys = [F(t[1]) for t in trials]
    xb, yb = sum(xs) / k, sum(ys) / k
    sxx = sum((x - xb) ** 2 for x in xs)
    sxy = sum((x - xb) * (y - yb) for x, y in zip(xs, ys))
    syy = sum((y - yb) ** 2 for y in ys)
    b = sxy / sxx
    a = yb - b * xb
    s2 = (syy - sxy * sxy / sxx) / (k - 2)
    x0f, sef = F(x0), F(se)
    value = a + b * x0f
    var = s2 * (1 + F(1, k) + (x0f - xb) ** 2 / sxx) + b * b * sef * sef
    sd = sqrt(mpf(var.numerator) / mpf(var.denominator))
    t = tq(level, k - 2)
    v = mpf(value.numerator) / mpf(value.denominator)
    t_str = ",".join(f"{x!r}:{y!r}" for x, y in trials)
    print("trial", name, repr(x0), repr(se), repr(level), repr(float(v)), repr(float(sd)),
          repr(float(v - t * sd)), repr(float(v + t * sd)), k - 2, t_str)


def affine(name, a, b, x0, se, level):
    z = zq(level)
    v = mpf(F(a) + F(b) * F(x0)) if False else None
    val = F(a) + F(b) * F(x0)
    v = mpf(val.numerator) / mpf(val.denominator)
    half = z * abs(mpf(F(b).numerator) / mpf(F(b).denominator)) * mpf(se)
    lo, hi = sorted((v - half, v + half))
    print("affine", name, repr(a), repr(b), repr(x0), repr(se), repr(level), repr(float(v)),
          repr(float(abs(mpf(b)) * mpf(se))), repr(float(lo)), repr(float(hi)))


def correction(name, sens, spec, vse, vsp, q, se, level, direction):
    S, P, a, d = F(sens), F(spec), F(vse), F(vsp)
    qf, V = F(q), F(se) ** 2
    J = S + P - 1
    p = (qf + P - 1) / J
    if direction == "back":
        value = p
        var = (V + p * p * a + (1 - p) ** 2 * d) / (J * J)
    else:
        value = S * qf + (1 - P) * (1 - qf)
        var = J * J * V + qf * qf * a + (1 - qf) ** 2 * d
    v = mpf(value.numerator) / mpf(value.denominator)
    sd = sqrt(mpf(var.numerator) / mpf(var.denominator))
    z = zq(level)
    lo, hi = max(v - z * sd, mpf(0)), min(v + z * sd, mpf(1))
    print("correction", name, repr(sens), repr(spec), repr(vse), repr(vsp), repr(q), repr(se),
          repr(level), direction, repr(float(v)), repr(float(sd)), repr(float(lo)), repr(float(hi)))


hand = [(0.0, 0.1), (0.5, 0.5), (1.0, 0.9), (1.5, 1.5)]
noisy20 = [(0.5 + 1.5 * i / 19, 0.1 + 0.9 * (0.5 + 1.5 * i / 19) + 0.1 * __import__("math").sin(i))
           for i in range(20)]
offset = [(1.0e6 + x, 0.3 + 0.8 * x + e) for x, e in
          [(0.0, 0.05), (0.4, -0.04), (1.1, 0.03), (1.5, -0.06), (2.2, 0.02)]]
three = [(0.0, 0.2), (1.0, 0.9), (2.0, 2.3)]
tied = [(0.0, 0.1), (0.0, 0.3), (1.0, 0.8), (1.0, 1.0), (2.0, 1.9)]
for level in (0.5, 0.9, 0.95, 0.99, 0.9999):
    trial(f"hand_{level}", hand, 0.8, 0.1, level)
trial("hand_zero_source_se", hand, 0.8, 0.0, 0.95)
trial("hand_at_min", hand, 0.0, 0.1, 0.95)
trial("hand_at_max", hand, 1.5, 0.1, 0.95)
trial("noisy_twenty", noisy20, 1.2, 0.1, 0.95)
trial("offset_1e6", offset, 1.0e6 + 0.7, 0.05, 0.95)
for level in (0.5, 0.95, 0.9999):
    trial(f"three_trials_df1_{level}", three, 1.3, 0.15, level)
trial("tied_x", tied, 0.5, 0.1, 0.95)
for level in (0.5, 0.95, 0.9999):
    affine(f"positive_{level}", 0.1, 0.6, 0.5, 0.05, level)
affine("negative_slope", 2.0, -0.75, 1.0e4, 0.02, 0.95)
correction("rogan_gladen_back", 0.9, 0.8, 0.0004, 0.0004, 0.31, 0.02, 0.95, "back")
correction("rogan_gladen_forward", 0.9, 0.8, 0.0004, 0.0004, 0.11 / 0.7, 0.02, 0.95, "forward")
correction("clipped_at_zero_back", 0.9, 0.8, 0.0009, 0.0009, 0.2005, 0.05, 0.95, "back")
correction("clipped_at_one_forward", 0.97, 0.9, 0.0004, 0.0004, 0.99, 0.05, 0.95, "forward")
correction("youden_just_above_min", 0.66, 0.65, 0.0004, 0.0004, 0.4, 0.02, 0.99, "back")
