# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""40-digit reference for the lagged-instrument Wald contrast.

Regenerate with:  uv run --script lagged_instrument_wald_reference.py > lagged_instrument_wald_reference.json

With lag L the instrument of time t is Z[t - L]; the aligned rows are t = L .. n - 1 (m = n - L of
them).  The effect is Cov(Y, Z) / Cov(T, Z) on the aligned rows, the first stage is T = a + p Z,
the structural intercept is mean(Y) - effect mean(T), the reduced-form outcome slope is
Cov(Y, Z) / Var(Z), the homoskedastic standard error is sqrt(RSS_s / (m - 2) * Szz / Szt^2) with
RSS_s the structural residual sum of squares at the Wald slope, and the first-stage F is
p^2 Szz / (RSS_f / (m - 2)).  The weak-instrument threshold refuses a finite F below it.
Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import json

import mpmath as mp

mp.mp.dps = 40


class Lcg:
    def __init__(self, s):
        self.s = s

    def u(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) % 2**64
        return (self.s >> 11) / float(1 << 53)

    def normal(self):
        return sum(self.u() for _ in range(12)) - 6.0


def s(x):
    return mp.nstr(x, 30)


def series(seed, n, lag, strength):
    rng = Lcg(seed)
    z = [rng.normal() for _ in range(n)]
    u = [rng.normal() for _ in range(n)]
    t = [0.0] * n
    y = [0.0] * n
    for i in range(n):
        zl = z[i - lag] if i >= lag else 0.0
        t[i] = 0.5 + strength * zl + 0.8 * u[i] + 0.5 * rng.normal()
        y[i] = -1.0 + 1.7 * t[i] + 1.1 * u[i] + 0.6 * rng.normal()
    return z, t, y


def wald(z, t, y, lag):
    n = len(z)
    zz = [mp.mpf(v) for v in z[: n - lag]]
    tt = [mp.mpf(v) for v in t[lag:]]
    yy = [mp.mpf(v) for v in y[lag:]]
    m = len(zz)
    zm, tm, ym = sum(zz) / m, sum(tt) / m, sum(yy) / m
    szz = sum((a - zm) ** 2 for a in zz)
    szt = sum((a - zm) * (b - tm) for a, b in zip(zz, tt))
    szy = sum((a - zm) * (b - ym) for a, b in zip(zz, yy))
    p = szt / szz
    a0 = tm - p * zm
    beta = szy / szt
    c0 = ym - beta * tm
    rss_f = sum((b - a0 - p * a) ** 2 for a, b in zip(zz, tt))
    rss_s = sum((c - c0 - beta * b) ** 2 for b, c in zip(tt, yy))
    df = m - 2
    return {
        "effect": s(beta), "intercept": s(c0), "first_stage_slope": s(p),
        "first_stage_intercept": s(a0), "reduced_form_slope": s(szy / szz),
        "se": s(mp.sqrt(rss_s / df * szz / szt**2)), "first_stage_f": s(p * p * szz / (rss_f / df)),
        "aligned_rows": m,
    }


def main():
    cases = []
    for name, seed, n, lag, strength in (
        ("lag1_strong", 91, 70, 1, 1.4),
        ("lag3_moderate", 92, 90, 3, 0.45),
        ("lag2_weak", 93, 40, 2, 0.12),
    ):
        z, t, y = series(seed, n, lag, strength)
        case = {"name": name, "instrument": z, "treatment": t, "outcome": y, "lag": lag}
        case.update(wald(z, t, y, lag))
        cases.append(case)
    print(json.dumps(cases))


main()
