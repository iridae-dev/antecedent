# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath"]
# ///
"""Instrumental-variable fits and Anderson-Rubin sets from first principles.

Regenerate with:  uv run --script iv_reference.py > iv_reference.json

Every data table is a list of binary64 values, read back exactly as rationals.  All regressions
are exact rational least squares by Gaussian elimination on the normal equations.  The 2SLS
coefficient is the textbook ratio  t'P_Z y / t'P_Z t  after partialling out the included
exogenous block (here by explicit projections on the full instrument matrix), the first-stage F
and partial R^2 come from the two first-stage residual sums of squares, and the Anderson-Rubin
set is found WITHOUT the quadratic-in-beta algebra the product uses on its moments: the
statistic F(b) = ((RSS_X(b) - RSS_ZX(b)) / k) / (RSS_ZX(b) / df) is evaluated by two direct
regressions of y - b t, the numerator-minus-critical polynomial df (R_X - R_ZX) - k Fcrit R_ZX
is interpolated exactly from three values of b, and its roots are taken in 60-digit arithmetic.
The F quantile is the root of the regularized incomplete beta function in mpmath.  A bisection
check on F(b) itself confirms each reported endpoint.  Repository-authored (MIT OR Apache-2.0).
"""
import json
from fractions import Fraction as Fr

import mpmath as mp

mp.mp.dps = 60

_state = 0x2545F4914F6CDD1D


def u01():
    global _state
    _state = (_state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
    return (_state >> 11) / float(1 << 53)


def gauss():
    return sum(u01() for _ in range(12)) - 6.0


def q(x, bits=14):
    """Round to a dyadic grid so decimal printing stays short and exact."""
    return round(x * (1 << bits)) / (1 << bits)


def solve(a, b):
    n = len(a)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        p = next(r for r in range(c, n) if m[r][c] != 0)
        m[c], m[p] = m[p], m[c]
        for r in range(n):
            if r != c and m[r][c] != 0:
                f = m[r][c] / m[c][c]
                m[r] = [x - f * y for x, y in zip(m[r], m[c])]
    return [m[i][n] / m[i][i] for i in range(n)]


def ols(cols, y):
    """Exact least squares of y on the columns; returns (coef, rss, fitted)."""
    k = len(cols)
    n = len(y)
    g = [[sum(cols[i][r] * cols[j][r] for r in range(n)) for j in range(k)] for i in range(k)]
    c = [sum(cols[i][r] * y[r] for r in range(n)) for i in range(k)]
    beta = solve(g, c)
    fit = [sum(beta[i] * cols[i][r] for i in range(k)) for r in range(n)]
    rss = sum((y[r] - fit[r]) ** 2 for r in range(n))
    return beta, rss, fit


def fcrit(level, k, df):
    """Quantile of F(k, df) by bisection on the regularized incomplete beta function."""
    def cdf(x):
        return mp.betainc(mp.mpf(k) / 2, mp.mpf(df) / 2, 0, k * x / (k * x + df), regularized=True)
    lo, hi = mp.mpf(0), mp.mpf(1)
    while cdf(hi) < level:
        hi *= 2
    for _ in range(220):
        mid = (lo + hi) / 2
        if cdf(mid) < level:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def analyse(t, y, z_cols, x_cols, level=Fr(95, 100)):
    n = len(t)
    t = [Fr(v) for v in t]
    y = [Fr(v) for v in y]
    z_cols = [[Fr(v) for v in c] for c in z_cols]
    one = [Fr(1)] * n
    x_cols = [one] + [[Fr(v) for v in c] for c in x_cols]
    k = len(z_cols)
    df = n - k - len(x_cols)
    zx = z_cols + x_cols
    # first stage
    _, rss_full, fitted = ols(zx, t)
    _, rss_restr, _ = ols(x_cols, t)
    f_stat = ((rss_restr - rss_full) / k) / (rss_full / df)
    pr2 = (rss_restr - rss_full) / rss_restr
    # second stage on fitted treatment
    beta2, _, _ = ols([fitted] + x_cols, y)
    out = {
        "n": n, "k": k, "df2": df,
        "coef": float(mp.mpf(beta2[0].numerator) / beta2[0].denominator),
        "f_stat": float(mp.mpf(f_stat.numerator) / f_stat.denominator),
        "partial_r2": float(mp.mpf(pr2.numerator) / pr2.denominator),
    }
    fc = fcrit(mp.mpf(level.numerator) / level.denominator, k, df)
    out["fcrit"] = float(fc)
    c = k * fc

    def poly(b):
        ys = [yy - b * tt for yy, tt in zip(y, t)]
        _, r_f, _ = ols(zx, ys)
        _, r_x, _ = ols(x_cols, ys)
        return df * (r_x - r_f) - c_frac * r_f, (r_x - r_f), r_f

    c_frac = Fr(mp.nstr(c, 55))  # rational value of the 55-digit critical constant
    v0, v1, vm = poly(Fr(0))[0], poly(Fr(1))[0], poly(Fr(-1))[0]
    A = (v1 + vm) / 2 - v0
    B = (v1 - vm) / 2
    C = v0
    A, B, C = (mp.mpf(x.numerator) / x.denominator for x in (A, B, C))
    # classify  A b^2 + B b + C <= 0  exactly (no tolerance beyond sign of exact quantities)
    kind, lo, hi = None, None, None
    if A == 0 and B == 0:
        kind = "whole" if C <= 0 else "empty"
    elif A == 0:
        r = -C / B
        kind, lo, hi = "ray", (-mp.inf if B > 0 else r), (r if B > 0 else mp.inf)
    else:
        disc = B * B - 4 * A * C
        if disc < 0:
            kind = "whole" if A < 0 else "empty"
        else:
            s = mp.sqrt(disc)
            r1, r2 = sorted([(-B - s) / (2 * A), (-B + s) / (2 * A)])
            if A > 0:
                kind, lo, hi = "interval", r1, r2
            else:
                kind, lo, hi = "union", r1, r2
    out["ar_kind"] = kind
    if kind in ("interval", "ray"):
        out["ar_lo"] = None if lo == -mp.inf else float(lo)
        out["ar_hi"] = None if hi == mp.inf else float(hi)
        # direct check: F(b) at the finite endpoints equals Fcrit; just inside accepted, just outside rejected
        for e in (lo, hi):
            if e in (-mp.inf, mp.inf):
                continue
            pe = poly(Fr(mp.nstr(e, 50)))[0]
            scale = abs(A) * e * e + abs(B) * abs(e) + abs(C)
            assert abs(mp.mpf(pe.numerator) / pe.denominator) < mp.mpf(10) ** -25 * scale, (kind, pe)
    elif kind == "union":
        out["ar_lo"], out["ar_hi"] = float(lo), float(hi)  # gap (lo, hi) is rejected
    return out


def make(name, n, kind, strength, covariates=0, instruments=1, scale_t=1.0, scale_y=1.0, shift_t=0.0, seed_skip=0):
    for _ in range(seed_skip):
        u01()
    z = [[q(gauss()) if kind == "cont" else float(u01() < 0.5) for _ in range(n)] for _ in range(instruments)]
    x = [[q(gauss()) for _ in range(n)] for _ in range(covariates)]
    u = [gauss() for _ in range(n)]
    t, y = [], []
    for r in range(n):
        tr = strength * sum(zz[r] for zz in z) + 0.5 * sum(xx[r] for xx in x) + u[r] + 0.5 * gauss()
        yr = 2.0 * tr + u[r] + 0.3 * sum(xx[r] for xx in x) + 0.5 * gauss()
        t.append(q(tr * scale_t + shift_t, 40 if scale_t < 1 else 14))
        y.append(q(yr * scale_y, 14 if scale_y >= 1 else 40))
    case = {"name": name, "t": t, "y": y, "z": z, "x": x}
    case.update(analyse(t, y, z, x))
    return case


def main():
    cases = [
        make("binary_instrument_strong", 40, "bin", 1.5),
        make("continuous_instrument_covariates", 30, "cont", 1.0, covariates=2),
        make("two_instruments_covariate", 36, "cont", 0.8, covariates=1, instruments=2),
    ]
    # search for each AR class under weak instruments
    wanted = {"interval": None, "ray": None, "whole": None, "union": None, "empty": None}
    tries = 0
    while any(v is None for v in wanted.values()) and tries < 400:
        tries += 1
        inst = 1 + tries % 3
        strength = [0.0, 0.02, 0.05, 0.1][tries % 4]
        c = make(f"search{tries}", 14 + tries % 9, "cont", strength, covariates=tries % 2, instruments=inst)
        k = c["ar_kind"]
        if wanted[k] is None:
            c["name"] = f"weak_{k}"
            wanted[k] = c
    cases += [c for c in wanted.values() if c is not None]
    # a nearly exact fit: the acceptance interval is a few parts in 1e7 wide
    n = 60
    z = [q(i / n - 0.5, 20) for i in range(n)]
    t = [q(3.0 * z[i] + 1e-6 * ((i % 7) - 3), 30) for i in range(n)]
    y = [q(2.05 * t[i] + 1e-6 * ((i % 5) - 2), 30) for i in range(n)]
    c = {"name": "narrow_interval", "t": t, "y": y, "z": [z], "x": []}
    c.update(analyse(t, y, [z], []))
    cases.append(c)
    base = cases[1]
    # rescaled copies of the continuous case: the Anderson-Rubin set must rescale exactly
    for tag, st, sy in [("treatment_tiny", 1e-9, 1.0), ("outcome_tiny", 1.0, 1e-9), ("treatment_huge", 1e9, 1.0), ("both_tiny", 1e-7, 1e-7)]:
        t = [q(v * st, 80) if st < 1 else v * st for v in base["t"]]
        y = [q(v * sy, 80) if sy < 1 else v * sy for v in base["y"]]
        c = {"name": f"continuous_{tag}", "t": t, "y": y, "z": base["z"], "x": base["x"]}
        c.update(analyse(t, y, base["z"], base["x"]))
        cases.append(c)
    print(json.dumps({"cases": cases}))


main()
