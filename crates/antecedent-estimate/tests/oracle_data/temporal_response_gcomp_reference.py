# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Horizon-specific linear g-computation of a temporal response, solved exactly.

Regenerate with:  uv run --script temporal_response_gcomp_reference.py > temporal_response_gcomp_reference.json

A case is a series (x treatment, y outcome, z confounder) and, per horizon h, a design written in
the definition's terms: the outcome at time t is regressed on an intercept, the treatment at time
t - h and the confounder at times t - l for each lag l of that horizon's adjustment set, over the
windows t >= max(h, largest l) (each horizon fits its own windows).  The g-computed level of the
mean curve at dose a is  mu_h(a) = beta_0 + beta_x a + sum_l beta_l mean(z_{t-l}),  the means
taken over that horizon's own design rows; the additive shift level is the same with a replaced by
mean(x_{t-h}) + shift.  All arithmetic is exact (fractions.Fraction, the binary64 inputs taken as
the exact rationals they are); only the final rounding to binary64 is inexact.
Repository-authored (MIT OR Apache-2.0); standard library only.
"""
import json
from fractions import Fraction as F

_state = 0x13198A2E03707344


def u01():
    global _state
    _state = (_state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
    return (_state >> 11) / float(1 << 53)


def normal():
    return sum(u01() for _ in range(12)) - 6.0


def solve(A, b):
    n = len(A)
    M = [row[:] + [bb] for row, bb in zip(A, b)]
    for c in range(n):
        p = next(r for r in range(c, n) if M[r][c] != 0)
        M[c], M[p] = M[p], M[c]
        piv = M[c][c]
        M[c] = [v / piv for v in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [a - f * bb for a, bb in zip(M[r], M[c])]
    return [row[n] for row in M]


def series(n, phi, z_noise, x_off, y_off, y_noise):
    z, x, y = [0.0] * n, [0.0] * n, [0.0] * n
    for t in range(n):
        zp = z[t - 1] if t >= 1 else 0.0
        z[t] = round(phi * zp + z_noise * normal(), 9)
        x[t] = round(0.6 * z[t] + 0.4 * zp + 0.8 * normal(), 9)
        x1 = x[t - 1] if t >= 1 else 0.0
        x2 = x[t - 2] if t >= 2 else 0.0
        z2 = z[t - 2] if t >= 2 else 0.0
        y[t] = round(1.5 * x1 + 0.7 * x2 + 0.5 * zp + 0.8 * z2 + y_noise * normal(), 9)
    return [v + x_off for v in x], [v + y_off for v in y], z


def fit_horizon(x, y, z, h, lags):
    n = len(y)
    max_lag = max([h] + lags)
    rows, ys = [], []
    for t in range(max_lag, n):
        rows.append([F(1), F(x[t - h])] + [F(z[t - l]) for l in lags])
        ys.append(F(y[t]))
    p = len(rows[0])
    A = [[sum(r[a] * r[b] for r in rows) for b in range(p)] for a in range(p)]
    rhs = [sum(r[a] * v for r, v in zip(rows, ys)) for a in range(p)]
    beta = solve(A, rhs)
    cbar = [sum(r[a] for r in rows) / len(rows) for a in range(p)]
    return beta, cbar, len(rows)


def level(beta, cbar, treatment):
    return sum(b * (treatment if i == 1 else c) for i, (b, c) in enumerate(zip(beta, cbar)))


def make_case(name, n, phi, z_noise, x_off, y_off, y_noise, doses, shift, lags_by_h):
    x, y, z = series(n, phi, z_noise, x_off, y_off, y_noise)
    horizons = sorted(lags_by_h)
    fits = {h: fit_horizon(x, y, z, h, lags_by_h[h]) for h in horizons}
    surface = []  # dose major, then horizon
    scale = []
    for a in doses:
        for h in horizons:
            beta, cbar, _ = fits[h]
            surface.append(float(level(beta, cbar, F(a))))
            scale.append(float(sum(abs(b * (F(a) if i == 1 else c)) for i, (b, c) in enumerate(zip(beta, cbar)))))
    shifted = [float(level(fits[h][0], fits[h][1], fits[h][1][1] + F(shift))) for h in horizons]
    return {
        "name": name,
        "x": x,
        "y": y,
        "z": z,
        "horizons": horizons,
        "adjustment_lags": {str(h): lags_by_h[h] for h in horizons},
        "doses": doses,
        "shift": shift,
        "rows": {str(h): fits[h][2] for h in horizons},
        "surface": surface,
        "surface_scale": scale,
        "shifted": shifted,
        "coefficients": {str(h): [float(b) for b in fits[h][0]] for h in horizons},
    }


LAGS = {1: [1, 2], 2: [2]}
cases = [
    make_case("ordinary", 70, 0.5, 1.0, 0.0, 0.0, 0.5, [-1.0, 0.0, 0.5, 2.5], 0.75, LAGS),
    make_case("outcome_offset_1e6_treatment_offset_1e3", 70, 0.5, 1.0, 1.0e3, 1.0e6, 0.5, [0.0, 999.0, 1000.5, 1002.0], -0.4, LAGS),
    make_case("near_collinear_confounder_lags", 70, 0.9995, 0.05, 0.0, 0.0, 0.3, [-1.0, 0.0, 1.0], 1.0, LAGS),
    make_case("tiny_sample", 9, 0.4, 1.0, 0.0, 0.0, 0.5, [0.0, 1.0], 0.5, LAGS),
    make_case("long_series", 400, 0.7, 1.0, 0.0, 0.0, 1.0, [-2.0, 0.0, 3.0], 0.3, LAGS),
]
print(json.dumps(cases, allow_nan=False))
