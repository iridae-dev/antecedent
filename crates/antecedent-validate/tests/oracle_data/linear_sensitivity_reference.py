# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath"]
# ///
"""Simulated-confounder linear sensitivity grid, by explicit perturbation in 50-digit arithmetic.

Regenerate with:  uv run --script linear_sensitivity_reference.py > linear_sensitivity_reference.json

The method injects a unit-variance confounder u, uncorrelated with [1, T, Y, Z], as
T' = T + a u and Y' = Y + b u, with a = sqrt(r / (1 - r)) SD(T | Z), b = -sign(ate) a SD(Y | Z) / SD(T | Z)
(SD(. | Z) is the residual standard deviation with divisor n - 1), refits Y' on [1, T', Z] and
reports the first grid value r at which the slope has changed sign (or vanished).  This script
builds one explicit such u (Gram-Schmidt against the design in 50-digit arithmetic, then scaled to
unit sample SD), forms T' and Y' for every r, and refits by solving the normal equations in
50-digit arithmetic, so it shares no algebra with the product's Gram shortcut.  It also records
the partial R^2 of u in T' on [1, Z, u] (which must equal r) and the closed-form tipping value
|rho| / (1 + |rho|) with rho the partial correlation of T and Y given Z.
Repository-authored (MIT OR Apache-2.0).
"""
import json

import mpmath as mp

mp.mp.dps = 50

_state = 0x5851F42D4C957F2D


def u01():
    global _state
    _state = (_state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
    return (_state >> 11) / float(1 << 53)


def gauss():
    return sum(u01() for _ in range(12)) - 6.0


def q(x, bits=16):
    return round(x * (1 << bits)) / (1 << bits)


def solve(a, b):
    n = len(a)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        m[c], m[p] = m[p], m[c]
        for r in range(n):
            if r != c:
                f = m[r][c] / m[c][c]
                m[r] = [x - f * y for x, y in zip(m[r], m[c])]
    return [m[i][n] / m[i][i] for i in range(n)]


def ols(cols, y):
    k, n = len(cols), len(y)
    g = [[sum(cols[i][r] * cols[j][r] for r in range(n)) for j in range(k)] for i in range(k)]
    c = [sum(cols[i][r] * y[r] for r in range(n)) for i in range(k)]
    beta = solve(g, c)
    res = [y[r] - sum(beta[i] * cols[i][r] for i in range(k)) for r in range(n)]
    return beta, res


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def analyse(t, y, zs, grid, levels=(1, 0)):
    n = len(t)
    t = [mp.mpf(v) for v in t]
    y = [mp.mpf(v) for v in y]
    zs = [[mp.mpf(v) for v in z] for z in zs]
    one = [mp.mpf(1)] * n
    base = [one] + zs
    _, et = ols(base, t)
    _, ey = ols(base, y)
    sd_t = mp.sqrt(dot(et, et) / (n - 1))  # residual vectors are mean zero (intercept in design)
    sd_y = mp.sqrt(dot(ey, ey) / (n - 1))
    beta0, _ = ols([one, t] + zs, y)
    ate0 = beta0[1]
    # explicit confounder: deterministic draw, residualised on [1, T, Y, Z], unit sample SD
    u = [mp.mpf(gauss()) for _ in range(n)]
    _, ur = ols([one, t, y] + zs, u)
    mean = sum(ur) / n
    sd_u = mp.sqrt(dot([v - mean for v in ur], [v - mean for v in ur]) / (n - 1))
    u = [v / sd_u for v in ur]
    direction = -1 if ate0 >= 0 else 1
    rho = dot(et, ey) / mp.sqrt(dot(et, et) * dot(ey, ey))
    rows = []
    for r in grid:
        rr = min(max(mp.mpf(r), 0), mp.mpf("0.999"))
        s = mp.sqrt(rr / (1 - rr))
        a, b = s * sd_t, direction * s * sd_y
        t2 = [ti + a * ui for ti, ui in zip(t, u)]
        y2 = [yi + b * ui for yi, ui in zip(y, u)]
        beta, _ = ols([one, t2] + zs, y2)
        # partial R^2 of u in T' given [1, Z, u]
        _, r1 = ols(base + [u], t2)
        _, r0 = ols(base, t2)
        pr2 = 1 - dot(r1, r1) / dot(r0, r0)
        rows.append({"r": r, "ate": float(beta[1]), "partial_r2": float(pr2)})
    tip = None
    sign0 = 1 if ate0 >= 0 else -1
    for row in rows:  # grid is ascending
        if row["ate"] == 0 or (1 if row["ate"] > 0 else -1) != sign0:
            tip = row["r"]
            break
    return {"ate": float(ate0), "rho": float(rho), "closed_form_tip": float(abs(rho) / (1 + abs(rho))),
            "grid": rows, "tip": tip}


def make(name, n, nz, tz, sign, noise, scale_t=1.0, scale_y=1.0):
    zs = [[q(gauss()) for _ in range(n)] for _ in range(nz)]
    u0 = [gauss() for _ in range(n)]
    t = [sum(tz * z[i] for z in zs) + 0.5 * u0[i] + 0.5 * gauss() for i in range(n)]
    y = [sign * 0.8 * t[i] + 0.4 * u0[i] + 0.3 * sum(z[i] for z in zs) + noise * gauss() for i in range(n)]
    t = [q(v * scale_t, 16 if scale_t >= 1 else 60) for v in t]
    y = [q(v * scale_y, 16 if scale_y >= 1 else 60) for v in y]
    grid = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
    case = {"name": name, "t": t, "y": y, "z": zs}
    case.update(analyse(t, y, zs, grid))
    return case


def main():
    cases = [
        make("no_covariate_positive", 50, 0, 0.0, 1, 3.0),
        make("strong_covariates_positive", 70, 2, 0.9, 1, 6.0),
        make("covariate_negative_effect", 60, 1, 0.8, -1, 5.0),
        make("modest_effect_tips_early", 80, 1, 0.3, 1, 9.0),
    ]
    base = cases[1]
    grid = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
    for tag, st, sy in [("outcome_in_tiny_units", 1.0, 1e-12), ("treatment_in_tiny_units", 1e-8, 1.0), ("both_in_tiny_units", 1e-6, 1e-12)]:
        t = [q(v * st, 80) if st < 1 else v * st for v in base["t"]]
        y = [q(v * sy, 80) if sy < 1 else v * sy for v in base["y"]]
        c = {"name": f"strong_covariates_{tag}", "t": t, "y": y, "z": base["z"]}
        c.update(analyse(t, y, base["z"], grid))
        cases.append(c)
    print(json.dumps({"cases": cases}))


main()
