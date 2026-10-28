# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Subset-of-regressors (Nystrom) RBF-GP residuals, evaluated in 50-digit arithmetic.

Regenerate with:  uv run --script gpdc_nystrom_reference.py > gpdc_nystrom_reference.json

For a conditioning set Z of d columns, n rows and m inducing rows R (a declared input of each case:
the product draws them from a seeded stream, the reference takes the draw as given), the predictor
residualizes a series y as follows, all in mpmath at 50 digits from the exact binary64 inputs:
  z_std[:, j] = (z[:, j] - mean) / sd  with the population sd (n divisor); a constant column is 0;
  K[a, b]     = exp(-0.5 |z_std[a] - z_std[b]|^2 / ell^2);
  N           = K_nR' K_nR + lambda K_RR;  N <- N + eps mean(diag N) I   (eps = 1e-10);
  c           = N^{-1} K_nR' (y - mean y);   residual = (y - mean y) - K_nR c.
With no conditioning column the residual is the centred series.  The distance correlation and the
permutation null applied to residuals are reviewed with ci.gpdc.
Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import json

import mpmath as mp

mp.mp.dps = 50

_state = 0x452821E638D01377


def u01():
    global _state
    _state = (_state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
    return (_state >> 11) / float(1 << 53)


def normal():
    return sum(u01() for _ in range(12)) - 6.0


def residuals(cols, zidx, ys, rows, ell, lam):
    n = len(ys[0])
    d = len(zidx)
    centred = []
    for y in ys:
        mu = sum(mp.mpf(v) for v in y) / n
        centred.append([mp.mpf(v) - mu for v in y])
    if d == 0:
        return centred
    zs = [[mp.mpf(0)] * d for _ in range(n)]
    for j, zc in enumerate(zidx):
        col = [mp.mpf(v) for v in cols[zc]]
        mu = sum(col) / n
        var = sum((v - mu) ** 2 for v in col) / n
        sd = mp.sqrt(var)
        if sd > 0:
            for r in range(n):
                zs[r][j] = (col[r] - mu) / sd
    ell2 = mp.mpf(ell) ** 2

    def k(a, b):
        d2 = sum((zs[a][j] - zs[b][j]) ** 2 for j in range(d))
        return mp.exp(-d2 / (2 * ell2))

    m = len(rows)
    knm = mp.matrix(n, m)
    for i in range(n):
        for j, r in enumerate(rows):
            knm[i, j] = k(i, r)
    normal_m = knm.T * knm
    for a in range(m):
        for b in range(m):
            normal_m[a, b] += mp.mpf(lam) * k(rows[a], rows[b])
    mean_diag = sum(normal_m[a, a] for a in range(m)) / m
    for a in range(m):
        normal_m[a, a] += mp.mpf("1e-10") * mean_diag
    out = []
    for yc in centred:
        rhs = knm.T * mp.matrix(yc)
        c = mp.lu_solve(normal_m, rhs)
        fitted = knm * c
        out.append([yc[i] - fitted[i] for i in range(n)])
    return out


def sample(n, kind):
    z1 = [round(normal(), 9) for _ in range(n)]
    z2 = [round(normal(), 9) for _ in range(n)]
    e1 = [normal() for _ in range(n)]
    e2 = [normal() for _ in range(n)]
    if kind == "ties":
        z1 = [float(round(v)) for v in z1]
        z2 = [float(round(v * 2) / 2) for v in z2]
    x = [round(1.5 * mp.sin(1.5 * z1[i]) + 0.8 * z2[i] + e1[i], 9) for i in range(n)]
    y = [round(1.5 * mp.tanh(1.5 * z1[i]) - 0.8 * z2[i] + e2[i], 9) for i in range(n)]
    x = [float(v) for v in x]
    y = [float(v) for v in y]
    return [x, y, z1, z2]


# (name, n, kind, z columns, inducing m, length scale, ridge, z affine transform (scale, shift))
SPECS = [
    ("rank_12_of_48", 48, "plain", [2, 3], 12, 1.0, 1e-3, (1.0, 0.0)),
    ("all_rows_inducing", 20, "plain", [2, 3], 64, 1.0, 1e-3, (1.0, 0.0)),
    ("one_conditioning_column", 40, "plain", [2], 10, 0.7, 1e-2, (1.0, 0.0)),
    ("z_far_from_unit_scale", 40, "plain", [2, 3], 10, 1.0, 1e-3, (1.0e4, 1.0e6)),
    ("tied_z_values_rank_deficient_kernel", 40, "ties", [2, 3], 12, 1.0, 1e-3, (1.0, 0.0)),
    ("tiny_sample_rank_3", 6, "plain", [2, 3], 3, 1.0, 1e-3, (1.0, 0.0)),
    ("wide_length_scale_smooth_kernel", 40, "plain", [2, 3], 8, 5.0, 1e-3, (1.0, 0.0)),
    ("empty_conditioning_set", 30, "plain", [], 8, 1.0, 1e-3, (1.0, 0.0)),
]
# The product's seeded draw of the inducing rows (ascending), frozen from the implementation;
# the test asserts the product still draws exactly these rows.
INDUCING = {
    "rank_12_of_48": [5, 13, 19, 21, 22, 24, 25, 26, 29, 34, 42, 45],
    "all_rows_inducing": list(range(20)),
    "one_conditioning_column": [4, 11, 15, 18, 22, 25, 28, 30, 33, 38],
    "z_far_from_unit_scale": [0, 5, 7, 10, 16, 18, 29, 33, 38, 39],
    "tied_z_values_rank_deficient_kernel": [0, 2, 4, 5, 7, 10, 16, 18, 29, 33, 38, 39],
    "tiny_sample_rank_3": [0, 2, 3],
    "wide_length_scale_smooth_kernel": [0, 7, 16, 18, 29, 33, 38, 39],
    "empty_conditioning_set": [],
}

cases = []
for name, n, kind, zidx, m, ell, lam, (sc, sh) in SPECS:
    cols = sample(n, kind)
    for j in (2, 3):
        cols[j] = [v * sc + sh for v in cols[j]]
    rows = INDUCING.get(name, list(range(n)) if m >= n else None)
    if rows is None:
        cases.append({"name": name, "columns": cols, "z": zidx, "m": m, "ell": ell, "ridge": lam})
        continue
    rx, ry = residuals(cols, zidx, [cols[0], cols[1]], rows, ell, lam)
    cases.append(
        {
            "name": name,
            "columns": cols,
            "z": zidx,
            "m": m,
            "ell": ell,
            "ridge": lam,
            "inducing": rows,
            "rx": [float(v) for v in rx],
            "ry": [float(v) for v in ry],
        }
    )
print(json.dumps(cases, allow_nan=False))
