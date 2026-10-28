# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Reference for the simulated-unobserved-confounder refuter at 60 digits.

Stages (the Rust test regenerates the confounder draws from the library RNG streams and asserts
they equal the frozen ones):

  uv run --script simulated_unobserved_confounder_reference.py data > data.json
  (Rust: UC_REFERENCE_DATA=data.json UC_REFERENCE_OUT=draws.json cargo test -p antecedent-validate \
         --test simulated_unobserved_confounder_reference emit -- --ignored)
  uv run --script simulated_unobserved_confounder_reference.py build draws.json \
         > simulated_unobserved_confounder_reference.json

Written from the stated definition, with no product code. With Z the adjustment covariates:
  sd_t = sample SD (n-1) of the residuals of T on (1, Z); sd_y likewise for Y;
  for each draw u (standard normal, supplied): T' = T + a*sd_t*u, Y' = Y + b*sd_y*u;
  ate' = coefficient of T' in the least-squares fit of Y' on (1, T', Z) -- u is NOT added to Z;
  centre = the unperturbed ate;  mean_delta = mean |ate' - centre|;  mean_ate = mean ate';
  std_delta = mean_delta / max(|centre|, 1e-3 * sd_y/sd_t);  passed = std_delta < threshold.

Repository-authored (MIT OR Apache-2.0).
"""
import json
import sys
from fractions import Fraction as F

import mpmath as mp

mp.mp.dps = 60
mpf = mp.mpf


def lcg(seed):
    state = seed
    while True:
        state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        yield F(state >> 40, 1 << 24)


def d4(x):
    return F(round(x * 10**4), 10**4)


def d64(x):
    return F(round(x * 64), 64)


def orthogonal_null_case(name, n):
    """t = i mod 2, z = i/n, y = 1 + z + e with e = +-1/2 in the pattern + - - + per block of
    four rows: e is exactly orthogonal to 1, t and z, so the full-fit effect is exactly zero while
    the residual SDs of T and Y given Z stay positive (n a multiple of 4, a power of two)."""
    t = [float(i % 2) for i in range(n)]
    z = [i / n for i in range(n)]
    e = [0.5 if i % 4 in (0, 3) else -0.5 for i in range(n)]
    y = [1.0 + z[i] + e[i] for i in range(n)]
    return {"name": name, "t": t, "y": y, "z": [z]}


def make_case(name, n, seed, nz, kind="mixed", z_offset=0, y_offset=0):
    g = lcg(seed)
    t, y, zs = [], [], [[] for _ in range(nz)]
    for _ in range(n):
        u = [next(g) for _ in range(nz + 2)]
        dec = d64 if kind == "exact_null" else d4
        z = [dec((u[k] - F(1, 2)) * 4 + z_offset) for k in range(nz)]
        z0 = z[0] - z_offset
        ti = dec(z0 / 2 + (u[nz] - F(1, 2)) * 2)
        if kind == "exact_null":
            yi = 2 * z0 + 3 * (z[1] - z_offset)
        else:
            yi = dec(1 + F(3, 5) * ti + z0 / 2 + (u[nz + 1] - F(1, 2)) + y_offset)
        t.append(ti)
        y.append(yi)
        for k in range(nz):
            zs[k].append(z[k])
    return {"name": name, "t": [float(v) for v in t], "y": [float(v) for v in y],
            "z": [[float(v) for v in c] for c in zs]}


CASES = [
    make_case("plain", 40, 41, 2),
    make_case("offset", 34, 42, 2, z_offset=100000, y_offset=1000000),
    orthogonal_null_case("exact_null_effect", 32),
    make_case("single_covariate", 26, 44, 1),
]
# (replicates, effect_on_treatment, effect_on_outcome, threshold)
SETTINGS = [(20, 0.5, 0.5, 1.0), (7, 0.5, 0.5, 0.1), (20, 0.0, 1.5, 1.0), (20, 2.0, 2.0, 1.0),
            (20, 0.0, 0.0, 1.0), (20, 0.5, 0.5, 0.05)]


def ols(cols, y):
    """cols: list of mpf lists (including the constant); returns beta (mp), 60 digits."""
    p = len(cols)
    a = mp.matrix(p, p)
    b = mp.matrix(p, 1)
    for i in range(p):
        for j in range(p):
            a[i, j] = mp.fsum(x * z for x, z in zip(cols[i], cols[j]))
        b[i] = mp.fsum(x * z for x, z in zip(cols[i], y))
    return mp.lu_solve(a, b)


def resid_sd(zcols, v):
    n = len(v)
    cols = [[mpf(1)] * n] + zcols
    beta = ols(cols, v)
    res = [v[i] - mp.fsum(beta[k] * cols[k][i] for k in range(len(cols))) for i in range(n)]
    m = mp.fsum(res) / n
    return mp.sqrt(mp.fsum((r - m) ** 2 for r in res) / (n - 1))


def main():
    if sys.argv[1] == "data":
        print(json.dumps(CASES))
        return
    draws = json.load(open(sys.argv[2]))
    out = []
    for case in CASES:
        n = len(case["t"])
        t = [mpf(v) for v in case["t"]]
        y = [mpf(v) for v in case["y"]]
        zc = [[mpf(v) for v in col] for col in case["z"]]
        one = [mpf(1)] * n
        centre = ols([one, t] + zc, y)[1]
        sd_t, sd_y = resid_sd(zc, t), resid_sd(zc, y)
        us = draws[case["name"]]
        runs = []
        for reps, a, b, thr in SETTINGS:
            ates = []
            for r in range(reps):
                u = [mpf(v) for v in us[r]]
                t2 = [t[i] + mpf(a) * sd_t * u[i] for i in range(n)]
                y2 = [y[i] + mpf(b) * sd_y * u[i] for i in range(n)]
                ates.append(ols([one, t2] + zc, y2)[1])
            mean_ate = mp.fsum(ates) / reps
            mean_delta = mp.fsum(abs(x - centre) for x in ates) / reps
            std_delta = mean_delta / max(abs(centre), mpf("1e-3") * sd_y / sd_t)
            runs.append({"replicates": reps, "a": a, "b": b, "threshold": thr,
                         "mean_ate": float(mean_ate), "std_delta": float(std_delta),
                         "passed": bool(std_delta < thr)})
        out.append({"case": case, "centre": float(centre), "sd_t": float(sd_t),
                    "sd_y": float(sd_y), "draws": us, "runs": runs})
    print(json.dumps(out))


main()
