# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Exact reference for the placebo-treatment, dummy-outcome, random-common-cause and
data-subset refuters.

Stages (the Rust test regenerates every perturbation from the library RNG streams and asserts
it equals the frozen one):

  uv run --script robustness_refuters_reference.py data > data.json
  (Rust: ROBUSTNESS_REFERENCE_DATA=data.json ROBUSTNESS_REFERENCE_OUT=perturbations.json
         cargo test -p antecedent-validate --test robustness_refuters_reference emit -- --ignored)
  uv run --script robustness_refuters_reference.py build perturbations.json > robustness_refuters_reference.json

Written from the definitions, with no product code:
  ate        = coefficient of T in the exact least-squares fit of Y on (1, T, Z...) (Fractions);
  placebo    : T replaced by the supplied noise (or by the supplied permutation of T), null 0;
  dummy      : Y replaced by the supplied noise, null 0;
  rcc        : the supplied noise appended to Z, null = the unperturbed fit;
  subset     : rows kept by the supplied mask, null = the unperturbed fit;
  mean = average of the replicate ates, sd = sample SD (n-1), z = (null - mean)/sd,
  p = erfc(|z|/sqrt(2)) at 60 digits, passed = p >= alpha; a spread of exactly zero gives
  p = 1 when the null equals the mean and 0 otherwise.

Repository-authored (MIT OR Apache-2.0).
"""
import json
import sys
from fractions import Fraction as F

import mpmath as mp

mp.mp.dps = 60


def lcg(seed):
    state = seed
    while True:
        state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        yield F(state >> 40, 1 << 24)


def dec_decimal(x, digits=4):
    return F(round(x * 10**digits), 10**digits)


def dec_dyadic(x):
    return F(round(x * 64), 64)


def make_case(name, n, seed, nz, treat_cut=F(1, 2), z_offset=0, y_offset=0, noise=F(2), dyadic=False):
    g = lcg(seed)
    dec = dec_dyadic if dyadic else dec_decimal
    slope = F(3, 4) if dyadic else F(4, 5)
    t, y, zs = [], [], [[] for _ in range(nz)]
    for _ in range(n):
        u = [next(g) for _ in range(nz + 2)]
        zi = [dec((u[k] - F(1, 2)) * 4 + z_offset) for k in range(nz)]
        ti = F(1) if u[nz] + F(1, 10) * (zi[0] - z_offset) > treat_cut else F(0)
        yi = dec(1 + slope * ti + F(1, 2) * (zi[0] - z_offset)
                 + (u[nz + 1] - F(1, 2)) * noise + y_offset)
        t.append(ti)
        y.append(yi)
        for k in range(nz):
            zs[k].append(zi[k])
    return {"name": name, "t": [float(v) for v in t], "y": [float(v) for v in y],
            "z": [[float(v) for v in col] for col in zs]}


CASES = [
    make_case("plain", 30, 21, 2),
    make_case("offset", 28, 22, 2, z_offset=100000, y_offset=1000000),
    make_case("noiseless", 24, 23, 1, noise=F(0), dyadic=True),
    make_case("unbalanced", 40, 24, 1, treat_cut=F(3, 4)),
]
SETTINGS = [(20, 0.05), (7, 0.5)]


def solve(a, b):
    n = len(a)
    m = [row[:] + [bb] for row, bb in zip(a, b)]
    for c in range(n):
        piv = next((r for r in range(c, n) if m[r][c] != 0), None)
        if piv is None:
            return None
        m[c], m[piv] = m[piv], m[c]
        inv = 1 / m[c][c]
        m[c] = [v * inv for v in m[c]]
        for r in range(n):
            if r != c and m[r][c] != 0:
                f = m[r][c]
                m[r] = [v - f * w for v, w in zip(m[r], m[c])]
    return [m[r][n] for r in range(n)]


def ate(t, y, zs, rows=None):
    rows = list(range(len(t))) if rows is None else rows
    cols = [[F(1)] * len(rows), [F(t[i]) for i in rows]] + [[F(z[i]) for i in rows] for z in zs]
    yy = [F(y[i]) for i in rows]
    p = len(cols)
    xtx = [[sum(a * b for a, b in zip(cols[i], cols[j])) for j in range(p)] for i in range(p)]
    xty = [sum(a * b for a, b in zip(cols[i], yy)) for i in range(p)]
    beta = solve(xtx, xty)
    return None if beta is None else beta[1]


def verdict(ates, null, alpha):
    m = len(ates)
    mean = sum(ates) / m
    var = sum((a - mean) ** 2 for a in ates) / (m - 1)
    if var == 0:
        p = mp.mpf(1) if null == mean else mp.mpf(0)
    else:
        sd = mp.sqrt(mp.mpf(var.numerator) / var.denominator)
        z = (mp.mpf(null.numerator) / null.denominator - mp.mpf(mean.numerator) / mean.denominator) / sd
        p = mp.erfc(abs(z) / mp.sqrt(2))
    return float(mean), float(p), bool(p >= alpha)


def main():
    if sys.argv[1] == "data":
        print(json.dumps(CASES))
        return
    pert = json.load(open(sys.argv[2]))
    out = []
    for case in CASES:
        t, y, zs = case["t"], case["y"], case["z"]
        n = len(t)
        p = pert[case["name"]]
        full = ate(t, y, zs)
        runs = []
        for reps, alpha in SETTINGS:
            res = {"replicates": reps, "alpha": alpha}
            # placebo (gaussian): null 0
            a = [ate(p["placebo_noise"][r], y, zs) for r in range(reps)]
            res["placebo"] = verdict(a, F(0), alpha)
            a = [ate(p["placebo_perm"][r], y, zs) for r in range(reps)]
            res["placebo_permute"] = verdict(a, F(0), alpha)
            a = [ate(t, p["dummy_noise"][r], zs) for r in range(reps)]
            res["dummy"] = verdict(a, F(0), alpha)
            a = [ate(t, y, zs + [p["rcc_noise"][r]]) for r in range(reps)]
            res["rcc"] = verdict(a, full, alpha)
            for frac in ("0.8", "0.55"):
                a = []
                for r in range(reps):
                    rows = [i for i in range(n) if p["subset_mask"][frac][r][i]]
                    v = ate(t, y, zs, rows)
                    a.append(v)
                res["subset_" + frac] = None if any(v is None for v in a) else verdict(a, full, alpha)
            runs.append(res)
        out.append({"case": case, "full": float(full), "perturbations": p, "runs": runs})
    print(json.dumps(out))


main()
