# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Exact reference for the static-mediation refuter suite (placebo mediator, random common
cause, random subsets, binary-mediator range overlap) on the one-mediator linear model with an
exogenous covariate:   T -> M, T -> Y, M -> Y, Z -> M, Z -> Y.

Stages (the Rust test regenerates the perturbations from the library RNG streams and asserts
they equal the frozen ones):

  uv run --script static_mediation_reference.py data > data.json
  (Rust: STATIC_MEDIATION_REFERENCE_DATA=data.json STATIC_MEDIATION_REFERENCE_OUT=perturbations.json
         cargo test -p antecedent-validate --test static_mediation_reference emit -- --ignored)
  uv run --script static_mediation_reference.py build perturbations.json > static_mediation_reference.json

Written from the product-of-coefficients definition, with no product code. In Fractions:
  a   = coefficient of T in the fit of M on (1, T, Z, extra...)
  c   = coefficient of T in the fit of Y on (1, T, M, Z, extra...);  b = coefficient of M there
  direct = c * delta, indirect = a * b * delta, total = direct + indirect   (delta = T contrast = 1)
The published estimate handed to the suite is the exact estimate plus a shift (0 or 1/2): the
random-common-cause and subset nulls are that published number, so a shifted estimate must fail.
Replicate r of the suite (20 of them):
  placebo : M replaced by the supplied noise, contrast Mediated, null 0;
  rcc     : the supplied noise appended to every regression, the query's own contrast, null = the
            unperturbed estimate of that contrast;
  subset  : the rows kept by the supplied Bernoulli(0.8) mask, the query's contrast, null as rcc.
  mean, sample SD (n-1), p = erfc(|null - mean| / (sd sqrt 2)) at 60 digits, passed = p >= 0.05.
Overlap: with binary T, the per-arm closed ranges of M intersect (touching ranges overlap).

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


def d4(x):
    return F(round(x * 10**4), 10**4)


def make_case(name, n, seed, m_offset=0, y_offset=0, treat_cut=F(1, 2), m_arms=None):
    g = lcg(seed)
    t, m, y, z = [], [], [], []
    for i in range(n):
        u = [next(g) for _ in range(4)]
        zi = d4((u[0] - F(1, 2)) * 4)
        ti = F(1) if u[1] > treat_cut else F(0)
        if m_arms is not None:
            ti = F(i % 2)
        mi = d4(F(7, 10) * ti + F(1, 2) * zi + (u[2] - F(1, 2)) * 2 + m_offset)
        if m_arms is not None:
            lo, hi = m_arms[int(ti)]
            mi = d4(lo + (hi - lo) * u[2])
            if i < 2:
                mi = lo if i == 0 else hi
            if i in (2, 3):
                mi = hi if ti == 0 else lo
        yi = d4(1 + F(3, 10) * ti + F(6, 10) * (mi - m_offset) + F(2, 5) * zi
                + (u[3] - F(1, 2)) * 2 + y_offset)
        t.append(ti); m.append(mi); y.append(yi); z.append(zi)
    return {"name": name, "t": [float(v) for v in t], "m": [float(v) for v in m],
            "y": [float(v) for v in y], "z": [float(v) for v in z]}


CASES = [
    make_case("plain", 60, 51),
    make_case("offset", 56, 52, m_offset=50000, y_offset=1000000),
    make_case("unbalanced", 64, 53, treat_cut=F(85, 100)),
    # arms: control m in [1, 2], treated m in [0, 1]: the ranges touch at exactly 1.
    make_case("touching_ranges", 48, 54, m_arms={0: (F(1), F(2)), 1: (F(0), F(1))}),
    make_case("disjoint_ranges", 48, 55, m_arms={0: (F(1000001, 1000000), F(2)), 1: (F(0), F(1))}),
]
CONTRASTS = ["Total", "NaturalDirect", "NaturalIndirect"]


def solve(a, b):
    n = len(a)
    mm = [row[:] + [bb] for row, bb in zip(a, b)]
    for c in range(n):
        piv = next((r for r in range(c, n) if mm[r][c] != 0), None)
        if piv is None:
            return None
        mm[c], mm[piv] = mm[piv], mm[c]
        inv = 1 / mm[c][c]
        mm[c] = [v * inv for v in mm[c]]
        for r in range(n):
            if r != c and mm[r][c] != 0:
                f = mm[r][c]
                mm[r] = [v - f * w for v, w in zip(mm[r], mm[c])]
    return [mm[r][n] for r in range(n)]


def fit(cols, y):
    p = len(cols)
    xtx = [[sum(a * b for a, b in zip(cols[i], cols[j])) for j in range(p)] for i in range(p)]
    xty = [sum(a * b for a, b in zip(cols[i], y)) for i in range(p)]
    return solve(xtx, xty)


def effects(t, m, y, z, extra, rows):
    sel = lambda v: [F(v[i]) for i in rows]
    one = [F(1)] * len(rows)
    ex = [sel(e) for e in extra]
    am = fit([one, sel(t), sel(z)] + ex, sel(m))
    ay = fit([one, sel(t), sel(m), sel(z)] + ex, sel(y))
    if am is None or ay is None:
        return None
    a, c, b = am[1], ay[1], ay[2]
    return {"Total": c + a * b, "NaturalDirect": c, "NaturalIndirect": a * b}


def verdict(vals, null):
    k = len(vals)
    mean = sum(vals) / k
    var = sum((v - mean) ** 2 for v in vals) / (k - 1)
    if var == 0:
        p = mp.mpf(1) if null == mean else mp.mpf(0)
    else:
        sd = mp.sqrt(mp.mpf(var.numerator) / var.denominator)
        z = (mp.mpf(null.numerator) / null.denominator - mp.mpf(mean.numerator) / mean.denominator) / sd
        p = mp.erfc(abs(z) / mp.sqrt(2))
    return float(mean), float(p), bool(p >= mp.mpf("0.05"))


def overlap(t, m):
    r = {0: [None, None], 1: [None, None]}
    for ti, mi in zip(t, m):
        g = r[int(ti)]
        g[0] = mi if g[0] is None else min(g[0], mi)
        g[1] = mi if g[1] is None else max(g[1], mi)
    return max(r[0][0], r[1][0]) <= min(r[0][1], r[1][1])


def main():
    if sys.argv[1] == "data":
        print(json.dumps(CASES))
        return
    pert = json.load(open(sys.argv[2]))
    out = []
    for case in CASES:
        t, m, y, z = case["t"], case["m"], case["y"], case["z"]
        n = len(t)
        p = pert[case["name"]]
        allrows = list(range(n))
        orig = effects(t, m, y, z, [], allrows)
        runs = []
        for contrast, shift in [(c, sh) for c in CONTRASTS for sh in (F(0), F(1, 2))]:
            placebo = [effects(t, p["placebo_noise"][r], y, z, [], allrows)["NaturalIndirect"]
                       for r in range(20)]
            rcc = [effects(t, m, y, z, [p["rcc_noise"][r]], allrows)[contrast] for r in range(20)]
            sub = []
            for r in range(20):
                rows = [i for i in range(n) if p["subset_mask"][r][i]]
                e = effects(t, m, y, z, [], rows)
                sub.append(None if e is None else e[contrast])
            published = orig[contrast] + shift
            run = {"contrast": contrast, "shift": float(shift), "original": float(orig[contrast]),
                   "placebo": verdict(placebo, F(0)),
                   "rcc": verdict(rcc, published)}
            run["subset"] = None if any(s is None for s in sub) else verdict(sub, published)
            runs.append(run)
        out.append({"case": case, "overlap": overlap(t, m), "perturbations": p, "runs": runs})
    print(json.dumps(out))


main()
