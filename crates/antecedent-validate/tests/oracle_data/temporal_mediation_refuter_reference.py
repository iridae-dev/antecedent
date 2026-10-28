# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Exact reference for the temporal-mediation refuter suite on lag-aligned windows of series with
gaps: placebo mediators, a random common cause, a contiguous 80-percent window, and the
binary-treatment mediator-support diagnostic.

Stages (the Rust test regenerates the perturbations from the library RNG streams and asserts
they equal the frozen ones):

  uv run --script temporal_mediation_refuter_reference.py data > data.json
  (Rust: TEMPORAL_MEDIATION_REFUTER_DATA=data.json TEMPORAL_MEDIATION_REFUTER_OUT=perturbations.json
         cargo test -p antecedent-validate --test temporal_mediation_refuter_reference emit -- --ignored)
  uv run --script temporal_mediation_refuter_reference.py build perturbations.json \
         > temporal_mediation_refuter_reference.json

Written from the definitions, with no product code.  A series has a treatment T, mediators
M_1..M_q, an outcome Y and a covariate Z, each with its own missing time points (the stored value
at a missing point is a large junk number that a correct reader never touches).  With treatment
lag h and covariate lags L the window ending at time t is intact when T[t-h], every M_j[t], Y[t]
and Z[t-l] (l in L) are observed and t >= max(h, L).  Over the intact windows, with W = [1, Z
lags, extra columns]:
  a_j  = coefficient of T in  M_j ~ [T, W]
  c'   = coefficient of T and b_j those of M_j in  Y ~ [T, M_1..M_q, W]
  c    = coefficient of T in  Y ~ [T, W]
  Total = delta c, Direct = delta c', Mediated = delta sum_j a_j b_j   (delta = active - control).
Replicate r (20 of them):
  placebo : every M_j replaced by library noise (stream 0xF0140000 + r for j = 0, 0xF0148000 +
            100 r + j beyond), validity kept; contrast Mediated; null 0;
  rcc     : M untouched, the j = 0 noise vector appended to W, the query's contrast, null = the
            published estimate;
  window  : one contiguous block of floor(0.8 n) rows, start floor(u (n - len + 1)) for the first
            uniform u of stream 0xF0150000 + r, rows outside dropped, windows recomputed inside it;
            the query's contrast, null as rcc.
  mean, sample SD (n-1), p = erfc(|null - mean| / (sd sqrt 2)) at 60 digits, passed = p >= 0.05.
Support (binary treatment only, i.e. exactly two distinct T values among the intact windows): per
mediator the closed ranges of M among the windows of each arm; the shared fraction is
max(0, min(hi) - max(lo)) / (max(hi) - min(lo)) (0 for zero width), the report keeps the least
shared mediator, and the ranges intersect when max(lo) <= min(hi).

Repository-authored (MIT OR Apache-2.0).
"""
import json
import math
import sys
from fractions import Fraction as F

import mpmath as mp

mp.mp.dps = 60
JUNK = 1.0e6


def lcg(seed):
    state = seed
    while True:
        state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        yield F(state >> 40, 1 << 24)


def d4(x):
    return F(round(x * 10**4), 10**4)


def make_case(name, n, seed, h, q, adj, missing, gaps=True, arms=None, levels=(0, 1), delta=1,
              force_t=None, force_m=None):
    """missing = {"t": [...], "m": [[...] per mediator], "y": [...], "z": [...]}."""
    g = lcg(seed)
    force_t = force_t or {}
    force_m = force_m or {}
    T, Z = [], []
    for i in range(n):
        u0, u1 = next(g), next(g)
        Z.append(d4((u0 - F(1, 2)) * 4))
        T.append(F(force_t.get(i, levels[int(u1 * len(levels))])))
    M = [[] for _ in range(q)]
    Y = []
    for t in range(n):
        tl, zl = T[max(t - h, 0)], Z[max(t - 1, 0)]
        row = []
        for j in range(q):
            u = next(g)
            if arms is not None:
                lo, hi = arms[j][int(tl)]
                m = d4(lo + (hi - lo) * u)
            else:
                m = d4((F(7, 10) - F(2, 5) * j) * tl + F(1, 2) * zl + (u - F(1, 2)) * 2)
            m = force_m.get((j, t), m)
            M[j].append(m)
            row.append(m)
        u = next(g)
        Y.append(d4(1 + F(3, 10) * tl + sum((F(9, 10) - F(1, 2) * j) * row[j] for j in range(q))
                    + F(2, 5) * zl + (u - F(1, 2)) * 2))
    miss = {"t": set(missing.get("t", [])), "y": set(missing.get("y", [])),
            "z": set(missing.get("z", [])),
            "m": [set(s) for s in missing.get("m", [[] for _ in range(q)])]}
    flt = lambda v, bad: [JUNK if i in bad else float(x) for i, x in enumerate(v)]
    return {
        "name": name, "h": h, "q": q, "adj": adj, "gaps": gaps, "delta": delta,
        "t": flt(T, miss["t"]), "m": [flt(M[j], miss["m"][j]) for j in range(q)],
        "y": flt(Y, miss["y"]), "z": flt(Z, miss["z"]),
        "miss_t": sorted(miss["t"]), "miss_m": [sorted(s) for s in miss["m"]],
        "miss_y": sorted(miss["y"]), "miss_z": sorted(miss["z"]),
    }


def arms_for(*pairs):
    return [{0: (F(a), F(b)), 1: (F(c), F(d))} for a, b, c, d in pairs]


CASES = [
    make_case("gapped_one_mediator", 52, 61, 1, 1, [1],
              {"t": [9], "m": [[20, 21]], "y": [40], "z": [33]}),
    make_case("two_mediators_lag2_scaled", 60, 62, 2, 2, [1, 2],
              {"t": [15], "m": [[], [27, 28, 29]], "y": [44]}, delta=2,
              arms=arms_for((0, 2, 1, 3), (0, 2, F(3, 2), F(7, 2)))),
    make_case("gapless_flag_off", 50, 63, 1, 1, [], {}, gaps=False),
    # arms touch at exactly 1 inside the intact windows; the dropped window 22 holds a mediator
    # value that would widen the shared range if the support read it
    make_case("touching_inside_windows", 56, 64, 1, 1, [1], {"y": [22]},
              arms=arms_for((1, 2, 0, 1)), force_t={21: 1, 29: 1, 30: 0},
              force_m={(0, 30): F(1), (0, 31): F(1), (0, 22): F(19, 10)}),
    # raw ranges overlap (row 22 is a treated-arm row valued 1.7) but the intact windows are disjoint
    make_case("disjoint_inside_windows", 56, 65, 1, 1, [1], {"y": [22]},
              arms=arms_for((F(3, 2), F(5, 2), 0, 1)), force_t={21: 1},
              force_m={(0, 22): F(17, 10)}),
    make_case("three_level_treatment", 48, 66, 1, 1, [1], {"m": [[14]]}, levels=(0, 1, 2)),
]
CONTRASTS = ["Total", "NaturalDirect", "Mediated"]


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


class Series:
    """T, M (list), Y, Z with per-variable observed flags."""

    def __init__(self, case, m_values=None, start=0, length=None):
        n = len(case["t"])
        stop = n if length is None else start + length
        sl = lambda v: v[start:stop]
        self.n = stop - start
        self.t = sl(case["t"])
        self.m = [sl(v) for v in (m_values if m_values is not None else case["m"])]
        self.y = sl(case["y"])
        self.z = sl(case["z"])
        ok = lambda bad: [(start + i) not in set(bad) for i in range(self.n)]
        self.ok_t = ok(case["miss_t"])
        self.ok_m = [ok(b) for b in case["miss_m"]]
        self.ok_y = ok(case["miss_y"])
        self.ok_z = ok(case["miss_z"])
        self.h, self.adj = case["h"], case["adj"]

    def windows(self):
        first = max([self.h] + self.adj)
        return [t for t in range(first, self.n)
                if self.ok_t[t - self.h] and all(o[t] for o in self.ok_m) and self.ok_y[t]
                and all(self.ok_z[t - l] for l in self.adj)]


def effects(s, extra=()):
    rows = s.windows()
    one = [F(1)] * len(rows)
    tt = [F(s.t[r - s.h]) for r in rows]
    ws = [[F(s.z[r - l]) for r in rows] for l in s.adj] + [[F(e[r]) for r in rows] for e in extra]
    ms = [[F(m[r]) for r in rows] for m in s.m]
    y = [F(s.y[r]) for r in rows]
    a = []
    for mj in ms:
        co = fit([one, tt] + ws, mj)
        a.append(co[1])
    cb = fit([one, tt] + ms + ws, y)
    cc = fit([one, tt] + ws, y)
    cp, b = cb[1], cb[2:2 + len(ms)]
    med = sum(x * z for x, z in zip(a, b))
    assert cc[1] == cp + med  # OLS identity: reduced form = direct + sum of path products
    return {"Total": cc[1], "NaturalDirect": cp, "Mediated": med}, len(rows)


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


def support(s):
    rows = s.windows()
    levels = sorted(set(s.t[r - s.h] for r in rows))
    if len(levels) != 2:
        return None
    fraction, intersect = None, True
    for m in s.m:
        rng = [[None, None], [None, None]]
        for r in rows:
            g = rng[int(s.t[r - s.h] == levels[1])]
            g[0] = F(m[r]) if g[0] is None else min(g[0], F(m[r]))
            g[1] = F(m[r]) if g[1] is None else max(g[1], F(m[r]))
        lo, hi = max(rng[0][0], rng[1][0]), min(rng[0][1], rng[1][1])
        width = max(rng[0][1], rng[1][1]) - min(rng[0][0], rng[1][0])
        frac = max(F(0), hi - lo) / width if width > 0 else F(0)
        fraction = frac if fraction is None else min(fraction, frac)
        intersect &= lo <= hi
    return {"fraction": float(fraction), "intersect": bool(intersect)}


def raw_overlap_would_pass(case):
    """Ranges over every observed row of the raw series (what a window-blind reader sees)."""
    s = Series(case)
    rows = [r for r in range(s.h, s.n) if s.ok_t[r - s.h] and all(o[r] for o in s.ok_m)]
    levels = sorted(set(s.t[r - s.h] for r in rows))
    out = True
    for m in s.m:
        rng = [[None, None], [None, None]]
        for r in rows:
            g = rng[int(s.t[r - s.h] == levels[1])]
            g[0] = F(m[r]) if g[0] is None else min(g[0], F(m[r]))
            g[1] = F(m[r]) if g[1] is None else max(g[1], F(m[r]))
        out &= max(rng[0][0], rng[1][0]) <= min(rng[0][1], rng[1][1])
    return out


def main():
    if sys.argv[1] == "data":
        print(json.dumps(CASES))
        return
    pert = json.load(open(sys.argv[2]))
    out = []
    for case in CASES:
        n, q, delta = len(case["t"]), case["q"], F(case["delta"])
        p = pert[case["name"]]
        full = Series(case)
        orig, nrows = effects(full)
        placebo = []
        rcc = []
        win = []
        for r in range(20):
            noise = p["noise"][r]
            pl, _ = effects(Series(case, m_values=noise))
            placebo.append(pl["Mediated"] * delta)
            rcc.append(effects(full, extra=[noise[0]])[0])
            length = min(max(int(math.floor(n * 0.8)), 1), n - 1)
            span = n - length
            start = min(int(math.floor(p["window_u"][r] * (span + 1.0))), span)
            win.append(effects(Series(case, start=start, length=length))[0])
        runs = []
        for contrast in CONTRASTS:
            for shift in (F(0), F(1, 2)):
                published = orig[contrast] * delta + shift
                runs.append({
                    "contrast": contrast, "shift": float(shift),
                    "original": float(orig[contrast] * delta),
                    "placebo": verdict(placebo, F(0)),
                    "rcc": verdict([x[contrast] * delta for x in rcc], published),
                    "window": verdict([x[contrast] * delta for x in win], published),
                })
        sup = support(full)
        if case["name"] == "disjoint_inside_windows":
            assert raw_overlap_would_pass(case) and sup["fraction"] == 0 and not sup["intersect"]
        if case["name"] == "touching_inside_windows":
            assert sup["fraction"] == 0 and sup["intersect"]
        out.append({"case": case, "rows": nrows, "support": sup, "perturbations": p, "runs": runs})
    print(json.dumps(out))


main()
