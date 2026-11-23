# /// script
# requires-python = ">=3.11"
# ///
"""Exact-rational reference for the iid row-bootstrap coverage refuter.

Stages (the Rust test regenerates the resample indices from the library RNG and asserts they
equal the frozen ones, so the frozen file cannot drift from the product's draws):

  uv run --offline --script bootstrap_refute_reference.py data   > data.json
  (Rust: BOOTSTRAP_REFERENCE_DATA=data.json BOOTSTRAP_REFERENCE_OUT=indices.json
         cargo test -p antecedent-validate --test bootstrap_refute_reference emit_indices -- --ignored)
  uv run --offline --script bootstrap_refute_reference.py build indices.json > bootstrap_refute_reference.json

The statistic is written from its definition, not from the product code:
  ate  = coefficient of T in the least-squares fit of Y on (1, T, Z...), exact in Fractions;
  replicates = n row draws with replacement; ates sorted ascending, m = number of replicates;
  lo_idx = round_half_up((1-level)/2 * (m-1)), hi_idx = round_half_up((1+level)/2 * (m-1));
  mean = average of the replicate ates; lo = mean - (mean - ates[lo_idx]); hi = ates[hi_idx];
  passed = lo <= full-sample ate <= hi;  comparison = hi - lo.

Repository-authored (MIT OR Apache-2.0); standard library only.
"""
import json
import sys
from fractions import Fraction as F


def lcg(seed):
    state = seed
    while True:
        state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        yield Fraction_from_bits(state >> 40)


def Fraction_from_bits(v):
    return F(v, 1 << 24)


def dec(x, digits=4):
    return F(round(x * 10**digits), 10**digits)


def make_case(name, n, seed, nz, t_binary=True, z_offset=0, y_offset=0, collinear=False):
    g = lcg(seed)
    t, y, zs = [], [], [[] for _ in range(nz)]
    for i in range(n):
        u = [next(g) for _ in range(nz + 2)]
        zi = [dec((u[k] - F(1, 2)) * 4 + z_offset) for k in range(nz)]
        if collinear and nz >= 2:
            zi[1] = dec(zi[0] * 2 + (u[nz] - F(1, 2)) / 1000)
        ti = F(1) if u[nz] + F(1, 10) * (zi[0] - z_offset) > F(1, 2) else F(0)
        if not t_binary:
            ti = dec((u[nz] - F(1, 2)) * 3 + (zi[0] - z_offset) / 2)
        yi = dec(1 + F(4, 5) * ti + F(1, 2) * (zi[0] - z_offset)
                 + ((u[nz + 1] - F(1, 2)) * 2) + y_offset)
        t.append(ti)
        y.append(yi)
        for k in range(nz):
            zs[k].append(zi[k])
    return {"name": name, "t": [float(v) for v in t], "y": [float(v) for v in y],
            "z": [[float(v) for v in col] for col in zs]}


CASES = [
    {"case": make_case("plain", 30, 11, 2), "settings": [[41, 0.95], [11, 0.5], [21, 0.9], [51, 0.1]]},
    {"case": make_case("offset", 28, 12, 2, z_offset=100000, y_offset=1000000), "settings": [[41, 0.95], [11, 0.5]]},
    {"case": make_case("near_collinear", 36, 13, 2, collinear=True), "settings": [[21, 0.9]]},
    {"case": make_case("continuous_treatment", 26, 14, 1, t_binary=False), "settings": [[31, 0.8]]},
    {"case": make_case("tiny", 7, 15, 1), "settings": [[21, 0.9]]},
]


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


def ate(t, y, zs, rows):
    cols = [[F(1)] * len(rows), [F(t[i]) for i in rows]] + [[F(z[i]) for i in rows] for z in zs]
    yy = [F(y[i]) for i in rows]
    p = len(cols)
    xtx = [[sum(a * b for a, b in zip(cols[i], cols[j])) for j in range(p)] for i in range(p)]
    xty = [sum(a * b for a, b in zip(cols[i], yy)) for i in range(p)]
    beta = solve(xtx, xty)
    return None if beta is None else beta[1]


def round_half_up(x):
    return int((x * 2 + 1) // 2)


def interval(ates, level):
    s = sorted(ates)
    m = len(s)
    lo_frac = (1 - level) / 2
    lo_i = min(round_half_up(lo_frac * (m - 1)), m - 1)
    hi_i = min(round_half_up((1 - lo_frac) * (m - 1)), m - 1)
    mean = sum(s) / m
    return mean, mean - (mean - s[lo_i]), s[hi_i], lo_i, hi_i


def main():
    if sys.argv[1] == "data":
        print(json.dumps([c["case"] for c in CASES]))
        return
    indices = json.load(open(sys.argv[2]))
    out = []
    for entry in CASES:
        case = entry["case"]
        t, y, zs = case["t"], case["y"], case["z"]
        n = len(t)
        full = ate(t, y, zs, list(range(n)))
        runs = []
        for (reps, level) in entry["settings"]:
            idx = indices[case["name"]][str(reps)]
            ates = [ate(t, y, zs, rows) for rows in idx]
            if any(a is None for a in ates):
                runs.append({"replicates": reps, "level": level, "indices": idx,
                             "singular": True})
                continue
            lvl = F(str(level))
            mean, lo, hi, lo_i, hi_i = interval(ates, lvl)
            runs.append({"replicates": reps, "level": level, "indices": idx, "singular": False,
                         "lo_idx": lo_i, "hi_idx": hi_i,
                         "mean": float(mean), "lo": float(lo), "hi": float(hi),
                         "width": float(hi - lo), "passed": bool(lo <= full <= hi),
                         "full": float(full), "ates": [float(a) for a in ates]})
        out.append({"case": case, "full": float(full), "runs": runs})
    print(json.dumps(out))


main()
