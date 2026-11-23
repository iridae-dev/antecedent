# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Exact references for the joint-law arithmetic of antecedent-stats.

Regenerate with:  uv run --script joint_law_reference.py > joint_law_reference.json

Coupling extremes are NOT taken from the comonotone / countermonotone / Makarov closed forms of
the product.  For two finite marginals whose masses are k/N, every vertex of the transportation
polytope is an integer table with margins (k_x), (k_y), and every functional below is linear in
the coupling, so its exact extreme is the extreme over all integer N-scaled contingency tables.
This script enumerates those tables in exact rational arithmetic (values are dyadic, so no
rounding enters) and reports, for each case, E[g] for five value functions, Cov, Var(aX+bY),
P(X+Y<=t) and P(X-Y<=t) over a grid of thresholds containing every atom sum and difference and
the points a quarter either side.  A value function whose mixed difference is not one-signed on
the lattice is recorded as "refuse" from a check over every pair of lattice rows and columns.

Moments of linear combinations, mixtures, correlation ranges, draw spreads, the Agresti-Coull
exceedance band, the normal interval and the equal-tail mixture quantile interval are evaluated
from their definitions (Fractions; mpmath at 50 digits for the normal quantile).

Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import itertools
import json
import random
from fractions import Fraction as F

import mpmath as mp

mp.mp.dps = 50
rng = random.Random(20261028)


def tables(rows, cols):
    """All non-negative integer tables with the given row and column sums."""
    if len(rows) == 1:
        yield [list(cols)]
        return

    def fill(r, remaining_cols):
        if r == len(rows) - 1:
            yield [list(remaining_cols)]
            return
        for row in compositions(rows[r], remaining_cols):
            rest = [c - x for c, x in zip(remaining_cols, row)]
            for tail in fill(r + 1, rest):
                yield [row] + tail

    yield from fill(0, list(cols))


def compositions(total, caps):
    if len(caps) == 1:
        if total <= caps[0]:
            yield [total]
        return
    for first in range(min(total, caps[0]) + 1):
        for rest in compositions(total - first, caps[1:]):
            yield [first] + rest


FUNCTIONALS = {
    "product": (lambda x, y: x * y, "super"),
    "ratio": (lambda x, y: x / y if y != 0 else None, "sub"),
    "minimum": (min, "super"),
    "maximum": (max, "sub"),
    "squared_difference": (lambda x, y: (x - y) ** 2, "sub"),
}


def lattice_sign_ok(g, xs, ys, kind):
    for i1, i2 in itertools.combinations(range(len(xs)), 2):
        for j1, j2 in itertools.combinations(range(len(ys)), 2):
            vals = [g(xs[i2], ys[j2]), g(xs[i1], ys[j1]), g(xs[i2], ys[j1]), g(xs[i1], ys[j2])]
            if any(v is None for v in vals):
                return False
            mixed = vals[0] + vals[1] - vals[2] - vals[3]
            if kind == "super" and mixed < 0:
                return False
            if kind == "sub" and mixed > 0:
                return False
    return True


def case(name, xs, kx, ys, ky, a, b):
    n = sum(kx)
    assert n == sum(ky)
    tabs = list(tables(kx, ky))
    out = {"name": name, "n": n, "x": [float(v) for v in xs], "kx": kx, "y": [float(v) for v in ys],
           "ky": ky, "a": float(a), "b": float(b)}

    def extremes(fn):
        vals = []
        for t in tabs:
            total = F(0)
            for i, row in enumerate(t):
                for j, c in enumerate(row):
                    if c:
                        v = fn(xs[i], ys[j])
                        total += F(c, n) * v
            vals.append(total)
        return min(vals), max(vals)

    funcs = {}
    for fname, (g, kind) in FUNCTIONALS.items():
        if lattice_sign_ok(g, xs, ys, kind):
            lo, hi = extremes(g)
            funcs[fname] = [float(lo), float(hi)]
        else:
            funcs[fname] = "refuse"
    out["functional"] = funcs
    mx = sum(F(k, n) * v for k, v in zip(kx, xs))
    my = sum(F(k, n) * v for k, v in zip(ky, ys))
    vx = sum(F(k, n) * (v - mx) ** 2 for k, v in zip(kx, xs))
    vy = sum(F(k, n) * (v - my) ** 2 for k, v in zip(ky, ys))
    lo, hi = extremes(lambda x, y: x * y)
    out["product_moment"] = [float(lo), float(hi)]
    out["covariance"] = [float(lo - mx * my), float(hi - mx * my)]
    cl, ch = lo - mx * my, hi - mx * my
    cand = [a * a * vx + b * b * vy + 2 * a * b * c for c in (cl, ch)]
    out["linear_variance"] = [float(min(cand)), float(max(cand))]
    out["mean_x"], out["mean_y"] = float(mx), float(my)
    out["var_x"], out["var_y"] = float(vx), float(vy)
    ts = sorted({x + y + d for x in xs for y in ys for d in (F(0), F(1, 4), F(-1, 4))}
                | {x - y + d for x in xs for y in ys for d in (F(0), F(1, 4), F(-1, 4))})
    sums, diffs = [], []
    for t in ts:
        for fn, store in ((lambda x, y: x + y, sums), (lambda x, y: x - y, diffs)):
            vals = []
            for tab in tabs:
                p = sum(F(c, n) for i, row in enumerate(tab) for j, c in enumerate(row)
                        if fn(xs[i], ys[j]) <= t)
                vals.append(p)
            store.append([float(t), float(min(vals)), float(max(vals))])
    out["sum_cdf"], out["difference_cdf"] = sums, diffs
    return out


def rand_atoms(m, lo, hi):
    return sorted(F(rng.randint(4 * lo, 4 * hi), 4) for _ in range(m))


def dedupe(vs):
    return sorted(set(vs))


def random_case(idx):
    n = rng.choice([4, 5, 6, 8])
    mx, my = rng.choice([2, 3]), rng.choice([2, 3])
    xs = dedupe(rand_atoms(mx, -3, 3))
    ys = dedupe(rand_atoms(my, -3, 3))
    kx = random_split(n, len(xs))
    ky = random_split(n, len(ys))
    a, b = rng.choice([F(1), F(-1), F(1, 2), F(3, 2)]), rng.choice([F(1), F(-1), F(1, 4), F(-2)])
    return case(f"random_{idx:02d}", xs, kx, ys, ky, a, b)


def random_split(n, parts):
    cuts = sorted(rng.sample(range(1, n), parts - 1)) if parts > 1 else []
    edges = [0] + cuts + [n]
    return [edges[i + 1] - edges[i] for i in range(parts)]


cases = [
    case("positive_support_ratio_valid", [F(1), F(2), F(4)], [2, 2, 2], [F(1, 2), F(1), F(2)], [1, 3, 2], F(1), F(-1)),
    case("ratio_denominator_crosses_zero_refused", [F(1), F(2)], [3, 3], [F(-1), F(1)], [3, 3], F(1), F(1)),
    case("ties_between_marginals", [F(0), F(1), F(2)], [2, 2, 2], [F(0), F(1), F(2)], [2, 2, 2], F(1), F(-1)),
    case("single_atoms", [F(3)], [5], [F(-2)], [5], F(2), F(3)),
    case("one_degenerate_one_spread", [F(1), F(5)], [1, 7], [F(2)], [8], F(1), F(1)),
    case("unequal_atom_counts_unequal_mass", [F(-2), F(0), F(1), F(3)], [1, 2, 1, 2], [F(-1), F(2)], [4, 2], F(1), F(-1)),
    case("bernoulli_like", [F(0), F(1)], [3, 5], [F(0), F(1)], [2, 6], F(1), F(-1)),
]
cases += [random_case(i) for i in range(12)]

# ------------------------------------------------------------- moments
def lin(coef, mean, cov):
    k = len(coef)
    return (float(sum(a * m for a, m in zip(coef, mean))),
            float(sum(coef[i] * coef[j] * cov[i][j] for i in range(k) for j in range(k))))


linear = []
for coef, mean, cov in [
    ([F(1), F(-1)], [F(5), F(2)], [[F(4), F(3)], [F(3), F(9)]]),
    ([F(1, 4), F(3, 4)], [F(5), F(2)], [[F(4), F(3)], [F(3), F(9)]]),
    ([F(1), F(2), F(-3)], [F(1, 2), F(-1), F(4)], [[F(2), F(1, 2), F(0)], [F(1, 2), F(3), F(1)], [F(0), F(1), F(5)]]),
    ([F(1), F(-1)], [F(0), F(0)], [[F(1), F(1)], [F(1), F(1)]]),  # perfectly correlated, variance 0
]:
    m, v = lin(coef, mean, cov)
    linear.append({"coef": [float(c) for c in coef], "mean": [float(c) for c in mean],
                   "cov": [float(c) for row in cov for c in row], "want_mean": m, "want_variance": v})

mixtures = []
for w, mu, var in [
    ([F(1, 4), F(3, 4)], [F(0), F(4)], [F(1), F(4)]),
    ([F(1, 8), F(3, 8), F(1, 2)], [F(-2), F(1), F(3)], [F(1, 2), F(0), F(5)]),
    ([F(1), F(0)], [F(7), F(-7)], [F(2), F(100)]),
]:
    mean = sum(a * b for a, b in zip(w, mu))
    within = sum(a * b for a, b in zip(w, var))
    between = sum(a * (m - mean) ** 2 for a, m in zip(w, mu))
    mixtures.append({"weights": [float(x) for x in w], "means": [float(x) for x in mu],
                     "variances": [float(x) for x in var], "mean": float(mean),
                     "within": float(within), "between": float(between), "variance": float(within + between)})

corr = []
for xm, xs_, ym, ys_, a, b, lo, hi in [
    (F(1), F(2), F(-1), F(3), F(1), F(1), F(-1, 2), F(3, 4)),
    (F(0), F(1), F(0), F(1), F(1), F(-1), F(-1), F(1)),
    (F(2), F(0), F(5), F(4), F(2), F(3), F(1, 4), F(1, 2)),
]:
    def at(r):
        return (xm * ym + r * xs_ * ys_, r * xs_ * ys_, a * a * xs_ ** 2 + b * b * ys_ ** 2 + 2 * a * b * r * xs_ * ys_)
    plo, phi = at(lo), at(hi)
    corr.append({"x_mean": float(xm), "x_sd": float(xs_), "y_mean": float(ym), "y_sd": float(ys_),
                 "a": float(a), "b": float(b), "lo": float(lo), "hi": float(hi),
                 "product_moment": [float(min(plo[0], phi[0])), float(max(plo[0], phi[0]))],
                 "covariance": [float(min(plo[1], phi[1])), float(max(plo[1], phi[1]))],
                 "linear_variance": [float(max(min(plo[2], phi[2]), 0)), float(max(max(plo[2], phi[2]), 0))]})

# ------------------------------------------------------- draws
def ppf(p):
    return float(mp.sqrt(2) * mp.erfinv(2 * mp.mpf(p) - 1))


spreads = []
for draws, eff in [([F(1), F(2), F(4), F(8)], None), ([F(-3), F(0), F(1, 2), F(2), F(2)], 2.0),
                   ([F(5), F(5)], None), ([F(1, 4), F(9, 4), F(-7, 4)], 3.0)]:
    n = len(draws)
    mean = sum(draws) / n
    var = sum((d - mean) ** 2 for d in draws) / (n - 1)
    size = F(n) if eff is None else F(eff)
    sd = mp.sqrt(mp.mpf(var.numerator) / var.denominator)
    se = mp.sqrt(mp.mpf(var.numerator) / var.denominator / (mp.mpf(size.numerator) / size.denominator))
    spreads.append({"draws": [float(d) for d in draws], "effective_size": eff, "mean": float(mean),
                    "variance": float(var), "sd": float(sd), "mean_se": float(se)})

exceed = []
for draws, thr, cutoff, level, eff in [
    ([float(i) for i in range(20)], 14.5, 0.25, 0.95, None),
    ([float(i) for i in range(20)], 100.0, 0.5, 0.9, None),
    ([float(i) for i in range(20)], -1.0, 0.5, 0.9, None),
    ([float(i % 7) for i in range(200)], 4.0, 0.3, 0.95, 50.0),
    ([float(i % 5) for i in range(1000)], 1.0, 0.6, 0.99, None),
]:
    n = len(draws)
    hits = sum(1 for d in draws if d > thr)
    p = mp.mpf(hits) / n
    size = mp.mpf(n if eff is None else eff)
    z = mp.sqrt(2) * mp.erfinv(2 * (mp.mpf(1) / 2 + mp.mpf(level) / 2) - 1)
    adj = (p * size + z * z / 2) / (size + z * z)
    se = mp.sqrt(adj * (1 - adj) / (size + z * z))
    band = z * se
    verdict = "act" if p - band > cutoff else "withhold" if p + band < cutoff else "unresolved"
    exceed.append({"draws": draws, "threshold": thr, "cutoff": cutoff, "level": level, "effective_size": eff,
                   "probability": float(p), "se": float(se), "band": float(band), "verdict": verdict})

normal_ints = []
for mean, var, level in [(1.5, 4.0, 0.95), (-2.0, 0.25, 0.5), (0.0, 1e-12, 0.999), (3.0, 0.0, 0.9), (0.0, 9.0, 0.6827)]:
    z = mp.sqrt(2) * mp.erfinv(mp.mpf(level))  # P(|Z|<=z)=level
    half = z * mp.sqrt(var)
    normal_ints.append({"mean": mean, "variance": var, "level": level, "lower": float(mean - half), "upper": float(mean + half)})

# mixture marginal: quantile interval and exceedance
mix = []
for w, comps, level, thr in [
    ([F(1, 4), F(3, 4)], [[F(0), F(1), F(2), F(3)], [F(2), F(5)]], F(3, 4), F(2)),
    ([F(1, 2), F(1, 2)], [[F(1), F(1), F(2)], [F(1), F(3), F(4), F(8)]], F(1, 2), F(1)),
    ([F(1, 8), F(7, 8)], [[F(-5)], [F(0), F(1), F(2), F(3), F(4), F(5), F(6), F(7)]], F(7, 8), F(3)),
]:
    atoms = {}
    for wi, c in zip(w, comps):
        for v in c:
            atoms[v] = atoms.get(v, F(0)) + wi / len(c)
    keys = sorted(atoms)
    cum = F(0)
    cdf = []
    for k in keys:
        cum += atoms[k]
        cdf.append((k, cum))

    def q(p):
        for k, c in cdf:
            if c >= p:
                return k
        return keys[-1]

    mean = sum(k * atoms[k] for k in keys)
    var = sum(atoms[k] * (k - mean) ** 2 for k in keys)
    pe = sum(wi * F(sum(1 for d in c if d > thr), len(c)) for wi, c in zip(w, comps))
    se2 = sum(wi * wi * F(sum(1 for d in c if d > thr), len(c)) * (1 - F(sum(1 for d in c if d > thr), len(c))) / len(c)
              for wi, c in zip(w, comps))
    mix.append({"weights": [float(x) for x in w], "components": [[float(d) for d in c] for c in comps],
                "level": float(level), "threshold": float(thr),
                "atoms": [[float(k), float(atoms[k])] for k in keys], "mean": float(mean), "variance": float(var),
                "lower": float(q((1 - level) / 2)), "upper": float(q((1 + level) / 2)),
                "exceedance": float(pe), "exceedance_se": float(mp.sqrt(mp.mpf(se2.numerator) / se2.denominator))})

print(json.dumps({"coupling": cases, "linear": linear, "mixtures": mixtures, "correlation": corr,
                  "spreads": spreads, "exceedance": exceed, "normal_interval": normal_ints,
                  "mixture_marginal": mix}, indent=1))
