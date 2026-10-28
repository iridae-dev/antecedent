# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath"]
# ///
"""Symbolic conditional mutual information and G-squared: statistic and exact within-stratum permutation p-value.

Regenerate with:  uv run --script symbolic_cmi_reference.py > symbolic_cmi_reference.json

For small discrete tables every arrangement of Y within the conditioning strata is enumerated
(a product of factorials of the stratum sizes, at most 40320).  The statistic is the
stratum-size-weighted mean over strata of at least two rows of the plug-in mutual information
MI = sum_{a,b} (c_ab/m) ln(c_ab m / (c_a. c_.b)), evaluated at 60 digits from integer counts.  The
exact p-value is the fraction of arrangements whose statistic is at least the observed one, where
G-squared is the sum over strata of 2 m MI.  Two statistics are equal when they differ by less than 1e-40 (they are equal as real numbers when
the count tables are equal up to relabeling; at 60 digits no other pair is that close for the
sizes here).  The product's p-value is a Monte Carlo add-one estimate, so its expectation is
(1 + R p_exact) / (1 + R) for R replicates.
Repository-authored (MIT OR Apache-2.0).
"""
import itertools, json
from collections import Counter
from math import factorial
from mpmath import mp, mpf, log

mp.dps = 60


def stratum_mi(xs, ys):
    m = len(xs)
    joint = Counter(zip(xs, ys)); cx = Counter(xs); cy = Counter(ys)
    return sum(mpf(c) / m * log(mpf(c) * m / (cx[a] * cy[b])) for (a, b), c in joint.items())


KIND = "mi"


def stratum_g2(xs, ys):
    # G^2 = 2 sum O ln(O / E) = 2 m MI for the plug-in table
    return 2 * len(xs) * stratum_mi(xs, ys)


def statistic(x, y, z):
    strata = {}
    for i, key in enumerate(z):
        strata.setdefault(key, []).append(i)
    num = mpf(0); den = 0
    for rows in strata.values():
        if len(rows) < 2:
            continue
        xs, ys = [x[i] for i in rows], [y[i] for i in rows]
        if KIND == "mi":
            num += len(rows) * stratum_mi(xs, ys)
            den += len(rows)
        else:
            num += stratum_g2(xs, ys)
            den = 1
    return num / den, strata


def exact_p(x, y, z):
    obs, strata = statistic(x, y, z)
    groups = list(strata.values())
    # distinct arrangements of y within each stratum, each with multiplicity
    per = []
    for rows in groups:
        vals = [y[i] for i in rows]
        arr = Counter(itertools.permutations(vals))
        per.append((rows, arr))
    total = 0; hit = 0
    tol = mpf(10) ** -40
    for combo in itertools.product(*[list(a.items()) for _, a in per]):
        yy = list(y); mult = 1
        for (rows, _), (perm, c) in zip(per, combo):
            for r, v in zip(rows, perm):
                yy[r] = v
            mult *= c
        s, _ = statistic(x, yy, z)
        total += mult
        if s >= obs - tol:
            hit += mult
    return float(obs), hit / total


cases = []


def add(name, x, y, z):
    global KIND
    KIND = "mi"
    obs, p = exact_p(x, y, z)
    KIND = "g2"
    g2, pg = exact_p(x, y, z)
    cases.append({"name": name, "x": x, "y": y, "z": z, "statistic": obs, "exact_p": p,
                  "g2_statistic": g2, "g2_exact_p": pg})


add("balanced_binary_unconditional", [0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 0, 1, 1, 1], [0] * 8)
add("balanced_binary_perfect_tie_rich", [0, 0, 1, 1, 0, 1, 0, 1], [0, 0, 1, 1, 1, 0, 0, 1], [0] * 8)
add("ternary_unconditional", [0, 1, 2, 0, 1, 2, 0, 1], [0, 1, 2, 1, 2, 0, 0, 1], [0] * 8)
add("two_strata", [0, 0, 1, 1, 0, 0, 1, 1], [0, 1, 1, 1, 0, 0, 1, 0], [0, 0, 0, 0, 1, 1, 1, 1])
add("three_strata_with_singleton", [0, 1, 0, 1, 0, 1, 1, 0, 0], [0, 1, 1, 0, 0, 1, 1, 1, 0], [0, 0, 0, 0, 1, 1, 1, 1, 2])
add("symbol_values_negative_and_large", [-3, -3, 7, 7, -3, 7, -3, 7], [10, 10, 40, 40, 40, 10, 10, 40], [0] * 8)


def g2_table(a, b, c, d):
    t = [[a, b], [c, d]]
    m = a + b + c + d
    r = [a + b, c + d]; cc = [a + c, b + d]
    tot = mpf(0)
    for i in range(2):
        for j in range(2):
            if t[i][j]:
                tot += 2 * t[i][j] * log(mpf(t[i][j]) * m / (r[i] * cc[j]))
    return tot


def g2_exact_p_2x2(strata):
    """strata: list of (x, y) binary columns; hypergeometric table law per stratum, exact."""
    from math import comb
    info = []
    for xs, ys in strata:
        n = len(xs); r1 = sum(xs); c1 = sum(ys); a_obs = sum(1 for u, v in zip(xs, ys) if u and v)
        rng = range(max(0, r1 + c1 - n), min(r1, c1) + 1)
        info.append((n, r1, c1, a_obs, rng))
    obs = sum(g2_table(a, r1 - a, c1 - a, n - r1 - c1 + a) for (n, r1, c1, a, _) in info)
    total = 1; hit = 0
    for n, r1, c1, _, rng in info:
        total *= comb(n, r1)
    tol = mpf(10) ** -40
    for combo in itertools.product(*[list(i[4]) for i in info]):
        w = 1; g = mpf(0)
        for a, (n, r1, c1, _, _) in zip(combo, info):
            w *= comb(c1, a) * comb(n - c1, r1 - a)
            g += g2_table(a, r1 - a, c1 - a, n - r1 - c1 + a)
        if g >= obs - tol:
            hit += w
    return float(obs), hit / total


g2cases = []


def add_g2(name, strata):
    x = [v for xs, _ in strata for v in xs]
    y = [v for _, ys in strata for v in ys]
    z = [k for k, (xs, _) in enumerate(strata) for _ in xs]
    g, p = g2_exact_p_2x2(strata)
    g2cases.append({"name": name, "x": x, "y": y, "z": z, "statistic": g, "exact_p": p})


bal = lambda a: [1] * a + [0] * (10 - a)
# one stratum of 24 with margins 12/12 and table [[9,3],[3,9]]: tied with [[3,9],[9,3]]
add_g2("single_stratum_24_rows_tied_mirror_tables",
       [([1] * 12 + [0] * 12, [1] * 9 + [0] * 3 + [1] * 3 + [0] * 9)])
add_g2("single_stratum_24_rows_independent_looking",
       [([1] * 12 + [0] * 12, [1] * 6 + [0] * 6 + [1] * 6 + [0] * 6)])
add_g2("two_strata_20_rows_each",
       [([1] * 10 + [0] * 10, [1] * 8 + [0] * 2 + [1] * 2 + [0] * 8),
        ([1] * 10 + [0] * 10, [1] * 3 + [0] * 7 + [1] * 7 + [0] * 3)])
print(json.dumps({"cases": cases, "g2_cases": g2cases}))
