# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Reference for adaptive_inference.participant_family (declared-correlation posterior).

Regenerate with:  uv run --offline --script participant_exact_reference.py > participant_exact_reference.json

Exact rational arithmetic only.  A participant of an arm with mean p and intra-participant
correlation rho gives m binary responses whose success count s has the beta-binomial mass

    P(s | p) = C(m, s) prod_{j<s} (p + j d) prod_{j<m-s} (1 - p + j d) / prod_{j<m} (1 + j d),
    d = rho / (1 - rho),

a polynomial of degree m in p.  With an integer-shape Beta(a, b) prior the posterior density of an arm
is proportional to a polynomial in p, so its normalizer, moments, distribution function, the
probability that an arm is largest (the integral of its density times the product of the other arms'
distribution functions), the expected maximum (1 minus the integral of the product of all arms'
distribution functions) and the exceedance of the control by a margin delta
(the integral over [delta, 1] of density times the control's distribution function at p - delta)
are finite sums of Fractions.  No quadrature and no special function enters.  Also: the
beta-binomial mass for m = 2..32, the power-prior arithmetic of a borrowed prior, and the
exploration-floor row of 2 to 6 arms defined as p_i' = max(floor, c p_i) with the unique c making
the row sum to one (found exactly from the sorted breakpoints, not by iterating the product's
clamp-and-rescale).

The correlation prior (estimated correlation) is not covered here.  Repository-authored
(MIT OR Apache-2.0); standard library only.
"""
import json
from fractions import Fraction as F
from math import comb


def txt(x):
    return f"{x.numerator}/{x.denominator}"


# ---- polynomial algebra in p (coefficient lists, index = power) ----

def pmul(a, b):
    out = [F(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                out[i + j] += x * y
    return out


def padd(a, b):
    n = max(len(a), len(b))
    return [(a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0) for i in range(n)]


def ppow(a, k):
    out = [F(1)]
    for _ in range(k):
        out = pmul(out, a)
    return out


def pint(a):
    """Antiderivative from 0."""
    return [F(0)] + [c / (i + 1) for i, c in enumerate(a)]


def peval(a, x):
    total = F(0)
    for c in reversed(a):
        total = total * x + c
    return total


def pdef(a, lo, hi):
    anti = pint(a)
    return peval(anti, hi) - peval(anti, lo)


def pshift(a, delta):
    """Coefficients of q(p) = a(p - delta)."""
    out = [F(0)]
    base = [-delta, F(1)]
    power = [F(1)]
    for c in a:
        out = padd(out, [c * x for x in power])
        power = pmul(power, base)
    return out


def arm_density(m, rho, a, b, counts):
    d = rho / (1 - rho)
    poly = pmul(ppow([F(0), F(1)], a - 1), ppow([F(1), F(-1)], b - 1))
    for s, n in enumerate(counts):
        if n == 0:
            continue
        like = [F(1)]
        for j in range(s):
            like = pmul(like, [j * d, F(1)])
        for j in range(m - s):
            like = pmul(like, [1 + j * d, F(-1)])
        poly = pmul(poly, ppow(like, n))
    z = pdef(poly, F(0), F(1))
    return [c / z for c in poly]


def functionals(m, rho, arms, margin):
    dens = [arm_density(m, rho, a, b, c) for (a, b, c) in arms]
    cdf = [pint(f) for f in dens]
    mean = [pdef(pmul([F(0), F(1)], f), F(0), F(1)) for f in dens]
    second = [pdef(pmul(ppow([F(0), F(1)], 2), f), F(0), F(1)) for f in dens]
    sd2 = [s - mu * mu for s, mu in zip(second, mean)]
    best = []
    for k, f in enumerate(dens):
        prod = f
        for j, g in enumerate(cdf):
            if j != k:
                prod = pmul(prod, g)
        best.append(pdef(prod, F(0), F(1)))
    allprod = [F(1)]
    for g in cdf:
        allprod = pmul(allprod, g)
    emax = 1 - pdef(allprod, F(0), F(1))
    loss = [emax - mu for mu in mean]
    exceed = []
    if margin is not None:
        for k in range(1, len(dens)):
            shifted = pshift(cdf[0], margin)
            exceed.append(pdef(pmul(dens[k], shifted), margin, F(1)))
    return mean, sd2, best, loss, exceed


cases = []


def case(name, m, rho, arms, margin=None):
    mean, sd2, best, loss, exceed = functionals(m, rho, arms, margin)
    assert sum(best) == 1
    cases.append(
        {
            "name": name,
            "responses": m,
            "rho": txt(rho),
            "margin": None if margin is None else txt(margin),
            "arms": [{"alpha": a, "beta": b, "counts": c} for (a, b, c) in arms],
            "mean": [float(x) for x in mean],
            "variance": [float(x) for x in sd2],
            "best": [float(x) for x in best],
            "loss": [float(x) for x in loss],
            "exceedance": [float(x) for x in exceed],
        }
    )


case("m3_quarter_two_arms", 3, F(1, 4), [(1, 1, [3, 5, 4, 6]), (2, 3, [1, 2, 6, 9])])
case("m6_half_three_arms_with_margin", 6, F(1, 2),
     [(1, 1, [2, 3, 4, 3, 2, 1, 0]), (3, 2, [0, 1, 2, 5, 4, 3, 2]), (1, 4, [4, 3, 2, 1, 0, 0, 0])], F(1, 10))
case("m4_independent_responses", 4, F(0), [(1, 1, [1, 4, 6, 4, 1]), (2, 2, [0, 1, 3, 6, 5])], F(1, 20))
case("m5_strong_correlation", 5, F(9, 10), [(1, 1, [8, 0, 0, 1, 0, 3]), (1, 1, [1, 0, 0, 0, 1, 10])])
case("m2_no_participants_is_the_prior", 2, F(1, 3), [(2, 3, [0, 0, 0]), (4, 1, [0, 0, 0])])
case("m3_extreme_counts_all_failures_against_all_successes", 3, F(1, 5), [(1, 1, [30, 0, 0, 0]), (1, 1, [0, 0, 0, 30])])
case("m3_large_sample_adjacent_arms", 3, F(1, 4), [(1, 1, [18, 20, 21, 21]), (1, 1, [17, 20, 22, 21])], F(0))
case("m8_four_arms", 8, F(1, 3),
     [(1, 1, [1, 2, 3, 4, 4, 3, 2, 1, 0]), (1, 1, [0, 1, 2, 3, 4, 4, 3, 2, 1]), (2, 2, [3, 3, 3, 3, 3, 3, 3, 3, 3]),
      (1, 3, [2, 2, 2, 2, 2, 2, 2, 2, 2])])


def pmf(m, mean, rho):
    d = rho / (1 - rho) if rho < 1 else None
    norm = F(1)
    for j in range(m):
        norm *= 1 + j * d
    out = []
    for s in range(m + 1):
        v = F(comb(m, s))
        for j in range(s):
            v *= mean + j * d
        for j in range(m - s):
            v *= 1 - mean + j * d
        out.append(v / norm)
    assert sum(out) == 1
    return out


pmfs = []
for m in (2, 3, 7, 16, 32):
    for mean in (F(1, 2), F(1, 100), F(99, 100), F(0), F(1)):
        for rho in (F(0), F(1, 4), F(9, 10), F(999, 1000)):
            pmfs.append({"responses": m, "mean": txt(mean), "rho": txt(rho), "pmf": [float(x) for x in pmf(m, mean, rho)]})

borrowing = []
for initial, sources in (
    ((1, 1), [(7, 20, F(1, 2))]),
    ((2, 3), [(0, 10, F(1)), (10, 10, F(1, 4))]),
    ((1, 1), []),
    ((1, 1), [(100000, 100000, F(1, 8)), (3, 1000, F(0))]),
):
    a = F(initial[0])
    b = F(initial[1])
    borrowed = F(0)
    for (s, n, disc) in sources:
        a += disc * s
        b += disc * (n - s)
        borrowed += disc * n
    borrowing.append(
        {
            "initial": list(initial),
            "sources": [{"successes": s, "trials": n, "discount": txt(disc)} for (s, n, disc) in sources],
            "alpha": float(a),
            "beta": float(b),
            "effective_sample_size": float(a + b),
            "borrowed_sample_size": float(borrowed),
        }
    )


def floored(row, floor):
    """p_i' = max(floor, c p_i) with the unique c making the sum one."""
    k = len(row)
    assert floor * k <= 1
    order = sorted(range(k), key=lambda i: row[i])
    for pinned in range(k):
        rest = [i for i in order[pinned:]]
        mass = sum(row[i] for i in rest)
        if mass == 0:
            continue
        c = (1 - pinned * floor) / mass
        low = all(c * row[i] <= floor for i in order[:pinned])
        high = all(c * row[i] >= floor for i in rest)
        if low and high:
            out = [max(floor, c * q) for q in row]
            assert sum(out) == 1
            return out
    raise AssertionError(row)


floors = []
for row, floor in (
    ([F(1, 2), F(1, 2)], F(1, 10)),
    ([F(97, 100), F(3, 100)], F(1, 10)),
    ([F(9, 10), F(1, 20), F(1, 20)], F(1, 10)),
    ([F(1, 100), F(1, 100), F(98, 100)], F(1, 5)),
    ([F(7, 10), F(2, 10), F(1, 40), F(1, 40) + F(1, 20)], F(3, 20)),
    ([F(1), F(0), F(0), F(0)], F(1, 8)),
    ([F(1, 5), F(1, 5), F(1, 5), F(1, 5), F(1, 5), F(0)], F(1, 6)),
    ([F(1, 3), F(1, 3), F(1, 3)], F(1, 3)),
):
    assert sum(row) == 1
    out = floored(row, floor)
    floors.append({"row": [txt(x) for x in row], "floor": txt(floor), "result": [float(x) for x in out]})

print(json.dumps({"cases": cases, "pmfs": pmfs, "borrowing": borrowing, "floors": floors}, indent=1))
