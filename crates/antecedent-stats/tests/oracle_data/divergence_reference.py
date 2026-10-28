# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Two-sample and divergence statistics in multiprecision and by exact enumeration.

Regenerate with:  uv run --script divergence_reference.py > divergence_reference.json

* Gaussian KL(N(m0,v0) || N(m1,v1)) = (v0/v1 - 1 - ln(v0/v1) + (m0-m1)^2/v1) / 2 at 100 digits.
* Welch statistic and two-sided p-value: Satterthwaite degrees of freedom, Student-t tail as a
  regularized incomplete beta at 100 digits, from the stored binary64 observations.
* Mann-Whitney: U from mid-ranks, normal approximation with the tie-corrected permutation
  variance.  The closed-form variance is verified here against the exact variance of U over
  every assignment of the pooled values to the two samples (rational arithmetic).
* Gaussian two-segment likelihood ratio n ln v0 - n1 ln v1 - n2 ln v2 and its chi-square(2)
  tail exp(-x/2); for segments below the asymptotic cutoff the exact randomization p-value over
  every partition of the pooled values.
* Biased MMD^2 with the RBF kernel and the median pairwise |difference| bandwidth; exact
  randomization p-value over every partition of the pooled values.
Repository-authored (MIT OR Apache-2.0); mpmath is used only to produce the frozen file.
"""
import json
from fractions import Fraction
from itertools import combinations

import mpmath as mp

mp.mp.dps = 100
_state = 0x2545F4914F6CDD1D


def u01():
    global _state
    _state = (_state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
    return (_state >> 11) / float(1 << 53)


def normal():
    return float(mp.sqrt(-2 * mp.log(mp.mpf(u01()) + mp.mpf(2) ** -60)) * mp.cos(2 * mp.pi * u01()))


def f(x):
    return float(x)


def kl(m0, v0, m1, v1):
    m0, v0, m1, v1 = map(mp.mpf, (m0, v0, m1, v1))
    r = v0 / v1
    return (r - 1 - mp.log(r) + (m0 - m1) ** 2 / v1) / 2


def welch(a, b):
    A = [mp.mpf(x) for x in a]
    B = [mp.mpf(x) for x in b]
    na, nb = len(A), len(B)
    ma, mb = sum(A) / na, sum(B) / nb
    va = sum((x - ma) ** 2 for x in A) / (na - 1)
    vb = sum((x - mb) ** 2 for x in B) / (nb - 1)
    qa, qb = va / na, vb / nb
    d = abs(ma - mb)
    t = d / mp.sqrt(qa + qb)
    df = (qa + qb) ** 2 / (qa**2 / (na - 1) + qb**2 / (nb - 1))
    p = mp.betainc(df / 2, mp.mpf(1) / 2, 0, df / (df + t * t), regularized=True)
    return d, p


def midranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = Fraction(i + 1 + j + 1, 2)
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def mann_whitney(a, b):
    na, nb = len(a), len(b)
    pooled = a + b
    ranks = midranks(pooled)
    ua = sum(ranks[:na]) - Fraction(na * (na + 1), 2)
    auc = ua / (na * nb)
    n = na + nb
    ties = {}
    for x in pooled:
        ties[x] = ties.get(x, 0) + 1
    tie_sum = sum(t**3 - t for t in ties.values())
    var = Fraction(na * nb, 12) * ((n + 1) - Fraction(tie_sum, n * (n - 1)))
    # exact variance of U over all assignments of pooled values to sample a
    us = []
    for idx in combinations(range(n), na):
        us.append(sum(ranks[i] for i in idx) - Fraction(na * (na + 1), 2))
    mean = sum(us) / len(us)
    exact_var = sum((u - mean) ** 2 for u in us) / len(us)
    assert mean == Fraction(na * nb, 2) and exact_var == var
    dev = abs(ua - Fraction(na * nb, 2))
    if dev == 0:
        z = mp.mpf(0)
    else:
        z = mp.mpf(dev.numerator) / dev.denominator / mp.sqrt(mp.mpf(var.numerator) / var.denominator)
    p = mp.erfc(z / mp.sqrt(2))
    return abs(auc - Fraction(1, 2)), p


def segvar(xs):
    m = sum(xs) / len(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)


def lr_stat(left, right):
    L = [mp.mpf(x) for x in left]
    R = [mp.mpf(x) for x in right]
    n1, n2 = len(L), len(R)
    n = n1 + n2
    return n * mp.log(segvar(L + R)) - n1 * mp.log(segvar(L)) - n2 * mp.log(segvar(R))


def lr_exact_p(left, right):
    pooled = left + right
    n1, n = len(left), len(left) + len(right)
    obs = lr_stat(left, right)
    ge = tot = 0
    for idx in combinations(range(n), n1):
        s = set(idx)
        l = [pooled[i] for i in range(n) if i in s]
        r = [pooled[i] for i in range(n) if i not in s]
        if min(segvar([mp.mpf(x) for x in l]), segvar([mp.mpf(x) for x in r])) == 0:
            ge += 1
        elif lr_stat(l, r) >= obs - mp.mpf(10) ** -30:
            ge += 1
        tot += 1
    return ge / tot


def rbf(x, y, h):
    return mp.exp(-((mp.mpf(x) - mp.mpf(y)) / h) ** 2 / 2)


def bandwidth(pooled):
    diffs = sorted(abs(mp.mpf(pooled[i]) - mp.mpf(pooled[j])) for i in range(len(pooled)) for j in range(i + 1, len(pooled)))
    med = diffs[len(diffs) // 2]
    if med > 0:
        return med
    return next((d for d in diffs if d > 0), mp.mpf(1))


def mmd2(a, b, h):
    na, nb = len(a), len(b)
    kxx = sum(rbf(x, y, h) for x in a for y in a) / na**2
    kyy = sum(rbf(x, y, h) for x in b for y in b) / nb**2
    kxy = sum(rbf(x, y, h) for x in a for y in b) / (na * nb)
    return kxx + kyy - 2 * kxy


def mmd_case(a, b):
    pooled = a + b
    h = bandwidth(pooled)
    obs = mmd2(a, b, h)
    n, na = len(pooled), len(a)
    ge = tot = 0
    for idx in combinations(range(n), na):
        s = set(idx)
        l = [pooled[i] for i in range(n) if i in s]
        r = [pooled[i] for i in range(n) if i not in s]
        if mmd2(l, r, h) >= obs - mp.mpf(10) ** -30:
            ge += 1
        tot += 1
    return obs, ge / tot, h


def main():
    out = {}
    kl_cases = []
    for m0, v0, m1, v1 in [
        (0.0, 1.0, 0.0, 1.0),
        (0.3, 1.0, -0.2, 2.5),
        (0.0, 1.0 + 1e-9, 0.0, 1.0),
        (1e-8, 1.0 - 3e-12, 0.0, 1.0),
        (0.0, 1e-6, 0.0, 1.0),
        (0.0, 1e6, 0.0, 1.0),
        (5.0, 0.01, 5.0, 100.0),
        (1.5e154, 1.0, 0.0, 1.0),
        (0.0, 1e-300, 0.0, 1e300),
    ]:
        kl_cases.append({"args": [m0, v0, m1, v1], "kl": f(kl(m0, v0, m1, v1))})
    out["kl"] = kl_cases

    welch_cases = []
    for na, nb, shift, scale in [(2, 2, 0.0, 1.0), (3, 5, 1.0, 2.0), (8, 6, 0.5, 1.0), (40, 35, 0.2, 1.0), (5, 5, 30.0, 0.5), (4, 7, 0.0, 1e-6)]:
        a = [normal() * scale for _ in range(na)]
        b = [normal() + shift for _ in range(nb)]
        d, p = welch(a, b)
        welch_cases.append({"a": a, "b": b, "stat": f(d), "p": f(p)})
    out["welch"] = welch_cases

    mw = []
    for a, b in [
        ([1.0, 2.0, 3.0, 4.0], [3.5, 5.0, 6.0, 7.0, 8.0]),
        ([1.0, 2.0, 2.0, 3.0, 3.0, 3.0], [2.0, 3.0, 3.0, 4.0, 4.0]),
        ([float(round(normal() * 2)) for _ in range(7)], [float(round(normal() * 2 + 1)) for _ in range(8)]),
        ([0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 1.0]),
        ([1.0, 1.0, 1.0], [1.0, 1.0]),
    ]:
        stat, p = mann_whitney(a, b)
        mw.append({"a": a, "b": b, "stat": f(stat), "p": f(p)})
    out["mann_whitney"] = mw

    lr = []
    for na, nb, sh, sc in [(30, 30, 0.0, 1.0), (45, 32, 0.4, 1.8), (60, 60, 3.0, 0.3), (35, 35, 12.0, 6.0)]:
        left = [normal() for _ in range(na)]
        right = [normal() * sc + sh for _ in range(nb)]
        s = lr_stat(left, right)
        lr.append({"left": left, "right": right, "stat": f(s), "p": f(mp.exp(-s / 2))})
    out["lr_asymptotic"] = lr

    lre = []
    for na, nb, sh, sc in [(4, 5, 0.0, 1.0), (5, 5, 2.0, 1.0), (3, 6, 0.0, 4.0), (6, 7, 0.2, 1.0)]:
        left = [normal() for _ in range(na)]
        right = [normal() * sc + sh for _ in range(nb)]
        lre.append({"left": left, "right": right, "stat": f(lr_stat(left, right)), "p_exact": lr_exact_p(left, right)})
    out["lr_exact"] = lre

    mm = []
    for na, nb, sh in [(4, 5, 0.0), (5, 5, 1.5), (3, 7, 0.4), (6, 6, 3.0)]:
        a = [normal() for _ in range(na)]
        b = [normal() + sh for _ in range(nb)]
        obs, pe, h = mmd_case(a, b)
        mm.append({"a": a, "b": b, "mmd2": f(obs), "p_exact": pe, "bandwidth": f(h)})
    # tied observations: the median pairwise distance is zero and the smallest positive one is used
    a, b = [0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 2.0]
    obs, pe, h = mmd_case(a, b)
    mm.append({"a": a, "b": b, "mmd2": f(obs), "p_exact": pe, "bandwidth": f(h)})
    out["mmd"] = mm
    print(json.dumps(out, separators=(",", ":")))


main()
