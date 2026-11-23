"""Exact rational reference for the Bayes action, expected regret (EVPI) and binomial EVSI.

Regenerate with:  python3 decision_exact_reference.py > decision_exact_reference.json

Written from the definitions, with fractions.Fraction throughout (draws, utilities, weights,
thresholds are dyadic so the inputs are exactly the f64 values the product receives):
  admissible set   F = { a : min over constraints of P(constraint | a) >= threshold }
  Bayes action     a* = argmax_{a in F} E[U(a, theta)]  (lowest index on a tie)
  EVPI / regret    E[ max_{a in F} U(a, theta) ] - E[ U(a*, theta) ]
  EVSI(n)          sum_{y=0..n} max_{a in F} E[ U(a, theta) P(y | theta, n) ]  -  max_{a in F} E[U(a, theta)]
with P(y | theta, n) the Binomial(n, theta) mass function and E over the (weighted) draws.
The constraint of every case is "theta <= cutoff_a" evaluated draw by draw.

Repository-authored (MIT OR Apache-2.0); standard library only.
"""
import json
import random
from fractions import Fraction as F
from math import comb


def analyse(thetas, weights, intercepts, slopes, cutoffs, threshold, ns):
    K, A = len(thetas), len(intercepts)
    w = weights or [F(1, K)] * K
    total = sum(w)
    w = [x / total for x in w]
    U = [[intercepts[a] + slopes[a] * thetas[k] for k in range(K)] for a in range(A)]
    if cutoffs is None:
        adm = list(range(A))
    else:
        # satisfaction is the unweighted fraction of draws, as the constraint callback reports it
        sat = [F(sum(1 for t in thetas if t <= cutoffs[a]), K) for a in range(A)]
        adm = [a for a in range(A) if sat[a] >= threshold]
    if not adm:
        return None
    eu = {a: sum(w[k] * U[a][k] for k in range(K)) for a in adm}
    best = adm[0]
    for a in adm:
        if eu[a] > eu[best]:
            best = a
    regret = sum(w[k] * (max(U[a][k] for a in adm) - U[best][k]) for k in range(K))
    evsi = {}
    for n in ns:
        tot = F(0)
        for y in range(n + 1):
            lik = [comb(n, y) * thetas[k] ** y * (1 - thetas[k]) ** (n - y) for k in range(K)]
            tot += max(sum(w[k] * lik[k] * (U[a][k] - U[best][k]) for k in range(K)) for a in adm)
        evsi[str(n)] = float(tot)
    return dict(chosen=best, expected_utility=float(eu[best]), regret=float(regret), evpi=float(regret),
                prior_eu=float(eu[best]), evsi=evsi, admissible=adm,
                satisfaction=None if cutoffs is None else [float(s) for s in sat])


def expected_max(lines, steps=4001):
    """E[max_a (c_a + d_a Z)], Z ~ N(0,1): composite Simpson between the crossings of the lines."""
    import math
    cuts = {-12.0, 12.0}
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            (ci, di), (cj, dj) = lines[i], lines[j]
            if di != dj:
                z = (cj - ci) / (di - dj)
                if -12.0 < z < 12.0:
                    cuts.add(z)
    cuts = sorted(cuts)
    pdf = lambda z: math.exp(-z * z / 2) / math.sqrt(2 * math.pi)
    total = 0.0
    for lo, hi in zip(cuts, cuts[1:]):
        h = (hi - lo) / (steps - 1)
        acc = 0.0
        for k in range(steps):
            z = lo + k * h
            wgt = 1 if k in (0, steps - 1) else (4 if k % 2 else 2)
            acc += wgt * max(c + d * z for c, d in lines) * pdf(z)
        total += acc * h / 3
    return total


def gaussian_cases(rng):
    out = []
    specs = [
        (0.5, 1.0, 12.0, [(0, 0), (0, 1)], [1, 4, 25, 400]),
        (0.0, 1.0, 1.0, [(0, 0), (-0.5, 1)], [1, 3, 50]),
        (2.0, 0.25, 4.0, [(0, 0), (-0.5, 1)], [1, 100]),          # break-even far from the prior mean
        (-1.0, 4.0, 0.5, [(1, -2), (0, 0), (-1, 3)], [1, 2, 9]),  # three actions
        (0.3, 2.0, 3.0, [(0.5, 1), (0.5, 1), (0, 0)], [2, 7]),    # identical lines
        (0.0, 1.0, 9.0, [(5, 0.0), (4, 0.5)], [1, 16]),           # large common offset
        (0.0, 1e-4, 1.0, [(0, 0), (0.01, 1)], [1, 1000]),         # tiny prior spread
    ]
    for mean, tau2, sigma2, lines, ns in specs:
        mu_lines = [(a + b * mean, b) for a, b in lines]
        prior_best = max(a for a, _ in mu_lines)
        ev = {}
        for n in ns:
            s2 = tau2 * (n * tau2) / (n * tau2 + sigma2)
            s = s2 ** 0.5
            ev[str(n)] = expected_max([(a, b * s) for a, b in mu_lines]) - prior_best
        tau = tau2 ** 0.5
        evpi = expected_max([(a, b * tau) for a, b in mu_lines]) - prior_best
        out.append(dict(mean=mean, variance=tau2, noise=sigma2, intercepts=[a for a, _ in lines],
                        slopes=[b for _, b in lines], ns=ns, evsi=ev, evpi=evpi))
    return out


def main():
    rng = random.Random(7)
    cases = []
    for idx in range(120):
        K = rng.randint(1, 7)
        A = rng.randint(2, 4)
        thetas = [F(rng.randint(0, 16), 16) for _ in range(K)]
        offset = rng.choice([0, 0, 1000, 10 ** 6])
        intercepts = [F(rng.randint(-8, 8), 4) + offset for _ in range(A)]
        slopes = [F(rng.randint(-12, 12), 4) for _ in range(A)]
        if idx % 5 == 0:  # exact ties between actions
            intercepts[1] = intercepts[0]
            slopes[1] = slopes[0]
        constrained = idx % 3 == 0
        cutoffs = [F(rng.randint(0, 16), 16) for _ in range(A)] if constrained else None
        threshold = rng.choice([F(1, 2), F(3, 4), F(1), F(1, 4)]) if constrained else F(1)
        weighted = (not constrained) and idx % 2 == 1
        weights = [F(1, 2 ** rng.randint(0, 20)) for _ in range(K)] if weighted else None
        ns = [1, 2, 5, 12, 60]
        out = analyse(thetas, weights, intercepts, slopes, cutoffs, threshold, ns)
        cases.append(dict(thetas=[float(t) for t in thetas], weights=None if weights is None else [float(x) for x in weights],
                          intercepts=[float(x) for x in intercepts], slopes=[float(x) for x in slopes],
                          cutoffs=None if cutoffs is None else [float(c) for c in cutoffs],
                          threshold=float(threshold), ns=ns, expect=out))
    print(json.dumps({"discrete": cases, "gaussian": gaussian_cases(rng)}, separators=(",", ":")))


main()
