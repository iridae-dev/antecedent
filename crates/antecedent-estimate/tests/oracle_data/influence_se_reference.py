# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Exact standard errors of a mean of influence values under each declared dependence.

Regenerate with:  uv run --script influence_se_reference.py > influence_se_reference.json

For influence values psi_i with u_i = psi_i - mean(psi) (n rows), the variance of the mean is
  homoskedastic and HC1:  sum u^2 / (n (n - 1)),   HC0:  sum u^2 / n^2,
  cluster:                (G / (G - 1)) sum_g s_g^2 / n^2, s_g = sum of u over cluster g,
  multiway (Cameron-Gelbach-Miller): sum over nonempty dimension subsets S of
                          (-1)^(|S| + 1) (G_S / (G_S - 1)) sum_{g in cells of S} s_g^2, over n^2,
  Newey-West (Bartlett): [sum u^2 + 2 sum_{k=1}^{L} (1 - k / (L + 1)) sum_t u_t u_{t-k}] / n^2 with
                          L = min(lag, n - 1) for consecutive indices, and for calendar labels the
                          scores sharing a label summed first, L = min(lag, time span) and pairs at
                          calendar gap k,
  panel cluster HAC:      (G / (G - 1)) sum_g [sum_t u_gt^2 + 2 sum_{k=1}^{L} (1 - k/(L+1)) sum_t u_gt u_g,t-k]
                          / n^2, with L = min(lag, widest within-cluster span) and k a calendar gap;
                          lag zero is, by the declared convention, the full cluster form above (every
                          within-cluster cross product at weight one), not the lag-zero Bartlett sum.
A two-way variance from inclusion-exclusion can be negative; it is reported as null (the product
refuses a materially negative one).
Fractions from the exact binary64 values; the square root is taken at the end in 40-digit
arithmetic.  Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import json
from fractions import Fraction as F
from itertools import combinations
import mpmath as mp

mp.mp.dps = 40


def sq(v):
    if v < 0:
        return None  # a negative inclusion-exclusion variance is refused by the product
    return float(mp.sqrt(mp.mpf(v.numerator) / v.denominator))


def demean(psi):
    m = sum(map(F, psi)) / len(psi)
    return [F(p) - m for p in psi]


def cluster_meat(u, labels):
    sums = {}
    for ui, g in zip(u, labels):
        sums[g] = sums.get(g, F(0)) + ui
    return sum(s * s for s in sums.values()), len(sums)


def cluster(u, labels):
    m, g = cluster_meat(u, labels)
    return F(g, g - 1) * m


def multiway(u, dims):
    total = F(0)
    for size in range(1, len(dims) + 1):
        for subset in combinations(range(len(dims)), size):
            cells = list(zip(*[dims[i] for i in subset]))
            m, g = cluster_meat(u, cells)
            total += (-1) ** (size + 1) * F(g, g - 1) * m
    return total


def nw_index(u, lag):
    n = len(u)
    L = min(lag, n - 1)
    meat = sum(x * x for x in u)
    for k in range(1, L + 1):
        w = 1 - F(k, L + 1)
        meat += 2 * w * sum(u[t] * u[t - k] for t in range(k, n))
    return meat


def nw_times(u, times, lag):
    by = {}
    for ui, t in zip(u, times):
        by[t] = by.get(t, F(0)) + ui
    ts = sorted(by)
    L = min(lag, ts[-1] - ts[0])
    meat = sum(v * v for v in by.values())
    for k in range(1, L + 1):
        w = 1 - F(k, L + 1)
        meat += 2 * w * sum(by[t] * by[t - k] for t in ts if (t - k) in by)
    return meat


def panel(u, labels, times, lag):
    groups = {}
    for ui, g, t in zip(u, labels, times):
        groups.setdefault(g, {})[t] = ui
    span = max(max(d) - min(d) for d in groups.values())
    L = min(lag, span)
    meat = F(0)
    for d in groups.values():
        meat += sum(v * v for v in d.values())
        for k in range(1, L + 1):
            w = 1 - F(k, L + 1)
            meat += 2 * w * sum(d[t] * d[t - k] for t in d if (t - k) in d)
    g = len(groups)
    return F(g, g - 1) * meat


psi = [2.5, -1.0, 0.75, 3.25, -2.0, 1.5, 0.25, -0.5, 4.0, -1.75, 0.5, 2.0]
n = len(psi)
u = demean(psi)
den = F(n * n)
cl = [0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 3, 0]
d1 = [0, 1, 0, 1, 2, 2, 0, 1, 2, 0, 1, 2]
d2 = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
d3 = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
# Irregular calendar labels with a repeated label and a gap wider than any lag.
times = [0, 1, 1, 2, 4, 5, 6, 7, 11, 12, 13, 14]
# Panel labels: unique (cluster, time) pairs, with a gap in one cluster.
ptimes = [1, 2, 1, 2, 3, 1, 3, 1, 2, 3, 4, 5]
out = dict(psi=psi, clusters=cl, dim1=d1, dim2=d2, dim3=d3, times=times, panel_times=ptimes, se={})
se = out["se"]
sumsq = sum(x * x for x in u)
se["homoskedastic"] = sq(sumsq / (n * (n - 1)))
se["hc1"] = sq(sumsq / (n * (n - 1)))
se["hc0"] = sq(sumsq / den)
se["cluster"] = sq(cluster(u, cl) / den)
se["multiway2"] = sq(multiway(u, [cl, d1]) / den)
se["multiway3"] = sq(multiway(u, [d1, d2, d3]) / den)
for lag in (0, 1, 3, 20):
    se[f"newey_west_{lag}"] = sq(nw_index(u, lag) / den)
for lag in (0, 2, 5, 30):
    se[f"newey_west_calendar_{lag}"] = sq(nw_times(u, times, lag) / den)
# A cluster-time panel needs unique (cluster, time) pairs.
assert len({(g, t) for g, t in zip(cl, ptimes)}) == n
for lag in (0, 1, 2, 9):
    se[f"panel_hac_{lag}"] = sq((cluster(u, cl) if lag == 0 else panel(u, cl, ptimes, lag)) / den)
print(json.dumps(out, separators=(",", ":")))
