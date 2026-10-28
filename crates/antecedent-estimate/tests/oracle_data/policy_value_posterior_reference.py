# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Exact laws of the Bayesian-bootstrap fixed-policy value and its paired difference.

Regenerate with:  uv run --script policy_value_posterior_reference.py > policy_value_posterior_reference.json

Binary augmented row score of action a (written from the estimand):
    s_i(a) = m_a(i) + 1{T_i = a} (y_i - m_a(i)) / pi_a(i) - cost_i 1{a_i},   pi_1 = e, pi_0 = 1 - e,
for the policy actions a_i, the reference actions r_i and their costs, and the paired difference
d_i = s_i(policy) - s_i(reference).  The posterior of a policy value is the Dirichlet(1, ..., 1)
weighted mean S = sum_i omega_i s_i, whose law for distinct scores is
    P(S <= x) = 1 - sum_i (s_i - x)_+^(n-1) / prod_{j != i} (s_i - s_j)
(B-spline law; for n = 2, S is uniform between the two scores).  The three columns share their
weights, so the law of the difference is the same formula applied to d_i.  The nuisance
predictions are distinct per row and the reference is the complement of the policy, so each
column has distinct scores.  Everything is Fraction arithmetic on the exact dyadic inputs.
Repository-authored (MIT OR Apache-2.0).
"""
import json
from fractions import Fraction as F


def cdf(c, x):
    n = len(c)
    total = F(0)
    for i, ci in enumerate(c):
        if ci > x:
            den = F(1)
            for j, cj in enumerate(c):
                if j != i:
                    den *= ci - cj
            total += (ci - x) ** (n - 1) / den
    return 1 - total


def scores(y, t, e, act, cost, m0, m1):
    out = []
    for i in range(len(y)):
        a = act[i]
        m = m1[i] if a else m0[i]
        pi = F(e[i]) if a else 1 - F(e[i])
        out.append(F(m) + (F(y[i]) - F(m) if t[i] == a else 0) / pi - F(cost) * a)
    return out


y = [2.5, 0.75, 3.25, 1.5, 4.0, 0.25, 2.0, 3.5, 1.25]
t = [True, False, True, True, False, False, True, False, True]
e = [0.25, 0.5, 0.75, 0.375, 0.625, 0.5, 0.25, 0.875, 0.5]
act = [True, True, False, True, False, True, False, False, True]
ref = [not a for a in act]
m0 = [v / 64 for v in (31, 47, 59, 13, 97, 7, 71, 83, 41)]
m1 = [v + 33 / 64 for v in m0]
ps = scores(y, t, e, act, 0.25, m0, m1)
rs = scores(y, t, e, ref, 0.125, m0, m1)
ds = [a - b for a, b in zip(ps, rs)]
n = len(y)
out = dict(y=y, assigned=t, propensity=e, actions=act, reference_actions=ref, cost=0.25, reference_cost=0.125, mu0=m0, mu1=m1)
for name, col in (("policy", ps), ("reference", rs), ("incremental", ds)):
    assert len(set(col)) == n
    mu = sum(col) / n
    grid = [mu + F(k, 4) for k in range(-6, 7)]
    out[name] = dict(mean=float(mu), grid=[float(x) for x in grid], cdf=[float(cdf(col, x)) for x in grid])
print(json.dumps(out, separators=(",", ":")))
