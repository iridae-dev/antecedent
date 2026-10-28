# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath"]
# ///
"""Fixed-policy value: inverse-probability, augmented, multi-action, uplift bins and finite-class regret.

Regenerate with:  uv run --script policy_value_reference.py > policy_value_reference.json

Every input is a binary64-exact dyadic rational (outcomes, propensities, costs, nuisance
predictions), so each score, mean and variance is exact `fractions.Fraction` arithmetic; only the
square root of a standard error and the normal quantile (mpmath, 50 digits) are not rational.
Definitions written from the estimand, not from the product:
  binary row score of action a: m_a + 1{T = a} (y - m_a) / pi_a   (m_a = 0 without nuisances;
    pi_1 = e, pi_0 = 1 - e), net of the row cost c * 1{a}; value = row mean; the paired
    difference scores policy - reference; SE^2 = sum (s - mean)^2 / (n (n - 1)).
  multi-action HT row score of action a: 1{T = a} y / pi_{row, T}, net of cost[a].
  CATE stratum, action a against control: scores 1{T=a} y / pi_a - 1{T=0} y / pi_0 over the rows of
    the stratum, mean and SE as above.
  uplift bin: scores y / e (treated), -y / (1 - e) (control), per bin; the singleton-bin
    function reports zero error, the interval function refuses a singleton.
  finite-class regret: candidate HT scores net of cost 1{a} c, the paired contrast with the selected
    candidate, regret = max(0, max_j d_j), simultaneous half-width z(1 - 0.05 / (2 (k - 1))) SE_j,
    lower = max(0, max_j (d_j - span_j)), upper = max(0, max_j (d_j + span_j)).
Values are written as nearest binary64.  Repository-authored (MIT OR Apache-2.0).
"""
import json
from fractions import Fraction as F
from mpmath import mp, mpf, sqrt, erfinv

mp.dps = 50


def q(v):
    return mpf(v.numerator) / mpf(v.denominator)


def fl(v):
    return float(v) if isinstance(v, F) else float(mpf(v))


def mean(v):
    return sum(v, F(0)) / len(v)


def se(v):
    m = mean(v)
    return sqrt(q(sum((x - m) ** 2 for x in v) / (len(v) * (len(v) - 1))))


def z(p):
    return mp.sqrt(2) * erfinv(2 * mpf(p) - 1)


def at(v, i):
    return v[0] if len(v) == 1 else v[i]


def binary_scores(y, t, e, act, cost, m0=None, m1=None):
    out = []
    for i in range(len(y)):
        a = act[i]
        m = (m1[i] if a else m0[i]) if m0 is not None else F(0)
        pi = at(e, i) if a else 1 - at(e, i)
        s = m + (F(y[i]) - m if t[i] == a else 0) / pi
        out.append(s - at(cost, i) * a)
    return out


cases = []


def binary_case(name, y, t, e, act, ref, cost, rcost, m0=None, m1=None):
    ps = binary_scores(y, t, e, act, cost, m0, m1)
    rs = binary_scores(y, t, e, ref, rcost, m0, m1)
    ds = [a - b for a, b in zip(ps, rs)]
    props = [at(e, i) for i in range(len(y))]
    cases.append({
        "kind": "binary", "name": name, "y": [float(v) for v in y], "t": [bool(v) for v in t],
        "e": [float(v) for v in e], "actions": [bool(v) for v in act], "reference": [bool(v) for v in ref],
        "costs": [float(v) for v in cost], "reference_costs": [float(v) for v in rcost],
        "mu0": None if m0 is None else [float(v) for v in m0],
        "mu1": None if m1 is None else [float(v) for v in m1],
        "policy_value": fl(mean(ps)), "reference_value": fl(mean(rs)), "incremental_value": fl(mean(ds)),
        "policy_se": fl(se(ps)), "reference_se": fl(se(rs)), "incremental_se": fl(se(ds)),
        "treatment_rate": sum(act) / len(y),
        "total_cost": fl(sum(at(cost, i) * act[i] for i in range(len(y)))),
        "propensity_min": float(min(props)), "propensity_max": float(max(props)),
        "ipw_policy_net": fl(mean([s for s in binary_scores(y, t, e, act, [F(0)])]) - sum(at(cost, i) * act[i] for i in range(len(y))) / len(y)),
        "ipw_reference_net": fl(mean(binary_scores(y, t, e, ref, [F(0)])) - sum(at(rcost, i) * ref[i] for i in range(len(y))) / len(y)),
    })


n = 14
y = [F(v, 4) for v in [9, 23, -5, 14, 30, 2, 11, 17, 8, 26, -3, 19, 5, 12]]
t = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0]
act = [1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0]
ref = [0] * n
e = [F(1, 2)]
m0 = [F(v, 4) for v in [8, 20, 1, 12, 28, 4, 10, 16, 6, 22, 0, 18, 7, 13]]
m1 = [v + F(3, 2) for v in m0]
binary_case("scalar_propensity_scalar_cost", y, t, e, act, ref, [F(1, 4)], [F(1, 4)], m0, m1)
binary_case("scalar_propensity_ipw_only", y, t, e, act, ref, [F(1, 4)], [F(1, 4)])
# extreme row propensities and row-aligned costs, a non-trivial reference
er = [F(1, 1024), F(1023, 1024), F(1, 2), F(1, 8), F(7, 8), F(1, 4), F(3, 4), F(1, 1024), F(1, 2), F(1023, 1024), F(1, 8), F(7, 8), F(1, 4), F(3, 4)]
cost = [F(v, 8) for v in [1, 2, 3, 1, 0, 2, 1, 4, 1, 2, 0, 3, 1, 2]]
rcost = [F(v, 8) for v in [2, 1, 1, 0, 3, 1, 2, 1, 0, 2, 1, 1, 3, 0]]
ref2 = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
binary_case("extreme_propensities_row_costs", y, t, er, act, ref2, cost, rcost, m0, m1)
binary_case("extreme_propensities_ipw_only", y, t, er, act, ref2, cost, rcost)
# outcome offset 1e7 (exact), all policies the same as assignment
yo = [F(10_000_000) + v for v in y]
binary_case("outcome_offset_1e7", yo, t, e, act, ref, [F(1, 4)], [F(1, 4)], [F(10_000_000) + v for v in m0], [F(10_000_000) + v + F(3, 2) for v in m0])
# two rows
binary_case("two_rows", [F(1), F(3, 2)], [1, 0], [F(1, 4)], [1, 1], [0, 0], [F(1, 8)], [F(1, 8)])

# multi-action, k = 3
rows = 12
labels = ["control", "a", "b"]
probs = []
base = [(F(1, 2), F(1, 4), F(1, 4)), (F(1, 4), F(1, 2), F(1, 4)), (F(1, 8), F(1, 8), F(3, 4)), (F(3, 4), F(1, 8), F(1, 8))]
for i in range(rows):
    probs.append(base[i % 4])
ym = [F(v, 4) for v in [7, 15, -2, 20, 9, 11, 3, 25, 6, 14, 1, 18]]
asg = [0, 1, 2, 1, 0, 2, 2, 1, 0, 0, 2, 1]
pol = [1, 2, 0, 1, 2, 0, 2, 1, 0, 2, 1, 0]
rf = [0, 1, 1, 0, 2, 0, 1, 2, 0, 1, 0, 2]
mcost = [F(0), F(1, 4), F(1, 2)]
mrcost = [F(0), F(1, 8), F(1, 4)]


def ht(policy, c):
    return [(ym[i] / probs[i][asg[i]] if asg[i] == policy[i] else F(0)) - c[policy[i]] for i in range(rows)]


mp_ = ht(pol, mcost)
mr_ = ht(rf, mrcost)
md_ = [a - b for a, b in zip(mp_, mr_)]
groups = ["g1"] * 6 + ["g2"] * 6
cate = []
for g in ["g1", "g2"]:
    idx = [i for i in range(rows) if groups[i] == g]
    for a in (1, 2):
        sc = [(ym[i] / probs[i][a] if asg[i] == a else (-ym[i] / probs[i][0] if asg[i] == 0 else F(0))) for i in idx]
        cate.append({"group": g, "action": labels[a], "effect": fl(mean(sc)), "se": fl(se(sc)),
                     "rows": len(idx), "action_rows": sum(asg[i] == a for i in idx),
                     "control_rows": sum(asg[i] == 0 for i in idx)})
cases.append({
    "kind": "multi", "name": "three_actions_two_strata", "y": [float(v) for v in ym], "assignment": asg,
    "actions": pol, "reference": rf, "propensities": [float(p) for r in probs for p in r],
    "costs": [float(c) for c in mcost], "reference_costs": [float(c) for c in mrcost],
    "groups": groups, "labels": labels,
    "policy_value": fl(mean(mp_)), "reference_value": fl(mean(mr_)), "incremental_value": fl(mean(md_)),
    "policy_se": fl(se(mp_)), "reference_se": fl(se(mr_)), "incremental_se": fl(se(md_)),
    "non_control_rate": sum(a != 0 for a in pol) / rows,
    "policy_cost": fl(sum(mcost[a] for a in pol)),
    "propensity_min": float(min(p for r in probs for p in r)), "propensity_max": float(max(p for r in probs for p in r)),
    "ipw_policy": fl(mean(mp_)), "ipw_reference": fl(mean(mr_)), "cate": cate,
})

# uplift bins
bins = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1]
ub = []
for b in range(3):
    sc = [(y[i] / at(er, i) if t[i] else -y[i] / (1 - at(er, i))) for i in range(n) if bins[i] == b]
    ub.append({"effect": fl(mean(sc)), "se": fl(se(sc)), "rows": len(sc)})
cases.append({"kind": "uplift", "name": "three_bins_extreme_propensities", "y": [float(v) for v in y],
              "t": [bool(v) for v in t], "e": [float(v) for v in er], "bins": bins, "bins_expected": ub})
single = [0] * 13 + [1]
sc1 = [(y[i] / at(er, i) if t[i] else -y[i] / (1 - at(er, i))) for i in range(13)]
cases.append({"kind": "uplift_singleton", "name": "singleton_bin", "y": [float(v) for v in y],
              "t": [bool(v) for v in t], "e": [float(v) for v in er], "bins": single,
              "first_effect": fl(mean(sc1)), "first_se": fl(se(sc1)),
              "second_effect": fl(y[13] / at(er, 13) if t[13] else -y[13] / (1 - at(er, 13)))})

# finite-class regret: 400 rows, k = 3
N = 400
yr = [F((37 * i * i + 11 * i) % 101 - 30, 4) for i in range(N)]
tr = [(i * 7 + i // 3) % 5 < 2 for i in range(N)]
pr = [F(1, 4) if i % 3 == 0 else (F(1, 2) if i % 3 == 1 else F(3, 4)) for i in range(N)]
cands = [[(i % 2 == 0) for i in range(N)], [(i % 3 != 0) for i in range(N)], [(i % 5 < 3) for i in range(N)]]
ccost = [F(1, 4)] * N
sc = []
for c in cands:
    sc.append([(yr[i] / (pr[i] if c[i] else 1 - pr[i]) if tr[i] == c[i] else F(0)) - ccost[i] * c[i] for i in range(N)])
means = [mean(s) for s in sc]
sel = 1
k = len(cands)
crit = z(1 - mpf("0.05") / (2 * (k - 1)))
lo = mpf(0); up = mpf(0); reg = F(0); ses = []
for j in range(k):
    if j == sel:
        ses.append(0.0); continue
    d = [a - b for a, b in zip(sc[j], sc[sel])]
    dm = means[j] - means[sel]
    s = se(d)
    ses.append(float(s))
    reg = max(reg, dm)
    lo = max(lo, q(dm) - crit * s); up = max(up, q(dm) + crit * s)
cases.append({"kind": "regret", "name": "three_candidates", "y": [float(v) for v in yr], "t": tr,
              "e": [float(v) for v in pr], "candidates": cands, "selected": sel, "costs": [0.25],
              "values": [fl(m) for m in means], "regret": fl(reg), "ses": ses,
              "interval": [float(lo), float(up)]})
print(json.dumps({"cases": cases}))
