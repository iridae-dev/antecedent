# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath"]
# ///
"""Longitudinal regime values from supplied sequential probabilities, in exact rational arithmetic.

Regenerate with:  uv run --script longitudinal_regime_reference.py > longitudinal_regime_reference.json

For subject i with realised treatment a_t, prescribed action d_t, observation indicator o_t,
treatment probability p_t, censoring (stay observed) probability c_t, and action probability
pi_t = p_t if d_t else 1 - p_t:

  matched weight (IPW):  W_i = prod_t 1/(pi_t c_t) over all t when a_t = d_t for every t, else 0;
      score S_i = W_i * Y_i if the terminal outcome is observed and the whole trajectory matches.
      value = mean S; SE^2 = sum (S - value)^2 / (n (n-1)); Kish size of the positive matched
      weights; the maximum matched weight; the smallest pi and c over every cell.
  g-formula: value = (sum of supplied period predictions) / n.
  sequential doubly robust, in its telescoped form with G_t = prod_{s<=t} 1[o_s, a_s = d_s]/(pi_s c_s):
      V = Q_0 + sum_{t=1}^{T-1} G_{t-1} (Q_t - Q_{t-1}) + G_{T-1} (Y - Q_{T-1}),
      value = mean V.
Everything is Fraction arithmetic on the exact binary64 inputs; the standard error is sqrt at 50
digits.  Values are written as nearest binary64.  Repository-authored (MIT OR Apache-2.0).
"""
import json
import random
from fractions import Fraction as F

from mpmath import mp, mpf, sqrt

mp.dps = 50
rng = random.Random(8128)


def make(name, n, T, pfloor, cfloor, y_offset=0.0, always_follow=0.0, action_mode="static1", obs_rate=0.9):
    d, a, p, c, o, q, y, ob = [], [], [], [], [], [], [], []
    for i in range(n):
        seen = True
        hist = []
        prev_a = None
        for t in range(T):
            pt = rng.uniform(pfloor, 1 - pfloor)
            ct = rng.uniform(cfloor, 1.0)
            if action_mode == "static1":
                dt = True
            elif action_mode == "alternate":
                dt = (t % 2 == 0)
            else:  # history dependent: repeat the previous realised action
                dt = True if prev_a is None else prev_a
            at = dt if rng.random() < always_follow else (rng.random() < pt)
            seen = seen and (rng.random() < max(obs_rate, ct))
            d.append(dt); a.append(at); p.append(pt); c.append(ct); o.append(seen)
            q.append(y_offset + rng.gauss(0, 1) + 0.3 * t)
            prev_a = at
        ob.append(o[-1])
        y.append(y_offset + rng.gauss(0, 1) + (1.0 if a[-1] else 0.0) if o[-1] else 0.0)
    case = dict(name=name, n=n, periods=T, floor=min(pfloor, cfloor) / 2, actions=d, treatment=a,
                p=p, c=c, observed_history=o, outcome_observed=ob, outcomes=y, q=q)
    case.update(solve(case))
    return case


def solve(cs):
    n, T = cs["n"], cs["periods"]
    d, a, o, ob = cs["actions"], cs["treatment"], cs["observed_history"], cs["outcome_observed"]
    P, C = [F(v) for v in cs["p"]], [F(v) for v in cs["c"]]
    Y, Q = [F(v) for v in cs["outcomes"]], [F(v) for v in cs["q"]]
    pi = [P[k] if d[k] else 1 - P[k] for k in range(n * T)]
    scores, weights, ipw_total = [], [], F(0)
    for i in range(n):
        follows = all(a[i * T + t] == d[i * T + t] for t in range(T))
        w = F(1)
        if follows:
            for t in range(T):
                w /= pi[i * T + t] * C[i * T + t]
        if follows and ob[i]:
            weights.append(w)
            scores.append(w * Y[i])
        else:
            scores.append(F(0))
    value = sum(scores) / n
    se_ip = None
    if n > 1:
        v = sum((s - value) ** 2 for s in scores) / (n * (n - 1))
        se_ip = float(sqrt(mpf(v.numerator) / mpf(v.denominator)))
    ess = float(sum(weights) ** 2 / sum(w * w for w in weights)) if weights else None
    out = dict(ipw=dict(value=float(value), se=se_ip, ess=ess, matched_fraction=len(weights) / n,
                        max_weight=float(max(weights)) if weights else None,
                        min_pi=float(min(pi)), min_c=float(min(C))))
    gvalue = sum(Q) / n
    out["gformula"] = dict(value=float(gvalue))
    # Sequential doubly robust, telescoped.
    vs = []
    max_w = F(1)
    for i in range(n):
        G = F(1)
        v = Q[i * T]
        for t in range(T):
            j = i * T + t
            G = G * (1 if (o[j] and a[j] == d[j]) else 0) / (pi[j] * C[j])
            if t + 1 < T:
                v += G * (Q[j + 1] - Q[j])
            else:
                v += G * ((Y[i] if ob[i] else F(0)) - Q[j])
        vs.append(v)
        w = F(1)
        for t in range(T):
            j = i * T + t
            if not (o[j] and a[j] == d[j]):
                break
            w /= pi[j] * C[j]
            max_w = max(max_w, w)
    out["sequential"] = dict(value=float(sum(vs) / n), max_weight=float(max_w),
                             observed_fraction=sum(ob) / n, scores=[float(v) for v in vs])
    return out


cases = [
    make("static_two_periods", 60, 2, 0.2, 0.8),
    make("floor_probabilities", 80, 3, 0.02, 0.05, always_follow=0.5),
    make("history_dependent", 70, 4, 0.1, 0.5, y_offset=0.0, action_mode="history"),
    make("outcome_offset_1e8", 60, 3, 0.15, 0.7, y_offset=1e8, always_follow=0.6),
    make("alternating_regime", 50, 3, 0.1, 0.6, action_mode="alternate", always_follow=0.5),
]
print(json.dumps(dict(cases=cases)))
