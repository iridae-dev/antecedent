# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Exact-rational Kaplan-Meier, Aalen-Johansen and restricted-mean references.

Regenerate with:  python3 product_limit_reference.py > product_limit_reference.json

Written from the definitions, in fractions.Fraction arithmetic with no rounding:
  risk set at event time t (left truncation):   R(t) = #{i : entry_i < t <= exit_i}
  Kaplan-Meier:     S(t) = prod_{t_j <= t} (1 - d_j / R(t_j)),   d_j = events at t_j (any cause)
  Aalen-Johansen:   F_c(t) = sum_{t_j <= t} S(t_j-) * d_{c,j} / R(t_j)
  RMST to tau:      integral_0^tau S(t) dt  (exact step integral)
Event-time ties, censoring tied with events, delayed entry, one-sided late entry, events exactly
at tau and censored subjects that leave before tau are generated from a fixed integer seed
(simple LCG; no library generator).  Times are small multiples of 1/4 so binary64 holds them
exactly.

Repository-authored (MIT OR Apache-2.0).
"""
import json
from fractions import Fraction as F


class Lcg:
    def __init__(self, seed):
        self.s = seed

    def next(self, m):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        return (self.s >> 33) % m


def curves(duration, code, treated, entry, tau, arm, cause):
    idx = [i for i in range(len(duration)) if treated[i] == arm]
    ev = sorted({duration[i] for i in idx if code[i] > 0 and duration[i] <= tau})
    surv, inc, prev, rmst = F(1), F(0), F(0), F(0)
    times, svals, ivals, minrisk = [F(0)], [F(1)], [F(0)], None
    for t in ev:
        rmst += (t - prev) * surv
        risk = sum(1 for i in idx if duration[i] >= t and (entry is None or entry[i] < t))
        d_all = sum(1 for i in idx if duration[i] == t and code[i] > 0)
        d_c = sum(1 for i in idx if duration[i] == t and code[i] == cause)
        inc += surv * F(d_c, risk)
        surv *= 1 - F(d_all, risk)
        times.append(t); svals.append(surv); ivals.append(inc)
        prev = t
        minrisk = risk if minrisk is None else min(minrisk, risk)
    rmst += (tau - prev) * surv
    return times, svals, ivals, rmst, minrisk


def at(times, vals, t):
    k = max(j for j, x in enumerate(times) if x <= t)
    return vals[k]


cases = []
for seed in range(1, 41):
    g = Lcg(seed)
    n = 14 + g.next(30)
    competing = seed % 2 == 0
    delayed = seed % 3 == 0
    duration, code, treated, entry = [], [], [], []
    for i in range(n):
        arm = i % 2 == 0 if seed % 5 else g.next(2) == 0
        e = F(0) if (not delayed or g.next(4) == 0) else F(1 + g.next(6), 4)
        d = e + F(1 + g.next(24), 4)
        c = 0 if g.next(3) == 0 else (1 + g.next(2) if competing else 1)
        duration.append(d); code.append(c); treated.append(arm); entry.append(e)
    if not any(t for t in treated) or all(treated):
        continue
    tau = F(1 + g.next(20), 4)
    if any(all(duration[i] < tau for i in range(n) if treated[i] == a) for a in (False, True)):
        continue
    if competing and len({c for c in code if c > 0}) < 2:
        continue
    if delayed and any(all(entry[i] != 0 for i in range(n) if treated[i] == a) for a in (False, True)):
        continue
    arms = [curves(duration, code, treated, entry if delayed else None, tau, a, 1) for a in (False, True)]
    grid = sorted({x for a in arms for x in a[0]} | {tau})
    out = {
        "seed": seed, "tau": float(tau), "competing": competing, "delayed": delayed,
        "duration": [float(x) for x in duration], "event_code": code, "treated": treated,
        "entry": [float(x) for x in entry] if delayed else None,
        "times": [float(t) for t in grid],
        "control": [float(at(arms[0][0], arms[0][2] if competing else arms[0][1], t)) for t in grid],
        "treated_curve": [float(at(arms[1][0], arms[1][2] if competing else arms[1][1], t)) for t in grid],
        "rmst_control": None if competing else float(arms[0][3]),
        "rmst_treated": None if competing else float(arms[1][3]),
        "minimum_event_risk_set": min([x for x in (arms[0][4], arms[1][4]) if x is not None], default=None),
    }
    cases.append(out)
print(json.dumps(cases))
