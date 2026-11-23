# /// script
# requires-python = ">=3.11"
# ///
"""Exact rational reference for the Kaplan-Meier inverse-probability-of-censoring weights.

Regenerate with:  uv run --script observation_ipcw_reference.py > observation_ipcw_reference.json

Rows have a time T_i, an indicator d_i (1 = outcome observed, 0 = censored) and an optional
delayed entry L_i <= T_i.  The risk set at t is {i : L_i <= t <= T_i}.  The censoring survival is
the product-limit  G(t) = prod_{censoring times s <= t} (1 - c(s) / r(s))  with c(s) the number of
censored rows at s and r(s) the risk-set size, and its left limit G(t-) is the product over
censoring times strictly before t (1 if none).  An observed row weighs G(L_i-) / G(T_i-) and a
censored row weighs 0.  The tail restriction is the first censoring time at which G falls below the
floor.  Every quantity is computed in fractions.Fraction from the exact binary64 inputs and
rounded once to binary64.  Also frozen: the Kaplan-Meier survival of the outcome, by the
Satten-Datta identity: with distinct times and no entry, (1/n) sum_i w_i 1{T_i <= t, d_i = 1} = 1 - S(t) at every t.
With tied times the identity is not exact (a tied observed row stays in the censoring risk set
while a tied censoring does not move that row's left limit), so it is frozen for distinct times only.
Repository-authored (MIT OR Apache-2.0).
"""
import json
import random
from fractions import Fraction as F


def risk(times, entries, t):
    return sum(1 for T, L in zip(times, entries) if L <= t <= T)


def censor_survival(times, events, entries):
    steps = []
    g = F(1)
    for s in sorted({T for T, d in zip(times, events) if d == 0}):
        c = sum(1 for T, d in zip(times, events) if d == 0 and T == s)
        r = risk(times, entries, s)
        g *= 1 - F(c, r)
        steps.append((s, g))
    return steps


def left_limit(steps, t):
    g = F(1)
    for s, v in steps:
        if s < t:
            g = v
    return g


def weights(times, events, entries, floor):
    steps = censor_survival(times, events, entries)
    out = []
    for T, d, L in zip(times, events, entries):
        out.append(F(0) if d == 0 else left_limit(steps, L) / left_limit(steps, T))
    tail = next((s for s, v in steps if v < floor), None)
    return out, tail


def km_outcome(times, events):
    s = F(1)
    curve = []
    for t in sorted({T for T, d in zip(times, events) if d == 1}):
        e = sum(1 for T, d in zip(times, events) if d == 1 and T == t)
        r = sum(1 for T in times if T >= t)
        s *= 1 - F(e, r)
        curve.append((t, s))
    return curve


def case(name, seed, n, ties, delayed, floor):
    rng = random.Random(seed)
    times, events, entries = [], [], []
    for _ in range(n):
        T = rng.randint(1, 12) / 2 if ties else rng.uniform(0.5, 6.0)
        times.append(T)
        events.append(1.0 if rng.random() < 0.62 else 0.0)
        entries.append(rng.uniform(0.0, T * 0.5) if delayed else None)
    tfr = [F(T) for T in times]
    # No entry is the entry -10^9: nothing precedes it, so its left limit is 1.
    sentinel = [F(-10**9) if L is None else F(L) for L in entries]
    w, tail = weights(tfr, events, sentinel, floor)
    out = {
        "name": name, "time": times, "event": events,
        "entry": None if not delayed else entries,
        "floor": floor,
        "weights": [float(x) for x in w],
        "tail_restriction": None if tail is None else float(tail),
    }
    if not delayed and not ties:
        curve = km_outcome(tfr, events)
        checks = []
        for t, s in curve:
            lhs = sum(wi for wi, T, d in zip(w, tfr, events) if T <= t and d == 1) / n
            assert lhs == 1 - s, (t, lhs, 1 - s)
            checks.append({"t": float(t), "one_minus_km": float(1 - s)})
        out["satten_datta"] = checks
    return out


def tail_case():
    # Every unit still at risk after time 4 is censored there: the censoring survival reaches 0.
    times = [1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.0, 4.0]
    events = [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    w, tail = weights([F(t) for t in times], events, [F(-10)] * 8, F(1, 100))
    # The rows with event 1 all occur before the tail, so the weights are finite.
    return {
        "name": "administrative_tail", "time": times, "event": events, "entry": None,
        "floor": 0.01, "weights": [float(x) for x in w],
        "tail_restriction": None if tail is None else float(tail),
    }


def main():
    print(json.dumps([
        case("distinct_times", 1, 28, False, False, 0.01),
        case("tied_times", 2, 34, True, False, 0.01),
        case("delayed_entry", 3, 30, True, True, 0.01),
        tail_case(),
    ]))


main()
