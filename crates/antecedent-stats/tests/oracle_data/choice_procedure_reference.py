# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Reference values for the exact law of sequential categorical-choice procedures.

Regenerate with:  uv run --script choice_procedure_reference.py > choice_procedure_reference.json

Written from the definition of the procedure, by a different algorithm from the product's
unit-by-unit dynamic program: the law of the counts after each look is built from the closed-form
probability of a whole batch (multinomial for a fixed truth, Dirichlet-multinomial for the prior
predictive), in exact rational arithmetic, and the live mass is convolved batch by batch. The
posterior probability that each option has the largest share and the posterior expected loss at
every reachable count vector are mpmath adaptive quadrature of the independent-gamma
representation (as in dirichlet_choice_reference.py).

  stop at look j in state c   iff   n_j >= min_observations
                                    and  max_k P(k best | c) >= p_min
                                    and  loss_{argmax}(c) <= loss_max
  the option selected is argmax_k P(k best | c).

Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import json
from fractions import Fraction
from itertools import product
from math import factorial

import mpmath as mp

mp.mp.dps = 20
mpf = mp.mpf


def pdf(a, x):
    return mp.exp((a - 1) * mp.log(x) - x - mp.loggamma(a)) if x > 0 else mpf(0)


def cdf(a, x):
    return mp.gammainc(a, 0, x, regularized=True) if x > 0 else mpf(0)


def breakpoints(alphas):
    pts = {mpf(0)}
    for a in alphas:
        sd = mp.sqrt(a)
        for k in (-6, -3, -1, 0, 1, 3, 6, 12):
            x = a + k * sd
            if x > 0:
                pts.add(x)
    pts.add(max(a + 40 * mp.sqrt(a) + 40 for a in alphas))
    return sorted(pts)


def best_and_loss(alphas):
    total = sum(alphas)
    best = []
    for k in range(len(alphas)):
        def f(x, k=k):
            v = pdf(alphas[k], x)
            for j, a in enumerate(alphas):
                if j != k:
                    v *= cdf(a, x)
            return v
        best.append(mp.quad(f, breakpoints(alphas)))
    emax = mp.quad(lambda x: 1 - mp.fprod(cdf(a, x) for a in alphas), breakpoints(alphas))
    share = emax / total
    return best, [share - a / total for a in alphas]


CACHE = {}


def assess(prior, counts):
    key = tuple(counts)
    if key not in CACHE:
        CACHE[key] = best_and_loss([mpf(a) + c for a, c in zip(prior, counts)])
    return CACHE[key]


def compositions(n, k):
    if k == 1:
        yield (n,)
        return
    for first in range(n + 1):
        for rest in compositions(n - first, k - 1):
            yield (first,) + rest


def batch_probability(kernel, counts_before, batch_counts):
    n = sum(batch_counts)
    k = len(batch_counts)
    coef = Fraction(factorial(n))
    for c in batch_counts:
        coef /= factorial(c)
    if kernel["kind"] == "fixed":
        p = coef
        for c, t in zip(batch_counts, kernel["theta"]):
            p *= Fraction(t) ** c
        return p
    alpha = [Fraction(a) for a in kernel["alpha"]]
    a0 = sum(alpha)
    seen = sum(counts_before)
    p = coef
    for a, before, c in zip(alpha, counts_before, batch_counts):
        for j in range(c):
            p *= (a + before + j)
    for j in range(n):
        p /= (a0 + seen + j)
    # the product above is the ordered-sequence probability times the multinomial coefficient
    return p


def run(case):
    prior = case["prior"]
    k = len(prior)
    rule = case["rule"]
    live = {tuple([0] * k): Fraction(1)}
    seen_total = 0
    atoms = []
    for look, batch in enumerate(case["batches"]):
        seen_total += batch
        nxt = {}
        for before, mass in live.items():
            for add in compositions(batch, k):
                p = batch_probability(case["kernel"], before, add)
                if p == 0:
                    continue
                after = tuple(b + a for b, a in zip(before, add))
                nxt[after] = nxt.get(after, Fraction(0)) + mass * p
        last = look + 1 == len(case["batches"])
        live = {}
        for counts, mass in nxt.items():
            best, loss = assess(prior, counts)
            top = max(range(k), key=lambda i: best[i])
            stop = (seen_total >= rule["min_observations"]
                    and best[top] >= mpf(Fraction(rule["min_probability_best"]).numerator) / Fraction(rule["min_probability_best"]).denominator
                    and loss[top] <= mpf(rule["max_expected_loss"]))
            if stop or last:
                atoms.append((look, stop, counts, top if stop else None, mass))
            else:
                live[counts] = mass
    selection = [Fraction(0)] * k
    undecided = Fraction(0)
    expected = Fraction(0)
    stops = [Fraction(0)] * len(case["batches"])
    cumulative = []
    t = 0
    for b in case["batches"]:
        t += b
        cumulative.append(t)
    for look, stop, counts, decision, mass in atoms:
        expected += cumulative[look] * mass
        if decision is None:
            undecided += mass
        else:
            selection[decision] += mass
        if stop:
            stops[look] += mass
    def f(x):
        return mp.nstr(mpf(x.numerator) / x.denominator, 17)
    return {
        "name": case["name"],
        "selection": [f(x) for x in selection],
        "undecided": f(undecided),
        "expected_enrolled": f(expected),
        "rule_stops": [f(x) for x in stops],
    }


CASES = [
    {"name": "two_options_fixed_clear", "prior": [1, 1], "batches": [4, 4, 4, 4],
     "rule": {"min_probability_best": "17/20", "max_expected_loss": "1", "min_observations": 4},
     "kernel": {"kind": "fixed", "theta": ["7/10", "3/10"]}},
    {"name": "two_options_fixed_tied", "prior": [1, 1], "batches": [4, 4, 4, 4],
     "rule": {"min_probability_best": "17/20", "max_expected_loss": "1", "min_observations": 4},
     "kernel": {"kind": "fixed", "theta": ["1/2", "1/2"]}},
    {"name": "two_options_predictive", "prior": [1, 1], "batches": [3, 3, 3, 3, 3],
     "rule": {"min_probability_best": "9/10", "max_expected_loss": "1", "min_observations": 6},
     "kernel": {"kind": "predictive", "alpha": [1, 1]}},
    {"name": "three_options_fixed_loss_binding", "prior": [1, 1, 1], "batches": [5, 5, 5],
     "rule": {"min_probability_best": "4/5", "max_expected_loss": "0.12", "min_observations": 5},
     "kernel": {"kind": "fixed", "theta": ["3/5", "1/4", "3/20"]}},
    {"name": "three_options_predictive_informative_prior", "prior": [2, 2, 2], "batches": [4, 4, 4],
     "rule": {"min_probability_best": "3/4", "max_expected_loss": "1", "min_observations": 4},
     "kernel": {"kind": "predictive", "alpha": [2, 2, 2]}},
]

print(json.dumps({"cases": [run(c) for c in CASES]}, indent=1))
