# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Exact rational reference for the transported augmented response grid.

Regenerate with:  uv run --script transport_response_grid_reference.py > transport_response_grid_reference.json

For source rows S_i = 1 with selection probability p_i, outcome Y_i, outcome regression m_i at the
observed treatment and a caller-supplied local weight w_{g,i} at grid point g, and target rows
S_i = 0 with outcome regression value mu_{g,i} at the grid point, the augmented transported mean is
    theta_g = (1/n_target) [ sum_{S_i=0} mu_{g,i}
                             + sum_{S_i=1} ((1-p_i)/p_i) w_{g,i} (Y_i - m_i) ]
evaluated in exact rational arithmetic (every binary64 input taken as the exact rational it is).
Source-correction weights are c_{g,i} = ((1-p_i)/p_i) w_{g,i}; the reported effective sample size of
a grid point is (sum |c|)^2 / sum c^2 (0 if all c = 0). The selection overlap is the min and max of
the source selection probabilities and the Kish size (sum o)^2 / sum o^2 of the odds
o_i = (1-p_i)/p_i with the count of odds above 10. Only the final rounding to binary64 is inexact.
Repository-authored (MIT OR Apache-2.0); standard library only.
"""
import json
import random
from fractions import Fraction as F


def reference(trial, p, y, m, grid, mu, w):
    n = len(trial)
    nt = sum(1 for t in trial if not t)
    means, ess = [], []
    for g in range(len(grid)):
        total = F(0)
        c = []
        for i in range(n):
            if trial[i]:
                odds = (1 - F(p[i])) / F(p[i])
                ci = odds * F(w[g * n + i])
                total += ci * (F(y[i]) - F(m[i]))
                c.append(ci)
            else:
                total += F(mu[g * n + i])
        means.append(total / nt)
        s2 = sum(x * x for x in c)
        ess.append(float(sum(abs(x) for x in c) ** 2 / s2) if s2 > 0 else 0.0)
    odds = [(1 - F(p[i])) / F(p[i]) for i in range(n) if trial[i]]
    ps = [p[i] for i in range(n) if trial[i]]
    so = sum(odds)
    s2 = sum(o * o for o in odds)
    sel = {
        "pmin": min(ps),
        "pmax": max(ps),
        "ess": float(so * so / s2) if s2 > 0 else 0.0,
        "extreme": sum(1 for o in odds if o > 10),
    }
    # condition number of the exact sum, to scale the comparison tolerance
    scale = []
    for g in range(len(grid)):
        a = F(0)
        for i in range(n):
            if trial[i]:
                a += abs((1 - F(p[i])) / F(p[i]) * F(w[g * n + i]) * (F(y[i]) - F(m[i])))
            else:
                a += abs(F(mu[g * n + i]))
        scale.append(float(a / nt))
    return [float(x) for x in means], ess, sel, scale


rng = random.Random(20261028)
cases = []


def add(name, trial, p, y, m, grid, mu, w):
    mean, ess, sel, scale = reference(trial, p, y, m, grid, mu, w)
    cases.append(
        dict(
            name=name, trial=trial, p=p, y=y, m=m, grid=grid, mu=mu, w=w,
            mean=mean, ess=ess, sel=sel, scale=scale,
        )
    )


def rnd(lo, hi, k=6):
    return round(rng.uniform(lo, hi), k)


# 1 balanced, ordinary weights
n = 12
trial = [i % 2 == 0 for i in range(n)]
p = [rnd(0.2, 0.8) for _ in range(n)]
y = [rnd(-2, 2) for _ in range(n)]
m = [rnd(-2, 2) for _ in range(n)]
grid = [-1.0, 0.0, 0.5, 2.0]
mu = [rnd(-3, 3) for _ in range(n * 4)]
w = [rnd(0.0, 2.0) for _ in range(n * 4)]
add("balanced", trial, p, y, m, grid, mu, w)

# 2 signed local-linear style weights (negative near the boundary), unbalanced 3 source / 9 target
n = 12
trial = [i < 3 for i in range(n)]
p = [rnd(0.05, 0.95) for _ in range(n)]
y = [rnd(-5, 5) for _ in range(n)]
m = [rnd(-5, 5) for _ in range(n)]
mu = [rnd(-5, 5) for _ in range(n * 3)]
w = [rnd(-1.5, 3.0) for _ in range(n * 3)]
add("signed_weights_unbalanced", trial, p, y, m, [0.0, 1.0, 2.0], mu, w)

# 3 extreme selection probabilities: odds 1e6 and 1e-6, several above the extreme threshold
n = 10
trial = [True, True, True, True, True, False, False, False, False, False]
p = [1e-6, 0.999999, 0.05, 0.5, 0.0909, 0.3, 0.4, 0.5, 0.6, 0.7]
y = [3.0, -1.0, 2.5, 0.25, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
m = [2.9, -0.9, 2.0, 0.3, 1.5, 0.1, 0.2, 0.3, 0.4, 0.5]
mu = [rnd(-1, 1) for _ in range(n * 2)]
w = [rnd(0.1, 1.0) for _ in range(n * 2)]
add("extreme_selection_odds", trial, p, y, m, [0.0, 1.0], mu, w)

# 4 outcome and regression offset 1e8: the correction is a difference of near-equal numbers
n = 10
trial = [i % 3 != 2 for i in range(n)]
p = [rnd(0.2, 0.8) for _ in range(n)]
off = 1.0e8
y = [off + rnd(-1, 1) for _ in range(n)]
m = [off + rnd(-1, 1) for _ in range(n)]
mu = [off + rnd(-1, 1) for _ in range(n * 2)]
w = [rnd(0.0, 2.0) for _ in range(n * 2)]
add("outcome_offset_1e8", trial, p, y, m, [0.0, 1.0], mu, w)

# 5 one source row, many target rows
n = 8
trial = [True] + [False] * 7
p = [0.25] + [0.5] * 7
y = [1.5] + [9.0] * 7
m = [1.0] + [-9.0] * 7
mu = [rnd(-1, 1) for _ in range(n * 2)]
w = [2.0] * (n * 2)
add("single_source_row", trial, p, y, m, [0.0, 1.0], mu, w)

# 6 one target row, many source rows, all ties in the probability
n = 8
trial = [True] * 7 + [False]
p = [0.5] * n
y = [rnd(-1, 1) for _ in range(n)]
m = [rnd(-1, 1) for _ in range(n)]
mu = [rnd(-1, 1) for _ in range(n * 2)]
w = [1.0] * (n * 2)
add("single_target_row_ties", trial, p, y, m, [0.0, 1.0], mu, w)

# 7 weights that vanish at one grid point (all-zero correction: effective size 0) and not at another
n = 6
trial = [True, True, True, False, False, False]
p = [0.4, 0.5, 0.6, 0.3, 0.3, 0.3]
y = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0]
m = [0.5, 1.5, 3.5, 0.0, 0.0, 0.0]
mu = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
w = [0.0] * 6 + [1.0, 0.0, -1.0, 0.0, 0.0, 0.0]
add("vanishing_weights_at_a_grid_point", trial, p, y, m, [0.0, 1.0], mu, w)

# 8 target rows carry outcome/weights that must not enter: large junk values
n = 6
trial = [True, True, False, False, True, False]
p = [0.35, 0.65, 0.5, 0.5, 0.45, 0.5]
y = [1.0, -1.0, 1e12, -1e12, 0.5, 777.0]
m = [0.5, -0.5, 1e12, 0.0, 0.25, 5.0]
mu = [rnd(-1, 1) for _ in range(n)]
w = [1.0, 2.0, 1e9, -1e9, 0.5, 3e8]
add("target_row_outcomes_and_weights_unused", trial, p, y, m, [0.5], mu, w)

print(json.dumps(cases, allow_nan=False))
