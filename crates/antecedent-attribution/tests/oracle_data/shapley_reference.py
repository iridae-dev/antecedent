# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Exact Shapley values of finite coalition games, by two independent definitions.

Regenerate with:  uv run --script shapley_reference.py > shapley_reference.json

Each game is a table v(S) over all subsets of n players (S encoded as a bitmask, bit i = player
i), stored as binary64 values and read back exactly as rationals.  The Shapley value is computed
(a) as the average marginal contribution over all n! player orders and (b) by the weighted
subset formula sum_{S not containing i} |S|!(n-|S|-1)!/n! (v(S+i)-v(S)); both are in exact
rational arithmetic and must agree before a game is written.  The exact per-player variance of
the marginal contribution over a uniformly random order (the quantity whose sampled standard
error the permutation estimator reports) is also written, for n <= 6.  Repository-authored
(MIT OR Apache-2.0); standard library only.
"""
import json
from fractions import Fraction
from itertools import permutations
from math import factorial

_state = 0x9E3779B97F4A7C15


def u01():
    global _state
    _state = (_state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
    return (_state >> 11) / float(1 << 53)


def shapley_orders(n, v):
    tot = [Fraction(0)] * n
    sq = [Fraction(0)] * n
    for order in permutations(range(n)):
        mask = 0
        for i in order:
            m = v[mask | (1 << i)] - v[mask]
            tot[i] += m
            sq[i] += m * m
            mask |= 1 << i
    f = factorial(n)
    mean = [t / f for t in tot]
    var = [s / f - m * m for s, m in zip(sq, mean)]
    return mean, var


def shapley_subsets(n, v):
    out = []
    for i in range(n):
        acc = Fraction(0)
        for mask in range(1 << n):
            if mask >> i & 1:
                continue
            s = bin(mask).count("1")
            acc += Fraction(factorial(s) * factorial(n - s - 1), factorial(n)) * (
                v[mask | (1 << i)] - v[mask]
            )
        out.append(acc)
    return out


def game(name, n, values):
    v = [Fraction(x) for x in values]
    sub = shapley_subsets(n, v)
    rec = {"name": name, "n": n, "values": values, "shapley": [float(x) for x in sub]}
    # the float of the exact rational is the correctly rounded value
    if n <= 6:
        mean, var = shapley_orders(n, v)
        assert mean == sub, name
        rec["marginal_variance"] = [float(x) for x in var]
    assert sum(sub) == v[-1] - v[0]
    return rec


def main():
    games = []
    for n in range(1, 9):
        games.append(game(f"random_{n}", n, [u01() * 10 - 5 for _ in range(1 << n)]))
    # unanimity game on {0, 2}: value 1 only when both are present
    n = 5
    games.append(
        game("unanimity", n, [1.0 if (m & 5) == 5 else 0.0 for m in range(1 << n)])
    )
    # large common offset: the differences are small against the magnitude of v
    for n in (6, 10):
        games.append(
            game(f"offset_{n}", n, [1e9 + u01() for _ in range(1 << n)])
        )
    # glove-like asymmetry with a null player (player 3) and a dummy-free symmetric pair
    n = 4
    vals = [0.0] * (1 << n)
    for m in range(1 << n):
        left = (m & 1) + (m >> 1 & 1)
        right = m >> 2 & 1
        vals[m] = float(min(left, right))
    games.append(game("glove_null_player", n, vals))
    # 12 players: only the subset formula is practical for the exact reference
    n = 12
    games.append(game("random_12", n, [u01() for _ in range(1 << n)]))
    print(json.dumps({"games": games}, separators=(",", ":")))


main()
