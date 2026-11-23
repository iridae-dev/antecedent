# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""50-digit reference for the delta-method standard error and clamped Wald interval of the sampled
counterfactual transport estimate on the unconfounded chain X -> M -> Y.

Regenerate with:  uv run --script sampled_transport_delta_reference.py > sampled_transport_delta_reference.json

With the identified functional P(y = 1 | do(x = 1)) = sum_m P(m | x = 1) P(y = 1 | m), written as a
function of the homogeneous cell weights w (X slowest, then M, then Y):
    f(w) = sum_m [ sum_y w(1, m, y) / sum_{m', y} w(1, m', y) ] [ sum_x w(x, m, 1) / sum_{x, y} w(x, m, y) ],
the plug-in is f at the observed cell shares p = counts / N, and the delta-method variance of a
multinomial table of N independent units is
    Var = (1 / N) [ sum_c p_c g_c^2 - ( sum_c p_c g_c )^2 ],    g_c = df / dw_c at w = p,
the gradient taken here by 50-digit numerical differentiation of the closed form.  The Wald interval
is f +- z se with z the 0.975 normal quantile, clamped to [0, 1].  Nothing here calls the product.
Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import json
import random

import mpmath as mp

mp.mp.dps = 50


def f(w):
    def cell(x, m, y):
        return w[x * 4 + m * 2 + y]

    total = mp.mpf(0)
    n_x1 = sum(cell(1, m, y) for m in range(2) for y in range(2))
    for m in range(2):
        p_m_given_x1 = sum(cell(1, m, y) for y in range(2)) / n_x1
        n_m = sum(cell(x, m, y) for x in range(2) for y in range(2))
        p_y1_given_m = sum(cell(x, m, 1) for x in range(2)) / n_m
        total += p_m_given_x1 * p_y1_given_m
    return total


def evaluate(counts):
    n = sum(counts)
    p = [mp.mpf(c) / n for c in counts]
    grad = []
    for c in range(8):
        grad.append(mp.diff(lambda t, c=c: f([p[i] + (t if i == c else 0) for i in range(8)]), 0))
    first = sum(p[c] * grad[c] for c in range(8))
    second = sum(p[c] * grad[c] ** 2 for c in range(8))
    var = (second - first**2) / n
    se = mp.sqrt(var)
    z = mp.sqrt(2) * mp.erfinv(mp.mpf("0.95"))
    point = f(p)
    return point, se, point - z * se, point + z * se


def cases():
    rng = random.Random(20261028)
    out = []
    probs = [0.5 * 0.75 * 0.875, 0.5 * 0.75 * 0.125, 0.5 * 0.25 * 0.125, 0.5 * 0.25 * 0.875]
    for name, n in [("large", 20000), ("moderate", 400), ("small", 64)]:
        while True:
            cells = [0] * 8
            for _ in range(n):
                x = rng.randint(0, 1)
                m = x ^ (rng.random() < 0.25)
                y = m ^ (rng.random() < 0.125)
                cells[x * 4 + int(m) * 2 + int(y)] += 1
            if min(cells) > 0:
                break
        out.append((name, cells))
    # hand-built tables near each boundary so the clamp binds
    out.append(("upper_clamp", [1, 12, 1, 10, 1, 1, 1, 10]))
    out.append(("lower_clamp", [40, 1, 12, 1, 2, 1, 1, 1]))
    return out


def main():
    result = []
    for name, counts in cases():
        point, se, lo, hi = evaluate(counts)
        result.append({"name": name, "counts": counts, "point": float(point), "se": float(se),
                       "lower": float(max(lo, 0)), "upper": float(min(hi, 1)),
                       "lower_clamped": bool(lo < 0), "upper_clamped": bool(hi > 1)})
    print(json.dumps(result, indent=1))


main()
