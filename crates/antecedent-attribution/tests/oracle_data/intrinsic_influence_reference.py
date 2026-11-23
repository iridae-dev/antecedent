# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Exact Shapley allocation of Var(E[Y | U_S]) for small structural models.

Regenerate with:  uv run --script intrinsic_influence_reference.py > intrinsic_influence_reference.json

Players are the independent noise terms U_i of the target and its ancestors.  The coalition value
is the definition  v(S) = Var( E[ Y | U_S ] ),  and the contribution of a player is its Shapley
value, the average over all n! orderings of the marginal contribution v(S + i) - v(S), computed
here by enumerating the orderings in Fraction arithmetic.
  * linear models: E[Y | U_S] = const + sum_{i in S} c_i U_i, c_i the total path coefficient of the
    noise of node i on the target (sum over directed paths of products of coefficients, found by
    expanding the structural equations), so v(S) = sum_{i in S} c_i^2 sigma_i^2;
  * the product model Y = b + A B + E with independent A ~ (m_a, s_a^2), B ~ (m_b, s_b^2),
    E ~ (0, s_e^2): E[Y | U_A] = b + m_b A, E[Y | U_A, U_B] = b + A B, so
    v({A}) = m_b^2 s_a^2, v({B}) = m_a^2 s_b^2, v({A, B}) = Var(AB) = m_a^2 s_b^2 + m_b^2 s_a^2 + s_a^2 s_b^2,
    and the independent additive term E adds s_e^2 to every coalition containing it.
Repository-authored (MIT OR Apache-2.0).
"""
import itertools
import json
from fractions import Fraction as F


def shapley(players, v):
    n = len(players)
    total = {p: F(0) for p in players}
    perms = list(itertools.permutations(players))
    for order in perms:
        seen = frozenset()
        for p in order:
            total[p] += v(seen | {p}) - v(seen)
            seen = seen | {p}
    return {p: total[p] / len(perms) for p in players}


def linear_case(name, n, edges, intercept, coef, sigma, target):
    """coef[i] maps parent id -> coefficient (as Fractions); sigma[i] is the noise scale."""
    # total coefficient of the noise of node i on the target
    memo = {}

    def total(node):  # d target / d noise(node)
        if node in memo:
            return memo[node]
        if node == target:
            memo[node] = F(1)
        else:
            memo[node] = sum((coef[c][node] * total(c) for (p, c) in edges if p == node), F(0))
        return memo[node]

    def ancestors(t):
        out, stack = {t}, [t]
        while stack:
            u = stack.pop()
            for p, c in edges:
                if c == u and p not in out:
                    out.add(p)
                    stack.append(p)
        return out

    players = sorted(ancestors(target))
    c2s2 = {p: total(p) ** 2 * sigma[p] ** 2 for p in players}
    phi = shapley(players, lambda S: sum((c2s2[p] for p in S), F(0)))
    return {"name": name, "kind": "linear", "n": n, "edges": edges, "target": target,
            "intercept": [float(x) for x in intercept],
            "coef": [{str(k): float(v) for k, v in d.items()} for d in coef],
            "sigma": [float(s) for s in sigma],
            "truth": {str(p): float(phi[p]) for p in players},
            "total_variance": float(sum(c2s2.values()))}


def product_case(name, ma, sa, mb, sb, se, b):
    ma, sa, mb, sb, se = map(F, (ma, sa, mb, sb, se))
    players = ["A", "B", "E"]

    def v(S):
        out = F(0)
        if "A" in S and "B" in S:
            out += ma ** 2 * sb ** 2 + mb ** 2 * sa ** 2 + sa ** 2 * sb ** 2
        elif "A" in S:
            out += mb ** 2 * sa ** 2
        elif "B" in S:
            out += ma ** 2 * sb ** 2
        if "E" in S:
            out += se ** 2
        return out

    phi = shapley(players, v)
    return {"name": name, "kind": "product", "ma": float(ma), "sa": float(sa), "mb": float(mb), "sb": float(sb),
            "se": float(se), "b": b, "truth": {p: float(phi[p]) for p in players},
            "total_variance": float(v(frozenset(players)))}


def main():
    out = []
    # target 0; root 3 feeds nodes 2 and 1 whose effects on the target cancel exactly; intercepts
    # near 1e7 and noise scales from 2^-10 to 30
    coef = [{2: F(2), 1: F(4)}, {3: F(-3, 8)}, {3: F(3, 4)}, {}]
    # path 3->2->0: (3/4)*2 = 3/2 ; path 3->1->0: (-3/8)*4 = -3/2 ; total zero
    out.append(linear_case("diamond_cancelling_paths", 4, [(3, 2), (3, 1), (2, 0), (1, 0)],
                           [F(10_000_000), F(-3_000_000), F(7), F(2_500_000)], coef,
                           [F(1, 2), F(30), F(1, 1024), F(1)], 0))
    # near-collinear parents: two parents are nearly the same function of one root
    coef2 = [{1: F(5), 2: F(-5)}, {3: F(1)}, {3: F(1)}, {}]
    out.append(linear_case("near_collinear_parents", 4, [(3, 1), (3, 2), (1, 0), (2, 0)],
                           [F(1_000_000), F(0), F(0), F(0)], coef2,
                           [F(1, 8), F(1, 1000), F(1, 1000), F(2)], 0))
    # deep chain with amplification
    coef3 = [{1: F(3)}, {2: F(3)}, {3: F(3)}, {}]
    out.append(linear_case("amplifying_chain", 4, [(3, 2), (2, 1), (1, 0)],
                           [F(0), F(1), F(2), F(3)], coef3, [F(1, 4), F(1, 2), F(1), F(2)], 0))
    out.append(product_case("asymmetric_product", F(2), F(1), F(1, 2), F(3, 2), F(1, 2), 100.0))
    out.append(product_case("offset_product", F(40), F(1, 2), F(-30), F(1, 4), F(1, 8), -50.0))
    print(json.dumps(out, indent=1))


main()
