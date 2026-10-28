# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath"]
# ///
"""Horvitz-Thompson excursion designs with cluster-robust ratio inference, in exact arithmetic.

Regenerate with:  uv run --script temporal_design_excursion_reference.py > temporal_design_excursion_reference.json

A decision row (unit, period t) is paired with the row of the same unit at t + lag when it is observed;
unpaired decisions are dropped.  For the pair with outcome Y, the arm-mean score of arm a is
1{A_t = a} Y / P(A_t = a), a contrast is the difference of arm-mean scores, and a coordinate is the
ratio  theta = sum_g S_g / sum_g n_g  of cluster sums of pair scores.  The covariance is the
linearised-ratio sandwich  G / (G - 1) sum_g r_g r_g' / (sum_g n_g)^2  with r_g = S_g - n_g theta, the
standard error its square root, the interval theta -/+ 1.959963984540054 se, and the Holm step-down
adjusted p-values of the family are formed from the two-sided normal p-values of theta / se.
Designs: two arms, four ordered arms (reference contrasts, adjacent steps, arm means), a two-factor
factorial (main effects, interaction, cell means) and within-cluster interference (direct, spillover and
total effect; the peer-exposure probability is prod_j (1 - p_j) over the other members observed that
period).  Arithmetic in fractions.Fraction; square roots and normal tails at 50 digits (mpmath).
Repository-authored (MIT OR Apache-2.0); mpmath is BSD-3-Clause.
"""
import json
from fractions import Fraction as F

from mpmath import mp, mpf, sqrt, erfc

mp.dps = 50
Z975 = mpf("1.959963984540054")
_state = 0x6C0356F2E8B3C6A9


def u01():
    global _state
    _state = (_state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
    return (_state >> 11) / float(1 << 53)


def mp_of(q):
    return mpf(q.numerator) / q.denominator


def summarize(pair_scores, clusters, coords):
    cl = sorted(set(clusters))
    G = len(cl)
    k = coords
    S = {c: [F(0)] * k for c in cl}
    n = {c: 0 for c in cl}
    for sc, c in zip(pair_scores, clusters):
        n[c] += 1
        S[c] = [a + b for a, b in zip(S[c], sc)]
    total = sum(n.values())
    theta = [sum(S[c][j] for c in cl) / total for j in range(k)]
    cov = [[F(0)] * k for _ in range(k)]
    for c in cl:
        r = [S[c][j] - n[c] * theta[j] for j in range(k)]
        for a in range(k):
            for b in range(k):
                cov[a][b] += r[a] * r[b]
    scale = F(G, G - 1) / F(total * total)
    cov = [[v * scale for v in row] for row in cov]
    se = [sqrt(mp_of(cov[j][j])) for j in range(k)]
    return theta, cov, se, G


def holm(theta, se, family):
    raw = [erfc(abs(mp_of(theta[j]) / se[j]) / sqrt(2)) for j in family]
    order = sorted(range(len(raw)), key=lambda i: raw[i])
    adj = [None] * len(raw)
    running = mpf(0)
    m = len(raw)
    for rank, i in enumerate(order):
        running = max(running, min(mpf(1), (m - rank) * raw[i]))
        adj[i] = running
    return [float(v) for v in adj]


def emit(name, design, rows, lag, scores_fn, family, clusters_of):
    index = {(r["unit"], r["period"]): i for i, r in enumerate(rows)}
    pairs = [(i, index[(r["unit"], r["period"] + lag)]) for i, r in enumerate(rows)
             if (r["unit"], r["period"] + lag) in index]
    unpaired = len(rows) - len(pairs)
    pair_scores, pair_clusters, minimum = [], [], F(1)
    for d, o in pairs:
        sc, p_real = scores_fn(d, F(rows[o]["y"]))
        pair_scores.append(sc)
        pair_clusters.append(clusters_of(d))
        minimum = min(minimum, p_real)
    k = len(pair_scores[0])
    theta, cov, se, G = summarize(pair_scores, pair_clusters, k)
    return {
        "name": name, "design": design, "lag": lag, "rows": rows, "pairs": len(pairs), "unpaired": unpaired,
        "clusters": G, "coordinates": k, "minimum_probability": float(minimum),
        "estimate": [float(t) for t in theta],
        "standard_error": [float(s) for s in se],
        "covariance": [[float(v) for v in row] for row in cov],
        "lower": [float(mp_of(t) - Z975 * s) for t, s in zip(theta, se)],
        "upper": [float(mp_of(t) + Z975 * s) for t, s in zip(theta, se)],
        "family": family, "holm": holm(theta, se, family),
    }


def panel(units, periods, drop_rate, make_row):
    rows = []
    for u in range(units):
        level = u01() - 0.5
        for t in range(periods):
            if u01() < drop_rate:
                continue
            rows.append(make_row(u, t, level))
    return rows


def two_arm_case():
    def make(u, t, level):
        p = 0.06 if u % 4 == 0 else (0.94 if u % 4 == 1 else 0.3 + 0.4 * u01())
        a = u01() < p
        return {"unit": u, "period": t, "a": a, "p": p, "y": 1.0e6 + level + 0.7 * a + (u01() - 0.5)}
    rows = panel(9, 7, 0.15, make)
    def scores(d, y):
        r = rows[d]
        p = F(r["p"])
        treated = y / p if r["a"] else F(0)
        control = F(0) if r["a"] else y / (1 - p)
        return [treated - control, treated, control], (p if r["a"] else 1 - p)
    return emit("two_arm_extreme_probabilities", "two_arm", rows, 1, scores, [0], lambda d: rows[d]["unit"])


def multi_arm_case():
    K = 4
    def make(u, t, level):
        raw = [0.1 + u01() for _ in range(K)]
        s = sum(raw)
        probs = [v / s for v in raw]
        r = u01()
        arm, acc = 0, 0.0
        for j, p in enumerate(probs):
            acc += p
            if r < acc:
                arm = j
                break
        else:
            arm = K - 1
        return {"unit": u, "period": t, "arm": arm, "probs": probs, "y": level + 0.4 * arm + (u01() - 0.5)}
    rows = panel(9, 6, 0.1, make)
    def scores(d, y):
        r = rows[d]
        p = [F(v) for v in r["probs"]]
        # the product requires a probability row summing to one within 1e-8; the Fractions are exact
        means = [F(0)] * K
        means[r["arm"]] = y / p[r["arm"]]
        out = [means[j] - means[0] for j in range(1, K)]
        out += [means[j] - means[j - 1] for j in range(2, K)]
        out += means
        return out, p[r["arm"]]
    return emit("four_arms_three_clusters", "multi_arm", rows, 1, scores, [0, 1, 2], lambda d: rows[d]["unit"] // 3)


def factorial_case():
    def make(u, t, level):
        pa, pb = 0.25 + 0.5 * u01(), 0.3 + 0.4 * u01()
        a, b = u01() < pa, u01() < pb
        return {"unit": u, "period": t, "a": a, "b": b, "pa": pa, "pb": pb,
                "y": level + 0.5 * a - 0.3 * b + 0.6 * a * b + (u01() - 0.5)}
    rows = panel(10, 6, 0.1, make)
    def scores(d, y):
        r = rows[d]
        pa, pb = F(r["pa"]), F(r["pb"])
        ra = pa if r["a"] else 1 - pa
        rb = pb if r["b"] else 1 - pb
        cell = int(r["a"]) + 2 * int(r["b"])
        mu = [F(0)] * 4
        mu[cell] = y / (ra * rb)
        main_a = F(1, 2) * ((mu[1] - mu[0]) + (mu[3] - mu[2]))
        main_b = F(1, 2) * ((mu[2] - mu[0]) + (mu[3] - mu[1]))
        inter = mu[3] - mu[1] - mu[2] + mu[0]
        return [main_a, main_b, inter] + mu, ra * rb
    return emit("factorial_cells", "factorial", rows, 1, scores, [0, 1, 2], lambda d: rows[d]["unit"])


def interference_case():
    groups, members, periods = 5, 3, 6
    rows = []
    for g in range(groups):
        for t in range(periods):
            for m in range(members):
                if m == 2 and (g + t) % 4 == 0:
                    continue  # a member unobserved that period
                p = 0.3 + 0.4 * u01()
                a = u01() < p
                rows.append({"unit": g * members + m, "group": g, "period": t, "a": a, "p": p})
    for r in rows:
        r["y"] = 0.5 * r["group"] + 0.8 * r["a"] + (u01() - 0.5)
    def scores(d, y):
        r = rows[d]
        peers = [q for q in rows if q["group"] == r["group"] and q["period"] == r["period"] and q is not r]
        none = F(1)
        any_t = False
        for q in peers:
            none *= 1 - F(q["p"])
            any_t = any_t or q["a"]
        own = F(r["p"])
        p_own = own if r["a"] else 1 - own
        p_peers = (1 - none) if any_t else none
        exposure = int(r["a"]) + 2 * int(any_t)
        mu = [F(0)] * 4
        mu[exposure] = y / (p_own * p_peers)
        return [mu[1] - mu[0], mu[2] - mu[0], mu[3] - mu[0]] + mu, p_own * p_peers
    return emit("clustered_interference", "interference", rows, 1, scores, [0, 1, 2], lambda d: rows[d]["group"])


print(json.dumps({"cases": [two_arm_case(), multi_arm_case(), factorial_case(), interference_case()]}, indent=1))
