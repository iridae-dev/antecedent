# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Whole-program randomization, finite-carryover history means and the Gaussian AR(1) history
posterior, in exact rational arithmetic.

Regenerate with:  uv run --script joint_design_reference.py > joint_design_reference.json

Randomized programs.  Unit i has arm a_i, known probabilities p_i, outcome y_i and population
weight w_i.  The Horvitz-Thompson mean of program j is  m_j = sum_{i: a_i = j} w_i y_i / p_i[j] / W
(W = sum w_i) and the estimating-equation sandwich is
    Cov_jl = n/(n-1) sum_i (s_ij - w_i m_j)(s_il - w_i m_l) / W^2,   s_ij = w_i y_i / p_i[j] 1{a_i = j}.
Finite-carryover histories.  Sequence s has candidate schedules with known probabilities; the
inclusion probability of history h at period t is the total probability of the schedules whose
window ending at t equals h (enumerated here), the score of h is sum over the eligible periods t of
y_t / pi_{s,t,h} where the realised schedule matches, and
    m_h = sum_s score_sh / (total eligible periods),
    Cov_hk = S/(S-1) sum_s (score_sh - T_s m_h)(score_sk - T_s m_k) / T^2
(S sequences, T_s eligible periods of sequence s, T their sum).
History posterior.  Eligible outcome t of sequence s equals theta_{slot(t)} + e_t where e is a
stationary AR(1) series with marginal variance v_s and coefficient rho_s, so the covariance of the
sequence's eligible errors is v_s rho_s^|t - u| (built explicitly and inverted, not the tridiagonal
precision); theta has independent N(mean_h, var_h) priors.  Posterior precision =
diag(1/var) + sum_s A_s' Sigma_s^-1 A_s and mean = precision^-1 (prior_mean/var + sum_s A_s'
Sigma_s^-1 y_s).  Every input is an exactly representable binary64; values are written as nearest
binary64.  Repository-authored (MIT OR Apache-2.0).
"""
import json
from fractions import Fraction as F

_state = 0x1F83D9ABFB41BD6B


def u01():
    global _state
    _state = (_state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
    return (_state >> 11) / float(1 << 53)


def dyadic_probs(m):
    """m strictly positive dyadic probabilities summing exactly to one (denominator 64)."""
    cuts = set()
    while len(cuts) < m - 1:
        cuts.add(1 + int(u01() * 63))
    cuts = sorted(cuts)
    parts = [b - a for a, b in zip([0] + cuts, cuts + [64])]
    return [p / 64 for p in parts]


def randomized(name, n, k, y_offset):
    units = []
    for i in range(n):
        arm = i % k if i < 2 * k else int(u01() * k) % k
        probs = dyadic_probs(k)
        units.append({"id": f"u{i}", "arm": arm, "probabilities": probs,
                      "outcome": y_offset + 4 * (u01() - 0.5) + arm,
                      "population_weight": 0.25 + int(u01() * 12) / 4})
    W = sum(F(u["population_weight"]) for u in units)
    s = [[F(0)] * k for _ in units]
    for i, u in enumerate(units):
        s[i][u["arm"]] = F(u["population_weight"]) * F(u["outcome"]) / F(u["probabilities"][u["arm"]])
    m = [sum(s[i][j] for i in range(n)) / W for j in range(k)]
    cov = [[F(n, n - 1) * sum((s[i][j] - F(units[i]["population_weight"]) * m[j])
                              * (s[i][l] - F(units[i]["population_weight"]) * m[l]) for i in range(n)) / W ** 2
            for l in range(k)] for j in range(k)]
    return {"kind": "randomized", "name": name, "k": k, "units": units,
            "means": [float(v) for v in m], "covariance": [[float(v) for v in r] for r in cov]}


def inv(A):
    n = len(A)
    M = [list(A[i]) + [F(int(i == j)) for j in range(n)] for i in range(n)]
    for c in range(n):
        piv = next(r for r in range(c, n) if M[r][c] != 0)
        M[c], M[piv] = M[piv], M[c]
        d = M[c][c]
        M[c] = [v / d for v in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [a - f * b for a, b in zip(M[r], M[c])]
    return [row[n:] for row in M]


def histories_case(name, S, width, y_offset):
    hist = [[a, b] for a in (0, 1) for b in (0, 1)] if width == 2 else \
           [[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)]
    import itertools
    seqs = []
    for _ in range(S):
        L = width + int(u01() * 2)  # full factorial schedule sets keep every history in support
        scheds = [list(c) for c in itertools.product((0, 1), repeat=L)]
        probs = dyadic_probs(len(scheds))
        realized = int(u01() * len(scheds)) % len(scheds)
        outcomes = [y_offset + 3 * (u01() - 0.5) + 0.7 * scheds[realized][t] for t in range(L)]
        seqs.append({"schedules": scheds, "probabilities": probs, "realized": realized, "outcomes": outcomes})
    for h in hist:
        assert any(seq["schedules"][seq["realized"]][t + 1 - width:t + 1] == h for seq in seqs
                   for t in range(width - 1, len(seq["outcomes"]))), "history without observed support"
    T_s, scores = [], []
    for seq in seqs:
        L = len(seq["outcomes"])
        T_s.append(L - width + 1)
        sc = [F(0)] * len(hist)
        real = seq["schedules"][seq["realized"]]
        for t in range(width - 1, L):
            for j, h in enumerate(hist):
                pi = sum(F(p) for s_, p in zip(seq["schedules"], seq["probabilities"]) if s_[t + 1 - width:t + 1] == h)
                if real[t + 1 - width:t + 1] == h:
                    sc[j] += F(seq["outcomes"][t]) / pi
        scores.append(sc)
    T = sum(T_s)
    m = [sum(sc[j] for sc in scores) / T for j in range(len(hist))]
    K = len(hist)
    cov = [[F(S, S - 1) * sum((scores[s][j] - T_s[s] * m[j]) * (scores[s][l] - T_s[s] * m[l]) for s in range(S)) / T ** 2
            for l in range(K)] for j in range(K)]
    # posterior
    prior_mean = [float(int(u01() * 8) / 4) for _ in range(K)]
    prior_var = [float(1 + int(u01() * 8) / 2) for _ in range(K)]
    seq_var = [float(0.5 + int(u01() * 8) / 4) for _ in range(S)]
    seq_rho = [float(int(u01() * 15) / 16 - 0.5) for _ in range(S)]
    prec = [[F(0)] * K for _ in range(K)]
    rhs = [F(0)] * K
    for j in range(K):
        prec[j][j] = 1 / F(prior_var[j])
        rhs[j] = F(prior_mean[j]) / F(prior_var[j])
    for s, seq in enumerate(seqs):
        real = seq["schedules"][seq["realized"]]
        L = len(seq["outcomes"])
        ts = list(range(width - 1, L))
        slots = [hist.index(real[t + 1 - width:t + 1]) for t in ts]
        v, r = F(seq_var[s]), F(seq_rho[s])
        Sig = [[v * r ** abs(a - b) for b in range(len(ts))] for a in range(len(ts))]
        P = inv(Sig)
        yv = [F(seq["outcomes"][t]) for t in ts]
        for a in range(len(ts)):
            rhs[slots[a]] += sum(P[a][b] * yv[b] for b in range(len(ts)))
            for b in range(len(ts)):
                prec[slots[a]][slots[b]] += P[a][b]
    cov_post = inv(prec)
    mean_post = [sum(cov_post[j][l] * rhs[l] for l in range(K)) for j in range(K)]
    return {"kind": "histories", "name": name, "width": width, "histories": hist, "sequences": seqs,
            "means": [float(v) for v in m], "covariance": [[float(v) for v in r_] for r_ in cov],
            "observed_support": [sum(1 for seq in seqs for t in range(width - 1, len(seq["outcomes"]))
                                     if seq["schedules"][seq["realized"]][t + 1 - width:t + 1] == h) for h in hist],
            "eligible_periods": T,
            "prior_mean": prior_mean, "prior_variance": prior_var, "sequence_variance": seq_var,
            "sequence_rho": seq_rho, "posterior_mean": [float(v) for v in mean_post],
            "posterior_covariance": [[float(v) for v in r_] for r_ in cov_post]}


cases = [
    randomized("three programs, unequal probabilities and weights", 24, 3, 0.0),
    randomized("four programs, outcome offset 1e6", 30, 4, 1e6),
    randomized("two programs, smallest sample", 4, 2, 0.0),
    histories_case("carryover one, unequal sequence lengths", 9, 2, 0.0),
    histories_case("carryover two, outcome offset 1e5", 30, 3, 1e5),
]
print(json.dumps({"cases": cases}))
