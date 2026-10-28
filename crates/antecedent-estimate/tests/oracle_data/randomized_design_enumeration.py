# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Exhaustive design enumeration for three randomized estimators, in exact rational arithmetic.

Regenerate with:  uv run --script randomized_design_enumeration.py > randomized_design_enumeration.json

Each design fixes potential outcomes and a known assignment law, then lists every realization of
the assignment with its probability and the exact value of the estimator the product computes:

* multi_arm: independent unit-level draws of one of several arms; the Horvitz-Thompson arm mean
  (1/n) sum_i 1{A_i=a} Y_i/p_ia and the observed-arm variance contribution
  (1/n^2) sum_i 1{A_i=a} (1-p_ia) Y_i^2 / p_ia^2.
* scores: the independent-row action contrast (1/n) sum_i (1{A_i=a} Y_i/p_ia - 1{A_i=r} Y_i/p_ir) with
  its sample-score variance sum_i (s_i - mean)^2 / (n (n-1)), for outcomes that may have the fixed
  coefficient term theta x_i of a pre-assignment covariate removed first.
* switchback_itt: independent Bernoulli(p) draws per unit-period grouped in sequences; the
  Horvitz-Thompson contrast (1/n) sum_i (A_i Y_i/p_i - (1-A_i) Y_i/(1-p_i)) and the sequence
  sandwich S/(S-1) sum_s (T_s - Tbar)^2 / n^2 over the sequence scores T_s.
* history: whole-schedule randomization of independent sequences; the joint-history mean
  sum_s sum_t 1{window = h} Y_t / P(window = h) divided by the number of eligible periods, and the
  centred cluster covariance S/(S-1) sum_s (T_sj - L_s m_j)(T_sk - L_s m_k) / N^2.

While generating, the script asserts, exactly, that every mean is unbiased for its potential-outcome
target over the whole realization set (with the estimating formula applied even when the
product would refuse for missing support) and that the expected variance estimate equals the
design variance plus the nonnegative between-cluster term the centring leaves, so the variance
estimates are conservative; for the multi-arm designs it also asserts that twice the summed
observed-arm variance contributions, the contrast bound the route reports, is at least the exact
contrast variance.  Probabilities are dyadic so every input is exact in binary64; results
are written as the nearest binary64.  Repository-authored (MIT OR Apache-2.0).
"""
import itertools
import json
from fractions import Fraction as F


def fl(x):
    return None if x is None else float(x)


# ---- multi-arm -------------------------------------------------------------------------------
def multi_arm_design(name, y_by_arm, probs):
    n, k = len(probs), len(probs[0])
    reals = []
    exp_val = [F(0)] * k
    exp_var = [F(0)] * k
    exp_sq = [[F(0)] * k for _ in range(k)]
    for assign in itertools.product(range(k), repeat=n):
        pr = F(1)
        for i, a in enumerate(assign):
            pr *= F(probs[i][a])
        val = [F(0)] * k
        var = [F(0)] * k
        sup = [0] * k
        for i, a in enumerate(assign):
            p, y = F(probs[i][a]), F(y_by_arm[a][i])
            val[a] += y / p / n
            var[a] += (1 - p) * y * y / (p * p) / (n * n)
            sup[a] += 1
        for a in range(k):
            exp_val[a] += pr * val[a]
            exp_var[a] += pr * var[a]
            for b in range(k):
                exp_sq[a][b] += pr * (val[a] - val[b]) ** 2
        ok = all(s > 0 for s in sup)
        reals.append(dict(assignment=list(assign), prob=fl(pr),
                          arms=[dict(value=fl(val[a]), variance=fl(var[a]), support=sup[a])
                                for a in range(k)] if ok else None))
    for a in range(k):
        target = sum(F(y) for y in y_by_arm[a]) / n
        assert exp_val[a] == target
        design_var = sum((1 - F(probs[i][a])) / F(probs[i][a]) * F(y_by_arm[a][i]) ** 2
                         for i in range(n)) / (n * n)
        assert exp_var[a] == design_var  # the observed-arm contribution is unbiased for it
    # the contrast bound the route uses, 2 (V_a + V_b), dominates the exact contrast variance
    for a in range(k):
        for b in range(a + 1, k):
            true_contrast_var = exp_sq[a][b] - (exp_val[a] - exp_val[b]) ** 2
            assert true_contrast_var <= 2 * (exp_var[a] + exp_var[b])
    return dict(name=name, outcomes_by_arm=y_by_arm, probabilities=probs, realizations=reals)


# ---- independent score contrast ---------------------------------------------------------------
def score_design(name, y_by_arm, probs, reference, action, theta=None, x=None):
    """Row score 1{A=action} Y/p_action - 1{A=reference} Y/p_reference over independent draws.

    With theta and x given the outcomes are the fixed-coefficient adjusted outcomes
    Y_i - theta x_i of a pre-assignment covariate x (the potential outcomes are adjusted first)."""
    n, k = len(probs), len(probs[0])
    adj = [[F(y_by_arm[a][i]) - (F(theta) * F(x[i]) if theta is not None else 0) for i in range(n)]
           for a in range(k)]
    reals, e_eff, e_var, e_eff_sq = [], F(0), F(0), F(0)
    for assign in itertools.product(range(k), repeat=n):
        pr = F(1)
        for i, a in enumerate(assign):
            pr *= F(probs[i][a])
        sc = []
        for i, a in enumerate(assign):
            y = adj[a][i]
            sc.append(y / F(probs[i][action]) if a == action else
                      (-y / F(probs[i][reference]) if a == reference else F(0)))
        eff = sum(sc) / n
        var = sum((v - eff) ** 2 for v in sc) / ((n - 1) * n)
        e_eff += pr * eff
        e_eff_sq += pr * eff * eff
        e_var += pr * var
        ok = assign.count(action) > 0 and assign.count(reference) > 0
        outcomes = [float(adj[a][i]) for i, a in enumerate(assign)]
        reals.append(dict(assignment=list(assign), outcomes=outcomes, prob=fl(pr),
                          fit=dict(effect=fl(eff), variance=fl(var),
                                   reference_support=assign.count(reference),
                                   action_support=assign.count(action)) if ok else None))
    tau = [adj[action][i] - adj[reference][i] for i in range(n)]
    assert e_eff == sum(tau) / n  # unbiased, whatever the fixed coefficient
    true_var = e_eff_sq - e_eff ** 2
    taubar = sum(tau) / n
    assert e_var == true_var + sum((t - taubar) ** 2 for t in tau) / (n * (n - 1))
    return dict(name=name, probabilities=probs, reference=reference, action=action,
                realizations=reals)


# ---- switchback ITT --------------------------------------------------------------------------
def switchback_design(name, lengths, p, y1, y0):
    n = sum(lengths)
    ids = [f"s{s}" for s, L in enumerate(lengths) for _ in range(L)]
    starts = [sum(lengths[:s]) for s in range(len(lengths))]
    S = len(lengths)
    reals = []
    e_eff = F(0)
    e_var = F(0)
    for assign in itertools.product([0, 1], repeat=n):
        pr = F(1)
        for i, a in enumerate(assign):
            pr *= F(p[i]) if a else 1 - F(p[i])
        T = []
        for s in range(S):
            tot = F(0)
            for i in range(starts[s], starts[s] + lengths[s]):
                tot += (F(y1[i]) / F(p[i])) if assign[i] else -(F(y0[i]) / (1 - F(p[i])))
            T.append(tot)
        eff = sum(T) / n
        Tbar = sum(T) / S
        var = F(S, S - 1) * sum((t - Tbar) ** 2 for t in T) / (n * n)
        e_eff += pr * eff
        e_var += pr * var
        both = 0 < sum(assign) < n
        reals.append(dict(assignment=list(assign), prob=fl(pr),
                          fit=dict(effect=fl(eff), variance=fl(var)) if both else None))
    target = sum(F(y1[i]) - F(y0[i]) for i in range(n)) / n
    assert e_eff == target
    mu = [sum(F(y1[i]) - F(y0[i]) for i in range(starts[s], starts[s] + lengths[s])) for s in range(S)]
    mubar = sum(mu) / S
    # variance of the sum of independent sequence scores
    true_var = F(0)
    for s in range(S):
        for i in range(starts[s], starts[s] + lengths[s]):
            q = F(p[i])
            true_var += F(y1[i]) ** 2 * (1 - q) / q + F(y0[i]) ** 2 * q / (1 - q) + 2 * F(y1[i]) * F(y0[i])
    true_var /= n * n
    extra = F(S, S - 1) * sum((m - mubar) ** 2 for m in mu) / (n * n)
    assert e_var == true_var + extra and extra >= 0
    return dict(name=name, lengths=lengths, ids=ids, probabilities=p, treated_outcomes=y1,
                control_outcomes=y0, realizations=reals)


# ---- history means ---------------------------------------------------------------------------
def history_design(name, n, schedules_per_seq, probs_per_seq, outcome, histories):
    S = len(schedules_per_seq)
    w = len(histories[0])
    total = S * (n - w + 1)
    reals = []
    exp_mean = [F(0)] * len(histories)
    for choice in itertools.product(*[range(len(sc)) for sc in schedules_per_seq]):
        pr = F(1)
        for s, c in enumerate(choice):
            pr *= F(probs_per_seq[s][c])
        scores, support = [], [0] * len(histories)
        outcomes = []
        for s, c in enumerate(choice):
            sched = schedules_per_seq[s][c]
            y = [outcome(s, sched, t) for t in range(n)]
            outcomes.append(y)
            sc = [F(0)] * len(histories)
            for t in range(w - 1, n):
                for j, h in enumerate(histories):
                    win = lambda sch: sch[t + 1 - w:t + 1] == h
                    prob = sum(F(pp) for sch, pp in zip(schedules_per_seq[s], probs_per_seq[s]) if win(sch))
                    if win(sched):
                        sc[j] += F(y[t]) / prob
                        support[j] += 1
            scores.append(sc)
        means = [sum(sc[j] for sc in scores) / total for j in range(len(histories))]
        for j in range(len(histories)):
            exp_mean[j] += pr * means[j]
        cov = [[sum(F(S, S - 1) * (scores[s][j] - (n - w + 1) * means[j])
                    * (scores[s][k] - (n - w + 1) * means[k]) for s in range(S)) / (total * total)
                for k in range(len(histories))] for j in range(len(histories))]
        ok = all(c > 0 for c in support)
        reals.append(dict(choice=list(choice), outcomes=outcomes, prob=fl(pr),
                          fit=dict(means=[fl(m) for m in means],
                                   covariance=[fl(c) for row in cov for c in row],
                                   support=support, eligible=total) if ok else None))
    for j, h in enumerate(histories):
        target = F(0)
        for s in range(S):
            for t in range(w - 1, n):
                # potential outcome at t if the window equals h: the outcome function of a schedule
                # whose window is h (identified by the outcome depending only on the window)
                probe = next(sch for sch in schedules_per_seq[s] if sch[t + 1 - w:t + 1] == h)
                target += F(outcome(s, probe, t))
        assert exp_mean[j] == target / total
    return dict(name=name, n=n, schedules=schedules_per_seq, probabilities=probs_per_seq,
                histories=histories, realizations=reals)


def main():
    d = {}
    d["multi_arm"] = [
        multi_arm_design("three_arms_mixed_probabilities",
                         [[1, 0, 2], [3, 5, -4], [7, 1, 2]],
                         [[0.5, 0.25, 0.25], [0.125, 0.125, 0.75], [0.0625, 0.4375, 0.5]]),
        multi_arm_design("offset_and_tiny_probability",
                         [[1e8 + 1, 1e8 + 3, 1e8 + 2], [1e8 - 1, 1e8 + 5, 1e8]],
                         [[2.0 ** -20, 1 - 2.0 ** -20], [0.5, 0.5], [0.75, 0.25]]),
        multi_arm_design("four_arms_two_units",
                         [[2, 3], [5, 7], [-1, 4], [0, 9]],
                         [[0.25] * 4, [0.5, 0.25, 0.125, 0.125]]),
    ]
    x = [1.0, -2.0, 0.5, 3.0, -1.5, 2.0]
    y2 = [[2.0, 1.0, 4.0, 0.0, 3.0, 5.0], [5.0, 3.0, 6.0, 4.0, 2.0, 9.0]]
    p2 = [[0.5, 0.5], [0.75, 0.25], [0.25, 0.75], [0.5, 0.5], [0.875, 0.125], [0.5, 0.5]]
    d["scores"] = [
        score_design("two_arm_unadjusted", y2, p2, 0, 1),
        score_design("two_arm_fixed_coefficient", y2, p2, 0, 1, theta=3.0, x=x),
        score_design("two_arm_offset_outcomes", [[1e8 + v for v in y2[0]], [1e8 + v for v in y2[1]]],
                     p2, 0, 1),
        score_design("three_arm", [[1.0, 2.0, 0.0, 3.0], [4.0, 1.0, 2.0, 5.0], [0.0, 6.0, 3.0, 1.0]],
                     [[0.5, 0.25, 0.25], [0.25, 0.5, 0.25], [0.125, 0.375, 0.5], [0.5, 0.25, 0.25]], 0, 2),
    ]
    d["switchback_itt"] = [
        switchback_design("equal_sequences", [2, 2, 2], [0.5, 0.25, 0.75, 0.5, 0.25, 0.75],
                          [3, 5, 2, 7, 1, 4], [1, 2, 2, 3, 0, 1]),
        switchback_design("unequal_sequences", [1, 3, 2], [0.5, 0.25, 0.5, 0.75, 0.5, 0.25],
                          [4, 2, 6, 1, 3, 5], [1, 0, 2, 1, 1, 2]),
        switchback_design("outcome_offset", [2, 2, 2], [0.5, 0.5, 0.25, 0.75, 0.5, 0.5],
                          [1e8 + 3, 1e8 + 1, 1e8 + 2, 1e8 + 5, 1e8, 1e8 + 4],
                          [1e8 + 1, 1e8, 1e8 + 2, 1e8 + 1, 1e8 + 3, 1e8 + 2]),
    ]
    sch = [[0, 0, 0], [0, 1, 1], [1, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]]
    prob = [0.25, 0.125, 0.25, 0.125, 0.125, 0.125]
    def carry(s, sched, t):
        w2 = tuple(sched[max(t - 1, 0):t + 1]) if t >= 1 else (sched[0],)
        return 10 * s + 2 * w2[-1] + 3 * (w2[0] if len(w2) == 2 else 0) + 5 * (w2[-1] * (w2[0] if len(w2) == 2 else 0))
    d["history"] = [
        history_design("pair_histories_unequal_laws", 3, [sch, sch[:5], sch],
                       [prob, [0.25, 0.25, 0.125, 0.25, 0.125], [0.5, 0.0625, 0.0625, 0.125, 0.125, 0.125]],
                       lambda s, sched, t: carry(s, sched, t), [[0, 0], [0, 1], [1, 0], [1, 1]]),
        history_design("single_period_histories", 3, [sch, sch],
                       [prob, prob], lambda s, sched, t: 10 * s + 4 * sched[t] + t, [[0], [1]]),
    ]
    json.dump(d, fp=__import__("sys").stdout, separators=(",", ":"))


main()
