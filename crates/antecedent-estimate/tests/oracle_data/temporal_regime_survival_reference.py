# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath"]
# ///
"""Inverse-probability-weighted survival under static regimes, at 50 digits.

Regenerate with:  uv run --script temporal_regime_survival_reference.py > temporal_regime_survival_reference.json

Person-period rows k = 0, 1, ... (kept while k < K) carry the confounder L_k, the treatment A_k, the
censoring indicator C_k and, when uncensored, the event indicator Y_k.  Two pooled logistic models
are fitted by maximum likelihood (Newton's method at 50 digits):
  treatment   P(A_k = 1 | 1, L_k, A_{k-1}, k/K)     (A_{-1} = 0),
  censoring   P(C_k = 1 | 1, L_k, A_k, k/K)          (omitted when no row is censored).
For a regime a in {1, 0} a subject contributes while its observed treatments equal a:
  W_k = prod_{j<=k} 1/(P(A_j = a | ...) (1 - P(C_j = 1 | L_j, a, j))),
the weighted discrete hazard of period k is h_k = sum_i W_ik Y_ik / sum_i W_ik over the subjects that
are regime-consistent and uncensored at k (a subject censored in period k contributes nothing to it),
and S_a(K) = prod_k (1 - h_k).  The estimate is [S_1 - S_0, S_1, S_0]; the smallest fitted probability
of a regime-consistent action over all regime-consistent rows is also written.  The synthetic data
are drawn from a sequential structural model with an LCG; values are written as the nearest binary64.
Repository-authored (MIT OR Apache-2.0); mpmath is BSD-3-Clause.
"""
import json
from math import exp as fexp, log, sqrt, cos, pi

from mpmath import mp, mpf, matrix, lu_solve, exp

mp.dps = 50
_state = 0x3F84D5B5B5470917


def u01():
    global _state
    _state = (_state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
    return (_state >> 11) / float(1 << 53)


def normal():
    return sqrt(-2.0 * log(1.0 - u01())) * cos(2.0 * pi * u01())


def sig(x):
    return 1.0 / (1.0 + fexp(-x))


def simulate(n_subjects, max_periods, censor_rate, a_strength, drop_censoring):
    subjects = []
    for _ in range(n_subjects):
        rows, prev, L = [], 0, normal()
        for k in range(max_periods):
            a = int(u01() < sig(-0.2 + a_strength * L + 0.5 * prev))
            c = 0 if drop_censoring else int(u01() < sig(censor_rate + 0.4 * L - 0.3 * a))
            y = 0
            if not c:
                y = int(u01() < sig(-1.6 - 0.8 * a + 0.5 * L))
            rows.append({"L": L, "A": a, "C": c, "Y": y})
            if c or y:
                break
            prev = a
            L = 0.6 * L + 0.3 * a + 0.5 * normal()
        subjects.append(rows)
    return subjects


def logit(rows, t):
    p = len(rows[0])
    beta = [mpf(0)] * p
    for _ in range(100):
        score = [mpf(0)] * p
        info = matrix(p, p)
        for r, ti in zip(rows, t):
            e = 1 / (1 + exp(-sum(beta[k] * r[k] for k in range(p))))
            for i in range(p):
                score[i] += (ti - e) * r[i]
                for j in range(p):
                    info[i, j] += e * (1 - e) * r[i] * r[j]
        step = lu_solve(info, matrix(score))
        beta = [beta[k] + step[k] for k in range(p)]
        if max(abs(s) for s in step) < mpf(10) ** -40:
            return beta
    raise RuntimeError("logistic maximum likelihood did not converge (separation)")


def prob(beta, row):
    return 1 / (1 + exp(-sum(b * r for b, r in zip(beta, row))))


def run(name, subjects, horizon, floor):
    kept = [s[:horizon] for s in subjects]
    tx, ty, cx, cy = [], [], [], []
    for s in kept:
        for k, r in enumerate(s):
            prev = s[k - 1]["A"] if k else 0
            tx.append([mpf(1), mpf(r["L"]), mpf(prev), mpf(k) / horizon])
            ty.append(mpf(r["A"]))
            cx.append([mpf(1), mpf(r["L"]), mpf(r["A"]), mpf(k) / horizon])
            cy.append(mpf(r["C"]))
    beta_a = logit(tx, ty)
    beta_c = logit(cx, cy) if any(v > 0 for v in cy) else None
    surv, minimum = {}, mpf(1)
    for arm in (1, 0):
        num, den = [mpf(0)] * horizon, [mpf(0)] * horizon
        for s in kept:
            cum = mpf(1)
            for k, r in enumerate(s):
                if r["A"] != arm:
                    break
                prev = s[k - 1]["A"] if k else 0
                p1 = prob(beta_a, [mpf(1), mpf(r["L"]), mpf(prev), mpf(k) / horizon])
                p_arm = p1 if arm else 1 - p1
                minimum = min(minimum, p_arm)
                cum /= p_arm
                if beta_c is not None:
                    pc = prob(beta_c, [mpf(1), mpf(r["L"]), mpf(arm), mpf(k) / horizon])
                    cum /= max(1 - pc, mpf(floor))
                if r["C"]:
                    break
                num[k] += cum * r["Y"]
                den[k] += cum
        s_arm = mpf(1)
        for k in range(horizon):
            s_arm *= 1 - num[k] / den[k]
        surv[arm] = s_arm
    rows = []
    for u, s in enumerate(subjects):
        for k, r in enumerate(s):
            rows.append({"unit": u, "period": k, "treatment": bool(r["A"]), "censored": bool(r["C"]),
                         "event": bool(r["Y"]), "L": r["L"]})
    return {
        "name": name, "horizon": horizon, "floor": floor, "rows": rows,
        "estimate": [float(surv[1] - surv[0]), float(surv[1]), float(surv[0])],
        "minimum_regime_probability": float(minimum),
        "person_periods": len(tx), "subjects": len(kept),
    }


cases = [
    run("censored_and_truncated", simulate(70, 6, -1.8, 0.8, False), 4, 0.01),
    run("no_censoring", simulate(60, 5, -9.0, 0.6, True), 3, 0.01),
    run("strong_confounding", simulate(140, 6, -1.8, 1.3, False), 4, 0.005),
    run("small_sample", simulate(36, 4, -1.5, 0.5, False), 3, 0.01),
]
print(json.dumps({"cases": cases}, indent=1))
