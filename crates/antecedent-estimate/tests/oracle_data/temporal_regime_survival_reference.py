# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath", "numpy"]
# ///
"""Parametric g-formula survival under static regimes: 50-digit model fits, quadrature-exact integral.

Regenerate with:  uv run --script temporal_regime_survival_reference.py > temporal_regime_survival_reference.json

Person-period rows k = 0, 1, ... (kept while k < K) carry the confounder L_k, the treatment A_k, the
censoring indicator C_k and, when uncensored, the event indicator Y_k.  Three models are fitted by
maximum likelihood at 50 digits:
  hazard      P(Y_k = 1 | 1, L_k, A_k, k/K) pooled logistic on the uncensored rows (Newton's method),
  transition  L_k = [1, L_{k-1}, A_{k-1}, k/K] b + sigma e on rows k >= 1 (least squares; sigma^2 is the
              residual sum of squares over the number of rows, the Gaussian maximum likelihood scale),
  treatment   P(A_k = 1 | 1, L_k, A_{k-1}, k/K)   (A_{-1} = 0), used only for the smallest fitted
              probability of a regime-consistent action over all regime-consistent rows.
For a regime a in {1, 0} every subject's baseline L_0 is carried through the fitted transition law
with A = a and S_a(K) is the subject average of E[prod_k (1 - h_k(L_k, a))], the expectation over the
Gaussian innovations evaluated by nested Gauss-Hermite quadrature (converged to 1e-9; checked here).
The estimate is [S_1 - S_0, S_1, S_0].  The product evaluates the same expectation by Monte Carlo on
16 fixed innovation paths per subject shared by both regimes, so it differs from the quadrature value
by Monte Carlo error only: `monte_carlo_tolerance` is five times the standard deviation of that error
over 400 independent innovation draws of the same scheme (per coordinate, absolute).  The synthetic
data are drawn from a sequential structural model with an LCG; values are written as the nearest
binary64.  Repository-authored (MIT OR Apache-2.0); mpmath is BSD-3-Clause, numpy is BSD-3-Clause.
"""
import json
from math import exp as fexp, log, sqrt, cos, pi

import numpy as np
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


def ols(rows, y):
    p = len(rows[0])
    gram, rhs = matrix(p, p), [mpf(0)] * p
    for r, yi in zip(rows, y):
        for i in range(p):
            rhs[i] += r[i] * yi
            for j in range(p):
                gram[i, j] += r[i] * r[j]
    beta = list(lu_solve(gram, matrix(rhs)))
    rss = sum((yi - sum(b * x for b, x in zip(beta, r))) ** 2 for r, yi in zip(rows, y))
    return beta, mp.sqrt(rss / len(y))


NODES, WEIGHTS = np.polynomial.hermite_e.hermegauss(20)
WEIGHTS = WEIGHTS / WEIGHTS.sum()
FINE_NODES, FINE_WEIGHTS = np.polynomial.hermite_e.hermegauss(28)
FINE_WEIGHTS = FINE_WEIGHTS / FINE_WEIGHTS.sum()
PATHS = 16


def expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def integrated_survival(l0, arm, horizon, bh, bl, sd, nodes, weights):
    """E[prod_k (1 - h_k)] by nested Gauss-Hermite quadrature, vectorised over the node grid."""
    l = np.full(1, l0)
    w = np.ones(1)
    surv = np.ones(1)
    for k in range(horizon):
        if k > 0:
            l = (bl[0] + bl[1] * l + bl[2] * arm + bl[3] * k / horizon)[:, None] + sd * nodes[None, :]
            w = (w[:, None] * weights[None, :]).ravel()
            surv = np.repeat(surv, len(nodes))
            l = l.ravel()
        surv = surv * (1 - expit(bh[0] + bh[1] * l + bh[2] * arm + bh[3] * k / horizon))
    return float((w * surv).sum())


def monte_carlo_survival(l0, arm, horizon, bh, bl, sd, noise):
    """The product's scheme: one fixed innovation path per (subject, draw), shared by both regimes."""
    total = 0.0
    for m in range(PATHS):
        l, s = l0, 1.0
        for k in range(horizon):
            if k > 0:
                l = bl[0] + bl[1] * l + bl[2] * arm + bl[3] * k / horizon + sd * noise[m, k]
            s *= 1 - expit(bh[0] + bh[1] * l + bh[2] * arm + bh[3] * k / horizon)
        total += s
    return total / PATHS


def run(name, subjects, horizon, floor):
    kept = [s[:horizon] for s in subjects]
    tx, ty, hx, hy, lx, ly = [], [], [], [], [], []
    for s in kept:
        for k, r in enumerate(s):
            prev = s[k - 1]["A"] if k else 0
            tx.append([mpf(1), mpf(r["L"]), mpf(prev), mpf(k) / horizon])
            ty.append(mpf(r["A"]))
            if not r["C"]:
                hx.append([mpf(1), mpf(r["L"]), mpf(r["A"]), mpf(k) / horizon])
                hy.append(mpf(r["Y"]))
            if k:
                lx.append([mpf(1), mpf(s[k - 1]["L"]), mpf(s[k - 1]["A"]), mpf(k) / horizon])
                ly.append(mpf(r["L"]))
    beta_a = logit(tx, ty)
    beta_h = logit(hx, hy)
    beta_l, sigma = ols(lx, ly)
    minimum = mpf(1)
    for arm in (1, 0):
        for s in kept:
            for k, r in enumerate(s):
                if r["A"] != arm:
                    break
                prev = s[k - 1]["A"] if k else 0
                p1 = prob(beta_a, [mpf(1), mpf(r["L"]), mpf(prev), mpf(k) / horizon])
                minimum = min(minimum, p1 if arm else 1 - p1)
    bh = [float(b) for b in beta_h]
    bl = [float(b) for b in beta_l]
    sd = float(sigma)
    l0s = [s[0]["L"] for s in kept]
    surv = {}
    for arm in (1, 0):
        values = [integrated_survival(l0, arm, horizon, bh, bl, sd, NODES, WEIGHTS) for l0 in l0s]
        fine = [integrated_survival(l0, arm, horizon, bh, bl, sd, FINE_NODES, FINE_WEIGHTS) for l0 in l0s]
        assert max(abs(a - b) for a, b in zip(values, fine)) < 1e-9, "quadrature has not converged"
        surv[arm] = sum(fine) / len(fine)
    estimate = [surv[1] - surv[0], surv[1], surv[0]]
    errors = []
    rng = np.random.default_rng(20240607)
    for _ in range(400):
        noise = rng.standard_normal((len(l0s), PATHS, horizon))
        a1 = np.mean([monte_carlo_survival(l0, 1, horizon, bh, bl, sd, noise[i]) for i, l0 in enumerate(l0s)])
        a0 = np.mean([monte_carlo_survival(l0, 0, horizon, bh, bl, sd, noise[i]) for i, l0 in enumerate(l0s)])
        errors.append([a1 - a0 - estimate[0], a1 - estimate[1], a0 - estimate[2]])
    tolerance = [float(5 * np.std(np.array(errors)[:, j], ddof=1)) for j in range(3)]
    rows = []
    for u, s in enumerate(subjects):
        for k, r in enumerate(s):
            rows.append({"unit": u, "period": k, "treatment": bool(r["A"]), "censored": bool(r["C"]),
                         "event": bool(r["Y"]), "L": r["L"]})
    return {
        "name": name, "horizon": horizon, "floor": floor, "rows": rows,
        "estimate": estimate, "monte_carlo_tolerance": tolerance,
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
