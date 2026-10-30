# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Family selection: every candidate's held-out score recomputed from its own exact fit.

Regenerate with:  uv run --script registry_family_selection_reference.py FOLDS.json > registry_family_selection_reference.json
where FOLDS.json is the fold assignment printed by the test when it differs from the frozen one
(`folds` below; `--data` prints the inputs the folds are cut for); the shuffled partition is the
library's seeded kernel and is the one input this script does not recompute.

A candidate's score is   mean over held-out rows of log p(y_i | parents_i)  -  k ln(n) / (2 n),
with the model refit on the training rows of every fold and the penalty's parameter count k the
registry's (p + 2 for linear-Gaussian, hierarchical-linear and BVAR; (levels - 1)(1 + p) for a
parent-conditional discrete or hierarchical-GLM fit; p + 5 for the conditional state space).  Each
refit is solved exactly (normal equations, 60 digits; Newton on the concave logit likelihoods):

  linear_gaussian  OLS;   hierarchical_linear  empirical-Bayes ridge;   bvar  ridge on the
  scale-standardised parents (phi = 0.2);   discrete  binary logit MLE;   hierarchical_glm
  ridge logit with the raw-slope empirical-Bayes strength recomputed on the training rows:
  linear OLS sampling variance uses the centered Gram inverse and the logit uses the
  unpenalized pilot's inverse observed Fisher information.

The conditional state space is scored prequentially: its parameters (a, q, r, x0, from 25 EM
iterations of the joint-law E-step, on the full-data OLS residual) are fixed, and the Kalman
one-step predictive density is evaluated on the *out-of-fold* OLS residual series, whose folds are
the declared validation scheme's.  Under the time-ordered scheme the leading block is never held
out, so its residuals come from the first fold's fit, and the score averages the held-out rows
only (rows from the first held-out block on); under the shuffled scheme every row is held out.

The constant family is inadmissible (-inf) whenever the outcome has any spread; it is reported as
failed, not scored.  Repository-authored (MIT OR Apache-2.0).
"""
import json
import random
import sys
from fractions import Fraction as F

import mpmath as mp

mp.mp.dps = 60
TOL = mp.mpf(10) ** -40
FOLDS_PATH = sys.argv[1] if len(sys.argv) > 1 else None


def mpf(f):
    f = F(f)
    return mp.mpf(f.numerator) / f.denominator


def wls_solve(D, y, w, penalty):
    n, k = len(D), len(D[0])
    A = mp.zeros(k, k)
    r = mp.zeros(k, 1)
    for i in range(n):
        for a in range(k):
            r[a] += w[i] * D[i][a] * y[i]
            for b in range(k):
                A[a, b] += w[i] * D[i][a] * D[i][b]
    for c in range(k):
        A[c, c] += penalty[c]
    return list(mp.lu_solve(A, r))


def sigma_of(rss, n, k, y):
    rms = mp.sqrt(sum(v * v for v in y) / n)
    floor = max(mp.mpf("1e-8") * rms, mp.mpf(2) ** -1022)
    return max(mp.sqrt(rss / max(n - k, 1)), floor)


def design(cols, rows):
    return [[mp.mpf(1)] + [c[i] for c in cols] for i in rows]


def rss_of(D, y, beta):
    return sum((y[i] - sum(beta[a] * D[i][a] for a in range(len(beta)))) ** 2 for i in range(len(y)))


def fit_lg(cols, y_all, rows):
    D, y = design(cols, rows), [y_all[i] for i in rows]
    beta = wls_solve(D, y, [mp.mpf(1)] * len(y), [0] * len(D[0]))
    return beta, sigma_of(rss_of(D, y, beta), len(y), len(D[0]), y)


def fit_hl(cols, y_all, rows):
    D, y = design(cols, rows), [y_all[i] for i in rows]
    n, p = len(y), len(cols)
    b0, s0 = fit_lg(cols, y_all, rows)
    if p:
        means = [sum(cols[j][i] for i in rows) / n for j in range(p)]
        G = mp.matrix([[sum((cols[a][i] - means[a]) * (cols[b][i] - means[b]) for i in rows)
                        for b in range(p)] for a in range(p)])
        inverse = G ** -1
        sampling = s0 * s0 * sum(inverse[j, j] for j in range(p)) / p
        tau2 = max(sum(b * b for b in b0[1:]) / p - sampling, mp.mpf("1e-8"))
        lam = min(max(s0 * s0 / tau2, mp.mpf("1e-6")), mp.mpf("1e6"))
    else:
        lam = mp.mpf(0)
    beta = wls_solve(D, y, [mp.mpf(1)] * n, [0] + [lam] * p)
    return beta, sigma_of(rss_of(D, y, beta), n, p + 1, y)


def fit_bvar(cols, y_all, rows):
    y = [y_all[i] for i in rows]
    n, p = len(y), len(cols)
    sc = []
    for c in cols:
        v = [c[i] for i in rows]
        m = sum(v) / n
        sd = mp.sqrt(sum((x - m) ** 2 for x in v) / n)
        sc.append(sd if sd > mp.mpf("1e-12") else mp.mpf(1))
    D = [[mp.mpf(1)] + [cols[j][i] / sc[j] for j in range(p)] for i in rows]
    b = wls_solve(D, y, [mp.mpf(1)] * n, [0] + [1 / mp.mpf("0.2")] * p)
    beta = [b[0]] + [b[1 + j] / sc[j] for j in range(p)]
    return beta, sigma_of(rss_of(design(cols, rows), y, beta), n, p + 1, y)


def gauss_lp(y, m, s):
    return -mp.log(2 * mp.pi * s * s) / 2 - (y - m) ** 2 / (2 * s * s)


def predict(beta, cols, i):
    return beta[0] + sum(beta[1 + j] * cols[j][i] for j in range(len(cols)))


# ---------------------------------------------------------------- EM and the Kalman density
def em(y, iters=25):
    n = len(y)
    a, q, r, x0 = mp.mpf("0.8"), mp.mpf(1), mp.mpf(1), y[0]
    for _ in range(iters):
        var, v = [], q
        for t in range(n):
            if t > 0:
                v = a * a * v + q
            var.append(v)
        cov = mp.zeros(n, n)
        for s in range(n):
            for t in range(s, n):
                cov[s, t] = cov[t, s] = a ** (t - s) * var[s]
        mu = mp.matrix([a ** t * x0 for t in range(n)])
        G = cov * mp.inverse(cov + r * mp.eye(n))
        m = mu + G * mp.matrix([y[t] - mu[t] for t in range(n)])
        C = cov - G * cov
        P = [C[t, t] for t in range(n)]
        lag = [mp.mpf(0)] + [C[t, t - 1] for t in range(1, n)]
        num = sum(lag[t] + m[t] * m[t - 1] for t in range(1, n))
        den = sum(P[t - 1] + m[t - 1] ** 2 for t in range(1, n))
        if den > mp.mpf("1e-12"):
            a = max(min(num / den, mp.mpf("0.999")), mp.mpf("-0.999"))
        x0 = m[0]
        q_acc = P[0] + sum(P[t] + m[t] ** 2 + a * a * (P[t - 1] + m[t - 1] ** 2)
                           - 2 * a * (lag[t] + m[t] * m[t - 1]) for t in range(1, n))
        q = max(q_acc / n, mp.mpf("1e-8"))
        r = max(sum(P[t] + (y[t] - m[t]) ** 2 for t in range(n)) / n, mp.mpf("1e-8"))
    return a, q, r, x0


def predictive_density(res, a, q, r, x0):
    x, p = x0, q
    out = []
    for t, v in enumerate(res):
        xp = x if t == 0 else a * x
        pp = p if t == 0 else a * a * p + q
        s = pp + r
        out.append(-mp.log(2 * mp.pi * s) / 2 - (v - xp) ** 2 / (2 * s))
        k = pp / s
        x = xp + k * (v - xp)
        p = (1 - k) * pp
    return out


# ---------------------------------------------------------------- logit fits
def newton_logit(X, y, penalty):
    n, p = len(X), len(X[0])
    beta = mp.zeros(p, 1)
    for _ in range(200):
        g, H = mp.zeros(p, 1), mp.zeros(p, p)
        for i in range(n):
            eta = sum(beta[j] * X[i][j] for j in range(p))
            mu = 1 / (1 + mp.e ** (-eta))
            for j in range(p):
                g[j] += (y[i] - mu) * X[i][j]
                for m_ in range(p):
                    H[j, m_] += mu * (1 - mu) * X[i][j] * X[i][m_]
        for j in range(p):
            g[j] -= penalty[j] * beta[j]
            H[j, j] += penalty[j]
        if mp.norm(g, p=mp.inf) < TOL:
            break
        beta = beta + mp.lu_solve(H, g)
    else:
        raise ValueError("logistic oracle did not converge in 200 iterations")
    return list(beta)


def fit_discrete(cols, y_all, rows):
    X = design(cols, rows)
    return newton_logit(X, [y_all[i] for i in rows], [0] * len(X[0]))


def fit_hglm(cols, y_all, rows):
    n, p = len(rows), len(cols)
    X = design(cols, rows)
    y = [y_all[i] for i in rows]
    if p:
        pilot = newton_logit(X, y, [0] * (p + 1))
        fisher = mp.zeros(p + 1, p + 1)
        for row in X:
            eta = sum(pilot[j] * row[j] for j in range(p + 1))
            mu = 1 / (1 + mp.exp(-eta))
            for i in range(p + 1):
                for j in range(p + 1):
                    fisher[i, j] += mu * (1 - mu) * row[i] * row[j]
        inverse = fisher ** -1
        tau2 = max(sum(b * b for b in pilot[1:]) / p
                   - sum(inverse[j,j] for j in range(1,p+1)) / p, mp.mpf("1e-8"))
        lam = min(max(1 / tau2, mp.mpf("1e-4")), mp.mpf("1e3"))
    else:
        lam = mp.mpf(1)
    return newton_logit(X, y, [0] + [lam] * p)


def logit_lp(beta, cols, y, i):
    eta = beta[0] + sum(beta[1 + j] * cols[j][i] for j in range(len(cols)))
    return y[i] * eta - mp.log(1 + mp.e ** eta)


# ---------------------------------------------------------------- folds
def time_ordered(n, purge):
    blocks = 6
    edge = lambda j: j * n // blocks
    return [([i for i in range(max(edge(k) - purge, 0))], list(range(edge(k), edge(k + 1))))
            for k in range(1, blocks)]


def shuffled(assign):
    k = max(assign) + 1
    return [([i for i, f in enumerate(assign) if f != fold], [i for i, f in enumerate(assign) if f == fold])
            for fold in range(k)]


def candidate(held, k, n):
    pen = mp.mpf(k) * mp.log(n) / (2 * n)
    return {"held_out": float(held), "penalty": float(pen), "score": float(held - pen)}


def continuous_scores(cols, y, folds, first_scored_from_folds):
    n, p = len(y), len(cols)
    out = {}
    for name, fit, k in [("linear_gaussian", fit_lg, p + 2), ("hierarchical_linear", fit_hl, p + 2),
                         ("bvar", fit_bvar, p + 2)]:
        tot, cnt = mp.mpf(0), 0
        for train, test in folds:
            beta, s = fit(cols, y, train)
            for i in test:
                tot += gauss_lp(y[i], predict(beta, cols, i), s)
                cnt += 1
        out[name] = candidate(tot / cnt, k, n)
    # conditional state space: full-data OLS, EM on the normalised residual, OOF residual series
    beta, _ = fit_lg(cols, y, list(range(n)))
    resid = [y[i] - predict(beta, cols, i) for i in range(n)]
    scale = max(abs(r) for r in resid)
    a, q, r, x0 = em([v / scale for v in resid])
    a, q, r, x0 = a, q * scale * scale, r * scale * scale, x0 * scale
    oof = [None] * n
    for train, test in folds:
        b, _ = fit_lg(cols, y, train)
        for i in test:
            oof[i] = y[i] - predict(b, cols, i)
    first = 0
    if first_scored_from_folds:
        b, _ = fit_lg(cols, y, folds[0][0])
        first = min(folds[0][1])
        for i in range(first):
            oof[i] = y[i] - predict(b, cols, i)
    assert all(v is not None for v in oof)
    dens = predictive_density(oof, a, q, r, x0)
    held = sum(dens[first:]) / (n - first)
    out["lgssm"] = candidate(held, p + 5, n)
    out["lgssm"]["parameters"] = {"a": float(a), "process_std": float(mp.sqrt(q)),
                                  "obs_std": float(mp.sqrt(r)), "initial_mean": float(x0)}
    return out


def binary_scores(cols, y, folds):
    n, p = len(y), len(cols)
    out = {}
    for name, fit in [("discrete", fit_discrete), ("hierarchical_glm", fit_hglm)]:
        tot, cnt = mp.mpf(0), 0
        for train, test in folds:
            beta = fit(cols, y, train)
            for i in test:
                tot += logit_lp(beta, cols, y, i)
                cnt += 1
        out[name] = candidate(tot / cnt, 1 * (1 + p), n)
    return out


def finish(scores):
    ranked = sorted(scores, key=lambda k: -scores[k]["score"])
    gap = scores[ranked[0]]["score"] - scores[ranked[1]]["score"]
    assert gap > 1e-6, ("winner too close to call", ranked, gap)
    return {"candidates": scores, "ranking": ranked, "selected": ranked[0], "gap": gap}


def main():
    rng = random.Random(41)
    q12 = lambda v: float(F(round(v * 4096), 4096))
    n = 48
    x1 = [q12(rng.uniform(-3, 3)) for _ in range(n)]
    x2 = [q12(rng.uniform(-2, 2)) for _ in range(n)]
    state, y = 0.0, []
    for a_, b_ in zip(x1, x2):
        state = 0.7 * state + rng.gauss(0, 0.5)
        y.append(q12(1.0 + 0.8 * a_ - 0.5 * b_ + state + rng.gauss(0, 0.2)))
    yb = []
    for a_, b_ in zip(x1, x2):
        z = 0.9 * a_ - 0.6 * b_ + 0.1
        yb.append(1.0 if rng.random() < 1 / (1 + 2.718281828459045 ** (-z)) else 0.0)

    if FOLDS_PATH == "--data":
        print(json.dumps({"x1": x1, "x2": x2, "y": y, "y_binary": yb}))
        return
    folds_in = json.load(open(FOLDS_PATH))
    cols = [[mpf(F(v)) for v in x1], [mpf(F(v)) for v in x2]]
    ymp = [mpf(F(v)) for v in y]
    ybmp = [mpf(F(v)) for v in yb]

    cont = {
        "shuffled": finish(continuous_scores(cols, ymp, shuffled(folds_in["continuous"]), False)),
        "time_ordered": finish(continuous_scores(cols, ymp, time_ordered(n, 2), True)),
    }
    binary = {
        "shuffled": finish(binary_scores(cols, ybmp, shuffled(folds_in["binary"]))),
        "time_ordered": finish(binary_scores(cols, ybmp, time_ordered(n, 1))),
    }
    print(json.dumps({
        "x1": x1, "x2": x2, "y": y, "y_binary": yb, "purge_continuous": 2, "purge_binary": 1,
        "folds": folds_in, "continuous": cont, "binary": binary,
    }, indent=1))


main()
