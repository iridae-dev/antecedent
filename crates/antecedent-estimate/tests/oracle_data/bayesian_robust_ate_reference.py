# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Cross-fitted weighted orthogonal ATE of the Bayesian-bootstrap route, for given weights and folds.

Regenerate in two stages (the row weights and fold ids are produced by the product's seeded
streams, so they are inputs of this reference, not outputs of it):

  python3 bayesian_robust_ate_reference.py inputs > inputs.json
  (the test's ignored dump test reads inputs.json and writes stream.json with the fold ids and
   the first three row-weight vectors of each case)
  uv run --script bayesian_robust_ate_reference.py final stream.json > bayesian_robust_ate_reference.json

For one weight vector w and fold ids f, every fold k is held out in turn.  On the training rows
the propensity is the maximiser of sum_i w_i [t_i log p_i + (1 - t_i) log(1 - p_i)] -
(lambda/2) sum_{j>=1} b_j^2 (Newton's method at 50 digits; the first column is the unpenalised
intercept); each arm's outcome regression is the weighted ridge solution of
(sum w x x' + lambda D) b = sum w x y over that arm's training rows, D = diag(0, 1, ..., 1).
On the held-out rows the propensity is clipped into [c, 1 - c] and the score is
psi_i = m1 - m0 + t (y - m1)/e - (1 - t)(y - m0)/(1 - e).  The draw is sum w_i psi_i / sum w_i.
Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import json
import sys


class Lcg:
    def __init__(self, s):
        self.s = s

    def u(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) % 2**64
        return (self.s >> 11) / float(1 << 53)


def make(name, n, p_cov, seed, folds, treat_share=0.5, offset=0.0, collinear=False, confound=1.0,
         clip=0.01, ridge=1.0, alternate=False, refused=False):
    g = Lcg(seed)
    covariates = []
    for j in range(p_cov):
        covariates.append([offset + (g.u() * 4 - 2) for _ in range(n)])
    if collinear and p_cov >= 2:
        covariates[1] = [covariates[0][i] + 1e-5 * (g.u() - 0.5) for i in range(n)]
    shift = {0.2: -1.4, 0.5: 0.0}.get(treat_share, 0.0)
    treatment, outcome = [], []
    for i in range(n):
        centred = covariates[0][i] - offset
        e = 1 / (1 + 2.718281828 ** (-(shift + confound * centred)))
        t = g.u() < e
        y = 1.0 + 2.0 * t + 0.7 * centred + 0.5 * (g.u() - 0.5)
        treatment.append(bool(t))
        outcome.append(y)
    if alternate:
        treatment = [i % 2 == 0 for i in range(n)]
    if all(treatment) or not any(treatment):
        treatment[0] = not treatment[0]
    ids = list(range(1000, 1000 + n))
    # a fixed permutation so the stable row ids are not in file order
    order = sorted(range(n), key=lambda i: (i * 7919 + seed) % 10007)
    ids = [ids[order[i]] for i in range(n)]
    return {"name": name, "row_ids": ids, "treatment": treatment, "outcome": outcome,
            "covariates": covariates, "folds": folds, "ridge": ridge, "clip": clip, "refused": refused}


CASES = [
    make("moderate", 40, 2, 1, 4),
    # three control rows in all: a training half holds one, fewer than the two coefficients of
    # the control outcome regression, so the product refuses instead of extrapolating
    make("too_few_control_rows_is_refused", 12, 1, 2, 2, refused=True),
    make("small_balanced_two_folds", 16, 1, 8, 2, alternate=True),
    make("unbalanced_arms", 60, 2, 3, 5, treat_share=0.2),
    make("covariate_offset", 40, 2, 4, 4, offset=1000.0),
    make("near_collinear_covariates", 40, 2, 5, 4, collinear=True),
    make("strong_confounding_clip_binds", 50, 1, 6, 5, confound=5.0, clip=0.05),
    make("three_covariates_weak_penalty", 45, 3, 7, 3, ridge=0.01),
]


def reference_draw(case, weights, fold_ids):
    import mpmath as mp

    mp.mp.dps = 50
    n = len(weights)
    lam = mp.mpf(case["ridge"])
    clip = mp.mpf(case["clip"])
    x = [[mp.mpf(1)] + [mp.mpf(c[i]) for c in case["covariates"]] for i in range(n)]
    t = case["treatment"]
    y = [mp.mpf(v) for v in case["outcome"]]
    w = [mp.mpf(v) for v in weights]
    p = len(x[0])

    def solve(a, b):
        return mp.lu_solve(mp.matrix(a), mp.matrix(b))

    def ridge_fit(rows):
        a = [[sum(w[i] * x[i][j] * x[i][k] for i in rows) + (lam if (j == k and j > 0) else 0) for k in range(p)] for j in range(p)]
        b = [sum(w[i] * x[i][j] * y[i] for i in rows) for j in range(p)]
        return list(solve(a, b))

    def logistic_fit(rows):
        beta = [mp.mpf(0)] * p
        for _ in range(100):
            grad = [mp.mpf(0)] * p
            hess = [[mp.mpf(0)] * p for _ in range(p)]
            for i in rows:
                eta = sum(beta[j] * x[i][j] for j in range(p))
                mu = 1 / (1 + mp.e ** (-eta))
                r = w[i] * ((1 if t[i] else 0) - mu)
                for j in range(p):
                    grad[j] += r * x[i][j]
                    for k in range(p):
                        hess[j][k] += w[i] * mu * (1 - mu) * x[i][j] * x[i][k]
            for j in range(1, p):
                grad[j] -= lam * beta[j]
                hess[j][j] += lam
            step = solve(hess, grad)
            beta = [beta[j] + step[j] for j in range(p)]
            if max(abs(s) for s in step) < mp.mpf(10) ** -40:
                break
        return beta

    e = [None] * n
    m0 = [None] * n
    m1 = [None] * n
    for k in sorted(set(fold_ids)):
        valid = [i for i in range(n) if fold_ids[i] == k]
        train = [i for i in range(n) if fold_ids[i] != k]
        beta = logistic_fit(train)
        b0 = ridge_fit([i for i in train if not t[i]])
        b1 = ridge_fit([i for i in train if t[i]])
        for i in valid:
            eta = sum(beta[j] * x[i][j] for j in range(p))
            raw = 1 / (1 + mp.e ** (-eta))
            e[i] = min(max(raw, clip), 1 - clip)
            m0[i] = sum(b0[j] * x[i][j] for j in range(p))
            m1[i] = sum(b1[j] * x[i][j] for j in range(p))
    total = mp.mpf(0)
    for i in range(n):
        ti = 1 if t[i] else 0
        psi = m1[i] - m0[i] + ti / e[i] * (y[i] - m1[i]) - (1 - ti) / (1 - e[i]) * (y[i] - m0[i])
        total += w[i] * psi
    return float(total / sum(w))


if sys.argv[1] == "inputs":
    print(json.dumps(CASES))
else:
    stream = json.load(open(sys.argv[2]))
    out = []
    for case, extra in zip(CASES, stream):
        case = dict(case)
        case["fold_ids"] = extra["fold_ids"]
        case["weights"] = extra["weights"]
        case["draws"] = [] if case["refused"] else [reference_draw(case, w, extra["fold_ids"]) for w in extra["weights"]]
        out.append(case)
    print(json.dumps(out))
