# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""One cross-fit fold of a ridge- or lasso-penalised logistic propensity with inner-CV penalty choice.

Regenerate with:  uv run --script penalized_propensity_reference.py [inner.json] > penalized_propensity_reference.json

The inner fold ids of each case are recorded (the seeded plan is a separate kernel; the in-crate
test asserts the product's plan equals them).  Per case, on the training rows only:
  1. every non-intercept column is centred by its (weighted) training mean and divided by its
     (weighted, population) standard deviation; a column whose standard deviation is at most
     1e-12 max(|mean|, 1) is zeroed.  The validation rows use the training mean and scale.
  2. for each penalty lambda of the ascending grid, the penalised logistic model is fitted on the
     inner training folds and its out-of-fold predictions score the weighted mean log loss
     (probabilities clipped to [1e-12, 1 - 1e-12]); objectives, on a SUM log likelihood with the
     intercept unpenalised:
         ridge  -loglik + (lambda / 2) |b|^2          (Newton's method, float64)
         lasso  -loglik + lambda |b|_1                (proximal Newton with cyclic coordinate
                descent; accepted only when the KKT conditions of the exact objective hold to 1e-10)
  3. the lambda with the smallest inner loss is chosen, a later (larger) lambda winning an exact tie;
     the model is refitted on all training rows and predicts the validation rows.
Outputs: the selected lambda, the inner losses, the validation probabilities and, for the lasso,
the selected support.  Repository-authored (MIT OR Apache-2.0).
"""
import json
import sys
import numpy as np

rng = np.random.default_rng(20261028)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def ridge_fit(X, y, w, lam):
    n, p = X.shape
    beta = np.zeros(p)
    pen = np.full(p, lam)
    pen[0] = 0.0
    for _ in range(200):
        pr = sigmoid(X @ beta)
        g = X.T @ (w * (y - pr)) - pen * beta
        H = (X * (w * pr * (1 - pr))[:, None]).T @ X + np.diag(pen)
        step = np.linalg.solve(H, g)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-14:
            break
    return beta


def soft(z, t):
    return np.sign(z) * np.maximum(np.abs(z) - t, 0.0)


def lasso_fit(X, y, w, lam):
    n, p = X.shape
    beta = np.zeros(p)
    for _ in range(500):
        eta = X @ beta
        pr = sigmoid(eta)
        W = np.maximum(w * pr * (1 - pr), 1e-12)
        z = eta + w * (y - pr) / W
        # weighted least squares + L1 by cyclic coordinate descent, to convergence
        b = beta.copy()
        for _ in range(20000):
            old = b.copy()
            for j in range(p):
                r = z - X @ b + X[:, j] * b[j]
                num = np.sum(W * X[:, j] * r)
                den = np.sum(W * X[:, j] ** 2)
                b[j] = num / den if j == 0 else (soft(num, lam) / den if den > 0 else 0.0)
            if np.max(np.abs(b - old)) < 1e-15:
                break
        step = b - beta
        # backtracking on the exact objective
        def obj(bb):
            e = X @ bb
            return np.sum(w * (np.logaddexp(0, e) - y * e)) + lam * np.sum(np.abs(bb[1:]))
        t = 1.0
        while obj(beta + t * step) > obj(beta) + 1e-14 and t > 1e-12:
            t /= 2
        beta = beta + t * step
        if np.max(np.abs(t * step)) < 1e-14:
            break
    grad = X.T @ (w * (y - sigmoid(X @ beta)))
    kkt = abs(grad[0])
    for j in range(1, p):
        kkt = max(kkt, abs(grad[j]) - lam if beta[j] == 0 else abs(grad[j] - lam * np.sign(beta[j])))
    assert kkt < 1e-9, kkt
    return beta


def fit(kind, X, y, w, lam):
    return ridge_fit(X, y, w, lam) if kind == "ridge" else lasso_fit(X, y, w, lam)


def standardise(train, valid, w):
    train, valid = train.copy(), valid.copy()
    for c in range(1, train.shape[1]):
        col = train[:, c]
        mass = w.sum()
        mean = np.sum(w * col) / mass
        sd = np.sqrt(np.sum(w * (col - mean) ** 2) / mass)
        scale = 1.0 / sd if sd > 1e-12 * max(abs(mean), 1.0) else 0.0
        train[:, c] = (col - mean) * scale
        valid[:, c] = (valid[:, c] - mean) * scale
    return train, valid


def run(case):
    kind = case["kind"]
    train = np.array(case["design_train"]).T if False else np.array(case["train_rows"])
    valid = np.array(case["valid_rows"])
    t = np.array(case["t_train"])
    w = np.array(case["weights"]) if case["weights"] is not None else np.ones(len(t))
    tr, va = standardise(train, valid, w)
    inner = np.array(case["inner_folds"])
    K = int(inner.max()) + 1
    best = None
    losses = []
    for lam in case["lambdas"]:
        oof = np.zeros(len(t))
        for f in range(K):
            m = inner != f
            b = fit(kind, tr[m], t[m], w[m], lam)
            oof[~m] = sigmoid(tr[~m] @ b)
        p = np.clip(oof, 1e-12, 1 - 1e-12)
        loss = np.sum(w * -(t * np.log(p) + (1 - t) * np.log(1 - p))) / w.sum()
        losses.append(float(loss))
        if best is None or loss <= best[1]:
            best = (lam, loss)
    b = fit(kind, tr, t, w, best[0])
    out = {"lambda": best[0], "inner_losses": losses, "valid_probability": [float(v) for v in sigmoid(va @ b)],
           "coefficients": [float(v) for v in b]}
    if kind == "lasso":
        out["support"] = [j for j in range(1, len(b)) if b[j] != 0.0]
    return out


def make(name, kind, n, nv, p, lambdas, weighted=False, offset=0.0, scale=1.0, constant_col=False, collinear=False):
    def rows(m):
        X = np.ones((m, 1 + p))
        for j in range(p):
            X[:, 1 + j] = offset + scale * rng.normal(size=m)
        if collinear and p >= 2:
            X[:, 2] = X[:, 1] + 0.01 * scale * rng.normal(size=m)
        if constant_col and p >= 1:
            X[:, p] = 7.0
        return X
    X, Xv = rows(n), rows(nv)
    z = 0.9 * (X[:, 1] - offset) / scale - 0.5 * (X[:, 2] - offset) / scale if p >= 2 else 0.9 * (X[:, 1] - offset) / scale
    t = (rng.uniform(size=n) < sigmoid(z)).astype(float)
    w = rng.uniform(0.2, 5.0, size=n) if weighted else None
    return {"name": name, "kind": kind, "train_rows": X.tolist(), "valid_rows": Xv.tolist(),
            "t_train": t.tolist(), "weights": None if w is None else w.tolist(), "lambdas": lambdas,
            "inner_k": 5, "seed": 5, "fold": 0}


if __name__ == "__main__":
    recorded = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else {}
    cases = [
        make("ridge, three covariates, default grid", "ridge", 80, 30, 3, [1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3]),
        make("lasso, five columns with a constant and a near-collinear pair", "lasso", 100, 30, 5,
             [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0], constant_col=True, collinear=True),
        make("weighted ridge", "ridge", 70, 25, 2, [1e-1, 1.0, 1e1, 1e2], weighted=True),
        make("ridge, covariate offset 1000 at scale 50", "ridge", 80, 25, 2, [1e-1, 1.0, 1e1, 1e2],
             offset=1000.0, scale=50.0),
    ]
    for case in cases:
        inner = recorded.get(case["name"])
        case["inner_folds"] = inner
        if inner is not None:
            case["expected"] = run(case)
    print(json.dumps({"cases": cases}))
