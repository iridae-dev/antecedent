# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Categorical and binary-logit mechanism fits against exact maximum likelihood, in 50 digits.

Regenerate with:  uv run --script registry_glm_reference.py > registry_glm_reference.json

Nothing here iterates a Fisher scoring loop shaped like the library's.  Each fit is the unique
maximiser of a concave log-likelihood, found by Newton's method on the raw (unstandardised)
design to a gradient below 1e-40 in 50-digit arithmetic:

  multinomial   l(B) = sum_i w_i log softmax(B x_i)[y_i], category 0 the reference with zero
                logits.  Weights are rescaled to mean one by the refit, which cannot move the
                maximiser (a common factor on every weight), and the reference does not rescale.
  hierarchical  maximise  sum_i [ y_i eta_i - log(1 + e^eta_i) ] - lambda/2 * sum_{j>=1} beta_j^2
                with the intercept unpenalised. An unpenalised logit pilot supplies b and
                observed Fisher information F = X' diag(mu_i (1 - mu_i)) X. The raw-slope
                prior variance is tau2 = max(mean(b_j^2) - mean((F^-1)_jj), 1e-8), and
                lambda = clamp(1/tau2, 1e-4, 1e3), in the summed-likelihood convention.

The fixed designs include a parent measured in years (offset 2020, spread 3) beside a unit-scale
parent, weights spanning three orders of magnitude, a binary outcome (the two-category branch), a
balanced zero-signal outcome that drives lambda to its upper clamp and others that leave it interior.
Repository-authored (MIT OR Apache-2.0).
"""
import json
import random
from fractions import Fraction as F

import mpmath as mp

mp.mp.dps = 50
TOL = mp.mpf(10) ** -40


def mpf(f):
    f = F(f)
    return mp.mpf(f.numerator) / f.denominator


def softmax_probs(B, x):
    eta = [mp.mpf(0)] + [sum(B[k][j] * x[j] for j in range(len(x))) for k in range(len(B))]
    m = max(eta)
    e = [mp.e ** (v - m) for v in eta]
    s = sum(e)
    return [v / s for v in e]


def multinomial_mle(X, cat, K, w):
    n, p = len(X), len(X[0])
    nfree = (K - 1) * p
    beta = mp.zeros(nfree, 1)
    for _ in range(200):
        B = [[beta[k * p + j] for j in range(p)] for k in range(K - 1)]
        g = mp.zeros(nfree, 1)
        H = mp.zeros(nfree, nfree)
        for i in range(n):
            pr = softmax_probs(B, X[i])
            for k in range(K - 1):
                r = (1 if cat[i] == k + 1 else 0) - pr[k + 1]
                for j in range(p):
                    g[k * p + j] += w[i] * r * X[i][j]
                for l in range(K - 1):
                    c = pr[k + 1] * ((1 if k == l else 0) - pr[l + 1])
                    for j in range(p):
                        for m_ in range(p):
                            H[k * p + j, l * p + m_] += w[i] * c * X[i][j] * X[i][m_]
        if mp.norm(g, p=mp.inf) < TOL:
            break
        beta = beta + mp.lu_solve(H, g)
    B = [[beta[k * p + j] for j in range(p)] for k in range(K - 1)]
    return [[mp.mpf(0)] * p] + B


def ridge_logit_mle(X, y, lam):
    n, p = len(X), len(X[0])
    beta = mp.zeros(p, 1)
    for _ in range(200):
        g = mp.zeros(p, 1)
        H = mp.zeros(p, p)
        for i in range(n):
            eta = sum(beta[j] * X[i][j] for j in range(p))
            mu = 1 / (1 + mp.e ** (-eta))
            for j in range(p):
                g[j] += (y[i] - mu) * X[i][j]
                for m_ in range(p):
                    H[j, m_] += mu * (1 - mu) * X[i][j] * X[i][m_]
        for j in range(1, p):
            g[j] -= lam * beta[j]
            H[j, j] += lam
        if mp.norm(g, p=mp.inf) < TOL:
            break
        beta = beta + mp.lu_solve(H, g)
    else:
        raise ValueError("logistic oracle did not converge in 200 iterations")
    return [beta[j] for j in range(p)]


def exact_solve(A, b):
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        piv = next(r for r in range(c, n) if M[r][c] != 0)
        M[c], M[piv] = M[piv], M[c]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c] / M[c][c]
                M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [M[i][n] / M[i][i] for i in range(n)]


def eb_lambda(cols, y):
    """EB strength from an independent high-precision unpenalised logit and Fisher inverse."""
    n = len(y)
    X = [[mp.mpf(1)] + [mpf(c[i]) for c in cols] for i in range(n)]
    outcome = [mpf(F(v)) for v in y]
    beta = ridge_logit_mle(X, outcome, mp.mpf(0))
    k = len(beta)
    fisher = mp.zeros(k, k)
    for row in X:
        eta = sum(beta[j] * row[j] for j in range(k))
        mu = 1 / (1 + mp.exp(-eta))
        for a in range(k):
            for b in range(k):
                fisher[a, b] += mu * (1 - mu) * row[a] * row[b]
    inverse = fisher ** -1
    p = k - 1
    tau2 = max(sum(b*b for b in beta[1:]) / p - sum(inverse[j,j] for j in range(1,k)) / p,
               mp.mpf("1e-8"))
    return min(max(1 / tau2, mp.mpf("1e-4")), mp.mpf("1e3"))


def fl(v):
    return float(v)


def discrete_case(name, x_cols, y, K, weights):
    n = len(y)
    X = [[mp.mpf(1)] + [mpf(c[i]) for c in x_cols] for i in range(n)]
    support = sorted(set(y))
    cat = [support.index(v) for v in y]
    assert len(support) == K, (name, support)
    out = {
        "name": name, "kind": "discrete", "x": [[fl(v) for v in c] for c in x_cols], "y": y,
        "support": support,
    }
    ones = [mp.mpf(1)] * n
    B = multinomial_mle(X, cat, K, ones)
    out["logits"] = [[fl(v) for v in row] for row in B]
    out["probs"] = [fl(sum(1 for c in cat if c == k) / n) for k in range(K)]
    if weights is not None:
        wmp = [mpf(F(v)) for v in weights]
        Bw = multinomial_mle(X, cat, K, wmp)
        out["weights"] = weights
        out["weighted_logits"] = [[fl(v) for v in row] for row in Bw]
        tot = sum(wmp)
        out["weighted_probs"] = [fl(sum(wmp[i] for i in range(n) if cat[i] == k) / tot) for k in range(K)]
    return out


def hierarchical_case(name, x_cols, y):
    n = len(y)
    lam = eb_lambda(x_cols, y)
    X = [[mp.mpf(1)] + [mpf(c[i]) for c in x_cols] for i in range(n)]
    beta = ridge_logit_mle(X, [mpf(F(v)) for v in y], lam)
    return {
        "name": name, "kind": "hierarchical", "x": [[fl(v) for v in c] for c in x_cols], "y": y,
        "lambda": fl(lam), "lambda_regime": "upper_clamp" if lam >= mp.mpf("1e3") else (
            "lower_clamp" if lam <= mp.mpf("1e-4") else "interior"),
        "coefficients": [fl(b) for b in beta],
        "p1": fl(sum(1 for v in y if v == 1.0) / n),
    }


def moments(col):
    n = len(col)
    m = sum(col) / n
    sd = mp.sqrt(sum((v - m) ** 2 for v in col) / n)
    return m, (sd if sd > mp.mpf("1e-12") else mp.mpf(1))


def basis_columns(cols, n, kind):
    """Columns (without the intercept) of the interaction or cubic-spline expansion."""
    ms = [moments(c) for c in cols]
    z = [[(c[i] - ms[j][0]) / ms[j][1] for i in range(n)] for j, c in enumerate(cols)]
    if kind == "interactions":
        return [z[0], z[1], [z[0][i] * z[1][i] for i in range(n)]]
    knots = []
    for j in range(2):
        vals = sorted(z[j])
        ks = []
        for q in (mp.mpf("0.25"), mp.mpf("0.5"), mp.mpf("0.75")):
            k = vals[int(mp.floor((len(vals) - 1) * q + mp.mpf("0.5")))]
            if not ks or abs(k - ks[-1]) >= mp.mpf("1e-9"):
                ks.append(k)
        assert len(ks) == 3
        knots.append(ks)
    per = []
    for j in range(2):
        cs = [z[j], [v ** 2 for v in z[j]], [v ** 3 for v in z[j]]]
        for k in knots[j]:
            cs.append([(v - k) ** 3 if v - k > 0 else mp.mpf(0) for v in z[j]])
        per.append(cs)
    terms = per[0] + per[1]
    for right in per[1]:
        terms.append([per[0][0][i] * right[i] for i in range(n)])
    for r, right in enumerate(per[0]):
        if r == 0:
            continue
        terms.append([per[1][0][i] * right[i] for i in range(n)])
    return terms


def basis_case(name, kind, x_cols, y, K, weights):
    n = len(y)
    cols = [[mpf(F(v)) for v in c] for c in x_cols]
    terms = basis_columns(cols, n, kind)
    X = [[mp.mpf(1)] + [t[i] for t in terms] for i in range(n)]
    support = sorted(set(y))
    assert len(support) == K
    cat = [support.index(v) for v in y]
    out = {"name": name, "kind": "discrete_" + kind, "x": [[fl(v) for v in c] for c in x_cols], "y": y,
           "support": support, "n_terms": len(terms)}
    B = multinomial_mle(X, cat, K, [mp.mpf(1)] * n)
    out["logits"] = [[fl(v) for v in row] for row in B]
    if weights is not None:
        wmp = [mpf(F(v)) for v in weights]
        Bw = multinomial_mle(X, cat, K, wmp)
        out["weights"] = weights
        out["weighted_logits"] = [[fl(v) for v in row] for row in Bw]
    return out


def main():
    rng = random.Random(11)

    def grid(lo, hi, den):
        return F(rng.randint(lo * den, hi * den), den)

    def categorical(n, K, shift=0, scale=1):
        x1 = [grid(-3, 3, 16) for _ in range(n)]
        x2 = [grid(-2, 2, 16) for _ in range(n)]
        cat = []
        for a, b in zip(x1, x2):
            eta = [0.0, 0.9 * float(a) - 0.5, -0.8 * float(b) + 0.3, 0.6 * float(a) - 0.7 * float(b)][:K]
            m = max(eta)
            pr = [2.718281828459045 ** (v - m) for v in eta]
            s = sum(pr)
            u = rng.random() * s
            acc = 0.0
            pick = K - 1
            for k, v in enumerate(pr):
                acc += v
                if u <= acc:
                    pick = k
                    break
            cat.append(pick)
        return x1, x2, cat

    cases = []
    n = 45
    x1, x2, cat = categorical(n, 3)
    w = [float(2.0 ** rng.choice([-5, -2, 0, 1, 3, 5])) for _ in range(n)]
    cases.append(discrete_case("three_categories", [[float(v) for v in x1], [float(v) for v in x2]],
                               [float(c) for c in cat], 3, w))
    # parent measured in years beside a unit-scale parent: the raw-scale logits are recovered exactly
    years = [float(F(2020) + F(round(float(v) * 16), 16) * 3) for v in x1]
    cases.append(discrete_case("years_parent", [years, [float(v) for v in x2]], [float(c) for c in cat], 3, None))
    n = 40
    x1, x2, cat = categorical(n, 2)
    w = [float(rng.choice([1, 2, 3, 5, 8])) for _ in range(n)]
    cases.append(discrete_case("two_categories", [[float(v) for v in x1], [float(v) for v in x2]],
                               [float(c) for c in cat], 2, w))
    x1, x2, cat = categorical(60, 4)
    cases.append(discrete_case("four_categories", [[float(v) for v in x1], [float(v) for v in x2]],
                               [float(c) for c in cat], 4, [float(rng.choice([1, 2, 4])) for _ in range(60)]))

    # multinomial logit on the interaction and the spline expansions of the parents
    for kind, n, K in [("interactions", 90, 3), ("spline", 320, 2)]:
        x1, x2, cat = categorical(n, K)
        # soften the labels so the larger expansion is not separated
        cat = [c if rng.random() < 0.85 else rng.randrange(K) for c in cat]
        wts = [float(rng.choice([1, 2, 3])) for _ in range(n)]
        cases.append(basis_case("basis_" + kind, kind, [[float(v) for v in x1], [float(v) for v in x2]],
                                [float(c) for c in cat], K, wts))

    # hierarchical binary logit: interior lambda, upper clamp
    n = 44
    x1 = [grid(-3, 3, 16) for _ in range(n)]
    x2 = [grid(-2, 2, 16) for _ in range(n)]
    yb = []
    for a, b in zip(x1, x2):
        z = 1.1 * float(a) - 0.7 * float(b) + 0.2
        yb.append(1.0 if rng.random() < 1 / (1 + 2.718281828459045 ** (-z)) else 0.0)
    cases.append(hierarchical_case("signal", [[float(v) for v in x1], [float(v) for v in x2]], yb))
    yw = [1.0 if rng.random() < 0.5 else 0.0 for _ in range(n)]
    cases.append(hierarchical_case("weak_signal", [[float(v) for v in x1], [float(v) for v in x2]], yw))
    # Every predictor row occurs with both binary outcomes: the unpenalized
    # logit has exactly zero slopes, so the EB variance floor forces the cap.
    balanced_x = [[float(v) for v in col for _ in range(2)] for col in (x1, x2)]
    balanced_y = [float(k) for _ in range(n) for k in (0, 1)]
    cases.append(hierarchical_case("balanced_zero_signal", balanced_x, balanced_y))
    off = [float(F(2020) + v * 3) for v in x1]
    cases.append(hierarchical_case("years_parent", [off, [float(v) for v in x2]], yb))
    print(json.dumps(cases, indent=1))


main()
