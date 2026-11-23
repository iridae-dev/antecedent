# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""40-digit reference for the joint vector-treatment OLS fit.

Regenerate with:  uv run --script vector_treatment_reference.py > vector_treatment_reference.json

A design of 48 rows with an intercept, two adjustment columns and three treatments (one of them
correlated with an adjustment column) and heteroskedastic noise.  The coefficient vector is the
ordinary least-squares solution b = (X'X)^-1 X'y evaluated at 40 digits from the exact binary64
inputs.  The covariance of the treatment block is, per kind,
  model based   s2 (X'X)^-1,              s2 = RSS / (n - p)
  HC0           (X'X)^-1 (sum e_i^2 x_i x_i') (X'X)^-1
  HC1           n / (n - p) times HC0
  HC2           weights e_i^2 / (1 - h_i)
  HC3           weights e_i^2 / (1 - h_i)^2,    h_i = x_i' (X'X)^-1 x_i.
A contrast w'b has variance w'Vw; the naive standard error drops the off-diagonals; the two-sided
normal p-value is erfc(|z| / sqrt 2); the Holm adjustment is the running maximum of
min(1, (m - rank) p_(rank)) over ascending p-values; the joint Wald statistic of R b = 0 is
(Rb)'(RVR')^-1(Rb) with the upper-tail chi-square p-value Q(m / 2, x / 2).
Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import json

import mpmath as mp

mp.mp.dps = 40


class Lcg:
    def __init__(self, s):
        self.s = s

    def u(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) % 2**64
        return (self.s >> 11) / float(1 << 53)

    def normal(self):
        return sum(self.u() for _ in range(12)) - 6.0


def data():
    rng = Lcg(20261028)
    n = 48
    a1 = [rng.normal() for _ in range(n)]
    a2 = [1.0 if rng.u() < 0.4 else 0.0 for _ in range(n)]
    t1 = [0.6 * a + rng.normal() for a in a1]
    t2 = [rng.normal() + 0.5 * b for b in a2]
    t3 = [1.0 if rng.u() < 0.5 else 0.0 for _ in range(n)]
    y = []
    for i in range(n):
        noise = (0.4 + 0.6 * abs(a1[i])) * rng.normal()
        y.append(1.0 + 0.8 * a1[i] - 0.5 * a2[i] + 1.5 * t1[i] - 0.7 * t2[i] + 0.3 * t3[i] + noise)
    return n, a1, a2, [t1, t2, t3], y


def solve(A, b):
    return mp.lu_solve(A, b)


def fit(n, adj, treat, y, kind):
    cols = [[1.0] * n] + adj + treat
    p = len(cols)
    X = mp.matrix(n, p)
    for j, c in enumerate(cols):
        for i in range(n):
            X[i, j] = mp.mpf(c[i])
    Y = mp.matrix([mp.mpf(v) for v in y])
    G = X.T * X
    Ginv = G ** -1
    b = Ginv * (X.T * Y)
    e = Y - X * b
    h = [(X[i, :] * Ginv * X[i, :].T)[0] for i in range(n)]
    if kind == "model_based":
        s2 = sum(v * v for v in e) / (n - p)
        V = s2 * Ginv
    else:
        M = mp.zeros(p, p)
        for i in range(n):
            if kind == "hc0":
                w = 1
            elif kind == "hc1":
                w = mp.mpf(n) / (n - p)
            elif kind == "hc2":
                w = 1 / (1 - h[i])
            else:
                w = 1 / (1 - h[i]) ** 2
            xi = X[i, :].T
            M += w * e[i] ** 2 * (xi * xi.T)
        V = Ginv * M * Ginv
    s2 = sum(v * v for v in e) / (n - p)
    t0 = 1 + len(adj)
    k = len(treat)
    bt = [b[t0 + i] for i in range(k)]
    Vt = mp.matrix(k, k)
    for i in range(k):
        for j in range(k):
            Vt[i, j] = (V[t0 + i, t0 + j] + V[t0 + j, t0 + i]) / 2
    return bt, Vt, s2


def pnorm(z):
    return mp.erfc(abs(z) / mp.sqrt(2))


def wald(R, b, V):
    Rm = mp.matrix(R)
    rb = Rm * mp.matrix(b)
    m = len(R)
    S = Rm * V * Rm.T
    stat = (rb.T * (S ** -1) * rb)[0]
    p = mp.gammainc(mp.mpf(m) / 2, stat / 2, mp.inf, regularized=True)
    return stat, p


def holm(ps):
    m = len(ps)
    order = sorted(range(m), key=lambda i: (ps[i], i))
    out = [None] * m
    run = mp.mpf(0)
    for rank, i in enumerate(order):
        run = max(run, min(mp.mpf(1), (m - rank) * ps[i]))
        out[i] = run
    return out


def s(x, d=30):
    return mp.nstr(x, d)


def main():
    n, a1, a2, treat, y = data()
    contrasts = [[1, -1, 0], [1, 0, -1], [0, 1, 1]]
    kinds = {}
    for kind in ("model_based", "hc0", "hc1", "hc2", "hc3"):
        b, V, s2 = fit(n, [a1, a2], treat, y, kind)
        k = len(b)
        se = [mp.sqrt(V[i, i]) for i in range(k)]
        coefs = []
        for i in range(k):
            z = b[i] / se[i]
            coefs.append({"estimate": s(b[i]), "se": s(se[i]), "z": s(z), "p": s(pnorm(z))})
        cs = []
        raw = []
        for w in contrasts:
            est = sum(w[i] * b[i] for i in range(k))
            var = sum(w[i] * V[i, j] * w[j] for i in range(k) for j in range(k))
            naive = sum(w[i] ** 2 * V[i, i] for i in range(k))
            z = est / mp.sqrt(var)
            raw.append(pnorm(z))
            cs.append({"weights": w, "estimate": s(est), "se": s(mp.sqrt(var)),
                       "naive_se": s(mp.sqrt(naive)), "z": s(z), "p": s(pnorm(z))})
        for c, a in zip(cs, holm(raw)):
            c["p_holm"] = s(a)
        stat, pv = wald([[1 if i == j else 0 for j in range(k)] for i in range(k)], b, V)
        stat2, pv2 = wald(contrasts[:2], b, V)
        kinds[kind] = {
            "residual_variance": s(s2),
            "covariance": [[s(V[i, j]) for j in range(k)] for i in range(k)],
            "coefficients": coefs,
            "contrasts": cs,
            "joint_wald": {"statistic": s(stat), "p": s(pv)},
            "contrast_wald": {"statistic": s(stat2), "p": s(pv2)},
        }
    print(json.dumps({"n": n, "adjustment": [a1, a2], "treatments": treat, "outcome": y,
                      "contrasts": contrasts, "kinds": kinds}))


main()
