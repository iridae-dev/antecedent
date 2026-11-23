# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""50-digit reference for the cohort linear conditional-effect fit and its Student-t interval.

Regenerate with:  uv run --script cohort_linear_effect_reference.py > cohort_linear_effect_reference.json

Model: y_i = phi_i' a + t_i phi_i' c + e_i with phi_i = (1, x_i1, .., x_ik), fitted by ordinary
least squares on the design (phi_i, t_i phi_i) of width p = 2 (k + 1).  The reported effect
coefficients are c with the HC3 covariance
    V = (X'X)^-1 (sum_i x_i x_i' e_i^2 / (1 - h_i)^2) (X'X)^-1,   h_i = x_i' (X'X)^-1 x_i,
restricted to the c block.  (The product centres the features before fitting and maps the effect
block back; a reparameterization by an invertible affine map leaves the fit and the HC3 sandwich
unchanged, so this reference fits the uncentred design directly.)  For a cohort with mean basis
phibar the mean contrast is c' phibar with standard error sqrt(phibar' V phibar), and the two-sided
interval at level L is point +- t_{n-p}((1+L)/2) se, the Student-t quantile being inverted from the
regularized incomplete beta function.  Nothing here calls the product.  Repository-authored
(MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import json
import random

import mpmath as mp

mp.mp.dps = 50


def fit(y, t, feats):
    n, k = len(y), len(feats)
    basis = k + 1
    p = 2 * basis
    rows = []
    for i in range(n):
        phi = [mp.mpf(1)] + [mp.mpf(f[i]) for f in feats]
        rows.append(phi + [phi_j * mp.mpf(t[i]) for phi_j in phi])
    X = mp.matrix(rows)
    inv = mp.inverse(X.T * X)
    beta = inv * (X.T * mp.matrix([mp.mpf(v) for v in y]))
    meat = mp.zeros(p, p)
    for i in range(n):
        x = X[i, :].T
        e = mp.mpf(y[i]) - (X[i, :] * beta)[0]
        h = (x.T * inv * x)[0]
        meat += (x * x.T) * (e * e / (1 - h) ** 2)
    V = inv * meat * inv
    c = [beta[basis + a] for a in range(basis)]
    Vc = mp.matrix(basis, basis)
    for a in range(basis):
        for b in range(basis):
            Vc[a, b] = V[basis + a, basis + b]
    return c, Vc, p


def t_quantile(prob, df):
    prob, df = mp.mpf(prob), mp.mpf(df)
    tail = 1 - prob

    def upper_tail(x):
        return mp.betainc(df / 2, mp.mpf(1) / 2, 0, df / (df + x * x), regularized=True) / 2

    guess = mp.sqrt(2) * mp.erfinv(2 * prob - 1)
    return mp.findroot(lambda x: upper_tail(x) - tail, guess if guess > 0 else mp.mpf(1))


def main():
    rng = random.Random(20261028)
    out = {"fits": [], "interval": None, "t_quantiles": []}
    for name, n, k in [("no_modifier", 40, 0), ("one_modifier", 120, 1), ("two_modifiers", 50, 2)]:
        feats = [[round(4 * rng.random() - 2, 6) for _ in range(n)] for _ in range(k)]
        t = [float(rng.random() < 0.5) for _ in range(n)]
        y = []
        for i in range(n):
            base = 2.0 + sum(f[i] for f in feats)
            tau = 1.0 + sum(0.5 * f[i] for f in feats)
            scale = 0.3 + 0.4 * abs(feats[0][i]) if feats else 0.5
            y.append(round(base + t[i] * tau + rng.gauss(0, scale), 6))
        c, V, p = fit(y, t, feats)
        out["fits"].append({"name": name, "outcome": y, "treatment": t, "features": feats, "width": p,
                            "coefficients": [float(v) for v in c],
                            "covariance": [float(V[a, b]) for a in range(k + 1) for b in range(k + 1)]})
        if name == "one_modifier":
            cohort = feats[0][:60]
            phibar = [mp.mpf(1), sum(mp.mpf(v) for v in cohort) / len(cohort)]
            point = sum(c[a] * phibar[a] for a in range(2))
            se = mp.sqrt(sum(phibar[a] * V[a, b] * phibar[b] for a in range(2) for b in range(2)))
            df = n - p
            out["interval"] = {"cohort": cohort, "point": float(point), "se": float(se), "df": df,
                               "levels": [{"level": lv, "critical": float(t_quantile(0.5 + lv / 2, df)),
                                           "lower": float(point - t_quantile(0.5 + lv / 2, df) * se),
                                           "upper": float(point + t_quantile(0.5 + lv / 2, df) * se)}
                                          for lv in (0.9, 0.95, 0.99)]}
    for df in (1, 2, 5, 30, 116, 1000):
        for prob in (0.95, 0.975, 0.995):
            out["t_quantiles"].append({"df": df, "p": prob, "want": float(t_quantile(prob, df))})
    print(json.dumps(out, indent=1))


main()
