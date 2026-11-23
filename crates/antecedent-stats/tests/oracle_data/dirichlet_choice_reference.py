# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Reference values for the Dirichlet categorical-choice posterior of the adaptive-inference module.

Regenerate with:  uv run --script dirichlet_choice_reference.py > dirichlet_choice_reference.json

Written from the definitions with mpmath (a different algorithm from the product's Gauss-Legendre
panels over the independent-gamma representation). With theta ~ Dirichlet(alpha), theta_k = G_k / S
for independent G_k ~ Gamma(alpha_k, 1) and S = sum G, and S independent of theta, so
  P(option k has the largest share) = P(G_k = max_j G_j) = int g_k(x) prod_{j != k} F_j(x) dx
  E[max_j theta_j]                  = E[max_j G_j] / sum alpha = (int_0^inf 1 - prod F_j) / sum alpha
  loss_k                            = E[max_j theta_j] - alpha_k / sum alpha
  ln ML(n)  (ordered sequence)      = lnGamma(A) - lnGamma(A + N) + sum_k [lnGamma(alpha_k + n_k) - lnGamma(alpha_k)]
Here F_j is the regularized lower incomplete gamma, computed by mp.gammainc.

Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import json

import mpmath as mp

mp.mp.dps = 20
mpf = mp.mpf


def pdf(a, x):
    return mp.exp((a - 1) * mp.log(x) - x - mp.loggamma(a)) if x > 0 else mpf(0)


def cdf(a, x):
    return mp.gammainc(a, 0, x, regularized=True) if x > 0 else mpf(0)


def breakpoints(alphas):
    pts = {mpf(0)}
    for a in alphas:
        sd = mp.sqrt(a)
        for k in (-6, -3, -1, 0, 1, 3, 6, 12):
            x = a + k * sd
            if x > 0:
                pts.add(x)
    hi = max(a + 40 * mp.sqrt(a) + 40 for a in alphas)
    pts.add(hi)
    return sorted(pts)


def best(alphas):
    out = []
    for k in range(len(alphas)):
        def f(x, k=k):
            v = pdf(alphas[k], x)
            for j, a in enumerate(alphas):
                if j != k:
                    v *= cdf(a, x)
            return v
        out.append(mp.quad(f, breakpoints(alphas)))
    return out


def losses(alphas):
    total = sum(alphas)
    emax = mp.quad(lambda x: 1 - mp.fprod(cdf(a, x) for a in alphas), breakpoints(alphas))
    share = emax / total
    return [share - a / total for a in alphas], share


def log_ml(alphas, counts):
    big_a, big_n = sum(alphas), sum(counts)
    return (mp.loggamma(big_a) - mp.loggamma(big_a + big_n)
            + sum(mp.loggamma(a + n) - mp.loggamma(a) for a, n in zip(alphas, counts)))


CASES = [
    [1, 1, 1],
    [2, 5],
    [3, 3, 3, 3],
    [1.5, 4.25, 9],
    [11, 8, 20, 5],
    [40, 35, 30],
    [1, 1, 1, 8],
    [250, 240, 260],
    [1, 1, 1, 1, 1, 1, 1, 1],
]
ML = [
    ([1, 1, 1], [3, 0, 2]),
    ([0.5, 0.5, 0.5, 0.5], [10, 4, 0, 1]),
    ([2, 3], [7, 5]),
    ([1, 1, 1, 1, 1], [100, 80, 60, 40, 20]),
]


def num(v):
    return mp.nstr(v, 17)


out = {"best": [], "losses": [], "log_ml": []}
for case in CASES:
    al = [mpf(str(a)) for a in case]
    out["best"].append({"alphas": case, "probabilities": [num(p) for p in best(al)]})
    ls, share = losses(al)
    out["losses"].append({"alphas": case, "expected_max_share": num(share), "losses": [num(v) for v in ls]})
for al, cn in ML:
    out["log_ml"].append({"alphas": al, "counts": cn, "value": num(log_ml([mpf(str(a)) for a in al], cn))})
print(json.dumps(out, indent=1))
