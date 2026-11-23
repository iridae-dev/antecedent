# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Reference values for the conjugate count posteriors and the power-prior discount selection.

Regenerate with:  uv run --script conjugate_counts_reference.py > conjugate_counts_reference.json

Written from the stated models at 50 digits, with no Antecedent code.

* Beta-Binomial: prior Beta(a, b), k successes in n trials.  Posterior Beta(a + k, b + n - k);
  CDF is the regularized incomplete beta; quantiles solve CDF(x) = p; the marginal likelihood of
  the count is C(n, k) B(a + k, b + n - k) / B(a, b); the posterior predictive of j successes in
  m trials is C(m, j) B(a' + j, b' + m - j) / B(a', b') with (a', b') the posterior.
* Gamma-Poisson: prior Gamma(shape a, rate b), counts y_i over exposures t_i.  Posterior
  Gamma(a + sum y, b + sum t); the marginal likelihood is
  prod_i t_i^y_i / y_i! * b^a / Gamma(a) * Gamma(a + sum y) / (b + sum t)^(a + sum y); the
  predictive count over exposure t is negative binomial.
* Empirical-Bayes discount: the power prior at discount alpha has source counts scaled by alpha;
  the score of alpha is the log marginal likelihood of the target data under it, and the
  maximizer over the stated grid (smallest alpha on a tie) is reported with its continuous
  refinement by golden-section search.
* Linear target: the composed power-path prior is N(m(alpha), sigma^2 V(alpha)) with precision
  P = P_baseline + alpha P_source and m = P^-1 (P_baseline m_b + alpha P_source m_s); the marginal
  law of y is N(X m, sigma^2 (I + X V X')) for a known sigma^2, and a multivariate t with
  2 a0 degrees of freedom, location X m and scale (b0 / a0)(I + X V X') for an Inv-Gamma(a0, b0)
  residual variance.  Scores are the log densities of y; the functional's posterior mean is
  c' (P + X'X)^-1 (P m + X'y).

Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""

import json
import random

import mpmath as mp

mp.mp.dps = 50
mpf = mp.mpf
rng = random.Random(20261116)

GRID = [mpf(i) / 20 for i in range(21)]


def f(x):
    return float(x)


def beta_cdf(a, b, x):
    return mp.betainc(a, b, 0, x, regularized=True)


def bisect(cdf, lo, hi, p):
    for _ in range(260):
        mid = (lo + hi) / 2
        if cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def beta_quantile(a, b, p):
    return bisect(lambda x: beta_cdf(a, b, x), mpf(0), mpf(1), p)


def gamma_cdf(a, b, x):
    return mp.gammainc(a, 0, b * x, regularized=True)


def gamma_quantile(a, b, p):
    hi = a / b + 40 * mp.sqrt(a) / b + 10
    return bisect(lambda x: gamma_cdf(a, b, x), mpf(0), hi, p)


def ln_choose(n, k):
    return mp.log(mp.binomial(n, k))


def ln_beta(a, b):
    return mp.log(mp.beta(a, b))


def bb_log_ml(a, b, k, n):
    return ln_choose(n, k) + ln_beta(a + k, b + n - k) - ln_beta(a, b)


def gp_log_ml(a, b, counts, exposures):
    s = sum(counts)
    e = sum(exposures)
    kernel = sum(y * mp.log(t) - mp.log(mp.factorial(y)) for y, t in zip(counts, exposures))
    return a * mp.log(b) - mp.log(mp.gamma(a)) + mp.log(mp.gamma(a + s)) - (a + s) * mp.log(b + e) + kernel


PROBS = [mpf("1e-6"), mpf("0.025"), mpf("0.25"), mpf("0.5"), mpf("0.975"), 1 - mpf("1e-6")]


def beta_case(a, b, k, n, m):
    a1, b1 = a + k, b + n - k
    out = {
        "prior": [f(a), f(b)],
        "successes": k,
        "trials": n,
        "posterior": [f(a1), f(b1)],
        "mean": f(a1 / (a1 + b1)),
        "variance": f(a1 * b1 / ((a1 + b1) ** 2 * (a1 + b1 + 1))),
        "probs": [f(p) for p in PROBS],
        "quantiles": [f(beta_quantile(a1, b1, p)) for p in PROBS],
        "cdf_at": [[f(x), f(beta_cdf(a1, b1, x))] for x in (mpf("0.05"), mpf("0.3"), mpf("0.5"), mpf("0.9"))],
        "log_marginal_likelihood": f(bb_log_ml(a, b, k, n)),
        "predictive_trials": m,
        "predictive_pmf": [f(mp.exp(bb_log_ml(a1, b1, j, m))) for j in range(m + 1)],
        "predictive_mean": f(m * a1 / (a1 + b1)),
        "predictive_variance": f(m * a1 * b1 * (a1 + b1 + m) / ((a1 + b1) ** 2 * (a1 + b1 + 1))),
    }
    return out


def gamma_case(a, b, counts, exposures, t_new, ymax):
    s = sum(counts)
    e = sum(exposures)
    a1, b1 = a + s, b + e
    pmf = []
    for y in range(ymax + 1):
        pmf.append(
            f(
                mp.exp(
                    mp.log(mp.gamma(y + a1))
                    - mp.log(mp.gamma(a1))
                    - mp.log(mp.factorial(y))
                    + a1 * mp.log(b1 / (b1 + t_new))
                    + y * mp.log(t_new / (b1 + t_new))
                )
            )
        )
    return {
        "prior": [f(a), f(b)],
        "counts": counts,
        "exposures": None if all(t == 1 for t in exposures) else [f(t) for t in exposures],
        "posterior": [f(a1), f(b1)],
        "mean": f(a1 / b1),
        "variance": f(a1 / b1**2),
        "probs": [f(p) for p in PROBS],
        "quantiles": [f(gamma_quantile(a1, b1, p)) for p in PROBS],
        "cdf_at": [[f(x), f(gamma_cdf(a1, b1, x))] for x in (a1 / b1 / 2, a1 / b1, 2 * a1 / b1)],
        "log_marginal_likelihood": f(gp_log_ml(a, b, counts, exposures)),
        "predictive_exposure": f(t_new),
        "predictive_pmf": pmf,
        "predictive_mean": f(t_new * a1 / b1),
        "predictive_variance": f(t_new * a1 / b1 * (1 + t_new / b1)),
    }


def golden(fn, lo, hi):
    g = (mp.sqrt(5) - 1) / 2
    c, d = hi - g * (hi - lo), lo + g * (hi - lo)
    for _ in range(200):
        if fn(c) > fn(d):
            hi = d
        else:
            lo = c
        c, d = hi - g * (hi - lo), lo + g * (hi - lo)
    return (lo + hi) / 2


def argmax_grid(scores):
    best = 0
    for i in range(1, len(scores)):
        if scores[i] > scores[best] + mpf("1e-12") * max(1, abs(scores[best])):
            best = i
    return best


def eb_beta(a0, b0, k0, n0, k, n):
    def score(alpha):
        return bb_log_ml(a0 + alpha * k0, b0 + alpha * (n0 - k0), k, n)

    scores = [score(al) for al in GRID]
    best = argmax_grid(scores)
    return {
        "base": [f(a0), f(b0)],
        "source": [k0, n0],
        "target": [k, n],
        "grid": [f(al) for al in GRID],
        "scores": [f(s) for s in scores],
        "selected_index": best,
        "continuous_maximizer": f(golden(score, mpf(0), mpf(1))),
        "estimates": [f((a0 + al * k0 + k) / (a0 + b0 + al * n0 + n)) for al in GRID],
    }


def eb_gamma(a0, b0, s0, e0, counts):
    def score(alpha):
        return gp_log_ml(a0 + alpha * s0, b0 + alpha * e0, counts, [mpf(1)] * len(counts))

    scores = [score(al) for al in GRID]
    best = argmax_grid(scores)
    s, n = sum(counts), len(counts)
    return {
        "base": [f(a0), f(b0)],
        "source": [s0, f(e0)],
        "counts": counts,
        "grid": [f(al) for al in GRID],
        "scores": [f(x) for x in scores],
        "selected_index": best,
        "continuous_maximizer": f(golden(score, mpf(0), mpf(1))),
        "estimates": [f((a0 + al * s0 + s) / (b0 + al * e0 + n)) for al in GRID],
    }


def linear_case(known_sigma2):
    n, p = 14, 2
    x = [[mpf(1), mpf(i % 2)] for i in range(n)]
    truth = [mpf("0.5"), mpf("1.0")]
    y = [truth[0] + truth[1] * x[i][1] + mpf(rng.gauss(0, 1)) for i in range(n)]
    mb, vb = [mpf(0), mpf(0)], [mpf(100), mpf(100)]
    ms, vs = [mpf("0.45"), mpf("1.4")], [mpf("0.05"), mpf("0.04")]
    c = [mpf(0), mpf(1)]
    X = mp.matrix(x)
    Y = mp.matrix(y)
    a0, b0 = mpf("0.001"), mpf("0.001")
    scores, estimates, sds = [], [], []
    for alpha in GRID:
        prec = [1 / vb[i] + alpha * (1 / vs[i]) for i in range(p)]
        mean = [(mb[i] / vb[i] + alpha * ms[i] / vs[i]) / prec[i] for i in range(p)]
        V = mp.diag([1 / q for q in prec])
        m = mp.matrix(mean)
        base = mp.eye(n) + X * V * X.T
        resid = Y - X * m
        quad = (resid.T * mp.inverse(base) * resid)[0]
        logdet = mp.log(mp.det(base))
        if known_sigma2 is not None:
            s2 = mpf(known_sigma2)
            score = -n / mpf(2) * mp.log(2 * mp.pi * s2) - logdet / 2 - quad / (2 * s2)
        else:
            nu = 2 * a0
            sigma_scale = b0 / a0
            score = (
                mp.log(mp.gamma((nu + n) / 2))
                - mp.log(mp.gamma(nu / 2))
                - n / mpf(2) * mp.log(nu * mp.pi)
                - (n * mp.log(sigma_scale) + logdet) / 2
                - (nu + n) / 2 * mp.log(1 + quad / (sigma_scale * nu))
            )
        scores.append(score)
        P = mp.diag(prec)
        Ln = P + X.T * X
        mn = mp.inverse(Ln) * (P * m + X.T * Y)
        estimates.append(sum(c[i] * mn[i] for i in range(p)))
        cov = mp.inverse(Ln)
        v = (mp.matrix([c]) * cov * mp.matrix(c))[0]
        if known_sigma2 is not None:
            sds.append(mp.sqrt(mpf(known_sigma2) * v))
        else:
            res = Y - X * mn
            q = (res.T * res)[0] + ((mn - m).T * P * (mn - m))[0]
            an = a0 + n / mpf(2)
            bn = b0 + q / 2
            sds.append(mp.sqrt(bn / an * v))
    return {
        "x": [[f(v) for v in row] for row in x],
        "y": [f(v) for v in y],
        "baseline": {"mean": [f(v) for v in mb], "variance": [f(v) for v in vb]},
        "source": {"mean": [f(v) for v in ms], "variance": [f(v) for v in vs]},
        "functional": [f(v) for v in c],
        "known_residual_variance": None if known_sigma2 is None else float(known_sigma2),
        "grid": [f(al) for al in GRID],
        "scores": [f(s) for s in scores],
        "selected_index": argmax_grid(scores),
        "estimates": [f(e) for e in estimates],
        "scales": [f(s) for s in sds],
    }


doc = {
    "tolerance": {"quantile_rel": 1e-9, "log_ml_abs": 1e-10, "pmf_abs": 1e-12},
    "beta": [
        beta_case(mpf(2), mpf(3), 7, 20, 6),
        beta_case(mpf("0.5"), mpf("0.5"), 0, 10, 5),
        beta_case(mpf(1), mpf(1), 50, 50, 4),
        beta_case(mpf(30), mpf(70), 3, 12, 8),
        beta_case(mpf("0.7"), mpf("1.3"), 480, 1000, 10),
    ],
    "gamma": [
        gamma_case(mpf(2), mpf(1), [3, 0, 4, 2], [mpf(1)] * 4, mpf(2), 12),
        gamma_case(mpf("0.5"), mpf("0.01"), [120, 98, 105, 131, 90], [mpf(1)] * 5, mpf(1), 220),
        gamma_case(mpf(1), mpf(1), [0], [mpf(1)], mpf(1), 8),
        gamma_case(mpf(3), mpf("0.5"), [4, 9, 2], [mpf("2.5"), mpf("4.0"), mpf("1.2")], mpf("3.3"), 30),
    ],
    "empirical_bayes_beta": [
        eb_beta(mpf(1), mpf(1), 30, 100, 9, 30),
        eb_beta(mpf(1), mpf(1), 90, 100, 12, 60),
        eb_beta(mpf(1), mpf(1), 40, 100, 12, 40),
        eb_beta(mpf(1), mpf(1), 50, 100, 30, 50),
        eb_beta(mpf(1), mpf(1), 60, 100, 9, 20),
    ],
    "empirical_bayes_gamma": [
        eb_gamma(mpf(1), mpf(1), 80, mpf(40), [2, 1, 3, 2, 2, 1, 4, 2]),
        eb_gamma(mpf(1), mpf(1), 150, mpf(50), [2, 1, 3, 2, 2, 1, 4, 2]),
        eb_gamma(mpf(1), mpf(1), 400, mpf(40), [2, 1, 3, 2, 2, 1, 4, 2]),
    ],
    "linear": [linear_case(1.0), linear_case(None)],
}
print(json.dumps(doc, indent=1))
