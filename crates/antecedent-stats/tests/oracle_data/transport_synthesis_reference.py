# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath"]
# ///
"""Transport-aware synthesis: exact GLS pooling, hierarchical borrowing and the transported prior.

Regenerate with:  uv run --script transport_synthesis_reference.py > transport_synthesis_reference.json

Pooling: each summary carries an estimate, a sampling variance and either no transport or a
transport (key, variance tau^2).  The covariance is the sampling covariance (diagonal unless one is
supplied) plus tau^2 added to the variance of every transported summary and to the covariance of
every pair of summaries that share a transport key.  The pooled value is the generalized least
squares estimate (1' S^-1 y)/(1' S^-1 1) with variance 1/(1' S^-1 1) and heterogeneity
(y - mu 1)' S^-1 (y - mu 1), all solved exactly in Fractions by Gaussian elimination on dyadic
inputs; the "without transport" variance repeats the solve with the transport terms removed.

Borrowing: summaries y_i with effective variance v_i + tau_t^2 are exchangeable draws
theta_i ~ N(mu, t^2), y_i | theta_i ~ N(theta_i, e_i), with a flat or normal prior on mu and a
discrete prior on t.  The marginal likelihood of y given t is evaluated from its definition (the
Gaussian density of y with covariance diag(e + t^2) plus the prior mean variance times the all-ones
matrix; for the flat prior the improper density up to a constant that does not depend on t),
by explicit determinant and inverse in 50-digit mpmath; the conditional posterior of mu and of each
theta_i are the standard Gaussian conditionals, and the reported posterior is the mixture over t.

Prior update: a transported claim (value m, sampling variance s, transport variance tau^2) is the
prior of a native normal likelihood made of studies with variances v_j: precision-weighted mean,
exact in Fractions.  Repository-authored (MIT OR Apache-2.0).
"""
import json
from fractions import Fraction as F

import mpmath as mp

mp.mp.dps = 50


def solve(a, b):
    n = len(a)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        p = next(r for r in range(c, n) if m[r][c] != 0)
        m[c], m[p] = m[p], m[c]
        for r in range(n):
            if r != c and m[r][c] != 0:
                f = m[r][c] / m[c][c]
                m[r] = [x - f * y for x, y in zip(m[r], m[c])]
    return [m[i][n] / m[i][i] for i in range(n)]


def gls(y, cov):
    k = len(y)
    ones = [F(1)] * k
    x = solve(cov, ones)
    den = sum(x)
    mu = sum(xi * yi for xi, yi in zip(x, y)) / den
    r = [yi - mu for yi in y]
    z = solve(cov, r)
    q = sum(ri * zi for ri, zi in zip(r, z))
    return mu, 1 / den, [xi / den for xi in x], q


def out(x):
    return {"value": float(x), "exact": f"{x.numerator}/{x.denominator}"}


def pooling(name, summaries, base=None, refused=False):
    k = len(summaries)
    cov = [[F(0)] * k for _ in range(k)]
    for i, s in enumerate(summaries):
        cov[i][i] = F(s["var"])
    if base is not None:
        cov = [[F(v) for v in row] for row in base]
    plain = [row[:] for row in cov]
    for i, a in enumerate(summaries):
        if a.get("key") is None:
            continue
        cov[i][i] += F(a["tau2"])
        for j in range(i + 1, k):
            if summaries[j].get("key") == a["key"]:
                cov[i][j] += F(a["tau2"])
                cov[j][i] += F(a["tau2"])
    y = [F(s["y"]) for s in summaries]
    mu, var, w, q = gls(y, cov)
    mu0, var0, _, _ = gls(y, plain)
    return {
        "name": name, "summaries": summaries, "base": base, "refused_as_singular": refused,
        "estimate": out(mu), "variance": out(var), "weights": [out(x) for x in w],
        "heterogeneity": out(q), "variance_without_transport": out(var0),
        "transport_variance_added": out(max(var - var0, F(0))),
        "covariance": [[out(v) for v in row] for row in cov],
    }


def borrowing(name, y, eff, taus, probs, prior, tau2s):
    k = len(y)
    ym = mp.matrix([mp.mpf(F(v).numerator) / F(v).denominator for v in y])
    logs, conds = [], []
    ones = mp.matrix([1] * k)
    for t, p in zip(taus, probs):
        t2 = mp.mpf(F(t).numerator) / F(t).denominator
        t2 = t2 * t2
        V = mp.diag([mp.mpf(F(e).numerator) / F(e).denominator + t2 for e in eff])
        Vi = V ** -1
        prec_data = (ones.T * Vi * ones)[0]
        if prior is None:
            muhat = (ones.T * Vi * ym)[0] / prec_data
            r = ym - muhat * ones
            quad = (r.T * Vi * r)[0]
            logm = -mp.log(mp.det(V)) / 2 - mp.log(prec_data) / 2 - quad / 2
            mean, var = muhat, 1 / prec_data
        else:
            m0 = mp.mpf(F(prior[0]).numerator) / F(prior[0]).denominator
            v0 = mp.mpf(F(prior[1]).numerator) / F(prior[1]).denominator
            S = V + v0 * (ones * ones.T)
            r = ym - m0 * ones
            logm = -mp.log(mp.det(S)) / 2 - (r.T * (S ** -1) * r)[0] / 2
            precision = 1 / v0 + prec_data
            mean = (m0 / v0 + (ones.T * Vi * ym)[0]) / precision
            var = 1 / precision
        pw = mp.mpf(F(p).numerator) / F(p).denominator
        logs.append(mp.log(pw) + logm)
        conds.append((t2, mean, var))
    peak = max(logs)
    un = [mp.e ** (l - peak) for l in logs]
    z = sum(un)
    post = [u / z for u in un]
    m1 = sum(pi * c[1] for pi, c in zip(post, conds))
    m2 = sum(pi * (c[2] + c[1] ** 2) for pi, c in zip(post, conds))
    sm, sv = [], []
    for i in range(k):
        mi = mp.mpf(0)
        si = mp.mpf(0)
        for pi, (t2, mean, var) in zip(post, conds):
            e = mp.mpf(F(eff[i]).numerator) / F(eff[i]).denominator
            b = t2 / (t2 + e)
            yi = ym[i]
            m = b * yi + (1 - b) * mean
            v = b * e + (1 - b) ** 2 * var
            mi += pi * m
            si += pi * (v + m * m)
        sm.append(mi)
        sv.append(si - mi * mi)
    f = lambda x: float(x)
    return {
        "name": name, "transport_variances": tau2s, "estimates": [float(v) for v in y], "effective_variances": [float(v) for v in eff],
        "tau_support": [float(t) for t in taus], "tau_probabilities": [float(p) for p in probs],
        "prior": None if prior is None else [float(prior[0]), float(prior[1])],
        "mean": f(m1), "variance": f(m2 - m1 * m1),
        "study_means": [f(v) for v in sm], "study_variances": [f(v) for v in sv],
        "tau_posterior": [f(v) for v in post],
    }


def prior_update(name, prior, likelihood):
    pv = F(prior["var"]) + F(prior["tau2"])
    pp = 1 / pv
    prec = sum(1 / F(s["var"]) for s in likelihood)
    wsum = sum(F(s["y"]) / F(s["var"]) for s in likelihood)
    tot = pp + prec
    return {
        "name": name, "prior": prior, "likelihood": likelihood,
        "mean": out((pp * F(prior["y"]) + wsum) / tot), "variance": out(1 / tot),
        "prior_variance": out(pv), "prior_weight": out(pp / tot), "likelihood_precision": out(prec),
    }


T = lambda y, v, key=None, tau2=None: {"y": y, "var": v, "key": key, "tau2": tau2}
pools = [
    pooling("shared_and_distinct_transports_with_cross_covariance",
            [T(1.0, 1.0), T(2.0, 0.5, "a", 0.25), T(3.5, 2.0, "a", 0.25), T(-1.0, 0.75, "b", 4.0)],
            base=[[1.0, 0.25, 0.0, 0.0], [0.25, 0.5, 0.125, 0.0], [0.0, 0.125, 2.0, 0.0], [0.0, 0.0, 0.0, 0.75]]),
    pooling("extreme_variance_ratio",
            [T(0.5, 2.0 ** -20), T(0.75, 2.0 ** 20, "t", 2.0 ** 10), T(-3.0, 1.0, "t", 2.0 ** 10)]),
    pooling("near_collinear_shared_transport",
            [T(1.0, 2.0 ** -12, "t", 2.0 ** 4), T(1.125, 2.0 ** -12, "t", 2.0 ** 4), T(2.0, 2.0 ** -12, "t", 2.0 ** 4)]),
    # correlation 1 - 4e-12: the exact GLS answer exists, but the routine's documented
    # correlation-scale singularity test (smallest eigenvalue <= 1e-10 of the largest) refuses it
    pooling("collinear_beyond_the_singularity_threshold",
            [T(1.0, 2.0 ** -30, "t", 2.0 ** 8), T(1.125, 2.0 ** -30, "t", 2.0 ** 8), T(2.0, 2.0 ** -30, "t", 2.0 ** 8)],
            refused=True),
    pooling("zero_transport_variance_is_native", [T(1.0, 1.0), T(2.0, 1.0, "t", 0.0)]),
    pooling("single_summary", [T(4.0, 0.5, "t", 2.0)]),
]
taus_a = [F(0), F(1, 4), F(1, 2), F(1), F(2)]
probs_a = [F(1, 8), F(1, 4), F(1, 4), F(1, 4), F(1, 8)]
borrows = [
    borrowing("flat_mean_prior_with_a_transported_study", [F(1), F(5, 2), F(-1, 2)],
              [F(1, 2), F(3, 2) + F(1, 4), F(1, 4)], taus_a, probs_a, None, [None, 0.25, None]),
    borrowing("normal_mean_prior_wide_spread", [F(10), F(-8), F(0), F(1, 2)],
              [F(1, 4), F(1, 2), F(2), F(1) + F(1, 8)], taus_a, probs_a, (F(0), F(25)), [None, None, None, 0.125]),
    borrowing("tiny_effective_variances_tau_zero_only", [F(1), F(1, 2)],
              [F(1, 1024), F(1, 1024)], [F(0)], [F(1)], None, [None, None]),
    borrowing("two_studies_single_large_tau", [F(3), F(-3)], [F(1), F(1)],
              [F(8)], [F(1)], (F(1), F(1, 4)), [0.5, None]),
]
updates = [
    prior_update("transported_claim_against_two_native_studies",
                 T(2.0, 0.5, "t", 0.5), [T(1.0, 0.25), T(1.5, 1.0)]),
    prior_update("very_uncertain_transport_yields_to_the_data",
                 T(10.0, 1.0, "t", 4096.0), [T(0.0, 0.0625)]),
]
print(json.dumps({"pools": pools, "borrows": borrows, "updates": updates}, separators=(",", ":")))
