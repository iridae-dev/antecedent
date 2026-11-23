# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0", "numpy>=1.26", "scipy>=1.14"]
# ///
"""Reference values for the hierarchical posterior of repeated categorical choices.

Regenerate with:  uv run --script hierarchical_choice_reference.py > hierarchical_choice_reference.json

Model: respondent i makes ordered choices with counts n_i (total m_i); theta_i ~ Dirichlet(kappa pi),
pi ~ Dirichlet(a), ln kappa ~ Normal(mu, sigma^2). Likelihood of respondent i:
  prod_k G(kappa pi_k + n_ik)/G(kappa pi_k) * G(kappa)/G(kappa + m_i)     (G = Gamma function).

Two independent methods, neither of them the product's panel scheme:

* two options: mpmath tanh-sinh double integration over (pi_1, ln kappa) with a cut at the tie
  pi_1 = 1/2, direct log-gamma, 15 digits. The interval is a root of the marginal distribution
  function.
* three options: (a) the smooth functionals (evidence, population means, heterogeneity) by the
  trapezoid rule in the coordinates u_k = ln(kappa pi_k), where the integrand
  lik * Dir(pi; a) * N(ln kappa) * prod_k pi_k is analytic and decays at both ends, so the rule
  converges geometrically; (b) the best-option probabilities and expected losses by scipy's
  adaptive vector cubature over the three regions {pi_k >= pi_j}, mapped to boxes.

Repository-authored (MIT OR Apache-2.0); mpmath, numpy and scipy are not product dependencies.
"""
import json
import math

import mpmath as mp
import numpy as np
from scipy.integrate import cubature
from scipy.optimize import brentq
from scipy.special import gammaln

mp.mp.dps = 18


# ---- model pieces (numpy) --------------------------------------------------------------------
def log_lik_np(alpha, kappa, hist):
    """alpha: (..., K); kappa: (...); hist: list of (counts, multiplicity)."""
    total = np.zeros(kappa.shape)
    for counts, mult in hist:
        m = sum(counts)
        term = gammaln(kappa) - gammaln(kappa + m)
        for k, n in enumerate(counts):
            term = term + gammaln(alpha[..., k] + n) - gammaln(alpha[..., k])
        total = total + mult * term
    return total


def log_prior_np(pi, v, a, mu, sigma):
    ln_norm = gammaln(sum(a)) - sum(gammaln(x) for x in a)
    lp = ln_norm + sum((a[k] - 1.0) * np.log(pi[..., k]) for k in range(len(a)))
    return lp - 0.5 * ((v - mu) / sigma) ** 2 - math.log(sigma) - 0.5 * math.log(2 * math.pi)


def group(counts):
    seen = {}
    for c in counts:
        seen[tuple(c)] = seen.get(tuple(c), 0) + 1
    return [(list(k), float(v)) for k, v in seen.items()]


# ---- K = 3: trapezoid in log-alpha ------------------------------------------------------------
def three_smooth(case):
    a, mu, sigma = case["a"], case["mu"], case["sigma"]
    hist = group(case["counts"])
    lo, hi, n = -7.0, 9.0, 120
    u = np.linspace(lo, hi, n)
    h = u[1] - u[0]
    # chunk over the first coordinate
    num = {"z": 0.0, "pi0": 0.0, "pi1": 0.0, "pi2": 0.0, "v": 0.0, "v2": 0.0, "rho": 0.0}
    shift = None
    chunks = []
    u1, u2, u3 = np.meshgrid(u, u, u, indexing="ij")
    alpha = np.stack([np.exp(u1), np.exp(u2), np.exp(u3)], axis=-1)
    kappa = alpha.sum(axis=-1)
    pi = alpha / kappa[..., None]
    v = np.log(kappa)
    logw = log_lik_np(alpha, kappa, hist) + log_prior_np(pi, v, a, mu, sigma) + np.log(pi).sum(axis=-1)
    shift = logw.max()
    w = np.exp(logw - shift) * h ** 3
    z = w.sum()
    out = {
        "log_evidence": math.log(z) + shift,
        "mean": [float((w * pi[..., k]).sum() / z) for k in range(3)],
        "log_kappa_mean": float((w * v).sum() / z),
        "log_kappa_sd": float(math.sqrt(max((w * v * v).sum() / z - ((w * v).sum() / z) ** 2, 0.0))),
        "correlation_mean": float((w / (kappa + 1.0)).sum() / z),
    }
    return out


# ---- K = 3: cubature of the best regions --------------------------------------------------------
def region_boxes(k):
    return [(1.0 / 3.0, 0.5, True), (0.5, 1.0, False)]


def three_regions(case):
    a, mu, sigma = case["a"], case["mu"], case["sigma"]
    hist = group(case["counts"])
    vlo, vhi = mu - 9 * sigma, mu + 9 * sigma
    best = [0.0] * 3
    weighted = [[0.0] * 3 for _ in range(3)]  # weighted[k][j] = integral of pi_j over best_k
    z = 0.0
    shift = None

    # reference constant from the smooth pass keeps exponents in range
    ref = case["_reference"]
    for k in range(3):
        others = [j for j in range(3) if j != k]
        for c_lo, c_hi, constrained in region_boxes(k):
            def f(x, k=k, others=others, constrained=constrained):
                c, t, v = x[:, 0], x[:, 1], x[:, 2]
                if constrained:
                    r_lo, r_hi = (1 - 2 * c) / (1 - c), c / (1 - c)
                else:
                    r_lo, r_hi = np.zeros_like(c), np.ones_like(c)
                r = r_lo + (r_hi - r_lo) * t
                pi = np.zeros((len(c), 3))
                pi[:, k] = c
                pi[:, others[0]] = (1 - c) * r
                pi[:, others[1]] = (1 - c) * (1 - r)
                jac = (1 - c) * (r_hi - r_lo)
                kappa = np.exp(v)
                alpha = kappa[:, None] * pi
                logt = log_lik_np(alpha, kappa, hist) + log_prior_np(pi, v, a, mu, sigma)
                dens = np.exp(logt - ref) * jac
                return np.column_stack([dens, dens * pi[:, 0], dens * pi[:, 1], dens * pi[:, 2]])
            res = cubature(f, [c_lo, 0.0, vlo], [c_hi, 1.0, vhi], rtol=1e-11, atol=0.0, rule="gk21", max_subdivisions=4000)
            val = res.estimate
            best[k] += val[0]
            for j in range(3):
                weighted[k][j] += val[1 + j]
            z += val[0]
    probs = [b / z for b in best]
    mean = [sum(weighted[k][j] for k in range(3)) / z for j in range(3)]
    emax = sum(weighted[k][k] for k in range(3)) / z
    return {"probability_best": probs, "expected_loss": [emax - m for m in mean], "mean_check": mean}


def three_interval(case, option, level):
    a, mu, sigma = case["a"], case["mu"], case["sigma"]
    hist = group(case["counts"])
    vlo, vhi = mu - 9 * sigma, mu + 9 * sigma
    others = [j for j in range(3) if j != option]
    ref = case["_reference"]

    def mass(x_hi, x_lo=0.0):
        def f(x):
            c, r, v = x[:, 0], x[:, 1], x[:, 2]
            pi = np.zeros((len(c), 3))
            pi[:, option] = c
            pi[:, others[0]] = (1 - c) * r
            pi[:, others[1]] = (1 - c) * (1 - r)
            kappa = np.exp(v)
            alpha = kappa[:, None] * pi
            logt = log_lik_np(alpha, kappa, hist) + log_prior_np(pi, v, a, mu, sigma)
            return (np.exp(logt - ref) * (1 - c))[:, None]
        return cubature(f, [x_lo, 0.0, vlo], [x_hi, 1.0, vhi], rtol=1e-11, atol=0.0, rule="gk21", max_subdivisions=4000).estimate[0]

    total = mass(1.0)
    tail = 0.5 * (1 - level)
    lo = brentq(lambda x: mass(x) / total - tail, 1e-9, 1 - 1e-9, xtol=1e-12)
    hi = brentq(lambda x: mass(x) / total - (1 - tail), 1e-9, 1 - 1e-9, xtol=1e-12)
    return [lo, hi]


# ---- K = 2: mpmath ---------------------------------------------------------------------------------
def two_options(case):
    a, mu, sigma = [mp.mpf(x) for x in case["a"]], mp.mpf(case["mu"]), mp.mpf(case["sigma"])
    hist = group(case["counts"])
    lg = mp.loggamma

    def logt(p, v):
        kappa = mp.e ** v
        pi = [p, 1 - p]
        ll = mp.mpf(0)
        for counts, mult in hist:
            m = sum(counts)
            term = lg(kappa) - lg(kappa + m)
            for k, n in enumerate(counts):
                term += lg(kappa * pi[k] + n) - lg(kappa * pi[k])
            ll += mult * term
        lp = lg(sum(a)) - lg(a[0]) - lg(a[1]) + (a[0] - 1) * mp.log(p) + (a[1] - 1) * mp.log(1 - p)
        return ll + lp - ((v - mu) / sigma) ** 2 / 2 - mp.log(sigma) - mp.log(2 * mp.pi) / 2

    vs = [mu + sigma * s for s in (-9, -4.5, -2, 0, 2, 4.5, 9)]
    # a reference level so that the exponentials stay in range
    ref = max(logt(mp.mpf(x), mu) for x in ("0.2", "0.4", "0.5", "0.6", "0.8"))

    def integral(g, p_lo, p_hi):
        return mp.quad(lambda p, v: g(p, v) * mp.e ** (logt(p, v) - ref), [p_lo, p_hi], vs)

    def i2(g, lo, hi):
        return mp.quad(lambda p: mp.quad(lambda v: g(p, v) * mp.e ** (logt(p, v) - ref), vs), [lo, hi])

    half = mp.mpf("0.5")
    one = lambda p, v: mp.mpf(1)
    z_hi = i2(one, half, 1)
    z_lo = i2(one, 0, half)
    z = z_hi + z_lo
    p_hi_w = i2(lambda p, v: p, half, 1)
    p_lo_w = i2(lambda p, v: 1 - p, 0, half)
    mean0 = (i2(lambda p, v: p, half, 1) + i2(lambda p, v: p, 0, half)) / z
    emax = (p_hi_w + p_lo_w) / z
    vm = (i2(lambda p, v: v, half, 1) + i2(lambda p, v: v, 0, half)) / z
    v2 = (i2(lambda p, v: v * v, half, 1) + i2(lambda p, v: v * v, 0, half)) / z
    rho = (i2(lambda p, v: 1 / (mp.e ** v + 1), half, 1) + i2(lambda p, v: 1 / (mp.e ** v + 1), 0, half)) / z

    def cdf(x):
        return i2(one, 0, x) / z

    level = mp.mpf("0.9")
    tail = (1 - level) / 2
    lo = mp.findroot(lambda x: cdf(x) - tail, mp.mpf("0.4"), tol=1e-14, solver="anderson") if False else None
    # bisection on the marginal distribution function (monotone, so robust)
    def bisect(target):
        a_, b_ = mp.mpf("1e-6"), 1 - mp.mpf("1e-6")
        for _ in range(48):
            mid = (a_ + b_) / 2
            if cdf(mid) < target:
                a_ = mid
            else:
                b_ = mid
        return (a_ + b_) / 2

    interval = [bisect(tail), bisect(1 - tail)]
    return {
        "log_evidence": float(mp.log(z) + ref),
        "mean": [float(mean0), float(1 - mean0)],
        "probability_best": [float(z_hi / z), float(z_lo / z)],
        "expected_loss": [float(emax - mean0), float(emax - (1 - mean0))],
        "log_kappa_mean": float(vm),
        "log_kappa_sd": float(mp.sqrt(v2 - vm * vm)),
        "correlation_mean": float(rho),
        "interval_option0_90": [float(x) for x in interval],
    }


CASES = [
    {"name": "two_options_moderate", "k": 2, "a": [1, 1], "mu": math.log(5), "sigma": 1.0,
     "counts": [[3, 1], [1, 3], [2, 2], [4, 0], [0, 4], [3, 1], [2, 2]]},
    {"name": "two_options_leaning_informative_prior", "k": 2, "a": [2, 3], "mu": math.log(10), "sigma": 0.7,
     "counts": [[5, 1], [4, 2], [6, 0], [3, 3], [5, 1], [4, 2]]},
    {"name": "three_options_symmetric", "k": 3, "a": [1, 1, 1], "mu": math.log(5), "sigma": 1.0,
     "counts": [[2, 1, 0], [0, 2, 1], [1, 0, 2], [1, 1, 1], [2, 1, 0], [0, 2, 1], [1, 0, 2]]},
    {"name": "three_options_leading", "k": 3, "a": [1, 1, 1], "mu": math.log(8), "sigma": 0.8,
     "counts": [[3, 1, 0], [2, 1, 1], [3, 0, 1], [4, 0, 0], [1, 2, 1], [2, 2, 0]]},
]


def run(case):
    if case["k"] == 2:
        out = two_options(case)
    else:
        out = three_smooth(case)
        case["_reference"] = out["log_evidence"]
        regions = three_regions(case)
        out["probability_best"] = regions["probability_best"]
        out["expected_loss"] = regions["expected_loss"]
        out["region_mean_check"] = regions["mean_check"]
        out["interval_option0_90"] = three_interval(case, 0, 0.9)
    return {
        "name": case["name"], "k": case["k"], "a": case["a"], "mu": case["mu"], "sigma": case["sigma"],
        "counts": case["counts"], **out,
    }


print(json.dumps({"cases": [run(c) for c in CASES]}, indent=1))
