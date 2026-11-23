# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""High-precision reference values for the conjugate normal expected value of sample information.

Regenerate with:  uv run --script evsi_conjugate_reference.py > evsi_conjugate_reference.json

For a prior theta ~ N(mu0, tau2), n observations of variance sigma2 and terminal actions with
utility alpha_a + beta_a * theta, the posterior mean after the sample mean ybar is
  m(ybar) = (mu0/tau2 + n ybar/sigma2) / (1/tau2 + n/sigma2)
and ybar ~ N(mu0, tau2 + sigma2/n) marginally.  This script does not use the preposterior
variance shortcut of the product code: it integrates the maximum over actions of the posterior
expected utility against the marginal density of ybar by adaptive Gauss-Legendre quadrature
split at every pairwise break-even point (mpmath.quad at 60 digits), and subtracts the best
prior expected utility.  Every number is evaluated from the exact binary64 inputs, so the
JSON stores inputs as hex-exact decimals via repr of the Python float.

Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import json
from fractions import Fraction as F

import mpmath as mp

mp.mp.dps = 60

# (name, mu0, tau2, n, sigma2, intercepts, slopes)
CASES = [
    ("two_actions_unit", 0.5, 1.0, 4, 12.0, [0.0, 0.0], [0.0, 1.0]),
    ("three_actions_interior_middle", -0.3, 2.5, 7, 3.0, [0.0, 0.4, 0.1], [-1.0, 0.0, 1.5]),
    ("five_actions_one_dominated", 0.2, 1.3, 3, 2.0, [0.0, 0.5, 0.2, -3.0, 0.3], [1.0, 0.0, -0.5, 0.1, 0.7]),
    ("equal_slopes_best_intercept_wins", 0.0, 1.0, 5, 4.0, [0.1, 0.3, 0.0], [1.0, 1.0, -1.0]),
    ("tiny_information_ratio", 0.0, 1.0, 1, 1.0e12, [0.0, 0.0], [0.0, 1.0]),
    ("huge_information_ratio", 0.1, 1.0, 1000000, 1.0e-6, [0.0, 0.0], [0.0, 1.0]),
    ("large_utility_offsets", 2.0, 0.5, 10, 5.0, [1.0e9, 1.0e9 + 1.0], [0.0, 3.0]),
    ("prior_far_from_break_even", 8.0, 1.0, 4, 12.0, [0.0, 0.0], [0.0, 1.0]),
    ("prior_at_break_even", 0.0, 4.0, 2, 3.0, [0.0, 0.0], [-2.0, 2.0]),
    ("tiny_prior_variance", 0.0, 1.0e-8, 3, 1.0, [0.0, 0.0], [0.0, 1.0]),
    ("many_actions_fan", 0.0, 1.0, 6, 2.0, [0.0, 0.15, 0.3, 0.45, 0.6, 0.75], [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0]),
]


def evsi(mu0, tau2, n, sigma2, alpha, beta):
    mu0, tau2, sigma2 = mp.mpf(mu0), mp.mpf(tau2), mp.mpf(sigma2)
    alpha = [mp.mpf(a) for a in alpha]
    beta = [mp.mpf(b) for b in beta]
    prec = 1 / tau2 + n / sigma2
    marg_var = tau2 + sigma2 / n
    marg_sd = mp.sqrt(marg_var)

    def post_mean(y):
        return (mu0 / tau2 + n * y / sigma2) / prec

    def best(m):
        return max(a + b * m for a, b in zip(alpha, beta))

    def integrand(y):
        z = (y - mu0) / marg_sd
        return best(post_mean(y)) * mp.exp(-z * z / 2) / (marg_sd * mp.sqrt(2 * mp.pi))

    # break-even posterior means -> sample means
    cuts = set()
    for i in range(len(alpha)):
        for j in range(i + 1, len(alpha)):
            if beta[i] != beta[j]:
                m = (alpha[j] - alpha[i]) / (beta[i] - beta[j])
                cuts.add((m * prec - mu0 / tau2) * sigma2 / n)
    lo, hi = mu0 - 40 * marg_sd, mu0 + 40 * marg_sd
    pts = sorted([lo, hi] + [c for c in cuts if lo < c < hi] + [mu0])
    total = mp.mpf(0)
    for a, b in zip(pts, pts[1:]):
        total += mp.quad(integrand, mp.linspace(a, b, 9))
    prior = best(mu0)
    # perfect information: E[max_a (alpha + beta theta)] over theta ~ N(mu0, tau2)
    def integrand_pi(t):
        z = (t - mu0) / mp.sqrt(tau2)
        return best(t) * mp.exp(-z * z / 2) / (mp.sqrt(tau2) * mp.sqrt(2 * mp.pi))
    cuts_t = set()
    for i in range(len(alpha)):
        for j in range(i + 1, len(alpha)):
            if beta[i] != beta[j]:
                cuts_t.add((alpha[j] - alpha[i]) / (beta[i] - beta[j]))
    sd = mp.sqrt(tau2)
    lo, hi = mu0 - 40 * sd, mu0 + 40 * sd
    pts = sorted([lo, hi] + [c for c in cuts_t if lo < c < hi] + [mu0])
    pi = mp.mpf(0)
    for a, b in zip(pts, pts[1:]):
        pi += mp.quad(integrand_pi, mp.linspace(a, b, 9))
    return total - prior, pi - prior, total, prior, prec


out = []
for name, mu0, tau2, n, sigma2, alpha, beta in CASES:
    g, evpi, pre, prior, prec = evsi(mu0, tau2, n, sigma2, alpha, beta)
    out.append(
        {
            "name": name,
            "prior_mean": mu0,
            "prior_variance": tau2,
            "sample_size": n,
            "noise_variance": sigma2,
            "intercept": alpha,
            "slope": beta,
            "gross_evsi": float(g),
            "evpi": float(evpi),
            "prior_expected_utility": float(prior),
            "preposterior_expected_utility": float(pre),
            "posterior_variance": float(1 / prec),
        }
    )
print(json.dumps(out, indent=1))
