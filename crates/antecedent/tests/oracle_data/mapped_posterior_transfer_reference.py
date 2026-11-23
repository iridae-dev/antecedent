# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Mapped posterior transfer: the target prior from a source's finite draws, in exact arithmetic.

Regenerate with:  uv run --script mapped_posterior_transfer_reference.py > mapped_posterior_transfer_reference.json

A source artifact holds N equally weighted joint draws of p coefficients and a constant residual
variance column sigma2.  The artifact's moments are those of the finite distribution of the draws
(denominator N): mean_i = sum_d x_di / N and cov_ij = sum_d (x_di - mean_i)(x_dj - mean_j) / N.
Under a complete bijection source name -> target name, the target coefficient k carries
  prior mean      mean_{s(k)},
  prior variance  max(cov_{s(k)s(k)}, 1e-12 sigma2) / sigma2           (V0 = Sigma / sigma2),
  correlation     rho_{kl} = cov_{s(k)s(l)} / sqrt(cov_{s(k)s(k)} cov_{s(l)s(l)}),
the correlation being carried (a dense V0) when it is positive definite and dropped to the diagonal
when it is not.  The same draws also give the sample covariance with denominator N - 1 used by
the posterior-artifact hydration.  The draws are binary64 numbers taken exactly; values are written
as the nearest binary64.  Repository-authored (MIT OR Apache-2.0); standard library only.
"""
import json
from fractions import Fraction as F
from math import isqrt

_state = 0xA4093822299F31D0


def u01():
    global _state
    _state = (_state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
    return (_state >> 11) / float(1 << 53)


def fsqrt(q, digits=60):
    scaled = q * 10 ** (2 * digits)
    return F(isqrt(scaled.numerator // scaled.denominator), 10**digits)


def moments(draws, ddof):
    n, p = len(draws), len(draws[0])
    mean = [sum(F(r[i]) for r in draws) / n for i in range(p)]
    cov = [[sum((F(r[i]) - mean[i]) * (F(r[j]) - mean[j]) for r in draws) / (n - ddof) for j in range(p)]
           for i in range(p)]
    return mean, cov


def case(name, n, offsets, scales, mix, sigma2, target_order):
    p = len(offsets)
    draws = []
    for _ in range(n):
        z = [u01() - 0.5 for _ in range(p)]
        # a fixed mixing matrix makes the coordinates correlated
        row = [offsets[i] + scales[i] * sum(mix[i][j] * z[j] for j in range(p)) for i in range(p)]
        draws.append(row + [sigma2])
    coef = [r[:p] for r in draws]
    mean, cov = moments(coef, 0)
    _, cov_unbiased = moments(coef, 1)
    s2 = F(sigma2)
    # target coefficient k reads source coefficient target_order[k]
    prior_mean = [float(mean[s]) for s in target_order]
    variance = [float(max(cov[s][s], F(1, 10**12) * s2) / s2) for s in target_order]
    corr = []
    for s in target_order:
        corr.append([float(cov[s][t] / fsqrt(cov[s][s] * cov[t][t])) if cov[s][s] > 0 and cov[t][t] > 0 else None
                     for t in target_order])
    return {
        "name": name, "p": p, "n": n, "sigma2": sigma2, "target_order": target_order,
        "draws": draws,
        "prior_mean": prior_mean, "prior_variance": variance, "correlation": corr,
        "sample_covariance": [[float(v) for v in row] for row in cov_unbiased],
    }


eye3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
mix3 = [[1.0, 0.0, 0.0], [0.8, 0.6, 0.0], [-0.3, 0.5, 0.9]]
cases = [
    case("cycle", 9, [0.2, -1.3, 4.0], [1.0, 2.0, 0.5], mix3, 2.5, [2, 0, 1]),
    case("offsets", 9, [1.0e6, -3.0e5, 0.0], [1.0e-3, 5.0e-4, 1.0], mix3, 0.04, [1, 2, 0]),
    case("near_collinear", 9, [0.0, 0.0, 1.0], [1.0, 1.0, 1.0],
         [[1.0, 0.0, 0.0], [1.0, 1.0e-4, 0.0], [0.0, 0.0, 1.0]], 1.0, [1, 0, 2]),
    case("identity_order", 6, [1.0, 2.0, 3.0], [1.0, 1.0, 1.0], mix3, 1.0, [0, 1, 2]),
    case("sigma2_large", 6, [0.0, 0.0, 0.0], [1.0e3, 1.0e3, 1.0e3], mix3, 1.0e4, [2, 1, 0]),
]
print(json.dumps({"cases": cases}, indent=1))
