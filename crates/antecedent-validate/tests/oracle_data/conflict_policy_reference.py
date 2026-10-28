# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Reference values for the prior-conflict alpha policy and the composition it feeds.

Regenerate with:  uv run --script conflict_policy_reference.py > conflict_policy_reference.json

Written from the policy's stated definition, at 60 digits:
  alpha' = alpha * 1{p > p_min} * exp(-kl_scale * kl)         (absent signals contribute 1)
  composed precision  = 1/v0 + sum_i alpha'_i / v_i
  composed mean       = (m0/v0 + sum_i alpha'_i m_i / v_i) / precision
and the Gaussian divergence KL(N(m0, v0) || N(m1, v1)) = 1/2 (v0/v1 - 1 - ln(v0/v1) + (m0-m1)^2/v1)
evaluated with mp.log1p-free high-precision arithmetic so that near-equal variances are exact.

Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import json

import mpmath as mp

mp.mp.dps = 80
mpf = mp.mpf


def shrink(alpha, p, kl, p_min, kl_scale):
    f = mpf(1)
    if p is not None and mpf(p) <= mpf(p_min):
        f = mpf(0)
    if kl is not None and mpf(kl) > 0 and mpf(kl_scale) > 0:
        f *= mp.exp(-mpf(kl_scale) * mpf(kl))
    return mpf(alpha) * f


SHRINK = []
for alpha in (0.0, 0.3, 1.0):
    for p in (None, 0.0, 0.05, 0.0500000000000001, 0.5, 1.0):
        for kl in (None, 0.0, 1e-12, 0.5, 2.0, 50.0, 800.0):
            for p_min, kl_scale in ((0.05, 1.0), (0.0, 0.0), (0.2, 3.5), (1.0, 1.0)):
                SHRINK.append(
                    {"alpha": alpha, "p": p, "kl": kl, "p_min": p_min, "kl_scale": kl_scale,
                     "applied": float(shrink(alpha, p, kl, p_min, kl_scale))}
                )

COMPOSE = []
for name, base, srcs in [
    ("two_sources_one_conflicting", (0.0, 4.0), [(1.0, 0.5, 0.8, 0.5, 0.1), (50.0, 0.25, 1.0, 0.001, 2.0)]),
    ("three_sources_graded", (-1.0, 2.0), [(0.0, 1.0, 1.0, 0.3, 0.7), (2.0, 0.1, 0.5, 0.06, 3.0), (-1.5, 3.0, 0.25, None, None)]),
    ("all_dropped_returns_baseline", (3.0, 0.7), [(9.0, 0.2, 1.0, 0.0, None), (-9.0, 0.2, 0.6, 0.04, 0.0)]),
    ("tight_source_heavily_discounted", (0.0, 100.0), [(5.0, 1e-4, 1.0, 0.9, 12.0)]),
]:
    p_min, kl_scale = 0.05, 1.0
    applied = [shrink(a, p, kl, p_min, kl_scale) for (_, _, a, p, kl) in srcs]
    prec = 1 / mpf(base[1]) + sum(a / mpf(v) for a, (_, v, _, _, _) in zip(applied, srcs))
    num = mpf(base[0]) / mpf(base[1]) + sum(a * mpf(m) / mpf(v) for a, (m, v, _, _, _) in zip(applied, srcs))
    COMPOSE.append(
        {"name": name, "baseline": list(base),
         "sources": [{"mean": m, "variance": v, "alpha": a, "p": p, "kl": kl} for (m, v, a, p, kl) in srcs],
         "applied": [float(a) for a in applied],
         "mean": float(num / prec), "variance": float(1 / prec)}
    )


def gkl(m0, v0, m1, v1):
    m0, v0, m1, v1 = map(mpf, (m0, v0, m1, v1))
    r = v0 / v1
    return (r - 1 - mp.log(r) + (m0 - m1) ** 2 / v1) / 2


KL = []
for m0, v0, m1, v1 in [
    (0.0, 1.0, 0.0, 1.0), (1.0, 1.0, 0.0, 1.0), (0.3, 2.0, -0.4, 0.5), (0.0, 1.0 + 1e-9, 0.0, 1.0),
    (0.0, 1.0 - 1e-12, 0.0, 1.0), (0.0, 1e-9, 0.0, 1.0), (0.0, 1e9, 0.0, 1.0), (0.0, 1.0, 0.0, 1e9),
    (0.0, 1.0, 0.0, 1e-9), (1e6, 1e-12, 1e6 + 3e-6, 1e-12), (2.0, 0.25, 2.0, 0.25 * (1 + 4e-16)),
    (0.0, 3.0, 0.0, 1.9999), (0.0, 3.0, 0.0, 2.0),
]:
    KL.append({"m0": m0, "v0": v0, "m1": m1, "v1": v1, "kl": float(gkl(m0, v0, m1, v1))})

print(json.dumps({"shrink": SHRINK, "compose": COMPOSE, "kl": KL}, indent=1))
