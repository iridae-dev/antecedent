"""Scalar linear-Gaussian state space against the joint Gaussian law, in 80-digit arithmetic.

Model: x_0 ~ N(m0, q^2), x_t = a x_{t-1} + q eps_t, y_t = x_t + r eta_t.  The reference never
runs a Kalman recursion: it writes down the covariance of (x_0..x_{n-1}) as
q^2 sum_{k<=min(s,t)} a^{s+t-2k}, adds r^2 on the diagonal for y, and obtains every filtered,
predicted and smoothed moment, the lag-one smoothed covariance and the per-row predictive
log density by Gaussian conditioning on the leading block of y.

    uv run --with mpmath python lgssm_joint_gaussian_reference.py > lgssm_joint_gaussian_reference.json
"""

import json
import math

from mpmath import matrix, mp, mpf, log, pi, lu_solve, inverse, det

mp.dps = 80

CASES = [
    ("moderate", 0.9, 0.5, 0.3, 0.2, 8, 1.0),
    ("random_walk_noisy_observation", 1.0, 1e-3, 10.0, 0.0, 8, 10.0),
    ("explosive", 1.5, 0.2, 0.4, 0.1, 8, 1.0),
    ("almost_noiseless_observation", -0.8, 3.0, 1e-6, -1.0, 8, 3.0),
    ("tiny_scale_near_unit_root", 0.999, 1e-8, 1e-8, 2e-8, 12, 1e-8),
    ("huge_scale", 0.7, 1e8, 1e7, 1e8, 10, 1e8),
]


def observations(n, scale, a):
    # deterministic, irregular, exactly representable inputs: the same binary64 values are
    # read by the product from the JSON this script prints
    return [float(scale * (math.sin(1.7 * t + 0.3) + 0.4 * math.cos(0.9 * t * t))) for t in range(n)]


def cond(sx, mu, y, m, r2):
    """Posterior of the n states given the first m observations (m may be 0)."""
    n = sx.rows
    if m == 0:
        return mu.copy(), sx.copy()
    cross = sx[0:n, 0:m]
    syy = sx[0:m, 0:m] + r2 * matrix([[1 if i == j else 0 for j in range(m)] for i in range(m)])
    resid = matrix(y[:m]) - mu[0:m]
    gain = cross * inverse(syy)
    mean = mu + gain * resid
    cov = sx - gain * cross.T
    return mean, cov


def main():
    out = {"cases": []}
    for name, a, q, r, m0, n, scale in CASES:
        a_, q_, r_, m0_ = mpf(a), mpf(q), mpf(r), mpf(m0)
        y = observations(n, scale, a)
        ym = [mpf(v) for v in y]
        mu = matrix([a_**t * m0_ for t in range(n)])
        sx = matrix(n, n)
        for s in range(n):
            for t in range(n):
                sx[s, t] = q_**2 * sum(a_ ** (s + t - 2 * k) for k in range(min(s, t) + 1))
        r2 = r_**2
        fm, fv, pm, pv, ll = [], [], [], [], []
        previous = mpf(0)
        for t in range(n):
            mean_p, cov_p = cond(sx, mu, ym, t, r2)
            pm.append(mean_p[t])
            pv.append(cov_p[t, t])
            mean_f, cov_f = cond(sx, mu, ym, t + 1, r2)
            fm.append(mean_f[t])
            fv.append(cov_f[t, t])
            m = t + 1
            syy = sx[0:m, 0:m] + r2 * matrix([[1 if i == j else 0 for j in range(m)] for i in range(m)])
            resid = matrix(ym[:m]) - mu[0:m]
            logp = -(m * log(2 * pi) + log(det(syy)) + (resid.T * lu_solve(syy, resid))[0]) / 2
            ll.append(logp - previous)
            previous = logp
        mean_s, cov_s = cond(sx, mu, ym, n, r2)
        out["cases"].append(
            {
                "name": name,
                "a": a, "process_std": q, "obs_std": r, "initial_mean": m0,
                "observations": y,
                "filtered_mean": [float(v) for v in fm],
                "filtered_variance": [float(v) for v in fv],
                "predicted_mean": [float(v) for v in pm],
                "predicted_variance": [float(v) for v in pv],
                "smoothed_mean": [float(mean_s[t]) for t in range(n)],
                "smoothed_variance": [float(cov_s[t, t]) for t in range(n)],
                "lag_one_covariance": [0.0] + [float(cov_s[t, t - 1]) for t in range(1, n)],
                "row_log_density": [float(v) for v in ll],
            }
        )
    print(json.dumps(out, indent=1))


main()
