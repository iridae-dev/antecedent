# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Dense-covariance reference for the joint-panel AR(1) plus random-intercept marginal law.

Regenerate with:  uv run --offline --script joint_panel_ar1_reference.py > joint_panel_ar1_reference.json

Written from the definition of the model, not from the product's precision-matrix algebra.  A
group's residual vector r (length n) is  r = u 1 + e  with  u ~ N(0, tau^2)  and e a stationary
Gaussian AR(1) process  e_t = rho e_{t-1} + sigma z_t,  so  Cov(e_i, e_j) = sigma^2 rho^|i-j| /
(1 - rho^2).  The covariance of r is  S = Cov(e) + tau^2 1 1'.  This script builds S as an exact
rational matrix from the double inputs, solves  S x = r  and  S y = 1  by exact Gauss-Jordan
elimination, takes the exact determinant, and returns

  log-likelihood      -1/2 (n ln(2 pi) + ln|S| + r' S^-1 r)
  intercept mean      E[u | r]   = tau^2 1' S^-1 r
  intercept variance  Var[u | r] = tau^2 - tau^4 1' S^-1 1

with every logarithm taken in the same 120-digit arithmetic.  Cases include a single observation, strong
positive and negative serial correlation (|rho| = 0.999), no correlation, a tiny and a huge
intercept scale, a tiny innovation scale, residuals offset by 1e6 (an intercept explains the
group mean), and one long group (n = 150).

Repository-authored (MIT OR Apache-2.0).
"""
import json
import random
import mpmath as mp

mp.mp.dps = 120


def case(name, r, rho, sigma, tau):
    n = len(r)
    rho_, sig_, tau_ = mp.mpf(rho), mp.mpf(sigma), mp.mpf(tau)
    s2, t2 = sig_ * sig_, tau_ * tau_
    cov = mp.matrix(n, n)
    for i in range(n):
        for j in range(n):
            cov[i, j] = s2 * rho_ ** abs(i - j) / (1 - rho_ * rho_) + t2
    rv = mp.matrix([mp.mpf(x) for x in r])
    ones = mp.matrix([1] * n)
    x = mp.lu_solve(cov, rv)
    y = mp.lu_solve(cov, ones)
    quad = sum(rv[i] * x[i] for i in range(n))
    logdet = mp.log(mp.det(cov))
    ll = -(n * mp.log(2 * mp.pi) + logdet + quad) / 2
    mean = t2 * sum(x[i] for i in range(n))
    var = t2 - t2 * t2 * sum(y[i] for i in range(n))
    return {"name": name, "r": r, "rho": rho, "sigma": sigma, "tau": tau, "n": n,
            "log_likelihood": float(ll), "intercept_mean": float(mean),
            "intercept_variance": float(var)}


def main():
    rnd = random.Random(20261028)
    nrm = lambda s, n: [rnd.gauss(0, s) for _ in range(n)]
    out = []
    out.append(case("single_observation", [0.3], 0.5, 0.2, 0.35))
    out.append(case("two_observations", [0.3, -0.1], 0.45, 0.2, 0.35))
    out.append(case("seven_typical", nrm(0.3, 7), 0.45, 0.2, 0.35))
    out.append(case("rho_zero", nrm(0.5, 9), 0.0, 0.4, 0.6))
    out.append(case("rho_near_one", nrm(0.5, 9), 0.999, 0.05, 0.5))
    out.append(case("rho_near_minus_one", nrm(0.5, 9), -0.999, 0.05, 0.5))
    out.append(case("tau_tiny", nrm(0.3, 8), 0.6, 0.2, 1e-3))
    out.append(case("tau_huge", nrm(0.3, 8), 0.6, 0.2, 1e3))
    out.append(case("sigma_tiny", nrm(0.3, 8), 0.3, 1e-3, 0.5))
    out.append(case("sigma_huge", nrm(0.3, 8), 0.3, 1e3, 0.5))
    out.append(case("offset_1e6_with_wide_intercept", [1e6 + x for x in nrm(0.2, 6)], 0.4, 0.2, 1e3))
    out.append(case("offset_1e6_with_narrow_intercept", [1e6 + x for x in nrm(0.2, 6)], 0.4, 0.2, 0.5))
    out.append(case("long_group_150", nrm(0.4, 150), 0.9, 0.2, 0.7))
    out.append(case("unbalanced_three_observations", [2.0, -2.0, 0.5], -0.7, 0.9, 0.2))
    json.dump(out, open("/dev/stdout", "w"), separators=(",", ":"))


main()
