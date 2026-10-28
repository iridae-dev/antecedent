# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy==2.4.6", "scipy==1.17.0"]
# ///
"""Exact posterior of the observed-data Gaussian temporal mechanism on tiny tables.

The outcome mechanism is y_t = b0 + b1 x_{t-1} + e_t, e_t ~ N(0, s2), rows t = 1..8 (row 0 is
an exact outcome used only as history).  Prior, the product's documented convention:
(b0, b1) | s2 ~ N(0, s2 * scale^2 I) and s2 ~ InverseGamma(1e-3, 1e-3).  The latent outcomes
are integrated out exactly: a missing (unselected) row contributes 1 (the selection indicator
is independent of the outcome given the conditioning set), a right-censored row contributes
the survival function P(y > c), a left-censored row the distribution function P(y < c), and
an exact row its density.  The remaining three-dimensional posterior (b0, b1, log s2) is
integrated by tensor-product trapezoid quadrature (exponentially convergent for these smooth,
rapidly decaying integrands) on a grid whose (b0, b1) window scales with sqrt(s2); every
number is recomputed on a finer grid and the pair is printed so the grid error is visible.

Reported per table: posterior mean and standard deviation of the cell values b0 (dose 0) and
b0 + b1 (dose 1), the posterior mean and standard deviation of b1, and P(b1 > threshold).

The second part gives the Gibbs full conditionals' exact moments: a single latent outcome with
a Gaussian prior-equivalent N(m, v) (own equation) times Gaussian child equations
y_child ~ N(a + b y, w), truncated to a side of a bound, for the sweep checks.

    uv run --script temporal_observed_bayes_reference.py > temporal_observed_bayes_reference.json

Imports no Antecedent code.  SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import json
import math

import numpy as np
from scipy.stats import norm

X = [0.3, -0.8, 1.1, 0.5, -0.4, 1.4, -1.2, 0.2, 0.9]  # x_0 .. x_8
# Latent outcomes y_1..y_8 (row 0 is history): 1 + 0.8 x_{t-1} + noise.
LATENT = [1.64, -0.14, 2.18, 1.2, 1.28, 1.82, 0.14, 0.76]
Y0 = 1.0
CENSORED = {2: -0.4, 4: 0.9, 7: -0.2}  # t -> bound (right: latent > bound)
THRESHOLD = 0.8


def tables():
    """name -> (kind, rows) with rows[t-1] = ('exact', y) | ('lower', c) | ('upper', c) | ('missing',)."""
    right = [("lower", CENSORED[t]) if t in CENSORED else ("exact", LATENT[t - 1]) for t in range(1, 9)]
    # Left censoring is the mirror image of right censoring in the outcome (not in x).
    left = [("upper", -CENSORED[t]) if t in CENSORED else ("exact", -LATENT[t - 1]) for t in range(1, 9)]
    selected = [("missing",) if t in CENSORED else ("exact", LATENT[t - 1]) for t in range(1, 9)]
    return {"right_censored": right, "left_censored": left, "selected": selected}


def posterior(rows, scale, nb, nu, widen):
    xs = np.array(X[:8])  # x_{t-1} for t = 1..8
    exact = [(i, r[1]) for i, r in enumerate(rows) if r[0] == "exact"]
    design = np.column_stack([np.ones(8), xs])
    ex_idx = [i for i, _ in exact]
    ex_y = np.array([y for _, y in exact])
    # Window centre and scale from the exact rows plus the prior ridge.
    lam = design[ex_idx].T @ design[ex_idx] + np.eye(2) / scale**2
    centre = np.linalg.solve(lam, design[ex_idx].T @ ex_y)
    lam_min = np.linalg.eigvalsh(design.T @ design + np.eye(2) / scale**2).min()
    us = np.linspace(-7.0, 6.5, nu)
    du = us[1] - us[0]
    a_ig = b_ig = 1e-3
    acc = {k: 0.0 for k in ("z", "m0", "m0sq", "m1", "m1sq", "mb", "mbsq", "p")}
    for u in us:
        s2 = math.exp(u)
        s = math.sqrt(s2)
        half = widen * s / math.sqrt(lam_min) + 1.0
        g0 = np.linspace(centre[0] - half, centre[0] + half, nb)
        g1 = np.linspace(centre[1] - half, centre[1] + half, nb)
        h = g0[1] - g0[0]
        B0, B1 = np.meshgrid(g0, g1, indexing="ij")
        logp = -(B0**2 + B1**2) / (2 * s2 * scale**2) - math.log(2 * math.pi * s2 * scale**2)
        logp += -(a_ig + 1) * u - b_ig / s2 + a_ig * math.log(b_ig) - math.lgamma(a_ig) + u  # IG density x Jacobian
        for i, row in enumerate(rows):
            mu = B0 + B1 * xs[i]
            if row[0] == "exact":
                logp += norm.logpdf(row[1], mu, s)
            elif row[0] == "lower":  # latent > bound
                logp += norm.logsf(row[1], mu, s)
            elif row[0] == "upper":  # latent < bound
                logp += norm.logcdf(row[1], mu, s)
        w = np.exp(logp) * h * h * du
        c1 = B0 + B1
        acc["z"] += w.sum()
        acc["m0"] += (w * B0).sum()
        acc["m0sq"] += (w * B0**2).sum()
        acc["m1"] += (w * c1).sum()
        acc["m1sq"] += (w * c1**2).sum()
        acc["mb"] += (w * B1).sum()
        acc["mbsq"] += (w * B1**2).sum()
        acc["p"] += (w * (B1 > THRESHOLD)).sum()
    z = acc["z"]
    out = {}
    for name, key in (("b0", "m0"), ("b0_plus_b1", "m1"), ("b1", "mb")):
        mean = acc[key] / z
        out[name + "_mean"] = mean
        out[name + "_sd"] = math.sqrt(acc[key + "sq"] / z - mean**2)
    out["p_b1_above"] = acc["p"] / z
    return out


def truncated_moments(prior_mean, prior_var, children, side, bound):
    """First two raw moments of y ~ N(m, v) x prod N(c_y; a + b y, w) restricted to y>bound
    ('lower'), y<bound ('upper') or unrestricted ('none')."""
    prec = 1 / prior_var
    lin = prior_mean / prior_var
    for (cy, a, b, w) in children:
        prec += b * b / w
        lin += b * (cy - a) / w
    m, sd = lin / prec, math.sqrt(1 / prec)
    if side == "none":
        mean, var = m, sd * sd
    else:
        alpha = (bound - m) / sd
        if side == "lower":
            lam = norm.pdf(alpha) / norm.sf(alpha)
            mean = m + sd * lam
            var = sd * sd * (1 + alpha * lam - lam * lam)
        else:
            lam = norm.pdf(alpha) / norm.cdf(alpha)
            mean = m - sd * lam
            var = sd * sd * (1 - alpha * lam - lam * lam)
    return {"precision": prec, "linear": lin, "mean": mean, "second": var + mean * mean}


def nig_moments(scale):
    """Exact moments of the conjugate coefficient/variance draw for a tiny complete table."""
    xs = np.array(X[:8])
    y = np.array(LATENT)
    design = np.column_stack([np.ones(8), xs])
    prec0 = np.eye(2) / scale**2
    vn_inv = design.T @ design + prec0
    vn = np.linalg.inv(vn_inv)
    mn = vn @ (design.T @ y)
    a_n = 1e-3 + 4.0
    b_n = 1e-3 + 0.5 * (np.sum((y - design @ mn) ** 2) + mn @ prec0 @ mn)
    return {
        "beta_mean": mn.tolist(),
        "sigma2_mean": b_n / (a_n - 1),
        "beta0_var": float(vn[0, 0] * b_n / (a_n - 1)),
        "beta1_var": float(vn[1, 1] * b_n / (a_n - 1)),
    }


def main():
    result = {"threshold": THRESHOLD, "tables": {}, "conditionals": {}, "conjugate": {}}
    for name, rows in tables().items():
        entry = {"rows": rows}
        for scale in (10.0, 0.5):
            coarse = posterior(rows, scale, 161, 241, 14.0)
            fine = posterior(rows, scale, 241, 401, 18.0)
            err = max(abs(coarse[k] - fine[k]) for k in fine)
            entry[f"scale_{scale}"] = fine
            entry[f"scale_{scale}_grid_difference"] = err
        result["tables"][name] = entry
    cases = {
        # own equation N(0.5, 2) truncated above 1.2, no children
        "lower_no_child": (0.5, 2.0, [], "lower", 1.2),
        "upper_no_child": (0.5, 2.0, [], "upper", -0.3),
        "missing_one_child": (0.5, 2.0, [(2.0, 0.3, 0.7, 0.8)], "none", 0.0),
        # lower-censored with an observed future child and a deep truncation (alpha > 0)
        "lower_with_child": (0.0, 1.0, [(1.5, 0.2, 0.9, 0.5)], "lower", 1.4),
        # latent initial-history row: prior N(0, 4) (the product's N(0, scale^2) initial law), one child
        "initial_history_lower": (0.0, 4.0, [(1.0, 0.5, 0.6, 1.5)], "lower", -0.5),
        "upper_with_two_children": (1.0, 0.5, [(0.1, -0.4, 0.5, 1.0), (-0.6, 0.2, -0.8, 0.3)], "upper", 0.2),
    }
    for name, (m, v, kids, side, bound) in cases.items():
        result["conditionals"][name] = {"prior_mean": m, "prior_var": v, "children": kids, "side": side,
                                        "bound": bound, **truncated_moments(m, v, kids, side, bound)}
    for scale in (10.0, 0.5):
        result["conjugate"][f"scale_{scale}"] = nig_moments(scale)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
