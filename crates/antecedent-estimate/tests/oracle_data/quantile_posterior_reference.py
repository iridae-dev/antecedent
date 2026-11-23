# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Exact posterior of the binned-Dirichlet quantile of one randomized arm.

Regenerate with:  uv run --script quantile_posterior_reference.py > quantile_posterior_reference.json

Within an arm the outcomes fall into the K + 1 bins cut by K increasing thresholds
(-inf, t_0], (t_0, t_1], ..., (t_{K-1}, inf); the bin masses D have a Dirichlet(alpha + counts)
posterior and the distribution function at the grid is F_j = D_0 + .. + D_j.  The tau-quantile
is the piecewise-linear inverse: it crosses in the bracket j >= 1 with F_{j-1} < tau <= F_j at
t_{j-1} + (tau - F_{j-1}) / D_j * (t_j - t_{j-1}), and the draw is unresolved when there is no
crossing or when D_j (t_{K-1} - t_0) / (t_j - t_{j-1}) < 1e-3.
For the crossing bracket j put A = F_{j-1}, B = D_j and C = the remaining bins; (A, B, C) is
Dirichlet(a_A, a_B, a_C), so B ~ Beta(a_B, a_A + a_C) and U = A / (1 - B) ~ Beta(a_A, a_C) is
independent of B.  Then, with lam = (x - t_{j-1}) / (t_j - t_{j-1}),
    P(cross in j, Q <= x, resolved) = int_{b >= delta_j} pdf_B(b) P(A in [tau - lam b, tau)) db,
a one-dimensional integral of a Beta tail, evaluated at 15 digits by Gauss-Legendre quadrature.  The probability of a resolved
draw, the distribution function of the quantile given resolution, its mean and its equal-tailed
95% interval follow; none of it uses the product's sampler or inversion.  Repository-authored
(MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import json

import mpmath as mp

mp.mp.dps = 15

GRID = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
CONTROL = [0.4, 0.8, 1.1, 1.3, 1.7, 2.2, 2.4, 2.9, 3.3, 3.6, 4.1, 4.6]
ACTIVE = [0.9, 1.4, 1.8, 2.1, 2.6, 2.8, 3.1, 3.4, 3.9, 4.3, 4.4, 4.8]


def counts(values, grid):
    c = [0] * (len(grid) + 1)
    for v in values:
        c[sum(1 for t in grid if t < v)] += 1
    return c


def beta_sf(x, a, b):
    if x <= 0: return mp.mpf(1)
    if x >= 1: return mp.mpf(0)
    return 1 - mp.betainc(a, b, 0, x, regularized=True)


def beta_pdf(x, a, b):
    return x ** (a - 1) * (1 - x) ** (b - 1) / mp.beta(a, b)


def arm(alpha, grid, tau):
    K = len(grid)
    span = mp.mpf(grid[-1]) - grid[0]
    tau = mp.mpf(tau)
    brackets = []
    for j in range(1, K):
        aA = sum(alpha[:j]); aB = alpha[j]; aC = sum(alpha[j + 1:])
        w = mp.mpf(grid[j]) - grid[j - 1]
        delta = mp.mpf(1) / 1000 * w / span
        brackets.append((j, mp.mpf(aA), mp.mpf(aB), mp.mpf(aC), w, delta))

    def cross(j, aA, aB, aC, w, delta, lam):
        """P(crossing in bracket j, A + lam B >= tau, guard satisfied)."""
        def integrand(b):
            lo = (tau - lam * b) / (1 - b)
            hi = tau / (1 - b)
            p = beta_sf(lo, aA, aC) - beta_sf(hi, aA, aC)
            return beta_pdf(b, aB, aA + aC) * p
        # The integrand has kinks where the Beta tail argument enters or leaves (0, 1):
        # lo = 0 at b = tau / lam and hi = 1 at b = 1 - tau; split the range there.
        kinks = [delta, 1 - tau, tau / lam if lam > 0 else mp.mpf(2), mp.mpf(1) / 2, mp.mpf(1)]
        pts = sorted({k for k in kinks if delta <= k <= 1})
        return mp.quad(integrand, pts, method='gauss-legendre', maxdegree=5)

    resolved = sum(cross(*br, 1) for br in brackets)

    def joint(x):
        x = mp.mpf(x)
        total = mp.mpf(0)
        for br in brackets:
            j, aA, aB, aC, w, delta = br
            lo_t = mp.mpf(grid[j - 1])
            if x >= lo_t + w:
                total += cross(*br, 1)
            elif x > lo_t:
                total += cross(*br, (x - lo_t) / w)
        return total
    cdf = lambda x: joint(x) / resolved
    mean = mp.mpf(grid[0]) + mp.quad(lambda x: 1 - cdf(x), grid, method='gauss-legendre', maxdegree=3)
    def q(p):
        lo, hi = mp.mpf(grid[0]), mp.mpf(grid[-1])
        for _ in range(26):
            mid = (lo + hi) / 2
            if cdf(mid) < p: lo = mid
            else: hi = mid
        return (lo + hi) / 2
    return {"resolved": float(resolved), "mean": float(mean), "q025": float(q(mp.mpf("0.025"))), "q975": float(q(mp.mpf("0.975"))),
            "cdf_at": [[x, float(cdf(x))] for x in (0.5, 1.5, 2.25, 2.75, 3.4, 4.2)]}


def case(name, concentration, tau):
    out = {"name": name, "concentration": concentration, "tau": tau, "grid": GRID, "control_data": CONTROL, "active_data": ACTIVE}
    for label, vals in (("control", CONTROL), ("treated", ACTIVE)):
        c = counts(vals, GRID)
        alpha = [mp.mpf(concentration) + k for k in c]
        out[label] = arm(alpha, GRID, tau)
        tot = sum(alpha)
        run = 0; cm = []
        for a in alpha[:-1]:
            run += a; cm.append(float(run / tot))
        out[label]["cdf_mean"] = cm
    return out


def main():
    print(json.dumps([case("moderate_prior_median", 0.5, 0.5), case("moderate_prior_upper_quartile", 0.5, 0.75),
                      case("vague_prior_lower_quartile", 0.01, 0.25), case("strong_prior_median", 20.0, 0.5)], indent=1))


if __name__ == "__main__":
    main()
