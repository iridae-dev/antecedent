# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Exact Laplace posterior of a static logistic propensity under a Gaussian coefficient prior.

Regenerate with:  uv run --script propensity_posterior_reference.py > propensity_posterior_reference.json

Model: treatment t_i ~ Bernoulli(expit(z_i'b)), z_i = (1, x_i), b_j ~ N(0, prior_sd^2) independently (the
intercept included).  The mode m solves  sum_i (t_i - expit(z_i'm)) z_i = m / prior_sd^2  (Newton at
40 digits) and the Laplace covariance is C = (Z' W Z + I / prior_sd^2)^-1 with W = diag(p_i (1 - p_i)) at m.
For a row z the linear predictor eta = z'b is Normal(z'm, z'Cz), so the score s = expit(eta) has
  P(s < clip) = Phi((logit(clip) - z'm) / sd), P(s > 1 - clip) = 1 - Phi((logit(1 - clip) - z'm) / sd),
and mean and standard deviation by Gauss-Hermite-free adaptive quadrature against the Normal density
of eta.  The score at the mode is expit(z'm).  Values are written as nearest binary64.
Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import json
import mpmath as mp

mp.mp.dps = 40


def expit(v):
    return 1 / (1 + mp.exp(-v))


def design(n):
    rows, t = [], []
    for i in range(n):
        x1 = (((i * 7) % 17) / 8.0) - 1.0
        x2 = 1.0 if (i % 3 == 0) else 0.0
        eta = -0.3 + 1.1 * x1 + 0.8 * x2
        u = ((i * 37 + 5) % 41) / 41.0
        rows.append([1.0, x1, x2])
        t.append(1.0 if u < float(expit(mp.mpf(eta))) else 0.0)
    return rows, t


def posterior(rows, t, sd):
    n, p = len(rows), len(rows[0])
    Z = mp.matrix(rows)
    m = mp.matrix(p, 1)
    for _ in range(200):
        pr = [expit(sum(m[j] * rows[i][j] for j in range(p))) for i in range(n)]
        g = mp.matrix(p, 1)
        H = mp.matrix(p, p)
        for i in range(n):
            for a in range(p):
                g[a] += (t[i] - pr[i]) * rows[i][a]
                for c in range(p):
                    H[a, c] += pr[i] * (1 - pr[i]) * rows[i][a] * rows[i][c]
        g -= m / mp.mpf(sd) ** 2
        H += mp.eye(p) / mp.mpf(sd) ** 2
        step = mp.lu_solve(H, g)
        m += step
        if mp.norm(step) < mp.mpf(10) ** -30:
            break
    pr = [expit(sum(m[j] * rows[i][j] for j in range(p))) for i in range(n)]
    H = mp.eye(p) / mp.mpf(sd) ** 2
    for i in range(n):
        for a in range(p):
            for c in range(p):
                H[a, c] += pr[i] * (1 - pr[i]) * rows[i][a] * rows[i][c]
    return m, H ** -1


def score_summary(m, C, z, clip):
    zv = mp.matrix(z)
    loc = (zv.T * m)[0]
    var = (zv.T * C * zv)[0]
    sd = mp.sqrt(var)
    dens = lambda e: mp.npdf(e, loc, sd)
    mean = mp.quad(lambda e: expit(e) * dens(e), [loc - 12 * sd, loc - 4 * sd, loc, loc + 4 * sd, loc + 12 * sd])
    second = mp.quad(lambda e: expit(e) ** 2 * dens(e), [loc - 12 * sd, loc - 4 * sd, loc, loc + 4 * sd, loc + 12 * sd])
    logit = lambda c: mp.log(mp.mpf(c) / (1 - mp.mpf(c)))
    outside = mp.ncdf((logit(clip) - loc) / sd) + 1 - mp.ncdf((logit(1 - mp.mpf(clip)) - loc) / sd)
    return dict(at_mode=float(expit(loc)), mean=float(mean), sd=float(mp.sqrt(second - mean ** 2)), outside_band=float(outside))


rows, t = design(80)
cases = []
for sd, clip in ((2.0, 0.2), (0.5, 0.3), (10.0, 0.1)):
    m, C = posterior(rows, t, sd)
    probes = [[1.0, -1.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [1.0, 0.5, 0.0]]
    cases.append(dict(prior_sd=sd, clip=clip, mode=[float(v) for v in m], probes=probes,
                      scores=[score_summary(m, C, z, clip) for z in probes]))
print(json.dumps(dict(rows=rows, treatment=t, cases=cases), separators=(",", ":")))
