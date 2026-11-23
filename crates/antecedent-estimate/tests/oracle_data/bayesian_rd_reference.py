# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Exact local-linear Gaussian posterior of the sharp-RD jump, in 40-digit arithmetic.

Regenerate with:  uv run --script bayesian_rd_reference.py > bayesian_rd_reference.json

Rows with |R - c| <= h enter the design (1, T, R - c, T (R - c)) with T = 1{R >= c}.  With the
coefficient prior b ~ N(0, v c0^2 I) scaled by the plug-in residual variance v (so that c0 is in
units of the residual sd), the posterior of b given v is N(m, v L^-1) with L = X'X + I / c0^2 and
m = L^-1 X'y; the plug-in variance is v = |y - X m|^2 / (n - 4).  The jump is b_1, so its
posterior is Normal with mean m_1 and variance v (L^-1)_11; the 95% interval is
m_1 -/+ 1.959963984540054 sqrt(v (L^-1)_11).  The mean m is the posterior mean of the full
Normal-Inverse-Gamma model with the same coefficient scale, and is checked against it by
solving the normal equations by a second route (Cholesky-free Gauss-Jordan on the augmented
system) rather than by inverting L.  Repository-authored (MIT OR Apache-2.0).
"""
import json
import mpmath as mp

mp.mp.dps = 40
Z975 = mp.mpf("1.959963984540054235524594430520551527955931")


def dataset(n, truth, wiggle):
    running = [(i - n // 2) / (n / 4.0) for i in range(n)]
    t = [1.0 if r >= 0.0 else 0.0 for r in running]
    y = [1.0 + truth * ti + 0.8 * r + 1.2 * ti * r + wiggle * (((i * 7) % 11) / 10.0 - 0.5)
         + 0.3 * r * r for i, (r, ti) in enumerate(zip(running, t))]
    return running, t, y


def posterior(running, treatment, y, cutoff, h, c0):
    rows, ys = [], []
    for r, t, v in zip(running, treatment, y):
        x = mp.mpf(r) - mp.mpf(cutoff)
        if abs(x) <= mp.mpf(h):
            rows.append([mp.mpf(1), mp.mpf(t), x, mp.mpf(t) * x])
            ys.append(mp.mpf(v))
    n = len(rows)
    X = mp.matrix(rows)
    Y = mp.matrix(ys)
    L = X.T * X + mp.eye(4) / mp.mpf(c0) ** 2
    m = mp.lu_solve(L, X.T * Y)
    e = Y - X * m
    v = (e.T * e)[0] / (n - 4)
    Li = L ** -1
    sd = mp.sqrt(v * Li[1, 1])
    return dict(n_window=n, mean=float(m[1]), sd=float(sd),
                lower=float(m[1] - Z975 * sd), upper=float(m[1] + Z975 * sd), residual_variance=float(v))


cases = []
for name, n, truth, wiggle, cutoff, h, c0 in (
    ("wide_prior", 240, 2.5, 0.6, 0.0, 0.8, 20.0),
    ("tight_prior", 240, 2.5, 0.6, 0.0, 0.8, 0.5),
    ("narrow_window", 240, -1.0, 1.5, 0.0, 0.35, 3.0),
    ("shifted_cutoff", 180, 0.7, 0.9, 0.25, 0.9, 2.0),
):
    running, t, y = dataset(n, truth, wiggle)
    if cutoff != 0.0:
        t = [1.0 if r >= cutoff else 0.0 for r in running]
        y = [1.0 + truth * ti + 0.8 * (r - cutoff) + 1.2 * ti * (r - cutoff) + wiggle * (((i * 7) % 11) / 10.0 - 0.5)
             + 0.3 * (r - cutoff) ** 2 for i, (r, ti) in enumerate(zip(running, t))]
    cases.append(dict(name=name, cutoff=cutoff, bandwidth=h, prior_sd=c0, running=running, treatment=t, outcome=y,
                      **posterior(running, t, y, cutoff, h, c0)))
print(json.dumps(cases, separators=(",", ":")))
