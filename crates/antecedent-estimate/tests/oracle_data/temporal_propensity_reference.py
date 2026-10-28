# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""40-digit reference for the temporal propensity (ridge mode and Laplace posterior).

Regenerate with:  uv run --script temporal_propensity_reference.py > temporal_propensity_reference.json

Rows are independent subjects with an intercept and two lag columns; the outcome is the
treatment indicator.  Ridge: the maximiser of
  sum_i [a_i eta_i - log(1 + e^eta_i)] - (lambda / 2) (b_1^2 + b_2^2),   eta = X b,
with the intercept unpenalized, found by Newton's method; the score at a lag history is
sigmoid(b0 + b1 z1 + b2 z2).  Posterior: independent Gaussian priors N(0, sd0^2) on the intercept and
N(0, sd1^2) on the slopes; the Laplace approximation has the log-posterior mode, covariance
(X'WX + diag(1/sd^2))^-1 with W = mu (1 - mu), and the posterior-mean score at a lag history is
the Gaussian expectation of sigmoid(eta), eta ~ N(x'm, x'Cx), by quadrature.
Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import json

import mpmath as mp

mp.mp.dps = 40


class Lcg:
    def __init__(self, s):
        self.s = s

    def u(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) % 2**64
        return (self.s >> 11) / float(1 << 53)

    def normal(self):
        return sum(self.u() for _ in range(12)) - 6.0


def s(x):
    return mp.nstr(x, 30)


sig = lambda e: 1 / (1 + mp.e ** (-e))
phi = lambda x: mp.e ** (-x * x / 2) / mp.sqrt(2 * mp.pi)


def data(n, seed):
    rng = Lcg(seed)
    rows, a = [], []
    for _ in range(n):
        z1, z2 = rng.normal(), rng.normal() * 3 + 5  # second lag on a different scale
        eta = -0.4 + 0.9 * z1 - 0.15 * (z2 - 5)
        rows.append([1.0, z1, z2])
        a.append(1.0 if rng.u() < sig(eta) else 0.0)
    return rows, a


def newton(rows, a, penalties):
    p = 3
    X = [mp.matrix(r) for r in rows]
    b = mp.zeros(p, 1)
    for _ in range(300):
        g = mp.zeros(p, 1)
        H = mp.zeros(p, p)
        for x, ai in zip(X, a):
            e = (x.T * b)[0]
            m = sig(e)
            g += (ai - m) * x
            H += m * (1 - m) * (x * x.T)
        for j in range(p):
            g[j] -= penalties[j] * b[j]
            H[j, j] += penalties[j]
        step = H ** -1 * g
        b += step
        if mp.norm(step) < mp.mpf("1e-35"):
            return b, H
    raise RuntimeError("no convergence")


def main():
    rows, a = data(220, 31337)
    histories = [[0.0, 5.0], [1.2, 3.0], [-0.8, 8.0]]
    ridge = []
    for lam in (0.5, 8.0, 400.0):
        b, _ = newton(rows, a, [0, lam, lam])
        ridge.append({
            "penalty": lam, "coefficients": [s(v) for v in b],
            "scores": [s(sig(b[0] + b[1] * h[0] + b[2] * h[1])) for h in histories],
        })
    post = []
    for sd0, sd1 in ((5.0, 2.0), (0.5, 0.5), (1.0, 0.1)):
        pen = [1 / mp.mpf(sd0) ** 2, 1 / mp.mpf(sd1) ** 2, 1 / mp.mpf(sd1) ** 2]
        b, H = newton(rows, a, pen)
        C = H ** -1
        means = []
        for h in histories:
            x = mp.matrix([1, h[0], h[1]])
            m = (x.T * b)[0]
            v = (x.T * C * x)[0]
            sd = mp.sqrt(v)
            means.append(s(mp.quad(lambda t: sig(m + sd * t) * phi(t), [-12, 0, 12])))
        post.append({
            "intercept_sd": sd0, "slope_sd": sd1, "mode": [s(v) for v in b],
            "covariance": [[s(C[i, j]) for j in range(3)] for i in range(3)],
            "mode_scores": [s(sig(b[0] + b[1] * h[0] + b[2] * h[1])) for h in histories],
            "posterior_mean_scores": means,
        })
    print(json.dumps({"rows": rows, "treatment": a, "histories": histories,
                      "ridge": ridge, "posterior": post}))


main()
