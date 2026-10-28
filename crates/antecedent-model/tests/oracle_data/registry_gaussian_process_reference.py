# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Gaussian-process regression by exact Cholesky, in 60-digit arithmetic.

Regenerate with:  uv run --script registry_gaussian_process_reference.py > registry_gaussian_process_reference.json

For each design the reference evaluates the negative log marginal likelihood of every one of the
25 cells of the fixed hyperparameter grid and keeps the first minimiser:
  length scale   multiples (0.25 0.5 1 2 4) of sqrt(mean over parents of the population variance),
  noise sd       multiples (0.05 0.1 0.2 0.5 1) of the outcome's population sd,
  signal variance the outcome's population variance, prior mean its sample mean,
  K = v exp(-|x - x'|^2 / (2 l^2)) + s^2 I,  alpha = K^-1 (y - mean),
  NLML = (y - mean)' alpha / 2 + log det K / 2 + n log(2 pi) / 2,
with log det K from an LDL' factorisation (mp.cholesky) and K^-1 by LU.  The script asserts the
winning cell leads the runner-up by more than 1e-6 in NLML, so the argmin is unambiguous, and
records the posterior mean at held-out points, the exact posterior variance (which the library
does not use: it scores a point as a Gaussian about the posterior mean with the fitted noise sd),
and the Gaussian log density about the posterior mean.
Repository-authored (MIT OR Apache-2.0).
"""
import json
import random
from fractions import Fraction as F

import mpmath as mp

mp.mp.dps = 60


def mpf(f):
    f = F(f)
    return mp.mpf(f.numerator) / f.denominator


def kern(a, b, var, ell):
    d2 = sum((u - v) ** 2 for u, v in zip(a, b))
    return var * mp.e ** (-d2 / (2 * ell * ell))


def case(name, n, p, rng, kind):
    q = lambda lo, hi, den: F(rng.randint(lo * den, hi * den), den)
    X = [[q(-3, 3, 32) for _ in range(p)] for _ in range(n)]
    if kind == "smooth":
        y = [F(round(float(mp.sin(mpf(r[0])) * 2 + (0 if p == 1 else mp.cos(mpf(r[1])))) * 4096), 4096)
             + q(-1, 1, 64) for r in X]
    else:  # offset scale: parents in years, outcome far from zero
        X = [[r[0] + 2020] + r[1:] for r in X]
        y = [F(1000) + F(round(float(mp.sin(mpf(r[0] - 2020)) * 50) * 64), 64) + q(-4, 4, 16) for r in X]
    Xm = [[mpf(v) for v in r] for r in X]
    ym = [mpf(v) for v in y]
    mean = sum(ym) / n
    var = sum((v - mean) ** 2 for v in ym) / n
    x_var = sum(sum((r[c] - sum(rr[c] for rr in Xm) / n) ** 2 for r in Xm) / n for c in range(p)) / p
    s_x, s_y = mp.sqrt(x_var), mp.sqrt(var)
    best = None
    scores = []
    for lm in (0.25, 0.5, 1, 2, 4):
        for sm in (0.05, 0.1, 0.2, 0.5, 1):
            ell, sd = lm * s_x, sm * s_y
            K = mp.matrix(n, n)
            for i in range(n):
                for j in range(n):
                    K[i, j] = kern(Xm[i], Xm[j], var, ell) + (sd * sd if i == j else 0)
            yc = mp.matrix([v - mean for v in ym])
            alpha = mp.lu_solve(K, yc)
            L = mp.cholesky(K)
            logdet = 2 * sum(mp.log(L[i, i]) for i in range(n))
            nlml = (yc.T * alpha)[0] / 2 + logdet / 2 + n * mp.log(2 * mp.pi) / 2
            scores.append(nlml)
            if best is None or nlml < best[0]:
                best = (nlml, ell, sd, alpha, K)
    scores.sort()
    assert scores[1] - scores[0] > mp.mpf("1e-6"), (name, scores[:2])
    nlml, ell, sd, alpha, K = best
    # held-out points
    tests = []
    for t in range(6):
        pt = [q(-3, 3, 32) + (2020 if kind != "smooth" and c == 0 else 0) for c in range(p)]
        pm = [mpf(v) for v in pt]
        kvec = mp.matrix([kern(pm, Xm[i], var, ell) for i in range(n)])
        mu = mean + (kvec.T * alpha)[0]
        post_var = var - (kvec.T * mp.lu_solve(K, kvec))[0]
        yt = mu + mpf(q(-1, 1, 16))
        lp = -mp.log(2 * mp.pi * sd * sd) / 2 - (yt - mu) ** 2 / (2 * sd * sd)
        tests.append({"x": [float(v) for v in pt], "mean": float(mu), "posterior_variance": float(post_var),
                      "y": float(yt), "log_density": float(lp)})
    return {"name": name, "x": [[float(v) for v in r] for r in X], "y": [float(v) for v in y],
            "length_scale": float(ell), "noise_std": float(sd), "variance": float(var), "mean": float(mean),
            "alpha": [float(a) for a in alpha], "nlml": float(nlml), "nlml_margin": float(scores[1] - scores[0]),
            "tests": tests}


def main():
    rng = random.Random(77)
    out = [case("one_parent", 24, 1, rng, "smooth"), case("two_parents", 26, 2, rng, "smooth"),
           case("offset_scale", 22, 2, rng, "offset")]
    print(json.dumps(out, indent=1))


main()
