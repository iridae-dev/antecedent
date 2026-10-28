# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Reference values for the Beta-Bernoulli posterior functionals of the adaptive-inference module.

Regenerate with:  uv run --script beta_bernoulli_reference.py > beta_bernoulli_reference.json

Written from the definitions at 25 digits with mpmath's tanh-sinh quadrature and regularized
incomplete beta (a different algorithm from the product's continued fraction and Gauss-Legendre
panels):
  I_x(a, b)                = mp.betainc(a, b, 0, x, regularized=True)
  P(arm k best)            = int f_k(x) prod_{j != k} F_j(x) dx
  P(theta_a - theta_b > m) = int_m^1 f_a(x) F_b(x - m) dx
  E[max_j theta_j]         = int_0^1 (1 - prod_j F_j(t)) dt,   loss_k = E[max] - E[theta_k]
For integer shapes the two-arm superiority is also summed exactly in rational arithmetic (see
exact_superior: conditioning on X = x, P(Y < x) is a binomial tail, which integrates against the
Beta density to a finite sum of Beta functions).

Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import json
from fractions import Fraction
from math import comb

import mpmath as mp

mp.mp.dps = 15
mpf = mp.mpf


def pdf(a, b, x):
    return mp.exp((a - 1) * mp.log(x) + (b - 1) * mp.log1p(-x) - mp.log(mp.beta(a, b))) if 0 < x < 1 else mpf(0)


def cdf(a, b, x):
    """Regularized incomplete beta, retrying with the reflection and more digits when the
    hypergeometric series of mpmath does not converge at the working precision."""
    if x <= 0:
        return mpf(0)
    if x >= 1:
        return mpf(1)
    for dps in (None, 30, 60):
        ctx = mp.workdps(dps) if dps else mp.workdps(mp.mp.dps)
        with ctx:
            for attempt in ("direct", "reflected"):
                try:
                    if attempt == "direct":
                        return +mp.betainc(a, b, 0, x, regularized=True)
                    return +(1 - mp.betainc(b, a, 0, 1 - x, regularized=True))
                except ValueError:
                    continue
    raise SystemExit(f"incomplete beta did not converge for ({a}, {b}) at {x}")


def points(shapes, lo=0, hi=1):
    pts = {mpf(lo), mpf(hi)}
    for a, b in shapes:
        mean = mpf(a) / (a + b)
        sd = mp.sqrt(mpf(a) * b / ((a + b) ** 2 * (a + b + 1)))
        for k in (-8, -4, -2, -1, 0, 1, 2, 4, 8):
            x = mean + k * sd
            if lo < x < hi:
                pts.add(x)
    return sorted(pts)


def best(shapes):
    out = []
    for k in range(len(shapes)):
        def f(x, k=k):
            v = pdf(*shapes[k], x)
            for j, s in enumerate(shapes):
                if j != k:
                    v *= cdf(*s, x)
            return v
        out.append(mp.quad(f, points(shapes)))
    return out


def exceeds(sa, sb, margin):
    m = mpf(margin)
    return mp.quad(lambda x: pdf(*sa, x) * (cdf(*sb, x - m) if x > m else 0), points([sa, sb], m, 1))


def losses(shapes):
    emax = mp.quad(lambda t: 1 - mp.fprod(cdf(*s, t) for s in shapes), points(shapes))
    return [emax - mpf(a) / (a + b) for a, b in shapes]


def exact_superior(a1, b1, a2, b2):
    """P(X > Y), X ~ Beta(a1, b1), Y ~ Beta(a2, b2), integer shapes.  Conditioning on X = x,
    P(Y < x) = P(Binomial(a2 + b2 - 1, x) >= a2); integrating the binomial tail against the Beta
    density gives a finite sum of Beta functions, evaluated in exact rationals."""
    n = a2 + b2 - 1
    total = Fraction(0)
    for j in range(a2, n + 1):
        # int x^(a1 - 1 + j) (1 - x)^(b1 - 1 + n - j) dx / B(a1, b1) = B(a1 + j, b1 + n - j) / B(a1, b1)
        num = Fraction(1) * comb(n, j) * Fraction(
            __import__("math").factorial(a1 + j - 1) * __import__("math").factorial(b1 + n - j - 1),
            __import__("math").factorial(a1 + b1 + n - 1),
        )
        den = Fraction(__import__("math").factorial(a1 - 1) * __import__("math").factorial(b1 - 1), __import__("math").factorial(a1 + b1 - 1))
        total += num / den
    return total


def ns(x):
    return mp.nstr(x, 22)


def main():
    out = {"best": [], "exceeds": [], "losses": [], "incomplete_beta": [], "exact_superior": []}
    arm_sets = [
        [(1, 1), (1, 1)],
        [(1, 1), (1, 1), (1, 1)],
        [(8, 4), (3, 9)],
        [(1, 40), (1, 1)],
        [(1.5, 2.5), (2.5, 1.5), (4.5, 6.5)],
        [(120, 880), (130, 870)],
        [(300, 700), (310, 690), (280, 720)],
        [(30, 70), (31, 69), (25, 75), (40, 60), (29.5, 70.5)],
        [(1, 100), (1, 150), (1, 200)],
        [(100, 1), (150, 1)],
    ]
    for shapes in arm_sets:
        out["best"].append({"shapes": shapes, "probabilities": [ns(v) for v in best(shapes)]})
        out["losses"].append({"shapes": shapes, "losses": [ns(v) for v in losses(shapes)]})
    for sa, sb in (((8, 4), (3, 9)), ((3, 2), (3, 2)), ((120, 880), (100, 900)), ((20.5, 30.5), (10.5, 40.5)), ((1, 1), (1, 1))):
        for margin in (0.0, 0.05, 0.3, 0.7, 0.999):
            out["exceeds"].append({"a": sa, "b": sb, "margin": margin, "p": ns(exceeds(sa, sb, margin))})
    for a, b in ((1, 1), (0.5, 0.5), (2, 5), (30.5, 2.25), (1000, 3000)):
        for x in (1e-9, 0.01, 0.25, 0.5, 0.9, 0.999999):
            out["incomplete_beta"].append({"a": a, "b": b, "x": x, "value": ns(cdf(a, b, mpf(x)))})
    for a1, b1, a2, b2 in ((8, 4, 3, 9), (1, 1, 1, 1), (2, 40, 20, 3), (51, 51, 50, 52), (1, 200, 1, 150), (300, 100, 290, 110), (3000, 7000, 2900, 7100),
                           (1, 3000, 1, 3001), (1, 100, 1, 150), (1, 20, 1, 30), (2, 3000, 1, 3000), (3000, 1, 3001, 1), (50, 1, 200, 1)):
        v = exact_superior(a1, b1, a2, b2)
        out["exact_superior"].append({"shapes": [[a1, b1], [a2, b2]], "numerator": str(v.numerator), "denominator": str(v.denominator),
                                      "float": ns(mpf(v.numerator) / v.denominator)})
    print(json.dumps(out, separators=(",", ":")))


main()
