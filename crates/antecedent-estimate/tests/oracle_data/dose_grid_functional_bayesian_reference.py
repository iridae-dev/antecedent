# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Reference for the joint polynomial dose-grid posterior (level, derivative, contrast).

Regenerate with:  uv run --script dose_grid_functional_bayesian_reference.py > dose_grid_functional_bayesian_reference.json

The curve is m(d) = sum_j beta_j z^j with z = (2d - lo - hi)/(hi - lo) and the prior
beta_j | s2 ~ N(0, s2 sj^2).  With A = X'X + diag(1/sj^2) and V = A^-1 the coefficient posterior
mean is V X'y.  For a known residual variance v the covariance is v V.  For an inverse-gamma
residual prior (a, b) the posterior is normal-inverse-gamma with a_n = a + n/2 and
b_n = b + (y'y - mean' A mean)/2; the coefficient marginal is Student-t with covariance
b_n/(a_n - 1) V.  A functional is linear in the coefficients, f = B beta, with B the rows of the
level z^j, the derivative (2/(hi-lo)) j z^(j-1) or the contrast difference of two level rows, so its
mean is B mean and its covariance is B Cov B'.  Everything is evaluated at 50 digits from the exact
binary64 inputs.  Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import json

import mpmath as mp

mp.mp.dps = 50


class Lcg:
    def __init__(self, s):
        self.s = s

    def u(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) % 2**64
        return (self.s >> 11) / float(1 << 53)


def rows_for(functional, points, width, lo, hi):
    lo, hi = mp.mpf(lo), mp.mpf(hi)
    z = lambda d: (2 * mp.mpf(d) - lo - hi) / (hi - lo)
    level = lambda d: [z(d) ** j for j in range(width)]
    slope = lambda d: [0 if j == 0 else 2 / (hi - lo) * j * z(d) ** (j - 1) for j in range(width)]
    if functional == "level":
        return [level(d) for d in points]
    if functional == "derivative":
        return [slope(d) for d in points]
    return [[a - b for a, b in zip(level(points[1]), level(points[0]))]]


def posterior(case):
    dose, y, lo, hi, s = case["dose"], case["outcome"], case["lo"], case["hi"], case["scales"]
    w = len(s)
    X = [[(((2 * mp.mpf(d) - mp.mpf(lo) - mp.mpf(hi)) / (mp.mpf(hi) - mp.mpf(lo))) ** j) for j in range(w)] for d in dose]
    A = mp.zeros(w, w)
    b = mp.zeros(w, 1)
    for row, yy in zip(X, y):
        for i in range(w):
            b[i] += row[i] * mp.mpf(yy)
            for j in range(w):
                A[i, j] += row[i] * row[j]
    for j in range(w):
        A[j, j] += 1 / mp.mpf(s[j]) ** 2
    V = A ** -1
    m = V * b
    if "known_variance" in case:
        scale = mp.mpf(case["known_variance"])
        extra = {}
    else:
        a, bb = mp.mpf(case["ig_shape"]), mp.mpf(case["ig_scale"])
        n = len(dose)
        yy = sum(mp.mpf(v) ** 2 for v in y)
        quad = sum(m[i] * A[i, j] * m[j] for i in range(w) for j in range(w))
        an, bn = a + mp.mpf(n) / 2, bb + (yy - quad) / 2
        scale = bn / (an - 1)
        extra = {"alpha_n": float(an), "beta_n": float(bn)}
    pts = [case["from"], case["to"]] if case["functional"] == "contrast" else case["grid"]
    B = rows_for(case["functional"], pts, w, lo, hi)
    mean = [sum(r[i] * m[i] for i in range(w)) for r in B]
    cov = []
    for r in B:
        vr = [sum(V[i, j] * r[j] for j in range(w)) for i in range(w)]
        for q in B:
            cov.append(scale * sum(vr[i] * q[i] for i in range(w)))
    return {"mean": [float(x) for x in mean], "cov": [float(x) for x in cov], "coef": [float(m[i]) for i in range(w)], **extra}


def make(name, n, lo, hi, offset, scales, functional, seed, **kw):
    g = Lcg(seed)
    dose = sorted(offset + lo + (hi - lo) * g.u() for _ in range(n))
    y = [1.0 + 0.7 * (d - offset - lo) - 0.2 * (d - offset - lo) ** 2 + 0.3 * (g.u() - 0.5) for d in dose]
    case = {"name": name, "dose": dose, "outcome": y, "lo": offset + lo, "hi": offset + hi,
            "scales": scales, "functional": functional}
    grid = [dose[len(dose) // 4], dose[len(dose) // 2], dose[-2]]
    if functional == "contrast":
        case["from"], case["to"] = dose[0], dose[-1]
    else:
        case["grid"] = grid
    case.update(kw)
    if "support" in case:
        case["lo"], case["hi"] = case.pop("support")
    case["expected"] = posterior(case)
    return case


cases = [
    make("level_known", 14, 0.0, 4.0, 0.0, [3.0, 2.0, 1.0], "level", 1, known_variance=0.25),
    make("derivative_known", 14, 0.0, 4.0, 0.0, [3.0, 2.0, 1.0], "derivative", 2, known_variance=0.25),
    make("contrast_known", 14, 0.0, 4.0, 0.0, [3.0, 2.0, 1.0], "contrast", 3, known_variance=0.25),
    # support and doses offset far from zero: the rescaling to [-1, 1] must not lose the curve
    make("offset_level", 12, 0.0, 4.0, 1.0e6, [3.0, 2.0, 1.0], "level", 4, known_variance=0.25),
    make("offset_derivative", 12, 0.0, 4.0, -3.0e5, [3.0, 2.0, 1.0], "derivative", 5, known_variance=0.25),
    # more coefficients than rows: the prior alone identifies the curve
    make("wider_than_rows", 4, 0.0, 2.0, 0.0, [2.0, 1.5, 1.0, 0.5, 0.25], "level", 6, known_variance=0.5),
    make("minimum_rows_contrast", 3, 0.0, 2.0, 0.0, [2.0, 1.0], "contrast", 7, known_variance=0.5),
    # a support wider than the doses, and doses pinned on the support ends
    make("wide_support_level", 16, 0.0, 4.0, 0.0, [3.0, 2.0, 1.0], "level", 8, known_variance=0.3),
    # the declared support is wider than the observed range
    make("support_wider_than_data", 16, 0.0, 2.0, 0.0, [3.0, 2.0, 1.0], "level", 9, known_variance=0.3, support=[-5.0, 7.0]),
    # tight and diffuse prior scales
    make("diffuse_prior", 14, 0.0, 4.0, 0.0, [1e6, 1e6, 1e6], "level", 10, known_variance=0.25),
    make("tight_prior", 14, 0.0, 4.0, 0.0, [1e-3, 1e-3, 1e-3], "level", 11, known_variance=0.25),
    make("inverse_gamma_level", 20, 0.0, 4.0, 0.0, [3.0, 2.0, 1.0], "level", 12, ig_shape=3.0, ig_scale=0.5),
    make("inverse_gamma_contrast", 20, 0.0, 4.0, 0.0, [3.0, 2.0, 1.0], "contrast", 13, ig_shape=4.0, ig_scale=0.2),
    make("inverse_gamma_derivative", 20, 0.0, 4.0, 2.0e5, [3.0, 2.0, 1.0], "derivative", 14, ig_shape=2.5, ig_scale=0.4),
]
print(json.dumps(cases))
