# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""40-digit reference for the two dose-curve posterior routes.

Regenerate with:  uv run --script dose_posterior_reference.py > dose_posterior_reference.json

Local quadratic (posterior_mean.rs).  At a dose d0 with Gaussian weights
w_i = exp(-(D_i - d0)^2 / (2 h^2)) and the design x_i = (1, u_i, u_i^2), u_i = D_i - d0:
  G = sum w_i x_i x_i',  r = sum w_i x_i y_i,  W = sum w_i,
  b0 = G^-1 r,           RSS = sum w_i (y_i - x_i'b0)^2,   v = RSS / (W - 3),
  G' = G + diag(0, v / s^2, v / s^2),   b = G'^-1 r,
and the published level and derivative are b[0] and b[1] (the intercept is not penalized).

Fixed group-dose policy (continuous_dose_posterior.rs).  Each group g has its own polynomial
curve m_g(d) = sum_j beta_j z(d)^j, z = (2d - lo - hi) / (hi - lo), with beta_j ~ N(0, v s_j^2)
given the residual variance v.  With D = diag(1 / s_j^2) and A = X'X + D the posterior mean is
A^-1 X'y; for a known v the covariance is v A^-1; for v ~ InvGamma(a, b) the posterior of v is
InvGamma(a + n/2, b + (y'y - m'A m) / 2) and the coefficient marginal is Student-t with
covariance (b_n / (a_n - 1)) A^-1.  The window functional of basis function j at a target t is
the Epanechnikov average  int_-1^1 0.75 (1 - u^2) z(t + h u)^j du  by adaptive quadrature.  The
policy value is sum_g pi_g f_g' beta_g with pi_g the observed group fraction.
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


def local_quadratic(dose, y, d0, h, scale):
    d0, h, scale = mp.mpf(d0), mp.mpf(h), mp.mpf(scale)
    G = mp.zeros(3, 3)
    r = mp.zeros(3, 1)
    W = mp.mpf(0)
    rows = []
    for d, yy in zip(dose, y):
        u = mp.mpf(d) - d0
        w = mp.e ** (-(u * u) / (2 * h * h))
        x = [1, u, u * u]
        for a in range(3):
            r[a] += w * x[a] * mp.mpf(yy)
            for c in range(3):
                G[a, c] += w * x[a] * x[c]
        W += w
        rows.append((w, x, mp.mpf(yy)))
    b0 = G ** -1 * r
    rss = sum(w * (yy - sum(x[a] * b0[a] for a in range(3))) ** 2 for w, x, yy in rows)
    v = rss / (W - 3)
    Gp = G.copy()
    Gp[1, 1] += v / scale**2
    Gp[2, 2] += v / scale**2
    b = Gp ** -1 * r
    return {
        "gram": [[s(G[a, c]) for c in range(3)] for a in range(3)],
        "rhs": [s(r[a]) for a in range(3)],
        "weight_sum": s(W),
        "weighted_rss": s(rss),
        "level": s(b[0]),
        "derivative": s(b[1]),
        "unpenalized_level": s(b0[0]),
        "unpenalized_derivative": s(b0[1]),
    }


def window_functional(t, h, lo, hi, width):
    t, h = mp.mpf(t), mp.mpf(h)
    out = []
    for j in range(width):
        f = lambda u: mp.mpf("0.75") * (1 - u * u) * ((2 * (t + h * u) - lo - hi) / (hi - lo)) ** j
        out.append(mp.quad(f, [-1, 0, 1]))
    return out


def group_posterior(dose, y, lo, hi, scales, residual):
    lo, hi = mp.mpf(lo), mp.mpf(hi)
    width = len(scales)
    n = len(dose)
    X = mp.matrix(n, width)
    for i, d in enumerate(dose):
        z = (2 * mp.mpf(d) - lo - hi) / (hi - lo)
        for j in range(width):
            X[i, j] = z**j
    Y = mp.matrix([mp.mpf(v) for v in y])
    A = X.T * X
    for j in range(width):
        A[j, j] += 1 / mp.mpf(scales[j]) ** 2
    Ainv = A ** -1
    m = Ainv * (X.T * Y)
    if residual[0] == "known":
        cov = mp.mpf(residual[1]) * Ainv
    else:
        a, b = mp.mpf(residual[1]), mp.mpf(residual[2])
        an = a + mp.mpf(n) / 2
        bn = b + ((Y.T * Y)[0] - (m.T * A * m)[0]) / 2
        cov = bn / (an - 1) * Ainv
    return m, cov


def policy(groups, lo, hi, scales, residual, h, rules):
    n = sum(len(g["dose"]) for g in groups)
    means = [mp.mpf(0)] * 3
    variances = [mp.mpf(0)] * 3
    for g in groups:
        pi = mp.mpf(len(g["dose"])) / n
        m, cov = group_posterior(g["dose"], g["outcome"], lo, hi, scales, residual)
        fp = window_functional(rules["policy"][g["name"]], h, lo, hi, len(scales))
        fr = window_functional(rules["reference"][g["name"]], h, lo, hi, len(scales))
        fd = [a - b for a, b in zip(fp, fr)]
        for slot, f in enumerate((fp, fr, fd)):
            fv = mp.matrix(f)
            means[slot] += pi * (fv.T * m)[0]
            variances[slot] += pi**2 * (fv.T * cov * fv)[0]
    return [s(x) for x in means], [s(x) for x in variances]


def main():
    rng = Lcg(7701)
    quad_cases = []
    dose = [round(rng.u() * 12, 6) for _ in range(34)]
    y = [1 + 0.3 * d - 0.04 * d * d + 0.5 * rng.normal() for d in dose]
    for d0, h, scale in ((6.0, 2.0, 0.05), (6.0, 2.0, 4.0), (2.5, 1.3, 0.5), (9.5, 3.0, 50.0)):
        case = local_quadratic(dose, y, d0, h, scale)
        case.update({"dose": dose, "outcome": y, "at": d0, "bandwidth": h, "scale": scale})
        quad_cases.append(case)

    groups = []
    for name, size, shift in (("A", 7, 0.0), ("B", 11, 0.6), ("C", 9, -0.3)):
        d = sorted(round(0.3 + rng.u() * 4.4, 6) for _ in range(size))
        yy = [1.0 + shift + 0.8 * x - 0.15 * x * x + 0.35 * rng.normal() for x in d]
        groups.append({"name": name, "dose": d, "outcome": yy})
    rules = {
        "policy": {"A": 1.7, "B": 2.9, "C": 2.2},
        "reference": {"A": 2.4, "B": 1.5, "C": 3.1},
    }
    lo, hi, scales, h = 0.0, 5.0, [3.0, 2.0, 1.0, 0.5], 0.9
    known = policy(groups, lo, hi, scales, ("known", 0.3), h, rules)
    ig = policy(groups, lo, hi, scales, ("ig", 6.0, 1.5), h, rules)
    local_rows = []
    for g in groups:
        for tgt in (rules["policy"][g["name"]], rules["reference"][g["name"]]):
            local_rows.append(sum(1 for d in g["dose"] if abs(d - tgt) < h))
    out = {
        "local_quadratic": quad_cases,
        "policy": {
            "groups": groups, "rules": rules, "support": [lo, hi], "scales": scales,
            "bandwidth": h, "known_variance": 0.3, "inverse_gamma": [6.0, 1.5],
            "minimum_local_rows": min(local_rows),
            "known": {"means": known[0], "variances": known[1]},
            "inverse_gamma_moments": {"means": ig[0], "variances": ig[1]},
        },
    }
    print(json.dumps(out))


main()
