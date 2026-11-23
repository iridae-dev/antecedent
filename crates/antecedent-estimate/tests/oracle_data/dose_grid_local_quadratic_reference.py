# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""High-precision reference for the Gaussian local quadratic dose-grid row.

Regenerate with:  uv run --script dose_grid_local_quadratic_reference.py > dose_grid_local_quadratic_reference.json

For a dose d0 the fit is the weighted least squares of y on (1, u, u^2), u = D - d0, with
Gaussian weights w = exp(-u^2 / (2 h^2)); the level is the intercept and the derivative the
slope.  The linear-smoother weight of row i is l_i = e_k' (X'WX)^-1 x_i w_i, the residual is
e_i = y_i - x_i'beta, and the sandwich standard error is sqrt(sum (l_i e_i)^2).  A contrast
standard error is sqrt(sum (l_to,i e_to,i - l_from,i e_from,i)^2).  The local Kish effective
sample size is (sum w)^2 / sum w^2.  The declared third-derivative bound M gives the bias bound
M/6 sum |l_i| |D_i - d0|^3, and the bias-aware critical value solves
Phi(c - t) - Phi(-c - t) = 0.95 at t = bias / se.  The sup-t critical value of a two-dose grid
is the 0.95 quantile of max(|Z1|, |Z2|) for a bivariate normal whose correlation is the
correlation of the two influence columns, by quadrature.  Everything is evaluated at 50 digits
from the exact binary64 inputs.  Repository-authored (MIT OR Apache-2.0); mpmath is not a
product dependency.
"""
import json
import math

import mpmath as mp

mp.mp.dps = 50


class Lcg:
    def __init__(self, s): self.s = s
    def u(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) % 2**64
        return (self.s >> 11) / float(1 << 53)


def fit(dose, y, d0, h, k):
    n = len(dose)
    d0, h = mp.mpf(d0), mp.mpf(h)
    us = [mp.mpf(d) - d0 for d in dose]
    ws = [mp.e ** (-(u * u) / (2 * h * h)) for u in us]
    A = mp.zeros(3, 3)
    b = mp.zeros(3, 1)
    for u, w, yy in zip(us, ws, y):
        x = [1, u, u * u]
        for a in range(3):
            b[a] += w * x[a] * mp.mpf(yy)
            for c in range(3):
                A[a, c] += w * x[a] * x[c]
    Ainv = A ** -1
    beta = Ainv * b
    ls, es = [], []
    for u, w, yy in zip(us, ws, y):
        x = mp.matrix([1, u, u * u])
        ls.append((Ainv * x)[k] * w)
        es.append(mp.mpf(yy) - (beta[0] + beta[1] * u + beta[2] * u * u))
    return beta, ls, es, ws, us


def folded(t):
    t = mp.mpf(t)
    cov = lambda c: (mp.erf((c - t) / mp.sqrt(2)) - mp.erf((-c - t) / mp.sqrt(2))) / 2
    return mp.findroot(lambda c: cov(c) - mp.mpf("0.95"), (mp.mpf(0), t + 10), solver="anderson", tol=1e-45, maxsteps=200)


def supt2(rho):
    rho = mp.mpf(rho)
    s = mp.sqrt(1 - rho * rho)
    Phi = lambda x: mp.erfc(-x / mp.sqrt(2)) / 2
    phi = lambda x: mp.e ** (-x * x / 2) / mp.sqrt(2 * mp.pi)

    def cover(c):
        return mp.quad(lambda z: phi(z) * (Phi((c - rho * z) / s) - Phi((-c - rho * z) / s)), [-c, 0, c])
    return mp.findroot(lambda c: cover(c) - mp.mpf("0.95"), (mp.mpf("1.9"), mp.mpf("2.6")), solver="anderson", tol=1e-30, maxsteps=100)


def point(dose, y, d0, h, M):
    out = {"dose": d0}
    for name, k in (("level", 0), ("derivative", 1)):
        beta, ls, es, ws, us = fit(dose, y, d0, h, k)
        se = mp.sqrt(sum((l * e) ** 2 for l, e in zip(ls, es)))
        bias = M / mp.mpf(6) * sum(abs(l) * abs(u) ** 3 for l, u in zip(ls, us))
        out[name] = float(beta[k])
        out[name + "_se"] = float(se)
        out[name + "_bias"] = float(bias)
        out[name + "_critical"] = float(folded(bias / se)) if se > 0 else None
        out[name + "_half"] = float(folded(bias / se) * se) if se > 0 else float(bias)
        out[name + "_influence"] = [float(l * e) for l, e in zip(ls, es)]
    ws = [mp.e ** (-((mp.mpf(d) - mp.mpf(d0)) ** 2) / (2 * mp.mpf(h) ** 2)) for d in dose]
    out["ess"] = float(sum(ws) ** 2 / sum(w * w for w in ws))
    out["sum_w"] = float(sum(ws))
    return out


def case(name, dose, y, h, grid, M, contrast):
    pts = [point(dose, y, g, h, M) for g in grid]
    c = None
    if contrast is not None:
        f, t = contrast
        _, lf, ef, _, uf = fit(dose, y, f, h, 0)
        bt, lt, et, _, ut = fit(dose, y, t, h, 0)
        bf = fit(dose, y, f, h, 0)[0]
        se = mp.sqrt(sum((a * b - c_ * d) ** 2 for a, b, c_, d in zip(lt, et, lf, ef)))
        bias = M / mp.mpf(6) * (sum(abs(l) * abs(u) ** 3 for l, u in zip(lt, ut)) + sum(abs(l) * abs(u) ** 3 for l, u in zip(lf, uf)))
        c = {"from": f, "to": t, "estimate": float(bt[0] - bf[0]), "se": float(se), "bias_sum": float(bias),
             "critical": float(folded(bias / se)), "half": float(folded(bias / se) * se)}
    out = {"name": name, "dose": dose, "outcome": y, "bandwidth": h, "grid": grid, "curvature_bound": M, "points": pts, "contrast": c}
    if len(grid) == 2:
        i0, i1 = pts[0]["level_influence"], pts[1]["level_influence"]
        s0 = math.sqrt(sum(v * v for v in i0)); s1 = math.sqrt(sum(v * v for v in i1))
        rho = sum(a * b for a, b in zip(i0, i1)) / (s0 * s1)
        out["band_rho"] = rho
        out["band_critical"] = float(supt2(rho))
    return out


def design(seed, n, lo, hi, offset=0.0, noise=0.3, hetero=True, fn=lambda d: math.sin(d) + 0.2 * d * d):
    r = Lcg(seed)
    dose = sorted(round(lo + (hi - lo) * r.u(), 3) + offset for _ in range(n))
    y = [round(fn(d - offset) + noise * (r.u() - 0.5) * (1 + (abs(d - offset) if hetero else 0)), 5) for d in dose]
    return dose, y


def main():
    cases = []
    d, y = design(1, 60, 0.0, 3.0)
    cases.append(case("smooth_interior", d, y, 0.5, [0.3, 1.2, 2.7], 3.0, (0.5, 2.5)))
    d, y = design(2, 60, 0.0, 3.0)
    cases.append(case("adjacent_grid_pair_high_correlation", d, y, 0.6, [1.4, 1.45], 2.0, None))
    d, y = design(3, 60, 0.0, 3.0)
    cases.append(case("distant_grid_pair_low_correlation", d, y, 0.3, [0.5, 2.5], 2.0, None))
    d, y = design(4, 50, 0.0, 3.0, offset=1.0e8)
    cases.append(case("dose_offset_1e8", d, y, 0.6, [1.0e8 + 0.5, 1.0e8 + 1.5, 1.0e8 + 2.5], 1.0, (1.0e8 + 0.5, 1.0e8 + 2.5)))
    d, y = design(5, 50, 0.0, 3.0, offset=-4.0e6, noise=0.05)
    cases.append(case("negative_offset_small_noise", d, y, 0.4, [-4.0e6 + 0.7, -4.0e6 + 1.9], 0.5, None))
    d, y = design(6, 40, 0.0, 2.0, noise=0.2)
    cases.append(case("wide_bandwidth_near_global_quadratic", d, y, 40.0, [d[3], 1.0, d[-4]], 1.0, (d[3], d[-4])))
    d, y = design(7, 45, 0.0, 6.0, noise=0.2)
    cases.append(case("narrow_bandwidth_weight_sum_near_floor", d, y, 0.4, [0.9, 3.1, 5.2], 0.7, (0.9, 5.2)))
    d, y = design(8, 60, 0.0, 3.0, noise=0.0, hetero=False, fn=lambda x: 1.0 + 2.0 * x - 0.5 * x * x)
    cases.append(case("near_exact_quadratic_tiny_residual", d, y, 0.7, [0.6, 1.5, 2.4], 2.0, (0.6, 2.4)))
    print(json.dumps(cases, indent=1))


if __name__ == "__main__":
    main()
