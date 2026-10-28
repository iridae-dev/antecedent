# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Exact reference for inverting a piecewise-linear interventional CDF at a level tau.

Regenerate with:  uv run --script quantile_inversion_reference.py > quantile_inversion_reference.json

The quantile of a CDF that is linear between grid thresholds is inf{c : F(c) >= tau}.  This
script finds it by scanning the segments of the exact rational CDF and solving the linear
equation on the first segment that reaches tau; it does not use the product's bracket
position, weight clamp or influence formula.  The influence function of the quantile is the
exact derivative of that whole map, obtained with dual numbers over Fractions: for a CDF
perturbed as F_j + eps * phi_j[i] the dual part of the quantile is the influence of row i.
For score tables the weighted means are formed from the weights, the CDF is 1 - mean, and the
influence of row i is n * w_i * dQ/dw_i (the sandwich normalisation of a ratio of weighted
sums), again by dual numbers through the unnormalised weights.  Every number is evaluated
from the exact binary64 inputs.  Repository-authored (MIT OR Apache-2.0); no product
dependency.
"""
import json
from fractions import Fraction as F


class Dual:
    __slots__ = ("a", "b")

    def __init__(self, a, b=F(0)):
        self.a, self.b = F(a), F(b)

    def _c(o):
        return o if isinstance(o, Dual) else Dual(o)

    def __add__(s, o):
        o = Dual._c(o); return Dual(s.a + o.a, s.b + o.b)
    __radd__ = __add__

    def __sub__(s, o):
        o = Dual._c(o); return Dual(s.a - o.a, s.b - o.b)

    def __rsub__(s, o):
        return Dual._c(o) - s

    def __mul__(s, o):
        o = Dual._c(o); return Dual(s.a * o.a, s.a * o.b + s.b * o.a)
    __rmul__ = __mul__

    def __truediv__(s, o):
        o = Dual._c(o)
        return Dual(s.a / o.a, (s.b * o.a - s.a * o.b) / (o.a * o.a))


def quantile(x, cdf, tau):
    """First segment [x_k, x_k+1] on which the linear CDF reaches tau (cdf[k] < tau <= cdf[k+1])."""
    for k in range(len(x) - 1):
        if cdf[k].a < tau <= cdf[k + 1].a:
            return x[k] + (Dual(tau) - cdf[k]) / (cdf[k + 1] - cdf[k]) * (x[k + 1] - x[k])
    return None


def fl(v):
    return float(v)


def direct_case(name, x, f, tau, phi, supported=None):
    xs = [F(v) for v in x]
    span = xs[-1] - xs[0]
    # accept/refuse by the stated density rule on the bracket
    probe = quantile(xs, [Dual(F(v)) for v in f], F(tau))
    out = {"name": name, "x": x, "cdf": f, "tau": tau, "phi": phi, "supported": supported}
    if probe is None:
        out["expect"] = "range"
        return out
    k = next(k for k in range(len(xs) - 1) if F(f[k]) < F(tau) <= F(f[k + 1]))
    density_norm = (F(f[k + 1]) - F(f[k])) / (xs[k + 1] - xs[k]) * span
    out["bracket"] = k
    out["normalised_density"] = fl(density_norm)
    if density_norm < F(1, 1000):
        out["expect"] = "density"
        return out
    n = len(phi[0])
    infl = []
    for i in range(n):
        cdf = [Dual(F(f[j]), F(phi[j][i])) for j in range(len(f))]
        q = quantile(xs, cdf, F(tau))
        infl.append(fl(q.b))
        qv = q.a
    out["expect"] = "ok"
    out["value"] = fl(qv)
    out["influence"] = infl
    out["density"] = fl((F(f[k + 1]) - F(f[k])) / (xs[k + 1] - xs[k]))
    return out


class Lcg:
    def __init__(self, s): self.s = s
    def u(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) % 2**64
        return (self.s >> 11) / float(1 << 53)


def phis(rng, m, n, scale=0.05):
    return [[round((rng.u() - 0.5) * 2 * scale, 6) for _ in range(n)] for _ in range(m)]


def direct_cases():
    r = Lcg(7)
    cases = []
    base_f = [0.05, 0.2, 0.45, 0.7, 0.93]
    for s in (1e-6, 1.0, 1e3, 1e6):
        x = [k * s for k in (0.0, 1.0, 2.5, 3.0, 7.0)]
        cases.append(direct_case(f"interior_scale_{s:g}", x, base_f, 0.3, phis(r, 5, 6)))
    x = [0.0, 1.0, 2.5, 3.0, 7.0]
    cases.append(direct_case("tau_equals_grid_cdf_value", x, base_f, 0.45, phis(r, 5, 6)))
    cases.append(direct_case("tau_just_above_grid_cdf_value", x, base_f, 0.45 + 2**-40, phis(r, 5, 6)))
    cases.append(direct_case("plateau_below_tau", x, [0.1, 0.4, 0.4, 0.4, 0.9], 0.6, phis(r, 5, 6)))
    cases.append(direct_case("plateau_at_tau_picks_first_reach", x, [0.1, 0.4, 0.4, 0.4, 0.9], 0.4, phis(r, 5, 6)))
    cases.append(direct_case("last_segment", x, base_f, 0.9, phis(r, 5, 6)))
    cases.append(direct_case("first_segment", x, base_f, 0.1, phis(r, 5, 6)))
    cases.append(direct_case("above_range", x, base_f, 0.95, phis(r, 5, 6)))
    cases.append(direct_case("below_range", x, base_f, 0.04, phis(r, 5, 6)))
    # the density rule: rise/width times span against 1e-3
    for s in (1e-6, 1.0, 1e6):
        xx = [k * s for k in (0.0, 1.0, 2.0, 3.0)]
        cases.append(direct_case(f"flat_bracket_refused_{s:g}", xx, [0.1, 0.5, 0.5002, 0.9], 0.50015, phis(r, 4, 3)))
        # rise 0.003 across a third of the span -> normalised density 9e-3 (accepted)
        cases.append(direct_case(f"thin_bracket_accepted_{s:g}", xx, [0.1, 0.5, 0.503, 0.9], 0.5015, phis(r, 4, 3)))
        # rise 2.4e-4 -> normalised density 7.2e-4 (refused)
        cases.append(direct_case(f"thin_bracket_refused_{s:g}", xx, [0.1, 0.5, 0.50024, 0.9], 0.50012, phis(r, 4, 3)))
    # steep bracket: near-jump CDF, tiny width
    cases.append(direct_case("near_jump", [0.0, 1.0, 1.0 + 2**-30, 2.0], [0.0 + 0.02, 0.3, 0.7, 0.98], 0.5, phis(r, 4, 4)))
    return cases


def table_case(name, seed, n, weight_kind, scale, tau):
    r = Lcg(seed)
    thresholds = [-0.5, 0.0, 0.7, 1.4]
    y = [round((r.u() + r.u() + r.u() - 1.5) * 2.0, 6) for _ in range(n)]
    w = [1.0] * n if weight_kind == "none" else [float(1 + (int(r.u() * 7))) * scale for _ in range(n)]
    cols = []
    for c in thresholds:
        col = [(1.0 if y[i] > c else 0.0) + round((r.u() - 0.5) * 0.2, 4) for i in range(n)]
        cols.append(col)
    # weighted means and the exact quantile with row-w_i dual perturbations
    wf = [F(v) for v in w]
    xs = [F(c) for c in thresholds]

    def q_with(wd):
        tot = sum(wd, Dual(0))
        cdf = []
        for col in cols:
            m = sum((wd[i] * F(col[i]) for i in range(n)), Dual(0)) / tot
            cdf.append(1 - m)
        return cdf, quantile(xs, cdf, F(tau))

    cdf0, q0 = q_with([Dual(v) for v in wf])
    assert q0 is not None, name
    infl = []
    for i in range(n):
        wd = [Dual(wf[j], wf[j] if j == i else 0) for j in range(n)]
        _, q = q_with(wd)
        infl.append(fl(n * q.b))
    return {
        "name": name, "thresholds": thresholds, "outcome": y, "columns": cols,
        "weights": None if weight_kind == "none" else w, "tau": tau,
        "cdf": [fl(c.a) for c in cdf0], "value": fl(q0.a), "influence": infl,
    }


def main():
    cases = {"direct": direct_cases(), "table": []}
    tau = 0.5
    cases["table"].append(table_case("unweighted", 11, 60, "none", 1.0, tau))
    for s in (1.0, 1e-200, 1e200):
        cases["table"].append(table_case(f"weighted_scale_{s:g}", 12, 60, "int", s, tau))
    print(json.dumps(cases, indent=1))


if __name__ == "__main__":
    main()
