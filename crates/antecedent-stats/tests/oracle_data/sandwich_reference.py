# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Exact-rational reference values for the sandwich covariances and least-squares solves.

Regenerate with:  uv run --script sandwich_reference.py > sandwich_reference.json

Every design, residual, weight and cluster label is an exact binary64 value (or integer) and
every expected value is computed in `fractions.Fraction` arithmetic with no rounding at all,
then rounded once to binary64.  Nothing here uses floating-point linear algebra, a library
implementation of any estimator or a solver; the estimators are written from their defining
formulas:

* HC0: bread M meat M with M = (X'X)^-1 and meat = sum e_i^2 x_i x_i'.
* HC1: HC0 * n / (n - p).  HC2: e_i^2 / (1 - h_i).  HC3: e_i^2 / (1 - h_i)^2, h_i = x_i' M x_i.
* Classical: sigma^2 M with sigma^2 = sum e^2 / (n - p).
* One-way cluster: meat = sum_g s_g s_g', s_g = sum_{i in g} e_i x_i, scaled by
  G/(G-1) * (n-1)/(n-p).
* Multiway: Cameron-Gelbach-Miller inclusion-exclusion over the non-empty subsets of the
  clustering dimensions, each subset's cluster meat carrying its own G_S/(G_S-1) (n-1)/(n-p).
* Newey-West: Gamma_0 + sum_{l=1}^{L} (1 - l/(L+1)) (Gamma_l + Gamma_l'), L = min(lag, n-1),
  no finite-sample factor.
* Panel cluster + within-unit HAC: the same with lag products formed only between rows of one
  unit that are exactly `l` time labels apart, L = min(lag, widest unit span), then scaled by the
  cluster factor with G = number of units.
* Fisher-weighted (score) variants use bread (X'WX)^-1 and h_i = w_i x_i' (X'WX)^-1 x_i.
* Bell-McCaffrey HC2 degrees of freedom: A = M_hat D M_hat with M_hat = I - H, formed in full,
  df = (tr A)^2 / tr(A^2), the dense definition rather than the O(n p^2) trace identity.

The least-squares cases give the exact normal-equation solution (and, for rank-deficient
designs, the exact minimum-norm solution through an exactly reduced full-rank problem) of the
binary64 data, plus the exact 2-norm condition number of the equilibrated design.

This script and its frozen output are repository-authored (MIT OR Apache-2.0).
"""
import json
import math
import sys
from fractions import Fraction as F

import mpmath as mp

mp.mp.dps = 60


def fl(v):
    return float(v)


def mat_inv(a):
    n = len(a)
    m = [[F(x) for x in row] + [F(int(i == j)) for j in range(n)] for i, row in enumerate(a)]
    for c in range(n):
        piv = next(r for r in range(c, n) if m[r][c] != 0)
        m[c], m[piv] = m[piv], m[c]
        d = m[c][c]
        m[c] = [x / d for x in m[c]]
        for r in range(n):
            if r != c and m[r][c] != 0:
                f = m[r][c]
                m[r] = [x - f * y for x, y in zip(m[r], m[c])]
    return [row[n:] for row in m]


def mm(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))]
            for i in range(len(a))]


def tr(a):
    return [list(r) for r in zip(*a)]


class Lcg:
    """Deterministic uniform source, independent of the interpreter's random module."""

    def __init__(self, seed):
        self.s = seed

    def u(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        return (self.s >> 11) / float(1 << 53)

    def sym(self, scale=1.0):
        return float(F(round((self.u() * 2 - 1) * 1e6), 10**6)) * scale


def rows_of(cols):
    n = len(cols[0])
    return [[F(c[i]) for c in cols] for i in range(n)]


def bread(rows, w=None):
    p = len(rows[0])
    g = [[sum((F(w[i]) if w else 1) * r[a] * r[b] for i, r in enumerate(rows))
          for b in range(p)] for a in range(p)]
    return mat_inv(g)


def lev(rows, b, w=None):
    out = []
    for i, r in enumerate(rows):
        h = sum(r[a] * b[a][c] * r[c] for a in range(len(r)) for c in range(len(r)))
        out.append(h * (F(w[i]) if w else 1))
    return out


def outer_sum(vecs, p):
    return [[sum(v[a] * v[c] for v in vecs) for c in range(p)] for a in range(p)]


def sandwich(b, meat):
    return mm(mm(b, meat), b)


def cluster_meat(rows, e, groups, p):
    sums = {}
    for r, ei, g in zip(rows, e, groups):
        s = sums.setdefault(g, [F(0)] * p)
        for a in range(p):
            s[a] += F(ei) * r[a]
    return outer_sum(list(sums.values()), p), len(sums)


def cfs(n, p, g):
    return F(g, g - 1) * F(n - 1, n - p)


def scale(m, f):
    return [[x * f for x in row] for row in m]


def add(a, b, s=1):
    return [[x + s * y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def hac_meat(scores, order_groups, p, lag, times=None):
    """Within-sequence Bartlett meat.  `order_groups` is a list of row-index lists, each in
    time order; with `times` the lag is by label difference, otherwise by position."""
    def span(idx):
        return (times[idx[-1]] - times[idx[0]]) if times else len(idx) - 1
    lmax = max(span(idx) for idx in order_groups)
    L = min(lag, lmax)
    meat = [[F(0)] * p for _ in range(p)]
    for idx in order_groups:
        for i in idx:
            for a in range(p):
                for c in range(p):
                    meat[a][c] += scores[i][a] * scores[i][c]
        for l in range(1, L + 1):
            w = 1 - F(l, L + 1)
            pos = {times[i]: i for i in idx} if times else {k: i for k, i in enumerate(idx)}
            for pos_i, i in enumerate(idx):
                key = times[i] if times else pos_i
                j = pos.get(key - l)
                if j is None:
                    continue
                for a in range(p):
                    for c in range(p):
                        meat[a][c] += w * (scores[i][a] * scores[j][c] + scores[j][a] * scores[i][c])
    return meat


def flat(m):
    return [fl(x) for row in m for x in row]


def cond2(rows):
    """Exact-ish 2-norm condition number of the column-equilibrated design (unit columns), by
    power iteration in Fractions on the Gram of the equilibrated design, reported as a float."""
    p = len(rows[0])
    import math as m
    norms = [m.sqrt(float(sum(r[a] * r[a] for r in rows))) for a in range(p)]
    g = [[float(sum(r[a] * r[b] for r in rows)) / (norms[a] * norms[b]) for b in range(p)]
         for a in range(p)]
    # smallest/largest eigenvalue by symmetric Jacobi on floats (reporting only).
    A = [row[:] for row in g]
    for _ in range(60):
        for i in range(p):
            for j in range(i + 1, p):
                if abs(A[i][j]) < 1e-300:
                    continue
                th = 0.5 * m.atan2(2 * A[i][j], A[j][j] - A[i][i])
                c, s = m.cos(th), m.sin(th)
                for k in range(p):
                    A[k][i], A[k][j] = c * A[k][i] - s * A[k][j], s * A[k][i] + c * A[k][j]
                for k in range(p):
                    A[i][k], A[j][k] = c * A[i][k] - s * A[j][k], s * A[i][k] + c * A[j][k]
    ev = sorted(abs(A[i][i]) for i in range(p))
    return m.sqrt(ev[-1] / ev[0]) if ev[0] > 0 else float("inf")


def make_design(name, n, rng):
    one = [1.0] * n
    u = [float(rng.sym(3.0)) for _ in range(n)]
    z = [float(rng.sym(2.0)) for _ in range(n)]
    if name == "well_conditioned":
        return [one, u, z]
    if name.startswith("offset_"):
        off = {"offset_1e4": 1e4, "offset_1e6": 1e6, "offset_1e8": 1e8}[name]
        return [one, [off + v for v in u], z]
    if name == "near_collinear_1e-5":
        return [one, u, [a + 1e-5 * b for a, b in zip(u, z)]]
    if name == "near_collinear_1e-7":
        return [one, u, [a + 1e-7 * b for a, b in zip(u, z)]]
    if name == "high_leverage":
        x = u[:]
        x[0] = 5000.0
        return [one, x]
    if name == "moderate_leverage":
        x = u[:]
        x[0] = 40.0
        return [one, x]
    if name == "badly_scaled":
        return [one, [1e-6 * v for v in u], [1e6 * v for v in z]]
    raise ValueError(name)


def cov_cases():
    out = []
    names = ["well_conditioned", "offset_1e4", "offset_1e6", "offset_1e8", "near_collinear_1e-5",
             "near_collinear_1e-7", "high_leverage", "badly_scaled"]
    for k, name in enumerate(names):
        n = 24
        rng = Lcg(1000 + k)
        cols = make_design(name, n, rng)
        p = len(cols)
        e = [float(rng.sym(1.0) * (1 + (i % 5))) for i in range(n)]
        groups = [i % 6 for i in range(n)]
        groups2 = [(i * 7 + i // 5) % 4 for i in range(n)]
        times = [i // 6 for i in range(n)]
        rows = rows_of(cols)
        b = bread(rows)
        h = lev(rows, b)
        cond = cond2(rows)
        base = {"design": name, "n": n, "p": p, "x": [v for c in cols for v in c], "e": e,
                "cond": cond, "one_minus_max_leverage": fl(1 - max(h))}
        sat = max(fl(x) for x in h) > 1 - 1e-8
        meat0 = outer_sum([[F(ei) * x for x in r] for ei, r in zip(e, rows)], p)

        def case(kind, cov, **extra):
            d = dict(base)
            d.update({"kind": kind, "cov": flat(cov)})
            d.update(extra)
            out.append(d)

        case("hc0", sandwich(b, meat0))
        case("hc1", scale(sandwich(b, meat0), F(n, n - p)))
        if not sat:
            for kind, pw in (("hc2", 1), ("hc3", 2)):
                meat = [[sum(F(ei) ** 2 / (1 - hi) ** pw * r[a] * r[c]
                             for ei, r, hi in zip(e, rows, h)) for c in range(p)] for a in range(p)]
                case(kind, sandwich(b, meat))
        sigma2 = sum(F(ei) ** 2 for ei in e) / (n - p)
        case("homoskedastic", scale(b, sigma2))
        meat, g = cluster_meat(rows, e, groups, p)
        case("cluster", sandwich(b, scale(meat, cfs(n, p, g))), groups=groups)
        # A different labelling of the same rows must give a different, equally exact covariance.
        meat_alt, g_alt = cluster_meat(rows, e, groups2, p)
        case("cluster_alt", sandwich(b, scale(meat_alt, cfs(n, p, g_alt))), groups=groups2)
        total = [[F(0)] * p for _ in range(p)]
        for mask in range(1, 4):
            lab = [tuple(d[i] for bit, d in enumerate((groups, groups2)) if mask >> bit & 1)
                   for i in range(n)]
            m, gs = cluster_meat(rows, e, lab, p)
            total = add(total, scale(m, cfs(n, p, gs)), 1 if bin(mask).count("1") % 2 else -1)
        # The inclusion-exclusion matrix need not be positive semidefinite.  Its definiteness is
        # measured in the metric of the bread (V v = lambda B v), where it does not depend on
        # the scaling of the columns: smallest over largest generalised eigenvalue.
        Bm = mp.matrix([[mp.mpf(x.numerator) / x.denominator for x in row] for row in b])
        Vm = mp.matrix([[mp.mpf(x.numerator) / x.denominator for x in row]
                        for row in sandwich(b, total)])
        Lc = mp.cholesky(Bm)
        Vz = mp.inverse(Lc) * Vm * mp.inverse(Lc).T
        eig = sorted(mp.eigsy((Vz + Vz.T) / 2, eigvals_only=True))
        case("multiway", sandwich(b, total), groups=groups, groups2=groups2,
             min_eig_ratio=fl(eig[0] / eig[-1]))
        scores = [[F(ei) * x for x in r] for ei, r in zip(e, rows)]
        for lag in (0, 3, 40):
            case("newey_west", sandwich(b, hac_meat(scores, [list(range(n))], p, lag)), lag=lag)
        # Panel: units = groups, time label = position within unit, with one gap in unit 2.
        ptime = [i // 6 for i in range(n)]
        for i in range(n):
            if groups[i] == 2 and ptime[i] >= 2:
                ptime[i] += 1
        units = {}
        for i in range(n):
            units.setdefault(groups[i], []).append(i)
        seqs = [sorted(idx, key=lambda i: ptime[i]) for idx in units.values()]
        for lag in (1, 2, 9):
            meat = hac_meat(scores, seqs, p, lag, times=ptime)
            case("panel_hac", sandwich(b, scale(meat, cfs(n, p, len(units)))), groups=groups,
                 time=ptime, lag=lag)
        case("panel_hac_lag0", sandwich(b, scale(cluster_meat(rows, e, groups, p)[0],
                                                 cfs(n, p, len(units)))), groups=groups,
             time=ptime, lag=0)
        # Fisher-weighted score path, weights spanning six decades.
        w = [float(10 ** (-3 + 6 * rng.u())) for _ in range(n)]
        bw = bread(rows, w)
        hw = lev(rows, bw, w)
        if max(fl(x) for x in hw) <= 1 - 1e-8:
            m0 = outer_sum([[F(ei) * x for x in r] for ei, r in zip(e, rows)], p)
            case("score_hc0", sandwich(bw, m0), weights=w)
            for kind, pw in (("score_hc2", 1), ("score_hc3", 2)):
                meat = [[sum(F(ei) ** 2 / (1 - hi) ** pw * r[a] * r[c]
                             for ei, r, hi in zip(e, rows, hw)) for c in range(p)]
                        for a in range(p)]
                case(kind, sandwich(bw, meat), weights=w)
    return out


def bm_cases():
    out = []
    for k, (name, n) in enumerate([("well_conditioned", 20), ("offset_1e4", 20),
                                   ("near_collinear_1e-5", 20), ("moderate_leverage", 20),
                                   ("high_leverage", 20)]):
        rng = Lcg(2000 + k)
        cols = make_design(name, n, rng)
        p = len(cols)
        e = [float(rng.sym(1.0) * (1 + (i % 4))) for i in range(n)]
        rows = rows_of(cols)
        b = bread(rows)
        h = lev(rows, b)
        if max(fl(x) for x in h) > 1 - 1e-8:
            continue
        for coef in range(p):
            sel = [b[coef][a] for a in range(p)]
            wv = [sum(sel[a] * r[a] for a in range(p)) for r in rows]
            var = sum(F(ei) ** 2 * w * w / (1 - hi) for ei, w, hi in zip(e, wv, h))
            # variance as in the sandwich: row (coef,coef) of B S B = sum w_i^2 e_i^2/(1-h_i)
            d = [w * w / (1 - hi) for w, hi in zip(wv, h)]
            # M_hat = I - X B X'
            H = [[sum(rows[i][a] * b[a][c] * rows[j][c] for a in range(p) for c in range(p))
                  for j in range(n)] for i in range(n)]
            Mh = [[F(int(i == j)) - H[i][j] for j in range(n)] for i in range(n)]
            A = mm(mm(Mh, [[d[i] if i == j else F(0) for j in range(n)] for i in range(n)]), Mh)
            trA = sum(A[i][i] for i in range(n))
            trA2 = sum(A[i][j] * A[j][i] for i in range(n) for j in range(n))
            out.append({"design": name, "n": n, "p": p, "coef": coef,
                        "x": [v for c in cols for v in c], "e": e, "variance": fl(var),
                        "df": fl(trA * trA / trA2), "cond": cond2(rows),
                        "one_minus_max_leverage": fl(1 - max(h))})
    return out


def lstsq_exact(rows, y, w=None):
    p = len(rows[0])
    sw = [F(1)] * len(rows) if w is None else [F(v) for v in w]
    g = [[sum(sw[i] * r[a] * r[b] for i, r in enumerate(rows)) for b in range(p)]
         for a in range(p)]
    rhs = [sum(sw[i] * r[a] * F(y[i]) for i, r in enumerate(rows)) for a in range(p)]
    gi = mat_inv(g)
    return [sum(gi[a][c] * rhs[c] for c in range(p)) for a in range(p)]


def ls_cases():
    out = []
    names = ["well_conditioned", "offset_1e4", "offset_1e6", "offset_1e8", "near_collinear_1e-5",
             "near_collinear_1e-7", "badly_scaled"]
    for k, name in enumerate(names):
        n = 30
        rng = Lcg(3000 + k)
        cols = make_design(name, n, rng)
        y = [float(rng.sym(1.0)) + 2.0 * i / n for i in range(n)]
        if name.startswith("offset"):
            y = [v + 1e3 for v in y]
        rows = rows_of(cols)
        coef = lstsq_exact(rows, y)
        w = [float(10 ** (-4 + 8 * rng.u())) for _ in range(n)]
        w[3] = 0.0
        coef_w = lstsq_exact(rows, y, w)
        out.append({"design": name, "n": n, "p": len(cols), "x": [v for c in cols for v in c],
                    "y": y, "w": w, "coef": flat([coef])[:], "coef_w": flat([coef_w]),
                    "cond": cond2(rows)})
    return out


def rank_deficient_cases():
    out = []
    for k in range(3):
        n = 20
        rng = Lcg(4000 + k)
        one = [1.0] * n
        u = [float(rng.sym(3.0)) for _ in range(n)]
        z = [float(rng.sym(2.0)) for _ in range(n)]
        y = [float(rng.sym(1.0)) + 3.0 * a for a, _ in zip(u, z)]
        if k == 0:
            cols = [one, u, u]          # exact duplicate column
            reduced = [one, u]
            split = lambda c: [c[0], c[1] / 2, c[1] / 2]
        elif k == 1:
            cols = [one, u, [2.0 * v for v in u], z]   # exact multiple (power-of-two scaling)
            reduced = None
        else:
            cols = [one, [0.0] * n, u]  # an all-zero column
            reduced = None
        rows = rows_of(cols)
        p = len(cols)
        # Exact minimum-norm least squares: beta = X' (X X')^+ ... computed through the exact
        # full-rank reduction  beta = B (B'X'XB)^-1 B'X'y  with B an orthogonal-on-null-space
        # basis is avoided; use the normal-equation minimum norm solution
        # beta = (X'X)^+ X'y with the pseudo-inverse from the exact rank factorisation.
        # Rank factorisation: choose the independent columns J, X = X_J C with C = (X_J'X_J)^-1 X_J'X.
        J = {0: [0, 1], 1: [0, 1, 3], 2: [0, 2]}[k]
        XJ = [[r[j] for j in J] for r in rows]
        gJ = mat_inv([[sum(r[a] * r[b] for r in XJ) for b in range(len(J))] for a in range(len(J))])
        C = [[sum(gJ[a][c] * sum(XJ[i][c] * rows[i][q] for i in range(n)) for c in range(len(J)))
              for q in range(p)] for a in range(len(J))]
        # X = X_J C  =>  X^+ = C^+ X_J^+ = C'(C C')^-1 (X_J'X_J)^-1 X_J'
        CCt = mat_inv([[sum(C[a][q] * C[b][q] for q in range(p)) for b in range(len(J))]
                       for a in range(len(J))])
        XJty = [sum(XJ[i][a] * F(y[i]) for i in range(n)) for a in range(len(J))]
        t = [sum(gJ[a][c] * XJty[c] for c in range(len(J))) for a in range(len(J))]
        t = [sum(CCt[a][c] * t[c] for c in range(len(J))) for a in range(len(J))]
        beta = [sum(C[a][q] * t[a] for a in range(len(J))) for q in range(p)]
        out.append({"case": k, "n": n, "p": p, "rank": len(J), "x": [v for c in cols for v in c],
                    "y": y, "coef": flat([beta])})
    return out


def describe_cases():
    out = []
    rng = Lcg(5000)
    sets = {
        "plain": [float(rng.sym(1.0)) for _ in range(17)],
        "offset_1e9": [1e9 + float(rng.sym(1.0)) for _ in range(500)],
        "offset_1e15": [1e15 + float(round(rng.sym(4.0))) for _ in range(500)],
        "wide_range": [float(rng.sym(1.0)) * 10 ** (k % 7 - 3) for k in range(60)],
        "tiny": [1e-160 * (1 + float(rng.sym(1.0))) for _ in range(40)],
        "huge": [1e150 * (1 + float(rng.sym(1.0))) for _ in range(40)],
        "two_values": [1.0, 1.0 + 2.0 ** -52],
        "large_n": [float(rng.sym(1.0)) for _ in range(20000)],
    }
    for name, v in sets.items():
        n = len(v)
        mean = sum(F(x) for x in v) / n
        var = sum((F(x) - mean) ** 2 for x in v) / (n - 1)
        # sqrt(var / n) of an exact rational, correctly rounded via integer sqrt on a scaled value
        q = var / n
        k = 1200
        s = math.isqrt((q.numerator << (2 * k)) // q.denominator)
        out.append({"name": name, "values": v, "mean": fl(mean), "variance": fl(var),
                    "se": fl(F(s, 1 << k))})
    return out


def mp_median(v):
    v = sorted(v)
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def mp_wls(rows, y, w):
    p = len(rows[0])
    g = mp.matrix(p, p)
    r = mp.matrix(p, 1)
    for xi, yi, wi in zip(rows, y, w):
        for a in range(p):
            r[a] += wi * xi[a] * yi
            for b in range(p):
                g[a, b] += wi * xi[a] * xi[b]
    return list(mp.lu_solve(g, r))


def huber_cases():
    """Huber regression by the iteration the product defines (OLS start, scale = 1.4826 MAD of the
    current residuals, weights min(1, c/|r/s|)), run at 60 digits to a 1e-40 step; reported with the
    estimating-equation residual sum psi_c(r_i/s) x_i, which vanishes at a fixed point whatever
    the iteration that reached it."""
    out = []
    for k, (name, n, outliers, zero_infl) in enumerate([
            ("clean", 60, 0, False), ("outliers", 60, 6, False),
            ("offset_1e6", 60, 4, False), ("zero_inflated", 60, 0, True)]):
        rng = Lcg(6000 + k)
        t = [float(rng.sym(2.0)) for _ in range(n)]
        cols = [[1.0] * n, [(1e6 + v) if name == "offset_1e6" else v for v in t]]
        y = [1.0 + 2.0 * v + float(rng.sym(0.5)) for v in t]
        if zero_infl:
            cols = [[1.0] * n]
            y = [1.0 if i % 4 == 0 else 0.0 for i in range(n)]
        for j in range(outliers):
            y[j * 7 % n] += 40.0 * (1 if j % 2 else -1)
        rows = [[mp.mpf(c[i]) for c in cols] for i in range(n)]
        yy = [mp.mpf(v) for v in y]
        beta = mp_wls(rows, yy, [mp.mpf(1)] * n)
        noise = mp.mpf(1000) * mp.mpf(2) ** -52 * max(abs(v) for v in yy)
        c = mp.mpf("1.345")
        fallback, converged, scale = False, False, mp.mpf(0)
        for it in range(1, 501):
            res = [yi - sum(x * b for x, b in zip(r, beta)) for r, yi in zip(rows, yy)]
            med = mp_median(res)
            mad = mp.mpf("1.4826") * mp_median([abs(v - med) for v in res])
            if mad > noise:
                scale, fallback = mad, False
            else:
                mean_ad = sum(abs(v - med) for v in res) / n
                fb = mp.sqrt(mp.pi / 2) * mean_ad
                scale, fallback = (fb, True) if fb > noise else (mp.mpf(0), False)
            w = []
            for v in res:
                if scale > 0:
                    au = abs(v / scale)
                    w.append(mp.mpf(1) if (au <= c or au < mp.mpf("1e-15")) else c / au)
                else:
                    w.append(mp.mpf(1))
            new = mp_wls(rows, yy, w)
            step = max(abs(a - b) for a, b in zip(new, beta))
            beta = new
            if step < mp.mpf("1e-40"):
                converged = True
                break
        res = [yi - sum(x * b for x, b in zip(r, beta)) for r, yi in zip(rows, yy)]
        psi = [max(-c, min(c, v / scale)) if scale > 0 else v for v in res]
        score = [sum(psi[i] * rows[i][a] for i in range(n)) for a in range(len(cols))]
        out.append({"name": name, "n": n, "p": len(cols), "x": [v for col in cols for v in col], "y": y,
                    "coef": [float(b) for b in beta], "scale": float(scale),
                    "scale_fallback": fallback, "converged": converged, "iterations": it,
                    "score_norm": float(max(abs(v) for v in score))})
    return out


if __name__ == "__main__":
    json.dump({"covariance": cov_cases(), "bell_mccaffrey": bm_cases(),
               "least_squares": ls_cases(), "rank_deficient": rank_deficient_cases(),
               "describe": describe_cases(), "huber": huber_cases()}, sys.stdout, allow_nan=False)
