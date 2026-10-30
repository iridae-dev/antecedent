"""Kennedy-style cross-fitted pseudo-outcome and local-quadratic curve, exactly.

Regenerate with:  python3 estimate_response_kennedy_dr_reference.py > estimate_response_kennedy_dr_reference.json

The data are dyadic rationals, so the product reads the numbers written here.  Everything is exact
(Fractions) until a Gaussian density or kernel needs an exponential, where mpmath at 50 digits is
used.  The pipeline follows the estimator's definition:

* folds: row i belongs to fold i mod K; the nuisances of a row's fold are fitted on the other rows;
* outcome nuisance: the penalized additive cubic-spline fit of y on (a, x) over the training rows,
  with quantile (type 7) knots, second-difference penalty, each smooth centered over the training
  rows; its fitted function F is unique, the treatment smooth is g_a(a) = f_a(a) - mean_train f_a;
* treatment nuisance: the same fit of a on x; its residual scale is
  sigma = sqrt(RSS / max(n_train - edf, 1)) with edf the hat-matrix trace (intercept included);
* conditional density pi_i = N(a_i; m(x_i), sigma^2); marginal density
  m_i = mean over the training rows j of N(a_i; m(x_j), sigma^2);
* marginalized outcome: offset = mean over all rows (fold fit) of F(a_j, x_j) - g_a(a_j), and
  M(a) = offset + g_a(a);
* pseudo-outcome phi_i = M(a_i) + (y_i - F(a_i, x_i)) sigma*m_i / max(sigma*pi_i, 1e-8);
  covariate term c_i = F(a_i, x_i) - M(a_i).

The curve at a grid point is the intercept of the Gaussian-kernel weighted local quadratic of phi on
(1, a - at, (a - at)^2) with the given bandwidth.  Repository-authored (MIT OR Apache-2.0); needs
mpmath (BSD-3-Clause) at run time, not redistributed.
"""
import json
from fractions import Fraction as F

from mpmath import mp, mpf, exp, pi, sqrt as msqrt, matrix, lu_solve

mp.dps = 50
_state = 0x510E527FADE682D1


def u01():
    global _state
    _state = (_state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
    return (_state >> 11) / float(1 << 53)


def dy(x, den):
    return F(int(round(x * den)), den)


def solve(A, b):
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        piv = next(r for r in range(c, n) if M[r][c] != 0)
        M[c], M[piv] = M[piv], M[c]
        d = M[c][c]
        M[c] = [x / d for x in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [M[i][n] for i in range(n)]


def inverse(A):
    n = len(A)
    cols = [solve(A, [F(int(i == j)) for i in range(n)]) for j in range(n)]
    return [[cols[j][i] for j in range(n)] for i in range(n)]


def quantile7(xs, p):
    s = sorted(xs)
    h = (len(s) - 1) * p
    lo = h.numerator // h.denominator
    hi = min(-(-h.numerator // h.denominator), len(s) - 1)
    return s[lo] if lo == hi else s[lo] + (h - lo) * (s[hi] - s[lo])


def quantile_knots(xs, K):
    lo, hi = min(xs), max(xs)
    if lo == hi:
        lo, hi = lo - 1, hi + 1
    interior = K - 4
    return [lo] * 4 + [quantile7(xs, F(i, interior + 1)) for i in range(1, interior + 1)] + [hi] * 4


def basis_row(x, knots, K):
    left, right = knots[3], knots[len(knots) - 4]
    x = min(max(x, left), right)
    last = max(i for i in range(len(knots) - 1) if knots[i] < knots[i + 1])
    N = []
    for i in range(len(knots) - 1):
        inside = knots[i] <= x < knots[i + 1] or (x == right and i == last)
        N.append(F(1) if inside else F(0))
    for d in range(1, 4):
        M = []
        for i in range(len(knots) - 1 - d):
            a = F(0) if knots[i + d] == knots[i] else (x - knots[i]) / (knots[i + d] - knots[i]) * N[i]
            b = F(0) if knots[i + d + 1] == knots[i + 1] else (knots[i + d + 1] - x) / (knots[i + d + 1] - knots[i + 1]) * N[i + 1]
            M.append(a + b)
        N = M
    return N[:K]


def penalty(K):
    P = [[F(0)] * K for _ in range(K)]
    for r in range(K - 2):
        d = {r: 1, r + 1: -2, r + 2: 1}
        for i, a in d.items():
            for j, b in d.items():
                P[i][j] += a * b
    return P


class Additive:
    """Penalized additive fit on columns `xs` (list of columns), uniform weights, fixed lambda.

    `penalty_fn(knots, K)` optionally replaces the coefficient second-difference penalty by another
    K x K matrix that is zero on constants (a constant coefficient shift is then free to leave
    out with the dropped first column of the later smooths).
    """

    def __init__(self, xs, y, K, lam, penalty_fn=None):
        lams = lam if isinstance(lam, list) else [lam] * len(xs)
        self.K, n = K, len(y)
        self.knots = [quantile_knots(c, K) for c in xs]
        bases = [[basis_row(v, k, K) for v in c] for c, k in zip(xs, self.knots)]
        cols, pen = [], []
        for j, B in enumerate(bases):
            keep = list(range(K)) if j == 0 else list(range(1, K))
            P = penalty(K) if penalty_fn is None else penalty_fn(self.knots[j], K)
            pen.append((len(cols), [[P[a][b] * lams[j] for b in keep] for a in keep]))
            for c in keep:
                cols.append([B[r][c] for r in range(n)])
        p = len(cols)
        A = [[sum(cols[a][r] * cols[b][r] for r in range(n)) for b in range(p)] for a in range(p)]
        XtX = [row[:] for row in A]
        for off, P in pen:
            for a in range(len(P)):
                for b in range(len(P)):
                    A[off + a][off + b] += P[a][b]
        rhs = [sum(cols[a][r] * y[r] for r in range(n)) for a in range(p)]
        # Exact LDL pivots certify strict positive definiteness of the reduced
        # penalized normal matrix; a unique penalized split is distinct from
        # identification by the data alone.
        L = [[F(int(i == j)) for j in range(p)] for i in range(p)]
        pivots = []
        for i in range(p):
            pivot = A[i][i] - sum(L[i][k] ** 2 * pivots[k] for k in range(i))
            assert pivot > 0, "penalized normal matrix is not positive definite"
            pivots.append(pivot)
            for j in range(i + 1, p):
                L[j][i] = (A[j][i] - sum(L[j][k] * L[i][k] * pivots[k] for k in range(i))) / pivot
        self.min_pivot = min(pivots)
        self.beta = solve(A, rhs)
        self.p, self.n, self.nsm = p, n, len(xs)
        Ainv = inverse(A)
        self.trace = sum(Ainv[a][b] * XtX[b][a] for a in range(p) for b in range(p))
        # coefficient offsets per smooth in the reparametrized vector
        self.offsets = [0] + [K] + [K + (K - 1) * (j - 1) for j in range(2, len(xs))] if len(xs) > 1 else [0]
        self.train_x = xs
        # centering constants: mean over the training rows of each smooth's raw function
        self.center = [sum(self.raw(j, c[r]) for r in range(n)) / n for j, c in enumerate(xs)]

    def raw(self, j, x):
        b = basis_row(x, self.knots[j], self.K)
        if j == 0:
            return sum(c * v for c, v in zip(self.beta[0:self.K], b))
        off = self.K + (self.K - 1) * (j - 1)
        return sum(c * v for c, v in zip(self.beta[off:off + self.K - 1], b[1:]))

    def derivative(self, j, x):
        """f_j'(x) of the uncentered smooth; zero outside [knots[3], knots[-4]) like the product."""
        kn = self.knots[j]
        left, right = kn[3], kn[len(kn) - 4]
        if x < left or x >= right:
            return F(0)
        N = []
        last = max(i for i in range(len(kn) - 1) if kn[i] < kn[i + 1])
        for i in range(len(kn) - 1):
            N.append(F(1) if kn[i] <= x < kn[i + 1] or (x == right and i == last) else F(0))
        for d in range(1, 3):
            M = []
            for i in range(len(kn) - 1 - d):
                a = F(0) if kn[i + d] == kn[i] else (x - kn[i]) / (kn[i + d] - kn[i]) * N[i]
                b = F(0) if kn[i + d + 1] == kn[i + 1] else (kn[i + d + 1] - x) / (kn[i + d + 1] - kn[i + 1]) * N[i + 1]
                M.append(a + b)
            N = M
        db = []
        for i in range(self.K):
            t1 = F(0) if kn[i + 3] == kn[i] else N[i] / (kn[i + 3] - kn[i])
            t2 = F(0) if kn[i + 4] == kn[i + 1] else N[i + 1] / (kn[i + 4] - kn[i + 1])
            db.append(3 * (t1 - t2))
        if j == 0:
            return sum(c * v for c, v in zip(self.beta[0:self.K], db))
        off = self.K + (self.K - 1) * (j - 1)
        return sum(c * v for c, v in zip(self.beta[off:off + self.K - 1], db[1:]))

    def smooth(self, j, x):
        return self.raw(j, x) - self.center[j]

    def intercept(self):
        return sum(self.center)

    def predict(self, row):
        return sum(self.raw(j, v) for j, v in enumerate(row))


def N(x, mean, sd):
    z = (mpf(x.numerator) / x.denominator - mean) / sd
    return exp(-z * z / 2) / (sd * msqrt(2 * pi))


def mf(x):
    x = F(x)
    return mpf(x.numerator) / x.denominator


def pseudo_outcomes(a, x, y, folds, K, lam):
    n = len(y)
    phi, c_term = [mpf(0)] * n, [mpf(0)] * n
    floor_rows = 0
    for f in range(folds):
        train = [i for i in range(n) if i % folds != f]
        valid = [i for i in range(n) if i % folds == f]
        ya = [y[i] for i in train]
        out = Additive([[a[i] for i in train], [x[i] for i in train]], ya, K, lam)
        tr = Additive([[x[i] for i in train]], [a[i] for i in train], K, lam)
        mt = {i: tr.predict([x[i]]) for i in range(n)}
        rss = sum((a[i] - mt[i]) ** 2 for i in train)
        edf = tr.trace
        df = max(F(len(train)) - edf, F(1))
        sigma = msqrt(mf(rss) / mf(df))
        # offset over all rows
        total = F(0)
        for i in range(n):
            mu = out.predict([a[i], x[i]])
            total += mu - out.smooth(0, a[i])
        offset = total / n
        for i in valid:
            mu = out.predict([a[i], x[i]])
            marg = offset + out.smooth(0, a[i])
            raw_pi = N(a[i], mf(mt[i]), sigma)
            if sigma * raw_pi <= mpf("1e-8"):
                floor_rows += 1
            scaled_pi = max(sigma * raw_pi, mpf("1e-8"))
            m_i = sum(N(a[i], mf(mt[j]), sigma) for j in train) / len(train)
            phi[i] = mf(marg) + mf(y[i] - mu) * sigma * m_i / scaled_pi
            c_term[i] = mf(mu - marg)
    return phi, c_term, floor_rows


def local_quadratic_level(a, phi, at, h):
    mp_at, mp_h = mf(at), mf(h)
    S = matrix(3, 3)
    b = matrix(3, 1)
    for ai, pi_ in zip(a, phi):
        t = mf(ai) - mp_at
        w = exp(-(t / mp_h) ** 2 / 2)
        xv = [mpf(1), t, t * t]
        for r in range(3):
            b[r] += w * xv[r] * pi_
            for c in range(3):
                S[r, c] += w * xv[r] * xv[c]
    return lu_solve(S, b)[0]


def mps(v):
    return mp.nstr(v, 40)


def build(name, a, x, y, folds, K, lam, h, grid):
    phi, c, floor_rows = pseudo_outcomes(a, x, y, folds, K, lam)
    return {
        "name": name, "folds": folds, "basis": K, "lambda": float(lam), "bandwidth": float(h),
        "grid": [float(g) for g in grid],
        "a": [str(float(v)) for v in a], "x": [str(float(v)) for v in x], "y": [str(float(v)) for v in y],
        "pseudo": [mps(v) for v in phi], "covariate_centered": [mps(v) for v in c],
        "floor_rows": floor_rows,
        "level": [mps(local_quadratic_level(a, phi, g, h)) for g in grid],
    }


def make_data(n, a_off, y_off, x_scale, a_noise=F(1, 2), outlier=False):
    x = [dy(u01() * 4 - 2, 1024) * x_scale for _ in range(n)]
    a = [a_off + dy(float(v / x_scale) * 0.6 + (u01() - 0.5) * float(a_noise) * 2, 1024) for v in x]
    y = [y_off + dy(1.5 * float(v - a_off) + 0.8 * float(w / x_scale) + (u01() - 0.5), 1024) for v, w in zip(a, x)]
    if outlier:
        a[n - 1] = a_off + 40
    return a, x, y


if __name__ == "__main__":
    cases = []
    a, x, y = make_data(60, F(0), F(0), F(1))
    cases.append(build("ordinary", a, x, y, 3, 5, F(1), F(3, 4), [F(-1, 2), F(0), F(1, 2)]))
    a, x, y = make_data(50, F(1000), F(1000000), F(250))
    cases.append(build("offsets_and_scales", a, x, y, 5, 5, F(1), F(3, 4), [F(999), F(1000), F(1001)]))
    a, x, y = make_data(49, F(0), F(0), F(1), outlier=True)
    cases.append(build("floor_binds", a, x, y, 2, 5, F(1), F(3, 4), [F(-1, 2), F(1, 2)]))
    a, x, y = make_data(48, F(0), F(0), F(1), a_noise=F(1, 8))
    cases.append(build("treatment_strongly_determined", a, x, y, 4, 5, F(1, 2), F(3, 4), [F(-1, 2), F(1, 2)]))
    # Near-collinearity makes the data-only split weak, but strict positive
    # definiteness of each penalized normal matrix gives a unique minimizer.
    a, x, y = make_data(48, F(0), F(0), F(1), a_noise=F(1, 64))
    near = build("treatment_nearly_determined", a, x, y, 4, 5, F(1, 2), F(3, 4), [F(-1, 2), F(1, 2)])
    near["outcome_min_ldl_pivots"] = [str(float(Additive(
        [[a[i] for i in range(len(a)) if i % 4 != f], [x[i] for i in range(len(x)) if i % 4 != f]],
        [y[i] for i in range(len(y)) if i % 4 != f], 5, F(1, 2)).min_pivot)) for f in range(4)]
    cases.append(near)
    a, x, y = make_data(720, F(0), F(0), F(1))
    cases.append(build("large_mixture", a, x, y, 5, 6, F(1), F(1, 2), [F(-1, 2), F(0), F(1, 2)]))
    print(json.dumps({"cases": cases}, indent=1))
