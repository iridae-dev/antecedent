# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Penalized additive spline fits in exact rational arithmetic.

Regenerate with:  uv run --script gam_reference.py > gam_reference.json

Everything is exact (``fractions.Fraction``): data are the exact values of the doubles written
to the file, cubic B-spline bases are evaluated from the recursive Cox-de Boor definition (zero
for a zero-width term), type-7 quantile knots from their definition, and the fit is the unique
minimizer of

    sum_i w_i (y_i - b0 - sum_j B_j(x_ij) beta_j)^2 + sum_j lambda_j beta_j' D2' D2 beta_j

with weights normalized to sum n.  The design shares one constant direction (every B-spline block
sums to one, D2 annihilates constants), so the minimizer is computed on the reparametrization
that keeps the first smooth in full and drops the first coefficient of every later one; the fitted
function is the same.  No backfitting, Cholesky or eigendecomposition is shared with the
implementation.  For auto-lambda the generalized cross-validation score
n RSS / (n - df)^2 is evaluated exactly on the 25-point grid with df either the hat-matrix trace
including the intercept (Craven-Wahba) or the trace of the centered smoother alone.
Repository-authored (MIT OR Apache-2.0).
"""
import json
from fractions import Fraction as F

GRID = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0,
        30.0, 100.0, 300.0, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6]
_state = 0x2545F4914F6CDD1D


def u01():
    global _state
    _state = (_state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
    return (_state >> 11) / float(1 << 53)


def normal():
    import math
    return math.sqrt(-2 * math.log(u01() + 2.0 ** -60)) * math.cos(2 * math.pi * u01())


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
    # degree-0 indicators; the last non-empty interval is closed on the right
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


def design(x_cols, specs):
    """Returns knots per smooth and the per-smooth basis matrices (rows)."""
    knots, bases = [], []
    for s in specs:
        xs = x_cols[s["raw_col"]]
        k = quantile_knots(xs, s["n_basis"])
        knots.append(k)
        bases.append([basis_row(x, k, s["n_basis"]) for x in xs])
    return knots, bases


def fit(bases, y, w, lams, return_hat=False):
    n = len(y)
    tot = sum(w)
    w = [v * n / tot for v in w]
    # reparametrized columns: smooth 0 in full, later smooths without their first coefficient
    cols, pen_blocks = [], []
    for j, B in enumerate(bases):
        K = len(B[0])
        keep = range(K) if j == 0 else range(1, K)
        P = penalty(K)
        pen_blocks.append((len(cols), [list(keep)], [[P[a][b] for b in keep] for a in keep], lams[j]))
        for c in keep:
            cols.append([B[r][c] for r in range(n)])
    p = len(cols)
    A = [[sum(w[r] * cols[a][r] * cols[b][r] for r in range(n)) for b in range(p)] for a in range(p)]
    for off, _, P, lam in pen_blocks:
        for a in range(len(P)):
            for b in range(len(P)):
                A[off + a][off + b] += lam * P[a][b]
    rhs = [sum(w[r] * cols[a][r] * y[r] for r in range(n)) for a in range(p)]
    beta = solve(A, rhs)
    fitted = [sum(beta[a] * cols[a][r] for a in range(p)) for r in range(n)]
    out = {"beta": beta, "fitted": fitted, "cols": cols, "A": A, "w": w}
    return out


def hat_trace(res):
    """tr(X (X'WX + lam P)^-1 X'W) over the reparametrized design (intercept included)."""
    cols, A, w = res["cols"], res["A"], res["w"]
    p, n = len(cols), len(w)
    Ainv = inverse(A)
    XtWX = [[sum(w[r] * cols[a][r] * cols[b][r] for r in range(n)) for b in range(p)] for a in range(p)]
    return sum(Ainv[a][b] * XtWX[b][a] for a in range(p) for b in range(p))


def predict(beta_res, knots, specs, xnew_cols):
    out = []
    nn = len(xnew_cols[0])
    rows = [[] for _ in range(nn)]
    for j, s in enumerate(specs):
        K = s["n_basis"]
        for r in range(nn):
            b = basis_row(xnew_cols[s["raw_col"]][r], knots[j], K)
            rows[r].extend(b if j == 0 else b[1:])
    return [sum(c * v for c, v in zip(beta_res["beta"], rows[r])) for r in range(nn)]


def fl(v):
    return float(v)


def case(name, x_cols, y, specs, lams, w=None, xnew=None):
    n = len(y)
    X = [[F(v) for v in col] for col in x_cols]
    Y = [F(v) for v in y]
    W = [F(v) for v in (w or [1.0] * n)]
    knots, bases = design(X, specs)
    res = fit(bases, Y, W, [F(l) for l in lams])
    out = {
        "name": name, "x_cols": x_cols, "y": y, "weights": w, "specs": specs, "lambdas": lams,
        "fitted": [fl(v) for v in res["fitted"]],
        "knots": [[fl(v) for v in k] for k in knots],
        "hat_trace": fl(hat_trace(res)),
    }
    if xnew is not None:
        XN = [[F(v) for v in col] for col in xnew]
        out["xnew"] = xnew
        out["prediction"] = [fl(v) for v in predict(res, knots, specs, XN)]
    # sum of per-smooth centered traces + intercept, the quantity edf_approx reports
    if len(specs) > 1:
        tr = F(1)
        n_ = len(Y)
        for j in range(len(specs)):
            single = fit([bases[j]], Y, W, [F(lams[j])])
            tr += hat_trace(single) - 1
        out["sum_of_single_smooth_traces"] = fl(tr)
    return out


def gcv_case(name, x, y, K, w=None):
    n = len(y)
    X = [[F(v) for v in x]]
    Y = [F(v) for v in y]
    W = [F(v) for v in (w or [1.0] * n)]
    specs = [{"raw_col": 0, "n_basis": K}]
    _, bases = design(X, specs)
    scores = {"with_intercept": [], "smoother_only": []}
    for lam in GRID:
        res = fit(bases, Y, W, [F(lam)])
        tr = hat_trace(res)
        wn = [v * n / sum(W) for v in W]
        rss = sum(wn[r] * (Y[r] - res["fitted"][r]) ** 2 for r in range(n))
        scores["with_intercept"].append(rss * n / (n - tr) ** 2)
        scores["smoother_only"].append(rss * n / (n - (tr - 1)) ** 2)
    pick = {k: GRID[min(range(len(GRID)), key=lambda i: v[i])] for k, v in scores.items()}
    margin = {}
    for k, v in scores.items():
        s = sorted(v)
        margin[k] = fl((s[1] - s[0]) / s[0])
    return {"name": name, "x": x, "y": y, "weights": w, "n_basis": K, "pick": pick, "margin": margin,
            "scores_with_intercept": [fl(v) for v in scores["with_intercept"]]}


def main():
    cases = []
    n = 40
    x = [u01() * 3 for _ in range(n)]
    y = [__import__("math").sin(2 * v) + 0.2 * normal() for v in x]
    xn = [-0.5, 0.0, 0.7, 1.5, 2.9, max(x), 3.5]
    cases.append(case("fixed_single", [x], y, [{"raw_col": 0, "n_basis": 8}], [0.7], xnew=[xn]))

    n = 60
    x1 = [u01() * 2 - 1 for _ in range(n)]
    x2 = [u01() * 4 for _ in range(n)]
    x2 = [0.5 * a + 0.5 * b for a, b in zip(x2, [3 * (v + 1) for v in x1])]  # concurvity
    y = [v ** 2 + 0.5 * __import__("math").cos(2 * u) + 0.1 * normal() for v, u in zip(x1, x2)]
    w = [0.1 + 3 * u01() for _ in range(n)]
    w[7] = 0.0
    cases.append(case("fixed_two_weighted", [x1, x2], y,
                      [{"raw_col": 0, "n_basis": 6}, {"raw_col": 1, "n_basis": 7}], [0.3, 5.0], w=w,
                      xnew=[[-1.2, -0.3, 0.0, 0.9, 1.0, 1.4], [-1.0, 0.2, 1.5, 2.0, 3.9, 4.5]]))

    x = [u01() * 3 for _ in range(30)]
    y = [0.5 * v + 0.3 * normal() for v in x]
    cases.append(case("large_lambda", [x], y, [{"raw_col": 0, "n_basis": 8}], [1e7]))

    x = [(u01() * 3) * 1e-9 for _ in range(36)]
    y = [(__import__("math").sin(2 * v * 1e9) + 0.1 * normal()) * 1e9 for v in x]
    cases.append(case("tiny_x_huge_y", [x], y, [{"raw_col": 0, "n_basis": 7}], [0.5]))

    x = [float(i % 3) for i in range(30)]
    y = [(i % 3) * 1.5 + 0.1 * normal() for i in range(30)]
    cases.append(case("discrete_x", [x], y, [{"raw_col": 0, "n_basis": 6}], [0.2],
                      xnew=[[0.0, 0.5, 1.0, 2.0, 2.5]]))

    gcv = []
    x = [u01() * 3 for _ in range(50)]
    gcv.append(gcv_case("smooth_signal", x, [__import__("math").sin(2 * v) + 0.2 * normal() for v in x], 10))
    x = [u01() * 3 for _ in range(50)]
    gcv.append(gcv_case("pure_noise", x, [normal() for _ in x], 10))
    x = [u01() * 3 for _ in range(50)]
    gcv.append(gcv_case("nearly_exact", x, [__import__("math").sin(2 * v) + 1e-5 * normal() for v in x], 10))
    x = [u01() * 3 for _ in range(25)]
    gcv.append(gcv_case("few_rows_many_bases", x, [v * v + 0.3 * normal() for v in x], 14))
    x = [u01() * 3 for _ in range(60)]
    w = [0.05 + 4 * u01() for _ in x]
    gcv.append(gcv_case("weighted", x, [__import__("math").sin(3 * v) + 0.3 * normal() for v in x], 9, w=w))
    json.dump({"cases": cases, "gcv": gcv, "grid": GRID}, __import__("sys").stdout, separators=(",", ":"))


main()
