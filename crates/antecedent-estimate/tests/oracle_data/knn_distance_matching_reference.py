# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath"]
# ///
"""M-nearest-neighbour covariate-distance matching with replacement, at 60 digits.

Regenerate with:  uv run --script knn_distance_matching_reference.py > knn_distance_matching_reference.json

Every covariate is z-scored on the estimation sample (mean zero, population standard deviation
one) and the distance between two units is the Euclidean distance between their z-scores.  Each
query is compared with every donor of the opposite arm; its M matches are the M donors with the
smallest (distance, donor position) pairs, and it is unmatched when its M-th nearest donor is
beyond the caliper (a distance equal to the caliper matches).  The matched difference is the
query's outcome minus the mean outcome of its M matches, less mu(z_q) - mean mu(z_match), where mu is the least-squares plane of the donor arm's
outcome on the z-scores (a donor arm with fewer rows than coefficients uses its mean).  ATT
queries the treated, ATC the controls (sign reversed) and ATE both.  The standard error is the
homoskedastic matching variance sigma^2 * W / n^2 with W = n + sum u_j^2 for one arm and
W = sum (1 + u_j)^2 over every unit for both, u_j = K_j / M the donor's use count over M,
sigma^2 the sample variance of the matched effects divided by 1 + 1/M.
Values are written as nearest binary64.  Repository-authored (MIT OR Apache-2.0).
"""
import json
from mpmath import mp, mpf, matrix, lu_solve, sqrt

mp.dps = 60
_state = 0x3C6EF372FE94F82B


def u01():
    global _state
    _state = (_state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
    return (_state >> 11) / float(1 << 53)


def zscore(x):
    n, p = len(x), len(x[0])
    cols = []
    for c in range(p):
        col = [mpf(row[c]) for row in x]
        mean = sum(col) / n
        sd = sqrt(sum((v - mean) ** 2 for v in col) / n)
        cols.append([(v - mean) / sd for v in col])
    return [[cols[c][i] for c in range(p)] for i in range(n)]


def fit_plane(z, y, rows):
    p = len(z[0]) + 1
    if len(rows) < p:
        m = sum(y[j] for j in rows) / len(rows)
        return lambda f: m
    A = matrix(p, p)
    b = matrix(p, 1)
    for j in rows:
        r = [mpf(1)] + list(z[j])
        for a in range(p):
            b[a] += r[a] * y[j]
            for c in range(p):
                A[a, c] += r[a] * r[c]
    beta = lu_solve(A, b)
    return lambda f: beta[0] + sum(beta[k + 1] * f[k] for k in range(p - 1))


def dist(a, b):
    return sqrt(sum((u - v) ** 2 for u, v in zip(a, b)))


def match_arm(z, y, queries, donors, caliper, m):
    mu = fit_plane(z, y, donors)
    out = []
    for q in queries:
        d = sorted((dist(z[q], z[j]), j) for j in donors)[:m]
        if caliper is not None and d[-1][0] > caliper:
            continue
        tied = [j for _, j in d]
        raw = y[q] - sum(y[j] for j in tied) / len(tied)
        bias = mu(z[q]) - sum(mu(z[j]) for j in tied) / len(tied)
        out.append((q, tied, raw - bias))
    return out


def estimate(z, y, t, target, caliper, m):
    n_all = len(t)
    treated = [i for i in range(n_all) if t[i] > 0.5]
    control = [i for i in range(n_all) if t[i] <= 0.5]
    usage = [mpf(0)] * n_all
    effects, sizes, passes = [], [], []
    if target in ("att", "ate"):
        passes.append((match_arm(z, y, treated, control, caliper, m), 1))
    if target in ("atc", "ate"):
        passes.append((match_arm(z, y, control, treated, caliper, m), -1))
    for res, sign in passes:
        for q, tied, d in res:
            effects.append(sign * d)
            sizes.append(len(tied))
            for j in tied:
                usage[j] += mpf(1) / len(tied)
    n = mpf(len(effects))
    tau = sum(effects) / n
    var = sum((e - tau) ** 2 for e in effects) / (n - 1)
    sigma2 = var / (1 + mpf(1) / m)
    if target == "ate":
        weight = sum((1 + u) ** 2 for u in usage)
    else:
        weight = n + sum(u * u for u in usage)
    return tau, sqrt(sigma2 * weight / n**2), len(effects)


def case(name, t, y, x, m, caliper=None):
    z = zscore(x)
    yy = [mpf(v) for v in y]
    cal = None if caliper is None else mpf(caliper)
    out = {"name": name, "neighbors": m, "t": t, "y": y, "x": x, "caliper": caliper, "targets": {}}
    for target in ("att", "atc", "ate"):
        tau, se, matched = estimate(z, yy, t, target, cal, m)
        out["targets"][target] = {"ate": float(tau), "se": float(se), "matched": matched}
    return out


def draw(n, p, pool=None, slope=0.6):
    t, y, x = [], [], []
    points = None
    if pool:
        points = [[u01() * 4 - 2 for _ in range(p)] for _ in range(pool)]
    for i in range(n):
        row = points[int(u01() * pool)] if points else [u01() * 4 - 2 for _ in range(p)]
        eta = slope * row[0] - 0.4 * (row[1] if p > 1 else 0.0) + 0.1
        ti = 1.0 if u01() < 1 / (1 + 2.718281828459045 ** (-eta)) else 0.0
        yi = 1.0 + 1.5 * ti + 0.9 * row[0] + 0.5 * (row[1] if p > 1 else 0.0) + (u01() - 0.5)
        t.append(ti)
        y.append(float(yi))
        x.append(list(row))
    return t, y, x


cases = []
t, y, x = draw(40, 2)
cases.append(case("two_covariates_m3", t, y, x, 3))
cases.append(case("two_covariates_m3_caliper", t, y, x, 3, 0.8))
t3, y3, x3 = draw(50, 3)
cases.append(case("three_covariates_m5", t3, y3, x3, 5))
t4, y4, x4 = draw(60, 2, pool=7)
cases.append(case("duplicated_rows_tie_m3", t4, y4, x4, 3))
cases.append(case("duplicated_rows_tie_m4", t4, y4, x4, 4))
cases.append(case("covariate_scale_1e14_m3", t, y, [[r[0], 1.0e14 * r[1]] for r in x], 3))
t6, y6, x6 = draw(24, 2, slope=0.2)
cases.append(case("small_sample_m2", t6, y6, x6, 2))
print(json.dumps({"cases": cases}))
