# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath"]
# ///
"""Definition-level references for the descriptive association estimator.

Regenerate in two stages (the resampling draws of the library are an input, not an output):

    uv run --script estimate_association_reference.py cases > estimate_association_reference.json
    cargo test -p antecedent-estimate --test estimate_association_reference \
        association_dump_library_streams -- --ignored --nocapture   # writes the streams file
    uv run --script estimate_association_reference.py final streams.json \
        > estimate_association_reference.json

Every case carries its data, so the references are fully determined.  Linear fits, the HC1 and
Bartlett Newey-West sandwiches, weighted group means and their linearised errors, the Pearson
error on the Kish count, the distance correlation and the leave-fold-out predictions are exact
rational arithmetic on the binary64 inputs; the logistic and Poisson maximum likelihood fits run
Newton at 80 digits; the Student-t quantile of the conjugate posterior is solved from the
regularised incomplete beta function.  Resampling statistics are recomputed from frozen index and
weight draws (captured from the library's own streams and checked for equality by the test).
Definitions:

* HC1 slope covariance  n/(n-p) A^-1 (sum s_i^2 x_i x_i') A^-1, A = sum I_i x_i x_i', s_i = w_i (y_i - mu_i),
  I_i = w_i V(mu_i), over the rows of positive weight.
* Newey-West  A^-1 [sum s^2 xx' + sum_{l=1..L} (1 - l/(L+1)) sum_t s_t s_{t-l} (x_t x_{t-l}' + x_{t-l} x_t')] A^-1
  with L = floor(4 (n/100)^(2/9)) clamped to [1, n-1].
* Group mean m_k = sum w y / sum w; influence psi_r = w_r (y_r - m_k)/sum_k w; variance sum psi^2 n_k/(n_k-1)
  (n_k rows of positive weight), or the Bartlett sum of the influence series for a series.
* Pearson: weighted correlation, error |1 - r^2| / sqrt(max(neff - 1, 1)), neff = (sum w)^2 / sum w^2.
* Distance correlation: V-statistic, sqrt(dCov2_xy / sqrt(dVar2_x dVar2_y)).
* Conjugate: beta | s2 ~ N(0, s2 Lambda0^-1), s2 ~ InvGamma(1, var_y); Student-t marginals.
Repository-authored (MIT OR Apache-2.0).
"""
import json
import math
import sys
from fractions import Fraction as F

import mpmath as mp

mp.mp.dps = 80
NAN = None  # a draw that did not fit
FIXED_B_INTERCEPT = 1.96


def mpf(v):
    if isinstance(v, F):
        return mp.mpf(v.numerator) / mp.mpf(v.denominator)
    return mp.mpf(v)


def tof(v):
    return float(mpf(v))


# ---------------------------------------------------------------- linear algebra, generic
def zero(v):
    return v == 0


def gauss_solve(a, b, tol=None):
    """Solve a x = b (b a list of columns' rows); None when singular."""
    n = len(a)
    m = [list(a[i]) + list(b[i]) for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if (tol is None and zero(m[piv][col])) or (tol is not None and abs(m[piv][col]) <= tol):
            return None
        m[col], m[piv] = m[piv], m[col]
        for r in range(n):
            if r != col:
                f = m[r][col] / m[col][col]
                if f != 0:
                    m[r] = [x - f * y for x, y in zip(m[r], m[col])]
    return [[m[i][n + j] / m[i][i] for j in range(len(b[0]))] for i in range(n)]


def identity(p, one):
    return [[one if i == j else one * 0 for j in range(p)] for i in range(p)]


def outer_add(acc, x, scale):
    p = len(x)
    for i in range(p):
        for j in range(p):
            acc[i][j] += scale * x[i] * x[j]


def to_exact(rows):
    return [[F(v) for v in row] for row in rows]


# ---------------------------------------------------------------- model fits
def expand(col):
    """Columns of one predictor: a numeric column is itself, a categorical column is its
    indicators of every present code except the reference (the declared one if present, else the
    lowest present code)."""
    if not isinstance(col, dict):
        return [col]
    present = sorted(set(col["codes"]))
    ref = col.get("reference")
    ref = ref if ref in present else present[0]
    return [[1.0 if c == lv else 0.0 for c in col["codes"]] for lv in present if lv != ref]


def design_columns(case):
    cols = case["columns"]
    reported = [[1.0] * len(cases_outcome(case))]
    for n in case["predictors"]:
        reported += expand(cols[n])
    rest = []
    for n in case["conditioning"]:
        rest += expand(cols[n])
    return reported, rest


def cases_outcome(case):
    return case["columns"][case["outcome"]]


def design_rows(case):
    """Rows of the design [1, predictors..., conditioning...]."""
    reported, rest = design_columns(case)
    allc = reported + rest
    return [[c[r] for c in allc] for r in range(len(allc[0]))]


def report_count(case):
    return len(design_columns(case)[0])


def family_of(case):
    return case["family"]


def fit_model(x, y, w, family):
    """Coefficients over the rows with w > 0, as exact or 80-digit numbers; None when singular."""
    keep = [i for i in range(len(y)) if w[i] > 0]
    p = len(x[0])
    if len(keep) <= p:
        return None
    if family == "continuous":
        X = to_exact([x[i] for i in keep])
        Y = [F(y[i]) for i in keep]
        W = [F(w[i]) for i in keep]
        a = [[F(0)] * p for _ in range(p)]
        b = [[F(0)] for _ in range(p)]
        for xi, yi, wi in zip(X, Y, W):
            outer_add(a, xi, wi)
            for j in range(p):
                b[j][0] += wi * xi[j] * yi
        sol = gauss_solve(a, b)
        return None if sol is None else [s[0] for s in sol]
    X = [[mpf(F(v)) for v in x[i]] for i in keep]
    Y = [mpf(F(y[i])) for i in keep]
    W = [mpf(F(w[i])) for i in keep]
    beta = [mp.mpf(0)] * p
    if family == "count":
        mean = sum(wi * yi for wi, yi in zip(W, Y)) / sum(W)
        if mean <= 0:
            return None
        beta[0] = mp.log(mean)
    for _ in range(200):
        a = [[mp.mpf(0)] * p for _ in range(p)]
        g = [[mp.mpf(0)] for _ in range(p)]
        for xi, yi, wi in zip(X, Y, W):
            eta = sum(c * b for c, b in zip(xi, beta))
            if family == "binary":
                mu = 1 / (1 + mp.exp(-eta))
                var = mu * (1 - mu)
            else:
                mu = mp.exp(eta)
                var = mu
            outer_add(a, xi, wi * var)
            for j in range(p):
                g[j][0] += wi * xi[j] * (yi - mu)
        step = gauss_solve(a, g, tol=mp.mpf(10) ** -60)
        if step is None:
            return None
        beta = [b + s[0] for b, s in zip(beta, step)]
        if max(abs(s[0]) for s in step) < mp.mpf(10) ** -60:
            return beta
    return None


def mean_function(family, eta):
    if family == "continuous":
        return eta
    if family == "binary":
        return 1 / (1 + mp.exp(-mpf(eta)))
    return mp.exp(mpf(eta))


def newey_west_lag(n):
    rule = max(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)), 1)
    return max(min(rule, n - 1), 1)


def sandwich(x, y, w, beta, family, modality):
    keep = [i for i in range(len(y)) if w[i] > 0]
    n, p = len(keep), len(x[0])
    exact = family == "continuous"
    conv = (lambda v: F(v)) if exact else (lambda v: mpf(F(v)))
    X = [[conv(v) for v in x[i]] for i in keep]
    s, info = [], []
    for k, i in enumerate(keep):
        eta = sum(c * b for c, b in zip(X[k], beta))
        mu = mean_function(family, eta)
        u = conv(y[i]) - mu
        if family == "binary":
            var = mu * (1 - mu)
        elif family == "count":
            var = mu
        else:
            var = conv(1)
        s.append(conv(w[i]) * u)
        info.append(conv(w[i]) * var)
    one = conv(1.0)
    a = [[one * 0] * p for _ in range(p)]
    for xi, ii in zip(X, info):
        outer_add(a, xi, ii)
    meat = [[one * 0] * p for _ in range(p)]
    for xi, si in zip(X, s):
        outer_add(meat, xi, si * si)
    if modality == "rows":
        f = conv(n) / conv(n - p)
        meat = [[f * v for v in row] for row in meat]
    else:
        lag = newey_west_lag(n)
        for ell in range(1, min(lag, n - 1) + 1):
            wt = 1 - conv(ell) / conv(lag + 1)
            for t in range(ell, n):
                c = wt * s[t] * s[t - ell]
                for i in range(p):
                    for j in range(p):
                        meat[i][j] += c * (X[t][i] * X[t - ell][j] + X[t - ell][i] * X[t][j])
    ainv = gauss_solve(a, identity(p, one), tol=None if exact else mp.mpf(10) ** -60)
    tmp = [[sum(ainv[i][k] * meat[k][j] for k in range(p)) for j in range(p)] for i in range(p)]
    return [[sum(tmp[i][k] * ainv[k][j] for k in range(p)) for j in range(p)] for i in range(p)]


# ---------------------------------------------------------------- strata
def strata_of(case):
    cols = case["columns"]
    keys_cols = case["predictors"] + case["conditioning"]
    n = len(cols[case["outcome"]])
    coarsen = {}
    for name in keys_cols:
        vals = cols[name]
        if isinstance(vals, dict):
            coarsen[name] = list(vals["codes"])
            continue
        distinct = sorted(set(vals))
        if len(distinct) > 64:
            srt = sorted(vals)
            total = len(srt)
            edges = [(b * total) // 64 for b in range(65)]
            upper = []
            for b in range(64):
                lo, hi = edges[b], edges[b + 1]
                if lo >= hi:
                    continue
                top = srt[hi - 1]
                if upper and top <= upper[-1]:
                    continue
                upper.append(top)
            code = []
            for v in vals:
                pos = sum(1 for t in upper if t < v)
                code.append(min(pos, len(upper) - 1))
        else:
            code = [distinct.index(v) for v in vals]
        coarsen[name] = code
    keys = [tuple(coarsen[nm][r] for nm in keys_cols) for r in range(n)]
    order = sorted(set(keys))
    return [order.index(k) for k in keys], len(order)


def lagged_variance(psi, lag):
    total = sum(v * v for v in psi)
    for ell in range(1, min(lag, len(psi) - 1) + 1):
        wt = 1 - F(ell) / F(lag + 1)
        total += 2 * wt * sum(psi[t] * psi[t - ell] for t in range(ell, len(psi)))
    return max(total, 0)


def strata_statistics(case, eff):
    of_row, k = strata_of(case)
    y = case["columns"][case["outcome"]]
    out = []
    for g in range(k):
        num = sum(F(eff[r]) * F(y[r]) for r in range(len(y)) if of_row[r] == g)
        den = sum(F(eff[r]) for r in range(len(y)) if of_row[r] == g)
        out.append(None if den <= 0 else tof(num / den))
    return out


def strata_errors(case, w):
    of_row, k = strata_of(case)
    y = case["columns"][case["outcome"]]
    n = len(y)
    out = []
    for g in range(k):
        members = [r for r in range(n) if of_row[r] == g]
        total = sum(F(w[r]) for r in members)
        positive = sum(1 for r in members if w[r] > 0)
        if positive < 2 or total <= 0:
            out.append(None)
            continue
        m = sum(F(w[r]) * F(y[r]) for r in members) / total
        psi = [F(w[r]) * (F(y[r]) - m) / total if of_row[r] == g else F(0) for r in range(n)]
        if case["modality"] == "series":
            var = lagged_variance(psi, newey_west_lag(n))
        else:
            var = sum(v * v for v in psi) * F(positive) / F(positive - 1)
        out.append(tof(mp.sqrt(mpf(var))))
    return out


# ---------------------------------------------------------------- correlations
def weighted_pearson(x, y, w):
    tw = sum(F(v) for v in w)
    if tw <= 0:
        return None
    mx = sum(F(a) * F(b) for a, b in zip(x, w)) / tw
    my = sum(F(a) * F(b) for a, b in zip(y, w)) / tw
    sxy = sum(F(c) * (F(a) - mx) * (F(b) - my) for a, b, c in zip(x, y, w))
    sxx = sum(F(c) * (F(a) - mx) ** 2 for a, c in zip(x, w))
    syy = sum(F(c) * (F(b) - my) ** 2 for b, c in zip(y, w))
    if sxx <= 0 or syy <= 0:
        return None
    return tof(mpf(sxy) / mp.sqrt(mpf(sxx) * mpf(syy)))


def pearson_error(r, w):
    tw = sum(F(v) for v in w)
    sq = sum(F(v) ** 2 for v in w)
    neff = mpf(tw * tw / sq)
    return tof(abs((1 - mpf(r) ** 2) / mp.sqrt(max(neff - 1, mp.mpf(1)))))


def distance_correlation(x, y):
    n = len(x)
    if n < 2:
        return 0.0

    def centred(v):
        v = [F(a) for a in v]
        d = [[abs(a - b) for b in v] for a in v]
        row = [sum(r) / n for r in d]
        tot = sum(row) / n
        return [[d[i][j] - row[i] - row[j] + tot for j in range(n)] for i in range(n)]

    a, b = centred(x), centred(y)

    def cov(p, q):
        return sum(p[i][j] * q[i][j] for i in range(n) for j in range(n)) / (n * n)

    dxx, dyy, dxy = cov(a, a), cov(b, b), cov(a, b)
    if dxx <= 0 or dyy <= 0:
        return 0.0
    return tof(mp.sqrt(max(mpf(dxy), 0) / mp.sqrt(mpf(dxx) * mpf(dyy))))


# ---------------------------------------------------------------- the reported statistic vector
def statistic(case, eff):
    """The reported statistic vector under effective row weights `eff` (None = did not fit)."""
    kind = case["kind"]
    if kind == "strata":
        return strata_statistics(case, eff)
    y = case["columns"][case["outcome"]]
    if kind == "model":
        x = design_rows(case)
        beta = fit_model(x, y, eff, case["family"])
        if beta is None:
            return [None] * (report_count(case))
        return [tof(b) for b in beta[: report_count(case)]]
    xs = case["columns"][case["predictors"][0]]
    if kind == "pearson":
        return [weighted_pearson(xs, y, eff)]
    copies = [int(round(v)) for v in eff]
    px = [xs[i] for i in range(len(y)) for _ in range(max(copies[i], 0))]
    py = [y[i] for i in range(len(y)) for _ in range(max(copies[i], 0))]
    return [distance_correlation(px, py)]


def case_weights(case):
    n = len(case["columns"][case["outcome"]])
    return case["columns"][case["weight"]] if case.get("weight") else [1.0] * n


def analytic(case):
    w = case_weights(case)
    kind = case["kind"]
    if kind == "strata":
        return strata_errors(case, w)
    if kind == "model":
        y = case["columns"][case["outcome"]]
        x = design_rows(case)
        beta = fit_model(x, y, w, case["family"])
        v = sandwich(x, y, w, beta, case["family"], case["modality"])
        return [tof(mp.sqrt(max(mpf(v[i][i]), 0))) for i in range(report_count(case))]
    if kind == "pearson" and case["modality"] == "rows":
        r = statistic(case, w)[0]
        return [pearson_error(r, w)]
    return [None]


# ---------------------------------------------------------------- resampling summaries
def sample_std(values):
    n = len(values)
    m = sum(F(v) for v in values) / n
    return tof(mp.sqrt(mpf(sum((F(v) - m) ** 2 for v in values) / (n - 1))))


def quantile7(sorted_values, p):
    d = len(sorted_values)
    h = (d - 1) * mp.mpf(p)
    lo = int(mp.floor(h))
    hi = min(lo + 1, d - 1)
    frac = h - lo
    return tof(mpf(F(sorted_values[lo])) * (1 - frac) + mpf(F(sorted_values[hi])) * frac)


def spread(values):
    if len(values) < 20:
        return None
    s = sorted(values)
    return [quantile7(s, 0.025), quantile7(s, 0.975)]


def multiplicities(index, n):
    m = [0.0] * n
    for i in index:
        m[i] += 1.0
    return m


def circular_scale(length, rows):
    b = min(max(length / rows, 0.0), 1.0)
    return (FIXED_B_INTERCEPT + 2.4389 * b + 3.7072 * b * b - 2.1055 * b * b * b) / FIXED_B_INTERCEPT


def bootstrap_expect(case, streams):
    n = len(case["columns"][case["outcome"]])
    w = case_weights(case)
    terms = len(case["term_count_hint"])
    draws = []
    for index in streams["indexes"]:
        mult = multiplicities(index, n)
        draws.append(statistic(case, [a * b for a, b in zip(w, mult)]))
    scale = circular_scale(streams["block_length"], n) if case["modality"] == "series" else 1.0
    attempted = len(draws)
    ok = attempted
    out = []
    for k in range(terms):
        vals = [d[k] for d in draws if d[k] is not None]
        ok = min(ok, len(vals))
        se = None
        if len(vals) >= 2 and len(vals) * 2 >= attempted:
            se = tof(mpf(sample_std(vals)) * mpf(scale))
        out.append({"se": se, "spread": spread(vals)})
    return {"terms": out, "ok": ok}


def bayes_boot_expect(case, streams):
    n = len(case["columns"][case["outcome"]])
    w = case_weights(case)
    terms = len(case["term_count_hint"])
    draws = []
    for weights in streams["weights"]:
        if case["modality"] == "series":
            ell = streams["block_length"]
            mult = [weights[r // ell] for r in range(n)]
        else:
            mult = weights
        draws.append(statistic(case, [a * b for a, b in zip(w, mult)]))
    out = []
    for k in range(terms):
        vals = [d[k] for d in draws if d[k] is not None]
        if len(vals) < 20:
            out.append(None)
            continue
        s = sorted(vals)
        mean = tof(sum(F(v) for v in vals) / len(vals))
        out.append(
            {"mean": mean, "sd": sample_std(vals), "lower": quantile7(s, 0.025), "upper": quantile7(s, 0.975)}
        )
    return out


# ---------------------------------------------------------------- conjugate posterior
def t_cdf(t, nu):
    t, nu = mp.mpf(t), mp.mpf(nu)
    x = nu / (nu + t * t)
    tail = mp.betainc(nu / 2, mp.mpf(1) / 2, 0, x, regularized=True) / 2
    return 1 - tail if t > 0 else tail


def t_ppf(q, nu):
    q = mp.mpf(q)
    return mp.findroot(lambda t: t_cdf(t, nu) - q, mp.mpf(2), tol=mp.mpf(10) ** -50, maxsteps=200)


def conjugate_expect(case, prior_scale):
    y = [F(v) for v in case["columns"][case["outcome"]]]
    n = len(y)
    my = sum(y) / n
    vy = sum((v - my) ** 2 for v in y) / n
    s2 = F(prior_scale) ** 2
    if case["kind"] == "strata":
        of_row, k = strata_of(case)
        designs = []
        for g in range(k):
            members = [r for r in range(n) if of_row[r] == g]
            designs.append(([[F(1)] for _ in members], [y[r] for r in members], [1 / s2]))
        results = [posterior(x, yy, prec, vy) for x, yy, prec in designs]
        return [r[0][0] for r in results]
    x = [[F(v) for v in row] for row in design_rows(case)]
    p = len(x[0])
    prec = [1 / s2]
    for c in range(1, p):
        col = [row[c] for row in x]
        mc = sum(col) / n
        prec.append(sum((v - mc) ** 2 for v in col) / n / s2)
    res = posterior(x, y, prec, vy)
    return res[1][: report_count(case)]


def posterior(x, y, prec, vy):
    """Returns (single functional summary list, summaries of every coefficient)."""
    n, p = len(y), len(x[0])
    lam = [[F(0)] * p for _ in range(p)]
    xty = [[F(0)] for _ in range(p)]
    for xi, yi in zip(x, y):
        outer_add(lam, xi, F(1))
        for j in range(p):
            xty[j][0] += xi[j] * yi
    for j in range(p):
        lam[j][j] += prec[j]
    inv = gauss_solve(lam, identity(p, F(1)))
    mean = [sum(inv[i][j] * xty[j][0] for j in range(p)) for i in range(p)]
    rss = sum((yi - sum(c * m for c, m in zip(xi, mean))) ** 2 for xi, yi in zip(x, y))
    shrink = sum(pr * m * m for pr, m in zip(prec, mean))
    shape = 1 + F(n, 2)
    scale = vy + (rss + shrink) / 2
    nu = 2 * shape
    out = []
    for j in range(p):
        loc = mean[j]
        sc = mp.sqrt(mpf(scale / shape * inv[j][j]))
        sd = sc * mp.sqrt(mpf(nu) / (mpf(nu) - 2))
        radius = sc * t_ppf(mp.mpf("0.975"), mpf(nu))
        out.append(
            {
                "mean": tof(loc),
                "sd": tof(sd),
                "lower": tof(mpf(loc) - radius),
                "upper": tof(mpf(loc) + radius),
            }
        )
    return [out[0]], out


# ---------------------------------------------------------------- prediction
def predict_expect(case, assignment, folds):
    y = case["columns"][case["outcome"]]
    w = case_weights(case)
    x = design_rows(case)
    n = len(y)
    family = case["family"]
    p = len(x[0])
    cases = [None] * n
    floor = 1e-12
    for f in range(folds):
        train = [0.0 if assignment[r] == f else w[r] for r in range(n)]
        beta = fit_model(x, y, train, family)
        keep = [r for r in range(n) if train[r] > 0]
        total = sum(F(train[r]) for r in keep)
        count = len(keep)
        base = sum(F(train[r]) * F(y[r]) for r in keep) / total
        resid_mp = mp.mpf(0)
        for r in keep:
            eta = sum(mpf(F(c)) * mpf(b) for c, b in zip(x[r], beta))
            mu = mean_function(family, eta)
            resid_mp += mpf(F(train[r])) * (mpf(F(y[r])) - mu) ** 2
        psd = mp.sqrt(max(resid_mp / (mpf(total) * (1 - mp.mpf(p) / count)), 0))
        psd = max(tof(psd), floor)
        bsd = mp.sqrt(max(sum(F(train[r]) * (F(y[r]) - base) ** 2 for r in keep) / total, 0))
        bsd = max(tof(bsd), floor)
        for r in range(n):
            if assignment[r] != f:
                continue
            eta = sum(mpf(F(c)) * mpf(b) for c, b in zip(x[r], beta))
            mu = tof(mean_function(family, eta))
            b = tof(base)
            if family == "binary":
                mu, b = min(max(mu, floor), 1 - floor), min(max(b, floor), 1 - floor)
            elif family == "count":
                mu, b = max(mu, floor), max(b, floor)
            cases[r] = {
                "outcome": y[r],
                "weight": w[r],
                "predicted": mu,
                "predicted_sd": psd,
                "baseline": b,
                "baseline_sd": bsd,
            }
    return cases


# ---------------------------------------------------------------- data
class Lcg:
    def __init__(self, seed):
        self.s = seed

    def unit(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        return (self.s >> 40) / float(1 << 24)

    def grid(self, lo, hi, denom=64):
        return round((lo + (hi - lo) * self.unit()) * denom) / denom


def make_cases():
    cases = []

    def add(**kw):
        kw.setdefault("conditioning", [])
        kw.setdefault("weight", None)
        kw.setdefault("modality", "rows")
        kw.setdefault("family", "continuous")
        cases.append(kw)

    # ---- linear HC1, offsets, conditioning column
    g = Lcg(11)
    n = 14
    x = [1000.0 + g.grid(-1, 1) for _ in range(n)]
    z = [float(i % 2) for i in range(n)]
    y = [1e7 + 0.75 * (xi - 1000.0) + 0.5 * zi + g.grid(-0.5, 0.5) for xi, zi in zip(x, z)]
    add(name="linear_hc1_offsets", kind="model", columns={"y": y, "x": x, "z": z}, outcome="y",
        predictors=["x"], conditioning=["z"])

    # ---- near collinear predictors (corr ~ 0.99999)
    g = Lcg(12)
    n = 30
    x1 = [g.grid(-4, 4) for _ in range(n)]
    x2 = [a + g.grid(-1, 1) / 512 for a in x1]
    y = [1 + 2 * a - 1.5 * b + g.grid(-1, 1) for a, b in zip(x1, x2)]
    add(name="linear_hc1_near_collinear", kind="model", columns={"y": y, "x1": x1, "x2": x2},
        outcome="y", predictors=["x1", "x2"])

    # ---- extreme weights, a zero weight and a high leverage point
    g = Lcg(13)
    n = 12
    x = [g.grid(-2, 2) for _ in range(n)]
    x[3] = 40.0
    y = [0.5 + 1.25 * a + g.grid(-1, 1) for a in x]
    w = [1e-3, 1.0, 1e3, 0.5, 0.0, 2.0, 1e3, 1e-3, 3.0, 1.0, 0.25, 8.0]
    add(name="linear_hc1_extreme_weights", kind="model", columns={"y": y, "x": x, "w": w},
        outcome="y", predictors=["x"], weight="w")

    # ---- smallest admissible sample: one residual degree of freedom
    add(name="linear_hc1_one_residual_df", kind="model",
        columns={"y": [1.0, 2.5, 2.0, 4.5], "x1": [0.0, 1.0, 2.0, 3.0], "x2": [1.0, 0.0, 1.0, 0.5]},
        outcome="y", predictors=["x1", "x2"])

    # ---- logistic and Poisson, weighted
    g = Lcg(14)
    n = 40
    x = [g.grid(-2, 2) for _ in range(n)]
    yb = [1.0 if (g.unit() < 1 / (1 + math.exp(-(0.3 + 0.9 * a)))) else 0.0 for a in x]
    wl = [g.grid(0.25, 4) for _ in range(n)]
    add(name="logistic_hc1_weighted", kind="model", family="binary",
        columns={"y": yb, "x": x, "w": wl}, outcome="y", predictors=["x"], weight="w")
    g = Lcg(15)
    x1 = [g.grid(-1, 1) for _ in range(n)]
    x2 = [float(i % 3) for i in range(n)]
    yc = [float(int(math.exp(0.4 + 0.5 * a + 0.2 * b) * g.unit() * 2.2)) for a, b in zip(x1, x2)]
    wp = [g.grid(0.5, 3) for _ in range(n)]
    add(name="poisson_hc1_weighted", kind="model", family="count",
        columns={"y": yc, "x1": x1, "x2": x2, "w": wp}, outcome="y", predictors=["x1"],
        conditioning=["x2"], weight="w")

    # ---- Newey-West, several lengths
    for nn, seed in ((12, 21), (60, 22), (100, 23)):
        g = Lcg(seed)
        x, y, prev_x, prev_y = [], [], 0.0, 0.0
        for _ in range(nn):
            prev_x = 0.75 * prev_x + g.grid(-1, 1)
            prev_y = 0.5 * prev_x + 0.6 * prev_y + g.grid(-1, 1)
            x.append(round(prev_x * 64) / 64)
            y.append(round(prev_y * 64) / 64)
        add(name=f"linear_newey_west_{nn}", kind="model", modality="series",
            columns={"y": y, "x": x}, outcome="y", predictors=["x"])
    g = Lcg(24)
    n = 50
    x = [g.grid(-2, 2) for _ in range(n)]
    ybs = [1.0 if g.unit() < 1 / (1 + math.exp(-(0.2 + a))) else 0.0 for a in x]
    add(name="logistic_newey_west", kind="model", family="binary", modality="series",
        columns={"y": ybs, "x": x}, outcome="y", predictors=["x"])
    yps = [float(int(math.exp(0.3 + 0.4 * a) * g.unit() * 2.0)) for a in x]
    add(name="poisson_newey_west", kind="model", family="count", modality="series",
        columns={"y": yps, "x": x}, outcome="y", predictors=["x"])

    # ---- group means
    g = Lcg(31)
    n = 24
    lv = [float(i % 4) for i in range(n)]
    lv[5] = 9.0  # a singleton stratum
    y = [2.0 * a + g.grid(-1, 1) for a in lv]
    add(name="strata_unweighted_with_singleton", kind="strata", columns={"y": y, "g": lv},
        outcome="y", predictors=["g"])
    w = [g.grid(0.1, 5) for _ in range(n)]
    w[0] = 0.0
    w[1] = 1e3
    add(name="strata_weighted_zero_weight_row", kind="strata", columns={"y": y, "g": lv, "w": w},
        outcome="y", predictors=["g"], weight="w")
    # a stratum of three rows whose weights are (1, 1, 0): two rows carry information
    add(name="strata_zero_weight_leaves_two_rows", kind="strata",
        columns={"y": [1.0, 3.0, 100.0, 5.0, 6.0, 7.0], "g": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                 "w": [1.0, 1.0, 0.0, 1.0, 2.0, 3.0]},
        outcome="y", predictors=["g"], weight="w")
    g = Lcg(32)
    n = 40
    a = [float(i % 2) for i in range(n)]
    b = [float((i // 2) % 3) for i in range(n)]
    yb = [1.0 if g.unit() < 0.3 + 0.2 * p else 0.0 for p in a]
    add(name="strata_probability_two_columns", kind="strata", family="binary",
        columns={"y": yb, "a": a, "b": b}, outcome="y", predictors=["a"], conditioning=["b"])
    g = Lcg(33)
    n = 60
    lv = [float(i % 5) for i in range(n)]
    y, prev = [], 0.0
    for k in range(n):
        prev = 0.6 * prev + g.grid(-1, 1)
        y.append(round((prev + lv[k]) * 64) / 64)
    add(name="strata_newey_west", kind="strata", modality="series", columns={"y": y, "g": lv},
        outcome="y", predictors=["g"])
    g = Lcg(34)
    n = 200
    xc = [float(i) for i in range(n)]
    y = [0.05 * a + g.grid(-1, 1) for a in xc]
    add(name="strata_coarsened_numeric", kind="strata", columns={"y": y, "x": xc},
        outcome="y", predictors=["x"])
    add(name="strata_marginal_law", kind="strata", columns={"y": [1.0, 2.0, 4.0, 8.0, 9.0]},
        outcome="y", predictors=[])

    # ---- categorical predictors: the declared reference level is the omitted indicator
    g = Lcg(71)
    n = 24
    codes = [i % 3 for i in range(n)]
    xnum = [g.grid(-1, 1) for _ in range(n)]
    y = [1.0 + 0.5 * c + 0.75 * a + g.grid(-0.5, 0.5) for c, a in zip(codes, xnum)]
    add(name="categorical_declared_reference_regression", kind="model",
        columns={"y": y, "c": {"codes": codes, "labels": ["a", "b", "c"], "reference": 2}, "x": xnum},
        outcome="y", predictors=["c"], conditioning=["x"])
    codes2 = [1 + (i % 3) for i in range(n)]
    add(name="categorical_absent_reference_falls_back_to_lowest_present", kind="model",
        columns={"y": y, "c": {"codes": codes2, "labels": ["a", "b", "c", "d"], "reference": 0}, "x": xnum},
        outcome="y", predictors=["c"], conditioning=["x"])
    add(name="categorical_strata", kind="strata",
        columns={"y": y, "c": {"codes": codes, "labels": ["a", "b", "c"], "reference": 2}},
        outcome="y", predictors=["c"])

    # ---- missing cells drop whole rows
    g = Lcg(81)
    n = 16
    x = [g.grid(-2, 2) for _ in range(n)]
    y = [1 + a + g.grid(-1, 1) for a in x]
    w = [g.grid(0.5, 2) for _ in range(n)]
    y[3] = None
    x[7] = None
    w[11] = None
    add(name="missing_cells_drop_rows", kind="model", columns={"y": y, "x": x, "w": w},
        outcome="y", predictors=["x"], weight="w")

    # ---- correlations
    g = Lcg(41)
    n = 15
    x = [g.grid(-3, 3) for _ in range(n)]
    y = [0.5 * a + g.grid(-2, 2) for a in x]
    add(name="pearson_unweighted", kind="pearson", columns={"y": y, "x": x}, outcome="y",
        predictors=["x"])
    w = [g.grid(0.1, 6) for _ in range(n)]
    w[2] = 0.0
    add(name="pearson_weighted_kish", kind="pearson", columns={"y": y, "x": x, "w": w},
        outcome="y", predictors=["x"], weight="w")
    add(name="pearson_three_rows_with_floor", kind="pearson",
        columns={"y": [1.0, 3.0, 2.0], "x": [0.0, 1.0, 2.5]}, outcome="y", predictors=["x"])
    add(name="pearson_series_has_no_error", kind="pearson", modality="series",
        columns={"y": y, "x": x}, outcome="y", predictors=["x"])
    xd = [-3.0, -2.0, -2.0, -1.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0]
    yd = [9.0, 4.0, 4.5, 1.0, 0.0, 0.25, 1.0, 4.0, 9.0, 8.0]
    add(name="distance_with_ties", kind="distance", columns={"y": yd, "x": xd}, outcome="y",
        predictors=["x"])
    add(name="distance_three_rows", kind="distance",
        columns={"y": [1.0, 5.0, 2.0], "x": [0.0, 1.0, 2.0]}, outcome="y", predictors=["x"])
    g = Lcg(42)
    x = [1e6 + g.grid(-1, 1) for _ in range(20)]
    y = [(a - 1e6) ** 2 + g.grid(-0.1, 0.1) for a in x]
    add(name="distance_offset_nonlinear", kind="distance", columns={"y": y, "x": x}, outcome="y",
        predictors=["x"])

    for c in cases:
        c["term_count_hint"] = term_hint(c)
    return cases


def term_hint(case):
    if case["kind"] == "model":
        return [0] * report_count(case)
    if case["kind"] == "strata":
        return [0] * strata_of(case)[1]
    return [0]


# resampling cases are attached to a base case with these options
def make_resampling():
    base = {c["name"]: c for c in make_cases()}
    plans = []

    def plan(name, source, **kw):
        c = json.loads(json.dumps(base[source]))
        c["name"] = name
        c.update(kw)
        c["term_count_hint"] = term_hint(c)
        plans.append(c)

    g = Lcg(51)
    n = 12
    x = [g.grid(-2, 2) for _ in range(n)]
    for k in range(10):
        x[k] = 1.0  # most rows tie, so a resample of only the tied rows is singular
    y = [0.5 + a + g.grid(-1, 1) for a in x]
    c = {"name": "row_bootstrap_linear_with_singular_draws", "kind": "model", "family": "continuous",
         "modality": "rows", "columns": {"y": y, "x": x}, "outcome": "y", "predictors": ["x"],
         "conditioning": [], "weight": None, "boot": 40}
    c["term_count_hint"] = term_hint(c)
    plans.append(c)
    plan("row_bootstrap_ten_replicates_has_no_spread", "linear_hc1_extreme_weights", boot=10)
    plan("row_bootstrap_strata_weighted", "strata_weighted_zero_weight_row", boot=30)
    plan("row_bootstrap_pearson", "pearson_unweighted", boot=30)
    c = {"name": "row_bootstrap_pearson_mostly_undefined_keeps_the_analytic_error", "kind": "pearson",
         "family": "continuous", "modality": "rows",
         "columns": {"y": [0.0, 1.0, 0.0, 0.0], "x": [0.0, 0.0, 0.0, 1.0]}, "outcome": "y",
         "predictors": ["x"], "conditioning": [], "weight": None, "boot": 40}
    c["term_count_hint"] = term_hint(c)
    plans.append(c)
    plan("row_bootstrap_distance", "distance_with_ties", boot=25)
    g = Lcg(52)
    n = 40
    x, y, px, py = [], [], 0.0, 0.0
    for _ in range(n):
        px = 0.8 * px + g.grid(-1, 1)
        py = 0.5 * px + 0.6 * py + g.grid(-1, 1)
        x.append(round(px * 64) / 64)
        y.append(round(py * 64) / 64)
    c = {"name": "circular_block_bootstrap_linear", "kind": "model", "family": "continuous",
         "modality": "series", "columns": {"y": y, "x": x}, "outcome": "y", "predictors": ["x"],
         "conditioning": [], "weight": None, "boot": 40}
    c["term_count_hint"] = term_hint(c)
    plans.append(c)
    c2 = json.loads(json.dumps(c))
    c2.update({"name": "circular_block_bootstrap_pearson", "kind": "pearson"})
    c2["term_count_hint"] = term_hint(c2)
    plans.append(c2)
    c3 = json.loads(json.dumps(c))
    c3.update({"name": "circular_block_bootstrap_strata", "kind": "strata"})
    c3["columns"] = {"y": y, "g": [float(i % 3) for i in range(n)]}
    c3["predictors"] = ["g"]
    c3["term_count_hint"] = term_hint(c3)
    plans.append(c3)

    # Bayesian bootstrap
    plan("bayes_bootstrap_logistic_weighted", "logistic_hc1_weighted", bayes={"draws": 30, "prior_scale": 2.0})
    plan("bayes_bootstrap_poisson_weighted", "poisson_hc1_weighted", bayes={"draws": 25, "prior_scale": 2.0})
    plan("bayes_bootstrap_linear_weighted", "linear_hc1_extreme_weights", bayes={"draws": 30, "prior_scale": 2.0})
    plan("bayes_bootstrap_pearson", "pearson_unweighted", bayes={"draws": 25, "prior_scale": 2.0})
    plan("bayes_bootstrap_blocks_linear", "linear_newey_west_60", bayes={"draws": 30, "prior_scale": 2.0})
    plan("bayes_bootstrap_blocks_strata", "strata_newey_west", bayes={"draws": 30, "prior_scale": 2.0})

    # conjugate posterior (no draws)
    plan("conjugate_linear_offset_outcome", "linear_hc1_offsets", bayes={"draws": 20, "prior_scale": 0.5})
    plan("conjugate_linear_near_collinear", "linear_hc1_near_collinear", bayes={"draws": 20, "prior_scale": 100.0})
    plan("conjugate_strata", "strata_unweighted_with_singleton", bayes={"draws": 20, "prior_scale": 1.5})
    plan("conjugate_marginal", "strata_marginal_law", bayes={"draws": 20, "prior_scale": 3.0})

    # prediction
    g = Lcg(61)
    n = 18
    x = [g.grid(-2, 2) for _ in range(n)]
    z = [float(i % 2) for i in range(n)]
    y = [1 + 2 * a - z[i] + g.grid(-1, 1) for i, a in enumerate(x)]
    w = [g.grid(0.5, 3) for _ in range(n)]
    c = {"name": "predictive_linear_random_folds", "kind": "model", "family": "continuous",
         "modality": "rows", "columns": {"y": y, "x": x, "z": z, "w": w}, "outcome": "y",
         "predictors": ["x"], "conditioning": ["z"], "weight": "w", "folds": 3}
    c["term_count_hint"] = term_hint(c)
    plans.append(c)
    plan("predictive_logistic_random_folds", "logistic_hc1_weighted", folds=4)
    plan("predictive_poisson_random_folds", "poisson_hc1_weighted", folds=4)
    plan("predictive_linear_series_blocks", "linear_newey_west_60", folds=4)
    plan("predictive_logistic_series_blocks", "logistic_newey_west", folds=3)
    return plans


def selection_cases():
    num = [0.5, 1.5, 2.25, 3.0, 4.5, 5.0]
    zo = [0.0, 1.0, 1.0, 0.0, 1.0, 0.0]
    cnt = [0.0, 3.0, 1.0, 2.0, 5.0, 2.0]
    cat = {"codes": [0, 1, 2, 0, 1, 2], "labels": ["a", "b", "c"], "reference": None}
    out = []

    def col(t, v):
        return {"type": t, "values": v}

    def add(name, outcome, preds, functional="auto", scale="schema", cond=None):
        spec = {"name": name, "outcome": outcome, "predictors": preds, "conditioning": cond or [],
                "functional": functional, "scale": scale}
        spec["expect"] = expected_selection(spec)
        out.append(spec)

    add("zero_one_continuous_column_is_binary", col("continuous", zo), [col("continuous", num)])
    add("negative_zero_counts_as_zero", col("continuous", [-0.0, 1.0, 1.0, -0.0, 1.0, 1.0]), [col("continuous", num)])
    add("only_zeros_stay_continuous", col("continuous", [0.0] * 6), [col("continuous", num)])
    add("unspecified_counts_are_counts", col("unspecified", cnt), [col("continuous", num)])
    add("count_type_with_fraction_is_continuous", col("count", [0.0, 1.5, 1.0, 2.0, 5.0, 2.0]), [col("continuous", num)])
    add("continuous_type_integers_stay_continuous", col("continuous", cnt), [col("continuous", num)])
    add("count_type_counts", col("count", cnt), [col("continuous", num)])
    add("schema_binary_outcome", col("binary", zo), [col("continuous", num)])
    add("categorical_outcome_refused", col("categorical", cat), [col("continuous", num)])
    add("ordinal_outcome_refused", col("ordinal", cat), [col("continuous", num)])
    for sc in ("continuous", "binary", "count"):
        add("declared_scale_" + sc, col("continuous", num), [col("continuous", num[::-1])], scale=sc)
    add("no_predictor_is_the_marginal_law", col("continuous", num), [])
    add("all_discrete_predictors_give_strata", col("continuous", num), [col("binary", zo), col("categorical", cat)])
    add("ordinal_predictor_is_discrete", col("continuous", num), [col("ordinal", cat)])
    add("one_numeric_predictor_gives_regression", col("continuous", num), [col("binary", zo), col("continuous", num)])
    add("numeric_conditioning_gives_regression", col("continuous", num), [col("binary", zo)], cond=[col("continuous", num)])
    add("discrete_conditioning_keeps_strata", col("binary", zo), [col("binary", zo[::-1])], cond=[col("categorical", cat)])
    add("conditional_binary_is_probability", col("binary", zo), [col("continuous", num)], functional="conditional")
    add("coefficients_count_is_poisson", col("count", cnt), [col("binary", zo)], functional="coefficients")
    add("coefficients_binary_is_logistic", col("binary", zo), [col("binary", zo[::-1])], functional="coefficients")
    add("pearson_numeric", col("continuous", num), [col("continuous", num[::-1])], functional="pearson")
    add("distance_numeric", col("continuous", num), [col("continuous", num[::-1])], functional="distance")
    add("pearson_categorical_predictor_refused", col("continuous", num), [col("categorical", cat)], functional="pearson")
    add("distance_ordinal_predictor_refused", col("continuous", num), [col("ordinal", cat)], functional="distance")
    add("pearson_two_predictors_refused", col("continuous", num), [col("continuous", num), col("continuous", zo)], functional="pearson")
    add("pearson_conditioning_refused", col("continuous", num), [col("continuous", num[::-1])], functional="pearson", cond=[col("continuous", zo)])
    add("predictive_continuous", col("continuous", num), [col("continuous", num[::-1])], functional="predictive")
    add("predictive_count", col("count", cnt), [col("continuous", num)], functional="predictive")
    add("predictive_binary", col("binary", zo), [col("continuous", num)], functional="predictive")
    add("coefficients_without_predictor_refused", col("continuous", num), [], functional="coefficients")
    add("predictive_without_predictor_refused", col("continuous", num), [], functional="predictive")
    return out


def expected_selection(spec):
    o = spec["outcome"]
    sc = spec["scale"]
    ok = {"continuous": "Continuous", "binary": "Binary", "count": "Count"}
    if sc != "schema":
        kind, source = ok[sc], "declared"
    elif o["type"] == "binary":
        kind, source = "Binary", "schema_binary"
    elif o["type"] in ("categorical", "ordinal", "vector"):
        return {"refused": True}
    else:
        vals = o["values"]
        ind = lambda v: v == 0.0 or v == 1.0
        if all(ind(v) for v in vals) and any(v == 0.0 for v in vals) and any(v == 1.0 for v in vals):
            kind, source = "Binary", "observed_zero_one"
        elif o["type"] in ("count", "unspecified") and all(v >= 0 and v == round(v) for v in vals):
            kind, source = "Count", "observed_count"
        else:
            kind, source = "Continuous", "default_continuous"
    f = spec["functional"]
    preds, cond = spec["predictors"], spec["conditioning"]
    if f in ("pearson", "distance"):
        if len(preds) != 1 or cond:
            return {"refused": True}
        if preds[0]["type"] in ("categorical", "ordinal"):
            return {"refused": True}
    if f in ("coefficients", "predictive") and not preds:
        return {"refused": True}
    discrete = all(c["type"] in ("binary", "categorical", "ordinal") for c in preds + cond)
    strata = "StratifiedProbability" if kind == "Binary" else "StratifiedMean"
    reg = {"Continuous": "LinearRegression", "Binary": "LogisticRegression", "Count": "PoissonRegression"}[kind]
    pred = {"Continuous": "PredictiveLinear", "Binary": "PredictiveLogistic", "Count": "PredictivePoisson"}[kind]
    if f == "auto":
        analysis = strata if (not preds or discrete) else reg
    elif f == "conditional":
        analysis = strata
    elif f == "coefficients":
        analysis = reg
    elif f == "pearson":
        analysis = "PearsonCorrelation"
    elif f == "distance":
        analysis = "DistanceCorrelation"
    else:
        analysis = pred
    return {"analysis": analysis, "kind": kind, "source": source}


def is_conjugate(case):
    return (case["family"] == "continuous" and case["modality"] == "rows" and not case["weight"]
            and case["kind"] in ("model", "strata"))


def clean(case):
    cols = case["columns"]
    n = len(cases_outcome(case))
    used = [case["outcome"]] + case["predictors"] + case["conditioning"] + ([case["weight"]] if case["weight"] else [])
    keep = [r for r in range(n) if all(not (not isinstance(cols[u], dict) and cols[u][r] is None) for u in used)]
    if len(keep) == n:
        return case, 0
    out = json.loads(json.dumps(case))
    for k, v in cols.items():
        if isinstance(v, dict):
            out["columns"][k]["codes"] = [v["codes"][r] for r in keep]
        else:
            out["columns"][k] = [v[r] for r in keep]
    return out, n - len(keep)


def main():
    stage = sys.argv[1]
    cases = make_cases()
    resampling = make_resampling()
    if stage == "cases":
        print(json.dumps({"cases": cases, "resampling": resampling, "selection": selection_cases()}))
        return
    streams = json.load(open(sys.argv[2]))
    out_cases = []
    for orig in cases:
        c, dropped = clean(orig)
        e = {"estimate": statistic(c, case_weights(c)), "se": analytic(c), "rows_dropped": dropped}
        orig = dict(orig)
        orig["expect"] = e
        out_cases.append(orig)
    out_res = []
    for c in resampling:
        c, _ = clean(c)
        s = streams[c["name"]]
        e = {"estimate": statistic(c, case_weights(c)), "se": analytic(c)}
        if "boot" in c:
            e["bootstrap"] = bootstrap_expect(c, s)
            c["streams"] = {"indexes": s["indexes"], "block_length": s.get("block_length")}
        if "bayes" in c:
            if is_conjugate(c):
                e["posterior_method"] = "conjugate_normal_inverse_gamma"
                e["posterior"] = conjugate_expect(c, c["bayes"]["prior_scale"])
            else:
                e["posterior_method"] = "bayesian_bootstrap"
                e["posterior"] = bayes_boot_expect(c, s)
                c["streams"] = {"weights": s["weights"], "block_length": s.get("block_length")}
        if "folds" in c:
            n = len(cases_outcome(c))
            if c["modality"] == "series":
                assignment = [r * c["folds"] // n for r in range(n)]
            else:
                assignment = s["assignment"]
            e["prediction"] = predict_expect(c, assignment, c["folds"])
            c["streams"] = {"assignment": assignment}
        c["expect"] = e
        out_res.append(c)
    print(json.dumps({"cases": out_cases, "resampling": out_res, "selection": selection_cases()}))


if __name__ == "__main__":
    main()
