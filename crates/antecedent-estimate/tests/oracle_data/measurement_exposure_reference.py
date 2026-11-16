# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath"]
# ///
"""Exposure misclassification, soft exposures, covariate-dependent detection and latent-class
repeat codings against independent multiprecision and exact rational computations.

Regenerate with:  uv run --script measurement_exposure_reference.py > measurement_exposure_reference.json

* exposure: the corrected shares and log odds ratio are the closed form at 50 digits; the
  variance is the delta method with every derivative taken by `mpmath.diff` of the corrected
  log odds ratio as an explicit function of the recorded shares and the rates, not by the
  library's analytic partials. Rates from validation counts carry binomial variances.
* soft exposure: exact rational least squares on (1, q, z); HC0 sandwich; the exposure
  uncertainty term beta^2 mean(q(1-q)) (M^-1)_11 / n; Rubin's rules over three exact 0/1 draws.
* reliability: exact bin means and gaps.
* detection: the logistic and table detection functions, the prevalence A / G with its Bernoulli
  variance, the gradient of the estimator in the logistic parameters by `mpmath.diff`, and the
  sharp bounds of a monotone envelope.
* latent class: EM to 1e-30 in 40-digit arithmetic on integer pattern counts that are not
  exactly a latent-class law, from a start that differs from the library's.
Repository-authored (MIT OR Apache-2.0).
"""

import itertools
import json
from fractions import Fraction as F

from mpmath import diff, exp, log, mp, mpf, sqrt

mp.dps = 50


def num(x):
    return float(x)


# ---------------------------------------------------------------- exposure


def log_or(p1, p0, se1, sp1, se0, sp0):
    pi1 = (p1 + sp1 - 1) / (se1 + sp1 - 1)
    pi0 = (p0 + sp0 - 1) / (se0 + sp0 - 1)
    return log(pi1 / (1 - pi1)) - log(pi0 / (1 - pi0))


def exposure_case(name, a1, n1, a0, n0, rates, variances, shared):
    """`rates` = (se1, sp1, se0, sp0) for cases then controls; `variances` the same order.

    With `shared` the case rates are applied to both outcome levels as ONE estimate."""
    p1, p0 = mpf(a1) / n1, mpf(a0) / n0
    se1, sp1, se0, sp0 = (mpf(r) for r in rates)
    if shared:
        se0, sp0 = se1, sp1
    f = log_or
    base = [p1, p0, se1, sp1, se0, sp0]
    value = f(*base)
    d_p1 = diff(lambda x: f(x, p0, se1, sp1, se0, sp0), p1)
    d_p0 = diff(lambda x: f(p1, x, se1, sp1, se0, sp0), p0)
    sampling = d_p1**2 * p1 * (1 - p1) / n1 + d_p0**2 * p0 * (1 - p0) / n0
    if shared:
        d_se = diff(lambda x: f(p1, p0, x, sp1, x, sp1), se1)
        d_sp = diff(lambda x: f(p1, p0, se1, x, se1, x), sp1)
        rate_var = d_se**2 * mpf(variances[0]) + d_sp**2 * mpf(variances[1])
    else:
        grads = [
            diff(lambda x: f(p1, p0, x, sp1, se0, sp0), se1),
            diff(lambda x: f(p1, p0, se1, x, se0, sp0), sp1),
            diff(lambda x: f(p1, p0, se1, sp1, x, sp0), se0),
            diff(lambda x: f(p1, p0, se1, sp1, se0, x), sp0),
        ]
        rate_var = sum(g**2 * mpf(v) for g, v in zip(grads, variances))
    return {
        "name": name,
        "counts": [a1, n1, a0, n0],
        "rates": [num(r) for r in rates],
        "rate_variances": [num(v) for v in variances],
        "shared": shared,
        "log_odds_ratio": num(value),
        "variance_sampling": num(sampling),
        "variance_rates": num(rate_var),
    }


def binomial_var(count, total):
    p = mpf(count) / total
    return p * (1 - p) / total


def exposure_cases():
    out = []
    out.append(
        exposure_case(
            "differential_declared",
            480,
            1000,
            180,
            1000,
            (0.9, 0.8, 0.7, 0.95),
            (4e-4, 9e-4, 6e-4, 2e-4),
            False,
        )
    )
    out.append(
        exposure_case(
            "shared_declared",
            480,
            1000,
            180,
            1000,
            (0.8, 0.9, 0.8, 0.9),
            (4e-4, 9e-4, 0, 0),
            True,
        )
    )
    # Rates estimated from validation samples, one per outcome level.
    val_controls = (70, 30, 190, 10)  # tp fn tn fp
    val_cases = (90, 10, 80, 20)
    se0 = mpf(70) / 100
    sp0 = mpf(190) / 200
    se1 = mpf(90) / 100
    sp1 = mpf(80) / 100
    vars_ = (
        binomial_var(90, 100),
        binomial_var(80, 100),
        binomial_var(70, 100),
        binomial_var(190, 200),
    )
    case = exposure_case(
        "differential_from_validation",
        420,
        900,
        210,
        1100,
        (se1, sp1, se0, sp0),
        vars_,
        False,
    )
    case["validation_controls"] = list(val_controls)
    case["validation_cases"] = list(val_cases)
    out.append(case)
    return out


# ---------------------------------------------------------------- soft exposure


def solve(a, b):
    n = len(a)
    m = [row[:] + [bv] for row, bv in zip(a, b)]
    for c in range(n):
        piv = next(r for r in range(c, n) if m[r][c] != 0)
        m[c], m[piv] = m[piv], m[c]
        d = m[c][c]
        m[c] = [v / d for v in m[c]]
        for r in range(n):
            if r != c and m[r][c] != 0:
                f = m[r][c]
                m[r] = [v - f * w for v, w in zip(m[r], m[c])]
    return [row[n] for row in m]


def invert(a):
    n = len(a)
    m = [row[:] + [F(int(i == j)) for j in range(n)] for i, row in enumerate(a)]
    for c in range(n):
        piv = next(r for r in range(c, n) if m[r][c] != 0)
        m[c], m[piv] = m[piv], m[c]
        d = m[c][c]
        m[c] = [v / d for v in m[c]]
        for r in range(n):
            if r != c and m[r][c] != 0:
                f = m[r][c]
                m[r] = [v - f * w for v, w in zip(m[r], m[c])]
    return [row[n:] for row in m]


def ols_hc0(y, columns):
    n = len(y)
    d = len(columns)
    xtx = [
        [sum(columns[a][i] * columns[b][i] for i in range(n)) for b in range(d)]
        for a in range(d)
    ]
    xty = [sum(columns[a][i] * y[i] for i in range(n)) for a in range(d)]
    beta = solve(xtx, xty)
    resid = [y[i] - sum(beta[a] * columns[a][i] for a in range(d)) for i in range(n)]
    bread = invert(xtx)
    meat = [
        [
            sum(columns[a][i] * columns[b][i] * resid[i] ** 2 for i in range(n))
            for b in range(d)
        ]
        for a in range(d)
    ]
    tmp = [
        [sum(bread[a][k] * meat[k][b] for k in range(d)) for b in range(d)]
        for a in range(d)
    ]
    cov = [
        [sum(tmp[a][k] * bread[k][b] for k in range(d)) for b in range(d)]
        for a in range(d)
    ]
    return beta, cov, bread


def soft_case():
    n = 12
    q = [F(k, 20) for k in (2, 4, 6, 8, 10, 12, 14, 16, 18, 3, 9, 15)]
    z = [F(v) for v in (1, -2, 0, 3, -1, 2, 0, -3, 1, 2, -2, 1)]
    y = [F(v, 10) for v in (49, -12, 37, 6, 50, 54, 8, 47, 41, 31, 3, 80)]
    ones = [F(1)] * n
    beta, cov, bread = ols_hc0(y, [ones, q, z])
    mean_var = sum(v * (1 - v) for v in q) / n
    # moment matrix M = X'X / n so (M^-1)_11 = n * (X'X)^-1_11.
    inv_m_11 = bread[1][1] * n
    exposure_term = beta[1] ** 2 * mean_var * inv_m_11 / n
    total = cov[1][1]
    exposure_term = min(max(exposure_term, F(0)), total)
    draws = [
        [F(v) for v in (0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1)],
        [F(v) for v in (0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1)],
        [F(v) for v in (1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1)],
    ]
    fits = [ols_hc0(y, [ones, d, z]) for d in draws]
    s = len(draws)
    mean_beta = [sum(f[0][k] for f in fits) / s for k in range(3)]
    within = sum(f[1][1][1] for f in fits) / s
    between = sum((f[0][1] - mean_beta[1]) ** 2 for f in fits) / (s - 1)
    rubin = within + (1 + F(1, s)) * between
    return {
        "outcome": [num(v) for v in y],
        "probability": [num(v) for v in q],
        "covariate": [num(v) for v in z],
        "coefficients": [num(b) for b in beta],
        "hc0_variance": num(total),
        "exposure_term": num(exposure_term),
        "mean_posterior_variance": num(mean_var),
        "draws": [[num(v) for v in d] for d in draws],
        "rubin_coefficient": num(mean_beta[1]),
        "rubin_within": num(within),
        "rubin_between": num(between),
        "rubin_total": num(rubin),
    }


def reliability_case():
    probability = [F(1, 20), F(2, 20), F(11, 20), F(13, 20), F(19, 20)]
    codings = [[0, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 1], [1, 0, 1, 1], [1, 1, 1, 1]]
    se, sp = F(9, 10), F(4, 5)
    bins = 2
    members = {}
    for q, c in zip(probability, codings):
        share = F(sum(c), len(c))
        expected = se * q + (1 - sp) * (1 - q)
        slot = min(int(q * bins), bins - 1)
        members.setdefault(slot, []).append((q, expected, share))
    out = []
    for slot in sorted(members):
        m = members[slot]
        u = len(m)
        mean_expected = sum(v[1] for v in m) / u
        mean_observed = sum(v[2] for v in m) / u
        diffs = [v[2] - v[1] for v in m]
        mean = sum(diffs) / u
        var = sum((d - mean) ** 2 for d in diffs) / (u - 1)
        out.append(
            {
                "units": u,
                "mean_expected": num(mean_expected),
                "mean_observed": num(mean_observed),
                "gap": num(mean_observed - mean_expected),
                "standard_error": num(sqrt(mpf(var.numerator) / var.denominator / u)),
            }
        )
    return {
        "probability": [num(v) for v in probability],
        "codings": [c for c in codings],
        "reference": [num(se), num(sp)],
        "bins": bins,
        "result": out,
    }


# ---------------------------------------------------------------- detection


def expit(x):
    return 1 / (1 + exp(-x))


def logistic_case():
    per = 500
    hits = [60, 110, 190, 240]
    covs = [0, 1, 2, 3]
    cov_matrix = [[mpf("0.004"), mpf("0.001")], [mpf("0.001"), mpf("0.002")]]
    fp, fp_var = mpf("0.02"), mpf("0.0001")
    n = per * len(covs)

    def prevalence(a, b, f):
        g = sum(per * expit(a + b * c) for c in covs) - n * f
        return (sum(hits) - n * f) / g

    a0, b0 = mpf(-1), mpf("0.9")
    pi = prevalence(a0, b0, fp)
    g = sum(per * expit(a0 + b0 * c) for c in covs) - n * fp
    sampling = 0
    for (c,) in zip(covs):
        d = expit(a0 + b0 * c)
        p = pi * d + (1 - pi) * fp
        sampling += per * p * (1 - p)
    sampling /= g**2
    grad = [
        diff(lambda x: prevalence(x, b0, fp), a0),
        diff(lambda x: prevalence(a0, x, fp), b0),
    ]
    d_fp = diff(lambda x: prevalence(a0, b0, x), fp)
    params = sum(
        grad[i] * cov_matrix[i][j] * grad[j] for i in range(2) for j in range(2)
    )
    params += d_fp**2 * fp_var
    return {
        "per": per,
        "hits": hits,
        "intercept": num(a0),
        "slope": num(b0),
        "intercept_variance": num(cov_matrix[0][0]),
        "covariance": num(cov_matrix[0][1]),
        "slope_variance": num(cov_matrix[1][1]),
        "false_positive": num(fp),
        "false_positive_variance": num(fp_var),
        "prevalence": num(pi),
        "variance_sampling": num(sampling),
        "variance_parameters": num(params),
    }


def bounds_case():
    per = 1000
    hits = [120, 180, 240, 300]
    anchors = [(0, F(3, 10), F(1, 2)), (3, F(9, 10), F(1))]
    lower_d, upper_d = [], []
    for c in range(4):
        lo = max([a[1] for a in anchors if a[0] <= c], default=F(0))
        hi = min([a[2] for a in anchors if a[0] >= c], default=F(1))
        lower_d.append(lo)
        upper_d.append(hi)
    n = per * 4
    a = F(sum(hits))
    g_hi = per * sum(upper_d)
    g_lo = per * sum(lower_d)
    return {
        "per": per,
        "hits": hits,
        "anchors": [[a_[0], num(a_[1]), num(a_[2])] for a_ in anchors],
        "lower": num(a / g_hi),
        "upper": num(min(F(1), a / g_lo)),
        "mean_lower": num(g_lo / n),
        "mean_upper": num(g_hi / n),
    }


# ---------------------------------------------------------------- latent class


def latent_class_case(counts, k, c, j):
    patterns = list(itertools.product(range(c), repeat=k))
    mp.dps = 40
    total = sum(counts)
    # A start unlike the library's: stagger the classes by pattern parity.
    resp = []
    for pat in patterns:
        s = sum(pat) / mpf(k * (c - 1))
        w = [mpf(1) / (1 + (s * (j - 1) - lv) ** 2 * 2) for lv in range(j)]
        t = sum(w)
        resp.append([x / t for x in w])
    pi = theta = None
    ll = mpf(0)
    for it in range(60000):
        pi = [
            sum(counts[p] * resp[p][lv] for p in range(len(patterns))) / total
            for lv in range(j)
        ]
        theta = []
        for kk in range(k):
            row = []
            for lv in range(j):
                cnt = [mpf(0)] * c
                for p, pat in enumerate(patterns):
                    cnt[pat[kk]] += counts[p] * resp[p][lv]
                s = sum(cnt)
                row.append([x / s for x in cnt])
            theta.append(row)
        new_resp, new_ll = [], mpf(0)
        for p, pat in enumerate(patterns):
            w = []
            for lv in range(j):
                v = pi[lv]
                for kk in range(k):
                    v *= theta[kk][lv][pat[kk]]
                w.append(v)
            t = sum(w)
            new_ll += counts[p] * log(t)
            new_resp.append([x / t for x in w])
        resp = new_resp
        if abs(new_ll - ll) < mpf(10) ** -32:
            ll = new_ll
            break
        ll = new_ll
    order = sorted(
        range(j),
        key=lambda lv: sum(
            m * sum(theta[kk][lv][m] for kk in range(k)) for m in range(c)
        ),
    )
    return {
        "k": k,
        "c": c,
        "j": j,
        "counts": counts,
        "prevalence": [num(pi[lv]) for lv in order],
        "response": [
            [[num(theta[kk][lv][m]) for m in range(c)] for lv in order]
            for kk in range(k)
        ],
        "log_likelihood": num(ll),
    }


# ---------------------------------------------------------------- seeded multiple imputation

MASK = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15


class SplitMix:
    """The library's SplitMix64 stream, written from its definition."""

    def __init__(self, seed):
        self.state = (seed ^ GOLDEN) & MASK

    def next_f64(self):
        self.state = (self.state + GOLDEN) & MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        z ^= z >> 31
        return (z >> 11) * (1.0 / (1 << 53))


def gauss_solve(a, b):
    """Gaussian elimination with partial pivoting in binary64."""
    n = len(a)
    m = [row[:] + [bv] for row, bv in zip(a, b)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(m[r][c]))
        m[c], m[piv] = m[piv], m[c]
        d = m[c][c]
        m[c] = [v / d for v in m[c]]
        for r in range(n):
            if r != c:
                f = m[r][c]
                m[r] = [v - f * w for v, w in zip(m[r], m[c])]
    return [row[n] for row in m]


def mixture_fit(y, q, z, rows, start):
    """EM for Y ~ N(b0 + b1 X + b2 Z, s2) with latent X ~ Bernoulli(q_i) over `rows`."""
    d = 3

    def design(i, x):
        return [1.0, x, z[i]]

    def m_step(resp):
        a = [[0.0] * d for _ in range(d)]
        b = [0.0] * d
        for slot, i in enumerate(rows):
            r = resp[slot]
            for x, w in ((1.0, r), (0.0, 1.0 - r)):
                if w <= 0.0:
                    continue
                v = design(i, x)
                for s_ in range(d):
                    b[s_] += w * v[s_] * y[i]
                    for t in range(d):
                        a[s_][t] += w * v[s_] * v[t]
        beta = gauss_solve(a, b)
        rss = 0.0
        for slot, i in enumerate(rows):
            r = resp[slot]
            for x, w in ((1.0, r), (0.0, 1.0 - r)):
                if w <= 0.0:
                    continue
                mu = sum(vv * cc for vv, cc in zip(design(i, x), beta))
                rss += w * (y[i] - mu) ** 2
        return beta, rss / len(rows)

    def responsibility(qi, yi, mu0, mu1, s2):
        if qi <= 0.0:
            return 0.0
        if qi >= 1.0:
            return 1.0
        odds = log(mpf(qi) / (1 - qi)) + ((yi - mu0) ** 2 - (yi - mu1) ** 2) / (
            2.0 * s2
        )
        return float(1 / (1 + exp(-odds)))

    def resp_of(beta, s2, i):
        mu0 = sum(vv * cc for vv, cc in zip(design(i, 0.0), beta))
        mu1 = sum(vv * cc for vv, cc in zip(design(i, 1.0), beta))
        return responsibility(q[i], y[i], mu0, mu1, s2)

    if start is None:
        beta, s2 = m_step([q[i] for i in rows])
    else:
        beta, s2 = start
    for _ in range(500):
        resp = [resp_of(beta, s2, i) for i in rows]
        nxt, ns2 = m_step(resp)
        change = max([abs(a - b) for a, b in zip(nxt, beta)] + [abs(ns2 - s2)])
        beta, s2 = nxt, ns2
        if change < 1e-9:
            break
    return beta, s2, resp_of


def ols_hc0_float(y, columns):
    n, d = len(y), len(columns)
    xtx = [
        [sum(columns[a][i] * columns[b][i] for i in range(n)) for b in range(d)]
        for a in range(d)
    ]
    xty = [sum(columns[a][i] * y[i] for i in range(n)) for a in range(d)]
    beta = gauss_solve(xtx, xty)
    resid = [y[i] - sum(beta[a] * columns[a][i] for a in range(d)) for i in range(n)]
    bread = [gauss_solve(xtx, [float(i == j) for i in range(d)]) for j in range(d)]
    meat = [
        [
            sum(columns[a][i] * columns[b][i] * resid[i] ** 2 for i in range(n))
            for b in range(d)
        ]
        for a in range(d)
    ]
    tmp = [
        [sum(bread[a][k] * meat[k][b] for k in range(d)) for b in range(d)]
        for a in range(d)
    ]
    cov = [
        [sum(tmp[a][k] * bread[k][b] for k in range(d)) for b in range(d)]
        for a in range(d)
    ]
    return beta, cov


def imputation_case():
    n = 60
    q = [0.05 + 0.9 * i / (n - 1) for i in range(n)]
    z = [((7 * i) % 11 - 5) / 5 for i in range(n)]
    y = [1.0 + 3.0 * q[i] + 0.5 * z[i] + ((13 * i) % 17 - 8) / 4 for i in range(n)]
    seed, draw_count = 4, 6
    full_beta, full_s2, _ = mixture_fit(y, q, z, list(range(n)), None)
    rng = SplitMix(seed)
    draws = []
    for _ in range(draw_count):
        rows = [min(int(rng.next_f64() * n), n - 1) for _ in range(n)]
        beta, s2, resp_of = mixture_fit(y, q, z, rows, (full_beta, full_s2))
        draws.append(
            [1.0 if rng.next_f64() < resp_of(beta, s2, i) else 0.0 for i in range(n)]
        )
    fits = [ols_hc0_float(y, [[1.0] * n, d, z]) for d in draws]
    mean_beta = sum(f[0][1] for f in fits) / draw_count
    within = sum(f[1][1][1] for f in fits) / draw_count
    between = sum((f[0][1] - mean_beta) ** 2 for f in fits) / (draw_count - 1)
    return {
        "outcome": y,
        "probability": q,
        "covariate": z,
        "seed": seed,
        "draw_count": draw_count,
        "full_coefficients": full_beta,
        "full_residual_variance": full_s2,
        "draws": draws,
        "coefficient": mean_beta,
        "within": within,
        "between": between,
        "total": within + (1 + 1 / draw_count) * between,
    }


def main():
    mp.dps = 50
    exposure = exposure_cases()
    soft = soft_case()
    imputation = imputation_case()
    reliability = reliability_case()
    logistic = logistic_case()
    bounds = bounds_case()
    mp.dps = 40
    binary = latent_class_case([4210, 610, 480, 410, 520, 390, 520, 1860], 3, 2, 2)
    ordinal = latent_class_case(
        [
            2900,
            330,
            90,
            300,
            420,
            150,
            90,
            120,
            130,
            280,
            330,
            200,
            330,
            900,
            540,
            120,
            330,
            330,
            90,
            150,
            300,
            120,
            340,
            420,
            130,
            330,
            1800,
        ],
        3,
        3,
        3,
    )
    print(
        json.dumps(
            {
                "exposure": exposure,
                "soft": soft,
                "imputation": imputation,
                "reliability": reliability,
                "logistic": logistic,
                "bounds": bounds,
                "latent_class": [binary, ordinal],
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
