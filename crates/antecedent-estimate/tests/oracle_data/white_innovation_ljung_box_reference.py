# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Exact Ljung-Box statistic of least-squares residuals, in 40-digit arithmetic.

Regenerate with:  uv run --script white_innovation_ljung_box_reference.py > white_innovation_ljung_box_reference.json

For a design X (n x p, an intercept first) and outcome y, the residuals are e = y - X (X'X)^-1 X'y,
solved here by Gauss-Jordan elimination on the normal equations in 40-digit arithmetic.  With
r_k = sum_{t>k} e_t e_{t-k} / sum_t e_t^2 the statistic is Q = n (n + 2) sum_{k=1..m} r_k^2 / (n - k)
and the p-value is the upper tail of a chi-square variable with m degrees of freedom,
Q(m/2, Q/2) the regularized upper incomplete gamma function.  Persistent-error and periodic-shock series are used, so
the p-values span a non-rejection and clear rejections.  The Normal-Inverse-Gamma posterior of the
treatment coefficient (prior scale 10, Inv-Gamma(1e-3, 1e-3), Student-t with mean mn and variance
bn / (an - 1) Vn) is also frozen for the untempered fit the gate licenses.
Repository-authored (MIT OR Apache-2.0).
"""
import json
import mpmath as mp

mp.mp.dps = 40


def series(n, phi, wiggle):
    state = 0.0
    noise, treatment, out = [], [], []
    for i in range(n):
        shock = wiggle * ((((i * 37 + 11) % 23) - 11) / 11.0)
        state = phi * state + shock
        noise.append(state)
        treatment.append(1.0 if (i * 5 + 3) % 7 < 3 else 0.0)
        out.append(0.5 + 1.5 * treatment[-1] + 0.2 * ((i % 9) / 9.0) + state)
    return treatment, [0.2 * ((i % 9) / 9.0) for i in range(n)], out


def ljung_box(treatment, covariate, y, lags):
    n = len(y)
    X = mp.matrix([[1, mp.mpf(treatment[i]), mp.mpf(covariate[i])] for i in range(n)])
    Y = mp.matrix([mp.mpf(v) for v in y])
    beta = mp.lu_solve(X.T * X, X.T * Y)
    e = [(Y - X * beta)[i] for i in range(n)]
    total = sum(v * v for v in e)
    q = mp.mpf(0)
    for k in range(1, lags + 1):
        r = sum(e[t] * e[t - k] for t in range(k, n)) / total
        q += r * r / (n - k)
    q *= n * (n + 2)
    p = mp.gammainc(mp.mpf(lags) / 2, q / 2, mp.inf, regularized=True)
    return float(q), float(p), nig_treatment(X, Y)


def nig_treatment(X, Y, c0=10.0):
    """Mean and sd of the treatment coefficient under b | s2 ~ N(0, s2 c0^2 I), s2 ~ Inv-Gamma(1e-3, 1e-3)."""
    n, p = X.rows, X.cols
    prec = X.T * X + mp.eye(p) / mp.mpf(c0) ** 2
    vn = prec ** -1
    mn = vn * (X.T * Y)
    an = mp.mpf("0.001") + mp.mpf(n) / 2
    bn = mp.mpf("0.001") + ((Y.T * Y)[0] - (mn.T * prec * mn)[0]) / 2
    return float(mn[1]), float(mp.sqrt(bn / (an - 1) * vn[1, 1]))


cases = []
for name, n, phi, wiggle, lags in (
    ("four_lags_persistent_error", 90, 0.7, 0.5, 4),
    ("four_lags_periodic_shocks", 90, 0.0, 0.5, 4),
    ("single_lag", 60, 0.4, 0.8, 1),
    ("twelve_lags", 120, 0.3, 0.6, 12),
):
    t, c, y = series(n, phi, wiggle)
    q, p, (mean, sd) = ljung_box(t, c, y, lags)
    cases.append(dict(name=name, lags=lags, treatment=t, covariate=c, outcome=y, q=q, p_value=p,
                      posterior_mean=mean, posterior_sd=sd))
print(json.dumps(cases, separators=(",", ":")))
