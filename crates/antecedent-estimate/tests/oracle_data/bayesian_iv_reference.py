# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Joint fixed-loading Bayesian IV posterior of the structural effect, in 40-digit arithmetic.

Regenerate with:  uv run --script bayesian_iv_reference.py > bayesian_iv_reference.json

Model: T = a + pi'Z + g'W + v, Y = mu + beta T + d'W + v + e, v ~ N(0, vt) and e ~ N(0, vy)
independent, independent N(0, tau^2) priors on every coefficient.  Because v = T - a - pi'Z - g'W,
the density of (T, Y) factorises as N(T; a + pi'Z + g'W, vt) * N(Y; mu + beta T + d'W + (T - a -
pi'Z - g'W), vy) with unit Jacobian, so the log posterior is quadratic in theta = (a, pi, g, mu,
d, beta).  It is written here directly from that factorisation, differentiated numerically (the
Hessian by central differences of the log density, exact for a quadratic up to rounding in 40
digits), and the posterior mean is the single Newton step m = -H^-1 grad(0).  This route shares no
design-matrix algebra with the product, which stacks the two equations as weighted regressions.
The posterior of beta is Normal(m_beta, (-H^-1)_beta,beta) and the 95% interval is
m_beta -/+ 1.959963984540054 sd.  The weak-instrument screen is the partial first-stage F of the
instrument block given the covariates, from two OLS fits of centred T.
Repository-authored (MIT OR Apache-2.0).
"""
import json
import mpmath as mp

mp.mp.dps = 40
Z975 = mp.mpf("1.959963984540054235524594430520551527955931")


def data(n, k, q):
    z = [[(((i * (3 + 2 * j) + 5 * j) % 17) - 8) / 8.0 for i in range(n)] for j in range(k)]
    w = [[(((i * (7 + j) + 2) % 11) - 5) / 5.0 for i in range(n)] for j in range(q)]
    t, y = [], []
    for i in range(n):
        v = (((i * 5 + 1) % 13) - 6) / 6.0 * 0.5
        e = (((i * 11 + 4) % 9) - 4) / 4.0 * 0.4
        ti = 0.3 + sum((0.8 - 0.3 * j) * z[j][i] for j in range(k)) + sum(0.4 * w[j][i] for j in range(q)) + v
        t.append(ti)
        y.append(1.0 + 1.5 * ti + sum(0.6 * w[j][i] for j in range(q)) + v + e)
    return z, w, t, y


def ols_rss(cols, y):
    n = len(y)
    if not cols:
        return sum(v * v for v in y)
    X = mp.matrix([[c[i] for c in cols] for i in range(n)])
    Y = mp.matrix(y)
    b = mp.lu_solve(X.T * X, X.T * Y)
    r = Y - X * b
    return (r.T * r)[0]


def centred(c):
    m = sum(c) / len(c)
    return [v - m for v in c]


def reference(z, w, t, y, vt, vy, tau):
    n, k, q = len(t), len(z), len(w)
    Z = [[mp.mpf(v) for v in c] for c in z]
    W = [[mp.mpf(v) for v in c] for c in w]
    T = [mp.mpf(v) for v in t]
    Y = [mp.mpf(v) for v in y]
    p = 3 + k + 2 * q

    def logpost(th):
        a, pi, g = th[0], th[1:1 + k], th[1 + k:1 + k + q]
        mu, d, beta = th[1 + k + q], th[2 + k + q:2 + k + 2 * q], th[-1]
        total = -sum(x * x for x in th) / (2 * mp.mpf(tau) ** 2)
        for i in range(n):
            mean_t = a + sum(pi[j] * Z[j][i] for j in range(k)) + sum(g[j] * W[j][i] for j in range(q))
            mean_y = mu + beta * T[i] + sum(d[j] * W[j][i] for j in range(q)) + (T[i] - mean_t)
            total -= (T[i] - mean_t) ** 2 / (2 * mp.mpf(vt)) + (Y[i] - mean_y) ** 2 / (2 * mp.mpf(vy))
        return total

    h = mp.mpf(10) ** -8
    zero = [mp.mpf(0)] * p

    def shifted(i, s, j=None, s2=0):
        th = list(zero)
        th[i] += s
        if j is not None:
            th[j] += s2
        return th

    f0 = logpost(zero)
    grad = mp.matrix([(logpost(shifted(i, h)) - logpost(shifted(i, -h))) / (2 * h) for i in range(p)])
    hess = mp.zeros(p, p)
    for i in range(p):
        hess[i, i] = (logpost(shifted(i, h)) - 2 * f0 + logpost(shifted(i, -h))) / h ** 2
        for j in range(i):
            val = (logpost(shifted(i, h, j, h)) - logpost(shifted(i, h, j, -h))
                   - logpost(shifted(i, -h, j, h)) + logpost(shifted(i, -h, j, -h))) / (4 * h * h)
            hess[i, j] = hess[j, i] = val
    cov = (-hess) ** -1
    m = cov * grad
    sd = mp.sqrt(cov[p - 1, p - 1])
    restricted = [centred(c) for c in w]
    unrestricted = restricted + [centred(c) for c in z]
    ct = centred(t)
    rss_r, rss_u = ols_rss(restricted, [mp.mpf(v) for v in ct]), ols_rss(unrestricted, [mp.mpf(v) for v in ct])
    # restricted/unrestricted regressors must be mpf for the exact solve
    f = ((rss_r - rss_u) / k) / (rss_u / (n - 1 - k - q))
    return dict(mean=float(m[p - 1]), sd=float(sd), lower=float(m[p - 1] - Z975 * sd),
                upper=float(m[p - 1] + Z975 * sd), first_stage_f=float(f))


cases = []
for name, n, k, q, vt, vy, tau in (
    ("single_instrument", 60, 1, 0, 0.25, 0.16, 5.0),
    ("two_instruments_two_covariates", 80, 2, 2, 0.3, 0.2, 3.0),
    ("tight_prior", 60, 1, 1, 0.25, 0.16, 0.7),
):
    z, w, t, y = data(n, k, q)
    cases.append(dict(name=name, n=n, instruments=z, covariates=w, treatment=t, outcome=y,
                      treatment_variance=vt, outcome_variance=vy, prior_sd=tau,
                      **reference(z, w, t, y, vt, vy, tau)))
print(json.dumps(cases, separators=(",", ":")))
