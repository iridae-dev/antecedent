# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Exact GLM g-computation contrast and delta-method standard error, in 40-digit arithmetic.

Regenerate with:  uv run --script glm_adjustment_reference.py > glm_adjustment_reference.json

Main-effects design X = (1, t, z1, z2), outcome family logit, probit or Poisson-log.  The maximum
likelihood coefficients are found by Fisher scoring to a 1e-35 fixed point in 40-digit arithmetic
(beta <- beta + (X'WX)^-1 X'u, with working weight w = m'(eta)^2 / V(mu) and score weight
u_i = (y_i - mu_i) m'(eta_i) / V(mu_i)).  The reported effect is the observed-row g-computation
contrast  (1/n) sum_i [m(x_i(active)'beta) - m(x_i(control)'beta)],  and its standard error is the
delta method  sqrt(g' (X'WX)^-1 g)  with
g = (1/n) sum_i [m'(eta_i^1) x_i(active) - m'(eta_i^0) x_i(control)]  and the expected (Fisher)
information at the maximum likelihood estimate.  Repository-authored (MIT OR Apache-2.0).
"""
import json
import mpmath as mp

mp.mp.dps = 40


def logit(eta):
    return 1 / (1 + mp.exp(-eta))


def logit_d(eta):
    m = logit(eta)
    return m * (1 - m)


def probit(eta):
    return mp.ncdf(eta)


def probit_d(eta):
    return mp.npdf(eta)


FAMILIES = {
    "logit": (logit, logit_d, lambda mu: mu * (1 - mu)),
    "probit": (probit, probit_d, lambda mu: mu * (1 - mu)),
    "poisson": (mp.exp, mp.exp, lambda mu: mu),
}


def data(n, family):
    z1 = [(((i * 7 + 3) % 19) - 9) / 9.0 for i in range(n)]
    z2 = [(((i * 5 + 1) % 13) - 6) / 6.0 for i in range(n)]
    t = [1.0 if (0.5 * z1[i] - 0.3 * z2[i] + (((i * 11) % 7) - 3) / 7.0) > 0 else 0.0 for i in range(n)]
    y = []
    for i in range(n):
        eta = -0.3 + 0.8 * t[i] + 0.7 * z1[i] - 0.4 * z2[i]
        if family == "poisson":
            lam = float(mp.exp(0.2 + 0.5 * t[i] + 0.4 * z1[i] - 0.3 * z2[i]))
            # deterministic count with mean close to lam: floor of lam plus a pattern in {0, 1}
            y.append(float(int(lam) + (1 if ((i * 13 + 5) % 10) / 10.0 < lam - int(lam) else 0)))
        else:
            p = float(logit(eta)) if family == "logit" else float(probit(eta / 1.7))
            y.append(1.0 if ((i * 37 + 11) % 100) / 100.0 < p else 0.0)
    return z1, z2, t, y


def reference(family, n):
    m, dm, var = FAMILIES[family]
    z1, z2, t, y = data(n, family)
    X = mp.matrix([[1, mp.mpf(t[i]), mp.mpf(z1[i]), mp.mpf(z2[i])] for i in range(n)])
    Y = mp.matrix([mp.mpf(v) for v in y])
    beta = mp.zeros(4, 1)
    if family == "poisson":
        beta[0] = mp.log(sum(Y) / n)
    for _ in range(200):
        eta = X * beta
        mu = [m(eta[i]) for i in range(n)]
        w = [dm(eta[i]) ** 2 / var(mu[i]) for i in range(n)]
        u = mp.matrix([(Y[i] - mu[i]) * dm(eta[i]) / var(mu[i]) for i in range(n)])
        info = X.T * mp.diag(w) * X
        step = info ** -1 * (X.T * u)
        beta += step
        if max(abs(step[k]) for k in range(4)) < mp.mpf(10) ** -35:
            break
    eta = X * beta
    mu = [m(eta[i]) for i in range(n)]
    w = [dm(eta[i]) ** 2 / var(mu[i]) for i in range(n)]
    info = X.T * mp.diag(w) * X
    effect = mp.mpf(0)
    grad = mp.zeros(4, 1)
    for i in range(n):
        x1 = mp.matrix([1, 1, X[i, 2], X[i, 3]])
        x0 = mp.matrix([1, 0, X[i, 2], X[i, 3]])
        e1, e0 = (x1.T * beta)[0], (x0.T * beta)[0]
        effect += (m(e1) - m(e0)) / n
        grad += (dm(e1) * x1 - dm(e0) * x0) / n
    se = mp.sqrt((grad.T * info ** -1 * grad)[0])
    return dict(family=family, treatment=t, z1=z1, z2=z2, outcome=y, ate=float(effect), se=float(se),
                treatment_coefficient=float(beta[1]))


print(json.dumps([reference("logit", 160), reference("probit", 160), reference("poisson", 140)], separators=(",", ":")))
