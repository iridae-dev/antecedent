# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath"]
# ///
"""Finite mixture of Gaussian linear regressions: maximum likelihood by independent EM at 40 digits.

Regenerate with:  uv run --script latent_class_reference.py > latent_class_reference.json

Model: y | a, x, class k ~ N(alpha_k + tau_k a + gamma_k' x, s2_k), P(class k) = pi_k.
This script does not reuse the product's initialisation: it starts EM from random hard labellings
drawn from its own generator (24 starts, float arithmetic), keeps the start with the highest
observed-data log-likelihood, and polishes it with EM iterations carried out at 40 digits until the
parameters stop moving (1e-30).  The reference solution is that stationary point.  Classes are put
in the canonical order (ascending tau, then alpha, then weight).  The log-likelihood
sum_i log sum_k pi_k N(y_i; mean_ik, s2_k), the responsibilities, the BIC with
K (d + 1) + K - 1 free parameters (d = 2 + number of covariates), the effective class sizes,
the separation (mean largest responsibility) and the mixture-average effect sum_k pi_k tau_k are
evaluated at it.  The global maximum is not proved: the data are well separated, the best of 24
independent starts is kept, and the other starts' log-likelihoods are recorded.
Values are written as nearest binary64.
Repository-authored (MIT OR Apache-2.0).
"""
import json
import math
from mpmath import mp, mpf, exp, log, pi as MPI

mp.dps = 40
_state = 0x510E527FADE682D1


def u01():
    global _state
    _state = (_state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
    return ((_state >> 11) + 0.5) / float(1 << 53)


def normal():
    return math.sqrt(-2 * math.log(u01())) * math.cos(2 * math.pi * u01())


def solve(A, b):
    n = len(A)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [M[i][n] / M[i][i] for i in range(n)]


def m_step(Z, y, R, one):
    n, d, K = len(y), len(Z[0]), len(R)
    pis, betas, s2s = [], [], []
    for k in range(K):
        w = R[k]
        A = [[sum(w[i] * Z[i][a] * Z[i][b] for i in range(n)) for b in range(d)] for a in range(d)]
        r = [sum(w[i] * Z[i][a] * y[i] for i in range(n)) for a in range(d)]
        beta = solve(A, r)
        nk = sum(w)
        sse = sum(w[i] * (y[i] - sum(beta[a] * Z[i][a] for a in range(d))) ** 2 for i in range(n))
        pis.append(nk / n)
        betas.append(beta)
        s2s.append(sse / nk)
    return pis, betas, s2s


def e_step(Z, y, params, exp_, log_, pi_):
    pis, betas, s2s = params
    n, d, K = len(y), len(Z[0]), len(pis)
    R = [[None] * n for _ in range(K)]
    ll = 0
    for i in range(n):
        terms = []
        for k in range(K):
            res = y[i] - sum(betas[k][a] * Z[i][a] for a in range(d))
            terms.append(log_(pis[k]) - (log_(2 * pi_ * s2s[k]) + res * res / s2s[k]) / 2)
        m = max(terms)
        tot = sum(exp_(t - m) for t in terms)
        lse = m + log_(tot)
        for k in range(K):
            R[k][i] = exp_(terms[k] - lse)
        ll += lse
    return R, ll


def fit(case):
    n = len(case["y"])
    K = case["classes"]
    Zf = [[1.0, case["treatment"][i]] + [c[i] for c in case["covariates"]] for i in range(n)]
    Zm = [[mpf(v) for v in row] for row in Zf]
    yf = case["y"]
    ym = [mpf(v) for v in yf]
    starts = []
    for s in range(24):
        labels = [int(u01() * K) % K for _ in range(n)]
        R = [[1.0 if labels[i] == k else 0.0 for i in range(n)] for k in range(K)]
        try:
            params = m_step(Zf, yf, R, 1.0)
            ll = -1e300
            for _ in range(400):
                R, ll = e_step(Zf, yf, params, math.exp, math.log, math.pi)
                params = m_step(Zf, yf, R, 1.0)
            if min(params[0]) > 0.03:
                starts.append((ll, params))
        except (ZeroDivisionError, ValueError, OverflowError):
            continue
    starts.sort(key=lambda t: -t[0])
    best = starts[0][1]
    # polish at 40 digits
    params = tuple([mpf(v) for v in best[0]]), tuple([[mpf(v) for v in b] for b in best[1]]), tuple(mpf(v) for v in best[2])
    for _ in range(400):
        R, ll = e_step(Zm, ym, params, exp, log, MPI)
        new = m_step(Zm, ym, R, mpf(1))
        delta = max(abs(a - b) for pa, pb in zip(new[1], params[1]) for a, b in zip(pa, pb))
        params = (tuple(new[0]), tuple(new[1]), tuple(new[2]))
        if delta < mpf(10) ** -32:
            break
    R, ll = e_step(Zm, ym, params, exp, log, MPI)
    order = sorted(range(K), key=lambda k: (params[1][k][1], params[1][k][0], params[0][k]))
    pis = [params[0][k] for k in order]
    betas = [params[1][k] for k in order]
    s2s = [params[2][k] for k in order]
    resp = [[float(R[k][i]) for k in order] for i in range(n)]
    d = len(Zf[0])
    return {
        "weights": [float(v) for v in pis],
        "beta": [[float(v) for v in b] for b in betas],
        "sigma2": [float(v) for v in s2s],
        "log_likelihood": float(ll),
        "bic": float(-2 * ll + (K * (d + 1) + K - 1) * log(mpf(n))),
        "effective_n": [float(sum(R[k])) for k in order],
        "separation": float(sum(max(R[k][i] for k in range(K)) for i in range(n)) / n),
        "mixture_average_effect": float(sum(p * b[1] for p, b in zip(pis, betas))),
        "responsibilities": resp,
        "start_log_likelihoods": [t[0] for t in starts],
    }


def make(name, n, K, weights, alphas, taus, gammas, sigmas, covs, binary, y_off=0.0, x_off=0.0):
    xs = [[x_off + normal() for _ in range(n)] for _ in range(covs)]
    a = [float(u01() < 0.5) if binary else normal() for _ in range(n)]
    ys = []
    for i in range(n):
        r, acc, k = u01(), 0.0, K - 1
        for j in range(K):
            acc += weights[j]
            if r < acc:
                k = j
                break
        mean = y_off + alphas[k] + taus[k] * a[i] + sum(gammas[k][c] * (xs[c][i] - x_off) for c in range(covs))
        ys.append(mean + sigmas[k] * normal())
    case = {"name": name, "classes": K, "y": ys, "treatment": a, "covariates": xs, "seed": 7}
    case["expected"] = fit(case)
    return case


cases = [
    make("two classes, binary treatment, one covariate", 200, 2, [0.5, 0.5], [0.0, 8.0], [-2.0, 3.0],
         [[0.5], [-0.5]], [0.5, 0.6], 1, True),
    make("three classes, continuous treatment, two covariates", 300, 3, [0.3, 0.4, 0.3], [-8.0, 0.0, 8.0],
         [-1.5, 0.5, 2.5], [[0.4, 0.0], [0.0, 0.5], [-0.3, 0.2]], [0.5, 0.5, 0.5], 2, False),
    make("unbalanced weights, outcome offset 1e4, covariate offset 100", 240, 2, [0.2, 0.8], [0.0, 9.0],
         [1.0, -2.0], [[0.3], [0.3]], [0.5, 0.5], 1, True, 1e4, 100.0),
    make("no covariates, continuous treatment", 200, 2, [0.4, 0.6], [0.0, 7.0], [2.0, -1.0], [[], []], [0.5, 0.5], 0, False),
]
print(json.dumps({"cases": cases}, indent=1))
