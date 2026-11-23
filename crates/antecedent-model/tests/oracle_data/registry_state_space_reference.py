# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Scalar linear-Gaussian state space: Kalman filter, smoother, EM fit and predictive density.

Regenerate with:  uv run --script registry_state_space_reference.py > registry_state_space_reference.json

Model: x_0 ~ N(x0, q), x_t = a x_{t-1} + N(0, q), y_t = x_t + N(0, r).

filter_cases   exact Fraction recursions (gain K = P^- / (P^- + r), P = (1 - K) P^-, smoother gain
               J = P a / P^-_{next}, lag covariance J P_s), each asserted equal, as rational numbers,
               to Gaussian conditioning on the joint law of the whole state path: the filtered and
               predicted moments condition on y_{<=t} and y_{<t}, the smoothed moments and the lag-one
               covariance on all of y.  The recursion therefore equals the joint law exactly; the
               f64 values written out are the roundings of those rationals.
density_cases  the one-step predictive log density sum, per row, from the exact predictive moments
               (the prequential score), including a case at scale 1e9 where the unit-free
               parametrisation matters.
em_cases       the fitted conditional state space of the registry: ordinary least squares of the
               outcome on [1 | parents] (exact rational), residuals scaled by their largest
               magnitude, then 25 EM iterations from (a, q, r, x0) = (0.8, 1, 1, y_0) whose E-step is
               the joint-Gaussian conditioning above in 60-digit arithmetic (no recursion) and whose
               M-step is the stationary point of the expected complete-data log likelihood
               (a clamped to +-0.999, q and r floored at 1e-8, the initial state's variance equal to
               q).  The clamp and floor states reached are recorded.
Repository-authored (MIT OR Apache-2.0).
"""
import json
import random
from fractions import Fraction as F

import mpmath as mp

mp.mp.dps = 60


def mpf(f):
    f = F(f)
    return mp.mpf(f.numerator) / f.denominator


# ---------------------------------------------------------------- exact rational recursions
def kalman(y, a, q, r, x0, p0):
    n = len(y)
    xf, pf, xp, pp = [], [], [], []
    x, p = x0, p0
    for t in range(n):
        xpt = x if t == 0 else a * x
        ppt = p if t == 0 else a * a * p + q
        s = ppt + r
        k = ppt / s
        x = xpt + k * (y[t] - xpt)
        p = (1 - k) * ppt
        xp.append(xpt), pp.append(ppt), xf.append(x), pf.append(p)
    xs, ps, lag = xf[:], pf[:], [F(0)] * n
    for t in range(n - 2, -1, -1):
        j = pf[t] * a / pp[t + 1]
        xs[t] = xf[t] + j * (xs[t + 1] - xp[t + 1])
        ps[t] = pf[t] + j * j * (ps[t + 1] - pp[t + 1])
        lag[t + 1] = j * ps[t + 1]
    return xf, pf, xp, pp, xs, ps, lag


def solve_exact(A, B):
    n = len(A)
    M = [A[i][:] + B[i][:] for i in range(n)]
    m = len(B[0])
    for c in range(n):
        piv = next(r for r in range(c, n) if M[r][c] != 0)
        M[c], M[piv] = M[piv], M[c]
        inv = 1 / M[c][c]
        M[c] = [v * inv for v in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [row[n:n + m] for row in M]


def joint_prior(n, a, q, x0, one):
    var = []
    v = q
    for t in range(n):
        if t > 0:
            v = a * a * v + q
        var.append(v)
    cov = [[one * 0 for _ in range(n)] for _ in range(n)]
    for s in range(n):
        for t in range(s, n):
            c = a ** (t - s) * var[s]
            cov[s][t] = c
            cov[t][s] = c
    mu = [a ** t * x0 for t in range(n)]
    return mu, cov


def condition_exact(y, a, q, r, x0, upto):
    """Mean, covariance of the whole state path given y[:upto] (rational)."""
    n = len(y)
    mu, cov = joint_prior(n, a, q, x0, F(0))
    if upto == 0:
        return mu, cov
    S = [[cov[i][j] + (r if i == j else 0) for j in range(upto)] for i in range(upto)]
    cross = [[cov[i][j] for j in range(upto)] for i in range(n)]  # n x upto
    # gain G = cross S^-1  ->  solve S G' = cross'
    Gt = solve_exact(S, [[cross[i][j] for i in range(n)] for j in range(upto)])  # upto x n
    G = [[Gt[j][i] for j in range(upto)] for i in range(n)]
    m = [mu[i] + sum(G[i][j] * (y[j] - mu[j]) for j in range(upto)) for i in range(n)]
    C = [[cov[i][k] - sum(G[i][j] * cross[k][j] for j in range(upto)) for k in range(n)] for i in range(n)]
    return m, C


def filter_case(name, a, q, r, x0, y):
    a, q, r, x0 = F(a), F(q), F(r), F(x0)
    y = [F(v) for v in y]
    n = len(y)
    xf, pf, xp, pp, xs, ps, lag = kalman(y, a, q, r, x0, q)
    for t in range(n):
        m, C = condition_exact(y, a, q, r, x0, t + 1)
        assert m[t] == xf[t] and C[t][t] == pf[t], (name, "filtered", t)
        m, C = condition_exact(y, a, q, r, x0, t)
        assert m[t] == xp[t] and C[t][t] == pp[t], (name, "predicted", t)
    m, C = condition_exact(y, a, q, r, x0, n)
    for t in range(n):
        assert m[t] == xs[t] and C[t][t] == ps[t], (name, "smoothed", t)
        if t > 0:
            assert C[t][t - 1] == lag[t], (name, "lag", t)
    fl = lambda v: [float(x) for x in v]
    return {"name": name, "a": float(a), "process_var": float(q), "obs_var": float(r), "x0": float(x0),
            "y": fl(y), "x_filtered": fl(xf), "p_filtered": fl(pf), "x_predicted": fl(xp),
            "p_predicted": fl(pp), "x_smoothed": fl(xs), "p_smoothed": fl(ps), "p_lag": fl(lag)}


def density_case(name, a, process_std, obs_std, x0, y, scale_note):
    a, ps_, os_, x0 = F(a), F(process_std), F(obs_std), F(x0)
    y = [F(v) for v in y]
    xf, pf, xp, pp, *_ = kalman(y, a, ps_ * ps_, os_ * os_, x0, ps_ * ps_)
    out = []
    for t in range(len(y)):
        s = pp[t] + os_ * os_
        e = y[t] - xp[t]
        out.append(float(-mp.log(2 * mp.pi * mpf(s)) / 2 - mpf(e * e) / (2 * mpf(s))))
    return {"name": name, "a": float(a), "process_std": float(ps_), "obs_std": float(os_),
            "initial_mean": float(x0), "y": [float(v) for v in y], "log_density": out, "note": scale_note}


# ---------------------------------------------------------------- EM through the joint law
def em(y, iters=25):
    n = len(y)
    a, q, r, x0 = mp.mpf("0.8"), mp.mpf(1), mp.mpf(1), y[0]
    clamped = floored = False
    for _ in range(iters):
        var = []
        v = q
        for t in range(n):
            if t > 0:
                v = a * a * v + q
            var.append(v)
        cov = mp.zeros(n, n)
        for s in range(n):
            for t in range(s, n):
                cov[s, t] = cov[t, s] = a ** (t - s) * var[s]
        mu = mp.matrix([a ** t * x0 for t in range(n)])
        S = cov + r * mp.eye(n)
        G = cov * mp.inverse(S)  # cov S^-1 (both symmetric)
        d = mp.matrix([y[t] - mu[t] for t in range(n)])
        m = mu + G * d
        C = cov - G * cov
        P = [C[t, t] for t in range(n)]
        lag = [mp.mpf(0)] + [C[t, t - 1] for t in range(1, n)]
        num = sum(lag[t] + m[t] * m[t - 1] for t in range(1, n))
        den = sum(P[t - 1] + m[t - 1] ** 2 for t in range(1, n))
        if den > mp.mpf("1e-12"):
            a = num / den
            if a > mp.mpf("0.999"):
                a, clamped = mp.mpf("0.999"), True
            elif a < mp.mpf("-0.999"):
                a, clamped = mp.mpf("-0.999"), True
        x0 = m[0]
        q_acc = P[0]
        for t in range(1, n):
            q_acc += P[t] + m[t] ** 2 + a * a * (P[t - 1] + m[t - 1] ** 2) - 2 * a * (lag[t] + m[t] * m[t - 1])
        q = q_acc / n
        if q < mp.mpf("1e-8"):
            q, floored = mp.mpf("1e-8"), True
        r_acc = sum(P[t] + (y[t] - m[t]) ** 2 for t in range(n))
        r = r_acc / n
        if r < mp.mpf("1e-8"):
            r, floored = mp.mpf("1e-8"), True
    return a, mp.sqrt(q), mp.sqrt(r), x0, clamped, floored


def ols(parents, y):
    n = len(y)
    D = [[F(1)] + [F(c[i]) for c in parents] for i in range(n)]
    k = len(D[0])
    A = [[sum(D[i][a_] * D[i][b] for i in range(n)) for b in range(k)] for a_ in range(k)]
    rhs = [[sum(D[i][a_] * F(y[i]) for i in range(n))] for a_ in range(k)]
    beta = [row[0] for row in solve_exact(A, rhs)]
    resid = [F(y[i]) - sum(beta[a_] * D[i][a_] for a_ in range(k)) for i in range(n)]
    return beta, resid


def em_case(name, parents, y):
    beta, resid = ols(parents, y)
    scale = max(abs(r) for r in resid)
    norm = [mpf(r / scale) for r in resid]
    a, ps_, os_, x0, clamped, floored = em(norm)
    s = mpf(scale)
    return {"name": name, "x": [[float(v) for v in c] for c in parents], "y": [float(v) for v in y],
            "intercept": float(beta[0]), "coeffs": [float(b) for b in beta[1:]],
            "a": float(a), "process_std": float(ps_ * s), "obs_std": float(os_ * s),
            "initial_mean": float(x0 * s), "a_clamped": clamped, "variance_floored": floored}


def main():
    rng = random.Random(5)
    q8 = lambda lo, hi: F(rng.randint(lo * 8, hi * 8), 8)

    filters = [
        filter_case("moderate", F(3, 4), F(1, 2), F(1, 4), 1, [q8(-3, 3) for _ in range(9)]),
        filter_case("negative_gain", F(-1, 2), 2, F(1, 8), F(-1, 2), [q8(-4, 4) for _ in range(8)]),
        filter_case("random_walk", 1, F(1, 16), 1, 0, [q8(-2, 2) for _ in range(10)]),
        filter_case("two_rows", F(1, 4), 1, 1, 2, [F(1, 2), F(-3, 2)]),
        filter_case("noisy_state_quiet_observation", F(9, 10) if False else F(7, 8), 4, F(1, 64), 1, [q8(-5, 5) for _ in range(7)]),
    ]
    big = F(10) ** 9
    ys = [q8(-3, 3) for _ in range(9)]
    densities = [
        density_case("unit_scale", F(3, 4), F(1, 2), F(1, 4), 1, ys, "baseline"),
        density_case("scale_1e9", F(3, 4), F(1, 2) * big, F(1, 4) * big, big, [v * big for v in ys],
                     "all noise scales and observations scaled by 1e9"),
        density_case("scale_1e-9", F(3, 4), F(1, 2) / big, F(1, 4) / big, 1 / big, [v / big for v in ys],
                     "all noise scales and observations scaled by 1e-9"),
    ]

    # EM series: an autoregressive state seen through noise, plus a regressor and a near-unit-root path
    n = 30
    state = 0.0
    ar = []
    for _ in range(n):
        state = 0.8 * state + rng.gauss(0, 0.6)
        ar.append(state + rng.gauss(0, 0.3))
    x = [float(q8(-2, 2)) for _ in range(n)]
    walk = []
    s = 0.0
    for _ in range(n):
        s += rng.gauss(0.15, 0.2)
        walk.append(s)
    r = lambda v: float(F(round(v * 4096), 4096))
    ems = [
        em_case("autoregressive_state", [], [r(v) + 3 for v in ar]),
        em_case("with_regressor", [x], [r(v) + 1.5 * xv + 2 for v, xv in zip(ar, x)]),
        em_case("drifting_path", [], [r(v) for v in walk]),
        # an exact ramp: a near-unit-root coefficient (the clamp is not reached in 25 iterations)
        em_case("ramp_near_unit_root", [], [float(F(t, 8)) for t in range(24)]),
    ]
    print(json.dumps({"filter_cases": filters, "density_cases": densities, "em_cases": ems}, indent=1))


main()
