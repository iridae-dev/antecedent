# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Exact responder-weighted contrast of a randomized trial with outcomes missing at random.

Regenerate with:  uv run --script responder_weighted_reference.py > responder_weighted_reference.json

Within each arm (n assigned units, observation indicator R_i, covariates x_i, design z_i = (1, x_i)):
  * the observation probability is the logistic maximum likelihood pi_i = expit(z_i'b), found by Newton
    iterations to 1e-30 in mpmath;
  * the arm mean is the Hajek ratio  mu = sum_i R_i y_i / pi_i  /  sum_i R_i / pi_i;
  * its variance is the stacked M-estimation sandwich of the logistic score s_i = (R_i - pi_i) z_i and
    the weighted estimating equation g_i = R_i (y_i - mu) / pi_i: with unknowns (b, mu),
        A = -(1/n) sum d/d(b, mu) of (s_i, g_i)   (a (p+1) x (p+1) matrix),
        B = (1/n) sum (s_i, g_i)(s_i, g_i)',
        Var(mu_hat) = [A^-1 B A^-T]_{mu mu} / n,
    which the product reports with the sample-variance divisor n - 1 in place of n, that is
    n/(n - 1) times the sandwich variance; the Kish size is (sum w)^2 / sum w^2 over responders with w = 1/pi.
The contrast is treated minus control, its variance the sum of the arm variances.  An arm in which
everyone responds has pi = 1, no observation-model term, and variance sum (y - mu)^2 / (n (n - 1)).
Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import json
import mpmath as mp

mp.mp.dps = 40


def expit(v):
    return 1 / (1 + mp.exp(-v))


def fit_logistic(z, r):
    n, p = len(z), len(z[0])
    b = mp.matrix(p, 1)
    for _ in range(100):
        pi = [expit(sum(b[j] * z[i][j] for j in range(p))) for i in range(n)]
        grad = mp.matrix(p, 1)
        hess = mp.matrix(p, p)
        for i in range(n):
            for a in range(p):
                grad[a] += (r[i] - pi[i]) * z[i][a]
                for c in range(p):
                    hess[a, c] += pi[i] * (1 - pi[i]) * z[i][a] * z[i][c]
        step = mp.lu_solve(hess, grad)
        b += step
        if mp.norm(step) < mp.mpf(10) ** -30:
            break
    return b, [expit(sum(b[j] * z[i][j] for j in range(p))) for i in range(n)]


def arm(y, r, x):
    n = len(y)
    if all(r):
        mu = sum(mp.mpf(v) for v in y) / n
        var = sum((mp.mpf(v) - mu) ** 2 for v in y) / (n * (n - 1))
        return dict(assigned=n, responders=n, mean=float(mu), variance=float(var), min_probability=1.0, max_probability=1.0, ess=float(n))
    z = [[mp.mpf(1)] + [mp.mpf(c) for c in row] for row in x]
    b, pi = fit_logistic(z, [mp.mpf(int(v)) for v in r])
    p = len(z[0])
    w = [1 / pi[i] for i in range(n) if r[i]]
    mu = sum(mp.mpf(y[i]) / pi[i] for i in range(n) if r[i]) / sum(w)
    # Stacked estimating functions psi_i = (s_i, g_i) and the derivative matrix A.
    A = mp.matrix(p + 1, p + 1)
    B = mp.matrix(p + 1, p + 1)
    for i in range(n):
        s = [(int(r[i]) - pi[i]) * z[i][a] for a in range(p)]
        g = (mp.mpf(int(r[i])) * (mp.mpf(y[i]) - mu) / pi[i]) if r[i] else mp.mpf(0)
        psi = s + [g]
        for a in range(p + 1):
            for c in range(p + 1):
                B[a, c] += psi[a] * psi[c] / n
        for a in range(p):
            for c in range(p):
                A[a, c] += pi[i] * (1 - pi[i]) * z[i][a] * z[i][c] / n
        if r[i]:
            # d g / d b = -(y - mu)(1 - pi)/pi z ;  d g / d mu = -1/pi.  A is minus the derivative.
            for c in range(p):
                A[p, c] += (mp.mpf(y[i]) - mu) * (1 - pi[i]) / pi[i] * z[i][c] / n
            A[p, p] += 1 / pi[i] / n
    Ai = A ** -1
    cov = Ai * B * Ai.T
    var = cov[p, p] / n * mp.mpf(n) / (n - 1)
    return dict(assigned=n, responders=sum(1 for v in r if v), mean=float(mu), variance=float(var),
                min_probability=float(min(pi[i] for i in range(n) if r[i])), max_probability=float(max(pi)),
                ess=float(sum(w) ** 2 / sum(v * v for v in w)))


def trial(name, n_each, seed_shift, treated_effect, all_observed_control=False):
    rows = []
    for g in (False, True):
        for i in range(n_each):
            k = i + (n_each if g else 0)
            x1 = (((k * 7 + seed_shift) % 19) / 9.0) - 1.0
            x2 = (((k * 5 + 3 * seed_shift) % 13) / 6.0) - 1.0
            y = 1.0 + 0.8 * x1 - 0.5 * x2 + (treated_effect if g else 0.0) + 0.3 * (((k * 11) % 7) / 3.0 - 1.0)
            # Observation depends on the covariates through a fixed deterministic threshold pattern
            # that keeps the logistic fit regular (neither separated nor saturated).
            u = ((k * 37 + 11) % 23) / 23.0
            pobs = float(expit(mp.mpf(0.4 + 0.9 * x1 - 0.6 * x2)))
            observed = u < pobs
            if all_observed_control and not g:
                observed = True
            rows.append(dict(y=y, observed=observed, treated=g, x1=x1, x2=x2))
    out = []
    for g in (False, True):
        sub = [r for r in rows if r["treated"] == g]
        out.append(arm([r["y"] for r in sub], [r["observed"] for r in sub], [[r["x1"], r["x2"]] for r in sub]))
    return dict(name=name, rows=rows, control=out[0], treated=out[1],
                effect=out[1]["mean"] - out[0]["mean"], variance=out[1]["variance"] + out[0]["variance"])


cases = [trial("two_covariates", 60, 1, 1.5), trial("different_pattern", 45, 4, -0.5),
         trial("control_all_observed", 40, 2, 0.8, all_observed_control=True)]
print(json.dumps(cases, separators=(",", ":")))
