# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""50-digit maximum-likelihood references for the generalised linear model fits.

Regenerate with:  uv run --script glm_reference.py > glm_reference.json

Each case is an exact binary64 design, outcome and (optionally) prior-weight vector.  The
expected coefficients are the maximiser of the (penalised) log-likelihood

    sum_i w_i l_i(beta) - (lambda / 2) |beta_penalised|^2

computed in `mpmath` at 50 digits by damped Fisher scoring on the exact log-likelihood (the
log-likelihood itself, not the working-response recursion the product uses, decides acceptance of
each step), iterated until the step is below 1e-40.  Families:

* logit  l = y log mu + (1-y) log(1-mu),  mu = 1/(1+e^-eta)
* probit same with mu = Phi(eta)
* Poisson l = y log mu - mu - log y!  (log link)
* NB2    l = lgamma(y+t) - lgamma(t) - lgamma(y+1) + t log(t/(t+mu)) + y log(mu/(t+mu)),  t = 1/alpha
* Gaussian identity (weighted least squares)

Reported alongside the coefficients: the deviance of each row convention the product uses, the
prior-weighted deviance, and for negative binomial the maximum-likelihood dispersion found by a
profile search on the exact weighted likelihood.  This script and its frozen output are
repository-authored (MIT OR Apache-2.0); mpmath is used only here, never at product run time.
"""
import json
import sys

import mpmath as mp

mp.mp.dps = 50


class Lcg:
    def __init__(self, seed):
        self.s = seed

    def u(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        return (self.s >> 11) / float(1 << 53)

    def sym(self, scale=1.0):
        return round((self.u() * 2 - 1) * 1e6) / 1e6 * scale


def sigma(eta):
    return 1 / (1 + mp.exp(-eta))


def Phi(eta):
    return mp.ncdf(eta)


def loglik_rows(family, y, eta, alpha=None):
    out = []
    for yi, e in zip(y, eta):
        yi = mp.mpf(yi)
        if family == "logit":
            out.append(-mp.log(1 + mp.exp(-e)) if yi == 1 else -mp.log(1 + mp.exp(e)))
        elif family == "probit":
            out.append(mp.log(Phi(e)) if yi == 1 else mp.log(Phi(-e)))
        elif family == "poisson":
            mu = mp.exp(e)
            out.append(yi * e - mu - mp.loggamma(yi + 1))
        elif family == "nb":
            mu = mp.exp(e)
            t = 1 / mp.mpf(alpha)
            out.append(mp.loggamma(yi + t) - mp.loggamma(t) - mp.loggamma(yi + 1)
                       + t * mp.log(t / (t + mu)) + yi * mp.log(mu / (t + mu)))
        elif family == "gaussian":
            out.append(-(yi - e) ** 2 / 2)
    return out


def weights_info(family, y, eta, alpha=None):
    """Per-row score multiplier s_i and expected information weight v_i (d l / d eta, -E d2 l / d eta2)."""
    s, v = [], []
    for yi, e in zip(y, eta):
        yi = mp.mpf(yi)
        if family == "logit":
            s.append(yi - sigma(e))
            v.append(sigma(e) * sigma(-e))
        elif family == "probit":
            ph = mp.npdf(e)
            s.append(ph / Phi(e) if yi == 1 else -ph / Phi(-e))
            v.append(ph * ph / (Phi(e) * Phi(-e)))
        elif family == "poisson":
            mu = mp.exp(e)
            s.append(yi - mu)
            v.append(mu)
        elif family == "nb":
            mu = mp.exp(e)
            s.append((yi - mu) / (1 + mp.mpf(alpha) * mu))
            v.append(mu / (1 + mp.mpf(alpha) * mu))
        elif family == "gaussian":
            s.append(yi - e)
            v.append(mp.mpf(1))
    return s, v


def fit(family, cols, y, w=None, lam=0, alpha=None, start=None):
    n, p = len(y), len(cols)
    w = [mp.mpf(1)] * n if w is None else [mp.mpf(v) for v in w]
    X = [[mp.mpf(c[i]) for c in cols] for i in range(n)]
    pen = [0] * p
    if lam:
        pen = [lam] * p
        if all(v == cols[0][0] for v in cols[0]):
            pen[0] = 0
    beta = [mp.mpf(0)] * p if start is None else [mp.mpf(b) for b in start]

    def objective(b):
        eta = [sum(X[i][k] * b[k] for k in range(p)) for i in range(n)]
        ll = sum(wi * l for wi, l in zip(w, loglik_rows(family, y, eta, alpha)))
        return ll - sum(pen[k] * b[k] ** 2 for k in range(p)) / 2

    cur = objective(beta)
    for _ in range(400):
        eta = [sum(X[i][k] * beta[k] for k in range(p)) for i in range(n)]
        s, v = weights_info(family, y, eta, alpha)
        g = mp.matrix(p, 1)
        H = mp.matrix(p, p)
        for i in range(n):
            for a in range(p):
                g[a] += w[i] * s[i] * X[i][a]
                for b2 in range(p):
                    H[a, b2] += w[i] * v[i] * X[i][a] * X[i][b2]
        for a in range(p):
            g[a] -= pen[a] * beta[a]
            H[a, a] += pen[a]
        step = mp.lu_solve(H, g)
        t = mp.mpf(1)
        while True:
            cand = [beta[k] + t * step[k] for k in range(p)]
            val = objective(cand)
            if val >= cur - mp.mpf(10) ** -45 or t < mp.mpf(2) ** -60:
                break
            t /= 2
        beta, cur = cand, val
        if max(abs(t * step[k]) for k in range(p)) < mp.mpf(10) ** -40:
            break
    return beta, cur


def deviance(family, y, eta, w=None, alpha=None):
    """Deviance contributions against the saturated model, weighted by w when given."""
    tot = mp.mpf(0)
    for i, (yi, e) in enumerate(zip(y, eta)):
        yi = mp.mpf(yi)
        wi = 1 if w is None else mp.mpf(w[i])
        if wi == 0:
            continue
        if family in ("logit", "probit"):
            d = -2 * loglik_rows(family, [yi], [e])[0]
        elif family == "poisson":
            mu = mp.exp(e)
            d = 2 * ((yi * mp.log(yi / mu) if yi > 0 else 0) - (yi - mu))
        elif family == "nb":
            mu = mp.exp(e)
            t = 1 / mp.mpf(alpha)
            d = 2 * ((yi * mp.log(yi / mu) if yi > 0 else 0) - (yi + t) * mp.log((yi + t) / (mu + t)))
        else:
            d = (yi - e) ** 2
        tot += wi * d
    return tot


def design(kind, n, rng):
    one = [1.0] * n
    u = [rng.sym(2.0) for _ in range(n)]
    z = [rng.sym(1.0) for _ in range(n)]
    if kind == "plain":
        return [one, u, z]
    if kind == "offset_1e3":
        return [one, [1e3 + v for v in u], z]
    if kind == "offset_1e5":
        return [one, [1e5 + v for v in u], z]
    if kind == "scaled":
        return [one, [1e-4 * v for v in u], [1e4 * v for v in z]]
    if kind == "binary":
        return [one, [float(i % 2) for i in range(n)]]
    raise ValueError(kind)


def draw(family, cols, rng, coef, alpha=None):
    n = len(cols[0])
    y = []
    for i in range(n):
        eta = sum(c[i] * b for c, b in zip(cols, coef))
        u = rng.u()
        if family in ("logit", "probit"):
            p = 1 / (1 + mp.exp(-eta)) if family == "logit" else mp.ncdf(eta)
            y.append(1.0 if u < p else 0.0)
        elif family == "gaussian":
            y.append(eta + rng.sym(1.0))
        elif family == "poisson" and coef[0] > 8:
            # large counts: a noisy integer near the mean (the likelihood is defined for any
            # non-negative real outcome through the gamma function)
            y.append(float(round(float(mp.exp(eta)) * (1 + 0.1 * rng.sym(1.0)))))
        else:
            mu = mp.exp(eta)
            # inverse-CDF draw from Poisson / NB2 with exact pmf accumulation
            k, cdf = 0, mp.mpf(0)
            while True:
                if family == "poisson":
                    pm = mp.exp(-mu + k * mp.log(mu) - mp.loggamma(k + 1))
                else:
                    t = 1 / mp.mpf(alpha)
                    pm = mp.exp(mp.loggamma(k + t) - mp.loggamma(t) - mp.loggamma(k + 1)
                                + t * mp.log(t / (t + mu)) + k * mp.log(mu / (t + mu)))
                cdf += pm
                if u < cdf or k > 5000:
                    break
                k += 1
            y.append(float(k))
    return y


def flt(v):
    return [float(x) for x in v]


def cases():
    out = []
    specs = [
        # (family, design, n, true coef, weighted, alpha)
        ("logit", "plain", 60, [0.3, 0.8, -0.5], False, None),
        ("logit", "plain", 60, [0.3, 0.8, -0.5], True, None),
        ("logit", "offset_1e3", 80, [-0.2, 0.0, 0.6], False, None),
        ("logit", "scaled", 80, [0.1, 3000.0, -0.002], False, None),
        ("logit", "plain", 120, [-1.0, 2.5, 1.5], False, None),
        ("probit", "plain", 60, [0.2, 0.5, -0.4], False, None),
        ("probit", "plain", 60, [0.2, 0.5, -0.4], True, None),
        ("probit", "offset_1e3", 80, [0.0, 0.0, 0.4], False, None),
        ("poisson", "plain", 60, [1.0, 0.4, -0.3], False, None),
        ("poisson", "plain", 60, [1.0, 0.4, -0.3], True, None),
        ("poisson", "offset_1e3", 60, [1.0, 0.0, 0.3], False, None),
        ("poisson", "plain", 40, [9.5, 0.2, -0.1], False, None),   # counts near 1e4
        ("poisson", "binary", 40, [-3.0, 1.0], False, None),        # mostly zeros
        ("nb", "plain", 80, [1.2, 0.3, -0.2], False, 0.4),
        ("nb", "plain", 80, [1.2, 0.3, -0.2], True, 0.4),
        ("gaussian", "plain", 40, [1.0, 2.0, -1.0], True, None),
        ("gaussian", "offset_1e5", 40, [1.0, 0.0, -1.0], False, None),
    ]
    for k, (family, dname, n, coef, weighted, alpha) in enumerate(specs):
        rng = Lcg(7000 + k)
        cols = design(dname, n, rng)
        y = draw(family, cols, rng, coef, alpha)
        w = None
        if weighted:
            w = [float(10 ** (-3 + 6 * rng.u())) for _ in range(n)]
            w[1] = 0.0
            w[2] = 1e6
        beta, ll = fit(family, cols, y, w, alpha=alpha)
        n_ = len(y)
        eta = [sum(mp.mpf(c[i]) * b for c, b in zip(cols, beta)) for i in range(n_)]
        mom = None
        if family == "nb":
            # Pearson method-of-moments dispersion from the exact Poisson fit (same weights):
            # max(sum m e^2 - max(sum m - p, 1), 0) / sum m mu, floored at 1e-8.
            pb, _ = fit("poisson", cols, y, w)
            pe = [sum(mp.mpf(c[i]) * b for c, b in zip(cols, pb)) for i in range(n_)]
            ww = [mp.mpf(1)] * n_ if w is None else [mp.mpf(v) for v in w]
            mus = [mp.exp(e) for e in pe]
            ss = sum(m * (mp.mpf(yi) - mu) ** 2 / mu for m, yi, mu in zip(ww, y, mus))
            df = max(sum(ww) - len(cols), mp.mpf(1))
            mom = float(max(max(ss - df, 0) / sum(m * mu for m, mu in zip(ww, mus)),
                            mp.mpf("1e-8")))
        rec = {"mom_alpha": mom, "family": family, "design": dname, "n": n, "p": len(cols),
               "x": [v for c in cols for v in c], "y": y, "w": w, "alpha": alpha,
               "coef": flt(beta),
               "deviance": float(deviance(family, y, eta, None, alpha)),
               "deviance_weighted": float(deviance(family, y, eta, w, alpha)) if w else None}
        out.append(rec)
    return out


def nb_profile_cases():
    """Maximum-likelihood NB2 dispersion: the fit's beta and alpha jointly maximise the likelihood."""
    out = []
    for k, (n, coef, alpha, weighted) in enumerate([(120, [1.0, 0.4, -0.3], 0.5, False),
                                                    (120, [1.0, 0.4, -0.3], 0.5, True),
                                                    (150, [2.0, 0.2, 0.1], 0.05, False)]):
        rng = Lcg(8000 + k)
        cols = design("plain", n, rng)
        y = draw("nb", cols, rng, coef, alpha)
        w = None
        if weighted:
            w = [float(1 + 4 * rng.u()) for _ in range(n)]

        def profile(a):
            beta, ll = fit("nb", cols, y, w, alpha=a)
            return ll

        # golden-section search on log alpha of the profile likelihood
        lo, hi = mp.log(mp.mpf("1e-3")), mp.log(mp.mpf(5))
        gr = (mp.sqrt(5) - 1) / 2
        x1, x2 = hi - gr * (hi - lo), lo + gr * (hi - lo)
        f1, f2 = profile(mp.exp(x1)), profile(mp.exp(x2))
        for _ in range(30):
            if f1 > f2:
                hi, x2, f2 = x2, x1, f1
                x1 = hi - gr * (hi - lo)
                f1 = profile(mp.exp(x1))
            else:
                lo, x1, f1 = x1, x2, f2
                x2 = lo + gr * (hi - lo)
                f2 = profile(mp.exp(x2))
        a = mp.exp((lo + hi) / 2)
        beta, ll = fit("nb", cols, y, w, alpha=a)
        out.append({"n": n, "p": 3, "x": [v for c in cols for v in c], "y": y, "w": w,
                    "alpha": float(a), "coef": flt(beta), "loglik": float(ll)})
    return out


def ridge_cases():
    """Penalised logistic / probit maximisers, including data that are completely separated."""
    out = []
    for k, (family, sep, lam, weighted) in enumerate([
            ("logit", False, 1e-4, False), ("logit", True, 1e-4, False),
            ("logit", True, 1e-4, True), ("probit", True, 1e-4, False),
            ("logit", False, 2.0, False), ("logit", True, 1e-8, False)]):
        rng = Lcg(9000 + k)
        n = 50
        one = [1.0] * n
        u = [rng.sym(2.0) for _ in range(n)]
        z = [rng.sym(1.0) for _ in range(n)]
        cols = [one, u, z]
        if sep:
            y = [1.0 if u[i] + 0.1 * z[i] > 0.05 else 0.0 for i in range(n)]
        else:
            y = draw(family, cols, rng, [0.1, 0.9, -0.6])
        w = [float(1 + 3 * rng.u()) for _ in range(n)] if weighted else None
        beta, ll = fit(family, cols, y, w, lam=lam)
        out.append({"family": family, "separated_data": sep, "lambda": lam, "n": n, "p": 3,
                    "x": [v for c in cols for v in c], "y": y, "w": w, "coef": flt(beta)})
    return out


def separation_cases():
    """Data whose unpenalised MLE does or does not exist, decided by exact enumeration of a
    separating hyperplane (a rational linear program), not by fitting."""
    out = []
    n = 24
    rng = Lcg(9500)
    base_u = [rng.sym(2.0) for _ in range(n)]
    designs = {
        "complete": [1.0 if v > 0.1 else 0.0 for v in base_u],
        "quasi_complete": None,
        "overlap": [1.0 if (v + rng.sym(1.5)) > 0.1 else 0.0 for v in base_u],
    }
    # quasi-complete: perfect separation except one tie row sharing the boundary value
    u = base_u[:]
    y = [1.0 if v > 0.1 else 0.0 for v in u]
    u[0], u[1] = 0.1, 0.1
    y[0], y[1] = 0.0, 1.0
    designs["quasi_complete"] = (u, y)
    for name, d in designs.items():
        if isinstance(d, tuple):
            uu, yy = d
        else:
            uu, yy = base_u, d
        # exact verdict: the MLE exists iff there is no (beta) with y_i=1 => x_i.beta >= 0 and
        # y_i=0 => x_i.beta <= 0, not all zero (Albert-Anderson).  With an intercept and one
        # covariate the cone can be tested exactly by checking the two directions in (b0, b1).
        from fractions import Fraction as F
        rows = [(F(1), F(v), 1 if yv == 1 else -1) for v, yv in zip(uu, yy)]
        # candidate direction set: rays through every pair of boundary lines plus axes
        cands = [(F(1), F(0)), (F(-1), F(0)), (F(0), F(1)), (F(0), F(-1))]
        pts = set()
        for a, b, _ in rows:
            pts.add((-b, a))
            pts.add((b, -a))
        cands += list(pts)
        separating = False
        for b0, b1 in cands:
            if (b0, b1) == (0, 0):
                continue
            vals = [(b0 * a + b1 * b) * s for a, b, s in rows]
            if all(v >= 0 for v in vals):
                separating = True
                break
        out.append({"name": name, "n": n, "p": 2, "x": [1.0] * n + list(uu), "y": list(yy),
                    "mle_exists": not separating})
    return out


if __name__ == "__main__":
    json.dump({"fits": cases(), "nb_dispersion": nb_profile_cases(), "ridge": ridge_cases(),
               "separation": separation_cases()}, sys.stdout, allow_nan=False)
