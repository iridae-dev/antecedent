# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath"]
# ///
"""Full-sample augmented inverse-probability weighting: trimmed ATE, ATT and ATC, at 60 digits.

Regenerate with:  uv run --script aipw_reference.py > aipw_reference.json

Propensity: unpenalised logistic maximum likelihood on all rows (Newton's method), clipped into
[c, 1-c] inside the score.  Outcome regressions: least squares of the outcome on [1, x] within
each arm.

Trimmed ATE.  Rows whose raw propensity lies outside [trim, 1-trim] leave both the outcome fits
and the average; the estimate is the mean over the kept rows R of
phi = mu_1 - mu_0 + T (y - mu_1)/e - (1 - T)(y - mu_0)/(1 - e).  Its variance is not written as an
influence formula: the whole fit (propensity on every row, arm regressions on R, the mean over R,
membership of R held fixed) is a function of case masses m_i, the derivative d_i of the estimate
with respect to every row's mass is taken by central difference at step 1e-30, and the influence
over ALL n rows is n d_i, so the rows trimmed from the average still enter through the propensity
they helped fit.  The plug-in SE^2 = n/(n-1) sum_i d_i^2 is reported as "se_plug_in"; the
published "se" is leverage-corrected (HC3-style): each influence value is divided by 1 - h_i, with
h_i the larger of 1/n and the row's least-squares leverage x_i'(X_a'X_a)^-1 x_i in its arm's
regression on R (trimmed rows: 1/n), and SE is the sample standard deviation of the corrected
values over sqrt(n).  (The influence of the kept rows alone, which leaves those rows out, differs
by up to a percent on these tables and is reported, uncorrected, as "kept_rows_only".)

ATT / ATC.  The estimate is the mean over the treated (controls) of the doubly robust score
(T/pi)(y - mu_0) - ((1-T)/pi) e/(1-e) (y - mu_0) for ATT, pi the treated share, and the mirror
for ATC with mu_1; the plug-in standard error ("se_plug_in") is the sample standard deviation of
the centred efficient influence function psi - tau T/pi (psi - tau (1-T)/pi) divided by sqrt(n),
and the published "se" divides each centred value by 1 - h_i first, h_i the larger of 1/n and the
row's leverage in the one arm regression the score uses (the controls for ATT, the treated for
ATC; 1/n for the other arm).  Values are written as
nearest binary64.  Repository-authored (MIT OR Apache-2.0).
"""
import json
from mpmath import mp, mpf, exp, matrix, lu_solve, sqrt

mp.dps = 60
_state = 0x3C6EF372FE94F82B


def u01():
    global _state
    _state = (_state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
    return (_state >> 11) / float(1 << 53)


def wls(rows, y, mass):
    p = len(rows[0])
    A, b = matrix(p, p), matrix(p, 1)
    for r, v, m in zip(rows, y, mass):
        for a in range(p):
            b[a] += m * r[a] * v
            for c in range(p):
                A[a, c] += m * r[a] * r[c]
    beta = lu_solve(A, b)
    return lambda r: sum(beta[k] * r[k] for k in range(p))


def leverage(rows):
    """Least-squares leverage x_i' (X'X)^-1 x_i of each row of one arm's regression."""
    p = len(rows[0])
    A = matrix(p, p)
    for r in rows:
        for a in range(p):
            for c in range(p):
                A[a, c] += r[a] * r[c]
    out = []
    for r in rows:
        s = lu_solve(A, matrix(list(r)))
        out.append(sum(r[k] * s[k] for k in range(p)))
    return out


def corrected_se(influence, lev):
    """Sample SD over sqrt(n) of the centred influence values divided by 1 - h_i."""
    n = len(influence)
    mean = sum(influence) / n
    j = [(v - mean) / (1 - min(h, mpf("0.99"))) for v, h in zip(influence, lev)]
    jm = sum(j) / n
    return sqrt(sum((v - jm) ** 2 for v in j) / (n - 1) / n)


def logit(rows, t, mass):
    p = len(rows[0])
    beta = [mpf(0)] * p
    for _ in range(200):
        score = [mpf(0)] * p
        info = matrix(p, p)
        for r, ti, m in zip(rows, t, mass):
            e = 1 / (1 + exp(-sum(beta[k] * r[k] for k in range(p))))
            for a in range(p):
                score[a] += m * (ti - e) * r[a]
                for c in range(p):
                    info[a, c] += m * e * (1 - e) * r[a] * r[c]
        step = lu_solve(info, matrix(score))
        beta = [beta[k] + step[k] for k in range(p)]
        if max(abs(s) for s in step) < mpf(10) ** -50:
            break
    return [1 / (1 + exp(-sum(beta[k] * r[k] for k in range(p)))) for r in rows]


def trimmed_ate(X, y, t, mass, keep, clip):
    n = len(t)
    e = logit(X, t, mass)
    ec = [min(max(v, mpf(clip)), 1 - mpf(clip)) if clip else v for v in e]
    f = {}
    for arm in (0, 1):
        idx = [i for i in keep if (t[i] > 0.5) == (arm == 1)]
        f[arm] = wls([X[i] for i in idx], [y[i] for i in idx], [mass[i] for i in idx])
    phi = {}
    for i in keep:
        m0, m1 = f[0](X[i]), f[1](X[i])
        phi[i] = m1 - m0 + t[i] * (y[i] - m1) / ec[i] - (1 - t[i]) * (y[i] - m0) / (1 - ec[i])
    den = sum(mass[i] for i in keep)
    return sum(mass[i] * phi[i] for i in keep) / den, e


def population(X, y, t, target, clip):
    n = len(t)
    mass = [mpf(1)] * n
    e = logit(X, t, mass)
    ec = [min(max(v, mpf(clip)), 1 - mpf(clip)) if clip else v for v in e]
    arm = 0 if target == "att" else 1  # outcome model of the comparison arm
    idx = [i for i in range(n) if (t[i] > 0.5) == (arm == 1)]
    f = wls([X[i] for i in idx], [y[i] for i in idx], mass)
    if target == "att":
        pi = sum(1 for i in range(n) if t[i] > 0.5) / mpf(n)
        psi = [(t[i] / pi) * (y[i] - f(X[i])) - ((1 - t[i]) / pi) * (ec[i] / (1 - ec[i])) * (y[i] - f(X[i])) for i in range(n)]
        share = [t[i] / pi for i in range(n)]
    else:
        pi = sum(1 for i in range(n) if t[i] <= 0.5) / mpf(n)
        # effect on the controls is mu_1 - y: the negative of the control-arm residual form
        psi = [-((1 - t[i]) / pi) * (y[i] - f(X[i])) + (t[i] / pi) * ((1 - ec[i]) / ec[i]) * (y[i] - f(X[i])) for i in range(n)]
        share = [(1 - t[i]) / pi for i in range(n)]
    tau = sum(psi) / n
    return tau, psi, share


def case(name, t, y, x, clip, trim):
    n = len(t)
    X = [[mpf(1)] + [mpf(c) for c in row] for row in x]
    yy = [mpf(v) for v in y]
    out = {"name": name, "t": t, "y": y, "x": x, "clip": clip, "trim": trim}
    # trimmed ATE
    raw = logit(X, t, [mpf(1)] * n)
    keep = [i for i in range(n) if trim is None or (trim <= raw[i] <= 1 - mpf(trim))]
    if trim is not None:
        h = mpf(10) ** -30
        est, _ = trimmed_ate(X, yy, t, [mpf(1)] * n, keep, clip)
        d = []
        for i in range(n):
            mass = [mpf(1)] * n
            mass[i] = 1 + h
            hi = trimmed_ate(X, yy, t, mass, keep, clip)[0]
            mass[i] = 1 - h
            lo = trimmed_ate(X, yy, t, mass, keep, clip)[0]
            d.append((hi - lo) / (2 * h))
        m = mpf(len(keep))
        nf = mpf(n)
        se_all = sqrt(nf / (nf - 1) * sum(v * v for v in d))
        se_kept = sqrt(m / (m - 1) * sum(d[i] ** 2 for i in keep))
        lev = [1 / nf] * n
        for arm in (0, 1):
            idx = [i for i in keep if (t[i] > 0.5) == (arm == 1)]
            for i, h in zip(idx, leverage([X[i] for i in idx])):
                lev[i] = max(lev[i], h)
        se = corrected_se([nf * v for v in d], lev)
        out["trimmed_ate"] = {"ate": float(est), "se": float(se), "se_plug_in": float(se_all),
                              "kept_rows_only": float(se_kept), "kept": len(keep)}
    return out


def pop_case(name, t, y, x, clip):
    n = len(t)
    X = [[mpf(1)] + [mpf(c) for c in row] for row in x]
    yy = [mpf(v) for v in y]
    out = {"name": name, "t": t, "y": y, "x": x, "clip": clip, "trim": None}
    for target in ("att", "atc"):
        tau, psi, share = population(X, yy, t, target, clip)
        centred = [psi[i] - tau * share[i] for i in range(n)]
        mean = sum(centred) / n
        var = sum((v - mean) ** 2 for v in centred) / (n - 1)
        arm = 0 if target == "att" else 1  # the one arm regression the score uses
        idx = [i for i in range(n) if (t[i] > 0.5) == (arm == 1)]
        lev = [1 / mpf(n)] * n
        for i, h in zip(idx, leverage([X[i] for i in idx])):
            lev[i] = max(lev[i], h)
        out[target] = {"ate": float(tau), "se": float(corrected_se(centred, lev)),
                       "se_plug_in": float(sqrt(var / n))}
    return out


def draw(n, p, slope=0.8):
    t, y, x = [], [], []
    for _ in range(n):
        row = [u01() * 4 - 2 for _ in range(p)]
        eta = slope * row[0] - 0.3 * (row[1] if p > 1 else 0.0)
        ti = 1.0 if u01() < 1 / (1 + 2.718281828459045 ** (-eta)) else 0.0
        yi = 1.0 + 1.5 * ti + 0.9 * row[0] + 0.4 * (row[1] if p > 1 else 0.0) + 0.5 * row[0] * ti + (u01() - 0.5)
        t.append(ti)
        y.append(float(yi))
        x.append(row)
    return t, y, x


cases = []
t, y, x = draw(60, 2, slope=1.6)
cases.append(case("trim_moderate", t, y, x, None, 0.15))
cases.append(case("trim_with_clip", t, y, x, 0.2, 0.1))
cases.append(case("trim_outcome_offset_1e7", t, [v + 1.0e7 for v in y], x, None, 0.15))
t2, y2, x2 = draw(80, 1, slope=2.4)
cases.append(case("trim_strong_selection", t2, y2, x2, None, 0.1))
t3, y3, x3 = draw(60, 2)
pops = [pop_case("population_two_covariates", t3, y3, x3, None),
        pop_case("population_clip", t2, y2, x2, 0.1),
        pop_case("population_outcome_offset_1e7", t3, [v + 1.0e7 for v in y3], x3, None)]
print(json.dumps({"cases": cases, "population_cases": pops}))
