"""Additive-GAM g-computation of an intervention response: level and HC2 sandwich error, exactly.

Regenerate with:  python3 estimate_response_intervention_gcomp_reference.py > estimate_response_intervention_gcomp_reference.json

The outcome is fitted by the penalized additive cubic-spline model with unpenalized treatment smooths
and adjuster smooths at the caller's lambda (exact Fraction minimizer, type-7 knots, from
estimate_response_kennedy_dr_reference.Additive).  An intervention policy gives each treatment a law:
Set a point mass, Shift the factual value plus a constant, Bernoulli(p) the levels 0 and 1, Categorical
the levels 0, 1, ... with the normalized probabilities, Gaussian(m, v) a normal law.  Because the fit
is additive, the policy mean of row i is the sum over the coordinates of E[f_j(V_j)] with V_j drawn from
the coordinate's law (the factual value when it is not intervened); a Gaussian expectation is integrated
by mpmath quadrature over the knot intervals of the clamped spline, not by a fixed Gauss-Hermite rule.
The estimate is the mean of the row means.  The error is the HC2 sandwich: with the reduced design
X = [1 | first K-1 basis columns of each smooth], S the second-difference penalty of each smooth
restricted to those columns, h_i = x_i'(X'X + S)^-1 x_i and e_i the fit residual, the row influence is
(rowmean_i - mean) + n (e_i / sqrt(1 - h_i)) x_i'(X'X + S)^-1 c, c the policy-averaged design row
(1, then for each column the mean over its law of the first K-1 basis values), centered, and
SE = sqrt(sum psi^2 / (n (n - 1))).  A policy that shifts every treatment by 0 has the observed outcome
mean, with influence y_i - mean(y).  Repository-authored (MIT OR Apache-2.0); needs mpmath
(BSD-3-Clause) at run time, not redistributed.
"""
import json

from mpmath import mp, mpf, exp, quad, sqrt, pi, erfinv, inf

import estimate_response_kennedy_dr_reference as kd
from estimate_response_kennedy_dr_reference import F, Additive, mf, basis_row, penalty, solve, inverse

mp.dps = 40


def numeric(policy):
    out = {}
    for key, value in policy.items():
        if key == "kind":
            out[key] = value
        elif key == "probs":
            out[key] = [float(F(v)) for v in value]
        else:
            out[key] = float(F(value))
    return out


def interp_fit(fit, j, v):
    return fit.raw(j, v)


def expectation_gaussian(fit, j, mean, var):
    """E[f_j(clamp(A))] for A ~ N(mean, var), by quadrature over the knot intervals."""
    if var == 0:
        return mf(fit.raw(j, mean))
    kn = sorted(set(fit.knots[j]))
    m, s = mf(mean), sqrt(mf(var))

    def g(a):
        x = F(a) if False else None
        return None

    def integrand(a):
        # evaluate the exact rational spline at the rational approximation of a (40 digits)
        from fractions import Fraction
        v = Fraction(str(mp.nstr(a, 38)))
        z = (a - m) / s
        return mf(fit.raw(j, v)) * exp(-z * z / 2) / (s * sqrt(2 * pi))

    pts = [-inf] + [mf(k) for k in kn] + [inf]
    return quad(integrand, pts)


def law(policy, factual):
    kind = policy["kind"]
    if kind == "set":
        return [(F(policy["value"]), F(1))]
    if kind == "shift":
        return [(factual + F(policy["delta"]), F(1))]
    if kind == "bernoulli":
        p = F(policy["p"])
        return [(F(v), w) for v, w in ((0, 1 - p), (1, p)) if w > 0]
    if kind == "categorical":
        probs = [F(p) for p in policy["probs"]]
        total = sum(probs)
        return [(F(i), p / total) for i, p in enumerate(probs) if p > 0]
    raise ValueError(kind)


def build(name, n, K, lam, treatments, policies, a_off=F(0), y_off=F(0), x_scale=F(1), zero_shift=False):
    x = [kd.dy(kd.u01() * 4 - 2, 1024) * x_scale for _ in range(n)]
    cols = [[a_off + kd.dy(float(v / x_scale) * 0.5 + (kd.u01() - 0.5) * 1.5, 1024) for v in x]]
    if treatments == 2:
        cols.append([kd.dy(-0.3 * float(v / x_scale) + (kd.u01() - 0.5) * 2, 1024) for v in x])
    y = [y_off + kd.dy(1.2 * float(cols[0][i] - a_off) + (-0.6 * float(cols[1][i]) if treatments == 2 else 0.0)
                       + 0.4 * float(cols[0][i] - a_off) ** 2 + 0.8 * float(x[i] / x_scale) + (kd.u01() - 0.5), 1024)
         for i in range(n)]
    t = treatments
    fit = Additive(cols + [x], y, K, [F(0)] * t + [lam])
    fitted = [fit.predict([c[i] for c in cols] + [x[i]]) for i in range(n)]
    resid = [y[i] - fitted[i] for i in range(n)]
    # row means
    def policy_mean(i):
        total = F(0) if True else None
        tot = mpf(0)
        for j in range(t + 1):
            factual = (cols + [x])[j][i]
            if j < t and policies[j] is not None:
                pol = policies[j]
                if pol["kind"] == "gaussian":
                    tot += expectation_gaussian(fit, j, F(pol["mean"]), F(pol["var"]))
                else:
                    tot += sum(mf(w) * mf(fit.raw(j, v)) for v, w in law(pol, factual))
            else:
                tot += mf(fit.raw(j, factual))
        return tot
    # a Gaussian expectation is the same for every row: cache by coordinate
    cache = {}
    def coord_mean(j, i):
        pol = policies[j] if j < t else None
        if pol is not None and pol["kind"] == "gaussian":
            key = (j,)
            if key not in cache:
                cache[key] = expectation_gaussian(fit, j, F(pol["mean"]), F(pol["var"]))
            return cache[key]
        factual = (cols + [x])[j][i]
        if pol is None:
            return mf(fit.raw(j, factual))
        return sum(mf(w) * mf(fit.raw(j, v)) for v, w in law(pol, factual))
    row_means = [sum(coord_mean(j, i) for j in range(t + 1)) for i in range(n)]
    estimate = sum(row_means) / n
    # sandwich
    ncol = t + 1
    design = [[F(1)] * n]
    offsets = []
    for j in range(ncol):
        offsets.append(len(design))
        for b in range(K - 1):
            design.append([basis_row((cols + [x])[j][i], fit.knots[j], K)[b] for i in range(n)])
    p = len(design)
    A = [[sum(design[a][i] * design[b][i] for i in range(n)) for b in range(p)] for a in range(p)]
    lam_of = [F(0)] * t + [lam]
    P = penalty(K)
    for j in range(ncol):
        o = offsets[j]
        for a in range(K - 1):
            for b in range(K - 1):
                A[o + a][o + b] += lam_of[j] * P[a][b]
    Ainv = inverse(A)
    lev = [sum(design[a][i] * Ainv[a][b] * design[b][i] for a in range(p) for b in range(p)) for i in range(n)]
    # policy-averaged design row
    c = [mpf(1)]
    for j in range(ncol):
        pol = policies[j] if j < t else None
        rows_avg = [mpf(0)] * (K - 1)
        if pol is not None and pol["kind"] == "shift":
            pts = [((cols + [x])[j][i] + F(pol["delta"]), F(1, n)) for i in range(n)]
        elif pol is not None and pol["kind"] == "gaussian":
            pts = None
        elif pol is not None:
            pts = law(pol, F(0))
        else:
            pts = [((cols + [x])[j][i], F(1, n)) for i in range(n)]
        if pts is None:
            kn = sorted(set(fit.knots[j]))
            m, s = mf(F(pol["mean"])), sqrt(mf(F(pol["var"])))
            from fractions import Fraction
            for b in range(K - 1):
                if F(pol["var"]) == 0:
                    rows_avg[b] = mf(basis_row(F(pol["mean"]), fit.knots[j], K)[b])
                else:
                    def integrand(a, b=b):
                        v = Fraction(str(mp.nstr(a, 38)))
                        z = (a - m) / s
                        return mf(basis_row(v, fit.knots[j], K)[b]) * exp(-z * z / 2) / (s * sqrt(2 * pi))
                    rows_avg[b] = quad(integrand, [-inf] + [mf(k) for k in kn] + [inf])
        else:
            for v, w in pts:
                br = basis_row(v, fit.knots[j], K)
                for b in range(K - 1):
                    rows_avg[b] += mf(w) * mf(br[b])
        c.extend(rows_avg)
    # solve (X'X+S) z = c in floating exact-ish: use mp
    from mpmath import matrix, lu_solve
    Am = matrix(p, p)
    for a in range(p):
        for b in range(p):
            Am[a, b] = mf(A[a][b])
    cm = matrix(p, 1)
    for a in range(p):
        cm[a] = c[a]
    zvec = lu_solve(Am, cm)
    if zero_shift:
        ybar = sum(mf(v) for v in y) / n
        psi = [mf(v) - ybar for v in y]
        estimate = ybar
    else:
        psi = []
        for i in range(n):
            e = mf(resid[i]) / sqrt(1 - mf(lev[i]))
            coef = n * e * sum(mf(design[a][i]) * zvec[a] for a in range(p))
            psi.append(row_means[i] - estimate + coef)
        mu = sum(psi) / n
        psi = [v - mu for v in psi]
    se = sqrt(sum(v * v for v in psi) / (n * (n - 1)))
    z = sqrt(2) * erfinv(mpf("0.95"))
    return {
        "name": name, "basis": K, "lambda": float(lam), "treatments": t, "policies": [numeric(p_) for p_ in policies],
        "columns": [[str(float(v)) for v in col] for col in cols], "x": [str(float(v)) for v in x],
        "y": [str(float(v)) for v in y],
        "estimate": mp.nstr(estimate, 30), "se": mp.nstr(se, 30),
        "interval": [mp.nstr(estimate - z * se, 30), mp.nstr(estimate + z * se, 30)],
    }


if __name__ == "__main__":
    cases = []
    base = dict(n=60, K=5, lam=F(1))
    for label, pol in [
        ("set", {"kind": "set", "value": "1/4"}),
        ("shift", {"kind": "shift", "delta": "3/10"}),
        ("shift_zero", {"kind": "shift", "delta": "0"}),
        ("bernoulli", {"kind": "bernoulli", "p": "3/10"}),
        ("categorical", {"kind": "categorical", "probs": ["1/5", "1/2", "3/10"]}),
        ("gaussian_inside", {"kind": "gaussian", "mean": "1/10", "var": "9/100"}),
        ("gaussian_wide", {"kind": "gaussian", "mean": "0", "var": "4"}),
    ]:
        cases.append(build(f"one_treatment_{label}", treatments=1, policies=[pol], zero_shift=(label == "shift_zero"), **base))
    cases.append(build("two_treatments_set_and_gaussian", n=80, K=5, lam=F(1), treatments=2,
                       policies=[{"kind": "set", "value": "-1/5"}, {"kind": "gaussian", "mean": "1/5", "var": "1/10"}]))
    cases.append(build("two_treatments_shift_and_categorical", n=80, K=5, lam=F(1), treatments=2,
                       policies=[{"kind": "shift", "delta": "1/5"}, {"kind": "categorical", "probs": ["1", "1"]}]))
    cases.append(build("offsets_and_scales", n=70, K=6, lam=F(1, 2), treatments=1, a_off=F(1000), y_off=F(100000),
                       x_scale=F(250), policies=[{"kind": "set", "value": "1000"}]))
    print(json.dumps({"cases": cases}, indent=1))
