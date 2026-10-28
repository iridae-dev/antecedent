"""Typed-bridge point mappings and their first-order propagated variances, in exact arithmetic.

Regenerate with:  python3 typed_bridges_reference.py > typed_bridges_reference.txt

Every input is a binary64 value taken as an exact rational (fractions.Fraction); every formula is
evaluated without rounding and the result is rounded once to the nearest binary64.
  trial-level line   b = Sxy/Sxx, a = ybar - b xbar, R^2 = Sxy^2/(Sxx Syy), s^2 = (Syy - Sxy^2/Sxx)/(k-2)
                     mean-line coefficient moments Var(b) = s^2/Sxx, Var(a) = s^2(1/k + xbar^2/Sxx),
                     Cov(a,b) = -s^2 xbar/Sxx
  affine propagation Var(a + bX) with Cov(b, X) = c and X ~ (x, V), a independent of X:
                     Var(a) + x^2 Var(b) + b^2 V + 2 x b c + 2 x Cov(a,b) + Var(b) V + c^2,
                     mean a + b x + c
  correction         p = (q + Sp - 1)/J, J = Se + Sp - 1, Var(p) = (V + p^2 Var(Se) + (1-p)^2 Var(Sp))/J^2;
                     forward q = Se p + (1-Sp)(1-p), Var(q) = J^2 V + p^2 Var(Se) + (1-p)^2 Var(Sp)
  nonlinear square   E[X^2] = m^2 + v
The bridged-claim value is the plug-in; the propagated mean adds c.  Repository-authored
(MIT OR Apache-2.0); standard library only.
"""
import json
from fractions import Fraction as F


def fl(x):
    return float(x)


def line(trials):
    k = len(trials)
    xs = [F(t[0]) for t in trials]
    ys = [F(t[1]) for t in trials]
    xb, yb = sum(xs) / k, sum(ys) / k
    sxx = sum((x - xb) ** 2 for x in xs)
    sxy = sum((x - xb) * (y - yb) for x, y in zip(xs, ys))
    syy = sum((y - yb) ** 2 for y in ys)
    b = sxy / sxx
    a = yb - b * xb
    rss = syy - sxy * sxy / sxx
    s2 = rss / (k - 2)
    return dict(k=k, xb=xb, sxx=sxx, b=b, a=a, r2=sxy * sxy / (sxx * syy), s2=s2,
                lo=min(xs), hi=max(xs))


def trial_case(name, trials, x0, se, cov):
    L = line(trials)
    x0f, V, c = F(x0), F(se) ** 2, F(cov)
    vb = L["s2"] / L["sxx"]
    va = L["s2"] * (F(1, L["k"]) + L["xb"] ** 2 / L["sxx"])
    cab = -L["s2"] * L["xb"] / L["sxx"]
    value = L["a"] + L["b"] * x0f
    var = va + x0f ** 2 * vb + L["b"] ** 2 * V + 2 * x0f * L["b"] * c + 2 * x0f * cab + vb * V + c * c
    assert c * c <= vb * V
    mean = value + c
    return {"name": name, "trials": trials, "x0": x0, "se": se, "cov": cov,
            "slope": fl(L["b"]), "intercept": fl(L["a"]), "r2": fl(L["r2"]), "value": fl(value),
            "mean": fl(mean), "variance": fl(var), "square_expectation": fl(mean * mean + var),
            "min": fl(L["lo"]), "max": fl(L["hi"])}


def affine_case(name, a, b, x0, se, va, vb, cab, cov):
    a, b, x0f, V, va, vb, cab, c = map(F, (a, b, x0, se, va, vb, cab, cov))
    V = V * V
    value = a + b * x0f
    var = va + x0f ** 2 * vb + b * b * V + 2 * x0f * b * c + 2 * x0f * cab + vb * V + c * c
    mean = value + c
    return {"name": name, "intercept": fl(a), "slope": fl(b), "x0": x0, "se": se,
            "intercept_variance": fl(va), "slope_variance": fl(vb), "covariance_ab": fl(cab),
            "cov": cov, "value": fl(value), "mean": fl(mean), "variance": fl(var),
            "square_expectation": fl(mean * mean + var)}


def correction_case(name, sens, spec, vse, vsp, q, se):
    S, P, a, d = F(sens), F(spec), F(vse), F(vsp)
    q, V = F(q), F(se) ** 2
    J = S + P - 1
    p = (q + P - 1) / J
    var_back = (V + p * p * a + (1 - p) ** 2 * d) / (J * J)
    qf = S * q + (1 - P) * (1 - q)
    var_fwd = J * J * V + q * q * a + (1 - q) ** 2 * d
    return {"name": name, "sensitivity": sens, "specificity": spec, "var_sensitivity": vse,
            "var_specificity": vsp, "value_back": fl(p), "q": fl(q), "se": se,
            "variance_back": fl(var_back), "value_forward": fl(qf), "variance_forward": fl(var_fwd)}


trial_sets = {
    "hand": [(0.0, 0.1), (0.5, 0.5), (1.0, 0.9), (1.5, 1.5)],
    "offset_1e6": [(1.0e6 + x, 0.3 + 0.8 * x + e) for x, e in
                   [(0.0, 0.05), (0.4, -0.04), (1.1, 0.03), (1.5, -0.06), (2.2, 0.02)]],
    "tight_x": [(1.0 + 1.0e-6 * i, 2.0 + 1.0e-6 * (3 * i + (-1) ** i) * 0.3) for i in range(6)],
    "negative_slope_twelve": [(0.1 * i, 3.0 - 0.7 * 0.1 * i + [0.03, -0.05, 0.02, 0.04, -0.01, -0.03,
                                                                 0.05, -0.02, 0.01, -0.04, 0.03, 0.0][i])
                              for i in range(12)],
    "three_trials": [(0.0, 0.2), (1.0, 0.9), (2.0, 2.3)],
    "tied_x": [(0.0, 0.1), (0.0, 0.3), (1.0, 0.8), (1.0, 1.0), (2.0, 1.9)],
}
cases = {"trial": [], "affine": [], "correction": []}
cases["trial"].append(trial_case("hand_independent", trial_sets["hand"], 0.8, 0.1, 0.0))
cases["trial"].append(trial_case("hand_at_min", trial_sets["hand"], 0.0, 0.1, 0.0))
cases["trial"].append(trial_case("hand_at_max_positive_cov", trial_sets["hand"], 1.5, 0.2, 0.006))
cases["trial"].append(trial_case("offset_1e6", trial_sets["offset_1e6"], 1.0e6 + 0.7, 0.05, 0.0))
cases["trial"].append(trial_case("tight_x", trial_sets["tight_x"], 1.0 + 2.0e-6, 0.01, 0.0))
cases["trial"].append(trial_case("negative_slope_negative_cov", trial_sets["negative_slope_twelve"], 0.55, 0.3, -0.0004))
cases["trial"].append(trial_case("three_trials", trial_sets["three_trials"], 1.3, 0.15, 0.0))
cases["trial"].append(trial_case("tied_x", trial_sets["tied_x"], 0.5, 0.1, 0.0))
cases["affine"].append(affine_case("independent", 0.1, 0.6, 0.5, 0.05, 0.0004, 0.0009, 0.0, 0.0))
cases["affine"].append(affine_case("covariance_and_correlated_mapping", -2.0, 3.5, 4.0, 0.2, 0.01, 0.04, 0.012, 0.01))
cases["affine"].append(affine_case("negative_slope_large_x", 5.0, -0.25, 1.0e5, 3.0, 0.5, 1.0e-6, -2.0e-4, -1.0e-3))
cases["correction"].append(correction_case("rogan_gladen", 0.9, 0.8, 0.0004, 0.0004, 0.31, 0.02))
cases["correction"].append(correction_case("near_min_youden", 0.62, 0.75, 0.0009, 0.0016, 0.5, 0.03))
cases["correction"].append(correction_case("at_attainable_ends_low", 0.95, 0.9, 0.0001, 0.0001, 0.1, 0.01))
cases["correction"].append(correction_case("at_attainable_ends_high", 0.95, 0.9, 0.0001, 0.0001, 0.95, 0.01))
def emit():
    for c in cases["trial"]:
        t = ",".join(f"{x!r}:{y!r}" for x, y in c["trials"])
        print("trial", c["name"], *[repr(c[k]) for k in ("x0", "se", "cov", "slope", "intercept", "r2",
              "value", "mean", "variance", "square_expectation", "min", "max")], t)
    for c in cases["affine"]:
        print("affine", c["name"], *[repr(c[k]) for k in ("intercept", "slope", "x0", "se",
              "intercept_variance", "slope_variance", "covariance_ab", "cov", "value", "mean",
              "variance", "square_expectation")])
    for c in cases["correction"]:
        print("correction", c["name"], *[repr(c[k]) for k in ("sensitivity", "specificity",
              "var_sensitivity", "var_specificity", "q", "se", "value_back", "variance_back",
              "value_forward", "variance_forward")])


emit()
