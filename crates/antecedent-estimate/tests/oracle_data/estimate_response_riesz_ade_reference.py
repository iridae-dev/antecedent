"""Cross-fitted Gaussian-score Riesz average derivative, exactly.

Regenerate with:  python3 estimate_response_riesz_ade_reference.py > estimate_response_riesz_ade_reference.json

The data are dyadic rationals (the generator and the exact penalized additive spline solver are
those of estimate_response_kennedy_dr_reference.py, imported so the two references share one
solver).  The pipeline follows the estimator's definition:

* folds: row i belongs to fold i mod K; the nuisances of a row's fold are fitted on the other rows;
* outcome nuisance: the penalized additive cubic-spline fit of y on (a, x) (or on a alone without
  adjusters) over the training rows at its unique exact solution; d mu / d a at a row is the
  derivative of the treatment smooth at the row's treatment, zero outside [knot 3, knot -4);
* treatment nuisance: the same fit of a on x, residual scale sigma = sqrt(RSS / max(n_train - edf, 1))
  with edf the hat-matrix trace; without adjusters, the training mean and the n-1 divisor;
* Riesz weight r_i = (a_i - m(x_i)) / sigma^2 with the held-out row's treatment mean m;
* score s_i = d mu/d a (a_i) + r_i (y_i - mu(a_i, x_i)); estimate = mean s; variance =
  sum (s - mean)^2 / (n - 1); se = sqrt(variance / n); interval = estimate -/+ z se with z the
  normal quantile of (1 + level) / 2; Kish ESS = (sum |r|)^2 / sum r^2.

Repository-authored (MIT OR Apache-2.0); needs mpmath (BSD-3-Clause) at run time, not redistributed.
"""
import json
import os
import sys
from fractions import Fraction as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mpmath import mp, mpf, sqrt as msqrt, erfinv  # noqa: E402

from estimate_response_kennedy_dr_reference import Additive, dy, make_data, mf, u01  # noqa: E402

mp.dps = 50


def scores(a, x, y, folds, K, lam):
    n = len(y)
    s, r = [mpf(0)] * n, [mpf(0)] * n
    for f in range(folds):
        train = [i for i in range(n) if i % folds != f]
        valid = [i for i in range(n) if i % folds == f]
        ya = [y[i] for i in train]
        if x is None:
            out = Additive([[a[i] for i in train]], ya, K, lam)
            mean = sum(a[i] for i in train) / len(train)
            rss = sum((a[i] - mean) ** 2 for i in train)
            df = F(max(len(train) - 1, 1))
            mt = {i: mean for i in range(n)}
        else:
            out = Additive([[a[i] for i in train], [x[i] for i in train]], ya, K, lam)
            tr = Additive([[x[i] for i in train]], [a[i] for i in train], K, lam)
            mt = {i: tr.predict([x[i]]) for i in range(n)}
            rss = sum((a[i] - mt[i]) ** 2 for i in train)
            df = max(F(len(train)) - tr.trace, F(1))
        sigma2 = mf(rss) / mf(df)
        for i in valid:
            mu = out.predict([a[i]] if x is None else [a[i], x[i]])
            d = out.derivative(0, a[i])
            r[i] = mf(a[i] - mt[i]) / sigma2
            s[i] = mf(d) + r[i] * mf(y[i] - mu)
    return s, r


def mps(v):
    return mp.nstr(v, 40)


def build(name, a, x, y, folds, K, lam, level, min_ess):
    s, r = scores(a, x, y, folds, K, lam)
    n = len(y)
    est = sum(s) / n
    var = sum((v - est) ** 2 for v in s) / (n - 1)
    se = msqrt(var / n)
    p = mpf(1) / 2 + mpf(level) / 2
    z = msqrt(2) * erfinv(2 * p - 1)
    ess = sum(abs(v) for v in r) ** 2 / sum(v * v for v in r)
    case = {
        "name": name, "folds": folds, "basis": K, "lambda": float(lam), "level": level,
        "min_ess": min_ess, "adjusters": 0 if x is None else 1,
        "a": [str(float(v)) for v in a], "y": [str(float(v)) for v in y],
        "scores": [mps(v) for v in s], "riesz": [mps(v) for v in r],
        "estimate": mps(est), "se": mps(se), "lower": mps(est - z * se), "upper": mps(est + z * se),
        "ess": mps(ess),
    }
    if x is not None:
        case["x"] = [str(float(v)) for v in x]
    return case


if __name__ == "__main__":
    cases = []
    a, x, y = make_data(60, F(0), F(0), F(1))
    cases.append(build("ordinary", a, x, y, 3, 5, F(1), 0.95, 1.0))
    a, x, y = make_data(50, F(1000), F(1000000), F(250))
    cases.append(build("offsets_and_scales", a, x, y, 5, 5, F(1), 0.9, 1.0))
    # One treatment value far outside the other rows: the held-out derivative is zero beyond the
    # training knot interior and the Riesz weight concentrates on that row.
    a, x, y = make_data(49, F(0), F(0), F(1), outlier=True)
    cases.append(build("outlier_beyond_knots", a, x, y, 2, 5, F(1), 0.95, 1.0))
    a, x, y = make_data(48, F(0), F(0), F(1), a_noise=F(1, 8))
    cases.append(build("treatment_strongly_determined", a, x, y, 4, 5, F(1, 2), 0.99, 1.0))
    # No adjusters: the treatment is its own training mean with the n - 1 divisor.
    a, _, y = make_data(48, F(0), F(0), F(1))
    cases.append(build("no_adjusters", a, None, y, 4, 5, F(1, 2), 0.95, 1.0))
    # A threshold above the Kish size of the first design, so the weak-overlap warning must fire.
    a, x, y = make_data(60, F(0), F(0), F(1))
    cases.append(build("weak_overlap_threshold", a, x, y, 3, 5, F(1), 0.95, 59.0))
    print(json.dumps({"cases": cases}, indent=1))
