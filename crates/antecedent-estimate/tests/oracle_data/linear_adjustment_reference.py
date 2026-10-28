# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Exact OLS and ridge adjusted treatment effect, standard errors and influence, in 40 digits.

Regenerate with:  uv run --script linear_adjustment_reference.py > linear_adjustment_reference.json

Design X = (1, t, z1, z2) (n rows, p = 4), outcome y with noise whose scale grows with |z1|, and a
declared treatment gap delta.  OLS: b = (X'X)^-1 X'y, residuals e, effect delta b_t and
    homoskedastic  se = |delta| sqrt( e'e / (n - p) * [(X'X)^-1]_tt ),
    HCk            se = |delta| sqrt( [(X'X)^-1 X' diag(w_i e_i^2) X (X'X)^-1]_tt ),
with w_i = 1 (HC0), n / (n - p) (HC1), 1 / (1 - h_i) (HC2) and 1 / (1 - h_i)^2 (HC3), where h_i is the
leverage x_i'(X'X)^-1 x_i.  The exported influence of row i is  n delta [(X'X)^-1 x_i]_t e_i.
Ridge with penalty lambda leaves the first (all-ones) column unpenalised: b = (X'X + lambda P)^-1 X'y
with P = diag(0, 1, 1, 1), residuals e = y - X b, effect delta b_t, no analytic standard error, and
the influence uses the bread (X'X + lambda P)^-1 and the ridge residuals.
Huber M-estimation with constant c = 1.345 on an outlier-contaminated copy of the outcome: iterate
b <- WLS(b) with weights min(1, c s / |r|), r = y - X b and s = 1.4826 median |r - median(r)|
(recomputed from the current residuals) until the coefficients stop moving (a 40-digit fixed
point); the effect is delta b_t.  The standard error is the homoskedastic Huber form the module
states (Huber 1981, 7.10): with scores u_i = clamp(r_i, -c s, c s), m = mean 1{|r_i| <= c s},
kappa = 1 + (p / n) (1 - m) / m and tau^2 = sum u_i^2 / (n - p) / m^2,
se = |delta| sqrt(kappa^2 tau^2 [(X'X)^-1]_tt)  with the unweighted Gram matrix.  The book was not
retrieved; this reproduces the formula as the module states it.
Repository-authored (MIT OR Apache-2.0).
"""
import json
import mpmath as mp

mp.mp.dps = 40


def data(n):
    z1 = [(((i * 7 + 3) % 19) - 9) / 9.0 for i in range(n)]
    z2 = [(((i * 5 + 1) % 13) - 6) / 6.0 for i in range(n)]
    t = [1.0 if (0.4 * z1[i] + 0.3 * z2[i] + (((i * 11) % 7) - 3) / 7.0) > 0 else 0.0 for i in range(n)]
    y = [0.7 + 1.3 * t[i] + 0.8 * z1[i] - 0.5 * z2[i]
         + (0.2 + 0.5 * abs(z1[i])) * ((((i * 13 + 2) % 17) - 8) / 8.0) for i in range(n)]
    return z1, z2, t, y


def fit(X, Y, penalty, delta):
    n, p = X.rows, X.cols
    P = mp.diag([0] + [1] * (p - 1)) * mp.mpf(penalty)
    bread = (X.T * X + P) ** -1
    b = bread * (X.T * Y)
    e = Y - X * b
    out = dict(ate=float(delta * b[1]))
    infl = []
    for i in range(n):
        xi = X[i, :].T
        infl.append(float(n * delta * (bread * xi)[1] * e[i]))
    out["influence"] = infl
    return out, bread, e


def sandwich(X, bread, e, weights, delta):
    n = X.rows
    meat = mp.zeros(X.cols, X.cols)
    for i in range(n):
        xi = X[i, :].T
        meat += weights[i] * e[i] ** 2 * (xi * xi.T)
    return float(abs(delta) * mp.sqrt((bread * meat * bread)[1, 1]))


n, delta = 60, 2.0
z1, z2, t, y = data(n)
X = mp.matrix([[1, mp.mpf(t[i]), mp.mpf(z1[i]), mp.mpf(z2[i])] for i in range(n)])
Y = mp.matrix([mp.mpf(v) for v in y])
ols, bread, e = fit(X, Y, 0, mp.mpf(delta))
p = X.cols
lev = [(X[i, :] * bread * X[i, :].T)[0] for i in range(n)]
ols["se"] = dict(
    homoskedastic=float(abs(delta) * mp.sqrt((e.T * e)[0] / (n - p) * bread[1, 1])),
    hc0=sandwich(X, bread, e, [1] * n, delta),
    hc1=sandwich(X, bread, e, [mp.mpf(n) / (n - p)] * n, delta),
    hc2=sandwich(X, bread, e, [1 / (1 - h) for h in lev], delta),
    hc3=sandwich(X, bread, e, [1 / (1 - h) ** 2 for h in lev], delta))
ridge = []
for lam in (0.5, 20.0):
    r, _, _ = fit(X, Y, lam, mp.mpf(delta))
    ridge.append(dict(lambda_=lam, **r))

def median(values):
    v = sorted(values)
    k = len(v)
    return v[k // 2] if k % 2 else (v[k // 2 - 1] + v[k // 2]) / 2


def huber(X, Y, c, delta):
    n, p = X.rows, X.cols
    b = (X.T * X) ** -1 * (X.T * Y)
    for _ in range(400):
        r = [(Y - X * b)[i] for i in range(n)]
        m0 = median(r)
        s = mp.mpf("1.4826") * median([abs(v - m0) for v in r])
        w = [mp.mpf(1) if abs(v) <= c * s else c * s / abs(v) for v in r]
        W = mp.diag(w)
        new = (X.T * W * X) ** -1 * (X.T * W * Y)
        if max(abs(new[i] - b[i]) for i in range(p)) < mp.mpf(10) ** -35:
            b = new
            break
        b = new
    r = [(Y - X * b)[i] for i in range(n)]
    m0 = median(r)
    s = mp.mpf("1.4826") * median([abs(v - m0) for v in r])
    bound = c * s
    scores = [max(-bound, min(bound, v)) for v in r]
    m = sum(1 for v in r if abs(v) <= bound) / mp.mpf(n)
    kappa = 1 + mp.mpf(p) / n * (1 - m) / m
    tau2 = sum(u * u for u in scores) / (n - p) / m ** 2
    inv = (X.T * X) ** -1
    return dict(ate=float(delta * b[1]), se=float(abs(delta) * mp.sqrt(kappa ** 2 * tau2 * inv[1, 1])),
                scale=float(s), unclipped=float(m))


y_out = [v + (7.0 if i % 11 == 0 else 0.0) for i, v in enumerate(y)]
Yo = mp.matrix([mp.mpf(v) for v in y_out])
hub = huber(X, Yo, mp.mpf("1.345"), mp.mpf(delta))
print(json.dumps(dict(delta=delta, treatment=t, z1=z1, z2=z2, outcome=y, ols=ols, ridge=ridge,
                      outcome_contaminated=y_out, huber=hub), separators=(",", ":")))
