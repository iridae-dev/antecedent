# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Exact Student-t reference for the Bayesian cohort-by-period event-time contrasts.

Regenerate with:  uv run --script bayesian_event_study_reference.py > bayesian_event_study_reference.json

A balanced panel of subjects over periods 1..5 with never-treated subjects (cohort 0) and two
adoption cohorts (3 and 4).  For a cohort g and every period t != g - 1 the contrast is the
group coefficient of the change score y_t - y_{g-1} on (1, 1{cohort = g}) over the never-treated
subjects and that cohort only, with b | s2 ~ N(0, s2 c^2 I) and s2 ~ InvGamma(a0, b0).  The
group coefficient has a Student-t marginal with 2 a0 + n degrees of freedom, location
(L^-1 X'y)_1 and squared scale (b/a) (L^-1)_11 for L = X'X + I / c^2, b = b0 + (|y - X m|^2 +
|m|^2 / c^2) / 2.  Everything is mpmath at 40 digits from the exact binary64 outcomes.
Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import json
import mpmath as mp

mp.mp.dps = 40
PERIODS = [1, 2, 3, 4, 5]
PRIOR = (3.0, 2.0, 1.5)


def t_cdf(t, nu):
    t, nu = mp.mpf(t), mp.mpf(nu)
    x = nu / (nu + t * t)
    tail = mp.betainc(nu / 2, mp.mpf(1) / 2, 0, x, regularized=True) / 2
    return 1 - tail if t > 0 else tail


def outcome(i, cohort, t):
    wob = (((i * 13 + t * 5) % 17) / 17.0 - 0.5)
    base = 1.0 + 0.2 * (i % 4) + 0.35 * t
    effect = 0.0
    if cohort and t >= cohort:
        effect = (1.5 if cohort == 3 else -0.5) * (t - cohort + 1)
    return base + effect + wob


subjects = []
for i in range(8):
    subjects.append((f"n{i}", 0))
for i in range(6):
    subjects.append((f"a{i}", 3))
for i in range(5):
    subjects.append((f"b{i}", 4))
rows = []
for k, (name, cohort) in enumerate(subjects):
    for t in PERIODS:
        rows.append(dict(subject=name, period=t, cohort=cohort, y=outcome(k, cohort, t)))
by = {name: {r["period"]: r["y"] for r in rows if r["subject"] == name} for name, _ in subjects}


def contrast(cohort, period):
    base = cohort - 1
    ctrl = [n for n, c in subjects if c == 0]
    trt = [n for n, c in subjects if c == cohort]
    y = [mp.mpf(by[n][period]) - mp.mpf(by[n][base]) for n in ctrl + trt]
    g = [0] * len(ctrl) + [1] * len(trt)
    n = len(y)
    X = mp.matrix([[1, gi] for gi in g])
    Y = mp.matrix(y)
    c, a0, b0 = map(mp.mpf, PRIOR)
    L = X.T * X + mp.eye(2) / c**2
    Li = L ** -1
    m = Li * (X.T * Y)
    e = Y - X * m
    a = a0 + mp.mpf(n) / 2
    b = b0 + ((e.T * e)[0] + (m.T * m)[0] / c**2) / 2
    scale2 = b / a * Li[1, 1]
    nu = 2 * a
    q = mp.findroot(lambda t: t_cdf(t, nu) - mp.mpf("0.975"), mp.mpf(2))
    s = mp.sqrt(scale2)
    return dict(cohort=cohort, period=period, event_time=period - cohort, treated=len(trt), comparison=len(ctrl),
                mean=float(m[1]), sd=float(mp.sqrt(scale2 * nu / (nu - 2))),
                lower=float(m[1] - q * s), upper=float(m[1] + q * s))


contrasts = [contrast(g, t) for g in (3, 4) for t in PERIODS if t != g - 1]
print(json.dumps(dict(prior=PRIOR, rows=rows, contrasts=contrasts), separators=(",", ":")))
