# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath"]
# ///
"""Front-door plug-in functional and its influence-function standard error, from the definition.

Regenerate with:  uv run --script frontdoor_functional_reference.py > frontdoor_functional_reference.json

The front-door interventional mean of a discrete treatment is
psi_a = sum_m P(m | T=a) sum_t' mu(m, t') P(T=t'), mu(m, t') = E[Y | M=m, T=t'].  Every
quantity is written as a function of per-row masses w_i (arm mass, cell and arm outcome means, or
per-arm weighted least squares), so the functional is a smooth map of the empirical measure.  The
effect is psi_active - psi_control at unit masses.  The influence of row i is n times the
derivative of the effect with respect to its mass, taken here by a central difference with step
1e-25 at 80 digits (the error is of order 1e-50), and the standard error is sqrt(sum of the squared
influences)/n.  Saturated: mu is the cell mean in each arm; arm-linear: mu is the per-arm weighted
least squares of Y on [1, M].  Values are written as nearest binary64.
Repository-authored (MIT OR Apache-2.0).
"""
import json
from mpmath import mp, mpf, matrix, lu_solve, sqrt

mp.dps = 80
_state = 0x3C6EF372FE94F82B


def u01():
    global _state
    _state = (_state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
    return (_state >> 11) / float(1 << 53)


def wls(rows, y, w):
    p = len(rows[0])
    A, b = matrix(p, p), matrix(p, 1)
    for r, v, wi in zip(rows, y, w):
        for a in range(p):
            b[a] += wi * r[a] * v
            for c in range(p):
                A[a, c] += wi * r[a] * r[c]
    beta = lu_solve(A, b)
    return lambda r: sum(beta[k] * r[k] for k in range(p))


def effect(t, y, m, w, active, control, model):
    n = len(t)
    arms = sorted(set(t))
    total = sum(w)
    arm_mass = {a: sum(w[i] for i in range(n) if t[i] == a) for a in arms}
    p = {a: arm_mass[a] / total for a in arms}
    if model == "saturated":
        cells = {}
        for i in range(n):
            key = (tuple(m[i]), t[i])
            c = cells.setdefault(key, [mpf(0), mpf(0)])
            c[0] += w[i] * y[i]
            c[1] += w[i]
        def mu(i, a):
            c = cells[(tuple(m[i]), a)]
            return c[0] / c[1]
    else:
        fits = {}
        for a in arms:
            rows = [i for i in range(n) if t[i] == a]
            fits[a] = wls([[mpf(1)] + [mpf(v) for v in m[i]] for i in rows],
                          [y[i] for i in rows], [w[i] for i in rows])
        def mu(i, a):
            return fits[a]([mpf(1)] + [mpf(v) for v in m[i]])
    psi = {}
    for a in (active, control):
        psi[a] = sum(w[i] * sum(p[b] * mu(i, b) for b in arms)
                     for i in range(n) if t[i] == a) / arm_mass[a]
    return psi[active] - psi[control]


def case(name, t, y, m, active, control, model):
    n = len(t)
    yy = [mpf(v) for v in y]
    mm = [[mpf(v) for v in row] for row in m]
    one = [mpf(1)] * n
    ate = effect(t, yy, mm, one, active, control, model)
    h = mpf(10) ** -25
    ssq = mpf(0)
    for i in range(n):
        up, dn = list(one), list(one)
        up[i] += h
        dn[i] -= h
        d = (effect(t, yy, mm, up, active, control, model)
             - effect(t, yy, mm, dn, active, control, model)) / (2 * h)
        ssq += d * d
    return {"name": name, "t": t, "y": y, "m": m, "active": active, "control": control,
            "model": model, "ate": float(ate), "se": float(sqrt(ssq))}


def discrete(n, arms, levels, k=1, tilt=1.0, offset=0.0):
    t, y, m = [], [], []
    for _ in range(n):
        ti = float(min(int(u01() * arms), arms - 1))
        row = [float(min(int(min(u01() * 0.8 + 0.2 * (ti / max(arms - 1, 1)) * tilt, 0.999) * levels), levels - 1))
               + offset for _ in range(k)]
        yi = 0.5 * ti + sum((j + 1) * v for j, v in enumerate(row)) + 0.7 * ti * row[0] + (u01() - 0.5)
        t.append(ti)
        y.append(float(yi))
        m.append(row)
    return t, y, m


cases = []
t, y, m = discrete(40, 2, 2)
cases.append(case("binary_saturated", t, y, m, 1.0, 0.0, "saturated"))
t, y, m = discrete(150, 3, 2, k=2, tilt=0.4)
for a, c in ((1.0, 0.0), (2.0, 0.0), (2.0, 1.0)):
    cases.append(case(f"three_arm_two_mediator_{int(a)}_{int(c)}", t, y, m, a, c, "saturated"))
t, y, m = discrete(30, 2, 2)
t = [1.0 if i < 5 else 0.0 for i in range(30)]
m = [[float(i % 2)] for i in range(30)]
cases.append(case("tiny_treated_arm", t, y, m, 1.0, 0.0, "saturated"))
t, y, m = discrete(50, 2, 6, k=1)
cases.append(case("continuous_like_arm_linear", t, y, [[v * 0.37 + 1.0e3 for v in r] for r in m],
                  1.0, 0.0, "arm_linear"))
t, y, m = discrete(60, 3, 5, k=2)
m2 = [[r[0] * 0.5 + 0.25, r[0] * 0.5 + 0.25 + 1.0e-3 * r[1]] for r in m]
cases.append(case("near_collinear_arm_linear", t, y, m2, 2.0, 0.0, "arm_linear"))
t, y, m = discrete(45, 2, 4, k=2, offset=1.0e4)
cases.append(case("mediator_offset_arm_linear", t, y, m, 1.0, 0.0, "arm_linear"))
print(json.dumps({"cases": cases}))
