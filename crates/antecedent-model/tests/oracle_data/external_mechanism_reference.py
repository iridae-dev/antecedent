# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Equilibrium likelihood, scores, contraction bound and solve of a three-equation feedback system.

Regenerate with:  uv run --script external_mechanism_reference.py > external_mechanism_reference.json

The system (variables 0, 1, 2; parameters theta = i0, p0, p1, i1, q, i2) is
    f0(x) = i0 + p0 tanh(p1 x1)          (external, reads x1)
    f1(x) = i1 + q sin(x0)               (external, reads x0)
    f2(x) = i2 + 0.3 x0 + 0.2 x1         (built in)
with disturbances u ~ N(0, S) and x_i = f_i(x) + u_i for every free i. For a row x observed with a
set C of clamped variables (the equations replaced by their clamp values), free = all \ C and
    data = -0.5 (r' S_FF^-1 r + log det S_FF + |F| log 2 pi),  r_i = x_i - f_i(x), i in F,
    logdet = log |det(I - J_FF)|,  J_ij = d f_i / d x_j for i, j in F,
the change of variables of the equilibrium map. Everything is evaluated in mpmath at 50 digits from
the exact binary64 inputs; scores are central differences of these two terms in each parameter at
50 digits (step 1e-15 of the scale), so they carry no analytic derivation shared with the product.
The contraction bound is the largest row sum of the declared Lipschitz bounds restricted to each
strongly connected block of the free system (blocks decided by hand for each clamp set below).
Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import json

import mpmath as mp

mp.mp.dps = 50

THETA = {"i0": 0.1, "p0": 0.5, "p1": 0.9, "i1": -0.2, "q": 0.5, "i2": 0.3}
ORDER = ["i0", "p0", "p1", "i1", "q", "i2"]
DECLARED = {0: {1: 0.45}, 1: {0: 0.5}, 2: {0: 0.3, 1: 0.2}}

COVS = {
    "correlated": [[1.0, 0.3, 0.1], [0.3, 0.8, -0.2], [0.1, -0.2, 1.5]],
    "near_singular_pair": [[1.0, 0.9999, 0.0], [0.9999, 1.0, 0.0], [0.0, 0.0, 1.0]],
    "noise_sd_1e-3": [[1e-6, 3e-7, 1e-7], [3e-7, 8e-7, -2e-7], [1e-7, -2e-7, 1.5e-6]],
}
# free variables -> strongly connected blocks of the free system
BLOCKS = {
    (): [[0, 1], [2]],
    (0,): [[1], [2]],
    (1,): [[0], [2]],
    (2,): [[0, 1]],
    (0, 1): [[2]],
}


def fs(theta, x):
    t = {k: mp.mpf(v) for k, v in theta.items()}
    x = [mp.mpf(v) for v in x]
    return [
        t["i0"] + t["p0"] * mp.tanh(t["p1"] * x[1]),
        t["i1"] + t["q"] * mp.sin(x[0]),
        t["i2"] + mp.mpf("0.3") * x[0] + mp.mpf("0.2") * x[1],
    ]


def jac(theta, x):
    t = {k: mp.mpf(v) for k, v in theta.items()}
    x = [mp.mpf(v) for v in x]
    sech2 = 1 / mp.cosh(t["p1"] * x[1]) ** 2
    return [
        [0, t["p0"] * t["p1"] * sech2, 0],
        [t["q"] * mp.cos(x[0]), 0, 0],
        [mp.mpf("0.3"), mp.mpf("0.2"), 0],
    ]


def terms(theta, rows, cov):
    data = mp.mpf(0)
    logdet = mp.mpf(0)
    for x, clamps in rows:
        free = [i for i in range(3) if i not in clamps]
        if not free:
            continue
        f = fs(theta, x)
        r = mp.matrix([mp.mpf(x[i]) - f[i] for i in free])
        S = mp.matrix([[mp.mpf(cov[i][j]) for j in free] for i in free])
        data += -mp.mpf("0.5") * ((r.T * mp.lu_solve(S, r))[0] + mp.log(mp.det(S)) + len(free) * mp.log(2 * mp.pi))
        J = jac(theta, x)
        M = mp.matrix([[(1 if a == b else 0) - J[i][j] for b, j in enumerate(free)] for a, i in enumerate(free)])
        logdet += mp.log(abs(mp.det(M)))
    return data, logdet


def scores(rows, cov, which):
    out = []
    for name in which:
        def g(t, name=name, which_term=None):
            th = dict(THETA)
            th[name] = mp.mpf(THETA[name]) + t
            return terms(th, rows, cov)
        d = mp.diff(lambda t: g(t)[0], 0, h=mp.mpf("1e-12"))
        e = mp.diff(lambda t: g(t)[1], 0, h=mp.mpf("1e-12"))
        out.append((float(d), float(e)))
    return out


def bound(clamps):
    worst = mp.mpf(0)
    for block in BLOCKS[tuple(sorted(clamps))]:
        for i in block:
            worst = max(worst, sum(mp.mpf(b) for j, b in DECLARED[i].items() if j in block))
    return float(worst)


def solve(u, clamps):
    free = [i for i in range(3) if i not in clamps]

    def build(xf):
        x = [mp.mpf(clamps[i]) if i in clamps else None for i in range(3)]
        for k, i in enumerate(free):
            x[i] = xf[k]
        return x

    def resid(*xf):
        x = build(list(xf))
        f = fs(THETA, x)
        return [x[i] - f[i] - mp.mpf(u[i]) for i in free]

    root = mp.findroot(resid, [mp.mpf(0)] * len(free), tol=mp.mpf(10) ** -40, maxsteps=200)
    root = [root[i] for i in range(len(free))] if hasattr(root, 'rows') else [root]
    return [float(v) for v in build(root)]


ROWSETS = {
    "no_clamps": [([0.4, -0.3, 0.9], ()), ([-1.2, 0.8, -0.1], ()), ([2.5, 1.7, 0.3], ()), ([0.0, 0.0, 0.0], ())],
    "clamp_1": [([0.4, -0.3, 0.9], (1,)), ([-1.2, 0.8, -0.1], (1,)), ([2.5, 1.7, 0.3], (1,))],
    "clamp_2": [([0.4, -0.3, 0.9], (2,)), ([-1.2, 0.8, -0.1], (2,))],
    "clamp_0_and_1": [([0.4, -0.3, 0.9], (0, 1)), ([-1.2, 0.8, -0.1], (0, 1))],
    "saturated_tanh": [([3.0, 50.0, 1.0], ()), ([-2.0, -80.0, 0.5], ())],
    "offset_1e3": [([1000.0, 1000.5, -999.0], ()), ([-1000.0, 999.5, 1001.0], (1,))],
}
# the near-singular and tiny-scale covariances are evaluated on the unclamped rows only
COV_ROWSETS = {
    "correlated": list(ROWSETS),
    "near_singular_pair": ["no_clamps", "clamp_2"],
    "noise_sd_1e-3": ["no_clamps", "clamp_1"],
}

likelihood = []
for cname, sets in COV_ROWSETS.items():
    for rname in sets:
        rows = [(x, c) for x, c in ROWSETS[rname]]
        data, logdet = terms(THETA, rows, COVS[cname])
        entry = {
            "cov": cname,
            "rows": rname,
            "data": float(data),
            "logdet": float(logdet),
            "data_scale": float(abs(data) + abs(logdet)),
        }
        if cname == "correlated":
            sc = scores(rows, COVS[cname], ORDER)
            entry["data_score"] = [s[0] for s in sc]
            entry["det_score"] = [s[1] for s in sc]
        likelihood.append(entry)

bounds = []
for clamps in BLOCKS:
    bounds.append({"clamps": list(clamps), "bound": bound(clamps)})

solves = []
for u, clamps in [
    ([0.4, -0.3, 0.9], {}),
    ([-1.5, 2.0, 0.1], {}),
    ([0.4, -0.3, 0.9], {1: 0.7}),
    ([0.4, -0.3, 0.9], {0: -0.5}),
    ([0.4, -0.3, 0.9], {2: 1.1}),
    ([0.4, -0.3, 0.9], {0: 0.2, 1: -0.4}),
    ([30.0, -20.0, 5.0], {}),
]:
    solves.append({"u": u, "clamps": {str(k): v for k, v in clamps.items()}, "x": solve(u, clamps)})

out = {
    "theta": THETA,
    "order": ORDER,
    "covs": COVS,
    "rowsets": {k: [{"x": x, "clamps": list(c)} for x, c in v] for k, v in ROWSETS.items()},
    "likelihood": likelihood,
    "bounds": bounds,
    "solves": solves,
}
print(json.dumps(out, allow_nan=False))
