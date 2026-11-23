# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Exact conjugate Gaussian posterior of the joint source-target transport model, in rationals.

Regenerate with:  uv run --script joint_bayesian_transport_reference.py > joint_bayesian_transport_reference.json

Model.  Source s has units i with treatment a, covariates x (p columns), outcome y and a known noise
variance sigma_s^2:  y = w'theta + v'gamma_s + noise, with  w = [a, a*phi(x), phi(x)]  (the
covariate main effects are invariant: varying block "intercept", v = [1]) or  w = [a, a*phi(x)] with
v = [1, phi(x)]  (varying block "intercept_and_covariates"), phi(x) = x for this record.  The
varying block is one per source or one shared by all sources.  The prior is Gaussian with a dense
covariance on theta and on each varying block, independent across blocks.  The posterior precision is
P0 + sum_s Z_s' Z_s / sigma_s^2 and the mean is the precision inverse times
P0 m0 + sum_s Z_s' y_s / sigma_s^2; both are solved here by exact Gauss-Jordan elimination in
Fractions on dyadic inputs (nothing is rounded before the final conversion).  The effect of a sample
is c' theta with c = [1, mean phi over the sample, 0 ...]; the effects listed are every source then
the target, with their exact joint mean and covariance.  Python standard library only.
Repository-authored (MIT OR Apache-2.0).
"""
import json
from fractions import Fraction as F


def inverse(a):
    n = len(a)
    m = [row[:] + [F(int(i == j)) for j in range(n)] for i, row in enumerate(a)]
    for c in range(n):
        p = next(r for r in range(c, n) if m[r][c] != 0)
        m[c], m[p] = m[p], m[c]
        d = m[c][c]
        m[c] = [x / d for x in m[c]]
        for r in range(n):
            if r != c and m[r][c] != 0:
                f = m[r][c]
                m[r] = [x - f * y for x, y in zip(m[r], m[c])]
    return [row[n:] for row in m]


def out(x):
    return {"value": float(x), "exact": f"{x.numerator}/{x.denominator}"}


def block_diag(blocks):
    n = sum(len(b) for b in blocks)
    m = [[F(0)] * n for _ in range(n)]
    o = 0
    for b in blocks:
        for i, row in enumerate(b):
            for j, v in enumerate(row):
                m[o + i][o + j] = F(v)
        o += len(b)
    return m


def phi(x, degree):
    row = []
    for v in x:
        pw = F(1)
        for _ in range(degree):
            pw *= F(v)
            row.append(pw)
    return row


def design(name, degree, varying, shared, inv_mean, inv_cov, var_mean, var_cov, sources, target):
    """sources: list of dict(a, y, x (list of columns), noise); target: dict(x columns)."""
    p = len(sources[0]["x"])
    m = p * degree
    q = 1 + m + (m if varying == "intercept" else 0)
    r = 1 if varying == "intercept" else 1 + m
    blocks = 1 if shared else len(sources)
    dim = q + blocks * r
    prior_cov = block_diag([inv_cov] + [var_cov] * blocks)
    prior_mean = [F(v) for v in inv_mean] + [F(v) for v in var_mean] * blocks
    p0 = inverse(prior_cov)
    prec = [row[:] for row in p0]
    rhs = [sum(p0[i][j] * prior_mean[j] for j in range(dim)) for i in range(dim)]
    means = []
    for s, src in enumerate(sources):
        n = len(src["y"])
        off = q + (0 if shared else s) * r
        sig = F(src["noise"])
        acc = [F(0)] * m
        for i in range(n):
            f = phi([col[i] for col in src["x"]], degree)
            acc = [u + v for u, v in zip(acc, f)]
            a = F(int(src["a"][i]))
            z = [F(0)] * dim
            z[0] = a
            for k in range(m):
                z[1 + k] = a * f[k]
            if varying == "intercept":
                for k in range(m):
                    z[1 + m + k] = f[k]
                z[off] = F(1)
            else:
                z[off] = F(1)
                for k in range(m):
                    z[off + 1 + k] = f[k]
            for j in range(dim):
                if z[j] != 0:
                    rhs[j] += z[j] * F(src["y"][i]) / sig
                    for k in range(dim):
                        if z[k] != 0:
                            prec[j][k] += z[j] * z[k] / sig
        means.append([v / n for v in acc])
    cov = inverse(prec)
    mu = [sum(cov[i][j] * rhs[j] for j in range(dim)) for i in range(dim)]
    nt = len(target["x"][0])
    acc = [F(0)] * m
    for i in range(nt):
        acc = [u + v for u, v in zip(acc, phi([col[i] for col in target["x"]], degree))]
    means.append([v / nt for v in acc])
    cvecs = []
    for fm in means:
        c = [F(0)] * dim
        c[0] = F(1)
        for k in range(m):
            c[1 + k] = fm[k]
        cvecs.append(c)
    e = len(cvecs)
    eff_mean = [sum(c[i] * mu[i] for i in range(dim)) for c in cvecs]
    eff_cov = [[sum(a[i] * cov[i][j] * b[j] for i in range(dim) for j in range(dim)) for b in cvecs] for a in cvecs]
    return {
        "name": name, "degree": degree, "varying": varying, "shared": shared,
        "features": p, "inv_mean": [float(v) for v in inv_mean], "inv_cov": [[float(v) for v in row] for row in inv_cov],
        "var_mean": [float(v) for v in var_mean], "var_cov": [[float(v) for v in row] for row in var_cov],
        "sources": [{"a": s["a"], "y": [float(v) for v in s["y"]], "x": [[float(v) for v in col] for col in s["x"]],
                     "noise": float(s["noise"])} for s in sources],
        "target": {"x": [[float(v) for v in col] for col in target["x"]]},
        "dim": dim,
        "posterior_mean": [out(v) for v in mu],
        "posterior_covariance": [[out(v) for v in row] for row in cov],
        "effect_mean": [out(v) for v in eff_mean],
        "effect_covariance": [[out(v) for v in row] for row in eff_cov],
    }


def eye(n, v):
    return [[F(v) if i == j else F(0) for j in range(n)] for i in range(n)]


def src(n, offset, a, noise, intercept=F(1), slope=F(1, 2), effect=F(2), cols=1, step=F(1, 8), lift=F(0)):
    xs = [[lift + F((7 * i + 3 * c + offset) % 11 - 5) * step for i in range(n)] for c in range(cols)]
    ys = []
    for i in range(n):
        t = F(a[i])
        ys.append(intercept + slope * xs[0][i] + t * (effect + F(1, 4) * xs[0][i]) + F((5 * i + offset) % 7 - 3) * F(1, 16))
    return {"a": [bool(v) for v in a], "y": ys, "x": xs, "noise": noise}


alt = lambda n: [i % 2 == 0 for i in range(n)]
dense3 = [[F(4), F(1), F(0)], [F(1), F(2), F(1, 2)], [F(0), F(1, 2), F(1)]]
dense5 = [[F(3), F(1, 2), 0, 0, 0], [F(1, 2), F(2), F(1, 4), 0, 0], [0, F(1, 4), F(2), F(1, 2), 0],
          [0, 0, F(1, 2), F(5, 2), F(1, 4)], [0, 0, 0, F(1, 4), F(3, 2)]]
dense5 = [[F(v) for v in row] for row in dense5]

designs = []
# 1  two sources, independent intercepts, dense invariant prior with nonzero mean
designs.append(design("dense_prior_two_sources", 1, "intercept", False,
                      [F(1, 2), F(-1, 4), F(0)], dense3, [F(1, 4)], eye(1, 4),
                      [src(8, 0, alt(8), F(1, 16)), src(6, 3, alt(6), F(1, 8))],
                      {"x": [[F(k - 3) * F(1, 8) for k in range(9)]]}))
# 2  covariate far from zero: intercept and covariate nearly collinear within each source
designs.append(design("offset_covariate_shared_block", 1, "intercept_and_covariates", True,
                      [F(0), F(0)], eye(2, 4), [F(0), F(0)], eye(2, 4),
                      [src(8, 1, alt(8), F(1, 64), lift=F(1024)), src(7, 5, alt(7), F(1, 64), lift=F(1024))],
                      {"x": [[F(1024) + F(k - 2) * F(1, 8) for k in range(5)]]}))
# 3  extreme weights and an unbalanced pair of sources (3 rows against 12)
designs.append(design("extreme_noise_ratio_unbalanced", 1, "intercept", False,
                      [F(0)] * 3, eye(3, 4), [F(0)], eye(1, 4),
                      [src(3, 2, [True, False, True], F(1, 2 ** 20)), src(12, 7, alt(12), F(4))],
                      {"x": [[F(k - 2) * F(1, 8) for k in range(5)]]}))
# 4  near-flat prior with an offset covariate
designs.append(design("near_flat_prior_offset", 1, "intercept", False,
                      [F(0)] * 3, eye(3, 2 ** 20), [F(0)], eye(1, 2 ** 20),
                      [src(9, 4, alt(9), F(1, 16), lift=F(256)), src(5, 8, alt(5), F(1, 16), lift=F(256))],
                      {"x": [[F(256) + F(k - 2) * F(1, 8) for k in range(5)]]}))
# 5  two covariates, the second exactly twice the first (the data cannot separate them; the prior can)
s5 = src(10, 6, alt(10), F(1, 16), cols=1)
s5["x"] = [s5["x"][0], [2 * v for v in s5["x"][0]]]
designs.append(design("collinear_covariates_proper_prior", 1, "intercept", False,
                      [F(0)] * 5, dense5, [F(0)], eye(1, 4), [s5],
                      {"x": [[F(k - 2) * F(1, 8) for k in range(5)], [F(k - 2) * F(1, 4) for k in range(5)]]}))
# the data cannot separate the two covariates; the product refuses such a design by rank, although the
# proper prior would give the exact posterior listed here
designs[-1]["refused_singular"] = True
# 6  single source with two rows per arm (smallest balanced sample)
designs.append(design("tiny_single_source", 1, "intercept_and_covariates", False,
                      [F(0), F(0)], eye(2, 1), [F(0), F(0)], eye(2, 1),
                      [src(4, 9, [True, False, True, False], F(1, 4))],
                      {"x": [[F(0), F(1, 8)]]}))
print(json.dumps({"designs": designs}, separators=(",", ":")))
