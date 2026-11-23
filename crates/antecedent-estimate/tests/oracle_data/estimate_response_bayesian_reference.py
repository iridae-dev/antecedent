# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Exact Normal-Inverse-Gamma posterior of a Gaussian response level, in 40-digit arithmetic.

Regenerate with:  uv run --script estimate_response_bayesian_reference.py > estimate_response_bayesian_reference.json

Model y = X b + e, e ~ N(0, s2 I), columns (1, treatment, other treatments, adjusters); prior
b | s2 ~ N(0, s2 c0^2 I), s2 ~ Inv-Gamma(1e-3, 1e-3).  With Vn = (X'X + I / c0^2)^-1, mn = Vn X'y,
an = 1e-3 + n/2 and bn = 1e-3 + (y'y - mn'(X'X + I / c0^2) mn)/2, a linear functional w'b has a
Student-t posterior with 2 an degrees of freedom, location w'mn and scale^2 (bn / an) w'Vn w, and
variance bn / (an - 1) w'Vn w.  The level of a response at a treatment level a is w'b with
w = (1, a, mean of the other treatments' policy values, mean of every adjuster): the fitted mean at
that treatment with every other column at its sample mean, which is the average over the observed
rows of a main-effects model.  A set policy contributes its value, a shift the sample mean plus the
shift, a Bernoulli(p) policy p, a Gaussian(m, v) policy m and a categorical policy its mean index
sum_i i p_i / sum p.  Student quantiles are found by root-finding on the incomplete beta function.
Repository-authored (MIT OR Apache-2.0); needs mpmath (BSD-3-Clause) at run time, not redistributed.
"""
import json
import mpmath as mp

mp.mp.dps = 40
A0 = B0 = mp.mpf("0.001")


def t_quantile(p, nu, loc, scale):
    def cdf(x):
        u = (x - loc) / scale
        w = nu / (nu + u * u)
        tail = mp.betainc(nu / 2, mp.mpf(1) / 2, 0, w, regularized=True) / 2
        return 1 - tail if u > 0 else tail
    return mp.findroot(lambda x: cdf(x) - p, (loc - 12 * scale, loc + 12 * scale), solver="illinois", tol=mp.mpf(10) ** -30)


def dataset(n, a_off, y_off, x_scale):
    z1 = [(((i * 7) % 13) / 6.0 - 1.0) * x_scale for i in range(n)]
    z2 = [((i * 5) % 11) / 5.0 - 1.0 for i in range(n)]
    a = [a_off + 0.5 * z1[i] / x_scale + (((i * 3) % 7) - 3) / 4.0 for i in range(n)]
    b = [0.3 * z2[i] + (((i * 13) % 9) - 4) / 5.0 for i in range(n)]
    y = [y_off + 1.5 * (a[i] - a_off) - 0.7 * b[i] + 0.5 * z1[i] / x_scale - z2[i] + 0.4 * (((i * 11) % 17) - 8) / 8.0
         for i in range(n)]
    return a, b, z1, z2, y


def build(name, n, c0, two, a_off=0.0, y_off=0.0, x_scale=1.0, queries=()):
    a, b, z1, z2, y = dataset(n, a_off, y_off, x_scale)
    cols = [a] + ([b] if two else []) + [z1, z2]
    X = mp.matrix([[mp.mpf(1), *[mp.mpf(c[i]) for c in cols]] for i in range(n)])
    Y = mp.matrix([mp.mpf(v) for v in y])
    p = X.cols
    prec = X.T * X + mp.eye(p) / mp.mpf(c0) ** 2
    Vn = prec ** -1
    mn = Vn * (X.T * Y)
    an = A0 + mp.mpf(n) / 2
    bn = B0 + ((Y.T * Y)[0] - (mn.T * prec * mn)[0]) / 2
    means = [sum(mp.mpf(v) for v in c) / n for c in cols]
    out = []
    expanded = []
    for query in queries:
        if query["kind"] == "curve":
            expanded.extend(dict(kind="curve", grid_point=g, levels=[g]) for g in query["grid"])
        else:
            expanded.append(query)
    for query in expanded:
        w = [mp.mpf(1)] + means[:]
        levels = query["levels"]
        for k, level in enumerate(levels):
            if level is not None:
                w[1 + k] = mp.mpf(level)
        # `shift` entries are offsets from the sample mean
        for k, shift in enumerate(query.get("shifts", [])):
            if shift is not None:
                w[1 + k] = means[k] + mp.mpf(shift)
        wv = mp.matrix(w)
        loc = (wv.T * mn)[0]
        quad = (wv.T * Vn * wv)[0]
        scale = mp.sqrt(bn / an * quad)
        nu = 2 * an
        out.append(dict(query=query, mean=float(loc), sd=float(mp.sqrt(bn / (an - 1) * quad)),
                        lower=float(t_quantile(mp.mpf("0.025"), nu, loc, scale)),
                        upper=float(t_quantile(mp.mpf("0.975"), nu, loc, scale))))
    return dict(name=name, prior_scale=c0, two_treatments=two, a=a, b=b, z1=z1, z2=z2, y=y, results=out)


cases = [
    build("curve_and_policies", 60, 10.0, False, queries=[
        dict(kind="curve", grid=[-0.5, 0.0, 0.5, 2.0], levels=[None]),
        dict(kind="set", levels=[0.3], policy=[dict(kind="set", value=0.3)]),
        dict(kind="shift", levels=[None], shifts=[0.4], policy=[dict(kind="shift", delta=0.4)]),
        dict(kind="bernoulli", levels=[0.3], policy=[dict(kind="bernoulli", p=0.3)]),
        dict(kind="gaussian", levels=[0.15], policy=[dict(kind="gaussian", mean=0.15, var=0.04)]),
        dict(kind="categorical", levels=[1.1], policy=[dict(kind="categorical", probs=[0.2, 0.5, 0.3])]),
    ]),
    build("two_treatments_joint", 70, 3.0, True, queries=[
        dict(kind="joint", levels=[0.2, -0.4], policy=[dict(kind="set", value=0.2), dict(kind="set", value=-0.4)]),
        dict(kind="joint", levels=[0.2, None], shifts=[None, 0.5], policy=[dict(kind="set", value=0.2), dict(kind="shift", delta=0.5)]),
    ]),
    build("offsets_and_scales", 60, 10.0, False, a_off=1000.0, y_off=100000.0, x_scale=250.0, queries=[
        dict(kind="curve", grid=[999.5, 1000.0, 1000.5], levels=[None]),
    ]),
]
print(json.dumps(cases, separators=(",", ":")))
