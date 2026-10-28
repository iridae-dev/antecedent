# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Exact Normal-Inverse-Gamma references for the three Bayesian difference-in-differences forms.

Regenerate with:  uv run --script bayesian_panel_did_reference.py > bayesian_panel_did_reference.json

Each fit is y | b, s2 ~ N(X b, s2 I), b | s2 ~ N(0, s2 c^2 I), s2 ~ InvGamma(a0, b0), whose
coefficient marginals are Student-t.  With L = X'X + I/c^2, m = L^-1 X'y, a = a0 + n/2 and
b = b0 + (|y - X m|^2 + |m|^2 / c^2) / 2 (the residual form, so nothing cancels), a linear
functional w'b has t_{2a} marginal with location w'm and squared scale (b/a) w'L^-1 w.
  * cross_section: columns 1, G, Post, G*Post; the effect is the interaction coefficient.
  * panel: y = post - pre on 1, G; the effect is the G coefficient.
  * augmented: each arm has its own fit of the change score on 1 and the covariates; the effect is
    the treated-arm functional at the treated covariate mean minus the comparison-arm functional at
    that mean, a difference of two independent Student-t variables; its mean and variance are
    exact, and its equal-tailed 95% interval is found by numerical convolution of the two
    distribution functions.
All arithmetic is in 40-digit mpmath from the exact binary64 inputs.  Repository-authored
(MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import json
import mpmath as mp

mp.mp.dps = 40


def noise(i):
    return ((i * 7) % 11) / 10.0 - 0.5


def t_cdf(t, nu):
    t, nu = mp.mpf(t), mp.mpf(nu)
    x = nu / (nu + t * t)
    tail = mp.betainc(nu / 2, mp.mpf(1) / 2, 0, x, regularized=True) / 2
    return 1 - tail if t > 0 else tail


def t_pdf(t, nu):
    t, nu = mp.mpf(t), mp.mpf(nu)
    return mp.gamma((nu + 1) / 2) / (mp.sqrt(nu * mp.pi) * mp.gamma(nu / 2)) * (1 + t * t / nu) ** (-(nu + 1) / 2)


def nig(cols, y, c, a0, b0, w):
    n, p = len(y), len(cols)
    X = mp.matrix(n, p)
    for j, col in enumerate(cols):
        for i in range(n):
            X[i, j] = mp.mpf(col[i])
    Y = mp.matrix([mp.mpf(v) for v in y])
    L = X.T * X + mp.eye(p) / mp.mpf(c) ** 2
    Li = L ** -1
    m = Li * (X.T * Y)
    e = Y - X * m
    a = mp.mpf(a0) + mp.mpf(n) / 2
    b = mp.mpf(b0) + ((e.T * e)[0] + (m.T * m)[0] / mp.mpf(c) ** 2) / 2
    W = mp.matrix([mp.mpf(v) for v in w])
    loc = (W.T * m)[0]
    scale2 = b / a * (W.T * Li * W)[0]
    return loc, scale2, 2 * a


def summarise(loc, scale2, nu):
    var = scale2 * nu / (nu - 2)
    s = mp.sqrt(scale2)
    q = mp.findroot(lambda t: t_cdf(t, nu) - mp.mpf("0.975"), mp.mpf(2))
    return dict(mean=float(loc), sd=float(mp.sqrt(var)), lower=float(loc - s * q), upper=float(loc + s * q))


def diff_summary(t1, t0):
    (l1, s1, n1), (l0, s0, n0) = t1, t0
    mean = l1 - l0
    var = s1 * n1 / (n1 - 2) + s0 * n0 / (n0 - 2)
    sd1, sd0 = mp.sqrt(s1), mp.sqrt(s0)

    # D = X1 - X0: P(D <= z) = integral f1(x) (1 - F0(x - z)) dx on standardised variables.
    def cdf(z):
        f = lambda u: t_pdf(u, n1) * (1 - t_cdf(((l1 + sd1 * u) - z - l0) / sd0, n0))
        return mp.quad(f, [-mp.inf, -20, -5, 0, 5, 20, mp.inf])

    lo = mp.findroot(lambda z: cdf(z) - mp.mpf("0.025"), mean - 2 * mp.sqrt(var), tol=1e-25)
    hi = mp.findroot(lambda z: cdf(z) - mp.mpf("0.975"), mean + 2 * mp.sqrt(var), tol=1e-25)
    return dict(mean=float(mean), sd=float(mp.sqrt(var)), lower=float(lo), upper=float(hi))


def cross_section(prior):
    y, g, post = [], [], []
    i = 0
    for gg in (0, 1):
        for pp in (0, 1):
            for _ in range(6):
                y.append(2.0 + gg + 0.5 * pp + 1.5 * gg * pp + noise(i))
                g.append(gg)
                post.append(pp)
                i += 1
    cols = [[1.0] * len(y), g, post, [a * b for a, b in zip(g, post)]]
    out = summarise(*nig(cols, y, *prior, [0, 0, 0, 1]))
    return dict(y=y, group=[bool(v) for v in g], post=[bool(v) for v in post], **out)


def panel_data():
    pre, post, tr, x = [], [], [], []
    for j in range(24):
        t = j >= 12
        xj = ((j * 5) % 7) / 3.0
        before = 1.0 + 0.3 * (j % 3) + noise(j)
        pre.append(before)
        post.append(before + 0.7 + 2.0 * t + 0.8 * xj + noise(j + 3))
        tr.append(t)
        x.append(xj)
    return pre, post, tr, x


def panel(prior):
    pre, post, tr, x = panel_data()
    change = [b - a for a, b in zip(pre, post)]
    out = summarise(*nig([[1.0] * 24, [float(t) for t in tr]], change, *prior, [0, 1]))
    return dict(pre=pre, post=post, treated=tr, **out)


def augmented(prior):
    pre, post, tr, x = panel_data()
    # A second covariate that is not collinear with the first.
    z = [((j * 3) % 5) / 2.0 - 1.0 for j in range(24)]
    change = [b - a for a, b in zip(pre, post)]
    xm = sum(xj for xj, t in zip(x, tr) if t) / sum(tr)
    zm = sum(zj for zj, t in zip(z, tr) if t) / sum(tr)
    arms = []
    for arm in (False, True):
        idx = [i for i in range(24) if tr[i] == arm]
        cols = [[1.0] * len(idx), [x[i] for i in idx], [z[i] for i in idx]]
        arms.append(nig(cols, [change[i] for i in idx], *prior, [1, xm, zm]))
    out = diff_summary(arms[1], arms[0])
    return dict(pre=pre, post=post, treated=tr, covariates=[x, z], **out)


cases = []
for name, prior in (("diffuse", (100.0, 1e-3, 1e-3)), ("informative", (0.8, 3.0, 2.0))):
    for form, fn in (("cross_section", cross_section), ("panel", panel), ("augmented", augmented)):
        cases.append(dict(name=f"{form}_{name}", form=form, coefficient_scale=prior[0],
                          variance_shape=prior[1], variance_scale=prior[2], **fn(prior)))
print(json.dumps(cases, separators=(",", ":")))
