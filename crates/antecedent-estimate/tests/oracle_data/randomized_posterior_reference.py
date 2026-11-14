# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Exact arm-contrast laws of the Bayesian randomized-trial posterior.

Regenerate with:  uv run --script randomized_posterior_reference.py > randomized_posterior_reference.json

Binary outcomes, arm k with s_k successes of m_k units under a Beta(a0, b0) prior:
  * superpopulation: theta_k ~ Beta(a0 + s_k, b0 + m_k - s_k) independently; for a contrast with
    exactly two nonzero weights a (positive) and -b the law of D = a theta_1 - b theta_0 has
    P(D <= z) = integral f_0(y) F_1((z + b y) / a) dy, evaluated by adaptive quadrature at 30 digits;
  * finite population: published under additive unit effects, where the contrast of the arm means
    over the trial's own units is the contrast of the theta_k, so it has the superpopulation law; the
    contrast theta_0 - (theta_1 + theta_2) / 2 has
    P(D <= z) = double integral f_1(y) f_2(x) F_0(z + (x + y) / 2) dy dx.
Gaussian outcomes, arm k with n_k units, mean ybar_k and sum of squares S_k under the
Normal-Inverse-Gamma prior (mu0, kappa0, a0, b0):
  kappa = kappa0 + n_k, mu = (kappa0 mu0 + n_k ybar) / kappa, a = a0 + n_k / 2,
  b = b0 + S_k / 2 + kappa0 n_k (ybar - mu0)^2 / (2 kappa);
  * superpopulation: mu_k ~ t_{2a}(mu, b / (a kappa));
  * finite population: the same law, under additive unit effects.
The contrast of two arms is a difference of independent Student variables, integrated like the Beta
case.  Everything is mpmath at 30 digits from the exact binary64 inputs.
Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import json
import mpmath as mp

mp.mp.dps = 30


def t_cdf(t, nu):
    t, nu = mp.mpf(t), mp.mpf(nu)
    x = nu / (nu + t * t)
    tail = mp.betainc(nu / 2, mp.mpf(1) / 2, 0, x, regularized=True) / 2
    return 1 - tail if t > 0 else tail


def t_pdf(t, nu):
    t, nu = mp.mpf(t), mp.mpf(nu)
    return mp.gamma((nu + 1) / 2) / (mp.sqrt(nu * mp.pi) * mp.gamma(nu / 2)) * (1 + t * t / nu) ** (-(nu + 1) / 2)


class Beta:
    def __init__(self, a, b): self.a, self.b = mp.mpf(a), mp.mpf(b)
    def cdf(self, x):
        x = mp.mpf(x)
        return mp.mpf(0) if x <= 0 else mp.mpf(1) if x >= 1 else mp.betainc(self.a, self.b, 0, x, regularized=True)
    def pdf(self, x): return mp.mpf(x) ** (self.a - 1) * (1 - mp.mpf(x)) ** (self.b - 1) / mp.beta(self.a, self.b)
    mean = property(lambda s: s.a / (s.a + s.b))
    var = property(lambda s: s.a * s.b / ((s.a + s.b) ** 2 * (s.a + s.b + 1)))
    support = (mp.mpf(0), mp.mpf(1))
    pts = [mp.mpf(0), mp.mpf(1)]


class Student:
    def __init__(self, loc, scale, nu): self.loc, self.scale, self.nu = mp.mpf(loc), mp.mpf(scale), mp.mpf(nu)
    def cdf(self, x): return t_cdf((mp.mpf(x) - self.loc) / self.scale, self.nu)
    def pdf(self, x): return t_pdf((mp.mpf(x) - self.loc) / self.scale, self.nu) / self.scale
    mean = property(lambda s: s.loc)
    var = property(lambda s: s.scale ** 2 * s.nu / (s.nu - 2))
    support = (-mp.inf, mp.inf)
    pts = property(lambda s: [-mp.inf, s.loc - 20 * s.scale, s.loc - 5 * s.scale, s.loc, s.loc + 5 * s.scale, s.loc + 20 * s.scale, mp.inf])


def diff_law(x1, x0):
    """Law of X1 - X0 for independent X1, X0 from P(D <= z) = int f_0(y) F_1(z + y) dy."""
    cdf = lambda z: mp.quad(lambda y: x0.pdf(y) * x1.cdf(z + y), x0.pts)
    mean = x1.mean - x0.mean
    sd = mp.sqrt(x1.var + x0.var)
    lo = mp.findroot(lambda z: cdf(z) - mp.mpf("0.025"), mean - 2 * sd, tol=1e-18)
    hi = mp.findroot(lambda z: cdf(z) - mp.mpf("0.975"), mean + 2 * sd, tol=1e-18)
    return dict(mean=float(mean), sd=float(sd), lower=float(lo), upper=float(hi), positive=float(1 - cdf(mp.mpf(0))))


def arm_binary(prior, s, m):
    return Beta(prior[0] + s, prior[1] + m - s)


def three_arm_law(lead, a, b):
    """Law of lead - a / 2 - b / 2 for independent Beta arms, by iterated quadrature at 20 digits."""
    def cdf(z):
        def inner(x):
            return mp.quad(lambda y: a.pdf(y) * lead.cdf(z + (x + y) / 2), a.pts)
        return mp.quad(lambda x: b.pdf(x) * inner(x), b.pts)
    mean = lead.mean - (a.mean + b.mean) / 2
    sd = mp.sqrt(lead.var + (a.var + b.var) / 4)
    with mp.workdps(20):
        lo = mp.findroot(lambda z: cdf(z) - mp.mpf("0.025"), mean - 2 * sd, tol=1e-12)
        hi = mp.findroot(lambda z: cdf(z) - mp.mpf("0.975"), mean + 2 * sd, tol=1e-12)
        positive = 1 - cdf(mp.mpf(0))
    return dict(mean=float(mean), sd=float(sd), lower=float(lo), upper=float(hi), positive=float(positive))


def nig_arm(prior, ys):
    mu0, k0, a0, b0 = map(mp.mpf, prior)
    n = len(ys)
    ybar = sum(mp.mpf(v) for v in ys) / n
    S = sum((mp.mpf(v) - ybar) ** 2 for v in ys)
    kappa = k0 + n
    mu = (k0 * mu0 + n * ybar) / kappa
    a = a0 + mp.mpf(n) / 2
    b = b0 + S / 2 + k0 * n * (ybar - mu0) ** 2 / (2 * kappa)
    return dict(n=n, sum=n * ybar, mu=mu, a=a, b=b, kappa=kappa)


def gaussian_laws(arms):
    return [Student(arm["mu"], mp.sqrt(arm["b"] / (arm["a"] * arm["kappa"])), 2 * arm["a"]) for arm in arms]


cases = []

# Binary, three arms, unequal sizes.
prior = (1.0, 2.0)
arms = [(7, 12), (4, 10), (9, 11)]
outcomes, assignment = [], []
for k, (s, m) in enumerate(arms):
    outcomes += [1.0] * s + [0.0] * (m - s)
    assignment += [k] * m
for est, finite in (("superpopulation", False), ("finite_population", True)):
    for weights in ([1.0, -1.0, 0.0], [0.0, 1.0, -1.0]) if not finite else ([1.0, -1.0, 0.0], [1.0, -0.5, -0.5], [0.0, 1.0, -1.0]):
        if -0.5 in weights:
            res = three_arm_law(*(arm_binary(prior, *arms[k]) for k in (0, 1, 2)))
        else:
            i1, i0 = weights.index(1.0), weights.index(-1.0)
            res = diff_law(arm_binary(prior, *arms[i1]), arm_binary(prior, *arms[i0]))
        cases.append(dict(model="bernoulli", prior=list(prior), estimand=est, outcome=outcomes, assignment=assignment,
                          weights=weights, arm_means=[float(arm_binary(prior, *a).mean) for a in arms],
                          arm_sds=[float(mp.sqrt(arm_binary(prior, *a).var)) for a in arms], **res))

# Gaussian, two arms, unequal sizes and spreads.
gprior = (0.5, 2.0, 3.0, 1.5)
ys = [[2.1, 3.4, 2.9, 4.2, 3.1, 2.6, 3.8, 2.2], [3.9, 5.5, 4.4, 6.1, 5.0, 4.7, 5.9, 6.4, 4.1, 5.2, 5.8, 4.9]]
outcomes = ys[0] + ys[1]
assignment = [0] * len(ys[0]) + [1] * len(ys[1])
garms = [nig_arm(gprior, y) for y in ys]
for est, finite in (("superpopulation", False), ("finite_population", True)):
    laws = gaussian_laws(garms)
    res = diff_law(laws[1], laws[0])
    cases.append(dict(model="gaussian", prior=list(gprior), estimand=est, outcome=outcomes, assignment=assignment,
                      weights=[-1.0, 1.0], arm_means=[float(a["mu"]) for a in garms],
                      arm_sds=[float(mp.sqrt(a["b"] / (a["a"] * a["kappa"]) * 2 * a["a"] / (2 * a["a"] - 2))) for a in garms], **res))
print(json.dumps(cases, separators=(",", ":")))
