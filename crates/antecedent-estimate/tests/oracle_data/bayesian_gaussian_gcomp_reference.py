# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Exact Normal-Inverse-Gamma posterior of the adjusted treatment coefficient, in 40-digit arithmetic.

Regenerate with:  uv run --script bayesian_gaussian_gcomp_reference.py > bayesian_gaussian_gcomp_reference.json

Model y = X b + e, e ~ N(0, s2 I), columns (1, t, covariates).  Prior b | s2 ~ N(0, s2 V0) with
V0 = c0^2 I (an absolute coefficient variance of s2 c0^2) and s2 ~ Inv-Gamma(a0, b0) with the
weakly informative a0 = b0 = 1e-3, or s2 known.  Then
    Vn = (V0^-1 + X'X)^-1,  mn = Vn X'y,  an = a0 + n/2,  bn = b0 + (y'y - mn' Vn^-1 mn)/2
and the marginal of every coefficient is Student-t with 2 an degrees of freedom, location mn and
scale^2 = (bn / an) Vn, so Var(b_j) = bn / (an - 1) (Vn)_jj.  With s2 known the posterior is
N(mn, s2 Vn).  The g-computation effect of a main-effects design is (active - control) b_t = b_t, so
its posterior is the marginal of column 1.  The data are written here, converted exactly to
mpmath values, and the Student-t quantiles are found by root-finding on the incomplete beta CDF.
Repository-authored (MIT OR Apache-2.0).
"""
import json
import mpmath as mp

mp.mp.dps = 40
A0 = B0 = mp.mpf("0.001")


def dataset(n):
    z1 = [((i * 7) % 13) / 6.0 - 1.0 for i in range(n)]
    z2 = [((i * 5) % 11) / 5.0 - 1.0 for i in range(n)]
    t = [1.0 if z1[i] + 0.3 * (((i * 3) % 7) - 3) / 3.0 > 0.0 else 0.0 for i in range(n)]
    y = [1.0 + 2.0 * t[i] + 0.5 * z1[i] - z2[i] + 0.4 * (((i * 11) % 17) - 8) / 8.0 for i in range(n)]
    return z1, z2, t, y


def t_quantile(p, nu, loc, scale):
    def cdf(x):
        u = (x - loc) / scale
        w = nu / (nu + u * u)
        tail = mp.betainc(nu / 2, mp.mpf(1) / 2, 0, w, regularized=True) / 2
        return 1 - tail if u > 0 else tail
    return mp.findroot(lambda x: cdf(x) - p, loc)


def posterior(columns, y, c0, known_s2):
    n = len(y)
    X = mp.matrix([[mp.mpf(1), *[mp.mpf(c[i]) for c in columns]] for i in range(n)])
    Y = mp.matrix([mp.mpf(v) for v in y])
    p = X.cols
    prec = X.T * X + mp.eye(p) / mp.mpf(c0) ** 2
    Vn = prec ** -1
    mn = Vn * (X.T * Y)
    if known_s2 is not None:
        s2 = mp.mpf(known_s2)
        sd = mp.sqrt(s2 * Vn[1, 1])
        z = mp.mpf("1.959963984540054235524594430520551527955931")
        return dict(mean=float(mn[1]), sd=float(sd), lower=float(mn[1] - z * sd),
                    upper=float(mn[1] + z * sd), nu=None, n=n)
    an = A0 + mp.mpf(n) / 2
    bn = B0 + ((Y.T * Y)[0] - (mn.T * prec * mn)[0]) / 2
    var = bn / (an - 1) * Vn[1, 1]
    scale = mp.sqrt(bn / an * Vn[1, 1])
    nu = 2 * an
    return dict(mean=float(mn[1]), sd=float(mp.sqrt(var)),
                lower=float(t_quantile(mp.mpf("0.025"), nu, mn[1], scale)),
                upper=float(t_quantile(mp.mpf("0.975"), nu, mn[1], scale)), nu=float(nu), n=n)


z1, z2, t, y = dataset(48)
cases = []
for name, backend, cols, c0, known in (
    ("adjusted_default_prior", "laplace_default", ["z1", "z2"], 10.0, None),
    ("adjusted_tight_prior", "conjugate", ["z1", "z2"], 0.8, None),
    ("treatment_only", "conjugate", [], 10.0, None),
    ("known_variance", "conjugate", ["z1"], 3.0, 0.09),
):
    columns = [{"z1": z1, "z2": z2}[c] for c in cols]
    cases.append(dict(name=name, backend=backend, covariates=cols, prior_scale=c0, known_variance=known,
                      treatment=t, outcome=y, z1=z1, z2=z2, **posterior([t, *columns], y, c0, known)))
print(json.dumps(cases, separators=(",", ":")))
