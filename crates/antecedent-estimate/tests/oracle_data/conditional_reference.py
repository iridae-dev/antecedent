# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Reference for the linear interaction route of the conditional effect (ConditionalLinearAdjustment).

Regenerate with:  uv run --script conditional_reference.py > conditional_reference.json

The design is [1, T, W_1..W_m, T*W_1..T*W_m, Z..]; beta = (X'X)^-1 X'y and e = y - X beta at 50
digits from the exact binary64 inputs.  With delta = active - control and wbar the modifier means:
  point = delta (beta_T + sum_k beta_TWk wbar_k)
  se^2  = s2 g'(X'X)^-1 g + sum_i (sum_k c_k (w_ik - wbar_k))^2 / (n (n-1)),
          g = delta (e_T + sum_k wbar_k e_TWk),  c_k = delta beta_TWk,  s2 = e'e / (n - p)
  arm mean mu_a = mean_i x_i(a)' beta with T set to the arm level (modifiers and Z as observed)
  arm influence_i = (x_i(a)'beta - mu_a) + n xbar(a)' (X'X)^-1 x_i e_i,
          xbar(a) = column means of the design with T = a and T*W = a wbar
  contrast influence_i = n g'(X'X)^-1 x_i e_i + sum_k c_k (w_ik - wbar_k).
Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import json

import mpmath as mp

mp.mp.dps = 50


class Lcg:
    def __init__(self, s):
        self.s = s

    def u(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) % 2**64
        return (self.s >> 11) / float(1 << 53)


def fit(case):
    t, y, ws, zs = case["t"], case["y"], case["w"], case["z"]
    n, m, nz = len(t), len(ws), len(zs)
    active, control = mp.mpf(case["active"]), mp.mpf(case["control"])
    delta = active - control
    p = 2 + 2 * m + nz
    X = []
    for i in range(n):
        row = [mp.mpf(1), mp.mpf(t[i])] + [mp.mpf(w[i]) for w in ws]
        row += [mp.mpf(t[i]) * mp.mpf(w[i]) for w in ws] + [mp.mpf(z[i]) for z in zs]
        X.append(row)
    Y = [mp.mpf(v) for v in y]
    A = mp.zeros(p, p)
    b = mp.zeros(p, 1)
    for r, yy in zip(X, Y):
        for i in range(p):
            b[i] += r[i] * yy
            for j in range(p):
                A[i, j] += r[i] * r[j]
    inv = A ** -1
    beta = inv * b
    e = [Y[i] - sum(X[i][j] * beta[j] for j in range(p)) for i in range(n)]
    s2 = sum(v * v for v in e) / (n - p)
    wbar = [sum(mp.mpf(v) for v in w) / n for w in ws]
    zbar = [sum(mp.mpf(v) for v in z) / n for z in zs]
    g = [mp.mpf(0)] * p
    g[1] = delta
    for k in range(m):
        g[2 + m + k] = delta * wbar[k]
    ginv = [sum(g[a] * inv[a, j] for a in range(p)) for j in range(p)]
    se_coef2 = s2 * sum(ginv[j] * g[j] for j in range(p))
    c = [delta * beta[2 + m + k] for k in range(m)]
    mean_term = sum(sum(c[k] * (mp.mpf(ws[k][i]) - wbar[k]) for k in range(m)) ** 2 for i in range(n)) / (n * (n - 1))
    point = delta * (beta[1] + sum(beta[2 + m + k] * wbar[k] for k in range(m)))

    def arm(level):
        level = mp.mpf(level)
        mu_i = []
        for i in range(n):
            v = beta[0] + beta[1] * level
            v += sum(beta[2 + k] * mp.mpf(ws[k][i]) for k in range(m))
            v += sum(beta[2 + m + k] * level * mp.mpf(ws[k][i]) for k in range(m))
            v += sum(beta[2 + 2 * m + k] * mp.mpf(zs[k][i]) for k in range(nz))
            mu_i.append(v)
        mu = sum(mu_i) / n
        xbar = [mp.mpf(1), level] + wbar + [level * w for w in wbar] + zbar
        xinv = [sum(xbar[a] * inv[a, j] for a in range(p)) for j in range(p)]
        infl = [mu_i[i] - mu + n * sum(xinv[j] * X[i][j] for j in range(p)) * e[i] for i in range(n)]
        return mu, infl

    mu0, if0 = arm(case["control"])
    mu1, if1 = arm(case["active"])
    contrast = [n * sum(ginv[j] * X[i][j] for j in range(p)) * e[i]
                + sum(c[k] * (mp.mpf(ws[k][i]) - wbar[k]) for k in range(m)) for i in range(n)]
    f = lambda v: float(v)
    return {"ate": f(point), "se": f(mp.sqrt(se_coef2 + mean_term)), "se_coefficient_part": f(mp.sqrt(se_coef2)),
            "mu0": f(mu0), "mu1": f(mu1), "if0": [f(v) for v in if0], "if1": [f(v) for v in if1],
            "contrast_influence": [f(v) for v in contrast]}


def make(name, n, m, nz, seed, treat=(0.0, 1.0), p_treat=0.5, noise=0.3, w_offset=0.0, w_scale=1.0,
         collinear=0.0, hetero=False):
    g = Lcg(seed)
    t = [treat[1] if g.u() < p_treat else treat[0] for _ in range(n)]
    if all(v == t[0] for v in t):
        t[0] = treat[1 - treat.index(t[0])]
    ws = [[w_offset + w_scale * (g.u() * 4 - 2) for _ in range(n)] for _ in range(m)]
    zs = [[g.u() * 3 - 1 for _ in range(n)] for _ in range(nz)]
    if collinear and nz:
        zs[0] = [ws[0][i] + collinear * (g.u() - 0.5) for i in range(n)]
    y = []
    for i in range(n):
        v = 1.0 + 1.5 * t[i] + sum((0.4 + 0.2 * k) * ws[k][i] for k in range(m))
        v += sum(0.3 * ws[k][i] * t[i] * (1 + k) for k in range(m)) + sum(0.5 * z[i] for z in zs)
        v += noise * (g.u() - 0.5) * (1 + (abs(ws[0][i] - w_offset) if hetero else 0))
        y.append(v)
    case = {"name": name, "t": t, "y": y, "w": ws, "z": zs, "active": treat[1], "control": treat[0]}
    case["expected"] = fit(case)
    return case


cases = [
    make("one_modifier", 40, 1, 0, 1),
    make("one_modifier_adjustment", 50, 1, 2, 2),
    make("two_modifiers_adjustment", 60, 2, 1, 3),
    make("three_modifiers", 45, 3, 0, 4),
    make("non_binary_levels", 40, 1, 1, 5, treat=(1.0, 3.0)),
    make("negative_levels_reversed", 40, 1, 1, 6, treat=(-2.0, 0.5)),
    make("unbalanced_arms", 30, 1, 1, 7, p_treat=0.12),
    make("offset_modifier", 40, 1, 1, 8, w_offset=1.0e4, w_scale=0.05),
    make("near_collinear_adjustment", 50, 1, 1, 9, collinear=1e-3),
    make("heteroskedastic", 60, 1, 1, 10, hetero=True, noise=2.0),
    make("minimum_rows", 8, 1, 0, 11, noise=0.5),
]
print(json.dumps(cases))
