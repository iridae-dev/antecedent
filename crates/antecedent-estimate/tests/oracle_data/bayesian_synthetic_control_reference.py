# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Simplex quadrature of the Bayesian synthetic-control posterior (independent errors).

Regenerate with:  uv run --script bayesian_synthetic_control_reference.py > bayesian_synthetic_control_reference.json

Three donors, eight pre-intervention periods t = 1..8 and three post periods t = 9..11.  With the
donor weights w on the simplex, a symmetric Dirichlet(alpha) prior, treated pre-period outcomes
y_t = sum_j w_j x_jt + e_t, e_t ~ N(0, s2) independent and s2 ~ Inv-Gamma(a, b), integrating s2 out
gives the weight posterior
    p(w | y) ~ prod_j w_j^(alpha - 1) * (b + RSS(w) / 2)^-(a + n / 2),   RSS(w) = sum_t (y_t - w'x_t)^2.
It is integrated on the triangle (w1, w2 >= 0, w1 + w2 <= 1) by composite tensor Gauss-Legendre quadrature
(12 nodes on each of 10 or 16 sub-intervals per axis, w2 scaled to [0, 1 - w1]; the two rules must agree to 1e-9), giving the
posterior mean and sd of each weight.  The effect is the realised average post-period effect:
gap(w) = mean_{t post} (y_t - w'x_t) minus the mean of three independent counterfactual errors, so
    E[effect] = E[gap(w)],   Var(effect) = Var(gap(w)) + E[s2] / 3,
with E[s2 | w] = (b + RSS(w) / 2) / (a + n / 2 - 1).  The pre-fit RMSE is that of the posterior-mean
weights.  The panel is the 3-donor panel with treated path 0.6 d1 + 0.4 d2 + small deterministic
noise and a post-period lift of 2, written here exactly as the test builds it.
Repository-authored (MIT OR Apache-2.0).
"""
import json
import mpmath as mp

mp.mp.dps = 20


def donors(t):
    tf = float(t)
    return [1.0 + 0.5 * tf + 0.3 * (((t * 5) % 7) / 3.0 - 1.0),
            3.0 - 0.2 * tf + 0.3 * (((t * 3) % 8) / 3.5 - 1.0),
            2.0 + 0.4 * ((t * 3) % 5)]


def treated(t, d):
    return 0.6 * d[0] + 0.4 * d[1] + 0.15 * (((t * 11) % 13) / 6.0 - 1.0) + (2.0 if t >= 9 else 0.0)


def gauss_legendre(count):
    """Nodes and weights on [0, 1] by Newton iteration on the Legendre polynomial."""
    nodes, weights = [], []
    for k in range(1, count + 1):
        x = mp.cos(mp.pi * (k - mp.mpf(1) / 4) / (count + mp.mpf(1) / 2))
        for _ in range(100):
            p0, p1 = mp.mpf(1), x
            for m in range(2, count + 1):
                p0, p1 = p1, ((2 * m - 1) * x * p1 - (m - 1) * p0) / m
            dp = count * (x * p1 - p0) / (x * x - 1)
            step = p1 / dp
            x -= step
            if abs(step) < mp.mpf(10) ** (-mp.mp.dps + 2):
                break
        p0, p1 = mp.mpf(1), x
        for m in range(2, count + 1):
            p0, p1 = p1, ((2 * m - 1) * x * p1 - (m - 1) * p0) / m
        dp = count * (x * p1 - p0) / (x * x - 1)
        nodes.append((x + 1) / 2)
        weights.append(1 / ((1 - x * x) * dp * dp))
    return nodes, weights


def composite(panels, per_panel=12):
    """Composite Gauss-Legendre rule on [0, 1]: `panels` equal sub-intervals of `per_panel` nodes."""
    base_nodes, base_weights = gauss_legendre(per_panel)
    nodes, weights = [], []
    for k in range(panels):
        for u, w in zip(base_nodes, base_weights):
            nodes.append((k + u) / panels)
            weights.append(w / panels)
    return nodes, weights


def reference(alpha, count, a=0.001, b=0.001):
    pre = list(range(1, 9))
    post = [9, 10, 11]
    X = {t: [mp.mpf(v) for v in donors(t)] for t in range(1, 12)}
    Y = {t: mp.mpf(treated(t, donors(t))) for t in range(1, 12)}
    n = len(pre)
    alpha = mp.mpf(alpha)
    a, b = mp.mpf(a), mp.mpf(b)
    nodes, weights = composite(count)
    keys = ["z", "w0", "w1", "w2", "q0", "q1", "q2", "g", "g2", "s2"]
    acc = dict.fromkeys(keys, mp.mpf(0))
    for u, wu in zip(nodes, weights):
        w1 = u
        span = 1 - w1
        for v, wv in zip(nodes, weights):
            w2 = span * v
            w = [w1, w2, 1 - w1 - w2]
            r = sum((Y[t] - sum(w[j] * X[t][j] for j in range(3))) ** 2 for t in pre)
            gap = sum(Y[t] - sum(w[j] * X[t][j] for j in range(3)) for t in post) / 3
            prior = (w[0] * w[1] * w[2]) ** (alpha - 1) if alpha != 1 else 1
            dens = prior * (b + r / 2) ** (-(a + mp.mpf(n) / 2)) * wu * wv * span
            acc["z"] += dens
            for j in range(3):
                acc[f"w{j}"] += dens * w[j]
                acc[f"q{j}"] += dens * w[j] ** 2
            acc["g"] += dens * gap
            acc["g2"] += dens * gap ** 2
            acc["s2"] += dens * (b + r / 2) / (a + mp.mpf(n) / 2 - 1)
    z = acc["z"]
    mean_w = [acc[f"w{j}"] / z for j in range(3)]
    sd_w = [mp.sqrt(acc[f"q{j}"] / z - mean_w[j] ** 2) for j in range(3)]
    e_gap = acc["g"] / z
    var_gap = acc["g2"] / z - e_gap ** 2
    e_s2 = acc["s2"] / z
    rss_mean = sum((Y[t] - sum(mean_w[j] * X[t][j] for j in range(3))) ** 2 for t in pre)
    return dict(alpha=float(alpha), panels=count, effect_mean=float(e_gap),
                effect_sd=float(mp.sqrt(var_gap + e_s2 / 3)), weight_means=[float(v) for v in mean_w],
                weight_sds=[float(v) for v in sd_w], pre_fit_rmse=float(mp.sqrt(rss_mean / n)))


# The result is checked for convergence in the number of Gauss-Legendre nodes.
cases = []
for alpha in (1, 20):
    coarse, fine = reference(alpha, 10), reference(alpha, 16)
    for key in ("effect_mean", "effect_sd", "pre_fit_rmse"):
        assert abs(coarse[key] - fine[key]) < 1e-9, (alpha, key, coarse[key], fine[key])
    cases.append(fine)
print(json.dumps(cases, separators=(",", ":")))
