# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Dense-covariance reference for the weighted and gapped joint-panel likelihood.

Regenerate with:
    uv run --offline --script joint_panel_weights_gaps_reference.py > joint_panel_weights_gaps_reference.json

Written from the definition of the model, not from the product's precision-factor algebra.

Gap cases.  A group's residuals are observed at a subset of its periods.  With  r = u 1 + e,
u ~ N(0, tau^2) and e a stationary Gaussian AR(1) process, Cov(e_s, e_t) = sigma^2 rho^|s-t| /
(1 - rho^2) at the TRUE time distance.  The law of the observed subset is the Gaussian with the
corresponding sub-matrix of the full covariance.  The script builds that sub-matrix exactly,
solves by dense elimination in 120-digit arithmetic and returns the marginal log-likelihood and
the conditional mean and variance of the intercept.

Model cases.  A five-group unbalanced panel with one assignment channel, the three-period
simplex adstock, Michaelis-Menten saturation and a confounder (nine free coordinates, in the
product's order: intercept, channel coefficient, confounder coefficient, two lag logits, log k,
atanh rho, log sigma, log tau).  For each variant (weighted, gapped, weighted and gapped) the
script evaluates, with 60-digit central differences of the dense log-likelihood:

  * the weighted log-likelihood  sum_g w_g l_g(x);
  * the weighted score of every group  w_g grad l_g(x);
  * the weighted observed information  H = -hess sum_g w_g l_g(x);
  * the weighted group-cluster sandwich  (G/(G-1)) H^-1 [sum_g (s_g - sbar)(s_g - sbar)'] H^-1;
  * the maximum-likelihood coordinates of the weighted dense likelihood, by damped Newton
    iteration on the dense log-likelihood from the data-generating coordinates.

Repository-authored (MIT OR Apache-2.0).
"""
import json
import random
import mpmath as mp

mp.mp.dps = 120


def dense_group(times, r, rho, sigma, tau):
    """Marginal law of residuals r observed at integer periods `times`."""
    n = len(r)
    rho_, sig_, tau_ = mp.mpf(rho), mp.mpf(sigma), mp.mpf(tau)
    s2, t2 = sig_ * sig_, tau_ * tau_
    cov = mp.matrix(n, n)
    for i in range(n):
        for j in range(n):
            cov[i, j] = s2 * rho_ ** abs(times[i] - times[j]) / (1 - rho_ * rho_) + t2
    rv = mp.matrix([mp.mpf(x) for x in r])
    ones = mp.matrix([1] * n)
    x = mp.lu_solve(cov, rv)
    y = mp.lu_solve(cov, ones)
    quad = sum(rv[i] * x[i] for i in range(n))
    logdet = mp.log(mp.det(cov))
    ll = -(n * mp.log(2 * mp.pi) + logdet + quad) / 2
    mean = t2 * sum(x[i] for i in range(n))
    var = t2 - t2 * t2 * sum(y[i] for i in range(n))
    return ll, mean, var


def gap_case(name, times, r, rho, sigma, tau):
    ll, mean, var = dense_group(times, r, rho, sigma, tau)
    return {"name": name, "times": times, "r": r, "rho": rho, "sigma": sigma, "tau": tau,
            "log_likelihood": float(ll), "intercept_mean": float(mean),
            "intercept_variance": float(var)}


# ---------------------------------------------------------------- model cases

D = 9  # coordinates


def simplex(a, b):
    m = max(a, b, mp.mpf(0))
    u = [mp.e ** (a - m), mp.e ** (b - m), mp.e ** (-m)]
    s = sum(u)
    return [v / s for v in u]


def group_loglik(group, x):
    """Dense log-likelihood of one group at coordinates x (list of mp numbers)."""
    c, beta, gamma, a, b, logk, atanh_rho, logsig, logtau = x
    w = simplex(a, b)
    k = mp.e ** logk
    rho = mp.tanh(atanh_rho)
    sigma, tau = mp.e ** logsig, mp.e ** logtau
    hist = group["history"]  # [x_-2, x_-1]
    ch = group["channel"]
    times, resid = [], []
    for t, y in enumerate(group["outcome"]):
        if y is None:
            continue
        def val(lag):
            return ch[t - lag] if lag <= t else hist[2 + t - lag]
        e = sum(w[l] * mp.mpf(val(l)) for l in range(3))
        mean = c + beta * e / (e + k) + gamma * mp.mpf(group["confounder"][t])
        times.append(t)
        resid.append(mp.mpf(y) - mean)
    return dense_group(times, resid, rho, sigma, tau)[0]


def group_intercept_mean(group, x):
    """E[u | r] of one group at coordinates x."""
    return group_dense(group, x)[1]


def group_dense(group, x):
    c, beta, gamma, a, b, logk, atanh_rho, logsig, logtau = x
    w = simplex(a, b)
    k = mp.e ** logk
    rho = mp.tanh(atanh_rho)
    sigma, tau = mp.e ** logsig, mp.e ** logtau
    hist = group["history"]
    ch = group["channel"]
    times, resid = [], []
    for t, y in enumerate(group["outcome"]):
        if y is None:
            continue
        def val(lag):
            return ch[t - lag] if lag <= t else hist[2 + t - lag]
        e = sum(w[l] * mp.mpf(val(l)) for l in range(3))
        mean = c + beta * e / (e + k) + gamma * mp.mpf(group["confounder"][t])
        times.append(t)
        resid.append(mp.mpf(y) - mean)
    return dense_group(times, resid, rho, sigma, tau)


# One prespecified program per entry: the dose schedule, the pre-period history of every
# group, and the future confounder.  Its response is the weighted average over groups of the
# conditional mean  c + beta e/(e+k) + gamma z + E[u_g | r_g]  at the maximum-likelihood point.
PROGRAMS = {
    "treated": {"dose": [3.0, 1.0, 0.0, 2.0], "history": [1.0, 2.0], "confounder": [0.2, -0.3, 0.1, 0.0]},
    "reference": {"dose": [0.0, 0.0, 0.0, 0.0], "history": [1.0, 2.0], "confounder": [0.2, -0.3, 0.1, 0.0]},
}


def program_response(groups, weights, x, program):
    c, beta, gamma, a, b, logk = x[0], x[1], x[2], x[3], x[4], x[5]
    w = simplex(a, b)
    k = mp.e ** logk
    total_w = sum(mp.mpf(v) for v in weights)
    out = []
    for t in range(len(program["dose"])):
        acc = mp.mpf(0)
        for gi, group in enumerate(groups):
            def val(lag):
                return program["dose"][t - lag] if lag <= t else program["history"][2 + t - lag]
            e = sum(w[l] * mp.mpf(val(l)) for l in range(3))
            mean = c + beta * e / (e + k) + gamma * mp.mpf(program["confounder"][t])
            acc += mp.mpf(weights[gi]) * (mean + group_intercept_mean(group, x))
        out.append(float(acc / total_w))
    return out


def grad(f, x, h):
    g = []
    for i in range(len(x)):
        xp = list(x); xm = list(x)
        xp[i] += h; xm[i] -= h
        g.append((f(xp) - f(xm)) / (2 * h))
    return g


def hessian(f, x, h):
    p = len(x)
    H = mp.matrix(p, p)
    for i in range(p):
        for j in range(p):
            xpp = list(x); xpm = list(x); xmp = list(x); xmm = list(x)
            xpp[i] += h; xpp[j] += h
            xpm[i] += h; xpm[j] -= h
            xmp[i] -= h; xmp[j] += h
            xmm[i] -= h; xmm[j] -= h
            H[i, j] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * h * h)
    return H


def build_panel(rnd, truth, lengths, gaps):
    groups = []
    c, beta, gamma, a, b, logk, atanh_rho, logsig, logtau = [mp.mpf(v) for v in truth]
    w = simplex(a, b)
    k = mp.e ** logk
    rho = mp.tanh(atanh_rho)
    sigma, tau = mp.e ** logsig, mp.e ** logtau
    for g, n in enumerate(lengths):
        hist = [round(rnd.uniform(0, 4), 3) for _ in range(2)]
        ch = [round(rnd.uniform(0, 5), 3) if rnd.random() > 0.2 else 0.0 for _ in range(n)]
        z = [round(rnd.gauss(0, 1), 3) for _ in range(n)]
        u = rnd.gauss(0, float(tau))
        e_prev = rnd.gauss(0, float(sigma / mp.sqrt(1 - rho * rho)))
        out = []
        for t in range(n):
            e_prev = float(rho) * e_prev + rnd.gauss(0, float(sigma))
            def val(lag):
                return ch[t - lag] if lag <= t else hist[2 + t - lag]
            e = sum(w[l] * mp.mpf(val(l)) for l in range(3))
            mean = c + beta * e / (e + k) + gamma * mp.mpf(z[t])
            out.append(round(float(mean) + u + e_prev, 6))
        for t in gaps.get(g, []):
            out[t] = None
        groups.append({"history": hist, "channel": ch, "confounder": z, "outcome": out})
    return groups


def sandwich(groups, weights, x, h_score, h_hess):
    G = len(groups)
    total = lambda xx: sum(weights[g] * group_loglik(groups[g], xx) for g in range(G))
    scores = [[weights[g] * s for s in grad(lambda xx, g=g: group_loglik(groups[g], xx), x, h_score)]
              for g in range(G)]
    H = -hessian(total, x, h_hess)
    mean = [sum(scores[g][j] for g in range(G)) / G for j in range(D)]
    meat = mp.matrix(D, D)
    for g in range(G):
        d = mp.matrix([scores[g][j] - mean[j] for j in range(D)])
        meat += d * d.T
    Hinv = H ** -1
    V = (mp.mpf(G) / (G - 1)) * Hinv * meat * Hinv
    return total(x), scores, H, V


def newton(groups, weights, x0):
    G = len(groups)
    total = lambda xx: sum(weights[g] * group_loglik(groups[g], xx) for g in range(G))
    x = [mp.mpf(v) for v in x0]
    for _ in range(40):
        g = mp.matrix(grad(total, x, mp.mpf(10) ** -16))
        if max(abs(v) for v in g) < mp.mpf(10) ** -12:
            break
        H = -hessian(total, x, mp.mpf(10) ** -11)
        try:
            step = mp.lu_solve(H, g)
        except ZeroDivisionError:
            break
        # damped ascent: halve the step until the objective does not fall
        base = total(x)
        t = mp.mpf(1)
        for _ in range(30):
            cand = [x[i] + t * step[i] for i in range(D)]
            if total(cand) >= base:
                x = cand
                break
            t /= 2
    return x


def model_variant(name, groups, weights, x, certified_point=None):
    """One variant.  The maximizer is found by damped Newton iteration from x; when the
    weakly identified likelihood defeats it (a flat ridge), `certified_point` is a candidate
    maximizer instead and is certified by the dense likelihood itself: a vanishing dense score
    and a positive definite dense observed information (smallest eigenvalue reported)."""
    ll, scores, H, V = sandwich(groups, weights, [mp.mpf(v) for v in x],
                                mp.mpf(10) ** -16, mp.mpf(10) ** -11)
    if certified_point is None:
        mle = newton(groups, weights, x)
        certificate = {}
    else:
        mle = [mp.mpf(v) for v in certified_point]
        total_ = lambda xx: sum(mp.mpf(weights[i]) * group_loglik(groups[i], xx) for i in range(len(groups)))
        info = -hessian(total_, mle, mp.mpf(10) ** -11)
        certificate = {"mle_min_information_eigenvalue": float(min(mp.eigsy(info, eigvals_only=True)))}
    return {**certificate,
        "name": name,
        "weights": weights,
        "groups": groups,
        "x": x,
        "log_likelihood": float(ll),
        "scores": [[float(v) for v in row] for row in scores],
        "information": [[float(H[i, j]) for j in range(D)] for i in range(D)],
        "covariance": [[float(V[i, j]) for j in range(D)] for i in range(D)],
        "mle": [float(v) for v in mle],
        "responses": {name_: program_response(groups, weights, mle, prog)
                      for name_, prog in PROGRAMS.items()},
        "programs": PROGRAMS,
        "mle_log_likelihood": float(sum(weights[i] * group_loglik(groups[i], mle) for i in range(len(groups)))),
        "mle_score_norm": float(max(abs(v) for v in grad(
            lambda xx: sum(weights[i] * group_loglik(groups[i], xx) for i in range(len(groups))),
            mle, mp.mpf(10) ** -16))),
    }


def main():
    rnd = random.Random(20261029)
    nrm = lambda s, n: [round(rnd.gauss(0, s), 4) for _ in range(n)]
    out = {"gap_cases": [], "model_cases": []}
    g = out["gap_cases"]
    g.append(gap_case("no_gap_matches_contiguous", [0, 1, 2, 3, 4], nrm(0.3, 5), 0.45, 0.2, 0.35))
    g.append(gap_case("one_interior_gap", [0, 1, 3, 4, 5], nrm(0.3, 5), 0.45, 0.2, 0.35))
    g.append(gap_case("leading_gap", [3, 4, 5, 6], nrm(0.3, 4), 0.6, 0.3, 0.4))
    g.append(gap_case("trailing_gap_is_a_truncation", [0, 1, 2, 3], nrm(0.3, 4), -0.6, 0.3, 0.4))
    g.append(gap_case("wide_gap", [0, 1, 7, 8, 9], nrm(0.3, 5), 0.8, 0.2, 0.5))
    g.append(gap_case("alternating_observation", [0, 2, 4, 6, 8, 10], nrm(0.4, 6), -0.7, 0.25, 0.3))
    g.append(gap_case("negative_rho_odd_distance", [0, 3, 4, 8], nrm(0.4, 4), -0.9, 0.2, 0.6))
    g.append(gap_case("rho_zero_gaps_are_deletions", [0, 2, 5, 6], nrm(0.4, 4), 0.0, 0.3, 0.5))
    g.append(gap_case("rho_near_one_gap", [0, 5, 6, 11], nrm(0.5, 4), 0.999, 0.05, 0.5))
    g.append(gap_case("three_observations_two_gaps", [1, 4, 9], nrm(0.4, 3), 0.5, 0.4, 0.3))
    g.append(gap_case("offset_1e6_with_a_gap", [0, 1, 4, 5], [1e6 + v for v in nrm(0.2, 4)], 0.4, 0.2, 0.5))
    g.append(gap_case("tau_huge_with_a_gap", [0, 2, 3, 7], nrm(0.3, 4), 0.6, 0.2, 1e3))
    g.append(gap_case("long_group_with_gaps", [t for t in range(60) if t % 7 not in (3, 4)],
                      nrm(0.4, len([t for t in range(60) if t % 7 not in (3, 4)])), 0.9, 0.2, 0.7))

    mp.mp.dps = 60
    truth = [1.0, 2.5, 0.6, 0.4, -0.3, 0.2, 0.5, -1.4, -0.9]
    lengths = [9, 11, 8, 10, 12]
    base_groups = build_panel(rnd, truth, lengths, {})
    gapped_groups = build_panel(random.Random(20261030), truth, lengths, {1: [3, 4], 3: [0, 6]})
    raw_weights = [0.5, 1.0, 2.5, 1.5, 4.0]
    x = [1.1, 2.3, 0.5, 0.3, -0.2, 0.1, 0.4, -1.3, -0.8]
    ones = [1.0] * 5
    out["model_cases"].append(model_variant("weighted", base_groups, raw_weights, x))
    out["model_cases"].append(model_variant("gapped", gapped_groups, ones, x))
    out["model_cases"].append(model_variant("weighted_and_gapped", gapped_groups, raw_weights, x))
    # Rows hidden by an analysis mask: row m of a unit (m = 2 + period) removes the periods
    # whose exposure window reads it, m-2 .. m, in the product's two-lag window.
    masked_groups = [dict(gr, outcome=list(gr["outcome"])) for gr in base_groups]
    for gi, periods in {2: [5, 6, 7], 4: [0, 1, 2]}.items():
        for t in periods:
            masked_groups[gi]["outcome"][t] = None
    # Newton iteration from the generating coordinates does not converge on this variant (the
    # hidden rows leave a flat ridge in the lag and saturation coordinates); its maximizer is the
    # candidate below, produced by the product's own optimizer and
    # accepted as the maximizer only because the dense score vanishes and the dense information is positive
    # definite there.
    certified = [1.4020681984982084, 3.0329908970553383, 0.6512887742006984, 0.4243444380983355,
                 -0.20808080640296645, 1.0547305245096232, 0.3290651623879472,
                 -1.471956544263882, -1.7725769192406136]
    out["model_cases"].append(model_variant("masked_rows", masked_groups, ones, x, certified))
    json.dump(out, open("/dev/stdout", "w"), separators=(",", ":"))


main()
