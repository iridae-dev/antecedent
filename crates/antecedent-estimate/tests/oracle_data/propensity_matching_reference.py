# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath"]
# ///
"""One-nearest-neighbour propensity matching with replacement, by exhaustive search at 60 digits.

Regenerate with:  uv run --script propensity_matching_reference.py > propensity_matching_reference.json

The propensity is the maximum-likelihood logistic fit (Newton's method).  Each query is compared
with every donor of the opposite arm; its matches are all donors at exactly the smallest distance
on the matching scale (the logit of the clipped score, or the clipped score itself), and none when
that distance exceeds the caliper (a distance equal to the caliper matches).  The matched
difference is the query's outcome minus the mean outcome of its matches, less the bias term
mu(x_q) - mean mu(x_match), where mu is the least-squares line of the donor arm's outcome on the
score.  ATT queries the treated, ATC the controls (sign reversed) and ATE both, averaging the
per-unit effects of all matched queries.  The standard error is the homoskedastic matching
variance sigma^2 * W / n^2 with W = n + sum u_j^2 for one arm and W = sum (1 + u_j)^2 over every
unit for both, u_j the share of matched queries a donor serves (a query shares its unit over its
tied donors), sigma^2 the sample variance of the matched effects (n-1 divisor) divided by the
mean of 1 + 1/(matches of the query).  Values are written as nearest binary64.
Repository-authored (MIT OR Apache-2.0).
"""
import json
from mpmath import mp, mpf, exp, log, matrix, lu_solve, sqrt

mp.dps = 60
_state = 0x6A09E667F3BCC908


def u01():
    global _state
    _state = (_state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
    return (_state >> 11) / float(1 << 53)


def fit_logit(t, X):
    n, p = len(t), len(X[0])
    beta = [mpf(0)] * p
    for _ in range(200):
        score = [mpf(0)] * p
        info = matrix(p, p)
        for i in range(n):
            e = 1 / (1 + exp(-sum(beta[k] * X[i][k] for k in range(p))))
            for a in range(p):
                score[a] += (t[i] - e) * X[i][a]
                for b in range(p):
                    info[a, b] += e * (1 - e) * X[i][a] * X[i][b]
        step = lu_solve(info, matrix(score))
        beta = [beta[k] + step[k] for k in range(p)]
        if max(abs(s) for s in step) < mpf(10) ** -50:
            break
    return [sum(beta[k] * X[i][k] for k in range(p)) for i in range(n)]


def line(xs, ys):
    n = mpf(len(xs))
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - slope * mx, slope


def match_arm(feat, y, queries, donors, caliper):
    a, b = line([feat[j] for j in donors], [y[j] for j in donors])
    mu = lambda f: a + b * f
    out = []
    for q in queries:
        dist = [(abs(feat[q] - feat[j]), j) for j in donors]
        best = min(d for d, _ in dist)
        if caliper is not None and best > caliper:
            continue
        tied = [j for d, j in dist if d == best]
        raw = y[q] - sum(y[j] for j in tied) / len(tied)
        bias = mu(feat[q]) - sum(mu(feat[j]) for j in tied) / len(tied)
        out.append((q, tied, raw - bias))
    return out


def estimate(feat, y, t, target, caliper):
    n_all = len(t)
    treated = [i for i in range(n_all) if t[i] > 0.5]
    control = [i for i in range(n_all) if t[i] <= 0.5]
    usage = [mpf(0)] * n_all
    effects, sizes = [], []
    passes = []
    if target in ("att", "ate"):
        passes.append((match_arm(feat, y, treated, control, caliper), 1))
    if target in ("atc", "ate"):
        passes.append((match_arm(feat, y, control, treated, caliper), -1))
    for res, sign in passes:
        for q, tied, d in res:
            effects.append(sign * d)
            sizes.append(len(tied))
            for j in tied:
                usage[j] += mpf(1) / len(tied)
    n = mpf(len(effects))
    tau = sum(effects) / n
    var = sum((e - tau) ** 2 for e in effects) / (n - 1)
    inflation = sum(1 + mpf(1) / s for s in sizes) / n
    sigma2 = var / inflation
    if target == "ate":
        weight = sum((1 + u) ** 2 for u in usage)
    else:
        weight = n + sum(u * u for u in usage)
    return tau, sqrt(sigma2 * weight / n**2), len(effects)


def case(name, t, y, x, clip, scale, caliper=None):
    n = len(t)
    X = [[mpf(1)] + [mpf(c) for c in row] for row in x]
    eta = fit_logit(t, X)
    if clip is not None:
        bound = log((1 - mpf(clip)) / mpf(clip))
        eta = [max(-bound, min(bound, e)) for e in eta]
    score = [1 / (1 + exp(-e)) for e in eta]
    if clip is not None:
        score = [max(mpf(clip), min(1 - mpf(clip), s)) for s in score]
    feat = eta if scale == "logit" else score
    yy = [mpf(v) for v in y]
    cal = None if caliper is None else mpf(caliper)
    out = {"name": name, "t": t, "y": y, "x": x, "clip": clip, "scale": scale,
           "caliper": caliper, "targets": {}}
    for target in ("att", "atc", "ate"):
        try:
            tau, se, matched = estimate(feat, yy, t, target, cal)
            out["targets"][target] = {"ate": float(tau), "se": float(se), "matched": matched}
        except ZeroDivisionError:
            out["targets"][target] = None
    return out


def draw(n, slope, levels=None):
    t, y, x = [], [], []
    for i in range(n):
        a = u01() * 4 - 2
        if levels:
            a = float(int(u01() * levels) - levels // 2)
        pr = 1 / (1 + 2.718281828459045 ** (-(slope * a + 0.1)))
        ti = 1.0 if u01() < pr else 0.0
        yi = 1.0 + 1.5 * ti + 0.9 * a + 0.3 * a * a * ti + (u01() - 0.5)
        t.append(ti)
        y.append(float(yi))
        x.append([a])
    return t, y, x


cases = []
t, y, x = draw(40, 0.8)
cases.append(case("continuous_logit_scale", t, y, x, None, "logit"))
cases.append(case("continuous_raw_scale", t, y, x, None, "raw"))
cases.append(case("logit_caliper", t, y, x, None, "logit", 0.2))
cases.append(case("raw_caliper", t, y, x, None, "raw", 0.03))
t2, y2, x2 = draw(60, 2.6)
cases.append(case("clip_ties_logit", t2, y2, x2, 0.15, "logit"))
cases.append(case("clip_ties_raw", t2, y2, x2, 0.15, "raw"))
cases.append(case("extreme_scores_no_clip", t2, y2, x2, None, "logit"))
t3, y3, x3 = draw(50, 0.7, levels=5)
cases.append(case("discrete_covariate_ties", t3, y3, x3, None, "logit"))
cases.append(case("outcome_offset_1e7", t, [v + 1.0e7 for v in y], x, None, "logit"))
t4, y4, x4 = draw(10, 0.6)
cases.append(case("tiny_sample", t4, y4, x4, None, "logit"))
print(json.dumps({"cases": cases}))
