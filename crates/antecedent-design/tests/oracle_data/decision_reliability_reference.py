"""Definition-level reference for the decision-reliability numerical error and derived intervals.

Regenerate with:  python3 decision_reliability_reference.py > decision_reliability_reference.json

Every quantity is written from its definition, not from the product's formulas:
  * iid standard error: sqrt(s^2 / n), s^2 with divisor n - 1;
  * batch means: b = max(floor(sqrt n), 2) (or fixed) non-overlapping batches of m = n // b draws,
    SE = sqrt(var(batch means) / b), ESS = clamp(b * var(first b*m draws) / var(batch means), 1, n);
  * self-normalized weights: mean = sum w y / sum w, SE = sqrt(sum (w (y - mean))^2) / sum w,
    ESS = (sum w)^2 / sum w^2;
  * quantile: the r-th order statistic with r = ceil(p n) computed on the decimal p, and its exact
    bootstrap standard error from the definition: the resampled r-th order statistic equals
    x_(i) with probability P(Bin(n, i/n) >= r) - P(Bin(n, (i-1)/n) >= r), evaluated with exact
    rational arithmetic and no truncation;
  * conditional value at risk: the Rockafellar-Uryasev maximum  max_z { z - sum (z - x_i)^+ / (alpha n) }
    over the data points z (lower tail), the upper tail on the negated data; its standard error
    is the sd of the series z* - (z* - x_i)^+ / alpha at the smallest maximiser z*;
  * Wald and Wilson intervals, the Bonferroni critical values, the multiple-comparisons-with-the-best
    set and the selection-corrected interval as stated in the module documentation.
Normal quantiles come from statistics.NormalDist (double precision).

Repository-authored (MIT OR Apache-2.0); standard library only.
"""
import json
import math
import random
from fractions import Fraction
from statistics import NormalDist

ND = NormalDist()


def zcrit(tail):
    return -ND.inv_cdf(tail)


def mean_var(xs):
    n = len(xs)
    m = math.fsum(xs) / n
    v = math.fsum((x - m) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
    return m, v


def n_batches(n, plan):
    b = max(math.isqrt(n), 2) if plan == "square_root" else plan
    assert 2 <= b <= n
    return b


def series_stats(series, weights, dep, plan):
    n = len(series)
    if weights is not None:
        sw = math.fsum(weights)
        mean = math.fsum(y * w for y, w in zip(series, weights)) / sw
        se = math.sqrt(math.fsum((w * (y - mean)) ** 2 for y, w in zip(series, weights))) / sw
        return mean, se, sw * sw / math.fsum(w * w for w in weights)
    if dep == "independent":
        m, v = mean_var(series)
        return m, math.sqrt(v / n), float(n)
    b = n_batches(n, plan)
    m = n // b
    means = [math.fsum(series[k * m:(k + 1) * m]) / m for k in range(b)]
    _, bv = mean_var(means)
    _, dv = mean_var(series[: b * m])
    ess = min(max(b * dv / bv, 1.0), float(n)) if bv > 0 else float(n)
    return math.fsum(series) / n, math.sqrt(bv / b), ess


def order_index(p, n):
    return min(max(math.ceil(Fraction(str(p)) * n), 1), n)


def binom_tail(n, r, x):
    return sum(math.comb(n, k) * x ** k * (1 - x) ** (n - k) for k in range(r, n + 1))


def mj_se(values, r):
    xs = sorted(values)
    n = len(xs)
    prev = binom_tail(n, r, Fraction(0))
    m1 = m2 = Fraction(0)
    for i in range(1, n + 1):
        cur = binom_tail(n, r, Fraction(i, n))
        w = cur - prev
        prev = cur
        x = Fraction(xs[i - 1])
        m1 += w * x
        m2 += w * x * x
    var = m2 - m1 * m1
    assert var >= 0
    return math.sqrt(float(var))


def batch_quantiles(values, b, m, p):
    return [sorted(values[k * m:(k + 1) * m])[order_index(p, m) - 1] for k in range(b)]


def cvar_lower_series(values, alpha):
    n = len(values)
    best = None
    for z in sorted(set(values)):
        v = z - math.fsum(max(z - x, 0.0) for x in values) / (alpha * n)
        if best is None or v > best[0] + 1e-12 * max(1.0, abs(v)):
            best = (v, z)
    value, z = best
    return value, [z - max(z - x, 0.0) / alpha for x in values]


def mean_like(values, f):
    if f["kind"] == "expected":
        return None, list(values)
    if f["kind"] == "threshold":
        t = f["threshold"]
        if f["side"] == "above":
            return None, [1.0 if u > t else 0.0 for u in values]
        return None, [1.0 if u <= t else 0.0 for u in values]
    if f["kind"] == "cvar":
        sign = 1.0 if f["tail"] == "lower" else -1.0
        value, series = cvar_lower_series([sign * u for u in values], f["alpha"])
        return sign * value, [sign * s for s in series]
    return None


def estimate(values, weights, dep, plan, f):
    n = len(values)
    ml = mean_like(values, f)
    if ml is not None:
        val, series = ml
        mean, se, ess = series_stats(series, weights, dep, plan)
        return dict(value=mean if val is None else val, se=se, ess=ess, n=n)
    p = f["p"]
    r = order_index(p, n)
    value = sorted(values)[r - 1]
    iid = mj_se(values, r)
    if dep == "independent":
        return dict(value=value, se=iid, ess=float(n), n=n)
    b = n_batches(n, plan)
    m = n // b
    _, v = mean_var(batch_quantiles(values, b, m, p))
    se = math.sqrt(v / b)
    ess = min(max(n * iid * iid / (se * se), 1.0), float(n)) if se > 0 else float(n)
    return dict(value=value, se=se, ess=ess, n=n)


def difference(a, b, weights, dep, plan, f, coupling):
    ea, eb = estimate(a, weights, dep, plan, f), estimate(b, weights, dep, plan, f)
    value = ea["value"] - eb["value"]
    va, vb = ea["se"] ** 2, eb["se"] ** 2
    out = dict(value=value, se_a=ea["se"], se_b=eb["se"])
    if coupling == "independent":
        se = math.sqrt(va + vb)
        ess = (va * ea["ess"] + vb * eb["ess"]) / (se * se) if se > 0 else min(ea["ess"], eb["ess"])
        return dict(out, se=se, ess=ess)
    if f["kind"] == "quantile":
        bb = n_batches(len(a), plan)
        m = len(a) // bb
        qa, qb = batch_quantiles(a, bb, m, f["p"]), batch_quantiles(b, bb, m, f["p"])
        _, v = mean_var([x - y for x, y in zip(qa, qb)])
        return dict(out, se=math.sqrt(v / bb), ess=None)
    sa, sb = mean_like(a, f)[1], mean_like(b, f)[1]
    _, se, ess = series_stats([x - y for x, y in zip(sa, sb)], weights, dep, plan)
    return dict(out, se=se, ess=ess)


def wald(est, tail, level):
    half = zcrit(tail) * est["se"]
    return dict(estimate=est["value"], se=est["se"], ess=est["ess"], lower=est["value"] - half,
                upper=est["value"] + half, level=level)


def wilson(est, tail, level):
    n, p, z = est["ess"], min(max(est["value"], 0.0), 1.0), zcrit(tail)
    z2 = z * z
    den = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / den
    half = z / den * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return dict(estimate=est["value"], se=est["se"], ess=n, lower=max(centre - half, 0.0),
                upper=min(centre + half, 1.0), level=level)


def top_share(greater, equal, k):
    return 0.0 if greater >= k else min(equal, k - greater) / equal


def derive(case):
    ids = sorted(case["candidates"])
    cols = {i: case["candidates"][i] for i in ids}
    n = len(cols[ids[0]])
    f, dep, plan, w = case["functional"], case["dependence"], case["plan"], case.get("weights")
    coupling = "shared" if case["shared"] else "independent"
    alpha = 1 - case["level"]
    ests = {i: estimate(cols[i], w, dep, plan, f) for i in ids}
    decided = None
    for i in ids:
        if decided is None or ests[i]["value"] > ests[decided]["value"]:
            decided = i
    rk = case.get("ranking")
    stm = {}
    if rk:
        for i in ids:
            g = [sum(1 for j in ids if cols[j][t] > cols[i][t]) for t in range(n)]
            e = [sum(1 for j in ids if cols[j][t] == cols[i][t]) for t in range(n)]
            entry = {"best": series_stats([top_share(g[t], e[t], 1) for t in range(n)], w, dep, plan)}
            entry["top"] = {k: series_stats([top_share(g[t], e[t], k) for t in range(n)], w, dep, plan)
                            for k in rk["top_k"]}
            if rk.get("baseline") and rk["baseline"] != i:
                bl = cols[rk["baseline"]]
                s = [1.0 if cols[i][t] > bl[t] else 0.0 if cols[i][t] < bl[t] else 0.5 for t in range(n)]
                entry["base"] = series_stats(s, w, dep, plan)
            stm[i] = entry
    diffs = {i: difference(cols[decided], cols[i], w, dep, plan, f, coupling) for i in ids if i != decided}
    n_prob = 0
    if rk:
        n_prob = sum(1 + len(rk["top_k"]) + (1 if "base" in stm[i] else 0) for i in ids)
    reported = len(ids) + len(diffs) + n_prob
    per_alpha = alpha if case["multiplicity"] == "marginal" else alpha / max(reported, 1)
    tail, level = per_alpha / 2, 1 - per_alpha
    out = {"decided": decided, "intervals_reported": reported}
    out["utilities"] = {i: wald(ests[i], tail, level) for i in ids}
    dd = {}
    for i, d in diffs.items():
        ess = d["ess"]
        if ess is None:
            ess = min(ests[decided]["ess"], ests[i]["ess"])
        dd[i] = wald(dict(value=d["value"], se=d["se"], ess=ess), tail, level)
    out["differences"] = dd
    out["probabilities"] = []
    if rk:
        for i in ids:
            def stat(s):
                return dict(value=s[0], se=s[1], ess=s[2])
            out["probabilities"].append(dict(id=i, quantity="best", **wilson(stat(stm[i]["best"]), tail, level)))
            for k in rk["top_k"]:
                out["probabilities"].append(dict(id=i, quantity=f"top_{k}", **wilson(stat(stm[i]["top"][k]), tail, level)))
            if "base" in stm[i]:
                out["probabilities"].append(dict(id=i, quantity="baseline", **wilson(stat(stm[i]["base"]), tail, level)))
    # confidence set for the best
    K = len(ids)
    if K < 2:
        members, crit, basis = list(ids), 0.0, "marginal_bonferroni"
    elif K <= 64:
        t = alpha / (K - 1)
        z = zcrit(t)
        members = []
        for j in ids:
            excluded = False
            for i in ids:
                if i == j:
                    continue
                d = difference(cols[i], cols[j], w, dep, plan, f, coupling)
                beats = d["value"] > z * d["se"] if d["se"] > 0 else d["value"] > 0
                if beats:
                    excluded = True
                    break
            if not excluded:
                members.append(j)
        crit, basis = z, "paired_comparisons_with_best"
    else:
        z = zcrit(alpha / (2 * K))
        floor = max(ests[i]["value"] - z * ests[i]["se"] for i in ids)
        members = [i for i in ids if ests[i]["value"] + z * ests[i]["se"] >= floor]
        crit, basis = z, "marginal_bonferroni"
    out["confidence_set"] = dict(basis=basis, members=members, critical=crit)
    d = ests[decided]
    ordinary, simult = zcrit(alpha / 2), zcrit(alpha / (2 * K))
    nh, ex = ordinary * d["se"], (simult - ordinary) * d["se"]
    out["corrected"] = dict(in_sample=d["value"], se=d["se"], numerical=nh, extra=ex,
                            lower=d["value"] - nh - ex, upper=d["value"] + nh + ex, critical=simult)
    return out


def grid(rng, k, n, step, scale):
    return [[round(rng.gauss(0, scale) / step) * step for _ in range(n)] for _ in range(k)]


def main():
    rng = random.Random(20261028)
    result = {"estimates": [], "differences": [], "derive": []}
    base = [round(rng.gauss(1.0, 2.0) * 4) / 4 for _ in range(60)]  # quarter grid: exact ties
    heavy = [rng.lognormvariate(0, 1.2) for _ in range(81)]
    ar = []
    prev = 0.0
    for _ in range(200):
        prev = 0.6 * prev + rng.gauss(0, 1)
        ar.append(prev)
    tiny = [3.0, 1.0, 2.0, 2.0, 5.0]
    weights = [math.exp(rng.gauss(0, 1.3)) for _ in range(60)]
    funcs = [
        {"kind": "expected"},
        {"kind": "threshold", "threshold": 1.0, "side": "above"},
        {"kind": "threshold", "threshold": 1.0, "side": "at_or_below"},
        {"kind": "quantile", "p": 0.5},
        {"kind": "quantile", "p": 0.1},
        {"kind": "quantile", "p": 0.95},
        {"kind": "cvar", "alpha": 0.2, "tail": "lower"},
        {"kind": "cvar", "alpha": 0.25, "tail": "upper"},
        {"kind": "cvar", "alpha": 1.0, "tail": "lower"},
        {"kind": "cvar", "alpha": 0.13, "tail": "lower"},
    ]
    for name, vals in [("tied_grid_60", base), ("lognormal_81", heavy), ("tiny_5", tiny)]:
        for f in funcs:
            if f["kind"] == "quantile" and f["p"] == 0.95 and len(vals) < 5:
                continue
            result["estimates"].append(dict(name=name, values=vals, weights=None, dependence="independent",
                                            plan="square_root", functional=f,
                                            expect=estimate(vals, None, "independent", "square_root", f)))
    for plan in ["square_root", 5, 40]:
        for f in funcs:
            result["estimates"].append(dict(name=f"ar1_200_plan_{plan}", values=ar, weights=None,
                                            dependence="autocorrelated", plan=plan, functional=f,
                                            expect=estimate(ar, None, "autocorrelated", plan, f)))
    for f in funcs[:3]:
        result["estimates"].append(dict(name="weighted_60", values=base, weights=weights, dependence="independent",
                                        plan="square_root", functional=f,
                                        expect=estimate(base, weights, "independent", "square_root", f)))
    # differences
    other = [x + rng.gauss(0, 0.5) for x in base]
    other_ar = [0.5 * x + 0.3 * rng.gauss(0, 1) for x in ar]
    for coupling in ["shared", "independent"]:
        for f in funcs:
            for a, b, dep, nm in [(base, other, "independent", "grid"), (ar, other_ar, "autocorrelated", "ar")]:
                cp = "independent" if coupling == "independent" else "shared"
                result["differences"].append(dict(name=nm, a=a, b=b, weights=None, dependence=dep,
                                                  plan="square_root", functional=f, coupling=cp,
                                                  expect=difference(a, b, None, dep, "square_root", f, cp)))
    result["differences"].append(dict(name="weighted", a=base, b=other, weights=weights, dependence="independent",
                                      plan="square_root", functional=funcs[0], coupling="shared",
                                      expect=difference(base, other, weights, "independent", "square_root", funcs[0], "shared")))
    result["differences"].append(dict(name="independent_lengths", a=base, b=other[:41], weights=None,
                                      dependence="independent", plan="square_root", functional=funcs[0],
                                      coupling="independent",
                                      expect=difference(base, other[:41], None, "independent", "square_root", funcs[0], "independent")))
    # derived uncertainty
    def cand(k, n, step, scale, shift=0.0, shared=False):
        common = [rng.gauss(0, 1) for _ in range(n)] if shared else None
        out = {}
        for j in range(k):
            col = []
            for t in range(n):
                noise = (0.8 * common[t] + 0.6 * rng.gauss(0, 1)) if shared else rng.gauss(0, 1)
                col.append(round((shift * j + scale * noise) / step) * step)
            out[chr(ord("A") + j)] = col
        return out
    scen = [
        ("independent_expected_marginal", dict(candidates=cand(3, 120, 0.25, 1.0, 0.15), shared=False, dependence="independent",
         plan="square_root", functional=funcs[0], level=0.95, multiplicity="marginal",
         ranking=None)),
        ("shared_expected_simultaneous", dict(candidates=cand(4, 150, 0.25, 1.0, 0.1, True), shared=True, dependence="independent",
         plan="square_root", functional=funcs[0], level=0.9, multiplicity="simultaneous",
         ranking=dict(baseline="B", top_k=[1, 2, 3]))),
        ("shared_quantile", dict(candidates=cand(3, 160, 0.5, 2.0, 0.2, True), shared=True, dependence="independent",
         plan="square_root", functional={"kind": "quantile", "p": 0.5}, level=0.95, multiplicity="marginal", ranking=None)),
        ("independent_cvar_upper", dict(candidates=cand(3, 140, 0.125, 1.5, -0.3), shared=False, dependence="independent",
         plan="square_root", functional={"kind": "cvar", "alpha": 0.25, "tail": "upper"}, level=0.99, multiplicity="marginal", ranking=None)),
        ("shared_threshold_ties", dict(candidates=cand(3, 100, 1.0, 0.7, 0.2, True), shared=True, dependence="independent",
         plan="square_root", functional={"kind": "threshold", "threshold": 0.0, "side": "above"}, level=0.8, multiplicity="marginal",
         ranking=dict(baseline=None, top_k=[2]))),
    ]
    wcase = dict(candidates=cand(3, 300, 0.25, 1.0, 0.2, True), shared=True, dependence="independent",
                 plan="square_root", functional=funcs[0], level=0.9, multiplicity="marginal",
                 ranking=dict(baseline="A", top_k=[2]),
                 weights=[math.exp(rng.gauss(0, 1.2)) for _ in range(300)])
    scen.append(("shared_extreme_weights", wcase))
    scen.append(("independent_seventy_candidates", dict(candidates={f"C{j:02d}": col for j, col in enumerate(
        [c for c in cand(70, 100, 0.25, 1.0, 0.01).values()])}, shared=False, dependence="independent",
        plan="square_root", functional=funcs[0], level=0.95, multiplicity="marginal", ranking=None)))
    ar_cols = {chr(ord("A") + j): [] for j in range(3)}
    prev_common = 0.0
    for _ in range(900):
        prev_common = 0.5 * prev_common + rng.gauss(0, 1)
        for j in range(3):
            ar_cols[chr(ord("A") + j)].append(round((prev_common + 0.1 * j + 0.5 * rng.gauss(0, 1)) * 8) / 8)
    scen.append(("autocorrelated_batch", dict(candidates=ar_cols, shared=True, dependence="autocorrelated",
                 plan="square_root", functional=funcs[0], level=0.95, multiplicity="marginal",
                 ranking=dict(baseline="A", top_k=[2]))))
    for name, c in scen:
        result["derive"].append(dict(name=name, **c, expect=derive(c)))
    print(json.dumps(result, indent=0, separators=(",", ":")))


main()
