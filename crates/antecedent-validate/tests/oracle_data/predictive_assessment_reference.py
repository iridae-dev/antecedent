# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Reference values for the predictive-assessment calibration diagnostics and score intervals.

Regenerate with:  uv run --script predictive_assessment_reference.py > predictive_assessment_reference.json

Written from the stated definitions at 60 digits.
  PIT           interval [F(y-), F(y)] of the outcome under its law (a point for a continuous law);
                the histogram spreads each case's unit mass uniformly over the interval across the
                bins; chi-square sum (count - n/b)^2/(n/b); max |count/n - 1/b|; KS distance of
                the mid-PIT (F(y-) + F(y))/2 from U(0,1), max over order statistics of
                (i+1)/n - u_(i) and u_(i) - i/n.
  coverage      central interval [Q(t), Q(1 - t)], t = (1 - level)/2, with Q the quantile function
                (smallest x with F(x) >= q, with a 1e-12 tolerance for the two finite families) or,
                for a sample of m draws, the type-6 order-statistic rank q (m + 1) clamped to
                [1, m], interpolated; coverage = share of outcomes inside, width = mean length.
  reliability   binary-event forecasts in b equal bins by forecast probability, mean forecast,
                observed frequency, and sum (n_b/n) |observed - forecast|.
  score interval   mean of the case scores, Student-t interval with half-width t_{1-(1-c)/2, df} * se,
                se^2 the iid s^2/n, the Bartlett long-run variance with lag L over the cases in time order
                (df n/(L+1) - 1) or the cluster-robust G/(G-1) sum_g (sum of centred scores)^2 / n^2
                (df G - 1).

Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import json
import random

import mpmath as mp

mp.mp.dps = 60
mpf = mp.mpf
rng = random.Random(31337)


def ncdf(x):
    return mp.ncdf(mpf(x))


def ninv(q):
    return mp.sqrt(2) * mp.erfinv(2 * mpf(q) - 1)


def crps_gauss(mean, sd, y):
    z = (mpf(y) - mpf(mean)) / mpf(sd)
    phi = mp.npdf(z)
    return mpf(sd) * (z * (2 * ncdf(z) - 1) + 2 * phi - 1 / mp.sqrt(mp.pi))


def t_cdf(x, df):
    x, df = mpf(x), mpf(df)
    z = df / (df + x * x)
    tail = mp.betainc(df / 2, mpf(1) / 2, 0, z, regularized=True) / 2
    return 1 - tail if x > 0 else tail


def t_ppf(p, df):
    return mp.findroot(lambda x: t_cdf(x, df) - mpf(p), mpf(2), tol=mpf(10) ** -40, maxsteps=200)


def pit_report(intervals, bins):
    n = len(intervals)
    counts = [mpf(0)] * bins
    mids = []
    for lo, hi in intervals:
        lo, hi = mpf(lo), mpf(hi)
        mids.append((lo + hi) / 2)
        if hi > lo:
            for b in range(bins):
                left, right = mpf(b) / bins, mpf(b + 1) / bins
                counts[b] += max(min(hi, right) - max(lo, left), 0) / (hi - lo)
        else:
            idx = min(int(mp.floor(lo * bins)), bins - 1) if lo > 0 else 0
            counts[max(idx, 0)] += 1
    expected = mpf(n) / bins
    srt = sorted(mids)
    ks = max(max(mpf(i + 1) / n - u, u - mpf(i) / n) for i, u in enumerate(srt))
    return {
        "counts": [float(c) for c in counts],
        "max_bin_deviation": float(max(abs(c / n - mpf(1) / bins) for c in counts)),
        "chi_square": float(sum((c - expected) ** 2 / expected for c in counts)),
        "ks": float(ks),
        "mid_pit": [float(m) for m in mids],
    }


def coverage_report(levels, per_case_interval, outcomes):
    out = []
    for level in levels:
        cov, width = 0, mpf(0)
        for law_interval, y in zip(per_case_interval(level), outcomes):
            lo, hi = law_interval
            cov += 1 if lo <= y <= hi else 0
            width += hi - lo
        out.append({"level": level, "coverage": cov / len(outcomes), "width": float(width / len(outcomes))})
    return out


def type6(sorted_draws, q):
    d = len(sorted_draws)
    h = min(max(mpf(q) * (d + 1), 1), d) - 1
    lo, hi = int(mp.floor(h)), int(mp.ceil(h))
    if lo == hi:
        return mpf(sorted_draws[lo])
    return mpf(sorted_draws[lo]) + (mpf(sorted_draws[hi]) - mpf(sorted_draws[lo])) * (h - lo)


def variance_mean(d, kind, lag=None, order=None, labels=None):
    n = len(d)
    if kind == "iid":
        m = sum(d) / n
        return sum((x - m) ** 2 for x in d) / (n - 1) / n
    if kind == "serial":
        dd = [d[i] for i in order]
        m = sum(dd) / n
        c = [x - m for x in dd]
        gamma = lambda k: sum(c[i + k] * c[i] for i in range(n - k)) / n
        lr = gamma(0) + sum(2 * (1 - mpf(k) / (lag + 1)) * gamma(k) for k in range(1, lag + 1))
        return max(lr, 0) / n
    m = sum(d) / n
    sums = {}
    for x, g in zip(d, labels):
        sums[g] = sums.get(g, 0) + (x - m)
    G = len(sums)
    return mpf(G) / (G - 1) * sum(s * s for s in sums.values()) / (n * n)


def interval(values, kind, level, **kw):
    n = len(values)
    var = variance_mean(values, kind, **kw)
    if kind == "iid":
        df = mpf(n - 1)
    elif kind == "serial":
        df = max(mpf(n) / (kw["lag"] + 1) - 1, 1)
    else:
        df = mpf(len(set(kw["labels"])) - 1)
    se = mp.sqrt(var)
    crit = t_ppf(1 - (1 - mpf(level)) / 2, df)
    m = sum(values) / n
    return {"mean": float(m), "se": float(se), "df": float(df), "lower": float(m - crit * se), "upper": float(m + crit * se)}


# ---------------- Gaussian, two predictors ----------------
N = 60
means = [0.15 * i - 4.0 for i in range(N)]
sd_a, sd_b = 1.0, 1.4
outcomes = [m + rng.gauss(0, 1.0) for m in means]
outcomes[7] += 4.5  # an outlier case
mean_b = [m + 0.3 for m in means]
perm = list(range(N))
rng.shuffle(perm)
time_index = perm  # a shuffled time index: the order of the cases in time is not the input order
labels = [rng.randrange(12) for _ in range(N)]
for g in range(12):
    labels[g] = g  # every one of the 12 clusters appears
LEVELS = [0.3, 0.8, 0.95]
BINS = 8
gauss_pit = pit_report([(ncdf((y - m) / sd_a),) * 2 for y, m in zip(outcomes, means)], BINS)
gauss_cov = coverage_report(
    LEVELS,
    lambda level: [(mpf(m) - ninv(1 - (1 - mpf(level)) / 2) * sd_a, mpf(m) + ninv(1 - (1 - mpf(level)) / 2) * sd_a) for m in means],
    outcomes,
)
crps_a = [crps_gauss(m, sd_a, y) for m, y in zip(means, outcomes)]
crps_b = [crps_gauss(m, sd_b, y) for m, y in zip(mean_b, outcomes)]
diff = [a - b for a, b in zip(crps_a, crps_b)]
order_time = sorted(range(N), key=lambda i: time_index[i])
gauss_intervals = {}
for level in (0.9, 0.5):
    gauss_intervals[str(level)] = {
        "iid_mean": interval(crps_a, "iid", level),
        "iid_diff": interval(diff, "iid", level),
        "serial_mean": interval(crps_a, "serial", level, lag=3, order=order_time),
        "serial_diff": interval(diff, "serial", level, lag=3, order=order_time),
        "cluster_mean": interval(crps_a, "cluster", level, labels=labels),
        "cluster_diff": interval(diff, "cluster", level, labels=labels),
    }
gauss = {"means": means, "mean_b": mean_b, "sd_a": sd_a, "sd_b": sd_b, "outcomes": outcomes,
         "time_index": time_index, "labels": labels, "levels": LEVELS, "bins": BINS,
         "pit": gauss_pit, "coverage": gauss_cov, "crps_mean": float(sum(crps_a) / N),
         "intervals": gauss_intervals}

# ---------------- Bernoulli ----------------
PROBS = [0.9, 0.8, 0.8, 0.1, 0.75, 0.25, 0.5, 0.05, 0.95, 0.6, 0.4, 0.99, 0.01, 0.3, 0.7]
bern_p = [PROBS[i % len(PROBS)] for i in range(60)]
bern_y = [1 if rng.random() < p else 0 for p in bern_p]
BLEVELS = [0.5, 0.6, 0.8, 0.9, 0.99]
def bern_interval(p, level):
    from fractions import Fraction as F
    tail = (1 - F(str(level))) / 2
    q0 = 1 - F(str(p))
    low = 0 if q0 >= tail else 1
    high = 0 if q0 >= 1 - tail else 1
    return (mpf(low), mpf(high))
bern_pit = pit_report([((0, 1 - p) if y == 0 else (1 - p, 1)) for p, y in zip(bern_p, bern_y)], 6)
bern_cov = coverage_report(BLEVELS, lambda level: [bern_interval(p, level) for p in bern_p], [mpf(y) for y in bern_y])
rel_bins = 7
rel = []
cnt, fs, hs = [0] * rel_bins, [mpf(0)] * rel_bins, [0] * rel_bins
for p, y in zip(bern_p, bern_y):
    b = min(int(mp.floor(mpf(p) * rel_bins)), rel_bins - 1)
    cnt[b] += 1
    fs[b] += mpf(p)
    hs[b] += y
gap = mpf(0)
for b in range(rel_bins):
    if cnt[b]:
        mf, of = fs[b] / cnt[b], mpf(hs[b]) / cnt[b]
    else:
        mf, of = mpf(0), mpf(0)
    gap += mpf(cnt[b]) / 60 * abs(of - mf)
    rel.append({"count": cnt[b], "mean_forecast": float(mf), "observed": float(of)})
bern = {"p": bern_p, "y": bern_y, "levels": BLEVELS, "pit": bern_pit, "pit_bins": 6, "coverage": bern_cov,
        "reliability_bins": rel_bins, "reliability": rel, "weighted_gap": float(gap),
        "mean_sd": float(sum(mp.sqrt(mpf(p) * (1 - mpf(p))) for p in bern_p) / 60)}

# ---------------- finite real law ----------------
SUP = [0.0, 1.0, 2.0, 3.0]
PATTERNS = [[0.1, 0.2, 0.3, 0.4], [0.25, 0.25, 0.25, 0.25], [0.7, 0.1, 0.1, 0.1], [0.05, 0.15, 0.3, 0.5]]
fin_p = [PATTERNS[i % 4] for i in range(40)]
fin_y = [rng.choice([0.0, 1.0, 2.0, 3.0, 1.5]) for _ in range(40)]
FLEVELS = [0.2, 0.5, 0.6, 0.9]
def fin_quantile(probs, q):
    from fractions import Fraction as F
    cum = F(0)
    for x, pm in zip(SUP, probs):
        cum += F(str(pm))
        if cum >= F(str(q)) - F(1, 10 ** 12):
            return mpf(x)
    return mpf(SUP[-1])
def fin_interval(probs, level):
    from fractions import Fraction as F
    tail = (1 - F(str(level))) / 2
    return (fin_quantile(probs, tail), fin_quantile(probs, 1 - tail))
def fin_pit(probs, y):
    below = sum(mpf(pm) for x, pm in zip(SUP, probs) if x < y)
    at = sum(mpf(pm) for x, pm in zip(SUP, probs) if x <= y)
    return (min(below, 1), min(at, 1))
fin_pit_r = pit_report([fin_pit(pr, y) for pr, y in zip(fin_p, fin_y)], 7)
fin_cov = coverage_report(FLEVELS, lambda level: [fin_interval(pr, level) for pr in fin_p], [mpf(y) for y in fin_y])
def fin_sd(pr):
    m = sum(mpf(x) * mpf(pm) for x, pm in zip(SUP, pr))
    return mp.sqrt(sum(mpf(pm) * (mpf(x) - m) ** 2 for x, pm in zip(SUP, pr)))
fin = {"support": SUP, "probs": fin_p, "y": fin_y, "levels": FLEVELS, "pit": fin_pit_r, "pit_bins": 7,
       "coverage": fin_cov, "mean_sd": float(sum(fin_sd(pr) for pr in fin_p) / 40)}

# ---------------- ensemble ----------------
ens = []
ens_y = []
for i in range(40):
    draws = [round(rng.gauss(0, 1), 1) for _ in range(7)]  # coarse: ties with the outcome occur
    ens.append(draws)
    ens_y.append(round(rng.gauss(0, 1), 1))
ELEVELS = [0.3, 0.5, 0.9]
def ens_interval(draws, level):
    s = sorted(mpf(x) for x in draws)
    t = (1 - mpf(level)) / 2
    return (type6(s, t), type6(s, 1 - t))
def ens_pit(draws, y):
    m = len(draws)
    return (mpf(sum(1 for x in draws if x < y)) / m, mpf(sum(1 for x in draws if x <= y)) / m)
ens_pit_r = pit_report([ens_pit(d, y) for d, y in zip(ens, ens_y)], 4)
ens_cov = coverage_report(ELEVELS, lambda level: [ens_interval(d, level) for d in ens], [mpf(y) for y in ens_y])
def ens_sd(d):
    m = sum(mpf(x) for x in d) / len(d)
    return mp.sqrt(sum((mpf(x) - m) ** 2 for x in d) / len(d))
ensd = {"draws": ens, "y": ens_y, "levels": ELEVELS, "pit": ens_pit_r, "pit_bins": 4, "coverage": ens_cov,
        "mean_sd": float(sum(ens_sd(d) for d in ens) / 40)}

print(json.dumps({"gauss": gauss, "bern": bern, "fin": fin, "ens": ensd}))
