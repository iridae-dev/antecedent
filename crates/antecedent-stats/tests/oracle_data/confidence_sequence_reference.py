# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Reference values for the bounded-mean confidence sequences, at 80 digits.

Regenerate with:  uv run --script confidence_sequence_reference.py > confidence_sequence_reference.json

Written from the formulas of the two constructions, not from the product source.

Mixture Hoeffding (Robbins 1970; Howard et al. 2021): with sigma = 1/2 and tau^2 = 1/(sigma^2 n0),
  |S_m| < sqrt((1 + m sigma^2 tau^2)/tau^2 * ln((1 + m sigma^2 tau^2)/alpha^2)),  radius = bound / m,
the sequence being the running intersection of mean +- radius with [0, 1].

Predictable plug-in empirical Bernstein (Waudby-Smith and Ramdas 2024):
  mu_hat_i = (1/2 + sum_{j<=i} X_j)/(i + 1),  s2_i = (1/4 + sum_{j<=i} (X_j - mu_hat_j)^2)/(i + 1),
  lambda_t = min(sqrt(2 ln(2/alpha)/(s2_{t-1} t ln(1 + t))), 3/4),  v_t = 4 (X_t - mu_hat_{t-1})^2,
  psi(l) = (-ln(1 - l) - l)/4,
  interval_t = sum lambda X / sum lambda  +-  (ln(2/alpha) + sum v psi(lambda)) / sum lambda.

Besides fixed streams, the miss probability of each sequence over every Bernoulli(p) path of length n
is enumerated exactly (probabilities in rational arithmetic, interval ends at 80 digits); Ville's
inequality bounds it by alpha.

Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import json
import random
from fractions import Fraction

import mpmath as mp

mp.mp.dps = 80
mpf = mp.mpf


def hoeffding_interval(count, total, alpha, n0):
    m, a, n0 = mpf(count), mpf(alpha), mpf(n0)
    sigma2 = mpf(1) / 4
    tau2 = 1 / (sigma2 * n0)
    bound = mp.sqrt((1 + m * sigma2 * tau2) / tau2 * mp.log((1 + m * sigma2 * tau2) / a**2))
    radius = bound / m
    centre = mpf(total) / m
    return centre - radius, centre + radius


class PlugIn:
    def __init__(self, alpha):
        self.level = mp.log(2 / mpf(alpha))
        self.t = 0
        self.sum = mpf(0)
        self.ss = mpf(0)
        self.tilt_sum = mpf(0)
        self.tilted = mpf(0)
        self.penalty = mpf(0)

    def push(self, x):
        x = mpf(x)
        t = self.t + 1
        mu_prev = (mpf(1) / 2 + self.sum) / t
        s2_prev = (mpf(1) / 4 + self.ss) / t
        lam = min(mp.sqrt(2 * self.level / (s2_prev * t * mp.log(1 + t))), mpf(3) / 4)
        psi = (-mp.log(1 - lam) - lam) / 4
        self.tilt_sum += lam
        self.tilted += lam * x
        self.penalty += 4 * (x - mu_prev) ** 2 * psi
        self.sum += x
        mu = (mpf(1) / 2 + self.sum) / (t + 1)
        self.ss += (x - mu) ** 2
        self.t = t
        return self.tilted / self.tilt_sum, (self.level + self.penalty) / self.tilt_sum


class Hoeffding:
    def __init__(self, alpha, n0):
        self.alpha, self.n0, self.t, self.sum = alpha, n0, 0, mpf(0)

    def push(self, x):
        self.t += 1
        self.sum += mpf(x)
        lo, hi = hoeffding_interval(self.t, self.sum, self.alpha, self.n0)
        return (lo + hi) / 2, (hi - lo) / 2


def run(kind, alpha, n0, values, checkpoints):
    seq = PlugIn(alpha) if kind == "plug_in" else Hoeffding(alpha, n0)
    lo, hi = mpf(0), mpf(1)
    out = []
    for i, x in enumerate(values, 1):
        c, r = seq.push(x)
        lo, hi = max(lo, c - r), min(hi, c + r)
        if i in checkpoints:
            out.append({"count": i, "lower": mp.nstr(lo, 40), "upper": mp.nstr(hi, 40), "empty": bool(lo > hi)})
    return out


def streams():
    rnd = random.Random(20261028)
    out = {}
    out["constant_zero"] = [0.0] * 400
    out["constant_one"] = [1.0] * 400
    out["constant_half"] = [0.5] * 400
    out["alternating"] = [float(i % 2) for i in range(400)]
    out["jump"] = [0.0] * 150 + [1.0] * 150
    out["proportions_p015"] = [sum(rnd.random() < 0.15 for _ in range(6)) / 6 for _ in range(600)]
    out["proportions_p05"] = [sum(rnd.random() < 0.5 for _ in range(6)) / 6 for _ in range(600)]
    out["uniform"] = [rnd.random() for _ in range(600)]
    out["tiny_variance"] = [0.5 + 1e-6 * (i % 3 - 1) for i in range(600)]
    out["bernoulli_rare"] = [float(rnd.random() < 0.01) for _ in range(2000)]
    return out


CHECK = {1, 2, 3, 7, 20, 100, 399, 400}


def miss_probability(kind, alpha, n0, p, n):
    """Exact probability, over Bernoulli(p) paths of length n, that the running intersection
    ever excludes p (or becomes empty)."""
    p_exact = Fraction(p).limit_denominator(10**6)
    target = mpf(p_exact.numerator) / p_exact.denominator
    total = Fraction(0)

    def walk(seq_state, depth, prob, lo, hi, values):
        nonlocal total
        if depth == n:
            return
        for x, w in ((0, 1 - p_exact), (1, p_exact)):
            if w == 0:
                continue
            seq = PlugIn(alpha) if kind == "plug_in" else Hoeffding(alpha, n0)
            for v in values:
                seq.push(v)
            c, r = seq.push(x)
            nlo, nhi = max(lo, c - r), min(hi, c + r)
            q = prob * w
            if nlo > target or nhi < target or nlo > nhi:
                # missed at this prefix; every continuation shares the miss
                total += q
                continue
            walk(None, depth + 1, q, nlo, nhi, values + [x])

    walk(None, 0, Fraction(1), mpf(0), mpf(1), [])
    return mp.nstr(mpf(total.numerator) / total.denominator, 40)


def main():
    out = {"streams": [], "radius": [], "miss": []}
    for name, values in streams().items():
        for alpha in (0.5, 0.05, 1e-6):
            for kind in ("plug_in", "hoeffding"):
                n0 = 50.0
                out["streams"].append(
                    {"name": name, "kind": kind, "alpha": alpha, "tuning": n0, "values": values,
                     "readings": run(kind, alpha, n0, values, {c for c in CHECK if c <= len(values)} | {len(values)})}
                )
    for count in (1, 2, 10, 1000, 10**6, 5 * 10**7):
        for alpha in (1e-9, 0.05, 0.99):
            for n0 in (1e-3, 1.0, 100.0, 1e6):
                total = count + n0
                m, a, n0m = mpf(count), mpf(alpha), mpf(n0)
                r = mpf(1) / 2 * mp.sqrt(total * mp.log(total / (n0m * a**2))) / m
                out["radius"].append({"count": count, "alpha": alpha, "tuning": n0, "radius": mp.nstr(r, 40)})
    for kind in ("plug_in", "hoeffding"):
        for alpha in (0.5, 0.3):
            for p in (0.5, 0.3, 0.05):
                n = 13
                out["miss"].append({"kind": kind, "alpha": alpha, "tuning": 4.0, "p": p, "n": n,
                                    "probability": miss_probability(kind, alpha, 4.0, p, n)})
    print(json.dumps(out, separators=(",", ":")))


main()
