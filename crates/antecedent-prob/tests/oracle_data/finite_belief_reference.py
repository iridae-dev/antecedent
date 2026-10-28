# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Reference values for the aligned finite Bayesian update under a known-variance Gaussian
experimental-mean likelihood, and the linear functional of aligned rows.

Regenerate with:  uv run --script finite_belief_reference.py > finite_belief_reference.json

Written from the stated model, at 80 digits.  Slot s carries a prior mass w_s, a target value
t_s and a paired affine map t_s = a_s + b_s * mu_s from the experimental mean mu_s to the target.
The observed sample mean ybar has the distribution N(mu_s, v / n), so
    posterior_s  proportional to  w_s * (2 pi v / n)^(-1/2) * exp(-(ybar - (t_s - a_s) / b_s)^2 / (2 v / n)),
normalised over the slots.  No Jacobian of the affine map enters, because the density is of the
observed mean, whose value is the same in every slot.  The functional is sum_j row_j * c_j.

Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import json
import random

import mpmath as mp

mp.mp.dps = 80
mpf = mp.mpf
rng = random.Random(20261028)


def posterior(prior, targets, a, b, ybar, v, n):
    var = mpf(v) / n
    logs = []
    for w, t, ai, bi in zip(prior, targets, a, b):
        if mpf(w) == 0:
            logs.append(None)
            continue
        mu = (mpf(t) - mpf(ai)) / mpf(bi)
        logs.append(mp.log(mpf(w)) - (mpf(ybar) - mu) ** 2 / (2 * var) - mp.log(2 * mp.pi * var) / 2)
    m = max(x for x in logs if x is not None)
    ws = [mp.exp(x - m) if x is not None else mpf(0) for x in logs]
    z = sum(ws)
    return [w / z for w in ws]


cases = []


def add(name, prior, targets, a, b, ybar, v, n):
    post = posterior(prior, targets, a, b, ybar, v, n)
    cases.append({"name": name, "prior": prior, "targets": targets, "intercepts": a, "slopes": b,
                  "observed_mean": ybar, "variance": v, "n": n,
                  "posterior": [mp.nstr(x, 30) for x in post]})


add("two_slots", [0.5, 0.5], [0.0, 1.0], [0.0, 0.0], [1.0, 1.0], 0.3, 1.0, 10)
add("negative_slope", [0.2, 0.3, 0.5], [-1.0, 0.5, 2.0], [0.1, -0.2, 0.0], [-2.0, 0.5, -1.5], 0.4, 2.0, 25)
add("far_tail_stable", [0.5, 0.5], [0.0, 30.0], [0.0, 0.0], [1.0, 1.0], 0.0, 1.0, 1000)
add("zero_prior_slot_stays_zero", [0.0, 0.4, 0.6], [0.0, 1.0, 2.0], [0.0] * 3, [1.0] * 3, 0.0, 1.0, 4)
add("huge_sample", [0.3, 0.7], [1.0, 1.0001], [0.0, 0.0], [1.0, 1.0], 1.00004, 1.0, 10**8)
add("tiny_variance", [0.5, 0.5], [0.0, 1e-3], [0.0, 0.0], [1.0, 1.0], 4e-4, 1e-9, 1)
for i in range(40):
    k = rng.randint(2, 12)
    prior = [round(rng.random(), 4) + 0.01 for _ in range(k)]
    targets = [round(rng.gauss(0, 2), 4) for _ in range(k)]
    a = [round(rng.gauss(0, 1), 4) for _ in range(k)]
    b = [round(rng.choice([-1, 1]) * (0.2 + rng.random() * 3), 4) for _ in range(k)]
    add(f"random_{i}", prior, targets, a, b, round(rng.gauss(0, 1), 4), round(0.1 + rng.random() * 5, 4), rng.randint(1, 200))

rows = []
for i in range(10):
    width = rng.randint(1, 5)
    count = rng.randint(1, 8)
    c = [round(rng.gauss(0, 3), 4) for _ in range(width)]
    data = [round(rng.gauss(0, 5), 4) for _ in range(width * count)]
    out = [mp.nstr(sum(mpf(data[r * width + j]) * mpf(c[j]) for j in range(width)), 30) for r in range(count)]
    rows.append({"coefficients": c, "rows": data, "values": out})
print(json.dumps({"conditioning": cases, "functional": rows}, indent=1))
