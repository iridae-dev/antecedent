# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Marginal sensitivity model ATE bounds by vertex enumeration, in exact rationals.

Regenerate with:  uv run --script msm_sensitivity_reference.py > msm_sensitivity_reference.json

For one stratum and arm with atoms (y_j, q_j), hidden fraction h (1 - e for the treated arm, e for
the control arm) and mean m = sum q_j y_j, the observed laws determine
    E[Y(arm) | x] = m + h (sum_j q_j rho_j y_j - m),
and the marginal sensitivity set is  { rho : 1/L <= rho_j <= L, sum_j q_j rho_j = 1 }.
The extremes of a linear function over this polytope are attained at a vertex, and a vertex has
every coordinate on a bound except at most one, which is then fixed by the equality.  This script
enumerates every such vertex (no greedy or quantile rule) and takes the extreme values.
ATE bounds: lower = sum_x P(x) (L_1 - U_0), upper = sum_x P(x) (U_1 - L_0).
Tipping point: the smallest L in [1, Lmax] at which the identified-side bound reaches the threshold,
by exact bisection to 1e-30 (the bounds are monotone in L).
Every input is an exactly representable binary64 value (each law is renormalised exactly, the
product accepts a sum within 1e-9 of one); bounds are written as nearest binary64.
Repository-authored (MIT OR Apache-2.0).
"""
import itertools
import json
from fractions import Fraction as F


def arm_range(atoms, hidden, lam):
    lo, hi = 1 / lam, lam
    n = len(atoms)
    total = sum(q for _, q in atoms)  # binary64 probabilities sum to one only to rounding
    atoms = [(y, q / total) for y, q in atoms]
    m = sum(q * y for y, q in atoms)
    best = [None, None]

    def consider(rho):
        v = sum(q * r * y for (y, q), r in zip(atoms, rho))
        for i, better in ((0, lambda a, b: a < b), (1, lambda a, b: a > b)):
            if best[i] is None or better(v, best[i]):
                best[i] = v

    for bits in itertools.product((lo, hi), repeat=n):
        if sum(q * r for (_, q), r in zip(atoms, bits)) == 1:
            consider(list(bits))
        for k in range(n):
            if atoms[k][1] == 0:
                continue
            others = sum(q * r for j, ((_, q), r) in enumerate(zip(atoms, bits)) if j != k)
            rk = (1 - others) / atoms[k][1]
            if lo <= rk <= hi:
                rho = list(bits)
                rho[k] = rk
                consider(rho)
    assert best[0] is not None
    return m + hidden * (best[0] - m), m + hidden * (best[1] - m)


def ate_range(strata, lam):
    lower = upper = F(0)
    for s in strata:
        e = F(s["propensity"])
        t = [(F(y), F(q)) for y, q in zip(s["treated"]["values"], s["treated"]["probabilities"])]
        c = [(F(y), F(q)) for y, q in zip(s["control"]["values"], s["control"]["probabilities"])]
        l1, u1 = arm_range(t, 1 - e, lam)
        l0, u0 = arm_range(c, e, lam)
        lower += F(s["mass"]) * (l1 - u0)
        upper += F(s["mass"]) * (u1 - l0)
    return lower, upper


def tipping(strata, lam_max, threshold):
    ident = ate_range(strata, F(1))[0]
    down = ident >= threshold
    reached = lambda lam: (ate_range(strata, lam)[0] <= threshold) if down else (ate_range(strata, lam)[1] >= threshold)
    if reached(F(1)):
        return {"status": "origin", "lambda": 1.0}
    if not reached(F(lam_max)):
        return {"status": "not_reached", "lambda": None}
    a, b = F(1), F(lam_max)
    for _ in range(100):
        mid = (a + b) / 2
        if reached(mid):
            b = mid
        else:
            a = mid
        if b - a < F(1, 10 ** 30):
            break
    return {"status": "bracketed", "lambda": float((a + b) / 2), "down": down}


def stratum(mass, e, t, c):
    return {"mass": mass, "propensity": e,
            "treated": {"values": t[0], "probabilities": t[1]},
            "control": {"values": c[0], "probabilities": c[1]}}


designs = [
    ("hand-derived binary two strata",
     [stratum(0.5, 0.5, ([0.0, 1.0], [0.2, 0.8]), ([0.0, 1.0], [0.6, 0.4])),
      stratum(0.5, 0.25, ([0.0, 1.0], [0.5, 0.5]), ([0.0, 1.0], [0.7, 0.3]))],
     [1.0, 1.5, 2.0, 7 / 3, 3.0, 7.0, 100.0], [0.0, 0.5]),
    ("zero-probability atom, tied values, unsorted input",
     [stratum(1.0, 0.5, ([2.0, 0.0, 1.0, 1.0], [0.25, 0.25, 0.0, 0.5]),
              ([1.0, 2.0, 0.0, 1.0], [0.5, 0.125, 0.125, 0.25]))],
     [1.0, 2.0, 3.0, 1.25], [0.0, 0.25]),
    ("five atoms, offset 1e6, extreme propensities, three strata",
     [stratum(0.5, 0.001, ([1e6 + 0.5, 1e6 - 3.0, 1e6 + 7.25, 1e6, 1e6 + 1.0], [0.125, 0.25, 0.0625, 0.5, 0.0625]),
              ([1e6 + 2.0, 1e6 - 1.0, 1e6 + 0.25, 1e6 + 5.0, 1e6 - 4.0], [0.25, 0.25, 0.125, 0.25, 0.125])),
      stratum(0.25, 0.999, ([-2.0, 0.5, 3.0, 4.5, -0.25], [0.5, 0.125, 0.125, 0.125, 0.125]),
              ([0.0, 1.5, -1.0, 2.25, 9.0], [0.25, 0.25, 0.25, 0.125, 0.125])),
      stratum(0.25, 0.5, ([5.0, 6.0], [0.75, 0.25]), ([5.5, 4.0], [0.5, 0.5]))],
     [1.0, 1.001, 1.5, 4.0, 1000.0, 1e6], [1e6 - 2.0, 3.0]),
    ("single atom per arm is identified at every Lambda",
     [stratum(0.5, 0.3, ([4.0], [1.0]), ([1.0], [1.0])), stratum(0.5, 0.6, ([2.0], [1.0]), ([3.0], [1.0]))],
     [1.0, 5.0, 50.0], [0.0]),
]

out = []
for name, strata, lams, thresholds in designs:
    pts = []
    for lam in lams:
        lo, hi = ate_range(strata, F(lam))
        pts.append({"lambda": lam, "lower": float(lo), "upper": float(hi)})
    lam_max = lams[-1] if len(lams) > 1 and lams[-1] > 1 else 2.0
    tips = []
    for th in thresholds:
        tips.append({"threshold": th, **tipping(strata, lam_max, F(th))})
    out.append({"name": name, "strata": strata, "points": pts, "lambda_max": lam_max, "tipping": tips})

print(json.dumps({"designs": out}, indent=1))
