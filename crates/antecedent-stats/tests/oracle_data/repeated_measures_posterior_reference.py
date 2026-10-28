# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Reference values for the posterior of arm means under repeated measures per unit.

Regenerate with:  uv run --script repeated_measures_posterior_reference.py > repeated_measures_posterior_reference.json

Written from the model with the full multivariate normal likelihood, at 50 digits: a unit in arm k
contributes y ~ N(mu_k 1, S) with S = sigma2 R built explicitly (compound symmetric, AR(1) or
within I + between 11'), and mu_k ~ N(m_k, p_k).  Conditioning on all units of the arm gives
  precision_k = 1/p_k + n_k (1' S^-1 1),     mean_k = (m_k/p_k + sum_i 1' S^-1 y_i) / precision_k
with S^-1 from mp.lu_solve, without the closed-form generalized-least-squares weights the product
uses.  The contrast posterior is normal with the difference of means and the sum of variances; its
equal-tailed interval uses mp.erfinv and P(contrast > 0) = Phi(mean/sd).

Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import json
import random

import mpmath as mp

mp.mp.dps = 50
mpf = mp.mpf


def txt(x):
    return mp.nstr(x, 30)


def covariance(structure):
    kind = structure["kind"]
    if kind == "cs":
        m, s2, rho = structure["m"], mpf(structure["variance"]), mpf(structure["rho"])
        return mp.matrix([[s2 * (1 if i == j else rho) for j in range(m)] for i in range(m)])
    if kind == "ar":
        m, s2, rho = structure["m"], mpf(structure["variance"]), mpf(structure["rho"])
        return mp.matrix([[s2 * rho ** abs(i - j) for j in range(m)] for i in range(m)])
    m, b, w = structure["m"], mpf(structure["between"]), mpf(structure["within"])
    return mp.matrix([[b + (w if i == j else 0) for j in range(m)] for i in range(m)])


def posterior(structure, priors, units, plus, minus, level):
    S = covariance(structure)
    m = structure["m"]
    ones = mp.matrix([1] * m)
    sinv1 = mp.lu_solve(S, ones)
    info = sum(sinv1)
    arms = []
    for k, (m0, pv) in enumerate(priors):
        mine = [u for u in units if u["arm"] == k]
        total = sum(sum(sinv1[j] * mpf(u["measurements"][j]) for j in range(m)) for u in mine)
        precision = 1 / mpf(pv) + len(mine) * info
        mean = (mpf(m0) / mpf(pv) + total) / precision
        arms.append({"units": len(mine), "mean": mean, "sd": mp.sqrt(1 / precision)})
    p, q = arms[plus], arms[minus]
    mean = p["mean"] - q["mean"]
    sd = mp.sqrt(p["sd"] ** 2 + q["sd"] ** 2)
    z = mp.sqrt(2) * mp.erfinv(level)
    return {"arms": [{"units": a["units"], "mean": txt(a["mean"]), "sd": txt(a["sd"])} for a in arms],
            "contrast_mean": txt(mean), "contrast_sd": txt(sd), "interval": [txt(mean - z * sd), txt(mean + z * sd)],
            "probability_positive": txt(mp.ncdf(mean / sd)), "unit_summary_variance": txt(1 / info)}


def main():
    rnd = random.Random(20261028)
    structures = [
        {"kind": "cs", "m": 2, "variance": 1.0, "rho": 0.5},
        {"kind": "cs", "m": 5, "variance": 2.5, "rho": 0.3},
        {"kind": "cs", "m": 4, "variance": 1.0, "rho": 0.999999},
        {"kind": "cs", "m": 4, "variance": 1.0, "rho": -1 / 3 + 1e-6},
        {"kind": "cs", "m": 10, "variance": 1e-4, "rho": -0.05},
        {"kind": "ar", "m": 3, "variance": 1.0, "rho": 0.7},
        {"kind": "ar", "m": 6, "variance": 4.0, "rho": -0.9},
        {"kind": "ar", "m": 8, "variance": 1.0, "rho": 0.99},
        {"kind": "hier", "m": 4, "between": 0.0, "within": 1.0},
        {"kind": "hier", "m": 5, "between": 2.0, "within": 0.5},
        {"kind": "hier", "m": 3, "between": 1e6, "within": 1.0},
        {"kind": "hier", "m": 20, "between": 1e-8, "within": 3.0},
    ]
    cases = []
    for st in structures:
        arms = rnd.choice([2, 3])
        priors = [(round(rnd.gauss(0, 1), 3), round(0.1 + 5 * rnd.random(), 3)) for _ in range(arms)]
        priors[0] = (priors[0][0], 1e6)  # a nearly flat prior
        n = rnd.randint(6, 40)
        units = []
        for i in range(n):
            arm = i % arms if i < arms else rnd.randrange(arms)
            units.append({"arm": arm, "measurements": [round(rnd.gauss(arm, 1.5) * 64) / 64 for _ in range(st["m"])],
                          "probability": round(0.1 + 0.8 * rnd.random(), 3)})
        for plus, minus, level in ((1, 0, 0.95), (0, arms - 1, 0.5), (arms - 1, 0, 0.999999)):
            cases.append({"structure": st, "priors": priors, "units": units, "plus": plus, "minus": minus, "level": level,
                          **posterior(st, priors, units, plus, minus, mpf(level))})
    print(json.dumps({"cases": cases}, separators=(",", ":")))


main()
