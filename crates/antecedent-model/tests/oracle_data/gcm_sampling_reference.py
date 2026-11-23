# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Weighting do-sampler, Gaussian kernel density and Silverman bandwidth, at 50 digits.

Regenerate with:  uv run --script gcm_sampling_reference.py > gcm_sampling_reference.json

Written from the definitions, with mpmath at 50 digits (the doubles handed to the product are
taken as exact):
  bandwidth     h = max( s * (4/(3n))^(1/5), (max - min) * 1e-6 ),  s the n-1 sample sd
                (Silverman's normal-reference rule); the weighting sampler also floors it at 1e-8
  kernel        K_h(u) = exp(-u^2 / (2 h^2)) / (h sqrt(2 pi))
  confounded    w_i = min( K_h(T_i - t0) / max(f(T_i | parents_i), 1e-300), 1e6 ),  f the normal
                density of the fitted treatment mechanism; weights are normalised to sum one,
                ESS = 1 / sum w_i^2, the estimate is sum w_i Y_i
  root          the outcomes of the units with |T_i - t0| <= tolerance, each with unit weight
  KDE density   (1/n) sum_i K_h(x - v_i), h floored at 1e-8
Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import json

import mpmath as mp

mp.mp.dps = 50
mpf = mp.mpf


def silverman(x):
    n = len(x)
    m = sum(mpf(v) for v in x) / n
    var = sum((mpf(v) - m) ** 2 for v in x) / (n - 1)
    rng = max(x) - min(x)
    return max(mp.sqrt(var) * (mpf(4) / (3 * n)) ** (mpf(1) / 5), mpf(rng) * mpf(10) ** -6)


def kernel(u, h):
    return mp.exp(-(u * u) / (2 * h * h)) / (h * mp.sqrt(2 * mp.pi))


def normal_density(t, mean, sd):
    z = (mpf(t) - mean) / sd
    return mp.exp(-z * z / 2) / (sd * mp.sqrt(2 * mp.pi))


def confounded(name, z, t, y, intercept, coef, sigma, t0):
    n = len(t)
    h = max(silverman(t), mpf(10) ** -8)
    raw, capped = [], 0
    for zi, ti in zip(z, t):
        dens = max(normal_density(ti, mpf(intercept) + mpf(coef) * mpf(zi), mpf(sigma)), mpf("1e-300"))
        w = kernel(mpf(ti) - mpf(t0), h) / dens
        if w > 1_000_000:
            w = mpf(1_000_000)
            capped += 1
        raw.append(w)
    total = sum(raw)
    w = [r / total for r in raw]
    ess = 1 / sum(x * x for x in w)
    mean = sum(wi * mpf(yi) for wi, yi in zip(w, y))
    floor = max(n * 0.05, 2.0)
    return {"name": name, "z": z, "t": t, "y": y, "intercept": intercept, "coef": coef, "sigma": sigma,
            "t0": t0, "bandwidth": float(h), "weights": [float(x) for x in w], "ess": float(ess),
            "mean": float(mean), "capped": capped, "degenerate": bool(ess < floor)}


def main():
    z = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
    t = [-2.75, -1.0, -1.75, 0.5, 0.25, 1.5, 0.5, 2.25, 1.0, 3.5]
    y = [1.0, -0.5, 2.0, 0.25, 3.5, -1.0, 0.75, 4.0, 2.5, -2.0]
    out = {"confounded": [], "root": [], "density": []}
    out["confounded"].append(confounded("central", z, t, y, 0.5, 1.5, 0.75, 0.5))
    out["confounded"].append(confounded("far_tail", z, t, y, 0.5, 1.5, 0.75, 6.5))
    # a narrow fitted conditional makes tiny densities, so some raw weights exceed the 1e6 cap
    out["confounded"].append(confounded("capped_weights", z, t, y, 0.0, 1.0, 0.0625, 1.0))
    # a treatment whose spread is far below the 1e-8 floor of the kernel width
    tt = [1.0 + k * 1e-10 for k in range(10)]
    out["confounded"].append(confounded("narrow_treatment_floor", z, tt, y, 1.0, 0.0, 1e-9, 1.0 + 4e-10))
    # large offsets
    out["confounded"].append(confounded("large_offset", z, [1e6 + v for v in t], y, 1e6, 1.5, 0.75, 1e6 + 0.5))

    tr = [0.0, 1.0, 1.0 + 5e-10, 1.0 - 2e-9, 0.0 + 4e-10, 2.0, 0.4, 0.6, 1.0, 0.0]
    yr = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    for name, tol, t0 in (("default", 1e-9, 1.0), ("exact", 0.0, 1.0), ("wide", 0.5, 0.5), ("none_in_window", 1e-9, 1.5)):
        sel = [yv for tv, yv in zip(tr, yr) if abs(mpf(tv) - mpf(t0)) <= mpf(tol)]
        out["root"].append({"name": name, "t": tr, "y": yr, "tolerance": tol, "t0": t0, "selected": sel})

    vals = [0.0, 0.5, 0.5, 2.0, -3.0, 1e-3]
    for bw in (0.3, 1e-12, None, 2.5):
        h = max(mpf(bw if bw is not None else 1.0), mpf("1e-8"))
        pts = [-3.0, 0.0, 0.5, 1.75, 40.0]
        out["density"].append({"values": vals, "bandwidth": bw, "points": pts,
                               "density": [float(sum(kernel(mpf(x) - mpf(v), h) for v in vals) / len(vals)) for x in pts]})
    out["bandwidth_samples"] = [{"x": x, "h": float(silverman(x))} for x in (
        [0.0, 1.0, 2.0, 3.0, 4.0], [1e6, 1e6 + 1, 1e6 - 1, 1e6 + 3, 1e6 + 0.5], [0.0, 1e9, 1e-9, 2e-9])]
    print(json.dumps(out, indent=1))


main()
