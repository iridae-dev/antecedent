# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""Reference for the published scalar standard error and its 95% interval.

Regenerate with:  uv run --script estimate_uncertainty_reference.py > estimate_uncertainty_reference.json

Two parts, each written from the stated rule and not from the product code:

selection -- every combination of an analytic SE, a bootstrap SE and first-stage state is
  enumerated and the published (method, SE, level, endpoints, withheld) is decided by the rule
  the facade documents: only a positive finite SE is published; a valid Anderson-Rubin set is
  published alone; with first-stage diagnostics and no valid set the bootstrap SE is never used
  and the analytic SE is published only if positive finite, else withheld; otherwise a positive
  finite bootstrap SE is preferred to a positive finite analytic SE; otherwise nothing.
critical -- the two-sided 0.95 critical value: the normal quantile at 0.975 and, where the
  estimate declares reference degrees of freedom, the Student-t quantile at 0.975, both found
  by 50-digit root finding on the exact distribution functions (regularized incomplete beta for
  t), together with the interval ate -+ crit*SE evaluated in 50 digits.
Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import itertools
import json

import mpmath as mp

mp.mp.dps = 50


def normal_q(p):
    f = lambda x: mp.ncdf(x) - p
    return mp.findroot(f, mp.mpf("1.96"), tol=mp.mpf(10) ** -45)


def t_cdf(x, df):
    x, df = mp.mpf(x), mp.mpf(df)
    tail = mp.betainc(df / 2, mp.mpf(1) / 2, 0, df / (df + x * x), regularized=True) / 2
    return 1 - tail if x >= 0 else tail


def t_q(p, df):
    f = lambda x: t_cdf(x, df) - p
    lo, hi = mp.mpf(0), mp.mpf(10) ** 7
    for _ in range(400):
        mid = (lo + hi) / 2
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def pos_fin(v):
    return v is not None and v == v and v not in (float("inf"),) and v > 0


# values: None = absent (bootstrap only); the string tags are decoded by the test
SE = {"nan": float("nan"), "zero": 0.0, "neg": -1.0, "inf": float("inf"), "ok": 0.15}
BOOT = {"none": None, "nan": float("nan"), "zero": 0.0, "ok": 0.2}
FIRST = {
    "absent": None,
    "no_ar": {"ar": None},
    "ar_ok": {"ar": [0.2, 1.8, 0.95]},
    "ar_inf": {"ar": [float("-inf"), float("inf"), 0.9]},
    "ar_nan": {"ar": [float("nan"), 1.8, 0.95]},
    "ar_level_one": {"ar": [0.2, 1.8, 1.0]},
    "ar_level_nan": {"ar": [0.2, 1.8, float("nan")]},
}


def enc(v):
    if v is None:
        return None
    if v != v:
        return "nan"
    if v == float("inf"):
        return "inf"
    if v == float("-inf"):
        return "-inf"
    return v


selection = []
for (sn, se), (bn, bo), (fn, fs) in itertools.product(SE.items(), BOOT.items(), FIRST.items()):
    out = {"method": "none", "se": None, "lower": None, "upper": None, "withheld": False}
    if fs is not None:
        ar = fs["ar"]
        ar_valid = (
            ar is not None
            and ar[0] == ar[0]
            and ar[1] == ar[1]
            and ar[2] == ar[2]
            and ar[2] not in (float("inf"), float("-inf"))
            and 0.0 <= ar[2] < 1.0
        )
        if ar_valid:
            out.update(method="anderson_rubin", lower=enc(ar[0]), upper=enc(ar[1]), level=ar[2])
        elif pos_fin(se):
            out.update(method="analytic_se", se=se, level=0.95)
        else:
            out.update(withheld=True)
    elif pos_fin(bo):
        out.update(method="bootstrap_se", se=bo, level=0.95)
    elif pos_fin(se):
        out.update(method="analytic_se", se=se, level=0.95)
    selection.append(
        {
            "analytic": enc(se),
            "bootstrap": enc(bo),
            "first_stage": None if fs is None else {"ar": None if fs["ar"] is None else [enc(x) for x in fs["ar"]]},
            "name": f"{sn}/{bn}/{fn}",
            "expected": out,
        }
    )

z = normal_q(mp.mpf("0.975"))
critical = []
for ate, se, df in [
    (1.0, 0.15, None),
    (-3.5e6, 2.5e5, None),
    (1e9, 1e-3, None),
    (0.25, 0.4, 1.0),
    (0.25, 0.4, 2.0),
    (0.25, 0.4, 3.0),
    (2.0, 0.07, 5.0),
    (2.0, 0.07, 9.0),
    (-1.0, 3.0, 30.0),
    (-1.0, 3.0, 7.5),
    (0.0, 1.0, 1000.0),
    (0.0, 1.0, 0.0),
]:
    # df <= 0 declares no usable reference: the normal quantile is used
    q = z if (df is None or df <= 0) else t_q(mp.mpf("0.975"), df)
    half = q * mp.mpf(se)
    critical.append(
        {
            "ate": ate,
            "se": se,
            "df": df,
            "critical": float(q),
            "lower": float(mp.mpf(ate) - half),
            "upper": float(mp.mpf(ate) + half),
        }
    )

print(json.dumps({"selection": selection, "critical": critical, "z": float(z)}, allow_nan=False))
