"""Exact rational reference for the inverse-density Epanechnikov dose response and fixed dose policy.

Regenerate with:  python3 estimate_policy_continuous_dose_reference.py > estimate_policy_continuous_dose_reference.json

Every input is a dyadic rational, so the product reads exactly the numbers written here.  For a
baseline group g and a target dose d the local weight of a row with dose x and supplied density f
is w = (3/4)(1 - s^2)/f with s = (x - d)/h, for the rows with |s| < 1 (a row exactly on the window
edge is out).  The response is sum(w y)/sum(w), the Kish effective size is (sum w)^2/sum(w^2), the
local sd is sqrt(sum w (y - response)^2/sum w).  The fixed-policy value is sum_g (n_g/n) response_g
at the group's policy (reference) dose; its score for row i of group g is
(n_g/n) w_i/sum(w) (y_i - response_g); the variance is m/(m-1) sum_i score_i^2 with m the smallest
group size, and the paired variance is the same on the score difference.  Values are written as
exact decimal strings (square roots to 50 digits).  Repository-authored (MIT OR Apache-2.0);
standard library only.
"""
import json
from decimal import Decimal, getcontext
from fractions import Fraction as F

getcontext().prec = 60
_state = 0x243F6A8885A308D3


def u01():
    global _state
    _state = (_state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
    return (_state >> 11) / float(1 << 53)


def dec(x):
    x = F(x)
    return str(Decimal(x.numerator) / Decimal(x.denominator))


def dsqrt(x):
    x = F(x)
    return str((Decimal(x.numerator) / Decimal(x.denominator)).sqrt())


def dyadic(x, den=1024):
    return F(int(round(x * den)), den)


def local(rows, target, h):
    """rows: list of (index, dose, density, outcome). Returns weights for rows inside the window."""
    out = []
    for i, x, f, y in rows:
        s = (x - target) / h
        if abs(s) >= 1:
            continue
        out.append((i, F(3, 4) * (1 - s * s) / f, f, y))
    return out


def response_cell(rows, target, h):
    w = local(rows, target, h)
    sw = sum(t[1] for t in w)
    sw2 = sum(t[1] ** 2 for t in w)
    resp = sum(t[1] * t[3] for t in w) / sw
    sd2 = sum(t[1] * (t[3] - resp) ** 2 for t in w) / sw
    return {
        "response": resp,
        "local_rows": len(w),
        "ess": sw * sw / sw2,
        "min_density": min(t[2] for t in w),
        "max_norm_weight": max(t[1] / sw for t in w),
        "sd2": sd2,
        "weights": w,
        "sw": sw,
    }


def make_case(name, groups, dose, density, outcome, h, support, targets, policy, reference, known):
    n = len(groups)
    names = sorted(set(groups))
    by = {g: [(i, dose[i], density[i], outcome[i]) for i in range(n) if groups[i] == g] for g in names}
    response = []
    for g in names:
        for t in targets:
            c = response_cell(by[g], t, h)
            response.append({
                "group": g, "target": dec(t), "response": dec(c["response"]),
                "local_rows": c["local_rows"], "ess": dec(c["ess"]),
                "min_density": dec(c["min_density"]), "max_norm_weight": dec(c["max_norm_weight"]),
                "local_sd": dsqrt(c["sd2"]),
            })
    value = {}
    scores = {"policy": [F(0)] * n, "reference": [F(0)] * n}
    vals = {"policy": F(0), "reference": F(0)}
    min_rows, min_ess, max_w, min_f = None, None, F(0), None
    mg = min(len(by[g]) for g in names)
    for g in names:
        fr = F(len(by[g]), n)
        for kind, rule in (("policy", policy), ("reference", reference)):
            c = response_cell(by[g], rule[g], h)
            vals[kind] += fr * c["response"]
            for i, w, f, y in c["weights"]:
                scores[kind][i] = fr * (w / c["sw"]) * (y - c["response"])
            min_rows = c["local_rows"] if min_rows is None else min(min_rows, c["local_rows"])
            min_ess = c["ess"] if min_ess is None else min(min_ess, c["ess"])
            max_w = max(max_w, c["max_norm_weight"])
            min_f = c["min_density"] if min_f is None else min(min_f, c["min_density"])
    corr = F(mg, mg - 1)
    var = lambda s: sum(t * t for t in s) * corr
    diff = [a - b for a, b in zip(scores["policy"], scores["reference"])]
    out = {
        "policy_value": dec(vals["policy"]), "reference_value": dec(vals["reference"]),
        "incremental_value": dec(vals["policy"] - vals["reference"]),
        "policy_variance": dec(var(scores["policy"])), "reference_variance": dec(var(scores["reference"])),
        "incremental_variance": dec(var(diff)),
        "minimum_local_rows": min_rows, "minimum_effective_sample_size": dec(min_ess),
        "maximum_normalized_weight": dec(max_w), "minimum_dose_density": dec(min_f),
        "interval_supported": bool(
            known and n >= 600 and mg >= 300 and min_rows >= 80 and min_ess >= 50
            and max_w <= F(1, 20) and min_f >= F(1, 5) and var(scores["policy"]) > 0
            and var(scores["reference"]) > 0 and var(diff) > 0),
    }
    return {
        "name": name, "groups": groups, "dose": [dec(x) for x in dose],
        "density": [dec(x) for x in density], "outcome": [dec(x) for x in outcome],
        "bandwidth": dec(h), "min_local_support": support, "targets": [dec(t) for t in targets],
        "policy": {g: dec(policy[g]) for g in names}, "reference": {g: dec(reference[g]) for g in names},
        "density_known": known, "response": response, "fixed": out,
    }


cases = []

# unbalanced groups of 4, 9 and 30 rows, an outcome offset of 1e9, densities over four decades and
# doses placed exactly on the window edge (which the open window excludes)
groups, dose, density, outcome = [], [], [], []
for g, k in (("a", 4), ("b", 9), ("c", 30)):
    for j in range(k):
        groups.append(g)
        dose.append(dyadic(u01() * 2 - 1))
        density.append(F(1, 2) ** int(u01() * 12 - 2) * (1 + F(j % 3, 4)))
        outcome.append(F(1000000000) + dyadic(u01() * 8, 64) + (3 if g == "b" else 0))
# edge ties: the window of group a at target 0 with h = 1 has edges +-1
for j, g in enumerate(groups):
    if g == "a" and j == 0:
        dose[j] = F(1)
    if g == "a" and j == 1:
        dose[j] = F(-1)
    if g == "a" and j == 2:
        dose[j] = F(0)
    if g == "a" and j == 3:
        dose[j] = F(1, 4)
cases.append(make_case("unbalanced_offset_edges", groups, dose, density, outcome, F(1), 2,
                       [F(0), F(1, 4), F(-1, 2)],
                       {"a": F(0), "b": F(1, 8), "c": F(-1, 4)}, {"a": F(1, 4), "b": F(-1, 8), "c": F(0)}, True))

# one row carries almost all the weight: density 2^-12 against 2^6
groups, dose, density, outcome = [], [], [], []
for g in ("p", "q"):
    for j in range(10):
        groups.append(g)
        dose.append(dyadic(0.45 + 0.01 * j, 1024))
        density.append(F(1, 4096) if j == 3 else F(64))
        outcome.append(dyadic(u01() * 4 - 2, 256) + (10 if (g == "q" and j == 3) else 0))
cases.append(make_case("one_row_dominates", groups, dose, density, outcome, F(1, 4), 2,
                       [F(1, 2)], {"p": F(1, 2), "q": F(9, 20)}, {"p": F(11, 20), "q": F(1, 2)}, True))

# an interval-supported design: 2 x 350 rows, constant density 1/2, h = 1/2, known density
groups, dose, density, outcome = [], [], [], []
for g in ("u", "v"):
    for j in range(350):
        groups.append(g)
        x = F(j, 350)
        dose.append(dyadic(float(x), 4096))
        density.append(F(1, 2) + (F(1, 4) if j % 7 == 0 else 0))
        outcome.append(dyadic(u01() * 6 - 3, 256) + (2 if g == "v" else 0) + dose[-1])
for known in (True, False):
    cases.append(make_case("interval_supported_known" if known else "interval_density_not_known",
                           groups, dose, density, outcome, F(1, 2), 80, [F(1, 2)],
                           {"u": F(3, 10), "v": F(7, 10)}, {"u": F(1, 2), "v": F(1, 2)}, known))
print(json.dumps({"cases": cases}, indent=1))
