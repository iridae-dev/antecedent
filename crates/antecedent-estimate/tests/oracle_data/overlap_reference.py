# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Exact overlap diagnostics: clip-sensitivity effective sizes, extreme weights, support and trimming.

Regenerate with:  uv run --script overlap_reference.py > overlap_reference.json

For propensities p_i, binary treatment t_i, a clip c in (0, 1/2) and a target, the observed-arm
weight at clipped e_i = min(max(p_i, c), 1 - c) is
    ATE/Custom: 1/e_i (treated), 1/(1 - e_i) (control);  ATT: 1 (treated), e_i/(1 - e_i) (control);
    ATC: (1 - e_i)/e_i (treated), 1 (control),
times the observation weight for Custom.  Kish size = (sum w)^2 / sum w^2, over all rows, the treated
rows and the control rows; the extreme count is #{w > 10}.  The grid is {c/2, c, min(2c, 0.49)} with
consecutive duplicates dropped.  Support is the share of propensities in [c, 1 - c] (or [trim, 1 - trim]
without a clip), the excluded share is the share strictly outside [trim, 1 - trim].  Fractions from
the exact binary64 inputs, rounded to nearest binary64 at the end.  Repository-authored
(MIT OR Apache-2.0).
"""
import json
import random
from fractions import Fraction as F

rng = random.Random(5)


def weight(target, t, e):
    if target in ("ate", "custom"):
        return 1 / e if t else 1 / (1 - e)
    if target == "att":
        return F(1) if t else e / (1 - e)
    return (1 - e) / e if t else F(1)


def kish(ws):
    s, q = sum(ws, F(0)), sum((w * w for w in ws), F(0))
    return float(s * s / q) if q > 0 else 0.0


def grid(p, t, target, clip, ow):
    cands = [F(clip) / 2, F(clip), F(min(clip * 2, 0.49))]
    thresholds = []
    for c in cands:
        if thresholds and abs(thresholds[-1] - c) < F(1, 10**15):
            continue
        thresholds.append(c)
    out = dict(thresholds=[float(c) for c in thresholds], ess=[], treated_ess=[], control_ess=[], extreme=[])
    for c in thresholds:
        ws = []
        for i, (pi, ti) in enumerate(zip(p, t)):
            e = min(max(F(pi), c), 1 - c)
            w = weight(target, ti, e)
            if ow is not None:
                w *= F(ow[i])
            ws.append(w)
        out["ess"].append(kish(ws))
        out["treated_ess"].append(kish([w for w, ti in zip(ws, t) if ti]))
        out["control_ess"].append(kish([w for w, ti in zip(ws, t) if not ti]))
        out["extreme"].append(sum(1 for w in ws if w > 10))
    return out


def case(name, n, target, clip, trim, use_ow, shape=0.7):
    p = [min(0.999, max(0.001, rng.betavariate(shape, shape))) for _ in range(n)]
    t = [rng.random() < pi for pi in p]
    ow = [rng.choice([0.5, 1.0, 1.5, 3.0]) for _ in range(n)] if use_ow else None
    lo = clip if clip is not None else (trim if trim is not None else 0.0)
    support = sum(1 for pi in p if F(lo) <= F(pi) <= 1 - F(lo)) / n
    excluded = sum(1 for pi in p if trim is not None and (F(pi) < F(trim) or F(pi) > 1 - F(trim))) / n
    res = dict(name=name, propensities=p, treatment=[1.0 if v else 0.0 for v in t], target=target, clip=clip, trim=trim,
               observation_weights=ow, min=min(p), max=max(p), support=support, excluded=excluded)
    if clip is not None:
        res["grid"] = grid(p, t, target, clip, ow)
    return res


cases = [
    case("ate_default_clip", 60, "ate", 0.01, None, False),
    case("att_clip_with_trim", 60, "att", 0.05, 0.05, False),
    case("atc_wide_clip", 50, "atc", 0.3, None, False),
    case("custom_weights", 60, "custom", 0.02, None, True),
    case("ate_tiny_clip", 80, "ate", 0.004, None, False, 0.2),
    case("trim_only", 40, "ate", None, 0.1, False),
]
print(json.dumps(cases, separators=(",", ":")))
