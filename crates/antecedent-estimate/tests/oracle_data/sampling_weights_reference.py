# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Sampling-weight normalisation, zero-weight accounting and Kish size, in exact rational arithmetic.

Regenerate with:  uv run --script sampling_weights_reference.py > sampling_weights_reference.json

A declared weight vector s (finite, non-negative) over the source rows and a mask of rows the
estimator keeps.  Rows of weight zero leave the mask and are counted.  The retained weights are
s_i * k / sum(s) (mean one over the k retained rows); the record holds k, the zero-weight count,
the Kish size (sum s)^2 / sum s^2 and sum s.  Every quantity is exact on the binary64 inputs;
the normalised weights are written as nearest binary64.  The weight vectors include scales of
1e308 and 1e-320 (sums that overflow or whose reciprocal overflows in binary64) and a ratio of
1e300 between the largest and smallest weight.
Repository-authored (MIT OR Apache-2.0).
"""
import json
import random
from fractions import Fraction as F

rng = random.Random(5)


def solve(weights, mask):
    keep = [m and w > 0 for m, w in zip(mask, weights)]
    zero = sum(1 for m, w in zip(mask, weights) if m and w == 0)
    kept = [F(w) for w, k in zip(weights, keep) if k]
    total = sum(kept)
    k = len(kept)
    return dict(
        weights=weights, mask=mask, zero_weight_rows=zero, rows_used=k,
        normalized=[float(w * k / total) for w in kept],
        ess=float(total * total / sum(w * w for w in kept)),
    )


cases = []
cases.append(dict(name="moderate", **solve([2.0, 0.0, 4.0, 6.0, 0.0, 1.5], [True, True, True, False, True, True])))
cases.append(dict(name="scale_1e308", **solve([1e308, 2.5e307, 5e307, 2.5e307], [True] * 4)))
cases.append(dict(name="scale_5e_320", **solve([4.94e-320, 1.235e-320, 2.47e-320, 1.235e-320], [True] * 4)))
cases.append(dict(name="ratio_1e300", **solve([1e-150, 1e150, 3.0, 1e-100], [True] * 4)))
cases.append(dict(name="random_spread", **solve([10 ** rng.uniform(-8, 8) for _ in range(30)], [True] * 30)))
print(json.dumps(dict(cases=cases)))
