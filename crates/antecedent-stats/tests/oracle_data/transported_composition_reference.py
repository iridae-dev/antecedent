# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Reference for joint_law.transported_composition.

Regenerate with:  uv run --offline --script transported_composition_reference.py > transported_composition_reference.json

Exact rational arithmetic on dyadic inputs.  Two parts, the two numeric routes of the joint-law
module that no other joint-law record covers:

* `paired`: population moments of two aligned draw columns, from the definition
  mean = sum / n, var = sum (x - mean)^2 / n, cov = sum (x - mean_x)(y - mean_y) / n, for columns
  at offsets up to 2^30 (where a one-pass formula would lose every digit), a single row, constant
  columns, perfectly anti-correlated columns and columns of 1 to 64 rows.
* `ranges`: for a declared correlation range [lo, hi] the quantities E[XY] = mx my + rho sx sy,
  Cov = rho sx sy and Var(aX + bY) = a^2 sx^2 + b^2 sy^2 + 2 a b rho sx sy are linear in rho, so
  the exact range is spanned by the endpoint values; the cases include negative and zero weights,
  a zero standard deviation, ranges touching -1 and 1 and means of size 2^20.

Repository-authored (MIT OR Apache-2.0); standard library only.
"""
import json
import random
from fractions import Fraction as F

rng = random.Random(20261028)


def fl(x):
    return float(x)


paired = []


def add_paired(name, xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs) / n
    vy = sum((y - my) ** 2 for y in ys) / n
    cv = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
    paired.append(
        {
            "name": name,
            "x": [fl(v) for v in xs],
            "y": [fl(v) for v in ys],
            "mean_x": fl(mx),
            "mean_y": fl(my),
            "var_x": fl(vx),
            "var_y": fl(vy),
            "covariance": fl(cv),
        }
    )


def dyadic(scale=1):
    return F(rng.randint(-1024, 1024), 256) * scale


for n in (1, 2, 5, 17, 64):
    add_paired(f"plain_{n}", [dyadic() for _ in range(n)], [dyadic() for _ in range(n)])
add_paired("offset_two_to_the_thirty", [F(2**30) + dyadic() for _ in range(20)], [F(-(2**30)) + dyadic() for _ in range(20)])
xs = [dyadic() for _ in range(30)]
add_paired("constant_columns", [F(3, 2)] * 8, [F(-5, 4)] * 8)
add_paired("perfectly_anticorrelated", xs, [-v for v in xs])
add_paired("identical_columns_at_offset", [F(2**25) + v for v in xs], [F(2**25) + v for v in xs])

ranges = []


def add_range(name, mx, sx, my, sy, a, b, lo, hi):
    def at(rho):
        return (mx * my + rho * sx * sy, rho * sx * sy, a * a * sx * sx + b * b * sy * sy + 2 * a * b * rho * sx * sy)

    p_lo, c_lo, v_lo = at(lo)
    p_hi, c_hi, v_hi = at(hi)
    ranges.append(
        {
            "name": name,
            "x_mean": fl(mx), "x_sd": fl(sx), "y_mean": fl(my), "y_sd": fl(sy),
            "a": fl(a), "b": fl(b), "lo": fl(lo), "hi": fl(hi),
            "product_moment": [fl(min(p_lo, p_hi)), fl(max(p_lo, p_hi))],
            "covariance": [fl(min(c_lo, c_hi)), fl(max(c_lo, c_hi))],
            "linear_variance": [fl(min(v_lo, v_hi)), fl(max(v_lo, v_hi))],
        }
    )


add_range("positive_weights", F(3, 2), F(1, 2), F(-2), F(3, 4), F(1), F(1), F(-1, 2), F(1, 2))
add_range("contrast_weights", F(3, 2), F(1, 2), F(-2), F(3, 4), F(1), F(-1), F(0), F(1))
add_range("full_range", F(0), F(2), F(0), F(5, 2), F(2), F(-3), F(-1), F(1))
add_range("negative_range", F(2**20), F(1, 4), F(-(2**20)), F(1, 8), F(-1, 2), F(3, 2), F(-1), F(-1, 2))
add_range("zero_standard_deviation", F(5), F(0), F(7), F(2), F(1), F(1), F(-1), F(1))
add_range("zero_weight", F(5), F(3), F(7), F(2), F(0), F(1), F(-1, 4), F(3, 4))
add_range("narrow", F(1), F(1), F(1), F(1), F(1), F(1), F(1, 4), F(1, 4) + F(1, 1024))
print(json.dumps({"paired": paired, "ranges": ranges}, indent=1))
