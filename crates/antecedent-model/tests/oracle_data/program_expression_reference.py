# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Value-expression semantics evaluated by an independent interpreter.

Regenerate with:  uv run --script program_expression_reference.py > program_expression_reference.json

Random typed expression trees over two input columns are evaluated row by row with the
definitions the expression language documents, in IEEE binary64 (Python floats):
  arithmetic    + - * and /; min and max return NaN if either operand is NaN
  clip          value, lower, upper: NaN if any is NaN or lower > upper, else the clamp
  comparison    IEEE comparisons (every ordered comparison with NaN is false, != is true)
  conditional   if c then a else b selects per row; the unselected branch is evaluated but its
                value (even NaN or infinite) is not used
  indicator     true is 1 and false is 0
  result        a row whose final value is not finite is a refusal ("refused")
The inputs include signed zeros, subnormals, values near the overflow threshold and exact ties,
so the generated cases hit division by zero, overflow, clipping with ties and inverted bounds,
and comparison of equal values.  Repository-authored (MIT OR Apache-2.0).
"""
import json
import math
import random

VALUES = [0.0, -0.0, 1.0, -1.0, 0.5, 2.5, -7.25, 3.0, 0.1, 1e-308, 5e-324, 1e154, 1e308, -1e308, 100.0]


def num(rng, depth):
    if depth == 0 or rng.random() < 0.25:
        r = rng.random()
        if r < 0.4:
            return ["var", rng.randrange(2)]
        return ["const", rng.choice(VALUES)]
    r = rng.random()
    if r < 0.08:
        return ["neg", num(rng, depth - 1)]
    if r < 0.55:
        return [rng.choice(["add", "sub", "mul", "div", "min", "max"]), num(rng, depth - 1), num(rng, depth - 1)]
    if r < 0.68:
        return ["clip", num(rng, depth - 1), num(rng, depth - 1), num(rng, depth - 1)]
    if r < 0.88:
        return ["if", boolean(rng, depth - 1), num(rng, depth - 1), num(rng, depth - 1)]
    return ["ind", boolean(rng, depth - 1)]


def boolean(rng, depth):
    r = rng.random()
    if depth == 0 or r < 0.55:
        return [rng.choice(["lt", "le", "gt", "ge", "eq", "ne"]), num(rng, max(depth - 1, 0)), num(rng, max(depth - 1, 0))]
    if r < 0.7:
        return ["not", boolean(rng, depth - 1)]
    return [rng.choice(["and", "or"]), boolean(rng, depth - 1), boolean(rng, depth - 1)]


def mn(a, b):
    if math.isnan(a) or math.isnan(b):
        return math.nan
    return a if a < b else b if b < a else (a if math.copysign(1, a) < 0 else b) if a == b == 0 else a


def mx(a, b):
    if math.isnan(a) or math.isnan(b):
        return math.nan
    return a if a > b else b if b > a else (a if math.copysign(1, a) > 0 else b) if a == b == 0 else a


def div(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        if a == 0 or math.isnan(a):
            return math.nan
        return math.copysign(math.inf, a) * math.copysign(1, b)


def arith(op, a, b):
    with_inf = {"add": lambda: a + b, "sub": lambda: a - b, "mul": lambda: a * b}
    if op in with_inf:
        return with_inf[op]()
    if op == "div":
        return div(a, b)
    return mn(a, b) if op == "min" else mx(a, b)


def ev(e, row):
    k = e[0]
    if k == "const":
        return e[1]
    if k == "var":
        return row[e[1]]
    if k == "neg":
        return -ev(e[1], row)
    if k in ("add", "sub", "mul", "div", "min", "max"):
        return arith(k, ev(e[1], row), ev(e[2], row))
    if k == "clip":
        v, lo, hi = ev(e[1], row), ev(e[2], row), ev(e[3], row)
        if math.isnan(v) or math.isnan(lo) or math.isnan(hi) or lo > hi:
            return math.nan
        return min(max(v, lo), hi)
    if k in ("lt", "le", "gt", "ge", "eq", "ne"):
        a, b = ev(e[1], row), ev(e[2], row)
        return {"lt": a < b, "le": a <= b, "gt": a > b, "ge": a >= b, "eq": a == b, "ne": a != b}[k]
    if k == "not":
        return not ev(e[1], row)
    if k == "and":
        a, b = ev(e[1], row), ev(e[2], row)
        return a and b
    if k == "or":
        a, b = ev(e[1], row), ev(e[2], row)
        return a or b
    if k == "if":
        c, t, f = ev(e[1], row), ev(e[2], row), ev(e[3], row)
        return t if c else f
    if k == "ind":
        return 1.0 if ev(e[1], row) else 0.0
    raise ValueError(k)


def main():
    rng = random.Random(41)
    cases = []
    for _ in range(220):
        tree = num(rng, rng.choice([2, 3, 4]))
        rows = []
        for _ in range(24):
            row = [rng.choice(VALUES), rng.choice(VALUES)]
            with_warning = ev(tree, row)
            rows.append({"in": row, "out": with_warning if math.isfinite(with_warning) else "refused"})
        cases.append({"expr": tree, "rows": rows})
    # JSON cannot carry -0.0 distinct from 0.0 by default; Python keeps "-0.0".
    print(json.dumps(cases))


main()
