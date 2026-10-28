"""Independent truth values for the certified global bound of a bounded objective expression.

Regenerate with:  python3 certified_program_search_reference.py > certified_program_search_reference.json

Each case is an expression tree over bounded controls with optional linear restrictions
sum_i c_i x_i <= b. The true maximum over the feasible set is found without interval arithmetic:
  * one control: a dense scan of 2^16 + 1 points of the feasible interval, then a golden-section
    polish inside the two cells around the best scan point (the objective is smooth or piecewise
    smooth with one extremum per cell for every case here);
  * two controls: a 1201 x 1201 scan of the box restricted to the feasible set, then coordinate
    golden-section polish from the best scan point.
The result is a high-accuracy value T of the true maximum (error recorded as `truth_error`, at most
1e-9 for one control and 1e-5 for two). The product's certificate must satisfy
    incumbent.lower <= T + slack  and  global_upper >= T - slack,
and, when it says Certified, T - incumbent.lower <= gap + truth_error.

Expressions use dyadic constants so the product and this script evaluate the same function.

Repository-authored (MIT OR Apache-2.0); standard library only.
"""
import json
import math
import random


def ev(e, x):
    op = e["op"]
    if op == "var":
        return x[e["i"]]
    if op == "const":
        return e["v"]
    if op == "neg":
        return -ev(e["x"], x)
    a, b = ev(e["l"], x), ev(e["r"], x)
    return {"add": a + b, "sub": a - b, "mul": a * b, "min": min(a, b), "max": max(a, b),
            "div": (a / b if b != 0 else math.nan)}[op]


V = lambda i: {"op": "var", "i": i}
C = lambda v: {"op": "const", "v": v}
B = lambda op, l, r: {"op": op, "l": l, "r": r}
N = lambda x: {"op": "neg", "x": x}


def feasible(x, controls, restrictions):
    if any(not (lo <= xi <= hi) for xi, (lo, hi) in zip(x, controls)):
        return False
    return all(sum(c * xi for c, xi in zip(r["coefficients"], x)) <= r["bound"] for r in restrictions)


def golden(f, lo, hi, iters=80):
    g = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    return max(fc, fd)


def truth_1d(e, controls, restrictions):
    lo, hi = controls[0]
    r = restrictions
    for q in r:
        c = q["coefficients"][0]
        if c > 0:
            hi = min(hi, q["bound"] / c)
        elif c < 0:
            lo = max(lo, q["bound"] / c)
        elif q["bound"] < 0:
            return None
    if lo > hi:
        return None
    n = 1 << 16
    pts = [lo + (hi - lo) * k / n for k in range(n + 1)]
    vals = [ev(e, [p]) for p in pts]
    k = max(range(n + 1), key=lambda i: vals[i])
    best = vals[k]
    a, b = pts[max(k - 1, 0)], pts[min(k + 1, n)]
    best = max(best, golden(lambda t: ev(e, [t]), a, b))
    return best, 1e-9


def truth_2d(e, controls, restrictions):
    (l0, h0), (l1, h1) = controls
    n = 1200
    best, arg = -math.inf, None
    for i in range(n + 1):
        x = l0 + (h0 - l0) * i / n
        for j in range(n + 1):
            y = l1 + (h1 - l1) * j / n
            if feasible([x, y], controls, restrictions):
                v = ev(e, [x, y])
                if v > best:
                    best, arg = v, (x, y)
    if arg is None:
        return None
    x, y = arg
    sx, sy = (h0 - l0) / n * 2, (h1 - l1) / n * 2
    for _ in range(6):
        fx = lambda t: ev(e, [t, y]) if feasible([t, y], controls, restrictions) else -math.inf
        v = golden(fx, max(l0, x - sx), min(h0, x + sx))
        # move to the better point along x
        cand = [max(l0, x - sx) + k * (min(h0, x + sx) - max(l0, x - sx)) / 200 for k in range(201)]
        x = max(cand, key=fx)
        fy = lambda t: ev(e, [x, t]) if feasible([x, t], controls, restrictions) else -math.inf
        cand = [max(l1, y - sy) + k * (min(h1, y + sy) - max(l1, y - sy)) / 200 for k in range(201)]
        y = max(cand, key=fy)
        best = max(best, fx(x), fy(y))
    return best, 1e-5


def case(name, e, controls, restrictions, initial, gap, splits):
    t = (truth_1d if len(controls) == 1 else truth_2d)(e, controls, restrictions)
    assert t is not None, name
    assert feasible(initial, controls, restrictions), name
    return dict(name=name, expr=e, controls=controls, restrictions=restrictions, initial=initial,
                gap=gap, max_splits=splits, truth=t[0], truth_error=t[1])


def main():
    x = V(0)
    y = V(1)
    one = C(1.0)
    cases = []
    cases.append(case("parabola", B("mul", x, B("sub", one, x)), [[0.0, 1.0]], [], [0.0], 1e-4, 100000))
    cases.append(case("cubic_two_interior_peaks", B("add", B("mul", B("mul", x, x), B("sub", C(3.0), B("mul", C(2.0), x))), B("mul", C(-2.0), B("mul", x, B("mul", x, B("mul", x, x))))), [[0.0, 2.0]], [], [0.0], 1e-4, 100000))
    pk = lambda at, h: B("sub", C(h), B("mul", B("sub", x, C(at)), B("sub", x, C(at))))
    cases.append(case("two_peaks_max", B("max", pk(0.25, 3.0), pk(0.75, 4.0)), [[0.0, 1.0]], [], [0.25], 1e-4, 100000))
    cases.append(case("two_peaks_min_valley", B("min", pk(0.25, 3.0), pk(0.75, 4.0)), [[0.0, 1.0]], [], [0.0], 1e-4, 100000))
    cases.append(case("budget_binds", x, [[0.0, 1.0]], [dict(coefficients=[1.0], bound=0.625)], [0.0], 1e-3, 5000))
    cases.append(case("budget_below_peak", B("mul", x, B("sub", one, x)), [[0.0, 1.0]], [dict(coefficients=[1.0], bound=0.25)], [0.0], 1e-4, 100000))
    cases.append(case("negative_lower_box", B("sub", B("mul", x, B("mul", x, x)), B("mul", C(0.75), x)), [[-1.0, 1.0]], [], [0.0], 1e-4, 100000))
    cases.append(case("ratio_separated_denominator", B("div", x, B("add", x, one)), [[0.0, 4.0]], [], [0.0], 1e-4, 100000))
    cases.append(case("ratio_with_interior_peak", B("div", B("mul", x, B("sub", C(2.0), x)), B("add", B("mul", x, x), one)), [[0.0, 2.0]], [], [0.0], 1e-4, 100000))
    cases.append(case("large_offset_small_variation", B("add", C(100000000.0), B("mul", C(0.0625), B("mul", x, B("sub", one, x)))), [[0.0, 1.0]], [], [0.0], 1e-3, 100000))
    cases.append(case("tiny_box", B("mul", x, B("sub", one, x)), [[0.5, 0.5000001]], [], [0.5], 1e-12, 100000))
    cases.append(case("fixed_control", B("mul", x, B("sub", one, x)), [[0.375, 0.375]], [], [0.375], 1e-12, 0))
    cases.append(case("kink_at_optimum", N(B("max", B("sub", x, C(0.3)), B("mul", C(2.0), B("sub", C(0.3), x)))), [[0.0, 1.0]], [], [0.0], 1e-4, 100000))
    cases.append(case("monotone_decreasing", N(x), [[0.0, 1.0]], [], [1.0], 1e-6, 1000))
    # two controls
    cases.append(case("separable_concave", B("add", B("mul", x, B("sub", one, x)), B("mul", C(2.0), B("mul", y, B("sub", one, y)))), [[0.0, 1.0], [0.0, 1.0]], [], [0.0, 0.0], 1e-3, 200000))
    cases.append(case("bilinear_budget", B("mul", x, y), [[0.0, 2.0], [0.0, 2.0]], [dict(coefficients=[1.0, 1.0], bound=2.0)], [0.0, 0.0], 1e-3, 200000))
    cases.append(case("bilinear_signed_box", B("mul", x, y), [[-1.0, 1.0], [-1.0, 0.5]], [], [0.0, 0.0], 1e-3, 200000))
    cases.append(case("saddle_with_budget", B("sub", B("mul", x, x), B("mul", y, y)), [[0.0, 1.0], [0.0, 1.0]], [dict(coefficients=[1.0, 1.0], bound=1.5)], [0.0, 0.0], 1e-3, 200000))
    cases.append(case("min_of_controls", B("min", B("mul", C(2.0), x), y), [[0.0, 1.0], [0.0, 1.0]], [dict(coefficients=[1.0, 1.0], bound=1.0)], [0.0, 0.0], 1e-3, 200000))
    cases.append(case("two_peaks_plane", B("max", B("sub", C(2.0), B("add", B("mul", B("sub", x, C(0.25)), B("sub", x, C(0.25))), B("mul", B("sub", y, C(0.25)), B("sub", y, C(0.25))))), B("sub", C(2.5), B("add", B("mul", B("sub", x, C(0.75)), B("sub", x, C(0.75))), B("mul", B("sub", y, C(0.75)), B("sub", y, C(0.75)))))), [[0.0, 1.0], [0.0, 1.0]], [], [0.25, 0.25], 1e-3, 400000))
    print(json.dumps(cases, separators=(",", ":")))


main()
