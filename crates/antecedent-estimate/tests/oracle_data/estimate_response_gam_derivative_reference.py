"""Additive-GAM plug-in Jacobian and directional derivative, exactly.

Regenerate with:  python3 estimate_response_gam_derivative_reference.py > estimate_response_gam_derivative_reference.json

The outcome is fitted by the penalized additive cubic-spline model over the treatments and the
adjusters, each treatment smooth unpenalized (lambda 0) and each adjuster smooth at the caller's
lambda, on the exact (Fraction) joint minimizer with type-7 quantile knots (estimate_response_kennedy_dr_reference.Additive).
For a treatment point `at`:

* gradient_j = f_j'(at_j), the derivative of the j-th treatment smooth by the degree-drop formula
  3 (N_{i,2} / (t_{i+3} - t_i) - N_{i+1,2} / (t_{i+4} - t_{i+1})), and exactly zero outside the half-open
  knot interior [t_3, t_{m-4});
* level = the mean over the sample rows of the fitted function at (at, x_i) (the covariate average of
  an additive model), clamped to the knot range;
* a Jacobian entry on a scale is the gradient (identity), at_j gradient (log-treatment), gradient / level
  (log-outcome) and at_j gradient / level (log-log), the last two needing a positive level; a
  directional derivative is the gradient dotted with the direction.

Repository-authored (MIT OR Apache-2.0); standard library only apart from the shared reference module.
"""
import json

import estimate_response_kennedy_dr_reference as kd
from estimate_response_kennedy_dr_reference import F, Additive


def mk(n, a_off, y_off, x_scale, two_treatments):
    x = [kd.dy(kd.u01() * 4 - 2, 1024) * x_scale for _ in range(n)]
    a1 = [a_off + kd.dy(float(v / x_scale) * 0.5 + (kd.u01() - 0.5) * 1.5, 1024) for v in x]
    a2 = [kd.dy(-0.3 * float(v / x_scale) + (kd.u01() - 0.5) * 2, 1024) for v in x]
    ys = []
    for scale_ in (1.0, -0.6):
        ys.append([y_off + kd.dy(scale_ * (1.5 * float(p - a_off) - 0.7 * float(q) + 0.5 * float(p - a_off) ** 2) + 0.8 * float(w / x_scale) + (kd.u01() - 0.5), 1024)
                   for p, q, w in zip(a1, a2, x)])
    return a1, a2, x, ys


def case(name, n, a_off, y_off, x_scale, two, K, lam, points):
    a1, a2, x, ys = mk(n, a_off, y_off, x_scale, two)
    treat = [a1, a2] if two else [a1]
    t = len(treat)
    results = []
    fits = [Additive(treat + [x], y, K, [F(0)] * t + [lam]) for y in ys]
    for at in points:
        grads, levels = [], []
        for fit in fits:
            grads.append([fit.derivative(j, at[j]) for j in range(t)])
            levels.append(sum(fit.predict(list(at) + [x[i]]) for i in range(n)) / n)
        entry = {"at": [float(v) for v in at],
                 "gradient": [[str(float(g)) if False else format_exact(g) for g in row] for row in grads],
                 "level": [format_exact(l) for l in levels]}
        results.append(entry)
    return {"name": name, "basis": K, "lambda": float(lam), "treatments": t,
            "a1": [str(float(v)) for v in a1], "a2": [str(float(v)) for v in a2], "x": [str(float(v)) for v in x],
            "y": [[str(float(v)) for v in y] for y in ys], "points": results}


def format_exact(v):
    from decimal import Decimal, getcontext
    getcontext().prec = 40
    v = F(v)
    return str(Decimal(v.numerator) / Decimal(v.denominator))


if __name__ == "__main__":
    cases = []
    cases.append(case("one_treatment", 70, F(0), F(0), F(1), False, 5, F(1),
                      [[F(-1, 5)], [F(3, 10)], [F(0)], [F(5)], [F(-5)]]))
    cases.append(case("two_treatments_two_outcomes", 90, F(0), F(0), F(1), True, 5, F(1),
                      [[F(1, 10), F(-1, 5)], [F(-3, 10), F(2, 5)]]))
    cases.append(case("offsets_and_scales", 80, F(1000), F(100000), F(250), False, 6, F(1, 2),
                      [[F(1000)], [F(1000) + F(1, 4)]]))
    print(json.dumps({"cases": cases}, indent=1))
