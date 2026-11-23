#!/usr/bin/env python3
"""Repository-authored mpmath reference for Gaussian observation log-likelihood terms.

Prints the 60-digit values (rounded to 17 significant digits) frozen in
crates/antecedent-stats/tests/gaussian_observation_reference.rs.  Each term is evaluated from
its definition with mpmath's normal CDF/PDF; nothing is shared with the product kernels.  Not a
pinned environment (mpmath is not a product dependency); run: python3 -I this_file.py
"""
from mpmath import mp, mpf, ncdf, npdf, log, nstr

mp.dps = 60


def show(label, value):
    print(label, nstr(value, 17))


def exact(y, mu, s):
    y, mu, s = map(mpf, (y, mu, s))
    return log(npdf((y - mu) / s) / s)


def left(b, mu, s):
    b, mu, s = map(mpf, (b, mu, s))
    return log(ncdf((b - mu) / s))


def right(b, mu, s):
    b, mu, s = map(mpf, (b, mu, s))
    return log(ncdf(-(b - mu) / s))


def interval_mass(l, u, mu, s):
    l, u, mu, s = map(mpf, (l, u, mu, s))
    zl, zu = (l - mu) / s, (u - mu) / s
    if zl > 0:  # upper tail: difference of survival values avoids cancelling near one
        return ncdf(-zl) - ncdf(-zu)
    return ncdf(zu) - ncdf(zl)


def interval(l, u, mu, s):
    return log(interval_mass(l, u, mu, s))


def truncated(v, l, u, mu, s):
    return exact(v, mu, s) - log(interval_mass(l, u, mu, s))


if __name__ == "__main__":
    show("exact(1.3, 0.4, 0.8)", exact(1.3, 0.4, 0.8))
    show("exact(-50, 0, 1)", exact(-50, 0, 1))
    show("left(0.5, 1, 2)", left(0.5, 1, 2))
    show("left(-40, 0, 1)", left(-40, 0, 1))
    show("left(9, 0, 1)", left(9, 0, 1))
    show("right(0.5, 1, 2)", right(0.5, 1, 2))
    show("right(40, 0, 1)", right(40, 0, 1))
    for l, u, mu, s in [
        (-1e-12, 1e-12, 0, 1), (-3e-9, 1e-9, 0, 1), (0.0, 1e-10, 0, 1), (-1e-12, 3e-12, 0, 2),
        (-1, 2, 0.5, 1.5), (8, 9, 0, 1), (-40, -39, 0, 1), (5, 5.000000001, 0, 1),
        (1, 1.000000000001, 0, 1), (-1.000000000001, -1, 0, 1), (2, 2.2, 0, 1), (2, 2.3, 0, 1),
        (0, 0.2, 0, 1), (0.01, 0.2, 0, 1), (20, 20.2, 0, 1),
    ]:
        show(f"interval({l}, {u}, {mu}, {s})", interval(l, u, mu, s))
    show("truncated(0.3, -1, 2, 0.5, 1.5)", truncated(0.3, -1, 2, 0.5, 1.5))
    show("truncated(0, -1e-12, 1e-12, 0, 1)", truncated(0.0, -1e-12, 1e-12, 0, 1))
    show("truncated(10.5, 10, 11, 0, 1)", truncated(10.5, 10, 11, 0, 1))
    show("truncated(0.2, -inf, 1, 0, 1)", exact(0.2, 0, 1) - log(ncdf(1)))
    show("truncated(0.2, -1, inf, 0, 1)", exact(0.2, 0, 1) - log(ncdf(1)))
