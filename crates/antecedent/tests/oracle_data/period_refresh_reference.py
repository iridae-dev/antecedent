"""Exact rational reference for the autoregressive period refresh.

Regenerate with:  python3 period_refresh_reference.py > period_refresh_reference.json

Every series value is a dyadic rational (an integer over 64 plus an exactly representable
offset), so the rows the product forms are the rows written here with no rounding.  For each
scenario the reference replays the declared periods: it forms the lag rows [1, v_{t-1}, ..., v_{t-L}] -> v_t
for the targets inside a period (and, when the period starts where the summarized one ended and the
mechanism is declared invariant, for the targets that reach back into the carried tail), solves the
normal equations of the retained rows with Fractions, and forms the conjugate normal-inverse-gamma
posterior of those rows under the stated prior by the definition
    Lambda = Lambda0 + X'X,  eta = Lambda0 m0 + X'y,  mean = Lambda^-1 eta,
    shape = a0 + n/2,        rate = b0 + (m0'Lambda0 m0 + y'y - mean' eta)/2.
Restarted periods (mechanism not declared invariant) are fitted on the new period alone.  Values
are written as 40-digit decimal strings of the exact rationals.  Repository-authored
(MIT OR Apache-2.0); standard library only.
"""
import json
from decimal import Decimal, getcontext
from fractions import Fraction as F

getcontext().prec = 60

_state = 0x9E3779B97F4A7C15


def u01():
    global _state
    _state = (_state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
    return (_state >> 11) / float(1 << 53)


def dec(x):
    x = F(x)
    return str(Decimal(x.numerator) / Decimal(x.denominator))


def solve(a, b):
    n = len(b)
    a = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        piv = next(r for r in range(c, n) if a[r][c] != 0)
        a[c], a[piv] = a[piv], a[c]
        for r in range(n):
            if r != c and a[r][c] != 0:
                f = a[r][c] / a[c][c]
                a[r] = [x - f * y for x, y in zip(a[r], a[c])]
    return [a[i][n] / a[i][i] for i in range(n)]


def lag_rows(series, first_time, lags):
    out = []
    for t in range(lags, len(series)):
        out.append((first_time + t, [F(1)] + [series[t - k] for k in range(1, lags + 1)], series[t]))
    return out


def fit(rows, prior):
    p = len(prior["mean"])
    xtx = [[sum(r[1][i] * r[1][j] for r in rows) for j in range(p)] for i in range(p)]
    xty = [sum(r[1][i] * r[2] for r in rows) for i in range(p)]
    yty = sum(r[2] * r[2] for r in rows)
    beta = solve(xtx, xty)
    l0, m0 = prior["precision"], prior["mean"]
    eta0 = [sum(l0[i][j] * m0[j] for j in range(p)) for i in range(p)]
    lam = [[l0[i][j] + xtx[i][j] for j in range(p)] for i in range(p)]
    eta = [eta0[i] + xty[i] for i in range(p)]
    mean = solve(lam, eta)
    c = sum(m0[i] * eta0[i] for i in range(p)) + yty
    rate = prior["rate"] + (c - sum(mean[i] * eta[i] for i in range(p))) / 2
    return {
        "n": len(rows),
        "row_ids": [r[0] for r in rows],
        "beta": [dec(b) for b in beta],
        "mean": [dec(m) for m in mean],
        "shape": dec(prior["shape"] + F(len(rows), 2)),
        "rate": dec(rate),
    }


def ar_series(n, offset, phi, noise_scale):
    v = [F(offset)]
    for _ in range(n - 1):
        eps = F(int((u01() - F(1, 2)) * 64), 64) * noise_scale
        v.append(F(offset) + F(phi) * (v[-1] - offset) + eps)
    return [F(int(x * 64), 64) for x in v]


def prior_of(p, scale):
    return {
        "mean": [F(0)] * p,
        "precision": [[(F(1, 2) + F(i, 5)) * scale if i == j else F(0) for j in range(p)] for i in range(p)],
        "shape": F(3),
        "rate": F(2),
    }


def scenario(name, lags, standing, specs, prior):
    """specs: list of (start, values) in order; period end = start + len(values)."""
    steps = []
    kept = []  # retained rows
    tail = []
    prev_end = None
    for k, (start, values) in enumerate(specs):
        end = start + len(values)
        if k == 0:
            kept = lag_rows(values, start, lags)
            outcome = "start"
        elif standing == "declared":
            contiguous = start == prev_end
            if contiguous:
                series = tail + values
                rows = lag_rows(series, start - len(tail), lags)
            else:
                rows = lag_rows(values, start, lags)
            kept = kept + rows
            outcome = "extended_contiguous" if contiguous else "extended_gap"
        else:
            kept = lag_rows(values, start, lags)
            outcome = "restarted"
        series_for_tail = (tail + values) if (standing == "declared" and k > 0 and start == prev_end) else values
        tail = series_for_tail[-lags:]
        prev_end = end
        if len(kept) == 0:
            steps.append({"start": start, "values": [dec(v) for v in values], "outcome": outcome, "refused": True})
            continue
        step = {"start": start, "values": [dec(v) for v in values], "outcome": outcome}
        step.update(fit(kept, prior))
        steps.append(step)
    return {
        "name": name,
        "lags": lags,
        "standing": standing,
        "prior_scale": dec(prior["_scale"]),
        "steps": steps,
    }


def with_scale(scale, p):
    pr = prior_of(p, scale)
    pr["_scale"] = scale
    return pr


cases = []
# offset 1e6 series, three contiguous periods
a = ar_series(95, 10000, F(3, 5), F(1, 3))
cases.append(scenario("offset_three_contiguous", 2, "declared",
                      [(0, a[:40]), (40, a[40:70]), (70, a[70:])], with_scale(F(1), 3)))
# the same shape at an offset of 1e6: the Gram of [1, v, v'] is conditioned near 1e14 relative to
# the prior, past what a double-precision Cholesky resolves, so the product may refuse it
a6 = ar_series(60, 1000000, F(3, 5), F(1, 3))
big = scenario("offset_million_may_refuse", 2, "declared",
               [(0, a6[:30]), (30, a6[30:])], with_scale(F(1), 3))
big["may_refuse"] = True
cases.append(big)
# a gap: the second period starts later than the first ended; the spanning rows are not formed
b = ar_series(70, 40, F(1, 2), F(1, 2))
cases.append(scenario("gap_not_filled", 2, "declared",
                      [(0, b[:30]), (35, b[30:55]), (60, b[55:])], with_scale(F(1), 3)))
# periods shorter than the lag order, contiguous and across a gap; lag order 3
c = ar_series(40, 5, F(2, 5), F(1, 2))
cases.append(scenario("tiny_periods_lags3", 3, "declared",
                      [(0, c[:12]), (12, c[12:14]), (14, c[14:15]), (20, c[15:17]), (22, c[17:30])],
                      with_scale(F(1), 4)))
# restart on the new period alone
d = ar_series(60, 3, F(1, 2), F(1, 2))
cases.append(scenario("undeclared_restart", 1, "undeclared",
                      [(0, d[:40]), (40, d[40:50]), (60, d[50:])], with_scale(F(1), 2)))
# nearly collinear columns: a slow ramp with tiny jitter, large offset, weak prior
e = [F(2000) + F(i, 64) + F(int((u01() - F(1, 2)) * 8), 4096) for i in range(60)]
e = [F(int(x * 4096), 4096) for x in e]
cases.append(scenario("near_collinear_ramp", 2, "declared",
                      [(0, e[:30]), (30, e[30:])], with_scale(F(1, 1000), 3)))
# a heavy prior against few rows
f = ar_series(14, 0, F(1, 2), F(2))
cases.append(scenario("strong_prior_few_rows", 2, "declared",
                      [(0, f[:7]), (7, f[7:])], with_scale(F(40), 3)))
print(json.dumps({"cases": cases}, indent=1))
