# /// script
# requires-python = ">=3.11"
# dependencies = ["mpmath==1.3.0"]
# ///
"""40-digit reference for the lagged linear predictor and its held-level prediction.

Regenerate with:  uv run --script temporal_linear_predictor_reference.py > temporal_linear_predictor_reference.json

The series are x, z and y (n = 64).  The target y(t) is regressed on an intercept, x(t-1), x(t-2)
and z(t-1) over the rows t = 2 .. n - 1 (the longest lag is two, so the first two times are
dropped).  The coefficients are the least-squares solution b = (X'X)^-1 X'y at 40 digits.  Holding
x at a level L replaces BOTH x columns by L (every lag of the held variable), keeps z(t-1) at its
observed value, and returns yhat(t) = b0 + (b1 + b2) L + b3 z(t-1) for every aligned row.
Repository-authored (MIT OR Apache-2.0); mpmath is not a product dependency.
"""
import json

import mpmath as mp

mp.mp.dps = 40


class Lcg:
    def __init__(self, s):
        self.s = s

    def u(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) % 2**64
        return (self.s >> 11) / float(1 << 53)

    def normal(self):
        return sum(self.u() for _ in range(12)) - 6.0


def s(v):
    return mp.nstr(v, 30)


def main():
    rng = Lcg(2718)
    n = 64
    x, z, y = [0.0] * n, [0.0] * n, [0.0] * n
    for t in range(n):
        x[t] = (0.6 * x[t - 1] if t else 0.0) + rng.normal()
        z[t] = (0.3 * z[t - 1] if t else 0.0) + 0.7 * x[t] + rng.normal()
        if t >= 2:
            y[t] = 0.5 + 1.1 * x[t - 1] - 0.4 * x[t - 2] + 0.8 * z[t - 1] + 0.5 * rng.normal()
    rows = list(range(2, n))
    X = mp.matrix(len(rows), 4)
    Y = mp.matrix([mp.mpf(y[t]) for t in rows])
    for i, t in enumerate(rows):
        X[i, 0], X[i, 1], X[i, 2], X[i, 3] = 1, mp.mpf(x[t - 1]), mp.mpf(x[t - 2]), mp.mpf(z[t - 1])
    b = (X.T * X) ** -1 * (X.T * Y)
    levels = [0.0, 1.0, -2.5]
    out = []
    for L in levels:
        out.append([s(b[0] + (b[1] + b[2]) * mp.mpf(L) + b[3] * mp.mpf(z[t - 1])) for t in rows])
    print(json.dumps({"x": x, "z": z, "y": y, "coefficients": [s(v) for v in b],
                      "levels": levels, "predictions": out}))


main()
