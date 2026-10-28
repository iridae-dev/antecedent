# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the scalar linear-Gaussian state-space smoother fixture.

Model: x_t = a x_{t-1} + w_t (Var w = q), y_t = x_t + v_t (Var v = r), x_0 ~ N(m0, p0)
before the first observation, so the first predicted law is (m0, p0).

The reference is the *joint Gaussian conditional*, not a Kalman recursion: the exact joint
covariance of (x_1..x_T, y_1..y_T) is built in rational arithmetic from the declared
parameters (Cov(x_s, x_t) = Var(x_s) a^(t-s) for s <= t), and the filtered,
predicted and smoothed laws are read off by Gaussian conditioning on the observations that
each quantity is allowed to see. The lag-one covariance Cov(x_t, x_{t-1} | y_1..y_T) is read
the same way. Nothing here is a Kalman gain or an RTS backward pass, so the product's
recursions are checked against the definition of the conditional law, not against a copy of
themselves.

Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/gcm/lgssm/reference.py --check

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction as F
from pathlib import Path

HERE = Path(__file__).resolve().parent


def solve(matrix, rhs):
    """Exact Gauss-Jordan solve of matrix X = rhs (rhs is a list of columns)."""
    n = len(matrix)
    aug = [list(row) + [col[i] for col in rhs] for i, row in enumerate(matrix)]
    for c in range(n):
        p = next(r for r in range(c, n) if aug[r][c] != 0)
        aug[c], aug[p] = aug[p], aug[c]
        piv = aug[c][c]
        aug[c] = [v / piv for v in aug[c]]
        for r in range(n):
            if r != c and aug[r][c] != 0:
                f = aug[r][c]
                aug[r] = [a - f * b for a, b in zip(aug[r], aug[c])]
    return [[aug[i][n + j] for i in range(n)] for j in range(len(rhs))]


def derive(parameters, observations):
    a = F(str(parameters["a"]))
    q = F(str(parameters["process_variance"]))
    r = F(str(parameters["observation_variance"]))
    m0 = F(str(parameters["initial_mean"]))
    p0 = F(str(parameters["initial_variance"]))
    y = [F(str(v)) for v in observations]
    n = len(y)
    # The fixture's first predicted law is (m0, p0) itself, so the first state is N(m0, p0)
    # and x_t = a x_{t-1} + w_t afterwards.
    mean = [m0]
    for _ in range(1, n):
        mean.append(a * mean[-1])
    var = [p0]
    for _ in range(1, n):
        var.append(a * a * var[-1] + q)
    cov = [[F(0)] * n for _ in range(n)]
    for s in range(n):
        cov[s][s] = var[s]
        for t in range(s + 1, n):
            cov[s][t] = var[s] * a ** (t - s)
            cov[t][s] = cov[s][t]

    def condition(k, upto):
        """Law of x_k given y_1..y_upto (1-based count upto, 0 for none)."""
        if upto == 0:
            return mean[k], var[k], None
        sig_xy = [cov[k][j] for j in range(upto)]
        sig_yy = [[cov[i][j] + (r if i == j else 0) for j in range(upto)] for i in range(upto)]
        w = solve(sig_yy, [sig_xy])[0]
        mu = mean[k] + sum(w[j] * (y[j] - mean[j]) for j in range(upto))
        v = var[k] - sum(w[j] * sig_xy[j] for j in range(upto))
        return mu, v, w

    out = {
        "filtered_mean": [], "filtered_variance": [], "predicted_mean": [],
        "predicted_variance": [], "smoothed_mean": [], "smoothed_variance": [],
        "lag_one_covariance": [],
    }
    for t in range(n):
        pm, pv, _ = condition(t, t)  # sees y_1..y_{t} (0-based count t)
        fm, fv, _ = condition(t, t + 1)
        sm, sv, _ = condition(t, n)
        out["predicted_mean"].append(pm)
        out["predicted_variance"].append(pv)
        out["filtered_mean"].append(fm)
        out["filtered_variance"].append(fv)
        out["smoothed_mean"].append(sm)
        out["smoothed_variance"].append(sv)
    # lag-one smoothed covariance Cov(x_t, x_{t-1} | y_1..y_n): joint conditioning.
    sig_yy = [[cov[i][j] + (r if i == j else 0) for j in range(n)] for i in range(n)]
    for t in range(n):
        if t == 0:
            out["lag_one_covariance"].append(F(0))
            continue
        wt = solve(sig_yy, [[cov[t][j] for j in range(n)]])[0]
        out["lag_one_covariance"].append(
            cov[t][t - 1] - sum(wt[j] * cov[t - 1][j] for j in range(n))
        )
    return out


def check(document):
    derived = derive(document["parameters"], document["observations"])
    problems = []
    tolerance = float(document["acceptance"]["atol"]) / 100
    for name, values in derived.items():
        frozen = document["reference"][name]
        if len(frozen) != len(values):
            problems.append(f"{name}: length {len(frozen)} != {len(values)}")
            continue
        for i, (want, got) in enumerate(zip(frozen, values)):
            if abs(float(got) - want) > tolerance:
                problems.append(f"{name}[{i}]: frozen {want!r} vs derived {float(got)!r}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.parse_args()
    document = json.loads((HERE / "expected.json").read_text(encoding="utf-8"))
    problems = check(document)
    for p in problems:
        print(p, file=sys.stderr)
    if problems:
        return 1
    print("lgssm reference matches expected.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
