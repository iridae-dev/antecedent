# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Independent derivation of the Gaussian-process marginal-likelihood grid selection.

For the declared data (24 points on linspace(-2.4, 2.4), outcome sin(1.3 x) + 0.18 cos(3.1 x))
the reference evaluates the exact negative log marginal likelihood of a zero-mean Gaussian process
on the mean-centred outcome with the squared-exponential kernel

    k(x, x') = variance * exp(-(x - x')^2 / (2 length_scale^2)),   noise variance noise_std^2,

over the declared hyperparameter grid (length scale = multiplier * the root-mean-square of the
parent, i.e. the declared ``parent_rms_sd``; noise std = multiplier * the outcome population sd;
signal variance = the outcome population variance), using a Cholesky factorisation written here
(NLML = 1/2 y' K^-1 y + sum log L_ii + n/2 log 2 pi), and reports the argmin and the
runner-up value. Standard library only; imports no Antecedent code and no statistics package.

    uv run --frozen --script conformance/gcm/gaussian_process/reference.py --check

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def cholesky(a):
    n = len(a)
    lower = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = a[i][j] - sum(lower[i][k] * lower[j][k] for k in range(j))
            lower[i][j] = math.sqrt(s) if i == j else s / lower[j][j]
    return lower


def nlml(x, centred, length_scale, noise_std, variance):
    n = len(x)
    k = [
        [
            variance * math.exp(-((x[i] - x[j]) ** 2) / (2 * length_scale**2))
            + (noise_std**2 if i == j else 0.0)
            for j in range(n)
        ]
        for i in range(n)
    ]
    lower = cholesky(k)
    z = [0.0] * n
    for i in range(n):
        z[i] = (centred[i] - sum(lower[i][j] * z[j] for j in range(i))) / lower[i][i]
    quad = sum(v * v for v in z)  # y' K^-1 y = |L^-1 y|^2
    logdet_half = sum(math.log(lower[i][i]) for i in range(n))
    return 0.5 * quad + logdet_half + 0.5 * n * math.log(2 * math.pi)


def derive(document):
    n = document["data"]["n"]
    x = [-2.4 + 4.8 * i / (n - 1) for i in range(n)]
    y = [math.sin(1.3 * v) + 0.18 * math.cos(3.1 * v) for v in x]
    mean = sum(y) / n
    variance = sum((v - mean) ** 2 for v in y) / n
    outcome_sd = math.sqrt(variance)
    parent_rms_sd = math.sqrt(sum(v * v for v in x) / n)
    centred = [v - mean for v in y]
    cells = []
    for lm in document["grid"]["length_scale_multipliers_of_parent_rms_sd"]:
        for nm in document["grid"]["noise_std_multipliers_of_outcome_sd"]:
            length_scale, noise_std = lm * parent_rms_sd, nm * outcome_sd
            cells.append((nlml(x, centred, length_scale, noise_std, variance), length_scale, noise_std))
    cells.sort()
    return {
        "length_scale": cells[0][1],
        "noise_std": cells[0][2],
        "variance": variance,
        "mean": mean,
        "nlml": cells[0][0],
        "runner_up_nlml": cells[1][0],
        "parent_rms_sd": parent_rms_sd,
        "outcome_sd": outcome_sd,
    }


def check(document):
    derived = derive(document)
    problems = []
    frozen = dict(document["reference"])
    frozen["parent_rms_sd"] = document["grid"]["parent_rms_sd"]
    frozen["outcome_sd"] = document["grid"]["outcome_sd"]
    for key, got in derived.items():
        want = frozen[key]
        if abs(got - want) > 1e-9 * max(1.0, abs(want)):
            problems.append(f"{key}: frozen {want!r} vs derived {got!r}")
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
    print("gaussian_process reference matches expected.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
