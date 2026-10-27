# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Exact conjugate-algebra reference for the Bayesian retention fixture.

Rebuilds the three declared designs row by row in exact rational arithmetic and evaluates
the closed-form posterior moments from the normal equations:

* Gaussian: 256 rows Y = 1 + 2T + noise (noise +-2/5), coefficient prior precision 1/100,
  known residual variance 1; effect = treatment coefficient.
* Quadratic basis: 240 rows over [1, T, Z, TZ, Z^2], prior precision 1/100, inverse-gamma
  shape and scale starting at 1/1000; the posterior effect variance is
  scale / (shape - 1) * (P^-1)[1][1] with shape = 1/1000 + n/2 and
  scale = 1/1000 + (RSS + 0.01 |b|^2) / 2.
* Joint internal candidate: two sources of 128 rows on the Gaussian design, invariant
  coefficient prior precision 1/100, varying intercept precision 1/4, coefficient order
  [T, 1].

No Antecedent code, no sampling and no frozen native draws.  The `prior_transfer` block
holds the declared structural effects of the source and target generators; it is a
bracketing bound for the product's transferred posterior and is not derived here.

    uv run --frozen --script conformance/recalculation/bayesian_retention/reference.py --check
    uv run --frozen --script conformance/recalculation/bayesian_retention/reference.py --self-test

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from fractions import Fraction as F
from pathlib import Path

HERE = Path(__file__).resolve().parent
ATOL = 1e-12


def inverse(matrix):
    n = len(matrix)
    m = [row[:] + [F(int(i == j)) for j in range(n)] for i, row in enumerate(matrix)]
    for i in range(n):
        pivot = next(r for r in range(i, n) if m[r][i] != 0)
        m[i], m[pivot] = m[pivot], m[i]
        m[i] = [v / m[i][i] for v in m[i]]
        for r in range(n):
            if r != i:
                m[r] = [v - m[r][i] * w for v, w in zip(m[r], m[i])]
    return [row[n:] for row in m]


def gram(design, prior_diag):
    k = len(design[0])
    p = [[sum(x[i] * x[j] for x in design) for j in range(k)] for i in range(k)]
    for i, d in enumerate(prior_diag):
        p[i][i] += d
    return p


def gaussian():
    rows, ys = [], []
    for _ in range(64):
        for t in (0, 1):
            for noise in (F(-2, 5), F(2, 5)):
                rows.append([F(1), F(t)])
                ys.append(1 + 2 * t + noise)
    inv = inverse(gram(rows, [F(1, 100)] * 2))
    moment = [sum(x[i] * y for x, y in zip(rows, ys)) for i in range(2)]
    mean = [sum(inv[i][j] * moment[j] for j in range(2)) for i in range(2)]
    return {"mean": mean[1], "variance": inv[1][1]}


def quadratic():
    rows, ys = [], []
    for _ in range(20):
        for t in (0, 1):
            for z in (-1, 0, 1):
                for noise in (F(-1, 5), F(1, 5)):
                    rows.append([F(1), F(t), F(z), F(t * z), F(z * z)])
                    ys.append(1 + 2 * t + F(1, 2) * z + F(2, 5) * t * z + F(3, 10) * z * z + noise)
    inv = inverse(gram(rows, [F(1, 100)] * 5))
    moment = [sum(x[i] * y for x, y in zip(rows, ys)) for i in range(5)]
    mean = [sum(inv[i][j] * moment[j] for j in range(5)) for i in range(5)]
    rss = sum((y - sum(b * v for b, v in zip(mean, x))) ** 2 for x, y in zip(rows, ys))
    penalty = F(1, 100) * sum(b * b for b in mean)
    shape = F(1, 1000) + F(len(rows), 2)
    scale = F(1, 1000) + (rss + penalty) / 2
    return {
        "coefficient_mean": mean,
        "effect_variance": scale / (shape - 1) * inv[1][1],
        "precision_inverse": inv,
        "nig_shape": shape,
        "nig_scale": scale,
    }


def joint():
    # Two independent 128-row sources; design columns [T, 1]; prior diag [1/100, 1/4].
    rows, ys = [], []
    for _ in range(2):
        for r in range(128):
            t = r % 2
            rows.append([F(t), F(1)])
            ys.append(F(3) if t else F(1))
    inv = inverse(gram(rows, [F(1, 100), F(1, 4)]))
    moment = [sum(x[i] * y for x, y in zip(rows, ys)) for i in range(2)]
    mean = [sum(inv[i][j] * moment[j] for j in range(2)) for i in range(2)]
    return {"mean": mean[0], "variance": inv[0][0], "precision": gram(rows, [F(1, 100), F(1, 4)]), "moment": moment}


def close(name, fixture, reference, problems):
    if abs(float(reference) - fixture) > ATOL:
        problems.append(f"{name}: fixture {fixture} reference {float(reference)}")


def check(document: dict) -> list[str]:
    problems = []
    g, q, j = gaussian(), quadratic(), joint()
    close("gaussian.mean", document["gaussian"]["mean"], g["mean"], problems)
    close("gaussian.variance", document["gaussian"]["variance"], g["variance"], problems)
    fq = document["quadratic_basis"]
    for i, (a, b) in enumerate(zip(fq["coefficient_mean"], q["coefficient_mean"], strict=True)):
        close(f"quadratic.coefficient_mean[{i}]", a, b, problems)
    close("quadratic.effect_variance", fq["effect_variance"], q["effect_variance"], problems)
    close("quadratic.nig_shape", fq["nig_shape"], q["nig_shape"], problems)
    close("quadratic.nig_scale", fq["nig_scale"], q["nig_scale"], problems)
    for i in range(5):
        for k in range(5):
            close(f"quadratic.precision_inverse[{i}][{k}]", fq["precision_inverse"][i][k], q["precision_inverse"][i][k], problems)
    close("joint.mean", document["joint_internal"]["mean"], j["mean"], problems)
    close("joint.variance", document["joint_internal"]["variance"], j["variance"], problems)
    if j["precision"] != [[F(12801, 100), 128], [128, F(1025, 4)]] or j["moment"] != [384, 512]:
        problems.append(f"joint precision/moment {j['precision']} {j['moment']}")
    transfer = document["prior_transfer"]
    if not transfer["source_effect"] < transfer["target_effect"]:
        problems.append("transfer bracket is empty")
    return problems


def self_test(document: dict) -> list[str]:
    failures = []
    wrong = copy.deepcopy(document)
    wrong["gaussian"]["mean"] += 1e-6
    if not check(wrong):
        failures.append("shifted gaussian mean accepted")
    wrong = copy.deepcopy(document)
    wrong["quadratic_basis"]["effect_variance"] *= 1.001
    if not check(wrong):
        failures.append("shifted quadratic variance accepted")
    wrong = copy.deepcopy(document)
    wrong["joint_internal"]["variance"] = document["gaussian"]["variance"]
    if not check(wrong):
        failures.append("single-source variance accepted for the joint candidate")
    wrong = copy.deepcopy(document)
    wrong["quadratic_basis"]["precision_inverse"][0][1] = 0.0
    if not check(wrong):
        failures.append("altered precision inverse accepted")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    document = json.loads((HERE / "expected.json").read_text(encoding="utf-8"))
    problems = self_test(document) if args.self_test else check(document)
    for problem in problems:
        print("FAIL", problem)
    if not problems:
        print("ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
