# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.1.3", "scipy==1.14.1"]
# ///
"""Independent Beta quadrature for a retained finite-model posterior functional.

Integrates the random conditional probability, rather than conditioning a table
of posterior means. No product sampler, random seed, graph engine or inference
implementation is used. Public Gauss-Jacobi nodes integrate independent Beta
coordinates; analytic Beta moments integrate the outcome tables exactly.
SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.special import roots_jacobi

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from reference_compare import values_close

PATH = Path(__file__).with_name("expected.json")
PARAMETERS = {"z": [101, 101], "s_given_z0": [41, 161], "s_given_z1": [161, 41],
              "y_given_s1_z0": [16, 36], "y_given_s1_z1": [46, 6]}


def beta_moments(parameters):
    a, b = parameters
    return a / (a + b), a * (a + 1) / ((a + b) * (a + b + 1))


def beta_quadrature(parameters, nodes):
    a, b = parameters
    x, weights = roots_jacobi(nodes, b - 1, a - 1)
    return (x + 1) / 2, weights / weights.sum()


def summarize(first, second):
    return {"mean": float(first), "sd": math.sqrt(float(second - first * first))}


def outputs(nodes):
    z, wz = beta_quadrature(PARAMETERS["z"], nodes)
    s0, ws0 = beta_quadrature(PARAMETERS["s_given_z0"], nodes)
    s1, ws1 = beta_quadrature(PARAMETERS["s_given_z1"], nodes)
    selected_one = z[:, None, None] * s1[None, None, :]
    selected_zero = (1 - z[:, None, None]) * s0[None, :, None]
    posterior_z = selected_one / (selected_one + selected_zero)
    weights = wz[:, None, None] * ws0[None, :, None] * ws1[None, None, :]
    m0, q0 = beta_moments(PARAMETERS["y_given_s1_z0"])
    m1, q1 = beta_moments(PARAMETERS["y_given_s1_z1"])
    def mixture(first, second):
        mean = (1 - first) * m0 + first * m1
        second_moment = (1 - 2 * first + second) * q0 + second * q1 + 2 * (first - second) * m0 * m1
        return summarize(mean, second_moment)
    observed = mixture(float(np.sum(weights * posterior_z)), float(np.sum(weights * posterior_z**2)))
    interventional = mixture(*beta_moments(PARAMETERS["z"]))
    return {"condition_s1": observed, "intervene_s1": interventional,
            "intervene_s1_condition_z1": summarize(m1, q1)}


def build():
    low, high = outputs(32), outputs(64)
    if not values_close(low, high, rtol=0, atol=1e-12):
        raise ValueError("quadrature has not converged within the declared bound")
    return {"schema_version": 1, "posterior_beta_parameters": PARAMETERS,
            "outputs": high, "reference_atol": 1e-12,
            "product_check": {"conditioning_draws": 8000, "conditioning_seed": 17,
                              "mean_standard_errors": 5.0, "relative_sd_tolerance": 0.05},
            "oracle": {"kind": "independent_reimplementation",
                       "generator": "conformance/composition/dirichlet_conditioning/reference.py",
                       "command": "uv run --frozen --script conformance/composition/dirichlet_conditioning/reference.py --check",
                       "method": "32/64-node tensor Beta quadrature plus analytic outcome-table first and second moments",
                       "limitations": "Fixed-table posterior functional moments only; finite quadrature agreement is a numerical convergence check, not a formal error certificate. Does not establish posterior interval coverage or sampling calibration."}}


def problems(document):
    expected = build()
    actual = copy.deepcopy(document)
    actual.get("oracle", {}).pop("reference_review", None)
    return [] if values_close(actual, expected, rtol=0, atol=1e-12) else ["posterior functional fixture differs"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    for package, version in {"numpy": "2.1.3", "scipy": "1.14.1"}.items():
        if importlib.metadata.version(package) != version:
            raise SystemExit(f"{package} must be {version}")
    if args.write:
        PATH.write_text(json.dumps(build(), indent=2) + "\n", encoding="utf-8")
    document = json.loads(PATH.read_text(encoding="utf-8"))
    if problems(document):
        raise SystemExit("\n".join(problems(document)))
    if args.self_test:
        for field, value in (("mean", 0.9), ("sd", 0.0)):
            changed = copy.deepcopy(document)
            changed["outputs"]["condition_s1"][field] = value
            if not problems(changed):
                raise SystemExit("incorrect posterior output passed")
        changed = copy.deepcopy(document)
        changed["reference_atol"] = 1.0
        if not problems(changed):
            raise SystemExit("altered reference tolerance passed")
    print(json.dumps(document["outputs"], indent=2))


if __name__ == "__main__":
    main()
