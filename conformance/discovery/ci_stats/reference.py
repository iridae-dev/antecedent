# /// script
# requires-python = ">=3.10,<3.11"
# dependencies = ["tigramite==5.2.1.30", "numpy==1.23.5", "scipy==1.10.1", "dcor", "scikit-learn"]
# ///
"""Retained black-box regeneration of the conditional-independence statistic references.

Reads only ``data.csv`` and runs the public ``tigramite`` tests GPDC, CMIknn (k = 5) and
Gsquared on the fixture's declared columns:

* ``gpdc_*``: x vs y given z (continuous);
* ``cmiknn_*``: x vs y given z, knn = 5, shuffle test (statistic exact, p-value Monte Carlo);
* ``gsquared_*``: xd vs yd with no conditioning set (discrete).

The frozen ``robust_parcorr_*`` entries are NOT regenerated: the call that produced them could
not be reconstructed (the public RobustParCorr calls tried give different values), and the
three ``*_error`` entries record failed probes.  Those entries are context, not evidence, and
the consuming test does not assert them.

    uv run --frozen --script conformance/discovery/ci_stats/reference.py --check
    uv run --frozen --script conformance/discovery/ci_stats/reference.py --self-test

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np
from tigramite.independence_tests.cmiknn import CMIknn
from tigramite.independence_tests.gpdc import GPDC
from tigramite.independence_tests.gsquared import Gsquared

HERE = Path(__file__).resolve().parent
STAT_RTOL = 1e-9
P_ATOL_MONTE_CARLO = 0.1


def columns() -> dict[str, np.ndarray]:
    with open(HERE / "data.csv", newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    names = rows[0]
    return {
        n: np.array([float(r[i]) for r in rows[1:]]).reshape(-1, 1)
        for i, n in enumerate(names)
    }


def outputs() -> dict:
    c = columns()
    out = {}
    for tag, y in (("dep", "y_dep"), ("ind", "y_ind")):
        s, p = GPDC().run_test_raw(c["x"], c[y], c["z"])
        out[f"gpdc_{tag}"] = {"statistic": float(s), "p_value": float(p), "mc": False}
        s, p = CMIknn(
            significance="shuffle_test", knn=5, shuffle_neighbors=5, sig_samples=1000, seed=1
        ).run_test_raw(c["x"], c[y], c["z"])
        out[f"cmiknn_{tag}"] = {"statistic": float(s), "p_value": float(p), "mc": True}
    for tag, y in (("dep", "yd_dep"), ("ind", "yd_ind")):
        s, p = Gsquared(significance="analytic").run_test_raw(c["xd"], c[y])
        out[f"gsquared_{tag}"] = {"statistic": float(s), "p_value": float(p), "mc": False}
    return out


def close(a: float, b: float, rtol: float, atol: float = 0.0) -> bool:
    return abs(a - b) <= atol + rtol * abs(b)


def check(document: dict) -> list[str]:
    problems = []
    if (document["atol_stat"], document["rtol_stat"], document["gpdc_atol_stat"]) != (0.15, 0.35, 0.5):
        problems.append("product comparison bands changed from the reviewed values")
    methods = document["reference"]["outputs"]["methods"]
    actual = outputs()
    for name, got in actual.items():
        frozen = methods.get(name)
        if frozen is None:
            problems.append(f"{name} missing from the frozen document")
            continue
        if not close(frozen["statistic"], got["statistic"], STAT_RTOL, 1e-12):
            problems.append(f"{name} statistic differs: {frozen['statistic']} vs {got['statistic']}")
        if got["mc"]:
            if abs(frozen["p_value"] - got["p_value"]) > P_ATOL_MONTE_CARLO:
                problems.append(f"{name} p-value differs beyond Monte Carlo error")
        elif not close(frozen["p_value"], got["p_value"], STAT_RTOL, 1e-12):
            problems.append(f"{name} p-value differs")
    if not actual:
        problems.append("empty selection")
    return problems


def self_test(document: dict) -> list[str]:
    failures = []
    for field, delta in (("statistic", 0.01), ("p_value", 0.3)):
        wrong = copy.deepcopy(document)
        wrong["reference"]["outputs"]["methods"]["gsquared_dep"][field] += delta
        if not check(wrong):
            failures.append(f"shifted gsquared {field} accepted")
    wrong = copy.deepcopy(document)
    wrong["reference"]["outputs"]["methods"]["gpdc_ind"]["statistic"] *= 1.5
    if not check(wrong):
        failures.append("shifted gpdc statistic accepted")
    wrong = copy.deepcopy(document)
    wrong["gpdc_atol_stat"] = 5.0
    if not check(wrong):
        failures.append("widened product band accepted")
    wrong = copy.deepcopy(document)
    del wrong["reference"]["outputs"]["methods"]["cmiknn_dep"]
    if not check(wrong):
        failures.append("dropped method accepted")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    document = json.loads((HERE / "expected.json").read_text(encoding="utf-8"))
    problems = self_test(document) if args.self_test else check(document)
    for p in problems:
        print("FAIL", p)
    if not problems:
        print("ok")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
