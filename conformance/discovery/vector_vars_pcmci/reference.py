# /// script
# requires-python = ">=3.10,<3.11"
# dependencies = ["tigramite==5.2.1.30", "numpy==1.23.5", "scipy==1.10.1"]
# ///
"""Retained black-box regeneration of the vector-variable PCMCI reference.

Reads only the fixture's declared series (``data.csv``), lag, level and FDR method, runs the
public ``tigramite`` PCMCI with partial-correlation tests, and reports the scalar logical
parents.  Neither Antecedent code nor tigramite source is read or stored.  The
``vector_vars_probe`` block of the frozen document is informational and is not regenerated.

    uv run --frozen --script conformance/discovery/vector_vars_pcmci/reference.py --check
    uv run --frozen --script conformance/discovery/vector_vars_pcmci/reference.py --self-test

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np
from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI

HERE = Path(__file__).resolve().parent
NAMES = ["x0", "x1", "y"]


def series() -> np.ndarray:
    with open(HERE / "data.csv", newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == NAMES
    return np.array([[float(v) for v in row] for row in rows[1:]])


def key(edge: dict) -> tuple:
    return (edge["target"], edge["source"], edge["source_lag"])


def recovered_parents(document: dict) -> list[dict]:
    data = series()
    assert data.shape[0] == document["n"]
    frame = pp.DataFrame(data, var_names=NAMES)
    pcmci = PCMCI(dataframe=frame, cond_ind_test=ParCorr(), verbosity=0)
    results = pcmci.run_pcmci(
        tau_min=1, tau_max=document["max_lag"], pc_alpha=None
    )
    q = pcmci.get_corrected_pvalues(
        p_matrix=results["p_matrix"],
        fdr_method=document["reference"]["outputs"]["fdr_method"],
        tau_min=1,
        tau_max=document["max_lag"],
    )
    found = []
    for source, target, lag in zip(*np.where(q < document["alpha"])):
        if lag == 0:
            continue
        found.append(
            {
                "source": NAMES[source],
                "source_lag": int(lag),
                "target": NAMES[target],
                "target_lag": 0,
            }
        )
    return sorted(found, key=key)


def check(document: dict) -> list[str]:
    problems = []
    if document["tolerance_class"] != "Exact":
        problems.append("comparison class changed from Exact")
    if document["reference"]["version"] != importlib.metadata.version("tigramite"):
        problems.append("reference version does not match the locked tigramite")
    expected = sorted(document["reference"]["outputs"]["recovered_parents"], key=key)
    if not expected:
        problems.append("empty recovered-parent set")
    actual = recovered_parents(document)
    if actual != expected:
        problems.append(f"recovered parents differ: {actual} vs {expected}")
    if expected != sorted(document["true_parents"], key=key):
        problems.append("reference parents differ from the generating SCM's parents")
    return problems


def self_test(document: dict) -> list[str]:
    failures = []
    wrong = copy.deepcopy(document)
    wrong["reference"]["outputs"]["recovered_parents"][0]["source_lag"] = 2
    if not check(wrong):
        failures.append("wrong lag accepted")
    wrong = copy.deepcopy(document)
    wrong["reference"]["outputs"]["recovered_parents"] = []
    if not check(wrong):
        failures.append("empty parent set accepted")
    wrong = copy.deepcopy(document)
    wrong["alpha"] = 0.9
    if not check(wrong):
        failures.append("changed level accepted")
    wrong = copy.deepcopy(document)
    wrong["reference"]["outputs"]["recovered_parents"][0]["source"] = "x1"
    if not check(wrong):
        failures.append("wrong source accepted")
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
