# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["dowhy==0.14"]
# ///
"""Regenerate the frozen linear ATE through DoWhy's public black-box API.

Read the retained CSV and declared graph; import no product or test code.
This fixed-input comparison runs no calibration, bootstrap or consumer replay.
SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

import networkx as nx
import pandas as pd
from dowhy import CausalModel

HERE = Path(__file__).resolve().parent
INPUT_SHA256 = "c2efe7c53819ca742952fbecb61d1b2a100c8663e75d41c9979158ae46c5c1bb"


def check(document: dict, csv: bytes) -> list[str]:
    problems = []
    if importlib.metadata.version("dowhy") != "0.14":
        return ["wrong DoWhy version"]
    if hashlib.sha256(csv).hexdigest() != INPUT_SHA256:
        return ["frozen CSV input changed"]
    if (document.get("atol"), document.get("rtol")) != (1e-10, 1e-8):
        problems.append("declared tolerance changed")
    if (
        document.get("n"),
        document.get("treatment"),
        document.get("outcome"),
        document.get("adjustment_set"),
        document.get("edges"),
    ) != (200, "t", "y", ["z"], [["z", "t"], ["z", "y"], ["t", "y"]]):
        return problems + ["declared scenario changed"]
    import io

    data = pd.read_csv(io.BytesIO(csv))
    graph = nx.DiGraph(document["edges"])
    model = CausalModel(data=data, treatment="t", outcome="y", graph=graph)
    identified = model.identify_effect(proceed_when_unidentifiable=False)
    estimated = model.estimate_effect(
        identified,
        method_name="backdoor.linear_regression",
        control_value=0,
        treatment_value=1,
        target_units="ate",
        test_significance=False,
        confidence_intervals=False,
    )
    value = float(estimated.value)
    reference = document.get("reference", {}).get("outputs", {})
    if reference.get("method") != "backdoor.linear_regression":
        problems.append("reference method changed")
    for label, expected in (
        ("reference_ate", document.get("reference_ate")),
        ("reference.outputs.estimate", reference.get("estimate")),
        ("true_ate", document.get("true_ate")),
    ):
        if not isinstance(expected, (int, float)) or not math.isclose(
            value, expected, abs_tol=1e-10, rel_tol=1e-8
        ):
            problems.append(f"{label} disagrees with DoWhy: {value}")
    # Independent structural truth: the recorded noiseless law is 1+2T+3Z.
    if document.get("true_ate") != 2.0:
        problems.append("structural coefficient truth changed")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    document = json.loads((HERE / "expected.json").read_text(encoding="utf-8"))
    csv = (HERE / "data.csv").read_bytes()
    problems = check(document, csv)
    if args.self_test:
        for field, wrong in (
            ("reference_ate", 3.0),
            ("true_ate", 3.0),
            ("atol", 1.0),
            ("n", 0),
            ("adjustment_set", []),
        ):
            changed = copy.deepcopy(document)
            changed[field] = wrong
            if not check(changed, csv):
                problems.append(f"fault escaped: {field}")
        changed = copy.deepcopy(document)
        changed["reference"]["outputs"]["estimate"] = 3.0
        if not check(changed, csv):
            problems.append("fault escaped: reference output")
        if not check(document, csv + b"corruption"):
            problems.append("fault escaped: corrupt input")
    print(
        "\n".join(problems)
        if problems
        else "DoWhy linear ATE reference and controls passed"
    )
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
