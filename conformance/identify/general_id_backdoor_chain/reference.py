# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["dowhy==0.14"]
# ///
"""Reproduce frozen positive and negative identification results through public DoWhy APIs.

No product implementation, numeric estimation or calibration is imported/run.
SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations
import argparse
import copy
import importlib.metadata
import json
from pathlib import Path
import pandas as pd
from dowhy import CausalModel

HERE = Path(__file__).resolve().parent
SCENARIOS = {
    "general_id_hedge": (
        "nonidentifiable_or_unidentified",
        "digraph { t -> y; U[latent]; U -> t; U -> y; }",
        ["t", "y"],
        None,
    ),
    "general_id_backdoor_chain": (
        "identifiable_backdoor",
        "digraph { z -> t; z -> y; t -> y; }",
        ["t", "y", "z"],
        "backdoor",
    ),
    "general_id_frontdoor": (
        "identifiable_frontdoor",
        "digraph { t -> m; m -> y; U[latent]; U -> t; U -> y; }",
        ["m", "t", "y"],
        "frontdoor",
    ),
}


def regenerate(name: str) -> dict:
    _, graph, observed, method = SCENARIOS[name]
    # Values are irrelevant to graph identification. U is deliberately absent:
    # it is unobserved, rather than an adjustment covariate.
    model = CausalModel(
        data=pd.DataFrame({n: [0.0, 1.0] for n in observed}),
        treatment="t",
        outcome="y",
        graph=graph,
    )
    result = model.identify_effect(proceed_when_unidentifiable=False)
    if method is None and any(result.estimands.values()):
        raise AssertionError(
            "DoWhy unexpectedly found a standard estimand on the bow arc"
        )
    if method is not None and not result.estimands.get(method):
        raise AssertionError(f"DoWhy did not identify {name} through {method}")
    return {"estimand": str(result), "estimand_type": str(result.estimand_type)}


def check(name: str, document: dict, actual: dict) -> list[str]:
    case, graph, _, method = SCENARIOS[name]
    problems = []
    if (
        document.get("case"),
        document.get("graph_dot"),
        document.get("treatment"),
        document.get("outcome"),
        document.get("expected_status_family"),
        document.get("tolerance_class"),
    ) != (case, graph, "t", "y", "identified" if method else "unidentified", "Exact"):
        problems.append("scenario or exact comparison changed")
    outputs = document.get("reference", {}).get("outputs", {})
    if outputs != actual:
        problems.append("frozen DoWhy output differs from regenerated result")
    text = outputs.get("estimand", "")
    if (
        method is not None
        and f"Estimand name: {method}\nEstimand expression:" not in text
    ):
        problems.append("required method has no executing estimand expression")
    if method is None and (
        "Estimand expression:" in text or text.count("No such variable(s) found!") != 4
    ):
        problems.append("negative fixture must refuse all four standard strategies")
    if (
        document.get("reference", {}).get("available"),
        document.get("reference", {}).get("version"),
    ) != (True, "0.14"):
        problems.append("required pinned comparator missing")
    if document.get("oracle", {}).get("kind") != "external_package":
        problems.append("incorrect oracle origin")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if importlib.metadata.version("dowhy") != "0.14":
        raise AssertionError("incorrect DoWhy version")
    problems = []
    for name in SCENARIOS:
        document = json.loads((HERE.parent / name / "expected.json").read_text())
        actual = regenerate(name)
        problems.extend(f"{name}: {p}" for p in check(name, document, actual))
        if args.self_test:
            for field, value in (
                ("graph_dot", "digraph { t -> y; }"),
                ("expected_status_family", "wrong_status"),
                ("tolerance_class", "StableFloat"),
            ):
                changed = copy.deepcopy(document)
                changed[field] = value
                if not check(name, changed, actual):
                    problems.append(f"fault escaped: {name}.{field}")
            for output in (
                {},
                {
                    "estimand": "Estimand name: frontdoor\nNo such variable(s) found!",
                    "estimand_type": actual["estimand_type"],
                },
            ):
                changed = copy.deepcopy(document)
                changed["reference"]["outputs"] = output
                if not check(name, changed, actual):
                    problems.append(f"fault escaped: {name} missing estimand")
            changed = copy.deepcopy(document)
            changed["oracle"]["kind"] = "closed_form"
            if not check(name, changed, actual):
                problems.append(f"fault escaped: {name} false origin")
    print(
        "\n".join(problems)
        if problems
        else "Three DoWhy identification references and fault controls passed"
    )
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
