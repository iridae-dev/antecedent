# /// script
# requires-python = ">=3.10,<3.11"
# dependencies = ["tigramite==5.2.1.30", "numpy==1.23.5", "scipy==1.10.1"]
# ///
"""Reproduce the retained LPCMCI motif inputs through Tigramite's public API.

No product or upstream implementation/test source is imported or translated.
This is fixed-input reference generation, not a discovery acceptance campaign.
SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np
from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.lpcmci import LPCMCI

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from reference_compare import values_close

PINS = {"tigramite": "5.2.1.30", "numpy": "1.23.5", "scipy": "1.10.1"}
INPUT_SHA256 = "5cca650c565d9dd45cf6375d73563169940f0e880858d84b1d1c38e6993d757d"


def input_digest(document: dict) -> str:
    inputs = [
        {key: case[key] for key in ("name", "data", "var_names", "parameters")}
        for case in document["cases"]
    ]
    return hashlib.sha256(
        json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def outputs(case: dict) -> dict:
    names = case["var_names"]
    data = pp.DataFrame(np.asarray(case["data"], dtype=float), var_names=names)
    result = LPCMCI(
        dataframe=data, cond_ind_test=ParCorr(significance="analytic"), verbosity=0
    ).run_lpcmci(**case["parameters"])
    graph = result["graph"]
    links = []
    # Public graph[i,j,tau] indexes source i@-tau -> target j@0.
    # Preserve both endpoint-relative entries at lag zero, including marks.
    for target in range(len(names)):
        for source in range(len(names)):
            for lag in range(graph.shape[2]):
                mark = str(graph[source, target, lag])
                if mark:
                    links.append(
                        {
                            "source": names[source],
                            "source_lag": lag,
                            "target": names[target],
                            "target_lag": 0,
                            "mark": mark,
                        }
                    )
    return {
        "graph_shape": list(graph.shape),
        "links": links,
        "val_matrix": result["val_matrix"].tolist(),
        "p_matrix": result["p_matrix"].tolist(),
    }


def canonical_links(links: list[dict]) -> list[str]:
    return sorted(json.dumps(link, sort_keys=True) for link in links)


def check(document: dict, computed: list[dict] | None = None) -> list[str]:
    if any(
        importlib.metadata.version(name) != version for name, version in PINS.items()
    ):
        return ["wrong comparator environment"]
    if input_digest(document) != INPUT_SHA256:
        return ["frozen scenario grid, inputs or parameters changed"]
    if document.get("regeneration_tolerance") != {"atol": 1e-12, "rtol": 1e-10}:
        return ["regeneration tolerance changed"]
    if computed is None:
        computed = [outputs(case) for case in document["cases"]]
    if len(computed) != len(document["cases"]) or not computed:
        return ["missing reference discovery"]
    problems = []
    for case, actual in zip(document["cases"], computed):
        expected = case["reference"]
        if expected.get("graph_shape") != actual["graph_shape"]:
            problems.append(f"{case['name']}: graph shape differs")
        if canonical_links(expected.get("links", [])) != canonical_links(
            actual["links"]
        ):
            problems.append(f"{case['name']}: endpoint-marked graph differs")
        for field in ("val_matrix", "p_matrix"):
            if not values_close(
                expected.get(field), actual[field], atol=1e-12, rtol=1e-10
            ):
                problems.append(f"{case['name']}: {field} differs")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    document = json.loads(
        Path(__file__).with_name("expected.json").read_text(encoding="utf-8")
    )
    computed = [outputs(case) for case in document["cases"]]
    problems = check(document, computed)
    if args.self_test:
        for mutation in ("output", "mark", "input", "tolerance", "empty"):
            changed = copy.deepcopy(document)
            if mutation == "output":
                changed["cases"][0]["reference"]["val_matrix"][0][0][1] += 0.1
            elif mutation == "mark":
                changed["cases"][0]["reference"]["links"][0]["mark"] = "<->"
            elif mutation == "input":
                changed["cases"][0]["data"][0][0] += 0.1
            elif mutation == "tolerance":
                changed["regeneration_tolerance"]["atol"] = 1.0
            else:
                changed["cases"] = []
            if not check(changed, computed):
                problems.append(f"fault escaped: {mutation}")
        if not check(document, []):
            problems.append("fault escaped: missing reference discovery")
    print(
        "\n".join(problems)
        if problems
        else "LPCMCI six-scenario reference and controls passed"
    )
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
