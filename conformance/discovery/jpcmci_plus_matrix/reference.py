# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["tigramite==5.2.9.7", "numpy==2.1.3", "scipy==1.14.1", "joblib==1.4.2"]
# ///
"""Regenerate pooled-context and caller-fixed-regime graph reference matrices.

Use only released Tigramite public APIs and recorded numeric inputs. The fixed
regime comparison runs PCMCI+ independently on the supplied segments; it does
not infer regimes or claim RPCMCI's alternating optimization was reproduced.
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
from tigramite.jpcmciplus import JPCMCIplus
from tigramite.pcmci import PCMCI

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from reference_compare import values_close

PINS = {"tigramite": "5.2.9.7", "numpy": "2.1.3", "scipy": "1.14.1", "joblib": "1.4.2"}
GRIDS = {
    "jpcmci_plus_matrix": (
        "environments",
        "reference",
        "0ca51cac7346a9c532ac67eab8941788545868e712078e417404b8110517c78c",
    ),
    "rpcmci_fixed_regime_matrix": (
        "segments",
        "reference_by_regime",
        "c44920ecf450b15300f5cf3c50bf0781cf701b920456b3a352b8fad71583a5ff",
    ),
}


def input_digest(document: dict, field: str) -> str:
    inputs = [
        {key: case[key] for key in ("name", "var_names", field, "parameters")}
        for case in document["cases"]
    ]
    return hashlib.sha256(
        json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def normalize(result: dict, names: list[str]) -> dict:
    graph = result["graph"]
    links = []
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


def outputs(case: dict, grid: str) -> list[dict]:
    names = case["var_names"]
    parameters = dict(case["parameters"])
    if grid == "jpcmci_plus_matrix":
        if parameters.pop("include_space_dummy") is not False:
            raise ValueError("this frozen grid has no space-dummy comparison")
        data = pp.DataFrame(
            {
                i: np.asarray(env, dtype=float)
                for i, env in enumerate(case["environments"])
            },
            analysis_mode="multiple",
            var_names=names,
        )
        algorithm = JPCMCIplus(
            node_classification={i: "system" for i in range(len(names))},
            dataframe=data,
            cond_ind_test=ParCorr(significance="analytic"),
            verbosity=0,
        )
        return [normalize(algorithm.run_jpcmciplus(**parameters), names)]
    if parameters.pop("alternating_iters") != 0:
        raise ValueError(
            "fixed-label reference cannot establish alternating regime inference"
        )
    results = []
    for segment in case["segments"]:
        data = pp.DataFrame(np.asarray(segment, dtype=float), var_names=names)
        algorithm = PCMCI(
            dataframe=data, cond_ind_test=ParCorr(significance="analytic"), verbosity=0
        )
        results.append(normalize(algorithm.run_pcmciplus(**parameters), names))
    return results


def check(
    document: dict, grid: str, computed: list[list[dict]] | None = None
) -> list[str]:
    if any(
        importlib.metadata.version(name) != version for name, version in PINS.items()
    ):
        return ["wrong comparator environment"]
    input_field, reference_field, digest = GRIDS[grid]
    if input_digest(document, input_field) != digest:
        return ["frozen scenario grid, inputs or parameters changed"]
    if document.get("regeneration_tolerance") != {"atol": 1e-12, "rtol": 1e-10}:
        return ["regeneration tolerance changed"]
    if computed is None:
        computed = [outputs(case, grid) for case in document["cases"]]
    if not computed or len(computed) != len(document["cases"]):
        return ["missing reference discovery"]
    problems = []
    for case, actuals in zip(document["cases"], computed):
        expected = case[reference_field]
        expected = [expected] if reference_field == "reference" else expected
        if len(expected) != len(actuals):
            problems.append(f"{case['name']}: regime count differs")
            continue
        for regime, (frozen, actual) in enumerate(zip(expected, actuals)):
            if (
                "graph_shape" in frozen
                and frozen["graph_shape"] != actual["graph_shape"]
            ):
                problems.append(f"{case['name']}/{regime}: graph shape differs")
            key = lambda links: sorted(
                json.dumps(link, sort_keys=True) for link in links
            )
            if key(frozen["links"]) != key(actual["links"]):
                problems.append(
                    f"{case['name']}/{regime}: endpoint-marked graph differs"
                )
            for field in ("val_matrix", "p_matrix"):
                if not values_close(
                    frozen[field], actual[field], atol=1e-12, rtol=1e-10
                ):
                    problems.append(f"{case['name']}/{regime}: {field} differs")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    problems = []
    for grid, (_, reference_field, _) in GRIDS.items():
        document = json.loads(
            (ROOT / "conformance/discovery" / grid / "expected.json").read_text(
                encoding="utf-8"
            )
        )
        computed = [outputs(case, grid) for case in document["cases"]]
        problems.extend(check(document, grid, computed))
        if args.self_test:
            for mutation in ("output", "mark", "input", "tolerance", "empty"):
                changed = copy.deepcopy(document)
                frozen = changed["cases"][0][reference_field]
                frozen = frozen if reference_field == "reference" else frozen[0]
                if mutation == "output":
                    frozen["val_matrix"][0][0][1] += 0.1
                elif mutation == "mark":
                    frozen["links"][0]["mark"] = "<->"
                elif mutation == "input":
                    changed["cases"][0]["parameters"]["tau_max"] += 1
                elif mutation == "tolerance":
                    changed["regeneration_tolerance"]["atol"] = 1.0
                else:
                    changed["cases"] = []
                if not check(changed, grid, computed):
                    problems.append(f"{grid}: fault escaped: {mutation}")
            if not check(document, grid, []):
                problems.append(f"{grid}: fault escaped: missing discovery")
        print(f"{grid}: {len(document['cases'])} fixed scenarios checked", flush=True)
    print(
        "\n".join(problems)
        if problems
        else "Pooled-context and fixed-regime references and controls passed"
    )
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
