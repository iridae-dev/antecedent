# /// script
# requires-python = ">=3.10,<3.11"
# dependencies = ["tigramite==5.2.1.30", "numpy==1.23.5", "scipy==1.10.1"]
# ///
"""Regenerate fixed scalar PCMCI/PCMCI+ graph references through public APIs.

Inputs are frozen CSVs; no product, upstream implementation or test source is read.
This performs no calibration or error-rate acceptance campaign.
SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path

import numpy as np
from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI

HERE = Path(__file__).resolve().parent
SPECS = {
    "pcmci_lag1": {
        "coordinates": {
            "n": 500,
            "alpha": 0.05,
            "max_lag": 2,
            "fdr": False,
            "scm": "x_t ~ N(0,1); y_t = 0.8 * x_{t-1} + 0.2 * N(0,1); seed=0",
            "tolerance_class": "Exact",
        },
        "csv_sha256": "7f9eee556c62096679889389cc002c8f0e4c91aa4d3bfbd2d1cf43188bc3458b",
    },
    "masked_mci_lag1": {
        "coordinates": {
            "n": 500,
            "alpha": 0.05,
            "max_lag": 2,
            "fdr": False,
            "scm": "x_t ~ U(-1,1); y_t = 0.8 * x_{t-1} + 0.2*noise; seed=0 LCG",
            "tolerance_class": "Exact",
            "mask": "hide every 7th row (analysis_mask)",
        },
        "csv_sha256": "dc3f76d4b5cb1f8c994a596b6fda139492d270057c6bb8a39dc1c2e8b4b25a46",
    },
    "pcmci_plus_lag0": {
        "coordinates": {
            "n": 500,
            "alpha": 0.05,
            "max_lag": 1,
            "fdr": False,
            "scm": "x_t ~ N(0,1); y_t = 0.8*x_{t-1} + 0.5*x_t + 0.2*N(0,1); seed=0",
            "tolerance_class": "Exact",
            "min_lag": 0,
        },
        "csv_sha256": "2316b96062ff8477dd490812abd6e272f63609248511078aee3294b334d30173",
    },
}


def regenerate(name: str, csv: bytes) -> dict:
    spec = SPECS[name]["coordinates"]
    raw = np.genfromtxt(io.BytesIO(csv), delimiter=",", names=True)
    columns = [n for n in raw.dtype.names if n != "mask"]
    values = np.column_stack([raw[n] for n in columns])
    mask = None
    if "mask" in raw.dtype.names:
        # CSV mask=1 means retained; Tigramite mask=True means excluded.
        mask = np.repeat((raw["mask"] == 0)[:, None], len(columns), axis=1)
    engine = PCMCI(
        dataframe=pp.DataFrame(values, mask=mask, var_names=columns),
        cond_ind_test=ParCorr(
            significance="analytic", mask_type="xyz" if mask is not None else None
        ),
        verbosity=0,
    )
    options = {
        "tau_min": spec.get("min_lag", 1),
        "tau_max": spec["max_lag"],
        "pc_alpha": spec["alpha"],
        "alpha_level": spec["alpha"],
        "max_conds_dim": 2,
        "fdr_method": "none",
    }
    if name == "pcmci_plus_lag0":
        options.pop("alpha_level")  # This release uses pc_alpha for PCMCI+ tests.
        result = engine.run_pcmciplus(**options)
    else:
        result = engine.run_pcmci(**options)
    graph = result["graph"]
    parents, links = [], []
    for i, source in enumerate(columns):
        for j, target in enumerate(columns):
            for lag in range(graph.shape[2]):
                mark = str(graph[i, j, lag])
                if not mark:
                    continue
                link = {
                    "source": source,
                    "source_lag": lag,
                    "target": target,
                    "target_lag": 0,
                }
                links.append(dict(link, mark=mark))
                if lag > 0 and mark == "-->":
                    parents.append(link)
    outputs = {
        "graph_shape": list(graph.shape),
        "recovered_parents": parents,
        "alpha": spec["alpha"],
        "max_lag": spec["max_lag"],
    }
    if name == "pcmci_plus_lag0":
        outputs["graph_links"] = links
    return outputs


def check(name: str, document: dict, csv: bytes, actual: dict) -> list[str]:
    spec = SPECS[name]
    problems = []
    if hashlib.sha256(csv).hexdigest() != spec["csv_sha256"]:
        problems.append("CSV input changed")
    if any(document.get(k) != v for k, v in spec["coordinates"].items()):
        problems.append("scenario or exact tolerance changed")
    if document.get("oracle", {}).get("kind") != "external_package":
        problems.append("incorrect oracle origin")
    reference = document.get("reference", {})
    if (reference.get("available"), reference.get("version")) != (True, "5.2.1.30"):
        problems.append("missing required pinned comparator")
    frozen = reference.get("outputs", {})
    for field, value in actual.items():
        expected = frozen.get(field)
        if isinstance(value, list) and field in ("recovered_parents", "graph_links"):
            value = sorted(json.dumps(v, sort_keys=True) for v in value)
            expected = (
                sorted(json.dumps(v, sort_keys=True) for v in expected)
                if isinstance(expected, list)
                else None
            )
        if value != expected:
            problems.append(
                f"{field} differs: regenerated={value!r}, frozen={expected!r}"
            )
    if (
        document.get("true_parents") != frozen.get("recovered_parents")
        and name != "pcmci_plus_lag0"
    ):
        problems.append("recorded SCM parent target differs from comparator")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    for package, version in (
        ("tigramite", "5.2.1.30"),
        ("numpy", "1.23.5"),
        ("scipy", "1.10.1"),
    ):
        if importlib.metadata.version(package) != version:
            raise AssertionError(f"incorrect {package} version")
    problems = []
    for name in SPECS:
        directory = HERE.parent / name
        document = json.loads((directory / "expected.json").read_text(encoding="utf-8"))
        csv = (directory / "data.csv").read_bytes()
        if hashlib.sha256(csv).hexdigest() != SPECS[name]["csv_sha256"]:
            problems.append(f"{name}: corrupt input; refusing regeneration")
            continue
        actual = regenerate(name, csv)
        problems.extend(f"{name}: {p}" for p in check(name, document, csv, actual))
        if args.self_test:
            for field, value in (
                ("alpha", 0.99),
                ("n", 0),
                ("tolerance_class", "StableFloat"),
            ):
                wrong = copy.deepcopy(document)
                wrong[field] = value
                if not check(name, wrong, csv, actual):
                    problems.append(f"fault escaped: {name}.{field}")
            wrong = copy.deepcopy(document)
            wrong["reference"]["outputs"]["recovered_parents"] = []
            if not check(name, wrong, csv, actual):
                problems.append(f"fault escaped: {name} removed graph")
            wrong = copy.deepcopy(document)
            wrong["reference"]["available"] = False
            if not check(name, wrong, csv, actual):
                problems.append(f"fault escaped: {name} unavailable comparator")
            if name == "pcmci_plus_lag0":
                wrong = copy.deepcopy(document)
                wrong["reference"]["outputs"]["graph_links"][0]["mark"] = "o-o"
                if not check(name, wrong, csv, actual):
                    problems.append("fault escaped: altered endpoint mark")
            wrong = copy.deepcopy(document)
            wrong["oracle"]["kind"] = "closed_form"
            if not check(name, wrong, csv, actual):
                problems.append(f"fault escaped: {name} false origin")
            if not check(name, document, csv + b"corruption", actual):
                problems.append(f"fault escaped: {name} corrupted input")
    print(
        "\n".join(problems)
        if problems
        else "Three scalar Tigramite references and fault controls passed"
    )
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
