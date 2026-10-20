# SPDX-License-Identifier: MIT OR Apache-2.0
"""Invertibly standardize training coordinates for remaining OU optimizer failures."""

import argparse
import json
import runpy
import time
from pathlib import Path

import numpy as np
from antecedent.cyclic import PreparedCyclicFamily

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--base-dir", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()
module = runpy.run_path(
    str(
        Path(__file__).resolve().parents[3]
        / "examples/python/cyclic_paper_experiments.py"
    )
)
base = args.base_dir.resolve()
out = args.output_dir.resolve()
out.mkdir(exist_ok=True, parents=True)
reports = []
for case in json.loads((base / "results.json").read_text())["cases"]:
    if case["converged"]:
        continue
    name = f"seed-{case['seed']}-{case['case']}"
    source = base / name
    request = json.loads((source / "request.json").read_text())
    old = json.loads((source / "native_report.json").read_text())["answer"]["result"][
        "fit"
    ]
    request["model"] = old["model"]
    data = np.array(request["regimes"][0]["values"])
    center = data.mean(axis=0)
    scale = data.std(axis=0)
    n = len(center)
    assert np.all(scale > 0)

    def transform(model, n=n, scale=scale, center=center):
        m = np.array(model["drift"]).reshape(n, n)
        b = np.array(model["forcing"])
        l = np.array(model["diffusion_factor"]).reshape(n, n)
        return dict(
            model,
            drift=(m * scale[None, :] / scale[:, None]).ravel().tolist(),
            forcing=((m @ center + b) / scale).tolist(),
            diffusion_factor=(l / scale[:, None]).ravel().tolist(),
        )

    request["model"] = transform(request["model"])

    def replacements(items, old=old, scale=scale, center=center):
        result = []
        for item in items:
            i = old["model"]["equations"].index(item["equation"])
            row = np.array(item["drift"])
            result.append(
                dict(
                    item,
                    drift=(row * scale / scale[i]).tolist(),
                    forcing=float((row @ center + item["forcing"]) / scale[i]),
                    diffusion_factor=(
                        np.array(item["diffusion_factor"]) / scale[i]
                    ).tolist(),
                )
            )
        return result

    for regime in request["regimes"]:
        regime["values"] = ((np.array(regime["values"]) - center) / scale).tolist()
        regime["replacements"] = replacements(regime["replacements"])
    request["worlds"] = [replacements(items) for items in request["worlds"]]
    bound = max(
        float(((np.abs(center)[None, :] + scale[None, :]) / scale[:, None]).max()),
        float((1 / scale).max()),
    )
    original_tolerance = request["policy"]["score_tolerance"]
    request["policy"]["score_tolerance"] = original_tolerance / bound
    request["policy"]["max_iterations"] = 20000
    start = time.monotonic()
    prepared = PreparedCyclicFamily(json.dumps(request, allow_nan=False))
    result = prepared.execute()
    fit = result.report["answer"]["result"]["fit"]
    target = out / name
    target.mkdir(exist_ok=True)
    module["write"](target / "request.json", request)
    module["write"](target / "native_report.json", result.report)
    report = {
        "seed": case["seed"],
        "case": case["case"],
        "approach": "training_coordinate_standardization_then_LBFGS_same_original_score_bound",
        "previous_identity": case["artifact"]["identity"],
        "center": center.tolist(),
        "scale": scale.tolist(),
        "original_coordinate_map": "X=center+diag(scale)*Z; all drift, forcing, diffusion and intervention rows transformed exactly",
        "original_score_upper_bound": fit["score_norm"] * bound,
        "original_score_tolerance": original_tolerance,
        "converged": fit["converged"],
        "iterations": fit["iterations"],
        "line_search_exhausted": fit["line_search_exhausted"],
        "failed_worlds": sum(
            "failure" in law for law in result.report["answer"]["result"]["laws"]
        ),
        "artifact": module["replay"](result, target),
        "wall_seconds": time.monotonic() - start,
    }
    module["write"](target / "metrics.json", report)
    reports.append(report)
    print(json.dumps(report), flush=True)
module["write"](
    out / "results.json",
    {
        "approach": "invertible affine training-coordinate reparameterization, retaining the full original parameter family and a conservative original-coordinate score tolerance; no heldout-based fit selection",
        "cases": reports,
    },
)
