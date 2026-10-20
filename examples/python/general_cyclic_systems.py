# SPDX-License-Identifier: MIT OR Apache-2.0
"""3.0 development workflow: fit overlapping nonlinear loops and replay an artifact.

The reference generator below is independent of Antecedent's solver. Training
contains observational and single-target intervention regimes; the queried joint
regime is held out. Covariance is specified, including a shared disturbance term.
Run with --output-dir to keep the artifact, request and separately retained identity.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

from antecedent.cyclic import Equation, EquilibriumModel


def run(output_dir: Path) -> None:
    random_source = random.Random(826)
    covariance = (0.0009, 0.00027, 0.0, 0.00027, 0.0009, 0.0, 0.0, 0.0, 0.0009)
    template = EquilibriumModel(
        (
            Equation("a", 0.0, (0.0, 0.1, 0.1)),
            Equation("b", 0.0, (0.1, 0.0, 0.0)),
            Equation("c", 0.0, (-0.1, 0.0, 0.0)),
        ),
        covariance,
    )
    rows = []
    for i in range(160):
        z = [random_source.gauss(0.0, 1.0) for _ in range(3)]
        disturbance = [
            0.03 * z[0],
            0.03 * (0.3 * z[0] + math.sqrt(0.91) * z[1]),
            0.03 * z[2],
        ]
        clamps = (
            []
            if i % 4 == 0
            else [{"variable": i % 4 - 1, "value": 1.0 if i % 8 < 4 else -1.0}]
        )
        imposed = {clamp["variable"]: clamp["value"] for clamp in clamps}
        a, b, c = 0.0, 0.0, 0.0
        for _ in range(100):
            # Independent reference generator, no native solve/fitting calls.
            a, b, c = (
                imposed.get(0, math.tanh(0.3 * b + 0.2 * c) + disturbance[0]),
                imposed.get(1, math.tanh(0.25 * a) + disturbance[1]),
                imposed.get(2, math.tanh(-0.2 * a) + disturbance[2]),
            )
        rows.append({"values": [a, b, c], "clamps": clamps})
    joint = [{"variable": 1, "value": 0.8}, {"variable": 2, "value": -0.6}]
    prepared = template.prepare(
        [[], joint],
        [[0.0, 0.0, 0.0]],
        factual=[0.2, 0.3, -0.1],
        parameters=[
            {"kind": "weight", "equation": i, "parent": j}
            for i, j in [(0, 1), (0, 2), (1, 0), (2, 0)]
        ],
        rows=rows,
    )
    result = prepared.execute()
    assert result.report["fit"]["converged"]
    prediction = result.report["worlds"][1][0]["values"][0]
    counterfactual = result.report["counterfactuals"][1]["values"][0]
    assert abs(prediction - math.tanh(0.12)) < 0.035
    assert abs(counterfactual - (math.tanh(0.12) + 0.2 - math.tanh(0.07))) < 0.035
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = output_dir / "nonlinear.ant"
    artifact.write_bytes(result.artifact)
    (output_dir / "expected-identity.json").write_text(
        json.dumps({"identity": result.identity}) + "\n"
    )
    (output_dir / "request.json").write_text(prepared.request_json + "\n")
    (output_dir / "report.json").write_text(json.dumps(result.report, indent=2) + "\n")
    consumer = (
        "import json,pathlib;from antecedent.cyclic import consume_artifact;"
        + f"r=consume_artifact(pathlib.Path({str(artifact.resolve())!r}).read_bytes(),expected_identity={result.identity!r});print(json.dumps(r.report,sort_keys=True))"
    )
    replay = subprocess.run(
        [sys.executable, "-c", consumer],
        cwd=output_dir,
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(replay.stdout) == result.report
    print(
        json.dumps(
            {
                "identity": result.identity,
                "joint_intervention_prediction": prediction,
                "counterfactual_prediction": counterfactual,
                "claim": result.report["claim"],
                "calibration": result.report["calibration"],
                "fresh_process_replay": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    arguments = parser.parse_args()
    if arguments.output_dir is None:
        with tempfile.TemporaryDirectory(prefix="antecedent-cyclic-example-") as folder:
            run(Path(folder))
    else:
        run(arguments.output_dir.resolve())
