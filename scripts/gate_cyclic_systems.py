#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Standalone cyclic implementation gate; calibration/attestation are separate.

Run with an installed current release wheel:
  python3 scripts/gate_cyclic_systems.py --python /path/to/venv/bin/python
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]
RUST_CHECKS = [
    ("antecedent-model", None),
    ("antecedent-graph", "sigma"),
    ("antecedent-graph", "bipartite"),
    ("antecedent-identify", "cyclic"),
    ("antecedent-discovery", "cyclic_fci"),
    ("antecedent-estimate", "cyclic"),
    ("antecedent-estimate", "polynomial_scm"),
    ("antecedent-estimate", "constraint_likelihood"),
    ("antecedent-estimate", "ou_mechanisms"),
    ("antecedent-prob", "metropolis"),
    ("antecedent-io", "cyclic"),
    ("antecedent", "cyclic"),
]


def run(command: list[str], **kwargs) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True, **kwargs)


def inventory() -> None:
    record = tomllib.loads((ROOT / "parity/cyclic/acceptance.toml").read_text())
    expected = {"Y18.1", "Y18.2"} | {f"Y19.{i}" for i in range(1, 9)}
    if {row["id"] for row in record["capability"]} != expected:
        raise ValueError("cyclic acceptance children differ from promoted scope")
    for row in record["capability"]:
        for path in row["rust"]:
            if not (ROOT / path).is_file():
                raise ValueError(f"missing cyclic implementation {path}")
    fixture = ROOT / "parity/cyclic/general-equilibrium"
    manifest = json.loads((fixture / "manifest.json").read_text())
    for key, path in [
        ("artifact_sha256", fixture / "nonlinear.ant"),
        ("generator_sha256", ROOT / manifest["generator"]),
    ]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest[key]:
            raise ValueError(f"stale independent fixture {path}")
    source = hashlib.sha256(
        (ROOT / "examples/python/cyclic_paper_experiments.py").read_bytes()
    ).hexdigest()
    for lane in ("nodags-linear", "nodags-relu", "bicycle"):
        path = ROOT / f"parity/cyclic/paper-experiments/{lane}.json"
        result = json.loads(path.read_text())
        if (
            result["manifest"]["script_sha256"] != source
            or not result["manifest"]["full"]
        ):
            raise ValueError(f"stale or reduced paper protocol: {lane}")
        if {trial["seed"] for trial in result["trials"]} != set(range(5)):
            raise ValueError(f"incomplete independent trials: {lane}")
        for trial in result["trials"]:
            cases = trial["cases"] if lane == "bicycle" else [trial]
            for case in cases:
                artifact = case["artifact"]
                if (
                    not artifact["fresh_process_cpu_replay"]
                    or len(artifact["identity"]) != 64
                ):
                    raise ValueError(f"missing independent artifact: {lane}")
            if lane == "bicycle":
                if trial["nodes"] != 10 or trial["samples_per_single_context"] != 250:
                    raise ValueError("incorrect published OU experiment size")
                if not cases[-1]["converged"]:
                    raise ValueError(
                        "full-intervention OU fit did not reach its declared tolerance"
                    )
                if any(c["independent_law_max_absolute_error"] > 1e-7 for c in cases):
                    raise ValueError("OU reference law disagreement")
            elif (
                trial["nodes"],
                trial["samples_per_single_intervention"],
                trial["training_rows"],
            ) != (20, 5000, 120000):
                raise ValueError("incorrect published nonlinear experiment size")
            elif not trial["fit_converged"]:
                raise ValueError(
                    "nonlinear fit did not reach its declared stopping rule"
                )
    directory = ROOT / "parity/cyclic/paper-experiments"
    bicycle = json.loads((directory / "bicycle.json").read_text())
    expected = {
        (t["seed"], f"single-contexts-{c['single_contexts']}")
        for t in bicycle["trials"]
        for c in t["cases"]
        if not c["converged"]
    }
    continued = json.loads((directory / "ou-continuations.json").read_text())["cases"]
    if {(r["seed"], r["case"]) for r in continued} != expected:
        raise ValueError("a budget-limited OU experiment has no named continuation")
    standardized = {
        (r["seed"], r["case"]): r
        for r in json.loads((directory / "ou-standardized.json").read_text())["cases"]
    }
    for row in continued:
        if not row["artifact"]["fresh_process_cpu_replay"]:
            raise ValueError("OU continuation was not independently replayed")
        if not row["converged"]:
            retry = standardized.get((row["seed"], row["case"]))
            if (
                retry is None
                or not retry["converged"]
                or retry["original_score_upper_bound"]
                > retry["original_score_tolerance"]
                or not retry["artifact"]["fresh_process_cpu_replay"]
            ):
                raise ValueError(
                    "OU coordinate-standardization continuation did not meet original score tolerance"
                )
    run([sys.executable, "scripts/check_v3_test_registry.py"], cwd=ROOT)
    run([sys.executable, "scripts/generate_example_catalog.py", "--check"], cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python with the current wheel, pytest, NumPy installed",
    )
    parser.add_argument(
        "--inventory-only",
        action="store_true",
        help="Only check scope, fixture and research-result integrity",
    )
    args = parser.parse_args()
    inventory()
    if args.inventory_only:
        return
    for crate, test in RUST_CHECKS:
        command = ["cargo", "test", "-p", crate, "--lib", "-j", "4"]
        if test:
            command.append(test)
        run(command, cwd=ROOT)
    with tempfile.TemporaryDirectory(prefix="antecedent-cyclic-gate-") as directory:
        target = Path(directory) / "test_cyclic.py"
        target.write_bytes((ROOT / "python/tests/test_cyclic.py").read_bytes())
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        run(
            [args.python, "-m", "pytest", "--noconftest", "-q", str(target)],
            cwd=directory,
            env=env,
        )
        fixture = ROOT / "parity/cyclic/general-equilibrium"
        code = (
            "import json,pathlib;from antecedent.cyclic import consume_artifact;"
            f"p=pathlib.Path({str(fixture)!r});"
            "expected=json.loads((p/'expected-identity.json').read_text());"
            "r=consume_artifact((p/'nonlinear.ant').read_bytes(),expected_identity=expected['identity']);"
            "assert r.report==json.loads((p/'report.json').read_text())"
        )
        run([args.python, "-c", code], cwd=directory, env=env)
    print(
        "Cyclic implementation gate PASSED; calibration and attestation were not run."
    )


if __name__ == "__main__":
    main()
