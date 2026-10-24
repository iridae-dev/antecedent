#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Execute X05 acceptance procedures and collect source-bound per-row evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "conformance/response/joint_panel/experiment_acceptance_report.json"
COMMANDS = [
    [
        "cargo",
        "test",
        "-p",
        "antecedent",
        "--lib",
        "experiment::matrix::tests",
        "--release",
        "--offline",
        "--",
        "--nocapture",
    ],
    [
        "cargo",
        "test",
        "-p",
        "antecedent",
        "--lib",
        "all_native_target_kinds_execute_full_experiment_matrix_against_independent_references",
        "--release",
        "--offline",
        "--",
        "--nocapture",
    ],
    [
        "cargo",
        "test",
        "-p",
        "antecedent",
        "--test",
        "suite_01",
        "joint_panel_workflow::experiment_chain::matrix",
        "--release",
        "--offline",
        "--",
        "--nocapture",
    ],
    [
        "cargo",
        "test",
        "-p",
        "antecedent",
        "--test",
        "suite_01",
        "joint_panel_workflow::actual_bayesian_panel_fit_preserves_mixing_diagnostics_and_shared_draw_axes",
        "--release",
        "--offline",
        "--",
        "--exact",
        "--nocapture",
    ],
]


def source_identity() -> str:
    """Hash actual executable procedure sources; a changing checkout cannot earn a report."""
    digest = hashlib.sha256()
    paths = []
    for directory in [
        "crates/antecedent/src/analysis/joint_panel",
        "crates/antecedent-estimate/src/joint_design",
        "crates/antecedent-contract/src/bridge",
        "crates/antecedent/tests/joint_panel_experiment_chain",
        "conformance/response/joint_panel",
    ]:
        paths.extend(
            p
            for p in (ROOT / directory).rglob("*")
            if p.is_file() and p.suffix in {".rs", ".py"}
        )
    paths.extend(
        ROOT / p
        for p in [
            "crates/antecedent/tests/joint_panel_experiment_chain.rs",
            "crates/antecedent/tests/joint_panel_workflow.rs",
            "scripts/run_joint_panel_experiment_acceptance.py",
        ]
    )
    for path in sorted(set(paths)):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def collect(directory: Path, identity: str, commands: list[dict]) -> dict:
    protocol = json.loads((directory / "protocol.json").read_text())
    records: dict[str, list] = {}
    for path in directory.glob("*.json"):
        if path.name == "protocol.json":
            continue
        record = json.loads(path.read_text())
        if record["source_identity"] != identity:
            raise ValueError("stale acceptance evidence")
        records.setdefault(record["row"], []).append(record)
    cases = {
        "factorial_multi_arm": ["compiled_world_factorial", "compiled_world_multi_arm"],
        "geo_experiment": ["compiled_world_geo", "native_weighted_geo"],
        "sequential_observational": [
            "compiled_world_longitudinal",
            "native_gaussian_longitudinal",
        ],
        "switchback": ["switchback"],
        "instrumented_dose": ["instrumented_dose", "native_local_complier"],
        "difference_in_differences": ["difference_in_differences"],
        "synthetic_control": ["synthetic_control"],
        "retained_panel_model_conditional": ["retained_panel"],
    }
    rows = []
    for obligation in protocol["rows"]:
        evidence = records.pop(obligation["id"], [])
        missing = sorted(
            set(cases[obligation["design"]]) - {e["case"] for e in evidence}
        )
        rows.append(
            {
                **obligation,
                "status": "unresolved" if missing else "passed",
                "missing_procedure_cases": missing,
                "executed_evidence": evidence,
            }
        )
    if records:
        raise ValueError(f"evidence outside advertised protocol: {list(records)}")
    return {
        "version": 1,
        "source_identity": identity,
        "update_semantics": protocol["modular_policy"],
        "release_calibration_license": False,
        "commands": commands,
        "counts": {
            "required_rows": len(rows),
            "passed_rows": sum(r["status"] == "passed" for r in rows),
            "unresolved_rows": sum(r["status"] != "passed" for r in rows),
        },
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dir", default=str(ROOT / "target"))
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        help="retain raw executable evidence in this directory",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="collect an existing source-bound evidence directory",
    )
    args = parser.parse_args()
    directory = args.evidence_dir or Path(
        tempfile.mkdtemp(prefix="antecedent-x05-acceptance-")
    )
    directory.mkdir(parents=True, exist_ok=True)
    identity = source_identity()
    env = {
        **os.environ,
        "CARGO_TARGET_DIR": args.target_dir,
        "ANTECEDENT_X05_ACCEPTANCE_DIR": str(directory),
        "ANTECEDENT_X05_SOURCE_ID": identity,
    }
    executions = []
    if not args.collect_only:
        for index, command in enumerate(COMMANDS):
            print(f"[{index + 1}/{len(COMMANDS)}] {' '.join(command)}", flush=True)
            code = subprocess.run(command, cwd=ROOT, env=env, check=False).returncode
            executions.append({"argv": command, "exit_code": code})
            if code:
                print(
                    f"Procedure failed; retained raw evidence: {directory}", flush=True
                )
                return code
    if source_identity() != identity:
        raise RuntimeError(
            "procedure source changed during execution; rerun on a stable candidate"
        )
    report = collect(directory, identity, executions)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["counts"], sort_keys=True), flush=True)
    return int(report["counts"]["unresolved_rows"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
