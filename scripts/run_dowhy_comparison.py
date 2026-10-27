# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["dowhy==0.14", "pytest==9.1.1", "pydantic==2.13.5"]
# ///
"""Run the existing public-API DoWhy comparison in a locked isolated environment.

uv run --frozen --script scripts/run_dowhy_comparison.py
No upstream implementation/test source is copied. Unlike ordinary pytest, this
explicit reference command fails on a missing/wrong comparator or skipped case.
It runs fixed scenarios, not a calibration or performance acceptance campaign.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]


def result_problems(path: Path, required: set[str]) -> list[str]:
    if not path.is_file():
        return ["comparison produced no JUnit execution report"]
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    passed = {
        case.get("name", "").split("[", 1)[0]
        for case in cases
        if not any(
            case.find(tag) is not None for tag in ("failure", "error", "skipped")
        )
    }
    problems = [
        f"required comparator assertion did not pass: {name}"
        for name in sorted(required - passed)
    ]
    if not cases:
        problems.append("empty comparison selection")
    if any(
        any(case.find(tag) is not None for tag in ("failure", "error", "skipped"))
        for case in cases
    ):
        problems.append("comparison includes failed, errored or skipped cases")
    return problems


def main() -> int:
    baseline = tomllib.loads(
        (ROOT / "parity/baselines/dowhy.toml").read_text(encoding="utf-8")
    )
    if importlib.metadata.version("dowhy") != baseline["version"]:
        raise SystemExit(
            "DoWhy version does not match the baseline; use the frozen script environment"
        )
    checked = subprocess.run(
        [sys.executable, str(ROOT / "scripts/native_build_identity.py"), "check"],
        cwd=ROOT,
        check=False,
    )
    if checked.returncode:
        return checked.returncode
    report = ROOT / "target/scientific-conformance/dowhy.xml"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.unlink(missing_ok=True)
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "python"),
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    command = [
        sys.executable,
        "-m",
        "pytest",
        str(ROOT / baseline["comparison_test"]),
        "-q",
        "--junitxml",
        str(report),
        "--strict-markers",
    ]
    result = subprocess.run(command, cwd=ROOT / "python", env=env, check=False)
    required = {row["checked_by"] for row in baseline["comparison"]}
    problems = result_problems(report, required)
    for problem in problems:
        print(f"FAIL: {problem}")
    receipt = {
        "kind": "fixed_scenario_external_comparison",
        "comparator": "dowhy",
        "version": baseline["version"],
        "command": command,
        "exit_code": result.returncode,
        "required_assertions": sorted(required),
        "problems": problems,
        "calibration_run": False,
        "grants_release_license": False,
    }
    report.with_suffix(".json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return result.returncode or int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
