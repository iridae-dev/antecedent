#!/usr/bin/env python3
"""Release-candidate receipts for the acceptance manifest.

The ten release-closure cases of parity/acceptance_manifest.toml are statements about the
whole manifest and the gates that execute it. Each is one function here, run on the real
registries by `scripts/gate_release_candidate.sh` and exercised on both a passing and a
failing fixture by scripts/test_check_release_closure.py:

    installed_wheel_problems     every documented example passes from an installed wheel
    practitioner_scale_problems  every practitioner-scale case is wired and measured
    refusal_boundary_problems    every documented refusal boundary refuses
    replay_identity_problems     replay identity holds for each artifact consumer
    calibrated_record_problems   each calibration cell has >= the case's replicates and an MCSE
    confirmation_problems        a boundary record is confirmed on fresh seeds before it closes
    construction_problems        each interval case binds a public test, a record and the attestation

The rest of the closure receipts live beside what they enforce: the manifest checker owns
"no planned case" and "no audit, limit, failed measurement or deferral closes a case"
(scripts/check_acceptance_manifest.py), scripts/ci_workflow.py owns the CI attestation, and
scripts/generate_calibration_backlog.py owns the ready-coordinate receipt.

    python3 scripts/check_release_closure.py
"""

from __future__ import annotations

import math
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

from _repo import REPO_ROOT as ROOT
from _repo import ensure_scripts_on_path

ensure_scripts_on_path()
import check_acceptance_manifest as manifest  # noqa: E402

IDENTITY = re.compile(
    r"replay\w* (?:reproduces|yields|in a fresh|identity matches)|replays? (?:in separate|results)"
    r"|bit-identical|identical (?:results|receipt)|replayed draws", re.I)
EQUALITY = re.compile(r"assert_eq!|assert_ne!|assertEqual|assert\s+[^\n]*==|\bis_identical|\.identity|to_bits")
MEASURED_ROLES = frozenset({"gated", "reported_level"})


def bound_cases(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [c for c in data["case"] if c["evidence"]["status"] == "bound"]


def test_body(case: dict[str, Any], root: Path) -> tuple[str, str, str] | None:
    """`(path, decoration, source with its same-file helpers)` of a bound case's test."""
    path, name = case["evidence"]["test"].split("::", 1)[0], case["evidence"]["test"].rsplit("::", 1)[1]
    source = root / path
    if not source.is_file():
        return None
    text = source.read_text(encoding="utf-8")
    found = manifest.test_source(text, source.suffix, name)
    if found is None:
        return None
    return path, found[0], manifest.with_helpers(text, source.suffix, found[1])


def installed_wheel_problems(candidate: str) -> list[str]:
    """The installed-wheel step of the release-candidate gate, read as text.

    The wheel must be installed into a fresh venv, the suite and every documented example
    must run with that venv's interpreter, from a directory that is not the source tree,
    and the interpreter must first prove it imports the wheel.
    """
    venv = '"$venv_dir/venv/bin/python"'
    problems = []
    if "maturin build" not in candidate or "uv pip install --python" not in candidate:
        problems.append("the gate does not build and install a wheel into a fresh venv")
    for label, needle in (
        ("proves the wheel is the imported package", "gate_release_candidate_2.py"),
        ("runs the Python suite (every shipped example and notebook)", "-m pytest"),
        ("executes the documented snippets and embedded examples", "run_doc_examples.py"),
    ):
        lines = [line for line in candidate.splitlines() if needle in line and venv in line]
        if not lines:
            problems.append(f"the installed-wheel step does not {label} with the venv interpreter")
    leave, doc_examples = candidate.find('cd "$run_dir"'), candidate.find("run_doc_examples.py")
    if leave < 0 or (0 <= doc_examples < leave):
        problems.append("the installed-wheel step does not leave the source tree before running the examples")
    return problems


def example_problems(root: Path = ROOT) -> list[str]:
    """Every shipped example must assert its own result and be run by the Python suite."""
    suite = root / "python/tests/test_examples_run.py"
    if not suite.is_file():
        return ["python/tests/test_examples_run.py does not exist"]
    text = suite.read_text(encoding="utf-8")
    problems = [] if "examples" in text and "runpy.run_path" in text else [
        "python/tests/test_examples_run.py does not execute examples/python"]
    return problems


def executed_by_the_gates(path: str, ci: str) -> bool:
    """Whether the CI `rust` job (verified by the CI run) or the release-candidate suite runs `path`."""
    if path.startswith(("python/tests/", "scripts/test_")):
        return True
    return path.startswith("crates/") and "cargo nextest run" in ci and "--workspace" in ci


def refusal_boundary_problems(data: dict[str, Any], root: Path = ROOT, ci: str = "") -> list[str]:
    """Every refusal case is bound to a live, un-ignored asserting test that a gate executes."""
    problems = []
    for case in data["case"]:
        if case["expected"]["kind"] != "refusal":
            continue
        cid = case["id"]
        if case["evidence"]["status"] != "bound":
            problems.append(f"{cid}: refusal boundary has no bound test")
            continue
        found = test_body(case, root)
        if found is None:
            problems.append(f"{cid}: bound refusal test cannot be read")
            continue
        path, decoration, _ = found
        if "#[ignore" in decoration:
            problems.append(f"{cid}: refusal test is ignored, so no gate checks the boundary")
        if not executed_by_the_gates(path, ci):
            problems.append(f"{cid}: no release gate executes {path}")
    return problems


def replay_identity_problems(data: dict[str, Any], root: Path = ROOT) -> list[str]:
    """Every case that promises replay or identity compares the replay with the original exactly."""
    problems = []
    for case in data["case"]:
        outputs = " ".join(case["expected"]["outputs"])
        if case["expected"]["kind"] not in ("value", "receipt") or not IDENTITY.search(outputs):
            continue
        if case["evidence"]["status"] != "bound":
            problems.append(f"{case['id']}: replay case has no bound test")
            continue
        found = test_body(case, root)
        if found is None or not EQUALITY.search(found[2]):
            problems.append(f"{case['id']}: bound test never compares the replay with the original")
    return problems


def records_of(case: dict[str, Any], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Coverage records the case's bound test emitted at the case's nominal level and size."""
    if split_size_problems(case):
        return []
    targets, size = case["targets"], case_size(case)
    wanted = case["evidence"]["test"]
    matched = []
    for record in records:
        if record.get("test") not in (wanted, wanted.rsplit("::", 1)[1]):
            continue
        if "sample_split_parent_set_effect_calibration.rs::" in wanted:
            fraction = case["configuration"].get("discovery_fraction")
            if fraction != 0.75 or "discovery_fraction_0_75" not in record.get("functional", "").split("."):
                continue
        if ("sample_split_weighted_effect_calibration.rs::" in wanted
                and case["configuration"].get("discovery_fraction") == 0.75
                and not {"discovery_fraction_0_75", "conflict_resolution_treatment_invariant"}
                    .issubset(record.get("functional", "").split("."))):
            continue
        variance = case["configuration"].get("interval_variance")
        if variance is not None and record.get("se_kind") != variance:
            continue
        suffix = case["configuration"].get("record_functional_suffix")
        if suffix is not None and record.get("functional", "").rsplit(".", 1)[-1] != suffix:
            continue
        if not math.isclose(record.get("nominal", math.nan), targets.get("nominal_coverage", math.nan), abs_tol=1e-9):
            continue
        if size is not None and not record["n_min"] <= size <= record["n_max"]:
            continue
        matched.append(record)
    return matched


def split_size_problems(case: dict[str, Any]) -> list[str]:
    """Check declared enrollment-to-analysis binding without changing enrollment claims."""
    config = case["configuration"]
    if not {"analysis_rows", "discovery_fraction"}.intersection(config):
        return []
    total, rows, fraction = (config.get(k) for k in (
        "participants", "analysis_rows", "discovery_fraction"))
    if (type(total) is not int or total <= 0 or type(rows) is not int or rows <= 0
            or isinstance(fraction, bool) or not isinstance(fraction, (int, float))
            or not math.isfinite(fraction) or not 0 < fraction < 1
            or not math.isclose(rows, total * (1 - fraction), rel_tol=0, abs_tol=1e-9)):
        return [f"{case['id']}: analysis_rows must equal participants * (1 - discovery_fraction) "
                "with positive integer counts and a declared fraction in (0, 1)"]
    if config.get("record_size_basis", "analysis_rows") not in {"analysis_rows", "procedure_rows"}:
        return [f"{case['id']}: unknown split record_size_basis"]
    return []


def case_size(case: dict[str, Any]) -> int | None:
    """Estimator record scope uses analysis rows; enrollment remains in the manifest."""
    if ("analysis_rows" in case["configuration"]
            and case["configuration"].get("record_size_basis", "analysis_rows") == "analysis_rows"):
        return case["configuration"]["analysis_rows"]
    return next((case["configuration"][k] for k in ("participants", "rows", "steps")
                 if isinstance(case["configuration"].get(k), int)), None)


def calibrated(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [c for c in bound_cases(data) if c["expected"]["kind"] == "calibrated_interval"]


def points(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [p for p in (record.get("grid") or [record]) if p.get("role", record.get("role")) in MEASURED_ROLES]


def configured_point(measured: list[dict[str, Any]], size: float | None) -> dict[str, Any] | None:
    """The measured point whose size range contains the case's configured size, else the nearest."""
    if not measured:
        return None
    if size is None:
        return measured[len(measured) // 2]

    def distance(point: dict[str, Any]) -> float:
        low, high = float(point.get("n_min", 0)), float(point.get("n_max", 0))
        return 0.0 if low <= size <= high else min(abs(size - low), abs(size - high))

    return min(measured, key=distance)


def calibrated_record_problems(data: dict[str, Any], records: list[dict[str, Any]]) -> list[str]:
    """The point at the case's configured size has >= the case's replicates; every measured point
    states a Monte Carlo SE and meets the coverage floor.

    The case's replicate minimum is a requirement on its own configuration. The record's other grid
    points extend the measured size range; they carry the ordinary first-pass replicates (and the
    harness rechecks any of them at 2,000 when its coverage is off nominal)."""
    problems = []
    for case in calibrated(data):
        cid, targets = case["id"], case["targets"]
        problems.extend(split_size_problems(case))
        matched = records_of(case, records)
        if not matched:
            problems.append(f"{cid}: no coverage record from {case['evidence']['test']} at nominal "
                            f"{targets['nominal_coverage']} spanning the configured size")
        size = case_size(case)
        for record in matched:
            measured = points(record)
            configured = configured_point(measured, size)
            for point in measured:
                where = f"{cid}: record {record['id']} point {point.get('point', 0)}"
                if point is configured and point.get("replicates", 0) < targets["min_replicates"]:
                    problems.append(f"{where} has {point.get('replicates', 0)} replicates, below {targets['min_replicates']}")
                mcse = point.get("mcse")
                if not isinstance(mcse, (int, float)) or not math.isfinite(mcse) or mcse <= 0:
                    problems.append(f"{where} states no Monte Carlo uncertainty")
                if point.get("observed", 0) < targets["coverage_floor"]:
                    problems.append(f"{where} covers {point.get('observed')}, below the floor {targets['coverage_floor']}: "
                                    "a failed measurement does not close a case")
    return problems


def confirmation_problems(data: dict[str, Any], records: list[dict[str, Any]]) -> list[str]:
    """A range claimed after a boundary needs a fresh-seed confirmation covering the configured size."""
    problems = []
    for record in records:
        if record.get("confirmed_passing_range") and not record.get("confirmation"):
            problems.append(f"record {record['id']}: claims a confirmed range with no confirmation")
    for case in calibrated(data):
        size = case_size(case)
        for record in records_of(case, records):
            if not record.get("boundary") and not any(p.get("boundary") for p in record.get("grid", [])):
                continue
            span = record.get("confirmed_passing_range")
            if not span or size is None or not span[0] <= size <= span[1]:
                problems.append(f"{case['id']}: record {record['id']} has a boundary and no independent "
                                f"confirmation covering size {size}")
                continue
            for point in record.get("confirmation", []):
                if point.get("replicates", 0) < case["targets"]["min_replicates"]:
                    problems.append(f"{case['id']}: confirmation of {record['id']} has {point.get('replicates', 0)} replicates")
    return problems


def attestation_problems(candidate: str, release: str) -> list[str]:
    """The release candidate runs the inventory, and the inventory runs the calibration attestation."""
    problems = []
    if "gate_release.sh" not in candidate:
        problems.append("gate_release_candidate.sh does not run gate_release.sh")
    if "gate_calibration_attestation.sh" not in release:
        problems.append("gate_release.sh does not run the calibration attestation (current source)")
    if "check_acceptance_manifest.py --require-bound" not in release:
        problems.append("gate_release.sh does not require every manifest case to be bound")
    return problems


def construction_problems(data: dict[str, Any], records: list[dict[str, Any]], candidate: str,
                          release: str, calibration_gate: str) -> list[str]:
    """Each interval case emits a record bound to its construction and source, and a gate registers it."""
    problems = attestation_problems(candidate, release)
    for case in bound_cases(data):
        if case["expected"]["kind"] not in manifest.INTERVAL_KINDS:
            continue
        cid, test = case["id"], case["evidence"]["test"]
        name = test.rsplit("::", 1)[1]
        if case["expected"]["kind"] == "calibrated_interval":
            if not re.search(rf"\b{re.escape(name)}\b", calibration_gate):
                problems.append(f"{cid}: scripts/gate_calibration.sh does not run {name}")
            for record in records_of(case, records):
                for field in ("calibration_sha", "surface_list_blob", "facets", "estimator", "interval_method"):
                    if not record.get(field):
                        problems.append(f"{cid}: record {record['id']} lacks {field}, so it is not bound to its construction and source")
    return problems


def practitioner_scale_problems(data: dict[str, Any], records: list[dict[str, Any]], calibration_gate: str) -> list[str]:
    """Every practitioner-scale case is bound, registered with the calibration gate and measured."""
    problems = []
    for case in data["case"]:
        if case["area"] != "practitioner_scale":
            continue
        cid = case["id"]
        if case["evidence"]["status"] != "bound":
            problems.append(f"{cid}: practitioner-scale case is not bound")
            continue
        name = case["evidence"]["test"].rsplit("::", 1)[1]
        if case["expected"]["kind"] == "calibrated_interval" and not re.search(rf"\brun_ps\s+{re.escape(name)}\b", calibration_gate):
            problems.append(f"{cid}: scripts/gate_calibration.sh does not run {name} as a practitioner-scale design")
    scale = {c["id"] for c in data["case"] if c["area"] == "practitioner_scale"}
    problems.extend(p for p in calibrated_record_problems(data, records) if p.split(":", 1)[0] in scale)
    return problems


def closure_problems(root: Path = ROOT) -> list[str]:
    data = tomllib.loads((root / "parity/acceptance_manifest.toml").read_text(encoding="utf-8"))
    records = tomllib.loads((root / "parity/coverage_records.toml").read_text(encoding="utf-8"))["record"]
    read = lambda rel: (root / rel).read_text(encoding="utf-8")  # noqa: E731
    candidate, release, calibration = (read(f"scripts/{n}") for n in (
        "gate_release_candidate.sh", "gate_release.sh", "gate_calibration.sh"))
    return [
        *installed_wheel_problems(candidate), *example_problems(root),
        *practitioner_scale_problems(data, records, calibration),
        *refusal_boundary_problems(data, root, read(".github/workflows/ci.yml")),
        *replay_identity_problems(data, root),
        *calibrated_record_problems(data, records), *confirmation_problems(data, records),
        *construction_problems(data, records, candidate, release, calibration),
    ]


def main() -> int:
    problems = closure_problems()
    if problems:
        print("release closure FAILED:")
        for problem in dict.fromkeys(problems):
            print(f"  {problem}")
        return 1
    print("release closure: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
