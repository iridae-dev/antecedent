#!/usr/bin/env python3
"""Validate the frozen acceptance manifest (parity/acceptance_manifest.toml).

Every `[[case]]` names a public workflow, a supported configuration, the required
outputs, frozen targets and a simulated representative dataset. Evidence starts as
`planned`; a `bound` case names `<path>::<fn>` and the test file must define that
function (Rust `fn name`, Python `def name`). `--require-bound` fails on any planned
case and is used at release closure.

    python3 scripts/check_acceptance_manifest.py
    python3 scripts/check_acceptance_manifest.py --require-bound
    python3 scripts/check_acceptance_manifest.py --self-test
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path
from typing import Any

from _repo import REPO_ROOT as ROOT

MANIFEST = ROOT / "parity/acceptance_manifest.toml"

AREAS = frozenset("""
ingest identifiers missing_data treatment_levels sampling_weights adaptive_capacity
adaptive_family adaptive_inference adaptive_validation discovery_scale
discovery_estimation discovery_knowledge discovery_splitting discovery_validation
calibration_ranges practitioner_scale interval_evidence posterior_draws
decision_reliability evidence_synthesis predictive_scores randomized_intervals
matching_scale fit_vs_artifact recalculation_bridge scalable_ci_sensitivity
graph_ceilings shared_exogenous transported_utility graph_posteriors survival_policy
discovered_transport release_closure intervention_programs policy_validation
intervention_callbacks policy_integration portable_artifacts mechanism_selection
graph_falsification noise_replay intrinsic_influence attribution_uncertainty
public_workflows performance_conformance dowhy_comparison release_conformance
""".split())
KINDS = frozenset({"value", "refusal", "interval", "calibrated_interval", "receipt"})
INTERVAL_KINDS = frozenset({"interval", "calibrated_interval"})
STATUSES = frozenset({"planned", "bound"})
CASE_FIELDS = ("id", "area", "workflow", "configuration", "expected", "targets", "dataset", "evidence")
ID = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*")
MIN_REPLICATES = 2000
MIN_REPORTING_RATE = 0.90
RATE_KEYS = ("min_reporting_rate", "nominal_coverage", "coverage_floor", "min_power",
             "min_best_arm_identification_rate", "simultaneous_coverage_floor",
             "max_false_superiority_rate")


PLANNING_TOKEN = re.compile(r"item|v\d+|[a-z]\d+")
ALLOWED_TOKENS = frozenset({"hc0", "hc1", "hc2", "hc3"})


def _label_problem(case_id: str) -> str | None:
    """Ids name the science; planning labels and version numbers are not allowed."""
    tokens = case_id.split("_")
    for token, following in zip(tokens, tokens[1:] + [""]):
        if token not in ALLOWED_TOKENS and PLANNING_TOKEN.fullmatch(token):
            return "id contains a planning label"
        if re.fullmatch(r"\d{1,2}", token) and re.fullmatch(r"\d{1,2}", following):
            return "id looks like a version number"
    return None


def _positive_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def case_problems(case: dict[str, Any]) -> list[str]:
    cid = case.get("id")
    prefix = f"{cid!r}" if isinstance(cid, str) else "<case without id>"
    errors: list[str] = []
    for field in CASE_FIELDS:
        if field not in case:
            errors.append(f"{prefix}: missing field {field}")
    if errors:
        return errors
    for extra in sorted(set(case) - set(CASE_FIELDS)):
        errors.append(f"{prefix}: unknown field {extra}")
    if not isinstance(cid, str) or not ID.fullmatch(cid):
        return [f"{prefix}: id must be scientific snake_case"]
    label = _label_problem(cid)
    if label:
        errors.append(f"{prefix}: {label}")
    if case["area"] not in AREAS:
        errors.append(f"{prefix}: unknown area {case['area']!r}")
    if not isinstance(case["workflow"], str) or not case["workflow"].strip():
        errors.append(f"{prefix}: workflow must name the public route")
    if not isinstance(case["configuration"], dict) or not case["configuration"]:
        errors.append(f"{prefix}: configuration must be a non-empty table")

    expected = case["expected"]
    kind = expected.get("kind") if isinstance(expected, dict) else None
    if kind not in KINDS:
        errors.append(f"{prefix}: expected.kind must be one of {sorted(KINDS)}")
    else:
        outputs = expected.get("outputs")
        if not isinstance(outputs, list) or not outputs or any(
            not isinstance(o, str) or not o.strip() for o in outputs
        ):
            errors.append(f"{prefix}: expected.outputs must list the required outputs")
        refusal = expected.get("refusal")
        if kind == "refusal" and not (isinstance(refusal, str) and refusal.strip()):
            errors.append(f"{prefix}: a refusal case must state expected.refusal")
        if kind != "refusal" and "refusal" in expected:
            errors.append(f"{prefix}: only a refusal case may state expected.refusal")

    targets = case["targets"]
    if not isinstance(targets, dict):
        errors.append(f"{prefix}: targets must be a table")
    else:
        for key in ("runtime_seconds", "memory_bytes"):
            if not _positive_number(targets.get(key)):
                errors.append(f"{prefix}: targets.{key} must be a positive number")
        if not isinstance(targets.get("rationale"), str) or not targets["rationale"].strip():
            errors.append(f"{prefix}: targets.rationale is required")
        for key in RATE_KEYS:
            if key in targets and not (isinstance(targets[key], (int, float))
                                       and not isinstance(targets[key], bool)
                                       and 0 < targets[key] <= 1):
                errors.append(f"{prefix}: targets.{key} must be in (0, 1]")
        if kind in INTERVAL_KINDS:
            nominal = targets.get("nominal_coverage")
            floor = targets.get("coverage_floor")
            if not isinstance(nominal, (int, float)) or not 0.5 < nominal < 1:
                errors.append(f"{prefix}: interval cases need targets.nominal_coverage in (0.5, 1)")
            elif not isinstance(floor, (int, float)) or not 0.5 < floor <= nominal:
                errors.append(f"{prefix}: interval cases need targets.coverage_floor in (0.5, nominal_coverage]")
        if kind == "calibrated_interval":
            reps = targets.get("min_replicates")
            if not isinstance(reps, int) or isinstance(reps, bool) or reps < MIN_REPLICATES:
                errors.append(f"{prefix}: calibrated cases need targets.min_replicates >= {MIN_REPLICATES}")
            rate = targets.get("min_reporting_rate")
            if not isinstance(rate, (int, float)) or rate < MIN_REPORTING_RATE:
                errors.append(f"{prefix}: calibrated cases need targets.min_reporting_rate >= {MIN_REPORTING_RATE}")
            if not any(key.startswith("max_median") for key in targets):
                errors.append(f"{prefix}: calibrated cases need a precision target (max_median_*)")

    dataset = case["dataset"]
    if not isinstance(dataset, dict):
        errors.append(f"{prefix}: dataset must be a table")
    else:
        if not isinstance(dataset.get("generator"), str) or not dataset["generator"].strip():
            errors.append(f"{prefix}: dataset.generator is required")
        seed = dataset.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            errors.append(f"{prefix}: dataset.seed must be a non-negative integer")
        if dataset.get("simulated_representative") is not True:
            errors.append(f"{prefix}: dataset.simulated_representative must be true")
        if "Simulated representative" not in str(dataset.get("provenance", "")):
            errors.append(f"{prefix}: dataset.provenance must state the dataset is a simulated representative")
        if "n" in dataset and not _positive_number(dataset["n"]):
            errors.append(f"{prefix}: dataset.n must be positive")

    errors.extend(evidence_problems(prefix, case["evidence"]))
    return errors


def evidence_problems(prefix: str, evidence: Any, root: Path = ROOT) -> list[str]:
    if not isinstance(evidence, dict) or evidence.get("status") not in STATUSES:
        return [f"{prefix}: evidence.status must be one of {sorted(STATUSES)}"]
    test = evidence.get("test")
    if evidence["status"] == "planned":
        return [f"{prefix}: a planned case must not name a test"] if test is not None else []
    if not isinstance(test, str) or "::" not in test:
        return [f"{prefix}: a bound case needs evidence.test = \"<path>::<fn>\""]
    path, name = test.split("::", 1)[0], test.rsplit("::", 1)[1]
    source = (root / path).resolve()
    if root.resolve() not in source.parents or not source.is_file():
        return [f"{prefix}: test file {path} does not exist in the repository"]
    if source.suffix == ".rs":
        pattern = rf"\bfn\s+{re.escape(name)}\b"
    elif source.suffix == ".py":
        pattern = rf"^\s*(?:async\s+)?def\s+{re.escape(name)}\b"
    else:
        return [f"{prefix}: test file {path} must be .rs or .py"]
    if not re.search(pattern, source.read_text(encoding="utf-8"), re.MULTILINE):
        return [f"{prefix}: {path} does not define {name}"]
    return []


def problems(data: dict[str, Any], require_bound: bool = False) -> list[str]:
    errors: list[str] = []
    header = data.get("manifest")
    if not isinstance(header, dict) or header.get("schema_version") != 1:
        errors.append("manifest.schema_version must be 1")
    cases = data.get("case")
    if not isinstance(cases, list) or not cases:
        return errors + ["manifest has no [[case]] entries"]
    seen: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            errors.append("case entry is not a table")
            continue
        errors.extend(case_problems(case))
        cid = case.get("id")
        if isinstance(cid, str):
            if cid in seen:
                errors.append(f"{cid!r}: duplicate id")
            seen.add(cid)
    covered = {c.get("area") for c in cases if isinstance(c, dict)}
    for area in sorted(AREAS - covered):
        errors.append(f"area {area} has no case")
    if require_bound:
        for case in cases:
            if isinstance(case, dict) and case.get("evidence", {}).get("status") != "bound":
                errors.append(f"{case.get('id')!r}: planned case at release closure")
    return errors


def self_test() -> int:
    def fixture() -> dict[str, Any]:
        cases = []
        for i, area in enumerate(sorted(AREAS)):
            cases.append({
                "id": f"case_for_{area}", "area": area, "workflow": "antecedent.analyze",
                "configuration": {"rows": 10},
                "expected": {"kind": "value", "outputs": ["x"]},
                "targets": {"runtime_seconds": 1, "memory_bytes": 1, "rationale": "r"},
                "dataset": {"generator": "g", "seed": i, "simulated_representative": True,
                            "provenance": "Simulated representative of the described study."},
                "evidence": {"status": "planned"},
            })
        return {"manifest": {"schema_version": 1}, "case": cases}

    def mutate(edit) -> dict[str, Any]:
        data = fixture()
        edit(data["case"][0], data)
        return data

    failures: list[str] = []

    def expect(label: str, data: dict[str, Any], needle: str | None, **kw: Any) -> None:
        found = problems(data, **kw)
        if needle is None and found:
            failures.append(f"{label}: unexpected {found[:2]}")
        elif needle is not None and not any(needle in e for e in found):
            failures.append(f"{label}: wanted {needle!r}, got {found[:2]}")

    expect("valid", fixture(), None)
    expect("duplicate id", mutate(lambda c, d: d["case"].append(dict(c))), "duplicate id")
    expect("unknown area", mutate(lambda c, d: c.update(area="nowhere")), "unknown area")
    expect("version in id", mutate(lambda c, d: c.update(id="feature_2_4")), "version")
    expect("planning label", mutate(lambda c, d: c.update(id="a1_ingest")), "planning label")
    expect("hc ids allowed", mutate(lambda c, d: c.update(id="hc0_covariance_100000_rows")), None)
    expect("missing area", mutate(lambda c, d: d["case"].pop()), "has no case")
    expect("no rationale", mutate(lambda c, d: c["targets"].pop("rationale")), "rationale")
    expect("refusal needs reason", mutate(lambda c, d: c["expected"].update(kind="refusal")), "expected.refusal")
    expect("calibrated needs replicates", mutate(lambda c, d: c["expected"].update(kind="calibrated_interval")), "nominal_coverage")
    expect("not simulated", mutate(lambda c, d: c["dataset"].update(simulated_representative=False)), "simulated_representative")
    expect("planned with test", mutate(lambda c, d: c["evidence"].update(test="x::y")), "must not name a test")
    expect("bound without test", mutate(lambda c, d: c["evidence"].update(status="bound")), "evidence.test")
    expect("bound missing file", mutate(lambda c, d: c["evidence"].update(status="bound", test="nope/missing.rs::t")), "does not exist")
    expect("bound ok", mutate(lambda c, d: c["evidence"].update(status="bound", test="scripts/check_acceptance_manifest.py::self_test")), None)
    expect("bound missing fn", mutate(lambda c, d: c["evidence"].update(status="bound", test="scripts/check_acceptance_manifest.py::absent_fn")), "does not define")
    expect("require bound", fixture(), "planned case at release closure", require_bound=True)
    if failures:
        print("check_acceptance_manifest self-test FAILED:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("check_acceptance_manifest self-test: ok")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--self-test" in args:
        return self_test()
    unknown = [a for a in args if a != "--require-bound"]
    if unknown:
        print(f"unknown arguments: {unknown}", file=sys.stderr)
        return 2
    data = tomllib.loads(MANIFEST.read_text(encoding="utf-8"))
    errors = problems(data, require_bound="--require-bound" in args)
    if errors:
        print("acceptance manifest check FAILED:")
        for error in errors:
            print(f"  {error}")
        return 1
    cases = data["case"]
    bound = sum(1 for c in cases if c["evidence"]["status"] == "bound")
    print(f"acceptance manifest: {len(cases)} cases, {bound} bound, {len(cases) - bound} planned")
    return 0


if __name__ == "__main__":
    sys.exit(main())
