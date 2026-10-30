#!/usr/bin/env python3
"""Validate the frozen acceptance manifest (parity/acceptance_manifest.toml).

Every `[[case]]` names a public workflow, a supported configuration, the required
outputs, frozen targets and a simulated representative dataset. Evidence starts as
`planned`; a `bound` case names `<path>::<fn>` and the test file must define that
function (Rust `fn name`, Python `def name`). `--require-bound` fails on any planned
case and is used at release closure. A bound case must also name a real test: a
live (not skipped, not deferred) function that asserts, never a script that merely
documents a limit, and its evidence may not be an audit, a configurable limit, a
failed measurement, documentation-only narrowing or a deferral. An `#[ignore]` test
closes only a calibrated construction case or a release-only scale case run by a named
gate script (`scripts/gate_*.sh`).

    python3 scripts/check_acceptance_manifest.py
    python3 scripts/check_acceptance_manifest.py --require-bound
    python3 scripts/check_acceptance_manifest.py --self-test
    python3 scripts/check_acceptance_manifest.py --scale-cases   # JSON for scripts/run_scale_cases.py
"""

from __future__ import annotations

import json
import ast
import io
import tokenize
import textwrap
import math
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

from _repo import REPO_ROOT as ROOT
from _repo import test_target

MANIFEST = ROOT / "parity/acceptance_manifest.toml"

AREAS = frozenset(
    [
        "ingest",
        "identifiers",
        "missing_data",
        "treatment_levels",
        "sampling_weights",
        "adaptive_capacity",
        "adaptive_family",
        "adaptive_inference",
        "adaptive_validation",
        "discovery_scale",
        "discovery_estimation",
        "discovery_knowledge",
        "discovery_splitting",
        "discovery_validation",
        "calibration_ranges",
        "practitioner_scale",
        "interval_evidence",
        "posterior_draws",
        "decision_reliability",
        "evidence_synthesis",
        "predictive_scores",
        "randomized_intervals",
        "matching_scale",
        "fit_vs_artifact",
        "recalculation_bridge",
        "scalable_ci_sensitivity",
        "graph_ceilings",
        "shared_exogenous",
        "transported_utility",
        "graph_posteriors",
        "survival_policy",
        "discovered_transport",
        "release_closure",
        "intervention_programs",
        "policy_validation",
        "intervention_callbacks",
        "policy_integration",
        "portable_artifacts",
        "mechanism_selection",
        "graph_falsification",
        "noise_replay",
        "intrinsic_influence",
        "attribution_uncertainty",
        "public_workflows",
        "performance_conformance",
        "dowhy_comparison",
        "release_conformance",
    ]
)
CALIBRATED_KINDS = frozenset({"calibrated_interval", "calibrated_statistical_test"})
KINDS = frozenset({"value", "refusal", "interval", "receipt"}) | CALIBRATED_KINDS
INTERVAL_KINDS = frozenset({"interval", "calibrated_interval"})
STATUSES = frozenset({"planned", "bound"})
CASE_FIELDS = (
    "id",
    "area",
    "workflow",
    "configuration",
    "expected",
    "targets",
    "dataset",
    "evidence",
)
ID = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*")
MIN_REPLICATES = 2000
MIN_REPORTING_RATE = 0.90
RATE_KEYS = (
    "min_reporting_rate",
    "nominal_coverage",
    "coverage_floor",
    "min_power",
    "min_best_arm_identification_rate",
    "simultaneous_coverage_floor",
    "max_false_superiority_rate",
)


SCALE_GATE = "scripts/gate_scale_cases.sh"
# Where an ignored release-only scale test may live: an integration test or the crate's own tests.
SCALE_TARGET = re.compile(r"crates/(?P<crate>[^/]+)/(?:tests/(?P<target>[^/]+)|src/.+)\.rs$")
EVIDENCE_FIELDS = frozenset({"status", "test"})
# What can never close a case, however it is spelled in the evidence.
FORBIDDEN_CLOSURE_KINDS = frozenset(
    {
        "audit",
        "limit",
        "configurable_limit",
        "failed_measurement",
        "documentation_only",
        "documentation_narrowing",
        "deferral",
        "deferred",
    }
)
DEFERRED_TEST = re.compile(
    r"\btodo!|\bunimplemented!|pytest\.(?:skip|xfail)|mark\.(?:skip|xfail)|unittest\.skip"
    r"|\bskipTest\b|\bexpectedFailure\b|#\[should_panic"
)
# Only actual assertion operations count; helper names are resolved to their
# bodies below, never treated as assertions merely by their spelling.
ASSERTS = re.compile(
    r"\b(?:assert|assert_eq|assert_ne|debug_assert|debug_assert_eq|debug_assert_ne|panic)!\s*\("
    r"|\bassert\b|\.raises\s*\(|\bself\.assert[A-Z]\w*\s*\(|\bnp\.testing\.assert_\w+\s*\(|\.unwrap_err\s*\("
)


def assertion_code(text: str, suffix: str) -> str:
    """Remove comments and string literals before following calls or assertions."""
    if suffix == ".rs":
        from _rust_source import lex
        return lex(text).skeleton
    text = textwrap.dedent(text)
    lines = text.splitlines(keepends=True)
    try:
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type not in (tokenize.COMMENT, tokenize.STRING):
                continue
            (start_line, start_col), (end_line, end_col) = token.start, token.end
            for index in range(start_line - 1, end_line):
                line = lines[index]
                lo = start_col if index == start_line - 1 else 0
                hi = end_col if index == end_line - 1 else len(line.rstrip("\r\n"))
                lines[index] = line[:lo] + " " * (hi - lo) + line[hi:]
    except (tokenize.TokenError, IndentationError):
        return ""
    return "".join(lines)


RUST_TEST_ATTRIBUTE = re.compile(r"#\[(?:tokio::)?test\b|#\[rstest|#\[test_case")


PLANNING_TOKEN = re.compile(r"item|v\d+|[a-z]\d+")
ALLOWED_TOKENS = frozenset({"hc0", "hc1", "hc2", "hc3"})


def _label_problem(case_id: str) -> str | None:
    """Ids name the science; planning labels and version numbers are not allowed."""
    tokens = case_id.split("_")
    for token, following in zip(tokens, tokens[1:] + [""], strict=True):
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
    config = case["configuration"]
    if isinstance(config, dict) and "calibration_grid_point" in config:
        point = config["calibration_grid_point"]
        if type(point) is not int or point not in (0, 1, 2) or kind not in CALIBRATED_KINDS:
            errors.append(
                f"{prefix}: calibration_grid_point must bind a calibrated case to integer 0, 1 or 2"
            )
    if kind not in KINDS:
        errors.append(f"{prefix}: expected.kind must be one of {sorted(KINDS)}")
    else:
        outputs = expected.get("outputs")
        if (
            not isinstance(outputs, list)
            or not outputs
            or any(not isinstance(o, str) or not o.strip() for o in outputs)
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
        if "target_change" in targets and (
            not isinstance(targets["target_change"], str) or not targets["target_change"].strip()
        ):
            errors.append(
                f"{prefix}: targets.target_change must state the reason a target was re-derived"
            )
        for key in RATE_KEYS:
            if key in targets and not (
                isinstance(targets[key], (int, float))
                and not isinstance(targets[key], bool)
                and 0 < targets[key] <= 1
            ):
                errors.append(f"{prefix}: targets.{key} must be in (0, 1]")
        if kind in INTERVAL_KINDS or kind == "calibrated_statistical_test":
            nominal = targets.get("nominal_coverage")
            floor = targets.get("coverage_floor")
            if not isinstance(nominal, (int, float)) or not 0.5 < nominal < 1:
                errors.append(f"{prefix}: interval cases need targets.nominal_coverage in (0.5, 1)")
            elif not isinstance(floor, (int, float)) or not 0.5 < floor <= nominal:
                errors.append(
                    f"{prefix}: interval cases need targets.coverage_floor in (0.5, nominal_coverage]"
                )
        if kind in CALIBRATED_KINDS:
            reps = targets.get("min_replicates")
            if not isinstance(reps, int) or isinstance(reps, bool) or reps < MIN_REPLICATES:
                errors.append(
                    f"{prefix}: calibrated cases need targets.min_replicates >= {MIN_REPLICATES}"
                )
            rate = targets.get("min_reporting_rate")
            if not isinstance(rate, (int, float)) or rate < MIN_REPORTING_RATE:
                errors.append(
                    f"{prefix}: calibrated cases need targets.min_reporting_rate >= {MIN_REPORTING_RATE}"
                )
            if kind == "calibrated_interval" and not any(
                key.startswith("max_median") for key in targets
            ):
                errors.append(f"{prefix}: calibrated cases need a precision target (max_median_*)")
            if kind == "calibrated_statistical_test":
                if (
                    case["configuration"].get("calibration_target")
                    != "conditional_null_rejection_rate"
                    or not isinstance(targets.get("type_i_alpha"), (int, float))
                    or not math.isclose(
                        targets["type_i_alpha"],
                        1 - targets.get("nominal_coverage", 0),
                        abs_tol=1e-12,
                    )
                ):
                    errors.append(
                        f"{prefix}: statistical-test calibration needs its explicit conditional Type-I target"
                    )
                if (
                    targets.get("precision_target") != "not_applicable"
                    or not isinstance(targets.get("precision_reason"), str)
                    or not targets["precision_reason"].strip()
                    or any(key.startswith("max_median") for key in targets)
                ):
                    errors.append(
                        f"{prefix}: statistical-test width must be explicitly inapplicable, never an interval precision claim"
                    )

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
            errors.append(
                f"{prefix}: dataset.provenance must state the dataset is a simulated representative"
            )
        if "n" in dataset and not _positive_number(dataset["n"]):
            errors.append(f"{prefix}: dataset.n must be positive")

    errors.extend(evidence_problems(prefix, case["evidence"], kind=kind))
    return errors


_DEFINITION = {
    ".rs": r"^(\s*)(?:pub(?:\([a-z]+\))?\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+{name}\b",
    ".py": r"^(\s*)(?:async\s+)?def\s+{name}\b",
}
_CALLED = {
    ".rs": re.compile(r"\b([A-Za-z_]\w*)\s*[(!]"),
    ".py": re.compile(r"\b([A-Za-z_]\w*)\s*\("),
}


def _attributes_above(lines: list[str], index: int) -> str:
    """Attribute, decorator and comment lines directly above line `index`, multi-line attributes whole."""
    above: list[str] = []
    in_attribute = False
    for previous in reversed(lines[:index]):
        stripped = previous.strip()
        if stripped.startswith(("#[", "@", "//")):
            in_attribute = False
        elif in_attribute or stripped.endswith("]"):
            in_attribute = not stripped.startswith("#[")
        else:
            break
        above.append(previous)
    return "\n".join(above)


def test_source(text: str, suffix: str, name: str) -> tuple[str, str] | None:
    """The attributes above `name` and its source (signature included); `None` if undefined."""
    lines = text.splitlines()
    if suffix == ".py":
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return None
        definitions = [node for node in ast.walk(tree)
                       if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name]
        if not definitions:
            return None
        node = min(definitions, key=lambda item: item.lineno)
        return _attributes_above(lines, node.lineno - 1), "\n".join(lines[node.lineno - 1:node.end_lineno])
    pattern = re.compile(_DEFINITION[suffix].format(name=re.escape(name)))
    for index, line in enumerate(lines):
        match = pattern.match(line)
        if not match:
            continue
        indent = match.group(1)
        body = [line]
        for following in lines[index + 1 :]:
            if suffix == ".rs":
                if following == f"{indent}}}":
                    break
            elif following.strip() and not following.startswith((indent + " ", indent + "\t")):
                break
            body.append(following)
        return _attributes_above(lines, index), "\n".join(body)
    return macro_test_source(text, name) if suffix == ".rs" else None


def macro_test_source(text: str, name: str) -> tuple[str, str] | None:
    """Recognize same-file test macros forwarding their first identifier unchanged.

    This deliberately supports only the simple declarative form used by the calibration
    roster. A mention in a comment, a non-test macro or renamed forwarding is not evidence.
    """
    if not re.search(r"!\s*\(\s*" + re.escape(name) + r"\s*[,)]", text):
        return None
    from _rust_source import lex

    source = lex(text)
    definitions: dict[str, tuple[str, str]] = {}
    for match in re.finditer(r"\bmacro_rules!\s+(\w+)\s*\{", source.skeleton):
        start, depth, end = match.end(), 1, match.end()
        while end < len(source.skeleton) and depth:
            depth += (source.skeleton[end] == "{") - (source.skeleton[end] == "}")
            end += 1
        if depth:
            continue
        body = source.skeleton[start : end - 1]
        binder = re.match(r"\s*\(\s*\$(\w+)\s*:\s*ident\s*[,)]", body)
        if binder:
            definitions[match.group(1)] = (binder.group(1), source.kept[start : end - 1])

    def resolve(macro: str, visited: set[str]) -> tuple[str, str] | None:
        if macro in visited or macro not in definitions:
            return None
        parameter, body = definitions[macro]
        skeleton = lex(body).skeleton
        generated = re.search(r"\bfn\s+\$" + re.escape(parameter) + r"\s*\(", skeleton)
        if generated and re.search(r"#\s*\[\s*test\s*\]", skeleton[: generated.start()]):
            return body[: generated.start()], body.replace("$" + parameter, name)
        for forwarded in re.finditer(
            r"\b(\w+)!\s*\(\s*\$" + re.escape(parameter) + r"\s*[,)]", skeleton
        ):
            result = resolve(forwarded.group(1), visited | {macro})
            if result:
                return result[0], body + "\n" + result[1]
        return None

    for invocation in re.finditer(
        r"\b(\w+)!\s*\(\s*" + re.escape(name) + r"\s*[,)]", source.skeleton
    ):
        result = resolve(invocation.group(1), set())
        if result:
            return result
    return None


def with_helpers(text: str, suffix: str, source: str, depth: int = 3) -> str:
    """`source` followed by the same-file helpers it calls, to `depth` levels."""
    if depth == 0:
        return assertion_code(source, suffix)
    expanded = [assertion_code(source, suffix)]
    for helper in sorted(set(_CALLED[suffix].findall(assertion_code(source, suffix)))):
        found = test_source(text, suffix, helper)
        if found is not None and found[1] != source:
            expanded.append(with_helpers(text, suffix, found[1], depth - 1))
    return "\n".join(expanded)


def scale_gate_wired(root: Path) -> bool:
    """Whether the release-candidate gate runs the release-only scale cases (SCALE_GATE)."""
    gate, candidate = root / SCALE_GATE, root / "scripts/gate_release_candidate.sh"
    return (
        gate.is_file()
        and candidate.is_file()
        and SCALE_GATE in candidate.read_text(encoding="utf-8")
    )


def scale_cases(data: dict[str, Any], root: Path = ROOT) -> list[dict[str, Any]]:
    """The bound `#[ignore]` release-only scale tests the scale gate must run, with their budgets."""
    cases = []
    for case in data.get("case", []):
        evidence = case.get("evidence", {})
        if evidence.get("status") != "bound" or case["expected"]["kind"] in CALIBRATED_KINDS:
            continue
        path, name = evidence["test"].split("::", 1)[0], evidence["test"].rsplit("::", 1)[1]
        target = SCALE_TARGET.match(path)
        source = root / path
        if target is None or not source.is_file():
            continue
        found = test_source(source.read_text(encoding="utf-8"), ".rs", name)
        if found is None or "#[ignore" not in found[0]:
            continue
        manifest = tomllib.loads(
            (root / "crates" / target["crate"] / "Cargo.toml").read_text(encoding="utf-8")
        )
        cargo_target = target["target"]
        if cargo_target is not None:
            cargo_target, prefix = test_target(target["crate"], cargo_target, root)
            name = prefix + name
        cases.append(
            {
                "case": case["id"],
                "package": manifest["package"]["name"],
                "target": cargo_target,
                "test": name,
                "runtime_seconds": case["targets"]["runtime_seconds"],
                "memory_bytes": case["targets"]["memory_bytes"],
            }
        )
    return cases


def _checker_in_a_gate(root: Path, path: str) -> bool:
    """A `scripts/check_*.py` or `gate_*.py` function stands for the checker a gate script runs."""
    name = Path(path).name
    return (
        path.startswith("scripts/")
        and name.startswith(("check_", "gate_"))
        and any(
            name in gate.read_text(encoding="utf-8")
            for gate in (root / "scripts").glob("gate_*.sh")
        )
    )


def closure_problems(
    prefix: str,
    evidence: dict[str, Any],
    kind: Any,
    root: Path,
    path: str,
    name: str,
    text: str,
    suffix: str,
) -> list[str]:
    """A bound test must be a live asserting test, and no forbidden kind may close the case."""
    errors: list[str] = []
    for key in sorted(set(evidence) - EVIDENCE_FIELDS):
        reason = "cannot close a case" if key in FORBIDDEN_CLOSURE_KINDS else "is not evidence"
        errors.append(f"{prefix}: evidence.{key} {reason}; only a live test binds a case")
    found = test_source(text, suffix, name)
    if found is None:
        return errors
    decoration, source = found
    bench = suffix == ".rs" and "/benches/" in f"/{path}"
    checker = suffix == ".py" and _checker_in_a_gate(root, path)
    where = f"{path}::{name}"
    if suffix == ".rs" and not bench and not RUST_TEST_ATTRIBUTE.search(decoration):
        errors.append(f"{prefix}: {where} is not a test function")
    if suffix == ".py" and not checker and not name.startswith("test"):
        errors.append(f"{prefix}: {where} is not a test function")
    if (
        "#[ignore" in decoration
        and kind not in CALIBRATED_KINDS
        and (SCALE_TARGET.match(path) is None or not scale_gate_wired(root))
    ):
        errors.append(
            f"{prefix}: {where} is #[ignore]d and {SCALE_GATE} does not run it in the "
            "release-candidate gate; only a calibrated construction case or a release-only "
            "scale case run by a named gate may bind an ignored test"
        )
    if DEFERRED_TEST.search(decoration + "\n" + source):
        errors.append(f"{prefix}: {where} skips or defers its check")
    if (
        kind != "calibrated_interval"
        and not checker
        and not ASSERTS.search(with_helpers(text, suffix, source))
    ):
        errors.append(
            f"{prefix}: {where} asserts nothing; a limit that is only documented does not close a case"
        )
    return errors


def evidence_problems(prefix: str, evidence: Any, root: Path = ROOT, kind: Any = None) -> list[str]:
    if not isinstance(evidence, dict) or evidence.get("status") not in STATUSES:
        return [f"{prefix}: evidence.status must be one of {sorted(STATUSES)}"]
    test = evidence.get("test")
    if evidence["status"] == "planned":
        return [f"{prefix}: a planned case must not name a test"] if test is not None else []
    if not isinstance(test, str) or "::" not in test:
        return [f'{prefix}: a bound case needs evidence.test = "<path>::<fn>"']
    path, name = test.split("::", 1)[0], test.rsplit("::", 1)[1]
    source = (root / path).resolve()
    if root.resolve() not in source.parents or not source.is_file():
        return [f"{prefix}: test file {path} does not exist in the repository"]
    if source.suffix not in {".rs", ".py"}:
        return [f"{prefix}: test file {path} must be .rs or .py"]
    text = source.read_text(encoding="utf-8")
    if test_source(text, source.suffix, name) is None:
        return [f"{prefix}: {path} does not define {name}"]
    return closure_problems(prefix, evidence, kind, root, path, name, text, source.suffix)


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
            cases.append(
                {
                    "id": f"case_for_{area}",
                    "area": area,
                    "workflow": "antecedent.analyze",
                    "configuration": {"rows": 10},
                    "expected": {"kind": "value", "outputs": ["x"]},
                    "targets": {"runtime_seconds": 1, "memory_bytes": 1, "rationale": "r"},
                    "dataset": {
                        "generator": "g",
                        "seed": i,
                        "simulated_representative": True,
                        "provenance": "Simulated representative of the described study.",
                    },
                    "evidence": {"status": "planned"},
                }
            )
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
    expect(
        "target change recorded",
        mutate(lambda c, d: c["targets"].update(target_change="Re-derived at the realized size.")),
        None,
    )
    expect(
        "empty target change",
        mutate(lambda c, d: c["targets"].update(target_change=" ")),
        "target_change",
    )
    expect(
        "target change not text",
        mutate(lambda c, d: c["targets"].update(target_change=3)),
        "target_change",
    )
    expect(
        "refusal needs reason",
        mutate(lambda c, d: c["expected"].update(kind="refusal")),
        "expected.refusal",
    )
    expect(
        "calibrated needs replicates",
        mutate(lambda c, d: c["expected"].update(kind="calibrated_interval")),
        "nominal_coverage",
    )
    expect(
        "not simulated",
        mutate(lambda c, d: c["dataset"].update(simulated_representative=False)),
        "simulated_representative",
    )
    expect(
        "planned with test",
        mutate(lambda c, d: c["evidence"].update(test="x::y")),
        "must not name a test",
    )
    expect(
        "bound without test",
        mutate(lambda c, d: c["evidence"].update(status="bound")),
        "evidence.test",
    )
    expect(
        "bound missing file",
        mutate(lambda c, d: c["evidence"].update(status="bound", test="nope/missing.rs::t")),
        "does not exist",
    )
    expect(
        "bound ok",
        mutate(
            lambda c, d: c["evidence"].update(
                status="bound",
                test="scripts/test_check_acceptance_manifest.py::test_require_bound_rejects_a_planned_case",
            )
        ),
        None,
    )
    expect(
        "bound missing fn",
        mutate(
            lambda c, d: c["evidence"].update(
                status="bound", test="scripts/test_check_acceptance_manifest.py::absent_fn"
            )
        ),
        "does not define",
    )
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
    if "--scale-cases" in args:
        print(json.dumps(scale_cases(tomllib.loads(MANIFEST.read_text(encoding="utf-8")))))
        return 0
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
