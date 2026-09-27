"""Generate exact graphless support licenses for Rust and documentation."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "parity/support_graphless.toml"
RUST = ROOT / "crates/antecedent/src/support_graphless_data.rs"
IO_RUST = ROOT / "crates/antecedent-io/src/support_graphless_data.rs"
DOC = ROOT / "docs/graphless-support-matrix.md"
KEYS = ("family", "design", "method", "inference_claim")
INTEGER_LIMITS = (
    "min_rows", "min_assignment_units_per_arm", "min_blocks", "min_block_arm",
    "min_factorial_cell", "min_action_rows", "min_reported_intervals",
    "min_policy_matches", "min_reference_matches", "min_bin_rows",
    "min_bin_arm_rows", "min_group_rows", "min_group_arm_rows",
)
BOOLEAN_REQUIREMENTS = (
    "requires_uncoupled_constraints", "requires_disjoint_nuisance_training",
    "requires_rank_ownership",
)


def load_rows() -> list[dict]:
    rows = tomllib.loads(SOURCE.read_text())["license"]
    seen: set[tuple[str, ...]] = set()
    for row in rows:
        key = tuple(row[field] for field in KEYS)
        if key in seen:
            raise ValueError(f"duplicate graphless support key: {key}")
        seen.add(key)
        for field in (*KEYS, "assignment_unit", "known_truth_test", "retained_route_test", "limitations"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"{key}: missing {field}")
        if row["assignment_unit"] not in ("unit", "cluster"):
            raise ValueError(f"{key}: unsupported assignment unit")
        if not any(field in row for field in INTEGER_LIMITS):
            raise ValueError(f"{key}: missing support threshold")
        for field in INTEGER_LIMITS:
            threshold = row.get(field, 0)
            if type(threshold) is not int or threshold < 0:
                raise ValueError(f"{key}: invalid {field}")
        maximum_covariates = row.get("max_covariates", 0)
        if type(maximum_covariates) is not int or maximum_covariates < 0:
            raise ValueError(f"{key}: invalid max_covariates")
        probability = row.get("min_probability", 0.0)
        if type(probability) not in (float, int) or not 0 <= probability <= 1:
            raise ValueError(f"{key}: invalid min_probability")
        if type(row.get("all_reported_intervals", False)) is not bool:
            raise ValueError(f"{key}: all_reported_intervals must be boolean")
        for field in BOOLEAN_REQUIREMENTS:
            if type(row.get(field, False)) is not bool:
                raise ValueError(f"{key}: {field} must be boolean")
        for evidence in ("known_truth_test", "retained_route_test"):
            path, sep, function = row[evidence].partition("::")
            if not sep or not re.fullmatch(r"[a-zA-Z_][a-zA-Z_0-9]*", function):
                raise ValueError(f"{key}: invalid {evidence} citation")
            source = ROOT / path
            if not source.is_file() or not re.search(rf"\bfn\s+{function}\s*\(", source.read_text()):
                raise ValueError(f"{key}: missing {evidence} function")
            if evidence == "known_truth_test" and "95_normal_interval" in row["inference_claim"]:
                body = re.search(
                    rf"\bfn\s+{function}\s*\(\)\s*\{{(.*?)(?=\n\s*#\[test\]|\Z)",
                    source.read_text(), re.DOTALL,
                )
                legacy_evidence = body is not None and all(token in body.group(1)
                    for token in ("REPLICATES: usize = 2_000", "0.93..=0.985", "covered", "interval_95"))
                policy_evidence = row["family"] == "policy_value" and body is not None and all(
                    token in body.group(1) for token in ("2_000", "coverage")) and any(
                    token in body.group(1) for token in ("interval_95", "intervals_95"))
                did_evidence = row["family"] == "difference_in_differences" and body is not None \
                    and "const REPLICATIONS: usize = 2_000" in source.read_text() \
                    and all(token in body.group(1) for token in ("covered", "interval_95", "0.925..=0.975"))
                if body is None or not (legacy_evidence or policy_evidence or did_evidence) or not any(
                    token in body.group(1) for token in ("truth", "target", "truths", "TRUTH")):
                    raise ValueError(f"{key}: interval evidence must run the 2,000-allocation known-truth coverage gate")
    return sorted(rows, key=lambda row: tuple(row[field] for field in KEYS))


def rust(rows: list[dict], *, io: bool = False) -> str:
    fields = (*KEYS, "assignment_unit", "known_truth_test", "retained_route_test", "limitations")
    lines = [
        "//! Generated from parity/support_graphless.toml; do not edit by hand.",
        *(["#[allow(dead_code)] // Evidence citations are used by the generator gate, not IO validation."] if io else []),
        "pub(super) struct GraphlessLicenseRow {",
        *(f"    pub(super) {field}: &'static str," for field in fields),
        *(f"    pub(super) {field}: usize," for field in INTEGER_LIMITS),
        "    pub(super) max_covariates: usize,",
        "    pub(super) min_probability: f64,",
        "    pub(super) all_reported_intervals: bool,",
        *(f"    pub(super) {field}: bool," for field in BOOLEAN_REQUIREMENTS),
        "}",
        "pub(super) const LICENSES: &[GraphlessLicenseRow] = &[",
    ]
    for row in rows:
        lines.append("    GraphlessLicenseRow {")
        lines.extend(f"        {field}: {json.dumps(row[field], ensure_ascii=False)}," for field in fields)
        lines.extend(f"        {field}: {row.get(field, 0)}," for field in INTEGER_LIMITS)
        lines.append(f"        max_covariates: {row.get('max_covariates', 0)},")
        lines.append(f"        min_probability: {float(row.get('min_probability', 0.0))},")
        lines.append(f"        all_reported_intervals: {str(row.get('all_reported_intervals', False)).lower()},")
        lines.extend(f"        {field}: {str(row.get(field, False)).lower()}," for field in BOOLEAN_REQUIREMENTS)
        lines.append("    },")
    return "\n".join([*lines, "];", ""])


def docs(rows: list[dict]) -> str:
    lines = [
        "# Graphless design-family support matrix",
        "",
        "This table is separate from the graph/structure support axes in",
        "[the geometric matrix](support-matrix.md). A query is licensed only when",
        "its exact family, design, method, inference claim, and observed",
        "assignment-unit counts match a row. All other combinations are refused.",
        "That refusal means no matrix license; off-axis point results may still",
        "execute. The route must validate its stated design and assumptions.",
        "",
        "| Family | Design | Method | Inference claim | Assignment support | Evidence |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        evidence = ", ".join(
            f"[`{row[field].split('::')[-1]}`](../{row[field].split('::')[0]})"
            for field in ("known_truth_test", "retained_route_test")
        )
        lines.append(
            "| " + " | ".join(
                [*(f"`{row[field]}`" for field in KEYS),
                 ", ".join([
                     *(f">= {row[field]} {field.removeprefix('min_').replace('_', ' ')}" for field in INTEGER_LIMITS if row.get(field)),
                     *(f"<= {row['max_covariates']} covariates" for _ in [0] if row.get("max_covariates")),
                     *(f"probability >= {row['min_probability']}" for _ in [0] if row.get("min_probability")),
                     "all intervals" if row.get("all_reported_intervals") else "interval published",
                     *(field.removeprefix("requires_").replace("_", " ") for field in BOOLEAN_REQUIREMENTS if row.get(field)),
                 ]),
                 evidence]
            ) + " |"
        )
    lines.append("")
    for row in rows:
        lines.append(f"**{row['design']} limits:** {row['limitations']}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--evidence-tests", action="store_true")
    args = parser.parse_args()
    rows = load_rows()
    if args.evidence_tests:
        citations = sorted({row[field] for row in rows for field in ("known_truth_test", "retained_route_test")})
        for citation in citations:
            path, function = citation.split("::")
            if path.startswith("crates/antecedent-estimate/src/"):
                print(f"antecedent-estimate\tlib\t-\t{function}")
            elif path.startswith("crates/antecedent/tests/"):
                print(f"antecedent\ttest\t{Path(path).stem}\t{function}")
            else:
                raise ValueError(f"unsupported graphless evidence target: {path}")
        return
    for target, expected in ((RUST, rust(rows)), (IO_RUST, rust(rows, io=True)), (DOC, docs(rows))):
        if args.check:
            if not target.exists() or target.read_text() != expected:
                raise SystemExit(f"{target.relative_to(ROOT)} is stale; regenerate graphless support")
        else:
            target.write_text(expected)


if __name__ == "__main__":
    main()
