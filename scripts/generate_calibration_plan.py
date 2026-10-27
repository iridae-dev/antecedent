#!/usr/bin/env python3
"""Inventory inherited calibration debt and the current inference contracts.

This renders declarations, not measurement or a license. Per-output calibration
designs must still cover diagnostics inside point-only records.
"""

from __future__ import annotations

import argparse
import tomllib
from collections import Counter
from copy import deepcopy

from _repo import REPO_ROOT as ROOT
from generate_calibration_backlog import READY_COORDINATES, ready_coordinate_problems

SOURCE = ROOT / "parity/support_licensed.toml"
OUTPUT = ROOT / "parity/calibration_plan.md"
PROMOTION = ROOT / "parity/promotion.toml"
LEDGER = ROOT / "parity/evidence_ledger.toml"
# Hand-maintained tail of the plan: the generator carries it over unchanged.
CAPACITY_HEADING = "## Capacity plan for the uncertainty, transport and decision coordinates"
AXES = ("query", "graph_class", "structure", "inference")
HISTORY_HEADING = "## Historical inherited backlog plan"
INFERENCE_HEADING = "## Current registered inference contracts"


def reviewed_backlog(cells: list[dict], records: list[dict]) -> list[dict]:
    """Accept the original reviewed allocation or its evidenced completion."""
    missing = [
        row for row in cells if row.get("calibration_reason") == "estimator_grid_not_measured"
    ]
    count = len({tuple(row[a] for a in AXES) for row in missing})
    if not missing:
        problems = ready_coordinate_problems(cells, records)
        if problems:
            raise ValueError("empty backlog lacks measured bindings: " + "; ".join(problems))
    elif len(missing) != 39 or count != 17:
        raise ValueError(
            f"backlog changed to {len(missing)} cells / {count} coordinates; review plan"
        )
    return missing


def inherited_history(existing: str) -> str:
    """Retain the original plan verbatim, clearly separated from current standing."""
    start = existing.find(HISTORY_HEADING)
    if start >= 0:
        end = existing.index(INFERENCE_HEADING, start)
        return existing[start:end].rstrip()
    if "These are 39 cells across 17 distinct coordinates." not in existing:
        raise ValueError("completed backlog requires the preserved original 39-cell plan")
    end = existing.index(INFERENCE_HEADING)
    original = existing[:end].replace("# Calibration backlog plan\n\n", "", 1).rstrip()
    return (
        HISTORY_HEADING + "\n\n"
        "The following is the original 39-cell / 17-coordinate allocation, retained unchanged "
        "as historical planning evidence. Its pending descriptions are not current status; "
        "the current bindings and measured records above govern.\n\n" + original
    )


def table_text(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def ledger_scope() -> tuple[list[dict], list[dict]]:
    """Return the promotion records the evidence ledger declares, and its cells.

    The ledger owns a subset of the merged promotion registry; the records it
    names are the ones whose routes, uncertainty methods and output allocations
    this inventory renders. A ledger cell naming no promotion record is an error.
    """
    registry = tomllib.loads(PROMOTION.read_text(encoding="utf-8"))["record"]
    cells = tomllib.loads(LEDGER.read_text(encoding="utf-8"))["cell"]
    owned = {cell["id"] for cell in cells}
    missing = owned - {record["id"] for record in registry}
    if missing:
        raise ValueError(f"ledger names unregistered records {sorted(missing)}")
    return [record for record in registry if record["id"] in owned], cells


def registered_inference() -> list[str]:
    records, cells = ledger_scope()
    by_route = {(cell["id"], cell["route"]): cell for cell in cells}
    expected = {(record["id"], route["name"]) for record in records for route in record["routes"]}
    if expected != set(by_route) or len(by_route) != len(cells):
        raise ValueError("inference inventory requires an exact route-to-evidence ledger")
    lines = [
        "## Current registered inference contracts",
        "",
        (
            f"The following inventory covers the {len(records)} promotion records owned by the evidence ledger and "
            f"{len(expected)} registered routes, including closed producers and independent "
            "consumers. It is generated from the promotion registry and evidence ledger. "
            "A point-only status can still carry unmeasured standard errors, p-values, "
            "posterior summaries or Monte Carlo diagnostics. The historical inherited backlog "
            "does not allocate designs for those outputs. Neither this table nor an "
            "allocated coverage ID establishes that a measurement suite exists or has run."
        ),
        "",
        (
            "Owners, theorem scope, evidence design and inference boundaries remain in "
            "the linked promotion record. Whole-method truth/design, nominal level, "
            "acceptance thresholds and activation must be allocated separately for each "
            "diagnostic or interval variant; input calibration never licenses a new "
            "transformation, retarget, posterior decision or ranking guarantee."
        ),
        "",
        (
            "| Record / owner | Route / standing | Claim and uncertainty boundary | "
            "Allocated calibration / independent consumer |"
        ),
        "| --- | --- | --- | --- |",
    ]
    for record in records:
        notes = record.get("inference_notes") or record["guarantee"]
        owner = ", ".join(record["owners"])
        for route in record["routes"]:
            cell = by_route[(record["id"], route["name"])]
            claim = route.get("claim", record["inference_claim"])
            calibration = ", ".join(record.get("coverage_records", [])) or cell["calibration"]
            lines.append(
                "| "
                + " | ".join(
                    table_text(value)
                    for value in (
                        f"`{record['id']}` / {owner}",
                        f"`{route['name']}` / {route['status']} ({record['status']})",
                        f"{claim}: {notes}",
                        f"{calibration}; consumer `{cell['artifact_consumer_fixture']}`",
                    )
                )
                + " |"
            )
    lines += [
        "",
        (
            "Source: `promotion.toml` and `evidence_ledger.toml`. "
            "Regenerate after changing a public route, uncertainty method or consumer."
        ),
        "",
    ]
    return lines


def disposition(row: dict) -> tuple[str, str]:
    if row["structure"] != "graph_posterior":
        raise ValueError("unreviewed backlog structure")
    quantity = row["query"]
    if quantity in {"ResponseCurve", "InterventionResponse"}:
        oracle = "a finite dose/horizon grid with per-atom truth and a declared pointwise or simultaneous target"
    elif quantity == "ConditionalEffect":
        oracle = (
            "a finite conditional-stratum SCM with support and per-atom conditional-effect truth"
        )
    else:
        oracle = "an enumerated graph/SCM family with per-atom ATE and unidentified-mass truth"
    if row["inference"] == "Bayesian":
        calibration = "posterior predictive/decision calibration for the actual mixed-atom claim"
    else:
        calibration = (
            "whole-method repeated sampling, including the graph-weight and data dependence rule"
        )
    return (
        "graph_mixture_no_whole_method_calibration",
        f"Freeze {oracle}; then measure {calibration}. Current atom values and supplied graph weights do not license an aggregate interval or posterior probability guarantee.",
    )


REQUIRED_INFERENCE_OUTPUT_FIELDS = {
    "id",
    "fields",
    "method",
    "implementation_status",
    "owner",
    "operation_or_object",
    "public_producers",
    "closed_producers",
    "independent_consumers",
    "method_source",
    "truth_design",
    "truth_fixtures",
    "nominal_level",
    "acceptance_threshold",
    "measurement_suite",
    "allocated_coverage_records",
    "inherited_coverage_records",
    "boundary",
    "activation_disposition",
    "allocation_status",
    "measurement_design_status",
    "truth_grid_allocation",
}
INFERENCE_IMPLEMENTATIONS = {
    "implemented_measured_scoped",
    "implemented_unmeasured",
    "candidate_public_closed",
    "implemented_exact_conditional",
    "implemented_model_conditional",
    "implemented_diagnostic",
    "inherited_only",
    "unimplemented_closed",
}


def per_output_inference(records: list[dict]) -> list[str]:
    """Render explicit method allocations without treating a draft as readiness."""
    families = []
    semantic = []
    for record in records:
        if not record.get("inference_inventory_disposition"):
            raise ValueError(f"{record['id']}: per-output source review is missing")
        outputs = record.get("inference_outputs")
        if not isinstance(outputs, list):
            raise TypeError(f"{record['id']}: explicit inference_outputs is required")
        if not outputs:
            semantic.append(record)
        seen = set()
        allocated = set(record.get("coverage_records", [])) | set(
            record.get("candidate_coverage_records", [])
        )
        historical = {entry["id"] for entry in record.get("historical_coverage_records", [])}
        if historical & allocated:
            raise ValueError(f"{record['id']}: historical allocations overlap selected IDs")
        for entry in record.get("historical_coverage_records", []):
            matching = [
                output for output in outputs if entry["id"] in output["allocated_coverage_records"]
            ]
            if entry.get("status") != "failed":
                raise ValueError(
                    f"{record['id']}: historical allocation must retain failed output standing"
                )
            if entry.get("disposition") == "retired_superseded":
                if matching or entry.get("replacement_coverage_id") not in allocated:
                    raise ValueError(
                        f"{record['id']}: retired history needs selected replacement and no active output"
                    )
            elif not matching or any(
                output.get("allocation_status") != "historical_measurement_failed_route_closed"
                for output in matching
            ):
                raise ValueError(
                    f"{record['id']}: historical allocation must retain failed output standing"
                )
        allocated |= historical
        for output in outputs:
            missing = REQUIRED_INFERENCE_OUTPUT_FIELDS - set(output)
            if missing or output.get("id") in seen:
                raise ValueError(f"{record['id']}: invalid output declaration {sorted(missing)}")
            seen.add(output["id"])
            if output["implementation_status"] not in INFERENCE_IMPLEMENTATIONS:
                raise ValueError(f"{record['id']}: unknown output implementation status")
            if output["owner"] not in record["owners"]:
                raise ValueError(f"{record['id']}: output has an unregistered owner")
            if not set(output["allocated_coverage_records"]) <= allocated:
                raise ValueError(f"{record['id']}: output allocates an unowned emitter ID")
            suite = output["measurement_suite"]
            if suite != "unallocated" and not (ROOT / suite.split("::", 1)[0]).is_file():
                raise ValueError(f"{record['id']}: nonexistent measurement suite {suite}")
            if output["implementation_status"] == "unimplemented_closed" and suite != "unallocated":
                raise ValueError(f"{record['id']}: missing method cannot have an executing suite")
            families.append((record, output))
    required = sum(
        output["allocation_status"] == "method_specific_harness_required_before_final_measurement"
        for _, output in families
    )
    prerequisites = sum(
        output["allocation_status"] == "method_prerequisite_missing_calibration_cannot_close"
        for _, output in families
    )
    frozen = sum(
        output["allocation_status"]
        in {
            "frozen_harness_compiled_measurement_pending",
            "implemented_numerical_precision_harness_unmeasured",
        }
        for _, output in families
    )
    lines = [
        "",
        "## Per-output inference methods and allocation",
        "",
        (
            f"All {len(records)} records declare an output disposition: {len(families)} "
            f"output families and {len(semantic)} records with no new numerical output. "
            f"There are {frozen} frozen candidate harness allocations, {required} output "
            f"families still requiring a method-specific harness, and {prerequisites} "
            "closed adjacent methods with an implementation prerequisite. These are "
            "separate from the inherited backlog. The branch is not ready for its final "
            "measurement while required designs/harnesses remain unallocated."
        ),
        "",
        (
            "This is explicit method ownership and design allocation, not itself measurement or "
            "a new license. Allocated candidate records describe future measurements; "
            "inherited records describe unchanged original methods. Neither allocation "
            "nor a numerical replay fixture validates a transformed estimator, covariance, "
            "interval, posterior decision or ranking. Public standing remains in the "
            "promotion routes. Mathematical supplied-law outputs and diagnostic labels "
            "must not be read as calibrated sampling guarantees."
        ),
        "",
        "| Record / output / owner | Fields / method / status | Producer and independent consumer | Truth / nominal / frozen rule or remaining work | Source / suite / allocation and activation |",
        "| --- | --- | --- | --- | --- |",
    ]
    for record, output in families:
        values = [
            f"{record['id']} / {output['id']} / {output['owner']}",
            ", ".join(output["fields"])
            + "; "
            + output["method"]
            + "; "
            + output["implementation_status"],
            "public: "
            + ", ".join(output["public_producers"])
            + "; closed: "
            + ", ".join(output["closed_producers"])
            + "; consumer: "
            + ", ".join(output["independent_consumers"]),
            output["truth_grid_allocation"]
            + "; nominal "
            + output["nominal_level"]
            + "; "
            + output["acceptance_threshold"],
            output["method_source"]
            + "; suite "
            + output["measurement_suite"]
            + "; new IDs: "
            + ", ".join(output["allocated_coverage_records"])
            + "; inherited IDs: "
            + ", ".join(output["inherited_coverage_records"])
            + "; "
            + output["allocation_status"]
            + "; "
            + output["activation_disposition"]
            + "; "
            + output["boundary"],
        ]
        lines.append("| " + " | ".join(table_text(value) for value in values) + " |")
    for record in semantic:
        lines.append(
            "| "
            + record["id"]
            + " | "
            + table_text(record["inference_inventory_disposition"])
            + " | Original source diagnostics/standing remain source-bound | No new numerical inference method declared | Re-review if numerical scope changes |"
        )
    return lines


def render() -> str:
    all_cells = tomllib.loads(SOURCE.read_text(encoding="utf-8"))["cell"]
    records = tomllib.loads((ROOT / "parity/coverage_records.toml").read_text(encoding="utf-8"))[
        "record"
    ]
    cells = reviewed_backlog(all_cells, records)
    counts = Counter(tuple(row[a] for a in AXES) for row in cells)
    cells.sort(
        key=lambda row: (
            row["structure"] != "explicit",
            *(row[a] for a in AXES),
            row.get("validation", ""),
        )
    )
    lines = [
        "# Calibration backlog plan",
        "",
        f"Generated from `support_licensed.toml` without changing its `estimator_grid_not_measured` markers. These are {len(cells)} cells across {len(counts)} distinct coordinates. The fixed-DAG point route is reviewed first; graph-posterior rows remain pending. A typed reason below explains why the missing calibration does not license an inferential claim. It is not a waiver or a coverage record.",
        "",
        "| Priority | Query | Graph | Structure | Inference | Validation | Estimator | Typed reason | Required next evidence |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in cells:
        reason, next_evidence = disposition(row)
        lines.append(
            "| "
            + " | ".join(
                (
                    "fixed graph first"
                    if row["structure"] == "explicit"
                    else "graph posterior later",
                    row["query"],
                    row["graph_class"],
                    row["structure"],
                    row["inference"],
                    row.get("validation", "none"),
                    ", ".join(row.get("estimators", [])),
                    f"`{reason}`",
                    next_evidence,
                )
            )
            + " |"
        )
    lines += [
        "",
        "No row above is relabeled in `support_licensed.toml`. The graph-posterior work starts only after a whole-method oracle and calibration design are frozen at each graph/query/provider coordinate; completion counts are never posterior probabilities, and unidentified or unevaluated mass cannot be renormalized away.",
        "",
    ]
    existing = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else ""
    if not cells:
        lines = [
            "# Calibration backlog plan",
            "",
            (
                "Generated from `support_licensed.toml`: 0 cells across 0 coordinates remain "
                "marked `estimator_grid_not_measured`. All seventeen reviewed coordinates now "
                "cite existing measured records without unresolved boundary points. These bindings "
                "retain their exact method and source scopes; current attestation and execution "
                "binding are still required. Historical planning alone grants no calibration."
            ),
            "",
            "## Completed reviewed coordinate bindings",
            "",
            "| Query | Graph | Structure | Inference | Validation | Exact coverage records |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for row in all_cells:
            if tuple(row[a] for a in AXES) in READY_COORDINATES:
                lines.append(
                    "| "
                    + " | ".join(
                        table_text(str(value))
                        for value in (
                            *(row[a] for a in AXES),
                            row.get("validation", "none"),
                            ", ".join(row["calibration"]),
                        )
                    )
                    + " |"
                )
        lines += ["", inherited_history(existing), ""]
    lines += registered_inference()
    lines += per_output_inference(ledger_scope()[0])
    text = "\n".join(lines)
    start = existing.find(CAPACITY_HEADING)
    if start >= 0:
        text = text.rstrip("\n") + "\n\n" + existing[start:]
    return text


def self_test() -> None:
    """Historical failures stay visible without replacing selected ownership."""
    records = ledger_scope()[0]
    source = next(record for record in records if record.get("historical_coverage_records"))
    per_output_inference([source])
    # Keep generic historical-allocation guards independent of retired live methods.
    source = deepcopy(source)
    for entry in source["historical_coverage_records"]:
        entry.pop("disposition", None)
        entry.pop("replacement_coverage_id", None)
    historical_output = deepcopy(source["inference_outputs"][0])
    historical_output["id"] = "synthetic.historical"
    historical_output["allocation_status"] = "historical_measurement_failed_route_closed"
    historical_output["allocated_coverage_records"] = [
        e["id"] for e in source["historical_coverage_records"]
    ]
    source["inference_outputs"].append(historical_output)
    historical_id = source["historical_coverage_records"][0]["id"]
    overlap = deepcopy(source)
    overlap.setdefault("coverage_records", []).append(historical_id)
    wrong_standing = deepcopy(source)
    wrong_standing["historical_coverage_records"][0]["status"] = "passed"
    unowned = deepcopy(source)
    unowned["historical_coverage_records"] = []
    for label, mutant, expected in (
        ("historical/selected overlap", overlap, "overlap selected"),
        ("historical relabeling", wrong_standing, "failed output standing"),
        ("lost historical ownership", unowned, "unowned emitter"),
    ):
        try:
            per_output_inference([mutant])
        except ValueError as error:
            if expected not in str(error):
                raise AssertionError(f"{label}: wrong refusal {error}") from error
        else:
            raise AssertionError(f"{label}: invalid allocation accepted")
    measured_cells = [
        dict(zip(AXES, coordinate, strict=True), validation="none", calibration=["measured"])
        for coordinate in READY_COORDINATES
    ]
    original_cells = [
        {
            **measured_cells[index % len(measured_cells)],
            "calibration_reason": "estimator_grid_not_measured",
        }
        for index in range(39)
    ]
    assert len(reviewed_backlog(original_cells, [])) == 39
    try:
        reviewed_backlog(original_cells[:-1], [])
    except ValueError as error:
        assert "review plan" in str(error)
    else:
        raise AssertionError("unreviewed partial backlog accepted")
    assert reviewed_backlog(measured_cells, [{"id": "measured", "boundary": False}]) == []
    for invalid in ([], [{"id": "measured", "boundary": True}]):
        try:
            reviewed_backlog(measured_cells, invalid)
        except ValueError as error:
            assert "empty backlog lacks measured bindings" in str(error)
        else:
            raise AssertionError("empty backlog accepted missing or failed evidence")
    original = (
        "# Calibration backlog plan\n\nThese are 39 cells across 17 distinct coordinates.\n| original row |\n\n"
        + INFERENCE_HEADING
    )
    history = inherited_history(original)
    assert "| original row |" in history
    assert inherited_history(history + "\n\n" + INFERENCE_HEADING) == history
    print("calibration plan historical allocation and completed-backlog self-test: ok")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    content = render()
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != content:
            parser.error("parity/calibration_plan.md is stale")
    else:
        OUTPUT.write_text(content, encoding="utf-8", newline="\n")
    print(
        "calibration plan: current backlog and historical allocation retained; "
        "current promotion routes and inference boundaries inventoried"
    )


if __name__ == "__main__":
    main()
