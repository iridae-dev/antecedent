#!/usr/bin/env python3
"""Inventory inherited calibration debt and the current 2.3 inference contracts.

This renders declarations, not measurement or a license. Per-output calibration
designs must still cover diagnostics inside point-only records.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "parity/support_licensed.toml"
OUTPUT = ROOT / "parity/2_3_calibration_plan.md"
PROMOTION = ROOT / "parity/promotion_2_3.toml"
LEDGER = ROOT / "parity/2_3_evidence_ledger.toml"
AXES = ("query", "graph_class", "structure", "inference")


def table_text(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def registered_inference() -> list[str]:
    records = tomllib.loads(PROMOTION.read_text())["record"]
    cells = tomllib.loads(LEDGER.read_text())["cell"]
    by_route = {(cell["id"], cell["route"]): cell for cell in cells}
    expected = {
        (record["id"], route["name"])
        for record in records
        for route in record["routes"]
    }
    if expected != set(by_route) or len(by_route) != len(cells):
        raise ValueError(
            "2.3 inference inventory requires an exact route-to-evidence ledger"
        )
    lines = [
        "## Current 2.3 registered inference contracts",
        "",
        (
            f"The following inventory covers all {len(records)} promotion records and "
            f"{len(expected)} registered routes, including closed producers and independent "
            "consumers. It is generated from the promotion registry and evidence ledger. "
            "A point-only status can still carry unmeasured standard errors, p-values, "
            "posterior summaries or Monte Carlo diagnostics. The inherited 40-cell backlog "
            "above does not allocate designs for those outputs. Neither this table nor an "
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
            calibration = (
                ", ".join(record.get("coverage_records", [])) or cell["calibration"]
            )
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
            "Source: `promotion_2_3.toml` and `2_3_evidence_ledger.toml`. "
            "Regenerate after changing a public route, uncertainty method or consumer."
        ),
        "",
    ]
    return lines


def disposition(row: dict) -> tuple[str, str]:
    if row["structure"] == "explicit":
        if (row["query"], row["graph_class"], row["inference"]) != (
            "NestedCounterfactualEffect",
            "Dag",
            "Frequentist",
        ):
            raise ValueError("unreviewed fixed-graph backlog coordinate")
        return (
            "point_only_no_interval",
            (
                "The fixed Markovian natural-direct-effect route publishes a point only. "
                "Build a separate whole-estimator interval and repeated-row known-SCM grid "
                "before claiming sampling coverage; retain its current point truth fixture."
            ),
        )
    if row["structure"] != "graph_posterior":
        raise ValueError("unreviewed backlog structure")
    quantity = row["query"]
    if quantity in {"ResponseCurve", "InterventionResponse"}:
        oracle = "a finite dose/horizon grid with per-atom truth and a declared pointwise or simultaneous target"
    elif quantity == "ConditionalEffect":
        oracle = "a finite conditional-stratum SCM with support and per-atom conditional-effect truth"
    else:
        oracle = "an enumerated graph/SCM family with per-atom ATE and unidentified-mass truth"
    if row["inference"] == "Bayesian":
        calibration = (
            "posterior predictive/decision calibration for the actual mixed-atom claim"
        )
    else:
        calibration = "whole-method repeated sampling, including the graph-weight and data dependence rule"
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
        for output in outputs:
            missing = REQUIRED_INFERENCE_OUTPUT_FIELDS - set(output)
            if missing or output.get("id") in seen:
                raise ValueError(
                    f"{record['id']}: invalid output declaration {sorted(missing)}"
                )
            seen.add(output["id"])
            if output["implementation_status"] not in INFERENCE_IMPLEMENTATIONS:
                raise ValueError(
                    f"{record['id']}: unknown output implementation status"
                )
            if output["owner"] not in record["owners"]:
                raise ValueError(f"{record['id']}: output has an unregistered owner")
            if not set(output["allocated_coverage_records"]) <= allocated:
                raise ValueError(
                    f"{record['id']}: output allocates an unowned emitter ID"
                )
            suite = output["measurement_suite"]
            if (
                suite != "unallocated"
                and not (ROOT / suite.split("::", 1)[0]).is_file()
            ):
                raise ValueError(
                    f"{record['id']}: nonexistent measurement suite {suite}"
                )
            if (
                output["implementation_status"] == "unimplemented_closed"
                and suite != "unallocated"
            ):
                raise ValueError(
                    f"{record['id']}: missing method cannot have an executing suite"
                )
            families.append((record, output))
    required = sum(
        output["allocation_status"]
        == "method_specific_harness_required_before_final_measurement"
        for _, output in families
    )
    prerequisites = sum(
        output["allocation_status"]
        == "method_prerequisite_missing_calibration_cannot_close"
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
            "This is explicit draft ownership and design allocation, not measurement or "
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
    cells = [
        row
        for row in tomllib.loads(SOURCE.read_text())["cell"]
        if row.get("calibration_reason") == "estimator_grid_not_measured"
    ]
    counts = Counter(tuple(row[a] for a in AXES) for row in cells)
    if len(cells) != 40 or len(counts) != 18:
        raise ValueError(
            f"backlog changed to {len(cells)} cells / {len(counts)} coordinates; review plan"
        )
    cells.sort(
        key=lambda row: (
            row["structure"] != "explicit",
            *(row[a] for a in AXES),
            row.get("validation", ""),
        )
    )
    lines = [
        "# 2.3 X11 calibration backlog plan",
        "",
        "Generated from `support_licensed.toml` without changing its `estimator_grid_not_measured` markers. These are 40 cells across 18 distinct coordinates. The fixed-DAG point route is reviewed first; graph-posterior rows remain pending. A typed reason below explains why the missing calibration does not license an inferential claim. It is not a waiver or a coverage record.",
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
    lines += registered_inference()
    lines += per_output_inference(tomllib.loads(PROMOTION.read_text())["record"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = render()
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text() != content:
            parser.error("2_3_calibration_plan.md is stale")
    else:
        OUTPUT.write_text(content)
    print(
        "2.3 calibration plan: inherited 40 cells / 18 coordinates retained; "
        "current promotion routes and inference boundaries inventoried"
    )


if __name__ == "__main__":
    main()
