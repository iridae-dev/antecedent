"""Validate source-audited manual interval recipes against original recordless receipts.

These are gate receipts, not coverage-record projections. Legacy measurements retain
legacy semantics; a frozen-family receipt is independently revalidated as such.
"""

from __future__ import annotations

import csv
import functools
import hashlib
import io
import json
import math
import re
import tomllib
from pathlib import Path
from typing import Any

import _rust_source as rust
import calibration_family as family
import collect_coverage_records as collector

REGISTRY = "parity/calibration_recipe_receipts.json"
Z95 = 1.959_963_984_540_054
WIDTH_DOMAIN = "release_2_4_1_missing_width_v1"
APPROVED_WIDTH_PLAN = "4a70af6a1e18f7024a7af2beffa8489bbf16766c8bf4521f4f1dbd3616c3d65e"
APPROVED_WIDTH_HELPER = "7d11a2a2d88e4fe730ca285ec9fbf71a73c9d7fa5c014bccc8221e7f7e7fe258"
APPROVED_NUMERICAL_AUDIT = "7cfbdf1ac9c2f2ac777c7fedc5f349bae6f293b99c30f4417da6257f652829f1"
SOURCE_PATH = "crates/antecedent-estimate/tests/observational_effects_calibration.rs"
DGP_PATH = "crates/antecedent-estimate/tests/common/observational_effects.rs"
COMMON_PATH = "crates/antecedent/tests/common/calibration.rs"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_asset(root: Path, descriptor: dict[str, str]) -> bytes:
    if set(descriptor) != {"path", "sha256"}:
        raise ValueError("recipe asset descriptor has unknown or missing fields")
    path = Path(descriptor["path"])
    if (
        path.is_absolute()
        or ".." in path.parts
        or not str(path).startswith("parity/recipe_evidence/")
    ):
        raise ValueError("recipe evidence must be a durable repository-relative asset")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to((root / "parity/recipe_evidence").resolve()):
        raise ValueError("recipe evidence symlink escapes its durable directory")
    data = resolved.read_bytes()
    if sha256(data) != descriptor["sha256"]:
        raise ValueError(f"recipe evidence hash mismatch: {path}")
    return data


@functools.cache
def source_blob(root: Path, path: str, sha: str) -> bytes:
    # Only immutable Git blobs are cached. Current files are always reread.
    return family.read_source(root, path, sha)


def function_bytes(data: bytes, name: str) -> bytes:
    text = data.decode("utf-8")
    skeleton = rust.lex(text).skeleton
    found = re.search(rf"\bfn {re.escape(name)}\s*\(", skeleton)
    if found is None:
        raise ValueError("audited numerical function is missing")
    start = found.start()
    cursor = skeleton.index("{", found.end()) + 1
    depth = 1
    while depth:
        depth += (skeleton[cursor] == "{") - (skeleton[cursor] == "}")
        cursor += 1
    return text[start:cursor].encode("utf-8")


def validate_numerical_source(root: Path, recipe: dict[str, Any]) -> None:
    proof_bytes = read_asset(root, recipe["numerical_source_audit"])
    if sha256(proof_bytes) != APPROVED_NUMERICAL_AUDIT:
        raise ValueError("unreviewed numerical dependency footprint")
    proof = json.loads(proof_bytes)
    for item, row in proof["same_source_files_and_functions"].items():
        path, separator, name = item.partition("::")
        measured = source_blob(root, path, recipe["calibration_sha"])
        current = (root / path).read_bytes()
        if separator:
            measured, current = function_bytes(measured, name), function_bytes(current, name)
        if sha256(measured) != row["c30_sha256"] or sha256(current) != row["c30_sha256"]:
            raise ValueError(f"numerical recipe dependency changed: {item}")
    original = tomllib.loads(
        source_blob(root, "Cargo.lock", recipe["calibration_sha"]).decode("utf-8")
    )
    current = tomllib.loads((root / "Cargo.lock").read_text(encoding="utf-8"))
    original_packages = [p for p in original["package"] if not p["name"].startswith("antecedent")]
    current_packages = [p for p in current["package"] if not p["name"].startswith("antecedent")]
    if original_packages != current_packages:
        raise ValueError("numerical recipe external dependency lock changed")


def upper_median(values: list[float]) -> float:
    if not values:
        raise ValueError("no finite reported widths")
    return sorted(values)[len(values) // 2]


def width_supplement(
    root: Path, spec: dict[str, Any], targets: dict[str, Any], recipe: dict[str, Any]
) -> float:
    """Recompute the predeclared width-only measurement; never infer coverage."""
    plan_bytes = read_asset(root, spec["plan"])
    plan = json.loads(plan_bytes)
    compiled = json.loads(read_asset(root, spec["compiled_artifact"]))
    if (
        sha256(plan_bytes) != APPROVED_WIDTH_PLAN
        or plan["helper_sha256"] != APPROVED_WIDTH_HELPER
        or plan["source_commit"] != recipe["calibration_sha"]
        or plan["harness"]["sha256"] != recipe["source_files"][SOURCE_PATH]
        or plan["status"] != "SEALED_PLAN_NO_SIMULATIONS"
        or plan["domain"] != WIDTH_DOMAIN
        or plan["attempts_per_case"] != 2048
        or plan["participants"] != 1000
        or plan["worker_count"] != 1
        or compiled["plan_sha256"] != sha256(plan_bytes)
    ):
        raise ValueError("width supplement differs from its predeclared plan")
    if sha256(read_asset(root, spec["helper"])) != plan["helper_sha256"]:
        raise ValueError("width supplement helper differs from sealed plan")
    if sha256(read_asset(root, spec["dgp"])) != plan["dgp"]["sha256"]:
        raise ValueError("width supplement generating source differs from plan")
    if sha256(read_asset(root, spec["runner"])) != plan["orchestration_sha256"]:
        raise ValueError("width supplement runner differs from sealed plan")
    if (
        plan["preoutcome_amendment"]["median_definition"]
        != "sort values; return values[len/2], including even sample sizes"
    ):
        raise ValueError("width supplement uses another median definition")
    salt = int.from_bytes(hashlib.sha256(WIDTH_DOMAIN.encode()).digest()[:8], "little")
    if int(plan["seed_salt_hex"], 16) != salt:
        raise ValueError("width supplement seed domain mismatch")
    case = spec["case"]
    bases = {"policy": 0x6B0000, "survival": 0x5E0000}
    if case not in bases:
        raise ValueError("unknown width-only case")
    raw = read_asset(root, spec["raw"])
    summary_bytes = read_asset(root, spec["summary"])
    summary = json.loads(summary_bytes)
    results = json.loads(read_asset(root, spec["results"]))
    matching = [r for r in results["results"] if r["case"] == case]
    if (
        results["plan_sha256"] != sha256(plan_bytes)
        or results["compiled_artifact"] != compiled
        or results["coverage_computed"] is not False
        or results["coverage_records_changed"] is not False
        or len(matching) != 1
        or matching[0]["exit_code"] != 0
        or matching[0]["raw_sha256"] != sha256(raw)
        or matching[0]["summary_sha256"] != sha256(summary_bytes)
        or summary["coverage_computed"] is not False
    ):
        raise ValueError("width raw outputs differ from their original completed receipt")
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
    if len(rows) != 2048:
        raise ValueError("width supplement must retain all2048 attempted rows")
    widths, truths = [], []
    for i, row in enumerate(rows):
        if (
            row["case"] != case
            or int(row["replicate"]) != i
            or int(row["seed"]) != ((bases[case] + i) ^ salt)
        ):
            raise ValueError("width supplement row or seed identity mismatch")
        if row["status"] == "refused":
            if row["width"] or row["abs_truth"]:
                raise ValueError("a refused width row carries an interval")
            continue
        if row["status"] != "reported":
            raise ValueError("unknown width reporting status")
        width, truth = float(row["width"]), float(row["abs_truth"])
        if not (math.isfinite(width) and width >= 0 and math.isfinite(truth) and truth > 0):
            raise ValueError("nonfinite width or nonpositive truth magnitude")
        widths.append(width)
        truths.append(truth)
    if len(widths) < 2000 or len(widths) / 2048 < targets["min_reporting_rate"]:
        raise ValueError("width supplement reporting/count target is not met")
    med_width, med_truth = upper_median(widths), upper_median(truths)
    ratio = upper_median([w / t for w, t in zip(widths, truths, strict=True)])
    if (
        summary["reported"] != len(widths)
        or summary["attempts"] != 2048
        or summary["case"] != case
        or not math.isclose(summary["median_width"], med_width, rel_tol=1e-12)
        or not math.isclose(summary["median_abs_truth"], med_truth, rel_tol=1e-12)
        or not math.isclose(
            summary["median_per_replicate_width_over_abs_truth"], ratio, rel_tol=1e-12
        )
    ):
        raise ValueError("saved width summary disagrees with the complete raw rows")
    return max(med_width / med_truth, ratio)


def truth_lower_bound(kind: str) -> float:
    if kind == "normal_threshold_policy_value":
        return 1 - 1.5 / math.sqrt(2 * math.pi)
    if kind == "six_cell_exponential_survival_difference":
        survival = []
        for arm in (0, 1):
            survival.append(
                sum(
                    math.exp(-0.25 * math.exp(-0.5 * arm + 0.7 * z1 + 0.5 * z2) * 2) / 6
                    for z1 in (0, 1)
                    for z2 in (-1, 0, 1)
                )
            )
        return survival[1] - survival[0]
    raise ValueError("unreviewed source truth bound")


def validate_recipe(
    root: Path, recipe: dict[str, Any], case: dict[str, Any], size: int | None
) -> dict[str, Any]:
    required = {
        "case_id",
        "evidence_kind",
        "coverage_semantics",
        "test",
        "sample_size",
        "nominal",
        "normal_critical_value",
        "calibration_sha",
        "group",
        "source_files",
        "gate_receipt",
        "coverage_log",
        "tally",
        "source_review",
        "numerical_source_audit",
        "width_evidence",
    }
    if set(recipe) != required:
        raise ValueError("unknown or missing recordless recipe fields")
    if (
        recipe["evidence_kind"] != "recordless_gate_manual_recipe"
        or recipe["coverage_semantics"] != "whole_procedure_refusals_are_misses"
        or recipe["case_id"] != case["id"]
        or recipe["test"] != case["evidence"]["test"]
        or recipe["sample_size"] != size
        or recipe["nominal"] != case["targets"]["nominal_coverage"]
        or recipe["normal_critical_value"] != Z95
        or recipe["nominal"] != 0.95
    ):
        raise ValueError("recordless recipe does not match the exact case/construction/size")
    source = recipe["calibration_sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", source):
        raise ValueError("recipe has no exact source identity")
    for path in (SOURCE_PATH, DGP_PATH, COMMON_PATH):
        if sha256(family.read_source(root, path, source)) != recipe["source_files"][path]:
            raise ValueError(f"recipe source audit mismatch: {path}")
    validate_numerical_source(root, recipe)
    gate = json.loads(read_asset(root, recipe["gate_receipt"]))
    if gate["calibration_sha"] != source or gate["group"] != recipe["group"]:
        raise ValueError("recordless receipt owner/source mismatch")
    log = read_asset(root, recipe["coverage_log"])
    # Keep the original basename: frozen-family auditing derives its group from it.
    log_path = root / recipe["coverage_log"]["path"]
    try:
        original = collector.gate_ledger([log_path], source)
    except SystemExit as error:
        raise ValueError(f"invalid original recordless decision log: {error}") from error
    if original != {recipe["group"]: gate}:
        raise ValueError("stored original gate receipt differs from its preserved raw log")
    if set(re.findall(rb"^calibration-measured-at ([0-9a-f]{40})$", log, re.M)) != {
        source.encode()
    }:
        raise ValueError("recordless coverage log source stamp mismatch")
    if b"calibration-record " in log or not re.search(
        rb"test result: ok\. [1-9][0-9]* passed; 0 failed;", log
    ):
        raise ValueError("recipe requires an original passed recordless gate")
    state_lines = read_asset(root, recipe["tally"]).decode("utf-8").splitlines()
    states = [json.loads(line) for line in state_lines]
    name = recipe["test"].rsplit("::", 1)[1]
    states = [s for s in states if s["key"] == f"{name}::{name}@0.950000"]
    if len(states) != 1:
        raise ValueError("recordless recipe tally identity is missing or ambiguous")
    state = states[0]
    if state["record"] is not None or state["level"] != recipe["nominal"]:
        raise ValueError("a recordless tally was relabelled as a typed record")
    values = [state[k] for k in ("covered", "scored", "skipped", "with_interval")]
    if any(type(v) is not int or v < 0 for v in values):
        raise ValueError("invalid raw coverage/reporting counts")
    covered, scored, skipped, reported = values
    total = scored + skipped
    if not 0 <= covered <= reported <= scored or total <= 0 or state["map_end"] != total:
        raise ValueError("inconsistent raw coverage/reporting denominator")
    length = state["length_sum"]
    if type(length) not in (int, float) or not math.isfinite(length) or length < 0 or reported == 0:
        raise ValueError("invalid saved interval lengths")
    if "decision_protocol" in gate:
        # The collector already revalidates the original full-family config, raw arithmetic,
        # superseded looks and actual nongrid execution point. Match its final cell to the tally.
        decisions = [
            r for r in gate["coverage_decisions"] if r["test"] == name and r["name"] == name
        ]
        if not decisions:
            raise ValueError("frozen-family recipe lacks its exact decision cell")
        final = max(decisions, key=lambda r: r["decision_protocol"]["planned_replicates"])
        decision = final["decision_protocol"]
        if (
            decision["covered"] != covered
            or decision["scored"] != total
            or not decision["compatible"]
            or state.get("decision_context")
            != {
                "config": gate["decision_protocol"],
                "measured_at": source,
                "grid_point": final["grid_point"],
            }
        ):
            raise ValueError("frozen-family decision differs from raw tally")
    else:
        # Legacy receipts have no retrospective FWER metadata. Recover exact counts from
        # the original human tally line, including refusals counted as observed misses.
        pattern = rf"calibration {re.escape(name)}:.*mean_length=([0-9.eE+-]+) \({covered}/{total} covered, {skipped} skipped\)"
        match = re.search(pattern.encode(), log)
        if not match:
            raise ValueError("legacy gate log and saved raw tally counts disagree")
        if not math.isclose(float(match[1]), length / reported, rel_tol=0, abs_tol=0.000_050_001):
            raise ValueError("legacy logged mean width disagrees with saved raw tally")
        if state.get("decision_context") is not None:
            raise ValueError("legacy receipt cannot borrow a future decision protocol")
    targets = case["targets"]
    if total < targets["min_replicates"] or covered / total < targets["coverage_floor"]:
        raise ValueError("recordless actual outcomes do not meet count/coverage targets")
    if reported / total < targets["min_reporting_rate"]:
        raise ValueError("recordless actual reporting rate is below its frozen floor")
    width = recipe["width_evidence"]
    if width["kind"] == "saved_mean_positive_truth_bound":
        expected = (
            "normal_threshold_policy_value"
            if name.startswith("policy_")
            else "six_cell_exponential_survival_difference"
        )
        if width["truth_model"] != expected or not name.startswith(
            ("policy_", "survival_difference_")
        ):
            raise ValueError("saved width bound uses another generating truth")
        bound = 2 * length / reported / truth_lower_bound(width["truth_model"])
    elif width["kind"] == "prospective_width_only_supplement":
        if size != 1000 or width["case"] != ("policy" if "policy_" in name else "survival"):
            raise ValueError("width supplement is bound to another source case or size")
        bound = width_supplement(root, width, targets, recipe)
    else:
        raise ValueError("missing source-audited width evidence")
    if bound > targets["max_median_width_over_effect"]:
        raise ValueError("recordless width evidence does not prove the frozen target")
    return {
        "outcomes": total,
        "reported": reported,
        "covered": covered,
        "coverage": covered / total,
        "reporting_rate": reported / total,
        "nominal_mcse": math.sqrt(recipe["nominal"] * (1 - recipe["nominal"]) / total),
        "median_width_over_effect_bound": bound,
        "receipt_kind": "recordless_gate",
    }


def validate_asserted_knn(
    root: Path, recipe: dict[str, Any], case: dict[str, Any], size: int | None
) -> dict[str, Any]:
    """Preserve the original KNN gate, including its exact printed width/reporting assertions."""
    required = {
        "case_id",
        "evidence_kind",
        "profile",
        "test",
        "sample_size",
        "size_basis",
        "execution_grid_point",
        "nominal",
        "effect",
        "covariates",
        "neighbors",
        "calibration_sha",
        "group",
        "source_files",
        "gate_receipt",
        "coverage_log",
        "tally",
        "source_review",
    }
    if set(recipe) != required or recipe["profile"] != "knn_population_linear.v1":
        raise ValueError("unknown source-asserted calibration profile")
    path = "crates/antecedent-estimate/tests/knn_matching_calibration.rs"
    test = "knn_matching_interval_ci_coverage"
    if (
        recipe["case_id"] != case["id"]
        or recipe["test"] != case["evidence"]["test"]
        or recipe["test"] != path + "::" + test
        or recipe["sample_size"] != size
        or recipe["size_basis"] != "generated_rows"
        or size not in (10000, 100000)
        or recipe["execution_grid_point"] != {10000: 0, 100000: 2}[size]
        or recipe["nominal"] != 0.95
        or recipe["nominal"] != case["targets"]["nominal_coverage"]
        or recipe["effect"] != 0.5
        or recipe["covariates"] != 8
        or recipe["neighbors"] != 3
        or case["configuration"].get("covariates") != 8
        or case["configuration"].get("k") != 3
        or case["expected"]["kind"] != "calibrated_interval"
    ):
        raise ValueError("KNN receipt does not match its exact source construction/size")
    if recipe["group"] != f"knn_matching_calibration__{test}.p{recipe['execution_grid_point']}":
        raise ValueError("KNN original group/grid identity mismatch")
    source = recipe["calibration_sha"]
    for rel in (path, COMMON_PATH):
        measured = source_blob(root, rel, source)
        if sha256(measured) != recipe["source_files"][rel]:
            raise ValueError("KNN source audit mismatch")
        if rel == path and (root / rel).read_bytes() != measured:
            raise ValueError("KNN source recipe changed since measurement")
    log = read_asset(root, recipe["coverage_log"])
    gate = json.loads(read_asset(root, recipe["gate_receipt"]))
    log_path = root / recipe["coverage_log"]["path"]
    if (
        gate["group"] != recipe["group"]
        or gate["calibration_sha"] != source
        or gate.get("protocol") != "population_linear"
        or collector.gate_ledger([log_path], source) != {recipe["group"]: gate}
    ):
        raise ValueError("KNN original passed gate/protocol mismatch")
    # The ordinary attestation gate owns current implementation compatibility.
    # Require this very original row to remain in its collected ledger; it will
    # acquire the reviewed KNN + shared-kernel + suite dependency facets.
    rows = tomllib.loads((root / "parity/calibration_gates.toml").read_text(encoding="utf-8"))[
        "gate"
    ]
    if [r for r in rows if r["group"] == gate["group"]] != [gate]:
        raise ValueError("KNN receipt is not bound to its currently attested original gate row")
    if (
        set(re.findall(rb"^calibration-measured-at ([0-9a-f]{40})$", log, re.M))
        != {source.encode()}
        or not re.search(rb"test result: ok\. [1-9][0-9]* passed; 0 failed;", log)
        or b"calibration-record " in log
    ):
        raise ValueError("KNN raw recordless receipt did not pass at its stated source")
    states = [
        json.loads(line) for line in read_asset(root, recipe["tally"]).decode("utf-8").splitlines()
    ]
    wanted = f"{test}::knn_matching_interval@0.950000"
    states = [row for row in states if row["key"] == wanted]
    if len(states) != 1:
        raise ValueError("KNN raw tally identity is missing or ambiguous")
    state = states[0]
    fields = [state[k] for k in ("scored", "skipped", "covered", "with_interval")]
    if any(type(x) is not int or x < 0 for x in fields):
        raise ValueError("invalid KNN actual counters")
    scored, skipped, covered, reported = fields
    total = scored + skipped
    if (
        state["record"] is not None
        or state["level"] != 0.95
        or not 0 <= covered <= reported <= scored
        or total != state["map_end"]
        or state.get("decision_context") is not None
    ):
        raise ValueError("invalid KNN legacy tally or retrospective protocol")
    match = re.search(
        rf"knn matching interval: rows {size}, reported (\d+) of (\d+), median width ([0-9.eE+-]+)".encode(),
        log,
    )
    if match is None or (int(match[1]), int(match[2])) != (reported, total):
        raise ValueError("KNN log size/reporting counters disagree")
    if not re.search(
        rf"calibration knn_matching_interval: nominal=0.95.*\({covered}/{total} covered, {skipped} skipped\)".encode(),
        log,
    ):
        raise ValueError("KNN original covered counts disagree")
    # Log formatting rounds the median to4decimal places. Use its upper bound,
    # while the original source independently asserts the unrounded target.
    width_ratio_bound = (float(match[3]) + 0.000_050_001) / recipe["effect"]
    targets = case["targets"]
    if (
        total < targets["min_replicates"]
        or reported < targets["min_replicates"]
        or covered / total < targets["coverage_floor"]
        or reported / total < targets["min_reporting_rate"]
        or width_ratio_bound > targets["max_median_width_over_effect"]
    ):
        raise ValueError("KNN exact count/coverage/reporting/width targets are not met")
    return {
        "receipt_kind": "recordless_gate",
        "outcomes": total,
        "reported": reported,
        "covered": covered,
        "coverage": covered / total,
        "median_width_over_effect_bound": width_ratio_bound,
    }


APPROVED_COUNT_PLAN = "098e843845449fc32c8ed927458b052175d2a2be8ecd226c5dfe158c107b619c"
COUNT_ORIGINAL_SOURCE = "c30e10304f4bd85a01ad81dd133a360ddc95c2f1"
COUNT_WITNESS_SOURCE = "767659966d856f0bb882196b5ff9e1aeb47e0a04"


@functools.cache
def count_observer_proof(path: str) -> str | None:
    """The reviewed terminal print changes no original primitive computation."""
    import _rust_equivalence as equivalence
    import calibration_facets as facets

    repo = facets.Repo()
    old = repo.rust_sources(COUNT_ORIGINAL_SOURCE)
    new = repo.rust_sources(COUNT_WITNESS_SOURCE)
    return equivalence.prove("rust_terminal_test_observer", path, old[path], new[path], old, new)


def validate_replayed_assertion(
    root: Path, recipe: dict[str, Any], case: dict[str, Any], size: int | None
) -> dict[str, Any]:
    """Supplement the original gate receipt with genuine same-protocol printed counts.

    This preserves the original receipt kind and legacy scientific decision. It is
    neither a new coverage record nor a retrospective family-wise decision.
    """
    required = {
        "case_id",
        "evidence_kind",
        "profile",
        "test",
        "sample_size",
        "nominal",
        "calibration_sha",
        "witness_sha",
        "group",
        "source_path",
        "gate_receipt",
        "witness_plan",
        "witness_log",
        "witness_receipt",
        "source_review",
    }
    profiles = {
        "gpdc_nystrom_conditional_null.v1": (
            "antecedent-stats",
            "crates/antecedent-stats/src/ci/calibration.rs",
            8000,
            "scalable_gpdc_replacement_validated_separately",
            "calibrated_statistical_test",
        ),
        "neighborhood_tipping_bracket.v1": (
            "antecedent-validate",
            "crates/antecedent-validate/src/sensitivity.rs",
            50000,
            "scalable_sensitivity_replacement_validated_separately",
            "calibrated_interval",
        ),
    }
    if (
        set(recipe) != required
        or recipe.get("profile") not in profiles
        or recipe.get("evidence_kind") != "replayed_source_asserted_recordless_gate"
    ):
        raise ValueError("unknown original assertion receipt profile")
    package, path, rows, test, kind = profiles[recipe["profile"]]
    if (
        recipe["case_id"] != case["id"]
        or case["id"] != test
        or recipe["test"] != case["evidence"]["test"]
        or recipe["test"] != path + "::" + test
        or recipe["source_path"] != path
        or recipe["group"] != package + "__" + test
        or recipe["sample_size"] != rows
        or size != rows
        or recipe["nominal"] != 0.95
        or case["targets"]["nominal_coverage"] != 0.95
        or recipe["calibration_sha"] != COUNT_ORIGINAL_SOURCE
        or recipe["witness_sha"] != COUNT_WITNESS_SOURCE
        or case["expected"]["kind"] != kind
    ):
        raise ValueError("original assertion receipt does not match its exact source/size/role")
    plan_bytes = read_asset(root, recipe["witness_plan"])
    if sha256(plan_bytes) != APPROVED_COUNT_PLAN:
        raise ValueError("count recovery lacks the reviewed sealed pre-outcome plan")
    plan = json.loads(plan_bytes)
    if plan["source"] != COUNT_WITNESS_SOURCE or plan["original_source"] != COUNT_ORIGINAL_SOURCE:
        raise ValueError("count recovery plan has another source/protocol")
    spec = [s for s in plan["cases"] if s["group"] == recipe["group"]]
    if len(spec) != 1 or spec[0]["profile"] != recipe["profile"] or spec[0]["rows"] != rows:
        raise ValueError("count recovery plan has another scientific recipe")
    expected_seed = spec[0].get("seed", spec[0].get("seed_start"))
    if (
        case["configuration"].get("rows") != rows
        or case["configuration"].get("replicates") != spec[0]["attempted"]
        or case.get("dataset", {}).get("seed") != expected_seed
        or case.get("dataset", {}).get("n") != rows
    ):
        raise ValueError("acceptance configuration does not match the original fixed source recipe")
    if count_observer_proof(path) is not None:
        raise ValueError("original primitive numerical protocol does not match witness source")
    if (root / path).read_bytes() != source_blob(root, path, COUNT_WITNESS_SOURCE):
        raise ValueError("current original assertion recipe changed after count recovery")
    gate = json.loads(read_asset(root, recipe["gate_receipt"]))
    if gate != {"group": recipe["group"], "passed": 1, "calibration_sha": COUNT_ORIGINAL_SOURCE}:
        raise ValueError("count recovery did not retain the original passed gate receipt")
    current_gates = tomllib.loads(
        (root / "parity/calibration_gates.toml").read_text(encoding="utf-8")
    )["gate"]
    if [g for g in current_gates if g["group"] == recipe["group"]] != [gate]:
        raise ValueError("original assertion receipt is not the currently attested gate row")
    log = read_asset(root, recipe["witness_log"])
    if (
        not re.search(rb"test result: ok\. 1 passed; 0 failed;", log)
        or ("test " + spec[0]["test"] + " ...").encode() not in log
    ):
        raise ValueError("count recovery did not execute its exact original scientific test")
    captures = re.findall(rb"calibration-assertion-receipt (\{[^\n]+\})", log)
    if len(captures) != 1:
        raise ValueError("count recovery has missing or ambiguous actual counters")
    counts = json.loads(captures[0])
    witness = json.loads(read_asset(root, recipe["witness_receipt"]))
    if (
        witness.get("schema") != "original_recordless_count_recovery_receipt.v1"
        or witness.get("source") != COUNT_WITNESS_SOURCE
        or witness.get("original_source") != COUNT_ORIGINAL_SOURCE
        or witness.get("plan_sha256") != APPROVED_COUNT_PLAN
        or witness.get("group") != recipe["group"]
        or witness.get("test") != spec[0]["test"]
        or witness.get("binary_sha256") != spec[0]["binary_sha256"]
        or witness.get("raw_log_sha256") != sha256(log)
        or witness.get("rust_harness_passed") is not True
        or witness.get("process_exit_code_observed") not in (None, 0)
        or witness.get("new_calibration_claim") is not False
    ):
        raise ValueError("count recovery receipt does not bind the original plan/raw execution")
    if witness.get("receipt") != counts:
        raise ValueError("recovered count receipt does not reproduce original raw bytes")
    if (
        counts.get("profile") != recipe["profile"]
        or counts.get("rows") != rows
        or counts.get("attempted") != spec[0]["attempted"]
    ):
        raise ValueError("count recovery has another source construction or attempted prefix")
    attempted, reported = counts["attempted"], counts.get("reported")
    if type(attempted) is not int or type(reported) is not int or not 0 < reported <= attempted:
        raise ValueError("invalid original reported/attempted counts")
    targets = case["targets"]
    if reported < targets["min_replicates"]:
        raise ValueError("actual conditional reported count is below the scored minimum")
    if reported / attempted < targets["min_reporting_rate"]:
        raise ValueError("original assertion reporting floor is not met")
    if kind == "calibrated_statistical_test":
        expected = {
            "profile",
            "rows",
            "attempted",
            "reported",
            "rejected",
            "alpha",
            "conditional_nonrejection",
            "width",
        }
        rejected = counts.get("rejected")
        if (
            set(counts) != expected
            or type(rejected) is not int
            or not 0 <= rejected <= reported
            or counts["alpha"] != 0.05
            or targets.get("type_i_alpha") != 0.05
            or case["configuration"].get("calibration_target") != "conditional_null_rejection_rate"
            or counts["width"] != "not_applicable"
            or targets.get("precision_target") != "not_applicable"
            or any(k.startswith("max_median") for k in targets)
        ):
            raise ValueError("GPDC receipt does not describe its statistical-test target")
        coverage = 1 - rejected / reported
        if not math.isclose(counts["conditional_nonrejection"], coverage, abs_tol=2e-16):
            raise ValueError("GPDC actual rejection and conditional counts disagree")
        covered = reported - rejected
    else:
        expected = {
            "profile",
            "rows",
            "attempted",
            "reported",
            "covered",
            "conditional_coverage",
            "median_width",
            "effect",
        }
        covered = counts.get("covered")
        if (
            set(counts) != expected
            or type(covered) is not int
            or not 0 <= covered <= reported
            or counts["effect"] != 0.5
            or not math.isfinite(counts["median_width"])
            or counts["median_width"] < 0
            or counts["median_width"] / 0.5 > targets["max_median_width_over_effect"]
        ):
            raise ValueError("sensitivity receipt does not meet its actual bracket width target")
        coverage = covered / reported
        if not math.isclose(counts["conditional_coverage"], coverage, abs_tol=2e-16):
            raise ValueError("sensitivity actual covered and reported counts disagree")
    if coverage < targets["coverage_floor"]:
        raise ValueError("original assertion conditional coverage floor is not met")
    return {
        "receipt_kind": "recordless_gate",
        "outcomes": attempted,
        "reported": reported,
        "covered": covered,
        "coverage": coverage,
        "coverage_semantics": "conditional_reported_original_assertion",
        "witness_source": COUNT_WITNESS_SOURCE,
        "original_source": COUNT_ORIGINAL_SOURCE,
    }


def validate_registry(
    root: Path, cases: list[dict[str, Any]], size_of: Any
) -> tuple[set[str], list[str]]:
    path = root / REGISTRY
    if not path.exists():
        return set(), []
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or set(data) != {"schema_version", "recipe"}:
        return set(), ["recordless recipe registry has an unknown schema"]
    by_id = {c["id"]: c for c in cases}
    valid, problems, seen = set(), [], set()
    for recipe in data["recipe"]:
        cid = recipe.get("case_id", "<missing>")
        if cid in seen or cid not in by_id:
            problems.append(f"{cid}: duplicate or unknown recordless recipe case")
            continue
        seen.add(cid)
        try:
            if recipe.get("evidence_kind") == "replayed_source_asserted_recordless_gate":
                validate_replayed_assertion(root, recipe, by_id[cid], size_of(by_id[cid]))
            elif recipe.get("evidence_kind") == "source_asserted_recordless_gate":
                validate_asserted_knn(root, recipe, by_id[cid], size_of(by_id[cid]))
            else:
                validate_recipe(root, recipe, by_id[cid], size_of(by_id[cid]))
            valid.add(cid)
        except (ValueError, KeyError, TypeError, OSError, ArithmeticError) as error:
            problems.append(f"{cid}: invalid recordless recipe receipt: {error}")
    return valid, problems
