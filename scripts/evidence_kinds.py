"""Shared evidence vocabulary; these categories are propositions, not a ranking.

Calibration is measured evidence with its own scope/commit attestation. No category
here, including agreement with an external package, grants a calibration license.
"""

import hashlib
import json
from pathlib import Path

EVIDENCE_KINDS = frozenset(
    {
        "implementation_exists",
        "internal_known_truth",
        "internal_cross_check",
        "frozen_external_oracle",
        "behavioral_parity",
        "contract_equivalence",
        "regression_pin",
        "process_attested",
    }
)
EXTERNAL_KINDS = frozenset({"frozen_external_oracle", "behavioral_parity"})
KNOWN_TRUTH_KINDS = EXTERNAL_KINDS | {"internal_known_truth"}
ORACLE_KINDS = frozenset(
    {
        "external_package",
        "closed_form",
        "enumeration",
        "independent_reimplementation",
        "regression_pin",
    }
)
SOFTWARE_KINDS = frozenset(
    {"implementation_exists", "internal_cross_check", "regression_pin"}
)


def evidence_roles(kind: str | None, activity: str | None = None) -> list[str]:
    """A scientific assertion is also a software regression check, never a license."""
    if activity == "statistical_smoke":
        return ["software", "statistical_smoke"]
    if kind == "process_attested":
        return ["process"]
    if kind in KNOWN_TRUTH_KINDS or kind == "contract_equivalence":
        return ["software", "scientific"]
    return ["software"]


def kind_problems(
    kind: str,
    oracle: dict | None,
    inference: str | None = None,
    fields: list[str] | None = None,
) -> list[str]:
    """Check claimed *origin*, leaving scientific sufficiency to review/execution."""
    if kind not in EVIDENCE_KINDS:
        return [f"unknown evidence_kind {kind!r}"]
    if kind not in KNOWN_TRUTH_KINDS:
        return []
    if not isinstance(oracle, dict) or oracle.get("kind") not in ORACLE_KINDS:
        return ["known-truth claim requires an oracle with a recognized origin"]
    origin = oracle["kind"]
    if "independent_fields" in oracle:
        independent = oracle["independent_fields"]
        if (
            not isinstance(independent, list)
            or not independent
            or not isinstance(fields, list)
            or not fields
            or any(
                not isinstance(field, str) or field not in independent
                for field in fields
            )
        ):
            return [
                "mixed fixture requires explicit evidence_fields restricted to its independent_fields"
            ]
    if kind in EXTERNAL_KINDS and origin != "external_package":
        return [f"{kind} requires external_package, found {origin}"]
    # A mixed fixture must be checked against the row's inference separately.
    if (
        kind == "internal_known_truth"
        and origin == "regression_pin"
        and (
            inference is None
            or inference not in oracle.get("independent_inferences", [])
        )
    ):
        return ["regression_pin has no independent reference for this inference"]
    return []


def fixture_kind_problems(root: Path, row: dict) -> list[str]:
    """Validate the same fixture origin/field scope in every consumer gate."""
    if row.get("evidence_kind") not in KNOWN_TRUTH_KINDS:
        return []
    fixture = row.get("known_truth_fixture")
    if not isinstance(fixture, str):
        return []  # Explicit inline references are validated separately.
    path = (root / fixture).resolve()
    if not path.is_relative_to(root.resolve()):
        return ["known_truth_fixture must remain inside the repository"]
    if path.suffix in {".rs", ".py"}:
        return []  # Inline truth is reviewed as source, not a JSON oracle.
    path = path / "expected.json" if path.is_dir() else path
    if not path.is_file():
        return [f"missing fixture: {fixture}"]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return [f"invalid fixture {fixture}: {error}"]
    if not isinstance(data, dict):
        return ["fixture must contain a JSON object"]
    if "evidence_roles" in data:
        roles = data["evidence_roles"]
        if (
            not isinstance(roles, list)
            or not roles
            or any(
                role not in ("software", "scientific", "statistical_smoke", "process")
                for role in roles
            )
        ):
            return ["invalid fixture evidence_roles"]
        if "scientific" not in roles:
            return ["software-only fixture cannot establish independent known truth"]
    return fixture_review_problems(root, data) + kind_problems(
        row["evidence_kind"],
        data.get("oracle"),
        row.get("inference"),
        row.get("evidence_fields"),
    )


def fixture_payload_digest(document: dict) -> str:
    payload = {**document}
    if isinstance(document.get("oracle"), dict):
        payload["oracle"] = {
            key: value
            for key, value in document["oracle"].items()
            if key != "reference_review"
        }
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def fixture_review_problems(root: Path, document: dict) -> list[str]:
    oracle = document.get("oracle", {})
    problems = reference_review_problems(root, oracle)
    if (
        isinstance(oracle, dict)
        and isinstance(oracle.get("reference_review"), dict)
        and oracle["reference_review"].get("fixture_payload_sha256")
        != fixture_payload_digest(document)
    ):
        problems.append(
            "reference_review stale fixture inputs, outputs, tolerances or origin metadata"
        )
    return problems


def reference_review_problems(root: Path, oracle: dict) -> list[str]:
    """Check a retained review's integrity, never infer independence from source scanning."""
    if not isinstance(oracle, dict):
        return ["oracle must be an object"]
    review = oracle.get("reference_review")
    if review is None:
        return []
    if not isinstance(review, dict):
        return ["reference_review must be an object"]
    problems = []
    for key in (
        "scope",
        "independence_basis",
        "historical_exposure",
        "redistribution",
        "limitations",
    ):
        if not isinstance(review.get(key), str) or not review[key].strip():
            problems.append(f"reference_review lacks {key}")
    artifacts = review.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return problems + ["reference_review requires reviewed artifacts"]
    roles = set()
    seen = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            problems.append("reference_review artifact must be an object")
            continue
        rel = artifact.get("path")
        role = artifact.get("role")
        if role not in (
            "generator",
            "environment_lock",
            "assertion_source",
            "comparison_helper",
            "input_fixture",
        ):
            problems.append("reference_review artifact has an unknown role")
        else:
            roles.add(role)
        if not isinstance(rel, str) or rel in seen:
            problems.append("reference_review requires unique artifact paths")
            continue
        seen.add(rel)
        path = (root / rel).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            problems.append(f"reference_review missing artifact: {rel}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != artifact.get("sha256"):
            problems.append(f"reference_review stale artifact: {rel}")
    if not {"generator", "environment_lock", "assertion_source"}.issubset(roles):
        problems.append(
            "reference_review must bind generator, environment lock and assertion source"
        )
    return problems


def inline_reference_problems(row: dict) -> list[str]:
    reference = row.get("known_truth_reference")
    if reference is None:
        return []
    if not isinstance(reference, dict):
        return ["known_truth_reference must be a table"]
    problems = []
    if reference.get("kind") not in {
        "closed_form",
        "enumeration",
        "independent_reimplementation",
    }:
        problems.append(
            "inline truth must identify an independent analytic/enumerated/reference origin"
        )
    expected = f"{row.get('evidence_test')}::{row.get('evidence_assertion')}"
    if reference.get("source") != expected:
        problems.append("inline truth source must name this exact executing assertion")
    for key in ("proposition", "tolerance", "limitations"):
        if not isinstance(reference.get(key), str) or not reference[key].strip():
            problems.append(f"inline truth {key} is required")
    return problems
