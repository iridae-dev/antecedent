"""Shared evidence vocabulary; these categories are propositions, not a ranking.

Calibration is measured evidence with its own scope/commit attestation. No category
here, including agreement with an external package, grants a calibration license.
"""

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
    kind: str, oracle: dict | None, inference: str | None = None
) -> list[str]:
    """Check claimed *origin*, leaving scientific sufficiency to review/execution."""
    if kind not in EVIDENCE_KINDS:
        return [f"unknown evidence_kind {kind!r}"]
    if kind not in KNOWN_TRUTH_KINDS:
        return []
    if not isinstance(oracle, dict) or oracle.get("kind") not in ORACLE_KINDS:
        return ["known-truth claim requires an oracle with a recognized origin"]
    origin = oracle["kind"]
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
