"""Host-integration contract for software built on Antecedent.

Antecedent is a library; it ships no agent. A host (a service, a notebook tool, an
LLM tool layer) compiles a request, reads what the compiled analysis supports, runs it,
and reads the claim. This module is what such a host needs to do that without parsing
prose:

- :func:`schema` -- the versioned JSON Schema for requests, contracts, claims, refusals
  and the ``supported`` / ``partial`` / ``unsupported`` statuses. It is generated from
  the Rust registries and enums that own each vocabulary and is the same file Rust
  embeds as ``antecedent::host::HOST_SCHEMA_TEXT``;
- :func:`claim_status`, :func:`operation_status` and :func:`inspection_status` -- the
  status of a claim or an operation, decided by the one Rust rule;
- :func:`refusal_codes`, :func:`refusal_code`, :func:`error_family`, :func:`refusal_record`,
  :func:`error_response` and :func:`claim_response` -- the stable registered refusal codes,
  exception families and response records, shared with Rust;
- :func:`fallback_response` -- the record of an opt-in descriptive fallback: the unchanged
  refusal beside the status of the noncausal claim.

A status is advice about use, never a verdict about the world: ``supported`` means the
route is licensed, identified under the declared structure and executed, with nothing
contradicting it; it does not certify the assumptions.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from typing import Any, Literal, cast

from . import _native
from .errors import CausalTypeError, CausalValueError

__all__ = [
    "CONTRACT_VERSION",
    "ErrorFamily",
    "HostStatus",
    "claim_record",
    "claim_response",
    "claim_status",
    "error_families",
    "error_family",
    "error_response",
    "fallback_response",
    "inspection_status",
    "operation_record",
    "operation_status",
    "reason_codes",
    "refusal_code",
    "refusal_codes",
    "refusal_record",
    "schema",
    "schema_text",
]

#: What a host may do with a claim or an operation.
HostStatus = Literal["supported", "partial", "unsupported"]


@cache
def _schema_text() -> str:
    return _native.host_schema_text()


def schema_text() -> str:
    """The committed schema exactly as generated (byte-identical to the file in the crate)."""
    return _schema_text()


def schema() -> dict[str, Any]:
    """The host-integration JSON Schema, parsed. Each call returns an independent copy."""
    loaded: dict[str, Any] = json.loads(_schema_text())
    return loaded


#: Version of the contract. A consumer must change when it moves; new enumeration
#: members and optional fields do not move it.
CONTRACT_VERSION: int = int(schema()["x-contract-version"])


def _enum(name: str) -> frozenset[str]:
    return frozenset(schema()["$defs"][name]["enum"])


def refusal_codes() -> frozenset[str]:
    """Registered runtime-refusal codes: every ``reason_code`` a refusal can carry."""
    return _enum("refusal_code")


def reason_codes() -> frozenset[str]:
    """Every registered reason code, including the calibration and product vocabularies."""
    return _enum("reason_code")


@dataclass(frozen=True)
class ErrorFamily:
    """One stable exception family."""

    code: str
    python_class: str
    parent: str | None
    origin: str
    #: The coarse family Rust reports for a failure Python narrows, when it narrows one.
    narrows: str | None


@cache
def _families() -> tuple[ErrorFamily, ...]:
    return tuple(ErrorFamily(**row) for row in schema()["x-error-families"])


def error_families() -> tuple[ErrorFamily, ...]:
    """The stable exception families, root first."""
    return _families()


def error_family(error: BaseException | type[BaseException]) -> str:
    """Stable family code of a failure, equal to the one Rust reports for the same error.

    The nearest registered family along the exception's class hierarchy; a Python-only
    narrowing (a type or value error, a structured refusal) reports the coarse family
    Rust uses. Anything outside the hierarchy is ``"error"``, the root.
    """
    cls = error if isinstance(error, type) else type(error)
    by_class = {family.python_class: family for family in _families()}
    by_code = {family.code: family for family in _families()}
    for klass in cls.__mro__:
        family = by_class.get(klass.__name__)
        if family is not None:
            while family.narrows is not None:
                family = by_code[family.narrows]
            return family.code
    return "error"


def _wire_digest(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).hex()
    if isinstance(value, (list, tuple)) and all(isinstance(item, int) for item in value):
        return bytes(value).hex()
    return str(value)


def _claim_parts(claim: Mapping[str, Any]) -> tuple[str, str, str, str]:
    """Kind and the three domain names, from a contract claim section or a claim record."""
    domains = claim.get("domains")
    if isinstance(domains, Mapping):
        identification = domains.get("identification")
        support = domains.get("support")
        evaluated = domains.get("evaluated")
    else:
        identification = claim.get("identification_domain")
        support = claim.get("support_domain")
        evaluated = claim.get("evaluated_domain")
    return (
        str(claim.get("kind")),
        str(identification),
        str(support),
        str(evaluated),
    )


def claim_status(claim: Mapping[str, Any]) -> HostStatus:
    """Status of a claim: ``result.inspect().contract["claim"]`` or a :func:`claim_record`.

    A name outside the registered vocabularies is ``"unsupported"``: a host that cannot
    read a claim must not use it.
    """
    status = _native.host_status_of_claim(*_claim_parts(claim))
    return cast("HostStatus", "unsupported" if status is None else status)


def operation_status(applicability: str, readiness: str | None = None) -> HostStatus:
    """Status of an operation from the capability report's applicability and readiness."""
    status = _native.host_status_of_operation(applicability, readiness)
    return cast("HostStatus", "unsupported" if status is None else status)


_APPLICABILITY = {
    "licensed": "licensed",
    "allowed_unlicensed": "unlicensed",
    "refused": "unlicensed",
    "not_applicable": "impossible",
}


def inspection_status(report: Any) -> HostStatus:
    """Status of an ``inspect()`` report, before or after execution.

    With a claim in the contract the claim decides. Before a claim exists, the support
    slot's matrix status gives the applicability, and a licensed cell whose
    identification is available is read as executable; a licensed cell without it is
    partial (prepare first), and an off-axis cell is partial (unknown support).
    """
    contract = getattr(report, "contract", None)
    claim = contract.get("claim") if isinstance(contract, Mapping) else None
    if isinstance(claim, Mapping) and claim.get("kind") is not None:
        return claim_status(claim)
    matrix = str(report.support.payload.get("matrix_status") or report.support.summary)
    applicability = _APPLICABILITY.get(matrix, "unknown")
    readiness = None
    if applicability == "licensed":
        readiness = "executable" if report.identification.available else "binding_missing"
    return operation_status(applicability, readiness)


def operation_record(report: Mapping[str, Any]) -> dict[str, Any]:
    """Host record of an operation report, shaped as the schema's ``operation_report``.

    ``report`` carries the capability report's fields: ``operation``, ``applicability``,
    ``readiness`` (or ``None``), ``coordinate`` (or ``None``), ``blockers`` (mappings with
    ``id``, ``reason`` and ``scientific``), ``next_actions`` (``id``, ``description``),
    ``neighbors`` and ``unmet_obligations`` (the records an ``inspect()`` capability report and
    :meth:`antecedent.repair.EvidencePlanStage.capability_report` give). The status is the one
    :func:`operation_status` reads from the applicability and readiness.
    """
    if not isinstance(report, Mapping):
        raise CausalTypeError("report must be a mapping of the capability report's fields")
    for key in ("operation", "applicability"):
        if not isinstance(report.get(key), str):
            raise CausalValueError(f"report needs the text field {key!r}")
    readiness = report.get("readiness")
    coordinate = report.get("coordinate")
    return {
        "operation": report["operation"],
        "applicability": report["applicability"],
        "readiness": None if readiness is None else str(readiness),
        "coordinate": None if coordinate is None else str(coordinate),
        "blockers": [
            {
                "id": str(b["id"]),
                "reason": str(b["reason"]),
                "scientific": bool(b["scientific"]),
            }
            for b in report.get("blockers", ())
        ],
        "next_actions": [
            {"id": str(a["id"]), "description": str(a["description"])}
            for a in report.get("next_actions", ())
        ],
        "neighbors": [dict(n) for n in report.get("neighbors", ())],
        "unmet_obligations": [dict(o) for o in report.get("unmet_obligations", ())],
        "status": operation_status(
            str(report["applicability"]), None if readiness is None else str(readiness)
        ),
    }


def claim_record(claim: Mapping[str, Any]) -> dict[str, Any]:
    """Host record of a claim, shaped as the schema's ``claim_envelope``."""
    kind, identification, support, evaluated = _claim_parts(claim)
    return {
        "claim_id": _wire_digest(claim.get("claim_id")),
        "kind": kind,
        "causal": kind != "associational",
        "domains": {
            "identification": identification,
            "support": support,
            "evaluated": evaluated,
        },
        "execution": _wire_digest(claim.get("execution")),
        "status": claim_status(claim),
    }


def claim_response(claim: Mapping[str, Any]) -> dict[str, Any]:
    """Host record of a claim as a ``host_response``.

    A refusal-kind claim is carried as the claim with status ``unsupported``; use
    :func:`error_response` for a failed call.
    """
    record = claim_record(claim)
    return {"status": record["status"], "claim": record}


def fallback_response(fallback: Any) -> dict[str, Any]:
    """Host record of a descriptive fallback, shaped as the schema's ``fallback_response``.

    ``fallback`` is the :class:`antecedent.fallback.AnalysisWithFallback` that
    ``analyze(..., on_refusal="describe")`` returned. The causal question stays refused:
    ``status`` and ``causal_status`` are ``"unsupported"`` and ``refusal`` is the unchanged
    refusal record. Beside them ``associational_status`` is the host status of the noncausal
    claim (or ``"unavailable"`` with its typed reason), and the associational claim carries
    ``causal: false`` and ranks no action.
    """
    associational_status, associational = fallback._associational_record()
    evidence = fallback.evidence_plan
    return {
        "status": "unsupported",
        "causal_status": "unsupported",
        "associational_status": associational_status,
        "refusal": refusal_record(fallback.refusal),
        "associational": associational,
        "evidence_plan": evidence.to_dict(),
        "notice": fallback.notice,
    }


def refusal_code(error: BaseException) -> str | None:
    """The registered refusal code ``error`` carries, or ``None``.

    A code outside the registered vocabulary is never reported, so a host never sees a
    code the schema does not list.
    """
    code = getattr(error, "reason_code", None)
    return code if code in refusal_codes() else None


def error_response(error: BaseException) -> dict[str, Any]:
    """Host record of a failed call as a ``host_response``: always ``unsupported``.

    ``refusal.family`` tells a host whether to retry (``resource``, ``cancelled``), ask for
    a review (``review_required``) or give up (``unsupported``).
    """
    return {"status": "unsupported", "refusal": refusal_record(error)}


def refusal_record(error: BaseException) -> dict[str, Any]:
    """Host record of a failed call, shaped as the schema's ``refusal``.

    ``reason_code`` is the registered code the error carries, or ``None``: a code that is
    not registered is never reported. ``family`` classifies every failure, coded or not.
    """
    remedy = getattr(error, "remedy", None)
    return {
        "status": "unsupported",
        "family": error_family(error),
        "reason_code": refusal_code(error),
        "message": str(error),
        "remedy": remedy if isinstance(remedy, str) else None,
    }
