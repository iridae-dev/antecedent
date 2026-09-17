"""Four reasoning slots shared by prepared contracts and result wrappers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from math import isfinite
from typing import Any


def _identity_hex(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return bytes(value).hex()
    except (TypeError, ValueError):
        return str(value)


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _variable_names(contract: Mapping[str, Any]) -> list[str]:
    """Schema-ordered variable names from a live (flat) or portable (decoded) contract."""
    flat = contract.get("variable_names")
    if isinstance(flat, str):
        return flat.split(",") if flat else []
    for section in ("target", "observation"):
        schema = _nested(contract, section, "schema", "variables")
        if isinstance(schema, list):
            ordered = sorted(
                (v for v in schema if isinstance(v, Mapping)), key=lambda v: v.get("id", 0)
            )
            return [str(v.get("name")) for v in ordered]
    names = _nested(contract, "identification", "schema_names")
    return [str(n) for n in names] if isinstance(names, list) else []


def _resolve_variables(value: Any, names: list[str]) -> Any:
    """Schema ids → variable names; ``None`` stays ``None``, unknown ids stay ids."""
    if isinstance(value, list):
        return [_resolve_variables(v, names) for v in value]
    if isinstance(value, str) and value.lstrip("-").isdigit():
        value = int(value)
    if isinstance(value, int) and not isinstance(value, bool) and 0 <= value < len(names):
        return names[value]
    return value


def _target_query(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Query kind, treatment/outcome names (and schema ids), and target population."""
    names = _variable_names(contract)
    query = _nested(contract, "target", "query")
    if isinstance(query, Mapping) and len(query) == 1:
        kind, body = next(iter(query.items()))
        body = body if isinstance(body, Mapping) else {}
        treatment_id = body.get("treatment", body.get("treatments"))
        outcome_id = body.get("outcome", body.get("outcomes"))
        population: Any = body.get("target_population")
    else:
        kind = contract.get("query_kind")
        treatment_id = contract.get("treatment")
        outcome_id = contract.get("outcome")
        population = contract.get("population") or contract.get("target_population")
    if isinstance(treatment_id, str) and treatment_id.lstrip("-").isdigit():
        treatment_id = int(treatment_id)
    if isinstance(outcome_id, str) and outcome_id.lstrip("-").isdigit():
        outcome_id = int(outcome_id)
    return {
        "target_population": population,
        "kind": kind,
        "treatment": _resolve_variables(treatment_id, names),
        "outcome": _resolve_variables(outcome_id, names),
        "treatment_id": treatment_id,
        "outcome_id": outcome_id,
    }


__all__ = [
    "ReasoningSlots",
    "SlotView",
    "describe_limitation",
    "display_mass",
    "mass_limitation",
]


def portable_evidence(
    section: Mapping[str, Any], body: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Execution evidence a portable ``analysis_result`` carries, per slot.

    The one projection both a live result and its loaded export report, so the
    identification method and adjustment set, the validation verdict (a failed
    one included) and counterfactual unit effects read the same before and
    after ``load(export())``.
    """
    names = _variable_names(section)
    raw_identification = body.get("identification")
    identification_wire = raw_identification if isinstance(raw_identification, Mapping) else {}
    estimands = identification_wire.get("estimands")
    first = (
        estimands[0]
        if isinstance(estimands, list) and estimands and isinstance(estimands[0], Mapping)
        else {}
    )
    raw_adjustment = list(first.get("adjustment_set") or [])
    unfolded = body.get("identification_variables")
    coordinates: list[dict[str, Any]] | None = None
    if isinstance(unfolded, list) and unfolded:
        # A temporal estimand adjusts on nodes of the unfolded window: each id
        # indexes `identification_variables` (schema variable, signed offset),
        # never the schema directly.
        coordinates = []
        for node in raw_adjustment:
            index = int(node) if str(node).lstrip("-").isdigit() else -1
            key = unfolded[index] if 0 <= index < len(unfolded) else None
            if not isinstance(key, Mapping):
                coordinates = None
                break
            coordinates.append(
                {
                    "name": _resolve_variables(key.get("variable"), names),
                    "offset": key.get("offset"),
                    "node": index,
                }
            )
    adjustment_set = (
        list(dict.fromkeys(item["name"] for item in coordinates))
        if coordinates is not None
        else _resolve_variables(raw_adjustment, names)
    )
    identification = {
        "method": first.get("method"),
        "adjustment_set": adjustment_set,
        **({"adjustment_coordinates": coordinates} if coordinates is not None else {}),
        "assumption_count": len(identification_wire.get("required_assumptions") or []),
        "derivation_step_count": len(identification_wire.get("derivation") or []),
    }
    reports = [dict(r) for r in body.get("refutations") or [] if isinstance(r, Mapping)]

    def field(diagnostic: Mapping[str, Any], name: str) -> str:
        for pair in diagnostic.get("fields") or []:
            if isinstance(pair, (list, tuple)) and len(pair) == 2 and pair[0] == name:
                return str(pair[1])
        return ""

    failures = [
        {"validator": field(d, "validator"), "reason": field(d, "reason")}
        for d in body.get("diagnostics") or []
        if isinstance(d, Mapping) and d.get("code") == "refute.validator.failed"
    ]
    count = len(reports) + len(failures)
    ran = count > 0
    validation = {
        "passed": ran and not failures and all(r.get("passed") is True for r in reports),
        "ran": ran,
        "count": count,
        "reports": reports,
        "computation_failures": failures,
    }
    uncertainty: dict[str, Any] = {}
    unit_effects = body.get("unit_effects")
    if isinstance(unit_effects, Mapping):
        uncertainty["unit_effects"] = dict(unit_effects)
    return {
        "identification": identification,
        "support": {"validation": validation},
        "uncertainty": uncertainty,
    }


def with_portable_evidence(slots: ReasoningSlots, evidence: Mapping[str, Any]) -> ReasoningSlots:
    """``slots`` with :func:`portable_evidence` merged into each slot's payload."""
    from dataclasses import replace

    def merged(view: SlotView, extra: Mapping[str, Any]) -> SlotView:
        return replace(view, payload={**view.payload, **extra})

    return replace(
        slots,
        identification=merged(slots.identification, evidence.get("identification", {})),
        support=merged(slots.support, evidence.get("support", {})),
        uncertainty=merged(slots.uncertainty, evidence.get("uncertainty", {})),
    )


def json_value(value: Any) -> Any:
    """Portable report data; preserve nonfinite bounds explicitly, never emit NaN."""
    if isinstance(value, Enum):
        return json_value(value.value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if isfinite(value) else {"value": None, "representation": str(value)}
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: json_value(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    return {"available": False, "reason": "not_json_serializable", "type": type(value).__name__}


@dataclass(frozen=True, slots=True)
class SlotView:
    available: bool
    reason: str | None
    summary: str
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ReasoningSlots:
    identification: SlotView
    support: SlotView
    uncertainty: SlotView
    assumptions: SlotView
    #: Compiled program identity (`contract["program"]`). Distinct from claim_id.
    program_id: str | None
    #: Execution claim identity (`contract["claim"]["claim_id"]`). Distinct from program_id.
    claim_id: str | None
    target_id: str | None = None
    identification_id: str | None = None
    identification_product_id: str | None = None
    inference_binding_id: str | None = None
    observation_id: str | None = None
    data_snapshot_id: str | None = None
    execution_id: str | None = None
    score_reuse_id: str | None = None
    target_weights_id: str | None = None
    contract: dict[str, Any] | None = None
    answer: Any = None
    calibration: Any = None
    diagnostics: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Structured report using native booleans, numbers, and explicit unavailable slots."""
        payload = json_value(self)
        contract = self.contract if isinstance(self.contract, Mapping) else {}
        payload["target"] = {"query": _target_query(contract)}
        payload["inference_binding"] = (
            contract.get("inference_binding") or self.inference_binding_id
        )
        return payload

    @classmethod
    def from_contract(cls, contract: Mapping[str, str]) -> ReasoningSlots:
        def slot(label: str, extras: dict[str, Any] | None = None) -> SlotView:
            payload = dict(extras or {})
            if label.startswith("unavailable:"):
                return SlotView(False, label.split(":", 1)[1], label, payload)
            return SlotView(True, None, label, payload)

        ident_payload: dict[str, Any] = {}
        if "identified_mass" in contract:
            ident_payload["identified_mass"] = float(contract["identified_mass"])
            ident_payload["unidentified_mass"] = float(contract["unidentified_mass"])
        for key in ("unevaluable_mass", "incomplete_search_mass"):
            if key in contract:
                ident_payload[key] = float(contract[key])
        for key in ("full_mass_scope", "search_capped"):
            if key in contract:
                ident_payload[key] = contract[key] == "true"
        support_payload = {
            "matrix_status": contract.get("matrix_status"),
            "matrix_coordinate": contract.get("matrix_coordinate"),
            "empirical": contract.get("empirical_support"),
        }
        return cls(
            identification=slot(
                contract.get("identification_status", "unavailable:missing"),
                ident_payload,
            ),
            support=slot(contract.get("matrix_status", "unavailable:missing"), support_payload),
            uncertainty=slot(
                contract.get("uncertainty", "unavailable:missing"),
                {"components": json.loads(contract["uncertainty_components"])}
                if "uncertainty_components" in contract
                else None,
            ),
            assumptions=slot(
                contract.get("assumptions", "unavailable:missing"),
                {"obligations": json.loads(contract["assumption_obligations"])}
                if "assumption_obligations" in contract
                else None,
            ),
            program_id=contract.get("program") or _nested(contract, "identities", "program"),
            claim_id=contract.get("claim_id") or _nested(contract, "claim", "claim_id"),
            target_id=contract.get("target") or _nested(contract, "identities", "target"),
            identification_id=contract.get("identification")
            or _nested(contract, "identities", "identification"),
            identification_product_id=contract.get("identification_product")
            or _nested(contract, "identities", "identification_product"),
            inference_binding_id=contract.get("inference_binding")
            or _nested(contract, "identities", "inference_binding"),
            observation_id=contract.get("observation")
            or _nested(contract, "identities", "observation"),
            data_snapshot_id=contract.get("data_snapshot")
            or _nested(contract, "identities", "data_snapshot"),
            execution_id=_nested(contract, "identities", "execution"),
            score_reuse_id=contract.get("score_reuse")
            or _nested(contract, "identities", "score_reuse"),
            target_weights_id=contract.get("target_weights")
            or _nested(contract, "identities", "target_weights"),
            contract=dict(contract) if contract else None,
        )

    @classmethod
    def from_result_section(
        cls,
        section: Mapping[str, Any],
        body: Mapping[str, Any] | None = None,
        *,
        answer: Any = None,
        calibration: Any = None,
    ) -> ReasoningSlots:
        """Inspect a portable execution contract. One owner for loaded identities."""

        def section_map(key: str) -> Mapping[str, Any]:
            value = section.get(key)
            return value if isinstance(value, Mapping) else {}

        reasoning = section_map("reasoning")
        identities = section_map("identities")
        claim = section_map("claim")
        body = body if isinstance(body, Mapping) else {}

        def slot(name: str) -> SlotView:
            raw = reasoning.get(name)
            raw = raw if isinstance(raw, Mapping) else {}
            value = raw.get("value")
            return SlotView(
                value is not None,
                raw.get("unavailable") if isinstance(raw.get("unavailable"), str) else None,
                (value.get("status", "available") if isinstance(value, Mapping) else "unavailable"),
                dict(value) if isinstance(value, Mapping) else {},
            )

        identification = slot("identification")
        support = slot("support")
        uncertainty = slot("uncertainty")
        assumptions = slot("assumptions")
        details = dict(uncertainty.payload)
        if body.get("standard_error") is not None:
            details["standard_error"] = body["standard_error"]
        response = body.get("response") or {}
        if isinstance(response, Mapping) and response.get("uncertainty") is not None:
            details["response"] = response["uncertainty"]
            if response["uncertainty"] != "none" and not uncertainty.available:
                uncertainty = SlotView(True, None, "response_specific", details)
            else:
                uncertainty = SlotView(
                    uncertainty.available, uncertainty.reason, uncertainty.summary, details
                )
        else:
            uncertainty = SlotView(
                uncertainty.available, uncertainty.reason, uncertainty.summary, details
            )
        structural = body.get("structural_response") or {}
        if (
            isinstance(structural, Mapping)
            and structural.get("identified_set_interval") is not None
        ):
            details = dict(uncertainty.payload)
            details["identified_set_interval"] = structural["identified_set_interval"]
            uncertainty = SlotView(
                uncertainty.available, uncertainty.reason, uncertainty.summary, details
            )
        if isinstance(response, Mapping) and response.get("support") is not None:
            support = SlotView(
                support.available,
                support.reason,
                support.summary,
                {**support.payload, "execution": response["support"]},
            )
        assumptions = SlotView(
            assumptions.available,
            assumptions.reason,
            assumptions.summary,
            {**assumptions.payload, "records": body.get("assumptions", [])},
        )
        slots = cls(
            identification=identification,
            support=support,
            uncertainty=uncertainty,
            assumptions=assumptions,
            program_id=_identity_hex(identities.get("program")),
            claim_id=_identity_hex(claim.get("claim_id")),
            target_id=_identity_hex(identities.get("target")),
            identification_id=_identity_hex(identities.get("identification")),
            identification_product_id=_identity_hex(identities.get("identification_product")),
            inference_binding_id=_identity_hex(identities.get("inference_binding")),
            observation_id=_identity_hex(identities.get("observation")),
            data_snapshot_id=_identity_hex(identities.get("data_snapshot")),
            execution_id=_identity_hex(identities.get("execution")),
            score_reuse_id=_identity_hex(identities.get("score_reuse")),
            target_weights_id=_identity_hex(identities.get("target_weights")),
            contract=dict(section),
            answer=answer,
            calibration=calibration,
            diagnostics=tuple(
                f"{d.get('code', '')}: {d.get('message', '')}"
                for d in body.get("diagnostics", [])
                if isinstance(d, Mapping)
            ),
        )
        return with_portable_evidence(slots, portable_evidence(section, body))

    def rendering_limitation(self) -> str | None:
        if not self.identification.available:
            return "identification_unavailable"
        for key in ("unidentified_mass", "unevaluable_mass", "incomplete_search_mass"):
            value = self.identification.payload.get(key)
            if isinstance(value, (float, int)) and value > 0:
                return key
        if self.identification.payload.get("search_capped"):
            return "incomplete_search"
        if self.identification.summary in ("not_identified", "unavailable"):
            return "identification_unavailable"
        if self.identification.summary == "partially_identified":
            return "identified_set"
        return None


#: Readable caveat per rendering-limitation id. Every renderer (reprs, HTML)
#: that withholds a point display names the id and this phrase.
LIMITATION_PHRASES: dict[str, str] = {
    "identified_set": "set-identified: a single number is a mixture over identified "
    "completions, not a point of the identified set",
    "unidentified_mass": "part of the structure gives no identified estimand; a single "
    "number averages only the structures where it is identified",
    "unevaluable_mass": "part of the identified structure could not be evaluated; a single "
    "number averages only the evaluated structures",
    "incomplete_search_mass": "the identification search did not finish for part of the structure",
    "incomplete_search": "the identification search was capped",
    "identification_unavailable": "not identified; there is no causal point",
    "absent_interval": "no scalar effect",
}


def describe_limitation(limitation: str) -> str:
    """``"<phrase> (<id>)"``; an id without a phrase renders as itself."""
    phrase = LIMITATION_PHRASES.get(limitation)
    return f"{phrase} ({limitation})" if phrase else limitation


def mass_limitation(mass: object, *, identified_set: bool = False) -> str | None:
    """Stable limitation id for unresolved structural mass or a set-valued claim."""
    if isinstance(mass, (int, float)) and mass > 0:
        return "unidentified_mass"
    if identified_set:
        return "identified_set"
    return None


def display_mass(structural: object, posterior: object) -> float | None:
    """The one unidentified mass a renderer shows, from the two a result can carry.

    Structural mass wins: it is the mass of the *claim* — the share of the
    completion enumeration that gives no identified estimand — while a graph
    posterior's unidentified mass describes the sampled structures behind it.
    A result carrying both must show the same number in every renderer, so
    this is the only precedence rule; :func:`mass_limitation` reads it, and so
    does the notebook callout.
    """
    for mass in (structural, posterior):
        if isinstance(mass, (int, float)) and not isinstance(mass, bool):
            return float(mass)
    return None
