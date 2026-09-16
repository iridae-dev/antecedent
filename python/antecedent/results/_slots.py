"""Four reasoning slots shared by prepared contracts and result wrappers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum, StrEnum
from math import isfinite
from typing import Any, Literal

from ..errors import RenderingLimitation


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


__all__ = [
    "ConsumerIntent",
    "ReasoningSlots",
    "RenderingLimitation",
    "SlotView",
    "mass_limitation",
    "require_scalar_display",
    "slots_from_prepared",
]


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


class ConsumerIntent(StrEnum):
    """Host actions. Only scientific intents route through preview/apply."""

    DISPLAY = "display_coordinates"
    PRESENTATION = "change_presentation"
    NEW_SUBGROUP = "new_subgroup"
    NEW_CAUSAL_TARGET = "new_causal_target"

    @property
    def kind(self) -> Literal["display", "presentation", "scientific"]:
        if self is ConsumerIntent.DISPLAY:
            return "display"
        if self is ConsumerIntent.PRESENTATION:
            return "presentation"
        return "scientific"

    def preview_name(self) -> str | None:
        return {
            ConsumerIntent.NEW_SUBGROUP: "filter_population",
            ConsumerIntent.NEW_CAUSAL_TARGET: "new_conditional_query",
        }.get(self)


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
    data_version: str | None = None
    contract: dict[str, Any] | None = None
    answer: Any = None
    calibration: Any = None
    diagnostics: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Structured report using native booleans, numbers, and explicit unavailable slots."""
        payload = json_value(self)
        contract = self.contract if isinstance(self.contract, Mapping) else {}
        population = contract.get("population") or contract.get("target_population")
        payload["target"] = {
            "query": {
                "target_population": population,
                "kind": contract.get("query_kind"),
                "treatment": contract.get("treatment"),
                "outcome": contract.get("outcome"),
            }
        }
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
            data_version=contract.get("data_snapshot")
            or _nested(contract, "identities", "data_snapshot"),
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
        reasoning = section.get("reasoning") if isinstance(section.get("reasoning"), Mapping) else {}
        identities = (
            section.get("identities") if isinstance(section.get("identities"), Mapping) else {}
        )
        claim = section.get("claim") if isinstance(section.get("claim"), Mapping) else {}
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
        if isinstance(structural, Mapping) and structural.get("identified_set_interval") is not None:
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
        return cls(
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
            data_version=_identity_hex(identities.get("data_snapshot")),
            contract=dict(section),
            answer=answer,
            calibration=calibration,
            diagnostics=tuple(
                f"{d.get('code', '')}: {d.get('message', '')}"
                for d in body.get("diagnostics", [])
                if isinstance(d, Mapping)
            ),
        )

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

    def display_effect(self, effect: float | None) -> float:
        return require_scalar_display(self.rendering_limitation(), effect)


def mass_limitation(mass: object, *, identified_set: bool = False) -> str | None:
    """Stable limitation id for unresolved structural mass or a set-valued claim."""
    if isinstance(mass, (int, float)) and mass > 0:
        return "unidentified_mass"
    if identified_set:
        return "identified_set"
    return None


def require_scalar_display(limitation: str | None, effect: float | None) -> float:
    if limitation is not None:
        raise RenderingLimitation(limitation)
    if effect is None:
        raise RenderingLimitation("absent_interval")
    return effect


def slots_from_prepared(prepared: object | None) -> ReasoningSlots | None:
    reasoning = getattr(prepared, "reasoning", None)
    if reasoning is None:
        return None
    return reasoning() if callable(reasoning) else reasoning
