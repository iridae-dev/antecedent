"""Four reasoning slots shared by prepared contracts and result wrappers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum, StrEnum
from math import isfinite
from typing import Any, Literal

from ..errors import RenderingLimitation

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
    claim_id: str | None
    data_version: str | None
    answer: Any = None
    calibration: Any = None
    diagnostics: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Structured report using native booleans, numbers, and explicit unavailable slots."""
        return json_value(self)

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
            claim_id=contract.get("program"),
            data_version=contract.get("data_snapshot"),
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
