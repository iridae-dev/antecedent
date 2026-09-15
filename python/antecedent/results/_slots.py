"""Four reasoning slots shared by prepared contracts and result wrappers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
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


class ConsumerIntent(str, Enum):
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
    claim_id: str
    data_version: str

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
            uncertainty=slot(contract.get("uncertainty", "unavailable:missing")),
            assumptions=slot(contract.get("assumptions", "unavailable:missing")),
            claim_id=contract["program"],
            data_version=contract["data_snapshot"],
        )

    def rendering_limitation(self) -> str | None:
        if not self.identification.available:
            return "identification_unavailable"
        return mass_limitation(self.identification.payload.get("unidentified_mass"))

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
