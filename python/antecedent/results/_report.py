"""Pydantic inspection and analyze result models. Rust remains the scientific source of truth."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_serializer

_HANDLE_ATTRS = ("_raw", "_prepared", "_execution")


class SlotModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    available: bool
    reason: str | None = None
    summary: str
    payload: dict[str, Any] = Field(default_factory=dict)


class InspectionReport(BaseModel):
    """Typed ``inspect()`` report. ``to_dict()`` is JSON-safe ``model_dump``."""

    model_config = ConfigDict(frozen=True, extra="allow", arbitrary_types_allowed=True)

    identification: SlotModel
    support: SlotModel
    uncertainty: SlotModel
    assumptions: SlotModel
    program_id: str | None = None
    claim_id: str | None = None
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
    diagnostics: tuple[str, ...] | list[str] = ()
    target: dict[str, Any] | None = None
    inference_binding: Any = None

    @field_serializer("answer", "calibration", when_used="json")
    def _json_slot(self, value: Any) -> Any:
        from ._slots import json_value

        return json_value(value)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

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


def _slot(view: Any) -> SlotModel:
    if isinstance(view, SlotModel):
        return view
    return SlotModel(
        available=view.available,
        reason=view.reason,
        summary=view.summary,
        payload=dict(view.payload),
    )


def _unwrap_causal(error: ValidationError) -> BaseException | None:
    from ..errors import CausalError

    for item in error.errors():
        cause = (item.get("ctx") or {}).get("error")
        if isinstance(cause, CausalError):
            return cause
    return None


class ResultModel(BaseModel):
    """Frozen analyze/response view. ``to_dict()`` is a JSON-safe walk."""

    model_config = ConfigDict(frozen=True, extra="allow", arbitrary_types_allowed=True)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if args:
            names = list(type(self).model_fields)
            if len(args) > len(names):
                raise TypeError(
                    f"{type(self).__name__}() takes {len(names)} positional "
                    f"arguments but {len(args)} were given"
                )
            for name, value in zip(names, args, strict=False):
                if name in kwargs:
                    raise TypeError(f"{type(self).__name__}() got multiple values for {name!r}")
                kwargs[name] = value
        try:
            super().__init__(**kwargs)
        except ValidationError as error:
            causal = _unwrap_causal(error)
            if causal is not None:
                raise causal from error
            raise

    def model_post_init(self, __context: Any) -> None:
        extra = dict(self.model_extra or {})
        lifted = False
        for name in _HANDLE_ATTRS:
            if name in extra:
                object.__setattr__(self, name, extra.pop(name))
                lifted = True
        if lifted:
            object.__setattr__(self, "__pydantic_extra__", extra or None)

    def to_dict(self) -> dict[str, Any]:
        from ._slots import json_value

        return json_value(self)


def copy_model(obj: Any, **updates: Any) -> Any:
    """``model_copy`` / ``dataclasses.replace`` for mixed result graphs."""
    if isinstance(obj, BaseModel):
        return obj.model_copy(update=updates)
    from dataclasses import replace

    return replace(obj, **updates)


def as_inspection(slots: Any) -> InspectionReport:
    """Wrap ``ReasoningSlots`` without flattening answer/calibration to dicts."""
    if isinstance(slots, InspectionReport):
        return slots
    if hasattr(slots, "identification") and hasattr(slots, "to_dict"):
        dumped = slots.to_dict()
        return InspectionReport(
            identification=_slot(slots.identification),
            support=_slot(slots.support),
            uncertainty=_slot(slots.uncertainty),
            assumptions=_slot(slots.assumptions),
            program_id=slots.program_id,
            claim_id=slots.claim_id,
            target_id=slots.target_id,
            identification_id=slots.identification_id,
            identification_product_id=slots.identification_product_id,
            inference_binding_id=slots.inference_binding_id,
            observation_id=slots.observation_id,
            data_snapshot_id=slots.data_snapshot_id,
            execution_id=slots.execution_id,
            score_reuse_id=slots.score_reuse_id,
            target_weights_id=slots.target_weights_id,
            contract=slots.contract,
            answer=slots.answer,
            calibration=slots.calibration,
            diagnostics=slots.diagnostics,
            target=dumped.get("target"),
            inference_binding=dumped.get("inference_binding"),
        )
    return InspectionReport.model_validate(dict(slots))
