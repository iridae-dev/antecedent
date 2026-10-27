"""Pydantic inspection and analyze result models. Rust remains the scientific source of truth."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from itertools import islice
from typing import Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationError,
    field_serializer,
    field_validator,
    model_serializer,
)
from pydantic.config import ExtraValues

from .._columns import coerce_fields
from .._freeze import freeze, thaw

InspectionStage = Literal[
    "prepared", "executed", "declared", "consumed", "exported", "unavailable", "refused"
]

_HANDLE_ATTRS = ("_raw", "_prepared", "_execution", "_live_study")


class SlotModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    available: bool
    reason: str | None = None
    summary: str
    payload: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("payload")
    @classmethod
    def _freeze_payload(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        return freeze(value)

    @field_serializer("payload")
    def _serialize_payload(self, value: Mapping[str, Any]) -> Any:
        return thaw(value)


class InspectionReport(BaseModel):
    """Typed ``inspect()`` report. ``to_dict()`` is JSON-safe ``model_dump``."""

    model_config = ConfigDict(frozen=True, extra="allow", arbitrary_types_allowed=True)

    identification: SlotModel
    support: SlotModel
    uncertainty: SlotModel
    assumptions: SlotModel
    program_id: str | None = None
    lowered_program_id: str | None = None
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
    contract: Mapping[str, Any] | None = None
    answer: Any = None
    calibration: Any = None
    diagnostics: tuple[str, ...] | list[str] = ()
    #: What the capability report suggests next (``id``, ``description``); never executed.
    #: A not-identified contract adds ``measure_variable:<name>``, ``run_experiment:<target>``,
    #: ``find_instrument`` and ``collect_panel``.
    next_actions: tuple[Mapping[str, Any], ...] = ()
    #: Licensed alternatives that change named premises; an explicit choice, never a fallback.
    neighbors: tuple[Mapping[str, Any], ...] = ()
    #: Unresolved obligations of the contract.
    unmet_obligations: tuple[Mapping[str, Any], ...] = ()
    #: Verified evidence obligations of a not-identified product, each with its proof step.
    evidence_obligations: tuple[Mapping[str, Any], ...] = ()
    target: Mapping[str, Any] | None = None
    inference_binding: Any = None
    stage: InspectionStage = "prepared"
    family: str | None = None
    details: Any = None
    operations: tuple[str, ...] = ()

    @field_validator(
        "contract",
        "target",
        "details",
        "diagnostics",
        "inference_binding",
        "answer",
        "calibration",
        "next_actions",
        "neighbors",
        "unmet_obligations",
        "evidence_obligations",
    )
    @classmethod
    def _freeze_details(cls, value: Any) -> Any:
        return freeze(value)

    @field_serializer(
        "contract",
        "target",
        "details",
        "inference_binding",
        "next_actions",
        "neighbors",
        "unmet_obligations",
        "evidence_obligations",
        when_used="json",
    )
    def _json_details(self, value: Any) -> Any:
        return thaw(value)

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        result = super().model_copy(update=update, deep=deep)
        for name in (
            "contract",
            "target",
            "details",
            "diagnostics",
            "inference_binding",
            "answer",
            "calibration",
            "next_actions",
            "neighbors",
            "unmet_obligations",
            "evidence_obligations",
        ):
            object.__setattr__(result, name, freeze(getattr(result, name)))
        return result

    def __repr__(self) -> str:
        return f"InspectionReport(stage={self.stage!r}, identification={self.identification.summary!r}, uncertainty={self.uncertainty.summary!r})"

    def __str__(self) -> str:
        return self.__repr__()

    def _repr_html_(self) -> str:
        from html import escape

        rows = "".join(
            f"<tr><th>{name}</th><td>{escape(slot.summary)}</td></tr>"
            for name, slot in (
                ("Identification", self.identification),
                ("Support", self.support),
                ("Uncertainty", self.uncertainty),
                ("Assumptions", self.assumptions),
            )
        )
        # Notebook display is a bounded summary, not a hidden JSON serialization
        # of an entire posterior or retained artifact.
        values = self.answer
        if values is not None:
            reading = f"answer={getattr(values, 'kind', 'structured')}"
            for name in ("value", "bounds", "detail"):
                value = getattr(values, name, None)
                if type(value) in (str, int, float, bool) or (
                    name == "bounds" and isinstance(value, tuple) and len(value) == 2
                ):
                    reading += f" {name}={str(value)[:160]}"
        else:
            reading = ""
        # Only the first few scalar entries of a mapping are shown; nested or large
        # records are never walked or serialized by a notebook display.
        for label, record in (("answer", values), ("details", self.details)):
            if isinstance(record, Mapping):
                shown = [
                    f"{key}={str(item)[:80]}"
                    for key, item in islice(record.items(), 32)
                    if type(item) in (str, int, float, bool)
                ][:8]
                if shown:
                    reading += ("\n" if reading else "") + f"{label}: " + " ".join(shown)
        detail = f"<pre>{escape(reading)}</pre>" if reading else ""
        limitation = self.rendering_limitation()
        warning = f"<p>{escape(limitation.replace('_', ' '))}</p>" if limitation else ""
        operations = (
            f"<p>Available: {escape(', '.join(self.operations))}</p>" if self.operations else ""
        )
        return f"<div><strong>{escape(self.family or 'Antecedent')} · {escape(self.stage)}</strong>{detail}<table>{rows}</table>{warning}{operations}</div>"

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


@contextmanager
def _causal_validation() -> Iterator[None]:
    """Preserve domain errors at every public result construction entry point."""
    try:
        yield
    except ValidationError as error:
        causal = _unwrap_causal(error)
        if causal is not None:
            raise causal from error
        raise


class ResultModel(BaseModel):
    """Frozen analyze/response view. ``to_dict()`` is a JSON-safe walk."""

    model_config = ConfigDict(frozen=True, extra="allow", arbitrary_types_allowed=True)
    _live_study: Any = PrivateAttr(default=None)

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
        with _causal_validation():
            super().__init__(**kwargs)

    @classmethod
    def model_validate(
        cls,
        obj: Any,
        *,
        strict: bool | None = None,
        extra: ExtraValues | None = None,
        from_attributes: bool | None = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Self:
        with _causal_validation():
            return super().model_validate(
                obj,
                strict=strict,
                extra=extra,
                from_attributes=from_attributes,
                context=context,
                by_alias=by_alias,
                by_name=by_name,
            )

    @classmethod
    def model_validate_json(
        cls,
        json_data: str | bytes | bytearray,
        *,
        strict: bool | None = None,
        extra: ExtraValues | None = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Self:
        with _causal_validation():
            return super().model_validate_json(
                json_data,
                strict=strict,
                extra=extra,
                context=context,
                by_alias=by_alias,
                by_name=by_name,
            )

    @classmethod
    def model_validate_strings(
        cls,
        obj: Any,
        *,
        strict: bool | None = None,
        extra: ExtraValues | None = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Self:
        with _causal_validation():
            return super().model_validate_strings(
                obj,
                strict=strict,
                extra=extra,
                context=context,
                by_alias=by_alias,
                by_name=by_name,
            )

    def model_post_init(self, __context: Any) -> None:
        extra = dict(self.__pydantic_extra__ or {})
        for name in _HANDLE_ATTRS:
            if name in extra:
                coerce_fields(self, **{name: extra.pop(name)})
        for name in type(self).model_fields:
            if not name.startswith("_"):
                object.__setattr__(self, name, freeze(getattr(self, name)))
        coerce_fields(
            self, __pydantic_extra__={name: freeze(value) for name, value in extra.items()} or None
        )

    @property
    def model_extra(self) -> Any:
        from types import MappingProxyType

        extra = self.__pydantic_extra__
        return MappingProxyType(extra) if extra is not None else None

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        result = super().model_copy(update=update, deep=deep)
        result.model_post_init(None)
        return result

    @model_serializer(mode="plain")
    def _serialize_result(self) -> dict[str, Any]:
        from ._slots import json_value

        return json_value(self)

    def to_dict(self) -> dict[str, Any]:
        return self._serialize_result()


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
        from ._slots import _target_query

        contract = slots.contract if isinstance(slots.contract, Mapping) else {}
        return InspectionReport(
            identification=_slot(slots.identification),
            support=_slot(slots.support),
            uncertainty=_slot(slots.uncertainty),
            assumptions=_slot(slots.assumptions),
            program_id=slots.program_id,
            lowered_program_id=getattr(slots, "lowered_program_id", None),
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
            next_actions=getattr(slots, "next_actions", ()),
            neighbors=getattr(slots, "neighbors", ()),
            unmet_obligations=getattr(slots, "unmet_obligations", ()),
            evidence_obligations=getattr(slots, "evidence_obligations", ()),
            target={"query": _target_query(contract)},
            inference_binding=contract.get("inference_binding") or slots.inference_binding_id,
        )
    return InspectionReport.model_validate(dict(slots))
