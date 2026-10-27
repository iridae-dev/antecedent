"""Owner-backed cyclic metadata pages; scientific records remain native."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from ._cyclic_typed import _with_execution
from ._freeze import freeze, thaw
from ._scientific import ScientificResult
from .errors import CausalTypeError, CausalValueError
from .execution import Execution


def page_bounds(offset: int, limit: int) -> None:
    for name, value in (("offset", offset), ("limit", limit)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise CausalTypeError(f"{name} must be an integer")
        if value < 0:
            raise CausalValueError(f"{name} must be nonnegative")
    if limit > 256:
        raise CausalValueError("cyclic page limit must be at most 256")


@dataclass(frozen=True, slots=True, repr=False)
class CyclicPage:
    """Explicitly selected native records and a bound source for the next page."""

    offset: int
    count: int
    items: tuple[Mapping[str, Any], ...] = field(repr=False)
    next_offset: int | None
    metadata: Mapping[str, Any] = field(repr=False)
    _read: Callable[..., str] = field(repr=False, compare=False)
    _key: str = field(repr=False, compare=False)
    _limit: int = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "items", tuple(freeze(item) for item in self.items))
        object.__setattr__(self, "metadata", freeze(self.metadata))

    def next(self, *, limit: int | None = None) -> CyclicPage | None:
        if self.next_offset is None:
            return None
        size = self._limit if limit is None else limit
        if size == 0:
            raise CausalValueError("advancing a cyclic page requires a positive limit")
        return native_page(self._read, self._key, self.next_offset, size)

    def to_dict(self) -> dict[str, Any]:
        return {
            **thaw(self.metadata),
            "offset": self.offset,
            "count": self.count,
            "items": thaw(self.items),
            "next_offset": self.next_offset,
        }

    def __repr__(self) -> str:
        return f"CyclicPage(offset={self.offset}, items={len(self.items)}, count={self.count}, next_offset={self.next_offset})"


def native_page(read: Callable[..., str], key: str, offset: int, limit: int) -> CyclicPage:
    page_bounds(offset, limit)
    raw = json.loads(read(offset=offset, limit=limit))
    count = raw.get("count", raw.get("column_count", 0))
    return CyclicPage(
        offset=raw.get("offset", offset),
        count=count,
        items=tuple(raw[key]),
        next_offset=raw.get("next_offset"),
        metadata={name: value for name, value in raw.items() if name != key},
        _read=read,
        _key=key,
        _limit=limit,
    )


def result_inspection(metadata: Mapping[str, Any], *, stage: Literal["executed", "consumed"]):
    from .outcomes import Answer
    from .results._report import InspectionReport, SlotModel

    contract = metadata.get("contract") or {}
    layers = contract.get("layers", {})
    identification = layers.get("identification")
    inference = layers.get("inference_binding")
    target = layers.get("target")
    graph = metadata.get("diagnostics", {}).get("graph") or {}
    assumptions = graph.get("assumptions")
    if assumptions is None and target is not None:
        assumptions = target.get("assumptions")
    unavailable = SlotModel(
        available=False, reason="not_retained_by_native_report", summary="unavailable"
    )
    identities = {
        f"{key}_id": value
        for key, value in contract.get("identities", {}).items()
        if value is not None
    }
    diagnostics = graph.get("diagnostics", ())
    return InspectionReport(
        identification=SlotModel(
            available=True, summary=contract["identification_kind"], payload=identification
        )
        if identification is not None
        else unavailable,
        support=unavailable,
        uncertainty=SlotModel(
            available=inference.get("method") != "none",
            reason="no_uncertainty_claim" if inference.get("method") == "none" else None,
            summary=str(inference.get("method")),
            payload=inference,
        )
        if inference is not None
        else unavailable,
        assumptions=SlotModel(
            available=True,
            summary="retained native assumption metadata",
            payload={"assumptions": assumptions},
        )
        if assumptions is not None
        else unavailable,
        contract=None,
        target=target,
        inference_binding=inference,
        answer=Answer("structured"),
        calibration=metadata.get("calibration"),
        diagnostics=tuple(item for item in diagnostics if isinstance(item, str)),
        details={**metadata, "contract_is_summary": bool(contract)},
        family=metadata["family"],
        stage=stage,
        lowered_program_id=contract.get("lowered_program_identity"),
        operations=(
            "inspect",
            "to_dict",
            "export",
            "assumptions",
            "interventions",
            "numeric",
            "evidence",
        ),
        **identities,
    )


@dataclass(frozen=True, slots=True, repr=False)
class RecoveredCyclicLaw(ScientificResult):
    """Retained experimental law, its draw origin, and original failed directions.

    New hard-clamp worlds prepare directly from this owner. The original
    uncertainty origin, failed slots, sample data and recovery derivation stay
    native and unchanged; preparation performs no further recovery fit.
    """

    _native: Any = field(repr=False)
    execution: Execution = field(repr=False, compare=False)
    _metadata: Mapping[str, Any] = field(repr=False)
    _stage: Literal["executed", "consumed"] = field(default="executed", repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_metadata", freeze(self._metadata))

    def _summary_fields(self) -> dict[str, Any]:
        return {
            "draw_count": self._metadata["draw_count"],
            "node_count": self._metadata["node_count"],
            "failed_mass": self._metadata["failed_mass"],
            "law_meaning": self._metadata["law_meaning"],
        }

    def inspect(self):
        from .outcomes import Answer
        from .results._report import InspectionReport, SlotModel

        unavailable = SlotModel(
            available=False, reason="not_established_by_recovered_law", summary="unavailable"
        )
        meaning = self._metadata["law_meaning"]
        return InspectionReport(
            identification=unavailable,
            support=SlotModel(
                available=True,
                summary="retained native failed mass",
                payload={"failed_mass": self._metadata["failed_mass"]},
            ),
            uncertainty=SlotModel(
                available=meaning is not None,
                summary=meaning or "supplied_draws_have_no_uncertainty_meaning",
                reason=None if meaning else "supplied_draws_have_no_uncertainty_meaning",
                payload={"draw_origin": self._metadata["draw_origin"]},
            ),
            assumptions=unavailable,
            answer=Answer("structured"),
            family="recovered_cyclic_law",
            stage=self._stage,
            details=self._metadata,
            operations=("inspect", "draw_status", "unresolved", "prepare", "to_dict"),
        )

    @_with_execution
    def prepare(
        self,
        worlds: Sequence[Any],
        *,
        variables: Sequence[str],
        outcome: str,
        execution: Execution | None = None,
    ):
        """Prepare hard-clamp worlds from this exact law without fitting it again.

        ``variables`` declares the original variable order, including dropped units.
        Native unit ids and draw positions retain their original alignment. A
        shift, soft mechanism or initial-state replacement requires another
        scientific bridge and keeps the native refusal. This is intervention
        evaluation, not factual abduction or a sampling-to-posterior conversion.
        """
        from ._cyclic_typed import _names, _worlds
        from .cyclic import PreparedCyclicFamily
        from .errors import CausalError
        from .outcomes import is_scientific_refusal, outcome_from_error

        assert execution is not None
        names = _names(variables)
        nodes = tuple(self._native.recovery_nodes())
        if any(node >= len(names) for node in nodes):
            raise CausalValueError("variables must preserve the original retained-unit ids")
        if not isinstance(outcome, str) or outcome not in names:
            raise CausalValueError("outcome must name a declared variable")
        outcome_id = names.index(outcome)
        if outcome_id not in nodes:
            raise CausalValueError("outcome must name a retained, unmasked unit")
        requested = tuple(worlds)
        wire = _worlds(requested, {name: index for index, name in enumerate(names)})
        try:
            handle = self._native.prepare_recovered_worlds(
                json.dumps(wire, allow_nan=False),
                outcome_id,
                max_bytes=None,
                cancel=None,
            )
        except CausalError as error:
            if not is_scientific_refusal(error):
                raise
            refusal = outcome_from_error(error, operation="recovered_cyclic_law.prepare").refusal
            return PreparedCyclicFamily._from_refused(
                None,
                names,
                refusal,
                execution,
                retained_law=(self, requested, names, outcome),
            )
        return PreparedCyclicFamily._from_retained_law(
            self,
            requested,
            names,
            outcome,
            handle,
            execution,
        )

    def draw_status(self, *, offset: int = 0, limit: int = 64) -> CyclicPage:
        return native_page(self._native.recovery_draw_status_json, "draw_status", offset, limit)

    def unresolved(self, draw: int, *, offset: int = 0, limit: int = 64) -> CyclicPage:
        from functools import partial

        if isinstance(draw, bool) or not isinstance(draw, int):
            raise CausalTypeError("draw must be an integer")
        if draw < 0:
            raise CausalValueError("draw must be nonnegative")
        read = partial(self._native.recovery_unresolved_json, draw)
        return native_page(read, "unresolved_directions", offset, limit)

    def to_dict(self, *, execution: Execution | None = None) -> dict[str, Any]:
        policy = self.execution if execution is None else execution
        if not isinstance(policy, Execution):
            raise CausalTypeError("execution must be an Execution")
        return json.loads(policy._call(self._native.recovery_law_report_json))
