"""Bounded native prepared metadata and explicitly requested inspection pages."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from ._freeze import freeze, thaw
from .errors import CausalStateError, CausalTypeError, CausalValueError

if TYPE_CHECKING:
    from .results._report import InspectionReport, SlotModel

PreparedPageKind = Literal[
    "assumption_obligations", "uncertainty_components", "evidence_obligations", "evidence_actions"
]

IdentificationPageKind = Literal[
    "identification_estimands",
    "identification_adjustment",
    "identification_assumptions",
    "identification_derivation",
]
PageKind = PreparedPageKind | IdentificationPageKind

_PAGE_METHODS: Mapping[PreparedPageKind, str] = {
    "assumption_obligations": "prepared_obligations_json",
    "uncertainty_components": "prepared_uncertainty_json",
    "evidence_obligations": "prepared_evidence_obligations_json",
    "evidence_actions": "prepared_evidence_actions_json",
}


def _position(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CausalTypeError(f"{name} must be an integer")
    if value < 0:
        raise CausalValueError(f"{name} must be nonnegative")


def _bounds(offset: int, limit: int) -> None:
    _position(offset, "offset")
    _position(limit, "limit")
    if limit > 256:
        raise CausalValueError("inspection page limit must be at most 256")


@dataclass(frozen=True, slots=True, repr=False, eq=False)
class InspectionPage:
    """One immutable native page; its bound reader keeps the source owner alive.

    Creating a page explicitly converts its selected records. Representing it
    never renders records or fetches another page. ``next`` is an explicit read.
    """

    kind: PageKind
    offset: int
    total: int | None
    limit: int
    items: tuple[Mapping[str, Any], ...] = field(repr=False)
    next_offset: int | None
    _read: Callable[..., str] = field(repr=False, compare=False)
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        _bounds(self.offset, self.limit)
        if self.total is not None:
            _position(self.total, "total")
        elif self.metadata.get("available") is not False:
            raise CausalStateError("native inspection page has no collection count")
        if len(self.items) > self.limit or not all(
            isinstance(item, Mapping) for item in self.items
        ):
            raise CausalStateError("native inspection page has invalid records")
        if self.next_offset is not None:
            _position(self.next_offset, "next_offset")
        object.__setattr__(self, "items", tuple(freeze(item) for item in self.items))
        object.__setattr__(self, "metadata", freeze(self.metadata))

    @property
    def available(self) -> bool:
        """Native collection availability; an absent collection has no known count."""
        return self.metadata.get("available", True) is not False

    @property
    def reason(self) -> str | None:
        """Native explanation for an unavailable collection, when retained."""
        return self.metadata.get("reason")

    def next(self, *, limit: int | None = None) -> InspectionPage | None:
        """Explicitly fetch the next retained page, or return None at the end."""
        if self.next_offset is None:
            return None
        size = self.limit if limit is None else limit
        _bounds(self.next_offset, size)
        if size == 0:
            raise CausalValueError("advancing an inspection page requires a positive limit")
        return _read_page(self.kind, self._read, self.next_offset, size)

    def to_dict(self) -> dict[str, Any]:
        """Explicitly copy this already requested page; never fetch more records."""
        return {
            "kind": self.kind,
            "offset": self.offset,
            "total": self.total,
            "items": thaw(self.items),
            "next_offset": self.next_offset,
            **thaw(self.metadata),
        }

    def __repr__(self) -> str:
        return (
            f"InspectionPage(kind={self.kind!r}, offset={self.offset}, "
            f"count={len(self.items)}, total={self.total}, next_offset={self.next_offset})"
        )

    def _repr_html_(self) -> str:
        from html import escape

        return f"<pre>{escape(repr(self))}</pre>"


def _read_page(kind: PageKind, read: Callable[..., str], offset: int, limit: int) -> InspectionPage:
    _bounds(offset, limit)
    raw = json.loads(read(offset=offset, limit=limit))
    if not isinstance(raw, Mapping):
        raise CausalStateError("native inspection page has an invalid header")
    if raw.get("available") is False:
        # Native absence is different from a successfully inspected empty set.
        # Keep its reason/selection and an unknown count, without fabricating
        # proof records or trying another scientific method.
        return InspectionPage(
            kind=kind,
            offset=offset,
            total=None,
            limit=limit,
            items=(),
            next_offset=None,
            _read=read,
            metadata=raw,
        )
    if raw.get("offset") != offset:
        raise CausalStateError("native inspection page has an invalid header")
    return InspectionPage(
        kind=kind,
        offset=raw["offset"],
        total=raw["total"],
        limit=limit,
        items=tuple(raw["items"]),
        next_offset=raw["next_offset"],
        _read=read,
        metadata={
            key: value
            for key, value in raw.items()
            if key not in {"offset", "total", "items", "next_offset"}
        },
    )


def inspection_page(study: Any, kind: PreparedPageKind, offset: int, limit: int) -> InspectionPage:
    _bounds(offset, limit)
    native = getattr(study._engine, "_native", None)
    read = getattr(native, _PAGE_METHODS[kind], None)
    if not callable(read):
        raise CausalStateError("this preparation has no native adapter for inspection pages")
    return _read_page(kind, read, offset, limit)


def _slot(name: str, raw: Mapping[str, Any], *, retained: Any = None) -> SlotModel:
    from .results._report import SlotModel

    # Native availability/status is authoritative. A count or a successful
    # preparation never establishes an empirical/statistical/scientific verdict.
    summary = raw.get("status", raw.get("matrix_status", raw.get("reason", name)))
    payload = dict(raw)
    if retained is not None:
        payload["retained"] = retained
    return SlotModel(
        available=raw["available"],
        reason=raw.get("reason"),
        summary=str(summary),
        payload=payload,
    )


def inspect_prepared(study: Any) -> InspectionReport:
    from .results._execution import CalibrationInfo
    from .results._report import InspectionReport

    native = getattr(study._engine, "_native", None)
    read = getattr(native, "prepared_inspection_json", None)
    if not callable(read):
        raise CausalStateError("this preparation has no native prepared-inspection adapter")
    raw = json.loads(read())
    identities = raw["identities"]
    retained = study.identification.inspect().metadata
    details = {
        key: value
        for key, value in raw.items()
        if key not in {"identities", "identification", "support", "uncertainty", "assumptions"}
    }
    details["snapshot_unavailable"] = identities.get("snapshot_unavailable")
    # These counts describe retained collections. Empty default arrays in the
    # common report have not materialized them; the canonical page methods do.
    details["collections"] = {
        "assumption_obligations": {
            "materialized": False,
            "count": raw["assumptions"]["obligation_count"],
        },
        "uncertainty_components": {
            "materialized": False,
            "count": raw["uncertainty"]["component_count"],
        },
        "evidence_obligations": {"materialized": False, "count": raw["evidence_obligation_count"]},
        "evidence_actions": {"materialized": False, "count": raw["evidence_action_count"]},
    }
    return InspectionReport(
        identification=_slot("identification", raw["identification"], retained=retained),
        support=_slot("support", raw["support"]),
        uncertainty=_slot("uncertainty", raw["uncertainty"]),
        assumptions=_slot("assumptions", raw["assumptions"]),
        target_id=identities["target_id"],
        identification_id=identities["identification_id"],
        identification_product_id=identities["identification_product_id"],
        program_id=identities["program_id"],
        inference_binding_id=identities["inference_binding_id"],
        observation_id=identities["observation_id"],
        data_snapshot_id=identities["data_snapshot_id"],
        contract=None,
        calibration=CalibrationInfo.from_contract(None),
        stage="prepared",
        family=raw["graph_class"],
        details=details,
        operations=(
            "inspect",
            "estimate",
            "rebind",
            "reconfigure",
            "assumption_obligations",
            "uncertainty_components",
            "evidence_obligations",
            "evidence_actions",
        ),
    )
