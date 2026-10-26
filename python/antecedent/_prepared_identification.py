"""Bounded navigation of a prepared study's retained native identification.

This reading is distinct from the standalone ``Identification`` declaration.
Composite products retain their own cases, horizons and posterior samples;
reading them never selects a representative proof or invents a single verdict.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import partial
from html import escape
from typing import Any

from ._freeze import freeze, thaw
from ._prepared_inspection import IdentificationPageKind, InspectionPage, _bounds, _read_page
from .errors import CausalStateError, CausalTypeError, CausalValueError
from .execution import Execution
from .outcomes import Refusal

__all__ = ["IdentificationSummary", "PreparedIdentification"]


def _position(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CausalTypeError(f"{name} must be an integer")
    if value < 0:
        raise CausalValueError(f"{name} must be nonnegative")


@dataclass(frozen=True, slots=True, repr=False)
class IdentificationSummary:
    """One node's bounded metadata, with no aggregate scientific inference.

    ``status`` is absent for products without one native aggregate status.
    ``status_unavailable`` explains that absence. ``metadata`` preserves native
    node-specific counts, masses, horizons and scope without combining them.
    """

    kind: str
    status: str | None
    status_unavailable: str | None
    method: str | None
    child_count: int
    metadata: Mapping[str, Any] = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", freeze(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Explicitly copy this metadata, without materializing any proof."""
        return thaw(self.metadata)

    def __repr__(self) -> str:
        return (
            f"IdentificationSummary(kind={self.kind[:80]!r}, status={str(self.status)[:80]!r}, "
            f"status_unavailable={str(self.status_unavailable)[:80]!r}, children={self.child_count})"
        )


@dataclass(frozen=True, slots=True, repr=False, eq=False)
class PreparedIdentification:
    """Immutable owner-backed identification node with bounded child paging.

    ``to_dict`` explicitly materializes the selected subtree. It can be large;
    native admission of that conversion remains separate from bounded reading.
    A child retains its native owner independently of the Python study.
    """

    _native: Any = field(repr=False, compare=False)
    execution: Execution = field(repr=False, compare=False)
    _unavailable: Mapping[str, Any] | None = field(default=None, repr=False, compare=False)
    refusal: Refusal | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.execution, Execution):
            raise CausalTypeError("execution must be an Execution declaration")
        if (self._native is None) != (self._unavailable is not None):
            raise CausalValueError("identification requires a native owner or unavailable metadata")
        object.__setattr__(self, "_unavailable", freeze(self._unavailable))

    @classmethod
    def from_native(cls, native: Any, execution: Execution) -> PreparedIdentification:
        """Retain the supplied immutable native view without converting its proof."""
        if native is None:
            raise CausalTypeError("a native identification view is required")
        return cls(native, execution)

    @classmethod
    def unavailable(
        cls,
        reason: str,
        *,
        kind: str = "unavailable",
        method: str | None = None,
        refusal: Refusal | None = None,
        details: Mapping[str, Any] | None = None,
        execution: Execution | None = None,
    ) -> PreparedIdentification:
        """Represent explicitly absent native proof, preserving a scientific refusal.

        This is for refused or externally declared preparations, never an
        exception fallback for a missing or failing native implementation.
        """
        if not isinstance(reason, str) or not reason:
            raise CausalValueError("unavailable identification requires a reason")
        if refusal is not None and not isinstance(refusal, Refusal):
            raise CausalTypeError("refusal must be a Refusal or None")
        return cls(
            None,
            Execution() if execution is None else execution,
            {
                "kind": kind,
                "status": None,
                "status_unavailable": reason,
                "method": method,
                "children": 0,
                "details": details or {},
            },
            refusal,
        )

    @property
    def summary(self) -> IdentificationSummary:
        if self._native is None:
            assert self._unavailable is not None
            raw = {key: value for key, value in self._unavailable.items() if key != "details"}
            raw["detail_count"] = len(self._unavailable["details"])
        else:
            raw = json.loads(self._native.summary_json())
        return IdentificationSummary(
            kind=raw["kind"],
            status=raw.get("status"),
            status_unavailable=raw.get("status_unavailable"),
            method=raw.get("method"),
            child_count=raw["children"],
            metadata=raw,
        )

    @property
    def kind(self) -> str:
        return self.summary.kind

    @property
    def status(self) -> str | None:
        return self.summary.status

    @property
    def status_unavailable(self) -> str | None:
        return self.summary.status_unavailable

    @property
    def method(self) -> str | None:
        return self.summary.method

    @property
    def child_count(self) -> int:
        return self.summary.child_count

    def child(self, index: int) -> PreparedIdentification:
        """Select one retained child, without reading its siblings or fitting."""
        _position(index, "index")
        if self._native is None:
            raise CausalValueError("unavailable identification has no retained children")
        return type(self).from_native(self._native.child(index), self.execution)

    def children(self, *, offset: int = 0, limit: int = 32) -> tuple[PreparedIdentification, ...]:
        """Read at most 256 immediate children at their original native positions."""
        _position(offset, "offset")
        _position(limit, "limit")
        if limit > 256:
            raise CausalValueError("identification page limit must be at most 256")
        if self._native is None:
            return ()
        return tuple(
            type(self).from_native(child, self.execution)
            for child in self._native.children(offset=offset, limit=limit)
        )

    def _detail_page(
        self,
        kind: IdentificationPageKind,
        method: str,
        offset: int,
        limit: int,
        *,
        estimand: int | None = None,
        selects_estimand: bool = False,
    ) -> InspectionPage:
        _bounds(offset, limit)
        if estimand is not None:
            _position(estimand, "estimand")
        if self._native is None:
            if self.refusal is not None:
                self.refusal.raise_error()
            raise CausalStateError("unavailable identification has no retained proof records")
        read = getattr(self._native, method, None)
        if not callable(read):
            raise CausalStateError("this identification has no native detail-page adapter")
        if selects_estimand:
            # The bound reader retains this exact native proof and selection for
            # later pages. Native support decides composite/ambiguous requests.
            read = partial(read, estimand=estimand)
        return _read_page(kind, read, offset, limit)

    def estimands(self, *, offset: int = 0, limit: int = 32) -> InspectionPage:
        """Page native alternatives on an explicitly selected proof.

        Functional IDs belong to that proof's arena. Composite nodes require
        explicit child selection; no first proof is chosen automatically.
        """
        return self._detail_page(
            "identification_estimands",
            "estimands_json",
            offset,
            limit,
        )

    def adjustment(
        self,
        *,
        estimand: int | None = None,
        offset: int = 0,
        limit: int = 32,
    ) -> InspectionPage:
        """Page named native coordinates; ambiguous alternatives require an index.

        Temporal offsets use the proof's retained indexer. Missing namespaces
        remain explicitly unavailable instead of treating dense IDs as names.
        """
        return self._detail_page(
            "identification_adjustment",
            "adjustment_json",
            offset,
            limit,
            estimand=estimand,
            selects_estimand=True,
        )

    def assumptions(
        self,
        *,
        estimand: int | None = None,
        offset: int = 0,
        limit: int = 32,
    ) -> InspectionPage:
        """Page ordered proof assumptions or one native estimand claim.

        The returned metadata preserves native scope, status and estimand.
        A missing claim never borrows another alternative's assumptions.
        """
        return self._detail_page(
            "identification_assumptions",
            "assumptions_json",
            offset,
            limit,
            estimand=estimand,
            selects_estimand=True,
        )

    def derivation(self, *, offset: int = 0, limit: int = 32) -> InspectionPage:
        """Page the selected proof's original ordered rules and explanations."""
        return self._detail_page(
            "identification_derivation",
            "derivation_json",
            offset,
            limit,
        )

    def inspect(self) -> IdentificationSummary:
        """A bounded metadata snapshot; no proof conversion or execution."""
        return self.summary

    def to_dict(self, *, execution: Execution | None = None) -> dict[str, Any]:
        """Materialize this complete native subtree and its schema names explicitly.

        Cancellation follows the selected Execution. This operation does not
        establish that the native JSON allocation is fully resource-admitted.
        """
        policy = self.execution if execution is None else execution
        if not isinstance(policy, Execution):
            raise CausalTypeError("execution must be an Execution declaration")
        if self._native is None:
            assert self._unavailable is not None
            return {
                "summary": self.summary.to_dict(),
                "details": thaw(self._unavailable["details"]),
                "children": [],
            }
        payload = json.loads(
            policy._call(self._native.materialize_json, seed=policy.seed, threads=policy.threads)
        )
        payload["names"] = self._native.schema_names()
        return payload

    def __bool__(self) -> bool:
        raise CausalTypeError(
            "read status and status_unavailable; identification has no boolean verdict"
        )

    def __repr__(self) -> str:
        summary = self.summary
        return f"PreparedIdentification(kind={summary.kind[:80]!r}, status={str(summary.status)[:80]!r}, children={summary.child_count})"

    def _repr_html_(self) -> str:
        summary = self.summary
        standing = summary.status if summary.status is not None else summary.status_unavailable
        return (
            f"<div><strong>Prepared identification · {escape(summary.kind[:80])}</strong>"
            f"<p>{escape(str(standing)[:160])}; {summary.child_count} retained children</p></div>"
        )
