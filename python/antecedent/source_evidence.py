"""Original scoped diagnostics and independently consumed source lineage.

Imported evidence retains its original claim; it supplies no executable state,
provider trust upgrade, transformed local support value or calibration license.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from . import _native
from .decision import Contract
from .joint_distribution import ScientificQuantity


@dataclass(frozen=True, slots=True)
class ScopedDiagnostic:
    id: str
    detail: str
    scope: str
    local_value: float | None
    global_values: tuple[float | None, ...]
    source_coordinate: ScientificQuantity


class SourceEvidence:
    """Producer diagnostics plus source bytes; metadata cannot issue native authority."""

    __slots__ = ("_native_handle", "_resolver", "_cache")

    def __init__(self, handle: _native.SourceEvidenceHandle):
        if not isinstance(handle, _native.SourceEvidenceHandle):
            raise TypeError("SourceEvidence requires a native validated evidence handle")
        self._native_handle: _native.SourceEvidenceHandle | None = handle
        self._resolver: Callable[[], _native.SourceEvidenceHandle] | None = None
        self._cache: dict[str, Any] | None = None

    @classmethod
    def _deferred(cls, resolver: Callable[[], _native.SourceEvidenceHandle]) -> SourceEvidence:
        result = cls.__new__(cls)
        result._native_handle, result._resolver, result._cache = None, resolver, None
        return result

    @property
    def _handle(self) -> _native.SourceEvidenceHandle:
        if self._native_handle is None:
            assert self._resolver is not None
            handle = self._resolver()
            if not isinstance(handle, _native.SourceEvidenceHandle):
                raise TypeError("native source evidence issuer returned no validated handle")
            self._native_handle, self._resolver = handle, None
        return self._native_handle

    @property
    def _summary(self) -> dict[str, Any]:
        if self._cache is None:
            self._cache = json.loads(self._handle.summary_json)
        return self._cache

    @property
    def source_artifact_digest(self) -> str:
        return str(self._summary["source_artifact_digest"])

    @property
    def original_acceptance(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._summary["original_acceptance"]))

    @property
    def resolution(self) -> dict[str, Any] | None:
        return json.loads(json.dumps(self._summary["resolution"]))

    def resolve_with(self, actual: Any) -> SourceEvidence:
        from .program_claims import NativeClaim

        if not isinstance(actual, NativeClaim):
            raise TypeError("source resolution requires an issued native claim")
        return SourceEvidence(self._handle.resolve_with(actual._native))

    @property
    def coordinates(self) -> tuple[ScientificQuantity, ...]:
        return tuple(ScientificQuantity._from_wire(q) for q in self._summary["coordinates"])

    @property
    def identities(self) -> dict[str, Any]:
        return dict(self._summary["source_identities"])

    @property
    def lineage(self):
        """Typed original derivation links; unresolved source status stays explicit."""
        from .external import LineageLink

        return tuple(
            LineageLink(
                item["id"],
                item["stage"],
                tuple(item["parents"]),
                item["digest"],
                tuple(item["parent_digests"]),
            )
            for item in self._summary["lineage"]
        )

    def stages_behind(self, link: str = "claim") -> frozenset[str]:
        by_id = {item.id: item for item in self.lineage}
        if link not in by_id:
            raise ValueError(f"unknown source lineage link {link!r}")
        seen: set[str] = set()
        stack = [link]
        while stack:
            current = stack.pop()
            if current not in seen:
                seen.add(current)
                stack.extend(by_id[current].parents)
        return frozenset(by_id[item].stage for item in seen)

    @property
    def provenance_id(self) -> str:
        return str(self._summary["provenance_id"])

    @property
    def diagnostics(self) -> tuple[dict[str, Any], ...]:
        return tuple(json.loads(json.dumps(value)) for value in self._summary["diagnostics"])

    @property
    def action_contributors(self) -> dict[str, tuple[ScientificQuantity, ...]]:
        return {
            value["action_id"]: tuple(
                ScientificQuantity._from_wire(q) for q in value["source_coordinates"]
            )
            for value in self._summary["action_contributors"]
        }

    def diagnostics_at(self, quantity: ScientificQuantity) -> tuple[ScopedDiagnostic, ...]:
        values = json.loads(self._handle.diagnostics_at(json.dumps(quantity._wire())))
        return tuple(
            ScopedDiagnostic(
                value["id"],
                value["detail"],
                value["scope"],
                value["local_value"],
                tuple(value["global_values"]),
                ScientificQuantity._from_wire(value["source_coordinate"]),
            )
            for value in values
        )

    def project(self, contract: Contract, actions: Sequence[str] | None = None) -> SourceEvidence:
        ids = [action.id for action in contract.actions] if actions is None else list(actions)
        return SourceEvidence._deferred(
            lambda: self._handle.project(json.dumps(contract._wire()), ids)
        )

    def original_artifact(self) -> bytes:
        return bytes(self._handle.original_artifact())

    def export(self) -> bytes:
        return bytes(self._handle.export())

    @classmethod
    def consume(cls, artifact: bytes) -> SourceEvidence:
        return cls(_native.SourceEvidenceHandle.consume(artifact))


__all__ = ["ScopedDiagnostic", "SourceEvidence"]
