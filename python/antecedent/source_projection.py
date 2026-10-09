"""Portable original-source projections; numerical replay supplies no native authority."""

from __future__ import annotations

import json
from typing import Any, Literal

from . import _native
from .decision import Contract
from .source_evidence import SourceEvidence

Projection = Literal["causal_contract", "attestation", "quantity_coordinates", "transformation"]


class SourceProjectionArtifact:
    """Original source plus replayed semantics, exact quantities or affine utilities.

    Native source dependencies remain unresolved until bundle consumption supplies
    an actual issued native claim. Utility units are declared by the caller.
    """

    __slots__ = ("_bytes", "_summary")

    def __init__(self, data: bytes, summary: str):
        self._bytes = bytes(data)
        self._summary: dict[str, Any] = json.loads(summary)

    @classmethod
    def produce(
        cls,
        source: SourceEvidence | bytes,
        projection: Projection,
        *,
        contract: Contract | None = None,
        action_id: str | None = None,
    ) -> SourceProjectionArtifact:
        """Replay a native source or original external claim; bind the operation."""
        if projection == "transformation":
            if contract is None:
                raise TypeError("affine transformation requires a decision Contract")
            declaration: Any = {"affine_decision": {"contract": json.dumps(contract._wire())}}
            if action_id is not None:
                declaration = {
                    "affine_functional": {
                        "contract": json.dumps(contract._wire()),
                        "action_id": action_id,
                    }
                }
        elif projection in ("causal_contract", "attestation", "quantity_coordinates"):
            if contract is not None or action_id is not None:
                raise TypeError("only a transformation takes a decision Contract")
            declaration = projection
        else:
            raise ValueError("unknown source projection")
        native = (
            isinstance(source, SourceEvidence)
            and source._summary.get("source_kind") != "external_bound_claim"
        )
        if not isinstance(source, (SourceEvidence, bytes)):
            raise TypeError("external source must be original claim artifact bytes")
        data = source.original_artifact() if isinstance(source, SourceEvidence) else source
        summary, encoded = _native.produce_source_projection(data, native, json.dumps(declaration))
        return cls(bytes(encoded), summary)

    @property
    def identity(self) -> str:
        return str(self._summary["identity"])

    @property
    def kind(self) -> str:
        return str(self._summary["kind"])

    @property
    def report(self) -> dict[str, Any]:
        """Recomputed descriptive report; no execution or calibration license."""
        return dict(json.loads(json.dumps(self._summary["report"])))

    def export(self) -> bytes:
        return self._bytes


def consume_source_projection(data: bytes, *, expected_identity: str) -> SourceProjectionArtifact:
    """Replay the original consumer under an independently retained identity."""
    if not isinstance(data, bytes):
        raise TypeError("source projection artifact must be bytes")
    return SourceProjectionArtifact(
        data, _native.consume_source_projection(data, expected_identity)
    )


__all__ = ["SourceProjectionArtifact", "consume_source_projection"]
