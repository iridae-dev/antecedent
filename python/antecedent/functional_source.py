"""Original finite-law functional requests and independently replayed derivation."""

from __future__ import annotations

import json
from typing import Any

from . import _native


class LawFunctionalArtifact:
    """Complete original law and functional request; replay issues no native authority."""

    __slots__ = ("_bytes", "_report")

    def __init__(self, data: bytes, report: str):
        self._bytes = bytes(data)
        self._report: dict[str, Any] = json.loads(report)

    @classmethod
    def _produce(
        cls, contract: Any, action: str, functional: Any, source: Any
    ) -> LawFunctionalArtifact:
        report, data = _native.export_law_functional_source(
            json.dumps(contract._wire()),
            action,
            json.dumps(functional._wire()),
            source._native_value,
        )
        return cls(bytes(data), report)

    @classmethod
    def consume(cls, data: bytes, *, expected_identity: str | None = None) -> LawFunctionalArtifact:
        return cls(data, _native.consume_law_functional_source(data, expected_identity))

    @property
    def identity(self) -> str:
        return str(self._report["identity"])

    def resolve_with(self, actual: Any) -> LawFunctionalArtifact:
        """Compare full rows and original source to a freshly issued opaque native claim."""
        from .program_claims import NativeClaim

        if not isinstance(actual, NativeClaim):
            raise TypeError("native law source resolution requires an issued native claim")
        return LawFunctionalArtifact(
            self._bytes, _native.resolve_law_functional_source(self._bytes, actual._native)
        )

    @property
    def report(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._report))

    def export(self) -> bytes:
        return self._bytes


__all__ = ["LawFunctionalArtifact"]
