"""Durable format-0.3 artifacts for causal queries and results.

The payload mapping is the canonical Rust wire representation.  This module only
adds the versioned container and variable-id/name table; it does not maintain a
second Python JSON schema.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

from ._native import (
    decode_causal_artifact as _decode_causal_artifact,
)
from ._native import (
    encode_causal_artifact as _encode_causal_artifact,
)

PayloadKind = Literal[
    "query",
    "response_result",
    "transport_identification",
    "transport_estimate",
    "interference_estimate",
]


@dataclass(frozen=True, slots=True)
class CausalArtifact:
    """Decoded causal payload plus its format and variable-name table."""

    artifact_id: str
    format_version: tuple[int, int]
    payload_kind: PayloadKind
    variable_names: Sequence[str]
    payload: Mapping[str, Any]


def dumps(
    payload_kind: PayloadKind,
    payload: Mapping[str, Any],
    *,
    variable_names: Sequence[str],
    artifact_id: str,
) -> bytes:
    """Encode one canonical query/result wire mapping as a format-0.3 artifact."""

    # `_encode_causal_artifact` now returns real Python `bytes` at the Rust
    # boundary (PyO3 `PyBytes`), so no `bytes(...)` coercion is needed here.
    return _encode_causal_artifact(
        payload_kind,
        list(variable_names),
        json.dumps(payload, allow_nan=False, separators=(",", ":")),
        artifact_id,
    )


def loads(data: bytes) -> CausalArtifact:
    """Decode and migrate a causal query/result artifact to the stable format."""

    decoded = _decode_causal_artifact(data)
    payload = json.loads(decoded.payload_json)
    if not isinstance(payload, dict):  # canonical wire payloads are object-shaped
        raise ValueError("decoded causal artifact payload is not an object")
    return CausalArtifact(
        decoded.artifact_id,
        (decoded.format_major, decoded.format_minor),
        cast(PayloadKind, decoded.payload_kind),
        tuple(decoded.variable_names),
        payload,
    )


__all__ = ["CausalArtifact", "PayloadKind", "dumps", "loads"]
