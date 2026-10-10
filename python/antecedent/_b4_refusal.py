"""Native-refusal-JSON helper for the structured refusals; the base class lives in ``errors``."""

from __future__ import annotations

import json

from .errors import StructuredRefusal

__all__ = ["StructuredRefusal", "refusal_from_json"]


def refusal_from_json(payload: str, cls: type[StructuredRefusal]) -> StructuredRefusal:
    """Build the structured refusal ``cls`` from a native refusal JSON string."""
    return cls(json.loads(payload))
