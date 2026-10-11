"""Native-refusal-JSON helpers for the structured refusals; the base class lives in ``errors``."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

from .errors import StructuredRefusal

__all__ = ["StructuredRefusal", "raise_refusal", "refusal_from_json"]

#: A refusal class, or a function choosing one from the decoded refusal mapping.
RefusalClass = type | Callable[[Mapping[str, Any]], type]


def refusal_from_json(payload: str | Mapping[str, Any], cls: RefusalClass) -> StructuredRefusal:
    """Build the refusal ``cls`` from a native refusal JSON string (or its decoded mapping).

    ``cls`` is either the refusal class itself or a function of the decoded mapping that returns
    it, for the modules whose class depends on the refusal's ``detail``.
    """
    wire = json.loads(payload) if isinstance(payload, str) else payload
    target = cls if isinstance(cls, type) else cls(wire)
    refusal: StructuredRefusal = target(wire)
    return refusal


def raise_refusal(payload: str | None, cls: RefusalClass) -> None:
    """Raise the refusal in the native refusal JSON ``payload``, when there is one."""
    if payload is not None:
        raise refusal_from_json(payload, cls)
