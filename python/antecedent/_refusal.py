"""Native-refusal-JSON helpers for the structured refusals; the base class lives in ``errors``."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

from .errors import StructuredRefusal

__all__ = ["StructuredRefusal", "raise_refusal"]

#: A refusal class, or a function choosing one from the decoded refusal mapping.
RefusalClass = type | Callable[[Mapping[str, Any]], type]


def raise_refusal(payload: str | None, cls: RefusalClass) -> None:
    """Raise the refusal in the native refusal JSON ``payload``, when there is one.

    ``cls`` is either the refusal class itself or a function of the decoded mapping that returns
    it, for the modules whose class depends on the refusal's ``detail``.
    """
    if payload is None:
        return
    wire = json.loads(payload)
    target = cls if isinstance(cls, type) else cls(wire)
    refusal: StructuredRefusal = target(wire)
    raise refusal
