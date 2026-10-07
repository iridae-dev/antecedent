"""Shared structured-refusal base for the B4 vector, categorical and compact-export modules."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .errors import CausalUnsupportedError


class StructuredRefusal(CausalUnsupportedError):
    """A :class:`CausalUnsupportedError` carrying a registered ``reason_code`` and a ``detail``.

    ``reason_code`` is the registered refusal code, ``detail`` the namespaced ``family.slot``
    slot (for example ``vector_treatment.adjustment_set_mismatch`` or
    ``compact_export.out_of_support``), ``stage`` the refusing stage, ``offending`` the field,
    quantity or identity the refusal is about (when there is one) and ``remedy`` what the caller
    can change.
    """

    def __init__(self, refusal: Mapping[str, Any]) -> None:
        detail = str(refusal["detail"])
        message = refusal.get("message")
        text = f"{detail}: {message}" if message else detail
        offending = refusal.get("offending")
        if offending:
            text = f"{text} at {offending}"
        super().__init__(text, reason_code=refusal["code"], remedy=refusal.get("remedy"))
        #: Refusing stage.
        self.stage: str = refusal.get("stage", "")
        #: Namespaced ``family.slot`` detail.
        self.detail: str = detail
        #: The field, quantity or identity part the refusal is about, when there is one.
        self.offending: str | None = offending


def refusal_from_json(payload: str, cls: type[StructuredRefusal]) -> StructuredRefusal:
    """Build the structured refusal ``cls`` from a native refusal JSON string."""
    return cls(json.loads(payload))
