"""The support decision of the unit a compiled program runs under.

Every compiled program resolves to one support unit, a capability record whose status comes
from the support model. A handle reads that decision from its program; reading it changes no
program identity.
"""

from __future__ import annotations

import json
from typing import Any

__all__ = ["read_unit_support"]


def read_unit_support(handle: Any | None) -> dict[str, Any] | None:
    """The support decision of the unit a compiled program runs under, as a dict.

    The keys are ``unit``, ``status``, ``supported`` and ``missing_evidence``: ``unit`` is the
    capability record id, ``status`` its support status, ``supported`` whether the unit is supported and
    ``missing_evidence`` the evidence an unsupported unit still lacks. ``None`` while no unit
    is bound to the program's route.
    """
    if handle is None:
        return None
    raw = handle.unit_support()
    return None if raw is None else dict(json.loads(raw))
