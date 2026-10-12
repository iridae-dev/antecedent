from __future__ import annotations

from ...errors import CausalUnsupportedError


def _no_native_authority() -> CausalUnsupportedError:
    """A display object edited or rebuilt away from its native execution cannot act."""
    return CausalUnsupportedError(
        "This display object has no native execution authority", reason_code="not_executed"
    )
