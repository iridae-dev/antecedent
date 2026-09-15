"""Single identification verdict table keyed by exact Rust Debug status strings."""

from __future__ import annotations

import warnings

VERDICTS: dict[str, str] = {
    "NonparametricallyIdentified": "identified",
    "IdentifiedUnderParametricRestrictions": "identified",
    "IdentifiedUnderPriorRestrictions": "identified",
    "PartiallyIdentified": "partially identified",
    "GraphDependent": "graph-dependent",
    "NotIdentified": "not identified",
}


def verdict_for(status: str) -> str:
    """Exact lookup. Unknown statuses warn and close as not identified."""
    verdict = VERDICTS.get(status)
    if verdict is not None:
        return verdict
    warnings.warn(
        f"unrecognized identification status {status!r}; treating as not identified",
        RuntimeWarning,
        stacklevel=2,
    )
    return "not identified"
