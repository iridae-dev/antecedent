"""Single identification renderer keyed by exact Rust Debug status strings.

Every human-readable identification label — ``Identification.verdict`` /
``statement``, ``IdentificationView`` / ``AnalysisResult`` reprs, and the
notebook HTML banner — comes from this module. There is no second table.
"""

from __future__ import annotations

import warnings
from typing import Literal

#: Four-value verdict per exact native status string.
VERDICTS: dict[str, str] = {
    "NonparametricallyIdentified": "identified",
    "IdentifiedUnderParametricRestrictions": "identified",
    "IdentifiedUnderPriorRestrictions": "identified",
    "PartiallyIdentified": "partially identified",
    "GraphDependent": "graph-dependent",
    "NotIdentified": "not identified",
}

#: Restriction that an identified verdict holds under. Priors never upgrade
#: identification and a parametric restriction is part of the claim, so an
#: identified label never drops it.
QUALIFIERS: dict[str, str] = {
    "IdentifiedUnderParametricRestrictions": "under parametric restrictions",
    "IdentifiedUnderPriorRestrictions": "under prior restrictions",
}

Tone = Literal["identified", "caution", "not_identified"]

_TONES: dict[str, Tone] = {
    "identified": "identified",
    "partially identified": "caution",
    "graph-dependent": "caution",
    "not identified": "not_identified",
}


def _warn_unknown(status: str) -> None:
    warnings.warn(
        f"unrecognized identification status {status!r}; treating as not identified",
        RuntimeWarning,
        stacklevel=3,
    )


def verdict_for(status: str) -> str:
    """Exact lookup. Unknown statuses warn and close as not identified."""
    verdict = VERDICTS.get(status)
    if verdict is not None:
        return verdict
    _warn_unknown(status)
    return "not identified"


def describe_status(status: str) -> str:
    """Verdict plus its restriction qualifier, e.g. ``identified under parametric restrictions``."""
    verdict = VERDICTS.get(status)
    if verdict is None:
        _warn_unknown(status)
        return "not identified"
    qualifier = QUALIFIERS.get(status)
    return f"{verdict} {qualifier}" if qualifier else verdict


def verdict_tone(status: str) -> Tone:
    """Display tone: identified, caution (partial / graph-dependent), or not identified."""
    verdict = VERDICTS.get(status)
    if verdict is None:
        _warn_unknown(status)
        return "not_identified"
    return _TONES[verdict]
