"""One identification-status vocabulary for every transport binding.

Every transport stage and decision reports ``identification_status`` in the
canonical spellings of :data:`IDENTIFICATION_STATUSES` (``identified``,
``proven_non_transportable``, ``missing_evidence``, ``not_certified``,
``budget_cancel``), which are the Rust ``TransportOutcomeKind`` names. The
spellings bindings used before 2.2 stay where they are serialized or public — a
stage's ``outcome`` (``exhausted``, ``named_route``, ``combined_identified``), a
scenario row's ``status`` (``structurally_unidentified``, ``unevaluated``) — and
are deprecated: :func:`identification_status` reads each as its canonical
spelling. Reason codes are unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from .._native import transport_identification_statuses as _native_statuses
from ..errors import CausalValueError

_canonical, _legacy = _native_statuses()

#: Canonical transport identification statuses, strongest first.
IDENTIFICATION_STATUSES: tuple[str, ...] = tuple(_canonical)

#: Deprecated binding-local spellings, each with the canonical status it reads as.
LEGACY_IDENTIFICATION_STATUSES: Mapping[str, str] = MappingProxyType(dict(_legacy))


def identification_status(value: Any) -> str:
    """The canonical identification status of a transport spelling, stage or decision.

    ``value`` is a status string in either spelling, an object with an
    ``identification_status`` or ``outcome`` attribute (a transport stage or
    identification), or a decision mapping with either key. Execution statuses
    such as ``support_failure`` are not identification statuses and raise.
    """
    if isinstance(value, str):
        spelling: Any = value
    elif isinstance(value, Mapping):
        spelling = value.get("identification_status") or value.get("outcome")
    else:
        spelling = getattr(value, "identification_status", None)
        if not isinstance(spelling, str):
            spelling = getattr(value, "outcome", None)
    if spelling in IDENTIFICATION_STATUSES:
        return str(spelling)
    if spelling in LEGACY_IDENTIFICATION_STATUSES:
        return LEGACY_IDENTIFICATION_STATUSES[spelling]
    raise CausalValueError(f"unknown transport identification status {spelling!r}")


__all__ = ["IDENTIFICATION_STATUSES", "LEGACY_IDENTIFICATION_STATUSES", "identification_status"]
