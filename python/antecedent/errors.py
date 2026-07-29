"""The exception surface: ``CausalError`` and its concrete subclasses.

Most exception types are defined in the native extension (``antecedent._native``)
and re-exported here unchanged. This module is the single place the frozen
public surface imports errors from.

``ReviewRequired`` is a real class defined here, subclassing the native
``CausalReviewError`` — so ``except antecedent.ReviewRequired`` and
``except antecedent.errors.CausalReviewError`` both still catch it. It is
registered with the native layer at import time (see
``_native.set_review_error_class``), which instantiates it for every
``CausalError::ReviewRequired`` raised from Rust; the two hand-rolled
``CausalReviewError`` construction sites in ``estimation.py`` are built the
same way, through ``build_review_error`` below. A raised review error carries:

- ``kind`` — which review gate tripped (e.g. ``"static_cpdag"``, ``"static_pag"``)
- ``algorithm`` — the discovery algorithm that produced the pending graph
- ``pending_edge_count`` — how many edges still need orientation review
- ``pending_edges`` — the actual pending edges, as a ``tuple[PendingEdge, ...]``
- ``hint`` — a human-readable suggestion for resolving the review
- ``message`` — the formatted error message
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ._native import (
    CausalAttributionError,
    CausalCancelledError,
    CausalCompileError,
    CausalCounterfactualError,
    CausalDataError,
    CausalDesignError,
    CausalDiscoveryError,
    CausalError,
    CausalEstimateError,
    CausalGraphError,
    CausalIdentifyError,
    CausalModelError,
    CausalResourceError,
    CausalReviewError,
    CausalSerializationError,
    CausalStateError,
    CausalUnsupportedError,
    CausalValidateError,
)
from ._native import set_review_error_class as _set_review_error_class


@dataclass(frozen=True, slots=True)
class PendingEdge:
    """One unreviewed edge from a `ReviewRequired`, with its endpoint marks.

    ``at_source`` / ``at_target`` are one of ``"tail"``, ``"arrow"``,
    ``"circle"``, or ``"conflict"`` — the same vocabulary ``GraphEdge`` uses
    elsewhere in this package. ``source`` / ``target`` are display identifiers
    for the endpoints (a dense variable id such as ``"V3"``, or a temporal key
    such as ``"V3@-1"``) rather than resolved schema names — see
    ``antecedent::error::PendingEdge`` in the Rust facade for why name
    resolution is out of scope this close to the graph structures.
    """

    source: str
    target: str
    at_source: str
    at_target: str


class ReviewRequired(CausalReviewError):
    """Raised when estimation is blocked on an incomplete graph review.

    Subclasses the native ``CausalReviewError`` so existing ``except
    CausalReviewError`` handlers keep working unchanged. Carries the
    structured attributes documented on the module docstring; ``pending_edges``
    is always a ``tuple[PendingEdge, ...]`` whose length equals
    ``pending_edge_count`` whenever the raising site had real edges in hand —
    see ``build_review_error`` and ``python/src/lib.rs``'s
    ``review_required_py_err`` for the two places that construct one.
    """


# Registers this class with the native layer so `CausalError::ReviewRequired`
# raised from Rust instantiates it (see `review_required_py_err` in
# `python/src/lib.rs`) instead of falling back to a bare `CausalReviewError`.
_set_review_error_class(ReviewRequired)


def build_review_error(
    message: str,
    *,
    kind: str,
    algorithm: str | None,
    pending_edge_count: int,
    hint: str,
    pending_edges: Sequence[PendingEdge] = (),
) -> ReviewRequired:
    """Construct a `ReviewRequired` with the standard structured attributes.

    The one Python-side construction path for a review-required error raised
    without going through the native discovery mapper — every such call site
    (see ``estimation.py``) should build its error through this function
    rather than hand-rolling ``ReviewRequired(...)`` plus a run of
    ``setattr`` calls. The native mapper (``python/src/lib.rs``) attaches the
    same attribute set for errors raised from Rust, so a caller never needs to
    know which side raised.

    ``pending_edges`` defaults to empty: pass the real edges whenever the
    caller has them, and only leave it empty when the review genuinely has no
    edge detail to offer (e.g. a query-shape rejection before discovery ever
    ran) — never as a placeholder for edges that exist but weren't collected.
    """
    err = ReviewRequired(message)
    err.kind = kind
    err.algorithm = algorithm
    err.pending_edge_count = pending_edge_count
    err.pending_edges = tuple(pending_edges)
    err.hint = hint
    err.message = message
    return err


def pending_edges(err: BaseException) -> tuple[PendingEdge, ...]:
    """Structured pending-edge list for a raised review error, when available.

    Normalizes the raised error's ``pending_edges`` attribute (a native
    ``CausalPendingEdge`` sequence or a ``PendingEdge`` tuple built by
    ``build_review_error``) into ``PendingEdge`` instances. Degrades
    gracefully to an empty tuple when the attribute is absent or empty, and
    skips any entry missing one of the four expected attributes rather than
    raising, so a partially-populated error still degrades rather than breaks
    callers.
    """
    raw = getattr(err, "pending_edges", None)
    if not raw:
        return ()
    out: list[PendingEdge] = []
    for entry in raw:
        try:
            out.append(
                PendingEdge(
                    source=str(entry.source),
                    target=str(entry.target),
                    at_source=str(entry.at_source),
                    at_target=str(entry.at_target),
                )
            )
        except AttributeError:
            continue
    return tuple(out)


__all__ = [
    "CausalAttributionError",
    "CausalCancelledError",
    "CausalCompileError",
    "CausalCounterfactualError",
    "CausalDataError",
    "CausalDesignError",
    "CausalDiscoveryError",
    "CausalEstimateError",
    "CausalError",
    "CausalGraphError",
    "CausalIdentifyError",
    "CausalModelError",
    "CausalResourceError",
    "CausalReviewError",
    "CausalSerializationError",
    "CausalStateError",
    "CausalUnsupportedError",
    "CausalValidateError",
    "PendingEdge",
    "ReviewRequired",
    "build_review_error",
    "pending_edges",
]
