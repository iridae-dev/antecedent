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

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from importlib import import_module
from typing import TYPE_CHECKING, Any

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
    CausalValidateError,
)
from ._native import (
    CausalCancelledError as CausalCancelled,
)
from ._native import (
    CausalUnsupportedError as _NativeUnsupported,
)
from ._native import runtime_refusal_codes as _runtime_refusal_codes
from ._native import set_not_identified_error_class as _set_not_identified_error_class
from ._native import set_review_error_class as _set_review_error_class
from ._native import set_unsupported_error_class as _set_unsupported_error_class
from ._native import set_value_error_class as _set_value_error_class

_RUNTIME_REFUSAL_CODES = frozenset(_runtime_refusal_codes())


def _registered_code(reason_code: str | None) -> str | None:
    """``reason_code`` if it is a registered runtime-refusal code; refuse any other."""
    if reason_code is not None and reason_code not in _RUNTIME_REFUSAL_CODES:
        raise ValueError(
            f"unregistered runtime reason code {reason_code!r}; "
            "add it to parity/reason_codes.toml with applies_to runtime_refusal"
        )
    return reason_code


class CausalTypeError(CausalValidateError, TypeError):
    """Input-validation failure: wrong argument type at a public entry point.

    Subclasses both :class:`CausalValidateError` (so ``except
    antecedent.CausalError`` and ``except CausalValidateError`` catch it) and
    the builtin ``TypeError`` (so an existing ``except TypeError`` /
    ``pytest.raises(TypeError)`` call site keeps working unchanged — this is
    additive, not a replacement for the builtin type). Raised by the
    coercion helpers in ``_coerce.py`` and the equivalent checks in
    ``estimation.py`` / ``discovery.py`` / ``accepted_graph.py`` where a
    caller-supplied argument (``graph=``, ``query=``, ``refute=``,
    ``latency=``, a discovery config, ...) is the wrong Python type.

    ``reason_code`` is an optional registered runtime-refusal code, checked
    like :class:`CausalUnsupportedError`'s.
    """

    def __init__(self, message: str = "", *, reason_code: str | None = None) -> None:
        super().__init__(message)
        self.reason_code = _registered_code(reason_code)


class CausalUnsupportedError(_NativeUnsupported):
    """A refusal, with an optional closed reason code.

    The native layer instantiates this same class for refusals raised in Rust
    (see ``set_unsupported_error_class``), reading the ``reason=<code>:``
    prefix into :attr:`reason_code`, so a caller catches one class either way.

    ``reason_code`` must be a registered runtime-refusal code
    (``parity/reason_codes.toml``, exposed as ``_native.runtime_refusal_codes()``);
    constructing the error with any other code raises ``ValueError`` so an
    unregistered code cannot be emitted.

    ``remedy`` is optional text naming what the caller can change to proceed.
    It is never part of the message; ``None`` (the default every
    :class:`CausalError` reads) when the refusal names none.
    """

    def __init__(
        self,
        message: str = "",
        *,
        reason_code: str | None = None,
        remedy: str | None = None,
    ) -> None:
        _registered_code(reason_code)
        text = f"reason={reason_code}: {message}" if reason_code else message
        super().__init__(text)
        self.reason_code = reason_code
        if remedy is not None:
            self.remedy = remedy


class CausalValueError(CausalValidateError, ValueError):
    """Input-validation failure: right type, invalid value at a public entry point.

    See :class:`CausalTypeError` — same rationale, for value checks (a
    correctly-typed argument whose value is out of range, missing a required
    companion, or otherwise not acceptable) rather than type checks.
    """

    def __init__(self, message: str = "", *, reason_code: str | None = None) -> None:
        super().__init__(message)
        self.reason_code = _registered_code(reason_code)


#: Reason code of a refusal that rejects a malformed argument rather than a scientific route.
INVALID_ARGUMENT = "invalid_argument"

_ARGUMENT_VARIANTS: dict[type, type] = {}


def _argument_variant(cls: type) -> type:
    """``cls`` extended with :class:`CausalValueError`, for ``invalid_argument`` refusals."""
    variant = _ARGUMENT_VARIANTS.get(cls)
    if variant is None:
        variant = type(
            cls.__name__,
            (cls, CausalValueError),
            {
                "__module__": cls.__module__,
                "__qualname__": cls.__qualname__,
                "__doc__": cls.__doc__,
            },
        )
        _ARGUMENT_VARIANTS[cls] = variant
    return variant


class StructuredRefusal(CausalUnsupportedError):
    """The shared base of every refusal that carries machine-readable fields.

    Built from a refusal mapping (the shape the native layer reports), it exposes the same
    fields on every refusal class so a caller can branch without parsing the message:

    - :attr:`code` (alias :attr:`reason_code`) is the registered refusal code;
    - :attr:`detail` is the namespaced ``family.slot`` (for example
      ``msm_sensitivity.lambda_below_one``);
    - :attr:`offending` is the field, quantity, node or identity the refusal is about, when there
      is one;
    - :attr:`remedy` is what the caller can change to proceed, when the refusal names it;
    - :attr:`stage` is the refusing stage and :attr:`message` the human-readable context.

    A refusal whose code is ``invalid_argument`` rejects a malformed argument, not a scientific
    route: it is also a :class:`CausalValueError`, so ``except CausalValueError`` catches it.
    """

    def __new__(cls, refusal: Any = None, *args: Any, **kwargs: Any) -> StructuredRefusal:
        code = refusal.get("code") if isinstance(refusal, Mapping) else kwargs.get("reason_code")
        target = cls
        if code == INVALID_ARGUMENT and not issubclass(cls, ValueError):
            target = _argument_variant(cls)
        return super().__new__(target)

    def __init__(self, refusal: Mapping[str, Any], *, text: str | None = None) -> None:
        detail = str(refusal["detail"])
        message = refusal.get("message")
        offending = refusal.get("offending")
        if text is None:
            text = f"{detail}: {message}" if message else detail
            if offending:
                text = f"{text} at {offending}"
        super().__init__(text, reason_code=refusal["code"], remedy=refusal.get("remedy"))
        #: Refusing stage.
        self.stage: str = refusal.get("stage", "")
        #: Namespaced ``family.slot`` detail.
        self.detail: str | None = detail
        #: Human-readable context, when the refusal carries one.
        self.message: str = str(message or "")
        #: The field, quantity, node or identity the refusal is about, when there is one.
        self.offending: str | None = offending

    @property
    def code(self) -> str | None:
        """The registered refusal code (the same value as :attr:`reason_code`)."""
        code: str | None = self.reason_code
        return code


class EffectNotIdentified(CausalUnsupportedError, CausalCompileError):
    """The question has no identified estimand under the declared structure.

    A refusal, never a result. ``reason_code`` is ``effect_not_identified``;
    ``identification_status`` is the status the search ended with, and
    ``search_complete`` / ``search_capped`` say whether it finished or stopped
    at a budget (a capped search is not a proof of non-identification). It
    subclasses :class:`CausalCompileError`, which this refusal was raised as
    before it carried its outcome.
    """

    identification_status: str = "not_identified"
    search_capped: bool = False
    search_complete: bool = True


_DISPLAY_NODE = re.compile(r"^V(\d+)(?:@(.+))?$")


def resolve_display_name(token: str, names: Sequence[str]) -> str:
    """Map a dense display id (``V3``, ``V3@-1``) onto a schema name."""
    match = _DISPLAY_NODE.match(token)
    if match is None:
        return token
    index = int(match.group(1))
    if index < 0 or index >= len(names):
        return token
    base = names[index]
    lag = match.group(2)
    return f"{base}@{lag}" if lag is not None else base


@dataclass(frozen=True, slots=True)
class PendingEdge:
    """One unreviewed edge from a `ReviewRequired`, with its endpoint marks.

    ``at_source`` / ``at_target`` are one of ``"tail"``, ``"arrow"``,
    ``"circle"``, or ``"conflict"`` — the same vocabulary ``GraphEdge`` uses
    elsewhere in this package. ``source`` / ``target`` are display identifiers
    (``V3``, ``V3@-1``) until :meth:`with_names` maps them onto schema names.
    """

    source: str
    target: str
    at_source: str
    at_target: str

    def with_names(self, names: Sequence[str]) -> PendingEdge:
        """Copy with ``V{i}`` display ids replaced by ``names[i]``."""
        return replace(
            self,
            source=resolve_display_name(self.source, names),
            target=resolve_display_name(self.target, names),
        )


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
# The same registration for refusals: a Rust refusal is this class, with its
# reason code already attached.
_set_unsupported_error_class(CausalUnsupportedError)
_set_not_identified_error_class(EffectNotIdentified)
_set_value_error_class(CausalValueError)


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


def named_pending_edges(
    edges: Sequence[PendingEdge], names: Sequence[str]
) -> tuple[PendingEdge, ...]:
    """Resolve a pending-edge list against column / graph node names."""
    return tuple(edge.with_names(names) for edge in edges)


def next_action(err: BaseException, edges: Sequence[PendingEdge] = ()) -> str:
    """Copy-paste next legal call, or why the refusal is terminal."""
    if isinstance(err, ReviewRequired) or type(err).__name__ == "ReviewRequired":
        kind = getattr(err, "kind", None) or "review"
        algorithm = getattr(err, "algorithm", None)
        via = f"{kind} via {algorithm}" if algorithm else str(kind)
        pending = edges or pending_edges(err)
        if pending:
            mapping = ", ".join(
                f"({edge.source!r}, {edge.target!r}): ({edge.at_source!r}, {edge.at_target!r})"
                for edge in pending
            )
            return (
                f"Review required ({via}, {len(pending)} pending). "
                f"Orient with accepted.review({{{mapping}}}) and re-run analyze."
            )
        hint = getattr(err, "hint", None)
        return f"Review required ({via}). {hint}" if hint else f"Review required ({via})."
    if isinstance(err, EffectNotIdentified):
        status = getattr(err, "identification_status", "not_identified")
        capped = getattr(err, "search_capped", False)
        search = "capped; not a proof of non-identification" if capped else "complete"
        return (
            f"Not identified ({status}; search {search}). "
            "Change the query or supply a more informative graph."
        )
    code = getattr(err, "reason_code", None) or type(err).__name__
    hint = getattr(err, "hint", None)
    if hint:
        return f"{code}: {hint}"
    return f"{code}: {err}"


#: Refusal classes defined beside the feature they refuse, re-exported here lazily (importing
#: their modules at the top of this one would be circular). ``name -> defining module``.
_LAZY_REFUSALS: dict[str, str] = {
    "CallbackUnavailableRefusal": "composition_bundle",
    "CompositionBundleRefusal": "composition_bundle",
    "DecisionRefusal": "decision",
    "EdgeDigestMismatchRefusal": "composition_bundle",
    "ExpectedIdentityMismatchRefusal": "composition_bundle",
    "GraphOrSnapshotMismatchRefusal": "composition_bundle",
    "IncompatibleVersionRefusal": "composition_bundle",
    "MechanismDiscrepancyRefusal": "mechanism_discrepancy",
    "MsmSensitivityRefusal": "msm_sensitivity",
    "NodeNotFoundRefusal": "composition_bundle",
    "OversizedRefusal": "composition_bundle",
    "ProviderRequestChangedRefusal": "composition_bundle",
    "RecalcNoLiveState": "recalc",
    "RecalcReceiptRefusal": "recalc",
    "RecalcRefusal": "recalc",
    "RecalcUnavailable": "recalc",
    "RepairArtifactRefusal": "repair",
    "RepairBudgetRefusal": "repair",
    "RepairRefusal": "repair",
    "ScenarioInvarianceRefusal": "scenario_invariance",
    "ScoreResumeRefusal": "recalc_cell",
    "ScoreResumeUnavailable": "recalc_cell",
    "SensitivityRefusal": "sensitivity_decision",
    "StudyCandidateRefusal": "repair",
    "SwappedEvidenceRefusal": "composition_bundle",
    "TamperedQuantityRefusal": "composition_bundle",
    "TransportedCounterfactualRefusal": "transported_counterfactual",
    "UnknownNodeKindRefusal": "composition_bundle",
    "UnsupportedLawRefusal": "composition_bundle",
}

if TYPE_CHECKING:
    from .composition_bundle import (
        CallbackUnavailableRefusal,
        CompositionBundleRefusal,
        EdgeDigestMismatchRefusal,
        ExpectedIdentityMismatchRefusal,
        GraphOrSnapshotMismatchRefusal,
        IncompatibleVersionRefusal,
        NodeNotFoundRefusal,
        OversizedRefusal,
        ProviderRequestChangedRefusal,
        SwappedEvidenceRefusal,
        TamperedQuantityRefusal,
        UnknownNodeKindRefusal,
        UnsupportedLawRefusal,
    )
    from .decision import DecisionRefusal
    from .mechanism_discrepancy import MechanismDiscrepancyRefusal
    from .msm_sensitivity import MsmSensitivityRefusal
    from .recalc import (
        RecalcNoLiveState,
        RecalcReceiptRefusal,
        RecalcRefusal,
        RecalcUnavailable,
    )
    from .recalc_cell import ScoreResumeRefusal, ScoreResumeUnavailable
    from .repair import (
        RepairArtifactRefusal,
        RepairBudgetRefusal,
        RepairRefusal,
        StudyCandidateRefusal,
    )
    from .scenario_invariance import ScenarioInvarianceRefusal
    from .sensitivity_decision import SensitivityRefusal
    from .transported_counterfactual import TransportedCounterfactualRefusal


def __getattr__(name: str) -> Any:
    module = _LAZY_REFUSALS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module}", __package__), name)
    globals()[name] = value
    return value


__all__ = [
    "CallbackUnavailableRefusal",
    "CompositionBundleRefusal",
    "DecisionRefusal",
    "EdgeDigestMismatchRefusal",
    "ExpectedIdentityMismatchRefusal",
    "GraphOrSnapshotMismatchRefusal",
    "IncompatibleVersionRefusal",
    "MechanismDiscrepancyRefusal",
    "MsmSensitivityRefusal",
    "NodeNotFoundRefusal",
    "OversizedRefusal",
    "ProviderRequestChangedRefusal",
    "RecalcNoLiveState",
    "RecalcReceiptRefusal",
    "RecalcRefusal",
    "RecalcUnavailable",
    "RepairArtifactRefusal",
    "RepairBudgetRefusal",
    "RepairRefusal",
    "ScenarioInvarianceRefusal",
    "ScoreResumeRefusal",
    "ScoreResumeUnavailable",
    "SensitivityRefusal",
    "StudyCandidateRefusal",
    "SwappedEvidenceRefusal",
    "TamperedQuantityRefusal",
    "TransportedCounterfactualRefusal",
    "UnknownNodeKindRefusal",
    "UnsupportedLawRefusal",
    "CausalAttributionError",
    "CausalCancelled",
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
    "CausalTypeError",
    "CausalUnsupportedError",
    "CausalValidateError",
    "CausalValueError",
    "EffectNotIdentified",
    "PendingEdge",
    "ReviewRequired",
    "StructuredRefusal",
    "build_review_error",
    "named_pending_edges",
    "next_action",
    "pending_edges",
    "resolve_display_name",
]
