"""Single omitted-default table for analyze / prepare / PreparedAnalysis.prepare."""

from __future__ import annotations

from typing import Any

from ._native import omitted_defaults

OMITTED: dict[str, Any] = omitted_defaults()

TEMPORAL_QUERY_KINDS = frozenset({"pulse", "sustained", "temporal_mediation"})
RESPONSE_QUERY_KINDS = frozenset(
    {
        "response_curve",
        "intervention_response",
        "counterfactual",
        "average_derivative",
        "point_derivative",
        "elasticity",
        "semi_elasticity",
        "directional_derivative",
        "response_jacobian",
    }
)


def is_temporal_query(query: Any = None, *, kind: str = "") -> bool:
    """True when the query carries temporal structure or a temporal kind tag."""
    resolved = kind or getattr(query, "kind", "") or ""
    return bool(getattr(query, "is_temporal", False)) or resolved in TEMPORAL_QUERY_KINDS


def resolve_omitted(
    *,
    kind: str,
    inference: Any,
    is_temporal: bool | None = None,
    query: Any = None,
    refute: Any,
    bootstrap: int | None,
    latency: Any,
) -> tuple[Any, int | None, Any]:
    """Apply the one omitted-default table. Latency is never injected.

    Static and Bayesian response-family queries keep bootstrap at 0 (analytic,
    influence-function, or posterior uncertainty). Only Frequentist temporal
    surfaces inherit the omitted replicate count.
    """
    if is_temporal is None:
        is_temporal = is_temporal_query(query, kind=kind)
    if refute is None:
        refute = "none" if kind in RESPONSE_QUERY_KINDS else OMITTED["refute"]
    if bootstrap is None:
        bayesian = type(inference).__name__ == "Bayesian"
        if kind in RESPONSE_QUERY_KINDS and (kind == "counterfactual" or bayesian or not is_temporal):
            bootstrap = 0
        else:
            bootstrap = OMITTED["bootstrap"]
    return refute, bootstrap, latency
