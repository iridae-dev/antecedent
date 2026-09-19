"""The omitted-default table, read from the Rust study builder.

``analyze`` / ``prepare`` / ``PreparedAnalysis.prepare`` never fill an omitted
``refute`` / ``bootstrap`` / ``latency`` themselves: an omitted value reaches
the study builder as omitted, and the builder applies this table, its
latency-tier mapping, and its refute downgrade. This module exposes what the
builder would use, for display and for tests.
"""

from __future__ import annotations

from typing import Any

from ._native import default_user_threads, omitted_defaults

#: ``bootstrap`` (on routes that resample), ``refute`` (before any cell
#: downgrade), ``latency`` (never injected), and the Bayesian draw budgets.
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


def resolve_threads(threads: int | None) -> int:
    """Omitted ``threads`` uses the machine; ``threads=1`` remains an explicit pin."""
    if threads is None:
        return int(default_user_threads())
    if threads < 1:
        raise ValueError("threads must be >= 1")
    return int(threads)


def is_temporal_query(query: Any = None, *, kind: str = "") -> bool:
    """True when the query carries temporal structure or a temporal kind tag."""
    resolved = kind or getattr(query, "kind", "") or ""
    return bool(getattr(query, "is_temporal", False)) or resolved in TEMPORAL_QUERY_KINDS
