"""Input coercion: the only module in the package allowed to accept union input types.

Every other public function takes one concrete type (a ``Dag``, a ``str``, a
``Mapping[str, NDArray]``, …) and relies on the five functions declared here —
``coerce_data``, ``coerce_graph``, ``coerce_query``, ``coerce_refute``,
``coerce_latency`` — to normalize whatever a caller passes (mapping,
DataFrame, edge list, ``Dag``, enum, string, bool, …) before it reaches
concrete-typed internals.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .data import EventFrame


def coerce_data(value: Any) -> tuple[list[str], list[NDArray[np.float64]]]:
    """Normalize tabular input to ``(names, float64 columns)``.

    Accepts, in order:

    - a ``(names, columns)`` pair (the packed form of the old two-positional-
      argument ``discover_*(names, columns)`` call shape);
    - an :class:`antecedent.data.EventFrame` (its ``names``/``columns`` fields
      taken directly — it has no ``to_numpy``, so ``as_columns`` alone would
      otherwise reject it as neither a mapping nor a DataFrame);
    - anything else :func:`antecedent._data.as_columns` already handles: a
      ``Mapping[str, array-like]``, a pandas DataFrame, or an equivalent
      frame-like object exposing ``columns`` + ``to_numpy``.

    ``PanelFrame`` and ``MultiEnvFrame`` are deliberately **not** accepted
    here: both hold more than one table (per-unit / per-environment column
    lists), and there is no existing single-table interpretation of either to
    preserve — coercing one would mean inventing a selection/pooling policy
    that does not exist in the code today. Callers needing those shapes use
    :func:`antecedent._data.as_multi_env_columns` or index the frame's
    per-unit/per-environment columns directly.
    """
    from ._data import as_columns, to_f64

    if isinstance(value, tuple) and len(value) == 2:
        names, columns = value
        return [str(n) for n in names], [to_f64(c) for c in columns]
    if isinstance(value, EventFrame):
        return list(value.names), [to_f64(c) for c in value.columns]
    return as_columns(value)


def coerce_graph(value: Any) -> Any:
    """Normalize a graph input to its canonical native representation.

    Mirrors the discrimination logic currently duplicated across
    ``estimation._static_edges`` / ``estimation._lagged_edges`` and the
    ``Pag``/``Cpdag``/``Admg`` special-casing in ``_analyze.handle_static_ate``:

    - ``Dag`` -> oriented ``(str, str)`` edge list.
    - ``Cpdag`` -> oriented ``(str, str)`` edge list; raises ``ValueError`` if
      undirected/ambiguous marks remain (fully oriented CPDAGs only — same
      rule as ``discovery.cpdag_oriented_edges``).
    - ``TemporalDag`` -> lagged ``(str, int, str, int)`` edge list.
    - ``TemporalCpdag`` -> coerced to a ``TemporalDag`` via
      ``try_into_temporal_dag()`` then lagged edges; raises ``ValueError`` with
      the same message ``_analyze.py`` raises today if that coercion fails.
    - A raw edge list: 2-tuples pass through as static edges, 4-tuples pass
      through as lagged edges.
    - ``Pag``, ``Admg``, ``TemporalPag`` -> returned unchanged. These have no
      single canonical edge-list form; native entry points
      (``analyze_ate_pag`` / ``analyze_ate_admg`` / ``analyze_temporal_pag``)
      accept the object directly, so there is nothing to normalize.
    """
    from .discovery import cpdag_oriented_edges
    from .graph import Admg, Cpdag, Dag, Pag, TemporalCpdag, TemporalDag, TemporalPag

    if isinstance(value, Dag):
        return [(str(a), str(b)) for a, b in value.edges()]
    if isinstance(value, Cpdag):
        return cpdag_oriented_edges(value, require_oriented=True)
    if isinstance(value, TemporalDag):
        return [(str(a), int(la), str(b), int(lb)) for a, la, b, lb in value.edges()]
    if isinstance(value, TemporalCpdag):
        try:
            dag = value.try_into_temporal_dag()
        except Exception as exc:  # noqa: BLE001 — surface orientation failures
            raise ValueError(
                "TemporalCpdag has undirected/conflict marks; orient edges "
                "(try_into_temporal_dag) before analyze, or use discovery review"
            ) from exc
        return [(str(a), int(la), str(b), int(lb)) for a, la, b, lb in dag.edges()]
    if isinstance(value, (Pag, Admg, TemporalPag)):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items = list(value)
        if not items:
            return items
        first = items[0]
        if len(first) == 2:
            return [(str(a), str(b)) for a, b in items]
        if len(first) == 4:
            return [(str(a), int(la), str(b), int(lb)) for a, la, b, lb in items]
    raise TypeError(
        f"unsupported graph type: {type(value)!r}; use a Dag/Cpdag/Pag/Admg/"
        "TemporalDag/TemporalCpdag/TemporalPag, a (str, str) edge list, or a "
        "(str, int, str, int) lagged edge list"
    )


def coerce_query(value: Any) -> Any:
    """Validate a query input and return it unchanged.

    Every query dataclass in :mod:`antecedent.query` carries a ``kind``
    discriminator; anything without one is not a supported query type.
    """
    from .query import (
        AverageEffect,
        ConditionalEffect,
        Counterfactual,
        InterventionalDistribution,
        MediationEffect,
        PathSpecificEffect,
        PulseEffect,
        SustainedEffect,
        TemporalMediationEffect,
    )

    valid = (
        AverageEffect,
        ConditionalEffect,
        Counterfactual,
        InterventionalDistribution,
        MediationEffect,
        PathSpecificEffect,
        PulseEffect,
        SustainedEffect,
        TemporalMediationEffect,
    )
    if isinstance(value, valid):
        return value
    names = ", ".join(c.__name__ for c in valid)
    raise TypeError(f"unsupported query type: {type(value)!r}; use one of {names}")


def coerce_refute(value: Any) -> str | bool:
    """Normalize a refute specification to a native-facing value.

    Accepts a ``Refute`` enum member (-> its wire string), a suite name
    string, or ``False`` (-> ``False``, meaning "no refutation").

    ``refute=True`` raises ``TypeError``. ``True`` carries no information
    about *which* suite to run; the code this consolidates
    (``estimation._resolve_latency_budget``) silently substitutes a
    mode-dependent default suite for it (``out_refute = mapped_refute if
    refute is True else refute``), which is undiscoverable from the call
    site. Pass ``refute="placebo"``, ``"cheap"``, ``"full"``, or a ``Refute``
    enum member instead.
    """
    from .ids import Refute

    if value is True:
        raise TypeError(
            "refute=True is ambiguous: it does not say which refutation suite "
            'to run. Pass refute="placebo", "cheap", "full", or a Refute enum '
            "member instead (or refute=False for no refutation)."
        )
    if value is False:
        return False
    if isinstance(value, Refute):
        return str(value)
    if isinstance(value, str):
        return value
    raise TypeError(
        f"unsupported refute type: {type(value)!r}; use a bool, a Refute enum "
        "member, or a suite name string"
    )


def coerce_latency(value: Any) -> str | None:
    """Normalize a latency specification to a native-facing string.

    Accepts a ``Latency`` enum member or a tier name string
    (``"interactive"`` / ``"standard"`` / ``"report"``) and returns the
    canonical lowercase tier string. ``None`` passes through unchanged — it
    means "no latency override" (``estimation._resolve_latency_budget`` skips
    tier mapping entirely in that case), which is not itself a tier.
    """
    from .ids import Latency

    if value is None:
        return None
    if isinstance(value, Latency):
        return str(value)
    if isinstance(value, str):
        key = value.strip().lower()
        valid = {str(m) for m in Latency}
        if key not in valid:
            raise ValueError(f"unknown latency={value!r}; use interactive|standard|report")
        return key
    raise TypeError(f"unsupported latency type: {type(value)!r}; use a str or Latency enum member")


__all__ = [
    "coerce_data",
    "coerce_graph",
    "coerce_latency",
    "coerce_query",
    "coerce_refute",
]
