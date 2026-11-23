"""Project declarations before owning a table, without interpreting its values."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ._columns import _declared_strings, analysis_columns, collect_column_names
from ._data import project_columns


def project_prepared_input(
    data: Any,
    *,
    query: Any,
    graph: Any,
    estimator: Any = None,
    estimator_config: Mapping[str, Any] | None = None,
    participant: str | None = None,
    missing: Any = None,
) -> Any:
    """Keep analysis, declared design and identifier columns before ownership.

    The existing analysis projection determines which graph families read a
    subset. Design columns remain available to resolve and check declarations;
    they are removed from numeric ingest by their existing consumers. Projection
    never casts, encodes or borrows values. Missing columns retain the existing
    consumer's validation, while selected duplicate names are refused explicitly.
    """
    from ._estimate_helpers import _unwrap_estimator
    from .accepted_graph import AcceptedGraph

    supplied = graph.graph if isinstance(graph, AcceptedGraph) else graph
    _estimator, config = _unwrap_estimator(estimator, estimator_config)
    projection = analysis_columns(query, supplied, config, missing=missing)
    if projection is None:
        return data
    required = set(projection.names)
    required.update(collect_column_names(query))
    _declared_strings(query, required)
    _declared_strings(config, required)
    if participant is not None:
        required.add(participant)
    return project_columns(data, sorted(required), subset=True)
