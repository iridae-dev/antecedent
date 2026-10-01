"""ADMG conditional transport on the classical complete-source family.

The target population's conditional interventional distribution
``P*(outcomes | do(treatments), conditioned_on)`` over a selection ADMG, from a
source that can run every experiment plus the target's observational law.

Rule 2 of the do-calculus first moves every conditioned variable it can into the
intervention set (the IDC reduction); the classical sID engine then decides the
remaining joint ``P*(outcomes, w'' | do(treatments, w'))``, and the answer is that
joint normalized at the requested ``w''``. The route is sound and incomplete: a
reduced joint that sID cannot certify is ``not_certified``, never a
non-transportability claim (the conditional completeness step is
paper-inherited and unverified). Exact laws only; points only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .._native import (
    consume_admg_conditional_transport_artifact as _consume_admg_conditional_artifact,
)
from .._native import (
    identify_admg_conditional_transport_stage as _identify_admg_conditional_stage,
)
from ..errors import CausalTypeError, CausalValueError
from ..graph import Admg
from ._impl import EvidenceCatalog, SelectionDiagram, _non_negative, _optional_non_negative
from ._multi_source import _names


@dataclass(frozen=True, slots=True)
class ConditionalTransportQuery:
    """Conditional query ``P*(outcomes | do(treatments), conditioned_on)`` in the target.

    ``diagram`` names the source (holding every experiment), the target and the
    variables whose mechanisms differ between them. At least one conditioned
    variable is required; outcomes, treatments and conditioned variables are
    disjoint.
    """

    diagram: SelectionDiagram
    outcomes: Sequence[str]
    conditioned_on: Sequence[str]
    treatments: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.diagram, SelectionDiagram):
            raise CausalTypeError("diagram must be a SelectionDiagram")
        outcomes = _names("outcomes", self.outcomes)
        conditioned = _names("conditioned_on", self.conditioned_on)
        treatments = _names("treatments", self.treatments, allow_empty=True)
        roles = [*outcomes, *conditioned, *treatments]
        if len(set(roles)) != len(roles):
            raise CausalValueError(
                "outcomes, treatments and conditioned variables must be distinct and disjoint"
            )
        object.__setattr__(self, "outcomes", outcomes)
        object.__setattr__(self, "conditioned_on", conditioned)
        object.__setattr__(self, "treatments", treatments)


def identify_admg_conditional_transport(
    *,
    graph: Admg,
    query: ConditionalTransportQuery,
    catalog: EvidenceCatalog,
    max_operations: int = 4096,
    max_depth: int = 24,
    max_support_rows: int = 1_000_000,
    memory_bytes: int | None = None,
    cancel: Any = None,
) -> Any:
    """Decide a bounded ADMG conditional transport query once.

    The returned stage's ``outcome`` is ``identified``, ``missing_evidence`` (the
    certified formula's leaves are not all in the catalog), ``not_certified`` or
    ``exhausted`` (a limit or cancellation stopped the one shared budget; the
    receipt lists explored and unevaluated stages). ``decision()`` explains it:
    the rule-2 moves, the remaining conditioned set, the reduced joint and, for
    ``not_certified``, the reduced joint's verified s-hedge as an inspection-only
    candidate (``"proof": false``). An identified stage offers ``prepare_exact``;
    counted laws (``prepare_empirical``) are refused as ``cell_not_licensed``.
    Limits above 4096 operations / depth 24, more than 6 observed variables, 3
    treatments or 3 conditioned variables refuse as ``route_not_supported``.
    """
    if not isinstance(graph, Admg):
        raise CausalTypeError("identify_admg_conditional_transport requires graph=Admg(...)")
    if not isinstance(query, ConditionalTransportQuery):
        raise CausalTypeError("query must be a ConditionalTransportQuery")
    if not isinstance(catalog, EvidenceCatalog):
        raise CausalTypeError("catalog must be an EvidenceCatalog")
    return _identify_admg_conditional_stage(
        graph,
        query.diagram.source,
        query.diagram.target,
        list(query.diagram.selections),
        list(query.outcomes),
        list(query.treatments),
        list(query.conditioned_on),
        catalog,
        max_operations=_non_negative("max_operations", max_operations),
        max_depth=_non_negative("max_depth", max_depth),
        max_support_rows=_non_negative("max_support_rows", max_support_rows),
        memory_bytes=_optional_non_negative("memory_bytes", memory_bytes),
        cancel=cancel,
    )


def consume_admg_conditional_transport_artifact(
    artifact: bytes,
    *,
    max_search_operations: int = 4096,
    max_search_depth: int = 24,
    max_operations: int = 10_000_000,
    max_depth: int = 256,
    max_support_rows: int | None = None,
    max_laws: int | None = None,
    max_law_cells: int | None = None,
    memory_bytes: int | None = None,
    cancel: Any = None,
) -> str:
    """Verify an exported conditional result and recompute its points.

    Stored limits above the consumer's, or a graph or query above the route's
    size bounds, refuse before any work. The consumer then
    checks the premises and data-identity digests, re-checks every rule-2 move,
    the maximality of the moved set and the reduced joint's proof under the
    producer's stored limits, re-decides the query under them, re-binds the
    leaves and recomputes every point bit for bit. Relabelled names or any other
    edit is refused with a typed ``reason_code``. The replay re-runs the same
    search and evaluator as the producer, so it is independent of the artifact,
    not of the implementation: a bug they share replays identically.
    """
    if not isinstance(artifact, bytes):
        raise CausalTypeError("artifact must be bytes")
    return _consume_admg_conditional_artifact(
        artifact,
        max_search_operations=_non_negative("max_search_operations", max_search_operations),
        max_search_depth=_non_negative("max_search_depth", max_search_depth),
        max_operations=_non_negative("max_operations", max_operations),
        max_depth=_non_negative("max_depth", max_depth),
        max_support_rows=_optional_non_negative("max_support_rows", max_support_rows),
        max_laws=_optional_non_negative("max_laws", max_laws),
        max_law_cells=_optional_non_negative("max_law_cells", max_law_cells),
        memory_bytes=_optional_non_negative("memory_bytes", memory_bytes),
        cancel=cancel,
    )


__all__ = [
    "ConditionalTransportQuery",
    "consume_admg_conditional_transport_artifact",
    "identify_admg_conditional_transport",
]
