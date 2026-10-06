"""Study planning over the X1 (mz) and X9 (mixed-source) catalogs.

Given a query that the multi-source (``route="mz"``) or mixed-source
(``route="mixed"``) route cannot identify from its catalog, and a declared
universe of at most sixteen candidate studies with integer costs, the planner
evaluates every subset of at most three studies in cost order and re-runs the
same route on each subset's hypothetical catalog, all under one shared search
budget. A subset is sufficient only when the re-run decision identifies and its
derivation re-checks against the hypothetical catalog.

The plan makes no numeric claim and no probability-of-success claim: a
sufficient proposal says that if the studies deliver exactly the declared
regimes (with mass at every level the formula reads), the same route would
identify the query. ``none_certified`` means nothing in this universe was
certified, never that the query is impossible. Minimality is claimed only when
every strictly cheaper subset was decided without a stop or an over-cap
decision, and means cost-minimal among the verified-derivable subsets of at most
three candidates of this declared universe, under this search and rule set: a
cheaper subset of four or more candidates is never examined, and a cheaper
subset the route refuses keeps the claim (it is only not sufficient under this
search).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .._native import plan_studies_stage as _plan_studies_stage
from .._native import replay_study_plan as _replay_study_plan
from ..errors import CausalTypeError, CausalValueError
from ..graph import Admg
from ._impl import EvidenceCatalog, _non_negative, _optional_non_negative
from ._mixed_source import MixedSourceQuery
from ._multi_source import MultiSourceZTransportQuery, _names

_MAX_OPERATIONS = 200_000
_MAX_DEPTH = {"mz": 24, "mixed": 16}


@dataclass(frozen=True, slots=True)
class StudyCandidate:
    """One study a planner may propose.

    ``population`` is the target or a declared source. ``interventions`` is
    empty for an observational study; ``levels`` lists the feasible level
    combinations (one regime each) or is ``None`` for every level (one
    unrestricted regime; the only form the mixed route accepts). ``measured`` is
    the jointly measured margin. ``cost_units`` is a positive integer;
    ``sample_budget`` only breaks ties. ``recruitment`` is recorded, never a
    sufficiency input.
    """

    id: str
    population: str
    measured: Sequence[str]
    cost_units: int
    recruitment: str
    interventions: Sequence[str] = ()
    levels: Sequence[Mapping[str, float]] | None = None
    sample_budget: int = 0
    requires: Sequence[str] = ()
    conflicts: Sequence[str] = ()
    feasibility: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise CausalValueError("candidate id must be a non-empty string")
        if not isinstance(self.population, str) or not self.population.strip():
            raise CausalValueError("candidate population must be a non-empty name")
        object.__setattr__(self, "measured", _names("measured", self.measured))
        object.__setattr__(
            self, "interventions", _names("interventions", self.interventions, allow_empty=True)
        )
        if isinstance(self.cost_units, bool) or not isinstance(self.cost_units, int):
            raise CausalTypeError("cost_units must be an integer")
        _non_negative("cost_units", self.cost_units)
        _non_negative("sample_budget", self.sample_budget)
        if self.levels is not None:
            levels = tuple(dict(level) for level in self.levels)
            for level in levels:
                if any(not math.isfinite(float(value)) for value in level.values()):
                    raise CausalValueError("candidate levels must be finite")
            object.__setattr__(self, "levels", levels)
        for name in ("requires", "conflicts", "feasibility"):
            values = tuple(getattr(self, name))
            if any(not isinstance(value, str) for value in values):
                raise CausalTypeError(f"{name} must be strings")
            object.__setattr__(self, name, values)


def plan_studies(
    *,
    graph: Admg,
    query: MultiSourceZTransportQuery | MixedSourceQuery,
    catalog: EvidenceCatalog,
    candidates: Sequence[StudyCandidate],
    max_operations: int = _MAX_OPERATIONS,
    max_depth: int | None = None,
    memory_bytes: int | None = None,
    cancel: Any = None,
) -> Any:
    """Plan the cheapest subset of at most three candidate studies that repairs the query.

    ``query`` selects the route: a :class:`MultiSourceZTransportQuery` plans over
    the mz route, a :class:`MixedSourceQuery` over the mixed-source route. The
    returned stage's ``outcome`` is ``sufficient``, ``none_certified`` or
    ``exhausted``; ``plan()`` lists the frozen failure, every subset's outcome,
    the ranked proposals with the regimes to ``deliver``, the factor or proof
    step each repairs and the margin the proof reads, the stop receipt and the
    minimality flag. ``receive(rank, catalog, provider_snapshot)`` accepts the
    arriving evidence and re-identifies it through the public route;
    ``export()`` yields an artifact :func:`replay_study_plan` replays.

    Refusals carry their reason code: ``invalid_argument`` for an invalid
    candidate (including a level label ``id#k`` a base regime already uses), a
    base regime without provider lineage (no binding outside the
    ``hypothetical:`` namespace) or a base that already identifies,
    ``route_not_supported`` for a bound
    (checked before any candidate compiles),
    ``transport_proven_non_transportable`` when the base is a checked
    obstruction over the declared controllable sets (a new population or a
    wider controllable set is outside this universe), and
    ``transport_budget_cancel`` when the budget stops the base decision.
    ``receive`` refuses a ``hypothetical:`` provider snapshot (a planning
    preview) and reports a stop of the public route as ``study_plan.budget``.
    """
    if not isinstance(graph, Admg):
        raise CausalTypeError("plan_studies requires graph=Admg(...)")
    if isinstance(query, MultiSourceZTransportQuery):
        route = "mz"
    elif isinstance(query, MixedSourceQuery):
        route = "mixed"
    else:
        raise CausalTypeError("query must be a MultiSourceZTransportQuery or MixedSourceQuery")
    if not isinstance(catalog, EvidenceCatalog):
        raise CausalTypeError("catalog must be an EvidenceCatalog")
    candidates = tuple(candidates)
    if any(not isinstance(candidate, StudyCandidate) for candidate in candidates):
        raise CausalTypeError("candidates must be StudyCandidate values")
    depth = _MAX_DEPTH[route] if max_depth is None else max_depth
    return _plan_studies_stage(
        graph,
        route,
        query.target,
        list(query.outcomes),
        list(query.treatments),
        [
            (
                source.population,
                list(source.controllable),
                dict(source.experiment_assignment),
                list(source.selections),
            )
            for source in query.sources
        ],
        catalog,
        [
            (
                c.id,
                c.population,
                list(c.interventions),
                None if c.levels is None else [dict(level) for level in c.levels],
                list(c.measured),
                c.recruitment,
                c.cost_units,
                c.sample_budget,
                list(c.requires),
                list(c.conflicts),
                list(c.feasibility),
            )
            for c in candidates
        ],
        max_operations=_non_negative("max_operations", max_operations),
        max_depth=_non_negative("max_depth", depth),
        memory_bytes=_optional_non_negative("memory_bytes", memory_bytes),
        cancel=cancel,
    )


def replay_study_plan(
    artifact: bytes,
    *,
    max_operations: int = _MAX_OPERATIONS,
    max_depth: int = 24,
    memory_bytes: int | None = None,
    cancel: Any = None,
) -> str:
    """Independently replay an exported study plan and return it as JSON.

    The consumer refuses stored limits above its own before any work, checks
    every digest, then re-plans under the stored limits and accepts only an
    identical plan (``study_plan.invalid_artifact`` otherwise); a cancelled
    replay is ``study_plan.budget``, never a verdict. Replay does not protect
    against a producer that states another base catalog or candidate universe
    and re-seals honestly: the artifact is then a correct plan of those stated
    inputs. Lineage (the base catalog's bindings and snapshots) is stored and
    digested; a plan refuses a base regime without a provider binding outside
    the ``hypothetical:`` namespace, so an artifact re-sealed with its base
    bindings cleared no longer replays. A binding renamed to another real
    snapshot is not detectable by replay; ``receive`` requires the arriving
    catalog to keep every stored base binding exactly.
    """
    if not isinstance(artifact, bytes):
        raise CausalTypeError("artifact must be bytes")
    return _replay_study_plan(
        artifact,
        max_operations=_non_negative("max_operations", max_operations),
        max_depth=_non_negative("max_depth", max_depth),
        memory_bytes=_optional_non_negative("memory_bytes", memory_bytes),
        cancel=cancel,
    )


__all__ = ["StudyCandidate", "plan_studies", "replay_study_plan"]
