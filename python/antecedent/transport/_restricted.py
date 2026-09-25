"""Route a declared controllable set through licensed single-source z-transport.

Classical and meta transport assume each source can experiment on every
variable. When every experimental source's declared interventions omit the
queried treatment, identification and execution stay on the z-transport stage.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..errors import CausalUnsupportedError, CausalValueError
from ..graph import Admg
from ._impl import (
    EmpiricalTable,
    Environment,
    EvidenceCatalog,
    EvidenceRegime,
    ExactDiscreteLaw,
    ExactTransportData,
    LearnedCategorical,
    RegimeBinding,
    RegimeSample,
    SelectionDiagram,
    StatisticalTransportData,
    TransportControls,
    TrialAipw,
    TrialAipwData,
    VariableCoordinate,
    ZTransportQuery,
    identify_z_transport,
)
from .._transport_results import TransportGridPoint, TransportUncertainty

SCOPE = "single_source_z_transport_cited_joints_sound_incomplete"
_COMBINATION = "z_transport.multi_source_combination_not_searched"


@dataclass(frozen=True, slots=True)
class RestrictedTransportIdentification:
    """Catalog-aware z-transport decision for one ordinary Transport question."""

    outcome: str
    reason: str | None
    rules: tuple[str, ...]
    formula: str
    source: str | None
    scope: str = SCOPE
    stage: Any = None
    laws: tuple[ExactDiscreteLaw, ...] = ()
    empirical: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "rules", tuple(self.rules))
        object.__setattr__(self, "laws", tuple(self.laws))


class RestrictedTransportExecution:
    """Point results from the z-transport stage, one target assignment at a time."""

    def __init__(
        self,
        *,
        points: tuple[TransportGridPoint, ...],
        rules: tuple[str, ...],
        prepared: Any,
    ) -> None:
        self.scope = SCOPE
        self.formula = SCOPE
        self.rules = rules
        self.points = points
        self._prepared = prepared
        first = points[0] if points else None
        self.uncertainty = None if first is None else first.uncertainty
        self.outcomes = () if first is None else tuple(first.means)

    def mean(self, outcome: str) -> float:
        if not self.points:
            raise CausalValueError("z-transport execution has no target assignment")
        return float(self.points[0].means[outcome])

    def export(self) -> bytes:
        if self._prepared is None:
            raise CausalValueError("estimate before exporting a z-transport artifact")
        return bytes(self._prepared.export())

    def inspect(self) -> Any:
        from ..results._report import InspectionReport, SlotModel

        identified = bool(self.points)
        interval = self.uncertainty
        available = interval is not None and interval.available
        return InspectionReport(
            identification=SlotModel(
                available=identified,
                summary=SCOPE if identified else "unavailable",
                payload={"scope": SCOPE, "rules": list(self.rules)},
            ),
            support=SlotModel(
                available=identified,
                summary="checked" if identified else "unavailable",
                payload={"scope": SCOPE},
            ),
            uncertainty=SlotModel(
                available=available,
                reason=None if interval is None else interval.reason,
                summary="pointwise_percentile" if available else "unavailable",
                payload={"scope": SCOPE},
            ),
            assumptions=SlotModel(
                available=True,
                summary="declared",
                payload={"scope": SCOPE, "rules": list(self.rules)},
            ),
        )


class _RestrictedNative:
    """PreparedAnalysis native adapter over ``ZTransportStage`` prepare/estimate."""

    def __init__(
        self,
        *,
        stage: Any,
        catalog: EvidenceCatalog,
        laws: tuple[ExactDiscreteLaw, ...],
        worlds: list[dict[str, float]],
        empirical: bool,
        limits: TransportControls,
        rules: tuple[str, ...],
        rebuild: Any,
    ) -> None:
        self.stage = stage
        self.catalog = catalog
        self.laws = laws
        self.worlds = worlds
        self.empirical = empirical
        self.limits = limits
        self.rules = rules
        self.rebuild = rebuild
        self._prepared: Any = None

    def replace_snapshot(self, data: Any, cancel: Any = None) -> None:
        del cancel
        self.laws = self.rebuild(data)
        self._prepared = None

    def estimate(self, cancel: Any = None) -> RestrictedTransportExecution:
        del cancel
        return self._execute()

    def refresh(self, data: Any, cancel: Any = None) -> RestrictedTransportExecution:
        self.replace_snapshot(data, cancel=cancel)
        return self._execute()

    def _execute(self) -> RestrictedTransportExecution:
        prepared = None
        points: list[TransportGridPoint] = []
        for world in self.worlds:
            kwargs = {
                "max_operations": self.limits.max_operations,
                "max_depth": self.limits.max_depth,
                "max_support_rows": self.limits.max_support_rows,
                "memory_bytes": self.limits.memory_bytes,
            }
            if self.empirical:
                prepared = self.stage.prepare_empirical(self.catalog, self.laws, world, **kwargs)
            else:
                prepared = self.stage.prepare_exact(self.catalog, self.laws, world, **kwargs)
            payload = json.loads(prepared.estimate())
            points.append(_point(world, payload))
        self._prepared = prepared
        return RestrictedTransportExecution(
            points=tuple(points), rules=self.rules, prepared=prepared
        )


def restricted_experiment(query: Any) -> bool:
    """True when no experimental source declares an experiment on the treatment."""

    from ._day1 import _question_parts

    treatments, _outcomes = _question_parts(query.question)
    experimental = [source for source in query.evidence.sources if source.kind == "experimental"]
    if not experimental:
        return False
    return all(not set(treatments).issubset(source.interventions) for source in experimental)


def identify_restricted(
    graph: Admg, query: Any, data: Any | None = None
) -> tuple[RestrictedTransportIdentification, EvidenceCatalog]:
    """Identify on the z-transport stage. Do not call classical or meta transport."""

    from ._day1 import _question_parts

    treatments, outcomes = _question_parts(query.question)
    sources = [
        source
        for source in query.evidence.sources
        if source.kind == "experimental" and not set(treatments).issubset(source.interventions)
    ]
    if len(sources) > 2:
        raise CausalUnsupportedError(
            "restricted-experiment transport searches at most two sources separately",
            reason_code=_COMBINATION,
        )
    if not sources:
        raise CausalValueError("restricted-experiment transport requires a controllable set")
    bundles = [_source_bundle(graph, query, data, source, outcomes, treatments) for source in sources]
    decisions = [bundle["stage"].decide(bundle["catalog"]) for bundle in bundles]
    identified = [
        (bundle, decision)
        for bundle, decision in zip(bundles, decisions, strict=True)
        if decision["outcome"] == "identified"
    ]
    if identified:
        bundle, decision = identified[0]
        rules = tuple(decision.get("proof", {}).get("rules") or ())
        return (
            RestrictedTransportIdentification(
                outcome="identified",
                reason=None,
                rules=rules,
                formula=SCOPE,
                source=bundle["source"].identity,
                stage=bundle["stage"],
                laws=bundle["laws"],
                empirical=bundle["empirical"],
            ),
            bundle["catalog"],
        )
    if len(decisions) == 2 and all(
        decision["outcome"] == "proven_non_transportable" for decision in decisions
    ):
        return (
            RestrictedTransportIdentification(
                outcome="proven_non_transportable",
                reason=None,
                rules=(),
                formula=SCOPE,
                source=None,
            ),
            bundles[0]["catalog"],
        )
    if len(sources) == 2:
        return (
            RestrictedTransportIdentification(
                outcome="not_certified",
                reason=_COMBINATION,
                rules=(),
                formula=SCOPE,
                source=None,
            ),
            bundles[0]["catalog"],
        )
    decision = decisions[0]
    reason = decision.get("reason")
    if decision["outcome"] == "missing_evidence" and not reason:
        reason = "transport.missing_evidence"
    return (
        RestrictedTransportIdentification(
            outcome=decision["outcome"],
            reason=reason,
            rules=tuple(decision.get("proof", {}).get("rules") or ()),
            formula=SCOPE,
            source=sources[0].identity,
            stage=bundles[0]["stage"],
            laws=bundles[0]["laws"],
            empirical=bundles[0]["empirical"],
        ),
        bundles[0]["catalog"],
    )


def prepare_restricted(
    data: Any,
    *,
    query: Any,
    graph: Admg,
    provider: Any = None,
    controls: TransportControls | None = None,
) -> Any:
    """Prepare the licensed z-transport stage for a restricted experiment."""

    from ..estimation import PreparedAnalysis

    if isinstance(provider, (LearnedCategorical, TrialAipw)) or isinstance(data, TrialAipwData):
        raise CausalUnsupportedError(
            "Restricted-experiment transport executes the exact or empirical plugin. "
            "LearnedCategorical and TrialAipw stay on the classical route.",
            reason_code="option_not_applicable",
        )
    identified, catalog = identify_restricted(graph, query, data)
    from ._day1 import lower_question

    shape, worlds = lower_question(query.question)
    limits = controls or TransportControls()
    study: PreparedAnalysis[Any] = PreparedAnalysis(None, kind="z_transport", query=query)
    study._transport_stage = {
        "identified": identified,
        "catalog": catalog,
        "bound": data,
        "shape": shape,
        "worlds": worlds,
        "provider": EmpiricalTable() if identified.empirical else "exact_law",
        "graph": graph,
    }
    if identified.outcome != "identified" or identified.stage is None:
        return study
    native = _RestrictedNative(
        stage=identified.stage,
        catalog=catalog,
        laws=identified.laws,
        worlds=worlds,
        empirical=identified.empirical,
        limits=limits,
        rules=identified.rules,
        rebuild=lambda fresh, graph=graph, query=query, source=identified.source: _rebuild_laws(
            graph, query, fresh, source
        ),
    )
    prepared: PreparedAnalysis[Any] = PreparedAnalysis(native, kind="z_transport", query=query)
    prepared._transport_stage = study._transport_stage
    return prepared


def _rebuild_laws(graph: Admg, query: Any, data: Any, source_name: str | None) -> tuple[ExactDiscreteLaw, ...]:
    from ._day1 import _question_parts

    treatments, outcomes = _question_parts(query.question)
    source = next(source for source in query.evidence.sources if source.identity == source_name)
    bundle = _source_bundle(graph, query, data, source, outcomes, treatments)
    return bundle["laws"]


def _source_bundle(
    graph: Admg,
    query: Any,
    data: Any,
    source: Any,
    outcomes: Sequence[str],
    treatments: Sequence[str],
) -> dict[str, Any]:
    empirical, laws, assignment = _laws_for_source(graph, query, data, source)
    stage = identify_z_transport(
        graph=graph,
        query=ZTransportQuery(
            SelectionDiagram(
                source.identity,
                query.target,
                source.selections or query.resolved_selections,
            ),
            outcomes=list(outcomes),
            treatments=list(treatments),
            controllable=list(source.interventions),
            experiment_assignment=assignment,
        ),
    )
    catalog = _catalog(graph, query, source, laws, assignment)
    return {
        "source": source,
        "stage": stage,
        "catalog": catalog,
        "laws": laws,
        "empirical": empirical,
        "assignment": assignment,
    }


def _laws_for_source(
    graph: Admg, query: Any, data: Any, source: Any
) -> tuple[bool, tuple[ExactDiscreteLaw, ...], dict[str, float]]:
    from ._day1 import _snapshot_digest, _split_worlds, _table_columns, _tables_by_source

    if isinstance(data, ExactTransportData):
        owned = tuple(law for law in data.laws if law.population == source.identity)
        return False, _retag_exact(source, owned), _assignment(source, owned)
    samples: list[RegimeSample] = []
    if isinstance(data, StatisticalTransportData):
        samples = [sample for sample in data.samples if sample.population == source.identity]
    elif data is not None:
        tables = _tables_by_source(data, query.evidence.sources)
        table = tables.get(source.identity)
        if table is not None:
            columns = _table_columns(table)
            snapshot = source.snapshot or _snapshot_digest(columns)
            for interventions, world in _split_worlds(columns, source.interventions):
                if not interventions:
                    continue
                samples.append(
                    RegimeSample(
                        source.identity,
                        source.identity,
                        snapshot,
                        world,
                        interventions=interventions,
                    )
                )
    laws = tuple(_law_from_sample(graph, source, sample) for sample in samples)
    return True, laws, _assignment_from_pairs(source, [sample.interventions for sample in samples])


def _retag_exact(source: Any, laws: tuple[ExactDiscreteLaw, ...]) -> tuple[ExactDiscreteLaw, ...]:
    retagged = []
    for law in laws:
        assignment = {name: float(value) for name, value in law.interventions}
        retagged.append(
            ExactDiscreteLaw(
                source.identity,
                _regime_id(source.identity, assignment),
                law.axes,
                law.probabilities,
                law.snapshot_identity,
                interventions=law.interventions,
                absolute_tolerance=law.absolute_tolerance,
                relative_tolerance=law.relative_tolerance,
                empirical_counts=law.empirical_counts,
            )
        )
    return tuple(retagged)


def _law_from_sample(graph: Admg, source: Any, sample: RegimeSample) -> ExactDiscreteLaw:
    names = [name for name in graph.nodes() if name in sample.columns]
    if not names:
        raise CausalValueError("restricted-experiment sample measured no graph variable")
    levels = []
    for name in names:
        observed = sorted({float(value) for value in sample.columns[name] if value is not None})
        if not observed:
            raise CausalValueError(f"restricted-experiment sample has no finite values for {name}")
        levels.append(observed)
    index = {combo: position for position, combo in enumerate(itertools.product(*levels))}
    counts = [0] * len(index)
    width = len(next(iter(sample.columns.values())))
    for row in range(width):
        observed = [sample.columns[name][row] for name in names]
        if any(value is None for value in observed):
            raise CausalValueError("restricted-experiment transport requires complete observations")
        key = tuple(float(value) for value in observed)
        slot = index.get(key)
        if slot is None:
            raise CausalValueError("restricted-experiment sample left the declared finite domain")
        counts[slot] += 1
    total = sum(counts)
    if total == 0:
        raise CausalValueError("restricted-experiment sample is empty")
    assignment = {name: float(value) for name, value in sample.interventions}
    return ExactDiscreteLaw(
        source.identity,
        _regime_id(source.identity, assignment),
        tuple((name, tuple(observed)) for name, observed in zip(names, levels, strict=True)),
        tuple(count / total for count in counts),
        sample.snapshot_identity,
        interventions=tuple(sample.interventions),
        empirical_counts=tuple(counts),
    )


def _assignment(source: Any, laws: Sequence[ExactDiscreteLaw]) -> dict[str, float]:
    return _assignment_from_pairs(source, [law.interventions for law in laws])


def _assignment_from_pairs(
    source: Any, groups: Sequence[Sequence[tuple[str, float]]]
) -> dict[str, float]:
    controllable = list(source.interventions)
    found = []
    for group in groups:
        values = {name: float(value) for name, value in group}
        if set(controllable).issubset(values):
            found.append({name: values[name] for name in controllable})
    if not found:
        return {}
    return dict(sorted(found, key=lambda item: tuple(sorted(item.items())))[0])


def _catalog(
    graph: Admg,
    query: Any,
    source: Any,
    laws: Sequence[ExactDiscreteLaw],
    assignment: Mapping[str, float],
) -> EvidenceCatalog:
    from ._day1 import _coordinates_from_columns

    columns: dict[str, list[float]] = {name: [] for name in source.interventions}
    for law in laws:
        for name, values in law.axes:
            columns.setdefault(name, []).extend(values)
        for name, value in law.interventions:
            columns.setdefault(name, []).append(float(value))
    if not columns:
        for name in graph.nodes():
            columns[name] = []
    extras = list(graph.nodes())
    coordinates = _coordinates_from_columns(columns, extras)
    environments = [
        Environment(
            source.identity,
            coordinates,
            selection_targets=source.selections or query.resolved_selections,
        )
    ]
    regimes = []
    bindings = []
    seen: set[str] = set()
    groups = list(laws)
    if not groups and assignment:
        regimes.append(
            EvidenceRegime(
                _regime_id(source.identity, assignment),
                source.identity,
                kind="experimental",
                interventions=list(source.interventions),
                intervention_values=dict(assignment),
                measured=tuple(name for name in graph.nodes() if name not in source.interventions),
            )
        )
    for law in groups:
        if law.regime in seen:
            continue
        seen.add(law.regime)
        values = {name: float(value) for name, value in law.interventions}
        regimes.append(
            EvidenceRegime(
                law.regime,
                source.identity,
                kind="experimental",
                interventions=tuple(values),
                intervention_values=values,
                measured=tuple(name for name, _values in law.axes),
            )
        )
        bindings.append(
            RegimeBinding(
                law.regime,
                law.snapshot_identity,
                schema_names=tuple(name for name, _values in law.axes),
                sampling=source.sampling,
                dependence=source.dependence or "unknown_dependence",
            )
        )
    if not regimes:
        regimes.append(
            EvidenceRegime(
                source.identity,
                source.identity,
                kind="experimental",
                interventions=list(source.interventions),
                measured=(),
            )
        )
    return EvidenceCatalog(
        environments=environments,
        regimes=tuple(regimes),
        bindings=tuple(bindings),
        target_sampling=query.evidence.target_sampling,
    )


def _regime_id(population: str, assignment: Mapping[str, float]) -> str:
    parts = ",".join(f"{name}={assignment[name]}" for name in sorted(assignment))
    return f"{population}:{parts}" if parts else population


def _point(world: Mapping[str, float], payload: Mapping[str, Any]) -> TransportGridPoint:
    outcome_names = list(payload["outcomes"])
    means = {}
    for index, name in enumerate(outcome_names):
        means[name] = sum(
            atom[index] * probability
            for atom, probability in zip(payload["atoms"], payload["probabilities"], strict=True)
        )
    interval = payload.get("interval") or {}
    uncertainty = None
    if interval.get("available"):
        uncertainty = TransportUncertainty(
            {
                "available": True,
                "reason": interval.get("reason"),
                "mean_intervals": tuple(
                    (row["outcome"], float(row["lower"]), float(row["upper"]))
                    for row in interval.get("mean_intervals") or ()
                ),
            }
        )
    elif interval:
        uncertainty = TransportUncertainty(
            {"available": False, "reason": interval.get("reason") or "no_interval_reported"}
        )
    return TransportGridPoint(
        {
            "at": dict(world),
            "status": payload.get("status") or "available",
            "means": means,
            "uncertainty": None if uncertainty is None else uncertainty.to_dict(),
        }
    )
