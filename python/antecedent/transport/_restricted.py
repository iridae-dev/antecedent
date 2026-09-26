"""Route a declared controllable set through licensed single-source z-transport.

Classical and meta transport assume each source can experiment on every
variable. When every experimental source's declared interventions omit the
queried treatment, identification and execution stay on the z-transport stage.
"""

from __future__ import annotations

import functools
import itertools
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, cast

from .._transport_results import TransportGridPoint, TransportUncertainty
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
    TransportInference,
    TrialAipw,
    TrialAipwData,
    ZTransportQuery,
    consume_z_transport_artifact,
    identify_z_transport,
)

SCOPE = "single_source_z_transport_cited_joints_sound_incomplete"
#: The native two-source decider's stable refusal detail. It is not itself a
#: registered reason code: the refusal is raised as ``transport_not_certified``
#: and names this detail in its message.
COMBINATION_DETAIL = "z_transport.multi_source_combination_not_searched"
COMBINATION_CODE = "transport_not_certified"

#: Failure-snapshot statuses (``ZTransportFailureStatus``) and the
#: identification outcome each re-derived status means.
_SNAPSHOT_OUTCOME = {
    "proof_obstruction": "proven_non_transportable",
    "missing_evidence": "missing_evidence",
    "unsupported_input": "not_certified",
    "exhausted_computation": "not_certified",
    "unresolved_identification": "not_certified",
}

_TRANSFORM_INTENTS = (
    "display_precision",
    "compatible_data_replace",
    "retarget",
    "filter_display",
    "filter_population",
    "new_conditional_query",
    "change_graph",
    "change_prior",
    "change_physical_policy",
    "average_unweighted_class",
)
#: Intents the z-transport handle honours: display edits and a compatible
#: snapshot replacement (``replace_snapshot`` / ``refresh``).
_ALLOWED_INTENTS = frozenset({"display_precision", "filter_display", "compatible_data_replace"})


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
    #: Human sentence for a missing-evidence outcome, in variable names.
    detail: str | None = None
    #: Structured missing-evidence object from the native decision.
    missing: Mapping[str, Any] | None = None
    #: Checked line-11 obstruction records, one per source, when proven.
    obstructions: tuple[Mapping[str, Any], ...] = ()
    #: For ``combined_identified``: one checked factor per source, each with its
    #: ``source``, ``outcomes``, ``treatments``, ``proof`` and binding
    #: ``inspection``. Empty for every single-source outcome.
    components: tuple[Mapping[str, Any], ...] = ()
    #: How the combined factors' product equals the target law, when combined:
    #: ``independent_disconnected_components`` or ``intervention_separated_groups``.
    combination: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "rules", tuple(self.rules))
        object.__setattr__(self, "laws", tuple(self.laws))
        object.__setattr__(self, "obstructions", tuple(self.obstructions))
        object.__setattr__(self, "components", tuple(self.components))


class RestrictedTransportExecution:
    """Point results from the z-transport stage, one target assignment at a time.

    ``artifacts`` holds one independently consumable z-transport artifact per
    target assignment, in ``points`` order.
    """

    def __init__(
        self,
        *,
        points: tuple[TransportGridPoint, ...],
        rules: tuple[str, ...],
        artifacts: tuple[bytes, ...] = (),
    ) -> None:
        self.scope = SCOPE
        self.formula = SCOPE
        self.rules = rules
        self.points = points
        self.artifacts = artifacts
        first = points[0] if points else None
        self.uncertainty = None if first is None else first.uncertainty
        self.outcomes = () if first is None else tuple(first.means)

    def mean(self, outcome: str) -> float:
        if not self.points:
            raise CausalValueError("z-transport execution has no target assignment")
        return float(self.points[0].means[outcome])

    def export(self) -> bytes:
        """The single-assignment artifact; a grid exports per point via ``artifacts``."""
        if not self.artifacts:
            raise CausalUnsupportedError(
                "estimate before exporting a z-transport artifact", reason_code="not_executed"
            )
        if len(self.artifacts) != 1:
            raise CausalUnsupportedError(
                "a z-transport grid execution carries one artifact per target assignment; "
                "read .artifacts, or export the whole result with result.export()",
                reason_code="option_not_applicable",
            )
        return self.artifacts[0]

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


def consume_restricted_artifacts(
    artifacts: Sequence[bytes], worlds: Sequence[Mapping[str, float]] = ()
) -> RestrictedTransportExecution:
    """Independently recheck one z-transport artifact per world and rebuild the execution.

    Every number comes from the native consumer, which rechecks the embedded
    proof, catalog binding and laws and recomputes the point. The loaded
    execution is point-only: no interval is replayed.
    """
    if not artifacts:
        raise CausalValueError("a restricted-experiment execution needs at least one artifact")
    if worlds and len(worlds) != len(artifacts):
        raise CausalValueError("one z-transport artifact per target assignment is required")
    points = []
    rules: tuple[str, ...] = ()
    for index, artifact in enumerate(artifacts):
        payload = json.loads(consume_z_transport_artifact(bytes(artifact)))
        rules = tuple(payload.get("proof", {}).get("rules") or rules)
        world = worlds[index] if worlds else {}
        points.append(_point(world, payload))
    return RestrictedTransportExecution(
        points=tuple(points), rules=rules, artifacts=tuple(bytes(a) for a in artifacts)
    )


def identification_from_snapshot(snapshot: bytes) -> RestrictedTransportIdentification:
    """Rebuild a non-identified decision from a rechecked failure snapshot."""
    from .. import _native

    report = _native.consume_z_transport_failure_snapshot(bytes(snapshot))
    status = str(report.get("status"))
    outcome = _SNAPSHOT_OUTCOME.get(status, "not_certified")
    obligations = list(report.get("obligations") or ())
    obstruction = report.get("z_obstruction")
    reason = None
    detail = None
    if outcome == "missing_evidence":
        reason = "z_transport.missing_evidence"
        detail = obligations[0] if obligations else None
    elif outcome == "not_certified":
        reason = obligations[0] if obligations else f"z_transport.{status}"
    return RestrictedTransportIdentification(
        outcome=outcome,
        reason=reason,
        rules=(),
        formula=SCOPE,
        source=None,
        detail=detail,
        obstructions=() if obstruction is None else (obstruction,),
    )


class _RestrictedNative:
    """PreparedAnalysis native adapter over one prepared z-transport handle per world.

    Every target assignment keeps its own ``PreparedZTransportStage``. A
    snapshot with the same catalog binding is rebound through the native
    ``refresh``; a snapshot that changes the binding (a new snapshot identity)
    rebuilds laws and catalog and prepares again, so the proof is rebound
    rather than executed against a stale catalog.
    """

    def __init__(
        self,
        *,
        stage: Any,
        catalog: EvidenceCatalog,
        laws: tuple[ExactDiscreteLaw, ...],
        worlds: Sequence[Mapping[str, float]],
        empirical: bool,
        limits: TransportControls,
        seed: int,
        rules: tuple[str, ...],
        rebuild: Callable[[Any], tuple[tuple[ExactDiscreteLaw, ...], EvidenceCatalog]],
    ) -> None:
        self.stage = stage
        self.catalog = catalog
        self.laws = laws
        self.worlds = [dict(world) for world in worlds]
        self.empirical = empirical
        self.limits = limits
        self.seed = seed
        self.rules = rules
        self.rebuild = rebuild
        self.last: RestrictedTransportExecution | None = None
        #: The prepared study's frozen inputs; the catalog follows a rebound snapshot.
        self.stage_snapshot: Any = None
        self._prepared = self._prepare_all(catalog, laws, cancel=limits.cancel)

    def _prepare_all(
        self, catalog: EvidenceCatalog, laws: tuple[ExactDiscreteLaw, ...], *, cancel: Any
    ) -> list[Any]:
        kwargs = {
            "max_operations": self.limits.max_operations,
            "max_depth": self.limits.max_depth,
            "max_support_rows": self.limits.max_support_rows,
            "memory_bytes": self.limits.memory_bytes,
            "seed": self.seed,
            "cancel": cancel,
        }
        prepare = self.stage.prepare_empirical if self.empirical else self.stage.prepare_exact
        return [prepare(catalog, laws, world, **kwargs) for world in self.worlds]

    def replace_snapshot(self, data: Any, cancel: Any = None) -> None:
        cancel = self.limits.cancel if cancel is None else cancel
        laws, catalog = self.rebuild(data)
        if catalog == self.catalog:
            done = 0
            try:
                for prepared in self._prepared:
                    prepared.refresh(laws, cancel=cancel)
                    done += 1
            except Exception:
                # Restore the handles already rebound so a failed refresh leaves
                # every world on the previous valid snapshot.
                for prepared in self._prepared[:done]:
                    prepared.refresh(self.laws, cancel=cancel)
                raise
        else:
            self._prepared = self._prepare_all(catalog, laws, cancel=cancel)
        self.laws = laws
        self.catalog = catalog
        self.last = None
        if self.stage_snapshot is not None:
            self.stage_snapshot["catalog"] = catalog
            self.stage_snapshot["bound"] = data

    def estimate(self, cancel: Any = None) -> RestrictedTransportExecution:
        cancel = self.limits.cancel if cancel is None else cancel
        points = []
        artifacts = []
        for world, prepared in zip(self.worlds, self._prepared, strict=True):
            payload = json.loads(prepared.estimate(cancel=cancel))
            points.append(_point(world, payload))
            artifacts.append(bytes(prepared.export()))
        self.last = RestrictedTransportExecution(
            points=tuple(points), rules=self.rules, artifacts=tuple(artifacts)
        )
        return self.last

    def refresh(self, data: Any, cancel: Any = None) -> RestrictedTransportExecution:
        self.replace_snapshot(data, cancel=cancel)
        return self.estimate(cancel=cancel)

    def export(self) -> bytes:
        """The last execution's artifacts, framed as one loadable transport view."""
        if self.last is None:
            raise CausalUnsupportedError(
                "estimate before exporting the z-transport study", reason_code="not_executed"
            )
        return self.envelope(self.last)

    #: Set by :func:`prepare_restricted`: encodes an execution as the loadable view.
    envelope: Callable[[RestrictedTransportExecution], bytes]

    def plan_summary(self) -> dict[str, str]:
        bindings = ",".join(f"{law.regime}:{law.snapshot_identity}" for law in self.laws)
        return {
            "plan_id": f"z_transport:{bindings}:{len(self.worlds)}",
            "structure_source": "explicit",
            "deterministic_reductions": "true",
            "kernels": (
                "z_transport_empirical_plugin_point"
                if self.empirical
                else "z_transport_exact_point"
            ),
        }

    def preview_transform(self, intent: str) -> dict[str, str]:
        if intent not in _TRANSFORM_INTENTS:
            raise CausalValueError(f"unknown transform intent {intent!r}")
        report = {
            "intent": intent,
            "refused": str(intent not in _ALLOWED_INTENTS).lower(),
            "obligations": "" if intent in _ALLOWED_INTENTS else "reprepare",
        }
        if intent not in _ALLOWED_INTENTS:
            report["refusal_code"] = "option_not_applicable"
            report["refusal"] = (
                f"reason=option_not_applicable: {intent} is not licensed on the "
                "z-transport handle; prepare again"
            )
        return report

    def inspection_json(self) -> str:
        execution = self.last or RestrictedTransportExecution(points=(), rules=self.rules)
        return json.dumps(execution.inspect().to_dict())

    def freeze(self) -> _RestrictedNative:
        frozen = _RestrictedNative.__new__(_RestrictedNative)
        frozen.__dict__.update(self.__dict__)
        frozen._prepared = list(self._prepared)
        return frozen


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
    """Identify on the z-transport stage. Do not call classical or meta transport.

    One source is decided by the native single-source decision; two sources go
    through the native two-source decider, which searches each source
    separately and refuses cross-source combination by name.
    """

    from ._day1 import _question_parts

    treatments, outcomes = _question_parts(query.question)
    sources = [
        source
        for source in query.evidence.sources
        if source.kind == "experimental" and not set(treatments).issubset(source.interventions)
    ]
    if len(sources) > 2:
        raise CausalUnsupportedError(
            f"{COMBINATION_DETAIL}: restricted-experiment transport searches at most two "
            f"sources separately, {len(sources)} were declared",
            reason_code=COMBINATION_CODE,
        )
    if not sources:
        raise CausalValueError("restricted-experiment transport requires a controllable set")
    bundles = [
        _source_bundle(graph, query, data, source, outcomes, treatments) for source in sources
    ]
    if len(bundles) == 1:
        decision = bundles[0]["stage"].decide(bundles[0]["catalog"])
        return _decided(bundles[0], decision), bundles[0]["catalog"]
    from .. import _native

    decision = _native.decide_two_source_z_transport_stage(
        graph,
        query.target,
        list(outcomes),
        list(treatments),
        [
            (
                bundle["source"].identity,
                list(bundle["source"].interventions),
                dict(bundle["assignment"]),
                list(bundle["source"].selections or query.resolved_selections),
            )
            for bundle in bundles
        ],
        [bundle["catalog"] for bundle in bundles],
    )
    if decision["outcome"] == "identified":
        winner = next(b for b in bundles if b["source"].identity == decision["source"])
        return _decided(winner, decision), winner["catalog"]
    if decision["outcome"] == "combined_identified":
        # Each factor is transported from one source under the joint-regime rule;
        # the two factors' product is the target law. The disconnected and the
        # connected intervention-separated cases share this shape and differ only
        # in `combination`.
        components = tuple(decision.get("components") or ())
        rules = tuple(
            rule
            for component in components
            for rule in (component.get("proof", {}).get("rules") or ())
        )
        return (
            RestrictedTransportIdentification(
                outcome="combined_identified",
                reason=None,
                rules=rules,
                formula=SCOPE,
                source=None,
                stage=bundles[0]["stage"],
                components=components,
                combination=decision.get("combination"),
            ),
            bundles[0]["catalog"],
        )
    if decision["outcome"] == "proven_non_transportable":
        return (
            RestrictedTransportIdentification(
                outcome="proven_non_transportable",
                reason=None,
                rules=(),
                formula=SCOPE,
                source=None,
                stage=bundles[0]["stage"],
                obstructions=tuple(decision.get("obstructions") or ()),
            ),
            bundles[0]["catalog"],
        )
    return (
        RestrictedTransportIdentification(
            outcome="not_certified",
            reason=decision.get("reason") or COMBINATION_DETAIL,
            rules=(),
            formula=SCOPE,
            source=None,
            stage=bundles[0]["stage"],
        ),
        bundles[0]["catalog"],
    )


def _decided(bundle: Mapping[str, Any], decision: Mapping[str, Any]) -> Any:
    outcome = decision["outcome"]
    missing = decision.get("missing")
    return RestrictedTransportIdentification(
        outcome=outcome,
        reason=decision.get("reason"),
        rules=tuple(decision.get("proof", {}).get("rules") or ()),
        formula=SCOPE,
        source=bundle["source"].identity,
        stage=bundle["stage"],
        laws=bundle["laws"],
        empirical=bundle["empirical"],
        detail=decision.get("detail"),
        missing=None if missing is None else dict(missing),
        obstructions=tuple(
            [decision["obstruction"]] if decision.get("obstruction") is not None else ()
        ),
    )


def prepare_restricted(
    data: Any,
    *,
    query: Any,
    graph: Admg,
    provider: Any = None,
    inference: TransportInference | None = None,
    controls: TransportControls | None = None,
) -> Any:
    """Prepare the licensed z-transport stage for a restricted experiment."""

    from ..estimation import PreparedAnalysis
    from ._day1 import lower_question, transport_stage
    from ._wrap import encode_restricted_execution

    if isinstance(provider, (LearnedCategorical, TrialAipw)) or isinstance(data, TrialAipwData):
        raise CausalUnsupportedError(
            "Restricted-experiment transport executes the exact or empirical plugin. "
            "LearnedCategorical and TrialAipw stay on the classical route.",
            reason_code="option_not_applicable",
        )
    identified, catalog = identify_restricted(graph, query, data)
    shape, worlds = lower_question(query.question)
    limits = controls or TransportControls()
    stage_snapshot = transport_stage(
        identified=identified,
        catalog=catalog,
        bound=data,
        shape=shape,
        worlds=worlds,
        provider=EmpiricalTable() if identified.empirical else "exact_law",
        graph=graph,
    )
    if identified.outcome != "identified" or identified.stage is None:
        study: PreparedAnalysis[Any] = PreparedAnalysis(None, kind="z_transport", query=query)
        study._transport_stage = stage_snapshot
        return study
    native = _RestrictedNative(
        stage=identified.stage,
        catalog=catalog,
        laws=identified.laws,
        worlds=worlds,
        empirical=identified.empirical,
        limits=limits,
        seed=(inference or TransportInference()).seed,
        rules=identified.rules,
        rebuild=functools.partial(_rebuild, graph, query, source_name=identified.source),
    )
    prepared: PreparedAnalysis[Any] = PreparedAnalysis(native, kind="z_transport", query=query)
    prepared._transport_stage = stage_snapshot
    native.stage_snapshot = stage_snapshot
    native.envelope = functools.partial(encode_restricted_execution, prepared)
    return prepared


def _rebuild(
    graph: Admg, query: Any, data: Any, source_name: str | None
) -> tuple[tuple[ExactDiscreteLaw, ...], EvidenceCatalog]:
    """Laws and the catalog they bind to, for a replacement snapshot."""
    from ._day1 import _question_parts

    treatments, outcomes = _question_parts(query.question)
    source = next(source for source in query.evidence.sources if source.identity == source_name)
    bundle = _source_bundle(graph, query, data, source, outcomes, treatments)
    return bundle["laws"], bundle["catalog"]


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
        return False, _retag_exact(source, owned), _assignment_from_pairs(source, owned)
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
    return True, laws, _assignment_from_pairs(source, samples)


def _retag_exact(source: Any, laws: tuple[ExactDiscreteLaw, ...]) -> tuple[ExactDiscreteLaw, ...]:
    return tuple(
        replace(
            law,
            population=source.identity,
            regime=_regime_id(source.identity, dict(law.interventions)),
        )
        for law in laws
    )


def _law_from_sample(graph: Admg, source: Any, sample: RegimeSample) -> ExactDiscreteLaw:
    """Complete-observation frequency joint on the sample's observed finite domain."""
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
        observed_row = [sample.columns[name][row] for name in names]
        if any(value is None for value in observed_row):
            raise CausalValueError("restricted-experiment transport requires complete observations")
        complete = cast(list[float], observed_row)
        key = tuple(float(value) for value in complete)
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


def _assignment_from_pairs(source: Any, items: Sequence[Any]) -> dict[str, float]:
    """The lowest complete controllable assignment among laws or samples."""
    controllable = list(source.interventions)
    found = []
    for item in items:
        values = {name: float(value) for name, value in item.interventions}
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
    """One-source catalog with one experimental regime per intervention world.

    The day-1 catalog builder keys regimes by source; the z-transport theorem
    cites one joint per concrete ``do(z)`` world, so the regimes here carry
    the intervention values and the bindings carry each law's snapshot.
    """
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
    if not laws and assignment:
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
    for law in laws:
        if law.regime in seen:
            continue
        seen.add(law.regime)
        intervention_values = {name: float(value) for name, value in law.interventions}
        regimes.append(
            EvidenceRegime(
                law.regime,
                source.identity,
                kind="experimental",
                interventions=tuple(intervention_values),
                intervention_values=intervention_values,
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
    parts = ",".join(f"{name}={float(assignment[name])}" for name in sorted(assignment))
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
