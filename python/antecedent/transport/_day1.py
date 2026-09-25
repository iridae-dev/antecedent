"""Day-1 transport query, evidence constructor, and verb compile helpers."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field
from typing import Any, Literal, get_args

import numpy as np

from .._data import as_columns
from ..errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from ..graph import Admg
from ..query import (
    AverageDerivative,
    AverageEffect,
    DirectionalDerivative,
    Elasticity,
    PointDerivative,
    ResponseCurve,
    ResponseJacobian,
    SemiElasticity,
)
from ._impl import (
    ClassicalTransportIdentification,
    DependenceGroupName,
    EmpiricalTable,
    Environment,
    EvidenceCatalog,
    EvidenceRegime,
    ExactTransportData,
    LearnedCategorical,
    RegimeBinding,
    RegimeKindName,
    RegimeSample,
    SamplingDesignName,
    SelectionDiagram,
    StatisticalTransportData,
    TargetSamplingName,
    TransportControls,
    TransportInference,
    TrialAipw,
    TrialAipwData,
    VariableCoordinate,
    VariableDomainName,
    identify_classical,
    identify_meta,
    inspect_catalog,
)
from ._impl import (
    prepare as prepare_stage,
)

_RESPONSE_FAMILY = (
    ResponseCurve,
    AverageDerivative,
    PointDerivative,
    Elasticity,
    SemiElasticity,
    DirectionalDerivative,
    ResponseJacobian,
)
_Question = (
    AverageEffect
    | ResponseCurve
    | AverageDerivative
    | PointDerivative
    | Elasticity
    | SemiElasticity
    | DirectionalDerivative
    | ResponseJacobian
)


def _require_name(field_name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CausalValueError(f"{field_name} must be a non-empty name")
    return value.strip()


def _as_sources(source: Source | Sequence[Source]) -> tuple[Source, ...]:
    if isinstance(source, Source):
        return (source,)
    sources = tuple(source)
    if not sources:
        raise CausalValueError("evidence requires at least one source")
    if not all(isinstance(item, Source) for item in sources):
        raise CausalTypeError("evidence.source must be Source or a sequence of Source")
    return sources


def _dependence_for(sampling: str, dependence: DependenceGroupName | None) -> DependenceGroupName:
    if dependence is not None:
        if dependence not in get_args(DependenceGroupName):
            raise CausalValueError(f"unknown dependence group {dependence!r}")
        return dependence
    return "independent_studies" if sampling == "independent" else "unknown_dependence"


@dataclass(frozen=True, slots=True)
class Source:
    """One source population and the regime that produced its evidence.

    Population identity, experimental vs observational status, the intervention
    set, and sampling/dependence are scientific claims. Column names and the
    snapshot digest are filled from ``data`` at prepare time.
    """

    identity: str
    _: KW_ONLY
    kind: RegimeKindName
    interventions: Sequence[str] = ()
    sampling: SamplingDesignName
    dependence: DependenceGroupName | None = None
    measured: Sequence[str] | None = None
    selections: Sequence[str] = ()
    snapshot: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "identity", _require_name("source identity", self.identity))
        if self.kind not in get_args(RegimeKindName):
            raise CausalValueError(f"unknown regime kind {self.kind!r}")
        if self.sampling not in get_args(SamplingDesignName):
            raise CausalValueError(f"unknown sampling design {self.sampling!r}")
        object.__setattr__(self, "interventions", tuple(self.interventions))
        object.__setattr__(self, "selections", tuple(self.selections))
        if self.measured is not None:
            object.__setattr__(self, "measured", tuple(self.measured))
        object.__setattr__(self, "dependence", _dependence_for(self.sampling, self.dependence))
        if self.kind == "experimental" and not self.interventions:
            raise CausalValueError("experimental sources require a non-empty intervention set")
        if self.kind == "observational" and self.interventions:
            raise CausalValueError("observational sources cannot carry hard interventions")


@dataclass(frozen=True, slots=True)
class Evidence:
    """Scientific transport claims. Clerical bindings come from ``data``."""

    source: Source | Sequence[Source]
    _: KW_ONLY
    target_sampling: TargetSamplingName

    def __post_init__(self) -> None:
        if self.target_sampling not in get_args(TargetSamplingName):
            raise CausalValueError(f"unknown target sampling {self.target_sampling!r}")
        object.__setattr__(self, "source", _as_sources(self.source))

    @property
    def sources(self) -> tuple[Source, ...]:
        return tuple(self.source)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class Transport:
    """Ordinary Antecedent question, asked in a target population.

    Source identity, intervention regime, and sampling claims live on
    :class:`Evidence`. Selection differences stay explicit.
    """

    question: _Question
    target: str
    evidence: Evidence
    _: KW_ONLY
    selections: Sequence[str] = ()
    kind: Literal["transport_compiler"] = field(
        default="transport_compiler", init=False, repr=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "target", _require_name("target", self.target))
        object.__setattr__(self, "selections", tuple(self.selections))
        if not isinstance(self.evidence, Evidence):
            raise CausalTypeError("evidence must be transport.Evidence")
        if not isinstance(self.question, (AverageEffect, *_RESPONSE_FAMILY)):
            raise CausalTypeError(
                "Transport.question must be AverageEffect or a response-family query"
            )
        if getattr(self.question, "target_population", None) is not None:
            raise CausalValueError(
                "Transport owns the target population; the inner question must not set "
                "target_population"
            )
        identities = [source.identity for source in self.evidence.sources]
        if self.target in identities:
            raise CausalValueError("target population must be distinct from every source")
        if len(set(identities)) != len(identities):
            raise CausalValueError("source identities must be distinct")

    @property
    def resolved_selections(self) -> tuple[str, ...]:
        return tuple(self.selections)


def _question_parts(question: _Question) -> tuple[list[str], list[str]]:
    if isinstance(question, (DirectionalDerivative, ResponseJacobian)):
        return list(question.treatments), list(question.outcomes)
    return [question.treatment], [question.outcome]


def lower_question(question: _Question) -> tuple[str, list[dict[str, float]]]:
    """Return ``("contrast" | "scalar" | "grid", intervention worlds)``."""

    treatments, _outcomes = _question_parts(question)
    if isinstance(question, AverageEffect):
        return (
            "contrast",
            [
                {treatments[0]: float(question.control_level)},
                {treatments[0]: float(question.active_level)},
            ],
        )
    if isinstance(question, ResponseCurve):
        grid = [float(value) for value in question.grid]
        if len(grid) == 1:
            return "scalar", [{treatments[0]: grid[0]}]
        return "grid", [{treatments[0]: value} for value in grid]
    at = getattr(question, "at", None)
    if isinstance(at, Mapping):
        return "scalar", [{name: float(at[name]) for name in treatments}]
    if isinstance(at, Sequence) and not isinstance(at, (str, bytes)):
        if len(treatments) == 1 and len(at) == 1:
            return "scalar", [{treatments[0]: float(at[0])}]
        return "scalar", [dict(zip(treatments, (float(v) for v in at), strict=True))]
    raise CausalValueError("transport question does not declare an intervention world")


def _snapshot_digest(columns: Mapping[str, Sequence[Any]]) -> str:
    digest = hashlib.sha256()
    for name in sorted(columns):
        digest.update(name.encode())
        digest.update(np.asarray(list(columns[name]), dtype=np.float64).tobytes())
    return digest.hexdigest()[:16]


def _domain_for(values: Sequence[Any]) -> tuple[VariableDomainName, int | None]:
    finite = [float(value) for value in values if value is not None and not math.isnan(value)]
    unique = sorted(set(finite))
    if unique and all(item in (0.0, 1.0) for item in unique):
        return "binary", None
    if unique and all(float(item).is_integer() for item in unique) and len(unique) <= 16:
        return "categorical", len(unique)
    return "unspecified", None


def _is_column_table(data: Any) -> bool:
    if isinstance(
        data, (ExactTransportData, StatisticalTransportData, TrialAipwData, EvidenceCatalog)
    ):
        return False
    if not isinstance(data, Mapping) or not data:
        return False
    sample = next(iter(data.values()))
    if isinstance(sample, Mapping):
        return False
    return not (hasattr(sample, "columns") and hasattr(sample, "to_numpy"))


def _table_columns(data: Any) -> dict[str, tuple[float | None, ...]]:
    if isinstance(data, Mapping) and data and isinstance(next(iter(data.values())), Mapping):
        raise CausalTypeError("expected a column table, not a mapping of tables")
    names, columns = as_columns(data)
    return {
        name: tuple(None if math.isnan(value) else float(value) for value in column)
        for name, column in zip(names, columns, strict=True)
    }


def _split_worlds(
    columns: Mapping[str, Sequence[float | None]],
    interventions: Sequence[str],
) -> list[tuple[tuple[tuple[str, float], ...], dict[str, tuple[float | None, ...]]]]:
    if not interventions or any(name not in columns for name in interventions):
        return [((), {name: tuple(values) for name, values in columns.items()})]
    n = len(next(iter(columns.values())))
    groups: dict[tuple[tuple[str, float], ...], list[int]] = {}
    for index in range(n):
        key = tuple((name, float(columns[name][index])) for name in interventions)  # type: ignore[arg-type]
        groups.setdefault(key, []).append(index)
    worlds = []
    for key, rows in groups.items():
        world = {
            name: tuple(values[i] for i in rows)
            for name, values in columns.items()
            if name not in interventions
        }
        worlds.append((key, world))
    return worlds


def _coordinates_from_columns(
    columns: Mapping[str, Sequence[Any]], extra: Sequence[str] = ()
) -> list[VariableCoordinate]:
    names = list(dict.fromkeys([*columns, *extra]))
    coords = []
    for name in names:
        if name in columns:
            domain, cardinality = _domain_for(columns[name])
        else:
            domain, cardinality = "unspecified", None
        coords.append(VariableCoordinate(name, domain, cardinality=cardinality))
    return coords


def _tables_by_source(data: Any, sources: Sequence[Source]) -> dict[str, Any]:
    if data is None:
        return {}
    if _is_column_table(data):
        if len(sources) != 1:
            raise CausalValueError(
                "a single table binds one source; pass a mapping of population tables"
            )
        return {sources[0].identity: data}
    if not isinstance(data, Mapping):
        raise CausalTypeError("transport data must be a table, a mapping of tables, or typed data")
    tables: dict[str, Any] = {}
    known = {source.identity for source in sources}
    for key, table in data.items():
        if key not in known:
            raise CausalValueError(
                f"data table {key!r} does not match a declared source; "
                "population identity is a scientific claim and is not inferred"
            )
        tables[str(key)] = table
    return tables


def catalog_from_evidence(
    query: Transport,
    data: Any | None = None,
    *,
    graph: Admg | None = None,
) -> tuple[EvidenceCatalog, ExactTransportData | StatisticalTransportData | TrialAipwData | None]:
    """Build an engine catalog. Infer only clerical facts from ``data``."""

    if isinstance(data, TrialAipwData):
        return _catalog_without_rows(query, graph), data
    if isinstance(data, ExactTransportData):
        return _catalog_from_exact(query, data, graph), data
    if isinstance(data, StatisticalTransportData):
        return _catalog_from_statistical(query, data, graph), data

    sources = query.evidence.sources
    tables = _tables_by_source(data, sources)
    environments: list[Environment] = []
    regimes: list[EvidenceRegime] = []
    bindings: list[RegimeBinding] = []
    samples: list[RegimeSample] = []
    graph_names = list(graph.nodes()) if graph is not None else []

    for source in sources:
        table = tables.get(source.identity)
        columns = _table_columns(table) if table is not None else {}
        worlds = _split_worlds(columns, source.interventions) if columns else [((), {})]
        measured = (
            tuple(source.measured)
            if source.measured is not None
            else tuple(
                name for name in (worlds[0][1] or columns) if name not in source.interventions
            )
            if (worlds[0][1] or columns)
            else ()
        )
        extras = [*graph_names, *source.interventions, *measured, *source.selections]
        domain_columns = dict(columns)
        for interventions, _world in worlds:
            for name, value in interventions:
                domain_columns.setdefault(name, ())
                domain_columns[name] = tuple(domain_columns[name]) + (value,)
        environments.append(
            Environment(
                source.identity,
                _coordinates_from_columns(domain_columns or {name: () for name in extras}, extras),
                selection_targets=source.selections or query.resolved_selections,
            )
        )
        regimes.append(
            EvidenceRegime(
                source.identity,
                source.identity,
                kind=source.kind,
                interventions=source.interventions,
                measured=measured,
            )
        )
        if table is None:
            continue
        snapshot = source.snapshot or _snapshot_digest(columns)
        bindings.append(
            RegimeBinding(
                source.identity,
                snapshot,
                schema_names=tuple(columns),
                sampling=source.sampling,
                dependence=source.dependence or "unknown_dependence",
            )
        )
        for interventions, world_columns in worlds:
            assigned = interventions or tuple((name, float("nan")) for name in source.interventions)
            if assigned and assigned[0][1] != assigned[0][1]:
                continue
            samples.append(
                RegimeSample(
                    source.identity,
                    source.identity,
                    snapshot,
                    world_columns or columns,
                    interventions=assigned if assigned and assigned[0][1] == assigned[0][1] else (),
                )
            )

    shared = []
    seen_vars: set[str] = set()
    for environment in environments:
        for coordinate in environment.variables:
            if coordinate.name in seen_vars:
                continue
            seen_vars.add(coordinate.name)
            shared.append(coordinate)
    for name in graph_names:
        if name not in seen_vars:
            shared.append(VariableCoordinate(name))
            seen_vars.add(name)
    environments.append(Environment(query.target, shared))
    catalog = EvidenceCatalog(
        environments=environments,
        regimes=regimes,
        bindings=bindings,
        target_sampling=query.evidence.target_sampling,
    )
    bound: StatisticalTransportData | None = (
        StatisticalTransportData(samples=tuple(samples)) if samples else None
    )
    return catalog, bound


def _catalog_without_rows(query: Transport, graph: Admg | None) -> EvidenceCatalog:
    graph_names = list(graph.nodes()) if graph is not None else []
    environments = []
    regimes = []
    for source in query.evidence.sources:
        extras = [
            *graph_names,
            *source.interventions,
            *(source.measured or ()),
            *source.selections,
        ]
        environments.append(
            Environment(
                source.identity,
                [VariableCoordinate(name) for name in dict.fromkeys(extras)],
                selection_targets=source.selections or query.resolved_selections,
            )
        )
        regimes.append(
            EvidenceRegime(
                source.identity,
                source.identity,
                kind=source.kind,
                interventions=source.interventions,
                measured=source.measured or (),
            )
        )
    environments.append(
        Environment(query.target, [VariableCoordinate(name) for name in graph_names])
    )
    return EvidenceCatalog(
        environments=environments,
        regimes=regimes,
        target_sampling=query.evidence.target_sampling,
    )


def _catalog_from_exact(
    query: Transport, data: ExactTransportData, graph: Admg | None
) -> EvidenceCatalog:
    columns: dict[str, list[float]] = {}
    for law in data.laws:
        for name, values in law.axes:
            columns.setdefault(name, [])
            columns[name].extend(values)
        for name, value in law.interventions:
            columns.setdefault(name, [])
            columns[name].append(float(value))
    catalog, _bound = catalog_from_evidence(query, None, graph=graph)
    environments = []
    for environment in catalog.environments:
        extras = [coordinate.name for coordinate in environment.variables]
        environments.append(
            Environment(
                environment.identity,
                _coordinates_from_columns(columns, extras),
                selection_targets=environment.selection_targets,
            )
        )
    by_source = {source.identity: source for source in query.evidence.sources}
    regimes = []
    seen_regimes: set[str] = set()
    for law in data.laws:
        source = by_source.get(law.population)
        if source is None or law.regime in seen_regimes:
            continue
        seen_regimes.add(law.regime)
        measured = source.measured or tuple(name for name, _values in law.axes)
        regimes.append(
            EvidenceRegime(
                law.regime,
                source.identity,
                kind=source.kind,
                interventions=source.interventions,
                measured=tuple(dict.fromkeys(measured)),
            )
        )
    bindings = []
    for law in data.laws:
        source = by_source.get(law.population)
        if source is None:
            continue
        bindings.append(
            RegimeBinding(
                law.regime,
                law.snapshot_identity,
                schema_names=tuple(name for name, _values in law.axes),
                sampling=source.sampling,
                dependence=source.dependence or "unknown_dependence",
            )
        )
    seen: set[tuple[str, str]] = set()
    unique = []
    for binding in bindings:
        key = (binding.regime, binding.snapshot_identity)
        if key in seen:
            continue
        seen.add(key)
        unique.append(binding)
    return EvidenceCatalog(
        environments=environments,
        regimes=regimes,
        bindings=unique,
        target_sampling=query.evidence.target_sampling,
    )


def _catalog_from_statistical(
    query: Transport, data: StatisticalTransportData, graph: Admg | None
) -> EvidenceCatalog:
    columns: dict[str, list[float | None]] = {}
    for sample in data.samples:
        for name, values in sample.columns.items():
            columns.setdefault(name, [])
            columns[name].extend(values)
        for name, value in sample.interventions:
            columns.setdefault(name, [])
            columns[name].append(float(value))
    for law in data.laws:
        for name, values in law.axes:
            columns.setdefault(name, [])
            columns[name].extend(values)
        for name, value in law.interventions:
            columns.setdefault(name, [])
            columns[name].append(float(value))
    catalog, _bound = catalog_from_evidence(query, None, graph=graph)
    environments = [
        Environment(
            environment.identity,
            _coordinates_from_columns(
                columns, [coordinate.name for coordinate in environment.variables]
            ),
            selection_targets=environment.selection_targets,
        )
        for environment in catalog.environments
    ]
    by_source = {source.identity: source for source in query.evidence.sources}
    regimes = []
    seen_regimes: set[str] = set()
    for sample in (*data.samples,):
        source = by_source.get(sample.population)
        if source is None or sample.regime in seen_regimes:
            continue
        seen_regimes.add(sample.regime)
        measured = source.measured or tuple(sample.columns)
        regimes.append(
            EvidenceRegime(
                sample.regime,
                source.identity,
                kind=source.kind,
                interventions=source.interventions,
                measured=tuple(dict.fromkeys(measured)),
            )
        )
    bindings = []
    seen_bindings: set[tuple[str, str]] = set()
    for sample in data.samples:
        source = by_source.get(sample.population)
        if source is None:
            continue
        key = (sample.regime, sample.snapshot_identity)
        if key in seen_bindings:
            continue
        seen_bindings.add(key)
        bindings.append(
            RegimeBinding(
                sample.regime,
                sample.snapshot_identity,
                schema_names=tuple(sample.columns),
                sampling=source.sampling,
                dependence=source.dependence or "unknown_dependence",
            )
        )
    for law in data.laws:
        source = by_source.get(law.population)
        if source is None:
            continue
        key = (law.regime, law.snapshot_identity)
        if key in seen_bindings:
            continue
        seen_bindings.add(key)
        bindings.append(
            RegimeBinding(
                law.regime,
                law.snapshot_identity,
                schema_names=tuple(name for name, _values in law.axes),
                sampling=source.sampling,
                dependence=source.dependence or "unknown_dependence",
            )
        )
    return EvidenceCatalog(
        environments=environments,
        regimes=regimes,
        bindings=bindings,
        target_sampling=query.evidence.target_sampling,
    )


def identify_transport(
    graph: Admg, query: Transport, data: Any | None = None
) -> tuple[Any, EvidenceCatalog]:
    if not isinstance(graph, Admg):
        raise CausalTypeError("transport identify requires graph=Admg(...)")
    from ._restricted import identify_restricted, restricted_experiment

    if restricted_experiment(query):
        return identify_restricted(graph, query, data)
    catalog, _bound = catalog_from_evidence(query, graph=graph)
    treatments, outcomes = _question_parts(query.question)
    sources = query.evidence.sources
    if len(sources) == 1:
        identified = identify_classical(
            graph,
            SelectionDiagram(sources[0].identity, query.target, query.resolved_selections),
            outcomes=outcomes,
            treatments=treatments,
        )
    else:
        identified = identify_meta(
            graph,
            catalog,
            target=query.target,
            outcomes=outcomes,
            treatments=treatments,
        )
    return identified, catalog


def _inner_phrase(query: Transport | None) -> str:
    if query is None:
        return "this query"
    from .._claim import query_phrase

    return query_phrase(query.question)


def _missing_source(search: Mapping[str, Any] | None) -> str:
    missing = (
        () if search is None else (search.get("missing_factors") or search.get("missing") or ())
    )
    if isinstance(missing, str):
        missing = [missing]
    if not missing:
        return "a source"
    first = missing[0]
    if isinstance(first, Mapping):
        return str(first.get("population") or first.get("source") or "a source")
    return str(first)


def missing_evidence_detail(
    identified: ClassicalTransportIdentification,
    catalog: EvidenceCatalog,
    *,
    search: Mapping[str, Any] | None = None,
    query: Transport | None = None,
) -> str:
    report = search if search is not None else inspect_catalog(identified, catalog)
    phrase = _inner_phrase(query)
    target = query.target if query is not None else "the target"
    source = _missing_source(report)
    return f"{phrase} is identified for {target}, but the required joint in {source} is unbound."


def not_certified_detail(
    identified: ClassicalTransportIdentification,
    query: Transport | None = None,
) -> str:
    phrase = _inner_phrase(query)
    target = query.target if query is not None else "the target"
    reason = getattr(identified, "reason", None)
    if identified.outcome == "proven_non_transportable":
        base = f"{phrase} is not transportable into {target} under this selection diagram."
        return f"{base} ({reason})" if reason else base
    if reason:
        return f"{phrase} is not certified for transport into {target} ({reason})."
    return f"{phrase} is not certified for transport into {target} (conservative refusal)."


def default_provider(
    data: Any, provider: EmpiricalTable | LearnedCategorical | TrialAipw | str | None
) -> EmpiricalTable | LearnedCategorical | TrialAipw | str | None:
    if provider is not None:
        return provider
    if isinstance(data, ExactTransportData):
        return None
    if isinstance(data, TrialAipwData):
        raise CausalUnsupportedError(
            "Trial AIPW requires known randomization and an explicit provider=TrialAipw(...). "
            "The conservative default is EmpiricalTable(); it will not auto-promote.",
            reason_code="option_not_applicable",
        )
    return EmpiricalTable()


def refuse_transport_only_kwargs(
    query: object,
    *,
    provider: object = None,
    inference: object = None,
    controls: object = None,
) -> None:
    if isinstance(query, Transport):
        return
    extra = []
    if provider is not None:
        extra.append("provider")
    if isinstance(inference, TransportInference):
        extra.append("inference")
    if controls is not None:
        extra.append("controls")
    if extra:
        names = ", ".join(f"{name}=" for name in extra)
        raise CausalUnsupportedError(
            f"{names} apply only to transport.Transport queries",
            reason_code="option_not_applicable",
        )


def prepare_transport(
    data: Any,
    *,
    query: Transport,
    graph: Admg,
    provider: EmpiricalTable | LearnedCategorical | TrialAipw | str | None = None,
    inference: TransportInference | None = None,
    controls: TransportControls | None = None,
    cancel: Any = None,
) -> Any:
    """Compile the T5–T9 request that PreparedAnalysis already knows."""

    from ..estimation import PreparedAnalysis
    from ._restricted import prepare_restricted, restricted_experiment

    if restricted_experiment(query):
        return prepare_restricted(
            data,
            query=query,
            graph=graph,
            provider=provider,
            controls=controls or TransportControls(),
        )

    catalog, bound = catalog_from_evidence(query, data, graph=graph)
    identified, _catalog = identify_transport(graph, query, data)
    catalog = _catalog if bound is None else catalog
    shape, worlds = lower_question(query.question)
    resolved = default_provider(bound if bound is not None else data, provider)
    exact = isinstance(bound if bound is not None else data, ExactTransportData)
    settings = inference if inference is not None else (None if exact else TransportInference())
    limits = controls or TransportControls()
    if cancel is not None and limits.cancel is None:
        limits = TransportControls(
            max_operations=limits.max_operations,
            max_depth=limits.max_depth,
            max_support_rows=limits.max_support_rows,
            memory_bytes=limits.memory_bytes,
            cancel=cancel,
        )
    if identified.outcome != "identified":
        study: PreparedAnalysis[Any] = PreparedAnalysis(None, kind="statistical_transport")
        study._query = query
        study._transport_stage = {
            "identified": identified,
            "catalog": catalog,
            "bound": bound,
            "shape": shape,
            "worlds": worlds,
            "provider": resolved,
            "graph": graph,
        }
        return study
    payload = bound if bound is not None else data
    if isinstance(resolved, TrialAipw) or isinstance(payload, TrialAipwData):
        if not isinstance(resolved, TrialAipw) or not isinstance(payload, TrialAipwData):
            raise CausalUnsupportedError(
                "Trial AIPW requires known randomization and an explicit "
                "provider=TrialAipw(...) with TrialAipwData. "
                "EmpiricalTable() is the conservative default and does not auto-promote.",
                reason_code="option_not_applicable",
            )
        if shape != "contrast":
            raise CausalUnsupportedError(
                "Trial AIPW estimates a binary contrast; use AverageEffect.",
                reason_code="option_not_applicable",
            )
        from ._impl import TrialAipwQuery, prepare_trial

        treatments, outcomes = _question_parts(query.question)
        sources = query.evidence.sources
        if len(sources) != 1:
            raise CausalValueError("Trial AIPW requires a single source")
        request = TrialAipwQuery(
            graph,
            SelectionDiagram(sources[0].identity, query.target, query.resolved_selections),
            treatments[0],
            outcomes[0],
        )
        study = prepare_trial(
            request,
            payload,
            provider=resolved,
            inference=settings or TransportInference(),
            controls=limits,
        )
        study._query = query
        study._transport_stage = {
            "identified": identified,
            "catalog": catalog,
            "bound": payload,
            "shape": shape,
            "worlds": worlds,
            "provider": resolved,
            "graph": graph,
        }
        return study
    if isinstance(payload, ExactTransportData) and isinstance(resolved, LearnedCategorical):
        raise CausalValueError("Exact laws do not accept learner or sampling inference settings")
    inference_knobs = settings or TransportInference()
    if shape == "scalar":
        # Distinct from `request` above (the TrialAipwQuery branch, which
        # always returns before this point): this local is legitimately
        # polymorphic across the exact/statistical split below, and
        # `prepare_stage` (the overloaded `prepare`) is called uniformly with
        # `provider`/`inference` regardless of which one it holds — the exact
        # branch relies on that call's own "exact laws do not accept learner
        # or sampling inference settings" check when `resolved`/`settings`
        # are not None.
        scalar_request: Any
        if isinstance(payload, ExactTransportData):
            from ._impl import ExactTransportQuery

            scalar_request = ExactTransportQuery(identified, catalog, worlds[0])
        else:
            from ._impl import StatisticalTransportQuery

            if not isinstance(payload, StatisticalTransportData):
                raise CausalValueError(
                    "Statistical transport needs a bound snapshot; pass a table or "
                    "StatisticalTransportData"
                )
            scalar_request = StatisticalTransportQuery(
                identified,
                catalog,
                worlds[0],
                bootstrap=inference_knobs.bootstrap,
                coverage_level=inference_knobs.coverage_level,
                estimator=resolved or EmpiricalTable(),
                seed=inference_knobs.seed,
            )
        study = prepare_stage(
            scalar_request, payload, provider=resolved, inference=settings, controls=limits
        )
    else:
        from ._impl import TransportResponseGridQuery

        if payload is None:
            raise CausalValueError("grid transport needs a bound snapshot")
        grid_request = TransportResponseGridQuery(
            identified,
            catalog,
            worlds,
            bootstrap=inference_knobs.bootstrap,
            coverage_level=inference_knobs.coverage_level,
            seed=inference_knobs.seed,
            estimator=resolved or "plugin",
        )
        study = prepare_stage(
            grid_request, payload, provider=resolved, inference=settings, controls=limits
        )
    study._query = query
    study._transport_stage = {
        "identified": identified,
        "catalog": catalog,
        "bound": payload,
        "shape": shape,
        "worlds": worlds,
        "provider": resolved,
        "graph": graph,
    }
    return study


#: Native catalog-search budget/cancellation messages
#: (``crates/antecedent-identify/src/sid/meta.rs``: ``"transport.cancelled"``,
#: ``"transport.binding_budget"``, ``"transport.memory_budget"``,
#: ``"transport.identification_budget"``). ``inspect_catalog`` raises a bare
#: ``ValueError`` for these (the native binding does not yet distinguish them
#: by exception type — see the module docstring note below), so the message
#: is the only signal available in Python; anything else is an unrecognized
#: failure and must propagate rather than be treated as an incomplete search.
_CATALOG_SEARCH_INCOMPLETE_REASONS = frozenset(
    {
        "transport.cancelled",
        "transport.binding_budget",
        "transport.memory_budget",
        "transport.identification_budget",
    }
)


def _catalog_search_incomplete_detail(query: Transport | None, reason: str) -> str:
    phrase = _inner_phrase(query)
    target = query.target if query is not None else "the target"
    return (
        f"{phrase} is identified for {target}, but the evidence catalog search "
        f"did not finish ({reason}); whether the required joint is bound is unknown."
    )


def _identification_from_restricted(graph: Admg, query: Transport, identified: Any) -> Any:
    from ..identify import Identification

    outcome = identified.outcome
    if outcome == "identified":
        status = "NonparametricallyIdentified"
        note = "identified"
    elif outcome == "missing_evidence":
        status = "NotIdentified"
        note = "missing_evidence"
    else:
        status = "NotIdentified"
        note = outcome
    certificate = {
        "outcome": note,
        "rules": list(identified.rules),
        "native_outcome": outcome,
        "scope": identified.scope,
        "source": identified.source,
        "missing_detail": (
            f"{_inner_phrase(query)} is identified for {query.target}, but {identified.reason} is unbound."
            if note == "missing_evidence"
            else None
        ),
        "not_certified_detail": None
        if note not in {"not_certified", "proven_non_transportable"}
        else not_certified_detail(identified, query=query),
        "engine": {
            "formula": identified.formula,
            "scope": identified.scope,
            "source": identified.source,
            "rules": list(identified.rules),
            "native_outcome": outcome,
            "reason": identified.reason,
        },
    }
    return Identification(
        status=status,
        method="identify.transport_sid",
        adjustment_set=[],
        graph=graph,
        query=query,
        identifier="transport.sid",
        certificate=certificate,
        assumption_count=len(identified.rules),
        derivation_step_count=len(identified.rules),
    )


def identification_from_transport(
    graph: Admg,
    query: Transport,
    *,
    catalog: EvidenceCatalog | None = None,
    identified: ClassicalTransportIdentification | None = None,
) -> Any:
    from ..identify import Identification

    identified, built = (
        (identified, catalog)
        if identified is not None and catalog is not None
        else identify_transport(graph, query)
    )
    catalog = built if catalog is None else catalog
    from ._restricted import RestrictedTransportIdentification

    if isinstance(identified, RestrictedTransportIdentification):
        return _identification_from_restricted(graph, query, identified)
    search = None
    search_incomplete_reason: str | None = None
    try:
        search = inspect_catalog(identified, catalog)
    except Exception as error:
        message = str(error)
        if message not in _CATALOG_SEARCH_INCOMPLETE_REASONS:
            raise
        search_incomplete_reason = message
    outcome = identified.outcome
    if search_incomplete_reason is not None:
        # The search never finished, so whether the required evidence is
        # bound is unknown — this can never be reported as "identified" with
        # nothing missing.
        status = "NonparametricallyIdentified" if outcome == "identified" else "NotIdentified"
        note = "catalog_search_incomplete"
    elif outcome == "identified" and search and search.get("outcome") == "missing_evidence":
        status = "NonparametricallyIdentified"
        note = "missing_evidence"
    elif outcome == "identified":
        status = "NonparametricallyIdentified"
        note = "identified"
    elif outcome == "missing_evidence":
        status = "NotIdentified"
        note = "missing_evidence"
    else:
        status = "NotIdentified"
        note = outcome
    certificate = {
        "outcome": note,
        "rules": list(identified.rules),
        "native_outcome": outcome,
        "missing_detail": (
            missing_evidence_detail(identified, catalog, search=search, query=query)
            if note == "missing_evidence"
            else _catalog_search_incomplete_detail(query, search_incomplete_reason)
            if note == "catalog_search_incomplete" and search_incomplete_reason is not None
            else None
        ),
        "not_certified_detail": None
        if note not in {"not_certified", "proven_non_transportable"} and outcome == "identified"
        else not_certified_detail(identified, query=query),
        "engine": {
            "formula": identified.formula,
            "catalog_search": search,
            "rules": list(identified.rules),
            "native_outcome": outcome,
            "catalog_search_incomplete_reason": search_incomplete_reason,
        },
    }
    return Identification(
        status=status,
        method="identify.transport_sid",
        adjustment_set=[],
        graph=graph,
        query=query,
        identifier="transport.sid",
        certificate=certificate,
        assumption_count=len(identified.rules),
        derivation_step_count=len(identified.rules),
    )


__all__ = [
    "Evidence",
    "Source",
    "Transport",
    "catalog_from_evidence",
    "default_provider",
    "identify_transport",
    "identification_from_transport",
    "lower_question",
    "missing_evidence_detail",
    "prepare_transport",
    "refuse_transport_only_kwargs",
]
