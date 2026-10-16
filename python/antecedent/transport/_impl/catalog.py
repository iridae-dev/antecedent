from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, TypedDict, get_args

from ..._columns import coerce_fields
from ...errors import CausalValueError
from ...graph import Admg
from .limits import _non_negative

if TYPE_CHECKING:
    from .exact import ExactTransportData
    from .statistical import EmpiricalTable, LearnedCategorical, StatisticalTransportData
    from .trial import TrialAipw, TrialAipwData


class TransportStage(TypedDict):
    """Snapshot of a not-yet-identified (or otherwise deferred) transport prepare.

    Frozen onto ``PreparedAnalysis._transport_stage`` by
    :func:`antecedent.transport._day1.prepare_transport` so a later
    ``inspect()`` / ``refresh()`` / result-wrap can re-derive identification
    and re-bind data without holding a native execution.
    """

    identified: Any
    catalog: EvidenceCatalog | None
    bound: ExactTransportData | StatisticalTransportData | TrialAipwData | None
    shape: str | None
    worlds: list[Mapping[str, float]]
    provider: EmpiricalTable | LearnedCategorical | TrialAipw | str | None
    graph: Admg | None


@dataclass(frozen=True, slots=True)
class SelectionDiagram:
    """Variables whose mechanisms may differ between one source and target."""

    source: str
    target: str
    selections: Sequence[str]

    def __post_init__(self) -> None:
        if not self.source.strip() or not self.target.strip():
            raise CausalValueError("source and target must be non-empty population names")
        if self.source == self.target:
            raise CausalValueError("source and target populations must be distinct")
        if len(set(self.selections)) != len(self.selections):
            raise CausalValueError("selections must not contain duplicates")
        if any(not value.strip() for value in self.selections):
            raise CausalValueError("selections must contain variable names")


@dataclass(frozen=True, slots=True)
class ZTransportQuery:
    """Query for the bounded single-source discrete z-transport route.

    The declared controllable variables bound the possible source experiments.
    A concrete assignment selects values for a positive formula; an empty
    assignment is valid for a catalog-aware negative theorem decision, which
    checks the complete experimental family instead.
    """

    diagram: SelectionDiagram
    outcomes: Sequence[str]
    treatments: Sequence[str]
    controllable: Sequence[str]
    experiment_assignment: Mapping[str, float]

    def __post_init__(self) -> None:
        for field_name in ("outcomes", "treatments", "controllable"):
            values = tuple(getattr(self, field_name))
            if not values or len(set(values)) != len(values) or any(not v.strip() for v in values):
                raise CausalValueError(
                    f"zTR {field_name} must be non-empty, distinct variable names"
                )
            coerce_fields(self, **{field_name: values})
        if not set(self.experiment_assignment).issubset(self.controllable):
            raise CausalValueError("zTR experiment assignments must name controllable variables")
        if any(not math.isfinite(value) for value in self.experiment_assignment.values()):
            raise CausalValueError("zTR experiment assignments must be finite")
        coerce_fields(
            self, experiment_assignment=MappingProxyType(dict(self.experiment_assignment))
        )


class ZTransportSensitivityResult(TypedDict):
    """Exact assumption range from discrete outcome-kernel contamination."""

    status: str
    estimand: str
    baseline: float
    assumption_range: dict[str, float]
    delta_domain: list[float]
    decision_threshold: float | None
    tipping_fraction: float | None
    minimizing_outcome_by_stratum: list[int]
    maximizing_outcome_by_stratum: list[int]
    interval_interpretation: str
    method: str
    baseline_binding: dict[str, Any]


EvidenceKindName = Literal["available", "manipulable", "proposed"]
RegimeKindName = Literal["observational", "experimental"]
DistributionAvailabilityName = Literal["joint", "separate_marginals"]
TargetSamplingName = Literal[
    "supplied_population_law",
    "representative_sample",
    "licensed_weighted_design",
    "convenience_sample",
]
DependenceGroupName = Literal["independent_studies", "linked_units", "unknown_dependence"]
SamplingDesignName = Literal["independent", "clustered", "unknown"]
VariableDomainName = Literal["unspecified", "continuous", "binary", "count", "categorical"]


@dataclass(frozen=True, slots=True)
class VariableCoordinate:
    """Shared variable coordinate used by source and target environments."""

    name: str
    domain: VariableDomainName = "unspecified"
    unit: str | None = None
    cardinality: int | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise CausalValueError("variable coordinate name must be non-empty")
        if self.domain not in get_args(VariableDomainName):
            raise CausalValueError(f"unknown variable domain {self.domain!r}")
        if self.domain != "categorical" and self.cardinality is not None:
            raise CausalValueError("cardinality only applies to categorical domains")
        if self.domain == "categorical" and (self.cardinality is None or self.cardinality < 1):
            raise CausalValueError("categorical domains require a positive cardinality")


@dataclass(frozen=True, slots=True)
class Environment:
    """One population in a transport problem."""

    identity: str
    variables: Sequence[VariableCoordinate] = ()
    selection_targets: Sequence[str] = ()

    def __post_init__(self) -> None:
        if not self.identity.strip():
            raise CausalValueError("environment identity must be non-empty")
        names = [coordinate.name for coordinate in self.variables]
        if len(set(names)) != len(names):
            raise CausalValueError("environment variable coordinates must be unique")
        if len(set(self.selection_targets)) != len(self.selection_targets):
            raise CausalValueError("selection targets must not contain duplicates")


@dataclass(frozen=True, slots=True)
class EvidenceRegime:
    """One observational or experimental evidence regime.

    ``do(A)`` and ``do(B)`` never imply ``do(A,B)``. Separate marginals never
    imply a joint measurement. Only ``available`` evidence can satisfy a factor.
    ``study`` names the study that produced the regime (``None``: only the
    population is known); independence between regimes is never assumed from
    their populations alone. A law collected in a sample selected on
    ``selected_on``, or produced by the model ``model_artifact``, never stands in
    for a population law.
    """

    id: str
    population: str
    kind: RegimeKindName = "observational"
    evidence_kind: EvidenceKindName = "available"
    interventions: Sequence[str] = ()
    measured: Sequence[str] = ()
    distribution: DistributionAvailabilityName = "joint"
    intervention_values: Mapping[str, float] = field(default_factory=dict)
    conditioned_on: Sequence[str] = ()
    study: str | None = None
    selected_on: Sequence[str] = ()
    model_artifact: str | None = None

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.population.strip():
            raise CausalValueError("regime id and population must be non-empty")
        if self.study is not None and not self.study.strip():
            raise CausalValueError("regime study must be non-empty when given")
        if self.model_artifact is not None and not self.model_artifact.strip():
            raise CausalValueError("regime model_artifact must be non-empty when given")
        if len(set(self.selected_on)) != len(self.selected_on):
            raise CausalValueError("regime selected_on variables must be unique")
        if self.kind not in get_args(RegimeKindName):
            raise CausalValueError(f"unknown regime kind {self.kind!r}")
        if self.evidence_kind not in get_args(EvidenceKindName):
            raise CausalValueError(f"unknown evidence kind {self.evidence_kind!r}")
        if self.distribution not in get_args(DistributionAvailabilityName):
            raise CausalValueError(f"unknown distribution availability {self.distribution!r}")
        if len(set(self.interventions)) != len(self.interventions):
            raise CausalValueError("regime interventions must be unique")
        if len(set(self.conditioned_on)) != len(self.conditioned_on) or any(
            v not in self.measured for v in self.conditioned_on
        ):
            raise CausalValueError("conditioning coordinates must be distinct measured variables")
        if self.conditioned_on and self.distribution != "joint":
            raise CausalValueError("conditioning requires a joint law")
        if len(set(self.measured)) != len(self.measured):
            raise CausalValueError("regime measured variables must be unique")
        if any(
            v not in self.interventions or not math.isfinite(x)
            for v, x in self.intervention_values.items()
        ):
            raise CausalValueError(
                "intervention values must be finite and name intervened variables"
            )
        if self.kind == "observational" and self.interventions:
            raise CausalValueError("observational regimes cannot carry hard interventions")
        if self.kind == "experimental" and not self.interventions:
            raise CausalValueError("experimental regimes require a non-empty intervention set")

    def available_experiment_on(self, population: str, variables: Sequence[str]) -> bool:
        return (
            self.evidence_kind == "available"
            and not self.selected_on
            and self.model_artifact is None
            and self.kind == "experimental"
            and self.population == population
            and set(self.interventions) == set(variables)
        )


@dataclass(frozen=True, slots=True)
class RegimeBinding:
    """Bind a dataset snapshot to one regime."""

    regime: str
    snapshot_identity: str
    schema_names: Sequence[str] = ()
    sampling: SamplingDesignName = "unknown"
    dependence: DependenceGroupName = "unknown_dependence"
    weights_snapshot: str | None = None
    dataset_identity: str | None = None

    def __post_init__(self) -> None:
        if not self.regime.strip() or not self.snapshot_identity.strip():
            raise CausalValueError("regime binding requires regime and snapshot identity")
        if self.dataset_identity is not None and not self.dataset_identity.strip():
            raise CausalValueError("dataset identity, when supplied, must be a non-empty name")
        if self.weights_snapshot is not None and self.weights_snapshot != self.snapshot_identity:
            raise CausalValueError("weights must be licensed for the bound snapshot")
        if self.sampling not in get_args(SamplingDesignName):
            raise CausalValueError(f"unknown sampling design {self.sampling!r}")
        if self.dependence not in get_args(DependenceGroupName):
            raise CausalValueError(f"unknown dependence group {self.dependence!r}")


@dataclass(frozen=True, slots=True)
class EvidenceCatalog:
    """Supplied evidence for one transport query. Multiple sources are first-class."""

    environments: Sequence[Environment] = ()
    regimes: Sequence[EvidenceRegime] = ()
    bindings: Sequence[RegimeBinding] = ()
    target_sampling: TargetSamplingName | None = None

    def __post_init__(self) -> None:
        identities = [environment.identity for environment in self.environments]
        if len(set(identities)) != len(identities):
            raise CausalValueError("catalog environments must have distinct identities")
        domains: dict[str, tuple[str, str | None, int | None]] = {}
        for environment in self.environments:
            for coordinate in environment.variables:
                existing = domains.get(coordinate.name)
                if existing is None:
                    domains[coordinate.name] = (
                        coordinate.domain,
                        coordinate.unit,
                        coordinate.cardinality,
                    )
                    continue
                existing_domain, existing_unit, existing_cardinality = existing
                if (
                    existing_domain != "unspecified"
                    and coordinate.domain != "unspecified"
                    and (
                        existing_domain != coordinate.domain
                        or (
                            existing_domain == "categorical"
                            and existing_cardinality != coordinate.cardinality
                        )
                    )
                ):
                    raise CausalValueError(
                        "catalog environments declare incompatible domains for the same variable"
                    )
                if (
                    existing_unit is not None
                    and coordinate.unit is not None
                    and existing_unit != coordinate.unit
                ):
                    raise CausalValueError(
                        "catalog environments declare incompatible units for the same variable"
                    )
                domains[coordinate.name] = (
                    coordinate.domain if existing_domain == "unspecified" else existing_domain,
                    existing_unit if existing_unit is not None else coordinate.unit,
                    coordinate.cardinality
                    if existing_domain == "unspecified"
                    else existing_cardinality,
                )
        regime_ids = [regime.id for regime in self.regimes]
        if len(set(regime_ids)) != len(regime_ids):
            raise CausalValueError("catalog regime ids must be unique")
        known = set(regime_ids)
        for binding in self.bindings:
            if binding.regime not in known:
                raise CausalValueError("regime binding names an unknown regime")
        if self.target_sampling is not None and self.target_sampling not in get_args(
            TargetSamplingName
        ):
            raise CausalValueError(f"unknown target sampling {self.target_sampling!r}")

    def source_experiment_variables(self, population: str) -> tuple[str, ...]:
        variables: list[str] = []
        for regime in self.regimes:
            if (
                regime.evidence_kind == "available"
                and regime.kind == "experimental"
                and regime.population == population
            ):
                variables.extend(regime.interventions)
        return tuple(sorted(set(variables)))

    def has_available_experiment(self, population: str, variables: Sequence[str]) -> bool:
        return any(regime.available_experiment_on(population, variables) for regime in self.regimes)

    @staticmethod
    def empty() -> EvidenceCatalog:
        return EvidenceCatalog()


@dataclass(frozen=True, slots=True)
class EvidenceCatalogDelta:
    """Immutable proposed laws for a transport-planning preview.

    The original catalog retains its evidence status. The returned preview is
    a temporary input for identification and binding; it does not contain data.
    """

    proposed_regimes: Sequence[EvidenceRegime]

    def __post_init__(self) -> None:
        ids = [regime.id for regime in self.proposed_regimes]
        if len(set(ids)) != len(ids):
            raise CausalValueError("catalog delta regime ids must be unique")
        if any(regime.evidence_kind != "proposed" for regime in self.proposed_regimes):
            raise CausalValueError("catalog delta requires proposed regimes")

    def preview(self, base: EvidenceCatalog) -> EvidenceCatalog:
        """Return a separate catalog with hypothetical results available."""
        if set(regime.id for regime in base.regimes) & set(
            regime.id for regime in self.proposed_regimes
        ):
            raise CausalValueError("catalog delta conflicts with a base regime")
        return EvidenceCatalog(
            environments=base.environments,
            regimes=(
                *base.regimes,
                *(replace(r, evidence_kind="available") for r in self.proposed_regimes),
            ),
            bindings=base.bindings,
            target_sampling=base.target_sampling,
        )


@dataclass(frozen=True, slots=True)
class ZTransportCandidate:
    """One proposed zTR study and its hypothetical catalog additions.

    ``catalog`` is the full hypothetical catalog, retaining the base entries
    and adding proposed regimes for this candidate. A verified proposal is
    returned only when its declared design can produce a checked,
    bound formula against the frozen failure snapshot.
    """

    id: str
    catalog: EvidenceCatalog
    design_kind: Literal["intervene", "measure"]
    targets: Sequence[str] = ()
    measured: Sequence[str] = ()
    cost: float = 0.0
    sample_budget: int = 0
    recruitment_sampling: str = "independent"
    feasibility_constraints: Sequence[str] = ()
    tag: int = 0

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise CausalValueError("candidate id must be non-empty")
        if self.design_kind not in ("intervene", "measure"):
            raise CausalValueError("design_kind must be 'intervene' or 'measure'")
        if not math.isfinite(self.cost) or self.cost < 0:
            raise CausalValueError("candidate cost must be a finite non-negative number")
        _non_negative("sample_budget", self.sample_budget)
        _non_negative("tag", self.tag)
        if not any(regime.evidence_kind == "proposed" for regime in self.catalog.regimes):
            raise CausalValueError("candidate catalog must add proposed regimes")
        coerce_fields(self, targets=tuple(self.targets))
        coerce_fields(self, measured=tuple(self.measured))
        coerce_fields(self, feasibility_constraints=tuple(self.feasibility_constraints))
