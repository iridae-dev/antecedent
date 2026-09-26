"""Single-source graphical transportability specifications.

This namespace describes structural population differences. It is distinct
from statistical prior/evidence transport in :mod:`antecedent.priors`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, TypedDict, get_args, overload

if TYPE_CHECKING:
    from ..estimation import PreparedAnalysis

import numpy as np

from .._defaults import OMITTED
from .._native import consume_z_transport_artifact as _consume_z_transport_artifact
from .._native import (
    consume_z_transport_sensitivity_artifact as _consume_z_transport_sensitivity_artifact,
)
from .._native import estimate_trial_transport as _estimate_trial_transport
from .._native import identify_transport as _identify_transport
from .._native import identify_z_transport_stage as _identify_z_transport_stage
from .._native import replay_z_transport_proposal as _replay_z_transport_proposal
from .._native import roundtrip_expr_arena as _roundtrip_expr_arena
from .._transport_results import (
    TransportContrast,
    TransportGridPoint,
    TransportUncertainty,
)
from ..errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from ..graph import Admg
from ..learners import LearnerSpec, Logistic, Ridge, _learner_wire
from ..query import (
    AverageDerivative,
    DirectionalDerivative,
    Elasticity,
    PointDerivative,
    ResponseCurve,
    ResponseJacobian,
    SemiElasticity,
)


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
            object.__setattr__(self, field_name, values)
        if not set(self.experiment_assignment).issubset(self.controllable):
            raise CausalValueError("zTR experiment assignments must name controllable variables")
        if any(not math.isfinite(value) for value in self.experiment_assignment.values()):
            raise CausalValueError("zTR experiment assignments must be finite")
        object.__setattr__(
            self, "experiment_assignment", MappingProxyType(dict(self.experiment_assignment))
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

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.population.strip():
            raise CausalValueError("regime id and population must be non-empty")
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
        object.__setattr__(self, "targets", tuple(self.targets))
        object.__setattr__(self, "measured", tuple(self.measured))
        object.__setattr__(self, "feasibility_constraints", tuple(self.feasibility_constraints))


@dataclass(frozen=True, slots=True)
class TransportQuery:
    """Transport a response query under a single-source selection diagram.

    ``query`` is the response requested in the target population, ``diagram``
    the selection diagram's populations and mechanism-selection targets, and
    ``source_experiments`` the variables randomized in the source.

    :func:`antecedent.analyze` estimates the licensed cell (explicit ``Admg``,
    Frequentist, validation ``none``): a mean :class:`antecedent.ResponseCurve`
    transported from trial to target by binary trial-to-target IPW. It reads
    three data columns named by the keyword-only fields: ``trial`` (source-trial
    membership, nonzero for trial rows), ``selection_probability``
    (``P(S=1 | X)`` on every row) and ``treatment_probability``
    (``P(A=1 | X, S=1)`` on trial rows). The selection diagram and these column
    bindings freeze at prepare; a refresh reads the columns from the new data.
    """

    query: object
    diagram: SelectionDiagram
    source_experiments: Sequence[str] = ()
    _: KW_ONLY
    catalog: EvidenceCatalog | None = None
    trial: str | None = None
    selection_probability: str | None = None
    treatment_probability: str | None = None
    kind: Literal["transport"] = field(default="transport", init=False, repr=False)

    def __post_init__(self) -> None:
        if self.query is None:
            raise CausalValueError("query must not be None")
        if len(set(self.source_experiments)) != len(self.source_experiments):
            raise CausalValueError("source_experiments must not contain duplicates")
        if any(not value.strip() for value in self.source_experiments):
            raise CausalValueError("source_experiments must contain variable names")
        if self.catalog is not None:
            derived = list(self.catalog.source_experiment_variables(self.diagram.source))
            explicit = list(self.source_experiments)
            if explicit and set(explicit) != set(derived):
                raise CausalValueError("source_experiments disagree with the evidence catalog")
            if not explicit:
                object.__setattr__(self, "source_experiments", tuple(derived))
        columns = (self.trial, self.selection_probability, self.treatment_probability)
        if any(column is not None for column in columns):
            if any(column is None for column in columns):
                raise CausalValueError(
                    "trial, selection_probability and treatment_probability name the trial "
                    "columns together; supply all three"
                )
            if any(not str(column).strip() for column in columns):
                raise CausalValueError("trial columns must be non-empty variable names")
            if len(set(columns)) != len(columns):
                raise CausalValueError("trial columns must be distinct")

    @property
    def trial_columns(self) -> tuple[str, str, str] | None:
        """``(trial, selection_probability, treatment_probability)`` when bound."""
        if self.trial is None or self.selection_probability is None:
            return None
        if self.treatment_probability is None:
            return None
        return (self.trial, self.selection_probability, self.treatment_probability)


@dataclass(frozen=True, slots=True)
class PopulationFactor:
    """One population-labelled distribution factor in a transport formula."""

    population: str
    variables: Sequence[str]
    conditioned_on: Sequence[str]
    interventions: Sequence[str]
    regime: int | None = None


@dataclass(frozen=True, slots=True)
class DirectFormula:
    factor: PopulationFactor


@dataclass(frozen=True, slots=True)
class StandardizationFormula:
    over: Sequence[str]
    source_response: PopulationFactor
    target_law: PopulationFactor


@dataclass(frozen=True, slots=True)
class RecursiveFactorizationFormula:
    sum_out: Sequence[str]
    factors: Sequence[PopulationFactor]


TransportFormula = DirectFormula | StandardizationFormula | RecursiveFactorizationFormula


@dataclass(frozen=True, slots=True)
class TransportCertificate:
    rule: str
    selection_targets: Sequence[str]


@dataclass(frozen=True, slots=True)
class NotCertifiedCertificate:
    """Conservative refusal; this is not a general non-transportability claim."""

    reason: str
    witness: Sequence[str]
    message: str


# Historical name denotes an inconclusive result, never an impossibility proof.
NonTransportableCertificate = NotCertifiedCertificate


@dataclass(frozen=True, slots=True)
class MissingEvidenceCertificate:
    """Required available evidence was absent. Distinct from :class:`NotCertifiedCertificate`."""

    reason: str
    missing: Sequence[str]
    message: str


@dataclass(frozen=True, slots=True)
class TransportIdentification:
    formula: TransportFormula | None
    certificate: TransportCertificate | NotCertifiedCertificate | MissingEvidenceCertificate
    outcome: str = "not_certified"
    provenance: Mapping[str, Any] = field(
        default_factory=lambda: {"operation_ids": ["identify.transport_sid"]}
    )
    pretty: str | None = None
    latex: str | None = None
    leaf_bindings: tuple[tuple[str, int | None], ...] = ()
    expr_root: int | None = None
    expr_wire_json: str | None = None
    # The native `TransportIdentificationResult` this was built from. `estimate_trial_effect`
    # forwards it to the native estimator so the refusal gate keys off the certificate that was
    # actually produced by `identify(...)`, not off this frozen dataclass's own (re-derivable
    # but caller-editable) `transportable` property. Opaque outside this module.
    _native: Any = field(default=None, repr=False, compare=False)

    @property
    def transportable(self) -> bool:
        return self.formula is not None


@dataclass(frozen=True, slots=True)
class OverlapDiagnostic:
    probability_min: float
    probability_max: float
    effective_sample_size: float
    extreme_weight_count: int


@dataclass(frozen=True, slots=True)
class TransportOverlapReport:
    selection: OverlapDiagnostic
    treatment: OverlapDiagnostic


@dataclass(frozen=True, slots=True)
class TrialTransportEstimate:
    # Rule id of the transport certificate that authorized this estimate, so the result
    # carries its own identification provenance rather than relying on the caller to
    # remember which `TransportIdentification` it passed to `estimate_trial_effect`.
    rule: str
    ipw: float
    aipw: float | None
    overlap: TransportOverlapReport
    provenance: Mapping[str, Any] = field(
        default_factory=lambda: {"operation_ids": ["estimate.trial_to_target"]}
    )


def _response_args(query: object) -> dict[str, Any]:
    supported = (
        ResponseCurve,
        AverageDerivative,
        PointDerivative,
        Elasticity,
        SemiElasticity,
        DirectionalDerivative,
        ResponseJacobian,
    )
    if not isinstance(query, supported):
        raise CausalTypeError("TransportQuery.query must be a response-family query")
    if getattr(query, "observation", None) is not None:
        raise CausalValueError("transport identification currently requires complete observation")
    if getattr(query, "observation_assumptions", ()):
        raise CausalValueError(
            "transport identification does not yet compose observation_assumptions"
        )
    if getattr(query, "target_population", None) is not None:
        raise CausalValueError(
            "TransportQuery owns its source/target populations; its embedded response "
            "must not set target_population"
        )
    at: list[float] | None
    if isinstance(query, (DirectionalDerivative, ResponseJacobian)):
        treatments = list(query.treatments)
        outcomes = list(query.outcomes)
        at = (
            [query.at[name] for name in treatments]
            if isinstance(query.at, Mapping)
            else list(query.at)
        )
    else:
        treatments = [query.treatment]
        outcomes = [query.outcome]
        at = [query.at] if hasattr(query, "at") else None
    direction = None
    if isinstance(query, DirectionalDerivative):
        direction = (
            [query.direction[name] for name in treatments]
            if isinstance(query.direction, Mapping)
            else list(query.direction)
        )
    scale = "identity"
    if isinstance(query, Elasticity):
        scale = "log_log"
    elif isinstance(query, SemiElasticity):
        scale = "log_treatment" if query.log_scale == "treatment" else "log_outcome"
    weighting = getattr(query, "weighting", None) or "observed"
    if not isinstance(weighting, str):
        raise CausalTypeError("AverageDerivative.weighting currently accepts 'observed'")
    return {
        "kind": query.kind,
        "treatments": treatments,
        "outcomes": outcomes,
        "grid": list(query.grid) if isinstance(query, ResponseCurve) else None,
        "at": at,
        "direction": direction,
        "order": getattr(query, "order", 1),
        "scale": scale,
        "weighting": weighting,
    }


def identify(*, graph: Admg, query: TransportQuery) -> TransportIdentification:
    """Identify a single-source graphical transport formula and its certificate."""

    if not isinstance(graph, Admg):
        raise CausalTypeError("transport.identify requires graph=Admg(...)")
    if not isinstance(query, TransportQuery):
        raise CausalTypeError("query must be a TransportQuery")
    raw = _identify_transport(
        graph,
        list(query.diagram.selections),
        query.diagram.source,
        query.diagram.target,
        list(query.source_experiments),
        catalog=query.catalog,
        **_response_args(query.query),
    )
    if not raw.transportable:
        if raw.reason is None or raw.message is None:
            raise RuntimeError("native transport refusal omitted its certificate")
        if raw.outcome == "missing_evidence":
            return TransportIdentification(
                None,
                MissingEvidenceCertificate(raw.reason, raw.selection_targets, raw.message),
                outcome=raw.outcome,
                _native=raw,
            )
        return TransportIdentification(
            None,
            NotCertifiedCertificate(raw.reason, raw.selection_targets, raw.message),
            outcome=raw.outcome,
            _native=raw,
        )
    factors = [
        PopulationFactor(population, variables, conditioned_on, interventions, regime)
        for population, variables, conditioned_on, interventions, regime in zip(
            raw.factor_populations,
            raw.factor_variables,
            raw.factor_conditioned_on,
            raw.factor_interventions,
            raw.factor_regimes,
            strict=True,
        )
    ]
    if raw.formula_kind == "direct":
        formula: TransportFormula = DirectFormula(factors[0])
    elif raw.formula_kind == "standardize":
        formula = StandardizationFormula(raw.marginalize, factors[0], factors[1])
    elif raw.formula_kind == "recursive_factorization":
        formula = RecursiveFactorizationFormula(raw.marginalize, factors)
    else:  # native result invariant
        raise RuntimeError(f"unknown native transport formula {raw.formula_kind!r}")
    if raw.rule is None:
        raise RuntimeError("native transport formula omitted its certificate rule")
    return TransportIdentification(
        formula,
        TransportCertificate(raw.rule, raw.selection_targets),
        outcome=raw.outcome,
        pretty=raw.expr_pretty,
        latex=raw.expr_latex,
        leaf_bindings=tuple(zip(raw.leaf_populations, raw.leaf_regimes, strict=True)),
        expr_root=raw.expr_root,
        expr_wire_json=raw.expr_wire_json,
        _native=raw,
    )


def _non_negative(name: str, value: int) -> int:
    """A non-negative int, refused as a typed value error before it reaches native code."""
    if value is None:
        raise CausalValueError(f"{name} must be a non-negative integer")
    if isinstance(value, bool) or not isinstance(value, int):
        raise CausalTypeError(f"{name} must be an integer")
    if value < 0:
        raise CausalValueError(f"{name} must be a non-negative integer, got {value}")
    return value


def _optional_non_negative(name: str, value: int | None) -> int | None:
    """:func:`_non_negative`, with ``None`` meaning "no limit"."""
    return None if value is None else _non_negative(name, value)


def identify_z_transport(
    *,
    graph: Admg,
    query: ZTransportQuery,
    max_steps: int = 100_000,
    max_depth: int = 256,
    max_support_rows: int = 1_000_000,
    memory_bytes: int | None = None,
    cancel: Any = None,
) -> Any:
    """Create a native stage for bounded single-source z-transport search.

    The returned stage exposes ``outcome`` and ``reason``. Positive formulas
    are checked before binding evidence; a negative search is not treated as a
    theorem obstruction unless the complete experimental-family premises hold.
    Call ``stage.decide(catalog)`` to check those premises and obtain a
    replayable bounded obstruction or a precise missing-evidence result.
    On an identified result, call ``prepare_exact`` or ``prepare_empirical``
    to obtain a point-only execution. The limits are retained by the stage and
    inherited by every decision, preparation and proposal made from it.
    """
    if not isinstance(graph, Admg):
        raise CausalTypeError("transport.identify_z_transport requires graph=Admg(...)")
    if not isinstance(query, ZTransportQuery):
        raise CausalTypeError("query must be a ZTransportQuery")
    return _identify_z_transport_stage(
        graph,
        list(query.diagram.selections),
        query.diagram.source,
        query.diagram.target,
        list(query.outcomes),
        list(query.treatments),
        list(query.controllable),
        dict(query.experiment_assignment),
        max_steps=_non_negative("max_steps", max_steps),
        max_depth=_non_negative("max_depth", max_depth),
        max_support_rows=_non_negative("max_support_rows", max_support_rows),
        memory_bytes=_optional_non_negative("memory_bytes", memory_bytes),
        cancel=cancel,
    )


def consume_z_transport_artifact(
    artifact: bytes, *, memory_bytes: int | None = None, cancel: Any = None
) -> str:
    """Independently verify and recompute an exported point-only zTR result."""
    if not isinstance(artifact, bytes):
        raise CausalTypeError("artifact must be bytes")
    return _consume_z_transport_artifact(
        artifact,
        memory_bytes=_optional_non_negative("memory_bytes", memory_bytes),
        cancel=cancel,
    )


def plan_z_transport_evidence(
    stage: Any,
    catalog: EvidenceCatalog,
    candidates: Sequence[ZTransportCandidate],
    *,
    failure_snapshot: bytes | None = None,
) -> tuple[dict[str, Any], tuple[Any, ...]]:
    """Assess candidates against a stage's frozen failure and catalog.

    Passing ``failure_snapshot`` reuses an exported snapshot and verifies that
    it exactly matches this stage and catalog before planning.
    """
    import json

    if not hasattr(stage, "plan_evidence"):
        raise CausalTypeError("stage must be returned by identify_z_transport")
    if failure_snapshot is not None and not isinstance(failure_snapshot, bytes):
        raise CausalTypeError("failure_snapshot must be bytes")
    report, proposals = stage.plan_evidence(catalog, list(candidates), failure_snapshot)
    return json.loads(report), tuple(proposals)


def consume_z_transport_sensitivity_artifact(
    artifact: bytes, *, memory_bytes: int | None = None, cancel: Any = None
) -> ZTransportSensitivityResult:
    """Independently verify and recompute a portable zTR sensitivity range."""
    if not isinstance(artifact, bytes):
        raise CausalTypeError("artifact must be bytes")
    return _consume_z_transport_sensitivity_artifact(
        artifact,
        memory_bytes=_optional_non_negative("memory_bytes", memory_bytes),
        cancel=cancel,
    )


def replay_z_transport_proposal(artifact: bytes) -> None:
    """Independently replay a portable hypothetical zTR study proposal."""
    if not isinstance(artifact, bytes):
        raise CausalTypeError("proposal artifact must be bytes")
    _replay_z_transport_proposal(artifact)


def reload_lowered_expression(
    identification: TransportIdentification,
) -> tuple[str, str, tuple[tuple[str, int | None], ...], tuple[int, ...]]:
    """Reload a lowered arena from its artifact JSON and return display/bindings."""

    if identification.expr_wire_json is None or identification.expr_root is None:
        raise CausalValueError("identification has no lowered expression to reload")
    pretty, latex, populations, regimes, free = _roundtrip_expr_arena(
        identification.expr_wire_json,
        identification.expr_root,
    )
    return pretty, latex, tuple(zip(populations, regimes, strict=True)), tuple(free)


def reload_lowered_program(identification: TransportIdentification) -> Any:
    """Return a bounded, checked executable program for an identified expression.

    The returned native program owns the validated arena and semantic schema.
    Exact evaluation requires an explicit catalog and provider laws; this
    function does not infer or fetch them.
    """
    if not isinstance(identification, TransportIdentification):
        raise CausalTypeError("reload_lowered_program requires a TransportIdentification")
    if identification._native is None or not identification.transportable:
        raise CausalValueError("identification has no checked expression to reload")
    return identification._native.reload_checked_program()


def restore_lowered_program(identification: TransportIdentification, wire_json: str) -> Any:
    """Independently import and verify a checked program against its identification."""
    if not isinstance(identification, TransportIdentification):
        raise CausalTypeError("restore_lowered_program requires a TransportIdentification")
    if identification._native is None or not identification.transportable:
        raise CausalValueError("identification has no checked expression to reload")
    return identification._native.load_checked_program(wire_json)


def estimate_trial_effect(
    identification: TransportIdentification,
    treatment: Sequence[bool],
    outcome: Sequence[float],
    trial: Sequence[bool],
    selection_probability: Sequence[float],
    treatment_probability: Sequence[float],
    *,
    mu0: Sequence[float] | None = None,
    mu1: Sequence[float] | None = None,
) -> TrialTransportEstimate:
    """Estimate a trial-to-target binary-treatment contrast by IPW and optional AIPW.

    This is an unlicensed utility: it calls the
    trial-to-target estimator directly and returns bare numbers, with no study,
    contract, export or calibration slot. ``antecedent.analyze(data, graph=Admg,
    query=TransportQuery(..., trial=, selection_probability=,
    treatment_probability=))`` is the licensed, study-retaining path.

    ``identification`` must be the result of :func:`identify` for this query.
    Identification and estimation stay separate operations, but the estimator
    now requires the certificate: when ``identification.transportable`` is
    ``False`` (a :class:`NotCertifiedCertificate`, i.e. ``identify``
    returned ``NotCertified``), this raises ``CausalEstimateError`` instead of
    returning a number for a quantity that was never shown to be identified.
    A refusal is a conservative "no rule applies", not a proof of
    non-transportability — see :class:`NotCertifiedCertificate`.

    The positional order is ``(treatment, outcome)``, matching Antecedent's
    scalar causal-query convention.
    """

    if not isinstance(identification, TransportIdentification):
        raise CausalTypeError(
            "estimate_trial_effect requires identification=transport.identify(...)"
        )
    if (mu0 is None) != (mu1 is None):
        raise CausalValueError("mu0 and mu1 must be supplied together")
    raw = _estimate_trial_transport(
        identification._native,
        np.asarray(outcome, dtype=np.float64),
        list(treatment),
        list(trial),
        np.asarray(selection_probability, dtype=np.float64),
        np.asarray(treatment_probability, dtype=np.float64),
        mu0=None if mu0 is None else np.asarray(mu0, dtype=np.float64),
        mu1=None if mu1 is None else np.asarray(mu1, dtype=np.float64),
    )
    return TrialTransportEstimate(
        raw.rule,
        raw.ipw,
        raw.aipw,
        TransportOverlapReport(
            OverlapDiagnostic(
                raw.selection_probability_min,
                raw.selection_probability_max,
                raw.selection_effective_sample_size,
                raw.selection_extreme_weight_count,
            ),
            OverlapDiagnostic(
                raw.treatment_probability_min,
                raw.treatment_probability_max,
                raw.treatment_effective_sample_size,
                raw.treatment_extreme_weight_count,
            ),
        ),
    )


__all__ = [
    "DependenceGroupName",
    "DirectFormula",
    "DistributionAvailabilityName",
    "Environment",
    "EvidenceCatalog",
    "EvidenceCatalogDelta",
    "EvidenceKindName",
    "EvidenceRegime",
    "MissingEvidenceCertificate",
    "NotCertifiedCertificate",
    "NonTransportableCertificate",
    "OverlapDiagnostic",
    "PopulationFactor",
    "RecursiveFactorizationFormula",
    "reload_lowered_expression",
    "reload_lowered_program",
    "restore_lowered_program",
    "RegimeBinding",
    "RegimeKindName",
    "SamplingDesignName",
    "SelectionDiagram",
    "StandardizationFormula",
    "TargetSamplingName",
    "TransportCertificate",
    "TransportIdentification",
    "TransportOverlapReport",
    "TransportQuery",
    "ZTransportQuery",
    "ZTransportCandidate",
    "ZTransportSensitivityResult",
    "TrialTransportEstimate",
    "VariableCoordinate",
    "VariableDomainName",
    "estimate_trial_effect",
    "identify",
    "identify_z_transport",
    "consume_z_transport_artifact",
    "plan_z_transport_evidence",
    "consume_z_transport_sensitivity_artifact",
    "replay_z_transport_proposal",
]


@dataclass(frozen=True, slots=True)
class ExactDiscreteLaw:
    """Dense exact law for one concrete intervention world, last axis fastest.

    A law is supplied as exact input; this does not assert that estimates from
    finite samples are error-free. Native validation occurs before evaluation.
    """

    population: str
    regime: str
    axes: tuple[tuple[str, tuple[float, ...]], ...]
    probabilities: tuple[float, ...]
    snapshot_identity: str
    interventions: tuple[tuple[str, float], ...] = ()
    absolute_tolerance: float = 1e-12
    relative_tolerance: float = 1e-10
    empirical_counts: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        # Freeze caller-owned sequences before retaining them as a snapshot.
        object.__setattr__(self, "axes", tuple((name, tuple(values)) for name, values in self.axes))
        object.__setattr__(self, "probabilities", tuple(self.probabilities))
        object.__setattr__(self, "interventions", tuple(tuple(item) for item in self.interventions))
        if self.empirical_counts is not None:
            counts = tuple(self.empirical_counts)
            if len(counts) != len(self.probabilities):
                raise CausalValueError("empirical_counts must have one count per probability")
            for count in counts:
                _non_negative("empirical_counts", count)
            object.__setattr__(self, "empirical_counts", counts)


@dataclass(frozen=True, slots=True)
class ExactTransportData:
    """Immutable collection of complete exact laws, separate from sampled data."""

    laws: tuple[ExactDiscreteLaw, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "laws", tuple(self.laws))


@dataclass(frozen=True, slots=True)
class ExactTransportDistribution:
    """Complete target distribution with no sampling uncertainty claim."""

    outcomes: tuple[str, ...]
    atoms: tuple[tuple[float, ...], ...]
    probabilities: tuple[float, ...]
    formula: str
    rules: tuple[str, ...]
    uncertainty: None = field(default=None, init=False)
    _execution: Any = field(default=None, repr=False, compare=False)

    def __repr__(self) -> str:
        state = "checked" if self._execution is not None else "unavailable"
        return (
            f"ExactTransportDistribution(outcomes={self.outcomes!r}, probabilities={self.probabilities!r}, "
            f"identification={state}, support={state}, uncertainty=not_applicable_exact_law, assumptions=declared)"
        )

    def inspect(self) -> Any:
        """Native four-slot reasoning and factor-level support for this execution."""
        import json

        from ..results._report import InspectionReport

        if self._execution is None:
            raise _no_native_authority()
        return InspectionReport(**json.loads(self._execution.inspection_json()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcomes": self.outcomes,
            "atoms": self.atoms,
            "probabilities": self.probabilities,
            "formula": self.formula,
            "rules": self.rules,
            "reasoning": self.inspect().to_dict(),
        }

    def export(self) -> bytes:
        """Export the immutable native execution, independent of edited display fields."""
        if self._execution is None:
            raise _no_native_authority()
        return bytes(self._execution.export())

    def contrast(self, reference: ExactTransportDistribution, outcome: str) -> float:
        """Difference of means from two complete target laws; no sampling interval."""
        if self.outcomes != reference.outcomes:
            raise CausalValueError("Contrast outcome coordinates must agree")
        return self.mean(outcome) - reference.mean(outcome)

    def mean(self, outcome: str) -> float:
        """Derive a numeric outcome mean from the full evaluated distribution."""
        coordinate = self.outcomes.index(outcome)
        return math.fsum(
            atom[coordinate] * p for atom, p in zip(self.atoms, self.probabilities, strict=True)
        )


@dataclass(frozen=True, slots=True)
class ClassicalTransportIdentification:
    """Theorem-stage result under all source experiments and target observation.

    Display fields cannot authorize execution. The retained native derivation
    remains authoritative; catalog binding is a separate, incomplete search.
    """

    outcome: str
    formula: str | None
    rules: tuple[str, ...]
    outcomes: tuple[str, ...]
    _native: Any = field(repr=False, compare=False)

    def export(self) -> bytes:
        """Export the retained checked proof or conservative/negative certificate."""
        return bytes(self._native.export())

    def inspect(self) -> Any:
        """Structural certificate and all four reasoning slots; no provider access."""
        import json

        from ..results._report import InspectionReport, SlotModel

        raw = dict(json.loads(self._native.certificate_json()))
        identified = self.outcome == "identified"
        return InspectionReport(
            identification=SlotModel(
                available=identified,
                reason=None if identified else self.outcome,
                summary=self.outcome,
                payload={"outcome": self.outcome, "engine": raw},
            ),
            support=SlotModel(
                available=identified,
                reason=None,
                summary="not_estimated" if identified else "unavailable",
                payload={"engine": raw},
            ),
            uncertainty=SlotModel(
                available=False,
                reason="not_estimated",
                summary="unavailable",
            ),
            assumptions=SlotModel(
                available=True,
                summary="declared",
                payload={"rules": list(self.rules), "engine": raw.get("reasoning", raw)},
            ),
        )


def consume_identification(artifact: bytes, **limits: Any) -> ClassicalTransportIdentification:
    """Independently verify a structural certificate, without rerunning identification."""
    from .._native import consume_transport_certificate

    native = consume_transport_certificate(artifact, **limits)
    return ClassicalTransportIdentification(
        native.outcome, native.formula, tuple(native.rules), tuple(native.outcomes), native
    )


def identify_meta(
    graph: Admg,
    catalog: EvidenceCatalog,
    *,
    target: str,
    outcomes: Sequence[str],
    treatments: Sequence[str],
    max_steps: int = 100_000,
    max_depth: int = 256,
    memory_bytes: int | None = None,
    cancel: Any = None,
) -> ClassicalTransportIdentification:
    """Identify with classical μsID and source-specific catalog selections.

    Completeness concerns full source experimental families; finite catalog
    binding remains a separate bounded search. No provider data are inspected.
    """
    from .._native import identify_meta_transport_stage

    native = identify_meta_transport_stage(
        graph,
        catalog,
        target,
        list(outcomes),
        list(treatments),
        max_steps=max_steps,
        max_depth=max_depth,
        memory_bytes=memory_bytes,
        cancel=cancel,
    )
    return ClassicalTransportIdentification(
        native.outcome,
        native.formula,
        tuple(native.rules),
        tuple(native.outcomes),
        native,
    )


def identify_classical(
    graph: Admg,
    selection: SelectionDiagram,
    *,
    outcomes: Sequence[str],
    treatments: Sequence[str],
    max_steps: int = 100_000,
    max_depth: int = 256,
    memory_bytes: int | None = None,
    cancel: Any = None,
) -> ClassicalTransportIdentification:
    """Identify under the classical *complete source experimental family*.

    The source experiments here are a theorem assumption, not a claim that a
    user's finite catalog supplies every experiment. Actual tables are bound
    separately by :func:`evaluate_exact`.
    """
    from .._native import identify_classical_transport_stage

    native = identify_classical_transport_stage(
        graph,
        list(selection.selections),
        selection.source,
        selection.target,
        list(outcomes),
        list(treatments),
        max_steps=max_steps,
        max_depth=max_depth,
        memory_bytes=memory_bytes,
        cancel=cancel,
    )
    return ClassicalTransportIdentification(
        native.outcome, native.formula, tuple(native.rules), tuple(native.outcomes), native
    )


def evaluate_exact(
    identification: ClassicalTransportIdentification,
    catalog: EvidenceCatalog,
    data: ExactTransportData,
    *,
    at: Mapping[str, float],
    max_operations: int = 10_000_000,
    max_depth: int = 256,
    max_support_rows: int = 1_000_000,
    memory_bytes: int | None = None,
    cancel: Any = None,
) -> ExactTransportDistribution:
    """Bind one checked derivation and evaluate its full target law.

    Missing catalog factors describe this derivation, not an impossibility
    proof. This stage does not fit models or estimate sampling uncertainty.
    """
    return prepare_exact(
        identification,
        catalog,
        data,
        at=at,
        max_operations=max_operations,
        max_depth=max_depth,
        max_support_rows=max_support_rows,
        memory_bytes=memory_bytes,
        cancel=cancel,
    ).estimate()


__all__ += [
    "ClassicalTransportIdentification",
    "ExactDiscreteLaw",
    "ExactTransportData",
    "ExactTransportDistribution",
    "identify_classical",
    "identify_meta",
    "evaluate_exact",
]


def _exact_distribution(native: Any, payload: Any) -> Any:
    atoms, probabilities, formula, rules = payload
    return ExactTransportDistribution(
        tuple(native.outcomes),
        tuple(tuple(row) for row in atoms),
        tuple(probabilities),
        formula,
        tuple(rules),
        _execution=native.freeze(),
    )


@dataclass(frozen=True, slots=True)
class RegimeSample:
    """Finite categorical sample for one regime and intervention world."""

    population: str
    regime: str
    snapshot_identity: str
    columns: Mapping[str, Sequence[float | None]]
    interventions: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        from types import MappingProxyType

        object.__setattr__(
            self,
            "columns",
            MappingProxyType(
                {
                    name: tuple(
                        None if value is None or math.isnan(value) else float(value)
                        for value in values
                    )
                    for name, values in self.columns.items()
                }
            ),
        )
        object.__setattr__(self, "interventions", tuple(tuple(item) for item in self.interventions))


@dataclass(frozen=True, slots=True)
class StatisticalTransportData:
    """Mixed supplied laws and estimated samples for one statistical prepare."""

    samples: tuple[RegimeSample, ...] = ()
    laws: tuple[ExactDiscreteLaw, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "samples", tuple(self.samples))
        object.__setattr__(self, "laws", tuple(self.laws))


@dataclass(frozen=True, slots=True)
class StatisticalTransportDistribution:
    """Plug-in target law with licensed or withheld sampling uncertainty."""

    outcomes: tuple[str, ...]
    atoms: tuple[tuple[float, ...], ...]
    probabilities: tuple[float, ...]
    formula: str
    rules: tuple[str, ...]
    uncertainty: TransportUncertainty | None = None
    _execution: Any = field(default=None, repr=False, compare=False)

    def inspect(self) -> Any:
        import json

        from ..results._report import InspectionReport

        if self._execution is None:
            raise _no_native_authority()
        return InspectionReport(**json.loads(self._execution.inspection_json()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcomes": self.outcomes,
            "atoms": self.atoms,
            "probabilities": self.probabilities,
            "formula": self.formula,
            "rules": self.rules,
            "uncertainty": self.uncertainty.to_dict() if self.uncertainty is not None else None,
            "reasoning": self.inspect().to_dict(),
        }

    def __repr__(self) -> str:
        state = "checked" if self._execution is not None else "unavailable"
        available = self._execution is not None and self.inspect().uncertainty.available
        return (
            f"StatisticalTransportDistribution(outcomes={self.outcomes!r}, probabilities={self.probabilities!r}, "
            f"identification={state}, support={state}, uncertainty={'pointwise_bootstrap' if available else 'unavailable'}, assumptions=declared)"
        )

    def mean(self, outcome: str) -> float:
        coordinate = self.outcomes.index(outcome)
        return math.fsum(
            atom[coordinate] * p for atom, p in zip(self.atoms, self.probabilities, strict=True)
        )

    def contrast(
        self, reference: StatisticalTransportDistribution, outcome: str
    ) -> TransportContrast:
        """Difference of means with paired bootstrap draws from compatible native runs."""
        if self._execution is None or reference._execution is None:
            raise _no_native_authority()
        import json

        result = dict(json.loads(self._execution.contrast(reference._execution, outcome)))
        if result["interval"] is not None:
            result["interval"] = tuple(result["interval"])
        return TransportContrast(result)

    def export(self) -> bytes:
        if self._execution is None:
            raise _no_native_authority()
        return bytes(self._execution.export())


def _no_native_authority() -> CausalUnsupportedError:
    """A display object edited or rebuilt away from its native execution cannot act."""
    return CausalUnsupportedError(
        "This display object has no native execution authority", reason_code="not_executed"
    )


@dataclass(frozen=True, slots=True)
class EmpiricalTable:
    """Unsmoothed empirical joint; empty cells retain sampling-zero semantics."""

    def _wire(self) -> dict[str, object]:
        return {"kind": "empirical_table"}


@dataclass(frozen=True, slots=True)
class LearnedCategorical:
    """Coherent finite categorical chain model, with empirical support checks.

    Model probabilities may extrapolate into empty joint cells. Such probabilities
    do not establish positivity; conditionals on unobserved strata still refuse.
    Bootstrap intervals are nominal and uncalibrated, not doubly robust.
    """

    learner: LearnerSpec | str = Logistic()

    def _wire(self) -> dict[str, object]:
        return {"kind": "learned_categorical", "learner": _learner_wire(self.learner)}


@dataclass(frozen=True, slots=True)
class StatisticalTransportQuery:
    """Checked identification plus empirical-table inference settings."""

    identification: ClassicalTransportIdentification
    catalog: EvidenceCatalog
    at: Mapping[str, float]
    bootstrap: int = OMITTED["transport_bootstrap"]
    coverage_level: float = OMITTED["transport_coverage_level"]
    estimator: EmpiricalTable | LearnedCategorical | str = "plugin"
    seed: int = 1

    def __post_init__(self) -> None:
        from types import MappingProxyType

        object.__setattr__(self, "at", MappingProxyType(dict(self.at)))


def prepare_statistical(
    identification: ClassicalTransportIdentification,
    catalog: EvidenceCatalog,
    data: StatisticalTransportData,
    *,
    at: Mapping[str, float],
    max_operations: int = 10_000_000,
    max_depth: int = 256,
    max_support_rows: int = 1_000_000,
    memory_bytes: int | None = None,
    cancel: Any = None,
    bootstrap: int = OMITTED["transport_bootstrap"],
    coverage_level: float = OMITTED["transport_coverage_level"],
    estimator: EmpiricalTable | LearnedCategorical | str = "plugin",
    seed: int = 1,
) -> PreparedAnalysis[StatisticalTransportDistribution]:
    """Prepare the empirical-table modality of the common PreparedAnalysis lifecycle."""
    from .._native import prepare_statistical_transport
    from ..estimation import PreparedAnalysis, _Controls

    native = prepare_statistical_transport(
        identification._native,
        catalog,
        data,
        dict(at),
        max_operations=max_operations,
        max_depth=max_depth,
        max_support_rows=max_support_rows,
        memory_bytes=memory_bytes,
        cancel=cancel,
        bootstrap=bootstrap,
        coverage_level=coverage_level,
        estimator=estimator,
        seed=seed,
    )
    return PreparedAnalysis(native, kind="statistical_transport", controls=_Controls(cancel=cancel))


def evaluate_statistical_grid(
    identification: ClassicalTransportIdentification,
    catalog: EvidenceCatalog,
    data: StatisticalTransportData,
    *,
    at: Sequence[Mapping[str, float]],
    **kwargs: Any,
) -> tuple[StatisticalTransportDistribution, ...]:
    """Evaluate a grid using shared dataset refits and shared successful replicate IDs.

    Intervals are pointwise. Use ``result.contrast(reference, outcome)`` to keep
    shared-factor covariance when comparing two points from the same execution.
    """
    if not at:
        raise CausalValueError("The treatment grid must not be empty")
    study = prepare_statistical(identification, catalog, data, at=at[0], **kwargs)
    points = study._native.estimate_grid([dict(point) for point in at], cancel=kwargs.get("cancel"))
    return tuple(_statistical_distribution(point, point.last_result()) for point in points)


def consume_statistical(
    artifact: bytes,
    *,
    max_operations: int = 10_000_000,
    max_depth: int = 256,
    max_support_rows: int = 1_000_000,
    memory_bytes: int | None = None,
    cancel: Any = None,
) -> PreparedAnalysis[StatisticalTransportDistribution]:
    """Verify portable proof and recomputed plug-in point; do not re-bootstrap.

    ``max_support_rows`` bounds the samples a later ``refresh`` or
    ``replace_snapshot`` may bind to the consumed handle.
    """
    from .. import _native
    from ..estimation import PreparedAnalysis, _Controls

    native = _native.consume_statistical_transport(
        artifact,
        max_operations=_non_negative("max_operations", max_operations),
        max_depth=_non_negative("max_depth", max_depth),
        max_support_rows=_non_negative("max_support_rows", max_support_rows),
        memory_bytes=_optional_non_negative("memory_bytes", memory_bytes),
        cancel=cancel,
    )
    return PreparedAnalysis(native, kind="statistical_transport", controls=_Controls(cancel=cancel))


@overload
def prepare(
    request: TrialAipwQuery,
    catalog: TrialAipwData,
    *,
    provider: TrialAipw | None = None,
    inference: TransportInference | None = None,
    controls: TransportControls | None = None,
) -> PreparedAnalysis[LearnedTrialEstimate]: ...


@overload
def prepare(
    request: ExactTransportQuery,
    catalog: ExactTransportData,
    *,
    controls: TransportControls | None = None,
) -> PreparedAnalysis[ExactTransportDistribution]: ...


@overload
def prepare(
    request: StatisticalTransportQuery,
    catalog: StatisticalTransportData,
    *,
    provider: EmpiricalTable | LearnedCategorical | str | None = None,
    inference: TransportInference | None = None,
    controls: TransportControls | None = None,
) -> PreparedAnalysis[StatisticalTransportDistribution]: ...


@overload
def prepare(
    request: TransportResponseGridQuery,
    catalog: ExactTransportData | StatisticalTransportData,
    *,
    provider: EmpiricalTable | LearnedCategorical | str | None = None,
    inference: TransportInference | None = None,
    controls: TransportControls | None = None,
) -> PreparedAnalysis[TransportResponseGrid]: ...


@overload
def prepare(
    request: ClassicalTransportIdentification,
    catalog: EvidenceCatalog,
    data: ExactTransportData,
    *,
    at: Mapping[str, float],
    controls: TransportControls | None = None,
) -> PreparedAnalysis[ExactTransportDistribution]: ...


@overload
def prepare(
    request: ClassicalTransportIdentification,
    catalog: EvidenceCatalog,
    data: StatisticalTransportData,
    *,
    at: Mapping[str, float],
    provider: EmpiricalTable | LearnedCategorical | str | None = None,
    inference: TransportInference | None = None,
    controls: TransportControls | None = None,
) -> PreparedAnalysis[StatisticalTransportDistribution]: ...


def prepare(
    request: ClassicalTransportIdentification
    | ExactTransportQuery
    | StatisticalTransportQuery
    | TransportResponseGridQuery
    | TrialAipwQuery,
    catalog: EvidenceCatalog
    | ExactTransportData
    | StatisticalTransportData
    | TrialAipwData
    | None = None,
    data: ExactTransportData | StatisticalTransportData | TrialAipwData | None = None,
    *,
    at: Mapping[str, float] | None = None,
    provider: EmpiricalTable | LearnedCategorical | TrialAipw | str | None = None,
    inference: TransportInference | None = None,
    controls: TransportControls | None = None,
) -> PreparedAnalysis[
    ExactTransportDistribution
    | StatisticalTransportDistribution
    | TransportResponseGrid
    | LearnedTrialEstimate
]:
    """Prepare a typed scalar, grid, or binary trial request on one lifecycle.

    Pass ``prepare(request, data, provider=..., inference=..., controls=...)``.
    The legacy ``prepare(identification, catalog, data, at=...)`` form remains.
    """
    provider_requested = provider is not None
    inference_requested = inference is not None
    controls = controls or TransportControls()
    if data is None and isinstance(
        catalog, (ExactTransportData, StatisticalTransportData, TrialAipwData)
    ):
        data, catalog = catalog, None
    if isinstance(request, TrialAipwQuery):
        if not isinstance(data, TrialAipwData) or (
            provider is not None and not isinstance(provider, TrialAipw)
        ):
            raise CausalTypeError("TrialAipwQuery requires TrialAipwData and a TrialAipw provider")
        if catalog is not None or at is not None:
            raise CausalValueError(
                "The binary trial request already declares its populations and contrast"
            )
        return prepare_trial(
            request,
            data,
            provider=provider or TrialAipw(),
            inference=inference or TransportInference(),
            controls=controls,
        )
    if isinstance(
        request, (ExactTransportQuery, StatisticalTransportQuery, TransportResponseGridQuery)
    ):
        if catalog is not None or at is not None:
            raise CausalValueError("The typed request already owns its catalog and coordinates")
        catalog = request.catalog
        if isinstance(request, (StatisticalTransportQuery, TransportResponseGridQuery)):
            inference = inference or TransportInference(
                request.bootstrap, request.coverage_level, request.seed
            )
            provider = provider or (
                EmpiricalTable() if request.estimator == "plugin" else request.estimator
            )
        if isinstance(request, TransportResponseGridQuery):
            if isinstance(data, ExactTransportData) and (
                provider_requested
                or inference_requested
                or not (
                    isinstance(request.estimator, EmpiricalTable) or request.estimator == "plugin"
                )
            ):
                raise CausalValueError(
                    "Exact grid laws do not accept learner or sampling inference settings"
                )
            if not isinstance(data, (ExactTransportData, StatisticalTransportData)) or isinstance(
                provider, TrialAipw
            ):
                raise CausalTypeError(
                    "Grid requests require exact or categorical-law data/providers"
                )
            settings = inference or TransportInference()
            return prepare_response_grid(
                request.identification,
                catalog,
                data,
                at=request.at,
                estimator=provider or EmpiricalTable(),
                bootstrap=settings.bootstrap,
                coverage_level=settings.coverage_level,
                seed=settings.seed,
                max_operations=controls.max_operations,
                max_depth=controls.max_depth,
                max_support_rows=controls.max_support_rows,
                memory_bytes=controls.memory_bytes,
                cancel=controls.cancel,
            )
        at = request.at
        request = request.identification
    if not isinstance(catalog, EvidenceCatalog) or at is None:
        raise CausalTypeError("Scalar transport requires a catalog and intervention coordinates")
    limits = dict(
        max_operations=controls.max_operations,
        max_depth=controls.max_depth,
        max_support_rows=controls.max_support_rows,
        memory_bytes=controls.memory_bytes,
        cancel=controls.cancel,
    )
    if isinstance(data, ExactTransportData):
        if provider is not None or inference is not None:
            raise CausalValueError(
                "Exact laws do not accept learner or sampling inference settings"
            )
        return prepare_exact(request, catalog, data, at=at, **limits)
    if not isinstance(data, StatisticalTransportData) or isinstance(provider, TrialAipw):
        raise CausalTypeError("Statistical transport requires categorical-law data/providers")
    settings = inference or TransportInference()
    return prepare_statistical(
        request,
        catalog,
        data,
        at=at,
        estimator=provider or EmpiricalTable(),
        bootstrap=settings.bootstrap,
        coverage_level=settings.coverage_level,
        seed=settings.seed,
        **limits,
    )


def _statistical_distribution(native: Any, payload: Any) -> StatisticalTransportDistribution:
    import json

    atoms, probabilities, formula, rules, uncertainty = payload
    parsed = json.loads(uncertainty) if isinstance(uncertainty, str) else uncertainty
    return StatisticalTransportDistribution(
        tuple(native.outcomes),
        tuple(tuple(row) for row in atoms),
        tuple(probabilities),
        formula,
        tuple(rules),
        uncertainty=TransportUncertainty(parsed),
        _execution=native.freeze(),
    )


__all__ += [
    "RegimeSample",
    "StatisticalTransportData",
    "StatisticalTransportDistribution",
    "StatisticalTransportQuery",
    "prepare",
    "prepare_statistical",
    "consume_statistical",
    "evaluate_statistical_grid",
]


def prepare_exact(
    identification: ClassicalTransportIdentification,
    catalog: EvidenceCatalog,
    data: ExactTransportData,
    *,
    at: Mapping[str, float],
    max_operations: int = 10_000_000,
    max_depth: int = 256,
    max_support_rows: int = 1_000_000,
    memory_bytes: int | None = None,
    cancel: Any = None,
) -> PreparedAnalysis[ExactTransportDistribution]:
    """Prepare the exact-law modality of the common PreparedAnalysis lifecycle."""
    from ..estimation import PreparedAnalysis, _Controls

    native = identification._native.prepare_exact(
        catalog,
        data.laws,
        dict(at),
        max_operations=max_operations,
        max_depth=max_depth,
        max_support_rows=max_support_rows,
        memory_bytes=memory_bytes,
        cancel=cancel,
    )
    return PreparedAnalysis(native, kind="exact_transport", controls=_Controls(cancel=cancel))


def consume_exact(
    artifact: bytes,
    *,
    max_operations: int = 10_000_000,
    max_depth: int = 256,
    memory_bytes: int | None = None,
    cancel: Any = None,
) -> PreparedAnalysis[ExactTransportDistribution]:
    """Verify portable proof, bindings, exact claims and all four reasoning slots.

    Uses embedded immutable laws; never fits models or fetches providers.
    """
    from .. import _native
    from ..estimation import PreparedAnalysis, _Controls

    native = _native.consume_exact_transport(
        artifact,
        max_operations=max_operations,
        max_depth=max_depth,
        memory_bytes=memory_bytes,
        cancel=cancel,
    )
    return PreparedAnalysis(native, kind="exact_transport", controls=_Controls(cancel=cancel))


__all__ += ["prepare_exact", "consume_exact"]


@dataclass(frozen=True, slots=True)
class ExactTransportQuery:
    """Checked identification, supplied evidence contract, and concrete target."""

    identification: ClassicalTransportIdentification
    catalog: EvidenceCatalog
    at: Mapping[str, float]

    def __post_init__(self) -> None:
        from types import MappingProxyType

        object.__setattr__(self, "at", MappingProxyType(dict(self.at)))


__all__ += ["ExactTransportQuery"]


def inspect_catalog(
    identification: ClassicalTransportIdentification,
    catalog: EvidenceCatalog,
    *,
    max_steps: int = 100_000,
    max_depth: int = 256,
    memory_bytes: int | None = None,
    cancel: Any = None,
) -> dict[str, Any]:
    """Inspect bounded alternatives and unmet evidence without fetching providers.

    ``exhausted`` concerns configured alternatives, never arbitrary finite-catalog
    completeness. Proposed future experiments cannot satisfy missing factors.
    """
    import json

    return dict(
        json.loads(
            identification._native.catalog_search(
                catalog,
                max_steps=max_steps,
                max_depth=max_depth,
                memory_bytes=memory_bytes,
                cancel=cancel,
            )
        )
    )


__all__ += ["inspect_catalog"]


def inspect_proof_graph(
    identification: ClassicalTransportIdentification,
    catalog: EvidenceCatalog,
    *,
    max_steps: int = 100_000,
    max_depth: int = 256,
) -> dict[str, Any]:
    """Return checked theorem steps and source-specific required factor leaves.

    Every leaf names its supplying regime or the exact binding failure. Proposed
    studies remain unavailable; use a hypothetical catalog delta for previews.
    """
    import json

    return dict(
        json.loads(
            identification._native.proof_graph_json(
                catalog,
                max_steps=max_steps,
                max_depth=max_depth,
            )
        )
    )


__all__ += ["inspect_proof_graph"]


@dataclass(frozen=True, slots=True)
class TransportResponseGridQuery:
    """Finite mean-response request; unsupported coordinates remain in the result."""

    identification: ClassicalTransportIdentification
    catalog: EvidenceCatalog
    at: Sequence[Mapping[str, float]]
    bootstrap: int = OMITTED["transport_bootstrap"]
    coverage_level: float = OMITTED["transport_coverage_level"]
    seed: int = 1
    estimator: EmpiricalTable | LearnedCategorical | str = "plugin"

    def __post_init__(self) -> None:
        from types import MappingProxyType

        object.__setattr__(self, "at", tuple(MappingProxyType(dict(point)) for point in self.at))


@dataclass(frozen=True, slots=True)
class TransportResponseGrid:
    """Native-authorized response family with explicit point-local failures."""

    points: tuple[TransportGridPoint, ...]
    execution_id: str
    _execution: Any = field(repr=False, compare=False)

    def mean(self, point: int, outcome: str) -> float:
        row = self.points[point]
        if row["status"] != "available":
            raise ValueError(f"Grid point {point} is unavailable: {row['detail']}")
        return float(row["means"][outcome])

    def contrast(self, left: int, right: int, outcome: str) -> TransportContrast:
        """Transform two points using native paired draws and retained parent identity."""
        import json

        result = dict(json.loads(self._execution.contrast(left, right, outcome)))
        if result["interval"] is not None:
            result["interval"] = tuple(result["interval"])
        return TransportContrast(result)

    def scalar_projection(self, point: int, outcome: str) -> dict[str, Any]:
        """Return a scalar and loss receipt; this projection cannot impersonate the full grid."""
        import json

        return dict(json.loads(self._execution.scalar_projection(point, outcome)))

    def inspect(self) -> Any:
        import json

        from ..results._report import InspectionReport

        return InspectionReport(**json.loads(self._execution.inspection_json()))

    def export(self) -> bytes:
        return bytes(self._execution.export())


def _response_grid(native: Any, payload: str) -> TransportResponseGrid:
    import json

    parsed = json.loads(payload)
    return TransportResponseGrid(
        tuple(TransportGridPoint(point) for point in parsed["points"]),
        parsed["execution_id"],
        native.freeze(),
    )


def prepare_response_grid(
    identification: ClassicalTransportIdentification,
    catalog: EvidenceCatalog,
    data: ExactTransportData | StatisticalTransportData,
    *,
    at: Sequence[Mapping[str, float]],
    estimator: EmpiricalTable | LearnedCategorical | str = "plugin",
    bootstrap: int = OMITTED["transport_bootstrap"],
    coverage_level: float = OMITTED["transport_coverage_level"],
    seed: int = 1,
    max_operations: int = 10_000_000,
    max_depth: int = 256,
    max_support_rows: int = 1_000_000,
    memory_bytes: int | None = None,
    cancel: Any = None,
) -> PreparedAnalysis[TransportResponseGrid]:
    """Prepare exact or empirical finite responses on the common study lifecycle."""
    from .. import _native
    from ..estimation import PreparedAnalysis, _Controls

    if not isinstance(data, (ExactTransportData, StatisticalTransportData)):
        raise TypeError("Grid data must be exact laws or explicit statistical providers")
    native = _native.prepare_transport_grid(
        identification._native,
        catalog,
        data,
        [dict(point) for point in at],
        statistical=isinstance(data, StatisticalTransportData),
        estimator=estimator,
        bootstrap=bootstrap,
        coverage_level=coverage_level,
        seed=seed,
        max_operations=max_operations,
        max_depth=max_depth,
        max_support_rows=max_support_rows,
        memory_bytes=memory_bytes,
        cancel=cancel,
    )
    return PreparedAnalysis(native, kind="transport_grid", controls=_Controls(cancel=cancel))


def consume_response_grid(
    artifact: bytes,
    *,
    max_operations: int = 10_000_000,
    max_depth: int = 256,
    memory_bytes: int | None = None,
    cancel: Any = None,
) -> PreparedAnalysis[TransportResponseGrid]:
    """Verify a grid without fetching, fitting, or rebuilding raw samples."""
    from .. import _native
    from ..estimation import PreparedAnalysis, _Controls

    native = _native.consume_transport_grid(
        artifact,
        max_operations=max_operations,
        max_depth=max_depth,
        memory_bytes=memory_bytes,
        cancel=cancel,
    )
    return PreparedAnalysis(native, kind="transport_grid", controls=_Controls(cancel=cancel))


__all__ += [
    "TransportResponseGridQuery",
    "TransportResponseGrid",
    "prepare_response_grid",
    "consume_response_grid",
]

__all__ += ["consume_identification"]

__all__ += ["EmpiricalTable", "LearnedCategorical"]


@dataclass(frozen=True, slots=True)
class TrialAipwData:
    """IID trial and target rows with known source randomization probabilities.

    For ``nested_cohort``, target rows are cohort nonparticipants. For
    ``independent_samples``, target rows are a separately sampled representative
    target population. Target outcomes/treatments are ignored; use finite placeholders.
    """

    covariates: Mapping[str, Sequence[float]]
    outcome: Sequence[float]
    treatment: Sequence[bool]
    source: Sequence[bool]
    randomization: Sequence[float]
    sampling: Literal["nested_cohort", "independent_samples"]

    def _json(self, names: Sequence[str]) -> str:
        import json

        unknown = set(self.covariates) - set(names)
        if unknown:
            raise ValueError(f"Unknown baseline covariates: {sorted(unknown)}")
        features = [i for i, name in enumerate(names) if name in self.covariates]
        return json.dumps(
            dict(
                features=features,
                covariates=[list(map(float, self.covariates[names[i]])) for i in features],
                outcome=list(map(float, self.outcome)),
                treatment=list(self.treatment),
                source=list(self.source),
                randomization=list(map(float, self.randomization)),
                sampling=self.sampling,
            ),
            allow_nan=False,
        )


@dataclass(frozen=True, slots=True)
class TrialAipw:
    """Nuisance configuration for the certified binary trial contrast."""

    outcome: LearnerSpec = Ridge()
    membership: LearnerSpec = Logistic()
    folds: int = 5


@dataclass(frozen=True, slots=True)
class TrialAipwQuery:
    graph: Admg
    diagram: SelectionDiagram
    treatment: str
    outcome: str


@dataclass(frozen=True, slots=True)
class TransportInference:
    """Frozen joint outer bootstrap; intervals remain nominal and uncalibrated."""

    bootstrap: int = OMITTED["transport_bootstrap"]
    coverage_level: float = OMITTED["transport_coverage_level"]
    seed: int = 1

    def __post_init__(self) -> None:
        _non_negative("bootstrap", self.bootstrap)
        _non_negative("seed", self.seed)
        if not isinstance(self.coverage_level, (int, float)) or not (
            0.0 < float(self.coverage_level) < 1.0
        ):
            raise CausalValueError("coverage_level must lie strictly between 0 and 1")


@dataclass(frozen=True, slots=True)
class TransportControls:
    """Physical resource limits, separate from scientific inference settings."""

    max_operations: int = 10_000_000
    max_depth: int = 256
    max_support_rows: int = 1_000_000
    memory_bytes: int | None = None
    cancel: Any = None

    def __post_init__(self) -> None:
        _non_negative("max_operations", self.max_operations)
        _non_negative("max_depth", self.max_depth)
        _non_negative("max_support_rows", self.max_support_rows)
        _optional_non_negative("memory_bytes", self.memory_bytes)


@dataclass(frozen=True, slots=True)
class TrialNuisanceDiagnostics:
    """Held-out role losses; these do not establish causal identification."""

    membership_logloss: float
    outcome_rmse: tuple[float, float]


@dataclass(frozen=True, slots=True)
class LearnedTrialEstimate:
    estimate: float
    interval: tuple[float, float] | None
    uncertainty_reason: str | None
    replicates: tuple[tuple[int, float], ...]
    failures: int
    overlap: Mapping[str, Any]
    diagnostics: TrialNuisanceDiagnostics
    _execution: Any = field(repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimate": self.estimate,
            "interval": list(self.interval) if self.interval is not None else None,
            "uncertainty_reason": self.uncertainty_reason,
            "replicates": [list(row) for row in self.replicates],
            "failures": self.failures,
            "overlap": {k: dict(v) for k, v in self.overlap.items()},
            "diagnostics": {
                "membership_logloss": self.diagnostics.membership_logloss,
                "outcome_rmse": list(self.diagnostics.outcome_rmse),
            },
            "reasoning": self.inspect().to_dict(),
        }

    def export(self) -> bytes:
        return bytes(self._execution.export())

    def inspect(self) -> Any:
        import json

        from ..results._report import InspectionReport

        return InspectionReport(**json.loads(self._execution.inspection_json()))


def _learned_trial(native: Any, payload: str) -> LearnedTrialEstimate:
    import json
    from types import MappingProxyType

    raw = json.loads(payload)
    return LearnedTrialEstimate(
        raw["estimate"],
        tuple(raw["interval"]) if raw["interval"] else None,
        raw["uncertainty_reason"],
        tuple((i, v) for i, v in raw["replicates"]),
        raw["failures"],
        MappingProxyType({k: MappingProxyType(v) for k, v in raw["overlap"].items()}),
        TrialNuisanceDiagnostics(
            raw["diagnostics"]["membership_logloss"], tuple(raw["diagnostics"]["outcome_rmse"])
        ),
        native.freeze(),
    )


def prepare_trial(
    query: TrialAipwQuery,
    data: TrialAipwData,
    *,
    provider: TrialAipw | None = None,
    inference: TransportInference | None = None,
    controls: TransportControls | None = None,
) -> PreparedAnalysis[LearnedTrialEstimate]:
    """Prepare a checked binary contrast; baseline covariates must match its certificate."""
    import json

    from .. import _native
    from ..estimation import PreparedAnalysis, _Controls

    provider = provider or TrialAipw()
    inference = inference or TransportInference()
    controls = controls or TransportControls()
    if (controls.max_operations, controls.max_depth, controls.max_support_rows) != (
        10_000_000,
        256,
        1_000_000,
    ):
        raise CausalUnsupportedError(
            "Discrete evaluator limits do not apply to trial AIPW; use memory_bytes and cancel",
            reason_code="option_not_applicable",
        )
    options = dict(
        outcome=_learner_wire(provider.outcome),
        membership=_learner_wire(provider.membership),
        folds=provider.folds,
        bootstrap=inference.bootstrap,
        coverage_level=inference.coverage_level,
    )
    native = _native.prepare_learned_trial(
        query.graph,
        list(query.diagram.selections),
        query.diagram.source,
        query.diagram.target,
        query.treatment,
        query.outcome,
        data,
        json.dumps(options),
        seed=inference.seed,
        memory_bytes=controls.memory_bytes,
        cancel=controls.cancel,
    )
    return PreparedAnalysis(
        native, kind="learned_trial", controls=_Controls(cancel=controls.cancel)
    )


__all__ += [
    "EmpiricalTable",
    "LearnedCategorical",
    "TrialAipw",
    "TrialAipwData",
    "TrialAipwQuery",
    "TransportInference",
    "TransportControls",
    "LearnedTrialEstimate",
    "prepare_trial",
]

__all__ += ["TransportContrast", "TransportUncertainty"]

__all__ += ["TrialNuisanceDiagnostics"]

__all__ += ["TransportGridPoint"]
