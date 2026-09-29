"""Randomization designs and exposure mappings for interference queries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field
from typing import Any, Literal

import numpy as np

from ._data import as_columns
from ._native import estimate_network_interference as _estimate_network_interference
from ._native import (
    estimate_observational_network_exposure as _estimate_observational_network_exposure,
)
from ._native import estimate_saturation_interference as _estimate_saturation_interference
from .errors import CausalEstimateError, CausalTypeError, CausalValueError


@dataclass(frozen=True, slots=True)
class BernoulliAssignment:
    """Independent assignment with one or unit-specific probabilities."""

    probabilities: float | Sequence[float]

    def __post_init__(self) -> None:
        values = (
            [self.probabilities]
            if isinstance(self.probabilities, (int, float))
            else self.probabilities
        )
        if not values or any(not 0.0 < value < 1.0 for value in values):
            raise CausalValueError("probabilities must be strictly between 0 and 1")


@dataclass(frozen=True, slots=True)
class CompleteRandomization:
    treated: int

    def __post_init__(self) -> None:
        if self.treated <= 0:
            raise CausalValueError("treated must be positive")


@dataclass(frozen=True, slots=True)
class ClusterRandomization:
    clusters: Sequence[int]
    treated_clusters: int

    def __post_init__(self) -> None:
        if not self.clusters or self.treated_clusters <= 0:
            raise CausalValueError("clusters and a positive treated_clusters are required")


@dataclass(frozen=True, slots=True)
class PartialInterference:
    """Explicit partial-interference assumption, checked against the supplied network.

    Cluster labels are in unit-row order. Every network edge must remain within a
    label, and the labels must describe the same partition used for cluster
    randomization. This is an assumption plus a topology check, not evidence that
    unrecorded interference is absent.
    """

    clusters: Sequence[int]

    def __post_init__(self) -> None:
        if len(self.clusters) == 0:
            raise CausalValueError("partial-interference clusters must not be empty")


@dataclass(frozen=True, slots=True)
class SaturationDesign:
    """Two-stage design: complete cluster allocation, then Bernoulli units.

    ``realized_saturation`` has one probability per row; rows in the same
    cluster must carry the same declared low or high saturation probability.
    Exactly ``high_clusters`` clusters receive the high saturation arm.
    """

    clusters: Sequence[int]
    low_probability: float
    high_probability: float
    high_clusters: int
    realized_saturation: Sequence[float]

    def __post_init__(self) -> None:
        if len(self.clusters) == 0 or len(self.realized_saturation) == 0:
            raise CausalValueError(
                "saturation design requires cluster labels and realized saturation"
            )
        if (
            not np.isfinite(self.low_probability)
            or not np.isfinite(self.high_probability)
            or not 0.0 < self.low_probability < self.high_probability < 1.0
        ):
            raise CausalValueError("saturation probabilities must satisfy 0 < low < high < 1")
        if self.high_clusters <= 0:
            raise CausalValueError("high_clusters must be positive")


@dataclass(frozen=True, slots=True)
class ObservedExposureDesign:
    """Observed network assignment with supplied exposure probabilities.

    ``assume_network_exchangeability`` declares no unmeasured network-exposure
    confounding given the variables used to supply the probability vectors.
    Antecedent checks positivity and network partition but cannot verify this
    assumption or external propensity fitting.
    """

    clusters: Sequence[int]
    propensity_from: Sequence[float]
    propensity_to: Sequence[float]
    propensity_provenance: Literal["known", "externally_estimated"]
    assume_network_exchangeability: bool

    def __post_init__(self) -> None:
        n = len(self.clusters)
        if n == 0 or len(self.propensity_from) != n or len(self.propensity_to) != n:
            raise CausalValueError(
                "observational exposure probabilities and clusters must align by row"
            )
        if self.propensity_provenance not in {"known", "externally_estimated"}:
            raise CausalValueError("propensity_provenance must be known or externally_estimated")
        if self.assume_network_exchangeability is not True:
            raise CausalValueError(
                "observational exposure requires assume_network_exchangeability=True"
            )
        for values in (self.propensity_from, self.propensity_to):
            if any(not np.isfinite(p) or p <= 0.0 or p > 1.0 for p in values):
                raise CausalValueError("exposure probabilities must lie in (0, 1]")


@dataclass(frozen=True, slots=True)
class SaturationEffectEstimate:
    horvitz_thompson: float
    hajek: float
    conservative_variance: float
    from_exposed_units: int
    to_exposed_units: int
    from_exposed_clusters: int
    to_exposed_clusters: int
    minimum_exposure_probability: float


@dataclass(frozen=True, slots=True)
class SaturationEffects:
    direct: SaturationEffectEstimate
    spillover: SaturationEffectEstimate
    total: SaturationEffectEstimate
    support_status: str = "unlicensed_point_utility"
    uncertainty_semantics: str = "Covariance-free variance bounds cover both randomization stages; no confidence interval or calibration claim is made."
    assumptions: tuple[str, ...] = (
        "Clusters are completely randomized to the declared low/high saturation counts.",
        "Units are independently Bernoulli assigned within each cluster at its realized saturation probability.",
        "Partial interference holds within the supplied network clusters; consistency holds.",
    )


@dataclass(frozen=True, slots=True)
class ObservationalNetworkExposureEstimate:
    horvitz_thompson: float
    hajek: float
    cluster_robust_variance: float
    from_exposed_units: int
    to_exposed_units: int
    from_exposed_clusters: int
    to_exposed_clusters: int
    minimum_exposure_probability: float
    maximum_exposure_probability: float
    clusters: int
    propensity_provenance: str
    uncertainty_semantics: str = (
        "Cluster sandwich variance for the HT contrast, treating supplied exposure "
        "propensities as fixed; no interval is reported and no calibration claim is made."
    )
    assumptions: tuple[str, ...] = (
        "No unmeasured network confounding conditional on the covariates used for the supplied exposure probabilities.",
        "Consistency, correct fixed network, and positivity at every requested exposure for every unit.",
        "Partial interference across the explicitly supplied cluster partition.",
        "Supplied exposure probabilities are correctly specified or known.",
    )
    support_status: str = "unlicensed_point_utility"


@dataclass(frozen=True, slots=True)
class OwnTreatment:
    """Exposure is the unit's own treatment only."""


@dataclass(frozen=True, slots=True)
class NeighborCount:
    """Exposure includes the number of treated incoming neighbors."""


@dataclass(frozen=True, slots=True)
class NeighborFraction:
    """Exposure includes the fraction of treated incoming neighbors."""


@dataclass(frozen=True, slots=True)
class WeightedNeighborExposure:
    """Exposure includes a weighted mean of incoming-neighbor treatment."""


@dataclass(frozen=True, slots=True)
class ExposureLevel:
    own: float
    neighbors: float = 0.0


@dataclass(frozen=True, slots=True)
class ExposureContrast:
    outcome: str
    from_: ExposureLevel
    to: ExposureLevel

    def __post_init__(self) -> None:
        if not self.outcome.strip():
            raise CausalValueError("outcome must be a non-empty variable name")
        if self.from_ == self.to:
            raise CausalValueError("exposure levels must be distinct")


@dataclass(frozen=True, slots=True)
class InterferenceQuery:
    """An exposure contrast on a fixed unit network.

    ``assignment`` is the declared assignment or observation design, ``exposure`` the exposure
    mapping and ``functional`` the contrast between two exposure levels.

    :func:`antecedent.analyze` estimates the licensed cell (explicit ``Dag``,
    Frequentist, validation ``none``): NeighborCount exposure under Bernoulli
    assignment, Horvitz–Thompson / Hájek with the conservative Young variance
    bound. A separate unlicensed construction accepts cluster randomization,
    NeighborFraction, an explicit partial-interference partition, and the
    total contrast from ``(0, 0)`` to ``(1, 1)``; its cluster-level variance
    has no interval claim. ``SaturationDesign`` also supports exact two-stage
    direct, spillover, and total exposure contrasts as separate point queries;
    its covariance-free variance proxy is not a confidence interval.
    ``ObservedExposureDesign`` uses supplied exposure probabilities and an
    explicit network-exchangeability declaration. It reports a point contrast
    and descriptive cluster variance without an interval.
    These paths need the keyword-only design facts: ``network``, the fixed
    directed exposure edges between unit rows (``NetworkEdge`` or
    ``(from, to[, weight])``; an empty sequence is a network without edges),
    and ``realized_assignment``, the binary assignment in unit-row order. Both
    freeze at prepare; the unit table is the data, so a refresh executes on
    new outcomes under the same network and assignment.
    """

    assignment: object
    exposure: object
    functional: ExposureContrast
    probability_draws: int = 10_000
    _: KW_ONLY
    network: Sequence[NetworkEdge | tuple[int, int] | tuple[int, int, float]] | None = None
    realized_assignment: Sequence[bool] | str | None = None
    partial_interference: PartialInterference | None = None
    kind: Literal["interference"] = field(default="interference", init=False, repr=False)
    _deferred_columns: bool = field(default=False, init=False, repr=False, compare=False)

    #: The row-aligned realized assignment may instead name a data column; ``network``
    #: is an edge list, not a per-row column, so it stays inline.
    _COLUMN_FIELDS = {"realized_assignment": "bool"}

    def __post_init__(self) -> None:
        from ._columns import defer_columns

        if defer_columns(self):
            # Hold the caller's column name unvalidated; resolve_columns rebuilds this
            # query against the data at prepare, re-running the checks below.
            object.__setattr__(self, "_deferred_columns", True)
            return
        if self.probability_draws <= 0:
            raise CausalValueError("probability_draws must be positive")
        if (self.network is None) != (self.realized_assignment is None):
            raise CausalValueError(
                "network and realized_assignment are the design together; supply both"
            )


@dataclass(frozen=True, slots=True)
class NetworkEdge:
    """Directed weighted exposure edge between unit-row indexes."""

    from_: int
    to: int
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.from_ < 0 or self.to < 0:
            raise CausalValueError("network edge indexes must be non-negative")
        if self.from_ == self.to:
            raise CausalValueError("network self-edges are not allowed")
        if not np.isfinite(self.weight) or self.weight < 0.0:
            raise CausalValueError("network edge weights must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class RandomizationContrast:
    horvitz_thompson: float
    hajek: float
    conservative_variance: float


@dataclass(frozen=True, slots=True)
class InterferencePointwiseInterval:
    """A 95% interval for one declared exposure contrast across independent clusters.

    ``first_stage_arm_clusters`` counts control/treated clusters for complete
    cluster randomization and low/high clusters for two-stage saturation.
    The interval applies only to this contrast, not to an exposure-response curve.
    """

    lower: float
    upper: float
    standard_error: float
    degrees_of_freedom: float
    first_stage_arm_clusters: tuple[int, int]
    method: str
    level: float = 0.95


@dataclass(frozen=True, slots=True)
class InterferenceEstimate:
    contrast: RandomizationContrast
    from_probability_method: str
    to_probability_method: str
    minimum_exposure_probability: float
    support: InterferenceSupport | None = None
    uncertainty_semantics: str = (
        "Conservative covariance-free Young variance bound under the declared "
        "randomization; it is not a cluster-robust interval or a calibrated CI."
    )
    provenance: Mapping[str, Any] = field(
        default_factory=lambda: {"operation_ids": ["stats.randomized_interference"]}
    )
    pointwise_interval: InterferencePointwiseInterval | None = None
    interval_unavailable_reason: str | None = None
    support_status: str = "unlicensed_point_utility"


@dataclass(frozen=True, slots=True)
class InterferenceSupport:
    """Observed exposure counts for a direct utility result."""

    from_exposed_units: int
    to_exposed_units: int
    from_observed_clusters: int | None
    to_observed_clusters: int | None
    minimum_exposure_probability: float
    partial_interference_checked: bool
    maximum_exposure_probability: float | None = None
    clusters: int | None = None


def _assignment_args(design: object) -> dict[str, Any]:
    if isinstance(design, ObservedExposureDesign):
        return {
            "assignment_kind": "observed_exposure",
            "assignment_probabilities": [],
            "treated": 0,
            "clusters": list(design.clusters),
            "treated_clusters": 0,
            "propensity_from": list(design.propensity_from),
            "propensity_to": list(design.propensity_to),
            "propensity_provenance": design.propensity_provenance,
            "assume_network_exchangeability": design.assume_network_exchangeability,
        }
    if isinstance(design, SaturationDesign):
        return {
            "assignment_kind": "saturation",
            "assignment_probabilities": [],
            "treated": 0,
            "clusters": list(design.clusters),
            "treated_clusters": design.high_clusters,
            "low_probability": design.low_probability,
            "high_probability": design.high_probability,
            "realized_saturation": list(design.realized_saturation),
        }
    if isinstance(design, BernoulliAssignment):
        probabilities = (
            [float(design.probabilities)]
            if isinstance(design.probabilities, (int, float))
            else list(design.probabilities)
        )
        return {
            "assignment_kind": "bernoulli",
            "assignment_probabilities": probabilities,
            "treated": 0,
            "clusters": [],
            "treated_clusters": 0,
        }
    if isinstance(design, CompleteRandomization):
        return {
            "assignment_kind": "complete",
            "assignment_probabilities": [],
            "treated": design.treated,
            "clusters": [],
            "treated_clusters": 0,
        }
    if isinstance(design, ClusterRandomization):
        return {
            "assignment_kind": "cluster",
            "assignment_probabilities": [],
            "treated": 0,
            "clusters": list(design.clusters),
            "treated_clusters": design.treated_clusters,
        }
    raise CausalTypeError("unsupported assignment design")


def _exposure_name(exposure: object) -> str:
    if isinstance(exposure, OwnTreatment):
        return "own_treatment"
    if isinstance(exposure, NeighborCount):
        return "neighbor_count"
    if isinstance(exposure, NeighborFraction):
        return "neighbor_fraction"
    if isinstance(exposure, WeightedNeighborExposure):
        return "weighted_neighbor_exposure"
    raise CausalTypeError("unsupported exposure mapping")


def _edge_values(
    edges: Sequence[NetworkEdge | tuple[int, int] | tuple[int, int, float]],
) -> list[tuple[int, int, float]]:
    values: list[tuple[int, int, float]] = []
    for edge in edges:
        if isinstance(edge, NetworkEdge):
            values.append((edge.from_, edge.to, edge.weight))
        elif len(edge) == 2:
            values.append((edge[0], edge[1], 1.0))
        else:
            values.append((edge[0], edge[1], edge[2]))
    return values


def estimate(
    data: Any,
    *,
    assignment: Sequence[bool],
    edges: Sequence[NetworkEdge | tuple[int, int] | tuple[int, int, float]],
    query: InterferenceQuery,
    seed: int = 1,
) -> InterferenceEstimate:
    """Estimate a known randomized exposure contrast on a fixed unit network.

    This is an unlicensed utility: every assignment
    design and exposure mapping, and ``seed`` as the exposure-probability Monte
    Carlo seed. It calls the interference estimator directly and returns bare
    numbers, with no study, contract, export or calibration slot.
    ``antecedent.analyze(data, graph=[], query=InterferenceQuery(..., network=,
    realized_assignment=))`` retains a study for the licensed Bernoulli cell
    and the separately checked, unlicensed cluster total-effect construction.
    """

    if not isinstance(query, InterferenceQuery):
        raise CausalTypeError("query must be an InterferenceQuery")
    if isinstance(query.assignment, ObservedExposureDesign):
        raise CausalValueError(
            "ObservedExposureDesign runs through analyze/prepare; use "
            "estimate_observational_network_exposure for the direct utility"
        )
    if isinstance(query.assignment, SaturationDesign):
        raise CausalValueError(
            "SaturationDesign runs through analyze/prepare; use estimate_saturation_effects "
            "for the direct utility"
        )
    names, columns = as_columns(data)
    try:
        outcome_index = names.index(query.functional.outcome)
    except ValueError as error:
        raise CausalValueError(
            f"outcome column {query.functional.outcome!r} is missing from data"
        ) from error
    edge_values = _edge_values(edges)
    n = len(columns[outcome_index])
    if len(assignment) != n:
        raise CausalValueError("assignment length must match data rows")
    if any(
        source < 0 or target < 0 or source >= n or target >= n for source, target, _ in edge_values
    ):
        raise CausalValueError("network edge index is outside data rows")
    partial_clusters: list[int] | None = None
    if query.partial_interference is not None:
        if not isinstance(query.assignment, ClusterRandomization):
            raise CausalValueError(
                "partial_interference requires ClusterRandomization so cluster assignment is explicit"
            )
        partial_clusters = list(query.partial_interference.clusters)
        assignment_clusters = list(query.assignment.clusters)
        if len(partial_clusters) != n or len(assignment_clusters) != n:
            raise CausalValueError(
                "partial-interference and assignment clusters must match data rows"
            )
        if _partition(partial_clusters) != _partition(assignment_clusters):
            raise CausalValueError(
                "partial-interference clusters must match the cluster-randomization partition"
            )
        if any(
            partial_clusters[source] != partial_clusters[target]
            for source, target, _ in edge_values
        ):
            raise CausalValueError(
                "partial-interference assumption violated: network edge crosses cluster boundary"
            )
    observed_levels = _observed_exposure(assignment, edge_values, query.exposure)
    from_rows = [
        i for i, level in enumerate(observed_levels) if _same_level(level, query.functional.from_)
    ]
    to_rows = [
        i for i, level in enumerate(observed_levels) if _same_level(level, query.functional.to)
    ]
    cluster_labels = partial_clusters
    if cluster_labels is None and isinstance(query.assignment, ClusterRandomization):
        cluster_labels = list(query.assignment.clusters)
    from_cluster_count = (
        len({cluster_labels[i] for i in from_rows}) if cluster_labels is not None else None
    )
    to_cluster_count = (
        len({cluster_labels[i] for i in to_rows}) if cluster_labels is not None else None
    )
    raw = _estimate_network_interference(
        columns[outcome_index],
        list(assignment),
        edge_values,
        exposure=_exposure_name(query.exposure),
        from_level=(query.functional.from_.own, query.functional.from_.neighbors),
        to_level=(query.functional.to.own, query.functional.to.neighbors),
        probability_draws=query.probability_draws,
        seed=seed,
        **_assignment_args(query.assignment),
    )
    return InterferenceEstimate(
        RandomizationContrast(
            raw.horvitz_thompson,
            raw.hajek,
            raw.conservative_variance,
        ),
        raw.from_probability_method,
        raw.to_probability_method,
        raw.minimum_exposure_probability,
        InterferenceSupport(
            len(from_rows),
            len(to_rows),
            from_cluster_count,
            to_cluster_count,
            raw.minimum_exposure_probability,
            partial_clusters is not None,
        ),
    )


def estimate_saturation_effects(
    data: Any,
    *,
    assignment: Sequence[bool],
    edges: Sequence[NetworkEdge | tuple[int, int] | tuple[int, int, float]],
    design: SaturationDesign,
    partial_interference: PartialInterference,
    exposure: NeighborCount | NeighborFraction | WeightedNeighborExposure,
    outcome: str,
    reference_neighbor_exposure: float,
    low_neighbor_exposure: float,
    high_neighbor_exposure: float,
) -> SaturationEffects:
    """Estimate direct, spillover, and total effects under two-stage saturation.

    Direct effect compares own treatment at the reference neighbor exposure.
    Spillover compares low/high neighbor exposure among untreated units. Total
    compares untreated/low with treated/high. Exact cluster allocations and
    within-cluster Bernoulli assignments provide each unit's exposure
    probabilities. This direct utility is not licensed on ``analyze``.
    """

    if not isinstance(design, SaturationDesign):
        raise CausalTypeError("design must be a SaturationDesign")
    if not isinstance(partial_interference, PartialInterference):
        raise CausalTypeError("partial_interference must explicitly state cluster labels")
    if not isinstance(exposure, (NeighborCount, NeighborFraction, WeightedNeighborExposure)):
        raise CausalTypeError(
            "saturation effects require NeighborCount, NeighborFraction, or WeightedNeighborExposure"
        )
    names, columns = as_columns(data)
    try:
        y = np.asarray(columns[names.index(outcome)], dtype=np.float64)
    except ValueError as error:
        raise CausalValueError(f"outcome column {outcome!r} is missing from data") from error
    n = len(y)
    assignments = _assignment_values(assignment, n)
    clusters = list(design.clusters)
    partial_clusters = list(partial_interference.clusters)
    saturation = list(design.realized_saturation)
    if len(clusters) != n or len(partial_clusters) != n or len(saturation) != n:
        raise CausalValueError("saturation and partial-interference vectors must match data rows")
    if _partition(clusters) != _partition(partial_clusters):
        raise CausalValueError(
            "partial-interference clusters must match saturation-design clusters"
        )
    edge_values = _edge_values(edges)
    if any(
        source < 0 or target < 0 or source >= n or target >= n for source, target, _ in edge_values
    ):
        raise CausalValueError("network edge index is outside data rows")
    if any(
        partial_clusters[source] != partial_clusters[target] for source, target, _ in edge_values
    ):
        raise CausalValueError(
            "partial-interference assumption violated: network edge crosses cluster boundary"
        )
    if any(
        not np.isfinite(value)
        or abs(value - design.low_probability) > 1e-12
        and abs(value - design.high_probability) > 1e-12
        for value in saturation
    ):
        raise CausalValueError(
            "realized saturation must equal the design's low or high probability"
        )
    if not all(
        np.isfinite([reference_neighbor_exposure, low_neighbor_exposure, high_neighbor_exposure])
    ):
        raise CausalValueError("neighbor exposure targets must be finite")
    if (
        low_neighbor_exposure < 0.0
        or high_neighbor_exposure < 0.0
        or reference_neighbor_exposure < 0.0
    ):
        raise CausalValueError("neighbor exposure targets must be non-negative")
    if abs(low_neighbor_exposure - high_neighbor_exposure) <= 1e-12:
        raise CausalValueError("low and high neighbor exposure targets must differ")
    raw = _estimate_saturation_interference(
        y,
        assignments,
        edge_values,
        clusters,
        saturation,
        design.low_probability,
        design.high_probability,
        design.high_clusters,
        _exposure_name(exposure),
        reference_neighbor_exposure,
        low_neighbor_exposure,
        high_neighbor_exposure,
    )
    direct, spillover, total = raw
    return SaturationEffects(
        SaturationEffectEstimate(*direct),
        SaturationEffectEstimate(*spillover),
        SaturationEffectEstimate(*total),
    )


def estimate_observational_network_exposure(
    data: Any,
    *,
    outcome: str,
    assignment: Sequence[bool],
    edges: Sequence[NetworkEdge | tuple[int, int] | tuple[int, int, float]],
    clusters: Sequence[int],
    partial_interference: PartialInterference,
    exposure: NeighborCount | NeighborFraction | WeightedNeighborExposure,
    from_level: ExposureLevel,
    to_level: ExposureLevel,
    propensity_from: float | Sequence[float],
    propensity_to: float | Sequence[float],
    propensity_provenance: str,
) -> ObservationalNetworkExposureEstimate:
    """Estimate an observational network-exposure contrast from supplied propensities.

    This point utility does not identify effects from the network alone. The
    caller must justify the declared network-confounding assumptions and the
    known or externally estimated probabilities for both queried exposures.
    """

    if not isinstance(partial_interference, PartialInterference):
        raise CausalTypeError("partial_interference must explicitly state cluster labels")
    if not isinstance(exposure, (NeighborCount, NeighborFraction, WeightedNeighborExposure)):
        raise CausalTypeError("observational network exposure requires a neighbor exposure mapping")
    if not isinstance(propensity_provenance, str) or propensity_provenance not in {
        "known",
        "externally_estimated",
    }:
        raise CausalValueError("propensity_provenance must be known or externally_estimated")
    if not isinstance(from_level, ExposureLevel) or not isinstance(to_level, ExposureLevel):
        raise CausalTypeError("from_level and to_level must be ExposureLevel values")
    if from_level == to_level:
        raise CausalValueError("from_level and to_level must differ")
    names, columns = as_columns(data)
    try:
        y = np.asarray(columns[names.index(outcome)], dtype=np.float64)
    except ValueError as error:
        raise CausalValueError(f"outcome column {outcome!r} is missing from data") from error
    n = len(y)
    assignments = _assignment_values(assignment, n)
    cluster_values = list(clusters)
    partial_clusters = list(partial_interference.clusters)
    if len(cluster_values) != n or len(partial_clusters) != n:
        raise CausalValueError("cluster labels must have one value per data row")
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 0
        or value > 2**32 - 1
        for value in cluster_values
    ):
        raise CausalValueError("cluster labels must be non-negative 32-bit integers")
    if _partition(cluster_values) != _partition(partial_clusters):
        raise CausalValueError("partial-interference clusters must match cluster labels")
    edge_values = _edge_values(edges)
    if any(
        source < 0 or target < 0 or source >= n or target >= n for source, target, _ in edge_values
    ):
        raise CausalValueError("network edge index is outside data rows")
    if any(
        partial_clusters[source] != partial_clusters[target] for source, target, _ in edge_values
    ):
        raise CausalValueError(
            "partial-interference assumption violated: network edge crosses cluster boundary"
        )
    p_from = _exposure_probability_vector(propensity_from, n, "propensity_from")
    p_to = _exposure_probability_vector(propensity_to, n, "propensity_to")
    try:
        raw = _estimate_observational_network_exposure(
            y,
            assignments,
            edge_values,
            [int(value) for value in cluster_values],
            _exposure_name(exposure),
            (from_level.own, from_level.neighbors),
            (to_level.own, to_level.neighbors),
            p_from,
            p_to,
            propensity_provenance,
        )
    except (ValueError, CausalEstimateError) as error:
        raise CausalValueError(str(error)) from error
    return ObservationalNetworkExposureEstimate(
        float(raw[0]),
        float(raw[1]),
        float(raw[2]),
        int(raw[3]),
        int(raw[4]),
        int(raw[5]),
        int(raw[6]),
        float(raw[7]),
        float(raw[8]),
        int(raw[9]),
        propensity_provenance,
    )


def _assignment_values(values: Sequence[bool], n: int) -> list[bool]:
    result = list(values)
    if len(result) != n:
        raise CausalValueError("assignment length must match data rows")
    if any(not isinstance(value, (bool, np.bool_)) for value in result):
        raise CausalTypeError("assignment must contain only booleans")
    return [bool(value) for value in result]


def _exposure_probability_vector(values: float | Sequence[float], n: int, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim == 0:
        vector = np.full(n, float(vector), dtype=np.float64)
    if vector.ndim != 1 or len(vector) != n:
        raise CausalValueError(f"{name} must be scalar or have one value per row")
    if not np.isfinite(vector).all() or ((vector <= 0.0) | (vector > 1.0)).any():
        raise CausalValueError(f"{name} values must lie in (0, 1]")
    return vector


def _partition(labels: Sequence[int]) -> frozenset[frozenset[int]]:
    groups: dict[int, set[int]] = {}
    for row, label in enumerate(labels):
        groups.setdefault(int(label), set()).add(row)
    return frozenset(frozenset(rows) for rows in groups.values())


def _same_level(left: tuple[float, float], right: ExposureLevel) -> bool:
    return abs(left[0] - right.own) <= 1e-12 and abs(left[1] - right.neighbors) <= 1e-12


def _observed_exposure(
    assignment: Sequence[bool],
    edges: Sequence[tuple[int, int, float]],
    exposure: object,
) -> list[tuple[float, float]]:
    incoming: list[list[tuple[int, float]]] = [[] for _ in assignment]
    for source, target, weight in edges:
        if source < 0 or target < 0 or source >= len(assignment) or target >= len(assignment):
            raise CausalValueError("network edge index is outside data rows")
        incoming[target].append((source, weight))
    levels = []
    for row, sources in enumerate(incoming):
        own = float(assignment[row])
        if isinstance(exposure, OwnTreatment):
            neighbors = 0.0
        elif isinstance(exposure, NeighborCount):
            neighbors = float(sum(assignment[source] for source, _ in sources))
        elif isinstance(exposure, NeighborFraction):
            neighbors = (
                float(sum(assignment[source] for source, _ in sources) / len(sources))
                if sources
                else 0.0
            )
        elif isinstance(exposure, WeightedNeighborExposure):
            total = sum(weight for _, weight in sources)
            neighbors = (
                sum(weight * assignment[source] for source, weight in sources) / total
                if total
                else 0.0
            )
        else:
            raise CausalTypeError("unsupported exposure mapping")
        levels.append((own, neighbors))
    return levels


__all__ = [
    "BernoulliAssignment",
    "ClusterRandomization",
    "CompleteRandomization",
    "ExposureContrast",
    "ExposureLevel",
    "InterferenceQuery",
    "InterferenceEstimate",
    "NetworkEdge",
    "NeighborCount",
    "NeighborFraction",
    "OwnTreatment",
    "RandomizationContrast",
    "PartialInterference",
    "SaturationDesign",
    "ObservedExposureDesign",
    "SaturationEffectEstimate",
    "SaturationEffects",
    "ObservationalNetworkExposureEstimate",
    "InterferenceSupport",
    "WeightedNeighborExposure",
    "estimate",
    "estimate_saturation_effects",
    "estimate_observational_network_exposure",
]
