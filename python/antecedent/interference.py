"""Randomization designs and exposure mappings for interference queries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field
from typing import Any, Literal

import numpy as np

from ._data import as_columns
from ._native import estimate_network_interference as _estimate_network_interference
from .errors import CausalTypeError, CausalValueError


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
    """A randomized exposure contrast on a fixed unit network.

    ``assignment`` is the known randomization design, ``exposure`` the exposure
    mapping and ``functional`` the contrast between two exposure levels.

    :func:`antecedent.analyze` estimates the licensed cell (explicit ``Dag``,
    Frequentist, validation ``none``): NeighborCount exposure under Bernoulli
    assignment, Horvitz–Thompson / Hájek with the conservative Young variance
    bound. It needs the keyword-only design facts: ``network``, the fixed
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
    realized_assignment: Sequence[bool] | None = None
    kind: Literal["interference"] = field(default="interference", init=False, repr=False)

    def __post_init__(self) -> None:
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
class InterferenceEstimate:
    contrast: RandomizationContrast
    from_probability_method: str
    to_probability_method: str
    minimum_exposure_probability: float
    provenance: Mapping[str, Any] = field(
        default_factory=lambda: {"operation_ids": ["stats.randomized_interference"]}
    )


def _assignment_args(design: object) -> dict[str, Any]:
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
    realized_assignment=))`` is the licensed, study-retaining path.
    """

    if not isinstance(query, InterferenceQuery):
        raise CausalTypeError("query must be an InterferenceQuery")
    names, columns = as_columns(data)
    try:
        outcome_index = names.index(query.functional.outcome)
    except ValueError as error:
        raise CausalValueError(
            f"outcome column {query.functional.outcome!r} is missing from data"
        ) from error
    edge_values = _edge_values(edges)
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
    )


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
    "WeightedNeighborExposure",
    "estimate",
]
