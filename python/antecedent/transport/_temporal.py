"""One finite two-step temporal transport sequence, exact and point-only.

The whole sequence ``do(A_1 = a_1, A_2 = a_2)`` is one longitudinal intervention
on an explicit two-slice unrolled selection diagram: baseline covariates, the
time-varying covariates observed before each action, one action per step over one
discrete alphabet, and one outcome after the last action. A selection target on a
coordinate is a time-indexed mechanism difference between source and target at
that coordinate's slice; every coordinate without one is assumed invariant at its
slice, and each assumption is reported per slice. Time step 1 and time step 2 are
never transported separately and multiplied. The claim is exact and point-only:
temporal sampling intervals, initial-state uncertainty and new-period refresh are
not licensed, and an interval request refuses.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .._native import (
    consume_temporal_transport_artifact as _consume_temporal_transport_artifact,
)
from .._native import prepare_temporal_transport_stage as _prepare_temporal_transport_stage
from ..errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from ..graph import Admg, Cpdag, Pag
from ._impl import EvidenceCatalog, VariableCoordinate, _non_negative, _optional_non_negative

HORIZON = 2


def _wire(
    coordinates: Sequence[VariableCoordinate],
) -> list[tuple[str, str, int | None, str | None]]:
    return [(c.name, c.domain, c.cardinality, c.unit) for c in coordinates]


@dataclass(frozen=True, slots=True)
class TemporalSequenceSpec:
    """An explicit two-slice unrolling: graph, places in time and mechanism differences.

    ``graph`` is the unrolled ADMG over the named coordinates (directed edges must
    not run backward in time; latent confounding is a bidirected edge). The slots
    place every coordinate: ``baseline`` covariates, the ``covariates`` observed
    before each of the two actions, the two ``actions`` and the ``outcome``.
    ``selections`` names the coordinates whose mechanism may differ between source
    and target, each at its own time slice; an action cannot be one. ``coordinates``
    declares every coordinate's name, finite domain and unit. Both actions share
    one alphabet of at most eight actions, at most twelve coordinates and 4096
    complete covariate histories are admitted, and the horizon is two.
    """

    graph: Admg
    baseline: Sequence[str]
    covariates: Sequence[Sequence[str]]
    actions: Sequence[str]
    outcome: str
    coordinates: Sequence[VariableCoordinate]
    selections: Sequence[str] = ()
    horizon: int = HORIZON

    def __post_init__(self) -> None:
        if isinstance(self.graph, Cpdag | Pag):
            raise CausalUnsupportedError(
                "temporal_transport.invalid_spec: supply the explicit unrolled ADMG; "
                "CPDAG/PAG-native temporal transport is not licensed",
                reason_code="invalid_argument",
            )
        if not isinstance(self.graph, Admg):
            raise CausalTypeError("graph must be an Admg")
        if self.horizon != HORIZON:
            raise CausalUnsupportedError(
                f"temporal_transport.horizon: horizon {self.horizon} is outside the "
                f"licensed horizon of {HORIZON}",
                reason_code="route_not_supported",
            )
        covariates = tuple(tuple(step) for step in self.covariates)
        actions = tuple(self.actions)
        if len(covariates) != HORIZON or len(actions) != HORIZON:
            raise CausalValueError(
                "temporal_transport.invalid_spec: declare the covariates and the action of "
                "each of the two steps",
                reason_code="invalid_argument",
            )
        if any(not isinstance(c, VariableCoordinate) for c in self.coordinates):
            raise CausalTypeError("coordinates must be VariableCoordinate values")
        selections = tuple(self.selections)
        if len(set(selections)) != len(selections):
            raise CausalValueError("temporal selections must be distinct")
        object.__setattr__(self, "baseline", tuple(self.baseline))
        object.__setattr__(self, "covariates", covariates)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "coordinates", tuple(self.coordinates))
        object.__setattr__(self, "selections", selections)


def prepare_temporal_transport_sequence(
    spec: TemporalSequenceSpec,
    *,
    sequence: Sequence[float],
    source: str,
    target: str,
    catalog: EvidenceCatalog,
    laws: Any,
    max_steps: int = 100_000,
    max_depth: int = 256,
    max_operations: int = 10_000_000,
    max_evaluation_depth: int = 256,
    max_support_rows: int = 1_000_000,
    memory_bytes: int | None = None,
    cancel: Any = None,
) -> Any:
    """Decide the whole sequence once, as one longitudinal intervention, and compile it.

    ``sequence`` is one action per step, in order: ``[a, b]`` and ``[b, a]`` are
    different interventions. The returned stage's ``estimate()`` reports the exact
    target response of the outcome, the history/horizon-local support, every
    invariance used per time slice, the time-varying confounders and the evidence
    each slice has; its claim is ``point_only``. ``max_steps`` and ``max_depth``
    are one search budget shared by the identification search, its proof replays
    and the growth of the history lattice, charged with live bytes against
    ``memory_bytes`` and observing ``cancel``; a stop is a receipt
    (``temporal_transport.history_budget``), never a non-identification verdict.
    ``laws`` are supplied exact laws (``ExactTransportData`` or a sequence).
    A complete history outside certified support refuses
    (``temporal_transport.history_outside_support``): the laws must serve every
    source factor the identified proof cites at every history of the lattice
    (reached by the target or not), which the report's ``evidence`` rows mark as
    ``cited_by_derivation``. ``refresh(laws)`` re-estimates same-window evidence
    under the same proof; another window, a longer horizon or a target law that
    moves the fixed initial state needs a new preparation. ``interval()`` always
    refuses.
    """
    if not isinstance(spec, TemporalSequenceSpec):
        raise CausalTypeError("spec must be a TemporalSequenceSpec")
    if not isinstance(catalog, EvidenceCatalog):
        raise CausalTypeError("catalog must be an EvidenceCatalog")
    return _prepare_temporal_transport_stage(
        spec.graph,
        list(spec.selections),
        (
            list(spec.baseline),
            [list(step) for step in spec.covariates],
            list(spec.actions),
            spec.outcome,
        ),
        _wire(spec.coordinates),
        spec.horizon,
        [float(a) for a in sequence],
        source,
        target,
        catalog,
        laws,
        max_steps=_non_negative("max_steps", max_steps),
        max_depth=_non_negative("max_depth", max_depth),
        max_operations=_non_negative("max_operations", max_operations),
        max_evaluation_depth=_non_negative("max_evaluation_depth", max_evaluation_depth),
        max_support_rows=_non_negative("max_support_rows", max_support_rows),
        memory_bytes=_optional_non_negative("memory_bytes", memory_bytes),
        cancel=cancel,
    )


def consume_temporal_transport_artifact(
    artifact: bytes,
    *,
    max_steps: int = 100_000,
    max_depth: int = 256,
    max_operations: int = 10_000_000,
    max_evaluation_depth: int = 256,
    max_support_rows: int = 1_000_000,
    max_laws: int = 256,
    memory_bytes: int | None = None,
    cancel: Any = None,
) -> str:
    """Independently replay an exported two-step sequence report.

    The sequence is re-decided under the producer's recorded limits (refused if
    they exceed the consumer's), recompiled and re-evaluated, and the report must
    match exactly. The horizon, the ordered sequence and the variable names the
    report is read with are bound into the artifact's identity.
    """
    if not isinstance(artifact, bytes):
        raise CausalTypeError("artifact must be bytes")
    return _consume_temporal_transport_artifact(
        artifact,
        max_steps=_non_negative("max_steps", max_steps),
        max_depth=_non_negative("max_depth", max_depth),
        max_operations=_non_negative("max_operations", max_operations),
        max_evaluation_depth=_non_negative("max_evaluation_depth", max_evaluation_depth),
        max_support_rows=_non_negative("max_support_rows", max_support_rows),
        max_laws=_non_negative("max_laws", max_laws),
        memory_bytes=_optional_non_negative("memory_bytes", memory_bytes),
        cancel=cancel,
    )


__all__ = [
    "TemporalSequenceSpec",
    "consume_temporal_transport_artifact",
    "prepare_temporal_transport_sequence",
]
