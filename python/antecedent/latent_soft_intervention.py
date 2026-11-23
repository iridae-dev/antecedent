"""Compiled, model-conditional discovery of latent additive soft interventions."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from . import _native
from ._artifact_protocol import ArtifactAcceptance, ArtifactExpectation, ExportedArtifact
from ._scientific import ScientificResult
from .errors import CausalValueError
from .execution import Execution, native_controls, resolve_execution

__all__ = [
    "FittedLatentSoftIntervention",
    "GaussianComponent",
    "LatentDiscovery",
    "LatentForecast",
    "LatentForecastRefit",
    "LatentRefits",
    "PreparedLatentSoftIntervention",
    "ReplayedLatentForecast",
    "TargetHypothesis",
    "prepare",
    "replay_forecast",
]

_KIND = "latent_soft_intervention_forecast_v1"

if TYPE_CHECKING:
    from .joint_law_propagation import PropagationClaim
    from .outcomes import Outcome


@dataclass(frozen=True, slots=True)
class TargetHypothesis:
    """A candidate pair of shift masks with its descriptive penalized fit gap."""

    target_sets: tuple[tuple[str, ...], tuple[str, ...]]
    training_gap: float


@dataclass(frozen=True, slots=True)
class LatentDiscovery:
    """The retained fitting evidence and additive shifts in source variable order."""

    program: str
    variables: tuple[str, ...]
    evidence: str
    release_supported: bool
    would_refuse: str | None
    log_evidence: float | None
    forecast_admissible: bool
    hypotheses: tuple[TargetHypothesis, ...]
    shifts: tuple[tuple[float, ...], ...]
    training_rows: int
    evaluation_rows: int


@dataclass(frozen=True, slots=True)
class GaussianComponent:
    """One Gaussian joint law in the forecast's coordinate order."""

    weight: float
    mean: tuple[float, ...]
    covariance: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class LatentForecast:
    """The model-conditional outcome mixture and baseline at a supplied context."""

    query: str
    variables: tuple[str, ...]
    baseline: GaussianComponent
    action: tuple[GaussianComponent, ...]
    mean_difference: tuple[float, ...]
    refits: LatentRefits | None

    def action_marginal(
        self, variable: str, *, execution: Execution | None = None
    ) -> Outcome[PropagationClaim]:
        """Propagate one action marginal conditional on the fitted law and weights."""
        from .joint_law_propagation import MixtureComponent, mixture

        try:
            index = self.variables.index(variable)
        except ValueError as error:
            raise CausalValueError(f"unknown forecast coordinate {variable!r}") from error
        diagonal = index * (len(self.variables) + 1)
        if any(index >= len(law.mean) or diagonal >= len(law.covariance) for law in self.action):
            raise CausalValueError("forecast moments are missing")
        return mixture(
            [
                MixtureComponent(
                    f"{self.query}:{index}:{version}",
                    self.query,
                    law.weight,
                    law.mean[index],
                    law.covariance[diagonal],
                )
                for version, law in enumerate(self.action)
            ],
            execution=execution,
        )


@dataclass(frozen=True, slots=True)
class LatentForecastRefit:
    """Aligned baseline and action from one complete fit; version labels are draw-local."""

    baseline: GaussianComponent
    action: tuple[GaussianComponent, ...]
    mean_difference: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class LatentRefits:
    """Diagnostic resampling spread across complete refits, without interval calibration."""

    attempted: int
    unavailable: int
    draws: tuple[LatentForecastRefit, ...]


@dataclass(frozen=True, slots=True)
class ReplayedLatentForecast(ScientificResult):
    """Discovery and forecast verified against a separately held identity."""

    discovery: LatentDiscovery
    forecast: LatentForecast
    acceptance: ArtifactAcceptance
    _artifact_bytes: bytes = field(repr=False)


def _discovery(wire: dict[str, Any]) -> LatentDiscovery:
    variables = tuple(wire["variables"])
    return LatentDiscovery(
        wire["program"],
        variables,
        wire["evidence"],
        wire["release_supported"],
        wire["would_refuse"],
        wire["log_evidence"],
        wire["forecast_admissible"],
        tuple(
            TargetHypothesis(
                (
                    tuple(variables[index] for index in item["target_sets"][0]),
                    tuple(variables[index] for index in item["target_sets"][1]),
                ),
                item["training_gap"],
            )
            for item in wire["hypotheses"]
        ),
        tuple(tuple(shift) for shift in wire["shifts"]),
        wire["training_rows"],
        wire["evaluation_rows"],
    )


def _component(wire: dict[str, Any]) -> GaussianComponent:
    return GaussianComponent(wire["weight"], tuple(wire["mean"]), tuple(wire["covariance"]))


def _forecast(wire: dict[str, Any]) -> LatentForecast:
    refits = wire["refits"]
    return LatentForecast(
        wire["query"],
        tuple(wire["variables"]),
        _component(wire["baseline"]),
        tuple(_component(item) for item in wire["action"]),
        tuple(wire["mean_difference"]),
        None
        if refits is None
        else LatentRefits(
            refits["attempted"],
            refits["unavailable"],
            tuple(
                LatentForecastRefit(
                    _component(draw["baseline"]),
                    tuple(_component(item) for item in draw["action"]),
                    tuple(draw["mean_difference"]),
                )
                for draw in refits["draws"]
            ),
        ),
    )


@dataclass(frozen=True, slots=True)
class PreparedLatentSoftIntervention:
    """Retained native compiler plan, inspectable before any fitting."""

    _native: Any = field(repr=False)
    context: tuple[str, ...]

    def inspect(self) -> dict[str, Any]:
        """Read the native compiler's plan view."""
        return json.loads(self._native.inspect())

    def unit_support(self) -> dict[str, Any] | None:
        """The support decision of the unit this plan's route runs under: ``unit``, ``status``,
        ``supported`` and ``missing_evidence``. ``None`` when no unit is bound to the route."""
        wire = self._native.unit_support()
        return None if wire is None else json.loads(wire)

    def execute(self, *, execution: Execution | None = None) -> FittedLatentSoftIntervention:
        """Execute the retained plan and keep its fitted mechanisms for forecast requests."""
        native = self._native.execute(
            **native_controls(execution, memory="memory_bytes", cancel="cancel")
        )
        return FittedLatentSoftIntervention(native, self.context)


@dataclass(frozen=True, slots=True)
class FittedLatentSoftIntervention:
    """Native fit owned by the lowered compiler program."""

    _native: Any = field(repr=False)
    context: tuple[str, ...]

    @property
    def discovery(self) -> LatentDiscovery:
        """Read the evidence for a split and its target hypotheses."""
        return _discovery(json.loads(self._native.discovery()))

    def _values(self, context: Mapping[str, float]) -> list[float]:
        if set(context) != set(self.context):
            raise CausalValueError("provide exactly the declared pre-intervention context")
        return [context[name] for name in self.context]

    def forecast(
        self, context: Mapping[str, float], *, execution: Execution | None = None
    ) -> LatentForecast:
        """Evaluate the recorded action and baseline at declared pre-intervention coordinates."""
        result = self._native.forecast(
            self._values(context),
            **native_controls(execution, memory="memory_bytes", cancel="cancel"),
        )
        return _forecast(json.loads(result))

    def export(
        self, context: Mapping[str, float], *, execution: Execution | None = None
    ) -> ExportedArtifact[ReplayedLatentForecast]:
        """Export the source observations and a query for native replay."""
        data, digest = self._native.export_forecast(
            self._values(context),
            **native_controls(execution, memory="memory_bytes", cancel="cancel"),
        )
        return ExportedArtifact(data, _KIND, ArtifactExpectation(_KIND, digest))


def prepare(
    *,
    variables: Sequence[str],
    edges: Sequence[tuple[str, str]],
    context: Sequence[str],
    candidates: Sequence[str],
    baseline: Sequence[Sequence[float]],
    action: Sequence[Sequence[float]],
    population: str,
    randomized: bool,
    version_exogenous_given_context: bool,
    refits: int = 0,
    search_operations: int = 41_000,
    execution: Execution | None = None,
) -> PreparedLatentSoftIntervention:
    """Compile a randomized recorded-action comparison on a declared continuous DAG.

    Rows use the variable order. Context is declared as pre-intervention DAG roots.
    Candidate targets name equations whose intercepts can shift. The assignment and
    exogeneity declarations are premises of the model. Preparation does no fitting.
    Set `refits` to a positive integer up to 32 for optional split-preserving joint
    resampling diagnostics. The native plan retains these fits across forecasts.
    ``execution.seed`` seeds the outcome-independent split, the numerical restarts and
    the refits, and enters the program identity.
    """
    execution = resolve_execution(execution)
    names = tuple(variables)
    context_names = tuple(context)
    index = {name: i for i, name in enumerate(names)}
    try:
        native_edges = [(index[parent], index[child]) for parent, child in edges]
        native_context = [index[name] for name in context_names]
        native_candidates = [index[name] for name in candidates]
    except KeyError as error:
        raise CausalValueError(f"unknown DAG variable {error.args[0]!r}") from error
    payload = json.dumps(
        {
            "variables": names,
            "edges": native_edges,
            "context": native_context,
            "candidates": native_candidates,
            "baseline": [list(row) for row in baseline],
            "action": [list(row) for row in action],
            "population": population,
            "randomized": randomized,
            "version_exogenous_given_context": version_exogenous_given_context,
            "seed": execution.seed,
            "refits": refits,
            "search_operations": search_operations,
        },
        separators=(",", ":"),
        allow_nan=False,
    )
    native = _native.latent_soft_intervention_prepare(
        payload, **native_controls(execution, memory="memory_bytes", cancel="cancel")
    )
    return PreparedLatentSoftIntervention(native, context_names)


def replay_forecast(
    data: bytes, *, expected_identity: str, execution: Execution | None = None
) -> ReplayedLatentForecast:
    """Recompile and execute the source and verify the reproduced forecast."""
    result = _native.latent_soft_intervention_replay(
        data,
        expected_identity,
        **native_controls(execution, memory="memory_bytes", cancel="cancel"),
    )
    wire = json.loads(result)
    return ReplayedLatentForecast(
        _discovery(wire["discovery"]),
        _forecast(wire["forecast"]),
        ArtifactAcceptance("replayed", True, True, False),
        data,
    )
