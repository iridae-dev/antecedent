"""Python declarations for native frozen modular experiment updates.

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .errors import CausalTypeError, CausalValueError


def acceptance_matrix() -> dict[str, Any]:
    """All advertised design/bridge/decision obligations, including implementation gaps.

    Rows describe required outcomes. They are never substituted for executed
    reference checks or release calibration measurements.
    """
    from . import _native

    return json.loads(_native.joint_panel_experiment_acceptance_matrix())


def quantity(handle: Any, functional: Any) -> Any:
    from .joint_distribution import ScientificQuantity

    return ScientificQuantity._from_wire(
        json.loads(handle.experiment_quantity(json.dumps(functional, allow_nan=False)))
    )


def condition(
    handle: Any,
    *,
    bridge: Any,
    ledger: Any,
    observed_mean: float,
    observation_variance: float,
    sample_size: int,
    mapping_dependency: str,
    intercepts: Sequence[float],
    slopes: Sequence[float],
    model: str,
    experiment: str,
    mapping: str,
    prior_origins: Sequence[str] = (),
    functional: Any = "terminal_mean",
    **execution: Any,
) -> PanelExperiment:
    from .evidence_ledger import EvidenceLedger
    from .scientific_bridge import BridgeContract

    if not isinstance(ledger, EvidenceLedger) or not isinstance(bridge, BridgeContract):
        raise CausalTypeError("condition_experiment needs a BridgeContract and an EvidenceLedger")
    declaration = {
        "bridge": bridge.to_wire(),
        "functional": functional,
        "observed_mean": observed_mean,
        "observation_variance": observation_variance,
        "sample_size": sample_size,
        "mapping_dependency": mapping_dependency,
        "intercepts": list(intercepts),
        "slopes": list(slopes),
        "model": model,
        "experiment": experiment,
        "mapping": mapping,
        "prior_origins": list(prior_origins),
    }
    return PanelExperiment(
        handle.condition_experiment(
            json.dumps(declaration, allow_nan=False), ledger._handle, **execution
        )
    )


def native_quantity(handle: Any, binding: Mapping[str, Any]) -> Any:
    """Bind a scalar projection while preserving the original method-native target kind."""
    from .joint_distribution import ScientificQuantity

    return ScientificQuantity._from_wire(
        json.loads(handle.experiment_quantity(json.dumps(dict(binding), allow_nan=False)))
    )


def condition_native(
    handle: Any, *, binding: Mapping[str, Any], **declaration: Any
) -> PanelExperiment:
    """Execute the target-preserving native experiment owner, with the ordinary bridge inputs."""
    binding_json = json.dumps(dict(binding), allow_nan=False)

    class BoundNativeTarget:
        def condition_experiment(self, payload: str, ledger: Any, **execution: Any) -> Any:
            return handle.condition_native_experiment(binding_json, payload, ledger, **execution)

    return condition(BoundNativeTarget(), **declaration)


@dataclass(frozen=True, slots=True)
class PanelFutureRegime:
    """The setting a future study would run in and how it connects to the retained response.

    ``bridge`` is the licensed :class:`~antecedent.scientific_bridge.BridgeContract` from that
    study's population, window, dose, duration, exposure, recency and spend to the retained
    contrast, with its transport premise, mapping uncertainty and source dependence;
    ``functional`` is the time functional of the retained worlds that the study's estimand
    corresponds to. ``intercepts`` and ``slopes`` are the paired per-draw affine realization of
    its mapping (one per retained draw slot) over the fitted source ``mapping_dependency``, and
    ``mapping_evidence`` the current ledger source of that mapping evidence.
    """

    bridge: Any
    mapping_dependency: str
    intercepts: Sequence[float]
    slopes: Sequence[float]
    mapping_evidence: str
    functional: Any = "terminal_mean"

    def _wire(self) -> dict[str, Any]:
        from .scientific_bridge import BridgeContract

        if not isinstance(self.bridge, BridgeContract):
            raise CausalTypeError("a future regime needs a BridgeContract")
        return {
            "bridge": self.bridge.to_wire(),
            "functional": self.functional,
            "mapping_dependency": self.mapping_dependency,
            "intercepts": list(self.intercepts),
            "slopes": list(self.slopes),
            "mapping_evidence": self.mapping_evidence,
        }


@dataclass(frozen=True, slots=True)
class PanelFutureStudy:
    """A candidate future study: sampling model and cost, and optionally its own regime.

    ``sample_size``, ``observation_variance`` (of one future observation, in the original
    experimental source units) and ``cost_utility`` (already in the decision's utility unit, or
    ``None`` for an uncosted catalog) describe the Gaussian sample. ``regime=None`` is another
    sample of the already observed estimand; a :class:`PanelFutureRegime` measures a different
    functional of the same retained joint response through its own licensed bridge and mapping.
    """

    id: str
    sample_size: int
    observation_variance: float
    cost_utility: float | None = None
    regime: PanelFutureRegime | None = None

    def _wire(self) -> dict[str, Any]:
        if self.regime is not None and not isinstance(self.regime, PanelFutureRegime):
            raise CausalTypeError("regime must be a PanelFutureRegime or None")
        return {
            "id": self.id,
            "sample_size": self.sample_size,
            "observation_variance": self.observation_variance,
            "cost_utility": self.cost_utility,
            "regime": None if self.regime is None else self.regime._wire(),
        }


@dataclass(frozen=True, slots=True)
class PanelExperiment:
    """Conditional posterior retaining aligned response and uncertain mapping slots.

    Every scientific consumption takes the caller's current evidence ledger.
    The update is frozen modular importance conditioning, never a joint refit.
    """

    _handle: Any

    def report(self, ledger: Any) -> dict[str, Any]:
        """Prior/posterior means, complete weights and exact scientific quantities."""
        return json.loads(self._handle.report(ledger._handle))

    def decide(self, contract: Mapping[str, Any] | Any, ledger: Any) -> dict[str, Any]:
        """Evaluate the existing decision contract under current evidence authority."""
        from .decision import Contract

        wire = contract._wire() if isinstance(contract, Contract) else contract
        return json.loads(self._handle.decide(json.dumps(wire, allow_nan=False), ledger._handle))

    def rank_next_samples(
        self,
        *,
        actions: Sequence[Mapping[str, Any]],
        candidates: Sequence[Mapping[str, Any]],
        ledger: Any,
        replicates: int = 199,
        **execution: Any,
    ) -> dict[str, Any]:
        """Rank Gaussian future samples using affine utilities over whole response rows.

        Each action declares id, intercept and coefficients in the report's exact
        quantity order. Every candidate declares id, sample_size,
        observation_variance and cost_utility (or None for an uncosted catalog).
        """
        declaration = {"actions": list(actions), "candidates": list(candidates)}
        return json.loads(
            self._handle.rank_next_samples(
                json.dumps(declaration, allow_nan=False),
                ledger._handle,
                replicates=replicates,
                **execution,
            )
        )

    def rank_next_studies(
        self,
        *,
        actions: Sequence[Mapping[str, Any]],
        studies: Sequence[PanelFutureStudy],
        ledger: Any,
        replicates: int = 199,
        **execution: Any,
    ) -> dict[str, Any]:
        """Rank candidate studies, each possibly in a regime other than the observed one.

        Each study is valued through the actual update it would cause: the retained draws,
        their posterior weights and the terminal responses stay exactly those of this update,
        and only what each draw predicts the study will observe changes with its regime. A
        regime whose bridge is unlicensed, whose mapping disagrees with the fitted source, or
        whose mapping evidence is missing, withdrawn or superseded in ``ledger`` is refused.
        ``actions`` are as for :meth:`rank_next_samples`. Every candidate reports its
        ``regime_identity`` (``None`` for the observed estimand).
        """
        rank = getattr(self._handle, "rank_next_studies", None)
        if rank is None:
            raise CausalValueError(
                "this experiment source cannot rank studies in another regime; use "
                "rank_next_samples for samples of the observed estimand"
            )
        wires = []
        for study in studies:
            if not isinstance(study, PanelFutureStudy):
                raise CausalTypeError("studies must be PanelFutureStudy declarations")
            wires.append(study._wire())
        declaration = {"actions": list(actions), "studies": wires}
        return json.loads(
            rank(
                json.dumps(declaration, allow_nan=False),
                ledger._handle,
                replicates=replicates,
                **execution,
            )
        )

    def export(self, ledger: Any, **execution: Any) -> bytes:
        """Export the complete original frozen panel and declared experiment inputs."""
        return bytes(self._handle.export(ledger._handle, **execution))

    def verify_current(self, artifact: bytes, ledger: Any, **execution: Any) -> None:
        """Bind replay to this update and the caller's current evidence authority."""
        self._handle.verify_current(artifact, ledger._handle, **execution)
