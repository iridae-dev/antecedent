"""Python declarations for native frozen modular experiment updates.

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .errors import CausalTypeError


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

    def export(self, ledger: Any, **execution: Any) -> bytes:
        """Export the complete original frozen panel and declared experiment inputs."""
        return bytes(self._handle.export(ledger._handle, **execution))

    def verify_current(self, artifact: bytes, ledger: Any, **execution: Any) -> None:
        """Bind replay to this update and the caller's current evidence authority."""
        self._handle.verify_current(artifact, ledger._handle, **execution)
