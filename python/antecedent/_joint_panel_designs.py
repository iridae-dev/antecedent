"""Typed design-native joint functionals, with the same scientific target in both modes."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from . import _native


def _json(value: Any) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"))


@dataclass(frozen=True)
class JointDesignResult:
    """An actual design-native law with original proof and aligned uncertainty."""

    _handle: Any

    def report(self) -> dict[str, Any]:
        return json.loads(self._handle.report())

    def experiment_quantity(
        self, functional: Mapping[str, Any] | str = "terminal_mean"
    ) -> dict[str, Any]:
        """Exact assembled contrast quantity for a retained experiment bridge."""
        from ._joint_panel_experiment import quantity

        return quantity(self._handle, functional)

    def condition_experiment(self, **declaration: Any) -> Any:
        """Apply the explicit modular likelihood update to this genuine posterior law."""
        from ._joint_panel_experiment import condition

        return condition(self._handle, **declaration)

    def decompose_reference_components(
        self,
        *,
        exposure: list[int],
        saturation: list[int],
        mechanism: list[int],
        hybrids: list[int],
        order: list[str] | None = None,
        require_identified: bool = True,
        **execution: Any,
    ) -> dict[str, Any]:
        """Check sixteen genuine hybrid worlds and propagate original joint uncertainty."""
        declaration = {
            "exposure": exposure,
            "saturation": saturation,
            "mechanism": mechanism,
            "hybrids": hybrids,
            "order": order or ["exposure", "saturation", "population", "mechanism"],
            "require_identified": require_identified,
        }
        return json.loads(
            self._handle.decompose_reference_components(_json(declaration), **execution)
        )

    def decompose_population_reference(
        self,
        *,
        baseline: int,
        strata: list[float],
        stratum_worlds: list[list[int]],
        before_weights: list[float],
        after_weights: list[float],
        exposure: list[int],
        saturation: list[int],
        mechanism: list[int],
        diagram: Mapping[str, Any],
        order: list[str] | None = None,
        require_identified: bool = True,
        **execution: Any,
    ) -> dict[str, Any]:
        """Transport supported root-covariate strata under a checked selection diagram.

        ``diagram`` declares node_count, directed and bidirected edges, and
        selection_targets. Only the baseline root's distribution may change;
        all remaining response mechanisms are explicitly assumed invariant.
        The returned proof records these premises and every actual source world.
        """
        declaration = {
            "baseline": baseline,
            "strata": strata,
            "stratum_worlds": stratum_worlds,
            "before_weights": before_weights,
            "after_weights": after_weights,
            "exposure": exposure,
            "saturation": saturation,
            "mechanism": mechanism,
            "diagram": dict(diagram),
            "order": order or ["exposure", "saturation", "population", "mechanism"],
            "require_identified": require_identified,
        }
        return json.loads(
            self._handle.decompose_population_reference(_json(declaration), **execution)
        )

    def export_reference_decomposition(
        self, declaration: Mapping[str, Any], **execution: Any
    ) -> bytes:
        """Retain an explicit hybrid component recipe and the original native design."""
        return bytes(
            self._handle.export_reference_decomposition(_json(dict(declaration)), **execution)
        )

    def consume_reference_decomposition_current(
        self, artifact: bytes, declaration: Mapping[str, Any], **execution: Any
    ) -> dict[str, Any]:
        """Recheck a reference artifact against current data, bindings and scope."""
        return json.loads(
            self._handle.consume_reference_decomposition_current(
                artifact, _json(dict(declaration)), **execution
            )
        )

    def export_population_reference(
        self, declaration: Mapping[str, Any], **execution: Any
    ) -> bytes:
        """Retain the complete design, component/population laws and transport diagram."""
        return bytes(
            self._handle.export_population_reference(_json(dict(declaration)), **execution)
        )

    def consume_population_reference_current(
        self, artifact: bytes, declaration: Mapping[str, Any], **execution: Any
    ) -> dict[str, Any]:
        """Compare caller-current data, components, distributions, diagram and scope."""
        return json.loads(
            self._handle.consume_population_reference_current(
                artifact, _json(dict(declaration)), **execution
            )
        )

    def export(self) -> bytes:
        """Retain the entire data, identifying functional and statistical execution."""
        return bytes(self._handle.export())

    def posterior_quantities(self) -> list[dict[str, Any]]:
        """Exact quantity bindings; frequentist covariance is never called a posterior."""
        return json.loads(self._handle.posterior_quantities())


@dataclass(frozen=True)
class PreparedJointDesign:
    """One checked complete-program design, separate from the nonlinear panel fitter."""

    _handle: Any

    def proof(self) -> dict[str, Any]:
        return json.loads(self._handle.proof())

    def execute(
        self, inference: Mapping[str, Any] | str = "frequentist", **execution: Any
    ) -> JointDesignResult:
        declaration = {"inference": inference} if isinstance(inference, str) else inference
        return JointDesignResult(self._handle.execute(_json(declaration), **execution))

    def consume_current(self, artifact: bytes, **execution: Any) -> JointDesignResult:
        """Match current source/design/target before consuming retained evidence."""
        return JointDesignResult(self._handle.consume_current(artifact, **execution))


def prepare_joint_design(
    target: Mapping[str, Any], proof: Mapping[str, Any], **execution: Any
) -> PreparedJointDesign:
    """Compile a ScientificTargetWire and typed version-one joint-design proof.

    Recipes use ``kind=randomized`` or ``kind=longitudinal`` with their actual
    design and unit/episode data. World bindings retain ordered program keys and
    scientific world IDs. Statistical inference is selected only during execute.
    """
    return PreparedJointDesign(
        _native.prepare_joint_design(_json(target), _json(proof), **execution)
    )


def prepare_joint_design_for_panel(
    panel_result: Any, proof: Mapping[str, Any], **execution: Any
) -> PreparedJointDesign:
    """Compile a native design for the exact target of an existing panel result."""
    return PreparedJointDesign(
        _native.prepare_joint_design_for_panel(panel_result._handle, _json(proof), **execution)
    )


def replay_joint_design(artifact: bytes, **execution: Any) -> dict[str, Any]:
    """Historical numerical replay, without caller-current scientific authority."""
    return json.loads(_native.replay_joint_design(artifact, **execution))


@dataclass(frozen=True)
class NativeJointDesignResult:
    """An actual structural/cohort/history/donor target, retaining its native meaning."""

    _handle: Any

    def report(self) -> dict[str, Any]:
        return json.loads(self._handle.report())

    def experiment_quantity(
        self,
        *,
        coefficients: list[float],
        outcome_units: str,
        local_ratio_regularization: dict[str, Any] | None = None,
    ) -> Any:
        """The exact native target projection, retaining structural/cohort/donor meaning."""
        from ._joint_panel_experiment import native_quantity

        return native_quantity(
            self._handle,
            {
                "coefficients": coefficients,
                "outcome_units": outcome_units,
                "local_ratio_regularization": local_ratio_regularization,
            },
        )

    def condition_experiment(
        self,
        *,
        coefficients: list[float],
        outcome_units: str,
        local_ratio_regularization: dict[str, Any] | None = None,
        **declaration: Any,
    ) -> Any:
        """Condition the actual native posterior without relabelling its target."""
        from ._joint_panel_experiment import condition_native

        return condition_native(
            self._handle,
            binding={
                "coefficients": coefficients,
                "outcome_units": outcome_units,
                "local_ratio_regularization": local_ratio_regularization,
            },
            **declaration,
        )

    def export(self) -> bytes:
        return bytes(self._handle.export())


@dataclass(frozen=True)
class PreparedNativeJointDesign:
    """Compiled exact native target; local effects remain local effects."""

    _handle: Any

    def identification(self) -> dict[str, Any]:
        return json.loads(self._handle.identification())

    def execute(
        self, inference: Mapping[str, Any] | str = "frequentist", **execution: Any
    ) -> NativeJointDesignResult:
        declaration = {"inference": inference} if isinstance(inference, str) else inference
        return NativeJointDesignResult(self._handle.execute(_json(declaration), **execution))

    def consume_current(self, artifact: bytes, **execution: Any) -> NativeJointDesignResult:
        return NativeJointDesignResult(self._handle.consume_current(artifact, **execution))


def prepare_native_joint_design(
    recipe: Mapping[str, Any], **execution: Any
) -> PreparedNativeJointDesign:
    """Compile any supported joint design through its actual native method owner.

    ``kind`` selects randomized, switchback, longitudinal, gaussian_longitudinal,
    instrumented, instrumented_local, difference_in_differences or synthetic_control. Source records and all
    design premises are explicit. Native targets preserve population, history,
    cohort and structural-feature meanings instead of converting them to ATEs.
    """
    return PreparedNativeJointDesign(
        _native.prepare_native_joint_design(_json(recipe), **execution)
    )


def replay_native_joint_design(artifact: bytes, **execution: Any) -> dict[str, Any]:
    """Replay frozen native proof and evidence; current use requires consume_current."""
    return json.loads(_native.replay_native_joint_design(artifact, **execution))


def replay_population_reference(artifact: bytes, **execution: Any) -> dict[str, Any]:
    """Independently replay historical population-reference numbers without current authority."""
    return json.loads(_native.replay_population_reference(artifact, **execution))


def replay_reference_decomposition(artifact: bytes, **execution: Any) -> dict[str, Any]:
    """Independently replay historical component numbers without current authority."""
    return json.loads(_native.replay_reference_decomposition(artifact, **execution))
