"""Retained multichannel panel execution through the Rust causal compiler.

Features use :mod:`antecedent.intervention_programs` expressions. Their variable
reads explicitly mean lag-weighted, Michaelis-Menten saturated channel values.
Python declares inputs; Rust checks the source/history/target, fits the model,
retains shared draws, checks support, and evaluates every world.

A fitted curve does not identify an intervention. Missing design premises remain
visible in ``inspect()`` and every result. Historical artifacts replay numerical
results without granting current scientific authority.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from . import _native
from ._artifact_protocol import (
    ExportedArtifact,
    artifact_export,
    artifact_from_receipt,
    container_bytes,
)
from ._joint_panel_designs import (
    JointDesignResult,
    NativeJointDesignResult,
    PreparedJointDesign,
    PreparedNativeJointDesign,
    prepare_joint_design,
    prepare_joint_design_for_panel,
    prepare_native_joint_design,
    replay_joint_design,
    replay_native_joint_design,
    replay_population_reference,
    replay_reference_decomposition,
    target_contrast,
)
from ._joint_panel_experiment import (
    HistoricalJointExperiment,
    HistoricalPanelExperiment,
    PanelExperiment,
    PanelFutureRegime,
    PanelFutureStudy,
    PanelJointComponent,
    PanelJointExperiment,
    replay_design_experiment,
    replay_joint_experiment,
    replay_native_experiment,
)
from ._joint_panel_experiment import acceptance_matrix as experiment_acceptance_matrix
from ._joint_panel_imputation import PanelImputationResult, PreparedPanelImputation
from ._panel_execution import run_panel
from .errors import CausalTypeError, CausalUnsupportedError, CausalValueError
from .execution import Execution
from .intervention_programs import Expr
from .missing import ImputationSpec
from .outcomes import outcome_operation, require_success

__all__ = [
    "JointResponseLaw",
    "PreparedJointResponse",
    "compile_joint_response",
    "HistoricalJointExperiment",
    "HistoricalPanelExperiment",
    "PanelJointComponent",
    "PanelJointExperiment",
    "replay_joint_experiment",
    "replay_design_experiment",
    "replay_native_experiment",
    "PanelProgramCertificate",
    "PanelExperiment",
    "PanelFutureRegime",
    "PanelFutureStudy",
    "experiment_acceptance_matrix",
    "JointDesignResult",
    "NativeJointDesignResult",
    "PreparedJointDesign",
    "PreparedNativeJointDesign",
    "prepare_joint_design",
    "prepare_joint_design_for_panel",
    "prepare_native_joint_design",
    "target_contrast",
    "replay_joint_design",
    "replay_native_joint_design",
    "replay_population_reference",
    "replay_reference_decomposition",
    "PanelCategoricalChannel",
    "PanelDummyChannel",
    "DummyCodedPanel",
    "dummy_code_panel",
    "PanelImputationResult",
    "PreparedPanelImputation",
    "prepare_imputed",
    "PanelResult",
    "PreparedPanel",
    "PreparedRandomizedProgramSelection",
    "RandomizedProgramSelection",
    "InstrumentedProgramSelection",
    "select_instrumented_program",
    "consume_instrumented_selection",
    "consume_program_certificate",
    "consume",
    "consume_randomized_selection",
    "prepare",
    "prepare_randomized_selection",
]


def _json(value: Any) -> str:
    try:
        return json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise CausalValueError(f"invalid joint panel declaration: {error}") from error


_ADSTOCKS = ("simplex", "geometric")
_SATURATIONS = ("michaelis_menten", "hill")
_SPATIAL_KERNELS = ("exponential", "gaussian")
_FAMILY_KEYS = frozenset(
    {"adstock", "saturation", "saturation_classes", "spatial_distances", "spatial_kernel"}
)


def _family_declaration(
    groups: Mapping[int, Any], family: Mapping[str, Any] | None
) -> dict[str, Any] | None:
    """Translate a family choice keyed by group ID into the Rust, group-ordered form.

    Alignment is by group identity: ``saturation_classes`` maps every group ID to a class
    ``0..m`` (or is ``"per_group"``), and ``spatial_distances`` maps every ordered pair of
    group IDs to a symmetric nonnegative distance. Omitted choices are the reviewed family:
    free three-period simplex adstock, one Michaelis-Menten half-saturation shared by every
    group and independent Gaussian group intercepts.
    """
    if family is None:
        return None
    unknown = set(family) - _FAMILY_KEYS
    if unknown:
        raise CausalValueError(f"unknown panel family choices: {sorted(unknown)}")
    ids = list(groups)
    adstock = family.get("adstock", "simplex")
    saturation = family.get("saturation", "michaelis_menten")
    if adstock not in _ADSTOCKS or saturation not in _SATURATIONS:
        raise CausalValueError(
            f"adstock must be one of {_ADSTOCKS} and saturation one of {_SATURATIONS}"
        )
    classes = family.get("saturation_classes")
    if classes is None:
        scope: dict[str, Any] = {"kind": "shared"}
    elif classes == "per_group":
        scope = {"kind": "classes", "assignment": list(range(len(ids)))}
    elif isinstance(classes, Mapping):
        if set(classes) != set(ids):
            raise CausalValueError("saturation_classes must assign exactly the declared groups")
        scope = {"kind": "classes", "assignment": [int(classes[group]) for group in ids]}
    else:
        raise CausalValueError("saturation_classes must be None, 'per_group' or a group mapping")
    spatial = None
    distances = family.get("spatial_distances")
    if distances is not None:
        kernel = family.get("spatial_kernel", "exponential")
        if kernel not in _SPATIAL_KERNELS:
            raise CausalValueError(f"spatial_kernel must be one of {_SPATIAL_KERNELS}")
        if set(distances) != set(ids) or any(set(distances[group]) != set(ids) for group in ids):
            raise CausalValueError("spatial_distances must cover every ordered pair of groups")
        spatial = {
            "kernel": kernel,
            "groups": [str(group) for group in ids],
            "distances": [float(distances[row][column]) for row in ids for column in ids],
        }
    elif "spatial_kernel" in family:
        raise CausalValueError("spatial_kernel requires spatial_distances")
    return {"adstock": adstock, "saturation": saturation, "scope": scope, "spatial": spatial}


@dataclass(frozen=True, slots=True)
class PanelProgramCertificate:
    """Global numerical bounds derived from an actual retained fitted response."""

    _handle: Any

    def report(self) -> dict[str, Any]:
        """Return feasible controls, global gap, shape proofs and fitted source identity.

        A bound certifies the declared fitted scalar objective. Statistical
        selected-value reliability requires its separate procedure validation.
        """
        return json.loads(self._handle.report())

    @artifact_export
    def export(self, *, execution: Execution | None = None) -> ExportedArtifact:
        """Export the frozen response, control bindings and replayable bound proof."""
        return artifact_from_receipt(self, run_panel(execution, self._handle.export))

    def verify_current(self, artifact: bytes, *, execution: Execution | None = None) -> None:
        """Replay and verify against this exact fitted response and search domain."""
        artifact = container_bytes(artifact)
        run_panel(execution, self._handle.verify_current, artifact)


@dataclass(frozen=True, slots=True)
class RandomizedProgramSelection:
    """Selected complete program evaluated on reserved original experimental units."""

    _handle: Any

    def report(self) -> dict[str, Any]:
        """Return held-out value/error, original split and whole-procedure diagnostics."""
        return json.loads(self._handle.report())

    @artifact_export
    def export(self) -> ExportedArtifact:
        """Export original observations, procedure and independently replayable results."""
        return artifact_from_receipt(self, self._handle.export())

    def verify_current(self, artifact: bytes, *, execution: Execution | None = None) -> None:
        """Replay against this exact current observational source and policy."""
        artifact = container_bytes(artifact)
        run_panel(execution, self._handle.verify_current, artifact)


@dataclass(frozen=True, slots=True)
class PreparedRandomizedProgramSelection:
    """A complete experimental selection procedure with a genuine observational split."""

    _handle: Any

    def execute(
        self,
        *,
        repetitions: int = 199,
        tie_tolerance: float = 0.0,
        execution: Execution | None = None,
    ) -> RandomizedProgramSelection:
        """Repeat generation, fitting, feasibility, search, selection and held-out evaluation.

        Resampling repeats the actual native estimators on whole original units;
        failed slots, ties, instability and training optimism remain visible.
        The selected value's uncertainty uses independent held-out observations.
        Bootstrap diagnostics grant no ordinary-bootstrap coverage claim at ties.
        """
        return RandomizedProgramSelection(
            run_panel(
                execution,
                self._handle.execute,
                repetitions=repetitions,
                tie_tolerance=tie_tolerance,
            )
        )

    def simultaneous_intervals(
        self, *, repetitions: int = 199, level: float = 0.95, execution: Execution | None = None
    ) -> dict[str, Any]:
        """Joint bootstrap max-statistic intervals over the fixed feasible program set.

        Each complete resample refits all candidates together. Data-dependent
        screening is refused here and must use the held-out execution instead.
        Statistical coverage requires the declared independent-unit bootstrap model.
        """
        return json.loads(
            run_panel(
                execution, self._handle.simultaneous_intervals, repetitions=repetitions, level=level
            )
        )


def prepare_randomized_selection(
    *,
    design: Mapping[str, Any],
    training: Sequence[Mapping[str, Any]],
    validation: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
    execution: Execution | None = None,
) -> PreparedRandomizedProgramSelection:
    """Compile complete-program selection with known randomization and reserved data.

    Design declares ``worlds``, ``sampling_unit``, ``population``, ``consistency``
    and ``between_unit_isolation``. Each unit declares original ``id``, assigned
    ``arm``, all known ``probabilities``, ``outcome`` and fixed preassignment
    ``population_weight``. Training and validation must have disjoint original IDs.

    Policy declares ``independent_sampling_frame``, ``utility_units``,
    ``benefit_multiplier``, per-world ``utility_costs`` and ``resource_costs``, a
    hard ``budget`` and ``generation`` (``kind: all`` or ``kind: training_screen``
    with ``top_k``). Common sampling and interference premises remain explicit;
    known random assignment alone does not establish population transport.
    """
    return PreparedRandomizedProgramSelection(
        run_panel(
            execution,
            _native.prepare_randomized_program_selection,
            _json(
                {
                    "design": dict(design),
                    "training": list(training),
                    "validation": list(validation),
                    "policy": dict(policy),
                }
            ),
        )
    )


def consume_program_certificate(
    artifact: bytes, *, execution: Execution | None = None
) -> dict[str, Any]:
    """Independently reconstruct the frozen response and repeat its numerical bound proof."""
    artifact = container_bytes(artifact)
    return json.loads(run_panel(execution, _native.consume_panel_program_certificate, artifact))


def consume_randomized_selection(
    artifact: bytes, *, execution: Execution | None = None
) -> dict[str, Any]:
    """Replay the original observational selection procedure without current authorization."""
    artifact = container_bytes(artifact)
    return json.loads(run_panel(execution, _native.consume_randomized_program_selection, artifact))


@dataclass(frozen=True, slots=True)
class InstrumentedProgramSelection:
    """Continuous structural-dose selection evaluated on reserved original observations."""

    _handle: Any

    def report(self) -> dict[str, Any]:
        """Selected doses, training search bounds, held-out value and resampling diagnostics."""
        return json.loads(self._handle.report())

    @artifact_export
    def export(self) -> ExportedArtifact:
        """Retain original data and the complete procedure for independent replay."""
        return artifact_from_receipt(self, self._handle.export())

    def verify_current(self, artifact: bytes, *, execution: Execution | None = None) -> None:
        """Reject changed data, structural premises, domain or algorithm."""
        artifact = container_bytes(artifact)
        run_panel(execution, self._handle.verify_current, artifact)


def select_instrumented_program(
    declaration: Mapping[str, Any],
    *,
    repetitions: int = 199,
    tie_tolerance: float = 0.0,
    execution: Execution | None = None,
) -> InstrumentedProgramSelection:
    """Fit joint structural features, search raw doses, and evaluate on held-out units.

    The declaration supplies the actual instrumented ``design``, ``features``
    binding (``raw_feature_indices`` and monomial ``powers``), original disjoint
    ``training`` and ``validation`` units, common ``independent_sampling_frame``,
    bounded ``controls``, linear ``restrictions``, optional ``budget``, explicit
    ``reference`` and ``initial`` programs, ``benefit_multiplier``, ``utility_units``,
    ``generation`` and search ``certificate`` options. Every unit supplies ``id``,
    ``endogenous``, ``instruments``, ``exogenous`` and ``outcome``.

    The target is population improvement under the declared homogeneous structural
    moment model. Held-out joint IV covariance is asymptotic and requires sufficient
    instrument strength. Search bounds concern fitted training utility; bootstrap
    diagnostics do not establish coverage at ties or grant a calibration license.
    """
    return InstrumentedProgramSelection(
        run_panel(
            execution,
            _native.select_instrumented_program,
            _json(dict(declaration)),
            repetitions=repetitions,
            tie_tolerance=tie_tolerance,
        )
    )


def consume_instrumented_selection(
    artifact: bytes, *, execution: Execution | None = None
) -> dict[str, Any]:
    """Replay the complete structural selection procedure without current authorization."""
    artifact = container_bytes(artifact)
    return json.loads(run_panel(execution, _native.consume_instrumented_selection, artifact))


@dataclass(frozen=True, slots=True)
class PanelResult:
    """One native shared-draw execution, retaining its compiled scientific scope."""

    _handle: Any

    def experiment_quantity(self, functional: Mapping[str, Any] | str = "terminal_mean") -> Any:
        """Return the exact compiled response quantity for an experimental bridge."""
        from ._joint_panel_experiment import quantity

        return quantity(self._handle, functional)

    def condition_experiment(self, **declaration: Any) -> PanelExperiment:
        """Condition this retained posterior using explicit bridge, mapping and evidence."""
        from ._joint_panel_experiment import condition

        return condition(self._handle, **declaration)

    def condition_excursion(
        self, *, evidence: Any, component: str, likelihood_assumption_id: str, **declaration: Any
    ) -> PanelExperiment:
        """Condition on a retained excursion through an explicit model-conditional bridge."""
        from ._joint_panel_experiment import condition_excursion

        return condition_excursion(
            self._handle,
            evidence=evidence,
            component=component,
            likelihood_assumption_id=likelihood_assumption_id,
            **declaration,
        )

    def condition_joint_excursion(self, **declaration: Any) -> Any:
        """Condition jointly on several components of one retained excursion.

        Takes ``evidence``, ``components`` (``PanelJointComponent``), ``likelihood_assumption_id``,
        ``ledger``, and the ``model``/``experiment``/``mapping`` origins. The observed contrasts
        and their covariance come from the retained evidence.
        """
        from ._joint_panel_experiment import condition_joint_excursion

        return condition_joint_excursion(self._handle, **declaration)

    @property
    def execution_facts(self) -> Any:
        """Engine-owned resource/operation measurements for this execution."""
        return self._handle.execution_facts

    def report(self) -> dict[str, Any]:
        """Actual means, failed draw slots, diagnostics, support and fit count."""
        return json.loads(self._handle.report())

    def scheduled_contrast(
        self, functional: Mapping[str, Any] | str = "terminal_mean", *, level: float = 0.95
    ) -> dict[str, Any]:
        """The target's assembled contrast of scheduled programs on the shared draws.

        ``functional`` is ``"terminal_mean"`` or a one-key mapping such as
        ``{"window_average": {"from": 0, "until": 3}}``. Every world is evaluated on the
        same parameter draw, so the interval carries the cross-world covariance; it
        is withheld when a slot failed or the fit does not authorize a complete law.
        The value is conditional on the fitted mechanism and grants no causal license.
        """
        return json.loads(self._handle.scheduled_contrast(_json(functional), level))

    @artifact_export
    def export(self, *, execution: Execution | None = None) -> ExportedArtifact:
        """Encode the actual frozen compiler product; performs no second fit."""
        return artifact_from_receipt(self, bytes(run_panel(execution, self._handle.export)))

    def certify_program(
        self,
        *,
        base_world: int,
        controls: Sequence[Mapping[str, Any]],
        initial: Sequence[float],
        budget: float | None = None,
        restrictions: Sequence[Mapping[str, Any]] = (),
        functional: Mapping[str, Any] | str = "terminal_mean",
        utility_offset: float = 0.0,
        utility_scale: float = 1.0,
        absolute_gap: float = 1e-4,
        max_splits: int = 10_000,
        execution: Execution | None = None,
    ) -> PanelProgramCertificate:
        """Optimize a compiler-derived fitted response with rigorous global bounds.

        Declare every channel/period control using ``channel``, ``offset``,
        ``lower``, ``upper`` and ``unit_cost``. Fix a coordinate explicitly by
        setting equal bounds. The response expression is derived in Rust from
        the retained lag, saturation, interaction and context mechanisms.
        ``SplitLimit`` retains a valid bound without claiming the requested gap.
        """
        declaration = {
            "base_world": base_world,
            "controls": list(controls),
            "initial": list(initial),
            "budget": budget,
            "restrictions": list(restrictions),
            "functional": functional,
            "utility_offset": utility_offset,
            "utility_scale": utility_scale,
            "options": {"absolute_gap": absolute_gap, "max_splits": max_splits},
        }
        return PanelProgramCertificate(
            run_panel(execution, self._handle.certify_program, _json(declaration))
        )

    def decompose_reference(
        self, declaration: Mapping[str, Any], *, execution: Execution | None = None
    ) -> dict[str, Any]:
        """Decompose a response change on the retained joint mechanism law.

        Declare ``before_world``/``after_world`` using compiled numeric world IDs,
        population weights in source group order, explicit ``time_weights``, and
        an ``order`` permutation of exposure, saturation, population, mechanism.
        ``periods`` maps before/after response periods to parameter indices in
        the same fitted path (all zero for an invariant fit).

        Returns all 16 hybrids, exact Shapley averages over 24 orders, sequential
        contributions and paired uncertainty slots. ``order_range`` describes
        order sensitivity and is not a confidence interval. This model-conditional
        reference decomposition does not establish unique causal attribution.
        """
        return json.loads(
            run_panel(execution, self._handle.decompose_reference, _json(declaration))
        )

    def search_program(
        self,
        *,
        base_world: int,
        controls: Sequence[Mapping[str, Any]],
        initial: Sequence[float],
        budget: float | None = None,
        restrictions: Sequence[Mapping[str, Any]] = (),
        functional: Mapping[str, Any] | str = "terminal_mean",
        utility_offset: float = 0.0,
        utility_scale: float = 1.0,
        absolute_gap: float = 1e-4,
        max_splits: int = 10_000,
        max_evaluations: int = 512,
        max_iterations: int = 128,
        initial_step: float = 0.5,
        step_tolerance: float = 1e-6,
        objective_tolerance: float = 0.0,
        uncertainty: str = "point_means",
        execution: Execution | None = None,
    ) -> dict[str, Any]:
        """Execute certified search when supported, otherwise bounded best-found search.

        Both routes evaluate this actual frozen fitted response and preserve its
        scientific scope. The best-found route retains its certificate refusal
        and stopping reason; an unavailable global gap remains absent.
        ``uncertainty="aligned_law"`` uses the complete original joint law and
        returns the selected action's actual law separately. Posterior meaning
        remains distinct from frequentist selected-value coverage.
        """
        declaration = {
            "uncertainty": uncertainty,
            "certificate": {
                "base_world": base_world,
                "controls": list(controls),
                "initial": list(initial),
                "budget": budget,
                "restrictions": list(restrictions),
                "functional": functional,
                "utility_offset": utility_offset,
                "utility_scale": utility_scale,
                "options": {"absolute_gap": absolute_gap, "max_splits": max_splits},
            },
            "poll_options": {
                "max_evaluations": max_evaluations,
                "max_iterations": max_iterations,
                "initial_step": initial_step,
                "step_tolerance": step_tolerance,
                "objective_tolerance": objective_tolerance,
            },
        }
        return json.loads(run_panel(execution, self._handle.search_program, _json(declaration)))

    def attempt_explanation(
        self,
        *,
        explanation: str,
        world: int,
        reference_world: int,
        before: tuple[int, int],
        after: tuple[int, int],
        execution: Execution | None = None,
    ) -> dict[str, Any]:
        """Execute the coarser total response contrast a finer explanation falls back to.

        ``explanation`` is the finer change being asked about (``"exposure"``,
        ``"saturation"``, ``"composition"`` or ``"mechanism"``). The executed contrast is the
        change of ``world``'s mean response from the inclusive ``before`` periods to the
        inclusive ``after`` periods, minus the same change for the flagged ``reference_world``,
        on the already fitted law. Unique attribution of the finer explanation stays refused
        (``unique_attribution_refusal``) and ``grants_unique_causal_attribution`` is false.
        """
        declaration = {
            "explanation": explanation,
            "world": world,
            "reference_world": reference_world,
            "before": list(before),
            "after": list(after),
        }
        return json.loads(
            run_panel(execution, self._handle.attempt_explanation, _json(declaration))
        )

    def attempt_identified_reference(
        self,
        declaration: Mapping[str, Any],
        *,
        fallback: Mapping[str, Any],
        execution: Execution | None = None,
    ) -> dict[str, Any]:
        """The identified reference decomposition, or the supplied coarser contrast.

        ``declaration`` is as for :meth:`decompose_reference` (``require_identified_reference``
        is not applied: an unidentified decomposition is executed as its ``fallback``, a mapping
        with the :meth:`attempt_explanation` arguments). ``route`` is ``"supported"`` when every
        hybrid is identified and ``"coarser"`` otherwise, with the conditional ``requested``
        numbers and the per-hybrid missing premises retained next to the ``executed`` contrast.
        """
        payload = {"reference": dict(declaration), "fallback": dict(fallback)}
        return json.loads(
            run_panel(execution, self._handle.attempt_identified_reference, _json(payload))
        )

    def attempt_design_adaptive_reference(
        self,
        declaration: Mapping[str, Any],
        *,
        fallback: Mapping[str, Any] | None = None,
        execution: Execution | None = None,
    ) -> dict[str, Any]:
        """The finest decomposition this design identifies, and the designs that earn the rest.

        Every component step whose two hybrids are both identified is executed on the aligned
        law; components with none are listed as ``unidentified_components`` and each missing
        premise is paired with the design that would earn it in ``next_designs``. When the
        full decomposition is not identified and ``fallback`` is supplied, the coarser total
        contrast is executed too (``coarser``).
        """
        payload = {
            "reference": dict(declaration),
            "fallback": None if fallback is None else dict(fallback),
        }
        return json.loads(
            run_panel(execution, self._handle.attempt_design_adaptive_reference, _json(payload))
        )

    def search_declared_programs(
        self,
        *,
        controls: Sequence[Mapping[str, Any]],
        budget: float | None = None,
        restrictions: Sequence[Mapping[str, Any]] = (),
        functional: Mapping[str, Any] | str = "terminal_mean",
        utility_offset: float = 0.0,
        utility_scale: float = 1.0,
        uncertainty: str = "point_means",
        max_candidates: int = 1024,
        execution: Execution | None = None,
    ) -> dict[str, Any]:
        """Exhaustively rank this panel's compiled worlds as candidate programs.

        Every control (``channel``, ``offset``, ``lower``, ``upper``, ``unit_cost``) binds one
        scheduled channel level; the declared bounds, linear ``budget`` and hard
        ``restrictions`` (``id``, ``coefficients``, ``bound``) exclude a world with a concrete
        reason (``rows[i]["feasibility"]``) before any value is attempted. The objective is the
        expected utility ``utility_offset + utility_scale * response`` of the fitted
        ``functional``. Exhaustion is over the finite set of compiled worlds and is not a
        continuous optimum: ``continuous_gap`` is absent and ``global_certificate`` is false.
        The standing is model-conditional and ``selected_value_reliability`` is unmeasured.
        """
        declaration = {
            "controls": list(controls),
            "budget": budget,
            "restrictions": list(restrictions),
            "functional": functional,
            "utility_offset": utility_offset,
            "utility_scale": utility_scale,
            "uncertainty": uncertainty,
            "max_candidates": max_candidates,
        }
        return json.loads(
            run_panel(execution, self._handle.search_declared_programs, _json(declaration))
        )


def _graph_node(node: Sequence[Any]) -> dict[str, Any]:
    """An unrolled-graph node: ``("context", i)``, ``("channel", c, t)`` or ``("outcome", t)``."""
    parts = tuple(node) if isinstance(node, Sequence) and not isinstance(node, str) else ()
    kind, rest = (parts[0], parts[1:]) if parts else (None, ())
    if kind == "context" and len(rest) == 1:
        return {"context": rest[0]}
    if kind == "channel" and len(rest) == 2:
        return {"channel": {"channel": rest[0], "period": rest[1]}}
    if kind == "outcome" and len(rest) == 1:
        return {"outcome": rest[0]}
    raise CausalValueError(
        "an unrolled graph node is ('context', i), ('channel', c, t) or ('outcome', t)",
        reason_code="invalid_argument",
    )


@dataclass(frozen=True, slots=True)
class PreparedPanel:
    """Native checked preparation with synchronized reuse of compatible fits."""

    _handle: Any

    def inspect(self) -> dict[str, Any]:
        """Inspect source/design obligations without fitting or simulation."""
        return json.loads(self._handle.inspect())

    def estimate(self, *, execution: Execution | None = None) -> PanelResult:
        """Run the retained Rust program under the caller's ``execution``."""
        return PanelResult(run_panel(execution, self._handle.estimate))

    def retarget(
        self, worlds: Sequence[Mapping[str, Any]], *, execution: Execution | None = None
    ) -> PreparedPanel:
        """Check replacement worlds against the same source and reuse compatible fits."""
        return PreparedPanel(run_panel(execution, self._handle.retarget, _json(list(worlds))))

    def with_unrolled_graph(
        self,
        *,
        periods: int,
        channels: int,
        outcome_periods: Sequence[int],
        contexts: int = 0,
        carryover_reach: int = 0,
        outcome_persistence: bool = False,
        channel_persistence: bool = False,
        history_feedback: bool = False,
        latent_confounding: Sequence[tuple[Sequence[Any], Sequence[Any]]] = (),
        execution: Execution | None = None,
    ) -> PreparedPanel:
        """Run general ID on a declared unrolled graph of this panel's program.

        ``periods`` and ``channels`` must equal the program's shape. Channel ``(c, t)``
        reaches the outcomes ``t .. t + carryover_reach``; ``outcome_persistence`` makes the
        previous outcome a parent of the next, ``history_feedback`` makes the assignment
        respond to it, and ``latent_confounding`` lists unmeasured common causes as pairs of
        nodes: ``("context", i)``, ``("channel", c, t)`` or ``("outcome", t)``. The verdict is
        for ``outcome_periods``.

        An identified verdict is a nonparametric statement about the observed law. It
        identifies the fitted panel only when ``prepare(..., gaussian_restriction=True)``
        declared that the fitted mechanism realizes it: then ``inspect()`` shows a separate
        ``joint_adjustment.gaussian_restriction`` attempt, ``identified`` becomes true, and
        every estimate re-checks residual normality and linearity on the fitted mechanism,
        withdrawing the claim if they flag it. A hedge, a graph of another shape, a
        restriction declared with a time-varying mechanism, history feedback, carryover that
        differs from the fitted lag order or serial dependence that does not match the AR(1)
        errors are refused. Nothing is identified without the declaration, and the claim never
        states nonparametric identification.
        """
        declaration = {
            "periods": periods,
            "channels": channels,
            "contexts": contexts,
            "carryover_reach": carryover_reach,
            "outcome_persistence": outcome_persistence,
            "channel_persistence": channel_persistence,
            "history_feedback": history_feedback,
            "latent_confounding": [
                [_graph_node(left), _graph_node(right)] for left, right in latent_confounding
            ],
            "outcome_periods": list(outcome_periods),
        }
        return PreparedPanel(
            run_panel(execution, self._handle.with_unrolled_graph, _json(declaration))
        )


@dataclass(frozen=True, slots=True)
class PanelCategoricalChannel:
    """A categorical assignment channel with its complete declared level set.

    The channel's source column holds one category label (or ``None`` for a missing cell)
    per row. ``levels`` must equal the set of labels the units actually assign, and
    ``reference`` (one of them) is coded as every indicator zero. Rust checks the level
    set, the reference and that every level is assigned in some unit; each failure is a
    refusal that names its cause, and nothing is recoded silently.
    """

    channel: str
    levels: tuple[str, ...]
    reference: str

    def __post_init__(self) -> None:
        levels = self.levels
        if isinstance(levels, str | bytes) or not isinstance(levels, Sequence):
            raise CausalTypeError(
                "categorical levels must be a sequence of labels", reason_code="invalid_argument"
            )
        levels = tuple(levels)
        if not (
            isinstance(self.channel, str)
            and isinstance(self.reference, str)
            and all(isinstance(level, str) for level in levels)
        ):
            raise CausalTypeError(
                "categorical channel, levels and reference must be strings",
                reason_code="invalid_argument",
            )
        object.__setattr__(self, "levels", levels)

    def _declaration(self) -> dict[str, Any]:
        return {"channel": self.channel, "levels": list(self.levels), "reference": self.reference}


@dataclass(frozen=True, slots=True)
class PanelDummyChannel:
    """One 0/1 indicator channel produced for a non-reference level."""

    name: str
    source: str
    level: str


@dataclass(frozen=True, slots=True)
class DummyCodedPanel:
    """A panel whose categorical channels are replaced by their indicator channels.

    ``groups`` has the shape :func:`prepare` takes (``None`` marks a missing cell), and
    ``channels`` lists the expanded channels in order: untouched channels, and for each
    categorical channel one ``"<channel>=<level>"`` indicator per non-reference level.
    Programs and worlds name the indicator channels.
    """

    groups: dict[int, dict[str, tuple[float | None, ...]]]
    channels: tuple[str, ...]
    outcome: str
    confounder: str
    season: str
    indicators: tuple[PanelDummyChannel, ...]


def _categorical_declarations(
    categorical: Sequence[PanelCategoricalChannel],
) -> list[dict[str, Any]]:
    if isinstance(categorical, str | bytes | Mapping) or not all(
        isinstance(declared, PanelCategoricalChannel) for declared in categorical
    ):
        raise CausalTypeError(
            "categorical must be a sequence of PanelCategoricalChannel",
            reason_code="invalid_argument",
        )
    return [declared._declaration() for declared in categorical]


def _source(
    groups: Mapping[int, Mapping[str, Sequence[Any]]],
    channels: Sequence[str],
    outcome: str,
    confounder: str,
    season: str,
) -> tuple[list[str], list[dict[str, Any]]]:
    """The declared source column names and each group's columns, in that order."""
    names = list(dict.fromkeys([*channels, outcome, confounder, season]))
    source = []
    for group, columns in groups.items():
        if set(columns) != set(names):
            raise CausalValueError("each group must supply exactly the declared source columns")
        source.append({"id": group, "columns": [list(columns[name]) for name in names]})
    return names, source


def dummy_code_panel(
    groups: Mapping[int, Mapping[str, Sequence[Any]]],
    *,
    channels: Sequence[str],
    outcome: str,
    confounder: str,
    season: str,
    interval_ns: int,
    categorical: Sequence[PanelCategoricalChannel],
    execution: Execution | None = None,
) -> DummyCodedPanel:
    """Rewrite declared categorical channels as 0/1 indicator channels, fitting nothing.

    A categorical assignment has no dose: its levels are alternatives. Each declared
    channel (labels in ``groups``) becomes one indicator per non-reference level, so a
    fitted response differs between levels by exactly the indicator coefficients. A cell
    with no label stays missing in every indicator of its channel. The same coding runs
    inside :func:`prepare` when ``categorical`` is passed there.

    The channel's domain is the set of labels the units assign. Refused, by name: a channel
    not among ``channels``, a level set that differs from that domain (so a declared level no
    unit assigns, and an assigned label nobody declared), a reference outside the level set,
    fewer than two levels, a channel declared twice, a channel with no label at all, and
    numbers where labels are expected or labels in an undeclared channel.
    """
    names, source = _source(groups, channels, outcome, confounder, season)
    declaration = {
        "names": names,
        "channels": list(channels),
        "outcome": outcome,
        "confounder": confounder,
        "season": season,
        "interval_ns": interval_ns,
        "groups": source,
        "categorical": _categorical_declarations(categorical),
    }
    raw = json.loads(run_panel(execution, _native.dummy_code_joint_panel, _json(declaration)))
    return DummyCodedPanel(
        groups={
            unit["id"]: {
                name: tuple(column)
                for name, column in zip(raw["names"], unit["columns"], strict=True)
            }
            for unit in raw["groups"]
        },
        channels=tuple(raw["channels"]),
        outcome=raw["outcome"],
        confounder=raw["confounder"],
        season=raw["season"],
        indicators=tuple(PanelDummyChannel(**indicator) for indicator in raw["indicators"]),
    )


def _declaration(
    groups: Mapping[int, Mapping[str, Sequence[Any]]],
    *,
    channels: Sequence[str],
    outcome: str,
    confounder: str,
    season: str,
    seasons: int,
    interval_ns: int,
    worlds: Sequence[Mapping[str, Any]],
    interactions: Mapping[str, Expr],
    options: Mapping[str, Any],
    lag_order: int,
    temporal: Mapping[str, Any] | None,
    functional: Mapping[str, Any] | None,
    assignment: Mapping[str, Any] | None,
    exogenous_context: bool,
    family: Mapping[str, Any] | None,
    categorical: Sequence[PanelCategoricalChannel],
    gaussian_restriction: bool,
) -> dict[str, Any]:
    """The one native declaration of a panel, shared by plain and imputed preparation."""
    names, source = _source(groups, channels, outcome, confounder, season)
    features = []
    for name, expression in interactions.items():
        if not isinstance(expression, Expr):
            raise CausalValueError("panel features require intervention_programs.Expr")
        features.append({"id": name, "expression": expression.node})
    declaration = {
        "names": names,
        "channels": list(channels),
        "outcome": outcome,
        "confounder": confounder,
        "season": season,
        "seasons": seasons,
        "lag_order": lag_order,
        "interval_ns": interval_ns,
        "groups": source,
        "worlds": list(worlds),
        "interactions": features,
        "options": dict(options),
        "temporal": temporal,
        "functional": functional,
        "assignment": assignment,
        "exogenous_context": exogenous_context,
        "gaussian_restriction": gaussian_restriction,
        "categorical": _categorical_declarations(categorical),
    }
    selected = _family_declaration(groups, family)
    if selected is not None:
        declaration["family"] = selected
    return declaration


def prepare(
    groups: Mapping[int, Mapping[str, Sequence[Any]]],
    *,
    channels: Sequence[str],
    outcome: str,
    confounder: str,
    season: str,
    seasons: int,
    interval_ns: int,
    worlds: Sequence[Mapping[str, Any]],
    interactions: Mapping[str, Expr],
    options: Mapping[str, Any],
    lag_order: int = 2,
    temporal: Mapping[str, Any] | None = None,
    functional: Mapping[str, Any] | None = None,
    assignment: Mapping[str, Any] | None = None,
    exogenous_context: bool = False,
    family: Mapping[str, Any] | None = None,
    categorical: Sequence[PanelCategoricalChannel] = (),
    gaussian_restriction: bool = False,
    execution: Execution | None = None,
) -> PreparedPanel:
    """Compile an explicitly declared panel through ``Study.prepare_joint_panel``.

    ``groups`` maps integer group IDs to complete named source columns. With lag
    order two, its first two rows are actual exposure prehistory; outcome and
    confounder may be ``None`` in those rows. Remaining rows must be complete.

    Every world supplies ``id``, ``reference``, keyed ``channels`` schedules and
    ``scenarios`` with ``group``, keyed two-value ``history``, ``confounder`` and
    ``season`` future arrays. No future context is extrapolated implicitly.

    ``options`` is the Rust ``PanelFitOptions`` declaration (optimizer, inference,
    initial); ``temporal`` is the same Rust change/smoothing specification.
    ``functional`` optionally supplies finite ``terms`` (world-index, coefficient)
    and a ``reference`` index. ``assignment`` uses the existing randomized-effect
    wire contract, checked against whole source-group programs in Rust.

    ``family`` optionally selects a broader mechanism family than the reviewed one
    (three-period simplex adstock, one shared Michaelis-Menten half-saturation,
    independent group intercepts): ``adstock`` is ``"simplex"`` or ``"geometric"``
    (one decay per channel over the observable window), ``saturation`` is
    ``"michaelis_menten"`` or ``"hill"`` (a free positive shape per channel),
    ``saturation_classes`` is ``None``, ``"per_group"`` or a map from every group ID
    to a class ``0..m`` with its own half-saturation, and ``spatial_distances`` maps
    every ordered pair of group IDs to a distance so group intercepts are spatially
    correlated (``spatial_kernel`` is ``"exponential"`` or ``"gaussian"``; the range
    is fitted). Priors for the extra coordinates go in
    ``options["inference"]["prior"]["family"]``. A family changes the fitted
    mechanism, never the identification requirements of the target.

    ``categorical`` declares :class:`PanelCategoricalChannel` assignment channels whose
    source columns hold category labels; each is dummy-coded as :func:`dummy_code_panel`
    describes, and worlds, scenario histories and interaction features name the
    ``"<channel>=<level>"`` indicator channels (``inspect()["indicators"]`` lists them).

    ``gaussian_restriction`` declares, on the target, that the conditional laws are the
    panel mechanism's own (a saturated lag-weighted linear mean with Gaussian group
    intercepts and stationary AR(1) errors independent of the program given the context).
    This is a user assumption, not a test. It identifies nothing by itself: it is
    consumed by :meth:`PreparedPanel.with_unrolled_graph`, which still requires general ID
    to identify the joint intervention on the declared unrolled graph.

    ``execution`` owns the seed, threads, memory limit and cancellation.
    Neither ``exogenous_context`` nor a finite fit establishes a causal license.
    """
    declaration = _declaration(
        groups,
        channels=channels,
        outcome=outcome,
        confounder=confounder,
        season=season,
        seasons=seasons,
        interval_ns=interval_ns,
        worlds=worlds,
        interactions=interactions,
        options=options,
        lag_order=lag_order,
        temporal=temporal,
        functional=functional,
        assignment=assignment,
        exogenous_context=exogenous_context,
        family=family,
        categorical=categorical,
        gaussian_restriction=gaussian_restriction,
    )
    return PreparedPanel(run_panel(execution, _native.prepare_joint_panel, _json(declaration)))


def prepare_imputed(
    groups: Mapping[int, Mapping[str, Sequence[Any]]],
    *,
    imputation: ImputationSpec,
    channels: Sequence[str],
    outcome: str,
    confounder: str,
    season: str,
    seasons: int,
    interval_ns: int,
    worlds: Sequence[Mapping[str, Any]],
    interactions: Mapping[str, Expr],
    options: Mapping[str, Any],
    lag_order: int = 2,
    functional: Mapping[str, Any] | None = None,
    assignment: Mapping[str, Any] | None = None,
    exogenous_context: bool = False,
    family: Mapping[str, Any] | None = None,
    categorical: Sequence[PanelCategoricalChannel] = (),
    gaussian_restriction: bool = False,
    execution: Execution | None = None,
) -> PreparedPanelImputation:
    """Prepare multiple imputation of a panel's missing outcomes and confounders.

    The panel is declared as for :func:`prepare` (a ``temporal`` mechanism is refused: its
    selection procedure is not carried through imputation). Each completed panel is
    prepared and fitted exactly as an observed one, and the ``m`` response vectors are
    pooled: Rubin's rules over frequentist fits (``options["inference"]`` needs at least
    two draws, the within-imputation uncertainty), or the stacked posterior of Bayesian
    fits. ``imputation`` is a :class:`antecedent.missing.ImputationSpec`; its ``domains``
    name the imputation model's own columns (``"outcome"`` and ``"confounder"``).

    Missing assignments and season levels are not imputed (an assignment is the
    intervention itself, a season level a design label), and a panel with an analysis mask
    is refused. Serial context is one neighbouring step of the stacked follow-up rows, so
    the first and last follow-up row of a unit read a row of the adjacent unit; the fit's
    own serial law is unchanged. Pooling grants no causal license, and a missingness
    mechanism other than missing-at-random is refused with its next attempt named.
    """
    if not isinstance(imputation, ImputationSpec):
        raise CausalTypeError(
            "imputation must be an antecedent.missing.ImputationSpec",
            reason_code="invalid_argument",
        )
    panel = _declaration(
        groups,
        channels=channels,
        outcome=outcome,
        confounder=confounder,
        season=season,
        seasons=seasons,
        interval_ns=interval_ns,
        worlds=worlds,
        interactions=interactions,
        options=options,
        lag_order=lag_order,
        temporal=None,
        functional=functional,
        assignment=assignment,
        exogenous_context=exogenous_context,
        family=family,
        categorical=categorical,
        gaussian_restriction=gaussian_restriction,
    )
    request = {
        "panel": panel,
        "imputation": {
            "model": imputation.model,
            "imputations": imputation.imputations,
            "iterations": imputation.iterations,
            "seed": imputation.seed,
            "assumption": imputation.assumption,
            "bootstrap_replicates": imputation.bootstrap_replicates,
            "domains": [list(domain) for domain in imputation.domains],
        },
    }
    return PreparedPanelImputation(
        run_panel(execution, _native.prepare_joint_panel_imputed, _json(request))
    )


def consume(artifact: bytes, *, execution: Execution | None = None) -> dict[str, Any]:
    """Replay the frozen Rust program in an independent historical consumer."""
    artifact = container_bytes(artifact)
    return json.loads(run_panel(execution, _native.consume_joint_panel, artifact))


@dataclass(frozen=True, slots=True)
class JointResponseLaw:
    """Joint means of every compiled world on one shared fit, with their full covariance.

    ``covariance`` (``uncertainty == "sampling_covariance"``) or the aligned posterior draws
    (``"aligned_posterior"``) carry the cross-world dependence; marginal intervals are never
    subtracted. ``identified_claim`` is false when conditional joint-world support is not
    established, and ``calibration`` is ``"unmeasured"``: a numerical result grants neither
    causal nor calibration license.
    """

    world_ids: tuple[int, ...]
    means: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    contrast: tuple[float, float] | None
    uncertainty: str
    draw_count: int
    failed_draws: int
    identified_claim: bool
    identification: str
    target: str
    source_digest: str
    calibration: str
    _handle: Any

    @classmethod
    def _from_handle(cls, handle: Any) -> JointResponseLaw:
        raw = json.loads(handle.report())
        contrast = raw["contrast"]
        return cls(
            world_ids=tuple(raw["world_ids"]),
            means=tuple(raw["means"]),
            covariance=tuple(tuple(row) for row in raw["covariance"]),
            contrast=None if contrast is None else (contrast["value"], contrast["se"]),
            uncertainty=raw["uncertainty"],
            draw_count=raw["draw_count"],
            failed_draws=raw["failed_draws"],
            identified_claim=raw["identified_claim"],
            identification=raw["identification"],
            target=raw["target"],
            source_digest=raw["source_digest"],
            calibration=raw["calibration"],
            _handle=handle,
        )

    @artifact_export
    def export(self, *, execution: Execution | None = None) -> ExportedArtifact:
        """Exact compiled inputs and the joint output, for independent replay."""
        return artifact_from_receipt(self, bytes(run_panel(execution, self._handle.export)))


@dataclass(frozen=True, slots=True)
class PreparedJointResponse:
    """Immutable worlds lowered from a prepared, identified static response.

    All worlds share one design and one fit. :meth:`retarget` recompiles only the world
    projections; a changed data, prior or kernel dependency needs a new compilation.
    """

    _handle: Any

    def inspect(self) -> dict[str, Any]:
        """Worlds, contrast coefficients and identification, without fitting anything."""
        return dict(json.loads(self._handle.inspect()))

    @outcome_operation
    def execute(self, *, execution: Execution | None = None) -> JointResponseLaw:
        """Fit once if necessary and jointly evaluate every world."""
        return JointResponseLaw._from_handle(run_panel(execution, self._handle.execute))

    @outcome_operation
    def retarget(
        self,
        *,
        outcome: str,
        worlds: Sequence[Mapping[str, Any]],
        contrast: Mapping[str, Any] | None = None,
        units: str = "unspecified",
        execution: Execution | None = None,
    ) -> PreparedJointResponse:
        """New world projections over the same source, prior and fit."""
        declaration = _response_declaration(outcome, worlds, contrast, units)
        return PreparedJointResponse(
            run_panel(execution, self._handle.retarget, _json(declaration))
        )

    @outcome_operation
    def consume_current(
        self, artifact: bytes, *, execution: Execution | None = None
    ) -> JointResponseLaw:
        """Replay an exported result and bind it to this live compiled program."""
        return JointResponseLaw._from_handle(
            run_panel(execution, self._handle.consume_current, container_bytes(artifact))
        )


def _response_declaration(
    outcome: str,
    worlds: Sequence[Mapping[str, Any]],
    contrast: Mapping[str, Any] | None,
    units: str,
) -> dict[str, Any]:
    if isinstance(worlds, str | bytes) or not all(isinstance(w, Mapping) for w in worlds):
        raise CausalTypeError("worlds must be mappings with id, levels and reference")
    return {
        "outcome": outcome,
        "units": units,
        "worlds": [
            {
                "id": w["id"],
                "reference": bool(w.get("reference", False)),
                "levels": dict(w["levels"]),
            }
            for w in worlds
        ],
        "contrast": None if contrast is None else dict(contrast),
    }


@outcome_operation
def compile_joint_response(
    study: Any,
    *,
    outcome: str,
    worlds: Sequence[Mapping[str, Any]],
    contrast: Mapping[str, Any] | None = None,
    units: str = "unspecified",
) -> PreparedJointResponse:
    """Lower worlds of an identified, prepared static response onto one shared Gaussian fit.

    ``study`` is a frequentist or Bayesian prepared static intervention-response study whose
    adjustment proof is retained. Each world is ``{"id": n, "levels": {name: level},
    "reference": bool}`` assigning exactly the study's intervention coordinates inside their
    observed range; ``contrast`` is ``{"kind": "difference", "plus": a, "minus": b,
    "reference": b}`` or ``{"kind": "linear", "terms": [[world, coefficient], ...],
    "reference": r}``. ``units`` must equal the outcome's schema units (``"unspecified"``
    when absent). Compilation fits nothing; execution fits once. The target keeps the
    Gaussian-linear, fixed-empirical-covariate premise, and worlds whose conditional joint
    support is not established stay model-conditional (``identified_claim`` false).
    """
    from .prepared import PreparedAnalysis

    if not isinstance(study, PreparedAnalysis):
        raise CausalTypeError("compile_joint_response requires a prepared study")
    if study._refused is not None:
        require_success(study._refused)
    native = getattr(study._engine, "_native", None)
    if not isinstance(native, _native.PreparedAnalysis):
        raise CausalUnsupportedError(
            "joint response compilation needs a prepared static response with a retained proof",
            reason_code="route_not_supported",
        )
    declaration = _response_declaration(outcome, worlds, contrast, units)
    handle = study.execution._call(
        _native.compile_joint_response,
        native,
        _json(declaration),
        seed=study.execution.seed,
        threads=study.execution.threads,
    )
    return PreparedJointResponse(handle)
