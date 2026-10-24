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
)
from ._joint_panel_experiment import PanelExperiment
from ._joint_panel_experiment import acceptance_matrix as experiment_acceptance_matrix
from .errors import CausalValueError
from .intervention_programs import Expr

__all__ = [
    "PanelProgramCertificate",
    "PanelExperiment",
    "experiment_acceptance_matrix",
    "JointDesignResult",
    "NativeJointDesignResult",
    "PreparedJointDesign",
    "PreparedNativeJointDesign",
    "prepare_joint_design",
    "prepare_joint_design_for_panel",
    "prepare_native_joint_design",
    "replay_joint_design",
    "replay_native_joint_design",
    "replay_population_reference",
    "replay_reference_decomposition",
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

    def export(self, **execution: Any) -> bytes:
        """Export the frozen response, control bindings and replayable bound proof."""
        return self._handle.export(**execution)

    def verify_current(self, artifact: bytes, **execution: Any) -> None:
        """Replay and verify against this exact fitted response and search domain."""
        self._handle.verify_current(artifact, **execution)


@dataclass(frozen=True, slots=True)
class RandomizedProgramSelection:
    """Selected complete program evaluated on reserved original experimental units."""

    _handle: Any

    def report(self) -> dict[str, Any]:
        """Return held-out value/error, original split and whole-procedure diagnostics."""
        return json.loads(self._handle.report())

    def export(self) -> bytes:
        """Export original observations, procedure and independently replayable results."""
        return self._handle.export()

    def verify_current(self, artifact: bytes, **execution: Any) -> None:
        """Replay against this exact current observational source and policy."""
        self._handle.verify_current(artifact, **execution)


@dataclass(frozen=True, slots=True)
class PreparedRandomizedProgramSelection:
    """A complete experimental selection procedure with a genuine observational split."""

    _handle: Any

    def execute(
        self, *, repetitions: int = 199, tie_tolerance: float = 0.0, **execution: Any
    ) -> RandomizedProgramSelection:
        """Repeat generation, fitting, feasibility, search, selection and held-out evaluation.

        Resampling repeats the actual native estimators on whole original units;
        failed slots, ties, instability and training optimism remain visible.
        The selected value's uncertainty uses independent held-out observations.
        Bootstrap diagnostics grant no ordinary-bootstrap coverage claim at ties.
        """
        return RandomizedProgramSelection(
            self._handle.execute(repetitions=repetitions, tie_tolerance=tie_tolerance, **execution)
        )

    def simultaneous_intervals(
        self, *, repetitions: int = 199, level: float = 0.95, **execution: Any
    ) -> dict[str, Any]:
        """Joint bootstrap max-statistic intervals over the fixed feasible program set.

        Each complete resample refits all candidates together. Data-dependent
        screening is refused here and must use the held-out execution instead.
        Statistical coverage requires the declared independent-unit bootstrap model.
        """
        return json.loads(
            self._handle.simultaneous_intervals(repetitions=repetitions, level=level, **execution)
        )


def prepare_randomized_selection(
    *,
    design: Mapping[str, Any],
    training: Sequence[Mapping[str, Any]],
    validation: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
    **execution: Any,
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
        _native.prepare_randomized_program_selection(
            _json(
                {
                    "design": dict(design),
                    "training": list(training),
                    "validation": list(validation),
                    "policy": dict(policy),
                }
            ),
            **execution,
        )
    )


def consume_program_certificate(artifact: bytes, **execution: Any) -> dict[str, Any]:
    """Independently reconstruct the frozen response and repeat its numerical bound proof."""
    return json.loads(_native.consume_panel_program_certificate(artifact, **execution))


def consume_randomized_selection(artifact: bytes, **execution: Any) -> dict[str, Any]:
    """Replay the original observational selection procedure without current authorization."""
    return json.loads(_native.consume_randomized_program_selection(artifact, **execution))


@dataclass(frozen=True, slots=True)
class InstrumentedProgramSelection:
    """Continuous structural-dose selection evaluated on reserved original observations."""

    _handle: Any

    def report(self) -> dict[str, Any]:
        """Selected doses, training search bounds, held-out value and resampling diagnostics."""
        return json.loads(self._handle.report())

    def export(self) -> bytes:
        """Retain original data and the complete procedure for independent replay."""
        return self._handle.export()

    def verify_current(self, artifact: bytes, **execution: Any) -> None:
        """Reject changed data, structural premises, domain or algorithm."""
        self._handle.verify_current(artifact, **execution)


def select_instrumented_program(
    declaration: Mapping[str, Any],
    *,
    repetitions: int = 199,
    tie_tolerance: float = 0.0,
    **execution: Any,
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
        _native.select_instrumented_program(
            _json(dict(declaration)),
            repetitions=repetitions,
            tie_tolerance=tie_tolerance,
            **execution,
        )
    )


def consume_instrumented_selection(artifact: bytes, **execution: Any) -> dict[str, Any]:
    """Replay the complete structural selection procedure without current authorization."""
    return json.loads(_native.consume_instrumented_selection(artifact, **execution))


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

    def export(self, **execution: Any) -> bytes:
        """Encode the actual frozen compiler product; performs no second fit."""
        return bytes(self._handle.export(**execution))

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
        **execution: Any,
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
            self._handle.certify_program(_json(declaration), **execution)
        )

    def decompose_reference(
        self, declaration: Mapping[str, Any], **execution: Any
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
        return json.loads(self._handle.decompose_reference(_json(declaration), **execution))

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
        **execution: Any,
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
        return json.loads(self._handle.search_program(_json(declaration), **execution))


@dataclass(frozen=True, slots=True)
class PreparedPanel:
    """Native checked preparation with synchronized reuse of compatible fits."""

    _handle: Any

    def inspect(self) -> dict[str, Any]:
        """Inspect source/design obligations without fitting or simulation."""
        return json.loads(self._handle.inspect())

    def estimate(self, **execution: Any) -> PanelResult:
        """Run the retained Rust program with caller seed/thread/memory/cancel policy."""
        return PanelResult(self._handle.estimate(**execution))

    def retarget(self, worlds: Sequence[Mapping[str, Any]], **execution: Any) -> PreparedPanel:
        """Check replacement worlds against the same source and reuse compatible fits."""
        return PreparedPanel(self._handle.retarget(_json(list(worlds)), **execution))


def prepare(
    groups: Mapping[int, Mapping[str, Sequence[float | None]]],
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
    **execution: Any,
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

    ``seed``, ``threads``, ``max_bytes`` and ``cancel`` are caller execution policy.
    Neither ``exogenous_context`` nor a finite fit establishes a causal license.
    """
    names = list(dict.fromkeys([*channels, outcome, confounder, season]))
    source = []
    for group, columns in groups.items():
        if set(columns) != set(names):
            raise CausalValueError("each group must supply exactly the declared source columns")
        source.append({"id": group, "columns": [list(columns[name]) for name in names]})
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
    }
    selected = _family_declaration(groups, family)
    if selected is not None:
        declaration["family"] = selected
    return PreparedPanel(_native.prepare_joint_panel(_json(declaration), **execution))


def consume(artifact: bytes, **execution: Any) -> dict[str, Any]:
    """Replay the frozen Rust program in an independent historical consumer."""
    return json.loads(_native.consume_joint_panel(artifact, **execution))
