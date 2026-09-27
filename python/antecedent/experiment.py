"""First-class randomized experiment designs and intention-to-treat queries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field
from typing import Literal

import numpy as np

from ._native import estimate_ancova_effect as _estimate_ancova_effect
from ._native import estimate_complier_effect as _estimate_complier_effect
from ._native import estimate_cuped_effect as _estimate_cuped_effect
from ._native import estimate_multi_arm_effects as _estimate_multi_arm_effects
from ._native import estimate_stratified_effect as _estimate_stratified_effect
from ._native import estimate_switchback_effect as _estimate_switchback_effect
from ._native import exact_randomization_test as _exact_randomization_test
from .errors import CausalUnsupportedError, CausalValueError
from .interference import (
    BernoulliAssignment,
    ClusterRandomization,
    CompleteRandomization,
    ExposureContrast,
    ExposureLevel,
    InterferenceQuery,
    NeighborCount,
)
from .interference import (
    estimate as estimate_interference,
)


@dataclass(frozen=True, slots=True)
class StratifiedRandomization:
    """Complete randomization independently within named blocks."""

    treated_per_block: Mapping[str, int]

    def __post_init__(self) -> None:
        counts = dict(self.treated_per_block)
        if not counts or any(not isinstance(key, str) or not key.strip() for key in counts):
            raise CausalValueError("treated_per_block must map non-empty block names to counts")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 2 for value in counts.values()):
            raise CausalValueError("each block needs at least two treated units for variance estimation")
        object.__setattr__(self, "treated_per_block", counts)


@dataclass(frozen=True, slots=True)
class FactorialRandomization:
    """Fixed four-cell randomization of two binary factors.

    Cell counts are ordered (primary=0, second=0), (1,0), (0,1), (1,1).
    ``RandomizedEffect`` reports both marginal main effects and their interaction.
    """

    second_factor_assignment: Sequence[bool]
    cell_counts: tuple[int, int, int, int]
    second_factor_arms: tuple[str, str] = ("off", "on")

    def __post_init__(self) -> None:
        object.__setattr__(self, "second_factor_assignment", tuple(self.second_factor_assignment))
        object.__setattr__(self, "cell_counts", tuple(self.cell_counts))
        object.__setattr__(self, "second_factor_arms", tuple(self.second_factor_arms))
        if any(type(value) is not bool for value in self.second_factor_assignment):
            raise CausalValueError("second_factor_assignment entries must be bool")
        if len(self.cell_counts) != 4 or any(type(count) is not int or count < 2 for count in self.cell_counts):
            raise CausalValueError("factorial cell_counts must have four counts of at least two")
        if len(self.second_factor_arms) != 2 or any(not isinstance(arm, str) or not arm.strip() for arm in self.second_factor_arms) or self.second_factor_arms[0] == self.second_factor_arms[1]:
            raise CausalValueError("second_factor_arms must contain two distinct non-empty labels")


@dataclass(frozen=True, slots=True)
class MultiArmExperimentDesign:
    """Independent randomized assignment among three or more named actions."""

    realized_assignment: Sequence[str]
    action_labels: Sequence[str]
    assignment_probabilities: Sequence[Sequence[float]]
    assignment_units: Sequence[str]
    outcome_units: Sequence[str]

    def __post_init__(self) -> None:
        for name in ("realized_assignment", "action_labels", "assignment_units", "outcome_units"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "assignment_probabilities", tuple(tuple(row) for row in self.assignment_probabilities))
        n = len(self.realized_assignment)
        labels = self.action_labels
        if n < 3 or len(labels) < 3 or len(set(labels)) != len(labels) or any(not isinstance(label, str) or not label.strip() for label in labels):
            raise CausalValueError("multi-arm design requires at least three distinct non-empty action labels")
        if len(self.assignment_units) != n or len(self.outcome_units) != n or len(self.assignment_probabilities) != n:
            raise CausalValueError("multi-arm assignment, probabilities, and unit labels must align")
        if any(action not in labels for action in self.realized_assignment) or set(self.realized_assignment) != set(labels):
            raise CausalValueError("every declared multi-arm action needs observed support")
        if any(not isinstance(unit, str) or not unit.strip() for unit in (*self.assignment_units, *self.outcome_units)) or len(set(self.assignment_units)) != n or len(set(self.outcome_units)) != n:
            raise CausalValueError("multi-arm assignment and outcome units must be distinct non-empty labels")
        probabilities = np.asarray(self.assignment_probabilities, dtype=np.float64)
        if probabilities.shape != (n, len(labels)) or not np.isfinite(probabilities).all() or np.any(probabilities <= 0) or np.any(probabilities >= 1) or not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-8):
            raise CausalValueError("multi-arm probability rows must be positive, finite, and sum to one")


@dataclass(frozen=True, slots=True)
class SwitchbackDesign:
    """Randomized treatment switching over periods within independent sequences.

    Probabilities are marginal assignment probabilities for each unit-period.
    The variance treats each sequence as an independent cluster and allows
    arbitrary dependence among periods within a sequence.
    """

    realized_assignment: Sequence[bool]
    sequence_ids: Sequence[str]
    period_ids: Sequence[str]
    assignment_probabilities: Sequence[float]
    treatment_arms: tuple[str, str] = ("control", "treated")

    def __post_init__(self) -> None:
        for name in ("realized_assignment", "sequence_ids", "period_ids", "assignment_probabilities"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        n = len(self.realized_assignment)
        if n < 2 or any(len(getattr(self, name)) != n for name in ("sequence_ids", "period_ids", "assignment_probabilities")):
            raise CausalValueError("switchback assignment, sequence, period, and probability rows must align")
        if any(type(value) is not bool for value in self.realized_assignment):
            raise CausalValueError("switchback realized_assignment entries must be bool")
        if any(not isinstance(value, str) or not value.strip() for value in (*self.sequence_ids, *self.period_ids)):
            raise CausalValueError("switchback sequence and period identifiers must be non-empty")
        if len(set(zip(self.sequence_ids, self.period_ids, strict=True))) != n:
            raise CausalValueError("period identifiers must be unique within each sequence")
        probabilities = np.asarray(self.assignment_probabilities, dtype=np.float64)
        if not np.isfinite(probabilities).all() or np.any(probabilities <= 0) or np.any(probabilities >= 1):
            raise CausalValueError("switchback assignment probabilities must be strictly between zero and one")
        if len(set(self.sequence_ids)) < 2:
            raise CausalValueError("switchback variance requires at least two independent sequences")
        if len(set(self.realized_assignment)) != 2:
            raise CausalValueError("switchback requires both observed treatment arms across the full schedule")
        if len(self.treatment_arms) != 2 or any(not isinstance(arm, str) or not arm.strip() for arm in self.treatment_arms) or self.treatment_arms[0] == self.treatment_arms[1]:
            raise CausalValueError("treatment_arms must contain two distinct non-empty labels")


@dataclass(frozen=True, slots=True)
class SwitchbackEffect:
    """ITT query for a unit-period randomized switchback experiment."""

    outcome: str
    design: SwitchbackDesign
    kind: Literal["switchback_effect"] = field(default="switchback_effect", init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, str) or not self.outcome.strip():
            raise CausalValueError("outcome must be a non-empty variable name")

    def estimate(self, data: object) -> SwitchbackEstimate:
        """Estimate marginal period ITT with sequence-clustered uncertainty."""
        from ._data import as_columns

        names, columns = as_columns(data)
        if self.outcome not in names:
            raise CausalValueError(f"outcome column {self.outcome!r} is missing from data")
        outcomes = np.asarray(columns[names.index(self.outcome)], dtype=np.float64)
        if outcomes.ndim != 1 or outcomes.size != len(self.design.realized_assignment):
            raise CausalValueError("outcome must have one value per switchback unit-period")
        try:
            raw = _estimate_switchback_effect(
                outcomes,
                list(self.design.realized_assignment),
                list(self.design.sequence_ids),
                np.asarray(self.design.assignment_probabilities, dtype=np.float64),
            )
        except ValueError as error:
            raise CausalValueError(str(error)) from error
        effect, standard_error, minimum_probability, sequences = map(float, raw)
        return SwitchbackEstimate(
            effect=effect,
            standard_error=standard_error,
            minimum_assignment_probability=minimum_probability,
            sequence_count=int(sequences),
            period_count=len(outcomes),
            treatment_arms=self.design.treatment_arms,
        )


@dataclass(frozen=True, slots=True)
class SwitchbackEstimate:
    effect: float
    standard_error: float
    minimum_assignment_probability: float
    sequence_count: int
    period_count: int
    treatment_arms: tuple[str, str]
    estimand: str = "unit_period_itt"
    uncertainty: str = "independent_sequence_cluster_sandwich_standard_error_no_interval"
    support_status: str = "unlicensed_direct_estimator_utility"
    assumptions: tuple[str, ...] = (
        "known marginal randomization probability for every unit-period",
        "independent assignment sequences with arbitrary within-sequence dependence",
        "no carryover from earlier period assignments to the current period outcome",
        "both treatment arms observed across the full schedule",
        "consistency and no interference between independent sequences",
    )


AssignmentDesign = (
    BernoulliAssignment | CompleteRandomization | ClusterRandomization | StratifiedRandomization | FactorialRandomization
)


@dataclass(frozen=True, slots=True)
class ExperimentDesign:
    """Randomization and unit mapping for a two-arm experiment.

    ``assignment_units`` and ``outcome_units`` are aligned to rows in the
    supplied outcome table. One assignment unit per outcome row is required
    for Bernoulli and complete randomization. Cluster assignment may repeat an
    assignment unit across outcome rows; its cluster IDs must match the design.
    The retained ``analyze`` route supports Bernoulli, complete, stratified,
    cluster, and fixed-cell factorial assignment. SwitchbackEffect uses the
    retained route with sequence and period metadata. ComplierEffect uses a
    separate retained encouragement query with observed treatment receipt.
    """

    assignment: AssignmentDesign
    realized_assignment: Sequence[bool]
    assignment_units: Sequence[str]
    outcome_units: Sequence[str]
    _: KW_ONLY
    blocks: Sequence[str] | None = None
    treatment_arms: tuple[str, str] = ("control", "treated")
    estimand: Literal["itt"] = "itt"
    kind: Literal["experiment_design"] = field(
        default="experiment_design", init=False, repr=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "realized_assignment", tuple(self.realized_assignment))
        object.__setattr__(self, "assignment_units", tuple(self.assignment_units))
        object.__setattr__(self, "outcome_units", tuple(self.outcome_units))
        if not isinstance(self.treatment_arms, str):
            object.__setattr__(self, "treatment_arms", tuple(self.treatment_arms))
        if self.blocks is not None:
            object.__setattr__(self, "blocks", tuple(self.blocks))
        if isinstance(self.assignment, BernoulliAssignment):
            probability = self.assignment.probabilities
            if not isinstance(probability, (int, float)):
                probability = tuple(probability)
            object.__setattr__(self, "assignment", BernoulliAssignment(probability))
        elif isinstance(self.assignment, ClusterRandomization):
            object.__setattr__(
                self,
                "assignment",
                ClusterRandomization(tuple(self.assignment.clusters), self.assignment.treated_clusters),
            )
        elif isinstance(self.assignment, StratifiedRandomization):
            object.__setattr__(
                self,
                "assignment",
                StratifiedRandomization(dict(self.assignment.treated_per_block)),
            )
        if not isinstance(self.assignment, (BernoulliAssignment, CompleteRandomization, ClusterRandomization, StratifiedRandomization, FactorialRandomization)):
            raise CausalValueError("unsupported experiment assignment design")
        n = len(self.realized_assignment)
        if n == 0 or len(self.assignment_units) != n or len(self.outcome_units) != n:
            raise CausalValueError(
                "realized_assignment, assignment_units, and outcome_units must have equal non-zero length"
            )
        if any(
            not isinstance(unit, str) or not unit.strip()
            for unit in (*self.assignment_units, *self.outcome_units)
        ):
            raise CausalValueError("unit identifiers must be non-empty")
        if self.blocks is not None and (
            len(self.blocks) != n
            or any(not isinstance(block, str) or not block.strip() for block in self.blocks)
        ):
            raise CausalValueError("blocks must contain one non-empty string per outcome row")
        if (
            isinstance(self.treatment_arms, str)
            or len(self.treatment_arms) != 2
            or any(not isinstance(arm, str) or not arm.strip() for arm in self.treatment_arms)
            or self.treatment_arms[0] == self.treatment_arms[1]
        ):
            raise CausalValueError("treatment_arms must be two distinct non-empty labels")
        if any(type(assigned) is not bool for assigned in self.realized_assignment):
            raise CausalValueError("realized_assignment entries must be bool")
        if len(set(self.outcome_units)) != n:
            raise CausalValueError("outcome_units must identify one outcome row each")
        if not isinstance(self.assignment, ClusterRandomization) and len(
            set(self.assignment_units)
        ) != n:
            raise CausalValueError(
                "Bernoulli and complete randomization require one assignment unit per outcome row"
            )
        if isinstance(self.assignment, BernoulliAssignment):
            probabilities = self.assignment.probabilities
            if not isinstance(probabilities, (int, float)) and len(probabilities) not in (1, n):
                raise CausalValueError(
                    "Bernoulli probabilities must be scalar or have one value per outcome row"
                )
        if isinstance(self.assignment, CompleteRandomization) and self.assignment.treated >= n:
            raise CausalValueError("complete randomization treated count must be less than n")
        if isinstance(self.assignment, CompleteRandomization) and sum(self.realized_assignment) != (
            self.assignment.treated
        ):
            raise CausalValueError(
                "realized_assignment must contain exactly the declared treated count"
            )
        if isinstance(self.assignment, ClusterRandomization):
            if len(self.assignment.clusters) != n:
                raise CausalValueError("cluster IDs must have one entry per outcome row")
            encoded = {name: i for i, name in enumerate(dict.fromkeys(self.assignment_units))}
            if list(self.assignment.clusters) != [encoded[name] for name in self.assignment_units]:
                raise CausalValueError(
                    "cluster randomization IDs must encode assignment_units in first-seen order"
                )
            if self.assignment.treated_clusters >= len(encoded):
                raise CausalValueError("treated_clusters must be less than the cluster count")
            assignments_by_cluster: dict[int, bool] = {}
            for cluster, assigned in zip(
                self.assignment.clusters, self.realized_assignment, strict=True
            ):
                if cluster in assignments_by_cluster and assignments_by_cluster[cluster] != assigned:
                    raise CausalValueError(
                        "realized_assignment must be constant within each randomized cluster"
                    )
                assignments_by_cluster[cluster] = assigned
            if sum(assignments_by_cluster.values()) != self.assignment.treated_clusters:
                raise CausalValueError(
                    "realized_assignment must treat exactly the declared cluster count"
                )
        if isinstance(self.assignment, StratifiedRandomization):
            if self.blocks is None:
                raise CausalValueError("stratified randomization requires one block label per row")
            observed_blocks = set(self.blocks)
            if observed_blocks != set(self.assignment.treated_per_block):
                raise CausalValueError("treated_per_block keys must match the observed block labels")
            for block, treated_count in self.assignment.treated_per_block.items():
                indices = [i for i, value in enumerate(self.blocks) if value == block]
                if treated_count > len(indices) - 2:
                    raise CausalValueError(
                        "each block needs at least two treated and two control assignment units"
                    )
                if sum(self.realized_assignment[i] for i in indices) != treated_count:
                    raise CausalValueError(
                        f"realized_assignment must contain exactly {treated_count} treated units in block {block!r}"
                    )
        if isinstance(self.assignment, FactorialRandomization):
            factorial = self.assignment
            if self.blocks is not None:
                raise CausalValueError("factorial design does not combine with block metadata")
            if len(factorial.second_factor_assignment) != n or sum(factorial.cell_counts) != n:
                raise CausalValueError("factorial second-factor assignment and four-cell counts must align with rows")
            observed = [0, 0, 0, 0]
            for primary, second in zip(self.realized_assignment, factorial.second_factor_assignment, strict=True):
                observed[int(primary) + 2 * int(second)] += 1
            if tuple(observed) != factorial.cell_counts:
                raise CausalValueError("observed factorial assignments must match the four declared cell counts")
        if self.estimand != "itt":
            raise CausalValueError("RandomizedEffect currently supports the ITT estimand")


@dataclass(frozen=True, slots=True)
class FixedCUPED:
    """Pre-assignment covariate with a coefficient fixed outside this trial's outcomes."""

    covariate: str
    coefficient: float

    def __post_init__(self) -> None:
        if not isinstance(self.covariate, str) or not self.covariate.strip():
            raise CausalValueError("CUPED covariate must be a non-empty column name")
        if not np.isfinite(self.coefficient):
            raise CausalValueError("fixed CUPED coefficient must be finite")


@dataclass(frozen=True, slots=True)
class RandomizedEffect:
    """Intention-to-treat contrast carried through the ordinary analysis API.

    Bernoulli, complete, stratified, and cluster designs use their corresponding
    design-based point estimates and variance contracts. Other design families
    are explicitly refused by the retained analysis route.
    """

    outcome: str
    design: ExperimentDesign | MultiArmExperimentDesign
    _: KW_ONLY
    cuped: FixedCUPED | None = None
    ancova_covariates: tuple[str, ...] = ()
    exact_randomization_test: bool = False
    kind: Literal["randomized_effect"] = field(default="randomized_effect", init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, str) or not self.outcome.strip():
            raise CausalValueError("outcome must be a non-empty variable name")
        if isinstance(self.design, MultiArmExperimentDesign):
            if self.cuped is not None or self.ancova_covariates or self.exact_randomization_test:
                raise CausalValueError("multi-arm contrasts do not combine with CUPED, ANCOVA, or exact Fisher inference")
            return
        if self.cuped is not None:
            if not isinstance(self.cuped, FixedCUPED) or self.cuped.covariate == self.outcome:
                raise CausalValueError("CUPED requires a distinct pre-assignment covariate")
            if not isinstance(self.design.assignment, BernoulliAssignment):
                raise CausalValueError("retained fixed CUPED currently requires Bernoulli assignment")
        if self.ancova_covariates:
            if not isinstance(self.ancova_covariates, tuple) or any(not isinstance(name, str) or not name.strip() for name in self.ancova_covariates) or len(set(self.ancova_covariates)) != len(self.ancova_covariates) or self.outcome in self.ancova_covariates:
                raise CausalValueError("ANCOVA requires distinct, non-empty pre-assignment covariate names separate from outcome")
            if self.cuped is not None or not isinstance(self.design.assignment, BernoulliAssignment):
                raise CausalValueError("retained ANCOVA requires Bernoulli assignment and cannot combine with fixed CUPED")
            if len(self.design.realized_assignment) <= len(self.ancova_covariates) + 2:
                raise CausalValueError("ANCOVA requires residual degrees of freedom")
        if isinstance(self.design.assignment, FactorialRandomization) and self.exact_randomization_test:
            raise CausalValueError("exact Fisher inference is only available for unadjusted complete two-arm designs")
        if self.exact_randomization_test and (
            type(self.exact_randomization_test) is not bool
            or self.cuped is not None
            or self.ancova_covariates
            or not isinstance(self.design.assignment, CompleteRandomization)
            or len(self.design.realized_assignment) > 20
        ):
            raise CausalValueError("exact randomization inference requires an unadjusted complete two-arm design with at most 20 units")

    def to_interference_query(self) -> InterferenceQuery:
        """Lower ITT to the shared native randomization estimator contract."""
        if isinstance(self.design.assignment, StratifiedRandomization):
            raise CausalValueError(
                "stratified randomization has a direct estimator but no retained analyze support cell"
            )
        return InterferenceQuery(
            self.design.assignment,
            NeighborCount(),
            ExposureContrast(
                self.outcome,
                ExposureLevel(0.0, 0.0),
                ExposureLevel(1.0, 0.0),
            ),
            network=(),
            realized_assignment=self.design.realized_assignment,
        )

    def estimate(self, data: object, *, seed: int = 1) -> RandomizedExperimentEstimate:
        """Run the native randomization estimator and retain the design record.

        This direct estimator utility supports the underlying Bernoulli,
        complete, and cluster assignment kernels. Its variance is a
        conservative design-based variance bound, not a calibrated interval
        or a support-matrix license. Exact randomization inference is available
        through the retained ``analyze`` or ``prepare`` route.
        """
        if isinstance(self.design, MultiArmExperimentDesign):
            raise CausalUnsupportedError("multi-arm RandomizedEffect requires analyze or prepare to retain every arm and the assignment record", reason_code="route_not_supported")
        if self.exact_randomization_test or self.ancova_covariates:
            raise CausalUnsupportedError(
                "exact randomization inference requires analyze or prepare; ANCOVA also requires this retained inference contract",
                reason_code="route_not_supported",
            )
        if isinstance(self.design.assignment, FactorialRandomization):
            raise CausalUnsupportedError(
                "factorial contrasts require analyze or prepare to retain all three estimands and their variance contracts",
                reason_code="route_not_supported",
            )
        if self.cuped is not None:
            raise CausalUnsupportedError(
                "fixed CUPED is available through analyze or prepare; the direct estimator does not retain its coefficient",
                reason_code="route_not_supported",
            )

        assignment_kind = (
            "bernoulli"
            if isinstance(self.design.assignment, BernoulliAssignment)
            else "stratified"
            if isinstance(self.design.assignment, StratifiedRandomization)
            else "complete"
            if isinstance(self.design.assignment, CompleteRandomization)
            else "cluster"
        )
        if isinstance(self.design.assignment, StratifiedRandomization):
            from ._data import as_columns

            names, columns = as_columns(data)
            if self.outcome not in names:
                raise CausalValueError(f"outcome column {self.outcome!r} is missing from data")
            estimated_raw = _estimate_stratified_effect(
                np.asarray(columns[names.index(self.outcome)], dtype=np.float64),
                list(self.design.realized_assignment),
                list(self.design.blocks or ()),
            )
            effect, variance_bound, minimum_probability, _ = map(float, estimated_raw)
        else:
            estimated = estimate_interference(
                data,
                assignment=self.design.realized_assignment,
                edges=(),
                query=self.to_interference_query(),
                seed=seed,
            )
            effect = estimated.contrast.horvitz_thompson
            variance_bound = estimated.contrast.conservative_variance
            minimum_probability = estimated.minimum_exposure_probability
        return RandomizedExperimentEstimate(
            effect=effect,
            variance_upper_bound=variance_bound,
            assignment_design=assignment_kind,
            assignment_units=tuple(self.design.assignment_units),
            outcome_units=tuple(self.design.outcome_units),
            blocks=self.design.blocks,
            treatment_arms=self.design.treatment_arms,
            control_units=len(self.design.realized_assignment) - sum(self.design.realized_assignment),
            treatment_units=sum(self.design.realized_assignment),
            minimum_assignment_probability=minimum_probability,
            uncertainty=(
                "stratified_neyman_variance_upper_bound_no_interval"
                if isinstance(self.design.assignment, StratifiedRandomization)
                else "complete_neyman_variance_upper_bound_no_interval"
                if isinstance(self.design.assignment, CompleteRandomization)
                else "cluster_conservative_variance_bound_no_interval"
                if isinstance(self.design.assignment, ClusterRandomization)
                else "bernoulli_ht_design_variance_no_interval"
            ),
            support_status="unlicensed_direct_estimator_utility",
        )


@dataclass(frozen=True, slots=True)
class RandomizedExperimentEstimate:
    """Native randomized estimate with retained design and unit mapping.

    The legacy ``variance_upper_bound`` field is a sequence sandwich variance
    estimate for switchback designs. Read ``uncertainty`` for its semantics.
    """

    effect: float
    variance_upper_bound: float
    assignment_design: Literal["bernoulli", "complete", "cluster", "stratified", "switchback", "factorial_2x2", "multi_arm"]
    assignment_units: tuple[str, ...]
    outcome_units: tuple[str, ...]
    blocks: tuple[str, ...] | None
    treatment_arms: tuple[str, str]
    control_units: int
    treatment_units: int
    minimum_assignment_probability: float
    uncertainty: str
    support_status: str
    periods: tuple[str, ...] | None = None
    estimand: Literal["itt", "cace_late", "treatment_on_treated", "factorial_primary_main_effect", "multi_arm_itt"] = "itt"
    intention_to_treat_effect: float | None = None
    first_stage_effect: float | None = None
    received_treatment: tuple[bool, ...] | None = None
    randomization_p_value: float | None = None
    randomization_allocations: int | None = None
    second_factor_effect: float | None = None
    factorial_interaction: float | None = None
    second_factor_variance: float | None = None
    factorial_interaction_variance: float | None = None
    multi_arm_values: tuple[tuple[str, float, float, int], ...] = ()
    standard_error: float | None = None
    interval_95: tuple[float, float] | None = None
    second_factor_interval_95: tuple[float, float] | None = None
    factorial_interaction_interval_95: tuple[float, float] | None = None
    multi_arm_intervals_95: tuple[tuple[float, float] | None, ...] = ()

    @property
    def multi_arm_contrasts(self) -> tuple[MultiArmContrast, ...]:
        """Named contrasts with variance bounds and supported pointwise intervals."""
        if not self.multi_arm_values:
            return ()
        control = self.multi_arm_values[0]
        return tuple(MultiArmContrast(label, value, value - control[1],
                                     2.0 * (variance + control[2]), support,
                                     self.multi_arm_intervals_95[index]
                                     if index < len(self.multi_arm_intervals_95) else None)
                     for index, (label, value, variance, support)
                     in enumerate(self.multi_arm_values[1:], start=1))

    @property
    def variance(self) -> float:
        """Design-aware variance estimate or bound identified by ``uncertainty``."""
        return self.variance_upper_bound


@dataclass(frozen=True, slots=True)
class ComplierEffect:
    """Wald CACE/LATE from Bernoulli encouragement and observed receipt.

    Exclusion, monotonicity, and independent assignment units are declared
    identifying assumptions. The retained result has no calibrated interval.
    """

    outcome: str
    design: ExperimentDesign
    received_treatment: Sequence[bool]
    kind: Literal["complier_effect"] = field(default="complier_effect", init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, str) or not self.outcome.strip():
            raise CausalValueError("outcome must be a non-empty variable name")
        if not isinstance(self.design.assignment, BernoulliAssignment):
            raise CausalValueError("retained CACE/LATE requires Bernoulli randomization")
        receipt = tuple(self.received_treatment)
        if len(receipt) != len(self.design.realized_assignment) or any(type(value) is not bool for value in receipt):
            raise CausalValueError("treatment receipt must be row-aligned booleans")
        object.__setattr__(self, "received_treatment", receipt)


@dataclass(frozen=True, slots=True)
class TreatmentOnTreated:
    """Effect among treatment recipients under one-sided noncompliance.

    Assignment is independent Bernoulli encouragement. The caller asserts
    exclusion and that a control-assigned unit could never receive treatment;
    observed control-arm receipt is checked. Under these claims recipients are
    compliers, so the Wald ratio identifies their mean treatment effect.
    """

    outcome: str
    design: ExperimentDesign
    received_treatment: Sequence[bool]
    kind: Literal["treatment_on_treated"] = field(default="treatment_on_treated", init=False, repr=False)

    def __post_init__(self) -> None:
        ComplierEffect(self.outcome, self.design, self.received_treatment)
        receipt = tuple(self.received_treatment)
        if any(received and not assigned for received, assigned in zip(receipt, self.design.realized_assignment, strict=True)):
            raise CausalValueError("treatment-on-treated requires no control-assigned treatment receipt")
        object.__setattr__(self, "received_treatment", receipt)


@dataclass(frozen=True, slots=True)
class ComplierEffectEstimate:
    """ITT and Wald CACE/LATE for a randomized binary encouragement."""

    intention_to_treat_effect: float
    first_stage_effect: float
    complier_average_causal_effect: float
    standard_error: float
    estimand: str = "cace_late"
    uncertainty: str = "asymptotic_influence_function_standard_error"
    support_status: str = "unlicensed_direct_estimator_utility"
    assumptions: tuple[str, ...] = (
        "random assignment of the encouragement with known positive probabilities",
        "exclusion restriction: assignment affects outcome only through treatment receipt",
        "monotonicity: no unit is induced to do the opposite by assignment",
        "non-zero first stage and consistency",
        "independent assignment units for the reported influence-function standard error",
    )


def estimate_complier_effect(
    outcomes: Sequence[float],
    assignment: Sequence[bool],
    received: Sequence[bool],
    propensity: float | Sequence[float],
) -> ComplierEffectEstimate:
    """Estimate ITT and Wald CACE/LATE under randomized noncompliance.

    This utility accepts unit-level known assignment probabilities and returns
    a point estimate plus an asymptotic influence-function standard error. It
    assumes exclusion and monotonicity; it does not report a confidence interval
    or add a licensed support cell.
    """
    y = np.asarray(outcomes, dtype=np.float64)
    if y.ndim != 1 or y.size < 2 or not np.isfinite(y).all():
        raise CausalValueError("outcomes must be a finite one-dimensional array with at least two rows")
    z = _binary_assignment(assignment, len(y), "assignment")
    d = _binary_assignment(received, len(y), "received")
    p = np.asarray(propensity, dtype=np.float64)
    if p.ndim == 0:
        p = np.full(len(y), float(p), dtype=np.float64)
    if p.ndim != 1 or len(p) not in (1, len(y)):
        raise CausalValueError("propensity must be scalar or one value per unit")
    if not np.isfinite(p).all() or np.any(p <= 0) or np.any(p >= 1):
        raise CausalValueError("assignment propensities must be strictly between zero and one")
    try:
        raw = _estimate_complier_effect(y, z, d, p)
    except ValueError as error:
        raise CausalValueError(str(error)) from error
    return ComplierEffectEstimate(*map(float, raw))


@dataclass(frozen=True, slots=True)
class CUPEDEstimate:
    effect: float
    adjustment_coefficient: float
    standard_error: float
    uncertainty: str = "asymptotic_independent_unit_standard_error"
    support_status: str = "unlicensed_direct_estimator_utility"
    assumptions: tuple[str, ...] = (
        "known independent Bernoulli assignment probabilities",
        "the pre-treatment covariate is measured before assignment",
        "independent assignment units",
    )


def estimate_cuped_effect(
    outcomes: Sequence[float],
    covariate: Sequence[float],
    assignment: Sequence[bool],
    propensity: float | Sequence[float],
) -> CUPEDEstimate:
    """Estimate a precision-adjusted Bernoulli ITT using one CUPED covariate."""
    y = np.asarray(outcomes, dtype=np.float64)
    x = np.asarray(covariate, dtype=np.float64)
    if y.ndim != 1 or y.size < 2 or x.shape != y.shape:
        raise CausalValueError("outcomes and covariate must be one-dimensional arrays with equal rows")
    z = _binary_assignment(assignment, len(y), "assignment")
    p = np.asarray(propensity, dtype=np.float64)
    if p.ndim == 0:
        p = np.full(len(y), float(p), dtype=np.float64)
    if p.ndim != 1 or len(p) not in (1, len(y)):
        raise CausalValueError("propensity must be scalar or one value per unit")
    try:
        raw = _estimate_cuped_effect(y, x, z, p)
    except ValueError as error:
        raise CausalValueError(str(error)) from error
    return CUPEDEstimate(*map(float, raw))


@dataclass(frozen=True, slots=True)
class ANCOVAEstimate:
    """Multi-covariate randomized ANCOVA with an HC0 standard error."""

    effect: float
    adjustment_coefficients: tuple[tuple[str, float], ...]
    standard_error: float
    treated_support: int
    control_support: int
    uncertainty: str = "hc0_independent_unit_standard_error_no_interval"
    support_status: str = "unlicensed_direct_estimator_utility"
    assumptions: tuple[str, ...] = (
        "independent Bernoulli assignment with a common known probability",
        "all adjustment covariates were measured before assignment",
        "linear additive outcome model with a common additive treatment effect",
        "independent outcome units and homoskedasticity not required for HC0 consistency",
    )


def estimate_ancova_effect(
    data: object,
    *,
    outcome: str,
    design: ExperimentDesign,
    covariates: Sequence[str],
) -> ANCOVAEstimate:
    """Fit a multi-covariate OLS ANCOVA for independent Bernoulli assignment.

    Returns the treatment coefficient, adjustment coefficients, and HC0
    independent-row sandwich standard error. The API requires a common
    Bernoulli propensity and does not produce a confidence interval or a
    support-matrix license.
    """
    from ._data import as_columns

    if not isinstance(design, ExperimentDesign) or not isinstance(design.assignment, BernoulliAssignment):
        raise CausalValueError("ANCOVA requires a Bernoulli ExperimentDesign")
    probabilities = design.assignment.probabilities
    probs = np.asarray(probabilities if not isinstance(probabilities, (int, float)) else [probabilities], dtype=np.float64)
    if probs.size > 1 and not np.allclose(probs, probs[0], rtol=0.0, atol=1e-12):
        raise CausalValueError("ANCOVA requires a common Bernoulli assignment probability")
    if len(set(design.assignment_units)) != len(design.assignment_units):
        raise CausalValueError("ANCOVA HC0 uncertainty requires independent unit-level assignment")
    covariate_names = tuple(covariates)
    if not covariate_names or len(set(covariate_names)) != len(covariate_names):
        raise CausalValueError("covariates must contain unique non-empty column names")
    if any(not isinstance(name, str) or not name.strip() for name in covariate_names):
        raise CausalValueError("covariates must contain unique non-empty column names")
    names, columns = as_columns(data)
    required = (outcome, *covariate_names)
    missing = [name for name in required if name not in names]
    if missing:
        raise CausalValueError(f"ANCOVA columns are missing: {', '.join(missing)}")
    y = np.asarray(columns[names.index(outcome)], dtype=np.float64)
    x = np.column_stack([np.asarray(columns[names.index(name)], dtype=np.float64) for name in covariate_names])
    if y.ndim != 1 or y.size != len(design.realized_assignment) or x.shape[0] != y.size:
        raise CausalValueError("ANCOVA outcomes, covariates, and design rows must align")
    try:
        raw = _estimate_ancova_effect(y, list(design.realized_assignment), x)
    except ValueError as error:
        raise CausalValueError(str(error)) from error
    effect, standard_error, coefficients, treated, control = raw
    return ANCOVAEstimate(
        float(effect),
        tuple((name, float(value)) for name, value in zip(covariate_names, coefficients, strict=True)),
        float(standard_error),
        int(treated),
        int(control),
    )


@dataclass(frozen=True, slots=True)
class RandomizationTest:
    observed_effect: float
    exact_two_sided_p_value: float
    assignments_enumerated: int
    null: str = "sharp_no_effect"
    uncertainty: str = "exact_randomization_test_under_declared_bernoulli_design"
    support_status: str = "unlicensed_direct_inference_utility"


def exact_randomization_test(
    outcomes: Sequence[float], assignment: Sequence[bool], propensity: float | Sequence[float]
) -> RandomizationTest:
    """Enumerate a two-sided Bernoulli randomization test of the sharp null.

    Exact enumeration is limited to 20 independent assignment units. This is a
    test of the sharp no-effect null, not a confidence interval or a weak-null
    test, and it does not create a licensed support coordinate.
    """
    y = np.asarray(outcomes, dtype=np.float64)
    if y.ndim != 1 or y.size == 0 or y.size > 20 or not np.isfinite(y).all():
        raise CausalValueError("outcomes must be finite with 1 to 20 assignment units")
    z = _binary_assignment(assignment, len(y), "assignment")
    p = np.asarray(propensity, dtype=np.float64)
    if p.ndim == 0:
        p = np.full(len(y), float(p), dtype=np.float64)
    if p.ndim != 1 or len(p) not in (1, len(y)):
        raise CausalValueError("propensity must be scalar or one value per unit")
    try:
        observed, p_value, allocations = _exact_randomization_test(y, z, p)
    except ValueError as error:
        raise CausalValueError(str(error)) from error
    return RandomizationTest(float(observed), float(p_value), int(allocations))


@dataclass(frozen=True, slots=True)
class MultiArmContrast:
    action: str
    value: float
    effect_vs_control: float
    variance_bound_vs_control: float
    observed_support: int
    interval_95: tuple[float, float] | None = None


@dataclass(frozen=True, slots=True)
class MultiArmExperimentEstimate:
    control: str
    arm_values: tuple[tuple[str, float], ...]
    contrasts: tuple[MultiArmContrast, ...]
    uncertainty: str = "covariance_free_variance_bound_no_interval"
    support_status: str = "unlicensed_direct_estimator_utility"
    assumptions: tuple[str, ...] = (
        "known randomized probabilities for every action and unit",
        "consistency, no interference, and independent assignment units",
        "each action has positive probability and observed support",
    )


def estimate_multi_arm_effect(
    data: object,
    *,
    outcome: str,
    assignment: Sequence[str],
    action_labels: Sequence[str],
    propensities: Sequence[Sequence[float]],
) -> MultiArmExperimentEstimate:
    """Estimate all arm means and contrasts to the first (control) action."""
    from ._data import as_columns

    names, columns = as_columns(data)
    if outcome not in names:
        raise CausalValueError(f"outcome column {outcome!r} is missing from data")
    y = np.asarray(columns[names.index(outcome)], dtype=np.float64)
    labels = tuple(action_labels)
    assigned = tuple(assignment)
    if len(labels) < 2 or any(not isinstance(label, str) or not label.strip() for label in labels):
        raise CausalValueError("action_labels must contain at least two non-empty labels")
    if len(set(labels)) != len(labels):
        raise CausalValueError("action_labels must be unique")
    if len(assigned) != len(y) or any(action not in labels for action in assigned):
        raise CausalValueError("assignment must name one declared action per outcome row")
    probabilities = np.asarray(propensities, dtype=np.float64)
    if probabilities.shape != (len(y), len(labels)):
        raise CausalValueError("propensities must have one row per unit and one column per action")
    try:
        raw = _estimate_multi_arm_effects(
            y, [labels.index(action) for action in assigned], probabilities
        )
    except ValueError as error:
        raise CausalValueError(str(error)) from error
    values = tuple(float(item[0]) for item in raw)
    variances = tuple(float(item[1]) for item in raw)
    supports = tuple(int(item[2]) for item in raw)
    control_value = values[0]
    contrasts = tuple(
        MultiArmContrast(
            labels[index], values[index], values[index] - control_value,
            2.0 * (variances[index] + variances[0]), supports[index],
        )
        for index in range(1, len(labels))
    )
    return MultiArmExperimentEstimate(labels[0], tuple(zip(labels, values, strict=True)), contrasts)


def _binary_assignment(values: Sequence[bool], n: int, name: str) -> list[bool]:
    raw = list(values)
    if len(raw) != n:
        raise CausalValueError(f"{name} must have one value per outcome row")
    if any(not isinstance(value, (bool, np.bool_)) for value in raw):
        raise CausalValueError(f"{name} must contain only booleans")
    return [bool(value) for value in raw]


__all__ = [
    "ANCOVAEstimate",
    "ComplierEffect",
    "ComplierEffectEstimate",
    "TreatmentOnTreated",
    "CUPEDEstimate",
    "FixedCUPED",
    "FactorialRandomization",
    "ExperimentDesign",
    "MultiArmContrast",
    "MultiArmExperimentDesign",
    "MultiArmExperimentEstimate",
    "RandomizedEffect",
    "RandomizedExperimentEstimate",
    "RandomizationTest",
    "StratifiedRandomization",
    "SwitchbackDesign",
    "SwitchbackEffect",
    "SwitchbackEstimate",
    "estimate_complier_effect",
    "estimate_cuped_effect",
    "estimate_ancova_effect",
    "exact_randomization_test",
    "estimate_multi_arm_effect",
]
