"""Binary treatment policies and held-out randomized evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import floor, isfinite
from numbers import Integral
from typing import Any

import numpy as np

from ._data import as_columns
from ._native import conditional_dose_response as _conditional_dose_response
from ._native import evaluate_binary_policy as _evaluate_binary_policy
from ._native import evaluate_binary_policy_doubly_robust as _evaluate_binary_policy_doubly_robust
from ._native import evaluate_multi_action_policy as _evaluate_multi_action_policy
from ._native import uplift_by_score as _uplift_by_score
from .errors import CausalTypeError, CausalValueError


@dataclass(frozen=True, slots=True)
class BinaryPolicy:
    """A binary treat/control recommendation for each evaluation row.

    ``capacity`` and ``max_treatment_rate`` constrain the number of treated
    rows. ``budget`` constrains the sum of their declared treatment costs.
    Costs may be scalar or one per row.
    """

    actions: Sequence[bool]
    capacity: int | None = None
    budget: float | None = None
    max_treatment_rate: float | None = None
    costs: float | Sequence[float] = 0.0

    def __post_init__(self) -> None:
        actions = tuple(self.actions)
        if any(not isinstance(action, (bool, np.bool_)) for action in actions):
            raise CausalValueError("policy actions must be bool values")
        if not actions:
            raise CausalValueError("policy actions must not be empty")
        object.__setattr__(self, "actions", tuple(bool(action) for action in actions))
        if self.capacity is not None and (
            isinstance(self.capacity, bool)
            or not isinstance(self.capacity, Integral)
            or self.capacity < 0
        ):
            raise CausalValueError("capacity must be a non-negative integer")
        if self.capacity is not None:
            object.__setattr__(self, "capacity", int(self.capacity))
        if self.budget is not None and (not isfinite(self.budget) or self.budget < 0.0):
            raise CausalValueError("budget must be finite and non-negative")
        if self.max_treatment_rate is not None and (
            isinstance(self.max_treatment_rate, bool)
            or not isinstance(self.max_treatment_rate, (int, float))
            or not isfinite(self.max_treatment_rate)
            or not 0.0 <= self.max_treatment_rate <= 1.0
        ):
            raise CausalValueError("max_treatment_rate must be in [0, 1]")

    @classmethod
    def top_k(
        cls,
        scores: Sequence[float],
        k: int,
        *,
        available: Sequence[bool] | None = None,
        budget: float | None = None,
        costs: float | Sequence[float] = 0.0,
    ) -> BinaryPolicy:
        """Choose the top ``k`` score rows, breaking ties by input order."""
        values = np.asarray(scores, dtype=np.float64)
        if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
            raise CausalValueError("scores must be a non-empty finite one-dimensional sequence")
        if isinstance(k, bool) or not isinstance(k, Integral) or not 0 <= k <= values.size:
            raise CausalValueError("k must be an integer between zero and the number of scores")
        can_treat = [True] * values.size if available is None else list(available)
        if len(can_treat) != values.size or any(
            not isinstance(value, (bool, np.bool_)) for value in can_treat
        ):
            raise CausalValueError("available must contain one bool per score")
        eligible = [i for i, can in enumerate(can_treat) if can]
        if k > len(eligible):
            raise CausalValueError("k exceeds the number of available treatment actions")
        ranked = sorted(eligible, key=lambda i: (-values[i], i))
        actions = [False] * values.size
        for index in ranked[: int(k)]:
            actions[index] = True
        return cls(actions, capacity=int(k), budget=budget, costs=costs)


@dataclass(frozen=True, slots=True)
class MultiActionPolicy:
    """Fixed multi-action recommendations; the first label is the control action."""

    action_labels: Sequence[str]
    recommendations: Sequence[str]
    costs: Sequence[float] | None = None
    capacities: Sequence[int] | None = None
    budget: float | None = None

    def __post_init__(self) -> None:
        labels = tuple(self.action_labels)
        recommendations = tuple(self.recommendations)
        if len(labels) < 2 or any(not isinstance(v, str) or not v.strip() for v in labels):
            raise CausalValueError("action_labels must contain at least two non-empty labels")
        if len(set(labels)) != len(labels):
            raise CausalValueError("action_labels must be unique")
        if not recommendations or any(v not in labels for v in recommendations):
            raise CausalValueError("recommendations must name a declared action for each row")
        object.__setattr__(self, "action_labels", labels)
        object.__setattr__(self, "recommendations", recommendations)
        if self.costs is not None:
            try:
                costs = tuple(float(v) for v in self.costs)
            except (TypeError, ValueError) as error:
                raise CausalValueError("costs must be numeric") from error
            if len(costs) != len(labels) or any(not isfinite(v) or v < 0 for v in costs):
                raise CausalValueError("costs must have one finite non-negative value per action")
            object.__setattr__(self, "costs", costs)
        if self.capacities is not None:
            capacities = tuple(self.capacities)
            if len(capacities) != len(labels) or any(
                isinstance(v, bool) or not isinstance(v, Integral) or v < 0 for v in capacities
            ):
                raise CausalValueError("capacities must have one non-negative integer per action")
            object.__setattr__(self, "capacities", tuple(int(v) for v in capacities))
        if self.budget is not None and (
            isinstance(self.budget, bool) or not isfinite(self.budget) or self.budget < 0
        ):
            raise CausalValueError("budget must be finite and non-negative")

    @classmethod
    def from_scores(
        cls,
        scores: Sequence[Sequence[float]],
        action_labels: Sequence[str],
        *,
        available: Sequence[Sequence[bool]] | None = None,
        costs: Sequence[float] | None = None,
        capacities: Sequence[int] | None = None,
        budget: float | None = None,
    ) -> MultiActionPolicy:
        """Choose the highest-score available action per row, ties by label order."""
        matrix = np.asarray(scores, dtype=np.float64)
        labels = tuple(action_labels)
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] != len(labels):
            raise CausalValueError("scores must be a non-empty row-by-action matrix")
        if not np.isfinite(matrix).all():
            raise CausalValueError("scores must be finite")
        if available is None:
            mask = np.ones(matrix.shape, dtype=bool)
        else:
            raw_mask = np.asarray(available, dtype=object)
            if raw_mask.shape != matrix.shape or any(
                not isinstance(value, (bool, np.bool_)) for value in raw_mask.flat
            ):
                raise CausalValueError("available must have one bool per score and action")
            mask = raw_mask.astype(bool)
        if mask.shape != matrix.shape or (~mask).all(axis=1).any():
            raise CausalValueError("each row must have at least one available action")
        picks = [labels[int(np.argmax(np.where(mask[i], matrix[i], -np.inf)))] for i in range(len(matrix))]
        return cls(labels, picks, costs, capacities, budget)


@dataclass(frozen=True, slots=True)
class PolicyEvaluation:
    """Point-only held-out value estimates for a binary treatment policy."""

    policy_value: float
    reference_value: float
    incremental_value: float
    relative_value_gap: float
    treatment_rate: float
    total_treatment_cost: float
    uncertainty: str = "point_only"
    evaluation_method: str = "randomized_ipw_heldout"


@dataclass(frozen=True, slots=True)
class DoublyRobustPolicyEvaluation:
    """Doubly robust value estimates and row-level score standard errors."""

    policy_value: float
    reference_value: float
    incremental_value: float
    relative_value_gap: float
    treatment_rate: float
    total_treatment_cost: float
    policy_value_standard_error: float
    reference_value_standard_error: float
    incremental_value_standard_error: float
    prediction_ownership: str
    propensity_min: float
    propensity_max: float
    uncertainty: str = "row_score_standard_error_independent_subjects"
    evaluation_method: str = "doubly_robust_randomized_heldout_or_cross_fitted"
    assumptions: tuple[str, ...] = (
    "Known randomized treatment propensities with strict treatment overlap.",
    "The supplied randomization propensities are correct; nuisance outcome predictions may be misspecified.",
        "Consistency and no interference between evaluation subjects.",
        "Policy recommendations and both outcome nuisance predictions were generated without using the corresponding evaluation subject's outcome.",
    )
    diagnostics: tuple[str, ...] = (
        "row-level standard errors assume independent evaluation subjects",
        "training/test disjointness or excluded-fold correspondence is checked from caller-supplied IDs only",
        "nuisance predictions and randomization claims are not independently authenticated",
    )
    support_status: str = "unlicensed_point_utility"


@dataclass(frozen=True, slots=True)
class PolicyValue:
    """Immutable query spec for held-out doubly robust binary policy value."""

    outcome: str
    assignment: Sequence[bool]
    propensity: float | Sequence[float]
    policy: BinaryPolicy
    mu0: Sequence[float]
    mu1: Sequence[float]
    evaluation_subject_ids: Sequence[str]
    training_subject_ids: Sequence[str] | None = None
    fold_ids: Sequence[int] | None = None
    prediction_excluded_fold_ids: Sequence[int] | None = None
    reference: BinaryPolicy | None = None
    _ownership: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, str) or not self.outcome:
            raise CausalValueError("outcome must be a non-empty column name")
        ownership = _prediction_ownership(
            self.evaluation_subject_ids, len(self.policy.actions), self.training_subject_ids,
            self.fold_ids, self.prediction_excluded_fold_ids,
        )
        object.__setattr__(self, "assignment", tuple(bool(v) for v in self.assignment))
        object.__setattr__(self, "mu0", tuple(float(v) for v in self.mu0))
        object.__setattr__(self, "mu1", tuple(float(v) for v in self.mu1))
        object.__setattr__(self, "evaluation_subject_ids", tuple(str(v) for v in self.evaluation_subject_ids))
        object.__setattr__(self, "_ownership", ownership)


@dataclass(frozen=True, slots=True)
class UpliftBin:
    rank: int
    effect: float
    standard_error: float
    evaluation_rows: int


@dataclass(frozen=True, slots=True)
class ConditionalDoseResponsePoint:
    baseline_group: str
    target_dose: float
    response: float
    local_rows: int
    effective_sample_size: float
    minimum_dose_density: float
    maximum_normalized_weight: float
    local_outcome_sd: float


@dataclass(frozen=True, slots=True)
class ConditionalDoseResponseEstimate:
    points: tuple[ConditionalDoseResponsePoint, ...]
    bandwidth: float
    density_provenance: str
    uncertainty: str = "point_only"
    evaluation_method: str = "stratified_triangular_kernel_inverse_density"
    policy_value_estimated: bool = False
    support_status: str = "unlicensed_point_utility"
    assumptions: tuple[str, ...] = (
        "Conditional exchangeability of continuous dose given the supplied baseline groups.",
        "Consistency, no interference, and correct dose density at observed doses.",
        "Continuous-dose positivity and adequate local support around each target dose in every group.",
    )
    diagnostics: tuple[str, ...] = (
        "local row count, effective sample size, density floor, and maximum normalized weight reported",
        "local outcome SD is descriptive and is not an inferential standard error",
        "does not estimate a learned continuous-dose policy value",
    )


def uplift_by_score(
    evaluation_data: Any,
    *,
    outcome: str,
    assignment: Sequence[bool],
    propensity: float | Sequence[float],
    scores: Sequence[float],
    bins: int = 10,
) -> tuple[UpliftBin, ...]:
    """Estimate held-out randomized uplift across descending score deciles.

    Scores must come from an independently fitted or cross-fitted model. The
    function does not verify fold separation or train a CATE model.
    """
    if isinstance(bins, bool) or not isinstance(bins, Integral) or bins < 1:
        raise CausalValueError("bins must be a positive integer")
    names, columns = as_columns(evaluation_data)
    try:
        values = np.asarray(columns[names.index(outcome)], dtype=np.float64)
    except ValueError as error:
        raise CausalValueError(f"outcome column {outcome!r} is missing from evaluation_data") from error
    score_values = np.asarray(scores, dtype=np.float64)
    if score_values.shape != values.shape or not np.isfinite(score_values).all():
        raise CausalValueError("scores must be finite and match evaluation rows")
    if len(assignment) != len(values) or any(
        not isinstance(value, (bool, np.bool_)) for value in assignment
    ):
        raise CausalValueError("assignment must contain one bool per evaluation row")
    if not 1 <= bins <= len(values):
        raise CausalValueError("bins must be between one and the number of evaluation rows")
    order = np.argsort(-score_values, kind="stable")
    bin_ids = np.empty(len(values), dtype=np.uint64)
    for rank, row in enumerate(order):
        bin_ids[row] = min(bins - 1, rank * bins // len(values))
    propensity_values = [float(propensity)] if np.isscalar(propensity) else list(propensity)
    raw = _uplift_by_score(
        values, [bool(value) for value in assignment], bin_ids.tolist(),
        np.asarray(propensity_values, dtype=np.float64), int(bins),
    )
    return tuple(UpliftBin(int(rank), float(effect), float(se), int(count))
                 for rank, effect, se, count in raw)


def estimate_continuous_dose_response(
    data: Any,
    *,
    outcome: str,
    dose: str,
    baseline_group: str,
    dose_density: str,
    target_doses: Sequence[float],
    bandwidth: float,
    density_provenance: str,
    min_local_support: int = 3,
) -> ConditionalDoseResponseEstimate:
    """Estimate conditional responses at target doses with local kernel weights.

    Within each supplied baseline group, this computes a Hájek mean using a
    triangular kernel centered at each target dose and inverse caller-supplied
    density at each observed dose. It is a response curve utility; it does not
    select or evaluate a dose policy.
    """

    # Preserve categorical group labels while converting only numeric columns.
    if isinstance(data, Mapping):
        names = [str(name) for name in data]
        columns = [np.asarray(data[name]) for name in data]
    elif hasattr(data, "columns") and hasattr(data, "__getitem__"):
        names = [str(name) for name in data.columns]
        columns = [np.asarray(data[name]) for name in data.columns]
    else:
        names, columns = as_columns(data)
    if len(set(names)) != len(names) or any(column.ndim != 1 for column in columns):
        raise CausalValueError("continuous-dose inputs require unique one-dimensional columns")
    if columns and len({len(column) for column in columns}) != 1:
        raise CausalValueError("continuous-dose input columns must have equal row counts")
    for name in (outcome, dose, baseline_group, dose_density):
        if name not in names:
            raise CausalValueError(f"required continuous-dose column {name!r} is missing")
    if not isinstance(density_provenance, str) or density_provenance not in {
        "known", "externally_estimated"
    }:
        raise CausalValueError("density_provenance must be known or externally_estimated")
    if not isfinite(bandwidth) or bandwidth <= 0.0:
        raise CausalValueError("bandwidth must be finite and positive")
    if isinstance(min_local_support, bool) or not isinstance(min_local_support, Integral) or min_local_support < 2:
        raise CausalValueError("min_local_support must be an integer of at least two")
    if len(target_doses) == 0:
        raise CausalValueError("target_doses must not be empty")
    try:
        values = np.asarray(columns[names.index(outcome)], dtype=np.float64)
        doses = np.asarray(columns[names.index(dose)], dtype=np.float64)
        densities = np.asarray(columns[names.index(dose_density)], dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise CausalValueError("outcome, dose, and density columns must be numeric") from error
    groups = list(columns[names.index(baseline_group)])
    if any(not isinstance(group, str) or not group for group in groups):
        raise CausalValueError("baseline group labels must be non-empty strings")
    targets = np.asarray(target_doses, dtype=np.float64)
    if targets.ndim != 1 or not np.isfinite(targets).all():
        raise CausalValueError("target_doses must be a finite one-dimensional sequence")
    try:
        raw_points = _conditional_dose_response(
            values,
            doses,
            groups,
            densities,
            targets.tolist(),
            float(bandwidth),
            min_local_support=int(min_local_support),
        )
    except ValueError as error:
        raise CausalValueError(str(error)) from error
    points = tuple(
        ConditionalDoseResponsePoint(
            str(group), float(target), float(response), int(local_rows),
            float(effective_n), float(min_density), float(max_weight), float(local_sd),
        )
        for group, target, response, local_rows, effective_n, min_density, max_weight, local_sd
        in raw_points
    )
    return ConditionalDoseResponseEstimate(
        points, float(bandwidth), density_provenance
    )


def evaluate_policy(
    evaluation_data: Any,
    *,
    outcome: str,
    assignment: Sequence[bool],
    propensity: float | Sequence[float],
    policy: BinaryPolicy,
    reference: BinaryPolicy | None = None,
    available: Sequence[bool] | None = None,
) -> PolicyEvaluation:
    """Evaluate precomputed recommendations on randomized evaluation rows.

    Supply outcomes and assignment from data held out from policy selection.
    Assignment propensities must be known and positive for both actions. This
    estimates value with the Horvitz--Thompson policy score and reports no
    interval; it does not train or select the policy.
    """

    if not isinstance(policy, BinaryPolicy):
        raise CausalTypeError("policy must be a BinaryPolicy")
    if reference is not None and not isinstance(reference, BinaryPolicy):
        raise CausalTypeError("reference must be a BinaryPolicy or None")
    if not isinstance(outcome, str) or not outcome.strip():
        raise CausalValueError("outcome must be a non-empty variable name")
    names, columns = as_columns(evaluation_data)
    try:
        values = np.asarray(columns[names.index(outcome)], dtype=np.float64)
    except ValueError as error:
        raise CausalValueError(f"outcome column {outcome!r} is missing from evaluation_data") from error
    n = len(values)
    if len(policy.actions) != n or len(assignment) != n:
        raise CausalValueError("policy actions and assignment must match evaluation row count")
    reference_actions = list(reference.actions) if reference else [False] * n
    if len(reference_actions) != n:
        raise CausalValueError("reference policy actions must match evaluation row count")
    if any(not isinstance(value, (bool, np.bool_)) for value in assignment):
        raise CausalValueError("assignment entries must be bool values")
    if available is not None and any(
        not isinstance(value, (bool, np.bool_)) for value in available
    ):
        raise CausalValueError("available entries must be bool values")
    probability_values = [float(propensity)] if np.isscalar(propensity) else list(propensity)
    cost_values = [float(policy.costs)] if np.isscalar(policy.costs) else list(policy.costs)
    capacity = policy.capacity
    if policy.max_treatment_rate is not None:
        rate_capacity = floor(policy.max_treatment_rate * n)
        capacity = rate_capacity if capacity is None else min(capacity, rate_capacity)
    reference_capacity = None if reference is None else reference.capacity
    if reference is not None and reference.max_treatment_rate is not None:
        rate_capacity = floor(reference.max_treatment_rate * n)
        reference_capacity = (
            rate_capacity
            if reference_capacity is None
            else min(reference_capacity, rate_capacity)
        )
    raw = _evaluate_binary_policy(
        values,
        [bool(value) for value in assignment],
        list(policy.actions),
        np.asarray(probability_values, dtype=np.float64),
        reference=reference_actions,
        costs=cost_values,
        reference_costs=(
            None
            if reference is None
            else [float(reference.costs)]
            if np.isscalar(reference.costs)
            else list(reference.costs)
        ),
        available=None if available is None else [bool(value) for value in available],
        capacity=capacity,
        budget=policy.budget,
        reference_capacity=reference_capacity,
        reference_budget=None if reference is None else reference.budget,
    )
    return PolicyEvaluation(*map(float, raw))


def _prediction_ownership(
    evaluation_subject_ids: Sequence[str],
    n: int,
    training_subject_ids: Sequence[str] | None,
    fold_ids: Sequence[int] | None,
    prediction_excluded_fold_ids: Sequence[int] | None,
) -> str:
    evaluation_ids = list(evaluation_subject_ids)
    if len(evaluation_ids) != n or any(not isinstance(value, str) or not value for value in evaluation_ids):
        raise CausalValueError("evaluation_subject_ids must contain one non-empty ID per evaluation row")
    if len(set(evaluation_ids)) != n:
        raise CausalValueError("evaluation_subject_ids must be unique for row-level standard errors")
    if training_subject_ids is not None:
        if fold_ids is not None or prediction_excluded_fold_ids is not None:
            raise CausalValueError("supply held-out training IDs or fold metadata, not both")
        training_ids = list(training_subject_ids)
        if not training_ids or any(not isinstance(value, str) or not value for value in training_ids):
            raise CausalValueError("training_subject_ids must contain non-empty IDs")
        overlap = set(evaluation_ids).intersection(training_ids)
        if overlap:
            raise CausalValueError("held-out ownership failure: training and evaluation subject IDs overlap")
        return "held_out_disjoint_subject_ids"
    if fold_ids is None or prediction_excluded_fold_ids is None:
        raise CausalValueError(
            "provide disjoint training_subject_ids or fold_ids and prediction_excluded_fold_ids"
        )
    folds = list(fold_ids)
    excluded = list(prediction_excluded_fold_ids)
    if len(folds) != n or len(excluded) != n:
        raise CausalValueError("fold ownership vectors must match evaluation rows")
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) or value < 0
        for value in (*folds, *excluded)
    ):
        raise CausalValueError("fold ownership IDs must be non-negative integers")
    if len(set(folds)) < 2:
        raise CausalValueError("cross-fitting requires at least two evaluation folds")
    if folds != excluded:
        raise CausalValueError(
            "cross-fit ownership failure: each prediction must exclude its evaluation row's fold"
        )
    return "caller_declared_cross_fitted_excluded_fold_ids"


def evaluate_policy_doubly_robust(
    evaluation_data: Any,
    *,
    outcome: str,
    assignment: Sequence[bool],
    propensity: float | Sequence[float],
    mu0: Sequence[float],
    mu1: Sequence[float],
    policy: BinaryPolicy,
    evaluation_subject_ids: Sequence[str],
    training_subject_ids: Sequence[str] | None = None,
    fold_ids: Sequence[int] | None = None,
    prediction_excluded_fold_ids: Sequence[int] | None = None,
    reference: BinaryPolicy | None = None,
    available: Sequence[bool] | None = None,
) -> DoublyRobustPolicyEvaluation:
    """Evaluate a fixed binary policy using randomized AIPW scores.

    Nuisance predictions must be from training subjects disjoint from evaluation
    subjects, or caller-declared fold-specific fits that exclude each evaluated
    fold. The metadata checks ownership shape/overlap but cannot prove which rows
    the nuisance models actually used.
    """

    if not isinstance(policy, BinaryPolicy):
        raise CausalTypeError("policy must be a BinaryPolicy")
    if reference is not None and not isinstance(reference, BinaryPolicy):
        raise CausalTypeError("reference must be a BinaryPolicy or None")
    ownership = _prediction_ownership(
        evaluation_subject_ids, len(policy.actions), training_subject_ids,
        fold_ids, prediction_excluded_fold_ids,
    )
    # Reuse the native HT path for the established recommendation, availability,
    # capacity, and budget validation contract.
    evaluate_policy(
        evaluation_data,
        outcome=outcome,
        assignment=assignment,
        propensity=propensity,
        policy=policy,
        reference=reference,
        available=available,
    )
    names, columns = as_columns(evaluation_data)
    try:
        values = np.asarray(columns[names.index(outcome)], dtype=np.float64)
    except ValueError as error:
        raise CausalValueError(f"outcome column {outcome!r} is missing from evaluation_data") from error
    n = len(values)
    mu0_values = np.asarray(mu0, dtype=np.float64)
    mu1_values = np.asarray(mu1, dtype=np.float64)
    if mu0_values.shape != (n,) or mu1_values.shape != (n,):
        raise CausalValueError("mu0 and mu1 must have one finite prediction per evaluation row")
    if not np.isfinite(mu0_values).all() or not np.isfinite(mu1_values).all():
        raise CausalValueError("mu0 and mu1 predictions must be finite")
    probability_values = np.asarray(
        [float(propensity)] if np.isscalar(propensity) else list(propensity), dtype=np.float64
    )
    raw = _evaluate_binary_policy_doubly_robust(
        values,
        [bool(value) for value in assignment],
        list(policy.actions),
        probability_values,
        mu0_values,
        mu1_values,
        reference=list(reference.actions) if reference is not None else None,
        costs=[float(policy.costs)] if np.isscalar(policy.costs) else list(policy.costs),
        reference_costs=(
            None if reference is None else
            [float(reference.costs)] if np.isscalar(reference.costs) else list(reference.costs)
        ),
    )
    return DoublyRobustPolicyEvaluation(
        *map(float, raw[:6]),
        *map(float, raw[6:9]),
        prediction_ownership=ownership,
        propensity_min=float(raw[9]),
        propensity_max=float(raw[10]),
    )


def evaluate_multi_action_policy(
    evaluation_data: Any,
    *,
    outcome: str,
    assignment: Sequence[str],
    propensities: Sequence[Sequence[float]],
    policy: MultiActionPolicy,
    reference: MultiActionPolicy | None = None,
    available: Sequence[Sequence[bool]] | None = None,
) -> PolicyEvaluation:
    """Evaluate fixed multi-action recommendations on randomized held-out rows.

    The result is an HT point estimate. It does not fit the score matrix,
    estimate CATEs, enforce a train/test split, or publish an interval.
    """
    if not isinstance(policy, MultiActionPolicy):
        raise CausalTypeError("policy must be a MultiActionPolicy")
    if reference is not None and not isinstance(reference, MultiActionPolicy):
        raise CausalTypeError("reference must be a MultiActionPolicy or None")
    if reference is not None and tuple(reference.action_labels) != tuple(policy.action_labels):
        raise CausalValueError("policy and reference action_labels must match")
    if not isinstance(outcome, str) or not outcome.strip():
        raise CausalValueError("outcome must be a non-empty variable name")
    names, columns = as_columns(evaluation_data)
    try:
        values = np.asarray(columns[names.index(outcome)], dtype=np.float64)
    except ValueError as error:
        raise CausalValueError(f"outcome column {outcome!r} is missing from evaluation_data") from error
    n = len(values)
    labels = tuple(policy.action_labels)
    if len(assignment) != n or len(policy.recommendations) != n:
        raise CausalValueError("assignment and recommendations must match evaluation row count")
    if any(action not in labels for action in assignment):
        raise CausalValueError("assignment contains an undeclared action label")
    reference_actions = (
        [labels[0]] * n if reference is None else list(reference.recommendations)
    )
    if len(reference_actions) != n:
        raise CausalValueError("reference recommendations must match evaluation row count")
    probs = np.asarray(propensities, dtype=np.float64)
    if probs.shape != (n, len(labels)):
        raise CausalValueError("propensities must have one column per action and one row per unit")
    availability = (
        [[True] * len(labels) for _ in range(n)] if available is None else [list(row) for row in available]
    )
    if len(availability) != n or any(
        len(row) != len(labels) or any(not isinstance(v, (bool, np.bool_)) for v in row)
        for row in availability
    ):
        raise CausalValueError("available must contain one bool per evaluation row and action")
    policy_capacities = list(policy.capacities or [n] * len(labels))
    reference_capacities = (
        [n] * len(labels)
        if reference is None or reference.capacities is None
        else list(reference.capacities)
    )
    encoded_assignment = [labels.index(action) for action in assignment]
    encoded_actions = [labels.index(action) for action in policy.recommendations]
    encoded_reference = [labels.index(action) for action in reference_actions]
    raw = _evaluate_multi_action_policy(
        values,
        encoded_assignment,
        encoded_actions,
        probs,
        reference=encoded_reference,
        costs=list(policy.costs or [0.0] * len(labels)),
        reference_costs=(
            None
            if reference is None
            else list(reference.costs or [0.0] * len(labels))
        ),
        available=availability,
        capacities=policy_capacities,
        reference_capacities=reference_capacities,
        budget=policy.budget,
        reference_budget=None if reference is None else reference.budget,
    )
    policy_value, reference_value, incremental, treatment_rate, total_cost = map(float, raw)
    return PolicyEvaluation(
        policy_value,
        reference_value,
        incremental,
        reference_value - policy_value,
        treatment_rate,
        total_cost,
    )


__all__ = [
    "BinaryPolicy",
    "MultiActionPolicy",
    "PolicyEvaluation",
    "DoublyRobustPolicyEvaluation",
    "UpliftBin",
    "ConditionalDoseResponsePoint",
    "ConditionalDoseResponseEstimate",
    "evaluate_multi_action_policy",
    "evaluate_policy",
    "evaluate_policy_doubly_robust",
    "uplift_by_score",
    "estimate_continuous_dose_response",
]
