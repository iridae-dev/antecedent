"""Point-only survival summaries for individually randomized two-arm studies.

The estimators use event-time risk sets and support right censoring, with an
explicit marginal independent-entry path for left-truncated observations.
They add no support-matrix license, conditional observation adjustment, or
interval inference.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np

from ._data import as_columns
from ._native import (
    randomized_cumulative_incidence as _randomized_cumulative_incidence,
)
from ._native import (
    randomized_cumulative_incidence_delayed_entry as _randomized_cumulative_incidence_delayed_entry,
)
from ._native import randomized_cumulative_incidence_ipcw as _randomized_cumulative_incidence_ipcw
from ._native import (
    randomized_survival as _randomized_survival,
)
from ._native import (
    randomized_survival_delayed_entry as _randomized_survival_delayed_entry,
)
from ._native import randomized_survival_ipcw as _randomized_survival_ipcw
from .errors import CausalValueError
from .observation import IndependentGiven


def _binary(values: Sequence[Any], name: str) -> list[bool]:
    result: list[bool] = []
    for value in values:
        if isinstance(value, (bool, np.bool_)) or (
            isinstance(value, (int, float, np.integer, np.floating)) and value in (0, 1)
        ):
            result.append(bool(value))
        else:
            raise CausalValueError(f"{name} values must be bool or encoded as 0/1")
    return result


def _validate_entry_contract(
    delayed_entry: str | None, assumption: IndependentGiven | None
) -> None:
    if delayed_entry is None:
        if assumption is not None and (
            not isinstance(assumption, IndependentGiven) or tuple(assumption.variables)
        ):
            raise CausalValueError(
                "the unadjusted survival route accepts only marginal IndependentGiven(())"
            )
        return
    if not isinstance(delayed_entry, str) or not delayed_entry.strip():
        raise CausalValueError("delayed_entry must be a non-empty column name")
    if not isinstance(assumption, IndependentGiven):
        raise CausalValueError(
            "delayed entry requires an explicit IndependentGiven observation assumption"
        )
    if tuple(assumption.variables):
        raise CausalValueError(
            "conditional delayed entry is unsupported; IndependentGiven must be empty"
        )


@dataclass(frozen=True, slots=True)
class KnownCensoringSurvival:
    """Known censoring survival columns on a shared grid through ``tau``.

    Columns contain one probability per subject and remain aligned with the
    data during prepared refresh. The caller owns the censoring model and its
    independent censoring claim; no model is fitted by this route.
    """

    times: tuple[float, ...]
    columns: tuple[str, ...]
    minimum_probability: float = 0.01

    def __post_init__(self) -> None:
        times = tuple(float(value) for value in self.times)
        columns = tuple(self.columns)
        if (len(times) < 2 or len(columns) != len(times) or times[0] != 0.0
            or any(not np.isfinite(value) for value in times)
            or any(right <= left for left, right in zip(times, times[1:], strict=False))
            or any(not isinstance(name, str) or not name.strip() for name in columns)
            or len(set(columns)) != len(columns)
            or not np.isfinite(self.minimum_probability)
            or not 0.0 < self.minimum_probability <= 1.0):
            raise CausalValueError("known censoring requires aligned increasing times, distinct columns, and a positive probability floor")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "columns", columns)


@dataclass(frozen=True, slots=True)
class SurvivalOutcome:
    """A two-arm randomized survival query using right-censored follow-up.

    Set ``randomized=True`` only for individual random assignment. The causal
    interpretation additionally assumes independent censoring within arm,
    consistency, and no interference. With ``delayed_entry``, the query also
    requires the existing observation contract's empty ``IndependentGiven``
    assumption; the data cannot establish these claims.
    """

    duration: str
    event_observed: str
    treatment: str
    tau: float
    randomized: bool = False
    delayed_entry: str | None = None
    observation_assumption: IndependentGiven | None = None
    known_censoring: KnownCensoringSurvival | None = None

    def __post_init__(self) -> None:
        for name in ("duration", "event_observed", "treatment"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise CausalValueError(f"{name} must be a non-empty column name")
        if len({self.duration, self.event_observed, self.treatment}) != 3:
            raise CausalValueError("duration, event_observed, and treatment columns must be distinct")
        if isinstance(self.tau, bool) or not isinstance(self.tau, (int, float)) or not np.isfinite(self.tau) or self.tau <= 0:
            raise CausalValueError("tau must be finite and positive")
        if not isinstance(self.randomized, bool):
            raise CausalValueError("randomized must be bool")
        if self.known_censoring is None:
            _validate_entry_contract(self.delayed_entry, self.observation_assumption)
        else:
            if not isinstance(self.known_censoring, KnownCensoringSurvival):
                raise CausalValueError("known_censoring must be KnownCensoringSurvival")
            if self.delayed_entry is not None:
                raise CausalValueError("known censoring survival does not combine with delayed entry")
            if not isinstance(self.observation_assumption, IndependentGiven):
                raise CausalValueError("known censoring requires an explicit IndependentGiven observation assumption")
            if abs(self.known_censoring.times[-1] - self.tau) > 1e-10:
                raise CausalValueError("known censoring time grid must end at tau")
        if self.delayed_entry in {self.duration, self.event_observed, self.treatment}:
            raise CausalValueError("delayed_entry must name a distinct column")


@dataclass(frozen=True, slots=True)
class SurvivalEstimate:
    """Kaplan-Meier step curves and RMST values, without interval estimates."""

    times: tuple[float, ...]
    control_survival: tuple[float, ...]
    treated_survival: tuple[float, ...]
    rmst_control: float
    rmst_treated: float
    rmst_difference: float
    tau: float
    uncertainty: str = "point_only"
    support_status: str = "unlicensed_point_utility"
    assumptions: tuple[str, ...] = (
        "individual_random_assignment",
        "independent_right_censoring_within_arm",
        "consistency",
        "no_interference",
    )


@dataclass(frozen=True, slots=True)
class IPCWSurvivalEstimate:
    """Caller-censoring-weighted survival/RMST point estimates (no intervals)."""

    times: tuple[float, ...]
    control_survival: tuple[float, ...]
    treated_survival: tuple[float, ...]
    rmst_control: float
    rmst_treated: float
    rmst_difference: float
    tau: float
    minimum_censoring_survival: float
    minimum_event_risk_set_control: int | None
    minimum_event_risk_set_treated: int | None
    censoring_survival_provenance: str = "caller_supplied_not_fitted_or_verified"
    uncertainty: str = "point_only"
    support_status: str = "unlicensed_point_utility"
    assumptions: tuple[str, ...] = (
        "individual_random_assignment",
        "correct_caller_supplied_conditional_censoring_survival",
        "independent_censoring_given_supplied_history",
        "sequential_censoring_positivity",
        "consistency",
        "no_interference",
    )


@dataclass(frozen=True, slots=True)
class IPCWCumulativeIncidenceEstimate:
    """Caller-censoring-weighted cause-specific CIF point estimates."""

    target_cause: int
    times: tuple[float, ...]
    control_incidence: tuple[float, ...]
    treated_incidence: tuple[float, ...]
    incidence_difference: float
    tau: float
    minimum_censoring_survival: float
    minimum_event_risk_set_control: int | None
    minimum_event_risk_set_treated: int | None
    censoring_survival_provenance: str = "caller_supplied_not_fitted_or_verified"
    uncertainty: str = "point_only"
    support_status: str = "unlicensed_point_utility"
    assumptions: tuple[str, ...] = (
        "individual_random_assignment",
        "all_event_causes_coded_distinctly",
        "correct_caller_supplied_conditional_censoring_survival",
        "independent_censoring_given_supplied_history",
        "sequential_censoring_positivity",
        "consistency",
        "no_interference",
    )


@dataclass(frozen=True, slots=True)
class CompetingRisksOutcome:
    """A two-arm randomized competing-risks query with coded event causes.

    ``event_cause == 0`` denotes right censoring; every positive integer is a
    distinct event cause. At least two event causes must be observed. The
    causal interpretation assumes all competing causes are captured and
    right censoring is independent within arm. Delayed entry uses the same
    explicit marginal ``IndependentGiven`` observation contract.
    """

    duration: str
    event_cause: str
    treatment: str
    target_cause: int
    tau: float
    randomized: bool = False
    delayed_entry: str | None = None
    observation_assumption: IndependentGiven | None = None
    known_censoring: KnownCensoringSurvival | None = None

    def __post_init__(self) -> None:
        for name in ("duration", "event_cause", "treatment"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise CausalValueError(f"{name} must be a non-empty column name")
        if len({self.duration, self.event_cause, self.treatment}) != 3:
            raise CausalValueError("duration, event_cause, and treatment columns must be distinct")
        if (
            isinstance(self.target_cause, bool)
            or not isinstance(self.target_cause, Integral)
            or self.target_cause <= 0
            or self.target_cause > np.iinfo(np.int64).max
        ):
            raise CausalValueError("target_cause must be a positive integer event code")
        if isinstance(self.tau, bool) or not isinstance(self.tau, (int, float)) or not np.isfinite(self.tau) or self.tau <= 0:
            raise CausalValueError("tau must be finite and positive")
        if not isinstance(self.randomized, bool):
            raise CausalValueError("randomized must be bool")
        if self.known_censoring is None:
            _validate_entry_contract(self.delayed_entry, self.observation_assumption)
        else:
            if not isinstance(self.known_censoring, KnownCensoringSurvival):
                raise CausalValueError("known_censoring must be KnownCensoringSurvival")
            if self.delayed_entry is not None:
                raise CausalValueError("known censoring survival does not combine with delayed entry")
            if not isinstance(self.observation_assumption, IndependentGiven):
                raise CausalValueError("known censoring requires an explicit IndependentGiven observation assumption")
            if abs(self.known_censoring.times[-1] - self.tau) > 1e-10:
                raise CausalValueError("known censoring time grid must end at tau")
        if self.delayed_entry in {self.duration, self.event_cause, self.treatment}:
            raise CausalValueError("delayed_entry must name a distinct column")


@dataclass(frozen=True, slots=True)
class CumulativeIncidenceEstimate:
    """Arm-specific Aalen–Johansen CIF step curves, without intervals."""

    target_cause: int
    times: tuple[float, ...]
    control_incidence: tuple[float, ...]
    treated_incidence: tuple[float, ...]
    incidence_difference: float
    tau: float
    uncertainty: str = "point_only"
    support_status: str = "unlicensed_point_utility"
    assumptions: tuple[str, ...] = (
        "individual_random_assignment",
        "all_event_causes_coded_distinctly",
        "independent_right_censoring_within_arm",
        "consistency",
        "no_interference",
    )


def estimate_survival(data: Any, query: SurvivalOutcome) -> SurvivalEstimate:
    """Estimate arm-specific survival curves and RMST through ``tau`` in Rust.

    Censored rows contribute to risk sets until their censoring time. Censoring
    is assumed independent of event time within each arm. Non-randomized
    assignments, conditional delayed entry, time-varying censoring adjustment,
    and competing-risk codes are refused by this binary-event API. Use
    :func:`estimate_cumulative_incidence` for coded competing events.
    """

    if not isinstance(query, SurvivalOutcome):
        raise CausalValueError("query must be a SurvivalOutcome")
    if query.known_censoring is not None:
        raise CausalValueError("known censoring survival belongs in analyze(data, query=...) for retained IPCW execution")
    if not query.randomized:
        raise CausalValueError("causal survival estimates currently require declared individual random assignment")
    names, columns = as_columns(data)
    required = [query.duration, query.event_observed, query.treatment]
    if query.delayed_entry is not None:
        required.append(query.delayed_entry)
    for name in required:
        if name not in names:
            raise CausalValueError(f"required survival column {name!r} is missing")
    duration = np.asarray(columns[names.index(query.duration)], dtype=np.float64)
    events = list(columns[names.index(query.event_observed)])
    treatment = list(columns[names.index(query.treatment)])
    if len(events) != len(duration) or len(treatment) != len(duration):
        raise CausalValueError("duration, event_observed, and treatment columns must have equal lengths")
    event_values = _binary(events, "event_observed")
    treated_values = _binary(treatment, "treatment")
    try:
        if query.delayed_entry is None:
            times, control, treated, rmst_control, rmst_treated = _randomized_survival(
                duration, event_values, treated_values, float(query.tau)
            )
        else:
            entry = np.asarray(columns[names.index(query.delayed_entry)], dtype=np.float64)
            times, control, treated, rmst_control, rmst_treated = (
                _randomized_survival_delayed_entry(
                    duration, entry, event_values, treated_values, float(query.tau)
                )
            )
    except ValueError as error:
        raise CausalValueError(str(error)) from error
    return SurvivalEstimate(
        tuple(float(value) for value in times),
        tuple(float(value) for value in control),
        tuple(float(value) for value in treated),
        float(rmst_control),
        float(rmst_treated),
        float(rmst_treated - rmst_control),
        float(query.tau),
        assumptions=(
            (
                "individual_random_assignment",
                "independent_right_censoring_within_arm",
                "consistency",
                "no_interference",
            )
            if query.delayed_entry is None
            else (
                "individual_random_assignment",
                "independent_right_censoring_within_arm",
                "independent_left_truncation_given_IndependentGiven_empty",
                "consistency",
                "no_interference",
            )
        ),
    )


def estimate_cumulative_incidence(
    data: Any, query: CompetingRisksOutcome
) -> CumulativeIncidenceEstimate:
    """Estimate cause-specific cumulative-incidence curves in native Rust.

    Code zero denotes right censoring and each positive code denotes one
    distinct event type. At each event time the estimator updates the target
    cause's incidence using the all-cause event-free survival immediately
    before that time, so other event causes compete in the risk set instead
    of being misclassified as ordinary censoring. An optional delayed-entry
    column requires ``IndependentGiven(())`` and uses the strict interval
    ``(entry, duration]`` for event-time risk sets.
    """

    if not isinstance(query, CompetingRisksOutcome):
        raise CausalValueError("query must be a CompetingRisksOutcome")
    if query.known_censoring is not None:
        raise CausalValueError("known censoring survival belongs in analyze(data, query=...) for retained IPCW execution")
    if not query.randomized:
        raise CausalValueError("causal cumulative incidence requires declared individual random assignment")
    names, columns = as_columns(data)
    required = [query.duration, query.event_cause, query.treatment]
    if query.delayed_entry is not None:
        required.append(query.delayed_entry)
    for name in required:
        if name not in names:
            raise CausalValueError(f"required competing-risks column {name!r} is missing")
    duration = np.asarray(columns[names.index(query.duration)], dtype=np.float64)
    raw_causes = list(columns[names.index(query.event_cause)])
    treatment = list(columns[names.index(query.treatment)])
    if len(raw_causes) != len(duration) or len(treatment) != len(duration):
        raise CausalValueError("duration, event_cause, and treatment columns must have equal lengths")
    causes: list[int] = []
    for value in raw_causes:
        if isinstance(value, (bool, np.bool_)):
            raise CausalValueError("event_cause values must be non-negative integer codes; zero denotes censoring")
        if isinstance(value, (int, np.integer)) or (
            isinstance(value, (float, np.floating)) and np.isfinite(value) and value.is_integer()
        ):
            code = int(value)
        else:
            raise CausalValueError("event_cause values must be non-negative integer codes; zero denotes censoring")
        if code < 0 or code > np.iinfo(np.int64).max:
            raise CausalValueError("event_cause values must fit in non-negative 64-bit integer codes")
        causes.append(code)
    treated_values = _binary(treatment, "treatment")
    try:
        if query.delayed_entry is None:
            times, control, treated = _randomized_cumulative_incidence(
                duration, causes, treated_values, float(query.tau), int(query.target_cause)
            )
        else:
            entry = np.asarray(columns[names.index(query.delayed_entry)], dtype=np.float64)
            times, control, treated = _randomized_cumulative_incidence_delayed_entry(
                duration, entry, causes, treated_values, float(query.tau), int(query.target_cause)
            )
    except ValueError as error:
        raise CausalValueError(str(error)) from error
    return CumulativeIncidenceEstimate(
        int(query.target_cause),
        tuple(float(value) for value in times),
        tuple(float(value) for value in control),
        tuple(float(value) for value in treated),
        float(treated[-1] - control[-1]),
        float(query.tau),
        assumptions=(
            (
                "individual_random_assignment",
                "all_event_causes_coded_distinctly",
                "independent_right_censoring_within_arm",
                "consistency",
                "no_interference",
            )
            if query.delayed_entry is None
            else (
                "individual_random_assignment",
                "all_event_causes_coded_distinctly",
                "independent_right_censoring_within_arm",
                "independent_left_truncation_given_IndependentGiven_empty",
                "consistency",
                "no_interference",
            )
        ),
    )


def estimate_survival_ipcw(
    data: Any,
    query: SurvivalOutcome,
    *,
    times: Sequence[float],
    censoring_survival: Any,
    minimum_probability: float = 0.01,
) -> IPCWSurvivalEstimate:
    """Estimate weighted survival curves and RMST with supplied censoring ``G``.

    ``censoring_survival[i, j]`` must be the subject-specific probability of
    remaining uncensored immediately before ``times[j]``. The caller supplies
    and is responsible for fitting/validating these probabilities. This utility
    has no interval inference and does not support delayed entry.
    """
    if not isinstance(query, SurvivalOutcome):
        raise CausalValueError("query must be a SurvivalOutcome")
    if query.known_censoring is not None:
        raise CausalValueError("query already carries known censoring columns; use analyze(data, query=...)")
    if not query.randomized:
        raise CausalValueError("causal survival estimates currently require declared individual random assignment")
    if query.delayed_entry is not None:
        raise CausalValueError("IPCW survival currently refuses delayed-entry queries")
    names, columns = as_columns(data)
    for name in (query.duration, query.event_observed, query.treatment):
        if name not in names:
            raise CausalValueError(f"required survival column {name!r} is missing")
    duration = np.asarray(columns[names.index(query.duration)], dtype=np.float64)
    events = _binary(list(columns[names.index(query.event_observed)]), "event_observed")
    treatment = _binary(list(columns[names.index(query.treatment)]), "treatment")
    grid = np.asarray(times, dtype=np.float64)
    g = np.asarray(censoring_survival, dtype=np.float64)
    if g.ndim != 2 or g.shape[0] != len(duration):
        raise CausalValueError("censoring_survival must be a subjects-by-times matrix")
    if not np.isfinite(minimum_probability) or not 0 < minimum_probability <= 1:
        raise CausalValueError("minimum_probability must be finite and in (0, 1]")
    try:
        (returned_times, control, treated, rmst_control, rmst_treated,
         minrisk_control, minrisk_treated, minimum_g) = _randomized_survival_ipcw(
            duration, events, treatment, grid.tolist(), g, float(query.tau), float(minimum_probability)
        )
    except ValueError as error:
        raise CausalValueError(str(error)) from error
    return IPCWSurvivalEstimate(
        tuple(float(value) for value in returned_times),
        tuple(float(value) for value in control),
        tuple(float(value) for value in treated),
        float(rmst_control),
        float(rmst_treated),
        float(rmst_treated - rmst_control),
        float(query.tau),
        float(minimum_g),
        minrisk_control,
        minrisk_treated,
    )


def estimate_cumulative_incidence_ipcw(
    data: Any,
    query: CompetingRisksOutcome,
    *,
    times: Sequence[float],
    censoring_survival: Any,
    minimum_probability: float = 0.01,
) -> IPCWCumulativeIncidenceEstimate:
    """Estimate an IPCW Aalen–Johansen CIF using caller-supplied censoring ``G``.

    ``G[i, j]`` is the subject-specific probability of remaining uncensored
    immediately before grid time ``times[j]``. Delayed entry is unsupported.
    The censoring model is neither fitted nor verified here; inference is
    point-only.
    """
    if not isinstance(query, CompetingRisksOutcome):
        raise CausalValueError("query must be a CompetingRisksOutcome")
    if query.known_censoring is not None:
        raise CausalValueError("query already carries known censoring columns; use analyze(data, query=...)")
    if not query.randomized:
        raise CausalValueError("causal cumulative incidence requires declared individual random assignment")
    if query.delayed_entry is not None:
        raise CausalValueError("IPCW cumulative incidence currently refuses delayed-entry queries")
    names, columns = as_columns(data)
    for name in (query.duration, query.event_cause, query.treatment):
        if name not in names:
            raise CausalValueError(f"required competing-risks column {name!r} is missing")
    duration = np.asarray(columns[names.index(query.duration)], dtype=np.float64)
    raw_causes = list(columns[names.index(query.event_cause)])
    treatment = _binary(list(columns[names.index(query.treatment)]), "treatment")
    if len(raw_causes) != len(duration):
        raise CausalValueError("duration, event_cause, and treatment columns must have equal lengths")
    causes: list[int] = []
    for value in raw_causes:
        if isinstance(value, (bool, np.bool_)):
            raise CausalValueError("event_cause values must be non-negative integer codes; zero denotes censoring")
        if isinstance(value, (int, np.integer)) or (
            isinstance(value, (float, np.floating)) and np.isfinite(value) and value.is_integer()
        ):
            code = int(value)
        else:
            raise CausalValueError("event_cause values must be non-negative integer codes; zero denotes censoring")
        if code < 0 or code > np.iinfo(np.int64).max:
            raise CausalValueError("event_cause values must fit in non-negative 64-bit integer codes")
        causes.append(code)
    g = np.asarray(censoring_survival, dtype=np.float64)
    grid = np.asarray(times, dtype=np.float64)
    if g.ndim != 2 or g.shape[0] != len(duration):
        raise CausalValueError("censoring_survival must be a subjects-by-times matrix")
    if not np.isfinite(minimum_probability) or not 0 < minimum_probability <= 1:
        raise CausalValueError("minimum_probability must be finite and in (0, 1]")
    try:
        (returned_times, control, treated, minrisk_control, minrisk_treated, minimum_g) = (
            _randomized_cumulative_incidence_ipcw(
                duration, causes, treatment, grid.tolist(), g, float(query.tau),
                int(query.target_cause), float(minimum_probability),
            )
        )
    except ValueError as error:
        raise CausalValueError(str(error)) from error
    return IPCWCumulativeIncidenceEstimate(
        int(query.target_cause),
        tuple(float(value) for value in returned_times),
        tuple(float(value) for value in control),
        tuple(float(value) for value in treated),
        float(treated[-1] - control[-1]),
        float(query.tau),
        float(minimum_g),
        minrisk_control,
        minrisk_treated,
    )


__all__ = [
    "CompetingRisksOutcome",
    "CumulativeIncidenceEstimate",
    "IPCWCumulativeIncidenceEstimate",
    "IPCWSurvivalEstimate",
    "KnownCensoringSurvival",
    "SurvivalEstimate",
    "SurvivalOutcome",
    "estimate_cumulative_incidence",
    "estimate_cumulative_incidence_ipcw",
    "estimate_survival_ipcw",
    "estimate_survival",
]
