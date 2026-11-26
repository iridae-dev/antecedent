"""Survival and competing-risk outcomes for randomized two-arm studies.

Declare a :class:`SurvivalOutcome` (survival curves and restricted mean survival time, RMST)
or a :class:`CompetingRisksOutcome` (cumulative incidence of one target cause) and pass it to
``antecedent.analyze(data, query=...)``; the answer is a :class:`SurvivalEstimate` or
:class:`CumulativeIncidenceEstimate` on ``result.value.survival``. Every contrast is
treated minus control at the horizon ``tau`` you name.

The native estimators use event-time risk sets for right censoring and an explicit marginal
independent-entry path for left truncation (delayed entry). Through ``analyze``, a
``Frequentist(bootstrap=...)`` request can add pointwise bootstrap intervals and a
simultaneous curve band when their subject-level inference gates pass; whether they meet a
``survival`` row of the graphless support matrix is reported on ``support_status``. The direct
utilities :func:`estimate_survival_ipcw` and :func:`estimate_cumulative_incidence_ipcw` report
points only and carry no support. No path fits or verifies a caller-supplied censoring model.

See the guide ``docs/guides/designs/survival-outcomes.md`` for the full workflow, and
:mod:`antecedent.observational_survival` when treatment was observed rather than randomized.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import _deprecation, _direct_status
from ._columns import coerce_fields
from ._data import as_columns
from ._native import randomized_cumulative_incidence_ipcw as _randomized_cumulative_incidence_ipcw
from ._native import randomized_survival_ipcw as _randomized_survival_ipcw
from .errors import CausalValueError
from .observation import IndependentGiven


def _binary(values: Any, name: str) -> NDArray[np.float64]:
    array = np.asarray(values)
    if array.ndim != 1:
        raise CausalValueError(f"{name} values must be a one-dimensional column")
    if array.dtype.kind not in "biuf" and not all(
        isinstance(value, (bool, np.bool_, int, float, np.integer, np.floating)) for value in array
    ):
        raise CausalValueError(f"{name} values must be bool or encoded as 0/1")
    return np.asarray(array, dtype=np.float64)


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
    """Caller-supplied censoring survival probabilities, as data columns on a grid through ``tau``.

    Pass as ``known_censoring=`` on :class:`SurvivalOutcome` or :class:`CompetingRisksOutcome`
    to weight the estimator by inverse probability of censoring (IPCW). Column ``columns[j]``
    holds, for each subject, the probability of remaining uncensored immediately before
    ``times[j]``. The columns stay aligned with the data rows when a prepared study is
    refreshed. You own the censoring model and its independent-censoring claim: no model is
    fitted, the probabilities are held fixed in every bootstrap resample and never refit, and
    the result labels them ``caller_supplied_fixed_not_fitted_or_verified``.

    Fields:

    * ``times`` -- strictly increasing finite grid starting at ``0.0``, at least two points;
      the last point must equal the query's ``tau``.
    * ``columns`` -- distinct, nonblank column names, one per grid point.
    * ``minimum_probability`` -- positivity floor in ``(0, 1]``, default ``0.01``; a supplied
      probability below it is refused when the study runs.

    Raises:
        CausalValueError: the grid, columns or floor violate the rules above.
    """

    times: tuple[float, ...]
    columns: tuple[str, ...]
    minimum_probability: float = 0.01

    def __post_init__(self) -> None:
        times = tuple(float(value) for value in self.times)
        columns = tuple(self.columns)
        if (
            len(times) < 2
            or len(columns) != len(times)
            or times[0] != 0.0
            or any(not np.isfinite(value) for value in times)
            or any(right <= left for left, right in zip(times, times[1:], strict=False))
            or any(not isinstance(name, str) or not name.strip() for name in columns)
            or len(set(columns)) != len(columns)
            or not np.isfinite(self.minimum_probability)
            or not 0.0 < self.minimum_probability <= 1.0
        ):
            raise CausalValueError(
                "known censoring requires aligned increasing times, distinct columns, and a positive probability floor"
            )
        coerce_fields(self, times=times)
        coerce_fields(self, columns=columns)


@dataclass(frozen=True, slots=True)
class StratifiedCensoringSurvival:
    """Fit a stratified reverse product-limit censoring learner on the evaluation subjects.

    Supply two binary baseline columns and the pre-entry population probabilities for entry
    stratum 0 and 1. Entry is independent of event, cause, censoring and the censoring stratum
    given arm and entry stratum. Censoring is independent of event and cause given arm and both
    strata. Inference resamples whole subjects within arms and refits the learner each time.
    """

    entry_stratum: str
    censoring_stratum: str
    population_weights: tuple[float, float]
    minimum_probability: float = 0.05

    def __post_init__(self) -> None:
        columns = (self.entry_stratum, self.censoring_stratum)
        weights = tuple(float(value) for value in self.population_weights)
        if (
            any(not isinstance(column, str) or not column.strip() for column in columns)
            or self.entry_stratum == self.censoring_stratum
            or len(weights) != 2
            or any(not np.isfinite(weight) or weight <= 0.0 for weight in weights)
            or abs(sum(weights) - 1.0) > 1e-12
            or not 0.0 < self.minimum_probability <= 1.0
        ):
            raise CausalValueError(
                "conditional survival requires distinct baseline strata, two positive "
                "source-population proportions summing to one, and a positive probability floor"
            )
        coerce_fields(self, population_weights=weights)


def _validate_stratified_censoring(query: SurvivalOutcome | CompetingRisksOutcome, event: str) -> None:
    learner = query.stratified_censoring
    if not isinstance(learner, StratifiedCensoringSurvival):
        raise CausalValueError("stratified_censoring must be StratifiedCensoringSurvival")
    strata = (learner.entry_stratum, learner.censoring_stratum)
    assumption = query.observation_assumption
    if (
        query.known_censoring is not None
        or not isinstance(assumption, IndependentGiven)
        or tuple(assumption.variables) != strata
        or any(name in (query.duration, event, query.treatment, query.delayed_entry) for name in strata)
        or (query.delayed_entry is not None and (
            not isinstance(query.delayed_entry, str) or not query.delayed_entry.strip()
        ))
    ):
        raise CausalValueError(
            "conditional survival requires distinct entry and censoring strata and their "
            "ordered IndependentGiven declaration with the fitted learner"
        )


@dataclass(frozen=True, slots=True)
class SurvivalOutcome:
    """A two-arm randomized survival query on right-censored follow-up, restricted to ``tau``.

    The estimands are the survival difference ``S_1(tau) - S_0(tau)`` and the RMST difference
    ``RMST_1(tau) - RMST_0(tau)``, where ``S_a(t) = P(T(a) > t)`` and
    ``RMST_a(tau) = E[min(T(a), tau)]``. Pass to ``antecedent.analyze(data, query=...)``; the
    answer is a :class:`SurvivalEstimate` on ``result.value.survival``.

    Fields:

    * ``duration`` -- column of event or censoring times (finite, nonnegative).
    * ``event_observed`` -- column marking an observed event (``True`` or 1) versus right
      censoring (``False`` or 0).
    * ``treatment`` -- binary arm column (1 treated, 0 control).
    * ``tau`` -- restriction horizon, finite and positive. Both arms need observed follow-up
      through it.
    * ``randomized`` -- set ``True`` only for individual random assignment. Default ``False``,
      which ``analyze`` and the IPCW utilities refuse (``route_not_supported`` from
      ``analyze``).
    * ``delayed_entry`` -- optional column of left-truncation entry times. Default ``None``.
    * ``observation_assumption`` -- an :class:`~antecedent.observation.IndependentGiven`
      declaration of how censoring (and entry) relate to the event time; ``analyze`` requires
      it. Without ``known_censoring`` it must be the marginal ``IndependentGiven(())``. With
      ``known_censoring`` it names the conditioning variables of the supplied probabilities,
      and must be empty when ``delayed_entry`` is also set. Default ``None``, which the
      constructor accepts only without ``delayed_entry`` and ``known_censoring``.
    * ``known_censoring`` -- optional :class:`KnownCensoringSurvival`. Default ``None``.

    The causal reading also assumes consistency and no interference. The data cannot
    establish any of these assumptions; the result lists them on ``assumptions``.

    Raises:
        CausalValueError: a column name is blank or repeated, ``tau`` is not finite and
            positive, ``randomized`` is not a ``bool``, the observation assumption does not
            match the rules above, or the ``known_censoring`` grid does not end at ``tau``.
    """

    duration: str
    event_observed: str
    treatment: str
    tau: float
    randomized: bool = False
    delayed_entry: str | None = None
    observation_assumption: IndependentGiven | None = None
    known_censoring: KnownCensoringSurvival | None = None
    stratified_censoring: StratifiedCensoringSurvival | None = None

    def __post_init__(self) -> None:
        for name in ("duration", "event_observed", "treatment"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise CausalValueError(f"{name} must be a non-empty column name")
        if len({self.duration, self.event_observed, self.treatment}) != 3:
            raise CausalValueError(
                "duration, event_observed, and treatment columns must be distinct"
            )
        if (
            isinstance(self.tau, bool)
            or not isinstance(self.tau, Real)
            or not np.isfinite(self.tau)
            or self.tau <= 0
        ):
            raise CausalValueError("tau must be finite and positive")
        coerce_fields(self, tau=float(self.tau))
        if not isinstance(self.randomized, bool):
            raise CausalValueError("randomized must be bool")
        if self.stratified_censoring is not None:
            _validate_stratified_censoring(self, self.event_observed)
        elif self.known_censoring is None:
            _validate_entry_contract(self.delayed_entry, self.observation_assumption)
        else:
            if not isinstance(self.known_censoring, KnownCensoringSurvival):
                raise CausalValueError("known_censoring must be KnownCensoringSurvival")
            if self.delayed_entry is not None:
                _validate_entry_contract(self.delayed_entry, self.observation_assumption)
            if not isinstance(self.observation_assumption, IndependentGiven):
                raise CausalValueError(
                    "known censoring requires an explicit IndependentGiven observation assumption"
                )
            if abs(self.known_censoring.times[-1] - self.tau) > 1e-10:
                raise CausalValueError("known censoring time grid must end at tau")
        if self.delayed_entry in {self.duration, self.event_observed, self.treatment}:
            raise CausalValueError("delayed_entry must name a distinct column")


@dataclass(frozen=True, slots=True)
class SurvivalDifferenceBand:
    """A simultaneous 95% band for the treatment-minus-control curve over the event-time grid.

    The band uses one common radius: the 95th percentile of the subject-stratified bootstrap
    suprema of centred curve differences, so its coverage statement is for the complete
    reported grid at once. The scalar RMST and fixed-horizon intervals are a separate pointwise
    construction. Read it from ``SurvivalEstimate.difference_band`` or
    ``CumulativeIncidenceEstimate.difference_band``.

    Fields:

    * ``times`` -- the observed event-time grid, including zero and the horizon.
    * ``difference`` -- the treatment-minus-control curve on that grid.
    * ``lower``, ``upper`` -- simultaneous band endpoints at each grid time.
    * ``replicates_ok`` -- bootstrap replicates that satisfied the estimator's support
      contract.
    * ``level`` -- ``0.95``.
    * ``method`` -- ``"subject_stratified_bootstrap_supremum"``.
    * ``support_status`` -- the support-matrix cell status (``"supported"``) when the band and
      its arm counts match a band row of the graphless support matrix; otherwise
      ``"unsupported_simultaneous_band"``.
    """

    times: tuple[float, ...]
    difference: tuple[float, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    replicates_ok: int
    level: float = 0.95
    method: str = "subject_stratified_bootstrap_supremum"
    support_status: str = _direct_status.UNSUPPORTED_SIMULTANEOUS_BAND
    minimum_horizon_risk_set: int | None = None


@dataclass(frozen=True, slots=True)
class SurvivalEstimate:
    """Arm survival curves and RMST from ``analyze`` with a :class:`SurvivalOutcome`.

    The curves are Kaplan-Meier (product-limit) step functions; with ``known_censoring`` they
    are inverse-probability-of-censoring weighted, and with ``delayed_entry`` they use
    left-truncated risk sets. Read ``rmst_difference`` and the curves first, then
    ``support_status`` and the intervals.

    Fields:

    * ``times`` -- event-time grid, including zero and ``tau``.
    * ``control_survival``, ``treated_survival`` -- arm survival at each grid time.
    * ``rmst_control``, ``rmst_treated`` -- restricted mean survival time up to ``tau``.
    * ``rmst_difference`` -- ``rmst_treated - rmst_control``.
    * ``tau`` -- the restriction horizon.
    * ``uncertainty`` -- ``"subject_stratified_percentile_bootstrap_pointwise_95"`` when the
      intervals were computed, otherwise ``"point_only_no_interval"``.
    * ``rmst_difference_interval`` -- pointwise 95% percentile bootstrap interval, or
      ``None``.
    * ``survival_at_tau_difference_interval`` -- pointwise 95% interval for
      ``S_1(tau) - S_0(tau)``, or ``None``. The two intervals cover their contrasts
      separately.
    * ``bootstrap_replicates_requested``, ``bootstrap_replicates_ok`` -- requested and valid
      bootstrap replicates, or ``None`` without a bootstrap.
    * ``censoring_survival_provenance`` --
      ``"caller_supplied_fixed_not_fitted_or_verified"`` with ``known_censoring``, else
      ``None``.
    * ``difference_band`` -- a :class:`SurvivalDifferenceBand`, or ``None``.
    * ``band_unavailable_reason`` -- why a bootstrap request produced no band, or ``None``.
    * ``support_status`` -- ``"supported"`` when the paired RMST and horizon intervals meet a
      ``survival`` row of the graphless support matrix; otherwise
      ``"supported_simultaneous_band"``, ``"unsupported_simultaneous_band"``,
      ``"unsupported_pointwise_interval"`` or ``"unsupported_point_utility"``, describing the
      most that was published.
    * ``assumptions`` -- the declared, unverified assumptions the causal reading rests on.
    """

    times: tuple[float, ...]
    control_survival: tuple[float, ...]
    treated_survival: tuple[float, ...]
    rmst_control: float
    rmst_treated: float
    rmst_difference: float
    tau: float
    uncertainty: str = "point_only"
    rmst_difference_interval: tuple[float, float] | None = None
    survival_at_tau_difference_interval: tuple[float, float] | None = None
    bootstrap_replicates_requested: int | None = None
    bootstrap_replicates_ok: int | None = None
    censoring_survival_provenance: str | None = None
    difference_band: SurvivalDifferenceBand | None = None
    band_unavailable_reason: str | None = None
    support_status: str = _direct_status.UNSUPPORTED_POINT_UTILITY
    assumptions: tuple[str, ...] = (
        "individual_random_assignment",
        "independent_right_censoring_within_arm",
        "consistency",
        "no_interference",
    )


@dataclass(frozen=True, slots=True)
class IPCWSurvivalEstimate:
    """Point estimates from :func:`estimate_survival_ipcw`: weighted survival curves and RMST.

    Fields as :class:`SurvivalEstimate` for ``times``, the curves, the RMST values and
    ``tau``, plus:

    * ``minimum_censoring_survival`` -- the smallest supplied censoring probability.
    * ``minimum_event_risk_set_control``, ``minimum_event_risk_set_treated`` -- the smallest
      risk set at an event time in each arm, or ``None``.
    * ``censoring_survival_provenance`` -- always
      ``"caller_supplied_not_fitted_or_verified"``.
    * ``uncertainty`` -- always ``"point_only"``; there are no intervals.
    * ``support_status`` -- always ``"unsupported_point_utility"``.
    * ``assumptions`` -- the declared, unverified assumptions, including correct supplied
      censoring probabilities and censoring positivity.

    The minimum probability and risk-set sizes are diagnostics, not inference.
    """

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
    support_status: str = _direct_status.UNSUPPORTED_POINT_UTILITY
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
    """Point estimates from :func:`estimate_cumulative_incidence_ipcw`.

    Weighted arm cumulative incidence functions (CIF) of ``target_cause`` and their difference
    at ``tau``.

    Fields:

    * ``target_cause`` -- the event code whose incidence is reported.
    * ``times`` -- the time grid, from zero through ``tau``.
    * ``control_incidence``, ``treated_incidence`` -- arm cumulative incidence at each time.
    * ``incidence_difference`` -- treated minus control at ``tau``.
    * ``tau``, ``minimum_censoring_survival``, ``minimum_event_risk_set_control``,
      ``minimum_event_risk_set_treated``, ``censoring_survival_provenance``, ``uncertainty``
      and ``support_status`` -- as :class:`IPCWSurvivalEstimate`.
    * ``assumptions`` -- the declared, unverified assumptions, including complete and
      distinct cause coding.
    """

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
    support_status: str = _direct_status.UNSUPPORTED_POINT_UTILITY
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
    """A two-arm randomized competing-risks query: cumulative incidence of one target cause.

    ``event_cause == 0`` denotes right censoring; every positive integer is a distinct event
    cause. Other causes are competing events that remove a subject from the event-free risk
    set; they are never treated as censoring. The estimand is the difference in target-cause
    cumulative incidence at ``tau``, by the Aalen-Johansen recursion. Pass to
    ``antecedent.analyze(data, query=...)``; the answer is a
    :class:`CumulativeIncidenceEstimate` on ``result.value.survival``.

    Fields are as :class:`SurvivalOutcome`, with ``event_cause`` (the column of nonnegative
    integer cause codes) in place of ``event_observed`` and:

    * ``target_cause`` -- the positive integer code whose incidence is reported. At least two
      distinct causes must be observed and the target must be among them; this is checked
      when the study runs.

    The causal reading assumes random assignment, complete and distinct cause coding,
    independent right censoring within arm, consistency and no interference.

    Raises:
        CausalValueError: as :class:`SurvivalOutcome`, or ``target_cause`` is not a positive
            integer that fits in 64 bits.
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
    stratified_censoring: StratifiedCensoringSurvival | None = None

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
        if (
            isinstance(self.tau, bool)
            or not isinstance(self.tau, Real)
            or not np.isfinite(self.tau)
            or self.tau <= 0
        ):
            raise CausalValueError("tau must be finite and positive")
        coerce_fields(self, tau=float(self.tau))
        if not isinstance(self.randomized, bool):
            raise CausalValueError("randomized must be bool")
        if self.stratified_censoring is not None:
            _validate_stratified_censoring(self, self.event_cause)
        elif self.known_censoring is None:
            _validate_entry_contract(self.delayed_entry, self.observation_assumption)
        else:
            if not isinstance(self.known_censoring, KnownCensoringSurvival):
                raise CausalValueError("known_censoring must be KnownCensoringSurvival")
            if self.delayed_entry is not None:
                _validate_entry_contract(self.delayed_entry, self.observation_assumption)
            if not isinstance(self.observation_assumption, IndependentGiven):
                raise CausalValueError(
                    "known censoring requires an explicit IndependentGiven observation assumption"
                )
            if abs(self.known_censoring.times[-1] - self.tau) > 1e-10:
                raise CausalValueError("known censoring time grid must end at tau")
        if self.delayed_entry in {self.duration, self.event_cause, self.treatment}:
            raise CausalValueError("delayed_entry must name a distinct column")


@dataclass(frozen=True, slots=True)
class CumulativeIncidenceEstimate:
    """Arm cumulative incidence curves from ``analyze`` with a :class:`CompetingRisksOutcome`.

    The curves are Aalen-Johansen estimates of the target cause's cumulative incidence. Read
    ``incidence_difference`` first, then ``support_status`` and the interval.

    Fields:

    * ``target_cause`` -- the reported event code.
    * ``minimum_horizon_risk_set`` -- the smaller arm count at risk at ``tau`` for
      delayed-entry incidence. Subjects enter strictly before ``tau`` and remain observed
      through it.
    * ``times`` -- event-time grid, including zero and ``tau``.
    * ``control_incidence``, ``treated_incidence`` -- arm cumulative incidence at each time.
    * ``incidence_difference`` -- treated minus control at ``tau``.
    * ``incidence_difference_interval`` -- pointwise 95% bootstrap interval for that
      difference, or ``None``.
    * ``tau``, ``uncertainty``, ``bootstrap_replicates_requested``,
      ``bootstrap_replicates_ok``, ``censoring_survival_provenance``, ``difference_band``,
      ``band_unavailable_reason``, ``support_status`` and ``assumptions`` -- as
      :class:`SurvivalEstimate`. Delayed-entry pointwise intervals have separate support
      rows for unweighted analysis and fixed known censoring survival. A supported
      simultaneous band has its own status (``"supported_simultaneous_band"``).
    """

    target_cause: int
    times: tuple[float, ...]
    control_incidence: tuple[float, ...]
    treated_incidence: tuple[float, ...]
    incidence_difference: float
    tau: float
    uncertainty: str = "point_only"
    incidence_difference_interval: tuple[float, float] | None = None
    bootstrap_replicates_requested: int | None = None
    bootstrap_replicates_ok: int | None = None
    censoring_survival_provenance: str | None = None
    difference_band: SurvivalDifferenceBand | None = None
    band_unavailable_reason: str | None = None
    support_status: str = _direct_status.UNSUPPORTED_POINT_UTILITY
    assumptions: tuple[str, ...] = (
        "individual_random_assignment",
        "all_event_causes_coded_distinctly",
        "independent_right_censoring_within_arm",
        "consistency",
        "no_interference",
    )
    minimum_horizon_risk_set: int | None = None


def difference_at_tau(
    times: Sequence[float], treated: Sequence[float], control: Sequence[float], tau: float
) -> float:
    """``treated - control`` at the grid point equal to ``tau`` (within ``1e-10``).

    ``times``, ``treated`` and ``control`` are aligned sequences, such as the fields of a
    :class:`SurvivalEstimate`.

    Raises:
        CausalValueError: the sequences differ in length, or ``tau`` is not a grid point.
    """
    if not len(times) == len(treated) == len(control):
        raise CausalValueError("times and arm curves must have equal lengths")
    for index, time in enumerate(times):
        if abs(time - tau) <= 1e-10:
            return float(treated[index] - control[index])
    raise CausalValueError(f"tau={tau} is not a point of the returned time grid")


def estimate_survival_ipcw(
    data: Any,
    query: SurvivalOutcome,
    *,
    times: Sequence[float],
    censoring_survival: Any,
    minimum_probability: float = 0.01,
) -> IPCWSurvivalEstimate:
    """IPCW estimate from a caller-supplied censoring survival matrix.

    Deprecated: this direct path runs without the causal compiler. Use ``analyze`` with a :class:`SurvivalOutcome` carrying ``known_censoring``; the
    probabilities go in data columns on the same grid.
    """
    _deprecation.direct_path_deprecated(
        "antecedent.survival.estimate_survival_ipcw",
        "antecedent.analyze(data, query=SurvivalOutcome(..., observation_assumption=IndependentGiven(...), known_censoring=KnownCensoringSurvival(times, columns)))",
    )
    return _estimate_survival_ipcw_direct(
        data,
        query,
        times=times,
        censoring_survival=censoring_survival,
        minimum_probability=minimum_probability,
    )


def _estimate_survival_ipcw_direct(
    data: Any,
    query: SurvivalOutcome,
    *,
    times: Sequence[float],
    censoring_survival: Any,
    minimum_probability: float = 0.01,
) -> IPCWSurvivalEstimate:
    """Estimate inverse-probability-of-censoring weighted survival curves and RMST.

    Use this when censoring depends on covariates and you have fitted the censoring survival
    ``G`` yourself. ``censoring_survival[i, j]`` is subject ``i``'s probability of remaining
    uncensored immediately before ``times[j]``. The estimator weights event and risk counts
    by ``1 / G`` at each event time and integrates the right-continuous curve for RMST. You
    are responsible for fitting and checking ``G``; it is neither fitted nor verified here.
    For intervals, put the probabilities in data columns and use
    :class:`KnownCensoringSurvival` with ``analyze`` instead.

    Parameters:

    * ``data`` -- a table holding the query's columns.
    * ``query`` -- a :class:`SurvivalOutcome` with ``randomized=True``, no ``delayed_entry``
      and no ``known_censoring``.
    * ``times`` -- strictly increasing grid from ``0`` through ``query.tau`` that includes
      every recorded duration at or before ``tau``.
    * ``censoring_survival`` -- a subjects-by-times matrix; each row starts at one, does not
      increase, and stays within ``[minimum_probability, 1]``.
    * ``minimum_probability`` -- positivity floor in ``(0, 1]``, default ``0.01``.

    Returns:
        An :class:`IPCWSurvivalEstimate` of points only; read ``rmst_difference`` and the
        curves, then the diagnostics ``minimum_censoring_survival`` and the minimum risk sets.

    Raises:
        CausalValueError: any input violates the rules above, a column is missing, an
            event or treatment value is not boolean or 0/1, an arm is empty or not followed
            through ``tau``, or the weighted event counts exceed the risk set.
        CausalResourceError: the copied grid exceeds the memory budget.
    """
    if not isinstance(query, SurvivalOutcome):
        raise CausalValueError("query must be a SurvivalOutcome")
    if query.known_censoring is not None or query.stratified_censoring is not None:
        raise CausalValueError(
            "query already declares a censoring procedure; use analyze(data, query=...)"
        )
    if not query.randomized:
        raise CausalValueError(
            "causal survival estimates currently require declared individual random assignment"
        )
    if query.delayed_entry is not None:
        raise CausalValueError("IPCW survival currently refuses delayed-entry queries")
    names, columns = as_columns(data)
    for name in (query.duration, query.event_observed, query.treatment):
        if name not in names:
            raise CausalValueError(f"required survival column {name!r} is missing")
    duration = np.asarray(columns[names.index(query.duration)], dtype=np.float64)
    events = _binary(columns[names.index(query.event_observed)], "event_observed")
    treatment = _binary(columns[names.index(query.treatment)], "treatment")
    grid = np.asarray(times, dtype=np.float64)
    g = np.asarray(censoring_survival, dtype=np.float64)
    if grid.ndim != 1:
        raise CausalValueError("times must be a one-dimensional grid")
    if g.ndim != 2 or g.shape[0] != len(duration):
        raise CausalValueError("censoring_survival must be a subjects-by-times matrix")
    if not np.isfinite(minimum_probability) or not 0 < minimum_probability <= 1:
        raise CausalValueError("minimum_probability must be finite and in (0, 1]")
    try:
        (
            returned_times,
            control,
            treated,
            rmst_control,
            rmst_treated,
            minrisk_control,
            minrisk_treated,
            minimum_g,
        ) = _randomized_survival_ipcw(
            duration,
            events,
            treatment,
            grid,
            g,
            float(query.tau),
            float(minimum_probability),
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


def _cause_array(values: Any) -> NDArray[np.float64] | NDArray[np.int64]:
    array = np.asarray(values)
    invalid_code = "event_cause values must be non-negative integer codes; zero denotes censoring"
    invalid_range = "event_cause values must fit in non-negative 64-bit integer codes"
    if array.ndim != 1:
        raise CausalValueError(invalid_code)
    if array.dtype.kind == "f":
        return np.asarray(array, dtype=np.float64)
    if array.dtype.kind in "iu":
        if array.dtype.kind == "u" and (array > np.iinfo(np.int64).max).any():
            raise CausalValueError(invalid_range)
        return np.asarray(array, dtype=np.int64)

    def code(value: Any) -> int:
        if isinstance(value, (bool, np.bool_)) or not (
            isinstance(value, (int, np.integer))
            or (
                isinstance(value, (float, np.floating))
                and np.isfinite(value)
                and value.is_integer()
            )
        ):
            raise CausalValueError(invalid_code)
        result = int(value)
        if result < 0 or result > np.iinfo(np.int64).max:
            raise CausalValueError(invalid_range)
        return result

    return np.fromiter((code(value) for value in array), dtype=np.int64, count=len(array))


def estimate_cumulative_incidence_ipcw(
    data: Any,
    query: CompetingRisksOutcome,
    *,
    times: Sequence[float],
    censoring_survival: Any,
    minimum_probability: float = 0.01,
) -> IPCWCumulativeIncidenceEstimate:
    """IPCW estimate from a caller-supplied censoring survival matrix.

    Deprecated: this direct path runs without the causal compiler. Use ``analyze`` with a :class:`CompetingRisksOutcome` carrying ``known_censoring``; the
    probabilities go in data columns on the same grid.
    """
    _deprecation.direct_path_deprecated(
        "antecedent.survival.estimate_cumulative_incidence_ipcw",
        "antecedent.analyze(data, query=CompetingRisksOutcome(..., observation_assumption=IndependentGiven(...), known_censoring=KnownCensoringSurvival(times, columns)))",
    )
    return _estimate_cumulative_incidence_ipcw_direct(
        data,
        query,
        times=times,
        censoring_survival=censoring_survival,
        minimum_probability=minimum_probability,
    )


def _estimate_cumulative_incidence_ipcw_direct(
    data: Any,
    query: CompetingRisksOutcome,
    *,
    times: Sequence[float],
    censoring_survival: Any,
    minimum_probability: float = 0.01,
) -> IPCWCumulativeIncidenceEstimate:
    """Estimate an inverse-probability-of-censoring weighted Aalen-Johansen cumulative incidence.

    The competing-risks counterpart of :func:`estimate_survival_ipcw`: target-cause failures
    and all-cause event-free survival are weighted by ``1 / G`` inside the Aalen-Johansen
    recursion. ``censoring_survival[i, j]`` is subject ``i``'s probability of remaining
    uncensored immediately before ``times[j]``. The censoring model is neither fitted nor
    verified here, and the result is points only.

    Parameters are as :func:`estimate_survival_ipcw`, with ``query`` a
    :class:`CompetingRisksOutcome` (``randomized=True``, no ``delayed_entry``, no
    ``known_censoring``). Cause codes must be nonnegative integers (zero is censoring), at
    least two distinct causes must be observed and ``target_cause`` must be among them.

    Returns:
        An :class:`IPCWCumulativeIncidenceEstimate`; read ``incidence_difference`` first.

    Raises:
        CausalValueError: as :func:`estimate_survival_ipcw`, or the cause codes violate the
            rules above.
        CausalResourceError: the copied grid exceeds the memory budget.
    """
    if not isinstance(query, CompetingRisksOutcome):
        raise CausalValueError("query must be a CompetingRisksOutcome")
    if query.known_censoring is not None or query.stratified_censoring is not None:
        raise CausalValueError(
            "query already declares a censoring procedure; use analyze(data, query=...)"
        )
    if not query.randomized:
        raise CausalValueError(
            "causal cumulative incidence requires declared individual random assignment"
        )
    if query.delayed_entry is not None:
        raise CausalValueError("IPCW cumulative incidence currently refuses delayed-entry queries")
    names, columns = as_columns(data, native_dtypes=True)
    for name in (query.duration, query.event_cause, query.treatment):
        if name not in names:
            raise CausalValueError(f"required competing-risks column {name!r} is missing")
    duration = np.asarray(columns[names.index(query.duration)], dtype=np.float64)
    causes = _cause_array(columns[names.index(query.event_cause)])
    treatment = _binary(columns[names.index(query.treatment)], "treatment")
    if len(causes) != len(duration):
        raise CausalValueError(
            "duration, event_cause, and treatment columns must have equal lengths"
        )
    g = np.asarray(censoring_survival, dtype=np.float64)
    grid = np.asarray(times, dtype=np.float64)
    if grid.ndim != 1:
        raise CausalValueError("times must be a one-dimensional grid")
    if g.ndim != 2 or g.shape[0] != len(duration):
        raise CausalValueError("censoring_survival must be a subjects-by-times matrix")
    if not np.isfinite(minimum_probability) or not 0 < minimum_probability <= 1:
        raise CausalValueError("minimum_probability must be finite and in (0, 1]")
    try:
        (returned_times, control, treated, minrisk_control, minrisk_treated, minimum_g) = (
            _randomized_cumulative_incidence_ipcw(
                duration,
                causes,
                treatment,
                grid,
                g,
                float(query.tau),
                int(query.target_cause),
                float(minimum_probability),
            )
        )
    except ValueError as error:
        raise CausalValueError(str(error)) from error
    return IPCWCumulativeIncidenceEstimate(
        int(query.target_cause),
        tuple(float(value) for value in returned_times),
        tuple(float(value) for value in control),
        tuple(float(value) for value in treated),
        difference_at_tau(returned_times, treated, control, float(query.tau)),
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
    "StratifiedCensoringSurvival",
    "SurvivalDifferenceBand",
    "SurvivalEstimate",
    "SurvivalOutcome",
    "estimate_cumulative_incidence_ipcw",
    "estimate_survival_ipcw",
]
