"""Validation: discovery stability, target-native refutation and sensitivity ranges.

* Discovery stability validators: thin wrappers over ``antecedent-validate`` stability checks
  (PCMCI block bootstrap, false-positive surrogates, parameter grids, orientation, null
  calibration, environment holdout, regime stability).
* Sensitivity ranges beyond the marginal sensitivity model, each an
  :class:`AssumptionRange` tied to its target (what is assumed, the range tested, the
  break-even value): :func:`evalue`, :func:`omitted_variable_bias`,
  :func:`omitted_variable_bias_from_data`, :func:`rosenbaum_bounds`,
  :func:`matched_sets_sensitivity`, :func:`amplify` and :func:`mediation_sensitivity`.
* Target-native refutation: :func:`target_refutation_plan` (the refuter catalogue of a target
  family and quadrant), :func:`derivative_consistency`, :func:`local_linearity`,
  :func:`prior_sensitivity_target` and :func:`posterior_predictive_target`. A prepared study
  refutes its own target with ``PreparedAnalysis.refute_target``.

Every number is computed in Rust; calibration of these diagnostics is unmeasured.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

from . import _native
from ._data import as_columns, as_multi_env_columns
from ._scientific import ScientificResult
from .errors import CausalTypeError, CausalValueError, StructuredRefusal
from .execution import Execution, native_controls
from .outcomes import Answer, Outcome, outcome_operation, require_success

if TYPE_CHECKING:
    from .prepared import PreparedAnalysis

from ._native import (
    sensitivity_amplify as _sensitivity_amplify,
)
from ._native import (
    sensitivity_evalue as _sensitivity_evalue,
)
from ._native import (
    sensitivity_evalue_posterior as _sensitivity_evalue_posterior,
)
from ._native import (
    sensitivity_evalue_vector as _sensitivity_evalue_vector,
)
from ._native import (
    sensitivity_matched_sets as _sensitivity_matched_sets,
)
from ._native import (
    sensitivity_mediation as _sensitivity_mediation,
)
from ._native import (
    sensitivity_mediation_paths as _sensitivity_mediation_paths,
)
from ._native import (
    sensitivity_mediation_posterior as _sensitivity_mediation_posterior,
)
from ._native import (
    sensitivity_mediation_vector as _sensitivity_mediation_vector,
)
from ._native import (
    sensitivity_msm_parameters as _sensitivity_msm_parameters,
)
from ._native import (
    sensitivity_msm_parameters_refit as _sensitivity_msm_parameters_refit,
)
from ._native import (
    sensitivity_omitted_variable_bias as _sensitivity_omitted_variable_bias,
)
from ._native import (
    sensitivity_omitted_variable_bias_fit as _sensitivity_omitted_variable_bias_fit,
)
from ._native import (
    sensitivity_ovb_posterior as _sensitivity_ovb_posterior,
)
from ._native import (
    sensitivity_ovb_vector as _sensitivity_ovb_vector,
)
from ._native import (
    sensitivity_rosenbaum as _sensitivity_rosenbaum,
)
from ._native import (
    sensitivity_rosenbaum_vector as _sensitivity_rosenbaum_vector,
)
from ._native import (
    target_derivative_consistency as _target_derivative_consistency,
)
from ._native import (
    target_local_linearity as _target_local_linearity,
)
from ._native import (
    target_posterior_predictive as _target_posterior_predictive,
)
from ._native import (
    target_prior_sensitivity as _target_prior_sensitivity,
)
from ._native import (
    target_refutation_plan as _target_refutation_plan,
)
from ._native import (
    validate_environment_holdout as _validate_environment_holdout,
)
from ._native import (
    validate_pcmci_alpha_sensitivity as _validate_pcmci_alpha_sensitivity,
)
from ._native import (
    validate_pcmci_block_bootstrap as _validate_pcmci_block_bootstrap,
)
from ._native import (
    validate_pcmci_ci_sensitivity as _validate_pcmci_ci_sensitivity,
)
from ._native import (
    validate_pcmci_false_positive as _validate_pcmci_false_positive,
)
from ._native import (
    validate_pcmci_lag_sensitivity as _validate_pcmci_lag_sensitivity,
)
from ._native import (
    validate_pcmci_plus_orientation as _validate_pcmci_plus_orientation,
)
from ._native import (
    validate_regime_stability as _validate_regime_stability,
)
from ._native import (
    validate_synthetic_null_calibration as _validate_synthetic_null_calibration,
)


def _run(
    fn: Callable[..., dict[str, Any]], execution: Execution | None, *args: Any, **kwargs: Any
) -> dict[str, Any]:
    """Run one native validator under the execution's policy and physical controls."""
    execution = Execution() if execution is None else execution
    return execution._call(
        fn, *args, **kwargs, **native_controls(execution, seed="seed", threads="threads")
    )


def _with_columns(
    fn: Callable[..., dict[str, Any]],
    data: Mapping[str, Any] | Any,
    *args: Any,
    execution: Execution | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    names, columns = as_columns(data)
    return _run(fn, execution, names, columns, *args, **kwargs)


def _with_multi_env(
    fn: Callable[..., dict[str, Any]],
    data: Sequence[Mapping[str, Any] | Any],
    *,
    execution: Execution | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    names, env_columns = as_multi_env_columns(data)
    return _run(fn, execution, names, env_columns, **kwargs)


def validate_pcmci_block_bootstrap(
    data: Mapping[str, Any] | Any,
    *,
    max_lag: int = 1,
    alpha: float = 0.05,
    fdr: bool = False,
    ci: str = "parcorr",
    replicates: int = 20,
    block_size: int = 20,
    execution: Execution | None = None,
) -> dict[str, Any]:
    return _with_columns(
        _validate_pcmci_block_bootstrap,
        data,
        max_lag=max_lag,
        alpha=alpha,
        fdr=fdr,
        ci=ci,
        replicates=replicates,
        block_size=block_size,
        execution=execution,
    )


def validate_pcmci_false_positive(
    data: Mapping[str, Any] | Any,
    *,
    max_lag: int = 1,
    alpha: float = 0.05,
    fdr: bool = False,
    ci: str = "parcorr",
    transform: str = "permute",
    replicates: int = 20,
    execution: Execution | None = None,
) -> dict[str, Any]:
    return _with_columns(
        _validate_pcmci_false_positive,
        data,
        max_lag=max_lag,
        alpha=alpha,
        fdr=fdr,
        ci=ci,
        transform=transform,
        replicates=replicates,
        execution=execution,
    )


def validate_pcmci_alpha_sensitivity(
    data: Mapping[str, Any] | Any,
    alphas: Sequence[float],
    *,
    max_lag: int = 1,
    fdr: bool = False,
    ci: str = "parcorr",
    execution: Execution | None = None,
) -> dict[str, Any]:
    return _with_columns(
        _validate_pcmci_alpha_sensitivity,
        data,
        list(alphas),
        max_lag=max_lag,
        fdr=fdr,
        ci=ci,
        execution=execution,
    )


def validate_pcmci_lag_sensitivity(
    data: Mapping[str, Any] | Any,
    max_lags: Sequence[int],
    *,
    alpha: float = 0.05,
    fdr: bool = False,
    ci: str = "parcorr",
    execution: Execution | None = None,
) -> dict[str, Any]:
    return _with_columns(
        _validate_pcmci_lag_sensitivity,
        data,
        [int(m) for m in max_lags],
        alpha=alpha,
        fdr=fdr,
        ci=ci,
        execution=execution,
    )


def validate_pcmci_ci_sensitivity(
    data: Mapping[str, Any] | Any,
    ci_names: Sequence[str],
    *,
    max_lag: int = 1,
    alpha: float = 0.05,
    fdr: bool = False,
    execution: Execution | None = None,
) -> dict[str, Any]:
    return _with_columns(
        _validate_pcmci_ci_sensitivity,
        data,
        list(ci_names),
        max_lag=max_lag,
        alpha=alpha,
        fdr=fdr,
        execution=execution,
    )


def validate_pcmci_plus_orientation(
    data: Mapping[str, Any] | Any,
    *,
    max_lag: int = 1,
    alpha: float = 0.05,
    fdr: bool = False,
    ci: str = "parcorr",
    replicates: int = 20,
    block_size: int = 20,
    execution: Execution | None = None,
) -> dict[str, Any]:
    return _with_columns(
        _validate_pcmci_plus_orientation,
        data,
        max_lag=max_lag,
        alpha=alpha,
        fdr=fdr,
        ci=ci,
        replicates=replicates,
        block_size=block_size,
        execution=execution,
    )


def validate_synthetic_null_calibration(
    *,
    max_lag: int = 1,
    alpha: float = 0.05,
    fdr: bool = False,
    ci: str = "parcorr",
    n_sim: int = 20,
    n_obs: int = 100,
    n_vars: int = 3,
    execution: Execution | None = None,
) -> dict[str, Any]:
    return _run(
        _validate_synthetic_null_calibration,
        execution,
        max_lag=max_lag,
        alpha=alpha,
        fdr=fdr,
        ci=ci,
        n_sim=n_sim,
        n_obs=n_obs,
        n_vars=n_vars,
    )


def validate_environment_holdout(
    data: Sequence[Mapping[str, Any] | Any],
    *,
    max_lag: int = 1,
    alpha: float = 0.05,
    fdr: bool = False,
    ci: str = "parcorr",
    n_discovery: int = 1,
    execution: Execution | None = None,
) -> dict[str, Any]:
    return _with_multi_env(
        _validate_environment_holdout,
        data,
        max_lag=max_lag,
        alpha=alpha,
        fdr=fdr,
        ci=ci,
        n_discovery=n_discovery,
        execution=execution,
    )


def validate_regime_stability(
    data: Mapping[str, Any] | Any,
    regimes: Sequence[int],
    *,
    max_lag: int = 1,
    alpha: float = 0.05,
    fdr: bool = False,
    ci: str = "parcorr",
    replicates: int = 10,
    block_size: int = 20,
    execution: Execution | None = None,
) -> dict[str, Any]:
    return _with_columns(
        _validate_regime_stability,
        data,
        [int(r) for r in regimes],
        max_lag=max_lag,
        alpha=alpha,
        fdr=fdr,
        ci=ci,
        replicates=replicates,
        block_size=block_size,
        execution=execution,
    )


# --- Target-native refutation and sensitivity ranges -----------------------------------


@dataclass(frozen=True)
class RefutationTarget:
    """The quantity a refutation or a sensitivity range speaks to.

    Two targets are the same exactly when their ``identity`` is equal; it binds the family,
    the quadrant, the coordinate labels and the defining ``source`` (the identity of the
    compiled program, for a prepared study's target).
    """

    family: str
    quadrant: str
    labels: tuple[str, ...]
    identity: str
    source: str

    @classmethod
    def _from_wire(cls, wire: Mapping[str, Any]) -> RefutationTarget:
        return cls(
            family=str(wire["family"]),
            quadrant=str(wire["quadrant"]),
            labels=tuple(str(label) for label in wire["labels"]),
            identity=str(wire["identity"]),
            source=str(wire["source"]),
        )


@dataclass(frozen=True)
class RangePoint:
    """One probed value of a sensitivity parameter and the target (or its test) there."""

    parameter: float | None
    target: float | None
    test: float | None


@dataclass(frozen=True)
class AssumptionRange:
    """A sensitivity analysis as an assumption range tied to its target.

    ``assumption`` names the untestable assumption and ``statement`` says how the range
    relaxes it; ``tested`` lists the probed values of ``parameter`` (``null_parameter`` is the
    value at which the assumption holds); ``break_even`` is the value at which the conclusion
    is explained away and ``break_even_significance`` the value at which it loses
    significance, when reached. ``identity`` binds the target identity and every number.
    """

    target: RefutationTarget
    assumption: str
    statement: str
    parameter: str
    null_parameter: float
    tested: tuple[RangePoint, ...]
    break_even: float | None
    break_even_significance: float | None
    identity: str
    calibration: str

    @classmethod
    def _from_wire(cls, wire: Mapping[str, Any]) -> AssumptionRange:
        return cls(
            target=RefutationTarget._from_wire(wire["target"]),
            assumption=str(wire["assumption"]),
            statement=str(wire["statement"]),
            parameter=str(wire["parameter"]),
            null_parameter=float(wire["null_parameter"]),
            tested=tuple(
                RangePoint(parameter=p["parameter"], target=p["target"], test=p["test"])
                for p in wire["tested"]
            ),
            break_even=wire["break_even"],
            break_even_significance=wire["break_even_significance"],
            identity=str(wire["identity"]),
            calibration=str(wire["calibration"]),
        )


@dataclass(frozen=True)
class SensitivityResult:
    """A sensitivity analysis: its summary numbers and its :class:`AssumptionRange`."""

    method: str
    summary: Mapping[str, Any]
    range: AssumptionRange | None


def _sensitivity(method: str, raw: str) -> SensitivityResult:
    wire = json.loads(raw)
    range_wire = wire.pop("range", None)
    return SensitivityResult(
        method=method,
        summary=wire,
        range=None if range_wire is None else AssumptionRange._from_wire(range_wire),
    )


@dataclass(frozen=True)
class TargetCheck:
    """One target-native refutation on the whole target vector."""

    refuter: str
    target: str
    original: tuple[float | None, ...]
    refuted: tuple[float | None, ...]
    statistic: str
    value: float | None
    df: int
    p_value: float | None
    passed: bool
    informative: bool
    replicates: int
    failure: str | None

    @classmethod
    def _from_wire(cls, wire: Mapping[str, Any]) -> TargetCheck:
        return cls(
            refuter=str(wire["refuter"]),
            target=str(wire["target"]),
            original=tuple(wire["original"]),
            refuted=tuple(wire["refuted"]),
            statistic=str(wire["statistic"]),
            value=wire["value"],
            df=int(wire["df"]),
            p_value=wire["p_value"],
            passed=bool(wire["passed"]),
            informative=bool(wire["informative"]),
            replicates=int(wire["replicates"]),
            failure=wire["failure"],
        )


@dataclass(frozen=True)
class RefuterRefusal:
    """A target-native refuter a route could not run, with the next attempt."""

    refuter: str
    reason: str
    next_attempt: str

    @property
    def code(self) -> Literal["refutation_not_applicable"]:
        """Stable standing; ``reason`` retains the native scientific explanation."""
        return "refutation_not_applicable"


@dataclass(frozen=True, slots=True)
class TargetRefutationSpec:
    """Scientific perturbations and budgets of a target-native validation run.

    ``None`` controls are resolved by the native graph or original declaration.
    A class posterior requires an explicit ``within_class`` assumption for a
    common-cause perturbation; no division of class mass is inferred.
    """

    suite: Literal["cheap", "placebo", "full"] = "full"
    replicates: int = 30
    alpha: float = 0.05
    subset_fraction: float = 0.8
    pre_trend_lead: int = 1
    negative_control_exposure: str | None = None
    negative_control_outcome: str | None = None
    mediator: str | None = None
    mediation_rho_bar: float = 0.1
    common_cause_strength: float = 0.0
    prior_scale_factors: tuple[float, ...] = (0.25, 4.0)
    prior_shift_tolerance: float = 0.5
    derivative_step: float | None = None
    derivative_tolerance: float = 0.1
    linearity_tolerance: float = 0.25
    gathered_resamples: bool = False
    within_class: Literal["uniform_over_members"] | None = None

    def __post_init__(self) -> None:
        import math

        if self.suite not in ("cheap", "placebo", "full"):
            raise CausalValueError("suite must be cheap, placebo or full")
        for name in ("replicates", "pre_trend_lead"):
            value = getattr(self, name)
            if type(value) is not int:
                raise CausalTypeError(f"{name} must be an integer")
            if not 1 <= value <= 2**32 - 1:
                raise CausalValueError(f"{name} must lie in [1, 2**32 - 1]")
        for name in ("alpha", "subset_fraction"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise CausalTypeError(f"{name} must be a number")
            if not 0 < value < 1:
                raise CausalValueError(f"{name} must lie in (0, 1)")
        for name in (
            "mediation_rho_bar",
            "common_cause_strength",
            "prior_shift_tolerance",
            "derivative_tolerance",
            "linearity_tolerance",
            "derivative_step",
        ):
            value = getattr(self, name)
            if value is None and name == "derivative_step":
                continue
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise CausalTypeError(f"{name} must be a number")
            if not math.isfinite(value) or (name != "common_cause_strength" and value < 0):
                raise CausalValueError(f"{name} must be finite and nonnegative")
            if name == "derivative_step" and value == 0:
                raise CausalValueError("derivative_step must be positive")
        for name in ("negative_control_exposure", "negative_control_outcome", "mediator"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise CausalTypeError(f"{name} must be a nonempty column name or None")
        if type(self.gathered_resamples) is not bool:
            raise CausalTypeError("gathered_resamples must be a bool")
        if self.within_class not in (None, "uniform_over_members"):
            raise CausalValueError("within_class must be uniform_over_members or None")
        if isinstance(self.prior_scale_factors, (str, bytes)):
            raise CausalTypeError("prior_scale_factors must be a sequence of positive numbers")
        try:
            factors = tuple(self.prior_scale_factors)
        except TypeError as error:
            raise CausalTypeError("prior_scale_factors must be a sequence") from error
        if any(type(value) not in (int, float) for value in factors):
            raise CausalTypeError("prior_scale_factors must contain numbers")
        if any(not math.isfinite(value) or value <= 0 for value in factors):
            raise CausalValueError("prior_scale_factors must be finite and positive")
        object.__setattr__(self, "prior_scale_factors", factors)


@dataclass(frozen=True, repr=False)
class TargetRefutation(ScientificResult):
    """The target-native suite of one prepared result: plan, reports, refusals, certificate."""

    target: RefutationTarget
    suite: str
    plan: tuple[str, ...]
    reports: tuple[TargetCheck, ...]
    refused: tuple[RefuterRefusal, ...]
    certificate: str
    calibration: str
    reused_original_result: bool = False
    original_executions: int | None = None
    program: str | None = None
    declaration: TargetRefutationSpec | None = None

    def _summary_fields(self) -> dict[str, Any]:
        return {
            "family": self.target.family,
            "suite": self.suite,
            "reports": len(self.reports),
            "refused": len(self.refused),
            "passed": self.passed,
            "calibration": self.calibration,
        }

    @property
    def answer(self) -> Answer:
        return Answer(
            "partial" if self.refused else "structured",
            detail="target_native_refutation",
            structured=self._summary_fields(),
        )

    @property
    def identities(self) -> Mapping[str, Any]:
        return {
            "program_id": self.program,
            "target_id": self.target.identity,
            "certificate": self.certificate,
        }

    @property
    def provenance(self) -> Mapping[str, Any]:
        return {
            "program": self.program,
            "target": self.target.identity,
            "reused_original_result": self.reused_original_result,
            "original_executions": self.original_executions,
            "calibration": self.calibration,
        }

    def inspect(self) -> Any:
        from .results._report import SlotModel

        return (
            super()
            .inspect()
            .model_copy(
                update={
                    "uncertainty": SlotModel(
                        available=False,
                        reason="diagnostic_unmeasured",
                        summary="Target-native diagnostic; operating characteristics unmeasured",
                    ),
                    "next_actions": tuple(
                        {"refuter": r.refuter, "action": r.next_attempt} for r in self.refused
                    ),
                    "assumptions": SlotModel(
                        available=True,
                        summary="Native target and declared perturbation assumptions",
                        payload={
                            "target_family": self.target.family,
                            "quadrant": self.target.quadrant,
                            "within_class": self.declaration.within_class
                            if self.declaration
                            else None,
                        },
                    ),
                    "program_id": self.program,
                    "target_id": self.target.identity,
                }
            )
        )

    @property
    def passed(self) -> bool | None:
        """Whether every informative report passed."""
        informative = tuple(report for report in self.reports if report.informative)
        return all(report.passed for report in informative) if informative else None

    @classmethod
    def _from_json(cls, raw: str) -> TargetRefutation:
        wire = json.loads(raw)
        return cls(
            target=RefutationTarget._from_wire(wire["target"]),
            suite=str(wire["suite"]),
            plan=tuple(str(k) for k in wire["plan"]),
            reports=tuple(TargetCheck._from_wire(r) for r in wire["reports"]),
            refused=tuple(
                RefuterRefusal(
                    refuter=str(r["refuter"]),
                    reason=str(r["reason"]),
                    next_attempt=str(r["next_attempt"]),
                )
                for r in wire["refused"]
            ),
            certificate=str(wire["certificate"]),
            calibration=str(wire["calibration"]),
            reused_original_result=bool(wire.get("reused_original_result", False)),
            original_executions=wire.get("original_executions"),
            program=wire.get("program"),
        )


@outcome_operation
def refute_target(
    source: PreparedAnalysis[Any] | Outcome[Any],
    *,
    checks: TargetRefutationSpec | None = None,
    execution: Execution | None = None,
) -> TargetRefutation:
    """Explicitly refit a study's complete target under declared perturbations.

    A retained outcome reuses its exact original native result. A preparation
    executes its original target once before the suite. Neither operation changes
    the source. For different observations, first use :func:`antecedent.rebind`.
    """
    from .estimation import _PreparedEngine as Engine
    from .prepared import PreparedAnalysis

    completed = isinstance(source, Outcome)
    if isinstance(source, Outcome):
        require_success(source)
        study = source.study
    else:
        study = source
    if not isinstance(study, PreparedAnalysis):
        if completed:
            raise StructuredRefusal(
                {
                    "code": "route_not_supported",
                    "detail": "target_refutation.live_result_required",
                    "message": "refitting requires the retained native result, study and observations",
                }
            )
        raise CausalTypeError(
            "refute_target requires a canonical prepared study or retained Outcome"
        )
    if study._refused is not None:
        require_success(study._refused)
    if checks is None:
        checks = TargetRefutationSpec()
    elif not isinstance(checks, TargetRefutationSpec):
        raise CausalTypeError("checks must be TargetRefutationSpec or None")
    controls = study.execution if execution is None else execution
    if not isinstance(controls, Execution):
        raise CausalTypeError("execution must be Execution or None")
    engine = study._engine
    if not isinstance(engine, Engine) or not isinstance(engine._native, _native.PreparedAnalysis):
        raise StructuredRefusal(
            {
                "code": "route_not_supported",
                "detail": "target_refutation.native_preparation_required",
                "message": "this scientific preparation has no target-native refit implementation",
            }
        )
    if (
        isinstance(source, Outcome)
        and getattr(source.value, "_execution", None) is not engine._native
    ):
        # Canonical estimation binds this exact immutable execution header to
        # both the published native-backed view and its retained study. A
        # program identity alone cannot distinguish another dataset or run.
        raise StructuredRefusal(
            {
                "code": "route_not_supported",
                "detail": "target_refutation.producer_mismatch",
                "message": "the supplied outcome is not the retained native result of this study",
                "next_actions": ({"action": "use the outcome's original retained study"},),
            }
        )
    report = TargetRefutation._from_json(
        controls._call(
            engine._native.refute_target_bound,
            **asdict(checks),
            require_completed=completed,
            seed=controls.seed,
            threads=controls.threads,
            on_stage=controls.on_stage,
        )
    )
    if not report.reports:
        raise StructuredRefusal(
            {
                "code": "route_not_supported",
                "detail": "target_refutation.no_applicable_refuters",
                "message": "none of the declared target refuters can assess this target",
                "causes": tuple(
                    {"code": "refutation_not_applicable", "refuter": r.refuter, "message": r.reason}
                    for r in report.refused
                ),
                "next_actions": tuple(
                    {"refuter": r.refuter, "action": r.next_attempt} for r in report.refused
                ),
                "target": report.target.identity,
                "certificate": report.certificate,
            }
        )
    return replace(report, declaration=checks)


def evalue(
    estimate: float,
    *,
    lower: float | None = None,
    upper: float | None = None,
    standard_error: float | None = None,
    scale: str = "risk_ratio",
    target_source: str | None = None,
) -> SensitivityResult:
    """E-values of an estimate and of its confidence limit nearest the null.

    ``scale`` is ``risk_ratio``, ``odds_ratio_rare``, ``odds_ratio_common``,
    ``hazard_ratio_rare``, ``hazard_ratio_common`` or ``standardized_mean_difference`` (pass
    ``standard_error`` instead of limits). VanderWeele and Ding (2017).
    """
    return _sensitivity(
        "evalue",
        _sensitivity_evalue(
            float(estimate),
            lower=None if lower is None else float(lower),
            upper=None if upper is None else float(upper),
            standard_error=None if standard_error is None else float(standard_error),
            scale=scale,
            target_source=target_source,
        ),
    )


def omitted_variable_bias(
    estimate: float,
    standard_error: float,
    dof: float,
    *,
    q: float = 1.0,
    alpha: float = 0.05,
    r2yz_extreme: float = 1.0,
    benchmarks: Sequence[tuple[str, float, float]] | None = None,
    kd: float = 1.0,
    ky: float | None = None,
    target_source: str | None = None,
) -> SensitivityResult:
    """Partial-R2 sensitivity of a linear-adjustment coefficient (Cinelli and Hazlett 2020).

    Reports the treatment's partial R2 with the outcome, the robustness values ``RV_q`` and
    ``RV_{q,alpha}``, bounds for confounders ``kd``/``ky`` times as strong as each benchmark
    ``(name, R2_{D~Xj|X-j}, R2_{Y~Xj|D,X-j})``, and the extreme-scenario curve as the range.
    """
    return _sensitivity(
        "omitted_variable_bias",
        _sensitivity_omitted_variable_bias(
            float(estimate),
            float(standard_error),
            float(dof),
            q=q,
            alpha=alpha,
            r2yz_extreme=r2yz_extreme,
            benchmarks=None
            if benchmarks is None
            else [(str(n), float(d), float(y)) for n, d, y in benchmarks],
            kd=kd,
            ky=ky,
            target_source=target_source,
        ),
    )


def _column(data: Mapping[str, Any], name: str) -> list[float]:
    return [float(v) for v in data[name]]


def omitted_variable_bias_from_data(
    data: Mapping[str, Any],
    *,
    outcome: str,
    treatment: str,
    covariates: Sequence[str],
    q: float = 1.0,
    alpha: float = 0.05,
    r2yz_extreme: float = 1.0,
    kd: float = 1.0,
    ky: float | None = None,
    target_source: str | None = None,
) -> SensitivityResult:
    """Fit ``outcome ~ treatment + covariates`` and benchmark against every covariate."""
    return _sensitivity(
        "omitted_variable_bias",
        _sensitivity_omitted_variable_bias_fit(
            _column(data, outcome),
            _column(data, treatment),
            [_column(data, c) for c in covariates],
            [str(c) for c in covariates],
            q=q,
            alpha=alpha,
            r2yz_extreme=r2yz_extreme,
            kd=kd,
            ky=ky,
            target_source=target_source,
        ),
    )


def rosenbaum_bounds(
    differences: Sequence[float],
    *,
    gammas: Sequence[float] | None = None,
    alpha: float = 0.05,
    target_source: str | None = None,
) -> SensitivityResult:
    """Wilcoxon signed-rank upper-bound p-values of matched pairs under hidden bias ``Gamma``,
    and the break-even ``Gamma`` (exact law up to 300 non-zero pairs)."""
    return _sensitivity(
        "rosenbaum",
        _sensitivity_rosenbaum(
            [float(d) for d in differences],
            gammas=None if gammas is None else [float(g) for g in gammas],
            alpha=alpha,
            target_source=target_source,
        ),
    )


def evalue_vector(
    estimates: Sequence[float],
    labels: Sequence[str],
    *,
    lowers: Sequence[float] | None = None,
    uppers: Sequence[float] | None = None,
    standard_errors: Sequence[float] | None = None,
    scale: str = "risk_ratio",
    family: str = "conditional_effect",
    temporal: bool = False,
    target_source: str | None = None,
    model: str | None = None,
) -> SensitivityResult:
    """E-values of every coordinate of a vector target, read at its weakest coordinate.

    ``model="mediation"`` reads the coordinates as mediation components (the assumption is no
    unmeasured mediator-outcome confounding) and ``model="marginal_structural"`` as marginal
    structural model parameters (the assumption is sequential exchangeability); ``None`` is a
    plain effect vector.

    One estimate (and limits, or standard errors on the standardized scale) per label: a
    conditional effect's strata, a temporal effect's horizons or a joint contrast's components.
    The claim that every coordinate keeps its sign breaks at the smallest coordinate E-value,
    which is the range's ``break_even``.
    """

    def floats(values: Sequence[float] | None) -> list[float] | None:
        return None if values is None else [float(v) for v in values]

    return _sensitivity(
        "evalue_vector",
        _sensitivity_evalue_vector(
            [float(e) for e in estimates],
            [str(label) for label in labels],
            lowers=floats(lowers),
            uppers=floats(uppers),
            standard_errors=floats(standard_errors),
            scale=scale,
            family=family,
            temporal=temporal,
            target_source=target_source,
            model=model,
        ),
    )


def evalue_posterior(
    draws: Sequence[Sequence[float]],
    labels: Sequence[str],
    *,
    scale: str = "risk_ratio",
    level: float = 0.95,
    family: str = "conditional_effect",
    temporal: bool = False,
    target_source: str | None = None,
    model: str | None = None,
) -> SensitivityResult:
    """E-values of a posterior over a vector target.

    ``model`` binds the range to the assumption of a mediation analysis (``"mediation"``) or a
    marginal structural model (``"marginal_structural"``), as in :func:`evalue_vector`.

    ``draws[k]`` are the posterior draws of coordinate ``k`` on ``scale``. Each draw is
    bias-adjusted at every probed confounder strength; the range reports the weakest
    coordinate's posterior median and, as ``test``, the joint posterior probability that every
    coordinate keeps its reported side of the null. ``break_even`` is the smallest E-value of a
    posterior median and ``break_even_significance`` the smallest E-value of a credible limit.
    """
    return _sensitivity(
        "evalue_posterior",
        _sensitivity_evalue_posterior(
            [[float(v) for v in column] for column in draws],
            [str(label) for label in labels],
            scale=scale,
            level=level,
            family=family,
            temporal=temporal,
            target_source=target_source,
            model=model,
        ),
    )


def omitted_variable_bias_vector(
    estimates: Sequence[float],
    standard_errors: Sequence[float],
    dofs: Sequence[float],
    labels: Sequence[str],
    *,
    q: float = 1.0,
    alpha: float = 0.05,
    r2yz_extreme: float = 1.0,
    family: str = "conditional_effect",
    temporal: bool = False,
    target_source: str | None = None,
    model: str | None = None,
) -> SensitivityResult:
    """Partial-R2 sensitivity of a vector of linear-adjustment coefficients.

    ``model`` binds the range to a mediation analysis (the coefficients are the mediator-outcome
    path) or a marginal structural model, as in :func:`evalue_vector`.

    The range is read at the weakest coordinate: its ``target`` is the smallest share of a
    reported estimate left, and ``break_even`` the smallest robustness value ``RV_q``.
    """
    return _sensitivity(
        "omitted_variable_bias_vector",
        _sensitivity_ovb_vector(
            [float(e) for e in estimates],
            [float(s) for s in standard_errors],
            [float(d) for d in dofs],
            [str(label) for label in labels],
            q=q,
            alpha=alpha,
            r2yz_extreme=r2yz_extreme,
            family=family,
            temporal=temporal,
            target_source=target_source,
            model=model,
        ),
    )


def omitted_variable_bias_posterior(
    coefficients: Sequence[Sequence[float]],
    bias_scale: Sequence[Sequence[float]],
    labels: Sequence[str],
    *,
    level: float = 0.95,
    r2yz_extreme: float = 1.0,
    family: str = "conditional_effect",
    temporal: bool = False,
    target_source: str | None = None,
    model: str | None = None,
) -> SensitivityResult:
    """Partial-R2 sensitivity of a posterior over a vector of linear-adjustment coefficients.

    ``model`` binds the range to a mediation analysis or a marginal structural model, as in
    :func:`evalue_vector`.

    ``bias_scale[k][d]`` is ``sd(Y_k | D, X) / sd(D | X)`` in draw ``d``; each draw moves toward
    the null by ``bias_scale * sqrt(R2yz * R2dz / (1 - R2dz))`` (Cinelli and Hazlett 2020).
    ``test`` is the joint posterior probability that every coordinate keeps its side.
    """
    return _sensitivity(
        "omitted_variable_bias_posterior",
        _sensitivity_ovb_posterior(
            [[float(v) for v in column] for column in coefficients],
            [[float(v) for v in column] for column in bias_scale],
            [str(label) for label in labels],
            level=level,
            r2yz_extreme=r2yz_extreme,
            family=family,
            temporal=temporal,
            target_source=target_source,
            model=model,
        ),
    )


def mediation_sensitivity_vector(
    treatment: Sequence[float],
    mediator: Sequence[float],
    outcomes: Sequence[Sequence[float]],
    covariates: Sequence[Sequence[float]],
    labels: Sequence[str],
    *,
    component: str = "indirect",
    temporal: bool = False,
    target_source: str | None = None,
) -> SensitivityResult:
    """Mediation sensitivity of one mediator to several outcomes (or horizons).

    The assumption is no mediator-outcome confounding (sequential ignorability); the range
    probes the error correlation ``rho`` of Imai, Keele and Yamamoto (2010). ``component`` is
    ``"indirect"`` or ``"direct"`` (the total effect, identified without the mediator
    assumption, minus the indirect effect). The range is read at the weakest outcome: ``target``
    is the smallest share of the reported component left and ``break_even`` the correlation of
    smallest magnitude at which some outcome's component reaches zero.
    """
    return _sensitivity(
        "mediation_vector",
        _sensitivity_mediation_vector(
            [float(v) for v in treatment],
            [float(v) for v in mediator],
            [[float(v) for v in column] for column in outcomes],
            [[float(v) for v in column] for column in covariates],
            [str(label) for label in labels],
            component=component,
            temporal=temporal,
            target_source=target_source,
        ),
    )


def mediation_path_sensitivity(
    treatment: Sequence[float],
    mediator: Sequence[float],
    outcomes: Sequence[Sequence[float]],
    covariates: Sequence[Sequence[float]],
    labels: Sequence[str],
    *,
    path: str = "indirect",
    q: float = 1.0,
    alpha: float = 0.05,
    r2yz_extreme: float = 1.0,
    temporal: bool = False,
    target_source: str | None = None,
) -> SensitivityResult:
    """Partial-R2 sensitivity of one path of a mediation analysis, per outcome.

    ``path`` is ``"treatment_mediator"`` (confounders of treatment and mediator),
    ``"mediator_outcome"`` (of mediator and outcome), ``"treatment_outcome"`` (of treatment and
    outcome given the mediator, the direct effect) or ``"indirect"`` (the product of the first
    two, both paths confounded at the tested strength). The range is read at the weakest
    outcome; for the indirect effect ``target`` is the product of the two paths' shares left,
    ``test`` the product of the shares of their adjusted limits, and ``break_even`` the smallest
    equal partial R2 at which the product falls to ``1 - q`` (at ``q = 1`` the smaller of the
    two paths' robustness values).
    """
    return _sensitivity(
        "mediation_paths",
        _sensitivity_mediation_paths(
            [float(v) for v in treatment],
            [float(v) for v in mediator],
            [[float(v) for v in column] for column in outcomes],
            [[float(v) for v in column] for column in covariates],
            [str(label) for label in labels],
            path=path,
            q=q,
            alpha=alpha,
            r2yz_extreme=r2yz_extreme,
            temporal=temporal,
            target_source=target_source,
        ),
    )


def mediation_sensitivity_posterior(
    beta_mediator: Sequence[Sequence[float]],
    sigma_total: Sequence[Sequence[float]],
    sigma_mediator: Sequence[Sequence[float]],
    rho_tilde: Sequence[Sequence[float]],
    total_effect: Sequence[Sequence[float]],
    labels: Sequence[str],
    *,
    component: str = "indirect",
    level: float = 0.95,
    temporal: bool = False,
    target_source: str | None = None,
) -> SensitivityResult:
    """Mediation sensitivity of a posterior over the regression summaries of several outcomes.

    Each argument is coordinate-major draws (``[k][d]``) of the mediator-regression treatment
    coefficient, the two residual standard deviations, their residual correlation and the
    total-effect coefficient. Every draw moves through the Imai-Keele-Yamamoto formula at each
    probed ``rho``; ``test`` is the joint posterior probability that every outcome keeps its
    side of zero, ``break_even`` the correlation of smallest magnitude at which a posterior
    median reaches zero and ``break_even_significance`` the same for the credible limit.
    """
    return _sensitivity(
        "mediation_posterior",
        _sensitivity_mediation_posterior(
            [[float(v) for v in column] for column in beta_mediator],
            [[float(v) for v in column] for column in sigma_total],
            [[float(v) for v in column] for column in sigma_mediator],
            [[float(v) for v in column] for column in rho_tilde],
            [[float(v) for v in column] for column in total_effect],
            [str(label) for label in labels],
            component=component,
            level=level,
            temporal=temporal,
            target_source=target_source,
        ),
    )


def msm_parameter_sensitivity(
    outcome: Sequence[float],
    exposures: Sequence[Sequence[float]],
    weights: Sequence[float],
    labels: Sequence[str],
    *,
    q: float = 1.0,
    alpha: float = 0.05,
    r2yz_extreme: float = 1.0,
    temporal: bool = False,
    target_source: str | None = None,
) -> SensitivityResult:
    """Partial-R2 sensitivity of the exposure parameters of a marginal structural model.

    The model ``outcome ~ 1 + exposures`` is fitted by weighted least squares with the subject
    ``weights`` (for example stabilized inverse-probability weights), held fixed, and a robust
    (HC1) standard error with the effective degrees of freedom of the weights. The assumption is
    sequential exchangeability; the range is read at the weakest parameter.
    """
    return _sensitivity(
        "msm_parameters",
        _sensitivity_msm_parameters(
            [float(v) for v in outcome],
            [[float(v) for v in column] for column in exposures],
            [float(v) for v in weights],
            [str(label) for label in labels],
            q=q,
            alpha=alpha,
            r2yz_extreme=r2yz_extreme,
            temporal=temporal,
            target_source=target_source,
        ),
    )


def msm_parameter_sensitivity_refit(
    outcome: Sequence[float],
    exposures: Sequence[Sequence[float]],
    treatments: Sequence[Sequence[float]],
    numerators: Sequence[Sequence[Sequence[float]]],
    denominators: Sequence[Sequence[Sequence[float]]],
    labels: Sequence[str],
    *,
    replicates: int = 200,
    q: float = 1.0,
    alpha: float = 0.05,
    r2yz_extreme: float = 1.0,
    temporal: bool = False,
    target_source: str | None = None,
    execution: Execution | None = None,
) -> SensitivityResult:
    """Partial-R2 sensitivity of marginal structural model parameters with estimated weights.

    ``treatments[k]`` is the 0/1 treatment of time point ``k`` and ``numerators[k]`` /
    ``denominators[k]`` the covariate columns of its stabilized-weight models (an empty numerator
    is the marginal treatment probability; the denominator is the measured history). Each
    bootstrap replicate resamples subjects and refits every weight model and the weighted
    regression, so ``standard_error`` carries the cost of estimating the weights;
    ``fixed_weight_standard_error`` is the robust error that holds them fixed (conservative for
    an estimated denominator model).
    """
    execution = Execution() if execution is None else execution
    return _sensitivity(
        "msm_parameters_refit",
        execution._call(
            _sensitivity_msm_parameters_refit,
            [float(v) for v in outcome],
            [[float(v) for v in column] for column in exposures],
            [[float(v) for v in column] for column in treatments],
            [[[float(v) for v in column] for column in block] for block in numerators],
            [[[float(v) for v in column] for column in block] for block in denominators],
            [str(label) for label in labels],
            replicates=int(replicates),
            **native_controls(execution, seed="seed"),
            q=q,
            alpha=alpha,
            r2yz_extreme=r2yz_extreme,
            temporal=temporal,
            target_source=target_source,
        ),
    )


def rosenbaum_bounds_vector(
    differences: Sequence[Sequence[float]],
    labels: Sequence[str],
    *,
    gammas: Sequence[float] | None = None,
    alpha: float = 0.05,
    family: str = "conditional_effect",
    temporal: bool = False,
    target_source: str | None = None,
) -> SensitivityResult:
    """Rosenbaum bounds of a vector of matched designs, read at the weakest coordinate.

    ``differences[k]`` are the matched-pair differences behind coordinate ``k``; ``test`` is the
    largest upper-bound p-value across coordinates and ``break_even_significance`` the smallest
    ``Gamma*`` (absent when some coordinate does not reject at ``Gamma = 1``).
    """
    return _sensitivity(
        "rosenbaum_vector",
        _sensitivity_rosenbaum_vector(
            [[float(d) for d in column] for column in differences],
            [str(label) for label in labels],
            gammas=None if gammas is None else [float(g) for g in gammas],
            alpha=alpha,
            family=family,
            temporal=temporal,
            target_source=target_source,
        ),
    )


def matched_sets_sensitivity(
    sets: Sequence[tuple[Sequence[float], int]],
    *,
    score: str = "aligned_ranks",
    gammas: Sequence[float] | None = None,
    alpha: float = 0.05,
) -> SensitivityResult:
    """Rosenbaum bounds for matched sets ``(outcomes, treated_index)`` with several controls
    (separable approximation of Gastwirth, Krieger and Rosenbaum 2000)."""
    return _sensitivity(
        "matched_sets",
        _sensitivity_matched_sets(
            [([float(y) for y in outcomes], int(treated)) for outcomes, treated in sets],
            score=score,
            gammas=None if gammas is None else [float(g) for g in gammas],
            alpha=alpha,
        ),
    )


def amplify(gamma: float, lam: float) -> float | None:
    """Outcome effect ``Delta`` an unobserved covariate with treatment effect ``lam`` needs to
    produce hidden bias ``gamma`` (Rosenbaum and Silber 2009); ``None`` when ``lam <= gamma``."""
    return _sensitivity_amplify(float(gamma), float(lam))


def mediation_sensitivity(
    data: Mapping[str, Any],
    *,
    treatment: str,
    mediator: str,
    outcome: str,
    covariates: Sequence[str] = (),
    target_source: str | None = None,
) -> SensitivityResult:
    """Indirect effect as a function of the mediator-outcome error correlation ``rho``
    (Imai, Keele and Yamamoto 2010); the break-even ``rho`` explains it away."""
    return _sensitivity(
        "mediation",
        _sensitivity_mediation(
            _column(data, treatment),
            _column(data, mediator),
            _column(data, outcome),
            [_column(data, c) for c in covariates],
            target_source=target_source,
        ),
    )


def target_refutation_plan(
    family: str,
    *,
    temporal: bool = False,
    bayesian: bool = False,
    suite: str = "full",
) -> tuple[str, ...]:
    """The target-native refuter catalogue of a target family in one quadrant.

    ``family`` is ``scalar_average_effect``, ``conditional_effect``, ``response_derivative``,
    ``temporal_effect``, ``mediation``, ``world_valued``, ``joint_contrast`` or ``attribution``; ``suite`` is
    ``cheap``, ``placebo`` or ``full``.
    """
    wire = json.loads(
        _target_refutation_plan(family, temporal=temporal, bayesian=bayesian, suite=suite)
    )
    return tuple(str(k) for k in wire["refuters"])


def derivative_consistency(
    response: Callable[[float], float],
    points: Sequence[float],
    reported: Sequence[float],
    *,
    step: float = 1e-2,
    tolerance: float = 1e-4,
) -> TargetCheck:
    """Reported derivatives against Richardson-extrapolated central differences of the
    fitted ``response`` callable."""
    wire = json.loads(
        _target_derivative_consistency(
            response,
            [float(x) for x in points],
            [float(d) for d in reported],
            step=step,
            tolerance=tolerance,
        )
    )
    return TargetCheck._from_wire(wire["report"])


def local_linearity(
    response: Callable[[float], float],
    points: Sequence[float],
    *,
    step: float = 1e-1,
    tolerance: float = 0.1,
    slope_floor: float = 1e-12,
) -> TargetCheck:
    """How far the fitted ``response`` bends over ``step`` relative to its slope."""
    wire = json.loads(
        _target_local_linearity(
            response,
            [float(x) for x in points],
            step=step,
            tolerance=tolerance,
            slope_floor=slope_floor,
        )
    )
    return TargetCheck._from_wire(wire["report"])


def _rows(draws: Any) -> list[list[float]]:
    return [[float(v) for v in row] for row in draws]


def prior_sensitivity_target(
    labels: Sequence[str],
    baseline: Any,
    alternatives: Mapping[str, Any],
    *,
    tolerance: float = 0.5,
    family: str = "world_valued",
    temporal: bool = False,
) -> TargetCheck:
    """Shift of the posterior of the whole target vector under alternative priors, in
    baseline posterior standard deviations (draws are ``draws x coordinates``)."""
    wire = json.loads(
        _target_prior_sensitivity(
            [str(label) for label in labels],
            _rows(baseline),
            [(str(name), _rows(draws)) for name, draws in alternatives.items()],
            tolerance=tolerance,
            family=family,
            temporal=temporal,
        )
    )
    return TargetCheck._from_wire(wire["report"])


def posterior_predictive_target(
    labels: Sequence[str],
    observed: Sequence[float],
    draws: Any,
    replicated: Any,
    *,
    alpha: float = 0.05,
    family: str = "world_valued",
    temporal: bool = False,
) -> TargetCheck:
    """Compare the observed target with targets recomputed from posterior replicates.

    The tail fraction is estimated from the supplied draws; zero means no exceedances
    were observed. The alpha threshold is a diagnostic convention, not a calibrated
    frequentist significance test, and the finite-draw uncertainty is not reported.
    """
    wire = json.loads(
        _target_posterior_predictive(
            [str(label) for label in labels],
            [float(v) for v in observed],
            _rows(draws),
            _rows(replicated),
            alpha=alpha,
            family=family,
            temporal=temporal,
        )
    )
    return TargetCheck._from_wire(wire["report"])


__all__ = [
    "AssumptionRange",
    "RangePoint",
    "RefutationTarget",
    "RefuterRefusal",
    "SensitivityResult",
    "TargetCheck",
    "TargetRefutation",
    "TargetRefutationSpec",
    "refute_target",
    "amplify",
    "derivative_consistency",
    "evalue",
    "evalue_posterior",
    "evalue_vector",
    "local_linearity",
    "matched_sets_sensitivity",
    "mediation_path_sensitivity",
    "mediation_sensitivity",
    "mediation_sensitivity_posterior",
    "mediation_sensitivity_vector",
    "msm_parameter_sensitivity",
    "msm_parameter_sensitivity_refit",
    "omitted_variable_bias",
    "omitted_variable_bias_from_data",
    "omitted_variable_bias_posterior",
    "omitted_variable_bias_vector",
    "posterior_predictive_target",
    "prior_sensitivity_target",
    "rosenbaum_bounds",
    "rosenbaum_bounds_vector",
    "target_refutation_plan",
    "validate_environment_holdout",
    "validate_pcmci_alpha_sensitivity",
    "validate_pcmci_block_bootstrap",
    "validate_pcmci_ci_sensitivity",
    "validate_pcmci_false_positive",
    "validate_pcmci_lag_sensitivity",
    "validate_pcmci_plus_orientation",
    "validate_regime_stability",
    "validate_synthetic_null_calibration",
]
