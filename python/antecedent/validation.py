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
from dataclasses import dataclass
from typing import Any

from ._data import as_columns, as_multi_env_columns
from ._native import (
    sensitivity_amplify as _sensitivity_amplify,
)
from ._native import (
    sensitivity_evalue as _sensitivity_evalue,
)
from ._native import (
    sensitivity_matched_sets as _sensitivity_matched_sets,
)
from ._native import (
    sensitivity_mediation as _sensitivity_mediation,
)
from ._native import (
    sensitivity_omitted_variable_bias as _sensitivity_omitted_variable_bias,
)
from ._native import (
    sensitivity_omitted_variable_bias_fit as _sensitivity_omitted_variable_bias_fit,
)
from ._native import (
    sensitivity_rosenbaum as _sensitivity_rosenbaum,
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


def _with_columns(
    fn: Callable[..., dict[str, Any]],
    data: Mapping[str, Any] | Any,
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    names, columns = as_columns(data)
    return fn(names, columns, *args, **kwargs)


def _with_multi_env(
    fn: Callable[..., dict[str, Any]], data: Sequence[Mapping[str, Any] | Any], **kwargs: Any
) -> dict[str, Any]:
    names, env_columns = as_multi_env_columns(data)
    return fn(names, env_columns, **kwargs)


def validate_pcmci_block_bootstrap(
    data: Mapping[str, Any] | Any,
    *,
    max_lag: int = 1,
    alpha: float = 0.05,
    fdr: bool = False,
    ci: str = "parcorr",
    replicates: int = 20,
    block_size: int = 20,
    seed: int = 1,
    threads: int | None = None,
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
        seed=seed,
        threads=threads,
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
    seed: int = 1,
    threads: int | None = None,
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
        seed=seed,
        threads=threads,
    )


def validate_pcmci_alpha_sensitivity(
    data: Mapping[str, Any] | Any,
    alphas: Sequence[float],
    *,
    max_lag: int = 1,
    fdr: bool = False,
    ci: str = "parcorr",
    seed: int = 1,
    threads: int | None = None,
) -> dict[str, Any]:
    return _with_columns(
        _validate_pcmci_alpha_sensitivity,
        data,
        list(alphas),
        max_lag=max_lag,
        fdr=fdr,
        ci=ci,
        seed=seed,
        threads=threads,
    )


def validate_pcmci_lag_sensitivity(
    data: Mapping[str, Any] | Any,
    max_lags: Sequence[int],
    *,
    alpha: float = 0.05,
    fdr: bool = False,
    ci: str = "parcorr",
    seed: int = 1,
    threads: int | None = None,
) -> dict[str, Any]:
    return _with_columns(
        _validate_pcmci_lag_sensitivity,
        data,
        [int(m) for m in max_lags],
        alpha=alpha,
        fdr=fdr,
        ci=ci,
        seed=seed,
        threads=threads,
    )


def validate_pcmci_ci_sensitivity(
    data: Mapping[str, Any] | Any,
    ci_names: Sequence[str],
    *,
    max_lag: int = 1,
    alpha: float = 0.05,
    fdr: bool = False,
    seed: int = 1,
    threads: int | None = None,
) -> dict[str, Any]:
    return _with_columns(
        _validate_pcmci_ci_sensitivity,
        data,
        list(ci_names),
        max_lag=max_lag,
        alpha=alpha,
        fdr=fdr,
        seed=seed,
        threads=threads,
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
    seed: int = 1,
    threads: int | None = None,
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
        seed=seed,
        threads=threads,
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
    seed: int = 1,
    threads: int | None = None,
) -> dict[str, Any]:
    return _validate_synthetic_null_calibration(
        max_lag=max_lag,
        alpha=alpha,
        fdr=fdr,
        ci=ci,
        n_sim=n_sim,
        n_obs=n_obs,
        n_vars=n_vars,
        seed=seed,
        threads=threads,
    )


def validate_environment_holdout(
    data: Sequence[Mapping[str, Any] | Any],
    *,
    max_lag: int = 1,
    alpha: float = 0.05,
    fdr: bool = False,
    ci: str = "parcorr",
    n_discovery: int = 1,
    seed: int = 1,
    threads: int | None = None,
) -> dict[str, Any]:
    return _with_multi_env(
        _validate_environment_holdout,
        data,
        max_lag=max_lag,
        alpha=alpha,
        fdr=fdr,
        ci=ci,
        n_discovery=n_discovery,
        seed=seed,
        threads=threads,
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
    seed: int = 1,
    threads: int | None = None,
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
        seed=seed,
        threads=threads,
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


@dataclass(frozen=True)
class TargetRefutation:
    """The target-native suite of one prepared result: plan, reports, refusals, certificate."""

    target: RefutationTarget
    suite: str
    plan: tuple[str, ...]
    reports: tuple[TargetCheck, ...]
    refused: tuple[RefuterRefusal, ...]
    certificate: str
    calibration: str

    @property
    def passed(self) -> bool:
        """Whether every informative report passed."""
        return all(report.passed for report in self.reports if report.informative)

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
        )


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
    ``temporal_effect``, ``mediation``, ``world_valued`` or ``joint_contrast``; ``suite`` is
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
    """Posterior-predictive p-value of the target quantity: the observed target against
    targets recomputed on data replicated from each posterior draw."""
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
    "amplify",
    "derivative_consistency",
    "evalue",
    "local_linearity",
    "matched_sets_sensitivity",
    "mediation_sensitivity",
    "omitted_variable_bias",
    "omitted_variable_bias_from_data",
    "posterior_predictive_target",
    "prior_sensitivity_target",
    "rosenbaum_bounds",
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
