"""Experiment planning: power, sample size and minimum detectable effect.

A plan declares a randomized design, the assumptions its operating characteristics rest on,
a criterion and a goal, and solves in one of three directions: the smallest size whose
criterion reaches its target (:class:`SampleSize`), the criterion at a given size
(:class:`Evaluate`), or the minimum detectable effect at a given size
(:class:`MinimumDetectableEffect`). The request is compiled natively to a lowered program
that retains every assumption in its identity, then solved analytically and, when a
:class:`Simulation` is declared, simulated end to end (data generation, the analysis
estimator the design is read with, the planned decision)::

    from antecedent.design import planning

    plan = planning.plan_experiment(
        planning.Cluster(mean_cluster_size=20, icc=0.05),
        planning.Assumptions.continuous(sd=1.0, effect=0.3, attrition=0.1),
        planning.Power(0.8),
        planning.SampleSize(),
        simulation=planning.Simulation(replicates=2000, seed=7),
    )
    plan.units, plan.analytic.value, plan.simulation.value

Designs: :class:`Bernoulli`, :class:`Complete`, :class:`Stratified`, :class:`Cluster`,
:class:`MultiArm`, :class:`Factorial`, :class:`Switchback`, :class:`GroupSequential`, and
:class:`Interference` / :class:`OutcomeAdaptive`, which are refused (with the next attempt
named) unless an exposure model or a pre-specified rule makes their operating
characteristics definable. Criteria: :class:`Power`, :class:`Assurance`,
:class:`PosteriorPrecision` and :class:`CorrectDecision`.

Every planned characteristic is a point statement under the declared assumptions; its
``calibration`` is ``"unmeasured"``. A plan feeds design ranking through
:meth:`PlanningResult.sampling_plan` and :meth:`PlanningResult.experiment_plan`, whose
sample budget is the planned sample size.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any, Literal

from .. import _native
from ..errors import CausalTypeError, CausalValueError
from .plans import Experiment, Sampling

__all__ = [
    "AnalyticCharacteristics",
    "Assumptions",
    "Assurance",
    "Bernoulli",
    "Cluster",
    "Complete",
    "ContainedInClusters",
    "CorrectDecision",
    "Evaluate",
    "Factorial",
    "FamilyCharacteristics",
    "FamilySimulation",
    "GroupSequential",
    "Interference",
    "MinimumDetectableEffect",
    "MultiArm",
    "NormalPrior",
    "OutcomeAdaptive",
    "PlanningResult",
    "PosteriorPrecision",
    "PosteriorSuccess",
    "Power",
    "SampleSize",
    "SequentialCharacteristics",
    "SequentialSimulation",
    "Significance",
    "SimulatedCharacteristics",
    "Simulation",
    "Stratified",
    "Stratum",
    "Switchback",
    "minimum_detectable_effect",
    "operating_characteristics",
    "plan_experiment",
    "planning_program_identity",
    "sample_size",
]


def _number(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CausalTypeError(f"{what} must be a number, not {type(value).__name__}")
    if not math.isfinite(value):
        raise CausalValueError(f"{what} must be finite, got {value}")
    return float(value)


def _optional_number(value: object, what: str) -> float | None:
    return None if value is None else _number(value, what)


def _integer(value: object, what: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CausalTypeError(f"{what} must be an integer, not {type(value).__name__}")
    if value < minimum:
        raise CausalValueError(f"{what} must be >= {minimum}, got {value}")
    return int(value)


def _choice(value: object, what: str, allowed: tuple[str, ...]) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise CausalValueError(f"{what} must be one of {allowed}, got {value!r}")
    return value


# --- designs -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Bernoulli:
    """Each unit is treated independently with probability ``r / (1 + r)``."""

    def _wire(self) -> dict[str, Any]:
        return {"kind": "bernoulli"}


@dataclass(frozen=True)
class Complete:
    """Fixed arm sizes in the allocation ratio (complete randomization)."""

    def _wire(self) -> dict[str, Any]:
        return {"kind": "complete"}


@dataclass(frozen=True)
class Stratum:
    """A stratum: its share of the units and its within-stratum outcome scale.

    ``within`` is the within-stratum outcome standard deviation for a continuous outcome and
    the control-arm rate for a binary one.
    """

    share: float
    within: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "share", _number(self.share, "share"))
        object.__setattr__(self, "within", _number(self.within, "within"))


@dataclass(frozen=True)
class Stratified:
    """Complete randomization within strata, analysed by the block-weighted contrast."""

    strata: tuple[Stratum, ...]

    def __post_init__(self) -> None:
        strata = tuple(self.strata)
        if not strata or not all(isinstance(s, Stratum) for s in strata):
            raise CausalTypeError("strata must be a non-empty sequence of Stratum")
        object.__setattr__(self, "strata", strata)

    def _wire(self) -> dict[str, Any]:
        return {
            "kind": "stratified",
            "strata": [{"share": s.share, "within": s.within} for s in self.strata],
        }


@dataclass(frozen=True)
class Cluster:
    """Whole clusters are randomized; the size counts clusters.

    The design effect is ``1 + ((1 + cv²) m − 1) ρ``. With unequal sizes
    (``cluster_size_cv > 0``) the cluster-total analysis depends on the outcome level, so a
    continuous outcome needs ``control_mean``.
    """

    mean_cluster_size: float
    icc: float
    cluster_size_cv: float = 0.0

    def __post_init__(self) -> None:
        for name in ("mean_cluster_size", "icc", "cluster_size_cv"):
            object.__setattr__(self, name, _number(getattr(self, name), name))

    def _wire(self) -> dict[str, Any]:
        return {
            "kind": "cluster",
            "mean_cluster_size": self.mean_cluster_size,
            "icc": self.icc,
            "cluster_size_cv": self.cluster_size_cv,
        }


@dataclass(frozen=True)
class MultiArm:
    """``treatment_arms`` arms against one shared control.

    ``comparison="many_to_one"`` tests each arm against control under ``multiplicity``
    (``"bonferroni"``, ``"holm"`` or ``"dunnett"``) and powers ``family_power``
    (``"per_comparison"``, ``"any"`` or ``"all"``); ``comparison="omnibus"`` is the one-way F
    test.
    """

    treatment_arms: int
    comparison: Literal["many_to_one", "omnibus"] = "many_to_one"
    multiplicity: Literal["bonferroni", "holm", "dunnett"] = "dunnett"
    family_power: Literal["per_comparison", "any", "all"] = "any"

    def __post_init__(self) -> None:
        _integer(self.treatment_arms, "treatment_arms", minimum=2)
        _choice(self.comparison, "comparison", ("many_to_one", "omnibus"))
        _choice(self.multiplicity, "multiplicity", ("bonferroni", "holm", "dunnett"))
        _choice(self.family_power, "family_power", ("per_comparison", "any", "all"))

    def _wire(self) -> dict[str, Any]:
        return {
            "kind": "multi_arm",
            "treatment_arms": self.treatment_arms,
            "comparison": self.comparison,
            "multiplicity": self.multiplicity,
            "family_power": self.family_power,
        }


@dataclass(frozen=True)
class Factorial:
    """Balanced two-by-two factorial; ``contrast`` is ``"main_effect"`` or ``"interaction"``."""

    contrast: Literal["main_effect", "interaction"] = "main_effect"

    def __post_init__(self) -> None:
        _choice(self.contrast, "contrast", ("main_effect", "interaction"))

    def _wire(self) -> dict[str, Any]:
        return {"kind": "factorial", "contrast": self.contrast}


@dataclass(frozen=True)
class Switchback:
    """Sequences switch arms in randomized blocks of periods; the size counts blocks per sequence.

    ``autocorrelation`` is the first-order autocorrelation of the outcome between periods,
    ``carryover`` the share of the direct effect carried into the next period, and
    ``washout_periods`` the periods at the start of each block left out of the analysis.
    Carryover without a washout is refused.
    """

    sequences: int
    periods_per_block: int
    autocorrelation: float = 0.0
    carryover: float = 0.0
    washout_periods: int = 0

    def __post_init__(self) -> None:
        _integer(self.sequences, "sequences", minimum=2)
        _integer(self.periods_per_block, "periods_per_block", minimum=1)
        _integer(self.washout_periods, "washout_periods")
        object.__setattr__(
            self, "autocorrelation", _number(self.autocorrelation, "autocorrelation")
        )
        object.__setattr__(self, "carryover", _number(self.carryover, "carryover"))

    def _wire(self) -> dict[str, Any]:
        return {
            "kind": "switchback",
            "sequences": self.sequences,
            "periods_per_block": self.periods_per_block,
            "autocorrelation": self.autocorrelation,
            "carryover": self.carryover,
            "washout_periods": self.washout_periods,
        }


@dataclass(frozen=True)
class GroupSequential:
    """Two-arm trial with pre-specified looks and an alpha-spending efficacy boundary.

    ``spending`` is ``"obrien_fleming_type"`` or ``"pocock_type"``; the last information
    fraction is 1. The size counts the maximum enrolment.
    """

    information_fractions: tuple[float, ...]
    spending: Literal["obrien_fleming_type", "pocock_type"] = "obrien_fleming_type"

    def __post_init__(self) -> None:
        fractions = tuple(_number(f, "information fraction") for f in self.information_fractions)
        object.__setattr__(self, "information_fractions", fractions)
        _choice(self.spending, "spending", ("obrien_fleming_type", "pocock_type"))

    def _wire(self) -> dict[str, Any]:
        return {
            "kind": "group_sequential",
            "information_fractions": list(self.information_fractions),
            "spending": self.spending,
        }


@dataclass(frozen=True)
class ContainedInClusters:
    """Exposure model: interference is contained in clusters randomized whole."""

    mean_cluster_size: float
    icc: float
    cluster_size_cv: float = 0.0


@dataclass(frozen=True)
class Interference:
    """Units interfere; without an exposure model no operating characteristic is defined."""

    exposure: ContainedInClusters | None = None

    def _wire(self) -> dict[str, Any]:
        exposure = self.exposure
        return {
            "kind": "interference",
            "exposure": None
            if exposure is None
            else {
                "mean_cluster_size": _number(exposure.mean_cluster_size, "mean_cluster_size"),
                "icc": _number(exposure.icc, "icc"),
                "cluster_size_cv": _number(exposure.cluster_size_cv, "cluster_size_cv"),
            },
        }


@dataclass(frozen=True)
class OutcomeAdaptive:
    """Allocation or stopping adapts to outcomes. Always refused, naming the next attempt."""

    rule: str | None = None

    def _wire(self) -> dict[str, Any]:
        return {"kind": "outcome_adaptive", "rule": self.rule}


Design = (
    Bernoulli
    | Complete
    | Stratified
    | Cluster
    | MultiArm
    | Factorial
    | Switchback
    | GroupSequential
    | Interference
    | OutcomeAdaptive
)

# --- assumptions -------------------------------------------------------------------------


@dataclass(frozen=True)
class Assumptions:
    """The declared assumptions every operating characteristic rests on.

    Build with :meth:`continuous` or :meth:`binary`. ``effect`` is the assumed difference of
    means or proportions (omit it when the goal solves for it or a Bayesian criterion's design
    prior replaces it); ``alpha`` is the total level of a ``"two_sided"`` test.
    """

    outcome: Literal["continuous", "binary"]
    sd: float | None = None
    control_mean: float | None = None
    control_rate: float | None = None
    effect: float | None = None
    alpha: float = 0.05
    sidedness: Literal["two_sided", "one_sided"] = "two_sided"
    allocation_ratio: float = 1.0
    attrition: float = 0.0
    covariate_r_squared: float = 0.0
    covariates: int = 0

    def __post_init__(self) -> None:
        _choice(self.outcome, "outcome", ("continuous", "binary"))
        _choice(self.sidedness, "sidedness", ("two_sided", "one_sided"))
        for name in ("sd", "control_mean", "control_rate", "effect"):
            object.__setattr__(self, name, _optional_number(getattr(self, name), name))
        for name in ("alpha", "allocation_ratio", "attrition", "covariate_r_squared"):
            object.__setattr__(self, name, _number(getattr(self, name), name))
        _integer(self.covariates, "covariates")
        if self.outcome == "continuous" and self.sd is None:
            raise CausalValueError("a continuous outcome needs sd")
        if self.outcome == "binary" and self.control_rate is None:
            raise CausalValueError("a binary outcome needs control_rate")

    @classmethod
    def continuous(cls, sd: float, **declared: Any) -> Assumptions:
        """A continuous outcome of standard deviation ``sd``."""
        return cls(outcome="continuous", sd=sd, **declared)

    @classmethod
    def binary(cls, control_rate: float, **declared: Any) -> Assumptions:
        """A binary outcome with control-arm rate ``control_rate``."""
        return cls(outcome="binary", control_rate=control_rate, **declared)

    def _wire(self) -> dict[str, Any]:
        wire: dict[str, Any] = {
            "outcome": self.outcome,
            "effect": self.effect,
            "alpha": self.alpha,
            "sidedness": self.sidedness,
            "allocation_ratio": self.allocation_ratio,
            "attrition": self.attrition,
            "covariate_r_squared": self.covariate_r_squared,
            "covariates": self.covariates,
        }
        if self.outcome == "continuous":
            wire.update(sd=self.sd, control_mean=self.control_mean)
        else:
            wire.update(control_rate=self.control_rate)
        return wire


# --- criteria ----------------------------------------------------------------------------


@dataclass(frozen=True)
class NormalPrior:
    """A normal prior on the effect; ``sd = 0`` is a point design prior."""

    mean: float
    sd: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "mean", _number(self.mean, "prior mean"))
        object.__setattr__(self, "sd", _number(self.sd, "prior sd"))

    def _wire(self) -> dict[str, float]:
        return {"mean": self.mean, "sd": self.sd}


@dataclass(frozen=True)
class Power:
    """Frequentist power of the planned test at the assumed effect."""

    target: float = 0.8

    def _wire(self) -> dict[str, Any]:
        return {"kind": "power", "target": _number(self.target, "target")}


@dataclass(frozen=True)
class Significance:
    """Success is rejection by the planned frequentist test (hybrid assurance)."""

    def _wire(self) -> dict[str, Any]:
        return {"kind": "significance"}


@dataclass(frozen=True)
class PosteriorSuccess:
    """Success is ``P(effect > threshold | data) >= probability`` under ``analysis_prior``."""

    analysis_prior: NormalPrior
    threshold: float = 0.0
    probability: float = 0.95

    def _wire(self) -> dict[str, Any]:
        return {
            "kind": "posterior",
            "analysis_prior": self.analysis_prior._wire(),
            "threshold": _number(self.threshold, "threshold"),
            "probability": _number(self.probability, "probability"),
        }


@dataclass(frozen=True)
class Assurance:
    """Probability of success averaged over a design prior on the effect."""

    design_prior: NormalPrior
    success: Significance | PosteriorSuccess = field(default_factory=Significance)
    target: float = 0.8

    def _wire(self) -> dict[str, Any]:
        return {
            "kind": "assurance",
            "design_prior": self.design_prior._wire(),
            "success": self.success._wire(),
            "target": _number(self.target, "target"),
        }


@dataclass(frozen=True)
class PosteriorPrecision:
    """Posterior standard deviation at or below ``max_posterior_sd`` (conjugate normal)."""

    analysis_prior: NormalPrior
    max_posterior_sd: float

    @classmethod
    def credible_half_width(
        cls, analysis_prior: NormalPrior, level: float, half_width: float
    ) -> PosteriorPrecision:
        """The target that bounds a ``level`` credible interval's half-width."""
        level = _number(level, "level")
        if not 0.0 < level < 1.0:
            raise CausalValueError(f"level must lie in (0, 1), got {level}")
        z = _normal_quantile(0.5 * (1.0 + level))
        return cls(analysis_prior, _number(half_width, "half_width") / z)

    def _wire(self) -> dict[str, Any]:
        return {
            "kind": "posterior_precision",
            "analysis_prior": self.analysis_prior._wire(),
            "max_posterior_sd": _number(self.max_posterior_sd, "max_posterior_sd"),
        }


@dataclass(frozen=True)
class CorrectDecision:
    """Probability that the posterior decision about ``effect > threshold`` is right."""

    design_prior: NormalPrior
    analysis_prior: NormalPrior
    threshold: float = 0.0
    probability: float = 0.95
    target: float = 0.8

    def _wire(self) -> dict[str, Any]:
        return {
            "kind": "correct_decision",
            "design_prior": self.design_prior._wire(),
            "analysis_prior": self.analysis_prior._wire(),
            "threshold": _number(self.threshold, "threshold"),
            "probability": _number(self.probability, "probability"),
            "target": _number(self.target, "target"),
        }


Criterion = Power | Assurance | PosteriorPrecision | CorrectDecision


def _normal_quantile(p: float) -> float:
    return NormalDist().inv_cdf(p)


# --- goals and simulation ----------------------------------------------------------------


@dataclass(frozen=True)
class SampleSize:
    """Smallest size whose criterion reaches its target, searched up to ``max_units``."""

    max_units: int = 10_000_000

    def _wire(self) -> dict[str, Any]:
        return {
            "kind": "sample_size",
            "max_units": _integer(self.max_units, "max_units", minimum=1),
        }


@dataclass(frozen=True)
class Evaluate:
    """The criterion at ``units`` randomization units."""

    units: int

    def _wire(self) -> dict[str, Any]:
        return {"kind": "evaluate", "units": _integer(self.units, "units", minimum=1)}


@dataclass(frozen=True)
class MinimumDetectableEffect:
    """Smallest effect the planned test detects with the target power at ``units``."""

    units: int

    def _wire(self) -> dict[str, Any]:
        return {
            "kind": "minimum_detectable_effect",
            "units": _integer(self.units, "units", minimum=1),
        }


Goal = SampleSize | Evaluate | MinimumDetectableEffect


@dataclass(frozen=True)
class Simulation:
    """Whole-procedure simulation: ``replicates`` keyed replicates from ``seed``."""

    replicates: int
    seed: int = 0

    def _wire(self) -> dict[str, int]:
        return {
            "replicates": _integer(self.replicates, "replicates", minimum=1),
            "seed": _integer(self.seed, "seed"),
        }


# --- results -----------------------------------------------------------------------------


@dataclass(frozen=True)
class FamilyCharacteristics:
    """Familywise characteristics of many-to-one comparisons."""

    any: float
    all: float
    per_comparison: float
    critical: tuple[float, ...]
    correlation: float


@dataclass(frozen=True)
class SequentialCharacteristics:
    """Operating characteristics of a group-sequential boundary."""

    critical: tuple[float, ...]
    look_units: tuple[int, ...]
    stop_probability_alternative: tuple[float, ...]
    stop_probability_null: tuple[float, ...]
    expected_units_alternative: float
    expected_units_null: float
    fixed_design_units: int | None


@dataclass(frozen=True)
class AnalyticCharacteristics:
    """Analytic operating characteristics at the planned size."""

    value: float
    estimator_variance: float
    noncentrality: float | None
    reference_df: float | None
    critical_value: float | None
    design_effect: float | None
    power: float | None
    family: FamilyCharacteristics | None
    sequential: SequentialCharacteristics | None
    method: str


@dataclass(frozen=True)
class FamilySimulation:
    """Simulated familywise rejection frequencies."""

    any: float
    all: float
    per_comparison: float


@dataclass(frozen=True)
class SequentialSimulation:
    """Simulated stopping frequencies and enrolment of a sequential design."""

    stop_frequency: tuple[float, ...]
    expected_units: float
    expected_units_se: float


@dataclass(frozen=True)
class SimulatedCharacteristics:
    """Simulated characteristics with their Monte Carlo standard error."""

    value: float
    monte_carlo_se: float | None
    replicates: int
    seed: int
    failed_replicates: int
    mean_estimate: float | None
    estimate_sd: float | None
    family: FamilySimulation | None
    sequential: SequentialSimulation | None
    agreement_z: float | None


@dataclass(frozen=True)
class PlanningResult:
    """A plan: the size, the effect, and the operating characteristics with their identities."""

    program_identity: str
    request_identity: str
    design: str
    analysis: str
    criterion: str
    goal: str
    unit: str
    units: int
    arm_units: tuple[int, ...]
    enrolled_individuals: float
    analysed_individuals: float
    sample_size: int
    effect: float | None
    target: float
    target_met: bool
    analytic: AnalyticCharacteristics
    simulation: SimulatedCharacteristics | None
    assumptions: tuple[str, ...]
    calibration: str

    def sampling_plan(self, cost_per_individual: float = 0.0, tag: int | None = None) -> Sampling:
        """A :class:`~antecedent.design.Sampling` plan adding this plan's sample size."""
        cost = _number(cost_per_individual, "cost_per_individual")
        return Sampling(
            self.sample_size,
            cost=cost * self.sample_size,
            sample_budget=self.sample_size,
            tag=tag,
        )

    def experiment_plan(
        self, targets: Sequence[int], cost_per_individual: float = 0.0, tag: int | None = None
    ) -> Experiment:
        """An :class:`~antecedent.design.Experiment` plan whose sample budget is this plan's."""
        cost = _number(cost_per_individual, "cost_per_individual")
        return Experiment(
            tuple(targets), cost=cost * self.sample_size, sample_budget=self.sample_size, tag=tag
        )


def _tuple(values: Sequence[Any] | None) -> tuple[Any, ...]:
    return tuple(values or ())


def _family(wire: Mapping[str, Any] | None) -> FamilyCharacteristics | None:
    if wire is None:
        return None
    return FamilyCharacteristics(
        any=wire["any"],
        all=wire["all"],
        per_comparison=wire["per_comparison"],
        critical=_tuple(wire["critical"]),
        correlation=wire["correlation"],
    )


def _sequential(wire: Mapping[str, Any] | None) -> SequentialCharacteristics | None:
    if wire is None:
        return None
    return SequentialCharacteristics(
        critical=_tuple(wire["critical"]),
        look_units=_tuple(wire["look_units"]),
        stop_probability_alternative=_tuple(wire["stop_probability_alternative"]),
        stop_probability_null=_tuple(wire["stop_probability_null"]),
        expected_units_alternative=wire["expected_units_alternative"],
        expected_units_null=wire["expected_units_null"],
        fixed_design_units=wire["fixed_design_units"],
    )


def _simulation(wire: Mapping[str, Any] | None) -> SimulatedCharacteristics | None:
    if wire is None:
        return None
    family = wire["family"]
    sequential = wire["sequential"]
    return SimulatedCharacteristics(
        value=wire["value"],
        monte_carlo_se=wire["monte_carlo_se"],
        replicates=wire["replicates"],
        seed=wire["seed"],
        failed_replicates=wire["failed_replicates"],
        mean_estimate=wire["mean_estimate"],
        estimate_sd=wire["estimate_sd"],
        family=None if family is None else FamilySimulation(**family),
        sequential=None
        if sequential is None
        else SequentialSimulation(
            stop_frequency=_tuple(sequential["stop_frequency"]),
            expected_units=sequential["expected_units"],
            expected_units_se=sequential["expected_units_se"],
        ),
        agreement_z=wire["agreement_z"],
    )


def _result(wire: Mapping[str, Any]) -> PlanningResult:
    analytic = wire["analytic"]
    return PlanningResult(
        program_identity=wire["program_identity"],
        request_identity=wire["request_identity"],
        design=wire["design"],
        analysis=wire["analysis"],
        criterion=wire["criterion"],
        goal=wire["goal"],
        unit=wire["unit"],
        units=wire["units"],
        arm_units=_tuple(wire["arm_units"]),
        enrolled_individuals=wire["enrolled_individuals"],
        analysed_individuals=wire["analysed_individuals"],
        sample_size=wire["sample_size"],
        effect=wire["effect"],
        target=wire["target"],
        target_met=wire["target_met"],
        analytic=AnalyticCharacteristics(
            value=analytic["value"],
            estimator_variance=analytic["estimator_variance"],
            noncentrality=analytic["noncentrality"],
            reference_df=analytic["reference_df"],
            critical_value=analytic["critical_value"],
            design_effect=analytic["design_effect"],
            power=analytic["power"],
            family=_family(analytic["family"]),
            sequential=_sequential(analytic["sequential"]),
            method=analytic["method"],
        ),
        simulation=_simulation(wire["simulation"]),
        assumptions=_tuple(wire["assumptions"]),
        calibration=wire["calibration"],
    )


# --- entry points ------------------------------------------------------------------------


def _request(
    design: Design,
    assumptions: Assumptions,
    criterion: Criterion,
    goal: Goal,
    simulation: Simulation | None,
) -> str:
    checks = (
        (design, "design", Design),
        (assumptions, "assumptions", Assumptions),
        (criterion, "criterion", Criterion),
        (goal, "goal", Goal),
    )
    for value, what, kind in checks:
        if not isinstance(value, kind):
            raise CausalTypeError(f"{what} must be a planning {what}, not {type(value).__name__}")
    if simulation is not None and not isinstance(simulation, Simulation):
        raise CausalTypeError("simulation must be a Simulation or None")
    return json.dumps(
        {
            "design": design._wire(),
            "assumptions": assumptions._wire(),
            "criterion": criterion._wire(),
            "goal": goal._wire(),
            "simulation": None if simulation is None else simulation._wire(),
        },
        allow_nan=False,
    )


def plan_experiment(
    design: Design,
    assumptions: Assumptions,
    criterion: Criterion,
    goal: Goal,
    *,
    simulation: Simulation | None = None,
    threads: int = 1,
    cancel: Any = None,
) -> PlanningResult:
    """Compile and solve a plan; simulate it when ``simulation`` is declared.

    Raises the native refusal (with ``reason_code`` and the next attempt in its message) for
    an invalid request, a design whose operating characteristics cannot be established, a
    target the design does not reach within the search bound, or a simulation the admission
    pool cannot hold.
    """
    request = _request(design, assumptions, criterion, goal, simulation)
    threads = _integer(threads, "threads", minimum=1)
    return _result(json.loads(_native.plan_experiment_py(request, threads, cancel)))


def sample_size(
    design: Design,
    assumptions: Assumptions,
    criterion: Criterion | None = None,
    *,
    max_units: int = 10_000_000,
    simulation: Simulation | None = None,
    threads: int = 1,
) -> PlanningResult:
    """Smallest size whose criterion (default 80 percent power) reaches its target."""
    return plan_experiment(
        design,
        assumptions,
        Power() if criterion is None else criterion,
        SampleSize(max_units),
        simulation=simulation,
        threads=threads,
    )


def operating_characteristics(
    design: Design,
    assumptions: Assumptions,
    units: int,
    criterion: Criterion | None = None,
    *,
    simulation: Simulation | None = None,
    threads: int = 1,
) -> PlanningResult:
    """The criterion (default power) of the design at ``units`` randomization units."""
    return plan_experiment(
        design,
        assumptions,
        Power() if criterion is None else criterion,
        Evaluate(units),
        simulation=simulation,
        threads=threads,
    )


def minimum_detectable_effect(
    design: Design,
    assumptions: Assumptions,
    units: int,
    power: float = 0.8,
    *,
    simulation: Simulation | None = None,
    threads: int = 1,
) -> PlanningResult:
    """Smallest effect detected with ``power`` at ``units``; read it from ``result.effect``."""
    return plan_experiment(
        design,
        assumptions,
        Power(power),
        MinimumDetectableEffect(units),
        simulation=simulation,
        threads=threads,
    )


def planning_program_identity(
    design: Design,
    assumptions: Assumptions,
    criterion: Criterion,
    goal: Goal,
    *,
    simulation: Simulation | None = None,
) -> str:
    """Identity of the lowered program the request compiles to; nothing is solved."""
    request = _request(design, assumptions, criterion, goal, simulation)
    return str(_native.planning_program_identity_py(request))
