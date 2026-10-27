"""Typed causal queries for the Python facade.

Every query dataclass keeps a short positional prefix of the fields that
identify *which variables* the query is about (treatment/outcome/mediator/…);
everything after that — contrasts, levels, options — is keyword-only, via the
``KW_ONLY`` sentinel. ``kind`` is a discriminator, never a caller-set value:
it is ``init=False`` on every class.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field
from math import isfinite
from typing import Literal

from ._columns import coerce_fields
from ._native import temporal_response_spec as _native_temporal_response_spec
from ._validate import count as _count
from .errors import CausalValueError


def _require_name(field_name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise CausalValueError(f"{field_name} must be a non-empty variable name")


def _require_names(field_name: str, values: Sequence[str]) -> None:
    if len(values) == 0:
        raise CausalValueError(f"{field_name} must contain at least one variable name")
    for value in values:
        _require_name(field_name, value)


def _require_finite(field_name: str, value: float) -> None:
    if not isfinite(value):
        raise CausalValueError(f"{field_name} must be finite, got {value!r}")


@dataclass(frozen=True, slots=True)
class Mean:
    """Default outcome functional ``E[Y(a)]``."""

    kind: Literal["mean"] = field(default="mean", init=False, repr=False)


@dataclass(frozen=True, slots=True)
class Exceedance:
    """Tail probability ``P(Y(a) > threshold)``."""

    threshold: float
    _: KW_ONLY
    kind: Literal["exceedance"] = field(default="exceedance", init=False, repr=False)

    def __post_init__(self) -> None:
        _require_finite("threshold", self.threshold)


@dataclass(frozen=True, slots=True)
class ExceedanceGrid:
    """Exceedance on a strictly increasing threshold grid."""

    thresholds: Sequence[float]
    _: KW_ONLY
    kind: Literal["exceedance_grid"] = field(default="exceedance_grid", init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.thresholds:
            raise CausalValueError("exceedance grid must be non-empty")
        prev = float("-inf")
        for value in self.thresholds:
            _require_finite("thresholds", float(value))
            if float(value) <= prev:
                raise CausalValueError("exceedance grid must be strictly increasing")
            prev = float(value)


@dataclass(frozen=True, slots=True)
class Quantile:
    """Finite-grid quantile ``F_a^{-1}(tau)`` on licensed Frequentist CDF paths.

    Effect queries return active minus control quantiles. ConditionalEffect
    standardizes arm CDFs over the retained modifier distribution before inversion.
    Joint InterventionResponse returns the requested cell's quantile level.
    Requires no refutation; uncertainty excludes grid selection and interpolation bias.
    """

    tau: float
    _: KW_ONLY
    kind: Literal["quantile"] = field(default="quantile", init=False, repr=False)

    def __post_init__(self) -> None:
        _require_finite("tau", self.tau)
        if not 0.0 < float(self.tau) < 1.0:
            raise CausalValueError("tau must lie in (0, 1)")


def coerce_outcome_functional(spec: object | None) -> dict[str, object] | None:
    """Normalize an outcome functional to the native wire dict."""
    if spec is None or isinstance(spec, Mean) or spec == "mean":
        return None
    if isinstance(spec, Exceedance):
        return {"kind": "exceedance", "threshold": float(spec.threshold)}
    if isinstance(spec, ExceedanceGrid):
        return {"kind": "exceedance_grid", "thresholds": [float(c) for c in spec.thresholds]}
    if isinstance(spec, Quantile):
        return {"kind": "quantile", "tau": float(spec.tau)}
    if isinstance(spec, Mapping):
        kind = str(spec.get("kind", "mean")).lower()
        if kind == "mean":
            return None
        if kind == "exceedance":
            return {"kind": "exceedance", "threshold": float(spec["threshold"])}
        if kind in {"exceedance_grid", "grid"}:
            return {
                "kind": "exceedance_grid",
                "thresholds": [float(c) for c in spec["thresholds"]],
            }
        if kind == "quantile":
            return {"kind": "quantile", "tau": float(spec["tau"])}
        raise CausalValueError(f"unknown outcome_functional kind {kind!r}")
    raise CausalValueError(f"unsupported outcome_functional type: {type(spec)!r}")


@dataclass(frozen=True, slots=True)
class TemporalResponseSpec:
    """Licensed temporal-response query policy, supplied by Rust.

    Python may duplicate error checking against this object. It must not
    define the cap, allowed policies, or lag default itself.
    """

    max_horizons: int
    max_cells: int
    allowed_policies: tuple[str, ...]
    default_policy: str
    default_treatment_lag: int


def _load_temporal_response_spec() -> TemporalResponseSpec:
    raw = _native_temporal_response_spec()
    return TemporalResponseSpec(
        max_horizons=int(raw["max_horizons"]),
        max_cells=int(raw["max_cells"]),
        allowed_policies=tuple(str(policy) for policy in raw["allowed_policies"]),
        default_policy=str(raw["default_policy"]),
        default_treatment_lag=int(raw["default_treatment_lag"]),
    )


temporal_response_spec = _load_temporal_response_spec()


def _validate_horizons(horizons: Sequence[int]) -> None:
    """Nonempty, ≥1, strictly increasing, capped — shared with TemporalResponseSpec."""
    if len(horizons) == 0:
        raise CausalValueError("horizons must be non-empty")
    if len(horizons) > temporal_response_spec.max_horizons:
        raise CausalValueError(
            f"horizons must contain at most {temporal_response_spec.max_horizons} entries"
        )
    prev_h: int | None = None
    for h in horizons:
        h = _count(h, "horizon", minimum=1, maximum=2**32 - 1, type_error=CausalValueError)
        if prev_h is not None and h <= prev_h:
            raise CausalValueError("horizons must be strictly increasing")
        prev_h = h


def _validate_max_history_lag(max_history_lag: int | None) -> None:
    """``max_history_lag`` is ``None`` or a nonnegative integer step count."""
    if max_history_lag is None:
        return
    _count(
        max_history_lag,
        "max_history_lag",
        minimum=0,
        maximum=2**32 - 1,
        type_error=CausalValueError,
    )
    if max_history_lag < 0:
        raise CausalValueError("max_history_lag must be non-negative")


@dataclass(frozen=True, slots=True)
class FeedbackMechanism:
    """Declared delayed-feedback dynamics ``s <- a s + b u + e`` of a temporal response.

    The dynamics are declared, not fitted: ``autoregression`` is ``a``,
    ``input_gain`` is ``b``, ``innovation_variance`` is the variance of ``e``, and
    ``initial_state`` is the level of the ``initial_law`` at the origin
    (``None`` is the contemporaneous observed law, a string names a law).
    ``delay`` is the feedback delay in ticks before the action re-enters the
    state; it is not ``treatment_lag``. ``level`` is the dose the pulse or
    sustained policy holds. A response that declares one publishes the
    finite-horizon rollout of these dynamics at every requested horizon instead
    of the fitted dose-by-horizon surface.
    """

    _: KW_ONLY
    delay: int
    level: float
    autoregression: float
    input_gain: float
    innovation_variance: float
    initial_law: str | None = None
    initial_state: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "delay", _count(self.delay, "delay", minimum=1, maximum=2**32 - 1))
        numbers = {
            "level": self.level,
            "autoregression": self.autoregression,
            "input_gain": self.input_gain,
            "innovation_variance": self.innovation_variance,
            "initial_state": self.initial_state,
        }
        for name, value in numbers.items():
            _require_finite(f"feedback {name}", value)
        if self.innovation_variance < 0:
            raise CausalValueError("feedback innovation_variance must not be negative")
        if self.initial_law is not None:
            _require_name("initial_law", self.initial_law)


def _validate_temporal(
    horizons: Sequence[int] | None,
    policy: str,
    treatment_lag: int,
    max_history_lag: int | None,
    feedback: FeedbackMechanism | None = None,
) -> None:
    """Shared temporal-attachment validation for ``ResponseCurve`` and
    ``InterventionResponse`` (ADR 0021). No-op when ``horizons`` is ``None``
    (the query is a static, non-temporal cell). Policy values come from
    :data:`temporal_response_spec`.
    """
    if horizons is None:
        if feedback is not None:
            raise CausalValueError("feedback requires horizons; a static cell has no dynamics")
        return
    if feedback is not None and not isinstance(feedback, FeedbackMechanism):
        raise CausalValueError("feedback must be a FeedbackMechanism")
    _validate_horizons(horizons)
    if policy not in temporal_response_spec.allowed_policies:
        allowed = "' or '".join(temporal_response_spec.allowed_policies)
        raise CausalValueError(f"policy must be '{allowed}'")
    if treatment_lag < 0:
        raise CausalValueError("treatment_lag must be non-negative")
    if max_history_lag is not None and max_history_lag < 0:
        raise CausalValueError("max_history_lag must be non-negative")


def _coordinate_values(
    field_name: str,
    coordinates: Sequence[float] | Mapping[str, float],
    treatments: Sequence[str],
) -> Sequence[float]:
    if isinstance(coordinates, Mapping):
        expected, actual = set(treatments), set(coordinates)
        if actual != expected:
            raise CausalValueError(
                f"{field_name} mapping keys must exactly match treatments; "
                f"missing={sorted(expected - actual)!r}, extra={sorted(actual - expected)!r}"
            )
        return [coordinates[name] for name in treatments]
    if len(coordinates) != len(treatments):
        raise CausalValueError(f"{field_name} must have one value per treatment")
    return coordinates


@dataclass(frozen=True, slots=True)
class AverageEffect:
    """Average treatment effect (static tabular).

    ``control_level`` and ``active_level`` name the two arms of the treatment column: numbers
    for a numerically coded column, or both labels (``str``) for a string- or category-coded
    one (``control_level="a", active_level="b"``). Labels are mapped to the arm codes and run
    the declared-level contrast of the numeric path; a row at any other label is refused
    (``treatment_level_undeclared``).
    """

    treatment: str
    outcome: str
    _: KW_ONLY
    control_level: float | str = 0.0
    active_level: float | str = 1.0
    target_population: object | None = None
    outcome_functional: object | None = None
    kind: Literal["average"] = field(default="average", init=False, repr=False)


@dataclass(frozen=True, slots=True)
class PulseEffect:
    """Temporal pulse intervention effect.

    ``max_history_lag`` bounds how many steps back the temporal unfolding
    looks for an adjustment set: covariates older than that many steps before
    the treatment or outcome are not adjusted for, and identification that
    needs them refuses and names this field. ``None`` (default) lets the
    unfolding grow until the adjustment set is certified or the graph's own
    chain bound is reached.

    ``weights="w"`` names a column of sampling weights, one per time step. Each lag window is
    weighted by the weight of the observation it ends at: the effect is the weighted-least-squares
    coefficient, windows keep their consecutive times (a window of weight zero stays, contributes
    nothing and is counted), weights are normalized to mean one, and the uncertainty is the
    dependence-honest circular-block bootstrap of the weighted windows. It is read by the
    frequentist single-window fit with ``refute=False``; sustained windows, conditional effects,
    Bayesian fits, refuters and series with gaps refuse (``option_not_applicable``).
    """

    treatment: str
    outcome: str
    _: KW_ONLY
    control_level: float = 0.0
    active_level: float = 1.0
    treatment_lag: int = temporal_response_spec.default_treatment_lag
    horizon_steps: int = 1
    max_history_lag: int | None = None
    target_population: object | None = None
    weights: str | None = None
    kind: Literal["pulse"] = field(default="pulse", init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_max_history_lag(self.max_history_lag)
        if self.weights is not None and (
            not isinstance(self.weights, str)
            or not self.weights.strip()
            or self.weights in (self.treatment, self.outcome)
        ):
            raise CausalValueError("weights must name a data column distinct from the variables")


@dataclass(frozen=True, slots=True)
class SustainedEffect:
    """Temporal sustained intervention effect.

    By default the intervention is at ``-treatment_lag``. ``window=(from_, until)``
    replaces that default with inclusive signed time offsets. Multi-step windows
    use sequential Gaussian g-computation on an explicit or accepted TemporalDag,
    or a Bayesian DBN graph-posterior mixture, with ``refute="none"``; the
    outcome is at ``horizon_steps - 1``. Cheap/full stay single-step.
    ``max_history_lag`` bounds the unfolding's history as on :class:`PulseEffect`.
    """

    treatment: str
    outcome: str
    _: KW_ONLY
    control_level: float = 0.0
    active_level: float = 1.0
    treatment_lag: int = temporal_response_spec.default_treatment_lag
    horizon_steps: int = 1
    window: tuple[int, int] | None = None
    max_history_lag: int | None = None
    target_population: object | None = None
    kind: Literal["sustained"] = field(default="sustained", init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_max_history_lag(self.max_history_lag)
        if self.window is not None:
            if len(self.window) != 2 or any(type(x) is not int for x in self.window):
                raise ValueError("window must be a pair of integer offsets")
            if self.window[0] > self.window[1]:
                raise ValueError("window must have from <= until")


@dataclass(frozen=True, slots=True)
class TemporalConditionalPulse:
    """One-modifier conditional pulse on a temporal DAG.

    The published point is the marginal effect ``(β_T + β_{T×W} · mean(W)) · (active − control)``
    on the unfolded temporal design. No interval is published.
    """

    treatment: str
    outcome: str
    modifier: str
    _: KW_ONLY
    control_level: float = 0.0
    active_level: float = 1.0
    treatment_lag: int = temporal_response_spec.default_treatment_lag
    horizon_steps: int = 1
    max_history_lag: int | None = None
    target_population: object | None = None
    kind: Literal["pulse"] = field(default="pulse", init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_max_history_lag(self.max_history_lag)


@dataclass(frozen=True, slots=True)
class TemporalConditionalSustained:
    """One-modifier conditional sustained effect on a single time step.

    Same marginal functional as :class:`TemporalConditionalPulse`. A multi-step
    window is not this license.
    """

    treatment: str
    outcome: str
    modifier: str
    _: KW_ONLY
    control_level: float = 0.0
    active_level: float = 1.0
    treatment_lag: int = temporal_response_spec.default_treatment_lag
    horizon_steps: int = 1
    window: tuple[int, int] | None = None
    max_history_lag: int | None = None
    target_population: object | None = None
    kind: Literal["sustained"] = field(default="sustained", init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_max_history_lag(self.max_history_lag)
        if self.window is not None and self.window != (
            -self.treatment_lag,
            -self.treatment_lag,
        ):
            raise ValueError("temporal conditional sustained effect is a single step")


@dataclass(frozen=True, slots=True)
class InterventionalDistribution:
    """Interventional distribution query (static)."""

    outcome: str
    _: KW_ONLY
    interventions: dict[str, float] = field(default_factory=dict)
    conditioning: Sequence[str] = ()
    target_population: object | None = None
    kind: Literal["distribution"] = field(default="distribution", init=False, repr=False)


@dataclass(frozen=True, slots=True)
class PathSpecificEffect:
    """Path-specific effect query (static)."""

    treatment: str
    outcome: str
    _: KW_ONLY
    path_nodes: Sequence[str] | None = None
    control_level: float = 0.0
    active_level: float = 1.0
    max_paths: int = 64
    max_len: int = 16
    target_population: object | None = None
    kind: Literal["path_specific"] = field(default="path_specific", init=False, repr=False)


@dataclass(frozen=True, slots=True)
class ConditionalEffect:
    """Conditional / context average effect given one effect modifier or a set of them.

    ``modifier`` is a column name or a non-empty sequence of distinct column
    names; :attr:`modifiers` is the normalized tuple. The frequentist
    interaction regression fits the whole set (the vector generalization of
    the one-modifier model), but its interval's coverage is measured for one
    modifier only, so a result for a modifier set carries the assumption
    ``conditional.joint_modifier_interval_uncalibrated``. Empty, repeated or
    non-string modifiers raise ``CausalValueError``.
    """

    treatment: str
    outcome: str
    modifier: str | Sequence[str]
    _: KW_ONLY
    control_level: float = 0.0
    active_level: float = 1.0
    target_population: object | None = None
    outcome_functional: object | None = None
    kind: Literal["conditional"] = field(default="conditional", init=False, repr=False)

    def __post_init__(self) -> None:
        raw = self.modifier
        if isinstance(raw, str):
            names: tuple[str, ...] = (raw,)
        elif isinstance(raw, Sequence):
            names = tuple(raw)
        else:
            raise CausalValueError(
                "ConditionalEffect modifier must be a column name or a sequence of column names"
            )
        if not names:
            raise CausalValueError("ConditionalEffect modifier must name at least one column")
        for name in names:
            _require_name("ConditionalEffect modifier", name)
        if len(set(names)) != len(names):
            raise CausalValueError("ConditionalEffect modifiers must be distinct column names")
        object.__setattr__(self, "modifier", names[0] if isinstance(raw, str) else names)

    @property
    def modifiers(self) -> tuple[str, ...]:
        """The effect modifiers as a tuple of column names."""
        raw = self.modifier
        return (raw,) if isinstance(raw, str) else tuple(raw)


@dataclass(frozen=True, slots=True)
class MediationEffect:
    """Static natural mediation on a linear additive DAG (no T-M interaction).

    ``direct`` / ``mediated`` alias natural direct / indirect effects in this
    model. Recanting and empty path sets refuse.
    """

    treatment: str
    outcome: str
    _: KW_ONLY
    mediators: Sequence[str]
    contrast: Literal["total", "direct", "mediated", "natural_direct", "natural_indirect"] = (
        "mediated"
    )
    control_level: float = 0.0
    active_level: float = 1.0
    target_population: object | None = None
    kind: Literal["mediation"] = field(default="mediation", init=False, repr=False)


@dataclass(frozen=True, slots=True)
class AnomalyReference:
    """Fixed location/scale for the anomaly IT score's marginal tail.

    The IT anomaly score is ``-log 2Φ(-|y - center|/scale)``. By default the
    reference is estimated from the target's observed marginal (robust median /
    ``1.4826·MAD``), so the score depends on the realized sample. Injecting a
    fixed ``(center, scale)`` makes the population mean anomaly score a fixed
    functional of the target's law, ``μ_A = E_Y[-log 2Φ(-|Y - center|/scale)]``.
    """

    center: float
    scale: float

    def __post_init__(self) -> None:
        center = float(self.center)
        scale = float(self.scale)
        if not isfinite(center) or not isfinite(scale) or scale <= 0.0:
            raise CausalValueError(
                "anomaly reference requires a finite center and a finite positive scale"
            )


@dataclass(frozen=True, slots=True)
class AnomalyAttribution:
    """GCM anomaly scores on a supplied explicit Dag.

    Licensed ``analyze(data, graph=Dag, query=AnomalyAttribution(...))`` at
    validation ``none``. Identification is parametric (``gcm.parametric``);
    estimator ``gcm.fit``. accepted, graph-posterior, and non-Dag graphs remain
    refused. A Bayesian run publishes a shared-Dirichlet-row-weight posterior of
    the target mean anomaly score; with a fixed ``reference`` that posterior is
    the Bayesian bootstrap of a fixed population functional.
    """

    targets: Sequence[str]
    _: KW_ONLY
    max_units: int = 100
    reference: AnomalyReference | None = None
    kind: Literal["anomaly_attribution"] = field(
        default="anomaly_attribution", init=False, repr=False
    )

    def __post_init__(self) -> None:
        _require_names("targets", self.targets)
        if int(self.max_units) <= 0:
            raise CausalValueError("max_units must be positive")
        if self.reference is not None and not isinstance(self.reference, AnomalyReference):
            raise CausalValueError("reference must be an AnomalyReference or None")


@dataclass(frozen=True, slots=True)
class ChangeAttribution:
    """GCM distribution-change Shapley on a supplied explicit Dag.

    Licensed ``analyze(data, graph=Dag, query=ChangeAttribution(...))`` at
    validation ``none``. Identification is parametric (``gcm.parametric``);
    estimator ``gcm.fit``. cheap/full, Bayesian, accepted, graph-posterior,
    and non-Dag graphs remain refused.
    """

    outcome: str
    _: KW_ONLY
    baseline_start: int
    baseline_end: int
    comparison_start: int
    comparison_end: int
    kind: Literal["change_attribution"] = field(
        default="change_attribution", init=False, repr=False
    )

    def __post_init__(self) -> None:
        _require_name("outcome", self.outcome)
        if int(self.baseline_end) <= int(self.baseline_start):
            raise CausalValueError("baseline range must be non-empty")
        if int(self.comparison_end) <= int(self.comparison_start):
            raise CausalValueError("comparison range must be non-empty")


@dataclass(frozen=True, slots=True)
class Counterfactual:
    """Two-world unit ITE ``Y(a) - Y(a0)`` via GCM abduction–action–prediction.

    This is not a factual-versus-counterfactual residual. Frequentist sampling
    uncertainty is unavailable. Bayesian ``none`` publishes a mechanism
    posterior and per-draw AAP unit effects. Nested and accepted forms stay
    refused.
    """

    treatment: str
    outcome: str
    _: KW_ONLY
    control_level: float = 0.0
    active_level: float = 1.0
    kind: Literal["counterfactual"] = field(default="counterfactual", init=False, repr=False)


@dataclass(frozen=True, slots=True)
class NestedCounterfactual:
    """Natural direct effect ``Y(1, M(0)) - Y(0, M(0))``.

    Licensed for one explicit three-node Markovian DAG with linear Gaussian
    mechanisms. The same abduced exogenous table is used in both worlds.
    """

    treatment: str
    mediator: str
    outcome: str
    _: KW_ONLY
    control_level: float = 0.0
    active_level: float = 1.0
    kind: Literal["nested_counterfactual"] = field(
        default="nested_counterfactual", init=False, repr=False
    )

    def __post_init__(self) -> None:
        _require_name("treatment", self.treatment)
        _require_name("mediator", self.mediator)
        _require_name("outcome", self.outcome)
        if len({self.treatment, self.mediator, self.outcome}) != 3:
            raise CausalValueError("treatment, mediator, and outcome must be distinct")
        if not isfinite(float(self.control_level)) or not isfinite(float(self.active_level)):
            raise CausalValueError("nested treatment levels must be finite")
        if float(self.control_level) == float(self.active_level):
            raise CausalValueError("nested treatment levels must be distinct")


@dataclass(frozen=True, slots=True)
class TemporalMediationEffect:
    """Temporal linear mediation (treatment → mediator → outcome).

    ``horizons=None`` defaults to ``[1]``. Multiple horizons return a
    horizon-indexed mediation grid; each Bayesian slice is pointwise and does
    not imply a joint cross-horizon posterior.
    """

    treatment: str
    mediator: str
    outcome: str
    _: KW_ONLY
    contrast: Literal["total", "direct", "mediated"] = "mediated"
    control_level: float = 0.0
    active_level: float = 1.0
    horizons: Sequence[int] | None = None
    target_population: object | None = None
    kind: Literal["temporal_mediation"] = field(
        default="temporal_mediation", init=False, repr=False
    )

    def __post_init__(self) -> None:
        hs = (1,) if self.horizons is None else tuple(self.horizons)
        _validate_horizons(hs)
        coerce_fields(self, horizons=hs)


@dataclass(frozen=True, slots=True)
class ResponseCurve:
    """Mean causal response ``a -> E[Y | do(A=a)]`` on an explicit grid.

    Keyword-only ``horizons`` / ``policy`` / ``max_history_lag`` attach a temporal
    dose × horizon surface (ADR 0021). When set, analysis requires series data and
    a ``TemporalDag``, ``TemporalCpdag``, or ``TemporalPag``. Class graphs retain
    per-completion responses and structural uncertainty. ``treatment_lag`` defaults to
    :attr:`temporal_response_spec.default_treatment_lag` (policy origin ``-lag``),
    matching :class:`PulseEffect` / :class:`SustainedEffect`. Keyword-only
    ``feedback`` declares a :class:`FeedbackMechanism`: the response is then the
    finite-horizon rollout of the declared dynamics, not the fitted surface.
    """

    treatment: str
    outcome: str
    _: KW_ONLY
    grid: Sequence[float]
    target_population: object | None = None
    observation: object | None = None
    observation_assumptions: Sequence[object] = ()
    horizons: Sequence[int] | None = None
    policy: str = temporal_response_spec.default_policy
    treatment_lag: int = temporal_response_spec.default_treatment_lag
    max_history_lag: int | None = None
    feedback: FeedbackMechanism | None = None
    kind: Literal["response_curve"] = field(default="response_curve", init=False, repr=False)

    def __post_init__(self) -> None:
        _require_name("outcome", self.outcome)
        _require_name("treatment", self.treatment)
        if len(self.grid) < 2:
            raise CausalValueError("grid must contain at least two treatment values")
        previous: float | None = None
        for value in self.grid:
            _require_finite("grid values", value)
            if previous is not None and value <= previous:
                raise CausalValueError("grid values must be strictly increasing")
            previous = value
        if self.horizons is not None:
            object.__setattr__(
                self,
                "horizons",
                tuple(
                    _count(h, f"horizons[{i}]", minimum=1, maximum=2**32 - 1)
                    for i, h in enumerate(self.horizons)
                ),
            )
        object.__setattr__(
            self,
            "treatment_lag",
            _count(self.treatment_lag, "treatment_lag", minimum=0, maximum=2**32 - 1),
        )
        if self.max_history_lag is not None:
            object.__setattr__(
                self,
                "max_history_lag",
                _count(self.max_history_lag, "max_history_lag", minimum=0, maximum=2**32 - 1),
            )
        _validate_temporal(
            self.horizons,
            self.policy,
            self.treatment_lag,
            self.max_history_lag,
            self.feedback,
        )
        if (
            self.horizons is not None
            and len(self.grid) * len(self.horizons) > temporal_response_spec.max_cells
        ):
            cells = len(self.grid) * len(self.horizons)
            raise CausalValueError(
                f"temporal response has {cells} dose-by-horizon cells; "
                f"materialization limit is {temporal_response_spec.max_cells}; "
                "coarsen the dose grid or request fewer horizons"
            )

    @property
    def is_temporal(self) -> bool:
        return self.horizons is not None


@dataclass(frozen=True, slots=True)
class AverageDerivative:
    """Population-average derivative of a continuous-treatment response."""

    treatment: str
    outcome: str
    _: KW_ONLY
    weighting: object | None = None
    target_population: object | None = None
    observation: object | None = None
    observation_assumptions: Sequence[object] = ()
    kind: Literal["average_derivative"] = field(
        default="average_derivative", init=False, repr=False
    )

    def __post_init__(self) -> None:
        _require_name("outcome", self.outcome)
        _require_name("treatment", self.treatment)


@dataclass(frozen=True, slots=True)
class PointDerivative:
    """Local derivative of a response curve at one treatment value."""

    treatment: str
    outcome: str
    _: KW_ONLY
    at: float
    order: int = 1
    target_population: object | None = None
    observation: object | None = None
    observation_assumptions: Sequence[object] = ()
    kind: Literal["point_derivative"] = field(default="point_derivative", init=False, repr=False)

    def __post_init__(self) -> None:
        _require_name("outcome", self.outcome)
        _require_name("treatment", self.treatment)
        _require_finite("at", self.at)
        if self.order not in (1, 2):
            raise CausalValueError(f"order must be 1 or 2, got {self.order!r}")


@dataclass(frozen=True, slots=True)
class Elasticity:
    """Log-outcome/log-treatment derivative at a positive treatment value."""

    treatment: str
    outcome: str
    _: KW_ONLY
    at: float
    target_population: object | None = None
    observation: object | None = None
    observation_assumptions: Sequence[object] = ()
    kind: Literal["elasticity"] = field(default="elasticity", init=False, repr=False)

    def __post_init__(self) -> None:
        _require_name("outcome", self.outcome)
        _require_name("treatment", self.treatment)
        _require_finite("at", self.at)
        if self.at <= 0.0:
            raise CausalValueError("Elasticity.at must be positive for the log-treatment scale")


@dataclass(frozen=True, slots=True)
class SemiElasticity:
    """Derivative using one logarithmic scale and one identity scale."""

    treatment: str
    outcome: str
    _: KW_ONLY
    at: float
    log_scale: Literal["treatment", "outcome"] = "treatment"
    target_population: object | None = None
    observation: object | None = None
    observation_assumptions: Sequence[object] = ()
    kind: Literal["semi_elasticity"] = field(default="semi_elasticity", init=False, repr=False)

    def __post_init__(self) -> None:
        _require_name("outcome", self.outcome)
        _require_name("treatment", self.treatment)
        _require_finite("at", self.at)
        if self.log_scale not in ("treatment", "outcome"):
            raise CausalValueError(
                f"log_scale must be 'treatment' or 'outcome', got {self.log_scale!r}"
            )
        if self.log_scale == "treatment" and self.at <= 0.0:
            raise CausalValueError("SemiElasticity.at must be positive when log_scale='treatment'")


@dataclass(frozen=True, slots=True)
class DirectionalDerivative:
    """Plug-in response derivative along a supplied direction (not renormalized).

    The additive-GAM plug-in gradient dotted with
    ``direction``. Frequentist results are point-only: ``uncertainty`` is empty
    and the warning ``response.derivative_interval_withheld`` explains that its
    coefficient-sandwich band is wired for calibration but stays closed until
    its coverage is measured. Bayesian inference publishes a pointwise credible
    band.
    """

    treatments: Sequence[str]
    outcomes: Sequence[str]
    _: KW_ONLY
    at: Sequence[float] | Mapping[str, float]
    direction: Sequence[float] | Mapping[str, float]
    target_population: object | None = None
    observation: object | None = None
    observation_assumptions: Sequence[object] = ()
    kind: Literal["directional_derivative"] = field(
        default="directional_derivative", init=False, repr=False
    )

    def __post_init__(self) -> None:
        _require_names("outcomes", self.outcomes)
        _require_names("treatments", self.treatments)
        at = _coordinate_values("at", self.at, self.treatments)
        direction = _coordinate_values("direction", self.direction, self.treatments)
        for value in at:
            _require_finite("at values", value)
        for value in direction:
            _require_finite("direction values", value)
        if not any(value != 0.0 for value in direction):
            raise CausalValueError("direction must contain at least one non-zero value")


@dataclass(frozen=True, slots=True)
class ResponseJacobian:
    """Low-dimensional Jacobian of outcomes with respect to treatments.

    Row-major outcomes x treatments from the additive-GAM plug-in gradient.
    Frequentist results are point-only: ``uncertainty``
    is empty and the warning ``response.derivative_interval_withheld`` explains
    that its coefficient-sandwich band is wired for calibration but stays closed
    until its coverage is measured. Bayesian inference publishes a pointwise
    credible band. The scalar :class:`Elasticity` query is a different
    estimator and publishes a Frequentist Fieller interval.
    """

    treatments: Sequence[str]
    outcomes: Sequence[str]
    _: KW_ONLY
    at: Sequence[float] | Mapping[str, float]
    target_population: object | None = None
    observation: object | None = None
    observation_assumptions: Sequence[object] = ()
    kind: Literal["response_jacobian"] = field(default="response_jacobian", init=False, repr=False)

    def __post_init__(self) -> None:
        _require_names("outcomes", self.outcomes)
        _require_names("treatments", self.treatments)
        at = _coordinate_values("at", self.at, self.treatments)
        for value in at:
            _require_finite("at values", value)


@dataclass(frozen=True, slots=True)
class InterventionResponse:
    """Mean outcome under an existing static, stochastic, or modified intervention.

    Keyword-only ``horizons`` / ``policy`` attach a temporal intervention path
    (ADR 0021), licensed on ``TemporalDag``, ``TemporalCpdag``, and ``TemporalPag``.
    Class graphs retain per-completion responses and structural uncertainty. Licensed policies are
    Soft(``constant``/``additive_shift``/``multiplicative``/``truncated_shift``) and ``Sequence`` of those overlays
    (multi-step on one variable, or joint at one time when every coordinate
    is identified). Multiplicative scales the structural assignment; truncated shift targets its
    population mean using ``f + clip(E[f] + delta, lower, upper) - E[f]`` under
    preceding interventions. Outcomes are not clipped. Nested ``Sequence`` is unsupported.
    Multi-step never collapses to the last step. ``treatment_lag``
    defaults to :attr:`temporal_response_spec.default_treatment_lag`, matching
    :class:`PulseEffect`. Keyword-only ``feedback`` declares a
    :class:`FeedbackMechanism`, as on :class:`ResponseCurve`.
    """

    outcome: str
    _: KW_ONLY
    intervention: object
    target_population: object | None = None
    observation: object | None = None
    observation_assumptions: Sequence[object] = ()
    horizons: Sequence[int] | None = None
    policy: str = temporal_response_spec.default_policy
    treatment_lag: int = temporal_response_spec.default_treatment_lag
    max_history_lag: int | None = None
    feedback: FeedbackMechanism | None = None
    outcome_functional: object | None = None
    kind: Literal["intervention_response"] = field(
        default="intervention_response", init=False, repr=False
    )

    def __post_init__(self) -> None:
        _require_name("outcome", self.outcome)
        if self.intervention is None:
            raise CausalValueError("intervention must not be None")
        if self.horizons is not None:
            object.__setattr__(
                self,
                "horizons",
                tuple(
                    _count(h, f"horizons[{i}]", minimum=1, maximum=2**32 - 1)
                    for i, h in enumerate(self.horizons)
                ),
            )
        object.__setattr__(
            self,
            "treatment_lag",
            _count(self.treatment_lag, "treatment_lag", minimum=0, maximum=2**32 - 1),
        )
        if self.max_history_lag is not None:
            object.__setattr__(
                self,
                "max_history_lag",
                _count(self.max_history_lag, "max_history_lag", minimum=0, maximum=2**32 - 1),
            )
        _validate_temporal(
            self.horizons,
            self.policy,
            self.treatment_lag,
            self.max_history_lag,
            self.feedback,
        )

    @property
    def is_temporal(self) -> bool:
        return self.horizons is not None


@dataclass(frozen=True, slots=True)
class TemporalQuery:
    """Attach temporal geometry to a typed derivative or unfolded cross-world target.

    Derivatives retain the requested horizon and policy on the core response query.
    Cross-world/distribution families use their existing one-lag unfolded mechanism;
    unsupported geometry is refused rather than changing that target. This does not
    denote the separately supplied two-slice SCM in temporal_counterfactual.
    """

    query: (
        AverageDerivative
        | PointDerivative
        | Elasticity
        | SemiElasticity
        | DirectionalDerivative
        | ResponseJacobian
        | Counterfactual
        | NestedCounterfactual
        | InterventionalDistribution
        | PathSpecificEffect
    )
    _: KW_ONLY
    horizons: Sequence[int] = (1,)
    policy: str = temporal_response_spec.default_policy
    treatment_lag: int = temporal_response_spec.default_treatment_lag
    max_history_lag: int | None = None
    window: tuple[int, int] | None = None
    kind: Literal["temporal_query"] = field(default="temporal_query", init=False, repr=False)

    def __post_init__(self) -> None:
        from ._study_lifecycle import retain

        supported = (
            AverageDerivative,
            PointDerivative,
            Elasticity,
            SemiElasticity,
            DirectionalDerivative,
            ResponseJacobian,
            Counterfactual,
            NestedCounterfactual,
            InterventionalDistribution,
            PathSpecificEffect,
        )
        if not isinstance(self.query, supported):
            raise CausalValueError(
                "TemporalQuery requires a derivative or unfolded cross-world query"
            )
        horizons = tuple(self.horizons)
        _validate_horizons(horizons)
        _validate_max_history_lag(self.max_history_lag)
        if self.policy not in temporal_response_spec.allowed_policies:
            raise CausalValueError(f"unsupported temporal policy {self.policy!r}")
        lag = _count(self.treatment_lag, "treatment_lag", minimum=0, maximum=2**31 - 1)
        if self.window is not None:
            if len(self.window) != 2 or any(type(value) is not int for value in self.window):
                raise CausalValueError("window must hold two integer offsets")
            if self.window[0] > self.window[1]:
                raise CausalValueError("window start must not exceed its end")
            object.__setattr__(self, "window", tuple(self.window))
        object.__setattr__(self, "query", retain(self.query))
        object.__setattr__(self, "horizons", horizons)
        object.__setattr__(self, "treatment_lag", lag)

    @property
    def is_temporal(self) -> bool:
        return True


__all__ = [
    "TemporalQuery",
    "AnomalyAttribution",
    "AnomalyReference",
    "AverageDerivative",
    "AverageEffect",
    "ChangeAttribution",
    "ConditionalEffect",
    "Counterfactual",
    "DirectionalDerivative",
    "Elasticity",
    "Exceedance",
    "ExceedanceGrid",
    "FeedbackMechanism",
    "InterventionalDistribution",
    "InterventionResponse",
    "Mean",
    "MediationEffect",
    "NestedCounterfactual",
    "PathSpecificEffect",
    "PulseEffect",
    "Quantile",
    "PointDerivative",
    "ResponseCurve",
    "ResponseJacobian",
    "SemiElasticity",
    "SustainedEffect",
    "TemporalConditionalPulse",
    "TemporalConditionalSustained",
    "TemporalMediationEffect",
    "TemporalResponseSpec",
    "coerce_outcome_functional",
    "temporal_response_spec",
]
