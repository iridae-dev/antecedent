"""Optimise bounded history-dependent intervention policies.

A policy family declares which observed quantities a rule may read (a variable now, or a
variable ``lag`` steps back), the rule's shape and a closed parameter box. Search is
derivative-free, seeded and replayable; every proposal is evaluated on the same shared draws of
a response source, so policy values and their paired contrasts share draws::

    from antecedent import adaptive_policy as ap

    families = [
        ap.Schedule("schedule", steps=10, action_bounds=(0.0, 1.0)),
        ap.Threshold("threshold", feature=0, action_bounds=(0.0, 1.0),
                     features=[ap.Feature(variable=0, range=(-4.0, 4.0))]),
    ]
    result = ap.search(families, source, baseline=0, constraints=ap.Constraints(
        cumulative_exposure=ap.ChanceBound(bound=4.0, max_violation_probability=0.1)))
    result.standing.label            # "fitted_model_optimization"
    result.contrasts                 # paired differences against the schedule baseline

The ``source`` is an object that simulates decision paths and owns its draws. It provides
``identity()``, ``n_draws(split)``, ``horizon()``, ``standing() -> (identified, basis)``,
``heldout_design() -> (kind, reason)`` with ``kind`` in ``independent_draws``,
``held_out_units`` or ``unavailable``, ``observable_variables() -> list[int] | None`` and
``rollout(family_id, theta, split) -> [(utility, actions, step_costs, step_exposures), ...]``.
For one split, draw ``i`` must use the same noise for every policy (common random numbers); the
``"optimization"`` and ``"held_out"`` splits must use independent noise.

Scientific standing is conservative. The result is a ``fitted_model_optimization``: the best
policy found inside the fitted response model, conditional on its assumptions. It is labelled
``identified_policy_effect`` only when requested with ``standing="identified_policy_effect"`` and
the source states an identification basis; otherwise the request is refused
(``adaptive_policy_identification_unsupported``). The selected policy is chosen on the
optimisation draws, so its ``in_sample`` value is optimistic; the ``held_out`` value is on draws
or units the selection never touched. A study design with no held-out evaluation is refused when
``held_out="required"`` (``adaptive_policy_heldout_unsupported``) and flagged otherwise. A policy
that reads a variable outside the source's observable set is refused
(``adaptive_policy_unobserved_feature``).

SPDX-License-Identifier: MIT OR Apache-2.0
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from ._native import CancellationToken
from ._native import adaptive_policy_search as _search

__all__ = [
    "AdaptivePolicyResult",
    "ChanceBound",
    "Constraints",
    "Feature",
    "FamilyResult",
    "LinearFeedback",
    "ModelSource",
    "PairedContrast",
    "Piecewise",
    "Schedule",
    "Standing",
    "Threshold",
    "Tree",
    "search",
    "search_model",
]


@dataclass(frozen=True)
class Feature:
    """One observed quantity a rule may read: ``variable`` now or ``lag`` steps back."""

    variable: int | str
    range: tuple[float, float]
    lag: int = 0

    def _spec(self) -> dict[str, Any]:
        return {"variable": self.variable, "lag": self.lag, "range": list(self.range)}


@dataclass(frozen=True)
class ChanceBound:
    """Upper bound on a path quantity with a tolerated violation probability (0 is hard)."""

    bound: float
    max_violation_probability: float = 0.0


@dataclass(frozen=True)
class Constraints:
    """Hard and chance constraints on rollout paths."""

    cumulative_cost: ChanceBound | None = None
    cumulative_exposure: ChanceBound | None = None
    per_step_action: tuple[float, float] | None = None

    def _spec(self) -> dict[str, Any]:
        def bound(value: ChanceBound | None) -> dict[str, float] | None:
            if value is None:
                return None
            return {
                "bound": value.bound,
                "max_violation_probability": value.max_violation_probability,
            }

        return {
            "cumulative_cost": bound(self.cumulative_cost),
            "cumulative_exposure": bound(self.cumulative_exposure),
            "per_step_action": None if self.per_step_action is None else list(self.per_step_action),
        }


@dataclass(frozen=True)
class _Family:
    id: str
    action_bounds: tuple[float, float]
    features: Sequence[Feature] = ()
    coefficient_range: tuple[float, float] = (-1.0, 1.0)

    def _kind(self) -> dict[str, Any]:  # pragma: no cover - overridden
        raise NotImplementedError

    def _spec(self) -> dict[str, Any]:
        spec: dict[str, Any] = {
            "id": self.id,
            "action_bounds": list(self.action_bounds),
            "coefficient_range": list(self.coefficient_range),
            "features": [f._spec() for f in self.features],
        }
        spec.update(self._kind())
        return spec


@dataclass(frozen=True)
class Schedule(_Family):
    """Open-loop schedule: one action per step (the fixed-schedule baseline)."""

    steps: int = 1

    def _kind(self) -> dict[str, Any]:
        return {"kind": "schedule", "steps": self.steps}


@dataclass(frozen=True)
class Threshold(_Family):
    """``below`` or ``at_or_above`` a threshold on one feature."""

    feature: int = 0

    def _kind(self) -> dict[str, Any]:
        return {"kind": "threshold", "feature": self.feature}


@dataclass(frozen=True)
class Piecewise(_Family):
    """Piecewise-constant on one feature with ``cuts`` ordered cut points."""

    feature: int = 0
    cuts: int = 1

    def _kind(self) -> dict[str, Any]:
        return {"kind": "piecewise", "feature": self.feature, "cuts": self.cuts}


@dataclass(frozen=True)
class LinearFeedback(_Family):
    """``u = clip(a + b . features)`` over the listed feature indices."""

    features_used: Sequence[int] = (0,)

    def _kind(self) -> dict[str, Any]:
        return {"kind": "linear_feedback", "features_used": list(self.features_used)}


@dataclass(frozen=True)
class Tree(_Family):
    """Complete binary decision tree; ``splits`` gives the feature tested at each internal node."""

    splits: Sequence[int] = (0,)

    def _kind(self) -> dict[str, Any]:
        return {"kind": "tree", "splits": list(self.splits)}


@dataclass(frozen=True)
class Standing:
    """Scientific label of an optimisation result."""

    label: str
    detail: str

    @property
    def identified(self) -> bool:
        """Whether the result is labelled an identified policy effect."""
        return self.label == "identified_policy_effect"


@dataclass(frozen=True)
class PairedContrast:
    """Mean paired difference of a policy minus the baseline over shared draws."""

    policy: str
    baseline: str
    split: str
    n_draws: int
    mean_difference: float
    standard_error: float


@dataclass(frozen=True)
class FamilyResult:
    """Best policy found in one family, with in-sample and held-out evaluation."""

    id: str
    theta: tuple[float, ...]
    in_sample: Mapping[str, Any]
    held_out: Mapping[str, Any] | None
    evaluations: int
    generations: int
    stop: str
    failed_attempts: int
    attempts: Sequence[Mapping[str, Any]] = field(repr=False, default=())


@dataclass(frozen=True)
class AdaptivePolicyResult:
    """Result of a bounded history-dependent policy search."""

    standing: Standing
    complete: bool
    flags: tuple[str, ...]
    selected: int | None
    selection_optimism: float | None
    families: tuple[FamilyResult, ...]
    contrasts: tuple[PairedContrast, ...]
    source_identity: str
    problem_identity: str
    record_identity: str
    admitted_bytes: int

    @property
    def selected_family(self) -> FamilyResult | None:
        """The best feasible family on the optimisation draws, if any is feasible."""
        return None if self.selected is None else self.families[self.selected]


def search(
    families: Sequence[_Family],
    source: Any,
    *,
    baseline: int | None = None,
    constraints: Constraints | None = None,
    objective: Literal["expected_utility", "lower_tail_mean"] = "expected_utility",
    tail_alpha: float | None = None,
    seed: int = 0,
    max_evaluations: int = 2000,
    population: int = 24,
    elite_fraction: float = 0.25,
    max_generations: int = 40,
    spread_tolerance: float = 1e-3,
    polish: bool = True,
    standing: Literal["fitted_model", "identified_policy_effect"] = "fitted_model",
    held_out: Literal["preferred", "required"] = "preferred",
    cancel: CancellationToken | None = None,
    _model_source: Mapping[str, Any] | None = None,
    _fitted: Any = None,
) -> AdaptivePolicyResult:
    """Search the declared policy families against ``source``.

    Raises a refusal for an unobserved feature, an unsupported identified-effect request or a
    required but unavailable held-out evaluation; raises ``CausalCancelledError`` on
    cancellation and ``CausalResourceError`` on memory refusal.
    """
    config: dict[str, Any] = {
        "families": [family._spec() for family in families],
        "baseline": baseline,
        "constraints": (constraints or Constraints())._spec(),
        "objective": objective,
        "tail_alpha": tail_alpha,
        "seed": seed,
        "max_evaluations": max_evaluations,
        "population": population,
        "elite_fraction": elite_fraction,
        "max_generations": max_generations,
        "spread_tolerance": spread_tolerance,
        "polish": polish,
        "standing": standing,
        "held_out": held_out,
    }
    if _fitted is not None:
        config["model_source"] = dict(_model_source or {})
        native = getattr(_fitted, "_native", _fitted)
        raw = json.loads(native.optimize_adaptive_policy(json.dumps(config), cancel))
    else:
        raw = json.loads(_search(json.dumps(config), source, cancel))
    label = raw["standing"]
    return AdaptivePolicyResult(
        standing=Standing(label["label"], label.get("note") or label.get("basis") or ""),
        complete=raw["complete"],
        flags=tuple(raw["flags"]),
        selected=raw["selected"],
        selection_optimism=raw["selection_optimism"],
        families=tuple(
            FamilyResult(
                id=f["id"],
                theta=tuple(f["theta"]),
                in_sample=f["in_sample"],
                held_out=f["held_out"],
                evaluations=f["evaluations"],
                generations=f["generations"],
                stop=f["stop"],
                failed_attempts=f["failed_attempts"],
                attempts=tuple(f["attempts"]),
            )
            for f in raw["families"]
        ),
        contrasts=tuple(PairedContrast(**c) for c in raw["contrasts"]),
        source_identity=raw["source_identity"],
        problem_identity=raw["problem_identity"],
        record_identity=raw["record_identity"],
        admitted_bytes=raw["admitted_bytes"],
    )


@dataclass(frozen=True)
class ModelSource:
    """How a fitted causal model is rolled out under a candidate policy.

    ``action`` is the variable the policy assigns each decision step; ``outcome`` is scored by the
    stage utility ``-(over * max(y - target, 0) + under * max(target - y, 0) + quadratic * (y -
    target)**2 + action_linear * |a| + action_quadratic * a**2)``, discounted over steps.
    ``observed`` lists the variables a rule may read when it acts. The policy decides every
    ``decision_interval`` steps; in between the action holds its last value. ``initial_state``
    gives the realised values before the first step (needed when a rule reads a lagged value).
    ``rows`` and ``held_out_rows`` are independent simulated units on the optimisation and
    held-out splits; every policy meets the same noise. Parameter uncertainty of the fit is not
    propagated, so the result is a fitted-model optimisation unless ``identified`` is declared
    with an ``identification_basis``.
    """

    action: str
    outcome: str
    horizon: int
    observed: Sequence[str]
    rows: int = 1000
    held_out_rows: int = 1000
    noise_seed: int = 0
    initial_state: Mapping[str, float] = field(default_factory=dict)
    target: float = 0.0
    over: float = 0.0
    under: float = 0.0
    quadratic: float = 0.0
    action_linear: float = 0.0
    action_quadratic: float = 0.0
    discount: float = 1.0
    unit_cost: float = 0.0
    exposure_weight: float = 0.0
    decision_interval: int = 1
    identified: bool = False
    identification_basis: str | None = None

    def _spec(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "outcome": self.outcome,
            "horizon": self.horizon,
            "observed": list(self.observed),
            "rows": self.rows,
            "held_out_rows": self.held_out_rows,
            "noise_seed": self.noise_seed,
            "initial_state": dict(self.initial_state),
            "utility": {
                "target": self.target,
                "over": self.over,
                "under": self.under,
                "quadratic": self.quadratic,
                "action_linear": self.action_linear,
                "action_quadratic": self.action_quadratic,
                "discount": self.discount,
            },
            "unit_cost": self.unit_cost,
            "exposure_weight": self.exposure_weight,
            "decision_interval": self.decision_interval,
            "identified": self.identified,
            "identification_basis": self.identification_basis,
        }


def search_model(
    fitted: Any,
    families: Sequence[_Family],
    model: ModelSource,
    **options: Any,
) -> AdaptivePolicyResult:
    """Search policy families against a fitted causal model (``antecedent.gcm.fit_gcm`` result).

    Features and the model source name variables by name. Takes the same options as
    :func:`search`.
    """
    return search(families, None, _model_source=model._spec(), _fitted=fitted, **options)
