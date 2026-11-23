"""History-dependent policy search through the native engine against an independent simulator.

The simulator is plain numpy with its own generator. Its closed-form reference (a stationary
Gaussian state, stage utility ``u (z - k)``, optimal rule "treat when z >= k") is recomputed in
``conformance/design/adaptive_policy_search/reference.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from antecedent import adaptive_policy as ap
from antecedent.errors import CausalUnsupportedError

HORIZON = 10
RHO, SIGMA, K = 0.6, 0.8, 0.5


def _closed_form_value() -> float:
    s = SIGMA / math.sqrt(1.0 - RHO * RHO)
    pdf = math.exp(-0.5 * (K / s) ** 2) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + math.erf(K / s / math.sqrt(2.0)))
    return HORIZON * (s * pdf - K * (1.0 - cdf))


class TreatmentSource:
    """Stationary AR(1) state observed before a dose in [0, 1]."""

    def __init__(self, identified=False, held_out="independent_draws", observable=(0,)):
        self.identified = identified
        self.held_out = held_out
        self.observable = observable
        self.n = {"optimization": 2500, "held_out": 3000 if held_out != "unavailable" else 0}
        self.noise = {}
        for split, seed in (("optimization", 101), ("held_out", 202)):
            rng = np.random.default_rng(seed)
            n = self.n[split]
            s0 = SIGMA / math.sqrt(1.0 - RHO * RHO)
            self.noise[split] = (rng.standard_normal(n) * s0, rng.standard_normal((HORIZON, n)))

    def identity(self):
        return "numpy-treatment-simulator"

    def n_draws(self, split):
        return self.n[split]

    def horizon(self):
        return HORIZON

    def standing(self):
        return (
            self.identified,
            "randomised sequential assignment" if self.identified else "fitted model",
        )

    def heldout_design(self):
        return (self.held_out, "no separated units" if self.held_out == "unavailable" else "")

    def observable_variables(self):
        return list(self.observable) if self.observable is not None else None

    def rollout(self, family_id, theta, split):
        z0, eps = self.noise[split]
        z = z0.copy()
        utility = np.zeros_like(z)
        actions = np.zeros((HORIZON, z.size))
        for t in range(HORIZON):
            if family_id == "schedule":
                u = np.full_like(z, theta[min(t, len(theta) - 1)])
            else:
                u = np.where(z >= theta[0], theta[2], theta[1])
            actions[t] = u
            utility += u * (z - K)
            z = RHO * z + SIGMA * eps[t]
        return [
            (float(utility[i]), list(actions[:, i]), list(actions[:, i]), list(actions[:, i]))
            for i in range(z.size)
        ]


def families():
    feature = ap.Feature(variable=0, range=(-4.0, 4.0))
    return [
        ap.Schedule("schedule", action_bounds=(0.0, 1.0), features=[feature], steps=HORIZON),
        ap.Threshold("threshold", action_bounds=(0.0, 1.0), features=[feature], feature=0),
    ]


def test_state_dependent_threshold_beats_the_best_fixed_schedule_at_the_closed_form():
    result = ap.search(families(), TreatmentSource(), baseline=0, seed=3, max_evaluations=500)
    optimum = _closed_form_value()
    held = result.families[1].held_out
    assert held is not None
    assert abs(held["mean_utility"] - optimum) < 4 * held["standard_error"] + 0.1
    schedule = result.families[0].held_out
    assert abs(schedule["mean_utility"]) < 4 * schedule["standard_error"] + 0.05
    assert result.selected == 1
    contrast = next(c for c in result.contrasts if c.split == "held_out")
    assert contrast.mean_difference > 0.8 * optimum
    assert contrast.mean_difference > 15 * contrast.standard_error
    assert result.standing.label == "fitted_model_optimization"
    assert not result.standing.identified
    assert result.selection_optimism is not None


def test_a_fixed_seed_replays_the_record_exactly():
    kwargs = {"baseline": 0, "seed": 9, "max_evaluations": 120}
    first = ap.search(families(), TreatmentSource(), **kwargs)
    again = ap.search(families(), TreatmentSource(), **kwargs)
    other = ap.search(families(), TreatmentSource(), **{**kwargs, "seed": 10})
    assert first.record_identity == again.record_identity
    assert first.record_identity != other.record_identity


def test_exposure_constraint_is_reported_on_paths():
    constraints = ap.Constraints(
        cumulative_exposure=ap.ChanceBound(bound=3.0, max_violation_probability=0.0)
    )
    result = ap.search(
        families()[1:], TreatmentSource(), constraints=constraints, seed=2, max_evaluations=400
    )
    report = result.families[0].in_sample["constraints"][0]
    assert report["id"] == "cumulative_exposure"
    assert report["violations"] == 0 and report["worst_observed"] <= 3.0 + 1e-12
    assert result.families[0].in_sample["feasible"]


def test_identified_policy_effect_is_refused_without_an_identification_basis():
    with pytest.raises(CausalUnsupportedError) as raised:
        ap.search(
            families()[1:],
            TreatmentSource(),
            standing="identified_policy_effect",
            max_evaluations=50,
        )
    assert raised.value.reason_code == "adaptive_policy_identification_unsupported"
    labelled = ap.search(
        families()[1:],
        TreatmentSource(identified=True),
        standing="identified_policy_effect",
        max_evaluations=60,
    )
    assert labelled.standing.identified


def test_held_out_is_required_or_flagged_and_unobserved_features_are_refused():
    source = TreatmentSource(held_out="unavailable")
    with pytest.raises(CausalUnsupportedError) as raised:
        ap.search(families()[1:], source, held_out="required", max_evaluations=50)
    assert raised.value.reason_code == "adaptive_policy_heldout_unsupported"
    flagged = ap.search(families()[1:], source, max_evaluations=50)
    assert flagged.families[0].held_out is None
    assert any("held-out evaluation unavailable" in flag for flag in flagged.flags)
    with pytest.raises(CausalUnsupportedError) as hidden:
        ap.search(families()[1:], TreatmentSource(observable=(7,)), max_evaluations=50)
    assert hidden.value.reason_code == "adaptive_policy_unobserved_feature"


def test_budget_exhaustion_is_reported_as_incomplete():
    result = ap.search(families()[1:], TreatmentSource(), max_evaluations=10)
    assert not result.complete
    assert result.families[0].stop == "evaluation_budget"


def test_a_fitted_model_is_a_response_source_with_hold_between_decisions():
    from antecedent.counterfactual import fit_gcm

    rng = np.random.default_rng(5)
    price = rng.normal(25.0, 4.0, 3000)
    demand = 50.0 - 0.8 * price + rng.normal(0.0, 1.0, 3000)
    fitted = fit_gcm(["price", "demand"], [price, demand], [("price", "demand")])
    feature = ap.Feature(variable="demand", lag=1, range=(0.0, 60.0))
    families = [
        ap.Schedule("schedule", action_bounds=(0.0, 60.0), features=[feature], steps=3),
        ap.LinearFeedback(
            "feedback",
            action_bounds=(0.0, 60.0),
            features=[feature],
            coefficient_range=(-2.0, 2.0),
            features_used=[0],
        ),
    ]
    model = ap.ModelSource(
        action="price",
        outcome="demand",
        horizon=6,
        observed=["demand"],
        rows=800,
        held_out_rows=800,
        noise_seed=4,
        initial_state={"demand": 30.0},
        target=30.0,
        quadratic=1.0,
        decision_interval=2,
    )
    result = ap.search_model(fitted, families, model, baseline=0, seed=1, max_evaluations=250)
    assert result.standing.label == "fitted_model_optimization"
    # Tracking loss is bounded below by the demand noise variance (about 1 per step).
    # Six steps at unit noise variance bound the tracking loss at -6; the declared stages reach
    # it (observed -5.95 for the schedule, -6.09 for the feedback rule, standard error 0.12).
    for family in result.families:
        held = family.held_out
        assert held is not None
        assert abs(held["mean_utility"] + 6.0) < 3 * held["standard_error"], held
