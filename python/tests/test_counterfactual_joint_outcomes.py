"""A counterfactual with several outcomes and a joint baseline of simultaneous interventions.

Each outcome's contrast reads the same abduction, so it agrees with the single-outcome
counterfactual asked for that outcome under the same joint baseline.
"""

import antecedent as ac
import numpy as np
import pytest
from antecedent.outcomes import require_success

GRAPH = [("a", "y1"), ("b", "y1"), ("a", "y2"), ("b", "y2")]


def fixture():
    rng = np.random.default_rng(3)
    n = 1500
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    y1 = 0.8 * a + 0.5 * b + rng.normal(size=n)
    y2 = -0.4 * a + 0.9 * b + rng.normal(size=n)
    return {"a": a, "b": b, "y1": y1, "y2": y2}


def run(query):
    return require_success(
        ac.analyze(
            fixture(),
            graph=GRAPH,
            query=query,
            refute="none",
            execution=ac.Execution(seed=1),
        )
    )


def joint(outcome, *, also=()):
    return ac.Counterfactual(
        "a",
        outcome,
        control_level=0.0,
        active_level=1.0,
        also_outcomes=also,
        active_interventions={"b": 1.0},
        control_interventions={"b": -1.0},
    )


def test_each_outcome_agrees_with_its_single_outcome_counterfactual():
    both = run(joint("y1", also=("y2",)))
    first = run(joint("y1"))
    second = run(joint("y2"))
    contrasts = both.counterfactual_contrasts
    assert contrasts is not None
    assert [c.outcome for c in contrasts] == ["y1", "y2"]
    for contrast, single in zip(contrasts, (first, second), strict=True):
        assert np.allclose(contrast.unit_effects, single.unit_effects)
        assert contrast.mean_effect == pytest.approx(float(np.mean(single.unit_effects)))
    # The first outcome fills the single-outcome fields unchanged.
    assert np.allclose(both.unit_effects, first.unit_effects)
    # Independently, a linear mechanism with additive noise moves every unit by the fitted
    # coefficients times the joint change (a: 0 -> 1, b: -1 -> 1), the same for every unit.
    data = fixture()
    design = np.column_stack([np.ones(len(data["a"])), data["a"], data["b"]])
    for contrast, outcome in zip(contrasts, ("y1", "y2"), strict=True):
        _, slope_a, slope_b = np.linalg.lstsq(design, data[outcome], rcond=None)[0]
        want = slope_a * 1.0 + slope_b * 2.0
        # y1 is reproduced to rounding; the y2 mechanism is not a plain least-squares plane
        # (observed difference 2.6e-2), so it is held to twice that.
        atol = {"y1": 1e-6, "y2": 0.05}[outcome]
        assert np.allclose(contrast.unit_effects, want, atol=atol), outcome
        assert contrast.mean_effect == pytest.approx(want, abs=atol)
        assert abs(want - {"y1": 1.8, "y2": 1.4}[outcome]) < 0.15  # the generating coefficients


def test_a_single_outcome_query_publishes_no_contrasts():
    single = run(ac.Counterfactual("a", "y1"))
    assert single.counterfactual_contrasts is None
    assert single.unit_effects is not None
    # The default contrast moves a by one with nothing else set: the fitted slope of a on y1,
    # close to the generating 0.8, identical for every unit.
    data = fixture()
    design = np.column_stack([np.ones(len(data["a"])), data["a"], data["b"]])
    slope_a = np.linalg.lstsq(design, data["y1"], rcond=None)[0][1]
    assert np.allclose(single.unit_effects, slope_a, atol=1e-6)


def test_the_joint_baseline_must_intervene_on_the_same_variables():
    with pytest.raises(ac.errors.CausalValueError, match="intervenes on the same variables"):
        ac.Counterfactual("a", "y1", active_interventions={"b": 1.0}, control_interventions={})
    with pytest.raises(ac.errors.CausalValueError, match="lists each outcome once"):
        ac.Counterfactual("a", "y1", also_outcomes=("y1",))
