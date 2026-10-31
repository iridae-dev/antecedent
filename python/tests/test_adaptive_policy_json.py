"""Policy-search configuration refuses wrongly typed or out-of-range members."""

from __future__ import annotations

import json

import numpy as np
import pytest
from antecedent import _native
from antecedent import adaptive_policy as ap

from test_adaptive_policy import TreatmentSource


def _config(**overrides):
    feature = {"variable": 0, "lag": 0, "range": [-4.0, 4.0]}
    config = {
        "families": [
            {
                "id": "threshold",
                "kind": "threshold",
                "feature": 0,
                "features": [feature],
                "action_bounds": [0.0, 1.0],
            }
        ],
        "seed": 1,
        "max_evaluations": 30,
    }
    config.update(overrides)
    return config


def _search(config):
    return _native.adaptive_policy_search(json.dumps(config), TreatmentSource())


def test_valid_configuration_runs():
    assert json.loads(_search(_config()))["families"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"held_out": True},
        {"standing": 3},
        {"objective": 1},
        {"polish": "yes"},
        {"max_evaluations": "5000"},
        {"population": 24.5},
        {"seed": -1},
        {"baseline": -1},
        {"spread_tolerance": "tight"},
    ],
)
def test_wrong_json_type_refused(overrides):
    with pytest.raises(ValueError):
        _search(_config(**overrides))


@pytest.mark.parametrize(
    "feature",
    [
        {"variable": 0, "lag": 1.0, "range": [-4.0, 4.0]},
        {"variable": 0, "lag": -1, "range": [-4.0, 4.0]},
        {"variable": 0, "lag": "1", "range": [-4.0, 4.0]},
        {"variable": -1, "lag": 0, "range": [-4.0, 4.0]},
        {"variable": 0.5, "lag": 0, "range": [-4.0, 4.0]},
    ],
)
def test_feature_members_must_be_exact(feature):
    config = _config()
    config["families"][0]["features"] = [feature]
    with pytest.raises(ValueError):
        _search(config)


def test_chance_bound_probability_type_is_checked():
    config = _config(
        constraints={"cumulative_cost": {"bound": 5.0, "max_violation_probability": "0.1"}}
    )
    with pytest.raises(ValueError, match="max_violation_probability"):
        _search(config)


def test_negative_variable_index_refused():
    feature = ap.Feature(variable=-1, range=(-4.0, 4.0))
    family = ap.Threshold("threshold", action_bounds=(0.0, 1.0), features=[feature], feature=0)
    with pytest.raises(ValueError, match="variable"):
        ap.search([family], TreatmentSource(), max_evaluations=30)


def test_rows_over_bound_refused():
    from antecedent.counterfactual import fit_gcm

    rng = np.random.default_rng(5)
    price = rng.normal(25.0, 4.0, 200)
    demand = 50.0 - 0.8 * price + rng.normal(0.0, 1.0, 200)
    fitted = fit_gcm(["price", "demand"], [price, demand], [("price", "demand")])
    feature = ap.Feature(variable="demand", lag=1, range=(0.0, 60.0))
    family = ap.Schedule("schedule", action_bounds=(0.0, 60.0), features=[feature], steps=3)
    model = ap.ModelSource(
        action="price",
        outcome="demand",
        horizon=3,
        observed=["demand"],
        rows=10**12,
        held_out_rows=10,
        initial_state={"demand": 30.0},
    )
    with pytest.raises(Exception, match="rows"):
        ap.search_model(fitted, [family], model, max_evaluations=10)
