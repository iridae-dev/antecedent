"""Actual compiled parameter experiments and population transport, with independent truth."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from antecedent import joint_panel
from antecedent.errors import CausalResourceError, CausalValueError
from antecedent.outcomes import require_success

ROOT = Path(__file__).resolve().parents[2]


def fixture():
    return json.loads(
        (ROOT / "conformance/response/joint_panel/population_reference.json").read_text(
            encoding="utf-8"
        )
    )


def truth():
    values = []
    for mask in range(16):
        exposure = 4.0 if mask & 1 else 1.0
        saturation = 2.0 if mask & 2 else 1.0
        population = 1.8 if mask & 4 else 1.2
        mechanism = 3.0 if mask & 8 else 1.0
        values.append(population * mechanism * exposure / (exposure + saturation))
    contributions = []
    for component in range(4):
        result = 0.0
        for mask in range(16):
            if mask & (1 << component):
                continue
            count = mask.bit_count()
            weight = 0.25 if count in (0, 3) else 1.0 / 12.0
            result += weight * (values[mask | (1 << component)] - values[mask])
        contributions.append(result)
    return values, contributions


@pytest.mark.parametrize("bayesian", [False, True])
def test_parameter_and_population_reference_decomposition_with_actual_source_and_replay(bayesian):
    data = fixture()
    prepared = joint_panel.prepare_joint_design(data["target"], data["proof"])
    result = require_success(prepared.execute(data["bayesian"] if bayesian else "frequentist"))
    declaration = data["population_reference"]
    report = result.decompose_population_reference(**declaration)
    expected_worlds, expected_contributions = truth()
    tolerance = 0.04 if bayesian else 1e-12
    point = report["decomposition"]["point"]
    assert point["hybrids"] == pytest.approx(expected_worlds, abs=tolerance)
    assert point["shapley"] == pytest.approx(expected_contributions, abs=tolerance)
    assert sum(point["shapley"]) == pytest.approx(point["total"], abs=1e-12)
    assert len(point["by_order"]) == len(point["orderings"]) == 24
    assert point["order_range"][0][1] > point["order_range"][0][0]
    assert report["scope"] == "identified_population_reference"
    assert report["transport"]["selection_targets"] == [3]
    assert report["grants_unique_causal_attribution"] is False
    covariance = report["covariance"]
    assert sum(sum(row) for row in covariance["shapley"]) == pytest.approx(
        covariance["total"], abs=1e-10
    )
    for draw in report["decomposition"]["draws"]:
        assert sum(draw["shapley"]) == pytest.approx(draw["total"], abs=1e-10)
    components = {
        "exposure": [0, 3],
        "saturation": [1],
        "mechanism": [2],
        "hybrids": declaration["stratum_worlds"][0],
        "order": declaration["order"],
        "require_identified": True,
    }
    component_report = result.decompose_reference_components(**components)
    component_artifact = bytes(
        result.export(quantity="reference_decomposition", declaration=components)
    )
    component_history = joint_panel.replay_reference_decomposition(component_artifact)
    component_current = result.consume_reference_decomposition_current(
        component_artifact, components
    )
    assert component_history["current_authorization"] is False
    assert (
        component_report["decomposition"]
        == component_history["decomposition"]
        == component_current["decomposition"]
    )
    artifact = bytes(result.export(quantity="population_reference", declaration=declaration))
    historical = joint_panel.replay_population_reference(artifact)
    current = result.consume_population_reference_current(artifact, declaration)
    assert historical["current_authorization"] is False
    assert historical["decomposition"] == current["decomposition"] == report["decomposition"]
    changed = copy.deepcopy(declaration)
    changed["after_weights"] = [0.4, 0.6]
    with pytest.raises(CausalValueError, match="caller-current"):
        result.consume_population_reference_current(artifact, changed)
    other_execution = require_success(
        prepared.execute("frequentist" if bayesian else data["bayesian"])
    )
    with pytest.raises(CausalValueError, match="statistical execution"):
        other_execution.consume_reference_decomposition_current(component_artifact, components)
    with pytest.raises(CausalValueError, match="statistical execution"):
        other_execution.consume_population_reference_current(artifact, declaration)


def test_population_only_change_and_missing_transport_premises():
    data = fixture()
    result = require_success(
        joint_panel.prepare_joint_design(data["target"], data["proof"]).execute("frequentist")
    )
    isolated = copy.deepcopy(data["population_reference"])
    isolated["stratum_worlds"] = [[0] * 16, [8] * 16]
    report = result.decompose_population_reference(**isolated)
    point = report["decomposition"]["point"]
    assert point["shapley"] == pytest.approx([0.0, 0.0, point["total"], 0.0], abs=1e-12)
    invalid = copy.deepcopy(data["population_reference"])
    invalid["diagram"]["selection_targets"] = [4]
    with pytest.raises(CausalValueError, match="selection"):
        result.decompose_population_reference(**invalid)
    invalid = copy.deepcopy(data["population_reference"])
    invalid["diagram"]["bidirected"] = [[3, 4]]
    with pytest.raises(CausalValueError, match="latent"):
        result.decompose_population_reference(**invalid)
    invalid = copy.deepcopy(data["population_reference"])
    invalid["saturation"].append(0)
    with pytest.raises(CausalValueError, match="disjoint"):
        result.decompose_population_reference(**invalid)
    with pytest.raises(CausalResourceError):
        result.decompose_population_reference(**data["population_reference"], max_bytes=1)
