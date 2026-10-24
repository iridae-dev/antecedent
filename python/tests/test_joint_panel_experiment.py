"""Real native-target experiment chain through the public Python workflow."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
from antecedent import decision
from antecedent import joint_panel as jp
from antecedent import scientific_bridge as sb
from antecedent.evidence_ledger import EvidenceLedger, SourceDeclaration, SourceRevision
from antecedent.joint_distribution import ScientificQuantity


@pytest.mark.parametrize("local", [False, True])
def test_native_target_experiment_update_decision_next_study_and_current_replay(local):
    units = [
        {
            "id": f"unit:{i}",
            "arm": i % 2,
            "probabilities": [0.5, 0.5],
            "outcome": (1.0 if i % 2 == 0 else 2.0) + (0.1 if i % 4 < 2 else -0.1),
            "population_weight": 1.0,
        }
        for i in range(32)
    ]
    prepared = jp.prepare_native_joint_design(
        {
            "kind": "randomized",
            "design": {
                "worlds": ["reference", "active"],
                "sampling_unit": "independent_cluster",
                "population": "declared_cluster_population",
                "consistency": "keys denote exact complete assigned programs",
                "between_unit_isolation": "independent randomized clusters without spillover",
            },
            "units": units,
        }
    )
    result = prepared.execute(
        {
            "inference": "randomized_gaussian",
            "prior": {
                "mean": [0.0, 0.0],
                "variance": [4.0, 4.0],
                "outcome_variance": [1.0] * len(units),
                "draws": 512,
                "seed": 91,
            },
        }
    )
    regularization = None
    coefficients = [-1.0, 1.0]
    target_kind = "whole_program_population_mean"
    if local:
        prepared = jp.prepare_native_joint_design(
            {
                "kind": "instrumented_local",
                "design": {
                    "key": "encouragement-compliers",
                    "population": "source-population",
                    "exclusion": "no direct encouragement effect",
                    "monotonicity": "no defiers",
                    "consistency_and_isolation": "consistent complete program and independent units",
                },
                "units": [
                    {
                        "id": f"local{i}",
                        "assigned": i % 2 == 0,
                        "received": i % 4 == 0,
                        "probability": 0.5,
                        "outcome": (2.0 if i % 8 == 0 else 4.0) if i % 4 == 0 else 0.0,
                    }
                    for i in range(128)
                ],
            }
        )
        result = prepared.execute(
            {"inference": "local_complier_bayesian", "draws": 512, "seed": 92}
        )
        coefficients = [1.0]
        target_kind = "encouragement_specific_complier_average_effect"
        regularization = {
            "max_abs_intercept": 0.0,
            "min_abs_slope": 1.0,
            "max_abs_slope": 1.0,
            "assumption": "The complete mapping law is exactly intercept zero and slope one.",
        }
    prior = result.report()
    draws = (
        [[r[0]] for r in prior["law"]["LocalComplier"]["posterior"]]
        if local
        else prior["law"]["Moments"]["Bayesian"]["draws"]
    )
    values = [math.fsum(v * c for v, c in zip(row, coefficients, strict=True)) for row in draws]
    target = result.experiment_quantity(
        coefficients=coefficients,
        outcome_units="outcome_unit",
        local_ratio_regularization=regularization,
    )
    assert target.functional_id.startswith(f"{target_kind}:linear:")
    source = replace(target, functional_id="experimental_sample_mean")
    setting = sb.ExposureSetting(
        target.population_id,
        "native_summary:0",
        "native_summary:0",
        1.0,
        "native_projection",
        1.0,
        "native_summary_window",
        target.regime_id,
    )
    bridge = sb.BridgeContract(
        "parameter_intervention_response",
        source,
        "model_parameter",
        target,
        "intervention_response",
        sb.Scope(target.population_id, target.regime_id),
        sb.AffineMapping(0.0, 1.0),
        assumptions=(
            sb.Assumption(
                "likelihood_model",
                "parameter_interpretation",
                "independent Gaussian sample mean maps to this exact native contrast",
                "declared_experiment",
            ),
        ),
        mapping_uncertainty=sb.MappingUncertainty("fixed_mapping", 0.0, 0.0, 0.0),
        source_dependence=sb.SourceDependence("slope_covariance", "retained_slots", 0.0),
        estimand_bridge=sb.EstimandBridge(
            "exact_translation",
            "declared_experiment",
            setting,
            setting,
            0.0,
            "native_window",
            0.0,
            "native_projection",
        ),
    )
    ledger = EvidenceLedger()
    dependency = prior["dependency"]
    model = ledger.add_source(SourceDeclaration(dependency, "fitted_model", [("model", "rows")]))
    experiment = ledger.add_source(
        SourceDeclaration("new_experiment", "experiment", [("new", "rows")])
    )
    mapping = ledger.add_source(SourceDeclaration("mapping", "fitted_model", [("mapping", "rows")]))
    observed = math.fsum(values) / len(values) + 0.3
    updated = result.condition_experiment(
        coefficients=coefficients,
        local_ratio_regularization=regularization,
        outcome_units="outcome_unit",
        bridge=bridge,
        ledger=ledger,
        observed_mean=observed,
        observation_variance=0.2,
        sample_size=4,
        mapping_dependency=dependency,
        intercepts=[0.0] * 512,
        slopes=[1.0] * 512,
        model=model,
        experiment=experiment,
        mapping=mapping,
    )
    report = updated.report(ledger)
    assert report["identification"]["target_kind"] == target_kind
    assert report["local_ratio_regularization"] == regularization
    anchor = min(values, key=lambda v: abs(observed - v))
    raw = [math.exp((v - anchor) * (2 * observed - v - anchor) / (2 * 0.2 / 4)) for v in values]
    norm = math.fsum(raw)
    assert report["posterior_weights"] == pytest.approx([v / norm for v in raw], abs=1e-12)
    mean = math.fsum(v * w for v, w in zip(values, report["posterior_weights"], strict=True))
    ranking = updated.rank_next_samples(
        actions=[
            {"id": "hold", "intercept": 0.0, "coefficients": [0.0] * len(coefficients)},
            {"id": "act", "intercept": -mean, "coefficients": coefficients},
        ],
        candidates=[
            {"id": "cheap", "sample_size": 4, "observation_variance": 1.0, "cost_utility": 0.01},
            {"id": "costly", "sample_size": 4, "observation_variance": 1.0, "cost_utility": 0.1},
        ],
        ledger=ledger,
        replicates=32,
    )["candidates"]
    assert ranking[0]["id"] == "cheap"
    assert ranking[0]["gross_evsi"] == ranking[1]["gross_evsi"]
    coordinates = tuple(ScientificQuantity.from_wire(q) for q in report["quantities"])
    zero = decision.x(0) - decision.x(0)
    expression = decision.const(-mean)
    for i, c in enumerate(coefficients):
        expression = expression + decision.x(i) * c
    terminal = decision.Contract(
        actions=(
            decision.Action("hold", inputs=coordinates, utility=zero, kind="policy"),
            decision.Action("act", inputs=coordinates, utility=expression, kind="policy"),
        ),
        utility_units=target.units,
        criterion=decision.Criterion.expected_utility(),
        target_population=target.population_id,
        constraints=(
            decision.Constraint(
                "binding_capacity", zero + 1.0, bound=0.0, units="capacity", applies_to=("act",)
            ),
        ),
    )
    terminal_result = updated.decide(terminal, ledger)
    feasible = [a["id"] for a in terminal_result["actions"] if a["admissible"]]
    assert feasible == ["hold"]
    candidates = [
        {"id": "cheap", "sample_size": 4, "observation_variance": 1.0, "cost_utility": 0.01}
    ]
    hold = {"id": "hold", "intercept": 0.0, "coefficients": [0.0] * len(coefficients)}
    restricted = updated.rank_next_samples(
        actions=[hold], candidates=candidates, ledger=ledger, replicates=8
    )["candidates"][0]
    assert restricted["gross_evsi"] == pytest.approx(0.0, abs=1e-12)
    assert restricted["net_value"] == pytest.approx(-0.01, abs=1e-12)
    assert restricted["admissible_action_identity"] != ranking[0]["admissible_action_identity"]
    renamed = updated.rank_next_samples(
        actions=[{**hold, "id": "different_action"}],
        candidates=candidates,
        ledger=ledger,
        replicates=8,
    )["candidates"][0]
    changed = updated.rank_next_samples(
        actions=[{**hold, "coefficients": coefficients}],
        candidates=candidates,
        ledger=ledger,
        replicates=8,
    )["candidates"][0]
    assert (
        len(
            {
                restricted["admissible_action_identity"],
                renamed["admissible_action_identity"],
                changed["admissible_action_identity"],
            }
        )
        == 3
    )
    artifact = updated.export(ledger)
    updated.verify_current(artifact, ledger)
    prepared.consume_current(result.export())
    ledger.revise_source(SourceRevision.withdraw(experiment, "withdrawn experiment"))
    with pytest.raises(ValueError, match="withdrawn|missing|superseded"):
        updated.verify_current(artifact, ledger)


def test_generated_acceptance_matrix_keeps_all_advertised_design_rows():
    matrix = jp.experiment_acceptance_matrix()
    assert len(matrix["rows"]) == 510
    assert {r["design"] for r in matrix["rows"]} == {
        "factorial_multi_arm",
        "switchback",
        "geo_experiment",
        "sequential_observational",
        "instrumented_dose",
        "difference_in_differences",
        "synthetic_control",
        "retained_panel_model_conditional",
    }
    assert len({r["id"] for r in matrix["rows"]}) == 510
