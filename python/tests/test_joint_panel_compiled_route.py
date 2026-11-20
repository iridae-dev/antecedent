"""Every joint-panel handle reports the compiled program it executes."""

from __future__ import annotations

import json
import math
import re
from dataclasses import replace
from pathlib import Path

from antecedent import joint_panel as jp
from antecedent import scientific_bridge as sb
from antecedent.evidence_ledger import EvidenceLedger, SourceDeclaration
from antecedent.outcomes import require_success

from test_joint_panel import declarations, request
from test_joint_panel_imputation import panel as imputed_panel

ROOT = Path(__file__).resolve().parents[2]
HEX64 = re.compile(r"[0-9a-f]{64}")


def native_recipe():
    return {
        "kind": "randomized",
        "design": {
            "worlds": ["reference", "active"],
            "sampling_unit": "independent_cluster",
            "population": "declared_cluster_population",
            "consistency": "keys denote exact complete assigned programs",
            "between_unit_isolation": "independent randomized clusters without spillover",
        },
        "units": [
            {
                "id": f"unit:{i}",
                "arm": i % 2,
                "probabilities": [0.5, 0.5],
                "outcome": (1.0 if i % 2 == 0 else 2.0) + (0.1 if i % 4 < 2 else -0.1),
                "population_weight": 1.0,
            }
            for i in range(16)
        ],
    }


def is_identity(value):
    return isinstance(value, str) and HEX64.fullmatch(value) is not None


def test_prepared_panel_and_result_name_the_response_route_and_share_one_identity():
    groups, worlds = declarations()
    prepared = request(groups, worlds)
    view = prepared.plan_view()
    assert view["route"] == "JointPanelResponse"
    assert is_identity(prepared.program_identity)
    assert prepared.inspect()["lowered_program_identity"] == prepared.program_identity
    result = prepared.estimate()
    assert result.program_identity == prepared.program_identity


def test_retarget_changes_the_program_identity_and_keeps_the_route():
    groups, worlds = declarations()
    prepared = request(groups, worlds)
    changed = [dict(world) for world in worlds]
    changed[1] = {**worlds[1], "channels": {**worlds[1]["channels"], "a": [2.0] * 24}}
    retargeted = prepared.retarget(changed)
    assert retargeted.plan_view()["route"] == "JointPanelResponse"
    assert is_identity(retargeted.program_identity)
    assert retargeted.program_identity != prepared.program_identity


def test_prepared_imputation_names_its_route():
    prepared = imputed_panel()
    assert prepared.plan_view()["route"] == "JointPanelImputation"
    assert is_identity(prepared.program_identity)


def test_joint_design_handles_name_the_design_route():
    native = jp.prepare_native_joint_design(native_recipe())
    assert native.plan_view()["route"] == "JointDesign"
    assert is_identity(native.program_identity)
    inference = {
        "inference": "randomized_gaussian",
        "prior": {
            "mean": [0.0, 0.0],
            "variance": [4.0, 4.0],
            "outcome_variance": [1.0] * 16,
            "draws": 32,
            "seed": 91,
        },
    }
    result = require_success(native.execute(inference))
    assert is_identity(result.program_identity)

    fixture = json.loads(
        (ROOT / "conformance/response/joint_panel/population_reference.json").read_text(
            encoding="utf-8"
        )
    )
    design = jp.prepare_joint_design(fixture["target"], fixture["proof"])
    assert design.plan_view()["route"] == "JointDesign"
    assert is_identity(design.program_identity)
    assert is_identity(require_success(design.execute(fixture["bayesian"])).program_identity)


def test_experiment_names_the_experiment_route():
    prepared = jp.prepare_native_joint_design(native_recipe())
    result = require_success(
        prepared.execute(
            {
                "inference": "randomized_gaussian",
                "prior": {
                    "mean": [0.0, 0.0],
                    "variance": [4.0, 4.0],
                    "outcome_variance": [1.0] * 16,
                    "draws": 32,
                    "seed": 91,
                },
            }
        )
    )
    prior = result.report()
    draws = prior["law"]["Moments"]["Bayesian"]["draws"]
    coefficients = [-1.0, 1.0]
    values = [math.fsum(v * c for v, c in zip(row, coefficients, strict=True)) for row in draws]
    target = result.experiment_quantity(coefficients=coefficients, outcome_units="outcome_unit")
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
        replace(target, functional_id="experimental_sample_mean"),
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
    updated = result.condition_experiment(
        coefficients=coefficients,
        outcome_units="outcome_unit",
        bridge=bridge,
        ledger=ledger,
        observed_mean=math.fsum(values) / len(values) + 0.3,
        observation_variance=0.2,
        sample_size=4,
        mapping_dependency=dependency,
        intercepts=[0.0] * len(values),
        slopes=[1.0] * len(values),
        model=model,
        experiment=experiment,
        mapping=mapping,
    )
    assert updated.plan_view()["route"] == "JointPanelExperiment"
    assert is_identity(updated.program_identity)
