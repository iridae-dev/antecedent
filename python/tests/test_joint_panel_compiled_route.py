"""Every joint-panel handle reports the compiled program it executes."""

from __future__ import annotations

import json
import math
import re
from dataclasses import replace
from pathlib import Path

from antecedent import Execution
from antecedent import joint_panel as jp
from antecedent import scientific_bridge as sb
from antecedent.evidence_ledger import EvidenceLedger, SourceDeclaration
from antecedent.outcomes import require_success

from test_joint_panel import declarations, request
from test_joint_panel_imputation import panel as imputed_panel
from test_panel_future_study import joint_excursion_declaration

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


def names_route(handle, route):
    """The handle's plan view names `route` and the program identity the handle reports."""
    view = handle.plan_view()
    assert view["route"] == route
    assert is_identity(handle.program_identity)
    assert view["identity"] == handle.program_identity
    assert handle.unit_support() == {
        "unit": "joint_response.interval_readings",
        "status": "supported",
        "supported": True,
        "missing_evidence": [],
    }


def test_prepared_panel_and_result_name_the_response_route_and_share_one_identity():
    groups, worlds = declarations()
    prepared = request(groups, worlds)
    view = prepared.plan_view()
    assert view["route"] == "JointPanelResponse"
    assert prepared.unit_support()["unit"] == "joint_response.interval_readings"
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
    assert prepared.unit_support()["supported"] is True
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


def test_certified_search_names_the_search_route():
    groups, worlds = declarations()
    # The search starts from a fitted mechanism, so the fit must converge.
    prepared = request(groups, worlds, max_iterations=2000, tolerance=1e-4)
    result = require_success(prepared.estimate(execution=Execution(seed=20261021)))
    controls = [
        {
            "channel": name,
            "offset": offset,
            "lower": 0.0,
            "upper": 2.0 if offset == 23 and name == "a" else 0.0,
            "unit_cost": 1.0 if offset == 23 and name == "a" else 0.0,
        }
        for offset in range(24)
        for name in ("a", "b", "c")
    ]
    certificate = result.certify_program(
        base_world=0,
        controls=controls,
        initial=[0.0] * len(controls),
        absolute_gap=1e-3,
        max_splits=8,
        execution=Execution(seed=20261021),
    )
    names_route(certificate, "JointPanelProgramSearch")
    assert certificate.program_identity != result.program_identity


def test_randomized_selection_names_the_selection_route_before_and_after_execution():
    design = {
        "worlds": ["a", "b"],
        "sampling_unit": "independent randomized cluster",
        "population": "common cluster sampling frame",
        "consistency": "complete program and common outcome measure",
        "between_unit_isolation": "no cross-cluster interference",
    }

    def units(stage):
        return [
            {
                "id": f"{stage}:{i}",
                "arm": i % 2,
                "probabilities": [0.5, 0.5],
                "outcome": [1.0, 1.5][i % 2] + 0.1 * (i % 5),
                "population_weight": 1.0,
            }
            for i in range(64)
        ]

    policy = {
        "independent_sampling_frame": "independent draws from the declared common frame",
        "utility_units": "outcome units",
        "benefit_multiplier": 1.0,
        "utility_costs": [0.0, 0.0],
        "resource_costs": [1.0, 1.0],
        "budget": 2.0,
        "generation": {"kind": "all"},
    }
    prepared = jp.prepare_randomized_selection(
        design=design, training=units("train"), validation=units("holdout"), policy=policy
    )
    names_route(prepared, "JointPanelProgramSelection")
    selected = require_success(prepared.execute(repetitions=3, execution=Execution(seed=13)))
    names_route(selected, "JointPanelProgramSelection")
    assert selected.program_identity == prepared.program_identity


def test_instrumented_selection_names_the_selection_route():
    def units(stage):
        rows = []
        for i in range(40):
            z = -1.0 + 2.0 * i / 39
            dose = 1.5 + z
            rows.append(
                {
                    "id": f"{stage}:{i}",
                    "endogenous": [dose, dose * dose],
                    "instruments": [z, z * z],
                    "exogenous": [1.0],
                    "outcome": 4 * dose - dose * dose,
                }
            )
        return rows

    declaration = {
        "design": {
            "contrasts": [["dose_increment", [1.0, 1.0]]],
            "structural_model": "homogeneous Y = beta1*d + beta2*d^2 + intercept + error",
            "moment_restrictions": "randomized Z; E[Z error]=E[Z^2 error]=0; exclusion",
            "population": "common independent unit population",
        },
        "features": {"raw_feature_indices": [0], "powers": [[1], [2]]},
        "training": units("training"),
        "validation": units("validation"),
        "independent_sampling_frame": "independent original units reserved before fitting",
        "controls": [{"lower": 0.0, "upper": 3.0, "unit_cost": 1.0, "utility_cost": 0.0}],
        "restrictions": [],
        "budget": 1.75,
        "reference": [0.0],
        "initial": [1.0],
        "benefit_multiplier": 1.0,
        "utility_units": "outcome units",
        "generation": {"kind": "prespecified"},
        "certificate": {"absolute_gap": 0.002, "max_splits": 256},
    }
    selected = jp.select_instrumented_program(
        declaration, repetitions=2, execution=Execution(seed=17)
    )
    names_route(selected, "JointPanelProgramSelection")


def test_joint_excursion_update_names_the_experiment_route():
    result, args, _ledger, _experiment = joint_excursion_declaration()
    updated = result.condition_joint_excursion(**args)
    names_route(updated, "JointPanelExperiment")
    assert updated.program_identity != result.program_identity
