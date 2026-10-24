"""Python declarations share the public Rust generic-panel compiler and executor."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from antecedent import joint_panel
from antecedent.errors import CausalResourceError
from antecedent.intervention_programs import clip, minimum, var

ROOT = Path(__file__).resolve().parents[2]


def declarations():
    with (ROOT / "conformance/response/joint_panel/randomized_interaction.csv").open() as source:
        rows = list(csv.DictReader(source))
    groups = {}
    scenarios = []
    for group in range(4):
        selected = [row for row in rows if int(row["group"]) == group]
        # CSV names remain independent of source-role binding.
        keys = list(selected[0])
        # Columns in this independent fixture: group,time,season,z,a,b,y.
        data = {
            "a": [float(row[keys[4]]) for row in selected],
            "b": [float(row[keys[5]]) for row in selected],
            "c": [2.0] * len(selected),
            "y": [float(row[keys[6]]) if row[keys[6]] else None for row in selected],
            "z": [float(row[keys[3]]) if row[keys[3]] else None for row in selected],
            "season": [float(row[keys[2]]) for row in selected],
        }
        groups[group] = data
        scenarios.append(
            {
                "group": group,
                "history": {name: data[name][:2] for name in ("a", "b", "c")},
                "confounder": data["z"][2:],
                "season": [int(value) for value in data["season"][2:]],
            }
        )
    worlds = [
        {
            "id": "baseline",
            "reference": True,
            "channels": {"a": [0.0] * 24, "b": [0.0] * 24, "c": [2.0] * 24},
            "scenarios": scenarios,
        },
        {
            "id": "joint",
            "reference": False,
            "channels": {"a": [4.0] * 24, "b": [4.0] * 24, "c": [2.0] * 24},
            "scenarios": scenarios,
        },
    ]
    return groups, worlds


def request(groups, worlds, functional=None, interactions=None):
    return joint_panel.prepare(
        groups,
        channels=["a", "b", "c"],
        outcome="y",
        confounder="z",
        season="season",
        seasons=4,
        interval_ns=604_800_000_000_000,
        worlds=worlds,
        interactions=interactions
        if interactions is not None
        else {"programmable": var("a") * var("b") * var("c") + minimum(var("c"), 0.25)},
        options={
            "optimizer": {"max_iterations": 1, "gradient_tolerance": 1e-12},
            "inference": {"kind": "frequentist", "draws": 4},
            "initial": None,
        },
        functional=functional,
        seed=20261021,
    )


def test_generic_python_panel_matches_shared_rust_mechanism_and_retains_failed_slots():
    groups, worlds = declarations()
    prepared = request(groups, worlds)
    assert prepared.inspect()["fits_performed"] == 0
    result = prepared.estimate(seed=20261021)
    report = result.report()
    assert report["fits_performed"] == 1
    assert not report["identified"]
    assert not report["diagnostics"]["optimizer_converged"]
    assert report["draws"] == [None] * 4
    assert report["diagnostics"]["calibration"] == "unmeasured"
    point = report["point_parameters"]
    parameters = point["parameters"]
    for world, response in zip(worlds, report["point"], strict=True):
        for time, actual in enumerate(response["means"]):
            expected = 0.0
            for scenario, group_intercept in zip(
                world["scenarios"], point["group_intercepts"], strict=True
            ):
                saturated = []
                for name, channel in zip(("a", "b", "c"), parameters["channels"], strict=True):
                    exposure = sum(
                        weight
                        * (
                            world["channels"][name][time - lag]
                            if lag <= time
                            else scenario["history"][name][2 + time - lag]
                        )
                        for lag, weight in enumerate(channel["lag"])
                    )
                    saturated.append(exposure / (exposure + parameters["half_saturation"]))
                feature = saturated[0] * saturated[1] * saturated[2] + min(saturated[2], 0.25)
                expected += (
                    parameters["intercept"]
                    + group_intercept
                    + sum(
                        channel["coefficient"] * value
                        for channel, value in zip(parameters["channels"], saturated, strict=True)
                    )
                    + parameters["interactions"][0]["coefficient"] * feature
                    + parameters["confounder"] * scenario["confounder"][time]
                    + parameters["season_effects"][scenario["season"][time]]
                ) / 4
            assert actual == pytest.approx(expected, abs=1e-12)
    retargeted = prepared.retarget(worlds, seed=20261021)
    del prepared
    assert retargeted.estimate(seed=20261021).report()["point"] == report["point"]
    assert retargeted.inspect()["fits_performed"] == 1
    replay = joint_panel.consume(result.export(seed=20261021), seed=20261021)
    assert replay["point"] == report["point"]
    assert replay["draws"] == [None] * 4
    assert replay["grants_current_authority"] is False


def test_python_panel_refuses_foreign_expression_read_without_fitting():
    groups, worlds = declarations()
    with pytest.raises(Exception, match="unknown variable|unknown source variable"):
        joint_panel.prepare(
            groups,
            channels=["a", "b", "c"],
            outcome="y",
            confounder="z",
            season="season",
            seasons=4,
            interval_ns=604_800_000_000_000,
            worlds=worlds,
            interactions={"wrong": var("absent")},
            options={
                "optimizer": {"max_iterations": 1, "gradient_tolerance": 1e-5},
                "inference": {"kind": "frequentist", "draws": 0},
                "initial": None,
            },
        )


def test_compiler_derived_program_certificate_replays_current_fit_and_domain():
    groups, worlds = declarations()
    prepared = request(groups, worlds)
    result = prepared.estimate(seed=20261021)
    controls = []
    for offset in range(24):
        for name in ("a", "b", "c"):
            controls.append(
                {
                    "channel": name,
                    "offset": offset,
                    "lower": 0.0,
                    "upper": 2.0 if offset == 23 and name == "a" else 0.0,
                    "unit_cost": 1.0 if offset == 23 and name == "a" else 0.0,
                }
            )
    certificate = result.certify_program(
        base_world=0,
        controls=controls,
        initial=[0.0] * len(controls),
        absolute_gap=1e-3,
        max_splits=2000,
        seed=20261021,
    )
    report = certificate.report()
    assert report["stop"] == "Certified"
    assert report["gap"] <= 1e-3
    assert report["scope"] == "model_conditional"
    assert report["selected_value_reliability"] == "unmeasured"
    assert len(report["controls"]) == len(controls)
    artifact = certificate.export(seed=20261021)
    certificate.verify_current(artifact, seed=17)
    historical = joint_panel.consume_program_certificate(artifact, seed=99)
    assert historical["domain_identity"] == report["domain_identity"]
    assert historical["gap"] == report["gap"]
    assert historical["grants_current_authority"] is False
    changed = result.certify_program(
        base_world=0,
        controls=controls,
        initial=[0.0] * len(controls),
        utility_scale=2.0,
        absolute_gap=1e-3,
        max_splits=2000,
        seed=20261021,
    )
    with pytest.raises(Exception, match="dependencies changed"):
        changed.verify_current(artifact, seed=17)
    assert prepared.inspect()["fits_performed"] == 1


def test_randomized_program_selection_uses_original_heldout_units_and_replays():
    design = {
        "worlds": ["a", "b", "over_budget"],
        "sampling_unit": "independent randomized cluster",
        "population": "common cluster sampling frame",
        "consistency": "complete program and common outcome measure",
        "between_unit_isolation": "no cross-cluster interference",
    }

    def units(stage):
        return [
            {
                "id": f"{stage}:{i}",
                "arm": i % 3,
                "probabilities": [1 / 3] * 3,
                "outcome": [1.0, 1.1, 3.0][i % 3],
                "population_weight": 1.0,
            }
            for i in range(240)
        ]

    policy = {
        "independent_sampling_frame": "independent draws from the declared common frame",
        "utility_units": "outcome units",
        "benefit_multiplier": 1.0,
        "utility_costs": [0.0] * 3,
        "resource_costs": [1.0, 1.0, 10.0],
        "budget": 2.0,
        "generation": {"kind": "all"},
    }
    prepared = joint_panel.prepare_randomized_selection(
        design=design, training=units("train"), validation=units("holdout"), policy=policy
    )
    result = prepared.execute(repetitions=20, seed=13)
    report = result.report()
    assert report["selected_program"] == "b"
    assert report["heldout"]["value"] == pytest.approx(1.1)
    assert len(report["repetitions"]) == 20
    artifact = result.export()
    result.verify_current(artifact, seed=99)
    replay = joint_panel.consume_randomized_selection(artifact, seed=99)
    assert replay["report"] == report
    assert replay["grants_current_authority"] is False
    intervals = prepared.simultaneous_intervals(repetitions=99, seed=13)
    assert intervals["ids"] == ["a", "b"]
    assert len(intervals["roots"]) == 99
    with pytest.raises(Exception, match="share original observational units"):
        joint_panel.prepare_randomized_selection(
            design=design, training=units("train"), validation=units("train"), policy=policy
        )


def test_continuous_instrumented_selection_fits_searches_holds_out_and_replays():
    def units(stage):
        result = []
        for i in range(40):
            z = -1.0 + 2.0 * i / 39
            dose = 1.5 + z
            result.append(
                {
                    "id": f"{stage}:{i}",
                    "endogenous": [dose, dose * dose],
                    "instruments": [z, z * z],
                    "exogenous": [1.0],
                    "outcome": 4 * dose - dose * dose,
                }
            )
        return result

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
    selected = joint_panel.select_instrumented_program(declaration, repetitions=3, seed=17)
    report = selected.report()
    assert report["selected"]["controls"] == pytest.approx([1.75], abs=0.002)
    assert report["heldout"]["value"] == pytest.approx(7 - 1.75**2, abs=0.005)
    assert len(report["repetitions"]) == 3
    artifact = selected.export()
    selected.verify_current(artifact, seed=99)
    replay = joint_panel.consume_instrumented_selection(artifact, seed=99)
    assert replay["report"] == report
    assert replay["grants_current_authority"] is False
    with pytest.raises(CausalResourceError):
        joint_panel.select_instrumented_program(declaration, repetitions=0, max_bytes=1)


def test_unsupported_certificate_expression_executes_actual_best_found_search():
    groups, worlds = declarations()
    prepared = request(groups, worlds, interactions={"clipped": clip(var("a"), 0.0, 0.5)})
    result = prepared.estimate(seed=20261021)
    controls = [
        {
            "channel": name,
            "offset": offset,
            "lower": 0.0,
            "upper": 2.0 if offset == 23 and name == "a" else 0.0,
            "unit_cost": 0.0,
        }
        for offset in range(24)
        for name in ("a", "b", "c")
    ]
    searched = result.search_program(
        base_world=0,
        controls=controls,
        initial=[0.0] * len(controls),
        max_evaluations=32,
        max_iterations=4,
        seed=20261021,
    )
    assert searched["route"] == "best_found"
    assert searched["gap"] is None
    assert searched["global_certificate"] is False
    assert "interval evaluator" in searched["certificate_refusal"]
    assert searched["best"]["value"] is not None
    assert prepared.inspect()["fits_performed"] == 1


def test_scheduled_contrast_assembles_on_the_shared_draws_and_refuses_separate_worlds():
    groups, worlds = declarations()
    separate = request(groups, worlds).estimate(seed=20261021)
    with pytest.raises(Exception, match="contrast target"):
        separate.scheduled_contrast()
    prepared = request(groups, worlds, functional={"terms": [(1, 1.0), (0, -1.0)], "reference": 0})
    result = prepared.estimate(seed=20261021)
    report = result.report()
    contrast = result.scheduled_contrast("terminal_mean", level=0.9)
    assert contrast["point"] == pytest.approx(
        report["point"][1]["means"][-1] - report["point"][0]["means"][-1], abs=1e-12
    )
    # Unconverged fit: failed slots stay failed and no interval is invented.
    assert contrast["draws"] == [None] * 4
    assert contrast["interval"] is None
    assert contrast["level"] == 0.9
    with pytest.raises(Exception, match="level"):
        result.scheduled_contrast("terminal_mean", level=1.5)


def test_reference_decomposition_uses_joint_fit_and_preserves_failed_slots():
    groups, worlds = declarations()
    prepared = request(groups, worlds)
    result = prepared.estimate(seed=20261021)
    declaration = {
        "before_world": 0,
        "after_world": 1,
        "before_population_weights": [0.25] * 4,
        "after_population_weights": [0.25] * 4,
        "time_weights": [0.0] * 23 + [1.0],
        "order": ["exposure", "saturation", "population", "mechanism"],
        "periods": {"before": [0] * 24, "after": [0] * 24},
    }
    decomposition = result.decompose_reference(declaration)
    report = result.report()
    expected = report["point"][1]["means"][-1] - report["point"][0]["means"][-1]
    assert decomposition["point"]["total"] == pytest.approx(expected, abs=1e-12)
    assert decomposition["point"]["shapley"] == pytest.approx([expected, 0.0, 0.0, 0.0])
    assert len(decomposition["point"]["hybrids"]) == 16
    assert len(decomposition["point"]["by_order"]) == 24
    assert decomposition["draws"] == [None] * 4
    assert decomposition["scope"] == "model_conditional_reference_decomposition"
    assert decomposition["grants_unique_causal_attribution"] is False
    assert prepared.inspect()["fits_performed"] == 1
    with pytest.raises(CausalResourceError):
        result.decompose_reference(declaration, max_bytes=1)
