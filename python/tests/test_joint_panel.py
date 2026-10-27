"""Python declarations share the public Rust generic-panel compiler and executor."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from antecedent import Execution, ResourceLimits, joint_panel
from antecedent.errors import CausalResourceError, CausalSerializationError, CausalValueError
from antecedent.intervention_programs import clip, minimum, var
from antecedent.outcomes import require_success

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
        execution=Execution(seed=20261021),
    )


def test_generic_python_panel_matches_shared_rust_mechanism_and_retains_failed_slots():
    groups, worlds = declarations()
    prepared = request(groups, worlds)
    assert prepared.inspect()["fits_performed"] == 0
    result = require_success(prepared.estimate(execution=Execution(seed=20261021)))
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
    retargeted = prepared.retarget(worlds, execution=Execution(seed=20261021))
    del prepared
    assert (
        require_success(retargeted.estimate(execution=Execution(seed=20261021))).report()["point"]
        == report["point"]
    )
    assert retargeted.inspect()["fits_performed"] == 1
    replay = require_success(
        joint_panel.consume(
            bytes(result.export(execution=Execution(seed=20261021))),
            execution=Execution(seed=20261021),
        )
    )
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
    result = require_success(prepared.estimate(execution=Execution(seed=20261021)))
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
        execution=Execution(seed=20261021),
    )
    report = certificate.report()
    assert report["stop"] == "Certified"
    assert report["gap"] <= 1e-3
    assert report["scope"] == "model_conditional"
    assert report["selected_value_reliability"] == "unmeasured"
    assert len(report["controls"]) == len(controls)
    artifact = bytes(certificate.export(execution=Execution(seed=20261021)))
    certificate.verify_current(artifact, execution=Execution(seed=17))
    historical = joint_panel.consume_program_certificate(artifact, execution=Execution(seed=99))
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
        execution=Execution(seed=20261021),
    )
    with pytest.raises(Exception, match="dependencies changed"):
        changed.verify_current(artifact, execution=Execution(seed=17))
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
    result = require_success(prepared.execute(repetitions=20, execution=Execution(seed=13)))
    report = result.report()
    assert report["selected_program"] == "b"
    assert report["heldout"]["value"] == pytest.approx(1.1)
    assert len(report["repetitions"]) == 20
    artifact = bytes(result.export())
    result.verify_current(artifact, execution=Execution(seed=99))
    replay = joint_panel.consume_randomized_selection(artifact, execution=Execution(seed=99))
    assert replay["report"] == report
    assert replay["grants_current_authority"] is False
    intervals = prepared.simultaneous_intervals(repetitions=99, execution=Execution(seed=13))
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
    selected = joint_panel.select_instrumented_program(
        declaration, repetitions=3, execution=Execution(seed=17)
    )
    report = selected.report()
    assert report["selected"]["controls"] == pytest.approx([1.75], abs=0.002)
    assert report["heldout"]["value"] == pytest.approx(7 - 1.75**2, abs=0.005)
    assert len(report["repetitions"]) == 3
    artifact = bytes(selected.export())
    selected.verify_current(artifact, execution=Execution(seed=99))
    replay = joint_panel.consume_instrumented_selection(artifact, execution=Execution(seed=99))
    assert replay["report"] == report
    assert replay["grants_current_authority"] is False
    with pytest.raises(CausalResourceError):
        joint_panel.select_instrumented_program(
            declaration, repetitions=0, execution=Execution(limits=ResourceLimits(memory_bytes=1))
        )


def test_unsupported_certificate_expression_executes_actual_best_found_search():
    groups, worlds = declarations()
    prepared = request(groups, worlds, interactions={"clipped": clip(var("a"), 0.0, 0.5)})
    result = require_success(prepared.estimate(execution=Execution(seed=20261021)))
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
        execution=Execution(seed=20261021),
    )
    assert searched["route"] == "best_found"
    assert searched["gap"] is None
    assert searched["global_certificate"] is False
    assert "interval evaluator" in searched["certificate_refusal"]
    assert searched["best"]["value"] is not None
    assert prepared.inspect()["fits_performed"] == 1


def finite_search_controls(*, budgeted=True):
    controls = []
    for offset in range(24):
        for name in ("a", "b", "c"):
            fixed = name == "c"
            controls.append(
                {
                    "channel": name,
                    "offset": offset,
                    "lower": 2.0 if fixed else 0.0,
                    "upper": 2.0 if fixed else 4.0,
                    "unit_cost": 1.0 if budgeted and not fixed else 0.0,
                }
            )
    return controls


def test_finite_program_search_ranks_compiled_worlds_with_concrete_exclusions():
    groups, worlds = declarations()
    prepared = request(groups, worlds)
    result = require_success(prepared.estimate(execution=Execution(seed=20261021)))
    terminal = [entry["means"][-1] for entry in result.report()["point"]]
    assert abs(terminal[1] - terminal[0]) > 1e-9, (
        "the worlds must differ for a ranking to mean anything"
    )
    controls = finite_search_controls()

    ranked = result.search_declared_programs(
        controls=controls, utility_offset=0.5, utility_scale=2.0, execution=Execution(seed=20261021)
    )
    rows = {row["id"]: row for row in ranked["rows"]}
    assert sorted(rows) == ["baseline", "joint"]
    # The objective is the declared affine utility of each world's own fitted terminal mean.
    assert rows["baseline"]["value"] == pytest.approx(0.5 + 2.0 * terminal[0], abs=1e-9)
    assert rows["joint"]["value"] == pytest.approx(0.5 + 2.0 * terminal[1], abs=1e-9)
    assert ranked["best"] == [max(rows, key=lambda name: rows[name]["value"])]
    # Scheduled cost is the linear unit-cost sum of the world's own channel levels.
    assert rows["baseline"]["cost"] == pytest.approx(0.0, abs=1e-12)
    assert rows["joint"]["cost"] == pytest.approx(24 * (4.0 + 4.0), abs=1e-9)
    assert rows["joint"]["controls"][:3] == [4.0, 4.0, 2.0]
    assert all(row["feasibility"] == {"kind": "feasible"} for row in rows.values())
    assert ranked["exhaustive"] is True
    assert ranked["global_certificate"] is False
    assert ranked["continuous_gap"] is None
    assert ranked["scope"] == "model_conditional"
    assert ranked["selected_value_reliability"] == "unmeasured"

    # A negative utility scale reverses the ranking; the same fitted response is reused.
    flipped = result.search_declared_programs(
        controls=controls, utility_scale=-1.0, execution=Execution(seed=20261021)
    )
    assert flipped["best"] == [min(rows, key=lambda name: rows[name]["value"])]
    assert flipped["candidate_set_identity"] == ranked["candidate_set_identity"]

    # A budget below the joint schedule's cost excludes it before any value is attempted.
    budgeted = result.search_declared_programs(
        controls=controls, budget=100.0, execution=Execution(seed=20261021)
    )
    by_id = {row["id"]: row for row in budgeted["rows"]}
    assert by_id["joint"]["feasibility"] == {"kind": "operational", "reason": "budget"}
    assert by_id["joint"]["value"] is None
    assert budgeted["best"] == ["baseline"]

    # A hard linear restriction names itself as the exclusion reason.
    restriction = [0.0] * len(controls)
    restriction[23 * 3] = 1.0
    restricted = result.search_declared_programs(
        controls=controls,
        restrictions=[{"id": "late_a_cap", "coefficients": restriction, "bound": 3.0}],
        execution=Execution(seed=20261021),
    )
    by_id = {row["id"]: row for row in restricted["rows"]}
    assert by_id["joint"]["feasibility"] == {"kind": "operational", "reason": "late_a_cap"}
    assert restricted["best"] == ["baseline"]

    # Controls must bind every actual channel at every schedule offset.
    with pytest.raises(Exception, match="every actual channel"):
        result.search_declared_programs(controls=controls[:-1], execution=Execution(seed=20261021))
    assert prepared.inspect()["fits_performed"] == 1


def test_scheduled_contrast_assembles_on_the_shared_draws_and_refuses_separate_worlds():
    groups, worlds = declarations()
    separate = require_success(request(groups, worlds).estimate(execution=Execution(seed=20261021)))
    with pytest.raises(Exception, match="contrast target"):
        separate.scheduled_contrast()
    prepared = request(groups, worlds, functional={"terms": [(1, 1.0), (0, -1.0)], "reference": 0})
    result = require_success(prepared.estimate(execution=Execution(seed=20261021)))
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
    result = require_success(prepared.estimate(execution=Execution(seed=20261021)))
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
    support = decomposition["hybrid_support"]
    assert [entry["mask"] for entry in support] == list(range(16))
    assert decomposition["identified_reference"] == all(
        not entry["unresolved"] for entry in support
    )
    assert decomposition["identification_identity"]
    assert prepared.inspect()["fits_performed"] == 1
    with pytest.raises(CausalResourceError):
        result.decompose_reference(
            declaration, execution=Execution(limits=ResourceLimits(memory_bytes=1))
        )


def reference_declaration():
    return {
        "before_world": 0,
        "after_world": 1,
        "before_population_weights": [0.25] * 4,
        "after_population_weights": [0.25] * 4,
        "time_weights": [0.0] * 23 + [1.0],
        "order": ["exposure", "saturation", "population", "mechanism"],
        "periods": {"before": [0] * 24, "after": [0] * 24},
    }


FALLBACK = {
    "explanation": "mechanism",
    "world": 1,
    "reference_world": 0,
    "before": (0, 5),
    "after": (18, 23),
}


def window_change_of_differences(report):
    """Independent oracle: (after - before) of the active world minus that of the reference."""

    def average(world, window):
        means = report["point"][world]["means"]
        return sum(means[window[0] : window[1] + 1]) / (window[1] - window[0] + 1)

    after, before = FALLBACK["after"], FALLBACK["before"]
    return (average(1, after) - average(1, before)) - (average(0, after) - average(0, before))


def test_coarser_explanation_executes_the_window_change_of_differences_on_the_fitted_law():
    groups, worlds = declarations()
    prepared = request(groups, worlds)
    result = require_success(prepared.estimate(execution=Execution(seed=20261021)))
    report = result.report()
    attempt = result.attempt_explanation(**FALLBACK)
    assert attempt["point"] == pytest.approx(window_change_of_differences(report), abs=1e-8)
    # Failed law slots stay failed in the same positions; none is dropped or invented.
    assert [d is None for d in attempt["draws"]] == [d is None for d in report["draws"]]
    assert attempt["unique_refusal"]["explanation"] == "mechanism"
    assert "unique" in attempt["unique_attribution_refusal"]
    assert attempt["grants_unique_causal_attribution"] is False
    assert attempt["scope"] == "model_conditional_coarser_contrast"
    assert attempt["target"] != attempt["parent_target"]
    # The derived response reuses the fitted law: no second fit.
    assert prepared.inspect()["fits_performed"] == 1
    for bad in (
        {**FALLBACK, "reference_world": 1},
        {**FALLBACK, "before": (0, 19)},
        {**FALLBACK, "after": (18, 24)},
        {**FALLBACK, "world": 7},
    ):
        with pytest.raises(CausalValueError, match="explanation"):
            result.attempt_explanation(**bad)
    with pytest.raises(CausalValueError, match="unknown variant"):
        result.attempt_explanation(**{**FALLBACK, "explanation": "everything"})
    with pytest.raises(CausalResourceError):
        result.attempt_explanation(
            **FALLBACK, execution=Execution(limits=ResourceLimits(memory_bytes=1))
        )


def test_identified_reference_attempt_retains_the_conditional_numbers_and_runs_the_fallback():
    groups, worlds = declarations()
    result = require_success(request(groups, worlds).estimate(execution=Execution(seed=20261021)))
    declaration = reference_declaration()
    attempt = result.attempt_identified_reference(declaration, fallback=FALLBACK)
    requested = result.decompose_reference(declaration)
    assert attempt["requested"]["point"] == requested["point"]
    assert attempt["requested"]["identified_reference"] == requested["identified_reference"]
    assert attempt["requested"]["grants_unique_causal_attribution"] is False
    if attempt["requested"]["identified_reference"]:
        assert attempt["route"] == "supported"
        assert "executed" not in attempt
    else:
        assert attempt["route"] == "coarser"
        assert attempt["executed"]["point"] == pytest.approx(
            window_change_of_differences(result.report()), abs=1e-8
        )
        missing = [
            h
            for h in attempt["requested"]["hybrid_support"]
            if h["unresolved"] and h["obligations"]
        ]
        assert missing, "an unidentified reference names its missing premises"
    with pytest.raises(CausalValueError, match="absent compiled world"):
        result.attempt_identified_reference({**declaration, "after_world": 9}, fallback=FALLBACK)


def test_design_adaptive_reference_lists_identified_steps_and_the_designs_that_earn_the_rest():
    groups, worlds = declarations()
    result = require_success(request(groups, worlds).estimate(execution=Execution(seed=20261021)))
    declaration = reference_declaration()
    attempt = result.attempt_design_adaptive_reference(declaration, fallback=FALLBACK)
    hybrids = attempt["requested"]["point"]["hybrids"]
    for step in attempt["identified_steps"]:
        bit = ["exposure", "saturation", "population", "mechanism"].index(step["component"])
        assert not step["context_mask"] & (1 << bit)
        assert step["point"] == pytest.approx(
            hybrids[step["context_mask"] | (1 << bit)] - hybrids[step["context_mask"]], abs=1e-12
        )
    stepped = {step["component"] for step in attempt["identified_steps"]}
    assert set(attempt["unidentified_components"]).isdisjoint(stepped)
    assert len(stepped) + len(attempt["unidentified_components"]) == 4
    missing = {
        obligation
        for hybrid in attempt["requested"]["hybrid_support"]
        for obligation in hybrid["obligations"]
    }
    assert {step["obligation"] for step in attempt["next_designs"]} == missing
    assert all(step["design"] and step["hybrids"] for step in attempt["next_designs"])
    assert attempt["complete"] == attempt["requested"]["identified_reference"]
    if attempt["complete"]:
        assert attempt["coarser"] is None
    else:
        assert attempt["coarser"]["point"] == pytest.approx(
            window_change_of_differences(result.report()), abs=1e-8
        )
        # Without a fallback the finer numbers and the next designs are all that is returned.
        assert result.attempt_design_adaptive_reference(declaration)["coarser"] is None


def family_request(groups, worlds, family, *, lag_order=2, inference=None):
    return joint_panel.prepare(
        groups,
        channels=["a", "b", "c"],
        outcome="y",
        confounder="z",
        season="season",
        seasons=4,
        interval_ns=604_800_000_000_000,
        worlds=worlds,
        interactions={"programmable": var("a") * var("b") * var("c") + minimum(var("c"), 0.25)},
        options={
            "optimizer": {"max_iterations": 1, "gradient_tolerance": 1e-12},
            "inference": inference or {"kind": "frequentist", "draws": 2},
            "initial": None,
        },
        lag_order=lag_order,
        family=family,
        execution=Execution(seed=20261021),
    )


BROADER_FAMILY = {
    "adstock": "geometric",
    "saturation": "hill",
    "saturation_classes": "per_group",
    "spatial_distances": {i: {j: abs(i - j) for j in range(4)} for i in range(4)},
}


def test_broader_family_is_evaluated_by_the_shared_rust_mechanism_and_replays_from_the_artifact():
    groups, worlds = declarations()
    prepared = family_request(groups, worlds, BROADER_FAMILY)
    result = require_success(prepared.estimate(execution=Execution(seed=20261021)))
    report = result.report()
    point = report["point_parameters"]
    parameters = point["parameters"]
    family = parameters["family"]
    # The family's coordinates are present, sized to the declaration and positive.
    assert len(family["hill_shape"]) == 3
    assert len(family["class_half_saturation"]) == 3  # four per-group classes, one shared first
    assert family["spatial_range"] > 0
    assert all(value > 0 for value in family["hill_shape"] + family["class_half_saturation"])
    for channel in parameters["channels"]:
        weights = channel["lag"]
        assert weights[1] / weights[0] == pytest.approx(weights[2] / weights[1], rel=1e-9)
        assert weights[1] < weights[0]
    half_saturations = [parameters["half_saturation"], *family["class_half_saturation"]]
    # An independent evaluation of the mechanism, written without the Rust code.
    for world, response in zip(worlds, report["point"], strict=True):
        for time, actual in enumerate(response["means"]):
            expected = 0.0
            for group, (scenario, group_intercept) in enumerate(
                zip(world["scenarios"], point["group_intercepts"], strict=True)
            ):
                saturated = []
                for name, channel, shape in zip(
                    ("a", "b", "c"), parameters["channels"], family["hill_shape"], strict=True
                ):
                    exposure = sum(
                        weight
                        * (
                            world["channels"][name][time - lag]
                            if lag <= time
                            else scenario["history"][name][2 + time - lag]
                        )
                        for lag, weight in enumerate(channel["lag"])
                    )
                    k = half_saturations[group]
                    saturated.append(exposure**shape / (exposure**shape + k**shape))
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
            assert actual == pytest.approx(expected, abs=1e-10)
    assert not report["identified"]
    assert report["diagnostics"]["calibration"] == "unmeasured"
    # The frozen artifact carries the family and replays the same numbers.
    replay = require_success(
        joint_panel.consume(
            bytes(result.export(execution=Execution(seed=20261021))),
            execution=Execution(seed=20261021),
        )
    )
    assert replay["point"] == report["point"]
    assert replay["draws"] == report["draws"]
    assert replay["grants_current_authority"] is False
    # Retargeting reuses the fitted broader-family mechanism.
    retargeted = prepared.retarget(worlds, execution=Execution(seed=20261021))
    assert (
        require_success(retargeted.estimate(execution=Execution(seed=20261021))).report()["point"]
        == report["point"]
    )


def test_reviewed_family_declaration_is_unchanged_and_family_choices_are_checked():
    groups, worlds = declarations()
    reviewed = require_success(
        request(groups, worlds).estimate(execution=Execution(seed=20261021))
    ).report()
    assert "family" not in reviewed["point_parameters"]["parameters"]
    explicit = family_request(
        groups, worlds, {"adstock": "simplex", "saturation": "michaelis_menten"}
    )
    # Naming the reviewed family explicitly is the reviewed model, not a different one.
    again = require_success(explicit.estimate(execution=Execution(seed=20261021))).report()
    assert again["point"] == reviewed["point"]
    assert "family" not in again["point_parameters"]["parameters"]
    # Geometric adstock has no meaning without carryover: refused before any fit.
    with pytest.raises(Exception, match=r"."):
        family_request(groups, worlds, {"adstock": "geometric"}, lag_order=0)
    with pytest.raises(Exception, match="unknown panel family"):
        family_request(groups, worlds, {"kernel": "matern"})
    with pytest.raises(Exception, match="saturation_classes"):
        family_request(groups, worlds, {"saturation_classes": {0: 0, 1: 1}})
    with pytest.raises(Exception, match="classes"):
        family_request(groups, worlds, {"saturation_classes": {0: 0, 1: 2, 2: 2, 3: 0}})
    with pytest.raises(Exception, match="spatial"):
        family_request(groups, worlds, {"spatial_distances": {0: {0: 0.0}}})
    asymmetric = {i: {j: float(i + 2 * j) * (i != j) for j in range(4)} for i in range(4)}
    with pytest.raises(Exception, match="symmetric"):
        family_request(groups, worlds, {"spatial_distances": asymmetric})


def test_broader_family_posterior_uses_the_extra_priors():
    groups, worlds = declarations()
    inference = {
        "kind": "bayesian",
        "prior": {
            "coefficient_sd": 10.0,
            "log_half_saturation_mean": 0.0,
            "log_half_saturation_sd": 1.0,
            "lag": [],
            "innovation_scale": 2.0,
            "group_scale": 2.0,
            "family": {
                "geometric_decay": [2.0, 2.0],
                "log_hill_shape_mean": 0.0,
                "log_hill_shape_sd": 0.4,
                "class_log_saturation_sd": 0.3,
                "log_spatial_range_mean": None,
                "log_spatial_range_sd": 1.0,
            },
        },
        "sampler": {
            "chains": 2,
            "warmup": 4,
            "draws": 8,
            "proposal_scale": 0.01,
            "max_rhat": 1.5,
            "min_ess": 1.0,
        },
    }
    prepared = family_request(groups, worlds, BROADER_FAMILY, inference=inference)
    report = require_success(prepared.estimate(execution=Execution(seed=20261021))).report()
    assert len(report["draws"]) == 16
    sampled = [draw for draw in report["draws"] if draw is not None]
    assert sampled
    for draw in sampled:
        assert len(draw) == len(worlds)
        assert all(len(response["means"]) == 24 for response in draw)


def whole_program_declarations():
    """Four independent groups, one per whole-program arm, with an exogenous context."""
    with (ROOT / "conformance/response/joint_panel/whole_program_interaction.csv").open() as source:
        rows = list(csv.DictReader(source))
    arms = [("reference", 0.0, 0.0), ("a", 4.0, 0.0), ("b", 0.0, 4.0), ("joint", 4.0, 4.0)]
    groups = {}
    scenarios = []
    for group in range(4):
        selected = [row for row in rows if int(row["group"]) == group]
        data = {
            "channel_a": [float(row["channel_a"]) for row in selected],
            "channel_b": [float(row["channel_b"]) for row in selected],
            "y": [float(row["outcome"]) if row["outcome"] else None for row in selected],
            "z": [float(row["z"]) if row["z"] else None for row in selected],
            "season": [float(row["season"]) for row in selected],
        }
        groups[group] = data
        scenarios.append(
            {
                "group": group,
                "history": {name: data[name][:2] for name in ("channel_a", "channel_b")},
                "confounder": data["z"][2:],
                "season": [int(value) for value in data["season"][2:]],
            }
        )
    worlds = [
        {
            "id": name,
            "reference": index == 0,
            "channels": {"channel_a": [a] * 24, "channel_b": [b] * 24},
            "scenarios": scenarios,
        }
        for index, (name, a, b) in enumerate(arms)
    ]
    assignment = {
        "design": {"kind": "multi_arm"},
        "outcome": 2,
        "realized_assignment": [False, True, True, True],
        "assignment_probabilities": [0.25] * 4,
        "assignment_units": ["0", "1", "2", "3"],
        "outcome_units": ["0", "1", "2", "3"],
        "treatment_arms": ["reference", "a"],
        "multi_arm_labels": [name for name, _, _ in arms],
        "multi_arm_assignment": [0, 1, 2, 3],
        "multi_arm_probabilities": [[0.25] * 4 for _ in range(4)],
    }
    return groups, worlds, assignment


def whole_program_request(*, exogenous_context, functional=None):
    groups, worlds, assignment = whole_program_declarations()
    return joint_panel.prepare(
        groups,
        channels=["channel_a", "channel_b"],
        outcome="y",
        confounder="z",
        season="season",
        seasons=4,
        interval_ns=604_800_000_000_000,
        worlds=worlds,
        interactions={"ab": var("channel_a") * var("channel_b")},
        options={
            "optimizer": {"max_iterations": 1, "gradient_tolerance": 1e-5},
            "inference": {"kind": "frequentist", "draws": 0},
            "initial": None,
        },
        functional=functional,
        assignment=assignment,
        exogenous_context=exogenous_context,
        execution=Execution(seed=20261021),
    )


def test_whole_program_assignment_is_identified_only_with_the_declared_exogenous_context():
    contrast = {"terms": [(3, 1.0), (1, -1.0), (2, -1.0), (0, 1.0)], "reference": 0}
    declared = whole_program_request(exogenous_context=True, functional=contrast).inspect()
    undeclared = whole_program_request(exogenous_context=False, functional=contrast).inspect()
    for inspection in (declared, undeclared):
        assert inspection["fits_performed"] == 0
        attempts = {attempt["id"]: attempt for attempt in inspection["attempts"]}
        assert "randomized.multi_arm_itt.whole_program" in attempts
    declared_attempt = {a["id"]: a for a in declared["attempts"]}[
        "randomized.multi_arm_itt.whole_program"
    ]
    assert declared["identified"] is True
    assert declared_attempt["outcome"] == "Identified"
    assert declared_attempt["next_attempt"] is None
    undeclared_attempt = {a["id"]: a for a in undeclared["attempts"]}[
        "randomized.multi_arm_itt.whole_program"
    ]
    assert undeclared["identified"] is False
    assert undeclared_attempt["outcome"] == "Rejected"
    assert "preassignment/exogenous" in undeclared_attempt["detail"]
    assert undeclared_attempt["next_attempt"] is not None


def test_panel_target_refuses_contrasts_over_worlds_that_were_not_declared():
    groups, worlds = declarations()
    for terms in ([(1, 1.0), (7, -1.0)], [(1, 1.0), (0, 0.0)], []):
        with pytest.raises(CausalValueError):
            request(groups, worlds, functional={"terms": terms, "reference": 0})
    with pytest.raises(CausalValueError):
        request(groups, worlds, functional={"terms": [(1, 1.0), (0, -1.0)], "reference": 9})


def test_historical_panel_consumers_refuse_malformed_artifacts():
    for consume in (
        joint_panel.consume,
        joint_panel.consume_program_certificate,
        joint_panel.consume_randomized_selection,
        joint_panel.consume_instrumented_selection,
    ):
        with pytest.raises((CausalValueError, CausalSerializationError)):
            consume(b"not a retained artifact")
