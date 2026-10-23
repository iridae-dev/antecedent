from __future__ import annotations

import antecedent as ant
import numpy as np
import pytest
from antecedent import interference
from antecedent.errors import CausalError, CausalUnsupportedError, CausalValueError
from antecedent.experiment import StratifiedRandomization


def _design(assignment: list[bool], *, randomization=None) -> ant.experiment.ExperimentDesign:
    return ant.experiment.ExperimentDesign(
        randomization or interference.BernoulliAssignment(0.5),
        assignment,
        [f"account-{i}" for i in range(len(assignment))],
        [f"row-{i}" for i in range(len(assignment))],
    )


def test_calibrated_complete_interval_is_retained_by_analyze_and_prepare():
    assignment = [i < 30 for i in range(60)]
    outcomes = np.asarray(
        [1.0 + 2.0 * float(assignment[i]) + 0.5 * np.sin(i * 0.3) for i in range(60)]
    )
    query = ant.experiment.RandomizedEffect(
        "outcome", _design(assignment, randomization=interference.CompleteRandomization(30))
    )
    data = {"outcome": outcomes}
    result = ant.analyze(data, query=query, refute="none")
    fit = result.randomized_effect
    assert fit is not None
    assert fit.interval_95 is not None
    assert fit.interval_95[0] < fit.effect < fit.interval_95[1]
    assert fit.standard_error > 0
    assert fit.uncertainty == "complete_neyman_normal_interval"
    assert fit.support_status == "licensed"
    assert result.evidence_status == "licensed"
    prepared = ant.prepare(data, query=query, refute="none")
    assert prepared.estimate(data).randomized_effect.interval_95 == fit.interval_95


def test_bernoulli_interval_has_exact_graphless_license():
    n = 400
    assignment = [i % 2 == 0 for i in range(n)]
    outcomes = np.asarray(
        [1.0 + 2.0 * float(assignment[i]) + 0.5 * np.sin(i * 0.3) for i in range(n)]
    )
    result = ant.analyze(
        {"outcome": outcomes}, query=ant.experiment.RandomizedEffect("outcome", _design(assignment))
    )
    assert result.randomized_effect.interval_95 is not None
    assert result.randomized_effect.support_status == "licensed"
    assert result.evidence_status == "licensed"


def test_blocked_interval_has_exact_graphless_license():
    n = 120
    assignment = [i % 30 < 15 for i in range(n)]
    blocks = [f"block-{i // 30}" for i in range(n)]
    design = ant.experiment.ExperimentDesign(
        StratifiedRandomization({f"block-{i}": 15 for i in range(4)}),
        assignment,
        [f"unit-{i}" for i in range(n)],
        [f"row-{i}" for i in range(n)],
        blocks=blocks,
    )
    outcomes = np.asarray(
        [1.0 + 2.0 * float(assignment[i]) + 0.5 * np.sin(i * 0.3) for i in range(n)]
    )
    result = ant.analyze(
        {"outcome": outcomes}, query=ant.experiment.RandomizedEffect("outcome", design)
    )
    assert result.randomized_effect.interval_95 is not None
    assert result.randomized_effect.support_status == "licensed"
    assert result.evidence_status == "licensed"


def test_factorial_three_intervals_have_exact_graphless_license():
    n = 120
    primary = [False] * 30 + [True] * 30 + [False] * 30 + [True] * 30
    second = [False] * 60 + [True] * 60
    design = _design(
        primary,
        randomization=ant.experiment.FactorialRandomization(
            second, (30, 30, 30, 30), ("no_message", "message")
        ),
    )
    outcomes = np.asarray(
        [
            1.0
            + 2.0 * float(primary[i])
            + float(second[i])
            + 0.5 * float(primary[i] and second[i])
            + 0.5 * np.sin(i * 0.3)
            for i in range(n)
        ]
    )
    result = ant.analyze(
        {"outcome": outcomes}, query=ant.experiment.RandomizedEffect("outcome", design)
    )
    fit = result.randomized_effect
    assert fit.interval_95 is not None
    assert fit.second_factor_interval_95 is not None
    assert fit.factorial_interaction_interval_95 is not None
    assert fit.support_status == "licensed"
    assert result.evidence_status == "licensed"


def test_calibrated_multi_action_intervals_are_pointwise_for_each_action():
    labels = ("control", "a", "b", "c")
    assignment = tuple(labels[i % 4] for i in range(400))
    design = ant.experiment.MultiArmExperimentDesign(
        assignment,
        labels,
        [[0.25] * 4 for _ in range(400)],
        [f"account-{i}" for i in range(400)],
        [f"row-{i}" for i in range(400)],
    )
    outcomes = np.asarray([1.0 + float(i % 4) + 0.5 * np.sin(i * 0.3) for i in range(400)])
    result = ant.analyze(
        {"outcome": outcomes},
        query=ant.experiment.RandomizedEffect("outcome", design),
        refute="none",
    )
    fit = result.randomized_effect
    assert fit is not None
    assert fit.interval_95 is not None
    assert fit.uncertainty == "multi_arm_ht_score_pointwise_normal_intervals"
    assert fit.multi_arm_intervals_95[0] is None
    assert len(fit.multi_arm_intervals_95) == 4
    assert [contrast.interval_95 for contrast in fit.multi_arm_contrasts] == list(
        fit.multi_arm_intervals_95[1:]
    )
    assert all(interval is not None for interval in fit.multi_arm_intervals_95[1:])
    assert fit.support_status == "licensed"


def test_randomized_effect_runs_in_the_retained_analysis_lifecycle():
    assignment = [True, False, True, False, True, False, True, False]
    data = {"outcome": 2.0 * np.asarray(assignment, dtype=float)}
    query = ant.experiment.RandomizedEffect("outcome", _design(assignment))

    result = ant.analyze(data, query=query, refute="none")

    assert result.answer.value == pytest.approx(2.0)
    assert result.study is not None
    assert result.query.kind == "randomized_effect"
    assert result.query.design.assignment_units == tuple(f"account-{i}" for i in range(8))
    assert result.randomized_effect is not None
    assert result.randomized_effect.effect == pytest.approx(2.0)
    assert result.randomized_effect.assignment_units == tuple(f"account-{i}" for i in range(8))
    assert result.randomized_effect.uncertainty == "bernoulli_ht_design_variance_no_interval"
    assert result.evidence_status == "off_axis"


def test_fixed_cuped_runs_through_retained_analyze_with_declared_uncertainty():
    assignment = [True, False] * 4
    covariate = np.asarray([0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
    outcomes = 2.0 * np.asarray(assignment, dtype=float) + 4.0 * covariate
    query = ant.experiment.RandomizedEffect(
        "outcome",
        _design(assignment),
        cuped=ant.experiment.FixedCUPED("baseline", 4.0),
    )
    result = ant.analyze({"outcome": outcomes, "baseline": covariate}, query=query, refute="none")

    assert result.answer.value == pytest.approx(2.0)
    assert result.randomized_effect.effect == pytest.approx(2.0)
    assert (
        result.randomized_effect.uncertainty
        == "bernoulli_fixed_cuped_ht_conservative_variance_no_interval"
    )
    assert result.randomized_effect.variance_upper_bound == pytest.approx(1.0)
    assert result.evidence_status == "off_axis"
    assert result.query.cuped == query.cuped
    assert any("fixed_pre_assignment_cuped" in item for item in result.assumptions or ())


def test_fixed_cuped_refuses_unsupported_design_and_missing_baseline():
    assignment = [True, False] * 4
    with pytest.raises(CausalValueError, match="Bernoulli"):
        ant.experiment.RandomizedEffect(
            "outcome",
            _design(assignment, randomization=interference.CompleteRandomization(4)),
            cuped=ant.experiment.FixedCUPED("baseline", 1.0),
        )
    with pytest.raises(Exception, match="baseline"):
        ant.analyze(
            {"outcome": np.asarray(assignment, dtype=float)},
            query=ant.experiment.RandomizedEffect(
                "outcome", _design(assignment), cuped=ant.experiment.FixedCUPED("baseline", 1.0)
            ),
            refute="none",
        )
    with pytest.raises(CausalUnsupportedError, match="through analyze or prepare"):
        ant.experiment.RandomizedEffect(
            "outcome", _design(assignment), cuped=ant.experiment.FixedCUPED("baseline", 1.0)
        ).estimate({"outcome": np.asarray(assignment, dtype=float), "baseline": np.zeros(8)})


def test_retained_ancova_uses_multiple_baselines_and_refuses_unsupported_combinations():
    assignment = [False, True, False, True, True, False, True, False]
    x1 = np.arange(8, dtype=float)
    x2 = np.asarray([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    outcomes = 3.0 + 2.0 * np.asarray(assignment, dtype=float) + 4.0 * x1 - x2
    query = ant.experiment.RandomizedEffect(
        "outcome", _design(assignment), ancova_covariates=("x1", "x2")
    )
    result = ant.analyze({"outcome": outcomes, "x1": x1, "x2": x2}, query=query, refute="none")
    assert result.answer.value == pytest.approx(2.0)
    assert result.randomized_effect.uncertainty == "bernoulli_ancova_hc0_variance_no_interval"
    assert result.randomized_effect.variance_upper_bound == pytest.approx(0.0, abs=1e-20)
    assert any("ancova_pre_assignment_covariates" in item for item in result.assumptions or ())
    with pytest.raises(CausalValueError, match="Bernoulli"):
        ant.experiment.RandomizedEffect(
            "outcome",
            _design(assignment, randomization=interference.CompleteRandomization(4)),
            ancova_covariates=("x1",),
        )
    with pytest.raises(CausalValueError, match="distinct"):
        ant.experiment.RandomizedEffect(
            "outcome", _design(assignment), ancova_covariates=("x1", "x1")
        )


def test_randomized_effect_rejects_non_itt_and_misaligned_units():
    with pytest.raises(CausalValueError, match="supports the ITT"):
        ant.experiment.ExperimentDesign(
            interference.BernoulliAssignment(0.5),
            [True, False],
            ["a", "b"],
            ["y1", "y2"],
            estimand="tot",  # type: ignore[arg-type]
        )
    with pytest.raises(CausalValueError, match="one assignment unit"):
        ant.experiment.ExperimentDesign(
            interference.BernoulliAssignment(0.5),
            [True, False],
            ["account", "account"],
            ["y1", "y2"],
        )


def test_complete_randomization_runs_through_retained_analyze_with_neyman_variance():
    assignment = [True, False, True, False, True, False, True, False]
    query = ant.experiment.RandomizedEffect(
        "outcome",
        _design(assignment, randomization=interference.CompleteRandomization(4)),
    )
    outcomes = np.asarray([2.0, 0.0, 4.0, 2.0, 6.0, 4.0, 8.0, 6.0])

    result = ant.analyze({"outcome": outcomes}, query=query, refute="none")

    assert result.answer.value == pytest.approx(2.0)
    assert result.randomized_effect.assignment_design == "complete"
    assert result.randomized_effect.control_units == 4
    assert result.randomized_effect.treatment_units == 4
    assert (
        result.randomized_effect.uncertainty == "complete_neyman_variance_upper_bound_no_interval"
    )
    assert result.randomized_effect.variance_upper_bound == pytest.approx(10.0 / 3.0)
    assert result.evidence_status == "off_axis"


def test_factorial_randomization_retains_main_effects_interaction_and_no_interval():
    primary = [False, False, True, True, False, False, True, True]
    second = [False] * 4 + [True] * 4
    factorial = ant.experiment.FactorialRandomization(
        second, (2, 2, 2, 2), ("no_message", "message")
    )
    design = _design(primary, randomization=factorial)
    query = ant.experiment.RandomizedEffect("outcome", design)
    data = {"outcome": np.asarray([0.0, 2.0, 2.0, 4.0, 1.0, 3.0, 5.0, 7.0])}

    result = ant.analyze(data, query=query, refute="none")

    assert result.answer.value == pytest.approx(3.0)
    assert result.randomized_effect.assignment_design == "factorial_2x2"
    assert result.randomized_effect.estimand == "factorial_primary_main_effect"
    assert result.randomized_effect.second_factor_effect == pytest.approx(2.0)
    assert result.randomized_effect.factorial_interaction == pytest.approx(2.0)
    assert result.randomized_effect.variance_upper_bound == pytest.approx(1.0)
    assert result.randomized_effect.second_factor_variance == pytest.approx(1.0)
    assert result.randomized_effect.factorial_interaction_variance == pytest.approx(4.0)
    assert (
        result.randomized_effect.uncertainty
        == "factorial_cell_neyman_variance_upper_bound_no_interval"
    )
    assert result.evidence_status == "off_axis"
    assert any("known_random_assignment" in assumption for assumption in result.assumptions or ())
    with pytest.raises(CausalUnsupportedError, match="require analyze or prepare"):
        query.estimate(data)
    with pytest.raises(CausalValueError, match="at least two"):
        ant.experiment.FactorialRandomization(second, (2, 3, 1, 2))


def test_stratified_randomization_runs_through_retained_analyze_with_blocked_variance():
    assignment = [True, False, True, False, True, False, True, False]
    blocks = ["north"] * 4 + ["south"] * 4
    design = ant.experiment.ExperimentDesign(
        StratifiedRandomization({"north": 2, "south": 2}),
        assignment,
        [f"unit-{i}" for i in range(8)],
        [f"outcome-{i}" for i in range(8)],
        blocks=blocks,
        treatment_arms=("control", "treated"),
    )
    result = ant.analyze(
        {"outcome": np.asarray([2.0, 0.0, 4.0, 2.0, 6.0, 4.0, 8.0, 6.0])},
        query=ant.experiment.RandomizedEffect("outcome", design),
        refute="none",
    )

    assert result.answer.value == pytest.approx(2.0)
    assert result.randomized_effect.assignment_design == "stratified"
    assert result.randomized_effect.blocks == tuple(blocks)
    assert result.randomized_effect.variance_upper_bound == pytest.approx(1.0)
    assert (
        result.randomized_effect.uncertainty == "stratified_neyman_variance_upper_bound_no_interval"
    )


def test_stratified_randomization_refuses_blocks_without_two_units_per_arm():
    with pytest.raises(CausalValueError, match="two treated and two control"):
        ant.experiment.ExperimentDesign(
            StratifiedRandomization({"only-block": 2}),
            [True, True, False],
            ["u0", "u1", "u2"],
            ["y0", "y1", "y2"],
            blocks=["only-block"] * 3,
        )


def test_cluster_assignment_runs_through_retained_analyze_with_cluster_variance():
    query = ant.experiment.RandomizedEffect(
        "outcome",
        ant.experiment.ExperimentDesign(
            interference.ClusterRandomization([0, 0, 1, 1, 2, 2, 3, 3], 2),
            [True, True, False, False, True, True, False, False],
            ["c0", "c0", "c1", "c1", "c2", "c2", "c3", "c3"],
            [f"y{i}" for i in range(8)],
        ),
    )
    result = ant.analyze(
        {"outcome": np.asarray([2.0, 2.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0])}, query=query
    )
    assert result.randomized_effect.effect == pytest.approx(2.0)
    assert result.randomized_effect.assignment_design == "cluster"
    assert result.randomized_effect.control_units == 2
    assert result.randomized_effect.treatment_units == 2
    assert result.randomized_effect.variance_upper_bound == pytest.approx(0.0)
    assert result.randomized_effect.uncertainty == "cluster_neyman_variance_upper_bound_no_interval"
    assert result.evidence_status == "off_axis"


def test_cluster_interval_has_exact_graphless_license():
    n = 60
    assignment = [i < 30 for i in range(n)]
    design = ant.experiment.ExperimentDesign(
        interference.ClusterRandomization(list(range(n)), 30),
        assignment,
        [f"cluster-{i}" for i in range(n)],
        [f"row-{i}" for i in range(n)],
    )
    outcomes = np.asarray(
        [1.0 + 2.0 * float(assignment[i]) + 0.5 * np.sin(i * 0.3) for i in range(n)]
    )
    result = ant.analyze(
        {"outcome": outcomes}, query=ant.experiment.RandomizedEffect("outcome", design)
    )
    assert result.randomized_effect.interval_95 is not None
    assert result.randomized_effect.support_status == "licensed"
    assert result.evidence_status == "licensed"


def test_cluster_assignment_with_block_metadata_is_refused_on_retained_route():
    query = ant.experiment.RandomizedEffect(
        "outcome",
        ant.experiment.ExperimentDesign(
            interference.ClusterRandomization([0, 0, 1, 1, 2, 2, 3, 3], 2),
            [True, True, False, False, True, True, False, False],
            ["c0", "c0", "c1", "c1", "c2", "c2", "c3", "c3"],
            [f"y{i}" for i in range(8)],
            blocks=["north"] * 8,
        ),
    )
    with pytest.raises(CausalUnsupportedError, match="does not combine with block metadata"):
        ant.analyze({"outcome": np.zeros(8)}, query=query)


@pytest.mark.parametrize("design_kind", ["complete", "cluster"])
def test_direct_native_randomized_effect_runs_design_kernel_and_retains_units(design_kind):
    if design_kind == "complete":
        design = ant.experiment.ExperimentDesign(
            interference.CompleteRandomization(2),
            [True, False, True, False],
            ["u0", "u1", "u2", "u3"],
            ["y0", "y1", "y2", "y3"],
            blocks=["north", "north", "south", "south"],
            treatment_arms=("usual_care", "new_protocol"),
        )
        outcomes = [12.0, 10.0, 12.0, 10.0]
    else:
        design = ant.experiment.ExperimentDesign(
            interference.ClusterRandomization([0, 0, 1, 1], 1),
            [True, True, False, False],
            ["c0", "c0", "c1", "c1"],
            ["y0", "y1", "y2", "y3"],
            blocks=("north", "north", "south", "south"),
            treatment_arms=("usual_care", "new_protocol"),
        )
        outcomes = [12.0, 12.0, 10.0, 10.0]
    estimate = ant.experiment.RandomizedEffect("outcome", design).estimate({"outcome": outcomes})
    assert estimate.effect == pytest.approx(2.0)
    assert estimate.assignment_design == design_kind
    assert estimate.assignment_units == tuple(design.assignment_units)
    assert estimate.outcome_units == tuple(design.outcome_units)
    assert estimate.blocks == ("north", "north", "south", "south")
    assert estimate.treatment_arms == ("usual_care", "new_protocol")
    assert estimate.uncertainty == (
        "complete_neyman_variance_upper_bound_no_interval"
        if design_kind == "complete"
        else "cluster_conservative_variance_bound_no_interval"
    )
    assert estimate.support_status == "unlicensed_direct_estimator_utility"


def test_cluster_randomization_validates_unit_map_and_realized_assignment():
    design = ant.experiment.ExperimentDesign(
        interference.ClusterRandomization([0, 0, 1, 1], 1),
        [True, True, False, False],
        ["c0", "c0", "c1", "c1"],
        ["y0", "y1", "y2", "y3"],
    )
    assert design.kind == "experiment_design"
    with pytest.raises(CausalValueError, match="constant within each randomized cluster"):
        ant.experiment.ExperimentDesign(
            interference.ClusterRandomization([0, 0, 1, 1], 1),
            [True, False, False, True],
            ["c0", "c0", "c1", "c1"],
            ["y0", "y1", "y2", "y3"],
        )


def test_stratified_randomization_runs_direct_blocked_native_estimator():
    assignment = [True, False, True, False, True, False, True, False]
    blocks = ["north"] * 4 + ["south"] * 4
    design = ant.experiment.ExperimentDesign(
        ant.experiment.StratifiedRandomization({"north": 2, "south": 2}),
        assignment,
        [f"a{i}" for i in range(8)],
        [f"y{i}" for i in range(8)],
        blocks=blocks,
    )
    result = ant.experiment.RandomizedEffect("outcome", design).estimate(
        {"outcome": [3.0, 0.0, 3.0, 0.0, 13.0, 10.0, 13.0, 10.0]}
    )
    assert result.effect == pytest.approx(3.0)
    assert result.assignment_design == "stratified"
    assert result.minimum_assignment_probability == pytest.approx(0.5)
    assert result.blocks == tuple(blocks)
    with pytest.raises(ValueError, match="no retained analyze support cell"):
        ant.experiment.RandomizedEffect("outcome", design).to_interference_query()


def test_exact_randomization_test_enumerates_the_bernoulli_assignment_space():
    result = ant.experiment.exact_randomization_test(
        [2.0, 0.0, 2.0, 0.0], [True, False, True, False], 0.5
    )
    assert result.observed_effect == pytest.approx(2.0)
    assert result.exact_two_sided_p_value == pytest.approx(0.5)
    assert result.assignments_enumerated == 16
    assert result.null == "sharp_no_effect"
    with pytest.raises(ValueError, match="1 to 20"):
        ant.experiment.exact_randomization_test([1.0] * 21, [True] * 21, 0.5)


def test_multi_arm_randomized_effect_recovers_known_arm_means_and_contrasts():
    labels = ("control", "dose_low", "dose_high")
    assignment = ["control", "dose_low", "dose_high"] * 2
    design = ant.experiment.MultiArmExperimentDesign(
        assignment,
        labels,
        [[1 / 3, 1 / 3, 1 / 3]] * 6,
        [f"unit-{i}" for i in range(6)],
        [f"row-{i}" for i in range(6)],
    )
    result = ant.analyze(
        {"y": [0.0, 2.0, 5.0, 0.0, 2.0, 5.0]},
        query=ant.experiment.RandomizedEffect("y", design),
        refute="none",
    ).randomized_effect
    assert [(arm[0], arm[1]) for arm in result.multi_arm_values] == [
        ("control", 0.0),
        ("dose_low", 2.0),
        ("dose_high", 5.0),
    ]
    assert [item.effect_vs_control for item in result.multi_arm_contrasts] == pytest.approx(
        [2.0, 5.0]
    )
    assert all(item.variance_bound_vs_control >= 0 for item in result.multi_arm_contrasts)
    assert result.uncertainty == "multi_arm_covariance_free_variance_bound_no_interval"
    with pytest.raises(CausalValueError, match="observed support"):
        ant.experiment.MultiArmExperimentDesign(
            ["control", "dose_low", "control"],
            labels,
            [[1 / 3] * 3] * 3,
            ("u0", "u1", "u2"),
            ("r0", "r1", "r2"),
        )


def test_multi_arm_randomized_effect_uses_retained_analyze_and_prepare():
    labels = ("control", "low", "high")
    assignment = ("control", "low", "high") * 2
    design = ant.experiment.MultiArmExperimentDesign(
        assignment,
        labels,
        [[1 / 3] * 3] * 6,
        [f"account-{i}" for i in range(6)],
        [f"row-{i}" for i in range(6)],
    )
    query = ant.experiment.RandomizedEffect("outcome", design)
    data = {"outcome": [0.0, 2.0, 5.0, 0.0, 2.0, 5.0]}
    result = ant.analyze(data, query=query, refute="none")
    assert result.answer.value == pytest.approx(2.0)
    fit = result.randomized_effect
    assert fit is not None
    assert [arm[0] for arm in fit.multi_arm_values] == list(labels)
    assert [arm[1] for arm in fit.multi_arm_values] == pytest.approx([0.0, 2.0, 5.0])
    assert [arm[3] for arm in fit.multi_arm_values] == [2, 2, 2]
    assert [contrast.effect_vs_control for contrast in fit.multi_arm_contrasts] == pytest.approx(
        [2.0, 5.0]
    )
    assert fit.uncertainty == "multi_arm_covariance_free_variance_bound_no_interval"
    assert fit.support_status == "unlicensed_off_matrix"
    assert result.evidence_status == "off_axis"
    prepared = ant.prepare(data, query=query, refute="none")
    refreshed = prepared.estimate(data)
    assert refreshed.randomized_effect.multi_arm_values == fit.multi_arm_values
    with pytest.raises(CausalValueError, match="observed support"):
        ant.experiment.MultiArmExperimentDesign(
            ("control", "low", "low"),
            labels,
            [[1 / 3] * 3] * 3,
            ("u0", "u1", "u2"),
            ("r0", "r1", "r2"),
        )


def test_switchback_itt_uses_sequence_clustered_native_variance():
    sequence_ids = [f"s{sequence}" for sequence in range(4) for _ in range(4)]
    assignment = [True, False, False, True] * 4
    outcomes = [
        10.0 * sequence + (2.0 if treated else 0.0)
        for sequence in range(4)
        for treated in (True, False, False, True)
    ]
    design = ant.experiment.SwitchbackDesign(
        assignment,
        sequence_ids,
        [f"p{period}" for _ in range(4) for period in range(4)],
        [0.5] * 16,
        treatment_arms=("off", "on"),
    )
    estimate = ant.experiment.SwitchbackEffect("y", design).estimate({"y": outcomes})
    assert estimate.effect == pytest.approx(2.0)
    assert estimate.sequence_count == 4
    assert estimate.period_count == 16
    assert estimate.treatment_arms == ("off", "on")
    assert (
        estimate.uncertainty == "independent_sequence_cluster_sandwich_standard_error_no_interval"
    )
    assert estimate.support_status == "unlicensed_direct_estimator_utility"
    assert "arbitrary within-sequence dependence" in estimate.assumptions[1]


def test_switchback_itt_runs_in_retained_analyze_with_period_identity():
    sequences = [f"s{sequence}" for sequence in range(4) for _ in range(4)]
    periods = [f"p{period}" for _ in range(4) for period in range(4)]
    assignment = [True, False, False, True] * 4
    outcomes = [
        10.0 * sequence + (sequence + 1.0 if treated else 0.0)
        for sequence in range(4)
        for treated in (True, False, False, True)
    ]
    query = ant.experiment.SwitchbackEffect(
        "y",
        ant.experiment.SwitchbackDesign(assignment, sequences, periods, [0.5] * 16, ("off", "on")),
    )
    result = ant.analyze({"y": outcomes}, query=query, refute="none")

    assert result.answer.value == pytest.approx(2.5)
    assert result.randomized_effect.variance_upper_bound == pytest.approx(5.0 / 12.0)
    assert result.randomized_effect.assignment_design == "switchback"
    assert result.randomized_effect.periods == tuple(periods)
    assert result.randomized_effect.assignment_units == tuple(sequences)
    assert (
        result.randomized_effect.uncertainty
        == "switchback_independent_sequence_sandwich_variance_no_interval"
    )
    assert result.evidence_status == "off_axis"
    assert any("switchback_no_carryover" in item for item in result.assumptions or ())


def test_switchback_supported_interval_uses_same_native_point_as_direct_utility():
    sequences = [f"s{i // 24}" for i in range(30 * 24)]
    periods = [f"p{i % 24}" for i in range(30 * 24)]
    assignment = [(i * 137 + (i // 24) * 31) % 11 < 5 for i in range(30 * 24)]
    outcomes = np.asarray(
        [
            1.0
            + 0.5 * np.sin((i // 24) * 0.7)
            + 0.3 * np.cos((i % 24) * 0.4)
            + (1.4 if assignment[i] else 0.0)
            for i in range(30 * 24)
        ]
    )
    query = ant.experiment.SwitchbackEffect(
        "y",
        ant.experiment.SwitchbackDesign(assignment, sequences, periods, [0.5] * len(assignment)),
    )
    retained = ant.analyze({"y": outcomes}, query=query, refute="none")
    direct = query.estimate({"y": outcomes})
    fit = retained.randomized_effect
    assert fit is not None and fit.interval_95 is not None
    assert fit.interval_95[0] < fit.effect < fit.interval_95[1]
    assert fit.effect == pytest.approx(direct.effect)
    assert fit.standard_error == pytest.approx(direct.standard_error)
    assert fit.uncertainty == "switchback_independent_sequence_student_interval"
    assert retained.evidence_status == "licensed"


def test_switchback_one_arm_within_sequence_is_valid_when_global_arms_observed():
    design = ant.experiment.SwitchbackDesign(
        [True, True, False, False],
        ["s0", "s0", "s1", "s1"],
        ["p0", "p1", "p0", "p1"],
        [0.5] * 4,
    )
    result = ant.experiment.SwitchbackEffect("y", design).estimate({"y": [2.0, 2.0, 1.0, 1.0]})
    assert result.effect == pytest.approx(1.0)


def test_switchback_retained_route_refuses_duplicate_periods():
    with pytest.raises(CausalValueError, match="unique within each sequence"):
        ant.experiment.SwitchbackDesign(
            [True, False, True, False],
            ["s0", "s0", "s1", "s1"],
            ["p0", "p0", "p0", "p1"],
            [0.5] * 4,
        )


def test_complier_effect_runs_in_retained_analyze_with_first_stage():
    assignment = [True, False] * 4
    design = ant.experiment.ExperimentDesign(
        ant.interference.BernoulliAssignment(0.5),
        assignment,
        [f"u{i}" for i in range(8)],
        [f"y{i}" for i in range(8)],
        treatment_arms=("control", "encouraged"),
    )
    receipt = [True, False, True, False, False, False, False, False]
    query = ant.experiment.ComplierEffect("y", design, receipt)
    result = ant.analyze(
        {"y": [5.0, 1.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0]}, query=query, refute="none"
    )
    effect = result.randomized_effect
    assert result.answer.value == pytest.approx(4.0)
    assert effect.estimand == "cace_late"
    assert effect.intention_to_treat_effect == pytest.approx(2.0)
    assert effect.first_stage_effect == pytest.approx(0.5)
    assert effect.variance == pytest.approx(16.0 / 7.0)
    assert effect.received_treatment == tuple(receipt)
    assert effect.uncertainty == "bernoulli_wald_cace_influence_variance_no_interval"
    assert result.evidence_status == "off_axis"
    assert any("exclusion_restriction" in item for item in result.assumptions or ())
    with pytest.raises(CausalValueError, match="Bernoulli"):
        ant.experiment.ComplierEffect(
            "y",
            ant.experiment.ExperimentDesign(
                ant.interference.CompleteRandomization(4),
                assignment,
                [f"u{i}" for i in range(8)],
                [f"y{i}" for i in range(8)],
            ),
            receipt,
        )


def test_one_sided_treatment_on_treated_retains_distinct_estimand_and_refuses_two_sided_receipt():
    assignment = [True, False] * 4
    design = ant.experiment.ExperimentDesign(
        ant.interference.BernoulliAssignment(0.5),
        assignment,
        [f"u{i}" for i in range(8)],
        [f"y{i}" for i in range(8)],
        treatment_arms=("control", "encouraged"),
    )
    receipt = [True, False, True, False, False, False, False, False]
    query = ant.experiment.TreatmentOnTreated("y", design, receipt)
    result = ant.analyze({"y": [5.0, 1.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0]}, query=query)
    effect = result.randomized_effect
    assert effect.estimand == "treatment_on_treated"
    assert effect.effect == pytest.approx(4.0)
    assert effect.first_stage_effect == pytest.approx(0.5)
    assert effect.interval_95 is None
    assert effect.uncertainty == "bernoulli_one_sided_tot_influence_variance_no_interval"
    assert result.evidence_status == "off_axis"
    assert any("one_sided_noncompliance" in item for item in result.assumptions or ())
    with pytest.raises(CausalValueError, match="no control-assigned treatment receipt"):
        ant.experiment.TreatmentOnTreated("y", design, [True, True] + [False] * 6)
    with pytest.raises(CausalValueError, match="Bernoulli"):
        ant.experiment.TreatmentOnTreated(
            "y",
            ant.experiment.ExperimentDesign(
                ant.interference.CompleteRandomization(4),
                assignment,
                [f"u{i}" for i in range(8)],
                [f"y{i}" for i in range(8)],
            ),
            receipt,
        )


def test_exact_complete_randomization_inference_retains_fisher_p_value():
    design = ant.experiment.ExperimentDesign(
        ant.interference.CompleteRandomization(2),
        [True, True, False, False],
        [f"u{i}" for i in range(4)],
        [f"y{i}" for i in range(4)],
    )
    query = ant.experiment.RandomizedEffect("y", design, exact_randomization_test=True)
    result = ant.analyze({"y": [1.0, 2.0, 3.0, 4.0]}, query=query, refute="none")
    assert result.answer.value == pytest.approx(-2.0)
    assert result.randomized_effect.randomization_p_value == pytest.approx(1.0 / 3.0)
    assert result.randomized_effect.randomization_allocations == 6
    assert result.randomized_effect.variance == pytest.approx(0.5)
    assert (
        result.randomized_effect.uncertainty == "complete_neyman_variance_upper_bound_no_interval"
    )
    assert result.evidence_status == "off_axis"
    assert any("fisher_sharp_null_two_sided" in item for item in result.assumptions or ())
    with pytest.raises(CausalUnsupportedError, match="requires analyze or prepare"):
        query.estimate({"y": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(CausalValueError, match="unadjusted complete"):
        ant.experiment.RandomizedEffect(
            "y",
            ant.experiment.ExperimentDesign(
                ant.interference.BernoulliAssignment(0.5),
                [True, False, True, False],
                [f"u{i}" for i in range(4)],
                [f"y{i}" for i in range(4)],
            ),
            exact_randomization_test=True,
        )


def test_switchback_refuses_unsupported_probability_and_single_sequence():
    with pytest.raises(CausalValueError, match="at least two independent sequences"):
        ant.experiment.SwitchbackDesign([True, False], ["s0", "s0"], ["p0", "p1"], [0.5, 0.5])
    with pytest.raises(CausalValueError, match="strictly between zero and one"):
        ant.experiment.SwitchbackDesign(
            [True, False, True, False],
            ["s0", "s0", "s1", "s1"],
            ["p0", "p1", "p0", "p1"],
            [0.5, 0.0, 0.5, 0.5],
        )


def test_multi_covariate_ancova_recovers_known_treatment_and_adjustment_effects():
    assignment = [index % 2 == 0 for index in range(12)]
    x1 = np.asarray([index - 5.5 for index in range(12)], dtype=float)
    x2 = np.asarray([(index % 4) - 1.5 for index in range(12)], dtype=float)
    y = 4.0 + 2.5 * np.asarray(assignment) + 1.2 * x1 - 0.7 * x2
    design = ant.experiment.ExperimentDesign(
        interference.BernoulliAssignment(0.5),
        assignment,
        [f"u{i}" for i in range(12)],
        [f"y{i}" for i in range(12)],
    )
    result = ant.experiment.estimate_ancova_effect(
        {"y": y, "pre_x1": x1, "pre_x2": x2},
        outcome="y",
        design=design,
        covariates=("pre_x1", "pre_x2"),
    )
    assert result.effect == pytest.approx(2.5)
    assert dict(result.adjustment_coefficients) == pytest.approx({"pre_x1": 1.2, "pre_x2": -0.7})
    assert result.treated_support == result.control_support == 6
    assert result.standard_error == pytest.approx(0.0, abs=1e-12)
    assert result.uncertainty == "hc0_independent_unit_standard_error_no_interval"
    assert result.support_status == "unlicensed_direct_estimator_utility"


def test_multi_covariate_ancova_refuses_collinear_and_missing_covariates():
    assignment = [index % 2 == 0 for index in range(8)]
    design = ant.experiment.ExperimentDesign(
        interference.BernoulliAssignment(0.5),
        assignment,
        [f"u{i}" for i in range(8)],
        [f"y{i}" for i in range(8)],
    )
    x = np.arange(8, dtype=float)
    with pytest.raises(CausalValueError, match="rank deficient|collinear"):
        ant.experiment.estimate_ancova_effect(
            {"y": x, "x1": x, "x2": 2 * x},
            outcome="y",
            design=design,
            covariates=("x1", "x2"),
        )
    with pytest.raises(CausalValueError, match="columns are missing"):
        ant.experiment.estimate_ancova_effect(
            {"y": x, "x1": x},
            outcome="y",
            design=design,
            covariates=("x1", "missing"),
        )


# ------------------------------------------------------------------ Bayesian arm contrasts
#
# The Bayesian routes are explicitly unlicensed and uncalibrated. References are computed here,
# independently of the library: the exact finite-population law by enumeration of the two
# Beta-Binomial missing-count laws, and the closed-form Beta moments of the superpopulation arms.

from math import lgamma, sqrt  # noqa: E402

from antecedent.design_models import (  # noqa: E402
    BernoulliArms,
    SwitchbackModel,
)

_TRIAL_TREATED = [True, False, False, True, False, True, False, True, False, False]
_TRIAL_Y = [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]


def _beta_binomial(missing: int, a: float, b: float) -> list[float]:
    def ln_beta(x: float, y: float) -> float:
        return lgamma(x) + lgamma(y) - lgamma(x + y)

    return [
        np.exp(
            lgamma(missing + 1)
            - lgamma(k + 1)
            - lgamma(missing - k + 1)
            + ln_beta(k + a, missing - k + b)
            - ln_beta(a, b)
        )
        for k in range(missing + 1)
    ]


def _finite_population_reference() -> tuple[float, float]:
    # 4 treated (3 successes) and 6 control (2 successes) under a uniform prior: the treated mean
    # over the 10 units is (3 + BB(6; 4, 2)) / 10, the control mean (2 + BB(4; 3, 5)) / 10.
    treated = _beta_binomial(6, 4.0, 2.0)
    control = _beta_binomial(4, 3.0, 5.0)
    mean = var = 0.0
    values = {}
    for i, p in enumerate(treated):
        for j, q in enumerate(control):
            d = ((3 + i) - (2 + j)) / 10.0
            values[d] = values.get(d, 0.0) + p * q
    mean = sum(d * w for d, w in values.items())
    var = sum(w * (d - mean) ** 2 for d, w in values.items())
    return mean, sqrt(var)


def _bayes(model, n_draws=4000):
    return ant.Bayesian(backend="conjugate", n_draws=n_draws, design_model=model)


def _two_arm_query():
    design = _design(_TRIAL_TREATED, randomization=interference.CompleteRandomization(4))
    return ant.experiment.RandomizedEffect("outcome", design)


def test_bayesian_ids_are_public():
    assert ant.Estimator.BAYESIAN_RANDOMIZED_FINITE_POPULATION.value == (
        "bayesian.randomized_finite_population"
    )
    assert ant.Estimator.BAYESIAN_RANDOMIZED_SUPERPOPULATION.value == (
        "bayesian.randomized_superpopulation"
    )
    assert ant.Estimator.BAYESIAN_SWITCHBACK.value == "bayesian.switchback_effect"


def test_bayesian_finite_population_is_the_default_and_matches_enumeration():
    mean, sd = _finite_population_reference()
    result = ant.analyze(
        {"outcome": np.asarray(_TRIAL_Y)},
        query=_two_arm_query(),
        inference=_bayes(BernoulliArms(1.0, 1.0)),
        seed=3,
        refute="none",
    )
    assert result.estimate.estimator_id == "bayesian.randomized_finite_population"
    assert result.randomized_effect is None
    assert result.evidence_status != "licensed"
    assert result.posterior.effect_mean == pytest.approx(mean, abs=5e-3)
    assert result.posterior.effect_sd == pytest.approx(sd, abs=6e-3)
    assert any("estimand=finite_population" in d for d in result.diagnostics)


def test_bayesian_superpopulation_is_selected_by_estimator_id():
    # Beta(4, 2) minus Beta(3, 5): exact moments of the independent Beta arm posteriors.
    mean = 4 / 6 - 3 / 8
    sd = sqrt(4 * 2 / (36 * 7) + 3 * 5 / (64 * 9))
    result = ant.analyze(
        {"outcome": np.asarray(_TRIAL_Y)},
        query=_two_arm_query(),
        inference=_bayes(BernoulliArms(1.0, 1.0)),
        estimator=ant.Estimator.BAYESIAN_RANDOMIZED_SUPERPOPULATION,
        seed=3,
        refute="none",
    )
    assert result.estimate.estimator_id == "bayesian.randomized_superpopulation"
    assert result.posterior.effect_mean == pytest.approx(mean, abs=5e-3)
    assert result.posterior.effect_sd == pytest.approx(sd, abs=6e-3)
    assert any("estimand=superpopulation" in d for d in result.diagnostics)


def test_bayesian_multi_arm_reports_each_action_against_the_reference():
    labels = ("control", "a", "b", "c")
    successes = (3, 5, 6, 9)
    assignment = tuple(labels[k] for k in range(4) for _ in range(10))
    outcomes = np.asarray([float(r < successes[k]) for k in range(4) for r in range(10)])
    design = ant.experiment.MultiArmExperimentDesign(
        assignment,
        labels,
        [[0.25] * 4 for _ in range(40)],
        [f"account-{i}" for i in range(40)],
        [f"row-{i}" for i in range(40)],
    )
    result = ant.analyze(
        {"outcome": outcomes},
        query=ant.experiment.RandomizedEffect("outcome", design),
        inference=_bayes(BernoulliArms(1.0, 1.0)),
        estimator=ant.Estimator.BAYESIAN_RANDOMIZED_SUPERPOPULATION,
        seed=5,
        refute="none",
    )
    # Beta(4, 8) against Beta(6, 6) (exact moments of the first contrast).
    var = 4 * 8 / (144 * 13) + 6 * 6 / (144 * 13)
    assert result.posterior.effect_mean == pytest.approx(6 / 12 - 4 / 12, abs=5e-3)
    assert result.posterior.effect_sd == pytest.approx(sqrt(var), abs=6e-3)


def test_bayesian_randomized_route_refuses_an_undeclared_prior_and_a_design_based_estimator():
    data = {"outcome": np.asarray(_TRIAL_Y)}
    with pytest.raises(CausalError, match="design_model") as caught:
        ant.analyze(
            data,
            query=_two_arm_query(),
            inference=ant.Bayesian(backend="conjugate", n_draws=2000),
            refute="none",
        )
    assert caught.value.reason_code == "required_option_missing"
    with pytest.raises(CausalError, match="estimator_inference_mismatch"):
        ant.analyze(
            data,
            query=_two_arm_query(),
            inference=_bayes(BernoulliArms(1.0, 1.0)),
            estimator=ant.Estimator.RANDOMIZED_NEYMAN_ITT,
            refute="none",
        )


_SB_ASSIGNMENT = [
    [True, False, True, False],
    [False, True, False, True],
    [True, True, False, False],
    [False, False, True, True],
]
_SB_OUTCOME = [
    [2.3, 1.1, 2.9, 0.8],
    [0.7, 2.2, 1.4, 2.6],
    [2.8, 3.1, 0.9, 1.3],
    [1.2, 0.6, 2.4, 3.0],
]


def _switchback_query():
    flat = [(s, t) for s in range(4) for t in range(4)]
    design = ant.experiment.SwitchbackDesign(
        [_SB_ASSIGNMENT[s][t] for s, t in flat],
        [f"s{s}" for s, _ in flat],
        [f"p{t}" for _, t in flat],
        [0.5] * 16,
    )
    outcomes = np.asarray([_SB_OUTCOME[s][t] for s, t in flat])
    return {"outcome": outcomes}, ant.experiment.SwitchbackEffect("outcome", design)


def _sb_model(carryover, dependence, rho=None):
    return SwitchbackModel(
        carryover=carryover,
        dependence=dependence,
        rho=rho,
        effect_precision=0.5,
        nuisance_precision=0.01,
        variance_shape=2.0,
        variance_scale=1.0,
    )


def test_bayesian_switchback_matches_the_whitened_least_squares_reference():
    data, query = _switchback_query()
    result = ant.analyze(
        data,
        query=query,
        inference=_bayes(_sb_model("none", "compound_symmetric", 0.4)),
        seed=7,
        refute="none",
    )
    assert result.estimate.estimator_id == "bayesian.switchback_effect"
    assert result.evidence_status != "licensed"
    # Student-t, 20 degrees of freedom: location and scale from the stacked whitened system.
    assert result.posterior.effect_mean == pytest.approx(1.547248493, abs=5e-3)
    assert result.posterior.effect_sd == pytest.approx(0.2040153784, abs=1e-2)
    assert result.posterior.q025 == pytest.approx(1.143518649, abs=1e-2)
    assert result.posterior.q975 == pytest.approx(1.950978337, abs=1e-2)


def test_bayesian_switchback_refuses_unrestricted_carryover_and_unspecified_dependence():
    data, query = _switchback_query()
    for model, detail, next_attempt in (
        (_sb_model("unrestricted", "independent"), "carryover_unrestricted", "washout"),
        (_sb_model("none", "unspecified"), "dependence_undeclared", "sensitivity"),
    ):
        with pytest.raises(CausalError) as caught:
            ant.analyze(data, query=query, inference=_bayes(model), refute="none")
        assert caught.value.reason_code == "cell_not_licensed"
        assert detail in str(caught.value) and next_attempt in str(caught.value)
