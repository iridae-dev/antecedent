from __future__ import annotations

import antecedent as ant
import numpy as np
import pytest
from antecedent import interference
from antecedent.errors import CausalUnsupportedError, CausalValueError
from antecedent.experiment import StratifiedRandomization


def _design(assignment: list[bool], *, randomization=None) -> ant.ExperimentDesign:
    return ant.ExperimentDesign(
        randomization or interference.BernoulliAssignment(0.5),
        assignment,
        [f"account-{i}" for i in range(len(assignment))],
        [f"row-{i}" for i in range(len(assignment))],
    )


def test_randomized_effect_runs_in_the_retained_analysis_lifecycle():
    assignment = [True, False, True, False, True, False, True, False]
    data = {"outcome": 2.0 * np.asarray(assignment, dtype=float)}
    query = ant.RandomizedEffect("outcome", _design(assignment))

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


def test_randomized_effect_rejects_non_itt_and_misaligned_units():
    with pytest.raises(CausalValueError, match="supports the ITT"):
        ant.ExperimentDesign(
            interference.BernoulliAssignment(0.5),
            [True, False],
            ["a", "b"],
            ["y1", "y2"],
            estimand="tot",  # type: ignore[arg-type]
        )
    with pytest.raises(CausalValueError, match="one assignment unit"):
        ant.ExperimentDesign(
            interference.BernoulliAssignment(0.5),
            [True, False],
            ["account", "account"],
            ["y1", "y2"],
        )


def test_complete_randomization_runs_through_retained_analyze_with_neyman_variance():
    assignment = [True, False, True, False, True, False, True, False]
    query = ant.RandomizedEffect(
        "outcome",
        _design(assignment, randomization=interference.CompleteRandomization(4)),
    )
    outcomes = np.asarray([2.0, 0.0, 4.0, 2.0, 6.0, 4.0, 8.0, 6.0])

    result = ant.analyze({"outcome": outcomes}, query=query, refute="none")

    assert result.answer.value == pytest.approx(2.0)
    assert result.randomized_effect.assignment_design == "complete"
    assert result.randomized_effect.control_units == 4
    assert result.randomized_effect.treatment_units == 4
    assert result.randomized_effect.uncertainty == "complete_neyman_variance_upper_bound_no_interval"
    assert result.randomized_effect.variance_upper_bound == pytest.approx(10.0 / 3.0)
    assert result.evidence_status == "off_axis"


def test_stratified_randomization_runs_through_retained_analyze_with_blocked_variance():
    assignment = [True, False, True, False, True, False, True, False]
    blocks = ["north"] * 4 + ["south"] * 4
    design = ant.ExperimentDesign(
        StratifiedRandomization({"north": 2, "south": 2}),
        assignment,
        [f"unit-{i}" for i in range(8)],
        [f"outcome-{i}" for i in range(8)],
        blocks=blocks,
        treatment_arms=("control", "treated"),
    )
    result = ant.analyze(
        {"outcome": np.asarray([2.0, 0.0, 4.0, 2.0, 6.0, 4.0, 8.0, 6.0])},
        query=ant.RandomizedEffect("outcome", design),
        refute="none",
    )

    assert result.answer.value == pytest.approx(2.0)
    assert result.randomized_effect.assignment_design == "stratified"
    assert result.randomized_effect.blocks == tuple(blocks)
    assert result.randomized_effect.variance_upper_bound == pytest.approx(1.0)
    assert result.randomized_effect.uncertainty == "stratified_neyman_variance_upper_bound_no_interval"


def test_stratified_randomization_refuses_blocks_without_two_units_per_arm():
    with pytest.raises(CausalValueError, match="two treated and two control"):
        ant.ExperimentDesign(
            StratifiedRandomization({"only-block": 2}),
            [True, True, False],
            ["u0", "u1", "u2"],
            ["y0", "y1", "y2"],
            blocks=["only-block"] * 3,
        )


def test_cluster_assignment_remains_an_explicit_retained_analyze_refusal():
    query = ant.RandomizedEffect(
        "outcome",
        ant.ExperimentDesign(
            interference.ClusterRandomization([0, 0, 1, 1], 1),
            [True, True, False, False],
            ["c0", "c0", "c1", "c1"],
            ["y0", "y1", "y2", "y3"],
        ),
    )
    with pytest.raises(CausalUnsupportedError, match="cluster, multi-arm, factorial, and switchback"):
        ant.analyze({"outcome": np.asarray([1.0, 2.0, 0.0, 1.0])}, query=query)


@pytest.mark.parametrize("design_kind", ["complete", "cluster"])
def test_direct_native_randomized_effect_runs_design_kernel_and_retains_units(design_kind):
    if design_kind == "complete":
        design = ant.ExperimentDesign(
            interference.CompleteRandomization(2),
            [True, False, True, False],
            ["u0", "u1", "u2", "u3"],
            ["y0", "y1", "y2", "y3"],
            blocks=["north", "north", "south", "south"],
            treatment_arms=("usual_care", "new_protocol"),
        )
        outcomes = [12.0, 10.0, 12.0, 10.0]
    else:
        design = ant.ExperimentDesign(
            interference.ClusterRandomization([0, 0, 1, 1], 1),
            [True, True, False, False],
            ["c0", "c0", "c1", "c1"],
            ["y0", "y1", "y2", "y3"],
            blocks=("north", "north", "south", "south"),
            treatment_arms=("usual_care", "new_protocol"),
        )
        outcomes = [12.0, 12.0, 10.0, 10.0]
    estimate = ant.RandomizedEffect("outcome", design).estimate(
        {"outcome": outcomes}
    )
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
    design = ant.ExperimentDesign(
        interference.ClusterRandomization([0, 0, 1, 1], 1),
        [True, True, False, False],
        ["c0", "c0", "c1", "c1"],
        ["y0", "y1", "y2", "y3"],
    )
    assert design.kind == "experiment_design"
    with pytest.raises(CausalValueError, match="constant within each randomized cluster"):
        ant.ExperimentDesign(
            interference.ClusterRandomization([0, 0, 1, 1], 1),
            [True, False, False, True],
            ["c0", "c0", "c1", "c1"],
            ["y0", "y1", "y2", "y3"],
        )


def test_stratified_randomization_runs_direct_blocked_native_estimator():
    assignment = [True, False, True, False, True, False, True, False]
    blocks = ["north"] * 4 + ["south"] * 4
    design = ant.ExperimentDesign(
        ant.experiment.StratifiedRandomization({"north": 2, "south": 2}),
        assignment,
        [f"a{i}" for i in range(8)],
        [f"y{i}" for i in range(8)],
        blocks=blocks,
    )
    result = ant.RandomizedEffect("outcome", design).estimate(
        {"outcome": [3.0, 0.0, 3.0, 0.0, 13.0, 10.0, 13.0, 10.0]}
    )
    assert result.effect == pytest.approx(3.0)
    assert result.assignment_design == "stratified"
    assert result.minimum_assignment_probability == pytest.approx(0.5)
    assert result.blocks == tuple(blocks)
    with pytest.raises(ValueError, match="no retained analyze support cell"):
        ant.RandomizedEffect("outcome", design).to_interference_query()


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
    result = ant.experiment.estimate_multi_arm_effect(
        {"y": [0.0, 2.0, 5.0, 0.0, 2.0, 5.0]},
        outcome="y",
        assignment=assignment,
        action_labels=labels,
        propensities=[[1 / 3, 1 / 3, 1 / 3]] * 6,
    )
    assert result.arm_values == (("control", 0.0), ("dose_low", 2.0), ("dose_high", 5.0))
    assert [item.effect_vs_control for item in result.contrasts] == pytest.approx([2.0, 5.0])
    assert all(item.variance_bound_vs_control >= 0 for item in result.contrasts)
    assert result.uncertainty == "covariance_free_variance_bound_no_interval"
    with pytest.raises(CausalValueError, match="every declared arm needs observed support"):
        ant.experiment.estimate_multi_arm_effect(
            {"y": [0.0, 2.0, 0.0]}, outcome="y",
            assignment=["control", "dose_low", "control"], action_labels=labels,
            propensities=[[1 / 3] * 3] * 3,
        )


def test_switchback_itt_uses_sequence_clustered_native_variance():
    sequence_ids = [f"s{sequence}" for sequence in range(4) for _ in range(4)]
    assignment = [True, False, False, True] * 4
    outcomes = [10.0 * sequence + (2.0 if treated else 0.0)
                for sequence in range(4) for treated in (True, False, False, True)]
    design = ant.SwitchbackDesign(
        assignment,
        sequence_ids,
        [f"p{period}" for _ in range(4) for period in range(4)],
        [0.5] * 16,
        treatment_arms=("off", "on"),
    )
    estimate = ant.SwitchbackEffect("y", design).estimate({"y": outcomes})
    assert estimate.effect == pytest.approx(2.0)
    assert estimate.sequence_count == 4
    assert estimate.period_count == 16
    assert estimate.treatment_arms == ("off", "on")
    assert estimate.uncertainty == "independent_sequence_cluster_sandwich_standard_error_no_interval"
    assert estimate.support_status == "unlicensed_direct_estimator_utility"
    assert "arbitrary within-sequence dependence" in estimate.assumptions[1]


def test_switchback_refuses_unsupported_probability_and_single_sequence():
    with pytest.raises(CausalValueError, match="at least two independent sequences"):
        ant.SwitchbackDesign(
            [True, False], ["s0", "s0"], ["p0", "p1"], [0.5, 0.5]
        )
    with pytest.raises(CausalValueError, match="strictly between zero and one"):
        ant.SwitchbackDesign(
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
    design = ant.ExperimentDesign(
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
    design = ant.ExperimentDesign(
        interference.BernoulliAssignment(0.5),
        assignment,
        [f"u{i}" for i in range(8)],
        [f"y{i}" for i in range(8)],
    )
    x = np.arange(8, dtype=float)
    with pytest.raises(CausalValueError, match="rank deficient|collinear"):
        ant.experiment.estimate_ancova_effect(
            {"y": x, "x1": x, "x2": 2 * x},
            outcome="y", design=design, covariates=("x1", "x2"),
        )
    with pytest.raises(CausalValueError, match="columns are missing"):
        ant.experiment.estimate_ancova_effect(
            {"y": x, "x1": x},
            outcome="y", design=design, covariates=("x1", "missing"),
        )
