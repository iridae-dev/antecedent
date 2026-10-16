//! Tests of the graphless design license rows.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[test]
fn fixed_dose_policy_requires_every_local_and_provenance_gate() {
    let key = GraphlessSupportKey {
        family: "continuous_dose_policy",
        design: "fixed_group_kernel",
        method: "inverse_density_kernel_paired_scores",
        inference_claim: "policy_reference_incremental_pointwise_95_normal_intervals",
    };
    let support = GraphlessAssignmentSupport {
        assignment_unit: "unit",
        rows: 600,
        min_group_rows: 300,
        min_local_rows: 80,
        min_effective_sample_size: 50.0,
        max_normalized_weight: 0.05,
        min_dose_density: 0.2,
        interval_95_published: true,
        reported_intervals: 3,
        all_reported_intervals: true,
        known_density: true,
        ..Default::default()
    };
    assert!(matches!(classify_graphless(key, support), GraphlessSupportStatus::Licensed { .. }));
    for weaker in [
        GraphlessAssignmentSupport { rows: 599, ..support },
        GraphlessAssignmentSupport { min_group_rows: 299, ..support },
        GraphlessAssignmentSupport { min_local_rows: 79, ..support },
        GraphlessAssignmentSupport { min_effective_sample_size: 49.0, ..support },
        GraphlessAssignmentSupport { max_normalized_weight: 0.051, ..support },
        GraphlessAssignmentSupport { min_dose_density: 0.19, ..support },
        GraphlessAssignmentSupport { reported_intervals: 2, ..support },
        GraphlessAssignmentSupport { all_reported_intervals: false, ..support },
        GraphlessAssignmentSupport { known_density: false, ..support },
        GraphlessAssignmentSupport { interval_95_published: false, ..support },
    ] {
        assert_eq!(classify_graphless(key, weaker), GraphlessSupportStatus::Refused);
    }
    assert_eq!(
        classify_graphless(GraphlessSupportKey { design: "exact_dose", ..key }, support),
        GraphlessSupportStatus::Refused
    );
}

#[test]
fn exact_randomized_design_claims_require_measured_assignment_support() {
    let complete = GraphlessSupportKey {
        family: "randomized_effect",
        design: "complete",
        method: "neyman_difference_in_means",
        inference_claim: "pointwise_95_normal_interval",
    };
    let unit_30 = GraphlessAssignmentSupport {
        assignment_unit: "unit",
        treated: 30,
        control: 30,
        interval_95_published: true,
        reported_intervals: 1,
        ..Default::default()
    };
    assert!(matches!(
        classify_graphless(complete, unit_30),
        GraphlessSupportStatus::Licensed { .. }
    ));
    assert_eq!(
        classify_graphless(complete, GraphlessAssignmentSupport { treated: 29, ..unit_30 }),
        GraphlessSupportStatus::Refused
    );
    assert_eq!(
        classify_graphless(
            complete,
            GraphlessAssignmentSupport { interval_95_published: false, ..unit_30 }
        ),
        GraphlessSupportStatus::Refused
    );
    assert_eq!(
        classify_graphless(
            GraphlessSupportKey { inference_claim: "simultaneous_95_band", ..complete },
            unit_30
        ),
        GraphlessSupportStatus::Refused
    );
    assert_eq!(
        classify_graphless(GraphlessSupportKey { design: "bernoulli", ..complete }, unit_30),
        GraphlessSupportStatus::Refused
    );
    let cluster = GraphlessSupportKey {
        family: "randomized_effect",
        design: "cluster",
        method: "neyman_unit_weighted_cluster_totals",
        inference_claim: "pointwise_95_normal_interval",
    };
    let cluster_30 = GraphlessAssignmentSupport {
        assignment_unit: "cluster",
        treated: 30,
        control: 30,
        interval_95_published: true,
        reported_intervals: 1,
        ..Default::default()
    };
    assert!(matches!(
        classify_graphless(cluster, cluster_30),
        GraphlessSupportStatus::Licensed { .. }
    ));
    assert_eq!(classify_graphless(cluster, unit_30), GraphlessSupportStatus::Refused);
    assert_eq!(
        classify_graphless(cluster, GraphlessAssignmentSupport { control: 29, ..cluster_30 }),
        GraphlessSupportStatus::Refused
    );
}

#[test]
#[expect(clippy::too_many_lines, reason = "one table of refusal cases, each a short literal")]
fn additional_randomized_interval_rows_refuse_each_missing_gate() {
    let binary = GraphlessAssignmentSupport {
        assignment_unit: "unit",
        treated: 200,
        control: 200,
        rows: 400,
        min_probability: 0.2,
        min_expected_arm_units: 200.0,
        interval_95_published: true,
        reported_intervals: 1,
        all_reported_intervals: true,
        ..Default::default()
    };
    let bernoulli = GraphlessSupportKey {
        family: "randomized_effect",
        design: "bernoulli",
        method: "independent_action_ht_score",
        inference_claim: "pointwise_95_normal_interval",
    };
    assert!(matches!(
        classify_graphless(bernoulli, binary),
        GraphlessSupportStatus::Licensed { .. }
    ));
    // The two-action score rule is arm support, not a row count: 300 balanced rows are licensed,
    // an arm the design expects to fill with 39 rows is not.
    assert!(matches!(
        classify_graphless(
            bernoulli,
            GraphlessAssignmentSupport {
                rows: 300,
                treated: 150,
                control: 150,
                min_expected_arm_units: 150.0,
                ..binary
            }
        ),
        GraphlessSupportStatus::Licensed { .. }
    ));
    assert_eq!(
        classify_graphless(
            bernoulli,
            GraphlessAssignmentSupport { min_expected_arm_units: 39.0, ..binary }
        ),
        GraphlessSupportStatus::Refused
    );
    assert_eq!(
        classify_graphless(bernoulli, GraphlessAssignmentSupport { treated: 29, ..binary }),
        GraphlessSupportStatus::Refused
    );
    assert_eq!(
        classify_graphless(
            bernoulli,
            GraphlessAssignmentSupport { min_probability: 0.19, ..binary }
        ),
        GraphlessSupportStatus::Refused
    );

    let blocked = GraphlessSupportKey {
        design: "stratified",
        method: "blocked_neyman_difference_in_means",
        ..bernoulli
    };
    let blocks = GraphlessAssignmentSupport {
        treated: 60,
        control: 60,
        blocks: 4,
        min_block_arm: 15,
        ..binary
    };
    assert!(matches!(classify_graphless(blocked, blocks), GraphlessSupportStatus::Licensed { .. }));
    assert_eq!(
        classify_graphless(blocked, GraphlessAssignmentSupport { min_block_arm: 14, ..blocks }),
        GraphlessSupportStatus::Refused
    );

    let factorial = GraphlessSupportKey {
        design: "factorial_2x2",
        method: "fixed_cell_neyman_contrasts",
        inference_claim: "three_pointwise_95_normal_intervals",
        ..bernoulli
    };
    let cells =
        GraphlessAssignmentSupport { min_factorial_cell: 30, reported_intervals: 3, ..binary };
    assert!(matches!(
        classify_graphless(factorial, cells),
        GraphlessSupportStatus::Licensed { .. }
    ));
    assert_eq!(
        classify_graphless(
            factorial,
            GraphlessAssignmentSupport { reported_intervals: 2, ..cells }
        ),
        GraphlessSupportStatus::Refused
    );

    let multi = GraphlessSupportKey {
        design: "multi_arm",
        method: "independent_action_ht_scores",
        inference_claim: "all_action_pointwise_95_normal_intervals",
        ..bernoulli
    };
    let actions =
        GraphlessAssignmentSupport { min_action_rows: 30, reported_intervals: 3, ..binary };
    assert!(matches!(classify_graphless(multi, actions), GraphlessSupportStatus::Licensed { .. }));
    assert_eq!(
        classify_graphless(
            multi,
            GraphlessAssignmentSupport { all_reported_intervals: false, ..actions }
        ),
        GraphlessSupportStatus::Refused
    );
    assert_eq!(
        classify_graphless(multi, GraphlessAssignmentSupport { min_action_rows: 29, ..actions }),
        GraphlessSupportStatus::Refused
    );
}

#[test]
fn precision_adjusted_bernoulli_rows_are_exact_and_cap_ancova_covariates() {
    let observed = GraphlessAssignmentSupport {
        assignment_unit: "unit",
        treated: 200,
        control: 200,
        rows: 400,
        min_probability: 0.2,
        min_expected_arm_units: 200.0,
        interval_95_published: true,
        reported_intervals: 1,
        covariates: 2,
        ..Default::default()
    };
    let ancova = GraphlessSupportKey {
        family: "randomized_effect",
        design: "bernoulli",
        method: "ancova_hc0",
        inference_claim: "pointwise_95_normal_interval",
    };
    assert!(matches!(
        classify_graphless(ancova, observed),
        GraphlessSupportStatus::Licensed { .. }
    ));
    assert_eq!(
        classify_graphless(ancova, GraphlessAssignmentSupport { covariates: 3, ..observed }),
        GraphlessSupportStatus::Refused
    );
    assert_eq!(
        classify_graphless(
            ancova,
            GraphlessAssignmentSupport { min_expected_arm_units: 39.0, ..observed }
        ),
        GraphlessSupportStatus::Refused
    );
    assert!(matches!(
        classify_graphless(
            ancova,
            GraphlessAssignmentSupport {
                rows: 300,
                treated: 150,
                control: 150,
                min_expected_arm_units: 150.0,
                ..observed
            }
        ),
        GraphlessSupportStatus::Licensed { .. }
    ));
    let cuped = GraphlessSupportKey { method: "fixed_cuped_ht_score", ..ancova };
    assert!(matches!(classify_graphless(cuped, observed), GraphlessSupportStatus::Licensed { .. }));
    assert_eq!(
        classify_graphless(GraphlessSupportKey { method: "wald_cace", ..ancova }, observed),
        GraphlessSupportStatus::Refused
    );
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "builds each ownership, support and interval variant of the exact policy rows and asserts the verdict of every row against the same fixture"
)]
fn exact_policy_rows_require_ownership_support_and_every_interval() {
    let binary = GraphlessSupportKey {
        family: "policy_value",
        design: "binary_ipw",
        method: "fixed_randomized_ipw_scores",
        inference_claim: "paired_policy_incremental_pointwise_95_normal_intervals",
    };
    let observed = GraphlessAssignmentSupport {
        assignment_unit: "unit",
        treated: 60,
        control: 60,
        rows: 120,
        min_probability: 0.5,
        interval_95_published: true,
        policy_matches: 60,
        reference_matches: 60,
        reported_intervals: 2,
        all_reported_intervals: true,
        uncoupled_constraints: true,
        ..Default::default()
    };
    assert!(matches!(
        classify_graphless(binary, observed),
        GraphlessSupportStatus::Licensed { .. }
    ));
    assert_eq!(
        classify_graphless(binary, GraphlessAssignmentSupport { policy_matches: 9, ..observed }),
        GraphlessSupportStatus::Refused
    );
    assert_eq!(
        classify_graphless(
            binary,
            GraphlessAssignmentSupport { uncoupled_constraints: false, ..observed }
        ),
        GraphlessSupportStatus::Refused
    );
    assert_eq!(
        classify_graphless(
            binary,
            GraphlessAssignmentSupport { min_probability: 0.49, ..observed }
        ),
        GraphlessSupportStatus::Refused
    );

    let uplift = GraphlessSupportKey {
        design: "binary_ipw_uplift",
        method: "fixed_randomized_ipw_frozen_rank_scores",
        inference_claim: "paired_policy_incremental_and_all_uplift_pointwise_95_normal_intervals",
        ..binary
    };
    let bins = GraphlessAssignmentSupport {
        rows: 600,
        min_bin_rows: 300,
        min_bin_arm_rows: 50,
        rank_ownership: true,
        reported_intervals: 4,
        ..observed
    };
    assert!(matches!(classify_graphless(uplift, bins), GraphlessSupportStatus::Licensed { .. }));
    assert_eq!(
        classify_graphless(uplift, GraphlessAssignmentSupport { min_bin_rows: 299, ..bins }),
        GraphlessSupportStatus::Refused
    );
    assert_eq!(
        classify_graphless(uplift, GraphlessAssignmentSupport { rank_ownership: false, ..bins }),
        GraphlessSupportStatus::Refused
    );
    assert_eq!(
        classify_graphless(
            uplift,
            GraphlessAssignmentSupport { all_reported_intervals: false, ..bins }
        ),
        GraphlessSupportStatus::Refused
    );

    let cate = GraphlessSupportKey {
        design: "multi_action_ipw_cate",
        method: "fixed_randomized_multi_action_ipw_fixed_group_scores",
        inference_claim: "paired_policy_incremental_and_all_cate_pointwise_95_normal_intervals",
        ..binary
    };
    let groups = GraphlessAssignmentSupport {
        rows: 600,
        min_probability: 0.2,
        min_action_rows: 200,
        min_group_rows: 300,
        min_group_arm_rows: 50,
        reported_intervals: 6,
        ..observed
    };
    assert!(matches!(classify_graphless(cate, groups), GraphlessSupportStatus::Licensed { .. }));
    assert_eq!(
        classify_graphless(cate, GraphlessAssignmentSupport { min_group_arm_rows: 49, ..groups }),
        GraphlessSupportStatus::Refused
    );
    assert_eq!(
        classify_graphless(cate, GraphlessAssignmentSupport { reported_intervals: 2, ..groups }),
        GraphlessSupportStatus::Refused
    );
}

#[test]
fn exact_two_period_did_rows_require_independent_cluster_cells() {
    let panel = GraphlessSupportKey {
        family: "difference_in_differences",
        design: "panel_2x2",
        method: "cluster_change_scores_cr1",
        inference_claim: "pointwise_95_normal_interval",
    };
    let observed = GraphlessAssignmentSupport {
        assignment_unit: "cluster",
        treated: 30,
        control: 30,
        interval_95_published: true,
        reported_intervals: 1,
        ..Default::default()
    };
    assert!(matches!(classify_graphless(panel, observed), GraphlessSupportStatus::Licensed { .. }));
    assert_eq!(
        classify_graphless(panel, GraphlessAssignmentSupport { treated: 29, ..observed }),
        GraphlessSupportStatus::Refused
    );
    let repeated = GraphlessSupportKey {
        design: "repeated_cross_section_2x2",
        method: "four_cell_cluster_scores_cr1",
        ..panel
    };
    let cells = GraphlessAssignmentSupport {
        treated: 60,
        control: 60,
        blocks: 4,
        min_block_arm: 30,
        ..observed
    };
    assert!(matches!(classify_graphless(repeated, cells), GraphlessSupportStatus::Licensed { .. }));
    assert_eq!(
        classify_graphless(repeated, GraphlessAssignmentSupport { min_block_arm: 29, ..cells }),
        GraphlessSupportStatus::Refused
    );
    assert_eq!(
        classify_graphless(repeated, GraphlessAssignmentSupport { blocks: 3, ..cells }),
        GraphlessSupportStatus::Refused
    );
    assert_eq!(
        classify_graphless(GraphlessSupportKey { design: "staggered", ..panel }, cells),
        GraphlessSupportStatus::Refused
    );
}
