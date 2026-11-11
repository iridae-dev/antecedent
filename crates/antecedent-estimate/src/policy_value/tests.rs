#![cfg_attr(
    test,
    allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "fixtures derive small nonnegative counts/indices from deterministic values"
    )
)]
use super::*;

#[test]
fn singleton_uplift_bin_keeps_its_point_with_undefined_standard_error() {
    let bins =
        uplift_by_score_bin(&[1.0, 1.0, 2.0], &[true, false, true], &[0, 1, 1], &[0.5], 2).unwrap();
    assert!((bins[0].effect - 2.0).abs() < 1e-12);
    assert!(bins[0].standard_error.is_nan());
    assert!((bins[1].effect - 1.0).abs() < 1e-12);
    assert!((bins[1].standard_error - 3.0).abs() < 1e-12);
}

#[test]
fn uplift_bin_count_beyond_the_rows_is_refused_without_sizing_by_it() {
    let refused = uplift_by_score_bin(&[1.0], &[true], &[0], &[0.5], 1 << 40);
    assert_eq!(refused.unwrap_err(), "every score bin must contain evaluation rows");
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "one end-to-end estimator fixture; splitting would fragment the flow"
)]
fn fixed_multi_action_cate_intervals_cover_group_truth_and_refuse_weak_support() {
    use std::sync::Arc;
    const PER_GROUP: usize = 300;
    const DRAWS: usize = 2_000;
    let n = 2 * PER_GROUP;
    let groups = (0..n)
        .map(|i| Arc::<str>::from(if i < PER_GROUP { "g0" } else { "g1" }))
        .collect::<Vec<_>>();
    let truths = [2.0, 3.0, 1.0, 4.0];
    let mut covered = [0_usize; 4];
    let mut supported = [0_usize; 4];
    let mut state = 0x8a71_b65d_53cc_f093_u64;
    let mut uniform = || crate::splitmix::splitmix64_unit(&mut state);
    for _ in 0..DRAWS {
        let assignment = (0..n).map(|_| (3.0 * uniform()).floor() as usize).collect::<Vec<_>>();
        let outcome = assignment
            .iter()
            .enumerate()
            .map(|(i, &action)| {
                let group = usize::from(i >= PER_GROUP);
                let action_effects = if group == 0 { [0.0, 2.0, 3.0] } else { [0.0, 1.0, 4.0] };
                1.0 + action_effects[action] + 2.0 * (uniform() - 0.5)
            })
            .collect::<Vec<_>>();
        let policy = antecedent_core::MultiActionPolicyInputs {
            action_labels: Arc::from([Arc::from("control"), Arc::from("A"), Arc::from("B")]),
            assignment: Arc::from(assignment),
            actions: Arc::from(vec![0; n]),
            reference: Arc::from(vec![0; n]),
            propensities: Arc::from((0..n).flat_map(|_| [1.0 / 3.0; 3]).collect::<Vec<_>>()),
            available: Arc::from(vec![true; n * 3]),
            costs: Arc::from([0.0; 3]),
            reference_costs: Arc::from([0.0; 3]),
            capacities: Arc::from([n; 3]),
            reference_capacities: Arc::from([n; 3]),
            budget: None,
            reference_budget: None,
            cate_groups: Arc::from(groups.clone()),
        };
        let points = evaluate_multi_action_cate(&outcome, &policy).unwrap();
        assert_eq!(points.len(), 4);
        for (index, point) in points.iter().enumerate() {
            if let Some(bounds) = point.interval_95 {
                supported[index] += 1;
                covered[index] +=
                    usize::from(bounds[0] <= truths[index] && truths[index] <= bounds[1]);
            }
        }
    }
    for (index, hits) in covered.into_iter().enumerate() {
        let rate = hits as f64 / supported[index] as f64;
        eprintln!("multi-action CATE {index}: {hits}/{}, coverage={rate:.4}", supported[index]);
        assert!(supported[index] >= 1_990);
        assert!((0.93..=0.97).contains(&rate), "CATE contrast {index} coverage {rate}");
    }

    // The interval requires at least 300 rows in each fixed baseline stratum.
    let mut assignment = (0..299).map(|i| i % 3).collect::<Vec<_>>();
    let outcome = assignment.iter().map(|&action| 1.0 + action as f64).collect::<Vec<_>>();
    let policy = antecedent_core::MultiActionPolicyInputs {
        action_labels: Arc::from([Arc::from("control"), Arc::from("A"), Arc::from("B")]),
        assignment: Arc::from(std::mem::take(&mut assignment)),
        actions: Arc::from(vec![0; 299]),
        reference: Arc::from(vec![0; 299]),
        propensities: Arc::from((0..299).flat_map(|_| [1.0 / 3.0; 3]).collect::<Vec<_>>()),
        available: Arc::from(vec![true; 299 * 3]),
        costs: Arc::from([0.0; 3]),
        reference_costs: Arc::from([0.0; 3]),
        capacities: Arc::from([299; 3]),
        reference_capacities: Arc::from([299; 3]),
        budget: None,
        reference_budget: None,
        cate_groups: Arc::from(vec![Arc::<str>::from("g0"); 299]),
    };
    assert!(
        evaluate_multi_action_cate(&outcome, &policy)
            .unwrap()
            .iter()
            .all(|p| p.interval_95.is_none())
    );
    let mut sparse = policy.clone();
    sparse.assignment = Arc::from(
        (0..300)
            .map(|i| {
                if i < 49 {
                    1
                } else if i < 149 {
                    0
                } else {
                    2
                }
            })
            .collect::<Vec<_>>(),
    );
    sparse.actions = Arc::from(vec![0; 300]);
    sparse.reference = Arc::from(vec![0; 300]);
    sparse.propensities = Arc::from((0..300).flat_map(|_| [1.0 / 3.0; 3]).collect::<Vec<_>>());
    sparse.available = Arc::from(vec![true; 300 * 3]);
    sparse.capacities = Arc::from([300; 3]);
    sparse.reference_capacities = Arc::from([300; 3]);
    sparse.cate_groups = Arc::from(vec![Arc::<str>::from("g0"); 300]);
    let outcome = sparse.assignment.iter().map(|&action| 1.0 + action as f64).collect::<Vec<_>>();
    let points = evaluate_multi_action_cate(&outcome, &sparse).unwrap();
    assert_eq!(points[0].observed_action_rows, 49);
    assert!(points[0].interval_95.is_none());
    let mut weak_overlap = sparse.clone();
    weak_overlap.assignment = Arc::from((0..300).map(|i| i % 3).collect::<Vec<_>>());
    weak_overlap.propensities =
        Arc::from((0..300).flat_map(|_| [0.45, 0.45, 0.10]).collect::<Vec<_>>());
    let outcome =
        weak_overlap.assignment.iter().map(|&action| 1.0 + action as f64).collect::<Vec<_>>();
    let points = evaluate_multi_action_cate(&outcome, &weak_overlap).unwrap();
    assert!(points[1].interval_95.is_none());
}

#[test]
fn boundary_probability_multi_action_cate_intervals_cover_known_truth() {
    use std::sync::Arc;
    const N: usize = 300;
    const DRAWS: usize = 2_000;
    let mut state = 0xb41d_8ac3_67f0_2219_u64;
    let mut uniform = || crate::splitmix::splitmix64_unit(&mut state);
    let mut covered = [0_usize; 2];
    let mut supported = [0_usize; 2];
    for _ in 0..DRAWS {
        let assignment = (0..N)
            .map(|_| {
                let u = uniform();
                if u < 0.5 {
                    0
                } else if u < 0.8 {
                    1
                } else {
                    2
                }
            })
            .collect::<Vec<_>>();
        let outcome = assignment
            .iter()
            .map(|&action| [1.0, 3.0, 4.0][action] + 2.0 * (uniform() - 0.5))
            .collect::<Vec<_>>();
        let policy = antecedent_core::MultiActionPolicyInputs {
            action_labels: Arc::from([Arc::from("control"), Arc::from("A"), Arc::from("B")]),
            assignment: Arc::from(assignment),
            actions: Arc::from(vec![0; N]),
            reference: Arc::from(vec![0; N]),
            propensities: Arc::from((0..N).flat_map(|_| [0.5, 0.3, 0.2]).collect::<Vec<_>>()),
            available: Arc::from(vec![true; N * 3]),
            costs: Arc::from([0.0; 3]),
            reference_costs: Arc::from([0.0; 3]),
            capacities: Arc::from([N; 3]),
            reference_capacities: Arc::from([N; 3]),
            budget: None,
            reference_budget: None,
            cate_groups: Arc::from(vec![Arc::<str>::from("g0"); N]),
        };
        let points = evaluate_multi_action_cate(&outcome, &policy).unwrap();
        for (index, point) in points.iter().enumerate() {
            if let Some(bounds) = point.interval_95 {
                supported[index] += 1;
                let truth = [2.0, 3.0][index];
                covered[index] += usize::from(bounds[0] <= truth && truth <= bounds[1]);
            }
        }
    }
    for (index, hits) in covered.into_iter().enumerate() {
        let coverage = hits as f64 / supported[index] as f64;
        eprintln!("boundary CATE {index}: {hits}/{}, coverage={coverage:.4}", supported[index]);
        assert!(supported[index] >= 1_700);
        assert!((0.93..=0.985).contains(&coverage));
    }
}

#[test]
fn held_out_uplift_bin_intervals_cover_fixed_rank_truth() {
    const BIN_ROWS: usize = 300;
    const DRAWS: usize = 2_000;
    let bins = (0..BIN_ROWS * 2).map(|i| usize::from(i >= BIN_ROWS)).collect::<Vec<_>>();
    let mut state = 0x3721_f0de_8b64_5c19_u64;
    let mut uniform = || crate::splitmix::splitmix64_unit(&mut state);
    let truths = [2.0, 0.5];
    let mut covered = [0_usize; 2];
    for _ in 0..DRAWS {
        let assignment = (0..BIN_ROWS * 2).map(|_| uniform() < 0.5).collect::<Vec<_>>();
        let outcome = (0..BIN_ROWS * 2)
            .map(|i| 1.0 + truths[bins[i]] * f64::from(assignment[i]) + 2.0 * (uniform() - 0.5))
            .collect::<Vec<_>>();
        let points = evaluate_uplift_bins(&outcome, &assignment, &[0.5], &bins, 2).unwrap();
        for (index, point) in points.iter().enumerate() {
            let interval = point
                .interval_95
                .expect("300 independently randomized subjects per fixed rank bin");
            covered[index] +=
                usize::from(interval[0] <= truths[index] && truths[index] <= interval[1]);
        }
    }
    for (index, hits) in covered.into_iter().enumerate() {
        let rate = hits as f64 / DRAWS as f64;
        eprintln!("uplift bin {index}: {hits}/{DRAWS}, coverage={rate:.4}");
        assert!((0.93..=0.97).contains(&rate), "pointwise 95% uplift-bin coverage {rate}");
    }
    let assignment = (0..299).map(|i| i % 2 == 0).collect::<Vec<_>>();
    let point =
        evaluate_uplift_bins(&vec![1.0; 299], &assignment, &[0.5], &vec![0; 299], 1).unwrap();
    assert!(point[0].interval_95.is_none());
    let assignment = (0..300).map(|i| i < 49).collect::<Vec<_>>();
    let point =
        evaluate_uplift_bins(&vec![1.0; 300], &assignment, &[0.5], &vec![0; 300], 1).unwrap();
    assert!(point[0].interval_95.is_none());
}

#[test]
fn recovers_known_constant_outcome_values() {
    let result = evaluate_policy_value_scores(
        &[1.0, 3.0, 1.0, 3.0],
        &BinaryPolicyDesign {
            assignment: &[false, true, false, true],
            actions: &[false, true, true, false],
            propensity: &[0.5],
            reference: &[false; 4],
            costs: &[0.0],
            reference_costs: &[0.0],
        },
        &[1.0; 4],
        &[3.0; 4],
    )
    .unwrap();
    assert!((result.policy_value - 2.0).abs() < 1e-12);
    assert!((result.reference_value - 1.0).abs() < 1e-12);
    assert!((result.incremental_value - 1.0).abs() < 1e-12);
    assert!(result.incremental_standard_error.is_finite());
}

#[test]
fn refuses_missing_overlap() {
    assert!(
        evaluate_policy_value_scores(
            &[1.0, 2.0],
            &BinaryPolicyDesign {
                assignment: &[false, true],
                actions: &[false, true],
                propensity: &[0.0],
                reference: &[false; 2],
                costs: &[0.0],
                reference_costs: &[0.0]
            },
            &[1.0; 2],
            &[2.0; 2]
        )
        .is_err()
    );
}

#[test]
fn ipw_policy_budget_uses_query_tolerance_for_policy_and_reference() {
    let outcomes = [1.0; 3];
    let actions = [true; 3];
    let binary = BinaryIpwDesign {
        assignment: &actions,
        actions: &actions,
        propensity: &[0.5],
        reference: Some(&actions),
        costs: &[0.1],
        reference_costs: None,
        available: None,
    };
    let limits =
        BinaryPolicyLimits { budget: Some(0.3), reference_budget: Some(0.3), ..Default::default() };
    assert!(evaluate_binary_ipw_policy(&outcomes, &binary, &limits).is_ok());
    let low = BinaryPolicyLimits { budget: Some(0.2999), ..limits };
    assert!(evaluate_binary_ipw_policy(&outcomes, &binary, &low).is_err());
    let low_reference = BinaryPolicyLimits { reference_budget: Some(0.2999), ..limits };
    assert!(evaluate_binary_ipw_policy(&outcomes, &binary, &low_reference).is_err());
    let overflow = BinaryIpwDesign { costs: &[f64::MAX], ..binary };
    assert!(
        evaluate_binary_ipw_policy(&outcomes, &overflow, &BinaryPolicyLimits::default()).is_err()
    );

    let selected = [1usize; 3];
    let probabilities = [0.5; 6];
    let multi = MultiActionIpwDesign {
        assigned: &selected,
        actions: &selected,
        probabilities: &probabilities,
        probability_rows: 3,
        action_count: 2,
        reference: Some(&selected),
        costs: Some(&[0.0, 0.1]),
        reference_costs: None,
        available: None,
    };
    let limits =
        MultiActionLimits { budget: Some(0.3), reference_budget: Some(0.3), ..Default::default() };
    assert!(evaluate_multi_action_ipw_policy(&outcomes, &multi, &limits).is_ok());
    assert!(
        evaluate_multi_action_ipw_policy(
            &outcomes,
            &multi,
            &MultiActionLimits { budget: Some(0.2999), ..limits }
        )
        .is_err()
    );
    assert!(
        evaluate_multi_action_ipw_policy(
            &outcomes,
            &multi,
            &MultiActionLimits { reference_budget: Some(0.2999), ..limits }
        )
        .is_err()
    );
}

#[test]
fn ipw_uses_known_assignment_probabilities_without_predictions() {
    let result = evaluate_policy_value_ipw_scores(
        &[1.0, 3.0, 1.0, 3.0],
        &BinaryPolicyDesign {
            assignment: &[false, true, false, true],
            actions: &[false, true, false, true],
            propensity: &[0.5],
            reference: &[false; 4],
            costs: &[0.0],
            reference_costs: &[0.0],
        },
    )
    .unwrap();
    assert!((result.policy_value - 4.0).abs() < 1e-12);
    assert!((result.reference_value - 1.0).abs() < 1e-12);
    assert!((result.incremental_value - 3.0).abs() < 1e-12);
    assert!(result.incremental_standard_error.is_finite());
}

#[test]
fn pointwise_intervals_use_paired_incremental_scores_and_refuse_weak_support() {
    let n = 200;
    let outcomes = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { 3.0 }).collect::<Vec<_>>();
    let assignment = (0..n).map(|i| i % 2 == 1).collect::<Vec<_>>();
    let actions = (0..n).map(|i| i % 3 == 0).collect::<Vec<_>>();
    let reference = vec![false; n];
    let score = evaluate_policy_value_ipw_scores(
        &outcomes,
        &BinaryPolicyDesign {
            assignment: &assignment,
            actions: &actions,
            propensity: &[0.5],
            reference: &reference,
            costs: &[0.25],
            reference_costs: &[0.0],
        },
    )
    .unwrap();
    let policy_matches = assignment.iter().zip(&actions).filter(|(a, b)| a == b).count();
    let reference_matches = assignment.iter().filter(|&&a| !a).count();
    let intervals = pointwise_intervals_95(&score, n, policy_matches, reference_matches).unwrap();
    let z = antecedent_stats::normal_ppf(0.975);
    assert!(
        (intervals.policy[0] - (score.policy_value - z * score.policy_standard_error)).abs()
            < 1e-12
    );
    assert!(
        (intervals.incremental[0]
            - (score.incremental_value - z * score.incremental_standard_error))
            .abs()
            < 1e-12
    );
    assert!(pointwise_intervals_95(&score, n, 9, reference_matches).is_none());
    assert!(pointwise_intervals_95(&score, 119, policy_matches, reference_matches).is_none());
}

#[test]
fn pointwise_policy_interval_repeated_sampling_coverage() {
    // Independently randomized evaluation subjects, fixed recommendations,
    // and a shared outcome shock for the two potential outcomes. A paired
    // difference score must retain the within-row covariance.
    let n = 120;
    let simulations = 2_000;
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut uniform = || crate::splitmix::splitmix64_unit(&mut state);
    let reference = vec![false; n];
    let truth_policy = 1.0 + 1.85 / 3.0;
    let truth_incremental = truth_policy - 1.0;
    let mut policy_hits_95 = 0;
    let mut incremental_hits_95 = 0;
    let mut policy_hits_90 = 0;
    let mut incremental_hits_90 = 0;
    for _ in 0..simulations {
        let actions = (0..n).map(|_| uniform() < 1.0 / 3.0).collect::<Vec<_>>();
        let assignment = (0..n).map(|_| uniform() < 0.5).collect::<Vec<_>>();
        let outcomes = assignment
            .iter()
            .map(|&a| 1.0 + 2.0 * f64::from(a) + (uniform() - 0.5) * 2.0)
            .collect::<Vec<_>>();
        let score = evaluate_policy_value_ipw_scores(
            &outcomes,
            &BinaryPolicyDesign {
                assignment: &assignment,
                actions: &actions,
                propensity: &[0.5],
                reference: &reference,
                costs: &[0.15],
                reference_costs: &[0.0],
            },
        )
        .unwrap();
        let policy_matches = assignment.iter().zip(&actions).filter(|(a, b)| a == b).count();
        let reference_matches = assignment.iter().filter(|&&a| !a).count();
        let interval =
            pointwise_intervals_95(&score, n, policy_matches, reference_matches).unwrap();
        policy_hits_95 +=
            usize::from(interval.policy[0] <= truth_policy && truth_policy <= interval.policy[1]);
        incremental_hits_95 += usize::from(
            interval.incremental[0] <= truth_incremental
                && truth_incremental <= interval.incremental[1],
        );
        let z90 = antecedent_stats::normal_ppf(0.95);
        policy_hits_90 += usize::from(
            (score.policy_value - truth_policy).abs() <= z90 * score.policy_standard_error,
        );
        incremental_hits_90 += usize::from(
            (score.incremental_value - truth_incremental).abs()
                <= z90 * score.incremental_standard_error,
        );
    }
    for (index, (hits, target)) in [
        (policy_hits_95, 0.95),
        (incremental_hits_95, 0.95),
        (policy_hits_90, 0.90),
        (incremental_hits_90, 0.90),
    ]
    .into_iter()
    .enumerate()
    {
        let rate = hits as f64 / f64::from(simulations);
        let mcse = (target * (1.0 - target) / f64::from(simulations)).sqrt();
        assert!(
            (rate - target).abs() <= 3.0 * mcse,
            "cell {index} coverage {rate} missed target {target}"
        );
    }
}

#[test]
fn held_out_aipw_and_constrained_multi_action_intervals_cover_known_values() {
    use std::sync::Arc;
    let n = 300;
    let simulations = 2_000;
    let mut state = 0xD1B5_4A32_4F6C_91E7_u64;
    let mut uniform = || crate::splitmix::splitmix64_unit(&mut state);
    let binary_reference = vec![false; n];
    let multi_reference = vec![0; n];
    let multi_costs = [0.0, 0.1, 0.2];
    let multi_truth = (1.0 + 1.9 + 3.8) / 3.0;
    let binary_truth = 1.0 + 1.85 / 3.0;
    let mut counts = [0_usize; 5];
    for _ in 0..simulations {
        // Baseline features and their fixed policy recommendations are
        // sampled independently for each evaluation subject.
        let binary_actions = (0..n).map(|_| uniform() < 1.0 / 3.0).collect::<Vec<_>>();
        let assignment = (0..n).map(|_| uniform() < 0.5).collect::<Vec<_>>();
        let outcomes = assignment
            .iter()
            .map(|&a| 1.0 + 2.0 * f64::from(a) + (uniform() - 0.5) * 2.0)
            .collect::<Vec<_>>();
        let score = evaluate_policy_value_scores(
            &outcomes,
            &BinaryPolicyDesign {
                assignment: &assignment,
                actions: &binary_actions,
                propensity: &[0.5],
                reference: &binary_reference,
                costs: &[0.15],
                reference_costs: &[0.0],
            },
            &vec![1.0; n],
            &vec![3.0; n],
        )
        .unwrap();
        let pm = assignment.iter().zip(&binary_actions).filter(|(a, b)| a == b).count();
        let rm = assignment.iter().filter(|&&a| !a).count();
        let ci = pointwise_intervals_95(&score, n, pm, rm).unwrap();
        counts[0] += usize::from(ci.policy[0] <= binary_truth && binary_truth <= ci.policy[1]);
        counts[1] += usize::from(
            ci.incremental[0] <= binary_truth - 1.0 && binary_truth - 1.0 <= ci.incremental[1],
        );
        let uplift = evaluate_uplift_bins(&outcomes, &assignment, &[0.5], &vec![0; n], 1).unwrap();
        let uplift_ci = uplift[0].interval_95.expect("one held-out randomized bin of 300 subjects");
        counts[4] += usize::from(uplift_ci[0] <= 2.0 && 2.0 <= uplift_ci[1]);

        let multi_actions = (0..n).map(|_| (uniform() * 3.0).floor() as usize).collect::<Vec<_>>();
        let multi_assignment = (0..n)
            .map(|_| {
                let u = uniform();
                if u < 0.5 {
                    0
                } else if u < 0.8 {
                    1
                } else {
                    2
                }
            })
            .collect::<Vec<_>>();
        let multi_outcomes = multi_assignment
            .iter()
            .map(|&a| [1.0, 2.0, 4.0][a] + (uniform() - 0.5) * 2.0)
            .collect::<Vec<_>>();
        let multi = antecedent_core::MultiActionPolicyInputs {
            action_labels: Arc::from([
                Arc::from("control"),
                Arc::from("small"),
                Arc::from("large"),
            ]),
            assignment: Arc::from(multi_assignment),
            actions: Arc::from(multi_actions.clone()),
            reference: Arc::from(multi_reference.clone()),
            propensities: Arc::from((0..n).flat_map(|_| [0.5, 0.3, 0.2]).collect::<Vec<_>>()),
            available: Arc::from(vec![true; n * 3]),
            costs: Arc::from(multi_costs),
            reference_costs: Arc::from([0.0, 0.0, 0.0]),
            capacities: Arc::from([n, n, n]),
            reference_capacities: Arc::from([n, n, n]),
            budget: Some(n as f64 * 0.2),
            reference_budget: Some(0.0),
            cate_groups: Arc::from([]),
        };
        let score = evaluate_multi_action_policy_value_scores(&multi_outcomes, &multi).unwrap();
        let pm = multi.assignment.iter().zip(multi.actions.iter()).filter(|(a, b)| a == b).count();
        let rm = multi.assignment.iter().filter(|&&a| a == 0).count();
        let ci = pointwise_intervals_95(&score, n, pm, rm).unwrap();
        counts[2] += usize::from(ci.policy[0] <= multi_truth && multi_truth <= ci.policy[1]);
        counts[3] += usize::from(
            ci.incremental[0] <= multi_truth - 1.0 && multi_truth - 1.0 <= ci.incremental[1],
        );
    }
    let mcse = (0.95_f64 * 0.05 / f64::from(simulations)).sqrt();
    for (index, hits) in counts.into_iter().enumerate() {
        let coverage = hits as f64 / f64::from(simulations);
        assert!((coverage - 0.95).abs() <= 3.0 * mcse, "cell {index} coverage {coverage}");
    }
}

#[test]
fn crossfit_aipw_policy_intervals_cover_known_values() {
    // Independently randomized evaluation subjects with five-fold
    // cross-fitted outcome nuisances: each fold's mu0/mu1 predictions are
    // the control and treated sample means computed on the other four
    // folds, so no row uses its own outcome. Cross-fitting removes
    // own-observation bias and the independent-row influence-function
    // variance stays valid, so the pointwise 95% coverage holds.
    const N: usize = 400;
    const FOLDS: usize = 5;
    let simulations = 2_000;
    let mut state = 0x51ED_270B_6A11_C3D7_u64;
    let mut uniform = || crate::splitmix::splitmix64_unit(&mut state);
    let cost = 0.15;
    let truth_policy = 1.0 + 2.0 * (1.0 / 3.0) - cost * (1.0 / 3.0);
    let truth_incremental = truth_policy - 1.0;
    let reference = vec![false; N];
    let mut policy_covered = 0usize;
    let mut incremental_covered = 0usize;
    for _ in 0..simulations {
        let actions = (0..N).map(|_| uniform() < 1.0 / 3.0).collect::<Vec<_>>();
        let assignment = (0..N).map(|_| uniform() < 0.5).collect::<Vec<_>>();
        let outcomes = assignment
            .iter()
            .map(|&a| 1.0 + 2.0 * f64::from(a) + (uniform() - 0.5) * 2.0)
            .collect::<Vec<_>>();
        let fold = (0..N).map(|i| i % FOLDS).collect::<Vec<_>>();
        let mut mu0 = vec![0.0; N];
        let mut mu1 = vec![0.0; N];
        for held in 0..FOLDS {
            let (mut treated_sum, mut treated_n, mut control_sum, mut control_n) =
                (0.0, 0usize, 0.0, 0usize);
            for i in 0..N {
                if fold[i] == held {
                    continue;
                }
                if assignment[i] {
                    treated_sum += outcomes[i];
                    treated_n += 1;
                } else {
                    control_sum += outcomes[i];
                    control_n += 1;
                }
            }
            let treated_mean = treated_sum / treated_n as f64;
            let control_mean = control_sum / control_n as f64;
            for i in 0..N {
                if fold[i] == held {
                    mu1[i] = treated_mean;
                    mu0[i] = control_mean;
                }
            }
        }
        let score = evaluate_policy_value_scores(
            &outcomes,
            &BinaryPolicyDesign {
                assignment: &assignment,
                actions: &actions,
                propensity: &[0.5],
                reference: &reference,
                costs: &[cost],
                reference_costs: &[0.0],
            },
            &mu0,
            &mu1,
        )
        .unwrap();
        let policy_matches = assignment.iter().zip(&actions).filter(|(a, b)| a == b).count();
        let reference_matches = assignment.iter().filter(|&&a| !a).count();
        let interval =
            pointwise_intervals_95(&score, N, policy_matches, reference_matches).unwrap();
        policy_covered +=
            usize::from(interval.policy[0] <= truth_policy && truth_policy <= interval.policy[1]);
        incremental_covered += usize::from(
            interval.incremental[0] <= truth_incremental
                && truth_incremental <= interval.incremental[1],
        );
    }
    for (label, hits) in [("policy", policy_covered), ("incremental", incremental_covered)] {
        let coverage = hits as f64 / f64::from(simulations);
        eprintln!(
            "crossfit AIPW {label} known-truth coverage: {hits}/{simulations} = {coverage:.4}"
        );
        assert!((0.93..=0.985).contains(&coverage), "crossfit {label} coverage {coverage}");
    }
}

#[test]
fn ranked_uplift_recovers_known_bin_contrasts_and_refuses_singletons() {
    let bins = evaluate_uplift_bins(
        &[9.0, 5.0, 9.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        &[true, false, true, false, true, false, true, false],
        &[0.5],
        &[0, 0, 0, 0, 1, 1, 1, 1],
        2,
    )
    .unwrap();
    assert!((bins[0].effect - 4.0).abs() < 1e-12);
    assert!(bins[1].effect.abs() < 1e-12);
    assert!(bins.iter().all(|bin| bin.standard_error.is_finite()));
    assert!(evaluate_uplift_bins(&[1.0, 2.0], &[false, true], &[0.5], &[0, 1], 2).is_err());
}

#[test]
fn finite_fixed_class_regret_simultaneous_interval_covers_oracle_gap() {
    const DRAWS: usize = 2_000;
    const N: usize = 400;
    // The even-subgroup policy is the class oracle. Its net gain is 0.9,
    // versus 0.3 for selected treat-all, so the finite-class regret truth
    // is 0.6. This is the target the simultaneous interval must cover.
    const ORACLE_REGRET_TRUTH: f64 = 0.6;
    let candidates = vec![
        vec![false; N],
        vec![true; N],
        (0..N).map(|i| i % 2 == 0).collect(),
        (0..N).map(|i| i % 2 == 1).collect(),
    ];
    let mut state = 0x6d3a_9f12_98e7_b40d_u64;
    let mut uniform = || crate::splitmix::splitmix64_unit(&mut state);
    for p in [0.2, 0.5] {
        let mut covered = 0;
        for _ in 0..DRAWS {
            let assignment = (0..N).map(|_| uniform() < p).collect::<Vec<_>>();
            let outcome = assignment
                .iter()
                .enumerate()
                .map(|(i, &treated)| {
                    1.0 + f64::from(i % 2 == 1) * 0.5
                        + if treated { if i % 2 == 0 { 2.0 } else { -1.0 } } else { 0.0 }
                        + 2.0 * (uniform() - 0.5)
                })
                .collect::<Vec<_>>();
            let estimate = evaluate_fixed_candidate_regret(
                &outcome,
                &assignment,
                &[p],
                &candidates,
                1,
                &[0.2],
                true,
            )
            .unwrap();
            covered += usize::from(
                estimate.interval_95[0] <= ORACLE_REGRET_TRUTH
                    && ORACLE_REGRET_TRUTH <= estimate.interval_95[1],
            );
            assert!(estimate.interval_95[0] >= 0.0);
            assert!(estimate.interval_95[0] <= estimate.interval_95[1]);
        }
        let coverage = covered as f64 / DRAWS as f64;
        println!(
            "finite-class regret p={p}: {covered}/{DRAWS} simultaneous intervals cover {ORACLE_REGRET_TRUTH}"
        );
        assert!(
            (0.94..=1.0).contains(&coverage),
            "p={p} simultaneous regret coverage {covered}/{DRAWS}"
        );
    }
}

#[test]
fn finite_fixed_class_regret_refuses_unowned_selection_and_sparse_support() {
    let n = 400;
    let outcome = vec![1.0; n];
    let assignments = (0..n).map(|i| i % 2 == 0).collect::<Vec<_>>();
    let candidates = vec![vec![false; n], vec![true; n]];
    let evaluate = |assignment: &[bool], p: f64, held_out| {
        evaluate_fixed_candidate_regret(
            &outcome,
            assignment,
            &[p],
            &candidates,
            0,
            &[0.0],
            held_out,
        )
    };
    assert!(evaluate(&assignments, 0.5, false).is_err());
    assert!(evaluate(&assignments, 0.1, true).is_err());
    assert!(evaluate(&vec![true; n], 0.5, true).is_err());
    assert!(
        evaluate_fixed_candidate_regret(
            &outcome,
            &assignments,
            &[0.5],
            &candidates,
            2,
            &[0.0],
            true,
        )
        .is_err()
    );
}

// Exact known truth over every assignment: each row has fixed potential outcomes and an
// independent known assignment law, so averaging an estimator over every assignment, weighted
// by its probability, is its exact expectation. A row score that depends only on the row's own
// assignment has expectation equal to that row's true net value, and its independent-row
// variance estimate has expectation equal to the estimator's variance plus the variation of
// the per-row truths.

const ENUM_Y0: [f64; 10] = [1.0, 2.5, 0.0, 3.0, 1.5, 4.0, 2.0, 0.5, 3.5, 1.0];
const ENUM_TAU: [f64; 10] = [2.0, -1.0, 0.5, 1.5, 0.0, 3.0, -0.5, 2.5, 1.0, -2.0];
const ENUM_P: [f64; 10] = [0.5, 0.3, 0.7, 0.4, 0.6, 0.5, 0.35, 0.65, 0.45, 0.55];
const ENUM_ACTIONS: [bool; 10] = [true, false, true, true, false, false, true, false, true, false];
const ENUM_REFERENCE: [bool; 10] =
    [false, false, true, false, false, true, false, false, false, true];
const ENUM_COST: f64 = 0.25;
const ENUM_REFERENCE_COST: f64 = 0.1;

fn assert_close(got: f64, want: f64, what: &str) {
    assert!((got - want).abs() <= 1e-10 * (1.0 + want.abs()), "{what}: got {got}, want {want}");
}

fn mean_of(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

// Expected independent-row variance estimate minus the estimator's variance.
fn truth_variation(truth: &[f64]) -> f64 {
    let n = truth.len() as f64;
    let center = mean_of(truth);
    truth.iter().map(|t| (t - center).powi(2)).sum::<f64>() / (n * (n - 1.0))
}

fn enum_potential(i: usize, treated: bool) -> f64 {
    ENUM_Y0[i] + if treated { ENUM_TAU[i] } else { 0.0 }
}

fn enum_assignment(mask: usize) -> (Vec<bool>, f64) {
    let assignment = (0..ENUM_Y0.len()).map(|i| (mask >> i) & 1 == 1).collect::<Vec<_>>();
    let weight = assignment
        .iter()
        .zip(ENUM_P)
        .map(|(&treated, p)| if treated { p } else { 1.0 - p })
        .product::<f64>();
    (assignment, weight)
}

fn enum_net_truth(recommended: &[bool], cost: f64) -> Vec<f64> {
    (0..ENUM_Y0.len())
        .map(|i| enum_potential(i, recommended[i]) - cost * f64::from(recommended[i]))
        .collect()
}

// Policy, reference and incremental scores (IPW, or AIPW with frozen predictions that do
// not depend on the evaluation assignment) and the frozen-rank uplift bins, over all 2^10
// assignments of ten rows with row-aligned known probabilities.
fn check_binary_policy_enumeration(nuisance: Option<(&[f64], &[f64])>) {
    let n = ENUM_Y0.len();
    let bins = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
    let costs = [ENUM_COST];
    let reference_costs = [ENUM_REFERENCE_COST];
    let policy_truth = enum_net_truth(&ENUM_ACTIONS, ENUM_COST);
    let reference_truth = enum_net_truth(&ENUM_REFERENCE, ENUM_REFERENCE_COST);
    let incremental_truth =
        policy_truth.iter().zip(&reference_truth).map(|(a, b)| a - b).collect::<Vec<_>>();
    let bin_truth = |bin: usize| -> Vec<f64> {
        (0..n).filter(|&i| bins[i] == bin).map(|i| ENUM_TAU[i]).collect()
    };
    let truths = [policy_truth, reference_truth, incremental_truth, bin_truth(0), bin_truth(1)];
    // Per quantity: E[estimate], E[estimate^2], E[standard error^2].
    let mut moments = [[0.0_f64; 3]; 5];
    let mut total = 0.0;
    for mask in 0_usize..(1 << n) {
        let (assignment, weight) = enum_assignment(mask);
        let outcome = (0..n).map(|i| enum_potential(i, assignment[i])).collect::<Vec<_>>();
        let design = BinaryPolicyDesign {
            assignment: &assignment,
            actions: &ENUM_ACTIONS,
            propensity: &ENUM_P,
            reference: &ENUM_REFERENCE,
            costs: &costs,
            reference_costs: &reference_costs,
        };
        let scores = match nuisance {
            Some((mu0, mu1)) => evaluate_policy_value_scores(&outcome, &design, mu0, mu1),
            None => evaluate_policy_value_ipw_scores(&outcome, &design),
        }
        .unwrap();
        let uplift = evaluate_uplift_bins(&outcome, &assignment, &ENUM_P, &bins, 2).unwrap();
        let draws = [
            (scores.policy_value, scores.policy_standard_error),
            (scores.reference_value, scores.reference_standard_error),
            (scores.incremental_value, scores.incremental_standard_error),
            (uplift[0].effect, uplift[0].standard_error),
            (uplift[1].effect, uplift[1].standard_error),
        ];
        for (moment, (estimate, standard_error)) in moments.iter_mut().zip(draws) {
            moment[0] += weight * estimate;
            moment[1] += weight * estimate * estimate;
            moment[2] += weight * standard_error * standard_error;
        }
        total += weight;
    }
    assert_close(total, 1.0, "assignment law");
    let labels =
        ["policy value", "reference value", "incremental value", "uplift bin 0", "uplift bin 1"];
    for ((moment, truth), label) in moments.iter().zip(&truths).zip(labels) {
        assert_close(moment[0], mean_of(truth), &format!("{label} is unbiased"));
        assert_close(
            moment[2],
            moment[1] - moment[0] * moment[0] + truth_variation(truth),
            &format!("{label} variance estimate"),
        );
    }
}

#[test]
fn binary_ipw_policy_scores_and_frozen_rank_uplift_bins_are_unbiased_over_every_assignment() {
    check_binary_policy_enumeration(None);
}

#[test]
fn held_out_aipw_policy_scores_and_uplift_bins_are_unbiased_over_every_assignment() {
    // Frozen predictions trained on other subjects, deliberately wrong: the augmented scores
    // stay exactly unbiased because the predictions do not depend on the evaluation draw.
    let mu0 = [0.5, 3.0, -1.0, 2.0, 1.0, 5.0, 0.0, 1.5, 2.0, 2.5];
    let mu1 = [4.0, 1.0, 0.0, 6.0, 1.0, 6.5, 3.0, 2.0, 3.0, 0.0];
    check_binary_policy_enumeration(Some((mu0.as_slice(), mu1.as_slice())));
}

#[test]
fn crossfit_aipw_policy_scores_are_unbiased_over_every_assignment() {
    // Five folds of two rows. Each row's predictions are the other folds' observed arm means
    // (zero when that arm is empty there), so they depend on the other rows' draws but never
    // on the row's own assignment or outcome. Conditional on the other rows, each augmented
    // score then has expectation equal to the row's true net value, so the cross-fitted
    // estimates are exactly unbiased over every assignment.
    const FOLDS: usize = 5;
    let n = ENUM_Y0.len();
    let costs = [ENUM_COST];
    let reference_costs = [ENUM_REFERENCE_COST];
    let policy_truth = mean_of(&enum_net_truth(&ENUM_ACTIONS, ENUM_COST));
    let reference_truth = mean_of(&enum_net_truth(&ENUM_REFERENCE, ENUM_REFERENCE_COST));
    let mut expected = [0.0_f64; 3];
    let mut total = 0.0;
    for mask in 0_usize..(1 << n) {
        let (assignment, weight) = enum_assignment(mask);
        let outcome = (0..n).map(|i| enum_potential(i, assignment[i])).collect::<Vec<_>>();
        let out_of_fold_mean = |i: usize, arm: bool| {
            let rows = (0..n)
                .filter(|&j| j % FOLDS != i % FOLDS && assignment[j] == arm)
                .map(|j| outcome[j])
                .collect::<Vec<_>>();
            if rows.is_empty() { 0.0 } else { mean_of(&rows) }
        };
        let mu0 = (0..n).map(|i| out_of_fold_mean(i, false)).collect::<Vec<_>>();
        let mu1 = (0..n).map(|i| out_of_fold_mean(i, true)).collect::<Vec<_>>();
        let scores = evaluate_policy_value_scores(
            &outcome,
            &BinaryPolicyDesign {
                assignment: &assignment,
                actions: &ENUM_ACTIONS,
                propensity: &ENUM_P,
                reference: &ENUM_REFERENCE,
                costs: &costs,
                reference_costs: &reference_costs,
            },
            &mu0,
            &mu1,
        )
        .unwrap();
        expected[0] += weight * scores.policy_value;
        expected[1] += weight * scores.reference_value;
        expected[2] += weight * scores.incremental_value;
        total += weight;
    }
    assert_close(total, 1.0, "assignment law");
    assert_close(expected[0], policy_truth, "cross-fitted policy value is unbiased");
    assert_close(expected[1], reference_truth, "cross-fitted reference value is unbiased");
    assert_close(
        expected[2],
        policy_truth - reference_truth,
        "cross-fitted incremental value is unbiased",
    );
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "one exact enumeration over the policy, reference and every group contrast"
)]
fn multi_action_policy_scores_and_group_contrasts_are_unbiased_over_every_assignment() {
    use std::sync::Arc;
    // Rows 0..3 (group g0) and 3..6 (group g1) are enumerated over all 3^6 action draws.
    // Rows 6..12 hold one row per group fixed at each action; their potential outcomes are
    // zero under every action, so their scores are the same whatever they draw and fixing
    // them only guarantees that every group observes every action.
    const PROBABILITIES: [f64; 3] = [0.5, 0.3, 0.2];
    let potential: [[f64; 3]; 12] = [
        [1.0, 3.0, 2.0],
        [2.0, 2.5, 4.0],
        [0.5, 1.0, 3.5],
        [3.0, 1.0, 5.0],
        [1.5, 4.0, 0.0],
        [2.5, 2.0, 3.0],
        [0.0; 3],
        [0.0; 3],
        [0.0; 3],
        [0.0; 3],
        [0.0; 3],
        [0.0; 3],
    ];
    let n = potential.len();
    let group_of = |i: usize| usize::from((3..6).contains(&i) || i >= 9);
    let fixed = [0, 1, 2, 0, 1, 2];
    let actions = [1, 2, 0, 2, 1, 1, 0, 1, 2, 2, 0, 1];
    let reference = [0, 0, 1, 0, 0, 2, 0, 0, 0, 1, 0, 0];
    let costs = [0.0, 0.2, 0.5];
    let reference_costs = [0.0, 0.1, 0.3];
    let base = antecedent_core::MultiActionPolicyInputs {
        action_labels: Arc::from([Arc::from("control"), Arc::from("A"), Arc::from("B")]),
        assignment: Arc::from(vec![0; n]),
        actions: Arc::from(actions),
        reference: Arc::from(reference),
        propensities: Arc::from((0..n).flat_map(|_| PROBABILITIES).collect::<Vec<_>>()),
        available: Arc::from(vec![true; n * 3]),
        costs: Arc::from(costs),
        reference_costs: Arc::from(reference_costs),
        capacities: Arc::from([n; 3]),
        reference_capacities: Arc::from([n; 3]),
        budget: None,
        reference_budget: None,
        cate_groups: Arc::from(
            (0..n).map(|i| Arc::<str>::from(["g0", "g1"][group_of(i)])).collect::<Vec<_>>(),
        ),
    };
    let policy_truth =
        (0..n).map(|i| potential[i][actions[i]] - costs[actions[i]]).collect::<Vec<_>>();
    let reference_truth = (0..n)
        .map(|i| potential[i][reference[i]] - reference_costs[reference[i]])
        .collect::<Vec<_>>();
    let incremental_truth =
        policy_truth.iter().zip(&reference_truth).map(|(a, b)| a - b).collect::<Vec<_>>();
    let contrast_truth = |group: usize, action: usize| -> Vec<f64> {
        (0..n)
            .filter(|&i| group_of(i) == group)
            .map(|i| potential[i][action] - potential[i][0])
            .collect()
    };
    let truths = [
        policy_truth,
        reference_truth,
        incremental_truth,
        contrast_truth(0, 1),
        contrast_truth(0, 2),
        contrast_truth(1, 1),
        contrast_truth(1, 2),
    ];
    // Per quantity: E[estimate], E[estimate^2], E[standard error^2].
    let mut moments = [[0.0_f64; 3]; 7];
    let mut total = 0.0;
    for code in 0_usize..729 {
        let mut assignment = vec![0_usize; n];
        let mut rest = code;
        for slot in assignment.iter_mut().take(6) {
            *slot = rest % 3;
            rest /= 3;
        }
        assignment[6..].copy_from_slice(&fixed);
        let weight = assignment[..6].iter().map(|&a| PROBABILITIES[a]).product::<f64>();
        let outcome = (0..n).map(|i| potential[i][assignment[i]]).collect::<Vec<_>>();
        let mut policy = base.clone();
        policy.assignment = Arc::from(assignment);
        let scores = evaluate_multi_action_policy_value_scores(&outcome, &policy).unwrap();
        let contrasts = evaluate_multi_action_cate(&outcome, &policy).unwrap();
        assert_eq!(contrasts.len(), 4);
        assert_eq!((contrasts[2].group.as_str(), contrasts[2].action.as_str()), ("g1", "A"));
        let draws = [
            (scores.policy_value, scores.policy_standard_error),
            (scores.reference_value, scores.reference_standard_error),
            (scores.incremental_value, scores.incremental_standard_error),
            (contrasts[0].effect, contrasts[0].standard_error),
            (contrasts[1].effect, contrasts[1].standard_error),
            (contrasts[2].effect, contrasts[2].standard_error),
            (contrasts[3].effect, contrasts[3].standard_error),
        ];
        for (moment, (estimate, standard_error)) in moments.iter_mut().zip(draws) {
            moment[0] += weight * estimate;
            moment[1] += weight * estimate * estimate;
            moment[2] += weight * standard_error * standard_error;
        }
        total += weight;
    }
    assert_close(total, 1.0, "assignment law");
    let labels = [
        "policy value",
        "reference value",
        "incremental value",
        "g0 A minus control",
        "g0 B minus control",
        "g1 A minus control",
        "g1 B minus control",
    ];
    for ((moment, truth), label) in moments.iter().zip(&truths).zip(labels) {
        assert_close(moment[0], mean_of(truth), &format!("{label} is unbiased"));
        assert_close(
            moment[2],
            moment[1] - moment[0] * moment[0] + truth_variation(truth),
            &format!("{label} variance estimate"),
        );
    }
}

#[test]
fn finite_class_candidate_values_are_unbiased_and_regret_is_their_oracle_gap() {
    // The first ten of 400 rows are enumerated over all 2^10 assignments. The other rows have
    // zero potential outcomes and a fixed alternating assignment, so their scores are the
    // same whatever they draw; they only supply each candidate's matching assignments.
    const N: usize = 400;
    const SELECTED: usize = 1;
    let m = ENUM_Y0.len();
    let cost = 0.2;
    let propensity = (0..N).map(|i| if i < m { ENUM_P[i] } else { 0.5 }).collect::<Vec<_>>();
    let candidates =
        vec![vec![false; N], vec![true; N], (0..N).map(|i| i % 2 == 0).collect::<Vec<_>>()];
    let potential = |i: usize, treated: bool| if i < m { enum_potential(i, treated) } else { 0.0 };
    let row_truth =
        |j: usize, i: usize| potential(i, candidates[j][i]) - cost * f64::from(candidates[j][i]);
    let values = (0..candidates.len())
        .map(|j| (0..N).map(|i| row_truth(j, i)).sum::<f64>() / N as f64)
        .collect::<Vec<_>>();
    let mut value_means = vec![0.0; candidates.len()];
    // Per candidate contrast against the selected one: E[contrast], E[contrast^2], E[se^2].
    let mut contrast_moments = vec![[0.0_f64; 3]; candidates.len()];
    let mut regret_mean = 0.0;
    let mut total = 0.0;
    for mask in 0_usize..(1 << m) {
        let (drawn, weight) = enum_assignment(mask);
        let assignment =
            (0..N).map(|i| if i < m { drawn[i] } else { i % 2 == 0 }).collect::<Vec<_>>();
        let outcome = (0..N).map(|i| potential(i, assignment[i])).collect::<Vec<_>>();
        let estimate = evaluate_fixed_candidate_regret(
            &outcome,
            &assignment,
            &propensity,
            &candidates,
            SELECTED,
            &[cost],
            true,
        )
        .unwrap();
        let best = estimate.candidate_values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert_close(
            estimate.regret,
            best - estimate.candidate_values[SELECTED],
            "plug-in regret is the best candidate minus the selected one",
        );
        for j in 0..candidates.len() {
            let contrast = estimate.candidate_values[j] - estimate.candidate_values[SELECTED];
            value_means[j] += weight * estimate.candidate_values[j];
            contrast_moments[j][0] += weight * contrast;
            contrast_moments[j][1] += weight * contrast * contrast;
            contrast_moments[j][2] += weight * estimate.contrast_standard_errors[j].powi(2);
        }
        regret_mean += weight * estimate.regret;
        total += weight;
    }
    assert_close(total, 1.0, "assignment law");
    for j in 0..candidates.len() {
        assert_close(value_means[j], values[j], &format!("candidate {j} value is unbiased"));
        let paired = (0..N).map(|i| row_truth(j, i) - row_truth(SELECTED, i)).collect::<Vec<_>>();
        let moment = contrast_moments[j];
        assert_close(moment[0], values[j] - values[SELECTED], &format!("contrast {j} is unbiased"));
        assert_close(
            moment[2],
            moment[1] - moment[0] * moment[0] + truth_variation(&paired),
            &format!("contrast {j} variance estimate"),
        );
    }
    // The class oracle gap is the regret target; the plug-in maximum can only overshoot it
    // in expectation.
    let oracle_gap = values.iter().copied().fold(f64::NEG_INFINITY, f64::max) - values[SELECTED];
    assert!(oracle_gap > 0.0, "the selected candidate is not the class oracle");
    assert!(
        regret_mean >= oracle_gap - 1e-12,
        "E[regret] {regret_mean} below the gap {oracle_gap}"
    );
}
