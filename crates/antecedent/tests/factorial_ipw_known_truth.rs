//! Known-truth checks of the independent-factor 2x2 contrasts and the net-value policy
//! estimators, through the compiled routes their declared estimators select.
//!
//! * A hand-worked 2x2 with every cell probability one quarter: the route returns the cell
//!   means, contrasts and Young bounds computed by hand, exactly (every number is dyadic).
//! * The full independent-factor assignment law of five units with unequal factor
//!   probabilities and fixed potential outcomes: over all 1,024 assignments, weighted by their
//!   probabilities, the Horvitz-Thompson contrasts average to the true contrasts, each mean bound
//!   dominates its contrast's design variance, and wherever every cell is observed the kernel the
//!   route calls equals the textbook formula written here; it refuses exactly the assignments
//!   that leave a cell empty.
//! * The binary and multi-action net-value routes, run on every assignment of a small
//!   randomized evaluation, average to the true net policy, reference and incremental values.
// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(clippy::float_cmp, reason = "dyadic fixtures are compared exactly")]

use std::sync::Arc;

use antecedent::prelude::ExecutionContext;
use antecedent::{
    EstimatorId, PolicyValueEstimate, RandomizedEffectEstimate, RandomizedEffectQuery, Study,
};
use antecedent_core::{
    CausalQuery, MultiActionPolicyInputs, PolicyValueQuery, RandomizationDesign, VariableId,
};
use antecedent_data::TabularData;

fn labels(prefix: &str, n: usize) -> Vec<Arc<str>> {
    (0..n).map(|i| Arc::<str>::from(format!("{prefix}-{i}"))).collect()
}

/// The `count` base-`base` digits of `code`, least significant first: one assignment of the
/// enumerated law.
fn digits(mut code: usize, base: usize, count: usize) -> Vec<usize> {
    (0..count)
        .map(|_| {
            let digit = code % base;
            code /= base;
            digit
        })
        .collect()
}

fn close(got: f64, want: f64, what: &str) {
    assert!((got - want).abs() <= 1e-12 * want.abs().max(1.0), "{what}: got {got}, want {want}");
}

/// Factor-A, factor-B and interaction weights over the cells `(0,0)`, `(0,1)`, `(1,0)`, `(1,1)`.
const WEIGHTS: [[f64; 4]; 3] =
    [[-0.5, -0.5, 0.5, 0.5], [-0.5, 0.5, -0.5, 0.5], [1.0, -1.0, -1.0, 1.0]];

fn run_factorial(y: &[f64], cells: &[usize], rows: &[Vec<f64>]) -> RandomizedEffectEstimate {
    let n = y.len();
    let query = RandomizedEffectQuery::with_design(
        RandomizationDesign::MultiArm {
            assignment: Arc::from(cells.to_vec()),
            probabilities: rows.to_vec().into(),
            arms: ["a0b0", "a0b1", "a1b0", "a1b1"].map(Arc::<str>::from).into(),
        },
        VariableId::from_raw(0),
        cells.iter().map(|cell| *cell != 0).collect::<Vec<_>>(),
        (0..n).map(|i| rows[i][cells[i]]).collect::<Vec<_>>(),
        labels("unit", n),
        labels("row", n),
        ("a0b0", "a0b1"),
    );
    let data = TabularData::from_f64_columns([("outcome", y)]).unwrap();
    Study::tabular(data)
        .query(query)
        .estimator(EstimatorId::RandomizedFactorialBernoulliHt)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(5))
        .unwrap()
        .into_randomized_effect()
        .unwrap()
}

#[test]
fn factorial_route_matches_the_hand_computed_cells_and_contrasts() {
    // Two rows per cell, both factors at one half, so every cell probability is 1/4 and each
    // observed row contributes 4y to its cell sum. Cell sums 16, 16, 16, 48 over n = 8 give
    // means 2, 2, 2, 6; the main effects are 2 and 2 and the interaction 4. Each cell variance
    // is (1/64)(3/4)(16) sum y^2: 15/8, 3/2, 3 and 111/8, so both main-effect bounds are their
    // sum 81/4 and the interaction bound is 81.
    let cells = [0, 0, 1, 1, 2, 2, 3, 3];
    let y = [1.0, 3.0, 2.0, 2.0, 4.0, 0.0, 5.0, 7.0];
    let rows = vec![vec![0.25; 4]; 8];
    let fit = run_factorial(&y, &cells, &rows);
    let means: Vec<f64> = fit.multi_arm_values.iter().map(|(_, mean, _, _)| *mean).collect();
    let variances: Vec<f64> =
        fit.multi_arm_values.iter().map(|(_, _, variance, _)| *variance).collect();
    assert_eq!(means, [2.0, 2.0, 2.0, 6.0]);
    assert_eq!(variances, [1.875, 1.5, 3.0, 13.875]);
    assert_eq!(fit.effect, 2.0);
    assert_eq!(fit.second_factor_effect, Some(2.0));
    assert_eq!(fit.factorial_interaction, Some(4.0));
    assert_eq!(fit.variance_upper_bound, 20.25);
    assert_eq!(fit.second_factor_variance, Some(20.25));
    assert_eq!(fit.factorial_interaction_variance, Some(81.0));
    assert_eq!((fit.control_units, fit.treatment_units), (4, 4));
    assert!(fit.interval_95.is_none(), "two rows per cell publish no interval");
}

#[test]
fn factorial_contrasts_are_unbiased_over_the_full_independent_factor_law() {
    // Five units, each with its own factor probabilities and four potential outcomes.
    let pa = [0.5, 0.25, 0.75, 0.5, 0.375];
    let pb = [0.5, 0.5, 0.25, 0.625, 0.75];
    let potential: [[f64; 4]; 5] = [
        [1.0, 2.0, 3.5, 5.0],
        [0.5, 1.5, 2.0, 4.0],
        [2.0, 2.5, 3.0, 6.5],
        [1.5, 1.0, 4.0, 3.0],
        [3.0, 4.5, 2.5, 7.0],
    ];
    let n = pa.len();
    let cell_probability = |unit: usize, cell: usize| {
        let qa = if cell / 2 == 1 { pa[unit] } else { 1.0 - pa[unit] };
        let qb = if cell % 2 == 1 { pb[unit] } else { 1.0 - pb[unit] };
        qa * qb
    };
    let truth: Vec<f64> = WEIGHTS
        .iter()
        .map(|weights| {
            (0..4)
                .map(|cell| {
                    weights[cell] * potential.iter().map(|row| row[cell]).sum::<f64>() / n as f64
                })
                .sum()
        })
        .collect();
    let (mut first, mut second, mut mean_bound) = ([0.0; 3], [0.0; 3], [0.0; 3]);
    let mut total_probability = 0.0;
    for code in 0..4_usize.pow(5) {
        let cells = digits(code, 4, n);
        let probability: f64 = (0..n).map(|unit| cell_probability(unit, cells[unit])).product();
        total_probability += probability;
        // Textbook Horvitz-Thompson cell means and diagonal variance estimates.
        let mut sums = [0.0; 4];
        let mut terms = [0.0; 4];
        let mut seen = [false; 4];
        let y: Vec<f64> = (0..n).map(|unit| potential[unit][cells[unit]]).collect();
        let q: Vec<f64> = (0..n).map(|unit| cell_probability(unit, cells[unit])).collect();
        for unit in 0..n {
            let cell = cells[unit];
            seen[cell] = true;
            sums[cell] += y[unit] / q[unit];
            terms[cell] += (1.0 - q[unit]) * (y[unit] / q[unit]).powi(2);
        }
        let means = sums.map(|sum| sum / n as f64);
        let variances = terms.map(|term| term / (n * n) as f64);
        for (k, weights) in WEIGHTS.iter().enumerate() {
            let estimate: f64 = (0..4).map(|cell| weights[cell] * means[cell]).sum();
            let bound: f64 =
                4.0 * (0..4).map(|cell| weights[cell].powi(2) * variances[cell]).sum::<f64>();
            first[k] += probability * estimate;
            second[k] += probability * estimate * estimate;
            mean_bound[k] += probability * bound;
        }
        // The kernel the compiled route calls equals the formula wherever every cell is seen,
        // and refuses exactly the assignments that leave a cell empty.
        let kernel =
            antecedent_estimate::factorial_bernoulli::estimate_factorial_2x2_cells(&y, &cells, &q);
        if seen.iter().all(|seen| *seen) {
            let fit = kernel.expect("every cell observed");
            for cell in 0..4 {
                close(fit.contrasts.cell_means[cell], means[cell], "cell mean");
                close(fit.cell_variances[cell], variances[cell], "cell variance");
            }
        } else {
            assert!(kernel.is_err(), "an empty cell is refused");
        }
    }
    close(total_probability, 1.0, "the assignment law sums to one");
    for k in 0..3 {
        close(first[k], truth[k], "unbiased contrast");
        let variance = second[k] - first[k] * first[k];
        assert!(
            mean_bound[k] + 1e-12 >= variance,
            "contrast {k}: mean Young bound {} below design variance {variance}",
            mean_bound[k]
        );
    }
}

fn policy_query(n: usize) -> PolicyValueQuery {
    PolicyValueQuery {
        outcome: VariableId::from_raw(0),
        assignment: Arc::from(vec![false; n]),
        propensity: Arc::from([0.5]),
        actions: Arc::from(vec![false; n]),
        reference: Arc::from(vec![false; n]),
        mu0: Arc::from([]),
        mu1: Arc::from([]),
        costs: Arc::from([0.0]),
        reference_costs: Arc::from([0.0]),
        evaluation_subject_ids: labels("eval", n).into(),
        disjoint_training_subjects: false,
        crossfit_fold_ownership_valid: false,
        global_constraints_present: false,
        multi_action: None,
        uplift_bins: Arc::from([]),
        uplift_bin_count: 0,
        uplift_training_subject_ids: Arc::from([]),
        regret: None,
    }
}

fn run_policy(
    outcome: &[f64],
    query: PolicyValueQuery,
    estimator: EstimatorId,
) -> PolicyValueEstimate {
    let data = TabularData::from_f64_columns([("outcome", outcome)]).unwrap();
    let context = ExecutionContext::for_tests(7);
    Study::tabular(data)
        .query(CausalQuery::PolicyValue(query))
        .estimator(estimator)
        .build()
        .unwrap()
        .run(&context)
        .unwrap()
        .into_policy_value()
        .unwrap()
}

#[test]
fn binary_net_value_route_is_unbiased_over_every_bernoulli_assignment() {
    // Four units, potential outcomes (Y(0), Y(1)), known probability one half, a fixed policy
    // and reference, and a treatment cost of one quarter for both.
    let potential = [(1.0, 2.0), (3.0, 3.5), (0.5, 2.5), (2.0, 1.0)];
    let actions = [true, false, true, true];
    let reference = [false, false, true, false];
    let cost = 0.25;
    let n = potential.len();
    let net = |policy: &[bool]| {
        (0..n)
            .map(|i| {
                let (untreated, treated) = potential[i];
                if policy[i] { treated - cost } else { untreated }
            })
            .sum::<f64>()
            / n as f64
    };
    let (truth_policy, truth_reference) = (net(&actions), net(&reference));
    let (mut policy_mean, mut reference_mean, mut incremental_mean) = (0.0, 0.0, 0.0);
    for code in 0..(1_usize << n) {
        let assignment: Vec<bool> = digits(code, 2, n).into_iter().map(|bit| bit == 1).collect();
        let outcome: Vec<f64> =
            (0..n).map(|i| if assignment[i] { potential[i].1 } else { potential[i].0 }).collect();
        let mut query = policy_query(n);
        query.assignment = assignment.into();
        query.actions = actions.into();
        query.reference = reference.into();
        query.costs = Arc::from([cost]);
        query.reference_costs = Arc::from([cost]);
        let fit = run_policy(&outcome, query, EstimatorId::RandomizedIpwPolicyNet);
        let weight = 1.0 / (1_usize << n) as f64;
        policy_mean += weight * fit.policy_value;
        reference_mean += weight * fit.reference_value;
        incremental_mean += weight * fit.incremental_value;
    }
    close(policy_mean, truth_policy, "policy value");
    close(reference_mean, truth_reference, "reference value");
    close(incremental_mean, truth_policy - truth_reference, "incremental value");
}

#[test]
fn multi_action_net_value_route_is_unbiased_over_every_assignment() {
    // Three units, three actions with known probabilities (1/2, 1/4, 1/4), per-action costs.
    let potential = [[1.0, 2.0, 3.0], [2.5, 1.5, 4.0], [0.5, 3.5, 2.0]];
    let probabilities = [0.5, 0.25, 0.25];
    let actions = [1_usize, 0, 2];
    let reference = [0_usize, 0, 1];
    let costs = [0.0, 0.25, 0.5];
    let n = potential.len();
    let k = probabilities.len();
    let net = |policy: &[usize]| {
        (0..n).map(|i| potential[i][policy[i]] - costs[policy[i]]).sum::<f64>() / n as f64
    };
    let (truth_policy, truth_reference) = (net(&actions), net(&reference));
    let (mut policy_mean, mut reference_mean, mut incremental_mean) = (0.0, 0.0, 0.0);
    for code in 0..k * k * k {
        let assigned = digits(code, k, n);
        let weight: f64 = assigned.iter().map(|&action| probabilities[action]).product();
        let outcome: Vec<f64> = (0..n).map(|i| potential[i][assigned[i]]).collect();
        let mut query = policy_query(n);
        query.assignment = Vec::new().into();
        query.propensity = Vec::new().into();
        query.actions = Vec::new().into();
        query.reference = Vec::new().into();
        query.costs = Vec::new().into();
        query.reference_costs = Vec::new().into();
        query.multi_action = Some(MultiActionPolicyInputs {
            action_labels: ["control", "A", "B"].map(Arc::<str>::from).into(),
            assignment: assigned.into(),
            actions: actions.into(),
            reference: reference.into(),
            propensities: (0..n).flat_map(|_| probabilities).collect::<Vec<_>>().into(),
            available: vec![true; n * k].into(),
            costs: costs.into(),
            reference_costs: costs.into(),
            capacities: [n; 3].into(),
            reference_capacities: [n; 3].into(),
            budget: None,
            reference_budget: None,
            cate_groups: Arc::from([]),
        });
        let fit = run_policy(&outcome, query, EstimatorId::RandomizedMultiActionIpwPolicyNet);
        policy_mean += weight * fit.policy_value;
        reference_mean += weight * fit.reference_value;
        incremental_mean += weight * fit.incremental_value;
    }
    close(policy_mean, truth_policy, "policy value");
    close(reference_mean, truth_reference, "reference value");
    close(incremental_mean, truth_policy - truth_reference, "incremental value");
}
