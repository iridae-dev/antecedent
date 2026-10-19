//! Observational policy value against truths computed from the generating process.
//!
//! The generating process has one normal confounder `z` that drives both the treatment and
//! the outcome: `E[Y | A = 0, z] = 1 + z` and a treatment effect `1.5 z`, so the best binary
//! policy treats when `z > 0` and has the closed-form value `1 + 1.5 phi(0)` where `phi(0)`
//! is the standard normal density at zero (`E[z; z > 0] = phi(0)`). A three-action process
//! with `E[Y | A = a, z] = (0, 1 + z / 2, 1 / 2 - z)` gives a second closed form. The tests
//! check that a prespecified policy, a policy learned on held-out rows and a cross-fitted
//! learned policy recover those values, that the naive follower mean does not, that a policy
//! scored on the rows it was learned from is refused, that selection inflates an in-sample
//! plug-in where cross-fitting does not, and that every refusal carries its own detail.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    reason = "test fixtures compare exact level codes and small counts"
)]

use std::sync::Arc;

use antecedent_core::{CausalRng, ExecutionContext};
use antecedent_estimate::EstimationError;
use antecedent_estimate::observational_policy::{
    KnownPropensity, ObservationalPolicyValue, PolicyData, PolicyEvaluation, PolicyObjective,
    PolicyOrigin, PolicyValueEstimate,
};
use antecedent_kernels::standard_normal;

#[path = "common/observational_effects.rs"]
mod dgp;

use dgp::{BinaryStudy as Binary, PHI_0, binary_study, offset_binary_study};

fn estimate(
    data: &Binary,
    request: &ObservationalPolicyValue,
    seed: u64,
) -> Result<PolicyValueEstimate, EstimationError> {
    let mut columns: Vec<&[f64]> = vec![&data.z];
    columns.extend(data.noise.iter().map(Vec::as_slice));
    request.estimate(
        &PolicyData { outcome: &data.outcome, treatment: &data.treatment, covariates: &columns },
        &ExecutionContext::for_tests(seed),
    )
}

fn binary_request(evaluation: PolicyEvaluation) -> ObservationalPolicyValue {
    ObservationalPolicyValue::new(vec![0.0, 1.0], PolicyObjective::Maximize, evaluation)
}

fn threshold_policy(data: &Binary) -> Arc<[f64]> {
    data.z.iter().map(|&z| f64::from(u8::from(z > 0.0))).collect()
}

fn within(value: f64, se: f64, truth: f64, multiples: f64) -> bool {
    (value - truth).abs() <= multiples * se
}

/// `E[Y | A = pi(z)]` for the policy "treat when z > 0" under `expit(steepness z + offset)`
/// assignment, by quadrature over the normal density: the mean outcome of the rows that
/// happened to follow the policy, which is what a naive analysis reports.
fn follower_mean(steepness: f64, offset: f64) -> f64 {
    let (mut numerator, mut denominator) = (0.0, 0.0);
    let steps = 40_000;
    for k in 0..=steps {
        let z = -10.0 + 20.0 * f64::from(k) / f64::from(steps);
        let treat_probability = 1.0 / (1.0 + (-(steepness * z + offset)).exp());
        let follows = if z > 0.0 { treat_probability } else { 1.0 - treat_probability };
        let value = if z > 0.0 { 1.0 + 2.5 * z } else { 1.0 + z };
        let density = dgp::phi(z);
        numerator += value * follows * density;
        denominator += follows * density;
    }
    numerator / denominator
}

#[test]
fn a_prespecified_policy_recovers_its_closed_form_value_where_the_follower_mean_does_not() {
    // Assignment expit(0.6 z + 1) favours treatment, so the rows that followed "treat when
    // z > 0" over-represent z > 0 and their mean outcome is biased upward by an amount the
    // quadrature above states.
    let (steepness, offset) = (0.6, 1.0);
    let data = offset_binary_study(6000, 21, 1.5, 0, steepness, offset);
    let truth = 1.0 + 1.5 * PHI_0;
    let policy = threshold_policy(&data);
    let request = binary_request(PolicyEvaluation::Prespecified {
        actions: Arc::clone(&policy),
        origin: PolicyOrigin::Prespecified,
    });
    let result = estimate(&data, &request, 3).unwrap();
    assert!(
        within(result.value, result.se, truth, 4.0),
        "{} vs {truth} (se {})",
        result.value,
        result.se
    );
    // Never and always treating: E[1 + z] = 1 and 1 + 1.5 E[z] = 1.
    let (never, always) = (&result.action_values[0], &result.action_values[1]);
    assert!(within(never.value, never.se, 1.0, 4.0), "{never:?}");
    assert!(within(always.value, always.se, 1.0, 4.0), "{always:?}");
    assert_eq!((never.level, always.level), (0.0, 1.0));
    let followers: Vec<f64> = (0..data.z.len())
        .filter(|&i| data.treatment[i] == policy[i])
        .map(|i| data.outcome[i])
        .collect();
    let naive = followers.iter().sum::<f64>() / followers.len() as f64;
    let expected_naive = follower_mean(steepness, offset);
    assert!(expected_naive - truth > 0.1, "analytic naive bias {}", expected_naive - truth);
    assert!((naive - expected_naive).abs() < 0.06, "naive {naive} analytic {expected_naive}");
    assert!((naive - truth).abs() > 0.1, "naive {naive} truth {truth}");
    assert_eq!(result.evaluation_rows, 6000);
}

#[test]
fn the_scores_replay_the_value_and_the_standard_error() {
    let data = binary_study(2000, 33, 1.5, 0, 1.0);
    let result = estimate(
        &data,
        &binary_request(PolicyEvaluation::Prespecified {
            actions: threshold_policy(&data),
            origin: PolicyOrigin::Prespecified,
        }),
        11,
    )
    .unwrap();
    let scores = &result.scores;
    let m = scores.len() as f64;
    let mean = scores.iter().sum::<f64>() / m;
    let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / (m - 1.0);
    assert!((mean - result.value).abs() < 1e-12);
    assert!((variance.sqrt() / m.sqrt() - result.se).abs() < 1e-12);
    assert_eq!(result.row_index.len(), scores.len());
    assert_eq!(result.policy_actions.len(), scores.len());
    // Seeded: the same seed reproduces the estimate exactly.
    let again = estimate(
        &data,
        &binary_request(PolicyEvaluation::Prespecified {
            actions: threshold_policy(&data),
            origin: PolicyOrigin::Prespecified,
        }),
        11,
    )
    .unwrap();
    assert_eq!(result, again);
}

#[test]
fn the_reference_comparison_is_the_paired_difference() {
    let data = binary_study(5000, 22, 1.5, 0, 1.0);
    let mut request = binary_request(PolicyEvaluation::Prespecified {
        actions: threshold_policy(&data),
        origin: PolicyOrigin::Prespecified,
    });
    request.reference = Some(0.0);
    let result = estimate(&data, &request, 4).unwrap();
    let reference = result.reference.unwrap();
    assert_eq!(reference.level, 0.0);
    assert!(within(reference.incremental_value, reference.incremental_se, 1.5 * PHI_0, 4.0));
    let gain = result.value - result.action_values[0].value;
    assert!((reference.incremental_value - gain).abs() < 1e-12);
}

#[test]
fn a_policy_learned_on_held_out_rows_is_scored_on_independent_rows() {
    let data = binary_study(8000, 23, 1.5, 0, 1.0);
    let request = binary_request(PolicyEvaluation::HeldOut { evaluation_fraction: 0.4 });
    let result = estimate(&data, &request, 5).unwrap();
    let optimum = 1.0 + 1.5 * PHI_0;
    // The learned threshold is within sampling error of zero; its regret is second order.
    assert!(within(result.value, result.se, optimum, 4.0) || (result.value - optimum).abs() < 0.03);
    assert!(result.value > 1.2, "learned policy value {}", result.value);
    // Each level holds out the ceiling of 40 percent of its rows.
    assert!((3200..=3202).contains(&result.evaluation_rows), "{}", result.evaluation_rows);
    assert_eq!(result.evaluation_rows + result.learning_rows, 8000);
    let evaluated: std::collections::HashSet<_> = result.row_index.iter().copied().collect();
    assert_eq!(evaluated.len() as u64, result.evaluation_rows);
    // The split is seeded: the same seed holds out the same rows, another does not.
    let again = estimate(&data, &request, 5).unwrap();
    assert_eq!(result.row_index, again.row_index);
    let other = estimate(&data, &request, 6).unwrap();
    assert_ne!(result.row_index, other.row_index);
    // Each treatment level is held out in proportion.
    let held_treated = result
        .row_index
        .iter()
        .filter(|i| data.treatment[usize::try_from(i.get()).expect("row")] == 1.0)
        .count() as f64;
    let all_treated = data.treatment.iter().sum::<f64>();
    assert!((held_treated / all_treated - 0.4).abs() < 0.01);
}

#[test]
fn cross_fitted_learning_scores_every_row_with_a_policy_that_never_saw_it() {
    let data = binary_study(6000, 24, 1.5, 0, 1.0);
    let result = estimate(&data, &binary_request(PolicyEvaluation::CrossFitted), 7).unwrap();
    let optimum = 1.0 + 1.5 * PHI_0;
    assert!(within(result.value, result.se, optimum, 4.0) || (result.value - optimum).abs() < 0.03);
    assert_eq!(result.evaluation_rows, 6000);
    assert!(result.learning_rows < 6000 && result.learning_rows >= 4700);
    assert_eq!(result.policy_actions.len(), 6000);
}

/// Least-squares fit of `y` on `[1, x...]` by normal equations (Gaussian elimination).
fn ols(rows: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
    let p = rows[0].len() + 1;
    let mut a = vec![vec![0.0; p + 1]; p];
    for (row, &target) in rows.iter().zip(y) {
        let x: Vec<f64> = std::iter::once(1.0).chain(row.iter().copied()).collect();
        for i in 0..p {
            for j in 0..p {
                a[i][j] += x[i] * x[j];
            }
            a[i][p] += x[i] * target;
        }
    }
    for col in 0..p {
        let pivot = (col..p).max_by(|&r, &s| a[r][col].abs().total_cmp(&a[s][col].abs())).unwrap();
        a.swap(col, pivot);
        for r in 0..p {
            if r != col {
                let factor = a[r][col] / a[col][col];
                for c in col..=p {
                    a[r][c] -= factor * a[col][c];
                }
            }
        }
    }
    (0..p).map(|i| a[i][p] / a[i][i]).collect()
}

/// The in-sample plug-in value of the greedy policy: fit each action's outcome model on
/// all rows, choose the best predicted action per row and average its prediction.
fn in_sample_plug_in(data: &Binary) -> f64 {
    let rows: Vec<Vec<f64>> = (0..data.z.len())
        .map(|i| std::iter::once(data.z[i]).chain(data.noise.iter().map(|c| c[i])).collect())
        .collect();
    let fit = |arm: f64| {
        let idx: Vec<usize> = (0..rows.len()).filter(|&i| data.treatment[i] == arm).collect();
        ols(
            &idx.iter().map(|&i| rows[i].clone()).collect::<Vec<_>>(),
            &idx.iter().map(|&i| data.outcome[i]).collect::<Vec<_>>(),
        )
    };
    let (b0, b1) = (fit(0.0), fit(1.0));
    let predict =
        |b: &[f64], x: &[f64]| b[0] + x.iter().zip(&b[1..]).map(|(a, c)| a * c).sum::<f64>();
    rows.iter().map(|x| predict(&b0, x).max(predict(&b1, x))).sum::<f64>() / rows.len() as f64
}

#[test]
fn selection_inflates_an_in_sample_value_where_cross_fitted_learning_does_not() {
    // No treatment effect at all: every policy has value E[Y] = 1 (E[1 + z] = 1), but with 30
    // covariates (one real, 29 noise) on 300 rows the greedy policy finds apparent gains
    // in-sample. Expected optimism: the fitted difference of the two arms' predictions at a row
    // is N(0, sigma^2 (h0 + h1)) with leverages h_a about p / n_a = 2p / n, and the greedy
    // value gains E[max(a, b)] - E[(a + b) / 2] = E|a - b| / 2 = sqrt(2 v / pi) / 2, i.e.
    // 0.5 * sqrt(2 / pi) * 0.5 * sqrt(4 p / n) = 0.13 for p = 31, n = 300, sigma = 0.5.
    let replicates = 30;
    let mut plug_in = Vec::new();
    let mut honest = Vec::new();
    for k in 0..replicates {
        // Randomized assignment with the declared probability 1/2, so the selection effect
        // comes from policy learning, not from a propensity model fitted to 30 covariates.
        let data = binary_study(300, 900 + k, 0.0, 29, 0.0);
        plug_in.push(in_sample_plug_in(&data) - 1.0);
        let mut request = binary_request(PolicyEvaluation::CrossFitted);
        request.known_propensity = Some(KnownPropensity::Constant(Arc::from([0.5, 0.5])));
        let result = estimate(&data, &request, k).unwrap();
        honest.push(result.value - 1.0);
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let sd = |v: &[f64]| {
        let m = mean(v);
        (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
    };
    assert!(mean(&plug_in) > 0.09, "in-sample plug-in bias {}", mean(&plug_in));
    assert!(
        mean(&honest).abs() < 3.0 * sd(&honest) / (replicates as f64).sqrt(),
        "cross-fitted bias {} (sd {})",
        mean(&honest),
        sd(&honest)
    );
}

#[test]
fn known_propensities_replace_the_fitted_model_and_are_validated() {
    // Randomized assignment: the declared 1/2 and the fitted model agree on the value.
    let data = binary_study(4000, 41, 1.5, 0, 0.0);
    let mut request = binary_request(PolicyEvaluation::Prespecified {
        actions: threshold_policy(&data),
        origin: PolicyOrigin::Prespecified,
    });
    request.known_propensity = Some(KnownPropensity::Constant(Arc::from([0.5, 0.5])));
    let known = estimate(&data, &request, 2).unwrap();
    let truth = 1.0 + 1.5 * PHI_0;
    assert!(within(known.value, known.se, truth, 4.0), "{} vs {truth}", known.value);
    assert!(known.action_values.iter().all(|a| (a.propensity_min - 0.5).abs() < 1e-12));
    // Per-row columns carry the same probabilities.
    let columns: Arc<[Arc<[f64]>]> =
        Arc::from([Arc::<[f64]>::from(vec![0.5; 4000]), Arc::<[f64]>::from(vec![0.5; 4000])]);
    request.known_propensity = Some(KnownPropensity::PerRow(columns));
    let per_row = estimate(&data, &request, 2).unwrap();
    assert_eq!(per_row.value, known.value);
    // Probabilities that do not sum to one, or fall below the floor, are refused.
    request.known_propensity = Some(KnownPropensity::Constant(Arc::from([0.5, 0.6])));
    let text = estimate(&data, &request, 2).unwrap_err().to_string();
    assert!(text.contains("policy_value.invalid_known_propensity"), "{text}");
    request.known_propensity = Some(KnownPropensity::Constant(Arc::from([0.995, 0.005])));
    let text = estimate(&data, &request, 2).unwrap_err().to_string();
    assert!(text.contains("policy_value.action_propensity_below_floor"), "{text}");
}

#[test]
fn a_policy_declared_as_learned_on_the_scoring_rows_is_refused() {
    let data = binary_study(1000, 25, 1.5, 0, 1.0);
    let request = binary_request(PolicyEvaluation::Prespecified {
        actions: threshold_policy(&data),
        origin: PolicyOrigin::LearnedOnEvaluationRows,
    });
    let error = estimate(&data, &request, 1).unwrap_err();
    let text = error.to_string();
    assert!(text.contains("policy_value.policy_learned_on_evaluation_rows"), "{text}");
    assert!(text.contains("held_out") && text.contains("cross_fitted_learning"), "{text}");
}

#[test]
fn a_policy_action_outside_the_declared_support_names_the_action() {
    let data = binary_study(1000, 26, 1.5, 0, 1.0);
    let mut actions = threshold_policy(&data).to_vec();
    actions[17] = 5.0;
    let request = binary_request(PolicyEvaluation::Prespecified {
        actions: Arc::from(actions),
        origin: PolicyOrigin::Prespecified,
    });
    let text = estimate(&data, &request, 1).unwrap_err().to_string();
    assert!(text.contains("policy_value.action_outside_support"), "{text}");
    assert!(text.contains('5'), "{text}");
}

#[test]
fn an_observed_treatment_outside_the_support_and_an_unpopulated_action_are_refused() {
    let mut data = binary_study(1000, 27, 1.5, 0, 1.0);
    data.treatment[3] = 2.0;
    let request = binary_request(PolicyEvaluation::CrossFitted);
    let text = estimate(&data, &request, 1).unwrap_err().to_string();
    assert!(text.contains("policy_value.undeclared_level"), "{text}");

    let data = binary_study(1000, 27, 1.5, 0, 1.0);
    let wide = ObservationalPolicyValue::new(
        vec![0.0, 1.0, 2.0],
        PolicyObjective::Maximize,
        PolicyEvaluation::CrossFitted,
    );
    let text = estimate(&data, &wide, 1).unwrap_err().to_string();
    assert!(text.contains("policy_value.action_not_populated"), "{text}");
}

#[test]
fn an_action_with_propensity_below_the_floor_is_a_positivity_refusal() {
    // Treatment is nearly deterministic in z, so one action has a vanishing propensity.
    let data = binary_study(3000, 28, 1.5, 0, 14.0);
    let request = binary_request(PolicyEvaluation::Prespecified {
        actions: threshold_policy(&data),
        origin: PolicyOrigin::Prespecified,
    });
    let text = estimate(&data, &request, 1).unwrap_err().to_string();
    assert!(text.contains("policy_value.action_propensity_below_floor"), "{text}");
}

#[test]
fn malformed_declarations_are_refused_with_their_own_details() {
    let data = binary_study(500, 29, 1.5, 0, 1.0);
    let mut request = binary_request(PolicyEvaluation::CrossFitted);
    request.reference = Some(9.0);
    assert!(
        estimate(&data, &request, 1)
            .unwrap_err()
            .to_string()
            .contains("policy_value.reference_outside_support")
    );
    let request = binary_request(PolicyEvaluation::HeldOut { evaluation_fraction: 1.5 });
    assert!(
        estimate(&data, &request, 1)
            .unwrap_err()
            .to_string()
            .contains("policy_value.invalid_evaluation_fraction")
    );
    let mut request = binary_request(PolicyEvaluation::CrossFitted);
    request.folds = 1;
    assert!(
        estimate(&data, &request, 1)
            .unwrap_err()
            .to_string()
            .contains("policy_value.invalid_folds")
    );
    let request = ObservationalPolicyValue::new(
        vec![0.0, 0.0],
        PolicyObjective::Maximize,
        PolicyEvaluation::CrossFitted,
    );
    assert!(
        estimate(&data, &request, 1)
            .unwrap_err()
            .to_string()
            .contains("policy_value.invalid_support")
    );
    let short = Arc::<[f64]>::from(vec![0.0; 10]);
    let request = binary_request(PolicyEvaluation::Prespecified {
        actions: short,
        origin: PolicyOrigin::Prespecified,
    });
    assert!(
        estimate(&data, &request, 1).unwrap_err().to_string().contains("policy_value.invalid_data")
    );
}

struct MultiAction {
    outcome: Vec<f64>,
    treatment: Vec<f64>,
    z: Vec<f64>,
}

fn multi_action_study(n: usize, seed: u64) -> MultiAction {
    let mut rng = CausalRng::from_seed(seed);
    let (mut outcome, mut treatment, mut zs) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..n {
        let z = standard_normal(&mut rng);
        let logits = [0.0, 0.5 * z, -0.5 * z];
        let top = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = logits.iter().map(|l| (l - top).exp()).collect();
        let total: f64 = weights.iter().sum();
        let draw = rng.next_f64() * total;
        let action = if draw < weights[0] {
            0
        } else if draw < weights[0] + weights[1] {
            1
        } else {
            2
        };
        let mean = [0.0, 1.0 + 0.5 * z, 0.5 - z][action];
        outcome.push(mean + 0.5 * standard_normal(&mut rng));
        treatment.push(action as f64);
        zs.push(z);
    }
    MultiAction { outcome, treatment, z: zs }
}

#[test]
fn a_three_action_policy_recovers_its_closed_form_value() {
    // Action 2 where z < 0, action 1 where z > 0:
    // E[(1/2 - z); z < 0] + E[1 + z/2; z > 0] = 1/4 + phi(0) + 1/2 + phi(0)/2.
    let truth = 0.75 + 1.5 * PHI_0;
    let data = multi_action_study(9000, 31);
    let actions: Arc<[f64]> = data.z.iter().map(|&z| if z < 0.0 { 2.0 } else { 1.0 }).collect();
    let mut request = ObservationalPolicyValue::new(
        vec![0.0, 1.0, 2.0],
        PolicyObjective::Maximize,
        PolicyEvaluation::Prespecified { actions, origin: PolicyOrigin::Prespecified },
    );
    request.reference = Some(0.0);
    let result = request
        .estimate(
            &PolicyData {
                outcome: &data.outcome,
                treatment: &data.treatment,
                covariates: &[&data.z],
            },
            &ExecutionContext::for_tests(8),
        )
        .unwrap();
    assert!(within(result.value, result.se, truth, 4.0), "{} vs {truth}", result.value);
    // Always action 0 has value 0, so the incremental value over it is the policy value.
    let reference = result.reference.unwrap();
    assert!(within(reference.incremental_value, reference.incremental_se, truth, 4.0));
    let always = |level: usize| result.action_values[level].value;
    assert!((always(0)).abs() < 0.1, "never-treat value {}", always(0));
    // E[1 + z/2] = 1 and E[1/2 - z] = 1/2.
    assert!((always(1) - 1.0).abs() < 0.1 && (always(2) - 0.5).abs() < 0.1);
    let shares: f64 = result.action_values.iter().map(|a| a.policy_share).sum();
    assert!((shares - 1.0).abs() < 1e-12);
    // A learned policy over three actions beats every single action.
    let learned = ObservationalPolicyValue::new(
        vec![0.0, 1.0, 2.0],
        PolicyObjective::Maximize,
        PolicyEvaluation::CrossFitted,
    )
    .estimate(
        &PolicyData { outcome: &data.outcome, treatment: &data.treatment, covariates: &[&data.z] },
        &ExecutionContext::for_tests(9),
    )
    .unwrap();
    assert!(learned.value > 1.0);
}

#[test]
fn minimizing_picks_the_action_with_the_smallest_predicted_outcome() {
    let data = multi_action_study(6000, 32);
    let learned = ObservationalPolicyValue::new(
        vec![0.0, 1.0, 2.0],
        PolicyObjective::Minimize,
        PolicyEvaluation::CrossFitted,
    )
    .estimate(
        &PolicyData { outcome: &data.outcome, treatment: &data.treatment, covariates: &[&data.z] },
        &ExecutionContext::for_tests(10),
    )
    .unwrap();
    // Action 0 has mean 0 everywhere; action 2 is lower for z > 1/2, action 1 for z < -2.
    // The minimizer's value is E[min(0, 1/2 - z, 1 + z/2)], well below the maximizer's.
    assert!(learned.value < 0.0);
    assert!(learned.action_values[0].policy_share > 0.4);
}
