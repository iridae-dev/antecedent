//! Known-truth check of the graph-imputation composite against an independent closed form.
//!
//! Every graph the root-treatment support retains identifies the treatment-to-outcome contrast
//! with the empty adjustment set, so each child fits the outcome on the design `[1, T]` with
//! the conjugate Gaussian model `beta | sigma2 ~ N(0, sigma2 * scale^2 * I)`. The posterior mean
//! of the treatment coefficient is then the ridge solution
//! `(X'X + I / scale^2)^-1 X'y`, whatever the graph weights are, and a covariate that no
//! adjustment set reads (here with missing cells the imputation must fill) cannot move it. The
//! expected value below is computed from the raw rows by solving those two normal equations
//! directly; nothing in it is read from the production path.
//!
//! Tolerance. The reported point is the mean of posterior draws, so it carries Monte-Carlo
//! error `sd / sqrt(imputations * draws)`, where `sd` is the posterior standard deviation of the
//! treatment coefficient (at most about 0.1 on this design with the prior scale used). With
//! 2 imputations of 20 000 draws that error is about 5e-4, and the tolerance of 5e-3 is ten
//! times it. The prior pulls the coefficient from the least-squares value 2 to about 1.6, so
//! the tolerance separates a correctly applied prior from an ignored or mis-scaled one by a
//! factor of about eighty.

use std::sync::Arc;

use antecedent::analysis::graph_imputation::{
    CompiledGraphImputation, GraphImputationSpec, GraphImputationSupport,
};
use antecedent_core::{ExecutionContext, VariableId};
use antecedent_data::{ImputationSpec, TabularData};

const ROWS: usize = 40;
const PRIOR_SCALE: f64 = 0.5;
const POSTERIOR_DRAWS: u32 = 20_000;
const TOLERANCE: f64 = 5e-3;

/// Balanced arms (20 and 20) with outcome `1 + 2 T + e`; the noise sums to zero within each arm,
/// so the arm means are exactly 1 and 3 and the least-squares effect is exactly 2.
fn rows() -> (Vec<f64>, Vec<f64>) {
    const NOISE: [f64; 10] = [0.1, -0.1, 0.05, -0.05, 0.2, -0.2, 0.15, -0.15, 0.0, 0.0];
    let mut treatment = Vec::with_capacity(ROWS);
    let mut outcome = Vec::with_capacity(ROWS);
    for row in 0..ROWS {
        let treated = f64::from(row % 2 == 1);
        // Row pairs (2k, 2k + 1) share a noise slot rotated per arm, which keeps each arm's
        // noise a full period of the zero-sum pattern.
        let slot = (row / 2 + usize::from(treated == 1.0) * 3) % NOISE.len();
        treatment.push(treated);
        outcome.push(1.0 + 2.0 * treated + NOISE[slot]);
    }
    (treatment, outcome)
}

/// The ridge posterior mean of the treatment coefficient, from the normal equations of the
/// design `[1, T]` with prior precision `1 / scale^2` on both coefficients.
fn ridge_treatment_effect(treatment: &[f64], outcome: &[f64], scale: f64) -> f64 {
    let lambda = 1.0 / (scale * scale);
    let n = treatment.len() as f64;
    let (mut sum_t, mut sum_y, mut sum_ty) = (0.0, 0.0, 0.0);
    for (t, y) in treatment.iter().zip(outcome) {
        sum_t += t;
        sum_y += y;
        sum_ty += t * y;
    }
    // [n + lambda, sum_t; sum_t, sum_t + lambda] beta = [sum_y, sum_ty], by Cramer's rule
    // (T is 0/1, so sum(T^2) = sum(T)).
    let (a, b, d) = (n + lambda, sum_t, sum_t + lambda);
    let determinant = a * d - b * b;
    (a * sum_ty - b * sum_y) / determinant
}

fn spec(support: GraphImputationSupport, variables: usize) -> GraphImputationSpec {
    GraphImputationSpec {
        treatment: VariableId::from_raw(0),
        outcome: VariableId::from_raw(1),
        support,
        imputation: ImputationSpec::new().with_imputations(2).with_iterations(2).with_seed(17),
        domains: Arc::from(&[1u8, 0, 0][..variables]),
        posterior_draws: POSTERIOR_DRAWS,
        prior_scale: PRIOR_SCALE,
        execution_seed: 19,
    }
}

fn check(point: &antecedent::analysis::graph_imputation::GraphImputationPoint, truth: f64) {
    assert!(
        (point.mean - truth).abs() < TOLERANCE,
        "mean {} differs from the closed form {truth} by more than {TOLERANCE}",
        point.mean
    );
    assert!(point.unidentified_by_table.iter().all(|mass| mass.abs() < f64::EPSILON));
    // The interval is the 2.5% and 97.5% nearest-rank quantiles of the same pooled draws, so it
    // brackets both the reported mean and the closed form, and its calibration is read, not
    // gated.
    let (lower, upper) = point.interval().unwrap();
    assert!(lower < point.mean && point.mean < upper, "[{lower}, {upper}] vs {}", point.mean);
    assert!(lower < truth && truth < upper, "[{lower}, {upper}] vs {truth}");
    point.calibration().unwrap();
}

#[test]
fn a_two_variable_design_has_the_ridge_posterior_mean_of_the_treatment_coefficient() {
    let (treatment, outcome) = rows();
    let truth = ridge_treatment_effect(&treatment, &outcome, PRIOR_SCALE);
    // The prior is far from negligible: the least-squares effect is exactly 2.
    assert!((truth - 2.0).abs() > 0.3, "{truth}");
    let data = TabularData::from_f64_columns([("t", &treatment[..]), ("y", &outcome[..])]).unwrap();
    let ctx = ExecutionContext::for_tests(19);
    let plan = CompiledGraphImputation::compile(
        data,
        &spec(GraphImputationSupport::TreatmentRootDirectEffect, 2),
        &ctx,
    )
    .unwrap();
    check(&plan.execute(&ctx).unwrap(), truth);
}

#[test]
fn a_covariate_with_imputed_cells_cannot_move_a_root_treatment_mixture() {
    let (treatment, outcome) = rows();
    let truth = ridge_treatment_effect(&treatment, &outcome, PRIOR_SCALE);
    // A covariate unrelated to the design, with every fourth cell missing so the chained
    // equations must fill it. Treatment is a root in every retained graph, so the identifying
    // adjustment set is empty in all six and the mixture mean equals the two-variable answer
    // whatever weight each graph receives.
    let covariate: Vec<f64> = (0..ROWS)
        .map(|row| if row % 4 == 3 { f64::NAN } else { ((row * 7) % 11) as f64 / 11.0 - 0.5 })
        .collect();
    let data = TabularData::from_f64_columns([
        ("t", &treatment[..]),
        ("y", &outcome[..]),
        ("x", &covariate[..]),
    ])
    .unwrap();
    let ctx = ExecutionContext::for_tests(19);
    let plan = CompiledGraphImputation::compile(
        data,
        &spec(GraphImputationSupport::TreatmentRootDirectEffect, 3),
        &ctx,
    )
    .unwrap();
    check(&plan.execute(&ctx).unwrap(), truth);
}
