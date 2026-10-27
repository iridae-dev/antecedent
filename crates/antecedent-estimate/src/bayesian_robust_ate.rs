//! Modular Bayesian-bootstrap cross-fitted orthogonal ATE.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{ExecutionContext, StreamDomain};

use crate::EstimationError;
use crate::crossfit_plan::{CrossfitPlan, FoldIdErrors, FoldSource};
use crate::util::dot;

/// Key of the seeded fold shuffle. The route carries no seed of its own and the fold
/// identity is retained and re-verified on refresh, so the plan is a function of the stable
/// row ids and the treatment arm alone; this constant only names the shuffle stream.
const FOLD_PLAN_SEED: u64 = 0xB007_F01D_0000_0001;

const FOLD_ERRORS: FoldIdErrors = FoldIdErrors {
    length: "invalid Bayesian robust ATE input or options",
    units: "invalid Bayesian robust ATE input or options",
    range: "invalid Bayesian robust ATE input or options",
};

/// Complete row-aligned input for the robust Bayesian ATE route.
#[derive(Clone, Debug, PartialEq)]
pub struct BayesianRobustAteInput {
    /// Unique stable identity for each observed row.
    pub row_ids: Vec<u64>,
    /// Binary treatment assignment.
    pub treatment: Vec<bool>,
    /// Numeric outcome.
    pub outcome: Vec<f64>,
    /// Raw baseline covariate columns, each length `n`.
    pub covariates: Vec<Vec<f64>>,
}

/// Fixed cross-fit and weighted nuisance settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BayesianRobustAteOptions {
    /// Number of posterior Bayesian-bootstrap row-weight draws.
    pub draws: usize,
    /// Number of cross-fit folds.
    pub folds: usize,
    /// Fixed ridge penalty on non-intercept coefficients in both nuisances.
    pub ridge_penalty: f64,
    /// Propensity clamp used in the orthogonal score.
    pub propensity_clip: f64,
    /// Central posterior interval coverage, e.g. 0.95.
    pub coverage: f64,
}

impl Default for BayesianRobustAteOptions {
    fn default() -> Self {
        Self { draws: 500, folds: 5, ridge_penalty: 1.0, propensity_clip: 0.01, coverage: 0.95 }
    }
}

/// Role/fold identity used to audit one weighted nuisance refit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BayesianRobustRefit {
    /// Nuisance role: `propensity`, `outcome_control`, or `outcome_treated`.
    pub role: &'static str,
    /// Held-out fold index.
    pub fold: usize,
    /// Fingerprint of the shared draw-specific Bayesian-bootstrap row weights.
    pub weight_fingerprint: u64,
}

/// Modular bootstrap pushforward for the cross-fitted orthogonal AIPW score.
#[derive(Clone, Debug, PartialEq)]
pub struct BayesianRobustAteResult {
    /// Posterior mean of the weighted orthogonal-score draws.
    pub estimate: f64,
    /// Equal-tailed posterior interval from the returned draws.
    pub interval: (f64, f64),
    /// Posterior standard deviation.
    pub standard_deviation: f64,
    /// Modular bootstrap pushforward draws, one per shared row-weight vector.
    pub draws: Vec<f64>,
    /// Stable row IDs in input order.
    pub row_ids: Vec<u64>,
    /// Fixed fold identity in input order.
    pub fold_ids: Vec<u16>,
    /// Nuisance implementation and fixed penalty.
    pub model_identity: String,
    /// Per-draw fit audit; each fold/role refit records the same weight fingerprint.
    pub refits: Vec<Vec<BayesianRobustRefit>>,
    /// Explicit interval interpretation.
    pub interval_interpretation: &'static str,
}

/// Fit a Bayesian-bootstrap modular posterior of cross-fitted orthogonal AIPW.
///
/// Every posterior draw samples one Exp(1) row-weight vector. The propensity, both
/// potential-outcome regressions, and final orthogonal-score average all use that
/// same vector. Fold membership is fixed by sorted stable row IDs within treatment
/// arm. Outcome nuisances use weighted ridge regression; the propensity nuisance
/// uses weighted ridge-logistic regression. The interval is a modular bootstrap
/// pushforward conditional on the fixed folds, linear nuisance bases, penalties,
/// and observed rows. Double robustness remains conditional on at least one nuisance
/// model being correctly specified and the usual causal assumptions; this interval
/// does not establish repeated-sampling coverage or causal identification.
///
/// # Errors
/// Invalid row layout, unsupported fold count, nonfinite inputs, or failed weighted fit.
pub fn estimate_bayesian_robust_ate(
    input: &BayesianRobustAteInput,
    options: BayesianRobustAteOptions,
    ctx: &ExecutionContext,
) -> Result<BayesianRobustAteResult, EstimationError> {
    validate(input, options)?;
    let n = input.row_ids.len();
    let plan = fold_plan(input, options.folds)?;
    let x = make_design(input);
    let _admitted = crate::util::admit_array::<f64>(ctx, options.draws, 2)?;
    let mut effects = Vec::with_capacity(options.draws);
    let mut refits = Vec::with_capacity(options.draws);
    for draw in 0..options.draws {
        if ctx.cancellation.is_cancelled() {
            return Err(EstimationError::unsupported("Bayesian robust ATE cancelled"));
        }
        let mut rng = ctx.rng.stream_for(StreamDomain::Bayesian, 0xB007_0000 | draw as u64);
        // Rescaled to mean one by the owner; the total check below sees the
        // rescaled sum (about `n`), which is non-finite or non-positive exactly
        // when the raw total was.
        let weights = antecedent_kernels::bayesian_bootstrap_weights(&mut rng, n, true);
        let total = weights.iter().sum::<f64>();
        if !total.is_finite() || total <= 0.0 {
            return Err(EstimationError::stats_msg("invalid Bayesian-bootstrap row weights"));
        }
        let fingerprint = fingerprint(&weights);
        let (estimate, audit) = fit_draw(input, &x, &plan, &weights, options, fingerprint)?;
        effects.push(estimate);
        refits.push(audit);
    }
    let estimate = crate::util::mean(&effects);
    let standard_deviation = crate::util::sample_std(&effects);
    let interval = interval(&effects, options.coverage);
    Ok(BayesianRobustAteResult {
        estimate,
        interval,
        standard_deviation,
        draws: effects,
        row_ids: input.row_ids.clone(),
        fold_ids: plan.fold_ids().iter().map(|&f| fold_u16(f)).collect::<Result<Vec<_>, _>>()?,
        model_identity: format!(
            "linear.weighted_ridge(lambda={}); logistic.weighted_ridge(lambda={})",
            options.ridge_penalty, options.ridge_penalty
        ),
        refits,
        interval_interpretation: "modular Bayesian-bootstrap pushforward; nuisance refits and orthogonal score share each draw's Exp(1) row weights",
    })
}

/// Bind the deterministic treatment-stratified cross-fit assignment for an
/// input. Checked preparation can retain this vector and verify it on refresh.
///
/// The ids come from the shared cross-fit plan: rows are keyed by the rank of their stable
/// row id, each treatment arm is shuffled with a fixed key and dealt round-robin, so every
/// fold holds both arms and the plan does not depend on the file order of the rows.
///
/// # Errors
/// Returns the same input and fold-count refusals as the estimator.
pub fn plan_bayesian_robust_ate_folds(
    input: &BayesianRobustAteInput,
    folds: usize,
) -> Result<Vec<u16>, EstimationError> {
    let options = BayesianRobustAteOptions { folds, ..BayesianRobustAteOptions::default() };
    validate(input, options)?;
    fold_plan(input, folds)?.fold_ids().iter().map(|&f| fold_u16(f)).collect()
}

fn validate(
    input: &BayesianRobustAteInput,
    opt: BayesianRobustAteOptions,
) -> Result<(), EstimationError> {
    let n = input.row_ids.len();
    if n < 4
        || input.treatment.len() != n
        || input.outcome.len() != n
        || input.covariates.iter().any(|c| c.len() != n)
        || input.row_ids.iter().copied().collect::<std::collections::HashSet<_>>().len() != n
        || input.outcome.iter().chain(input.covariates.iter().flatten()).any(|v| !v.is_finite())
        || opt.draws < 2
        || opt.folds < 2
        || opt.folds > n / 2
        || u16::try_from(opt.folds).is_err()
        || !opt.ridge_penalty.is_finite()
        || opt.ridge_penalty <= 0.0
        || !opt.propensity_clip.is_finite()
        || !(0.0..0.5).contains(&opt.propensity_clip)
        || !opt.coverage.is_finite()
        || !(0.0..1.0).contains(&opt.coverage)
        || opt.coverage == 0.0
    {
        return Err(EstimationError::data_msg("invalid Bayesian robust ATE input or options"));
    }
    if !input.treatment.iter().any(|t| *t) || !input.treatment.iter().any(|t| !*t) {
        return Err(EstimationError::unsupported_coded(
            antecedent_core::reason_code!("positivity_violated"),
            "Bayesian robust ATE requires both treatment arms",
        ));
    }
    Ok(())
}

/// The shared cross-fit plan of a validated input.
fn fold_plan(
    input: &BayesianRobustAteInput,
    folds: usize,
) -> Result<CrossfitPlan, EstimationError> {
    let n = input.row_ids.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| input.row_ids[i]);
    let mut units = Vec::with_capacity(n);
    for (rank, &i) in order.iter().enumerate() {
        units.push((
            i,
            antecedent_data::GlobalId::new(u64::try_from(rank).map_err(|_| {
                EstimationError::data_msg(
                    "address the row by GlobalId and keep ChunkIndex inside one bounded chunk. A u32 backend feature flag is not that attempt",
                )
            })?),
        ));
    }
    let mut placed = vec![antecedent_data::GlobalId::new(0); n];
    for (i, id) in units {
        placed[i] = id;
    }
    let units = placed;
    let strata: Vec<u32> = input.treatment.iter().map(|&t| u32::from(t)).collect();
    CrossfitPlan::derive(
        n,
        folds,
        FoldSource::Stratified { strata: &strata, units: &units, seed: FOLD_PLAN_SEED },
        &FOLD_ERRORS,
    )
}

/// A fold id as stored in the result; validation bounds the fold count by `u16::MAX`.
fn fold_u16(fold: u32) -> Result<u16, EstimationError> {
    u16::try_from(fold)
        .map_err(|_| EstimationError::data_msg("invalid Bayesian robust ATE input or options"))
}

fn make_design(input: &BayesianRobustAteInput) -> Vec<Vec<f64>> {
    (0..input.row_ids.len())
        .map(|i| std::iter::once(1.0).chain(input.covariates.iter().map(|c| c[i])).collect())
        .collect()
}

fn fit_draw(
    input: &BayesianRobustAteInput,
    x: &[Vec<f64>],
    plan: &CrossfitPlan,
    weights: &[f64],
    opt: BayesianRobustAteOptions,
    weight_fingerprint: u64,
) -> Result<(f64, Vec<BayesianRobustRefit>), EstimationError> {
    let n = x.len();
    let mut e = vec![0.0; n];
    let mut mu0 = vec![0.0; n];
    let mut mu1 = vec![0.0; n];
    let mut audit = Vec::with_capacity(opt.folds * 3);
    for fold in 0..plan.folds() {
        let (train, valid) = (plan.train(fold), plan.valid(fold));
        let p = fit_logistic(x, &input.treatment, weights, train, opt.ridge_penalty)?;
        let raw: Vec<f64> =
            valid.iter().map(|&i| antecedent_kernels::sigmoid(dot(&p, &x[i]))).collect();
        let guarded = crate::learn_nuisance::guarded_propensity(&raw, Some(opt.propensity_clip))?;
        for (&i, &ehat) in valid.iter().zip(&guarded) {
            e[i] = ehat;
        }
        audit.push(BayesianRobustRefit { role: "propensity", fold, weight_fingerprint });
        for arm in [false, true] {
            let arm_rows: Vec<usize> =
                train.iter().copied().filter(|&i| input.treatment[i] == arm).collect();
            let beta = fit_ridge(x, &input.outcome, weights, &arm_rows, opt.ridge_penalty)?;
            let predictions = if arm { &mut mu1 } else { &mut mu0 };
            for &i in valid {
                predictions[i] = dot(&beta, &x[i]);
            }
            audit.push(BayesianRobustRefit {
                role: if arm { "outcome_treated" } else { "outcome_control" },
                fold,
                weight_fingerprint,
            });
        }
    }
    let weight_sum = weights.iter().sum::<f64>();
    let score = (0..n)
        .map(|i| {
            let t = f64::from(input.treatment[i]);
            let psi = mu1[i] - mu0[i] + t / e[i] * (input.outcome[i] - mu1[i])
                - (1.0 - t) / (1.0 - e[i]) * (input.outcome[i] - mu0[i]);
            weights[i] * psi
        })
        .sum::<f64>()
        / weight_sum;
    if !score.is_finite() {
        return Err(EstimationError::stats_msg("nonfinite Bayesian robust ATE score"));
    }
    Ok((score, audit))
}

fn fit_ridge(
    x: &[Vec<f64>],
    y: &[f64],
    weights: &[f64],
    rows: &[usize],
    lambda: f64,
) -> Result<Vec<f64>, EstimationError> {
    if rows.len() < x.first().map_or(1, Vec::len) {
        return Err(EstimationError::data_msg(
            "weighted outcome nuisance has too few training rows",
        ));
    }
    let p = x[0].len();
    let mut gram = vec![0.0; p * p];
    let mut rhs = vec![0.0; p];
    for &i in rows {
        for j in 0..p {
            rhs[j] += weights[i] * x[i][j] * y[i];
            for k in 0..p {
                gram[j * p + k] += weights[i] * x[i][j] * x[i][k];
            }
        }
    }
    solve_penalized(&mut gram, &rhs, lambda)
}

fn fit_logistic(
    x: &[Vec<f64>],
    t: &[bool],
    weights: &[f64],
    rows: &[usize],
    lambda: f64,
) -> Result<Vec<f64>, EstimationError> {
    if !rows.iter().any(|&i| t[i]) || !rows.iter().any(|&i| !t[i]) {
        return Err(EstimationError::data_msg(
            "weighted propensity fold is missing a treatment arm",
        ));
    }
    let p = x[0].len();
    let nrows = rows.len();
    let mut design = vec![0.0; nrows * p];
    for (r, &i) in rows.iter().enumerate() {
        for j in 0..p {
            design[j * nrows + r] = x[i][j];
        }
    }
    let y: Vec<f64> = rows.iter().map(|&i| f64::from(t[i])).collect();
    let row_weights: Vec<f64> = rows.iter().map(|&i| weights[i]).collect();
    // The shared GLM engine: damped IRLS with the intercept column unpenalised, the same
    // weighted-ridge objective the outcome nuisances solve in closed form.
    crate::util::fit_logistic_ridge(&design, nrows, p, &y, Some(&row_weights), lambda)
}

/// Ridge-penalise every non-intercept diagonal entry of the weighted Gram and solve.
fn solve_penalized(
    gram: &mut [f64],
    rhs: &[f64],
    lambda: f64,
) -> Result<Vec<f64>, EstimationError> {
    let p = rhs.len();
    for j in 1..p {
        gram[j * p + j] += lambda;
    }
    solve_spd_system(gram, rhs)
}

/// Solve the symmetric positive-definite weighted nuisance system with the shared Cholesky rule.
fn solve_spd_system(a: &[f64], b: &[f64]) -> Result<Vec<f64>, EstimationError> {
    crate::util::solve_spd(a, b, b.len())
        .ok_or_else(|| EstimationError::stats_msg("singular weighted nuisance system"))
}

/// Domain tag of the draw-specific weight fingerprint.
const WEIGHT_FINGERPRINT_TAG: &str = "antecedent.estimate.bayesian_robust_ate.weights";

/// Fingerprint of one draw's row-weight vector, recorded on each refit of the draw.
fn fingerprint(values: &[f64]) -> u64 {
    let mut writer = antecedent_core::CanonicalWriter::new(WEIGHT_FINGERPRINT_TAG);
    writer.word(values.len() as u64);
    for &v in values {
        writer.real(v);
    }
    crate::util::canonical_word(&writer)
}
/// Equal-tailed posterior interval of the draws under the crate's exchangeable-rank rule
/// (rank `p·(D + 1)`, so a finite draw count covers at its level).
fn interval(draws: &[f64], coverage: f64) -> (f64, f64) {
    let mut sorted = draws.to_vec();
    sorted.sort_by(f64::total_cmp);
    antecedent_stats::equal_tail_interval_sorted(
        &sorted,
        coverage,
        antecedent_stats::QuantileRule::ExchangeableRank,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(
        n: usize,
        nonlinear_propensity: bool,
        nonlinear_outcome: bool,
    ) -> BayesianRobustAteInput {
        let x: Vec<f64> = (0..n).map(|i| ((i * 7919 % n) as f64 / n as f64) * 2.0 - 1.0).collect();
        let mut rng = ExecutionContext::for_tests(611).rng.stream_for(StreamDomain::Estimate, 91);
        let treatment: Vec<bool> = x
            .iter()
            .map(|v| {
                let z = if nonlinear_propensity { 0.25 + 1.2 * v * v } else { -0.1 + 0.8 * v };
                rng.next_f64() < antecedent_kernels::sigmoid(z)
            })
            .collect();
        let outcome = x
            .iter()
            .zip(&treatment)
            .map(|(v, t)| {
                let baseline = if nonlinear_outcome { 1.4 * v * v } else { 0.8 * v };
                baseline + if *t { 2.0 } else { 0.0 }
            })
            .collect();
        BayesianRobustAteInput {
            row_ids: (0..n).map(|i| 100_000 + (i as u64 * 37 % n as u64)).collect(),
            treatment,
            outcome,
            covariates: vec![x],
        }
    }

    fn run(nonlinear_propensity: bool, nonlinear_outcome: bool) -> BayesianRobustAteResult {
        let input = data(800, nonlinear_propensity, nonlinear_outcome);
        estimate_bayesian_robust_ate(
            &input,
            BayesianRobustAteOptions {
                draws: 80,
                folds: 4,
                ridge_penalty: 0.05,
                propensity_clip: 0.01,
                coverage: 0.95,
            },
            &ExecutionContext::for_tests(882),
        )
        .unwrap()
    }

    #[test]
    fn one_correct_nuisance_protects_each_single_misspecification_case() {
        let outcome_only = run(false, true); // propensity correct, outcome basis misspecified
        let propensity_only = run(true, false); // outcome correct, propensity basis misspecified
        assert!((outcome_only.estimate - 2.0).abs() < 0.25, "{}", outcome_only.estimate);
        assert!((propensity_only.estimate - 2.0).abs() < 0.25, "{}", propensity_only.estimate);
    }

    #[test]
    fn joint_misspecification_executes_without_a_double_robustness_claim() {
        let result = run(true, true);
        assert_eq!(result.draws.len(), 80);
        assert!(result.estimate.is_finite());
        assert!(result.interval.0.is_finite() && result.interval.1.is_finite());
    }

    #[test]
    fn every_role_and_fold_shares_weight_draw_and_fixed_row_folds() {
        let input = data(200, false, false);
        let result = estimate_bayesian_robust_ate(
            &input,
            BayesianRobustAteOptions {
                draws: 3,
                folds: 4,
                ridge_penalty: 0.05,
                propensity_clip: 0.01,
                coverage: 0.9,
            },
            &ExecutionContext::for_tests(15),
        )
        .unwrap();
        assert_eq!(result.row_ids, input.row_ids);
        assert_eq!(result.fold_ids, plan_bayesian_robust_ate_folds(&input, 4).unwrap());
        for draw in &result.refits {
            assert_eq!(draw.len(), 12);
            assert!(draw.iter().all(|fit| fit.weight_fingerprint == draw[0].weight_fingerprint));
            for fold in 0..4 {
                assert_eq!(draw.iter().filter(|fit| fit.fold == fold).count(), 3);
                assert!(draw.iter().any(|fit| fit.fold == fold && fit.role == "propensity"));
                assert!(draw.iter().any(|fit| fit.fold == fold && fit.role == "outcome_control"));
                assert!(draw.iter().any(|fit| fit.fold == fold && fit.role == "outcome_treated"));
            }
        }
    }

    #[test]
    fn the_nuisance_solve_is_scale_free_and_refuses_collinear_columns() {
        // Weighted Gram entries of order 1e-14 were refused by the former absolute 1e-12 pivot.
        let mut gram = vec![1e-14, 0.0, 0.0, 4e-14];
        let beta = solve_penalized(&mut gram, &[1e-14, 8e-14], 0.0).unwrap();
        assert!((beta[0] - 1.0).abs() < 1e-9 && (beta[1] - 2.0).abs() < 1e-9, "{beta:?}");
        let mut collinear = vec![1.0, 2.0, 2.0, 4.0];
        let e = solve_penalized(&mut collinear, &[1.0, 2.0], 0.0).unwrap_err();
        assert!(e.to_string().contains("singular weighted nuisance system"), "{e}");
    }

    #[test]
    fn weight_fingerprint_is_the_canonical_stream_of_the_weight_bits() {
        let weights = [0.5_f64, 1.0, 1.5];
        let mut writer = antecedent_core::CanonicalWriter::new(
            "antecedent.estimate.bayesian_robust_ate.weights",
        );
        writer.word(3);
        for w in weights {
            writer.word(w.to_bits());
        }
        let digest = writer.finish();
        let expected = u64::from_le_bytes(digest[..8].try_into().unwrap());
        assert_eq!(fingerprint(&weights), expected);
        assert_ne!(fingerprint(&[0.5, 1.0, 1.500_000_000_000_000_2]), expected);
        assert_eq!(expected, 18_274_849_298_044_415_906_u64);
    }
}
