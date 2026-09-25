//! Modular Bayesian-bootstrap cross-fitted orthogonal ATE.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{ExecutionContext, StreamDomain};

use crate::EstimationError;

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
    let fold_ids = plan_bayesian_robust_ate_folds(input, options.folds)?;
    let x = make_design(input);
    let mut effects = Vec::with_capacity(options.draws);
    let mut refits = Vec::with_capacity(options.draws);
    for draw in 0..options.draws {
        if ctx.cancellation.is_cancelled() {
            return Err(EstimationError::unsupported("Bayesian robust ATE cancelled"));
        }
        let mut rng = ctx.rng.stream_for(StreamDomain::Bayesian, 0xB007_0000 | draw as u64);
        let mut weights: Vec<f64> =
            (0..n).map(|_| -rng.next_f64().max(f64::MIN_POSITIVE).ln()).collect();
        let total = weights.iter().sum::<f64>();
        if !total.is_finite() || total <= 0.0 {
            return Err(EstimationError::stats_msg("invalid Bayesian-bootstrap row weights"));
        }
        for weight in &mut weights {
            *weight *= n as f64 / total;
        }
        let fingerprint = fingerprint(&weights);
        let (estimate, audit) = fit_draw(input, &x, &fold_ids, &weights, options, fingerprint)?;
        effects.push(estimate);
        refits.push(audit);
    }
    let estimate = effects.iter().sum::<f64>() / effects.len() as f64;
    let standard_deviation = (effects.iter().map(|v| (v - estimate).powi(2)).sum::<f64>()
        / (effects.len() - 1) as f64)
        .sqrt();
    let interval = interval(&effects, options.coverage);
    Ok(BayesianRobustAteResult {
        estimate,
        interval,
        standard_deviation,
        draws: effects,
        row_ids: input.row_ids.clone(),
        fold_ids,
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
/// # Errors
/// Returns the same input and fold-count refusals as the estimator.
pub fn plan_bayesian_robust_ate_folds(
    input: &BayesianRobustAteInput,
    folds: usize,
) -> Result<Vec<u16>, EstimationError> {
    let options = BayesianRobustAteOptions { folds, ..BayesianRobustAteOptions::default() };
    validate(input, options)?;
    Ok(fixed_folds(input, folds))
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
        return Err(EstimationError::unsupported(
            "Bayesian robust ATE requires both treatment arms",
        ));
    }
    Ok(())
}

fn fixed_folds(input: &BayesianRobustAteInput, folds: usize) -> Vec<u16> {
    let mut order: Vec<usize> = (0..input.row_ids.len()).collect();
    order.sort_by_key(|&i| input.row_ids[i]);
    let mut counts = [0usize; 2];
    let mut result = vec![0u16; order.len()];
    for i in order {
        let arm = usize::from(input.treatment[i]);
        result[i] = u16::try_from(counts[arm] % folds)
            .expect("validated fold count fits the stored fold identity");
        counts[arm] += 1;
    }
    result
}

fn make_design(input: &BayesianRobustAteInput) -> Vec<Vec<f64>> {
    (0..input.row_ids.len())
        .map(|i| std::iter::once(1.0).chain(input.covariates.iter().map(|c| c[i])).collect())
        .collect()
}

fn fit_draw(
    input: &BayesianRobustAteInput,
    x: &[Vec<f64>],
    fold_ids: &[u16],
    weights: &[f64],
    opt: BayesianRobustAteOptions,
    weight_fingerprint: u64,
) -> Result<(f64, Vec<BayesianRobustRefit>), EstimationError> {
    let n = x.len();
    let mut e = vec![0.0; n];
    let mut mu0 = vec![0.0; n];
    let mut mu1 = vec![0.0; n];
    let mut audit = Vec::with_capacity(opt.folds * 3);
    for fold in 0..opt.folds {
        let train: Vec<usize> = (0..n).filter(|&i| usize::from(fold_ids[i]) != fold).collect();
        let valid: Vec<usize> = (0..n).filter(|&i| usize::from(fold_ids[i]) == fold).collect();
        let p = fit_logistic(x, &input.treatment, weights, &train, opt.ridge_penalty)?;
        for &i in &valid {
            e[i] = logistic(dot(&p, &x[i])).clamp(opt.propensity_clip, 1.0 - opt.propensity_clip);
        }
        audit.push(BayesianRobustRefit { role: "propensity", fold, weight_fingerprint });
        for arm in [false, true] {
            let arm_rows: Vec<usize> =
                train.iter().copied().filter(|&i| input.treatment[i] == arm).collect();
            let beta = fit_ridge(x, &input.outcome, weights, &arm_rows, opt.ridge_penalty)?;
            let predictions = if arm { &mut mu1 } else { &mut mu0 };
            for &i in &valid {
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
    let mut gram = vec![vec![0.0; p]; p];
    let mut rhs = vec![0.0; p];
    for &i in rows {
        for j in 0..p {
            rhs[j] += weights[i] * x[i][j] * y[i];
            for k in 0..p {
                gram[j][k] += weights[i] * x[i][j] * x[i][k];
            }
        }
    }
    solve_penalized(gram, rhs, lambda)
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
    let mut beta = vec![0.0; p];
    let mut converged = false;
    for _ in 0..80 {
        let mut info = vec![vec![0.0; p]; p];
        let mut score = vec![0.0; p];
        for &i in rows {
            let prob = logistic(dot(&beta, &x[i]));
            let variance = (prob * (1.0 - prob)).max(1e-8);
            for j in 0..p {
                score[j] += weights[i] * x[i][j] * (f64::from(t[i]) - prob);
                for k in 0..p {
                    info[j][k] += weights[i] * variance * x[i][j] * x[i][k];
                }
            }
        }
        for j in 1..p {
            info[j][j] += lambda;
            score[j] -= lambda * beta[j];
        }
        let step = solve(info, score)?;
        let max_step = step.iter().map(|v| v.abs()).fold(0.0, f64::max);
        for j in 0..p {
            beta[j] += step[j].clamp(-2.0, 2.0);
        }
        if max_step < 1e-8 {
            converged = true;
            break;
        }
    }
    if !converged || beta.iter().any(|v| !v.is_finite()) {
        return Err(EstimationError::stats_msg("weighted logistic nuisance did not converge"));
    }
    Ok(beta)
}

fn solve_penalized(
    mut gram: Vec<Vec<f64>>,
    rhs: Vec<f64>,
    lambda: f64,
) -> Result<Vec<f64>, EstimationError> {
    for (j, row) in gram.iter_mut().enumerate().skip(1) {
        row[j] += lambda;
    }
    solve(gram, rhs)
}

fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Result<Vec<f64>, EstimationError> {
    let p = b.len();
    for col in 0..p {
        let pivot = (col..p).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs())).unwrap();
        if a[pivot][col].abs() < 1e-12 {
            return Err(EstimationError::stats_msg("singular weighted nuisance system"));
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        let d = a[col][col];
        for j in col..p {
            a[col][j] /= d;
        }
        b[col] /= d;
        for i in 0..p {
            if i == col {
                continue;
            }
            let f = a[i][col];
            for j in col..p {
                a[i][j] -= f * a[col][j];
            }
            b[i] -= f * b[col];
        }
    }
    Ok(b)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn logistic(x: f64) -> f64 {
    1.0 / (1.0 + (-x.clamp(-30.0, 30.0)).exp())
}
fn fingerprint(values: &[f64]) -> u64 {
    values
        .iter()
        .fold(0xcbf2_9ce4_8422_2325_u64, |h, x| (h ^ x.to_bits()).wrapping_mul(0x100_0000_01b3))
}
fn interval(draws: &[f64], coverage: f64) -> (f64, f64) {
    let mut sorted = draws.to_vec();
    sorted.sort_by(f64::total_cmp);
    let q = |p: f64| {
        // Validated coverage gives a probability and `draws` contains at least two entries.
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "rank is clamped to the nonempty draw slice"
        )]
        sorted[((p * (sorted.len() + 1) as f64).floor() as usize).clamp(1, sorted.len()) - 1]
    };
    (q((1.0 - coverage) / 2.0), q((1.0 + coverage) / 2.0))
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
                rng.next_f64() < logistic(z)
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
        assert_eq!(result.fold_ids, fixed_folds(&input, 4));
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
}
