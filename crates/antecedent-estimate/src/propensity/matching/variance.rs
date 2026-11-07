//! Analytic variance calculations for matching estimators.

use super::{MatchedArms, MatchedEffects, MatchingSample, fit_linear_mean, predict_linear};
use crate::error::EstimationError;
use crate::se::AnalyticSeKind;
use antecedent_core::TargetPopulation;

/// What the analytic matching variance reads beside the SE kind.
pub(super) struct MatchedVariance<'a> {
    pub(super) sample: &'a MatchingSample<'a>,
    pub(super) target: &'a TargetPopulation,
    pub(super) arms: &'a MatchedArms,
    pub(super) effects: &'a MatchedEffects,
    pub(super) target_weights: Option<&'a [f64]>,
    pub(super) ate: f64,
}

impl MatchedVariance<'_> {
    /// Coefficient of each sample row's outcome in the weighted matched estimator.
    fn coefficients(&self, weights: &[f64]) -> Vec<f64> {
        let mass: f64 = self
            .effects
            .effect_rows
            .iter()
            .map(|&row| weights.get(row).copied().unwrap_or(0.0))
            .sum();
        let query_shares: Vec<f64> = self
            .effects
            .effect_rows
            .iter()
            .map(|&row| weights.get(row).copied().unwrap_or(0.0) / mass)
            .collect();
        custom_unit_coefficients(
            weights,
            &self.effects.effect_rows,
            &self.effects.donor_usage(Some(&query_shares)),
            &self.arms.treated_idx,
            &self.arms.control_idx,
            self.sample.treatment.len(),
            matches!(self.target, TargetPopulation::Untreated),
        )
    }

    /// The population variance form, when requested.
    fn population_se(&self) -> Result<Option<f64>, EstimationError> {
        if !self.sample.population_variance {
            return Ok(None);
        }
        let unit_weights = vec![1.0; self.sample.treatment.len()];
        let weights = self.target_weights.unwrap_or(&unit_weights);
        let coefficients = self.coefficients(weights);
        Ok(Some(population_matching_se(
            &self.arms.treated_feat,
            &self.arms.treated_y,
            &self.arms.control_feat,
            &self.arms.control_y,
            &self.arms.treated_idx,
            &self.arms.control_idx,
            self.sample.features,
            self.sample.dim,
            &self.effects.effect_rows,
            &coefficients,
            weights,
        )?))
    }

    /// The analytic standard error of the requested kind.
    pub(super) fn analytic_se(&self, se_kind: AnalyticSeKind) -> Result<f64, EstimationError> {
        // Under a custom target or sampling weights the estimator is the weighted matched mean,
        // so each unit's outcome enters with its query weight plus the weights of the queries it
        // donates to; the equal-weight `(1 + K)/n` coefficients would describe a different
        // estimator.
        let unit_weights = vec![1.0; self.sample.treatment.len()];
        let weights = self.target_weights.unwrap_or(&unit_weights);
        let coefficients = self.coefficients(weights);
        let population_se = self.population_se()?;
        let effects = self.effects;
        Ok(match se_kind {
            AnalyticSeKind::Homoskedastic if population_se.is_some() => {
                population_se.unwrap_or(f64::NAN)
            }
            AnalyticSeKind::Homoskedastic => weighted_abadie_imbens_se(
                &effects.per_unit_effects,
                self.ate,
                &effects.effect_rows,
                &coefficients,
                weights,
                1.0 / effects.mean_noise_inflation(),
            ),
            AnalyticSeKind::Hc0
            | AnalyticSeKind::Hc1
            | AnalyticSeKind::Hc2
            | AnalyticSeKind::Hc3 => {
                return Err(EstimationError::unsupported_coded(
                    antecedent_core::reason_code!("estimator_inference_mismatch"),
                    "matching does not implement HC0–HC3 sandwich SEs; use Homoskedastic (Abadie–Imbens)",
                ));
            }
            // A matched effect mixes the query unit's outcome with its donors' outcomes, and
            // donors come from other clusters and time points. Charging the whole per-query
            // residual to the query's cluster (or time) mis-assigns the donor noise, so a
            // cluster, multiway or HAC sandwich over those contributions is not a valid
            // variance for the matching estimator.
            //
            // Kept as a refusal rather than derived, because no clustered influence
            // function is available for this exact estimator:
            // - Fixed-M nearest-neighbour matching is not asymptotically linear in an influence
            //   function of one observation; its only linear representation is the
            //   martingale one, √N(τ̂ − τ) ≈ N^{-1/2} Σ_i [τ(X_i) − τ + (2W_i − 1)(1 + K_i/M) ε_i]
            //   (Abadie–Imbens 2006, Econometrica 74(1); 2012, JASA 107(498)), proved for
            //   random samples. Its K_i distribution and the negligibility of the
            //   bias-corrected conditional bias are not established under within-cluster or
            //   serial dependence of the covariates and of the matches themselves.
            // - Even granting that representation, a cluster sum needs per-unit ε_i
            //   (within-cluster cross products ε_i ε_j) and per-unit τ(X_i). Abadie–Imbens
            //   recover only the diagonal σ²(X_i) from same-arm matches; here the bias
            //   correction is a linear fit on the match feature (the one-dimensional score
            //   for propensity matching), which is not a consistent μ_w(X), and the
            //   published SE is a pooled ½·Var(τ̂_i) proxy, not the conditional estimator.
            // - Propensity matching on an estimated score needs the Abadie–Imbens (2016,
            //   Econometrica 84(2)) estimated-score adjustment, which is not implemented even
            //   for iid data; a clustered version would also need it.
            // - Resampling does not fill the gap: the (cluster) bootstrap is inconsistent for
            //   fixed-M matching (Abadie–Imbens 2008, Econometrica 76(6)), and the wild
            //   bootstrap of the martingale representation (Otsu–Rai 2017, JASA 112(520)) is
            //   proved for iid samples only.
            AnalyticSeKind::Cluster
            | AnalyticSeKind::Multiway
            | AnalyticSeKind::NeweyWest { .. }
            | AnalyticSeKind::PanelClusterHac { .. } => {
                return Err(EstimationError::refused(
                    antecedent_core::reason_code!("estimator_inference_mismatch"),
                    "matching has no valid cluster, multiway or HAC standard error (donor outcomes \
                     belong to other clusters and times than the matched unit); use Homoskedastic \
                     (Abadie–Imbens), or AIPW, which implements cluster, multiway and HAC standard \
                     errors, for clustered or serially dependent data",
                ));
            }
        })
    }
}

/// The row-label vectors of the SE specification must cover every row.
/// Coefficient `c_i` of each sample row's outcome in the weighted pooled matching estimator
/// `τ̂ = Σ_q w̃_q τ̂_q` (`w̃_q = w_q / M`, `M = Σ_q w_q` over the matched queries): the row's own
/// query weight `w_i / M` plus the weighted donor reuse `K^w_i / M`, where
/// `K^w_i = Σ_{q: donor(q) = i} w_q` is the weight of the queries row `i` donates to.
///
/// Only queries retained by the caliper contribute their own `w_i / M`. A dropped
/// unit may still donate to other queries and then contributes only its donor share.
///
/// At constant weights `w_i = w` this is `c_i = (1 + K_i) / n` (ATE) and `c_q = 1/n`,
/// `c_j = K_j / n` (ATT), the coefficients behind the unweighted `(1 + K_j)²` and `n + ΣK²`
/// sums, whatever the scale of `w`.
#[allow(
    clippy::too_many_arguments,
    reason = "the coefficients need the weights, both arms' rows and the donor mapping"
)]
fn custom_unit_coefficients(
    target_weights: &[f64],
    effect_rows: &[usize],
    donor_usage: &[f64],
    treated_idx: &[usize],
    control_idx: &[usize],
    nrows: usize,
    donors_are_treated: bool,
) -> Vec<f64> {
    let n_control = control_idx.len();
    let weight = |row: usize| target_weights.get(row).copied().unwrap_or(0.0);
    let mass: f64 = effect_rows.iter().map(|&r| weight(r)).sum();
    let mut coefficient = vec![0.0; nrows];
    for &row in effect_rows {
        coefficient[row] += weight(row) / mass;
    }
    // `donor_usage` holds, per donor, the share of the matched queries' weight it carries
    // (each query's weight split equally over its donors).
    for (donor, &share) in donor_usage.iter().enumerate() {
        if share == 0.0 {
            continue;
        }
        // The ATC match draws its donors from the treated arm alone, indexed without the
        // control offset the pooled (ATE) donor list carries for its ATC half.
        let donor_row = if donors_are_treated {
            treated_idx[donor]
        } else if donor < n_control {
            control_idx[donor]
        } else {
            treated_idx[donor - n_control]
        };
        coefficient[donor_row] += share;
    }
    coefficient
}

/// Abadie–Imbens homoskedastic SE of the weighted matched mean:
/// `Var = σ̂² Σ_i c_i²`, `σ̂² = ½ · Σ_q w̃_q (τ̂_q − τ̂)² / (1 − Σ_q w̃_q²)`.
///
/// The denominator is the reliability-weights degrees-of-freedom correction of the weighted
/// variance, so it generalizes the unweighted `Var(τ̂ᵢ)/2` with its `n − 1` divisor. At
/// constant weights `w̃_q = 1/n`, so `1 − Σ w̃² = (n − 1)/n`,
/// `σ̂² = ½ Σ(τ̂_q − τ̂)² / (n − 1)` and, with the coefficients of
/// [`custom_unit_coefficients`], `Var = σ̂² Σ_j (1 + K_j)² / n²` (ATE) or
/// `σ̂² (n + ΣK²) / n²` (ATT/ATC): exactly [`abadie_imbens_se`], independent of the scale of
/// the weights. A single matched query has no variance estimate and returns `NaN`.
fn weighted_abadie_imbens_se(
    effects: &[f64],
    ate: f64,
    effect_rows: &[usize],
    coefficients: &[f64],
    target_weights: &[f64],
    noise_share: f64,
) -> f64 {
    let weight = |row: usize| target_weights.get(row).copied().unwrap_or(0.0);
    let mass: f64 = effect_rows.iter().map(|&r| weight(r)).sum();
    let concentration: f64 = effect_rows.iter().map(|&r| (weight(r) / mass).powi(2)).sum();
    let denominator = 1.0 - concentration;
    if denominator <= 0.0 {
        return f64::NAN;
    }
    let sigma2: f64 = noise_share
        * effects
            .iter()
            .zip(effect_rows)
            .map(|(e, &r)| weight(r) / mass * (e - ate).powi(2))
            .sum::<f64>()
        / denominator;
    (sigma2 * coefficients.iter().map(|c| c * c).sum::<f64>()).max(0.0).sqrt()
}

/// Abadie–Imbens population variance: outcome-noise donor reuse plus the variance of the
/// conditional treatment effect over the target population. The fitted arm means are the
/// linear models already declared by matching's bias correction. Estimating noise from the
/// entire matched-contrast spread would charge effect heterogeneity to donor noise as well.
#[allow(clippy::too_many_arguments, reason = "arm regressions and aligned matched population")]
fn population_matching_se(
    treated_features: &[f64],
    treated_y: &[f64],
    control_features: &[f64],
    control_y: &[f64],
    treated_rows: &[usize],
    control_rows: &[usize],
    features: &[f64],
    dim: usize,
    effect_rows: &[usize],
    coefficients: &[f64],
    weights: &[f64],
) -> Result<f64, EstimationError> {
    let beta1 = fit_linear_mean(treated_features, treated_y, dim)
        .ok_or_else(|| EstimationError::stats_msg("matching arm outcome model failed"))?;
    let beta0 = fit_linear_mean(control_features, control_y, dim)
        .ok_or_else(|| EstimationError::stats_msg("matching arm outcome model failed"))?;
    let n = treated_y.len() + control_y.len();
    let parameters = beta1.len() + beta0.len();
    if n <= parameters || effect_rows.len() < 2 {
        return Err(EstimationError::data_msg(
            "population matching variance requires residual degrees of freedom",
        ));
    }
    let mut residual_sum = 0.0;
    for (x, y, beta) in
        [(treated_features, treated_y, &beta1), (control_features, control_y, &beta0)]
    {
        for (row, outcome) in y.iter().enumerate() {
            residual_sum += (outcome - predict_linear(beta, x, dim, row)).powi(2);
        }
    }
    let noise = residual_sum / (n - parameters) as f64;
    let mass = effect_rows.iter().map(|&row| weights[row]).sum::<f64>();
    let contrasts: Vec<f64> = effect_rows
        .iter()
        .map(|&row| {
            predict_linear(&beta1, features, dim, row) - predict_linear(&beta0, features, dim, row)
        })
        .collect();
    let mean = effect_rows
        .iter()
        .zip(&contrasts)
        .map(|(&row, contrast)| weights[row] * contrast)
        .sum::<f64>()
        / mass;
    let count = effect_rows.len() as f64;
    let heterogeneity = effect_rows
        .iter()
        .zip(&contrasts)
        .map(|(&row, contrast)| (weights[row] / mass * (contrast - mean)).powi(2))
        .sum::<f64>()
        * count
        / (count - 1.0);
    let target_design: Vec<f64> = std::iter::once(1.0)
        .chain((0..dim).map(|c| {
            effect_rows.iter().map(|&row| weights[row] * features[row * dim + c]).sum::<f64>()
                / mass
        }))
        .collect();
    let norm = bias_adjusted_coefficient_norm(
        treated_features,
        dim,
        treated_rows,
        coefficients,
        &target_design,
    )? + bias_adjusted_coefficient_norm(
        control_features,
        dim,
        control_rows,
        coefficients,
        &target_design,
    )?;
    Ok((noise * norm + heterogeneity).sqrt())
}

/// Exact coefficients of outcome noise after the shared OLS bias correction. If the raw
/// donor coefficient vector is c, the corrected vector is c + X(X'X)^-1(a-X'c), where a is
/// the target query's covariate mean. Squaring raw donor coefficients alone loses the fitted
/// regression's variance and its covariance with the matched outcomes.
pub(super) fn bias_adjusted_coefficient_norm(
    features: &[f64],
    dim: usize,
    rows: &[usize],
    coefficients: &[f64],
    target: &[f64],
) -> Result<f64, EstimationError> {
    let p = dim + 1;
    let mut gram = vec![0.0; p * p];
    let mut mismatch = target.to_vec();
    let row = |i: usize| {
        std::iter::once(1.0)
            .chain(features[i * dim..(i + 1) * dim].iter().copied())
            .collect::<Vec<_>>()
    };
    for (i, &source) in rows.iter().enumerate() {
        let x = row(i);
        for c in 0..p {
            mismatch[c] -= x[c] * coefficients[source];
            for d in 0..p {
                gram[c * p + d] += x[c] * x[d];
            }
        }
    }
    let correction = crate::util::solve_spd(&gram, &mismatch, p)
        .ok_or_else(|| EstimationError::stats_msg("singular matching bias-adjustment variance"))?;
    Ok(rows
        .iter()
        .enumerate()
        .map(|(i, &source)| {
            (coefficients[source] + row(i).iter().zip(&correction).map(|(x, b)| x * b).sum::<f64>())
                .powi(2)
        })
        .sum())
}
