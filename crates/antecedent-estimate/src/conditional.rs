//! Conditional ATE with effect modifiers.
//!
//! Fits `Y ~ 1 + T + W + T×W` and reports the average treatment effect
//! marginalized over observed modifier values:
//! `ATE = (β_T + β_{T×W} · Ē[W]) · (active − control)` for a single modifier.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg_attr(
    test,
    allow(
        clippy::cast_possible_truncation,
        reason = "test fixtures compare exact constants and index with small literals"
    )
)]

use std::sync::Arc;

use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSet, AssumptionSource,
    AssumptionStatus, AverageEffectQuery, ConditionalEffectQuery, ExecutionContext,
    TargetPopulation, VariableId,
};
use antecedent_data::TabularData;
use antecedent_expr::IdentifiedEstimand;
use antecedent_stats::{DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace};

use crate::adjustment::{EffectEstimate, intervention_f64};
use crate::error::EstimationError;
use crate::gcomp::linear_predictors;
use crate::overlap::OverlapPolicy;
use crate::util::require_explicit_override;

/// Remedy named when the frequentist interaction regression, configured with
/// `joint_modifiers` off, is asked for more than one modifier. Its estimand,
/// delta-method SE and calibration record are for one modifier; a joint-modifier
/// frequentist interval would be an unmeasured coordinate, so the refusal points at
/// the routes that take a modifier set instead. The adjustment handoff is named as a
/// capability; Python appends the call.
const MULTI_MODIFIER_REMEDY: &str = "the frequentist interaction regression fits one modifier; \
for several modifiers use the Bayesian conditional estimator \
(Rust: `.estimator(EstimatorId::BayesianConditional).inference(InferenceMode::Bayesian(..))`), \
or hand the identified adjustment set to an external CATE learner \
(capability: adjustment handoff), or fit one `ConditionalEffect` per modifier";

/// Per-arm plugin scores for a conditional linear fit.
///
/// `influence[a]` is aligned with [`Self::row_index`]. Means are
/// `μ_a = n⁻¹ Σ μ̂(a, W_i, Z_i)`.
#[derive(Clone, Debug)]
pub struct ConditionalArmScores {
    /// Interventional arm means `[μ_control, μ_active]`.
    pub means: [f64; 2],
    /// Plugin IF for each arm mean.
    pub influence: [Vec<f64>; 2],
    /// Original-row index of each complete-case score.
    pub row_index: Vec<antecedent_data::GlobalId>,
    /// Complete-case treatment values (same order as [`Self::row_index`]).
    pub treatment: Vec<f64>,
}

impl ConditionalArmScores {
    fn with_means(mut self, means: [f64; 2]) -> Self {
        self.means = means;
        self
    }
}

/// Conditional linear adjustment ATE.
#[derive(Clone, Debug)]
pub struct ConditionalLinearAdjustment {
    /// Overlap policy (must be explicit override).
    pub overlap: OverlapPolicy,
    /// Backend.
    pub backend: FaerBackend,
    /// Master seed of the cross-fit fold plan used by the binary AIPW arm scores.
    pub fold_seed: u64,
    /// Accept a set of several effect modifiers (the default). The regression, delta-method
    /// standard error and influence functions are the exact vector generalization of the
    /// one-modifier model, but the interval's coverage is measured for one modifier only, so an
    /// accepted set carries an assumption record naming the interval as uncalibrated. Set this
    /// to `false` to refuse a modifier set instead.
    pub joint_modifiers: bool,
}

impl Default for ConditionalLinearAdjustment {
    fn default() -> Self {
        Self::new()
    }
}

impl ConditionalLinearAdjustment {
    /// Defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            overlap: OverlapPolicy::ExplicitOverride,
            backend: FaerBackend,
            fold_seed: 0,
            joint_modifiers: true,
        }
    }

    /// Seed the cross-fit fold plan of the binary AIPW arm scores (the run's master seed).
    #[must_use]
    pub const fn with_fold_seed(mut self, fold_seed: u64) -> Self {
        self.fold_seed = fold_seed;
        self
    }

    /// Accept a set of several effect modifiers, with an interval whose coverage has not been
    /// measured for a modifier set (see [`Self::joint_modifiers`]); this is the default.
    #[must_use]
    pub const fn with_joint_modifiers(mut self) -> Self {
        self.joint_modifiers = true;
        self
    }

    /// Set the overlap policy. Must remain [`OverlapPolicy::ExplicitOverride`].
    #[must_use]
    pub const fn with_overlap(mut self, overlap: OverlapPolicy) -> Self {
        self.overlap = overlap;
        self
    }

    /// Estimate conditional ATE from a [`ConditionalEffectQuery`].
    ///
    /// # Errors
    ///
    /// Empty modifiers, unsupported populations, or OLS failures.
    pub fn estimate(
        &self,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        query: &ConditionalEffectQuery,
        ctx: &ExecutionContext,
    ) -> Result<EffectEstimate, EstimationError> {
        let _ = ctx;
        require_explicit_override(
            self.overlap,
            "ConditionalLinearAdjustment requires ExplicitOverride overlap policy",
        )?;
        query.validate()?;
        self.estimate_with_means(data, estimand, query).map(|(estimate, _)| estimate)
    }

    /// Estimate the contrast and the two interventional arm means.
    ///
    /// # Errors
    ///
    /// Same refusals as [`Self::estimate`].
    pub fn estimate_with_means(
        &self,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        query: &ConditionalEffectQuery,
    ) -> Result<(EffectEstimate, [f64; 2]), EstimationError> {
        require_explicit_override(
            self.overlap,
            "ConditionalLinearAdjustment requires ExplicitOverride overlap policy",
        )?;
        query.validate()?;
        self.estimate_ate(data, estimand, &query.inner).map(|(estimate, arms, _)| (estimate, arms))
    }

    /// Estimate the contrast, arm means, and per-arm influence functions.
    ///
    /// Binary 0/1 treatments use cross-fitted AIPW scores (nonparametric EIF
    /// under back-door, positivity, and nuisance rates): the interaction model is not
    /// fitted, the result carries the propensity policy and overlap report actually applied
    /// and a declared `conditional.crossfit_aipw_substitution` assumption. Non-binary levels
    /// keep the linear plugin IF under the explicit override.
    ///
    /// # Errors
    ///
    /// Same refusals as [`Self::estimate`], or AIPW fold/overlap failure.
    pub fn estimate_with_arm_scores(
        &self,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        query: &ConditionalEffectQuery,
    ) -> Result<(EffectEstimate, ConditionalArmScores), EstimationError> {
        require_explicit_override(
            self.overlap,
            "ConditionalLinearAdjustment requires ExplicitOverride overlap policy",
        )?;
        query.validate()?;
        let modifiers = query.inner.effect_modifiers.len();
        if query.inner.target_population != TargetPopulation::AllObserved
            || modifiers == 0
            || (modifiers > 1 && !self.joint_modifiers)
        {
            let message = "conditional arm scores require AllObserved and exactly one modifier";
            return Err(
                if query.inner.target_population == TargetPopulation::AllObserved
                    && query.inner.effect_modifiers.len() > 1
                {
                    EstimationError::unsupported_with_remedy(message, MULTI_MODIFIER_REMEDY)
                } else {
                    EstimationError::unsupported(message)
                },
            );
        }
        if !matches!(query.inner.outcome_functional, antecedent_core::OutcomeFunctional::Mean) {
            return Err(EstimationError::unsupported_coded(
                antecedent_core::reason_code!("route_not_supported"),
                "conditional arm scores require an already transformed outcome and the Mean functional",
            ));
        }
        if binary_zero_one(&query.inner)? {
            let (scores, overlap, overlap_report) =
                aipw_conditional_arm_scores(data, estimand, &query.inner, self.fold_seed)?;
            let n = scores.influence[0].len() as f64;
            let contrast: Vec<f64> =
                scores.influence[1].iter().zip(&scores.influence[0]).map(|(a, b)| a - b).collect();
            let mean = contrast.iter().sum::<f64>() / n;
            let var = contrast.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
            // The caller asked for the regression path (explicit override); for a 0/1 treatment
            // the number below is the cross-fitted AIPW contrast instead, built under a
            // clipped propensity model. Report the policy that actually ran, its overlap
            // report, and name the substitution.
            let mut assumptions = AssumptionSet::default();
            assumptions.push(crossfit_substitution_record());
            if modifiers > 1 {
                assumptions.push(joint_modifier_record());
            }
            let estimate = EffectEstimate::new(
                scores.means[1] - scores.means[0],
                (var / n).sqrt(),
                assumptions,
                overlap,
            )
            .with_overlap_report(overlap_report)
            .with_influence(Some(Arc::from(contrast)));
            return Ok((estimate, scores));
        }
        let (estimate, means, scores) = self.estimate_ate(data, estimand, &query.inner)?;
        Ok((estimate, scores.with_means(means)))
    }

    /// Estimate from an [`AverageEffectQuery`] with non-empty modifiers.
    ///
    /// # Errors
    ///
    /// Empty modifiers or OLS failures.
    #[allow(
        clippy::too_many_lines,
        reason = "validate the query, build the interaction design, fit, then derive the point, delta-method SE and influence from the same coefficients and inverse"
    )]
    pub fn estimate_ate(
        &self,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        query: &AverageEffectQuery,
    ) -> Result<(EffectEstimate, [f64; 2], ConditionalArmScores), EstimationError> {
        if query.outcome_functional.quantile_level().is_some() {
            return Err(EstimationError::unsupported_coded(
                antecedent_core::reason_code!("route_not_supported"),
                "conditional quantiles require CDF-grid inversion through Study; a mean regression cannot estimate a quantile",
            ));
        }
        if query.effect_modifiers.is_empty() {
            return Err(EstimationError::unsupported_coded(
                antecedent_core::reason_code!("required_option_missing"),
                "ConditionalLinearAdjustment requires effect modifiers",
            ));
        }
        let m = query.effect_modifiers.len();
        if m > 1 && !self.joint_modifiers {
            return Err(EstimationError::unsupported_with_remedy(
                "ConditionalLinearAdjustment currently supports one effect modifier",
                MULTI_MODIFIER_REMEDY,
            ));
        }
        if query.target_population != TargetPopulation::AllObserved {
            return Err(EstimationError::refused(
                antecedent_core::reason_code!("population_not_estimable"),
                "ConditionalLinearAdjustment only supports AllObserved",
            ));
        }
        if !estimand.is_adjustment_shaped() {
            return Err(EstimationError::IncompatibleEstimand {
                message: "ConditionalLinearAdjustment expects an adjustment-shaped estimand",
            });
        }
        let active = intervention_f64(&query.active)?;
        let control = intervention_f64(&query.control)?;
        let delta = active - control;
        if delta == 0.0 {
            return Err(EstimationError::unsupported(
                "active and control treatment levels must differ",
            ));
        }

        let modifier_ids = &query.effect_modifiers;
        let extra_z: Vec<VariableId> =
            estimand.adjustment_set.iter().copied().filter(|z| !modifier_ids.contains(z)).collect();
        let mut ids = vec![query.treatment, query.outcome];
        ids.extend_from_slice(modifier_ids);
        ids.extend_from_slice(&extra_z);
        let crate::prepare::AdmittedColumns { data, row_mask, receipt } =
            crate::prepare::admit_estimator_columns(data, &ids, crate::prepare::FINITE_TABLE_ONLY)?;
        let t = data.float64_selected(query.treatment, &row_mask).map_err(EstimationError::from)?;
        let y = data.float64_selected(query.outcome, &row_mask).map_err(EstimationError::from)?;
        let w_columns = modifier_ids
            .iter()
            .map(|&id| data.float64_selected(id, &row_mask).map_err(EstimationError::from))
            .collect::<Result<Vec<_>, _>>()?;
        let w: Vec<&[f64]> = w_columns.iter().map(|column| &column[..]).collect();
        let n = t.len();
        if n < 8 {
            return Err(EstimationError::data_msg("too few complete rows for conditional ATE"));
        }

        // Design: [1, T, W_1..W_m, T*W_1..T*W_m, Z...]; skip Z that is already a modifier. With
        // one modifier this is the `[1, T, W, T*W, Z...]` interaction model.
        let n_z = extra_z.len();
        let layout = InteractionLayout { m, n_z };
        let ncols = layout.ncols();
        let mut design = vec![0.0; n * ncols];
        for i in 0..n {
            design[i] = 1.0;
            design[n + i] = t[i];
            for (k, column) in w.iter().enumerate() {
                design[(2 + k) * n + i] = column[i];
                design[(2 + m + k) * n + i] = t[i] * column[i];
            }
        }
        for (k, &z) in extra_z.iter().enumerate() {
            let base = (layout.z_start() + k) * n;
            data.float64_selected_into(z, &row_mask, &mut design[base..base + n])
                .map_err(EstimationError::from)?;
        }

        let mut ws = LeastSquaresWorkspace::default();
        let fit = self
            .backend
            .least_squares(&design, n, ncols, &y, &mut ws)
            .map_err(crate::error::EstimationError::from)?;
        // QR residuals are `y − Xβ` in the same order `ols_sigma2` sums, so RSS / (n−p)
        // matches that helper without a second pass over the design.
        let sigma2 = fit.rss / (n.saturating_sub(ncols)).max(1) as f64;
        let coef = fit.coefficients;
        let residuals = fit.residuals;

        // One Gram of this design. Cholesky's inverse is the variance factor when the
        // factor exists; Gauss–Jordan remains the inverse when that factor is refused.
        // The two floors differ, so a conditional SE can move.
        let inv = gram_inverse(&design, n, ncols)
            .ok_or_else(|| EstimationError::stats_msg("singular design in conditional ATE"))?;

        let n_f = n as f64;
        let w_bar: Vec<f64> = w.iter().map(|column| column.iter().sum::<f64>() / n_f).collect();
        // Marginal ATE at mean W: (β_T + Σ_k β_{T×W_k} * Ē[W_k]) * delta
        let mut cate_at_mean = coef[1];
        for (k, mean) in w_bar.iter().enumerate() {
            cate_at_mean += coef[layout.interaction(k)] * mean;
        }
        let point = cate_at_mean * delta;

        // Delta-method SE with g = δ·(e_T + Σ_k w̄_k·e_{T×W_k}) for the coefficients,
        // plus the sampling variance of the w̄_k themselves: the reported functional
        // averages the CATE over the modifier distribution, so its IF carries
        // δ·Σ_k β_{T×W_k}·(w_ik − w̄_k) (asymptotically uncorrelated with the OLS
        // score term). Treating w̄ as fixed under-covers whenever some β_{T×W_k} ≠ 0.
        let mut g = vec![0.0; ncols];
        g[1] = delta;
        for (k, mean) in w_bar.iter().enumerate() {
            g[layout.interaction(k)] = delta * mean;
        }
        let modifier_terms: Vec<f64> =
            (0..m).map(|k| delta * coef[layout.interaction(k)]).collect();
        let w_var_of_mean = modifier_mean_variance(&modifier_terms, &w, &w_bar);
        let se_coef = crate::util::delta_method_se(&inv, ncols, &g, sigma2);
        let se_analytic = (se_coef * se_coef + w_var_of_mean).sqrt();

        let contrast_leverage = inverse_weighted_leverage(&inv, &design, n, ncols, &g);
        let influence: Vec<f64> = contrast_leverage
            .iter()
            .zip(&residuals)
            .enumerate()
            .map(|(i, (g_inv_x, residual))| {
                let mut value = n_f * g_inv_x * residual;
                for (k, column) in w.iter().enumerate() {
                    value += modifier_terms[k] * (column[i] - w_bar[k]);
                }
                value
            })
            .collect();

        let mut mu0 = 0.0;
        let mut mu1 = 0.0;
        let zterms =
            linear_predictors(&design[layout.z_start() * n..], n, n_z, &coef[layout.z_start()..]);
        for (i, zterm) in zterms.iter().enumerate() {
            mu0 += mean_response(&coef, layout, control, &w, i) + zterm;
            mu1 += mean_response(&coef, layout, active, &w, i) + zterm;
        }
        mu0 /= n_f;
        mu1 /= n_f;

        let mut z_bar = vec![0.0; n_z];
        for (k, zmean) in z_bar.iter_mut().enumerate() {
            let base = (layout.z_start() + k) * n;
            *zmean = design[base..base + n].iter().sum::<f64>() / n_f;
        }
        let fit = InteractionFit {
            coef: &coef,
            inv: &inv,
            design: &design,
            residuals: &residuals,
            w: &w,
            z_bar: &z_bar,
            n,
            layout,
        };
        let if0 = plugin_arm_influence(&fit, control, mu0);
        let if1 = plugin_arm_influence(&fit, active, mu1);
        let mut row_index = Vec::with_capacity(n);
        for (i, &keep) in row_mask.iter().enumerate() {
            if keep {
                row_index.push(antecedent_data::GlobalId::new(u64::try_from(i).map_err(|_| {
                        EstimationError::data_msg(
                            "address the row by GlobalId and keep ChunkIndex inside one bounded chunk. A u32 backend feature flag is not that attempt",
                        )
                    })?));
            }
        }
        let scores = ConditionalArmScores {
            means: [mu0, mu1],
            influence: [if0, if1],
            row_index,
            treatment: t.into_owned(),
        };
        let mut assumptions = AssumptionSet::default();
        if m > 1 {
            assumptions.push(joint_modifier_record());
        }

        Ok((
            EffectEstimate::new(point, se_analytic, assumptions, OverlapPolicy::ExplicitOverride)
                .with_preparation(Some(crate::prepare::preparation_summary(&receipt)))
                .with_influence(Some(Arc::from(influence))),
            [mu0, mu1],
            scores,
        ))
    }
}

/// Column layout of the interaction design `[1, T, W_1..W_m, T×W_1..T×W_m, Z_1..Z_nz]`: the
/// one-modifier design is `[1, T, W, T×W, Z..]`.
#[derive(Clone, Copy, Debug)]
struct InteractionLayout {
    /// Number of effect modifiers.
    m: usize,
    /// Number of further adjustment covariates.
    n_z: usize,
}

impl InteractionLayout {
    /// Total design columns.
    const fn ncols(self) -> usize {
        2 + 2 * self.m + self.n_z
    }
    /// Column of the treatment-by-modifier product of modifier `k`.
    const fn interaction(self, k: usize) -> usize {
        2 + self.m + k
    }
    /// First adjustment-covariate column.
    const fn z_start(self) -> usize {
        2 + 2 * self.m
    }
}

/// Fitted mean response of row `row` at treatment `level` before the adjustment-covariate
/// terms: `β_0 + β_T·level + Σ_k β_{W_k}·w_k + Σ_k β_{T×W_k}·level·w_k`, accumulated left to
/// right.
fn mean_response(
    coef: &[f64],
    layout: InteractionLayout,
    level: f64,
    w: &[&[f64]],
    row: usize,
) -> f64 {
    let mut value = coef[0] + coef[1] * level;
    for (k, column) in w.iter().enumerate() {
        value += coef[2 + k] * column[row];
    }
    for (k, column) in w.iter().enumerate() {
        value += coef[layout.interaction(k)] * level * column[row];
    }
    value
}

/// Sampling variance of `Σ_k c_k · w̄_k` for the modifier means: with one modifier the
/// `c² · Var(w̄)` product, otherwise the variance of the weighted mean of the centred rows.
fn modifier_mean_variance(terms: &[f64], w: &[&[f64]], w_bar: &[f64]) -> f64 {
    let n = w.first().map_or(0, |column| column.len()) as f64;
    if let ([term], [column], [mean]) = (terms, w, w_bar) {
        let w_var_of_mean =
            column.iter().map(|wi| (wi - mean).powi(2)).sum::<f64>() / (n * (n - 1.0));
        return term * term * w_var_of_mean;
    }
    let rows = w.first().map_or(0, |column| column.len());
    let centred_square_sum: f64 = (0..rows)
        .map(|i| {
            let weighted: f64 = terms
                .iter()
                .zip(w)
                .zip(w_bar)
                .map(|((c, column), mean)| c * (column[i] - mean))
                .sum();
            weighted * weighted
        })
        .sum();
    centred_square_sum / (n * (n - 1.0))
}

/// The fitted `Y ~ 1 + T + W + T×W + Z` interaction model an arm's plug-in influence is
/// formed from.
struct InteractionFit<'a> {
    coef: &'a [f64],
    /// `(XᵀX)⁻¹`, row-major `ncols × ncols`.
    inv: &'a [f64],
    /// Column-major design.
    design: &'a [f64],
    residuals: &'a [f64],
    /// Effect-modifier values, one column per modifier.
    w: &'a [&'a [f64]],
    /// Column means of the `Z` covariates.
    z_bar: &'a [f64],
    n: usize,
    layout: InteractionLayout,
}

fn plugin_arm_influence(fit: &InteractionFit<'_>, level: f64, mean: f64) -> Vec<f64> {
    let InteractionFit { coef, inv, design, residuals, w, z_bar, n, layout } = *fit;
    let (n_z, ncols) = (layout.n_z, layout.ncols());
    let n_f = n as f64;
    let mut xbar = vec![0.0; ncols];
    xbar[0] = 1.0;
    xbar[1] = level;
    for (k, column) in w.iter().enumerate() {
        let w_bar: f64 = column.iter().sum::<f64>() / n_f;
        xbar[2 + k] = w_bar;
        xbar[layout.interaction(k)] = level * w_bar;
    }
    for (k, &zmean) in z_bar.iter().enumerate() {
        xbar[layout.z_start() + k] = zmean;
    }
    // Column-outer: every row adds its terms in the order a row-by-row walk would, from the
    // same starting value, so each score is bit-identical to that walk.
    let mut mu: Vec<f64> = (0..n).map(|i| mean_response(coef, layout, level, w, i)).collect();
    for k in 0..n_z {
        let column = &design[(layout.z_start() + k) * n..(layout.z_start() + k + 1) * n];
        for (slot, &x) in mu.iter_mut().zip(column) {
            *slot += coef[layout.z_start() + k] * x;
        }
    }
    let mut xbar_inv_x = vec![0.0; n];
    for (a, &weight) in xbar.iter().enumerate() {
        let inv_x = linear_predictors(design, n, ncols, &inv[a * ncols..(a + 1) * ncols]);
        for (slot, value) in xbar_inv_x.iter_mut().zip(inv_x) {
            *slot += weight * value;
        }
    }
    mu.iter()
        .zip(&xbar_inv_x)
        .zip(residuals)
        .map(|((mu_i, leverage), residual)| (mu_i - mean) + n_f * leverage * residual)
        .collect()
}

/// `(XᵀX)⁻¹` from one Gram of `design`. The Cholesky factor supplies the inverse
/// when it exists; the same Gram uses Gauss–Jordan when that factor is refused.
fn gram_inverse(design: &[f64], nrows: usize, ncols: usize) -> Option<Vec<f64>> {
    let mut xtx = vec![0.0; ncols.saturating_mul(ncols)];
    antecedent_stats::form_xtx(design, nrows, ncols, &mut xtx);
    if let Some(chol) = antecedent_stats::cholesky_spd(&xtx, ncols) {
        Some(antecedent_kernels::chol_inverse(&chol, ncols))
    } else {
        antecedent_stats::invert_square(&xtx, ncols)
    }
}

/// `Σ_a g_a · (inv · x_i)_a` of every row `x_i` of the column-major `n × ncols` design: the
/// row's leverage along the contrast `g` under the inverse Gram `inv` (row-major).
/// Components with `g_a == 0` contribute nothing and are skipped.
fn inverse_weighted_leverage(
    inv: &[f64],
    design: &[f64],
    n: usize,
    ncols: usize,
    g: &[f64],
) -> Vec<f64> {
    let mut leverage = vec![0.0; n];
    for (a, &weight) in g.iter().enumerate().take(ncols) {
        if weight == 0.0 {
            continue;
        }
        let inv_x = linear_predictors(design, n, ncols, &inv[a * ncols..(a + 1) * ncols]);
        for (slot, value) in leverage.iter_mut().zip(inv_x) {
            *slot += weight * value;
        }
    }
    leverage
}

fn binary_zero_one(query: &AverageEffectQuery) -> Result<bool, EstimationError> {
    let active = intervention_f64(&query.active)?;
    let control = intervention_f64(&query.control)?;
    Ok((active - 1.0).abs() <= 1e-12 && control.abs() <= 1e-12)
}

/// Declared record that a binary-treatment conditional fit ran the cross-fitted AIPW estimator
/// rather than the linear interaction model.
fn crossfit_substitution_record() -> AssumptionRecord {
    AssumptionRecord {
        assumption: Assumption::Custom {
            id: Arc::from("conditional.crossfit_aipw_substitution"),
            description: Arc::from(
                "For a 0/1 treatment the conditional arm means and contrast are the cross-fitted AIPW scores (five folds, logistic propensity clipped at the reported bound, per-arm OLS outcome models), not the linear interaction model. Consistency needs a correct propensity or outcome model and positivity; the reported overlap describes the propensity clipping applied.",
            ),
        },
        source: AssumptionSource::AlgorithmDefault {
            algorithm: Arc::from("estimate.conditional.aipw"),
        },
        scope: AssumptionScope::Estimation,
        status: AssumptionStatus::Declared,
    }
}

/// Declared record that a modifier-set interval is the exact vector generalization of the
/// one-modifier model, with coverage measured for one modifier only.
fn joint_modifier_record() -> AssumptionRecord {
    AssumptionRecord {
        assumption: Assumption::Custom {
            id: Arc::from("conditional.joint_modifier_interval_uncalibrated"),
            description: Arc::from(
                "The conditional contrast conditions on a set of several effect modifiers. The regression, delta-method standard error and influence functions are the vector generalization of the one-modifier model, but the interval's coverage has been measured for one modifier only, so it is not a calibrated coverage claim for a modifier set.",
            ),
        },
        source: AssumptionSource::AlgorithmDefault {
            algorithm: Arc::from("estimate.conditional.joint_modifiers"),
        },
        scope: AssumptionScope::Estimation,
        status: AssumptionStatus::Declared,
    }
}

fn aipw_conditional_arm_scores(
    data: &TabularData,
    estimand: &IdentifiedEstimand,
    query: &AverageEffectQuery,
    fold_seed: u64,
) -> Result<
    (ConditionalArmScores, OverlapPolicy, Option<crate::overlap::OverlapReport>),
    EstimationError,
> {
    if query.effect_modifiers.is_empty() {
        return Err(EstimationError::unsupported_coded(
            antecedent_core::reason_code!("required_option_missing"),
            "AIPW conditional scores require an effect modifier",
        ));
    }
    let active = intervention_f64(&query.active)?;
    let control = intervention_f64(&query.control)?;
    // The nuisance models condition on every modifier, then on the rest of the adjustment set.
    let mut adj: Vec<VariableId> = query.effect_modifiers.to_vec();
    adj.extend(
        estimand.adjustment_set.iter().copied().filter(|z| !query.effect_modifiers.contains(z)),
    );
    let mut arena = antecedent_expr::CausalExprArena::new();
    let functional = arena.backdoor_ate(
        query.treatment,
        query.outcome,
        &adj,
        antecedent_core::Value::f64(active),
        antecedent_core::Value::f64(control),
    );
    let aipw_estimand =
        IdentifiedEstimand::backdoor("backdoor.adjustment", Arc::from(adj), functional);
    let aipw_query =
        AverageEffectQuery::with_levels(query.treatment, query.outcome, control, active)
            .with_outcome_functional(query.outcome_functional.clone());
    let overlap = crate::propensity::default_propensity_overlap();
    let mut problem = crate::propensity::prepare_propensity_problem_with_registry(
        data,
        &aipw_estimand,
        &aipw_query,
        overlap,
        None,
    )?;
    problem.fold_seed = fold_seed;
    let table = crate::crossfit_aipw::build_binary_scores(
        &problem,
        query.treatment,
        &crate::crossfit_aipw::thresholds_of(&query.outcome_functional),
        crate::crossfit_aipw::DEFAULT_AIPW_FOLDS,
        &antecedent_stats::GlmOptions::default(),
        FaerBackend,
    )?;
    if !crate::retarget::score_weighted_support(&table, &vec![1.0; table.n_rows]).overlap_ok {
        return Err(EstimationError::unsupported_coded(
            antecedent_core::reason_code!("positivity_violated"),
            "conditional AIPW arm scores require supported out-of-fold propensity overlap",
        ));
    }
    let c0 = table.column(0)?;
    let c1 = table.column(1)?;
    let n = table.n_rows as f64;
    let mu0 = c0.iter().sum::<f64>() / n;
    let mu1 = c1.iter().sum::<f64>() / n;
    let e_hat = table
        .columns
        .iter()
        .position(|c| c.arm == 1 && c.threshold.is_none())
        .and_then(|col| table.propensities.get(col * table.n_rows..(col + 1) * table.n_rows));
    let report = e_hat.map(|scores| {
        crate::propensity::propensity_overlap_report(
            &problem,
            scores,
            None,
            Some(crate::overlap::IpwTarget::Ate),
        )
    });
    Ok((
        ConditionalArmScores {
            means: [mu0, mu1],
            influence: [c0.to_vec(), c1.to_vec()],
            row_index: table.row_index.to_vec(),
            treatment: table.observed_arm.iter().map(|&a| f64::from(a)).collect(),
        },
        overlap,
        report,
    ))
}

#[cfg(test)]
mod tests {
    use antecedent_core::{
        AverageEffectQuery, CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet,
        ValueType, VariableId,
    };
    use antecedent_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
    };
    use antecedent_expr::IdentifiedEstimand;

    use super::*;

    #[test]
    fn conditional_ate_runs() {
        let n = 200usize;
        let mut b = CausalSchemaBuilder::new();
        for name in ["t", "y", "w"] {
            b.add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        }
        let schema = b.build().unwrap();
        let t: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 0.0 } else { 1.0 }).collect();
        let w: Vec<f64> = (0..n).map(|i| (i % 5) as f64).collect();
        let y: Vec<f64> =
            t.iter().zip(w.iter()).map(|(&ti, &wi)| 1.0 + 2.0 * ti + 0.5 * ti * wi).collect();
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(t),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    Arc::from(y),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(2),
                    Arc::from(w),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        let data = TabularData::new(storage);
        let q = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
            .with_effect_modifiers([VariableId::from_raw(2)]);
        let cq = ConditionalEffectQuery::try_new(q).unwrap();
        let estimand = IdentifiedEstimand::backdoor(
            "backdoor.adjustment",
            Arc::from([]),
            antecedent_expr::ExprId::from_raw(0),
        );
        let est = ConditionalLinearAdjustment::new()
            .estimate(&data, &estimand, &cq, &ExecutionContext::for_tests(2))
            .unwrap();
        // True ATE at mean W≈2: 2 + 0.5*2 = 3
        assert!((est.ate - 3.0).abs() < 0.3);
        // Noiseless design → analytic SE ≈ 0 but must not claim exact certainty via a hard 0
        // when noise is present; with this noiseless fit SE is 0 or NaN-free.
        assert!(est.se_analytic().is_finite());
    }

    #[test]
    fn conditional_ate_se_positive_with_noise() {
        let n = 200usize;
        let mut b = CausalSchemaBuilder::new();
        for name in ["t", "y", "w"] {
            b.add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        }
        let schema = b.build().unwrap();
        let t: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 0.0 } else { 1.0 }).collect();
        let w: Vec<f64> = (0..n).map(|i| (i % 5) as f64).collect();
        let y: Vec<f64> = t
            .iter()
            .zip(w.iter())
            .enumerate()
            .map(|(i, (&ti, &wi))| 1.0 + 2.0 * ti + 0.5 * ti * wi + 0.4 * ((i % 7) as f64 - 3.0))
            .collect();
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(t),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    Arc::from(y),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(2),
                    Arc::from(w),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        let data = TabularData::new(storage);
        let q = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
            .with_effect_modifiers([VariableId::from_raw(2)]);
        let cq = ConditionalEffectQuery::try_new(q).unwrap();
        let estimand = IdentifiedEstimand::backdoor(
            "backdoor.adjustment",
            Arc::from([]),
            antecedent_expr::ExprId::from_raw(0),
        );
        let est = ConditionalLinearAdjustment::new()
            .estimate(&data, &estimand, &cq, &ExecutionContext::for_tests(2))
            .unwrap();
        assert!(
            est.se_analytic().is_finite() && est.se_analytic() > 0.0,
            "se={}",
            est.se_analytic()
        );
    }

    #[test]
    fn conditional_effect_matches_pinned_statsmodels_oracle() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../conformance/estimate/conditional_effects/expected.json"
        ))
        .unwrap();
        let n = fixture["data"]["n"].as_u64().unwrap() as usize;
        let mut builder = CausalSchemaBuilder::new();
        for name in ["t", "y", "w"] {
            builder
                .add_variable(
                    name,
                    ValueType::Continuous,
                    SmallRoleSet::from_hint(RoleHint::Context),
                    None,
                    None,
                    MeasurementSpec::default(),
                )
                .unwrap();
        }
        let schema = builder.build().unwrap();
        let t: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
        let w: Vec<f64> = (0..n).map(|i| (i % 7) as f64 - 3.0).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let ti = t[i];
                let wi = w[i];
                1.2 + 1.8 * ti - 0.4 * wi
                    + 0.65 * ti * wi
                    + 0.25 * (0.31 * i as f64).sin()
                    + 0.1 * (0.17 * i as f64).cos()
            })
            .collect();
        let columns = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(t),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    Arc::from(y),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(2),
                    Arc::from(w),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let data =
            TabularData::new(OwnedColumnarStorage::try_new(schema, columns, None, None).unwrap());
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
                .with_effect_modifiers([VariableId::from_raw(2)]);
        let conditional = ConditionalEffectQuery::try_new(query).unwrap();
        let estimand = IdentifiedEstimand::backdoor(
            "backdoor.adjustment",
            Arc::from([]),
            antecedent_expr::ExprId::from_raw(0),
        );
        let actual = ConditionalLinearAdjustment::new()
            .estimate(&data, &estimand, &conditional, &ExecutionContext::for_tests(2))
            .unwrap();
        let tolerance = fixture["acceptance"]["atol"].as_f64().unwrap();
        assert!(
            (actual.ate - fixture["reference"]["ate_at_modifier_mean"].as_f64().unwrap()).abs()
                <= tolerance
        );
        // The statsmodels oracle is the coefficient delta-method SE at the
        // observed modifier mean (w̄ held fixed). The reported functional
        // averages over the modifier distribution, so its SE adds the sampling
        // variance of w̄, δ²·β_{T×W}²·Σ(w−w̄)²/(n(n−1)); remove it and compare.
        let coefficients = fixture["reference"]["coefficients"].as_array().unwrap();
        let b_tw = coefficients[3].as_f64().unwrap();
        let w_bar = fixture["reference"]["modifier_mean"].as_f64().unwrap();
        let w_var_of_mean = (0..n).map(|i| ((i % 7) as f64 - 3.0 - w_bar).powi(2)).sum::<f64>()
            / (n as f64 * (n as f64 - 1.0));
        let coefficient_part = (actual.se_analytic().powi(2) - b_tw * b_tw * w_var_of_mean).sqrt();
        assert!(
            (coefficient_part - fixture["reference"]["analytic_se"].as_f64().unwrap()).abs()
                <= tolerance,
            "coefficient-part SE {coefficient_part} vs statsmodels oracle"
        );
        assert!(actual.se_analytic() > coefficient_part, "the modifier-mean term must be added");
    }

    /// Both one-modifier refusals keep their message and name the
    /// multi-modifier routes only when several modifiers were asked for.
    #[test]
    fn multi_modifier_refusals_name_the_bayesian_and_handoff_routes() {
        let column: Vec<f64> = (0..16).map(f64::from).collect();
        let data = TabularData::from_f64_columns([
            ("t", column.as_slice()),
            ("y", column.as_slice()),
            ("w", column.as_slice()),
            ("v", column.as_slice()),
        ])
        .unwrap();
        let estimand = IdentifiedEstimand::backdoor(
            "backdoor.adjustment",
            Arc::from([]),
            antecedent_expr::ExprId::from_raw(0),
        );
        let two = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
            .with_effect_modifiers([VariableId::from_raw(2), VariableId::from_raw(3)]);
        let mut estimator =
            ConditionalLinearAdjustment::new().with_overlap(OverlapPolicy::ExplicitOverride);
        estimator.joint_modifiers = false;
        let error = estimator.estimate_ate(&data, &estimand, &two).unwrap_err();
        assert_eq!(
            error.to_string(),
            "ConditionalLinearAdjustment currently supports one effect modifier"
        );
        assert_eq!(error.remedy(), Some(MULTI_MODIFIER_REMEDY));
        let error = estimator
            .estimate_with_arm_scores(
                &data,
                &estimand,
                &ConditionalEffectQuery::try_new(two.clone()).unwrap(),
            )
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            "conditional arm scores require AllObserved and exactly one modifier"
        );
        assert_eq!(error.remedy(), Some(MULTI_MODIFIER_REMEDY));
        assert!(MULTI_MODIFIER_REMEDY.contains("EstimatorId::BayesianConditional"));
        assert!(MULTI_MODIFIER_REMEDY.contains("capability: adjustment handoff"));
        assert!(!MULTI_MODIFIER_REMEDY.contains("antecedent.handoff.econml"));
    }

    /// The default estimator accepts a modifier set without a declaration.
    #[test]
    fn the_default_estimator_accepts_a_modifier_set() {
        assert!(ConditionalLinearAdjustment::new().joint_modifiers);
        assert!(ConditionalLinearAdjustment::default().joint_modifiers);
    }

    /// A modifier set fits the vector interaction model: the point estimate is the
    /// effect at the modifier means, the interval is flagged uncalibrated, and the
    /// one-modifier route is unchanged.
    #[test]
    fn a_declared_modifier_set_recovers_the_effect_at_the_modifier_means() {
        let n = 60usize;
        let t: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
        let w: Vec<f64> = (0..n).map(|i| ((i * 7) % 11) as f64 / 5.0).collect();
        let v: Vec<f64> = (0..n).map(|i| ((i * 3) % 7) as f64 / 3.0).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| 1.0 + 2.0 * t[i] + 0.5 * w[i] + 0.7 * v[i] + t[i] * (0.3 * w[i] + 0.4 * v[i]))
            .collect();
        let data = TabularData::from_f64_columns([
            ("t", t.as_slice()),
            ("y", y.as_slice()),
            ("w", w.as_slice()),
            ("v", v.as_slice()),
        ])
        .unwrap();
        let estimand = IdentifiedEstimand::backdoor(
            "backdoor.adjustment",
            Arc::from([]),
            antecedent_expr::ExprId::from_raw(0),
        );
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
                .with_effect_modifiers([VariableId::from_raw(2), VariableId::from_raw(3)]);
        let estimator = ConditionalLinearAdjustment::new().with_joint_modifiers();
        let (estimate, means, scores) = estimator.estimate_ate(&data, &estimand, &query).unwrap();
        let w_bar = w.iter().sum::<f64>() / n as f64;
        let v_bar = v.iter().sum::<f64>() / n as f64;
        let expected = 2.0 + 0.3 * w_bar + 0.4 * v_bar;
        assert!((estimate.ate - expected).abs() < 1e-8, "{} vs {expected}", estimate.ate);
        assert!((means[1] - means[0] - expected).abs() < 1e-8);
        assert_eq!(scores.influence[0].len(), n);
        assert!(estimate.se_analytic() > 0.0, "the modifier-mean term is a sampling variance");
        assert!(
            estimate.provenance.assumptions.entries.iter().any(|record| matches!(
                &record.assumption,
                Assumption::Custom { id, .. }
                    if id.as_ref() == "conditional.joint_modifier_interval_uncalibrated"
            )),
            "the modifier-set interval names itself uncalibrated"
        );
    }

    #[test]
    fn arm_plugin_ifs_are_finite_and_contrast_like() {
        let n = 80usize;
        let mut b = CausalSchemaBuilder::new();
        for name in ["t", "y", "w"] {
            b.add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        }
        let schema = b.build().unwrap();
        let t: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 0.0 } else { 1.0 }).collect();
        let w: Vec<f64> = (0..n).map(|i| (i % 5) as f64).collect();
        let y: Vec<f64> = t
            .iter()
            .zip(w.iter())
            .enumerate()
            .map(|(i, (&ti, &wi))| 1.0 + 2.0 * ti + 0.5 * ti * wi + 0.2 * ((i % 3) as f64))
            .collect();
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(t),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(1),
                    Arc::from(y),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(2),
                    Arc::from(w),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
        ];
        let data =
            TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
        let q = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
            .with_effect_modifiers([VariableId::from_raw(2)]);
        let cq = ConditionalEffectQuery::try_new(q).unwrap();
        let estimand = IdentifiedEstimand::backdoor(
            "backdoor.adjustment",
            Arc::from([]),
            antecedent_expr::ExprId::from_raw(0),
        );
        let (estimate, scores) = ConditionalLinearAdjustment::new()
            .estimate_with_arm_scores(&data, &estimand, &cq)
            .unwrap();
        assert_eq!(scores.influence[0].len(), n);
        assert!(estimate.diagnostics.overlap_report.is_some(), "overlap report missing");
        assert_eq!(estimate.provenance.overlap, crate::propensity::default_propensity_overlap());
        let substituted = crossfit_substitution_record();
        assert!(estimate.provenance.assumptions.entries.iter().any(|r| *r == substituted));
        assert!(scores.influence[0].iter().all(|v| v.is_finite()));
        assert!(scores.influence[1].iter().all(|v| v.is_finite()));
        let contrast: Vec<f64> =
            scores.influence[0].iter().zip(&scores.influence[1]).map(|(a, b)| b - a).collect();
        assert!(contrast.iter().any(|v| v.abs() > 1e-8));
    }

    /// The row-by-row loops the column-outer kernels replaced.
    fn reference_plugin_arm_influence(fit: &InteractionFit<'_>, level: f64, mean: f64) -> Vec<f64> {
        let InteractionFit { coef, inv, design, residuals, w, z_bar, n, layout } = *fit;
        let (ncols, n_z) = (layout.ncols(), layout.n_z);
        let w = w[0];
        let n_f = n as f64;
        let mut xbar = vec![0.0; ncols];
        xbar[0] = 1.0;
        xbar[1] = level;
        let w_bar: f64 = w.iter().sum::<f64>() / n_f;
        xbar[2] = w_bar;
        xbar[3] = level * w_bar;
        for (k, &zmean) in z_bar.iter().enumerate() {
            xbar[4 + k] = zmean;
        }
        let mut out = vec![0.0; n];
        for i in 0..n {
            let mut mu_i = coef[0] + coef[1] * level + coef[2] * w[i] + coef[3] * level * w[i];
            for k in 0..n_z {
                mu_i += coef[4 + k] * design[(4 + k) * n + i];
            }
            let mut xbar_inv_x = 0.0;
            for a in 0..ncols {
                let mut inv_x = 0.0;
                for b in 0..ncols {
                    inv_x += inv[a * ncols + b] * design[b * n + i];
                }
                xbar_inv_x += xbar[a] * inv_x;
            }
            out[i] = (mu_i - mean) + n_f * xbar_inv_x * residuals[i];
        }
        out
    }

    #[test]
    fn column_outer_scores_are_bit_identical_to_the_row_first_loops() {
        let (n, n_z) = (1_111usize, 3usize);
        let ncols = 4 + n_z;
        let mut state = 0x0DDB_1A5E_5BAD_5EED_u64;
        let mut draw = || {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            (state >> 33) as f64 / 2f64.powi(31) - 0.5
        };
        let w: Vec<f64> = (0..n).map(|_| draw()).collect();
        let t: Vec<f64> = (0..n).map(|i| f64::from(u8::from(i % 2 == 0))).collect();
        let mut design = vec![0.0; n * ncols];
        for i in 0..n {
            design[i] = 1.0;
            design[n + i] = t[i];
            design[2 * n + i] = w[i];
            design[3 * n + i] = t[i] * w[i];
        }
        for k in 0..n_z {
            for i in 0..n {
                design[(4 + k) * n + i] = draw();
            }
        }
        let y: Vec<f64> = (0..n).map(|i| 1.0 + 0.5 * t[i] + draw()).collect();
        let coef: Vec<f64> = (0..ncols).map(|j| 0.4 - 0.13 * j as f64).collect();
        let inv: Vec<f64> = (0..ncols * ncols)
            .map(|k| if k / ncols == k % ncols { 0.01 } else { 0.001 * draw() })
            .collect();
        let g = {
            let mut g = vec![0.0; ncols];
            g[1] = 0.7;
            g[3] = -0.2;
            g
        };

        let mut residuals = vec![0.0; n];
        for i in 0..n {
            let mut pred = 0.0;
            for j in 0..ncols {
                pred += coef[j] * design[j * n + i];
            }
            residuals[i] = y[i] - pred;
        }
        let new_residuals: Vec<f64> = linear_predictors(&design, n, ncols, &coef)
            .iter()
            .zip(&y)
            .map(|(pred, y)| y - pred)
            .collect();
        assert!(residuals.iter().zip(&new_residuals).all(|(a, b)| a.to_bits() == b.to_bits()));

        let leverage = inverse_weighted_leverage(&inv, &design, n, ncols, &g);
        for i in 0..n {
            let mut g_inv_x = 0.0;
            for a in 0..ncols {
                if g[a] == 0.0 {
                    continue;
                }
                let mut inv_x = 0.0;
                for b in 0..ncols {
                    inv_x += inv[a * ncols + b] * design[b * n + i];
                }
                g_inv_x += g[a] * inv_x;
            }
            assert_eq!(leverage[i].to_bits(), g_inv_x.to_bits());
        }

        let z_bar: Vec<f64> = (0..n_z)
            .map(|k| design[(4 + k) * n..(5 + k) * n].iter().sum::<f64>() / n as f64)
            .collect();
        let fit = InteractionFit {
            coef: &coef,
            inv: &inv,
            design: &design,
            residuals: &residuals,
            w: &[&w[..]],
            z_bar: &z_bar,
            n,
            layout: InteractionLayout { m: 1, n_z },
        };
        for (level, mean) in [(0.0, 0.3), (1.0, -0.2)] {
            let new = plugin_arm_influence(&fit, level, mean);
            let reference = reference_plugin_arm_influence(&fit, level, mean);
            assert!(new.iter().zip(&reference).all(|(a, b)| a.to_bits() == b.to_bits()));
        }
    }

    #[test]
    #[ignore = "timing harness: cargo test --release -p antecedent-estimate --lib timing_ -- --ignored --nocapture"]
    fn timing_conditional_ate_wide_adjustment() {
        let (n, n_z) = (200_000usize, 8usize);
        let mut b = CausalSchemaBuilder::new();
        for index in 0..3 + n_z {
            b.add_variable(
                format!("v{index}").as_str(),
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        }
        let schema = b.build().unwrap();
        let mut state = 0x2545_F491_4F6C_DD1D_u64;
        let mut draw = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) as f64 / 2f64.powi(31) - 0.5
        };
        let mut columns: Vec<Vec<f64>> = Vec::new();
        columns.push((0..n).map(|i| f64::from(u8::from(i % 2 == 0))).collect());
        columns.push((0..n).map(|_| draw()).collect());
        columns.push((0..n).map(|_| draw()).collect());
        for _ in 0..n_z {
            columns.push((0..n).map(|_| draw()).collect());
        }
        let cols = columns
            .into_iter()
            .enumerate()
            .map(|(index, values)| {
                OwnedColumn::Float64(
                    Float64Column::new(
                        VariableId::from_raw(u32::try_from(index).unwrap()),
                        Arc::from(values),
                        ValidityBitmap::all_valid(n),
                    )
                    .unwrap(),
                )
            })
            .collect();
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        let data = TabularData::new(storage);
        let q = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
            .with_effect_modifiers([VariableId::from_raw(2)]);
        let cq = ConditionalEffectQuery::try_new(q).unwrap();
        let adjustment: Vec<VariableId> =
            (3..3 + n_z).map(|index| VariableId::from_raw(u32::try_from(index).unwrap())).collect();
        let estimand = IdentifiedEstimand::backdoor(
            "backdoor.adjustment",
            Arc::from(adjustment),
            antecedent_expr::ExprId::from_raw(0),
        );
        let est = ConditionalLinearAdjustment::new();
        let ctx = ExecutionContext::for_tests(2);
        let start = std::time::Instant::now();
        let reps = 5;
        for _ in 0..reps {
            std::hint::black_box(est.estimate(&data, &estimand, &cq, &ctx).unwrap());
        }
        println!(
            "conditional estimate n={n} n_z={n_z}: {:.3} ms/call",
            start.elapsed().as_secs_f64() * 1e3 / f64::from(reps)
        );
    }
}
