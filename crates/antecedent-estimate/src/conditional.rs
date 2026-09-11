//! Conditional ATE with effect modifiers.
//!
//! Fits `Y ~ 1 + T + W + T×W` and reports the average treatment effect
//! marginalized over observed modifier values:
//! `ATE = (β_T + β_{T×W} · Ē[W]) · (active − control)` for a single modifier.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    clippy::similar_names
)]

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, ConditionalEffectQuery, ExecutionContext, TargetPopulation,
    VariableId,
};
use antecedent_data::TabularData;
use antecedent_expr::IdentifiedEstimand;
use antecedent_stats::{DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace};

use crate::adjustment::{EffectEstimate, intervention_f64};
use crate::error::EstimationError;
use crate::overlap::OverlapPolicy;
use crate::util::require_explicit_override;

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
    pub row_index: Vec<u32>,
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
        Self { overlap: OverlapPolicy::ExplicitOverride, backend: FaerBackend }
    }

    /// Set the overlap policy. Must remain [`OverlapPolicy::ExplicitOverride`].
    #[must_use]
    pub const fn with_overlap(mut self, overlap: OverlapPolicy) -> Self {
        self.overlap = overlap;
        self
    }

    /// Set the dense linear-algebra backend used for the interaction-model OLS fit.
    #[must_use]
    pub const fn with_backend(mut self, backend: FaerBackend) -> Self {
        self.backend = backend;
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
    /// under back-door, positivity, and nuisance rates). Non-binary levels keep
    /// the linear plugin IF and are labeled as such.
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
        if query.inner.target_population != TargetPopulation::AllObserved
            || query.inner.effect_modifiers.len() != 1
        {
            return Err(EstimationError::unsupported(
                "conditional arm scores require AllObserved and exactly one modifier",
            ));
        }
        if !matches!(query.inner.outcome_functional, antecedent_core::OutcomeFunctional::Mean) {
            return Err(EstimationError::unsupported(
                "conditional arm scores require an already transformed outcome and the Mean functional",
            ));
        }
        if binary_zero_one(&query.inner)? {
            let scores = aipw_conditional_arm_scores(data, estimand, &query.inner)?;
            let n = scores.influence[0].len() as f64;
            let contrast: Vec<f64> =
                scores.influence[1].iter().zip(&scores.influence[0]).map(|(a, b)| a - b).collect();
            let mean = contrast.iter().sum::<f64>() / n;
            let var = contrast.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
            let estimate = EffectEstimate::new(
                scores.means[1] - scores.means[0],
                (var / n).sqrt(),
                AssumptionSet::default(),
                OverlapPolicy::ExplicitOverride,
            )
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
    #[allow(clippy::too_many_lines)]
    pub fn estimate_ate(
        &self,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        query: &AverageEffectQuery,
    ) -> Result<(EffectEstimate, [f64; 2], ConditionalArmScores), EstimationError> {
        if query.outcome_functional.quantile_level().is_some() {
            return Err(EstimationError::unsupported(
                "conditional quantiles require CDF-grid inversion through Study; a mean regression cannot estimate a quantile",
            ));
        }
        if query.effect_modifiers.is_empty() {
            return Err(EstimationError::unsupported(
                "ConditionalLinearAdjustment requires effect modifiers",
            ));
        }
        if query.effect_modifiers.len() != 1 {
            return Err(EstimationError::unsupported(
                "ConditionalLinearAdjustment currently supports one effect modifier",
            ));
        }
        if query.target_population != TargetPopulation::AllObserved {
            return Err(EstimationError::unsupported(
                "ConditionalLinearAdjustment only supports AllObserved",
            ));
        }
        if estimand.method_kind().ok() != Some(antecedent_expr::EstimandMethod::BackdoorAdjustment)
        {
            return Err(EstimationError::IncompatibleEstimand {
                message: "ConditionalLinearAdjustment expects backdoor.adjustment",
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

        let w_id = query.effect_modifiers[0];
        let extra_z: Vec<VariableId> =
            estimand.adjustment_set.iter().copied().filter(|&z| z != w_id).collect();
        let mut ids = vec![query.treatment, query.outcome, w_id];
        ids.extend_from_slice(&extra_z);
        let row_mask = data.complete_case_mask(&ids).map_err(EstimationError::from)?;
        let t = data.float64_masked(query.treatment, &row_mask).map_err(EstimationError::from)?;
        let y = data.float64_masked(query.outcome, &row_mask).map_err(EstimationError::from)?;
        let w = data.float64_masked(w_id, &row_mask).map_err(EstimationError::from)?;
        let n = t.len();
        if n < 8 {
            return Err(EstimationError::data_msg("too few complete rows for conditional ATE"));
        }

        // Design: [1, T, W, T*W, Z...]; skip Z that is already the modifier.
        let n_z = extra_z.len();
        let ncols = 4 + n_z;
        let mut design = vec![0.0; n * ncols];
        for i in 0..n {
            design[i] = 1.0;
            design[n + i] = t[i];
            design[2 * n + i] = w[i];
            design[3 * n + i] = t[i] * w[i];
        }
        for (k, &z) in extra_z.iter().enumerate() {
            let zcol = data.float64_masked(z, &row_mask).map_err(EstimationError::from)?;
            let base = (4 + k) * n;
            design[base..base + n].copy_from_slice(&zcol);
        }

        let mut ws = LeastSquaresWorkspace::default();
        let fit = self
            .backend
            .least_squares(&design, n, ncols, &y, &mut ws)
            .map_err(crate::util::stats_err)?;
        let coef = fit.coefficients;

        let inv = crate::util::xtx_inverse(&design, n, ncols)
            .ok_or_else(|| EstimationError::stats_msg("singular design in conditional ATE"))?;

        let w_bar: f64 = w.iter().sum::<f64>() / n as f64;
        // Marginal ATE at mean W: (β_T + β_{TW} * Ē[W]) * delta
        let point = (coef[1] + coef[3] * w_bar) * delta;

        // Delta-method SE with g = δ·(e_T + w̄·e_{T×W}), treating w̄ as fixed.
        let sigma2 = crate::util::ols_sigma2(&design, n, ncols, &y, &coef);
        let mut g = vec![0.0; ncols];
        g[1] = delta;
        g[3] = delta * w_bar;
        let se_analytic = crate::util::delta_method_se(&inv, ncols, &g, sigma2);

        let mut residuals = vec![0.0; n];
        for i in 0..n {
            let mut pred = 0.0;
            for j in 0..ncols {
                pred += coef[j] * design[j * n + i];
            }
            residuals[i] = y[i] - pred;
        }
        let n_f = n as f64;
        let mut influence = vec![0.0; n];
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
            influence[i] = n_f * g_inv_x * residuals[i];
        }

        let _ = Arc::clone(&estimand.method);

        let mut mu0 = 0.0;
        let mut mu1 = 0.0;
        for i in 0..n {
            let mut zterm = 0.0;
            for k in 0..n_z {
                zterm += coef[4 + k] * design[(4 + k) * n + i];
            }
            mu0 += coef[0] + coef[1] * control + coef[2] * w[i] + coef[3] * control * w[i] + zterm;
            mu1 += coef[0] + coef[1] * active + coef[2] * w[i] + coef[3] * active * w[i] + zterm;
        }
        mu0 /= n_f;
        mu1 /= n_f;

        let mut z_bar = vec![0.0; n_z];
        for (k, zmean) in z_bar.iter_mut().enumerate() {
            let base = (4 + k) * n;
            *zmean = design[base..base + n].iter().sum::<f64>() / n_f;
        }
        let if0 = plugin_arm_influence(
            control, mu0, &coef, &inv, &design, &residuals, &w, &z_bar, n, ncols, n_z,
        );
        let if1 = plugin_arm_influence(
            active, mu1, &coef, &inv, &design, &residuals, &w, &z_bar, n, ncols, n_z,
        );
        let mut row_index = Vec::with_capacity(n);
        for (i, &keep) in row_mask.iter().enumerate() {
            if keep {
                row_index.push(u32::try_from(i).unwrap_or(u32::MAX));
            }
        }
        let scores = ConditionalArmScores {
            means: [mu0, mu1],
            influence: [if0, if1],
            row_index,
            treatment: t.clone(),
        };

        Ok((
            EffectEstimate::new(
                point,
                se_analytic,
                AssumptionSet::default(),
                OverlapPolicy::ExplicitOverride,
            )
            .with_influence(Some(Arc::from(influence))),
            [mu0, mu1],
            scores,
        ))
    }
}

#[allow(clippy::too_many_arguments)]
fn plugin_arm_influence(
    level: f64,
    mean: f64,
    coef: &[f64],
    inv: &[f64],
    design: &[f64],
    residuals: &[f64],
    w: &[f64],
    z_bar: &[f64],
    n: usize,
    ncols: usize,
    n_z: usize,
) -> Vec<f64> {
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

fn binary_zero_one(query: &AverageEffectQuery) -> Result<bool, EstimationError> {
    let active = intervention_f64(&query.active)?;
    let control = intervention_f64(&query.control)?;
    Ok((active - 1.0).abs() <= 1e-12 && control.abs() <= 1e-12)
}

fn aipw_conditional_arm_scores(
    data: &TabularData,
    estimand: &IdentifiedEstimand,
    query: &AverageEffectQuery,
) -> Result<ConditionalArmScores, EstimationError> {
    if query.effect_modifiers.is_empty() {
        return Err(EstimationError::unsupported(
            "AIPW conditional scores require an effect modifier",
        ));
    }
    let w_id = query.effect_modifiers[0];
    let mut adj: Vec<VariableId> =
        estimand.adjustment_set.iter().copied().filter(|&z| z != w_id).collect();
    adj.insert(0, w_id);
    let mut arena = antecedent_expr::CausalExprArena::new();
    let functional = arena.backdoor_ate(
        query.treatment,
        query.outcome,
        &adj,
        antecedent_core::Value::f64(1.0),
        antecedent_core::Value::f64(0.0),
    );
    let aipw_estimand =
        IdentifiedEstimand::backdoor("backdoor.adjustment", Arc::from(adj), functional);
    let aipw_query = AverageEffectQuery::binary_ate(query.treatment, query.outcome)
        .with_outcome_functional(query.outcome_functional.clone());
    let problem = crate::propensity::prepare_propensity_problem_with_registry(
        data,
        &aipw_estimand,
        &aipw_query,
        crate::propensity::default_propensity_overlap(),
        None,
    )?;
    let table = crate::crossfit_aipw::build_binary_scores(
        &problem,
        query.treatment,
        &crate::crossfit_aipw::thresholds_of(&query.outcome_functional),
        crate::crossfit_aipw::DEFAULT_AIPW_FOLDS,
        &antecedent_stats::GlmOptions::default(),
        FaerBackend,
    )?;
    if !crate::retarget::score_weighted_support(&table, &vec![1.0; table.n_rows]).overlap_ok {
        return Err(EstimationError::unsupported(
            "conditional AIPW arm scores require supported out-of-fold propensity overlap",
        ));
    }
    let c0 = table.column(0)?;
    let c1 = table.column(1)?;
    let n = table.n_rows as f64;
    let mu0 = c0.iter().sum::<f64>() / n;
    let mu1 = c1.iter().sum::<f64>() / n;
    Ok(ConditionalArmScores {
        means: [mu0, mu1],
        influence: [c0.to_vec(), c1.to_vec()],
        row_index: table.row_index.to_vec(),
        treatment: table.observed_arm.iter().map(|&a| f64::from(a)).collect(),
    })
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
        assert!(est.se_analytic.is_finite());
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
        assert!(est.se_analytic.is_finite() && est.se_analytic > 0.0, "se={}", est.se_analytic);
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
        assert!(
            (actual.se_analytic - fixture["reference"]["analytic_se"].as_f64().unwrap()).abs()
                <= tolerance
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
        let (_, scores) = ConditionalLinearAdjustment::new()
            .estimate_with_arm_scores(&data, &estimand, &cq)
            .unwrap();
        assert_eq!(scores.influence[0].len(), n);
        assert!(scores.influence[0].iter().all(|v| v.is_finite()));
        assert!(scores.influence[1].iter().all(|v| v.is_finite()));
        let contrast: Vec<f64> =
            scores.influence[0].iter().zip(&scores.influence[1]).map(|(a, b)| b - a).collect();
        assert!(contrast.iter().any(|v| v.abs() > 1e-8));
    }
}
