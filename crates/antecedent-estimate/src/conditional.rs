//! Conditional ATE with effect modifiers .
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
};
use antecedent_data::TabularData;
use antecedent_expr::IdentifiedEstimand;
use antecedent_stats::{DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace};

use crate::adjustment::{EffectEstimate, intervention_f64};
use crate::error::EstimationError;
use crate::overlap::OverlapPolicy;
use crate::util::require_explicit_override;

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
        self.estimate_ate(data, estimand, &query.inner)
    }

    /// Estimate from an [`AverageEffectQuery`] with non-empty modifiers.
    ///
    /// # Errors
    ///
    /// Empty modifiers or OLS failures.
    pub fn estimate_ate(
        &self,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        query: &AverageEffectQuery,
    ) -> Result<EffectEstimate, EstimationError> {
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
        let mut ids = vec![query.treatment, query.outcome, w_id];
        ids.extend_from_slice(&estimand.adjustment_set);
        let row_mask = data.complete_case_mask(&ids).map_err(EstimationError::from)?;
        let t = data.float64_masked(query.treatment, &row_mask).map_err(EstimationError::from)?;
        let y = data.float64_masked(query.outcome, &row_mask).map_err(EstimationError::from)?;
        let w = data.float64_masked(w_id, &row_mask).map_err(EstimationError::from)?;
        let n = t.len();
        if n < 8 {
            return Err(EstimationError::data_msg("too few complete rows for conditional ATE"));
        }

        // Design: [1, T, W, T*W, Z...]
        let n_z = estimand.adjustment_set.len();
        let ncols = 4 + n_z;
        let mut design = vec![0.0; n * ncols];
        for i in 0..n {
            design[i] = 1.0;
            design[n + i] = t[i];
            design[2 * n + i] = w[i];
            design[3 * n + i] = t[i] * w[i];
        }
        for (k, &z) in estimand.adjustment_set.iter().enumerate() {
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

        let _ = Arc::clone(&estimand.method);

        Ok(EffectEstimate::new(
            point,
            se_analytic,
            AssumptionSet::default(),
            OverlapPolicy::ExplicitOverride,
        ))
    }
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
}
