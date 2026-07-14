//! Conditional ATE with effect modifiers (Phase 9 / `dowhy.estimate.conditional`).
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

use causal_core::{
    AssumptionSet, AverageEffectQuery, ConditionalEffectQuery, ExecutionContext, TargetPopulation,
};
use causal_data::TabularData;
use causal_expr::IdentifiedEstimand;
use causal_stats::{FaerBackend, form_xtx, invert_square};

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
        query.validate().map_err(|e| EstimationError::UnsupportedQuery(e.to_string()))?;
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
            return Err(EstimationError::UnsupportedQuery(
                "ConditionalLinearAdjustment requires effect modifiers".into(),
            ));
        }
        if query.effect_modifiers.len() != 1 {
            return Err(EstimationError::UnsupportedQuery(
                "ConditionalLinearAdjustment currently supports one effect modifier".into(),
            ));
        }
        if query.target_population != TargetPopulation::AllObserved {
            return Err(EstimationError::UnsupportedQuery(
                "ConditionalLinearAdjustment only supports AllObserved".into(),
            ));
        }
        if estimand.method_kind().ok() != Some(causal_expr::EstimandMethod::BackdoorAdjustment) {
            return Err(EstimationError::IncompatibleEstimand {
                message: "ConditionalLinearAdjustment expects backdoor.adjustment",
            });
        }
        let active = intervention_f64(&query.active)?;
        let control = intervention_f64(&query.control)?;
        let delta = active - control;
        if delta == 0.0 {
            return Err(EstimationError::UnsupportedQuery(
                "active and control treatment levels must differ".into(),
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

        let mut xtx = vec![0.0; ncols * ncols];
        let mut xty = vec![0.0; ncols];
        form_xtx(&design, n, ncols, &mut xtx);
        for c in 0..ncols {
            let col = &design[c * n..(c + 1) * n];
            xty[c] = col.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        }
        let inv = invert_square(&xtx, ncols)
            .ok_or_else(|| EstimationError::stats_msg("singular design in conditional ATE"))?;
        let mut coef = vec![0.0; ncols];
        for i in 0..ncols {
            let mut s = 0.0;
            for j in 0..ncols {
                s += inv[i * ncols + j] * xty[j];
            }
            coef[i] = s;
        }

        let w_bar: f64 = w.iter().sum::<f64>() / n as f64;
        // Marginal ATE at mean W: (β_T + β_{TW} * Ē[W]) * delta
        let point = (coef[1] + coef[3] * w_bar) * delta;
        let _ = self.backend;
        let _ = Arc::clone(&estimand.method);

        Ok(EffectEstimate {
            ate: point,
            se_analytic: 0.0,
            se_bootstrap: None,
            assumptions: AssumptionSet::default(),
            overlap: OverlapPolicy::ExplicitOverride,
            overlap_report: None,
            retained_memory_bytes: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use causal_core::{
        AverageEffectQuery, CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet,
        ValueType, VariableId,
    };
    use causal_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
    };
    use causal_expr::IdentifiedEstimand;

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
            causal_expr::ExprId::from_raw(0),
        );
        let est = ConditionalLinearAdjustment::new()
            .estimate(&data, &estimand, &cq, &ExecutionContext::for_tests(2))
            .unwrap();
        // True ATE at mean W≈2: 2 + 0.5*2 = 3
        assert!((est.ate - 3.0).abs() < 0.3);
    }
}
