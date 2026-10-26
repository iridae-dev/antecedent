//! Streamed sufficient-statistics reduction for two-stage least squares.
//!
//! The design `[1 | W… | Z… | T]` is never built. Rows are read in bounded batches straight
//! from the source columns, through the complete-case mask, into the weighted-Gaussian capacity
//! engine; its moments (`XᵀX`, `Xᵀy`, `yᵀy`) are the sufficient statistics of the whole
//! problem. The treatment coefficient, the first-stage excluded-instrument test and the
//! homoskedastic Anderson–Rubin set follow in closed form from those moments
//! ([`antecedent_stats::fit_2sls_from_moments`]), so memory is one batch and a `p × p` Gram
//! instead of the `n × p` design.
//!
//! The reduction is the normal equation, so it squares the design's condition number: it
//! agrees with the materialized pivoted-QR fit to a relative tolerance set by that condition
//! number, not bit for bit. It publishes what the resident fit publishes: the instrumented
//! effect, the first-stage diagnostics and the Anderson–Rubin set, and never a Wald or
//! bootstrap standard error.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::execution_facts::OutputRetention;
use antecedent_core::physical_choice::PhysicalChoice;
use antecedent_core::{AssumptionSet, AverageEffectQuery, TargetPopulation};
use antecedent_data::materialize::{PreparationControls, PreparationProduct, prepare_table};
use antecedent_data::{ColumnView, MissingPolicy, TableView, TabularData};
use antecedent_expr::IdentifiedEstimand;
use antecedent_io_core::capacity_account::{
    CapacityControl, CapacityOptions, run_weighted_capacity,
};
use antecedent_stats::{FirstStageDiagnostics, IvMoments};

use crate::adjustment::EffectEstimate;
use crate::error::EstimationError;
use crate::iv::TwoStageLeastSquares;
use crate::overlap::OverlapPolicy;
use crate::prepare::treatment_contrast;
use crate::se::AnalyticSeKind;
use crate::streamed_rows::{MaskedRows, RowMode};

/// Level of the Anderson–Rubin set, as the resident fit publishes it.
const AR_LEVEL: f64 = 0.95;

/// Effect and diagnostics from a streamed reduction, with the facts of the run.
#[derive(Clone, Debug)]
pub struct StreamedIvEstimate {
    /// Instrumented effect: the treatment coefficient times the active − control contrast.
    pub ate: f64,
    /// Complete-case rows reduced.
    pub n_obs: u64,
    /// Second-stage coefficients: the treatment first, then the intercept and adjustment columns.
    pub coefficients: Vec<f64>,
    /// First-stage excluded-instrument test, with the Anderson–Rubin set attached (or the
    /// reason it is withheld), scaled to the published contrast.
    pub first_stage: FirstStageDiagnostics,
    /// Bytes reserved against the ceiling for this run.
    pub reserved_bytes: u64,
    /// Passes over the source rows.
    pub source_passes: u64,
    /// Batches read in the first pass.
    pub batches: u64,
    /// The executed backend choice, with the alternatives rejected.
    pub choice: PhysicalChoice,
}

impl StreamedIvEstimate {
    /// The estimate the resident fit would publish: the effect with no Wald or bootstrap
    /// standard error, the first-stage diagnostics and the licensed uncertainty set.
    #[must_use]
    pub fn into_effect(
        self,
        assumptions: AssumptionSet,
        overlap: OverlapPolicy,
        se_kind: AnalyticSeKind,
    ) -> EffectEstimate {
        EffectEstimate::new(self.ate, f64::NAN, assumptions, overlap)
            .with_n_obs(self.n_obs)
            .with_first_stage_diagnostics(Some(self.first_stage))
            .with_se_kind(se_kind)
    }
}

impl TwoStageLeastSquares {
    /// Whether this estimator's result follows in closed form from streamed sufficient
    /// statistics. Two-stage least squares publishes no standard error, so the standard-error
    /// kind only decides whether the Anderson–Rubin set is published or withheld; clustering,
    /// multiway ids and panel times are the only options the reduction cannot honour.
    #[must_use]
    pub fn supports_streamed_reduction(&self) -> bool {
        self.inference.cluster_ids.is_none()
            && self.inference.multiway_ids.is_none()
            && self.inference.panel_times.is_none()
    }

    /// Estimate the instrumented effect by streaming the rows in batches of `batch_rows`,
    /// holding at most `ceiling_bytes`, without building the design.
    ///
    /// # Errors
    ///
    /// The estimator is not streamable, the estimand or query is not an all-observed IV
    /// problem, a column is not float64, a block is rank deficient, or the reduction exceeds
    /// its ceiling.
    pub fn estimate_streamed(
        &self,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        query: &AverageEffectQuery,
        batch_rows: usize,
        ceiling_bytes: u64,
    ) -> Result<StreamedIvEstimate, EstimationError> {
        let delta = self.validate_streamed_query(estimand, query)?;
        let mut ids =
            Vec::with_capacity(2 + estimand.instruments.len() + estimand.adjustment_set.len());
        ids.push(query.treatment);
        ids.push(query.outcome);
        ids.extend_from_slice(&estimand.instruments);
        ids.extend_from_slice(&estimand.adjustment_set);
        let PreparationProduct { row_mask, .. } = prepare_table(
            data,
            &ids,
            MissingPolicy::CompleteCase,
            &[],
            PreparationControls::finite_resident(),
        )?;
        let column = |id| -> Result<&[f64], EstimationError> {
            match data.column(id)? {
                ColumnView::Float64(c) => Ok(c.values.as_slice()),
                _ => Err(EstimationError::unsupported(
                    "a streamed reduction reads float64 design columns",
                )),
            }
        };
        // Design column order is [1 | adjustment… | instruments… | treatment]: the exogenous
        // block, the excluded instruments, then the endogenous column the moments end with.
        let mut columns = Vec::with_capacity(1 + ids.len());
        for &id in estimand.adjustment_set.iter() {
            columns.push(column(id)?);
        }
        for &id in estimand.instruments.iter() {
            columns.push(column(id)?);
        }
        columns.push(column(query.treatment)?);
        let outcome = column(query.outcome)?;
        let x_ncols = 1 + estimand.adjustment_set.len();
        let z_ncols = estimand.instruments.len();
        let mut source = MaskedRows::new(true, columns, RowMode::Plain { outcome }, &row_mask);
        let p = x_ncols + z_ncols + 1;
        let options = CapacityOptions::new(batch_rows, ceiling_bytes, OutputRetention::SummaryOnly);
        let result = run_weighted_capacity(CapacityControl::Glm, &mut source, &options)?
            .into_resident()
            .ok_or_else(|| EstimationError::data_msg("the reduction summary was not resident"))?;
        let n = usize::try_from(result.residual.positive_weight_rows).unwrap_or(usize::MAX);
        let moments = IvMoments {
            gram: &result.cross_products.xtwx,
            cross_outcome: &result.cross_products.xtwy,
            outcome_sum_squares: result.cross_products.ytwy,
            x_ncols,
            z_ncols,
            nrows: n,
        };
        debug_assert_eq!(result.cross_products.xtwx.len(), p * p);
        let fit =
            antecedent_stats::fit_2sls_from_moments(&moments).map_err(crate::error::EstimationError::from)?;
        let mut first_stage = fit.first_stage_diagnostics;
        if matches!(self.inference.se_kind, AnalyticSeKind::Homoskedastic) {
            let (set, reason) =
                antecedent_stats::anderson_rubin_confidence_set_from_moments(&moments, AR_LEVEL)
                    .map_err(crate::error::EstimationError::from)?;
            // Scale structural-coefficient endpoints to the published contrast.
            first_stage.anderson_rubin = set.map(|(lo, hi, level)| {
                let (mut lo_a, mut hi_a) = (lo * delta, hi * delta);
                if delta < 0.0 {
                    std::mem::swap(&mut lo_a, &mut hi_a);
                }
                (lo_a, hi_a, level)
            });
            first_stage.uncertainty_withheld = reason;
        } else {
            first_stage.anderson_rubin = None;
            first_stage.uncertainty_withheld = Some("anderson_rubin_requires_homoskedastic");
        }
        Ok(StreamedIvEstimate {
            ate: fit.coefficients[0] * delta,
            n_obs: result.residual.positive_weight_rows,
            coefficients: fit.coefficients,
            first_stage,
            reserved_bytes: result.charges.total_bytes,
            source_passes: result.source_passes,
            batches: result.batches,
            choice: *result.physical_choice(),
        })
    }
    /// Validate the supported IV estimand and population before scanning rows.
    fn validate_streamed_query(
        &self,
        estimand: &IdentifiedEstimand,
        query: &AverageEffectQuery,
    ) -> Result<f64, EstimationError> {
        if !self.supports_streamed_reduction() {
            return Err(EstimationError::unsupported(
                "a streamed reduction needs two-stage least squares without clustering, \
                 multiway ids or panel times",
            ));
        }
        crate::util::require_explicit_override(
            self.overlap,
            "IV estimators require ExplicitOverride overlap policy (not propensity-based)",
        )?;
        if estimand.method_kind().ok() != Some(antecedent_expr::EstimandMethod::Iv) {
            return Err(EstimationError::IncompatibleEstimand {
                message: "IV estimators expect an \"iv\" estimand",
            });
        }
        if estimand.instruments.is_empty() {
            return Err(EstimationError::IncompatibleEstimand {
                message: "IV estimators require a non-empty instrument set",
            });
        }
        query.validate()?;
        if !query.effect_modifiers.is_empty() {
            return Err(EstimationError::unsupported(
                "IV estimators do not support effect modifiers",
            ));
        }
        if query.target_population != TargetPopulation::AllObserved {
            return Err(EstimationError::refused(
                antecedent_core::reason_code!("population_not_estimable"),
                "IV estimators only support TargetPopulation::AllObserved",
            ));
        }
        let (_, _, delta) = treatment_contrast(&query.active, &query.control)?;
        Ok(delta)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use antecedent_core::{AssumptionSet, AverageEffectQuery, ExecutionContext, VariableId};
    use antecedent_data::TabularData;
    use antecedent_expr::{ExprId, IdentifiedEstimand};

    use super::*;
    use crate::iv::TwoStageLeastSquaresWorkspace;

    fn noise(i: usize, k: usize) -> f64 {
        (((i * 31 + k * 977) as f64 * 12.9898).sin() * 43758.5453).fract() - 0.5
    }

    /// `t = 0.8 z1 + 0.5 z2 + 0.4 w + u`, `y = 1 + 2 t + 0.3 w + 1.5 u + noise`.
    fn fixture(rows: usize) -> (TabularData, IdentifiedEstimand, AverageEffectQuery) {
        let z1: Vec<f64> = (0..rows).map(|i| noise(i, 1) * 3.0).collect();
        let z2: Vec<f64> = (0..rows).map(|i| noise(i, 2) * 2.0).collect();
        let w: Vec<f64> = (0..rows).map(|i| noise(i, 3) * 2.0).collect();
        let u: Vec<f64> = (0..rows).map(|i| noise(i, 4)).collect();
        let t: Vec<f64> =
            (0..rows).map(|i| 0.8 * z1[i] + 0.5 * z2[i] + 0.4 * w[i] + u[i]).collect();
        let y: Vec<f64> = (0..rows)
            .map(|i| 1.0 + 2.0 * t[i] + 0.3 * w[i] + 1.5 * u[i] + 0.2 * noise(i, 5))
            .collect();
        let data = TabularData::from_f64_columns([
            ("t", t.as_slice()),
            ("y", y.as_slice()),
            ("z1", z1.as_slice()),
            ("z2", z2.as_slice()),
            ("w", w.as_slice()),
        ])
        .unwrap();
        let mut estimand = IdentifiedEstimand::instrumental(
            "iv",
            Arc::from([VariableId::from_raw(2), VariableId::from_raw(3)]),
            ExprId::from_raw(0),
        );
        estimand.adjustment_set = Arc::from([VariableId::from_raw(4)]);
        let query = AverageEffectQuery::with_levels(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            0.0,
            2.0,
        );
        (data, estimand, query)
    }

    fn close(left: f64, right: f64) -> bool {
        (left - right).abs() <= 1e-7 * left.abs().max(right.abs()).max(1.0)
    }

    #[test]
    fn the_streamed_reduction_equals_the_resident_fit_diagnostics_and_ar_set() {
        let (data, estimand, query) = fixture(4_000);
        let estimator = TwoStageLeastSquares::new().with_bootstrap_replicates(0);
        assert!(estimator.supports_streamed_reduction());
        let prepared = estimator.prepare(&data, &estimand, &query).unwrap();
        let ctx = ExecutionContext::for_tests(1);
        let mut workspace = TwoStageLeastSquaresWorkspace::default();
        let resident =
            estimator.fit(&prepared, &mut workspace, &ctx, AssumptionSet::new()).unwrap();
        // A batch far smaller than the row count: many batches, one window.
        let streamed = estimator.estimate_streamed(&data, &estimand, &query, 97, 1 << 24).unwrap();
        assert!(streamed.batches > 1, "the rows were read in several batches");
        assert_eq!(Some(streamed.n_obs), resident.provenance.n_obs, "complete cases");
        assert!(close(streamed.ate, resident.ate), "ate {} vs {}", streamed.ate, resident.ate);
        let want = resident.diagnostics.first_stage.as_ref().unwrap();
        let got = &streamed.first_stage;
        assert_eq!((got.df1, got.df2), (want.df1, want.df2));
        assert!(
            close(got.f_statistic, want.f_statistic),
            "F {} vs {}",
            got.f_statistic,
            want.f_statistic
        );
        assert!(close(got.partial_r2, want.partial_r2));
        assert_eq!(got.uncertainty_withheld, want.uncertainty_withheld);
        match (got.anderson_rubin, want.anderson_rubin) {
            (Some((lo, hi, level)), Some((want_lo, want_hi, want_level))) => {
                assert!(
                    close(lo, want_lo) && close(hi, want_hi),
                    "AR [{lo}, {hi}] vs [{want_lo}, {want_hi}]"
                );
                assert_eq!(level, want_level);
            }
            (None, None) => {}
            other => panic!("the Anderson-Rubin sets disagree: {other:?}"),
        }
        // The published estimate carries no Wald or bootstrap standard error either way.
        let published = streamed.into_effect(
            AssumptionSet::new(),
            estimator.overlap,
            estimator.inference.se_kind,
        );
        assert!(!published.se_analytic().is_finite());
        assert!(published.se_bootstrap().is_none());
    }

    #[test]
    fn a_ceiling_below_the_window_refuses_and_a_clustered_estimator_is_not_streamable() {
        let (data, estimand, query) = fixture(500);
        let estimator = TwoStageLeastSquares::new().with_bootstrap_replicates(0);
        assert!(estimator.estimate_streamed(&data, &estimand, &query, 64, 8).is_err());
        let clustered = estimator.clone().with_cluster_ids(vec![0; 500]);
        assert!(!clustered.supports_streamed_reduction());
        assert!(clustered.estimate_streamed(&data, &estimand, &query, 64, 1 << 20).is_err());
        let robust = estimator.with_se_kind(AnalyticSeKind::Hc1);
        let streamed = robust.estimate_streamed(&data, &estimand, &query, 64, 1 << 20).unwrap();
        assert!(streamed.first_stage.anderson_rubin.is_none());
        assert_eq!(
            streamed.first_stage.uncertainty_withheld,
            Some("anderson_rubin_requires_homoskedastic")
        );
    }
}
