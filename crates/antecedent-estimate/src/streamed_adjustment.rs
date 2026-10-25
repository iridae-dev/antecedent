//! Streamed normal-equation reduction for the linear adjustment estimator.
//!
//! The design `[1 | T | Z…]` is never built. Rows are read in bounded batches straight from
//! the source columns, through the complete-case mask, into the weighted Gaussian
//! reduction: one pass accumulates `XᵀX`, `Xᵀy` and `yᵀy`, the coefficients solve from
//! them, and a second pass accumulates the residual sum of squares and the `HC0` meat
//! `Σ e² x xᵀ`. The treatment coefficient, its homoskedastic standard error
//! (`RSS/(n−p)·[(XᵀX)⁻¹]ₜₜ`) and its `HC0` standard error follow in closed form from
//! those sufficient statistics, so memory is one batch and a `p × p` Gram instead of
//! the `n × p` design.
//!
//! The reduction is the normal equation, so it squares the design's condition number:
//! it agrees with the materialized least-squares fit to a relative tolerance set by that
//! condition number, not bit for bit. Only the closed-form route is streamed: ordinary
//! least squares, unit weights, homoskedastic or `HC0` standard error, all-observed
//! target, numeric columns, no bootstrap, clustering, panel structure or contrasts.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::execution_facts::OutputRetention;
use antecedent_core::physical_choice::PhysicalChoice;
use antecedent_core::{AverageEffectQuery, TargetPopulation};
use antecedent_data::materialize::{PreparationControls, PreparationProduct, prepare_table};
use antecedent_data::{ColumnView, MissingPolicy, TableView, TabularData};
use antecedent_expr::IdentifiedEstimand;
use antecedent_io_core::capacity_account::{
    CapacityControl, CapacityOptions, DesignShape, WeightedCapacityError, WeightedRowSource,
    run_weighted_capacity,
};
use antecedent_stats::invert_square;

use crate::adjustment::{LinearAdjustmentAte, LinearFitKind};
use crate::error::EstimationError;
use crate::prepare::{
    require_adjustment_shaped, treatment_contrast, validate_ate_query_with_targets,
};
use crate::se::AnalyticSeKind;

/// Effect and standard error from a streamed reduction, with the facts of the run.
#[derive(Clone, Debug)]
pub struct StreamedAdjustmentEstimate {
    /// Average effect: the treatment coefficient times the active − control contrast.
    pub ate: f64,
    /// Analytic standard error of [`Self::ate`], of the estimator's configured kind.
    pub se_analytic: f64,
    /// Complete-case rows reduced.
    pub n_obs: u64,
    /// All coefficients in design order: intercept, treatment, adjustment columns.
    pub coefficients: Vec<f64>,
    /// Bytes reserved against the ceiling for this run.
    pub reserved_bytes: u64,
    /// Passes over the source rows (the cross-product pass and the residual pass).
    pub source_passes: u64,
    /// Batches read in the cross-product pass.
    pub batches: u64,
    /// The executed backend choice, with the alternatives rejected.
    pub choice: PhysicalChoice,
}

/// Source columns read through the complete-case mask, one batch at a time.
struct MaskedColumns<'a> {
    treatment: &'a [f64],
    outcome: &'a [f64],
    covariates: Vec<&'a [f64]>,
    mask: &'a [bool],
    cursor: usize,
    bound: Option<usize>,
}

impl WeightedRowSource for MaskedColumns<'_> {
    fn ncols(&self) -> usize {
        2 + self.covariates.len()
    }

    fn input_ownership_bytes(&self) -> u64 {
        // The mask is the only resident structure the reduction keeps; the columns are
        // the caller's table.
        u64::try_from(self.mask.len()).unwrap_or(u64::MAX)
    }

    fn decoder_reservation(&self, batch_rows: usize) -> Result<u64, WeightedCapacityError> {
        if batch_rows == 0 { Err(WeightedCapacityError::BatchBound) } else { Ok(0) }
    }

    fn prepare_decoder(&mut self, batch_rows: usize) -> Result<(), WeightedCapacityError> {
        if batch_rows == 0 {
            return Err(WeightedCapacityError::BatchBound);
        }
        self.bound = Some(batch_rows);
        Ok(())
    }

    fn allocated_decoder_bytes(&self) -> u64 {
        0
    }

    fn rewind(&mut self) -> Result<(), WeightedCapacityError> {
        self.cursor = 0;
        Ok(())
    }

    fn read_batch(
        &mut self,
        max_rows: usize,
        x_rowmajor: &mut [f64],
        y: &mut [f64],
        weights: &mut [f64],
    ) -> Result<usize, WeightedCapacityError> {
        let bound = self.bound.ok_or(WeightedCapacityError::DecoderNotPrepared)?;
        if max_rows == 0 || max_rows > bound {
            return Err(WeightedCapacityError::ReadPastBatch { requested: max_rows, bound });
        }
        let ncols = self.ncols();
        if x_rowmajor.len() < max_rows * ncols || y.len() < max_rows || weights.len() < max_rows {
            return Err(WeightedCapacityError::Shape(DesignShape::BatchWindow));
        }
        let mut taken = 0;
        while taken < max_rows && self.cursor < self.mask.len() {
            let row = self.cursor;
            self.cursor += 1;
            if !self.mask[row] {
                continue;
            }
            let cells = &mut x_rowmajor[taken * ncols..(taken + 1) * ncols];
            cells[0] = 1.0;
            cells[1] = self.treatment[row];
            for (cell, column) in cells[2..].iter_mut().zip(&self.covariates) {
                *cell = column[row];
            }
            y[taken] = self.outcome[row];
            weights[taken] = 1.0;
            taken += 1;
        }
        Ok(taken)
    }
}

impl LinearAdjustmentAte {
    /// Whether this estimator's result follows in closed form from streamed sufficient
    /// statistics: ordinary least squares with unit weights and a homoskedastic or `HC0`
    /// standard error, no bootstrap, clustering, panel structure or contrasts.
    #[must_use]
    pub fn supports_streamed_reduction(&self) -> bool {
        matches!(self.fit_kind, LinearFitKind::Ols)
            && self.bootstrap_replicates == 0
            && matches!(self.se_kind, AnalyticSeKind::Homoskedastic | AnalyticSeKind::Hc0)
            && self.sampling_weights.is_none()
            && self.cluster_ids.is_none()
            && self.source_cluster_ids.is_none()
            && self.multiway_ids.is_none()
            && self.panel_times.is_none()
            && self.contrasts.is_empty()
    }

    /// Estimate the average effect by streaming the rows in batches of `batch_rows`,
    /// holding at most `ceiling_bytes`, without building the design.
    ///
    /// # Errors
    ///
    /// The estimator is not streamable ([`Self::supports_streamed_reduction`]), the
    /// estimand or query is not an all-observed adjustment, a design column is not
    /// float64, the rows are rank deficient, or the reduction exceeds its ceiling.
    pub fn estimate_streamed(
        &self,
        data: &TabularData,
        estimand: &IdentifiedEstimand,
        query: &AverageEffectQuery,
        batch_rows: usize,
        ceiling_bytes: u64,
    ) -> Result<StreamedAdjustmentEstimate, EstimationError> {
        if !self.supports_streamed_reduction() {
            return Err(EstimationError::unsupported(
                "a streamed reduction needs least squares with unit weights, a homoskedastic or \
                 HC0 standard error, no bootstrap, clustering, panel structure or contrasts",
            ));
        }
        crate::util::require_explicit_override(
            self.overlap,
            "LinearAdjustmentAte requires ExplicitOverride overlap policy",
        )?;
        require_adjustment_shaped(
            estimand,
            "LinearAdjustmentAte expects an adjustment-shaped estimand",
        )?;
        validate_ate_query_with_targets(query)?;
        if query.target_population != TargetPopulation::AllObserved {
            return Err(EstimationError::unsupported(
                "a streamed reduction estimates the all-observed population",
            ));
        }
        let (_, _, delta) = treatment_contrast(&query.active, &query.control)?;
        let mut ids = Vec::with_capacity(2 + estimand.adjustment_set.len());
        ids.push(query.treatment);
        ids.push(query.outcome);
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
        let mut covariates = Vec::with_capacity(estimand.adjustment_set.len());
        for &id in estimand.adjustment_set.iter() {
            covariates.push(column(id)?);
        }
        let mut source = MaskedColumns {
            treatment: column(query.treatment)?,
            outcome: column(query.outcome)?,
            covariates,
            mask: &row_mask,
            cursor: 0,
            bound: None,
        };
        let p = source.ncols();
        let options = CapacityOptions::new(batch_rows, ceiling_bytes, OutputRetention::SummaryOnly);
        let result = run_weighted_capacity(CapacityControl::Glm, &mut source, &options)
            .map_err(|error| EstimationError::data_msg(error.to_string()))?
            .into_resident()
            .ok_or_else(|| EstimationError::data_msg("the reduction summary was not resident"))?;
        let t_col = 1;
        let n = result.residual.positive_weight_rows;
        let se_coef = if self.se_kind == AnalyticSeKind::Hc0 {
            result.robust_covariance[t_col * p + t_col].max(0.0).sqrt()
        } else {
            let sigma2 = result.residual.weighted_rss / (n as f64 - p as f64).max(1.0);
            invert_square(&result.cross_products.xtwx, p)
                .map_or(f64::NAN, |inverse| (sigma2 * inverse[t_col * p + t_col].max(0.0)).sqrt())
        };
        Ok(StreamedAdjustmentEstimate {
            ate: result.coefficients[t_col] * delta,
            se_analytic: se_coef * delta.abs(),
            n_obs: n,
            coefficients: result.coefficients.clone(),
            reserved_bytes: result.charges.total_bytes,
            source_passes: result.source_passes,
            batches: result.batches,
            choice: *result.physical_choice(),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use antecedent_core::{AssumptionSet, AverageEffectQuery, ExecutionContext, VariableId};
    use antecedent_data::TabularData;
    use antecedent_expr::{ExprId, IdentifiedEstimand};

    use crate::adjustment::{EstimationWorkspace, LinearAdjustmentAte};
    use crate::estimator::Estimator;
    use crate::se::AnalyticSeKind;

    fn noise(i: usize) -> f64 {
        ((i as f64 * 12.9898).sin() * 43758.5453).fract().abs() - 0.5
    }

    /// `t` depends on `z1`; `y = 1 + 2 t + 0.5 z1 - 0.3 z2 + noise`.
    fn fixture(rows: usize) -> (TabularData, IdentifiedEstimand, AverageEffectQuery) {
        let z1: Vec<f64> = (0..rows).map(|i| noise(i) * 4.0).collect();
        let z2: Vec<f64> = (0..rows).map(|i| noise(i + 7919) * 3.0).collect();
        let t: Vec<f64> =
            (0..rows).map(|i| if z1[i] + 2.0 * noise(i + 31) > 0.0 { 1.0 } else { 0.0 }).collect();
        let y: Vec<f64> = (0..rows)
            .map(|i| 1.0 + 2.0 * t[i] + 0.5 * z1[i] - 0.3 * z2[i] + 0.4 * noise(i + 104_729))
            .collect();
        let data = TabularData::from_f64_columns([
            ("t", t.as_slice()),
            ("y", y.as_slice()),
            ("z1", z1.as_slice()),
            ("z2", z2.as_slice()),
        ])
        .unwrap();
        let estimand = IdentifiedEstimand::backdoor(
            "backdoor.adjustment",
            Arc::from([VariableId::from_raw(2), VariableId::from_raw(3)]),
            ExprId::from_raw(0),
        );
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        (data, estimand, query)
    }

    fn close(left: f64, right: f64, tolerance: f64) -> bool {
        (left - right).abs() <= tolerance * left.abs().max(right.abs()).max(1.0)
    }

    #[test]
    fn the_streamed_reduction_equals_the_materialized_fit_and_both_standard_errors() {
        let (data, estimand, query) = fixture(4_000);
        for kind in [AnalyticSeKind::Homoskedastic, AnalyticSeKind::Hc0] {
            let estimator = LinearAdjustmentAte {
                bootstrap_replicates: 0,
                ..LinearAdjustmentAte::new().with_se_kind(kind)
            };
            assert!(estimator.supports_streamed_reduction());
            let prep = estimator.prepare(&data, &estimand, &query).unwrap();
            let mut workspace = EstimationWorkspace::default();
            let ctx = ExecutionContext::for_tests(1);
            let resident =
                estimator.fit(&prep, &mut workspace, &ctx, AssumptionSet::new()).unwrap();
            // A batch far smaller than the row count: many batches, one window.
            let streamed =
                estimator.estimate_streamed(&data, &estimand, &query, 97, 1 << 24).unwrap();
            assert!(streamed.batches > 1, "{kind:?}: the rows were read in several batches");
            assert_eq!(streamed.source_passes, 2, "{kind:?}");
            assert_eq!(Some(streamed.n_obs), resident.provenance.n_obs, "{kind:?}: complete cases");
            assert!(close(streamed.ate, resident.ate, 1e-9), "{kind:?}: ate");
            assert!(
                close(streamed.se_analytic, resident.se_analytic(), 1e-9),
                "{kind:?}: se {} vs {}",
                streamed.se_analytic,
                resident.se_analytic()
            );
        }
    }

    #[test]
    fn under_a_budget_the_design_cannot_fit_the_streamed_reduction_runs_and_is_chosen() {
        use antecedent_core::physical_choice::{
            BackendSupport, PhysicalAlternative, RejectionReason, ResourceCapability, RouteChoice,
            RouteWork,
        };
        use antecedent_core::route_memory::RouteBudget;
        let (data, estimand, query) = fixture(4_000);
        let estimator =
            LinearAdjustmentAte { bootstrap_replicates: 0, ..LinearAdjustmentAte::new() };
        let design_bytes = 4_000_u64 * 4 * 8;
        let budget = design_bytes / 2;
        // The reduction runs under a ceiling smaller than the design it never builds.
        let streamed = estimator.estimate_streamed(&data, &estimand, &query, 97, budget).unwrap();
        assert!(streamed.reserved_bytes <= budget, "{} bytes reserved", streamed.reserved_bytes);
        // The route-level choice over the same shape agrees: both resident alternatives
        // are rejected for want of memory and the streamed reduction is chosen.
        let everything = PhysicalAlternative::ALL
            .into_iter()
            .fold(BackendSupport::none(), |support, alternative| support.with(alternative, true));
        let routed = RouteChoice::select(
            RouteWork::new(4_000, 4, 2).batch_rows(97),
            everything,
            RouteBudget::new(budget),
        )
        .unwrap();
        assert_eq!(routed.selected(), PhysicalAlternative::StreamedReduction);
        let reason = |alternative| {
            routed
                .choice()
                .rejected()
                .iter()
                .find(|rejected| rejected.alternative() == alternative)
                .map(|rejected| rejected.reason())
        };
        assert_eq!(
            reason(PhysicalAlternative::DirectPreparedNumericViews),
            Some(RejectionReason::MissingResourceCapability {
                capability: ResourceCapability::PreparedColumns,
            })
        );
        assert_eq!(
            reason(PhysicalAlternative::MaterializedDesign),
            Some(RejectionReason::MissingResourceCapability {
                capability: ResourceCapability::DesignMemory,
            })
        );
    }

    #[test]
    fn an_estimator_without_a_closed_form_reduction_is_refused() {
        let (data, estimand, query) = fixture(200);
        let bootstrap =
            LinearAdjustmentAte { bootstrap_replicates: 50, ..LinearAdjustmentAte::new() };
        assert!(!bootstrap.supports_streamed_reduction());
        assert!(bootstrap.estimate_streamed(&data, &estimand, &query, 64, 1 << 20).is_err());
        let robust = LinearAdjustmentAte {
            bootstrap_replicates: 0,
            ..LinearAdjustmentAte::new().with_se_kind(AnalyticSeKind::Hc3)
        };
        assert!(!robust.supports_streamed_reduction());
    }

    #[test]
    fn a_ceiling_below_the_window_refuses_instead_of_reading() {
        let (data, estimand, query) = fixture(500);
        let estimator =
            LinearAdjustmentAte { bootstrap_replicates: 0, ..LinearAdjustmentAte::new() };
        assert!(estimator.estimate_streamed(&data, &estimand, &query, 64, 8).is_err());
    }
}
