//! Temporal linear adjustment estimator.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::too_many_arguments
)]

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, ExecutionContext, Lag, TargetPopulation, TemporalEffectQuery, VariableId,
};
use antecedent_data::{
    DiscoveryEstimationSplit, LaggedColumn, LaggedSampleWorkspace, TemporalIndexer, TimeSeriesData,
};
use antecedent_expr::IdentifiedEstimand;
use antecedent_stats::CompiledDesign;

use crate::adjustment::{
    EffectEstimate, EstimationWorkspace, LinearAdjustmentAte, PreparedEstimationProblem,
    intervention_f64,
};
use crate::error::EstimationError;
use crate::overlap::OverlapPolicy;

/// Temporal linear adjustment for unfolded backdoor estimands.
#[derive(Clone, Debug)]
pub struct TemporalLinearAdjustment {
    /// Shared OLS / bootstrap machinery.
    pub inner: LinearAdjustmentAte,
}

impl Default for TemporalLinearAdjustment {
    fn default() -> Self {
        Self::new()
    }
}

impl TemporalLinearAdjustment {
    /// Defaults match [`LinearAdjustmentAte::new`].
    #[must_use]
    pub fn new() -> Self {
        Self { inner: LinearAdjustmentAte::new() }
    }

    /// Set the shared OLS / bootstrap machinery.
    #[must_use]
    pub fn with_inner(mut self, inner: LinearAdjustmentAte) -> Self {
        self.inner = inner;
        self
    }

    /// Prepare a lag-aligned design from series + unfolded identification.
    ///
    /// Adjustment `VariableId`s are interpreted as **dense unfolded node ids**
    /// (as returned by [`antecedent_identify::TemporalBackdoorIdentifier`]).
    ///
    /// `extra_contemporaneous` are schema (lag-0) covariates appended to the design
    /// after unfolded adjustment — used by temporal RCC refuters.
    ///
    /// # Errors
    ///
    /// Incompatible estimand, missing columns, or sample preparation failures.
    pub fn prepare(
        &self,
        data: &TimeSeriesData,
        estimand: &IdentifiedEstimand,
        query: &TemporalEffectQuery,
        indexer: &TemporalIndexer,
        split: Option<&DiscoveryEstimationSplit>,
        policy: &antecedent_core::KernelPolicy,
    ) -> Result<PreparedEstimationProblem, EstimationError> {
        self.prepare_with_extras(data, estimand, query, indexer, split, policy, &[])
    }

    /// Like [`Self::prepare`], with optional lag-0 schema covariates.
    ///
    /// # Errors
    ///
    /// Incompatible estimand, missing columns, or sample preparation failures.
    pub fn prepare_with_extras(
        &self,
        data: &TimeSeriesData,
        estimand: &IdentifiedEstimand,
        query: &TemporalEffectQuery,
        indexer: &TemporalIndexer,
        split: Option<&DiscoveryEstimationSplit>,
        policy: &antecedent_core::KernelPolicy,
        extra_contemporaneous: &[VariableId],
    ) -> Result<PreparedEstimationProblem, EstimationError> {
        if self.inner.overlap != OverlapPolicy::ExplicitOverride {
            return Err(EstimationError::Overlap {
                message: "temporal linear adjustment requires ExplicitOverride overlap policy",
            });
        }
        if !matches!(
            estimand.method_kind().ok(),
            Some(
                antecedent_expr::EstimandMethod::TemporalBackdoorUnfolded
                    | antecedent_expr::EstimandMethod::BackdoorAdjustment
            )
        ) {
            return Err(EstimationError::IncompatibleEstimand {
                message: "TemporalLinearAdjustment expects temporal.backdoor.unfolded",
            });
        }
        query.validate()?;

        refuse_multi_step_schedule(&query.policy)?;
        if query.target_population != TargetPopulation::AllObserved {
            return Err(EstimationError::TargetPopulation);
        }
        // Lagged samples are anchored at the latest queried outcome. Temporal
        // query offsets are absolute around the policy origin, so horizons > 1
        // otherwise appear as unsupported positive ("future") lags even though
        // the same design is representable by shifting every column together.
        let sample_anchor = query.outcome_offset().max(0);
        let t_lag = offset_to_lag(query.try_treatment_offset()? - sample_anchor)?;
        let y_lag = offset_to_lag(query.outcome_offset() - sample_anchor)?;

        let mut cols =
            Vec::with_capacity(2 + estimand.adjustment_set.len() + extra_contemporaneous.len());
        cols.push(LaggedColumn { variable: query.treatment, lag: t_lag });
        cols.push(LaggedColumn { variable: query.outcome, lag: y_lag });

        let mut adj_keys = Vec::new();
        for &dense_var in estimand.adjustment_set.iter() {
            let key = indexer
                .key_of(dense_var.raw())
                .map_err(|e| EstimationError::data_msg(e.to_string()))?;
            let lag = offset_to_lag(key.offset - sample_anchor).map_err(|_| {
                EstimationError::unsupported(
                    "adjustment node sits after this horizon's sample anchor and \
                     cannot be lag-aligned",
                )
            })?;
            cols.push(LaggedColumn { variable: key.variable, lag });
            adj_keys.push(key.variable);
        }
        let lag0 = Lag::from_raw(0);
        for &vid in extra_contemporaneous {
            cols.push(LaggedColumn { variable: vid, lag: lag0 });
            adj_keys.push(vid);
        }

        let max_lag = cols.iter().map(|c| c.lag.raw()).max().unwrap_or(0);
        let plan = data
            .plan_lagged_sample(max_lag, Arc::<[LaggedColumn]>::from(cols))
            .map_err(EstimationError::from)?;
        let mut sample_ws = LaggedSampleWorkspace::default();
        let prep = plan.prepare(data, &mut sample_ws, policy).map_err(EstimationError::from)?;

        let n = prep.n;
        let (row_start, row_end) = if let Some(s) = split {
            // Map estimation time range into prepared sample rows (aligned at max_lag).
            let est_start = s.estimation.start.saturating_sub(max_lag as usize);
            let est_end = s.estimation.end.saturating_sub(max_lag as usize).min(n);
            if est_start >= est_end {
                return Err(EstimationError::data_msg(
                    "estimation split empty after lag alignment",
                ));
            }
            (est_start, est_end)
        } else {
            (0, n)
        };
        let nrows = row_end - row_start;
        let t = &prep.column(0)[row_start..row_end];
        let y = &prep.column(1)[row_start..row_end];
        let cov_refs: Vec<(VariableId, &[f64])> = adj_keys
            .iter()
            .enumerate()
            .map(|(i, &vid)| (vid, &prep.column(2 + i)[row_start..row_end]))
            .collect();
        let selected: Vec<usize> = (0..nrows).collect();
        let design = CompiledDesign::linear_adjustment(t, &cov_refs, y, &selected)
            .map_err(EstimationError::from)?;

        let active = intervention_f64(&query.active)?;
        let control = intervention_f64(&query.control)?;
        let treatment_delta = active - control;
        if treatment_delta == 0.0 {
            return Err(EstimationError::unsupported(
                "active and control treatment levels must differ",
            ));
        }

        Ok(PreparedEstimationProblem {
            design,
            method: Arc::from("temporal.linear.adjustment"),
            adjustment_set: Arc::from(adj_keys),
            overlap: self.inner.overlap,
            treatment_delta,
            target_population: TargetPopulation::AllObserved,
            treatment: Arc::from(t),
            active,
            control,
        })
    }

    /// Prepare a stacked panel design (no cross-unit lag windows) with unit cluster ids
    /// and per-row panel times.
    ///
    /// Returns `(problem, cluster_ids, panel_times)` where `cluster_ids[row] = unit_id`
    /// and `panel_times` are consecutive within-unit indices `0..n_unit` for the prepared
    /// (lag-aligned) rows of each unit.
    ///
    /// # Errors
    ///
    /// Empty panel, incompatible estimand, or per-unit preparation failures.
    pub fn prepare_panel(
        &self,
        panel: &antecedent_data::PanelData,
        estimand: &IdentifiedEstimand,
        query: &TemporalEffectQuery,
        indexer: &TemporalIndexer,
        split: Option<&DiscoveryEstimationSplit>,
        policy: &antecedent_core::KernelPolicy,
    ) -> Result<(PreparedEstimationProblem, Vec<u32>, Vec<i64>), EstimationError> {
        if panel.unit_count() == 0 {
            return Err(EstimationError::data_msg("panel needs ≥1 unit"));
        }
        let mut all_t = Vec::new();
        let mut all_y = Vec::new();
        let mut all_covs: Vec<(VariableId, Vec<f64>)> = Vec::new();
        let mut cluster_ids = Vec::new();
        let mut panel_times = Vec::new();
        let mut adj_keys: Vec<VariableId> = Vec::new();
        let mut active = 0.0;
        let mut control = 0.0;
        let mut treatment_delta = 0.0;
        let mut first = true;

        for unit in panel.units() {
            let prep = self.prepare(&unit.series, estimand, query, indexer, split, policy)?;
            if first {
                active = prep.active;
                control = prep.control;
                treatment_delta = prep.treatment_delta;
                adj_keys = prep.adjustment_set.to_vec();
                all_covs = adj_keys.iter().map(|&id| (id, Vec::new())).collect();
                first = false;
            }
            let n = prep.treatment.len();
            all_t.extend_from_slice(&prep.treatment);
            all_y.extend_from_slice(&prep.design.outcome);
            // Covariates are columns 2.. of the column-major design matrix.
            let nrows = prep.design.nrows;
            for (i, (_id, dest)) in all_covs.iter_mut().enumerate() {
                let base = (2 + i) * nrows;
                dest.extend_from_slice(&prep.design.matrix[base..base + nrows]);
            }
            cluster_ids.extend(std::iter::repeat_n(unit.unit_id, n));
            // Prepared rows are consecutive in calendar time after lag alignment.
            for t_idx in 0..n {
                let t_label = i64::try_from(t_idx)
                    .map_err(|_| EstimationError::data_msg("panel time index does not fit i64"))?;
                panel_times.push(t_label);
            }
        }

        let cov_refs: Vec<(VariableId, &[f64])> =
            all_covs.iter().map(|(id, v)| (*id, v.as_slice())).collect();
        let selected: Vec<usize> = (0..all_t.len()).collect();
        let design = CompiledDesign::linear_adjustment(&all_t, &cov_refs, &all_y, &selected)
            .map_err(EstimationError::from)?;

        Ok((
            PreparedEstimationProblem {
                design,
                method: Arc::from("temporal.linear.adjustment.panel"),
                adjustment_set: Arc::from(adj_keys),
                overlap: self.inner.overlap,
                treatment_delta,
                target_population: TargetPopulation::AllObserved,
                treatment: Arc::from(all_t),
                active,
                control,
            },
            cluster_ids,
            panel_times,
        ))
    }

    /// Fit using the shared linear-adjustment path.
    ///
    /// # Errors
    ///
    /// OLS / bootstrap failures.
    pub fn fit(
        &self,
        problem: &PreparedEstimationProblem,
        workspace: &mut EstimationWorkspace,
        ctx: &ExecutionContext,
        assumptions: AssumptionSet,
    ) -> Result<EffectEstimate, EstimationError> {
        self.inner.fit(problem, workspace, ctx, assumptions)
    }
}

/// Multi-step schedules identify a contrast over *every* treatment-time node;
/// this estimator regresses a single treatment column and cannot honor that
/// estimand — refuse rather than estimate a one-node proxy.
fn refuse_multi_step_schedule(
    policy: &antecedent_core::TemporalPolicy,
) -> Result<(), EstimationError> {
    match policy {
        antecedent_core::TemporalPolicy::Dynamic { active_at, .. } if active_at.len() != 1 => {
            Err(EstimationError::unsupported(
                "TemporalPolicy::Dynamic with multiple active steps is not supported by \
                 temporal linear adjustment (use a single-step schedule)",
            ))
        }
        antecedent_core::TemporalPolicy::Sustained { from, until } if until > from => {
            Err(EstimationError::unsupported(
                "TemporalPolicy::Sustained spanning multiple steps is not supported by \
                 temporal linear adjustment: the identified estimand contrasts every \
                 treatment-time node, and a single-column regression would silently \
                 estimate a one-node proxy. Use a single-step window (from == until).",
            ))
        }
        _ => Ok(()),
    }
}

fn offset_to_lag(offset: i32) -> Result<Lag, EstimationError> {
    if offset > 0 {
        return Err(EstimationError::unsupported(
            "positive offsets (future treatment/outcome) unsupported for temporal adjustment",
        ));
    }
    let lag = u32::try_from(-offset)
        .map_err(|_| EstimationError::unsupported("offset does not fit lag"))?;
    Ok(Lag::from_raw(lag))
}

#[cfg(test)]
#[allow(clippy::many_single_char_names)]
mod tests {
    use antecedent_core::{
        CausalSchemaBuilder, DistributionRef, ExecutionContext, Lag, MeasurementSpec,
        PredicateExpr, RoleHint, SmallRoleSet, TargetPopulation, TemporalEffectQuery,
        TemporalPolicy, ValueType, VariableId,
    };
    use antecedent_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
        TimeSeriesData, ValidityBitmap,
    };
    use antecedent_graph::{TemporalDag, ensure_lagged};
    use antecedent_identify::TemporalBackdoorIdentifier;

    use super::*;

    fn series() -> (TimeSeriesData, TemporalDag) {
        let n = 240usize;
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "x",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        b.add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        let mut x = vec![0.0; n];
        let mut y = vec![0.0; n];
        for t in 1..n {
            x[t] = ((t as f64) * 0.07).sin();
            y[t] = 0.8 * x[t - 1] + 0.1 * (0.19 * t as f64).cos();
        }
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(0),
                    Arc::from(x),
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
        ];
        let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
        let data = TimeSeriesData::try_new(
            storage,
            TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
        )
        .unwrap();
        let mut g = TemporalDag::empty();
        let x1 = ensure_lagged(&mut g, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let y0 = ensure_lagged(&mut g, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        g.insert_directed(x1, y0).unwrap();
        (data, g)
    }

    #[test]
    fn recovers_lagged_effect() {
        let (data, g) = series();
        let q = TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
            .with_policy(TemporalPolicy::pulse(-1))
            .with_horizon_steps(1)
            .with_max_history_lag(Some(1));
        let id_res = TemporalBackdoorIdentifier::new().identify_temporal(&g, &q).unwrap();
        let estimand = id_res.result.estimands.first().unwrap();
        let est = TemporalLinearAdjustment::new();
        let prep = est
            .prepare(
                &data,
                estimand,
                &q,
                &id_res.indexer,
                None,
                &ExecutionContext::for_tests(1).kernel_policy,
            )
            .unwrap();
        let mut ws = EstimationWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let mut est2 = TemporalLinearAdjustment::new();
        est2.inner.bootstrap_replicates = 0;
        let effect = est2.fit(&prep, &mut ws, &ctx, id_res.result.required_assumptions).unwrap();
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../conformance/estimate/temporal_adjustment/expected.json"
        ))
        .unwrap();
        let expected = fixture["reference"]["coefficient"].as_f64().unwrap();
        let tolerance = fixture["acceptance"]["atol"].as_f64().unwrap();
        assert!(
            (effect.ate - expected).abs() <= tolerance,
            "ate={} expected {expected}",
            effect.ate
        );
    }

    #[test]
    fn rejects_planned_target_populations() {
        let (data, g) = series();
        let base =
            TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
                .with_policy(TemporalPolicy::pulse(-1))
                .with_horizon_steps(1)
                .with_max_history_lag(Some(1));
        let id_res = TemporalBackdoorIdentifier::new().identify_temporal(&g, &base).unwrap();
        let estimand = id_res.result.estimands.first().unwrap();
        let est = TemporalLinearAdjustment::new();
        let policy = &ExecutionContext::for_tests(1).kernel_policy;
        for population in [
            TargetPopulation::Treated,
            TargetPopulation::Predicate(PredicateExpr::named("cohort_a")),
            TargetPopulation::CustomDistribution(DistributionRef::from_raw(1)),
        ] {
            let q = base.clone().with_target_population(population);
            let err = est.prepare(&data, estimand, &q, &id_res.indexer, None, policy).unwrap_err();
            assert!(matches!(err, EstimationError::TargetPopulation));
        }
    }
}
