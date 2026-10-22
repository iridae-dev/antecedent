//! Temporal linear adjustment estimator.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg_attr(
    test,
    allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "test fixtures compare exact constants and index with small literals"
    )
)]

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, ExecutionContext, HistoryWindow, InitialLaw, KernelPolicy, Lag,
    TargetPopulation, TemporalEffectQuery, VariableId,
};
use antecedent_data::{
    DiscoveryEstimationSplit, LaggedColumn, LaggedSampleWorkspace, SamplingRegularity,
    TemporalIndexer, TimeIndex, TimeSeriesData,
};
use antecedent_expr::IdentifiedEstimand;
use antecedent_model::history_operators::{
    ActionTiming, CensoringAssignmentWeight, HistoryConstruction, HistoryOperatorError, Horizon,
};
use antecedent_stats::{CompiledDesign, DenseLinearAlgebra};

use crate::adjustment::{
    EffectEstimate, EstimationWorkspace, LinearAdjustmentAte, PreparedEstimationProblem,
    intervention_f64,
};
use crate::error::EstimationError;
use crate::overlap::OverlapPolicy;
use crate::se::AnalyticSeKind;

/// RNG stream base for the one-series circular-block row bootstrap.
const TEMPORAL_BLOCK_STREAM: u64 = 0x7B10_C000_0000;

/// How [`TemporalLinearAdjustment::fit_dependence_honest`] computed its SE.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TemporalDependenceSe {
    /// Circular-block length in lag-aligned rows:
    /// [`crate::temporal_block::dependence_block_length`] of the unfolded window
    /// over the treatment influence and every normal-equation score — at least
    /// `max(history + horizon, ⌈n^{1/3}⌉)`, lengthened for a persistently
    /// dependent score. When no replicates were requested
    /// ([`Self::replicates_attempted`] is 0) the scores are not scanned and this
    /// is the plain rule length.
    pub block_length: usize,
    /// Lag-aligned rows in the fitted (and resampled) design.
    pub rows: usize,
    /// Effective rows of the treatment-coefficient score at [`Self::block_length`]
    /// ([`crate::temporal_block::score_effective_rows`]).
    pub effective_rows: f64,
    /// Bootstrap replicates evaluated (0 when none were requested).
    pub replicates_attempted: u32,
    /// Bartlett kernel-bias factor of the treatment-coefficient score at
    /// [`Self::block_length`] ([`crate::temporal_block::kernel_bias_scale`]),
    /// applied to `se_bootstrap` together with the fixed-b factor.
    pub kernel_bias: f64,
}

/// [`crate::temporal_block::dependence_block_length`] of one series' prepared
/// design: the unfolded window (`history + horizon` slices), the treatment
/// influence and every normal-equation score of the regression (the intercept's
/// score is the residual series).
/// The lag windows that carry weight: all of them without sampling weights, else the windows of
/// positive weight (a zero-weight window stays in the design to keep consecutive times but
/// contributes no score).
fn analysed_windows(prep: &PreparedEstimationProblem) -> usize {
    prep.sampling_weights
        .map_or(prep.design.nrows, |record| usize::try_from(record.rows_used).unwrap_or(usize::MAX))
        .min(prep.design.nrows)
}

fn single_window_block_length(
    prep: &PreparedEstimationProblem,
    indexer: &TemporalIndexer,
    influence: Option<&[f64]>,
) -> Result<usize, EstimationError> {
    let rows = prep.design.nrows;
    let structural_span = (indexer.history() as usize + indexer.horizon() as usize).max(1);
    // The same regression was just fitted for the point estimate, so a refusal here means
    // the block length cannot be sized on the intercept's residual score; falling back to
    // the rule length would understate the dependence without saying so.
    let normal_scores = crate::temporal_block::normal_equation_scores(
        &prep.design.matrix,
        rows,
        prep.design.ncols,
        &prep.design.outcome,
    )
    .ok_or_else(|| {
        EstimationError::stats_msg(
            "normal-equation scores of the temporal adjustment regression are unavailable, \
             so the dependence-aware block length cannot be sized",
        )
    })?;
    let scores: Vec<&[f64]> =
        influence.into_iter().chain(normal_scores.iter().map(Vec::as_slice)).collect();
    // Windows of weight zero contribute no score: the block length is sized on the windows
    // that carry the effect, not on the padded series.
    let counted = analysed_windows(prep);
    Ok(crate::temporal_block::dependence_block_length(structural_span, counted, &scores))
}

/// What a lag-aligned temporal adjustment design is prepared from.
#[derive(Clone, Copy)]
pub struct TemporalDesignInputs<'a> {
    /// The observed series.
    pub data: &'a TimeSeriesData,
    /// Unfolded backdoor identification.
    pub estimand: &'a IdentifiedEstimand,
    /// The temporal effect query.
    pub query: &'a TemporalEffectQuery,
    /// Dense-node to (variable, offset) indexer of the unfolded graph.
    pub indexer: &'a TemporalIndexer,
    /// Discovery/estimation split, when the series was split.
    pub split: Option<&'a DiscoveryEstimationSplit>,
    /// Kernel policy for the lagged-sample preparation.
    pub policy: &'a KernelPolicy,
}

/// Which lag-aligned windows of a series with gaps are intact.
///
/// Window `r` is the one ending at series time `first_time + r`; time keeps its original
/// grid, so a missing time point removes the windows that read it and moves no other row.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GapSelection {
    /// Series time of window 0.
    pub first_time: usize,
    /// Whether each window has every planned column observed.
    pub complete: Vec<bool>,
}

impl GapSelection {
    /// Segment id of every intact window, in time order: windows adjacent on the original axis
    /// share a segment and a gap starts a new one.
    #[must_use]
    pub fn segments(&self) -> Arc<[u32]> {
        crate::serial_dependence::window_segments(&self.complete)
    }

    /// Series times of the windows that were dropped, ascending.
    #[must_use]
    pub fn dropped_times(&self) -> Vec<usize> {
        self.complete
            .iter()
            .enumerate()
            .filter(|(_, intact)| !**intact)
            .map(|(r, _)| self.first_time + r)
            .collect()
    }
}

/// Apply the series' sampling weights to a lag-aligned design: window `r` is weighted by the
/// weight of the observation it ends at, series time `first_time + r`.
///
/// A window of weight zero stays in the design, scaled to nothing, so the windows keep their
/// consecutive times (the block bootstrap and the HAC lags need them) and contribute no score.
/// The weights are normalized to mean one over the windows of positive weight. A series with
/// gaps, whose windows are not consecutive, refuses.
///
/// # Errors
///
/// `route_unsupported` for a gapped series; `invalid` for weights that do not cover the windows
/// or leave none with a positive weight.
fn weight_windows(
    design: CompiledDesign,
    data: &TimeSeriesData,
    first_time: usize,
    windows: usize,
    allow_gaps: bool,
) -> Result<(CompiledDesign, Option<crate::sampling_weights::SamplingWeightRecord>), EstimationError>
{
    let Some(weights) = data.storage().weights() else {
        return Ok((design, None));
    };
    if allow_gaps {
        return Err(crate::sampling_weights::unsupported_route(
            "sampling weights are read by consecutive lag windows; a series with gaps has none",
        ));
    }
    let Some(window_weights) = weights.get(first_time..first_time + windows) else {
        return Err(crate::sampling_weights::invalid(
            "the series' sampling weights do not cover its lag-aligned windows",
        ));
    };
    let positive: Vec<f64> = window_weights.iter().copied().filter(|w| *w > 0.0).collect();
    if positive.is_empty() {
        return Err(crate::sampling_weights::invalid(
            "no lag window has a positive sampling weight; the weighted population is empty",
        ));
    }
    let zero_weight_windows = (window_weights.len() - positive.len()) as u64;
    let retained = crate::sampling_weights::RetainedSamplingWeights::from_positive(
        &positive,
        zero_weight_windows,
    );
    let scale = positive.len() as f64 / positive.iter().sum::<f64>();
    let normalized: Vec<f64> = window_weights.iter().map(|w| w * scale).collect();
    Ok((crate::sampling_weights::root_weight_design(design, &normalized), Some(retained.record)))
}

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
        self.prepare_with_extras(
            &TemporalDesignInputs { data, estimand, query, indexer, split, policy },
            &[],
        )
    }

    /// Like [`Self::prepare`], with optional lag-0 schema covariates.
    ///
    /// # Errors
    ///
    /// Incompatible estimand, missing columns, or sample preparation failures.
    pub fn prepare_with_extras(
        &self,
        inputs: &TemporalDesignInputs<'_>,
        extra_contemporaneous: &[VariableId],
    ) -> Result<PreparedEstimationProblem, EstimationError> {
        self.prepare_aligned_with_extras(inputs, extra_contemporaneous).map(|(prep, _)| prep)
    }

    /// Like [`Self::prepare`], also returning which series times the design rows
    /// read: design row `r` is the lag window ending at series time
    /// `rows.first_time + r`. Multi-design bootstraps use this to refit every
    /// design on the same resampled times
    /// ([`crate::temporal_block::aligned_block_bootstrap`]).
    ///
    /// # Errors
    ///
    /// As [`Self::prepare`].
    pub fn prepare_aligned(
        &self,
        data: &TimeSeriesData,
        estimand: &IdentifiedEstimand,
        query: &TemporalEffectQuery,
        indexer: &TemporalIndexer,
        split: Option<&DiscoveryEstimationSplit>,
        policy: &antecedent_core::KernelPolicy,
    ) -> Result<(PreparedEstimationProblem, crate::temporal_block::AlignedRows), EstimationError>
    {
        self.prepare_aligned_with_extras(
            &TemporalDesignInputs { data, estimand, query, indexer, split, policy },
            &[],
        )
    }

    fn prepare_aligned_with_extras(
        &self,
        inputs: &TemporalDesignInputs<'_>,
        extra_contemporaneous: &[VariableId],
    ) -> Result<(PreparedEstimationProblem, crate::temporal_block::AlignedRows), EstimationError>
    {
        self.prepare_windows(inputs, extra_contemporaneous, false)
            .map(|(prep, aligned, _)| (prep, aligned))
    }

    /// Like [`Self::prepare`], on a series that may have missing time points.
    ///
    /// Windows that read a missing value are dropped and every other window keeps its own
    /// time and its own lags, so nothing is concatenated across a gap. The design holds the
    /// intact windows in time order. Valid for the point estimate under outcome and
    /// covariate missingness that is independent of the innovation given the retained
    /// lagged regressors; resampling consecutive windows (the block bootstrap) is not
    /// defined across gaps, so a caller must not bootstrap the returned design.
    ///
    /// # Errors
    ///
    /// As [`Self::prepare`], and a split series (its row ranges index complete windows).
    pub fn prepare_with_gaps(
        &self,
        data: &TimeSeriesData,
        estimand: &IdentifiedEstimand,
        query: &TemporalEffectQuery,
        indexer: &TemporalIndexer,
        policy: &antecedent_core::KernelPolicy,
    ) -> Result<(PreparedEstimationProblem, GapSelection), EstimationError> {
        let (prep, _, selection) = self.prepare_windows(
            &TemporalDesignInputs { data, estimand, query, indexer, split: None, policy },
            &[],
            true,
        )?;
        let selection = selection.ok_or(EstimationError::Unsupported {
            message: "gap-aware preparation did not report its windows",
        })?;
        Ok((prep, selection))
    }

    /// [`Self::prepare_with_gaps`] with lag-0 schema covariates, for refits of a perturbed
    /// series that keeps its gaps (the refuters).
    ///
    /// # Errors
    ///
    /// As [`Self::prepare_with_gaps`].
    pub fn prepare_with_gaps_extras(
        &self,
        inputs: &TemporalDesignInputs<'_>,
        extra_contemporaneous: &[VariableId],
    ) -> Result<(PreparedEstimationProblem, GapSelection), EstimationError> {
        let inputs = TemporalDesignInputs { split: None, ..*inputs };
        let (prep, _, selection) = self.prepare_windows(&inputs, extra_contemporaneous, true)?;
        let selection = selection.ok_or(EstimationError::Unsupported {
            message: "gap-aware preparation did not report its windows",
        })?;
        Ok((prep, selection))
    }

    #[allow(
        clippy::too_many_lines,
        reason = "resolve lag-aligned columns, prepare the sample, then split and gather the design inputs; every step reads the same aligned row range"
    )]
    fn prepare_windows(
        &self,
        inputs: &TemporalDesignInputs<'_>,
        extra_contemporaneous: &[VariableId],
        allow_gaps: bool,
    ) -> Result<
        (PreparedEstimationProblem, crate::temporal_block::AlignedRows, Option<GapSelection>),
        EstimationError,
    > {
        let TemporalDesignInputs { data, estimand, query, indexer, split, policy } = *inputs;
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
        // The anchor is the history operator's admitted horizon, not the query's
        // step count.
        let sample_anchor = outcome_sample_anchor(data.time_index(), query)?.max(0);
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
        let (prep, complete) = if allow_gaps {
            let gapped = plan
                .prepare_with_gaps(data, &mut sample_ws, policy)
                .map_err(EstimationError::from)?;
            (gapped.sample, Some(gapped.complete))
        } else {
            (plan.prepare(data, &mut sample_ws, policy).map_err(EstimationError::from)?, None)
        };

        let split = match split {
            Some(split) => Some(
                split
                    .bind_embargo(
                        antecedent_core::DeclaredSamplingDesign::DependentTimeSeries,
                        max_lag,
                    )
                    .map_err(EstimationError::from)?,
            ),
            None => None,
        };
        let split = split.as_ref();
        let n = prep.n;
        let (row_start, row_end) = if let Some(s) = split {
            // Prepared row `r` is the lag window `[r, r + max_lag]`. A row belongs to the
            // estimation half only when its whole window does: `r >= estimation.start` (so
            // no lagged regressor reads a discovery-half observation) and
            // `r + max_lag < estimation.end`.
            let est_start = s.estimation.start;
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
        // Intact windows only, in time order; every window when the series has no gaps.
        let selected: Vec<usize> = match &complete {
            Some(intact) => (0..nrows).filter(|&r| intact[row_start + r]).collect(),
            None => (0..nrows).collect(),
        };
        if allow_gaps && selected.len() < 3 {
            return Err(EstimationError::data_msg(
                "fewer than three complete lag windows remain after dropping windows that read a missing value",
            ));
        }
        // The design holds only the selected windows; `selected` labels each design row with
        // its source window.
        let gather = |values: &[f64]| -> Vec<f64> { selected.iter().map(|&r| values[r]).collect() };
        let (t_sel, y_sel) = (gather(t), gather(y));
        let cov_sel: Vec<(VariableId, Vec<f64>)> =
            cov_refs.iter().map(|(id, values)| (*id, gather(values))).collect();
        let cov_sel_refs: Vec<(VariableId, &[f64])> =
            cov_sel.iter().map(|(id, values)| (*id, values.as_slice())).collect();
        let design = CompiledDesign::linear_adjustment(&t_sel, &cov_sel_refs, &y_sel, &selected)
            .map_err(EstimationError::from)?;
        let (design, sampling_weights) =
            weight_windows(design, data, max_lag as usize + row_start, nrows, allow_gaps)?;
        let selection = complete.as_ref().map(|intact| GapSelection {
            first_time: max_lag as usize + row_start,
            complete: intact[row_start..row_end].to_vec(),
        });

        let active = intervention_f64(&query.active)?;
        let control = intervention_f64(&query.control)?;
        let treatment_delta = active - control;
        if treatment_delta == 0.0 {
            return Err(EstimationError::unsupported(
                "active and control treatment levels must differ",
            ));
        }

        let aligned = crate::temporal_block::AlignedRows {
            first_time: max_lag as usize + row_start,
            rows: nrows,
        };
        Ok((
            PreparedEstimationProblem {
                design,
                method: Arc::from("temporal.linear.adjustment"),
                adjustment_set: Arc::from(adj_keys),
                overlap: self.inner.overlap,
                treatment_delta,
                target_population: TargetPopulation::AllObserved,
                treatment: selected.iter().map(|&r| t[r]).collect(),
                active,
                control,
                sampling_weights,
                preparation: antecedent_data::materialize::PreparationReceipt::composed_finite(
                    selected.len(),
                )?,
            },
            aligned,
            selection,
        ))
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
        let units = self.prepare_panel_units(panel, estimand, query, indexer, split, policy)?;
        let ids: Vec<u32> = panel.units().iter().map(|unit| unit.unit_id).collect();
        self.stack_panel_units(&units.iter().collect::<Vec<_>>(), &ids)
    }

    /// Prepare every panel unit's own lag-aligned design (one [`Self::prepare`] per
    /// unit, so no lag window crosses a unit boundary), in panel order.
    ///
    /// A unit cluster bootstrap prepares once and restacks these blocks per replicate
    /// with [`Self::stack_panel_units`].
    ///
    /// # Errors
    ///
    /// Empty panel, incompatible estimand, or per-unit preparation failures.
    pub fn prepare_panel_units(
        &self,
        panel: &antecedent_data::PanelData,
        estimand: &IdentifiedEstimand,
        query: &TemporalEffectQuery,
        indexer: &TemporalIndexer,
        split: Option<&DiscoveryEstimationSplit>,
        policy: &antecedent_core::KernelPolicy,
    ) -> Result<Vec<PreparedEstimationProblem>, EstimationError> {
        if panel.unit_count() == 0 {
            return Err(EstimationError::data_msg("panel needs ≥1 unit"));
        }
        panel
            .units()
            .iter()
            .map(|unit| self.prepare(&unit.series, estimand, query, indexer, split, policy))
            .collect()
    }

    /// Stack per-unit prepared blocks into one pooled design.
    ///
    /// `cluster_ids[k]` labels every row of `units[k]`; a resampled panel passes one
    /// fresh id per draw so a unit drawn twice forms two clusters. Returns
    /// `(problem, row_cluster_ids, panel_times)` with consecutive within-block times.
    ///
    /// # Errors
    ///
    /// No blocks, mismatched ids, blocks from different designs, or design failure.
    pub fn stack_panel_units(
        &self,
        units: &[&PreparedEstimationProblem],
        cluster_ids: &[u32],
    ) -> Result<(PreparedEstimationProblem, Vec<u32>, Vec<i64>), EstimationError> {
        let Some(first) = units.first() else {
            return Err(EstimationError::data_msg("panel needs ≥1 unit"));
        };
        if cluster_ids.len() != units.len() {
            return Err(EstimationError::data_msg("one cluster id per stacked panel unit"));
        }
        let adj_keys = first.adjustment_set.to_vec();
        let total_rows: usize = units.iter().map(|unit| unit.design.nrows).sum();
        let mut all_t = Vec::with_capacity(total_rows);
        let mut all_y = Vec::with_capacity(total_rows);
        let mut all_covs: Vec<(VariableId, Vec<f64>)> =
            adj_keys.iter().map(|&id| (id, Vec::with_capacity(total_rows))).collect();
        let mut row_clusters = Vec::with_capacity(total_rows);
        let mut panel_times = Vec::with_capacity(total_rows);
        for (prep, &cluster) in units.iter().zip(cluster_ids) {
            if prep.adjustment_set.as_ref() != adj_keys.as_slice()
                || prep.design.ncols != first.design.ncols
            {
                return Err(EstimationError::data_msg(
                    "stacked panel units must share one adjustment design",
                ));
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
            row_clusters.extend(std::iter::repeat_n(cluster, n));
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
        let rows = all_t.len();

        Ok((
            PreparedEstimationProblem {
                design,
                method: Arc::from("temporal.linear.adjustment.panel"),
                adjustment_set: Arc::from(adj_keys),
                overlap: self.inner.overlap,
                treatment_delta: first.treatment_delta,
                target_population: TargetPopulation::AllObserved,
                treatment: Arc::from(all_t),
                active: first.active,
                control: first.control,
                sampling_weights: None,
                preparation: antecedent_data::materialize::PreparationReceipt::composed_finite(
                    rows,
                )?,
            },
            row_clusters,
            panel_times,
        ))
    }

    /// Fit one series' prepared design with dependence-honest uncertainty.
    ///
    /// Lag-aligned rows of one series are serially dependent (overlapping lag
    /// windows, autocorrelated residuals, MA(h−1) errors at horizon h), so neither
    /// the homoskedastic OLS SE nor an iid row bootstrap is valid here:
    ///
    /// - `se_bootstrap` (when `inner.bootstrap_replicates > 0`) is a circular-block
    ///   bootstrap over consecutive lag-aligned rows
    ///   ([`crate::temporal_block::row_block_bootstrap_vec`]); each replicate refits
    ///   the design. The block length is
    ///   [`crate::temporal_block::dependence_block_length`]: at least
    ///   [`antecedent_data::circular_block_length`] of the unfolded window
    ///   (`history + horizon` slices) and the row count, lengthened when the
    ///   treatment influence or any normal-equation score of the regression
    ///   (the residual included) is persistently dependent.
    ///   The replicate SD carries the circular fixed-b factor of that length and
    ///   the Bartlett kernel-bias factor of the treatment influence
    ///   ([`crate::temporal_block::kernel_bias_scale`]).
    /// - `se_analytic` is NaN. No analytic SE is calibrated here: the iid OLS SE
    ///   ignores the dependence, and a Newey–West HAC SE at the same bandwidth
    ///   under-covered in calibration (0.82–0.88 at nominal 0.90).
    ///
    /// `inner.se_kind` and the iid bootstrap of [`LinearAdjustmentAte`] are never used.
    ///
    /// `prep` comes from [`Self::prepare`] on one series (rows are consecutive in
    /// time; under a split, the estimation window only); `indexer` is the
    /// identification indexer that sets the structural span.
    ///
    /// # Errors
    ///
    /// Point-fit failures. Failed replicates are counted, not raised.
    pub fn fit_dependence_honest(
        &self,
        prep: &PreparedEstimationProblem,
        indexer: &TemporalIndexer,
        ctx: &ExecutionContext,
        assumptions: AssumptionSet,
    ) -> Result<(EffectEstimate, TemporalDependenceSe), EstimationError> {
        let rows = prep.design.nrows;
        let fitter = self.point_fitter();
        let mut workspace = EstimationWorkspace::default();
        let mut point = fitter.fit_point(prep, &mut workspace, assumptions)?;
        point.uncertainty.analytic.se = f64::NAN;

        let replicates = self.inner.bootstrap_replicates;
        // The dependence-aware length refits the design for its normal-equation
        // scores and runs a Politis–White scan on each; without replicates no
        // interval is published, so the rule length stands in (the diagnostic
        // says so through `replicates_attempted == 0`).
        let block_length = if replicates > 0 {
            single_window_block_length(prep, indexer, point.scores.influence.as_deref())?
        } else {
            let structural_span = (indexer.history() as usize + indexer.horizon() as usize).max(1);
            antecedent_data::circular_block_length(structural_span, analysed_windows(prep))
        };
        let target: Vec<&[f64]> = point.scores.influence.as_deref().into_iter().collect();
        let kernel_bias = crate::temporal_block::kernel_bias_scale(&target, block_length);
        let boot = if replicates > 0 {
            // The gather buffers hold the design and outcome of one resample.
            let _gather =
                crate::util::admit_gather_scratch(ctx, rows, prep.design.ncols + 1, replicates)?;
            let mut x_boot = vec![0.0; rows * prep.design.ncols];
            let mut y_boot = vec![0.0; rows];
            let draws = crate::temporal_block::row_block_bootstrap_vec(
                rows,
                block_length,
                replicates,
                TEMPORAL_BLOCK_STREAM,
                ctx,
                |row_src| {
                    fitter
                        .ate_on_row_indices_into(
                            prep,
                            &mut workspace,
                            row_src,
                            &mut x_boot,
                            &mut y_boot,
                        )
                        .ok()
                        .map(|ate| vec![ate])
                },
            );
            draws.ensure_admitted()?;
            Some(draws)
        } else {
            None
        };
        let info = TemporalDependenceSe {
            block_length,
            rows,
            effective_rows: crate::temporal_block::score_effective_rows(
                &point.scores.influence.as_deref().into_iter().collect::<Vec<_>>(),
                block_length,
            ),
            replicates_attempted: boot.as_ref().map_or(0, |b| b.attempted),
            kernel_bias,
        };
        let boot = boot.map(|mut b| {
            b.kernel_bias = kernel_bias;
            b.se_result(0)
        });
        let mut estimate = point.with_bootstrap(boot);
        // The geometry the interval resampled, so a check of that interval
        // (`bootstrap.ci_coverage`) can reuse the length instead of re-deriving it.
        estimate.uncertainty.block.resampling =
            Some(crate::adjustment::BlockResampling { block_length, rows, kernel_bias });
        Ok((estimate, info))
    }

    /// [`Self::fit_dependence_honest`] for a design built by [`Self::prepare_with_gaps`].
    ///
    /// The circular blocks run over the original window axis, so two windows are in one block
    /// only when they are adjacent in time; a replicate keeps the intact windows its blocks
    /// cover, so a block never joins windows across a gap and the replicate size varies.
    /// The block length is the structural rule (history plus horizon, at least the cube root
    /// of the window count) on the window axis. The Bartlett kernel-bias inflation of
    /// [`Self::fit_dependence_honest`] is not applied: it is defined for consecutive scores.
    ///
    /// # Errors
    ///
    /// Point-fit failures. Failed replicates are counted, not raised.
    pub fn fit_dependence_honest_gapped(
        &self,
        prep: &PreparedEstimationProblem,
        selection: &GapSelection,
        indexer: &TemporalIndexer,
        ctx: &ExecutionContext,
        assumptions: AssumptionSet,
    ) -> Result<(EffectEstimate, TemporalDependenceSe), EstimationError> {
        let rows = prep.design.nrows;
        let windows = selection.complete.len();
        let fitter = self.point_fitter();
        let mut workspace = EstimationWorkspace::default();
        let mut point = fitter.fit_point(prep, &mut workspace, assumptions)?;
        point.uncertainty.analytic.se = f64::NAN;
        let structural_span = (indexer.history() as usize + indexer.horizon() as usize).max(1);
        let block_length = antecedent_data::circular_block_length(structural_span, windows);
        let replicates = self.inner.bootstrap_replicates;
        let p = prep.design.ncols;
        let boot = if replicates > 0 {
            // A resample keeps at most every window, so one gather holds `windows` rows.
            let _gather = crate::util::admit_gather_scratch(ctx, windows, p + 1, replicates)?;
            let draws = crate::temporal_block::gapped_row_block_bootstrap(
                &selection.complete,
                block_length,
                replicates,
                TEMPORAL_BLOCK_STREAM,
                ctx,
                |row_src| {
                    if row_src.len() <= p {
                        return None;
                    }
                    let mut x_boot = vec![0.0; row_src.len() * p];
                    let mut y_boot = vec![0.0; row_src.len()];
                    fitter
                        .ate_on_row_indices_into(
                            prep,
                            &mut workspace,
                            row_src,
                            &mut x_boot,
                            &mut y_boot,
                        )
                        .ok()
                        .map(|ate| vec![ate])
                },
            );
            draws.ensure_admitted()?;
            Some(draws)
        } else {
            None
        };
        let info = TemporalDependenceSe {
            block_length,
            rows,
            effective_rows: crate::temporal_block::score_effective_rows(
                &point.scores.influence.as_deref().into_iter().collect::<Vec<_>>(),
                block_length,
            ),
            replicates_attempted: boot.as_ref().map_or(0, |b| b.attempted),
            kernel_bias: 1.0,
        };
        let boot = boot.map(|b| b.se_result(0));
        let mut estimate = point.with_bootstrap(boot);
        estimate.uncertainty.block.resampling = Some(crate::adjustment::BlockResampling {
            block_length,
            rows: windows,
            kernel_bias: 1.0,
        });
        Ok((estimate, info))
    }

    /// Circular-block length [`Self::fit_dependence_honest`] resamples `prep` with,
    /// so a check of that interval (`bootstrap.ci_coverage`) can resample the same
    /// blocks without running the bootstrap.
    ///
    /// # Errors
    ///
    /// Point-fit failures.
    pub fn dependence_block_length(
        &self,
        prep: &PreparedEstimationProblem,
        indexer: &TemporalIndexer,
    ) -> Result<usize, EstimationError> {
        let point = self.point_fitter().fit_point(
            prep,
            &mut EstimationWorkspace::default(),
            AssumptionSet::default(),
        )?;
        single_window_block_length(prep, indexer, point.scores.influence.as_deref())
    }

    /// The OLS point fitter of [`Self::fit_dependence_honest`] (no iid bootstrap or
    /// cluster SEs).
    fn point_fitter(&self) -> LinearAdjustmentAte {
        LinearAdjustmentAte {
            bootstrap_replicates: 0,
            se_kind: AnalyticSeKind::Homoskedastic,
            cluster_ids: None,
            multiway_ids: None,
            panel_times: None,
            ..self.inner.clone()
        }
    }

    /// Fit using the shared linear-adjustment path.
    ///
    /// This is the iid path of [`LinearAdjustmentAte`] (homoskedastic analytic SE,
    /// iid bootstrap over prepared rows). One-series temporal effects use
    /// [`Self::fit_dependence_honest`] instead.
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

/// Marker between a variable name and its lag in temporal coefficient names
/// (`coef_pressure@lag1`). Names carrying it bind prior transfer by lag.
pub(crate) const TEMPORAL_COEF_LAG_MARKER: &str = "@lag";

/// Whether a durable coefficient name carries a temporal lag coordinate.
#[must_use]
pub fn is_temporal_coefficient_name(name: &str) -> bool {
    name.contains(TEMPORAL_COEF_LAG_MARKER)
}

/// Durable, lag-aware coefficient names for the design [`TemporalLinearAdjustment::prepare`]
/// builds: `intercept`, `coef_<treatment>@lag<k>`, then `coef_<covariate>@lag<k>` in
/// adjustment order. Lags are relative to the design's sample anchor (the latest queried
/// outcome), i.e. they are the regression coordinates a transferred coefficient binds to.
///
/// # Errors
///
/// The same offset / indexer failures as `prepare`.
pub fn temporal_coefficient_names(
    data: &TimeSeriesData,
    estimand: &IdentifiedEstimand,
    query: &TemporalEffectQuery,
    indexer: &TemporalIndexer,
) -> Result<Arc<[Arc<str>]>, EstimationError> {
    use antecedent_data::TableView;
    let schema = data.schema();
    let name = |id: VariableId| {
        schema.get(id).map_or_else(|_| format!("var_{}", id.raw()), |v| v.name.as_ref().to_string())
    };
    let sample_anchor = outcome_sample_anchor(data.time_index(), query)?.max(0);
    let t_lag = offset_to_lag(query.try_treatment_offset()? - sample_anchor)?;
    let mut names: Vec<Arc<str>> = vec![
        Arc::from("intercept"),
        Arc::from(format!(
            "coef_{}{TEMPORAL_COEF_LAG_MARKER}{}",
            name(query.treatment),
            t_lag.raw()
        )),
    ];
    for &dense_var in estimand.adjustment_set.iter() {
        let key = indexer
            .key_of(dense_var.raw())
            .map_err(|e| EstimationError::data_msg(e.to_string()))?;
        let lag = offset_to_lag(key.offset - sample_anchor)?;
        names.push(Arc::from(format!(
            "coef_{}{TEMPORAL_COEF_LAG_MARKER}{}",
            name(key.variable),
            lag.raw()
        )));
    }
    Ok(Arc::from(names))
}

/// History, action time, censoring and assignment weights, and horizon admitted
/// together. The horizon is the operator's; a step count is only what that
/// horizon covers after the cadence check.
#[allow(
    dead_code,
    reason = "the adjustment reads the horizon steps; action time and weights are admitted with it"
)]
struct AdmittedTemporalHistory {
    history: HistoryConstruction,
    action: ActionTiming,
    weights: CensoringAssignmentWeight,
}

#[allow(
    dead_code,
    reason = "tests read the admitted history, action time, and weights beside the horizon"
)]
impl AdmittedTemporalHistory {
    const fn horizon(&self) -> Horizon {
        self.history.frame().horizon()
    }

    const fn history(&self) -> HistoryWindow {
        self.history.history()
    }

    const fn action_ns(&self) -> u64 {
        self.action.at_ns()
    }

    const fn censoring(&self) -> f64 {
        self.weights.censoring()
    }

    const fn assignment(&self) -> f64 {
        self.weights.assignment()
    }

    /// Ticks from the origin to the admitted horizon end.
    ///
    /// # Errors
    ///
    /// The tick count does not fit in `u32`.
    fn horizon_steps(&self) -> Result<u32, EstimationError> {
        let ticks = self.horizon().end_ns() / self.history.frame().interval_ns();
        u32::try_from(ticks)
            .map_err(|_| EstimationError::unsupported("admitted horizon does not fit in u32 steps"))
    }
}

/// Admit `horizon` with the history, action time, and weights the operators record.
///
/// # Errors
///
/// The history operators refuse the cadence, the horizon, the initial law, the
/// action time, or a weight.
#[must_use = "a refused history is not a horizon"]
fn admit_temporal_history(
    cadence: &TimeIndex,
    horizon: Horizon,
    initial_law: InitialLaw,
    history: HistoryWindow,
    action_ns: u64,
    censoring: f64,
    assignment: f64,
) -> Result<AdmittedTemporalHistory, EstimationError> {
    let history =
        HistoryConstruction::try_new(cadence.clone(), horizon, initial_law.clone(), history)
            .map_err(|error| history_operator_refusal(&error))?;
    let action = ActionTiming::try_new(cadence.clone(), horizon, initial_law.clone(), action_ns)
        .map_err(|error| history_operator_refusal(&error))?;
    let weights = CensoringAssignmentWeight::try_new(
        cadence.clone(),
        horizon,
        initial_law,
        censoring,
        assignment,
    )
    .map_err(|error| history_operator_refusal(&error))?;
    Ok(AdmittedTemporalHistory { history, action, weights })
}

fn history_operator_refusal(error: &HistoryOperatorError) -> EstimationError {
    EstimationError::data_msg(error.to_string())
}

/// Nanosecond end of `horizon_steps` on `cadence`.
///
/// An irregular cadence has no tick the end can land on. The end passed to the
/// operator is then off the cadence, and the operator refuses it.
///
/// # Errors
///
/// `horizon_steps` times the regular interval does not fit in `u64`.
fn query_horizon_end(cadence: &TimeIndex, horizon_steps: u32) -> Result<Horizon, EstimationError> {
    let SamplingRegularity::Regular { interval_ns } = cadence.regularity else {
        return Ok(Horizon::ending_at(1));
    };
    let end_ns = u64::from(horizon_steps)
        .checked_mul(interval_ns)
        .ok_or(EstimationError::unsupported("horizon end in nanoseconds does not fit in u64"))?;
    Ok(Horizon::ending_at(end_ns))
}

/// Sample anchor of `query`: one less than the horizon the operators admitted.
///
/// # Errors
///
/// The operators refuse the query horizon on `cadence`.
fn outcome_sample_anchor(
    cadence: &TimeIndex,
    query: &TemporalEffectQuery,
) -> Result<i32, EstimationError> {
    let history = match query.max_history_lag {
        Some(lag) => HistoryWindow::max_lag(lag),
        // With no declared cap the planner retains every tick the series has.
        None => HistoryWindow::planner_default()
            .resolve(u32::try_from(cadence.length).unwrap_or(u32::MAX)),
    };
    let admitted = admit_temporal_history(
        cadence,
        query_horizon_end(cadence, query.horizon_steps)?,
        InitialLaw::contemporaneous_observed(),
        history,
        0,
        1.0,
        1.0,
    )?;
    let steps = admitted.horizon_steps()?;
    Ok(i32::try_from(steps.saturating_sub(1)).unwrap_or(i32::MAX))
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

/// Posterior-mean coefficients under an isotropic Gaussian prior of standard
/// deviation `scale` on every non-intercept column, using the least-squares
/// residual variance of `ols_coef` as a plug-in. Solved as the ridge-augmented
/// least-squares problem; the intercept column (the first) is unpenalized.
fn slope_ridge_coefficients(
    design: &[f64],
    n: usize,
    ncols: usize,
    y: &[f64],
    ols_coef: &[f64],
    scale: f64,
    invalid_scale: &'static str,
) -> Result<Vec<f64>, EstimationError> {
    if !(scale.is_finite() && scale > 0.0) {
        return Err(EstimationError::unsupported(invalid_scale));
    }
    let sigma2 = crate::util::ols_sigma2(design, n, ncols, y, ols_coef).max(1e-12);
    let penalty = sigma2 / (scale * scale);
    let aug_n = n + (ncols - 1);
    let mut aug_x = vec![0.0; aug_n * ncols];
    let mut aug_y = vec![0.0; aug_n];
    for col in 0..ncols {
        aug_x[col * aug_n..col * aug_n + n].copy_from_slice(&design[col * n..(col + 1) * n]);
    }
    aug_y[..n].copy_from_slice(y);
    let root = penalty.sqrt();
    for col in 1..ncols {
        let row = n + col - 1;
        aug_x[col * aug_n + row] = root;
    }
    let mut aug_ws = antecedent_stats::LeastSquaresWorkspace::default();
    Ok(antecedent_stats::FaerBackend
        .least_squares(&aug_x, aug_n, ncols, &aug_y, &mut aug_ws)
        .map_err(crate::util::stats_err)?
        .coefficients)
}

/// Marginal temporal conditional effect for one modifier.
///
/// The unfolded regression is `Y ~ 1 + T + W + T×W + Z`. The published point is
/// `(β_T + β_{T×W} · mean(W)) · (active − control)`, the same functional as the
/// static one-modifier conditional effect, on the lag-aligned rows of the
/// temporal backdoor design. `prior_scale` is the isotropic Gaussian prior
/// standard deviation on the slope coefficients. `None` is ordinary least
/// squares. `Some` replaces that point with the posterior mean under that
/// prior, using the least-squares residual variance as a plug-in. No interval
/// is returned.
///
/// # Errors
///
/// The query is not a single-modifier conditional effect, the design is
/// singular or too small, or the temporal sample cannot be aligned.
pub fn temporal_conditional_marginal_effect(
    data: &TimeSeriesData,
    estimand: &IdentifiedEstimand,
    query: &TemporalEffectQuery,
    indexer: &TemporalIndexer,
    policy: &antecedent_core::KernelPolicy,
    prior_scale: Option<f64>,
) -> Result<f64, EstimationError> {
    temporal_conditional_marginal_effect_with_gaps(
        data,
        estimand,
        query,
        indexer,
        policy,
        prior_scale,
        false,
    )
    .map(|(effect, _)| effect)
}

/// [`temporal_conditional_marginal_effect`] on a series that may have missing time points.
///
/// With `allow_gaps` the windows that read a missing value are dropped and the surviving
/// windows keep their own times and lags (nothing is concatenated across a gap); without it a
/// missing value refuses as before.
///
/// # Errors
///
/// As [`temporal_conditional_marginal_effect`]. Also returns, with `allow_gaps`, which windows
/// were intact.
pub fn temporal_conditional_marginal_effect_with_gaps(
    data: &TimeSeriesData,
    estimand: &IdentifiedEstimand,
    query: &TemporalEffectQuery,
    indexer: &TemporalIndexer,
    policy: &antecedent_core::KernelPolicy,
    prior_scale: Option<f64>,
    allow_gaps: bool,
) -> Result<(f64, Option<GapSelection>), EstimationError> {
    if query.effect_modifiers.len() != 1 {
        return Err(EstimationError::unsupported(
            "temporal conditional effect fits exactly one modifier",
        ));
    }
    query.validate()?;
    if query.target_population != TargetPopulation::AllObserved {
        return Err(EstimationError::TargetPopulation);
    }
    let active = intervention_f64(&query.active)?;
    let control = intervention_f64(&query.control)?;
    let delta = active - control;
    if delta == 0.0 {
        return Err(EstimationError::unsupported(
            "active and control treatment levels must differ",
        ));
    }
    let sample_anchor = query.outcome_offset().max(0);
    let t_lag = offset_to_lag(query.try_treatment_offset()? - sample_anchor)?;
    let y_lag = offset_to_lag(query.outcome_offset() - sample_anchor)?;
    let modifier = query.effect_modifiers[0];
    let mut cols = vec![
        LaggedColumn { variable: query.treatment, lag: t_lag },
        LaggedColumn { variable: query.outcome, lag: y_lag },
        LaggedColumn { variable: modifier, lag: y_lag },
    ];
    let mut adjustment = Vec::new();
    for &dense_var in estimand.adjustment_set.iter() {
        let key = indexer
            .key_of(dense_var.raw())
            .map_err(|e| EstimationError::data_msg(e.to_string()))?;
        if key.variable == modifier
            || key.variable == query.treatment
            || key.variable == query.outcome
        {
            continue;
        }
        let lag = offset_to_lag(key.offset - sample_anchor).map_err(|_| {
            EstimationError::unsupported(
                "adjustment node sits after this horizon's sample anchor and cannot be lag-aligned",
            )
        })?;
        cols.push(LaggedColumn { variable: key.variable, lag });
        adjustment.push(key.variable);
    }
    let max_lag = cols.iter().map(|c| c.lag.raw()).max().unwrap_or(0);
    let plan = data
        .plan_lagged_sample(max_lag, Arc::<[LaggedColumn]>::from(cols))
        .map_err(EstimationError::from)?;
    let mut sample_ws = LaggedSampleWorkspace::default();
    let (prep, selection) = if allow_gaps {
        let (prep, complete) = plan
            .prepare_complete_windows(data, &mut sample_ws, policy)
            .map_err(EstimationError::from)?;
        (prep, Some(GapSelection { first_time: max_lag as usize, complete }))
    } else {
        (plan.prepare(data, &mut sample_ws, policy).map_err(EstimationError::from)?, None)
    };
    let n = prep.n;
    if n < 8 {
        return Err(EstimationError::data_msg(
            "too few aligned rows for a temporal conditional effect",
        ));
    }
    let t = prep.column(0);
    let y = prep.column(1);
    let w = prep.column(2);
    let n_z = adjustment.len();
    let ncols = 4 + n_z;
    let mut design = vec![0.0; n * ncols];
    for i in 0..n {
        design[i] = 1.0;
        design[n + i] = t[i];
        design[2 * n + i] = w[i];
        design[3 * n + i] = t[i] * w[i];
    }
    for k in 0..n_z {
        let column = prep.column(3 + k);
        let base = (4 + k) * n;
        design[base..base + n].copy_from_slice(column);
    }
    let mut ws = antecedent_stats::LeastSquaresWorkspace::default();
    let fit = antecedent_stats::FaerBackend
        .least_squares(&design, n, ncols, y, &mut ws)
        .map_err(crate::util::stats_err)?;
    let mut coef = fit.coefficients;
    if let Some(scale) = prior_scale {
        coef = slope_ridge_coefficients(
            &design,
            n,
            ncols,
            y,
            &coef,
            scale,
            "temporal conditional prior scale must be finite and positive",
        )?;
    }
    let w_bar = w.iter().sum::<f64>() / n as f64;
    Ok(((coef[1] + coef[3] * w_bar) * delta, selection))
}

/// The outcome, treatments and their time offsets of a [`temporal_linear_slopes`] fit.
#[derive(Clone, Copy, Debug)]
pub struct TemporalSlopeSpec<'a> {
    /// Outcome variable (at the outcome offset).
    pub outcome: VariableId,
    /// Treatments regressed on, none equal to the outcome.
    pub treatments: &'a [VariableId],
    /// Offset of the treatments from the policy origin.
    pub treatment_offset: i32,
    /// Offset of the outcome from the policy origin.
    pub outcome_offset: i32,
    /// Treatments read at the outcome's own lag rather than the treatment offset.
    pub contemporaneous: &'a [VariableId],
}

/// Intercept and partial slopes of one outcome on the named treatments.
///
/// The design is the unfolded temporal back-door regression: intercept, each
/// treatment at the policy origin, then the identified adjustment nodes.
/// `prior_scale` replaces the slope coefficients with their posterior means
/// under independent `N(0, scale²)` priors and the OLS residual variance; the
/// intercept stays unpenalized. No interval is computed.
///
/// # Errors
///
/// Refuses an empty treatment list, a treatment equal to the outcome, a
/// non-positive prior scale, fewer than eight aligned rows, or a lag the
/// sample cannot place.
pub fn temporal_linear_slopes(
    data: &TimeSeriesData,
    estimand: &IdentifiedEstimand,
    indexer: &TemporalIndexer,
    spec: &TemporalSlopeSpec<'_>,
    policy: &antecedent_core::KernelPolicy,
    prior_scale: Option<f64>,
) -> Result<(f64, Vec<f64>), EstimationError> {
    let TemporalSlopeSpec {
        outcome,
        treatments,
        treatment_offset,
        outcome_offset,
        contemporaneous,
    } = *spec;
    if treatments.is_empty() || treatments.contains(&outcome) {
        return Err(EstimationError::unsupported(
            "temporal linear slope needs treatments distinct from the outcome",
        ));
    }
    let sample_anchor = outcome_offset.max(0);
    let t_lag = offset_to_lag(treatment_offset - sample_anchor)?;
    let y_lag = offset_to_lag(outcome_offset - sample_anchor)?;
    let mut cols = Vec::with_capacity(treatments.len() + 1);
    for &treatment in treatments {
        let lag = if contemporaneous.contains(&treatment) { y_lag } else { t_lag };
        cols.push(LaggedColumn { variable: treatment, lag });
    }
    cols.push(LaggedColumn { variable: outcome, lag: y_lag });
    let treatment_count = treatments.len();
    let mut adjustment = Vec::new();
    for &dense_var in estimand.adjustment_set.iter() {
        let key = indexer
            .key_of(dense_var.raw())
            .map_err(|e| EstimationError::data_msg(e.to_string()))?;
        if key.variable == outcome || treatments.contains(&key.variable) {
            continue;
        }
        let lag = offset_to_lag(key.offset - sample_anchor).map_err(|_| {
            EstimationError::unsupported(
                "adjustment node sits after this horizon's sample anchor and cannot be lag-aligned",
            )
        })?;
        cols.push(LaggedColumn { variable: key.variable, lag });
        adjustment.push(key.variable);
    }
    let max_lag = cols.iter().map(|c| c.lag.raw()).max().unwrap_or(0);
    let plan = data
        .plan_lagged_sample(max_lag, Arc::<[LaggedColumn]>::from(cols))
        .map_err(EstimationError::from)?;
    let mut sample_ws = LaggedSampleWorkspace::default();
    let prep = plan.prepare(data, &mut sample_ws, policy).map_err(EstimationError::from)?;
    let n = prep.n;
    if n < 8 {
        return Err(EstimationError::data_msg("too few aligned rows for a temporal linear slope"));
    }
    let y = prep.column(treatment_count);
    let n_z = adjustment.len();
    let ncols = 1 + treatment_count + n_z;
    let mut design = vec![0.0; n * ncols];
    design[..n].fill(1.0);
    for k in 0..treatment_count {
        let column = prep.column(k);
        design[(1 + k) * n..(2 + k) * n].copy_from_slice(column);
    }
    for k in 0..n_z {
        let column = prep.column(treatment_count + 1 + k);
        let base = (1 + treatment_count + k) * n;
        design[base..base + n].copy_from_slice(column);
    }
    let mut ws = antecedent_stats::LeastSquaresWorkspace::default();
    let fit = antecedent_stats::FaerBackend
        .least_squares(&design, n, ncols, y, &mut ws)
        .map_err(crate::util::stats_err)?;
    let mut coef = fit.coefficients;
    if let Some(scale) = prior_scale {
        coef = slope_ridge_coefficients(
            &design,
            n,
            ncols,
            y,
            &coef,
            scale,
            "temporal slope prior scale must be finite and positive",
        )?;
    }
    Ok((coef[0], coef[1..=treatment_count].to_vec()))
}

/// Moments of the nested natural-direct-effect regression on a time series.
///
/// The design is the same empty-adjustment regression
/// [`temporal_linear_slopes`] fits for a nested point: intercept, treatment
/// one step before the outcome, and the mediator at the outcome time. A
/// positive `prior_scale` adds the slope-only ridge that function uses, so
/// the intercept moment stays the aligned row count.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NestedDirectFitMoments {
    /// Aligned rows that entered the gram.
    pub complete_case_rows: u64,
    /// Row-major gram of `[1, treatment, mediator]`.
    pub gram: [f64; 9],
    /// Cross product of that design with the outcome.
    pub outcome_cross: [f64; 3],
}

/// Outcome-regression moments behind a temporal nested natural direct effect.
///
/// # Errors
///
/// Same row, lag, and prior-scale refusals as [`temporal_linear_slopes`] for
/// this two-treatment design, plus a non-finite aligned column.
pub fn nested_direct_fit_moments(
    data: &TimeSeriesData,
    treatment: VariableId,
    mediator: VariableId,
    outcome: VariableId,
    policy: &KernelPolicy,
    prior_scale: Option<f64>,
) -> Result<NestedDirectFitMoments, EstimationError> {
    if treatment == outcome || mediator == outcome || treatment == mediator {
        return Err(EstimationError::unsupported(
            "temporal nested moments need distinct treatment, mediator, and outcome",
        ));
    }
    let t_lag = offset_to_lag(-1)?;
    let y_lag = offset_to_lag(0)?;
    let cols = [
        LaggedColumn { variable: treatment, lag: t_lag },
        LaggedColumn { variable: mediator, lag: y_lag },
        LaggedColumn { variable: outcome, lag: y_lag },
    ];
    let plan = data
        .plan_lagged_sample(1, Arc::<[LaggedColumn]>::from(cols))
        .map_err(EstimationError::from)?;
    let mut sample_ws = LaggedSampleWorkspace::default();
    let prep = plan.prepare(data, &mut sample_ws, policy).map_err(EstimationError::from)?;
    let n = prep.n;
    if n < 8 {
        return Err(EstimationError::data_msg("too few aligned rows for a temporal linear slope"));
    }
    let y = prep.column(2);
    let treatment_column = prep.column(0);
    let mediator_column = prep.column(1);
    if y.iter().chain(treatment_column).chain(mediator_column).any(|value| !value.is_finite()) {
        return Err(EstimationError::data_msg(
            "temporal nested moments require finite treatment, mediator, and outcome rows",
        ));
    }
    let ncols = 3;
    let mut design = vec![0.0; n * ncols];
    design[..n].fill(1.0);
    design[n..2 * n].copy_from_slice(treatment_column);
    design[2 * n..3 * n].copy_from_slice(mediator_column);
    let mut gram = [0.0_f64; 9];
    let mut outcome_cross = [0.0_f64; 3];
    for i in 0..n {
        let row = [design[i], design[n + i], design[2 * n + i]];
        for left in 0..3 {
            outcome_cross[left] += row[left] * y[i];
            for right in 0..3 {
                gram[left * 3 + right] += row[left] * row[right];
            }
        }
    }
    if let Some(scale) = prior_scale {
        if !(scale.is_finite() && scale > 0.0) {
            return Err(EstimationError::unsupported(
                "temporal slope prior scale must be finite and positive",
            ));
        }
        let mut ws = antecedent_stats::LeastSquaresWorkspace::default();
        let coef = antecedent_stats::FaerBackend
            .least_squares(&design, n, ncols, y, &mut ws)
            .map_err(crate::util::stats_err)?
            .coefficients;
        let sigma2 = crate::util::ols_sigma2(&design, n, ncols, y, &coef).max(1e-12);
        let penalty = sigma2 / (scale * scale);
        gram[4] += penalty;
        gram[8] += penalty;
    }
    let Ok(complete_case_rows) = u64::try_from(n) else {
        return Err(EstimationError::data_msg("aligned row count does not fit the moment record"));
    };
    Ok(NestedDirectFitMoments { complete_case_rows, gram, outcome_cross })
}

#[cfg(test)]
mod tests {
    use antecedent_core::{
        CausalSchemaBuilder, DistributionRef, ExecutionContext, HistoryWindow, InitialLaw, Lag,
        MeasurementSpec, PredicateExpr, RoleHint, SmallRoleSet, TargetPopulation,
        TemporalEffectQuery, TemporalPolicy, ValueType, VariableId,
    };
    use antecedent_data::{
        Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
        TimeSeriesData, ValidityBitmap,
    };
    use antecedent_graph::{TemporalDag, ensure_lagged};
    use antecedent_identify::TemporalBackdoorIdentifier;
    use antecedent_model::history_operators::Horizon;

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
    fn the_adjustment_horizon_is_the_operator_horizon() {
        let cadence =
            TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 10 }, length: 4 };
        let horizon = Horizon::ending_at(40);
        let admitted = admit_temporal_history(
            &cadence,
            horizon,
            InitialLaw::contemporaneous_observed(),
            HistoryWindow::max_lag(2),
            20,
            1.0,
            0.5,
        )
        .unwrap();
        assert_eq!(admitted.horizon(), horizon);
        assert_eq!(admitted.horizon().end_ns(), 40);
        assert_eq!(admitted.horizon_steps().unwrap(), 4);
        assert_eq!(admitted.history(), HistoryWindow::max_lag(2));
        assert_eq!(admitted.action_ns(), 20);
        assert_eq!(admitted.censoring().to_bits(), 1.0f64.to_bits());
        assert_eq!(admitted.assignment().to_bits(), 0.5f64.to_bits());

        let (data, g) = series();
        let query =
            TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
                .with_policy(TemporalPolicy::pulse(-1))
                .with_horizon_steps(1)
                .with_max_history_lag(Some(1));
        let id_res = TemporalBackdoorIdentifier::new().identify_temporal(&g, &query).unwrap();
        let steps = outcome_sample_anchor(data.time_index(), &query).unwrap();
        assert_eq!(steps, query.outcome_offset());
        let estimand = id_res.result.estimands.first().unwrap();
        TemporalLinearAdjustment::new()
            .prepare(
                &data,
                estimand,
                &query,
                &id_res.indexer,
                None,
                &ExecutionContext::for_tests(1).kernel_policy,
            )
            .unwrap();
    }

    #[test]
    fn an_off_cadence_horizon_is_refused() {
        let cadence =
            TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 10 }, length: 4 };
        let Err(error) = admit_temporal_history(
            &cadence,
            Horizon::ending_at(25),
            InitialLaw::contemporaneous_observed(),
            HistoryWindow::planner_default(),
            0,
            1.0,
            1.0,
        ) else {
            panic!("an off-cadence horizon was admitted");
        };
        let message = error.to_string();
        assert!(message.contains("crosses the cadence"), "{message}");
        assert!(message.contains("25"), "{message}");

        let (data, g) = series();
        let irregular = TimeSeriesData::try_new(
            data.storage().clone(),
            TimeIndex {
                regularity: SamplingRegularity::Irregular,
                length: data.time_index().length,
            },
        )
        .unwrap();
        let query =
            TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
                .with_policy(TemporalPolicy::pulse(-1))
                .with_horizon_steps(1)
                .with_max_history_lag(Some(1));
        let id_res = TemporalBackdoorIdentifier::new().identify_temporal(&g, &query).unwrap();
        let route_error = TemporalLinearAdjustment::new()
            .prepare(
                &irregular,
                id_res.result.estimands.first().unwrap(),
                &query,
                &id_res.indexer,
                None,
                &ExecutionContext::for_tests(1).kernel_policy,
            )
            .unwrap_err();
        let route_message = route_error.to_string();
        assert!(route_message.contains("crosses the cadence"), "{route_message}");
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
    fn dependence_block_length_is_the_published_interval_block() {
        let (data, g) = series();
        let q = TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
            .with_policy(TemporalPolicy::pulse(-1))
            .with_horizon_steps(1)
            .with_max_history_lag(Some(1));
        let id_res = TemporalBackdoorIdentifier::new().identify_temporal(&g, &q).unwrap();
        let estimand = id_res.result.estimands.first().unwrap();
        let ctx = ExecutionContext::for_tests(3);
        let mut est = TemporalLinearAdjustment::new();
        est.inner.bootstrap_replicates = 8;
        let prep =
            est.prepare(&data, estimand, &q, &id_res.indexer, None, &ctx.kernel_policy).unwrap();
        let (_, info) =
            est.fit_dependence_honest(&prep, &id_res.indexer, &ctx, AssumptionSet::new()).unwrap();
        // A check of the interval (no replicates of its own) resamples the same blocks.
        est.inner.bootstrap_replicates = 0;
        let block = est.dependence_block_length(&prep, &id_res.indexer).unwrap();
        assert_eq!(block, info.block_length);
        // The smooth residual is persistent, so its normal-equation score lengthens
        // the blocks past the n^(1/3) rule.
        assert!(block > antecedent_data::circular_block_length(2, info.rows), "block={block}");
    }

    #[test]
    fn estimation_split_rows_read_no_discovery_observation() {
        let (data, g) = series();
        let q = TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
            .with_policy(TemporalPolicy::pulse(-1))
            .with_horizon_steps(1)
            .with_max_history_lag(Some(1));
        let id_res = TemporalBackdoorIdentifier::new().identify_temporal(&g, &q).unwrap();
        let estimand = id_res.result.estimands.first().unwrap();
        let est = TemporalLinearAdjustment::new();
        let policy = &ExecutionContext::for_tests(1).kernel_policy;
        // Discovery [0, 99), embargo [99, 100), estimation [100, 240). max_lag = 1, so a
        // gap of 0 would leak across that embargo. Prepared row r spans [r, r + 1].
        let split = DiscoveryEstimationSplit::from_sizes(240, 99, 1, 140).unwrap();
        let leaked = DiscoveryEstimationSplit::from_sizes(240, 100, 0, 140).unwrap();
        let error = est
            .prepare_aligned(&data, estimand, &q, &id_res.indexer, Some(&leaked), policy)
            .expect_err("a zero gap leaks across the lag embargo");
        assert!(error.to_string().contains("embargo"), "{error}");
        let (prep, aligned) = est
            .prepare_aligned(&data, estimand, &q, &id_res.indexer, Some(&split), policy)
            .unwrap();
        // First row's window is [100, 101] (lagged regressor at time 100, no discovery
        // time), so it ends at 101; the last row's window ends at the series end (239).
        assert_eq!(aligned.first_time, 101);
        assert_eq!(aligned.rows, 139);
        assert_eq!(aligned.first_time + aligned.rows, 240);
        assert_eq!(prep.design.nrows, 139);
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
    #[test]
    #[allow(
        clippy::too_many_lines,
        reason = "the fixture builds the schema, series and storage inline before the moment assertions"
    )]
    #[allow(
        clippy::float_cmp,
        reason = "the intercept moment is the exact integer row count, summed from ones"
    )]
    fn nested_direct_moments_solve_to_the_fitted_slope() {
        let n = 80usize;
        let mut b = CausalSchemaBuilder::new();
        for (name, role) in [
            ("t", RoleHint::TreatmentCandidate),
            ("m", RoleHint::Context),
            ("y", RoleHint::OutcomeCandidate),
        ] {
            b.add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(role),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        }
        let schema = b.build().unwrap();
        let mut t = vec![0.0; n];
        let mut m = vec![0.0; n];
        let mut y = vec![0.0; n];
        for step in 1..n {
            t[step] = ((step as f64) * 0.17).sin();
            m[step] = 0.3 * t[step] + ((step as f64) * 0.41).cos();
            y[step] = 1.4 * t[step - 1] + 0.7 * m[step] + 0.01 * ((step as f64) * 0.09).sin();
        }
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
                    Arc::from(m),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(2),
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
        let estimand = antecedent_expr::IdentifiedEstimand::new(
            "backdoor.adjustment",
            Arc::from([]),
            Arc::from([]),
            Arc::from([]),
            antecedent_expr::ExprId::from_raw(0),
            None,
        );
        let indexer = TemporalIndexer::new(3, 1, 1).unwrap();
        let policy = ExecutionContext::for_tests(1).kernel_policy;
        for prior in [None, Some(10.0)] {
            let moments = nested_direct_fit_moments(
                &data,
                VariableId::from_raw(0),
                VariableId::from_raw(1),
                VariableId::from_raw(2),
                &policy,
                prior,
            )
            .unwrap();
            assert_eq!(moments.gram[0], moments.complete_case_rows as f64);
            let (_, slopes) = temporal_linear_slopes(
                &data,
                &estimand,
                &indexer,
                &TemporalSlopeSpec {
                    outcome: VariableId::from_raw(2),
                    treatments: &[VariableId::from_raw(0), VariableId::from_raw(1)],
                    treatment_offset: -1,
                    outcome_offset: 0,
                    contemporaneous: &[VariableId::from_raw(1)],
                },
                &policy,
                prior,
            )
            .unwrap();
            let beta = solve_moments(moments.gram, moments.outcome_cross).unwrap();
            let delta = 1.0 - 0.0;
            assert!(
                (beta[1] * delta - slopes[0] * delta).abs()
                    <= 1e-8 * (slopes[0] * delta).abs().max(1.0),
                "prior={prior:?} solved={} fitted={}",
                beta[1] * delta,
                slopes[0] * delta
            );
        }
    }

    fn solve_moments(gram: [f64; 9], cross: [f64; 3]) -> Option<[f64; 3]> {
        let rows: [[f64; 3]; 3] =
            std::array::from_fn(|r| [gram[3 * r], gram[3 * r + 1], gram[3 * r + 2]]);
        let inverse = crate::util::invert_fixed(&rows)?;
        Some(std::array::from_fn(|r| (0..3).map(|c| inverse[r][c] * cross[c]).sum()))
    }
}
