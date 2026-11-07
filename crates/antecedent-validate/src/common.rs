//! Shared refuter types and data transforms.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    AverageEffectQuery, ExecutionContext, KernelPolicy, TemporalEffectQuery, VariableId,
};
use antecedent_data::TemporalIndexer;
use antecedent_data::{
    ColumnView, DataError, DiscoveryEstimationSplit, Float64Column, OwnedColumn,
    OwnedColumnarStorage, PanelData, TableView, TabularData, TimeIndex, TimeSeriesData,
    ValidityBitmap,
};

use crate::panel_slice::PanelSliceTemplate;
use antecedent_estimate::{
    ConditionalLinearAdjustment, EffectEstimate, EstimationWorkspace, LinearAdjustmentAte,
    OverlapPolicy, TemporalLinearAdjustment,
};
use antecedent_identify::IdentifiedEstimand;

use crate::error::ValidationError;
use crate::target_native::RefutationTarget;

mod data;
mod propensity;
mod replicates;
mod report;
pub(crate) use propensity::{
    DiagnosticOverlap, binary_treatment, diagnostic_overlap_report_with, fit_diagnostic_propensity,
};
pub use report::{RefutationReport, RefutationScope};

pub(crate) use replicates::{
    STABILITY_ALPHA, STABILITY_REPLICATES, admit_cells, check_cancelled, check_replicate_count,
    finish_p_report, map_drawn_problem_replicates, map_drawn_replicates, map_problem_replicates,
    map_replicates,
};

#[cfg(test)]
pub(crate) use replicates::replicate_p_value;

pub(crate) use data::{
    column_major_dot, complete_case_rows, fill_gaussian, float64_full, full_sample_refit_ate,
    masked_sample_sd, sample_sd, temporal_diagnostic_design, validity_from_flags,
    with_contiguous_row_window, with_resampled_rows, with_row_subset,
};

/// A [`ValidationError::NotApplicable`] with a fixed message.
pub(crate) fn not_applicable(message: &'static str) -> ValidationError {
    ValidationError::NotApplicable { message }
}

/// Require one sensitivity input per coordinate of a vector target.
///
/// # Errors
///
/// `found` differs from the target's dimension.
pub(crate) fn require_dimension(
    target: &RefutationTarget,
    found: usize,
) -> Result<(), ValidationError> {
    if found == target.dimension() {
        Ok(())
    } else {
        Err(not_applicable(
            "a sensitivity range of a vector target needs one input per coordinate of the target",
        ))
    }
}

/// Context for lag-aware temporal refits (series or panel).
#[derive(Clone, Copy, Debug)]
pub struct TemporalRefitContext<'a> {
    /// Unfolded temporal indexer from identification.
    pub indexer: &'a TemporalIndexer,
    /// Temporal effect query (pulse / sustained).
    pub temporal_query: &'a TemporalEffectQuery,
    /// Optional discovery/estimation split.
    pub split: Option<&'a DiscoveryEstimationSplit>,
    /// Kernel policy for lag sample preparation.
    pub kernel_policy: &'a KernelPolicy,
    /// Series time index, or `None` when refitting a panel.
    pub time_index: Option<&'a TimeIndex>,
    /// Panel units when refitting stacked panel designs.
    pub panel: Option<&'a PanelData>,
    /// The series has missing time points. Refits go through the gap-aware design (intact
    /// lag windows at their original times), so a perturbation that keeps the gaps in place
    /// is refit without any window spanning one.
    pub gaps: bool,
}

impl TemporalRefitContext<'_> {
    /// True when this context targets a panel (stacked cluster design).
    #[must_use]
    pub fn is_panel(&self) -> bool {
        self.panel.is_some()
    }
}

/// Estimator-specific refit for perturbation checks whose causal model is not a
/// single adjustment regression. The same estimator must be used after mutation.
pub trait EffectRefit: std::fmt::Debug {
    /// Refit the original causal functional on perturbed data, optionally adding
    /// independent contemporaneous nuisance covariates to every mechanism.
    fn refit(
        &self,
        data: &TabularData,
        extra_contemporaneous: &[VariableId],
        ctx: &ExecutionContext,
    ) -> Result<EffectEstimate, ValidationError>;

    /// Whether a static row bootstrap may resample only the problem's witness
    /// treatment, outcome, adjustment and modifier columns. Implementations may
    /// opt in only when the refit reads no other row-varying input.
    fn static_bootstrap_uses_witness_columns(&self) -> bool {
        false
    }

    /// Prepare the composed estimator once on the original series so resampling
    /// checks can refit it on lag-aligned rows (each row keeping its original lag
    /// window) instead of rebuilding lags on a resampled raw series. `None` when
    /// this refitter has no aligned-row evaluation.
    fn prepare_aligned(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Option<Result<AlignedRefit, ValidationError>> {
        let _ = (data, ctx);
        None
    }

    /// [`Self::prepare_aligned`] on a series that may have missing time points, with the
    /// intact-window map of the original window axis (`None` for a complete series).
    ///
    /// The default has no gap-aware evaluation and reports every series as complete.
    fn prepare_aligned_windows(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Option<Result<(AlignedRefit, Option<WindowCompleteness>), ValidationError>> {
        self.prepare_aligned(data, ctx).map(|prepared| prepared.map(|refit| (refit, None)))
    }
}

/// Refit of a composed estimator on a row map of its lag-aligned rows
/// (`None` when the fit fails).
pub(crate) type AlignedRowEstimate = Box<dyn FnMut(&[usize]) -> Option<f64>>;

/// A composed temporal estimator prepared for evaluation on lag-aligned rows.
pub struct AlignedRefit {
    /// Lag-aligned rows of the prepared design.
    pub rows: usize,
    /// Circular-block length, in aligned rows, of the interval being checked (the
    /// same rule that interval's block bootstrap uses, e.g.
    /// [`antecedent_estimate::dependence_block_length`] over its estimating scores).
    pub block_length: usize,
    /// When [`Self::estimate`] is not the estimator whose point estimate is checked
    /// but a stand-in for it (e.g. the least-squares contrast for a posterior
    /// mean), what it stands in for and when that is valid; reported with the check.
    pub stand_in: Option<Arc<str>>,
    /// Refit on the aligned rows of a row map.
    pub estimate: AlignedRowEstimate,
}

/// Which lag windows of a series with missing time points are intact, in time order.
///
/// The prepared design's rows are the intact windows, and a block bootstrap resamples
/// blocks of the original window axis so that no block spans a gap.
pub type WindowCompleteness = Vec<bool>;

impl std::fmt::Debug for AlignedRefit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AlignedRefit")
            .field("rows", &self.rows)
            .field("block_length", &self.block_length)
            .field("stand_in", &self.stand_in)
            .finish_non_exhaustive()
    }
}

/// Inputs shared by effect refuters.
#[derive(Clone, Copy, Debug)]
pub struct RefutationProblem<'a> {
    /// Tabular data (series storage wrap, or stacked panel rows for mutation).
    pub data: &'a TabularData,
    /// Identified estimand (backdoor adjustment).
    pub estimand: &'a IdentifiedEstimand,
    /// Average-effect query (levels / population).
    pub query: &'a AverageEffectQuery,
    /// Original point estimate.
    pub original: &'a EffectEstimate,
    /// Estimator id used for the original fit (e.g. `linear.adjustment.ate`), when known.
    pub estimator: Option<&'a str>,
    /// When set, refits use [`TemporalLinearAdjustment`] on the lag-aligned design.
    pub temporal: Option<TemporalRefitContext<'a>>,
    /// Optional refit of the original composed causal model.
    pub effect_refit: Option<&'a dyn EffectRefit>,
    /// Prepare-time per-unit slice plan. Filled by [`crate::PreparedRefutation::compile`];
    /// callers constructing a problem directly leave this unset. A replicate loop
    /// compiles one plan from [`Self::temporal`] and reuses it; a single refit
    /// compiles its own.
    pub(crate) panel_slices: Option<&'a PanelSliceTemplate<'a>>,
    /// The treatment is a numeric dose whose effect is the slope times the contrast width
    /// (the lag-aligned diagnostic rows of a temporal design), so a refit never reads a
    /// discrete dose with several integer values as a two-level contrast.
    pub(crate) dose_treatment: bool,
}

impl<'a> RefutationProblem<'a> {
    /// Construct a refutation problem. Panel slice reuse is compiled by
    /// [`crate::PreparedRefutation::compile`].
    #[must_use]
    pub fn new(
        data: &'a TabularData,
        estimand: &'a IdentifiedEstimand,
        query: &'a AverageEffectQuery,
        original: &'a EffectEstimate,
        estimator: Option<&'a str>,
        temporal: Option<TemporalRefitContext<'a>>,
    ) -> Self {
        Self {
            data,
            estimand,
            query,
            original,
            estimator,
            temporal,
            effect_refit: None,
            panel_slices: None,
            dose_treatment: false,
        }
    }

    /// Read the treatment as a numeric dose in every refit of this problem.
    pub(crate) fn with_dose_treatment(mut self) -> Self {
        self.dose_treatment = true;
        self
    }

    /// Preserve the original composed estimator during every perturbation.
    #[must_use]
    pub fn with_effect_refit(mut self, refit: &'a dyn EffectRefit) -> Self {
        self.effect_refit = Some(refit);
        self
    }

    pub(crate) fn with_panel_slices(mut self, slices: Option<&'a PanelSliceTemplate<'a>>) -> Self {
        self.panel_slices = slices;
        self
    }
}

/// What a refit reads of a [`RefutationProblem`] apart from its composed-estimator refit.
///
/// The refit is a `dyn` without a `Sync` bound, so it cannot be shared with worker threads;
/// everything else can, and the refit travels beside this view as a separate argument.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SharedProblem<'a, 'plan> {
    pub(crate) data: &'a TabularData,
    pub(crate) estimand: &'a IdentifiedEstimand,
    pub(crate) query: &'a AverageEffectQuery,
    pub(crate) temporal: Option<TemporalRefitContext<'a>>,
    /// `'plan` may be shorter than `'a` when a replicate loop owns the plan.
    pub(crate) panel_slices: Option<&'plan PanelSliceTemplate<'a>>,
}

impl<'a> RefutationProblem<'a> {
    pub(crate) fn shared(&self) -> SharedProblem<'a, 'a> {
        SharedProblem {
            data: self.data,
            estimand: self.estimand,
            query: self.query,
            temporal: self.temporal,
            panel_slices: self.panel_slices,
        }
    }
}

impl RefutationProblem<'_> {
    /// Treatment variable from the query.
    #[must_use]
    pub fn treatment(&self) -> VariableId {
        self.query.treatment
    }

    /// Outcome variable from the query.
    #[must_use]
    pub fn outcome(&self) -> VariableId {
        self.query.outcome
    }

    /// Schema columns that define the estimate's complete-case sample.
    ///
    /// Treatment, outcome, and effect modifiers, plus the adjustment set when
    /// `include_adjustment` is set. A temporal problem's adjustment ids are dense
    /// node ids, not columns, so those callers pass `false`; modifiers are still
    /// included, because a missing modifier drops the row from the conditional fit.
    /// Duplicate ids (a modifier that is also an adjustment column) are omitted.
    pub(crate) fn estimation_ids(&self, include_adjustment: bool) -> Vec<VariableId> {
        let mut ids = Vec::with_capacity(
            2 + usize::from(include_adjustment) * self.estimand.adjustment_set.len()
                + self.query.effect_modifiers.len(),
        );
        ids.push(self.treatment());
        ids.push(self.outcome());
        if include_adjustment {
            for &id in self.estimand.adjustment_set.iter() {
                if !ids.contains(&id) {
                    ids.push(id);
                }
            }
        }
        for &id in self.query.effect_modifiers.iter() {
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
        ids
    }
}

/// Replace a refuter column while preserving its validity, mask, and weights.
/// Perturbations must not impute missing observations or change the fitted sample.
pub(crate) fn with_replaced_float(
    data: &TabularData,
    id: VariableId,
    values: Arc<[f64]>,
) -> Result<TabularData, ValidationError> {
    let storage = data.storage();
    let mut columns = storage.columns().to_vec();
    let column = columns.get_mut(id.as_usize()).ok_or(DataError::UnknownVariable { id })?;
    let OwnedColumn::Float64(original) = column else {
        return Err(DataError::TypeMismatch { id, expected: "float64" }.into());
    };
    *column = OwnedColumn::Float64(Float64Column::new(id, values, original.validity.clone())?);
    Ok(TabularData::new(OwnedColumnarStorage::try_new(
        storage.schema().clone(),
        columns,
        storage.analysis_mask().cloned(),
        storage.weights().map(Arc::from),
    )?))
}

/// Append an independent continuous covariate; returns new data and its id.
pub(crate) fn with_extra_float(
    data: &TabularData,
    name: &str,
    values: Arc<[f64]>,
) -> Result<(TabularData, VariableId), ValidationError> {
    data.with_appended_float(name, values).map_err(ValidationError::from)
}

/// Fit the supported scalar once (no nested bootstrap pools).
///
/// Queries with effect modifiers refit [`ConditionalLinearAdjustment`] — the same
/// interaction model that produced the supported CATE-at-mean — not main-effects ATE.
pub(crate) fn fit_once(
    estimator: &LinearAdjustmentAte,
    data: &TabularData,
    estimand: &IdentifiedEstimand,
    query: &AverageEffectQuery,
    workspace: &mut EstimationWorkspace,
    ctx: &ExecutionContext,
) -> Result<EffectEstimate, ValidationError> {
    if !query.effect_modifiers.is_empty() {
        // The conditional estimator is a different model. Settings that would
        // change its answer and that it cannot carry are a refusal, not a silent drop.
        if estimator.inference.se_kind != antecedent_estimate::AnalyticSeKind::Homoskedastic
            || estimator.inference.cluster_ids.is_some()
            || estimator.inference.source_cluster_ids.is_some()
            || estimator.sampling_weights.is_some()
            || estimator.inference.multiway_ids.is_some()
            || estimator.inference.panel_times.is_some()
            || !matches!(estimator.fit_kind, antecedent_estimate::LinearFitKind::Ols)
        {
            return Err(ValidationError::estimation_msg(
                "conditional refit cannot apply the caller's cluster, panel, SE, or non-OLS fit; those settings would be dropped",
            ));
        }
        return ConditionalLinearAdjustment::new()
            .with_overlap(estimator.overlap)
            .estimate_ate(data, estimand, query)
            .map(|(estimate, _, _)| estimate)
            .map_err(ValidationError::from);
    }
    let prep = estimator.prepare(data, estimand, query).map_err(ValidationError::from)?;
    estimator
        .fit(&prep, workspace, ctx, antecedent_core::AssumptionSet::new())
        .map_err(ValidationError::from)
}

/// Static or temporal effect refit (bootstrap disabled).
///
/// `caller_estimator` carries the refuter's configured [`LinearAdjustmentAte`] (SE kind,
/// cluster/multiway/panel-time ids, fit family, backend); the refit honors it on both the
/// static and temporal branches so a caller who set e.g. `se_kind = AnalyticSeKind::Cluster`
/// gets cluster-robust SEs on every refit replicate, not silently-downgraded homoskedastic
/// OLS. Two fields are always forced regardless of what the caller configured:
/// `bootstrap_replicates = 0` (refit replicates must never nest their own bootstrap pools)
/// and `overlap = OverlapPolicy::ExplicitOverride` (refuters mutate a single column and refit
/// the same design; re-running propensity/overlap diagnostics on a noised or permuted column
/// is neither meaningful nor how the original estimate was gated).
pub(crate) fn refit_effect(
    problem: &RefutationProblem<'_>,
    data: &TabularData,
    estimand: &IdentifiedEstimand,
    extra_contemporaneous: &[VariableId],
    caller_estimator: &LinearAdjustmentAte,
    workspace: &mut EstimationWorkspace,
    ctx: &ExecutionContext,
) -> Result<EffectEstimate, ValidationError> {
    refit_effect_shared(
        &problem.shared(),
        problem.effect_refit,
        data,
        estimand,
        extra_contemporaneous,
        caller_estimator,
        workspace,
        ctx,
    )
}

/// [`refit_effect`] on the shareable part of a problem plus its composed-estimator refit.
#[allow(
    clippy::too_many_arguments,
    reason = "the problem's refit travels beside its shared view, which keeps the view Sync"
)]
pub(crate) fn refit_effect_shared(
    problem: &SharedProblem<'_, '_>,
    effect_refit: Option<&dyn EffectRefit>,
    data: &TabularData,
    estimand: &IdentifiedEstimand,
    extra_contemporaneous: &[VariableId],
    caller_estimator: &LinearAdjustmentAte,
    workspace: &mut EstimationWorkspace,
    ctx: &ExecutionContext,
) -> Result<EffectEstimate, ValidationError> {
    if let Some(refit) = effect_refit {
        return refit.refit(data, extra_contemporaneous, ctx);
    }
    let Some(temporal) = problem.temporal else {
        let est = forced_refit_estimator(caller_estimator);
        return fit_once(&est, data, estimand, problem.query, workspace, ctx);
    };
    let estimator =
        TemporalLinearAdjustment::new().with_inner(forced_refit_estimator(caller_estimator));
    if let Some(panel) = temporal.panel {
        let owned;
        let slices = if let Some(plan) = problem.panel_slices {
            plan
        } else {
            owned = PanelSliceTemplate::from_panel(panel, problem.data)?;
            &owned
        };
        let rebuilt = slices.apply_stacked(data)?;
        let prep = if extra_contemporaneous.is_empty() {
            let (prep, _cluster_ids, _panel_times) = estimator
                .prepare_panel(
                    &rebuilt,
                    estimand,
                    temporal.temporal_query,
                    temporal.indexer,
                    temporal.split,
                    temporal.kernel_policy,
                )
                .map_err(ValidationError::from)?;
            prep
        } else {
            panel_prepare_with_extras(
                &estimator,
                &rebuilt,
                estimand,
                &temporal,
                extra_contemporaneous,
            )?
        };
        return estimator
            .fit(&prep, workspace, ctx, antecedent_core::AssumptionSet::new())
            .map_err(ValidationError::from);
    }
    let time_index = temporal.time_index.ok_or(ValidationError::NotApplicable {
        message: "temporal series refit requires time_index",
    })?;
    // Most refuters (noise-replace, resample) keep every row, so the original time index
    // (built for the un-subset series length) still applies verbatim. `DataSubsetRefuter`'s
    // contiguous-window subset (`with_contiguous_row_window`) physically shortens the series,
    // so the index length must follow the data it actually describes; the sampling interval
    // and regularity kind are unaffected by *which* contiguous window was kept.
    let series_time_index = if data.row_count() == time_index.length {
        time_index.clone()
    } else {
        TimeIndex { regularity: time_index.regularity.clone(), length: data.row_count() }
    };
    let series = TimeSeriesData::try_new(data.storage().clone(), series_time_index)
        .map_err(ValidationError::from)?;
    let inputs = antecedent_estimate::TemporalDesignInputs {
        data: &series,
        estimand,
        query: temporal.temporal_query,
        indexer: temporal.indexer,
        split: temporal.split,
        policy: temporal.kernel_policy,
    };
    let prep = if temporal.gaps {
        estimator
            .prepare_with_gaps_extras(&inputs, extra_contemporaneous)
            .map_err(ValidationError::from)?
            .0
    } else {
        estimator
            .prepare_with_extras(&inputs, extra_contemporaneous)
            .map_err(ValidationError::from)?
    };
    estimator
        .fit(&prep, workspace, ctx, antecedent_core::AssumptionSet::new())
        .map_err(ValidationError::from)
}

fn panel_prepare_with_extras(
    estimator: &TemporalLinearAdjustment,
    panel: &PanelData,
    estimand: &IdentifiedEstimand,
    temporal: &TemporalRefitContext<'_>,
    extra: &[VariableId],
) -> Result<antecedent_estimate::PreparedEstimationProblem, ValidationError> {
    // Stack per-unit prepare_with_extras designs (mirrors prepare_panel).
    let mut all_t = Vec::new();
    let mut all_y = Vec::new();
    let mut all_covs: Vec<(VariableId, Vec<f64>)> = Vec::new();
    let mut adj_keys: Vec<VariableId> = Vec::new();
    let mut active = 0.0;
    let mut control = 0.0;
    let mut treatment_delta = 0.0;
    let mut first = true;
    for unit in panel.units() {
        let prep = estimator
            .prepare_with_extras(
                &antecedent_estimate::TemporalDesignInputs {
                    data: &unit.series,
                    estimand,
                    query: temporal.temporal_query,
                    indexer: temporal.indexer,
                    split: temporal.split,
                    policy: temporal.kernel_policy,
                },
                extra,
            )
            .map_err(ValidationError::from)?;
        if first {
            active = prep.active;
            control = prep.control;
            treatment_delta = prep.treatment_delta;
            adj_keys = prep.adjustment_set.to_vec();
            all_covs = adj_keys.iter().map(|&id| (id, Vec::new())).collect();
            first = false;
        }
        all_t.extend_from_slice(&prep.treatment);
        all_y.extend_from_slice(&prep.design.outcome);
        let nrows = prep.design.nrows;
        for (i, (_id, dest)) in all_covs.iter_mut().enumerate() {
            let base = (2 + i) * nrows;
            dest.extend_from_slice(&prep.design.matrix[base..base + nrows]);
        }
    }
    let cov_refs: Vec<(VariableId, &[f64])> =
        all_covs.iter().map(|(id, v)| (*id, v.as_slice())).collect();
    let selected: Vec<usize> = (0..all_t.len()).collect();
    let design =
        antecedent_stats::CompiledDesign::linear_adjustment(&all_t, &cov_refs, &all_y, &selected)
            .map_err(ValidationError::from)?;
    let rows = all_t.len();
    Ok(antecedent_estimate::PreparedEstimationProblem {
        design,
        method: Arc::from("temporal.linear.adjustment.panel"),
        adjustment_set: Arc::from(adj_keys),
        overlap: OverlapPolicy::ExplicitOverride,
        treatment_delta,
        target_population: antecedent_core::TargetPopulation::AllObserved,
        treatment: Arc::from(all_t),
        active,
        control,
        treatment_levels: None,
        sampling_weights: None,
        preparation: antecedent_data::materialize::PreparationReceipt::composed_finite(rows)?,
    })
}

/// Stack panel units into one tabular table (row-major concat) for refute mutations.
pub fn stack_panel_tabular(panel: &PanelData) -> Result<TabularData, ValidationError> {
    let total = panel.total_rows();
    let schema = panel.schema().clone();
    let n_cols = schema.len();
    let mut col_vals: Vec<Vec<f64>> = (0..n_cols).map(|_| Vec::with_capacity(total)).collect();
    for u in panel.units() {
        for (j, dest) in col_vals.iter_mut().enumerate() {
            let id =
                VariableId::from_raw(u32::try_from(j).map_err(|_| {
                    ValidationError::data_msg("panel column index exceeds u32::MAX")
                })?);
            let vals = u.series.float64_values(id).map_err(ValidationError::from)?;
            dest.extend_from_slice(&vals);
        }
    }
    let mut cols = Vec::with_capacity(n_cols);
    for (j, values) in col_vals.into_iter().enumerate() {
        let id = VariableId::from_raw(
            u32::try_from(j)
                .map_err(|_| ValidationError::data_msg("panel column index exceeds u32::MAX"))?,
        );
        cols.push(OwnedColumn::Float64(
            Float64Column::new(id, Arc::from(values), ValidityBitmap::all_valid(total))
                .map_err(ValidationError::from)?,
        ));
    }
    let storage =
        OwnedColumnarStorage::try_new(schema, cols, None, None).map_err(ValidationError::from)?;
    Ok(TabularData::new(storage))
}

/// Linear adjustment with nested bootstrap disabled (refuters / sensitivity grids).
#[must_use]
pub(crate) fn linear_estimator_no_bootstrap() -> LinearAdjustmentAte {
    let mut estimator = LinearAdjustmentAte::new();
    estimator.inference.bootstrap_replicates = 0;
    estimator
}

/// Clone `caller_estimator` and force the two overrides every refit needs regardless of the
/// caller's config: no nested bootstrap, and no propensity/overlap re-diagnosis. See
/// [`refit_effect`] for why these two are pinned rather than threaded through.
#[must_use]
pub(crate) fn forced_refit_estimator(
    caller_estimator: &LinearAdjustmentAte,
) -> LinearAdjustmentAte {
    let mut estimator = caller_estimator.clone();
    estimator.inference.bootstrap_replicates = 0;
    estimator.overlap = OverlapPolicy::ExplicitOverride;
    estimator
}

/// Which column a noise-replace refuter overwrites.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NoiseReplaceTarget {
    /// Replace the treatment column.
    Treatment,
    /// Replace the outcome column.
    Outcome,
}

/// Settings of one noise-replace refuter run.
#[derive(Clone, Copy, Debug)]
pub(crate) struct NoiseReplaceSpec {
    pub replicates: u32,
    pub alpha: f64,
    pub target: NoiseReplaceTarget,
    /// Base of the per-replicate noise stream ids.
    pub stream_base: u64,
    pub refuter_id: &'static str,
    pub failure_label: &'static str,
}

/// Shared placebo / dummy-outcome loop: replace a column with Gaussian noise and refit.
///
/// Under OLS / linear adjustment the coefficient on independent noise is asymptotically
/// zero by construction, so this procedure has no power against a wrong causal claim.
/// Reports are therefore `informative: false` (diagnostic only, not a falsifier).
/// A non-finite replicate estimate is a failed computation, not evidence: it is a
/// [`ValidationError`] (rendered `Failed` by the suite), never a verdict.
pub(crate) fn noise_replace_refute(
    problem: &RefutationProblem<'_>,
    _workspace: &mut EstimationWorkspace,
    ctx: &ExecutionContext,
    estimator: &LinearAdjustmentAte,
    spec: &NoiseReplaceSpec,
) -> Result<RefutationReport, ValidationError> {
    let NoiseReplaceSpec { replicates, alpha, target, stream_base, refuter_id, failure_label } =
        *spec;
    check_replicate_count(replicates)?;
    let replace_id = match target {
        NoiseReplaceTarget::Treatment => problem.treatment(),
        NoiseReplaceTarget::Outcome => problem.outcome(),
    };
    let n = problem.data.row_count();
    let ates = map_problem_replicates(
        ctx,
        problem,
        replicates,
        EstimationWorkspace::default,
        |problem, refit, workspace, ctx, r| {
            let mut noise = vec![0.0; n];
            fill_gaussian(&mut noise, ctx, crate::streams::replicate_stream(stream_base, r));
            let data = with_replaced_float(problem.data, replace_id, Arc::from(noise))?;
            let est = refit_effect_shared(
                problem,
                refit,
                &data,
                problem.estimand,
                &[],
                estimator,
                workspace,
                ctx,
            )?;
            Ok(est.ate)
        },
    )?;
    // OLS recovery of ~0 under independent noise cannot falsify the claim.
    finish_p_report(refuter_id, problem.original.ate, &ates, 0.0, alpha, |mean_ate, p_value| {
        format!(
            "{failure_label}: null value 0 is atypical under the replicate distribution \
             (mean {mean_ate}, p={p_value} < alpha={alpha})"
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "a zero-spread replicate set yields exactly p = 1 or p = 0 by construction, so exact equality is intended"
    )]
    fn replicate_p_value_zero_sd_matches_hypothesis_is_one() {
        // All replicates identical (SD = 0) and equal to the hypothesized value.
        let p = replicate_p_value(&[1.5, 1.5, 1.5, 1.5], 1.5).unwrap();
        assert_eq!(p, 1.0);
    }

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "a zero-spread replicate set yields exactly p = 1 or p = 0 by construction, so exact equality is intended"
    )]
    fn replicate_p_value_zero_sd_away_from_hypothesis_is_zero() {
        let p = replicate_p_value(&[1.5, 1.5, 1.5, 1.5], 0.0).unwrap();
        assert_eq!(p, 0.0);
    }

    #[test]
    fn replicate_p_value_matches_normal_tail_closed_form() {
        // Samples {-1, 1}: mean 0, sample sd sqrt(2). Hypothesized 2 gives z = sqrt(2), and
        // the two-sided normal tail is erfc(z / sqrt(2)) = erfc(1) = 0.157299207050285.
        let p = replicate_p_value(&[-1.0, 1.0], 2.0).unwrap();
        assert!((p - 0.157_299_207_050_285_13).abs() < 1e-12, "p={p}");
    }

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "the fixed reference fixture preserves its exact values and bounded setup"
    )]
    fn replicate_p_value_is_invariant_to_positive_and_negative_unit_scaling() {
        let expected = replicate_p_value(&[-1.0, 1.0], 2.0).unwrap();
        for scale in [1e-14, -1e-14, 1e14, -1e14] {
            let actual = replicate_p_value(&[-scale, scale], 2.0 * scale).unwrap();
            assert!((actual - expected).abs() < 1e-12, "scale {scale}: {actual} vs {expected}");
        }
        assert_eq!(replicate_p_value(&[0.0, 0.0], 0.0).unwrap(), 1.0);
        assert_eq!(replicate_p_value(&[0.0, 0.0], 1e-14).unwrap(), 0.0);
    }

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "the fixed reference fixture preserves its exact values and bounded setup"
    )]
    fn roundoff_degeneracy_is_relative_and_does_not_hide_real_tiny_effects() {
        let base = 2.0_f64.to_bits();
        let samples = [
            f64::from_bits(base - 64),
            f64::from_bits(base - 65),
            f64::from_bits(base - 63),
            f64::from_bits(base - 64),
        ];
        assert_eq!(replicate_p_value(&samples, 2.0).unwrap(), 1.0);
        assert!(replicate_p_value(&samples, f64::from_bits(base - 1024)).unwrap() < 0.05);
        for units in [1e-200, -1e-200, 1e100, -1e100] {
            let converted = samples.map(|x| x * units);
            assert_eq!(replicate_p_value(&converted, 2.0 * units).unwrap(), 1.0);
            assert!(replicate_p_value(&[-units, units], 10.0 * units).unwrap() < 0.05);
        }
        assert_eq!(replicate_p_value(&[0.0, 0.0], 1e-200).unwrap(), 0.0);
    }

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "the fixed reference fixture preserves its exact values and bounded setup"
    )]
    fn replicate_p_value_handles_extreme_units_and_large_offsets() {
        let reference = replicate_p_value(&[-1.0, 1.0], 2.0).unwrap();
        for scale in [1e-200, -1e-200, 1e200, -1e200] {
            let actual = replicate_p_value(&[-scale, scale], 2.0 * scale).unwrap();
            assert!((actual - reference).abs() < 1e-12, "scale {scale}: {actual}");
        }
        let actual = replicate_p_value(&[1e12 - 1.0, 1e12 + 1.0], 1e12 + 2.0).unwrap();
        assert!((actual - reference).abs() < 1e-12);
        assert_eq!(replicate_p_value(&[-f64::MAX, f64::MAX], 0.0).unwrap(), 1.0);
        assert_eq!(replicate_p_value(&[0.0, f64::from_bits(1)], f64::MAX).unwrap(), 0.0);
    }

    #[test]
    fn replicate_p_value_refuses_a_non_finite_replicate() {
        // A blown-up refit is neither a pass (p = 1) nor a refutation (p = 0): it is an
        // operational failure the caller must see as an error.
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let err = replicate_p_value(&[1.0, 2.0, bad, 3.0], 0.0).unwrap_err();
            assert!(matches!(err, ValidationError::Refused { .. }), "{err:?}");
        }
    }

    #[test]
    fn replicate_p_value_refuses_a_non_finite_hypothesis_or_too_few_samples() {
        let samples = [1.0, 2.0, 3.0, 4.0];
        assert!(replicate_p_value(&samples, f64::NAN).is_err());
        assert!(replicate_p_value(&samples, f64::INFINITY).is_err());
        assert!(matches!(
            replicate_p_value(&[1.0], 0.0),
            Err(ValidationError::NotApplicable { .. })
        ));
    }

    #[test]
    fn replicate_count_has_only_a_lower_bound_and_streams_stay_disjoint() {
        assert!(check_replicate_count(1).is_err());
        assert!(check_replicate_count(2).is_ok());
        assert!(check_replicate_count(1 << 20).is_ok());
        let base = crate::streams::PLACEBO_REFIT;
        // Historical streams are unchanged below the aliasing distance.
        assert_eq!(crate::streams::replicate_stream(base, 7), base + 7);
        // Beyond it a replicate never reuses the neighbouring refuter's base-plus-offset.
        let neighbour = crate::streams::PLACEBO_SHUFFLE;
        assert_ne!(crate::streams::replicate_stream(base, 0x1000), neighbour);
        assert_ne!(
            crate::streams::replicate_stream(base, 0x1001),
            crate::streams::replicate_stream(base, 0x1000)
        );
    }
}
