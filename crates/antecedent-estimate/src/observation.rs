//! Estimator-side handling of explicit outcome-observation mechanisms.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::float_cmp, clippy::too_many_lines)]

use std::sync::Arc;

use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSet, AssumptionSource,
    AssumptionStatus, CausalResponse, Diagnostic, DiagnosticKind, DiagnosticSeverity,
    IdentificationStatus, ObservationAssumption, ObservationSpec, ParametricAssumption,
    ResponseFunctional, ResponseQuery, ResponseUncertainty, SupportDiagnostic, TemporalNodeKey,
    VariableId,
};
use antecedent_data::{TableView, TabularData, TimeSeriesData};
use antecedent_stats::{
    FaerBackend, GaussianObservation, GlmDesignRef, GlmFamily, GlmOptions, LeastSquaresWorkspace,
    fit_glm, fit_observation_logistic, gaussian_observation_log_likelihood, kaplan_meier_ipcw,
    selected_outcome_pseudo_values,
};

use crate::{ContinuousResponseEstimator, EstimationError};

/// Selected-outcome correction used after fitting the observation model.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum SelectedOutcomeCorrection {
    /// Inverse-probability weighting, `R Y / p(X)`.
    Ipw,
    /// Augmented inverse-probability weighting with a Gaussian linear outcome nuisance.
    Aipw,
}

/// Numerical choices for observation-mechanism estimation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObservationEstimatorOptions {
    /// Selected-outcome correction.
    pub selected_correction: SelectedOutcomeCorrection,
    /// Floor for fitted observation probabilities.
    pub observation_probability_floor: f64,
    /// Floor for estimated censoring survival.
    pub censoring_survival_floor: f64,
    /// Deterministic row-index folds for augmented-path nuisance cross-fitting.
    ///
    /// Used by [`SelectedOutcomeCorrection::Aipw`] only; see [`ObservationMechanismEstimator`]
    /// for why the inverse-probability path keeps its in-sample fit.
    pub crossfit_folds: usize,
}

impl Default for ObservationEstimatorOptions {
    fn default() -> Self {
        Self {
            selected_correction: SelectedOutcomeCorrection::Aipw,
            observation_probability_floor: 0.01,
            censoring_survival_floor: 0.01,
            crossfit_folds: 5,
        }
    }
}

/// Out-of-fold selected-outcome nuisances, one prediction per source row.
struct CrossFittedSelectedNuisances {
    probabilities: Vec<f64>,
    outcome_predictions: Vec<f64>,
}

/// How selected-AIPW cross-fitting folds are assigned to observation rows.
#[derive(Clone, Copy)]
enum FoldLayout<'a> {
    /// Row `i` of an exchangeable table goes to fold `i % folds`.
    Interleaved,
    /// Row `i` sits at series position `positions[i]` in `0..span` and goes to the
    /// contiguous time block `positions[i]·folds / span`. Repeated positions (a block
    /// bootstrap drawing the same tuple twice) share a fold, and a validation row's
    /// serial neighbours mostly share its fold rather than training its nuisances.
    TimeBlocks { positions: &'a [usize], span: usize },
}

impl FoldLayout<'_> {
    fn fold_of(self, row: usize, folds: usize) -> usize {
        match self {
            Self::Interleaved => row % folds,
            Self::TimeBlocks { positions, span } => (positions[row] * folds / span).min(folds - 1),
        }
    }
}

/// Extra selected-AIPW outcome-model regressors: the downstream design columns the
/// declared observation conditioning set does not already carry.
struct OutcomeModelExtras {
    /// Column-major `rows × ncols`, aligned with the observation table rows.
    values: Vec<f64>,
    ncols: usize,
    /// Leading table rows whose extra regressors fall before the series start. Those rows
    /// keep the conditioning-only outcome model.
    lead: usize,
}

/// Regressors of every horizon's lag-aligned response design, as offsets relative to
/// the outcome time: the treatment at the policy offset and each identified adjustment
/// node, for a curve or single Set / Shift / Soft temporal response.
///
/// These are the columns the selected-AIPW outcome nuisance must condition on for the
/// pseudo-outcome regression to stay orthogonal to the estimated selection probability
/// (see [`ObservationMechanismEstimator::adjust_temporal_series`]).
///
/// # Errors
///
/// A query without a temporal attachment or treatment/outcome pair, a horizon count that
/// differs from `identifications`, or an adjustment node missing from its indexer.
pub fn temporal_curve_outcome_regressors(
    query: &ResponseQuery,
    identifications: &[(&antecedent_expr::IdentifiedEstimand, &antecedent_data::TemporalIndexer)],
) -> Result<Vec<TemporalNodeKey>, EstimationError> {
    let temporal = query.temporal.as_ref().ok_or_else(|| {
        EstimationError::unsupported(
            "temporal observation correction requires ResponseQuery.temporal",
        )
    })?;
    if temporal.horizons.len() != identifications.len() {
        return Err(EstimationError::unsupported(
            "temporal observation regressors need one identification per horizon",
        ));
    }
    let (treatment, _) = query.functional.primary_pair().ok_or_else(|| {
        EstimationError::unsupported("response query has no treatment/outcome pair")
    })?;
    let treatment_offset = temporal.treatment_offset()?;
    let mut keys = Vec::new();
    for (&horizon, &(estimand, indexer)) in temporal.horizons.iter().zip(identifications) {
        let outcome_offset = i32::try_from(horizon.saturating_sub(1)).unwrap_or(i32::MAX);
        keys.push(TemporalNodeKey {
            variable: treatment,
            offset: treatment_offset.saturating_sub(outcome_offset),
        });
        for &dense in estimand.adjustment_set.iter() {
            let key = indexer
                .key_of(dense.raw())
                .map_err(|e| EstimationError::data_msg(e.to_string()))?;
            keys.push(TemporalNodeKey {
                variable: key.variable,
                offset: key.offset.saturating_sub(outcome_offset),
            });
        }
    }
    keys.sort();
    keys.dedup();
    Ok(keys)
}

/// Parents of the outcome's stationary mechanism in `graph`, as offsets relative to the
/// outcome time: the regressors of the unfolded sequential outcome regression that a
/// Sequence overlay refits.
///
/// The outcome itself is also listed at `−lag` for every lagged edge out of the outcome
/// into a mechanism that can reach the outcome (its own or an ancestor's), i.e. every lag
/// at which the outcome can enter the unfolded Sequence design as a regressor.
/// [`ObservationMechanismEstimator::adjust_temporal_series`] refuses a Sequence whose
/// regressors carry the outcome at a nonzero offset: the correction replaces the outcome
/// column with pseudo-outcomes, which would then be regressors (errors in variables).
/// The selected-AIPW outcome nuisance never conditions on outcome columns.
#[must_use]
pub fn temporal_sequence_outcome_regressors(
    graph: &antecedent_graph::TemporalDag,
    outcome: VariableId,
) -> Vec<TemporalNodeKey> {
    let mut keys = Vec::new();
    // Template variables with a directed path to the outcome (the outcome included).
    let mut reaches_outcome = vec![outcome];
    loop {
        let before = reaches_outcome.len();
        for edge in graph.edges() {
            let Some((from, to)) = edge.parent_child() else {
                continue;
            };
            let (Some(from_key), Some(to_key)) = (graph.temporal_key(from), graph.temporal_key(to))
            else {
                continue;
            };
            if reaches_outcome.contains(&to_key.variable)
                && !reaches_outcome.contains(&from_key.variable)
            {
                reaches_outcome.push(from_key.variable);
            }
        }
        if reaches_outcome.len() == before {
            break;
        }
    }
    for edge in graph.edges() {
        let Some((from, to)) = edge.parent_child() else {
            continue;
        };
        let (Some(from_key), Some(to_key)) = (graph.temporal_key(from), graph.temporal_key(to))
        else {
            continue;
        };
        if from_key.variable == outcome
            && from_key.offset < to_key.offset
            && reaches_outcome.contains(&to_key.variable)
        {
            keys.push(TemporalNodeKey {
                variable: outcome,
                offset: from_key.offset.saturating_sub(to_key.offset),
            });
        }
    }
    for (index, _) in graph.nodes().iter().enumerate() {
        let id = antecedent_graph::DenseNodeId::from_raw(u32::try_from(index).unwrap_or(u32::MAX));
        let Some(key) = graph.temporal_key(id) else {
            continue;
        };
        if key.variable != outcome {
            continue;
        }
        for &parent in graph.parents(id) {
            if let Some(parent_key) = graph.temporal_key(parent) {
                keys.push(TemporalNodeKey {
                    variable: parent_key.variable,
                    offset: parent_key.offset.saturating_sub(key.offset),
                });
            }
        }
    }
    keys.sort();
    keys.dedup();
    keys
}

/// Pseudo-outcome representation produced by a supported observation mechanism.
#[derive(Clone, Debug, PartialEq)]
pub struct ObservationAdjustedOutcome {
    /// One pseudo-value per source row.
    ///
    /// Values already incorporate inverse-probability weighting and must not be weighted a
    /// second time by [`Self::weights`].
    pub values: Vec<f64>,
    /// Observation/IPCW weights exposed for diagnostics only.
    /// Unobserved/censored rows have zero weight.
    pub weights: Vec<f64>,
    /// Stable method identifier.
    pub method: Arc<str>,
}

/// Explicit observation-mechanism estimator.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ObservationMechanismEstimator {
    /// Numerical and method options.
    pub options: ObservationEstimatorOptions,
}

impl ObservationMechanismEstimator {
    /// Construct with explicit options.
    #[must_use]
    pub const fn new(options: ObservationEstimatorOptions) -> Self {
        Self { options }
    }

    /// Produce IPW/AIPW or IPCW pseudo-outcomes for a response query.
    ///
    /// Selected outcomes require exactly one explicit
    /// [`ObservationAssumption::OutcomeIndependentGiven`] claim and fit a logistic
    /// observation model on those variables. Right/left censoring uses Kaplan–Meier for an
    /// empty independence set and a log-linear Cox censoring hazard for nonempty sets. `delayed_entry` is supported for right censoring;
    /// left censoring with delayed entry and interval censoring/truncation are refused here.
    ///
    /// Under [`SelectedOutcomeCorrection::Aipw`] both nuisances are cross-fit over
    /// `crossfit_folds` deterministic row-index folds (row `i` in fold `i % folds`), so no
    /// row's pseudo-value is built from a model that saw it — the sample-splitting
    /// condition the AIPW double-robustness argument assumes. The temporal entry points
    /// ([`Self::adjust_temporal_series`], [`Self::adjust_temporal_anchors`]) use
    /// contiguous time-block folds keyed by series position instead. A fold that cannot support either model is refused, never silently
    /// refit in sample. [`SelectedOutcomeCorrection::Ipw`] deliberately keeps its in-sample
    /// maximum-likelihood propensity, which is the published estimator.
    ///
    /// Cross-fitting removes the in-sample nuisance bias; it does not by itself license an
    /// interval, and this path still publishes no standard error.
    ///
    /// # Errors
    ///
    /// Unsupported mechanism/assumption combinations, malformed columns, positivity failure,
    /// or nuisance-model failure.
    pub fn adjusted_outcome(
        &self,
        data: &TabularData,
        query: &ResponseQuery,
        delayed_entry: Option<VariableId>,
    ) -> Result<ObservationAdjustedOutcome, EstimationError> {
        self.adjusted_outcome_with_extras(data, query, delayed_entry, None, FoldLayout::Interleaved)
    }

    fn adjusted_outcome_with_extras(
        &self,
        data: &TabularData,
        query: &ResponseQuery,
        delayed_entry: Option<VariableId>,
        extras: Option<&OutcomeModelExtras>,
        layout: FoldLayout<'_>,
    ) -> Result<ObservationAdjustedOutcome, EstimationError> {
        query.validate()?;
        self.validate_options()?;
        match &query.observation {
            ObservationSpec::Selected { observed, indicator, .. } => {
                if delayed_entry.is_some() {
                    return Err(EstimationError::unsupported(
                        "delayed entry applies only to right-censoring IPCW",
                    ));
                }
                self.selected(data, query, *observed, *indicator, extras, layout)
            }
            ObservationSpec::RightCensored { observed, censoring, event, .. } => {
                self.censored(data, query, *observed, *censoring, *event, delayed_entry, false)
            }
            ObservationSpec::LeftCensored { observed, censoring, event, .. } => {
                if delayed_entry.is_some() {
                    return Err(EstimationError::unsupported(
                        "delayed entry is not defined for left-censoring sign reversal",
                    ));
                }
                self.censored(data, query, *observed, *censoring, *event, None, true)
            }
            ObservationSpec::Complete => Err(EstimationError::unsupported(
                "complete outcomes do not require observation-mechanism correction",
            )),
            ObservationSpec::IntervalCensored { .. } | ObservationSpec::Truncated { .. } => {
                Err(EstimationError::unsupported(
                    "interval censoring and truncation require the opt-in Gaussian likelihood",
                ))
            }
        }
    }

    /// Estimate an observation-adjusted scalar mean-response curve.
    ///
    /// This composition first constructs a selected-outcome or censoring-adjusted
    /// pseudo-outcome, then applies the continuous-treatment response estimator.
    ///
    /// Censoring IPCW and selected IPW use the Horvitz–Thompson transform
    /// `Y* = Y · W`. Censored or IPW-unselected rows remain with `Y* = 0`; that
    /// is the HT contribution, not a complete-case drop. Default selected
    /// correction is AIPW: unselected rows receive the outcome-regression
    /// prediction `m(X)`, not zero.
    ///
    /// For selected outcomes, the declared
    /// [`ObservationAssumption::OutcomeIndependentGiven`] set must include the treatment and
    /// every causal adjustment variable. This containment makes the observation correction
    /// conditionally valid for the downstream response regression. Marginal Kaplan–Meier IPCW
    /// continues to require the stronger unconditional observation-independence claim.
    ///
    /// The returned point curve does **not** carry the complete-data response bands: those bands
    /// omit uncertainty from the estimated observation mechanism and would overstate precision.
    /// Joint nuisance-and-curve uncertainty is therefore reported as [`ResponseUncertainty::None`]
    /// with an explicit diagnostic warning.
    ///
    /// # Errors
    ///
    /// Non-curve response functionals, unsupported observation mechanisms or assumptions,
    /// selected-outcome conditioning that omits treatment/adjustment variables, and any
    /// observation or response-estimation failure are refused.
    pub fn estimate_mean_curve(
        &self,
        response_estimator: &ContinuousResponseEstimator,
        data: &TabularData,
        query: &ResponseQuery,
        delayed_entry: Option<VariableId>,
        identification_status: IdentificationStatus,
        assumptions: AssumptionSet,
    ) -> Result<CausalResponse, EstimationError> {
        let (outcome, treatment) = match &query.functional {
            ResponseFunctional::MeanCurve { outcome, treatment } => (*outcome, treatment.variable),
            _ => {
                return Err(EstimationError::unsupported(
                    "observation-adjusted response composition currently supports MeanCurve only",
                ));
            }
        };
        if let ObservationSpec::Selected { .. } = query.observation {
            let conditioning = exact_outcome_independence(query)?;
            if !conditioning.contains(&treatment)
                || response_estimator
                    .adjustment_set
                    .iter()
                    .any(|variable| !conditioning.contains(variable))
            {
                return Err(EstimationError::unsupported(
                    "selected-outcome response correction requires OutcomeIndependentGiven to include the treatment and every causal adjustment variable",
                ));
            }
        }
        let conditional_censoring = matches!(
            query.observation,
            ObservationSpec::RightCensored { .. } | ObservationSpec::LeftCensored { .. }
        ) && !censoring_independence(query)?.is_empty();
        let mut assumptions = assumptions;
        if conditional_censoring {
            let conditioning = censoring_independence(query)?;
            if !conditioning.contains(&treatment)
                || response_estimator
                    .adjustment_set
                    .iter()
                    .any(|variable| !conditioning.contains(variable))
            {
                return Err(EstimationError::unsupported(
                    "conditional censoring response requires IndependentGiven to include treatment and every causal adjustment variable",
                ));
            }
            assumptions.push(AssumptionRecord {
                assumption: Assumption::ParametricRestriction(ParametricAssumption {
                    id: Arc::from("observation.cox_proportional_hazards"),
                    description: Arc::from("Conditional censoring uses a proportional-hazards model with log-linear effects of the declared covariates and Breslow ties; correct conditional censoring survival and positivity are required."),
                }),
                source: AssumptionSource::AlgorithmDefault { algorithm: Arc::from("estimate.observation_cox_ipcw") },
                scope: AssumptionScope::Estimation, status: AssumptionStatus::Declared,
            });
        }
        if response_estimator.options.simultaneous_replicates.is_some() {
            return Err(EstimationError::unsupported(
                "simultaneous bands are unavailable for observation-adjusted response curves",
            ));
        }
        let adjusted = self.adjusted_outcome(data, query, delayed_entry)?;
        let adjusted_data = data
            .with_replaced_float(outcome, Arc::from(adjusted.values))
            .map_err(EstimationError::from)?;
        let mut complete_query = query.clone();
        complete_query.observation = ObservationSpec::Complete;
        complete_query.observation_assumptions = Arc::from([]);
        let mut response = response_estimator.estimate_identified(
            &adjusted_data,
            &complete_query,
            identification_status,
            assumptions,
        )?;
        let (minimum_weight, maximum_weight, effective_sample_size) =
            diagnostic_weight_summary(&adjusted.weights);
        response.uncertainty = ResponseUncertainty::None;
        response.provenance_id = Arc::from("estimate.response.observation_adjusted");
        response.support.diagnostics.push(SupportDiagnostic {
            id: Arc::from("response.observation_adjustment_weights"),
            values: Arc::from([minimum_weight, maximum_weight, effective_sample_size]),
            detail: Arc::from(
                "minimum positive weight, maximum weight, and Kish effective sample size; weights are diagnostic only and were already incorporated into the pseudo-outcome",
            ),
        });
        response.support.warnings.push(Diagnostic::new(
            "response.observation_joint_uncertainty_unavailable",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            "point estimate includes observation correction; uncertainty is omitted because complete-data curve bands do not account for the estimated observation mechanism",
        ));
        response.support.warnings.push(Diagnostic::new(
            "response.observation_adjustment_method",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            adjusted.method.clone(),
        ));
        if adjusted.method.as_ref() == "observation.selected.complete_collapse.v1" {
            response.support.warnings.push(Diagnostic::new(
                "response.observation_selected_complete_collapse",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Warning,
                "every selection indicator is 1, so the selected-outcome correction is the raw recorded outcome; a mis-coded indicator would look like no selection",
            ));
        }
        Ok(response)
    }

    /// Observation correction on a temporal curve: same 1.3 primitives, with
    /// nonempty `IndependentGiven` / `OutcomeIndependentGiven` lag-aligned at the
    /// policy treatment offset.
    ///
    /// `adjustment` is the union of identified horizon adjustment variables.
    /// Selected and conditional-censoring pairs must name the treatment and every
    /// such variable.
    ///
    /// `outcome_regressors` are the columns (offsets relative to the outcome time) of the
    /// downstream response regressions that will consume the pseudo-outcome
    /// ([`temporal_curve_outcome_regressors`] / [`temporal_sequence_outcome_regressors`]).
    /// The selected-AIPW outcome nuisance conditions on the declared set at the policy
    /// offset **and** on those columns. With the declared set alone, a downstream design
    /// that also regresses on, e.g., the treatment at a second lag leaves the pseudo-outcome
    /// residual `Y − m` correlated with that regressor, so the downstream coefficients are
    /// protected only by the selection model and are first-order sensitive to its
    /// estimation error (not orthogonal in `π̂`). With the downstream columns in `m`,
    /// `E[Y* | design] = E[Y | design, R = 1]` whatever `π̂`, which equals the latent
    /// regression when selection is ignorable given the design columns. The selection
    /// model keeps the declared set. Censoring IPCW and plain selected IPW ignore
    /// `outcome_regressors`; so do columns of the observed outcome or indicator (they are
    /// not completely observed) and columns already in the declared set at the policy
    /// offset. Rows whose extra regressors would fall before the series start keep the
    /// declared-set outcome model.
    ///
    /// # Errors
    ///
    /// Unlicensed pair, missing containment, delayed entry, or primitive failure.
    pub fn adjust_temporal_series(
        &self,
        data: &TimeSeriesData,
        query: &ResponseQuery,
        adjustment: &[VariableId],
        outcome_regressors: &[TemporalNodeKey],
    ) -> Result<(TimeSeriesData, ObservationAdjustedOutcome), EstimationError> {
        query.validate()?;
        query.require_licensed_temporal_observation()?;
        if query.observation == ObservationSpec::Complete {
            return Err(EstimationError::unsupported(
                "complete outcomes do not require observation-mechanism correction",
            ));
        }
        let temporal = query.temporal.as_ref().ok_or_else(|| {
            EstimationError::unsupported(
                "temporal observation correction requires ResponseQuery.temporal",
            )
        })?;
        let offset = temporal.treatment_offset()?;
        if offset > 0 {
            return Err(EstimationError::unsupported(
                "temporal observation conditioning cannot use a future treatment offset",
            ));
        }
        let (treatment, outcome) = query.functional.primary_pair().ok_or_else(|| {
            EstimationError::unsupported("response query has no treatment/outcome pair")
        })?;
        require_temporal_observation_containment(query, treatment, adjustment)?;
        require_unlagged_sequence_outcome(query, outcome, outcome_regressors)?;
        let conditioning = temporal_conditioning_ids(query)?;
        let (table, start) = if conditioning.is_empty() {
            (TabularData::new(data.storage().clone()), 0)
        } else {
            lag_aligned_observation_table(data, conditioning, offset)?
        };
        let extras = self.outcome_model_extras(
            data,
            query,
            offset,
            outcome_regressors,
            &(start..data.row_count()).collect::<Vec<_>>(),
        )?;
        // Contiguous time-block folds over the lag-aligned rows `start..n`.
        let positions: Vec<usize> = (0..table.row_count()).collect();
        let layout = FoldLayout::TimeBlocks { positions: &positions, span: positions.len().max(1) };
        let subset =
            self.adjusted_outcome_with_extras(&table, query, None, extras.as_ref(), layout)?;
        let mut values = data.float64_values(outcome)?;
        let mut weights = vec![0.0; values.len()];
        if start == 0 {
            values = subset.values;
            weights = subset.weights;
        } else if subset.values.len() + start != values.len() {
            return Err(EstimationError::unsupported(
                "temporal observation subset does not cover the lag-aligned series",
            ));
        } else {
            values[start..].copy_from_slice(&subset.values);
            weights[start..].copy_from_slice(&subset.weights);
        }
        let adjusted =
            ObservationAdjustedOutcome { values: values.clone(), weights, method: subset.method };
        let series = data.with_replaced_float(outcome, Arc::from(values))?;
        Ok((series, adjusted))
    }

    /// Tuple-level replicate of [`Self::adjust_temporal_series`] for block bootstraps.
    ///
    /// `anchors` are outcome-time row indices of `data` (repeats allowed, in resample
    /// order). Each anchor contributes its whole lag-aligned observation row — the
    /// outcome and indicators at the anchor, the conditioning set at the policy treatment
    /// offset — so no replicate row pairs values from different resampled blocks. The
    /// observation nuisance is refit on exactly those rows; the returned pseudo-outcomes
    /// align with `anchors`. Selected-AIPW folds are the same contiguous time blocks as
    /// [`Self::adjust_temporal_series`], keyed by each anchor's source position, so a
    /// tuple drawn more than once lands in a single fold and never trains the nuisance
    /// that predicts its own copy. `outcome_regressors` are as in
    /// [`Self::adjust_temporal_series`]; every anchor must reach all of them
    /// ([`Self::observation_row_lag`]).
    ///
    /// # Errors
    ///
    /// The refusals of [`Self::adjust_temporal_series`], or an anchor whose lag-aligned
    /// row does not exist.
    pub fn adjust_temporal_anchors(
        &self,
        data: &TimeSeriesData,
        query: &ResponseQuery,
        adjustment: &[VariableId],
        outcome_regressors: &[TemporalNodeKey],
        anchors: &[usize],
    ) -> Result<Vec<f64>, EstimationError> {
        query.validate()?;
        query.require_licensed_temporal_observation()?;
        if query.observation == ObservationSpec::Complete {
            return Err(EstimationError::unsupported(
                "complete outcomes do not require observation-mechanism correction",
            ));
        }
        let temporal = query.temporal.as_ref().ok_or_else(|| {
            EstimationError::unsupported(
                "temporal observation correction requires ResponseQuery.temporal",
            )
        })?;
        let offset = temporal.treatment_offset()?;
        if offset > 0 {
            return Err(EstimationError::unsupported(
                "temporal observation conditioning cannot use a future treatment offset",
            ));
        }
        let (treatment, _) = query.functional.primary_pair().ok_or_else(|| {
            EstimationError::unsupported("response query has no treatment/outcome pair")
        })?;
        require_temporal_observation_containment(query, treatment, adjustment)?;
        let conditioning = temporal_conditioning_ids(query)?;
        let lag = if conditioning.is_empty() { 0 } else { offset.unsigned_abs() as usize };
        let n = data.row_count();
        if anchors.iter().any(|&anchor| anchor < lag || anchor >= n) {
            return Err(EstimationError::unsupported(
                "temporal observation anchor has no lag-aligned observation row",
            ));
        }
        let extras = self.outcome_model_extras(data, query, offset, outcome_regressors, anchors)?;
        if extras.as_ref().is_some_and(|extras| extras.lead > 0) {
            return Err(EstimationError::unsupported(
                "temporal observation anchor has no lag-aligned outcome-model row",
            ));
        }
        let covariates = gather_anchor_columns(data, conditioning, anchors, lag)?;
        match &query.observation {
            ObservationSpec::Selected { observed, indicator, .. } => {
                if conditioning.iter().any(|id| id == observed || id == indicator) {
                    return Err(EstimationError::unsupported(
                        "observation-model conditions cannot include observed outcome or indicator",
                    ));
                }
                let y = gather_anchor_column(data, *observed, anchors, 0)?;
                let r = gather_anchor_column(data, *indicator, anchors, 0)?;
                let positions: Vec<usize> = anchors.iter().map(|&anchor| anchor - lag).collect();
                let layout = FoldLayout::TimeBlocks { positions: &positions, span: n - lag };
                Ok(self
                    .selected_from_columns(
                        &y,
                        &r,
                        &covariates,
                        conditioning.len(),
                        extras.as_ref(),
                        layout,
                    )?
                    .values)
            }
            ObservationSpec::RightCensored { observed, censoring, event, .. }
            | ObservationSpec::LeftCensored { observed, censoring, event, .. } => {
                if conditioning.iter().any(|v| [observed, censoring, event].contains(&v)) {
                    return Err(EstimationError::unsupported(
                        "censoring conditioning cannot include recorded outcome, censoring time, or event indicator",
                    ));
                }
                let y = gather_anchor_column(data, *observed, anchors, 0)?;
                let c = gather_anchor_column(data, *censoring, anchors, 0)?;
                let d = gather_anchor_column(data, *event, anchors, 0)?;
                let reverse = matches!(query.observation, ObservationSpec::LeftCensored { .. });
                Ok(self
                    .censored_from_columns(
                        &y,
                        &c,
                        &d,
                        &covariates,
                        conditioning.len(),
                        None,
                        reverse,
                    )?
                    .values)
            }
            _ => Err(EstimationError::unsupported(
                "temporal observation correction supports selected and left/right censoring only",
            )),
        }
    }

    /// Largest lag (in rows before the outcome time) an observation row reads: the
    /// declared conditioning set at the policy offset and, for selected AIPW, every
    /// retained outcome-model regressor. Tuple bootstraps start their anchors at least
    /// this far into the series.
    ///
    /// # Errors
    ///
    /// A query without a supported observation assumption or temporal attachment.
    pub fn observation_row_lag(
        &self,
        query: &ResponseQuery,
        outcome_regressors: &[TemporalNodeKey],
    ) -> Result<usize, EstimationError> {
        if query.observation == ObservationSpec::Complete {
            return Ok(0);
        }
        let offset = query
            .temporal
            .as_ref()
            .ok_or_else(|| {
                EstimationError::unsupported(
                    "temporal observation correction requires ResponseQuery.temporal",
                )
            })?
            .treatment_offset()?;
        let conditioning = temporal_conditioning_ids(query)?;
        let base = if conditioning.is_empty() { 0 } else { offset.unsigned_abs() as usize };
        let extra = self
            .retained_outcome_regressors(query, offset, outcome_regressors)?
            .iter()
            .map(|key| key.offset.unsigned_abs() as usize)
            .max()
            .unwrap_or(0);
        Ok(base.max(extra))
    }

    /// Outcome-model regressors kept for the selected-AIPW nuisance (see
    /// [`Self::adjust_temporal_series`]); empty for every other correction.
    fn retained_outcome_regressors(
        &self,
        query: &ResponseQuery,
        offset: i32,
        outcome_regressors: &[TemporalNodeKey],
    ) -> Result<Vec<TemporalNodeKey>, EstimationError> {
        let ObservationSpec::Selected { latent, observed, indicator } = &query.observation else {
            return Ok(Vec::new());
        };
        if self.options.selected_correction != SelectedOutcomeCorrection::Aipw {
            return Ok(Vec::new());
        }
        let conditioning = exact_outcome_independence(query)?;
        let mut keys: Vec<TemporalNodeKey> = outcome_regressors
            .iter()
            .copied()
            .filter(|key| {
                key.offset <= 0
                    && ![*latent, *observed, *indicator].contains(&key.variable)
                    && !(key.offset == offset && conditioning.contains(&key.variable))
            })
            .collect();
        keys.sort();
        keys.dedup();
        Ok(keys)
    }

    /// Column-major extra outcome-model regressors for the observation rows whose outcome
    /// times are `rows` (series indices, in table order).
    fn outcome_model_extras(
        &self,
        data: &TimeSeriesData,
        query: &ResponseQuery,
        offset: i32,
        outcome_regressors: &[TemporalNodeKey],
        rows: &[usize],
    ) -> Result<Option<OutcomeModelExtras>, EstimationError> {
        let keys = self.retained_outcome_regressors(query, offset, outcome_regressors)?;
        if keys.is_empty() || rows.is_empty() {
            return Ok(None);
        }
        let reach = keys.iter().map(|key| key.offset.unsigned_abs() as usize).max().unwrap_or(0);
        let lead = rows.iter().take_while(|&&row| row < reach).count();
        if rows[lead..].iter().any(|&row| row < reach) {
            return Err(EstimationError::unsupported(
                "temporal observation rows must reach every outcome-model regressor",
            ));
        }
        let mut values = Vec::with_capacity(rows.len() * keys.len());
        for key in &keys {
            let lag = key.offset.unsigned_abs() as usize;
            let full = data.float64_values(key.variable)?;
            for &row in rows {
                values.push(if row >= lag { full[row - lag] } else { f64::NAN });
            }
        }
        if values.chunks(rows.len()).any(|column| column[lead..].iter().any(|v| !v.is_finite())) {
            return Err(EstimationError::unsupported(
                "observation-model covariates must be completely observed and finite",
            ));
        }
        Ok(Some(OutcomeModelExtras { values, ncols: keys.len(), lead }))
    }

    /// Evaluate the opt-in Gaussian likelihood for the query's observation mechanism.
    ///
    /// The query must explicitly declare
    /// `ObservationAssumption::Structural("gaussian_observation_likelihood")`. `means` must
    /// contain one latent-outcome mean per source row. Selected outcomes are not supported by
    /// this likelihood path.
    ///
    /// # Errors
    ///
    /// Missing structural opt-in, unsupported mechanism, malformed data, or invalid likelihood.
    pub fn gaussian_log_likelihood(
        &self,
        data: &TabularData,
        query: &ResponseQuery,
        means: &[f64],
        sigma: f64,
    ) -> Result<f64, EstimationError> {
        query.validate()?;
        if means.len() != data.row_count() {
            return Err(EstimationError::unsupported(
                "Gaussian observation means must align with source rows",
            ));
        }
        let opted_in = query.observation_assumptions.iter().any(|assumption| {
            matches!(assumption, ObservationAssumption::Structural(name) if name.as_ref() == "gaussian_observation_likelihood")
        });
        if !opted_in {
            return Err(EstimationError::unsupported(
                "Gaussian censoring/truncation likelihood requires an explicit structural opt-in",
            ));
        }
        let observations = gaussian_observations(data, &query.observation)?;
        Ok(gaussian_observation_log_likelihood(&observations, means, sigma)?)
    }

    fn validate_options(&self) -> Result<(), EstimationError> {
        if !self.options.observation_probability_floor.is_finite()
            || !(0.0..0.5).contains(&self.options.observation_probability_floor)
            || !self.options.censoring_survival_floor.is_finite()
            || !(0.0..1.0).contains(&self.options.censoring_survival_floor)
            || self.options.crossfit_folds < 2
        {
            return Err(EstimationError::unsupported("invalid observation-estimator options"));
        }
        Ok(())
    }

    fn selected(
        &self,
        data: &TabularData,
        query: &ResponseQuery,
        observed_id: VariableId,
        indicator_id: VariableId,
        extras: Option<&OutcomeModelExtras>,
        layout: FoldLayout<'_>,
    ) -> Result<ObservationAdjustedOutcome, EstimationError> {
        let conditioning = exact_outcome_independence(query)?;
        if conditioning.iter().any(|id| *id == observed_id || *id == indicator_id) {
            return Err(EstimationError::unsupported(
                "observation-model conditions cannot include observed outcome or indicator",
            ));
        }
        let observed = data.float64_values(observed_id)?;
        let indicator = data.float64_values(indicator_id)?;
        let covariates = read_complete_columns(data, conditioning)?;
        self.selected_from_columns(
            &observed,
            &indicator,
            &covariates,
            conditioning.len(),
            extras,
            layout,
        )
    }

    fn selected_from_columns(
        &self,
        observed: &[f64],
        indicator: &[f64],
        covariates: &[f64],
        n_cov: usize,
        extras: Option<&OutcomeModelExtras>,
        layout: FoldLayout<'_>,
    ) -> Result<ObservationAdjustedOutcome, EstimationError> {
        if indicator.iter().all(|&r| r == 1.0) {
            if observed.iter().any(|value| !value.is_finite()) {
                return Err(EstimationError::unsupported("selected outcomes must be finite"));
            }
            return Ok(ObservationAdjustedOutcome {
                values: observed.to_vec(),
                weights: vec![1.0; observed.len()],
                method: Arc::from("observation.selected.complete_collapse.v1"),
            });
        }
        let (probabilities, outcome_predictions) = match self.options.selected_correction {
            // Plain IPW keeps the in-sample maximum-likelihood propensity. That is the
            // published estimator, and estimating the propensity in sample is what makes it
            // efficient — cross-fitting here would deviate from the citation to no end.
            SelectedOutcomeCorrection::Ipw => {
                let fit = fit_observation_logistic(
                    indicator,
                    covariates,
                    n_cov,
                    self.options.observation_probability_floor,
                )?;
                (fit.probabilities, None)
            }
            // The augmented path is where sample splitting earns its keep: the AIPW
            // double-robustness and asymptotic-linearity arguments assume the nuisances were
            // not fit on the row they are evaluated at.
            SelectedOutcomeCorrection::Aipw => {
                let nuisances = self
                    .crossfit_selected_nuisances(observed, indicator, covariates, extras, layout)?;
                (nuisances.probabilities, Some(nuisances.outcome_predictions))
            }
        };
        let values = selected_outcome_pseudo_values(
            observed,
            indicator,
            &probabilities,
            outcome_predictions.as_deref(),
        )?;
        let weights = indicator
            .iter()
            .zip(&probabilities)
            .map(|(&r, &p)| if r == 1.0 { 1.0 / p } else { 0.0 })
            .collect();
        Ok(ObservationAdjustedOutcome {
            values,
            weights,
            method: Arc::from(match self.options.selected_correction {
                SelectedOutcomeCorrection::Ipw => "observation.selected.logistic_ipw.v1",
                SelectedOutcomeCorrection::Aipw => "observation.selected.crossfit_logistic_aipw.v1",
            }),
        })
    }

    /// Fit both selected-outcome nuisances out of fold.
    ///
    /// Folds follow `layout`: the deterministic row-index folds used elsewhere in this
    /// crate for exchangeable rows, contiguous time blocks for temporal rows. Every row
    /// receives a probability and an outcome prediction from a model fit without it (or any
    /// copy of it: repeated positions share a fold). A fold
    /// whose training rows cannot support either model is refused rather than quietly
    /// falling back to an in-sample fit, which would reintroduce exactly the bias the
    /// splitting removes.
    ///
    /// With `extras`, the outcome regression also conditions on those columns (the
    /// selection model does not); rows before `extras.lead` keep the conditioning-only
    /// outcome regression, fit on the same training fold.
    fn crossfit_selected_nuisances(
        &self,
        observed: &[f64],
        indicator: &[f64],
        covariates: &[f64],
        extras: Option<&OutcomeModelExtras>,
        layout: FoldLayout<'_>,
    ) -> Result<CrossFittedSelectedNuisances, EstimationError> {
        let n = indicator.len();
        let folds = self.options.crossfit_folds;
        let ncols = if n == 0 { 0 } else { covariates.len() / n };
        if extras.is_some_and(|extras| extras.values.len() != n * extras.ncols) {
            return Err(EstimationError::unsupported(
                "outcome-model regressors must align with observation rows",
            ));
        }
        // Declared conditioning columns followed by the extra outcome-model columns.
        let augmented = extras.map(|extras| {
            let mut values = covariates.to_vec();
            values.extend_from_slice(&extras.values);
            (values, ncols + extras.ncols, extras.lead)
        });
        if folds > n {
            return Err(EstimationError::unsupported(
                "cross-fitting folds cannot exceed observed rows",
            ));
        }
        let mut probabilities = vec![f64::NAN; n];
        let mut outcome_predictions = vec![f64::NAN; n];
        let fold_of: Vec<usize> = (0..n).map(|row| layout.fold_of(row, folds)).collect();
        for fold in 0..folds {
            let train: Vec<usize> = (0..n).filter(|&i| fold_of[i] != fold).collect();
            let valid: Vec<usize> = (0..n).filter(|&i| fold_of[i] == fold).collect();
            if valid.is_empty() {
                continue;
            }
            let train_indicator: Vec<f64> = train.iter().map(|&i| indicator[i]).collect();
            let train_covariates = subset_colmajor(covariates, n, ncols, &train);
            let fit = fit_observation_logistic(
                &train_indicator,
                &train_covariates,
                ncols,
                self.options.observation_probability_floor,
            )
            .map_err(|_| {
                EstimationError::unsupported(
                    "a cross-fitting fold cannot support the observation model; reduce crossfit_folds or supply more rows covering both observed and unobserved outcomes",
                )
            })?;
            let lead = augmented.as_ref().map_or(0, |(_, _, lead)| *lead);
            let train_observed: Vec<f64> = train.iter().map(|&i| observed[i]).collect();
            let base = if augmented.is_none() || valid.iter().any(|&row| row < lead) {
                Some(fit_selected_outcome_regression(
                    &train_observed,
                    &train_indicator,
                    &train_covariates,
                    ncols,
                )?)
            } else {
                None
            };
            let augmented_fit = augmented
                .as_ref()
                .map(|(values, width, lead)| {
                    let rows: Vec<usize> = train.iter().copied().filter(|&i| i >= *lead).collect();
                    let train_values = subset_colmajor(values, n, *width, &rows);
                    let train_observed: Vec<f64> = rows.iter().map(|&i| observed[i]).collect();
                    let train_indicator: Vec<f64> = rows.iter().map(|&i| indicator[i]).collect();
                    fit_selected_outcome_regression(
                        &train_observed,
                        &train_indicator,
                        &train_values,
                        *width,
                    )
                })
                .transpose()?;
            for &row in &valid {
                let features = covariate_row(covariates, n, ncols, row);
                probabilities[row] = fit.probability_at(&features)?;
                outcome_predictions[row] = match (&augmented, &augmented_fit, &base) {
                    (Some((values, width, lead)), Some(coefficients), _) if row >= *lead => {
                        predict_linear(coefficients, &covariate_row(values, n, *width, row))
                    }
                    (_, _, Some(coefficients)) => predict_linear(coefficients, &features),
                    _ => f64::NAN,
                };
            }
        }
        if probabilities.iter().chain(&outcome_predictions).any(|value| !value.is_finite()) {
            return Err(EstimationError::unsupported(
                "cross-fitted observation nuisances produced a non-finite prediction",
            ));
        }
        Ok(CrossFittedSelectedNuisances { probabilities, outcome_predictions })
    }

    #[allow(clippy::too_many_arguments)]
    fn censored(
        &self,
        data: &TabularData,
        query: &ResponseQuery,
        observed_id: VariableId,
        censoring_id: VariableId,
        event_id: VariableId,
        delayed_entry: Option<VariableId>,
        reverse: bool,
    ) -> Result<ObservationAdjustedOutcome, EstimationError> {
        let conditioning = censoring_independence(query)?;
        if conditioning.iter().any(|v| [observed_id, censoring_id, event_id].contains(v)) {
            return Err(EstimationError::unsupported(
                "censoring conditioning cannot include recorded outcome, censoring time, or event indicator",
            ));
        }
        if !conditioning.is_empty() && delayed_entry.is_some() {
            return Err(EstimationError::unsupported(
                "conditional Cox IPCW does not support delayed entry",
            ));
        }
        for id in [observed_id, censoring_id, event_id] {
            if (0..data.row_count())
                .any(|row| !data.column(id).is_ok_and(|c| c.validity().is_valid(row)))
            {
                return Err(EstimationError::unsupported(
                    "censoring inputs must be completely observed",
                ));
            }
        }
        let observed = data.float64_values(observed_id)?;
        let censoring = data.float64_values(censoring_id)?;
        let event = data.float64_values(event_id)?;
        let covariates = if conditioning.is_empty() {
            Vec::new()
        } else {
            read_complete_columns(data, conditioning)?
        };
        let entry_values = delayed_entry.map(|id| data.float64_values(id)).transpose()?;
        self.censored_from_columns(
            &observed,
            &censoring,
            &event,
            &covariates,
            conditioning.len(),
            entry_values.as_deref(),
            reverse,
        )
    }

    fn censored_from_columns(
        &self,
        observed: &[f64],
        censoring: &[f64],
        event: &[f64],
        covariates: &[f64],
        n_cov: usize,
        delayed_entry: Option<&[f64]>,
        reverse: bool,
    ) -> Result<ObservationAdjustedOutcome, EstimationError> {
        if observed.iter().chain(censoring).any(|v| !v.is_finite()) {
            return Err(EstimationError::unsupported(
                "censoring times and recorded outcomes must be finite",
            ));
        }
        for i in 0..observed.len() {
            let compatible =
                if reverse { observed[i] >= censoring[i] } else { observed[i] <= censoring[i] };
            if !compatible || (event[i] == 0.0 && observed[i] != censoring[i]) {
                return Err(EstimationError::unsupported(
                    "recorded outcome is incompatible with its censoring value/event",
                ));
            }
        }
        let transformed: Vec<f64> =
            observed.iter().map(|&value| if reverse { -value } else { value }).collect();
        let weights = if n_cov == 0 {
            kaplan_meier_ipcw(
                &transformed,
                event,
                delayed_entry,
                self.options.censoring_survival_floor,
            )?
        } else {
            antecedent_stats::cox_ipcw(
                &transformed,
                event,
                covariates,
                n_cov,
                self.options.censoring_survival_floor,
            )?
            .weights
        };
        // Horvitz–Thompson transform: Y* = Y · (δ / G). Censored rows contribute
        // exactly 0; uncensored rows are upweighted by 1/G. The unweighted mean
        // of Y* is the IPCW mean. Dropping the zeros would be complete-case
        // analysis, not IPCW.
        let values = observed.iter().zip(&weights).map(|(&y, &w)| y * w).collect();
        Ok(ObservationAdjustedOutcome {
            values,
            weights,
            method: Arc::from(if n_cov > 0 {
                if reverse {
                    "observation.left_censored.cox_ipcw_sign_reversal.v1"
                } else {
                    "observation.right_censored.cox_ipcw.v1"
                }
            } else if reverse {
                "observation.left_censored.km_ipcw_sign_reversal.v1"
            } else if delayed_entry.is_some() {
                "observation.right_censored.km_ipcw_delayed_entry.v1"
            } else {
                "observation.right_censored.km_ipcw.v1"
            }),
        })
    }
}

/// Refuse an observation-adjusted Sequence whose unfolded design carries the outcome at
/// a nonzero lag (see [`temporal_sequence_outcome_regressors`]). The correction replaces
/// the outcome column by pseudo-outcomes, so every lagged outcome regressor would be a
/// noisy proxy of the latent outcome and attenuate its coefficient.
fn require_unlagged_sequence_outcome(
    query: &ResponseQuery,
    outcome: VariableId,
    outcome_regressors: &[TemporalNodeKey],
) -> Result<(), EstimationError> {
    let sequence = crate::plan_from_response_query(query)?
        .and_then(|plan| plan.mechanism_overlays())
        .is_some();
    if sequence && outcome_regressors.iter().any(|key| key.variable == outcome && key.offset != 0) {
        return Err(EstimationError::unsupported(
            "observation-adjusted Sequence responses require the outcome to enter the unfolded design only at the outcome time; a lagged outcome regressor would be replaced by pseudo-outcomes (errors in variables)",
        ));
    }
    Ok(())
}

fn temporal_conditioning_ids(query: &ResponseQuery) -> Result<&[VariableId], EstimationError> {
    match &query.observation {
        ObservationSpec::Selected { .. } => exact_outcome_independence(query),
        ObservationSpec::RightCensored { .. } | ObservationSpec::LeftCensored { .. } => {
            censoring_independence(query)
        }
        _ => Err(EstimationError::unsupported(
            "temporal observation correction supports selected and left/right censoring only",
        )),
    }
}

pub(crate) fn require_temporal_observation_containment(
    query: &ResponseQuery,
    treatment: VariableId,
    adjustment: &[VariableId],
) -> Result<(), EstimationError> {
    match &query.observation {
        ObservationSpec::Selected { .. } => {
            let conditioning = exact_outcome_independence(query)?;
            if !conditioning.contains(&treatment)
                || adjustment.iter().any(|variable| !conditioning.contains(variable))
            {
                return Err(EstimationError::unsupported(
                    "selected-outcome response correction requires OutcomeIndependentGiven to include the treatment and every causal adjustment variable",
                ));
            }
        }
        ObservationSpec::RightCensored { .. } | ObservationSpec::LeftCensored { .. } => {
            let conditioning = censoring_independence(query)?;
            if !conditioning.is_empty()
                && (!conditioning.contains(&treatment)
                    || adjustment.iter().any(|variable| !conditioning.contains(variable)))
            {
                return Err(EstimationError::unsupported(
                    "conditional censoring response requires IndependentGiven to include treatment and every causal adjustment variable",
                ));
            }
        }
        _ => {}
    }
    Ok(())
}

fn lag_aligned_observation_table(
    data: &TimeSeriesData,
    conditioning: &[VariableId],
    offset: i32,
) -> Result<(TabularData, usize), EstimationError> {
    let lag = offset.unsigned_abs() as usize;
    let n = data.row_count();
    if n <= lag {
        return Err(EstimationError::unsupported(
            "temporal observation conditioning requires more rows than the treatment lag",
        ));
    }
    let start = lag;
    let rows = n - start;
    let schema = data.schema().clone();
    let mut owned = Vec::with_capacity(schema.len());
    for var in schema.variables() {
        let full = data.float64_values(var.id)?;
        let slice = if conditioning.contains(&var.id) {
            full[..rows].to_vec()
        } else {
            full[start..].to_vec()
        };
        owned.push((var.name.clone(), slice));
    }
    let pairs: Vec<(&str, &[f64])> =
        owned.iter().map(|(name, values)| (name.as_ref(), values.as_slice())).collect();
    let table = TabularData::try_from_schema_f64(schema, pairs)?;
    Ok((table, start))
}

fn diagnostic_weight_summary(weights: &[f64]) -> (f64, f64, f64) {
    let minimum =
        weights.iter().copied().filter(|weight| *weight > 0.0).fold(f64::INFINITY, f64::min);
    let maximum = weights.iter().copied().fold(0.0_f64, f64::max);
    let sum = weights.iter().sum::<f64>();
    let sum_squares = weights.iter().map(|weight| weight * weight).sum::<f64>();
    let effective_sample_size = if sum_squares > 0.0 { sum * sum / sum_squares } else { 0.0 };
    (if minimum.is_finite() { minimum } else { 0.0 }, maximum, effective_sample_size)
}

fn exact_outcome_independence(query: &ResponseQuery) -> Result<&[VariableId], EstimationError> {
    let mut claims =
        query.observation_assumptions.iter().filter_map(|assumption| match assumption {
            ObservationAssumption::OutcomeIndependentGiven(vars) => Some(vars.as_ref()),
            _ => None,
        });
    let Some(first) = claims.next() else {
        return Err(EstimationError::unsupported(
            "selected-outcome correction requires OutcomeIndependentGiven",
        ));
    };
    if claims.next().is_some() || query.observation_assumptions.len() != 1 {
        return Err(EstimationError::unsupported(
            "selected-outcome correction requires exactly one supported observation assumption",
        ));
    }
    Ok(first)
}

fn censoring_independence(query: &ResponseQuery) -> Result<&[VariableId], EstimationError> {
    if query.observation_assumptions.len() != 1 {
        return Err(EstimationError::unsupported(
            "IPCW requires exactly one independence assumption",
        ));
    }
    match &query.observation_assumptions[0] {
        ObservationAssumption::IndependentGiven(vars) => Ok(vars),
        // Preserve the previously accepted empty outcome-independence spelling.
        ObservationAssumption::OutcomeIndependentGiven(vars) if vars.is_empty() => Ok(vars),
        _ => {
            Err(EstimationError::unsupported("conditional censoring requires IndependentGiven(Z)"))
        }
    }
}

fn gather_anchor_column(
    data: &TimeSeriesData,
    id: VariableId,
    anchors: &[usize],
    shift: usize,
) -> Result<Vec<f64>, EstimationError> {
    let full = data.float64_values(id)?;
    Ok(anchors.iter().map(|&anchor| full[anchor - shift]).collect())
}

fn gather_anchor_columns(
    data: &TimeSeriesData,
    ids: &[VariableId],
    anchors: &[usize],
    shift: usize,
) -> Result<Vec<f64>, EstimationError> {
    let mut values = Vec::with_capacity(anchors.len() * ids.len());
    for &id in ids {
        let full = data.float64_values(id)?;
        for &anchor in anchors {
            let value = full[anchor - shift];
            if !value.is_finite() {
                return Err(EstimationError::unsupported(
                    "observation-model covariates must be completely observed and finite",
                ));
            }
            values.push(value);
        }
    }
    Ok(values)
}

fn read_complete_columns(
    data: &TabularData,
    variables: &[VariableId],
) -> Result<Vec<f64>, EstimationError> {
    let mut values = Vec::with_capacity(data.row_count() * variables.len());
    for &variable in variables {
        let column = data.float64_values(variable)?;
        if column.iter().any(|value| !value.is_finite())
            || (0..data.row_count())
                .any(|row| !data.column(variable).is_ok_and(|c| c.validity().is_valid(row)))
        {
            return Err(EstimationError::unsupported(
                "observation-model covariates must be completely observed and finite",
            ));
        }
        values.extend(column);
    }
    Ok(values)
}

/// Column-major covariates for a subset of rows, preserving column order.
fn subset_colmajor(covariates: &[f64], n: usize, ncols: usize, rows: &[usize]) -> Vec<f64> {
    let mut out = vec![0.0; rows.len() * ncols];
    for col in 0..ncols {
        for (position, &row) in rows.iter().enumerate() {
            out[col * rows.len() + position] = covariates[col * n + row];
        }
    }
    out
}

/// One row's covariate values, excluding the intercept.
fn covariate_row(covariates: &[f64], n: usize, ncols: usize, row: usize) -> Vec<f64> {
    (0..ncols).map(|col| covariates[col * n + row]).collect()
}

/// Linear prediction from an intercept-leading coefficient vector.
fn predict_linear(coefficients: &[f64], features: &[f64]) -> f64 {
    coefficients[0]
        + coefficients[1..].iter().zip(features).map(|(beta, value)| beta * value).sum::<f64>()
}

/// Selected-outcome regression coefficients, intercept first.
///
/// Returns coefficients rather than fitted values so the caller decides which rows the model
/// is evaluated on; returning in-sample predictions would make out-of-fold use impossible.
fn fit_selected_outcome_regression(
    observed: &[f64],
    indicator: &[f64],
    covariates: &[f64],
    ncols: usize,
) -> Result<Vec<f64>, EstimationError> {
    let n = observed.len();
    let rows: Vec<usize> = (0..n).filter(|&i| indicator[i] == 1.0).collect();
    if rows.len() <= ncols + 1 {
        return Err(EstimationError::unsupported(
            "too few selected rows for augmented outcome regression",
        ));
    }
    let train_n = rows.len();
    let mut train_x = vec![1.0; train_n * (ncols + 1)];
    for col in 0..ncols {
        for (r, &source) in rows.iter().enumerate() {
            train_x[(col + 1) * train_n + r] = covariates[col * n + source];
        }
    }
    let train_y: Vec<f64> = rows.iter().map(|&i| observed[i]).collect();
    if train_y.iter().any(|v| !v.is_finite()) {
        return Err(EstimationError::unsupported("selected outcomes must be finite"));
    }
    let mut workspace = LeastSquaresWorkspace::default();
    let fit = fit_glm(
        GlmFamily::GaussianIdentity,
        GlmDesignRef { x_colmajor: &train_x, nrows: train_n, ncols: ncols + 1, y: &train_y },
        &FaerBackend,
        &mut workspace,
        &GlmOptions::default(),
    )?;
    fit.require_ok()?;
    Ok(fit.coefficients)
}

fn gaussian_observations(
    data: &TabularData,
    spec: &ObservationSpec,
) -> Result<Vec<GaussianObservation>, EstimationError> {
    Ok(match spec {
        ObservationSpec::Complete => {
            return Err(EstimationError::unsupported(
                "complete Gaussian outcomes use the ordinary complete-data likelihood",
            ));
        }
        ObservationSpec::Selected { .. } => {
            return Err(EstimationError::unsupported(
                "selected outcomes use logistic IPW/AIPW, not the Gaussian observation likelihood",
            ));
        }
        ObservationSpec::RightCensored { observed, event, .. } => {
            let y = data.float64_values(*observed)?;
            let delta = data.float64_values(*event)?;
            binary_events(&delta)?;
            y.into_iter()
                .zip(delta)
                .map(|(value, d)| {
                    if d == 1.0 {
                        GaussianObservation::Exact(value)
                    } else {
                        GaussianObservation::RightCensored(value)
                    }
                })
                .collect()
        }
        ObservationSpec::LeftCensored { observed, event, .. } => {
            let y = data.float64_values(*observed)?;
            let delta = data.float64_values(*event)?;
            binary_events(&delta)?;
            y.into_iter()
                .zip(delta)
                .map(|(value, d)| {
                    if d == 1.0 {
                        GaussianObservation::Exact(value)
                    } else {
                        GaussianObservation::LeftCensored(value)
                    }
                })
                .collect()
        }
        ObservationSpec::IntervalCensored { lower, upper, .. } => {
            let lower = data.float64_values(*lower)?;
            let upper = data.float64_values(*upper)?;
            lower
                .into_iter()
                .zip(upper)
                .map(|(lower, upper)| GaussianObservation::IntervalCensored { lower, upper })
                .collect()
        }
        ObservationSpec::Truncated { observed, lower, upper, .. } => {
            let value = data.float64_values(*observed)?;
            let lower = lower.map(|id| data.float64_values(id)).transpose()?;
            let upper = upper.map(|id| data.float64_values(id)).transpose()?;
            (0..data.row_count())
                .map(|i| GaussianObservation::Truncated {
                    value: value[i],
                    lower: lower.as_ref().map_or(f64::NEG_INFINITY, |v| v[i]),
                    upper: upper.as_ref().map_or(f64::INFINITY, |v| v[i]),
                })
                .collect()
        }
    })
}

fn binary_events(events: &[f64]) -> Result<(), EstimationError> {
    if events.iter().all(|&event| event == 0.0 || event == 1.0) {
        Ok(())
    } else {
        Err(EstimationError::unsupported("censoring event indicators must be binary"))
    }
}

#[cfg(test)]
mod tests {
    use antecedent_core::{
        ContinuousDomain, GridSpec, ObservationAssumption, ResponseFunctional, ResponseQuery,
    };

    use super::*;

    fn response_query(outcome: VariableId, treatment: VariableId) -> ResponseQuery {
        ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome,
            treatment: ContinuousDomain::new(treatment, GridSpec::Values(Arc::from([-0.1, 0.1]))),
        })
    }

    /// Selection depends on `x`, so a complete-case mean is biased and only a correction
    /// that uses the observation model recovers `E[Y] = 2 + 3 E[x]`.
    fn selection_biased_sample(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let x: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let r: Vec<f64> =
            x.iter().enumerate().map(|(i, &x)| f64::from(x > 0.25 || i % 3 == 0)).collect();
        let y: Vec<f64> = x
            .iter()
            .zip(&r)
            .map(|(&x, &r)| if r == 1.0 { 2.0 + 3.0 * x } else { f64::NAN })
            .collect();
        (x, r, y)
    }

    fn selected_query() -> ResponseQuery {
        response_query(VariableId::from_raw(1), VariableId::from_raw(0)).with_observation(
            ObservationSpec::Selected {
                latent: VariableId::from_raw(1),
                observed: VariableId::from_raw(1),
                indicator: VariableId::from_raw(2),
            },
            [ObservationAssumption::OutcomeIndependentGiven(Arc::from([VariableId::from_raw(3)]))],
        )
    }

    fn selected_table(x: &[f64], y: &[f64], r: &[f64]) -> TabularData {
        TabularData::from_f64_columns([("a", x), ("y", y), ("r", r), ("x", x)]).unwrap()
    }

    #[test]
    fn selected_aipw_nuisances_are_fit_without_the_row_they_are_applied_to() {
        // The property cross-fitting exists to provide. Row 0 falls in fold 0, so the
        // probability used for it must come from a model fit on the rows outside fold 0 —
        // not from the all-rows fit, which is what this path used to do.
        let n = 100;
        let (x, r, y) = selection_biased_sample(n);
        let data = selected_table(&x, &y, &r);
        let estimator = ObservationMechanismEstimator::default();
        let adjusted = estimator.adjusted_outcome(&data, &selected_query(), None).unwrap();
        assert_eq!(adjusted.method.as_ref(), "observation.selected.crossfit_logistic_aipw.v1");

        let folds = estimator.options.crossfit_folds;
        let train: Vec<usize> = (0..n).filter(|i| i % folds != 0).collect();
        let train_indicator: Vec<f64> = train.iter().map(|&i| r[i]).collect();
        let train_covariates: Vec<f64> = train.iter().map(|&i| x[i]).collect();
        let out_of_fold = fit_observation_logistic(
            &train_indicator,
            &train_covariates,
            1,
            estimator.options.observation_probability_floor,
        )
        .unwrap()
        .probability_at(&[x[0]])
        .unwrap();

        let in_sample =
            fit_observation_logistic(&r, &x, 1, estimator.options.observation_probability_floor)
                .unwrap()
                .probabilities[0];

        assert_eq!(r[0], 1.0, "row 0 must be selected for its weight to be 1/p");
        let used = 1.0 / adjusted.weights[0];
        assert!(
            (used - out_of_fold).abs() < 1e-9,
            "row 0 used p={used}, expected the out-of-fold p={out_of_fold}"
        );
        assert!(
            (used - in_sample).abs() > 1e-12,
            "out-of-fold and in-sample probabilities coincide; the test cannot tell them apart"
        );
    }

    #[test]
    fn crossfit_selected_aipw_recovers_the_latent_mean_under_biased_selection() {
        let n = 150;
        let (x, r, y) = selection_biased_sample(n);
        let data = selected_table(&x, &y, &r);
        let adjusted = ObservationMechanismEstimator::default()
            .adjusted_outcome(&data, &selected_query(), None)
            .unwrap();

        let truth = x.iter().map(|&x| 2.0 + 3.0 * x).sum::<f64>() / n as f64;
        let corrected = adjusted.values.iter().sum::<f64>() / n as f64;
        let complete_case = {
            let selected: Vec<f64> =
                y.iter().zip(&r).filter(|&(_, &r)| r == 1.0).map(|(&y, _)| y).collect();
            selected.iter().sum::<f64>() / selected.len() as f64
        };
        assert!(
            (corrected - truth).abs() < 1e-6,
            "cross-fitted AIPW gave {corrected}, truth {truth}"
        );
        assert!(
            (complete_case - truth).abs() > 0.1,
            "the complete-case mean must be visibly biased or this proves nothing"
        );
    }

    #[test]
    fn a_fold_that_cannot_support_the_observation_model_is_refused() {
        // Every unobserved row sits in one fold, so the other folds train on selected rows
        // only and the logistic model has no contrast. Refusing is the point: silently
        // refitting in sample would return exactly the biased answer cross-fitting removes.
        let n = 60usize;
        let x: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let r: Vec<f64> = (0..n).map(|i| f64::from(i % 5 != 0)).collect();
        let y: Vec<f64> = x
            .iter()
            .zip(&r)
            .map(|(&x, &r)| if r == 1.0 { 2.0 + 3.0 * x } else { f64::NAN })
            .collect();
        let data = selected_table(&x, &y, &r);
        let error = ObservationMechanismEstimator::default()
            .adjusted_outcome(&data, &selected_query(), None)
            .unwrap_err();
        assert!(error.to_string().contains("cross-fitting fold"), "got {error}");
    }

    #[test]
    fn selected_aipw_requires_and_uses_explicit_outcome_independence() {
        let x: Vec<f64> = (0..80).map(|i| f64::from(i) / 79.0).collect();
        let r: Vec<f64> = (0..80).map(|i| f64::from(i % 3 != 0)).collect();
        let y: Vec<f64> = x
            .iter()
            .zip(&r)
            .map(|(&x, &r)| if r == 1.0 { 2.0 + 3.0 * x } else { f64::NAN })
            .collect();
        let data = TabularData::from_f64_columns([
            ("a", x.as_slice()),
            ("y", y.as_slice()),
            ("r", r.as_slice()),
            ("x", x.as_slice()),
        ])
        .unwrap();
        let mut query = response_query(VariableId::from_raw(1), VariableId::from_raw(0));
        query = query.with_observation(
            ObservationSpec::Selected {
                latent: VariableId::from_raw(1),
                observed: VariableId::from_raw(1),
                indicator: VariableId::from_raw(2),
            },
            [ObservationAssumption::OutcomeIndependentGiven(Arc::from([VariableId::from_raw(3)]))],
        );
        let adjusted =
            ObservationMechanismEstimator::default().adjusted_outcome(&data, &query, None).unwrap();
        assert!(
            adjusted.values.iter().zip(&x).all(|(&got, &x)| (got - (2.0 + 3.0 * x)).abs() < 1e-8)
        );
    }

    #[test]
    fn selected_aipw_composes_into_point_curve_without_invalid_bands() {
        let a: Vec<f64> = (0..80).map(|i| f64::from(i) / 79.0).collect();
        let r: Vec<f64> = (0..80).map(|i| f64::from(i % 3 != 0)).collect();
        let y: Vec<f64> = a
            .iter()
            .zip(&r)
            .map(|(&a, &r)| if r == 1.0 { 2.0 + 3.0 * a } else { f64::NAN })
            .collect();
        let data = TabularData::from_f64_columns([
            ("a", a.as_slice()),
            ("y", y.as_slice()),
            ("r", r.as_slice()),
        ])
        .unwrap();
        let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: VariableId::from_raw(1),
            treatment: ContinuousDomain::new(
                VariableId::from_raw(0),
                GridSpec::Values(Arc::from([0.2, 0.8])),
            ),
        })
        .with_observation(
            ObservationSpec::Selected {
                latent: VariableId::from_raw(1),
                observed: VariableId::from_raw(1),
                indicator: VariableId::from_raw(2),
            },
            [ObservationAssumption::OutcomeIndependentGiven(Arc::from([VariableId::from_raw(0)]))],
        );
        let response = ObservationMechanismEstimator::default()
            .estimate_mean_curve(
                &ContinuousResponseEstimator::new(Arc::from([])),
                &data,
                &query,
                None,
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
            )
            .unwrap();
        assert_eq!(response.uncertainty, ResponseUncertainty::None);
        assert_eq!(response.provenance_id.as_ref(), "estimate.response.observation_adjusted");
        assert!(
            response.support.warnings.iter().any(|warning| warning.code.as_ref()
                == "response.observation_joint_uncertainty_unavailable")
        );
    }

    #[test]
    fn selected_curve_refuses_missing_treatment_in_observation_conditioning() {
        let values: Vec<f64> = (0..40).map(f64::from).collect();
        let selected = vec![1.0; values.len()];
        let data = TabularData::from_f64_columns([
            ("a", values.as_slice()),
            ("y", values.as_slice()),
            ("r", selected.as_slice()),
            ("x", values.as_slice()),
        ])
        .unwrap();
        let query = response_query(VariableId::from_raw(1), VariableId::from_raw(0))
            .with_observation(
                ObservationSpec::Selected {
                    latent: VariableId::from_raw(1),
                    observed: VariableId::from_raw(1),
                    indicator: VariableId::from_raw(2),
                },
                [ObservationAssumption::OutcomeIndependentGiven(Arc::from([
                    VariableId::from_raw(3),
                ]))],
            );
        let error = ObservationMechanismEstimator::default()
            .estimate_mean_curve(
                &ContinuousResponseEstimator::new(Arc::from([])),
                &data,
                &query,
                None,
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
            )
            .unwrap_err();
        assert!(error.to_string().contains("include the treatment"));
    }

    #[test]
    fn conditional_cox_separation_fails_closed() {
        let values = [1.0, 2.0, 3.0, 4.0];
        let event = [0.0, 1.0, 1.0, 1.0];
        let data = TabularData::from_f64_columns([
            ("a", values.as_slice()),
            ("y", values.as_slice()),
            ("c", values.as_slice()),
            ("d", event.as_slice()),
        ])
        .unwrap();
        let query = response_query(VariableId::from_raw(1), VariableId::from_raw(0))
            .with_observation(
                ObservationSpec::RightCensored {
                    latent: VariableId::from_raw(1),
                    observed: VariableId::from_raw(1),
                    censoring: VariableId::from_raw(2),
                    event: VariableId::from_raw(3),
                },
                [ObservationAssumption::IndependentGiven(Arc::from([VariableId::from_raw(0)]))],
            );
        let error = ObservationMechanismEstimator::default()
            .adjusted_outcome(&data, &query, None)
            .unwrap_err();
        assert!(error.to_string().contains("Cox"), "{error}");
    }

    #[test]
    fn gaussian_interval_likelihood_is_explicitly_opt_in() {
        let lower = [-1.0, 0.0];
        let upper = [0.0, 1.0];
        let treatment = [0.0, 1.0];
        let data = TabularData::from_f64_columns([
            ("a", treatment.as_slice()),
            ("lo", lower.as_slice()),
            ("hi", upper.as_slice()),
        ])
        .unwrap();
        let base = response_query(VariableId::from_raw(1), VariableId::from_raw(0));
        let spec = ObservationSpec::IntervalCensored {
            latent: VariableId::from_raw(1),
            lower: VariableId::from_raw(1),
            upper: VariableId::from_raw(2),
        };
        let without = base.clone().with_observation(spec.clone(), Arc::from([]));
        assert!(
            ObservationMechanismEstimator::default()
                .gaussian_log_likelihood(&data, &without, &[0.0, 0.0], 1.0)
                .is_err()
        );
        let with = base.with_observation(
            spec,
            [ObservationAssumption::Structural(Arc::from("gaussian_observation_likelihood"))],
        );
        assert!(
            ObservationMechanismEstimator::default()
                .gaussian_log_likelihood(&data, &with, &[0.0, 0.0], 1.0)
                .unwrap()
                .is_finite()
        );
    }

    #[test]
    fn no_selection_and_no_censoring_collapse_to_observed_values() {
        let values = [1.0, 2.0, 3.0, 4.0];
        let selected = [1.0; 4];
        let far_censor = [10.0; 4];
        let data = TabularData::from_f64_columns([
            ("a", values.as_slice()),
            ("y", values.as_slice()),
            ("r", selected.as_slice()),
            ("c", far_censor.as_slice()),
        ])
        .unwrap();
        let selected_query = response_query(VariableId::from_raw(1), VariableId::from_raw(0))
            .with_observation(
                ObservationSpec::Selected {
                    latent: VariableId::from_raw(1),
                    observed: VariableId::from_raw(1),
                    indicator: VariableId::from_raw(2),
                },
                [ObservationAssumption::OutcomeIndependentGiven(Arc::from([]))],
            );
        let right_query = response_query(VariableId::from_raw(1), VariableId::from_raw(0))
            .with_observation(
                ObservationSpec::RightCensored {
                    latent: VariableId::from_raw(1),
                    observed: VariableId::from_raw(1),
                    censoring: VariableId::from_raw(3),
                    event: VariableId::from_raw(2),
                },
                [ObservationAssumption::IndependentGiven(Arc::from([]))],
            );
        let estimator = ObservationMechanismEstimator::default();
        let selected_adjusted = estimator.adjusted_outcome(&data, &selected_query, None).unwrap();
        let right_adjusted = estimator.adjusted_outcome(&data, &right_query, None).unwrap();
        assert_eq!(selected_adjusted.values, values);
        assert_eq!(right_adjusted.values, values);
        assert_eq!(selected_adjusted.weights, vec![1.0; 4]);
        assert_eq!(right_adjusted.weights, vec![1.0; 4]);
        assert_eq!(selected_adjusted.method.as_ref(), "observation.selected.complete_collapse.v1");
        let n = 40;
        let large: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let ones = vec![1.0; n];
        let large_data = TabularData::from_f64_columns([
            ("a", large.as_slice()),
            ("y", large.as_slice()),
            ("r", ones.as_slice()),
        ])
        .unwrap();
        let collapse_query = ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: VariableId::from_raw(1),
            treatment: ContinuousDomain::new(
                VariableId::from_raw(0),
                GridSpec::Values(Arc::from([0.2, 0.8])),
            ),
        })
        .with_observation(
            ObservationSpec::Selected {
                latent: VariableId::from_raw(1),
                observed: VariableId::from_raw(1),
                indicator: VariableId::from_raw(2),
            },
            [ObservationAssumption::OutcomeIndependentGiven(Arc::from([VariableId::from_raw(0)]))],
        );
        let collapsed = ObservationMechanismEstimator::default()
            .estimate_mean_curve(
                &ContinuousResponseEstimator::new(Arc::from([])),
                &large_data,
                &collapse_query,
                None,
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
            )
            .unwrap();
        assert!(collapsed.support.warnings.iter().any(|warning| {
            warning.code.as_ref() == "response.observation_selected_complete_collapse"
        }));
    }

    #[test]
    fn selected_aipw_pseudo_values_match_observation_primitives_fixture() {
        let pin: serde_json::Value = serde_json::from_str(include_str!(
            "../../../conformance/response/observation_primitives/expected.json"
        ))
        .unwrap();
        let sel = &pin["selected_outcome"];
        let nums = |key: &str| -> Vec<f64> {
            sel[key].as_array().unwrap().iter().map(|v| v.as_f64().unwrap_or(f64::NAN)).collect()
        };
        let mu = nums("outcome_regression");
        let got = selected_outcome_pseudo_values(
            &nums("observed"),
            &nums("indicator"),
            &nums("probabilities"),
            Some(mu.as_slice()),
        )
        .unwrap();
        let expected = nums("expected_aipw_pseudo_values");
        let atol = pin["tolerance"]["atol"].as_f64().unwrap_or(1e-12);
        assert!(
            got.iter().zip(&expected).all(|(a, b)| (a - b).abs() <= atol),
            "ObservationMechanismEstimator primitives must consume observation_primitives, got {got:?} expected {expected:?}"
        );
    }

    /// Two-lag selected series: `Y_s = 1 + 2 T_{s-1} + 1.5 T_{s-2}` (no noise), selection
    /// on `T_{s-1}` only; unselected outcomes are recorded as 0.
    fn two_lag_selected_series(n: usize) -> TimeSeriesData {
        let t: Vec<f64> = (0..n).map(|i| ((i * 37 + 11) % 101) as f64 / 50.0 - 1.0).collect();
        let lag = |s: usize, l: usize| s.checked_sub(l).map_or(0.0, |i| t[i]);
        let r: Vec<f64> = (0..n).map(|s| f64::from(lag(s, 1) > -0.4 || (s * 13) % 7 < 3)).collect();
        let y: Vec<f64> = (0..n)
            .map(|s| if r[s] == 1.0 { 1.0 + 2.0 * lag(s, 1) + 1.5 * lag(s, 2) } else { 0.0 })
            .collect();
        TimeSeriesData::from_f64_columns(
            [("t", t.as_slice()), ("y", y.as_slice()), ("r", r.as_slice())],
            1,
        )
        .unwrap()
    }

    fn temporal_selected_query(horizons: Vec<u32>) -> ResponseQuery {
        response_query(VariableId::from_raw(1), VariableId::from_raw(0))
            .with_temporal(
                antecedent_core::TemporalResponseSpec::new(
                    horizons,
                    antecedent_core::TemporalPolicy::pulse(-1),
                    None,
                )
                .unwrap(),
            )
            .with_observation(
                ObservationSpec::Selected {
                    latent: VariableId::from_raw(1),
                    observed: VariableId::from_raw(1),
                    indicator: VariableId::from_raw(2),
                },
                [ObservationAssumption::OutcomeIndependentGiven(Arc::from([
                    VariableId::from_raw(0),
                ]))],
            )
    }

    #[test]
    fn selected_aipw_outcome_model_conditions_on_the_downstream_design_columns() {
        let data = two_lag_selected_series(240);
        let t = data.float64_values(VariableId::from_raw(0)).unwrap();
        let r = data.float64_values(VariableId::from_raw(2)).unwrap();
        let query = temporal_selected_query(vec![1]);
        let estimator = ObservationMechanismEstimator::default();
        let key = |offset| TemporalNodeKey { variable: VariableId::from_raw(0), offset };
        // An unselected row's pseudo-outcome is the outcome-model prediction. With T@-2 in
        // the outcome model it is the noiseless latent outcome; with the declared T@-1 alone
        // it cannot track T@-2.
        let (_, with_design) =
            estimator.adjust_temporal_series(&data, &query, &[], &[key(-1), key(-2)]).unwrap();
        let (_, declared_only) =
            estimator.adjust_temporal_series(&data, &query, &[], &[key(-1)]).unwrap();
        let mut worst_design = 0.0_f64;
        let mut worst_declared = 0.0_f64;
        for s in 2..t.len() {
            if r[s] == 0.0 {
                let latent = 1.0 + 2.0 * t[s - 1] + 1.5 * t[s - 2];
                worst_design = worst_design.max((with_design.values[s] - latent).abs());
                worst_declared = worst_declared.max((declared_only.values[s] - latent).abs());
            }
        }
        assert!(worst_design < 1e-8, "design-column outcome model misses: {worst_design}");
        assert!(worst_declared > 0.5, "declared-set outcome model cannot track T@-2");
        // Row 1 has no T@-2: it keeps the declared-set outcome model, never a NaN.
        assert!(with_design.values[1].is_finite());
        // The observation row now reaches two rows back; anchors must respect that.
        assert_eq!(estimator.observation_row_lag(&query, &[key(-1), key(-2)]).unwrap(), 2);
        assert_eq!(estimator.observation_row_lag(&query, &[key(-1)]).unwrap(), 1);
        let anchors: Vec<usize> = (1..t.len()).collect();
        assert!(
            estimator
                .adjust_temporal_anchors(&data, &query, &[], &[key(-1), key(-2)], &anchors)
                .is_err()
        );
        let anchors: Vec<usize> = (2..t.len()).collect();
        let replicate = estimator
            .adjust_temporal_anchors(&data, &query, &[], &[key(-1), key(-2)], &anchors)
            .unwrap();
        assert!(
            replicate.iter().zip(&with_design.values[2..]).all(|(a, b)| (a - b).abs() < 1e-8),
            "on noiseless rows the exact outcome model makes the replicate fold-invariant"
        );
    }

    /// Noisy one-lag selected series: `Y_s = 1 + 2 T_{s-1} + ε_s`, selection on `T_{s-1}`.
    fn noisy_selected_series(n: usize) -> TimeSeriesData {
        let t: Vec<f64> = (0..n).map(|i| ((i * 37 + 11) % 101) as f64 / 50.0 - 1.0).collect();
        let lag = |s: usize| s.checked_sub(1).map_or(0.0, |i| t[i]);
        let r: Vec<f64> = (0..n).map(|s| f64::from(lag(s) > -0.4 || (s * 13) % 7 < 3)).collect();
        let y: Vec<f64> = (0..n)
            .map(|s| {
                let noise = ((s * 53 + 7) % 97) as f64 / 48.0 - 1.0;
                if r[s] == 1.0 { 1.0 + 2.0 * lag(s) + noise } else { 0.0 }
            })
            .collect();
        TimeSeriesData::from_f64_columns(
            [("t", t.as_slice()), ("y", y.as_slice()), ("r", r.as_slice())],
            1,
        )
        .unwrap()
    }

    #[test]
    fn replicate_folds_follow_source_positions_so_duplicated_tuples_share_a_fold() {
        let n = 200;
        let data = noisy_selected_series(n);
        let query = temporal_selected_query(vec![1]);
        let estimator = ObservationMechanismEstimator::default();
        let (_, series) = estimator.adjust_temporal_series(&data, &query, &[], &[]).unwrap();
        // Every distinct anchor once, in order: the replicate uses the same time-block
        // folds as the full-series fit and reproduces it exactly.
        let anchors: Vec<usize> = (1..n).collect();
        let replicate =
            estimator.adjust_temporal_anchors(&data, &query, &[], &[], &anchors).unwrap();
        assert!(replicate.iter().zip(&series.values[1..]).all(|(a, b)| (a - b).abs() < 1e-10));
        // A block bootstrap that draws every tuple twice, adjacently: copies share a fold,
        // so both receive the same out-of-fold nuisances (a copy in the training fold
        // would make the two pseudo-values differ).
        let doubled: Vec<usize> = (1..n).flat_map(|anchor| [anchor, anchor]).collect();
        let replicate =
            estimator.adjust_temporal_anchors(&data, &query, &[], &[], &doubled).unwrap();
        for pair in replicate.chunks(2) {
            assert!((pair[0] - pair[1]).abs() < 1e-12, "duplicated tuple split across folds");
        }
        let folds = estimator.options.crossfit_folds;
        let positions: Vec<usize> = doubled.iter().map(|&anchor| anchor - 1).collect();
        let layout = FoldLayout::TimeBlocks { positions: &positions, span: n - 1 };
        let fold_of: Vec<usize> =
            (0..doubled.len()).map(|row| layout.fold_of(row, folds)).collect();
        assert!(fold_of.windows(2).all(|w| w[0] <= w[1]), "time-block folds are contiguous");
        assert_eq!(fold_of.last().copied(), Some(folds - 1));
    }

    #[test]
    fn sequence_regressors_flag_every_lag_at_which_the_outcome_feeds_its_ancestry() {
        // T@-1 → Y@0, Y@-2 → T@0 (lagged outcome into an ancestral mechanism), and
        // Y@-1 → Z@0 with Z a pure descendant (not in the outcome's design).
        let (t, y, z) = (VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(3));
        let mut graph = antecedent_graph::TemporalDag::empty();
        let lagged = |graph: &mut antecedent_graph::TemporalDag, v, lag| {
            antecedent_graph::ensure_lagged(graph, v, antecedent_core::Lag::from_raw(lag)).unwrap()
        };
        let (t1, t0, y0, y1, y2, z0) = (
            lagged(&mut graph, t, 1),
            lagged(&mut graph, t, 0),
            lagged(&mut graph, y, 0),
            lagged(&mut graph, y, 1),
            lagged(&mut graph, y, 2),
            lagged(&mut graph, z, 0),
        );
        graph.insert_directed(t1, y0).unwrap();
        graph.insert_directed(y2, t0).unwrap();
        graph.insert_directed(y1, z0).unwrap();
        let keys = temporal_sequence_outcome_regressors(&graph, y);
        let key = |variable, offset| TemporalNodeKey { variable, offset };
        assert_eq!(keys, vec![key(t, -1), key(y, -2)]);
        let query = temporal_selected_query(vec![1]);
        // A curve query is not a Sequence: the outcome key does not refuse it.
        assert!(require_unlagged_sequence_outcome(&query, y, &keys).is_ok());
    }

    #[test]
    fn downstream_design_columns_follow_the_curve_horizons_and_sequence_parents() {
        let mut graph = antecedent_graph::TemporalDag::empty();
        let y0 = antecedent_graph::ensure_lagged(
            &mut graph,
            VariableId::from_raw(1),
            antecedent_core::Lag::CONTEMPORANEOUS,
        )
        .unwrap();
        for lag in [1, 2] {
            let t = antecedent_graph::ensure_lagged(
                &mut graph,
                VariableId::from_raw(0),
                antecedent_core::Lag::from_raw(lag),
            )
            .unwrap();
            graph.insert_directed(t, y0).unwrap();
        }
        let key = |offset| TemporalNodeKey { variable: VariableId::from_raw(0), offset };
        assert_eq!(
            temporal_sequence_outcome_regressors(&graph, VariableId::from_raw(1)),
            vec![key(-2), key(-1)]
        );
        // Plain IPW and censoring never take extra outcome-model columns.
        let ipw = ObservationMechanismEstimator::new(ObservationEstimatorOptions {
            selected_correction: SelectedOutcomeCorrection::Ipw,
            ..ObservationEstimatorOptions::default()
        });
        let query = temporal_selected_query(vec![1]);
        assert_eq!(ipw.observation_row_lag(&query, &[key(-1), key(-2)]).unwrap(), 1);
    }
}
