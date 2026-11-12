//! Linear sequential g-computation on an identified unfolded DAG.
//!
//! Multi-step Sustained and multi-step / joint `Sequence` overlays share this
//! engine. Sustained is the active-minus-control contrast over a window;
//! Sequence evaluates the interventional level under per-node Set / Soft
//! mean overlays (constant, shift, multiplicative, and bounded shift).
//! Fitted stationary Bayesian mechanisms can be retained for predictive checks;
//! facade validation refits this same engine for each perturbed graph atom.
//! No second identifier.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

// Dense ids originate as u32; nonnegative row indices and bounded replicate counts are intentional.
#![cfg_attr(
    test,
    allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "test fixtures compare exact constants and index with small literals"
    )
)]

mod bayesian;
mod bootstrap;
mod design;
use bayesian::BayesianComposition;
use bootstrap::block_bootstrap_estimate;
use design::{extend_mechanism_design, weight_sequential_windows};

use std::sync::Arc;

use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSet, AssumptionSource,
    AssumptionStatus, CausalRng, ExecutionContext, IdentificationStatus, Lag, ParametricAssumption,
    TargetPopulation, TemporalEffectQuery, TemporalNodeKey, TemporalPolicy, VariableId,
};
use antecedent_data::{LaggedColumn, TemporalIndexer, TimeSeriesData};
use antecedent_expr::{EstimandMethod, IdentifiedEstimand};
use antecedent_graph::{DenseNodeId, TemporalDag};
use antecedent_model::history_operators::{HistoryScan, SequentialRegression};
use antecedent_prob::{BayesLikelihood, PosteriorDraws, PosteriorQuantityKind, PosteriorSchema};
use antecedent_stats::{
    CompiledDesign, DesignColumnMap, DesignColumnRole, LeastSquaresFit, LeastSquaresWorkspace,
    StatsError,
};

use crate::serial_dependence::{DependenceScope, SerialDependence};
use crate::{
    BayesianGCompWorkspace, BayesianGComputationAte, CausalPosterior, EffectEstimate,
    EstimationError, OverlapPolicy, PreparedBayesianProblem,
};

/// Stream salt of the root-mean shift draws, apart from every mechanism's own stream.
const ROOT_MEAN_STREAM: u64 = 0x2007_3A11_C0DE_51DE;

/// The observed series and the identified unfolded design every sequential estimator fits:
/// the temporal graph, its dense-node indexer and the identified unfolded estimand.
#[derive(Clone, Copy)]
pub struct SequentialProblem<'a> {
    /// The observed series.
    pub data: &'a TimeSeriesData,
    /// The temporal DAG the unfolded design comes from.
    pub graph: &'a TemporalDag,
    /// Dense-node to (variable, offset) indexer of the unfolded graph.
    pub indexer: &'a TemporalIndexer,
    /// The identified unfolded estimand.
    pub estimand: &'a IdentifiedEstimand,
}

/// The outcome node and per-node overlays of a sequence level.
///
/// `O` is [`SequentialMechanismOverlay`] or the plain [`SequentialNodeOverlay`].
#[derive(Clone, Copy)]
pub struct SequenceTarget<'a, O = SequentialMechanismOverlay> {
    /// Outcome template variable.
    pub outcome: VariableId,
    /// Absolute time offset of the outcome copy.
    pub outcome_offset: i32,
    /// The per-node overlays; no two may assign the same (variable, time).
    pub overlays: &'a [O],
}

/// How a sequential estimate is made uncertain and what it is published with: the
/// identification status and assumptions it carries, and whether a Frequentist block
/// bootstrap or a Bayesian composition supplies the interval.
#[derive(Clone)]
pub struct SequentialInference<'a> {
    /// Identification status; the design must be point identified.
    pub status: IdentificationStatus,
    /// Assumptions the identification requires, carried onto the estimate.
    pub assumptions: AssumptionSet,
    /// Frequentist circular-block bootstrap replicates (ignored when `bayesian` is set).
    pub bootstrap_replicates: u32,
    /// Bayesian composition of independent Gaussian mechanism posteriors, when requested.
    pub bayesian: Option<&'a BayesianGComputationAte>,
}

/// One intervened unfolded node and the supported overlay applied there.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SequentialNodeOverlay {
    /// Template variable.
    pub variable: VariableId,
    /// Absolute time offset of the intervened copy.
    pub offset: i32,
    /// Hard `Set` / Soft `constant` level. `None` means an additive shift.
    pub level: Option<f64>,
    /// Additive shift applied to the factual mean when [`Self::level`] is `None`.
    pub shift: f64,
}

impl SequentialNodeOverlay {
    /// Assigned value under a linear additive mechanism.
    #[must_use]
    pub fn assigned(self, factual_mean: f64) -> f64 {
        self.level.unwrap_or(factual_mean + self.shift)
    }
}

/// Structural multiplication or population-mean-targeting shift on an unfolded node.
///
/// Applies `clamp(multiplier * mean + shift, lower, upper)` to the propagated
/// mean. A bounded shift replaces `f` by `f + clamp(mu + shift, lower, upper) - mu`,
/// where `mu = E[f]` under preceding interventions. Parent effects and innovations
/// remain; individual outcomes need not satisfy the bounds. Multiplication uses
/// `multiplier * f`, whose expectation is propagated exactly by the linear engine.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SequentialMechanismOverlay {
    /// Existing Set/Shift coordinates and parameters.
    pub node: SequentialNodeOverlay,
    /// Propagated-mean multiplier.
    pub multiplier: f64,
    /// Optional inclusive bounds on the shifted propagated mean.
    pub bounds: Option<(f64, f64)>,
}

impl From<SequentialNodeOverlay> for SequentialMechanismOverlay {
    fn from(node: SequentialNodeOverlay) -> Self {
        Self { node, multiplier: 1.0, bounds: None }
    }
}

impl SequentialMechanismOverlay {
    /// Record the population-dependent mechanism defining a bounded mean shift.
    #[must_use]
    pub fn mean_target_assumption() -> antecedent_core::AssumptionRecord {
        antecedent_core::AssumptionRecord {
            assumption: antecedent_core::Assumption::ParametricRestriction(
                antecedent_core::ParametricAssumption {
                    id: "temporal.soft.population_mean_target".into(),
                    description: "truncated_shift defines f_policy = f + clip(mu + delta, lower, upper) - mu, \
                        where mu = E[f] under preceding interventions; parent effects and innovations \
                        remain, and realized outcomes need not satisfy the bounds. This is a \
                        population-mean-targeting policy, not stochastic clipping. Multiplicative \
                        defines factor * f; linear downstream mechanisms propagate these means exactly".into(),
                },
            ),
            source: antecedent_core::AssumptionSource::AlgorithmDefault {
                algorithm: "temporal.sequential.gcomp".into(),
            },
            scope: antecedent_core::AssumptionScope::Estimation,
            status: antecedent_core::AssumptionStatus::Declared,
        }
    }

    /// Apply the deterministic propagated-mean mechanism.
    #[must_use]
    pub fn assigned(self, mean: f64) -> f64 {
        let value = self.node.level.unwrap_or(self.multiplier * mean + self.node.shift);
        self.bounds.map_or(value, |(lower, upper)| value.clamp(lower, upper))
    }
}

/// Natural mean of a fitted linear mechanism: `β₀ + Σ_k β_{k+1} · values[parent_k]`.
pub(crate) fn linear_natural(coefficients: &[f64], parents: &[usize], values: &[f64]) -> f64 {
    coefficients[0]
        + parents
            .iter()
            .enumerate()
            .map(|(k, &node)| coefficients[k + 1] * values[node])
            .sum::<f64>()
}

/// The g-formula level of `outcome` on an unfolded linear SEM: walk `order` (a topological
/// order of the needed nodes) and give each node the overlay-assigned value of its natural
/// mean. A hard `Set` overwrites the node without reading its mechanism; every other node
/// asks `natural(node, values_so_far)`. The one owner of overlay application for the
/// propagated level, shared by the sequential engine, the response tuples and the
/// observed-data posterior.
pub(crate) fn propagate_linear_level(
    order: &[usize],
    node_count: usize,
    outcome: usize,
    overlay_at: impl Fn(usize) -> Option<SequentialMechanismOverlay>,
    natural: impl Fn(usize, &[f64]) -> f64,
) -> f64 {
    let mut values = vec![0.0; node_count];
    for &i in order {
        let overlay = overlay_at(i);
        if let Some(hard) = overlay.filter(|overlay| overlay.node.level.is_some()) {
            values[i] = hard.assigned(0.0);
            continue;
        }
        let mean = natural(i, &values);
        values[i] = overlay.map_or(mean, |overlay| overlay.assigned(mean));
    }
    values[outcome]
}

/// What the sequential engine returns for one outcome node.
#[derive(Clone, Copy, Debug)]
enum SequentialEval {
    /// Active-minus-control contrast; every intervened node is overwritten by `delta`.
    Contrast { delta: f64 },
    /// Interventional outcome level under the per-node overlays.
    Level,
}

/// Fit every non-intervened ancestor's linear mechanism in the identified
/// unfolded DAG. Propagate the active-minus-control difference in topological
/// order, overwriting **every** treatment-time node in the sustained window.
/// Frequentist uncertainty is a circular-block bootstrap over consecutive
/// lag-aligned rows, refitting every equation on the same rows, with the
/// replicate SD scaled by the circular-Bartlett fixed-b factor
/// ([`crate::temporal_block::circular_fixed_b_scale`]). Bayesian uncertainty uses independent Gaussian priors across
/// stationary mechanisms, sharing each coefficient draw across its time copies;
/// each mechanism's Gaussian likelihood is tempered by its long-run-variance
/// ratio ([`crate::serial_dependence`]), so the composed interval is a
/// generalized posterior with a serial-dependence correction.
pub fn estimate_sustained_window(
    problem: &SequentialProblem<'_>,
    query: &TemporalEffectQuery,
    inference: SequentialInference<'_>,
    ctx: &ExecutionContext,
) -> Result<(EffectEstimate, Option<CausalPosterior>), EstimationError> {
    estimate_sustained_window_with_validation(problem, query, inference, ctx, None)
}

/// A Frequentist multi-step Sustained contrast on a series that may have missing time points.
///
/// Windows that read a missing value are dropped as a whole (every mechanism is fitted on the
/// intact windows, each at its own time with its own lags); the block bootstrap draws circular
/// blocks on the original window axis, so no block joins windows across a gap, and applies no
/// kernel-bias widening. A Bayesian composition tempers each mechanism's likelihood by the
/// segment-wise long-run variance of its own intact windows
/// ([`crate::SerialDependence::SegmentedTempering`]).
///
/// # Errors
///
/// As [`estimate_sustained_window`]; a Bayesian inference; fewer than three intact windows.
pub fn estimate_sustained_window_with_gaps(
    problem: &SequentialProblem<'_>,
    query: &TemporalEffectQuery,
    inference: SequentialInference<'_>,
    ctx: &ExecutionContext,
    validation: Option<&mut Vec<SequentialBayesianMechanism>>,
) -> Result<(EffectEstimate, Option<CausalPosterior>, Option<crate::GapSelection>), EstimationError>
{
    estimate_sustained_window_gaps_if(problem, query, inference, ctx, validation, true)
}

/// [`estimate_sustained_window_with_gaps`] that admits gaps only when `gaps` is set; without it
/// a series with a missing value refuses as [`estimate_sustained_window`] does.
///
/// # Errors
///
/// As [`estimate_sustained_window_with_gaps`].
pub fn estimate_sustained_window_gaps_if(
    problem: &SequentialProblem<'_>,
    query: &TemporalEffectQuery,
    inference: SequentialInference<'_>,
    ctx: &ExecutionContext,
    validation: Option<&mut Vec<SequentialBayesianMechanism>>,
    gaps: bool,
) -> Result<(EffectEstimate, Option<CausalPosterior>, Option<crate::GapSelection>), EstimationError>
{
    let mut budget = antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
    estimate_sustained_window_gaps_if_within(
        problem,
        query,
        inference,
        ctx,
        validation,
        gaps,
        &mut budget,
    )
}

/// Fit one sustained window using an operation-wide temporal budget.
pub fn estimate_sustained_window_gaps_if_within(
    problem: &SequentialProblem<'_>,
    query: &TemporalEffectQuery,
    inference: SequentialInference<'_>,
    ctx: &ExecutionContext,
    validation: Option<&mut Vec<SequentialBayesianMechanism>>,
    gaps: bool,
    budget: &mut antecedent_core::TemporalBudget,
) -> Result<(EffectEstimate, Option<CausalPosterior>, Option<crate::GapSelection>), EstimationError>
{
    let (overlays, delta) =
        sustained_contrast_overlays(query, problem.estimand, inference.status, budget, ctx)?;
    if inference.bayesian.is_some_and(|est| est.likelihood != BayesLikelihood::GaussianIdentity) {
        return Err(EstimationError::unsupported_coded(
            antecedent_core::reason_code!("route_not_supported"),
            "sequential Bayesian g-computation requires GaussianIdentity",
        ));
    }
    if inference.bayesian.is_some_and(|est| est.backend == crate::BayesianBackendKind::Hmc) {
        return Err(EstimationError::unsupported_coded(
            antecedent_core::reason_code!("estimator_inference_mismatch"),
            "composed sustained HMC requires derived-contrast chain diagnostics; use conjugate or Laplace",
        ));
    }
    estimate_sequential_with_gaps(
        problem,
        &SequenceTarget {
            outcome: query.outcome,
            outcome_offset: query.outcome_offset()?,
            overlays: &overlays,
        },
        SequentialEval::Contrast { delta },
        inference,
        ctx,
        query.effect_modifiers.as_ref(),
        validation,
        gaps,
        budget,
    )
}

/// A fitted stationary mechanism, retained for predictive validation of the actual SEM.
#[derive(Clone, Debug)]
pub struct SequentialBayesianMechanism {
    /// Child variable shared by the unfolded time copies.
    pub variable: VariableId,
    /// Unique observed rows used by this likelihood.
    pub prepared: PreparedBayesianProblem,
    /// Posterior over this mechanism's coefficients and noise scale.
    pub posterior: CausalPosterior,
}

/// Fit a sustained policy and optionally retain its actual stationary Bayesian mechanisms.
pub fn estimate_sustained_window_with_validation(
    problem: &SequentialProblem<'_>,
    query: &TemporalEffectQuery,
    inference: SequentialInference<'_>,
    ctx: &ExecutionContext,
    validation: Option<&mut Vec<SequentialBayesianMechanism>>,
) -> Result<(EffectEstimate, Option<CausalPosterior>), EstimationError> {
    let mut budget = antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
    let (overlays, delta) =
        sustained_contrast_overlays(query, problem.estimand, inference.status, &mut budget, ctx)?;
    let bayesian = inference.bayesian;
    if bayesian.is_some_and(|est| est.likelihood != BayesLikelihood::GaussianIdentity) {
        return Err(EstimationError::unsupported_coded(
            antecedent_core::reason_code!("route_not_supported"),
            "sequential Bayesian g-computation requires GaussianIdentity",
        ));
    }
    if bayesian.is_some_and(|est| est.backend == crate::BayesianBackendKind::Hmc) {
        return Err(EstimationError::unsupported_coded(
            antecedent_core::reason_code!("estimator_inference_mismatch"),
            "composed sustained HMC requires derived-contrast chain diagnostics; use conjugate or Laplace",
        ));
    }
    estimate_sequential(
        problem,
        &SequenceTarget {
            outcome: query.outcome,
            outcome_offset: query.outcome_offset()?,
            overlays: &overlays,
        },
        SequentialEval::Contrast { delta },
        inference,
        ctx,
        query.effect_modifiers.as_ref(),
        validation,
        &mut budget,
    )
}

struct SustainedOverlays {
    values: Vec<SequentialMechanismOverlay>,
    _lease: antecedent_core::execution::AdmissionLease,
}
impl std::ops::Deref for SustainedOverlays {
    type Target = [SequentialMechanismOverlay];
    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

/// Validate a multi-step Sustained contrast and build its per-time overlays and
/// the active-minus-control `delta`.
fn sustained_contrast_overlays(
    query: &TemporalEffectQuery,
    estimand: &IdentifiedEstimand,
    status: IdentificationStatus,
    budget: &mut antecedent_core::TemporalBudget,
    ctx: &ExecutionContext,
) -> Result<(SustainedOverlays, f64), EstimationError> {
    if ctx.cancellation.is_cancelled() {
        return Err(antecedent_core::TemporalMaterializationError::Cancelled.into());
    }
    budget.charge_work(query.effect_modifiers.len() as u64)?;
    let validation = antecedent_core::admit_temporal_intervention_validation(
        [&query.control, &query.active],
        budget,
        ctx,
    )?;
    query.validate()?;
    drop(validation);
    let TemporalPolicy::Sustained { from, until } = query.policy else {
        return Err(EstimationError::unsupported_coded(
            antecedent_core::reason_code!("route_not_supported"),
            "sequential estimator requires a sustained window",
        ));
    };
    if from >= until
        || query.target_population != TargetPopulation::AllObserved
        || estimand.method_kind().ok() != Some(EstimandMethod::TemporalBackdoorUnfolded)
        || !matches!(
            status,
            IdentificationStatus::NonparametricallyIdentified
                | IdentificationStatus::IdentifiedUnderParametricRestrictions
        )
    {
        return Err(EstimationError::unsupported(
            "sequential estimator requires an identified unfolded multi-step contrast and AllObserved",
        ));
    }
    let active = crate::adjustment::intervention_f64(&query.active)?;
    let delta = active - crate::adjustment::intervention_f64(&query.control)?;
    let count = query.policy.offset_count()?;
    budget.charge_offsets(count)?;
    budget.charge_work(count)?;
    let bytes = count
        .checked_mul(256)
        .and_then(|n| n.checked_add(512))
        .ok_or(antecedent_core::TemporalMaterializationError::Limit { dimension: "bytes" })?;
    if bytes > budget.limits().max_bytes {
        return Err(
            antecedent_core::TemporalMaterializationError::Limit { dimension: "bytes" }.into()
        );
    }
    let lease = ctx.acquire_admission(0, bytes).map_err(|e| match e {
        antecedent_core::execution::AdmissionError::Cancelled => {
            antecedent_core::TemporalMaterializationError::Cancelled
        }
        other => antecedent_core::TemporalMaterializationError::Admission(other),
    })?;
    let mut overlays = Vec::new();
    overlays
        .try_reserve_exact(
            usize::try_from(count)
                .map_err(|_| antecedent_core::TemporalMaterializationError::Allocation)?,
        )
        .map_err(|_| antecedent_core::TemporalMaterializationError::Allocation)?;
    for offset in from..=until {
        if ctx.cancellation.is_cancelled() {
            return Err(antecedent_core::TemporalMaterializationError::Cancelled.into());
        }
        overlays.push(SequentialMechanismOverlay::from(SequentialNodeOverlay {
            variable: query.treatment,
            offset,
            level: Some(active),
            shift: 0.0,
        }));
    }
    Ok((SustainedOverlays { values: overlays, _lease: lease }, delta))
}

/// A Frequentist multi-step Sustained contrast prepared once for repeated OLS
/// evaluation on lag-aligned rows of the original series.
///
/// Multi-design shared bootstraps (structural mixtures over temporal classes or
/// DBN posteriors, and the `bootstrap.ci_coverage` refuter) resample *series
/// times*, not raw rows: [`Self::estimate_on_rows`] refits every mechanism on the
/// selected aligned rows, each of which keeps its intact lag window.
#[derive(Debug)]
pub struct SequentialContrastDesign {
    setup: SequentialSetup,
}

impl SequentialContrastDesign {
    /// Prepare the identified multi-step Sustained contrast on `data`.
    ///
    /// # Errors
    ///
    /// The same refusals as [`estimate_sustained_window`], or lag-alignment failures.
    pub fn prepare(
        problem: &SequentialProblem<'_>,
        query: &TemporalEffectQuery,
        status: IdentificationStatus,
        ctx: &ExecutionContext,
    ) -> Result<Self, EstimationError> {
        Self::prepare_with_gaps(problem, query, status, ctx, false)
    }

    /// [`Self::prepare`], on a series that may have missing time points when `gaps` is set.
    ///
    /// The design holds the intact windows in time order; [`Self::gapped_rows`] says which
    /// windows those are, so a shared bootstrap resamples blocks of the original window axis.
    ///
    /// # Errors
    ///
    /// As [`Self::prepare`].
    pub fn prepare_with_gaps(
        problem: &SequentialProblem<'_>,
        query: &TemporalEffectQuery,
        status: IdentificationStatus,
        ctx: &ExecutionContext,
        gaps: bool,
    ) -> Result<Self, EstimationError> {
        let mut budget =
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
        let (overlays, delta) =
            sustained_contrast_overlays(query, problem.estimand, status, &mut budget, ctx)?;
        let setup = SequentialSetup::build_with_gaps(
            problem,
            &SequenceTarget {
                outcome: query.outcome,
                outcome_offset: query.outcome_offset()?,
                overlays: &overlays,
            },
            SequentialEval::Contrast { delta },
            query.effect_modifiers.as_ref(),
            ctx,
            gaps,
            &mut budget,
        )?;
        if setup.sampling_weights.is_some() {
            return Err(crate::sampling_weights::unsupported_route(
                "a shared bootstrap over a prepared sustained design refits unweighted windows; \
                 sampling weights are read by the direct frequentist sustained contrast",
            ));
        }
        Ok(Self { setup })
    }

    /// Series times of the aligned rows (row `r` is time `first_time + r`).
    #[must_use]
    pub fn aligned_rows(&self) -> crate::temporal_block::AlignedRows {
        crate::temporal_block::AlignedRows {
            first_time: self.setup.max_lag as usize,
            rows: self.setup.n,
        }
    }

    /// Whether any window of the series was dropped for a missing value. A design with
    /// gaps resamples through [`Self::gapped_rows`], not [`Self::aligned_rows`].
    #[must_use]
    pub fn has_gaps(&self) -> bool {
        self.setup.window_complete.as_ref().is_some_and(|c| c.iter().any(|intact| !intact))
    }

    /// The windows of the series this design reads and which of them are intact; its rows are
    /// the intact windows in time order.
    #[must_use]
    pub fn gapped_rows(&self) -> crate::temporal_block::GappedAlignedRows {
        match &self.setup.window_complete {
            Some(complete) => crate::temporal_block::GappedAlignedRows {
                first_time: self.setup.max_lag as usize,
                complete: complete.clone(),
            },
            None => crate::temporal_block::GappedAlignedRows::contiguous(self.aligned_rows()),
        }
    }

    /// The dropped series times, when the design has gaps.
    #[must_use]
    pub fn gap_selection(&self) -> Option<crate::GapSelection> {
        self.has_gaps().then(|| {
            let rows = self.gapped_rows();
            crate::GapSelection { first_time: rows.first_time, complete: rows.complete }
        })
    }

    /// Unfolded window span in series times (`max_lag + 1`).
    #[must_use]
    pub fn structural_span(&self) -> usize {
        self.setup.max_lag as usize + 1
    }

    /// Point contrast on every aligned row.
    ///
    /// # Errors
    ///
    /// Least-squares failures.
    pub fn estimate(&self) -> Result<f64, EstimationError> {
        self.setup.evaluate(None, &mut LeastSquaresWorkspace::default())
    }

    /// Contrast refit on the aligned rows `rows` (any length; indices `< rows()`).
    ///
    /// # Errors
    ///
    /// Out-of-range rows or least-squares failures.
    pub fn estimate_on_rows(&self, rows: &[usize]) -> Result<f64, EstimationError> {
        self.estimate_on_rows_into(
            rows,
            &mut LeastSquaresWorkspace::default(),
            &mut Vec::new(),
            &mut Vec::new(),
        )
    }

    /// [`Self::estimate_on_rows`] writing each mechanism gather into caller-owned buffers.
    ///
    /// # Errors
    ///
    /// Out-of-range rows or least-squares failures.
    pub fn estimate_on_rows_into(
        &self,
        rows: &[usize],
        ls_ws: &mut LeastSquaresWorkspace,
        x_boot: &mut Vec<f64>,
        y_boot: &mut Vec<f64>,
    ) -> Result<f64, EstimationError> {
        if rows.iter().any(|&r| r >= self.setup.n) {
            return Err(EstimationError::data_msg("aligned row index out of range"));
        }
        self.setup.evaluate_into(Some(rows), ls_ws, x_boot, y_boot)
    }

    /// Per-row influence of the contrast (OLS scores of every mechanism mapped
    /// through the contrast gradient), for [`crate::temporal_block::effective_rows`].
    #[must_use]
    pub fn influence(&self) -> Option<Vec<f64>> {
        let full = self.setup.fit_full(&mut LeastSquaresWorkspace::default());
        self.setup.influence_from(&self.setup.coefficients_of(&full).ok()?)
    }

    /// Every mechanism's OLS normal-equation scores on the aligned rows
    /// ([`crate::temporal_block::normal_equation_scores`]), for
    /// [`crate::temporal_block::dependence_block_length`].
    #[must_use]
    pub fn normal_equation_scores(&self) -> Vec<Vec<f64>> {
        let full = self.setup.fit_full(&mut LeastSquaresWorkspace::default());
        self.setup.normal_equation_scores_of(&full)
    }

    /// Circular-block length of this contrast's Frequentist block-bootstrap
    /// interval ([`estimate_sustained_window`] with replicates), so a check of that
    /// interval resamples the same blocks.
    #[must_use]
    pub fn block_length(&self) -> usize {
        let full = self.setup.fit_full(&mut LeastSquaresWorkspace::default());
        let influence = self
            .setup
            .coefficients_of(&full)
            .ok()
            .and_then(|base| self.setup.influence_from(&base));
        self.setup.dependence_block_length_with(influence.as_deref(), &full)
    }
}

/// Sequential g-computation of the interventional **level** under per-node
/// Set / Soft constant / Soft shift overlays. Shares the Sustained engine.
///
/// # Errors
///
/// Empty or duplicate overlays, unidentified estimand, or fit failures.
pub fn estimate_sequence_overlays(
    problem: &SequentialProblem<'_>,
    target: &SequenceTarget<'_, SequentialNodeOverlay>,
    inference: SequentialInference<'_>,
    ctx: &ExecutionContext,
) -> Result<(EffectEstimate, Option<CausalPosterior>), EstimationError> {
    let mechanisms =
        target.overlays.iter().copied().map(SequentialMechanismOverlay::from).collect::<Vec<_>>();
    estimate_sequence_mechanisms(
        problem,
        &SequenceTarget {
            outcome: target.outcome,
            outcome_offset: target.outcome_offset,
            overlays: &mechanisms,
        },
        inference,
        ctx,
    )
}

/// Estimate a sequence of deterministic propagated-mean mechanism interventions.
/// Multiplication scales the structural assignment; bounded shifts target its population mean.
///
/// # Errors
/// Invalid parameters, duplicate coordinates, unidentified design, or fit failures.
pub fn estimate_sequence_mechanisms(
    problem: &SequentialProblem<'_>,
    target: &SequenceTarget<'_>,
    mut inference: SequentialInference<'_>,
    ctx: &ExecutionContext,
) -> Result<(EffectEstimate, Option<CausalPosterior>), EstimationError> {
    let overlays = target.overlays;
    let (estimand, status, bayesian) = (problem.estimand, inference.status, inference.bayesian);
    let assumptions = &mut inference.assumptions;
    if overlays.iter().any(|overlay| overlay.bounds.is_some()) {
        let restriction = SequentialMechanismOverlay::mean_target_assumption();
        if !assumptions.entries.contains(&restriction) {
            assumptions.push(restriction);
        }
    }
    if overlays.is_empty() {
        return Err(EstimationError::unsupported("sequential overlay requires at least one step"));
    }
    if overlays.iter().any(|overlay| {
        overlay.node.level.is_some_and(|level| !level.is_finite())
            || !overlay.node.shift.is_finite()
            || !overlay.multiplier.is_finite()
            || overlay.bounds.is_some_and(|(lo, hi)| !lo.is_finite() || !hi.is_finite() || lo > hi)
    }) {
        return Err(EstimationError::unsupported(
            "sequential overlays require finite levels and shifts",
        ));
    }
    if estimand.method_kind().ok() != Some(EstimandMethod::TemporalBackdoorUnfolded)
        || !matches!(
            status,
            IdentificationStatus::NonparametricallyIdentified
                | IdentificationStatus::IdentifiedUnderParametricRestrictions
        )
    {
        return Err(EstimationError::unsupported(
            "sequential overlay requires an identified unfolded design",
        ));
    }
    if bayesian.is_some_and(|est| est.likelihood != BayesLikelihood::GaussianIdentity) {
        return Err(EstimationError::unsupported_coded(
            antecedent_core::reason_code!("route_not_supported"),
            "sequential Bayesian g-computation requires GaussianIdentity",
        ));
    }
    if bayesian.is_some_and(|est| est.backend == crate::BayesianBackendKind::Hmc) {
        return Err(EstimationError::unsupported_coded(
            antecedent_core::reason_code!("estimator_inference_mismatch"),
            "composed sequence HMC requires derived-contrast chain diagnostics; use conjugate or Laplace",
        ));
    }
    let mut budget = antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
    estimate_sequential(
        problem,
        target,
        SequentialEval::Level,
        inference,
        ctx,
        &[],
        None,
        &mut budget,
    )
}

/// RNG stream base for the sequential circular-block row bootstrap.
const SEQUENTIAL_BLOCK_STREAM: u64 = 0x5120_0000;

/// The identified unfolded design as fitted mechanisms over one set of
/// lag-aligned rows: every mechanism's OLS design reads the same rows, row `r`
/// being the unfolded window that ends at series time `max_lag + r`.
#[derive(Debug)]
struct SequentialSetup {
    // Admission for graph-shaped setup buffers retained by this design.
    _temporal_lease: antecedent_core::execution::AdmissionLease,
    node_count: usize,
    order: Vec<usize>,
    parents: Vec<Vec<usize>>,
    designs: Vec<Option<CompiledDesign>>,
    intervention: Vec<bool>,
    overlay_at: Vec<Option<SequentialMechanismOverlay>>,
    /// Aligned values of needed nodes without a fitted mechanism (their factual
    /// mean enters `Level` propagation).
    free_columns: Vec<Option<Vec<f64>>>,
    factual: Vec<f64>,
    outcome: usize,
    n: usize,
    max_lag: u32,
    eval: SequentialEval,
    /// Named modifiers. Empty leaves every mechanism the parent regression it was.
    modifiers: Vec<VariableId>,
    /// Coefficient slots of `parent × modifier`, aligned with `parents`.
    /// `interaction_at[node][parent]` is one slot per modifier, in modifier order.
    interaction_at: Vec<Vec<Vec<usize>>>,
    /// Modifier columns at each fitted node's lag, used to recompute `E[W]` on a row map.
    node_modifier_values: Vec<Vec<Vec<f64>>>,
    /// `Some(complete)` when the series has gaps and the rows are its intact windows only:
    /// `complete[w]` says whether the window at position `w` of the original axis is one.
    window_complete: Option<Vec<bool>>,
    /// What declared sampling weights did to the windows; every mechanism's design is then
    /// the root-weighted design of its windows, so each solve is weighted least squares.
    sampling_weights: Option<crate::sampling_weights::SamplingWeightRecord>,
}

/// Full-sample OLS solves of a [`SequentialSetup`], one per fitted mechanism in fitting order.
struct FullFit {
    fits: Vec<(usize, Result<LeastSquaresFit, StatsError>)>,
}

impl SequentialSetup {
    #[cfg(test)]
    fn build(
        problem: &SequentialProblem<'_>,
        target: &SequenceTarget<'_>,
        eval: SequentialEval,
        modifiers: &[VariableId],
        ctx: &ExecutionContext,
    ) -> Result<Self, EstimationError> {
        let mut budget =
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
        Self::build_with_gaps(problem, target, eval, modifiers, ctx, false, &mut budget)
    }

    #[allow(
        clippy::cast_possible_truncation,
        reason = "dense node ids and row indices are u32 by construction, so every index below the graph's node count fits"
    )]
    #[allow(
        clippy::too_many_lines,
        reason = "resolve the unfolded graph, order the needed nodes, plan the lagged columns and gather the per-mechanism designs; every step reads the same node order"
    )]
    fn build_with_gaps(
        problem: &SequentialProblem<'_>,
        target: &SequenceTarget<'_>,
        eval: SequentialEval,
        modifiers: &[VariableId],
        ctx: &ExecutionContext,
        gaps: bool,
        budget: &mut antecedent_core::TemporalBudget,
    ) -> Result<Self, EstimationError> {
        let SequentialProblem { data, graph, indexer, .. } = *problem;
        let SequenceTarget { outcome, outcome_offset, overlays } = *target;
        let modifiers =
            if matches!(eval, SequentialEval::Contrast { .. }) { modifiers } else { &[] };
        let unfolded = graph.unfold_within(indexer.clone(), budget, ctx)?;
        let dag = &unfolded.dag;
        let node_count = dag.node_count();
        let mut edges = 0u64;
        budget.charge_work(node_count as u64)?;
        for i in 0..node_count {
            if ctx.cancellation.is_cancelled() {
                return Err(antecedent_core::TemporalMaterializationError::Cancelled.into());
            }
            edges = edges.checked_add(dag.parents(DenseNodeId::from_index(i)).len() as u64).ok_or(
                antecedent_core::TemporalMaterializationError::Limit { dimension: "work" },
            )?;
        }
        let work = (node_count as u64)
            .checked_mul(32)
            .and_then(|n| edges.checked_mul(128).and_then(|e| n.checked_add(e)))
            .ok_or(antecedent_core::TemporalMaterializationError::Limit { dimension: "work" })?;
        budget.charge_work(work)?;
        let bytes = (node_count as u64)
            .checked_mul(1024)
            .and_then(|n| edges.checked_mul(512).and_then(|e| n.checked_add(e)))
            .and_then(|n| n.checked_add(4096))
            .ok_or(antecedent_core::TemporalMaterializationError::Limit { dimension: "bytes" })?;
        if bytes > budget.limits().max_bytes {
            return Err(antecedent_core::TemporalMaterializationError::Limit {
                dimension: "bytes",
            }
            .into());
        }
        let temporal_lease = ctx.acquire_admission(0, bytes).map_err(|error| match error {
            antecedent_core::execution::AdmissionError::Cancelled => {
                antecedent_core::TemporalMaterializationError::Cancelled
            }
            other => antecedent_core::TemporalMaterializationError::Admission(other),
        })?;
        let outcome = indexer
            .dense_id(TemporalNodeKey { variable: outcome, offset: outcome_offset })?
            as usize;
        let mut intervention = vec![false; node_count];
        let mut overlay_at = vec![None; node_count];
        for &overlay in overlays {
            if ctx.cancellation.is_cancelled() {
                return Err(antecedent_core::TemporalMaterializationError::Cancelled.into());
            }
            let dense = indexer.dense_id(TemporalNodeKey {
                variable: overlay.node.variable,
                offset: overlay.node.offset,
            })? as usize;
            if intervention[dense] {
                return Err(EstimationError::unsupported(
                    "Sequence assigns the same (variable, time) twice; refuse rather than collapse",
                ));
            }
            intervention[dense] = true;
            overlay_at[dense] = Some(overlay);
        }
        let mut needed = vec![false; node_count];
        let mut pending = vec![outcome];
        while let Some(i) = pending.pop() {
            if ctx.cancellation.is_cancelled() {
                return Err(antecedent_core::TemporalMaterializationError::Cancelled.into());
            }
            if needed[i] {
                continue;
            }
            needed[i] = true;
            // A hard Set/constant cuts incoming edges. An additive shift replaces
            // f_i(pa_i, e_i) with f_i(pa_i, e_i) + delta, so its parents and
            // fitted mechanism remain part of the g-formula.
            let hard_intervention =
                overlay_at[i].is_some_and(|overlay| overlay.node.level.is_some());
            if !hard_intervention {
                pending.extend(
                    dag.parents(DenseNodeId::from_raw(i as u32)).iter().map(|p| p.as_usize()),
                );
            }
        }
        let order: Vec<_> = dag
            .topological_order()
            .ok_or_else(|| EstimationError::data_msg("unfolded graph is not acyclic"))?
            .into_iter()
            .map(DenseNodeId::as_usize)
            .filter(|&i| needed[i])
            .collect();
        let anchor = order
            .iter()
            .map(|&i| indexer.key_of(i as u32).map(|k| k.offset))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .max()
            .unwrap_or(0)
            .max(0);
        let mut columns = Vec::new();
        let mut column_of = vec![0; node_count];
        let mut max_lag = 0;
        for &i in &order {
            let key = indexer.key_of(i as u32)?;
            let lag = u32::try_from(anchor - key.offset)
                .map_err(|_| EstimationError::unsupported("invalid unfolded time offset"))?;
            max_lag = max_lag.max(lag);
            column_of[i] = columns.len();
            columns.push(LaggedColumn { variable: key.variable, lag: Lag::from_raw(lag) });
        }
        let mut modifier_at: Vec<(VariableId, u32, usize)> = Vec::new();
        if !modifiers.is_empty() {
            for &i in &order {
                let hard = overlay_at[i].is_some_and(|overlay| overlay.node.level.is_some());
                let parent_ids: Vec<usize> = dag
                    .parents(DenseNodeId::from_raw(i as u32))
                    .iter()
                    .map(|parent| parent.as_usize())
                    .collect();
                if hard || parent_ids.is_empty() || !parent_ids.iter().any(|&p| intervention[p]) {
                    continue;
                }
                let key = indexer.key_of(i as u32)?;
                let lag = u32::try_from(anchor - key.offset)
                    .map_err(|_| EstimationError::unsupported("invalid unfolded time offset"))?;
                max_lag = max_lag.max(lag);
                for &modifier in modifiers {
                    if modifier_at
                        .iter()
                        .any(|(variable, stored, _)| *variable == modifier && *stored == lag)
                    {
                        continue;
                    }
                    modifier_at.push((modifier, lag, columns.len()));
                    columns.push(LaggedColumn { variable: modifier, lag: Lag::from_raw(lag) });
                }
            }
        }
        let mut scan = HistoryScan::plan(data, max_lag, Arc::from(columns.clone()))?;
        let (sample, window_complete) = scan.windows(data, &ctx.kernel_policy, gaps)?;
        let n = sample.n;
        if n < 3 {
            return Err(EstimationError::unsupported(
                "sustained window requires at least three complete aligned rows",
            ));
        }
        let mut designs = vec![None; node_count];
        let mut parents = vec![Vec::new(); node_count];
        let mut free_columns = vec![None; node_count];
        let mut interaction_at = vec![Vec::new(); node_count];
        let mut node_modifier_values = vec![Vec::new(); node_count];
        for &i in &order {
            if ctx.cancellation.is_cancelled() {
                return Err(antecedent_core::TemporalMaterializationError::Cancelled.into());
            }
            parents[i] =
                dag.parents(DenseNodeId::from_raw(i as u32)).iter().map(|p| p.as_usize()).collect();
            // Stable coefficient positions across time copies, independent of dense
            // node order in the unfolding.
            let child = indexer.key_of(i as u32)?;
            let mut keyed = Vec::with_capacity(parents[i].len());
            for &p in &parents[i] {
                let parent = indexer.key_of(p as u32)?;
                let lag = child.offset.checked_sub(parent.offset).ok_or_else(|| {
                    EstimationError::unsupported("unfolded parent lag overflowed")
                })?;
                keyed.push((p, parent.variable, lag));
            }
            keyed.sort_by_key(|&(_, variable, lag)| (variable.raw(), lag));
            parents[i] = keyed.iter().map(|(p, _, _)| *p).collect();
            let hard_intervention =
                overlay_at[i].is_some_and(|overlay| overlay.node.level.is_some());
            if hard_intervention || parents[i].is_empty() {
                free_columns[i] = Some(sample.column(column_of[i]).to_vec());
                continue;
            }
            let t = sample.column(column_of[parents[i][0]]);
            let covs: Vec<_> = parents[i]
                .iter()
                .skip(1)
                .map(|&p| (VariableId::from_raw(p as u32), sample.column(column_of[p])))
                .collect();
            let design =
                CompiledDesign::linear_adjustment(t, &covs, sample.column(column_of[i]), &[])?;
            let parent_values: Vec<&[f64]> =
                parents[i].iter().map(|&p| sample.column(column_of[p])).collect();
            let intervened: Vec<bool> = parents[i].iter().map(|&p| intervention[p]).collect();
            let child_lag = columns[column_of[i]].lag.raw();
            let modifier_values: Vec<Vec<f64>> = if intervened.iter().any(|flag| *flag) {
                modifiers
                    .iter()
                    .map(|modifier| {
                        let index = modifier_at
                            .iter()
                            .find(|(variable, lag, _)| variable == modifier && *lag == child_lag)
                            .map(|(_, _, index)| *index)
                            .ok_or_else(|| {
                                EstimationError::unsupported(
                                    "sustained modifier is missing from the aligned window",
                                )
                            })?;
                        Ok(sample.column(index).to_vec())
                    })
                    .collect::<Result<Vec<_>, EstimationError>>()?
            } else {
                Vec::new()
            };
            let modifier_refs: Vec<&[f64]> = modifier_values.iter().map(Vec::as_slice).collect();
            let parent_variables: Vec<VariableId> =
                keyed.iter().map(|(_, variable, _)| *variable).collect();
            let modifier_is_parent: Vec<bool> =
                modifiers.iter().map(|modifier| parent_variables.contains(modifier)).collect();
            let (design, slots) = extend_mechanism_design(
                design,
                &parent_values,
                &intervened,
                &modifier_refs,
                modifiers,
                &modifier_is_parent,
            )?;
            interaction_at[i] = slots;
            if !modifier_values.is_empty() {
                node_modifier_values[i] = modifier_values;
            }
            designs[i] = Some(design);
        }
        let sampling_weights = weight_sequential_windows(
            problem.data,
            &mut designs,
            max_lag as usize,
            n,
            gaps || !modifiers.is_empty() || !matches!(eval, SequentialEval::Contrast { .. }),
        )?;
        let mut factual = vec![0.0; node_count];
        for &i in &order {
            let col = sample.column(column_of[i]);
            factual[i] = col.iter().sum::<f64>() / n as f64;
        }
        Ok(Self {
            _temporal_lease: temporal_lease,
            node_count,
            order,
            parents,
            designs,
            intervention,
            overlay_at,
            free_columns,
            factual,
            outcome,
            n,
            max_lag,
            eval,
            modifiers: modifiers.to_vec(),
            interaction_at,
            node_modifier_values,
            window_complete,
            sampling_weights,
        })
    }

    fn modifier_mean(&self, node: usize, modifier: usize, rows: Option<&[usize]>) -> f64 {
        let column = &self.node_modifier_values[node][modifier];
        if let Some(rows) = rows.filter(|rows| !rows.is_empty()) {
            rows.iter().map(|&row| column[row]).sum::<f64>() / rows.len() as f64
        } else {
            column.iter().sum::<f64>() / self.n as f64
        }
    }

    /// Parent slope in the contrast. An intervened parent that names modifiers
    /// contributes `β_p + Σ_m β_{p×W_m} E[W_m]`; the modifier's own level does not.
    fn marginal_parent_slope(
        &self,
        coefficients: &[f64],
        node: usize,
        parent: usize,
        rows: Option<&[usize]>,
    ) -> f64 {
        let mut slope = coefficients[parent + 1];
        for (modifier, &slot) in self.interaction_at[node][parent].iter().enumerate() {
            slope += coefficients[slot] * self.modifier_mean(node, modifier, rows);
        }
        slope
    }

    /// Per-row score of the modifier means a `Contrast` reads:
    /// `Σ_{i,m} ∂contrast/∂w̄_{i,m} · (w_{i,m,t} − w̄_{i,m})` over fitted nodes `i` and
    /// modifiers `m`. The contrast evaluates every interaction at the sample mean of its
    /// modifier, so the mean's sampling error moves the contrast even with the
    /// coefficients held fixed. The contrast is linear in each single mean, so the
    /// gradient is the exact change from replacing that mean's interaction slots with a
    /// unit step. `None` for a `Level` and for a contrast that reads no modifier mean.
    fn modifier_mean_scores(&self, base: &[Vec<f64>]) -> Option<Vec<f64>> {
        if self.modifiers.is_empty() || !matches!(self.eval, SequentialEval::Contrast { .. }) {
            return None;
        }
        let value = self.propagate(base, &self.factual, None);
        let mut score = vec![0.0; self.n];
        let mut any = false;
        for &i in &self.order {
            if base[i].is_empty() {
                continue;
            }
            for (modifier, column) in self.node_modifier_values[i].iter().enumerate() {
                // A unit step in this mean adds each parent's interaction coefficient to
                // that parent's main slope.
                let mut stepped = base.to_vec();
                for (parent, slots) in self.interaction_at[i].iter().enumerate() {
                    if let Some(&slot) = slots.get(modifier) {
                        stepped[i][parent + 1] += base[i][slot];
                    }
                }
                let slope = self.propagate(&stepped, &self.factual, None) - value;
                if !slope.is_finite() || slope == 0.0 {
                    continue;
                }
                let mean = self.modifier_mean(i, modifier, None);
                for (s, w) in score.iter_mut().zip(column) {
                    *s += slope * (w - mean);
                }
                any = true;
            }
        }
        any.then_some(score)
    }

    fn propagate(&self, coefficients: &[Vec<f64>], factual: &[f64], rows: Option<&[usize]>) -> f64 {
        match self.eval {
            SequentialEval::Contrast { delta } => {
                let mut differences = vec![0.0; self.node_count];
                for &i in &self.order {
                    differences[i] = if self.intervention[i] {
                        delta
                    } else {
                        self.parents[i]
                            .iter()
                            .enumerate()
                            .map(|(p, &node)| {
                                self.marginal_parent_slope(&coefficients[i], i, p, rows)
                                    * differences[node]
                            })
                            .sum()
                    };
                }
                differences[self.outcome]
            }
            SequentialEval::Level => propagate_linear_level(
                &self.order,
                self.node_count,
                self.outcome,
                |i| self.overlay_at.get(i).copied().flatten(),
                |i, values| {
                    if coefficients[i].is_empty() {
                        factual[i]
                    } else {
                        linear_natural(&coefficients[i], &self.parents[i], values)
                    }
                },
            ),
        }
    }

    /// OLS coefficients of every fitted mechanism on all aligned rows, or on the
    /// row map `rows` (any length).
    fn fit_ols(
        &self,
        rows: Option<&[usize]>,
        ls_ws: &mut LeastSquaresWorkspace,
        x_boot: &mut Vec<f64>,
        y_boot: &mut Vec<f64>,
    ) -> Result<Vec<Vec<f64>>, EstimationError> {
        let n = self.n;
        let mut coefficients = vec![Vec::new(); self.node_count];
        for &i in &self.order {
            if let Some(design) = &self.designs[i] {
                let (nrows, ncols) = if let Some(rows) = rows {
                    let m = rows.len();
                    let p = design.ncols;
                    x_boot.resize(m * p, 0.0);
                    y_boot.resize(m, 0.0);
                    for c in 0..p {
                        let column = &design.matrix[c * n..(c + 1) * n];
                        for (r, &src) in rows.iter().enumerate() {
                            x_boot[c * m + r] = column[src];
                        }
                    }
                    for (r, &src) in rows.iter().enumerate() {
                        y_boot[r] = design.outcome[src];
                    }
                    (m, p)
                } else {
                    x_boot.clear();
                    x_boot.extend_from_slice(&design.matrix);
                    y_boot.clear();
                    y_boot.extend_from_slice(&design.outcome);
                    (design.nrows, design.ncols)
                };
                coefficients[i] =
                    SequentialRegression::solve_design(x_boot, nrows, ncols, y_boot, None, ls_ws)?
                        .coefficients;
            }
        }
        Ok(coefficients)
    }

    /// Every mechanism's OLS fit on all aligned rows, residuals included, in fitting order.
    ///
    /// One solve per mechanism serves the point value, the contrast influence and the
    /// normal-equation scores of the block length: each of those reads the same fit, so
    /// none refits.
    fn fit_full(&self, ls_ws: &mut LeastSquaresWorkspace) -> FullFit {
        let fits = self
            .order
            .iter()
            .filter_map(|&i| {
                let design = self.designs[i].as_ref()?;
                let fit = SequentialRegression::solve_design(
                    &design.matrix,
                    design.nrows,
                    design.ncols,
                    &design.outcome,
                    None,
                    ls_ws,
                );
                Some((i, fit))
            })
            .collect();
        FullFit { fits }
    }

    /// Coefficients of every fitted mechanism by node, or the first failed solve.
    fn coefficients_of(&self, full: &FullFit) -> Result<Vec<Vec<f64>>, EstimationError> {
        let mut coefficients = vec![Vec::new(); self.node_count];
        for (i, fit) in &full.fits {
            let fit = fit.as_ref().map_err(|error| EstimationError::from(error.clone()))?;
            coefficients[*i].clone_from(&fit.coefficients);
        }
        Ok(coefficients)
    }

    /// Point value on all aligned rows, or refit on the row map `rows`. A `Level`
    /// refit also re-reads the factual means of mechanism-free nodes on `rows`.
    fn evaluate(
        &self,
        rows: Option<&[usize]>,
        ls_ws: &mut LeastSquaresWorkspace,
    ) -> Result<f64, EstimationError> {
        self.evaluate_into(rows, ls_ws, &mut Vec::new(), &mut Vec::new())
    }

    fn evaluate_into(
        &self,
        rows: Option<&[usize]>,
        ls_ws: &mut LeastSquaresWorkspace,
        x_boot: &mut Vec<f64>,
        y_boot: &mut Vec<f64>,
    ) -> Result<f64, EstimationError> {
        let coefficients = self.fit_ols(rows, ls_ws, x_boot, y_boot)?;
        Ok(match (self.eval, rows) {
            (SequentialEval::Level, Some(rows)) if !rows.is_empty() => {
                let mut factual = self.factual.clone();
                for &i in &self.order {
                    if let Some(column) = &self.free_columns[i] {
                        factual[i] =
                            rows.iter().map(|&r| column[r]).sum::<f64>() / rows.len() as f64;
                    }
                }
                self.propagate(&coefficients, &factual, None)
            }
            _ => self.propagate(&coefficients, &self.factual, rows),
        })
    }

    /// Gradient of the propagated value in every fitted coefficient. The value is
    /// multilinear in the coefficients, so central differences are exact up to rounding.
    fn gradient(&self, base: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let mut gradient: Vec<Vec<f64>> = base.iter().map(|c| vec![0.0; c.len()]).collect();
        for &i in &self.order {
            for k in 0..base[i].len() {
                let h = 1e-6 * base[i][k].abs().max(1.0);
                let mut up = base.to_vec();
                let mut down = base.to_vec();
                up[i][k] += h;
                down[i][k] -= h;
                gradient[i][k] = (self.propagate(&up, &self.factual, None)
                    - self.propagate(&down, &self.factual, None))
                    / (2.0 * h);
            }
        }
        gradient
    }

    /// Gradient of a `Level` in the factual mean of every mechanism-free node, as
    /// `(node, ∂level/∂mean)` for the nodes the level actually reads. The level is linear
    /// in each mean up to a bounded overlay's clamp, so central differences are exact
    /// away from the clamp. Empty for a `Contrast`, whose differences cancel every mean.
    fn root_mean_gradient(&self, base: &[Vec<f64>]) -> Vec<(usize, f64)> {
        if !matches!(self.eval, SequentialEval::Level) {
            return Vec::new();
        }
        self.order
            .iter()
            .filter(|&&i| self.free_columns[i].is_some())
            .filter_map(|&i| {
                let h = 1e-6 * self.factual[i].abs().max(1.0);
                let (mut up, mut down) = (self.factual.clone(), self.factual.clone());
                up[i] += h;
                down[i] -= h;
                let slope = (self.propagate(base, &up, None) - self.propagate(base, &down, None))
                    / (2.0 * h);
                (slope.is_finite() && slope != 0.0).then_some((i, slope))
            })
            .collect()
    }

    /// Per-row score of the root means the level reads: `Σ_r ∂level/∂x̄_r · (x_{r,t} − x̄_r)`,
    /// the influence of the sample means inside the propagated level. A persistent
    /// exogenous root carries its full long-run variance into the level even when every
    /// regression score is white. `None` when the level reads no root mean.
    fn root_mean_scores(&self, base: &[Vec<f64>]) -> Option<Vec<f64>> {
        let gradient = self.root_mean_gradient(base);
        if gradient.is_empty() {
            return None;
        }
        let mut score = vec![0.0; self.n];
        for (i, slope) in gradient {
            let column = self.free_columns[i].as_ref()?;
            for (s, x) in score.iter_mut().zip(column) {
                *s += slope * (x - self.factual[i]);
            }
        }
        Some(score)
    }

    /// Per-row influence of the propagated value: `Σ_i n·x_{i,r}ᵀ (X_iᵀX_i)⁻¹ g_i · e_{i,r}`
    /// over fitted mechanisms `i` with gradient `g_i`, plus, for a `Level`, the root-mean
    /// scores ([`Self::root_mean_scores`]). Both terms are `O_p(1)`: the influence feeds the
    /// block length and the kernel-bias factor as well as the effective-row count, so its
    /// scale is not immaterial. `base` holds the full-sample OLS coefficients by node.
    fn influence_from(&self, base: &[Vec<f64>]) -> Option<Vec<f64>> {
        let gradient = self.gradient(base);
        let n = self.n;
        let mut score = self.root_mean_scores(base).unwrap_or_else(|| vec![0.0; n]);
        for &i in &self.order {
            let Some(design) = &self.designs[i] else {
                continue;
            };
            let p = design.ncols;
            let xtx = crate::util::gram(&design.matrix[..n * p], n, p);
            let v = crate::util::solve_spd(&xtx, &gradient[i], p)?;
            for (r, slot) in score.iter_mut().enumerate() {
                let (mut fitted, mut xv) = (0.0, 0.0);
                for (c, (coef, vc)) in base[i].iter().zip(&v).enumerate() {
                    let x = design.matrix[c * n + r];
                    fitted += x * coef;
                    xv += x * vc;
                }
                *slot += n as f64 * xv * (design.outcome[r] - fitted);
            }
        }
        score.iter().all(|s| s.is_finite()).then_some(score)
    }

    /// [`crate::temporal_block::normal_equation_scores_of_residuals`] of every mechanism
    /// whose solve in `full` succeeded, from the residuals that solve holds.
    fn normal_equation_scores_of(&self, full: &FullFit) -> Vec<Vec<f64>> {
        full.fits
            .iter()
            .filter_map(|(i, fit)| {
                let design = self.designs[*i].as_ref()?;
                let fit = fit.as_ref().ok()?;
                (design.nrows != 0 && design.ncols != 0).then(|| {
                    crate::temporal_block::normal_equation_scores_of_residuals(
                        &design.matrix,
                        design.nrows,
                        design.ncols,
                        &fit.residuals,
                    )
                })
            })
            .flatten()
            .collect()
    }

    /// [`crate::temporal_block::dependence_block_length`] of the unfolded span over the
    /// contrast influence (already computed) plus every mechanism's normal-equation scores
    /// from the shared full-sample fit.
    fn dependence_block_length_with(&self, influence: Option<&[f64]>, full: &FullFit) -> usize {
        let normal = self.normal_equation_scores_of(full);
        let scores: Vec<&[f64]> =
            influence.into_iter().chain(normal.iter().map(Vec::as_slice)).collect();
        crate::temporal_block::dependence_block_length(self.max_lag as usize + 1, self.n, &scores)
    }
}

#[allow(clippy::too_many_arguments, reason = "the inputs describe one estimation procedure")]
fn estimate_sequential(
    problem: &SequentialProblem<'_>,
    target: &SequenceTarget<'_>,
    eval: SequentialEval,
    inference: SequentialInference<'_>,
    ctx: &ExecutionContext,
    modifiers: &[VariableId],
    validation: Option<&mut Vec<SequentialBayesianMechanism>>,
    budget: &mut antecedent_core::TemporalBudget,
) -> Result<(EffectEstimate, Option<CausalPosterior>), EstimationError> {
    estimate_sequential_with_gaps(
        problem, target, eval, inference, ctx, modifiers, validation, false, budget,
    )
    .map(|(effect, posterior, _)| (effect, posterior))
}

#[allow(
    clippy::too_many_arguments,
    reason = "the sequential estimator's inputs plus the gap switch; each is a distinct caller input"
)]
fn estimate_sequential_with_gaps(
    problem: &SequentialProblem<'_>,
    target: &SequenceTarget<'_>,
    eval: SequentialEval,
    inference: SequentialInference<'_>,
    ctx: &ExecutionContext,
    modifiers: &[VariableId],
    validation: Option<&mut Vec<SequentialBayesianMechanism>>,
    gaps: bool,
    budget: &mut antecedent_core::TemporalBudget,
) -> Result<(EffectEstimate, Option<CausalPosterior>, Option<crate::GapSelection>), EstimationError>
{
    let SequentialInference { status, assumptions, bootstrap_replicates, bayesian } = inference;
    if gaps && bayesian.is_some() && !matches!(eval, SequentialEval::Contrast { .. }) {
        return Err(EstimationError::unsupported_coded(
            antecedent_core::reason_code!("route_not_supported"),
            "a series with gaps is combined with the composed Bayesian posterior for the sustained contrast only: a level also reads root-mean scores with a consecutive-row standard error",
        ));
    }
    let setup =
        SequentialSetup::build_with_gaps(problem, target, eval, modifiers, ctx, gaps, budget)?;
    if bayesian.is_some() && setup.sampling_weights.is_some() {
        return Err(crate::sampling_weights::unsupported_route(
            "sampling weights are read by the frequentist sustained contrast; the composed \
             Bayesian posterior fits unweighted mechanisms",
        ));
    }
    let selection = setup
        .window_complete
        .clone()
        .map(|complete| crate::GapSelection { first_time: setup.max_lag as usize, complete });
    let mut ls_ws = LeastSquaresWorkspace::default();
    if let Some(estimator) = bayesian {
        let composition =
            BayesianComposition { setup: &setup, problem, eval, estimator, status, ctx, gaps };
        let (effect, posterior) = composition.estimate(assumptions, validation, &mut ls_ws)?;
        return Ok((effect, Some(posterior), selection));
    }
    let effect =
        block_bootstrap_estimate(&setup, assumptions, bootstrap_replicates, ctx, &mut ls_ws)?;
    Ok((effect, None, selection))
}

#[cfg(test)]
mod tests {
    use antecedent_core::{StreamDomain, TargetPopulation};
    use antecedent_graph::ensure_lagged;

    use super::*;

    /// `Z_s` exogenous, `T_s = 0.5 Z_s + u`, `Y_s = 1 + 2 T_{s-1} + 0.8 T_{s-2} + 0.5 Z_{s-1} + e`.
    fn fixture() -> (TimeSeriesData, TemporalDag) {
        let n = 240usize;
        let ctx = ExecutionContext::for_tests(11);
        let mut rng = ctx.rng.stream_for(StreamDomain::Estimate, 3);
        let mut draw = || 2.0 * rng.next_f64() - 1.0;
        let z: Vec<f64> = (0..n).map(|_| draw()).collect();
        let t: Vec<f64> = z.iter().map(|z| 0.5 * z + draw()).collect();
        let y: Vec<f64> = (0..n)
            .map(|s| {
                let at = |v: &[f64], lag: usize| s.checked_sub(lag).map_or(0.0, |i| v[i]);
                1.0 + 2.0 * at(&t, 1) + 0.8 * at(&t, 2) + 0.5 * at(&z, 1) + 0.3 * draw()
            })
            .collect();
        let data = TimeSeriesData::from_f64_columns(
            [("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())],
            1,
        )
        .unwrap();
        let mut graph = TemporalDag::empty();
        let id = VariableId::from_raw;
        let mut edge = |from: u32, from_lag: u32, to: u32, to_lag: u32| {
            let src = ensure_lagged(&mut graph, id(from), Lag::from_raw(from_lag)).unwrap();
            let dst = ensure_lagged(&mut graph, id(to), Lag::from_raw(to_lag)).unwrap();
            graph.insert_directed(src, dst).unwrap();
        };
        edge(2, 0, 0, 0);
        edge(0, 1, 1, 0);
        edge(0, 2, 1, 0);
        edge(2, 1, 1, 0);
        (data, graph)
    }

    // Both offsets carry the same Set level (0.5): these tests exercise a
    // multi-offset hard-set schedule, not a per-node dose difference (which
    // `resolve_schedule_active_level` certifies at a level-free reference).
    fn hard_overlays() -> Vec<SequentialMechanismOverlay> {
        [(-2, 0.5), (-1, 0.5)]
            .into_iter()
            .map(|(offset, level)| {
                SequentialNodeOverlay {
                    variable: VariableId::from_raw(0),
                    offset,
                    level: Some(level),
                    shift: 0.0,
                }
                .into()
            })
            .collect()
    }

    /// The propagated level is the g-formula of a chain: with `x0 = 1`, `x1 = 2 + 3 x0`,
    /// `x2 = 1 + 0.5 x1 - x0`, the natural level is `1 + 0.5 * 5 - 1 = 2.5`; a hard set of
    /// `x1` to 4 cuts its mechanism and gives `1 + 0.5 * 4 - 1 = 2`; a shift of 1 on `x1`
    /// gives `1 + 0.5 * 6 - 1 = 3`.
    #[test]
    fn linear_level_walk_matches_the_hand_computed_g_formula() {
        let coefficients = [vec![], vec![2.0, 3.0], vec![1.0, 0.5, -1.0]];
        let parents = [vec![], vec![0], vec![1, 0]];
        let run = |overlay_on_x1: Option<SequentialMechanismOverlay>| {
            propagate_linear_level(
                &[0, 1, 2],
                3,
                2,
                |i| if i == 1 { overlay_on_x1 } else { None },
                |i, values| {
                    if coefficients[i].is_empty() {
                        1.0
                    } else {
                        linear_natural(&coefficients[i], &parents[i], values)
                    }
                },
            )
        };
        let node = |level, shift| {
            SequentialMechanismOverlay::from(SequentialNodeOverlay {
                variable: VariableId::from_raw(1),
                offset: 0,
                level,
                shift,
            })
        };
        assert!((run(None) - 2.5).abs() < 1e-12);
        assert!((run(Some(node(Some(4.0), 0.0))) - 2.0).abs() < 1e-12);
        assert!((run(Some(node(None, 1.0))) - 3.0).abs() < 1e-12);
    }

    /// Repeating a Bayesian sequence with the same seed preserves every retained draw.
    #[test]
    fn bayesian_sequence_repeats_draws_for_the_same_seed() {
        let (data, graph) = fixture();
        let overlays = hard_overlays();
        let schedule: Vec<(VariableId, i32, Option<f64>)> =
            overlays.iter().map(|o| (o.node.variable, o.node.offset, o.node.level)).collect();
        let identified = antecedent_identify::TemporalBackdoorIdentifier::new()
            .identify_temporal_schedule(
                &graph,
                VariableId::from_raw(1),
                0,
                &schedule,
                None,
                TargetPopulation::AllObserved,
            )
            .unwrap();
        let estimand = identified.result.estimands.first().unwrap();
        let estimator = BayesianGComputationAte::new();
        let run = || {
            let (_, posterior) = estimate_sequence_mechanisms(
                &SequentialProblem {
                    data: &data,
                    graph: &graph,
                    indexer: &identified.indexer,
                    estimand,
                },
                &SequenceTarget {
                    outcome: VariableId::from_raw(1),
                    outcome_offset: 0,
                    overlays: &overlays,
                },
                SequentialInference {
                    status: IdentificationStatus::NonparametricallyIdentified,
                    assumptions: AssumptionSet::new(),
                    bootstrap_replicates: 0,
                    bayesian: Some(&estimator),
                },
                &ExecutionContext::for_tests(5),
            )
            .unwrap();
            posterior.unwrap().draws.values
        };
        let first = run();
        let second = run();
        assert!(!first.is_empty());
        assert_eq!(first.len(), second.len());
        assert!(first.iter().zip(second.iter()).all(|(a, b)| a.to_bits() == b.to_bits()));
    }

    /// A Bayesian Sequence level is tempered like a Sustained contrast (its mechanisms carry
    /// the long-run tempering note and assumption, not the iid likelihood) and records the
    /// root-mean shift it adds.
    #[test]
    fn bayesian_level_is_tempered_and_discloses_its_root_mean_shift() {
        let (data, graph) = fixture();
        let overlays = hard_overlays();
        let schedule: Vec<(VariableId, i32, Option<f64>)> =
            overlays.iter().map(|o| (o.node.variable, o.node.offset, o.node.level)).collect();
        let id_res = antecedent_identify::TemporalBackdoorIdentifier::new()
            .identify_temporal_schedule(
                &graph,
                VariableId::from_raw(1),
                0,
                &schedule,
                None,
                TargetPopulation::AllObserved,
            )
            .unwrap();
        let estimand = id_res.result.estimands.first().cloned().expect("identified estimand");
        let bayes = BayesianGComputationAte::new();
        let (_, posterior) = estimate_sequence_mechanisms(
            &SequentialProblem {
                data: &data,
                graph: &graph,
                indexer: &id_res.indexer,
                estimand: &estimand,
            },
            &SequenceTarget {
                outcome: VariableId::from_raw(1),
                outcome_offset: 0,
                overlays: &overlays,
            },
            SequentialInference {
                status: IdentificationStatus::NonparametricallyIdentified,
                assumptions: AssumptionSet::new(),
                bootstrap_replicates: 0,
                bayesian: Some(&bayes),
            },
            &ExecutionContext::for_tests(5),
        )
        .unwrap();
        let posterior = posterior.expect("Bayesian posterior");
        let has = |id: &str| {
            posterior.assumptions.entries.iter().any(|a| {
                matches!(&a.assumption, Assumption::ParametricRestriction(p) if p.id.as_ref() == id)
            })
        };
        assert!(has(crate::serial_dependence::DEPENDENCE_ASSUMPTION_ID), "tempering recorded");
        assert!(has("temporal.sequential.level_posterior"), "root-mean shift recorded");
        assert!(
            crate::serial_dependence::tempering_kappa_from_notes(&posterior.diagnostics.notes)
                .is_some_and(|kappa| kappa >= 1.0)
        );
    }

    /// A level reads the sample mean of its exogenous root `Z_{s-1}` through the outcome
    /// mechanism's `Z` coefficient, so `∂level/∂z̄` is that fitted coefficient and the
    /// root-mean score of row `t` is `β_Z (z_t − z̄)`. A contrast cancels every mean.
    #[test]
    fn level_reads_the_root_mean_through_its_outcome_coefficient() {
        let (data, graph) = fixture();
        let overlays = hard_overlays();
        let schedule: Vec<(VariableId, i32, Option<f64>)> =
            overlays.iter().map(|o| (o.node.variable, o.node.offset, o.node.level)).collect();
        let id_res = antecedent_identify::TemporalBackdoorIdentifier::new()
            .identify_temporal_schedule(
                &graph,
                VariableId::from_raw(1),
                0,
                &schedule,
                None,
                TargetPopulation::AllObserved,
            )
            .unwrap();
        let estimand = id_res.result.estimands.first().cloned().expect("identified estimand");
        let ctx = ExecutionContext::for_tests(5);
        let build = |eval| {
            SequentialSetup::build(
                &SequentialProblem {
                    data: &data,
                    graph: &graph,
                    indexer: &id_res.indexer,
                    estimand: &estimand,
                },
                &SequenceTarget {
                    outcome: VariableId::from_raw(1),
                    outcome_offset: 0,
                    overlays: &overlays,
                },
                eval,
                &[],
                &ctx,
            )
            .unwrap()
        };
        let setup = build(SequentialEval::Level);
        let mut ws = LeastSquaresWorkspace::default();
        let base = setup.fit_ols(None, &mut ws, &mut Vec::new(), &mut Vec::new()).unwrap();
        let gradient = setup.root_mean_gradient(&base);
        assert_eq!(gradient.len(), 1, "only Z(-1) is an unintervened root: {gradient:?}");
        let (root, slope) = gradient[0];
        // Outcome parents sort as T(-1), T(-2), Z(-1): coefficients [1, T1, T2, Z1].
        let beta_z = base[setup.outcome][3];
        assert!((slope - beta_z).abs() < 1e-6, "slope {slope} vs beta_Z {beta_z}");
        assert!((beta_z - 0.5).abs() < 0.1, "the fitted Z coefficient is near its truth");
        let scores = setup.root_mean_scores(&base).unwrap();
        let column = setup.free_columns[root].as_ref().unwrap();
        let mean = column.iter().sum::<f64>() / column.len() as f64;
        for (score, x) in scores.iter().zip(column) {
            assert!((score - slope * (x - mean)).abs() < 1e-12);
        }
        assert!(scores.iter().sum::<f64>().abs() < 1e-9);
        // The level's influence carries the root term on top of the regression scores.
        let influence = setup.influence_from(&base).unwrap();
        assert_eq!(influence.len(), scores.len());
        assert!(influence.iter().sum::<f64>().abs() < 1e-6, "influence is mean zero");
        let contrast = build(SequentialEval::Contrast { delta: 1.0 });
        assert!(contrast.root_mean_gradient(&base).is_empty());
    }

    /// The point value, influence and normal-equation scores read one full-sample fit; each is
    /// bit-identical to the refit it replaces.
    #[test]
    fn shared_full_fit_reproduces_the_refits_bitwise() {
        let (data, graph) = fixture();
        let overlays = hard_overlays();
        let schedule: Vec<(VariableId, i32, Option<f64>)> =
            overlays.iter().map(|o| (o.node.variable, o.node.offset, o.node.level)).collect();
        let id_res = antecedent_identify::TemporalBackdoorIdentifier::new()
            .identify_temporal_schedule(
                &graph,
                VariableId::from_raw(1),
                0,
                &schedule,
                None,
                TargetPopulation::AllObserved,
            )
            .unwrap();
        let estimand = id_res.result.estimands.first().cloned().expect("identified estimand");
        let ctx = ExecutionContext::for_tests(5);
        for eval in [SequentialEval::Contrast { delta: 1.0 }, SequentialEval::Level] {
            let setup = SequentialSetup::build(
                &SequentialProblem {
                    data: &data,
                    graph: &graph,
                    indexer: &id_res.indexer,
                    estimand: &estimand,
                },
                &SequenceTarget {
                    outcome: VariableId::from_raw(1),
                    outcome_offset: 0,
                    overlays: &overlays,
                },
                eval,
                &[],
                &ctx,
            )
            .unwrap();
            let mut ws = LeastSquaresWorkspace::default();
            let full = setup.fit_full(&mut ws);
            let coefficients = setup.coefficients_of(&full).unwrap();
            let refit = setup.fit_ols(None, &mut ws, &mut Vec::new(), &mut Vec::new()).unwrap();
            assert_eq!(coefficients, refit);
            let point = setup.propagate(&coefficients, &setup.factual, None);
            assert_eq!(point.to_bits(), setup.evaluate(None, &mut ws).unwrap().to_bits());
            let refit_scores: Vec<Vec<f64>> = setup
                .order
                .iter()
                .filter_map(|&i| setup.designs[i].as_ref())
                .filter_map(|d| {
                    crate::temporal_block::normal_equation_scores(
                        &d.matrix, d.nrows, d.ncols, &d.outcome,
                    )
                })
                .flatten()
                .collect();
            let shared_scores = setup.normal_equation_scores_of(&full);
            assert!(!shared_scores.is_empty());
            assert_eq!(shared_scores.len(), refit_scores.len());
            for (shared, refit) in shared_scores.iter().zip(&refit_scores) {
                assert!(shared.iter().zip(refit).all(|(a, b)| a.to_bits() == b.to_bits()));
            }
        }
    }

    /// Cost of the block-length path of one sequential estimate: the contrast influence plus
    /// every mechanism's normal-equation scores, on a series long enough for the fits to matter.
    #[test]
    #[ignore = "timing harness: cargo test --release -p antecedent-estimate --lib timing_ -- --ignored --nocapture"]
    fn timing_sequential_block_length_path() {
        let n = 20_000usize;
        let ctx = ExecutionContext::for_tests(11);
        let mut rng = ctx.rng.stream_for(StreamDomain::Estimate, 3);
        let mut draw = || 2.0 * rng.next_f64() - 1.0;
        let z: Vec<f64> = (0..n).map(|_| draw()).collect();
        let t: Vec<f64> = z.iter().map(|z| 0.5 * z + draw()).collect();
        let y: Vec<f64> = (0..n)
            .map(|s| {
                let at = |v: &[f64], lag: usize| s.checked_sub(lag).map_or(0.0, |i| v[i]);
                1.0 + 2.0 * at(&t, 1) + 0.8 * at(&t, 2) + 0.5 * at(&z, 1) + 0.3 * draw()
            })
            .collect();
        let data = TimeSeriesData::from_f64_columns(
            [("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())],
            1,
        )
        .unwrap();
        let mut graph = TemporalDag::empty();
        let id = VariableId::from_raw;
        let mut edge = |from: u32, from_lag: u32, to: u32, to_lag: u32| {
            let src = ensure_lagged(&mut graph, id(from), Lag::from_raw(from_lag)).unwrap();
            let dst = ensure_lagged(&mut graph, id(to), Lag::from_raw(to_lag)).unwrap();
            graph.insert_directed(src, dst).unwrap();
        };
        edge(2, 0, 0, 0);
        edge(0, 1, 1, 0);
        edge(0, 2, 1, 0);
        edge(2, 1, 1, 0);
        let overlays = hard_overlays();
        let schedule: Vec<(VariableId, i32, Option<f64>)> =
            overlays.iter().map(|o| (o.node.variable, o.node.offset, o.node.level)).collect();
        let id_res = antecedent_identify::TemporalBackdoorIdentifier::new()
            .identify_temporal_schedule(
                &graph,
                VariableId::from_raw(1),
                0,
                &schedule,
                None,
                TargetPopulation::AllObserved,
            )
            .unwrap();
        let estimand = id_res.result.estimands.first().cloned().expect("identified estimand");
        let setup = SequentialSetup::build(
            &SequentialProblem {
                data: &data,
                graph: &graph,
                indexer: &id_res.indexer,
                estimand: &estimand,
            },
            &SequenceTarget {
                outcome: VariableId::from_raw(1),
                outcome_offset: 0,
                overlays: &overlays,
            },
            SequentialEval::Contrast { delta: 1.0 },
            &[],
            &ctx,
        )
        .unwrap();
        let mut ws = LeastSquaresWorkspace::default();
        let reps = 20;
        let start = std::time::Instant::now();
        let mut checksum = 0usize;
        for _ in 0..reps {
            let full = setup.fit_full(&mut ws);
            let base = setup.coefficients_of(&full).unwrap();
            let influence = setup.influence_from(&base);
            checksum += setup.dependence_block_length_with(influence.as_deref(), &full);
        }
        println!(
            "sequential influence + block length n={n}: {:.3} ms/call (checksum {checksum})",
            start.elapsed().as_secs_f64() * 1e3 / f64::from(reps)
        );
    }
}
