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
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]

use std::sync::Arc;

use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSet, AssumptionSource,
    AssumptionStatus, CausalRng, ExecutionContext, IdentificationStatus, Lag, ParametricAssumption,
    TargetPopulation, TemporalEffectQuery, TemporalNodeKey, TemporalPolicy, VariableId,
};
use antecedent_data::{LaggedColumn, LaggedSampleWorkspace, TemporalIndexer, TimeSeriesData};
use antecedent_expr::{EstimandMethod, IdentifiedEstimand};
use antecedent_graph::{DenseNodeId, TemporalDag};
use antecedent_prob::{BayesLikelihood, PosteriorDraws, PosteriorQuantityKind, PosteriorSchema};
use antecedent_stats::{CompiledDesign, DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace};

use crate::serial_dependence::{DependenceScope, SerialDependence};
use crate::{
    BayesianGCompWorkspace, BayesianGComputationAte, CausalPosterior, EffectEstimate,
    EstimationError, OverlapPolicy, PreparedBayesianProblem,
};

/// Stream salt of the root-mean shift draws, apart from every mechanism's own stream.
const ROOT_MEAN_STREAM: u64 = 0x2007_3A11_C0DE_51DE;

/// One intervened unfolded node and the licensed overlay applied there.
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
    data: &TimeSeriesData,
    graph: &TemporalDag,
    indexer: &TemporalIndexer,
    estimand: &IdentifiedEstimand,
    query: &TemporalEffectQuery,
    status: IdentificationStatus,
    assumptions: AssumptionSet,
    bootstrap_replicates: u32,
    bayesian: Option<&BayesianGComputationAte>,
    ctx: &ExecutionContext,
) -> Result<(EffectEstimate, Option<CausalPosterior>), EstimationError> {
    estimate_sustained_window_with_validation(
        data,
        graph,
        indexer,
        estimand,
        query,
        status,
        assumptions,
        bootstrap_replicates,
        bayesian,
        ctx,
        None,
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
    data: &TimeSeriesData,
    graph: &TemporalDag,
    indexer: &TemporalIndexer,
    estimand: &IdentifiedEstimand,
    query: &TemporalEffectQuery,
    status: IdentificationStatus,
    assumptions: AssumptionSet,
    bootstrap_replicates: u32,
    bayesian: Option<&BayesianGComputationAte>,
    ctx: &ExecutionContext,
    validation: Option<&mut Vec<SequentialBayesianMechanism>>,
) -> Result<(EffectEstimate, Option<CausalPosterior>), EstimationError> {
    let (overlays, delta) = sustained_contrast_overlays(query, estimand, status)?;
    if bayesian.is_some_and(|est| est.likelihood != BayesLikelihood::GaussianIdentity) {
        return Err(EstimationError::unsupported(
            "sequential Bayesian g-computation requires GaussianIdentity",
        ));
    }
    if bayesian.is_some_and(|est| est.backend == crate::BayesianBackendKind::Hmc) {
        return Err(EstimationError::unsupported(
            "composed sustained HMC requires derived-contrast chain diagnostics; use conjugate or Laplace",
        ));
    }
    estimate_sequential(
        data,
        graph,
        indexer,
        estimand,
        query.outcome,
        query.outcome_offset(),
        &overlays,
        SequentialEval::Contrast { delta },
        status,
        assumptions,
        bootstrap_replicates,
        bayesian,
        ctx,
        validation,
    )
}

/// Validate a multi-step Sustained contrast and build its per-time overlays and
/// the active-minus-control `delta`.
fn sustained_contrast_overlays(
    query: &TemporalEffectQuery,
    estimand: &IdentifiedEstimand,
    status: IdentificationStatus,
) -> Result<(Vec<SequentialMechanismOverlay>, f64), EstimationError> {
    query.validate()?;
    let TemporalPolicy::Sustained { from, until } = query.policy else {
        return Err(EstimationError::unsupported(
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
    let overlays = (from..=until)
        .map(|offset| {
            SequentialMechanismOverlay::from(SequentialNodeOverlay {
                variable: query.treatment,
                offset,
                level: Some(active),
                shift: 0.0,
            })
        })
        .collect();
    Ok((overlays, delta))
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
        data: &TimeSeriesData,
        graph: &TemporalDag,
        indexer: &TemporalIndexer,
        estimand: &IdentifiedEstimand,
        query: &TemporalEffectQuery,
        status: IdentificationStatus,
        ctx: &ExecutionContext,
    ) -> Result<Self, EstimationError> {
        let (overlays, delta) = sustained_contrast_overlays(query, estimand, status)?;
        let setup = SequentialSetup::build(
            data,
            graph,
            indexer,
            query.outcome,
            query.outcome_offset(),
            &overlays,
            SequentialEval::Contrast { delta },
            ctx,
        )?;
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
        self.setup.influence(&mut LeastSquaresWorkspace::default())
    }

    /// Every mechanism's OLS normal-equation scores on the aligned rows
    /// ([`crate::temporal_block::normal_equation_scores`]), for
    /// [`crate::temporal_block::dependence_block_length`].
    #[must_use]
    pub fn normal_equation_scores(&self) -> Vec<Vec<f64>> {
        self.setup.normal_equation_scores()
    }

    /// Circular-block length of this contrast's Frequentist block-bootstrap
    /// interval ([`estimate_sustained_window`] with replicates), so a check of that
    /// interval resamples the same blocks.
    #[must_use]
    pub fn block_length(&self) -> usize {
        self.setup.dependence_block_length(&mut LeastSquaresWorkspace::default())
    }
}

/// Sequential g-computation of the interventional **level** under per-node
/// Set / Soft constant / Soft shift overlays. Shares the Sustained engine.
///
/// # Errors
///
/// Empty or duplicate overlays, unidentified estimand, or fit failures.
pub fn estimate_sequence_overlays(
    data: &TimeSeriesData,
    graph: &TemporalDag,
    indexer: &TemporalIndexer,
    estimand: &IdentifiedEstimand,
    outcome: VariableId,
    outcome_offset: i32,
    overlays: &[SequentialNodeOverlay],
    status: IdentificationStatus,
    assumptions: AssumptionSet,
    bootstrap_replicates: u32,
    bayesian: Option<&BayesianGComputationAte>,
    ctx: &ExecutionContext,
) -> Result<(EffectEstimate, Option<CausalPosterior>), EstimationError> {
    let mechanisms =
        overlays.iter().copied().map(SequentialMechanismOverlay::from).collect::<Vec<_>>();
    estimate_sequence_mechanisms(
        data,
        graph,
        indexer,
        estimand,
        outcome,
        outcome_offset,
        &mechanisms,
        status,
        assumptions,
        bootstrap_replicates,
        bayesian,
        ctx,
    )
}

/// Estimate a sequence of deterministic propagated-mean mechanism interventions.
/// Multiplication scales the structural assignment; bounded shifts target its population mean.
///
/// # Errors
/// Invalid parameters, duplicate coordinates, unidentified design, or fit failures.
pub fn estimate_sequence_mechanisms(
    data: &TimeSeriesData,
    graph: &TemporalDag,
    indexer: &TemporalIndexer,
    estimand: &IdentifiedEstimand,
    outcome: VariableId,
    outcome_offset: i32,
    overlays: &[SequentialMechanismOverlay],
    status: IdentificationStatus,
    mut assumptions: AssumptionSet,
    bootstrap_replicates: u32,
    bayesian: Option<&BayesianGComputationAte>,
    ctx: &ExecutionContext,
) -> Result<(EffectEstimate, Option<CausalPosterior>), EstimationError> {
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
        return Err(EstimationError::unsupported(
            "sequential Bayesian g-computation requires GaussianIdentity",
        ));
    }
    if bayesian.is_some_and(|est| est.backend == crate::BayesianBackendKind::Hmc) {
        return Err(EstimationError::unsupported(
            "composed sequence HMC requires derived-contrast chain diagnostics; use conjugate or Laplace",
        ));
    }
    estimate_sequential(
        data,
        graph,
        indexer,
        estimand,
        outcome,
        outcome_offset,
        overlays,
        SequentialEval::Level,
        status,
        assumptions,
        bootstrap_replicates,
        bayesian,
        ctx,
        None,
    )
}

/// RNG stream base for the sequential circular-block row bootstrap.
const SEQUENTIAL_BLOCK_STREAM: u64 = 0x5120_0000;

/// The identified unfolded design as fitted mechanisms over one set of
/// lag-aligned rows: every mechanism's OLS design reads the same rows, row `r`
/// being the unfolded window that ends at series time `max_lag + r`.
#[derive(Debug)]
struct SequentialSetup {
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
}

impl SequentialSetup {
    fn build(
        data: &TimeSeriesData,
        graph: &TemporalDag,
        indexer: &TemporalIndexer,
        outcome: VariableId,
        outcome_offset: i32,
        overlays: &[SequentialMechanismOverlay],
        eval: SequentialEval,
        ctx: &ExecutionContext,
    ) -> Result<Self, EstimationError> {
        let unfolded =
            graph.unfold(indexer.clone()).map_err(|e| EstimationError::data_msg(e.to_string()))?;
        let dag = &unfolded.dag;
        let node_count = dag.node_count();
        let outcome = indexer
            .dense_id(TemporalNodeKey { variable: outcome, offset: outcome_offset })
            .map_err(|e| EstimationError::data_msg(e.to_string()))? as usize;
        let mut intervention = vec![false; node_count];
        let mut overlay_at = vec![None; node_count];
        for &overlay in overlays {
            let dense = indexer
                .dense_id(TemporalNodeKey {
                    variable: overlay.node.variable,
                    offset: overlay.node.offset,
                })
                .map_err(|e| EstimationError::data_msg(e.to_string()))?
                as usize;
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
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| EstimationError::data_msg(e.to_string()))?
            .into_iter()
            .max()
            .unwrap_or(0)
            .max(0);
        let mut columns = Vec::new();
        let mut column_of = vec![0; node_count];
        let mut max_lag = 0;
        for &i in &order {
            let key =
                indexer.key_of(i as u32).map_err(|e| EstimationError::data_msg(e.to_string()))?;
            let lag = u32::try_from(anchor - key.offset)
                .map_err(|_| EstimationError::unsupported("invalid unfolded time offset"))?;
            max_lag = max_lag.max(lag);
            column_of[i] = columns.len();
            columns.push(LaggedColumn { variable: key.variable, lag: Lag::from_raw(lag) });
        }
        let plan = data.plan_lagged_sample(max_lag, Arc::from(columns))?;
        let mut workspace = LaggedSampleWorkspace::default();
        let sample = plan.prepare(data, &mut workspace, &ctx.kernel_policy)?;
        let n = sample.n;
        if n < 3 {
            return Err(EstimationError::unsupported(
                "sustained window requires at least three complete aligned rows",
            ));
        }
        let mut designs = vec![None; node_count];
        let mut parents = vec![Vec::new(); node_count];
        let mut free_columns = vec![None; node_count];
        for &i in &order {
            parents[i] =
                dag.parents(DenseNodeId::from_raw(i as u32)).iter().map(|p| p.as_usize()).collect();
            // Stable coefficient positions across time copies, independent of dense
            // node order in the unfolding.
            let child = indexer.key_of(i as u32).expect("unfolded node");
            parents[i].sort_by_key(|&p| {
                let parent = indexer.key_of(p as u32).expect("unfolded parent");
                (parent.variable.raw(), child.offset - parent.offset)
            });
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
            designs[i] = Some(CompiledDesign::linear_adjustment(
                t,
                &covs,
                sample.column(column_of[i]),
                &[],
            )?);
        }
        let mut factual = vec![0.0; node_count];
        for &i in &order {
            let col = sample.column(column_of[i]);
            factual[i] = col.iter().sum::<f64>() / n as f64;
        }
        Ok(Self {
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
        })
    }

    fn propagate(&self, coefficients: &[Vec<f64>], factual: &[f64]) -> f64 {
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
                            .map(|(p, &node)| coefficients[i][p + 1] * differences[node])
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
                    FaerBackend.least_squares(x_boot, nrows, ncols, y_boot, ls_ws)?.coefficients;
            }
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
                self.propagate(&coefficients, &factual)
            }
            _ => self.propagate(&coefficients, &self.factual),
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
                gradient[i][k] = (self.propagate(&up, &self.factual)
                    - self.propagate(&down, &self.factual))
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
                let slope = (self.propagate(base, &up) - self.propagate(base, &down)) / (2.0 * h);
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
    /// scale is not immaterial.
    fn influence(&self, ls_ws: &mut LeastSquaresWorkspace) -> Option<Vec<f64>> {
        let base = self.fit_ols(None, ls_ws, &mut Vec::new(), &mut Vec::new()).ok()?;
        let gradient = self.gradient(&base);
        let n = self.n;
        let mut score = self.root_mean_scores(&base).unwrap_or_else(|| vec![0.0; n]);
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

    /// [`crate::temporal_block::normal_equation_scores`] of every fitted mechanism.
    fn normal_equation_scores(&self) -> Vec<Vec<f64>> {
        self.order
            .iter()
            .filter_map(|&i| self.designs[i].as_ref())
            .filter_map(|d| {
                crate::temporal_block::normal_equation_scores(
                    &d.matrix, d.nrows, d.ncols, &d.outcome,
                )
            })
            .flatten()
            .collect()
    }

    /// [`crate::temporal_block::dependence_block_length`] of the unfolded span over
    /// the contrast influence plus every mechanism's normal-equation scores.
    fn dependence_block_length(&self, ls_ws: &mut LeastSquaresWorkspace) -> usize {
        self.dependence_block_length_with(self.influence(ls_ws).as_deref())
    }

    /// [`Self::dependence_block_length`] with the contrast influence already computed.
    fn dependence_block_length_with(&self, influence: Option<&[f64]>) -> usize {
        let normal = self.normal_equation_scores();
        let scores: Vec<&[f64]> =
            influence.into_iter().chain(normal.iter().map(Vec::as_slice)).collect();
        crate::temporal_block::dependence_block_length(self.max_lag as usize + 1, self.n, &scores)
    }
}

fn estimate_sequential(
    data: &TimeSeriesData,
    graph: &TemporalDag,
    indexer: &TemporalIndexer,
    estimand: &IdentifiedEstimand,
    outcome: VariableId,
    outcome_offset: i32,
    overlays: &[SequentialMechanismOverlay],
    eval: SequentialEval,
    status: IdentificationStatus,
    assumptions: AssumptionSet,
    bootstrap_replicates: u32,
    bayesian: Option<&BayesianGComputationAte>,
    ctx: &ExecutionContext,
    mut validation: Option<&mut Vec<SequentialBayesianMechanism>>,
) -> Result<(EffectEstimate, Option<CausalPosterior>), EstimationError> {
    let setup =
        SequentialSetup::build(data, graph, indexer, outcome, outcome_offset, overlays, eval, ctx)?;
    let order = &setup.order;
    let parents = &setup.parents;
    let designs = &setup.designs;
    let mut ls_ws = LeastSquaresWorkspace::default();
    if let Some(estimator) = bayesian {
        // Both the Sustained contrast and a Sequence level temper each stationary mechanism
        // by the long-run-variance ratio of the target's own linear combination of that
        // mechanism's coefficients: the gradient of the composed value, summed over the
        // mechanism's unfolded time copies (coefficient positions are stable across
        // copies). The value is linear in each coefficient, so central differences at the
        // OLS fit are exact up to rounding.
        let ols_base = setup.fit_ols(None, &mut ls_ws, &mut Vec::new(), &mut Vec::new()).ok();
        let contrast_gradient = ols_base.as_ref().map(|base| {
            let per_node = setup.gradient(base);
            let mut gradients: std::collections::HashMap<VariableId, Vec<f64>> =
                std::collections::HashMap::new();
            for &i in order {
                if base[i].is_empty() {
                    continue;
                }
                let variable = indexer.key_of(i as u32).expect("unfolded node").variable;
                let entry = gradients.entry(variable).or_insert_with(|| vec![0.0; base[i].len()]);
                for k in 0..base[i].len().min(entry.len()) {
                    entry[k] += per_node[i][k];
                }
            }
            gradients
        });
        // A level also reads the sample means of its exogenous roots, held fixed in the
        // g-formula. Their sampling error is a first-order (delta-method) shift of the
        // level: the score of the root means combined by their gradient, whose long-run
        // variance sets the shift's standard deviation.
        let root_mean_se = ols_base
            .as_ref()
            .and_then(|base| setup.root_mean_scores(base))
            .map_or(0.0, |scores| crate::serial_dependence::mean_standard_error(&scores));
        let mut mechanism_posts = vec![None; setup.node_count];
        let mut mechanism_of = vec![0; setup.node_count];
        let mut mechanisms: Vec<(VariableId, Vec<LaggedColumn>, usize)> = Vec::new();
        let mut count = usize::MAX;
        for &i in order {
            if designs[i].is_some() {
                let child = indexer.key_of(i as u32).expect("unfolded node");
                let parent_columns: Vec<_> = parents[i]
                    .iter()
                    .map(|&p| {
                        let parent = indexer.key_of(p as u32).expect("unfolded parent");
                        LaggedColumn {
                            variable: parent.variable,
                            lag: Lag::from_raw(
                                u32::try_from(child.offset - parent.offset).expect("causal lag"),
                            ),
                        }
                    })
                    .collect();
                if let Some((_, columns, owner)) =
                    mechanisms.iter().find(|(v, _, _)| *v == child.variable)
                {
                    if columns != &parent_columns {
                        return Err(EstimationError::unsupported(
                            "stationary mechanism has inconsistent unfolded parents",
                        ));
                    }
                    mechanism_of[i] = *owner;
                    continue;
                }
                // Fit each stationary mechanism once on its unique observed
                // time-series rows. Overlapping unfolded windows must not
                // multiply the likelihood or create independent copies of beta.
                let mut columns = parent_columns.clone();
                columns.push(LaggedColumn { variable: child.variable, lag: Lag::CONTEMPORANEOUS });
                let max_lag = columns.iter().map(|c| c.lag.raw()).max().unwrap_or(0);
                let plan = data.plan_lagged_sample(max_lag, Arc::from(columns))?;
                let mut workspace = LaggedSampleWorkspace::default();
                let sample = plan.prepare(data, &mut workspace, &ctx.kernel_policy)?;
                let covs: Vec<_> = parent_columns
                    .iter()
                    .enumerate()
                    .skip(1)
                    .map(|(j, c)| (c.variable, sample.column(j)))
                    .collect();
                let design = CompiledDesign::linear_adjustment(
                    sample.column(0),
                    &covs,
                    sample.column(parent_columns.len()),
                    &[],
                )?;
                let prep = PreparedBayesianProblem {
                    design,
                    method: estimand.method.clone(),
                    adjustment_set: Arc::from([]),
                    active: 1.0,
                    control: 0.0,
                    overlap: OverlapPolicy::ExplicitOverride,
                    coef_names: None,
                    unit_ids: None,
                    // Rows are time-ordered, so the stated model is the serial-dependence-
                    // corrected generalized posterior for the Sustained contrast and the
                    // Sequence level alike, each tempered along its own gradient.
                    serial_dependence: {
                        // Without an OLS gradient (rank-deficient unfolded design) the
                        // sum of slopes stands in for the direction; a level also reads
                        // every intercept.
                        let direction = contrast_gradient
                            .as_ref()
                            .and_then(|g| g.get(&child.variable).cloned())
                            .unwrap_or_else(|| {
                                let intercept = match eval {
                                    SequentialEval::Contrast { .. } => 0.0,
                                    SequentialEval::Level => 1.0,
                                };
                                (0..=parent_columns.len())
                                    .map(|k| if k == 0 { intercept } else { 1.0 })
                                    .collect()
                            });
                        SerialDependence::LongRunTempering(DependenceScope::Direction(Arc::from(
                            direction,
                        )))
                    },
                };
                let mut est = estimator.clone();
                est.seed = est.seed.wrapping_add(u64::from(child.variable.raw()));
                let post = est.fit(&prep, status, &mut BayesianGCompWorkspace::default(), ctx)?;
                if let Some(contexts) = validation.as_deref_mut() {
                    contexts.push(SequentialBayesianMechanism {
                        variable: child.variable,
                        prepared: prep.clone(),
                        posterior: post.clone(),
                    });
                }
                count = count.min(post.draws.n_draws);
                mechanism_posts[i] = Some(post);
                mechanism_of[i] = i;
                mechanisms.push((child.variable, parent_columns, i));
            }
        }
        let mut posterior = mechanism_posts.iter().flatten().next().cloned().ok_or_else(|| {
            EstimationError::unsupported("sustained contrast has no fitted outcome mechanism")
        })?;
        let mut coefficient_columns = vec![Vec::new(); setup.node_count];
        let mut coefficients = vec![Vec::new(); setup.node_count];
        for &i in order {
            let Some(design) = &designs[i] else {
                continue;
            };
            let post = mechanism_posts[mechanism_of[i]].as_ref().expect("stationary fit");
            coefficient_columns[i] = (0..design.ncols)
                .map(|index| {
                    post.draws
                        .schema
                        .quantities
                        .iter()
                        .position(|q| {
                            matches!(
                                q,
                                PosteriorQuantityKind::Coefficient { index: j, .. }
                                    if *j == index
                            )
                        })
                        .ok_or_else(|| EstimationError::stats_msg("missing sequential coefficient"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            coefficients[i].resize(design.ncols, 0.0);
        }
        let mut root_rng = CausalRng::from_seed(estimator.seed ^ ROOT_MEAN_STREAM);
        let mut values = Vec::with_capacity(count);
        for draw in 0..count {
            for &i in order {
                if designs[i].is_some() {
                    let post = mechanism_posts[mechanism_of[i]].as_ref().expect("stationary fit");
                    for (value, &column) in coefficients[i].iter_mut().zip(&coefficient_columns[i])
                    {
                        *value = post.draws.column(column)?[draw];
                    }
                }
            }
            let level = setup.propagate(&coefficients, &setup.factual);
            values.push(if root_mean_se > 0.0 {
                level + root_mean_se * antecedent_kernels::standard_normal(&mut root_rng)
            } else {
                level
            });
        }
        let draws = PosteriorDraws::from_column_major(
            PosteriorSchema {
                quantities: Arc::from([PosteriorQuantityKind::Effect {
                    name: Arc::from("sustained_window"),
                }]),
            },
            count,
            values,
        )?;
        posterior.summaries = draws.summarize();
        posterior.draws = draws;
        for post in mechanism_posts.iter().flatten().skip(1) {
            posterior.assumptions.entries.extend(post.assumptions.entries.iter().cloned());
            for note in &post.diagnostics.notes {
                if !posterior.diagnostics.notes.contains(note) {
                    posterior.diagnostics.notes.push(Arc::clone(note));
                }
            }
        }
        if matches!(eval, SequentialEval::Level) {
            posterior.assumptions.entries.push(AssumptionRecord {
                assumption: Assumption::ParametricRestriction(ParametricAssumption {
                    id: "temporal.sequential.level_posterior".into(),
                    description: "each stationary mechanism's likelihood is tempered by the \
                        long-run-variance ratio of the level's own gradient (rows are \
                        time-ordered), and the sample means of exogenous root nodes the level \
                        reads enter as a first-order (delta-method) normal shift whose \
                        variance is the long-run variance of their gradient-weighted scores"
                        .into(),
                }),
                source: AssumptionSource::AlgorithmDefault {
                    algorithm: "temporal.sequential.linear_sem".into(),
                },
                scope: AssumptionScope::Estimation,
                status: AssumptionStatus::Declared,
            });
        }
        posterior.assumptions.entries.extend(assumptions.entries);
        let effect = EffectEstimate::new(
            posterior.summaries.mean[0],
            posterior.summaries.sd[0],
            posterior.assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        );
        return Ok((effect, Some(posterior)));
    }
    let point = setup.evaluate(None, &mut ls_ws)?;
    // Circular blocks of consecutive lag-aligned rows (every row keeps its intact
    // unfolded window), every mechanism refit on the same rows, and the replicate
    // SD scaled by the circular-Bartlett fixed-b factor and the kernel-bias
    // factor of the contrast influence, as on the single-window path
    // ([`crate::temporal_block`]). Blocks are at least the unfolded span and
    // lengthen when the contrast's estimating score, or any mechanism's
    // normal-equation score, is persistently dependent. The influence and block
    // length are returned on the estimate so callers can report the resampling
    // geometry without rebuilding the design. The lengthening refits every
    // mechanism for its scores and scans each; without replicates no interval
    // is published and the rule length is returned instead.
    let influence = setup.influence(&mut ls_ws);
    let block_length = if bootstrap_replicates > 0 {
        setup.dependence_block_length_with(influence.as_deref())
    } else {
        antecedent_data::circular_block_length(setup.max_lag as usize + 1, setup.n)
    };
    let target: Vec<&[f64]> = influence.as_deref().into_iter().collect();
    let boot = crate::temporal_block::row_block_bootstrap_vec(
        setup.n,
        block_length,
        bootstrap_replicates,
        SEQUENTIAL_BLOCK_STREAM,
        ctx,
        |rows| setup.evaluate(Some(rows), &mut ls_ws).ok().map(|value| vec![value]),
    )
    .with_kernel_bias(&target);
    let kernel_bias = boot.kernel_bias;
    let se = boot.se_result(0);
    let mut effect = EffectEstimate::from_parts(
        point,
        f64::NAN,
        se.se,
        (bootstrap_replicates > 0).then_some(se.replicates_ok),
        (bootstrap_replicates > 0).then_some(se.replicates_failed),
        ctx.cancellation.is_cancelled(),
        false,
        assumptions,
        OverlapPolicy::ExplicitOverride,
        None,
        None,
    );
    effect.influence = influence.map(Arc::from);
    effect.block_resampling =
        Some(crate::adjustment::BlockResampling { block_length, rows: setup.n, kernel_bias });
    Ok((effect, None))
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

    fn hard_overlays() -> Vec<SequentialMechanismOverlay> {
        [(-2, 0.5), (-1, -0.25)]
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
            &data,
            &graph,
            &id_res.indexer,
            &estimand,
            VariableId::from_raw(1),
            0,
            &overlays,
            IdentificationStatus::NonparametricallyIdentified,
            AssumptionSet::new(),
            0,
            Some(&bayes),
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
        let ctx = ExecutionContext::for_tests(5);
        let build = |eval| {
            SequentialSetup::build(
                &data,
                &graph,
                &id_res.indexer,
                VariableId::from_raw(1),
                0,
                &overlays,
                eval,
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
        let influence = setup.influence(&mut ws).unwrap();
        assert_eq!(influence.len(), scores.len());
        assert!(influence.iter().sum::<f64>().abs() < 1e-6, "influence is mean zero");
        let contrast = build(SequentialEval::Contrast { delta: 1.0 });
        assert!(contrast.root_mean_gradient(&base).is_empty());
    }
}
