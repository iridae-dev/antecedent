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
    clippy::cast_precision_loss,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, ExecutionContext, IdentificationStatus, Lag, TargetPopulation,
    TemporalEffectQuery, TemporalNodeKey, TemporalPolicy, VariableId,
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
/// replicate SD scaled by the Kiefer–Vogelsang fixed-b factor
/// ([`crate::temporal_block`]). Bayesian uncertainty uses independent Gaussian priors across
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
        if rows.iter().any(|&r| r >= self.setup.n) {
            return Err(EstimationError::data_msg("aligned row index out of range"));
        }
        self.setup.evaluate(Some(rows), &mut LeastSquaresWorkspace::default())
    }

    /// Per-row influence of the contrast (OLS scores of every mechanism mapped
    /// through the contrast gradient), for [`crate::temporal_block::effective_rows`].
    #[must_use]
    pub fn influence(&self) -> Option<Vec<f64>> {
        self.setup.influence(&mut LeastSquaresWorkspace::default())
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
            SequentialEval::Level => {
                let mut values = vec![0.0; self.node_count];
                for &i in &self.order {
                    let natural = if coefficients[i].is_empty() {
                        factual[i]
                    } else {
                        coefficients[i][0]
                            + self.parents[i]
                                .iter()
                                .enumerate()
                                .map(|(p, &node)| coefficients[i][p + 1] * values[node])
                                .sum::<f64>()
                    };
                    values[i] = match self.overlay_at.get(i).copied().flatten() {
                        Some(overlay) => overlay.assigned(natural),
                        None => natural,
                    };
                }
                values[self.outcome]
            }
        }
    }

    /// OLS coefficients of every fitted mechanism on all aligned rows, or on the
    /// row map `rows` (any length).
    fn fit_ols(
        &self,
        rows: Option<&[usize]>,
        ls_ws: &mut LeastSquaresWorkspace,
    ) -> Result<Vec<Vec<f64>>, EstimationError> {
        let n = self.n;
        let mut coefficients = vec![Vec::new(); self.node_count];
        for &i in &self.order {
            if let Some(design) = &self.designs[i] {
                let (matrix, y) = if let Some(rows) = rows {
                    (
                        design
                            .matrix
                            .chunks(n)
                            .flat_map(|column| rows.iter().map(|&r| column[r]))
                            .collect::<Vec<_>>(),
                        rows.iter().map(|&r| design.outcome[r]).collect::<Vec<_>>(),
                    )
                } else {
                    (design.matrix.to_vec(), design.outcome.to_vec())
                };
                coefficients[i] = FaerBackend
                    .least_squares(&matrix, y.len(), design.ncols, &y, ls_ws)?
                    .coefficients;
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
        let coefficients = self.fit_ols(rows, ls_ws)?;
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

    /// Per-row influence of the propagated value: `Σ_i n·x_{i,r}ᵀ (X_iᵀX_i)⁻¹ g_i · e_{i,r}`
    /// over fitted mechanisms `i` with contrast gradient `g_i`. Used only for the
    /// effective-row count of the estimating score, so its scale is immaterial.
    fn influence(&self, ls_ws: &mut LeastSquaresWorkspace) -> Option<Vec<f64>> {
        let base = self.fit_ols(None, ls_ws).ok()?;
        let gradient = self.gradient(&base);
        let n = self.n;
        let mut score = vec![0.0; n];
        for &i in &self.order {
            let Some(design) = &self.designs[i] else {
                continue;
            };
            let p = design.ncols;
            let column = |c: usize| &design.matrix[c * n..(c + 1) * n];
            let mut xtx = vec![0.0; p * p];
            for a in 0..p {
                for b in 0..=a {
                    let v: f64 = column(a).iter().zip(column(b)).map(|(x, y)| x * y).sum();
                    xtx[a * p + b] = v;
                    xtx[b * p + a] = v;
                }
            }
            let v = solve_symmetric_positive(&xtx, p, &gradient[i])?;
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
}

/// Solve `A x = b` for a small symmetric positive-definite `A` (row-major `p×p`)
/// by Cholesky; `None` when `A` is not numerically positive definite.
#[allow(clippy::many_single_char_names, clippy::needless_range_loop)]
fn solve_symmetric_positive(a: &[f64], p: usize, b: &[f64]) -> Option<Vec<f64>> {
    let mut l = vec![0.0; p * p];
    for i in 0..p {
        for j in 0..=i {
            let mut sum = a[i * p + j];
            for k in 0..j {
                sum -= l[i * p + k] * l[j * p + k];
            }
            if i == j {
                if sum <= 0.0 || !sum.is_finite() {
                    return None;
                }
                l[i * p + i] = sum.sqrt();
            } else {
                l[i * p + j] = sum / l[j * p + j];
            }
        }
    }
    let mut y = vec![0.0; p];
    for i in 0..p {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[i * p + k] * y[k];
        }
        y[i] = sum / l[i * p + i];
    }
    let mut x = vec![0.0; p];
    for i in (0..p).rev() {
        let mut sum = y[i];
        for k in i + 1..p {
            sum -= l[k * p + i] * x[k];
        }
        x[i] = sum / l[i * p + i];
    }
    Some(x)
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
        // Sustained contrasts temper each stationary mechanism by the long-run-variance
        // ratio of the contrast's own linear combination of that mechanism's
        // coefficients: the gradient of the composed contrast, summed over the
        // mechanism's unfolded time copies (coefficient positions are stable across
        // copies). The contrast is linear in each coefficient, so central differences
        // at the OLS fit are exact up to rounding.
        let contrast_gradient = match eval {
            SequentialEval::Contrast { .. } => setup.fit_ols(None, &mut ls_ws).ok().map(|base| {
                let per_node = setup.gradient(&base);
                let mut gradients: std::collections::HashMap<VariableId, Vec<f64>> =
                    std::collections::HashMap::new();
                for &i in order {
                    if base[i].is_empty() {
                        continue;
                    }
                    let variable = indexer.key_of(i as u32).expect("unfolded node").variable;
                    let entry =
                        gradients.entry(variable).or_insert_with(|| vec![0.0; base[i].len()]);
                    for k in 0..base[i].len().min(entry.len()) {
                        entry[k] += per_node[i][k];
                    }
                }
                gradients
            }),
            SequentialEval::Level => None,
        };
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
                    // Sustained contrasts are licensed cells whose stated model is the
                    // serial-dependence-corrected generalized posterior; Sequence levels
                    // keep the disclosed iid likelihood.
                    serial_dependence: match eval {
                        SequentialEval::Contrast { .. } => {
                            // Without an OLS gradient (rank-deficient unfolded design) the
                            // sum of slopes stands in for the contrast direction.
                            let direction = contrast_gradient
                                .as_ref()
                                .and_then(|g| g.get(&child.variable).cloned())
                                .unwrap_or_else(|| {
                                    (0..=parent_columns.len())
                                        .map(|k| if k == 0 { 0.0 } else { 1.0 })
                                        .collect()
                                });
                            SerialDependence::LongRunTempering(DependenceScope::Direction(
                                Arc::from(direction),
                            ))
                        }
                        SequentialEval::Level => SerialDependence::Iid,
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
            values.push(setup.propagate(&coefficients, &setup.factual));
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
    // SD scaled by the Kiefer–Vogelsang fixed-b factor, as on the single-window
    // path ([`crate::temporal_block`]). Blocks are at least the unfolded span and
    // lengthen when the contrast's estimating score is persistently dependent.
    let block_length = if bootstrap_replicates > 0 {
        let influence = setup.influence(&mut ls_ws);
        let scores: Vec<&[f64]> = influence.as_deref().into_iter().collect();
        crate::temporal_block::dependence_block_length(setup.max_lag as usize + 1, setup.n, &scores)
    } else {
        0
    };
    let boot = crate::temporal_block::row_block_bootstrap_vec(
        setup.n,
        block_length,
        bootstrap_replicates,
        SEQUENTIAL_BLOCK_STREAM,
        ctx,
        |rows| setup.evaluate(Some(rows), &mut ls_ws).ok().map(|value| vec![value]),
    );
    let se = boot.se_result(0);
    Ok((
        EffectEstimate::from_parts(
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
        ),
        None,
    ))
}
