//! Observed-data Bayesian Gaussian temporal mechanisms.
//!
//! Data augmentation samples selected/missing and censored outcomes, conditional
//! on every Gaussian child equation as well as the outcome equation. Stationary
//! coefficients are shared across time. Under the declared observation
//! independence and distinct nuisance parameters/priors, the selection-indicator
//! or censoring-bound likelihood factors out of the outcome posterior. Censoring
//! event inequalities remain in the outcome likelihood; no estimated IPCW weights or
//! corrected pseudo-outcomes are treated as complete Gaussian data.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::float_cmp,
    clippy::doc_markdown,
    clippy::cast_sign_loss,
    clippy::needless_range_loop
)]

use crate::{
    BayesianGComputationAte, CausalPosterior, EstimationError, SequentialMechanismOverlay,
    SequentialNodeOverlay, TemporalInterventionPlan, plan_from_response_query,
};
use antecedent_core::{
    Assumption, AssumptionRecord, AssumptionScope, AssumptionSet, AssumptionSource,
    AssumptionStatus, CausalResponse, CausalRng, Diagnostic, DiagnosticKind, DiagnosticSeverity,
    ExecutionContext, HorizonIdentification, IdentificationStatus, ObservationSpec,
    ParametricAssumption, ResponseFunctional, ResponseIdentification, ResponseQuery,
    ResponseUncertainty, ResponseValue, SupportDiagnostic, SupportRegion, SupportReport,
    SupportStatus, VariableId,
};
use antecedent_data::{TableView, TemporalIndexer, TimeSeriesData};
use antecedent_expr::IdentifiedEstimand;
use antecedent_graph::{DenseNodeId, TemporalDag};
use antecedent_kernels::standard_normal;
use antecedent_prob::{
    BayesDesignRef, BayesFitOptions, BayesLikelihood, GaussianCoefficientPrior,
    HessianFactorization, InferenceDiagnostics, LaplaceWorkspace, PosteriorDraws,
    PosteriorQuantityKind, PosteriorSchema, PriorSet, PriorSpec, fit_conjugate_gaussian,
    mcmc_summary,
};
use std::{collections::BTreeMap, sync::Arc};

#[derive(Clone, Copy, Debug)]
enum Observation {
    Exact(f64),
    Missing,
    Lower(f64),
    Upper(f64),
}

struct Mechanism {
    variable: VariableId,
    parents: Vec<(VariableId, usize)>,
    history: usize,
    beta: Vec<f64>,
    variance: f64,
    matrix: Vec<f64>,
    design_initialized: bool,
    design_varies: bool,
    y: Vec<f64>,
    prior: PriorSet,
    workspace: LaplaceWorkspace,
}
impl Mechanism {
    fn mean(&self, data: &BTreeMap<VariableId, Vec<f64>>, row: usize) -> f64 {
        self.beta[0]
            + self
                .parents
                .iter()
                .enumerate()
                .map(|(j, (v, lag))| self.beta[j + 1] * data[v][row - lag])
                .sum::<f64>()
    }
    fn draw(
        &mut self,
        data: &BTreeMap<VariableId, Vec<f64>>,
        rng: &mut CausalRng,
    ) -> Result<(), EstimationError> {
        let n = self.y.len();
        if !self.design_initialized || self.design_varies {
            self.matrix[..n].fill(1.0);
            for (j, (v, lag)) in self.parents.iter().enumerate() {
                self.matrix[(j + 1) * n..(j + 2) * n]
                    .copy_from_slice(&data[v][self.history - lag..self.history - lag + n]);
            }
            // X changes only when a parent is the augmented latent outcome.
            // The conjugate backend always recomputes X'y and y'y, so its X'X
            // cache remains valid when only the response changes between sweeps.
            self.workspace.invalidate_design();
            self.design_initialized = true;
        }
        self.y.copy_from_slice(&data[&self.variable][self.history..]);
        let fit = fit_conjugate_gaussian(
            BayesDesignRef {
                x_colmajor: &self.matrix,
                nrows: n,
                ncols: self.beta.len(),
                y: &self.y,
                weights: None,
                offsets: None,
            },
            &self.prior,
            &BayesFitOptions { n_draws: 1, seed: rng.next_u64(), ..BayesFitOptions::default() },
            &mut self.workspace,
        )?;
        for j in 0..self.beta.len() {
            self.beta[j] = fit.draws.column(j)?[0];
        }
        self.variance = fit.draws.column(self.beta.len())?[0];
        Ok(())
    }
}

fn read_column(data: &TimeSeriesData, variable: VariableId) -> Result<Vec<f64>, EstimationError> {
    let mut values = data.float64_values(variable)?;
    let column = data.column(variable)?;
    for (i, value) in values.iter_mut().enumerate() {
        if !column.validity().is_valid(i) {
            *value = f64::NAN;
        }
    }
    Ok(values)
}

fn observations(
    data: &TimeSeriesData,
    query: &ResponseQuery,
    outcome: VariableId,
) -> Result<Vec<Observation>, EstimationError> {
    query.require_licensed_temporal_observation()?;
    let (latent, observed, indicator, censor) = match query.observation {
        ObservationSpec::Selected { latent, observed, indicator } => {
            (latent, observed, indicator, None)
        }
        ObservationSpec::RightCensored { latent, observed, event, censoring } => {
            (latent, observed, event, Some((censoring, true)))
        }
        ObservationSpec::LeftCensored { latent, observed, event, censoring } => {
            (latent, observed, event, Some((censoring, false)))
        }
        _ => {
            return Err(EstimationError::unsupported(
                "observed temporal posterior needs selection or one-sided censoring",
            ));
        }
    };
    if latent != outcome {
        return Err(EstimationError::unsupported(
            "observation latent must be the response outcome",
        ));
    }
    let y = read_column(data, observed)?;
    let flags = read_column(data, indicator)?;
    let bounds = censor.map(|(v, _)| read_column(data, v)).transpose()?;
    let mut result = Vec::with_capacity(y.len());
    for i in 0..y.len() {
        if flags[i] != 0.0 && flags[i] != 1.0 {
            return Err(EstimationError::data_msg(
                "observation indicator must be complete binary data",
            ));
        }
        if flags[i] == 1.0 {
            if !y[i].is_finite() {
                return Err(EstimationError::data_msg("observed outcome must be finite"));
            }
            if let Some((_, right)) = censor {
                let bound = bounds.as_ref().expect("censoring column")[i];
                if !bound.is_finite() || (right && y[i] > bound) || (!right && y[i] < bound) {
                    return Err(EstimationError::data_msg(
                        "event outcome contradicts its censoring bound",
                    ));
                }
            }
            result.push(Observation::Exact(y[i]));
        } else if let Some((_, right)) = censor {
            let bound = bounds.as_ref().expect("censoring column")[i];
            if !bound.is_finite() {
                return Err(EstimationError::data_msg("censoring bound must be finite"));
            }
            result.push(if right { Observation::Lower(bound) } else { Observation::Upper(bound) });
        } else {
            result.push(Observation::Missing);
        }
    }
    Ok(result)
}

// Robert's exponential rejection proposal avoids inverse-CDF cancellation in
// the normal tail; ordinary rejection has acceptance at least 1/2 for a <= 0.
fn lower_normal(a: f64, rng: &mut CausalRng) -> Result<f64, EstimationError> {
    if !a.is_finite() {
        return Err(EstimationError::stats_msg("nonfinite standardized censoring bound"));
    }
    for _ in 0..100_000 {
        if a <= 0.0 {
            let z = standard_normal(rng);
            if z >= a {
                return Ok(z);
            }
        } else {
            let rate = 0.5 * a + 0.5 * a.hypot(2.0);
            let z = a - rng.next_f64().max(f64::MIN_POSITIVE).ln() / rate;
            if rng.next_f64() <= (-0.5 * (z - rate).powi(2)).exp() {
                return Ok(z);
            }
        }
    }
    Err(EstimationError::stats_msg("truncated Gaussian sampler failed to accept"))
}
fn latent_draw(
    mean: f64,
    sd: f64,
    observation: Observation,
    rng: &mut CausalRng,
) -> Result<f64, EstimationError> {
    let value = match observation {
        Observation::Exact(value) => value,
        Observation::Missing => mean + sd * standard_normal(rng),
        Observation::Lower(bound) => {
            (mean + sd * lower_normal((bound - mean) / sd, rng)?).max(bound)
        }
        Observation::Upper(bound) => {
            (mean - sd * lower_normal((mean - bound) / sd, rng)?).min(bound)
        }
    };
    if value.is_finite() {
        Ok(value)
    } else {
        Err(EstimationError::stats_msg("nonfinite latent outcome draw"))
    }
}

type MechanismTemplates = BTreeMap<VariableId, Vec<(VariableId, usize)>>;

// One canonical order for coefficient fitting and compiled response evaluation.
fn mechanism_templates(
    graph: &TemporalDag,
    outcome: VariableId,
) -> Result<MechanismTemplates, EstimationError> {
    let mut parents: BTreeMap<VariableId, Vec<(VariableId, usize)>> = BTreeMap::new();
    parents.insert(outcome, Vec::new());
    for node in 0..graph.node_count() {
        let child = graph.temporal_key(DenseNodeId::from_raw(node as u32)).expect("temporal node");
        if child.offset != 0 {
            continue;
        }
        let entry = parents.entry(child.variable).or_default();
        for &p in graph.parents(DenseNodeId::from_raw(node as u32)) {
            let parent = graph.temporal_key(p).expect("temporal parent");
            entry.push((
                parent.variable,
                parent.offset.checked_neg().and_then(|lag| usize::try_from(lag).ok()).ok_or_else(
                    || EstimationError::unsupported("invalid parent lag in temporal model"),
                )?,
            ));
        }
        entry.sort();
    }
    parents.retain(|v, p| *v == outcome || !p.is_empty());
    Ok(parents)
}

fn mechanisms(
    data: &TimeSeriesData,
    graph: &TemporalDag,
    outcome: VariableId,
    scale: f64,
) -> Result<Vec<Mechanism>, EstimationError> {
    let parents = mechanism_templates(graph, outcome)?;
    parents
        .into_iter()
        .map(|(variable, parents)| {
            let history = parents.iter().map(|(_, lag)| *lag).max().unwrap_or(0);
            let n = data
                .row_count()
                .checked_sub(history)
                .filter(|&n| n > parents.len() + 2)
                .ok_or_else(|| {
                    EstimationError::data_msg("too few time rows for observed-data mechanism")
                })?;
            let p = parents.len() + 1;
            let mut prior = PriorSet::weakly_informative(p);
            prior.specs.retain(|s| !matches!(s, PriorSpec::GaussianCoefficients(_)));
            prior.specs.push(PriorSpec::GaussianCoefficients(GaussianCoefficientPrior::isotropic(
                p, scale,
            )));
            let design_varies = parents.iter().any(|(v, _)| *v == outcome);
            Ok(Mechanism {
                variable,
                parents,
                history,
                beta: vec![0.0; p],
                variance: 1.0,
                matrix: vec![0.0; n * p],
                design_initialized: false,
                design_varies,
                y: vec![0.0; n],
                prior,
                workspace: LaplaceWorkspace::default(),
            })
        })
        .collect()
}

fn sweep(
    data: &mut BTreeMap<VariableId, Vec<f64>>,
    outcome: VariableId,
    observations: &[Observation],
    models: &[Mechanism],
    initial_variance: f64,
    rng: &mut CausalRng,
) -> Result<(), EstimationError> {
    let owner = models.iter().find(|m| m.variable == outcome).expect("outcome mechanism");
    for (row, &observation) in observations.iter().enumerate() {
        if matches!(observation, Observation::Exact(_)) {
            continue;
        }
        let (mut precision, mut linear) = if row >= owner.history {
            (1.0 / owner.variance, owner.mean(data, row) / owner.variance)
        } else {
            (1.0 / initial_variance, 0.0)
        };
        // All observed and latent descendants contribute to the conditional.
        for model in models {
            for (j, &(v, lag)) in model.parents.iter().enumerate() {
                let child = row + lag;
                if v != outcome || child < model.history || child >= observations.len() {
                    continue;
                }
                let b = model.beta[j + 1];
                let without = model.mean(data, child) - b * data[&outcome][row];
                precision += b * b / model.variance;
                linear += b * (data[&model.variable][child] - without) / model.variance;
            }
        }
        let value = latent_draw(linear / precision, 1.0 / precision.sqrt(), observation, rng)?;
        data.get_mut(&outcome).expect("outcome")[row] = value;
    }
    Ok(())
}

struct EvaluationNode {
    variable: VariableId,
    // Absent for exogenous variables and copies with any parent outside the
    // finite unfolding: these copies use the empirical boundary distribution.
    mechanism: Option<(usize, Vec<usize>)>,
    overlay: Option<SequentialMechanismOverlay>,
}
struct Evaluation {
    order: Vec<usize>,
    nodes: Vec<EvaluationNode>,
    outcome: usize,
}
impl Evaluation {
    fn new(
        graph: &TemporalDag,
        indexer: &TemporalIndexer,
        outcome: VariableId,
        horizon: u32,
        overlays: Vec<SequentialMechanismOverlay>,
    ) -> Result<Self, EstimationError> {
        let unfolded =
            graph.unfold(indexer.clone()).map_err(|e| EstimationError::data_msg(e.to_string()))?;
        let mut order: Vec<usize> = unfolded
            .dag
            .topological_order()
            .ok_or_else(|| EstimationError::data_msg("cyclic unfolding"))?
            .into_iter()
            .map(DenseNodeId::as_usize)
            .collect();
        let keys = (0..unfolded.dag.node_count())
            .map(|i| indexer.key_of(i as u32).map_err(|e| EstimationError::data_msg(e.to_string())))
            .collect::<Result<Vec<_>, _>>()?;
        let positions = keys
            .iter()
            .enumerate()
            .map(|(i, key)| ((key.variable, key.offset), i))
            .collect::<BTreeMap<_, _>>();
        let templates = mechanism_templates(graph, outcome)?
            .into_iter()
            .enumerate()
            .map(|(i, (v, p))| (v, (i, p)))
            .collect::<BTreeMap<_, _>>();
        let mut nodes = keys
            .iter()
            .map(|key| {
                let mechanism = templates.get(&key.variable).and_then(|(model, parents)| {
                    parents
                        .iter()
                        .map(|(variable, lag)| {
                            i32::try_from(*lag)
                                .ok()
                                .and_then(|lag| key.offset.checked_sub(lag))
                                .and_then(|offset| positions.get(&(*variable, offset)).copied())
                        })
                        .collect::<Option<Vec<_>>>()
                        .map(|parents| (*model, parents))
                });
                EvaluationNode { variable: key.variable, mechanism, overlay: None }
            })
            .collect::<Vec<_>>();
        for overlay in overlays {
            let position =
                positions.get(&(overlay.node.variable, overlay.node.offset)).ok_or_else(|| {
                    EstimationError::data_msg("intervention outside temporal unfolding")
                })?;
            if nodes[*position].overlay.replace(overlay).is_some() {
                return Err(EstimationError::data_msg(
                    "duplicate temporal intervention assignment",
                ));
            }
        }
        let offset = i32::try_from(horizon)
            .map_err(|_| EstimationError::data_msg("horizon overflow"))?
            .checked_sub(1)
            .ok_or_else(|| EstimationError::data_msg("horizon overflow"))?;
        let outcome = *positions
            .get(&(outcome, offset))
            .ok_or_else(|| EstimationError::data_msg("outcome outside temporal unfolding"))?;
        // Evaluate only dependencies of the requested cell. The indexer can
        // include unrelated schema columns without an empirical model/mean.
        let mut needed = vec![false; nodes.len()];
        let mut pending = vec![outcome];
        while let Some(i) = pending.pop() {
            if needed[i] {
                continue;
            }
            needed[i] = true;
            if nodes[i].overlay.is_some_and(|overlay| overlay.node.level.is_some()) {
                continue;
            }
            if let Some((_, parents)) = &nodes[i].mechanism {
                pending.extend(parents.iter().copied());
            }
        }
        order.retain(|&i| needed[i]);
        Ok(Self { order, nodes, outcome })
    }
    fn value(&self, models: &[Mechanism], means: &BTreeMap<VariableId, f64>) -> f64 {
        let mut values = vec![0.0; self.nodes.len()];
        for &i in &self.order {
            let node = &self.nodes[i];
            if let Some(overlay) = node.overlay.filter(|overlay| overlay.node.level.is_some()) {
                values[i] = overlay.assigned(0.0);
                continue;
            }
            let natural = node.mechanism.as_ref().map_or_else(
                || means[&node.variable],
                |(model, parents)| {
                    models[*model].beta[0]
                        + parents
                            .iter()
                            .enumerate()
                            .map(|(j, &parent)| models[*model].beta[j + 1] * values[parent])
                            .sum::<f64>()
                },
            );
            values[i] = node.overlay.map_or(natural, |overlay| overlay.assigned(natural));
        }
        values[self.outcome]
    }
}

/// Infer a temporal mean surface from the selected/censored Gaussian likelihood.
///
/// Four data-augmentation chains share each stationary mechanism across time.
/// At least 1024 retained sweeps per chain and 512 warmup sweeps are used; the
/// ordinary R-hat/ESS publication gate applies. Observation nuisance parameters
/// have distinct independent priors. Independence applies to the selection-indicator
/// or censoring-bound trajectory; censoring event inequalities remain in the likelihood.
/// Initial latent history has independent N(0, `prior_scale²`) priors. This is a
/// parametric observed-data model, not a Bayesian IPCW pseudo-likelihood.
///
/// # Errors
/// Invalid observation data, incompatible priors/likelihood, numerical failure,
/// cancellation, or failure of the MCMC publication gate.
pub fn estimate_observed_temporal_response(
    data: &TimeSeriesData,
    graph: &TemporalDag,
    identifications: &[(&IdentifiedEstimand, &TemporalIndexer)],
    query: &ResponseQuery,
    status: IdentificationStatus,
    mut assumptions: AssumptionSet,
    estimator: &BayesianGComputationAte,
    ctx: &ExecutionContext,
) -> Result<(CausalResponse, CausalPosterior), EstimationError> {
    query.validate()?;
    if ctx.cancellation.is_cancelled() {
        return Err(EstimationError::data_msg("observed temporal posterior cancelled"));
    }
    if estimator.likelihood != BayesLikelihood::GaussianIdentity || estimator.prior.is_some() {
        return Err(EstimationError::unsupported(
            "observed temporal mechanisms require Gaussian likelihood and independent isotropic mechanism priors",
        ));
    }
    if !estimator.prior_scale.is_finite() || estimator.prior_scale <= 0.0 {
        return Err(EstimationError::data_msg("invalid observed-mechanism prior scale"));
    }
    let temporal = query
        .temporal
        .as_ref()
        .ok_or_else(|| EstimationError::unsupported("temporal query required"))?;
    if identifications.len() != temporal.horizons.len()
        || !matches!(
            status,
            IdentificationStatus::NonparametricallyIdentified
                | IdentificationStatus::IdentifiedUnderParametricRestrictions
        )
    {
        return Err(EstimationError::unsupported(
            "observed temporal posterior requires every horizon identified",
        ));
    }
    let (outcome, treatment, doses) = match &query.functional {
        ResponseFunctional::MeanCurve { outcome, treatment } => {
            (*outcome, treatment.variable, Some(treatment.grid.values()?))
        }
        ResponseFunctional::InterventionResponse { outcome, interventions } => {
            (*outcome, interventions[0].target_variables()[0], None)
        }
        _ => {
            return Err(EstimationError::unsupported(
                "observed temporal posterior supports mean curves and intervention levels",
            ));
        }
    };
    let mut adjustment = Vec::new();
    for &(estimand, indexer) in identifications {
        for id in estimand.adjustment_set.iter() {
            adjustment.push(
                indexer
                    .key_of(id.raw())
                    .map_err(|e| EstimationError::data_msg(e.to_string()))?
                    .variable,
            );
        }
    }
    crate::observation::require_temporal_observation_containment(query, treatment, &adjustment)?;
    for assumption in query.observation_assumptions.iter() {
        let (antecedent_core::ObservationAssumption::IndependentGiven(conditioning)
        | antecedent_core::ObservationAssumption::OutcomeIndependentGiven(conditioning)) =
            assumption
        else {
            continue;
        };
        for &v in conditioning.iter() {
            if v == outcome || read_column(data, v)?.iter().any(|x| !x.is_finite()) {
                return Err(EstimationError::data_msg(
                    "observation conditioning must be fully observed and exclude the latent outcome",
                ));
            }
        }
    }
    let obs = observations(data, query, outcome)?;
    let mut base = BTreeMap::new();
    for node in 0..graph.node_count() {
        let v =
            graph.temporal_key(DenseNodeId::from_raw(node as u32)).expect("temporal node").variable;
        if v != outcome && !base.contains_key(&v) {
            let values = read_column(data, v)?;
            if values.iter().any(|v| !v.is_finite()) {
                return Err(EstimationError::data_msg(
                    "non-outcome model variables must be observed",
                ));
            }
            base.insert(v, values);
        }
    }
    let exact: Vec<_> = obs
        .iter()
        .filter_map(|o| if let Observation::Exact(y) = o { Some(*y) } else { None })
        .collect();
    if exact.len() < 3 {
        return Err(EstimationError::data_msg(
            "at least three observed outcomes required for temporal posterior",
        ));
    }
    let center = exact.iter().sum::<f64>() / exact.len() as f64;
    base.insert(
        outcome,
        obs.iter()
            .map(|o| match *o {
                Observation::Exact(y) => y,
                Observation::Lower(y) => center.max(y),
                Observation::Upper(y) => center.min(y),
                Observation::Missing => center,
            })
            .collect(),
    );
    let planned = plan_from_response_query(query)?;
    if planned
        .as_ref()
        .and_then(TemporalInterventionPlan::mechanism_overlays)
        .is_some_and(|overlays| overlays.iter().any(|overlay| overlay.bounds.is_some()))
    {
        assumptions.push(SequentialMechanismOverlay::mean_target_assumption());
    }

    let mut evaluations = Vec::new();
    let mut grid = Vec::new();
    for dose in doses.clone().unwrap_or_else(|| vec![0.0]) {
        for (&horizon, &(_, indexer)) in temporal.horizons.iter().zip(identifications) {
            let overlays = if let Some(plan) = &planned {
                if let Some(overlays) = plan.mechanism_overlays() {
                    overlays
                } else {
                    let TemporalInterventionPlan::Single { treatment, level, shift } = *plan else {
                        unreachable!()
                    };
                    temporal
                        .policy
                        .active_offsets()
                        .map_err(|e| EstimationError::data_msg(e.to_string()))?
                        .iter()
                        .map(|&at| {
                            SequentialNodeOverlay { variable: treatment, offset: at, level, shift }
                                .into()
                        })
                        .collect()
                }
            } else {
                temporal
                    .policy
                    .active_offsets()
                    .map_err(|e| EstimationError::data_msg(e.to_string()))?
                    .iter()
                    .map(|&at| {
                        SequentialNodeOverlay {
                            variable: treatment,
                            offset: at,
                            level: Some(dose),
                            shift: 0.0,
                        }
                        .into()
                    })
                    .collect()
            };
            evaluations.push(Evaluation::new(graph, indexer, outcome, horizon, overlays)?);
            if doses.is_some() {
                grid.push(dose);
            }
            grid.push(f64::from(horizon));
        }
    }
    let chains = 4;
    let per_chain = estimator.n_draws.div_ceil(chains).max(1024);
    let warmup = 512;
    let count = chains
        .checked_mul(per_chain)
        .ok_or_else(|| EstimationError::data_msg("posterior draw count overflow"))?;
    let cells = evaluations.len();
    // Inspect dimensions before allocating any mechanism design or workspace.
    // Chains run sequentially, so only one chain's mechanism buffers are live.
    let dimensions = mechanism_templates(graph, outcome)?
        .values()
        .map(|parents| {
            let history = parents.iter().map(|(_, lag)| *lag).max().unwrap_or(0);
            (data.row_count().saturating_sub(history), parents.len().saturating_add(1))
        })
        .collect::<Vec<_>>();
    let (n_parameters, bytes) =
        posterior_buffer_budget(&dimensions, data.row_count(), base.len(), count, cells)?;
    if ctx.memory.hard_limit_bytes.is_some_and(|limit| bytes as u128 > u128::from(limit)) {
        return Err(EstimationError::data_msg("observed temporal posterior exceeds memory budget"));
    }
    let mut draws = vec![0.0; count * cells];
    let mut parameters = Vec::with_capacity(count * n_parameters);
    for chain in 0..chains {
        let mut rng = CausalRng::from_seed(estimator.seed.wrapping_add(chain as u64 * 0x10001));
        let mut values = base.clone();
        let mut models = mechanisms(data, graph, outcome, estimator.prior_scale)?;
        for iteration in 0..warmup + per_chain {
            if ctx.cancellation.is_cancelled() {
                return Err(EstimationError::data_msg("observed temporal posterior cancelled"));
            }
            for model in &mut models {
                model.draw(&values, &mut rng)?;
            }
            sweep(&mut values, outcome, &obs, &models, estimator.prior_scale.powi(2), &mut rng)?;
            if iteration < warmup {
                continue;
            }
            for model in &models {
                parameters.extend_from_slice(&model.beta);
                parameters.push(model.variance);
            }
            let means =
                values.iter().map(|(&v, x)| (v, x.iter().sum::<f64>() / x.len() as f64)).collect();
            for (cell, evaluation) in evaluations.iter().enumerate() {
                let value = evaluation.value(&models, &means);
                draws[cell * count + chain * per_chain + iteration - warmup] = value;
                parameters.push(value);
            }
        }
    }
    let summary = mcmc_summary(&parameters, chains, per_chain, n_parameters);
    let mut diagnostics = InferenceDiagnostics::analytic("temporal.observed_gaussian.gibbs");
    diagnostics.factorization = HessianFactorization::Mcmc;
    diagnostics.n_chains = Some(chains as u32);
    diagnostics.n_warmup = Some(warmup as u32);
    diagnostics.iterations = per_chain as u32;
    diagnostics.rhat_max = Some(summary.max_rhat);
    diagnostics.ess_bulk_min = Some(summary.min_bulk_ess);
    diagnostics.ess_tail_min = Some(summary.min_tail_ess);
    diagnostics.n_postwarmup_divergences = Some(0);
    diagnostics.n_divergences = Some(0);
    diagnostics.n_warmup_divergences = Some(0);
    diagnostics.mean_accept_prob = Some(1.0);
    diagnostics.max_abs_delta_h = Some(0.0);
    diagnostics.notes.push("Exact Gibbs conditional updates have unit acceptance; no Hamiltonian integration is performed (zero integration-error convention).".into());
    diagnostics.all_chains_moved =
        Some(antecedent_prob::all_chains_moved(&parameters, chains, per_chain, n_parameters));
    diagnostics.converged = diagnostics.mcmc_publication_ok();
    if !diagnostics.allows_posterior() {
        return Err(EstimationError::stats_msg(format!(
            "observed temporal Gibbs publication gate: R-hat={}, bulk ESS={}, tail ESS={}",
            summary.max_rhat, summary.min_bulk_ess, summary.min_tail_ess
        )));
    }
    let observation_independence = if matches!(query.observation, ObservationSpec::Selected { .. })
    {
        "selection-indicator trajectory is independent of the full latent outcome trajectory"
    } else {
        "censoring-bound trajectory is independent of the full latent outcome trajectory; \
         the event indicator is determined by the outcome and censoring bound"
    };
    assumptions.push(AssumptionRecord {
        assumption: Assumption::ParametricRestriction(ParametricAssumption {
            id: "temporal.observed_gaussian_sem".into(),
            description: format!(
                "time-homogeneous linear Gaussian mechanisms with independent innovations \
                 and mechanism priors (no AR stability prior); {observation_independence}, \
                 conditional on the declared fully observed conditioning trajectory, with \
                 distinct independent nuisance priors; latent initial history has \
                 independent N(0, prior_scale²) priors"
            )
            .into(),
        }),
        source: AssumptionSource::AlgorithmDefault {
            algorithm: "temporal.observed_gaussian.gibbs".into(),
        },
        scope: AssumptionScope::Estimation,
        status: AssumptionStatus::Declared,
    });
    let schema = PosteriorSchema {
        quantities: (0..cells)
            .map(|cell| PosteriorQuantityKind::Scalar {
                name: format!("response.cell.{cell}").into(),
            })
            .collect::<Vec<_>>()
            .into(),
    };
    let draws = PosteriorDraws::from_column_major(schema, count, draws)?;
    let summaries = draws.summarize();
    let mut lower = Vec::new();
    let mut upper = Vec::new();
    for cell in 0..cells {
        let mut x = draws.column(cell)?.to_vec();
        x.sort_by(f64::total_cmp);
        lower.push(quantile(&x, 0.025));
        upper.push(quantile(&x, 0.975));
    }
    let horizon_identification = temporal
        .horizons
        .iter()
        .zip(identifications)
        .map(|(&horizon, &(id, indexer))| {
            Ok(HorizonIdentification {
                horizon,
                status,
                method: id.method.clone(),
                adjustment: id
                    .adjustment_set
                    .iter()
                    .map(|v| {
                        indexer
                            .key_of(v.raw())
                            .map_err(|e| EstimationError::data_msg(e.to_string()))
                    })
                    .collect::<Result<Vec<_>, _>>()?
                    .into(),
            })
        })
        .collect::<Result<Vec<_>, EstimationError>>()?;
    let dimension = if doses.is_some() { 2 } else { 1 };
    let minima = (0..dimension)
        .map(|axis| grid.chunks(dimension).map(|point| point[axis]).fold(f64::INFINITY, f64::min))
        .collect::<Vec<_>>();
    let maxima = (0..dimension)
        .map(|axis| {
            grid.chunks(dimension).map(|point| point[axis]).fold(f64::NEG_INFINITY, f64::max)
        })
        .collect::<Vec<_>>();
    let response = CausalResponse {
        estimand: query.functional.clone(),
        identification_status: status,
        estimate: ResponseIdentification::PointIdentified(if doses.is_none() && cells == 1 {
            ResponseValue::Scalar(summaries.mean[0])
        } else {
            ResponseValue::Surface {
                grid: grid.into(),
                dimension,
                mean: Arc::clone(&summaries.mean),
            }
        }),
        uncertainty: ResponseUncertainty::PointwiseBand {
            level: 0.95,
            lower: lower.into(),
            upper: upper.into(),
        },
        support: SupportReport {
            status: SupportStatus::Extrapolative,
            query_region: SupportRegion { minima: minima.into(), maxima: maxima.into() },
            point_status: Some(vec![SupportStatus::Extrapolative; cells].into()),
            diagnostics: vec![SupportDiagnostic {
                id: "response.observation_posterior".into(),
                values: Arc::from([
                    count as f64,
                    summary.max_rhat,
                    summary.min_bulk_ess,
                    summary.min_tail_ess,
                ]),
                detail: "retained draws, maximum R-hat, minimum bulk ESS and tail ESS \
                         for observed-data Gaussian mechanism posterior"
                    .into(),
            }],
            warnings: vec![Diagnostic::new(
                "response.observation_gaussian_model",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "credible bands integrate censored/missing outcomes and mechanism parameters \
                 under a Gaussian temporal model; the selection-indicator or censoring-bound \
                 nuisance likelihood factors out under the declared independence and distinct \
                 priors, while censoring event inequalities remain in the outcome likelihood; \
                 empirical joint longitudinal support is unassessed",
            )],
        },
        assumptions: assumptions.clone(),
        provenance_id: "estimate.temporal_observed_bayes".into(),
        horizon_identification: Some(horizon_identification.into()),
        interaction_structurally_zero: false,
    };
    let posterior = CausalPosterior {
        draws,
        summaries,
        identification: status,
        prior_sensitivity: None,
        conflict_summary: None,
        diagnostics,
        assumptions,
        unidentified_mass: 0.0,
        early_stopped: false,
    };
    Ok((response, posterior))
}
// Conservative numeric-buffer peak, excluding caller-owned input. The dense
// allowance covers design, response, cached Gram/Hessian/factor matrices,
// conjugate posterior temporaries, and coefficient/prior/sampling scratch.
fn posterior_buffer_budget(
    dimensions: &[(usize, usize)],
    rows: usize,
    variables: usize,
    draws: usize,
    cells: usize,
) -> Result<(usize, usize), EstimationError> {
    let overflow = || EstimationError::data_msg("observed posterior buffer size overflow");
    let mut parameters = cells;
    let mut work = 0_usize;
    for &(n, p) in dimensions {
        parameters =
            parameters.checked_add(p).and_then(|v| v.checked_add(1)).ok_or_else(overflow)?;
        let dense = p.checked_mul(p).and_then(|v| v.checked_mul(8)).ok_or_else(overflow)?;
        let linear = p.checked_mul(16).ok_or_else(overflow)?;
        let design = p.checked_add(4).and_then(|v| v.checked_mul(n)).ok_or_else(overflow)?;
        work = work
            .checked_add(dense)
            .and_then(|v| v.checked_add(linear))
            .and_then(|v| v.checked_add(design))
            .ok_or_else(overflow)?;
    }
    // Base + chain data, observations/initialization, and diagnostic sorting,
    // ranks, split-chain and autocovariance scratch for one parameter at a time.
    work = variables
        .checked_mul(2)
        .and_then(|v| v.checked_add(4))
        .and_then(|v| v.checked_mul(rows))
        .and_then(|v| v.checked_add(work))
        .ok_or_else(overflow)?;
    let bytes = parameters
        .checked_add(cells)
        .and_then(|v| v.checked_add(24))
        .and_then(|v| v.checked_mul(draws))
        .and_then(|v| v.checked_add(work))
        .and_then(|v| v.checked_mul(std::mem::size_of::<f64>()))
        .ok_or_else(overflow)?;
    Ok((parameters, bytes))
}

#[cfg(test)]
mod memory_budget_tests {
    use super::*;
    #[test]
    fn preflight_accounts_for_dense_workspace_and_rejects_overflow() {
        let (parameters, bytes) = posterior_buffer_budget(&[(100, 20)], 100, 3, 4096, 2).unwrap();
        assert_eq!(parameters, 23);
        let posterior_only = 4096 * (parameters + 2) * 8;
        assert!(bytes > posterior_only + (100 * 20 + 8 * 20 * 20) * 8);
        assert!(posterior_buffer_budget(&[(100, usize::MAX)], 100, 3, 4096, 2).is_err());
        assert!(posterior_buffer_budget(&[(100, 20)], 100, 3, usize::MAX, 2).is_err());
    }
}

fn quantile(x: &[f64], p: f64) -> f64 {
    let pos = p * (x.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    x[lo] + (x[hi] - x[lo]) * (pos - lo as f64)
}

#[cfg(test)]
mod evaluation_boundary_tests {
    use super::*;
    use antecedent_core::Lag;
    use antecedent_graph::ensure_lagged;

    #[test]
    fn incomplete_history_is_empirical_before_forward_propagation() {
        let variable = VariableId::from_raw(0);
        let mut graph = TemporalDag::empty();
        let lagged = ensure_lagged(&mut graph, variable, Lag::from_raw(1)).unwrap();
        let current = ensure_lagged(&mut graph, variable, Lag::CONTEMPORANEOUS).unwrap();
        graph.insert_directed(lagged, current).unwrap();
        // Extra schema variable 1 has no data/model; it is not an ancestor.
        let indexer = TemporalIndexer::new(2, 1, 1).unwrap();
        let data = TimeSeriesData::from_f64_columns([("y", &[3.0; 10][..])], 1).unwrap();
        let mut models = mechanisms(&data, &graph, variable, 1.0).unwrap();
        models[0].beta = vec![2.0, 0.5];
        let means = BTreeMap::from([(variable, 3.0)]);
        let natural = Evaluation::new(&graph, &indexer, variable, 1, Vec::new()).unwrap();
        // Boundary y[-1] is empirical 3, not 2 + .5*3; only y[0] is propagated.
        assert!((natural.value(&models, &means) - 3.5).abs() < 1e-12);
        let intervention = Evaluation::new(
            &graph,
            &indexer,
            variable,
            1,
            vec![
                SequentialNodeOverlay { variable, offset: -1, level: Some(10.0), shift: 0.0 }
                    .into(),
            ],
        )
        .unwrap();
        assert!((intervention.value(&models, &means) - 7.0).abs() < 1e-12);
        let hard_outcome = Evaluation::new(
            &graph,
            &indexer,
            variable,
            1,
            vec![
                SequentialNodeOverlay { variable, offset: 0, level: Some(9.0), shift: 0.0 }.into(),
            ],
        )
        .unwrap();
        // A hard level needs neither its own natural mean nor any ancestors.
        assert_eq!(hard_outcome.order.len(), 1);
        assert!((hard_outcome.value(&[], &BTreeMap::new()) - 9.0).abs() < 1e-12);
    }
}

#[cfg(test)]
mod sampler_tests {
    use super::*;
    #[test]
    fn one_sided_normal_draws_match_half_normal_moments_and_retain_tails() {
        let mut rng = CausalRng::from_seed(79);
        let mut sum = 0.0;
        let mut second = 0.0;
        for _ in 0..40_000 {
            let z = lower_normal(0.0, &mut rng).unwrap();
            assert!(z >= 0.0);
            sum += z;
            second += z * z;
        }
        assert!((sum / 40_000.0 - (2.0 / std::f64::consts::PI).sqrt()).abs() < 0.012);
        assert!((second / 40_000.0 - 1.0).abs() < 0.025);
        let mut tail_mean = 0.0;
        for _ in 0..10_000 {
            let z = lower_normal(10.0, &mut rng).unwrap();
            assert!(z >= 10.0);
            tail_mean += z;
        }
        assert!((tail_mean / 10_000.0 - 10.098_093_234).abs() < 0.008);
    }
    #[test]
    fn latent_markov_blanket_includes_observed_future_child() {
        let outcome = VariableId::from_raw(0);
        let model = Mechanism {
            variable: outcome,
            parents: vec![(outcome, 1)],
            history: 1,
            beta: vec![0.0, 0.5],
            variance: 1.0,
            matrix: vec![],
            design_initialized: false,
            design_varies: true,
            y: vec![],
            prior: PriorSet::weakly_informative(2),
            workspace: LaplaceWorkspace::default(),
        };
        let obs = [Observation::Exact(1.0), Observation::Missing, Observation::Exact(3.0)];
        let mut data = BTreeMap::from([(outcome, vec![1.0, 0.0, 3.0])]);
        let mut rng = CausalRng::from_seed(94);
        let mut sum = 0.0;
        let mut second = 0.0;
        for _ in 0..20_000 {
            sweep(&mut data, outcome, &obs, std::slice::from_ref(&model), 100.0, &mut rng).unwrap();
            let value = data[&outcome][1];
            sum += value;
            second += value * value;
        }
        let mean = sum / 20_000.0;
        assert!((mean - 1.6).abs() < 0.02);
        assert!((second / 20_000.0 - mean * mean - 0.8).abs() < 0.025);
    }
}

#[cfg(test)]
mod design_cache_tests {
    use super::*;
    use antecedent_core::Lag;
    use antecedent_graph::ensure_lagged;

    #[test]
    fn cache_matches_fresh_fit_when_outcome_and_latent_predictors_change() {
        let id = VariableId::from_raw;
        let x = (0..24).map(f64::from).collect::<Vec<_>>();
        let y = x.iter().map(|x| 1.0 + 2.0 * x).collect::<Vec<_>>();
        let z = x.iter().map(|x| 3.0 - x).collect::<Vec<_>>();
        let data = TimeSeriesData::from_f64_columns(
            [("x", x.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())],
            1,
        )
        .unwrap();
        let mut graph = TemporalDag::empty();
        let nodes = (0..3)
            .map(|v| ensure_lagged(&mut graph, id(v), Lag::CONTEMPORANEOUS).unwrap())
            .collect::<Vec<_>>();
        graph.insert_directed(nodes[0], nodes[1]).unwrap();
        graph.insert_directed(nodes[1], nodes[2]).unwrap();
        let mut cached = mechanisms(&data, &graph, id(1), 2.0).unwrap();
        let mut fresh = mechanisms(&data, &graph, id(1), 2.0).unwrap();
        assert!(!cached[0].design_varies);
        assert!(cached[1].design_varies);
        let mut values = BTreeMap::from([(id(0), x), (id(1), y), (id(2), z)]);
        for iteration in 0..3 {
            for value in values.get_mut(&id(1)).unwrap() {
                *value += 0.75;
            }
            for (cached, fresh) in cached.iter_mut().zip(&mut fresh) {
                fresh.design_initialized = false;
                cached.draw(&values, &mut CausalRng::from_seed(iteration)).unwrap();
                fresh.draw(&values, &mut CausalRng::from_seed(iteration)).unwrap();
                assert_eq!(cached.beta, fresh.beta);
                assert_eq!(cached.variance.to_bits(), fresh.variance.to_bits());
            }
        }
    }
}
