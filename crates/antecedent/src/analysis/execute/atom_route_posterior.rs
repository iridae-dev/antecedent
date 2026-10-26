// SPDX-License-Identifier: MIT OR Apache-2.0
//! Derivative and multi-coordinate responses, ADMG conditional effects and counterfactuals over a
//! supplied graph posterior on tabular data.
//!
//! Every atom of the posterior is a structure: a DAG, an ADMG (or the contemporaneous projection
//! of a temporal ADMG), a CPDAG or a PAG. Each atom answers the query with the executor the same
//! query has on a supplied structure of that kind, so an atom's value, identification and
//! uncertainty are exactly those of the explicit cell: the continuous response estimator on a
//! DAG, the derivative identifier of a latent-confounded graph, the IDC contrast of an ADMG, the
//! structural-mechanism counterfactual of a DAG or class, the ID* plug-in of a latent-confounded
//! graph and the completion envelope of a class. The posterior moves only which structure
//! answers.
//!
//! The atoms are then mixed by posterior probability:
//!
//! * an atom whose identifier refuses the query is unidentified posterior mass, and an atom the
//!   identifier accepts but whose estimator cannot be evaluated is unevaluable mass; neither is
//!   redistributed, so the published value is conditional on the atoms that identify and
//!   evaluate and the result states the mass it does not cover;
//! * the atoms identify one target by structure-specific formulas (adjustment sets, district
//!   formulas, class completions), so the estimand an atom contributes is the target and the
//!   mixture is a posterior-weighted mean of the atoms' values;
//! * an atom that is a class whose completions disagree keeps its identified set, and the
//!   mixture publishes the union of the atoms' sets beside the point;
//! * a derivative, Jacobian or multi-coordinate level is mixed coordinate by coordinate, and each
//!   graph's own typed answer is retained beside its weight on the structural mixture.
//!
//! Uncertainty of the mixture of several atoms. Frequentist: when every atom reports its row
//! influence the mixture is the influence of the weighted sum (one shared sample); otherwise every
//! atom is re-run on the same resampled tables and the standard error is the spread of the
//! draw-wise mixture of the atoms' replicates. Bayesian: the atoms ran in one context, so draw
//! `d` of every atom is the same draw of the row weights, and the pooled posterior is the
//! draw-wise posterior-weighted mixture; atoms whose draws differ in count publish no aggregate
//! and keep their per-atom posteriors.

use std::collections::HashMap;

use antecedent_core::{ConditionalEffectQuery, CounterfactualQuery, ResponseQuery};
use antecedent_data::TableView;
use antecedent_discovery::adjacency_mask::{
    admg_from_adjacency_mask, cpdag_from_adjacency_mask, dag_from_adjacency_mask,
    pag_from_adjacency_mask,
};
use antecedent_discovery::{AdjacencyMask, GraphPosteriorAtomKind};

use super::atom_mix::{AtomBoundRule, AtomEvalView, AtomMixInput, MixedAtomEvals, mix_atom_evals};
use super::response_path::{
    mix_response_values, mix_support_reports, response_identified_value, response_primary_pair,
    response_scalar_summary,
};
use super::{
    AcceptedGraph, Admg, Arc, BootstrapAccounting, CausalError, CausalPosterior, CausalQuery, Dag,
    Diagnostic, DiagnosticKind, DiagnosticSeverity, EffectEstimate, ExecutionContext,
    GraphIdentFlag, GraphPosterior, IdentificationStatus, IdentifiedEstimand, IdentifiedKeyWeights,
    Intervention, OverlapPolicy, Pag, PhysicalExecutionPlan, PosteriorMassSummary, RefuteSuite,
    ResponseIdentification, ResponseUncertainty, ResponseValue, StudyResult, TabularData,
    VariableId, WeightedGraphSamples, admg_has_bidirected, admg_without_latents_to_dag,
    effect_from_posterior, resolve_posterior_status, response_functional_is_derivative,
};
use crate::analysis::identification_cache::CachedGraphPosteriorAtomIdentification;
use crate::analysis::proof_support::graph_posterior_withheld_diagnostic;
use crate::result::{PrimaryEstimate, ResultPayload, StructuralAggregationPolicy};

/// Stable code of the disclosure every atom-route graph-posterior result carries.
pub(crate) const ATOM_ROUTE_POSTERIOR_CODE: &str = "estimate.graph_posterior.atom_route";

/// Stable code of the disclosure that states why a mixture publishes no aggregate interval.
const UNCERTAINTY_WITHHELD_CODE: &str = "estimate.graph_posterior.uncertainty_withheld";

/// Method tag every posterior atom's estimand carries: the atoms answer one target by
/// structure-specific identifications, so the estimand identity is the target's.
const ATOM_ESTIMAND_METHOD: &str = "graph_posterior.atom_route";

/// Stream base of the shared resampled tables of a mixture's replicates.
const SHARED_REPLICATE_STREAM: u64 = 0x6A7B_0C10_5E00_0001;

/// Whether a static response is answered atom by atom through the explicit executor of each
/// atom's structure: a derivative family functional, or an intervention level over several
/// coordinates.
#[must_use]
pub(crate) fn atom_route_response(query: &ResponseQuery) -> bool {
    response_functional_is_derivative(&query.functional)
        || matches!(
            &query.functional,
            antecedent_core::ResponseFunctional::InterventionResponse { interventions, .. }
                if interventions.len() > 1
        )
}

/// The query a graph posterior of this family answers.
#[derive(Clone, Copy)]
enum AtomQuery<'a> {
    Response(&'a ResponseQuery),
    Conditional(&'a ConditionalEffectQuery),
    Counterfactual(&'a CounterfactualQuery),
}

impl<'a> AtomQuery<'a> {
    fn of(query: &'a CausalQuery) -> Result<Self, CausalError> {
        match query {
            CausalQuery::Response(q) => {
                q.validate()?;
                Ok(Self::Response(q))
            }
            CausalQuery::ConditionalEffect(q) => {
                q.validate()?;
                Ok(Self::Conditional(q))
            }
            CausalQuery::Counterfactual(q) => {
                q.validate()?;
                Ok(Self::Counterfactual(q))
            }
            _ => Err(CausalError::Compile {
                message: "an atom-route graph posterior answers a derivative or multi-coordinate \
                          response, an ADMG conditional effect or a counterfactual"
                    .into(),
            }),
        }
    }

    /// Whether a validation suite is the atoms' own: a derivative's refuter state runs on each
    /// atom's estimand, and no other query of this family has a refuter an atom could run.
    fn validates_per_atom(self) -> bool {
        matches!(self, Self::Response(q) if response_functional_is_derivative(&q.functional))
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Response(_) => "response",
            Self::Conditional(_) => "conditional effect",
            Self::Counterfactual(_) => "counterfactual",
        }
    }

    /// The treatment and outcome the plan and the result name.
    fn treatment_outcome(self) -> Result<(VariableId, VariableId), CausalError> {
        match self {
            Self::Response(q) => response_primary_pair(&q.functional),
            Self::Conditional(q) => Ok((q.inner.treatment, q.inner.outcome)),
            Self::Counterfactual(q) => {
                endpoints(q.interventions.first(), q.outcomes.first().copied())
            }
        }
    }
}

fn endpoints(
    intervention: Option<&Intervention>,
    outcome: Option<VariableId>,
) -> Result<(VariableId, VariableId), CausalError> {
    intervention.and_then(Intervention::primary_variable).zip(outcome).ok_or_else(|| {
        CausalError::Compile {
            message: "a cross-world query over a graph posterior needs one intervention and one \
                      outcome"
                .into(),
        }
    })
}

/// The structure of one posterior sample.
enum AtomGraph {
    Dag(Dag),
    Admg(Admg),
    Cpdag(antecedent_graph::Cpdag),
    Pag(Pag),
}

impl AtomGraph {
    /// The structure of sample `index`; `None` when the sample is not a member of the
    /// posterior's graph class.
    fn of(gp: &GraphPosterior, index: usize) -> Option<Self> {
        let mask = gp.atoms[index];
        let n_vars = gp.n_vars;
        match gp.atom_kind {
            GraphPosteriorAtomKind::Dag => {
                dag_from_adjacency_mask(mask, n_vars).ok().map(Self::Dag)
            }
            GraphPosteriorAtomKind::Admg | GraphPosteriorAtomKind::TemporalAdmg => {
                admg_from_adjacency_mask(mask, n_vars).ok().map(Self::Admg)
            }
            GraphPosteriorAtomKind::Cpdag => {
                cpdag_from_adjacency_mask(mask, n_vars).ok().map(Self::Cpdag)
            }
            GraphPosteriorAtomKind::Pag => {
                let mark =
                    gp.mark_atoms.as_ref().map_or(AdjacencyMask::EMPTY, |marks| marks[index]);
                pag_from_adjacency_mask(mask, mark, n_vars).ok().map(Self::Pag)
            }
            _ => None,
        }
    }

    fn accepted(&self) -> Option<AcceptedGraph> {
        match self {
            Self::Dag(graph) => Some(AcceptedGraph::dag(graph.clone())),
            Self::Admg(graph) => Some(AcceptedGraph::admg(graph.clone())),
            Self::Cpdag(graph) => AcceptedGraph::cpdag(graph.clone()).ok(),
            Self::Pag(graph) => Some(AcceptedGraph::pag(graph.clone())),
        }
    }
}

/// What one posterior atom produced.
enum AtomRun {
    /// The explicit executor of the atom's structure answered.
    Fit(Box<StudyResult>),
    /// The atom is not a member of the posterior's class, or its identifier refuses the query.
    Unidentified,
    /// The identifier accepted the atom but its estimator could not be evaluated.
    Unevaluable,
}

/// One distinct graph of the posterior with the study that answers the query on it.
struct AtomRecord {
    key: u64,
    graph: Option<AtomGraph>,
    study: Option<super::Study>,
    run: AtomRun,
}

/// One atom's contribution to the mixture.
struct AtomEval {
    key: u64,
    status: IdentificationStatus,
    estimand: Option<IdentifiedEstimand>,
    value: Option<f64>,
    posterior: Option<CausalPosterior>,
    assumptions: Option<antecedent_core::AssumptionSet>,
    bounds: Option<(f64, f64)>,
    partial: bool,
    analytic_se: Option<f64>,
    influence: Option<Vec<f64>>,
    uncertainty: Option<antecedent_estimate::EstimateUncertainty>,
}

impl AtomEvalView for AtomEval {
    fn key(&self) -> u64 {
        self.key
    }

    fn status(&self) -> IdentificationStatus {
        self.status
    }

    fn estimand(&self) -> Option<&IdentifiedEstimand> {
        self.estimand.as_ref()
    }

    fn scalar(&self) -> Option<f64> {
        self.value.filter(|value| value.is_finite())
    }

    fn posterior(&self) -> Option<&CausalPosterior> {
        self.posterior.as_ref()
    }

    fn assumptions(&self) -> Option<&antecedent_core::AssumptionSet> {
        self.assumptions.as_ref()
    }

    fn identified_bounds(&self) -> Option<(f64, f64)> {
        self.bounds
    }

    fn partial(&self) -> bool {
        self.partial
    }

    fn analytic_se(&self) -> Option<f64> {
        self.analytic_se
    }

    fn influence(&self) -> Option<&[f64]> {
        self.influence.as_deref()
    }

    fn reported_uncertainty(&self) -> Option<antecedent_estimate::EstimateUncertainty> {
        self.uncertainty
    }
}

/// The estimand an atom contributes: the target, whatever structure-specific identification
/// (adjustment set, district formula, class completion) produced the atom's value, so atoms
/// that identify one target differently still agree on what they answer.
fn target_estimand(estimand: &IdentifiedEstimand) -> IdentifiedEstimand {
    let mut target = estimand.clone();
    target.method = Arc::from(ATOM_ESTIMAND_METHOD);
    target.adjustment_set = Arc::from([]);
    target.instruments = Arc::from([]);
    target.mediators = Arc::from([]);
    target
}

/// The scalar identified set an atom's own class envelope published, when it published one.
fn atom_identified_bounds(result: &StudyResult) -> Option<(f64, f64)> {
    let set = result.structural_response.as_ref()?.identified_set.as_ref()?;
    let lower = *set.lower.first()?;
    let upper = *set.upper.first()?;
    (lower.is_finite() && upper.is_finite() && lower <= upper).then_some((lower, upper))
}

/// Shared replicates refit each atom without recursively resampling or validating it.
fn quiet_bootstrap_study(study: &super::Study) -> super::Study {
    let mut quiet = study.clone();
    quiet.bootstrap_replicates = 0;
    quiet.refute = RefuteSuite::None;
    quiet
}

/// The first coordinate of a response payload, the scalar the mixer accounts an atom by.
fn first_coordinate(value: &ResponseValue) -> Option<f64> {
    match value {
        ResponseValue::Scalar(value) => Some(*value),
        ResponseValue::Vector(values) | ResponseValue::Jacobian { values, .. } => {
            values.first().copied()
        }
        ResponseValue::Surface { mean, .. } => mean.first().copied(),
        ResponseValue::Envelope(_) => None,
    }
}

/// The scalar the mixer accounts an atom's result by.
fn mixing_scalar(kind: AtomQuery<'_>, result: &StudyResult) -> Option<f64> {
    let value = match kind {
        AtomQuery::Response(_) => {
            first_coordinate(&result.response().and_then(response_identified_value)?)?
        }
        AtomQuery::Conditional(_) | AtomQuery::Counterfactual(_) => result.effect(),
    };
    value.is_finite().then_some(value)
}

/// The typed value an atom's result carries into the structural mixture.
fn atom_value(kind: AtomQuery<'_>, result: &StudyResult) -> Option<ResponseValue> {
    match kind {
        AtomQuery::Response(_) => result.response().and_then(response_identified_value),
        AtomQuery::Conditional(_) | AtomQuery::Counterfactual(_) => {
            let effect = result.effect();
            effect.is_finite().then_some(ResponseValue::Scalar(effect))
        }
    }
}

fn atom_eval(
    kind: AtomQuery<'_>,
    key: u64,
    result: &StudyResult,
    rows: usize,
    frequentist: bool,
) -> AtomEval {
    let effect = result.estimate.as_effect();
    AtomEval {
        key,
        status: result.identification.status,
        estimand: Some(target_estimand(&result.estimand)),
        value: mixing_scalar(kind, result),
        posterior: result.posterior.clone(),
        assumptions: effect.map(|effect| effect.provenance.assumptions.clone()),
        bounds: atom_identified_bounds(result),
        partial: result.identification.status == IdentificationStatus::PartiallyIdentified,
        analytic_se: effect
            .map(EffectEstimate::se_analytic)
            .filter(|se| se.is_finite() && *se >= 0.0),
        influence: if frequentist {
            effect
                .and_then(|effect| effect.scores.influence.as_ref())
                .filter(|influence| influence.len() == rows)
                .map(|influence| influence.to_vec())
        } else {
            None
        },
        uncertainty: effect.map(|effect| effect.uncertainty),
    }
}

/// The plan with the strategy pair another route of the same atom would compile to.
fn with_binding(
    physical: &PhysicalExecutionPlan,
    (identifier, estimator): (Arc<str>, Arc<str>),
) -> Arc<PhysicalExecutionPlan> {
    let mut plan = physical.clone();
    plan.logical.record.identifier = Some(identifier);
    plan.logical.record.estimator = Some(estimator);
    Arc::new(plan)
}

/// Answer the query on the structure of one posterior atom with the explicit executor of that
/// structure.
#[allow(
    clippy::too_many_lines,
    reason = "one table of the explicit executor per (query, structure) pair"
)]
fn answer_on_atom(
    atom: &super::Study,
    graph: &AtomGraph,
    data: &TabularData,
    kind: AtomQuery<'_>,
    physical: &Arc<PhysicalExecutionPlan>,
    ctx: &ExecutionContext,
) -> Result<StudyResult, CausalError> {
    let unsupported = |message: &'static str| CausalError::Unsupported { message };
    match kind {
        AtomQuery::Response(query) => match graph {
            AtomGraph::Dag(dag) => atom.execute_response(data, dag, query, physical, None, ctx),
            AtomGraph::Admg(admg) => {
                if admg_has_bidirected(admg) {
                    if response_functional_is_derivative(&query.functional) {
                        atom.execute_admg_derivative(data, admg, query, physical, ctx)
                    } else {
                        atom.execute_admg_response(data, admg, query, physical, ctx)
                    }
                } else {
                    // A latent-free ADMG is the DAG route, as a supplied one is.
                    let dag = admg_without_latents_to_dag(admg)?;
                    let plan = with_binding(physical, atom.resolve_response_pair(query));
                    atom.execute_response(data, &dag, query, &plan, None, ctx)
                }
            }
            AtomGraph::Cpdag(_) => atom.execute_class_response(data, query, physical, ctx),
            AtomGraph::Pag(pag) => {
                let mut plan = (**physical).clone();
                plan.resolved_static_pag = Some(pag.clone());
                atom.execute_class_response(data, query, &plan, ctx)
            }
        },
        AtomQuery::Conditional(query) => match graph {
            AtomGraph::Admg(admg) => {
                if admg_has_bidirected(admg) {
                    atom.execute_admg_conditional(data, admg, query, physical, ctx)
                } else {
                    let dag = admg_without_latents_to_dag(admg)?;
                    let plan = with_binding(physical, atom.resolve_conditional_pair());
                    atom.execute_conditional(data, &dag, query, &plan, ctx)
                }
            }
            AtomGraph::Dag(dag) => {
                let plan = with_binding(physical, atom.resolve_conditional_pair());
                atom.execute_conditional(data, dag, query, &plan, ctx)
            }
            AtomGraph::Cpdag(_) | AtomGraph::Pag(_) => Err(unsupported(
                "a Cpdag or Pag conditional effect is answered by the class conditional route",
            )),
        },
        AtomQuery::Counterfactual(query) => match graph {
            AtomGraph::Dag(dag) => atom.execute_counterfactual(data, dag, query, physical, ctx),
            AtomGraph::Admg(_) | AtomGraph::Cpdag(_) | AtomGraph::Pag(_) => {
                atom.execute_counterfactual_class(data, query, physical, ctx)
            }
        },
    }
}

/// Whether an error is the query's own refusal on every structure (a shape the explicit route
/// does not execute) rather than a failure of one atom.
fn is_query_refusal(error: &CausalError) -> bool {
    matches!(
        error,
        CausalError::Unsupported { .. } | CausalError::Support { .. } | CausalError::Compile { .. }
    ) && !error.is_identification_refusal()
}

/// Draw `d` of the mixture of the atoms' scalar response draws: the posterior-weighted sum of
/// draw `d` of every atom. `None` when an atom retains no draws or the counts differ, so the
/// atoms are not draws of one shared run.
fn pooled_credible_draws(fits: &[(&AtomRecord, &StudyResult, f64)]) -> Option<Vec<Vec<f64>>> {
    let mut columns: Vec<(f64, Arc<[f64]>)> = Vec::with_capacity(fits.len());
    for (_, result, weight) in fits {
        let ResponseUncertainty::Scalar { draws: Some(draws), .. } =
            &result.response()?.uncertainty
        else {
            return None;
        };
        columns.push((*weight, Arc::clone(&draws.values)));
    }
    let n = columns.first()?.1.len();
    let total: f64 = columns.iter().map(|(weight, _)| *weight).sum();
    if n < 2 || !total.is_finite() || total <= 0.0 || columns.iter().any(|(_, c)| c.len() != n) {
        return None;
    }
    Some(
        (0..n)
            .map(|draw| {
                vec![columns.iter().map(|(weight, column)| weight / total * column[draw]).sum()]
            })
            .collect(),
    )
}

/// Circular-block row indices of one resample of `n` rows (`block == 1` is the i.i.d. row
/// bootstrap).
fn resample_rows(n: usize, block: usize, rng: &mut antecedent_core::CausalRng) -> Vec<usize> {
    let block = block.clamp(1, n.max(1));
    let blocks = n.div_ceil(block);
    let mut rows = Vec::with_capacity(blocks * block);
    for _ in 0..blocks {
        let start = antecedent_kernels::unbiased_index(rng, n);
        rows.extend((0..block).map(|offset| (start + offset) % n));
    }
    rows.truncate(n);
    rows
}

/// The draw-wise mixture of the atoms' replicates on shared resampled tables.
struct SharedReplicates {
    se: Option<f64>,
    ok: u32,
    failed: u32,
    note: String,
}

impl super::Study {
    /// Refusals every execution of this family of graph posteriors shares.
    fn ensure_atom_route_licensed(
        &self,
        gp: &GraphPosterior,
        kind: AtomQuery<'_>,
    ) -> Result<(), CausalError> {
        if gp.lag_masks.is_some() || gp.max_lag.is_some() {
            return Err(CausalError::Unsupported {
                message: "a static query over a graph posterior requires static atoms; this \
                          posterior carries DBN lag structure",
            });
        }
        if self.refute != RefuteSuite::None && !kind.validates_per_atom() {
            return Err(CausalError::Unsupported {
                message: "a counterfactual or ADMG conditional effect over a graph posterior is \
                          validated by the functional refuter state of the whole procedure, not \
                          by an atom's executor",
            });
        }
        Ok(())
    }

    /// The study that answers the query on one atom: this study with the atom's structure in
    /// place of the posterior, and no refuter or retained identification of the posterior.
    fn posterior_atom_study(&self, graph: &AtomGraph, keep_refute: bool) -> Option<super::Study> {
        let accepted = graph.accepted()?;
        let mut atom = self.clone();
        atom.graph_posterior = None;
        atom.graph = Arc::new(accepted);
        atom.refute = if keep_refute { self.refute } else { RefuteSuite::None };
        atom.custom_validators.clear();
        atom.clear_identification_where(|cache| {
            matches!(
                cache,
                crate::analysis::IdentificationCache::GraphPosterior(_)
                    | crate::analysis::IdentificationCache::Static(_)
                    | crate::analysis::IdentificationCache::Pag(_)
                    | crate::analysis::IdentificationCache::Cpdag(_)
            )
        });
        Some(atom)
    }

    /// Run the explicit executor on every distinct graph of the posterior, in order of first
    /// appearance and in one context so that draw `d` of every atom is one draw of the row
    /// weights. An atom's refusal or failure is its outcome; only a cancellation or a resource
    /// refusal ends the run.
    fn run_posterior_atoms(
        &self,
        data: &TabularData,
        gp: &GraphPosterior,
        kind: AtomQuery<'_>,
        physical: &Arc<PhysicalExecutionPlan>,
        ctx: &ExecutionContext,
    ) -> Result<Vec<AtomRecord>, CausalError> {
        let mut records: Vec<AtomRecord> = Vec::new();
        let mut seen: std::collections::HashSet<u64> = std::collections::HashSet::new();
        let mut refusal: Option<CausalError> = None;
        for index in 0..gp.n_graphs {
            let key = gp.graph_keys[index];
            if !seen.insert(key) {
                continue;
            }
            if ctx.cancellation.is_cancelled() {
                return Err(CausalError::Cancelled {
                    stage: crate::analysis::stage::STAGE_ESTIMATE_POINT,
                });
            }
            let graph = AtomGraph::of(gp, index);
            let study = graph
                .as_ref()
                .and_then(|graph| self.posterior_atom_study(graph, kind.validates_per_atom()));
            let run = match (graph.as_ref(), study.as_ref()) {
                (Some(graph_ref), Some(study_ref)) => {
                    match answer_on_atom(study_ref, graph_ref, data, kind, physical, ctx) {
                        Ok(result) => AtomRun::Fit(Box::new(result)),
                        Err(
                            error @ (CausalError::Cancelled { .. } | CausalError::Resource { .. }),
                        ) => {
                            return Err(error);
                        }
                        Err(error) if error.is_identification_refusal() => AtomRun::Unidentified,
                        Err(error) => {
                            if refusal.is_none() && is_query_refusal(&error) {
                                refusal = Some(error);
                            }
                            AtomRun::Unevaluable
                        }
                    }
                }
                _ => AtomRun::Unidentified,
            };
            records.push(AtomRecord { key, graph, study, run });
        }
        let fitted = records.iter().any(|record| matches!(record.run, AtomRun::Fit(_)));
        if !fitted {
            if let Some(error) = refusal {
                return Err(error);
            }
        }
        Ok(records)
    }

    /// Re-run every contributing atom on shared resampled tables and read the spread of the
    /// draw-wise mixture of their values.
    fn shared_row_replicates(
        &self,
        data: &TabularData,
        atoms: &[(f64, &super::Study, &AtomGraph)],
        kind: AtomQuery<'_>,
        physical: &Arc<PhysicalExecutionPlan>,
        ctx: &ExecutionContext,
    ) -> Result<SharedReplicates, CausalError> {
        let requested = self.bootstrap_replicates;
        let unavailable =
            |note: &str| SharedReplicates { se: None, ok: 0, failed: 0, note: note.to_owned() };
        if requested < 2 {
            return Ok(unavailable(
                "no bootstrap replicates were requested, so the mixture of several atoms \
                 publishes no interval",
            ));
        }
        let n = data.row_count();
        let block = match self.sampling_design {
            Some(antecedent_core::DeclaredSamplingDesign::ClusteredPanel) => {
                return Ok(unavailable(
                    "the declared clustered design needs whole clusters, and the table carries \
                     no cluster labels to draw them from",
                ));
            }
            Some(antecedent_core::DeclaredSamplingDesign::DependentTimeSeries) => {
                antecedent_data::circular_block_length(1, n)
            }
            _ => 1,
        };
        let ids: Vec<VariableId> =
            data.schema().variables().iter().map(|variable| variable.id).collect();
        let columns: Vec<Vec<f64>> =
            match ids.iter().map(|&id| data.float64_values(id)).collect::<Result<_, _>>() {
                Ok(columns) => columns,
                Err(_) => {
                    return Ok(unavailable(
                        "a column cannot be read as floats, so the table cannot be resampled",
                    ));
                }
            };
        let total: f64 = atoms.iter().map(|(weight, _, _)| *weight).sum();
        if !total.is_finite() || total <= 0.0 {
            return Ok(unavailable("the contributing atoms carry no posterior mass"));
        }
        let quiet: Vec<_> = atoms
            .iter()
            .map(|(weight, study, graph)| (*weight, quiet_bootstrap_study(study), *graph))
            .collect();
        let mut values: Vec<f64> = Vec::new();
        let mut failed = 0_u32;
        'replicates: for replicate in 0..requested {
            if ctx.cancellation.is_cancelled() {
                return Err(CausalError::Cancelled {
                    stage: crate::analysis::stage::STAGE_ESTIMATE_POINT,
                });
            }
            let mut rng = ctx.rng.stream_for(
                antecedent_core::StreamDomain::Resample,
                SHARED_REPLICATE_STREAM.wrapping_add(u64::from(replicate)),
            );
            let picked = resample_rows(n, block, &mut rng);
            let replacements: Vec<(VariableId, Arc<[f64]>)> = ids
                .iter()
                .zip(&columns)
                .map(|(&id, column)| (id, picked.iter().map(|&row| column[row]).collect()))
                .collect();
            let Ok(table) = data.with_replaced_floats(&replacements) else {
                failed += 1;
                continue;
            };
            let mut mixed = 0.0;
            for (weight, study, graph) in &quiet {
                let value = answer_on_atom(study, graph, &table, kind, physical, ctx)
                    .ok()
                    .and_then(|result| mixing_scalar(kind, &result));
                let Some(value) = value else {
                    failed += 1;
                    continue 'replicates;
                };
                mixed += weight * value;
            }
            values.push(mixed / total);
        }
        let ok = u32::try_from(values.len()).unwrap_or(u32::MAX);
        let se = (values.len() >= 2
            && super::bootstrap_has_enough_successes(values.len(), requested as usize))
        .then(|| {
            let count = values.len() as f64;
            let mean = values.iter().sum::<f64>() / count;
            let variance = values.iter().map(|value| (value - mean) * (value - mean)).sum::<f64>()
                / (count - 1.0);
            variance.sqrt()
        })
        .filter(|se| se.is_finite());
        Ok(SharedReplicates {
            se,
            ok,
            failed,
            note: format!(
                "every contributing atom was re-run on each of {requested} shared resampled \
                 tables ({} drawn in blocks of {block} row(s)); the standard error is the spread \
                 of the draw-wise posterior-weighted mixture over {ok} surviving replicates",
                if block == 1 { "rows" } else { "consecutive rows" },
            ),
        })
    }

    /// The pooled posterior of the mixture: draw `d` of the mixture is the posterior-weighted
    /// sum of draw `d` of every atom. `None` when an atom publishes no effect column, or the
    /// atoms' draws differ in count so they are not draws of one shared run.
    fn pool_atom_posteriors(
        items: &[(f64, &CausalPosterior)],
        status: IdentificationStatus,
        (unidentified_mass, unevaluable_mass): (f64, f64),
    ) -> Option<CausalPosterior> {
        let (_, first) = items.first()?;
        let first_column = first.effect_column()?;
        let n_draws = first.draws.n_draws;
        let total: f64 = items.iter().map(|(weight, _)| *weight).sum();
        if n_draws < 2 || !total.is_finite() || total <= 0.0 {
            return None;
        }
        let mut pooled = vec![0.0; n_draws];
        for (weight, posterior) in items {
            if posterior.draws.n_draws != n_draws {
                return None;
            }
            let column = posterior.draws.column(posterior.effect_column()?).ok()?;
            if column.len() != n_draws || column.iter().any(|draw| !draw.is_finite()) {
                return None;
            }
            for (slot, draw) in pooled.iter_mut().zip(column) {
                *slot += weight / total * draw;
            }
        }
        let quantity = first.draws.schema.quantities.get(first_column)?.clone();
        let schema = antecedent_prob::PosteriorSchema { quantities: Arc::from([quantity]) };
        let draws = antecedent_prob::PosteriorDraws::from_column_major(
            schema,
            n_draws,
            Arc::<[f64]>::from(pooled),
        )
        .ok()?;
        let mut posterior = (*first).clone();
        posterior.summaries = draws.summarize();
        posterior.draws = draws;
        posterior.identification = status;
        posterior.prior_sensitivity = None;
        posterior.unidentified_mass = unidentified_mass;
        posterior.unevaluable_mass = unevaluable_mass;
        posterior.diagnostics.notes.push(Arc::from(
            "pooled over the graph posterior: draw d is the posterior-weighted sum of draw d of \
             every contributing atom, all run in one context so that draw d is one draw of the \
             row weights",
        ));
        for (_, other) in items.iter().skip(1) {
            posterior.assumptions.extend_unique(&other.assumptions.entries);
        }
        Some(posterior)
    }

    /// Answer a derivative or multi-coordinate response, an ADMG conditional effect or a
    /// counterfactual over the atoms of a tabular graph posterior.
    ///
    /// # Errors
    ///
    /// A posterior with DBN lag structure, a validation suite other than `none`, no atom that
    /// identifies and evaluates the query (the refusal of the explicit route when every atom
    /// refused it), or a cancelled run.
    #[allow(
        clippy::too_many_lines,
        reason = "one mixture: the atoms, their masses, the typed value, the interval and the finish read one frozen posterior"
    )]
    pub(super) fn execute_atom_route_graph_posterior(
        &self,
        data: &TabularData,
        gp: &GraphPosterior,
        physical: &Arc<PhysicalExecutionPlan>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let kind = AtomQuery::of(&self.query)?;
        self.ensure_atom_route_licensed(gp, kind)?;
        let records = self.run_posterior_atoms(data, gp, kind, physical, ctx)?;
        let by_key: HashMap<u64, usize> =
            records.iter().enumerate().map(|(position, record)| (record.key, position)).collect();

        // The sample flags: a refused atom is unidentified mass, an atom the identifier
        // accepted stays identified whether or not its estimator evaluated.
        let flags: Vec<GraphIdentFlag> = (0..gp.n_graphs)
            .map(|index| match by_key.get(&gp.graph_keys[index]).map(|&p| &records[p].run) {
                Some(AtomRun::Fit(_) | AtomRun::Unevaluable) => GraphIdentFlag::Identified,
                _ => GraphIdentFlag::Unidentified,
            })
            .collect();
        let graphs = WeightedGraphSamples::new(gp.weights.to_vec(), flags, gp.graph_keys.to_vec())?;
        let unidentified_mass = graphs.unidentified_mass();
        let key_weights = IdentifiedKeyWeights::new(&graphs);
        let frequentist = self.inference.is_frequentist();
        let rows = data.row_count();

        let mut evals: Vec<AtomEval> = Vec::with_capacity(records.len());
        let mut proofs: Vec<CachedGraphPosteriorAtomIdentification> = Vec::new();
        for record in &records {
            match &record.run {
                AtomRun::Unidentified => {}
                AtomRun::Unevaluable => evals.push(AtomEval {
                    key: record.key,
                    status: IdentificationStatus::NotIdentified,
                    estimand: None,
                    value: None,
                    posterior: None,
                    assumptions: None,
                    bounds: None,
                    partial: false,
                    analytic_se: None,
                    influence: None,
                    uncertainty: None,
                }),
                AtomRun::Fit(result) => {
                    let eval = atom_eval(kind, record.key, result, rows, frequentist);
                    if eval.value.is_some() {
                        proofs.push(CachedGraphPosteriorAtomIdentification {
                            key: record.key,
                            estimand: target_estimand(&result.estimand),
                            identification: result.identification.clone(),
                            transport: None,
                        });
                    }
                    evals.push(eval);
                }
            }
        }
        if proofs.is_empty() {
            return Err(CausalError::not_identified(
                IdentificationStatus::NotIdentified,
                false,
                &format!(
                    "no atom of the graph posterior identifies and evaluates the {}",
                    kind.label()
                ),
            ));
        }
        let label = kind.label();
        let mut mixed = mix_atom_evals(
            &evals,
            AtomMixInput {
                graphs: &graphs,
                proofs: &proofs,
                unidentified_mass,
                full_mass_scope: true,
                truncated_atoms: 0,
                bounds: AtomBoundRule::SetOrPoint,
                estimand_fallback: true,
            },
            |n, mixable| {
                format!(
                    "{label} graph-posterior mix: no evaluable atom value (evals={n}, \
                     mixable={mixable})"
                )
            },
            || format!("{label} graph-posterior mix: missing estimand"),
        )?;
        mixed.identification.status = resolve_posterior_status(
            mixed.identification.status,
            mixed.policy,
            PosteriorMassSummary {
                unidentified: unidentified_mass,
                unevaluable: mixed.mixture.unevaluable_mass,
                ..PosteriorMassSummary::default()
            },
        );

        // The atoms that contributed a value, with their posterior mass.
        let fits: Vec<(&AtomRecord, &StudyResult, f64)> = records
            .iter()
            .filter_map(|record| match &record.run {
                AtomRun::Fit(result) if mixing_scalar(kind, result).is_some() => {
                    Some((record, &**result, key_weights.get(record.key)))
                }
                _ => None,
            })
            .collect();
        let representative = fits
            .iter()
            .max_by(|left, right| left.2.total_cmp(&right.2))
            .map(|(record, result, _)| (*record, *result))
            .ok_or_else(|| CausalError::Compile {
                message: format!("{label} graph-posterior mix: no contributing atom"),
            })?;
        let contributing = fits.len();
        let mut diagnostics: Vec<Diagnostic> = Vec::new();

        // The published estimate and posterior.
        let MixedAtomEvals { policy, identification, mut estimate, posterior, mut mixture, .. } =
            mixed;
        let mut published_posterior = posterior;
        let mut mean_se: Option<f64> = None;
        if contributing == 1 {
            // One atom carries the whole conditional mixture: its own uncertainty is the
            // mixture's.
            if let Some(own) = representative.1.estimate.as_effect() {
                estimate.uncertainty = own.uncertainty;
                estimate.scores.influence.clone_from(&own.scores.influence);
            }
            if published_posterior.is_none() {
                published_posterior.clone_from(&representative.1.posterior);
            }
            if let Some(posterior) = published_posterior.as_mut() {
                posterior.unidentified_mass = mixture.unidentified_mass;
                posterior.unevaluable_mass = mixture.unevaluable_mass;
            }
        } else if self.inference.is_bayesian() {
            let items: Vec<(f64, &CausalPosterior)> = fits
                .iter()
                .filter_map(|(_, result, weight)| result.posterior.as_ref().map(|p| (*weight, p)))
                .collect();
            let scalar_only = fits.iter().all(|(_, result, _)| {
                matches!(atom_value(kind, result), Some(ResponseValue::Scalar(_)))
            });
            let pooled = (items.len() == contributing && scalar_only)
                .then(|| {
                    Self::pool_atom_posteriors(
                        &items,
                        identification.status,
                        (mixture.unidentified_mass, mixture.unevaluable_mass),
                    )
                })
                .flatten();
            if let Some(pooled) = pooled {
                if let Ok(pooled_estimate) = effect_from_posterior(&pooled) {
                    estimate = pooled_estimate;
                    mean_se = Some(estimate.se_analytic()).filter(|se| se.is_finite());
                }
                published_posterior = Some(pooled);
                diagnostics.push(Diagnostic::new(
                    ATOM_ROUTE_POSTERIOR_CODE,
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    "pooled posterior: draw d of the mixture is the posterior-weighted sum of \
                     draw d of every contributing atom, all run in one context so that draw d \
                     is one draw of the row weights",
                ));
            } else {
                diagnostics.push(graph_posterior_withheld_diagnostic());
                diagnostics.push(Diagnostic::new(
                    UNCERTAINTY_WITHHELD_CODE,
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Warning,
                    "no aggregate posterior: the atoms' draws differ in count or publish no \
                     effect column, so they are not draws of one shared run; the per-atom \
                     posteriors are retained on the structural mixture",
                ));
            }
        } else if !estimate.se_analytic().is_finite() && estimate.uncertainty.bootstrap.se.is_none()
        {
            let atoms: Vec<(f64, &super::Study, &AtomGraph)> = fits
                .iter()
                .filter_map(|(record, _, weight)| {
                    Some((*weight, record.study.as_ref()?, record.graph.as_ref()?))
                })
                .collect();
            let replicates = self.shared_row_replicates(data, &atoms, kind, physical, ctx)?;
            diagnostics.push(Diagnostic::new(
                if replicates.se.is_some() {
                    ATOM_ROUTE_POSTERIOR_CODE
                } else {
                    UNCERTAINTY_WITHHELD_CODE
                },
                DiagnosticKind::Scientific,
                if replicates.se.is_some() {
                    DiagnosticSeverity::Info
                } else {
                    DiagnosticSeverity::Warning
                },
                replicates.note.clone(),
            ));
            if replicates.se.is_some() {
                mean_se = replicates.se;
                estimate = EffectEstimate::from_parts(
                    estimate.ate,
                    f64::NAN,
                    BootstrapAccounting {
                        se: replicates.se,
                        replicates_ok: Some(replicates.ok),
                        replicates_failed: Some(replicates.failed),
                        cancelled: false,
                        early_stopped: false,
                    },
                    estimate.provenance.assumptions.clone(),
                    OverlapPolicy::ExplicitOverride,
                    None,
                    None,
                );
            }
        } else {
            mean_se = Some(estimate.se_analytic())
                .filter(|se| se.is_finite())
                .or(estimate.uncertainty.bootstrap.se);
            diagnostics.push(Diagnostic::new(
                ATOM_ROUTE_POSTERIOR_CODE,
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "standard error of the frozen-weight mixture from the joint row influence of the \
                 contributing atoms on the shared sample",
            ));
        }

        // The typed value of the answer.
        let (treatment, outcome) = kind.treatment_outcome()?;
        let mut result = representative.1.clone();
        let mut payload = if contributing == 1 {
            representative.1.payload.clone()
        } else {
            ResultPayload::Scalar
        };
        match kind {
            AtomQuery::Response(query) => {
                let items: Vec<(f64, ResponseValue)> = fits
                    .iter()
                    .filter_map(|(_, result, weight)| {
                        result
                            .response()
                            .and_then(response_identified_value)
                            .map(|value| (*weight, value))
                    })
                    .collect();
                let refs: Vec<(f64, &ResponseValue)> =
                    items.iter().map(|(weight, value)| (*weight, value)).collect();
                if let Some(base) = representative.1.response() {
                    let mut response = base.clone();
                    let graph_dependent =
                        identification.status == IdentificationStatus::GraphDependent;
                    let value = mix_response_values(&refs).ok();
                    response.estimand = query.functional.clone();
                    response.identification_status = identification.status;
                    response.estimate = match (&value, graph_dependent) {
                        (Some(value), false) => {
                            ResponseIdentification::PointIdentified(value.clone())
                        }
                        _ => ResponseIdentification::GraphDependent(
                            fits.iter()
                                .filter_map(|(record, result, _)| {
                                    result
                                        .response()
                                        .and_then(response_identified_value)
                                        .map(|value| (record.key, value))
                                })
                                .collect(),
                        ),
                    };
                    if contributing > 1 {
                        let level = match &base.uncertainty {
                            ResponseUncertainty::Scalar { level, .. } => *level,
                            _ => crate::result::REPORTED_SE_INTERVAL_LEVEL,
                        };
                        response.uncertainty = match (&value, mean_se) {
                            (Some(ResponseValue::Scalar(point)), Some(se)) => {
                                let z = antecedent_stats::normal_ppf(0.5 + level / 2.0);
                                ResponseUncertainty::Scalar {
                                    standard_error: se,
                                    level,
                                    lower: point - z * se,
                                    upper: point + z * se,
                                    interpretation: if self.inference.is_bayesian() {
                                        antecedent_core::IntervalInterpretation::Credible
                                    } else {
                                        antecedent_core::IntervalInterpretation::Confidence
                                    },
                                    draws: None,
                                }
                            }
                            _ => ResponseUncertainty::None,
                        };
                        if self.inference.is_bayesian() {
                            if let Some(ResponseValue::Scalar(point)) = &value {
                                if let Some(draws) = pooled_credible_draws(&fits) {
                                    super::temporal_point_uncertainty::attach_posterior_response_band(
                                        &mut response,
                                        &draws,
                                        &[*point],
                                    );
                                    diagnostics.push(Diagnostic::new(
                                        ATOM_ROUTE_POSTERIOR_CODE,
                                        DiagnosticKind::Scientific,
                                        DiagnosticSeverity::Info,
                                        "pooled credible draws: draw d of the mixture is the \
                                         posterior-weighted sum of draw d of every contributing \
                                         atom's response draws, all run in one context so that \
                                         draw d is one draw of the row weights",
                                    ));
                                } else {
                                    diagnostics.push(Diagnostic::new(
                                        UNCERTAINTY_WITHHELD_CODE,
                                        DiagnosticKind::Scientific,
                                        DiagnosticSeverity::Warning,
                                        "no aggregate credible interval: the atoms' response \
                                         draws differ in count or are not retained, so they are \
                                         not draws of one shared run; each atom's own interval is \
                                         kept on the structural mixture",
                                    ));
                                }
                            }
                        }
                        let supports: Vec<&antecedent_core::SupportReport> = fits
                            .iter()
                            .filter_map(|(_, result, _)| {
                                result.response().map(|response| &response.support)
                            })
                            .collect();
                        if !supports.is_empty() {
                            response.support = mix_support_reports(&supports);
                        }
                        for (_, other, _) in fits
                            .iter()
                            .filter(|(record, _, _)| !std::ptr::eq(*record, representative.0))
                        {
                            if let Some(other) = other.response() {
                                response.assumptions.extend_unique(&other.assumptions.entries);
                            }
                        }
                    }
                    response.provenance_id =
                        Arc::from("estimate.response.graph_posterior.atom_route");
                    if let Some(value) = value {
                        mixture.conditional_on_identified = Some(value);
                    }
                    let (scalar, se) = response_scalar_summary(&response);
                    estimate = EffectEstimate::from_parts(
                        scalar,
                        se,
                        BootstrapAccounting::default(),
                        response.assumptions.clone(),
                        OverlapPolicy::ExplicitOverride,
                        None,
                        None,
                    );
                    if contributing == 1 {
                        if let Some(own) = representative.1.estimate.as_effect() {
                            estimate.uncertainty = own.uncertainty;
                            estimate.scores.influence.clone_from(&own.scores.influence);
                        }
                    }
                    payload = ResultPayload::Response(response);
                }
            }
            AtomQuery::Conditional(_) | AtomQuery::Counterfactual(_) => {}
        }

        // The structural atoms carry each graph's own typed value, and for a response its full
        // per-graph answer.
        for atom in &mut mixture.atoms {
            let Some(&position) = by_key.get(&atom.graph_key) else { continue };
            if let AtomRun::Fit(each) = &records[position].run {
                if let Some(value) = atom_value(kind, each) {
                    atom.value = Some(value);
                }
                if matches!(kind, AtomQuery::Response(_)) {
                    atom.response = each.response().cloned();
                }
            }
        }

        let identified_mass = mixture.identified_mass;
        let unevaluable_mass = mixture.unevaluable_mass;
        let unidentified_reported = mixture.unidentified_mass;
        diagnostics.push(
            Diagnostic::new(
                ATOM_ROUTE_POSTERIOR_CODE,
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                format!(
                    "each {} posterior atom answered the {} with the executor of its own \
                     structure, and the atoms are mixed by posterior probability over the atoms \
                     that identify and evaluate it ({contributing} of {} distinct graph(s)); an \
                     atom the identifier refuses is unidentified mass and an atom whose \
                     estimator cannot be evaluated is unevaluable mass, neither redistributed",
                    gp.atom_kind.as_str(),
                    kind.label(),
                    records.len(),
                ),
            )
            .with_fields([
                ("atoms", contributing.to_string()),
                ("unidentified_mass", unidentified_reported.to_string()),
                ("unevaluable_mass", unevaluable_mass.to_string()),
            ]),
        );
        super::push_graph_posterior_structural_aggregation_diagnostic(
            &mut diagnostics,
            policy,
            identified_mass,
            unidentified_reported,
            unevaluable_mass,
            mixture.subsampled_out_mass,
        );
        if matches!(policy, StructuralAggregationPolicy::GraphDependentAtoms) {
            diagnostics.push(Diagnostic::new(
                UNCERTAINTY_WITHHELD_CODE,
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Warning,
                "the atoms did not agree on what they identify, so no aggregate value is \
                 published; per-atom values are retained on the structural mixture",
            ));
        }
        result.diagnostics.extend(diagnostics);
        result.identification = identification;
        result.certificate = None;
        result.estimate = PrimaryEstimate::Effect(estimate);
        result.posterior = published_posterior;
        result.structural_response = Some(mixture);
        // A derivative's refuter state ran on each atom's own estimand: the reports of one
        // refuter are mixed by the posterior weight of the atoms that produced them, a mixed
        // verdict passes only if every contributing atom passes, and a refuter that ran on part
        // of the mass says so.
        let per_atom: Vec<(f64, Vec<antecedent_validate::RefutationReport>)> =
            fits.iter().map(|(_, each, weight)| (*weight, each.refutations.clone())).collect();
        let (mixed_reports, coverage_notes) = if kind.validates_per_atom() {
            super::mix_atom_refutation_reports(&per_atom)
        } else {
            (Vec::new(), Vec::new())
        };
        result.refutations = mixed_reports;
        result.diagnostics.extend(coverage_notes);
        result.predictive_checks =
            fits.iter().flat_map(|(_, each, _)| each.predictive_checks.iter().cloned()).collect();
        if kind.validates_per_atom() && self.refute != RefuteSuite::None {
            let listed = fits
                .iter()
                .map(|(record, each, weight)| {
                    format!("{:x}:weight={weight}:reports={}", record.key, each.refutations.len())
                })
                .collect::<Vec<_>>()
                .join(",");
            result.diagnostics.push(Diagnostic::new(
                "refute.graph_posterior.per_atom",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                format!(
                    "cheap/full validation ran the derivative refuter state on each \
                     contributing atom's own estimand; the reports of one refuter are mixed \
                     by posterior weight and pass only if every contributing atom passes \
                     (atoms: {listed})"
                ),
            ));
        }
        result.payload = payload;
        result.treatment = Some(treatment);
        result.outcome = outcome;
        result.rebind_interval(self.inference.is_bayesian());
        Ok(result)
    }
}
