// SPDX-License-Identifier: MIT OR Apache-2.0
//! Mediation, path-restricted natural effects and natural direct effects over a supplied graph
//! posterior on tabular data.
//!
//! Every atom of the posterior is a structure: a DAG, an ADMG (or the contemporaneous projection
//! of a temporal ADMG), a CPDAG or a PAG. Each atom is answered by the executor the same query
//! has on a supplied structure of that kind, so an atom's value, identification and uncertainty
//! are exactly those of the explicit cell: the Dag route's own plug-in on a DAG, the
//! path-specific identifier and finite-discrete plug-in on a mixed graph, and the class
//! envelope (every DAG completion of a CPDAG, every ADMG sharing the m-separation model of a MAG
//! completion of a PAG) on a class. The posterior moves only which structure answers.
//!
//! The atoms are then mixed by posterior probability, one mixture per component of the answer:
//!
//! * a mediation has three components (total, natural direct and natural indirect effects), each
//!   mixed over the atoms that identify it; a component no atom identifies is reported absent and
//!   an atom whose identifier refuses a component other than the requested one keeps that
//!   component's mass as unidentified mass of that component only;
//! * a path-restricted or natural direct effect has one component.
//!
//! An atom the identifier refuses (a recanting district, a hedge, a class whose members all
//! refuse) is unidentified posterior mass, and an atom whose plug-in cannot be evaluated is
//! unevaluable mass. Neither is redistributed: the published value is conditional on the atoms
//! that identify and evaluate, the result states the mass it does not cover, and the status
//! resolves to graph-dependent when either exceeds the tolerance. An atom that is a class whose
//! completions disagree keeps its identified set, and the mixture publishes the union of the
//! atoms' sets beside the point.
//!
//! Uncertainty of the mixture. One contributing atom carries the whole conditional mixture, so its
//! own interval and posterior are the mixture's. Several atoms, Frequentist: every atom is re-run
//! on the same resampled tables and the standard error is the spread of the draw-wise
//! posterior-weighted mixture. Several atoms, Bayesian: the atoms ran in one context, so draw `d`
//! of every atom is the same draw of the row weights and the pooled posterior is the draw-wise
//! posterior-weighted mixture. An interval that cannot be formed is withheld and the result says
//! why. Cheap and full validation run each atom's own suite and mix the reports by posterior
//! weight.

use std::collections::{HashMap, HashSet};

use antecedent_core::{
    InterventionalDistributionQuery, MediationQuery, NestedCounterfactualQuery,
    PathSpecificEffectQuery, VariableId,
};
use antecedent_data::TableView;
use antecedent_discovery::adjacency_mask::{
    admg_from_adjacency_mask, cpdag_from_adjacency_mask, dag_from_adjacency_mask,
    pag_from_adjacency_mask,
};
use antecedent_discovery::dbn_posterior::temporal_pag_from_dbn_atoms;
use antecedent_discovery::{AdjacencyMask, GraphPosteriorAtomKind, LagMask};
use antecedent_estimate::{
    DistributionAtom, InterventionalDistributionEstimate, TemporalMediationEstimate,
};
use antecedent_graph::Admg;

use super::atom_mix::{AtomBoundRule, AtomEvalView, AtomMixInput, MixedAtomEvals, mix_atom_evals};
use super::class_distribution::{DistributionClass, envelope_of_members};
use super::mediation_mixed::{ClassStructure, component_slot};
use super::static_path::DistributionGraph;
use super::{
    AcceptedGraph, Arc, BootstrapAccounting, CausalError, CausalPosterior, CausalQuery, Diagnostic,
    DiagnosticKind, DiagnosticSeverity, EffectEstimate, EstimatorId, ExecutionContext,
    GraphIdentFlag, GraphPosterior, IdentificationStatus, IdentifiedEstimand,
    IdentifiedExecuteExtras, IdentifiedExecuteFinish, IdentifierId, Instant, OverlapPolicy,
    PathSpecificMixedSource, PhysicalExecutionPlan, PosteriorMassSummary, RefuteSuite, Strategy,
    StudyResult, TabularData, TimeSeriesData, WeightedGraphSamples, effect_from_posterior,
    elapsed_ns, mass_fields, mix_atom_refutation_reports, overlap_diagnostic,
    overlap_limit_diagnostic, provenance_ids, resolve_posterior_status,
};
use crate::analysis::identification_cache::CachedGraphPosteriorAtomIdentification;
use crate::analysis::proof_support::graph_posterior_withheld_diagnostic;
use crate::planner::RouteDefaults;

/// Stable code of the disclosure every cross-world graph-posterior result carries.
pub(crate) const CROSS_WORLD_POSTERIOR_CODE: &str = "estimate.graph_posterior.cross_world";

/// Stream base of the shared resampled tables of a mixture's replicates.
const SHARED_REPLICATE_STREAM: u64 = 0x6A7B_0C10_5E00_0002;

/// Stable code of the disclosure of how the uncertainty of a mixture of several atoms was formed.
const UNCERTAINTY_CODE: &str = "estimate.graph_posterior.cross_world_uncertainty";

/// Method tag every posterior atom's estimand carries: the atoms answer one target by
/// structure-specific identifications, so the estimand identity is the target's.
const ATOM_ESTIMAND_METHOD: &str = "graph_posterior.cross_world";

/// Names of the components of a mediation, in publication order.
const MEDIATION_COMPONENTS: [&str; 3] = ["total", "direct", "indirect"];

/// The query a cross-world graph posterior answers.
#[derive(Clone, Copy)]
enum CrossWorldKind<'a> {
    Mediation(&'a MediationQuery),
    PathSpecific(&'a PathSpecificEffectQuery),
    Nested(&'a NestedCounterfactualQuery),
    Distribution(&'a InterventionalDistributionQuery),
}

impl<'a> CrossWorldKind<'a> {
    fn of(query: &'a antecedent_core::CausalQuery) -> Result<Self, CausalError> {
        match query {
            CausalQuery::Mediation(q) => {
                q.validate()?;
                Ok(Self::Mediation(q))
            }
            CausalQuery::PathSpecific(q) => {
                q.validate()?;
                Ok(Self::PathSpecific(q))
            }
            CausalQuery::NestedCounterfactual(q) => {
                q.validate()?;
                Ok(Self::Nested(q))
            }
            CausalQuery::Distribution(q) => {
                q.validate()?;
                Ok(Self::Distribution(q))
            }
            _ => Err(CausalError::Compile {
                message: "a cross-world graph posterior answers a mediation, a path-restricted \
                          natural effect, a natural direct effect or an interventional \
                          distribution"
                    .into(),
            }),
        }
    }

    fn treatment_outcome(self) -> Result<(VariableId, VariableId), CausalError> {
        Ok(match self {
            Self::Mediation(q) => (q.treatment, q.outcome),
            Self::PathSpecific(q) => (q.treatment, q.outcome),
            Self::Nested(q) => (q.treatment, q.outcome),
            Self::Distribution(q) => {
                let treatment = q
                    .interventions
                    .first()
                    .and_then(antecedent_core::Intervention::primary_variable)
                    .ok_or_else(|| CausalError::Compile {
                        message: "distribution query missing intervention target".into(),
                    })?;
                let outcome = *q.outcomes.first().ok_or_else(|| CausalError::Compile {
                    message: "distribution query missing outcome".into(),
                })?;
                (treatment, outcome)
            }
        })
    }

    /// Components of the answer.
    const fn components(self) -> usize {
        match self {
            Self::Mediation(_) => MEDIATION_COMPONENTS.len(),
            Self::PathSpecific(_) | Self::Nested(_) | Self::Distribution(_) => 1,
        }
    }

    /// The component the query asks for.
    fn requested(self) -> usize {
        match self {
            Self::Mediation(q) => component_slot(q.contrast),
            Self::PathSpecific(_) | Self::Nested(_) | Self::Distribution(_) => 0,
        }
    }

    /// The estimator the query's explicit cell on a DAG uses.
    fn dag_estimator(self) -> EstimatorId {
        match self {
            Self::Mediation(_) => EstimatorId::StaticMediationLinear,
            Self::PathSpecific(_) | Self::Nested(_) => EstimatorId::FunctionalEffect,
            Self::Distribution(_) => EstimatorId::FunctionalDistribution,
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Mediation(_) => "mediation",
            Self::PathSpecific(_) => "path-restricted natural effect",
            Self::Nested(_) => "natural direct effect",
            Self::Distribution(_) => "interventional distribution",
        }
    }

    /// One component's published value on an atom that answered, `None` where the atom's
    /// identifier refused that component.
    fn component_value(self, result: &StudyResult, slot: usize) -> Option<f64> {
        match self {
            Self::Mediation(_) => {
                if slot == self.requested() {
                    return Some(result.effect());
                }
                let decomposition = result.mediation()?;
                match slot {
                    0 => decomposition.total,
                    1 => decomposition.direct,
                    _ => decomposition.mediated,
                }
            }
            Self::PathSpecific(_) | Self::Nested(_) | Self::Distribution(_) => {
                Some(result.effect())
            }
        }
    }
}

/// What one posterior atom produced.
enum AtomRun {
    /// The explicit executor of the atom's structure answered.
    Fit(Box<StudyResult>),
    /// The atom is not a member of the posterior's class, or its identifier refuses the effect.
    Unidentified,
    /// The identifier accepted the atom but its plug-in could not be evaluated.
    Unevaluable,
}

/// One atom's contribution to one component's mixture.
struct ComponentEval {
    key: u64,
    status: IdentificationStatus,
    estimand: Option<IdentifiedEstimand>,
    value: Option<f64>,
    posterior: Option<CausalPosterior>,
    assumptions: Option<antecedent_core::AssumptionSet>,
    bounds: Option<(f64, f64)>,
    uncertainty: Option<antecedent_estimate::EstimateUncertainty>,
}

impl AtomEvalView for ComponentEval {
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
        false
    }

    fn analytic_se(&self) -> Option<f64> {
        None
    }

    fn influence(&self) -> Option<&[f64]> {
        None
    }

    fn reported_uncertainty(&self) -> Option<antecedent_estimate::EstimateUncertainty> {
        self.uncertainty
    }
}

/// The spread of the draw-wise mixture of the atoms' values on shared resampled tables.
struct ReplicateSe {
    se: Option<f64>,
    ok: u32,
    failed: u32,
    note: String,
}

/// Re-runs the contributing atoms (key and posterior mass) on shared resampled tables.
type Replicate<'a> = &'a dyn Fn(&[(u64, f64)]) -> Result<ReplicateSe, CausalError>;

/// The runs of every distinct atom of a posterior, with the samples they stand for.
struct CrossWorldRuns<'a> {
    kind: CrossWorldKind<'a>,
    /// Posterior weight and atom key of every sample.
    samples: (&'a [f64], &'a [u64]),
    /// Keys of the distinct atoms, aligned with `runs`.
    keys: Vec<u64>,
    runs: Vec<AtomRun>,
    /// What the atoms are, for the disclosure.
    structure: &'a str,
    /// Identifier and estimator the result records.
    strategy: (IdentifierId, EstimatorId),
    started: Instant,
    /// How a frequentist mixture of several atoms forms its standard error, when it can.
    replicates: Option<Replicate<'a>>,
}

/// Whether the answer of an atom has no scalar mean to mix: the interventional distribution of
/// an outcome without a finite mean.
fn scalar_free(kind: CrossWorldKind<'_>, value: f64) -> bool {
    matches!(kind, CrossWorldKind::Distribution(_)) && !value.is_finite()
}

/// One component's mixture over the atoms.
struct ComponentMix {
    mixed: MixedAtomEvals,
    unidentified_mass: f64,
    atoms: usize,
}

/// The requested mixture and the component disclosures needed to publish it.
struct CrossWorldComponents {
    requested: ComponentMix,
    mediation: [Option<f64>; 3],
    report: Vec<String>,
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

/// Whether an error is the query's own refusal on every structure (a shape the explicit route
/// does not execute) rather than a failure of one atom.
fn is_query_refusal(error: &CausalError) -> bool {
    matches!(
        error,
        CausalError::Unsupported { .. } | CausalError::Support { .. } | CausalError::Compile { .. }
    ) && !error.is_identification_refusal()
}

/// What an atom's failed executor means for the mixture: a cancellation or a resource refusal
/// ends the run, the identifier's refusal is the atom's unidentified mass, and any other failure
/// is its unevaluable mass. The first failure that is the query's own refusal is kept in
/// `refusal`, and is the answer when no atom fits at all.
fn atom_outcome_of(
    error: CausalError,
    refusal: &mut Option<CausalError>,
) -> Result<AtomRun, CausalError> {
    if matches!(error, CausalError::Cancelled { .. } | CausalError::Resource { .. }) {
        return Err(error);
    }
    if error.is_identification_refusal() {
        return Ok(AtomRun::Unidentified);
    }
    if refusal.is_none() && is_query_refusal(&error) {
        *refusal = Some(error);
    }
    Ok(AtomRun::Unevaluable)
}

/// The refusal of the query itself when no atom was fitted, else the runs.
fn require_a_fit(
    runs: Vec<AtomRun>,
    refusal: Option<CausalError>,
) -> Result<Vec<AtomRun>, CausalError> {
    if !runs.iter().any(|run| matches!(run, AtomRun::Fit(_))) {
        if let Some(error) = refusal {
            return Err(error);
        }
    }
    Ok(runs)
}

impl super::Study {
    /// Refusals every execution of a cross-world graph posterior shares.
    fn ensure_cross_world_posterior_licensed(
        &self,
        gp: &GraphPosterior,
    ) -> Result<(), CausalError> {
        if gp.lag_masks().is_some() || gp.max_lag().is_some() {
            return Err(CausalError::Unsupported {
                message: "a static cross-world query over a graph posterior requires static \
                          atoms; this posterior carries DBN lag structure",
            });
        }
        if let (CausalQuery::Distribution(query), false) =
            (&self.query, gp.atom_kind() == GraphPosteriorAtomKind::Dag)
        {
            // A mixed graph or a class reads an unconditional finite-discrete table.
            super::static_path::ensure_admg_distribution_licensed(query, self.settings.refute)?;
        }
        Ok(())
    }

    /// Answer the query on the structure of one posterior atom with the explicit executor of
    /// that structure.
    fn cross_world_on_atom(
        &self,
        data: &TabularData,
        gp: &GraphPosterior,
        index: usize,
        kind: CrossWorldKind<'_>,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let mask = gp.atoms()[index].clone();
        let n_vars = gp.n_vars();
        let not_member = || {
            CausalError::not_identified(
                IdentificationStatus::NotIdentified,
                false,
                "the posterior atom is not a member of the posterior's graph class",
            )
        };
        match gp.atom_kind() {
            GraphPosteriorAtomKind::Dag => {
                let dag = dag_from_adjacency_mask(mask, n_vars).map_err(|_| not_member())?;
                self.cross_world_on_dag(data, &dag, kind, physical, ctx)
            }
            GraphPosteriorAtomKind::Admg | GraphPosteriorAtomKind::TemporalAdmg => {
                let admg = admg_from_adjacency_mask(mask, n_vars).map_err(|_| not_member())?;
                self.cross_world_on_admg(data, &admg, kind, physical, ctx)
            }
            GraphPosteriorAtomKind::Cpdag => {
                let cpdag = cpdag_from_adjacency_mask(mask, n_vars).map_err(|_| not_member())?;
                match kind {
                    CrossWorldKind::Mediation(q) => self.execute_mediation_class(
                        data,
                        ClassStructure::Cpdag(&cpdag),
                        q,
                        physical,
                        ctx,
                    ),
                    CrossWorldKind::PathSpecific(q) => self.execute_class_path_specific(
                        data,
                        &cpdag,
                        PathSpecificMixedSource::PathSpecific(q),
                        physical,
                        ctx,
                    ),
                    CrossWorldKind::Nested(q) => self.execute_class_path_specific(
                        data,
                        &cpdag,
                        PathSpecificMixedSource::Nested(q),
                        physical,
                        ctx,
                    ),
                    CrossWorldKind::Distribution(q) => self.execute_class_distribution(
                        data,
                        DistributionClass::Cpdag(&cpdag),
                        q,
                        physical,
                        ctx,
                    ),
                }
            }
            GraphPosteriorAtomKind::Pag => {
                let mark = gp.mark_atom(index);
                let pag = pag_from_adjacency_mask(mask, mark, n_vars).map_err(|_| not_member())?;
                match kind {
                    CrossWorldKind::Mediation(q) => self.execute_mediation_class(
                        data,
                        ClassStructure::Pag(&pag),
                        q,
                        physical,
                        ctx,
                    ),
                    CrossWorldKind::PathSpecific(q) => self.execute_pag_path_specific(
                        data,
                        &pag,
                        PathSpecificMixedSource::PathSpecific(q),
                        physical,
                        ctx,
                    ),
                    CrossWorldKind::Nested(q) => self.execute_pag_path_specific(
                        data,
                        &pag,
                        PathSpecificMixedSource::Nested(q),
                        physical,
                        ctx,
                    ),
                    CrossWorldKind::Distribution(q) => self.execute_class_distribution(
                        data,
                        DistributionClass::Pag(&pag),
                        q,
                        physical,
                        ctx,
                    ),
                }
            }
            _ => Err(CausalError::Unsupported {
                message: "a cross-world query over a graph posterior reads Dag, Admg, \
                          TemporalAdmg, Cpdag or Pag atoms",
            }),
        }
    }

    /// Run the explicit executor on every distinct atom of the posterior, in order of first
    /// appearance. An atom's refusal or failure is its outcome; only a cancellation ends the
    /// run.
    fn run_cross_world_atoms(
        &self,
        data: &TabularData,
        gp: &GraphPosterior,
        kind: CrossWorldKind<'_>,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<(Vec<u64>, Vec<AtomRun>), CausalError> {
        let mut keys: Vec<u64> = Vec::new();
        let mut firsts: Vec<usize> = Vec::new();
        let mut seen = HashSet::new();
        for index in 0..gp.n_graphs() {
            if seen.insert(gp.graph_keys()[index]) {
                keys.push(gp.graph_keys()[index]);
                firsts.push(index);
            }
        }
        let mut runs = Vec::with_capacity(firsts.len());
        let mut refusal = None;
        for index in firsts {
            if ctx.cancellation.is_cancelled() {
                return Err(CausalError::Cancelled {
                    stage: crate::analysis::stage::STAGE_ESTIMATE_POINT,
                });
            }
            runs.push(match self.cross_world_on_atom(data, gp, index, kind, physical, ctx) {
                Ok(result) => AtomRun::Fit(Box::new(result)),
                Err(error) => atom_outcome_of(error, &mut refusal)?,
            });
        }
        Ok((keys, require_a_fit(runs, refusal)?))
    }

    fn cross_world_on_dag(
        &self,
        data: &TabularData,
        dag: &antecedent_graph::Dag,
        kind: CrossWorldKind<'_>,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        match kind {
            CrossWorldKind::Mediation(q) => {
                self.execute_static_mediation_total(data, dag, q, physical, ctx)
            }
            CrossWorldKind::PathSpecific(q) => {
                self.execute_path_specific(data, dag, q, physical, ctx)
            }
            CrossWorldKind::Nested(q) => {
                let admg = Admg::from_dag(dag).map_err(CausalError::rejected)?;
                self.execute_path_specific_mixed(
                    data,
                    &admg,
                    PathSpecificMixedSource::Nested(q),
                    physical,
                    ctx,
                )
            }
            CrossWorldKind::Distribution(q) => {
                self.execute_distribution(data, DistributionGraph::Dag(dag), q, physical, None, ctx)
            }
        }
    }

    fn cross_world_on_admg(
        &self,
        data: &TabularData,
        admg: &Admg,
        kind: CrossWorldKind<'_>,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        match kind {
            CrossWorldKind::Mediation(q) => {
                self.execute_mediation_mixed(data, admg, q, physical, ctx)
            }
            CrossWorldKind::PathSpecific(q) => self.execute_path_specific_mixed(
                data,
                admg,
                PathSpecificMixedSource::PathSpecific(q),
                physical,
                ctx,
            ),
            CrossWorldKind::Nested(q) => self.execute_path_specific_mixed(
                data,
                admg,
                PathSpecificMixedSource::Nested(q),
                physical,
                ctx,
            ),
            CrossWorldKind::Distribution(q) => self.execute_distribution(
                data,
                DistributionGraph::Admg(admg),
                q,
                physical,
                None,
                ctx,
            ),
        }
    }

    /// Mix one component over the atoms. `None` when no atom identifies it.
    fn mix_cross_world_component(
        samples: (&[f64], &[u64]),
        kind: CrossWorldKind<'_>,
        slot: usize,
        by_key: &HashMap<u64, &AtomRun>,
    ) -> Result<Option<ComponentMix>, CausalError> {
        let (weights, sample_keys) = samples;
        let requested = slot == kind.requested();
        let mut flags = Vec::with_capacity(sample_keys.len());
        let mut evals: Vec<ComponentEval> = Vec::new();
        let mut proofs: Vec<CachedGraphPosteriorAtomIdentification> = Vec::new();
        let mut seen = HashSet::new();
        for &key in sample_keys {
            let Some(run) = by_key.get(&key) else {
                flags.push(GraphIdentFlag::Unidentified);
                continue;
            };
            let first = seen.insert(key);
            match run {
                AtomRun::Unidentified => flags.push(GraphIdentFlag::Unidentified),
                AtomRun::Unevaluable => {
                    flags.push(GraphIdentFlag::Identified);
                    if first {
                        evals.push(ComponentEval {
                            key,
                            status: IdentificationStatus::NotIdentified,
                            estimand: None,
                            value: None,
                            posterior: None,
                            assumptions: None,
                            bounds: None,
                            uncertainty: None,
                        });
                    }
                }
                AtomRun::Fit(result) => match kind.component_value(result, slot) {
                    None => flags.push(GraphIdentFlag::Unidentified),
                    Some(value) => {
                        flags.push(GraphIdentFlag::Identified);
                        if first {
                            let estimand = target_estimand(&result.estimand);
                            let effect = result.estimate.as_effect();
                            evals.push(ComponentEval {
                                key,
                                status: result.identification.status,
                                estimand: Some(estimand.clone()),
                                // A distribution over levels without a mean (a categorical
                                // outcome) is evaluable: it carries a placeholder scalar
                                // that the publication removes, so its mass is the atom's
                                // identified mass, not unevaluable mass.
                                value: Some(if scalar_free(kind, value) { 0.0 } else { value }),
                                posterior: requested.then(|| result.posterior.clone()).flatten(),
                                assumptions: effect
                                    .map(|effect| effect.provenance.assumptions.clone()),
                                bounds: atom_identified_bounds(result),
                                uncertainty: requested
                                    .then(|| effect.map(|effect| effect.uncertainty))
                                    .flatten(),
                            });
                            proofs.push(CachedGraphPosteriorAtomIdentification {
                                key,
                                estimand,
                                identification: result.identification.clone(),
                                transport: None,
                            });
                        }
                    }
                },
            }
        }
        if proofs.is_empty() {
            return Ok(None);
        }
        let graphs = WeightedGraphSamples::new(weights.to_vec(), flags, sample_keys.to_vec())?;
        let unidentified_mass = graphs.unidentified_mass();
        let atoms = proofs.len();
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
        Ok(Some(ComponentMix { mixed, unidentified_mass, atoms }))
    }

    /// Mix a mediation, path-restricted natural effect or natural direct effect over the atoms
    /// of a tabular graph posterior.
    ///
    /// # Errors
    ///
    /// A posterior with DBN lag structure, a validation suite other than `none`, no atom that
    /// identifies the requested effect, or a cancelled run.
    pub(super) fn execute_cross_world_graph_posterior(
        &self,
        data: &TabularData,
        gp: &GraphPosterior,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        self.ensure_cross_world_posterior_licensed(gp)?;
        let kind = CrossWorldKind::of(&self.query)?;
        let (keys, runs) = self.run_cross_world_atoms(data, gp, kind, physical, ctx)?;
        let rerun = |atoms: &[(u64, f64)]| {
            self.cross_world_shared_replicates(data, gp, kind, physical, ctx, atoms)
        };
        self.publish_cross_world(
            CrossWorldRuns {
                kind,
                samples: (&gp.weights()[..], &gp.graph_keys()[..]),
                keys,
                runs,
                structure: gp.atom_kind().as_str(),
                strategy: (
                    Self::cross_world_identifier(kind),
                    Self::cross_world_estimator(gp, kind),
                ),
                started,
                replicates: Some(&rerun),
            },
            physical,
        )
    }

    /// Mix and select the requested component while retaining the other mediation values.
    fn cross_world_components(
        samples: (&[f64], &[u64]),
        kind: CrossWorldKind<'_>,
        by_key: &HashMap<u64, &AtomRun>,
    ) -> Result<CrossWorldComponents, CausalError> {
        let requested = kind.requested();
        let mut mixes: Vec<Option<ComponentMix>> = Vec::with_capacity(kind.components());
        for slot in 0..kind.components() {
            mixes.push(Self::mix_cross_world_component(samples, kind, slot, by_key)?);
        }
        let value = |slot: usize| {
            mixes[slot]
                .as_ref()
                .filter(|mix| mix.mixed.mixable_scalar)
                .map(|mix| mix.mixed.estimate.ate)
        };
        let (total, direct, mediated) = match kind {
            CrossWorldKind::Mediation(_) => (value(0), value(1), value(2)),
            _ => (None, None, None),
        };
        let component_report: Vec<String> = mixes
            .iter()
            .enumerate()
            .map(|(slot, mix)| {
                let name = match kind {
                    CrossWorldKind::Mediation(_) => MEDIATION_COMPONENTS[slot],
                    _ => "effect",
                };
                match mix {
                    Some(mix) => format!(
                        "{name}: {} atom(s), unidentified_mass={}, unevaluable_mass={}",
                        mix.atoms, mix.unidentified_mass, mix.mixed.mixture.unevaluable_mass
                    ),
                    None => format!("{name}: absent (no atom identifies it)"),
                }
            })
            .collect();
        let requested = mixes.swap_remove(requested).ok_or_else(|| {
            CausalError::not_identified(
                IdentificationStatus::NotIdentified,
                false,
                &format!(
                    "no atom of the graph posterior identifies the requested {}",
                    kind.label()
                ),
            )
        })?;

        Ok(CrossWorldComponents {
            requested,
            mediation: [total, direct, mediated],
            report: component_report,
        })
    }

    fn cross_world_publication_diagnostics(
        &self,
        kind: CrossWorldKind<'_>,
        class: &str,
        mixed: &MixedAtomEvals,
        accounting: (usize, f64, &[String]),
        notes: (Vec<Diagnostic>, Vec<Diagnostic>),
    ) -> Vec<Diagnostic> {
        let (atoms, unidentified_mass, component_report) = accounting;
        let (refute_notes, uncertainty_notes) = notes;
        let mut diagnostics = mixed.identification.diagnostics.clone();
        if self.settings.inference.is_bayesian() && mixed.posterior.is_none() {
            diagnostics.push(graph_posterior_withheld_diagnostic());
        }
        diagnostics.extend(refute_notes);
        diagnostics.extend(uncertainty_notes);
        diagnostics.push(
            Diagnostic::new(
                CROSS_WORLD_POSTERIOR_CODE,
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                format!(
                    "each {class} posterior atom answered the {} with the executor of its own \
                     structure (a Dag's plug-in, a mixed graph's path-specific identifier, a \
                     class's completion envelope), and the atoms are mixed by posterior \
                     probability per component over the atoms that identify it; components: [{}]; \
                     an atom the identifier refuses is unidentified mass and an atom whose \
                     plug-in cannot be evaluated is unevaluable mass, neither redistributed",
                    kind.label(),
                    component_report.join("; ")
                ),
            )
            .with_fields([
                ("atoms", atoms.to_string()),
                ("unidentified_mass", unidentified_mass.to_string()),
                ("unevaluable_mass", mixed.mixture.unevaluable_mass.to_string()),
            ]),
        );
        diagnostics.push(
            Diagnostic::new(
                "estimate.graph_posterior.structural_aggregation",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                format!(
                    "policy={}; weight_basis=posterior_probability; completion enumeration is \
                     not posterior probability; identified_mass={}; unidentified_mass={}",
                    mixed.policy.as_str(),
                    mixed.mixture.identified_mass,
                    mixed.mixture.unidentified_mass
                ),
            )
            .with_fields(mass_fields(
                Some(mixed.mixture.identified_mass),
                mixed.mixture.unidentified_mass,
            )),
        );
        diagnostics.push(overlap_diagnostic(mixed.estimate.provenance.overlap));
        diagnostics
            .extend(overlap_limit_diagnostic(mixed.estimate.diagnostics.overlap_report.as_ref()));

        diagnostics
    }

    /// Mix each atom's own refuter verdicts using the same contributing posterior mass.
    fn cross_world_refutations(
        &self,
        mixed: &MixedAtomEvals,
        by_key: &HashMap<u64, &AtomRun>,
    ) -> (Vec<antecedent_validate::RefutationReport>, Vec<Diagnostic>) {
        // Each atom ran the suite its own structure has, against its own estimate; the
        // reports of one refuter mix by the posterior weight of the atoms that produced them,
        // and a mixed verdict passes only if every contributing atom passes.
        if self.settings.refute == RefuteSuite::None {
            (Vec::new(), Vec::new())
        } else {
            let per_atom: Vec<(f64, Vec<antecedent_validate::RefutationReport>)> = mixed
                .mixture
                .atoms
                .iter()
                .filter(|atom| atom.value.is_some())
                .filter_map(|atom| match by_key.get(&atom.graph_key).copied() {
                    Some(AtomRun::Fit(result)) => Some((atom.weight, result.refutations.clone())),
                    _ => None,
                })
                .collect();
            mix_atom_refutation_reports(&per_atom)
        }
    }

    /// Mix the runs of every distinct atom, one mixture per component, and publish the
    /// component the query asks for with the others beside it.
    ///
    /// # Errors
    ///
    /// No atom identifies the requested component, or a mixture has no evaluable atom.
    fn publish_cross_world(
        &self,
        input: CrossWorldRuns<'_>,
        physical: &PhysicalExecutionPlan,
    ) -> Result<StudyResult, CausalError> {
        let CrossWorldRuns {
            kind,
            samples,
            keys,
            runs,
            structure: class,
            strategy,
            started,
            replicates,
        } = input;
        let by_key: HashMap<u64, &AtomRun> = keys.iter().copied().zip(runs.iter()).collect();

        let CrossWorldComponents { requested: component, mediation, report: component_report } =
            Self::cross_world_components(samples, kind, &by_key)?;
        let [total, direct, mediated] = mediation;
        let ComponentMix { mut mixed, unidentified_mass, atoms } = component;

        let (treatment, outcome) = kind.treatment_outcome()?;
        let no_mean = matches!(kind, CrossWorldKind::Distribution(_))
            && by_key.values().any(
                |run| matches!(run, AtomRun::Fit(result) if scalar_free(kind, result.effect())),
            );
        let uncertainty_notes = if no_mean {
            Vec::new()
        } else {
            self.mixture_uncertainty(&mut mixed, &by_key, replicates)?
        };
        let distribution = match kind {
            CrossWorldKind::Distribution(_) => Self::mixed_distribution(&by_key, &mixed),
            _ => None,
        };
        let (refutations, refute_notes) = self.cross_world_refutations(&mixed, &by_key);
        let diagnostics = self.cross_world_publication_diagnostics(
            kind,
            class,
            &mixed,
            (atoms, unidentified_mass, &component_report),
            (refute_notes, uncertainty_notes),
        );

        let algo =
            physical.logical.record.discovery_algorithm.as_deref().unwrap_or("graph_posterior");
        let MixedAtomEvals {
            policy,
            identification,
            estimand,
            mut estimate,
            posterior,
            mut mixture,
            ..
        } = mixed;
        if no_mean {
            strip_scalar_free_response(&mut estimate, &mut mixture, &by_key);
        }
        let mut finish = IdentifiedExecuteFinish::new(
            physical,
            identification,
            estimand,
            estimate.clone(),
            Strategy::new(strategy.0, strategy.1),
            treatment,
            outcome,
        )
        .identify_cached(false)
        .wall_time_ns(elapsed_ns(started));
        if distribution.is_some() {
            finish = finish.distribution(distribution);
        }
        if !refutations.is_empty() {
            finish = finish.refutations(refutations);
        }
        if matches!(kind, CrossWorldKind::Mediation(_)) {
            finish = finish.mediation(Some(TemporalMediationEstimate {
                effect: estimate,
                total,
                direct,
                mediated,
            }));
        }
        Ok(self.finish_identified_execute(finish.extras(IdentifiedExecuteExtras {
            identify_provenance: Some(provenance_ids("discover.graph_posterior", algo)),
            estimate_provenance: Some(provenance_ids(
                "estimate.cross_world_graph_posterior",
                policy.as_str(),
            )),
            n_draws: posterior.as_ref().map(|p| u32::try_from(p.draws.n_draws).unwrap_or(u32::MAX)),
            posterior,
            diagnostics: Some(diagnostics),
            structural_response: Some(mixture),
            ..Default::default()
        })))
    }

    /// Mix the natural mediation decomposition over the atoms of a graph posterior of temporal
    /// PAGs.
    ///
    /// Each atom is a stationary template read as a supplied `TemporalPag` is: unfolded over
    /// the window of the pulse and its outcome horizon, every stationary MAG completion
    /// marginalized onto it and read through the ADMGs that share its MAG, the decomposition
    /// identified only when every candidate identifies the requested contrast and evaluated by
    /// the plug-in on the lag-aligned window table. An atom whose window leaves the requested
    /// contrast unidentified keeps its mass as unidentified mass; an atom whose completions
    /// identify different functionals publishes its identified set, and the mixture publishes
    /// the union of the atoms' sets beside the posterior-weighted point.
    ///
    /// # Errors
    ///
    /// A posterior that is not a temporal PAG posterior, a validation suite other than `none`,
    /// no atom that identifies the requested contrast, or a cancelled run.
    pub(super) fn execute_temporal_pag_graph_posterior_mediation(
        &self,
        series: &TimeSeriesData,
        gp: &GraphPosterior,
        query: &MediationQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        query.validate()?;
        let (GraphPosteriorAtomKind::Pag, Some(lag_masks), Some(max_lag)) =
            (gp.atom_kind(), gp.lag_masks(), gp.max_lag())
        else {
            return Err(CausalError::Unsupported {
                message: "a temporal PAG graph posterior needs Pag atoms with per-atom lag masks \
                          and max_lag",
            });
        };
        let variables: Vec<VariableId> =
            series.schema().variables().iter().map(|variable| variable.id).collect();
        let mark_of = |index: usize| gp.mark_atom(index);
        // One run per distinct template; a sample is keyed by the first sample that holds it.
        let mut first_of: HashMap<(AdjacencyMask, LagMask, AdjacencyMask), usize> = HashMap::new();
        let mut firsts: Vec<usize> = Vec::new();
        let mut sample_keys: Vec<u64> = Vec::with_capacity(gp.n_graphs());
        for index in 0..gp.n_graphs() {
            let first = *first_of
                .entry((gp.atoms()[index].clone(), lag_masks[index].clone(), mark_of(index)))
                .or_insert_with(|| {
                    firsts.push(index);
                    index
                });
            sample_keys.push(u64::try_from(first).unwrap_or(u64::MAX));
        }
        let mut keys = Vec::with_capacity(firsts.len());
        let mut runs = Vec::with_capacity(firsts.len());
        let mut refusal = None;
        for index in firsts {
            if ctx.cancellation.is_cancelled() {
                return Err(CausalError::Cancelled {
                    stage: crate::analysis::stage::STAGE_ESTIMATE_POINT,
                });
            }
            keys.push(u64::try_from(index).unwrap_or(u64::MAX));
            let template = temporal_pag_from_dbn_atoms(
                &gp.atoms()[index],
                &lag_masks[index],
                mark_of(index),
                gp.n_vars(),
                max_lag,
                &variables,
            );
            let Ok(template) = template else {
                runs.push(AtomRun::Unidentified);
                continue;
            };
            let mut member = self.clone();
            member.graph = Arc::new(AcceptedGraph::temporal_pag(template));
            runs.push(match member.execute_temporal_pag_mediation(series, query, physical, ctx) {
                Ok(result) => AtomRun::Fit(Box::new(result)),
                Err(error) => atom_outcome_of(error, &mut refusal)?,
            });
        }
        let rerun = |atoms: &[(u64, f64)]| {
            self.temporal_pag_shared_replicates(
                series,
                (gp, &lag_masks[..], max_lag),
                query,
                physical,
                ctx,
                atoms,
            )
        };
        self.publish_cross_world(
            CrossWorldRuns {
                kind: CrossWorldKind::Mediation(query),
                samples: (&gp.weights()[..], &sample_keys),
                keys,
                runs: require_a_fit(runs, refusal)?,
                structure: "temporal_pag",
                strategy: (IdentifierId::PathSpecificNatural, EstimatorId::FunctionalEffect),
                started,
                replicates: Some(&rerun),
            },
            physical,
        )
    }

    /// Re-run every contributing temporal-PAG atom on shared resampled series and read the spread
    /// of the draw-wise mixture of their requested values.
    ///
    /// The series is resampled in circular blocks of consecutive periods, never shorter than
    /// the window a pulse and its outcome horizon span, so every atom reads the same resampled
    /// series and the windows its template needs stay intact inside a block.
    fn temporal_pag_shared_replicates(
        &self,
        series: &TimeSeriesData,
        posterior: (&GraphPosterior, &[LagMask], u32),
        query: &MediationQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
        atoms: &[(u64, f64)],
    ) -> Result<ReplicateSe, CausalError> {
        let (gp, lag_masks, max_lag) = posterior;
        let requested = self.settings.bootstrap_replicates;
        let unavailable =
            |note: &str| ReplicateSe { se: None, ok: 0, failed: 0, note: note.to_owned() };
        if requested < 2 {
            return Ok(unavailable(
                "no bootstrap replicates were requested, so the mixture of several atoms \
                 publishes no interval",
            ));
        }
        if matches!(
            self.declared.sampling_design,
            Some(
                antecedent_core::DeclaredSamplingDesign::ClusteredPanel
                    | antecedent_core::DeclaredSamplingDesign::IndependentSubjectTrajectories
            )
        ) {
            return Ok(unavailable(
                "the declared design needs whole subjects or clusters, and one series carries \
                 no labels to draw them from",
            ));
        }
        let n = series.row_count();
        let horizon = query.horizons.iter().copied().max().unwrap_or(1);
        let window = max_lag as usize + horizon as usize + 1;
        let block = antecedent_data::circular_block_length(1, n).max(window);
        let ids: Vec<VariableId> =
            series.schema().variables().iter().map(|variable| variable.id).collect();
        let columns: Vec<Vec<f64>> =
            match ids.iter().map(|&id| series.float64_values(id)).collect::<Result<_, _>>() {
                Ok(columns) => columns,
                Err(_) => {
                    return Ok(unavailable(
                        "a column cannot be read as floats, so the series cannot be resampled",
                    ));
                }
            };
        let total: f64 = atoms.iter().map(|(_, weight)| *weight).sum();
        if !total.is_finite() || total <= 0.0 {
            return Ok(unavailable("the contributing atoms carry no posterior mass"));
        }
        let members =
            match self.temporal_pag_replicate_members((gp, lag_masks, max_lag), &ids, atoms) {
                Ok(members) => members,
                Err(note) => return Ok(unavailable(note)),
            };
        let slot = CrossWorldKind::Mediation(query).requested();
        let tabular = TabularData::new(series.storage().clone());
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
                SHARED_REPLICATE_STREAM.wrapping_add(0x100 + u64::from(replicate)),
            );
            let picked = resample_rows(n, block, &mut rng);
            let resampled = resampled_series(series, &tabular, &ids, &columns, &picked);
            let Some(resampled) = resampled else {
                failed += 1;
                continue;
            };
            let mut mixed = 0.0;
            for (weight, member) in &members {
                let value = member
                    .execute_temporal_pag_mediation(&resampled, query, physical, ctx)
                    .ok()
                    .and_then(|result| {
                        CrossWorldKind::Mediation(query).component_value(&result, slot)
                    })
                    .filter(|value| value.is_finite());
                let Some(value) = value else {
                    failed += 1;
                    continue 'replicates;
                };
                mixed += weight * value;
            }
            values.push(mixed / total);
        }
        let ok = u32::try_from(values.len()).unwrap_or(u32::MAX);
        let se = replicate_standard_error(&values, requested);
        Ok(ReplicateSe {
            se,
            ok,
            failed,
            note: format!(
                "every contributing temporal-PAG atom was re-run on each of {requested} shared \
                 resampled series (circular blocks of {block} consecutive periods, no shorter \
                 than the {window}-period window); the standard error is the spread of the \
                 draw-wise posterior-weighted mixture over {ok} surviving replicates"
            ),
        })
    }

    /// Resolve every contributing template before any shared resampling stream is drawn.
    fn temporal_pag_replicate_members(
        &self,
        posterior: (&GraphPosterior, &[LagMask], u32),
        ids: &[VariableId],
        atoms: &[(u64, f64)],
    ) -> Result<Vec<(f64, Self)>, &'static str> {
        let (gp, lag_masks, max_lag) = posterior;
        let mark_of = |index: usize| gp.mark_atom(index);
        let mut members: Vec<(f64, Self)> = Vec::with_capacity(atoms.len());
        for (key, weight) in atoms {
            let index = usize::try_from(*key).unwrap_or(usize::MAX);
            if index >= gp.n_graphs() {
                return Err("a contributing atom is outside the posterior");
            }
            let Ok(template) = temporal_pag_from_dbn_atoms(
                &gp.atoms()[index],
                &lag_masks[index],
                mark_of(index),
                gp.n_vars(),
                max_lag,
                ids,
            ) else {
                return Err("a contributing atom no longer builds its template");
            };
            let mut member = self.clone();
            member.settings.bootstrap_replicates = 0;
            member.settings.refute = RefuteSuite::None;
            member.graph = Arc::new(AcceptedGraph::temporal_pag(template));
            members.push((*weight, member));
        }
        Ok(members)
    }

    /// The uncertainty of the mixture of the atoms that contribute a value.
    ///
    /// One contributing atom carries the whole conditional mixture, so its own interval and
    /// posterior are the mixture's. Several atoms: Frequentist re-runs every atom on shared
    /// resampled tables and reads the spread of the draw-wise posterior-weighted mixture;
    /// Bayesian pools the atoms' draws (all run in one context, so draw `d` of every atom is one
    /// draw of the row weights). When neither can be formed the interval is withheld and the
    /// result says why.
    fn mixture_uncertainty(
        &self,
        mixed: &mut MixedAtomEvals,
        by_key: &HashMap<u64, &AtomRun>,
        replicates: Option<Replicate<'_>>,
    ) -> Result<Vec<Diagnostic>, CausalError> {
        let contributing: Vec<(u64, f64, &StudyResult)> = mixed
            .mixture
            .atoms
            .iter()
            .filter(|atom| atom.value.is_some())
            .filter_map(|atom| match by_key.get(&atom.graph_key).copied() {
                Some(AtomRun::Fit(result)) => Some((atom.graph_key, atom.weight, &**result)),
                _ => None,
            })
            .collect();
        let mut notes = Vec::new();
        if let [(_, _, only)] = contributing.as_slice() {
            adopt_single_atom_uncertainty(mixed, only);
            return Ok(notes);
        }
        if contributing.is_empty() {
            return Ok(notes);
        }
        if self.settings.inference.is_bayesian() {
            let items: Vec<(f64, &CausalPosterior)> = contributing
                .iter()
                .filter_map(|(_, weight, result)| result.posterior.as_ref().map(|p| (*weight, p)))
                .collect();
            let pooled = (items.len() == contributing.len())
                .then(|| {
                    pool_atom_posteriors(
                        &items,
                        mixed.identification.status,
                        (mixed.mixture.unidentified_mass, mixed.mixture.unevaluable_mass),
                    )
                })
                .flatten();
            if let Some(pooled) = pooled {
                if let Ok(pooled_estimate) = effect_from_posterior(&pooled) {
                    mixed.estimate = pooled_estimate;
                }
                mixed.posterior = Some(pooled);
                notes.push(Diagnostic::new(
                    UNCERTAINTY_CODE,
                    DiagnosticKind::Scientific,
                    DiagnosticSeverity::Info,
                    "pooled posterior: draw d of the mixture is the posterior-weighted sum of \
                     draw d of every contributing atom, all run in one context so that draw d is \
                     one draw of the row weights",
                ));
            } else {
                notes.push(withheld_uncertainty(
                    "no aggregate posterior: the atoms' draws differ in count or publish no \
                     effect column, so they are not draws of one shared run; the per-atom \
                     posteriors are retained on the structural mixture"
                        .to_owned(),
                ));
            }
            return Ok(notes);
        }
        let Some(replicate) = replicates else {
            notes.push(withheld_uncertainty(
                "the standard error of a mixture of several atoms needs the atoms re-run on \
                 shared resampled tables, which this route does not form; the per-atom intervals \
                 are retained on the structural mixture"
                    .to_owned(),
            ));
            return Ok(notes);
        };
        let atoms: Vec<(u64, f64)> =
            contributing.iter().map(|(key, weight, _)| (*key, *weight)).collect();
        let shared = replicate(&atoms)?;
        if let Some(se) = shared.se {
            mixed.estimate = EffectEstimate::from_parts(
                mixed.estimate.ate,
                f64::NAN,
                BootstrapAccounting {
                    se: Some(se),
                    replicates_ok: Some(shared.ok),
                    replicates_failed: Some(shared.failed),
                    cancelled: false,
                    early_stopped: false,
                },
                mixed.estimate.provenance.assumptions.clone(),
                OverlapPolicy::ExplicitOverride,
                None,
                None,
            );
            notes.push(Diagnostic::new(
                UNCERTAINTY_CODE,
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                shared.note,
            ));
        } else {
            notes.push(withheld_uncertainty(shared.note));
        }
        Ok(notes)
    }

    /// Re-run every contributing atom on shared resampled tables and read the spread of the
    /// draw-wise mixture of their requested values.
    fn cross_world_shared_replicates(
        &self,
        data: &TabularData,
        gp: &GraphPosterior,
        kind: CrossWorldKind<'_>,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
        atoms: &[(u64, f64)],
    ) -> Result<ReplicateSe, CausalError> {
        let requested = self.settings.bootstrap_replicates;
        let unavailable =
            |note: &str| ReplicateSe { se: None, ok: 0, failed: 0, note: note.to_owned() };
        if requested < 2 {
            return Ok(unavailable(
                "no bootstrap replicates were requested, so the mixture of several atoms \
                 publishes no interval",
            ));
        }
        let n = data.row_count();
        let block = match self.declared.sampling_design {
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
        let total: f64 = atoms.iter().map(|(_, weight)| *weight).sum();
        if !total.is_finite() || total <= 0.0 {
            return Ok(unavailable("the contributing atoms carry no posterior mass"));
        }
        let mut first_index: HashMap<u64, usize> = HashMap::new();
        for index in 0..gp.n_graphs() {
            first_index.entry(gp.graph_keys()[index]).or_insert(index);
        }
        let mut quiet = self.clone();
        quiet.settings.bootstrap_replicates = 0;
        quiet.settings.refute = RefuteSuite::None;
        let slot = kind.requested();
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
            for (key, weight) in atoms {
                let value = first_index
                    .get(key)
                    .and_then(|&index| {
                        quiet.cross_world_on_atom(&table, gp, index, kind, physical, ctx).ok()
                    })
                    .and_then(|result| kind.component_value(&result, slot))
                    .filter(|value| value.is_finite());
                let Some(value) = value else {
                    failed += 1;
                    continue 'replicates;
                };
                mixed += weight * value;
            }
            values.push(mixed / total);
        }
        let ok = u32::try_from(values.len()).unwrap_or(u32::MAX);
        let se = replicate_standard_error(&values, requested);
        Ok(ReplicateSe {
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

    /// The estimator the result records: the Dag cell's own for a DAG posterior, the functional
    /// plug-in for every other structure.
    fn cross_world_estimator(gp: &GraphPosterior, kind: CrossWorldKind<'_>) -> EstimatorId {
        match (gp.atom_kind(), kind) {
            (GraphPosteriorAtomKind::Dag, _) => kind.dag_estimator(),
            (_, CrossWorldKind::Distribution(_)) => EstimatorId::FunctionalDistribution,
            _ => EstimatorId::FunctionalEffect,
        }
    }

    /// The interventional distribution a mixture of atoms publishes: the posterior-weighted mean
    /// probability of every outcome level over the atoms that identify and evaluate it, with
    /// the mean the mixture published. An atom a member lacks has probability zero there. The
    /// sampling intervals of the levels are those of a fit to one structure, so a mixture of
    /// several atoms publishes none; a mixture of one atom keeps that atom's own.
    fn mixed_distribution(
        by_key: &HashMap<u64, &AtomRun>,
        mixed: &MixedAtomEvals,
    ) -> Option<InterventionalDistributionEstimate> {
        let mut members: Vec<&[DistributionAtom]> = Vec::new();
        let mut weights: Vec<f64> = Vec::new();
        let mut template: Option<&InterventionalDistributionEstimate> = None;
        for atom in &mixed.mixture.atoms {
            if atom.value.is_none() {
                continue;
            }
            let Some(AtomRun::Fit(result)) = by_key.get(&atom.graph_key).copied() else {
                continue;
            };
            let Some(distribution) = result.distribution() else { continue };
            members.push(&distribution.atoms[..]);
            weights.push(atom.weight);
            template.get_or_insert(distribution);
        }
        let template = template?;
        if members.len() == 1 {
            return Some(template.clone());
        }
        let envelope = envelope_of_members(&members, &weights);
        let mut distribution = template.clone();
        distribution.atoms = envelope
            .iter()
            .map(|level| DistributionAtom {
                outcomes: Arc::clone(&level.outcomes),
                conditioning: Arc::clone(&level.conditioning),
                probability: level.mean,
            })
            .collect();
        distribution.mean = if template.mean.is_finite() { mixed.estimate.ate } else { f64::NAN };
        distribution.se_analytic = f64::NAN;
        distribution.se_bootstrap = None;
        distribution.bootstrap_replicates_ok = None;
        distribution.bootstrap_replicates_failed = None;
        distribution.atom_uncertainty = Arc::from([]);
        distribution.mean_interval = None;
        Some(distribution)
    }

    /// The identifier the query's explicit cell uses.
    fn cross_world_identifier(kind: CrossWorldKind<'_>) -> IdentifierId {
        match kind {
            CrossWorldKind::Distribution(_) => RouteDefaults::DISTRIBUTION.identifier,
            _ => IdentifierId::PathSpecificNatural,
        }
    }

    /// The estimator and identifier a cross-world graph posterior compiles to.
    pub(super) fn cross_world_posterior_strategy(
        &self,
        gp: &GraphPosterior,
    ) -> Result<(IdentifierId, EstimatorId), CausalError> {
        let kind = CrossWorldKind::of(&self.query)?;
        Ok((Self::cross_world_identifier(kind), Self::cross_world_estimator(gp, kind)))
    }
}

/// What the explicit class cell of one CPDAG or PAG posterior atom identifies, for the contract
/// that binds the atom to the mixture.
pub(crate) struct ClassAtomIdentification {
    /// The class envelope read as one identification product.
    pub(crate) result: antecedent_identify::IdentificationResult,
    /// Completions examined.
    pub(crate) cases: usize,
    /// Enumeration weight of the identified completions.
    pub(crate) identified_weight: f64,
    /// Enumeration weight of the unidentified completions.
    pub(crate) unidentified_weight: f64,
    /// Completions whose search was capped.
    pub(crate) truncated_completions: usize,
}

impl ClassAtomIdentification {
    fn from_envelope<G>(
        envelope: &antecedent_identify::IdentificationEnvelope<G>,
        query: CausalQuery,
    ) -> Self {
        Self {
            result: super::envelope_to_identification_result_for(envelope, query),
            cases: envelope.cases.len(),
            identified_weight: envelope.identified_weight.0,
            unidentified_weight: envelope.unidentified_weight.0,
            truncated_completions: envelope.truncated_completions,
        }
    }
}

impl super::Study {
    /// Identify a path-restricted effect, a natural direct effect or an interventional
    /// distribution over the class of posterior sample `index`, exactly as the explicit class
    /// executor does. `None` when the sample is no member of the posterior's class or the class
    /// does not identify (its mass is then unidentified).
    ///
    /// # Errors
    ///
    /// A failure other than an identification refusal.
    pub(crate) fn class_atom_identification(
        &self,
        gp: &GraphPosterior,
        index: usize,
        ctx: &ExecutionContext,
    ) -> Result<Option<ClassAtomIdentification>, CausalError> {
        let kind = CrossWorldKind::of(&self.query)?;
        let identifier = Self::cross_world_identifier(kind);
        let query = match kind {
            CrossWorldKind::Distribution(q) => CausalQuery::Distribution(q.clone()),
            CrossWorldKind::PathSpecific(q) => {
                super::static_path::path_specific_source_parts(
                    PathSpecificMixedSource::PathSpecific(q),
                )
                .0
            }
            CrossWorldKind::Nested(q) => {
                super::static_path::path_specific_source_parts(PathSpecificMixedSource::Nested(q)).0
            }
            CrossWorldKind::Mediation(_) => return Ok(None),
        };
        let distribution = matches!(kind, CrossWorldKind::Distribution(_));
        let mask = gp.atoms()[index].clone();
        let refusal = |error: CausalError| {
            if error.is_identification_refusal() { Ok(None) } else { Err(error) }
        };
        match gp.atom_kind() {
            GraphPosteriorAtomKind::Cpdag => {
                let Ok(cpdag) = cpdag_from_adjacency_mask(mask, gp.n_vars()) else {
                    return Ok(None);
                };
                let envelope = if distribution {
                    crate::strategy_table::identify_cpdag_distribution_in_context(
                        identifier,
                        &cpdag,
                        &query,
                        Some(ctx),
                    )
                } else {
                    crate::strategy_table::identify_cpdag_path_specific_in_context(
                        identifier,
                        &cpdag,
                        &query,
                        Some(ctx),
                    )
                };
                let envelope = match envelope {
                    Ok(envelope) => envelope,
                    Err(error) => return refusal(error),
                };
                if matches!(envelope.status, IdentificationStatus::NotIdentified)
                    || envelope.identified_weight.0 <= 0.0
                {
                    return Ok(None);
                }
                Ok(Some(ClassAtomIdentification::from_envelope(&envelope, self.query.clone())))
            }
            GraphPosteriorAtomKind::Pag => {
                let mark = gp.mark_atom(index);
                let Ok(pag) = pag_from_adjacency_mask(mask, mark, gp.n_vars()) else {
                    return Ok(None);
                };
                let envelope = if distribution {
                    crate::strategy_table::identify_pag_distribution_in_context(
                        identifier,
                        &pag,
                        &query,
                        Some(ctx),
                    )
                } else {
                    crate::strategy_table::identify_pag_path_specific_in_context(
                        identifier,
                        &pag,
                        &query,
                        Some(ctx),
                    )
                };
                let envelope = match envelope {
                    Ok(envelope) => envelope,
                    Err(error) => return refusal(error),
                };
                // A candidate ADMG the identifier refuses refuses the whole class.
                if envelope.unidentified_weight.0 > 0.0
                    || envelope.identified_weight.0 <= 0.0
                    || matches!(envelope.status, IdentificationStatus::NotIdentified)
                {
                    return Ok(None);
                }
                Ok(Some(ClassAtomIdentification::from_envelope(&envelope, self.query.clone())))
            }
            _ => Ok(None),
        }
    }
}

/// Resample `n` rows with replacement, in circular blocks of `block` consecutive rows.
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

/// The pooled posterior of a mixture: draw `d` is the posterior-weighted sum of draw `d` of every
/// atom. `None` when an atom publishes no effect column, or the atoms' draws differ in count so
/// they are not draws of one shared run.
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
        "pooled over the graph posterior: draw d is the posterior-weighted sum of draw d of every \
         contributing atom, all run in one context so that draw d is one draw of the row weights",
    ));
    for (_, other) in items.iter().skip(1) {
        posterior.assumptions.extend_unique(&other.assumptions.entries);
    }
    Some(posterior)
}

fn replicate_standard_error(values: &[f64], requested: u32) -> Option<f64> {
    (values.len() >= 2 && super::bootstrap_has_enough_successes(values.len(), requested as usize))
        .then(|| {
            let count = values.len() as f64;
            let mean = values.iter().sum::<f64>() / count;
            let variance = values.iter().map(|value| (value - mean) * (value - mean)).sum::<f64>()
                / (count - 1.0);
            variance.sqrt()
        })
        .filter(|se| se.is_finite())
}

fn adopt_single_atom_uncertainty(mixed: &mut MixedAtomEvals, only: &StudyResult) {
    if let Some(own) = only.estimate.as_effect() {
        mixed.estimate.uncertainty = own.uncertainty;
        mixed.estimate.scores.influence.clone_from(&own.scores.influence);
    }
    if mixed.posterior.is_none() {
        mixed.posterior.clone_from(&only.posterior);
    }
    if let Some(posterior) = mixed.posterior.as_mut() {
        posterior.unidentified_mass = mixed.mixture.unidentified_mass;
        posterior.unevaluable_mass = mixed.mixture.unevaluable_mass;
    }
}

fn resampled_series(
    series: &TimeSeriesData,
    tabular: &TabularData,
    ids: &[VariableId],
    columns: &[Vec<f64>],
    picked: &[usize],
) -> Option<TimeSeriesData> {
    let replacements: Vec<(VariableId, Arc<[f64]>)> = ids
        .iter()
        .zip(columns)
        .map(|(&id, column)| (id, picked.iter().map(|&row| column[row]).collect()))
        .collect();
    tabular.with_replaced_floats(&replacements).ok().and_then(|table| {
        TimeSeriesData::try_new(table.storage().clone(), series.time_index().clone()).ok()
    })
}

fn withheld_uncertainty(note: String) -> Diagnostic {
    Diagnostic::new(UNCERTAINTY_CODE, DiagnosticKind::Scientific, DiagnosticSeverity::Warning, note)
}

/// Publish outcome levels without exposing the placeholder scalar or a scalar identified set.
fn strip_scalar_free_response(
    estimate: &mut EffectEstimate,
    mixture: &mut crate::result::StructuralResponseMixture,
    by_key: &HashMap<u64, &AtomRun>,
) {
    // The levels are the answer; the placeholder scalar and the set and conditional
    // value read from it are not published, and the masses stay as mixed.
    *estimate = EffectEstimate::new(
        f64::NAN,
        f64::NAN,
        estimate.provenance.assumptions.clone(),
        OverlapPolicy::ExplicitOverride,
    );
    mixture.identified_set = None;
    mixture.identified_set_interval = None;
    mixture.conditional_on_identified = None;
    for atom in &mut mixture.atoms {
        if matches!(by_key.get(&atom.graph_key).copied(), Some(AtomRun::Fit(_))) {
            atom.value = None;
        }
    }
}
