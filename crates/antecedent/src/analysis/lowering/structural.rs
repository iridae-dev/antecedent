//! Lowering of the static mediation and structural-model cells: mediation and nested
//! counterfactuals on a DAG, and the counterfactual, attribution and change queries
//! on a DAG or a lag-aligned temporal DAG.
//!
//! Mediation lowers to the natural-effect theorem target, built here once for the
//! static and temporal mediation executors alike. The structural-model queries
//! lower to their typed functional, with one mechanism model per graph node from
//! the candidate registry the route fits, and the Dirichlet row-weight posterior the
//! Bayesian procedures draw.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::ClaimKind;
use antecedent_core::{
    AssumptionSet, CausalQuery, Conditioning, CrossWorldPremise, GraphFamily, HistoryWindow,
    InferenceSelection, InitialLaw, Lag, LikelihoodModel, MediationArms, MediationComponent,
    MediationContrast, MediationQuery, MediationTarget, ModelAssumptions, NodeRef, ObservationUnit,
    OrdinaryBayesianLikelihood, OutcomeQuantity, Prior, QuantityUnits, ScientificTarget,
    ScoreFrequentist, ScoreModel, TargetForm, TemporalPolicy, TimeFrame, VariableId, WorldBinding,
};
use antecedent_data::{TableView, TabularData};
use antecedent_graph::{Dag, DenseNodeId, TemporalDag};

use super::LoweringContext;
use super::static_response::{
    CellProgram, RealizationChoice, Resampling, assemble, declared_uncertainty,
    inference_selection, record_estimator, retain_or_defer, retained_static, slot, union,
};
use crate::analysis::execute::{
    ANOMALY_DIRICHLET_BACKEND, CHANGE_DIRICHLET_BACKEND, COUNTERFACTUAL_DIRICHLET_BACKEND,
    RouteInput, StaticMediationSource,
};
use crate::error::CausalError;
use crate::planner::lowered::{
    ClaimDemand, FunctionalKind, LoweredProgram, ProblemTarget, UncertaintyReading,
};
use crate::planner::statistical::{
    ModelRole, ModelSlot, PosteriorAlgorithm, PosteriorPlan, Realization,
};
use crate::{EstimatorId, InferenceMode};

/// Why a mediation route quotes no adjustment-design width.
const MEDIATION_WIDTH: &str = "linear mediation fits one regression per mediator and one \
                               for the outcome, not one quoted adjustment design";
/// Why a structural-model route quotes no adjustment-design width.
const MECHANISM_WIDTH: &str = "structural mechanisms are fitted node by node from a candidate \
                               registry, not on one quoted adjustment design";
/// Candidate set the attribution and change routes fit, chosen per node by validation score.
const STANDARD_REGISTRY: &str = "gcm.registry.standard.best_score";
/// Candidate set the counterfactual route fits: the standard families beside the
/// heterogeneity-capable ones, chosen per node by validation score.
const HETEROGENEITY_REGISTRY: &str = "gcm.registry.heterogeneity.best_score";
/// Prior of the Bayesian structural-model procedures: one shared Dirichlet(1, …, 1)
/// weight vector over the observed rows per draw (a Bayesian bootstrap).
const DIRICHLET_ROW_PRIOR: &str = "dirichlet_row_weights(1)";
/// Score of mechanism-change detection.
const MECHANISM_CHANGE_SCORE: &str = "gcm.mechanism_change.likelihood_ratio";

/// Lower one cell of this family.
pub(super) fn lower(cx: &LoweringContext<'_>) -> Result<LoweredProgram, CausalError> {
    match cx.input {
        Some(RouteInput::StaticMediationDag { data, source, .. }) => {
            lower_mediation(cx, data, source)
        }
        Some(RouteInput::GcmDag { graph, .. }) => {
            lower_structural(cx, static_mechanisms(graph, registry(&cx.study.query)))
        }
        Some(
            RouteInput::GcmAttributeDag { graph, .. } | RouteInput::GcmSimulateDag { graph, .. },
        ) => {
            let family = super::discovery::mechanism_label(cx.study);
            lower_structural(cx, static_mechanisms(graph, &family))
        }
        Some(RouteInput::GcmTemporal { .. }) => {
            // The route classifies as this cell only on a supplied temporal DAG.
            let graph = cx
                .study
                .graph
                .as_temporal_dag()
                .ok_or_else(|| super::not_lowered_by("structural", cx))?;
            lower_structural(cx, lagged_mechanisms(graph, registry(&cx.study.query)))
        }
        _ => Err(super::not_lowered_by("structural", cx)),
    }
}

/// Natural-effect mediation on a supplied DAG, from a mediation or a nested
/// counterfactual query.
fn lower_mediation(
    cx: &LoweringContext<'_>,
    data: &TabularData,
    source: StaticMediationSource<'_>,
) -> Result<LoweredProgram, CausalError> {
    let study = cx.study;
    let query = match source {
        StaticMediationSource::Mediation(query) => query.clone(),
        StaticMediationSource::Nested(query) => query.as_mediation_query(),
    };
    let identification = retain_or_defer(cx)?;
    let cache = retained_static(&identification);
    let assumptions = cache
        .as_ref()
        .map_or_else(AssumptionSet::new, |cache| cache.identification.required_assumptions.clone());
    // A mediation the theorem target cannot state (for example arms that are not
    // portable world actions) lowers to its typed path-specific functional.
    let target = theorem_mediation_target(
        &query,
        data.schema(),
        &assumptions,
        inference_selection(&study.inference),
        None,
    )
    .map_or_else(
        |_| super::query_functional(study, FunctionalKind::PathSpecific, Some(query.outcome)),
        ProblemTarget::Target,
    );
    let adjustment: &[VariableId] = match cache.as_ref() {
        Some(cache) => &cache.estimand.adjustment_set,
        None => &[],
    };
    let family = record_estimator(cx).unwrap_or(EstimatorId::StaticMediationLinear).kernel_label();
    let treatment = [query.treatment];
    let mediator_inputs = union(&treatment, adjustment);
    let mut models: Vec<ModelSlot> = query
        .mediators
        .iter()
        .map(|&mediator| slot(ModelRole::Mediator, family, &mediator_inputs, Some(mediator)))
        .collect();
    let outcome_inputs = union(&union(&treatment, &query.mediators), adjustment);
    models.push(slot(ModelRole::Outcome, family, &outcome_inputs, Some(query.outcome)));
    assemble(
        cx,
        CellProgram {
            target,
            identification,
            realization: RealizationChoice::Declared,
            models,
            folds: None,
            resampling: match study.inference {
                // The posterior carries the uncertainty; no bootstrap is drawn.
                InferenceMode::Bayesian(_) => Resampling::Off,
                InferenceMode::Frequentist => Resampling::Declared,
            },
            intervenes: true,
            width: MEDIATION_WIDTH,
            claim: ClaimDemand {
                uncertainty: declared_uncertainty(&study.inference),
                kind: Some(ClaimKind::Point),
            },
        },
    )
}

/// A counterfactual, attribution or change query on a fitted structural model.
fn lower_structural(
    cx: &LoweringContext<'_>,
    models: Vec<ModelSlot>,
) -> Result<LoweredProgram, CausalError> {
    let study = cx.study;
    let kind = match &study.query {
        CausalQuery::Counterfactual(_) => FunctionalKind::Counterfactual,
        CausalQuery::AnomalyAttribution(_) => FunctionalKind::AnomalyAttribution,
        CausalQuery::ChangeAttribution(_) => FunctionalKind::ChangeAttribution,
        CausalQuery::MechanismChange(_) => FunctionalKind::MechanismChange,
        CausalQuery::UnitChange(_) => FunctionalKind::UnitChange,
        CausalQuery::Attribute(_) => FunctionalKind::Attribution,
        CausalQuery::Simulate(_) => FunctionalKind::Simulation,
        _ => return Err(super::not_lowered_by("structural", cx)),
    };
    let (_, outcome) = crate::analysis::execute::gcm_query_vars(&study.query)?;
    let identification = retain_or_defer(cx)?;
    let backend = match kind {
        FunctionalKind::Counterfactual => Some(COUNTERFACTUAL_DIRICHLET_BACKEND),
        FunctionalKind::AnomalyAttribution => Some(ANOMALY_DIRICHLET_BACKEND),
        FunctionalKind::ChangeAttribution => Some(CHANGE_DIRICHLET_BACKEND),
        _ => None,
    };
    let (realization, posterior, uncertainty) = match (kind, backend, &study.inference) {
        (_, Some(backend), InferenceMode::Bayesian(config)) => {
            let (realization, posterior) = dirichlet_row_weights(backend, config.n_draws)?;
            (realization, Some(posterior), UncertaintyReading::Posterior)
        }
        // Detection is a likelihood-ratio test with a sampling law, whatever the
        // study's inference mode.
        (FunctionalKind::MechanismChange, _, _) => (
            Realization::Frequentist(ScoreFrequentist::new(
                ScoreModel::new(MECHANISM_CHANGE_SCORE).map_err(declaration_error)?,
            )),
            None,
            UncertaintyReading::Sampling,
        ),
        // Unit effects, anomaly scores and change shares are computed under the fitted
        // mechanisms and publish no sampling interval.
        _ => (
            Realization::ExactPoint {
                evaluator: Arc::from(record_estimator(cx).unwrap_or(EstimatorId::GcmFit).as_str()),
            },
            None,
            UncertaintyReading::None,
        ),
    };
    assemble(
        cx,
        CellProgram {
            target: super::query_functional(study, kind, Some(outcome)),
            identification,
            realization: RealizationChoice::Override { realization, posterior },
            models,
            folds: None,
            resampling: Resampling::Off,
            intervenes: matches!(kind, FunctionalKind::Counterfactual | FunctionalKind::Simulation),
            width: MECHANISM_WIDTH,
            claim: ClaimDemand { uncertainty, kind: Some(ClaimKind::Point) },
        },
    )
}

/// The candidate registry the route for `query` fits.
fn registry(query: &CausalQuery) -> &'static str {
    if matches!(query, CausalQuery::Counterfactual(_)) {
        HETEROGENEITY_REGISTRY
    } else {
        STANDARD_REGISTRY
    }
}

/// One mechanism per node of a static DAG, reading the node's parents.
fn static_mechanisms(graph: &Dag, family: &str) -> Vec<ModelSlot> {
    let nodes = graph.nodes();
    graph
        .node_ids()
        .zip(nodes)
        .map(|(id, node)| {
            let parents = parent_variables(nodes, graph.parents(id));
            slot(ModelRole::Mechanism, family, &parents, Some(node.variable()))
        })
        .collect()
}

/// One mechanism per contemporaneous node of a temporal DAG, reading the variables of
/// its parents at every lag. The fit runs on the lag-aligned unfolding of this graph.
fn lagged_mechanisms(graph: &TemporalDag, family: &str) -> Vec<ModelSlot> {
    let nodes = graph.nodes();
    let ids = (0..nodes.len()).map(DenseNodeId::from_index);
    ids.zip(nodes)
        .filter(|(_, node)| node.lag().is_some_and(Lag::is_contemporaneous))
        .map(|(id, node)| {
            let mut parents = parent_variables(nodes, graph.parents(id));
            parents.sort_unstable();
            parents.dedup();
            slot(ModelRole::Mechanism, family, &parents, Some(node.variable()))
        })
        .collect()
}

fn parent_variables(nodes: &[NodeRef], parents: &[DenseNodeId]) -> Vec<VariableId> {
    parents
        .iter()
        .filter_map(|parent| nodes.get(parent.as_usize()).copied().map(NodeRef::variable))
        .collect()
}

/// The Bayesian bootstrap the structural-model procedures draw: each draw refits the
/// mechanisms under one shared Dirichlet(1, …, 1) row-weight vector, which is the
/// conjugate posterior of the empirical row law.
fn dirichlet_row_weights(
    backend: &str,
    draws: usize,
) -> Result<(Realization, PosteriorPlan), CausalError> {
    let prior = Prior::new(DIRICHLET_ROW_PRIOR).map_err(declaration_error)?;
    let likelihood = LikelihoodModel::new(backend).map_err(declaration_error)?;
    Ok((
        Realization::Bayesian(OrdinaryBayesianLikelihood::new(likelihood, prior.clone())),
        PosteriorPlan { algorithm: PosteriorAlgorithm::Conjugate, draws, prior },
    ))
}

fn declaration_error(error: antecedent_core::DeclarationError) -> CausalError {
    CausalError::Compile { message: error.to_string() }
}

/// The natural-effect scientific target of a mediation query.
///
/// The natural-effect and linear temporal theorems carry their own premises. This
/// target is retained beside their validated proofs, never accepted by the generic
/// joint-world support validator. Temporal frames describe the observed lag-window law
/// already averaged by the estimator; they do not introduce independent initial-state
/// draws.
///
/// `temporal` is `(horizon, history)` for a temporal mediation, `None` for a static one.
///
/// # Errors
///
/// The query is invalid, its outcome is not in `schema`, or a target component is
/// refused.
pub(crate) fn theorem_mediation_target(
    query: &MediationQuery,
    schema: &antecedent_core::CausalSchema,
    assumptions: &AssumptionSet,
    inference: InferenceSelection,
    temporal: Option<(u32, u32)>,
) -> Result<ScientificTarget, CausalError> {
    query.validate()?;
    let component = match query.contrast {
        MediationContrast::Total => MediationComponent::Total,
        MediationContrast::Direct => MediationComponent::Direct,
        MediationContrast::NaturalDirect => MediationComponent::NaturalDirect,
        MediationContrast::Mediated => MediationComponent::Mediated,
        MediationContrast::NaturalIndirect => MediationComponent::NaturalIndirect,
    };
    let (graph, unit, time, schedule) = match temporal {
        None => (GraphFamily::Dag, "row", TimeFrame::static_decision(), None),
        Some((horizon, history)) => (
            GraphFamily::TemporalDag,
            "lag_window",
            TimeFrame::temporal(
                horizon,
                HistoryWindow::max_lag(history),
                InitialLaw::contemporaneous_observed(),
            )
            .map_err(mediation_target_error)?,
            // TemporalMediationIdentifier sets T at -1 and Y at h-1 (estimator lag h).
            Some(TemporalPolicy::pulse(-1)),
        ),
    };
    let binding = WorldBinding::new(
        query.target_population.clone(),
        ObservationUnit::new(unit).map_err(mediation_target_error)?,
        OutcomeQuantity::mean(
            query.outcome,
            QuantityUnits::new(schema.get(query.outcome)?.unit.as_deref().unwrap_or("unspecified"))
                .map_err(mediation_target_error)?,
        )
        .map_err(mediation_target_error)?,
        Conditioning::unconditional(),
    )
    .map_err(mediation_target_error)?;
    let arms = MediationArms::new(
        query.treatment,
        query.mediators.iter().copied(),
        query.control.clone(),
        query.active.clone(),
        component,
    )
    .map_err(mediation_target_error)?;
    let premises = if component == MediationComponent::Total {
        Vec::new()
    } else {
        vec![
            CrossWorldPremise::Composition,
            CrossWorldPremise::CrossWorldIndependence,
            CrossWorldPremise::PathSpecificNested,
        ]
    };
    let mediation = MediationTarget::new(binding, time, schedule, arms, premises)
        .map_err(mediation_target_error)?;
    Ok(ScientificTarget::new(
        TargetForm::Mediation(mediation),
        ModelAssumptions::new(graph, assumptions.clone()),
        inference,
    ))
}

/// The refusal wording the mediation executors have always reported.
fn mediation_target_error(error: antecedent_core::ScientificTargetError) -> CausalError {
    CausalError::Compile { message: format!("mediation scientific target: {error}") }
}

#[cfg(test)]
#[path = "structural_tests.rs"]
mod tests;
