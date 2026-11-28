//! Compiled planning and ranking of the evidence a caller could collect next.
//!
//! Each entry compiles a plan or a ranking over declared candidates, graphs or posteriors into a
//! program of its own family in the planning route family, and executes the same kernel the
//! direct entry calls on the same inputs:
//!
//! | entry | family | kernel |
//! |---|---|---|
//! | [`compile_identification_repair`] | `IdentificationRepair` | `repair`, which re-runs the failed contract's own theorem checker |
//! | [`compile_study_addition_plan`] | `StudyAdditionPlanning` | `plan_studies` |
//! | [`compile_z_transport_failure_snapshot`], [`compile_z_transport_evidence_plan`] | `ZTransportEvidencePlanning` | `snapshot_z_transport_failure`, `plan_z_transport_evidence` |
//! | [`compile_design_ranking`] | `DesignRanking` | `DesignRanker::rank` |
//! | [`compile_choice_study_ranking`] | `ChoiceStudyRanking` | `rank_studies_with_context`, `rank_hierarchical_studies` |
//! | [`compile_graph_identification`], [`compile_structure_identification`] | `StructureIdentification` | `identifies`, `structure_posterior`, `identified_under_intervention` |
//!
//! A plan is hypothetical: no candidate is evidence until it arrives, a search a budget stopped
//! reports what it did not examine, and a ranking is advisory and runs no analysis.
//!
//! A refusal of the kernel is returned unchanged as [`TerminalError::Kernel`]; the compiler's
//! own refusals (re-entry, a release build's support gate, an evaluation context
//! other than the compiled one) are [`TerminalError::Refused`].
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::EvidenceObligation;
use antecedent_core::{
    CanonicalWriter, EvidenceCatalog, ExecutionContext, InterventionAssignment, SearchLimits,
    Value, VariableId,
};
use antecedent_design::choice_evsi::{
    ChoiceEvsiError, StudyCandidate as ChoiceStudyCandidate, StudyRank, rank_hierarchical_studies,
    rank_studies_with_context,
};
use antecedent_design::structure_identification::{
    GraphIdentFlag, GraphPosteriorSample, StructureIdentificationError,
    identified_under_intervention, identifies, structure_posterior,
};
use antecedent_design::{
    BackdoorRepairFamily, CandidateDesign, DesignConstraints, DesignError, DesignEvaluationContext,
    DesignObjective, DesignRankConfig, DesignRanker, DurableStudyCandidate, FailureRoute,
    FrontDoorRepairFamily, HedgeRepairFamily, InstrumentRepairFamily, RepairError, RepairFamily,
    RepairLimits, RepairObjective, RepairReport, ScoredDesignRanking, SensitivityRepairFamily,
    StudyCandidate, StudyPlanError, StudyPlanResult, TransportRepairFamily,
    ZTransportFailureSnapshot, ZTransportPlanResult, ZTransportPlanSpec, ZTransportPlanningError,
    ZTransportRepairFamily, plan_studies, plan_z_transport_evidence, snapshot_z_transport_failure,
};
use antecedent_design::{DesignCost, FailedContract, TransportEvidenceCandidate};
use antecedent_graph::{Admg, Dag, SelectionDiagram};
use antecedent_identify::{SidLimits, StudyPlanLimits, StudyPlanRoute, ZTransportQuery};
use antecedent_io_core::transport_catalog_wire::{EvidenceCatalogDeltaWire, EvidenceCatalogWire};
use antecedent_io_core::transport_query_wire::{
    MixedSourceQueryWire, MzTransportQueryWire, ZTransportQueryWire,
};
use antecedent_prob::WeightedGraphSamples;
use antecedent_stats::adaptive_inference::HierarchicalChoicePrior;

use crate::analysis::supplied_identity::{
    Canonical, Declaration, Wire, encoded, unencodable, unencoded_variant, variant,
};
pub use crate::analysis::supplied_terminal::TerminalError;
use crate::analysis::supplied_terminal::{
    Reading, TerminalSpec, admit_compile, compiled_terminal, lower,
};
use crate::error::CausalError;
use crate::planner::lowered::{Composition, EvidenceRole, FunctionalKind, ProgramCell};

// --- identification repair -------------------------------------------------------------------

/// A failed contract whose evidence obligations a repair search tries to close.
#[derive(Clone, Debug)]
#[allow(
    clippy::large_enum_variant,
    reason = "a contract is built once per repair search and moved into its program"
)]
pub enum RepairContract {
    /// A theorem route of a failed identification: back-door, front-door, instrument, hedge or
    /// sensitivity.
    Route(FailureRoute),
    /// A catalog-aware classical transport contract missing evidence.
    Transport(TransportRepairFamily),
    /// A checked z-transport proof missing required factors.
    ZTransport(ZTransportRepairFamily),
}

impl RepairContract {
    /// The repair family whose theorem checker decides sufficiency.
    #[must_use]
    pub fn family(&self) -> &dyn RepairFamily {
        match self {
            Self::Route(route) => route.family(),
            Self::Transport(family) => family,
            Self::ZTransport(family) => family,
        }
    }
}

impl From<FailureRoute> for RepairContract {
    fn from(route: FailureRoute) -> Self {
        Self::Route(route)
    }
}

impl From<BackdoorRepairFamily> for RepairContract {
    fn from(family: BackdoorRepairFamily) -> Self {
        Self::Route(FailureRoute::Backdoor(Box::new(family)))
    }
}

impl From<FrontDoorRepairFamily> for RepairContract {
    fn from(family: FrontDoorRepairFamily) -> Self {
        Self::Route(FailureRoute::FrontDoor(Box::new(family)))
    }
}

impl From<InstrumentRepairFamily> for RepairContract {
    fn from(family: InstrumentRepairFamily) -> Self {
        Self::Route(FailureRoute::Instrument(Box::new(family)))
    }
}

impl From<HedgeRepairFamily> for RepairContract {
    fn from(family: HedgeRepairFamily) -> Self {
        Self::Route(FailureRoute::Hedge(Box::new(family)))
    }
}

impl From<SensitivityRepairFamily> for RepairContract {
    fn from(family: SensitivityRepairFamily) -> Self {
        Self::Route(FailureRoute::Sensitivity(Box::new(family)))
    }
}

impl From<TransportRepairFamily> for RepairContract {
    fn from(family: TransportRepairFamily) -> Self {
        Self::Transport(family)
    }
}

impl From<ZTransportRepairFamily> for RepairContract {
    fn from(family: ZTransportRepairFamily) -> Self {
        Self::ZTransport(family)
    }
}

compiled_terminal! {
    /// A compiled identification-repair search over declared candidate studies.
    #[derive(Clone, Debug)]
    CompiledIdentificationRepair {
        contract: RepairContract,
        candidates: Vec<DurableStudyCandidate>,
        objective: RepairObjective,
        limits: RepairLimits,
    }
}

/// Compile the search for the candidates (and bounded subsets of them) that repair a failed
/// contract.
///
/// The program's identity covers the contract's family and stable identity, its unresolved
/// obligations, every candidate, the objective and the limits. Nothing is searched at compile.
///
/// # Errors
/// `public_builder_reentry` inside a lowered dispatch.
pub fn compile_identification_repair(
    contract: impl Into<RepairContract>,
    candidates: Vec<DurableStudyCandidate>,
    objective: RepairObjective,
    limits: RepairLimits,
    ctx: &ExecutionContext,
) -> Result<CompiledIdentificationRepair, CausalError> {
    admit_compile(ctx)?;
    let contract = contract.into();
    encodable(&contract)?;
    let question = encoded("supplied.identification_repair.v1", || {
        Declaration::new("antecedent.identification_repair.v2")
            .field("contract", &contract)
            // The search reads the candidates in semantic-id order (a stable sort; a repeated id
            // is refused as a duplicate), so they are written in that order and never deduplicated.
            .field(
                "candidates",
                &KernelOrdered::by(
                    &candidates,
                    DurableStudyCandidate::semantic_id,
                    durable_candidate,
                ),
            )
            .field("objective", &RepairObjectiveField(objective))
            .field("limits", &RepairLimitsField(limits))
            .finish()
    })?;
    let lowered = lower(TerminalSpec {
        cell: ProgramCell::IdentificationRepair,
        plan_id: "supplied.identification_repair.v1",
        functional: FunctionalKind::StructureSearch,
        graph_class: "failed_identification_contract",
        question,
        evaluator: "repair::theorem_checker_search",
        reading: Reading::Exact,
        role: EvidenceRole::Trial,
        composition: Composition::Single,
        decision: None,
        claim: None,
    })?;
    Ok(CompiledIdentificationRepair { lowered, contract, candidates, objective, limits })
}

impl CompiledIdentificationRepair {
    /// The declared contract.
    #[must_use]
    pub const fn contract(&self) -> &RepairContract {
        &self.contract
    }

    /// The declared candidates.
    #[must_use]
    pub fn candidates(&self) -> &[DurableStudyCandidate] {
        &self.candidates
    }

    /// Search the candidates and their bounded subsets with the family's own theorem checker.
    ///
    /// # Errors
    /// The search's own refusals (bounds beyond the declared ones, cost units that disagree, a
    /// budget stopped before the search), or the compiler's.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<RepairReport, TerminalError<RepairError>> {
        let (scoped, publication) = self.enter(ctx)?;
        publication.finish(
            antecedent_design::repair(
                self.contract.family(),
                &self.candidates,
                self.objective,
                self.limits,
                &scoped,
            )
            .map_err(TerminalError::Kernel),
        )
    }
}

// --- study addition planning -----------------------------------------------------------------

compiled_terminal! {
    /// A compiled plan of the cheapest declared studies that would make a failed multi-source
    /// or mixed-source decision identify.
    #[derive(Clone, Debug)]
    CompiledStudyAdditionPlan {
        graph: Admg,
        route: StudyPlanRoute,
        catalog: EvidenceCatalog,
        candidates: Vec<StudyCandidate>,
        limits: StudyPlanLimits,
    }
}

/// Compile a study-addition plan.
///
/// # Errors
/// `public_builder_reentry` inside a lowered dispatch.
pub fn compile_study_addition_plan(
    graph: Admg,
    route: StudyPlanRoute,
    catalog: EvidenceCatalog,
    candidates: Vec<StudyCandidate>,
    limits: StudyPlanLimits,
    ctx: &ExecutionContext,
) -> Result<CompiledStudyAdditionPlan, CausalError> {
    admit_compile(ctx)?;
    let question = encoded("supplied.study_addition_planning.v1", || {
        Declaration::new("antecedent.study_addition_plan.v2")
            .field("graph", &graph)
            .field("route", &RouteField(&route))
            .field("catalog", &CatalogField(&catalog))
            // The planner reads the candidates sorted by id (a stable sort), so they are written in
            // that order and never deduplicated.
            .field(
                "candidates",
                &KernelOrdered::by(
                    &candidates,
                    |c: &StudyCandidate| Arc::clone(&c.id),
                    study_candidate,
                ),
            )
            .field("limits", &StudyPlanLimitsField(limits))
            .finish()
    })?;
    let lowered = lower(TerminalSpec {
        cell: ProgramCell::StudyAdditionPlanning,
        plan_id: "supplied.study_addition_planning.v1",
        functional: FunctionalKind::StructureSearch,
        graph_class: "admg",
        question,
        evaluator: "study_planner::plan_studies",
        reading: Reading::Exact,
        role: EvidenceRole::TransportSource,
        composition: Composition::Pooled { sources: candidates.len() },
        decision: None,
        claim: None,
    })?;
    Ok(CompiledStudyAdditionPlan { lowered, graph, route, catalog, candidates, limits })
}

impl CompiledStudyAdditionPlan {
    /// Plan the cheapest sufficient subset under the one declared budget.
    ///
    /// # Errors
    /// The planner's own refusals, or the compiler's.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<StudyPlanResult, TerminalError<StudyPlanError>> {
        let (scoped, publication) = self.enter(ctx)?;
        publication.finish(
            plan_studies(
                &self.graph,
                &self.route,
                &self.catalog,
                &self.candidates,
                self.limits,
                &scoped,
            )
            .map_err(TerminalError::Kernel),
        )
    }
}

// --- z-transport evidence planning -----------------------------------------------------------

compiled_terminal! {
    /// A compiled capture of a failed z-transport identification under declared limits.
    #[derive(Clone, Debug)]
    CompiledZTransportFailureSnapshot {
        diagram: SelectionDiagram,
        query: ZTransportQuery,
        catalog: EvidenceCatalog,
        limits: SidLimits,
    }
}

/// Compile the capture of a failed z-transport identification.
///
/// # Errors
/// `public_builder_reentry` inside a lowered dispatch.
pub fn compile_z_transport_failure_snapshot(
    diagram: SelectionDiagram,
    query: ZTransportQuery,
    catalog: EvidenceCatalog,
    limits: SidLimits,
    ctx: &ExecutionContext,
) -> Result<CompiledZTransportFailureSnapshot, CausalError> {
    admit_compile(ctx)?;
    let question = encoded("supplied.z_transport_failure_snapshot.v1", || {
        Declaration::new("antecedent.z_transport_failure_snapshot.v2")
            .field("diagram", &DiagramField(&diagram))
            .field("query", &Wire(&ZTransportQueryWire::from_query(&query)))
            .field("catalog", &CatalogField(&catalog))
            .field("limits", &SidLimitsField(limits))
            .finish()
    })?;
    let lowered = lower(TerminalSpec {
        cell: ProgramCell::ZTransportEvidencePlanning,
        plan_id: "supplied.z_transport_failure_snapshot.v1",
        functional: FunctionalKind::StructureSearch,
        graph_class: "selection_diagram",
        question,
        evaluator: "z_transport_planner::snapshot_failure",
        reading: Reading::Exact,
        role: EvidenceRole::TransportSource,
        composition: Composition::Transported,
        decision: None,
        claim: None,
    })?;
    Ok(CompiledZTransportFailureSnapshot { lowered, diagram, query, catalog, limits })
}

impl CompiledZTransportFailureSnapshot {
    /// Run the supported identifier and freeze its failed outcome.
    ///
    /// # Errors
    /// The snapshot's own refusals (no failure to snapshot, an invalid catalog, an
    /// unclassified identification error), or the compiler's.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<ZTransportFailureSnapshot, TerminalError<ZTransportPlanningError>> {
        let (scoped, publication) = self.enter(ctx)?;
        publication.finish(
            snapshot_z_transport_failure(
                &self.diagram,
                &self.query,
                &self.catalog,
                self.limits,
                &scoped,
            )
            .map_err(TerminalError::Kernel),
        )
    }
}

compiled_terminal! {
    /// A compiled assessment of declared z-transport evidence candidates against a frozen
    /// failure.
    #[derive(Clone, Debug)]
    CompiledZTransportEvidencePlan {
        snapshot: Arc<ZTransportFailureSnapshot>,
        spec: ZTransportPlanSpec,
    }
}

/// Compile the assessment of `spec`'s candidates against the frozen `snapshot`.
///
/// # Errors
/// `public_builder_reentry` inside a lowered dispatch.
pub fn compile_z_transport_evidence_plan(
    snapshot: Arc<ZTransportFailureSnapshot>,
    spec: ZTransportPlanSpec,
    ctx: &ExecutionContext,
) -> Result<CompiledZTransportEvidencePlan, CausalError> {
    admit_compile(ctx)?;
    let question = encoded("supplied.z_transport_evidence_plan.v1", || {
        Declaration::new("antecedent.z_transport_evidence_plan.v2")
            .field("snapshot", &SnapshotField(&snapshot))
            .field("spec", &PlanSpecField(&spec))
            .finish()
    })?;
    let lowered = lower(TerminalSpec {
        cell: ProgramCell::ZTransportEvidencePlanning,
        plan_id: "supplied.z_transport_evidence_plan.v1",
        functional: FunctionalKind::StructureSearch,
        graph_class: "selection_diagram",
        question,
        evaluator: "z_transport_planner::plan_evidence",
        reading: Reading::Exact,
        role: EvidenceRole::TransportSource,
        composition: Composition::Pooled { sources: spec.candidates.len() },
        decision: None,
        claim: None,
    })?;
    Ok(CompiledZTransportEvidencePlan { lowered, snapshot, spec })
}

impl CompiledZTransportEvidencePlan {
    /// Assess up to the declared number of candidates and rank the checked proposals.
    ///
    /// # Errors
    /// The planner's own refusals (a zero evaluation budget, cancellation), or the compiler's.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<ZTransportPlanResult, TerminalError<ZTransportPlanningError>> {
        let (scoped, publication) = self.enter(ctx)?;
        publication.finish(
            plan_z_transport_evidence(&self.snapshot, &self.spec, &scoped)
                .map_err(TerminalError::Kernel),
        )
    }
}

// --- design ranking --------------------------------------------------------------------------

/// The identity of a design evaluation context: the graph posterior, every unlock list and flag
/// vector, the effect-width and model-likelihood inputs, and the decision registry. A registry
/// holds utility, constraint and signal callbacks and generic state draws, which have no
/// canonical encoding: it is recorded as a bound callback whose declared identity covers what
/// it declares (each registered problem's id, action and constraint counts and chance
/// threshold, a normal prior's moments or the number of state draws, and the signal's name).
fn evaluation_digest<A, O>(eval: &DesignEvaluationContext<'_, A, O>) -> [u8; 32] {
    let mut declaration = Declaration::new("antecedent.design_ranking.evaluation.v2");
    declaration
        .field("graphs", &GraphSamplesField(eval.graphs))
        .field("effect_width", &eval.effect_width.map(EffectWidthField))
        .field("model_loglik", &eval.model_loglik.map(ModelLoglikField))
        .field("query_id_unlock", &eval.query_id_unlock.map(QueryUnlockField))
        .field("env_id_unlock", &eval.env_id_unlock.map(EnvironmentUnlockField))
        .field("identified_under_intervention", &eval.identified_under_intervention.map(FlagsField))
        .field("graph_features", &eval.graph_features.map(<[u32]>::to_vec));
    match eval.decisions {
        Some(registry) => {
            declaration.field("decisions", &true);
            declaration.callback("decision_registry", Some(registry_identity(registry).as_str()));
        }
        None => {
            declaration.field("decisions", &false);
        }
    }
    declaration.finish()
}

compiled_terminal! {
    /// A compiled ranking of candidate designs under an objective.
    #[derive(Clone, Debug)]
    CompiledDesignRanking {
        config: DesignRankConfig,
        constraints: DesignConstraints,
        objective: DesignObjective,
        candidates: Vec<CandidateDesign>,
        evaluation: [u8; 32],
    }
}

/// Compile the ranking `ranker` gives `candidates` under `objective` and `eval`.
///
/// The program's identity covers the ranker's configuration and hard constraints, the
/// objective, every candidate and the evaluation context's digest (the graph posterior, its
/// unlock lists and flags). Nothing is scored at compile.
///
/// # Errors
/// `public_builder_reentry` inside a lowered dispatch.
pub fn compile_design_ranking<A, O>(
    ranker: &DesignRanker,
    objective: DesignObjective,
    candidates: Vec<CandidateDesign>,
    eval: &DesignEvaluationContext<'_, A, O>,
    ctx: &ExecutionContext,
) -> Result<CompiledDesignRanking, CausalError> {
    admit_compile(ctx)?;
    let evaluation = encoded("supplied.design_ranking.v1", || evaluation_digest(eval))?;
    let question = encoded("supplied.design_ranking.v1", || {
        Declaration::new("antecedent.design_ranking.v2")
            .field("config", &RankConfigField(&ranker.config))
            .field("constraints", &ConstraintsField(&ranker.constraints))
            .field("objective", &ObjectiveField(&objective))
            // The ranking reports candidates by their declared index, so their order is kept.
            .field("candidates", &candidates.iter().map(CandidateField).collect::<Vec<_>>())
            .field("evaluation", &Bytes32(evaluation))
            .finish()
    })?;
    let lowered = lower(TerminalSpec {
        cell: ProgramCell::DesignRanking,
        plan_id: "supplied.design_ranking.v1",
        functional: FunctionalKind::StructureSearch,
        graph_class: "weighted_graph_posterior",
        question,
        evaluator: "ranker::rank",
        reading: Reading::MonteCarlo,
        role: EvidenceRole::DiscoveredStructure,
        composition: Composition::AtomMixture { atoms: eval.graphs.n_samples },
        decision: None,
        claim: None,
    })?;
    Ok(CompiledDesignRanking {
        lowered,
        config: ranker.config.clone(),
        constraints: ranker.constraints.clone(),
        objective,
        candidates,
        evaluation,
    })
}

impl CompiledDesignRanking {
    /// Rank the candidates under the evaluation context the program was compiled for.
    ///
    /// # Errors
    /// An evaluation context other than the compiled one, the ranker's own refusals, or the
    /// compiler's.
    pub fn execute<A, O>(
        &self,
        eval: &DesignEvaluationContext<'_, A, O>,
        ctx: &ExecutionContext,
    ) -> Result<ScoredDesignRanking, TerminalError<DesignError>>
    where
        A: Clone + Sync,
        O: Clone + Sync,
    {
        let (scoped, publication) = self.enter(ctx)?;
        if evaluation_digest(eval) != self.evaluation {
            return Err(TerminalError::Refused(CausalError::Compile {
                message: "design_ranking: the evaluation context differs from the one the \
                          ranking was compiled for; compile the ranking for this context"
                    .to_owned(),
            }));
        }
        publication.finish(
            DesignRanker::new()
                .with_config(self.config.clone())
                .with_constraints(self.constraints.clone())
                .rank(&self.objective, &self.candidates, eval, &scoped)
                .map_err(TerminalError::Kernel),
        )
    }
}

// --- choice study ranking --------------------------------------------------------------------

/// The posterior a choice-study ranking values further choices under.
#[derive(Clone, Debug, PartialEq)]
pub enum ChoicePosterior {
    /// A pooled Dirichlet posterior of the option shares; Monte Carlo beyond the enumeration
    /// cap uses `seed`.
    Pooled {
        /// Dirichlet posterior shapes, one per option.
        alpha: Vec<f64>,
        /// Seed of the Monte Carlo stream.
        seed: u64,
    },
    /// A hierarchical choice posterior: the declared prior and every respondent's counts.
    Hierarchical {
        /// The declared hierarchical prior.
        prior: HierarchicalChoicePrior,
        /// Choice counts per respondent.
        respondents: Vec<Vec<u32>>,
    },
}

compiled_terminal! {
    /// A compiled ranking of further choice studies by expected value of sample information.
    #[derive(Clone, Debug)]
    CompiledChoiceStudyRanking {
        posterior: ChoicePosterior,
        values: Vec<f64>,
        candidates: Vec<ChoiceStudyCandidate>,
    }
}

/// Compile the ranking of `candidates` by EVSI net of cost under `posterior`.
///
/// # Errors
/// `public_builder_reentry` inside a lowered dispatch.
pub fn compile_choice_study_ranking(
    posterior: ChoicePosterior,
    values: Vec<f64>,
    candidates: Vec<ChoiceStudyCandidate>,
    ctx: &ExecutionContext,
) -> Result<CompiledChoiceStudyRanking, CausalError> {
    admit_compile(ctx)?;
    let question = encoded("supplied.choice_study_ranking.v1", || {
        Declaration::new("antecedent.choice_study_ranking.v2")
            .field("posterior", &posterior)
            // Values are per option and candidates are reported in declared order: both ordered.
            .field("values", &values)
            .field("candidates", &candidates.iter().map(ChoiceCandidateField).collect::<Vec<_>>())
            .finish()
    })?;
    let lowered = lower(TerminalSpec {
        cell: ProgramCell::ChoiceStudyRanking,
        plan_id: "supplied.choice_study_ranking.v1",
        functional: FunctionalKind::PolicyValue,
        graph_class: "categorical_choice",
        question,
        evaluator: match &posterior {
            ChoicePosterior::Pooled { .. } => "choice_evsi::rank_studies",
            ChoicePosterior::Hierarchical { .. } => "choice_evsi::rank_hierarchical_studies",
        },
        reading: Reading::Posterior,
        role: EvidenceRole::PriorArtifact,
        composition: Composition::Single,
        decision: None,
        claim: Some(antecedent_contract::ClaimKind::Point),
    })?;
    Ok(CompiledChoiceStudyRanking { lowered, posterior, values, candidates })
}

impl CompiledChoiceStudyRanking {
    /// Rank the candidate studies.
    ///
    /// # Errors
    /// The ranking's own refusals (inputs outside their domain, a work or admission limit,
    /// cancellation), or the compiler's.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<Vec<StudyRank>, TerminalError<ChoiceEvsiError>> {
        let (scoped, publication) = self.enter(ctx)?;
        publication.finish(
            match &self.posterior {
                ChoicePosterior::Pooled { alpha, seed } => {
                    rank_studies_with_context(alpha, &self.values, &self.candidates, *seed, &scoped)
                }
                ChoicePosterior::Hierarchical { prior, respondents } => rank_hierarchical_studies(
                    prior,
                    respondents,
                    &self.values,
                    &self.candidates,
                    &scoped,
                ),
            }
            .map_err(TerminalError::Kernel),
        )
    }
}

// --- structure identification ----------------------------------------------------------------

/// The identification flags of a structure posterior.
#[derive(Clone, Debug, PartialEq)]
pub struct StructureIdentificationReadout {
    /// The posterior with each structure flagged by the complete ID algorithm.
    pub posterior: WeightedGraphSamples,
    /// Each structure's flag on its mutilated graph, when interventions were declared.
    pub under_intervention: Option<Vec<GraphIdentFlag>>,
}

compiled_terminal! {
    /// A compiled run of the complete ID algorithm on one graph.
    #[derive(Clone, Debug)]
    CompiledGraphIdentification {
        graph: Admg,
        treatment: VariableId,
        outcome: VariableId,
    }
}

fn structure_spec(question: [u8; 32], plan_id: &'static str, atoms: usize) -> TerminalSpec {
    TerminalSpec {
        cell: ProgramCell::StructureIdentification,
        plan_id,
        functional: FunctionalKind::StructureSearch,
        graph_class: "admg",
        question,
        evaluator: "structure_identification::id_algorithm",
        reading: Reading::Exact,
        role: EvidenceRole::DiscoveredStructure,
        composition: if atoms > 1 {
            Composition::AtomMixture { atoms }
        } else {
            Composition::Single
        },
        decision: None,
        claim: None,
    }
}

/// Compile whether the complete ID algorithm identifies `P(outcome | do(treatment))` in
/// `graph`.
///
/// # Errors
/// `public_builder_reentry` inside a lowered dispatch.
pub fn compile_graph_identification(
    graph: Admg,
    treatment: VariableId,
    outcome: VariableId,
    ctx: &ExecutionContext,
) -> Result<CompiledGraphIdentification, CausalError> {
    admit_compile(ctx)?;
    let question = encoded("supplied.graph_identification.v1", || {
        Declaration::new("antecedent.graph_identification.v2")
            .field("graph", &graph)
            .field("treatment", &treatment)
            .field("outcome", &outcome)
            .finish()
    })?;
    let lowered = lower(structure_spec(question, "supplied.graph_identification.v1", 1))?;
    Ok(CompiledGraphIdentification { lowered, graph, treatment, outcome })
}

impl CompiledGraphIdentification {
    /// Run the ID algorithm.
    ///
    /// # Errors
    /// Unknown variables or a budget stop of the identifier (never read as
    /// non-identification), or the compiler's refusals.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<GraphIdentFlag, TerminalError<StructureIdentificationError>> {
        let (_scoped, publication) = self.enter(ctx)?;
        publication.finish(
            identifies(&self.graph, self.treatment, self.outcome).map_err(TerminalError::Kernel),
        )
    }
}

compiled_terminal! {
    /// A compiled identification of every structure of a weighted graph posterior, and of each
    /// structure's mutilated graph under declared interventions.
    #[derive(Clone, Debug)]
    CompiledStructureIdentification {
        samples: Vec<GraphPosteriorSample>,
        treatment: VariableId,
        outcome: VariableId,
        interventions: Option<Vec<VariableId>>,
    }
}

/// Compile the identification flags of `samples` for `P(outcome | do(treatment))`, and, when
/// `interventions` are declared, of each structure after an experiment that sets them.
///
/// # Errors
/// `public_builder_reentry` inside a lowered dispatch.
pub fn compile_structure_identification(
    samples: Vec<GraphPosteriorSample>,
    treatment: VariableId,
    outcome: VariableId,
    interventions: Option<Vec<VariableId>>,
    ctx: &ExecutionContext,
) -> Result<CompiledStructureIdentification, CausalError> {
    admit_compile(ctx)?;
    let question = encoded("supplied.structure_identification.v1", || {
        Declaration::new("antecedent.structure_identification.v2")
            // The flags are reported per sample in declared order, so the samples stay ordered.
            .field("samples", &samples.iter().map(SampleField).collect::<Vec<_>>())
            .field("treatment", &treatment)
            .field("outcome", &outcome)
            // Ordered: an unknown target is reported by the first one met.
            .field("interventions", &interventions)
            .finish()
    })?;
    let lowered =
        lower(structure_spec(question, "supplied.structure_identification.v1", samples.len()))?;
    Ok(CompiledStructureIdentification { lowered, samples, treatment, outcome, interventions })
}

impl CompiledStructureIdentification {
    /// Flag every structure, and every mutilated structure when interventions were declared.
    ///
    /// # Errors
    /// An empty or invalid posterior, an unknown variable or target, a budget stop of the
    /// identifier, or the compiler's refusals.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<StructureIdentificationReadout, TerminalError<StructureIdentificationError>> {
        let (_scoped, publication) = self.enter(ctx)?;
        let posterior = structure_posterior(&self.samples, self.treatment, self.outcome)
            .map_err(TerminalError::Kernel)?;
        let under_intervention = self
            .interventions
            .as_ref()
            .map(|targets| {
                identified_under_intervention(&self.samples, self.treatment, self.outcome, targets)
            })
            .transpose()
            .map_err(TerminalError::Kernel)?;
        publication.finish(Ok(StructureIdentificationReadout { posterior, under_intervention }))
    }
}

// --- canonical encodings of the declared inputs ---------------------------------------------
//
// Every encoder writes named-order fields through the shared `Canonical` rules
// (`crate::analysis::supplied_identity`): integers as words, reals as their bits, texts
// length-prefixed, enum variants by tag. Where the kernel or io-core already defines a canonical
// identity or wire for a value, the encoder writes that and says so.

/// A list the kernel reads in the order of a key it sorts by (a stable sort): written in that
/// order, so a declaration order the kernel ignores never reaches the identity, while items with
/// equal keys keep their declared order and a repeated item is kept (the kernel refuses or
/// reports it), never deduplicated.
struct KernelOrdered<'a, T> {
    items: Vec<&'a T>,
    encode: fn(&T, &mut CanonicalWriter),
}

impl<'a, T> KernelOrdered<'a, T> {
    fn by<K: Ord>(
        items: &'a [T],
        key: impl Fn(&T) -> K,
        encode: fn(&T, &mut CanonicalWriter),
    ) -> Self {
        let mut items: Vec<&T> = items.iter().collect();
        items.sort_by_key(|item| key(item));
        Self { items, encode }
    }
}

impl<T> Canonical for KernelOrdered<'_, T> {
    fn encode(&self, out: &mut CanonicalWriter) {
        out.word(self.items.len() as u64);
        for item in &self.items {
            (self.encode)(item, out);
        }
    }
}

/// A 32-byte digest, length-prefixed.
struct Bytes32([u8; 32]);

impl Canonical for Bytes32 {
    fn encode(&self, out: &mut CanonicalWriter) {
        out.bytes(&self.0);
    }
}

fn search_limits(out: &mut CanonicalWriter, limits: SearchLimits) {
    out.word(limits.operations as u64).word(limits.depth as u64);
}

fn texts(out: &mut CanonicalWriter, values: &[Arc<str>]) {
    out.word(values.len() as u64);
    for value in values {
        out.text(value);
    }
}

fn variables(out: &mut CanonicalWriter, values: &[VariableId]) {
    out.word(values.len() as u64);
    for value in values {
        value.encode(out);
    }
}

/// A DAG's canonical identity wire (sorted edge lists), or a recorded refusal ([`unencodable`]),
/// then the label of every dense node in dense order (the wire indexes dense nodes; the labels
/// name variables).
fn dag_field(out: &mut CanonicalWriter, dag: &Dag) {
    match antecedent_io_core::identity::dag_identity(dag) {
        Ok(wire) => Wire(&wire).encode(out),
        Err(_) => unencodable(out, "graph"),
    }
    dag.nodes().encode(out);
}

/// The treatment, outcome, population and current joint law of a frozen failed contract.
fn failed_contract(out: &mut CanonicalWriter, contract: &dyn FailedContract) {
    contract.treatment().encode(out);
    contract.outcome().encode(out);
    out.text(contract.population());
    // `base_law` is the family's sorted set of jointly observed variables.
    variables(out, &contract.base_law());
}

/// Refuse a contract whose failure route the identity encoding below has no arm for.
/// `FailureRoute` is non-exhaustive; a route added later is refused until it gains an arm,
/// rather than taking a fallback identity it could share with another route.
fn encodable(contract: &RepairContract) -> Result<(), CausalError> {
    match contract {
        RepairContract::Route(
            FailureRoute::Backdoor(_)
            | FailureRoute::FrontDoor(_)
            | FailureRoute::Instrument(_)
            | FailureRoute::Hedge(_)
            | FailureRoute::Sensitivity(_),
        )
        | RepairContract::Transport(_)
        | RepairContract::ZTransport(_) => Ok(()),
        RepairContract::Route(_) => {
            Err(unencoded_variant("supplied.identification_repair.v1", "failure route"))
        }
    }
}

impl Canonical for RepairContract {
    /// The family's stable name and contract label, every unresolved obligation (in the family's
    /// order, each by its own canonical text, [`EvidenceObligation::canonical`]), then the frozen
    /// premises the family exposes: the graph by its io-core identity wire and the failed
    /// contract's variables, the selection diagram, query and base catalog of a transport
    /// contract, or the frozen failure of a z-transport contract (whose identity the contract
    /// label carries). A hedge's certificate is derived from its graph and is not written.
    fn encode(&self, out: &mut CanonicalWriter) {
        let family = self.family();
        out.text(family.family_id()).text(&family.contract_id());
        let obligations: Vec<String> =
            family.unresolved_obligations().iter().map(EvidenceObligation::canonical).collect();
        out.texts(&obligations);
        match self {
            Self::Route(route) => match route {
                FailureRoute::Backdoor(family) => {
                    variant(out, "backdoor");
                    dag_field(out, family.dag());
                    failed_contract(out, &**family);
                }
                FailureRoute::FrontDoor(family) => {
                    variant(out, "front_door");
                    dag_field(out, family.dag());
                    failed_contract(out, &**family);
                }
                FailureRoute::Instrument(family) => {
                    variant(out, "instrument");
                    dag_field(out, family.dag());
                    failed_contract(out, &**family);
                }
                FailureRoute::Hedge(family) => {
                    variant(out, "hedge");
                    family.admg().encode(out);
                    variables(out, family.surrogates());
                    failed_contract(out, &**family);
                }
                FailureRoute::Sensitivity(family) => {
                    variant(out, "sensitivity");
                    dag_field(out, family.dag());
                    failed_contract(out, &**family);
                }
                // A route added later never reaches here: `encodable` refuses it at compile, so
                // it cannot share an identity with another route.
                _ => {
                    variant(out, "unencoded_route");
                }
            },
            Self::Transport(family) => {
                variant(out, "transport");
                DiagramField(family.diagram()).encode(out);
                let query = family.query();
                out.text(&query.source).text(&query.target);
                variables(out, &query.treatments);
                variables(out, &query.outcomes);
                CatalogField(family.base_catalog()).encode(out);
                SidLimitsField(family.sid_limits()).encode(out);
            }
            Self::ZTransport(family) => {
                variant(out, "z_transport");
                SnapshotField(family.snapshot()).encode(out);
                SidLimitsField(family.sid_limits()).encode(out);
            }
        }
    }
}

/// A repair candidate by its semantic id (the kernel's own canonical, order-independent content
/// identity, [`DurableStudyCandidate::semantic_id`]) and its label.
fn durable_candidate(candidate: &DurableStudyCandidate, out: &mut CanonicalWriter) {
    out.text(&candidate.semantic_id()).text(&candidate.label);
}

struct RepairObjectiveField(RepairObjective);

impl Canonical for RepairObjectiveField {
    fn encode(&self, out: &mut CanonicalWriter) {
        match self.0 {
            RepairObjective::MinimizeCost => variant(out, "minimize_cost"),
            RepairObjective::MinimizeSampleBudget => variant(out, "minimize_sample_budget"),
        };
    }
}

struct RepairLimitsField(RepairLimits);

impl Canonical for RepairLimitsField {
    fn encode(&self, out: &mut CanonicalWriter) {
        search_limits(out, self.0.search);
        out.word(self.0.memory_limit_bytes);
    }
}

struct StudyPlanLimitsField(StudyPlanLimits);

impl Canonical for StudyPlanLimitsField {
    fn encode(&self, out: &mut CanonicalWriter) {
        search_limits(out, self.0.search);
        out.word(self.0.memory_limit_bytes);
    }
}

struct SidLimitsField(SidLimits);

impl Canonical for SidLimitsField {
    fn encode(&self, out: &mut CanonicalWriter) {
        out.word(self.0.steps as u64).word(self.0.depth as u64);
    }
}

/// A study-planning route in its canonical form ([`StudyPlanRoute::canonical`], sources in
/// canonical order), by its io-core query wire.
struct RouteField<'a>(&'a StudyPlanRoute);

impl Canonical for RouteField<'_> {
    fn encode(&self, out: &mut CanonicalWriter) {
        match self.0.canonical() {
            StudyPlanRoute::Mixed(query) => {
                variant(out, "mixed");
                Wire(&MixedSourceQueryWire::from_query(&query)).encode(out);
            }
            StudyPlanRoute::Mz(query) => {
                variant(out, "mz");
                Wire(&MzTransportQueryWire::from_query(&query)).encode(out);
            }
        }
    }
}

/// An evidence catalog by the io-core wire of its canonical form
/// ([`EvidenceCatalog::canonicalized`]), the encoding the z-transport planner digests; a catalog
/// that does not canonicalize is written by the wire of its declared form under its own tag.
struct CatalogField<'a>(&'a EvidenceCatalog);

impl Canonical for CatalogField<'_> {
    fn encode(&self, out: &mut CanonicalWriter) {
        if let Ok(canonical) = self.0.canonicalized() {
            variant(out, "canonical");
            Wire(&EvidenceCatalogWire::from_catalog(&canonical)).encode(out);
        } else {
            variant(out, "declared");
            Wire(&EvidenceCatalogWire::from_catalog(self.0)).encode(out);
        }
    }
}

/// A selection diagram by io-core's transport identity wire (canonical graph, sorted selection
/// targets).
struct DiagramField<'a>(&'a SelectionDiagram);

impl Canonical for DiagramField<'_> {
    fn encode(&self, out: &mut CanonicalWriter) {
        match antecedent_io_core::transport_identity(Some(self.0), None) {
            Ok(wire) => Wire(&wire).encode(out),
            Err(_) => unencodable(out, "selection diagram"),
        }
        // The wire indexes dense nodes; each dense node's label names its variable.
        self.0.causal_graph().nodes().encode(out);
    }
}

/// A frozen z-transport failure by its inspection wire, with the graph's edge lists and the
/// selection targets sorted as the planner's own failure digest sorts them.
struct SnapshotField<'a>(&'a ZTransportFailureSnapshot);

impl Canonical for SnapshotField<'_> {
    fn encode(&self, out: &mut CanonicalWriter) {
        match self.0.to_wire() {
            Ok(mut wire) => {
                wire.graph.directed.sort_unstable();
                wire.graph.bidirected.sort_unstable();
                wire.selection_targets.sort_unstable();
                Wire(&wire).encode(out);
            }
            Err(_) => unencodable(out, "z-transport failure snapshot"),
        }
    }
}

fn design_cost(out: &mut CanonicalWriter, cost: &DesignCost) {
    out.real(cost.amount).word(cost.sample_budget);
}

/// A z-transport plan's candidates in declared order (the planner evaluates the first
/// `max_evaluated` and refuses a repeated id against the earlier one), each with its catalog
/// delta by the io-core delta wire.
struct PlanSpecField<'a>(&'a ZTransportPlanSpec);

impl Canonical for PlanSpecField<'_> {
    fn encode(&self, out: &mut CanonicalWriter) {
        out.word(self.0.candidates.len() as u64);
        for candidate in &self.0.candidates {
            transport_candidate(out, candidate);
        }
        SidLimitsField(self.0.identification_limits).encode(out);
        out.word(self.0.max_evaluated as u64);
    }
}

fn transport_candidate(out: &mut CanonicalWriter, candidate: &TransportEvidenceCandidate) {
    out.text(&candidate.id);
    CandidateField(&candidate.design).encode(out);
    design_cost(out, &candidate.cost);
    Wire(&EvidenceCatalogDeltaWire::from_delta(&candidate.delta)).encode(out);
    out.text(&candidate.recruitment_sampling);
    texts(out, &candidate.feasibility_constraints);
}

/// A study-planning candidate's declared fields; its variable and text lists are written in
/// declared order.
fn study_candidate(candidate: &StudyCandidate, out: &mut CanonicalWriter) {
    out.text(&candidate.id).text(&candidate.population);
    variables(out, &candidate.interventions);
    variables(out, &candidate.measured);
    match &candidate.levels {
        Some(levels) => {
            out.word(1).word(levels.len() as u64);
            for level in levels.iter() {
                out.word(level.len() as u64);
                for assignment in level.iter() {
                    intervention_assignment(out, assignment);
                }
            }
        }
        None => {
            out.word(0);
        }
    }
    out.text(&candidate.recruitment);
    out.word(candidate.cost.units).word(candidate.cost.sample_budget);
    texts(out, &candidate.requires);
    texts(out, &candidate.conflicts);
    texts(out, &candidate.feasibility_constraints);
}

fn intervention_assignment(out: &mut CanonicalWriter, assignment: &InterventionAssignment) {
    assignment.variable.encode(out);
    match &assignment.value {
        Value::Float64(value) => {
            variant(out, "float64").real(*value);
        }
        Value::Int64(value) => {
            variant(out, "int64").word(u64::from_le_bytes(value.to_le_bytes()));
        }
        Value::Bool(value) => {
            variant(out, "bool").word(u64::from(*value));
        }
        Value::Category(value) => {
            variant(out, "category").word(u64::from(*value));
        }
        Value::Label(value) => {
            variant(out, "label").text(value);
        }
    }
}

struct CandidateField<'a>(&'a CandidateDesign);

impl Canonical for CandidateField<'_> {
    fn encode(&self, out: &mut CanonicalWriter) {
        match self.0 {
            CandidateDesign::IncreaseSamplingRate(plan) => {
                variant(out, "increase_sampling_rate").word(plan.tag).word(plan.additional_samples);
                design_cost(out, &plan.cost);
            }
            CandidateDesign::Intervene(plan) => {
                variant(out, "intervene").word(plan.tag);
                variables(out, &plan.targets);
                design_cost(out, &plan.cost);
            }
            CandidateDesign::Measure(plan) => {
                variant(out, "measure").word(plan.tag);
                variables(out, &plan.variables);
                design_cost(out, &plan.cost);
            }
            CandidateDesign::ObserveEnvironment(plan) => {
                variant(out, "observe_environment")
                    .word(plan.tag)
                    .word(u64::from(plan.environment.raw()))
                    .word(plan.additional_rows);
                design_cost(out, &plan.cost);
            }
        }
    }
}

struct RankConfigField<'a>(&'a DesignRankConfig);

impl Canonical for RankConfigField<'_> {
    fn encode(&self, out: &mut CanonicalWriter) {
        out.word(u64::from(self.0.min_batches))
            .word(u64::from(self.0.max_batches))
            .word(u64::from(self.0.batch_size))
            .real(self.0.rank_uncertainty_threshold);
    }
}

struct ConstraintsField<'a>(&'a DesignConstraints);

impl Canonical for ConstraintsField<'_> {
    fn encode(&self, out: &mut CanonicalWriter) {
        out.opt_real(self.0.max_cost).opt_word(self.0.max_sample_budget);
    }
}

struct ObjectiveField<'a>(&'a DesignObjective);

impl Canonical for ObjectiveField<'_> {
    fn encode(&self, out: &mut CanonicalWriter) {
        match self.0 {
            DesignObjective::IncreaseIdentificationProbability { query } => {
                variant(out, "increase_identification_probability").word(u64::from(query.raw()));
            }
            DesignObjective::ReduceGraphEntropy => {
                variant(out, "reduce_graph_entropy");
            }
            DesignObjective::ReduceEffectPosteriorWidth { query } => {
                variant(out, "reduce_effect_posterior_width").word(u64::from(query.raw()));
            }
            DesignObjective::DistinguishModels { models } => {
                variant(out, "distinguish_models").word(models.len() as u64);
                for model in models.iter() {
                    out.word(u64::from(model.raw()));
                }
            }
            DesignObjective::ReduceDecisionRegret { decision } => {
                variant(out, "reduce_decision_regret").word(u64::from(decision.raw()));
            }
        }
    }
}

/// A weighted graph posterior: every per-structure array in sample order.
struct GraphSamplesField<'a>(&'a WeightedGraphSamples);

impl Canonical for GraphSamplesField<'_> {
    fn encode(&self, out: &mut CanonicalWriter) {
        let graphs = self.0;
        out.word(graphs.n_samples as u64).reals(&graphs.weights).words(&graphs.graph_keys);
        FlagsField(&graphs.identified).encode(out);
        graphs.edge_marginals.as_deref().map(<[f64]>::to_vec).encode(out);
        graphs.orientation_marginals.as_deref().map(<[f64]>::to_vec).encode(out);
    }
}

struct FlagsField<'a>(&'a [GraphIdentFlag]);

impl Canonical for FlagsField<'_> {
    fn encode(&self, out: &mut CanonicalWriter) {
        out.word(self.0.len() as u64);
        for flag in self.0 {
            match flag {
                GraphIdentFlag::Identified => variant(out, "identified"),
                GraphIdentFlag::Unidentified => variant(out, "unidentified"),
            };
        }
    }
}

struct EffectWidthField<'a>(&'a antecedent_design::EffectWidthContext);

impl Canonical for EffectWidthField<'_> {
    fn encode(&self, out: &mut CanonicalWriter) {
        let context = self.0;
        out.word(context.n).real(context.sigma2).word(context.treatment_col as u64);
        out.reals(&context.xtx);
        match &context.intervention_design {
            Some(design) => {
                out.word(1).word(design.n).real(design.sigma2).reals(&design.xtx);
            }
            None => {
                out.word(0);
            }
        }
        match &context.environment_grams {
            Some(grams) => {
                out.word(1).word(grams.len() as u64);
                for gram in grams.iter() {
                    out.word(u64::from(gram.environment.raw()))
                        .word(gram.n)
                        .opt_real(gram.sigma2)
                        .reals(&gram.xtx);
                }
            }
            None => {
                out.word(0);
            }
        }
        match &context.measure_columns {
            Some(columns) => {
                out.word(1).word(columns.len() as u64);
                for column in columns.iter() {
                    column.variable.encode(out);
                    out.real(column.self_dot).reals(&column.cross).opt_real(column.sigma2_after);
                }
            }
            None => {
                out.word(0);
            }
        }
    }
}

struct ModelLoglikField<'a>(&'a antecedent_design::ModelLoglikDraws);

impl Canonical for ModelLoglikField<'_> {
    fn encode(&self, out: &mut CanonicalWriter) {
        out.word(self.0.models.len() as u64);
        for model in self.0.models.iter() {
            out.word(u64::from(model.raw()));
        }
        out.word(self.0.n_draws as u64).reals(&self.0.loglik);
    }
}

type QueryUnlock = (antecedent_core::QueryId, Arc<[VariableId]>);

/// A query's unlock lists in declared order (the ranker reads the first entry for a query),
/// each list as a set: the ranker only asks whether a candidate's variable or environment is in
/// it, so its order and a repetition never reach the identity.
struct QueryUnlockField<'a>(&'a [QueryUnlock]);

/// The raw ids of an unlock list, sorted and deduplicated.
fn raw_set(raw: impl Iterator<Item = u32>) -> Vec<u64> {
    let mut words: Vec<u64> = raw.map(u64::from).collect();
    words.sort_unstable();
    words.dedup();
    words
}

impl Canonical for QueryUnlockField<'_> {
    fn encode(&self, out: &mut CanonicalWriter) {
        out.word(self.0.len() as u64);
        for (query, unlocked) in self.0 {
            out.word(u64::from(query.raw()));
            out.words(&raw_set(unlocked.iter().map(|variable| variable.raw())));
        }
    }
}

type EnvironmentUnlock = (antecedent_core::QueryId, Arc<[antecedent_core::EnvironmentId]>);

struct EnvironmentUnlockField<'a>(&'a [EnvironmentUnlock]);

impl Canonical for EnvironmentUnlockField<'_> {
    fn encode(&self, out: &mut CanonicalWriter) {
        out.word(self.0.len() as u64);
        for (query, environments) in self.0 {
            out.word(u64::from(query.raw()));
            out.words(&raw_set(environments.iter().map(|environment| environment.raw())));
        }
    }
}

/// The declared identity of a decision registry's callbacks: each registered problem's raw id,
/// action and constraint counts and chance threshold, a normal prior's moments or the number of
/// state draws, and the signal's name. The utilities, constraints, signal model and generic
/// state values themselves are runtime bindings.
fn registry_identity<A, O>(registry: &antecedent_design::DecisionRegistry<A, O>) -> String {
    let mut out = CanonicalWriter::new("antecedent.design_ranking.decision_registry.v1");
    let registered: Vec<(usize, &antecedent_design::DecisionProblem<A, O>)> = registry
        .problems
        .iter()
        .enumerate()
        .filter_map(|(index, problem)| problem.as_ref().map(|problem| (index, problem)))
        .collect();
    out.word(registered.len() as u64);
    for (index, problem) in registered {
        out.word(index as u64)
            .word(problem.actions.len() as u64)
            .word(problem.constraints.len() as u64)
            .real(problem.chance_threshold);
    }
    match &registry.prior {
        antecedent_design::DecisionPrior::Draws(draws) => {
            variant(&mut out, "draws").word(draws.len() as u64);
        }
        antecedent_design::DecisionPrior::Normal { mean, variance } => {
            variant(&mut out, "normal").real(*mean).real(*variance);
        }
    }
    out.text(registry.signal.name());
    out.finish_hex()
}

impl Canonical for ChoicePosterior {
    fn encode(&self, out: &mut CanonicalWriter) {
        match self {
            Self::Pooled { alpha, seed } => {
                variant(out, "pooled").reals(alpha).word(*seed);
            }
            Self::Hierarchical { prior, respondents } => {
                variant(out, "hierarchical")
                    .reals(&prior.shapes)
                    .real(prior.log_concentration_mean)
                    .real(prior.log_concentration_sd);
                // Respondents in declared order, each with its counts per option.
                out.word(respondents.len() as u64);
                for counts in respondents {
                    out.u32s(counts);
                }
            }
        }
    }
}

struct ChoiceCandidateField<'a>(&'a ChoiceStudyCandidate);

impl Canonical for ChoiceCandidateField<'_> {
    fn encode(&self, out: &mut CanonicalWriter) {
        out.text(&self.0.label).word(u64::from(self.0.observations)).real(self.0.cost);
    }
}

struct SampleField<'a>(&'a GraphPosteriorSample);

impl Canonical for SampleField<'_> {
    fn encode(&self, out: &mut CanonicalWriter) {
        out.real(self.0.weight).word(self.0.key);
        self.0.graph.encode(out);
    }
}

#[cfg(test)]
#[path = "design_planning_programs_site_tests.rs"]
mod site_tests;
