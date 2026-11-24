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
//! own refusals (re-entry, cancellation, a release build's support gate, an evaluation context
//! other than the compiled one) are [`TerminalError::Refused`].
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{CanonicalWriter, EvidenceCatalog, ExecutionContext, VariableId};
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
use antecedent_graph::{Admg, SelectionDiagram};
use antecedent_identify::{SidLimits, StudyPlanLimits, StudyPlanRoute, ZTransportQuery};
use antecedent_prob::WeightedGraphSamples;
use antecedent_stats::adaptive_inference::HierarchicalChoicePrior;

pub use crate::analysis::supplied_terminal::TerminalError;
use crate::analysis::supplied_terminal::{
    Reading, TerminalSpec, admit_compile, compiled_terminal, declaration_digest, lower,
};
use crate::error::CausalError;
use crate::planner::lowered::{Composition, EvidenceRole, FunctionalKind, ProgramCell};

// --- identification repair -------------------------------------------------------------------

/// A failed contract whose evidence obligations a repair search tries to close.
#[derive(Clone, Debug)]
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
/// `public_builder_reentry` inside a lowered dispatch, or cancellation.
pub fn compile_identification_repair(
    contract: impl Into<RepairContract>,
    candidates: Vec<DurableStudyCandidate>,
    objective: RepairObjective,
    limits: RepairLimits,
    ctx: &ExecutionContext,
) -> Result<CompiledIdentificationRepair, CausalError> {
    admit_compile(ctx)?;
    let contract = contract.into();
    let family = contract.family();
    let question = declaration_digest(
        "antecedent.identification_repair.v1",
        &[
            &family.family_id(),
            &family.contract_id(),
            &family.unresolved_obligations(),
            &candidates,
            &objective,
            &limits,
        ],
    );
    let lowered = lower(TerminalSpec {
        cell: ProgramCell::IdentificationRepair,
        plan_id: "identification_repair.v1",
        functional: FunctionalKind::StructureSearch,
        graph_class: "failed_identification_contract",
        question,
        evaluator: "repair.theorem_checker_search",
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
        let scoped = self.enter(ctx)?;
        antecedent_design::repair(
            self.contract.family(),
            &self.candidates,
            self.objective,
            self.limits,
            &scoped,
        )
        .map_err(TerminalError::Kernel)
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
/// `public_builder_reentry` inside a lowered dispatch, or cancellation.
pub fn compile_study_addition_plan(
    graph: Admg,
    route: StudyPlanRoute,
    catalog: EvidenceCatalog,
    candidates: Vec<StudyCandidate>,
    limits: StudyPlanLimits,
    ctx: &ExecutionContext,
) -> Result<CompiledStudyAdditionPlan, CausalError> {
    admit_compile(ctx)?;
    let question = declaration_digest(
        "antecedent.study_addition_plan.v1",
        &[&graph, &route, &catalog, &candidates, &limits],
    );
    let lowered = lower(TerminalSpec {
        cell: ProgramCell::StudyAdditionPlanning,
        plan_id: "study_addition_planning.v1",
        functional: FunctionalKind::StructureSearch,
        graph_class: "admg",
        question,
        evaluator: "study_planner.plan_studies",
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
        let scoped = self.enter(ctx)?;
        plan_studies(
            &self.graph,
            &self.route,
            &self.catalog,
            &self.candidates,
            self.limits,
            &scoped,
        )
        .map_err(TerminalError::Kernel)
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
/// `public_builder_reentry` inside a lowered dispatch, or cancellation.
pub fn compile_z_transport_failure_snapshot(
    diagram: SelectionDiagram,
    query: ZTransportQuery,
    catalog: EvidenceCatalog,
    limits: SidLimits,
    ctx: &ExecutionContext,
) -> Result<CompiledZTransportFailureSnapshot, CausalError> {
    admit_compile(ctx)?;
    let question = declaration_digest(
        "antecedent.z_transport_failure_snapshot.v1",
        &[&diagram, &query, &catalog, &limits],
    );
    let lowered = lower(TerminalSpec {
        cell: ProgramCell::ZTransportEvidencePlanning,
        plan_id: "z_transport_failure_snapshot.v1",
        functional: FunctionalKind::StructureSearch,
        graph_class: "selection_diagram",
        question,
        evaluator: "z_transport_planner.snapshot_failure",
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
        let scoped = self.enter(ctx)?;
        snapshot_z_transport_failure(
            &self.diagram,
            &self.query,
            &self.catalog,
            self.limits,
            &scoped,
        )
        .map_err(TerminalError::Kernel)
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
/// `public_builder_reentry` inside a lowered dispatch, or cancellation.
pub fn compile_z_transport_evidence_plan(
    snapshot: Arc<ZTransportFailureSnapshot>,
    spec: ZTransportPlanSpec,
    ctx: &ExecutionContext,
) -> Result<CompiledZTransportEvidencePlan, CausalError> {
    admit_compile(ctx)?;
    let question =
        declaration_digest("antecedent.z_transport_evidence_plan.v1", &[&snapshot, &spec]);
    let lowered = lower(TerminalSpec {
        cell: ProgramCell::ZTransportEvidencePlanning,
        plan_id: "z_transport_evidence_plan.v1",
        functional: FunctionalKind::StructureSearch,
        graph_class: "selection_diagram",
        question,
        evaluator: "z_transport_planner.plan_evidence",
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
        let scoped = self.enter(ctx)?;
        plan_z_transport_evidence(&self.snapshot, &self.spec, &scoped)
            .map_err(TerminalError::Kernel)
    }
}

// --- design ranking --------------------------------------------------------------------------

/// The digest of a design evaluation context: the graph posterior, every unlock list and flag
/// vector, the effect-width and model-likelihood inputs, and whether a decision registry is
/// supplied. A registry holds utility callbacks, which have no canonical rendering; the
/// ranking reads the registry the caller passes at execution.
fn evaluation_digest<A, O>(eval: &DesignEvaluationContext<'_, A, O>) -> [u8; 32] {
    let mut writer = CanonicalWriter::new("antecedent.design_ranking.evaluation.v1");
    writer
        .text(&format!("{:?}", eval.graphs))
        .text(&format!("{:?}", eval.effect_width))
        .text(&format!("{:?}", eval.model_loglik))
        .text(&format!("{:?}", eval.query_id_unlock))
        .text(&format!("{:?}", eval.env_id_unlock))
        .text(&format!("{:?}", eval.identified_under_intervention))
        .text(&format!("{:?}", eval.graph_features))
        .word(u64::from(eval.decisions.is_some()));
    if eval.decisions.is_some() {
        writer.note_runtime_callback();
    }
    writer.finish()
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
/// `public_builder_reentry` inside a lowered dispatch, or cancellation.
pub fn compile_design_ranking<A, O>(
    ranker: &DesignRanker,
    objective: DesignObjective,
    candidates: Vec<CandidateDesign>,
    eval: &DesignEvaluationContext<'_, A, O>,
    ctx: &ExecutionContext,
) -> Result<CompiledDesignRanking, CausalError> {
    admit_compile(ctx)?;
    let evaluation = evaluation_digest(eval);
    let question = declaration_digest(
        "antecedent.design_ranking.v1",
        &[&ranker.config, &ranker.constraints, &objective, &candidates, &evaluation],
    );
    let lowered = lower(TerminalSpec {
        cell: ProgramCell::DesignRanking,
        plan_id: "design_ranking.v1",
        functional: FunctionalKind::StructureSearch,
        graph_class: "weighted_graph_posterior",
        question,
        evaluator: "ranker.rank",
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
        let scoped = self.enter(ctx)?;
        if evaluation_digest(eval) != self.evaluation {
            return Err(TerminalError::Refused(CausalError::Compile {
                message: "design_ranking: the evaluation context differs from the one the \
                          ranking was compiled for; compile the ranking for this context"
                    .to_owned(),
            }));
        }
        DesignRanker::new()
            .with_config(self.config.clone())
            .with_constraints(self.constraints.clone())
            .rank(&self.objective, &self.candidates, eval, &scoped)
            .map_err(TerminalError::Kernel)
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
/// `public_builder_reentry` inside a lowered dispatch, or cancellation.
pub fn compile_choice_study_ranking(
    posterior: ChoicePosterior,
    values: Vec<f64>,
    candidates: Vec<ChoiceStudyCandidate>,
    ctx: &ExecutionContext,
) -> Result<CompiledChoiceStudyRanking, CausalError> {
    admit_compile(ctx)?;
    let question = declaration_digest(
        "antecedent.choice_study_ranking.v1",
        &[&posterior, &values, &candidates],
    );
    let lowered = lower(TerminalSpec {
        cell: ProgramCell::ChoiceStudyRanking,
        plan_id: "choice_study_ranking.v1",
        functional: FunctionalKind::PolicyValue,
        graph_class: "categorical_choice",
        question,
        evaluator: match &posterior {
            ChoicePosterior::Pooled { .. } => "choice_evsi.rank_studies",
            ChoicePosterior::Hierarchical { .. } => "choice_evsi.rank_hierarchical_studies",
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
        let scoped = self.enter(ctx)?;
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
        .map_err(TerminalError::Kernel)
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
        evaluator: "structure_identification.id_algorithm",
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
/// `public_builder_reentry` inside a lowered dispatch, or cancellation.
pub fn compile_graph_identification(
    graph: Admg,
    treatment: VariableId,
    outcome: VariableId,
    ctx: &ExecutionContext,
) -> Result<CompiledGraphIdentification, CausalError> {
    admit_compile(ctx)?;
    let question =
        declaration_digest("antecedent.graph_identification.v1", &[&graph, &treatment, &outcome]);
    let lowered = lower(structure_spec(question, "graph_identification.v1", 1))?;
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
        let _scoped = self.enter(ctx)?;
        identifies(&self.graph, self.treatment, self.outcome).map_err(TerminalError::Kernel)
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
/// `public_builder_reentry` inside a lowered dispatch, or cancellation.
pub fn compile_structure_identification(
    samples: Vec<GraphPosteriorSample>,
    treatment: VariableId,
    outcome: VariableId,
    interventions: Option<Vec<VariableId>>,
    ctx: &ExecutionContext,
) -> Result<CompiledStructureIdentification, CausalError> {
    admit_compile(ctx)?;
    let question = declaration_digest(
        "antecedent.structure_identification.v1",
        &[&samples, &treatment, &outcome, &interventions],
    );
    let lowered = lower(structure_spec(question, "structure_identification.v1", samples.len()))?;
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
        let _scoped = self.enter(ctx)?;
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
        Ok(StructureIdentificationReadout { posterior, under_intervention })
    }
}
