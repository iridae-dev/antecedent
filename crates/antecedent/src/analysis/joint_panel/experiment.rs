//! Conditional experiment updates over one compiled panel posterior and existing decision ports.
//!
//! Every update, decision and next-study ranking executes the `JointPanelExperiment` program
//! of its operation. A public entry point lowers that program as a child of the program whose
//! law it reads, enters the program's dispatch, passes the release support gate and runs the
//! unchanged update with the dispatch-scoped context; the result keeps the caller's context
//! and carries the program it executed.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::compiled::{self, Entry};
use super::{
    Arc, ExecutionContext, JointPanelError, JointPanelRequest, JointPanelResponseFunctional,
    JointPanelResult, JointPanelState, PreparedStudy, SemanticDigest, engine, source,
};
use crate::analysis::lowering::joint_panel::experiment::{ExperimentOperation, lower};
use crate::planner::lowered::{
    CarrierDemand, DecisionProgram, EvsiProgram, JointPanelRoute, LoweredProgram,
};
use crate::planner::{PlanStage, PlanView};
use antecedent_contract::bridge::{BridgeContract, SourceMappingDependence};
use antecedent_contract::evidence_ledger::{EvidenceLedger, Overlap, SourceId, SourceStatus};
use antecedent_core::CanonicalWriter;
use antecedent_design::DesignError;
use antecedent_design::decision::DecisionProblem;
use antecedent_design::decision_contract::DecisionContract;
use antecedent_design::preposterior::{DecisionSignal, PreposteriorAnalysis};
use antecedent_io::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionTrust,
};

mod future_regime;
mod joint;
mod law;
mod matrix;
mod native;
pub use future_regime::{
    PanelFutureRegime, PanelFutureStudy, PanelFutureStudyUpdate, PanelFutureStudyValue,
};
pub use joint::{
    HistoricalJointExperiment, PanelJointCombination, PanelJointExperimentComponent,
    PanelJointExperimentResult, PanelJointExperimentUpdate, PanelJointSensitivity,
};
pub use law::{
    HistoricalJointLawExperiment, JointLawExperimentRequest, JointLawExperimentResult,
    LawExperimentLimits,
};
pub use matrix::{
    JointExperimentAcceptanceMatrix, JointExperimentAcceptanceRow,
    joint_experiment_acceptance_matrix,
};
pub use native::{
    HistoricalNativeJointExperiment, LocalRatioGaussianRegularization, NativeExperimentBinding,
    NativeExperimentDecision, NativeJointExperimentResult,
};

/// Current ledger origins of every likelihood/prior/mapping used by an update.
#[derive(Clone, Debug)]
pub struct PanelExperimentEvidence {
    /// Model source declaration; its label must equal the exact fit dependency digest.
    pub model: SourceId,
    /// All prior origins, including data-derived priors; absence explicitly means elicited.
    pub prior_origins: Vec<SourceId>,
    /// New independent experimental likelihood evidence.
    pub experiment: SourceId,
    /// Mapping evidence; its dependence on the model remains explicitly declared.
    pub mapping: SourceId,
}

/// One paired affine map per original shared parameter slot.
#[derive(Clone, Debug)]
pub struct PanelExperimentMapping {
    /// Exact fitted dependency whose draw slots these mappings accompany.
    pub dependency: SemanticDigest,
    /// Mapping draw intercepts, in source parameter-slot order.
    pub intercepts: Vec<f64>,
    /// Mapping draw slopes, in the same order; none may be zero.
    pub slopes: Vec<f64>,
}

/// Explicit Gaussian experimental sufficient-statistic likelihood and full estimand translation.
#[derive(Clone, Debug)]
pub struct PanelExperimentUpdate {
    /// Existing bridge owner checks population, regime, assumptions and transport first.
    pub bridge: BridgeContract,
    /// Time functional applied identically to each complete world response.
    pub functional: JointPanelResponseFunctional,
    /// Observed experimental sample mean, in the bridge source quantity's units.
    pub observed_mean: f64,
    /// Known variance of one experimental observation, not a fitted standard error.
    pub observation_variance: f64,
    /// Actual independent experimental sample size.
    pub sample_size: u64,
    /// Explicit modular policy: experiment likelihood reweights shared panel/mapping
    /// draws; no feedback refit of the original measurement/assignment mechanisms.
    pub modular_policy: PanelExperimentModularPolicy,
    /// Exact source/mapping pairing; uncertainty is never replaced by its mean mapping.
    pub mapping: PanelExperimentMapping,
    /// Declared origin/overlap authority, checked again on every current consumption.
    pub evidence: PanelExperimentEvidence,
}

/// The declared scientific reconciliation used by this conditional update.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PanelExperimentModularPolicy {
    /// Importance conditioning of the retained joint law, with original fitting modules frozen.
    FrozenJointImportance,
}

impl PanelExperimentModularPolicy {
    /// Stable `snake_case` name.
    const fn as_str(self) -> &'static str {
        match self {
            Self::FrozenJointImportance => "frozen_joint_importance",
        }
    }
}

impl PanelExperimentMapping {
    /// Write the fitted dependency and every paired realization in slot order.
    fn write(&self, writer: &mut CanonicalWriter) {
        writer.bytes(self.dependency.as_bytes()).reals(&self.intercepts).reals(&self.slopes);
    }

    /// Bytes of the paired realizations an update reads.
    fn quoted_bytes(&self) -> u64 {
        let slots = self.intercepts.len().saturating_add(self.slopes.len());
        u64::try_from(slots).unwrap_or(u64::MAX).saturating_mul(F64_BYTES)
    }
}

impl PanelExperimentEvidence {
    /// Identity of the declared origins: model, experiment, mapping and every prior origin,
    /// in declaration order.
    pub(crate) fn origins_identity(&self) -> SemanticDigest {
        let mut writer = CanonicalWriter::new("antecedent.joint_panel.experiment_origins.v1");
        writer
            .bytes(self.model.digest().as_bytes())
            .bytes(self.experiment.digest().as_bytes())
            .bytes(self.mapping.digest().as_bytes())
            .word(self.prior_origins.len() as u64);
        for origin in &self.prior_origins {
            writer.bytes(origin.digest().as_bytes());
        }
        SemanticDigest::from_bytes(writer.finish())
    }
}

impl PanelExperimentUpdate {
    /// Identity of the independent experiment evidence this update reads: its bridge,
    /// response functional, Gaussian sample-mean likelihood, modular policy, paired mapping
    /// and declared origins. It is the experiment an update program cites as trial evidence.
    pub(crate) fn evidence_identity(&self) -> SemanticDigest {
        let mut writer = CanonicalWriter::new("antecedent.joint_panel.experiment_evidence.v1");
        writer.bytes(self.bridge.identity().as_bytes());
        write_functional(&mut writer, &self.functional);
        writer
            .real(self.observed_mean)
            .real(self.observation_variance)
            .word(self.sample_size)
            .text(self.modular_policy.as_str());
        self.mapping.write(&mut writer);
        writer.bytes(self.evidence.origins_identity().as_bytes());
        SemanticDigest::from_bytes(writer.finish())
    }

    /// Bytes of the experiment evidence an update reads: its sufficient statistics and the
    /// paired mapping realizations. A quote for the physical layer; nothing is admitted.
    pub(crate) fn evidence_bytes(&self) -> u64 {
        self.mapping.quoted_bytes().saturating_add(3 * F64_BYTES)
    }
}

/// Bytes of one `f64`, the unit of every experiment quote.
const F64_BYTES: u64 = std::mem::size_of::<f64>() as u64;

/// Write a response functional in canonical form.
fn write_functional(writer: &mut CanonicalWriter, functional: &JointPanelResponseFunctional) {
    match functional {
        JointPanelResponseFunctional::TerminalMean => {
            writer.text("terminal_mean");
        }
        JointPanelResponseFunctional::WindowAverage { from, until } => {
            writer.text("window_average").word(*from as u64).word(*until as u64);
        }
        JointPanelResponseFunctional::WindowChange { before, after } => {
            writer
                .text("window_change")
                .word(before.0 as u64)
                .word(before.1 as u64)
                .word(after.0 as u64)
                .word(after.1 as u64);
        }
        JointPanelResponseFunctional::WeightedSum { coefficients } => {
            writer.text("weighted_sum").reals(coefficients);
        }
    }
}

/// Bytes of the aligned draws of a conditioned law, the read of a decision or ranking.
fn law_bytes(law: &DistributionArtifact) -> u64 {
    u64::try_from(law.draws().len()).unwrap_or(u64::MAX).saturating_mul(F64_BYTES)
}

/// Lower the experiment program of `operation` as a child of the program `parent`.
fn experiment_program(
    operation: ExperimentOperation,
    parent: SemanticDigest,
    experiment: SemanticDigest,
    experiment_bytes: u64,
    decision: Option<DecisionProgram>,
) -> Result<Arc<LoweredProgram>, JointPanelError> {
    Ok(Arc::new(lower(operation, parent, experiment, experiment_bytes, decision)?))
}

/// Enter the dispatch of an experiment `program` from a public entry point and pass the
/// release support gate with the scoped context. The update runs with the returned context.
///
/// # Errors
/// `public_builder_reentry` when `ctx` is already inside a lowered dispatch, and the
/// `not_supported` refusal of a release build whose experiment capability is not supported.
fn enter_experiment(
    ctx: &ExecutionContext,
    program: &LoweredProgram,
) -> Result<ExecutionContext, JointPanelError> {
    let scoped = compiled::enter(ctx, program)?;
    compiled::require_release_support(&scoped, JointPanelRoute::Experiment)?;
    Ok(scoped)
}

/// The decision program of a decision contract evaluated on a conditioned posterior law.
///
/// A contract that does not validate carries no contract identity; the decision evaluation
/// then refuses it exactly as it does outside the compiler route.
fn contract_decision(contract: &DecisionContract) -> DecisionProgram {
    DecisionProgram {
        carrier: CarrierDemand::ParameterPosterior,
        utility: DecisionProgram::utility_form(None),
        actions: contract.actions.len(),
        constraints: contract.constraints.len(),
        chance_threshold: contract
            .constraints
            .iter()
            .map(|constraint| constraint.min_probability)
            .fold(0.0, f64::max),
        utility_unit: Some(Arc::from(contract.utility_units.as_str())),
        contract_identity: contract.identity().ok().map(Arc::from),
        information_value: None,
        scenario_source: None,
    }
}

/// The conditioned law a decision or next-study ranking reads, as its program cites it.
struct ConditionedLaw {
    /// Identity of the program that conditioned the law.
    program: SemanticDigest,
    /// Identity of the conditioned law as the decision reads it.
    law: SemanticDigest,
    /// Bytes of the conditioned law's aligned draws.
    bytes: u64,
    /// The experiment whose evidence the conditioned law already summarizes.
    experiment: SourceId,
}

impl ConditionedLaw {
    /// Lower the decision program of `contract` over this law and pass its dispatch entry and
    /// release support gate with `ctx`, the context the law was conditioned under: the
    /// decision evaluation takes no context of its own.
    ///
    /// # Errors
    /// As [`enter_experiment`].
    fn enter_decision(
        &self,
        ctx: &ExecutionContext,
        contract: &DecisionContract,
    ) -> Result<(), JointPanelError> {
        let program = lower(
            ExperimentOperation::Decide,
            self.program,
            self.law,
            self.bytes,
            Some(contract_decision(contract)),
        )?;
        enter_experiment(ctx, &program)?;
        Ok(())
    }

    /// Lower the next-study ranking program over this law and enter its dispatch from a
    /// public entry point. Returns the program identity every ranked value carries and the
    /// context the ranking runs with.
    ///
    /// The program's information value lists the candidates in canonical order, the
    /// experiment the law already summarizes, the caller's seed and one batch of
    /// `replicates` draws per candidate without early stopping; a replicate count beyond
    /// `u32::MAX` is quoted at `u32::MAX`. Its experiment identity binds the law, every
    /// candidate's sampling model and cost, and `regimes`, the identities of the declared
    /// future-study regimes.
    ///
    /// # Errors
    /// A Monte Carlo store whose quote overflows `u64`, and the refusals of
    /// [`enter_experiment`].
    fn enter_ranking<A>(
        &self,
        ctx: &ExecutionContext,
        problem: &DecisionProblem<A, PanelExperimentDecisionState>,
        candidates: &[PanelFutureSample],
        regimes: &[SemanticDigest],
        replicates: usize,
    ) -> Result<(SemanticDigest, ExecutionContext), JointPanelError> {
        let mut catalog = CanonicalWriter::new("antecedent.joint_panel.next_study_catalog.v1");
        catalog.bytes(self.law.as_bytes()).word(candidates.len() as u64);
        for candidate in candidates {
            catalog
                .text(&candidate.id)
                .word(candidate.sample_size)
                .real(candidate.observation_variance)
                .opt_real(candidate.cost_utility);
        }
        catalog.word(regimes.len() as u64);
        for regime in regimes {
            catalog.bytes(regime.as_bytes());
        }
        let decision = DecisionProgram {
            carrier: CarrierDemand::ParameterPosterior,
            utility: DecisionProgram::utility_form(None),
            actions: problem.actions.len(),
            constraints: problem.constraints.len(),
            chance_threshold: problem.chance_threshold,
            utility_unit: None,
            contract_identity: None,
            information_value: Some(self.information_value(candidates, replicates, ctx)?),
            scenario_source: None,
        };
        let program = lower(
            ExperimentOperation::RankNext,
            self.program,
            SemanticDigest::from_bytes(catalog.finish()),
            self.bytes,
            Some(decision),
        )?;
        let scoped = enter_experiment(ctx, &program)?;
        Ok((program.identity(), scoped))
    }

    /// The value-of-information layer of a next-study ranking; see [`Self::enter_ranking`].
    fn information_value(
        &self,
        candidates: &[PanelFutureSample],
        replicates: usize,
        ctx: &ExecutionContext,
    ) -> Result<EvsiProgram, JointPanelError> {
        let ids = candidates
            .iter()
            .map(|candidate| candidate.id.as_str())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .map(Arc::<str>::from)
            .collect::<Arc<[Arc<str>]>>();
        let rank = EvsiProgram::schedule(1, 1, u32::try_from(replicates).unwrap_or(u32::MAX), 0.0);
        let monte_carlo =
            EvsiProgram::quote_tasks(candidates.len() as u64, rank.draws_per_candidate())
                .ok_or_else(|| JointPanelError::compile("next-study Monte Carlo quote overflow"))?;
        let costed = !candidates.is_empty()
            && candidates.iter().all(|candidate| candidate.cost_utility.is_some());
        Ok(EvsiProgram {
            candidates: ids,
            prior_observation_ids: Arc::from([Arc::<str>::from(self.experiment.to_hex())]),
            rng_seed: ctx.rng.master_seed(),
            mc_error_tolerance: 0.0,
            require_net_value: false,
            tie_tolerance: 0.0,
            max_candidates: candidates.len() as u64,
            rank,
            // Declared costs are already in the decision's utility unit.
            cost_map: costed
                .then(|| EvsiProgram::cost_mapping("decision_utility", "decision_utility", 1.0)),
            monte_carlo,
        })
    }
}

/// Common prepared state carrying the exact original panel program and declared update.
#[derive(Clone, Debug)]
pub struct PanelExperimentState {
    panel: PreparedStudy<JointPanelState>,
    update: PanelExperimentUpdate,
    lowered: Arc<LoweredProgram>,
}

/// Conditional Bayesian update retaining the whole aligned joint response, not marginal intervals.
#[derive(Clone, Debug)]
pub struct PanelExperimentResult {
    original: JointPanelResult,
    update: Arc<PanelExperimentUpdate>,
    law: DistributionArtifact,
    prior_means: Vec<f64>,
    posterior_means: Vec<f64>,
    dependency: SemanticDigest,
    retention: Arc<antecedent_core::execution::AdmissionLease>,
    program: ExecutedProgram,
}

/// The experiment program an update executed, and the context of the public entry point
/// that ran it, which later decisions on the update run under.
#[derive(Clone, Debug)]
struct ExecutedProgram {
    lowered: Arc<LoweredProgram>,
    caller: ExecutionContext,
}

impl ExecutedProgram {
    /// The executed `lowered` program, kept with `caller`, the public entry point's context.
    fn new(lowered: Arc<LoweredProgram>, caller: &ExecutionContext) -> Self {
        Self { lowered, caller: caller.clone() }
    }

    /// Inspect the executed program.
    fn plan_view(&self) -> PlanView {
        PlanView::of_program(Arc::clone(&self.lowered), PlanStage::Compiled)
    }

    /// Identity of the executed program.
    fn identity(&self) -> SemanticDigest {
        self.lowered.identity()
    }
}

/// Retained experimental law whose scientific consumption requires caller-current evidence.
/// Numerical extraction is historical component data and carries no current authorization.
#[derive(Clone, Debug)]
pub struct PanelExperimentDecisionSource {
    source: antecedent_design::scientific_source::ScientificDecisionSource,
    update: Arc<PanelExperimentUpdate>,
    original_dependency: SemanticDigest,
    retention: Arc<antecedent_core::execution::AdmissionLease>,
    /// Other components whose evidence shaped the same joint weights; empty for a single update.
    companions: Vec<Arc<PanelExperimentUpdate>>,
}
impl PanelExperimentDecisionSource {
    /// Exact prefix-world basis underlying this response functional.
    #[must_use]
    pub fn target(&self) -> &antecedent_core::ScientificTarget {
        self.source.target()
    }
    /// Conditional proof retained through the experimental update.
    pub fn identification(&self) -> &antecedent_identify::CompiledIdentificationProduct {
        self.source.identification()
    }
    /// Actual sparse temporal projection.
    #[must_use]
    pub fn projection(
        &self,
    ) -> Option<&antecedent_design::scientific_source::ScientificSourceProjection> {
        self.source.projection()
    }
    /// Complete projected question identity.
    #[must_use]
    pub fn question_identity(&self) -> SemanticDigest {
        self.source.question_identity()
    }
    /// Exact experimental update dependency.
    #[must_use]
    pub fn dependency_identity(&self) -> SemanticDigest {
        self.source.dependency_identity()
    }
    /// Explicit historical numerical component; this grants no current scientific permission.
    #[must_use]
    pub fn as_numerical_source(
        &self,
    ) -> &antecedent_design::scientific_source::ScientificSourcePayload {
        self.source.as_numerical_source()
    }
    /// Check the caller's current evidence before scientific consumption.
    /// # Errors
    /// Withdrawal, supersession, mismatched origin or unsupported causal scope.
    pub fn authorize(
        &self,
        scope: antecedent_design::scientific_source::ScientificConsumptionScope,
        ledger: &EvidenceLedger,
    ) -> Result<(), JointPanelError> {
        self.retention.observe_current();
        validate_update(&self.update, ledger)?;
        validate_model_dependency(self.original_dependency, &self.update, ledger)?;
        for companion in &self.companions {
            validate_update(companion, ledger)?;
            validate_model_dependency(self.original_dependency, companion, ledger)?;
        }
        if scope
            == antecedent_design::scientific_source::ScientificConsumptionScope::IdentifiedCausal
        {
            return Err(JointPanelError::Unsupported("the frozen modular experiment update requires its additional likelihood and transport model; causal authority for that combined model is not inferred from the original identification proof".into()));
        }
        self.source.authorize(scope).map_err(super::search::map_scientific_error)
    }
    /// Evaluate a decision only under current caller-supplied ledger authority.
    /// # Errors
    /// Current scientific authorization or original decision semantics refuse consumption.
    pub fn evaluate(
        &self,
        contract: &antecedent_design::decision_contract::DecisionContract,
        scope: antecedent_design::scientific_source::ScientificConsumptionScope,
        ledger: &EvidenceLedger,
    ) -> Result<antecedent_design::scientific_source::ScopedDecisionResult, JointPanelError> {
        self.authorize(scope, ledger)?;
        self.source.evaluate(contract, scope).map_err(super::search::map_scientific_error)
    }
    /// [`Self::evaluate`] through the demand executor, charged to `ctx`'s admission pool.
    /// # Errors
    /// As [`Self::evaluate`], and the memory budget of `ctx`.
    pub fn evaluate_admitted(
        &self,
        contract: &antecedent_design::decision_contract::DecisionContract,
        scope: antecedent_design::scientific_source::ScientificConsumptionScope,
        ledger: &EvidenceLedger,
        ctx: &ExecutionContext,
    ) -> Result<antecedent_design::scientific_source::ScopedDecisionResult, JointPanelError> {
        self.authorize(scope, ledger)?;
        self.source
            .evaluate_admitted(contract, scope, ctx)
            .map_err(super::search::map_scientific_error)
    }
    /// Execute the common finite search under caller-current evidence.
    /// # Errors
    /// Current authority, search semantics, admission or cancellation refuses execution.
    pub fn search(
        &self,
        search: &antecedent_design::program_search::PreparedFiniteProgramSearch,
        scope: antecedent_design::scientific_source::ScientificConsumptionScope,
        ledger: &EvidenceLedger,
        ctx: &ExecutionContext,
    ) -> Result<antecedent_design::scientific_source::ScopedProgramSearchResult, JointPanelError>
    {
        self.authorize(scope, ledger)?;
        self.source.search(search, scope, ctx).map_err(super::search::map_scientific_error)
    }
}

impl JointPanelResult {
    /// Condition this retained posterior on an independent experiment without refitting.
    ///
    /// The supplied paired mapping remains attached to the original draw slots. This
    /// is the same frozen modular update as `with_experiment`, with caller-current
    /// bridge and evidence checks repeated before every scientific consumption.
    ///
    /// The update executes the `JointPanelExperiment` condition program, a child of this
    /// response's program with the experiment as trial evidence; the result carries it.
    ///
    /// # Errors
    /// A sampling distribution, unsupported translation, stale evidence, mismatched
    /// mapping dependency or invalid likelihood cannot become a posterior update; a call
    /// inside another program's lowered dispatch is refused.
    pub fn condition_experiment(
        &self,
        update: PanelExperimentUpdate,
        ledger: &EvidenceLedger,
        ctx: &ExecutionContext,
    ) -> Result<PanelExperimentResult, JointPanelError> {
        let lowered = experiment_program(
            ExperimentOperation::Condition,
            self.program_identity(),
            update.evidence_identity(),
            update.evidence_bytes(),
            None,
        )?;
        let scoped = enter_experiment(ctx, &lowered)?;
        self.condition_experiment_in(update, ledger, &scoped, lowered, ctx)
    }

    /// The update of [`Self::condition_experiment`] inside the dispatch of `lowered`, whose
    /// release support gate `scoped` has passed. The result carries `lowered` and keeps
    /// `caller`, the context of the public entry point.
    pub(crate) fn condition_experiment_in(
        &self,
        update: PanelExperimentUpdate,
        ledger: &EvidenceLedger,
        scoped: &ExecutionContext,
        lowered: Arc<LoweredProgram>,
        caller: &ExecutionContext,
    ) -> Result<PanelExperimentResult, JointPanelError> {
        validate_update(&update, ledger)?;
        source::check_cancelled(scoped)?;
        update_result(self.clone(), update, ledger, scoped, ExecutedProgram::new(lowered, caller))
    }

    /// Exact assembled contrast coordinate for an experiment-to-model bridge.
    ///
    /// # Errors
    /// Requires a checked contrast and a valid projection covering its actual periods.
    pub fn experiment_quantity(
        &self,
        functional: &JointPanelResponseFunctional,
    ) -> Result<antecedent_contract::ScientificQuantity, JointPanelError> {
        contrast_weights(self.request())?;
        let means = self.numerical_project_mean_source(functional)?;
        let mut quantity = means
            .coordinates
            .first()
            .cloned()
            .ok_or_else(|| JointPanelError::compile("empty experimental target"))?;
        quantity.regime_id = format!("contrast:{}", self.request().target.identity());
        quantity.functional_id =
            format!("contrast:{}:{}", self.request().target.identity(), quantity.functional_id);
        Ok(quantity)
    }
}

impl PreparedStudy<JointPanelState> {
    /// Compile an experiment update through the same retained panel execution owner.
    ///
    /// The handle retains the `JointPanelExperiment` condition program, a child of this
    /// panel's program with the experiment as trial evidence. Lowering fits nothing.
    ///
    /// # Errors
    /// A bridge without a complete estimand translation, invalid likelihood/mapping, or
    /// withdrawn, overlapping, unknown-origin evidence refuses before fitting.
    pub fn with_experiment(
        &self,
        update: PanelExperimentUpdate,
        ledger: &EvidenceLedger,
    ) -> Result<PreparedStudy<PanelExperimentState>, JointPanelError> {
        validate_update(&update, ledger)?;
        if matches!(update.functional, JointPanelResponseFunctional::WindowChange { .. }) {
            return Err(JointPanelError::Unsupported(
                "experiment updates of two-window changes require an explicit two-window estimand translation; one enclosing window cannot authorize them".into(),
            ));
        }
        if !matches!(self.request().options.inference, engine::PanelInference::Bayesian { .. }) {
            return Err(JointPanelError::Unsupported(
                "cluster sampling draws cannot become a Bayesian experimental prior".into(),
            ));
        }
        let lowered = experiment_program(
            ExperimentOperation::Condition,
            self.program_identity(),
            update.evidence_identity(),
            update.evidence_bytes(),
            None,
        )?;
        Ok(PreparedStudy { state: PanelExperimentState { panel: self.clone(), update, lowered } })
    }
}

impl PreparedStudy<PanelExperimentState> {
    /// The experiment program this handle executes: the update of the parent panel's
    /// program by the declared experiment.
    #[must_use]
    pub fn lowered(&self) -> &Arc<LoweredProgram> {
        &self.state.lowered
    }

    /// Inspect the experiment program without fitting or conditioning.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        PlanView::of_program(Arc::clone(&self.state.lowered), PlanStage::Prepared)
    }

    /// Identity of the experiment program this handle executes.
    #[must_use]
    pub fn program_identity(&self) -> SemanticDigest {
        self.state.lowered.identity()
    }

    /// Execute the original shared mechanism once, then condition its complete response law.
    ///
    /// The experiment program's dispatch encloses the panel estimate, which runs as its
    /// child; the result carries the experiment program and keeps `ctx`.
    ///
    /// # Errors
    /// Fit, draw alignment, bridge quantity, current authority, numerical/resource or
    /// cancellation failures remain refusals; updating does not discharge identification.
    /// A call inside another program's lowered dispatch is refused.
    pub fn estimate(
        &self,
        ledger: &EvidenceLedger,
        ctx: &ExecutionContext,
    ) -> Result<PanelExperimentResult, JointPanelError> {
        let scoped = enter_experiment(ctx, &self.state.lowered)?;
        validate_update(&self.state.update, ledger)?;
        source::check_cancelled(&scoped)?;
        let mut original =
            self.state.panel.estimate_entry(Entry::Child(self.program_identity()), &scoped)?;
        // The panel result keeps the caller's worker context, not the dispatch-scoped one.
        original.context = ctx.serial_inner();
        let program = ExecutedProgram::new(Arc::clone(&self.state.lowered), ctx);
        update_result(original, self.state.update.clone(), ledger, &scoped, program)
    }
}

impl PanelExperimentResult {
    /// The experiment program this update executed: a child of the program of the panel
    /// response it conditioned, with the experiment as trial evidence.
    #[must_use]
    pub fn lowered(&self) -> &Arc<LoweredProgram> {
        &self.program.lowered
    }
    /// Inspect the experiment program this update executed.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        self.program.plan_view()
    }
    /// Identity of the experiment program this update executed.
    #[must_use]
    pub fn program_identity(&self) -> SemanticDigest {
        self.program.identity()
    }
    /// The conditioned law a decision or ranking on this update reads.
    fn conditioned(&self) -> ConditionedLaw {
        ConditionedLaw {
            program: self.program.identity(),
            law: self.dependency,
            bytes: law_bytes(&self.law),
            experiment: self.update.evidence.experiment,
        }
    }
    /// Original compiled result and retained identification obligations, without promotion.
    #[must_use]
    pub fn original(&self) -> &JointPanelResult {
        &self.original
    }
    /// This chain is conditional on the declared likelihood and paired mapping law.
    #[must_use]
    pub const fn scope(&self) -> &'static str {
        "model_conditional_experimental_update;calibration_not_measured"
    }
    /// Prior and updated means in the exact projected world coordinate order.
    ///
    /// # Errors
    /// Withdrawal, supersession, changed origins or invalid bridge authority refuse access.
    pub fn response_means(
        &self,
        ledger: &EvidenceLedger,
    ) -> Result<(&[f64], &[f64]), JointPanelError> {
        self.validate_current(ledger)?;
        Ok((&self.prior_means, &self.posterior_means))
    }
    /// Actual weighted aligned law accepted by existing decision evaluation.
    ///
    /// # Errors
    /// Current evidence/bridge authority must remain valid.
    pub fn numerical_law_source(
        &self,
        ledger: &EvidenceLedger,
    ) -> Result<&DistributionArtifact, JointPanelError> {
        self.validate_current(ledger)?;
        Ok(&self.law)
    }
    /// Updated law inseparable from its exact projected target and conditional standing.
    ///
    /// # Errors
    /// Current evidence, bridge authority or exact projection binding must remain valid.
    pub fn law_source(
        &self,
        ledger: &EvidenceLedger,
    ) -> Result<PanelExperimentDecisionSource, JointPanelError> {
        self.validate_current(ledger)?;
        let weights = self.original.projection(&self.update.functional)?.1;
        // The shared projection owner admits the borrowed updated law before
        // cloning it and retains the actual prefix target/map/payload reservation.
        let source = self.original.projected_law_source_with_dependency(
            &self.law,
            Some(weights),
            self.dependency,
        )?;
        Ok(PanelExperimentDecisionSource {
            source,
            update: Arc::clone(&self.update),
            original_dependency: self.original.dependency_identity(),
            retention: Arc::clone(&self.retention),
            companions: Vec::new(),
        })
    }

    /// Evaluate an existing decision contract on the updated whole law.
    ///
    /// The evaluation executes the `JointPanelExperiment` decision program, a child of this
    /// update's program, under the context the update was conditioned with.
    ///
    /// # Errors
    /// Existing decision semantics and current authority are enforced without relaxation.
    pub fn decide(
        &self,
        contract: &antecedent_design::decision_contract::DecisionContract,
        ledger: &EvidenceLedger,
    ) -> Result<antecedent_design::scientific_source::ScopedDecisionResult, JointPanelError> {
        self.conditioned().enter_decision(&self.program.caller, contract)?;
        self.law_source(ledger)?.evaluate(
            contract,
            antecedent_design::scientific_source::ScientificConsumptionScope::ModelConditional,
            ledger,
        )
    }
    fn validate_current(&self, ledger: &EvidenceLedger) -> Result<(), JointPanelError> {
        validate_update(&self.update, ledger)?;
        validate_model_origin(&self.original, &self.update, ledger)
    }
}

fn validate_update(
    update: &PanelExperimentUpdate,
    ledger: &EvidenceLedger,
) -> Result<(), JointPanelError> {
    update
        .bridge
        .check_support()
        .map_err(|error| JointPanelError::compile(format!("{error:?}")))?;
    if update.bridge.estimand_bridge.is_none()
        || update.bridge.source_dependence.is_none()
        || !update.observed_mean.is_finite()
        || !update.observation_variance.is_finite()
        || update.observation_variance <= 0.0
        || update.sample_size == 0
    {
        return Err(JointPanelError::compile(
            "experiment needs full population/window/dose/duration/exposure/recency/spend translation, dependence and a finite Gaussian sample-mean model",
        ));
    }
    let evidence = &update.evidence;
    let mut ids = vec![evidence.model, evidence.experiment, evidence.mapping];
    ids.extend(&evidence.prior_origins);
    for id in ids {
        if !ledger.source(id).is_some_and(|source| source.status == SourceStatus::Active) {
            return Err(JointPanelError::Unsupported(
                "experimental update evidence is missing, withdrawn or superseded".into(),
            ));
        }
    }
    for origin in
        [evidence.model, evidence.mapping].into_iter().chain(evidence.prior_origins.iter().copied())
    {
        if !matches!(
            ledger.overlap(origin, evidence.experiment).map_err(JointPanelError::compile)?,
            Overlap::Independent
        ) {
            return Err(JointPanelError::Unsupported(
                "experiment overlaps a model/prior/mapping origin or its origin is unknown".into(),
            ));
        }
    }
    if matches!(update.bridge.source_dependence, Some(SourceMappingDependence::Independent { .. }))
        && !matches!(
            ledger.overlap(evidence.model, evidence.mapping).map_err(JointPanelError::compile)?,
            Overlap::Independent
        )
    {
        return Err(JointPanelError::Unsupported(
            "mapping independence contradicts current ledger origins".into(),
        ));
    }
    if update.mapping.intercepts.len() != update.mapping.slopes.len()
        || update.mapping.intercepts.is_empty()
        || update.mapping.intercepts.iter().any(|value| !value.is_finite())
        || update.mapping.slopes.iter().any(|value| !value.is_finite() || *value == 0.0)
    {
        return Err(JointPanelError::compile(
            "mapping needs one finite nonzero-slope affine realization per retained source draw slot",
        ));
    }
    Ok(())
}
fn validate_model_origin(
    original: &JointPanelResult,
    update: &PanelExperimentUpdate,
    ledger: &EvidenceLedger,
) -> Result<(), JointPanelError> {
    validate_model_dependency(original.dependency_identity(), update, ledger)
}
fn validate_model_dependency(
    dependency: SemanticDigest,
    update: &PanelExperimentUpdate,
    ledger: &EvidenceLedger,
) -> Result<(), JointPanelError> {
    let model = ledger
        .source(update.evidence.model)
        .ok_or_else(|| JointPanelError::compile("missing model origin"))?;
    if update.mapping.dependency != dependency || model.declaration.label != dependency.to_hex() {
        return Err(JointPanelError::Unsupported(
            "experiment mapping/evidence refers to another fitted source or numerical dependency"
                .into(),
        ));
    }
    Ok(())
}

fn validate_bridge_projection(
    original: &JointPanelResult,
    update: &PanelExperimentUpdate,
) -> Result<(), JointPanelError> {
    validate_bridge_window(original, &update.functional, &update.bridge)
}
/// A bridge may authorize only the exact projected periods and quantity it names.
fn validate_bridge_window(
    original: &JointPanelResult,
    functional: &JointPanelResponseFunctional,
    bridge: &BridgeContract,
) -> Result<(), JointPanelError> {
    let quantity = original.experiment_quantity(functional)?;
    let translation = bridge
        .estimand_bridge
        .as_ref()
        .ok_or_else(|| JointPanelError::compile("missing experimental translation"))?;
    let horizon = original.request().worlds[0].program.horizon();
    if original.request().worlds.iter().any(|world| world.horizon() != horizon) {
        return Err(JointPanelError::Unsupported(
            "one experimental bridge window cannot authorize a contrast across different program horizons".into(),
        ));
    }
    let (from, until) = match functional {
        JointPanelResponseFunctional::TerminalMean => (horizon - 1, horizon - 1),
        JointPanelResponseFunctional::WindowAverage { from, until } => (*from, *until),
        JointPanelResponseFunctional::WeightedSum { .. } => (0, horizon - 1),
        JointPanelResponseFunctional::WindowChange { .. } => {
            return Err(JointPanelError::Unsupported(
                "experiment updates of two-window changes require an explicit two-window estimand translation; one enclosing window cannot authorize them".into(),
            ));
        }
    };
    if translation.target.time_start != format!("program_period:{from}")
        || translation.target.time_end != format!("program_period:{until}")
        || translation.target.duration.to_bits() != ((until - from + 1) as f64).to_bits()
        || translation.target.duration_units != "panel_period"
    {
        return Err(JointPanelError::compile(
            "target bridge window/duration must name the exact projected program periods; calendar transport is not inferred",
        ));
    }
    if bridge.target.quantity != quantity {
        return Err(JointPanelError::compile(
            "experimental bridge must target the exact assembled response functional, units, population, horizon and source",
        ));
    }
    Ok(())
}

/// Condition `original`'s projected posterior on `update` with the dispatch-scoped `ctx`; the
/// result carries `program`.
#[allow(
    clippy::too_many_lines,
    reason = "one update keeps its origin, posterior, mapping, admission and law checks together"
)]
fn update_result(
    original: JointPanelResult,
    update: PanelExperimentUpdate,
    ledger: &EvidenceLedger,
    ctx: &ExecutionContext,
    program: ExecutedProgram,
) -> Result<PanelExperimentResult, JointPanelError> {
    validate_model_origin(&original, &update, ledger)?;
    let law = original.numerical_project_law_source(&update.functional)?;
    if !matches!(
        law.semantic(),
        antecedent_io::quantity_wire::DistributionMeaningWire::CausalFunctionalPosterior
            | antecedent_io::quantity_wire::DistributionMeaningWire::ParameterPosterior
    ) {
        return Err(JointPanelError::Unsupported(
            "experimental likelihood updates require a posterior, never estimator sampling draws"
                .into(),
        ));
    }
    let coefficients = contrast_weights(original.request())?;
    validate_bridge_projection(&original, &update)?;
    let count = law.n_draws();
    let width = coefficients.len();
    if update.mapping.intercepts.len() != count {
        return Err(JointPanelError::compile(
            "mapping/source draw slots differ; implicit independent pairing is forbidden",
        ));
    }
    let bytes = count
        .checked_mul(width + 8)
        .and_then(|n| n.checked_mul(16))
        .and_then(|n| u64::try_from(n).ok())
        .ok_or_else(|| JointPanelError::compile("experiment quote overflow"))?;
    let retention = Arc::new(admit(ctx, bytes)?);
    let source_values =
        antecedent_prob::finite_belief::linear_functional_rows(law.draws(), &coefficients)
            .map_err(JointPanelError::compile)?;
    validate_mapping_moments(&update, &source_values)?;
    let prior_weights =
        law.metadata().weights.clone().unwrap_or_else(|| vec![1.0 / count as f64; count]);
    let weights = antecedent_prob::finite_belief::AffineGaussianExperiment {
        observed_mean: update.observed_mean,
        observation_variance: update.observation_variance,
        sample_size: update.sample_size,
        target_values: &source_values,
        intercepts: &update.mapping.intercepts,
        slopes: &update.mapping.slopes,
    }
    .condition(&prior_weights)
    .map_err(JointPanelError::compile)?;
    let prior_mass = prior_weights.iter().sum::<f64>();
    let mut prior_means = vec![0.0; width];
    for (row, weight) in law.draws().chunks_exact(width).zip(&prior_weights) {
        for (mean, value) in prior_means.iter_mut().zip(row) {
            *mean += weight / prior_mass * value;
        }
    }
    let mut posterior_means = vec![0.0; width];
    for (row, weight) in law.draws().chunks_exact(width).zip(&weights) {
        for (mean, value) in posterior_means.iter_mut().zip(row) {
            *mean += weight * value;
        }
    }
    if prior_means.iter().chain(&posterior_means).any(|mean| !mean.is_finite()) {
        return Err(JointPanelError::compile("experiment response mean overflow"));
    }
    let mut metadata = law.metadata().clone();
    metadata.weights = Some(weights.clone());
    let mut origins = vec![
        update.evidence.model.to_hex(),
        update.evidence.experiment.to_hex(),
        update.evidence.mapping.to_hex(),
    ];
    origins.extend(update.evidence.prior_origins.iter().map(SourceId::to_hex));
    let dependency = SemanticDigest::from_bytes(
        antecedent_io::joint_panel_experiment_artifact::experiment_update_identity(
            &metadata.identity.causal_contract_id,
            original.dependency_identity(),
            update.bridge.identity(),
            update.observed_mean,
            update.observation_variance,
            update.sample_size,
            &update.mapping.intercepts,
            &update.mapping.slopes,
            &weights,
            &origins,
        ),
    );
    metadata.identity.causal_contract_id = dependency.to_hex();
    // The declared experiment/mapping likelihood is an additional conditional model,
    // never a new nonparametric identification result.
    metadata.identity.semantic =
        antecedent_io::quantity_wire::DistributionMeaningWire::CausalFunctionalPosterior;
    metadata.identity.provider_id = "antecedent.compiled_joint_panel.conditional_experiment".into();
    metadata.calibration = DistributionCalibration::NotMeasured;
    metadata.trust = DistributionTrust::Unverified;
    let law = DistributionArtifact::new(metadata, law.draws().to_vec())
        .map_err(JointPanelError::compile)?;
    source::check_cancelled(ctx)?;
    Ok(PanelExperimentResult {
        original,
        update: Arc::new(update),
        law,
        prior_means,
        posterior_means,
        dependency,
        retention,
        program,
    })
}

fn contrast_weights(request: &JointPanelRequest) -> Result<Vec<f64>, JointPanelError> {
    let antecedent_core::TargetForm::Contrast(contrast) = request.target.form() else {
        return Err(JointPanelError::compile(
            "experimental update must name a checked assembled contrast, not a free coefficient vector",
        ));
    };
    let mut weights = vec![0.0; request.worlds.len()];
    for term in contrast.terms() {
        let index = request
            .worlds
            .iter()
            .position(|world| world.world == term.world)
            .ok_or_else(|| JointPanelError::compile("missing contrast world"))?;
        weights[index] = term.coefficient;
    }
    Ok(weights)
}
fn validate_mapping_moments(
    update: &PanelExperimentUpdate,
    source: &[f64],
) -> Result<(), JointPanelError> {
    update
        .bridge
        .validate_finite_affine_mapping(&update.mapping.intercepts, &update.mapping.slopes, source)
        .map_err(|error| JointPanelError::compile(format!("{error:?}")))
}

/// Existing terminal utility callbacks receive complete jointly aligned projected responses.
#[derive(Clone, Debug)]
pub struct PanelExperimentDecisionState {
    /// One mean functional per original world, in the exact projected coordinate order.
    pub responses: Vec<f64>,
    /// Experimental observation mean under that same parameter/mapping draw.
    pub experiment_mean: f64,
}

/// A smaller, explicitly supported next-study attempt: additional independent sampling
/// of the same mapped experimental estimand under its declared Gaussian model.
#[derive(Clone, Debug)]
pub struct PanelFutureSample {
    /// Stable candidate identity.
    pub id: String,
    /// Future independent observation count.
    pub sample_size: u64,
    /// Variance of one future observation, in the original experimental source units.
    pub observation_variance: f64,
    /// Optional cost already expressed in the decision utility unit; absent stays absent.
    pub cost_utility: Option<f64>,
}

/// Value of a declared future study, conditional on the retained model and update.
#[derive(Clone, Debug)]
pub struct PanelFutureSampleValue {
    /// Scientific scope of the complete update/selection procedure.
    pub scope: antecedent_design::scientific_source::ScientificConsumptionScope,
    /// Full question whose joint responses were updated.
    pub target_identity: SemanticDigest,
    /// Retained design/support proof, unchanged by fitting or updating.
    pub identification_identity: SemanticDigest,
    /// Exact source, bridge, observed-data and update dependency.
    pub dependency_identity: SemanticDigest,
    /// Exact projected question, including its sparse temporal functional.
    pub question_identity: SemanticDigest,
    /// Exact explicit terminal action domain used by every future-data update.
    pub admissible_action_identity: SemanticDigest,
    /// Actual prefix-world basis and projection used by terminal utility evaluation.
    pub projection: Option<Arc<antecedent_design::scientific_source::ScientificSourceProjection>>,
    retention: Arc<antecedent_core::execution::AdmissionLease>,
    /// Candidate identity.
    pub id: String,
    /// Expected improvement before study cost.
    pub gross_evsi: f64,
    /// Monte Carlo standard error for the complete finite-belief preposterior computation.
    pub standard_error: f64,
    /// Gross value minus cost only when cost was declared in the utility unit.
    pub net_value: Option<f64>,
    program: SemanticDigest,
}

impl PanelFutureSampleValue {
    /// Shared admitted reservation for the complete ranking output and projection.
    #[must_use]
    pub fn retained_admitted_bytes(&self) -> u64 {
        self.retention.observe_current();
        self.retention.remainder().bytes()
    }
    /// Identity of the `JointPanelExperiment` ranking program that valued this study: a
    /// child of the program that conditioned the ranked law.
    #[must_use]
    pub const fn program_identity(&self) -> SemanticDigest {
        self.program
    }
}

fn projection_retained_bytes(
    projection: Option<&antecedent_design::scientific_source::ScientificSourceProjection>,
) -> Result<usize, JointPanelError> {
    let quantity_bytes = |quantity: &antecedent_contract::ScientificQuantity| {
        let strings = [
            &quantity.variable_id,
            &quantity.variable_name,
            &quantity.units,
            &quantity.population_id,
            &quantity.regime_id,
            &quantity.functional_id,
            &quantity.transform_id,
        ];
        let mut bytes =
            strings.iter().try_fold(0_usize, |bytes, value| bytes.checked_add(value.len()))?;
        for condition in &quantity.conditioning {
            bytes = bytes
                .checked_add(std::mem::size_of_val(condition))?
                .checked_add(condition.variable_id.len())?
                .checked_add(condition.value_id.len())?;
        }
        Some(bytes)
    };
    let Some(projection) = projection else {
        return Ok(0);
    };
    let bytes = (|| {
        let mut bytes = std::mem::size_of_val(projection);
        for basis in &projection.basis {
            bytes = bytes
                .checked_add(std::mem::size_of_val(basis))?
                .checked_add(quantity_bytes(&basis.quantity)?)?;
        }
        for output in &projection.outputs {
            bytes = bytes
                .checked_add(std::mem::size_of_val(output))?
                .checked_add(quantity_bytes(&output.quantity)?)?
                .checked_add(
                    output.terms.len().checked_mul(std::mem::size_of::<(usize, f64)>())?,
                )?;
        }
        Some(bytes)
    })();
    bytes.ok_or_else(|| JointPanelError::compile("future-study projection quote overflow"))
}

fn validate_future_sample_catalog(
    candidates: &[PanelFutureSample],
    replicates: usize,
) -> Result<(), JointPanelError> {
    // Candidate and replicate counts are the caller's work budget: retention is admitted
    // against the memory limit and every replicate checks cancellation.
    if candidates.is_empty()
        || replicates < 2
        || candidates.iter().any(|candidate| {
            candidate.id.trim().is_empty()
                || candidate.sample_size == 0
                || !candidate.observation_variance.is_finite()
                || candidate.observation_variance <= 0.0
                || candidate.observation_variance / candidate.sample_size as f64 == 0.0
                || candidate.cost_utility.is_some_and(|cost| !cost.is_finite() || cost < 0.0)
        })
    {
        return Err(JointPanelError::compile(
            "invalid future-sample catalog, Gaussian model, declared utility cost or integration budget",
        ));
    }
    let mut ids = std::collections::BTreeSet::new();
    if candidates.iter().any(|candidate| !ids.insert(&candidate.id)) {
        return Err(JointPanelError::compile("duplicate future sample identity"));
    }
    let costed = candidates.iter().filter(|candidate| candidate.cost_utility.is_some()).count();
    if costed != 0 && costed != candidates.len() {
        return Err(JointPanelError::compile(
            "a net-value ranking needs a utility-unit cost for every candidate, or for none; a partly costed catalog would silently rank by gross value",
        ));
    }
    Ok(())
}

struct PanelExperimentSignal {
    variance: f64,
}
impl DecisionSignal<PanelExperimentDecisionState> for PanelExperimentSignal {
    fn name(&self) -> &'static str {
        "joint_panel.declared_gaussian_experiment"
    }
    fn log_likelihood(
        &self,
        statistic: f64,
        n: u64,
        states: &[PanelExperimentDecisionState],
        out: &mut [f64],
    ) -> Result<(), DesignError> {
        let values = states.iter().map(|state| state.experiment_mean).collect::<Vec<_>>();
        antecedent_design::preposterior::GaussianMeanSignal::new(self.variance)?
            .log_likelihood(statistic, n, &values, out)
    }
    fn sample_statistic(
        &self,
        state: &PanelExperimentDecisionState,
        n: u64,
        rng: &mut antecedent_core::CausalRng,
    ) -> Result<f64, DesignError> {
        antecedent_design::preposterior::GaussianMeanSignal::new(self.variance)?.sample_statistic(
            &state.experiment_mean,
            n,
            rng,
        )
    }
}

impl PanelExperimentResult {
    /// Exact mean-functional coordinates used by response and terminal decision callbacks.
    ///
    /// # Errors
    /// Current authority or quantity conversion fails.
    pub fn quantities(
        &self,
        ledger: &EvidenceLedger,
    ) -> Result<Vec<antecedent_contract::ScientificQuantity>, JointPanelError> {
        self.validate_current(ledger)?;
        self.law
            .metadata()
            .identity
            .quantities
            .iter()
            .cloned()
            .map(|quantity| {
                antecedent_contract::ScientificQuantity::try_from(quantity)
                    .map_err(JointPanelError::compile)
            })
            .collect()
    }
    /// Rank future sample sizes with the existing weighted preposterior engine.
    ///
    /// The same mapped experiment is sampled, its likelihood updates the exact current
    /// finite posterior weights, and the caller's terminal actions/utility are evaluated
    /// on every whole response row. This does not rank unsupported new populations,
    /// exposure mechanisms or nonlinear mappings. Utilities declare their own units;
    /// optional candidate costs must already be expressed in that same unit.
    ///
    /// The ranking executes the `JointPanelExperiment` ranking program, a child of this
    /// update's program; every value carries its identity.
    ///
    /// # Errors
    /// Invalid study/cost/replicate bounds, current authority, resources/cancellation,
    /// unsupported weighted constraints or callback failures refuse the ranking; a call
    /// inside another program's lowered dispatch is refused.
    pub fn rank_next_samples<A: serde::Serialize>(
        &self,
        problem: &DecisionProblem<A, PanelExperimentDecisionState>,
        candidates: &[PanelFutureSample],
        replicates: usize,
        ledger: &EvidenceLedger,
        ctx: &ExecutionContext,
    ) -> Result<Vec<PanelFutureSampleValue>, JointPanelError> {
        let (program, scoped) =
            self.conditioned().enter_ranking(ctx, problem, candidates, &[], replicates)?;
        self.retention.observe_current();
        let scientific = self.law_source(ledger)?;
        let coefficients = contrast_weights(self.original.request())?;
        rank_law_next_samples(
            &ExperimentRankingReceipt::new(&scientific, program),
            &self.law,
            &self.update,
            &coefficients,
            problem,
            candidates,
            replicates,
            &scoped,
        )
    }
}
struct ExperimentRankingReceipt {
    target_identity: SemanticDigest,
    identification_identity: SemanticDigest,
    dependency_identity: SemanticDigest,
    question_identity: SemanticDigest,
    projection: Option<Arc<antecedent_design::scientific_source::ScientificSourceProjection>>,
    /// Identity of the ranking program the values are computed by.
    program: SemanticDigest,
}
impl ExperimentRankingReceipt {
    /// The receipt of a ranking program `program` over the law of `source`.
    fn new(source: &PanelExperimentDecisionSource, program: SemanticDigest) -> Self {
        Self {
            target_identity: source.target().identity(),
            identification_identity: source.identification().identity(),
            dependency_identity: source.dependency_identity(),
            question_identity: source.question_identity(),
            projection: source.projection().cloned().map(Arc::new),
            program,
        }
    }
}
#[allow(
    clippy::too_many_arguments,
    reason = "the ranking reads its receipt, law, update, contrast, decision, catalog and budget"
)]
fn rank_law_next_samples<A: serde::Serialize>(
    receipt: &ExperimentRankingReceipt,
    law: &DistributionArtifact,
    update: &PanelExperimentUpdate,
    coefficients: &[f64],
    problem: &DecisionProblem<A, PanelExperimentDecisionState>,
    candidates: &[PanelFutureSample],
    replicates: usize,
    ctx: &ExecutionContext,
) -> Result<Vec<PanelFutureSampleValue>, JointPanelError> {
    rank_law_next_studies(
        receipt,
        law,
        update,
        coefficients,
        problem,
        candidates,
        &vec![None; candidates.len()],
        replicates,
        ctx,
    )
}
/// Ranking core. `candidate_means[i]`, when present, is the experimental mean each retained
/// draw slot implies for candidate `i`'s own regime; absent means the update's own estimand.
#[allow(
    clippy::too_many_arguments,
    reason = "the ranking reads its receipt, law, update, contrast, decision, catalog and budget"
)]
fn rank_law_next_studies<A: serde::Serialize>(
    receipt: &ExperimentRankingReceipt,
    law: &DistributionArtifact,
    update: &PanelExperimentUpdate,
    coefficients: &[f64],
    problem: &DecisionProblem<A, PanelExperimentDecisionState>,
    candidates: &[PanelFutureSample],
    candidate_means: &[Option<Vec<f64>>],
    replicates: usize,
    ctx: &ExecutionContext,
) -> Result<Vec<PanelFutureSampleValue>, JointPanelError> {
    source::check_cancelled(ctx)?;
    validate_future_sample_catalog(candidates, replicates)?;
    if candidate_means.len() != candidates.len()
        || candidate_means
            .iter()
            .flatten()
            .any(|means| means.len() != law.n_draws() || means.iter().any(|mean| !mean.is_finite()))
    {
        return Err(JointPanelError::compile(
            "each regime-specific study needs one finite experimental mean per retained draw slot",
        ));
    }
    let output_retention = future_study_output_retention(receipt, problem, candidates, ctx)?;
    let admissible_action_identity = future_study_action_identity(receipt, problem)?;
    let projection = receipt.projection.clone();
    let width = coefficients.len();
    let weights =
        law.metadata().weights.as_ref().ok_or_else(|| {
            JointPanelError::compile("conditional law must retain posterior weights")
        })?;
    let bytes = future_preposterior_bytes(law, problem.actions.len(), weights.len())?;
    let _lease = admit(ctx, bytes)?;
    let states = law
        .draws()
        .chunks_exact(width)
        .enumerate()
        .map(|(slot, row)| {
            let target =
                row.iter().zip(coefficients).map(|(value, weight)| value * weight).sum::<f64>();
            PanelExperimentDecisionState {
                responses: row.to_vec(),
                experiment_mean: (target - update.mapping.intercepts[slot])
                    / update.mapping.slopes[slot],
            }
        })
        .collect::<Vec<_>>();
    if states.iter().any(|state| !state.experiment_mean.is_finite()) {
        return Err(JointPanelError::compile("future experiment mapped mean overflow"));
    }
    let mut rows = Vec::with_capacity(candidates.len());
    let base_means = states.iter().map(|state| state.experiment_mean).collect::<Vec<_>>();
    let mut states = states;
    for (candidate, regime_means) in candidates.iter().zip(candidate_means) {
        source::check_cancelled(ctx)?;
        // The study's own regime decides what each retained draw predicts it will observe;
        // the responses and weights the terminal decision uses are unchanged.
        for (slot, state) in states.iter_mut().enumerate() {
            state.experiment_mean = regime_means.as_ref().map_or(base_means[slot], |m| m[slot]);
        }
        let signal = PanelExperimentSignal { variance: candidate.observation_variance };
        let analysis = PreposteriorAnalysis::new_weighted(problem, &states, weights, &signal)
            .map_err(JointPanelError::compile)?;
        let mut samples = antecedent_stats::Welford::new();
        for replicate in 0..replicates {
            source::check_cancelled(ctx)?;
            let mut rng =
                antecedent_design::preposterior::evsi_replicate_rng(&ctx.rng, replicate as u64);
            samples.push(
                analysis
                    .sample_evsi(candidate.sample_size, &mut rng)
                    .map_err(JointPanelError::compile)?,
            );
        }
        rows.push(PanelFutureSampleValue {
            scope:
                antecedent_design::scientific_source::ScientificConsumptionScope::ModelConditional,
            target_identity: receipt.target_identity,
            identification_identity: receipt.identification_identity,
            dependency_identity: receipt.dependency_identity,
            question_identity: receipt.question_identity,
            admissible_action_identity,
            projection: projection.clone(),
            retention: Arc::clone(&output_retention),
            id: candidate.id.clone(),
            gross_evsi: samples.mean(),
            standard_error: samples.stderr_of_mean(),
            net_value: candidate.cost_utility.map(|cost| samples.mean() - cost),
            program: receipt.program,
        });
    }
    // A net ranking is meaningful only when every candidate declares a comparable cost.
    let net = rows.iter().all(|row| row.net_value.is_some());
    rows.sort_by(|left, right| {
        right
            .net_value
            .filter(|_| net)
            .unwrap_or(right.gross_evsi)
            .total_cmp(&left.net_value.filter(|_| net).unwrap_or(left.gross_evsi))
            .then_with(|| left.id.cmp(&right.id))
    });
    Ok(rows)
}

fn future_preposterior_bytes(
    law: &DistributionArtifact,
    n_actions: usize,
    n_weights: usize,
) -> Result<u64, JointPanelError> {
    // The existing decision owner retains both its action-by-state utility table
    // and the corresponding regret gaps, in addition to aligned response states.
    let action_cells = n_actions
        .checked_mul(n_weights)
        .and_then(|cells| cells.checked_mul(3))
        .ok_or_else(|| JointPanelError::compile("preposterior action table quote overflow"))?;
    law.draws()
        .len()
        .checked_mul(3)
        .and_then(|cells| cells.checked_add(action_cells))
        .and_then(|cells| cells.checked_add(n_weights.checked_mul(16)?))
        .and_then(|cells| cells.checked_mul(8))
        .and_then(|n| u64::try_from(n).ok())
        .ok_or_else(|| JointPanelError::compile("preposterior quote overflow"))
}

fn admit(
    ctx: &ExecutionContext,
    bytes: u64,
) -> Result<antecedent_core::execution::AdmissionLease, JointPanelError> {
    let limit = ctx.memory_limit_bytes();
    if limit.is_some_and(|limit| bytes > limit) {
        return Err(engine::PanelError::Resource(
            "configured experiment/update memory limit".into(),
        )
        .into());
    }
    ctx.acquire_admission(0, bytes)
        .map_err(|error| engine::PanelError::Resource(error.to_string()).into())
}

impl PanelExperimentResult {
    /// Preserve the exact experiment, uncertain mapping, selected origins and frozen panel.
    /// # Errors
    /// Current withdrawal/supersession, projection, bridge/ledger or resource failure.
    pub fn artifact_program(
        &self,
        ledger: &EvidenceLedger,
    ) -> Result<
        antecedent_io::joint_panel_experiment_artifact::PanelExperimentExecutionProgram,
        antecedent_io::joint_panel_artifact::PanelArtifactError,
    > {
        use antecedent_io::joint_panel_experiment_artifact::{
            PanelExperimentExecutionProgram, PanelExperimentExecutionSource, PanelExperimentOrigins,
        };
        let invalid = |error: JointPanelError| {
            antecedent_io::joint_panel_artifact::PanelArtifactError::Invalid(error.to_string())
        };
        self.validate_current(ledger).map_err(invalid)?;
        let (_, projection) = self.original.projection(&self.update.functional).map_err(invalid)?;
        let prior =
            self.original.numerical_project_law_source(&self.update.functional).map_err(invalid)?;
        PanelExperimentExecutionProgram::from_source(PanelExperimentExecutionSource {
            panel: self.original.artifact_program()?,
            bridge: &self.update.bridge,
            ledger,
            origins: PanelExperimentOrigins {
                model: self.update.evidence.model.to_hex(),
                experiment: self.update.evidence.experiment.to_hex(),
                mapping: self.update.evidence.mapping.to_hex(),
                priors: self.update.evidence.prior_origins.iter().map(SourceId::to_hex).collect(),
            },
            projection,
            prior_metadata: prior.metadata().clone(),
            observed_mean: self.update.observed_mean,
            observation_variance: self.update.observation_variance,
            sample_size: self.update.sample_size,
            intercepts: &self.update.mapping.intercepts,
            slopes: &self.update.mapping.slopes,
            mapping_dependency: self.update.mapping.dependency,
        })
    }
    /// Encode the actual conditional update without fitting another mechanism.
    /// # Errors
    /// All current authority, original artifact, serialization, resource or cancellation refusals.
    pub fn export(
        &self,
        ledger: &EvidenceLedger,
        ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, antecedent_io::joint_panel_artifact::PanelArtifactError> {
        antecedent_io::joint_panel_experiment_artifact::encode_panel_experiment(
            self.artifact_program(ledger)?,
            &self.law,
            ctx,
        )
    }
    /// Verify historical evaluation/update against actual current result and live ledger.
    /// # Errors
    /// Any changed selected authority, numerical request or frozen fit law refuses consumption.
    pub fn consume_current(
        &self,
        bytes: &[u8],
        ledger: &EvidenceLedger,
        ctx: &ExecutionContext,
    ) -> Result<Self, antecedent_io::joint_panel_artifact::PanelArtifactError> {
        self.validate_current(ledger).map_err(|error| {
            antecedent_io::joint_panel_artifact::PanelArtifactError::Invalid(error.to_string())
        })?;
        let replay =
            antecedent_io::joint_panel_experiment_artifact::consume_panel_experiment(bytes, ctx)?;
        if replay.program().identity()? != self.artifact_program(ledger)?.identity()?
            || replay.dependency_identity() != self.dependency
        {
            return Err(antecedent_io::joint_panel_artifact::PanelArtifactError::Mismatch);
        }
        Ok(self.clone())
    }
}

/// Admit all retained result rows and their shared action-domain/projection ownership.
fn future_study_output_retention<A: serde::Serialize>(
    receipt: &ExperimentRankingReceipt,
    problem: &DecisionProblem<A, PanelExperimentDecisionState>,
    candidates: &[PanelFutureSample],
    ctx: &ExecutionContext,
) -> Result<Arc<antecedent_core::execution::AdmissionLease>, JointPanelError> {
    let mut action_size = law::WireCounter(0);
    serde_json::to_writer(&mut action_size, &problem.actions)
        .map_err(|error| JointPanelError::compile(error.to_string()))?;
    let projection_bytes = projection_retained_bytes(receipt.projection.as_deref())?
        .checked_add(action_size.0.checked_mul(2).ok_or_else(|| {
            JointPanelError::compile("action-domain serialization quote overflow")
        })?)
        .and_then(|n| n.checked_add(std::any::type_name::<A>().len()))
        .and_then(|n| n.checked_add(256))
        .ok_or_else(|| JointPanelError::compile("future-study shared ownership quote overflow"))?;
    let output_bytes = candidates
        .iter()
        .try_fold(projection_bytes, |bytes, candidate| {
            bytes
                .checked_add(std::mem::size_of::<PanelFutureSampleValue>())?
                .checked_add(candidate.id.len())
        })
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| JointPanelError::compile("future-study output quote overflow"))?;
    Ok(Arc::new(admit(ctx, output_bytes)?))
}

/// Bind the actual serialized admissible actions, their Rust type, and the ranking question.
fn future_study_action_identity<A: serde::Serialize>(
    receipt: &ExperimentRankingReceipt,
    problem: &DecisionProblem<A, PanelExperimentDecisionState>,
) -> Result<SemanticDigest, JointPanelError> {
    let mut action_writer =
        antecedent_core::CanonicalWriter::new("antecedent.future_study.explicit_action_domain.v1");
    action_writer
        .bytes(receipt.question_identity.as_bytes())
        .text(std::any::type_name::<A>())
        .bytes(
            &serde_json::to_vec(&problem.actions)
                .map_err(|error| JointPanelError::compile(error.to_string()))?,
        );
    Ok(SemanticDigest::from_bytes(action_writer.finish()))
}

impl crate::analysis::program_handle::CompiledProgramHandle
    for PreparedStudy<PanelExperimentState>
{
    fn program(&self) -> &std::sync::Arc<crate::planner::lowered::LoweredProgram> {
        &self.state.lowered
    }
}

#[cfg(test)]
mod catalog_size_tests {
    use super::{PanelFutureSample, validate_future_sample_catalog};

    fn candidates(count: usize) -> Vec<PanelFutureSample> {
        (0..count)
            .map(|i| PanelFutureSample {
                id: format!("study-{i}"),
                sample_size: 10,
                observation_variance: 1.0,
                cost_utility: None,
            })
            .collect()
    }

    #[test]
    fn catalog_and_replicate_counts_are_not_capped_by_literals() {
        assert!(validate_future_sample_catalog(&candidates(2_000), 250_000).is_ok());
        assert!(validate_future_sample_catalog(&candidates(0), 1_000).is_err());
        assert!(validate_future_sample_catalog(&candidates(3), 1).is_err());
    }
}

#[cfg(test)]
mod program_tests {
    use super::{
        Arc, ConditionedLaw, DecisionProblem, ExecutionContext, PanelExperimentDecisionState,
        PanelFutureSample, SemanticDigest, SourceId,
    };
    use antecedent_design::DesignError;
    use antecedent_design::decision::Utility;

    struct Flat;

    impl Utility<usize, PanelExperimentDecisionState> for Flat {
        fn evaluate_batch(
            &self,
            _: &[usize],
            _: &[PanelExperimentDecisionState],
            out: &mut [f64],
        ) -> Result<(), DesignError> {
            out.fill(0.0);
            Ok(())
        }
    }

    fn law() -> ConditionedLaw {
        ConditionedLaw {
            program: SemanticDigest::from_bytes([1; 32]),
            law: SemanticDigest::from_bytes([2; 32]),
            bytes: 64,
            experiment: SourceId::from_digest(SemanticDigest::from_bytes([3; 32])),
        }
    }

    fn sample(id: &str, cost_utility: Option<f64>) -> PanelFutureSample {
        PanelFutureSample {
            id: id.into(),
            sample_size: 10,
            observation_variance: 1.0,
            cost_utility,
        }
    }

    #[test]
    fn a_ranking_program_binds_its_catalog_and_runs_inside_its_own_dispatch() {
        let ctx = ExecutionContext::for_tests(5);
        let problem = DecisionProblem::new(vec![0_usize, 1], Arc::new(Flat), vec![]);
        let catalog = [sample("b", None), sample("a", None)];
        let (program, scoped) = law().enter_ranking(&ctx, &problem, &catalog, &[], 9).unwrap();
        assert_eq!(scoped.lowered_dispatch(), Some(program));
        let resized =
            [sample("b", None), PanelFutureSample { sample_size: 20, ..sample("a", None) }];
        let (other, _) = law().enter_ranking(&ctx, &problem, &resized, &[], 9).unwrap();
        assert_ne!(program, other, "a candidate's sampling model is part of the program");
        let regime = [SemanticDigest::from_bytes([4; 32])];
        let (with_regime, _) = law().enter_ranking(&ctx, &problem, &catalog, &regime, 9).unwrap();
        assert_ne!(program, with_regime, "a declared regime is part of the program");
        assert!(
            law().enter_ranking(&scoped, &problem, &catalog, &[], 9).is_err(),
            "a ranking inside another program's dispatch is refused"
        );
    }

    #[test]
    fn the_information_value_lists_candidates_in_canonical_order() {
        let ctx = ExecutionContext::for_tests(5);
        let catalog = [sample("b", Some(1.0)), sample("a", Some(2.0))];
        let value = law().information_value(&catalog, 9, &ctx).unwrap();
        let ids = value.candidates().iter().map(|id| &**id).collect::<Vec<_>>();
        assert_eq!(ids, ["a", "b"]);
        assert_eq!(value.monte_carlo().draws(), 9);
        assert_eq!(value.prior_observation_ids().len(), 1);
        assert!(value.cost_map().is_some());
        let uncosted = law().information_value(&[sample("a", None)], 9, &ctx).unwrap();
        assert!(uncosted.cost_map().is_none());
    }
}
