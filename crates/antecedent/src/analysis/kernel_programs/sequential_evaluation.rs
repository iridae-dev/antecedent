//! Exact evaluation of a declared finite-horizon contingent policy, as one program family.
//!
//! `SequentialPolicyEvaluation` is bound to `sequential_evaluation.policy_evaluation`. Compiling
//! stores the caller's inputs and the digest of their canonical bytes. Executing passes that
//! unit's support gate and then calls the kernel function the direct path calls, with those
//! inputs:
//!
//! * [`CompiledPolicyEvaluation::evaluate`] calls `evaluate_policy`
//! * [`CompiledOffPolicyEvaluation::evaluate`] calls `evaluate_off_policy` (the caller's ledger
//!   is execution state, not part of the digest)
//! * [`CompiledPolicyRollout::evaluate`] calls `rollout_policy`
//! * [`CompiledPolicyComparison::evaluate`] calls `compare_policies`
//! * [`CompiledValueOfCollectingEvidence::evaluate`] calls `value_of_collecting_evidence`
//! * [`CompiledSequentialEvaluationSeal::seal`] calls [`SequentialEvaluationArtifact::new`]
//!
//! `SequentialPolicyReliability` keeps calling `evaluate_policy` and `rollout_policy` after its
//! own gate. These handles do not compile again.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::DataClassification;
use antecedent_contract::evidence_ledger::EvidenceLedger;
use antecedent_core::{CanonicalWriter, ExecutionContext, NotSupported};
use antecedent_design::sequential_evaluation::{
    ActionSpec, ActionSupport, ComparisonReport, ComparisonSpec, ContingentPolicy,
    EvaluationLimits, FiniteSequentialModel, HistoryStep, LoggedStep, LoggedTrajectory,
    ObservationChannel, OffPolicyEstimate, PolicyEvaluation, PolicySpec, RolloutConfig,
    RolloutEstimate, SequentialEvaluationError, SequentialRequest, SequentialResult,
    StoppingBoundary, VoiCheck, VoiRequest, compare_policies, evaluate_off_policy, evaluate_policy,
    rollout_policy, value_of_collecting_evidence,
};
use antecedent_design::sequential_evaluation_artifact::{
    OffPolicyInput, SequentialEvaluationArtifact,
};

use super::{KernelProgram, KernelProgramShape, KernelRealization};
use crate::analysis::Study;
use crate::analysis::supplied_identity::{Canonical, Declaration};
use crate::error::CausalError;
use crate::planner::PlanView;
use crate::planner::lowered::{
    EvidenceRole, FunctionalKind, LoweredProgram, ProgramCell, UncertaintyReading,
};

const GRAPH: &str = "finite_horizon_dynamic_structural_model";

const G_FORMULA_PLAN: &str = "sequential_evaluation.policy_evaluation.g_formula";
const OFF_POLICY_PLAN: &str = "sequential_evaluation.policy_evaluation.off_policy";
const ROLLOUT_PLAN: &str = "sequential_evaluation.policy_evaluation.rollout";
const COMPARE_PLAN: &str = "sequential_evaluation.policy_evaluation.compare";
const VALUE_OF_EVIDENCE_PLAN: &str = "sequential_evaluation.policy_evaluation.value_of_evidence";
const SEAL_PLAN: &str = "sequential_evaluation.policy_evaluation.seal";

const G_FORMULA_TAG: &str = "antecedent.sequential_evaluation.g_formula.v1";
const OFF_POLICY_TAG: &str = "antecedent.sequential_evaluation.off_policy.v1";
const ROLLOUT_TAG: &str = "antecedent.sequential_evaluation.rollout.v1";
const COMPARE_TAG: &str = "antecedent.sequential_evaluation.compare.v1";
const VALUE_OF_EVIDENCE_TAG: &str = "antecedent.sequential_evaluation.value_of_evidence.v1";
const SEAL_TAG: &str = "antecedent.sequential_evaluation.seal.v1";

/// What would make the unit's support refusal evaluable.
const GATE_REMEDY: &str =
    "run a development build, which records the missing evidence, or use a supported route";

const fn family(
    plan_id: &'static str,
    evaluator: &'static str,
    design: &'static str,
) -> KernelProgramShape {
    KernelProgramShape {
        cell: ProgramCell::SequentialPolicyEvaluation,
        plan_id,
        functional: FunctionalKind::PolicyValue,
        evaluator,
        realization: KernelRealization::Exact,
        uncertainty: UncertaintyReading::None,
        role: EvidenceRole::Observations,
        graph_class: GRAPH,
        data: DataClassification::Temporal,
        design,
    }
}

const G_FORMULA: KernelProgramShape = family(
    G_FORMULA_PLAN,
    "sequential_g_formula",
    "exact sequential g-formula of a declared contingent policy",
);
const OFF_POLICY: KernelProgramShape = family(
    OFF_POLICY_PLAN,
    "sequential_importance_weighted_mean",
    "importance-weighted mean of a declared log under a declared behaviour policy",
);
const ROLLOUT: KernelProgramShape = family(
    ROLLOUT_PLAN,
    "sequential_common_random_rollout",
    "common-random-number rollout of a declared contingent policy",
);
const COMPARE: KernelProgramShape = family(
    COMPARE_PLAN,
    "sequential_policy_comparison",
    "exact comparison of declared contingent policies",
);
const VALUE_OF_EVIDENCE: KernelProgramShape = family(
    VALUE_OF_EVIDENCE_PLAN,
    "sequential_value_of_collecting_evidence",
    "value of one declared observation before a declared commit",
);
const SEAL: KernelProgramShape = family(
    SEAL_PLAN,
    "sequential_evaluation_artifact",
    "sealed replay of a declared sequential evaluation",
);

fn word_len(len: usize) -> u64 {
    u64::try_from(len).expect("a collection length fits in a canonical word")
}

fn declare(tag: &str, fields: impl FnOnce(&mut Declaration)) -> [u8; 32] {
    let mut declaration = Declaration::new(tag);
    fields(&mut declaration);
    declaration.finish()
}

fn open<I>(
    shape: &KernelProgramShape,
    input: I,
    canonical: [u8; 32],
    ctx: &ExecutionContext,
) -> Result<KernelProgram<I>, CausalError> {
    KernelProgram::compile(shape, input, &canonical, ctx)
}

/// The support refusal carried on the kernel error the direct path already returns.
///
/// No `sequential_evaluation.*` detail is paired with `not_supported`, so the gate's own code
/// and detail pass through. The detail is the route name the gate already set.
fn evaluation_refusal(refusal: &NotSupported) -> SequentialEvaluationError {
    SequentialEvaluationError {
        code: refusal.code(),
        detail: refusal.detail.to_owned(),
        message: refusal.message().to_owned(),
        remedy: GATE_REMEDY,
    }
}

// ---------------------------------------------------------------------------
// Canonical bytes. Each collection is classified from the kernel's read.
// ---------------------------------------------------------------------------

impl Canonical for HistoryStep {
    fn encode(&self, out: &mut CanonicalWriter) {
        self.action.encode(out);
        self.observation.encode(out);
    }
}

impl Canonical for ActionSpec {
    fn encode(&self, out: &mut CanonicalWriter) {
        out.text(&self.label).text(self.role.as_str());
        out.real(self.utility_cost).real(self.resource_use);
        // Row-major transition, positional in the declared state order.
        self.transition.encode(out);
        match &self.channel {
            Some(ObservationChannel { measurement }) => {
                out.presence(true);
                measurement.identities().model.encode(out);
            }
            None => {
                out.presence(false);
            }
        }
        self.immediate_utility.encode(out);
        match &self.delayed {
            Some(delayed) => {
                out.presence(true);
                delayed.delay.encode(out);
                delayed.utility.encode(out);
            }
            None => {
                out.presence(false);
            }
        }
    }
}

impl Canonical for StoppingBoundary {
    fn encode(&self, out: &mut CanonicalWriter) {
        out.text(&self.label);
        // The kernel sums `belief[state]` and refuses a repeated state. Order is not a read.
        // Sorted, not deduplicated: a duplicate the kernel would refuse stays visible.
        let mut states = self.states.clone();
        states.sort_unstable();
        states.encode(out);
        out.opt_real(self.upper).opt_real(self.lower);
        self.stop_action.encode(out);
    }
}

impl Canonical for ActionSupport {
    fn encode(&self, out: &mut CanonicalWriter) {
        // Default flags are positional in the action index.
        self.default_allowed().encode(out);
        // Overrides are a set keyed by history. `overrides` yields canonical history order.
        let overrides: Vec<_> = self.overrides().collect();
        out.word(word_len(overrides.len()));
        for (history, flags) in overrides {
            history.steps().encode(out);
            flags.encode(out);
        }
    }
}

impl Canonical for FiniteSequentialModel {
    fn encode(&self, out: &mut CanonicalWriter) {
        // `label` is a display name. The kernel's own identity omits it, and evaluation does
        // not read it.
        out.text(&self.population).text(&self.regime).text(&self.period).text(&self.state_variable);
        self.states.encode(out);
        self.initial.encode(out);
        self.horizon.encode(out);
        self.actions.encode(out);
        self.terminal_utility.encode(out);
        out.text(self.censoring.as_str());
        out.opt_real(self.resource_budget);
        // Boundaries are positional: the first crossed boundary wins.
        self.stopping.encode(out);
        self.support.encode(out);
    }
}

impl Canonical for ContingentPolicy {
    fn encode(&self, out: &mut CanonicalWriter) {
        out.text(self.version());
        // Decisions are a set keyed by history. `entries` yields canonical history order.
        // Each stored distribution is already sorted by action; the kernel looks up by action.
        let entries: Vec<_> = self.entries().collect();
        out.word(word_len(entries.len()));
        for (history, distribution) in entries {
            history.steps().encode(out);
            distribution.encode(out);
        }
    }
}

impl Canonical for PolicySpec {
    fn encode(&self, out: &mut CanonicalWriter) {
        match self {
            Self::Contingent(policy) => {
                out.text("contingent");
                policy.encode(out);
            }
            Self::ReweightedStaticScores { scores, weights } => {
                // The kernel reads only the lengths, in the refusal. The sequences are still
                // the declared table, in the order supplied, so two tables do not share a digest.
                out.text("reweighted_static_scores");
                scores.encode(out);
                weights.encode(out);
            }
            Self::LearnedFromEvaluationData { rule } => {
                out.text("learned_from_evaluation_data");
                out.text(rule);
            }
        }
    }
}

impl Canonical for LoggedStep {
    fn encode(&self, out: &mut CanonicalWriter) {
        self.action.encode(out);
        self.observation.encode(out);
        out.real(self.assignment_probability);
    }
}

impl Canonical for LoggedTrajectory {
    fn encode(&self, out: &mut CanonicalWriter) {
        out.text(&self.id);
        // Steps are a trajectory's decision cycles, oldest first.
        self.steps.encode(out);
        out.real(self.outcome_utility);
    }
}

/// A log in the order `evaluate_off_policy` reads it: sorted by trajectory id.
///
/// Equal ids keep their original relative order and are not collapsed. The kernel refuses a
/// repeated id, so an accepted log's order is the id order alone.
struct LogById<'a>(&'a [LoggedTrajectory]);

impl Canonical for LogById<'_> {
    fn encode(&self, out: &mut CanonicalWriter) {
        let mut order: Vec<usize> = (0..self.0.len()).collect();
        order
            .sort_by(|&left, &right| self.0[left].id.cmp(&self.0[right].id).then(left.cmp(&right)));
        out.word(word_len(order.len()));
        for index in order {
            self.0[index].encode(out);
        }
    }
}

impl Canonical for OffPolicyInput {
    fn encode(&self, out: &mut CanonicalWriter) {
        self.behaviour.encode(out);
        LogById(&self.trajectories).encode(out);
    }
}

impl Canonical for ComparisonSpec {
    fn encode(&self, out: &mut CanonicalWriter) {
        out.text(&self.label).text(self.kind.as_str());
        self.policy.encode(out);
    }
}

impl Canonical for VoiRequest {
    fn encode(&self, out: &mut CanonicalWriter) {
        self.collect_action.encode(out);
        // Positional: the utility table is this order, and a tie keeps the first index.
        self.commit_actions.encode(out);
    }
}

impl Canonical for SequentialRequest {
    fn encode(&self, out: &mut CanonicalWriter) {
        out.opt_text(self.target_population.as_deref()).opt_text(self.target_period.as_deref());
    }
}

impl Canonical for EvaluationLimits {
    fn encode(&self, out: &mut CanonicalWriter) {
        out.word(self.max_nodes);
        self.max_receipts.encode(out);
    }
}

impl Canonical for RolloutConfig {
    fn encode(&self, out: &mut CanonicalWriter) {
        out.word(self.seed);
        out.word(self.rollouts);
    }
}

// ---------------------------------------------------------------------------
// g-formula
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct PolicyEvaluationInput {
    model: FiniteSequentialModel,
    policy: ContingentPolicy,
    limits: EvaluationLimits,
}

fn policy_evaluation_bytes(input: &PolicyEvaluationInput) -> [u8; 32] {
    declare(G_FORMULA_TAG, |declaration| {
        declaration
            .field("model", &input.model)
            .field("policy", &input.policy)
            .field("limits", &input.limits);
    })
}

/// A compiled exact enumeration of one declared contingent policy.
#[derive(Clone, Debug)]
pub struct CompiledPolicyEvaluation {
    program: KernelProgram<PolicyEvaluationInput>,
}

impl Study {
    /// Compile `evaluate_policy` for `model`, `policy` and `limits`. Compiling executes nothing.
    ///
    /// # Errors
    ///
    /// Compiling from inside a lowered dispatch, or cancellation.
    pub fn compile_evaluate_policy(
        model: FiniteSequentialModel,
        policy: ContingentPolicy,
        limits: EvaluationLimits,
        ctx: &ExecutionContext,
    ) -> Result<CompiledPolicyEvaluation, CausalError> {
        let input = PolicyEvaluationInput { model, policy, limits };
        let canonical = policy_evaluation_bytes(&input);
        Ok(CompiledPolicyEvaluation { program: open(&G_FORMULA, input, canonical, ctx)? })
    }
}

impl CompiledPolicyEvaluation {
    /// The compiled program's plan view, including the support decision of its unit.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        self.program.view()
    }

    /// Every layer of the compiled program.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        self.program.program_layers()
    }

    /// The model the program binds.
    #[must_use]
    pub fn model(&self) -> &FiniteSequentialModel {
        &self.program.input().model
    }

    /// The policy the program binds.
    #[must_use]
    pub fn policy(&self) -> &ContingentPolicy {
        &self.program.input().policy
    }

    /// Pass the support gate, then evaluate the policy as `evaluate_policy` does.
    ///
    /// # Errors
    ///
    /// The support refusal (`not_supported`, detail `SequentialPolicyEvaluation`) when the unit
    /// is not supported, and every refusal of `evaluate_policy`.
    pub fn evaluate(&self, ctx: &ExecutionContext) -> SequentialResult<PolicyEvaluation> {
        let publication = self.program.gate(ctx).map_err(|refusal| evaluation_refusal(&refusal))?;
        let input = self.program.input();
        publication.finish(evaluate_policy(&input.model, &input.policy, input.limits, ctx))
    }
}

// ---------------------------------------------------------------------------
// Off-policy
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct OffPolicyProgram {
    model: FiniteSequentialModel,
    target: ContingentPolicy,
    behaviour: ContingentPolicy,
    log: Vec<LoggedTrajectory>,
    limits: EvaluationLimits,
}

fn off_policy_bytes(input: &OffPolicyProgram) -> [u8; 32] {
    declare(OFF_POLICY_TAG, |declaration| {
        declaration
            .field("model", &input.model)
            .field("target", &input.target)
            .field("behaviour", &input.behaviour)
            .field("log", &LogById(&input.log))
            .field("limits", &input.limits);
    })
}

/// A compiled importance-weighted evaluation of one declared log.
#[derive(Clone, Debug)]
pub struct CompiledOffPolicyEvaluation {
    program: KernelProgram<OffPolicyProgram>,
}

impl Study {
    /// Compile `evaluate_off_policy` for the model, the target, the logging policy, the log and
    /// the limits. The evidence ledger is passed at execution. Compiling executes nothing.
    ///
    /// # Errors
    ///
    /// Compiling from inside a lowered dispatch, or cancellation.
    pub fn compile_evaluate_off_policy(
        model: FiniteSequentialModel,
        target: ContingentPolicy,
        behaviour: ContingentPolicy,
        log: Vec<LoggedTrajectory>,
        limits: EvaluationLimits,
        ctx: &ExecutionContext,
    ) -> Result<CompiledOffPolicyEvaluation, CausalError> {
        let input = OffPolicyProgram { model, target, behaviour, log, limits };
        let canonical = off_policy_bytes(&input);
        Ok(CompiledOffPolicyEvaluation { program: open(&OFF_POLICY, input, canonical, ctx)? })
    }
}

impl CompiledOffPolicyEvaluation {
    /// The compiled program's plan view, including the support decision of its unit.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        self.program.view()
    }

    /// Every layer of the compiled program.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        self.program.program_layers()
    }

    /// The logged trajectories the program binds, in the order they were declared.
    #[must_use]
    pub fn log(&self) -> &[LoggedTrajectory] {
        &self.program.input().log
    }

    /// Pass the support gate, then weight the log as `evaluate_off_policy` does.
    ///
    /// `ledger` is the caller's ledger. The kernel updates it only when the evaluation succeeds.
    ///
    /// # Errors
    ///
    /// The support refusal (`not_supported`, detail `SequentialPolicyEvaluation`) when the unit
    /// is not supported, and every refusal of `evaluate_off_policy`.
    pub fn evaluate(
        &self,
        ledger: &mut EvidenceLedger,
        ctx: &ExecutionContext,
    ) -> SequentialResult<OffPolicyEstimate> {
        let publication = self.program.gate(ctx).map_err(|refusal| evaluation_refusal(&refusal))?;
        let input = self.program.input();
        publication.finish(evaluate_off_policy(
            &input.model,
            &input.target,
            &input.behaviour,
            &input.log,
            ledger,
            input.limits,
            ctx,
        ))
    }
}

// ---------------------------------------------------------------------------
// Rollout
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct RolloutProgram {
    model: FiniteSequentialModel,
    policy: ContingentPolicy,
    config: RolloutConfig,
}

fn rollout_bytes(input: &RolloutProgram) -> [u8; 32] {
    declare(ROLLOUT_TAG, |declaration| {
        declaration
            .field("model", &input.model)
            .field("policy", &input.policy)
            .field("config", &input.config);
    })
}

/// A compiled common-random-number rollout of one declared policy.
#[derive(Clone, Debug)]
pub struct CompiledPolicyRollout {
    program: KernelProgram<RolloutProgram>,
}

impl Study {
    /// Compile `rollout_policy` for `model`, `policy` and `config`. Compiling executes nothing.
    ///
    /// # Errors
    ///
    /// Compiling from inside a lowered dispatch, or cancellation.
    pub fn compile_rollout_policy(
        model: FiniteSequentialModel,
        policy: ContingentPolicy,
        config: RolloutConfig,
        ctx: &ExecutionContext,
    ) -> Result<CompiledPolicyRollout, CausalError> {
        let input = RolloutProgram { model, policy, config };
        let canonical = rollout_bytes(&input);
        Ok(CompiledPolicyRollout { program: open(&ROLLOUT, input, canonical, ctx)? })
    }
}

impl CompiledPolicyRollout {
    /// The compiled program's plan view, including the support decision of its unit.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        self.program.view()
    }

    /// Every layer of the compiled program.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        self.program.program_layers()
    }

    /// The rollout configuration the program binds.
    #[must_use]
    pub fn config(&self) -> RolloutConfig {
        self.program.input().config
    }

    /// Pass the support gate, then roll the policy out as `rollout_policy` does.
    ///
    /// # Errors
    ///
    /// The support refusal (`not_supported`, detail `SequentialPolicyEvaluation`) when the unit
    /// is not supported, and every refusal of `rollout_policy`.
    pub fn evaluate(&self, ctx: &ExecutionContext) -> SequentialResult<RolloutEstimate> {
        let publication = self.program.gate(ctx).map_err(|refusal| evaluation_refusal(&refusal))?;
        let input = self.program.input();
        publication.finish(rollout_policy(&input.model, &input.policy, input.config, ctx))
    }
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct CompareProgram {
    model: FiniteSequentialModel,
    specs: Vec<ComparisonSpec>,
    voi: Option<VoiRequest>,
    request: SequentialRequest,
    limits: EvaluationLimits,
}

fn compare_bytes(input: &CompareProgram) -> [u8; 32] {
    declare(COMPARE_TAG, |declaration| {
        declaration
            .field("model", &input.model)
            .field("specs", &input.specs)
            .field("voi", &input.voi)
            .field("request", &input.request)
            .field("limits", &input.limits);
    })
}

/// A compiled comparison of declared policies, optionally with the value of one observation.
#[derive(Clone, Debug)]
pub struct CompiledPolicyComparison {
    program: KernelProgram<CompareProgram>,
}

impl Study {
    /// Compile `compare_policies` for the model, the comparisons, the optional value-of-information
    /// request, the population request and the limits. Compiling executes nothing.
    ///
    /// # Errors
    ///
    /// Compiling from inside a lowered dispatch, or cancellation.
    pub fn compile_compare_policies(
        model: FiniteSequentialModel,
        specs: Vec<ComparisonSpec>,
        voi: Option<VoiRequest>,
        request: SequentialRequest,
        limits: EvaluationLimits,
        ctx: &ExecutionContext,
    ) -> Result<CompiledPolicyComparison, CausalError> {
        let input = CompareProgram { model, specs, voi, request, limits };
        let canonical = compare_bytes(&input);
        Ok(CompiledPolicyComparison { program: open(&COMPARE, input, canonical, ctx)? })
    }
}

impl CompiledPolicyComparison {
    /// The compiled program's plan view, including the support decision of its unit.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        self.program.view()
    }

    /// Every layer of the compiled program.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        self.program.program_layers()
    }

    /// The comparisons the program binds, in declared order. The first is the baseline.
    #[must_use]
    pub fn specs(&self) -> &[ComparisonSpec] {
        &self.program.input().specs
    }

    /// Pass the support gate, then compare the policies as `compare_policies` does.
    ///
    /// # Errors
    ///
    /// The support refusal (`not_supported`, detail `SequentialPolicyEvaluation`) when the unit
    /// is not supported, and every refusal of `compare_policies`.
    pub fn evaluate(&self, ctx: &ExecutionContext) -> SequentialResult<ComparisonReport> {
        let publication = self.program.gate(ctx).map_err(|refusal| evaluation_refusal(&refusal))?;
        let input = self.program.input();
        publication.finish(compare_policies(
            &input.model,
            input.specs.clone(),
            input.voi.as_ref(),
            &input.request,
            input.limits,
            ctx,
        ))
    }
}

// ---------------------------------------------------------------------------
// Value of collecting evidence
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct ValueOfEvidenceProgram {
    model: FiniteSequentialModel,
    request: VoiRequest,
    limits: EvaluationLimits,
}

fn value_of_evidence_bytes(input: &ValueOfEvidenceProgram) -> [u8; 32] {
    declare(VALUE_OF_EVIDENCE_TAG, |declaration| {
        declaration
            .field("model", &input.model)
            .field("request", &input.request)
            .field("limits", &input.limits);
    })
}

/// A compiled value of collecting one declared observation before committing.
#[derive(Clone, Debug)]
pub struct CompiledValueOfCollectingEvidence {
    program: KernelProgram<ValueOfEvidenceProgram>,
}

impl Study {
    /// Compile `value_of_collecting_evidence` for `model`, `request` and `limits`. Compiling
    /// executes nothing.
    ///
    /// # Errors
    ///
    /// Compiling from inside a lowered dispatch, or cancellation.
    pub fn compile_value_of_collecting_evidence(
        model: FiniteSequentialModel,
        request: VoiRequest,
        limits: EvaluationLimits,
        ctx: &ExecutionContext,
    ) -> Result<CompiledValueOfCollectingEvidence, CausalError> {
        let input = ValueOfEvidenceProgram { model, request, limits };
        let canonical = value_of_evidence_bytes(&input);
        Ok(CompiledValueOfCollectingEvidence {
            program: open(&VALUE_OF_EVIDENCE, input, canonical, ctx)?,
        })
    }
}

impl CompiledValueOfCollectingEvidence {
    /// The compiled program's plan view, including the support decision of its unit.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        self.program.view()
    }

    /// Every layer of the compiled program.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        self.program.program_layers()
    }

    /// The value-of-information request the program binds.
    #[must_use]
    pub fn request(&self) -> &VoiRequest {
        &self.program.input().request
    }

    /// Pass the support gate, then score the observation as `value_of_collecting_evidence` does.
    ///
    /// # Errors
    ///
    /// The support refusal (`not_supported`, detail `SequentialPolicyEvaluation`) when the unit
    /// is not supported, and every refusal of `value_of_collecting_evidence`.
    pub fn evaluate(&self, ctx: &ExecutionContext) -> SequentialResult<VoiCheck> {
        let publication = self.program.gate(ctx).map_err(|refusal| evaluation_refusal(&refusal))?;
        let input = self.program.input();
        publication.finish(value_of_collecting_evidence(
            &input.model,
            &input.request,
            input.limits,
            ctx,
        ))
    }
}

// ---------------------------------------------------------------------------
// Seal
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct SealProgram {
    model: FiniteSequentialModel,
    policy: ContingentPolicy,
    off_policy: Option<OffPolicyInput>,
    rollout: Option<RolloutConfig>,
    limits: EvaluationLimits,
}

fn seal_bytes(input: &SealProgram) -> [u8; 32] {
    declare(SEAL_TAG, |declaration| {
        declaration
            .field("model", &input.model)
            .field("policy", &input.policy)
            .field("off_policy", &input.off_policy)
            .field("rollout", &input.rollout)
            .field("limits", &input.limits);
    })
}

/// A compiled seal of one declared sequential evaluation.
#[derive(Clone, Debug)]
pub struct CompiledSequentialEvaluationSeal {
    program: KernelProgram<SealProgram>,
}

impl Study {
    /// Compile the seal of `model` and `policy`, with the optional off-policy input, the optional
    /// rollout and the limits that [`SequentialEvaluationArtifact::new`] receives. Compiling
    /// executes nothing.
    ///
    /// # Errors
    ///
    /// Compiling from inside a lowered dispatch, or cancellation.
    pub fn compile_sequential_evaluation_seal(
        model: FiniteSequentialModel,
        policy: ContingentPolicy,
        off_policy: Option<OffPolicyInput>,
        rollout: Option<RolloutConfig>,
        limits: EvaluationLimits,
        ctx: &ExecutionContext,
    ) -> Result<CompiledSequentialEvaluationSeal, CausalError> {
        let input = SealProgram { model, policy, off_policy, rollout, limits };
        let canonical = seal_bytes(&input);
        Ok(CompiledSequentialEvaluationSeal { program: open(&SEAL, input, canonical, ctx)? })
    }
}

impl CompiledSequentialEvaluationSeal {
    /// The compiled program's plan view, including the support decision of its unit.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        self.program.view()
    }

    /// Every layer of the compiled program.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        self.program.program_layers()
    }

    /// Pass the support gate, then seal as [`SequentialEvaluationArtifact::new`] does.
    ///
    /// # Errors
    ///
    /// The support refusal (`not_supported`) when the unit is not supported, and every error of
    /// [`SequentialEvaluationArtifact::new`].
    pub fn seal(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<SequentialEvaluationArtifact, antecedent_io::IoError> {
        let publication = self.program.gate(ctx).map_err(|refusal| super::io_refusal(&refusal))?;
        let input = self.program.input();
        publication.finish(SequentialEvaluationArtifact::new(
            &input.model,
            &input.policy,
            input.off_policy.as_ref(),
            input.rollout,
            input.limits,
            ctx,
        ))
    }
}

#[cfg(test)]
mod site_tests {
    use antecedent_design::sequential_evaluation::{
        ActionRole, CensoringRule, ComparisonKind, HistoryKey, HistoryStep,
    };

    use super::{
        ActionSpec, ActionSupport, ComparisonSpec, ContingentPolicy, EvaluationLimits,
        FiniteSequentialModel, LoggedStep, LoggedTrajectory, OffPolicyInput, PolicySpec,
        RolloutConfig, SequentialRequest, StoppingBoundary, VoiRequest,
    };
    use super::{
        CompareProgram, OffPolicyProgram, PolicyEvaluationInput, RolloutProgram, SealProgram,
        ValueOfEvidenceProgram, compare_bytes, off_policy_bytes, policy_evaluation_bytes,
        rollout_bytes, seal_bytes, value_of_evidence_bytes,
    };
    use crate::identity_laws::{bit_exact, ordered_laws, set_laws};

    fn inspection(labels: &[&str], boundary: &[usize], utility: f64) -> FiniteSequentialModel {
        let actions = labels
            .iter()
            .map(|label| ActionSpec {
                label: (*label).to_owned(),
                role: ActionRole::Commit,
                utility_cost: 0.0,
                resource_use: 0.0,
                transition: vec![1.0, 0.0, 0.0, 1.0],
                channel: None,
                immediate_utility: vec![utility, -utility],
                delayed: None,
            })
            .collect();
        let stopping = if boundary.is_empty() {
            Vec::new()
        } else {
            vec![StoppingBoundary {
                label: "stop".into(),
                states: boundary.to_vec(),
                upper: Some(0.9),
                lower: None,
                stop_action: 0,
            }]
        };
        FiniteSequentialModel {
            label: "display".into(),
            population: "plant_a".into(),
            regime: "operations".into(),
            period: "2026".into(),
            state_variable: "unit_condition".into(),
            states: vec!["sound".into(), "faulty".into()],
            initial: vec![0.5, 0.5],
            horizon: 2,
            actions,
            terminal_utility: vec![0.0, 0.0],
            censoring: CensoringRule::Refuse,
            resource_budget: None,
            stopping,
            support: ActionSupport::everywhere(labels.len()),
        }
    }

    fn root_policy() -> ContingentPolicy {
        ContingentPolicy::new("v").decide(HistoryKey::root(), 0)
    }

    fn distinct_states(states: &[usize]) -> bool {
        if states.is_empty() || states.iter().any(|state| *state >= 2) {
            return false;
        }
        let mut sorted = states.to_vec();
        sorted.sort_unstable();
        sorted.windows(2).all(|pair| pair[0] != pair[1])
    }

    fn g_bytes(model: &FiniteSequentialModel, policy: &ContingentPolicy) -> [u8; 32] {
        policy_evaluation_bytes(&PolicyEvaluationInput {
            model: model.clone(),
            policy: policy.clone(),
            limits: EvaluationLimits::default(),
        })
    }

    /// Boundary states are a set, actions and support flags are ordered, and the display label
    /// is not part of the digest. `identity` is one operation's bytes of a model.
    fn model_laws(identity: impl Fn(&FiniteSequentialModel) -> [u8; 32]) {
        set_laws(&[0_usize, 1], |states| {
            if !distinct_states(states) {
                return None;
            }
            Some(identity(&inspection(&["ship", "repair"], states, 0.0)))
        });
        ordered_laws(&["ship", "repair"], |labels| Some(identity(&inspection(labels, &[], 0.0))));
        bit_exact(|utility| Some(identity(&inspection(&["ship"], &[], utility))));
        let mut labeled = inspection(&["ship"], &[], 1.0);
        let digest = identity(&labeled);
        labeled.label = "renamed".into();
        assert_eq!(digest, identity(&labeled), "the display label is not part of the identity");
    }

    fn unique_ids(log: &[LoggedTrajectory]) -> bool {
        let mut ids: Vec<&str> = log.iter().map(|trajectory| trajectory.id.as_str()).collect();
        ids.sort_unstable();
        ids.windows(2).all(|pair| pair[0] != pair[1])
    }

    #[test]
    fn site_sequential_policy_evaluation_boundaries_are_a_set_and_actions_are_ordered() {
        model_laws(|model| g_bytes(model, &root_policy()));
        set_laws(
            &[Choice { extended: false, action: 0 }, Choice { extended: true, action: 1 }],
            |choices| {
                let mut policy = ContingentPolicy::new("v");
                for choice in choices {
                    let history = if choice.extended {
                        HistoryKey::root().extended(0, 0)
                    } else {
                        HistoryKey::root()
                    };
                    policy = policy.decide(history, choice.action);
                }
                Some(g_bytes(&inspection(&["ship", "repair"], &[], 0.0), &policy))
            },
        );
        ordered_laws(
            &[HistoryStep { action: 0, observation: 0 }, HistoryStep { action: 1, observation: 0 }],
            |steps| {
                let policy = ContingentPolicy::new("v")
                    .decide(HistoryKey::from_steps(steps.iter().copied()), 0);
                Some(g_bytes(&inspection(&["ship", "repair"], &[], 0.0), &policy))
            },
        );
        set_laws(&[(0_usize, 0.25_f64), (1_usize, 0.75)], |distribution| {
            let policy = ContingentPolicy::new("v")
                .decide_stochastic(HistoryKey::root(), distribution.to_vec())
                .ok()?;
            Some(g_bytes(&inspection(&["ship", "repair"], &[], 0.0), &policy))
        });
        set_laws(&[HistoryKey::root(), HistoryKey::root().extended(0, 0)], |histories| {
            let mut support = ActionSupport::everywhere(2);
            for history in histories {
                support = support.with_history(history.clone(), vec![true, false]);
            }
            let mut model = inspection(&["ship", "repair"], &[], 0.0);
            model.support = support;
            Some(g_bytes(&model, &root_policy()))
        });
        ordered_laws(&[true, false], |flags| {
            let mut model = inspection(&["ship", "repair"], &[], 0.0);
            model.support =
                ActionSupport::everywhere(2).with_history(HistoryKey::root(), flags.to_vec());
            Some(g_bytes(&model, &root_policy()))
        });
    }

    #[derive(Clone, Debug, PartialEq)]
    struct Choice {
        extended: bool,
        action: usize,
    }

    fn off_input(log: Vec<LoggedTrajectory>) -> OffPolicyProgram {
        OffPolicyProgram {
            model: inspection(&["ship"], &[], 0.0),
            target: root_policy(),
            behaviour: root_policy(),
            log,
            limits: EvaluationLimits::default(),
        }
    }

    #[test]
    fn site_sequential_off_policy_log_is_a_set_and_steps_are_ordered() {
        model_laws(|model| {
            off_policy_bytes(&OffPolicyProgram {
                model: model.clone(),
                target: root_policy(),
                behaviour: root_policy(),
                log: Vec::new(),
                limits: EvaluationLimits::default(),
            })
        });
        set_laws(
            &[
                LoggedTrajectory {
                    id: "a".into(),
                    steps: vec![LoggedStep {
                        action: 0,
                        observation: None,
                        assignment_probability: 1.0,
                    }],
                    outcome_utility: 1.0,
                },
                LoggedTrajectory {
                    id: "b".into(),
                    steps: vec![LoggedStep {
                        action: 1,
                        observation: None,
                        assignment_probability: 1.0,
                    }],
                    outcome_utility: -1.0,
                },
            ],
            |log| {
                if !unique_ids(log) {
                    return None;
                }
                Some(off_policy_bytes(&off_input(log.to_vec())))
            },
        );
        ordered_laws(
            &[
                LoggedStep { action: 0, observation: None, assignment_probability: 1.0 },
                LoggedStep { action: 1, observation: Some(0), assignment_probability: 0.5 },
            ],
            |steps| {
                Some(off_policy_bytes(&off_input(vec![LoggedTrajectory {
                    id: "a".into(),
                    steps: steps.to_vec(),
                    outcome_utility: 0.0,
                }])))
            },
        );
    }

    #[test]
    fn site_sequential_rollout_boundaries_are_a_set_and_actions_are_ordered() {
        model_laws(|model| {
            rollout_bytes(&RolloutProgram {
                model: model.clone(),
                policy: root_policy(),
                config: RolloutConfig { seed: 11, rollouts: 4 },
            })
        });
    }

    fn comparison(label: &str) -> ComparisonSpec {
        ComparisonSpec {
            label: label.to_owned(),
            kind: ComparisonKind::ActNow,
            policy: PolicySpec::Contingent(
                ContingentPolicy::new(label).decide(HistoryKey::root(), 0),
            ),
        }
    }

    fn compare_input(model: FiniteSequentialModel, specs: Vec<ComparisonSpec>) -> CompareProgram {
        CompareProgram {
            model,
            specs,
            voi: None,
            request: SequentialRequest::default(),
            limits: EvaluationLimits::default(),
        }
    }

    #[test]
    fn site_sequential_comparisons_are_ordered_and_boundaries_are_a_set() {
        model_laws(|model| compare_bytes(&compare_input(model.clone(), vec![comparison("act")])));
        ordered_laws(&["act", "wait"], |labels| {
            Some(compare_bytes(&compare_input(
                inspection(&["ship"], &[], 0.0),
                labels.iter().map(|label| comparison(label)).collect(),
            )))
        });
        ordered_laws(&[1.0_f64, 2.0], |scores| {
            let spec = ComparisonSpec {
                label: "table".into(),
                kind: ComparisonKind::ActNow,
                policy: PolicySpec::ReweightedStaticScores {
                    scores: scores.to_vec(),
                    weights: vec![1.0, 1.0],
                },
            };
            Some(compare_bytes(&compare_input(inspection(&["ship"], &[], 0.0), vec![spec])))
        });
        ordered_laws(&[0_usize, 1], |actions| {
            let mut input =
                compare_input(inspection(&["ship", "repair"], &[], 0.0), vec![comparison("act")]);
            input.voi = Some(VoiRequest { collect_action: 0, commit_actions: actions.to_vec() });
            Some(compare_bytes(&input))
        });
        let learned = |rule: &str| {
            compare_bytes(&compare_input(
                inspection(&["ship"], &[], 0.0),
                vec![ComparisonSpec {
                    label: "learned".into(),
                    kind: ComparisonKind::ActNow,
                    policy: PolicySpec::LearnedFromEvaluationData { rule: rule.to_owned() },
                }],
            ))
        };
        assert_ne!(learned("fit"), learned("other"));
    }

    #[test]
    fn site_sequential_commit_actions_are_ordered_and_boundaries_are_a_set() {
        let bytes = |model: &FiniteSequentialModel, actions: &[usize]| {
            value_of_evidence_bytes(&ValueOfEvidenceProgram {
                model: model.clone(),
                request: VoiRequest { collect_action: 0, commit_actions: actions.to_vec() },
                limits: EvaluationLimits::default(),
            })
        };
        model_laws(|model| bytes(model, &[0]));
        ordered_laws(&[0_usize, 1], |actions| {
            Some(bytes(&inspection(&["ship", "repair"], &[], 0.0), actions))
        });
    }

    #[test]
    fn site_sequential_seal_log_is_a_set_and_actions_are_ordered() {
        model_laws(|model| {
            seal_bytes(&SealProgram {
                model: model.clone(),
                policy: root_policy(),
                off_policy: None,
                rollout: None,
                limits: EvaluationLimits::default(),
            })
        });
        set_laws(
            &[
                LoggedTrajectory {
                    id: "a".into(),
                    steps: vec![LoggedStep {
                        action: 0,
                        observation: None,
                        assignment_probability: 1.0,
                    }],
                    outcome_utility: 1.0,
                },
                LoggedTrajectory {
                    id: "b".into(),
                    steps: vec![LoggedStep {
                        action: 0,
                        observation: None,
                        assignment_probability: 1.0,
                    }],
                    outcome_utility: -1.0,
                },
            ],
            |log| {
                if !unique_ids(log) {
                    return None;
                }
                Some(seal_bytes(&SealProgram {
                    model: inspection(&["ship"], &[], 0.0),
                    policy: root_policy(),
                    off_policy: Some(OffPolicyInput {
                        behaviour: root_policy(),
                        trajectories: log.to_vec(),
                    }),
                    rollout: Some(RolloutConfig { seed: 11, rollouts: 4 }),
                    limits: EvaluationLimits::default(),
                }))
            },
        );
    }
}
