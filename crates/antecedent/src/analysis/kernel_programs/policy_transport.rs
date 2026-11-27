//! The three policy-transport evaluators as compiled program families.
//!
//! `evaluate_transported_policy` and `evaluate_temporal_policy` take the caller's context;
//! `transport_adaptive_trial` does not. Each compile binds an owned copy of the borrowed
//! request (the kernels borrow the model, policy, contract and declarations) and uses the
//! context for the support gate. Execute rebuilds that request and calls the same kernel.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::DataClassification;
use antecedent_contract::adaptive_design::AdaptiveDesignContract;
use antecedent_core::measurement_model::MeasurementModel;
use antecedent_core::{CanonicalWriter, ExecutionContext, NotSupported, SemanticDigest};
use antecedent_design::adaptive_trial_transport::{
    AdaptiveTransportEstimate, AdaptiveTransportRequest, TrialMechanism, TrialSelectionDiagram,
    transport_adaptive_trial,
};
use antecedent_design::sequential_evaluation::{
    ActionRole, ActionSpec, CensoringRule, ContingentPolicy, DelayedOutcome, EvaluationLimits,
    FiniteSequentialModel, HistoryKey, SequentialEvaluationError, StoppingBoundary,
};
use antecedent_design::temporal_policy_transport::{
    ActionAlphabet, HistoryAlphabet, TemporalPolicyEvaluation, TemporalPolicyRequest,
    TemporalTarget, TimeInvariance, evaluate_temporal_policy,
};
use antecedent_design::transported_policy::{
    ActionUtility, MechanismStatus, ModelMechanism, SelectionDiagram, TargetUtility,
    TransportedPolicyEvaluation, TransportedPolicyRequest, evaluate_transported_policy,
};

use super::{KernelProgram, KernelProgramShape, KernelRealization};
use crate::analysis::Study;
use crate::analysis::supplied_identity::{
    Canonical, Declaration, SetOf, encoded, unencodable, variant,
};
use crate::error::CausalError;
use crate::planner::PlanView;
use crate::planner::lowered::{
    EvidenceRole, FunctionalKind, LoweredProgram, ProgramCell, UncertaintyReading,
};

const TAG_POLICY: &str = "antecedent.kernel_program.transported_policy";
const TAG_TEMPORAL: &str = "antecedent.kernel_program.temporal_policy";
const TAG_ADAPTIVE: &str = "antecedent.kernel_program.adaptive_trial_transport";

const POLICY: KernelProgramShape = KernelProgramShape {
    cell: ProgramCell::TransportedPolicyEvaluation,
    plan_id: "transport.policy_evaluation",
    functional: FunctionalKind::PolicyValue,
    evaluator: "transported_policy_evaluation",
    realization: KernelRealization::Exact,
    uncertainty: UncertaintyReading::None,
    role: EvidenceRole::TransportSource,
    graph_class: "finite_sequential_selection_diagram",
    data: DataClassification::MultiEnvironment,
    design: "exact g-formula of a contingent policy under a declared selection diagram",
};

const TEMPORAL: KernelProgramShape = KernelProgramShape {
    cell: ProgramCell::TemporalPolicyEvaluation,
    plan_id: "transport.period_shift_policy",
    functional: FunctionalKind::PolicyValue,
    evaluator: "temporal_policy_evaluation",
    realization: KernelRealization::Exact,
    uncertainty: UncertaintyReading::None,
    role: EvidenceRole::TransportSource,
    graph_class: "finite_sequential_time_invariance",
    data: DataClassification::MultiEnvironment,
    design: "exact g-formula of a contingent policy under a declared time invariance",
};

const ADAPTIVE: KernelProgramShape = KernelProgramShape {
    cell: ProgramCell::AdaptiveTrialTransport,
    plan_id: "transport.adaptive_trial_source",
    functional: FunctionalKind::AssignmentContrast,
    evaluator: "adaptive_trial_transport",
    realization: KernelRealization::Exact,
    uncertainty: UncertaintyReading::None,
    role: EvidenceRole::TransportSource,
    graph_class: "trial_selection_diagram",
    data: DataClassification::MultiEnvironment,
    design: "Hajek arm means of an adaptive trial standardized to a target stratum law",
};

/// Owned copy of a [`TransportedPolicyRequest`]. The kernel borrows; the handle keeps what it
/// needs to borrow again.
#[derive(Clone, Debug)]
struct OwnedPolicy {
    source: FiniteSequentialModel,
    policy: ContingentPolicy,
    target_population: String,
    target_initial: Option<Vec<f64>>,
    diagram: Option<SelectionDiagram>,
    target_utility: Option<TargetUtility>,
    limits: EvaluationLimits,
}

/// Owned copy of a [`TemporalPolicyRequest`].
#[derive(Clone, Debug)]
struct OwnedTemporal {
    source: FiniteSequentialModel,
    policy: ContingentPolicy,
    target: TemporalTarget,
    target_initial: Option<Vec<f64>>,
    invariance: Option<TimeInvariance>,
    limits: EvaluationLimits,
}

/// Owned copy of an [`AdaptiveTransportRequest`]. The contract is stored whole so execute can
/// pass the same contract the kernel borrows; its digest is [`AdaptiveDesignContract::identity`].
#[derive(Clone, Debug)]
struct OwnedTrial {
    contract: Option<AdaptiveDesignContract>,
    trial_population: String,
    unit_strata: Vec<usize>,
    stratum_labels: Vec<String>,
    target_stratum_law: Option<Vec<f64>>,
    diagram: Option<TrialSelectionDiagram>,
    treatment_arm: usize,
    control_arm: usize,
}

impl OwnedPolicy {
    fn from_request(request: &TransportedPolicyRequest<'_>) -> Self {
        Self {
            source: request.source.clone(),
            policy: request.policy.clone(),
            target_population: request.target_population.to_owned(),
            target_initial: request.target_initial.map(ToOwned::to_owned),
            diagram: request.diagram.cloned(),
            target_utility: request.target_utility.cloned(),
            limits: request.limits,
        }
    }

    fn request(&self) -> TransportedPolicyRequest<'_> {
        TransportedPolicyRequest {
            source: &self.source,
            policy: &self.policy,
            target_population: &self.target_population,
            target_initial: self.target_initial.as_deref(),
            diagram: self.diagram.as_ref(),
            target_utility: self.target_utility.as_ref(),
            limits: self.limits,
        }
    }
}

impl OwnedTemporal {
    fn from_request(request: &TemporalPolicyRequest<'_>) -> Self {
        Self {
            source: request.source.clone(),
            policy: request.policy.clone(),
            target: request.target.clone(),
            target_initial: request.target_initial.map(ToOwned::to_owned),
            invariance: request.invariance.cloned(),
            limits: request.limits,
        }
    }

    fn request(&self) -> TemporalPolicyRequest<'_> {
        TemporalPolicyRequest {
            source: &self.source,
            policy: &self.policy,
            target: &self.target,
            target_initial: self.target_initial.as_deref(),
            invariance: self.invariance.as_ref(),
            limits: self.limits,
        }
    }
}

impl OwnedTrial {
    fn from_request(request: &AdaptiveTransportRequest<'_>) -> Self {
        Self {
            contract: request.contract.cloned(),
            trial_population: request.trial_population.to_owned(),
            unit_strata: request.unit_strata.to_vec(),
            stratum_labels: request.stratum_labels.to_vec(),
            target_stratum_law: request.target_stratum_law.map(ToOwned::to_owned),
            diagram: request.diagram.cloned(),
            treatment_arm: request.treatment_arm,
            control_arm: request.control_arm,
        }
    }

    fn request(&self) -> AdaptiveTransportRequest<'_> {
        AdaptiveTransportRequest {
            contract: self.contract.as_ref(),
            trial_population: &self.trial_population,
            unit_strata: &self.unit_strata,
            stratum_labels: &self.stratum_labels,
            target_stratum_law: self.target_stratum_law.as_deref(),
            diagram: self.diagram.as_ref(),
            treatment_arm: self.treatment_arm,
            control_arm: self.control_arm,
        }
    }
}

fn encode_role(role: ActionRole, out: &mut CanonicalWriter) {
    match role {
        ActionRole::Continue => {
            variant(out, "continue");
        }
        ActionRole::Commit => {
            variant(out, "commit");
        }
        _ => unencodable(out, "action role"),
    }
}

fn encode_censoring(rule: CensoringRule, out: &mut CanonicalWriter) {
    match rule {
        CensoringRule::Refuse => {
            variant(out, "refuse");
        }
        CensoringRule::Forfeit => {
            variant(out, "forfeit");
        }
        _ => unencodable(out, "censoring rule"),
    }
}

fn encode_status(status: MechanismStatus, out: &mut CanonicalWriter) {
    match status {
        MechanismStatus::Invariant => {
            variant(out, "invariant");
        }
        MechanismStatus::Shifted => {
            variant(out, "shifted");
        }
        _ => unencodable(out, "mechanism status"),
    }
}

fn encode_model_mechanism(mechanism: ModelMechanism, out: &mut CanonicalWriter) {
    match mechanism {
        ModelMechanism::InitialState => {
            variant(out, "initial_state");
        }
        ModelMechanism::Transition { action } => {
            variant(out, "transition");
            action.encode(out);
        }
        ModelMechanism::Observation { action } => {
            variant(out, "observation");
            action.encode(out);
        }
        ModelMechanism::Utility => {
            variant(out, "utility");
        }
        _ => unencodable(out, "model mechanism"),
    }
}

fn encode_trial_mechanism(mechanism: TrialMechanism, out: &mut CanonicalWriter) {
    match mechanism {
        TrialMechanism::StratumLaw => {
            variant(out, "stratum_law");
        }
        TrialMechanism::OutcomeGivenStratumAndArm => {
            variant(out, "outcome_given_stratum_and_arm");
        }
        _ => unencodable(out, "trial mechanism"),
    }
}

fn encode_channel(model: &MeasurementModel, out: &mut CanonicalWriter) {
    let identity: SemanticDigest = model.identities().model;
    identity.encode(out);
}

fn encode_delayed(delayed: &DelayedOutcome, out: &mut CanonicalWriter) {
    delayed.delay.encode(out);
    delayed.utility.encode(out);
}

fn encode_action(action: &ActionSpec, out: &mut CanonicalWriter) {
    action.label.encode(out);
    encode_role(action.role, out);
    action.utility_cost.encode(out);
    action.resource_use.encode(out);
    // Row-major, current state then next state: the g-formula indexes it.
    action.transition.encode(out);
    match &action.channel {
        Some(channel) => {
            out.presence(true);
            encode_channel(&channel.measurement, out);
        }
        None => {
            out.presence(false);
        }
    }
    action.immediate_utility.encode(out);
    match &action.delayed {
        Some(delayed) => {
            out.presence(true);
            encode_delayed(delayed, out);
        }
        None => {
            out.presence(false);
        }
    }
}

fn encode_boundary(boundary: &StoppingBoundary, out: &mut CanonicalWriter) {
    boundary.label.encode(out);
    // The mass is a sum, so order is irrelevant, and a repeated state changes the sum.
    // The model's own identity sorts without dropping a repeat.
    let mut states = boundary.states.clone();
    states.sort_unstable();
    states.encode(out);
    boundary.upper.encode(out);
    boundary.lower.encode(out);
    boundary.stop_action.encode(out);
}

fn encode_history(history: &HistoryKey, out: &mut CanonicalWriter) {
    out.word(history.steps().len() as u64);
    for step in history.steps() {
        step.action.encode(out);
        step.observation.encode(out);
    }
}

/// The model, minus its display label, which neither the g-formula nor the model's identity reads.
fn encode_model(model: &FiniteSequentialModel, out: &mut CanonicalWriter) {
    model.population.encode(out);
    model.regime.encode(out);
    model.period.encode(out);
    model.state_variable.encode(out);
    model.states.encode(out);
    model.initial.encode(out);
    model.horizon.encode(out);
    model.terminal_utility.encode(out);
    encode_censoring(model.censoring, out);
    model.resource_budget.encode(out);
    out.word(model.actions.len() as u64);
    for action in &model.actions {
        encode_action(action, out);
    }
    // Boundaries are checked in the order they are declared.
    out.word(model.stopping.len() as u64);
    for boundary in &model.stopping {
        encode_boundary(boundary, out);
    }
    model.support.default_allowed().encode(out);
    let overrides: Vec<Override<'_>> =
        model.support.overrides().map(|(history, allowed)| Override { history, allowed }).collect();
    SetOf(&overrides).encode(out);
}

/// One policy decision. The distribution is stored sorted by action and read in that order.
struct Decision<'a> {
    history: &'a HistoryKey,
    distribution: &'a [(usize, f64)],
}

impl Canonical for Decision<'_> {
    fn encode(&self, out: &mut CanonicalWriter) {
        encode_history(self.history, out);
        self.distribution.encode(out);
    }
}

/// One support override. Histories are a map.
struct Override<'a> {
    history: &'a HistoryKey,
    allowed: &'a [bool],
}

impl Canonical for Override<'_> {
    fn encode(&self, out: &mut CanonicalWriter) {
        encode_history(self.history, out);
        self.allowed.encode(out);
    }
}

/// Decisions are a map from history to a distribution.
fn encode_policy(policy: &ContingentPolicy, out: &mut CanonicalWriter) {
    policy.version().encode(out);
    let decisions: Vec<Decision<'_>> = policy
        .entries()
        .map(|(history, distribution)| Decision { history, distribution })
        .collect();
    SetOf(&decisions).encode(out);
}

fn encode_mechanism_rows(rows: &[(ModelMechanism, MechanismStatus)], out: &mut CanonicalWriter) {
    // The diagram and the invariance identity sort these rows and keep a duplicate.
    let mut rows = rows.to_vec();
    rows.sort();
    out.word(rows.len() as u64);
    for (mechanism, status) in rows {
        encode_model_mechanism(mechanism, out);
        encode_status(status, out);
    }
}

fn encode_utility(utility: &TargetUtility, out: &mut CanonicalWriter) {
    utility.provenance.source.encode(out);
    utility.provenance.digest.encode(out);
    utility.terminal.encode(out);
    // One entry per source action, in the source's action order.
    out.word(utility.actions.len() as u64);
    for action in &utility.actions {
        encode_action_utility(action, out);
    }
}

fn encode_action_utility(action: &ActionUtility, out: &mut CanonicalWriter) {
    action.cost.encode(out);
    action.immediate.encode(out);
    action.delayed.encode(out);
}

fn encode_limits(limits: EvaluationLimits, out: &mut CanonicalWriter) {
    limits.max_nodes.encode(out);
    limits.max_receipts.encode(out);
}

fn encode_alphabet(alphabet: &HistoryAlphabet, out: &mut CanonicalWriter) {
    // Actions are in action-index order; symbols are in symbol-index order.
    out.word(alphabet.actions.len() as u64);
    for action in &alphabet.actions {
        encode_alphabet_action(action, out);
    }
}

fn encode_alphabet_action(action: &ActionAlphabet, out: &mut CanonicalWriter) {
    action.label.encode(out);
    action.symbols.encode(out);
}

impl Canonical for OwnedPolicy {
    fn encode(&self, out: &mut CanonicalWriter) {
        encode_model(&self.source, out);
        encode_policy(&self.policy, out);
        self.target_population.encode(out);
        self.target_initial.encode(out);
        match &self.diagram {
            Some(diagram) => {
                out.presence(true);
                diagram.source_population.encode(out);
                diagram.target_population.encode(out);
                encode_mechanism_rows(&diagram.mechanisms, out);
            }
            None => {
                out.presence(false);
            }
        }
        match &self.target_utility {
            Some(utility) => {
                out.presence(true);
                encode_utility(utility, out);
            }
            None => {
                out.presence(false);
            }
        }
        encode_limits(self.limits, out);
    }
}

impl Canonical for OwnedTemporal {
    fn encode(&self, out: &mut CanonicalWriter) {
        encode_model(&self.source, out);
        encode_policy(&self.policy, out);
        self.target.period.encode(out);
        self.target.horizon.encode(out);
        encode_alphabet(&self.target.alphabet, out);
        self.target_initial.encode(out);
        match &self.invariance {
            Some(invariance) => {
                out.presence(true);
                invariance.source_period.encode(out);
                invariance.target_period.encode(out);
                invariance.horizon.encode(out);
                encode_mechanism_rows(&invariance.mechanisms, out);
            }
            None => {
                out.presence(false);
            }
        }
        encode_limits(self.limits, out);
    }
}

impl Canonical for OwnedTrial {
    fn encode(&self, out: &mut CanonicalWriter) {
        match &self.contract {
            Some(contract) => {
                out.presence(true);
                contract.identity().encode(out);
            }
            None => {
                out.presence(false);
            }
        }
        self.trial_population.encode(out);
        // Enrollment order: index i is unit i.
        self.unit_strata.encode(out);
        // Index i is stratum i.
        self.stratum_labels.encode(out);
        self.target_stratum_law.encode(out);
        match &self.diagram {
            Some(diagram) => {
                out.presence(true);
                diagram.source_population.encode(out);
                diagram.target_population.encode(out);
                let mut rows = diagram.mechanisms.clone();
                rows.sort();
                out.word(rows.len() as u64);
                for (mechanism, status) in rows {
                    encode_trial_mechanism(mechanism, out);
                    encode_status(status, out);
                }
            }
            None => {
                out.presence(false);
            }
        }
        self.treatment_arm.encode(out);
        self.control_arm.encode(out);
    }
}

fn declare<T: Canonical>(tag: &str, plan: &str, input: &T) -> Result<[u8; 32], CausalError> {
    encoded(plan, || {
        let mut declaration = Declaration::new(tag);
        declaration.field("request", input);
        declaration.finish()
    })
}

fn policy_bytes(input: &OwnedPolicy) -> Result<[u8; 32], CausalError> {
    declare(TAG_POLICY, POLICY.plan_id, input)
}

fn temporal_bytes(input: &OwnedTemporal) -> Result<[u8; 32], CausalError> {
    declare(TAG_TEMPORAL, TEMPORAL.plan_id, input)
}

fn adaptive_bytes(input: &OwnedTrial) -> Result<[u8; 32], CausalError> {
    declare(TAG_ADAPTIVE, ADAPTIVE.plan_id, input)
}

fn support(refusal: &NotSupported) -> SequentialEvaluationError {
    SequentialEvaluationError {
        code: refusal.code(),
        detail: refusal.detail.to_owned(),
        message: refusal.message().to_owned(),
        remedy: "support this transport before evaluating it",
    }
}

/// A compiled evaluation of a contingent policy in another population.
#[derive(Clone, Debug)]
pub struct CompiledTransportedPolicy {
    program: KernelProgram<OwnedPolicy>,
}

impl CompiledTransportedPolicy {
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

    /// Pass the support gate, then call `evaluate_transported_policy` with the bound request.
    ///
    /// # Errors
    /// The support refusal, and every refusal of `evaluate_transported_policy`.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<TransportedPolicyEvaluation, SequentialEvaluationError> {
        self.program.gate(ctx).map_err(|refusal| support(&refusal))?;
        evaluate_transported_policy(&self.program.input().request(), ctx)
    }
}

/// A compiled evaluation of a contingent policy in another period.
#[derive(Clone, Debug)]
pub struct CompiledTemporalPolicy {
    program: KernelProgram<OwnedTemporal>,
}

impl CompiledTemporalPolicy {
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

    /// Pass the support gate, then call `evaluate_temporal_policy` with the bound request.
    ///
    /// # Errors
    /// The support refusal, and every refusal of `evaluate_temporal_policy`.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<TemporalPolicyEvaluation, SequentialEvaluationError> {
        self.program.gate(ctx).map_err(|refusal| support(&refusal))?;
        evaluate_temporal_policy(&self.program.input().request(), ctx)
    }
}

/// A compiled transport of an adaptive trial's arm means.
#[derive(Clone, Debug)]
pub struct CompiledAdaptiveTrialTransport {
    program: KernelProgram<OwnedTrial>,
}

impl CompiledAdaptiveTrialTransport {
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

    /// Pass the support gate, then call `transport_adaptive_trial` with the bound request.
    /// The kernel takes no context.
    ///
    /// # Errors
    /// The support refusal, and every refusal of `transport_adaptive_trial`.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<AdaptiveTransportEstimate, SequentialEvaluationError> {
        self.program.gate(ctx).map_err(|refusal| support(&refusal))?;
        transport_adaptive_trial(&self.program.input().request())
    }
}

impl Study {
    /// Compile `evaluate_transported_policy` for this request. The request is borrowed for the
    /// call; the program owns the copies execute borrows again. Compiling executes nothing.
    ///
    /// # Errors
    /// Cancellation, compiling from inside a lowered dispatch, or a declaration with no encoding.
    pub fn compile_transported_policy(
        request: &TransportedPolicyRequest<'_>,
        ctx: &ExecutionContext,
    ) -> Result<CompiledTransportedPolicy, CausalError> {
        let owned = OwnedPolicy::from_request(request);
        let bytes = policy_bytes(&owned)?;
        let program = KernelProgram::compile(&POLICY, owned, &bytes, ctx)?;
        Ok(CompiledTransportedPolicy { program })
    }

    /// Compile `evaluate_temporal_policy` for this request.
    ///
    /// # Errors
    /// Cancellation, compiling from inside a lowered dispatch, or a declaration with no encoding.
    pub fn compile_temporal_policy(
        request: &TemporalPolicyRequest<'_>,
        ctx: &ExecutionContext,
    ) -> Result<CompiledTemporalPolicy, CausalError> {
        let owned = OwnedTemporal::from_request(request);
        let bytes = temporal_bytes(&owned)?;
        let program = KernelProgram::compile(&TEMPORAL, owned, &bytes, ctx)?;
        Ok(CompiledTemporalPolicy { program })
    }

    /// Compile `transport_adaptive_trial` for this request. The context is the gate's; execute
    /// calls the kernel with the request and no context.
    ///
    /// # Errors
    /// Cancellation, compiling from inside a lowered dispatch, or a declaration with no encoding.
    pub fn compile_adaptive_trial(
        request: &AdaptiveTransportRequest<'_>,
        ctx: &ExecutionContext,
    ) -> Result<CompiledAdaptiveTrialTransport, CausalError> {
        let owned = OwnedTrial::from_request(request);
        let bytes = adaptive_bytes(&owned)?;
        let program = KernelProgram::compile(&ADAPTIVE, owned, &bytes, ctx)?;
        Ok(CompiledAdaptiveTrialTransport { program })
    }
}

#[cfg(test)]
mod site_tests {
    use super::*;
    use crate::identity_laws::{
        ADVERSARIAL_TEXT, bit_exact, injective, multiset_laws, ordered_laws, set_laws,
    };
    use antecedent_design::sequential_evaluation::{ActionSupport, HistoryStep};

    fn model(population: &str, regime: &str) -> FiniteSequentialModel {
        FiniteSequentialModel {
            label: "ignored".to_owned(),
            population: population.to_owned(),
            regime: regime.to_owned(),
            period: "2026".to_owned(),
            state_variable: "state".to_owned(),
            states: vec!["sound".to_owned(), "faulty".to_owned()],
            initial: vec![0.5, 0.5],
            horizon: 1,
            actions: vec![ActionSpec {
                label: "hold".to_owned(),
                role: ActionRole::Commit,
                utility_cost: 0.0,
                resource_use: 0.0,
                transition: vec![1.0, 0.0, 0.0, 1.0],
                channel: None,
                immediate_utility: vec![0.0, 0.0],
                delayed: None,
            }],
            terminal_utility: vec![0.0, 0.0],
            censoring: CensoringRule::Refuse,
            resource_budget: None,
            stopping: Vec::new(),
            support: ActionSupport::everywhere(1),
        }
    }

    fn policy_of(keys: &[HistoryKey]) -> ContingentPolicy {
        let mut policy = ContingentPolicy::new("v");
        for key in keys {
            policy = policy.decide(key.clone(), 0);
        }
        policy
    }

    fn owned(
        population: &str,
        regime: &str,
        law: &[f64],
        policy: ContingentPolicy,
        rows: &[(ModelMechanism, MechanismStatus)],
    ) -> OwnedPolicy {
        OwnedPolicy {
            source: model(population, regime),
            policy,
            target_population: "target".to_owned(),
            target_initial: Some(law.to_vec()),
            diagram: Some(SelectionDiagram {
                source_population: population.to_owned(),
                target_population: "target".to_owned(),
                mechanisms: rows.to_vec(),
            }),
            target_utility: None,
            limits: EvaluationLimits { max_nodes: 10, max_receipts: 10 },
        }
    }

    fn policy_digest(owned: &OwnedPolicy) -> [u8; 32] {
        policy_bytes(owned).unwrap()
    }

    /// `declare` for a transported policy: decisions are a set (the policy's map), the target
    /// law is ordered by state, diagram rows are a multiset, and population and regime are
    /// framed. A law component enters by its bits.
    #[test]
    fn site_transported_policy_decisions_are_a_set_and_the_target_law_is_ordered() {
        let quiet = HistoryKey::from_steps([HistoryStep { action: 0, observation: 0 }]);
        let alarm = HistoryKey::from_steps([HistoryStep { action: 0, observation: 1 }]);
        let rows = [
            (ModelMechanism::InitialState, MechanismStatus::Shifted),
            (ModelMechanism::Utility, MechanismStatus::Invariant),
        ];
        set_laws(&[quiet, alarm], |keys| {
            Some(policy_digest(&owned("plant", "operations", &[0.5, 0.5], policy_of(keys), &rows)))
        });
        ordered_laws(&[0.2, 0.8], |law| {
            Some(policy_digest(&owned(
                "plant",
                "operations",
                law,
                ContingentPolicy::new("v"),
                &rows,
            )))
        });
        multiset_laws(&rows, |rows| {
            Some(policy_digest(&owned(
                "plant",
                "operations",
                &[0.5, 0.5],
                ContingentPolicy::new("v"),
                rows,
            )))
        });
        bit_exact(|first| {
            Some(policy_digest(&owned(
                "plant",
                "operations",
                &[first, 0.5],
                ContingentPolicy::new("v"),
                &rows,
            )))
        });
        injective(
            &ADVERSARIAL_TEXT,
            |parts| *parts,
            |&(population, regime)| {
                Some(policy_digest(&owned(
                    population,
                    regime,
                    &[0.5, 0.5],
                    ContingentPolicy::new("v"),
                    &rows,
                )))
            },
        );
        // The display label is not read.
        let mut labeled =
            owned("plant", "operations", &[0.5, 0.5], ContingentPolicy::new("v"), &rows);
        labeled.source.label = "other".to_owned();
        let base = owned("plant", "operations", &[0.5, 0.5], ContingentPolicy::new("v"), &rows);
        assert_eq!(policy_digest(&labeled), policy_digest(&base));
    }

    fn temporal_of(
        symbols: &[String],
        rows: &[(ModelMechanism, MechanismStatus)],
    ) -> OwnedTemporal {
        OwnedTemporal {
            source: model("plant", "operations"),
            policy: ContingentPolicy::new("v"),
            target: TemporalTarget {
                period: "2027".to_owned(),
                horizon: 1,
                alphabet: HistoryAlphabet {
                    actions: vec![ActionAlphabet {
                        label: "hold".to_owned(),
                        symbols: symbols.to_vec(),
                    }],
                },
            },
            target_initial: Some(vec![0.4, 0.6]),
            invariance: Some(TimeInvariance {
                source_period: "2026".to_owned(),
                target_period: "2027".to_owned(),
                horizon: 1,
                mechanisms: rows.to_vec(),
            }),
            limits: EvaluationLimits { max_nodes: 10, max_receipts: 10 },
        }
    }

    /// `declare` for a period shift: observation symbols are ordered by symbol index, and the
    /// invariance rows are a multiset.
    #[test]
    fn site_temporal_policy_symbols_are_ordered_and_mechanisms_are_a_multiset() {
        let rows = [
            (ModelMechanism::InitialState, MechanismStatus::Shifted),
            (ModelMechanism::Utility, MechanismStatus::Invariant),
        ];
        ordered_laws(&["quiet".to_owned(), "alarm".to_owned()], |symbols| {
            temporal_bytes(&temporal_of(symbols, &rows)).ok()
        });
        multiset_laws(&rows, |rows| temporal_bytes(&temporal_of(&[], rows)).ok());
    }

    fn trial_of(strata: &[usize], rows: &[(TrialMechanism, MechanismStatus)]) -> OwnedTrial {
        OwnedTrial {
            contract: None,
            trial_population: "trial".to_owned(),
            unit_strata: strata.to_vec(),
            stratum_labels: vec!["mild".to_owned(), "severe".to_owned()],
            target_stratum_law: Some(vec![0.25, 0.75]),
            diagram: Some(TrialSelectionDiagram {
                source_population: "trial".to_owned(),
                target_population: "clinic".to_owned(),
                mechanisms: rows.to_vec(),
            }),
            treatment_arm: 1,
            control_arm: 0,
        }
    }

    /// `declare` for an adaptive trial: unit strata are in enrollment order, and the diagram
    /// rows are a multiset. The stratum law enters by its bits.
    #[test]
    fn site_adaptive_trial_strata_are_ordered_and_mechanisms_are_a_multiset() {
        let rows = [
            (TrialMechanism::StratumLaw, MechanismStatus::Shifted),
            (TrialMechanism::OutcomeGivenStratumAndArm, MechanismStatus::Invariant),
        ];
        ordered_laws(&[0, 1, 0], |strata| adaptive_bytes(&trial_of(strata, &rows)).ok());
        multiset_laws(&rows, |rows| adaptive_bytes(&trial_of(&[0, 1], rows)).ok());
        bit_exact(|mass| {
            let mut trial = trial_of(&[0, 1], &rows);
            trial.target_stratum_law = Some(vec![mass, 0.75]);
            adaptive_bytes(&trial).ok()
        });
    }
}
