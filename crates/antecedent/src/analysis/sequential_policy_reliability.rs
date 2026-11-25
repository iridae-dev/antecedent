//! Compiler-owned reliability diagnostics for a fixed finite set of sequential policies.

use crate::planner::PlanView;
use crate::planner::lowered::{
    CarrierDemand, ClaimDemand, Composition, DecisionProgram, DesignShape, EvidenceDecisionProgram,
    EvidenceRole, EvidenceSourceRef, FunctionalKind, IdentificationLayer, LoweredProgram,
    NonGraphicalIdentification, PhysicalLayer, ProblemTarget, ProgramCell, ProgramIdentityScope,
    ScientificProblem, StandaloneCompositionLoweredParts, UncertaintyReading,
};
use crate::planner::statistical::{
    Realization, ResampleUnit, ResamplingMethod, ResamplingPlan, StatisticalProgram,
};
use antecedent_contract::evidence_ledger::EvidenceLedger;
use antecedent_contract::{ClaimKind, DataClassification};
use antecedent_core::{
    CanonicalWriter, ExecutionContext, PhysicalEffect, PhysicalOperator, SemanticDigest,
};
use antecedent_design::decision_reliability::{
    DecisionDraws, ReliabilityReport, ReliabilityRequest, assess,
};
use antecedent_design::sequential_evaluation::{
    ContingentPolicy, EvaluationLimits, FiniteSequentialModel, LoggedTrajectory, PolicySpec,
    RolloutConfig, bootstrap_decision_draws, bootstrap_off_policy, evaluate_policy,
    rollout_decision_draws, rollout_policy,
};
use antecedent_design::sequential_evaluation_artifact::{
    ModelWire, PolicyWire, TrajectoryWire, model_from_wire_body, model_to_wire_body,
    policy_from_wire, policy_to_wire, trajectories_from_wire, trajectories_to_wire,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// A refusal of the sequential producer, diagnostic, or portable replay.
#[derive(Debug, thiserror::Error)]
pub enum SequentialReliabilityError {
    /// Invalid or mismatched declaration.
    #[error("sequential policy reliability: {0}")]
    Invalid(&'static str),
    /// Sequential evaluation refused.
    #[error("sequential policy evaluation: {0}")]
    Evaluation(#[from] antecedent_design::sequential_evaluation::SequentialEvaluationError),
    /// Reliability assessment refused.
    #[error("decision reliability: {0}")]
    Reliability(#[from] antecedent_design::decision_reliability::ReliabilityError),
    /// Portable representation refused.
    #[error("sequential reliability artifact: {0}")]
    Io(#[from] antecedent_io::IoError),
}

/// Raw evaluation mode; every policy is fixed before evaluating these data.
#[derive(Clone, Debug)]
pub enum SequentialReliabilityMode {
    /// Common-random-stream rollouts from the declared model.
    Rollout(RolloutConfig),
    /// Shared whole-trajectory bootstrap of ordinary importance-weighted estimates.
    OffPolicy {
        /// Policy that generated the logged trajectories.
        behaviour: ContingentPolicy,
        /// Logged trajectories resampled whole.
        log: Vec<LoggedTrajectory>,
        /// Seed of the bootstrap random stream.
        seed: u64,
        /// Number of bootstrap replicates, at least 2.
        replicates: u64,
        /// Smallest acceptable effective sample size, finite and at least 2.
        minimum_effective_sample_size: f64,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
enum ModeWire {
    Rollout {
        seed: u64,
        rollouts: u64,
    },
    OffPolicy {
        behaviour: PolicyWire,
        log: Vec<TrajectoryWire>,
        seed: u64,
        replicates: u64,
        minimum_effective_sample_size_bits: u64,
    },
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Declaration {
    model: ModelWire,
    policies: Vec<PolicyWire>,
    unit: String,
    mode: ModeWire,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Artifact {
    version: u32,
    declaration: Declaration,
    program: [u8; 32],
    draws_digest: String,
    report_digest: String,
}

/// Compiled six-layer program and retained raw declaration.
pub struct SequentialPolicyReliabilityPlan {
    declaration: Declaration,
    model: FiniteSequentialModel,
    policies: Vec<ContingentPolicy>,
    mode: SequentialReliabilityMode,
    lowered: Arc<LoweredProgram>,
}
/// Diagnostic report and the aligned producer draws that support it.
pub struct SequentialPolicyReliabilityResult {
    /// Aligned per-policy utilities or bootstrap estimates.
    pub draws: DecisionDraws,
    /// Point diagnostic; no coverage calibration is claimed.
    pub report: ReliabilityReport,
    /// Identity of the compiler program.
    pub lowered_program_identity: SemanticDigest,
}

/// Compile fixed policies and raw inputs before evaluating any candidate.
///
/// # Errors
/// A learned or static-score policy, duplicate candidate, invalid declaration or lowering failure.
pub fn compile_sequential_policy_reliability(
    model: FiniteSequentialModel,
    policies: Vec<PolicySpec>,
    unit: String,
    mode: SequentialReliabilityMode,
) -> Result<SequentialPolicyReliabilityPlan, SequentialReliabilityError> {
    compile_sequential_policy_reliability_as(model, policies, unit, mode, Question::Current)
}

/// Which question digest a program binds: the current one, or the released one (the logged
/// trajectories in declared order) a retained artifact was compiled under.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Question {
    Current,
    V1,
}

/// The declaration as the question writes it. Policies are ordered (draws and the report are
/// indexed by policy); the off-policy log is a set of trajectories keyed by id (the estimator
/// and the bootstrap read it sorted by id and refuse a repeated id), so the current question
/// writes it sorted by id. The released question kept the declared order.
fn declaration_bytes(
    declaration: &Declaration,
    version: Question,
) -> Result<Vec<u8>, antecedent_io::IoError> {
    match version {
        Question::Current => {
            let mut canonical = declaration.clone();
            if let ModeWire::OffPolicy { log, .. } = &mut canonical.mode {
                log.sort_by(|a, b| a.id.cmp(&b.id));
            }
            antecedent_io::to_cbor(&canonical)
        }
        Question::V1 => antecedent_io::to_cbor(declaration),
    }
}

#[allow(clippy::too_many_lines, reason = "one validation per declared mode and policy field")]
fn compile_sequential_policy_reliability_as(
    model: FiniteSequentialModel,
    policies: Vec<PolicySpec>,
    unit: String,
    mode: SequentialReliabilityMode,
    version: Question,
) -> Result<SequentialPolicyReliabilityPlan, SequentialReliabilityError> {
    let policies: Vec<_> =
        policies.into_iter().map(PolicySpec::into_contingent).collect::<Result<_, _>>()?;
    if policies.len() < 2 || unit.trim().is_empty() {
        return Err(SequentialReliabilityError::Invalid(
            "at least two fixed policies and a utility unit are required",
        ));
    }
    model.validate()?;
    for policy in &policies {
        policy.validate()?;
    }
    match &mode {
        SequentialReliabilityMode::Rollout(config)
            if config.rollouts < 2
                || config.rollouts > antecedent_design::sequential_evaluation::MAX_ROLLOUTS =>
        {
            return Err(SequentialReliabilityError::Invalid(
                "rollout count outside supported range",
            ));
        }
        SequentialReliabilityMode::OffPolicy {
            behaviour,
            log,
            replicates,
            minimum_effective_sample_size,
            ..
        } => {
            behaviour.validate()?;
            if log.is_empty()
                || log.len() > antecedent_design::sequential_evaluation::MAX_LOGGED_TRAJECTORIES
                || *replicates < 2
                || *replicates > antecedent_design::sequential_evaluation::MAX_ROLLOUTS
                || !minimum_effective_sample_size.is_finite()
                || *minimum_effective_sample_size < 2.0
            {
                return Err(SequentialReliabilityError::Invalid("invalid bootstrap declaration"));
            }
            // The log is a set of trajectories keyed by id: the estimator refuses a repeated
            // id when it runs, so the program refuses it before it is named.
            let mut trajectory_ids: Vec<&str> = log.iter().map(|t| t.id.as_str()).collect();
            trajectory_ids.sort_unstable();
            if trajectory_ids.windows(2).any(|pair| pair[0] == pair[1]) {
                return Err(SequentialReliabilityError::Invalid(
                    "a trajectory identity appears more than once in the log",
                ));
            }
        }
        SequentialReliabilityMode::Rollout(_) => {}
    }
    let mut ids: Vec<_> = policies.iter().map(ContingentPolicy::identity_hex).collect();
    ids.sort();
    if ids.windows(2).any(|w| w[0] == w[1]) {
        return Err(SequentialReliabilityError::Invalid("policy identities must be unique"));
    }
    let mode_wire = match &mode {
        SequentialReliabilityMode::Rollout(c) => {
            ModeWire::Rollout { seed: c.seed, rollouts: c.rollouts }
        }
        SequentialReliabilityMode::OffPolicy {
            behaviour,
            log,
            seed,
            replicates,
            minimum_effective_sample_size,
        } => ModeWire::OffPolicy {
            behaviour: policy_to_wire(behaviour),
            log: trajectories_to_wire(log),
            seed: *seed,
            replicates: *replicates,
            minimum_effective_sample_size_bits: minimum_effective_sample_size.to_bits(),
        },
    };
    let declaration = Declaration {
        model: model_to_wire_body(&model),
        policies: policies.iter().map(policy_to_wire).collect(),
        unit,
        mode: mode_wire,
    };
    let bytes = declaration_bytes(&declaration, version)?;
    let mut writer = CanonicalWriter::new(match version {
        Question::Current => "antecedent.sequential_policy_reliability.v2",
        Question::V1 => "antecedent.sequential_policy_reliability.v1",
    });
    writer.bytes(&bytes);
    let identity = writer.finish();
    let design = DesignShape::quoted(1, 1, true);
    let scientific = ScientificProblem {
        runtime_query: None,
        identity_scope: ProgramIdentityScope::Semantic,
        target: ProblemTarget::QueryFunctional {
            kind: FunctionalKind::PolicyValue,
            outcome: None,
            query: identity,
        },
        plan_id: Arc::from("sequential_policy_reliability.v1"),
        data: DataClassification::Tabular,
        graph_class: "finite_sequential_policy",
        query_variables: Arc::from([]),
        source_digest: Some(identity),
        support_coordinate: None,
        supported: false,
        support_status: None,
    };
    let identification = IdentificationLayer::NonGraphical(Arc::new(
        NonGraphicalIdentification::SharedEvidenceContract { contract: identity },
    ));
    let (realization, resampling, uncertainty) = match &mode {
        SequentialReliabilityMode::Rollout(_) => {
            (Realization::SequentialKnownLawMonteCarlo, None, UncertaintyReading::MonteCarlo)
        }
        SequentialReliabilityMode::OffPolicy { replicates, .. } => {
            let count = u32::try_from(*replicates).map_err(|_| {
                SequentialReliabilityError::Invalid("too many bootstrap replicates")
            })?;
            (
                Realization::SequentialTrajectoryBootstrap,
                Some(ResamplingPlan {
                    method: ResamplingMethod::Bootstrap,
                    replicates: count,
                    unit: ResampleUnit::Subject,
                }),
                UncertaintyReading::Sampling,
            )
        }
    };
    let statistical = StatisticalProgram {
        identifier: None,
        estimator: None,
        realization,
        models: Arc::from([]),
        folds: None,
        resampling,
        posterior: None,
        design,
        operators: Arc::from([]),
    };
    let evidence = EvidenceDecisionProgram {
        sources: Arc::from([EvidenceSourceRef {
            role: EvidenceRole::ChainSource,
            digest: Some(identity),
        }]),
        composition: Composition::Chained { stages: 2 },
        decision: Some(DecisionProgram {
            carrier: CarrierDemand::DecisionDraws,
            utility: DecisionProgram::utility_form(None),
            actions: policies.len(),
            constraints: 0,
            chance_threshold: 0.0,
            utility_unit: Some(Arc::from(declaration.unit.as_str())),
            contract_identity: None,
            information_value: None,
            scenario_source: Some(Arc::from("sequential_policy_reliability.v1")),
        }),
        planning: None,
        chain: None,
    };
    let physical = PhysicalLayer {
        operators: Arc::from([
            PhysicalOperator::declare(PhysicalEffect::Read)
                .build()
                .map_err(|_| SequentialReliabilityError::Invalid("invalid read operator"))?,
            PhysicalOperator::declare(PhysicalEffect::Intervene).build().map_err(|_| {
                SequentialReliabilityError::Invalid("invalid policy intervention operator")
            })?,
        ]),
        design,
    };
    let lowered = Arc::new(
        StandaloneCompositionLoweredParts {
            cell: ProgramCell::SequentialPolicyReliability,
            scientific,
            identification,
            statistical,
            evidence,
            physical,
            claim: ClaimDemand { uncertainty, kind: Some(ClaimKind::PolicyValue) },
        }
        .finish(),
    );
    Ok(SequentialPolicyReliabilityPlan { declaration, model, policies, mode, lowered })
}

impl SequentialPolicyReliabilityPlan {
    /// Six-layer compiled program.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        &self.lowered
    }
    /// Inspect the compiled program.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        crate::analysis::program_handle::CompiledProgramHandle::plan_view(self)
    }
    /// Exact prospective action-choice study scope for fixed policies. It
    /// binds the model, policy set and sampling design, excluding realized
    /// trajectories and random seeds. No reliability claim is earned here.
    ///
    /// # Errors
    /// The design cannot be encoded.
    pub fn decision_reliability_scope(&self) -> Result<SemanticDigest, SequentialReliabilityError> {
        let mut writer =
            CanonicalWriter::new("antecedent.sequential_policy_reliability.action_choice_study.v1");
        writer
            .bytes(&antecedent_io::to_cbor(&self.declaration.model)?)
            .bytes(&antecedent_io::to_cbor(&self.declaration.policies)?)
            .text(&self.declaration.unit);
        match &self.mode {
            SequentialReliabilityMode::Rollout(config) => {
                writer.text("common_stream_known_law_rollout_v1").word(config.rollouts);
            }
            SequentialReliabilityMode::OffPolicy {
                behaviour,
                log,
                replicates,
                minimum_effective_sample_size,
                ..
            } => {
                writer
                    .text("whole_trajectory_bootstrap_uncalibrated_v1")
                    .bytes(&antecedent_io::to_cbor(&policy_to_wire(behaviour))?)
                    .word(log.len() as u64)
                    .word(*replicates)
                    .word(minimum_effective_sample_size.to_bits());
            }
        }
        Ok(SemanticDigest::from_bytes(writer.finish()))
    }
    /// Rerun the raw evaluation and compute the diagnostic report.
    ///
    /// # Errors
    /// Any support, evaluation, carrier or diagnostic refusal.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<SequentialPolicyReliabilityResult, SequentialReliabilityError> {
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        // A release build refuses an unsupported unit; the refusal rides `Invalid` with its
        // `reason=not_supported:` wire message, which the Python binding raises as such.
        crate::analysis::program_handle::gate_program(&scoped, self)
            .map_err(|refused| SequentialReliabilityError::Invalid(refused.message()))?;
        // Enumerate reachable histories first: sampling cannot establish support.
        for policy in &self.policies {
            evaluate_policy(&self.model, policy, EvaluationLimits::default(), &scoped)?;
        }
        let draws = match &self.mode {
            SequentialReliabilityMode::Rollout(config) => {
                let rows = self
                    .policies
                    .iter()
                    .map(|policy| rollout_policy(&self.model, policy, *config, &scoped))
                    .collect::<Result<Vec<_>, _>>()?;
                rollout_decision_draws(&rows, &self.declaration.unit)?
            }
            SequentialReliabilityMode::OffPolicy {
                behaviour,
                log,
                seed,
                replicates,
                minimum_effective_sample_size,
            } => {
                let mut ledger = EvidenceLedger::new();
                let bootstrap = bootstrap_off_policy(
                    &self.model,
                    &self.policies,
                    behaviour,
                    log,
                    &mut ledger,
                    EvaluationLimits::default(),
                    *seed,
                    *replicates,
                    *minimum_effective_sample_size,
                    &scoped,
                )?;
                bootstrap_decision_draws(&bootstrap, &self.declaration.unit)?
            }
        };
        let report = assess(&draws, None, &ReliabilityRequest::expected_utility(), &scoped)?;
        Ok(SequentialPolicyReliabilityResult {
            draws,
            report,
            lowered_program_identity: self.lowered.identity(),
        })
    }
    /// Export raw declaration and diagnostic digests for independent recomputation.
    ///
    /// # Errors
    /// A foreign or changed result, evaluation refusal, oversized artifact or encoding failure.
    pub fn export(
        &self,
        result: &SequentialPolicyReliabilityResult,
        ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, SequentialReliabilityError> {
        if result.lowered_program_identity != self.lowered.identity() {
            return Err(SequentialReliabilityError::Invalid("foreign result"));
        }
        let replay = self.execute(ctx)?;
        if replay.draws != result.draws || replay.report != result.report {
            return Err(SequentialReliabilityError::Invalid("result differs from raw evaluation"));
        }
        let report_digest = report_digest(&replay.report)?;
        let bytes = antecedent_io::to_cbor(&Artifact {
            version: 1,
            declaration: self.declaration.clone(),
            program: *self.lowered.identity().as_bytes(),
            draws_digest: replay.draws.digest_hex(),
            report_digest,
        })?;
        if bytes.len() > 80 * 1024 * 1024 {
            return Err(SequentialReliabilityError::Invalid("artifact too large"));
        }
        Ok(bytes)
    }
}
fn report_digest(report: &ReliabilityReport) -> Result<String, SequentialReliabilityError> {
    let bytes = antecedent_io::to_cbor(report)?;
    let mut writer = CanonicalWriter::new("antecedent.sequential_policy_reliability.report.v1");
    writer.bytes(&bytes);
    Ok(writer.finish_hex())
}
/// Consume a portable declaration by rerunning raw model or log evaluation.
///
/// # Errors
/// Corruption, a changed program, or a replay mismatch.
pub fn consume_sequential_policy_reliability(
    bytes: &[u8],
    expected: Option<SemanticDigest>,
    ctx: &ExecutionContext,
) -> Result<SequentialPolicyReliabilityResult, SequentialReliabilityError> {
    if bytes.len() > 80 * 1024 * 1024 {
        return Err(SequentialReliabilityError::Invalid("artifact too large"));
    }
    let artifact: Artifact = antecedent_io::from_cbor(bytes)?;
    if artifact.version != 1 {
        return Err(SequentialReliabilityError::Invalid("unsupported artifact version"));
    }
    let d = artifact.declaration;
    let compile_as = |version| -> Result<_, SequentialReliabilityError> {
        let model = model_from_wire_body(&d.model)?;
        let policies = d
            .policies
            .iter()
            .map(policy_from_wire)
            .map(|p| p.map(PolicySpec::Contingent))
            .collect::<Result<Vec<_>, _>>()?;
        let mode = match &d.mode {
            ModeWire::Rollout { seed, rollouts } => {
                SequentialReliabilityMode::Rollout(RolloutConfig {
                    seed: *seed,
                    rollouts: *rollouts,
                })
            }
            ModeWire::OffPolicy {
                behaviour,
                log,
                seed,
                replicates,
                minimum_effective_sample_size_bits,
            } => SequentialReliabilityMode::OffPolicy {
                behaviour: policy_from_wire(behaviour)?,
                log: trajectories_from_wire(log)?,
                seed: *seed,
                replicates: *replicates,
                minimum_effective_sample_size: f64::from_bits(*minimum_effective_sample_size_bits),
            },
        };
        compile_sequential_policy_reliability_as(model, policies, d.unit.clone(), mode, version)
    };
    let mut plan = compile_as(Question::Current)?;
    if artifact.program != *plan.lowered.identity().as_bytes() {
        // An artifact retained under the released question still verifies: it is recompiled
        // under that question and must match it exactly.
        plan = compile_as(Question::V1)?;
    }
    if artifact.program != *plan.lowered.identity().as_bytes()
        || expected.is_some_and(|v| v != plan.lowered.identity())
    {
        return Err(SequentialReliabilityError::Invalid("program identity mismatch"));
    }
    let result = plan.execute(ctx)?;
    if result.draws.digest_hex() != artifact.draws_digest
        || report_digest(&result.report)? != artifact.report_digest
    {
        return Err(SequentialReliabilityError::Invalid("replayed diagnostic differs"));
    }
    Ok(result)
}

impl crate::analysis::program_handle::CompiledProgramHandle for SequentialPolicyReliabilityPlan {
    fn program(&self) -> &std::sync::Arc<crate::planner::lowered::LoweredProgram> {
        &self.lowered
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_design::sequential_evaluation::{
        ActionRole, ActionSpec, ActionSupport, CensoringRule, HistoryKey, LoggedStep,
    };

    fn fixture() -> (FiniteSequentialModel, Vec<PolicySpec>) {
        let action = |label: &str, utility: Vec<f64>| ActionSpec {
            label: label.into(),
            role: ActionRole::Commit,
            utility_cost: 0.0,
            resource_use: 0.0,
            transition: vec![1.0, 0.0, 0.0, 1.0],
            channel: None,
            immediate_utility: utility,
            delayed: None,
        };
        let model = FiniteSequentialModel {
            label: "test".into(),
            population: "p".into(),
            regime: "r".into(),
            period: "t".into(),
            state_variable: "state".into(),
            states: vec!["a".into(), "b".into()],
            initial: vec![0.5, 0.5],
            horizon: 1,
            actions: vec![action("a", vec![2.0, 0.0]), action("b", vec![0.0, 1.0])],
            terminal_utility: vec![0.0, 0.0],
            censoring: CensoringRule::Refuse,
            resource_budget: None,
            stopping: vec![],
            support: ActionSupport::everywhere(2),
        };
        let policies = vec![
            PolicySpec::Contingent(ContingentPolicy::new("a").decide(HistoryKey::root(), 0)),
            PolicySpec::Contingent(ContingentPolicy::new("b").decide(HistoryKey::root(), 1)),
        ];
        (model, policies)
    }

    #[test]
    fn rollout_compiles_and_replays_raw_declaration() {
        let (model, policies) = fixture();
        let plan = compile_sequential_policy_reliability(
            model,
            policies,
            "utility".into(),
            SequentialReliabilityMode::Rollout(RolloutConfig { seed: 5, rollouts: 128 }),
        )
        .unwrap();
        let ctx = ExecutionContext::for_tests(1);
        let result = plan.execute(&ctx).unwrap();
        let bytes = plan.export(&result, &ctx).unwrap();
        let replay =
            consume_sequential_policy_reliability(&bytes, Some(plan.lowered().identity()), &ctx)
                .unwrap();
        assert_eq!(result.draws, replay.draws);
        assert_eq!(result.report, replay.report);
        let mut altered = result;
        altered.draws.candidates[0].utilities[0] += 1.0;
        assert!(plan.export(&altered, &ctx).is_err());
    }

    #[test]
    fn learned_policy_and_unsupported_action_are_refused() {
        let (mut model, mut policies) = fixture();
        policies.push(PolicySpec::LearnedFromEvaluationData { rule: "same log".into() });
        assert!(
            compile_sequential_policy_reliability(
                model.clone(),
                policies,
                "utility".into(),
                SequentialReliabilityMode::Rollout(RolloutConfig { seed: 1, rollouts: 32 })
            )
            .is_err()
        );
        model.support = ActionSupport::by_action(vec![true, false]);
        let (_, policies) = fixture();
        let plan = compile_sequential_policy_reliability(
            model,
            policies,
            "utility".into(),
            SequentialReliabilityMode::Rollout(RolloutConfig { seed: 1, rollouts: 32 }),
        )
        .unwrap();
        assert!(plan.execute(&ExecutionContext::for_tests(1)).is_err());
    }

    #[test]
    fn bootstrap_replays_whole_trajectories() {
        let (model, policies) = fixture();
        let behaviour = ContingentPolicy::new("logging")
            .decide_stochastic(HistoryKey::root(), vec![(0, 0.5), (1, 0.5)])
            .unwrap();
        let log = (0..16)
            .map(|i| LoggedTrajectory {
                id: format!("{i}"),
                steps: vec![LoggedStep {
                    action: i % 2,
                    observation: None,
                    assignment_probability: 0.5,
                }],
                outcome_utility: (i % 4) as f64,
            })
            .collect();
        let plan = compile_sequential_policy_reliability(
            model,
            policies,
            "utility".into(),
            SequentialReliabilityMode::OffPolicy {
                behaviour,
                log,
                seed: 3,
                replicates: 64,
                minimum_effective_sample_size: 2.0,
            },
        )
        .unwrap();
        let ctx = ExecutionContext::for_tests(1);
        let result = plan.execute(&ctx).unwrap();
        let replay =
            consume_sequential_policy_reliability(&plan.export(&result, &ctx).unwrap(), None, &ctx)
                .unwrap();
        assert_eq!(result.draws, replay.draws);
    }

    fn query_of(lowered: &LoweredProgram) -> [u8; 32] {
        match lowered.scientific().target() {
            ProblemTarget::QueryFunctional { query, .. } => *query,
            _ => panic!("the program target is a query functional"),
        }
    }

    #[test]
    fn site_sequential_policy_reliability_declaration_orders_policies_and_binds_mode() {
        let identity = |actions: &[usize], unit: &str, seed: u64, rollouts: u64| {
            let (model, _) = fixture();
            let policies = actions
                .iter()
                .map(|&action| {
                    PolicySpec::Contingent(
                        ContingentPolicy::new(format!("always-{action}"))
                            .decide(HistoryKey::root(), action),
                    )
                })
                .collect();
            compile_sequential_policy_reliability(
                model,
                policies,
                unit.into(),
                SequentialReliabilityMode::Rollout(RolloutConfig { seed, rollouts }),
            )
            .ok()
            .map(|plan| query_of(plan.lowered()))
        };
        // The candidates are reported, drawn and assessed in declared order.
        crate::identity_laws::ordered_laws(&[0, 1], |actions: &[usize]| {
            identity(actions, "utility", 5, 128)
        });
        crate::identity_laws::all_distinct(&[
            identity(&[0, 1], "utility", 5, 128),
            identity(&[0, 1], "points", 5, 128),
            identity(&[0, 1], "utility", 6, 128),
            identity(&[0, 1], "utility", 5, 129),
        ]);
    }

    #[test]
    fn site_sequential_policy_reliability_report_digest_reads_the_report() {
        let (model, policies) = fixture();
        let plan = compile_sequential_policy_reliability(
            model,
            policies,
            "utility".into(),
            SequentialReliabilityMode::Rollout(RolloutConfig { seed: 5, rollouts: 128 }),
        )
        .unwrap();
        let report = plan.execute(&ExecutionContext::for_tests(1)).unwrap().report;
        let mut unit = report.clone();
        unit.utility_unit = "points".into();
        let mut decided = report.clone();
        decided.decided_id = "another-policy".into();
        let mut draws = report.clone();
        draws.draws_digest = "0".repeat(64);
        crate::identity_laws::all_distinct(&[
            report_digest(&report).unwrap(),
            report_digest(&unit).unwrap(),
            report_digest(&decided).unwrap(),
            report_digest(&draws).unwrap(),
        ]);
    }

    fn logging_policy() -> ContingentPolicy {
        ContingentPolicy::new("logging")
            .decide_stochastic(HistoryKey::root(), vec![(0, 0.5), (1, 0.5)])
            .unwrap()
    }

    /// A logged trajectory whose content follows from its id, so reordering the log moves
    /// whole trajectories.
    fn trajectory(id: &str) -> LoggedTrajectory {
        let byte = id.as_bytes().first().copied().unwrap_or(0);
        LoggedTrajectory {
            id: id.into(),
            steps: vec![LoggedStep {
                action: usize::from(byte % 2),
                observation: None,
                assignment_probability: 0.5,
            }],
            outcome_utility: f64::from(byte % 4),
        }
    }

    fn off_policy_plan(ids: &[&str], version: Question) -> Option<SequentialPolicyReliabilityPlan> {
        let (model, policies) = fixture();
        compile_sequential_policy_reliability_as(
            model,
            policies,
            "utility".into(),
            SequentialReliabilityMode::OffPolicy {
                behaviour: logging_policy(),
                log: ids.iter().map(|id| trajectory(id)).collect(),
                seed: 3,
                replicates: 64,
                minimum_effective_sample_size: 2.0,
            },
            version,
        )
        .ok()
    }

    #[test]
    fn site_sequential_policy_reliability_off_policy_log_is_a_set() {
        // The estimator and the bootstrap read the log sorted by trajectory id and a repeated
        // id is refused (the repeated case is refused at compile and skipped by the law).
        crate::identity_laws::set_laws(&["a", "b", "c", "d"], |ids: &[&str]| {
            off_policy_plan(ids, Question::Current).map(|plan| query_of(plan.lowered()))
        });
        assert!(off_policy_plan(&["a", "b", "a"], Question::Current).is_none());
        // The released question wrote the log in declared order.
        let released =
            |ids: &[&str]| off_policy_plan(ids, Question::V1).map(|plan| query_of(plan.lowered()));
        assert_ne!(released(&["a", "b", "c"]), released(&["c", "a", "b"]));
    }

    #[test]
    fn site_sequential_policy_reliability_released_identity_still_verifies() {
        let ids: Vec<String> = (0..16).map(|i| format!("{i}")).collect();
        let ids: Vec<&str> = ids.iter().map(String::as_str).collect();
        let released =
            off_policy_plan(&ids, Question::V1).expect("compiles under the released question");
        let current = off_policy_plan(&ids, Question::Current).expect("compiles");
        assert_ne!(released.lowered().identity(), current.lowered().identity());
        let ctx = ExecutionContext::for_tests(1);
        let result = released.execute(&ctx).expect("the released program runs");
        let bytes = released.export(&result, &ctx).expect("it exports");
        let replay = consume_sequential_policy_reliability(
            &bytes,
            Some(released.lowered().identity()),
            &ctx,
        )
        .expect("an artifact retained under the released question still verifies");
        assert_eq!(replay.draws, result.draws);
    }
}
