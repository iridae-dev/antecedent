//! Compile an explicit joint scenario carrier into layer four and execute its
//! robust terminal without coercing an envelope into a mean or a posterior.
use crate::analysis::contract::CausalContract;
use crate::analysis::decision_contract_facade::BoundDecisionContract;
use crate::analysis::lowering::CompileGoal;
use crate::planner::PhysicalExecutionPlan;
use crate::planner::lowered::{CarrierDemand, DecisionProgram};
use crate::{CausalError, PreparedStudy, Study};
use antecedent_core::{CanonicalWriter, ExecutionContext};
use antecedent_design::decision_contract::StructuralPolicy;
use antecedent_design::decision_robust_artifact::{
    JointScenarioCarrier, MAX_COMPILED_ROBUST_ITEMS,
};
use antecedent_io::IoError;
use std::sync::Arc;

fn io(error: IoError) -> CausalError {
    match error {
        IoError::Admission(antecedent_core::execution::AdmissionError::Cancelled) => {
            CausalError::Cancelled { stage: "compiled_robust_decision" }
        }
        IoError::Admission(error) => CausalError::Resource { message: error.to_string() },
        other => CausalError::Serialization(other),
    }
}
fn refused(message: &str) -> CausalError {
    io(IoError::Refused {
        code: antecedent_core::reason_code!("decision_contract_unsatisfied"),
        message: format!(
            "{}: {message}",
            antecedent_core::reason_details::decision_joint_scenarios::CARRIER
        ),
    })
}

struct DeclarationCounter(u64);
impl std::io::Write for DeclarationCounter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0 = self
            .0
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| std::io::Error::other("declaration size overflow"))?;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
fn declaration_bytes(decision: &BoundDecisionContract) -> Result<u64, CausalError> {
    use std::io::Write;
    let mut count = DeclarationCounter(4096);
    write!(&mut count, "{decision:?}")
        .map_err(|e| CausalError::Resource { message: e.to_string() })?;
    count
        .0
        .checked_mul(2)
        .ok_or_else(|| CausalError::Resource { message: "declaration size overflow".into() })
}

/// A compiler-native robust terminal over an explicit joint scenario set.
/// It performs no scalar fit and grants no posterior or expected-utility
/// interpretation across structures. Scientific standing is conditional on
/// the supplied joint laws and the declared scenario search/exhaustiveness.
#[derive(Clone, Debug)]
pub struct CompiledRobustDecision {
    plan: Arc<PhysicalExecutionPlan>,
    decision: Arc<BoundDecisionContract>,
    source: JointScenarioCarrier,
    program_identity: String,
    _declaration_retention: Arc<antecedent_core::execution::AdmissionLease>,
}
impl CompiledRobustDecision {
    /// The shared compiler plan. Layer four names the set carrier and exact source.
    #[must_use]
    pub fn plan(&self) -> &PhysicalExecutionPlan {
        &self.plan
    }
    /// The decision bound to the study's target and identification.
    #[must_use]
    pub fn decision_contract(&self) -> &BoundDecisionContract {
        &self.decision
    }
    /// Exact joint source, retaining incomplete search and unidentified states.
    #[must_use]
    pub fn source(&self) -> &JointScenarioCarrier {
        &self.source
    }
    /// Lowered program identity including the robust source and bound decision.
    #[must_use]
    pub fn program_identity(&self) -> &str {
        &self.program_identity
    }
    /// Explicit scope of every result, independent of producer trust labels.
    #[must_use]
    pub const fn standing(&self) -> &'static str {
        "conditional_on_declared_joint_scenarios"
    }
    /// Run the existing robust engine; never fit the anchor study's scalar estimator.
    /// # Errors
    /// Quantity/policy/refusal, cancellation or insufficient admission.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<antecedent_design::decision_robust_artifact::ScenarioDecision, CausalError> {
        self.source.evaluate(self.decision.decision(), ctx).map_err(io)
    }
    /// Recompute and seal the terminal with its compiler/source/contract identities.
    /// # Errors
    /// The execution errors or serialization failure.
    pub fn export(&self, ctx: &ExecutionContext) -> Result<Vec<u8>, CausalError> {
        self.source
            .seal_decision(self.decision.decision(), self.program_identity(), ctx)
            .map_err(io)
    }
    /// Independently reconstruct the source and rerun the terminal against this
    /// separately retained compiler identity. Bytes alone cannot recreate a study.
    /// # Errors
    /// Any source, compiler, contract or numerical mismatch, or admission failure.
    pub fn consume(
        &self,
        bytes: &[u8],
        ctx: &ExecutionContext,
    ) -> Result<antecedent_design::decision_robust_artifact::ScenarioDecision, CausalError> {
        self.source
            .replay_decision(self.decision.decision(), self.program_identity(), bytes, ctx)
            .map_err(io)
    }
}

fn compile(
    study: &Study,
    causal: &CausalContract,
    decision: BoundDecisionContract,
    source: JointScenarioCarrier,
    ctx: &ExecutionContext,
) -> Result<CompiledRobustDecision, CausalError> {
    crate::analysis::lowering::refuse_reentry(ctx)?;
    if ctx.cancellation.is_cancelled() {
        return Err(CausalError::Cancelled { stage: "compile_robust_decision" });
    }
    let target = causal.identities.target.to_hex();
    let identification = causal.identities.identification.to_hex();
    if decision.causal_target() != target
        || decision.causal_identification() != identification
        || source.causal_target() != target
        || source.causal_identification() != identification
    {
        return Err(refused(
            "joint scenarios and decision must bind this study's exact causal target and identification",
        ));
    }
    if causal
        .reasoning
        .support
        .as_ref()
        .is_none_or(|support| support.matrix_status.as_ref() != "supported")
    {
        return Err(refused(
            "explicit joint scenarios do not reopen an unsupported anchor study; choose a supported support cell",
        ));
    }
    let contract = &decision.decision().contract;
    if contract.structural_policy == StructuralPolicy::BayesOverStructures
        || contract.actions.len() > MAX_COMPILED_ROBUST_ITEMS
        || contract.constraints.len() > MAX_COMPILED_ROBUST_ITEMS
    {
        return Err(refused(
            "use invariant, maximin or report-only policy over at most 256 actions and constraints; a robust set is not a posterior",
        ));
    }
    decision.decision().identity().map_err(|e| refused(&format!("{e:?}")))?;
    let declaration_retention = Arc::new(
        ctx.acquire_admission(0, declaration_bytes(&decision)?)
            .map_err(|e| io(IoError::Admission(e)))?,
    );
    let source = source.admit(ctx).map_err(io)?;
    let program = DecisionProgram {
        carrier: CarrierDemand::IdentifiedSet,
        utility: DecisionProgram::utility_form(None),
        actions: contract.actions.len(),
        constraints: contract.constraints.len(),
        chance_threshold: contract
            .constraints
            .iter()
            .map(|c| c.min_probability)
            .fold(1.0, f64::min),
        utility_unit: Some(Arc::from(contract.utility_units.as_str())),
        contract_identity: Some(Arc::from(decision.identity())),
        information_value: None,
        scenario_source: Some(Arc::from(source.identity())),
    };
    let mut writer = CanonicalWriter::new("antecedent.lowered_program.joint_scenarios.source.v1");
    writer.text(source.identity());
    let goal =
        CompileGoal::RobustDecision { program, source: writer.finish(), atoms: source.len() };
    let plan = study.compile_for_goal(&goal, ctx)?;
    let program_identity = plan
        .lowered_program_identity()
        .ok_or_else(|| refused("compiler did not attach a robust lowered program"))?
        .to_hex();
    Ok(CompiledRobustDecision {
        plan: Arc::new(plan),
        decision: Arc::new(decision),
        source,
        program_identity,
        _declaration_retention: declaration_retention,
    })
}
impl Study {
    /// Compile a robust terminal from explicitly joint finite scenarios. Marginal
    /// envelopes alone cannot supply cross-action dependence and continue to refuse.
    /// # Errors
    /// Causal/source mismatch, unsupported policy, invalid declaration or compiler/admission failure.
    pub fn compile_for_robust_decision(
        &self,
        decision: BoundDecisionContract,
        source: JointScenarioCarrier,
        ctx: &ExecutionContext,
    ) -> Result<CompiledRobustDecision, CausalError> {
        compile(self, &self.inspect()?, decision, source, ctx)
    }
}
impl PreparedStudy {
    /// Compile the same robust terminal through a prepared handle's frozen contract.
    /// No estimate or refit is run. All source rows and scenario probabilities stay explicit.
    /// # Errors
    /// The same checked bindings and compiler refusals as `Study::compile_for_robust_decision`.
    pub fn compile_for_robust_decision(
        &self,
        decision: BoundDecisionContract,
        source: JointScenarioCarrier,
        ctx: &ExecutionContext,
    ) -> Result<CompiledRobustDecision, CausalError> {
        compile(self.study(), &self.contract()?, decision, source, ctx)
    }
}
