//! Compile an explicit joint scenario carrier into layer four and execute its
//! robust terminal without coercing an envelope into a mean or a posterior.
//!
//! The robust decision is a parent program over the anchor study's compiled program
//! ([`crate::analysis::attached_decision`]): the anchor keeps its own support cell, and the
//! decision is gated on its own unit before it runs.
use crate::analysis::attached_decision::{ParentDecisionProgram, ParentSpec, lower_parent};
use crate::analysis::contract::CausalContract;
use crate::analysis::decision_contract_facade::BoundDecisionContract;
use crate::analysis::lowering::CompileGoal;
use crate::analysis::supplied_identity::Declaration;
use crate::planner::lowered::{CarrierDemand, Composition, DecisionProgram, ProgramCell};
use crate::planner::{PhysicalExecutionPlan, PlanView};
use crate::{CausalError, PreparedStudy, Study};
use antecedent_core::{CanonicalWriter, ExecutionContext, SemanticDigest};
use antecedent_design::decision_contract::StructuralPolicy;
use antecedent_design::decision_robust_artifact::{
    JointScenarioCarrier, MAX_COMPILED_ROBUST_ITEMS,
};
use antecedent_io::IoError;
use std::sync::Arc;

/// Tag of the robust decision's parent program over the anchor study's program.
const PARENT_TAG: &str = "antecedent.attached_decision.joint_scenarios.v1";
/// Tag of the robust decision's own declaration.
const DECLARATION_TAG: &str = "antecedent.attached_decision.joint_scenarios.declaration.v1";

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
    /// The decision's own program over the anchor study's program, gated on the decision's
    /// unit.
    parent: ParentDecisionProgram,
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
    /// The plan view of the decision's own program: a parent program over the anchor study's
    /// program, whose [`PlanView::unit_support`] is the support decision of the decision's
    /// unit (`decision.compiled_joint_scenarios`). The anchor keeps its own support cell.
    #[must_use]
    pub fn decision_plan_view(&self) -> PlanView {
        self.parent.plan_view()
    }
    /// Identity of the decision's own program: the anchor's program identity and the
    /// decision's declaration (the bound decision and the joint source).
    #[must_use]
    pub fn decision_program_identity(&self) -> SemanticDigest {
        self.parent.identity()
    }
    /// Explicit scope of every result, independent of producer trust labels.
    #[must_use]
    pub const fn standing(&self) -> &'static str {
        "conditional_on_declared_joint_scenarios"
    }
    /// Run the existing robust engine; never fit the anchor study's scalar estimator. The
    /// decision's own program passes its support gate first: a release build refuses a
    /// decision whose unit is not supported, a development build records it and runs.
    /// # Errors
    /// The release refusal of the decision's unit; quantity/policy/refusal, cancellation or
    /// insufficient admission.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<antecedent_design::decision_robust_artifact::ScenarioDecision, CausalError> {
        self.parent.gate(ctx)?;
        self.source.evaluate(self.decision.decision(), ctx).map_err(io)
    }
    /// Recompute and seal the terminal with its compiler/source/contract identities.
    /// # Errors
    /// The execution errors or serialization failure.
    pub fn export(&self, ctx: &ExecutionContext) -> Result<Vec<u8>, CausalError> {
        self.parent.gate(ctx)?;
        self.source
            .seal_decision(self.decision.decision(), self.program_identity(), ctx)
            .map_err(io)
    }
    /// Independently reconstruct the source and rerun the terminal against this
    /// separately retained compiler identity. Bytes alone cannot recreate a study.
    /// # Errors
    /// The release refusal of the decision's unit; any source, compiler, contract or numerical
    /// mismatch, or admission failure.
    pub fn consume(
        &self,
        bytes: &[u8],
        ctx: &ExecutionContext,
    ) -> Result<antecedent_design::decision_robust_artifact::ScenarioDecision, CausalError> {
        self.parent.gate(ctx)?;
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
    let parent_program = program.clone();
    let goal =
        CompileGoal::RobustDecision { program, source: writer.finish(), atoms: source.len() };
    let plan = study.compile_for_goal(&goal, ctx)?;
    let anchor = plan
        .lowered_program_identity()
        .ok_or_else(|| refused("compiler did not attach a robust lowered program"))?;
    let parent = lower_parent(ParentSpec {
        cell: ProgramCell::JointScenarioAttachedDecision,
        plan_id: "attached.joint_scenario_decision.v1",
        tag: PARENT_TAG,
        children: &[Some(anchor)],
        decision: Declaration::new(DECLARATION_TAG)
            .field("decision", decision.identity())
            .field("source", source.identity())
            .finish(),
        evaluator: "decision_robust_artifact::JointScenarioCarrier::evaluate",
        composition: Composition::ScenarioSet { scenarios: source.len() },
        program: parent_program,
        claim: None,
    })?;
    Ok(CompiledRobustDecision {
        plan: Arc::new(plan),
        decision: Arc::new(decision),
        source,
        program_identity: anchor.to_hex(),
        parent,
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

#[cfg(test)]
mod site_tests {
    use super::*;
    use crate::RefuteSuite;
    use crate::analysis::attached_decision::parent_identity;
    use crate::analysis::program_handle::CompiledProgramHandle;
    use crate::identity_laws::all_distinct;
    use crate::planner::lowered::{EvidenceRole, NonGraphicalIdentification};
    use antecedent_contract::{QuantityRole, ScientificQuantity};
    use antecedent_core::{AverageEffectQuery, VariableId};
    use antecedent_data::TabularData;
    use antecedent_design::decision_adapters::{
        ClaimProbability, SuppliedClaim, adapt_finite_scenarios,
    };
    use antecedent_design::decision_contract::{
        ActionKind, AdmissibilityRules, AdmissibleDecisionContract, DecisionAction,
        DecisionContract, DecisionCriterion, UtilityExpr,
    };
    use antecedent_design::decision_robustness::AtomSupport;
    use antecedent_design::decision_structural::AtomEvidence;
    use antecedent_estimate::LinearAdjustmentAte;
    use antecedent_graph::{Dag, DenseNodeId};
    use antecedent_io::distribution_artifact::{
        DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
        DistributionProvenance, DistributionTrust, DrawAlignment,
    };
    use antecedent_io::quantity_wire::DistributionMeaningWire;

    fn prepared(ctx: &ExecutionContext) -> PreparedStudy {
        let t = [0., 1., 0., 1., 0., 1., 0., 1.];
        let z = [0., 0., 1., 1., 0., 0., 1., 1.];
        let y = [0., 2., 1., 3., 1., 3., 2., 4.];
        let data = TabularData::from_f64_columns(vec![
            ("t", t.as_slice()),
            ("y", y.as_slice()),
            ("z", z.as_slice()),
        ])
        .unwrap();
        let mut graph = Dag::with_variables(3);
        for (a, b) in [(2, 0), (2, 1), (0, 1)] {
            graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
        Study::tabular(data)
            .graph(graph)
            .query(AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1)))
            .estimator(LinearAdjustmentAte::new().with_bootstrap_replicates(0))
            .refute(RefuteSuite::None)
            .build()
            .unwrap()
            .prepare(ctx)
            .unwrap()
    }

    fn quantity(name: &str) -> ScientificQuantity {
        ScientificQuantity {
            variable_id: name.into(),
            variable_name: name.into(),
            role: QuantityRole::Outcome,
            units: "units".into(),
            population_id: "target".into(),
            regime_id: "do(joint)".into(),
            horizon: 0,
            functional_id: "outcome".into(),
            conditioning: vec![],
            transform_id: "identity".into(),
        }
    }

    fn law(values: Vec<f64>) -> Box<DistributionArtifact> {
        let identity = DistributionIdentity::new(
            DistributionMeaningWire::InterventionalPredictive,
            &[quantity("a"), quantity("b")],
            DrawAlignment::Joint,
            DistributionProvenance {
                source_id: "declared".into(),
                provider_id: "enumeration".into(),
                rng_id: "none".into(),
                snapshot_id: "fixture".into(),
                causal_contract_id: "conditional".into(),
            },
        )
        .unwrap();
        Box::new(
            DistributionArtifact::new(
                DistributionMetadata {
                    version: 1,
                    identity,
                    axes: ["draw".into(), "quantity".into()],
                    shape: [2, 2],
                    weights: None,
                    supported: None,
                    calibration: DistributionCalibration::Exact,
                    trust: DistributionTrust::Unverified,
                    legacy_posterior: None,
                    legacy_bindings: None,
                },
                values,
            )
            .unwrap(),
        )
    }

    fn contract(policy: StructuralPolicy) -> AdmissibleDecisionContract {
        AdmissibleDecisionContract {
            contract: DecisionContract {
                actions: vec![
                    DecisionAction {
                        id: "pair".into(),
                        kind: ActionKind::Intervention,
                        inputs: vec![quantity("a"), quantity("b")],
                        utility: UtilityExpr::Mul(
                            Box::new(UtilityExpr::Input(0)),
                            Box::new(UtilityExpr::Input(1)),
                        ),
                    },
                    DecisionAction {
                        id: "safe".into(),
                        kind: ActionKind::Intervention,
                        inputs: vec![quantity("a")],
                        utility: UtilityExpr::sum(
                            UtilityExpr::product(UtilityExpr::Const(0.), UtilityExpr::Input(0)),
                            UtilityExpr::Const(0.3),
                        ),
                    },
                ],
                utility_units: "units".into(),
                criterion: DecisionCriterion::PosteriorExpectedUtility,
                constraints: vec![],
                target_population: "target".into(),
                horizon: 0,
                structural_policy: policy,
            },
            rules: AdmissibilityRules::default(),
        }
    }

    fn scenario(id: &str, values: Vec<f64>) -> SuppliedClaim {
        SuppliedClaim {
            id: id.into(),
            probability: ClaimProbability::CompletionCount(1),
            evidence: AtomEvidence::Evaluated(law(values)),
            support: AtomSupport::supported(),
        }
    }

    fn source(
        prepared: &PreparedStudy,
        claims: Vec<SuppliedClaim>,
        ctx: &ExecutionContext,
    ) -> JointScenarioCarrier {
        let causal = prepared.contract().unwrap();
        JointScenarioCarrier::checked(
            &causal.identities.target.to_hex(),
            &causal.identities.identification.to_hex(),
            "common paired states across actions",
            true,
            None,
            adapt_finite_scenarios(claims, StructuralPolicy::ReportOnly).unwrap(),
            ctx,
        )
        .unwrap()
    }

    /// The anchor's joint-scenario source digest and the parent's question digest.
    fn digests(compiled: &CompiledRobustDecision) -> ([u8; 32], [u8; 32]) {
        let anchor = compiled.plan().lowered().expect("a lowered anchor");
        let joint: Vec<[u8; 32]> = anchor
            .evidence()
            .sources()
            .iter()
            .filter(|source| source.role() == EvidenceRole::JointScenarios)
            .filter_map(|source| source.digest())
            .collect();
        assert_eq!(joint.len(), 1, "the anchor reads one joint scenario source");
        let question = match compiled.parent.program().non_graphical_identification() {
            Some(NonGraphicalIdentification::SharedEvidenceContract { contract }) => *contract,
            other => panic!("the parent declares its question as a shared contract: {other:?}"),
        };
        (joint[0], question)
    }

    /// The anchor's source digest is the carrier's identity under its own tag, and the parent's
    /// question is the anchor's identity then the declaration of (bound decision, source) in
    /// that fixed field order; a changed decision or source changes every digest it reaches.
    #[test]
    fn site_compile_robust_source_and_declaration_frame_their_parts() {
        let ctx = ExecutionContext::for_tests(43);
        let prepared = prepared(&ctx);
        let first = || {
            vec![
                scenario("positive", vec![0., 0., 1., 1.]),
                scenario("negative", vec![0., 1., 1., 0.]),
            ]
        };
        let compile = |claims: Vec<SuppliedClaim>, policy: StructuralPolicy| {
            let causal = prepared.contract().unwrap();
            prepared
                .compile_for_robust_decision(
                    BoundDecisionContract::bind(&causal, contract(policy)).expect("binds"),
                    source(&prepared, claims, &ctx),
                    &ctx,
                )
                .unwrap()
        };
        let base = compile(first(), StructuralPolicy::Maximin);
        let other_decision = compile(first(), StructuralPolicy::ReportOnly);
        let other_source =
            compile(vec![scenario("positive", vec![0., 0., 1., 1.])], StructuralPolicy::Maximin);
        for compiled in [&base, &other_decision, &other_source] {
            let (joint, question) = digests(compiled);
            let mut writer =
                CanonicalWriter::new("antecedent.lowered_program.joint_scenarios.source.v1");
            writer.text(compiled.source().identity());
            assert_eq!(joint, writer.finish(), "the source digest frames the carrier identity");
            let anchor = compiled.plan().lowered_program_identity().expect("a lowered anchor");
            let declaration = Declaration::new(DECLARATION_TAG)
                .field("decision", compiled.decision_contract().identity())
                .field("source", compiled.source().identity())
                .finish();
            assert_eq!(
                question,
                parent_identity(PARENT_TAG, &[Some(anchor)], declaration),
                "the question is the anchor then (decision, source) in field order"
            );
        }
        assert_eq!(digests(&base), digests(&compile(first(), StructuralPolicy::Maximin)));
        let (base_joint, base_question) = digests(&base);
        let (decision_joint, decision_question) = digests(&other_decision);
        let (source_joint, source_question) = digests(&other_source);
        assert_eq!(base_joint, decision_joint, "the source digest reads only the source");
        all_distinct(&[base_joint, source_joint]);
        all_distinct(&[base_question, decision_question, source_question]);
        all_distinct(&[
            base.decision_program_identity(),
            other_decision.decision_program_identity(),
            other_source.decision_program_identity(),
        ]);
    }
}
