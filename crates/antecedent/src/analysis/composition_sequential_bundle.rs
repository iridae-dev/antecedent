//! Live raw sequential evaluation, aligned draws and diagnostic bundle.
use super::*;
use antecedent_design::decision_reliability::{assess, DecisionDraws, ReliabilityReport, ReliabilityRequest};

const FIT: &str = "sequential_reliability_fit_v1";
const DRAWS: &str = "sequential_decision_draws_v1";
const DIAGNOSTIC: &str = "sequential_reliability_diagnostic_v1";
const PORTABLE: &str = "sequential_reliability_bundle_v1";
const DRAW_FACT: &str = "sequential_draws_digest";
const REPORT_FACT: &str = "sequential_diagnostic_digest";

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FitWire { feature: String, seed: u64, program: [u8; 32], artifact: Vec<u8> }
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DrawWire { feature: String, draws: DecisionDraws }
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DiagnosticWire { feature: String, draws: DecisionDraws, report: ReliabilityReport }
fn report_digest(report: &ReliabilityReport) -> Result<String, IoError> {
    Ok(CanonicalWriter::digest_bytes_hex("antecedent.sequential_bundle.report.v1", &antecedent_io::to_cbor(report)?))
}
fn failed(reason: impl Into<String>) -> NodeFailure { NodeFailure { stage: BundleStage::SwappedEvidence, reason: reason.into() } }
fn facts(draws: &DecisionDraws, report: Option<&ReliabilityReport>) -> Result<std::collections::BTreeMap<String, String>, IoError> {
    let mut facts = std::collections::BTreeMap::new(); facts.insert(DRAW_FACT.into(), draws.digest_hex());
    if let Some(report) = report { facts.insert(REPORT_FACT.into(), report_digest(report)?); }
    Ok(facts)
}
pub(super) struct SequentialFitVerifier { pub(super) ctx: ExecutionContext }
impl NodeVerifier for SequentialFitVerifier {
    fn kind(&self) -> NodeKind { NodeKind::SequentialPolicyReliabilityFit }
    fn verify(&self, _: &str, bytes: &[u8], _: &[UpstreamNode<'_>]) -> Result<VerifiedNode, NodeFailure> {
        if bytes.len() > MAX_BUNDLE_RECOMPUTE_BYTES { return Err(failed("sequential fit exceeds bound")); }
        let wire: FitWire = antecedent_io::from_cbor(bytes).map_err(|e| failed(e.to_string()))?;
        if wire.feature != FIT { return Err(failed("sequential fit feature differs")); }
        let ctx = fit_node_context(&self.ctx, Some(wire.seed));
        let live = super::super::sequential_policy_reliability::consume_sequential_policy_reliability(
            &wire.artifact, Some(SemanticDigest::from_bytes(wire.program)), &ctx).map_err(|e| fit_node_failure(&ctx, &e))?;
        Ok(VerifiedNode { identity: CanonicalWriter::digest_bytes_hex("antecedent.sequential_bundle.fit.v1", bytes),
            facts: facts(&live.draws, Some(&live.report)).map_err(|e| failed(e.to_string()))?, values: Vec::new() })
    }
}
pub(super) struct SequentialDrawsVerifier;
impl NodeVerifier for SequentialDrawsVerifier {
    fn kind(&self) -> NodeKind { NodeKind::SequentialDecisionDraws }
    fn verify(&self, _: &str, bytes: &[u8], _: &[UpstreamNode<'_>]) -> Result<VerifiedNode, NodeFailure> {
        if bytes.len() > MAX_BUNDLE_RECOMPUTE_BYTES { return Err(failed("sequential draws exceed bound")); }
        let wire: DrawWire = antecedent_io::from_cbor(bytes).map_err(|e| failed(e.to_string()))?;
        if wire.feature != DRAWS { return Err(failed("draw feature differs")); }
        wire.draws.validate().map_err(|e| failed(e.to_string()))?;
        Ok(VerifiedNode { identity: CanonicalWriter::digest_bytes_hex("antecedent.sequential_bundle.draws.v1", bytes),
            facts: facts(&wire.draws, None).map_err(|e| failed(e.to_string()))?, values: Vec::new() })
    }
}
pub(super) struct SequentialDiagnosticVerifier { pub(super) ctx: ExecutionContext }
impl NodeVerifier for SequentialDiagnosticVerifier {
    fn kind(&self) -> NodeKind { NodeKind::SequentialReliabilityDiagnostic }
    fn verify(&self, _: &str, bytes: &[u8], _: &[UpstreamNode<'_>]) -> Result<VerifiedNode, NodeFailure> {
        if bytes.len() > MAX_BUNDLE_RECOMPUTE_BYTES { return Err(failed("diagnostic exceeds bound")); }
        let wire: DiagnosticWire = antecedent_io::from_cbor(bytes).map_err(|e| failed(e.to_string()))?;
        if wire.feature != DIAGNOSTIC { return Err(failed("diagnostic feature differs")); }
        let report = assess(&wire.draws, None, &ReliabilityRequest::expected_utility(), &self.ctx)
            .map_err(|e| fit_node_failure(&self.ctx, &e))?;
        if wire.report != report { return Err(failed("diagnostic differs from live assessment")); }
        Ok(VerifiedNode { identity: CanonicalWriter::digest_bytes_hex("antecedent.sequential_bundle.diagnostic.v1", bytes),
            facts: facts(&wire.draws, Some(&report)).map_err(|e| failed(e.to_string()))?, values: Vec::new() })
    }
}

/// Failure to replay the sequential evidence, carrier or diagnostic.
#[derive(Debug, thiserror::Error)]
pub enum SequentialReliabilityBundleError {
    #[error("sequential reliability bundle: {0}")]
    Bundle(#[from] ChainBundleRecomputeError),
    #[error("sequential reliability bundle: {0}")]
    Invalid(&'static str),
}
impl BundleBuilder {
    /// Embed an independently replayed raw sequential policy evaluation.
    pub fn add_sequential_policy_reliability_fit(&mut self, id: &str, artifact: &[u8], program: SemanticDigest,
        ctx: &ExecutionContext) -> Result<(), SequentialReliabilityBundleError> {
        self.ensure_new(id).map_err(ChainBundleRecomputeError::Bundle)?;
        let bytes = antecedent_io::to_cbor(&FitWire { feature: FIT.into(), seed: ctx.rng.master_seed(), program: *program.as_bytes(),
            artifact: artifact.to_vec() }).map_err(ChainBundleRecomputeError::Io)?;
        self.add_sequential_node(id, NodeKind::SequentialPolicyReliabilityFit, bytes,
            &SequentialFitVerifier { ctx: ctx.clone() })
    }
    /// Embed a fully disclosed aligned decision-draw carrier.
    pub fn add_sequential_decision_draws(&mut self, id: &str, draws: &DecisionDraws)
        -> Result<(), SequentialReliabilityBundleError> {
        self.ensure_new(id).map_err(ChainBundleRecomputeError::Bundle)?;
        let bytes = antecedent_io::to_cbor(&DrawWire { feature: DRAWS.into(), draws: draws.clone() }).map_err(ChainBundleRecomputeError::Io)?;
        self.add_sequential_node(id, NodeKind::SequentialDecisionDraws, bytes, &SequentialDrawsVerifier)
    }
    /// Embed a diagnostic which is recomputed from its declared draws on replay.
    pub fn add_sequential_reliability_diagnostic(&mut self, id: &str, draws: &DecisionDraws,
        report: &ReliabilityReport, ctx: &ExecutionContext) -> Result<(), SequentialReliabilityBundleError> {
        self.ensure_new(id).map_err(ChainBundleRecomputeError::Bundle)?;
        let bytes = antecedent_io::to_cbor(&DiagnosticWire { feature: DIAGNOSTIC.into(), draws: draws.clone(), report: report.clone() })
            .map_err(ChainBundleRecomputeError::Io)?;
        self.add_sequential_node(id, NodeKind::SequentialReliabilityDiagnostic, bytes, &SequentialDiagnosticVerifier { ctx: ctx.clone() })
    }
    fn add_sequential_node(&mut self, id: &str, kind: NodeKind, bytes: Vec<u8>, verifier: &dyn NodeVerifier)
        -> Result<(), SequentialReliabilityBundleError> {
        if bytes.len() > self.limits.max_embedded_bytes { return Err(SequentialReliabilityBundleError::Invalid("node exceeds bound")); }
        let inspected = verifier.verify(id, &bytes, &[]).map_err(|e| ChainBundleRecomputeError::Bundle(node_refused(id, &e)))?;
        self.nodes.push(BundleNode::embedded(id, kind, &inspected.identity, bytes)); Ok(())
    }
}
/// Compiler-owned producer, draw carrier and reliability diagnostic.
pub struct CompiledSequentialReliabilityBundle {
    bundle: CompositionBundle, producer: String, draws: String, terminal: String,
    program: SemanticDigest, lowered: Arc<LoweredProgram>,
}
/// Live point diagnostic identities; this does not claim calibration.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SequentialReliabilityBundleReceipt {
    pub bundle_identity: String, pub draws_digest: String, pub diagnostic_digest: String,
    pub producer_program_identity: [u8; 32], pub lowered_program_identity: [u8; 32],
    pub dependencies: Vec<ReplayedBundleDependency>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PortableWire { feature: String, bundle: Vec<u8>, bundle_identity: String, producer: String,
    draws: String, terminal: String, receipt: SequentialReliabilityBundleReceipt }
fn embedded<'a>(bundle: &'a CompositionBundle, id: &str, kind: NodeKind) -> Result<&'a [u8], SequentialReliabilityBundleError> {
    let n = bundle.nodes().iter().find(|n| n.id == id).ok_or(SequentialReliabilityBundleError::Invalid("node missing"))?;
    if n.kind != kind { return Err(SequentialReliabilityBundleError::Invalid("node kind differs")); }
    match &n.source { NodeSource::Embedded { bytes } => Ok(bytes), _ => Err(SequentialReliabilityBundleError::Invalid("node must be embedded")) }
}
/// Compile a direct producer -> aligned draws -> live diagnostic graph.
pub fn compile_sequential_reliability_node_bundle(bundle: CompositionBundle, producer: &str, draws: &str,
    terminal: &str) -> Result<CompiledSequentialReliabilityBundle, SequentialReliabilityBundleError> {
    for (from, to) in [(producer, draws), (draws, terminal)] {
        if !bundle.edges().iter().any(|e| e.from == from && e.to == to) {
            return Err(SequentialReliabilityBundleError::Invalid("direct producer, carrier and diagnostic edges required"));
        }
    }
    let fit: FitWire = antecedent_io::from_cbor(embedded(&bundle, producer, NodeKind::SequentialPolicyReliabilityFit)?)
        .map_err(ChainBundleRecomputeError::Io)?;
    let carrier: DrawWire = antecedent_io::from_cbor(embedded(&bundle, draws, NodeKind::SequentialDecisionDraws)?)
        .map_err(ChainBundleRecomputeError::Io)?;
    let diagnostic: DiagnosticWire = antecedent_io::from_cbor(embedded(&bundle, terminal, NodeKind::SequentialReliabilityDiagnostic)?)
        .map_err(ChainBundleRecomputeError::Io)?;
    if fit.feature != FIT || carrier.feature != DRAWS || diagnostic.feature != DIAGNOSTIC || carrier.draws != diagnostic.draws {
        return Err(SequentialReliabilityBundleError::Invalid("node features or diagnostic draw carrier differ"));
    }
    let program = SemanticDigest::from_bytes(fit.program);
    let digest = CanonicalWriter::digest_bytes("antecedent.sequential_bundle.graph.v1", bundle.identity().as_bytes());
    let mut writer = CanonicalWriter::new("antecedent.sequential_bundle.question.v1");
    writer.bytes(&digest).text(producer).text(draws).text(terminal).bytes(program.as_bytes());
    let query = writer.finish();
    let design = DesignShape::quoted(carrier.draws.candidates.len() as u64, 1, true);
    let scientific = ScientificProblem { runtime_query: None, identity_scope: ProgramIdentityScope::Semantic,
        target: ProblemTarget::QueryFunctional { kind: FunctionalKind::PolicyValue, outcome: None, query },
        plan_id: Arc::from("sequential_policy_reliability_node_bundle_v1"), data: DataClassification::Tabular,
        graph_class: "sequential_policy_reliability_bundle", query_variables: Arc::from([]), source_digest: Some(digest),
        support_coordinate: None, supported: false, support_status: None };
    let identification = IdentificationLayer::NonGraphical(Arc::new(NonGraphicalIdentification::SharedEvidenceContract { contract: query }));
    let statistical = StatisticalProgram { identifier: None, estimator: None,
        realization: Realization::ExactPoint { evaluator: Arc::from("sequential_reliability_live_diagnostic_v1") },
        models: Arc::from([]), folds: None, resampling: None, posterior: None, design, operators: Arc::from([]) };
    let evidence = EvidenceDecisionProgram { sources: Arc::from([EvidenceSourceRef { role: EvidenceRole::Observations, digest: Some(digest) }]),
        composition: Composition::Chained { stages: bundle.nodes().len() + 1 }, decision: None, planning: None, chain: None };
    let mut operators = Vec::new(); for effect in [PhysicalEffect::Read, PhysicalEffect::Fit, PhysicalEffect::Condition] {
        operators.push(PhysicalOperator::declare(effect).build().map_err(|_| SequentialReliabilityBundleError::Invalid("physical operator refused"))?);
    }
    let physical = PhysicalLayer { operators: Arc::from(operators), design };
    let lowered = Arc::new(StandaloneCompositionLoweredParts { cell: ProgramCell::SequentialPolicyReliabilityNodeBundleRecomputation,
        scientific, identification, statistical, evidence, physical,
        claim: ClaimDemand { uncertainty: UncertaintyReading::None, kind: None } }.finish());
    Ok(CompiledSequentialReliabilityBundle { bundle, producer: producer.into(), draws: draws.into(), terminal: terminal.into(), program, lowered })
}
impl CompiledSequentialReliabilityBundle {
    pub fn lowered(&self) -> &LoweredProgram { &self.lowered }
    pub fn plan_view(&self) -> PlanView { PlanView::of_program(Arc::clone(&self.lowered), PlanStage::Compiled) }
    /// Replay the raw producer and independently reassess the exact retained draws.
    pub fn execute(&self, ctx: &ExecutionContext) -> Result<SequentialReliabilityBundleReceipt, SequentialReliabilityBundleError> {
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        let bytes = export_bundle(&self.bundle, "sequential-reliability-bundle").map_err(ChainBundleRecomputeError::Bundle)?;
        let consumed = native_fit_bundle_consumer(&scoped).consume(&bytes, &BundleLimits::default(), self.bundle.identity(), &SuppliedSources::default())
            .map_err(ChainBundleRecomputeError::Bundle)?;
        if !consumed.all_replayed() { return Err(SequentialReliabilityBundleError::Invalid("bundle dependency did not replay")); }
        let fit = consumed.node(&self.producer).ok_or(SequentialReliabilityBundleError::Invalid("producer absent"))?;
        let carrier = consumed.node(&self.draws).ok_or(SequentialReliabilityBundleError::Invalid("draws absent"))?;
        let terminal = consumed.node(&self.terminal).ok_or(SequentialReliabilityBundleError::Invalid("diagnostic absent"))?;
        let wire: DrawWire = antecedent_io::from_cbor(embedded(&self.bundle, &self.draws, NodeKind::SequentialDecisionDraws)?)
            .map_err(ChainBundleRecomputeError::Io)?;
        let report = assess(&wire.draws, None, &ReliabilityRequest::expected_utility(), &scoped)
            .map_err(|_| SequentialReliabilityBundleError::Invalid("live diagnostic refused"))?;
        let live = facts(&wire.draws, Some(&report)).map_err(ChainBundleRecomputeError::Io)?;
        if fit.facts.get(DRAW_FACT) != live.get(DRAW_FACT) || fit.facts.get(REPORT_FACT) != live.get(REPORT_FACT)
            || carrier.facts.get(DRAW_FACT) != live.get(DRAW_FACT) || terminal.facts.get(DRAW_FACT) != live.get(DRAW_FACT)
            || terminal.facts.get(REPORT_FACT) != live.get(REPORT_FACT) {
            return Err(SequentialReliabilityBundleError::Invalid("producer, draw carrier and live diagnostic differ"));
        }
        Ok(SequentialReliabilityBundleReceipt { bundle_identity: consumed.identity().into(), draws_digest: live[DRAW_FACT].clone(),
            diagnostic_digest: live[REPORT_FACT].clone(), producer_program_identity: *self.program.as_bytes(),
            lowered_program_identity: *self.lowered.identity().as_bytes(), dependencies: consumed.nodes().iter().map(|n| ReplayedBundleDependency {
                node_id: n.id.clone(), artifact_identity: n.identity.clone(), chain_digest: n.chain_digest.clone() }).collect() })
    }
    pub fn export_portable(&self, receipt: &SequentialReliabilityBundleReceipt, ctx: &ExecutionContext)
        -> Result<Vec<u8>, SequentialReliabilityBundleError> {
        if self.execute(ctx)? != *receipt { return Err(SequentialReliabilityBundleError::Invalid("receipt differs")); }
        let bytes = antecedent_io::to_cbor(&PortableWire { feature: PORTABLE.into(),
            bundle: export_bundle(&self.bundle, "portable-sequential-reliability").map_err(ChainBundleRecomputeError::Bundle)?,
            bundle_identity: self.bundle.identity().into(), producer: self.producer.clone(), draws: self.draws.clone(),
            terminal: self.terminal.clone(), receipt: receipt.clone() }).map_err(ChainBundleRecomputeError::Io)?;
        if bytes.len() > MAX_BUNDLE_RECOMPUTE_BYTES { return Err(SequentialReliabilityBundleError::Invalid("portable bundle exceeds bound")); }
        Ok(bytes)
    }
}
/// Independently reconstruct the graph and recompute its sequential diagnostic.
pub fn consume_portable_sequential_reliability_node_bundle(bytes: &[u8], expected_bundle: &str,
    expected_program: SemanticDigest, ctx: &ExecutionContext) -> Result<SequentialReliabilityBundleReceipt, SequentialReliabilityBundleError> {
    if bytes.len() > MAX_BUNDLE_RECOMPUTE_BYTES { return Err(SequentialReliabilityBundleError::Invalid("portable bundle exceeds bound")); }
    let wire: PortableWire = antecedent_io::from_cbor(bytes).map_err(ChainBundleRecomputeError::Io)?;
    if wire.feature != PORTABLE || wire.bundle_identity != expected_bundle { return Err(SequentialReliabilityBundleError::Invalid("portable identity differs")); }
    let bundle = CompositionBundle::from_bytes(&wire.bundle, expected_bundle, &BundleLimits::default()).map_err(ChainBundleRecomputeError::Bundle)?;
    let plan = compile_sequential_reliability_node_bundle(bundle, &wire.producer, &wire.draws, &wire.terminal)?;
    if plan.lowered.identity() != expected_program { return Err(SequentialReliabilityBundleError::Invalid("portable compiler differs")); }
    let live = plan.execute(ctx)?;
    if live != wire.receipt { return Err(SequentialReliabilityBundleError::Invalid("portable receipt differs")); }
    Ok(live)
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_design::sequential_evaluation::{ActionRole, ActionSpec, ActionSupport, CensoringRule,
        ContingentPolicy, FiniteSequentialModel, HistoryKey, PolicySpec, RolloutConfig};
    use super::super::super::sequential_policy_reliability::{compile_sequential_policy_reliability, SequentialReliabilityMode};

    fn fixture() -> (FiniteSequentialModel, Vec<PolicySpec>) {
        let action = |label: &str, utility: Vec<f64>| ActionSpec { label: label.into(), role: ActionRole::Commit,
            utility_cost: 0.0, resource_use: 0.0, transition: vec![1.0, 0.0, 0.0, 1.0],
            channel: None, immediate_utility: utility, delayed: None };
        let model = FiniteSequentialModel { label: "test".into(), population: "p".into(), regime: "r".into(),
            period: "t".into(), state_variable: "state".into(), states: vec!["a".into(), "b".into()],
            initial: vec![0.5, 0.5], horizon: 1, actions: vec![action("a", vec![2.0, 0.0]), action("b", vec![0.0, 1.0])],
            terminal_utility: vec![0.0, 0.0], censoring: CensoringRule::Refuse, resource_budget: None,
            stopping: vec![], support: ActionSupport::everywhere(2) };
        let policies = vec![PolicySpec::Contingent(ContingentPolicy::new("a").decide(HistoryKey::root(), 0)),
            PolicySpec::Contingent(ContingentPolicy::new("b").decide(HistoryKey::root(), 1))];
        (model, policies)
    }
    #[test]
    #[ignore = "source-only sequential diagnostic bundle; no calibration earned"]
    fn producer_carrier_and_diagnostic_recompute_portably() {
        let (model, policies) = fixture();
        let producer = compile_sequential_policy_reliability(model, policies, "utility".into(),
            SequentialReliabilityMode::Rollout(RolloutConfig { seed: 5, rollouts: 128 })).unwrap();
        let ctx = ExecutionContext::for_tests(1); let result = producer.execute(&ctx).unwrap();
        let artifact = producer.export(&result, &ctx).unwrap();
        let mut builder = BundleBuilder::new();
        builder.add_sequential_policy_reliability_fit("fit", &artifact, producer.lowered().identity(), &ctx).unwrap();
        builder.add_sequential_decision_draws("draws", &result.draws).unwrap();
        builder.add_sequential_reliability_diagnostic("diagnostic", &result.draws, &result.report, &ctx).unwrap();
        assert!(compile_sequential_reliability_node_bundle(builder.build().unwrap(), "fit", "draws", "diagnostic").is_err());
        builder.connect("fit", "draws").unwrap(); builder.connect("draws", "diagnostic").unwrap();
        let bundle = builder.build().unwrap();
        let plan = compile_sequential_reliability_node_bundle(bundle.clone(), "fit", "draws", "diagnostic").unwrap();
        assert_eq!(plan.plan_view().route().as_str(), "SequentialPolicyReliabilityNodeBundleRecomputation");
        let receipt = plan.execute(&ctx).unwrap();
        assert_eq!(receipt.draws_digest, result.draws.digest_hex());
        let portable = plan.export_portable(&receipt, &ctx).unwrap();
        assert_eq!(consume_portable_sequential_reliability_node_bundle(&portable, bundle.identity(), plan.lowered().identity(), &ctx).unwrap(), receipt);
        assert!(consume_portable_sequential_reliability_node_bundle(&portable, "wrong", plan.lowered().identity(), &ctx).is_err());
        let mut changed = result.draws.clone(); changed.candidates[0].utilities[0] += 1.0;
        assert!(BundleBuilder::new().add_sequential_reliability_diagnostic("bad", &changed, &result.report, &ctx).is_err());
        let mut mixed = BundleBuilder::new();
        mixed.add_sequential_policy_reliability_fit("fit", &artifact, producer.lowered().identity(), &ctx).unwrap();
        mixed.add_sequential_decision_draws("draws", &changed).unwrap();
        let changed_report = assess(&changed, None, &ReliabilityRequest::expected_utility(), &ctx).unwrap();
        mixed.add_sequential_reliability_diagnostic("diagnostic", &changed, &changed_report, &ctx).unwrap();
        mixed.connect("fit", "draws").unwrap(); mixed.connect("draws", "diagnostic").unwrap();
        assert!(compile_sequential_reliability_node_bundle(mixed.build().unwrap(), "fit", "draws", "diagnostic")
            .unwrap().execute(&ctx).is_err());
    }
}
