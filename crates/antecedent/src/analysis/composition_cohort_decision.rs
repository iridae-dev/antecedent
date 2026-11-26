//! Compiler replay of an affine decision over a retained, weighted cohort.

use std::sync::Arc;

use antecedent_contract::DataClassification;
use antecedent_core::{
    ByteEstimates, CanonicalWriter, ExecutionContext, PhysicalEffect, PhysicalOperator,
    SemanticDigest,
};
use antecedent_design::cohort_decision_input::cohort_mean_binding;
use antecedent_design::composition_boundary::SupportPolicy;
use antecedent_design::composition_verifiers::{
    CohortDecisionSourceVerifier, register_standard_verifiers,
};
use antecedent_design::decision_artifact::{
    DecisionContractArtifact, DecisionResultArtifact, bound_mean_result_to_bytes,
    decision_result_fingerprint,
};
use antecedent_design::decision_contract::{DecisionContract, SourceRepresentation};
use antecedent_design::decision_eval::DecisionResult;
use serde::{Deserialize, Serialize};

use super::cohort_decision::{CohortDecisionSource, CohortExpectation, evaluate_cohort_decision};
use super::composition::{
    BundleBuilder, BundleConsumer, BundleError, BundleLimits, BundleNode, BundleStage,
    CompositionBundle, NodeKind, NodeSource, ReplayedBundleDependency, SuppliedSources,
    export_bundle,
};
use super::composition_decision_terminal::{
    DecisionBundleRecomputationReceipt, DecisionBundleRecomputeError,
};
use crate::planner::PlanView;
use crate::planner::lowered::{
    CarrierDemand, ClaimDemand, Composition, DecisionProgram, DesignShape, EvidenceDecisionProgram,
    EvidenceRole, EvidenceSourceRef, FunctionalKind, IdentificationLayer, LoweredProgram,
    NonGraphicalIdentification, PhysicalLayer, ProblemTarget, ProgramCell, ProgramIdentityScope,
    ScientificProblem, StandaloneCompositionLoweredParts, UncertaintyReading,
};
use crate::planner::statistical::{Realization, StatisticalProgram};
use crate::support::unit::{SupportKey, require_program, require_program_unscoped};
use crate::support::units_generated::COMPOSITION_BUNDLE;

const COMPILE: &str = "antecedent.composition.compile_cohort_decision_bundle_recomputation";
const CONSUME: &str =
    "antecedent.composition.consume_portable_cohort_decision_bundle_recomputation";
const PRODUCE: &str = "antecedent.composition.cohort_decision_bytes";
const FEATURE: &str = "cohort_decision_bundle_recomputation_portable_v1";
const MAX_PORTABLE_BYTES: usize = 32 * 1024 * 1024;

type Error = DecisionBundleRecomputeError;

fn refused(stage: BundleStage, reason: impl Into<String>, node: Option<&str>) -> Error {
    BundleError::Refused { stage, reason: reason.into(), offending: node.map(str::to_owned) }.into()
}

fn check_mean_requirement(contract: &DecisionContract) -> Result<(), Error> {
    if contract
        .source_requirement()?
        .is_some_and(|r| !r.any_of.contains(&SourceRepresentation::Mean))
    {
        return Err(refused(
            BundleStage::UnsupportedLaw,
            "a cohort contrast supports affine expectation only",
            None,
        ));
    }
    Ok(())
}

fn score(
    contract: &DecisionContract,
    source: &CohortDecisionSource,
) -> Result<DecisionResult, Error> {
    check_mean_requirement(contract)?;
    let evaluated = evaluate_cohort_decision(contract, source, &SupportPolicy::require_all())
        .map_err(|error| Error::Evaluation(format!("{error:?}")))?;
    let [result] = evaluated.results.try_into().map_err(|_: Vec<DecisionResult>| {
        Error::Invalid("the decision requires at least two actions on the supported cohort")
    })?;
    Ok(result)
}

/// Export the affine decision on a cohort for use as a decision-result node.
///
/// Connect the result to both the source and the decision contract using bundle edges.
///
/// # Errors
/// A law-valued request, unsupported coordinates, or encoding failure.
pub fn cohort_decision_bytes(
    contract: &DecisionContract,
    source: &CohortDecisionSource,
    artifact_id: &str,
) -> Result<Vec<u8>, Error> {
    require_program_unscoped(&SupportKey::Unit(COMPOSITION_BUNDLE), PRODUCE)?;
    let result = score(contract, source)?;
    Ok(bound_mean_result_to_bytes(&result, &cohort_mean_binding(source), artifact_id)?)
}

impl BundleBuilder {
    /// Add the validated cohort decision source to the bundle.
    ///
    /// # Errors
    /// An invalid or oversized cohort artifact, or a repeated node id.
    pub fn add_cohort_decision_source(
        &mut self,
        node_id: Option<&str>,
        bytes: &[u8],
    ) -> Result<String, BundleError> {
        self.add_artifact(node_id, NodeKind::CohortDecisionSource, bytes)
    }
}

/// A composition with one supported cohort mean and one affine decision terminal.
#[derive(Clone, Debug)]
pub struct CompiledCohortDecisionBundleRecomputation {
    bundle: CompositionBundle,
    source_node: String,
    terminal_node: String,
    contract: DecisionContract,
    contract_identity: String,
    lowered: Arc<LoweredProgram>,
}

fn embedded(node: &BundleNode) -> Result<&[u8], Error> {
    match &node.source {
        NodeSource::Embedded { bytes } => Ok(bytes),
        NodeSource::Reference { .. } => Err(refused(
            BundleStage::CallbackUnavailable,
            "the decision needs the embedded source or contract artifact",
            Some(&node.id),
        )),
    }
}

fn by_id<'a>(bundle: &'a CompositionBundle, id: &str) -> Result<&'a BundleNode, Error> {
    bundle.nodes().iter().find(|node| node.id == id).ok_or_else(|| {
        refused(BundleStage::GraphOrSnapshotMismatch, "bundle node is absent", Some(id))
    })
}

fn direct_parent<'a>(
    bundle: &'a CompositionBundle,
    terminal: &str,
    kind: NodeKind,
) -> Result<&'a BundleNode, Error> {
    let mut matches = bundle.nodes().iter().filter(|node| {
        node.kind == kind
            && bundle.edges().iter().any(|edge| edge.from == node.id && edge.to == terminal)
    });
    let parent = matches.next().filter(|_| matches.next().is_none()).ok_or_else(|| {
        refused(
            BundleStage::GraphOrSnapshotMismatch,
            format!("the terminal needs exactly one upstream {}", kind.as_str()),
            Some(terminal),
        )
    })?;
    embedded(parent)?;
    Ok(parent)
}

/// Compile a retained cohort → affine decision graph without executing the fitted model.
///
/// # Errors
/// A missing, referenced or ambiguous source or contract, an invalid contract, or a
/// decision whose functional requires a law.
pub fn compile_cohort_decision_bundle_recomputation(
    bundle: CompositionBundle,
    terminal_node: &str,
) -> Result<CompiledCohortDecisionBundleRecomputation, Error> {
    require_program_unscoped(&SupportKey::Unit(COMPOSITION_BUNDLE), COMPILE)?;
    let terminal = by_id(&bundle, terminal_node)?;
    if terminal.kind != NodeKind::DecisionResult {
        return Err(Error::Invalid("the terminal must be a decision result node"));
    }
    embedded(terminal)?;
    let source = direct_parent(&bundle, terminal_node, NodeKind::CohortDecisionSource)?;
    let contract_node = direct_parent(&bundle, terminal_node, NodeKind::DecisionContract)?;
    let contract =
        DecisionContractArtifact::from_bytes(embedded(contract_node)?, &contract_node.identity)?;
    check_mean_requirement(contract.contract())?;
    let lowered = lower(&bundle, source, terminal_node, &contract)?;
    let source_node = source.id.clone();
    Ok(CompiledCohortDecisionBundleRecomputation {
        bundle,
        source_node,
        terminal_node: terminal_node.into(),
        contract: contract.contract().clone(),
        contract_identity: contract.identity().into(),
        lowered: Arc::new(lowered),
    })
}

#[allow(clippy::too_many_lines, reason = "one declaration for each layer of the cohort decision")]
fn lower(
    bundle: &CompositionBundle,
    source: &BundleNode,
    terminal: &str,
    contract: &DecisionContractArtifact,
) -> Result<LoweredProgram, Error> {
    let source_digest = CanonicalWriter::digest_bytes(
        "antecedent.cohort_decision_bundle.source.v1",
        source.identity.as_bytes(),
    );
    let mut question = CanonicalWriter::new("antecedent.cohort_decision_bundle.question.v1");
    question.text(bundle.identity()).text(&source.id).text(terminal).text(contract.identity());
    let query = question.finish();
    let design = DesignShape::quoted(1, 1, true);
    let scientific = ScientificProblem {
        runtime_query: None,
        identity_scope: ProgramIdentityScope::Semantic,
        target: ProblemTarget::QueryFunctional {
            kind: FunctionalKind::PropagatedChain,
            outcome: None,
            query,
        },
        plan_id: Arc::from("cohort_decision_bundle_recomputation.v1"),
        data: DataClassification::Tabular,
        graph_class: "bundle_decision_dependency_graph",
        query_variables: Arc::from([]),
        source_digest: Some(source_digest),
        support_coordinate: None,
        supported: false,
        support_status: None,
    };
    let identification = IdentificationLayer::NonGraphical(Arc::new(
        NonGraphicalIdentification::SharedEvidenceContract { contract: query },
    ));
    let statistical = StatisticalProgram {
        identifier: None,
        estimator: None,
        realization: Realization::ExactPoint { evaluator: Arc::from("cohort_decision.affine") },
        models: Arc::from([]),
        folds: None,
        resampling: None,
        posterior: None,
        design,
        operators: Arc::from([]),
    };
    let evidence = EvidenceDecisionProgram {
        sources: Arc::from([EvidenceSourceRef {
            role: EvidenceRole::ChainSource,
            digest: Some(source_digest),
        }]),
        composition: Composition::Chained { stages: bundle.nodes().len() },
        decision: Some(DecisionProgram {
            carrier: CarrierDemand::Mean,
            utility: DecisionProgram::utility_form(None),
            actions: contract.contract().actions.len(),
            constraints: 0,
            chance_threshold: 0.0,
            utility_unit: Some(Arc::from(contract.contract().utility_units.as_str())),
            contract_identity: Some(Arc::from(contract.identity())),
            information_value: None,
            scenario_source: None,
        }),
        planning: None,
        chain: None,
    };
    let physical = PhysicalLayer {
        operators: Arc::from([
            PhysicalOperator::declare(PhysicalEffect::Read)
                .estimates(ByteEstimates::absent().with_output(embedded(source)?.len() as u64))
                .build()
                .map_err(|_| Error::Invalid("cohort source read cannot be lowered"))?,
            PhysicalOperator::declare(PhysicalEffect::Condition)
                .build()
                .map_err(|_| Error::Invalid("affine decision cannot be lowered"))?,
        ]),
        design,
    };
    Ok(StandaloneCompositionLoweredParts {
        cell: ProgramCell::CohortDecisionBundleRecomputation,
        scientific,
        identification,
        statistical,
        evidence,
        physical,
        claim: ClaimDemand { uncertainty: UncertaintyReading::None, kind: None },
    }
    .finish())
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PortableCohortDecisionBundle {
    version: u32,
    required_feature: String,
    bundle_bytes: Vec<u8>,
    receipt: DecisionBundleRecomputationReceipt,
}

impl CompiledCohortDecisionBundleRecomputation {
    /// The inspectable six-layer program.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        &self.lowered
    }

    /// Inspect the compiled route without evaluating the cohort.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        crate::analysis::program_handle::CompiledProgramHandle::plan_view(self)
    }

    /// Replay the graph and score the affine decision from the model and cohort.
    ///
    /// # Errors
    /// A failed dependency, resource stop, or a sealed result that differs from the live result.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<DecisionBundleRecomputationReceipt, Error> {
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        require_program(&scoped, &SupportKey::Unit(COMPOSITION_BUNDLE), COMPILE)?;
        if scoped.cancellation.is_cancelled() {
            return Err(refused(
                BundleStage::ExecutionCancelled,
                "cohort decision replay was cancelled",
                Some(&self.source_node),
            ));
        }
        let limits = BundleLimits::default();
        let bytes = export_bundle(&self.bundle, "cohort-decision-replay")?;
        let consumer = register_standard_verifiers(
            BundleConsumer::new().register(Box::new(CohortDecisionSourceVerifier::new(&scoped))),
        );
        let consumed = consumer.consume(
            &bytes,
            &limits,
            self.bundle.identity(),
            &SuppliedSources::default(),
        )?;
        consumed.require_verified()?;
        if !consumed.all_replayed() {
            return Err(Error::Invalid("every cohort decision dependency must replay"));
        }
        let source_node = by_id(&self.bundle, &self.source_node)?;
        let source_bytes = embedded(source_node)?;
        let _lease = scoped.acquire_serial_numerical_admission(source_bytes.len() as u64).map_err(
            |error| {
                let stage = if scoped.cancellation.is_cancelled() {
                    BundleStage::ExecutionCancelled
                } else {
                    BundleStage::ResourceRefused
                };
                refused(stage, error.to_string(), Some(&self.source_node))
            },
        )?;
        let source = CohortDecisionSource::consume_with(
            source_bytes,
            &CohortExpectation::for_identity(&source_node.identity),
            &scoped,
        )
        .map_err(|error| {
            refused(BundleStage::SwappedEvidence, error.to_string(), Some(&self.source_node))
        })?;
        let terminal = by_id(&self.bundle, &self.terminal_node)?;
        let binding = cohort_mean_binding(&source);
        let stored = DecisionResultArtifact::from_bytes(
            embedded(terminal)?,
            &self.contract_identity,
            &binding,
        )?;
        let live = score(&self.contract, &source)?;
        let fingerprint = decision_result_fingerprint(&live, &binding)?;
        if decision_result_fingerprint(stored.result(), &binding)? != fingerprint {
            return Err(refused(
                BundleStage::TamperedQuantity,
                "the retained decision differs from the live cohort decision",
                Some(&self.terminal_node),
            ));
        }
        Ok(DecisionBundleRecomputationReceipt {
            bundle_identity: consumed.identity().to_owned(),
            terminal_node: terminal.id.clone(),
            terminal_identity: terminal.identity.clone(),
            decision_fingerprint: fingerprint,
            lowered_program_identity: *self.lowered.identity().as_bytes(),
            dependencies: consumed
                .nodes()
                .iter()
                .map(|node| ReplayedBundleDependency {
                    node_id: node.id.clone(),
                    artifact_identity: node.identity.clone(),
                    chain_digest: node.chain_digest.clone(),
                })
                .collect(),
        })
    }

    /// Embed the graph and the independently reproduced receipt.
    ///
    /// # Errors
    /// A failed replay, a changed receipt, or an oversized portable artifact.
    pub fn export_portable(
        &self,
        receipt: &DecisionBundleRecomputationReceipt,
        ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, Error> {
        if self.execute(ctx)? != *receipt {
            return Err(Error::Invalid("cohort decision bundle receipt differs"));
        }
        let bytes = antecedent_io::to_cbor(&PortableCohortDecisionBundle {
            version: 1,
            required_feature: FEATURE.into(),
            bundle_bytes: export_bundle(&self.bundle, "portable-cohort-decision")?,
            receipt: receipt.clone(),
        })?;
        if bytes.len() > MAX_PORTABLE_BYTES {
            return Err(refused(
                BundleStage::Oversized,
                "the portable cohort decision exceeds its byte bound",
                None,
            ));
        }
        Ok(bytes)
    }
}

/// Reconstruct and replay a cohort composition under the retained graph and program identities.
///
/// # Errors
/// Foreign version, graph or program, or a dependency or result that differs on replay.
pub fn consume_portable_cohort_decision_bundle_recomputation(
    bytes: &[u8],
    expected_bundle: &str,
    expected_program: SemanticDigest,
    ctx: &ExecutionContext,
) -> Result<DecisionBundleRecomputationReceipt, Error> {
    require_program(ctx, &SupportKey::Unit(COMPOSITION_BUNDLE), CONSUME)?;
    if bytes.len() > MAX_PORTABLE_BYTES {
        return Err(refused(
            BundleStage::Oversized,
            "the portable cohort decision exceeds its byte bound",
            None,
        ));
    }
    let wire: PortableCohortDecisionBundle = antecedent_io::from_cbor(bytes)?;
    if wire.version != 1 || wire.required_feature != FEATURE {
        return Err(refused(
            BundleStage::IncompatibleVersion,
            "the portable cohort decision version differs",
            None,
        ));
    }
    let bundle = CompositionBundle::from_bytes(
        &wire.bundle_bytes,
        expected_bundle,
        &BundleLimits::default(),
    )?;
    let plan = compile_cohort_decision_bundle_recomputation(bundle, &wire.receipt.terminal_node)?;
    if plan.lowered.identity() != expected_program {
        return Err(Error::Invalid("cohort decision bundle program differs"));
    }
    let receipt = plan.execute(ctx)?;
    if receipt != wire.receipt {
        return Err(Error::Invalid("cohort decision bundle live receipt differs"));
    }
    Ok(receipt)
}

impl crate::analysis::program_handle::CompiledProgramHandle
    for CompiledCohortDecisionBundleRecomputation
{
    fn program(&self) -> &Arc<LoweredProgram> {
        &self.lowered
    }
}
