//! Live predictive and parameter-posterior decision terminals for embedded bundles.
//!
//! The bundle consumer checks every dependency, then this compiler evaluates
//! the contract on the bound aligned source law and compares its complete
//! decision fingerprint with the embedded terminal. A portable consumer
//! reconstructs the same source and contract from the embedded graph.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::DataClassification;
use antecedent_core::{
    ByteEstimates, CanonicalWriter, ExecutionContext, PhysicalEffect, PhysicalOperator,
    SemanticDigest,
};
use antecedent_design::decision_artifact::{
    DecisionContractArtifact, DecisionResultArtifact, decision_result_fingerprint, source_digest,
};
use antecedent_design::decision_contract::{
    DecisionContract, DecisionContractError, DecisionCriterion,
};
use antecedent_design::decision_eval::evaluate_contract;
use antecedent_io::IoError;
use antecedent_io::distribution_artifact::{DistributionArtifact, DistributionIdentity};
use antecedent_io::quantity_wire::DistributionMeaningWire;
use serde::{Deserialize, Serialize};

use super::composition::{
    BundleError, BundleLimits, CompositionBundle, NodeKind, NodeSource, ReplayedBundleDependency,
    SuppliedSources, consume, export_bundle,
};
use crate::planner::lowered::{
    CarrierDemand, ClaimDemand, Composition, DecisionProgram, DesignShape, EvidenceDecisionProgram,
    EvidenceRole, EvidenceSourceRef, FunctionalKind, IdentificationLayer, LoweredProgram,
    NonGraphicalIdentification, PhysicalLayer, ProblemTarget, ProgramCell, ProgramIdentityScope,
    ScientificProblem, StandaloneCompositionLoweredParts, UncertaintyReading,
};
use crate::planner::statistical::{Realization, StatisticalProgram};
use crate::planner::{PlanStage, PlanView};

const FEATURE: &str = "decision_bundle_recomputation_portable_v1";
const POSTERIOR_FEATURE: &str = "posterior_decision_bundle_recomputation_portable_v1";
const MAX_BYTES: usize = 32 * 1024 * 1024;

/// A bundle whose decision terminal is recomputed from its compiler-bound
/// contract and aligned predictive or parameter-posterior distribution.
#[derive(Clone, Debug)]
pub struct CompiledDecisionBundleRecomputation {
    bundle: CompositionBundle,
    terminal_node: String,
    contract: DecisionContract,
    source: DistributionArtifact,
    contract_identity: String,
    source_digest: String,
    lowered: Arc<LoweredProgram>,
    posterior: bool,
}

/// A live decision fingerprint and every independently replayed dependency.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionBundleRecomputationReceipt {
    /// Independently consumed graph identity.
    pub bundle_identity: String,
    /// Embedded decision result node.
    pub terminal_node: String,
    /// Independently checked result artifact identity.
    pub terminal_identity: String,
    /// Fingerprint of the freshly evaluated decision.
    pub decision_fingerprint: String,
    /// Identity of the outer six-layer compiler program.
    pub lowered_program_identity: [u8; 32],
    /// Every embedded dependency replayed by its own consumer.
    pub dependencies: Vec<ReplayedBundleDependency>,
}

/// Refusal of the bundle, carrier, decision or portable replay.
#[derive(Debug, thiserror::Error)]
pub enum DecisionBundleRecomputeError {
    /// Bundle structure or embedded dependency refused.
    #[error("decision bundle: {0}")]
    Bundle(#[from] BundleError),
    /// Contract or carrier requirements refused.
    #[error("decision bundle contract: {0}")]
    Contract(#[from] DecisionContractError),
    /// Live decision evaluation refused.
    #[error("decision bundle evaluation: {0}")]
    Evaluation(String),
    /// Artifact encoding, decoding or replay refused.
    #[error("decision bundle artifact: {0}")]
    Io(#[from] IoError),
    /// An embedded binding or live result differs.
    #[error("decision bundle: {0}")]
    Invalid(&'static str),
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PortableDecisionBundleWire {
    version: u32,
    required_feature: String,
    bundle_bytes: Vec<u8>,
    bundle_identity: String,
    terminal_node: String,
    contract_identity: String,
    source_identity: DistributionIdentity,
    source_digest: String,
    receipt: DecisionBundleRecomputationReceipt,
}

/// Compile a predictive decision terminal with its complete embedded graph.
/// The contract, source metadata and joint draws, graph and node all enter a
/// separate six-layer identity before any utility is scored.
///
/// # Errors
/// Missing terminal, invalid contract or unsupported source-law meaning.
#[allow(clippy::too_many_lines, reason = "all six layers bind one predictive decision terminal")]
pub fn compile_decision_bundle_recomputation(
    bundle: CompositionBundle,
    terminal_node: &str,
    contract: DecisionContract,
    source: DistributionArtifact,
) -> Result<CompiledDecisionBundleRecomputation, DecisionBundleRecomputeError> {
    compile_decision_bundle(bundle, terminal_node, contract, source, false)
}

/// Compile a nonpredictive decision over a parameter posterior. Outcome-risk
/// coordinates and identified-set decisions are refused.
///
/// # Errors
/// Missing terminal, invalid contract or unsupported posterior carrier.
pub fn compile_posterior_decision_bundle_recomputation(
    bundle: CompositionBundle,
    terminal_node: &str,
    contract: DecisionContract,
    source: DistributionArtifact,
) -> Result<CompiledDecisionBundleRecomputation, DecisionBundleRecomputeError> {
    compile_decision_bundle(bundle, terminal_node, contract, source, true)
}

fn compile_decision_bundle(
    bundle: CompositionBundle,
    terminal_node: &str,
    contract: DecisionContract,
    source: DistributionArtifact,
    posterior: bool,
) -> Result<CompiledDecisionBundleRecomputation, DecisionBundleRecomputeError> {
    let node = bundle
        .nodes()
        .iter()
        .find(|node| node.id == terminal_node)
        .ok_or(DecisionBundleRecomputeError::Invalid("decision terminal is absent"))?;
    if node.kind != NodeKind::DecisionResult || !matches!(&node.source, NodeSource::Embedded { .. })
    {
        return Err(DecisionBundleRecomputeError::Invalid(
            "decision terminal must be an embedded decision result",
        ));
    }
    if posterior {
        if !matches!(source.semantic(), DistributionMeaningWire::ParameterPosterior)
            || !matches!(
                contract.criterion,
                DecisionCriterion::PosteriorExpectedUtility
                    | DecisionCriterion::PosteriorExpectedLoss
            )
            || contract
                .actions
                .iter()
                .flat_map(|action| &action.inputs)
                .any(|quantity| quantity.functional_id == "outcome")
        {
            return Err(DecisionBundleRecomputeError::Invalid(
                "posterior decision requires parameter draws and a non-outcome expected utility or loss",
            ));
        }
    } else if !matches!(
        source.semantic(),
        DistributionMeaningWire::PosteriorPredictive
            | DistributionMeaningWire::InterventionalPredictive
            | DistributionMeaningWire::EmpiricalOutcome
    ) {
        return Err(DecisionBundleRecomputeError::Invalid(
            "live decision terminal requires a predictive outcome law",
        ));
    }
    if contract.source_requirement()?.is_some_and(|required| required.is_identified_set()) {
        return Err(DecisionBundleRecomputeError::Invalid(
            "an identified-set decision requires a separate scenario carrier",
        ));
    }
    let contract_identity = contract.identity()?;
    let source_digest = source_digest(&source);
    let source_bytes = source.to_bytes("decision-bundle-source")?;
    let bundle_digest = CanonicalWriter::digest_bytes(
        "antecedent.decision_bundle.graph.v1",
        bundle.identity().as_bytes(),
    );
    let source_identity_digest =
        CanonicalWriter::digest_bytes("antecedent.decision_bundle.source.v1", &source_bytes);
    let mut writer = CanonicalWriter::new("antecedent.decision_bundle.question.v1");
    writer
        .bytes(&bundle_digest)
        .text(terminal_node)
        .text(&contract_identity)
        .text(&source_digest)
        .bytes(&source_identity_digest);
    let query = writer.finish();
    let design =
        DesignShape::quoted(source.n_draws() as u64, source.quantities().len() as u64, true);
    let scientific = ScientificProblem {
        runtime_query: None,
        identity_scope: ProgramIdentityScope::Semantic,
        target: ProblemTarget::QueryFunctional {
            kind: FunctionalKind::PropagatedChain,
            outcome: None,
            query,
        },
        plan_id: Arc::from(if posterior {
            "posterior_decision_bundle_recomputation.v1"
        } else {
            "decision_bundle_recomputation.v1"
        }),
        data: DataClassification::Tabular,
        graph_class: "bundle_decision_dependency_graph",
        query_variables: Arc::from(Vec::new()),
        source_digest: Some(source_identity_digest),
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
        realization: if posterior {
            Realization::DeclaredPosterior {
                evaluator: Arc::from("decision_bundle.live_parameter_posterior_terminal"),
            }
        } else {
            Realization::DeclaredPredictive {
                evaluator: Arc::from("decision_bundle.live_predictive_terminal"),
            }
        },
        models: Arc::from([]),
        folds: None,
        resampling: None,
        posterior: None,
        design,
        operators: Arc::from([]),
    };
    let decision = DecisionProgram {
        carrier: if posterior {
            CarrierDemand::ParameterPosterior
        } else {
            CarrierDemand::PredictiveLaw
        },
        utility: DecisionProgram::utility_form(None),
        actions: contract.actions.len(),
        constraints: contract.constraints.len(),
        chance_threshold: contract
            .constraints
            .iter()
            .map(|constraint| constraint.min_probability)
            .fold(0.0_f64, f64::max),
        utility_unit: Some(Arc::from(contract.utility_units.as_str())),
        contract_identity: Some(Arc::from(contract_identity.as_str())),
        information_value: None,
        scenario_source: None,
    };
    let evidence = EvidenceDecisionProgram {
        sources: Arc::from([
            EvidenceSourceRef {
                role: EvidenceRole::ChainSource,
                digest: Some(source_identity_digest),
            },
            EvidenceSourceRef { role: EvidenceRole::TransportSource, digest: Some(bundle_digest) },
        ]),
        composition: Composition::Chained { stages: bundle.nodes().len() + 1 },
        decision: Some(decision),
        planning: None,
        chain: None,
    };
    let physical = PhysicalLayer {
        operators: Arc::from([
            PhysicalOperator::declare(PhysicalEffect::Read)
                .estimates(ByteEstimates::absent().with_output(source_bytes.len() as u64))
                .build()
                .map_err(|_| {
                    DecisionBundleRecomputeError::Invalid("decision source read cannot be lowered")
                })?,
            PhysicalOperator::declare(PhysicalEffect::Condition).build().map_err(|_| {
                DecisionBundleRecomputeError::Invalid("decision scoring cannot be lowered")
            })?,
        ]),
        design,
    };
    let lowered = Arc::new(
        StandaloneCompositionLoweredParts {
            cell: if posterior {
                ProgramCell::PosteriorDecisionBundleRecomputation
            } else {
                ProgramCell::BundleRecomputation
            },
            scientific,
            identification,
            statistical,
            evidence,
            physical,
            claim: ClaimDemand {
                uncertainty: if posterior {
                    UncertaintyReading::Posterior
                } else {
                    UncertaintyReading::Predictive
                },
                kind: None,
            },
        }
        .finish(),
    );
    Ok(CompiledDecisionBundleRecomputation {
        bundle,
        terminal_node: terminal_node.to_owned(),
        contract,
        source,
        contract_identity,
        source_digest,
        lowered,
        posterior,
    })
}

impl CompiledDecisionBundleRecomputation {
    /// Complete bundle and decision terminal program.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram {
        &self.lowered
    }

    /// Inspect the compiled route without scoring its decision.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        PlanView::of_program(Arc::clone(&self.lowered), PlanStage::Compiled)
    }

    /// Replay every embedded dependency, score the decision and
    /// compare its whole result with the embedded terminal.
    ///
    /// # Errors
    /// Failed dependency, changed source/contract or changed utility result.
    pub fn execute(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<DecisionBundleRecomputationReceipt, DecisionBundleRecomputeError> {
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        if scoped.cancellation.is_cancelled() {
            return Err(DecisionBundleRecomputeError::Invalid("decision bundle was cancelled"));
        }
        let bytes = export_bundle(&self.bundle, "decision-bundle-recomputation")?;
        let consumed = consume(&bytes, self.bundle.identity(), &SuppliedSources::default())?;
        if !consumed.all_replayed() {
            return Err(DecisionBundleRecomputeError::Invalid(
                "every decision bundle dependency must replay",
            ));
        }
        let node = self
            .bundle
            .nodes()
            .iter()
            .find(|node| node.id == self.terminal_node)
            .ok_or(DecisionBundleRecomputeError::Invalid("decision terminal disappeared"))?;
        let NodeSource::Embedded { bytes: terminal_bytes } = &node.source else {
            return Err(DecisionBundleRecomputeError::Invalid("decision terminal is a reference"));
        };
        let stored = DecisionResultArtifact::from_bytes(
            terminal_bytes,
            &self.contract_identity,
            &self.source_digest,
        )?;
        stored.replay(&self.contract, &self.source)?;
        let live = evaluate_contract(&self.contract, &self.source)
            .map_err(|error| DecisionBundleRecomputeError::Evaluation(error.to_string()))?;
        let fingerprint = decision_result_fingerprint(&live, &self.source_digest)?;
        if decision_result_fingerprint(stored.result(), &self.source_digest)? != fingerprint {
            return Err(DecisionBundleRecomputeError::Invalid(
                "embedded decision differs from live compiler scoring",
            ));
        }
        let dependencies = consumed
            .nodes()
            .iter()
            .map(|node| ReplayedBundleDependency {
                node_id: node.id.clone(),
                artifact_identity: node.identity.clone(),
                chain_digest: node.chain_digest.clone(),
            })
            .collect();
        Ok(DecisionBundleRecomputationReceipt {
            bundle_identity: consumed.identity().to_owned(),
            terminal_node: self.terminal_node.clone(),
            terminal_identity: node.identity.clone(),
            decision_fingerprint: fingerprint,
            lowered_program_identity: *self.lowered.identity().as_bytes(),
            dependencies,
        })
    }

    /// Embed the full graph and decision receipt for another process.
    ///
    /// # Errors
    /// Changed live decision or oversized portable artifact.
    pub fn export_portable(
        &self,
        receipt: &DecisionBundleRecomputationReceipt,
        ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, DecisionBundleRecomputeError> {
        if self.execute(ctx)? != *receipt {
            return Err(DecisionBundleRecomputeError::Invalid("decision bundle receipt differs"));
        }
        let bytes = antecedent_io::to_cbor(&PortableDecisionBundleWire {
            version: 1,
            required_feature: if self.posterior { POSTERIOR_FEATURE } else { FEATURE }.into(),
            bundle_bytes: export_bundle(&self.bundle, "portable-decision-bundle")?,
            bundle_identity: self.bundle.identity().to_owned(),
            terminal_node: self.terminal_node.clone(),
            contract_identity: self.contract_identity.clone(),
            source_identity: self.source.metadata().identity.clone(),
            source_digest: self.source_digest.clone(),
            receipt: receipt.clone(),
        })?;
        if bytes.len() > MAX_BYTES {
            return Err(DecisionBundleRecomputeError::Invalid(
                "portable decision bundle exceeds its byte bound",
            ));
        }
        Ok(bytes)
    }
}

/// Reconstruct contract and predictive source from the embedded graph,
/// recompile its outer program and compare the live terminal receipt.
///
/// # Errors
/// Foreign feature, graph, source, contract or program; failed dependency or
/// changed decision value.
#[allow(clippy::too_many_lines, reason = "the portable consumer reconstructs each bound node")]
pub fn consume_portable_decision_bundle_recomputation(
    bytes: &[u8],
    expected_bundle: &str,
    expected_program: SemanticDigest,
    ctx: &ExecutionContext,
) -> Result<DecisionBundleRecomputationReceipt, DecisionBundleRecomputeError> {
    consume_portable_decision_bundle(bytes, expected_bundle, expected_program, ctx, false)
}

/// Independently reconstruct and score a parameter-posterior decision bundle.
///
/// # Errors
/// Changed graph, carrier, program, dependency or result.
pub fn consume_portable_posterior_decision_bundle_recomputation(
    bytes: &[u8],
    expected_bundle: &str,
    expected_program: SemanticDigest,
    ctx: &ExecutionContext,
) -> Result<DecisionBundleRecomputationReceipt, DecisionBundleRecomputeError> {
    consume_portable_decision_bundle(bytes, expected_bundle, expected_program, ctx, true)
}

fn consume_portable_decision_bundle(
    bytes: &[u8],
    expected_bundle: &str,
    expected_program: SemanticDigest,
    ctx: &ExecutionContext,
    posterior: bool,
) -> Result<DecisionBundleRecomputationReceipt, DecisionBundleRecomputeError> {
    if bytes.len() > MAX_BYTES {
        return Err(DecisionBundleRecomputeError::Invalid(
            "portable decision bundle exceeds its byte bound",
        ));
    }
    let wire: PortableDecisionBundleWire = antecedent_io::from_cbor(bytes)?;
    let expected_feature = if posterior { POSTERIOR_FEATURE } else { FEATURE };
    if wire.version != 1
        || wire.required_feature != expected_feature
        || wire.bundle_identity != expected_bundle
    {
        return Err(DecisionBundleRecomputeError::Invalid(
            "portable decision bundle feature differs",
        ));
    }
    let bundle = CompositionBundle::from_bytes(
        &wire.bundle_bytes,
        expected_bundle,
        &BundleLimits::default(),
    )?;
    let contract_nodes: Vec<_> = bundle
        .nodes()
        .iter()
        .filter(|node| {
            node.kind == NodeKind::DecisionContract && node.identity == wire.contract_identity
        })
        .collect();
    if contract_nodes.len() != 1 {
        return Err(DecisionBundleRecomputeError::Invalid(
            "portable decision contract is absent or ambiguous",
        ));
    }
    let NodeSource::Embedded { bytes: contract_bytes } = &contract_nodes[0].source else {
        return Err(DecisionBundleRecomputeError::Invalid(
            "portable decision contract is a reference",
        ));
    };
    let contract = DecisionContractArtifact::from_bytes(contract_bytes, &wire.contract_identity)?
        .contract()
        .clone();
    let mut matched_source = None;
    for node in bundle.nodes().iter().filter(|node| node.kind == NodeKind::Distribution) {
        if let NodeSource::Embedded { bytes } = &node.source
            && let Ok(source) = DistributionArtifact::from_bytes(bytes, &wire.source_identity)
            && source_digest(&source) == wire.source_digest
        {
            if matched_source.replace(source).is_some() {
                return Err(DecisionBundleRecomputeError::Invalid(
                    "portable decision source is ambiguous",
                ));
            }
        }
    }
    let source = matched_source.ok_or(DecisionBundleRecomputeError::Invalid(
        "portable predictive decision source is absent",
    ))?;
    let plan = compile_decision_bundle(bundle, &wire.terminal_node, contract, source, posterior)?;
    if plan.lowered().identity() != expected_program {
        return Err(DecisionBundleRecomputeError::Invalid(
            "portable decision bundle compiler differs",
        ));
    }
    let live = plan.execute(ctx)?;
    if live != wire.receipt {
        return Err(DecisionBundleRecomputeError::Invalid(
            "portable decision bundle live receipt differs",
        ));
    }
    Ok(live)
}
