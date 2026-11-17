//! Composition facade: build, export and independently consume a portable composed result.
//!
//! A [`BundleBuilder`] collects the parts of a composed decision as nodes of a
//! typed graph: artifacts embedded from their bytes (a distribution, an external
//! claim, a decision contract and result, a sensitivity artifact, a study ranking,
//! an inverse query, a repair report), references to a provider or data source the
//! consumer must supply, dependency edges, and declared evidence relationships.
//! [`BundleBuilder::build`] seals the graph into a
//! [`CompositionBundle`]; [`consume`] decodes exported bytes in another process,
//! verifies every embedded artifact through its own consumer and reports each node
//! as verified, reference-unresolved or failed at a named stage.
//!
//! Native fitted nodes use context-bound verifiers to recompile and rerun their
//! portable source before a live bundle terminal is accepted. A reference node is never claimed
//! replayed, a decision resting on an external mean is labelled point-only
//! attested, and a decision that needs an aligned joint law is refused over an
//! upstream mean-only claim. The bundle identity the consumer passes to
//! [`consume`] is retained independently of the bytes.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub use antecedent_design::composition_bundle::{
    BundleConsumer, BundleEdge, BundleError, BundleLimits, BundleNode, BundleStage, ClaimLabel,
    CompositionBundle, ConsumedBundle, ConsumedNode, EvidenceRelationship, FACT_EVIDENCE_DIGEST,
    FACT_LAW, FACT_QUANTITY_DIGEST, FACT_REQUEST, FACT_REQUIRES_LAW, FACT_SNAPSHOT, FACT_TRUST,
    LAW_JOINT, LAW_MEAN_ONLY, NodeFailure, NodeKind, NodeSource, NodeStatus, ProviderOrData,
    SuppliedData, SuppliedProvider, SuppliedSources, NodeVerifier, UpstreamNode, VerifiedNode,
};
pub use antecedent_design::composition_verifiers::{
    ArtifactDescription, FACT_COORDINATES, FACT_MEAN_SOURCE, FACT_PROVIDER_BINDING,
    FACT_REQUIRES_DISTRIBUTION, FACT_SOURCE_DIGEST, LAW_MARGINAL, TRUST_ATTESTED, TRUST_NATIVE,
    TRUST_UNVERIFIED, TRUST_VERIFIED, VERIFIABLE_KINDS, describe_artifact, detect_node_kind,
    evidence_relationship_bytes, mean_decision_bytes, mean_source_of, standard_consumer,
};
pub use antecedent_design::decision_artifact::mean_result_to_bytes;

use std::sync::Arc;
use antecedent_contract::DataClassification;
use antecedent_contract::bridge::{BridgeContract, ValueKind};
use antecedent_core::{
    ByteEstimates, CanonicalWriter, ExecutionContext, PhysicalEffect,
    PhysicalOperator, SemanticDigest,
};
use antecedent_contract::propagation_chain::ChainRefusal;
use antecedent_io::propagation_chain_record::PropagationChainRecordWire;
use antecedent_io::IoError;
use antecedent_io::distribution_artifact::DistributionArtifact;
use antecedent_io::quantity_wire::{DistributionMeaningWire, ScientificQuantityWire};
use serde::{Deserialize, Serialize};
use super::propagation_chain::{PropagationChainPlan, compile_propagation_chain};
use super::scientific_bridge::{
    PreparedScientificBridgeDistribution, ScientificBridgeRefusal,
    consume_scientific_bridge_distribution_artifact,
};
use crate::planner::lowered::{
    ClaimDemand, Composition, DesignShape, EvidenceDecisionProgram,
    EvidenceRole, EvidenceSourceRef,
    FunctionalKind, IdentificationLayer, LoweredProgram,
    NonGraphicalIdentification, PhysicalLayer, ProblemTarget, ProgramCell,
    ProgramIdentityScope, ScientificProblem, StandaloneCompositionLoweredParts,
    UncertaintyReading,
};
use crate::planner::statistical::{NestedFitLaw, Realization, StatisticalProgram};
use crate::planner::{PlanStage, PlanView};

fn refused(stage: BundleStage, reason: &str, offending: Option<&str>) -> BundleError {
    BundleError::Refused {
        stage,
        reason: reason.to_owned(),
        offending: offending.map(str::to_owned),
    }
}

fn graph_refused(reason: &str, offending: Option<&str>) -> BundleError {
    refused(BundleStage::EdgeDigestMismatch, reason, offending)
}

fn node_refused(id: &str, failure: &NodeFailure) -> BundleError {
    refused(failure.stage, &failure.reason, Some(id))
}

/// Collects the nodes and dependency edges of a composed result.
///
/// Node ids are chosen by the caller or derived from the artifact itself
/// (`<kind>:<first 12 hex of its recomputed identity>`), so a bundle built from the
/// same parts has the same identity whatever order they were added in.
#[derive(Clone, Debug)]
pub struct BundleBuilder {
    nodes: Vec<BundleNode>,
    dependencies: Vec<(String, String)>,
    limits: BundleLimits,
}

impl Default for BundleBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl BundleBuilder {
    /// An empty builder under the default bounds.
    #[must_use]
    pub fn new() -> Self {
        Self { nodes: Vec::new(), dependencies: Vec::new(), limits: BundleLimits::default() }
    }

    /// The bounds the bundle is built and exported under.
    #[must_use]
    pub fn limits(&self) -> &BundleLimits {
        &self.limits
    }

    /// Ids of the nodes added so far, in insertion order.
    #[must_use]
    pub fn node_ids(&self) -> Vec<&str> {
        self.nodes.iter().map(|n| n.id.as_str()).collect()
    }

    fn position(&self, id: &str) -> Option<usize> {
        self.nodes.iter().position(|n| n.id == id)
    }

    fn ensure_new(&self, id: &str) -> Result<(), BundleError> {
        if id.trim().is_empty() {
            return Err(graph_refused("a node id is blank", Some(id)));
        }
        if self.position(id).is_some() {
            return Err(graph_refused("a node id is repeated", Some(id)));
        }
        Ok(())
    }

    fn ensure_known(&self, id: &str) -> Result<usize, BundleError> {
        self.position(id).ok_or_else(|| graph_refused("the node is not in the bundle", Some(id)))
    }

    /// Embed an artifact of the given kind from its container bytes. The artifact
    /// is decoded through its own consumer and its identity recomputed, so bytes
    /// that are not that kind of artifact refuse here. Returns the node id.
    ///
    /// # Errors
    /// An artifact its own consumer refuses (with that stage), a blank or repeated
    /// id, or an embedded artifact over the bundle bound.
    pub fn add_artifact(
        &mut self,
        node_id: Option<&str>,
        kind: NodeKind,
        bytes: &[u8],
    ) -> Result<String, BundleError> {
        if bytes.len() > self.limits.max_embedded_bytes {
            return Err(refused(
                BundleStage::Oversized,
                "an embedded artifact exceeds its bound",
                node_id,
            ));
        }
        let described = describe_artifact(kind, bytes)
            .map_err(|failure| node_refused(node_id.unwrap_or(kind.as_str()), &failure))?;
        let id = node_id.map_or_else(
            || {
                let short: String = described.identity.chars().take(12).collect();
                format!("{}:{short}", kind.as_str())
            },
            str::to_owned,
        );
        self.ensure_new(&id)?;
        self.nodes.push(BundleNode::embedded(&id, kind, &described.identity, bytes.to_vec()));
        Ok(id)
    }

    /// Add a versioned native cyclic posterior producer node. Its artifact is
    /// independently recompiled and refitted before the node is accepted.
    ///
    /// # Errors
    /// Changed producer program, failed fit, repeated id or a size bound.
    pub fn add_cyclic_posterior_fit(
        &mut self, node_id: &str, artifact: &[u8], program: SemanticDigest,
        ctx: &ExecutionContext,
    ) -> Result<(), CyclicPosteriorBundleError> {
        self.ensure_new(node_id).map_err(ChainBundleRecomputeError::Bundle)?;
        let bytes = antecedent_io::to_cbor(&CyclicPosteriorFitWire {
            version: 2, required_feature: CYCLIC_POSTERIOR_FIT_FEATURE_V2.into(),
            replay_seed: Some(ctx.rng.master_seed()), producer_program: *program.as_bytes(), artifact: artifact.to_vec(),
        }).map_err(ChainBundleRecomputeError::Io)?;
        if bytes.len() > self.limits.max_embedded_bytes {
            return Err(CyclicPosteriorBundleError::Invalid("fit node exceeds its embedded bound"));
        }
        let inspected = CyclicPosteriorFitVerifier { ctx: ctx.clone() }
            .verify(node_id, &bytes, &[])
            .map_err(|failure| ChainBundleRecomputeError::Bundle(node_refused(node_id, &failure)))?;
        self.nodes.push(BundleNode::embedded(node_id, NodeKind::CyclicPosteriorFit,
            &inspected.identity, bytes));
        Ok(())
    }

    /// Add an independently reidentified and refitted trial-IPW producer.
    ///
    /// # Errors
    /// The producer replay, graph id or embedded byte limit refuses.
    pub fn add_trial_ipw_transport_fit(
        &mut self, node_id: &str, artifact: &[u8], program: SemanticDigest,
        ctx: &ExecutionContext,
    ) -> Result<(), TrialIpwBundleError> {
        self.ensure_new(node_id).map_err(ChainBundleRecomputeError::Bundle)?;
        let bytes = antecedent_io::to_cbor(&TrialIpwFitWire {
            version: 2, required_feature: TRIAL_IPW_FIT_FEATURE_V2.into(),
            replay_seed: Some(ctx.rng.master_seed()), producer_program: *program.as_bytes(), artifact: artifact.to_vec(),
        }).map_err(ChainBundleRecomputeError::Io)?;
        if bytes.len() > self.limits.max_embedded_bytes {
            return Err(TrialIpwBundleError::Invalid("trial fit exceeds its embedded bound"));
        }
        let inspected = TrialIpwFitVerifier { ctx: ctx.clone() }
            .verify(node_id, &bytes, &[])
            .map_err(|failure| ChainBundleRecomputeError::Bundle(node_refused(node_id, &failure)))?;
        self.nodes.push(BundleNode::embedded(node_id, NodeKind::TrialIpwTransportFit,
            &inspected.identity, bytes));
        Ok(())
    }

    /// Add an independently reidentified and refitted graph-imputation transport producer.
    ///
    /// # Errors
    /// The producer replay, graph id or embedded byte limit refuses.
    pub fn add_graph_imputation_transport_fit(
        &mut self, node_id: &str, artifact: &[u8], program: SemanticDigest,
        ctx: &ExecutionContext,
    ) -> Result<(), GraphImputationTransportBundleError> {
        self.ensure_new(node_id).map_err(ChainBundleRecomputeError::Bundle)?;
        let bytes = antecedent_io::to_cbor(&GraphImputationTransportFitWire {
            version: 2, required_feature: GRAPH_IMPUTATION_TRANSPORT_FIT_FEATURE_V2.into(),
            replay_seed: Some(ctx.rng.master_seed()), producer_program: *program.as_bytes(), artifact: artifact.to_vec(),
        }).map_err(ChainBundleRecomputeError::Io)?;
        if bytes.len() > self.limits.max_embedded_bytes {
            return Err(GraphImputationTransportBundleError::Invalid("graph transport fit exceeds its embedded bound"));
        }
        let inspected = GraphImputationTransportFitVerifier { ctx: ctx.clone() }
            .verify(node_id, &bytes, &[])
            .map_err(|failure| ChainBundleRecomputeError::Bundle(node_refused(node_id, &failure)))?;
        self.nodes.push(BundleNode::embedded(node_id, NodeKind::GraphImputationTransportFit,
            &inspected.identity, bytes));
        Ok(())
    }

    /// Add an independently reidentified and refitted joint transport effect producer.
    ///
    /// # Errors
    /// The producer replay, graph id or embedded byte limit refuses.
    pub fn add_joint_transport_effect_fit(
        &mut self, node_id: &str, artifact: &[u8], program: SemanticDigest,
        ctx: &ExecutionContext,
    ) -> Result<(), JointTransportEffectBundleError> {
        self.ensure_new(node_id).map_err(ChainBundleRecomputeError::Bundle)?;
        let bytes = antecedent_io::to_cbor(&JointTransportEffectFitWire {
            version: 2, required_feature: JOINT_TRANSPORT_EFFECT_FIT_FEATURE_V2.into(),
            replay_seed: Some(ctx.rng.master_seed()), producer_program: *program.as_bytes(), artifact: artifact.to_vec(),
        }).map_err(ChainBundleRecomputeError::Io)?;
        if bytes.len() > self.limits.max_embedded_bytes {
            return Err(JointTransportEffectBundleError::Invalid("joint transport effect fit exceeds its embedded bound"));
        }
        let inspected = JointTransportEffectFitVerifier { ctx: ctx.clone() }
            .verify(node_id, &bytes, &[])
            .map_err(|failure| ChainBundleRecomputeError::Bundle(node_refused(node_id, &failure)))?;
        self.nodes.push(BundleNode::embedded(node_id, NodeKind::JointTransportEffectFit,
            &inspected.identity, bytes));
        Ok(())
    }

    /// Embed an artifact whose kind is read from its container.
    ///
    /// # Errors
    /// A container that is not a bundle artifact (`unknown_node_kind`), or any
    /// refusal of [`Self::add_artifact`].
    pub fn add_detected_artifact(
        &mut self,
        node_id: Option<&str>,
        bytes: &[u8],
    ) -> Result<String, BundleError> {
        let Some(kind) = detect_node_kind(bytes) else {
            return Err(refused(
                BundleStage::UnknownNodeKind,
                "the bytes are not a container of a known artifact kind",
                node_id,
            ));
        };
        self.add_artifact(node_id, kind, bytes)
    }

    /// Add a node that is not embedded: an artifact held elsewhere that needs a
    /// supplied provider (at an exact request) or data source, and is never claimed
    /// replayed.
    ///
    /// # Errors
    /// A blank or repeated id.
    pub fn add_reference(
        &mut self,
        node_id: &str,
        kind: NodeKind,
        identity: &str,
        requires: ProviderOrData,
    ) -> Result<(), BundleError> {
        self.ensure_new(node_id)?;
        self.nodes.push(BundleNode::reference(node_id, kind, identity, requires));
        Ok(())
    }

    fn update_reference(
        &mut self,
        node_id: &str,
        change: impl FnOnce(BundleNode) -> BundleNode,
    ) -> Result<(), BundleError> {
        let position = self.ensure_known(node_id)?;
        if !matches!(self.nodes[position].source, NodeSource::Reference { .. }) {
            return Err(graph_refused(
                "facts and inspected values are declared on reference nodes only",
                Some(node_id),
            ));
        }
        let node = self.nodes.remove(position);
        self.nodes.insert(position, change(node));
        Ok(())
    }

    /// Declare an (unverified) fact of a reference node, such as its law or its
    /// snapshot, so the bundle can cross-check it against connected nodes.
    ///
    /// # Errors
    /// An unknown node, or an embedded one (it publishes its own facts).
    pub fn declare_fact(
        &mut self,
        node_id: &str,
        key: &str,
        value: &str,
    ) -> Result<(), BundleError> {
        self.update_reference(node_id, |node| node.with_fact(key, value))
    }

    /// Keep a declared, unverified value of a reference node, so an inspected
    /// result survives when the reference cannot be resolved.
    ///
    /// # Errors
    /// An unknown node, or an embedded one.
    pub fn declare_inspected(
        &mut self,
        node_id: &str,
        key: &str,
        value: f64,
    ) -> Result<(), BundleError> {
        self.update_reference(node_id, |node| node.with_inspected(key, value))
    }

    /// Declare that `dependent` was derived from `upstream`. The upstream node's
    /// Merkle digest then enters the dependent node's digest and the edge.
    ///
    /// # Errors
    /// An unknown node, a self-dependency or a repeated edge.
    pub fn connect(&mut self, upstream: &str, dependent: &str) -> Result<(), BundleError> {
        self.ensure_known(upstream)?;
        self.ensure_known(dependent)?;
        if upstream == dependent {
            return Err(graph_refused("a node depends on itself", Some(upstream)));
        }
        let edge = (upstream.to_owned(), dependent.to_owned());
        if self.dependencies.contains(&edge) {
            return Err(graph_refused("an edge is repeated", Some(upstream)));
        }
        self.dependencies.push(edge);
        Ok(())
    }

    /// Declare how the evidence behind two nodes depends on each other. The
    /// declaration is an embedded node both are connected to, so it enters the
    /// Merkle chain and is checked against the nodes it names (a pair declared
    /// independent that rests on the same evidence is refused as swapped evidence).
    /// The pair is unordered. Returns the declaration node's id.
    ///
    /// # Errors
    /// An unknown or identical pair, or a pair already related.
    pub fn relate(
        &mut self,
        left: &str,
        right: &str,
        relationship: EvidenceRelationship,
    ) -> Result<String, BundleError> {
        self.ensure_known(left)?;
        self.ensure_known(right)?;
        if left == right {
            return Err(graph_refused("a relationship names two distinct nodes", Some(left)));
        }
        let (first, second) = if left <= right { (left, right) } else { (right, left) };
        let id = format!("relation:{first}:{second}");
        self.ensure_new(&id)?;
        let bytes = evidence_relationship_bytes(relationship, first, second);
        self.add_artifact(Some(&id), NodeKind::EvidenceRelationship, &bytes)?;
        self.connect(first, &id)?;
        self.connect(second, &id)?;
        Ok(id)
    }

    /// Seal the graph into a bundle.
    ///
    /// # Errors
    /// A bound exceeded (`oversized`), or a dangling edge or a cycle
    /// (`edge_digest_mismatch`).
    pub fn build(&self) -> Result<CompositionBundle, BundleError> {
        CompositionBundle::new(self.nodes.clone(), &self.dependencies, &self.limits)
    }
}

/// Serialize a bundle through the checksummed container under the default bounds.
///
/// # Errors
/// A blank artifact id or a result over the bound.
pub fn export_bundle(
    bundle: &CompositionBundle,
    artifact_id: &str,
) -> Result<Vec<u8>, BundleError> {
    bundle.to_bytes(artifact_id, &BundleLimits::default())
}

/// Consume exported bundle bytes under the default bounds with the standard
/// verifiers, requiring the bundle identity the consumer retained independently of
/// the bytes.
///
/// # Errors
/// An oversized or corrupt container, an unsupported version, an unknown node kind,
/// an edge digest that differs from its recomputation, or an identity different
/// from `expected_identity`. A node-level failure does not refuse the bundle; it is
/// reported on that node.
pub fn consume(
    bytes: &[u8],
    expected_identity: &str,
    supplied: &SuppliedSources,
) -> Result<ConsumedBundle, BundleError> {
    consume_with_limits(bytes, &BundleLimits::default(), expected_identity, supplied)
}

/// [`consume`] under explicit bounds.
///
/// # Errors
/// As [`consume`].
pub fn consume_with_limits(
    bytes: &[u8],
    limits: &BundleLimits,
    expected_identity: &str,
    supplied: &SuppliedSources,
) -> Result<ConsumedBundle, BundleError> {
    standard_consumer().consume(bytes, limits, expected_identity, supplied)
}

const BUNDLE_RECOMPUTE_FEATURE: &str = "chain_bundle_recomputation_v1";
const PORTABLE_BUNDLE_RECOMPUTE_FEATURE: &str = "chain_bundle_recomputation_portable_v2";
const MAX_BUNDLE_RECOMPUTE_BYTES: usize = 16 * 1024 * 1024;

/// A bundle whose dependencies are all independently replayable, with one
/// propagation-chain terminal recomputed from its retained compiler program.
#[derive(Clone, Debug)]
pub struct CompiledChainBundleRecomputation {
    bundle: CompositionBundle,
    terminal_node: String,
    chain: PropagationChainPlan,
    lowered: Arc<LoweredProgram>,
}

/// One dependency that passed the bundle's artifact consumer in this run.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayedBundleDependency {
    /// Node identity within the bundle.
    pub node_id: String,
    /// Recomputed artifact identity.
    pub artifact_identity: String,
    /// Bundle Merkle digest after its upstream dependencies were bound.
    pub chain_digest: String,
}

/// An actual compiler execution and all dependency replay outcomes, with no
/// claim that an external source was refitted or a new calibration was earned.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ChainBundleRecomputationReceipt {
    /// Recomputed bundle identity.
    pub bundle_identity: String,
    /// Terminal bundle node.
    pub terminal_node: String,
    /// Recomputed propagation-chain artifact identity.
    pub terminal_identity: String,
    /// Six-layer program that executed the terminal.
    pub lowered_program_identity: [u8; 32],
    /// Every embedded dependency's verified replay identity and Merkle digest.
    pub dependencies: Vec<ReplayedBundleDependency>,
}

/// Refusal of a bundle compiler, artifact consumer or live terminal execution.
#[derive(Debug, thiserror::Error)]
pub enum ChainBundleRecomputeError {
    /// Bundle structure or embedded artifact verification refused.
    #[error("bundle recomputation: {0}")]
    Bundle(#[from] BundleError),
    /// The terminal chain declaration or execution refused.
    #[error("bundle chain compiler: {0}")]
    Chain(#[from] ChainRefusal),
    /// A replay artifact could not be decoded or encoded.
    #[error("bundle recomputation artifact: {0}")]
    Io(#[from] IoError),
    /// A required replay or terminal binding is missing or changed.
    #[error("bundle recomputation: {0}")]
    Invalid(&'static str),
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ChainBundleRecomputationWire {
    version: u32,
    required_feature: String,
    receipt: ChainBundleRecomputationReceipt,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PortableChainBundleRecomputationWire {
    version: u32,
    required_feature: String,
    bundle_bytes: Vec<u8>,
    bundle_identity: String,
    terminal_node: String,
    receipt: ChainBundleRecomputationReceipt,
}

/// Lower one fully embedded bundle and its executable chain terminal into a
/// new six-layer program. The graph, each node identity and the terminal
/// declaration are bound before execution. Other embedded dependencies are
/// checked by their own independent consumers at execution.
///
/// # Errors
/// A missing or non-embedded terminal, or a physical operator refusal.
pub fn compile_chain_bundle_recomputation(
    bundle: CompositionBundle,
    terminal_node: &str,
    chain: PropagationChainPlan,
) -> Result<CompiledChainBundleRecomputation, ChainBundleRecomputeError> {
    let node = bundle.nodes().iter().find(|node| node.id == terminal_node)
        .ok_or(ChainBundleRecomputeError::Invalid("terminal node is absent from the bundle"))?;
    if node.kind != NodeKind::PropagationChain
        || !matches!(&node.source, NodeSource::Embedded { .. })
    {
        return Err(ChainBundleRecomputeError::Invalid(
            "the executable terminal must be an embedded propagation chain",
        ));
    }
    let bundle_digest = CanonicalWriter::digest_bytes(
        "antecedent.bundle_recomputation.bundle.v1", bundle.identity().as_bytes(),
    );
    let mut writer = CanonicalWriter::new("antecedent.bundle_recomputation.question.v1");
    writer.bytes(&bundle_digest).text(terminal_node).bytes(chain.identity().as_bytes());
    let query = writer.finish();
    let scientific = ScientificProblem {
        runtime_query: None, identity_scope: ProgramIdentityScope::Semantic,
        target: ProblemTarget::QueryFunctional {
            kind: FunctionalKind::PropagatedChain, outcome: None, query,
        },
        plan_id: Arc::from("chain_bundle_recomputation.v1"),
        data: DataClassification::Tabular, graph_class: "bundle_dependency_graph",
        query_variables: Arc::from(Vec::new()), source_digest: Some(bundle_digest),
        support_coordinate: None, supported: false, support_status: None,
    };
    let identification = IdentificationLayer::NonGraphical(Arc::new(
        NonGraphicalIdentification::SharedEvidenceContract { contract: query },
    ));
    let mut sources = chain.lowered().evidence().sources().to_vec();
    sources.push(EvidenceSourceRef {
        role: EvidenceRole::ChainSource, digest: Some(bundle_digest),
    });
    let evidence = EvidenceDecisionProgram {
        sources: Arc::from(sources),
        composition: Composition::Chained {
            stages: chain.plan().stages.len() + bundle.nodes().len(),
        },
        decision: chain.lowered().evidence().decision().cloned(), planning: None,
        chain: chain.lowered().propagation_chain().cloned(),
    };
    let mut operators = chain.lowered().physical().operators().to_vec();
    let quote = (bundle.nodes().len() as u64).saturating_mul(128);
    operators.push(PhysicalOperator::declare(PhysicalEffect::Read)
        .estimates(ByteEstimates::absent().with_output(quote))
        .build().map_err(|_| ChainBundleRecomputeError::Invalid(
            "the bundle replay physical operator is invalid",
        ))?);
    let physical = PhysicalLayer {
        operators: Arc::from(operators),
        design: chain.lowered().physical().design(),
    };
    let lowered = Arc::new(StandaloneCompositionLoweredParts {
        cell: ProgramCell::BundleRecomputation,
        scientific, identification,
        statistical: chain.lowered().statistical().clone(),
        evidence, physical, claim: *chain.lowered().claim(),
    }.finish());
    let chain = chain.within_lowered(Arc::clone(&lowered))?;
    Ok(CompiledChainBundleRecomputation {
        bundle, terminal_node: terminal_node.to_owned(), chain, lowered,
    })
}

impl CompiledChainBundleRecomputation {
    /// The bundle and live terminal's six-layer compiler program.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram { &self.lowered }

    /// Inspect the compiled bundle recomputation without executing it.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        PlanView::of_program(Arc::clone(&self.lowered), PlanStage::Compiled)
    }

    /// Replay every embedded dependency and execute the terminal program.
    /// A reference node or failed artifact prevents an execution receipt.
    ///
    /// # Errors
    /// A failed dependency, changed terminal law or compiler refusal.
    pub fn execute(&self, ctx: &ExecutionContext)
        -> Result<ChainBundleRecomputationReceipt, ChainBundleRecomputeError> {
        let bytes = export_bundle(&self.bundle, "chain-bundle-recomputation")?;
        let consumed = consume(&bytes, self.bundle.identity(), &SuppliedSources::default())?;
        if !consumed.all_replayed() {
            return Err(ChainBundleRecomputeError::Invalid(
                "every bundle dependency must pass an embedded artifact consumer",
            ));
        }
        let node = self.bundle.nodes().iter().find(|node| node.id == self.terminal_node)
            .ok_or(ChainBundleRecomputeError::Invalid("terminal node disappeared"))?;
        let NodeSource::Embedded { bytes } = &node.source else {
            return Err(ChainBundleRecomputeError::Invalid("terminal node is a reference"));
        };
        let recorded = PropagationChainRecordWire::decode(bytes)
            .map_err(|error| ChainBundleRecomputeError::Chain(error.to_refusal()))?;
        recorded.consume(Some(&node.identity), ctx)
            .map_err(|error| ChainBundleRecomputeError::Chain(error.to_refusal()))?;
        let live = self.chain.execute(ctx)?;
        if &recorded != live.record() || node.identity != live.identity() {
            return Err(ChainBundleRecomputeError::Invalid(
                "the embedded terminal differs from the compiler's live result",
            ));
        }
        let dependencies = consumed.nodes().iter().map(|node| ReplayedBundleDependency {
            node_id: node.id.clone(),
            artifact_identity: node.identity.clone(),
            chain_digest: node.chain_digest.clone(),
        }).collect();
        Ok(ChainBundleRecomputationReceipt {
            bundle_identity: consumed.identity().to_owned(),
            terminal_node: self.terminal_node.clone(),
            terminal_identity: live.identity().to_owned(),
            lowered_program_identity: *self.lowered.identity().as_bytes(),
            dependencies,
        })
    }

    /// Export a receipt after independently checking its live execution.
    ///
    /// # Errors
    /// A changed receipt or artifact encoding refusal.
    pub fn export(&self, receipt: &ChainBundleRecomputationReceipt, ctx: &ExecutionContext)
        -> Result<Vec<u8>, ChainBundleRecomputeError> {
        if self.execute(ctx)? != *receipt {
            return Err(ChainBundleRecomputeError::Invalid("bundle execution receipt differs"));
        }
        let bytes = antecedent_io::to_cbor(&ChainBundleRecomputationWire {
            version: 1, required_feature: BUNDLE_RECOMPUTE_FEATURE.into(),
            receipt: receipt.clone(),
        })?;
        if bytes.len() > MAX_BUNDLE_RECOMPUTE_BYTES {
            return Err(ChainBundleRecomputeError::Invalid("bundle recomputation receipt exceeds its byte bound"));
        }
        Ok(bytes)
    }

    /// Export the embedded graph and live receipt so a fresh process can
    /// reconstruct the terminal compiler without holding this plan.
    ///
    /// # Errors
    /// A changed receipt or a container over the portable byte limit.
    pub fn export_portable(
        &self, receipt: &ChainBundleRecomputationReceipt, ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, ChainBundleRecomputeError> {
        if self.execute(ctx)? != *receipt {
            return Err(ChainBundleRecomputeError::Invalid("bundle execution receipt differs"));
        }
        let bytes = antecedent_io::to_cbor(&PortableChainBundleRecomputationWire {
            version: 2,
            required_feature: PORTABLE_BUNDLE_RECOMPUTE_FEATURE.into(),
            bundle_bytes: export_bundle(&self.bundle, "portable-chain-bundle-recomputation")?,
            bundle_identity: self.bundle.identity().to_owned(),
            terminal_node: self.terminal_node.clone(),
            receipt: receipt.clone(),
        })?;
        if bytes.len() > MAX_BUNDLE_RECOMPUTE_BYTES {
            return Err(ChainBundleRecomputeError::Invalid(
                "portable bundle receipt exceeds its byte bound",
            ));
        }
        Ok(bytes)
    }

    /// Independently re-run every dependency consumer and the retained chain,
    /// then compare the complete receipt under a separately retained program id.
    ///
    /// # Errors
    /// A foreign feature, changed program or changed replay result.
    pub fn consume(
        &self, bytes: &[u8], expected_program: SemanticDigest,
        ctx: &ExecutionContext,
    ) -> Result<ChainBundleRecomputationReceipt, ChainBundleRecomputeError> {
        if bytes.len() > MAX_BUNDLE_RECOMPUTE_BYTES {
            return Err(ChainBundleRecomputeError::Invalid("bundle recomputation receipt exceeds its byte bound"));
        }
        let wire: ChainBundleRecomputationWire = antecedent_io::from_cbor(bytes)?;
        if wire.version != 1 || wire.required_feature != BUNDLE_RECOMPUTE_FEATURE
            || expected_program != self.lowered.identity()
        {
            return Err(ChainBundleRecomputeError::Invalid("bundle recomputation feature or program differs"));
        }
        let live = self.execute(ctx)?;
        if wire.receipt != live {
            return Err(ChainBundleRecomputeError::Invalid("bundle dependency or terminal replay differs"));
        }
        Ok(live)
    }
}

/// Rebuild a portable bundle and its chain declaration, compile the full
/// six-layer terminal program, replay every dependency and compare the live
/// receipt against independently retained bundle and program identities.
///
/// # Errors
/// A foreign feature, changed graph or program, failed embedded consumer or
/// changed live terminal value.
pub fn consume_portable_chain_bundle_recomputation(
    bytes: &[u8], expected_bundle: &str, expected_program: SemanticDigest,
    ctx: &ExecutionContext,
) -> Result<ChainBundleRecomputationReceipt, ChainBundleRecomputeError> {
    if bytes.len() > MAX_BUNDLE_RECOMPUTE_BYTES {
        return Err(ChainBundleRecomputeError::Invalid("portable bundle receipt exceeds its byte bound"));
    }
    let wire: PortableChainBundleRecomputationWire = antecedent_io::from_cbor(bytes)?;
    if wire.version != 2 || wire.required_feature != PORTABLE_BUNDLE_RECOMPUTE_FEATURE
        || wire.bundle_identity != expected_bundle
    {
        return Err(ChainBundleRecomputeError::Invalid("portable bundle feature or identity differs"));
    }
    let bundle = CompositionBundle::from_bytes(
        &wire.bundle_bytes, expected_bundle, &BundleLimits::default(),
    )?;
    let terminal = bundle.nodes().iter().find(|node| node.id == wire.terminal_node)
        .ok_or(ChainBundleRecomputeError::Invalid("portable terminal node is absent"))?;
    if terminal.kind != NodeKind::PropagationChain {
        return Err(ChainBundleRecomputeError::Invalid("portable terminal is not a propagation chain"));
    }
    let NodeSource::Embedded { bytes: terminal_bytes } = &terminal.source else {
        return Err(ChainBundleRecomputeError::Invalid("portable terminal is not embedded"));
    };
    let recorded = PropagationChainRecordWire::decode(terminal_bytes)
        .map_err(|error| ChainBundleRecomputeError::Chain(error.to_refusal()))?;
    let chain = compile_propagation_chain(recorded, ctx)?;
    let plan = compile_chain_bundle_recomputation(bundle, &wire.terminal_node, chain)?;
    if plan.lowered().identity() != expected_program {
        return Err(ChainBundleRecomputeError::Invalid("portable bundle compiler program differs"));
    }
    let live = plan.execute(ctx)?;
    if wire.receipt != live {
        return Err(ChainBundleRecomputeError::Invalid("portable bundle live receipt differs"));
    }
    Ok(live)
}

const BRIDGE_BUNDLE_FEATURE: &str = "bridge_bundle_recomputation_portable_v1";

/// Fully embedded bundle with a live predictive scientific bridge terminal.
/// The source table, checked bridge contract and seed are bound before the
/// bundle's dependency consumers and terminal mapping run.
#[derive(Clone, Debug)]
pub struct CompiledBridgeBundleRecomputation {
    bundle: CompositionBundle,
    terminal_node: String,
    bridge: PreparedScientificBridgeDistribution,
    source: DistributionArtifact,
    seed: u64,
    lowered: Arc<LoweredProgram>,
}

/// Live mapped bridge identity and every independently replayed dependency.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BridgeBundleRecomputationReceipt {
    /// Independently consumed bundle graph identity.
    pub bundle_identity: String,
    /// Scientific bridge terminal node.
    pub terminal_node: String,
    /// Freshly derived bridge artifact identity.
    pub terminal_identity: String,
    /// Identity of the complete outer compiler program.
    pub lowered_program_identity: [u8; 32],
    /// Every dependency verified by its own embedded artifact consumer.
    pub dependencies: Vec<ReplayedBundleDependency>,
}

/// Refusal of a bridge bundle compiler, dependency or terminal replay.
#[derive(Debug, thiserror::Error)]
pub enum BridgeBundleRecomputeError {
    /// Bundle structure or dependency consumer refused.
    #[error("bridge bundle recomputation: {0}")]
    Bundle(#[from] BundleError),
    /// Bridge contract, derivation or artifact consumer refused.
    #[error("bridge bundle recomputation bridge: {0}")]
    Bridge(#[from] ScientificBridgeRefusal),
    /// Source artifact encoding or portable receipt codec refused.
    #[error("bridge bundle recomputation artifact: {0}")]
    Io(#[from] IoError),
    /// Embedded node, source or fresh result differs.
    #[error("bridge bundle recomputation: {0}")]
    Invalid(&'static str),
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PortableBridgeBundleRecomputationWire {
    version: u32,
    required_feature: String,
    bundle_bytes: Vec<u8>,
    bundle_identity: String,
    terminal_node: String,
    receipt: BridgeBundleRecomputationReceipt,
}

/// Lower a fully embedded bundle and its predictive scientific bridge terminal
/// together. The declared contract, source law, seed, graph and terminal node
/// all enter a new six-layer identity; no mapping is run at compilation.
///
/// # Errors
/// Missing or non-embedded terminal, unsupported predictive carrier, bridge
/// contract, source artifact or physical lowering refusal.
pub fn compile_bridge_bundle_recomputation(
    bundle: CompositionBundle, terminal_node: &str,
    contract: BridgeContract, source: DistributionArtifact, seed: u64,
) -> Result<CompiledBridgeBundleRecomputation, BridgeBundleRecomputeError> {
    let node = bundle.nodes().iter().find(|node| node.id == terminal_node)
        .ok_or(BridgeBundleRecomputeError::Invalid("bridge terminal node is absent"))?;
    if node.kind != NodeKind::ScientificBridge
        || !matches!(&node.source, NodeSource::Embedded { .. })
    {
        return Err(BridgeBundleRecomputeError::Invalid(
            "bridge terminal must be an embedded scientific bridge",
        ));
    }
    if contract.source_value != ValueKind::PredictiveLaw
        || contract.target_value != ValueKind::PredictiveLaw
        || !matches!(source.semantic(),
            DistributionMeaningWire::PosteriorPredictive
                | DistributionMeaningWire::InterventionalPredictive
                | DistributionMeaningWire::EmpiricalOutcome)
        || !source.quantities().contains(&ScientificQuantityWire::from(&contract.source.quantity))
        || source.n_draws() < 2
    {
        return Err(BridgeBundleRecomputeError::Invalid(
            "bridge terminal needs a checked predictive source coordinate and aligned draws",
        ));
    }
    let bridge = PreparedScientificBridgeDistribution::prepare(contract)?;
    let source_bytes = source.to_bytes("bridge-bundle-source")?;
    let source_digest = CanonicalWriter::digest_bytes(
        "antecedent.bridge_bundle_recomputation.source.v1", &source_bytes,
    );
    let bundle_digest = CanonicalWriter::digest_bytes(
        "antecedent.bridge_bundle_recomputation.bundle.v1", bundle.identity().as_bytes(),
    );
    let mut writer = CanonicalWriter::new("antecedent.bridge_bundle_recomputation.question.v1");
    writer.bytes(&bundle_digest).text(terminal_node).bytes(&source_digest)
        .bytes(bridge.contract_identity().as_bytes()).word(seed);
    let query = writer.finish();
    let design = DesignShape::quoted(source.n_draws() as u64, 1, true);
    let scientific = ScientificProblem {
        runtime_query: None, identity_scope: ProgramIdentityScope::Semantic,
        target: ProblemTarget::QueryFunctional {
            kind: FunctionalKind::PropagatedChain, outcome: None, query,
        },
        plan_id: Arc::from("bridge_bundle_recomputation.v1"),
        data: DataClassification::Tabular, graph_class: "bundle_bridge_dependency_graph",
        query_variables: Arc::from(Vec::new()), source_digest: Some(source_digest),
        support_coordinate: None, supported: false, support_status: None,
    };
    let identification = IdentificationLayer::NonGraphical(Arc::new(
        NonGraphicalIdentification::SharedEvidenceContract { contract: query },
    ));
    let statistical = StatisticalProgram {
        identifier: None, estimator: None,
        realization: Realization::DeclaredPredictive {
            evaluator: Arc::from("scientific_bridge.bundle_terminal"),
        },
        models: Arc::from([]), folds: None, resampling: None,
        posterior: None, design, operators: Arc::from([]),
    };
    let evidence = EvidenceDecisionProgram {
        sources: Arc::from([
            EvidenceSourceRef { role: EvidenceRole::ChainSource, digest: Some(source_digest) },
            EvidenceSourceRef { role: EvidenceRole::TransportSource, digest: Some(bundle_digest) },
        ]),
        composition: Composition::Chained { stages: bundle.nodes().len() + 1 },
        decision: None, planning: None, chain: None,
    };
    let physical = PhysicalLayer {
        operators: Arc::from([
            PhysicalOperator::declare(PhysicalEffect::Read)
                .estimates(ByteEstimates::absent().with_output(source_bytes.len() as u64))
                .build().map_err(|_| BridgeBundleRecomputeError::Invalid(
                    "bridge bundle source read cannot be lowered",
                ))?,
            PhysicalOperator::declare(PhysicalEffect::Transport)
                .estimates(ByteEstimates::absent().with_output(
                    (source.n_draws() as u64).saturating_mul(8),
                )).build().map_err(|_| BridgeBundleRecomputeError::Invalid(
                    "bridge bundle transport cannot be lowered",
                ))?,
        ]), design,
    };
    let lowered = Arc::new(StandaloneCompositionLoweredParts {
        cell: ProgramCell::BridgeBundleRecomputation,
        scientific, identification, statistical, evidence, physical,
        claim: ClaimDemand { uncertainty: UncertaintyReading::Predictive, kind: None },
    }.finish());
    Ok(CompiledBridgeBundleRecomputation {
        bundle, terminal_node: terminal_node.to_owned(), bridge, source, seed, lowered,
    })
}

impl CompiledBridgeBundleRecomputation {
    /// Complete bridge-terminal bundle compiler program.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram { &self.lowered }

    /// Inspect the retained bundle and terminal schedule without mapping.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        PlanView::of_program(Arc::clone(&self.lowered), PlanStage::Compiled)
    }

    /// Consume every embedded dependency and derive the scientific bridge
    /// from the compiler-bound source, contract and seed.
    ///
    /// # Errors
    /// A reference, failed artifact or changed live bridge refuses a receipt.
    pub fn execute(&self, ctx: &ExecutionContext)
        -> Result<BridgeBundleRecomputationReceipt, BridgeBundleRecomputeError> {
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        if scoped.cancellation.is_cancelled() {
            return Err(BridgeBundleRecomputeError::Invalid("bridge bundle execution was cancelled"));
        }
        let bytes = export_bundle(&self.bundle, "bridge-bundle-recomputation")?;
        let consumed = consume(&bytes, self.bundle.identity(), &SuppliedSources::default())?;
        if !consumed.all_replayed() {
            return Err(BridgeBundleRecomputeError::Invalid(
                "every bridge bundle dependency must pass its artifact consumer",
            ));
        }
        let node = self.bundle.nodes().iter().find(|node| node.id == self.terminal_node)
            .ok_or(BridgeBundleRecomputeError::Invalid("bridge terminal disappeared"))?;
        let NodeSource::Embedded { bytes: terminal_bytes } = &node.source else {
            return Err(BridgeBundleRecomputeError::Invalid("bridge terminal became a reference"));
        };
        let stored = consume_scientific_bridge_distribution_artifact(
            terminal_bytes, Some(&node.identity),
        )?;
        let live = self.bridge.derive(&self.source, None, self.seed)?;
        let exported = live.export("bridge-bundle-live")?;
        if exported.identity != node.identity || stored.identity != exported.identity
            || stored.bridged.claim().claim.claim_id != live.claim().claim.claim_id
        {
            return Err(BridgeBundleRecomputeError::Invalid(
                "embedded bridge differs from the live compiler terminal",
            ));
        }
        let dependencies = consumed.nodes().iter().map(|node| ReplayedBundleDependency {
            node_id: node.id.clone(), artifact_identity: node.identity.clone(),
            chain_digest: node.chain_digest.clone(),
        }).collect();
        Ok(BridgeBundleRecomputationReceipt {
            bundle_identity: consumed.identity().to_owned(),
            terminal_node: self.terminal_node.clone(),
            terminal_identity: exported.identity,
            lowered_program_identity: *self.lowered.identity().as_bytes(),
            dependencies,
        })
    }

    /// Export an independently reconstructible graph and terminal receipt.
    ///
    /// # Errors
    /// Changed live result or oversized portable artifact.
    pub fn export_portable(
        &self, receipt: &BridgeBundleRecomputationReceipt, ctx: &ExecutionContext,
    ) -> Result<Vec<u8>, BridgeBundleRecomputeError> {
        if self.execute(ctx)? != *receipt {
            return Err(BridgeBundleRecomputeError::Invalid("bridge bundle receipt differs"));
        }
        let bytes = antecedent_io::to_cbor(&PortableBridgeBundleRecomputationWire {
            version: 1, required_feature: BRIDGE_BUNDLE_FEATURE.into(),
            bundle_bytes: export_bundle(&self.bundle, "portable-bridge-bundle-recomputation")?,
            bundle_identity: self.bundle.identity().to_owned(),
            terminal_node: self.terminal_node.clone(), receipt: receipt.clone(),
        })?;
        if bytes.len() > MAX_BUNDLE_RECOMPUTE_BYTES {
            return Err(BridgeBundleRecomputeError::Invalid("portable bridge bundle exceeds its byte bound"));
        }
        Ok(bytes)
    }
}

/// Reconstruct the terminal bridge declaration from its independently
/// consumed artifact, recompile the outer program and compare its live receipt
/// under independently retained graph and program identities.
///
/// # Errors
/// Foreign feature, changed graph, mapping, source, seed or dependency.
pub fn consume_portable_bridge_bundle_recomputation(
    bytes: &[u8], expected_bundle: &str, expected_program: SemanticDigest,
    ctx: &ExecutionContext,
) -> Result<BridgeBundleRecomputationReceipt, BridgeBundleRecomputeError> {
    if bytes.len() > MAX_BUNDLE_RECOMPUTE_BYTES {
        return Err(BridgeBundleRecomputeError::Invalid("portable bridge bundle exceeds its byte bound"));
    }
    let wire: PortableBridgeBundleRecomputationWire = antecedent_io::from_cbor(bytes)?;
    if wire.version != 1 || wire.required_feature != BRIDGE_BUNDLE_FEATURE
        || wire.bundle_identity != expected_bundle
    {
        return Err(BridgeBundleRecomputeError::Invalid("portable bridge bundle feature or identity differs"));
    }
    let bundle = CompositionBundle::from_bytes(
        &wire.bundle_bytes, expected_bundle, &BundleLimits::default(),
    )?;
    let terminal = bundle.nodes().iter().find(|node| node.id == wire.terminal_node)
        .ok_or(BridgeBundleRecomputeError::Invalid("portable bridge terminal is absent"))?;
    if terminal.kind != NodeKind::ScientificBridge {
        return Err(BridgeBundleRecomputeError::Invalid("portable terminal is not a scientific bridge"));
    }
    let NodeSource::Embedded { bytes: terminal_bytes } = &terminal.source else {
        return Err(BridgeBundleRecomputeError::Invalid("portable bridge terminal is not embedded"));
    };
    let stored = consume_scientific_bridge_distribution_artifact(
        terminal_bytes, Some(&terminal.identity),
    )?;
    let plan = compile_bridge_bundle_recomputation(
        bundle, &wire.terminal_node, stored.bridged.claim().contract.clone(),
        stored.bridged.source().clone(), stored.bridged.claim().stream,
    )?;
    if plan.lowered().identity() != expected_program {
        return Err(BridgeBundleRecomputeError::Invalid("portable bridge bundle compiler differs"));
    }
    let live = plan.execute(ctx)?;
    if live != wire.receipt {
        return Err(BridgeBundleRecomputeError::Invalid("portable bridge bundle live receipt differs"));
    }
    Ok(live)
}

// Version 1 used the caller's seed. Version 2 binds the producer's replay
// stream in the node while retaining the caller's cancellation/admission pool.
fn fit_node_version_supported(version: u32, feature: &str, seed: Option<u64>,
    legacy: &str, current: &str) -> bool {
    (version == 1 && feature == legacy && seed.is_none())
        || (version == 2 && feature == current && seed.is_some())
}

fn fit_node_context(ctx: &ExecutionContext, seed: Option<u64>) -> ExecutionContext {
    let mut scoped = ctx.clone();
    if let Some(seed) = seed { scoped.rng = antecedent_core::RngFactory::from_seed(seed); }
    scoped
}

fn fit_node_failure(ctx: &ExecutionContext, error: &(dyn std::error::Error + 'static)) -> NodeFailure {
    let mut stage = if ctx.cancellation.is_cancelled() {
        BundleStage::ExecutionCancelled
    } else { BundleStage::SwappedEvidence };
    let mut cause = Some(error);
    while let Some(current) = cause {
        if let Some(io) = current.downcast_ref::<IoError>() {
            stage = match io {
                IoError::UnsupportedVersion { .. } | IoError::UnsupportedFormat { .. } => BundleStage::IncompatibleVersion,
                IoError::TooLarge => BundleStage::Oversized,
                IoError::StreamCancelled => BundleStage::ExecutionCancelled,
                IoError::Admission(antecedent_core::execution::AdmissionError::Cancelled) => BundleStage::ExecutionCancelled,
                IoError::Admission(_) | IoError::StreamBudgetRefused { .. } => BundleStage::ResourceRefused,
                _ => stage,
            };
        }
        if let Some(admission) = current.downcast_ref::<antecedent_core::execution::AdmissionError>() {
            stage = if matches!(admission, antecedent_core::execution::AdmissionError::Cancelled) {
                BundleStage::ExecutionCancelled
            } else { BundleStage::ResourceRefused };
        }
        if matches!(current.downcast_ref::<antecedent_model::nonlinear_equilibrium::EquilibriumError>(),
            Some(antecedent_model::nonlinear_equilibrium::EquilibriumError::Resource(_)))
            || matches!(current.downcast_ref::<crate::error::CausalError>(), Some(crate::error::CausalError::Resource { .. }))
        {
            if stage != BundleStage::ExecutionCancelled { stage = BundleStage::ResourceRefused; }
        }
        cause = current.source();
    }
    NodeFailure { stage, reason: error.to_string() }
}

const CYCLIC_POSTERIOR_FIT_FEATURE: &str = "cyclic_posterior_fit_bundle_node_v1";
const CYCLIC_POSTERIOR_FIT_FEATURE_V2: &str = "cyclic_posterior_fit_bundle_node_v2";
const CYCLIC_FIT_CHAIN_FACT: &str = "cyclic_posterior_fit_chain_identity";

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CyclicPosteriorFitWire {
    version: u32,
    required_feature: String,
    producer_program: [u8; 32],
    artifact: Vec<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    replay_seed: Option<u64>,
}

struct CyclicPosteriorFitVerifier { ctx: ExecutionContext }

impl NodeVerifier for CyclicPosteriorFitVerifier {
    fn kind(&self) -> NodeKind { NodeKind::CyclicPosteriorFit }

    fn verify(&self, _node_id: &str, bytes: &[u8], _upstream: &[UpstreamNode<'_>])
        -> Result<VerifiedNode, NodeFailure> {
        let fail = |reason: String| NodeFailure { stage: BundleStage::SwappedEvidence, reason };
        if bytes.len() > MAX_BUNDLE_RECOMPUTE_BYTES {
            return Err(NodeFailure { stage: BundleStage::Oversized, reason: "cyclic posterior fit node exceeds its bound".into() });
        }
        let wire: CyclicPosteriorFitWire = antecedent_io::from_cbor(bytes)
            .map_err(|error| fail(error.to_string()))?;
        if !fit_node_version_supported(wire.version, &wire.required_feature, wire.replay_seed, CYCLIC_POSTERIOR_FIT_FEATURE, CYCLIC_POSTERIOR_FIT_FEATURE_V2) {
            return Err(NodeFailure { stage: BundleStage::IncompatibleVersion, reason: "cyclic posterior fit feature differs".into() });
        }
        let node_ctx = fit_node_context(&self.ctx, wire.replay_seed);
        let producer = super::cyclic_posterior_action_decision::consume_cyclic_posterior_action_decision(
            &wire.artifact, SemanticDigest::from_bytes(wire.producer_program), &node_ctx,
        ).map_err(|error| fit_node_failure(&node_ctx, &error))?;
        let mut facts = std::collections::BTreeMap::new();
        facts.insert(CYCLIC_FIT_CHAIN_FACT.into(), producer.decision.chain.identity().into());
        Ok(VerifiedNode {
            identity: CanonicalWriter::digest_bytes_hex("antecedent.cyclic_posterior_fit_node.v1", bytes),
            facts, values: Vec::new(),
        })
    }
}

/// A bundle consumer able to refit native cyclic posterior nodes with `ctx`.
/// Other embedded artifacts retain their usual independent verifiers.
#[must_use]
pub fn cyclic_posterior_bundle_consumer(ctx: &ExecutionContext) -> BundleConsumer {
    native_fit_bundle_consumer(ctx)
}

/// Consumer with the current native fitted-node verifiers. Each verifier
/// recompiles from the embedded portable evidence under the supplied context.
#[must_use]
pub fn native_fit_bundle_consumer(ctx: &ExecutionContext) -> BundleConsumer {
    standard_consumer()
        .register(Box::new(CyclicPosteriorFitVerifier { ctx: ctx.clone() }))
        .register(Box::new(TrialIpwFitVerifier { ctx: ctx.clone() }))
        .register(Box::new(GraphImputationTransportFitVerifier { ctx: ctx.clone() }))
        .register(Box::new(JointTransportEffectFitVerifier { ctx: ctx.clone() }))
}

/// A bundle terminal backed by a cyclic posterior fit. The producer artifact
/// retains the raw training evidence and is recompiled on every execution.
#[derive(Clone, Debug)]
pub struct CompiledCyclicPosteriorBundleRecomputation {
    chain: CompiledChainBundleRecomputation,
    producer_artifact: Vec<u8>,
    producer_program: SemanticDigest,
    lowered: Arc<LoweredProgram>,
}

/// Receipt for a refitted cyclic posterior and its independently consumed bundle.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CyclicPosteriorBundleReceipt {
    /// The independently replayed terminal and graph.
    pub bundle: ChainBundleRecomputationReceipt,
    /// Identity of the fitted producer's compiler program.
    pub producer_program_identity: [u8; 32],
    /// Identity of the complete bundle and producer compiler program.
    pub lowered_program_identity: [u8; 32],
}

/// Refusal of the producer fit or bundle recomputation.
#[derive(Debug, thiserror::Error)]
pub enum CyclicPosteriorBundleError {
    /// A bundle, chain or codec operation refused.
    #[error("cyclic posterior bundle: {0}")]
    Bundle(#[from] ChainBundleRecomputeError),
    /// The raw-evidence posterior producer refused reexecution.
    #[error("cyclic posterior producer: {0}")]
    Producer(#[from] super::cyclic_posterior_action_decision::CyclicPosteriorDecisionError),
    /// The portable artifact or its bindings differed.
    #[error("cyclic posterior bundle: {0}")]
    Invalid(&'static str),
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PortableCyclicPosteriorBundleWire {
    version: u32,
    required_feature: String,
    bundle_bytes: Vec<u8>,
    bundle_identity: String,
    terminal_node: String,
    producer_artifact: Vec<u8>,
    producer_program: [u8; 32],
    receipt: CyclicPosteriorBundleReceipt,
}

const CYCLIC_POSTERIOR_BUNDLE_FEATURE: &str = "cyclic_posterior_bundle_recomputation_v1";

/// Compile the bundle and native fitted producer into one six-layer program.
/// The embedded chain must be the chain generated by the producer; this is
/// checked by refitting from the producer's portable raw evidence on execution.
/// Compilation itself does not fit the model.
///
/// # Errors
/// A missing or malformed chain, an oversized producer, or a lowering refusal.
pub fn compile_cyclic_posterior_bundle_recomputation(
    bundle: CompositionBundle, terminal_node: &str, producer_artifact: Vec<u8>,
    producer_program: SemanticDigest, ctx: &ExecutionContext,
) -> Result<CompiledCyclicPosteriorBundleRecomputation, CyclicPosteriorBundleError> {
    if producer_artifact.len() > MAX_BUNDLE_RECOMPUTE_BYTES {
        return Err(CyclicPosteriorBundleError::Invalid("producer artifact exceeds its bound"));
    }
    let node = bundle.nodes().iter().find(|node| node.id == terminal_node)
        .ok_or(CyclicPosteriorBundleError::Invalid("terminal node is absent"))?;
    if node.kind != NodeKind::PropagationChain {
        return Err(CyclicPosteriorBundleError::Invalid("terminal is not a propagation chain"));
    }
    let NodeSource::Embedded { bytes } = &node.source else {
        return Err(CyclicPosteriorBundleError::Invalid("terminal is not embedded"));
    };
    let recorded = PropagationChainRecordWire::decode(bytes)
        .map_err(|error| ChainBundleRecomputeError::Chain(error.to_refusal()))?;
    let chain = compile_propagation_chain(recorded, ctx)
        .map_err(ChainBundleRecomputeError::Chain)?;
    let chain = compile_chain_bundle_recomputation(bundle, terminal_node, chain)?;
    let artifact_digest = CanonicalWriter::digest_bytes(
        "antecedent.cyclic_posterior_bundle.producer.v1", &producer_artifact,
    );
    let mut writer = CanonicalWriter::new("antecedent.cyclic_posterior_bundle.question.v1");
    writer.bytes(chain.lowered().identity().as_bytes())
        .bytes(producer_program.as_bytes()).bytes(&artifact_digest);
    let query = writer.finish();
    let mut scientific = chain.lowered().scientific().clone();
    scientific.target = ProblemTarget::QueryFunctional {
        kind: FunctionalKind::PropagatedChain, outcome: None, query,
    };
    scientific.plan_id = Arc::from("cyclic_posterior_bundle_recomputation.v1");
    scientific.source_digest = Some(artifact_digest);
    let identification = IdentificationLayer::NonGraphical(Arc::new(
        NonGraphicalIdentification::SharedEvidenceContract { contract: query },
    ));
    let mut evidence = chain.lowered().evidence().clone();
    let mut sources = evidence.sources.to_vec();
    sources.push(EvidenceSourceRef { role: EvidenceRole::Observations, digest: Some(artifact_digest) });
    evidence.sources = Arc::from(sources);
    let mut operators = Vec::new();
    for effect in [PhysicalEffect::Read, PhysicalEffect::Fit, PhysicalEffect::Intervene] {
        operators.push(PhysicalOperator::declare(effect).build().map_err(|_| {
            CyclicPosteriorBundleError::Invalid("producer physical operator cannot be lowered")
        })?);
    }
    operators.extend_from_slice(chain.lowered().physical().operators());
    let physical = PhysicalLayer {
        operators: Arc::from(operators), design: chain.lowered().physical().design(),
    };
    let lowered = Arc::new(StandaloneCompositionLoweredParts {
        cell: ProgramCell::CyclicPosteriorBundleRecomputation, scientific, identification,
        statistical: chain.lowered().statistical().clone(), evidence, physical,
        claim: *chain.lowered().claim(),
    }.finish());
    Ok(CompiledCyclicPosteriorBundleRecomputation {
        chain, producer_artifact, producer_program, lowered,
    })
}

impl CompiledCyclicPosteriorBundleRecomputation {
    /// Inspect the full six-layer program before fitting.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram { &self.lowered }

    /// Inspect the compiled route without fitting or executing the bundle.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        PlanView::of_program(Arc::clone(&self.lowered), PlanStage::Compiled)
    }

    /// Refit the producer, then independently consume and execute the bundle.
    ///
    /// # Errors
    /// Changed raw evidence, posterior, generated chain or bundle dependency.
    pub fn execute(&self, ctx: &ExecutionContext)
        -> Result<CyclicPosteriorBundleReceipt, CyclicPosteriorBundleError> {
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        let producer = super::cyclic_posterior_action_decision::consume_cyclic_posterior_action_decision(
            &self.producer_artifact, self.producer_program, &scoped,
        )?;
        let bundle = self.chain.execute(&scoped)?;
        if producer.decision.chain.identity() != bundle.terminal_identity {
            return Err(CyclicPosteriorBundleError::Invalid(
                "refitted producer chain differs from the embedded terminal",
            ));
        }
        Ok(CyclicPosteriorBundleReceipt {
            bundle, producer_program_identity: *self.producer_program.as_bytes(),
            lowered_program_identity: *self.lowered.identity().as_bytes(),
        })
    }

    /// Export the graph, raw-evidence producer artifact and verified receipt.
    ///
    /// # Errors
    /// A changed execution or oversized artifact.
    pub fn export_portable(&self, receipt: &CyclicPosteriorBundleReceipt, ctx: &ExecutionContext)
        -> Result<Vec<u8>, CyclicPosteriorBundleError> {
        if self.execute(ctx)? != *receipt {
            return Err(CyclicPosteriorBundleError::Invalid("bundle receipt differs"));
        }
        let bytes = antecedent_io::to_cbor(&PortableCyclicPosteriorBundleWire {
            version: 1, required_feature: CYCLIC_POSTERIOR_BUNDLE_FEATURE.into(),
            bundle_bytes: export_bundle(&self.chain.bundle, "cyclic-posterior-bundle")
                .map_err(ChainBundleRecomputeError::Bundle)?,
            bundle_identity: self.chain.bundle.identity().to_owned(),
            terminal_node: self.chain.terminal_node.clone(),
            producer_artifact: self.producer_artifact.clone(),
            producer_program: *self.producer_program.as_bytes(), receipt: receipt.clone(),
        }).map_err(ChainBundleRecomputeError::Io)?;
        if bytes.len() > MAX_BUNDLE_RECOMPUTE_BYTES {
            return Err(CyclicPosteriorBundleError::Invalid("portable bundle exceeds its bound"));
        }
        Ok(bytes)
    }
}

/// Independently reconstruct, recompile and execute a portable cyclic bundle.
/// Both identities must be retained independently of the artifact.
///
/// # Errors
/// Changed graph, compiler program, producer evidence or fitted terminal.
pub fn consume_portable_cyclic_posterior_bundle_recomputation(
    bytes: &[u8], expected_bundle: &str, expected_program: SemanticDigest,
    ctx: &ExecutionContext,
) -> Result<CyclicPosteriorBundleReceipt, CyclicPosteriorBundleError> {
    if bytes.len() > MAX_BUNDLE_RECOMPUTE_BYTES {
        return Err(CyclicPosteriorBundleError::Invalid("portable bundle exceeds its bound"));
    }
    let wire: PortableCyclicPosteriorBundleWire = antecedent_io::from_cbor(bytes)
        .map_err(ChainBundleRecomputeError::Io)?;
    if wire.version != 1 || wire.required_feature != CYCLIC_POSTERIOR_BUNDLE_FEATURE
        || wire.bundle_identity != expected_bundle
    {
        return Err(CyclicPosteriorBundleError::Invalid("portable feature or bundle identity differs"));
    }
    let bundle = CompositionBundle::from_bytes(&wire.bundle_bytes, expected_bundle, &BundleLimits::default())
        .map_err(ChainBundleRecomputeError::Bundle)?;
    let plan = compile_cyclic_posterior_bundle_recomputation(
        bundle, &wire.terminal_node, wire.producer_artifact,
        SemanticDigest::from_bytes(wire.producer_program), ctx,
    )?;
    if plan.lowered().identity() != expected_program {
        return Err(CyclicPosteriorBundleError::Invalid("portable compiler program differs"));
    }
    let live = plan.execute(ctx)?;
    if live != wire.receipt {
        return Err(CyclicPosteriorBundleError::Invalid("portable receipt differs"));
    }
    Ok(live)
}

/// A compiler-owned chain terminal whose cyclic fitted source is an embedded
/// bundle dependency, independently refitted by the bundle consumer.
#[derive(Clone, Debug)]
pub struct CompiledCyclicPosteriorNodeBundle {
    inner: CompiledCyclicPosteriorBundleRecomputation,
    producer_node: String,
    lowered: Arc<LoweredProgram>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PortableCyclicPosteriorNodeBundleWire {
    version: u32,
    required_feature: String,
    bundle_bytes: Vec<u8>,
    bundle_identity: String,
    producer_node: String,
    terminal_node: String,
    receipt: CyclicPosteriorBundleReceipt,
}

const CYCLIC_NODE_BUNDLE_FEATURE: &str = "cyclic_posterior_node_bundle_recomputation_v1";

/// Compile a live chain terminal with its embedded native fit dependency.
/// The graph must contain a direct fit-to-terminal edge. No fitting occurs
/// during compilation; the node verifier performs it at execution.
///
/// # Errors
/// Missing nodes or edge, malformed fit declaration or chain compiler refusal.
pub fn compile_cyclic_posterior_node_bundle_recomputation(
    bundle: CompositionBundle, producer_node: &str, terminal_node: &str,
    ctx: &ExecutionContext,
) -> Result<CompiledCyclicPosteriorNodeBundle, CyclicPosteriorBundleError> {
    if !bundle.edges().iter().any(|edge| edge.from == producer_node && edge.to == terminal_node) {
        return Err(CyclicPosteriorBundleError::Invalid("fit must be a direct terminal dependency"));
    }
    let node = bundle.nodes().iter().find(|node| node.id == producer_node)
        .ok_or(CyclicPosteriorBundleError::Invalid("fit node is absent"))?;
    if node.kind != NodeKind::CyclicPosteriorFit {
        return Err(CyclicPosteriorBundleError::Invalid("producer node is not a cyclic posterior fit"));
    }
    let NodeSource::Embedded { bytes } = &node.source else {
        return Err(CyclicPosteriorBundleError::Invalid("fit node is not embedded"));
    };
    let wire: CyclicPosteriorFitWire = antecedent_io::from_cbor(bytes)
        .map_err(ChainBundleRecomputeError::Io)?;
    if !fit_node_version_supported(wire.version, &wire.required_feature, wire.replay_seed, CYCLIC_POSTERIOR_FIT_FEATURE, CYCLIC_POSTERIOR_FIT_FEATURE_V2) {
        return Err(CyclicPosteriorBundleError::Invalid("fit node feature differs"));
    }
    let inner = compile_cyclic_posterior_bundle_recomputation(
        bundle, terminal_node, wire.artifact, SemanticDigest::from_bytes(wire.producer_program), ctx,
    )?;
    let mut writer = CanonicalWriter::new("antecedent.cyclic_posterior_node_bundle.question.v1");
    writer.bytes(inner.lowered.identity().as_bytes()).text(producer_node).text(terminal_node);
    let query = writer.finish();
    let mut scientific = inner.lowered.scientific().clone();
    scientific.target = ProblemTarget::QueryFunctional {
        kind: FunctionalKind::PropagatedChain, outcome: None, query,
    };
    scientific.plan_id = Arc::from("cyclic_posterior_node_bundle_recomputation.v1");
    let identification = IdentificationLayer::NonGraphical(Arc::new(
        NonGraphicalIdentification::SharedEvidenceContract { contract: query },
    ));
    let mut statistical = inner.lowered.statistical().clone();
    statistical.realization = Realization::NestedFit {
        producer_program: inner.producer_program, method: Arc::from("cyclic_posterior_action_decision_v1"),
        law: NestedFitLaw::Predictive,
    };
    let lowered = Arc::new(StandaloneCompositionLoweredParts {
        cell: ProgramCell::CyclicPosteriorNodeBundleRecomputation, scientific, identification,
        statistical, evidence: inner.lowered.evidence().clone(),
        physical: inner.lowered.physical().clone(), claim: *inner.lowered.claim(),
    }.finish());
    Ok(CompiledCyclicPosteriorNodeBundle { inner, producer_node: producer_node.into(), lowered })
}

impl CompiledCyclicPosteriorNodeBundle {
    /// Complete compiler program, binding the graph, fit and terminal.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram { &self.lowered }

    /// Refit the embedded source through the bundle consumer and execute the
    /// terminal chain, requiring the source's fresh chain identity to match.
    ///
    /// # Errors
    /// A failed embedded verifier, changed fitted source or terminal.
    pub fn execute(&self, ctx: &ExecutionContext)
        -> Result<CyclicPosteriorBundleReceipt, CyclicPosteriorBundleError> {
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        let bundle = &self.inner.chain.bundle;
        let bytes = export_bundle(bundle, "cyclic-posterior-node-bundle")
            .map_err(ChainBundleRecomputeError::Bundle)?;
        let consumed = cyclic_posterior_bundle_consumer(&scoped).consume(
            &bytes, &BundleLimits::default(), bundle.identity(), &SuppliedSources::default(),
        ).map_err(ChainBundleRecomputeError::Bundle)?;
        if !consumed.all_replayed() {
            return Err(CyclicPosteriorBundleError::Invalid("bundle dependency did not replay"));
        }
        let fit = consumed.node(&self.producer_node)
            .ok_or(CyclicPosteriorBundleError::Invalid("consumed fit is absent"))?;
        let terminal = bundle.nodes().iter().find(|node| node.id == self.inner.chain.terminal_node)
            .ok_or(CyclicPosteriorBundleError::Invalid("terminal disappeared"))?;
        let NodeSource::Embedded { bytes: terminal_bytes } = &terminal.source else {
            return Err(CyclicPosteriorBundleError::Invalid("terminal became a reference"));
        };
        let recorded = PropagationChainRecordWire::decode(terminal_bytes)
            .map_err(|error| ChainBundleRecomputeError::Chain(error.to_refusal()))?;
        recorded.consume(Some(&terminal.identity), &scoped)
            .map_err(|error| ChainBundleRecomputeError::Chain(error.to_refusal()))?;
        let live = self.inner.chain.chain.execute(&scoped).map_err(ChainBundleRecomputeError::Chain)?;
        if &recorded != live.record() || terminal.identity != live.identity()
            || fit.facts.get(CYCLIC_FIT_CHAIN_FACT).map(String::as_str) != Some(live.identity())
        {
            return Err(CyclicPosteriorBundleError::Invalid("refitted source and live terminal differ"));
        }
        let dependencies = consumed.nodes().iter().map(|node| ReplayedBundleDependency {
            node_id: node.id.clone(), artifact_identity: node.identity.clone(),
            chain_digest: node.chain_digest.clone(),
        }).collect();
        Ok(CyclicPosteriorBundleReceipt {
            bundle: ChainBundleRecomputationReceipt {
                bundle_identity: consumed.identity().into(),
                terminal_node: self.inner.chain.terminal_node.clone(),
                terminal_identity: live.identity().into(),
                lowered_program_identity: *self.inner.chain.lowered.identity().as_bytes(),
                dependencies,
            },
            producer_program_identity: *self.inner.producer_program.as_bytes(),
            lowered_program_identity: *self.lowered.identity().as_bytes(),
        })
    }

    /// Export the embedded graph and independently replayable receipt.
    ///
    /// # Errors
    /// Changed live execution or oversized artifact.
    pub fn export_portable(&self, receipt: &CyclicPosteriorBundleReceipt, ctx: &ExecutionContext)
        -> Result<Vec<u8>, CyclicPosteriorBundleError> {
        if self.execute(ctx)? != *receipt {
            return Err(CyclicPosteriorBundleError::Invalid("node bundle receipt differs"));
        }
        let bytes = antecedent_io::to_cbor(&PortableCyclicPosteriorNodeBundleWire {
            version: 1, required_feature: CYCLIC_NODE_BUNDLE_FEATURE.into(),
            bundle_bytes: export_bundle(&self.inner.chain.bundle, "portable-cyclic-node-bundle")
                .map_err(ChainBundleRecomputeError::Bundle)?,
            bundle_identity: self.inner.chain.bundle.identity().into(),
            producer_node: self.producer_node.clone(), terminal_node: self.inner.chain.terminal_node.clone(),
            receipt: receipt.clone(),
        }).map_err(ChainBundleRecomputeError::Io)?;
        if bytes.len() > MAX_BUNDLE_RECOMPUTE_BYTES {
            return Err(CyclicPosteriorBundleError::Invalid("portable node bundle exceeds its bound"));
        }
        Ok(bytes)
    }
}

/// Independently reconstruct and refit a portable native-fit bundle.
///
/// # Errors
/// Changed graph, fit, compiler identity, terminal or receipt.
pub fn consume_portable_cyclic_posterior_node_bundle_recomputation(
    bytes: &[u8], expected_bundle: &str, expected_program: SemanticDigest, ctx: &ExecutionContext,
) -> Result<CyclicPosteriorBundleReceipt, CyclicPosteriorBundleError> {
    if bytes.len() > MAX_BUNDLE_RECOMPUTE_BYTES {
        return Err(CyclicPosteriorBundleError::Invalid("portable node bundle exceeds its bound"));
    }
    let wire: PortableCyclicPosteriorNodeBundleWire = antecedent_io::from_cbor(bytes)
        .map_err(ChainBundleRecomputeError::Io)?;
    if wire.version != 1 || wire.required_feature != CYCLIC_NODE_BUNDLE_FEATURE
        || wire.bundle_identity != expected_bundle
    {
        return Err(CyclicPosteriorBundleError::Invalid("portable node feature or bundle differs"));
    }
    let bundle = CompositionBundle::from_bytes(&wire.bundle_bytes, expected_bundle, &BundleLimits::default())
        .map_err(ChainBundleRecomputeError::Bundle)?;
    let plan = compile_cyclic_posterior_node_bundle_recomputation(
        bundle, &wire.producer_node, &wire.terminal_node, ctx,
    )?;
    if plan.lowered().identity() != expected_program {
        return Err(CyclicPosteriorBundleError::Invalid("portable node compiler differs"));
    }
    let live = plan.execute(ctx)?;
    if live != wire.receipt {
        return Err(CyclicPosteriorBundleError::Invalid("portable node receipt differs"));
    }
    Ok(live)
}

const TRIAL_IPW_FIT_FEATURE: &str = "trial_ipw_transport_fit_bundle_node_v1";
const TRIAL_IPW_FIT_FEATURE_V2: &str = "trial_ipw_transport_fit_bundle_node_v2";
const TRIAL_IPW_NODE_FEATURE: &str = "trial_ipw_node_bundle_recomputation_v1";
const TRIAL_IPW_CHAIN_FACT: &str = "trial_ipw_transport_fit_chain_identity";

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct TrialIpwFitWire {
    version: u32, required_feature: String,
    producer_program: [u8; 32], artifact: Vec<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    replay_seed: Option<u64>,
}

struct TrialIpwFitVerifier { ctx: ExecutionContext }

impl NodeVerifier for TrialIpwFitVerifier {
    fn kind(&self) -> NodeKind { NodeKind::TrialIpwTransportFit }

    fn verify(&self, _node_id: &str, bytes: &[u8], _upstream: &[UpstreamNode<'_>])
        -> Result<VerifiedNode, NodeFailure> {
        let fail = |reason: String| NodeFailure { stage: BundleStage::SwappedEvidence, reason };
        if bytes.len() > MAX_BUNDLE_RECOMPUTE_BYTES {
            return Err(NodeFailure { stage: BundleStage::Oversized, reason: "trial IPW fit node exceeds its bound".into() });
        }
        let wire: TrialIpwFitWire = antecedent_io::from_cbor(bytes)
            .map_err(|error| fail(error.to_string()))?;
        if !fit_node_version_supported(wire.version, &wire.required_feature, wire.replay_seed, TRIAL_IPW_FIT_FEATURE, TRIAL_IPW_FIT_FEATURE_V2) {
            return Err(NodeFailure { stage: BundleStage::IncompatibleVersion, reason: "trial IPW fit feature differs".into() });
        }
        let node_ctx = fit_node_context(&self.ctx, wire.replay_seed);
        let producer = super::trial_ipw_transport_decision::consume_trial_ipw_transport_decision(
            &wire.artifact, SemanticDigest::from_bytes(wire.producer_program), &node_ctx,
        ).map_err(|error| fit_node_failure(&node_ctx, &error))?;
        let mut facts = std::collections::BTreeMap::new();
        facts.insert(TRIAL_IPW_CHAIN_FACT.into(), producer.decision.chain.identity().into());
        Ok(VerifiedNode {
            identity: CanonicalWriter::digest_bytes_hex("antecedent.trial_ipw_transport_fit_node.v1", bytes),
            facts, values: Vec::new(),
        })
    }
}

/// Consumer that independently reidentifies and refits trial-IPW nodes.
/// It requires a runtime context; the generic consumer does not replay these nodes.
#[must_use]
pub fn trial_ipw_bundle_consumer(ctx: &ExecutionContext) -> BundleConsumer {
    native_fit_bundle_consumer(ctx)
}

/// Refusal of a trial-IPW fit node or compiler-owned bundle terminal.
#[derive(Debug, thiserror::Error)]
pub enum TrialIpwBundleError {
    /// Bundle structure, chain or codec refused.
    #[error("trial IPW bundle: {0}")]
    Bundle(#[from] ChainBundleRecomputeError),
    /// Invalid fit or terminal binding.
    #[error("trial IPW bundle: {0}")]
    Invalid(&'static str),
}

/// A trial-IPW fit and live chain, compiled together before replay.
#[derive(Clone, Debug)]
pub struct CompiledTrialIpwNodeBundle {
    chain: CompiledChainBundleRecomputation,
    producer_node: String,
    producer_program: SemanticDigest,
    lowered: Arc<LoweredProgram>,
}

/// Whole-route receipt; no sampling coverage or decision reliability is earned.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrialIpwNodeBundleReceipt {
    /// The independently consumed graph and freshly executed chain.
    pub bundle: ChainBundleRecomputationReceipt,
    /// Trial-IPW producer compiler identity.
    pub producer_program_identity: [u8; 32],
    /// Outer compiler identity.
    pub lowered_program_identity: [u8; 32],
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PortableTrialIpwNodeBundleWire {
    version: u32, required_feature: String, bundle_bytes: Vec<u8>,
    bundle_identity: String, producer_node: String, terminal_node: String,
    receipt: TrialIpwNodeBundleReceipt,
}

/// Compile an embedded trial-IPW fit and directly dependent chain terminal.
/// The node graph, fit bytes and program identity enter a distinct outer
/// six-layer program; compilation does not fit the raw evidence.
///
/// # Errors
/// Missing node or edge, malformed fit declaration or chain compiler refusal.
pub fn compile_trial_ipw_node_bundle_recomputation(
    bundle: CompositionBundle, producer_node: &str, terminal_node: &str, ctx: &ExecutionContext,
) -> Result<CompiledTrialIpwNodeBundle, TrialIpwBundleError> {
    if !bundle.edges().iter().any(|edge| edge.from == producer_node && edge.to == terminal_node) {
        return Err(TrialIpwBundleError::Invalid("trial fit must be a direct terminal dependency"));
    }
    let node = bundle.nodes().iter().find(|node| node.id == producer_node)
        .ok_or(TrialIpwBundleError::Invalid("trial fit node is absent"))?;
    if node.kind != NodeKind::TrialIpwTransportFit {
        return Err(TrialIpwBundleError::Invalid("producer is not a trial IPW fit"));
    }
    let NodeSource::Embedded { bytes } = &node.source else {
        return Err(TrialIpwBundleError::Invalid("trial fit is not embedded"));
    };
    let wire: TrialIpwFitWire = antecedent_io::from_cbor(bytes).map_err(ChainBundleRecomputeError::Io)?;
    if !fit_node_version_supported(wire.version, &wire.required_feature, wire.replay_seed, TRIAL_IPW_FIT_FEATURE, TRIAL_IPW_FIT_FEATURE_V2) {
        return Err(TrialIpwBundleError::Invalid("trial fit feature differs"));
    }
    let producer_program = SemanticDigest::from_bytes(wire.producer_program);
    let fit_digest = CanonicalWriter::digest_bytes("antecedent.trial_ipw_bundle.fit.v1", bytes);
    let terminal = bundle.nodes().iter().find(|node| node.id == terminal_node)
        .ok_or(TrialIpwBundleError::Invalid("trial terminal is absent"))?;
    if terminal.kind != NodeKind::PropagationChain {
        return Err(TrialIpwBundleError::Invalid("trial terminal is not a propagation chain"));
    }
    let NodeSource::Embedded { bytes } = &terminal.source else {
        return Err(TrialIpwBundleError::Invalid("trial terminal is not embedded"));
    };
    let record = PropagationChainRecordWire::decode(bytes)
        .map_err(|error| ChainBundleRecomputeError::Chain(error.to_refusal()))?;
    let child = compile_propagation_chain(record, ctx).map_err(ChainBundleRecomputeError::Chain)?;
    let chain = compile_chain_bundle_recomputation(bundle, terminal_node, child)?;
    let mut writer = CanonicalWriter::new("antecedent.trial_ipw_bundle.question.v1");
    writer.bytes(chain.lowered().identity().as_bytes()).text(producer_node)
        .bytes(producer_program.as_bytes()).bytes(&fit_digest);
    let query = writer.finish();
    let mut scientific = chain.lowered().scientific().clone();
    scientific.target = ProblemTarget::QueryFunctional {
        kind: FunctionalKind::PropagatedChain, outcome: None, query,
    };
    scientific.plan_id = Arc::from("trial_ipw_node_bundle_recomputation.v1");
    scientific.source_digest = Some(fit_digest);
    let identification = IdentificationLayer::NonGraphical(Arc::new(
        NonGraphicalIdentification::SharedEvidenceContract { contract: query },
    ));
    let mut evidence = chain.lowered().evidence().clone();
    let mut sources = evidence.sources.to_vec();
    sources.push(EvidenceSourceRef { role: EvidenceRole::Observations, digest: Some(fit_digest) });
    evidence.sources = Arc::from(sources);
    let mut operators = Vec::new();
    for effect in [PhysicalEffect::Read, PhysicalEffect::Fit] {
        operators.push(PhysicalOperator::declare(effect).build().map_err(|_| {
            TrialIpwBundleError::Invalid("trial fit physical operator cannot be lowered")
        })?);
    }
    operators.extend_from_slice(chain.lowered().physical().operators());
    let physical = PhysicalLayer {
        operators: Arc::from(operators), design: chain.lowered().physical().design(),
    };
    let mut statistical = chain.lowered().statistical().clone();
    statistical.realization = Realization::NestedFit {
        producer_program, method: Arc::from("trial_ipw_transport_effect_decision_v1"), law: NestedFitLaw::Posterior,
    };
    let lowered = Arc::new(StandaloneCompositionLoweredParts {
        cell: ProgramCell::TrialIpwNodeBundleRecomputation, scientific, identification,
        statistical, evidence, physical,
        claim: *chain.lowered().claim(),
    }.finish());
    Ok(CompiledTrialIpwNodeBundle { chain, producer_node: producer_node.into(), producer_program, lowered })
}

impl CompiledTrialIpwNodeBundle {
    /// Complete distinct compiler identity for this trial fit and graph.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram { &self.lowered }

    /// Inspect the outer program without fitting.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        PlanView::of_program(Arc::clone(&self.lowered), PlanStage::Compiled)
    }

    /// Reidentify and refit the source through the bundle consumer, then
    /// independently execute its direct chain terminal.
    ///
    /// # Errors
    /// A failed dependency, changed fitted chain or terminal.
    pub fn execute(&self, ctx: &ExecutionContext) -> Result<TrialIpwNodeBundleReceipt, TrialIpwBundleError> {
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        let bundle = &self.chain.bundle;
        let bytes = export_bundle(bundle, "trial-ipw-node-bundle").map_err(ChainBundleRecomputeError::Bundle)?;
        let consumed = trial_ipw_bundle_consumer(&scoped).consume(
            &bytes, &BundleLimits::default(), bundle.identity(), &SuppliedSources::default(),
        ).map_err(ChainBundleRecomputeError::Bundle)?;
        if !consumed.all_replayed() {
            return Err(TrialIpwBundleError::Invalid("trial bundle dependency did not replay"));
        }
        let fit = consumed.node(&self.producer_node).ok_or(TrialIpwBundleError::Invalid("trial fit disappeared"))?;
        let terminal = bundle.nodes().iter().find(|node| node.id == self.chain.terminal_node)
            .ok_or(TrialIpwBundleError::Invalid("trial terminal disappeared"))?;
        let NodeSource::Embedded { bytes } = &terminal.source else {
            return Err(TrialIpwBundleError::Invalid("trial terminal became a reference"));
        };
        let record = PropagationChainRecordWire::decode(bytes)
            .map_err(|error| ChainBundleRecomputeError::Chain(error.to_refusal()))?;
        record.consume(Some(&terminal.identity), &scoped)
            .map_err(|error| ChainBundleRecomputeError::Chain(error.to_refusal()))?;
        let live = self.chain.chain.execute(&scoped).map_err(ChainBundleRecomputeError::Chain)?;
        if &record != live.record() || terminal.identity != live.identity()
            || fit.facts.get(TRIAL_IPW_CHAIN_FACT).map(String::as_str) != Some(live.identity())
        {
            return Err(TrialIpwBundleError::Invalid("refitted trial source and live chain differ"));
        }
        let dependencies = consumed.nodes().iter().map(|node| ReplayedBundleDependency {
            node_id: node.id.clone(), artifact_identity: node.identity.clone(),
            chain_digest: node.chain_digest.clone(),
        }).collect();
        Ok(TrialIpwNodeBundleReceipt {
            bundle: ChainBundleRecomputationReceipt {
                bundle_identity: consumed.identity().into(), terminal_node: self.chain.terminal_node.clone(),
                terminal_identity: live.identity().into(),
                lowered_program_identity: *self.chain.lowered.identity().as_bytes(), dependencies,
            },
            producer_program_identity: *self.producer_program.as_bytes(),
            lowered_program_identity: *self.lowered.identity().as_bytes(),
        })
    }

    /// Export the graph and verified whole-route receipt for fresh replay.
    ///
    /// # Errors
    /// Changed receipt, codec or byte-bound refusal.
    pub fn export_portable(&self, receipt: &TrialIpwNodeBundleReceipt, ctx: &ExecutionContext)
        -> Result<Vec<u8>, TrialIpwBundleError> {
        if self.execute(ctx)? != *receipt { return Err(TrialIpwBundleError::Invalid("trial bundle receipt differs")); }
        let bytes = antecedent_io::to_cbor(&PortableTrialIpwNodeBundleWire {
            version: 1, required_feature: TRIAL_IPW_NODE_FEATURE.into(),
            bundle_bytes: export_bundle(&self.chain.bundle, "portable-trial-ipw-node-bundle")
                .map_err(ChainBundleRecomputeError::Bundle)?,
            bundle_identity: self.chain.bundle.identity().into(), producer_node: self.producer_node.clone(),
            terminal_node: self.chain.terminal_node.clone(), receipt: receipt.clone(),
        }).map_err(ChainBundleRecomputeError::Io)?;
        if bytes.len() > MAX_BUNDLE_RECOMPUTE_BYTES {
            return Err(TrialIpwBundleError::Invalid("portable trial bundle exceeds its bound"));
        }
        Ok(bytes)
    }
}

/// Independently recompile and replay an embedded trial-IPW bundle.
///
/// # Errors
/// Changed graph, fit, compiler, terminal or receipt.
pub fn consume_portable_trial_ipw_node_bundle_recomputation(
    bytes: &[u8], expected_bundle: &str, expected_program: SemanticDigest, ctx: &ExecutionContext,
) -> Result<TrialIpwNodeBundleReceipt, TrialIpwBundleError> {
    if bytes.len() > MAX_BUNDLE_RECOMPUTE_BYTES {
        return Err(TrialIpwBundleError::Invalid("portable trial bundle exceeds its bound"));
    }
    let wire: PortableTrialIpwNodeBundleWire = antecedent_io::from_cbor(bytes).map_err(ChainBundleRecomputeError::Io)?;
    if wire.version != 1 || wire.required_feature != TRIAL_IPW_NODE_FEATURE || wire.bundle_identity != expected_bundle {
        return Err(TrialIpwBundleError::Invalid("portable trial bundle feature or identity differs"));
    }
    let bundle = CompositionBundle::from_bytes(&wire.bundle_bytes, expected_bundle, &BundleLimits::default())
        .map_err(ChainBundleRecomputeError::Bundle)?;
    let plan = compile_trial_ipw_node_bundle_recomputation(bundle, &wire.producer_node, &wire.terminal_node, ctx)?;
    if plan.lowered().identity() != expected_program {
        return Err(TrialIpwBundleError::Invalid("portable trial bundle compiler differs"));
    }
    let live = plan.execute(ctx)?;
    if live != wire.receipt { return Err(TrialIpwBundleError::Invalid("portable trial bundle receipt differs")); }
    Ok(live)
}

const GRAPH_IMPUTATION_TRANSPORT_FIT_FEATURE: &str = "graph_imputation_transport_fit_bundle_node_v1";
const GRAPH_IMPUTATION_TRANSPORT_FIT_FEATURE_V2: &str = "graph_imputation_transport_fit_bundle_node_v2";
const GRAPH_IMPUTATION_TRANSPORT_NODE_FEATURE: &str = "graph_imputation_transport_node_bundle_recomputation_v1";
const GRAPH_IMPUTATION_TRANSPORT_CHAIN_FACT: &str = "graph_imputation_transport_fit_chain_identity";

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct GraphImputationTransportFitWire {
    version: u32, required_feature: String,
    producer_program: [u8; 32], artifact: Vec<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    replay_seed: Option<u64>,
}

struct GraphImputationTransportFitVerifier { ctx: ExecutionContext }

impl NodeVerifier for GraphImputationTransportFitVerifier {
    fn kind(&self) -> NodeKind { NodeKind::GraphImputationTransportFit }

    fn verify(&self, _node_id: &str, bytes: &[u8], _upstream: &[UpstreamNode<'_>])
        -> Result<VerifiedNode, NodeFailure> {
        let fail = |reason: String| NodeFailure { stage: BundleStage::SwappedEvidence, reason };
        if bytes.len() > MAX_BUNDLE_RECOMPUTE_BYTES {
            return Err(NodeFailure { stage: BundleStage::Oversized, reason: "graph-imputation transport fit node exceeds its bound".into() });
        }
        let wire: GraphImputationTransportFitWire = antecedent_io::from_cbor(bytes)
            .map_err(|error| fail(error.to_string()))?;
        if !fit_node_version_supported(wire.version, &wire.required_feature, wire.replay_seed, GRAPH_IMPUTATION_TRANSPORT_FIT_FEATURE, GRAPH_IMPUTATION_TRANSPORT_FIT_FEATURE_V2) {
            return Err(NodeFailure { stage: BundleStage::IncompatibleVersion, reason: "graph-imputation transport fit feature differs".into() });
        }
        let node_ctx = fit_node_context(&self.ctx, wire.replay_seed);
        let producer = super::graph_imputation_transport_decision::consume_graph_imputation_transport_decision(
            &wire.artifact, SemanticDigest::from_bytes(wire.producer_program), &node_ctx,
        ).map_err(|error| fit_node_failure(&node_ctx, &error))?;
        let mut facts = std::collections::BTreeMap::new();
        facts.insert(GRAPH_IMPUTATION_TRANSPORT_CHAIN_FACT.into(), producer.decision.chain.identity().into());
        Ok(VerifiedNode {
            identity: CanonicalWriter::digest_bytes_hex("antecedent.graph_imputation_transport_fit_node.v1", bytes),
            facts, values: Vec::new(),
        })
    }
}

/// Consumer that independently reidentifies and refits graph-imputation transport nodes.
/// It requires a runtime context; the generic consumer does not replay these nodes.
#[must_use]
pub fn graph_imputation_transport_bundle_consumer(ctx: &ExecutionContext) -> BundleConsumer {
    native_fit_bundle_consumer(ctx)
}

/// Refusal of a graph-imputation transport fit node or compiler-owned bundle terminal.
#[derive(Debug, thiserror::Error)]
pub enum GraphImputationTransportBundleError {
    /// Bundle structure, chain or codec refused.
    #[error("graph-imputation transport bundle: {0}")]
    Bundle(#[from] ChainBundleRecomputeError),
    /// Invalid fit or terminal binding.
    #[error("graph-imputation transport bundle: {0}")]
    Invalid(&'static str),
}

/// A graph-imputation transport fit and live chain, compiled together before replay.
#[derive(Clone, Debug)]
pub struct CompiledGraphImputationTransportNodeBundle {
    chain: CompiledChainBundleRecomputation,
    producer_node: String,
    producer_program: SemanticDigest,
    lowered: Arc<LoweredProgram>,
}

/// Whole-route receipt; no sampling coverage or decision reliability is earned.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GraphImputationTransportNodeBundleReceipt {
    /// The independently consumed graph and freshly executed chain.
    pub bundle: ChainBundleRecomputationReceipt,
    /// Graph-imputation transport producer compiler identity.
    pub producer_program_identity: [u8; 32],
    /// Outer compiler identity.
    pub lowered_program_identity: [u8; 32],
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PortableGraphImputationTransportNodeBundleWire {
    version: u32, required_feature: String, bundle_bytes: Vec<u8>,
    bundle_identity: String, producer_node: String, terminal_node: String,
    receipt: GraphImputationTransportNodeBundleReceipt,
}

/// Compile an embedded graph-imputation transport fit and directly dependent chain terminal.
/// The node graph, fit bytes and program identity enter a distinct outer
/// six-layer program; compilation does not fit the raw evidence.
///
/// # Errors
/// Missing node or edge, malformed fit declaration or chain compiler refusal.
pub fn compile_graph_imputation_transport_node_bundle_recomputation(
    bundle: CompositionBundle, producer_node: &str, terminal_node: &str, ctx: &ExecutionContext,
) -> Result<CompiledGraphImputationTransportNodeBundle, GraphImputationTransportBundleError> {
    if !bundle.edges().iter().any(|edge| edge.from == producer_node && edge.to == terminal_node) {
        return Err(GraphImputationTransportBundleError::Invalid("graph transport fit must be a direct terminal dependency"));
    }
    let node = bundle.nodes().iter().find(|node| node.id == producer_node)
        .ok_or(GraphImputationTransportBundleError::Invalid("graph transport fit node is absent"))?;
    if node.kind != NodeKind::GraphImputationTransportFit {
        return Err(GraphImputationTransportBundleError::Invalid("producer is not a graph-imputation transport fit"));
    }
    let NodeSource::Embedded { bytes } = &node.source else {
        return Err(GraphImputationTransportBundleError::Invalid("graph transport fit is not embedded"));
    };
    let wire: GraphImputationTransportFitWire = antecedent_io::from_cbor(bytes).map_err(ChainBundleRecomputeError::Io)?;
    if !fit_node_version_supported(wire.version, &wire.required_feature, wire.replay_seed, GRAPH_IMPUTATION_TRANSPORT_FIT_FEATURE, GRAPH_IMPUTATION_TRANSPORT_FIT_FEATURE_V2) {
        return Err(GraphImputationTransportBundleError::Invalid("graph transport fit feature differs"));
    }
    let producer_program = SemanticDigest::from_bytes(wire.producer_program);
    let fit_digest = CanonicalWriter::digest_bytes("antecedent.graph_imputation_transport_bundle.fit.v1", bytes);
    let terminal = bundle.nodes().iter().find(|node| node.id == terminal_node)
        .ok_or(GraphImputationTransportBundleError::Invalid("graph transport terminal is absent"))?;
    if terminal.kind != NodeKind::PropagationChain {
        return Err(GraphImputationTransportBundleError::Invalid("graph transport terminal is not a propagation chain"));
    }
    let NodeSource::Embedded { bytes } = &terminal.source else {
        return Err(GraphImputationTransportBundleError::Invalid("graph transport terminal is not embedded"));
    };
    let record = PropagationChainRecordWire::decode(bytes)
        .map_err(|error| ChainBundleRecomputeError::Chain(error.to_refusal()))?;
    let child = compile_propagation_chain(record, ctx).map_err(ChainBundleRecomputeError::Chain)?;
    let chain = compile_chain_bundle_recomputation(bundle, terminal_node, child)?;
    let mut writer = CanonicalWriter::new("antecedent.graph_imputation_transport_bundle.question.v1");
    writer.bytes(chain.lowered().identity().as_bytes()).text(producer_node)
        .bytes(producer_program.as_bytes()).bytes(&fit_digest);
    let query = writer.finish();
    let mut scientific = chain.lowered().scientific().clone();
    scientific.target = ProblemTarget::QueryFunctional {
        kind: FunctionalKind::PropagatedChain, outcome: None, query,
    };
    scientific.plan_id = Arc::from("graph_imputation_transport_node_bundle_recomputation.v1");
    scientific.source_digest = Some(fit_digest);
    let identification = IdentificationLayer::NonGraphical(Arc::new(
        NonGraphicalIdentification::SharedEvidenceContract { contract: query },
    ));
    let mut evidence = chain.lowered().evidence().clone();
    let mut sources = evidence.sources.to_vec();
    sources.push(EvidenceSourceRef { role: EvidenceRole::Observations, digest: Some(fit_digest) });
    evidence.sources = Arc::from(sources);
    let mut operators = Vec::new();
    for effect in [PhysicalEffect::Read, PhysicalEffect::Fit] {
        operators.push(PhysicalOperator::declare(effect).build().map_err(|_| {
            GraphImputationTransportBundleError::Invalid("graph transport fit physical operator cannot be lowered")
        })?);
    }
    operators.extend_from_slice(chain.lowered().physical().operators());
    let physical = PhysicalLayer {
        operators: Arc::from(operators), design: chain.lowered().physical().design(),
    };
    let mut statistical = chain.lowered().statistical().clone();
    statistical.realization = Realization::NestedFit {
        producer_program, method: Arc::from("graph_imputation_transport_decision"), law: NestedFitLaw::Posterior,
    };
    let lowered = Arc::new(StandaloneCompositionLoweredParts {
        cell: ProgramCell::GraphImputationTransportNodeBundleRecomputation, scientific, identification,
        statistical, evidence, physical,
        claim: *chain.lowered().claim(),
    }.finish());
    Ok(CompiledGraphImputationTransportNodeBundle { chain, producer_node: producer_node.into(), producer_program, lowered })
}

impl CompiledGraphImputationTransportNodeBundle {
    /// Complete distinct compiler identity for this graph transport fit and graph.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram { &self.lowered }

    /// Inspect the outer program without fitting.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        PlanView::of_program(Arc::clone(&self.lowered), PlanStage::Compiled)
    }

    /// Reidentify and refit the source through the bundle consumer, then
    /// independently execute its direct chain terminal.
    ///
    /// # Errors
    /// A failed dependency, changed fitted chain or terminal.
    pub fn execute(&self, ctx: &ExecutionContext) -> Result<GraphImputationTransportNodeBundleReceipt, GraphImputationTransportBundleError> {
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        let bundle = &self.chain.bundle;
        let bytes = export_bundle(bundle, "graph-transport-node-bundle").map_err(ChainBundleRecomputeError::Bundle)?;
        let consumed = graph_imputation_transport_bundle_consumer(&scoped).consume(
            &bytes, &BundleLimits::default(), bundle.identity(), &SuppliedSources::default(),
        ).map_err(ChainBundleRecomputeError::Bundle)?;
        if !consumed.all_replayed() {
            return Err(GraphImputationTransportBundleError::Invalid("graph transport bundle dependency did not replay"));
        }
        let fit = consumed.node(&self.producer_node).ok_or(GraphImputationTransportBundleError::Invalid("graph transport fit disappeared"))?;
        let terminal = bundle.nodes().iter().find(|node| node.id == self.chain.terminal_node)
            .ok_or(GraphImputationTransportBundleError::Invalid("graph transport terminal disappeared"))?;
        let NodeSource::Embedded { bytes } = &terminal.source else {
            return Err(GraphImputationTransportBundleError::Invalid("graph transport terminal became a reference"));
        };
        let record = PropagationChainRecordWire::decode(bytes)
            .map_err(|error| ChainBundleRecomputeError::Chain(error.to_refusal()))?;
        record.consume(Some(&terminal.identity), &scoped)
            .map_err(|error| ChainBundleRecomputeError::Chain(error.to_refusal()))?;
        let live = self.chain.chain.execute(&scoped).map_err(ChainBundleRecomputeError::Chain)?;
        if &record != live.record() || terminal.identity != live.identity()
            || fit.facts.get(GRAPH_IMPUTATION_TRANSPORT_CHAIN_FACT).map(String::as_str) != Some(live.identity())
        {
            return Err(GraphImputationTransportBundleError::Invalid("refitted graph transport source and live chain differ"));
        }
        let dependencies = consumed.nodes().iter().map(|node| ReplayedBundleDependency {
            node_id: node.id.clone(), artifact_identity: node.identity.clone(),
            chain_digest: node.chain_digest.clone(),
        }).collect();
        Ok(GraphImputationTransportNodeBundleReceipt {
            bundle: ChainBundleRecomputationReceipt {
                bundle_identity: consumed.identity().into(), terminal_node: self.chain.terminal_node.clone(),
                terminal_identity: live.identity().into(),
                lowered_program_identity: *self.chain.lowered.identity().as_bytes(), dependencies,
            },
            producer_program_identity: *self.producer_program.as_bytes(),
            lowered_program_identity: *self.lowered.identity().as_bytes(),
        })
    }

    /// Export the graph and verified whole-route receipt for fresh replay.
    ///
    /// # Errors
    /// Changed receipt, codec or byte-bound refusal.
    pub fn export_portable(&self, receipt: &GraphImputationTransportNodeBundleReceipt, ctx: &ExecutionContext)
        -> Result<Vec<u8>, GraphImputationTransportBundleError> {
        if self.execute(ctx)? != *receipt { return Err(GraphImputationTransportBundleError::Invalid("graph transport bundle receipt differs")); }
        let bytes = antecedent_io::to_cbor(&PortableGraphImputationTransportNodeBundleWire {
            version: 1, required_feature: GRAPH_IMPUTATION_TRANSPORT_NODE_FEATURE.into(),
            bundle_bytes: export_bundle(&self.chain.bundle, "portable-graph-transport-node-bundle")
                .map_err(ChainBundleRecomputeError::Bundle)?,
            bundle_identity: self.chain.bundle.identity().into(), producer_node: self.producer_node.clone(),
            terminal_node: self.chain.terminal_node.clone(), receipt: receipt.clone(),
        }).map_err(ChainBundleRecomputeError::Io)?;
        if bytes.len() > MAX_BUNDLE_RECOMPUTE_BYTES {
            return Err(GraphImputationTransportBundleError::Invalid("portable graph transport bundle exceeds its bound"));
        }
        Ok(bytes)
    }
}

/// Independently recompile and replay an embedded graph-imputation transport bundle.
///
/// # Errors
/// Changed graph, fit, compiler, terminal or receipt.
pub fn consume_portable_graph_imputation_transport_node_bundle_recomputation(
    bytes: &[u8], expected_bundle: &str, expected_program: SemanticDigest, ctx: &ExecutionContext,
) -> Result<GraphImputationTransportNodeBundleReceipt, GraphImputationTransportBundleError> {
    if bytes.len() > MAX_BUNDLE_RECOMPUTE_BYTES {
        return Err(GraphImputationTransportBundleError::Invalid("portable graph transport bundle exceeds its bound"));
    }
    let wire: PortableGraphImputationTransportNodeBundleWire = antecedent_io::from_cbor(bytes).map_err(ChainBundleRecomputeError::Io)?;
    if wire.version != 1 || wire.required_feature != GRAPH_IMPUTATION_TRANSPORT_NODE_FEATURE || wire.bundle_identity != expected_bundle {
        return Err(GraphImputationTransportBundleError::Invalid("portable graph transport bundle feature or identity differs"));
    }
    let bundle = CompositionBundle::from_bytes(&wire.bundle_bytes, expected_bundle, &BundleLimits::default())
        .map_err(ChainBundleRecomputeError::Bundle)?;
    let plan = compile_graph_imputation_transport_node_bundle_recomputation(bundle, &wire.producer_node, &wire.terminal_node, ctx)?;
    if plan.lowered().identity() != expected_program {
        return Err(GraphImputationTransportBundleError::Invalid("portable graph transport bundle compiler differs"));
    }
    let live = plan.execute(ctx)?;
    if live != wire.receipt { return Err(GraphImputationTransportBundleError::Invalid("portable graph transport bundle receipt differs")); }
    Ok(live)
}

const JOINT_TRANSPORT_EFFECT_FIT_FEATURE: &str = "joint_transport_effect_fit_bundle_node_v1";
const JOINT_TRANSPORT_EFFECT_FIT_FEATURE_V2: &str = "joint_transport_effect_fit_bundle_node_v2";
const JOINT_TRANSPORT_EFFECT_NODE_FEATURE: &str = "joint_transport_effect_node_bundle_recomputation_v1";
const JOINT_TRANSPORT_EFFECT_CHAIN_FACT: &str = "joint_transport_effect_fit_chain_identity";

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct JointTransportEffectFitWire {
    version: u32, required_feature: String,
    producer_program: [u8; 32], artifact: Vec<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    replay_seed: Option<u64>,
}

struct JointTransportEffectFitVerifier { ctx: ExecutionContext }

impl NodeVerifier for JointTransportEffectFitVerifier {
    fn kind(&self) -> NodeKind { NodeKind::JointTransportEffectFit }

    fn verify(&self, _node_id: &str, bytes: &[u8], _upstream: &[UpstreamNode<'_>])
        -> Result<VerifiedNode, NodeFailure> {
        let fail = |reason: String| NodeFailure { stage: BundleStage::SwappedEvidence, reason };
        if bytes.len() > MAX_BUNDLE_RECOMPUTE_BYTES {
            return Err(NodeFailure { stage: BundleStage::Oversized, reason: "joint transport effect fit node exceeds its bound".into() });
        }
        let wire: JointTransportEffectFitWire = antecedent_io::from_cbor(bytes)
            .map_err(|error| fail(error.to_string()))?;
        if !fit_node_version_supported(wire.version, &wire.required_feature, wire.replay_seed, JOINT_TRANSPORT_EFFECT_FIT_FEATURE, JOINT_TRANSPORT_EFFECT_FIT_FEATURE_V2) {
            return Err(NodeFailure { stage: BundleStage::IncompatibleVersion, reason: "joint transport effect fit feature differs".into() });
        }
        let node_ctx = fit_node_context(&self.ctx, wire.replay_seed);
        let producer = super::joint_transport_effect_decision::consume_joint_transport_effect_decision(
            &wire.artifact, SemanticDigest::from_bytes(wire.producer_program), &node_ctx,
        ).map_err(|error| fit_node_failure(&node_ctx, &error))?;
        let mut facts = std::collections::BTreeMap::new();
        facts.insert(JOINT_TRANSPORT_EFFECT_CHAIN_FACT.into(), producer.decision.chain.identity().into());
        Ok(VerifiedNode {
            identity: CanonicalWriter::digest_bytes_hex("antecedent.joint_transport_effect_fit_node.v1", bytes),
            facts, values: Vec::new(),
        })
    }
}

/// Consumer that independently reidentifies and refits joint transport effect nodes.
/// It requires a runtime context; the generic consumer does not replay these nodes.
#[must_use]
pub fn joint_transport_effect_bundle_consumer(ctx: &ExecutionContext) -> BundleConsumer {
    native_fit_bundle_consumer(ctx)
}

/// Refusal of a joint transport effect fit node or compiler-owned bundle terminal.
#[derive(Debug, thiserror::Error)]
pub enum JointTransportEffectBundleError {
    /// Bundle structure, chain or codec refused.
    #[error("joint transport effect bundle: {0}")]
    Bundle(#[from] ChainBundleRecomputeError),
    /// Invalid fit or terminal binding.
    #[error("joint transport effect bundle: {0}")]
    Invalid(&'static str),
}

/// A joint transport effect fit and live chain, compiled together before replay.
#[derive(Clone, Debug)]
pub struct CompiledJointTransportEffectNodeBundle {
    chain: CompiledChainBundleRecomputation,
    producer_node: String,
    producer_program: SemanticDigest,
    lowered: Arc<LoweredProgram>,
}

/// Whole-route receipt; no sampling coverage or decision reliability is earned.
#[derive(Clone, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct JointTransportEffectNodeBundleReceipt {
    /// The independently consumed graph and freshly executed chain.
    pub bundle: ChainBundleRecomputationReceipt,
    /// Joint source-target transport producer compiler identity.
    pub producer_program_identity: [u8; 32],
    /// Outer compiler identity.
    pub lowered_program_identity: [u8; 32],
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PortableJointTransportEffectNodeBundleWire {
    version: u32, required_feature: String, bundle_bytes: Vec<u8>,
    bundle_identity: String, producer_node: String, terminal_node: String,
    receipt: JointTransportEffectNodeBundleReceipt,
}

/// Compile an embedded joint transport effect fit and directly dependent chain terminal.
/// The node graph, fit bytes and program identity enter a distinct outer
/// six-layer program; compilation does not fit the raw evidence.
///
/// # Errors
/// Missing node or edge, malformed fit declaration or chain compiler refusal.
pub fn compile_joint_transport_effect_node_bundle_recomputation(
    bundle: CompositionBundle, producer_node: &str, terminal_node: &str, ctx: &ExecutionContext,
) -> Result<CompiledJointTransportEffectNodeBundle, JointTransportEffectBundleError> {
    if !bundle.edges().iter().any(|edge| edge.from == producer_node && edge.to == terminal_node) {
        return Err(JointTransportEffectBundleError::Invalid("joint transport effect fit must be a direct terminal dependency"));
    }
    let node = bundle.nodes().iter().find(|node| node.id == producer_node)
        .ok_or(JointTransportEffectBundleError::Invalid("joint transport effect fit node is absent"))?;
    if node.kind != NodeKind::JointTransportEffectFit {
        return Err(JointTransportEffectBundleError::Invalid("producer is not a joint transport effect fit"));
    }
    let NodeSource::Embedded { bytes } = &node.source else {
        return Err(JointTransportEffectBundleError::Invalid("joint transport effect fit is not embedded"));
    };
    let wire: JointTransportEffectFitWire = antecedent_io::from_cbor(bytes).map_err(ChainBundleRecomputeError::Io)?;
    if !fit_node_version_supported(wire.version, &wire.required_feature, wire.replay_seed, JOINT_TRANSPORT_EFFECT_FIT_FEATURE, JOINT_TRANSPORT_EFFECT_FIT_FEATURE_V2) {
        return Err(JointTransportEffectBundleError::Invalid("joint transport effect fit feature differs"));
    }
    let producer_program = SemanticDigest::from_bytes(wire.producer_program);
    let fit_digest = CanonicalWriter::digest_bytes("antecedent.joint_transport_effect_bundle.fit.v1", bytes);
    let terminal = bundle.nodes().iter().find(|node| node.id == terminal_node)
        .ok_or(JointTransportEffectBundleError::Invalid("joint transport effect terminal is absent"))?;
    if terminal.kind != NodeKind::PropagationChain {
        return Err(JointTransportEffectBundleError::Invalid("joint transport effect terminal is not a propagation chain"));
    }
    let NodeSource::Embedded { bytes } = &terminal.source else {
        return Err(JointTransportEffectBundleError::Invalid("joint transport effect terminal is not embedded"));
    };
    let record = PropagationChainRecordWire::decode(bytes)
        .map_err(|error| ChainBundleRecomputeError::Chain(error.to_refusal()))?;
    let child = compile_propagation_chain(record, ctx).map_err(ChainBundleRecomputeError::Chain)?;
    let chain = compile_chain_bundle_recomputation(bundle, terminal_node, child)?;
    let mut writer = CanonicalWriter::new("antecedent.joint_transport_effect_bundle.question.v1");
    writer.bytes(chain.lowered().identity().as_bytes()).text(producer_node)
        .bytes(producer_program.as_bytes()).bytes(&fit_digest);
    let query = writer.finish();
    let mut scientific = chain.lowered().scientific().clone();
    scientific.target = ProblemTarget::QueryFunctional {
        kind: FunctionalKind::PropagatedChain, outcome: None, query,
    };
    scientific.plan_id = Arc::from("joint_transport_effect_node_bundle_recomputation.v1");
    scientific.source_digest = Some(fit_digest);
    let identification = IdentificationLayer::NonGraphical(Arc::new(
        NonGraphicalIdentification::SharedEvidenceContract { contract: query },
    ));
    let mut evidence = chain.lowered().evidence().clone();
    let mut sources = evidence.sources.to_vec();
    sources.push(EvidenceSourceRef { role: EvidenceRole::Observations, digest: Some(fit_digest) });
    evidence.sources = Arc::from(sources);
    let mut operators = Vec::new();
    for effect in [PhysicalEffect::Read, PhysicalEffect::Fit] {
        operators.push(PhysicalOperator::declare(effect).build().map_err(|_| {
            JointTransportEffectBundleError::Invalid("joint transport effect fit physical operator cannot be lowered")
        })?);
    }
    operators.extend_from_slice(chain.lowered().physical().operators());
    let physical = PhysicalLayer {
        operators: Arc::from(operators), design: chain.lowered().physical().design(),
    };
    let mut statistical = chain.lowered().statistical().clone();
    statistical.realization = Realization::NestedFit {
        producer_program, method: Arc::from("joint_transport_effect_decision_v1"), law: NestedFitLaw::Posterior,
    };
    let lowered = Arc::new(StandaloneCompositionLoweredParts {
        cell: ProgramCell::JointTransportEffectNodeBundleRecomputation, scientific, identification,
        statistical, evidence, physical,
        claim: *chain.lowered().claim(),
    }.finish());
    Ok(CompiledJointTransportEffectNodeBundle { chain, producer_node: producer_node.into(), producer_program, lowered })
}

impl CompiledJointTransportEffectNodeBundle {
    /// Complete distinct compiler identity for this joint transport effect fit and graph.
    #[must_use]
    pub fn lowered(&self) -> &LoweredProgram { &self.lowered }

    /// Inspect the outer program without fitting.
    #[must_use]
    pub fn plan_view(&self) -> PlanView {
        PlanView::of_program(Arc::clone(&self.lowered), PlanStage::Compiled)
    }

    /// Reidentify and refit the source through the bundle consumer, then
    /// independently execute its direct chain terminal.
    ///
    /// # Errors
    /// A failed dependency, changed fitted chain or terminal.
    pub fn execute(&self, ctx: &ExecutionContext) -> Result<JointTransportEffectNodeBundleReceipt, JointTransportEffectBundleError> {
        let scoped = ctx.enter_lowered_dispatch(self.lowered.identity());
        let bundle = &self.chain.bundle;
        let bytes = export_bundle(bundle, "joint-transport-effect-node-bundle").map_err(ChainBundleRecomputeError::Bundle)?;
        let consumed = joint_transport_effect_bundle_consumer(&scoped).consume(
            &bytes, &BundleLimits::default(), bundle.identity(), &SuppliedSources::default(),
        ).map_err(ChainBundleRecomputeError::Bundle)?;
        if !consumed.all_replayed() {
            return Err(JointTransportEffectBundleError::Invalid("joint transport effect bundle dependency did not replay"));
        }
        let fit = consumed.node(&self.producer_node).ok_or(JointTransportEffectBundleError::Invalid("joint transport effect fit disappeared"))?;
        let terminal = bundle.nodes().iter().find(|node| node.id == self.chain.terminal_node)
            .ok_or(JointTransportEffectBundleError::Invalid("joint transport effect terminal disappeared"))?;
        let NodeSource::Embedded { bytes } = &terminal.source else {
            return Err(JointTransportEffectBundleError::Invalid("joint transport effect terminal became a reference"));
        };
        let record = PropagationChainRecordWire::decode(bytes)
            .map_err(|error| ChainBundleRecomputeError::Chain(error.to_refusal()))?;
        record.consume(Some(&terminal.identity), &scoped)
            .map_err(|error| ChainBundleRecomputeError::Chain(error.to_refusal()))?;
        let live = self.chain.chain.execute(&scoped).map_err(ChainBundleRecomputeError::Chain)?;
        if &record != live.record() || terminal.identity != live.identity()
            || fit.facts.get(JOINT_TRANSPORT_EFFECT_CHAIN_FACT).map(String::as_str) != Some(live.identity())
        {
            return Err(JointTransportEffectBundleError::Invalid("refitted joint transport effect source and live chain differ"));
        }
        let dependencies = consumed.nodes().iter().map(|node| ReplayedBundleDependency {
            node_id: node.id.clone(), artifact_identity: node.identity.clone(),
            chain_digest: node.chain_digest.clone(),
        }).collect();
        Ok(JointTransportEffectNodeBundleReceipt {
            bundle: ChainBundleRecomputationReceipt {
                bundle_identity: consumed.identity().into(), terminal_node: self.chain.terminal_node.clone(),
                terminal_identity: live.identity().into(),
                lowered_program_identity: *self.chain.lowered.identity().as_bytes(), dependencies,
            },
            producer_program_identity: *self.producer_program.as_bytes(),
            lowered_program_identity: *self.lowered.identity().as_bytes(),
        })
    }

    /// Export the graph and verified whole-route receipt for fresh replay.
    ///
    /// # Errors
    /// Changed receipt, codec or byte-bound refusal.
    pub fn export_portable(&self, receipt: &JointTransportEffectNodeBundleReceipt, ctx: &ExecutionContext)
        -> Result<Vec<u8>, JointTransportEffectBundleError> {
        if self.execute(ctx)? != *receipt { return Err(JointTransportEffectBundleError::Invalid("joint transport effect bundle receipt differs")); }
        let bytes = antecedent_io::to_cbor(&PortableJointTransportEffectNodeBundleWire {
            version: 1, required_feature: JOINT_TRANSPORT_EFFECT_NODE_FEATURE.into(),
            bundle_bytes: export_bundle(&self.chain.bundle, "portable-joint-transport-effect-node-bundle")
                .map_err(ChainBundleRecomputeError::Bundle)?,
            bundle_identity: self.chain.bundle.identity().into(), producer_node: self.producer_node.clone(),
            terminal_node: self.chain.terminal_node.clone(), receipt: receipt.clone(),
        }).map_err(ChainBundleRecomputeError::Io)?;
        if bytes.len() > MAX_BUNDLE_RECOMPUTE_BYTES {
            return Err(JointTransportEffectBundleError::Invalid("portable joint transport effect bundle exceeds its bound"));
        }
        Ok(bytes)
    }
}

/// Independently recompile and replay an embedded joint transport effect bundle.
///
/// # Errors
/// Changed graph, fit, compiler, terminal or receipt.
pub fn consume_portable_joint_transport_effect_node_bundle_recomputation(
    bytes: &[u8], expected_bundle: &str, expected_program: SemanticDigest, ctx: &ExecutionContext,
) -> Result<JointTransportEffectNodeBundleReceipt, JointTransportEffectBundleError> {
    if bytes.len() > MAX_BUNDLE_RECOMPUTE_BYTES {
        return Err(JointTransportEffectBundleError::Invalid("portable joint transport effect bundle exceeds its bound"));
    }
    let wire: PortableJointTransportEffectNodeBundleWire = antecedent_io::from_cbor(bytes).map_err(ChainBundleRecomputeError::Io)?;
    if wire.version != 1 || wire.required_feature != JOINT_TRANSPORT_EFFECT_NODE_FEATURE || wire.bundle_identity != expected_bundle {
        return Err(JointTransportEffectBundleError::Invalid("portable joint transport effect bundle feature or identity differs"));
    }
    let bundle = CompositionBundle::from_bytes(&wire.bundle_bytes, expected_bundle, &BundleLimits::default())
        .map_err(ChainBundleRecomputeError::Bundle)?;
    let plan = compile_joint_transport_effect_node_bundle_recomputation(bundle, &wire.producer_node, &wire.terminal_node, ctx)?;
    if plan.lowered().identity() != expected_program {
        return Err(JointTransportEffectBundleError::Invalid("portable joint transport effect bundle compiler differs"));
    }
    let live = plan.execute(ctx)?;
    if live != wire.receipt { return Err(JointTransportEffectBundleError::Invalid("portable joint transport effect bundle receipt differs")); }
    Ok(live)
}

#[cfg(test)]
mod compiler_recomputation_tests {
    use super::*;
    use antecedent_contract::propagation_chain::{
        ChainReading, LawMeaning, StageKind, TableConstruction,
    };
    use antecedent_io::propagation_chain_record::{
        ChainCoefficientWire, ChainOpWire, ChainSourceWire, ChainStageWire,
    };
    use crate::analysis::propagation_chain::{PropagationChain, compile_propagation_chain};

    #[test]
    fn a_bundle_terminal_is_recomputed_by_its_compiler_and_replayed() {
        let ctx = ExecutionContext::for_tests(23);
        let draft = PropagationChain::draft(
            TableConstruction::JointDraws, ChainReading::Spread, "output",
            vec![ChainSourceWire::draws(
                "source", LawMeaning::Sampling, "aligned-bundle", &[1.0, 2.0, 3.0],
            )],
            vec![ChainStageWire::new(
                "output", StageKind::DerivedQuantity, &["source"],
                ChainOpWire::affine(
                    ChainCoefficientWire::fixed(2.0), ChainCoefficientWire::fixed(0.0),
                ),
            )],
        );
        let chain = compile_propagation_chain(draft, &ctx).expect("chain compiler");
        let sealed = chain.execute(&ctx).expect("source execution");
        let mut builder = BundleBuilder::new();
        builder.add_artifact(
            Some("terminal"), NodeKind::PropagationChain,
            &sealed.export().expect("chain bytes"),
        ).expect("embedded chain");
        let bundle = builder.build().expect("bundle");
        let plan = compile_chain_bundle_recomputation(bundle, "terminal", chain)
            .expect("bundle compiler");
        assert_eq!(plan.plan_view().route().as_str(), "BundleRecomputation");
        let receipt = plan.execute(&ctx).expect("live bundle replay");
        assert_eq!(receipt.dependencies.len(), 1);
        assert_eq!(receipt.terminal_identity, sealed.identity());
        let bytes = plan.export(&receipt, &ctx).expect("recalculation receipt");
        assert_eq!(plan.consume(&bytes, plan.lowered().identity(), &ctx)
            .expect("independent replay"), receipt);
        let portable = plan.export_portable(&receipt, &ctx).expect("portable artifact");
        assert_eq!(consume_portable_chain_bundle_recomputation(
            &portable, plan.bundle.identity(), plan.lowered().identity(), &ctx,
        ).expect("fresh compiler replay"), receipt);
    }
}

#[cfg(test)]
#[path = "composition_cyclic_bundle_tests.rs"]
mod cyclic_bundle_tests;

#[cfg(test)]
#[path = "composition_joint_transport_bundle_tests.rs"]
mod joint_transport_bundle_tests;
