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
//! Nothing here replays an executable study. A reference node is never claimed
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
    SuppliedData, SuppliedProvider, SuppliedSources,
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
use antecedent_core::{
    ByteEstimates, CanonicalWriter, ExecutionContext, PhysicalEffect,
    PhysicalOperator, SemanticDigest,
};
use antecedent_contract::propagation_chain::ChainRefusal;
use antecedent_io::propagation_chain_record::PropagationChainRecordWire;
use antecedent_io::IoError;
use serde::{Deserialize, Serialize};
use super::propagation_chain::PropagationChainPlan;
use crate::planner::lowered::{
    Composition, EvidenceDecisionProgram, EvidenceRole, EvidenceSourceRef,
    FunctionalKind, IdentificationLayer, LoweredProgram,
    NonGraphicalIdentification, PhysicalLayer, ProblemTarget, ProgramCell,
    ProgramIdentityScope, ScientificProblem, StandaloneCompositionLoweredParts,
};
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
    }
}
