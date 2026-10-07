//! C3 facade: build, export and independently consume a portable composed result.
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

    /// An empty builder under explicit bounds.
    #[must_use]
    pub fn with_limits(limits: BundleLimits) -> Self {
        Self { nodes: Vec::new(), dependencies: Vec::new(), limits }
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
