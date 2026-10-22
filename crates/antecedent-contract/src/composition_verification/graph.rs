//! The claim-level dependency graph: nodes are claims and objects with
//! content identities, edges are typed dependencies.
//!
//! A [`CompositionGraph`] is a snapshot. Two snapshots (earlier and current)
//! are compared by [`super::revalidate`]. The graph stores nodes and edges in
//! ordered containers, so declaration order never changes an identity, a
//! topological order or a classification.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use antecedent_core::{CanonicalWriter, SemanticDigest};

use crate::evidence_ledger::SourceId;
use antecedent_core::named_variants;

/// Version word written into every graph identity.
pub const COMPOSITION_GRAPH_VERSION: u64 = 1;
/// Most nodes in one graph.
pub const MAX_NODES: usize = 512;
/// Most edges in one graph.
pub const MAX_EDGES: usize = 4_096;
/// Most assumptions, unsupported coordinates or evidence sources on one node.
pub const MAX_ITEMS_PER_NODE: usize = 64;
/// Longest text field in bytes.
pub const MAX_TEXT_BYTES: usize = 1_024;

/// What a node stands for.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum NodeKind {
    /// A cohort or study source (observations).
    CohortSource,
    /// A data snapshot.
    Data,
    /// A measurement model relating an indicator to a construct.
    MeasurementModel,
    /// A composition of fitted mechanisms.
    MechanismComposition,
    /// A scientific bridge between quantities.
    Bridge,
    /// A result registered in the evidence ledger.
    LedgerResult,
    /// A transported claim about a target population.
    TransportedResult,
    /// A decision resting on claims.
    Decision,
    /// An external provider at an exact request.
    ExternalProvider,
    /// A selection diagram.
    SelectionDiagram,
    /// A target population.
    TargetPopulation,
    /// Any other reported claim.
    Claim,
}

named_variants! {
    NodeKind {
        CohortSource => "cohort_source",
        Data => "data",
        MeasurementModel => "measurement_model",
        MechanismComposition => "mechanism_composition",
        Bridge => "bridge",
        LedgerResult => "ledger_result",
        TransportedResult => "transported_result",
        Decision => "decision",
        ExternalProvider => "external_provider",
        SelectionDiagram => "selection_diagram",
        TargetPopulation => "target_population",
        Claim => "claim",
    }
}

impl NodeKind {
    /// Whether the node is a boundary transformation (it changes the meaning,
    /// population or representation of what flows through it).
    #[must_use]
    pub const fn is_boundary(self) -> bool {
        matches!(
            self,
            Self::MeasurementModel
                | Self::MechanismComposition
                | Self::Bridge
                | Self::TransportedResult
        )
    }
}

/// The kind of dependence an edge states. The kind decides how a change or a
/// withdrawal upstream reaches the downstream claim.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum DependencyKind {
    /// The downstream numbers were computed from the upstream data.
    Data,
    /// The downstream result counts the upstream evidence.
    Evidence,
    /// The downstream result uses a model or representation at a version.
    ModelVersion,
    /// The downstream coordinates lie inside the upstream support.
    Support,
    /// The downstream result relies on the upstream provider's trust.
    ProviderTrust,
    /// The downstream claim inherits the upstream calibration.
    Calibration,
    /// The downstream result reuses the upstream structural derivation
    /// (a graph, an identification, a bridge form), not its fit.
    StructuralDerivation,
}

named_variants! {
    DependencyKind {
        Data => "data",
        Evidence => "evidence",
        ModelVersion => "model_version",
        Support => "support",
        ProviderTrust => "provider_trust",
        Calibration => "calibration",
        StructuralDerivation => "structural_derivation",
    }
}

/// Whether a node's own coordinates are supported.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SupportStanding {
    /// Every coordinate the node answers lies inside its support.
    Supported,
    /// These coordinates lie outside the support. The node is not licensed.
    Unsupported {
        /// Named coordinates (sorted, distinct, non-empty).
        coordinates: Vec<String>,
    },
}

/// Whether a provider node's trust holds at its current identity.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[non_exhaustive]
pub enum TrustStanding {
    /// The node is not a provider; trust is not at issue.
    NotApplicable,
    /// The provider's trust holds at this identity.
    Held,
    /// The provider's trust no longer holds (replaced, unverified, revoked).
    Withdrawn,
    /// The producer did not state whether trust holds. Unknown standing is not
    /// a license: the node is withdrawn for provider trust until it is stated.
    Undeclared,
}

named_variants! {
    TrustStanding {
        NotApplicable => "not_applicable",
        Held => "held",
        Withdrawn => "withdrawn",
        Undeclared => "undeclared",
    }
}

impl TrustStanding {
    /// The standing of a provider at a verification level. An unverified
    /// provider holds no trust; every other level holds the trust it was
    /// granted. Metadata never raises the level.
    #[must_use]
    pub const fn of_provider(trust: crate::ProviderTrust) -> Self {
        match trust {
            crate::ProviderTrust::Unverified => Self::Withdrawn,
            _ => Self::Held,
        }
    }
}

/// What a node says about its uncertainty guarantee.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CalibrationDeclaration {
    /// A point value only. No interval, posterior or probability guarantee is
    /// claimed, so no calibration is needed.
    PointOnly,
    /// An uncertainty statement is wanted at this node but no record backs it.
    Required,
    /// A calibration record established at exactly this coordinate.
    Held {
        /// Identity of the calibration record.
        record: SemanticDigest,
        /// Coordinate the record was established at (method, target
        /// population, regime, quantity); it must equal the node's coordinate.
        coordinate: SemanticDigest,
    },
    /// A calibration record claimed from an upstream node.
    Inherited {
        /// The upstream node the record is claimed from.
        from: String,
        /// Identity of the calibration record.
        record: SemanticDigest,
        /// The coordinate the record is claimed to hold at.
        coordinate: SemanticDigest,
    },
}

/// The three standings a claim rests on beyond its identity: whether its
/// coordinates are supported, whether its provider's trust holds, and what
/// uncertainty guarantee it states.
///
/// A producing route states all three explicitly when it builds a node, so
/// that reusing a stage artifact never reads as a license by default. A route
/// that cannot state one uses [`Self::undeclared`], which is refused rather
/// than assumed fine.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClaimStanding {
    /// Whether the node's own coordinates are supported.
    pub support: SupportStanding,
    /// Whether the node's provider trust holds.
    pub trust: TrustStanding,
    /// The node's uncertainty guarantee.
    pub calibration: CalibrationDeclaration,
}

impl ClaimStanding {
    /// A natively computed point value: supported, no provider, no interval or
    /// posterior guarantee claimed.
    #[must_use]
    pub const fn native_point() -> Self {
        Self {
            support: SupportStanding::Supported,
            trust: TrustStanding::NotApplicable,
            calibration: CalibrationDeclaration::PointOnly,
        }
    }

    /// A standing the producer did not state: supported coordinates are not
    /// assumed to hold trust, and an uncertainty statement is wanted without a
    /// record. A node built with it is withdrawn for provider trust.
    #[must_use]
    pub const fn undeclared() -> Self {
        Self {
            support: SupportStanding::Supported,
            trust: TrustStanding::Undeclared,
            calibration: CalibrationDeclaration::Required,
        }
    }

    /// A node that rests on a provider at the given verification level.
    #[must_use]
    pub const fn provider(
        trust: crate::ProviderTrust,
        calibration: CalibrationDeclaration,
    ) -> Self {
        Self {
            support: SupportStanding::Supported,
            trust: TrustStanding::of_provider(trust),
            calibration,
        }
    }

    /// Replace the calibration declaration.
    #[must_use]
    pub fn with_calibration(mut self, calibration: CalibrationDeclaration) -> Self {
        self.calibration = calibration;
        self
    }

    /// Declare coordinates outside the support.
    #[must_use]
    pub fn with_unsupported(mut self, coordinates: Vec<String>) -> Self {
        self.support = SupportStanding::Unsupported { coordinates };
        self
    }
}

/// One claim or object.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompositionNode {
    /// Unique stable key within the graph.
    pub key: String,
    /// What it stands for.
    pub kind: NodeKind,
    /// Content identity: changes whenever the data, fit, numbers or
    /// declarations it was computed from change.
    pub identity: SemanticDigest,
    /// Identity of the structural derivation alone (graph, identification,
    /// bridge form), when it has one. Unchanged under a data replacement.
    pub structure: Option<SemanticDigest>,
    /// Model or representation version label, when it has one.
    pub representation: Option<String>,
    /// Coordinate the claim is evaluated at (target population, regime,
    /// quantity). A calibration record holds at one coordinate only.
    pub coordinate: SemanticDigest,
    /// Whether its own coordinates are supported.
    pub support: SupportStanding,
    /// Whether its provider trust holds, for provider nodes.
    pub trust: TrustStanding,
    /// Its uncertainty guarantee.
    pub calibration: CalibrationDeclaration,
    /// Assumptions this node introduces at its boundary.
    pub assumptions: Vec<String>,
    /// Name of the transformation this node applies, when it applies one.
    pub transformation: Option<String>,
    /// Ledger sources whose observations this node counts.
    pub evidence: Vec<SourceId>,
}

impl CompositionNode {
    /// A supported point-only node with no structure, version or evidence.
    #[must_use]
    pub fn new(key: impl Into<String>, kind: NodeKind, identity: SemanticDigest) -> Self {
        Self {
            key: key.into(),
            kind,
            identity,
            structure: None,
            representation: None,
            coordinate: SemanticDigest::from_bytes([0; 32]),
            support: SupportStanding::Supported,
            trust: TrustStanding::NotApplicable,
            calibration: CalibrationDeclaration::PointOnly,
            assumptions: Vec::new(),
            transformation: None,
            evidence: Vec::new(),
        }
    }

    /// Set the structural derivation identity.
    #[must_use]
    pub const fn with_structure(mut self, structure: SemanticDigest) -> Self {
        self.structure = Some(structure);
        self
    }

    /// Set the model or representation version.
    #[must_use]
    pub fn with_representation(mut self, representation: impl Into<String>) -> Self {
        self.representation = Some(representation.into());
        self
    }

    /// Set the evaluation coordinate.
    #[must_use]
    pub const fn with_coordinate(mut self, coordinate: SemanticDigest) -> Self {
        self.coordinate = coordinate;
        self
    }

    /// Declare coordinates outside the support.
    #[must_use]
    pub fn with_unsupported(mut self, coordinates: Vec<String>) -> Self {
        self.support = SupportStanding::Unsupported { coordinates };
        self
    }

    /// Set the support, trust and calibration standings together.
    #[must_use]
    pub fn with_standing(mut self, standing: ClaimStanding) -> Self {
        self.support = standing.support;
        self.trust = standing.trust;
        self.calibration = standing.calibration;
        self
    }

    /// The node's current standings.
    #[must_use]
    pub fn standing(&self) -> ClaimStanding {
        ClaimStanding {
            support: self.support.clone(),
            trust: self.trust,
            calibration: self.calibration.clone(),
        }
    }

    /// Set the provider trust standing.
    #[must_use]
    pub const fn with_trust(mut self, trust: TrustStanding) -> Self {
        self.trust = trust;
        self
    }

    /// Set the calibration declaration.
    #[must_use]
    pub fn with_calibration(mut self, calibration: CalibrationDeclaration) -> Self {
        self.calibration = calibration;
        self
    }

    /// Declare the assumptions introduced at this boundary.
    #[must_use]
    pub fn with_assumptions(mut self, assumptions: Vec<String>) -> Self {
        self.assumptions = assumptions;
        self
    }

    /// Name the transformation applied.
    #[must_use]
    pub fn with_transformation(mut self, name: impl Into<String>) -> Self {
        self.transformation = Some(name.into());
        self
    }

    /// Declare the ledger sources counted.
    #[must_use]
    pub fn with_evidence(mut self, evidence: Vec<SourceId>) -> Self {
        self.evidence = evidence;
        self
    }
}

/// One dependency: `downstream` depends on `upstream` in the stated way.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CompositionEdge {
    /// The dependent node.
    pub downstream: String,
    /// The node depended on.
    pub upstream: String,
    /// How it depends.
    pub kind: DependencyKind,
}

/// Why a graph declaration is refused.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum GraphError {
    /// A blank, over-long or otherwise malformed text field.
    InvalidText(&'static str),
    /// A node key declared twice.
    DuplicateNode(String),
    /// A node key that is not in the graph.
    UnknownNode(String),
    /// A node that depends on itself.
    SelfDependency(String),
    /// An edge that would close a cycle.
    Cycle {
        /// The dependent node.
        downstream: String,
        /// The node depended on.
        upstream: String,
    },
    /// A bound was exceeded.
    LimitExceeded(&'static str),
    /// An unsupported node declares no coordinates.
    EmptyUnsupported(String),
    /// A node inherits calibration from a node it has no calibration edge to.
    InheritedWithoutEdge {
        /// The inheriting node.
        node: String,
        /// The node named as the source.
        from: String,
    },
}

impl GraphError {
    /// The registered reason code every graph declaration error is surfaced
    /// under; the namespaced detail is the start of its display text.
    #[must_use]
    pub const fn reason_code(&self) -> &'static str {
        antecedent_core::reason_code!("invalid_argument")
    }
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidText(what) => {
                write!(f, "composition_verification.invalid_text: {what} is blank or too long")
            }
            Self::DuplicateNode(key) => {
                write!(f, "composition_verification.duplicate_node: {key} is declared twice")
            }
            Self::UnknownNode(key) => {
                write!(f, "composition_verification.unknown_node: {key} is not in the graph")
            }
            Self::SelfDependency(key) => {
                write!(f, "composition_verification.self_dependency: {key} depends on itself")
            }
            Self::Cycle { downstream, upstream } => write!(
                f,
                "composition_verification.cycle: {downstream} depending on {upstream} closes a \
                 cycle"
            ),
            Self::LimitExceeded(what) => {
                write!(f, "composition_verification.limit_exceeded: {what}")
            }
            Self::EmptyUnsupported(key) => write!(
                f,
                "composition_verification.empty_unsupported: {key} is unsupported but names no \
                 coordinate"
            ),
            Self::InheritedWithoutEdge { node, from } => write!(
                f,
                "composition_verification.inherited_without_edge: {node} inherits calibration \
                 from {from} without a calibration dependency on it"
            ),
        }
    }
}

impl std::error::Error for GraphError {}

fn text(value: &str, what: &'static str) -> Result<(), GraphError> {
    if value.trim().is_empty() || value.len() > MAX_TEXT_BYTES {
        Err(GraphError::InvalidText(what))
    } else {
        Ok(())
    }
}

fn digest(writer: &CanonicalWriter) -> SemanticDigest {
    SemanticDigest::from_bytes(writer.finish())
}

/// A set of claims and objects with typed dependencies.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CompositionGraph {
    nodes: BTreeMap<String, CompositionNode>,
    edges: BTreeSet<CompositionEdge>,
}

impl CompositionGraph {
    /// An empty graph.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a node. Assumptions, unsupported coordinates and evidence are
    /// stored sorted and distinct, so declaration order carries no meaning.
    ///
    /// # Errors
    /// A malformed or duplicate node, or a bound exceeded.
    pub fn add_node(&mut self, mut node: CompositionNode) -> Result<(), GraphError> {
        text(&node.key, "node key")?;
        if let Some(version) = &node.representation {
            text(version, "representation version")?;
        }
        if let Some(name) = &node.transformation {
            text(name, "transformation name")?;
        }
        for assumption in &node.assumptions {
            text(assumption, "assumption")?;
        }
        if let SupportStanding::Unsupported { coordinates } = &mut node.support {
            for coordinate in coordinates.iter() {
                text(coordinate, "unsupported coordinate")?;
            }
            coordinates.sort_unstable();
            coordinates.dedup();
            if coordinates.is_empty() {
                return Err(GraphError::EmptyUnsupported(node.key));
            }
            if coordinates.len() > MAX_ITEMS_PER_NODE {
                return Err(GraphError::LimitExceeded("unsupported coordinates"));
            }
        }
        if let CalibrationDeclaration::Inherited { from, .. } = &node.calibration {
            text(from, "calibration source")?;
        }
        node.assumptions.sort_unstable();
        node.assumptions.dedup();
        node.evidence.sort_unstable();
        node.evidence.dedup();
        if node.assumptions.len() > MAX_ITEMS_PER_NODE || node.evidence.len() > MAX_ITEMS_PER_NODE {
            return Err(GraphError::LimitExceeded("items on one node"));
        }
        if self.nodes.contains_key(&node.key) {
            return Err(GraphError::DuplicateNode(node.key));
        }
        if self.nodes.len() >= MAX_NODES {
            return Err(GraphError::LimitExceeded("nodes"));
        }
        self.nodes.insert(node.key.clone(), node);
        Ok(())
    }

    /// Add a dependency of `downstream` on `upstream`. Adding the same edge
    /// twice is a no-op.
    ///
    /// # Errors
    /// An unknown node, a self-dependency, a cycle or a bound exceeded.
    pub fn add_edge(
        &mut self,
        downstream: &str,
        upstream: &str,
        kind: DependencyKind,
    ) -> Result<(), GraphError> {
        for key in [downstream, upstream] {
            if !self.nodes.contains_key(key) {
                return Err(GraphError::UnknownNode(key.to_owned()));
            }
        }
        if downstream == upstream {
            return Err(GraphError::SelfDependency(downstream.to_owned()));
        }
        if self.ancestors_of(upstream).contains(downstream) {
            return Err(GraphError::Cycle {
                downstream: downstream.to_owned(),
                upstream: upstream.to_owned(),
            });
        }
        let edge = CompositionEdge {
            downstream: downstream.to_owned(),
            upstream: upstream.to_owned(),
            kind,
        };
        if !self.edges.contains(&edge) && self.edges.len() >= MAX_EDGES {
            return Err(GraphError::LimitExceeded("edges"));
        }
        self.edges.insert(edge);
        Ok(())
    }

    /// The node under `key`.
    #[must_use]
    pub fn node(&self, key: &str) -> Option<&CompositionNode> {
        self.nodes.get(key)
    }

    /// Nodes in key order.
    pub fn nodes(&self) -> impl Iterator<Item = &CompositionNode> {
        self.nodes.values()
    }

    /// Edges in canonical order.
    pub fn edges(&self) -> impl Iterator<Item = &CompositionEdge> {
        self.edges.iter()
    }

    /// Number of nodes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the graph has no node.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// The dependencies of `key`: `(upstream, kind)` in canonical order.
    #[must_use]
    pub fn dependencies(&self, key: &str) -> Vec<(&str, DependencyKind)> {
        self.edges
            .iter()
            .filter(|edge| edge.downstream == key)
            .map(|edge| (edge.upstream.as_str(), edge.kind))
            .collect()
    }

    /// Everything `key` depends on, directly or not (excluding `key`).
    #[must_use]
    pub fn ancestors_of(&self, key: &str) -> BTreeSet<&str> {
        let mut seen: BTreeSet<&str> = BTreeSet::new();
        let mut stack = vec![key];
        while let Some(current) = stack.pop() {
            for edge in self.edges.iter().filter(|edge| edge.downstream == current) {
                if seen.insert(edge.upstream.as_str()) {
                    stack.push(edge.upstream.as_str());
                }
            }
        }
        seen
    }

    /// Node keys with every dependency before its dependents. Ties break by
    /// key, so the order is a function of the content only.
    #[must_use]
    pub fn topological_order(&self) -> Vec<&str> {
        let mut remaining: BTreeMap<&str, usize> = self
            .nodes
            .keys()
            .map(|key| {
                let count = self.edges.iter().filter(|e| e.downstream == *key).count();
                (key.as_str(), count)
            })
            .collect();
        let mut ready: BTreeSet<&str> =
            remaining.iter().filter(|(_, n)| **n == 0).map(|(k, _)| *k).collect();
        let mut order = Vec::with_capacity(self.nodes.len());
        while let Some(next) = ready.pop_first() {
            order.push(next);
            for edge in self.edges.iter().filter(|e| e.upstream == next) {
                if let Some(count) = remaining.get_mut(edge.downstream.as_str()) {
                    *count -= 1;
                    if *count == 0 {
                        ready.insert(edge.downstream.as_str());
                    }
                }
            }
        }
        order
    }

    /// Check what edge-by-edge insertion cannot: a node that claims inherited
    /// calibration must exist upstream and carry a calibration dependency.
    ///
    /// # Errors
    /// [`GraphError::InheritedWithoutEdge`] or [`GraphError::UnknownNode`].
    pub fn validate(&self) -> Result<(), GraphError> {
        for node in self.nodes.values() {
            if let CalibrationDeclaration::Inherited { from, .. } = &node.calibration {
                if !self.nodes.contains_key(from) {
                    return Err(GraphError::UnknownNode(from.clone()));
                }
                let edge = CompositionEdge {
                    downstream: node.key.clone(),
                    upstream: from.clone(),
                    kind: DependencyKind::Calibration,
                };
                if !self.edges.contains(&edge) {
                    return Err(GraphError::InheritedWithoutEdge {
                        node: node.key.clone(),
                        from: from.clone(),
                    });
                }
            }
        }
        Ok(())
    }

    /// Canonical identity of the whole graph.
    #[must_use]
    pub fn identity(&self) -> SemanticDigest {
        let mut writer = CanonicalWriter::new("antecedent.composition_verification.graph");
        writer.word(COMPOSITION_GRAPH_VERSION).word(self.nodes.len() as u64);
        for node in self.nodes.values() {
            encode_node(&mut writer, node);
        }
        writer.word(self.edges.len() as u64);
        for edge in &self.edges {
            writer.text(&edge.downstream).text(&edge.upstream).text(edge.kind.as_str());
        }
        digest(&writer)
    }
}

pub(crate) fn encode_calibration(writer: &mut CanonicalWriter, c: &CalibrationDeclaration) {
    match c {
        CalibrationDeclaration::PointOnly => {
            writer.text("point_only");
        }
        CalibrationDeclaration::Required => {
            writer.text("required");
        }
        CalibrationDeclaration::Held { record, coordinate } => {
            writer.text("held").bytes(record.as_bytes()).bytes(coordinate.as_bytes());
        }
        CalibrationDeclaration::Inherited { from, record, coordinate } => {
            writer
                .text("inherited")
                .text(from)
                .bytes(record.as_bytes())
                .bytes(coordinate.as_bytes());
        }
    }
}

fn encode_node(writer: &mut CanonicalWriter, node: &CompositionNode) {
    writer.text(&node.key).text(node.kind.as_str()).bytes(node.identity.as_bytes());
    match &node.structure {
        Some(structure) => writer.word(1).bytes(structure.as_bytes()),
        None => writer.word(0),
    };
    writer.opt_text(node.representation.as_deref()).bytes(node.coordinate.as_bytes());
    match &node.support {
        SupportStanding::Supported => writer.text("supported"),
        SupportStanding::Unsupported { coordinates } => {
            writer.text("unsupported").texts(coordinates)
        }
    };
    writer.text(node.trust.as_str());
    encode_calibration(writer, &node.calibration);
    writer.texts(&node.assumptions).opt_text(node.transformation.as_deref());
    writer.word(node.evidence.len() as u64);
    for source in &node.evidence {
        writer.bytes(source.digest().as_bytes());
    }
}
