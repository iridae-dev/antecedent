//! The per-claim lineage report.
//!
//! [`CompositionView`] answers, for one claim: which upstream claims and
//! evidence stand behind it, through which transformations and bridges, which
//! of its inputs share data, at which model and representation versions,
//! which coordinates are unsupported, which assumptions each boundary
//! introduced, and what happened to calibration (inherited, lost, newly
//! required). It reads a [`GraphRevalidation`]; it computes no statistic.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;
use std::fmt;

use antecedent_core::{CancellationToken, CanonicalWriter, SemanticDigest};

use crate::evidence_ledger::{
    EvidenceLedger, LedgerError, Overlap, RelationKind, Revalidation, SourceId,
};

use super::graph::{CompositionGraph, DependencyKind, GraphError, NodeKind, SupportStanding};
use super::revalidate::{CalibrationLineage, GraphRevalidation};

/// Default bound on source-pair overlap checks that a caller of untrusted input may apply by
/// cancelling the query; [`shared_data_relations`] itself checks the cancellation token between
/// source pairs and imposes no fixed bound.
pub const MAX_OVERLAP_CHECKS: usize = 250_000;

/// Why a view or a shared-data query could not be produced.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ViewError {
    /// The graph does not contain the claim or does not validate.
    Graph(GraphError),
    /// The ledger refused an overlap query (an unknown source).
    Ledger(LedgerError),
    /// A caller-imposed overlap work bound was exceeded. The query itself imposes none.
    LimitExceeded,
    /// The cancellation token fired.
    Cancelled,
}

impl fmt::Display for ViewError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Graph(error) => write!(f, "{error}"),
            Self::Ledger(error) => write!(f, "{error}"),
            Self::LimitExceeded => f.write_str(
                "composition_verification.limit_exceeded: the shared-data query exceeds its \
                 overlap-check bound",
            ),
            Self::Cancelled => f.write_str("composition_verification.cancelled: cancelled"),
        }
    }
}

impl std::error::Error for ViewError {}

impl From<GraphError> for ViewError {
    fn from(error: GraphError) -> Self {
        Self::Graph(error)
    }
}

/// Two nodes whose evidence overlaps (or may overlap).
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct SharedDataRelation {
    /// The earlier node key (keys are ordered).
    pub first: String,
    /// The later node key.
    pub second: String,
    /// `shared_data`, `shared_prior`, `shared_fitted_model` or
    /// `unknown_dependence`; independent pairs are not listed.
    pub relation: &'static str,
    /// Observations shared, summed over the source pairs that share any.
    /// Zero for an unknown dependence.
    pub shared_observations: u64,
}

const fn rank(kind: RelationKind) -> u8 {
    match kind {
        RelationKind::SharedData => 0,
        RelationKind::SharedFittedModel => 1,
        RelationKind::SharedPrior => 2,
        RelationKind::UnknownDependence => 3,
        _ => 4,
    }
}

/// Shared-data relationships between the nodes of `graph`, from the ledger's
/// row-level overlap. Independence is never assumed: a pair whose sources are
/// not both known is listed as an unknown dependence.
///
/// # Errors
/// A source the ledger does not hold, or cancellation.
pub fn shared_data_relations(
    graph: &CompositionGraph,
    ledger: &EvidenceLedger,
    cancel: &CancellationToken,
) -> Result<Vec<SharedDataRelation>, ViewError> {
    let carriers: Vec<(&str, &[SourceId])> = graph
        .nodes()
        .filter(|n| !n.evidence.is_empty())
        .map(|n| (n.key.as_str(), n.evidence.as_slice()))
        .collect();
    let mut out = Vec::new();
    for (i, (first, first_sources)) in carriers.iter().enumerate() {
        for (second, second_sources) in &carriers[i + 1..] {
            if cancel.is_cancelled() {
                return Err(ViewError::Cancelled);
            }
            let mut best: Option<RelationKind> = None;
            let mut shared = 0_u64;
            for a in *first_sources {
                if cancel.is_cancelled() {
                    return Err(ViewError::Cancelled);
                }
                for b in *second_sources {
                    let relation = ledger.relation(*a, *b).map_err(ViewError::Ledger)?;
                    if relation == RelationKind::Independent {
                        continue;
                    }
                    if let Overlap::SharedObservations { shared: n } =
                        ledger.overlap(*a, *b).map_err(ViewError::Ledger)?
                    {
                        shared = shared.saturating_add(n);
                    }
                    if best.is_none_or(|current| rank(relation) < rank(current)) {
                        best = Some(relation);
                    }
                }
            }
            if let Some(kind) = best {
                out.push(SharedDataRelation {
                    first: (*first).to_owned(),
                    second: (*second).to_owned(),
                    relation: kind.as_str(),
                    shared_observations: shared,
                });
            }
        }
    }
    out.sort();
    Ok(out)
}

/// One upstream object in a claim's lineage.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LineageEntry {
    /// Node key.
    pub key: String,
    /// What it is.
    pub kind: NodeKind,
    /// Content identity.
    pub identity: SemanticDigest,
    /// Model or representation version.
    pub representation: Option<String>,
    /// Classification against the earlier snapshot.
    pub class: Revalidation,
    /// Its calibration state.
    pub calibration: CalibrationLineage,
}

/// One dependency inside the lineage.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct LineageEdge {
    /// The dependent node.
    pub downstream: String,
    /// The node depended on.
    pub upstream: String,
    /// How it depends.
    pub kind: DependencyKind,
}

/// A transformation or bridge on the way to the claim.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BoundaryTransformation {
    /// The node that applies it.
    pub node: String,
    /// What the node is.
    pub kind: NodeKind,
    /// The declared transformation name (the kind's name when none).
    pub name: String,
}

/// An assumption introduced at one boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BoundaryAssumption {
    /// The node that introduces it.
    pub node: String,
    /// The statement.
    pub statement: String,
}

/// A coordinate outside the support of some node in the lineage.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UnsupportedCoordinate {
    /// The node whose support it lies outside of.
    pub node: String,
    /// The coordinate.
    pub coordinate: String,
}

/// The lineage report of one claim.
#[derive(Clone, Debug, PartialEq)]
pub struct CompositionView {
    /// The claim.
    pub claim: LineageEntry,
    /// Upstream objects, every dependency before its dependents.
    pub upstream: Vec<LineageEntry>,
    /// Dependencies among the claim and its upstream objects.
    pub edges: Vec<LineageEdge>,
    /// Ledger sources counted anywhere in the lineage (sorted, distinct).
    pub evidence: Vec<SourceId>,
    /// Transformations and bridges, in dependency order.
    pub transformations: Vec<BoundaryTransformation>,
    /// Shared-data relationships among nodes of the lineage.
    pub shared_data: Vec<SharedDataRelation>,
    /// Coordinates outside the support of any node in the lineage.
    pub unsupported: Vec<UnsupportedCoordinate>,
    /// Assumptions each boundary introduced, in dependency order.
    pub assumptions: Vec<BoundaryAssumption>,
}

impl CompositionView {
    /// The lineage of `key`. `shared` is the relation list from
    /// [`shared_data_relations`] (empty when no ledger is available, in which
    /// case the view reports no shared-data relationship and says nothing
    /// about independence).
    ///
    /// # Errors
    /// A key that is not in the graph.
    pub fn of(
        graph: &CompositionGraph,
        revalidation: &GraphRevalidation,
        key: &str,
        shared: &[SharedDataRelation],
    ) -> Result<Self, ViewError> {
        let unknown = || ViewError::Graph(GraphError::UnknownNode(key.to_owned()));
        let focus = graph.node(key).ok_or_else(unknown)?;
        let mut members: BTreeSet<&str> = graph.ancestors_of(key);
        members.insert(key);
        let entry = |k: &str| -> Result<LineageEntry, ViewError> {
            let node = graph.node(k).ok_or_else(unknown)?;
            let claim = revalidation.claim(k).ok_or_else(unknown)?;
            Ok(LineageEntry {
                key: node.key.clone(),
                kind: node.kind,
                identity: node.identity,
                representation: node.representation.clone(),
                class: claim.class,
                calibration: claim.calibration.clone(),
            })
        };
        let mut upstream = Vec::new();
        let mut transformations = Vec::new();
        let mut assumptions = Vec::new();
        let mut unsupported = Vec::new();
        let mut evidence: BTreeSet<SourceId> = BTreeSet::new();
        for k in graph.topological_order() {
            if !members.contains(k) {
                continue;
            }
            let node = graph.node(k).ok_or_else(unknown)?;
            if k != key {
                upstream.push(entry(k)?);
            }
            if node.kind.is_boundary() || node.transformation.is_some() {
                transformations.push(BoundaryTransformation {
                    node: node.key.clone(),
                    kind: node.kind,
                    name: node
                        .transformation
                        .clone()
                        .unwrap_or_else(|| node.kind.as_str().to_owned()),
                });
            }
            assumptions.extend(node.assumptions.iter().map(|statement| BoundaryAssumption {
                node: node.key.clone(),
                statement: statement.clone(),
            }));
            if let SupportStanding::Unsupported { coordinates } = &node.support {
                unsupported.extend(coordinates.iter().map(|coordinate| UnsupportedCoordinate {
                    node: node.key.clone(),
                    coordinate: coordinate.clone(),
                }));
            }
            evidence.extend(node.evidence.iter().copied());
        }
        let mut edges: Vec<LineageEdge> = graph
            .edges()
            .filter(|e| members.contains(e.downstream.as_str()))
            .map(|e| LineageEdge {
                downstream: e.downstream.clone(),
                upstream: e.upstream.clone(),
                kind: e.kind,
            })
            .collect();
        edges.sort();
        let mut shared_data: Vec<SharedDataRelation> = shared
            .iter()
            .filter(|r| members.contains(r.first.as_str()) && members.contains(r.second.as_str()))
            .cloned()
            .collect();
        shared_data.sort();
        Ok(Self {
            claim: entry(&focus.key)?,
            upstream,
            edges,
            evidence: evidence.into_iter().collect(),
            transformations,
            shared_data,
            unsupported,
            assumptions,
        })
    }

    /// Model and representation versions in the lineage (the claim included),
    /// as `(node, version)` in dependency order.
    #[must_use]
    pub fn versions(&self) -> Vec<(&str, &str)> {
        self.upstream
            .iter()
            .chain(std::iter::once(&self.claim))
            .filter_map(|e| e.representation.as_deref().map(|v| (e.key.as_str(), v)))
            .collect()
    }

    /// Identity of the whole report.
    #[must_use]
    pub fn identity(&self) -> SemanticDigest {
        let mut w = CanonicalWriter::new("antecedent.composition_verification.view");
        let entries: Vec<&LineageEntry> =
            self.upstream.iter().chain(std::iter::once(&self.claim)).collect();
        w.word(entries.len() as u64);
        for e in entries {
            w.text(&e.key)
                .text(e.kind.as_str())
                .bytes(e.identity.as_bytes())
                .opt_text(e.representation.as_deref())
                .text(e.class.as_str())
                .text(e.calibration.state.as_str())
                .opt_text(e.calibration.from.as_deref())
                .word(u64::from(e.calibration.newly_required))
                .opt_text(e.calibration.detail);
        }
        w.word(self.edges.len() as u64);
        for e in &self.edges {
            w.text(&e.downstream).text(&e.upstream).text(e.kind.as_str());
        }
        w.word(self.evidence.len() as u64);
        for s in &self.evidence {
            w.bytes(s.digest().as_bytes());
        }
        w.word(self.transformations.len() as u64);
        for t in &self.transformations {
            w.text(&t.node).text(t.kind.as_str()).text(&t.name);
        }
        w.word(self.shared_data.len() as u64);
        for r in &self.shared_data {
            w.text(&r.first).text(&r.second).text(r.relation).word(r.shared_observations);
        }
        w.word(self.unsupported.len() as u64);
        for u in &self.unsupported {
            w.text(&u.node).text(&u.coordinate);
        }
        w.word(self.assumptions.len() as u64);
        for a in &self.assumptions {
            w.text(&a.node).text(&a.statement);
        }
        SemanticDigest::from_bytes(w.finish())
    }
}
