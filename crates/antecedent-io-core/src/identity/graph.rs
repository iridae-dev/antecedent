//! Canonical graph identity wires and conversions.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    AdjacencyMask, Admg, AdmgWire, Cpdag, CpdagWire, Dag, DagWire, Deserialize,
    DiscoveryTemporalGraphWire, EndpointWire, GraphPosterior, GraphPosteriorAtomKind,
    IDENTITY_FORMAT, IoError, MarkedEdge, NodeRef, Pag, PagWire, Serialize, TemporalAdmg,
    TemporalCpdag, TemporalDag, TemporalGraphWire, TemporalNodeKeyWire, TemporalPag, VariableId,
    admg_from_adjacency_mask, admg_to_wire, cpdag_from_adjacency_mask, cpdag_to_wire,
    dag_from_adjacency_mask, dag_to_wire, endpoint_to_wire, pag_from_adjacency_mask, pag_to_wire,
    temporal_cpdag_from_dbn_atoms, temporal_dag_from_dbn_atoms, temporal_dag_to_wire,
    temporal_pag_from_dbn_atoms, to_cbor,
};

/// Temporal ADMG identity: lagged nodes plus directed and bidirected edges.
///
/// Node order is `(variable, lag)`. Edge endpoints are indexes into that order.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct TemporalAdmgIdentityWire {
    /// Lagged nodes.
    pub nodes: Vec<TemporalNodeKeyWire>,
    /// Directed edges `(from, to)`.
    pub directed: Vec<(u32, u32)>,
    /// Bidirected edges `(a, b)` with `a < b`.
    pub bidirected: Vec<(u32, u32)>,
}

/// Graph payload for identification identity. Uses existing wire types.
///
/// Edge lists are in canonical (sorted) order, so a graph's identity never
/// depends on insertion order.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum GraphIdentityWire {
    /// Static DAG.
    Dag(DagWire),
    /// Static ADMG.
    Admg(AdmgWire),
    /// Static CPDAG.
    Cpdag(CpdagWire),
    /// Static PAG.
    Pag(PagWire),
    /// Temporal DAG (nodes sorted by `(variable, lag)`).
    TemporalDag(TemporalGraphWire),
    /// Temporal ADMG (nodes sorted by `(variable, lag)`; edges re-indexed).
    TemporalAdmg(TemporalAdmgIdentityWire),
    /// Temporal class with lagged/context nodes and marked edges.
    TemporalClass(TemporalClassIdentityWire),
    /// No causal graph was supplied; design metadata defines the experiment.
    RandomizedTrial,
    /// Graph-posterior atoms: every atom's structure and weight.
    GraphPosterior {
        /// Declared graph class of the posterior atoms.
        graph_class: String,
        /// Number of retained atoms.
        n_atoms: u64,
        /// Atom graphs and weights, in posterior order.
        atoms: Vec<PosteriorAtomIdentityWire>,
    },
}

impl GraphIdentityWire {
    /// Scientific projection hashed into identification premises.
    ///
    /// Posterior atoms are a weighted set: atoms are sorted canonically and
    /// the position-derived [`PosteriorAtomIdentityWire::execution_key`] is
    /// cleared, so reordering atoms or renumbering cache keys preserves
    /// identity while any graph or weight change moves it.
    #[must_use]
    pub fn premises(&self) -> Self {
        match self {
            Self::GraphPosterior { graph_class, n_atoms, atoms } => {
                let mut keyed: Vec<(Vec<u8>, PosteriorAtomIdentityWire)> = atoms
                    .iter()
                    .map(|atom| {
                        let atom = PosteriorAtomIdentityWire { execution_key: 0, ..atom.clone() };
                        (to_cbor(&atom).unwrap_or_default(), atom)
                    })
                    .collect();
                keyed.sort_by(|left, right| left.0.cmp(&right.0));
                Self::GraphPosterior {
                    graph_class: graph_class.clone(),
                    n_atoms: *n_atoms,
                    atoms: keyed.into_iter().map(|(_, atom)| atom).collect(),
                }
            }
            other => other.clone(),
        }
    }
}

/// Temporal CPDAG / PAG identity: lagged edges plus contemporaneous marks.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct TemporalClassIdentityWire {
    /// `temporal_cpdag.v2` or `temporal_pag.v2`.
    pub kind: String,
    /// Nodes in dense order.
    pub nodes: Vec<IdentityNodeWire>,
    /// Marked edges, including middle marks.
    pub edges: Vec<IdentityMarkedEdgeWire>,
}

/// Node identity that includes lagged and contemporaneous bindings.
///
/// The derived order is the canonical node order of a temporal class identity.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum IdentityNodeWire {
    /// Static variable.
    Static {
        /// Variable raw id.
        variable: u32,
    },
    /// Lagged temporal node.
    Lagged {
        /// Variable raw id.
        variable: u32,
        /// Lag magnitude (`0` = contemporaneous).
        lag: u32,
    },
    /// Slot of a finite unfolding (`variable` at signed `offset` from the analysis origin).
    Unfolded {
        /// Variable raw id.
        variable: u32,
        /// Signed time offset (negative = history).
        offset: i32,
    },
    /// Context / environment node.
    Context {
        /// Variable raw id.
        variable: u32,
        /// Optional environment id.
        environment: Option<u32>,
    },
}

impl IdentityNodeWire {
    fn from_node(node: NodeRef) -> Self {
        match node {
            NodeRef::Static(variable) => Self::Static { variable: variable.raw() },
            NodeRef::Lagged { variable, lag } => {
                Self::Lagged { variable: variable.raw(), lag: lag.raw() }
            }
            NodeRef::Unfolded { variable, offset } => {
                Self::Unfolded { variable: variable.raw(), offset }
            }
            NodeRef::Context { variable, environment } => Self::Context {
                variable: variable.raw(),
                environment: environment.map(antecedent_core::EnvironmentId::raw),
            },
        }
    }
}

/// Marked edge with middle mark (LPCMCI).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct IdentityMarkedEdgeWire {
    /// Endpoint A.
    pub a: u32,
    /// Endpoint B.
    pub b: u32,
    /// Mark at A.
    pub at_a: EndpointWire,
    /// Mark at B.
    pub at_b: EndpointWire,
    /// Middle mark (`unknown`, `left`, `right`, `both`, `empty`).
    pub middle: String,
}

fn middle_wire(mark: antecedent_graph::MiddleMark) -> &'static str {
    match mark {
        antecedent_graph::MiddleMark::Unknown => "unknown",
        antecedent_graph::MiddleMark::Left => "left",
        antecedent_graph::MiddleMark::Right => "right",
        antecedent_graph::MiddleMark::Both => "both",
        antecedent_graph::MiddleMark::Empty => "empty",
    }
}

fn marked_identity_edge(edge: MarkedEdge) -> IdentityMarkedEdgeWire {
    IdentityMarkedEdgeWire {
        a: edge.a.raw(),
        b: edge.b.raw(),
        at_a: endpoint_to_wire(edge.at_a),
        at_b: endpoint_to_wire(edge.at_b),
        middle: middle_wire(edge.middle).into(),
    }
}

/// DAG wire with edges in canonical `(from, to)` order.
///
/// # Errors
///
/// Non-directed edges.
pub fn canonical_dag_wire(dag: &Dag) -> Result<DagWire, IoError> {
    let mut wire = dag_to_wire(dag)?;
    wire.edges.sort_unstable();
    Ok(wire)
}

/// Temporal-DAG wire with nodes sorted by `(variable, lag)` and edges
/// re-indexed and sorted, so equal graphs built in any order encode equally.
///
/// # Errors
///
/// Non-lagged nodes.
pub fn canonical_temporal_dag_wire(graph: &TemporalDag) -> Result<TemporalGraphWire, IoError> {
    let wire = temporal_dag_to_wire(graph)?;
    let mut order: Vec<usize> = (0..wire.nodes.len()).collect();
    order.sort_by_key(|&index| (wire.nodes[index].variable, wire.nodes[index].lag));
    let mut position = vec![0u32; wire.nodes.len()];
    for (new, &old) in order.iter().enumerate() {
        position[old] = u32::try_from(new).map_err(|_| IoError::TooLarge)?;
    }
    let remap = |index: u32| {
        position.get(index as usize).copied().ok_or_else(|| {
            IoError::Convert("temporal edge references a node outside the graph".into())
        })
    };
    let mut directed = wire
        .directed
        .iter()
        .map(|&(from, to)| Ok((remap(from)?, remap(to)?)))
        .collect::<Result<Vec<_>, IoError>>()?;
    directed.sort_unstable();
    Ok(DiscoveryTemporalGraphWire {
        kind: wire.kind,
        nodes: order.iter().map(|&index| wire.nodes[index]).collect(),
        directed,
    })
}

/// Encode a DAG for identification identity.
///
/// # Errors
///
/// Non-directed edges.
pub fn dag_identity(dag: &Dag) -> Result<GraphIdentityWire, IoError> {
    Ok(GraphIdentityWire::Dag(canonical_dag_wire(dag)?))
}

/// Encode an ADMG for identification identity (sorted edge lists).
///
/// # Errors
///
/// Capacity overflow.
pub fn admg_identity(admg: &Admg) -> Result<GraphIdentityWire, IoError> {
    Ok(GraphIdentityWire::Admg(canonical_admg_wire(admg)?))
}

pub(super) fn canonical_admg_wire(admg: &Admg) -> Result<AdmgWire, IoError> {
    let mut wire = admg_to_wire(admg)?;
    wire.directed.sort_unstable();
    for pair in &mut wire.bidirected {
        if pair.0 > pair.1 {
            *pair = (pair.1, pair.0);
        }
    }
    wire.bidirected.sort_unstable();
    wire.bidirected.dedup();
    Ok(wire)
}

/// Encode a CPDAG for identification identity (sorted edge lists).
///
/// # Errors
///
/// Unexpected marks.
pub fn cpdag_identity(cpdag: &Cpdag) -> Result<GraphIdentityWire, IoError> {
    let mut wire = cpdag_to_wire(cpdag)?;
    wire.directed.sort_unstable();
    wire.undirected.sort_unstable();
    Ok(GraphIdentityWire::Cpdag(wire))
}

/// Encode a PAG for identification identity (edges sorted by endpoints).
///
/// # Errors
///
/// Non-static nodes.
pub fn pag_identity(pag: &Pag) -> Result<GraphIdentityWire, IoError> {
    let mut wire = pag_to_wire(pag)?;
    wire.edges.sort_by_key(|edge| (edge.a, edge.b));
    Ok(GraphIdentityWire::Pag(wire))
}

/// Encode a temporal DAG for identification identity.
///
/// # Errors
///
/// Non-lagged nodes.
pub fn temporal_dag_identity(graph: &TemporalDag) -> Result<GraphIdentityWire, IoError> {
    Ok(GraphIdentityWire::TemporalDag(canonical_temporal_dag_wire(graph)?))
}

/// Encode a temporal ADMG for identification identity.
///
/// # Errors
///
/// Non-lagged nodes or an edge index outside the node list.
pub fn temporal_admg_identity(graph: &TemporalAdmg) -> Result<GraphIdentityWire, IoError> {
    Ok(GraphIdentityWire::TemporalAdmg(canonical_temporal_admg_wire(graph)?))
}

fn canonical_temporal_admg_wire(graph: &TemporalAdmg) -> Result<TemporalAdmgIdentityWire, IoError> {
    use antecedent_core::NodeRef;
    let mut nodes = Vec::with_capacity(graph.node_count());
    for node in graph.nodes() {
        match node {
            NodeRef::Lagged { variable, lag } => {
                nodes.push(TemporalNodeKeyWire { variable: variable.raw(), lag: lag.raw() });
            }
            other => {
                return Err(IoError::Convert(format!(
                    "temporal ADMG identity requires lagged nodes, got {other:?}"
                )));
            }
        }
    }
    let mut order: Vec<usize> = (0..nodes.len()).collect();
    order.sort_by_key(|&index| (nodes[index].variable, nodes[index].lag));
    let mut position = vec![0u32; nodes.len()];
    for (new, &old) in order.iter().enumerate() {
        position[old] = u32::try_from(new).map_err(|_| IoError::TooLarge)?;
    }
    let remap = |index: u32| {
        position.get(index as usize).copied().ok_or_else(|| {
            IoError::Convert("temporal ADMG edge references a node outside the graph".into())
        })
    };
    let mut directed = Vec::new();
    let mut bidirected = Vec::new();
    for edge in graph.edges() {
        if edge.is_bidirected() {
            let mut a = remap(edge.a.raw())?;
            let mut b = remap(edge.b.raw())?;
            if a > b {
                std::mem::swap(&mut a, &mut b);
            }
            bidirected.push((a, b));
        } else if let Some((from, to)) = edge.parent_child() {
            directed.push((remap(from.raw())?, remap(to.raw())?));
        }
    }
    directed.sort_unstable();
    bidirected.sort_unstable();
    bidirected.dedup();
    Ok(TemporalAdmgIdentityWire {
        nodes: order.into_iter().map(|index| nodes[index]).collect(),
        directed,
        bidirected,
    })
}

/// Encode a temporal CPDAG, including lagged and contemporaneous edges.
///
/// Nodes are sorted and edges re-indexed, so insertion order never reaches the digest. Nodes
/// with one key are one node (the graph treats them as the same variable at the same lag), so
/// a graph that repeats a node digests as the graph that does not.
///
/// # Errors
///
/// An edge index outside the node list, an edge joining two copies of one node, or two
/// different edges between one pair of nodes once copies are merged.
pub fn temporal_cpdag_identity(graph: &TemporalCpdag) -> Result<GraphIdentityWire, IoError> {
    temporal_class_identity(TEMPORAL_CPDAG_KIND, graph.nodes(), graph.edges())
}

/// Encode a temporal PAG, including lagged and contemporaneous edges.
///
/// Canonical as [`temporal_cpdag_identity`].
///
/// # Errors
///
/// As [`temporal_cpdag_identity`].
pub fn temporal_pag_identity(graph: &TemporalPag) -> Result<GraphIdentityWire, IoError> {
    temporal_class_identity(TEMPORAL_PAG_KIND, graph.nodes(), graph.edges())
}

/// Kind of a temporal CPDAG identity. The `.v2` marks the sorted, merged-node encoding; the
/// earlier encoding kept insertion order under `temporal_cpdag`.
const TEMPORAL_CPDAG_KIND: &str = "temporal_cpdag.v2";
/// Kind of a temporal PAG identity; see [`TEMPORAL_CPDAG_KIND`].
const TEMPORAL_PAG_KIND: &str = "temporal_pag.v2";

fn temporal_class_identity(
    kind: &'static str,
    nodes: &[NodeRef],
    edges: impl IntoIterator<Item = MarkedEdge>,
) -> Result<GraphIdentityWire, IoError> {
    let keys: Vec<IdentityNodeWire> =
        nodes.iter().copied().map(IdentityNodeWire::from_node).collect();
    let mut distinct = keys.clone();
    distinct.sort();
    distinct.dedup();
    let mut position = Vec::with_capacity(keys.len());
    for key in &keys {
        let index = distinct.binary_search(key).map_err(|_| {
            IoError::Convert("temporal class node is missing from its own node set".into())
        })?;
        position.push(u32::try_from(index).map_err(|_| IoError::TooLarge)?);
    }
    let remap = |index: u32| {
        position.get(index as usize).copied().ok_or_else(|| {
            IoError::Convert("temporal class edge references a node outside the graph".into())
        })
    };
    let mut marked = Vec::new();
    for edge in edges {
        let mut wire = marked_identity_edge(edge);
        wire.a = remap(wire.a)?;
        wire.b = remap(wire.b)?;
        if wire.a == wire.b {
            return Err(IoError::Convert(
                "temporal class edge joins two copies of one node".into(),
            ));
        }
        if wire.a > wire.b {
            std::mem::swap(&mut wire.a, &mut wire.b);
            std::mem::swap(&mut wire.at_a, &mut wire.at_b);
        }
        marked.push(wire);
    }
    marked.sort_by_key(|edge| (edge.a, edge.b));
    marked.dedup();
    if marked.windows(2).any(|pair| (pair[0].a, pair[0].b) == (pair[1].a, pair[1].b)) {
        return Err(IoError::Convert(
            "temporal class has different edges between one pair of nodes".into(),
        ));
    }
    Ok(GraphIdentityWire::TemporalClass(TemporalClassIdentityWire {
        kind: kind.into(),
        nodes: distinct,
        edges: marked,
    }))
}

/// Structure of one graph-posterior atom.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum PosteriorAtomGraphWire {
    /// Static DAG over the schema variables (canonical edge order).
    Static(DagWire),
    /// Static ADMG (canonical directed and bidirected edge order).
    Admg(AdmgWire),
    /// Static CPDAG (canonical directed and undirected edge order).
    Cpdag(CpdagWire),
    /// Static PAG (canonical marked edge order).
    Pag(PagWire),
    /// Lagged and contemporaneous DBN structure (canonical temporal wire).
    Temporal(TemporalGraphWire),
    /// Lagged ADMG atom (canonical temporal mixed-graph wire).
    TemporalAdmg(TemporalAdmgIdentityWire),
    /// Temporal class structure (lagged/context nodes and marked edges).
    TemporalClass(TemporalClassIdentityWire),
    /// Posterior weight whose mask is not a graph of the atom class.
    ///
    /// Execution keeps this weight as unidentified mass. It is not rewritten
    /// into another graph class.
    Unrepresentable,
}

/// Durable graph-posterior atom: structure plus posterior weight.
///
/// The variable namespace is the enclosing premises' `schema_names`.
/// `execution_key` maps the atom to the local cache key used by the frozen
/// handle; it is audit data and is excluded from the hashed premises.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PosteriorAtomIdentityWire {
    /// Identity format.
    pub format: u16,
    /// Posterior weight as IEEE-754 bits.
    pub weight_bits: u64,
    /// Atom structure.
    pub graph: PosteriorAtomGraphWire,
    /// Local envelope key for this frozen handle (not hashed).
    pub execution_key: u64,
}

/// Durable identities for every graph-posterior atom, static or DBN.
///
/// # Errors
///
/// Invalid adjacency or lag masks, or graph encode failure.
pub fn graph_posterior_atom_identities(
    posterior: &GraphPosterior,
) -> Result<Vec<PosteriorAtomIdentityWire>, IoError> {
    let ids: Vec<VariableId> = (0..posterior.n_vars())
        .map(|i| u32::try_from(i).map(VariableId::from_raw).map_err(|_| IoError::TooLarge))
        .collect::<Result<_, _>>()?;
    let max_lag = posterior.max_lag().unwrap_or(1);
    let mut atoms = Vec::with_capacity(posterior.n_graphs());
    for (index, (adjacency, &weight)) in
        posterior.atoms().iter().zip(posterior.weights().iter()).enumerate()
    {
        let graph = posterior_atom_graph_wire(posterior, index, adjacency, max_lag, &ids)?;
        atoms.push(PosteriorAtomIdentityWire {
            format: IDENTITY_FORMAT,
            weight_bits: weight.to_bits(),
            graph,
            execution_key: crate::error::fit_u64(index)?,
        });
    }
    Ok(atoms)
}

#[allow(
    clippy::too_many_lines,
    reason = "flat field-by-field wire check or conversion; the arms are independent and splitting would only scatter the list"
)]
fn posterior_atom_graph_wire(
    posterior: &GraphPosterior,
    index: usize,
    adjacency: &AdjacencyMask,
    max_lag: u32,
    ids: &[VariableId],
) -> Result<PosteriorAtomGraphWire, IoError> {
    let n_vars = posterior.n_vars();
    if let Some(lag_masks) = posterior.lag_masks() {
        let lag_mask = lag_masks
            .get(index)
            .ok_or_else(|| IoError::Convert("posterior lag masks must align with atoms".into()))?;
        return match posterior.atom_kind() {
            GraphPosteriorAtomKind::Cpdag => {
                let graph =
                    temporal_cpdag_from_dbn_atoms(adjacency, lag_mask, n_vars, max_lag, ids)
                        .map_err(|err| IoError::Convert(err.to_string()))?;
                let GraphIdentityWire::TemporalClass(wire) = temporal_cpdag_identity(&graph)?
                else {
                    return Err(IoError::Convert(
                        "temporal CPDAG posterior atom must encode as TemporalClass".into(),
                    ));
                };
                Ok(PosteriorAtomGraphWire::TemporalClass(wire))
            }
            GraphPosteriorAtomKind::TemporalAdmg => {
                // A lagged temporal-ADMG atom is its stationary template: the contemporaneous
                // ADMG, the lagged directed edges and the lagged latent-confounder edges.
                let atom = posterior.temporal_admg_atom(index).ok_or_else(|| {
                    IoError::Convert("posterior lag masks must align with atoms".into())
                })?;
                let Ok(temporal) = atom.to_template(n_vars, max_lag, ids) else {
                    return Ok(PosteriorAtomGraphWire::Unrepresentable);
                };
                let GraphIdentityWire::TemporalAdmg(wire) = temporal_admg_identity(&temporal)?
                else {
                    return Err(IoError::Convert(
                        "temporal ADMG posterior atom must encode as TemporalAdmg".into(),
                    ));
                };
                Ok(PosteriorAtomGraphWire::TemporalAdmg(wire))
            }
            GraphPosteriorAtomKind::Pag => {
                let mark_mask = posterior.mark_atom(index);
                let graph = temporal_pag_from_dbn_atoms(
                    adjacency, lag_mask, mark_mask, n_vars, max_lag, ids,
                )
                .map_err(|err| IoError::Convert(err.to_string()))?;
                let GraphIdentityWire::TemporalClass(wire) = temporal_pag_identity(&graph)? else {
                    return Err(IoError::Convert(
                        "temporal PAG posterior atom must encode as TemporalClass".into(),
                    ));
                };
                Ok(PosteriorAtomGraphWire::TemporalClass(wire))
            }
            _ => {
                let graph = temporal_dag_from_dbn_atoms(adjacency, lag_mask, n_vars, max_lag, ids)
                    .map_err(|err| IoError::Convert(err.to_string()))?;
                Ok(PosteriorAtomGraphWire::Temporal(canonical_temporal_dag_wire(&graph)?))
            }
        };
    }
    match posterior.atom_kind() {
        GraphPosteriorAtomKind::Admg => {
            let Ok(admg) = admg_from_adjacency_mask(adjacency, n_vars) else {
                return Ok(PosteriorAtomGraphWire::Unrepresentable);
            };
            Ok(PosteriorAtomGraphWire::Admg(canonical_admg_wire(&admg)?))
        }
        GraphPosteriorAtomKind::TemporalAdmg => {
            let Ok(admg) = admg_from_adjacency_mask(adjacency, n_vars) else {
                return Ok(PosteriorAtomGraphWire::Unrepresentable);
            };
            let temporal = TemporalAdmg::from_static_contemporaneous(&admg)
                .map_err(|err| IoError::Convert(err.to_string()))?;
            let GraphIdentityWire::TemporalAdmg(wire) = temporal_admg_identity(&temporal)? else {
                return Err(IoError::Convert(
                    "temporal ADMG posterior atom must encode as TemporalAdmg".into(),
                ));
            };
            Ok(PosteriorAtomGraphWire::TemporalAdmg(wire))
        }
        GraphPosteriorAtomKind::Cpdag => {
            // A mask that is no CPDAG is unidentified mass, as for the other static kinds.
            let Ok(cpdag) = cpdag_from_adjacency_mask(adjacency, n_vars) else {
                return Ok(PosteriorAtomGraphWire::Unrepresentable);
            };
            let GraphIdentityWire::Cpdag(wire) = cpdag_identity(&cpdag)? else {
                return Err(IoError::Convert("CPDAG posterior atom must encode as Cpdag".into()));
            };
            Ok(PosteriorAtomGraphWire::Cpdag(wire))
        }
        GraphPosteriorAtomKind::Pag => {
            let mark_mask = posterior.mark_atom(index);
            let Ok(pag) = pag_from_adjacency_mask(adjacency, mark_mask, n_vars) else {
                return Ok(PosteriorAtomGraphWire::Unrepresentable);
            };
            let GraphIdentityWire::Pag(wire) = pag_identity(&pag)? else {
                return Err(IoError::Convert("PAG posterior atom must encode as Pag".into()));
            };
            Ok(PosteriorAtomGraphWire::Pag(wire))
        }
        _ => {
            // An atom that is not a DAG carries unidentified mass; it has no graph to encode.
            let Ok(dag) = dag_from_adjacency_mask(adjacency, n_vars) else {
                return Ok(PosteriorAtomGraphWire::Unrepresentable);
            };
            Ok(PosteriorAtomGraphWire::Static(canonical_dag_wire(&dag)?))
        }
    }
}
