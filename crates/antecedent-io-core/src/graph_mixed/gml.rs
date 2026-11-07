//! GML interchange for mixed causal graphs.

use super::{
    Admg, AdmgWire, Cpdag, EndpointWire, GraphCpdagWire, GraphLimits, IoError, MarkedEdgeWire, Pag,
    PagWire, admg_from_wire_with_limits, admg_to_wire, canon, cpdag_from_wire_with_limits,
    cpdag_to_wire, endpoint_str, pag_from_wire_with_limits, pag_to_wire, parse_endpoint,
    without_names,
};
use crate::graph_gml;
use std::borrow::Cow;
use std::fmt::Write as _;

// ── GML ─────────────────────────────────────────────────────────────────────

without_names! {
/// Parse GML into a [`Cpdag`] (edge attr `undirected 1`).
///
/// # Errors
///
/// Malformed GML.
    pub fn cpdag_from_gml(gml: &str) -> Cpdag = cpdag_with_names_from_gml
}

/// Parse GML into a [`Cpdag`] (edge attr `undirected 1`) plus its node
/// labels, in dense-id order.
///
/// # Errors
///
/// Malformed GML.
pub fn cpdag_with_names_from_gml(gml: &str) -> Result<(Cpdag, Vec<String>), IoError> {
    cpdag_with_names_from_gml_with_limits(gml, &GraphLimits::default())
}

/// [`cpdag_with_names_from_gml`] under the caller's graph bounds: a document over `limits` is refused.
///
/// # Errors
///
/// As [`cpdag_with_names_from_gml`], and [`IoError::Convert`] when a bound of `limits` is exceeded.
pub fn cpdag_with_names_from_gml_with_limits(
    gml: &str,
    limits: &GraphLimits,
) -> Result<(Cpdag, Vec<String>), IoError> {
    let g = parse_gml(gml)?;
    let cpdag = cpdag_from_wire_with_limits(
        &GraphCpdagWire {
            node_count: g.node_count,
            directed: g.directed,
            undirected: g.undirected,
        },
        limits,
    )?;
    Ok((cpdag, g.names))
}

/// Serialize a [`Cpdag`] to GML.
///
/// # Errors
///
/// Wire conversion.
pub fn cpdag_to_gml(cpdag: &Cpdag, names: Option<&[String]>) -> Result<String, IoError> {
    let wire = cpdag_to_wire(cpdag)?;
    Ok(emit_gml(
        wire.node_count,
        names,
        wire.directed.iter().map(|&(a, b)| GmlEdge::directed(a, b)),
        wire.undirected.iter().map(|&(a, b)| GmlEdge::undirected(a, b)),
    ))
}

without_names! {
/// Parse GML into a [`Pag`] (edge attrs `mark_a` / `mark_b`).
///
/// # Errors
///
/// Malformed GML.
    pub fn pag_from_gml(gml: &str) -> Pag = pag_with_names_from_gml
}

/// Parse GML into a [`Pag`] (edge attrs `mark_a` / `mark_b`) plus its node
/// labels, in dense-id order.
///
/// # Errors
///
/// Malformed GML.
pub fn pag_with_names_from_gml(gml: &str) -> Result<(Pag, Vec<String>), IoError> {
    pag_with_names_from_gml_with_limits(gml, &GraphLimits::default())
}

/// [`pag_with_names_from_gml`] under the caller's graph bounds: a document over `limits` is refused.
///
/// # Errors
///
/// As [`pag_with_names_from_gml`], and [`IoError::Convert`] when a bound of `limits` is exceeded.
pub fn pag_with_names_from_gml_with_limits(
    gml: &str,
    limits: &GraphLimits,
) -> Result<(Pag, Vec<String>), IoError> {
    let g = parse_gml(gml)?;
    let mut edges = g.marked;
    for &(a, b) in &g.directed {
        edges.push(MarkedEdgeWire { a, b, at_a: EndpointWire::Tail, at_b: EndpointWire::Arrow });
    }
    for &(a, b) in &g.undirected {
        edges.push(MarkedEdgeWire { a, b, at_a: EndpointWire::Tail, at_b: EndpointWire::Tail });
    }
    for &(a, b) in &g.bidirected {
        edges.push(MarkedEdgeWire { a, b, at_a: EndpointWire::Arrow, at_b: EndpointWire::Arrow });
    }
    let pag = pag_from_wire_with_limits(&PagWire { node_count: g.node_count, edges }, limits)?;
    Ok((pag, g.names))
}

/// Serialize a [`Pag`] to GML.
///
/// # Errors
///
/// Wire conversion.
pub fn pag_to_gml(pag: &Pag, names: Option<&[String]>) -> Result<String, IoError> {
    let wire = pag_to_wire(pag)?;
    Ok(emit_gml(
        wire.node_count,
        names,
        wire.edges.iter().map(|e| GmlEdge::marked(e.a, e.b, e.at_a, e.at_b)),
        std::iter::empty(),
    ))
}

without_names! {
/// Parse GML into an [`Admg`] (edge attr `bidirected 1`).
///
/// # Errors
///
/// Malformed GML.
    pub fn admg_from_gml(gml: &str) -> Admg = admg_with_names_from_gml
}

/// Parse GML into an [`Admg`] (edge attr `bidirected 1`) plus its node
/// labels, in dense-id order.
///
/// # Errors
///
/// Malformed GML.
pub fn admg_with_names_from_gml(gml: &str) -> Result<(Admg, Vec<String>), IoError> {
    admg_with_names_from_gml_with_limits(gml, &GraphLimits::default())
}

/// [`admg_with_names_from_gml`] under the caller's graph bounds: a document over `limits` is refused.
///
/// # Errors
///
/// As [`admg_with_names_from_gml`], and [`IoError::Convert`] when a bound of `limits` is exceeded.
pub fn admg_with_names_from_gml_with_limits(
    gml: &str,
    limits: &GraphLimits,
) -> Result<(Admg, Vec<String>), IoError> {
    let g = parse_gml(gml)?;
    let admg = admg_from_wire_with_limits(
        &AdmgWire { node_count: g.node_count, directed: g.directed, bidirected: g.bidirected },
        limits,
    )?;
    Ok((admg, g.names))
}

/// Serialize an [`Admg`] to GML.
///
/// # Errors
///
/// Wire conversion.
pub fn admg_to_gml(admg: &Admg, names: Option<&[String]>) -> Result<String, IoError> {
    let wire = admg_to_wire(admg)?;
    Ok(emit_gml(
        wire.node_count,
        names,
        wire.directed.iter().map(|&(a, b)| GmlEdge::directed(a, b)),
        wire.bidirected.iter().map(|&(a, b)| GmlEdge::bidirected(a, b)),
    ))
}

struct GmlGraph {
    node_count: u32,
    directed: Vec<(u32, u32)>,
    undirected: Vec<(u32, u32)>,
    bidirected: Vec<(u32, u32)>,
    marked: Vec<MarkedEdgeWire>,
    /// Node labels in dense-id order.
    names: Vec<String>,
}

enum GmlEdge {
    Dir(u32, u32),
    Undir(u32, u32),
    Bi(u32, u32),
    Mark(u32, u32, EndpointWire, EndpointWire),
}

impl GmlEdge {
    fn directed(a: u32, b: u32) -> Self {
        Self::Dir(a, b)
    }
    fn undirected(a: u32, b: u32) -> Self {
        Self::Undir(a, b)
    }
    fn bidirected(a: u32, b: u32) -> Self {
        Self::Bi(a, b)
    }
    fn marked(a: u32, b: u32, at_a: EndpointWire, at_b: EndpointWire) -> Self {
        Self::Mark(a, b, at_a, at_b)
    }
}

fn parse_gml(gml: &str) -> Result<GmlGraph, IoError> {
    let doc = graph_gml::parse_gml_document(gml)?;
    let mut g = GmlGraph {
        node_count: u32::try_from(doc.names.len()).map_err(|_| IoError::TooLarge)?,
        directed: Vec::new(),
        undirected: Vec::new(),
        bidirected: Vec::new(),
        marked: Vec::new(),
        names: doc.names,
    };
    for e in doc.edges {
        let flag = |k: &str| e.attrs.get(k).is_some_and(|v| v != "0");
        match (e.attrs.get("mark_a"), e.attrs.get("mark_b")) {
            (Some(a), Some(b)) => g.marked.push(MarkedEdgeWire {
                a: e.from,
                b: e.to,
                at_a: parse_endpoint(a)?,
                at_b: parse_endpoint(b)?,
            }),
            (None, None) if flag("bidirected") => g.bidirected.push(canon(e.from, e.to)),
            (None, None) if flag("undirected") => g.undirected.push(canon(e.from, e.to)),
            (None, None) => g.directed.push((e.from, e.to)),
            _ => {
                return Err(IoError::Convert(
                    "GML edge sets only one of mark_a / mark_b; both are required".into(),
                ));
            }
        }
    }
    Ok(g)
}

#[allow(clippy::match_same_arms, reason = "the cases retain distinct wire or replay semantics")]
fn emit_gml(
    node_count: u32,
    names: Option<&[String]>,
    edges_a: impl Iterator<Item = GmlEdge>,
    edges_b: impl Iterator<Item = GmlEdge>,
) -> String {
    // Writing into a `String` cannot fail, so the `fmt::Result`s below are
    // discarded explicitly.
    let mut out = String::from("graph [\n  directed 1\n");
    for i in 0..node_count {
        let label = gml_label(names, i);
        out.push_str("  node [\n    id \"");
        graph_gml::push_escaped_gml(&mut out, &label);
        out.push_str("\"\n    label \"");
        graph_gml::push_escaped_gml(&mut out, &label);
        out.push_str("\"\n  ]\n");
    }
    for e in edges_a.chain(edges_b) {
        let (a, b) = match e {
            GmlEdge::Dir(a, b) | GmlEdge::Undir(a, b) | GmlEdge::Bi(a, b) => (a, b),
            GmlEdge::Mark(a, b, _, _) => (a, b),
        };
        out.push_str("  edge [\n    source \"");
        graph_gml::push_escaped_gml(&mut out, &gml_label(names, a));
        out.push_str("\"\n    target \"");
        graph_gml::push_escaped_gml(&mut out, &gml_label(names, b));
        out.push_str("\"\n");
        match e {
            GmlEdge::Dir(..) => {}
            GmlEdge::Undir(..) => out.push_str("    undirected \"1\"\n"),
            GmlEdge::Bi(..) => out.push_str("    bidirected \"1\"\n"),
            GmlEdge::Mark(_, _, at_a, at_b) => {
                let _ = write!(
                    out,
                    "    mark_a \"{}\"\n    mark_b \"{}\"\n",
                    endpoint_str(at_a),
                    endpoint_str(at_b)
                );
            }
        }
        out.push_str("  ]\n");
    }
    out.push(']');
    out
}

/// The name of node `i`, or its index when the graph carries no names.
fn gml_label(names: Option<&[String]>, i: u32) -> Cow<'_, str> {
    names
        .and_then(|n| n.get(i as usize))
        .map_or_else(|| Cow::Owned(i.to_string()), |name| Cow::Borrowed(name.as_str()))
}
