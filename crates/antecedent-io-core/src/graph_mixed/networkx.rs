//! `NetworkX` node-link interchange for mixed causal graphs.

use super::{
    Admg, AdmgWire, Cpdag, EndpointWire, GraphCpdagWire, GraphLimits, IoError, MarkedEdgeWire, Pag,
    PagWire, admg_from_wire_with_limits, admg_to_wire, canon, cpdag_from_wire_with_limits,
    cpdag_to_wire, endpoint_str, pag_from_wire_with_limits, pag_to_wire, parse_endpoint,
    without_names,
};
use crate::graph_networkx::NetworkXNode;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

// ── NetworkX node-link ──────────────────────────────────────────────────────

without_names! {
/// Parse `NetworkX` node-link JSON into a [`Cpdag`].
///
/// # Errors
///
/// Malformed JSON.
    pub fn cpdag_from_networkx_node_link(json: &str) -> Cpdag = cpdag_with_names_from_networkx_node_link
}

/// Parse `NetworkX` node-link JSON into a [`Cpdag`] plus its node names
/// (the document's node `id` values, stringified, in dense-id order).
///
/// # Errors
///
/// Malformed JSON.
pub fn cpdag_with_names_from_networkx_node_link(
    json: &str,
) -> Result<(Cpdag, Vec<String>), IoError> {
    cpdag_with_names_from_networkx_node_link_with_limits(json, &GraphLimits::default())
}

/// [`cpdag_with_names_from_networkx_node_link`] under the caller's graph bounds: a document over `limits` is refused.
///
/// # Errors
///
/// As [`cpdag_with_names_from_networkx_node_link`], and [`IoError::Convert`] when a bound of `limits` is exceeded.
pub fn cpdag_with_names_from_networkx_node_link_with_limits(
    json: &str,
    limits: &GraphLimits,
) -> Result<(Cpdag, Vec<String>), IoError> {
    let (order, links) = parse_nx(json)?;
    let mut directed = Vec::new();
    let mut undirected = Vec::new();
    for link in links {
        if link.undirected == Some(true) {
            undirected.push(canon(link.from, link.to));
        } else {
            directed.push((link.from, link.to));
        }
    }
    let node_count = u32::try_from(order.len()).map_err(|_| IoError::TooLarge)?;
    let g =
        cpdag_from_wire_with_limits(&GraphCpdagWire { node_count, directed, undirected }, limits)?;
    Ok((g, order))
}

/// Serialize a [`Cpdag`] to `NetworkX` node-link JSON.
///
/// # Errors
///
/// JSON failures.
pub fn cpdag_to_networkx_node_link(
    cpdag: &Cpdag,
    names: Option<&[String]>,
) -> Result<String, IoError> {
    let wire = cpdag_to_wire(cpdag)?;
    let mut links = Vec::new();
    for &(a, b) in &wire.directed {
        links.push(nx_link(a, b, names, None, None, None, None));
    }
    for &(a, b) in &wire.undirected {
        links.push(nx_link(a, b, names, Some(true), None, None, None));
    }
    emit_nx(wire.node_count, names, links)
}

without_names! {
/// Parse `NetworkX` node-link JSON into a [`Pag`].
///
/// # Errors
///
/// Malformed JSON.
    pub fn pag_from_networkx_node_link(json: &str) -> Pag = pag_with_names_from_networkx_node_link
}

/// Parse `NetworkX` node-link JSON into a [`Pag`] plus its node names (the
/// document's node `id` values, stringified, in dense-id order).
///
/// # Errors
///
/// Malformed JSON.
pub fn pag_with_names_from_networkx_node_link(json: &str) -> Result<(Pag, Vec<String>), IoError> {
    pag_with_names_from_networkx_node_link_with_limits(json, &GraphLimits::default())
}

/// [`pag_with_names_from_networkx_node_link`] under the caller's graph bounds: a document over `limits` is refused.
///
/// # Errors
///
/// As [`pag_with_names_from_networkx_node_link`], and [`IoError::Convert`] when a bound of `limits` is exceeded.
pub fn pag_with_names_from_networkx_node_link_with_limits(
    json: &str,
    limits: &GraphLimits,
) -> Result<(Pag, Vec<String>), IoError> {
    let (order, links) = parse_nx(json)?;
    let mut edges = Vec::new();
    for link in links {
        let (at_a, at_b) = if let (Some(a), Some(b)) = (link.mark_a, link.mark_b) {
            (parse_endpoint(&a)?, parse_endpoint(&b)?)
        } else if link.bidirected == Some(true)
            || link.dir.as_deref().is_some_and(|d| d.eq_ignore_ascii_case("both"))
        {
            (EndpointWire::Arrow, EndpointWire::Arrow)
        } else if link.undirected == Some(true) {
            (EndpointWire::Tail, EndpointWire::Tail)
        } else {
            (EndpointWire::Tail, EndpointWire::Arrow)
        };
        let (a, b, at_a, at_b) = if link.from <= link.to {
            (link.from, link.to, at_a, at_b)
        } else {
            (link.to, link.from, at_b, at_a)
        };
        edges.push(MarkedEdgeWire { a, b, at_a, at_b });
    }
    let node_count = u32::try_from(order.len()).map_err(|_| IoError::TooLarge)?;
    let g = pag_from_wire_with_limits(&PagWire { node_count, edges }, limits)?;
    Ok((g, order))
}

/// Serialize a [`Pag`] to `NetworkX` node-link JSON.
///
/// # Errors
///
/// JSON failures.
pub fn pag_to_networkx_node_link(pag: &Pag, names: Option<&[String]>) -> Result<String, IoError> {
    let wire = pag_to_wire(pag)?;
    let links = wire
        .edges
        .iter()
        .map(|e| {
            nx_link(
                e.a,
                e.b,
                names,
                None,
                None,
                Some(endpoint_str(e.at_a).into()),
                Some(endpoint_str(e.at_b).into()),
            )
        })
        .collect();
    emit_nx(wire.node_count, names, links)
}

without_names! {
/// Parse `NetworkX` node-link JSON into an [`Admg`].
///
/// # Errors
///
/// Malformed JSON.
    pub fn admg_from_networkx_node_link(json: &str) -> Admg = admg_with_names_from_networkx_node_link
}

/// Parse `NetworkX` node-link JSON into an [`Admg`] plus its node names
/// (the document's node `id` values, stringified, in dense-id order).
///
/// # Errors
///
/// Malformed JSON.
pub fn admg_with_names_from_networkx_node_link(json: &str) -> Result<(Admg, Vec<String>), IoError> {
    admg_with_names_from_networkx_node_link_with_limits(json, &GraphLimits::default())
}

/// [`admg_with_names_from_networkx_node_link`] under the caller's graph bounds: a document over `limits` is refused.
///
/// # Errors
///
/// As [`admg_with_names_from_networkx_node_link`], and [`IoError::Convert`] when a bound of `limits` is exceeded.
pub fn admg_with_names_from_networkx_node_link_with_limits(
    json: &str,
    limits: &GraphLimits,
) -> Result<(Admg, Vec<String>), IoError> {
    let (order, links) = parse_nx(json)?;
    let mut directed = Vec::new();
    let mut bidirected = Vec::new();
    for link in links {
        if link.bidirected == Some(true)
            || link.dir.as_deref().is_some_and(|d| d.eq_ignore_ascii_case("both"))
        {
            bidirected.push(canon(link.from, link.to));
        } else {
            directed.push((link.from, link.to));
        }
    }
    let node_count = u32::try_from(order.len()).map_err(|_| IoError::TooLarge)?;
    let g = admg_from_wire_with_limits(&AdmgWire { node_count, directed, bidirected }, limits)?;
    Ok((g, order))
}

/// Serialize an [`Admg`] to `NetworkX` node-link JSON.
///
/// # Errors
///
/// JSON failures.
pub fn admg_to_networkx_node_link(
    admg: &Admg,
    names: Option<&[String]>,
) -> Result<String, IoError> {
    let wire = admg_to_wire(admg)?;
    let mut links = Vec::new();
    for &(a, b) in &wire.directed {
        links.push(nx_link(a, b, names, None, None, None, None));
    }
    for &(a, b) in &wire.bidirected {
        links.push(nx_link(a, b, names, None, Some(true), None, None));
    }
    emit_nx(wire.node_count, names, links)
}

#[derive(Clone, Debug, Deserialize)]
struct NxDoc {
    directed: bool,
    #[serde(default)]
    multigraph: bool,
    nodes: Vec<NetworkXNode>,
    links: Vec<NxLinkRaw>,
}

#[derive(Clone, Debug, Deserialize)]
struct NxLinkRaw {
    source: JsonValue,
    target: JsonValue,
    #[serde(default)]
    undirected: Option<bool>,
    #[serde(default)]
    bidirected: Option<bool>,
    #[serde(default)]
    dir: Option<String>,
    #[serde(default)]
    mark_a: Option<String>,
    #[serde(default)]
    mark_b: Option<String>,
}

impl crate::graph_networkx::NodeLinkEndpoints for NxLinkRaw {
    fn endpoints(&self) -> (&JsonValue, &JsonValue) {
        (&self.source, &self.target)
    }
}

struct NxLink {
    from: u32,
    to: u32,
    undirected: Option<bool>,
    bidirected: Option<bool>,
    dir: Option<String>,
    mark_a: Option<String>,
    mark_b: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
struct NxOut {
    directed: bool,
    multigraph: bool,
    graph: JsonValue,
    nodes: Vec<NetworkXNode>,
    links: Vec<NxOutLink>,
}

#[derive(Clone, Debug, Serialize)]
struct NxOutLink {
    source: JsonValue,
    target: JsonValue,
    #[serde(skip_serializing_if = "Option::is_none")]
    undirected: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    bidirected: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mark_a: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mark_b: Option<String>,
}

fn parse_nx(json: &str) -> Result<(Vec<String>, Vec<NxLink>), IoError> {
    let doc: NxDoc =
        serde_json::from_str(json).map_err(|e| IoError::Convert(format!("json: {e}")))?;
    crate::graph_networkx::require_simple_directed(doc.directed, doc.multigraph)?;
    let (order, bound) = crate::graph_networkx::bind_node_link(&doc.nodes, &doc.links)?;
    let links = bound
        .into_iter()
        .zip(doc.links)
        .map(|((from, to), link)| NxLink {
            from,
            to,
            undirected: link.undirected,
            bidirected: link.bidirected,
            dir: link.dir,
            mark_a: link.mark_a,
            mark_b: link.mark_b,
        })
        .collect();
    Ok((order, links))
}

fn nx_link(
    a: u32,
    b: u32,
    names: Option<&[String]>,
    undirected: Option<bool>,
    bidirected: Option<bool>,
    mark_a: Option<String>,
    mark_b: Option<String>,
) -> NxOutLink {
    NxOutLink {
        source: node_id(a, names),
        target: node_id(b, names),
        undirected,
        bidirected,
        mark_a,
        mark_b,
    }
}

fn emit_nx(
    node_count: u32,
    names: Option<&[String]>,
    links: Vec<NxOutLink>,
) -> Result<String, IoError> {
    let nodes = (0..node_count).map(|i| NetworkXNode { id: node_id(i, names) }).collect();
    let doc = NxOut {
        directed: true,
        multigraph: false,
        graph: JsonValue::Object(serde_json::Map::new()),
        nodes,
        links,
    };
    serde_json::to_string_pretty(&doc).map_err(|e| IoError::Convert(format!("json: {e}")))
}

fn node_id(i: u32, names: Option<&[String]>) -> JsonValue {
    names
        .and_then(|n| n.get(i as usize))
        .cloned()
        .map(JsonValue::String)
        .unwrap_or(JsonValue::Number(i.into()))
}
