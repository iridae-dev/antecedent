//! PAG / CPDAG / ADMG interchange (DOT / JSON / GML / `NetworkX`).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;

use antecedent_graph::{Admg, Cpdag, Pag};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::convert::{
    admg_from_wire, admg_to_wire, cpdag_from_wire, cpdag_to_wire, pag_from_wire, pag_to_wire,
};
use crate::error::IoError;
use crate::graph_dot::{self, Lexer};
use crate::graph_gml::{self, Tok};
use crate::graph_networkx::{NetworkXNode, json_id_to_string};
use crate::wire::{AdmgWire, CpdagWire, EndpointWire, MarkedEdgeWire, PagWire};

// ── JSON ────────────────────────────────────────────────────────────────────

/// JSON document for a static PAG.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PagJson {
    /// Discriminator.
    #[serde(default = "pag_kind")]
    pub kind: String,
    /// Dense node count.
    pub node_count: u32,
    /// Marked edges.
    pub edges: Vec<MarkedEdgeWire>,
    /// Optional variable names.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub variable_names: Option<Vec<String>>,
}

fn pag_kind() -> String {
    "pag".into()
}

/// JSON document for a static CPDAG.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CpdagJson {
    /// Discriminator.
    #[serde(default = "cpdag_kind")]
    pub kind: String,
    /// Dense node count.
    pub node_count: u32,
    /// Directed edges.
    pub directed: Vec<(u32, u32)>,
    /// Undirected edges.
    pub undirected: Vec<(u32, u32)>,
    /// Optional variable names.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub variable_names: Option<Vec<String>>,
}

fn cpdag_kind() -> String {
    "cpdag".into()
}

/// JSON document for a static ADMG.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct AdmgJson {
    /// Discriminator.
    #[serde(default = "admg_kind")]
    pub kind: String,
    /// Dense node count.
    pub node_count: u32,
    /// Directed edges.
    pub directed: Vec<(u32, u32)>,
    /// Bidirected edges.
    pub bidirected: Vec<(u32, u32)>,
    /// Optional variable names.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub variable_names: Option<Vec<String>>,
}

fn admg_kind() -> String {
    "admg".into()
}

fn json_from_str<T: for<'de> Deserialize<'de>>(json: &str) -> Result<T, IoError> {
    serde_json::from_str(json).map_err(|e| IoError::Convert(format!("json: {e}")))
}

fn json_to_pretty<T: Serialize>(doc: &T) -> Result<String, IoError> {
    serde_json::to_string_pretty(doc).map_err(|e| IoError::Convert(format!("json: {e}")))
}

/// Emit JSON for a graph after wire conversion (shared pag/cpdag/admg path).
macro_rules! graph_to_json {
    ($name:ident, $Graph:ty, $to_wire:ident, $Json:ident, $kind:expr, { $($field:ident),+ $(,)? }) => {
        /// Serialize to JSON.
        ///
        /// # Errors
        ///
        /// Wire / JSON failures.
        pub fn $name(graph: &$Graph, names: Option<&[String]>) -> Result<String, IoError> {
            let wire = $to_wire(graph)?;
            check_optional_names(names, wire.node_count)?;
            let doc = $Json {
                kind: $kind,
                node_count: wire.node_count,
                $($field: wire.$field,)+
                variable_names: names.map(<[String]>::to_vec),
            };
            json_to_pretty(&doc)
        }
    };
}

/// Parse JSON into a [`Pag`].
///
/// # Errors
///
/// JSON / structure errors.
pub fn pag_from_json(json: &str) -> Result<Pag, IoError> {
    pag_with_names_from_json(json).map(|(g, _names)| g)
}

/// Parse JSON into a [`Pag`] plus its node names.
///
/// When the document has no `variable_names`, dense-index strings are
/// returned instead.
///
/// # Errors
///
/// JSON / structure errors.
pub fn pag_with_names_from_json(json: &str) -> Result<(Pag, Vec<String>), IoError> {
    let doc: PagJson = json_from_str(json)?;
    let names = names_or_default(doc.variable_names.as_deref(), doc.node_count)?;
    let g = pag_from_wire(&PagWire { node_count: doc.node_count, edges: doc.edges })?;
    Ok((g, names))
}

graph_to_json!(pag_to_json, Pag, pag_to_wire, PagJson, pag_kind(), { edges });

/// Parse JSON into a [`Cpdag`].
///
/// # Errors
///
/// JSON / structure errors.
pub fn cpdag_from_json(json: &str) -> Result<Cpdag, IoError> {
    cpdag_with_names_from_json(json).map(|(g, _names)| g)
}

/// Parse JSON into a [`Cpdag`] plus its node names.
///
/// When the document has no `variable_names`, dense-index strings are
/// returned instead.
///
/// # Errors
///
/// JSON / structure errors.
pub fn cpdag_with_names_from_json(json: &str) -> Result<(Cpdag, Vec<String>), IoError> {
    let doc: CpdagJson = json_from_str(json)?;
    let names = names_or_default(doc.variable_names.as_deref(), doc.node_count)?;
    let g = cpdag_from_wire(&CpdagWire {
        node_count: doc.node_count,
        directed: doc.directed,
        undirected: doc.undirected,
    })?;
    Ok((g, names))
}

graph_to_json!(
    cpdag_to_json,
    Cpdag,
    cpdag_to_wire,
    CpdagJson,
    cpdag_kind(),
    { directed, undirected }
);

/// Parse JSON into an [`Admg`].
///
/// # Errors
///
/// JSON / structure errors.
pub fn admg_from_json(json: &str) -> Result<Admg, IoError> {
    admg_with_names_from_json(json).map(|(g, _names)| g)
}

/// Parse JSON into an [`Admg`] plus its node names.
///
/// When the document has no `variable_names`, dense-index strings are
/// returned instead.
///
/// # Errors
///
/// JSON / structure errors.
pub fn admg_with_names_from_json(json: &str) -> Result<(Admg, Vec<String>), IoError> {
    let doc: AdmgJson = json_from_str(json)?;
    let names = names_or_default(doc.variable_names.as_deref(), doc.node_count)?;
    let g = admg_from_wire(&AdmgWire {
        node_count: doc.node_count,
        directed: doc.directed,
        bidirected: doc.bidirected,
    })?;
    Ok((g, names))
}

graph_to_json!(
    admg_to_json,
    Admg,
    admg_to_wire,
    AdmgJson,
    admg_kind(),
    { directed, bidirected }
);

fn check_optional_names(names: Option<&[String]>, node_count: u32) -> Result<(), IoError> {
    if let Some(n) = names {
        if n.len() != node_count as usize {
            return Err(IoError::Convert(format!(
                "variable_names length {} must equal node_count {node_count}",
                n.len()
            )));
        }
    }
    Ok(())
}

/// Validated names, or dense-index strings when the document carries none.
fn names_or_default(names: Option<&[String]>, node_count: u32) -> Result<Vec<String>, IoError> {
    check_optional_names(names, node_count)?;
    Ok(names.map_or_else(|| (0..node_count).map(|i| i.to_string()).collect(), <[String]>::to_vec))
}

// ── DOT ─────────────────────────────────────────────────────────────────────

/// Parse DOT into a [`Cpdag`] (`->` directed, `--` undirected).
///
/// # Errors
///
/// Malformed DOT or illegal CPDAG marks.
pub fn cpdag_from_dot(dot: &str) -> Result<Cpdag, IoError> {
    cpdag_with_names_from_dot(dot).map(|(g, _names)| g)
}

/// Parse DOT into a [`Cpdag`] (`->` directed, `--` undirected) plus its node
/// labels, in dense-id order.
///
/// # Errors
///
/// Malformed DOT or illegal CPDAG marks.
pub fn cpdag_with_names_from_dot(dot: &str) -> Result<(Cpdag, Vec<String>), IoError> {
    let parsed = parse_dot(dot, true)?;
    let mut directed = Vec::new();
    let mut undirected = Vec::new();
    for e in parsed.edges {
        match e.kind {
            DotKind::Directed => directed.push((e.from, e.to)),
            DotKind::Undirected => undirected.push(canon(e.from, e.to)),
            _ => {
                return Err(IoError::Convert("CPDAG DOT accepts only -> and -- edges".into()));
            }
        }
    }
    let g = cpdag_from_wire(&CpdagWire { node_count: parsed.node_count, directed, undirected })?;
    Ok((g, parsed.names))
}

/// Serialize a [`Cpdag`] to DOT.
///
/// # Errors
///
/// Wire conversion failures.
pub fn cpdag_to_dot(cpdag: &Cpdag, names: Option<&[String]>) -> Result<String, IoError> {
    let wire = cpdag_to_wire(cpdag)?;
    let use_names = names.is_some_and(|n| n.len() == wire.node_count as usize);
    let mut out = String::from("digraph {\n");
    for &(a, b) in &wire.directed {
        emit_simple_edge(&mut out, a, b, "->", names, use_names);
    }
    for &(a, b) in &wire.undirected {
        emit_simple_edge(&mut out, a, b, "--", names, use_names);
    }
    emit_isolated(&mut out, wire.node_count, &wire.directed, &wire.undirected, names, use_names);
    out.push('}');
    Ok(out)
}

/// Parse DOT into a [`Pag`].
///
/// Plain `->` is tail→arrow; `--` is undirected; `dir=both` is bidirected;
/// otherwise `mark_a` / `mark_b` attributes.
///
/// # Errors
///
/// Malformed DOT or illegal PAG marks.
pub fn pag_from_dot(dot: &str) -> Result<Pag, IoError> {
    pag_with_names_from_dot(dot).map(|(g, _names)| g)
}

/// Parse DOT into a [`Pag`] plus its node labels, in dense-id order.
///
/// Plain `->` is tail→arrow; `--` is undirected; `dir=both` is bidirected;
/// otherwise `mark_a` / `mark_b` attributes.
///
/// # Errors
///
/// Malformed DOT or illegal PAG marks.
pub fn pag_with_names_from_dot(dot: &str) -> Result<(Pag, Vec<String>), IoError> {
    let parsed = parse_dot(dot, true)?;
    let mut edges = Vec::new();
    for e in parsed.edges {
        let (at_a, at_b) = match e.kind {
            DotKind::Directed => (EndpointWire::Tail, EndpointWire::Arrow),
            DotKind::Undirected => (EndpointWire::Tail, EndpointWire::Tail),
            DotKind::Bidirected => (EndpointWire::Arrow, EndpointWire::Arrow),
            DotKind::Marked { at_a, at_b } => (at_a, at_b),
        };
        let (a, b, at_a, at_b) =
            if e.from <= e.to { (e.from, e.to, at_a, at_b) } else { (e.to, e.from, at_b, at_a) };
        edges.push(MarkedEdgeWire { a, b, at_a, at_b });
    }
    let g = pag_from_wire(&PagWire { node_count: parsed.node_count, edges })?;
    Ok((g, parsed.names))
}

/// Serialize a [`Pag`] to DOT.
///
/// # Errors
///
/// Wire conversion failures.
pub fn pag_to_dot(pag: &Pag, names: Option<&[String]>) -> Result<String, IoError> {
    let wire = pag_to_wire(pag)?;
    let use_names = names.is_some_and(|n| n.len() == wire.node_count as usize);
    let mut out = String::from("digraph {\n");
    let mut seen = vec![false; wire.node_count as usize];
    for e in &wire.edges {
        seen[e.a as usize] = true;
        seen[e.b as usize] = true;
        out.push(' ');
        match (e.at_a, e.at_b) {
            (EndpointWire::Tail, EndpointWire::Arrow) => {
                push_label(&mut out, e.a, names, use_names);
                out.push_str(" -> ");
                push_label(&mut out, e.b, names, use_names);
                out.push_str(";\n");
            }
            (EndpointWire::Arrow, EndpointWire::Tail) => {
                push_label(&mut out, e.b, names, use_names);
                out.push_str(" -> ");
                push_label(&mut out, e.a, names, use_names);
                out.push_str(";\n");
            }
            (EndpointWire::Tail, EndpointWire::Tail) => {
                push_label(&mut out, e.a, names, use_names);
                out.push_str(" -- ");
                push_label(&mut out, e.b, names, use_names);
                out.push_str(";\n");
            }
            (EndpointWire::Arrow, EndpointWire::Arrow) => {
                push_label(&mut out, e.a, names, use_names);
                out.push_str(" -> ");
                push_label(&mut out, e.b, names, use_names);
                out.push_str(" [dir=both];\n");
            }
            (at_a, at_b) => {
                push_label(&mut out, e.a, names, use_names);
                out.push_str(" -> ");
                push_label(&mut out, e.b, names, use_names);
                out.push_str(&format!(
                    " [mark_a={}, mark_b={}];\n",
                    endpoint_str(at_a),
                    endpoint_str(at_b)
                ));
            }
        }
    }
    for (i, present) in seen.iter().enumerate() {
        if !*present {
            out.push(' ');
            push_label(
                &mut out,
                u32::try_from(i).map_err(|_| IoError::TooLarge)?,
                names,
                use_names,
            );
            out.push_str(";\n");
        }
    }
    out.push('}');
    Ok(out)
}

/// Parse DOT into an [`Admg`] (`->` directed; `dir=both` bidirected).
///
/// # Errors
///
/// Malformed DOT or illegal ADMG structure.
pub fn admg_from_dot(dot: &str) -> Result<Admg, IoError> {
    admg_with_names_from_dot(dot).map(|(g, _names)| g)
}

/// Parse DOT into an [`Admg`] (`->` directed; `dir=both` bidirected) plus
/// its node labels, in dense-id order.
///
/// # Errors
///
/// Malformed DOT or illegal ADMG structure.
pub fn admg_with_names_from_dot(dot: &str) -> Result<(Admg, Vec<String>), IoError> {
    let parsed = parse_dot(dot, false)?;
    let mut directed = Vec::new();
    let mut bidirected = Vec::new();
    for e in parsed.edges {
        match e.kind {
            DotKind::Directed => directed.push((e.from, e.to)),
            DotKind::Bidirected
            | DotKind::Marked { at_a: EndpointWire::Arrow, at_b: EndpointWire::Arrow } => {
                bidirected.push(canon(e.from, e.to));
            }
            _ => {
                return Err(IoError::Convert(
                    "ADMG DOT accepts only -> and dir=both / arrow-arrow marks".into(),
                ));
            }
        }
    }
    let g = admg_from_wire(&AdmgWire { node_count: parsed.node_count, directed, bidirected })?;
    Ok((g, parsed.names))
}

/// Serialize an [`Admg`] to DOT.
///
/// # Errors
///
/// Wire conversion failures.
pub fn admg_to_dot(admg: &Admg, names: Option<&[String]>) -> Result<String, IoError> {
    let wire = admg_to_wire(admg)?;
    let use_names = names.is_some_and(|n| n.len() == wire.node_count as usize);
    let mut out = String::from("digraph {\n");
    for &(a, b) in &wire.directed {
        emit_simple_edge(&mut out, a, b, "->", names, use_names);
    }
    for &(a, b) in &wire.bidirected {
        out.push(' ');
        push_label(&mut out, a, names, use_names);
        out.push_str(" -> ");
        push_label(&mut out, b, names, use_names);
        out.push_str(" [dir=both];\n");
    }
    emit_isolated(&mut out, wire.node_count, &wire.directed, &wire.bidirected, names, use_names);
    out.push('}');
    Ok(out)
}

#[derive(Clone, Debug)]
enum DotKind {
    Directed,
    Undirected,
    Bidirected,
    Marked { at_a: EndpointWire, at_b: EndpointWire },
}

#[derive(Clone, Debug)]
struct DotEdge {
    from: u32,
    to: u32,
    kind: DotKind,
}

struct DotGraph {
    node_count: u32,
    edges: Vec<DotEdge>,
    /// Node labels in dense-id order.
    names: Vec<String>,
}

fn parse_dot(dot: &str, allow_undirected: bool) -> Result<DotGraph, IoError> {
    let mut lexer = Lexer::new(dot);
    lexer.skip_ws_and_comments();
    let kw = lexer.expect_ident()?.to_ascii_lowercase();
    if kw != "digraph" && kw != "graph" {
        return Err(IoError::Convert(format!("expected digraph/graph, found `{kw}`")));
    }
    lexer.skip_ws_and_comments();
    if lexer.peek_ident().is_some() {
        let _ = lexer.expect_ident()?;
    }
    lexer.expect_char('{')?;
    let mut order = Vec::new();
    let mut index = HashMap::new();
    let mut edges = Vec::new();
    loop {
        lexer.skip_ws_and_comments();
        if lexer.eat_char('}') {
            break;
        }
        if lexer.eof() {
            return Err(IoError::Convert("unexpected end of DOT input".into()));
        }
        if lexer.eat_char(';') {
            continue;
        }
        let from = lexer.expect_node_id()?;
        lexer.skip_ws_and_comments();
        if lexer.peek_char() == Some('[') {
            let _ = lexer.parse_attr_list()?;
            graph_dot::intern(&from, &mut order, &mut index)?;
            let _ = lexer.eat_char(';');
            continue;
        }
        if lexer.eat_char(';') {
            graph_dot::intern(&from, &mut order, &mut index)?;
            continue;
        }
        let directed = lexer.eat_str("->");
        let undirected = !directed && lexer.eat_str("--");
        if !directed && !undirected {
            return Err(IoError::Convert(format!(
                "expected edge or node statement near `{}`",
                lexer.snippet()
            )));
        }
        if undirected && !allow_undirected {
            return Err(IoError::Convert("undirected edges (--) are not supported".into()));
        }
        lexer.skip_ws_and_comments();
        let to = lexer.expect_node_id()?;
        lexer.skip_ws_and_comments();
        let attrs =
            if lexer.peek_char() == Some('[') { lexer.parse_attr_list()? } else { HashMap::new() };
        let _ = lexer.eat_char(';');
        let fi = graph_dot::intern(&from, &mut order, &mut index)?;
        let ti = graph_dot::intern(&to, &mut order, &mut index)?;
        edges.push(DotEdge { from: fi, to: ti, kind: classify_edge(directed, &attrs)? });
    }
    if order.is_empty() {
        return Err(IoError::Convert("DOT graph has no nodes".into()));
    }
    let pairs: Vec<(u32, u32)> = edges.iter().map(|e| (e.from, e.to)).collect();
    if let Some(remapped) = graph_dot::remap_numeric_dense(&order, &pairs)? {
        let map: HashMap<u32, u32> = order
            .iter()
            .enumerate()
            .filter_map(|(i, label)| {
                let i_u32 = u32::try_from(i).ok()?;
                label.parse::<u32>().ok().map(|n| (i_u32, n))
            })
            .collect();
        let edges = edges
            .into_iter()
            .map(|e| DotEdge {
                from: *map.get(&e.from).unwrap_or(&e.from),
                to: *map.get(&e.to).unwrap_or(&e.to),
                kind: e.kind,
            })
            .collect();
        // Dense id == numeric label in this case, so the label carries no
        // information beyond the dense index; fall back to index strings.
        let names = (0..remapped.node_count).map(|i| i.to_string()).collect();
        return Ok(DotGraph { node_count: remapped.node_count, edges, names });
    }
    let node_count = u32::try_from(order.len()).map_err(|_| IoError::TooLarge)?;
    Ok(DotGraph { node_count, edges, names: order })
}

fn classify_edge(directed: bool, attrs: &HashMap<String, String>) -> Result<DotKind, IoError> {
    if let (Some(a), Some(b)) = (attrs.get("mark_a"), attrs.get("mark_b")) {
        return Ok(DotKind::Marked { at_a: parse_endpoint(a)?, at_b: parse_endpoint(b)? });
    }
    if attrs.get("dir").is_some_and(|d| d.eq_ignore_ascii_case("both")) {
        return Ok(DotKind::Bidirected);
    }
    Ok(if directed { DotKind::Directed } else { DotKind::Undirected })
}

// ── GML ─────────────────────────────────────────────────────────────────────

/// Parse GML into a [`Cpdag`] (edge attr `undirected 1`).
///
/// # Errors
///
/// Malformed GML.
pub fn cpdag_from_gml(gml: &str) -> Result<Cpdag, IoError> {
    cpdag_with_names_from_gml(gml).map(|(g, _names)| g)
}

/// Parse GML into a [`Cpdag`] (edge attr `undirected 1`) plus its node
/// labels, in dense-id order.
///
/// # Errors
///
/// Malformed GML.
pub fn cpdag_with_names_from_gml(gml: &str) -> Result<(Cpdag, Vec<String>), IoError> {
    let g = parse_gml(gml)?;
    let cpdag = cpdag_from_wire(&CpdagWire {
        node_count: g.node_count,
        directed: g.directed,
        undirected: g.undirected,
    })?;
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

/// Parse GML into a [`Pag`] (edge attrs `mark_a` / `mark_b`).
///
/// # Errors
///
/// Malformed GML.
pub fn pag_from_gml(gml: &str) -> Result<Pag, IoError> {
    pag_with_names_from_gml(gml).map(|(g, _names)| g)
}

/// Parse GML into a [`Pag`] (edge attrs `mark_a` / `mark_b`) plus its node
/// labels, in dense-id order.
///
/// # Errors
///
/// Malformed GML.
pub fn pag_with_names_from_gml(gml: &str) -> Result<(Pag, Vec<String>), IoError> {
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
    let pag = pag_from_wire(&PagWire { node_count: g.node_count, edges })?;
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

/// Parse GML into an [`Admg`] (edge attr `bidirected 1`).
///
/// # Errors
///
/// Malformed GML.
pub fn admg_from_gml(gml: &str) -> Result<Admg, IoError> {
    admg_with_names_from_gml(gml).map(|(g, _names)| g)
}

/// Parse GML into an [`Admg`] (edge attr `bidirected 1`) plus its node
/// labels, in dense-id order.
///
/// # Errors
///
/// Malformed GML.
pub fn admg_with_names_from_gml(gml: &str) -> Result<(Admg, Vec<String>), IoError> {
    let g = parse_gml(gml)?;
    let admg = admg_from_wire(&AdmgWire {
        node_count: g.node_count,
        directed: g.directed,
        bidirected: g.bidirected,
    })?;
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

struct GmlParseState {
    order: Vec<String>,
    index: HashMap<String, u32>,
    directed: Vec<(u32, u32)>,
    undirected: Vec<(u32, u32)>,
    bidirected: Vec<(u32, u32)>,
    marked: Vec<MarkedEdgeWire>,
}

impl GmlParseState {
    fn into_graph(self) -> Result<GmlGraph, IoError> {
        Ok(GmlGraph {
            node_count: u32::try_from(self.order.len()).map_err(|_| IoError::TooLarge)?,
            directed: self.directed,
            undirected: self.undirected,
            bidirected: self.bidirected,
            marked: self.marked,
            names: self.order,
        })
    }
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

fn parse_gml_node(state: &mut GmlParseState, tokens: &[Tok], i: &mut usize) -> Result<(), IoError> {
    *i += 1;
    graph_gml::expect_char(tokens, i, '[')?;
    let mut id = None;
    let mut label = None;
    while *i < tokens.len() && !matches!(&tokens[*i], Tok::Char(']')) {
        let key = graph_gml::expect_any_ident(tokens, i)?.to_ascii_lowercase();
        let val = graph_gml::expect_value(tokens, i)?;
        match key.as_str() {
            "id" => id = Some(val),
            "label" => label = Some(val),
            _ => {}
        }
    }
    graph_gml::expect_char(tokens, i, ']')?;
    let name = label.or(id).ok_or_else(|| IoError::Convert("node missing id".into()))?;
    graph_dot::intern(&name, &mut state.order, &mut state.index)?;
    Ok(())
}

fn parse_gml_edge(state: &mut GmlParseState, tokens: &[Tok], i: &mut usize) -> Result<(), IoError> {
    *i += 1;
    graph_gml::expect_char(tokens, i, '[')?;
    let mut source = None;
    let mut target = None;
    let mut is_undirected = false;
    let mut is_bidirected = false;
    let mut mark_a = None;
    let mut mark_b = None;
    while *i < tokens.len() && !matches!(&tokens[*i], Tok::Char(']')) {
        let key = graph_gml::expect_any_ident(tokens, i)?.to_ascii_lowercase();
        let val = graph_gml::expect_value(tokens, i)?;
        match key.as_str() {
            "source" => source = Some(val),
            "target" => target = Some(val),
            "undirected" => is_undirected = val != "0",
            "bidirected" => is_bidirected = val != "0",
            "mark_a" => mark_a = Some(val),
            "mark_b" => mark_b = Some(val),
            _ => {}
        }
    }
    graph_gml::expect_char(tokens, i, ']')?;
    let s = source.ok_or_else(|| IoError::Convert("edge missing source".into()))?;
    let t = target.ok_or_else(|| IoError::Convert("edge missing target".into()))?;
    let from = graph_dot::intern(&s, &mut state.order, &mut state.index)?;
    let to = graph_dot::intern(&t, &mut state.order, &mut state.index)?;
    if let (Some(a), Some(b)) = (mark_a, mark_b) {
        state.marked.push(MarkedEdgeWire {
            a: from,
            b: to,
            at_a: parse_endpoint(&a)?,
            at_b: parse_endpoint(&b)?,
        });
    } else if is_bidirected {
        state.bidirected.push(canon(from, to));
    } else if is_undirected {
        state.undirected.push(canon(from, to));
    } else {
        state.directed.push((from, to));
    }
    Ok(())
}

fn parse_gml(gml: &str) -> Result<GmlGraph, IoError> {
    let tokens = graph_gml::tokenize(gml)?;
    let mut i = 0;
    graph_gml::expect_ident(&tokens, &mut i, "graph")?;
    graph_gml::expect_char(&tokens, &mut i, '[')?;
    let mut directed_flag = None;
    let mut state = GmlParseState {
        order: Vec::new(),
        index: HashMap::new(),
        directed: Vec::new(),
        undirected: Vec::new(),
        bidirected: Vec::new(),
        marked: Vec::new(),
    };
    while i < tokens.len() {
        if matches!(&tokens[i], Tok::Char(']')) {
            break;
        }
        match &tokens[i] {
            Tok::Ident(k) if k.eq_ignore_ascii_case("directed") => {
                i += 1;
                directed_flag = Some(graph_gml::expect_number(&tokens, &mut i)? != 0.0);
            }
            Tok::Ident(k) if k.eq_ignore_ascii_case("node") => {
                parse_gml_node(&mut state, &tokens, &mut i)?;
            }
            Tok::Ident(k) if k.eq_ignore_ascii_case("edge") => {
                parse_gml_edge(&mut state, &tokens, &mut i)?;
            }
            Tok::Ident(_) => {
                i += 1;
                if i < tokens.len() {
                    let _ = graph_gml::expect_value(&tokens, &mut i);
                }
            }
            other => return Err(IoError::Convert(format!("unexpected GML token {other:?}"))),
        }
    }
    if directed_flag != Some(true) {
        return Err(IoError::Convert("GML graph must be directed 1".into()));
    }
    if state.order.is_empty() {
        return Err(IoError::Convert("empty GML graph".into()));
    }
    state.into_graph()
}

fn emit_gml(
    node_count: u32,
    names: Option<&[String]>,
    edges_a: impl Iterator<Item = GmlEdge>,
    edges_b: impl Iterator<Item = GmlEdge>,
) -> String {
    let mut out = String::from("graph [\n  directed 1\n");
    for i in 0..node_count {
        let label = names.and_then(|n| n.get(i as usize)).cloned().unwrap_or_else(|| i.to_string());
        out.push_str(&format!("  node [\n    id \"{label}\"\n    label \"{label}\"\n  ]\n"));
    }
    for e in edges_a.chain(edges_b) {
        let (a, b, extra) = match e {
            GmlEdge::Dir(a, b) => (a, b, String::new()),
            GmlEdge::Undir(a, b) => (a, b, "    undirected \"1\"\n".into()),
            GmlEdge::Bi(a, b) => (a, b, "    bidirected \"1\"\n".into()),
            GmlEdge::Mark(a, b, at_a, at_b) => (
                a,
                b,
                format!(
                    "    mark_a \"{}\"\n    mark_b \"{}\"\n",
                    endpoint_str(at_a),
                    endpoint_str(at_b)
                ),
            ),
        };
        let sa = names.and_then(|n| n.get(a as usize)).cloned().unwrap_or_else(|| a.to_string());
        let sb = names.and_then(|n| n.get(b as usize)).cloned().unwrap_or_else(|| b.to_string());
        out.push_str(&format!("  edge [\n    source \"{sa}\"\n    target \"{sb}\"\n{extra}  ]\n"));
    }
    out.push(']');
    out
}

// ── NetworkX node-link ──────────────────────────────────────────────────────

/// Parse `NetworkX` node-link JSON into a [`Cpdag`].
///
/// # Errors
///
/// Malformed JSON.
pub fn cpdag_from_networkx_node_link(json: &str) -> Result<Cpdag, IoError> {
    cpdag_with_names_from_networkx_node_link(json).map(|(g, _names)| g)
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
    let g = cpdag_from_wire(&CpdagWire { node_count, directed, undirected })?;
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

/// Parse `NetworkX` node-link JSON into a [`Pag`].
///
/// # Errors
///
/// Malformed JSON.
pub fn pag_from_networkx_node_link(json: &str) -> Result<Pag, IoError> {
    pag_with_names_from_networkx_node_link(json).map(|(g, _names)| g)
}

/// Parse `NetworkX` node-link JSON into a [`Pag`] plus its node names (the
/// document's node `id` values, stringified, in dense-id order).
///
/// # Errors
///
/// Malformed JSON.
pub fn pag_with_names_from_networkx_node_link(json: &str) -> Result<(Pag, Vec<String>), IoError> {
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
    let g = pag_from_wire(&PagWire { node_count, edges })?;
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

/// Parse `NetworkX` node-link JSON into an [`Admg`].
///
/// # Errors
///
/// Malformed JSON.
pub fn admg_from_networkx_node_link(json: &str) -> Result<Admg, IoError> {
    admg_with_names_from_networkx_node_link(json).map(|(g, _names)| g)
}

/// Parse `NetworkX` node-link JSON into an [`Admg`] plus its node names
/// (the document's node `id` values, stringified, in dense-id order).
///
/// # Errors
///
/// Malformed JSON.
pub fn admg_with_names_from_networkx_node_link(json: &str) -> Result<(Admg, Vec<String>), IoError> {
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
    let g = admg_from_wire(&AdmgWire { node_count, directed, bidirected })?;
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
    if !doc.directed {
        return Err(IoError::Convert("NetworkX graph must be directed".into()));
    }
    if doc.multigraph {
        return Err(IoError::Convert("NetworkX multigraph not supported".into()));
    }
    let mut order = Vec::new();
    let mut index = HashMap::new();
    for n in &doc.nodes {
        let name = json_id_to_string(&n.id)?;
        graph_dot::intern(&name, &mut order, &mut index)?;
    }
    let mut links = Vec::new();
    for link in &doc.links {
        let s = json_id_to_string(&link.source)?;
        let t = json_id_to_string(&link.target)?;
        let from = graph_dot::intern(&s, &mut order, &mut index)?;
        let to = graph_dot::intern(&t, &mut order, &mut index)?;
        links.push(NxLink {
            from,
            to,
            undirected: link.undirected,
            bidirected: link.bidirected,
            dir: link.dir.clone(),
            mark_a: link.mark_a.clone(),
            mark_b: link.mark_b.clone(),
        });
    }
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

// ── helpers ─────────────────────────────────────────────────────────────────

fn canon(a: u32, b: u32) -> (u32, u32) {
    if a <= b { (a, b) } else { (b, a) }
}

fn parse_endpoint(s: &str) -> Result<EndpointWire, IoError> {
    match s.to_ascii_lowercase().as_str() {
        "tail" => Ok(EndpointWire::Tail),
        "arrow" => Ok(EndpointWire::Arrow),
        "circle" => Ok(EndpointWire::Circle),
        "conflict" => Ok(EndpointWire::Conflict),
        other => Err(IoError::Convert(format!(
            "unknown endpoint mark `{other}` (expected tail|arrow|circle|conflict)"
        ))),
    }
}

fn endpoint_str(e: EndpointWire) -> &'static str {
    match e {
        EndpointWire::Tail => "tail",
        EndpointWire::Arrow => "arrow",
        EndpointWire::Circle => "circle",
        EndpointWire::Conflict => "conflict",
    }
}

fn push_label(out: &mut String, id: u32, names: Option<&[String]>, use_names: bool) {
    if use_names {
        graph_dot::push_quoted(out, &names.expect("checked")[id as usize]);
    } else {
        out.push_str(&id.to_string());
    }
}

fn emit_simple_edge(
    out: &mut String,
    a: u32,
    b: u32,
    op: &str,
    names: Option<&[String]>,
    use_names: bool,
) {
    out.push(' ');
    push_label(out, a, names, use_names);
    out.push(' ');
    out.push_str(op);
    out.push(' ');
    push_label(out, b, names, use_names);
    out.push_str(";\n");
}

fn emit_isolated(
    out: &mut String,
    node_count: u32,
    a: &[(u32, u32)],
    b: &[(u32, u32)],
    names: Option<&[String]>,
    use_names: bool,
) {
    let mut seen = vec![false; node_count as usize];
    for &(x, y) in a.iter().chain(b.iter()) {
        if (x as usize) < seen.len() {
            seen[x as usize] = true;
        }
        if (y as usize) < seen.len() {
            seen[y as usize] = true;
        }
    }
    for (i, present) in seen.iter().enumerate() {
        if !*present {
            if let Ok(id) = u32::try_from(i) {
                out.push(' ');
                push_label(out, id, names, use_names);
                out.push_str(";\n");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use antecedent_graph::{DenseNodeId, Endpoint, MarkedEdge, MiddleMark};

    use super::*;

    #[test]
    fn cpdag_json_dot_gml_networkx_round_trip() {
        let mut g = Cpdag::with_variables(3);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        g.insert_undirected(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
        let names = vec!["a".into(), "b".into(), "c".into()];
        assert_eq!(
            cpdag_from_json(&cpdag_to_json(&g, Some(&names)).unwrap()).unwrap().edges().len(),
            2
        );
        assert_eq!(
            cpdag_from_dot(&cpdag_to_dot(&g, Some(&names)).unwrap()).unwrap().node_count(),
            3
        );
        assert_eq!(
            cpdag_from_gml(&cpdag_to_gml(&g, Some(&names)).unwrap()).unwrap().node_count(),
            3
        );
        assert_eq!(
            cpdag_from_networkx_node_link(&cpdag_to_networkx_node_link(&g, Some(&names)).unwrap())
                .unwrap()
                .node_count(),
            3
        );
    }

    #[test]
    fn pag_json_dot_round_trip() {
        let mut g = Pag::with_variables(2);
        g.insert_marked(MarkedEdge {
            a: DenseNodeId::from_raw(0),
            b: DenseNodeId::from_raw(1),
            at_a: Endpoint::Circle,
            at_b: Endpoint::Arrow,
            middle: MiddleMark::Empty,
        })
        .unwrap();
        let back = pag_from_json(&pag_to_json(&g, None).unwrap()).unwrap();
        assert!(back.has_edge(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)));
        let e = pag_from_dot(&pag_to_dot(&g, None).unwrap())
            .unwrap()
            .edge_between(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1))
            .unwrap();
        assert_eq!((e.at_a, e.at_b), (Endpoint::Circle, Endpoint::Arrow));
    }

    #[test]
    fn admg_json_networkx_gml_round_trip() {
        let mut g = Admg::with_variables(3);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        g.insert_bidirected(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
        assert_eq!(
            admg_from_json(&admg_to_json(&g, None).unwrap())
                .unwrap()
                .bidirected_neighbors(DenseNodeId::from_raw(1))
                .len(),
            1
        );
        assert_eq!(
            admg_from_networkx_node_link(&admg_to_networkx_node_link(&g, None).unwrap())
                .unwrap()
                .node_count(),
            3
        );
        assert_eq!(admg_from_gml(&admg_to_gml(&g, None).unwrap()).unwrap().node_count(), 3);
        assert_eq!(admg_from_dot(&admg_to_dot(&g, None).unwrap()).unwrap().node_count(), 3);
    }

    #[test]
    fn cpdag_with_names_round_trip_preserves_labels_all_codecs() {
        let mut g = Cpdag::with_variables(3);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        g.insert_undirected(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
        let names = vec!["a".to_string(), "b".to_string(), "c".to_string()];

        let (_g, back) =
            cpdag_with_names_from_json(&cpdag_to_json(&g, Some(&names)).unwrap()).unwrap();
        assert_eq!(back, names);
        let (_g, back) =
            cpdag_with_names_from_dot(&cpdag_to_dot(&g, Some(&names)).unwrap()).unwrap();
        assert_eq!(back, names);
        let (_g, back) =
            cpdag_with_names_from_gml(&cpdag_to_gml(&g, Some(&names)).unwrap()).unwrap();
        assert_eq!(back, names);
        let (_g, back) = cpdag_with_names_from_networkx_node_link(
            &cpdag_to_networkx_node_link(&g, Some(&names)).unwrap(),
        )
        .unwrap();
        assert_eq!(back, names);
    }

    #[test]
    fn pag_with_names_round_trip_preserves_labels_all_codecs() {
        let mut g = Pag::with_variables(2);
        g.insert_marked(MarkedEdge {
            a: DenseNodeId::from_raw(0),
            b: DenseNodeId::from_raw(1),
            at_a: Endpoint::Circle,
            at_b: Endpoint::Arrow,
            middle: MiddleMark::Empty,
        })
        .unwrap();
        let names = vec!["x".to_string(), "y".to_string()];

        let (_g, back) = pag_with_names_from_json(&pag_to_json(&g, Some(&names)).unwrap()).unwrap();
        assert_eq!(back, names);
        let (_g, back) = pag_with_names_from_dot(&pag_to_dot(&g, Some(&names)).unwrap()).unwrap();
        assert_eq!(back, names);
        let (_g, back) = pag_with_names_from_gml(&pag_to_gml(&g, Some(&names)).unwrap()).unwrap();
        assert_eq!(back, names);
        let (_g, back) = pag_with_names_from_networkx_node_link(
            &pag_to_networkx_node_link(&g, Some(&names)).unwrap(),
        )
        .unwrap();
        assert_eq!(back, names);
    }

    #[test]
    fn admg_with_names_round_trip_preserves_labels_all_codecs() {
        let mut g = Admg::with_variables(3);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        g.insert_bidirected(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
        let names = vec!["z".to_string(), "t".to_string(), "y".to_string()];

        let (_g, back) =
            admg_with_names_from_json(&admg_to_json(&g, Some(&names)).unwrap()).unwrap();
        assert_eq!(back, names);
        let (_g, back) = admg_with_names_from_dot(&admg_to_dot(&g, Some(&names)).unwrap()).unwrap();
        assert_eq!(back, names);
        let (_g, back) = admg_with_names_from_gml(&admg_to_gml(&g, Some(&names)).unwrap()).unwrap();
        assert_eq!(back, names);
        let (_g, back) = admg_with_names_from_networkx_node_link(
            &admg_to_networkx_node_link(&g, Some(&names)).unwrap(),
        )
        .unwrap();
        assert_eq!(back, names);
    }

    #[test]
    fn with_names_nameless_falls_back_to_dense_index() {
        let mut g = Cpdag::with_variables(2);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let (_g, names) = cpdag_with_names_from_json(&cpdag_to_json(&g, None).unwrap()).unwrap();
        assert_eq!(names, vec!["0".to_string(), "1".to_string()]);
    }
}
