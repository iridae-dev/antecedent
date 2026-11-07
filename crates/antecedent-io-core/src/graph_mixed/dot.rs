//! DOT interchange for mixed causal graphs.

use super::{
    Admg, AdmgWire, Cpdag, EndpointWire, GraphCpdagWire, GraphLimits, IoError, MarkedEdgeWire, Pag,
    PagWire, admg_from_wire_with_limits, admg_to_wire, canon, cpdag_from_wire_with_limits,
    cpdag_to_wire, endpoint_str, pag_from_wire_with_limits, pag_to_wire, parse_endpoint,
    without_names,
};
use crate::graph_dot;
use std::collections::HashMap;
use std::fmt::Write as _;

// ── DOT ─────────────────────────────────────────────────────────────────────

without_names! {
    /// Parse DOT into a [`Cpdag`] (`->` directed, `--` undirected).
    ///
    /// # Errors
    ///
    /// Malformed DOT or illegal CPDAG marks.
    pub fn cpdag_from_dot(dot: &str) -> Cpdag = cpdag_with_names_from_dot
}

/// Parse DOT into a [`Cpdag`] (`->` directed, `--` undirected) plus its node
/// labels, in dense-id order.
///
/// # Errors
///
/// Malformed DOT or illegal CPDAG marks.
pub fn cpdag_with_names_from_dot(dot: &str) -> Result<(Cpdag, Vec<String>), IoError> {
    cpdag_with_names_from_dot_with_limits(dot, &GraphLimits::default())
}

/// [`cpdag_with_names_from_dot`] under the caller's graph bounds: a document over `limits` is refused.
///
/// # Errors
///
/// As [`cpdag_with_names_from_dot`], and [`IoError::Convert`] when a bound of `limits` is exceeded.
pub fn cpdag_with_names_from_dot_with_limits(
    dot: &str,
    limits: &GraphLimits,
) -> Result<(Cpdag, Vec<String>), IoError> {
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
    let g = cpdag_from_wire_with_limits(
        &GraphCpdagWire { node_count: parsed.node_count, directed, undirected },
        limits,
    )?;
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

without_names! {
/// Parse DOT into a [`Pag`].
///
/// Plain `->` is tail→arrow; `--` is undirected; `dir=both` is bidirected;
/// `dir=back` / `dir=none` and `arrowhead` / `arrowtail` shapes (`normal`,
/// `odot`, `none`) follow Graphviz; otherwise `mark_a` / `mark_b` attributes.
///
/// # Errors
///
/// Malformed DOT or illegal PAG marks.
    pub fn pag_from_dot(dot: &str) -> Pag = pag_with_names_from_dot
}

/// Parse DOT into a [`Pag`] plus its node labels, in dense-id order.
///
/// Plain `->` is tail→arrow; `--` is undirected; `dir=both` is bidirected;
/// `dir=back` / `dir=none` and `arrowhead` / `arrowtail` shapes (`normal`,
/// `odot`, `none`) follow Graphviz; otherwise `mark_a` / `mark_b` attributes.
///
/// # Errors
///
/// Malformed DOT or illegal PAG marks.
pub fn pag_with_names_from_dot(dot: &str) -> Result<(Pag, Vec<String>), IoError> {
    pag_with_names_from_dot_with_limits(dot, &GraphLimits::default())
}

/// [`pag_with_names_from_dot`] under the caller's graph bounds: a document over `limits` is refused.
///
/// # Errors
///
/// As [`pag_with_names_from_dot`], and [`IoError::Convert`] when a bound of `limits` is exceeded.
pub fn pag_with_names_from_dot_with_limits(
    dot: &str,
    limits: &GraphLimits,
) -> Result<(Pag, Vec<String>), IoError> {
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
    let g = pag_from_wire_with_limits(&PagWire { node_count: parsed.node_count, edges }, limits)?;
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
                let _ = writeln!(
                    out,
                    " [mark_a={}, mark_b={}];",
                    endpoint_str(at_a),
                    endpoint_str(at_b)
                );
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

without_names! {
/// Parse DOT into an [`Admg`] (`->` directed; `dir=both` bidirected).
///
/// # Errors
///
/// Malformed DOT or illegal ADMG structure.
    pub fn admg_from_dot(dot: &str) -> Admg = admg_with_names_from_dot
}

/// Parse DOT into an [`Admg`] (`->` directed; `dir=both` bidirected) plus
/// its node labels, in dense-id order.
///
/// # Errors
///
/// Malformed DOT or illegal ADMG structure.
pub fn admg_with_names_from_dot(dot: &str) -> Result<(Admg, Vec<String>), IoError> {
    admg_with_names_from_dot_with_limits(dot, &GraphLimits::default())
}

/// [`admg_with_names_from_dot`] under the caller's graph bounds: a document over `limits` is refused.
///
/// # Errors
///
/// As [`admg_with_names_from_dot`], and [`IoError::Convert`] when a bound of `limits` is exceeded.
pub fn admg_with_names_from_dot_with_limits(
    dot: &str,
    limits: &GraphLimits,
) -> Result<(Admg, Vec<String>), IoError> {
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
    let g = admg_from_wire_with_limits(
        &AdmgWire { node_count: parsed.node_count, directed, bidirected },
        limits,
    )?;
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
    let parsed = graph_dot::parse_dot_graph(
        dot,
        graph_dot::DotParseOptions {
            allow_graph_keyword: true,
            allow_undirected,
            empty_message: "DOT graph has no nodes",
        },
    )?;
    let mut edges = Vec::with_capacity(parsed.edges.len());
    for edge in parsed.edges {
        let (kind, swapped) = classify_edge(edge.directed, &edge.attrs)?;
        let (from, to) = if swapped { (edge.to, edge.from) } else { (edge.from, edge.to) };
        edges.push(DotEdge { from, to, kind });
    }
    let order = parsed.names;
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

/// Read the endpoint marks of one DOT edge `from <op> to [attrs]`.
///
/// Explicit `mark_a` / `mark_b` must come as a pair. Otherwise the marks
/// follow Graphviz: `dir` (`forward` default for `->`, `none` for `--`;
/// also `back` / `both`) decides which ends are drawn, and a drawn end takes
/// its `arrowhead` / `arrowtail` shape (`normal` → arrow, `odot` → circle,
/// `none` → tail). The result is `(mark at from, mark at to, swapped)`,
/// normalised so a single arrowhead always points at `to`.
fn classify_edge(
    directed: bool,
    attrs: &HashMap<String, String>,
) -> Result<(DotKind, bool), IoError> {
    let (at_a, at_b) = match (attrs.get("mark_a"), attrs.get("mark_b")) {
        (Some(a), Some(b)) => {
            if ["dir", "arrowhead", "arrowtail"].iter().any(|k| attrs.contains_key(*k)) {
                return Err(IoError::Convert(
                    "DOT edge mixes mark_a/mark_b with dir/arrowhead/arrowtail".into(),
                ));
            }
            (parse_endpoint(a)?, parse_endpoint(b)?)
        }
        (None, None) => dot_drawn_marks(directed, attrs)?,
        _ => {
            return Err(IoError::Convert(
                "DOT edge sets only one of mark_a / mark_b; both are required".into(),
            ));
        }
    };
    Ok(match (at_a, at_b) {
        (EndpointWire::Tail, EndpointWire::Arrow) => (DotKind::Directed, false),
        (EndpointWire::Arrow, EndpointWire::Tail) => (DotKind::Directed, true),
        (EndpointWire::Tail, EndpointWire::Tail) => (DotKind::Undirected, false),
        (EndpointWire::Arrow, EndpointWire::Arrow) => (DotKind::Bidirected, false),
        (at_a, at_b) => (DotKind::Marked { at_a, at_b }, false),
    })
}

fn dot_drawn_marks(
    directed: bool,
    attrs: &HashMap<String, String>,
) -> Result<(EndpointWire, EndpointWire), IoError> {
    let dir = attrs.get("dir").map_or(if directed { "forward" } else { "none" }, String::as_str);
    let (tail_drawn, head_drawn) = match dir.to_ascii_lowercase().as_str() {
        "forward" => (false, true),
        "back" => (true, false),
        "both" => (true, true),
        "none" => (false, false),
        other => return Err(IoError::Convert(format!("unsupported DOT edge dir={other}"))),
    };
    let end = |key: &str, drawn: bool| -> Result<EndpointWire, IoError> {
        match (attrs.get(key), drawn) {
            (None, drawn) => Ok(if drawn { EndpointWire::Arrow } else { EndpointWire::Tail }),
            (Some(_), false) => Err(IoError::Convert(format!(
                "DOT edge sets {key} on an end that dir={dir} does not draw"
            ))),
            (Some(shape), true) => match shape.to_ascii_lowercase().as_str() {
                "normal" => Ok(EndpointWire::Arrow),
                "odot" => Ok(EndpointWire::Circle),
                "none" => Ok(EndpointWire::Tail),
                other => Err(IoError::Convert(format!("unsupported DOT {key}={other}"))),
            },
        }
    };
    Ok((end("arrowtail", tail_drawn)?, end("arrowhead", head_drawn)?))
}

fn push_label(out: &mut String, id: u32, names: Option<&[String]>, use_names: bool) {
    match names.filter(|_| use_names).and_then(|names| names.get(id as usize)) {
        Some(name) => graph_dot::push_quoted(out, name),
        None => {
            let _ = write!(out, "{id}");
        }
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
