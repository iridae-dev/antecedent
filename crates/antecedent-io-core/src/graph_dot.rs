//! DOT digraph subset import/export for causal DAGs.
//!
//! Supports `digraph [name] { A -> B; ... }` with identifier or quoted node ids.
//! Attribute lists `[...]` are skipped. Undirected edges and subgraphs are rejected.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;
use std::fmt::Write as _;

use antecedent_graph::Dag;

use crate::convert::{GraphLimits, dag_from_wire_with_limits, dag_to_wire};
use crate::error::IoError;
use crate::graph_json::without_names;
use crate::wire::DagWire;

without_names! {
    /// Parse a DOT digraph into a [`Dag`].
    ///
    /// Node labels become dense indices in first-seen order. Numeric labels that
    /// form a contiguous `0..n` set keep those indices when every node is numeric.
    ///
    /// # Errors
    ///
    /// Malformed DOT, undirected edges, cycles, or empty graphs.
    pub fn dag_from_dot(dot: &str) -> Dag = dag_with_names_from_dot
}

/// Parse a DOT digraph into a [`Dag`] plus its node labels.
///
/// Labels are returned in dense-id order. When every label is numeric and
/// forms a contiguous `0..n` set (see [`dag_from_dot`]), dense ids follow
/// those numeric values and the returned names are the dense-index strings,
/// since the document carries no distinct name information in that case.
///
/// # Errors
///
/// Malformed DOT, undirected edges, cycles, or empty graphs.
pub fn dag_with_names_from_dot(dot: &str) -> Result<(Dag, Vec<String>), IoError> {
    dag_with_names_from_dot_with_limits(dot, &GraphLimits::default())
}

/// [`dag_with_names_from_dot`] under the caller's graph bounds: a document over `limits` is refused.
///
/// # Errors
///
/// As [`dag_with_names_from_dot`], and [`IoError::Convert`] when a bound of `limits` is exceeded.
pub fn dag_with_names_from_dot_with_limits(
    dot: &str,
    limits: &GraphLimits,
) -> Result<(Dag, Vec<String>), IoError> {
    let (wire, names) = dag_wire_and_names_from_dot(dot)?;
    let dag = dag_from_wire_with_limits(&wire, limits)?;
    Ok((dag, names))
}

/// Serialize a [`Dag`] to a DOT digraph string.
///
/// When `names` is `Some` and length matches `node_count`, those labels are
/// used (quoted). Otherwise nodes are emitted as dense integer ids.
///
/// # Errors
///
/// Wire conversion failures (non-static / non-directed edges).
pub fn dag_to_dot(dag: &Dag, names: Option<&[String]>) -> Result<String, IoError> {
    let wire = dag_to_wire(dag)?;
    dag_wire_to_dot(&wire, names)
}

/// Parse DOT into [`DagWire`] (shared with JSON path for tests).
pub fn dag_wire_from_dot(dot: &str) -> Result<DagWire, IoError> {
    dag_wire_and_names_from_dot(dot).map(|(wire, _names)| wire)
}

/// Graphviz `node [..]`, `edge [..]` and `graph [..]` set defaults for later statements;
/// they name no variable. A quoted `"node"` is an ordinary node id.
pub(crate) fn is_default_attribute_statement(id: &str, quoted: bool, next: Option<char>) -> bool {
    !quoted
        && next == Some('[')
        && matches!(id.to_ascii_lowercase().as_str(), "node" | "edge" | "graph")
}

/// Parse DOT into [`DagWire`] plus node labels in dense-id order.
fn dag_wire_and_names_from_dot(dot: &str) -> Result<(DagWire, Vec<String>), IoError> {
    let parsed = parse_dot_graph(
        dot,
        DotParseOptions {
            allow_graph_keyword: false,
            allow_undirected: false,
            empty_message: "DOT digraph has no nodes",
        },
    )?;
    let mut edges = Vec::with_capacity(parsed.edges.len());
    for edge in &parsed.edges {
        refuse_unrepresentable_dag_edge_attrs(&edge.attrs)?;
        edges.push((edge.from, edge.to));
    }
    finish_dag_wire(&parsed.names, edges, parsed.node_count)
}

fn finish_dag_wire(
    names: &[String],
    edges: Vec<(u32, u32)>,
    node_count: u32,
) -> Result<(DagWire, Vec<String>), IoError> {
    match remap_numeric_dense(names, &edges)? {
        Some(wire) => {
            let names = (0..wire.node_count).map(|i| i.to_string()).collect();
            Ok((wire, names))
        }
        None => Ok((DagWire { node_count, edges }, names.to_vec())),
    }
}

/// How a class reads one DOT document. The parser records edges; the class
/// decides which marks it can represent.
#[derive(Clone, Copy)]
pub(crate) struct DotParseOptions {
    /// `graph` as well as `digraph`. A DAG document rejects `graph`.
    pub allow_graph_keyword: bool,
    /// `--` edges. A DAG document rejects them.
    pub allow_undirected: bool,
    /// Error when the document names no node.
    pub empty_message: &'static str,
}

/// One edge before a graph class interprets its marks.
pub(crate) struct ParsedDotEdge {
    pub from: u32,
    pub to: u32,
    /// `->` rather than `--`.
    pub directed: bool,
    pub attrs: HashMap<String, String>,
}

/// Nodes in first-seen order and edges interned in that same order.
pub(crate) struct ParsedDot {
    pub node_count: u32,
    pub names: Vec<String>,
    pub edges: Vec<ParsedDotEdge>,
}

/// Parse DOT into a class-neutral document.
///
/// Endpoint order is first-seen, before any class swaps an edge to point a
/// single arrowhead at `to`. Attribute meaning is left to the caller.
pub(crate) fn parse_dot_graph(dot: &str, options: DotParseOptions) -> Result<ParsedDot, IoError> {
    let mut lexer = Lexer::new(dot);
    lexer.skip_ws_and_comments();
    let kw = lexer.expect_ident()?.to_ascii_lowercase();
    let digraph = kw == "digraph";
    if !(digraph || options.allow_graph_keyword && kw == "graph") {
        let expected = if options.allow_graph_keyword { "digraph/graph" } else { "digraph" };
        return Err(IoError::Convert(format!("expected {expected}, found `{kw}`")));
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
        let (from, quoted) = lexer.expect_statement_id()?;
        lexer.skip_ws_and_comments();
        if is_default_attribute_statement(&from, quoted, lexer.peek_char()) {
            lexer.skip_attr_list()?;
            let _ = lexer.eat_char(';');
            continue;
        }
        if lexer.peek_char() == Some('[') {
            lexer.skip_attr_list()?;
            lexer.skip_ws_and_comments();
            intern(&from, &mut order, &mut index)?;
            let _ = lexer.eat_char(';');
            continue;
        }
        if lexer.eat_char(';') {
            intern(&from, &mut order, &mut index)?;
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
        if undirected && !options.allow_undirected {
            return Err(IoError::Convert("undirected edges (--) are not supported".into()));
        }
        lexer.skip_ws_and_comments();
        let to = lexer.expect_node_id()?;
        lexer.skip_ws_and_comments();
        let attrs =
            if lexer.peek_char() == Some('[') { lexer.parse_attr_list()? } else { HashMap::new() };
        lexer.skip_ws_and_comments();
        let _ = lexer.eat_char(';');
        let from_id = intern(&from, &mut order, &mut index)?;
        let to_id = intern(&to, &mut order, &mut index)?;
        edges.push(ParsedDotEdge { from: from_id, to: to_id, directed, attrs });
    }
    if order.is_empty() {
        return Err(IoError::Convert(options.empty_message.into()));
    }
    let node_count = u32::try_from(order.len()).map_err(|_| IoError::TooLarge)?;
    Ok(ParsedDot { node_count, names: order, edges })
}

/// Emit DOT from [`DagWire`].
///
/// When `names` is `Some` and its length equals `node_count`, those labels are emitted
/// (quoted); otherwise nodes are emitted as dense integer ids. Isolated nodes always appear
/// so a re-import keeps `node_count` and the ids.
///
/// # Errors
///
/// An edge endpoint at or beyond `node_count`.
pub fn dag_wire_to_dot(wire: &DagWire, names: Option<&[String]>) -> Result<String, IoError> {
    let node_count = wire.node_count as usize;
    if let Some(&(from, to)) = wire
        .edges
        .iter()
        .find(|&&(from, to)| from as usize >= node_count || to as usize >= node_count)
    {
        return Err(IoError::Convert(format!(
            "edge ({from}, {to}) names a node outside node_count {node_count}"
        )));
    }
    let names = names.filter(|names| names.len() == node_count);
    let mut out = String::from("digraph {\n");
    for &(from, to) in &wire.edges {
        if let Some(names) = names {
            out.push(' ');
            push_quoted(&mut out, &names[from as usize]);
            out.push_str(" -> ");
            push_quoted(&mut out, &names[to as usize]);
            out.push_str(";\n");
        } else {
            let _ = writeln!(out, " {from} -> {to};");
        }
    }
    let mut seen = vec![false; node_count];
    for &(from, to) in &wire.edges {
        seen[from as usize] = true;
        seen[to as usize] = true;
    }
    for (i, present) in seen.iter().enumerate() {
        if *present {
            continue;
        }
        if let Some(names) = names {
            out.push(' ');
            push_quoted(&mut out, &names[i]);
            out.push_str(";\n");
        } else {
            let _ = writeln!(out, " {i};");
        }
    }
    out.push('}');
    Ok(out)
}

/// Refuse edge attributes that would change DAG semantics if ignored.
fn refuse_unrepresentable_dag_edge_attrs(attrs: &HashMap<String, String>) -> Result<(), IoError> {
    if let Some(dir) = attrs.get("dir") {
        if !dir.eq_ignore_ascii_case("forward") {
            return Err(IoError::Convert(format!(
                "DAG DOT cannot represent edge attribute dir={dir}; use admg_from_dot or pag_from_dot"
            )));
        }
    }
    for (key, trivial) in [("arrowhead", "normal"), ("arrowtail", "none")] {
        if let Some(shape) = attrs.get(key) {
            if !shape.eq_ignore_ascii_case(trivial) {
                return Err(IoError::Convert(format!(
                    "DAG DOT cannot represent edge attribute {key}={shape}; use pag_from_dot"
                )));
            }
        }
    }
    if attrs.contains_key("mark_a") || attrs.contains_key("mark_b") {
        return Err(IoError::Convert(
            "DAG DOT cannot represent mark_a/mark_b edge attributes; use pag_from_dot".into(),
        ));
    }
    Ok(())
}

pub(crate) fn push_quoted(out: &mut String, s: &str) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' | '\\' => {
                out.push('\\');
                out.push(ch);
            }
            _ => out.push(ch),
        }
    }
    out.push('"');
}

pub(crate) fn intern(
    label: &str,
    order: &mut Vec<String>,
    index: &mut HashMap<String, u32>,
) -> Result<u32, IoError> {
    if let Some(&i) = index.get(label) {
        return Ok(i);
    }
    let i = u32::try_from(order.len()).map_err(|_| IoError::TooLarge)?;
    order.push(label.to_string());
    index.insert(label.to_string(), i);
    Ok(i)
}

pub(crate) fn remap_numeric_dense(
    order: &[String],
    edges: &[(u32, u32)],
) -> Result<Option<DagWire>, IoError> {
    let mut nums = Vec::with_capacity(order.len());
    for label in order {
        let Ok(n) = label.parse::<u32>() else {
            return Ok(None);
        };
        nums.push(n);
    }
    let n = u32::try_from(order.len()).map_err(|_| IoError::TooLarge)?;
    let mut seen = vec![false; order.len()];
    for &v in &nums {
        if v >= n {
            return Ok(None);
        }
        if seen[v as usize] {
            return Ok(None);
        }
        seen[v as usize] = true;
    }
    if seen.iter().any(|b| !*b) {
        return Ok(None);
    }
    // order[i] had numeric label nums[i]; map old dense i -> nums[i]
    let mut mapped_edges = Vec::with_capacity(edges.len());
    for &(f, t) in edges {
        mapped_edges.push((nums[f as usize], nums[t as usize]));
    }
    Ok(Some(DagWire { node_count: n, edges: mapped_edges }))
}

pub(crate) struct Lexer<'a> {
    pub(crate) src: &'a str,
    pub(crate) pos: usize,
}

impl<'a> Lexer<'a> {
    pub(crate) fn new(src: &'a str) -> Self {
        Self { src, pos: 0 }
    }

    pub(crate) fn eof(&self) -> bool {
        self.pos >= self.src.len()
    }

    pub(crate) fn peek_char(&self) -> Option<char> {
        self.src[self.pos..].chars().next()
    }

    pub(crate) fn bump(&mut self) -> Option<char> {
        let ch = self.peek_char()?;
        self.pos += ch.len_utf8();
        Some(ch)
    }

    pub(crate) fn eat_char(&mut self, expected: char) -> bool {
        if self.peek_char() == Some(expected) {
            self.bump();
            true
        } else {
            false
        }
    }

    pub(crate) fn eat_str(&mut self, s: &str) -> bool {
        if self.src[self.pos..].starts_with(s) {
            self.pos += s.len();
            true
        } else {
            false
        }
    }

    pub(crate) fn skip_ws_and_comments(&mut self) {
        loop {
            while matches!(self.peek_char(), Some(c) if c.is_whitespace()) {
                self.bump();
            }
            if self.src[self.pos..].starts_with("//") {
                while let Some(c) = self.bump() {
                    if c == '\n' {
                        break;
                    }
                }
                continue;
            }
            if self.src[self.pos..].starts_with("/*") {
                self.pos += 2;
                while self.pos < self.src.len() && !self.src[self.pos..].starts_with("*/") {
                    self.pos += self.src[self.pos..].chars().next().map_or(1, char::len_utf8);
                }
                if self.src[self.pos..].starts_with("*/") {
                    self.pos += 2;
                }
                continue;
            }
            break;
        }
    }

    pub(crate) fn expect_char(&mut self, expected: char) -> Result<(), IoError> {
        self.skip_ws_and_comments();
        if self.eat_char(expected) {
            Ok(())
        } else {
            Err(IoError::Convert(format!("expected `{expected}` near `{}`", self.snippet())))
        }
    }

    pub(crate) fn peek_ident(&self) -> Option<&str> {
        let rest = &self.src[self.pos..];
        let mut chars = rest.chars();
        let first = chars.next()?;
        if !(first.is_ascii_alphabetic() || first == '_') {
            return None;
        }
        let mut len = first.len_utf8();
        for c in chars {
            if c.is_ascii_alphanumeric() || c == '_' || c == '.' {
                len += c.len_utf8();
            } else {
                break;
            }
        }
        Some(&rest[..len])
    }

    pub(crate) fn expect_ident(&mut self) -> Result<&'a str, IoError> {
        self.skip_ws_and_comments();
        let Some(id) = self.peek_ident() else {
            return Err(IoError::Convert(format!("expected identifier near `{}`", self.snippet())));
        };
        let len = id.len();
        let start = self.pos;
        self.pos += len;
        Ok(&self.src[start..start + len])
    }

    /// Start of a statement: the id plus whether it was a quoted string. A bare
    /// `node` / `edge` / `graph` opens a default-attribute statement, not a variable.
    pub(crate) fn expect_statement_id(&mut self) -> Result<(String, bool), IoError> {
        self.skip_ws_and_comments();
        let quoted = self.peek_char() == Some('"');
        Ok((self.expect_node_id()?, quoted))
    }

    pub(crate) fn expect_node_id(&mut self) -> Result<String, IoError> {
        self.skip_ws_and_comments();
        if self.eat_char('"') {
            let mut s = String::new();
            while let Some(c) = self.bump() {
                match c {
                    '"' => return Ok(s),
                    '\\' => {
                        let Some(n) = self.bump() else {
                            return Err(IoError::Convert(
                                "unterminated escape in DOT string".into(),
                            ));
                        };
                        s.push(n);
                    }
                    _ => s.push(c),
                }
            }
            return Err(IoError::Convert("unterminated quoted node id".into()));
        }
        // Number or bare identifier.
        if let Some(c) = self.peek_char() {
            if c.is_ascii_digit() || c == '-' {
                let start = self.pos;
                self.bump();
                while matches!(self.peek_char(), Some(d) if d.is_ascii_digit()) {
                    self.bump();
                }
                return Ok(self.src[start..self.pos].to_string());
            }
        }
        Ok(self.expect_ident()?.to_string())
    }

    pub(crate) fn skip_attr_list(&mut self) -> Result<(), IoError> {
        let _ = self.parse_attr_list()?;
        Ok(())
    }

    /// Parse `[key=value, ...]` into a map (values unquoted where possible).
    pub(crate) fn parse_attr_list(&mut self) -> Result<HashMap<String, String>, IoError> {
        self.expect_char('[')?;
        let mut attrs = HashMap::new();
        loop {
            self.skip_ws_and_comments();
            if self.eat_char(']') {
                break;
            }
            let key = self.expect_ident()?.to_ascii_lowercase();
            self.skip_ws_and_comments();
            self.expect_char('=')?;
            self.skip_ws_and_comments();
            let val = if self.peek_char() == Some('"') {
                self.expect_node_id()?
            } else if let Some(id) = self.peek_ident() {
                let s = id.to_string();
                self.pos += id.len();
                s
            } else if matches!(self.peek_char(), Some(c) if c.is_ascii_digit() || c == '-') {
                self.expect_node_id()?
            } else {
                return Err(IoError::Convert(format!(
                    "expected attribute value near `{}`",
                    self.snippet()
                )));
            };
            attrs.insert(key, val);
            self.skip_ws_and_comments();
            let _ = self.eat_char(',');
        }
        Ok(attrs)
    }

    pub(crate) fn snippet(&self) -> String {
        let mut end = self.pos.saturating_add(24).min(self.src.len());
        while !self.src.is_char_boundary(end) {
            end -= 1;
        }
        self.src[self.pos..end].replace('\n', "\\n")
    }
}

#[cfg(test)]
mod tests {
    use antecedent_graph::DenseNodeId;

    use super::*;

    #[test]
    fn invalid_dot_with_multibyte_error_context_returns_an_error() {
        for text in
            ["digraph { a x€€€€€€€€€€€€€€€€€€€€€€€ }", "digraph { a [x=€€€€€€€€€€€€€€€€€€€€] }"]
        {
            let result = std::panic::catch_unwind(|| dag_wire_from_dot(text));
            assert!(matches!(result, Ok(Err(IoError::Convert(_)))));
        }
        let lexer = Lexer::new("x€€€€€€€€€");
        assert_eq!(lexer.snippet(), "x€€€€€€€");
    }

    #[test]
    fn round_trip_numeric_dot() {
        let mut dag = Dag::with_variables(3);
        dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        dag.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
        let s = dag_to_dot(&dag, None).unwrap();
        let back = dag_from_dot(&s).unwrap();
        assert_eq!(back.node_count(), 3);
        assert!(back.reaches(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)));
    }

    #[test]
    fn default_attribute_statements_are_not_variables() {
        let dot = "digraph G { graph [rankdir=LR]; node [shape=box]; edge [color=red]; \
                   a -> b; }";
        let (dag, names) = dag_with_names_from_dot(dot).unwrap();
        assert_eq!(names, vec!["a".to_string(), "b".to_string()]);
        assert_eq!(dag.node_count(), 2);
        // A quoted id is a real variable even when it spells a keyword.
        let (quoted, names) = dag_with_names_from_dot("digraph { \"node\" -> b; }").unwrap();
        assert_eq!(names, vec!["node".to_string(), "b".to_string()]);
        assert_eq!(quoted.node_count(), 2);
    }

    #[test]
    fn with_names_round_trip_preserves_labels() {
        let mut dag = Dag::with_variables(2);
        dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let names = vec!["x".to_string(), "y".to_string()];
        let s = dag_to_dot(&dag, Some(&names)).unwrap();
        let (back, back_names) = dag_with_names_from_dot(&s).unwrap();
        assert_eq!(back.node_count(), 2);
        assert_eq!(back_names, names);
    }

    #[test]
    fn with_names_nameless_numeric_falls_back_to_dense_index() {
        let (dag, names) = dag_with_names_from_dot("digraph { 0 -> 1; 1 -> 2; }").unwrap();
        assert_eq!(dag.node_count(), 3);
        assert_eq!(names, vec!["0".to_string(), "1".to_string(), "2".to_string()]);
    }

    #[test]
    fn named_nodes_and_attrs() {
        let dot = r#"
        digraph G {
          "treatment" -> "outcome" [label="causes"];
          confounder -> treatment;
          confounder -> outcome;
        }
        "#;
        let dag = dag_from_dot(dot).unwrap();
        assert_eq!(dag.node_count(), 3);
    }

    #[test]
    fn rejects_undirected() {
        let err = dag_from_dot("digraph { a -- b; }").unwrap_err();
        assert!(err.to_string().contains("undirected"));
    }

    #[test]
    fn refuses_dir_both_edge_attrs() {
        let err = dag_from_dot("digraph { 0 -> 1 [dir=both]; }").unwrap_err();
        assert!(matches!(err, IoError::Convert(_)));
        assert!(err.to_string().contains("dir="));
    }

    #[test]
    fn unnamed_export_keeps_isolated_nodes() {
        let mut dag = Dag::with_variables(3);
        dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let s = dag_to_dot(&dag, None).unwrap();
        assert!(s.contains(" 2;"), "isolated node must appear in DOT: {s}");
        let back = dag_from_dot(&s).unwrap();
        assert_eq!(back.node_count(), 3);
        assert!(back.reaches(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)));
    }
}
