//! DOT digraph subset import/export for causal DAGs (DESIGN.md §24).
//!
//! Supports `digraph [name] { A -> B; ... }` with identifier or quoted node ids.
//! Attribute lists `[...]` are skipped. Undirected edges and subgraphs are rejected.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;

use causal_graph::Dag;

use crate::convert::{dag_from_wire, dag_to_wire};
use crate::error::IoError;
use crate::wire::DagWire;

/// Parse a DOT digraph into a [`Dag`].
///
/// Node labels become dense indices in first-seen order. Numeric labels that
/// form a contiguous `0..n` set keep those indices when every node is numeric.
///
/// # Errors
///
/// Malformed DOT, undirected edges, cycles, or empty graphs.
pub fn dag_from_dot(dot: &str) -> Result<Dag, IoError> {
    let wire = dag_wire_from_dot(dot)?;
    dag_from_wire(&wire)
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
    Ok(dag_wire_to_dot(&wire, names))
}

/// Parse DOT into [`DagWire`] (shared with JSON path for tests).
pub fn dag_wire_from_dot(dot: &str) -> Result<DagWire, IoError> {
    let mut lexer = Lexer::new(dot);
    lexer.skip_ws_and_comments();
    let kw = lexer.expect_ident()?.to_ascii_lowercase();
    if kw != "digraph" {
        return Err(IoError::Convert(format!("expected digraph, found `{kw}`")));
    }
    lexer.skip_ws_and_comments();
    if lexer.peek_ident().is_some() {
        let _ = lexer.expect_ident()?;
        lexer.skip_ws_and_comments();
    }
    lexer.expect_char('{')?;
    let mut order: Vec<String> = Vec::new();
    let mut index: HashMap<String, u32> = HashMap::new();
    let mut edges: Vec<(u32, u32)> = Vec::new();

    loop {
        lexer.skip_ws_and_comments();
        if lexer.eat_char('}') {
            break;
        }
        if lexer.eof() {
            return Err(IoError::Convert("unexpected end of DOT input".into()));
        }
        // Skip empty statements.
        if lexer.eat_char(';') {
            continue;
        }
        let from = lexer.expect_node_id()?;
        lexer.skip_ws_and_comments();
        // Optional attribute list after a lone node declaration.
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
        if lexer.eat_str("->") {
            lexer.skip_ws_and_comments();
            let to = lexer.expect_node_id()?;
            lexer.skip_ws_and_comments();
            if lexer.peek_char() == Some('[') {
                lexer.skip_attr_list()?;
                lexer.skip_ws_and_comments();
            }
            let _ = lexer.eat_char(';');
            let fi = intern(&from, &mut order, &mut index)?;
            let ti = intern(&to, &mut order, &mut index)?;
            edges.push((fi, ti));
            continue;
        }
        if lexer.eat_str("--") {
            return Err(IoError::Convert("undirected edges (--) are not supported".into()));
        }
        return Err(IoError::Convert(format!(
            "expected edge or node statement near `{}`",
            lexer.snippet()
        )));
    }

    if order.is_empty() {
        return Err(IoError::Convert("DOT digraph has no nodes".into()));
    }

    // Prefer numeric dense ids when every label is an integer in 0..n.
    let remapped = remap_numeric_dense(&order, &edges)?;
    Ok(remapped.unwrap_or(DagWire {
        node_count: u32::try_from(order.len()).map_err(|_| IoError::TooLarge)?,
        edges,
    }))
}

/// Emit DOT from [`DagWire`].
#[must_use]
pub fn dag_wire_to_dot(wire: &DagWire, names: Option<&[String]>) -> String {
    let mut out = String::from("digraph {\n");
    let use_names = names.is_some_and(|n| n.len() == wire.node_count as usize);
    for &(from, to) in &wire.edges {
        if use_names {
            let names = names.expect("checked");
            out.push(' ');
            push_quoted(&mut out, &names[from as usize]);
            out.push_str(" -> ");
            push_quoted(&mut out, &names[to as usize]);
            out.push_str(";\n");
        } else {
            out.push_str(&format!(" {from} -> {to};\n"));
        }
    }
    // Ensure isolated nodes appear when names are provided.
    if use_names {
        let names = names.expect("checked");
        let mut seen = vec![false; wire.node_count as usize];
        for &(f, t) in &wire.edges {
            seen[f as usize] = true;
            seen[t as usize] = true;
        }
        for (i, present) in seen.iter().enumerate() {
            if !*present {
                out.push(' ');
                push_quoted(&mut out, &names[i]);
                out.push_str(";\n");
            }
        }
    } else if wire.edges.is_empty() {
        for i in 0..wire.node_count {
            out.push_str(&format!(" {i};\n"));
        }
    }
    out.push('}');
    out
}

fn push_quoted(out: &mut String, s: &str) {
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

fn intern(
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

fn remap_numeric_dense(order: &[String], edges: &[(u32, u32)]) -> Result<Option<DagWire>, IoError> {
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

struct Lexer<'a> {
    src: &'a str,
    pos: usize,
}

impl<'a> Lexer<'a> {
    fn new(src: &'a str) -> Self {
        Self { src, pos: 0 }
    }

    fn eof(&self) -> bool {
        self.pos >= self.src.len()
    }

    fn peek_char(&self) -> Option<char> {
        self.src[self.pos..].chars().next()
    }

    fn bump(&mut self) -> Option<char> {
        let ch = self.peek_char()?;
        self.pos += ch.len_utf8();
        Some(ch)
    }

    fn eat_char(&mut self, expected: char) -> bool {
        if self.peek_char() == Some(expected) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn eat_str(&mut self, s: &str) -> bool {
        if self.src[self.pos..].starts_with(s) {
            self.pos += s.len();
            true
        } else {
            false
        }
    }

    fn skip_ws_and_comments(&mut self) {
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

    fn expect_char(&mut self, expected: char) -> Result<(), IoError> {
        self.skip_ws_and_comments();
        if self.eat_char(expected) {
            Ok(())
        } else {
            Err(IoError::Convert(format!("expected `{expected}` near `{}`", self.snippet())))
        }
    }

    fn peek_ident(&self) -> Option<&str> {
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

    fn expect_ident(&mut self) -> Result<&'a str, IoError> {
        self.skip_ws_and_comments();
        let Some(id) = self.peek_ident() else {
            return Err(IoError::Convert(format!("expected identifier near `{}`", self.snippet())));
        };
        let len = id.len();
        let start = self.pos;
        self.pos += len;
        Ok(&self.src[start..start + len])
    }

    fn expect_node_id(&mut self) -> Result<String, IoError> {
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

    fn skip_attr_list(&mut self) -> Result<(), IoError> {
        self.expect_char('[')?;
        let mut depth = 1;
        while let Some(c) = self.bump() {
            match c {
                '[' => depth += 1,
                ']' => {
                    depth -= 1;
                    if depth == 0 {
                        return Ok(());
                    }
                }
                '"' => {
                    while let Some(q) = self.bump() {
                        if q == '\\' {
                            self.bump();
                        } else if q == '"' {
                            break;
                        }
                    }
                }
                _ => {}
            }
        }
        Err(IoError::Convert("unterminated attribute list".into()))
    }

    fn snippet(&self) -> String {
        let end = (self.pos + 24).min(self.src.len());
        self.src[self.pos..end].replace('\n', "\\n")
    }
}

#[cfg(test)]
mod tests {
    use causal_graph::DenseNodeId;

    use super::*;

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
}
