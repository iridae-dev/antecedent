//! GML digraph subset import/export for causal DAGs (DESIGN.md §24).
//!
//! Supports pinned baseline-style:
//! `graph [ directed 1 node [ id "Z" label "Z" ] edge [ source "Z" target "X" ] ]`
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;

use causal_graph::Dag;

use crate::convert::{dag_from_wire, dag_to_wire};
use crate::error::IoError;
use crate::wire::DagWire;

/// Parse a GML digraph into a [`Dag`].
///
/// # Errors
///
/// Malformed GML, undirected graphs, cycles.
pub fn dag_from_gml(gml: &str) -> Result<Dag, IoError> {
    dag_from_wire(&dag_wire_from_gml(gml)?)
}

/// Serialize a [`Dag`] to GML.
///
/// # Errors
///
/// Wire conversion failures.
pub fn dag_to_gml(dag: &Dag, names: Option<&[String]>) -> Result<String, IoError> {
    Ok(dag_wire_to_gml(&dag_to_wire(dag)?, names))
}

/// Parse GML into [`DagWire`].
///
/// # Errors
///
/// Malformed / undirected GML.
pub fn dag_wire_from_gml(gml: &str) -> Result<DagWire, IoError> {
    let tokens = tokenize(gml)?;
    let mut i = 0;
    expect_ident(&tokens, &mut i, "graph")?;
    expect_char(&tokens, &mut i, '[')?;

    let mut directed: Option<bool> = None;
    let mut order: Vec<String> = Vec::new();
    let mut index: HashMap<String, u32> = HashMap::new();
    let mut edges: Vec<(u32, u32)> = Vec::new();

    while i < tokens.len() {
        if matches!(&tokens[i], Tok::Char(']')) {
            break;
        }
        match &tokens[i] {
            Tok::Ident(k) if k.eq_ignore_ascii_case("directed") => {
                i += 1;
                let v = expect_number(&tokens, &mut i)?;
                directed = Some(v != 0.0);
            }
            Tok::Ident(k) if k.eq_ignore_ascii_case("node") => {
                i += 1;
                expect_char(&tokens, &mut i, '[')?;
                let mut id: Option<String> = None;
                let mut label: Option<String> = None;
                while i < tokens.len() && !matches!(&tokens[i], Tok::Char(']')) {
                    let key = expect_any_ident(&tokens, &mut i)?.to_ascii_lowercase();
                    let val = expect_value(&tokens, &mut i)?;
                    match key.as_str() {
                        "id" => id = Some(val),
                        "label" => label = Some(val),
                        _ => {}
                    }
                }
                expect_char(&tokens, &mut i, ']')?;
                let name = label.or(id).ok_or_else(|| IoError::Convert("node missing id".into()))?;
                intern(&name, &mut order, &mut index)?;
            }
            Tok::Ident(k) if k.eq_ignore_ascii_case("edge") => {
                i += 1;
                expect_char(&tokens, &mut i, '[')?;
                let mut source: Option<String> = None;
                let mut target: Option<String> = None;
                while i < tokens.len() && !matches!(&tokens[i], Tok::Char(']')) {
                    let key = expect_any_ident(&tokens, &mut i)?.to_ascii_lowercase();
                    let val = expect_value(&tokens, &mut i)?;
                    match key.as_str() {
                        "source" => source = Some(val),
                        "target" => target = Some(val),
                        _ => {}
                    }
                }
                expect_char(&tokens, &mut i, ']')?;
                let s = source.ok_or_else(|| IoError::Convert("edge missing source".into()))?;
                let t = target.ok_or_else(|| IoError::Convert("edge missing target".into()))?;
                let from = intern(&s, &mut order, &mut index)?;
                let to = intern(&t, &mut order, &mut index)?;
                edges.push((from, to));
            }
            Tok::Ident(_) => {
                // Skip unknown key/value pairs at graph level.
                i += 1;
                if i < tokens.len() {
                    let _ = expect_value(&tokens, &mut i);
                }
            }
            other => {
                return Err(IoError::Convert(format!("unexpected GML token {other:?}")));
            }
        }
    }

    if directed != Some(true) {
        return Err(IoError::Convert("GML graph must be directed 1".into()));
    }
    if order.is_empty() {
        return Err(IoError::Convert("empty GML graph".into()));
    }

    // Prefer numeric contiguous labels when all nodes are numeric 0..n-1.
    let wire = remap_numeric_if_possible(&order, &edges);
    Ok(wire)
}

/// Emit GML from wire.
#[must_use]
pub fn dag_wire_to_gml(wire: &DagWire, names: Option<&[String]>) -> String {
    let mut out = String::from("graph [\n  directed 1\n");
    for i in 0..wire.node_count {
        let label = names
            .and_then(|n| n.get(i as usize))
            .cloned()
            .unwrap_or_else(|| i.to_string());
        out.push_str(&format!(
            "  node [\n    id \"{label}\"\n    label \"{label}\"\n  ]\n"
        ));
    }
    for &(a, b) in &wire.edges {
        let sa = names.and_then(|n| n.get(a as usize)).cloned().unwrap_or_else(|| a.to_string());
        let sb = names.and_then(|n| n.get(b as usize)).cloned().unwrap_or_else(|| b.to_string());
        out.push_str(&format!(
            "  edge [\n    source \"{sa}\"\n    target \"{sb}\"\n  ]\n"
        ));
    }
    out.push(']');
    out
}

fn remap_numeric_if_possible(order: &[String], edges: &[(u32, u32)]) -> DagWire {
    let n = order.len() as u32;
    let all_numeric = order.iter().all(|s| s.parse::<u32>().is_ok());
    if all_numeric {
        let mut vals: Vec<u32> = order.iter().map(|s| s.parse().unwrap()).collect();
        vals.sort_unstable();
        if vals.iter().copied().eq(0..n) {
            let mut map = HashMap::new();
            for (i, s) in order.iter().enumerate() {
                map.insert(i as u32, s.parse::<u32>().unwrap());
            }
            let edges = edges
                .iter()
                .map(|&(a, b)| (*map.get(&a).unwrap(), *map.get(&b).unwrap()))
                .collect();
            return DagWire { node_count: n, edges };
        }
    }
    DagWire { node_count: n, edges: edges.to_vec() }
}

fn intern(name: &str, order: &mut Vec<String>, index: &mut HashMap<String, u32>) -> Result<u32, IoError> {
    if let Some(&id) = index.get(name) {
        return Ok(id);
    }
    let id = u32::try_from(order.len()).map_err(|_| IoError::TooLarge)?;
    order.push(name.to_owned());
    index.insert(name.to_owned(), id);
    Ok(id)
}

#[derive(Debug)]
enum Tok {
    Ident(String),
    String(String),
    Number(f64),
    Char(char),
}

fn tokenize(input: &str) -> Result<Vec<Tok>, IoError> {
    let mut out = Vec::new();
    let bytes = input.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        if c == '#' {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        if c == '"' {
            i += 1;
            let start = i;
            while i < bytes.len() && bytes[i] != b'"' {
                i += 1;
            }
            if i >= bytes.len() {
                return Err(IoError::Convert("unterminated GML string".into()));
            }
            let s = String::from_utf8_lossy(&bytes[start..i]).into_owned();
            i += 1;
            out.push(Tok::String(s));
            continue;
        }
        if c == '[' || c == ']' {
            out.push(Tok::Char(c));
            i += 1;
            continue;
        }
        if c.is_ascii_digit() || c == '-' || c == '+' {
            let start = i;
            i += 1;
            while i < bytes.len()
                && ((bytes[i] as char).is_ascii_digit()
                    || bytes[i] == b'.'
                    || bytes[i] == b'e'
                    || bytes[i] == b'E'
                    || bytes[i] == b'-'
                    || bytes[i] == b'+')
            {
                i += 1;
            }
            let s = std::str::from_utf8(&bytes[start..i]).unwrap_or("");
            let n: f64 = s
                .parse()
                .map_err(|_| IoError::Convert(format!("bad GML number `{s}`")))?;
            out.push(Tok::Number(n));
            continue;
        }
        if c.is_ascii_alphabetic() || c == '_' {
            let start = i;
            i += 1;
            while i < bytes.len() {
                let ch = bytes[i] as char;
                if ch.is_ascii_alphanumeric() || ch == '_' {
                    i += 1;
                } else {
                    break;
                }
            }
            out.push(Tok::Ident(String::from_utf8_lossy(&bytes[start..i]).into_owned()));
            continue;
        }
        return Err(IoError::Convert(format!("unexpected GML char `{c}`")));
    }
    Ok(out)
}

fn expect_ident(tokens: &[Tok], i: &mut usize, want: &str) -> Result<(), IoError> {
    let got = expect_any_ident(tokens, i)?;
    if !got.eq_ignore_ascii_case(want) {
        return Err(IoError::Convert(format!("expected `{want}`, got `{got}`")));
    }
    Ok(())
}

fn expect_any_ident(tokens: &[Tok], i: &mut usize) -> Result<String, IoError> {
    match tokens.get(*i) {
        Some(Tok::Ident(s)) => {
            *i += 1;
            Ok(s.clone())
        }
        _ => Err(IoError::Convert("expected identifier".into())),
    }
}

fn expect_char(tokens: &[Tok], i: &mut usize, c: char) -> Result<(), IoError> {
    match tokens.get(*i) {
        Some(Tok::Char(x)) if *x == c => {
            *i += 1;
            Ok(())
        }
        _ => Err(IoError::Convert(format!("expected `{c}`"))),
    }
}

fn expect_number(tokens: &[Tok], i: &mut usize) -> Result<f64, IoError> {
    match tokens.get(*i) {
        Some(Tok::Number(n)) => {
            *i += 1;
            Ok(*n)
        }
        _ => Err(IoError::Convert("expected number".into())),
    }
}

fn expect_value(tokens: &[Tok], i: &mut usize) -> Result<String, IoError> {
    match tokens.get(*i) {
        Some(Tok::String(s)) => {
            *i += 1;
            Ok(s.clone())
        }
        Some(Tok::Ident(s)) => {
            *i += 1;
            Ok(s.clone())
        }
        Some(Tok::Number(n)) => {
            *i += 1;
            if n.fract() == 0.0 && *n >= 0.0 {
                Ok(format!("{}", *n as i64))
            } else {
                Ok(n.to_string())
            }
        }
        _ => Err(IoError::Convert("expected value".into())),
    }
}

#[cfg(test)]
mod tests {
    use causal_graph::DenseNodeId;

    use super::*;

    #[test]
    fn gml_round_trip_named() {
        let gml = r#"graph [
  directed 1
  node [ id "Z" label "Z" ]
  node [ id "X" label "X" ]
  node [ id "Y" label "Y" ]
  edge [ source "Z" target "X" ]
  edge [ source "X" target "Y" ]
]"#;
        let dag = dag_from_gml(gml).unwrap();
        assert_eq!(dag.node_count(), 3);
        let out = dag_to_gml(&dag, Some(&["Z".into(), "X".into(), "Y".into()])).unwrap();
        let back = dag_from_gml(&out).unwrap();
        assert_eq!(back.node_count(), 3);
    }

    #[test]
    fn rejects_undirected() {
        let gml = "graph [ directed 0 node [ id 0 ] node [ id 1 ] edge [ source 0 target 1 ] ]";
        let err = dag_from_gml(gml).unwrap_err();
        assert!(matches!(err, IoError::Convert(_)));
    }

    #[test]
    fn numeric_ids() {
        let mut dag = Dag::with_variables(2);
        dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let s = dag_to_gml(&dag, None).unwrap();
        let back = dag_from_gml(&s).unwrap();
        assert!(back.reaches(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)));
    }
}
