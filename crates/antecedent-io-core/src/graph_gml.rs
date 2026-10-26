//! GML digraph subset import/export for causal DAGs.
//!
//! Supports pinned baseline-style:
//! `graph [ directed 1 node [ id "Z" label "Z" ] edge [ source "Z" target "X" ] ]`
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;
use std::fmt::Write as _;

use antecedent_graph::Dag;

use crate::convert::{dag_from_wire, dag_to_wire};
use crate::error::IoError;
use crate::graph_dot;
use crate::graph_json::without_names;
use crate::wire::DagWire;

without_names! {
    /// Parse a GML digraph into a [`Dag`].
    ///
    /// # Errors
    ///
    /// Malformed GML, undirected graphs, cycles.
    pub fn dag_from_gml(gml: &str) -> Dag = dag_with_names_from_gml
}

/// Parse a GML digraph into a [`Dag`] plus its node labels.
///
/// Labels are returned in dense-id order. When every label is numeric and
/// forms a contiguous `0..n` set, dense ids follow those numeric values and
/// the returned names are the dense-index strings, since the document
/// carries no distinct name information in that case.
///
/// # Errors
///
/// Malformed GML, undirected graphs, cycles.
pub fn dag_with_names_from_gml(gml: &str) -> Result<(Dag, Vec<String>), IoError> {
    let (wire, names) = dag_wire_and_names_from_gml(gml)?;
    Ok((dag_from_wire(&wire)?, names))
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
    dag_wire_and_names_from_gml(gml).map(|(wire, _names)| wire)
}

/// Parse GML into [`DagWire`] plus node labels in dense-id order.
fn dag_wire_and_names_from_gml(gml: &str) -> Result<(DagWire, Vec<String>), IoError> {
    let doc = parse_gml_document(gml)?;
    let edges: Vec<(u32, u32)> = doc.edges.iter().map(|e| (e.from, e.to)).collect();
    // Prefer numeric contiguous labels when all nodes are numeric 0..n-1.
    match graph_dot::remap_numeric_dense(&doc.names, &edges)? {
        // Dense id == numeric label in this case, so the label carries no
        // information beyond the dense index; fall back to index strings.
        Some(wire) => {
            let names = (0..wire.node_count).map(|i| i.to_string()).collect();
            Ok((wire, names))
        }
        None => {
            let node_count = u32::try_from(doc.names.len()).map_err(|_| IoError::TooLarge)?;
            Ok((DagWire { node_count, edges }, doc.names))
        }
    }
}

/// One GML `edge` block, bound to declared node ids.
pub(crate) struct GmlRawEdge {
    pub(crate) from: u32,
    pub(crate) to: u32,
    /// Remaining edge attributes (lowercased keys), e.g. `undirected`, `mark_a`.
    pub(crate) attrs: HashMap<String, String>,
}

/// A parsed `graph [ directed 1 ... ]` document: nodes in declaration order
/// and edges bound against node `id` keys.
pub(crate) struct GmlDocument {
    /// Display names (`label`, else `id`) in dense-id (declaration) order.
    pub(crate) names: Vec<String>,
    pub(crate) edges: Vec<GmlRawEdge>,
}

/// Parse the shared GML subset used by every graph kind.
///
/// Nodes are collected before any edge is bound, so declaration order does
/// not matter; an edge endpoint naming no declared node and a node `id`
/// declared twice are both refused rather than silently repaired.
pub(crate) fn parse_gml_document(gml: &str) -> Result<GmlDocument, IoError> {
    let tokens = tokenize(gml)?;
    let mut i = 0;
    expect_ident(&tokens, &mut i, "graph")?;
    expect_char(&tokens, &mut i, '[')?;

    let mut directed: Option<bool> = None;
    let mut nodes: Vec<HashMap<String, String>> = Vec::new();
    let mut raw_edges: Vec<HashMap<String, String>> = Vec::new();
    while i < tokens.len() {
        if matches!(&tokens[i], Tok::Char(']')) {
            break;
        }
        match &tokens[i] {
            Tok::Ident(k) if k.eq_ignore_ascii_case("directed") => {
                i += 1;
                directed = Some(expect_number(&tokens, &mut i)? != 0.0);
            }
            Tok::Ident(k) if k.eq_ignore_ascii_case("node") => {
                i += 1;
                nodes.push(parse_attr_block(&tokens, &mut i)?);
            }
            Tok::Ident(k) if k.eq_ignore_ascii_case("edge") => {
                i += 1;
                raw_edges.push(parse_attr_block(&tokens, &mut i)?);
            }
            Tok::Ident(_) => {
                // Skip unknown key/value pairs (and nested blocks such as `graphics [ ... ]`)
                // at graph level; a key without a value is malformed.
                i += 1;
                skip_value(&tokens, &mut i)?;
            }
            other => return Err(IoError::Convert(format!("unexpected GML token {other:?}"))),
        }
    }
    expect_char(&tokens, &mut i, ']')?;
    if directed != Some(true) {
        return Err(IoError::Convert("GML graph must be directed 1".into()));
    }
    if nodes.is_empty() {
        return Err(IoError::Convert("empty GML graph".into()));
    }

    let mut id_order: Vec<String> = Vec::new();
    let mut id_index: HashMap<String, u32> = HashMap::new();
    let mut names = Vec::with_capacity(nodes.len());
    for mut node in nodes {
        let id = node.remove("id").ok_or_else(|| IoError::Convert("node missing id".into()))?;
        if id_index.contains_key(&id) {
            return Err(IoError::Convert(format!("GML node id `{id}` is declared twice")));
        }
        graph_dot::intern(&id, &mut id_order, &mut id_index)?;
        names.push(node.remove("label").unwrap_or(id));
    }
    let bind = |attrs: &mut HashMap<String, String>, key: &str| -> Result<u32, IoError> {
        let v = attrs.remove(key).ok_or_else(|| IoError::Convert(format!("edge missing {key}")))?;
        id_index
            .get(&v)
            .copied()
            .ok_or_else(|| IoError::Convert(format!("GML edge {key} `{v}` names no declared node")))
    };
    let mut edges = Vec::with_capacity(raw_edges.len());
    for mut attrs in raw_edges {
        let from = bind(&mut attrs, "source")?;
        let to = bind(&mut attrs, "target")?;
        edges.push(GmlRawEdge { from, to, attrs });
    }
    Ok(GmlDocument { names, edges })
}

/// Parse `[ key value ... ]` into a map with lowercased keys.
fn parse_attr_block(tokens: &[Tok], i: &mut usize) -> Result<HashMap<String, String>, IoError> {
    expect_char(tokens, i, '[')?;
    let mut attrs = HashMap::new();
    while *i < tokens.len() && !matches!(&tokens[*i], Tok::Char(']')) {
        let key = expect_any_ident(tokens, i)?.to_ascii_lowercase();
        if matches!(tokens.get(*i), Some(Tok::Char('['))) {
            // Nested block (`graphics [ ... ]`, `LabelGraphics [ ... ]`): not a node or edge
            // attribute this importer reads.
            skip_block(tokens, i)?;
            continue;
        }
        let val = expect_value(tokens, i)?;
        attrs.insert(key, val);
    }
    expect_char(tokens, i, ']')?;
    Ok(attrs)
}

/// Emit GML from wire.
#[must_use]
pub fn dag_wire_to_gml(wire: &DagWire, names: Option<&[String]>) -> String {
    let label_of = |i: u32| -> std::borrow::Cow<'_, str> {
        names.and_then(|n| n.get(i as usize)).map_or_else(
            || std::borrow::Cow::Owned(i.to_string()),
            |name| std::borrow::Cow::Borrowed(name.as_str()),
        )
    };
    let mut out = String::from("graph [\n  directed 1\n");
    for i in 0..wire.node_count {
        let label = escape_gml_string(&label_of(i));
        let _ = write!(out, "  node [\n    id \"{label}\"\n    label \"{label}\"\n  ]\n");
    }
    for &(a, b) in &wire.edges {
        let (sa, sb) = (escape_gml_string(&label_of(a)), escape_gml_string(&label_of(b)));
        let _ = write!(out, "  edge [\n    source \"{sa}\"\n    target \"{sb}\"\n  ]\n");
    }
    out.push(']');
    out
}

/// GML strings cannot contain a raw `"`; the format's escapes are the HTML entities
/// `&quot;` and `&amp;`.
pub(crate) fn escape_gml_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '"' => out.push_str("&quot;"),
            _ => out.push(ch),
        }
    }
    out
}

fn unescape_gml_string(s: &str) -> String {
    s.replace("&quot;", "\"").replace("&amp;", "&")
}

#[derive(Debug)]
pub(crate) enum Tok {
    Ident(String),
    String(String),
    /// A number as written; ids keep their spelling instead of round-tripping through `f64`.
    Number(String),
    Char(char),
}

pub(crate) fn tokenize(input: &str) -> Result<Vec<Tok>, IoError> {
    let mut out = Vec::new();
    let mut chars = input.char_indices().peekable();
    while let Some(&(start, c)) = chars.peek() {
        if c.is_whitespace() {
            chars.next();
            continue;
        }
        if c == '#' {
            while chars.next_if(|&(_, ch)| ch != '\n').is_some() {}
            continue;
        }
        if c == '"' {
            chars.next();
            let body = start + 1;
            let end = chars
                .by_ref()
                .find(|&(_, ch)| ch == '"')
                .map(|(index, _)| index)
                .ok_or_else(|| IoError::Convert("unterminated GML string".into()))?;
            out.push(Tok::String(unescape_gml_string(&input[body..end])));
            continue;
        }
        if c == '[' || c == ']' {
            out.push(Tok::Char(c));
            chars.next();
            continue;
        }
        if c.is_ascii_digit() || c == '-' || c == '+' {
            chars.next();
            let mut end = start + 1;
            while let Some((index, ch)) = chars
                .next_if(|&(_, ch)| ch.is_ascii_digit() || matches!(ch, '.' | 'e' | 'E' | '-' | '+'))
            {
                end = index + ch.len_utf8();
            }
            let text = &input[start..end];
            text.parse::<f64>()
                .map_err(|_| IoError::Convert(format!("bad GML number `{text}`")))?;
            out.push(Tok::Number(text.to_owned()));
            continue;
        }
        if c.is_ascii_alphabetic() || c == '_' {
            chars.next();
            let mut end = start + 1;
            while let Some((index, ch)) =
                chars.next_if(|&(_, ch)| ch.is_ascii_alphanumeric() || ch == '_')
            {
                end = index + ch.len_utf8();
            }
            out.push(Tok::Ident(input[start..end].to_owned()));
            continue;
        }
        return Err(IoError::Convert(format!("unexpected GML char `{c}`")));
    }
    Ok(out)
}

pub(crate) fn expect_ident(tokens: &[Tok], i: &mut usize, want: &str) -> Result<(), IoError> {
    let got = expect_any_ident(tokens, i)?;
    if !got.eq_ignore_ascii_case(want) {
        return Err(IoError::Convert(format!("expected `{want}`, got `{got}`")));
    }
    Ok(())
}

pub(crate) fn expect_any_ident(tokens: &[Tok], i: &mut usize) -> Result<String, IoError> {
    match tokens.get(*i) {
        Some(Tok::Ident(s)) => {
            *i += 1;
            Ok(s.clone())
        }
        _ => Err(IoError::Convert("expected identifier".into())),
    }
}

pub(crate) fn expect_char(tokens: &[Tok], i: &mut usize, c: char) -> Result<(), IoError> {
    match tokens.get(*i) {
        Some(Tok::Char(x)) if *x == c => {
            *i += 1;
            Ok(())
        }
        _ => Err(IoError::Convert(format!("expected `{c}`"))),
    }
}

pub(crate) fn expect_number(tokens: &[Tok], i: &mut usize) -> Result<f64, IoError> {
    match tokens.get(*i) {
        Some(Tok::Number(text)) => {
            let value = text
                .parse::<f64>()
                .map_err(|_| IoError::Convert(format!("bad GML number `{text}`")))?;
            *i += 1;
            Ok(value)
        }
        _ => Err(IoError::Convert("expected number".into())),
    }
}

/// Skip a balanced `[ ... ]` block, including nested blocks.
fn skip_block(tokens: &[Tok], i: &mut usize) -> Result<(), IoError> {
    expect_char(tokens, i, '[')?;
    let mut depth = 1_usize;
    while depth > 0 {
        match tokens.get(*i) {
            Some(Tok::Char('[')) => depth += 1,
            Some(Tok::Char(']')) => depth -= 1,
            Some(_) => {}
            None => return Err(IoError::Convert("unterminated GML block".into())),
        }
        *i += 1;
    }
    Ok(())
}

/// Skip the value of an ignored key: a scalar or a nested block.
fn skip_value(tokens: &[Tok], i: &mut usize) -> Result<(), IoError> {
    if matches!(tokens.get(*i), Some(Tok::Char('['))) {
        skip_block(tokens, i)
    } else {
        expect_value(tokens, i).map(|_| ())
    }
}

/// The text of a numeric token as an identifier: integers are normalized (`+5` and `005`
/// read as `5`, `-0` as `0`), anything else keeps its spelling, so distinct ids never
/// collapse through floating point.
fn number_text(text: &str) -> String {
    text.parse::<i64>().map_or_else(|_| text.to_owned(), |integer| integer.to_string())
}

pub(crate) fn expect_value(tokens: &[Tok], i: &mut usize) -> Result<String, IoError> {
    match tokens.get(*i) {
        Some(Tok::String(s) | Tok::Ident(s)) => {
            *i += 1;
            Ok(s.clone())
        }
        Some(Tok::Number(text)) => {
            *i += 1;
            Ok(number_text(text))
        }
        _ => Err(IoError::Convert("expected value".into())),
    }
}

#[cfg(test)]
mod tests {
    use antecedent_graph::DenseNodeId;

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
    fn networkx_style_gml_binds_edges_to_id_not_label() {
        // NetworkX writes numeric `id` and string `label`; edges reference `id`.
        let gml = r#"graph [
  directed 1
  node [ id 0 label "Z" ]
  node [ id 1 label "X" ]
  node [ id 2 label "Y" ]
  edge [ source 0 target 1 ]
]"#;
        let (dag, names) = dag_with_names_from_gml(gml).unwrap();
        assert_eq!(dag.node_count(), 3);
        assert_eq!(names, vec!["Z".to_string(), "X".to_string(), "Y".to_string()]);
        assert!(dag.reaches(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)));
        assert!(!dag.reaches(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)));
    }

    #[test]
    fn labels_with_quotes_and_ampersands_survive_a_round_trip() {
        let dag = dag_from_gml(
            "graph [ directed 1 node [ id 0 label \"a\" ] node [ id 1 label \"b\" ] \
                              edge [ source 0 target 1 ] ]",
        )
        .unwrap();
        let wire = crate::convert::dag_to_wire(&dag).unwrap();
        let names = vec!["say \"hi\"".to_string(), "a&b".to_string()];
        let gml = dag_wire_to_gml(&wire, Some(&names));
        assert!(gml.contains("say &quot;hi&quot;"), "{gml}");
        assert!(gml.contains("a&amp;b"), "{gml}");
        let (back, back_names) = dag_with_names_from_gml(&gml).unwrap();
        assert_eq!(back_names, names);
        assert!(back.reaches(
            antecedent_graph::DenseNodeId::from_raw(0),
            antecedent_graph::DenseNodeId::from_raw(1)
        ));
    }

    #[test]
    fn with_names_round_trip_preserves_labels() {
        let mut dag = Dag::with_variables(2);
        dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let names = vec!["x".to_string(), "y".to_string()];
        let s = dag_to_gml(&dag, Some(&names)).unwrap();
        let (back, back_names) = dag_with_names_from_gml(&s).unwrap();
        assert_eq!(back.node_count(), 2);
        assert_eq!(back_names, names);
    }

    #[test]
    fn with_names_nameless_falls_back_to_dense_index() {
        let mut dag = Dag::with_variables(2);
        dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let s = dag_to_gml(&dag, None).unwrap();
        let (_back, names) = dag_with_names_from_gml(&s).unwrap();
        assert_eq!(names, vec!["0".to_string(), "1".to_string()]);
    }

    #[test]
    fn rejects_undirected() {
        let gml = "graph [ directed 0 node [ id 0 ] node [ id 1 ] edge [ source 0 target 1 ] ]";
        let err = dag_from_gml(gml).unwrap_err();
        assert!(matches!(err, IoError::Convert(_)));
    }

    #[test]
    fn edge_to_undeclared_node_is_refused() {
        let gml = r#"graph [ directed 1 node [ id "A" ] node [ id "B" ]
            edge [ source "A" target "C" ] ]"#;
        let err = dag_from_gml(gml).unwrap_err();
        assert!(matches!(&err, IoError::Convert(m) if m.contains("`C`")), "{err:?}");
        let err = crate::graph_mixed::cpdag_from_gml(gml).unwrap_err();
        assert!(matches!(&err, IoError::Convert(m) if m.contains("`C`")), "{err:?}");
    }

    #[test]
    fn duplicate_node_id_is_refused() {
        let gml = r#"graph [ directed 1 node [ id "A" label "a1" ] node [ id "A" label "a2" ]
            node [ id "B" ] edge [ source "A" target "B" ] ]"#;
        let err = dag_from_gml(gml).unwrap_err();
        assert!(matches!(&err, IoError::Convert(m) if m.contains("twice")), "{err:?}");
        let err = crate::graph_mixed::admg_from_gml(gml).unwrap_err();
        assert!(matches!(&err, IoError::Convert(m) if m.contains("twice")), "{err:?}");
    }

    #[test]
    fn edge_before_node_declaration_keeps_the_label() {
        let gml = r#"graph [ directed 1 edge [ source "a" target "b" ]
            node [ id "a" label "Alpha" ] node [ id "b" label "Beta" ] ]"#;
        let (dag, names) = dag_with_names_from_gml(gml).unwrap();
        assert_eq!(names, vec!["Alpha".to_string(), "Beta".to_string()]);
        assert!(dag.reaches(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)));
        let (_g, names) = crate::graph_mixed::pag_with_names_from_gml(gml).unwrap();
        assert_eq!(names, vec!["Alpha".to_string(), "Beta".to_string()]);
    }

    #[test]
    fn numeric_ids() {
        let mut dag = Dag::with_variables(2);
        dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let s = dag_to_gml(&dag, None).unwrap();
        let back = dag_from_gml(&s).unwrap();
        assert!(back.reaches(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)));
    }

    #[test]
    fn nested_blocks_are_skipped_and_do_not_refuse_real_documents() {
        let gml = r##"graph [ directed 1 comment "x" meta [ a 1 inner [ b 2 ] ]
            node [ id 0 label "A" graphics [ x 1.5 y 2 w 10 ] ]
            node [ id 1 label "B" LabelGraphics [ text "B" ] ]
            edge [ source 0 target 1 graphics [ fill "#000" ] ] ]"##;
        let (dag, names) = dag_with_names_from_gml(gml).unwrap();
        assert_eq!(names, vec!["A".to_string(), "B".to_string()]);
        assert!(dag.reaches(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)));
    }

    #[test]
    fn a_graph_level_key_without_a_value_or_an_open_block_is_refused() {
        assert!(dag_from_gml("graph [ directed 1 node [ id 0 ] foo ]").is_err());
        assert!(dag_from_gml("graph [ directed 1 node [ id 0 ] foo [ a 1 ").is_err());
        assert!(dag_from_gml("graph [ directed 1 node [ id 0 ]").is_err());
    }

    #[test]
    fn large_integer_ids_do_not_collapse_through_floating_point() {
        let gml = "graph [ directed 1 node [ id 9007199254740993 ] \
                   node [ id 9007199254740992 ] edge [ source 9007199254740993 \
                   target 9007199254740992 ] ]";
        let (dag, names) = dag_with_names_from_gml(gml).unwrap();
        assert_eq!(dag.node_count(), 2);
        assert_eq!(names.len(), 2);
        assert_ne!(names[0], names[1]);
    }

    #[test]
    fn non_ascii_labels_survive_tokenizing() {
        let gml = "graph [ directed 1 node [ id 0 label \"caf\u{e9} \u{a0}x\" ] ]";
        let (_dag, names) = dag_with_names_from_gml(gml).unwrap();
        assert_eq!(names, vec!["caf\u{e9} \u{a0}x".to_string()]);
    }
}
