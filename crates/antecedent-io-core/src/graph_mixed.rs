//! PAG / CPDAG / ADMG interchange (DOT / JSON / GML / `NetworkX`).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod dot;
mod gml;

pub use dot::{
    admg_from_dot, admg_to_dot, admg_with_names_from_dot, admg_with_names_from_dot_with_limits,
    cpdag_from_dot, cpdag_to_dot, cpdag_with_names_from_dot, cpdag_with_names_from_dot_with_limits,
    pag_from_dot, pag_to_dot, pag_with_names_from_dot, pag_with_names_from_dot_with_limits,
};
mod networkx;

pub use networkx::{
    admg_from_networkx_node_link, admg_to_networkx_node_link,
    admg_with_names_from_networkx_node_link, admg_with_names_from_networkx_node_link_with_limits,
    cpdag_from_networkx_node_link, cpdag_to_networkx_node_link,
    cpdag_with_names_from_networkx_node_link, cpdag_with_names_from_networkx_node_link_with_limits,
    pag_from_networkx_node_link, pag_to_networkx_node_link, pag_with_names_from_networkx_node_link,
    pag_with_names_from_networkx_node_link_with_limits,
};

pub use gml::{
    admg_from_gml, admg_to_gml, admg_with_names_from_gml, admg_with_names_from_gml_with_limits,
    cpdag_from_gml, cpdag_to_gml, cpdag_with_names_from_gml, cpdag_with_names_from_gml_with_limits,
    pag_from_gml, pag_to_gml, pag_with_names_from_gml, pag_with_names_from_gml_with_limits,
};

use antecedent_graph::{Admg, Cpdag, Pag};
use serde::{Deserialize, Serialize};

use crate::convert::{
    GraphLimits, admg_from_wire_with_limits, admg_to_wire, cpdag_from_wire_with_limits,
    cpdag_to_wire, pag_from_wire_with_limits, pag_to_wire,
};
use crate::error::IoError;
use crate::graph_json::{check_optional_names, json_from_str, json_to_pretty, without_names};
use crate::wire::{AdmgWire, EndpointWire, GraphCpdagWire, MarkedEdgeWire, PagWire};

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

without_names! {
    /// Parse JSON into a [`Pag`].
    ///
    /// # Errors
    ///
    /// JSON / structure errors.
    pub fn pag_from_json(json: &str) -> Pag = pag_with_names_from_json
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
    pag_with_names_from_json_with_limits(json, &GraphLimits::default())
}

/// [`pag_with_names_from_json`] under the caller's graph bounds: a document over `limits` is refused.
///
/// # Errors
///
/// As [`pag_with_names_from_json`], and [`IoError::Convert`] when a bound of `limits` is exceeded.
pub fn pag_with_names_from_json_with_limits(
    json: &str,
    limits: &GraphLimits,
) -> Result<(Pag, Vec<String>), IoError> {
    let doc: PagJson = json_from_str(json)?;
    let names = names_or_default(doc.variable_names.as_deref(), doc.node_count, limits)?;
    let g = pag_from_wire_with_limits(
        &PagWire { node_count: doc.node_count, edges: doc.edges },
        limits,
    )?;
    Ok((g, names))
}

graph_to_json!(pag_to_json, Pag, pag_to_wire, PagJson, pag_kind(), { edges });

without_names! {
    /// Parse JSON into a [`Cpdag`].
    ///
    /// # Errors
    ///
    /// JSON / structure errors.
    pub fn cpdag_from_json(json: &str) -> Cpdag = cpdag_with_names_from_json
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
    cpdag_with_names_from_json_with_limits(json, &GraphLimits::default())
}

/// [`cpdag_with_names_from_json`] under the caller's graph bounds: a document over `limits` is refused.
///
/// # Errors
///
/// As [`cpdag_with_names_from_json`], and [`IoError::Convert`] when a bound of `limits` is exceeded.
pub fn cpdag_with_names_from_json_with_limits(
    json: &str,
    limits: &GraphLimits,
) -> Result<(Cpdag, Vec<String>), IoError> {
    let doc: CpdagJson = json_from_str(json)?;
    let names = names_or_default(doc.variable_names.as_deref(), doc.node_count, limits)?;
    let g = cpdag_from_wire_with_limits(
        &GraphCpdagWire {
            node_count: doc.node_count,
            directed: doc.directed,
            undirected: doc.undirected,
        },
        limits,
    )?;
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

without_names! {
    /// Parse JSON into an [`Admg`].
    ///
    /// # Errors
    ///
    /// JSON / structure errors.
    pub fn admg_from_json(json: &str) -> Admg = admg_with_names_from_json
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
    admg_with_names_from_json_with_limits(json, &GraphLimits::default())
}

/// [`admg_with_names_from_json`] under the caller's graph bounds: a document over `limits` is refused.
///
/// # Errors
///
/// As [`admg_with_names_from_json`], and [`IoError::Convert`] when a bound of `limits` is exceeded.
pub fn admg_with_names_from_json_with_limits(
    json: &str,
    limits: &GraphLimits,
) -> Result<(Admg, Vec<String>), IoError> {
    let doc: AdmgJson = json_from_str(json)?;
    let names = names_or_default(doc.variable_names.as_deref(), doc.node_count, limits)?;
    let g = admg_from_wire_with_limits(
        &AdmgWire {
            node_count: doc.node_count,
            directed: doc.directed,
            bidirected: doc.bidirected,
        },
        limits,
    )?;
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

/// Validated names, or dense-index strings when the document carries none.
fn names_or_default(
    names: Option<&[String]>,
    node_count: u32,
    limits: &GraphLimits,
) -> Result<Vec<String>, IoError> {
    crate::convert::check_graph_budget_within(node_count, 0, limits)?;
    check_optional_names(names, node_count)?;
    Ok(names.map_or_else(|| (0..node_count).map(|i| i.to_string()).collect(), <[String]>::to_vec))
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
        let json = pag_to_json(&g, None).unwrap();
        assert_eq!(
            json,
            "{\n  \"kind\": \"pag\",\n  \"node_count\": 2,\n  \"edges\": [\n    {\n      \"a\": 0,\n      \"b\": 1,\n      \"at_a\": \"circle\",\n      \"at_b\": \"arrow\"\n    }\n  ]\n}"
        );
        let back = pag_from_json(&json).unwrap();
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
    fn admg_dot_dir_both_round_trip_preserves_bidirected() {
        let mut g = Admg::with_variables(2);
        g.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let dot = admg_to_dot(&g, None).unwrap();
        assert!(dot.contains("dir=both"), "expected dir=both in DOT: {dot}");
        let back = admg_from_dot(&dot).unwrap();
        assert_eq!(back.bidirected_neighbors(DenseNodeId::from_raw(0)).len(), 1);
        assert!(back.children(DenseNodeId::from_raw(0)).is_empty());
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

    fn pag_marks(dot: &str) -> Vec<(u32, u32, EndpointWire, EndpointWire)> {
        let wire = pag_to_wire(&pag_from_dot(dot).unwrap()).unwrap();
        wire.edges.iter().map(|e| (e.a, e.b, e.at_a, e.at_b)).collect()
    }

    #[test]
    fn dot_arrowhead_arrowtail_map_to_pag_marks() {
        // causal-learn writes `a o-> b` as dir=both with odot / normal shapes.
        let marks = pag_marks("digraph { a -> b [dir=both, arrowtail=odot, arrowhead=normal]; }");
        assert_eq!(marks, vec![(0, 1, EndpointWire::Circle, EndpointWire::Arrow)]);
        let marks = pag_marks("digraph { a -> b [dir=both, arrowtail=none, arrowhead=normal]; }");
        assert_eq!(marks, vec![(0, 1, EndpointWire::Tail, EndpointWire::Arrow)]);
        let marks = pag_marks("digraph { a -> b [dir=both, arrowtail=odot, arrowhead=odot]; }");
        assert_eq!(marks, vec![(0, 1, EndpointWire::Circle, EndpointWire::Circle)]);
        // An o-> edge must not load into an ADMG as <->.
        assert!(
            admg_from_dot("digraph { a -> b [dir=both, arrowtail=odot, arrowhead=normal]; }")
                .is_err()
        );
    }

    #[test]
    fn dot_dir_back_and_none_are_honoured() {
        assert_eq!(
            pag_marks("digraph { a -> b [dir=back]; }"),
            vec![(0, 1, EndpointWire::Arrow, EndpointWire::Tail)]
        );
        assert_eq!(
            pag_marks("digraph { a -> b [dir=none]; }"),
            vec![(0, 1, EndpointWire::Tail, EndpointWire::Tail)]
        );
        let admg = admg_from_dot("digraph { a -> b [dir=back]; }").unwrap();
        let wire = admg_to_wire(&admg).unwrap();
        assert_eq!(wire.directed, vec![(1, 0)]);
        assert!(admg_from_dot("digraph { a -> b [dir=none]; }").is_err());
    }

    #[test]
    fn dot_half_marks_and_unknown_shapes_are_refused() {
        for dot in [
            "digraph { a -> b [mark_a=circle]; }",
            "digraph { a -> b [mark_b=arrow]; }",
            "digraph { a -> b [dir=sideways]; }",
            "digraph { a -> b [arrowhead=diamond]; }",
            "digraph { a -> b [arrowtail=odot]; }",
            "digraph { a -> b [mark_a=tail, mark_b=arrow, dir=both]; }",
        ] {
            assert!(pag_from_dot(dot).is_err(), "{dot}");
            assert!(admg_from_dot(dot).is_err(), "{dot}");
        }
    }

    #[test]
    fn dag_dot_refuses_non_trivial_arrow_shapes() {
        use crate::graph_dot::dag_from_dot;
        assert!(dag_from_dot("digraph { a -> b [arrowhead=odot]; }").is_err());
        assert!(dag_from_dot("digraph { a -> b [arrowtail=odot]; }").is_err());
        assert!(dag_from_dot("digraph { a -> b [arrowhead=normal, arrowtail=none]; }").is_ok());
    }

    #[test]
    fn with_names_nameless_falls_back_to_dense_index() {
        let mut g = Cpdag::with_variables(2);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let (_g, names) = cpdag_with_names_from_json(&cpdag_to_json(&g, None).unwrap()).unwrap();
        assert_eq!(names, vec!["0".to_string(), "1".to_string()]);
    }
}
