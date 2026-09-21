//! `NetworkX`-compatible JSON graph interchange (`node_link` / adjacency).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;

use antecedent_graph::Dag;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::convert::{dag_from_wire, dag_to_wire};
use crate::error::IoError;
use crate::graph_dot;
use crate::wire::DagWire;

/// `NetworkX` `node_link_data` subset.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct NetworkXNodeLink {
    /// Must be true.
    pub directed: bool,
    /// Multigraph flag (must be false for DAGs).
    #[serde(default)]
    pub multigraph: bool,
    /// Graph attributes (ignored).
    #[serde(default)]
    pub graph: JsonValue,
    /// Nodes.
    pub nodes: Vec<NetworkXNode>,
    /// Links.
    pub links: Vec<NetworkXLink>,
}

/// Node entry.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct NetworkXNode {
    /// Node id (string or number).
    pub id: JsonValue,
}

/// Link entry.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct NetworkXLink {
    /// Source id.
    pub source: JsonValue,
    /// Target id.
    pub target: JsonValue,
}

/// `NetworkX` `adjacency_data` document.
///
/// Real NetworkX documents keep a parallel top-level `adjacency` array: entry
/// `i` lists the out-neighbors of `nodes[i]`, each neighbor carrying an `id`
/// field (plus optional edge attributes).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct NetworkXAdjacency {
    /// Must be true.
    pub directed: bool,
    /// Multigraph.
    #[serde(default)]
    pub multigraph: bool,
    /// Graph attrs.
    #[serde(default)]
    pub graph: JsonValue,
    /// Nodes (ids only; adjacency is parallel, not nested).
    pub nodes: Vec<NetworkXNode>,
    /// Parallel to `nodes`: each entry is that node's out-neighbor list.
    pub adjacency: Vec<Vec<NetworkXAdjNeighbor>>,
}

/// One out-neighbor entry in a NetworkX adjacency list.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct NetworkXAdjNeighbor {
    /// Neighbor id.
    pub id: JsonValue,
}

/// Parse `NetworkX` node-link JSON into a [`Dag`].
///
/// # Errors
///
/// Undirected / malformed JSON / cycles.
pub fn dag_from_networkx_node_link(json: &str) -> Result<Dag, IoError> {
    dag_with_names_from_networkx_node_link(json).map(|(dag, _names)| dag)
}

/// Parse `NetworkX` node-link JSON into a [`Dag`] plus its node names.
///
/// Names are the document's node `id` values (stringified), in dense-id
/// order. A document with plain sequential integer ids (as emitted by
/// [`dag_to_networkx_node_link`] with `names = None`) yields dense-index
/// strings, since the document carries no distinct name information in
/// that case.
///
/// # Errors
///
/// Undirected / malformed JSON / cycles.
pub fn dag_with_names_from_networkx_node_link(json: &str) -> Result<(Dag, Vec<String>), IoError> {
    let (wire, names) = dag_wire_and_names_from_networkx_node_link(json)?;
    Ok((dag_from_wire(&wire)?, names))
}

/// Serialize a [`Dag`] to `NetworkX` node-link JSON.
///
/// # Errors
///
/// Wire / JSON failures.
pub fn dag_to_networkx_node_link(dag: &Dag, names: Option<&[String]>) -> Result<String, IoError> {
    let doc = networkx_node_link_from_wire(&dag_to_wire(dag)?, names);
    serde_json::to_string_pretty(&doc).map_err(|e| IoError::Convert(format!("json: {e}")))
}

/// Parse node-link JSON to wire.
///
/// # Errors
///
/// Undirected or parse errors.
pub fn dag_wire_from_networkx_node_link(json: &str) -> Result<DagWire, IoError> {
    dag_wire_and_names_from_networkx_node_link(json).map(|(wire, _names)| wire)
}

/// Parse node-link JSON to wire plus node names (document `id` values) in
/// dense-id order.
fn dag_wire_and_names_from_networkx_node_link(
    json: &str,
) -> Result<(DagWire, Vec<String>), IoError> {
    let doc: NetworkXNodeLink =
        serde_json::from_str(json).map_err(|e| IoError::Convert(format!("json: {e}")))?;
    if !doc.directed {
        return Err(IoError::Convert("NetworkX graph must be directed".into()));
    }
    if doc.multigraph {
        return Err(IoError::Convert("NetworkX multigraph not supported".into()));
    }
    let mut order = Vec::new();
    let mut index = HashMap::new();
    let mut kinds = NodeIdKinds::default();
    for n in &doc.nodes {
        let name = kinds.name(&n.id)?;
        graph_dot::intern(&name, &mut order, &mut index)?;
    }
    let mut edges = Vec::new();
    for link in &doc.links {
        let s = kinds.name(&link.source)?;
        let t = kinds.name(&link.target)?;
        let from = graph_dot::intern(&s, &mut order, &mut index)?;
        let to = graph_dot::intern(&t, &mut order, &mut index)?;
        edges.push((from, to));
    }
    let node_count = u32::try_from(order.len()).map_err(|_| IoError::TooLarge)?;
    Ok((DagWire { node_count, edges }, order))
}

/// Build node-link document from wire.
#[must_use]
pub fn networkx_node_link_from_wire(wire: &DagWire, names: Option<&[String]>) -> NetworkXNodeLink {
    let nodes = (0..wire.node_count)
        .map(|i| {
            let id = names
                .and_then(|n| n.get(i as usize))
                .cloned()
                .map(JsonValue::String)
                .unwrap_or(JsonValue::Number(i.into()));
            NetworkXNode { id }
        })
        .collect();
    let links = wire
        .edges
        .iter()
        .map(|&(a, b)| {
            let source = names
                .and_then(|n| n.get(a as usize))
                .cloned()
                .map(JsonValue::String)
                .unwrap_or(JsonValue::Number(a.into()));
            let target = names
                .and_then(|n| n.get(b as usize))
                .cloned()
                .map(JsonValue::String)
                .unwrap_or(JsonValue::Number(b.into()));
            NetworkXLink { source, target }
        })
        .collect();
    NetworkXNodeLink {
        directed: true,
        multigraph: false,
        graph: JsonValue::Object(serde_json::Map::new()),
        nodes,
        links,
    }
}

/// Parse `NetworkX` adjacency JSON into a [`Dag`].
///
/// # Errors
///
/// Undirected / malformed / cycles.
pub fn dag_from_networkx_adjacency(json: &str) -> Result<Dag, IoError> {
    dag_with_names_from_networkx_adjacency(json).map(|(dag, _names)| dag)
}

/// Parse `NetworkX` adjacency JSON into a [`Dag`] plus its node names.
///
/// Names are the document's node `id` values (stringified), in dense-id
/// order. A document with plain sequential integer ids (as emitted by
/// [`dag_to_networkx_adjacency`] with `names = None`) yields dense-index
/// strings, since the document carries no distinct name information in
/// that case.
///
/// Expects NetworkX `adjacency_data` shape: top-level `adjacency` parallel
/// to `nodes`. Documents without that field (or with length mismatch) are
/// refused — never returned as an edgeless success.
///
/// # Errors
///
/// Undirected / malformed / cycles / unsupported adjacency shape.
pub fn dag_with_names_from_networkx_adjacency(json: &str) -> Result<(Dag, Vec<String>), IoError> {
    let doc: NetworkXAdjacency =
        serde_json::from_str(json).map_err(|e| IoError::Convert(format!("json: {e}")))?;
    if !doc.directed {
        return Err(IoError::Convert("NetworkX graph must be directed".into()));
    }
    if doc.multigraph {
        return Err(IoError::Convert("NetworkX multigraph not supported".into()));
    }
    if doc.adjacency.len() != doc.nodes.len() {
        return Err(IoError::Convert(format!(
            "NetworkX adjacency length {} must equal nodes length {}",
            doc.adjacency.len(),
            doc.nodes.len()
        )));
    }
    let mut order = Vec::new();
    let mut index = HashMap::new();
    let mut kinds = NodeIdKinds::default();
    for n in &doc.nodes {
        let name = kinds.name(&n.id)?;
        graph_dot::intern(&name, &mut order, &mut index)?;
    }
    let mut edges = Vec::new();
    for (i, nbrs) in doc.adjacency.iter().enumerate() {
        let from_name = kinds.name(&doc.nodes[i].id)?;
        let from = *index.get(&from_name).ok_or_else(|| {
            IoError::Convert(format!("NetworkX adjacency missing node `{from_name}`"))
        })?;
        for nbr in nbrs {
            let to_name = kinds.name(&nbr.id)?;
            let to = graph_dot::intern(&to_name, &mut order, &mut index)?;
            edges.push((from, to));
        }
    }
    let node_count = u32::try_from(order.len()).map_err(|_| IoError::TooLarge)?;
    let dag = dag_from_wire(&DagWire { node_count, edges })?;
    Ok((dag, order))
}

/// Serialize a [`Dag`] to `NetworkX` adjacency JSON.
///
/// Emits NetworkX `adjacency_data` shape (top-level `adjacency` parallel to
/// `nodes`).
///
/// # Errors
///
/// Wire / JSON failures.
pub fn dag_to_networkx_adjacency(dag: &Dag, names: Option<&[String]>) -> Result<String, IoError> {
    let wire = dag_to_wire(dag)?;
    let mut children: HashMap<u32, Vec<u32>> = HashMap::new();
    for &(a, b) in &wire.edges {
        children.entry(a).or_default().push(b);
    }
    let nodes = (0..wire.node_count)
        .map(|i| {
            let id = names
                .and_then(|n| n.get(i as usize))
                .cloned()
                .map(JsonValue::String)
                .unwrap_or(JsonValue::Number(i.into()));
            NetworkXNode { id }
        })
        .collect();
    let adjacency = (0..wire.node_count)
        .map(|i| {
            children
                .get(&i)
                .into_iter()
                .flatten()
                .map(|&t| {
                    let id = names
                        .and_then(|n| n.get(t as usize))
                        .cloned()
                        .map(JsonValue::String)
                        .unwrap_or(JsonValue::Number(t.into()));
                    NetworkXAdjNeighbor { id }
                })
                .collect()
        })
        .collect();
    let doc = NetworkXAdjacency {
        directed: true,
        multigraph: false,
        graph: JsonValue::Object(serde_json::Map::new()),
        nodes,
        adjacency,
    };
    serde_json::to_string_pretty(&doc).map_err(|e| IoError::Convert(format!("json: {e}")))
}

/// Node-id spellings seen in one document. NetworkX keys nodes by Python value, so the
/// integer `1` and the string `"1"` are different nodes; names here are strings, so a
/// document that uses both spellings of one name cannot be represented faithfully.
#[derive(Default)]
pub(crate) struct NodeIdKinds(HashMap<String, bool>);

impl NodeIdKinds {
    pub(crate) fn name(&mut self, id: &JsonValue) -> Result<String, IoError> {
        let name = json_id_to_string(id)?;
        let numeric = id.is_number();
        if *self.0.entry(name.clone()).or_insert(numeric) != numeric {
            return Err(IoError::Convert(format!(
                "NetworkX node id `{name}` appears both as a number and as a string; \
                 these are distinct nodes and cannot share one variable name"
            )));
        }
        Ok(name)
    }
}

fn json_id_to_string(v: &JsonValue) -> Result<String, IoError> {
    match v {
        JsonValue::String(s) => Ok(s.clone()),
        JsonValue::Number(n) => Ok(n.to_string()),
        other => Err(IoError::Convert(format!("unsupported node id {other}"))),
    }
}

#[cfg(test)]
mod tests {
    use antecedent_graph::DenseNodeId;

    use super::*;

    #[test]
    fn integer_and_string_ids_of_one_spelling_are_refused_not_merged() {
        let doc = r#"{"directed": true, "multigraph": false, "graph": {},
            "nodes": [{"id": 1}, {"id": "1"}], "links": []}"#;
        let error = dag_from_networkx_node_link(doc).unwrap_err().to_string();
        assert!(error.contains("both as a number and as a string"), "{error}");
        let same_kind = r#"{"directed": true, "multigraph": false, "graph": {},
            "nodes": [{"id": "a"}, {"id": "b"}], "links": [{"source": "a", "target": "b"}]}"#;
        assert_eq!(dag_from_networkx_node_link(same_kind).unwrap().node_count(), 2);
    }

    #[test]
    fn node_link_round_trip() {
        let mut dag = Dag::with_variables(2);
        dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let names = vec!["X".into(), "Y".into()];
        let s = dag_to_networkx_node_link(&dag, Some(&names)).unwrap();
        let back = dag_from_networkx_node_link(&s).unwrap();
        assert_eq!(back.node_count(), 2);
        assert!(back.reaches(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)));
    }

    #[test]
    fn node_link_with_names_round_trip_preserves_labels() {
        let mut dag = Dag::with_variables(2);
        dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let names = vec!["X".to_string(), "Y".to_string()];
        let s = dag_to_networkx_node_link(&dag, Some(&names)).unwrap();
        let (back, back_names) = dag_with_names_from_networkx_node_link(&s).unwrap();
        assert_eq!(back.node_count(), 2);
        assert_eq!(back_names, names);
    }

    #[test]
    fn node_link_with_names_nameless_falls_back_to_dense_index() {
        let mut dag = Dag::with_variables(2);
        dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let s = dag_to_networkx_node_link(&dag, None).unwrap();
        let (_back, names) = dag_with_names_from_networkx_node_link(&s).unwrap();
        assert_eq!(names, vec!["0".to_string(), "1".to_string()]);
    }

    #[test]
    fn rejects_undirected_node_link() {
        let json =
            r#"{"directed":false,"multigraph":false,"graph":{},"nodes":[{"id":0}],"links":[]}"#;
        assert!(dag_from_networkx_node_link(json).is_err());
    }

    #[test]
    fn adjacency_round_trip() {
        let mut dag = Dag::with_variables(2);
        dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let s = dag_to_networkx_adjacency(&dag, None).unwrap();
        let back = dag_from_networkx_adjacency(&s).unwrap();
        assert!(back.reaches(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)));
    }

    #[test]
    fn genuine_networkx_adjacency_document_loads_edges() {
        // NetworkX `adjacency_data` shape: parallel top-level `adjacency`.
        let json = r#"{
  "directed": true,
  "multigraph": false,
  "graph": [],
  "nodes": [{"id": "Z"}, {"id": "X"}, {"id": "Y"}],
  "adjacency": [
    [{"id": "X"}],
    [{"id": "Y"}],
    []
  ]
}"#;
        let (dag, names) = dag_with_names_from_networkx_adjacency(json).unwrap();
        assert_eq!(names, vec!["Z".to_string(), "X".to_string(), "Y".to_string()]);
        assert_eq!(dag.node_count(), 3);
        assert!(dag.reaches(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)));
        assert!(dag.reaches(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)));
    }

    #[test]
    fn adjacency_without_parallel_array_is_refused() {
        // Nested-in-node adjacency (non-NetworkX) must not load as edgeless success.
        let json = r#"{
  "directed": true,
  "multigraph": false,
  "graph": {},
  "nodes": [{"id": 0, "adjacency": [{"1": {}}]}, {"id": 1, "adjacency": []}]
}"#;
        let err = dag_from_networkx_adjacency(json).unwrap_err();
        assert!(matches!(err, IoError::Convert(_)));
    }

    #[test]
    fn adjacency_with_names_round_trip_preserves_labels() {
        let mut dag = Dag::with_variables(2);
        dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let names = vec!["a".to_string(), "b".to_string()];
        let s = dag_to_networkx_adjacency(&dag, Some(&names)).unwrap();
        let (back, back_names) = dag_with_names_from_networkx_adjacency(&s).unwrap();
        assert_eq!(back.node_count(), 2);
        assert_eq!(back_names, names);
    }

    #[test]
    fn adjacency_with_names_nameless_falls_back_to_dense_index() {
        let mut dag = Dag::with_variables(2);
        dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let s = dag_to_networkx_adjacency(&dag, None).unwrap();
        let (_back, names) = dag_with_names_from_networkx_adjacency(&s).unwrap();
        assert_eq!(names, vec!["0".to_string(), "1".to_string()]);
    }
}
