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

/// `NetworkX` `adjacency_data` subset.
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
    /// Nodes with adjacency maps.
    pub nodes: Vec<NetworkXAdjNode>,
}

/// Adjacency node.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct NetworkXAdjNode {
    /// Id.
    pub id: JsonValue,
    /// Out-neighbors → attr object (attrs ignored).
    #[serde(default)]
    pub adjacency: Vec<HashMap<String, JsonValue>>,
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
    for n in &doc.nodes {
        let name = json_id_to_string(&n.id)?;
        graph_dot::intern(&name, &mut order, &mut index)?;
    }
    let mut edges = Vec::new();
    for link in &doc.links {
        let s = json_id_to_string(&link.source)?;
        let t = json_id_to_string(&link.target)?;
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
/// # Errors
///
/// Undirected / malformed / cycles.
pub fn dag_with_names_from_networkx_adjacency(json: &str) -> Result<(Dag, Vec<String>), IoError> {
    let doc: NetworkXAdjacency =
        serde_json::from_str(json).map_err(|e| IoError::Convert(format!("json: {e}")))?;
    if !doc.directed {
        return Err(IoError::Convert("NetworkX graph must be directed".into()));
    }
    let mut order = Vec::new();
    let mut index = HashMap::new();
    for n in &doc.nodes {
        let name = json_id_to_string(&n.id)?;
        graph_dot::intern(&name, &mut order, &mut index)?;
    }
    let mut edges = Vec::new();
    for n in &doc.nodes {
        let from_name = json_id_to_string(&n.id)?;
        let from = *index.get(&from_name).unwrap();
        for adj in &n.adjacency {
            for key in adj.keys() {
                let to = graph_dot::intern(key, &mut order, &mut index)?;
                edges.push((from, to));
            }
        }
    }
    let node_count = u32::try_from(order.len()).map_err(|_| IoError::TooLarge)?;
    let dag = dag_from_wire(&DagWire { node_count, edges })?;
    Ok((dag, order))
}

/// Serialize a [`Dag`] to `NetworkX` adjacency JSON.
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
            let adjacency = children
                .get(&i)
                .into_iter()
                .flatten()
                .map(|&t| {
                    let key = names
                        .and_then(|n| n.get(t as usize))
                        .cloned()
                        .unwrap_or_else(|| t.to_string());
                    let mut m = HashMap::new();
                    m.insert(key, JsonValue::Object(serde_json::Map::new()));
                    m
                })
                .collect();
            NetworkXAdjNode { id, adjacency }
        })
        .collect();
    let doc = NetworkXAdjacency {
        directed: true,
        multigraph: false,
        graph: JsonValue::Object(serde_json::Map::new()),
        nodes,
    };
    serde_json::to_string_pretty(&doc).map_err(|e| IoError::Convert(format!("json: {e}")))
}

pub(crate) fn json_id_to_string(v: &JsonValue) -> Result<String, IoError> {
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
