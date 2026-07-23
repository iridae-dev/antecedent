//! `NetworkX`-compatible JSON graph interchange (`node_link` / adjacency).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;

use antecedent_graph::Dag;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::convert::{dag_from_wire, dag_to_wire};
use crate::error::IoError;
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
    dag_from_wire(&dag_wire_from_networkx_node_link(json)?)
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
        intern(&name, &mut order, &mut index)?;
    }
    let mut edges = Vec::new();
    for link in &doc.links {
        let s = json_id_to_string(&link.source)?;
        let t = json_id_to_string(&link.target)?;
        let from = intern(&s, &mut order, &mut index)?;
        let to = intern(&t, &mut order, &mut index)?;
        edges.push((from, to));
    }
    Ok(DagWire { node_count: u32::try_from(order.len()).map_err(|_| IoError::TooLarge)?, edges })
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
    let doc: NetworkXAdjacency =
        serde_json::from_str(json).map_err(|e| IoError::Convert(format!("json: {e}")))?;
    if !doc.directed {
        return Err(IoError::Convert("NetworkX graph must be directed".into()));
    }
    let mut order = Vec::new();
    let mut index = HashMap::new();
    for n in &doc.nodes {
        let name = json_id_to_string(&n.id)?;
        intern(&name, &mut order, &mut index)?;
    }
    let mut edges = Vec::new();
    for n in &doc.nodes {
        let from_name = json_id_to_string(&n.id)?;
        let from = *index.get(&from_name).unwrap();
        for adj in &n.adjacency {
            for key in adj.keys() {
                let to = intern(key, &mut order, &mut index)?;
                edges.push((from, to));
            }
        }
    }
    dag_from_wire(&DagWire {
        node_count: u32::try_from(order.len()).map_err(|_| IoError::TooLarge)?,
        edges,
    })
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

fn intern(
    name: &str,
    order: &mut Vec<String>,
    index: &mut HashMap<String, u32>,
) -> Result<u32, IoError> {
    if let Some(&id) = index.get(name) {
        return Ok(id);
    }
    let id = u32::try_from(order.len()).map_err(|_| IoError::TooLarge)?;
    order.push(name.to_owned());
    index.insert(name.to_owned(), id);
    Ok(id)
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
}
