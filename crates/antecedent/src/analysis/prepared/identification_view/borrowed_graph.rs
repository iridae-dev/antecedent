//! Borrowed graph records in the existing explicit identification read model.
use antecedent_core::NodeRef;
use antecedent_graph::{Dag, DenseNodeId, Pag, TemporalDag};
use serde::{
    Serialize,
    ser::{SerializeSeq, SerializeStruct},
};

pub(super) struct DagView<'a>(pub(super) &'a Dag);
impl Serialize for DagView<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut wire = s.serialize_struct("DagWire", 2)?;
        wire.serialize_field(
            "node_count",
            &u32::try_from(self.0.node_count()).map_err(serde::ser::Error::custom)?,
        )?;
        wire.serialize_field("edges", &DagEdges(self.0))?;
        wire.end()
    }
}
struct DagEdges<'a>(&'a Dag);
impl Serialize for DagEdges<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut seq = s.serialize_seq(Some(self.0.edges().count()))?;
        for edge in self.0.edges() {
            let (a, b) = edge.parent_child().ok_or_else(|| {
                serde::ser::Error::custom("non-directed edge in DAG wire encoding")
            })?;
            seq.serialize_element(&(a.raw(), b.raw()))?;
        }
        seq.end()
    }
}
pub(super) struct PagView<'a>(pub(super) &'a Pag);
impl Serialize for PagView<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut wire = s.serialize_struct("PagWire", 2)?;
        wire.serialize_field(
            "node_count",
            &u32::try_from(self.0.node_count()).map_err(serde::ser::Error::custom)?,
        )?;
        wire.serialize_field("edges", &PagEdges(self.0))?;
        wire.end()
    }
}
fn endpoint(mark: antecedent_graph::Endpoint) -> antecedent_io::EndpointWire {
    match mark {
        antecedent_graph::Endpoint::Tail => antecedent_io::EndpointWire::Tail,
        antecedent_graph::Endpoint::Arrow => antecedent_io::EndpointWire::Arrow,
        antecedent_graph::Endpoint::Circle => antecedent_io::EndpointWire::Circle,
        antecedent_graph::Endpoint::Conflict => antecedent_io::EndpointWire::Conflict,
    }
}
struct PagEdges<'a>(&'a Pag);
impl Serialize for PagEdges<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut count = 0;
        for index in 0..self.0.node_count() {
            let a = DenseNodeId::from_raw(u32::try_from(index).map_err(serde::ser::Error::custom)?);
            count += self.0.neighbors(a).filter(|(b, _, _)| a.raw() < b.raw()).count();
        }
        let mut seq = s.serialize_seq(Some(count))?;
        for index in 0..self.0.node_count() {
            let a = DenseNodeId::from_raw(u32::try_from(index).map_err(serde::ser::Error::custom)?);
            for (b, at_a, at_b) in self.0.neighbors(a) {
                if a.raw() < b.raw() {
                    seq.serialize_element(&antecedent_io::MarkedEdgeWire {
                        a: a.raw(),
                        b: b.raw(),
                        at_a: endpoint(at_a),
                        at_b: endpoint(at_b),
                    })?;
                }
            }
        }
        seq.end()
    }
}
pub(super) struct TemporalCompletion<'a>(
    pub(super) &'a antecedent_identify::TemporalCompletionGraph,
);
impl Serialize for TemporalCompletion<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self.0 {
            antecedent_identify::TemporalCompletionGraph::Dag(graph) => {
                let mut wire = s.serialize_struct("TemporalGraph", 2)?;
                wire.serialize_field("kind", "temporal_dag")?;
                wire.serialize_field("graph", &TemporalDagView(graph))?;
                wire.end()
            }
            antecedent_identify::TemporalCompletionGraph::Mag(graph) => {
                let mut wire = s.serialize_struct("TemporalGraph", 3)?;
                wire.serialize_field("kind", "temporal_mag")?;
                wire.serialize_field("nodes", &Nodes(graph.nodes()))?;
                wire.serialize_field("edges", &MagEdges(graph))?;
                wire.end()
            }
        }
    }
}
struct TemporalDagView<'a>(&'a TemporalDag);
impl Serialize for TemporalDagView<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut wire = s.serialize_struct("DiscoveryTemporalGraphWire", 3)?;
        wire.serialize_field("kind", "temporal_dag")?;
        wire.serialize_field("nodes", &TemporalNodes(self.0.nodes()))?;
        wire.serialize_field("directed", &TemporalEdges(self.0))?;
        wire.end()
    }
}
struct TemporalNodes<'a>(&'a [NodeRef]);
impl Serialize for TemporalNodes<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut seq = s.serialize_seq(Some(self.0.len()))?;
        for node in self.0 {
            match node {
                NodeRef::Lagged { variable, lag } => {
                    seq.serialize_element(&antecedent_io::TemporalNodeKeyWire {
                        variable: variable.raw(),
                        lag: lag.raw(),
                    })?
                }
                _ => return Err(serde::ser::Error::custom("temporal wire requires Lagged nodes")),
            }
        }
        seq.end()
    }
}
struct TemporalEdges<'a>(&'a TemporalDag);
impl Serialize for TemporalEdges<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let count = self.0.edges().filter(|e| e.parent_child().is_some()).count();
        let mut seq = s.serialize_seq(Some(count))?;
        for edge in self.0.edges() {
            if let Some((a, b)) = edge.parent_child() {
                seq.serialize_element(&(a.raw(), b.raw()))?;
            }
        }
        seq.end()
    }
}
struct Nodes<'a>(&'a [NodeRef]);
impl Serialize for Nodes<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut seq = s.serialize_seq(Some(self.0.len()))?;
        for node in self.0 {
            seq.serialize_element(&Node(node))?;
        }
        seq.end()
    }
}
struct Node<'a>(&'a NodeRef);
impl Serialize for Node<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut wire = s.serialize_struct(
            "NodeRef",
            match self.0 {
                NodeRef::Static(_) => 2,
                _ => 3,
            },
        )?;
        match self.0 {
            NodeRef::Static(v) => {
                wire.serialize_field("kind", "static")?;
                wire.serialize_field("variable", &v.raw())?;
            }
            NodeRef::Lagged { variable, lag } => {
                wire.serialize_field("kind", "lagged")?;
                wire.serialize_field("variable", &variable.raw())?;
                wire.serialize_field("lag", &lag.raw())?;
            }
            NodeRef::Unfolded { variable, offset } => {
                wire.serialize_field("kind", "unfolded")?;
                wire.serialize_field("variable", &variable.raw())?;
                wire.serialize_field("offset", offset)?;
            }
            NodeRef::Context { variable, environment } => {
                wire.serialize_field("kind", "context")?;
                wire.serialize_field("variable", &variable.raw())?;
                wire.serialize_field("environment", &environment.map(|v| v.raw()))?;
            }
        }
        wire.end()
    }
}
struct MagEdges<'a>(&'a antecedent_graph::TemporalPag);
impl Serialize for MagEdges<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut count = 0;
        for index in 0..self.0.node_count() {
            let a = DenseNodeId::from_raw(u32::try_from(index).map_err(serde::ser::Error::custom)?);
            count += self.0.neighbors(a).filter(|(b, _, _)| a.raw() < b.raw()).count();
        }
        let mut seq = s.serialize_seq(Some(count))?;
        for index in 0..self.0.node_count() {
            let a = DenseNodeId::from_raw(u32::try_from(index).map_err(serde::ser::Error::custom)?);
            for (b, _, _) in self.0.neighbors(a) {
                if a.raw() < b.raw() {
                    let edge = self.0.edge_between(a, b).ok_or_else(|| {
                        serde::ser::Error::custom("retained temporal MAG edge is missing")
                    })?;
                    seq.serialize_element(&MagEdge(&edge))?;
                }
            }
        }
        seq.end()
    }
}
struct MagEdge<'a>(&'a antecedent_graph::MarkedEdge);
impl Serialize for MagEdge<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let e = self.0;
        let mut wire = s.serialize_struct("MarkedEdge", 5)?;
        wire.serialize_field("a", &e.a.raw())?;
        wire.serialize_field("b", &e.b.raw())?;
        wire.serialize_field("at_a", &endpoint(e.at_a))?;
        wire.serialize_field("at_b", &endpoint(e.at_b))?;
        wire.serialize_field(
            "middle",
            match e.middle {
                antecedent_graph::MiddleMark::Unknown => "unknown",
                antecedent_graph::MiddleMark::Left => "left",
                antecedent_graph::MiddleMark::Right => "right",
                antecedent_graph::MiddleMark::Both => "both",
                antecedent_graph::MiddleMark::Empty => "empty",
            },
        )?;
        wire.end()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{Lag, VariableId};
    use antecedent_graph::{Endpoint, MarkedEdge, MiddleMark, TemporalPag};
    #[test]
    fn borrowed_static_and_temporal_graphs_preserve_existing_coordinate_order_and_marks() {
        let mut dag = Dag::with_variables(3);
        dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
        dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        assert_eq!(
            serde_json::to_value(DagView(&dag)).unwrap(),
            serde_json::to_value(antecedent_io::dag_to_wire(&dag).unwrap()).unwrap()
        );
        let pag = antecedent_io::pag_from_wire(&antecedent_io::PagWire {
            node_count: 3,
            edges: vec![antecedent_io::MarkedEdgeWire {
                a: 0,
                b: 2,
                at_a: antecedent_io::EndpointWire::Circle,
                at_b: antecedent_io::EndpointWire::Arrow,
            }],
        })
        .unwrap();
        assert_eq!(
            serde_json::to_value(PagView(&pag)).unwrap(),
            serde_json::to_value(antecedent_io::pag_to_wire(&pag).unwrap()).unwrap()
        );
        let mut temporal = TemporalDag::empty();
        let a = temporal.add_lagged(VariableId::from_raw(2), Lag::from_raw(1)).unwrap();
        let b = temporal.add_lagged(VariableId::from_raw(0), Lag::from_raw(0)).unwrap();
        temporal.insert_directed(a, b).unwrap();
        let graph = antecedent_identify::TemporalCompletionGraph::Dag(temporal);
        assert_eq!(
            serde_json::to_value(TemporalCompletion(&graph)).unwrap(),
            super::super::temporal_graph(&graph).unwrap()
        );
        let mut temporal = TemporalPag::empty();
        let a = temporal.add_lagged(VariableId::from_raw(2), Lag::from_raw(1)).unwrap();
        let b = temporal.add_lagged(VariableId::from_raw(0), Lag::from_raw(0)).unwrap();
        temporal
            .insert_marked(MarkedEdge {
                a,
                b,
                at_a: Endpoint::Circle,
                at_b: Endpoint::Arrow,
                middle: MiddleMark::Left,
            })
            .unwrap();
        let graph = antecedent_identify::TemporalCompletionGraph::Mag(temporal);
        assert_eq!(
            serde_json::to_value(TemporalCompletion(&graph)).unwrap(),
            super::super::temporal_graph(&graph).unwrap()
        );
    }
}
