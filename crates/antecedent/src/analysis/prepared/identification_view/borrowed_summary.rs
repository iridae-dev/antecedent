//! Heap-free summary projection for explicit streamed proof materialization.
use super::*;
use serde::{
    Serialize, Serializer,
    ser::{SerializeMap, SerializeStruct},
};

#[derive(Clone)]
enum EntryValue<'a> {
    Null,
    Text(&'a str),
    Shared(Arc<str>),
    Usize(usize),
    U64(u64),
    U32(u32),
    I32(i32),
    Float(f64),
    Bool(bool),
    Summary(&'a PreparedStudy, Node<'a>),
    Indexer(&'a antecedent_core::TemporalIndexer),
}
impl Serialize for EntryValue<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Null => serializer.serialize_none(),
            Self::Text(v) => serializer.serialize_str(v),
            Self::Shared(v) => serializer.serialize_str(v),
            Self::Usize(v) => v.serialize(serializer),
            Self::U64(v) => serializer.serialize_u64(*v),
            Self::U32(v) => serializer.serialize_u32(*v),
            Self::I32(v) => serializer.serialize_i32(*v),
            Self::Float(v) => serializer.serialize_f64(*v),
            Self::Bool(v) => serializer.serialize_bool(*v),
            Self::Summary(study, node) => {
                BorrowedSummary(PreparedIdentificationView { study, node: *node })
                    .serialize(serializer)
            }
            Self::Indexer(i) => {
                let mut map = serializer.serialize_struct("TemporalNamespace", 3)?;
                map.serialize_field("history", &i.history())?;
                map.serialize_field("horizon", &i.horizon())?;
                map.serialize_field("variable_count", &i.variable_count())?;
                map.end()
            }
        }
    }
}
macro_rules! convert {
    ($type:ty,$variant:ident) => {
        impl<'a> From<$type> for EntryValue<'a> {
            fn from(value: $type) -> Self {
                Self::$variant(value)
            }
        }
    };
}
convert!(usize, Usize);
convert!(u64, U64);
convert!(u32, U32);
convert!(i32, I32);
convert!(f64, Float);
convert!(bool, Bool);
impl<'a> From<&'a str> for EntryValue<'a> {
    fn from(v: &'a str) -> Self {
        Self::Text(v)
    }
}
impl<'a> From<Arc<str>> for EntryValue<'a> {
    fn from(v: Arc<str>) -> Self {
        Self::Shared(v)
    }
}
impl<'a, T: Into<EntryValue<'a>>> From<Option<T>> for EntryValue<'a> {
    fn from(v: Option<T>) -> Self {
        v.map_or(Self::Null, Into::into)
    }
}

struct Fields<'a> {
    entries: [Option<(&'static str, EntryValue<'a>)>; 32],
    length: usize,
}
impl<'a> Fields<'a> {
    fn new() -> Self {
        Self { entries: std::array::from_fn(|_| None), length: 0 }
    }
    fn put(&mut self, name: &'static str, value: impl Into<EntryValue<'a>>) {
        let value = value.into();
        for entry in self.entries[..self.length].iter_mut().flatten() {
            if entry.0 == name {
                entry.1 = value;
                return;
            }
        }
        assert!(self.length < self.entries.len(), "bounded identification summary field limit");
        self.entries[self.length] = Some((name, value));
        self.length += 1;
    }
    fn get(&self, name: &str) -> EntryValue<'a> {
        self.entries[..self.length]
            .iter()
            .flatten()
            .find(|e| e.0 == name)
            .map_or(EntryValue::Null, |e| e.1.clone())
    }
    fn status(&mut self, status: antecedent_core::IdentificationStatus) {
        self.put("status", status.as_str());
        self.put("status_domain", "identification");
        self.put("status_unavailable", EntryValue::Null);
    }
    fn envelope<G>(&mut self, e: &antecedent_identify::IdentificationEnvelope<G>) {
        self.status(e.status);
        self.put("identified_weight", e.identified_weight.0);
        self.put("unidentified_weight", e.unidentified_weight.0);
        self.put("truncated_completions", e.truncated_completions);
        self.put("case_count", e.cases.len());
        self.put("has_invariant", e.invariant.is_some());
    }
    fn transport(&mut self, p: &antecedent_identify::TransportIdentification) {
        let outcome = p.outcome();
        self.put("status", outcome.kind.as_str());
        self.put("status_domain", "transport_outcome");
        self.put("status_unavailable", EntryValue::Null);
        self.put("reason", outcome.reason);
        self.put("location_factor", outcome.location.as_ref().and_then(|l| l.factor.clone()));
        self.put(
            "location_node_count",
            outcome.location.as_ref().map_or(0, |l| l.graph_nodes.len()),
        );
    }
}
impl Serialize for Fields<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // Match the existing serde_json::Value object's lexical key order using
        // a fixed stack index, without allocating a map or copying record text.
        let mut indices: [usize; 32] = std::array::from_fn(|index| index);
        indices[..self.length].sort_unstable_by_key(|&i| self.entries[i].as_ref().unwrap().0);
        let mut map = serializer.serialize_map(Some(self.length))?;
        for &i in &indices[..self.length] {
            let (name, value) = self.entries[i].as_ref().unwrap();
            map.serialize_entry(name, value)?;
        }
        map.end()
    }
}

pub(super) struct BorrowedSummary<'a>(PreparedIdentificationView<'a>);
impl<'a> BorrowedSummary<'a> {
    pub(super) fn new(view: &PreparedIdentificationView<'a>) -> Self {
        Self(PreparedIdentificationView { study: view.study, node: view.node })
    }
    fn fields(&self) -> Fields<'a> {
        let view = &self.0;
        let study = view.study;
        let mut fields = Fields::new();
        fields.put("kind", view.node.kind());
        fields.put("children", view.child_count());
        fields.put("status", EntryValue::Null);
        fields.put("status_unavailable", "composite_identification");
        match view.node {
            Node::Root => {
                fields.put("graph_class", study.study().graph.class().as_str());
                fields.put(
                    "cache_kind",
                    study.study().identification.as_ref().map(IdentificationCache::kind),
                );
                fields.put("max_completions", study.study().declared.max_completions);
                fields.put("method", study.plan().binding().identifier.named().map(|i| i.as_str()));
                if let Some(plan) = study.plan().lowered() {
                    let node = Node::Layer(plan.identification_layer());
                    let nested =
                        BorrowedSummary(PreparedIdentificationView { study, node }).fields();
                    fields.put("layer", nested.get("layer"));
                    fields.put("status_unavailable", nested.get("status_unavailable"));
                    fields.put("layer_details", EntryValue::Summary(study, node));
                }
                if let Some(cache) = study.study().identification.as_ref() {
                    let node = Node::Cache(cache);
                    let nested =
                        BorrowedSummary(PreparedIdentificationView { study, node }).fields();
                    fields.put("status", nested.get("status"));
                    fields.put("status_unavailable", nested.get("status_unavailable"));
                    fields.put("status_domain", nested.get("status_domain"));
                    fields.put("retained", EntryValue::Summary(study, node));
                }
            }
            Node::Layer(layer) => {
                let (name, reason) = match layer {
                    IdentificationLayer::Retained(_) => ("retained", "composite_identification"),
                    IdentificationLayer::Deferred { .. } => ("deferred", "deferred_identification"),
                    IdentificationLayer::PerExecution { .. } => {
                        ("per_execution", "per_execution_identification")
                    }
                    IdentificationLayer::NonGraphical(_) => {
                        ("non_graphical", "non_graphical_identification")
                    }
                    IdentificationLayer::NotClaimed { .. } => {
                        ("not_claimed", "identification_not_claimed")
                    }
                };
                fields.put("layer", name);
                fields.put("status_unavailable", reason);
                match layer {
                    IdentificationLayer::Deferred { reason, .. }
                    | IdentificationLayer::PerExecution { reason } => fields.put("reason", *reason),
                    IdentificationLayer::NotClaimed { refusal } => {
                        fields.put("refusal", refusal.clone())
                    }
                    IdentificationLayer::NonGraphical(proof) => {
                        fields.put("non_graphical", proof.as_str())
                    }
                    _ => {}
                }
            }
            Node::Cache(cache) => match cache {
                IdentificationCache::Static(c) => fields.status(c.identification.status),
                IdentificationCache::Pag(c) => fields.envelope(&c.envelope),
                IdentificationCache::Cpdag(c) => fields.envelope(&c.envelope),
                IdentificationCache::TemporalClass(c) => {
                    fields.envelope(&c.envelope.envelope);
                    fields.put("horizon_count", c.by_horizon.len());
                }
                IdentificationCache::Temporal(c) => fields.put("horizon_count", c.by_horizon.len()),
                IdentificationCache::GraphPosterior(c) => {
                    fields.put("sample_count", c.graphs.n_samples);
                    fields.put("atom_count", c.atoms.len());
                    fields.put("class_atom_count", c.class_atoms.len());
                    fields.put("mass_available", "per_sample");
                }
                IdentificationCache::DbnPosterior(c) => {
                    fields.put("sample_count", c.graphs.n_samples);
                    fields.put("atom_count", c.atoms.len());
                    fields.put("horizon_demotion_count", c.horizon_demotions.len());
                    fields.put("mass_available", "per_sample");
                }
                IdentificationCache::TemporalClassPosterior(c) => {
                    fields.put("sample_count", c.graphs.n_samples);
                    fields.put("class_atom_count", c.class_atoms.len());
                    fields.put("mass_available", "per_sample");
                }
                IdentificationCache::Transport(p) => fields.transport(p),
            },
            Node::Transport(p) => fields.transport(p),
            Node::Proof(proof, indexer, weight) => {
                fields.status(proof.status);
                fields.put("assumption_count", proof.required_assumptions.len());
                fields.put("derivation_step_count", proof.derivation.steps.len());
                fields.put("estimand_count", proof.estimands.len());
                fields.put("estimand_claim_count", proof.estimand_claims.len());
                fields.put("candidates_examined", proof.performance.candidates_examined);
                fields.put("sets_returned", proof.performance.sets_returned);
                fields.put("weight", weight);
                fields.put("temporal_namespace", indexer.map(EntryValue::Indexer));
            }
            Node::Pag(e) => fields.envelope(e),
            Node::Cpdag(e) => fields.envelope(e),
            Node::TemporalEnvelope(e) | Node::HorizonEnvelope(_, e) => fields.envelope(&e.envelope),
            Node::Horizon(entry) => {
                fields.put("horizon", entry.horizon);
                fields.status(entry.identification.status);
            }
            Node::ClassAtom(a) => {
                fields.put("key", a.key);
                fields.status(a.identification.status);
                fields.put("identified_weight", a.identified_weight);
                fields.put("truncated_completions", a.truncated_completions);
            }
            Node::TemporalAtom(a) => {
                fields.put("key", a.key);
                fields.status(a.identification.status);
                fields.put("identified_weight", a.identified_weight);
                fields.put("truncated_completions", a.truncated_completions);
            }
            Node::GraphAtom(a) => {
                fields.put("key", a.key);
                fields.status(a.identification.status);
            }
            Node::DbnAtom(a) => {
                fields.put("key", a.key);
                fields.status(a.identification.status);
            }
            Node::ClassCase(a, index) => {
                fields.status(a.cases[index].status);
                fields.put("weight", a.cases[index].weight);
            }
            Node::PagCase(c) => {
                fields.status(c.result.status);
                fields.put("weight", c.weight.0);
            }
            Node::DagCase(c) => {
                fields.status(c.result.status);
                fields.put("weight", c.weight.0);
            }
            Node::TemporalCase(c, indexer) => {
                fields.status(c.result.status);
                fields.put("weight", c.weight.0);
                fields.put("temporal_namespace", indexer.map(EntryValue::Indexer));
            }
            Node::Absent => fields.put("status_unavailable", "component_not_identified"),
            Node::Sample(graphs, index) => {
                fields.put("key", graphs.graph_keys.get(index).copied());
                fields.put("weight", graphs.weights.get(index).copied());
                fields.put(
                    "identified",
                    graphs
                        .identified
                        .get(index)
                        .map(|flag| matches!(flag, antecedent_prob::GraphIdentFlag::Identified)),
                );
            }
            _ => {}
        }
        if let Node::HorizonEnvelope(horizon, _) = view.node {
            fields.put("horizon", horizon);
        }
        fields
    }
}
impl Serialize for BorrowedSummary<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.fields().serialize(serializer)
    }
}
