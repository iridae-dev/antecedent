//! Stream the retained identification hierarchy directly into one admitted buffer.
use super::borrowed_graph::{DagView, PagView, TemporalCompletion};
use super::borrowed_summary::BorrowedSummary;
use super::*;
use antecedent_io_core::admitted_json::{AdmittedJson, serialize_admitted};
use antecedent_io_core::analysis_wire::BorrowedIdentificationWire;
use antecedent_io_core::transport_interference_wire::BorrowedTransportIdentificationWire;
use serde::{
    Serialize,
    ser::{SerializeMap, SerializeSeq, SerializeStruct},
};

impl PreparedIdentificationView<'_> {
    /// Explicit full proof conversion into an exactly admitted native JSON owner.
    ///
    /// Retained proof trees, compiled declarations, graph cases and schema names
    /// are serialized by borrowing native storage. The output lease stays with
    /// the returned owner through its final consumer; no heap Value tree or
    /// copied identification DTO is constructed before admission.
    /// # Errors
    /// Cancellation, admission/allocation failure, or canonical wire conversion
    /// failure of the retained proof. Missing producer state is an implementation error.
    pub fn materialize_admitted(
        &self,
        context: &antecedent_core::ExecutionContext,
    ) -> Result<AdmittedJson, crate::CausalError> {
        if context.cancellation.is_cancelled() {
            return Err(crate::CausalError::Cancelled { stage: "identification_materialize" });
        }
        let program =
            self.study.program_cache().get().ok_or_else(|| crate::CausalError::Compile {
                message: "prepared identification has no retained compiled query wire".into(),
            })?;
        serialize_admitted(
            &Tree {
                view: PreparedIdentificationView { study: self.study, node: self.node },
                encoded: program.retained_query_wire(),
                names: true,
                context,
            },
            context,
        )
        .map_err(|error| match error {
            antecedent_io_core::IoError::StreamCancelled => {
                crate::CausalError::Cancelled { stage: "identification_materialize" }
            }
            antecedent_io_core::IoError::StreamBudgetRefused { message } => {
                crate::CausalError::Resource { message }
            }
            other => crate::CausalError::from(other),
        })
    }
}

struct Tree<'a> {
    view: PreparedIdentificationView<'a>,
    encoded: &'a antecedent_io::CausalQueryWire,
    names: bool,
    context: &'a antecedent_core::ExecutionContext,
}
impl Serialize for Tree<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let view = &self.view;
        let mut object = s.serialize_map(None)?;
        // JSON object order is not a scientific identity; array/record order is.
        // Existing canonical wire serializers remain the authority for proof fields.
        match view.node {
            Node::Proof(proof, _, _) => {
                let wire = BorrowedIdentificationWire::new(proof, view.study.query(), self.encoded)
                    .map_err(serde::ser::Error::custom)?;
                object.serialize_entry("identification", &wire)?;
            }
            Node::Transport(proof) => object
                .serialize_entry("transport", &BorrowedTransportIdentificationWire::new(proof))?,
            Node::Samples(samples) => {
                object.serialize_entry("edge_marginals", &samples.edge_marginals.as_deref())?;
                object.serialize_entry(
                    "orientation_marginals",
                    &samples.orientation_marginals.as_deref(),
                )?;
            }
            Node::Layer(IdentificationLayer::NonGraphical(proof)) => {
                object.serialize_entry("non_graphical", &NonGraphical(proof))?
            }
            Node::PagCase(case) => object.serialize_entry("graph", &PagView(&case.graph))?,
            Node::DagCase(case) => object.serialize_entry("graph", &DagView(&case.graph))?,
            Node::TemporalCase(case, _) => {
                // The completion digest is traversed in place: no sorted copy is allocated outside admission.
                let fingerprint = case
                    .graph
                    .fingerprint_without_allocation(&|| self.context.cancellation.is_cancelled())
                    .ok_or_else(|| {
                        serde::ser::Error::custom("native JSON serialization cancelled")
                    })?;
                object.serialize_entry("fingerprint", &fingerprint)?;
                object.serialize_entry("graph", &TemporalCompletion(&case.graph))?;
            }
            Node::Horizon(entry) => {
                object.serialize_entry("selected_estimand", &Selected(&entry.estimand))?
            }
            Node::Selected(_, estimand) => {
                object.serialize_entry("selected_estimand", &Selected(estimand))?
            }
            Node::ClassCase(atom, index) => object.serialize_entry(
                "selected_estimand",
                &atom.cases[index].estimand.as_ref().map(Selected),
            )?,
            Node::ClassAtom(atom) => {
                object.serialize_entry("invariant", &atom.invariant.as_ref().map(Selected))?
            }
            Node::TemporalAtom(atom) => {
                object.serialize_entry("invariant", &atom.invariant.as_ref().map(Selected))?
            }
            Node::GraphAtom(atom) => {
                object.serialize_entry("selected_estimand", &Selected(&atom.estimand))?
            }
            Node::DbnAtom(atom) => {
                object.serialize_entry("selected_estimand", &Selected(&atom.estimand))?
            }
            Node::Cache(IdentificationCache::Static(cache)) => {
                object.serialize_entry("selected_estimand", &Selected(&cache.estimand))?
            }
            Node::Pag(envelope) => envelope_fields(&mut object, envelope)?,
            Node::Cpdag(envelope) => envelope_fields(&mut object, envelope)?,
            Node::TemporalEnvelope(envelope) | Node::HorizonEnvelope(_, envelope) => {
                envelope_fields(&mut object, &envelope.envelope)?
            }
            Node::Cache(IdentificationCache::DbnPosterior(cache)) => {
                object.serialize_entry("demotion", &Demotion(&cache.identify_demotion))?;
                object.serialize_entry(
                    "horizon_demotions",
                    &HorizonDemotions(&cache.horizon_demotions),
                )?;
            }
            _ => {}
        }
        object.serialize_entry("summary", &BorrowedSummary::new(view))?;
        object.serialize_entry("children", &Children(self))?;
        if self.names {
            object.serialize_entry("names", &Names(view.study.schema()))?;
        }
        object.end()
    }
}
struct Children<'a, 'b>(&'a Tree<'b>);
impl Serialize for Children<'_, '_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let source = self.0;
        let mut seq = s.serialize_seq(Some(source.view.child_count()))?;
        for index in 0..source.view.child_count() {
            let child = source.view.child(index).ok_or_else(|| {
                serde::ser::Error::custom("retained identification child is missing")
            })?;
            seq.serialize_element(&Tree {
                view: child,
                encoded: source.encoded,
                names: false,
                context: source.context,
            })?;
        }
        seq.end()
    }
}
struct Names<'a>(&'a antecedent_core::CausalSchema);
impl Serialize for Names<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.collect_seq(self.0.variables().iter().map(|v| v.name.as_ref()))
    }
}
struct Variables<'a>(&'a [antecedent_core::VariableId]);
impl Serialize for Variables<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.collect_seq(self.0.iter().map(|v| v.raw()))
    }
}
struct Selected<'a>(&'a antecedent_expr::IdentifiedEstimand);
impl Serialize for Selected<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let e = self.0;
        let mut object = s.serialize_struct("SelectedEstimand", 6)?;
        object.serialize_field("method", e.method.as_ref())?;
        object.serialize_field("adjustment_set", &Variables(&e.adjustment_set))?;
        object.serialize_field("instruments", &Variables(&e.instruments))?;
        object.serialize_field("mediators", &Variables(&e.mediators))?;
        object.serialize_field("functional", &e.functional.raw())?;
        object.serialize_field(
            "rd_design",
            &e.rd_design.map(|rd| antecedent_io::RdDesignWire {
                running_variable: rd.running_variable.raw(),
                cutoff: rd.cutoff,
                bandwidth: rd.bandwidth,
            }),
        )?;
        object.end()
    }
}
fn envelope_fields<M: SerializeMap, G>(
    object: &mut M,
    envelope: &antecedent_identify::IdentificationEnvelope<G>,
) -> Result<(), M::Error> {
    object.serialize_entry("invariant", &envelope.invariant.as_ref().map(Selected))?;
    object.serialize_entry("critical_graph_features", &Features(&envelope.critical_graph_features))
}
struct Features<'a>(&'a [antecedent_identify::GraphFeature]);
impl Serialize for Features<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut seq = s.serialize_seq(Some(self.0.len()))?;
        for feature in self.0 {
            seq.serialize_element(&Feature(feature))?;
        }
        seq.end()
    }
}
struct Feature<'a>(&'a antecedent_identify::GraphFeature);
impl Serialize for Feature<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut object = s.serialize_struct("GraphFeature", 2)?;
        object.serialize_field("kind", self.0.kind.as_ref())?;
        object.serialize_field("detail", self.0.detail.as_ref())?;
        object.end()
    }
}
struct Demotion<'a>(&'a crate::analysis::identification_cache::DbnIdentifyDemotion);
impl Serialize for Demotion<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let d = self.0;
        let mut object = s.serialize_struct("DbnIdentifyDemotion", 4)?;
        object.serialize_field("invalid_graph", &d.invalid_graph)?;
        object.serialize_field("identify_failed", &d.identify_failed)?;
        object.serialize_field("not_identified", &d.not_identified)?;
        object.serialize_field("no_estimand", &d.no_estimand)?;
        object.end()
    }
}
struct HorizonDemotions<'a>(
    &'a [(u32, crate::analysis::identification_cache::DbnIdentifyDemotion)],
);
impl Serialize for HorizonDemotions<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut seq = s.serialize_seq(Some(self.0.len()))?;
        for (horizon, demotion) in self.0 {
            seq.serialize_element(&HorizonDemotion(*horizon, demotion))?;
        }
        seq.end()
    }
}
struct HorizonDemotion<'a>(u32, &'a crate::analysis::identification_cache::DbnIdentifyDemotion);
impl Serialize for HorizonDemotion<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut object = s.serialize_struct("HorizonDemotion", 2)?;
        object.serialize_field("horizon", &self.0)?;
        object.serialize_field("demotion", &Demotion(self.1))?;
        object.end()
    }
}
struct NonGraphical<'a>(&'a crate::planner::lowered::NonGraphicalIdentification);
impl Serialize for NonGraphical<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use crate::planner::lowered::NonGraphicalIdentification::*;
        let mut object = s.serialize_map(None)?;
        match self.0 {
            SpecifiedModelConditional { model } => {
                object.serialize_entry("kind", "specified_model_conditional")?;
                object.serialize_entry("model", model)?;
            }
            ExperimentalRecovery { design, unresolved } => {
                object.serialize_entry("kind", "experimental_recovery")?;
                object.serialize_entry("design", design)?;
                object.serialize_entry("unresolved", &Unresolved(unresolved))?;
            }
            CyclicIoScm { source } => {
                object.serialize_entry("kind", "cyclic_io_scm")?;
                object.serialize_entry("source", source)?;
            }
            SharedEvidenceContract { contract } => {
                object.serialize_entry("kind", "shared_evidence_contract")?;
                object.serialize_entry("contract", contract)?;
            }
        }
        object.end()
    }
}
struct Unresolved<'a>(&'a [(antecedent_core::VariableId, Arc<[antecedent_core::VariableId]>)]);
impl Serialize for Unresolved<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut seq = s.serialize_seq(Some(self.0.len()))?;
        for (target, missing) in self.0 {
            seq.serialize_element(&UnresolvedEntry(*target, missing))?;
        }
        seq.end()
    }
}
struct UnresolvedEntry<'a>(antecedent_core::VariableId, &'a [antecedent_core::VariableId]);
impl Serialize for UnresolvedEntry<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut object = s.serialize_struct("UnresolvedDirection", 2)?;
        object.serialize_field("target", &self.0.raw())?;
        object.serialize_field("missing", &Variables(self.1))?;
        object.end()
    }
}
