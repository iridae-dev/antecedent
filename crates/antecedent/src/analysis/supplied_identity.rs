//! Canonical identity of the declared inputs of a program over supplied evidence.
//!
//! A program family's identity is a [`CanonicalWriter`] stream (length-prefixed fields hashed with
//! BLAKE3) under the family's versioned tag. [`Declaration`] writes every declared input as a named
//! field, so a field never runs into the next. No value is rendered through `Debug` or `Display`.
//!
//! Encoding rules ([`Canonical`]):
//!
//! | Value | Encoding |
//! |---|---|
//! | integer, index, id | one little-endian word |
//! | real | its IEEE-754 bit pattern (`-0.0` and `0.0` differ, NaN payloads are kept) |
//! | text | length-prefixed UTF-8 |
//! | option | a presence word, then the value |
//! | ordered list | a count word, then each item in order: reordering it changes the identity |
//! | set ([`SetOf`]) | each item's sub-digest, sorted ascending and deduplicated, after a count word: reordering or repeating an item does not change the identity |
//! | enum | a variant tag text, then the variant's fields |
//! | wire value ([`Wire`]) | its CBOR wire encoding (fields in declaration order, floats in the shortest exact width, so injective on bits), length-prefixed; a value the wire cannot encode writes its conversion error under its own tag |
//! | decision contract | its canonical declaration identity (actions and constraints as a set, as the contract identity encodes them), then the action and constraint ids in declaration order, since results are reported in that order |
//! | aligned law | its decision-source digest (quantities, draws and weights) |
//! | graph | the graph's canonical identity wire (sorted edge lists), then every dense node's label in dense order |
//!
//! A runtime callback (a utility, a response source, a signal provider) cannot be encoded: the
//! writer records that one is bound ([`CanonicalWriter::note_runtime_callback`]) and writes the
//! callback's declared identity when the declaration carries one.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::ScientificQuantity;
use antecedent_core::response::IdentifiedSet;
use antecedent_core::{CanonicalWriter, NodeRef, SupportStatus, VariableId};
use antecedent_design::decision_adapters::{ClaimProbability, IdentifiedUtility, SuppliedClaim};
use antecedent_design::decision_contract::{AdmissibleDecisionContract, DecisionContract};
use antecedent_design::decision_robustness::{AtomSupport, InputSupport};
use antecedent_design::decision_structural::AtomEvidence;
use antecedent_graph::admg::Admg;
use antecedent_io::distribution_artifact::DistributionArtifact;

/// Tag of a set item's sub-digest.
const SET_ITEM: &str = "antecedent.supplied_identity.set_item.v1";

/// A value with a canonical encoding.
pub(crate) trait Canonical {
    /// Write the value's canonical encoding.
    fn encode(&self, out: &mut CanonicalWriter);
}

/// The identity of one program's declared inputs.
pub(crate) struct Declaration {
    writer: CanonicalWriter,
}

impl Declaration {
    /// Start the declaration of a family, under its versioned `tag`.
    pub(crate) fn new(tag: &str) -> Self {
        Self { writer: CanonicalWriter::new(tag) }
    }

    /// Write the declared input `name`.
    pub(crate) fn field(&mut self, name: &str, value: &(impl Canonical + ?Sized)) -> &mut Self {
        self.writer.text(name);
        value.encode(&mut self.writer);
        self
    }

    /// Record a runtime callback bound as `name`, with its declared identity when the
    /// declaration carries one.
    pub(crate) fn callback(&mut self, name: &str, declared: Option<&str>) -> &mut Self {
        self.writer.note_runtime_callback();
        self.writer.text(name).opt_text(declared);
        self
    }

    /// Whether a runtime callback is bound.
    #[cfg_attr(not(test), allow(dead_code, reason = "read by the identity tests"))]
    pub(crate) const fn has_callback(&self) -> bool {
        self.writer.has_runtime_callback()
    }

    /// The identity.
    pub(crate) fn finish(&self) -> [u8; 32] {
        self.writer.finish()
    }
}

/// A set-valued input: its items' sub-digests sorted and deduplicated, so neither the order
/// nor a repetition of an item changes the identity. Use it only for an input the evaluator
/// reads as a set.
pub(crate) struct SetOf<'a, T>(pub(crate) &'a [T]);

impl<T: Canonical> Canonical for SetOf<'_, T> {
    fn encode(&self, out: &mut CanonicalWriter) {
        let mut items: Vec<[u8; 32]> = self
            .0
            .iter()
            .map(|item| {
                let mut writer = CanonicalWriter::new(SET_ITEM);
                item.encode(&mut writer);
                writer.finish()
            })
            .collect();
        items.sort_unstable();
        items.dedup();
        out.word(items.len() as u64);
        for item in &items {
            out.bytes(item);
        }
    }
}

/// A value whose CBOR wire encoding is canonical: a serde wire type of scalar fields, ordered
/// lists and no hash-ordered maps.
pub(crate) struct Wire<'a, T>(pub(crate) &'a T);

impl<T: serde::Serialize> Canonical for Wire<'_, T> {
    fn encode(&self, out: &mut CanonicalWriter) {
        match antecedent_io::to_cbor(self.0) {
            Ok(bytes) => out.text("cbor").bytes(&bytes),
            Err(error) => out.text("unencodable").text(&error.to_string()),
        };
    }
}

/// The refusal of a declaration whose identity cannot be encoded because a non-exhaustive enum
/// from another crate gained a variant the encoding has no arm for. Writing such a variant
/// under a shared fallback tag would let two different declarations share an identity, so the
/// compile refuses until the encoding gains the arm.
pub(crate) fn unencoded_variant(plan: &str, what: &str) -> crate::CausalError {
    crate::compile_reason!(
        "route_not_supported",
        "{plan}.identity: the identity encoding has no arm for this {what} variant, so the \
         compiled route cannot bind it"
    )
}

/// An enum variant: its tag, then its fields.
pub(crate) fn variant<'w>(out: &'w mut CanonicalWriter, tag: &str) -> &'w mut CanonicalWriter {
    out.text(tag)
}

impl Canonical for bool {
    fn encode(&self, out: &mut CanonicalWriter) {
        out.word(u64::from(*self));
    }
}

impl Canonical for u32 {
    fn encode(&self, out: &mut CanonicalWriter) {
        out.word(u64::from(*self));
    }
}

impl Canonical for u64 {
    fn encode(&self, out: &mut CanonicalWriter) {
        out.word(*self);
    }
}

impl Canonical for usize {
    fn encode(&self, out: &mut CanonicalWriter) {
        out.word(*self as u64);
    }
}

impl Canonical for f64 {
    fn encode(&self, out: &mut CanonicalWriter) {
        out.real(*self);
    }
}

impl Canonical for str {
    fn encode(&self, out: &mut CanonicalWriter) {
        out.text(self);
    }
}

impl Canonical for String {
    fn encode(&self, out: &mut CanonicalWriter) {
        out.text(self);
    }
}

impl<T: Canonical + ?Sized> Canonical for &T {
    fn encode(&self, out: &mut CanonicalWriter) {
        (**self).encode(out);
    }
}

impl<T: Canonical + ?Sized> Canonical for Box<T> {
    fn encode(&self, out: &mut CanonicalWriter) {
        (**self).encode(out);
    }
}

impl<T: Canonical + ?Sized> Canonical for Arc<T> {
    fn encode(&self, out: &mut CanonicalWriter) {
        (**self).encode(out);
    }
}

impl<T: Canonical> Canonical for Option<T> {
    fn encode(&self, out: &mut CanonicalWriter) {
        match self {
            Some(value) => {
                out.word(1);
                value.encode(out);
            }
            None => {
                out.word(0);
            }
        }
    }
}

impl<T: Canonical> Canonical for [T] {
    fn encode(&self, out: &mut CanonicalWriter) {
        out.word(self.len() as u64);
        for item in self {
            item.encode(out);
        }
    }
}

impl<T: Canonical> Canonical for Vec<T> {
    fn encode(&self, out: &mut CanonicalWriter) {
        self.as_slice().encode(out);
    }
}

impl<A: Canonical, B: Canonical> Canonical for (A, B) {
    fn encode(&self, out: &mut CanonicalWriter) {
        self.0.encode(out);
        self.1.encode(out);
    }
}

impl Canonical for VariableId {
    fn encode(&self, out: &mut CanonicalWriter) {
        out.word(u64::from(self.raw()));
    }
}

impl Canonical for SupportStatus {
    fn encode(&self, out: &mut CanonicalWriter) {
        out.text(self.as_str());
    }
}

impl Canonical for ScientificQuantity {
    fn encode(&self, out: &mut CanonicalWriter) {
        Wire(&antecedent_io_core::quantity_wire::ScientificQuantityWire::from(self)).encode(out);
    }
}

impl Canonical for IdentifiedSet<f64> {
    fn encode(&self, out: &mut CanonicalWriter) {
        out.real(self.lower).real(self.upper);
    }
}

impl Canonical for DecisionContract {
    fn encode(&self, out: &mut CanonicalWriter) {
        out.text(&self.declaration_identity());
        let actions: Vec<&str> = self.actions.iter().map(|action| action.id.as_str()).collect();
        let constraints: Vec<&str> = self.constraints.iter().map(|c| c.id.as_str()).collect();
        out.texts(&actions).texts(&constraints);
    }
}

impl Canonical for AdmissibleDecisionContract {
    fn encode(&self, out: &mut CanonicalWriter) {
        out.text(&self.declaration_identity());
        let actions: Vec<&str> =
            self.contract.actions.iter().map(|action| action.id.as_str()).collect();
        let constraints: Vec<&str> =
            self.contract.constraints.iter().map(|c| c.id.as_str()).collect();
        out.texts(&actions).texts(&constraints);
    }
}

impl Canonical for DistributionArtifact {
    fn encode(&self, out: &mut CanonicalWriter) {
        out.text(&antecedent_design::decision_artifact::source_digest(self));
    }
}

impl Canonical for NodeRef {
    fn encode(&self, out: &mut CanonicalWriter) {
        match *self {
            Self::Static(variable) => {
                variant(out, "static").word(u64::from(variable.raw()));
            }
            Self::Lagged { variable, lag } => {
                variant(out, "lagged").word(u64::from(variable.raw())).word(u64::from(lag.raw()));
            }
            Self::Unfolded { variable, offset } => {
                variant(out, "unfolded")
                    .word(u64::from(variable.raw()))
                    .word(u64::from(offset < 0))
                    .word(u64::from(offset.unsigned_abs()));
            }
            Self::Context { variable, environment } => {
                variant(out, "context")
                    .word(u64::from(variable.raw()))
                    .opt_word(environment.map(|id| u64::from(id.raw())));
            }
        }
    }
}

impl Canonical for Admg {
    fn encode(&self, out: &mut CanonicalWriter) {
        match antecedent_io_core::identity::admg_identity(self) {
            Ok(wire) => Wire(&wire).encode(out),
            Err(error) => {
                out.text("unencodable").text(&error.to_string());
            }
        }
        // The identity wire indexes dense nodes and records only their count; the label of
        // each dense node, in dense order, says which variable the edges are about.
        self.nodes().encode(out);
    }
}

impl Canonical for ClaimProbability {
    fn encode(&self, out: &mut CanonicalWriter) {
        match self {
            Self::Unspecified => {
                variant(out, "unspecified");
            }
            Self::Genuine(probability) => {
                variant(out, "genuine").real(*probability);
            }
            Self::CompletionCount(count) => {
                variant(out, "completion_count").word(*count);
            }
        }
    }
}

impl Canonical for AtomEvidence {
    fn encode(&self, out: &mut CanonicalWriter) {
        match self {
            Self::Evaluated(law) => {
                variant(out, "evaluated");
                law.encode(out);
            }
            Self::Unidentified => {
                variant(out, "unidentified");
            }
            Self::Unevaluated(reason) => {
                variant(out, "unevaluated").text(reason);
            }
        }
    }
}

impl Canonical for InputSupport {
    fn encode(&self, out: &mut CanonicalWriter) {
        out.text(&self.action_id).word(self.input as u64);
        self.status.encode(out);
    }
}

impl Canonical for AtomSupport {
    fn encode(&self, out: &mut CanonicalWriter) {
        self.overall.encode(out);
        // Ordered: a lookup reads the first entry for an input.
        self.per_input.encode(out);
    }
}

impl Canonical for SuppliedClaim {
    fn encode(&self, out: &mut CanonicalWriter) {
        out.text(&self.id);
        self.probability.encode(out);
        self.evidence.encode(out);
        self.support.encode(out);
    }
}

impl Canonical for IdentifiedUtility {
    fn encode(&self, out: &mut CanonicalWriter) {
        out.text(&self.action_id);
        self.utility.encode(out);
        self.hard_exclusions.encode(out);
        self.support.encode(out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(build: impl FnOnce(&mut Declaration)) -> [u8; 32] {
        let mut declaration = Declaration::new("antecedent.supplied_identity.test.v1");
        build(&mut declaration);
        declaration.finish()
    }

    #[test]
    fn an_ordered_list_is_order_significant_and_a_set_is_not() {
        let ordered = |items: &[u64]| {
            digest(|d| {
                d.field("items", items);
            })
        };
        let set = |items: &[u64]| {
            digest(|d| {
                d.field("items", &SetOf(items));
            })
        };
        assert_eq!(ordered(&[1, 2]), ordered(&[1, 2]));
        assert_ne!(ordered(&[1, 2]), ordered(&[2, 1]));
        assert_eq!(set(&[1, 2]), set(&[2, 1]));
        assert_eq!(set(&[1, 2]), set(&[2, 1, 2]));
        assert_ne!(set(&[1, 2]), set(&[1, 3]));
    }

    #[test]
    fn fields_are_framed_and_reals_are_their_bits() {
        let texts = |a: &str, b: &str| {
            digest(|d| {
                d.field("a", a).field("b", b);
            })
        };
        assert_ne!(texts("ab", "c"), texts("a", "bc"));
        let real = |x: f64| {
            digest(|d| {
                d.field("x", &x);
            })
        };
        assert_ne!(real(0.0), real(-0.0));
        assert_eq!(real(0.1), real(0.1));
    }

    #[test]
    fn a_graph_is_its_sorted_edges_and_its_node_labels() {
        use antecedent_graph::DenseNodeId;
        let graph = |labels: &[u32], edges: &[(u32, u32)]| {
            let mut admg = Admg::empty();
            for &label in labels {
                admg.add_node(NodeRef::Static(VariableId::from_raw(label))).unwrap();
            }
            for &(from, to) in edges {
                admg.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to))
                    .unwrap();
            }
            digest(|d| {
                d.field("graph", &admg);
            })
        };
        let base = graph(&[0, 1, 2], &[(0, 1), (1, 2)]);
        assert_eq!(base, graph(&[0, 1, 2], &[(1, 2), (0, 1)]), "edge insertion order");
        assert_ne!(base, graph(&[0, 1, 2], &[(0, 1), (0, 2)]), "a changed edge");
        assert_ne!(base, graph(&[0, 1, 5], &[(0, 1), (1, 2)]), "a changed node label");
    }

    #[test]
    fn a_variant_with_no_encoding_arm_refuses_the_compile() {
        match unencoded_variant("some.plan.v1", "request") {
            crate::CausalError::Compile { message } => {
                assert!(message.contains("route_not_supported"), "{message}");
                assert!(message.contains("some.plan.v1.identity"), "{message}");
                assert!(message.contains("no arm for this request variant"), "{message}");
            }
            other => panic!("expected a compile refusal, got {other:?}"),
        }
    }

    #[test]
    fn a_callback_is_recorded_with_its_declared_identity() {
        let mut declaration = Declaration::new("antecedent.supplied_identity.test.v1");
        assert!(!declaration.has_callback());
        declaration.callback("utility", Some("declared"));
        assert!(declaration.has_callback());
        let other = digest(|d| {
            d.callback("utility", Some("other"));
        });
        assert_ne!(declaration.finish(), other);
    }
}
