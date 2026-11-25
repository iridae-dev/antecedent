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
//! | program identity ([`SemanticDigest`]) | its 32 digest bytes, length-prefixed |
//! | option | a presence word, then the value |
//! | ordered list | a count word, then each item in order: reordering it changes the identity |
//! | set ([`SetOf`]) | each item's sub-digest, sorted ascending and deduplicated, after a count word ([`CanonicalWriter::digest_set`]): reordering or repeating an item does not change the identity |
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
use antecedent_core::{CanonicalWriter, NodeRef, SemanticDigest, SupportStatus, VariableId};
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
        let items: Vec<[u8; 32]> = self
            .0
            .iter()
            .map(|item| {
                let mut writer = CanonicalWriter::new(SET_ITEM);
                item.encode(&mut writer);
                writer.finish()
            })
            .collect();
        out.digest_set(items);
    }
}

/// A value whose CBOR wire encoding is canonical: a serde wire type of scalar fields, ordered
/// lists and no hash-ordered maps.
pub(crate) struct Wire<'a, T>(pub(crate) &'a T);

impl<T: serde::Serialize> Canonical for Wire<'_, T> {
    fn encode(&self, out: &mut CanonicalWriter) {
        match antecedent_io::to_cbor(self.0) {
            Ok(bytes) => {
                out.text("cbor").bytes(&bytes);
            }
            Err(_) => unencodable(out, "cbor wire value"),
        }
    }
}

thread_local! {
    /// The first value an encoding on this thread could not encode, since the innermost
    /// [`encoded`] began.
    static UNENCODABLE: std::cell::RefCell<Option<&'static str>> =
        const { std::cell::RefCell::new(None) };
}

/// Record that the value `what` has no canonical encoding. The writer receives a fixed marker,
/// never an error's text; the identity it is part of is never used, because [`encoded`] refuses
/// the compile that encoded it.
pub(crate) fn unencodable(out: &mut CanonicalWriter, what: &'static str) {
    UNENCODABLE.with(|slot| {
        slot.borrow_mut().get_or_insert(what);
    });
    out.text("unencodable").text(what);
}

/// Run `identity`, which encodes a declaration, and refuse the compile of `plan` when a value
/// it encoded has no canonical encoding ([`unencodable`]). A nested call sees only its own
/// encoding; the enclosing call still sees what the nested one recorded.
///
/// # Errors
/// `route_not_supported` when a value could not be encoded.
pub(crate) fn encoded<R>(
    plan: &str,
    identity: impl FnOnce() -> R,
) -> Result<R, crate::CausalError> {
    let outer = UNENCODABLE.with(|slot| slot.borrow_mut().take());
    let value = identity();
    let inner = UNENCODABLE.with(|slot| {
        let mut slot = slot.borrow_mut();
        let inner = slot.take();
        *slot = outer.or(inner);
        inner
    });
    match inner {
        Some(what) => Err(crate::compile_reason!(
            "route_not_supported",
            "{plan}.identity: the declared {what} has no canonical encoding, so the compiled \
             route cannot bind it"
        )),
        None => Ok(value),
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
                out.presence(true);
                value.encode(out);
            }
            None => {
                out.presence(false);
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

impl Canonical for SemanticDigest {
    fn encode(&self, out: &mut CanonicalWriter) {
        out.bytes(self.as_bytes());
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
            Err(_) => unencodable(out, "graph"),
        }
        // The identity wire indexes dense nodes and records only their count; the label of
        // each dense node, in dense order, says which variable the edges are about.
        self.nodes().encode(out);
    }
}

/// A discovery failure by its stable kind and registered reason code, never its message (which
/// carries input values and may change wording).
impl Canonical for antecedent_discovery::DiscoveryError {
    fn encode(&self, out: &mut CanonicalWriter) {
        variant(out, "discovery").text(self.kind()).opt_text(self.refusal_code());
    }
}

/// A selective-discovery refusal by its registered reason code and namespaced detail, or the
/// discovery failure it wraps; never its remedy text.
impl Canonical for antecedent_discovery::selective::SelectiveDiscoveryError {
    fn encode(&self, out: &mut CanonicalWriter) {
        if let Self::Discovery(error) = self {
            error.encode(out);
        } else {
            variant(out, "selective").opt_text(self.code()).opt_text(self.detail());
        }
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
    use crate::identity_laws::{
        ADVERSARIAL_TEXT, all_distinct, bit_exact, injective, ordered_laws, set_laws,
    };

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
    fn site_supplied_identity_an_unencodable_value_refuses_and_never_hashes_error_text() {
        let digest = |fail: bool| {
            encoded("some.plan.v1", || {
                let mut writer = CanonicalWriter::new("antecedent.supplied_identity.test.v1");
                writer.text("graph");
                if fail {
                    unencodable(&mut writer, "graph");
                }
                writer.finish()
            })
        };
        let accepted = digest(false).expect("an encodable declaration is accepted");
        assert_eq!(accepted, digest(false).expect("repeatable"));
        match digest(true) {
            Err(crate::CausalError::Compile { message }) => {
                assert!(message.contains("route_not_supported"), "{message}");
                assert!(message.contains("some.plan.v1.identity"), "{message}");
                assert!(message.contains("declared graph has no canonical encoding"), "{message}");
            }
            other => panic!("expected a compile refusal, got {other:?}"),
        }
        // A refusal does not leak into the next declaration.
        assert_eq!(digest(false).expect("the slot is cleared"), accepted);
        // A nested refusal reaches the enclosing declaration, and the nested call refuses too.
        let outer = encoded("outer.plan.v1", || {
            assert!(digest(true).is_err());
        });
        assert!(outer.is_err(), "the enclosing declaration sees the nested refusal");
        // An outer refusal recorded before a nested, accepted declaration survives it.
        let outer = encoded("outer.plan.v1", || {
            let mut writer = CanonicalWriter::new("antecedent.supplied_identity.test.v1");
            unencodable(&mut writer, "graph");
            assert!(digest(false).is_ok(), "the nested declaration sees only its own values");
        });
        assert!(outer.is_err(), "the outer refusal survives a nested accepted declaration");
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

    fn encoded_value(value: &(impl Canonical + ?Sized)) -> [u8; 32] {
        digest(|d| {
            d.field("value", value);
        })
    }

    fn quantity(variable: &str) -> ScientificQuantity {
        ScientificQuantity {
            variable_id: variable.into(),
            variable_name: variable.into(),
            role: antecedent_contract::QuantityRole::Outcome,
            units: "units".into(),
            population_id: "target".into(),
            regime_id: "do(a=1)".into(),
            horizon: 0,
            functional_id: "outcome".into(),
            conditioning: vec![],
            transform_id: "identity".into(),
        }
    }

    fn contract_of(actions: &[&str], constraints: &[&str]) -> DecisionContract {
        use antecedent_design::decision_contract::{
            ActionKind, DecisionAction, DecisionCriterion, HardConstraint, StructuralPolicy,
            UtilityExpr,
        };
        DecisionContract {
            actions: actions
                .iter()
                .map(|id| DecisionAction {
                    id: (*id).to_owned(),
                    kind: ActionKind::Intervention,
                    inputs: vec![quantity(id)],
                    utility: UtilityExpr::Input(0),
                })
                .collect(),
            utility_units: "units".into(),
            criterion: DecisionCriterion::PosteriorExpectedUtility,
            constraints: constraints
                .iter()
                .map(|id| HardConstraint {
                    id: (*id).to_owned(),
                    expr: UtilityExpr::Input(0),
                    bound: 1.0,
                    min_probability: 1.0,
                    units: "units".into(),
                    applies_to: vec![],
                })
                .collect(),
            target_population: "target".into(),
            horizon: 0,
            structural_policy: StructuralPolicy::Maximin,
        }
    }

    fn admissible_of(actions: &[&str], constraints: &[&str]) -> AdmissibleDecisionContract {
        AdmissibleDecisionContract {
            contract: contract_of(actions, constraints),
            rules: antecedent_design::decision_contract::AdmissibilityRules::default(),
        }
    }

    #[test]
    fn site_supplied_identity_set_of_sorts_and_deduplicates_framed_items() {
        set_laws(&[3_u64, 1, 2], |items| Some(encoded_value(&SetOf(items))));
        set_laws(&["b".to_owned(), "a".to_owned(), "c".to_owned()], |items| {
            Some(encoded_value(&SetOf(items)))
        });
        // Each item is its own framed sub-digest, so two different items never merge.
        injective(&ADVERSARIAL_TEXT, |p| *p, |(a, b)| Some(encoded_value(&SetOf(&[(*a, *b)]))));
        bit_exact(|x| Some(encoded_value(&SetOf(&[x, 1.0]))));
        // A set is not the list of the same items.
        all_distinct(&[encoded_value(&SetOf(&[1_u64, 2])), encoded_value(&[1_u64, 2][..])]);
    }

    #[test]
    fn site_supplied_identity_list_is_counted_and_ordered() {
        ordered_laws(&[1_u64, 2, 3], |items| Some(encoded_value(items)));
        ordered_laws(&["a".to_owned(), "b".to_owned()], |items| Some(encoded_value(items)));
        // The count word frames nested lists: [[1], [2, 3]] is not [[1, 2], [3]].
        let nested =
            [vec![vec![1_u64], vec![2, 3]], vec![vec![1, 2], vec![3]], vec![vec![1, 2, 3]]];
        injective(&nested, Clone::clone, |lists| Some(encoded_value(lists)));
        injective(&ADVERSARIAL_TEXT, |p| *p, |(a, b)| Some(encoded_value(&[*a, *b][..])));
        bit_exact(|x| Some(encoded_value(&[x][..])));
        // An option's presence word frames it.
        all_distinct(&[
            encoded_value(&None::<u64>),
            encoded_value(&Some(0_u64)),
            encoded_value(&Some(1_u64)),
        ]);
    }

    #[test]
    fn site_supplied_identity_fields_are_named_and_framed() {
        // A field's name and its value are framed: neither a renamed field nor a value that
        // moves between fields shares an identity.
        let pairs = [("a", "v"), ("b", "v"), ("a", "w"), ("ab", ""), ("a", "b")];
        injective(
            &pairs,
            |p| *p,
            |(name, value)| {
                Some(digest(|d| {
                    d.field(name, *value);
                }))
            },
        );
        injective(
            &ADVERSARIAL_TEXT,
            |p| *p,
            |(a, b)| {
                Some(digest(|d| {
                    d.field("a", *a).field("b", *b);
                }))
            },
        );
    }

    #[test]
    fn site_supplied_identity_decision_contract_orders_actions_and_constraints() {
        let actions = ["A", "B", "C"];
        let constraints = ["c1", "c2", "c3"];
        // Results are reported in declaration order, so the ids are ordered after the
        // contract's own set identity.
        ordered_laws(&actions, |items| Some(encoded_value(&contract_of(items, &constraints))));
        ordered_laws(&constraints, |items| Some(encoded_value(&contract_of(&actions, items))));
        ordered_laws(&actions, |items| Some(encoded_value(&admissible_of(items, &constraints))));
        ordered_laws(&constraints, |items| Some(encoded_value(&admissible_of(&actions, items))));
        injective(
            &ADVERSARIAL_TEXT,
            |p| *p,
            |(a, b)| Some(encoded_value(&contract_of(&[*a, *b], &[]))),
        );
        let mut changed = contract_of(&actions, &constraints);
        changed.constraints[0].bound = 2.0;
        all_distinct(&[
            encoded_value(&contract_of(&actions, &constraints)),
            encoded_value(&changed),
            encoded_value(&contract_of(&actions[..2], &constraints)),
            encoded_value(&contract_of(&actions, &constraints[..2])),
        ]);
    }

    #[test]
    fn site_supplied_identity_atom_support_and_exclusions_are_ordered() {
        use antecedent_design::decision_robustness::InputSupport;
        let entry = |action: &str, input: usize, status: SupportStatus| InputSupport {
            action_id: action.into(),
            input,
            status,
        };
        let entries = [
            entry("A", 0, SupportStatus::Supported),
            entry("A", 1, SupportStatus::WeakOverlap),
            entry("B", 0, SupportStatus::Extrapolative),
        ];
        // Ordered: status_of reads the first entry for an input.
        let support = |per_input: &[InputSupport]| AtomSupport {
            overall: SupportStatus::Supported,
            per_input: per_input.to_vec(),
        };
        ordered_laws(&entries, |items| Some(encoded_value(&support(items))));
        injective(
            &ADVERSARIAL_TEXT,
            |p| *p,
            |(a, b)| {
                Some(encoded_value(&support(&[
                    entry(a, 0, SupportStatus::Supported),
                    entry(b, 0, SupportStatus::Supported),
                ])))
            },
        );
        let utility = |exclusions: &[&str], lower: f64| IdentifiedUtility {
            action_id: "A".into(),
            utility: IdentifiedSet { lower, upper: 2.0 },
            hard_exclusions: exclusions.iter().map(|id| (*id).to_owned()).collect(),
            support: AtomSupport::supported(),
        };
        // Ordered: the exclusions are reported with the action in the order supplied.
        ordered_laws(&["c1", "c2", "c3"], |items| Some(encoded_value(&utility(items, 1.0))));
        bit_exact(|lower| Some(encoded_value(&utility(&[], lower))));
        all_distinct(&[
            encoded_value(&AtomSupport::supported()),
            encoded_value(&AtomSupport::unassessed()),
            encoded_value(&support(&entries[..1])),
        ]);
    }

    #[test]
    fn site_supplied_identity_graph_edges_are_a_set_and_labels_dense_ordered() {
        use antecedent_graph::DenseNodeId;
        let graph = |labels: &[u32], edges: &[(u32, u32)]| -> Option<[u8; 32]> {
            let mut admg = Admg::empty();
            for &label in labels {
                admg.add_node(NodeRef::Static(VariableId::from_raw(label))).ok()?;
            }
            for &(from, to) in edges {
                admg.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to))
                    .ok()?;
            }
            Some(encoded_value(&admg))
        };
        let labels = [0_u32, 1, 2, 3];
        set_laws(&[(0_u32, 1_u32), (1, 2), (2, 3)], |edges| graph(&labels, edges));
        // The dense labels say which variable each edge is about.
        ordered_laws(&[0_u32, 1, 2], |labels| graph(labels, &[(0, 1)]));
    }

    #[test]
    fn site_supplied_identity_node_ref_variants_are_tagged() {
        use antecedent_core::{EnvironmentId, Lag};
        let variable = VariableId::from_raw(1);
        all_distinct(&[
            encoded_value(&NodeRef::Static(variable)),
            encoded_value(&NodeRef::Static(VariableId::from_raw(2))),
            encoded_value(&NodeRef::Lagged { variable, lag: Lag::from_raw(0) }),
            encoded_value(&NodeRef::Lagged { variable, lag: Lag::from_raw(1) }),
            encoded_value(&NodeRef::Unfolded { variable, offset: 1 }),
            encoded_value(&NodeRef::Unfolded { variable, offset: -1 }),
            encoded_value(&NodeRef::Unfolded { variable, offset: 0 }),
            encoded_value(&NodeRef::Context { variable, environment: None }),
            encoded_value(&NodeRef::Context {
                variable,
                environment: Some(EnvironmentId::from_raw(0)),
            }),
        ]);
    }

    #[test]
    fn site_supplied_identity_wire_lists_keep_their_order() {
        ordered_laws(&[1_u64, 2, 3], |items| Some(encoded_value(&Wire(&items.to_vec()))));
        injective(&ADVERSARIAL_TEXT, |p| *p, |(a, b)| Some(encoded_value(&Wire(&(*a, *b)))));
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
