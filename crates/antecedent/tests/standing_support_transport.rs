//! A carried claim's support in a composition graph: the graph's support facet has no state
//! between supported and outside support, so a claim whose label is weaker than `supported`
//! but inside the support its transport premise establishes states that premise as an
//! assumption, and a label outside support names its coordinates as unsupported.

use antecedent::analysis::composition_verification::{
    SupportStanding, library_transported_node, library_transported_standing, support_label_premise,
};

#[test]
fn a_weaker_label_is_supported_only_under_its_stated_premise() {
    assert_eq!(support_label_premise("supported"), None);
    assert_eq!(
        support_label_premise("weak_overlap").as_deref(),
        Some("support_label:weak_overlap")
    );
    assert_eq!(
        support_label_premise("extrapolative").as_deref(),
        Some("support_label:extrapolative")
    );
    // Outside support, missing evidence or no support label at all: the coordinates are
    // named unsupported instead, so no premise could make the claim supported.
    for label in ["outside_empirical_support", "missing_evidence", "not_a_label", ""] {
        assert_eq!(support_label_premise(label), None, "{label}");
        assert!(matches!(
            library_transported_standing(label, &[]).support,
            SupportStanding::Unsupported { .. }
        ));
    }
}

#[test]
fn a_library_transported_node_states_its_support_premise_once() {
    let node = library_transported_node(
        "carried",
        "cohort_transport",
        "route-identity",
        "clinic_b",
        "extrapolative",
        vec!["selection acts on age only".to_owned(), "support_label:extrapolative".to_owned()],
    );
    assert_eq!(node.support, SupportStanding::Supported);
    assert_eq!(
        node.assumptions,
        vec!["selection acts on age only".to_owned(), "support_label:extrapolative".to_owned()],
        "the premise is stated, and stated once"
    );
    let plain = library_transported_node(
        "carried",
        "cohort_transport",
        "route-identity",
        "clinic_b",
        "extrapolative",
        vec![],
    );
    assert_eq!(plain.assumptions, vec!["support_label:extrapolative".to_owned()]);
    // A supported label needs no premise; an outside label is unsupported, not premised.
    let supported = library_transported_node(
        "carried",
        "cohort_transport",
        "id",
        "clinic_b",
        "supported",
        vec![],
    );
    assert!(supported.assumptions.is_empty());
    let outside = library_transported_node(
        "carried",
        "cohort_transport",
        "id",
        "clinic_b",
        "outside_empirical_support",
        vec![],
    );
    assert!(outside.assumptions.is_empty());
    assert!(matches!(outside.support, SupportStanding::Unsupported { .. }));
}
