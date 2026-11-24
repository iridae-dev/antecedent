//! Identity sites of antecedent-contract (formal/identity_sites.toml), checked on production
//! code through the public API with the category laws of `common`.
//!
//! Each `site_*` test builds the production input from small generated parts, enumerates
//! every reordering of the canonicalized collection, and applies the law of the collection's
//! class: a set keeps its identity under reordering and repetition (or production refuses the
//! repeat) and changes it when an element is dropped; a multiset changes it on repetition; an
//! ordered list changes it when two elements swap; text joined into an identity is injective.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

mod common;

use std::ops::Range;

use antecedent_contract::evidence_ledger::{
    CombinationRequest, EvidenceLedger, Lineage, ObservationId, PoolingDependence,
    SourceDeclaration, SourceId, SourceKind, SynthesisOperation, TransportIdentity,
};
use antecedent_core::SemanticDigest;
use proptest::prelude::*;

/// Every ordering of `items` (at most 5).
fn orderings<T: Clone>(items: &[T]) -> Vec<Vec<T>> {
    assert!(items.len() <= 5);
    if items.len() <= 1 {
        return vec![items.to_vec()];
    }
    let mut out = Vec::new();
    for index in 0..items.len() {
        let mut rest = items.to_vec();
        let head = rest.remove(index);
        for mut tail in orderings(&rest) {
            tail.insert(0, head.clone());
            out.push(tail);
        }
    }
    out
}

fn check_set<T: Clone + std::fmt::Debug + PartialEq, I: PartialEq + std::fmt::Debug>(
    items: &[T],
    identity: impl Fn(&[T]) -> Option<I>,
) {
    for ordering in orderings(items) {
        common::set_laws(items, &ordering, &identity).unwrap();
    }
}

fn check_multiset<T: Clone + std::fmt::Debug + PartialEq, I: PartialEq + std::fmt::Debug>(
    items: &[T],
    identity: impl Fn(&[T]) -> Option<I>,
) {
    for ordering in orderings(items) {
        common::multiset_laws(items, &ordering, &identity).unwrap();
    }
}

fn check_ordered<T: Clone + std::fmt::Debug + PartialEq, I: PartialEq + std::fmt::Debug>(
    items: &[T],
    identity: impl Fn(&[T]) -> Option<I>,
) {
    for i in 0..items.len() {
        for j in 0..items.len() {
            common::ordered_laws(items, i, j, &identity).unwrap();
        }
    }
}

/// A set the site accepts only in its canonical (sorted) spelling: `items` in that spelling
/// is admitted, every other ordering and a repeat are refused, and dropping an element
/// changes the identity.
fn check_canonical_set<T: Clone + std::fmt::Debug + PartialEq, I: PartialEq + std::fmt::Debug>(
    items: &[T],
    identity: impl Fn(&[T]) -> Option<I>,
) {
    let base = identity(items).expect("the canonical spelling is admitted");
    for ordering in orderings(items) {
        if ordering.as_slice() != items {
            assert!(identity(&ordering).is_none(), "a non-canonical spelling was admitted");
        }
    }
    if let Some(first) = items.first() {
        let mut repeated = items.to_vec();
        repeated.insert(1, first.clone());
        assert!(identity(&repeated).is_none(), "a repeated element was admitted");
        if let Some(smaller) = identity(&items[1..]) {
            assert_ne!(smaller, base, "dropping an element left the identity unchanged");
        }
    }
}

fn digest(byte: u8) -> SemanticDigest {
    SemanticDigest::from_bytes([byte; 32])
}

// ---------------------------------------------------------------------------
// Evidence ledger.
// ---------------------------------------------------------------------------

fn obs(range: Range<u32>) -> Vec<ObservationId> {
    range.map(|i| ObservationId::from_key("cohort", &i.to_le_bytes())).collect()
}

fn data(label: &str, range: Range<u32>) -> SourceDeclaration {
    SourceDeclaration::new(label, SourceKind::ObservationSet).with_observations(obs(range))
}

/// A ledger holding three disjoint observation sets, and their ids.
fn three_sources() -> (EvidenceLedger, [SourceId; 3]) {
    let mut ledger = EvidenceLedger::new();
    let a = ledger.add_source(data("a", 0..10)).unwrap();
    let b = ledger.add_source(data("b", 10..20)).unwrap();
    let c = ledger.add_source(data("c", 20..30)).unwrap();
    (ledger, [a, b, c])
}

/// `contract.evidence_ledger.source.SourceDeclaration.identity` and
/// `contract.evidence_ledger.ledger.EvidenceLedger.canonical_source`: a source's observation
/// rows are a set.
#[test]
fn site_source_declaration_observation_rows() {
    let rows = obs(0..4);
    check_set(&rows, |listed| {
        EvidenceLedger::new()
            .add_source(
                SourceDeclaration::new("rows", SourceKind::ObservationSet)
                    .with_observations(listed.iter().copied()),
            )
            .ok()
    });
}

/// `contract.evidence_ledger.source.Lineage.encode`: the inputs of a derived source are a set
/// (the ledger refuses a repeated input).
#[test]
fn site_lineage_inputs() {
    let (_, ids) = three_sources();
    check_set(&ids, |listed| {
        let (mut ledger, _) = three_sources();
        ledger
            .add_source(
                SourceDeclaration::new("derived", SourceKind::ExternalClaim)
                    .with_lineage(Lineage::DerivedFromUsedData { inputs: listed.to_vec() }),
            )
            .ok()
    });
}

/// The identity of one combination of `ids`, with what it counted.
fn pooled(
    ids: &[SourceId],
) -> Option<(SemanticDigest, Option<SemanticDigest>, Vec<SemanticDigest>, u64)> {
    let (mut ledger, _) = three_sources();
    let result = ledger
        .combine(CombinationRequest {
            label: "pooled".to_owned(),
            operation: SynthesisOperation::StatisticalPooling {
                sources: ids.to_vec(),
                dependence: PoolingDependence::Independent,
            },
            replaces: None,
        })
        .ok()?;
    let record = ledger.result(result)?;
    Some((
        result.digest(),
        record.observations_digest,
        record.transport_identities.clone(),
        record.mass.distinct_observations,
    ))
}

/// `contract.evidence_ledger.operation.SynthesisOperation.encode` and
/// `contract.evidence_ledger.ledger.EvidenceLedger.account`: the sources of a combination are
/// a set, and so are the observations and transports it counts.
#[test]
fn site_synthesis_operation_sources() {
    let (_, ids) = three_sources();
    check_set(&ids, pooled);
}

/// `contract.evidence_ledger.ledger.EvidenceLedger.mass_account` and
/// `contract.evidence_ledger.overlap.union_sorted`: the accounted sources are a set and their
/// overlapping observations are counted once.
#[test]
fn site_mass_account_sources() {
    let mut ledger = EvidenceLedger::new();
    let ids = [
        ledger.add_source(data("a", 0..10)).unwrap(),
        ledger.add_source(data("b", 5..15)).unwrap(),
        ledger.add_source(data("c", 12..20)).unwrap(),
    ];
    check_set(&ids, |listed| {
        let mass = ledger.mass_account(listed).ok()?;
        Some((mass.sources, mass.declared_slots, mass.distinct_observations))
    });
    assert_eq!(ledger.mass_account(&ids).unwrap().distinct_observations, 20);
}

/// `contract.evidence_ledger.ledger.EvidenceLedger.state_digest`: sources and results are
/// listed by their unique ids, so the state depends on what the ledger holds and its history
/// head, and any change of a standing moves it.
#[test]
fn site_state_digest() {
    let (first, ids) = three_sources();
    let (second, _) = three_sources();
    assert_eq!(first.state_digest(), second.state_digest());
    let mut withdrawn = three_sources().0;
    withdrawn.withdraw_source(ids[1], "retracted").unwrap();
    assert_ne!(withdrawn.state_digest(), first.state_digest());
    let mut grown = three_sources().0;
    grown.add_source(data("d", 30..40)).unwrap();
    assert_ne!(grown.state_digest(), first.state_digest());
}

fn transport(populations: &[&str], assumptions: &[&str]) -> TransportIdentity {
    TransportIdentity {
        source_populations: populations.iter().map(|p| (*p).to_owned()).collect(),
        target_population: "target".to_owned(),
        selection_diagram: digest(7),
        assumptions: assumptions.iter().map(|a| (*a).to_owned()).collect(),
    }
}

/// `contract.evidence_ledger.source.TransportIdentity.encode`: populations and assumptions are
/// multisets in `digest` (as documented) and sets in `set_digest`, the form the ledger and
/// every offer and handoff read.
#[test]
fn site_transport_identity_lists() {
    let populations = ["p", "q", "r"];
    check_multiset(&populations, |listed| Some(transport(listed, &["s"]).digest()));
    check_set(&populations, |listed| Some(transport(listed, &["s"]).set_digest()));
    let assumptions = ["s", "t", "u"];
    check_multiset(&assumptions, |listed| Some(transport(&["p"], listed).digest()));
    check_set(&assumptions, |listed| Some(transport(&["p"], listed).set_digest()));
}

proptest! {
    #![proptest_config(common::config())]

    /// The transport lists are framed: moving text between a population and an assumption,
    /// or splitting one entry in two, always changes the set digest.
    #[test]
    fn site_transport_identity_framing(
        left in (prop::collection::vec(common::adversarial_text(), 1..3), prop::collection::vec(common::adversarial_text(), 1..3)),
        right in (prop::collection::vec(common::adversarial_text(), 1..3), prop::collection::vec(common::adversarial_text(), 1..3)),
    ) {
        let build = |(populations, assumptions): &(Vec<String>, Vec<String>)| TransportIdentity {
            source_populations: populations.clone(),
            target_population: "target".to_owned(),
            selection_diagram: digest(7),
            assumptions: assumptions.clone(),
        };
        common::injective(
            &left,
            &right,
            |parts| {
                let sets = build(parts).as_sets();
                (sets.source_populations, sets.assumptions)
            },
            |parts| Some(build(parts).set_digest()),
        )?;
    }
}

// ---------------------------------------------------------------------------
// Provider transport.
// ---------------------------------------------------------------------------

mod provider {
    use antecedent_contract::interface_contract::{
        ContractOperation, EvidenceBasis, SupportDomain,
    };
    use antecedent_contract::provider_transport::{
        ProviderTransportSource, SourceTrustRequirement, TransportedClaimOffer,
    };
    use antecedent_contract::{
        DistributionMeaning, ExternalCapability, ExternalScientificObject, ExternalTrustState,
        LawProviderContract, ProviderObjectIdentity, QuantityRole, ScientificQuantity,
        VerificationProbe, VerificationProbeKind, verify_external_object,
    };

    use super::{SourceId, TransportIdentity, check_set, digest};

    pub(super) fn quantity(population: &str, regime: &str) -> ScientificQuantity {
        ScientificQuantity {
            variable_id: "schema:recovery".into(),
            variable_name: "recovery".into(),
            role: QuantityRole::Outcome,
            units: "days".into(),
            population_id: population.into(),
            regime_id: regime.into(),
            horizon: 0,
            functional_id: "outcome".into(),
            conditioning: Vec::new(),
            transform_id: "identity".into(),
        }
    }

    fn law() -> ExternalScientificObject {
        ExternalScientificObject::Law(LawProviderContract {
            identity: ProviderObjectIdentity {
                provider_id: "lab".into(),
                object_id: "trial-a-law".into(),
                version_id: "1.0".into(),
                snapshot_id: "snap-1".into(),
                request_id: "req-1".into(),
            },
            quantities: vec![quantity("trial_a", "do(a=1)")],
            meaning: DistributionMeaning::InterventionalPredictive,
            capabilities: vec![ExternalCapability::Mean, ExternalCapability::Sample],
        })
    }

    fn verified(object: &ExternalScientificObject) -> ExternalTrustState {
        use VerificationProbeKind as P;
        let probes: Vec<VerificationProbe> =
            [P::KnownTruth, P::Shape, P::Support, P::Moments, P::SeededBehavior]
                .into_iter()
                .map(|kind| VerificationProbe {
                    kind,
                    observed: 2.0,
                    expected: 2.0,
                    tolerance: 1e-9,
                })
                .collect();
        ExternalTrustState::ExactRequestVerified(Box::new(
            verify_external_object(object, &probes).expect("probes agree with the known truth"),
        ))
    }

    fn source(assumptions: &[String], evidence: &[SourceId]) -> ProviderTransportSource {
        let object = law();
        let trust = verified(&object);
        ProviderTransportSource {
            component: "trial_a_provider".into(),
            object,
            trust,
            quantity: quantity("trial_a", "do(a=1)"),
            semantic_identity: "estimand:recovery_under_treatment".into(),
            support: SupportDomain::Real,
            assumptions: assumptions.to_vec(),
            evidence: EvidenceBasis::Declared(evidence.iter().copied().collect()),
            requirement: SourceTrustRequirement::ExactRequestVerified,
        }
    }

    fn evidence() -> Vec<SourceId> {
        let (_, ids) = super::three_sources();
        ids.to_vec()
    }

    /// `contract.provider_transport.AdmittedProviderSource.canonical_identity`: assumptions and
    /// evidence are sets.
    #[test]
    fn site_admitted_provider_source_sets() {
        let assumptions: Vec<String> =
            ["invariant", "no unmeasured", "positivity"].into_iter().map(String::from).collect();
        check_set(&assumptions, |listed| {
            source(listed, &evidence()).admit().ok().map(|admitted| admitted.canonical_identity())
        });
        let ids = evidence();
        check_set(&ids, |listed| {
            source(&assumptions, listed).admit().ok().map(|admitted| admitted.canonical_identity())
        });
    }

    fn offer(transport: TransportIdentity, evidence: &[SourceId]) -> TransportedClaimOffer {
        TransportedClaimOffer {
            component: "transport_result".into(),
            port: "effect".into(),
            quantity: quantity("target", "do(a=1)"),
            semantic_identity: "estimand:recovery_under_treatment".into(),
            support: SupportDomain::Real,
            operations: [ContractOperation::Mean].into_iter().collect(),
            uncertainty: None,
            transport,
            evidence: evidence.iter().copied().collect(),
        }
    }

    fn transport(populations: &[String], assumptions: &[String]) -> TransportIdentity {
        TransportIdentity {
            source_populations: populations.to_vec(),
            target_population: "target".into(),
            selection_diagram: digest(7),
            assumptions: assumptions.to_vec(),
        }
    }

    /// `contract.provider_transport.TransportedClaimOffer.canonical_identity`: the transport's
    /// populations and assumptions are sets (as the ledger reads them), and so is the evidence.
    #[test]
    fn site_transported_claim_offer_sets() {
        let populations: Vec<String> = ["trial_a", "trial_b", "trial_c"].map(String::from).to_vec();
        let assumptions: Vec<String> = ["invariant", "s-admissible"].map(String::from).to_vec();
        let identity = |offer: TransportedClaimOffer| {
            offer.validate().ok().map(|()| offer.canonical_identity())
        };
        check_set(&populations, |listed| {
            identity(offer(transport(listed, &assumptions), &evidence()))
        });
        check_set(&assumptions, |listed| {
            identity(offer(transport(&populations, listed), &evidence()))
        });
        let ids = evidence();
        check_set(&ids, |listed| identity(offer(transport(&populations, &assumptions), listed)));
    }
}

// ---------------------------------------------------------------------------
// Callback protocol.
// ---------------------------------------------------------------------------

mod callback {
    use std::collections::BTreeSet;

    use antecedent_contract::callback_protocol::{
        CallbackDeclaration, CallbackDependency, CallbackInput, CallbackKind, CallbackOperation,
        CallbackQuery, CallbackRequest, CallbackTrust, DependencyDeclaration, DependencyKind,
        InputSupplier, ReplayContract,
    };
    use antecedent_contract::interface_contract::{EvidenceBasis, SupportDomain};
    use antecedent_contract::{DistributionMeaning, ProviderObjectIdentity};

    use super::{check_canonical_set, check_set};

    fn gaussian() -> CallbackDeclaration {
        CallbackDeclaration {
            component: "gaussian_law".into(),
            kind: CallbackKind::Law,
            identity: ProviderObjectIdentity {
                provider_id: "lab".into(),
                object_id: "gaussian".into(),
                version_id: "1.0".into(),
                snapshot_id: "snap-1".into(),
                request_id: "req-1".into(),
            },
            quantity: super::provider::quantity("adults", "do(a=1)"),
            semantic_identity: "estimand:outcome_law".into(),
            meaning: DistributionMeaning::InterventionalPredictive,
            support: SupportDomain::Real,
            operations: [CallbackOperation::Mean, CallbackOperation::Sample].into_iter().collect(),
            update: None,
            inputs: Vec::new(),
            dependencies: DependencyDeclaration::Complete(vec![CallbackDependency::new(
                DependencyKind::Model,
                "parameters",
                "mu=2;sd=1.5",
            )]),
            replay: ReplayContract::DeterministicPure,
            trust: CallbackTrust::ExternallyAttested { attestor: "lab".into() },
            evidence: EvidenceBasis::Declared(BTreeSet::new()),
        }
    }

    fn input(name: &str) -> CallbackInput {
        let base = gaussian();
        CallbackInput {
            name: name.into(),
            quantity: base.quantity.clone(),
            semantic_identity: format!("estimand:{name}"),
            support: base.support,
            operation: CallbackOperation::Mean,
            uncertainty: None,
            supplier: InputSupplier::External,
        }
    }

    fn identity(declaration: &CallbackDeclaration) -> Option<antecedent_core::SemanticDigest> {
        declaration.validate().ok().map(|()| declaration.canonical_identity())
    }

    /// `contract.callback_protocol.declaration.CallbackDeclaration.canonical_identity`: inputs
    /// and dependencies are sets (validation refuses a repeat); categorical levels are a set
    /// accepted only in its canonical (strictly ascending) spelling.
    #[test]
    fn site_callback_declaration_collections() {
        let inputs = [input("a"), input("b"), input("c")];
        check_set(&inputs, |listed| {
            let mut declaration = gaussian();
            declaration.inputs = listed.to_vec();
            identity(&declaration)
        });
        let dependencies = [
            CallbackDependency::new(DependencyKind::Model, "parameters", "mu=2"),
            CallbackDependency::new(DependencyKind::Environment, "runtime", "1"),
            CallbackDependency::new(DependencyKind::Model, "weights", "w=3"),
        ];
        check_set(&dependencies, |listed| {
            let mut declaration = gaussian();
            declaration.dependencies = DependencyDeclaration::Complete(listed.to_vec());
            identity(&declaration)
        });
        let levels: Vec<String> = ["high", "low", "mid"].map(String::from).to_vec();
        // An output is never categorical; an input may be, and its levels are the set.
        check_canonical_set(&levels, |listed| {
            let mut declaration = gaussian();
            let mut categorical = input("group");
            categorical.support = SupportDomain::Categorical { levels: listed.to_vec() };
            categorical.quantity.units = "category".into();
            declaration.inputs = vec![categorical];
            identity(&declaration)
        });
    }

    /// `contract.callback_protocol.plan.CallbackRequest.identity`: input digests are a set keyed
    /// by input name.
    #[test]
    fn site_callback_request_input_digests() {
        let declaration = gaussian();
        let pairs = [("a", "d1"), ("b", "d2"), ("c", "d3")];
        check_set(&pairs, |listed| {
            let mut request = CallbackRequest::new(CallbackQuery::Mean, 1);
            for (name, digest) in listed {
                request.input_digests.insert((*name).to_owned(), (*digest).to_owned());
            }
            Some(request.identity(&declaration))
        });
    }
}

// ---------------------------------------------------------------------------
// Interface contracts and their composition.
// ---------------------------------------------------------------------------

mod interface {
    use antecedent_contract::interface_contract::{
        ComponentConnection, ConsumedPort, ContractOperation, CyclicComponentDeclaration,
        InterfaceContract, ProducedPort, SupportDomain, check_composition,
        check_composition_with_cyclic_components,
    };

    use super::{check_set, digest};

    fn produced(name: &str) -> ProducedPort {
        ProducedPort::new(
            name,
            super::provider::quantity("adults", "do(a=1)"),
            format!("estimand:{name}"),
            SupportDomain::Real,
            [ContractOperation::Mean],
        )
    }

    fn consumed(name: &str) -> ConsumedPort {
        ConsumedPort::new(
            name,
            super::provider::quantity("adults", "do(a=1)"),
            format!("estimand:{name}"),
            SupportDomain::Real,
            [ContractOperation::Mean],
        )
    }

    fn contract(
        component: &str,
        produces: &[ProducedPort],
        consumes: &[ConsumedPort],
    ) -> InterfaceContract {
        InterfaceContract {
            component: component.into(),
            produces: produces.to_vec(),
            consumes: consumes.to_vec(),
        }
    }

    fn identity(contract: &InterfaceContract) -> Option<antecedent_core::SemanticDigest> {
        contract.validate().ok().map(|()| contract.canonical_identity())
    }

    /// `contract.interface_contract.declaration.InterfaceContract.canonical_identity`: produced
    /// and consumed ports are sets by name (validation refuses a repeated name).
    #[test]
    fn site_interface_contract_ports() {
        let produces = [produced("a"), produced("b"), produced("c")];
        check_set(&produces, |listed| identity(&contract("p", listed, &[])));
        let consumes = [consumed("a"), consumed("b"), consumed("c")];
        check_set(&consumes, |listed| identity(&contract("p", &[produced("out")], listed)));
    }

    fn components() -> Vec<InterfaceContract> {
        vec![
            contract("p", &[produced("effect")], &[]),
            contract("c1", &[produced("one")], &[consumed("effect")]),
            contract("c2", &[produced("two")], &[consumed("effect")]),
        ]
    }

    fn connections() -> Vec<ComponentConnection> {
        vec![
            ComponentConnection::new("p", "effect", "c1", "effect"),
            ComponentConnection::new("p", "effect", "c2", "effect"),
        ]
    }

    /// `contract.interface_contract.composition.compose` and `decide_connections`: components
    /// and connections are sets; the feed-forward order is derived, never caller order.
    #[test]
    fn site_composition_components_and_connections() {
        let all = components();
        check_set(&all[1..], |listed| {
            let mut declared = vec![all[0].clone()];
            declared.extend(listed.iter().cloned());
            let wired: Vec<ComponentConnection> = connections()
                .into_iter()
                .filter(|c| declared.iter().any(|d| d.component == c.consumer))
                .collect();
            check_composition(&declared, &wired).ok().map(|composed| composed.identity)
        });
        let wires = connections();
        check_set(&wires, |listed| {
            check_composition(&components(), listed).ok().map(|composed| composed.identity)
        });
    }

    /// `contract.interface_contract.composition.check_composition_with_cyclic_components`: the
    /// declared cyclic products are a set (a repeated component is refused).
    #[test]
    fn site_composition_cyclic_components() {
        let products = [
            CyclicComponentDeclaration {
                component: "c1".into(),
                program: digest(1),
                cyclic_solution: digest(2),
            },
            CyclicComponentDeclaration {
                component: "c2".into(),
                program: digest(3),
                cyclic_solution: digest(4),
            },
        ];
        check_set(&products, |listed| {
            check_composition_with_cyclic_components(&components(), &connections(), listed, None)
                .ok()
                .map(|composed| composed.identity)
        });
    }
}

// ---------------------------------------------------------------------------
// Composition verification.
// ---------------------------------------------------------------------------

mod composition {
    use std::collections::BTreeMap;

    use antecedent_contract::composition_verification::{
        CompositionGraph, CompositionNode, CompositionView, DependencyKind, NodeKind,
        SharedDataRelation, revalidate,
    };

    use super::{check_multiset, check_set, digest};

    fn node(key: &str, kind: NodeKind, id: u8) -> CompositionNode {
        CompositionNode::new(key, kind, digest(id))
    }

    fn nodes() -> Vec<CompositionNode> {
        vec![
            node("cohort", NodeKind::CohortSource, 1),
            node("data", NodeKind::Data, 2),
            node("result", NodeKind::LedgerResult, 3),
            node("claim", NodeKind::Claim, 4),
        ]
    }

    fn edges() -> Vec<(&'static str, &'static str, DependencyKind)> {
        vec![
            ("result", "cohort", DependencyKind::Data),
            ("result", "data", DependencyKind::Evidence),
            ("claim", "result", DependencyKind::Data),
        ]
    }

    fn graph(
        nodes: &[CompositionNode],
        edges: &[(&str, &str, DependencyKind)],
    ) -> Option<CompositionGraph> {
        let mut g = CompositionGraph::new();
        for n in nodes {
            g.add_node(n.clone()).ok()?;
        }
        for (down, up, kind) in edges {
            g.add_edge(down, up, *kind).ok()?;
        }
        Some(g)
    }

    /// `contract.composition_verification.graph.CompositionGraph.identity`: nodes (by unique
    /// key) and edges are sets.
    #[test]
    fn site_composition_graph_nodes_and_edges() {
        let all = nodes();
        check_set(&all[..2], |listed| {
            let mut declared = listed.to_vec();
            declared.extend(all[2..].iter().cloned());
            let wired: Vec<_> = edges()
                .into_iter()
                .filter(|(d, u, _)| {
                    declared.iter().any(|n| n.key == *d) && declared.iter().any(|n| n.key == *u)
                })
                .collect();
            graph(&declared, &wired).map(|g| g.identity())
        });
        let wires = edges();
        check_set(&wires, |listed| graph(&nodes(), listed).map(|g| g.identity()));
    }

    /// `contract.composition_verification.graph.CompositionGraph.add_node`: a node's assumptions,
    /// evidence and unsupported coordinates are sets.
    #[test]
    fn site_composition_node_sets() {
        let with = |n: CompositionNode| {
            let mut declared = nodes();
            declared[2] = n;
            graph(&declared, &edges()).map(|g| g.identity())
        };
        let assumptions: Vec<String> =
            ["exchangeability", "positivity", "sufficiency"].map(String::from).to_vec();
        check_set(&assumptions, |listed| {
            with(node("result", NodeKind::LedgerResult, 3).with_assumptions(listed.to_vec()))
        });
        let (_, ids) = super::three_sources();
        check_set(&ids, |listed| {
            with(node("result", NodeKind::LedgerResult, 3).with_evidence(listed.to_vec()))
        });
        let coordinates: Vec<String> =
            ["age above 90", "income above 200k"].map(String::from).to_vec();
        check_set(&coordinates, |listed| {
            with(node("result", NodeKind::LedgerResult, 3).with_unsupported(listed.to_vec()))
        });
    }

    /// `contract.composition_verification.revalidate.GraphRevalidation.identity`: the removed
    /// nodes and the per-claim reasons are sets: the previous graph's insertion order never
    /// reaches the identity, and removing another node changes it.
    #[test]
    fn site_graph_revalidation_removed() {
        let current = graph(&nodes(), &edges()).unwrap();
        let extra = [
            node("old_a", NodeKind::Data, 10),
            node("old_b", NodeKind::Data, 11),
            node("old_c", NodeKind::Data, 12),
        ];
        check_set(&extra, |listed| {
            let mut previous_nodes = nodes();
            previous_nodes.extend(listed.iter().cloned());
            let previous = graph(&previous_nodes, &edges())?;
            revalidate(&previous, &current, &BTreeMap::new()).ok().map(|r| r.identity())
        });
    }

    /// `contract.composition_verification.view.CompositionView.of`: lineage edges and evidence
    /// are sets; the shared-data relations a caller passes are listed sorted, a repeat kept.
    #[test]
    fn site_composition_view() {
        let current = graph(&nodes(), &edges()).unwrap();
        let revalidation = revalidate(&current, &current, &BTreeMap::new()).unwrap();
        let relations = [
            SharedDataRelation {
                first: "cohort".into(),
                second: "data".into(),
                relation: "shared_data",
                shared_observations: 3,
            },
            SharedDataRelation {
                first: "data".into(),
                second: "result".into(),
                relation: "shared_prior",
                shared_observations: 0,
            },
        ];
        check_multiset(&relations, |listed| {
            CompositionView::of(&current, &revalidation, "claim", listed).ok()
        });
        let wires = edges();
        check_set(&wires, |listed| {
            let g = graph(&nodes(), listed)?;
            let r = revalidate(&g, &g, &BTreeMap::new()).ok()?;
            CompositionView::of(&g, &r, "claim", &[]).ok()
        });
    }
}

// ---------------------------------------------------------------------------
// Scientific bridges.
// ---------------------------------------------------------------------------

mod bridge {
    use antecedent_contract::bridge::{
        AssumptionKind, BridgeAssumption, BridgeContract, BridgeEndpoint, BridgeKind,
        BridgeMapping, BridgeScope, QuantityClass, TrialEffectPair, TrialLevelValidation,
        ValueKind,
    };
    use antecedent_core::CanonicalWriter;

    use super::check_multiset;

    fn assumption(id: &str) -> BridgeAssumption {
        BridgeAssumption {
            id: id.into(),
            kind: AssumptionKind::MeasurementModel,
            statement: format!("{id} holds"),
            evidence: None,
        }
    }

    fn contract(assumptions: &[BridgeAssumption], mapping: BridgeMapping) -> BridgeContract {
        BridgeContract {
            kind: BridgeKind::LatentMeasuredOutcome,
            source: BridgeEndpoint {
                quantity: super::provider::quantity("adults", "observational"),
                class: QuantityClass::LatentConstruct,
            },
            target: BridgeEndpoint {
                quantity: super::provider::quantity("adults", "observational"),
                class: QuantityClass::LatentConstruct,
            },
            source_value: ValueKind::Mean,
            target_value: ValueKind::Mean,
            evidence_scope: BridgeScope {
                population_id: "adults".into(),
                regime_id: "observational".into(),
            },
            mapping,
            assumptions: assumptions.to_vec(),
            transport: None,
            mapping_uncertainty: None,
            source_dependence: None,
            estimand_bridge: None,
        }
    }

    /// `contract.bridge.contract.BridgeContract.identity`: assumptions are ordered by id and by
    /// their own encoding, so listing order never reaches the identity; a repeat is kept
    /// (validation refuses a repeated id, so a valid contract's assumptions are a set).
    #[test]
    fn site_bridge_contract_assumptions() {
        let assumptions = [assumption("a"), assumption("b"), assumption("c")];
        check_multiset(&assumptions, |listed| {
            Some(contract(listed, BridgeMapping::Identity).identity())
        });
    }

    fn trial_mapping(trials: &[TrialEffectPair]) -> BridgeMapping {
        BridgeMapping::TrialLevelLinear(TrialLevelValidation {
            trials: trials.to_vec(),
            min_trials: 3,
            min_r_squared: 0.5,
        })
    }

    fn mapping_digest(mapping: &BridgeMapping) -> [u8; 32] {
        let mut writer = CanonicalWriter::new("antecedent.adversarial.bridge_mapping");
        mapping.write_canonical(&mut writer);
        writer.finish()
    }

    /// `contract.bridge.mapping.BridgeMapping.write_canonical`: validation trials are a multiset
    /// (a repeated trial is more evidence), written by bit pattern.
    #[test]
    fn site_bridge_mapping_trials() {
        let trials =
            [(0.0, 0.1), (0.5, 0.5), (1.0, 0.9)].map(|(surrogate_effect, terminal_effect)| {
                TrialEffectPair { surrogate_effect, terminal_effect }
            });
        check_multiset(&trials, |listed| Some(mapping_digest(&trial_mapping(listed))));
        crate::common::bit_exact(|value| {
            Some(mapping_digest(&trial_mapping(&[TrialEffectPair {
                surrogate_effect: value,
                terminal_effect: 1.0,
            }])))
        })
        .unwrap();
    }
}

// ---------------------------------------------------------------------------
// Program binding.
// ---------------------------------------------------------------------------

mod program {
    use antecedent_contract::program_binding::ProgramBinding;
    use proptest::prelude::*;

    use super::check_canonical_set;

    fn binding() -> ProgramBinding {
        ProgramBinding {
            graph_id: "graph:g1".into(),
            contract_id: "contract:c1".into(),
            treatment_id: "schema:dose".into(),
            outcome_id: "schema:y".into(),
            population_id: "target".into(),
            intervention_kind: "do".into(),
            horizon: 0,
            dose_grid: vec![0.5, 1.0, 2.0],
            dose_units: "mg".into(),
            outcome_units: "mmHg".into(),
            functional_id: "mean".into(),
            transform_id: "identity".into(),
        }
    }

    fn identity(binding: &ProgramBinding) -> Option<String> {
        binding.validate().ok().map(|()| binding.identity())
    }

    /// `contract.program_binding.ProgramBinding.identity`: the dose grid is a set admitted only
    /// strictly increasing, written by bit pattern.
    #[test]
    fn site_program_binding_dose_grid() {
        let grid = binding().dose_grid;
        check_canonical_set(&grid, |listed| {
            let mut b = binding();
            b.dose_grid = listed.to_vec();
            identity(&b)
        });
        crate::common::bit_exact(|dose| {
            let mut b = binding();
            b.dose_grid = vec![dose, 1.0];
            identity(&b)
        })
        .unwrap();
    }

    proptest! {
        #![proptest_config(crate::common::config())]

        /// The text fields are framed: two bindings share an identity only when every field
        /// agrees, whatever separators the text holds.
        #[test]
        fn site_program_binding_framing(
            left in (crate::common::adversarial_text(), crate::common::adversarial_text()),
            right in (crate::common::adversarial_text(), crate::common::adversarial_text()),
        ) {
            let build = |(units, functional): &(String, String)| {
                let mut b = binding();
                b.dose_units = format!("u{units}");
                b.functional_id = format!("f{functional}");
                b
            };
            crate::common::injective(&left, &right, Clone::clone, |parts| identity(&build(parts)))?;
        }
    }
}

// ---------------------------------------------------------------------------
// Adaptive and choice designs.
// ---------------------------------------------------------------------------

mod adaptive {
    use antecedent_contract::adaptive_design::{
        AdaptiveDesignContract, AdaptiveDesignDeclaration, AssignmentFamily, ChoiceDesignContract,
        ChoiceDesignDeclaration, ChoiceLook, ChoiceRespondent, DelayMechanism, InterimAnalysis,
        MonitoringStrategy, OutcomeAvailability, OutcomeStructure, PolicyEpoch, ProcedureCell,
        RetainedChoiceSource, StoppingRuleDeclaration, StoppingRuleKind, StoppingSpec,
    };

    use super::{check_ordered, check_set, digest};

    fn declaration() -> AdaptiveDesignDeclaration {
        let cell = ProcedureCell {
            assignment: AssignmentFamily::ThompsonBetaBernoulli,
            outcome: OutcomeStructure::SingleOutcome,
            monitoring: MonitoringStrategy::FixedInterimLooks,
            stopping: StoppingRuleKind::Efficacy,
        };
        let assigned_arm = vec![0, 1, 0, 1, 1, 0, 1, 0];
        AdaptiveDesignDeclaration {
            label: "trial".to_owned(),
            cell,
            arm_labels: vec!["control".to_owned(), "treatment".to_owned()],
            assigned_arm: assigned_arm.clone(),
            policy_history: Some(vec![
                PolicyEpoch { first_unit: 0, policy_identity: digest(1) },
                PolicyEpoch { first_unit: 4, policy_identity: digest(2) },
            ]),
            assignment_probabilities: Some(
                (0..8)
                    .map(|unit| if unit >= 4 { vec![0.3, 0.7] } else { vec![0.5, 0.5] })
                    .collect(),
            ),
            exposure: Some(assigned_arm),
            availability: Some(OutcomeAvailability {
                available_at_look: (0..8).map(|unit| Some(u32::from(unit >= 4))).collect(),
                delay: DelayMechanism::Immediate,
            }),
            outcomes: [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0].iter().map(|&y| vec![y]).collect(),
            interim_analyses: Some(vec![
                InterimAnalysis { units_enrolled: 4, information_fraction: 0.5 },
                InterimAnalysis { units_enrolled: 8, information_fraction: 1.0 },
            ]),
            stopping_rule: Some(StoppingRuleDeclaration {
                kind: StoppingRuleKind::Efficacy,
                spec: StoppingSpec::PosteriorProbabilityBest { threshold: 0.95 },
                ended_by_rule: false,
            }),
            probability_floor: 0.05,
        }
    }

    /// `contract.adaptive_design.declaration.identity_of`: arms, per-unit assignments and
    /// outcomes are ordered (unit order is enrolment order; arm labels name arm indices).
    #[test]
    fn site_adaptive_design_identity_lists_are_ordered() {
        let identity = |d: AdaptiveDesignDeclaration| {
            AdaptiveDesignContract::new(d).ok().map(|c| c.identity())
        };
        let arms = declaration().arm_labels;
        check_ordered(&arms, |listed| {
            let mut d = declaration();
            d.arm_labels = listed.to_vec();
            identity(d)
        });
        let outcomes = declaration().outcomes;
        check_ordered(&outcomes[..4], |listed| {
            let mut d = declaration();
            d.outcomes[..4].clone_from_slice(listed);
            identity(d)
        });
    }

    /// `contract.adaptive_design.accounting.AdaptiveDesignContract.interim_evidence`: each look's
    /// rows are a set registered through the ledger (whose observation rows are a set,
    /// `site_source_declaration_observation_rows`); two contracts over the same units account
    /// identically and every look reuses the earlier rows.
    #[test]
    fn site_adaptive_interim_evidence() {
        let contract = AdaptiveDesignContract::new(declaration()).unwrap();
        let first = contract.interim_evidence().unwrap();
        let second =
            AdaptiveDesignContract::new(declaration()).unwrap().interim_evidence().unwrap();
        assert_eq!(first.look_sources, second.look_sources);
        assert_eq!(first.mass, second.mass);
        assert_eq!(first.mass.distinct_observations, 8);
    }

    fn choice(sources: Vec<RetainedChoiceSource>) -> ChoiceDesignDeclaration {
        let labels: Vec<String> = ["a", "b", "c"].map(String::from).to_vec();
        ChoiceDesignDeclaration {
            label: "wave".into(),
            cell: ProcedureCell {
                assignment: AssignmentFamily::FixedRandomization,
                outcome: OutcomeStructure::CategoricalChoice,
                monitoring: MonitoringStrategy::FinalAnalysisOnly,
                stopping: StoppingRuleKind::NoStopping,
            },
            option_labels: labels,
            meaning: "chosen in the pricing task".into(),
            population: "customers".into(),
            respondents: (0..6)
                .map(|i| {
                    let mut counts = vec![0, 0, 0];
                    counts[(i % 3) as usize] = 1;
                    ChoiceRespondent { id: format!("r{i}"), counts }
                })
                .collect(),
            initial_prior: vec![1.0; 3],
            retained_sources: sources,
            looks: Some(vec![ChoiceLook { respondents: 6 }]),
            stopping_rule: Some(StoppingRuleDeclaration {
                kind: StoppingRuleKind::NoStopping,
                spec: StoppingSpec::NoStopping,
                ended_by_rule: false,
            }),
        }
    }

    fn source(ids: &[String]) -> RetainedChoiceSource {
        RetainedChoiceSource {
            label: "earlier".into(),
            option_labels: ["a", "b", "c"].map(String::from).to_vec(),
            meaning: "chosen in the pricing task".into(),
            population: "customers".into(),
            counts: vec![3.0, 0.0, 0.0],
            observation_ids: ids.to_vec(),
            discount: 0.5,
            transfer_rationale: None,
            posterior_proof: None,
        }
    }

    fn choice_identity(ids: &[String]) -> Option<antecedent_core::SemanticDigest> {
        ChoiceDesignContract::new(choice(vec![source(ids)])).ok().map(|c| c.identity())
    }

    /// `contract.adaptive_design.choice.identity_of` and `contract.adaptive_design.choice.account`:
    /// a retained source's observation ids are a set of rows: reordering keeps the identity,
    /// and `p` and `p#0` name one row.
    #[test]
    fn site_choice_retained_rows_are_a_set() {
        let ids: Vec<String> = ["p1", "p2", "p3"].map(String::from).to_vec();
        check_set(&ids, choice_identity);
        let spelled: Vec<String> = ["p1#0", "p2", "p3#0"].map(String::from).to_vec();
        assert_eq!(choice_identity(&ids), choice_identity(&spelled));
        // The contract identity stored under version 1 still verifies.
        let contract = ChoiceDesignContract::new(choice(vec![source(&ids)])).unwrap();
        assert!(contract.accepts_identity_hex(&contract.identity().to_hex()));
        assert!(!contract.accepts_identity_hex(&"0".repeat(64)));
    }
}

// ---------------------------------------------------------------------------
// Transport revalidation.
// ---------------------------------------------------------------------------

mod transport_revalidation {
    use std::collections::BTreeMap;

    use antecedent_contract::composition_verification::SupportStanding;
    use antecedent_contract::transport_revalidation::{
        SceneClaim, SceneClaimKind, SceneSource, SceneTarget, TransportScene, revalidate_transport,
    };

    use super::{check_set, digest};

    fn scene(sources: &[SceneSource], claim_sources: &[String]) -> TransportScene {
        let mut effect =
            SceneClaim::new("effect", SceneClaimKind::Transported, digest(31), digest(50));
        effect.structure = Some(digest(40));
        effect.sources = claim_sources.to_vec();
        TransportScene {
            target: SceneTarget { identity: digest(10), support: SupportStanding::Supported },
            selection_diagram: digest(20),
            sources: sources.to_vec(),
            claims: vec![effect],
        }
    }

    fn sources() -> Vec<SceneSource> {
        vec![
            SceneSource { key: "trial".into(), identity: digest(1) },
            SceneSource { key: "registry".into(), identity: digest(2) },
            SceneSource { key: "cohort".into(), identity: digest(3) },
        ]
    }

    /// `contract.transport_revalidation.TransportRevalidation.identity`: a scene's sources (by
    /// key) and a claim's sources are sets; the claims and reasons are listed by key.
    #[test]
    fn site_transport_revalidation_sets() {
        let identity = |scene: &TransportScene| {
            revalidate_transport(scene, scene, &BTreeMap::new()).ok().map(|r| r.identity())
        };
        let all: Vec<String> = ["trial", "registry", "cohort"].map(String::from).to_vec();
        check_set(&sources(), |listed| identity(&scene(listed, &all[..1])));
        check_set(&all, |listed| identity(&scene(&sources(), listed)));
    }
}
