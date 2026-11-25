//! Standing laws of the contract crate over generated inputs, against the production
//! functions the standing-site inventory (`formal/standing_sites.toml`) names: a bridged
//! claim asserts nothing its source does not establish for the target and keeps every
//! obligation its source left unresolved; a composition graph's support facet and an
//! undeclared trust only ever lower a claim; and a ledger result resting on one a
//! combination replaced is never read as valid, even before the downstream walk.

use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::Arc;

use antecedent_contract::bridge::{
    AssumptionKind, BridgeAssumption, BridgeContract, BridgeEndpoint, BridgeKind, BridgeMapping,
    BridgeScope, BridgeSource, QuantityClass, SOURCE_ASSUMPTIONS_UNAVAILABLE, ValueKind,
    derive_bridged_claim,
};
use antecedent_contract::composition_verification::{
    CompositionGraph, CompositionNode, DependencyKind, NodeKind, PriorAuthority, SupportStanding,
    TrustStanding, revalidate,
};
use antecedent_contract::evidence_ledger::{
    CombinationRequest, EvidenceLedger, ObservationId, PoolingDependence, ResultId, Revalidation,
    SourceDeclaration, SourceKind, SynthesisOperation,
};
use antecedent_contract::{
    AssumptionSlot, ClaimDomains, ClaimEnvelope, ClaimKind, DomainStatus, ObligationKind,
    ObligationRecord, ObligationScope, QuantityRole, ReasoningView, ScientificQuantity,
    SlotAvailability,
};
use antecedent_core::assumption::{AssumptionSource, AssumptionStatus};
use antecedent_core::{ContractIdentities, SemanticDigest};
use proptest::prelude::*;

fn config() -> ProptestConfig {
    ProptestConfig { cases: 192, failure_persistence: None, ..ProptestConfig::default() }
}

fn digest(byte: u8) -> SemanticDigest {
    SemanticDigest::from_bytes([byte; 32])
}

// -- bridges ----------------------------------------------------------------------------------

const STATUSES: [DomainStatus; 8] = [
    DomainStatus::Identified,
    DomainStatus::Supported,
    DomainStatus::Evaluated,
    DomainStatus::OutsideScope,
    DomainStatus::Contradicted,
    DomainStatus::Unknown,
    DomainStatus::Limited,
    DomainStatus::NotClaimed,
];
const KINDS: [ObligationKind; 6] = [
    ObligationKind::UserAssertion,
    ObligationKind::GraphicalImplication,
    ObligationKind::EmpiricalDiagnostic,
    ObligationKind::FailedCheck,
    ObligationKind::Uncheckable,
    ObligationKind::CheckNotRun,
];
const ASSUMPTION_STATUSES: [AssumptionStatus; 4] = [
    AssumptionStatus::Declared,
    AssumptionStatus::Supported,
    AssumptionStatus::Contradicted,
    AssumptionStatus::Untestable,
];

fn quantity(variable: &str, units: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable.into(),
        variable_name: variable.into(),
        role: QuantityRole::Outcome,
        units: units.into(),
        population_id: "adults".into(),
        regime_id: "do(t=1)".into(),
        horizon: 0,
        functional_id: "average_effect".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

fn contract() -> BridgeContract {
    BridgeContract {
        kind: BridgeKind::SurrogateTerminalOutcome,
        source: BridgeEndpoint {
            quantity: quantity("trial:biomarker", "mmol_per_l"),
            class: QuantityClass::SurrogateEndpoint,
        },
        target: BridgeEndpoint {
            quantity: quantity("trial:survival", "log_hazard"),
            class: QuantityClass::TerminalOutcome,
        },
        source_value: ValueKind::Mean,
        target_value: ValueKind::Mean,
        evidence_scope: BridgeScope { population_id: "adults".into(), regime_id: "do(t=1)".into() },
        mapping: BridgeMapping::Affine { intercept: 0.06, slope: 0.92 },
        assumptions: vec![BridgeAssumption {
            id: "validation".into(),
            kind: AssumptionKind::SurrogateValidation,
            statement: "the surrogate is validated at the trial level".into(),
            evidence: Some("meta-analysis:ckd-2024".into()),
        }],
        transport: None,
        mapping_uncertainty: None,
        source_dependence: None,
        estimand_bridge: None,
    }
}

/// The source's assumption slot: unavailable, or a list of (kind, status) obligations.
type SourceSlot = Option<Vec<(usize, usize)>>;

fn source_slot() -> impl Strategy<Value = SourceSlot> {
    prop::option::of(prop::collection::vec((0usize..KINDS.len(), 0usize..4), 0..5))
}

fn source_claim(domains: [usize; 3], slot: &SourceSlot, value: f64) -> ClaimEnvelope {
    let assumptions = match slot {
        None => SlotAvailability::unavailable("fixture"),
        Some(records) => SlotAvailability::Available(AssumptionSlot::new(
            records
                .iter()
                .enumerate()
                .map(|(index, &(kind, status))| {
                    ObligationRecord::new(
                        format!("source_premise_{index}"),
                        ObligationScope::Program,
                        AssumptionSource::UserDeclared,
                        KINDS[kind],
                        ASSUMPTION_STATUSES[status],
                        "a premise of the source claim",
                    )
                })
                .collect::<Vec<_>>(),
        )),
    };
    ClaimEnvelope::new(
        digest(1),
        ContractIdentities::new(
            digest(10),
            digest(11),
            Some(digest(12)),
            Some(digest(13)),
            digest(14),
            digest(15),
            digest(16),
        ),
        ClaimKind::Point,
        Some(value),
        Some(Arc::from("mmol_per_l")),
        ReasoningView::new(
            SlotAvailability::unavailable("fixture"),
            SlotAvailability::unavailable("fixture"),
            SlotAvailability::unavailable("fixture"),
            assumptions,
        ),
        ClaimDomains::new(STATUSES[domains[0]], STATUSES[domains[1]], STATUSES[domains[2]]),
        None,
        [Arc::<str>::from("data:study")],
    )
}

// -- composition graphs ----------------------------------------------------------------------

const TRUSTS: [TrustStanding; 4] = [
    TrustStanding::NotApplicable,
    TrustStanding::Held,
    TrustStanding::Withdrawn,
    TrustStanding::Undeclared,
];

const DEPENDENCIES: [DependencyKind; 4] = [
    DependencyKind::Data,
    DependencyKind::Evidence,
    DependencyKind::Support,
    DependencyKind::StructuralDerivation,
];

/// Per node: identity byte before and now, trust index before and now, and the kind of its
/// dependency on the node before it.
type NodeSpec = (u8, u8, usize, usize, usize);

fn nodes() -> impl Strategy<Value = Vec<NodeSpec>> {
    prop::collection::vec((1u8..3, 1u8..3, 0usize..4, 0usize..4, 0usize..4), 1..6)
}

fn chain_graph(spec: &[NodeSpec], now: bool, unsupported: &[bool]) -> CompositionGraph {
    let mut graph = CompositionGraph::new();
    for (index, &(id_before, id_now, trust_before, trust_now, _)) in spec.iter().enumerate() {
        let (id, trust) = if now { (id_now, trust_now) } else { (id_before, trust_before) };
        let mut node = CompositionNode::new(format!("n{index}"), NodeKind::Claim, digest(id))
            .with_trust(TRUSTS[trust]);
        if unsupported.get(index).copied().unwrap_or(false) {
            node.support = SupportStanding::Unsupported { coordinates: vec![format!("c{index}")] };
        }
        graph.add_node(node).expect("a well-formed node");
    }
    for (index, &(.., kind)) in spec.iter().enumerate().skip(1) {
        graph
            .add_edge(&format!("n{index}"), &format!("n{}", index - 1), DEPENDENCIES[kind])
            .expect("an acyclic edge");
    }
    graph
}

// -- ledgers ------------------------------------------------------------------------------------

fn obs(range: Range<u32>) -> Vec<ObservationId> {
    range.map(|i| ObservationId::from_key("cohort", &i.to_le_bytes())).collect()
}

fn reuse(label: String, of: ResultId) -> CombinationRequest {
    CombinationRequest {
        label,
        operation: SynthesisOperation::EvidenceReuse { of },
        replaces: None,
    }
}

proptest! {
    #![proptest_config(config())]

    /// A bridged claim asserts no domain about the target quantity, whatever its source
    /// asserts about its own, and carries every obligation its source left unresolved, or
    /// one standing for them when the source's assumptions are unavailable.
    #[test]
    fn bridged_claim_never_asserts_more_than_its_source(
        domains in [0usize..8, 0usize..8, 0usize..8],
        slot in source_slot(),
        value in 0.0f64..1.5,
    ) {
        let contract = contract();
        let source_claim = source_claim(domains, &slot, value);
        let source = BridgeSource { claim: &source_claim, quantity: &contract.source.quantity };
        let bridged = derive_bridged_claim(&contract, &source).expect("a supported bridge");
        for status in [
            bridged.claim.domains.identification,
            bridged.claim.domains.support,
            bridged.claim.domains.evaluated,
        ] {
            prop_assert_eq!(status, DomainStatus::Unknown);
        }
        prop_assert!(!bridged.claim.reasoning.identification.is_available());
        prop_assert!(!bridged.claim.reasoning.support.is_available());
        let carried = &bridged.claim.reasoning.assumptions.as_ref().expect("assumptions").obligations;
        prop_assert!(carried.iter().all(ObligationRecord::is_unresolved));
        let has = |id: &str| carried.iter().any(|record| &*record.id == id);
        let receipt_has = |id: &str| {
            bridged.receipt.unresolved.iter().any(|item| **item == *format!("source_assumption:{id}"))
        };
        match &source_claim.reasoning.assumptions {
            SlotAvailability::Available(slot) => {
                for record in slot.obligations.iter() {
                    prop_assert_eq!(has(&record.id), record.is_unresolved(), "{}", record.id);
                    prop_assert_eq!(receipt_has(&record.id), record.is_unresolved());
                }
                prop_assert!(!has(SOURCE_ASSUMPTIONS_UNAVAILABLE));
            }
            SlotAvailability::Unavailable { .. } => {
                prop_assert!(has(SOURCE_ASSUMPTIONS_UNAVAILABLE));
                prop_assert!(receipt_has(SOURCE_ASSUMPTIONS_UNAVAILABLE));
            }
        }
        prop_assert!(!bridged.receipt.equivalent_claim());
    }

    /// In a composition graph the support facet only ever lowers a claim: stating any set of
    /// nodes unsupported never improves any node's class or authority, and a node whose
    /// provider trust is undeclared is never still valid and never keeps its authority.
    #[test]
    fn graph_support_and_undeclared_trust_only_lower(
        spec in nodes(),
        marks in prop::collection::vec(any::<bool>(), 6),
    ) {
        let previous = chain_graph(&spec, false, &[]);
        let plain = chain_graph(&spec, true, &[]);
        let marked = chain_graph(&spec, true, &marks);
        let base = revalidate(&previous, &plain, &BTreeMap::new()).expect("a valid graph");
        let lowered = revalidate(&previous, &marked, &BTreeMap::new()).expect("a valid graph");
        for (index, &(.., trust_now, _)) in spec.iter().enumerate() {
            let key = format!("n{index}");
            let a = base.claim(&key).expect("classified");
            let b = lowered.claim(&key).expect("classified");
            prop_assert!(b.class >= a.class, "{} {:?} -> {:?}", key, a.class, b.class);
            if marks[index] {
                prop_assert_eq!(b.class, Revalidation::Withdrawn, "{} is unsupported", key);
            }
            if TRUSTS[trust_now] == TrustStanding::Undeclared {
                prop_assert_ne!(a.class, Revalidation::StillValid, "{} trust undeclared", key);
                prop_assert_ne!(a.authority, PriorAuthority::Retained);
            }
        }
    }

    /// Replacing a pooled result never leaves a result resting on it valid: before the
    /// downstream walk every reuse of it already reads as needing recomputation and is
    /// refused, and the walk then records exactly the classes that were read.
    #[test]
    fn ledger_replace_never_leaves_a_dependent_valid(
        widths in prop::collection::vec(1u32..20, 2..5),
        chain in 1usize..4,
    ) {
        let mut ledger = EvidenceLedger::new();
        let mut start = 0;
        let mut sources = Vec::new();
        for (index, width) in widths.iter().enumerate() {
            let declaration = SourceDeclaration::new(format!("s{index}"), SourceKind::ObservationSet)
                .with_observations(obs(start..start + width));
            start += width;
            sources.push(ledger.add_source(declaration).expect("a source"));
        }
        let pooling = |label: &str| CombinationRequest {
            label: label.to_owned(),
            operation: SynthesisOperation::StatisticalPooling {
                sources: sources.clone(),
                dependence: PoolingDependence::Independent,
            },
            replaces: None,
        };
        let pooled = ledger.combine(pooling("pooled")).expect("pooled");
        let mut dependents = Vec::new();
        let mut of = pooled;
        for step in 0..chain {
            of = ledger.combine(reuse(format!("cite{step}"), of)).expect("a citation");
            dependents.push(of);
        }
        let mut again = pooling("pooled_again");
        again.replaces = Some(pooled);
        let replacement = ledger.combine(again).expect("a replacement");
        for &dependent in &dependents {
            prop_assert_ne!(ledger.standing(dependent), Some(Revalidation::StillValid));
            prop_assert!(ledger.require_valid(dependent).is_err());
        }
        prop_assert_eq!(ledger.standing(replacement), Some(Revalidation::StillValid));
        let read = ledger.classification();
        let walked = ledger.revalidate_downstream().expect("the walk");
        prop_assert_eq!(walked, read);
        for &dependent in &dependents {
            prop_assert_ne!(ledger.standing(dependent), Some(Revalidation::StillValid));
        }
    }
}
