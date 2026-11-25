//! Site tests of the question digests of the supplied-claim, identified-set, contract-law and
//! decision-functional families: each compiles the public entry and reads the program
//! identity, so the laws run on the production declaration.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::identity_laws::{ADVERSARIAL_TEXT, all_distinct, bit_exact, injective, ordered_laws};
use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::{SemanticDigest, SupportStatus};
use antecedent_design::decision_adapters::ClaimProbability;
use antecedent_design::decision_contract::{
    ActionKind, AdmissibilityRules, DecisionAction, DecisionCriterion, UtilityExpr,
};
use antecedent_design::decision_robustness::InputSupport;
use antecedent_io::distribution_artifact::{
    DistributionCalibration, DistributionIdentity, DistributionMetadata, DistributionProvenance,
    DistributionTrust, DrawAlignment,
};
use antecedent_io::quantity_wire::DistributionMeaningWire;

fn quantity(variable: &str, regime: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable.into(),
        variable_name: variable.into(),
        role: QuantityRole::Outcome,
        units: "units".into(),
        population_id: "target".into(),
        regime_id: regime.into(),
        horizon: 0,
        functional_id: "outcome".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

fn columns() -> [ScientificQuantity; 2] {
    [quantity("a", "do(a=1)"), quantity("b", "do(a=0)")]
}

/// An exact law over the two columns with equally likely `rows`.
fn law(rows: &[[f64; 2]]) -> DistributionArtifact {
    let columns = columns();
    let identity = DistributionIdentity::new(
        DistributionMeaningWire::InterventionalPredictive,
        &columns,
        DrawAlignment::Joint,
        DistributionProvenance {
            source_id: "structure".into(),
            provider_id: "exact-law".into(),
            rng_id: "deterministic_exact".into(),
            snapshot_id: "enumeration".into(),
            causal_contract_id: "checked".into(),
        },
    )
    .expect("a valid law identity");
    DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity,
            axes: ["draw".into(), "quantity".into()],
            shape: [rows.len(), 2],
            weights: None,
            supported: None,
            calibration: DistributionCalibration::Exact,
            trust: DistributionTrust::Unverified,
            legacy_posterior: None,
            legacy_bindings: None,
        },
        rows.iter().flat_map(|row| row.iter().copied()).collect(),
    )
    .expect("a valid law")
}

fn contract() -> DecisionContract {
    let [a, b] = columns();
    let action = |id: &str, input: ScientificQuantity| DecisionAction {
        id: id.into(),
        kind: ActionKind::Intervention,
        inputs: vec![input],
        utility: UtilityExpr::Input(0),
    };
    DecisionContract {
        actions: vec![action("A", a), action("B", b)],
        utility_units: "units".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::Maximin,
    }
}

fn admissible() -> AdmissibleDecisionContract {
    AdmissibleDecisionContract { contract: contract(), rules: AdmissibilityRules::default() }
}

fn claim(id: &str, payoff: f64, probability: ClaimProbability) -> SuppliedClaim {
    SuppliedClaim {
        id: id.into(),
        probability,
        evidence: AtomEvidence::Evaluated(Box::new(law(&[[payoff, 1.0]]))),
        support: AtomSupport::supported(),
    }
}

fn claims_of(items: &[(&str, f64)]) -> Vec<SuppliedClaim> {
    items.iter().map(|(id, payoff)| claim(id, *payoff, ClaimProbability::Unspecified)).collect()
}

fn claims_identity(claims: SuppliedClaimSet) -> Option<SemanticDigest> {
    let ctx = ExecutionContext::for_tests(7);
    compile_supplied_claim_decision(admissible(), claims, &ctx)
        .ok()
        .map(|program| program.identity())
}

#[test]
fn site_claim_decision_programs_claims_are_ordered_and_their_reading_tagged() {
    let items = [("g1", 5.0), ("g2", 6.0), ("g3", 7.0)];
    // Ordered: the adapted atoms and the per-atom results keep the claims' order.
    ordered_laws(&items, |items| {
        claims_identity(SuppliedClaimSet::GraphDependent(claims_of(items)))
    });
    ordered_laws(&items, |items| {
        claims_identity(SuppliedClaimSet::FiniteScenarios(claims_of(items)))
    });
    ordered_laws(&items, |items| {
        claims_identity(SuppliedClaimSet::WeightedGraphAtoms(claims_of(items)))
    });
    let point = |id: &str, payoff: f64| SuppliedClaimSet::Point {
        id: id.into(),
        evidence: claim(id, payoff, ClaimProbability::Unspecified).evidence,
        support: AtomSupport::supported(),
    };
    let identities: Vec<SemanticDigest> = [
        SuppliedClaimSet::GraphDependent(claims_of(&items)),
        SuppliedClaimSet::WeightedGraphAtoms(claims_of(&items)),
        SuppliedClaimSet::FiniteScenarios(claims_of(&items)),
        SuppliedClaimSet::FiniteScenarios(claims_of(&items[..2])),
        point("g1", 5.0),
        point("g2", 5.0),
        point("g1", 6.0),
    ]
    .into_iter()
    .map(|claims| claims_identity(claims).expect("the declaration compiles"))
    .collect();
    all_distinct(&identities);
    // A claim's probability is written by its bits, and its id is framed.
    bit_exact(|p| {
        claims_identity(SuppliedClaimSet::WeightedGraphAtoms(vec![claim(
            "g1",
            5.0,
            ClaimProbability::Genuine(p),
        )]))
    });
    injective(
        &ADVERSARIAL_TEXT,
        |p| *p,
        |(a, b)| {
            claims_identity(SuppliedClaimSet::FiniteScenarios(claims_of(&[(*a, 5.0), (*b, 5.0)])))
        },
    );
}

#[test]
fn site_claim_decision_programs_point_support_entries_are_ordered() {
    let entry =
        |input: usize, status: SupportStatus| InputSupport { action_id: "A".into(), input, status };
    let entries = [
        entry(0, SupportStatus::Supported),
        entry(1, SupportStatus::WeakOverlap),
        entry(2, SupportStatus::Extrapolative),
    ];
    // Ordered: a lookup reads the first entry for an input.
    ordered_laws(&entries, |items| {
        claims_identity(SuppliedClaimSet::Point {
            id: "only".into(),
            evidence: claim("only", 5.0, ClaimProbability::Unspecified).evidence,
            support: AtomSupport { overall: SupportStatus::Supported, per_input: items.to_vec() },
        })
    });
}

fn carrier_identity(actions: Vec<IdentifiedAction>) -> Option<SemanticDigest> {
    let ctx = ExecutionContext::for_tests(7);
    compile_identified_set_decision(admissible(), IdentifiedSetCarrier::new(actions), &ctx)
        .ok()
        .map(|program| program.identity())
}

fn utility(action: &str, lower: f64, exclusions: &[&str]) -> IdentifiedUtility {
    IdentifiedUtility {
        action_id: action.into(),
        utility: IdentifiedSet { lower, upper: 9.0 },
        hard_exclusions: exclusions.iter().map(|id| (*id).to_owned()).collect(),
        support: AtomSupport::supported(),
    }
}

fn inputs(action: &str, sets: &[IdentifiedSet<f64>]) -> IdentifiedAction {
    IdentifiedAction::Inputs {
        action_id: action.into(),
        inputs: sets.to_vec(),
        hard_exclusions: vec![],
        support: AtomSupport::supported(),
    }
}

#[test]
fn site_claim_decision_programs_carrier_actions_inputs_and_exclusions_are_ordered() {
    // Ordered: the enclosed utilities are reported in the carrier's order.
    let actions = [
        IdentifiedAction::Utility(utility("A", 1.0, &[])),
        IdentifiedAction::Utility(utility("B", 2.0, &[])),
        inputs("C", &[IdentifiedSet { lower: 0.0, upper: 1.0 }]),
    ];
    ordered_laws(&actions, |items| carrier_identity(items.to_vec()));
    // Ordered: one identified set per declared input, by position.
    let sets = [
        IdentifiedSet { lower: 1.0, upper: 2.0 },
        IdentifiedSet { lower: 3.0, upper: 4.0 },
        IdentifiedSet { lower: 5.0, upper: 6.0 },
    ];
    ordered_laws(&sets, |items| carrier_identity(vec![inputs("A", items)]));
    // Ordered: the exclusions are reported with the action in the order supplied.
    ordered_laws(&["c1", "c2", "c3"], |items| {
        carrier_identity(vec![IdentifiedAction::Utility(utility("A", 1.0, items))])
    });
    bit_exact(|lower| carrier_identity(vec![IdentifiedAction::Utility(utility("A", lower, &[]))]));
    bit_exact(|lower| carrier_identity(vec![inputs("A", &[IdentifiedSet { lower, upper: 9.0 }])]));
    // The two declarations of one interval are different variants.
    let identities: Vec<SemanticDigest> = [
        vec![IdentifiedAction::Utility(utility("A", 1.0, &[]))],
        vec![inputs("A", &[IdentifiedSet { lower: 1.0, upper: 9.0 }])],
        vec![],
    ]
    .into_iter()
    .map(|actions| carrier_identity(actions).expect("the carrier compiles"))
    .collect();
    all_distinct(&identities);
}

fn means_source(
    pairs: &[(ScientificQuantity, f64)],
    estimate: MeanEstimate,
    provider: &str,
    snapshot: &str,
) -> ContractSource {
    ContractSource::Means(MeanSource {
        coordinates: pairs.iter().map(|(coordinate, _)| coordinate.clone()).collect(),
        means: pairs.iter().map(|(_, mean)| *mean).collect(),
        estimate,
        provider_id: provider.into(),
        snapshot_id: snapshot.into(),
        causal_contract_id: "checked".into(),
        rng_id: "none:mean_grid".into(),
    })
}

fn source_identity(source: ContractSource) -> Option<SemanticDigest> {
    let ctx = ExecutionContext::for_tests(7);
    compile_contract_decision(contract(), source, &ctx).ok().map(|program| program.identity())
}

#[test]
fn site_claim_decision_programs_means_are_ordered_pairs_and_the_source_tagged() {
    let coordinate = |regime: &str| quantity("y", regime);
    let pairs =
        [(coordinate("do(a=0)"), 1.0), (coordinate("do(a=1)"), 3.0), (coordinate("do(a=2)"), 5.0)];
    // Ordered: a mean aligns with its coordinate by position.
    ordered_laws(&pairs, |items| {
        source_identity(means_source(items, MeanEstimate::Frequentist, "lab", "snap"))
    });
    bit_exact(|mean| {
        source_identity(means_source(
            &[(coordinate("do(a=0)"), mean)],
            MeanEstimate::Frequentist,
            "lab",
            "snap",
        ))
    });
    injective(
        &ADVERSARIAL_TEXT,
        |p| *p,
        |(provider, snapshot)| {
            source_identity(means_source(&pairs, MeanEstimate::Frequentist, provider, snapshot))
        },
    );
    let identities: Vec<SemanticDigest> = [
        means_source(&pairs, MeanEstimate::Frequentist, "lab", "snap"),
        means_source(&pairs, MeanEstimate::Posterior, "lab", "snap"),
        ContractSource::Law(law(&[[1.0, 2.0], [3.0, 4.0]])),
        ContractSource::Law(law(&[[3.0, 4.0], [1.0, 2.0]])),
    ]
    .into_iter()
    .map(|source| source_identity(source).expect("the source compiles"))
    .collect();
    all_distinct(&identities);
}

#[test]
fn site_claim_decision_programs_functional_frames_its_parts() {
    let ctx = ExecutionContext::for_tests(7);
    let identity = |action: &str, functional: DecisionFunctional, rows: &[[f64; 2]]| {
        compile_decision_functional(contract(), action, functional, law(rows), &ctx)
            .expect("the functional compiles")
            .identity()
    };
    let rows = [[1.0, 2.0], [3.0, 4.0]];
    all_distinct(&[
        identity("A", DecisionFunctional::Expectation, &rows[..]),
        identity("B", DecisionFunctional::Expectation, &rows[..]),
        identity("A", DecisionFunctional::Variance, &rows[..]),
        identity("A", DecisionFunctional::Quantile { p: 0.5 }, &rows[..]),
        identity("A", DecisionFunctional::Quantile { p: 0.25 }, &rows[..]),
        identity("A", DecisionFunctional::Expectation, &[[1.0, 2.0], [3.0, 5.0]][..]),
    ]);
    bit_exact(|p| Some(identity("A", DecisionFunctional::Quantile { p }, &rows[..])));
}
