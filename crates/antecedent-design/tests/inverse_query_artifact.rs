//! F7: the durable inverse-query artifact (`inverse_functional_query_v1`).
//!
//! Ordered actions `a0, a1, a2` have outcome laws Bernoulli(1/4, 1/2, 3/4), realized as
//! four equiprobable aligned rows (see `tests/inverse_query.rs`):
//!
//! ```text
//! row   y(a0)  y(a1)  y(a2)
//!  0      1      1      1
//!  1      0      1      1
//!  2      0      0      1
//!  3      0      0      0
//! ```
//!
//! `E[Y]` is `1/4, 1/2, 3/4`; the left-inverse median is `0, 0, 1`. The artifact stores
//! the query, the embedded laws and means, and the per-action status table; the consumer
//! re-evaluates through the same engine and refuses any resealed mutation.

use antecedent_core::{QuantityRole, ScientificQuantity};
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, StructuralPolicy, UtilityExpr,
};
use antecedent_design::decision_eval::{DecisionEvalError, MeanSource};
use antecedent_design::decision_structural::{AtomEvidence, StructuralAtom};
use antecedent_design::inverse_query::{
    Comparison, ExistenceClaim, FeasibilityStatus, ForwardClaim, ForwardEvidence, GridScope,
    InverseConstraint, InverseQuery, InverseQueryError, SelectionOutcome, SelectionRule,
};
use antecedent_design::inverse_query_artifact::{
    ExistenceWire, InverseQueryArtifact, SelectionOutcomeWire, decode_parts, encode_parts,
};
use antecedent_io::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io::error::IoError;
use antecedent_io::quantity_wire::DistributionMeaningWire;

const REGIMES: [&str; 3] = ["do(a=0)", "do(a=1)", "do(a=2)"];
const IDS: [&str; 3] = ["a0", "a1", "a2"];
const A0: [f64; 4] = [1.0, 0.0, 0.0, 0.0];
const A1: [f64; 4] = [1.0, 1.0, 0.0, 0.0];
const A2: [f64; 4] = [1.0, 1.0, 1.0, 0.0];

fn quantity(variable: &str, regime: &str, functional: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable.into(),
        variable_name: variable.into(),
        role: QuantityRole::Outcome,
        units: "units".into(),
        population_id: "target".into(),
        regime_id: regime.into(),
        horizon: 0,
        functional_id: functional.into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

fn law(a1: &[f64], snapshot: &str) -> DistributionArtifact {
    let columns = [
        (quantity("y", REGIMES[0], "outcome"), A0.to_vec()),
        (quantity("y", REGIMES[1], "outcome"), a1.to_vec()),
        (quantity("y", REGIMES[2], "outcome"), A2.to_vec()),
    ];
    let quantities: Vec<ScientificQuantity> = columns.iter().map(|(q, _)| q.clone()).collect();
    let identity = DistributionIdentity::new(
        DistributionMeaningWire::InterventionalPredictive,
        &quantities,
        DrawAlignment::Joint,
        DistributionProvenance {
            source_id: "enumerated".into(),
            provider_id: "exact-law".into(),
            rng_id: "deterministic_exact".into(),
            snapshot_id: snapshot.into(),
            causal_contract_id: "checked-contract".into(),
        },
    )
    .unwrap();
    let rows = columns[0].1.len();
    let mut draws = Vec::with_capacity(rows * columns.len());
    for row in 0..rows {
        for (_, values) in &columns {
            draws.push(values[row]);
        }
    }
    DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity,
            axes: ["draw".into(), "quantity".into()],
            shape: [rows, columns.len()],
            weights: None,
            supported: None,
            calibration: DistributionCalibration::Exact,
            trust: DistributionTrust::Unverified,
            legacy_posterior: None,
            legacy_bindings: None,
        },
        draws,
    )
    .unwrap()
}

fn contract_with(functional: &str) -> DecisionContract {
    DecisionContract {
        actions: IDS
            .iter()
            .zip(REGIMES)
            .map(|(id, regime)| DecisionAction {
                id: (*id).into(),
                kind: ActionKind::Intervention,
                inputs: vec![quantity("y", regime, functional)],
                utility: UtilityExpr::Input(0),
            })
            .collect(),
        utility_units: "units".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
    }
}

fn query_on(
    contract: DecisionContract,
    constraints: Vec<InverseConstraint>,
    scope: GridScope,
) -> InverseQuery {
    InverseQuery {
        contract,
        grid_order: IDS.iter().map(|id| (*id).to_owned()).collect(),
        grid_scope: scope,
        constraints,
        selection: SelectionRule::FirstInGridOrder,
        tolerance: 1e-12,
        max_evaluations: None,
    }
}

fn mean_at_least(target: f64) -> InverseConstraint {
    InverseConstraint::TargetMean { target, comparison: Comparison::AtLeast }
}

fn median_at_least(target: f64) -> InverseConstraint {
    InverseConstraint::TargetQuantile { p: 0.5, target, comparison: Comparison::AtLeast }
}

fn point_law() -> ForwardEvidence {
    ForwardEvidence {
        point: Some(ForwardClaim::Law(Box::new(law(&A1, "enumeration-1")))),
        ..ForwardEvidence::default()
    }
}

fn sealed(constraints: Vec<InverseConstraint>, scope: GridScope) -> InverseQueryArtifact {
    InverseQueryArtifact::new(query_on(contract_with("outcome"), constraints, scope), point_law())
        .unwrap()
}

fn message(error: &IoError) -> String {
    error.to_string()
}

#[test]
fn f7_artifact_round_trips_the_status_table_and_replays_through_the_engine() {
    let artifact = sealed(vec![median_at_least(1.0)], GridScope::FiniteEnumeration);
    // Medians by hand: F_a0(0) = 3/4, F_a1(0) = 1/2, F_a2(0) = 1/4 against level 1/2, so the
    // left-inverse quantile is 0, 0, 1 and only a2 reaches 1.
    assert_eq!(artifact.result().feasible_actions, vec!["a2".to_owned()]);
    assert_eq!(artifact.result().selected.as_deref(), Some("a2"));
    assert_eq!(artifact.result().selection, SelectionOutcome::Selected);
    assert!(artifact.result().selection_certified);
    assert!(artifact.result().exhaustive_over_declared_set);

    let bytes = artifact.to_bytes("inverse-query-1").unwrap();
    let consumed = InverseQueryArtifact::from_bytes(&bytes, None).unwrap();
    assert_eq!(consumed.result(), artifact.result());
    assert_eq!(consumed.result_wire(), artifact.result_wire());
    assert_eq!(consumed.identity(), artifact.identity());
    // The consumer's own retained identity also matches.
    let retained = artifact.identity().clone();
    InverseQueryArtifact::from_bytes(&bytes, Some(&retained)).unwrap();
}

#[test]
fn f7_premises_and_data_digests_are_kept_apart() {
    let base = sealed(vec![mean_at_least(0.5)], GridScope::FiniteEnumeration);
    // Same premises, different numbers: only the data digest moves.
    let other_numbers = InverseQueryArtifact::new(
        query_on(contract_with("outcome"), vec![mean_at_least(0.5)], GridScope::FiniteEnumeration),
        ForwardEvidence {
            point: Some(ForwardClaim::Law(Box::new(law(&[1.0, 1.0, 1.0, 0.0], "enumeration-1")))),
            ..ForwardEvidence::default()
        },
    )
    .unwrap();
    assert_eq!(base.identity().premises_digest, other_numbers.identity().premises_digest);
    assert_ne!(base.identity().data_digest, other_numbers.identity().data_digest);
    assert_ne!(base.identity().digest, other_numbers.identity().digest);
    // Same numbers, different target: only the premises digest moves.
    let other_target = sealed(vec![mean_at_least(0.7)], GridScope::FiniteEnumeration);
    assert_ne!(base.identity().premises_digest, other_target.identity().premises_digest);
    assert_eq!(base.identity().data_digest, other_target.identity().data_digest);
}

#[test]
fn f7_artifact_keeps_the_feasibility_fields_distinct() {
    let scenario = |id: &str, a1: &[f64]| StructuralAtom {
        id: id.into(),
        probability: Some(0.5),
        evidence: AtomEvidence::Evaluated(Box::new(law(a1, id))),
    };
    // Scenario "high" makes a1's mean 3/4, scenario "low" makes it 1/4: conflicting answers
    // for a target mean of 1/2.
    let evidence = ForwardEvidence {
        scenarios: Some(vec![
            scenario("high", &[1.0, 1.0, 1.0, 0.0]),
            scenario("low", &[1.0, 0.0, 0.0, 0.0]),
        ]),
        ..point_law()
    };
    let artifact = InverseQueryArtifact::new(
        query_on(contract_with("outcome"), vec![mean_at_least(0.5)], GridScope::FiniteEnumeration),
        evidence,
    )
    .unwrap();
    let a1 = &artifact.result().actions[1];
    assert_eq!(a1.point, Some(FeasibilityStatus::Feasible), "the point field answers on its own");
    assert_eq!(a1.all_scenario, Some(FeasibilityStatus::StructurallyAmbiguous));
    assert_eq!(a1.interval_region, None, "no interval evidence, so no interval verdict");
    assert_eq!(a1.identified_set, None);
    let posterior = a1.posterior_probability.as_ref().unwrap();
    assert!((posterior.feasible_mass - 0.5).abs() < 1e-12);
    assert!((posterior.infeasible_mass - 0.5).abs() < 1e-12);

    let bytes = artifact.to_bytes("inverse-query-fields").unwrap();
    let consumed = InverseQueryArtifact::from_bytes(&bytes, None).unwrap();
    assert_eq!(consumed.result_wire(), artifact.result_wire());
    assert_eq!(consumed.result().actions[1].all_scenario, a1.all_scenario);
}

#[test]
fn f7_artifact_embeds_means_and_a_probability_on_them_refuses() {
    let means = MeanSource {
        coordinates: REGIMES.iter().map(|r| quantity("y", r, "mean")).collect(),
        means: vec![0.25, 0.5, 0.75],
        provider_id: "lab".into(),
        snapshot_id: "snap-9".into(),
        causal_contract_id: "checked-contract".into(),
        rng_id: "none:mean_grid".into(),
    };
    let evidence =
        ForwardEvidence { point: Some(ForwardClaim::Means(means)), ..ForwardEvidence::default() };
    let artifact = InverseQueryArtifact::new(
        query_on(contract_with("mean"), vec![mean_at_least(0.5)], GridScope::FiniteEnumeration),
        evidence.clone(),
    )
    .unwrap();
    assert_eq!(artifact.result().selected.as_deref(), Some("a1"));
    let bytes = artifact.to_bytes("inverse-query-means").unwrap();
    let consumed = InverseQueryArtifact::from_bytes(&bytes, None).unwrap();
    assert_eq!(consumed.result_wire(), artifact.result_wire());

    // A mean never yields an outcome quantile: the shared engine refuses, so no artifact.
    let error = InverseQueryArtifact::new(
        query_on(contract_with("mean"), vec![median_at_least(1.0)], GridScope::FiniteEnumeration),
        evidence,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        InverseQueryError::Engine(DecisionEvalError::MeanSourceInsufficient { .. })
    ));
}

#[test]
fn f7_consumer_refuses_resealed_mutations() {
    let artifact = sealed(vec![mean_at_least(0.5)], GridScope::FiniteEnumeration);
    let bytes = artifact.to_bytes("inverse-query-tamper").unwrap();
    let (meta, contract, data) = decode_parts(&bytes).unwrap();

    // A changed selection in the stored table does not replay.
    let mut forged = meta.clone();
    forged.result.selected = Some("a0".into());
    let tampered = encode_parts(&forged, &contract, &data, "tampered").unwrap();
    let error = InverseQueryArtifact::from_bytes(&tampered, None).unwrap_err();
    assert!(message(&error).contains("does not replay"), "{error}");

    // A forged selection_certified or exhaustive flag does not replay either.
    let mut forged = meta.clone();
    forged.result.exhaustive_over_declared_set = !forged.result.exhaustive_over_declared_set;
    let tampered = encode_parts(&forged, &contract, &data, "tampered").unwrap();
    assert!(InverseQueryArtifact::from_bytes(&tampered, None).is_err());

    // A changed target in the stored query is caught by the recomputed premises digest.
    let mut forged = meta.clone();
    forged.query.constraints =
        vec![antecedent_design::inverse_query_artifact::ConstraintWire::TargetMean {
            target: 0.9,
            comparison: antecedent_design::inverse_query_artifact::ComparisonWire::AtLeast,
        }];
    let tampered = encode_parts(&forged, &contract, &data, "tampered").unwrap();
    let error = InverseQueryArtifact::from_bytes(&tampered, None).unwrap_err();
    assert!(message(&error).contains("premises digest"), "{error}");

    // A changed draw is caught by the recomputed data digest.
    let mut numbers = data.clone();
    numbers[0] ^= 1;
    let tampered = encode_parts(&meta, &contract, &numbers, "tampered").unwrap();
    let error = InverseQueryArtifact::from_bytes(&tampered, None).unwrap_err();
    assert!(message(&error).contains("data digest"), "{error}");

    // A self-consistent resealed artifact for a different target is internally valid, but
    // not under the identity the consumer retained.
    let other = sealed(vec![mean_at_least(0.7)], GridScope::FiniteEnumeration);
    let other_bytes = other.to_bytes("other").unwrap();
    InverseQueryArtifact::from_bytes(&other_bytes, None).unwrap();
    let error =
        InverseQueryArtifact::from_bytes(&other_bytes, Some(artifact.identity())).unwrap_err();
    assert!(message(&error).contains("retained identity"), "{error}");
}

#[test]
fn f7_other_versions_and_layouts_are_refused() {
    let artifact = sealed(vec![mean_at_least(0.5)], GridScope::FiniteEnumeration);
    let bytes = artifact.to_bytes("inverse-query-version").unwrap();
    let (meta, contract, data) = decode_parts(&bytes).unwrap();
    let mut future = meta;
    future.version = 2;
    let reencoded = encode_parts(&future, &contract, &data, "future").unwrap();
    assert!(matches!(
        InverseQueryArtifact::from_bytes(&reencoded, None),
        Err(IoError::UnsupportedVersion { version: 2 })
    ));
    assert!(InverseQueryArtifact::from_bytes(&bytes[..bytes.len() / 2], None).is_err());
}

#[test]
fn f7_continuous_sample_artifact_never_claims_global_feasibility() {
    // No sampled point reaches a mean of 0.9; that does not show the continuum has none.
    let none_found = sealed(vec![mean_at_least(0.9)], GridScope::ContinuousSample);
    assert_eq!(none_found.result().existence, ExistenceClaim::Undetermined);
    assert!(!none_found.result().exhaustive_over_declared_set);
    // One feasible sampled point is a witness, not a global claim.
    let found = sealed(vec![mean_at_least(0.5)], GridScope::ContinuousSample);
    assert_eq!(found.result().existence, ExistenceClaim::FoundFeasibleAction);
    assert!(!found.result().exhaustive_over_declared_set);
    let bytes = found.to_bytes("continuous").unwrap();
    let consumed = InverseQueryArtifact::from_bytes(&bytes, None).unwrap();
    assert!(!consumed.result().exhaustive_over_declared_set);

    // A resealed artifact that upgrades the sample to a global claim is refused before replay.
    let bytes = none_found.to_bytes("continuous-none").unwrap();
    let (meta, contract, data) = decode_parts(&bytes).unwrap();
    let mut forged = meta.clone();
    forged.result.exhaustive_over_declared_set = true;
    forged.result.existence = ExistenceWire::NoFeasibleActionInDeclaredSet;
    forged.result.selection = SelectionOutcomeWire::NoFeasibleAction;
    let tampered = encode_parts(&forged, &contract, &data, "forged").unwrap();
    let error = InverseQueryArtifact::from_bytes(&tampered, None).unwrap_err();
    assert!(message(&error).contains("global_feasibility_claim"), "{error}");
    assert!(message(&error).contains("inverse_functional_unsupported"), "{error}");

    // The same table on a finite enumeration is a legitimate claim and replays.
    let finite = sealed(vec![mean_at_least(0.9)], GridScope::FiniteEnumeration);
    assert_eq!(finite.result().existence, ExistenceClaim::NoFeasibleActionInDeclaredSet);
    assert!(finite.result().exhaustive_over_declared_set);
}
