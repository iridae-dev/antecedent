//! 2.2B X10: prepared lifecycle, export and independent consumption of exact
//! binary observation recovery, with the artifact mutation set.
#![allow(clippy::float_cmp, reason = "the mutation needs two cells with different masses")]

#[path = "../../antecedent-estimate/tests/support/recovery_scm.rs"]
mod recovery_scm;

use std::sync::Arc;

use antecedent::{
    ObservationRecoveryResult, PreparedObservationRecovery, StudyBuilder,
    consume_observation_recovery_artifact,
};
use antecedent_core::{ExecutionContext, Value};
use antecedent_expr::{Assignment, ExactDiscreteLaw, ExactEvaluationLimits, LawTolerance};
use antecedent_graph::{Admg, DenseNodeId, NodeRef};
use antecedent_identify::{RecoveredEffectQuery, RecoveryDetail, RecoveryLimits};
use antecedent_io::recovery_artifact::{
    RecoveryArtifactError, RecoveryArtifactWire, RecoveryConsumeLimits,
};
use recovery_scm::{MModel, POPULATION, SNAPSHOT, v};

const EDGES: [(u32, u32); 3] = [(2, 0), (2, 1), (0, 1)];

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

fn model() -> MModel {
    // Treatment X0, outcome X1, confounder O; R0 <- O, R1 <- X0.
    MModel::new(2, 1, &EDGES, &[vec![2], vec![0]], 17).unwrap()
}

fn effect(model: &MModel) -> RecoveredEffectQuery {
    let mut graph = Admg::empty();
    for n in 0..model.k + model.m {
        graph.add_node(NodeRef::Static(v(n))).unwrap();
    }
    for (a, b) in EDGES {
        graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    RecoveredEffectQuery { graph, outcomes: Arc::from([v(1)]), treatments: Arc::from([v(0)]) }
}

fn requests() -> Vec<Assignment> {
    [0.0, 1.0].iter().map(|x| Assignment::from_pairs([(v(0), Value::f64(*x))])).collect()
}

fn prepare(model: &MModel) -> PreparedObservationRecovery {
    StudyBuilder::observation_recovery(
        model.graph.clone(),
        &model.query(),
        model.catalog(),
        Some(effect(model)),
        RecoveryLimits::default(),
        model.observed_law(),
        requests(),
        ExactEvaluationLimits::default(),
        &ctx(),
    )
    .unwrap()
}

fn names(model: &MModel) -> Vec<String> {
    (0..model.nodes()).map(|n| format!("n{n}")).collect()
}

fn exported(model: &MModel) -> (Vec<u8>, ObservationRecoveryResult) {
    let prepared = prepare(model);
    let result = prepared.estimate(&ctx()).unwrap();
    (result.export_named(&prepared, &names(model)).unwrap(), result)
}

fn p_one(point: &antecedent_expr::ExactDistribution) -> f64 {
    point.probabilities[point.atoms.iter().position(|a| a[0].as_f64() == Some(1.0)).unwrap()]
}

#[test]
fn estimate_runs_the_retained_derivation_after_builder_disposal() {
    let model = model();
    let builder = prepare(&model);
    let prepared = builder.clone();
    drop(builder);
    let result = prepared.estimate(&ctx()).unwrap();
    assert!(
        result
            .recovered()
            .law()
            .probabilities()
            .iter()
            .zip(model.truth())
            .all(|(a, b)| (a - b).abs() < 1e-12)
    );
    for (point, level) in result.effects().iter().zip([0usize, 1]) {
        assert!((p_one(point) - model.interventional(1, 0, level)).abs() < 1e-12);
    }
    // The retained derivation is the plan: recovery formula plus downstream effect.
    let plan = prepared.derivation();
    assert_eq!(plan.record().factors.len(), 2);
    assert!(plan.effect().is_some());
    assert_eq!(prepared.requests().len(), 2);
}

#[test]
fn refresh_keeps_the_derivation_and_refuses_another_snapshot() {
    let model = model();
    let prepared = prepare(&model);
    let identity = prepared.derivation().identity();
    // A law of the same snapshot (another SCM of the same graph): the derivation is
    // kept, the point follows the new law.
    let other = MModel::new(2, 1, &EDGES, &[vec![2], vec![0]], 18).unwrap();
    let refreshed = prepared.refresh(other.observed_law(), &ctx()).unwrap();
    assert_eq!(refreshed.derivation().identity(), identity);
    let result = refreshed.estimate(&ctx()).unwrap();
    assert!(
        result
            .recovered()
            .law()
            .probabilities()
            .iter()
            .zip(other.truth())
            .all(|(a, b)| (a - b).abs() < 1e-12)
    );
    // Another snapshot needs a new preparation.
    let law = other.observed_law();
    let moved = ExactDiscreteLaw::try_new(
        POPULATION,
        law.regime(),
        Vec::new(),
        law.axes().to_vec(),
        law.probabilities().to_vec(),
        "snap-moved",
        LawTolerance::default(),
    )
    .unwrap();
    let error = prepared.refresh(moved, &ctx()).unwrap_err();
    assert_eq!(error.reason_code(), Some("invalid_argument"));
    assert!(error.to_string().contains("recovery.invalid_observed_law"), "{error}");
    let _ = SNAPSHOT;
}

#[test]
fn a_nonrecoverable_decision_refuses_preparation_with_its_witness_code() {
    let censored = MModel::new(2, 1, &EDGES, &[vec![2], vec![1]], 17).unwrap();
    let error = StudyBuilder::observation_recovery(
        censored.graph.clone(),
        &censored.query(),
        censored.catalog(),
        None,
        RecoveryLimits::default(),
        censored.observed_law(),
        Vec::new(),
        ExactEvaluationLimits::default(),
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(error.reason_code(), Some("transport_proven_non_transportable"));
    assert!(error.to_string().contains("recovery.nonrecoverable_witness"));
}

#[test]
fn exported_point_is_recomputed_by_an_independent_consumer() {
    let model = model();
    let (bytes, live) = exported(&model);
    let consumed =
        consume_observation_recovery_artifact(&bytes, RecoveryConsumeLimits::default(), &ctx())
            .unwrap();
    let bits = |p: &[f64]| p.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(consumed.recovered().law().probabilities()),
        bits(live.recovered().law().probabilities())
    );
    for (a, b) in consumed.effects().iter().zip(live.effects()) {
        assert_eq!(bits(&a.probabilities), bits(&b.probabilities));
    }
    assert_eq!(consumed.recovered().descriptor(), live.recovered().descriptor());
}

#[test]
fn the_io_consumer_replays_under_the_stored_limits() {
    let model = model();
    let (bytes, _) = exported(&model);
    let consumed =
        RecoveryArtifactWire::consume_with_limits(&bytes, RecoveryConsumeLimits::default(), &ctx())
            .unwrap();
    assert_eq!(consumed.wire.derivation, *consumed.derivation.record());
    consumed.wire.check_variable_names(&names(&model)).unwrap();
    assert_eq!(
        consumed.wire.check_variable_names(&[]).unwrap_err(),
        RecoveryArtifactError::NamesMismatch
    );
    // The producer refuses to export what a default consumer would refuse.
    let too_many = StudyBuilder::observation_recovery(
        model.graph.clone(),
        &model.query(),
        model.catalog(),
        Some(effect(&model)),
        RecoveryLimits::default(),
        model.observed_law(),
        (0..65).map(|_| requests()[0].clone()).collect(),
        ExactEvaluationLimits::default(),
        &ctx(),
    )
    .unwrap();
    let result = too_many.estimate(&ctx()).unwrap();
    let error = result.export(&too_many).unwrap_err();
    assert_eq!(error.reason_code(), Some("route_not_supported"));
    // A consumer whose maxima are below the producer's stored limits refuses first.
    let mut tight = RecoveryConsumeLimits::default();
    tight.decision.search.operations = 10;
    let Err(error) = RecoveryArtifactWire::consume_with_limits(&bytes, tight, &ctx()) else {
        panic!("limits above the consumer's maxima must refuse");
    };
    assert_eq!(error.reason_code(), Some("route_not_supported"));
    assert!(error.to_string().contains("recovery.bounds_exceeded"), "{error}");
}

fn reseal(wire: &mut RecoveryArtifactWire) {
    wire.premises_digest = wire.expected_premises_digest().unwrap();
    wire.data_digest = wire.expected_data_digest().unwrap();
}

fn consume(wire: &RecoveryArtifactWire) -> Result<(), RecoveryArtifactError> {
    RecoveryArtifactWire::consume_typed(
        &wire.export().unwrap(),
        RecoveryConsumeLimits::default(),
        &ctx(),
    )
    .map(|_| ())
}

#[test]
fn a_mutated_artifact_fails_independent_consumption() {
    let model = model();
    let (bytes, _) = exported(&model);
    let original = RecoveryArtifactWire::decode(&bytes).unwrap();
    consume(&original).unwrap();
    let proof = |error: RecoveryArtifactError| matches!(error, RecoveryArtifactError::ProofMismatch(e) if e.detail == RecoveryDetail::InvalidDerivation);

    // Premises edited without resealing.
    let mut premises = original.clone();
    premises.query.population = "elsewhere".into();
    assert_eq!(consume(&premises).unwrap_err(), RecoveryArtifactError::PremisesMismatch);
    // Data digest edited.
    let mut data = original.clone();
    data.data_digest = "00".into();
    assert_eq!(consume(&data).unwrap_err(), RecoveryArtifactError::DataIdentityMismatch);
    // A regime the derivation never cites, added or edited without resealing: the
    // whole catalog is bound into the data digest, not only the observed regime.
    let mut unrelated = original.clone();
    let mut extra = unrelated.catalog.regimes[0].clone();
    extra.id = 8;
    extra.label = None;
    unrelated.catalog.regimes.push(extra);
    assert_eq!(consume(&unrelated).unwrap_err(), RecoveryArtifactError::DataIdentityMismatch);
    let mut relabelled = original.clone();
    relabelled.catalog.regimes[0].label = Some("renamed".into());
    assert_eq!(consume(&relabelled).unwrap_err(), RecoveryArtifactError::DataIdentityMismatch);
    // The observed table swapped without resealing: data identity, not replay.
    let mut swapped = original.clone();
    let cells = &swapped.observed_law.probabilities;
    let (i, j) = (0..cells.len())
        .flat_map(|i| (0..cells.len()).map(move |j| (i, j)))
        .find(|(i, j)| cells[*i] > 0.0 && cells[*j] > 0.0 && cells[*i] != cells[*j])
        .unwrap();
    swapped.observed_law.probabilities.swap(i, j);
    assert_eq!(consume(&swapped).unwrap_err(), RecoveryArtifactError::DataIdentityMismatch);
    // Result: a recovered cell and an effect cell one ulp off.
    let mut recovered = original.clone();
    recovered.recovered.probabilities[0] =
        f64::from_bits(recovered.recovered.probabilities[0].to_bits() + 1);
    assert_eq!(consume(&recovered).unwrap_err(), RecoveryArtifactError::RecoveredMismatch);
    let mut effect = original.clone();
    effect.effects[1].probabilities[0] =
        f64::from_bits(effect.effects[1].probabilities[0].to_bits() + 1);
    assert_eq!(consume(&effect).unwrap_err(), RecoveryArtifactError::EffectMismatch);
    // Re-sealed semantic edits keep both digests valid and fail replay for the right reason.
    let mut margin = original.clone();
    margin.derivation.margins[0].identity.push('x');
    reseal(&mut margin);
    assert!(proof(consume(&margin).unwrap_err()), "margin identity");
    let mut premise = original.clone();
    premise.derivation.premises.pop();
    reseal(&mut premise);
    assert!(proof(consume(&premise).unwrap_err()), "proof premise");
    let mut edge = original.clone();
    edge.graph.directed.push((2, 4)); // O -> R1: a different m-graph.
    reseal(&mut edge);
    assert!(proof(consume(&edge).unwrap_err()), "graph");
    let mut factor = original.clone();
    factor.derivation.factors[1].conditioning = vec![2];
    reseal(&mut factor);
    assert!(proof(consume(&factor).unwrap_err()), "factor");
    let mut law = original.clone();
    let other = MModel::new(2, 1, &EDGES, &[vec![2], vec![0]], 99).unwrap().observed_law();
    law.observed_law.probabilities = other.probabilities().to_vec();
    reseal(&mut law);
    assert_eq!(consume(&law).unwrap_err(), RecoveryArtifactError::RecoveredMismatch);
    let mut request = original.clone();
    request.requests.swap(0, 1);
    reseal(&mut request);
    assert_eq!(consume(&request).unwrap_err(), RecoveryArtifactError::EffectMismatch);
    let mut snapshot = original.clone();
    let mut catalog = snapshot.catalog.to_catalog().unwrap();
    Arc::make_mut(&mut catalog.bindings)[0].snapshot_identity = Arc::from("snap-renamed");
    snapshot.catalog =
        antecedent_io::transport_catalog_wire::EvidenceCatalogWire::from_catalog(&catalog);
    snapshot.observed_law.snapshot = "snap-renamed".into();
    reseal(&mut snapshot);
    assert!(
        proof(consume(&snapshot).unwrap_err()),
        "catalog snapshot changes every margin identity"
    );
    // Limits above the consumer's maxima, a claim other than point_only.
    let mut limits = original.clone();
    limits.derivation.receipt.operations_limit = 60_000;
    reseal(&mut limits);
    assert!(matches!(consume(&limits).unwrap_err(), RecoveryArtifactError::LimitsExceeded(_)));
    let mut claim = original.clone();
    claim.uncertainty = "calibrated".into();
    assert!(matches!(consume(&claim).unwrap_err(), RecoveryArtifactError::UnsupportedSemantics(_)));
    // Every kind maps to a registered code and an X10 detail.
    assert_eq!(
        RecoveryArtifactError::PremisesMismatch.refusal(),
        ("transport_not_certified", "recovery.invalid_derivation")
    );
    assert_eq!(
        RecoveryArtifactError::LimitsExceeded("x").refusal(),
        ("route_not_supported", "recovery.bounds_exceeded")
    );
}

#[test]
fn wire_structs_deny_unknown_fields() {
    let model = model();
    let (bytes, _) = exported(&model);
    let mut value: ciborium::Value = ciborium::from_reader(bytes.as_slice()).unwrap();
    if let ciborium::Value::Map(entries) = &mut value {
        entries.push((ciborium::Value::Text("smuggled".into()), ciborium::Value::Bool(true)));
    }
    let mut tampered = Vec::new();
    ciborium::into_writer(&value, &mut tampered).unwrap();
    assert!(RecoveryArtifactWire::decode(&tampered).is_err());
    let json = serde_json::json!({"population": "p", "observed_regime": 0, "partially_observed": [], "fully_observed": [], "extra": 1});
    assert!(
        serde_json::from_value::<antecedent_io::recovery_artifact::RecoveryQueryWire>(json)
            .is_err()
    );
    let record =
        serde_json::to_value(RecoveryArtifactWire::decode(&bytes).unwrap().derivation).unwrap();
    let mut extra = record;
    extra["smuggled"] = serde_json::json!(1);
    assert!(
        serde_json::from_value::<antecedent_identify::RecoveryDerivationRecord>(extra).is_err()
    );
}
