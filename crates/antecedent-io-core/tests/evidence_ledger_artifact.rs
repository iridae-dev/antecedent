//! Evidence ledger artifact: a fresh consumer replays the ledger, recomputes
//! the syntheses and refuses semantic mutations.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::evidence_ledger::{
    CombinationRequest, EvidenceLedger, ObservationId, PoolingDependence, ResultId, Revalidation,
    SourceDeclaration, SourceId, SourceKind, SynthesisOperation,
};
use antecedent_io_core::evidence_ledger_artifact::{
    EvidenceLedgerArtifact, EvidenceLedgerArtifactError, EvidenceLedgerMeta, SynthesisMethod,
    SynthesisRecord, covariance_digest, covariance_has_cross_terms,
};
use antecedent_stats::synthesis::{MeanPrior, TauPrior};

const STUDY_ESTIMATES: [f64; 3] = [0.0, 2.0, 6.0];
const STUDY_VARIANCES: [f64; 3] = [1.0, 2.0, 4.0];

struct Fixture {
    ledger: EvidenceLedger,
    records: Vec<SynthesisRecord>,
    s1: SourceId,
    s2: SourceId,
    s3: SourceId,
    pooled: ResultId,
    covariance: ResultId,
    envelope: ResultId,
    borrowed: ResultId,
}

fn rows(range: std::ops::Range<u32>) -> Vec<ObservationId> {
    range.map(|i| ObservationId::from_key("cohort", &i.to_le_bytes())).collect()
}

fn data(label: &str, range: std::ops::Range<u32>) -> SourceDeclaration {
    SourceDeclaration::new(label, SourceKind::ObservationSet).with_observations(rows(range))
}

fn pooling(label: &str, ids: &[SourceId], dependence: PoolingDependence) -> CombinationRequest {
    CombinationRequest {
        label: label.to_owned(),
        operation: SynthesisOperation::StatisticalPooling { sources: ids.to_vec(), dependence },
        replaces: None,
    }
}

fn fixture() -> Fixture {
    let mut ledger = EvidenceLedger::new();
    let s1 = ledger.add_source(data("study_1", 0..40)).expect("s1");
    let s2 = ledger.add_source(data("study_2", 40..90)).expect("s2");
    let s3 = ledger.add_source(data("study_3", 90..120)).expect("s3");
    // a fourth study re-using 10 of study 3's observations
    let s4 = ledger.add_source(data("study_4", 110..150)).expect("s4");
    let vendor = ledger
        .add_source(SourceDeclaration::new("vendor_claim", SourceKind::ExternalClaim))
        .expect("vendor");

    let pooled = ledger
        .combine(pooling("independent_pool", &[s1, s2, s3], PoolingDependence::Independent))
        .expect("pool");

    // y = (1, 3), standard errors (1, 2), correlation 1/2: S = [[1, 1], [1, 4]]
    let covariance_matrix = [1.0, 1.0, 1.0, 4.0];
    let covariance = ledger
        .combine(pooling(
            "covariance_pool",
            &[s3, s4],
            PoolingDependence::DeclaredCovariance {
                covariance: covariance_digest(2, &covariance_matrix),
                has_cross_covariance: covariance_has_cross_terms(2, &covariance_matrix),
            },
        ))
        .expect("covariance pool");

    let envelope = ledger
        .combine(pooling(
            "range_pool",
            &[s3, vendor],
            PoolingDependence::AssumptionRange { rho_lo: 0.0, rho_hi: 0.9 },
        ))
        .expect("range pool");

    let borrowed = ledger
        .combine(CombinationRequest {
            label: "hierarchical".to_owned(),
            operation: SynthesisOperation::BayesianBorrowing {
                priors: Vec::new(),
                likelihood: vec![s1, s2, s3],
            },
            replaces: None,
        })
        .expect("borrow");

    let records = vec![
        SynthesisRecord {
            result: pooled,
            sources: vec![s1, s2, s3],
            method: SynthesisMethod::RandomEffectsDerSimonianLaird {
                estimates: STUDY_ESTIMATES.to_vec(),
                variances: STUDY_VARIANCES.to_vec(),
            },
        },
        SynthesisRecord {
            result: covariance,
            sources: vec![s3, s4],
            method: SynthesisMethod::FixedEffectGls {
                estimates: vec![1.0, 3.0],
                covariance: covariance_matrix.to_vec(),
            },
        },
        SynthesisRecord {
            result: envelope,
            sources: vec![s3, vendor],
            method: SynthesisMethod::DependenceEnvelope {
                estimates: vec![1.0, 3.0],
                standard_errors: vec![1.0, 2.0],
                rho_lo: 0.0,
                rho_hi: 0.9,
            },
        },
        SynthesisRecord {
            result: borrowed,
            sources: vec![s1, s2, s3],
            method: SynthesisMethod::HierarchicalBorrowing {
                estimates: STUDY_ESTIMATES.to_vec(),
                variances: STUDY_VARIANCES.to_vec(),
                mean_prior: MeanPrior::Flat,
                tau_prior: TauPrior { support: vec![1.0], probabilities: vec![1.0] },
            },
        },
    ];
    Fixture { ledger, records, s1, s2, s3, pooled, covariance, envelope, borrowed }
}

fn close(actual: f64, expected: f64, what: &str) {
    assert!((actual - expected).abs() <= 1e-9, "{what}: expected {expected}, got {actual}");
}

fn sealed() -> (Fixture, EvidenceLedgerMeta) {
    let f = fixture();
    let (_, meta) = EvidenceLedgerArtifact::seal(&f.ledger, &f.records).expect("seal");
    (f, meta)
}

fn reseal(mut meta: EvidenceLedgerMeta) -> EvidenceLedgerMeta {
    EvidenceLedgerArtifact::reseal_identity(&mut meta);
    meta
}

// -- round trip --------------------------------------------------------------------------------

/// Oracle values: the stats-layer hand derivations (`DerSimonian`-Laird tau^2 = 38/7,
/// GLS weights (1, 0), envelope [-1/7, 7/5] with variance maximum 1 at 1/2, hierarchical
/// borrowing at tau = 1 giving 56/31 and 30/31).
#[test]
fn a_fresh_consumer_replays_the_ledger_and_recomputes_the_syntheses() {
    let f = fixture();
    let bytes = EvidenceLedgerArtifact::to_bytes(&f.ledger, &f.records, "evidence-ledger-1")
        .expect("export");
    let consumed = EvidenceLedgerArtifact::from_bytes(&bytes, None).expect("consume");

    // the replayed ledger equals the producer's, operation by operation
    assert_eq!(consumed.ledger().head(), f.ledger.head());
    assert_eq!(consumed.ledger().state_digest(), f.ledger.state_digest());
    assert_eq!(consumed.ledger().operations().len(), f.ledger.operations().len());
    let mass = &consumed.ledger().result(f.covariance).expect("result").mass;
    assert_eq!(
        (mass.declared_slots, mass.distinct_observations, mass.repeated_slots),
        (70, 60, 10)
    );

    let by_result = |id: ResultId| {
        consumed.syntheses().iter().find(|s| s.result == id).expect("synthesis").clone()
    };
    let dl = by_result(f.pooled);
    assert_eq!(dl.method, "random_effects_dersimonian_laird");
    close(dl.values[0], 3330.0 / 1457.0, "DL estimate");
    close(dl.values[1], 25740.0 / 10199.0, "DL variance");
    close(dl.values[2], 38.0 / 7.0, "DL tau^2");
    close(dl.values[3], 52.0 / 7.0, "Q");

    let gls = by_result(f.covariance);
    assert_eq!(gls.method, "fixed_effect_gls");
    close(gls.values[0], 1.0, "GLS estimate");
    close(gls.values[1], 1.0, "GLS variance");
    close(gls.values[3], 1.0, "weight 1");
    close(gls.values[4], 0.0, "weight 2");

    let env = by_result(f.envelope);
    assert_eq!(env.method, "dependence_envelope");
    let expected = [-1.0 / 7.0, 0.9, 1.4, 0.0, 19.0 / 35.0, 0.9, 1.0, 0.5];
    for (value, want) in env.values.iter().zip(expected) {
        close(*value, want, "envelope");
    }
    // the ledger labels the result an assumption range, and refuses an interval from it
    let record = consumed.ledger().result(f.envelope).expect("result");
    assert_eq!(record.claim_class.as_str(), "assumption_range");
    assert!(consumed.ledger().request_interval(f.envelope).is_err());

    let hb = by_result(f.borrowed);
    close(hb.values[0], 56.0 / 31.0, "posterior mean");
    close(hb.values[1], 30.0 / 31.0, "posterior variance");

    // same bytes, same identity; and a consumer that retained the identity accepts it
    let identity = consumed.identity().to_owned();
    let again = EvidenceLedgerArtifact::from_bytes(&bytes, Some(&identity)).expect("retained");
    assert_eq!(again.identity(), identity);
    let (sealed_again, _) = EvidenceLedgerArtifact::seal(&f.ledger, &f.records).expect("seal");
    assert_eq!(sealed_again.identity(), identity);
}

#[test]
fn withdrawal_and_supersession_survive_export_and_revalidate_dependents() {
    let mut f = fixture();
    // an external model built from the pooled result, then withdrawing a study
    let declaration = f
        .ledger
        .declaration_from_result(f.pooled, "downstream_model", SourceKind::FittedModel)
        .expect("declaration");
    let model = f.ledger.add_source(declaration).expect("model");
    let downstream = f
        .ledger
        .combine(pooling(
            "model_and_s2",
            &[model, f.s2],
            PoolingDependence::AssumptionRange { rho_lo: 0.0, rho_hi: 0.5 },
        ))
        .expect("downstream");
    f.ledger.withdraw_source(f.s1, "study retracted").expect("withdraw");

    // pooling three studies and losing one still pools two: recompute
    assert_eq!(f.ledger.standing(f.pooled), Some(Revalidation::Recompute));
    assert_eq!(f.ledger.standing(f.borrowed), Some(Revalidation::Recompute));
    assert_eq!(f.ledger.standing(f.covariance), Some(Revalidation::StillValid));
    // the model was built from the pooled studies, so it is lost through lineage and the
    // two-member pool it sat in has one member left: withdrawn
    assert_eq!(f.ledger.standing(downstream), Some(Revalidation::Withdrawn));

    // syntheses of results the withdrawal retired are dropped by the producer
    let records: Vec<SynthesisRecord> =
        f.records.iter().filter(|r| r.result == f.covariance).cloned().collect();
    let bytes =
        EvidenceLedgerArtifact::to_bytes(&f.ledger, &records, "evidence-ledger-2").expect("export");
    let consumed = EvidenceLedgerArtifact::from_bytes(&bytes, None).expect("consume");
    assert_eq!(consumed.ledger().standing(f.pooled), Some(Revalidation::Recompute));
    assert_eq!(consumed.ledger().standing(downstream), Some(Revalidation::Withdrawn));
    assert_eq!(consumed.ledger().standing(f.covariance), Some(Revalidation::StillValid));
    // the consumer's ledger also refuses to build on the withdrawn study
    let mut replayed = consumed.ledger().clone();
    let error = replayed
        .combine(pooling("again", &[f.s1, f.s3], PoolingDependence::Independent))
        .expect_err("withdrawn study");
    assert_eq!(error.code, "evidence_source_withdrawn");
}

// -- semantic mutations ----------------------------------------------------------------------------

#[test]
fn an_unresealed_edit_changes_the_identity_or_fails_replay() {
    let (_, mut meta) = sealed();
    // one unit in the last place is inside the replay tolerance, so only the identity
    // (which covers the stored bits) catches it
    meta.syntheses[0].values[0] ^= 1;
    assert!(matches!(
        EvidenceLedgerArtifact::from_meta(&meta, None),
        Err(EvidenceLedgerArtifactError::IdentityMismatch { field: "identity" })
    ));
    let (_, mut meta) = sealed();
    meta.identity = "0".repeat(64);
    assert!(matches!(
        EvidenceLedgerArtifact::from_meta(&meta, None),
        Err(EvidenceLedgerArtifactError::IdentityMismatch { field: "identity" })
    ));
}

#[test]
fn a_resealed_synthesis_value_edit_is_refused() {
    let (_, mut meta) = sealed();
    // move the pooled estimate by 1e-6: far beyond the replay tolerance
    let value = f64::from_bits(meta.syntheses[0].values[0]) + 1e-6;
    meta.syntheses[0].values[0] = value.to_bits();
    let error = EvidenceLedgerArtifact::from_meta(&reseal(meta), None).expect_err("refused");
    assert_eq!(error, EvidenceLedgerArtifactError::SynthesisMismatch("values"));
    let (code, detail, _) = error.refusal().expect("registered refusal");
    assert_eq!(code, "invalid_argument");
    assert_eq!(detail, "evidence_ledger.synthesis_mismatch");
}

#[test]
fn a_resealed_classification_edit_after_a_withdrawal_is_refused() {
    let mut f = fixture();
    f.ledger.withdraw_source(f.s1, "retracted").expect("withdraw");
    let (_, mut meta) = EvidenceLedgerArtifact::seal(&f.ledger, &[]).expect("seal");
    let op = meta.operations.last_mut().expect("withdrawal");
    assert_eq!(op.kind, "withdraw");
    let entry = op
        .effect
        .revalidation
        .iter_mut()
        .find(|r| r.class == "recompute")
        .expect("a dependent needs recomputation");
    entry.class = "still_valid".to_owned();
    let error = EvidenceLedgerArtifact::from_meta(&reseal(meta), None).expect_err("refused");
    assert_eq!(error, EvidenceLedgerArtifactError::ReplayMismatch("operation effect"));
}

#[test]
fn a_resealed_declaration_edit_is_refused_on_replay() {
    let (_, mut meta) = sealed();
    // make study 2 share an observation with study 1: its identity changes, so every
    // later reference to the stored identity no longer resolves
    let first_row =
        meta.operations[0].source.as_ref().and_then(|s| s.observations.as_ref()).expect("rows")[0]
            .clone();
    let rows =
        meta.operations[1].source.as_mut().and_then(|s| s.observations.as_mut()).expect("rows");
    rows[0] = first_row;
    let error = EvidenceLedgerArtifact::from_meta(&reseal(meta), None).expect_err("refused");
    // A semantic refusal found by replay, not corruption or a malformed value: either the
    // ledger rejects the stored operation or its recomputed identity differs from the stored one.
    assert!(
        matches!(
            error,
            EvidenceLedgerArtifactError::Refused { .. }
                | EvidenceLedgerArtifactError::ReplayMismatch("operation identity")
        ),
        "{error}"
    );
    assert!(error.refusal().is_some(), "{error}");
}

#[test]
fn a_resealed_operation_reorder_or_reason_edit_is_refused() {
    let (_, mut meta) = sealed();
    meta.operations.swap(0, 1);
    let error = EvidenceLedgerArtifact::from_meta(&reseal(meta), None).expect_err("refused");
    assert_eq!(error, EvidenceLedgerArtifactError::ReplayMismatch("operation identity"));

    let mut f = fixture();
    f.ledger.withdraw_source(f.s1, "retracted").expect("withdraw");
    let (_, mut meta) = EvidenceLedgerArtifact::seal(&f.ledger, &[]).expect("seal");
    meta.operations.last_mut().expect("withdrawal").reason = Some("a different reason".to_owned());
    let error = EvidenceLedgerArtifact::from_meta(&reseal(meta), None).expect_err("refused");
    assert_eq!(error, EvidenceLedgerArtifactError::ReplayMismatch("operation identity"));
}

#[test]
fn a_resealed_covariance_or_range_edit_no_longer_binds_to_the_ledger_result() {
    let (_, mut meta) = sealed();
    // a diagonal covariance with the same studies: the ledger bound the cross covariance
    let gls = meta.syntheses.iter_mut().find(|s| s.method == "fixed_effect_gls").expect("gls");
    gls.covariance = [1.0_f64, 0.0, 0.0, 4.0].iter().map(|v| v.to_bits()).collect();
    let error = EvidenceLedgerArtifact::from_meta(&reseal(meta), None).expect_err("refused");
    assert_eq!(error, EvidenceLedgerArtifactError::SynthesisMismatch("covariance binding"));

    let (_, mut meta) = sealed();
    let env =
        meta.syntheses.iter_mut().find(|s| s.method == "dependence_envelope").expect("envelope");
    env.rho_hi = Some(0.95_f64.to_bits());
    let error = EvidenceLedgerArtifact::from_meta(&reseal(meta), None).expect_err("refused");
    assert_eq!(error, EvidenceLedgerArtifactError::SynthesisMismatch("assumption range"));

    // a synthesis pointed at a different set of studies
    let (_, mut meta) = sealed();
    let dl = meta
        .syntheses
        .iter_mut()
        .find(|s| s.method == "random_effects_dersimonian_laird")
        .expect("dl");
    dl.sources.pop();
    dl.estimates.pop();
    dl.variances.pop();
    let error = EvidenceLedgerArtifact::from_meta(&reseal(meta), None).expect_err("refused");
    assert_eq!(error, EvidenceLedgerArtifactError::SynthesisMismatch("sources"));
}

#[test]
fn a_synthesis_method_that_the_ledger_operation_does_not_license_is_refused() {
    let mut f = fixture();
    // random effects over the assumption-range result: the ledger recorded no independence
    f.records.push(SynthesisRecord {
        result: f.envelope,
        sources: f.records[2].sources.clone(),
        method: SynthesisMethod::FixedEffectInverseVariance {
            estimates: vec![1.0, 3.0],
            variances: vec![1.0, 4.0],
        },
    });
    let error = EvidenceLedgerArtifact::seal(&f.ledger, &f.records).expect_err("refused");
    assert_eq!(
        error,
        EvidenceLedgerArtifactError::SynthesisMismatch("method for the ledger operation")
    );
}

#[test]
fn a_consistently_resealed_artifact_of_another_history_fails_the_retained_identity() {
    let (_, meta) = sealed();
    // a different, internally consistent history: one more source
    let mut f = fixture();
    f.ledger.add_source(data("study_5", 500..510)).expect("add");
    let (_, other) = EvidenceLedgerArtifact::seal(&f.ledger, &f.records).expect("seal");
    let retained = meta.identity;
    let error = EvidenceLedgerArtifact::from_meta(&other, Some(&retained)).expect_err("refused");
    assert_eq!(error, EvidenceLedgerArtifactError::IdentityMismatch { field: "retained_identity" });
}

#[test]
fn versions_markers_and_corruption_are_refused() {
    let (f, meta) = sealed();
    let mut newer = meta.clone();
    newer.version = 2;
    assert!(matches!(
        EvidenceLedgerArtifact::from_meta(&newer, None),
        Err(EvidenceLedgerArtifactError::UnsupportedVersion { version: 2 })
    ));
    let mut foreign = meta;
    foreign.feature = "something_else".to_owned();
    assert!(matches!(
        EvidenceLedgerArtifact::from_meta(&foreign, None),
        Err(EvidenceLedgerArtifactError::UnsupportedSemantics("feature marker"))
    ));
    let mut bytes = EvidenceLedgerArtifact::to_bytes(&f.ledger, &f.records, "evidence-ledger-3")
        .expect("bytes");
    let middle = bytes.len() / 2;
    bytes[middle] ^= 0xff;
    assert!(matches!(
        EvidenceLedgerArtifact::from_bytes(&bytes, None),
        Err(EvidenceLedgerArtifactError::Container(antecedent_io_core::IoError::ChecksumMismatch { section })) if section == "evidence_ledger_meta"
    ));
    assert!(matches!(
        EvidenceLedgerArtifact::from_bytes(&bytes[..bytes.len() / 3], None),
        Err(EvidenceLedgerArtifactError::Decode(m)) if m.contains("failed to fill whole buffer")
    ));
    assert!(matches!(
        EvidenceLedgerArtifact::to_bytes(&f.ledger, &f.records, "  "),
        Err(EvidenceLedgerArtifactError::Encode(m)) if m == "missing artifact id"
    ));
}

#[test]
fn an_operation_the_ledger_refuses_cannot_be_smuggled_through_a_resealed_artifact() {
    let (_, mut meta) = sealed();
    // the stored pool is edited to name one study twice; the identity is resealed
    let combine = meta
        .operations
        .iter_mut()
        .find(|op| op.request.as_ref().is_some_and(|r| r.label == "independent_pool"))
        .expect("pool");
    // list the first study twice in place of the third
    let request = combine.request.as_mut().expect("request");
    let first = request.sources[0].clone();
    request.sources[2] = first;
    let error = EvidenceLedgerArtifact::from_meta(&reseal(meta), None).expect_err("refused");
    assert!(matches!(error, EvidenceLedgerArtifactError::Refused { .. }), "{error}");
    let (code, detail, _) = error.refusal().expect("registered refusal");
    assert_eq!(code, "evidence_double_counted");
    assert_eq!(detail, "evidence_ledger.repeated_source");
}

// -- transport, reuse and repeated use through export ----------------------------------------------

#[test]
fn a_transported_claim_keeps_its_transport_identity_through_export() {
    use antecedent_contract::evidence_ledger::TransportIdentity;
    let mut ledger = EvidenceLedger::new();
    let native = ledger.add_source(data("target_cohort", 0..30)).expect("native");
    let identity = TransportIdentity {
        source_populations: vec!["trial_b".to_owned(), "trial_a".to_owned()],
        target_population: "target".to_owned(),
        selection_diagram: antecedent_core::SemanticDigest::from_bytes([7; 32]),
        assumptions: vec!["s-admissible adjustment set".to_owned()],
    };
    let transported = ledger
        .add_source(
            SourceDeclaration::new("effect_from_trials", SourceKind::ExternalClaim)
                .with_observations(rows(100..160))
                .with_transport(identity),
        )
        .expect("transported");
    let result = ledger
        .combine(CombinationRequest {
            label: "transport".to_owned(),
            operation: SynthesisOperation::CausalTransport {
                transported: vec![transported],
                native: vec![native],
            },
            replaces: None,
        })
        .expect("transport");
    let (_, mut meta) = EvidenceLedgerArtifact::seal(&ledger, &[]).expect("seal");
    let consumed = EvidenceLedgerArtifact::from_meta(&meta, None).expect("consume");
    let record = consumed.ledger().result(result).expect("result");
    assert_eq!(record.transport_identities.len(), 1);
    let stored = consumed.ledger().source(transported).expect("source");
    let carried = stored.declaration.transport.as_ref().expect("transport identity");
    assert_eq!(carried.target_population, "target");
    // canonical order of populations
    assert_eq!(carried.source_populations, vec!["trial_a".to_owned(), "trial_b".to_owned()]);

    // dropping the transport identity from the stored declaration and resealing is refused:
    // the source identity changes and the operations that cite it no longer replay
    let add_transported = meta
        .operations
        .iter_mut()
        .find(|op| op.source.as_ref().is_some_and(|s| s.label == "effect_from_trials"))
        .expect("add");
    add_transported.source.as_mut().expect("source").transport = None;
    let error = EvidenceLedgerArtifact::from_meta(&reseal(meta), None).expect_err("refused");
    // A semantic refusal found by replay, not corruption or a malformed value: either the
    // ledger rejects the stored operation or its recomputed identity differs from the stored one.
    assert!(
        matches!(
            error,
            EvidenceLedgerArtifactError::Refused { .. }
                | EvidenceLedgerArtifactError::ReplayMismatch("operation identity")
        ),
        "{error}"
    );
    assert!(error.refusal().is_some(), "{error}");
}

#[test]
fn evidence_reuse_survives_export_with_mass_counted_once() {
    let mut ledger = EvidenceLedger::new();
    let a = ledger.add_source(data("a", 0..30)).expect("a");
    let b = ledger.add_source(data("b", 20..50)).expect("b");
    let pooled = ledger
        .combine(pooling(
            "covariance_pool",
            &[a, b],
            PoolingDependence::DeclaredCovariance {
                covariance: covariance_digest(2, &[1.0, 0.5, 0.5, 2.0]),
                has_cross_covariance: true,
            },
        ))
        .expect("pooled");
    let cited = ledger
        .combine(CombinationRequest {
            label: "cite".to_owned(),
            operation: SynthesisOperation::EvidenceReuse { of: pooled },
            replaces: None,
        })
        .expect("reuse");
    let bytes =
        EvidenceLedgerArtifact::to_bytes(&ledger, &[], "evidence-ledger-reuse").expect("bytes");
    let consumed = EvidenceLedgerArtifact::from_bytes(&bytes, None).expect("consume");
    let original = consumed.ledger().result(pooled).expect("result").mass.clone();
    assert_eq!(
        (original.declared_slots, original.distinct_observations, original.repeated_slots),
        (60, 50, 10)
    );
    assert_eq!(consumed.ledger().result(cited).expect("result").mass, original);
}

#[test]
fn repeated_observation_use_survives_export_flagged() {
    let mut ledger = EvidenceLedger::new();
    let a = ledger.add_source(data("a", 0..30)).expect("a");
    let repeated = ledger
        .combine(CombinationRequest {
            label: "final_analysis".to_owned(),
            operation: SynthesisOperation::RepeatedObservationUse {
                sources: vec![a],
                stage: "confirmation".to_owned(),
            },
            replaces: None,
        })
        .expect("repeated use");
    let bytes =
        EvidenceLedgerArtifact::to_bytes(&ledger, &[], "evidence-ledger-repeat").expect("bytes");
    let consumed = EvidenceLedgerArtifact::from_bytes(&bytes, None).expect("consume");
    assert!(consumed.ledger().result(repeated).expect("result").data_reuse);
}

#[test]
fn an_exported_result_carries_no_interval_license() {
    let f = fixture();
    let bytes = EvidenceLedgerArtifact::to_bytes(&f.ledger, &f.records, "evidence-ledger-4")
        .expect("export");
    let consumed = EvidenceLedgerArtifact::from_bytes(&bytes, None).expect("consume");
    for result in [f.pooled, f.covariance, f.envelope, f.borrowed] {
        let error = consumed.ledger().request_interval(result).expect_err("closed");
        assert_eq!(error.code, "cell_not_licensed");
        assert_eq!(error.detail, "evidence_ledger.interval_not_licensed");
    }
    assert_eq!(
        consumed.ledger().result(f.envelope).expect("result").claim_class.as_str(),
        "assumption_range"
    );
}
