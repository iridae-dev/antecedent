//! Evidence ledger: overlap, mass accounting, double-counting refusals,
//! distinct operations, transport provenance and downstream revalidation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::ops::Range;

use antecedent_contract::EvidenceDependence;
use antecedent_contract::evidence_ledger::{
    ClaimClass, CombinationRequest, EvidenceLedger, LedgerError, LedgerOperation, Lineage,
    MAX_COMBINATION_SOURCES, ObservationId, Overlap, PoolingDependence, RelationKind, ResultId,
    Revalidation, SourceDeclaration, SourceId, SourceKind, SynthesisKind, SynthesisOperation,
    TransportIdentity,
};
use antecedent_core::SemanticDigest;

fn obs(range: Range<u32>) -> Vec<ObservationId> {
    range.map(|i| ObservationId::from_key("cohort", &i.to_le_bytes())).collect()
}

fn data(label: &str, range: Range<u32>) -> SourceDeclaration {
    SourceDeclaration::new(label, SourceKind::ObservationSet).with_observations(obs(range))
}

fn digest(byte: u8) -> SemanticDigest {
    SemanticDigest::from_bytes([byte; 32])
}

fn transport(target: &str) -> TransportIdentity {
    TransportIdentity {
        source_populations: vec!["trial_a".to_owned(), "trial_b".to_owned()],
        target_population: target.to_owned(),
        selection_diagram: digest(7),
        assumptions: vec!["s-admissible adjustment set".to_owned()],
    }
}

fn pooling(label: &str, ids: &[SourceId], dependence: PoolingDependence) -> CombinationRequest {
    CombinationRequest {
        label: label.to_owned(),
        operation: SynthesisOperation::StatisticalPooling { sources: ids.to_vec(), dependence },
        replaces: None,
    }
}

fn expect_refusal(error: &LedgerError, code: &str, detail: &str) {
    assert_eq!(error.code, code, "{error}");
    assert_eq!(error.detail, detail, "{error}");
}

fn add(ledger: &mut EvidenceLedger, declaration: SourceDeclaration) -> SourceId {
    ledger.add_source(declaration).expect("source is accepted")
}

// -- mass accounting -------------------------------------------------------------------------

/// A = rows 0..100, B = rows 50..150 (50 shared), C = rows 200..210, D = a duplicate of A.
/// Slots are summed per source; distinct observations are the union.
#[test]
fn mass_accounting_counts_each_observation_once() {
    let mut ledger = EvidenceLedger::new();
    let a = add(&mut ledger, data("a", 0..100));
    let b = add(&mut ledger, data("b", 50..150));
    let c = add(&mut ledger, data("c", 200..210));
    let d = add(
        &mut ledger,
        SourceDeclaration::new("a_copy", SourceKind::ObservationSet)
            .with_lineage(Lineage::Duplicate { of: a }),
    );

    let all = ledger.mass_account(&[a, b, c]).expect("mass");
    assert_eq!(all.sources, 3);
    assert_eq!(all.declared_slots, 210);
    assert_eq!(all.distinct_observations, 160);
    assert_eq!(all.repeated_slots, 50);
    assert!(all.unknown_sources.is_empty());

    let disjoint = ledger.mass_account(&[a, c]).expect("mass");
    assert_eq!(
        (disjoint.declared_slots, disjoint.distinct_observations, disjoint.repeated_slots),
        (110, 110, 0)
    );

    // a duplicate adds slots but no observations
    let duplicated = ledger.mass_account(&[a, d]).expect("mass");
    assert_eq!(
        (duplicated.declared_slots, duplicated.distinct_observations, duplicated.repeated_slots),
        (200, 100, 100)
    );
    assert_eq!(ledger.shared_observations(a, b).expect("overlap").len(), 50);
}

#[test]
fn a_registered_result_records_the_once_counted_mass() {
    let mut ledger = EvidenceLedger::new();
    let a = add(&mut ledger, data("a", 0..100));
    let b = add(&mut ledger, data("b", 50..150));
    // pooling with a declared covariance that carries the cross term is licensed
    let covariance = digest(9);
    let result = ledger
        .combine(pooling(
            "pooled_with_cross_terms",
            &[a, b],
            PoolingDependence::DeclaredCovariance { covariance, has_cross_covariance: true },
        ))
        .expect("covariance-preserving pooling is licensed");
    let record = ledger.result(result).expect("record");
    assert_eq!(record.mass.declared_slots, 200);
    assert_eq!(record.mass.distinct_observations, 150);
    assert_eq!(record.mass.repeated_slots, 50);
    assert!(record.dependence_declared);
    assert_eq!(record.claim_class, ClaimClass::PointOnly);
    assert!(record.observations_digest.is_some());
}

// -- refusals of silent double counting ----------------------------------------------------------

#[test]
fn independent_disjoint_sources_pool() {
    let mut ledger = EvidenceLedger::new();
    let a = add(&mut ledger, data("a", 0..10));
    let b = add(&mut ledger, data("b", 10..20));
    assert_eq!(ledger.overlap(a, b).expect("overlap"), Overlap::Independent);
    assert_eq!(ledger.relation(a, b).expect("relation"), RelationKind::Independent);
    let result =
        ledger.combine(pooling("p", &[a, b], PoolingDependence::Independent)).expect("pooled");
    assert_eq!(ledger.standing(result), Some(Revalidation::StillValid));
}

#[test]
fn a_duplicate_re_entering_is_refused() {
    let mut ledger = EvidenceLedger::new();
    let a = add(&mut ledger, data("a", 0..10));
    let copy = add(
        &mut ledger,
        SourceDeclaration::new("a_again", SourceKind::ObservationSet)
            .with_lineage(Lineage::Duplicate { of: a }),
    );
    let error = ledger
        .combine(pooling("p", &[a, copy], PoolingDependence::Independent))
        .expect_err("a duplicate is not independent evidence");
    expect_refusal(&error, "evidence_double_counted", "evidence_ledger.reentry_duplicate");
    assert_eq!(error.sources.len(), 2);
}

#[test]
fn a_transform_re_entering_is_refused() {
    let mut ledger = EvidenceLedger::new();
    let a = add(&mut ledger, data("a", 0..10));
    let scaled = add(
        &mut ledger,
        SourceDeclaration::new("a_standardized", SourceKind::Experiment)
            .with_lineage(Lineage::Transform { of: a, transform: digest(3) }),
    );
    let error = ledger
        .combine(pooling("p", &[a, scaled], PoolingDependence::Independent))
        .expect_err("a transform of the same observations is not new evidence");
    expect_refusal(&error, "evidence_double_counted", "evidence_ledger.reentry_transform");
}

/// A prior built from the same data as the likelihood (a power prior), and a
/// posterior fed back next to the likelihood it already contains.
#[test]
fn prior_plus_likelihood_re_entry_is_refused() {
    let mut ledger = EvidenceLedger::new();
    let likelihood = add(&mut ledger, data("trial", 0..40));
    let power_prior = add(
        &mut ledger,
        SourceDeclaration::new("power_prior_from_trial", SourceKind::Prior)
            .with_observations(obs(0..40)),
    );
    let error = ledger
        .combine(CombinationRequest {
            label: "borrow".to_owned(),
            operation: SynthesisOperation::BayesianBorrowing {
                priors: vec![power_prior],
                likelihood: vec![likelihood],
            },
            replaces: None,
        })
        .expect_err("the data would inform the estimate twice");
    expect_refusal(
        &error,
        "evidence_double_counted",
        "evidence_ledger.reentry_prior_plus_likelihood",
    );

    // the posterior carries the likelihood's observations; pooling it with
    // the likelihood counts them again
    let elicited = add(
        &mut ledger,
        SourceDeclaration::new("elicited", SourceKind::Prior).with_observations(Vec::new()),
    );
    let posterior = add(
        &mut ledger,
        SourceDeclaration::new("posterior", SourceKind::Prior)
            .with_lineage(Lineage::PriorPlusLikelihood { prior: elicited, likelihood }),
    );
    let error = ledger
        .combine(pooling("p", &[posterior, likelihood], PoolingDependence::Independent))
        .expect_err("posterior already contains the likelihood");
    expect_refusal(
        &error,
        "evidence_double_counted",
        "evidence_ledger.reentry_prior_plus_likelihood",
    );

    // an elicited prior with no data origin is licensed beside the likelihood
    ledger
        .combine(CombinationRequest {
            label: "borrow_elicited".to_owned(),
            operation: SynthesisOperation::BayesianBorrowing {
                priors: vec![elicited],
                likelihood: vec![likelihood],
            },
            replaces: None,
        })
        .expect("an elicited prior has no data in common with the likelihood");
}

#[test]
fn an_external_claim_built_from_used_data_is_refused() {
    let mut ledger = EvidenceLedger::new();
    let trial = add(&mut ledger, data("trial", 0..25));
    let claim = add(
        &mut ledger,
        SourceDeclaration::new("vendor_effect", SourceKind::ExternalClaim)
            .with_lineage(Lineage::DerivedFromUsedData { inputs: vec![trial] }),
    );
    let error = ledger
        .combine(pooling("p", &[trial, claim], PoolingDependence::Independent))
        .expect_err("the claim was built from the same observations");
    expect_refusal(
        &error,
        "evidence_double_counted",
        "evidence_ledger.reentry_derived_from_used_data",
    );
    // the derived claim inherits the observations through lineage
    assert_eq!(ledger.mass_account(&[trial, claim]).expect("mass").repeated_slots, 25);

    // the same refusal when the claim only declares the observations it used
    let declared = add(
        &mut ledger,
        SourceDeclaration::new("vendor_effect_declared", SourceKind::ExternalClaim)
            .with_observations(obs(10..20)),
    );
    let error = ledger
        .combine(pooling("p2", &[trial, declared], PoolingDependence::Independent))
        .expect_err("declared rows overlap");
    expect_refusal(
        &error,
        "evidence_double_counted",
        "evidence_ledger.reentry_derived_from_used_data",
    );
}

#[test]
fn undeclared_lineage_with_shared_rows_and_repeated_sources_are_refused() {
    let mut ledger = EvidenceLedger::new();
    let a = add(&mut ledger, data("a", 0..10));
    let b = add(&mut ledger, data("b", 5..15));
    let error = ledger
        .combine(pooling("p", &[a, b], PoolingDependence::Independent))
        .expect_err("shared rows");
    expect_refusal(&error, "evidence_double_counted", "evidence_ledger.shared_observations");
    let error = ledger
        .combine(pooling("p", &[a, a], PoolingDependence::Independent))
        .expect_err("one source twice");
    expect_refusal(&error, "evidence_double_counted", "evidence_ledger.repeated_source");
}

#[test]
fn the_same_evidence_is_not_added_twice_and_rows_are_not_repeated_in_a_source() {
    let mut ledger = EvidenceLedger::new();
    add(&mut ledger, data("a", 0..10));
    let error = ledger.add_source(data("a", 0..10)).expect_err("same identity");
    expect_refusal(&error, "evidence_double_counted", "evidence_ledger.duplicate_source");

    let mut rows = obs(0..5);
    rows.push(rows[0]);
    let error = ledger
        .add_source(
            SourceDeclaration::new("dup_rows", SourceKind::ObservationSet).with_observations(rows),
        )
        .expect_err("a repeated observation inside one source");
    expect_refusal(&error, "evidence_double_counted", "evidence_ledger.duplicate_row");
}

#[test]
fn lineage_must_agree_with_declared_rows() {
    let mut ledger = EvidenceLedger::new();
    let a = add(&mut ledger, data("a", 0..10));
    let error = ledger
        .add_source(
            SourceDeclaration::new("not_a_copy", SourceKind::ObservationSet)
                .with_observations(obs(0..9))
                .with_lineage(Lineage::Duplicate { of: a }),
        )
        .expect_err("a duplicate has the same rows");
    expect_refusal(&error, "invalid_argument", "evidence_ledger.lineage_rows_mismatch");
    let error = ledger
        .add_source(
            SourceDeclaration::new("not_a_subset", SourceKind::ObservationSet)
                .with_observations(obs(5..15))
                .with_lineage(Lineage::Subset { of: a }),
        )
        .expect_err("a subset lies inside its parent");
    expect_refusal(&error, "invalid_argument", "evidence_ledger.lineage_rows_mismatch");
    // a proper subset is accepted and overlaps its parent
    let subset = add(
        &mut ledger,
        SourceDeclaration::new("half", SourceKind::ObservationSet)
            .with_observations(obs(0..5))
            .with_lineage(Lineage::Subset { of: a }),
    );
    assert_eq!(
        ledger.overlap(a, subset).expect("overlap"),
        Overlap::SharedObservations { shared: 5 }
    );
}

// -- unknown dependence ---------------------------------------------------------------------------

#[test]
fn unknown_overlap_is_never_promoted_to_independent() {
    let mut ledger = EvidenceLedger::new();
    let a = add(&mut ledger, data("a", 0..10));
    let vendor = add(&mut ledger, SourceDeclaration::new("vendor", SourceKind::ExternalClaim));
    assert!(matches!(ledger.overlap(a, vendor).expect("overlap"), Overlap::Unknown { .. }));
    assert_eq!(ledger.dependence(a, vendor).expect("dependence"), EvidenceDependence::Unknown);
    assert_eq!(ledger.relation(a, vendor).expect("relation"), RelationKind::UnknownDependence);

    let error = ledger
        .combine(pooling("p", &[a, vendor], PoolingDependence::Independent))
        .expect_err("independence is not assumed");
    expect_refusal(&error, "evidence_overlap_unknown", "evidence_ledger.overlap_unknown");

    // a diagonal covariance states independence, which is not known either
    let error = ledger
        .combine(pooling(
            "diag",
            &[a, vendor],
            PoolingDependence::DeclaredCovariance {
                covariance: digest(1),
                has_cross_covariance: false,
            },
        ))
        .expect_err("a diagonal covariance does not establish independence");
    expect_refusal(&error, "evidence_overlap_unknown", "evidence_ledger.overlap_unknown");

    // an assumption range is licensed, and is labelled as one
    let ranged = ledger
        .combine(pooling(
            "ranged",
            &[a, vendor],
            PoolingDependence::AssumptionRange { rho_lo: 0.0, rho_hi: 0.9 },
        ))
        .expect("an assumption range does not turn unknown into independent");
    let record = ledger.result(ranged).expect("record");
    assert_eq!(record.claim_class, ClaimClass::AssumptionRange);
    assert!(record.mass.unknown_sources.contains(&vendor));
    assert!(record.observations_digest.is_none());
    // and nothing licenses an interval from it
    let error = ledger.request_interval(ranged).expect_err("closed");
    expect_refusal(&error, "cell_not_licensed", "evidence_ledger.interval_not_licensed");

    // a covariance with cross terms is licensed point-only
    ledger
        .combine(pooling(
            "cross",
            &[a, vendor],
            PoolingDependence::DeclaredCovariance {
                covariance: digest(2),
                has_cross_covariance: true,
            },
        ))
        .expect("the dependence is modelled, not assumed away");
}

#[test]
fn an_empty_observation_set_overlaps_nothing_even_unknown_evidence() {
    let mut ledger = EvidenceLedger::new();
    let elicited = add(
        &mut ledger,
        SourceDeclaration::new("elicited", SourceKind::Prior).with_observations(Vec::new()),
    );
    let vendor = add(&mut ledger, SourceDeclaration::new("vendor", SourceKind::ExternalClaim));
    assert_eq!(ledger.overlap(elicited, vendor).expect("overlap"), Overlap::Independent);
}

#[test]
fn assumption_ranges_must_stay_inside_the_definite_interval() {
    let mut ledger = EvidenceLedger::new();
    let a = add(&mut ledger, data("a", 0..10));
    let b = add(&mut ledger, SourceDeclaration::new("b", SourceKind::ExternalClaim));
    let c = add(&mut ledger, SourceDeclaration::new("c", SourceKind::ExternalClaim));
    for (lo, hi) in [(0.3, 0.3), (0.0, 1.0), (-0.5, 0.2), (f64::NAN, 0.5)] {
        let error = ledger
            .combine(pooling(
                "r",
                &[a, b, c],
                PoolingDependence::AssumptionRange { rho_lo: lo, rho_hi: hi },
            ))
            .expect_err("range outside (-1/(k-1), 1)");
        expect_refusal(&error, "invalid_argument", "evidence_ledger.invalid_declaration");
    }
}

// -- distinct operations ----------------------------------------------------------------------------

/// The same overlapping pair (A, B share 50 observations) through every operation.
#[test]
fn each_operation_has_its_own_policy_on_the_same_overlapping_pair() {
    let mut ledger = EvidenceLedger::new();
    let a = add(&mut ledger, data("a", 0..100));
    let b = add(&mut ledger, data("b", 50..150));

    // statistical pooling as independent: refused
    let error = ledger
        .combine(pooling("pool", &[a, b], PoolingDependence::Independent))
        .expect_err("pooling");
    expect_refusal(&error, "evidence_double_counted", "evidence_ledger.shared_observations");

    // Bayesian borrowing: likelihood sources must be independent as well
    let error = ledger
        .combine(CombinationRequest {
            label: "borrow".to_owned(),
            operation: SynthesisOperation::BayesianBorrowing {
                priors: Vec::new(),
                likelihood: vec![a, b],
            },
            replaces: None,
        })
        .expect_err("borrowing");
    expect_refusal(&error, "evidence_double_counted", "evidence_ledger.shared_observations");

    // causal transport: native A and a claim transported from data overlapping A
    let transported = add(
        &mut ledger,
        SourceDeclaration::new("transported_b", SourceKind::ExternalClaim)
            .with_observations(obs(50..150))
            .with_transport(transport("target")),
    );
    let error = ledger
        .combine(CombinationRequest {
            label: "transport".to_owned(),
            operation: SynthesisOperation::CausalTransport {
                transported: vec![transported],
                native: vec![a],
            },
            replaces: None,
        })
        .expect_err("transport");
    expect_refusal(&error, "evidence_double_counted", "evidence_ledger.native_and_transported");

    // repeated use of the same observations is a declared, distinct operation
    let reuse_stage = ledger
        .combine(CombinationRequest {
            label: "second_stage".to_owned(),
            operation: SynthesisOperation::RepeatedObservationUse {
                sources: vec![a, b],
                stage: "confirmation".to_owned(),
            },
            replaces: None,
        })
        .expect("repeated use is licensed and counted once");
    let record = ledger.result(reuse_stage).expect("record");
    assert!(record.data_reuse);
    assert_eq!(record.operation.kind(), SynthesisKind::RepeatedObservationUse);
    assert_eq!(record.mass.distinct_observations, 150);

    // covariance-preserving pooling is licensed; citing it again counts once
    let pooled = ledger
        .combine(pooling(
            "pool_cov",
            &[a, b],
            PoolingDependence::DeclaredCovariance {
                covariance: digest(4),
                has_cross_covariance: true,
            },
        ))
        .expect("pooling with cross terms");
    let cited = ledger
        .combine(CombinationRequest {
            label: "cite".to_owned(),
            operation: SynthesisOperation::EvidenceReuse { of: pooled },
            replaces: None,
        })
        .expect("evidence reuse");
    let original = ledger.result(pooled).expect("record").mass.clone();
    assert_eq!(ledger.result(cited).expect("record").mass, original);
    assert_eq!(ledger.result(cited).expect("record").mass.distinct_observations, 150);

    // a diagonal covariance with shared rows names the shared rows
    let error = ledger
        .combine(pooling(
            "diag",
            &[a, b],
            PoolingDependence::DeclaredCovariance {
                covariance: digest(5),
                has_cross_covariance: false,
            },
        ))
        .expect_err("diagonal covariance with shared rows");
    expect_refusal(
        &error,
        "evidence_double_counted",
        "evidence_ledger.diagonal_covariance_shared_rows",
    );
}

#[test]
fn unknown_dependence_is_refused_by_every_operation_that_needs_it_known() {
    let mut ledger = EvidenceLedger::new();
    let a = add(&mut ledger, data("a", 0..10));
    let prior_with_unknown_origin =
        add(&mut ledger, SourceDeclaration::new("p", SourceKind::Prior));
    let claim = add(
        &mut ledger,
        SourceDeclaration::new("claim", SourceKind::ExternalClaim)
            .with_transport(transport("target")),
    );
    let refused = |ledger: &mut EvidenceLedger, operation: SynthesisOperation| {
        ledger
            .combine(CombinationRequest { label: "x".to_owned(), operation, replaces: None })
            .expect_err("unknown dependence")
    };
    let error = refused(
        &mut ledger,
        SynthesisOperation::BayesianBorrowing {
            priors: vec![prior_with_unknown_origin],
            likelihood: vec![a],
        },
    );
    expect_refusal(&error, "evidence_overlap_unknown", "evidence_ledger.overlap_unknown");
    let error = refused(
        &mut ledger,
        SynthesisOperation::CausalTransport { transported: vec![claim], native: vec![a] },
    );
    expect_refusal(&error, "evidence_overlap_unknown", "evidence_ledger.overlap_unknown");
    let error = refused(
        &mut ledger,
        SynthesisOperation::RepeatedObservationUse {
            sources: vec![claim],
            stage: "later".to_owned(),
        },
    );
    expect_refusal(&error, "evidence_overlap_unknown", "evidence_ledger.overlap_unknown");
}

#[test]
fn operation_arity_and_kinds_are_checked() {
    let mut ledger = EvidenceLedger::new();
    let a = add(&mut ledger, data("a", 0..10));
    let b = add(&mut ledger, data("b", 10..20));
    let error = ledger
        .combine(pooling("one", &[a], PoolingDependence::Independent))
        .expect_err("a pool needs two");
    expect_refusal(&error, "invalid_argument", "evidence_ledger.operation_arity");
    let error = ledger
        .combine(CombinationRequest {
            label: "not_a_prior".to_owned(),
            operation: SynthesisOperation::BayesianBorrowing {
                priors: vec![b],
                likelihood: vec![a],
            },
            replaces: None,
        })
        .expect_err("a prior has kind prior");
    expect_refusal(&error, "invalid_argument", "evidence_ledger.invalid_declaration");
}

// -- transport provenance --------------------------------------------------------------------------

#[test]
fn a_transported_claim_never_counts_as_native_target_data() {
    let mut ledger = EvidenceLedger::new();
    let native = add(&mut ledger, data("target_cohort", 0..30));
    let transported = add(
        &mut ledger,
        SourceDeclaration::new("effect_from_trials", SourceKind::ExternalClaim)
            .with_observations(obs(100..160))
            .with_transport(transport("target")),
    );
    let result = ledger
        .combine(CombinationRequest {
            label: "transport".to_owned(),
            operation: SynthesisOperation::CausalTransport {
                transported: vec![transported],
                native: vec![native],
            },
            replaces: None,
        })
        .expect("disjoint native and transported evidence");
    let record = ledger.result(result).expect("record");
    assert_eq!(record.transport_identities.len(), 1);
    assert_eq!(
        record.transport_identities[0],
        ledger
            .source(transported)
            .expect("source")
            .declaration
            .transport
            .as_ref()
            .expect("t")
            .digest()
    );

    // transport-then-synthesize: pooled, the transport identity travels with it
    let pooled = ledger
        .combine(pooling("pooled", &[native, transported], PoolingDependence::Independent))
        .expect("disjoint");
    assert_eq!(ledger.result(pooled).expect("record").transport_identities.len(), 1);

    // a claim without its identity cannot be transported
    let bare = add(&mut ledger, data("bare_claim_data", 300..310));
    let error = ledger
        .combine(CombinationRequest {
            label: "bad".to_owned(),
            operation: SynthesisOperation::CausalTransport {
                transported: vec![bare],
                native: vec![native],
            },
            replaces: None,
        })
        .expect_err("transport identity missing");
    expect_refusal(&error, "invalid_argument", "evidence_ledger.transport_identity_missing");

    // a transported claim in the native slot
    let error = ledger
        .combine(CombinationRequest {
            label: "bad".to_owned(),
            operation: SynthesisOperation::CausalTransport {
                transported: vec![transported],
                native: vec![transported],
            },
            replaces: None,
        })
        .expect_err("same source both native and transported");
    expect_refusal(&error, "evidence_double_counted", "evidence_ledger.native_and_transported");

    // another transported claim in the native slot
    let other = add(
        &mut ledger,
        SourceDeclaration::new("other_transported", SourceKind::ExternalClaim)
            .with_observations(obs(400..410))
            .with_transport(transport("target")),
    );
    let error = ledger
        .combine(CombinationRequest {
            label: "bad".to_owned(),
            operation: SynthesisOperation::CausalTransport {
                transported: vec![transported],
                native: vec![other],
            },
            replaces: None,
        })
        .expect_err("a transported claim is not native");
    expect_refusal(&error, "invalid_argument", "evidence_ledger.transported_as_native");

    // raw observations are not transported
    let error = ledger
        .add_source(data("raw", 500..510).with_transport(transport("target")))
        .expect_err("observations are not transported");
    expect_refusal(&error, "invalid_argument", "evidence_ledger.invalid_declaration");
}

/// The same source as native and as transported: pooling refuses it even
/// when a covariance is declared, because the category error is not a
/// dependence to be modelled.
#[test]
fn the_same_source_native_and_transported_is_refused_even_with_a_covariance() {
    let mut ledger = EvidenceLedger::new();
    let native = add(&mut ledger, data("target_cohort", 0..30));
    let transported = add(
        &mut ledger,
        SourceDeclaration::new("claim_from_target_rows", SourceKind::FittedModel)
            .with_observations(obs(10..40))
            .with_transport(transport("target")),
    );
    let error = ledger
        .combine(pooling(
            "p",
            &[native, transported],
            PoolingDependence::DeclaredCovariance {
                covariance: digest(6),
                has_cross_covariance: true,
            },
        ))
        .expect_err("native and transported");
    expect_refusal(&error, "evidence_double_counted", "evidence_ledger.native_and_transported");
}

// -- operations and downstream revalidation -------------------------------------------------------------

struct Scenario {
    ledger: EvidenceLedger,
    s1: SourceId,
    s2: SourceId,
    s3: SourceId,
    model: SourceId,
    r_s1_s2: ResultId,
    r_s2_s3: ResultId,
    r_all: ResultId,
    r_cite: ResultId,
    r_model: ResultId,
}

fn scenario() -> Scenario {
    let mut ledger = EvidenceLedger::new();
    let s1 = add(&mut ledger, data("s1", 0..10));
    let s2 = add(&mut ledger, data("s2", 10..20));
    let s3 = add(&mut ledger, data("s3", 20..30));
    let r_s1_s2 =
        ledger.combine(pooling("s1_s2", &[s1, s2], PoolingDependence::Independent)).expect("ok");
    let r_s2_s3 =
        ledger.combine(pooling("s2_s3", &[s2, s3], PoolingDependence::Independent)).expect("ok");
    let r_all =
        ledger.combine(pooling("all", &[s1, s2, s3], PoolingDependence::Independent)).expect("ok");
    let r_cite = ledger
        .combine(CombinationRequest {
            label: "cite_s1_s2".to_owned(),
            operation: SynthesisOperation::EvidenceReuse { of: r_s1_s2 },
            replaces: None,
        })
        .expect("ok");
    // a fitted model built from s1; a result pooling it with s3
    let model = add(
        &mut ledger,
        SourceDeclaration::new("model_on_s1", SourceKind::FittedModel)
            .with_lineage(Lineage::DerivedFromUsedData { inputs: vec![s1] }),
    );
    let r_model = ledger
        .combine(pooling("model_s3", &[model, s3], PoolingDependence::Independent))
        .expect("ok");
    Scenario { ledger, s1, s2, s3, model, r_s1_s2, r_s2_s3, r_all, r_cite, r_model }
}

#[test]
fn withdrawing_a_source_classifies_and_refuses_dependents() {
    let mut s = scenario();
    let applied = s.ledger.withdraw_source(s.s1, "consent withdrawn").expect("withdraw").clone();
    assert_eq!(applied.effect.retired_source, Some(s.s1));
    let class = |id: ResultId| {
        applied.effect.revalidation.iter().find(|r| r.result == id).expect("classified").class
    };
    // pooling s1 and s2 loses a member of two: nothing left to pool
    assert_eq!(class(s.r_s1_s2), Revalidation::Withdrawn);
    // citing a withdrawn result is itself withdrawn
    assert_eq!(class(s.r_cite), Revalidation::Withdrawn);
    // untouched
    assert_eq!(class(s.r_s2_s3), Revalidation::StillValid);
    // three studies minus one still pool: recompute
    assert_eq!(class(s.r_all), Revalidation::Recompute);
    // the model derived from s1 is lost through lineage; one member of two remains
    assert_eq!(class(s.r_model), Revalidation::Withdrawn);

    // new work cannot rest on the withdrawn source, directly or through lineage
    let error = s
        .ledger
        .combine(pooling("again", &[s.s1, s.s3], PoolingDependence::Independent))
        .expect_err("withdrawn");
    expect_refusal(&error, "evidence_source_withdrawn", "evidence_ledger.source_withdrawn");
    let error = s
        .ledger
        .combine(pooling("via_model", &[s.model, s.s2], PoolingDependence::Independent))
        .expect_err("the model rests on a withdrawn source");
    expect_refusal(&error, "evidence_source_withdrawn", "evidence_ledger.source_withdrawn");
    // a withdrawn result is not usable and cannot be cited again
    let error = s.ledger.require_valid(s.r_s1_s2).expect_err("withdrawn result");
    expect_refusal(&error, "evidence_source_withdrawn", "evidence_ledger.result_not_valid");
    let error = s
        .ledger
        .combine(CombinationRequest {
            label: "cite_again".to_owned(),
            operation: SynthesisOperation::EvidenceReuse { of: s.r_s1_s2 },
            replaces: None,
        })
        .expect_err("withdrawn result");
    expect_refusal(&error, "evidence_source_withdrawn", "evidence_ledger.result_not_valid");
    // a withdrawn source cannot be withdrawn twice
    let error = s.ledger.withdraw_source(s.s1, "again").expect_err("already withdrawn");
    expect_refusal(&error, "evidence_source_withdrawn", "evidence_ledger.source_withdrawn");

    // the explicit revalidation reports the same classification and is stable
    let first = s.ledger.revalidate_downstream().expect("revalidate");
    let second = s.ledger.revalidate_downstream().expect("revalidate");
    assert_eq!(first, second);
    assert_eq!(first, applied.effect.revalidation);
    assert_eq!(s.ledger.standing(s.r_s2_s3), Some(Revalidation::StillValid));
    assert_eq!(s.ledger.standing(s.r_all), Some(Revalidation::Recompute));
}

#[test]
fn superseding_a_model_marks_dependents_for_recompute_and_a_recompute_replaces_them() {
    let mut s = scenario();
    let successor = SourceDeclaration::new("model_v2_on_s1", SourceKind::FittedModel)
        .with_lineage(Lineage::DerivedFromUsedData { inputs: vec![s.s1] });
    let applied = s.ledger.supersede_source(s.model, successor).expect("supersede").clone();
    let model_v2 = applied.effect.created_source.expect("successor");
    assert_eq!(applied.effect.retired_source, Some(s.model));
    let class = |id: ResultId| {
        applied.effect.revalidation.iter().find(|r| r.result == id).expect("classified").class
    };
    assert_eq!(class(s.r_model), Revalidation::Recompute);
    assert_eq!(class(s.r_all), Revalidation::StillValid);

    // the superseded model cannot enter new results and points to its successor
    let error = s
        .ledger
        .combine(pooling("stale", &[s.model, s.s2], PoolingDependence::Independent))
        .expect_err("superseded");
    expect_refusal(&error, "evidence_source_withdrawn", "evidence_ledger.source_superseded");
    assert!(error.message.contains(&model_v2.to_hex()));

    // recompute on the successor; the old result is retired and preserved
    let recomputed = s
        .ledger
        .combine(CombinationRequest {
            label: "model_s3".to_owned(),
            operation: SynthesisOperation::StatisticalPooling {
                sources: vec![model_v2, s.s3],
                dependence: PoolingDependence::Independent,
            },
            replaces: Some(s.r_model),
        })
        .expect("recompute");
    assert_eq!(s.ledger.standing(recomputed), Some(Revalidation::StillValid));
    assert_eq!(s.ledger.standing(s.r_model), Some(Revalidation::Withdrawn));
    assert!(s.ledger.result(s.r_model).is_some(), "the original claim is preserved");
    let error = s
        .ledger
        .combine(CombinationRequest {
            label: "twice".to_owned(),
            operation: SynthesisOperation::StatisticalPooling {
                sources: vec![model_v2, s.s2],
                dependence: PoolingDependence::Independent,
            },
            replaces: Some(s.r_model),
        })
        .expect_err("already replaced");
    expect_refusal(&error, "invalid_argument", "evidence_ledger.unknown_result");
}

#[test]
fn update_replace_and_supersede_have_distinct_rules() {
    let mut ledger = EvidenceLedger::new();
    let s = add(&mut ledger, data("s", 0..10));
    let model = add(
        &mut ledger,
        SourceDeclaration::new("m", SourceKind::FittedModel).with_observations(obs(0..10)),
    );

    // update only grows
    let error = ledger.update_source(s, data("s_shrunk", 0..5)).expect_err("drops rows");
    expect_refusal(&error, "invalid_argument", "evidence_ledger.update_drops_rows");
    let grown = ledger.update_source(s, data("s_grown", 0..20)).expect("grows").clone();
    assert_eq!(grown.operation.name(), "update");
    // replace may drop (a correction)
    let corrected = ledger
        .replace_source(grown.effect.created_source.expect("id"), data("s_corrected", 5..12))
        .expect("replace")
        .clone();
    assert_eq!(corrected.operation.name(), "replace");
    // supersede is for a model or a claim
    let error = ledger
        .supersede_source(corrected.effect.created_source.expect("id"), data("x", 0..1))
        .expect_err("observations are updated or replaced, not superseded");
    expect_refusal(&error, "invalid_argument", "evidence_ledger.supersede_kind_mismatch");
    let error = ledger
        .supersede_source(
            model,
            SourceDeclaration::new("claim", SourceKind::ExternalClaim)
                .with_observations(obs(0..10)),
        )
        .expect_err("a claim does not supersede a model");
    expect_refusal(&error, "invalid_argument", "evidence_ledger.supersede_kind_mismatch");
    let error = ledger
        .replace_source(
            model,
            SourceDeclaration::new("claim", SourceKind::ExternalClaim)
                .with_observations(obs(0..10)),
        )
        .expect_err("kinds match");
    expect_refusal(&error, "invalid_argument", "evidence_ledger.kind_mismatch");
    // a retired source is not changed again
    let error = ledger.update_source(s, data("s_again", 0..30)).expect_err("retired");
    expect_refusal(&error, "evidence_source_withdrawn", "evidence_ledger.source_superseded");
}

#[test]
fn an_update_of_unknown_observations_is_not_verifiable() {
    let mut ledger = EvidenceLedger::new();
    let vendor = add(&mut ledger, SourceDeclaration::new("v", SourceKind::ExternalClaim));
    let error = ledger
        .update_source(vendor, SourceDeclaration::new("v2", SourceKind::ExternalClaim))
        .expect_err("monotone growth cannot be shown");
    expect_refusal(&error, "evidence_overlap_unknown", "evidence_ledger.overlap_unknown");
}

#[test]
fn updating_a_source_marks_dependents_for_recompute() {
    let mut s = scenario();
    let applied = s.ledger.update_source(s.s2, data("s2_more", 10..25)).expect("update").clone();
    let class = |id: ResultId| {
        applied.effect.revalidation.iter().find(|r| r.result == id).expect("classified").class
    };
    assert_eq!(class(s.r_s1_s2), Revalidation::Recompute);
    assert_eq!(class(s.r_s2_s3), Revalidation::Recompute);
    assert_eq!(class(s.r_all), Revalidation::Recompute);
    assert_eq!(class(s.r_cite), Revalidation::Recompute);
    assert_eq!(class(s.r_model), Revalidation::StillValid);
}

// -- identity, determinism, atomicity -----------------------------------------------------------------

#[test]
fn operation_identities_chain_and_replay_identically() {
    let build = |order: &[usize]| {
        let mut ledger = EvidenceLedger::new();
        let declarations = [data("a", 0..10), data("b", 10..20), data("c", 20..30)];
        let ids: Vec<SourceId> =
            order.iter().map(|&i| add(&mut ledger, declarations[i].clone())).collect();
        (ledger, ids)
    };
    let (first, ids_first) = build(&[0, 1, 2]);
    let (again, ids_again) = build(&[0, 1, 2]);
    assert_eq!(ids_first, ids_again);
    assert_eq!(first.head(), again.head());
    assert_eq!(first.state_digest(), again.state_digest());
    // every operation has its own identity
    let operations = first.operations();
    for (i, left) in operations.iter().enumerate() {
        for right in &operations[i + 1..] {
            assert_ne!(left.id, right.id);
        }
    }
    // source identity is content, independent of the order they were added in;
    // the history (head) records the order
    let (reordered, _) = build(&[2, 0, 1]);
    for source in first.sources() {
        assert!(reordered.source(source.id).is_some());
    }
    assert_ne!(first.head(), reordered.head());
}

#[test]
fn a_result_identity_ignores_the_order_of_its_sources() {
    let mut ledger = EvidenceLedger::new();
    let a = add(&mut ledger, data("a", 0..10));
    let b = add(&mut ledger, data("b", 10..20));
    let c = add(&mut ledger, data("c", 20..30));
    let forward =
        ledger.combine(pooling("p", &[a, b, c], PoolingDependence::Independent)).expect("pooled");
    let mut twin = EvidenceLedger::new();
    let a2 = add(&mut twin, data("a", 0..10));
    let b2 = add(&mut twin, data("b", 10..20));
    let c2 = add(&mut twin, data("c", 20..30));
    let backward =
        twin.combine(pooling("p", &[c2, a2, b2], PoolingDependence::Independent)).expect("pooled");
    assert_eq!(forward, backward);
}

#[test]
fn a_refused_operation_leaves_the_ledger_unchanged() {
    let mut ledger = EvidenceLedger::new();
    let a = add(&mut ledger, data("a", 0..10));
    let b = add(&mut ledger, data("b", 5..15));
    let digest_before = ledger.state_digest();
    let head_before = ledger.head();
    let count = ledger.operations().len();
    ledger
        .apply(LedgerOperation::Combine {
            request: pooling("p", &[a, b], PoolingDependence::Independent),
        })
        .expect_err("refused");
    assert_eq!(ledger.state_digest(), digest_before);
    assert_eq!(ledger.head(), head_before);
    assert_eq!(ledger.operations().len(), count);
    assert!(ledger.results().is_empty());
}

#[test]
fn entry_overlaps_are_reported_but_not_refused() {
    let mut ledger = EvidenceLedger::new();
    let a = add(&mut ledger, data("a", 0..10));
    let applied = ledger.apply(LedgerOperation::Add { source: data("b", 5..15) }).expect("add");
    assert_eq!(applied.effect.detected_overlaps.len(), 1);
    assert_eq!(applied.effect.detected_overlaps[0].existing, a);
    assert_eq!(applied.effect.detected_overlaps[0].shared, 5);
}

// -- composition-vocabulary adapter ------------------------------------------------------------------

#[test]
fn relations_use_the_composition_vocabulary() {
    let mut ledger = EvidenceLedger::new();
    let rows = add(&mut ledger, data("rows", 0..10));
    let prior = add(
        &mut ledger,
        SourceDeclaration::new("power_prior", SourceKind::Prior).with_observations(obs(100..110)),
    );
    let model = add(
        &mut ledger,
        SourceDeclaration::new("model", SourceKind::FittedModel).with_observations(obs(200..210)),
    );
    let derived = |ledger: &mut EvidenceLedger, label: &str, parent: SourceId| {
        add(
            ledger,
            SourceDeclaration::new(label, SourceKind::ExternalClaim)
                .with_lineage(Lineage::DerivedFromUsedData { inputs: vec![parent] }),
        )
    };
    let from_rows = (derived(&mut ledger, "r1", rows), derived(&mut ledger, "r2", rows));
    let from_prior = (derived(&mut ledger, "p1", prior), derived(&mut ledger, "p2", prior));
    let from_model = (derived(&mut ledger, "m1", model), derived(&mut ledger, "m2", model));
    assert_eq!(ledger.relation(from_rows.0, from_rows.1).expect("r"), RelationKind::SharedData);
    assert_eq!(ledger.relation(from_prior.0, from_prior.1).expect("r"), RelationKind::SharedPrior);
    assert_eq!(
        ledger.relation(from_model.0, from_model.1).expect("r"),
        RelationKind::SharedFittedModel
    );
    assert_eq!(RelationKind::SharedFittedModel.as_str(), "shared_fitted_model");
    assert_eq!(ledger.dependence(from_rows.0, from_rows.1).expect("d"), EvidenceDependence::Shared);
    assert_eq!(ledger.dependence(rows, prior).expect("d"), EvidenceDependence::Independent);
    // the first four synthesis kinds carry the composition operations' names
    assert_eq!(SynthesisKind::StatisticalPooling.as_str(), "statistical_pooling");
    assert_eq!(SynthesisKind::BayesianBorrowing.as_str(), "bayesian_borrowing");
    assert_eq!(SynthesisKind::CausalTransport.as_str(), "causal_transport");
    assert_eq!(SynthesisKind::EvidenceReuse.as_str(), "evidence_reuse");
}

#[test]
fn counts_are_not_capped_and_blank_labels_are_refused() {
    let mut ledger = EvidenceLedger::new();
    let count = u32::try_from(MAX_COMBINATION_SOURCES).expect("the default bound fits u32") + 1;
    let ids: Vec<SourceId> = (0..count)
        .map(|i| {
            #[allow(
                clippy::range_plus_one,
                reason = "`data` takes a half-open `Range<u32>`, not an inclusive range"
            )]
            let one = i..i + 1;
            add(&mut ledger, data(&format!("s{i}"), one))
        })
        .collect();
    ledger
        .combine(pooling("many_sources", &ids, PoolingDependence::Independent))
        .expect("a combination may name more sources than the decoder default");
    let error = ledger
        .add_source(SourceDeclaration::new("  ", SourceKind::ObservationSet))
        .expect_err("empty label");
    expect_refusal(&error, "invalid_argument", "evidence_ledger.invalid_declaration");
}

// -- evidence reuse and repeated observation use -----------------------------------------------

#[test]
fn evidence_reuse_counts_the_cited_result_once() {
    let mut ledger = EvidenceLedger::new();
    let a = add(&mut ledger, data("a", 0..30));
    let b = add(&mut ledger, data("b", 30..50));
    let pooled =
        ledger.combine(pooling("pooled", &[a, b], PoolingDependence::Independent)).expect("pooled");
    let cite = |ledger: &mut EvidenceLedger, label: &str| {
        ledger
            .combine(CombinationRequest {
                label: label.to_owned(),
                operation: SynthesisOperation::EvidenceReuse { of: pooled },
                replaces: None,
            })
            .expect("reuse")
    };
    let first = cite(&mut ledger, "second_analysis");
    let second = cite(&mut ledger, "third_analysis");
    assert_ne!(first, second);
    let original = ledger.result(pooled).expect("record").mass.clone();
    assert_eq!(original.distinct_observations, 50);
    // each citation reports the same once-counted mass, not a multiple of it
    for id in [first, second] {
        let record = ledger.result(id).expect("record");
        assert_eq!(record.mass, original);
        assert_eq!(record.operation.kind(), SynthesisKind::EvidenceReuse);
        assert_eq!(ledger.result_sources(id), ledger.result_sources(pooled));
    }
}

#[test]
fn repeated_observation_use_is_flagged_and_stays_flagged_when_cited() {
    let mut ledger = EvidenceLedger::new();
    let a = add(&mut ledger, data("a", 0..30));
    let repeated = ledger
        .combine(CombinationRequest {
            label: "interim_then_final".to_owned(),
            operation: SynthesisOperation::RepeatedObservationUse {
                sources: vec![a],
                stage: "final_analysis".to_owned(),
            },
            replaces: None,
        })
        .expect("repeated use is declared");
    let record = ledger.result(repeated).expect("record");
    assert!(record.data_reuse);
    assert_eq!(record.mass.distinct_observations, 30);
    let cited = ledger
        .combine(CombinationRequest {
            label: "cite".to_owned(),
            operation: SynthesisOperation::EvidenceReuse { of: repeated },
            replaces: None,
        })
        .expect("reuse");
    assert!(ledger.result(cited).expect("record").data_reuse);
    // a stage with no name, or no sources, is not a declaration
    let error = ledger
        .combine(CombinationRequest {
            label: "nameless".to_owned(),
            operation: SynthesisOperation::RepeatedObservationUse {
                sources: vec![a],
                stage: String::new(),
            },
            replaces: None,
        })
        .expect_err("a stage is named");
    expect_refusal(&error, "invalid_argument", "evidence_ledger.operation_arity");
}

#[test]
fn a_reuse_of_a_stale_result_is_refused() {
    let mut ledger = EvidenceLedger::new();
    let a = add(&mut ledger, data("a", 0..30));
    let b = add(&mut ledger, data("b", 30..50));
    let c = add(&mut ledger, data("c", 50..60));
    let pooled = ledger
        .combine(pooling("pooled", &[a, b, c], PoolingDependence::Independent))
        .expect("pooled");
    ledger.update_source(c, data("c_more", 50..70)).expect("update");
    assert_eq!(ledger.standing(pooled), Some(Revalidation::Recompute));
    let error = ledger
        .combine(CombinationRequest {
            label: "cite".to_owned(),
            operation: SynthesisOperation::EvidenceReuse { of: pooled },
            replaces: None,
        })
        .expect_err("a stale result is recomputed before it is cited");
    expect_refusal(&error, "evidence_source_withdrawn", "evidence_ledger.result_not_valid");
}

#[test]
fn no_interval_or_posterior_is_licensed_from_pooled_evidence() {
    let mut ledger = EvidenceLedger::new();
    let a = add(&mut ledger, data("a", 0..10));
    let b = add(&mut ledger, data("b", 10..20));
    let pooled =
        ledger.combine(pooling("p", &[a, b], PoolingDependence::Independent)).expect("pooled");
    let record = ledger.result(pooled).expect("record");
    // the ledger licenses a point-only statement
    assert_eq!(record.claim_class, ClaimClass::PointOnly);
    let error = ledger.request_interval(pooled).expect_err("the interval route is closed");
    expect_refusal(&error, "cell_not_licensed", "evidence_ledger.interval_not_licensed");
    assert!(error.message.contains("point_only"), "{}", error.message);
}
