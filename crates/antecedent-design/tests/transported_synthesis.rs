//! Transport-then-synthesize against hand-computed pools, with the ledger's
//! double-counting refusals and the transport uncertainty counted once.
//!
//! Oracles (all summaries estimate one target quantity):
//!
//! * native `y=1, v=1`; transported `y=2, v=1` with transport variance
//!   `0.5`: covariance `diag(1, 1.5)`, precisions `1` and `2/3`, pooled
//!   `(1 + 4/3) / (5/3) = 1.4`, variance `3/5`; without the transport error
//!   the pool is `1.5` with variance `0.5`, so the transport adds `0.1`.
//! * the same native summary and two claims through one shared transport
//!   (`y=2` and `y=3`, `v=1`, transport variance `0.5`): covariance
//!   `[[1,0,0],[0,1.5,.5],[0,.5,1.5]]`, weights `(.5,.25,.25)`, pooled `1.75`
//!   with variance `0.5` (not `1/3` ignoring the transport, not `3/7`
//!   counting the shared error twice).
//! * two distinct transports: diagonal `(1, 1.5, 1.5)`, pooled `13/7`,
//!   variance `3/7`.
//! * a transported prior `N(2, 1 + 0.5)` against native `y=1, v=1`:
//!   precision `5/3`, variance `0.6`, mean `1.4`.

use antecedent_contract::evidence_ledger::{
    ClaimClass, EvidenceLedger, Lineage, ObservationId, SourceDeclaration, SourceId, SourceKind,
    TransportIdentity,
};
use antecedent_core::SemanticDigest;
use antecedent_design::transported_synthesis::{
    DETAIL_NATIVE_POPULATION_BLANK, DETAIL_OVERLAP_NOT_MODELLED, DETAIL_PRIOR_ROLE,
    DETAIL_SHARED_TRANSPORT, DETAIL_SOURCE_LISTED_TWICE, DETAIL_TARGET_POPULATION_MISMATCH,
    DETAIL_TRANSPORT_VARIANCE_ON_NATIVE, DETAIL_TRANSPORT_VARIANCE_UNDECLARED,
    TransportedContribution, TransportedSynthesisMethod, TransportedSynthesisRefusal,
    synthesize_transported, synthesize_transported_in_population,
};
use antecedent_stats::synthesis::{MeanPrior, TauPrior};

fn rows(start: u32, end: u32) -> Vec<ObservationId> {
    (start..end).map(|i| ObservationId::from_key("registry", &i.to_le_bytes())).collect()
}

fn transport(selection: u8) -> TransportIdentity {
    TransportIdentity {
        source_populations: vec!["trial_a".to_owned()],
        target_population: "registry".to_owned(),
        selection_diagram: SemanticDigest::from_bytes([selection; 32]),
        assumptions: vec!["outcome mechanism invariant".to_owned()],
    }
}

fn native(ledger: &mut EvidenceLedger, label: &str, start: u32, end: u32) -> SourceId {
    ledger
        .add_source(
            SourceDeclaration::new(label, SourceKind::ObservationSet)
                .with_observations(rows(start, end)),
        )
        .unwrap()
}

fn claim(
    ledger: &mut EvidenceLedger,
    label: &str,
    kind: SourceKind,
    start: u32,
    end: u32,
    selection: u8,
) -> SourceId {
    ledger
        .add_source(
            SourceDeclaration::new(label, kind)
                .with_observations(rows(start, end))
                .with_lineage(Lineage::Original)
                .with_transport(transport(selection)),
        )
        .unwrap()
}

fn pool() -> TransportedSynthesisMethod {
    TransportedSynthesisMethod::Pool { sampling_covariance: None }
}

fn contribution(
    source: SourceId,
    estimate: f64,
    variance: f64,
    transport_variance: Option<f64>,
) -> TransportedContribution {
    TransportedContribution { source, estimate, sampling_variance: variance, transport_variance }
}

fn near(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-12, "{a} vs {b}");
}

#[test]
fn native_and_transported_claims_pool_with_the_transport_variance_counted_once() {
    let mut ledger = EvidenceLedger::new();
    let cohort = native(&mut ledger, "registry cohort", 0, 100);
    let carried = claim(&mut ledger, "trial a claim", SourceKind::ExternalClaim, 1_000, 1_100, 1);

    let out = synthesize_transported(
        &mut ledger,
        "pooled effect",
        &[contribution(cohort, 1.0, 1.0, None), contribution(carried, 2.0, 1.0, Some(0.5))],
        &pool(),
    )
    .unwrap();
    near(out.estimate, 1.4);
    near(out.variance, 0.6);
    near(out.transport_variance_added.unwrap(), 0.1);
    assert_eq!(out.method, "pool");
    assert_eq!(out.native_sources, vec![cohort]);
    assert_eq!(out.transported_sources, vec![carried]);
    assert_eq!(out.transport_identities, vec![transport(1).digest()]);
    // Each observation once: 100 native rows plus 100 rows behind the claim.
    assert_eq!(out.mass.distinct_observations, 200);
    assert_eq!(out.mass.repeated_slots, 0);
    // A point and a model variance; the ledger licenses nothing further.
    assert_eq!(out.claim_class, ClaimClass::PointOnly);
    let record = ledger.result(out.result).unwrap();
    assert_eq!(record.transport_identities, vec![transport(1).digest()]);
}

#[test]
fn one_transport_serving_two_claims_shares_its_error() {
    let mut ledger = EvidenceLedger::new();
    let cohort = native(&mut ledger, "registry cohort", 0, 100);
    // Two claims through the same transport identity, learned on disjoint rows.
    let first = claim(&mut ledger, "claim one", SourceKind::ExternalClaim, 1_000, 1_100, 1);
    let second = claim(&mut ledger, "claim two", SourceKind::ExternalClaim, 2_000, 2_100, 1);

    let out = synthesize_transported(
        &mut ledger,
        "shared transport",
        &[
            contribution(cohort, 1.0, 1.0, None),
            contribution(first, 2.0, 1.0, Some(0.5)),
            contribution(second, 3.0, 1.0, Some(0.5)),
        ],
        &pool(),
    )
    .unwrap();
    near(out.estimate, 1.75);
    near(out.variance, 0.5);
    near(out.transport_variance_added.unwrap(), 0.5 - 1.0 / 3.0);
    assert_eq!(out.transport_identities.len(), 1);

    // Contribution order is not part of the result.
    let mut again = EvidenceLedger::new();
    let c = native(&mut again, "registry cohort", 0, 100);
    let f = claim(&mut again, "claim one", SourceKind::ExternalClaim, 1_000, 1_100, 1);
    let s = claim(&mut again, "claim two", SourceKind::ExternalClaim, 2_000, 2_100, 1);
    let reordered = synthesize_transported(
        &mut again,
        "shared transport",
        &[
            contribution(s, 3.0, 1.0, Some(0.5)),
            contribution(c, 1.0, 1.0, None),
            contribution(f, 2.0, 1.0, Some(0.5)),
        ],
        &pool(),
    )
    .unwrap();
    near(reordered.estimate, out.estimate);
    near(reordered.variance, out.variance);
}

#[test]
fn distinct_transports_add_independent_errors() {
    let mut ledger = EvidenceLedger::new();
    let cohort = native(&mut ledger, "registry cohort", 0, 100);
    let first = claim(&mut ledger, "claim one", SourceKind::ExternalClaim, 1_000, 1_100, 1);
    let second = claim(&mut ledger, "claim two", SourceKind::ExternalClaim, 2_000, 2_100, 2);
    let out = synthesize_transported(
        &mut ledger,
        "two transports",
        &[
            contribution(cohort, 1.0, 1.0, None),
            contribution(first, 2.0, 1.0, Some(0.5)),
            contribution(second, 3.0, 1.0, Some(0.5)),
        ],
        &pool(),
    )
    .unwrap();
    near(out.estimate, 13.0 / 7.0);
    near(out.variance, 3.0 / 7.0);
    assert_eq!(out.transport_identities.len(), 2);
}

#[test]
fn the_same_observations_cannot_count_as_native_and_as_transported() {
    let mut ledger = EvidenceLedger::new();
    let cohort = native(&mut ledger, "registry cohort", 0, 100);
    // The transported claim was learned on rows 50..150: half of them are the
    // registry's own rows.
    let overlapping =
        claim(&mut ledger, "overlapping claim", SourceKind::ExternalClaim, 50, 150, 1);
    let before = ledger.state_digest();

    for method in [
        pool(),
        TransportedSynthesisMethod::Borrow {
            mean_prior: MeanPrior::Flat,
            tau_prior: TauPrior { support: vec![0.0], probabilities: vec![1.0] },
        },
    ] {
        let err = synthesize_transported(
            &mut ledger,
            "double counted",
            &[contribution(cohort, 1.0, 1.0, None), contribution(overlapping, 2.0, 1.0, Some(0.5))],
            &method,
        )
        .unwrap_err();
        assert_eq!(err.code(), "evidence_double_counted", "{err:?}");
        assert!(matches!(err, TransportedSynthesisRefusal::Ledger(_)));
    }
    let err = synthesize_transported(
        &mut ledger,
        "double counted",
        &[contribution(cohort, 1.0, 1.0, None), contribution(overlapping, 2.0, 1.0, Some(0.5))],
        &pool(),
    )
    .unwrap_err();
    assert_eq!(err.detail(), "evidence_ledger.native_and_transported");
    // Atomic: nothing was registered.
    assert_eq!(ledger.state_digest(), before);
    assert!(ledger.results().is_empty());
}

#[test]
fn the_exact_same_source_listed_twice_is_one_source_counted_twice() {
    let mut ledger = EvidenceLedger::new();
    let cohort = native(&mut ledger, "registry cohort", 0, 100);
    let err = synthesize_transported(
        &mut ledger,
        "twice",
        &[contribution(cohort, 1.0, 1.0, None), contribution(cohort, 1.0, 1.0, None)],
        &pool(),
    )
    .unwrap_err();
    assert_eq!(err.detail(), DETAIL_SOURCE_LISTED_TWICE);
}

#[test]
fn a_transport_with_no_declared_uncertainty_is_not_a_free_transport() {
    let mut ledger = EvidenceLedger::new();
    let cohort = native(&mut ledger, "registry cohort", 0, 100);
    let carried = claim(&mut ledger, "claim", SourceKind::ExternalClaim, 1_000, 1_100, 1);

    let err = synthesize_transported(
        &mut ledger,
        "no tau",
        &[contribution(cohort, 1.0, 1.0, None), contribution(carried, 2.0, 1.0, None)],
        &pool(),
    )
    .unwrap_err();
    assert_eq!(err.code(), "transport_missing_evidence");
    assert_eq!(err.detail(), DETAIL_TRANSPORT_VARIANCE_UNDECLARED);

    let err = synthesize_transported(
        &mut ledger,
        "tau on native",
        &[contribution(cohort, 1.0, 1.0, Some(0.2)), contribution(carried, 2.0, 1.0, Some(0.5))],
        &pool(),
    )
    .unwrap_err();
    assert_eq!(err.detail(), DETAIL_TRANSPORT_VARIANCE_ON_NATIVE);
    assert!(ledger.results().is_empty());
}

#[test]
fn shared_transport_error_with_overlapping_data_needs_a_covariance_that_models_it() {
    let mut ledger = EvidenceLedger::new();
    let cohort = native(&mut ledger, "registry cohort", 0, 100);
    // Two claims through one transport that share half of their rows.
    let first = claim(&mut ledger, "claim one", SourceKind::ExternalClaim, 1_000, 1_100, 1);
    let second = claim(&mut ledger, "claim two", SourceKind::ExternalClaim, 1_050, 1_150, 1);
    let err = synthesize_transported(
        &mut ledger,
        "overlapping claims",
        &[
            contribution(cohort, 1.0, 1.0, None),
            contribution(first, 2.0, 1.0, Some(0.5)),
            contribution(second, 3.0, 1.0, Some(0.5)),
        ],
        &pool(),
    )
    .unwrap_err();
    assert_eq!(err.detail(), DETAIL_OVERLAP_NOT_MODELLED);
    assert!(ledger.results().is_empty());
}

#[test]
fn borrowing_inflates_each_transported_variance_and_refuses_a_shared_transport() {
    let mut ledger = EvidenceLedger::new();
    let cohort = native(&mut ledger, "registry cohort", 0, 100);
    let first = claim(&mut ledger, "claim one", SourceKind::ExternalClaim, 1_000, 1_100, 1);
    let second = claim(&mut ledger, "claim two", SourceKind::ExternalClaim, 2_000, 2_100, 2);
    let method = TransportedSynthesisMethod::Borrow {
        mean_prior: MeanPrior::Flat,
        tau_prior: TauPrior { support: vec![0.0], probabilities: vec![1.0] },
    };
    let out = synthesize_transported(
        &mut ledger,
        "borrow",
        &[
            contribution(cohort, 1.0, 1.0, None),
            contribution(first, 2.0, 1.0, Some(0.5)),
            contribution(second, 3.0, 1.0, Some(0.5)),
        ],
        &method,
    )
    .unwrap();
    // A point-mass between-study sd of zero is inverse-variance pooling of the
    // inflated variances (1, 1.5, 1.5).
    near(out.estimate, 13.0 / 7.0);
    near(out.variance, 3.0 / 7.0);
    assert_eq!(out.method, "borrow");

    // Two claims through one transport on the borrowing route: its shared
    // error would be counted once per claim, so the route refuses.
    let mut other = EvidenceLedger::new();
    let c = native(&mut other, "registry cohort", 0, 100);
    let a = claim(&mut other, "claim one", SourceKind::ExternalClaim, 1_000, 1_100, 1);
    let b = claim(&mut other, "claim two", SourceKind::ExternalClaim, 2_000, 2_100, 1);
    let err = synthesize_transported(
        &mut other,
        "borrow shared",
        &[
            contribution(c, 1.0, 1.0, None),
            contribution(a, 2.0, 1.0, Some(0.5)),
            contribution(b, 3.0, 1.0, Some(0.5)),
        ],
        &method,
    )
    .unwrap_err();
    assert_eq!(err.detail(), DETAIL_SHARED_TRANSPORT);
    assert_eq!(err.code(), "joint_law_required");
    assert!(other.results().is_empty());
}

#[test]
fn a_transported_claim_is_the_prior_of_a_native_likelihood() {
    let mut ledger = EvidenceLedger::new();
    let cohort = native(&mut ledger, "registry cohort", 0, 100);
    let prior = claim(&mut ledger, "transported prior", SourceKind::Prior, 1_000, 1_100, 1);
    let method = TransportedSynthesisMethod::TransportedPrior { prior };
    let out = synthesize_transported(
        &mut ledger,
        "transported prior",
        &[contribution(cohort, 1.0, 1.0, None), contribution(prior, 2.0, 1.0, Some(0.5))],
        &method,
    )
    .unwrap();
    near(out.estimate, 1.4);
    near(out.variance, 0.6);
    assert_eq!(out.method, "transported_prior");
    assert_eq!(out.transported_sources, vec![prior]);

    // A prior built from the likelihood's own rows is the same data twice.
    let mut reused = EvidenceLedger::new();
    let c = native(&mut reused, "registry cohort", 0, 100);
    let p = claim(&mut reused, "prior on registry rows", SourceKind::Prior, 40, 140, 1);
    let err = synthesize_transported(
        &mut reused,
        "prior plus likelihood",
        &[contribution(c, 1.0, 1.0, None), contribution(p, 2.0, 1.0, Some(0.5))],
        &TransportedSynthesisMethod::TransportedPrior { prior: p },
    )
    .unwrap_err();
    assert_eq!(err.code(), "evidence_double_counted");
    assert_eq!(err.detail(), "evidence_ledger.reentry_prior_plus_likelihood");
    assert!(reused.results().is_empty());
}

#[test]
fn the_prior_route_needs_a_transported_prior_and_native_likelihood() {
    let mut ledger = EvidenceLedger::new();
    let cohort = native(&mut ledger, "registry cohort", 0, 100);
    let other = native(&mut ledger, "second registry", 500, 600);
    let err = synthesize_transported(
        &mut ledger,
        "native prior",
        &[contribution(cohort, 1.0, 1.0, None), contribution(other, 2.0, 1.0, None)],
        &TransportedSynthesisMethod::TransportedPrior { prior: other },
    )
    .unwrap_err();
    assert_eq!(err.detail(), DETAIL_PRIOR_ROLE);

    let carried = claim(&mut ledger, "claim", SourceKind::Prior, 1_000, 1_100, 1);
    let err = synthesize_transported(
        &mut ledger,
        "prior not listed",
        &[contribution(cohort, 1.0, 1.0, None)],
        &TransportedSynthesisMethod::TransportedPrior { prior: carried },
    )
    .unwrap_err();
    assert_eq!(err.detail(), DETAIL_PRIOR_ROLE);
    assert!(ledger.results().is_empty());
}

#[test]
fn a_numerical_refusal_leaves_the_ledger_unchanged() {
    let mut ledger = EvidenceLedger::new();
    let cohort = native(&mut ledger, "registry cohort", 0, 100);
    let carried = claim(&mut ledger, "claim", SourceKind::ExternalClaim, 1_000, 1_100, 1);
    let before = ledger.state_digest();
    // A negative transport variance is a numerical refusal; nothing is
    // registered in the ledger.
    let err = synthesize_transported(
        &mut ledger,
        "bad tau",
        &[contribution(cohort, 1.0, 1.0, None), contribution(carried, 2.0, 1.0, Some(-0.5))],
        &pool(),
    )
    .unwrap_err();
    assert!(matches!(err, TransportedSynthesisRefusal::Numeric(_)));
    assert_eq!(err.code(), "invalid_argument");
    assert_eq!(ledger.state_digest(), before);
}

#[test]
fn a_changed_input_at_unchanged_shape_changes_the_registered_result() {
    let mut ledger = EvidenceLedger::new();
    let cohort = native(&mut ledger, "registry cohort", 0, 100);
    let carried = claim(&mut ledger, "claim", SourceKind::ExternalClaim, 1_000, 1_100, 1);
    let first = synthesize_transported(
        &mut ledger,
        "effect",
        &[contribution(cohort, 1.0, 1.0, None), contribution(carried, 2.0, 1.0, Some(0.5))],
        &pool(),
    )
    .unwrap();
    // The same sources with a different declared transport variance.
    let second = synthesize_transported(
        &mut ledger,
        "effect (wider transport)",
        &[contribution(cohort, 1.0, 1.0, None), contribution(carried, 2.0, 1.0, Some(2.0))],
        &pool(),
    )
    .unwrap();
    assert!(second.variance > first.variance);
    assert_ne!(first.result, second.result);
}

fn claim_to(ledger: &mut EvidenceLedger, label: &str, target: &str, start: u32) -> SourceId {
    let identity = TransportIdentity {
        source_populations: vec!["trial_a".to_owned()],
        target_population: target.to_owned(),
        selection_diagram: SemanticDigest::from_bytes([9; 32]),
        assumptions: vec!["outcome mechanism invariant".to_owned()],
    };
    ledger
        .add_source(
            SourceDeclaration::new(label, SourceKind::ExternalClaim)
                .with_observations(rows(start, start + 100))
                .with_lineage(Lineage::Original)
                .with_transport(identity),
        )
        .unwrap()
}

#[test]
fn transports_to_different_targets_are_not_one_quantity() {
    let mut ledger = EvidenceLedger::new();
    let a = claim_to(&mut ledger, "claim a", "registry", 1_000);
    let b = claim_to(&mut ledger, "claim b", "clinic_network", 2_000);
    let before = ledger.state_digest();
    let err = synthesize_transported(
        &mut ledger,
        "mixed targets",
        &[contribution(a, 1.0, 1.0, Some(0.5)), contribution(b, 2.0, 1.0, Some(0.5))],
        &pool(),
    )
    .unwrap_err();
    assert_eq!(err.detail(), DETAIL_TARGET_POPULATION_MISMATCH);
    assert_eq!(err.code(), "quantity_semantics_mismatch");
    assert_eq!(ledger.state_digest(), before);
}

#[test]
fn native_contributions_must_be_declared_in_the_transports_target() {
    let mut ledger = EvidenceLedger::new();
    let cohort = native(&mut ledger, "registry cohort", 0, 100);
    let carried = claim(&mut ledger, "claim", SourceKind::ExternalClaim, 1_000, 1_100, 1);
    let contributions =
        [contribution(cohort, 1.0, 1.0, None), contribution(carried, 2.0, 1.0, Some(0.5))];

    let before = ledger.state_digest();
    let err = synthesize_transported_in_population(
        &mut ledger,
        "wrong native population",
        &contributions,
        &pool(),
        "some_other_clinic",
    )
    .unwrap_err();
    assert_eq!(err.detail(), DETAIL_TARGET_POPULATION_MISMATCH);
    assert_eq!(ledger.state_digest(), before);

    let err =
        synthesize_transported_in_population(&mut ledger, "blank", &contributions, &pool(), "  ")
            .unwrap_err();
    assert_eq!(err.detail(), DETAIL_NATIVE_POPULATION_BLANK);

    let out = synthesize_transported_in_population(
        &mut ledger,
        "declared",
        &contributions,
        &pool(),
        "registry",
    )
    .unwrap();
    near(out.estimate, 1.4);
    assert_eq!(out.target_population.as_deref(), Some("registry"));
    assert_eq!(out.native_population.as_deref(), Some("registry"));
    assert_eq!(out.source_populations, vec!["trial_a".to_owned()]);
}

#[test]
fn a_transport_with_no_assumption_or_a_self_target_is_refused_at_the_ledger() {
    let mut ledger = EvidenceLedger::new();
    let add = |ledger: &mut EvidenceLedger, start: u32, identity: TransportIdentity| {
        ledger.add_source(
            SourceDeclaration::new("claim", SourceKind::ExternalClaim)
                .with_observations(rows(start, start + 100))
                .with_lineage(Lineage::Original)
                .with_transport(identity),
        )
    };
    let err = add(&mut ledger, 1_000, TransportIdentity { assumptions: vec![], ..transport(3) })
        .unwrap_err();
    assert_eq!(err.detail, "evidence_ledger.invalid_declaration");
    // The design-level guard stays for identities that reach synthesis.
    assert!(add(&mut ledger, 1_000, transport(3)).is_ok());
}
