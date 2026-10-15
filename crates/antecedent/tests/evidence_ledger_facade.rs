//! Facade integration of the evidence ledger: overlap before numbers,
//! point-only syntheses against hand derivations, revision with downstream
//! revalidation and stale-use refusal, an atomic refresh, and an artifact a
//! consumer replays.
//!
//! Hand oracles:
//! * y = (1, 3), variances (1, 4): estimate 7/5, variance 4/5, Q 4/5, weights (4/5, 1/5).
//! * y = (1, 3), covariance [[1, 1], [1, 4]]: weights (1, 0), estimate 1, variance 1.
//! * y = (0, 2, 6), v = (1, 2, 4): Q = 52/7, tau^2 = 38/7, `DerSimonian`-Laird
//!   estimate 3330/1457, variance 25740/10199, I^2 = 19/26.
//! * tau = 1 with a flat mean prior on the same studies: mean 56/31, variance
//!   30/31; with a N(0, 1) prior on the mean: 56/61 and 30/61.
//! * y = (1, 3), se = (1, 2): the pooled estimate over a common correlation
//!   rho is (7 - 8 rho)/(5 - 4 rho) (decreasing: 7/5 at 0, -1/7 at 0.9) and
//!   its variance (4 - 4 rho^2)/(5 - 4 rho), which peaks at rho = 1/2 with
//!   value 1 and is 0.76/1.4 at 0.9.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::ops::Range;

use antecedent::analysis::evidence_ledger::{
    ClaimClass, Lineage, ObservationId, Overlap, RelationKind, ResultId, Revalidation,
    SourceDeclaration, SourceId, SourceKind, TransportIdentity,
};
use antecedent::{
    BetweenStudy, BorrowingRequest, DependenceRangeRequest, EvidenceLedgerRefusal,
    EvidenceLedgerSession, PoolingDeclaration, PoolingRequest, SourceChange, SourceRevision,
    StudyBuilder, StudyEstimate, StudySummary, SynthesisValue, consume_evidence_ledger_artifact,
};
use antecedent_core::{ExecutionContext, SemanticDigest};
use antecedent_io::evidence_ledger_artifact::{
    EvidenceLedgerArtifact, EvidenceLedgerArtifactError, SynthesisMethod, SynthesisRecord,
};
use antecedent_stats::synthesis::{MeanPrior, TauPrior};

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

fn close(actual: f64, expected: f64, what: &str) {
    assert!((actual - expected).abs() <= 1e-12, "{what}: expected {expected}, got {actual}");
}

fn rows(range: Range<u32>) -> Vec<ObservationId> {
    range.map(|i| ObservationId::from_key("cohort", &i.to_le_bytes())).collect()
}

fn data(label: &str, range: Range<u32>) -> SourceDeclaration {
    SourceDeclaration::new(label, SourceKind::ObservationSet).with_observations(rows(range))
}

fn summary(source: SourceId, estimate: f64, variance: f64) -> StudySummary {
    StudySummary { source, estimate, variance }
}

fn pooling(label: &str, summaries: Vec<StudySummary>) -> PoolingRequest {
    PoolingRequest { label: label.to_owned(), summaries, replaces: None }
}

fn expect(refusal: &EvidenceLedgerRefusal, code: &str, detail: &str) {
    assert_eq!(refusal.code, code, "{refusal}");
    assert_eq!(refusal.detail, detail, "{refusal}");
}

struct Scenario {
    session: EvidenceLedgerSession,
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

/// Three disjoint studies, pools over pairs and all three, a citation of one
/// pool, and a fitted model built from study 1 pooled with study 3.
fn scenario() -> Scenario {
    let mut session = StudyBuilder::evidence_ledger();
    let s1 = session.add_source(data("s1", 0..10)).unwrap();
    let s2 = session.add_source(data("s2", 10..20)).unwrap();
    let s3 = session.add_source(data("s3", 20..30)).unwrap();
    let c = ctx();
    let r_s1_s2 = session
        .pool_summaries(
            pooling("s1_s2", vec![summary(s1, 1.0, 1.0), summary(s2, 3.0, 4.0)]),
            PoolingDeclaration::Independent,
            &c,
        )
        .unwrap()
        .result;
    let r_s2_s3 = session
        .pool_summaries(
            pooling("s2_s3", vec![summary(s2, 3.0, 4.0), summary(s3, 2.0, 1.0)]),
            PoolingDeclaration::Independent,
            &c,
        )
        .unwrap()
        .result;
    let r_all = session
        .pool_summaries(
            pooling(
                "all",
                vec![summary(s1, 0.0, 1.0), summary(s2, 2.0, 2.0), summary(s3, 6.0, 4.0)],
            ),
            PoolingDeclaration::Independent,
            &c,
        )
        .unwrap()
        .result;
    let r_cite = session.reuse_result("cite_s1_s2", r_s1_s2).unwrap();
    let model = session
        .add_source(
            SourceDeclaration::new("model_on_s1", SourceKind::FittedModel)
                .with_lineage(Lineage::DerivedFromUsedData { inputs: vec![s1] }),
        )
        .unwrap();
    let r_model = session
        .pool_summaries(
            pooling("model_s3", vec![summary(model, 0.5, 1.0), summary(s3, 2.0, 1.0)]),
            PoolingDeclaration::Independent,
            &c,
        )
        .unwrap()
        .result;
    Scenario { session, s1, s2, s3, model, r_s1_s2, r_s2_s3, r_all, r_cite, r_model }
}

// ---------------------------------------------------------------------------
// Overlap and pooling
// ---------------------------------------------------------------------------

#[test]
fn a_pooled_value_matches_the_hand_calculation_and_mass_counts_each_observation_once() {
    let mut session = StudyBuilder::evidence_ledger();
    let s1 = session.add_source(data("study_1", 0..40)).unwrap();
    let s2 = session.add_source(data("study_2", 40..90)).unwrap();
    let outcome = session
        .pool_summaries(
            pooling("pool", vec![summary(s1, 1.0, 1.0), summary(s2, 3.0, 4.0)]),
            PoolingDeclaration::Independent,
            &ctx(),
        )
        .unwrap();
    let SynthesisValue::FixedEffect(fit) = &outcome.value else {
        panic!("independent pooling is a fixed-effect pool");
    };
    close(fit.estimate, 1.4, "estimate");
    close(fit.variance, 0.8, "variance");
    close(fit.heterogeneity_q, 0.8, "Q");
    close(fit.weights[0], 0.8, "weight 0");
    close(fit.weights[1], 0.2, "weight 1");
    assert_eq!(outcome.value.claim_label(), "point_only");
    assert_eq!(outcome.claim_class, ClaimClass::PointOnly);
    assert_eq!(outcome.mass.sources, 2);
    assert_eq!(outcome.mass.declared_slots, 90);
    assert_eq!(outcome.mass.distinct_observations, 90);
    assert_eq!(outcome.mass.repeated_slots, 0);
    assert!(outcome.mass.unknown_sources.is_empty());
    assert_eq!(session.synthesis(outcome.result).unwrap(), Some(&outcome.value));

    // The account of two disjoint studies, and of an overlapping pair.
    let account = session.account_observations(&[s1, s2]).unwrap();
    assert_eq!(account.pairs[0].overlap, Overlap::Independent);
    assert_eq!(account.pairs[0].relation, RelationKind::Independent);
    let s4 = session.add_source(data("study_4", 80..130)).unwrap();
    let account = session.account_observations(&[s2, s4]).unwrap();
    assert_eq!(account.pairs[0].overlap, Overlap::SharedObservations { shared: 10 });
    // 50 + 50 slots over 90 distinct observations: 10 would be double counted.
    assert_eq!(account.mass.declared_slots, 100);
    assert_eq!(account.mass.distinct_observations, 90);
    assert_eq!(account.mass.repeated_slots, 10);
}

#[test]
fn overlap_refusals_precede_the_numbers_and_leave_the_session_unchanged() {
    let mut session = StudyBuilder::evidence_ledger();
    let s3 = session.add_source(data("s3", 90..120)).unwrap();
    let s4 = session.add_source(data("s4", 110..150)).unwrap();
    let vendor =
        session.add_source(SourceDeclaration::new("vendor", SourceKind::ExternalClaim)).unwrap();
    let head = session.head_hex();
    let state = session.state_digest_hex();

    // Shared rows declared independent: the numbers are not even looked at
    // (the variances here would be refused).
    let refusal = session
        .pool_summaries(
            pooling("p", vec![summary(s3, 1.0, -1.0), summary(s4, 3.0, -1.0)]),
            PoolingDeclaration::Independent,
            &ctx(),
        )
        .unwrap_err();
    expect(&refusal, "evidence_double_counted", "evidence_ledger.shared_observations");
    // Undeclared observations are unknown, never independent.
    let refusal = session
        .pool_summaries(
            pooling("p", vec![summary(s3, 1.0, 1.0), summary(vendor, 3.0, 4.0)]),
            PoolingDeclaration::Independent,
            &ctx(),
        )
        .unwrap_err();
    expect(&refusal, "evidence_overlap_unknown", "evidence_ledger.overlap_unknown");
    // A diagonal covariance over shared rows is the same double count.
    let refusal = session
        .pool_summaries(
            pooling("p", vec![summary(s3, 1.0, 1.0), summary(s4, 3.0, 4.0)]),
            PoolingDeclaration::DeclaredCovariance { covariance: vec![1.0, 0.0, 0.0, 4.0] },
            &ctx(),
        )
        .unwrap_err();
    expect(&refusal, "evidence_double_counted", "evidence_ledger.diagonal_covariance_shared_rows");
    // One source listed twice.
    let refusal = session
        .pool_summaries(
            pooling("p", vec![summary(s3, 1.0, 1.0), summary(s3, 2.0, 1.0)]),
            PoolingDeclaration::Independent,
            &ctx(),
        )
        .unwrap_err();
    expect(&refusal, "evidence_double_counted", "evidence_ledger.repeated_source");
    // None of the refusals changed the session.
    assert_eq!(session.head_hex(), head);
    assert_eq!(session.state_digest_hex(), state);
    assert!(session.ledger().results().is_empty());

    // A duplicate re-entering through lineage is refused as a duplicate.
    let copy = session
        .add_source(
            SourceDeclaration::new("s3_again", SourceKind::ObservationSet)
                .with_lineage(Lineage::Duplicate { of: s3 }),
        )
        .unwrap();
    let refusal = session
        .pool_summaries(
            pooling("p", vec![summary(s3, 1.0, 1.0), summary(copy, 1.0, 1.0)]),
            PoolingDeclaration::Independent,
            &ctx(),
        )
        .unwrap_err();
    expect(&refusal, "evidence_double_counted", "evidence_ledger.reentry_duplicate");

    // The numbers refuse after the overlap check passes, and the staged
    // registration is discarded with them.
    let s5 = session.add_source(data("s5", 500..520)).unwrap();
    let before = session.state_digest_hex();
    let before_head = session.head_hex();
    let refusal = session
        .pool_summaries(
            pooling("p", vec![summary(s3, 1.0, 1.0), summary(s5, 3.0, -4.0)]),
            PoolingDeclaration::Independent,
            &ctx(),
        )
        .unwrap_err();
    expect(&refusal, "invalid_argument", "evidence_ledger.invalid_declaration");
    assert_eq!(session.state_digest_hex(), before);
    assert_eq!(session.head_hex(), before_head);
    assert!(session.ledger().results().is_empty());
}

#[test]
fn a_declared_covariance_with_cross_terms_licenses_the_shared_pair() {
    let mut session = StudyBuilder::evidence_ledger();
    let s3 = session.add_source(data("s3", 90..120)).unwrap();
    let s4 = session.add_source(data("s4", 110..150)).unwrap();
    // y = (1, 3), standard errors (1, 2), correlation 1/2: S = [[1, 1], [1, 4]].
    let outcome = session
        .pool_summaries(
            pooling("shared", vec![summary(s3, 1.0, 1.0), summary(s4, 3.0, 4.0)]),
            PoolingDeclaration::DeclaredCovariance { covariance: vec![1.0, 1.0, 1.0, 4.0] },
            &ctx(),
        )
        .unwrap();
    let SynthesisValue::FixedEffect(fit) = &outcome.value else {
        panic!("GLS is a fixed-effect pool")
    };
    close(fit.estimate, 1.0, "estimate");
    close(fit.variance, 1.0, "variance");
    close(fit.weights[0], 1.0, "weight 0");
    close(fit.weights[1], 0.0, "weight 1");
    // The shared rows are counted once: 30 + 40 slots, 60 distinct.
    assert_eq!(outcome.mass.declared_slots, 70);
    assert_eq!(outcome.mass.distinct_observations, 60);
    assert_eq!(outcome.mass.repeated_slots, 10);
    assert!(session.result(outcome.result).unwrap().dependence_declared);

    // The covariance must be a covariance: eigenvalues 3 and -1 refuse, and
    // the wrong shape is refused before the ledger sees it.
    let s5 = session.add_source(data("s5", 500..520)).unwrap();
    let s6 = session.add_source(data("s6", 600..620)).unwrap();
    let refusal = session
        .pool_summaries(
            pooling("bad", vec![summary(s5, 1.0, 1.0), summary(s6, 2.0, 1.0)]),
            PoolingDeclaration::DeclaredCovariance { covariance: vec![1.0, 2.0, 2.0, 1.0] },
            &ctx(),
        )
        .unwrap_err();
    expect(&refusal, "invalid_argument", "evidence_ledger.invalid_declaration");
    let refusal = session
        .pool_summaries(
            pooling("short", vec![summary(s5, 1.0, 1.0), summary(s6, 2.0, 1.0)]),
            PoolingDeclaration::DeclaredCovariance { covariance: vec![1.0, 0.0, 4.0] },
            &ctx(),
        )
        .unwrap_err();
    expect(&refusal, "invalid_argument", "evidence_ledger.invalid_declaration");
    assert_eq!(session.ledger().results().len(), 1);
}

#[test]
fn pooling_is_invariant_to_the_order_of_the_studies() {
    let build = |order: &[usize]| {
        let mut session = StudyBuilder::evidence_ledger();
        let ids = [
            session.add_source(data("a", 0..10)).unwrap(),
            session.add_source(data("b", 10..20)).unwrap(),
            session.add_source(data("c", 20..30)).unwrap(),
        ];
        let y = [0.0, 2.0, 6.0];
        let v = [1.0, 2.0, 4.0];
        let summaries: Vec<StudySummary> =
            order.iter().map(|&i| summary(ids[i], y[i], v[i])).collect();
        session
            .pool_summaries(pooling("p", summaries), PoolingDeclaration::Independent, &ctx())
            .unwrap()
    };
    let a = build(&[0, 1, 2]);
    let b = build(&[2, 0, 1]);
    // The result identity is a function of the source set, not its order.
    assert_eq!(a.result, b.result);
    let (SynthesisValue::FixedEffect(x), SynthesisValue::FixedEffect(y)) = (&a.value, &b.value)
    else {
        panic!("fixed effect");
    };
    assert!((x.estimate - y.estimate).abs() < 1e-12);
    assert!((x.variance - y.variance).abs() < 1e-12);
}

// ---------------------------------------------------------------------------
// Random effects, borrowing, dependence range
// ---------------------------------------------------------------------------

#[test]
fn random_effects_match_the_dersimonian_laird_derivation() {
    let mut session = StudyBuilder::evidence_ledger();
    let ids: Vec<SourceId> = (0..3)
        .map(|i| session.add_source(data(&format!("s{i}"), i * 10..(i + 1) * 10)).unwrap())
        .collect();
    let y = [0.0, 2.0, 6.0];
    let v = [1.0, 2.0, 4.0];
    let summaries: Vec<StudySummary> = (0..3).map(|i| summary(ids[i], y[i], v[i])).collect();
    let outcome = session
        .random_effects(pooling("re", summaries.clone()), BetweenStudy::DerSimonianLaird, &ctx())
        .unwrap();
    let SynthesisValue::RandomEffects(fit) = &outcome.value else { panic!("random effects") };
    close(fit.q, 52.0 / 7.0, "Q");
    assert_eq!(fit.degrees_of_freedom, 2);
    close(fit.tau_squared, 38.0 / 7.0, "tau^2");
    close(fit.estimate, 3330.0 / 1457.0, "estimate");
    close(fit.variance, 25740.0 / 10199.0, "variance");
    close(fit.i_squared, 19.0 / 26.0, "I^2");

    // REML is a different, registered estimator of the same quantity.
    let reml = session
        .random_effects(
            PoolingRequest { label: "re_reml".into(), summaries, replaces: None },
            BetweenStudy::RestrictedMaximumLikelihood,
            &ctx(),
        )
        .unwrap();
    let SynthesisValue::RandomEffects(r) = &reml.value else { panic!("random effects") };
    assert!(r.tau_squared >= 0.0 && r.estimate > 0.0 && r.estimate < 6.0);
    close(r.q, 52.0 / 7.0, "Q is method independent");
    // Fewer than two studies cannot estimate between-study variance.
    let one = session.add_source(data("lonely", 100..110)).unwrap();
    let refusal = session
        .random_effects(
            pooling("one", vec![summary(one, 1.0, 1.0)]),
            BetweenStudy::DerSimonianLaird,
            &ctx(),
        )
        .unwrap_err();
    assert_eq!(refusal.code, "invalid_argument");
}

fn three_studies(session: &mut EvidenceLedgerSession) -> Vec<StudySummary> {
    let ids: Vec<SourceId> = (0..3)
        .map(|i| session.add_source(data(&format!("study_{i}"), i * 10..(i + 1) * 10)).unwrap())
        .collect();
    [0.0, 2.0, 6.0]
        .iter()
        .zip([1.0, 2.0, 4.0])
        .zip(&ids)
        .map(|((&y, v), &id)| summary(id, y, v))
        .collect()
}

#[test]
fn hierarchical_borrowing_matches_the_closed_form_at_a_single_tau() {
    let mut session = StudyBuilder::evidence_ledger();
    let studies = three_studies(&mut session);
    let outcome = session
        .hierarchical_borrowing(
            BorrowingRequest {
                label: "flat".into(),
                priors: vec![],
                likelihood: studies.clone(),
                mean_prior: MeanPrior::Flat,
                tau_prior: TauPrior { support: vec![1.0], probabilities: vec![1.0] },
                replaces: None,
            },
            &ctx(),
        )
        .unwrap();
    let SynthesisValue::Hierarchical(fit) = &outcome.value else { panic!("hierarchical") };
    close(fit.mean, 56.0 / 31.0, "flat mean");
    close(fit.variance, 30.0 / 31.0, "flat variance");
    assert_eq!(outcome.value.claim_label(), "point_only");

    let normal = session
        .hierarchical_borrowing(
            BorrowingRequest {
                label: "normal".into(),
                priors: vec![],
                likelihood: studies,
                mean_prior: MeanPrior::Normal { mean: 0.0, variance: 1.0 },
                tau_prior: TauPrior { support: vec![1.0], probabilities: vec![1.0] },
                replaces: None,
            },
            &ctx(),
        )
        .unwrap();
    let SynthesisValue::Hierarchical(fit) = &normal.value else { panic!("hierarchical") };
    close(fit.mean, 56.0 / 61.0, "normal-prior mean");
    close(fit.variance, 30.0 / 61.0, "normal-prior variance");
}

#[test]
fn a_prior_built_from_the_likelihoods_data_is_double_counting() {
    let mut session = StudyBuilder::evidence_ledger();
    let studies = three_studies(&mut session);
    // A power prior made from rows study 1 already contributed.
    let power = session
        .add_source(
            SourceDeclaration::new("power_prior", SourceKind::Prior).with_observations(rows(0..5)),
        )
        .unwrap();
    let request = |priors: Vec<SourceId>| BorrowingRequest {
        label: "borrow".into(),
        priors,
        likelihood: studies.clone(),
        mean_prior: MeanPrior::Normal { mean: 0.0, variance: 1.0 },
        tau_prior: TauPrior { support: vec![0.5, 1.0], probabilities: vec![0.5, 0.5] },
        replaces: None,
    };
    let refusal = session.hierarchical_borrowing(request(vec![power]), &ctx()).unwrap_err();
    assert_eq!(refusal.code, "evidence_double_counted");
    assert!(session.ledger().results().is_empty());

    // An elicited prior has an empty observation set and overlaps nothing.
    let elicited = session
        .add_source(SourceDeclaration::new("elicited", SourceKind::Prior).with_observations(vec![]))
        .unwrap();
    let outcome = session.hierarchical_borrowing(request(vec![elicited]), &ctx()).unwrap();
    let SynthesisValue::Hierarchical(fit) = &outcome.value else { panic!("hierarchical") };
    assert!((fit.tau_posterior.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    // The numbers are the session's; the artifact format carries one summary
    // per source, so this result's numbers are listed as not carried.
    let export = session.export("borrowing").unwrap();
    assert!(export.carried.is_empty());
    assert_eq!(export.not_carried.len(), 1);
    assert_eq!(export.not_carried[0].0, outcome.result);
}

#[test]
fn the_dependence_range_is_an_exact_labelled_envelope() {
    let mut session = StudyBuilder::evidence_ledger();
    let s3 = session.add_source(data("s3", 0..30)).unwrap();
    let vendor =
        session.add_source(SourceDeclaration::new("vendor", SourceKind::ExternalClaim)).unwrap();
    let request = |lo: f64, hi: f64| DependenceRangeRequest {
        label: "range".into(),
        studies: vec![
            StudyEstimate { source: s3, estimate: 1.0, standard_error: 1.0 },
            StudyEstimate { source: vendor, estimate: 3.0, standard_error: 2.0 },
        ],
        rho_lo: lo,
        rho_hi: hi,
        replaces: None,
    };
    let outcome = session.dependence_range(request(0.0, 0.9), &ctx()).unwrap();
    assert_eq!(outcome.claim_class, ClaimClass::AssumptionRange);
    assert_eq!(outcome.value.claim_label(), "assumption_range");
    let SynthesisValue::DependenceEnvelope(env) = &outcome.value else { panic!("envelope") };
    assert!((env.estimate_max - 1.4).abs() < 1e-9, "{}", env.estimate_max);
    assert!(env.rho_at_estimate_max.abs() < 1e-9);
    assert!((env.estimate_min + 1.0 / 7.0).abs() < 1e-9, "{}", env.estimate_min);
    assert!((env.rho_at_estimate_min - 0.9).abs() < 1e-9);
    // The variance peaks inside the range, at rho = s_min / s_max = 1/2.
    assert!((env.variance_max - 1.0).abs() < 1e-9, "{}", env.variance_max);
    assert!((env.rho_at_variance_max - 0.5).abs() < 1e-9, "{}", env.rho_at_variance_max);
    assert!((env.variance_min - 0.76 / 1.4).abs() < 1e-9, "{}", env.variance_min);
    // The unknown source stays unknown in the accounting.
    assert!(session.result(outcome.result).unwrap().mass.unknown_sources.contains(&vendor));

    // A range that is not strictly inside the definite interval refuses.
    for (lo, hi) in [(0.3, 0.3), (0.0, 1.0), (-1.5, 0.2)] {
        let refusal = session.dependence_range(request(lo, hi), &ctx()).unwrap_err();
        assert_eq!(refusal.code, "invalid_argument", "({lo}, {hi})");
    }
}

// ---------------------------------------------------------------------------
// Other operations
// ---------------------------------------------------------------------------

fn transport_identity(target: &str) -> TransportIdentity {
    TransportIdentity {
        source_populations: vec!["trial_a".into(), "trial_b".into()],
        target_population: target.into(),
        selection_diagram: SemanticDigest::from_bytes([7; 32]),
        assumptions: vec!["s-admissible adjustment set".into()],
    }
}

#[test]
fn a_transported_claim_is_never_native_target_data() {
    let mut session = StudyBuilder::evidence_ledger();
    let native = session.add_source(data("target_cohort", 0..30)).unwrap();
    let transported = session
        .add_source(
            SourceDeclaration::new("effect_from_trials", SourceKind::ExternalClaim)
                .with_observations(rows(100..160))
                .with_transport(transport_identity("target")),
        )
        .unwrap();
    let result =
        session.transport_claim("transport", vec![transported], vec![native], None).unwrap();
    assert_eq!(session.result(result).unwrap().transport_identities.len(), 1);
    // Accounting-only results carry no numbers.
    assert_eq!(session.synthesis(result).unwrap(), None);

    // The same source native and transported is a category error.
    let refusal =
        session.transport_claim("bad", vec![transported], vec![transported], None).unwrap_err();
    expect(&refusal, "evidence_double_counted", "evidence_ledger.native_and_transported");
    // A claim without its transport identity cannot be transported.
    let bare = session.add_source(data("bare", 300..310)).unwrap();
    let refusal = session.transport_claim("bad", vec![bare], vec![native], None).unwrap_err();
    expect(&refusal, "invalid_argument", "evidence_ledger.transport_identity_missing");
    // Another transported claim is not native evidence.
    let other = session
        .add_source(
            SourceDeclaration::new("other_transported", SourceKind::ExternalClaim)
                .with_observations(rows(400..410))
                .with_transport(transport_identity("target")),
        )
        .unwrap();
    let refusal = session.transport_claim("bad", vec![transported], vec![other], None).unwrap_err();
    expect(&refusal, "invalid_argument", "evidence_ledger.transported_as_native");
}

#[test]
fn citing_a_result_counts_its_observations_once_and_refuses_a_stale_one() {
    let mut session = StudyBuilder::evidence_ledger();
    let a = session.add_source(data("a", 0..30)).unwrap();
    let b = session.add_source(data("b", 30..50)).unwrap();
    let c = session.add_source(data("c", 50..60)).unwrap();
    let pooled = session
        .pool_summaries(
            pooling(
                "pooled",
                vec![summary(a, 1.0, 1.0), summary(b, 2.0, 1.0), summary(c, 3.0, 1.0)],
            ),
            PoolingDeclaration::Independent,
            &ctx(),
        )
        .unwrap();
    assert_eq!(pooled.mass.distinct_observations, 60);
    let first = session.reuse_result("second_analysis", pooled.result).unwrap();
    let second = session.reuse_result("third_analysis", pooled.result).unwrap();
    assert_ne!(first, second);
    for id in [first, second] {
        // Each citation reports the once-counted mass, not a multiple of it.
        assert_eq!(session.result(id).unwrap().mass, pooled.mass);
    }
    // A stale result is recomputed before it is cited.
    session
        .revise_source(SourceRevision::Update { target: c, replacement: data("c_more", 50..70) })
        .unwrap();
    assert_eq!(session.standing(pooled.result), Some(Revalidation::Recompute));
    let refusal = session.reuse_result("cite", pooled.result).unwrap_err();
    expect(&refusal, "evidence_source_withdrawn", "evidence_ledger.result_not_valid");
}

#[test]
fn repeated_observation_use_is_flagged_and_stays_flagged_when_cited() {
    let mut session = StudyBuilder::evidence_ledger();
    let a = session.add_source(data("a", 0..30)).unwrap();
    let repeated = session.repeat_observations("interim_then_final", vec![a], "final").unwrap();
    let record = session.result(repeated).unwrap();
    assert!(record.data_reuse);
    assert_eq!(record.mass.distinct_observations, 30);
    let cited = session.reuse_result("cite", repeated).unwrap();
    assert!(session.result(cited).unwrap().data_reuse);
    // A stage with no name is not a declaration.
    let refusal = session.repeat_observations("nameless", vec![a], "").unwrap_err();
    expect(&refusal, "invalid_argument", "evidence_ledger.operation_arity");
    // Undeclared observations cannot be accounted: their reuse is unknown.
    let vendor =
        session.add_source(SourceDeclaration::new("vendor", SourceKind::ExternalClaim)).unwrap();
    let refusal = session.repeat_observations("vendor_again", vec![vendor], "final").unwrap_err();
    assert_eq!(refusal.code, "evidence_overlap_unknown");
}

#[test]
fn a_pooled_interval_inside_the_measured_scope_is_bound_to_its_record() {
    // Five studies of 100 observations each: inside the measured fixed-effect
    // and random-effects constructions.
    let mut session = StudyBuilder::evidence_ledger();
    let sources: Vec<SourceId> = (0..5)
        .map(|i| session.add_source(data(&format!("study_{i}"), i * 100..(i + 1) * 100)).unwrap())
        .collect();
    let summaries = |shift: f64| -> Vec<StudySummary> {
        sources
            .iter()
            .enumerate()
            .map(|(i, s)| summary(*s, 0.5 + shift * (i as f64 - 2.0), 0.01 + 0.002 * i as f64))
            .collect()
    };
    let c = ctx();
    let fixed = session
        .pool_summaries(pooling("fixed", summaries(0.01)), PoolingDeclaration::Independent, &c)
        .unwrap()
        .result;
    let random = session
        .random_effects(pooling("random", summaries(0.05)), BetweenStudy::DerSimonianLaird, &c)
        .unwrap()
        .result;
    for (result, method) in [(fixed, "wald_z"), (random, "hksj_t")] {
        let interval = session.pooled_interval(result).unwrap();
        assert_eq!(interval.method, method);
        assert!(interval.lower < interval.estimate && interval.estimate < interval.upper);
        assert!((interval.level - 0.95).abs() < 1e-12);
        let slot = &interval.report.scalars[0].calibration;
        assert_eq!(slot.status, "calibrated");
        assert!(slot.record_id.is_some());
    }
    // Three studies are outside the measured study counts: typed refusal.
    let three = session
        .pool_summaries(
            pooling("three", summaries(0.01)[..3].to_vec()),
            PoolingDeclaration::Independent,
            &c,
        )
        .unwrap()
        .result;
    let refusal = session.pooled_interval(three).unwrap_err();
    expect(&refusal, "cell_not_licensed", "evidence_ledger.interval_not_measured");
}

#[test]
fn a_pooled_interval_refuses_without_a_measured_record() {
    let s = scenario();
    // The small fixture has no coverage record of its sample size: the numeric
    // result is refused as unmeasured, never reported without a record.
    let refusal = s.session.pooled_interval(s.r_all).unwrap_err();
    assert_eq!(refusal.code, "cell_not_licensed");
    assert!(
        refusal.detail == "evidence_ledger.interval_not_measured"
            || refusal.detail == "evidence_ledger.interval_not_licensed",
        "{}",
        refusal.detail
    );
    let unknown = ResultId::from_hex(&"ab".repeat(32)).unwrap();
    let refusal = s.session.pooled_interval(unknown).unwrap_err();
    assert_eq!(refusal.code, "invalid_argument");
}

// ---------------------------------------------------------------------------
// Revision, stale use and refresh
// ---------------------------------------------------------------------------

#[test]
fn withdrawing_a_source_classifies_dependents_and_refuses_their_use() {
    let mut s = scenario();
    let report = s
        .session
        .revise_source(SourceRevision::Withdraw {
            target: s.s1,
            reason: "consent withdrawn".into(),
        })
        .unwrap();
    let class = |id: ResultId| {
        report.revalidation.iter().find(|r| r.result == id).expect("classified").class
    };
    assert_eq!(class(s.r_s1_s2), Revalidation::Withdrawn);
    assert_eq!(class(s.r_cite), Revalidation::Withdrawn);
    assert_eq!(class(s.r_s2_s3), Revalidation::StillValid);
    assert_eq!(class(s.r_all), Revalidation::Recompute);
    // The model derived from s1 is lost through lineage.
    assert_eq!(class(s.r_model), Revalidation::Withdrawn);
    assert_eq!(report.retired_source, Some(s.s1));

    // Stale numbers are not served; valid ones are.
    for stale in [s.r_s1_s2, s.r_cite, s.r_all, s.r_model] {
        let refusal = s.session.synthesis(stale).unwrap_err();
        expect(&refusal, "evidence_source_withdrawn", "evidence_ledger.result_not_valid");
    }
    assert!(s.session.synthesis(s.r_s2_s3).unwrap().is_some());
    // New work cannot rest on the withdrawn source, directly or by lineage.
    let refusal = s
        .session
        .pool_summaries(
            pooling("again", vec![summary(s.s1, 1.0, 1.0), summary(s.s3, 2.0, 1.0)]),
            PoolingDeclaration::Independent,
            &ctx(),
        )
        .unwrap_err();
    expect(&refusal, "evidence_source_withdrawn", "evidence_ledger.source_withdrawn");
    let refusal = s
        .session
        .pool_summaries(
            pooling("via_model", vec![summary(s.model, 0.5, 1.0), summary(s.s2, 3.0, 4.0)]),
            PoolingDeclaration::Independent,
            &ctx(),
        )
        .unwrap_err();
    expect(&refusal, "evidence_source_withdrawn", "evidence_ledger.source_withdrawn");
    let refusal = s.session.reuse_result("cite_again", s.r_s1_s2).unwrap_err();
    expect(&refusal, "evidence_source_withdrawn", "evidence_ledger.result_not_valid");

    // The recompute on the remaining studies names the result it replaces and
    // the original is preserved but unusable.
    let recomputed = s
        .session
        .pool_summaries(
            PoolingRequest {
                label: "all_minus_s1".into(),
                summaries: vec![summary(s.s2, 2.0, 2.0), summary(s.s3, 6.0, 4.0)],
                replaces: Some(s.r_all),
            },
            PoolingDeclaration::Independent,
            &ctx(),
        )
        .unwrap();
    assert_eq!(s.session.standing(recomputed.result), Some(Revalidation::StillValid));
    assert_eq!(s.session.standing(s.r_all), Some(Revalidation::Withdrawn));
    assert!(s.session.result(s.r_all).is_some());
    assert!(s.session.synthesis(s.r_all).is_err());
    // y = (2, 6), v = (2, 4): estimate (1 + 1.5) / (3/4) = 10/3, variance 4/3.
    let SynthesisValue::FixedEffect(fit) = &recomputed.value else { panic!("fixed effect") };
    close(fit.estimate, 10.0 / 3.0, "estimate");
    close(fit.variance, 4.0 / 3.0, "variance");
}

#[test]
fn superseding_a_model_marks_dependents_for_recompute() {
    let mut s = scenario();
    let successor = SourceDeclaration::new("model_v2_on_s1", SourceKind::FittedModel)
        .with_lineage(Lineage::DerivedFromUsedData { inputs: vec![s.s1] });
    let report = s
        .session
        .revise_source(SourceRevision::Supersede { target: s.model, replacement: successor })
        .unwrap();
    let successor_id = report.created_source.unwrap();
    let class = |id: ResultId| {
        report.revalidation.iter().find(|r| r.result == id).expect("classified").class
    };
    assert_eq!(class(s.r_model), Revalidation::Recompute);
    assert_eq!(class(s.r_all), Revalidation::StillValid);
    // A result that needs recomputation is refused, not served.
    let refusal = s.session.synthesis(s.r_model).unwrap_err();
    expect(&refusal, "evidence_source_withdrawn", "evidence_ledger.result_not_valid");
    // The superseded model cannot enter new results.
    let refusal = s
        .session
        .pool_summaries(
            pooling("stale", vec![summary(s.model, 0.5, 1.0), summary(s.s2, 3.0, 4.0)]),
            PoolingDeclaration::Independent,
            &ctx(),
        )
        .unwrap_err();
    expect(&refusal, "evidence_source_withdrawn", "evidence_ledger.source_superseded");
    assert!(refusal.message.contains(&successor_id.to_hex()));
    // Recompute on the successor.
    let recomputed = s
        .session
        .pool_summaries(
            PoolingRequest {
                label: "model_s3".into(),
                summaries: vec![summary(successor_id, 0.7, 1.0), summary(s.s3, 2.0, 1.0)],
                replaces: Some(s.r_model),
            },
            PoolingDeclaration::Independent,
            &ctx(),
        )
        .unwrap();
    assert_eq!(s.session.standing(recomputed.result), Some(Revalidation::StillValid));
    assert!(s.session.synthesis(recomputed.result).unwrap().is_some());
}

#[test]
fn refresh_is_all_or_nothing_and_keeps_untouched_results() {
    let s = scenario();
    let head = s.session.head_hex();
    let state = s.session.state_digest_hex();
    let result_count = s.session.ledger().results().len();

    let refreshed = s
        .session
        .refresh(vec![
            SourceChange::Add(data("s4", 200..230)),
            SourceChange::Revise(SourceRevision::Withdraw {
                target: s.s1,
                reason: "consent withdrawn".into(),
            }),
        ])
        .unwrap();
    assert_eq!(refreshed.reports.len(), 2);
    // Kept: results whose sources are untouched, with their numbers.
    assert!(refreshed.session.synthesis(s.r_s2_s3).unwrap().is_some());
    // Not served: results whose input changed, until a recompute replaces them.
    assert!(refreshed.session.synthesis(s.r_all).is_err());
    let class = |id: ResultId| {
        refreshed.classification.iter().find(|r| r.result == id).expect("classified").class
    };
    assert_eq!(class(s.r_all), Revalidation::Recompute);
    assert_eq!(class(s.r_s1_s2), Revalidation::Withdrawn);
    // The operation history is chained and the new head differs.
    assert_ne!(refreshed.session.head_hex(), head);
    assert_eq!(refreshed.session.ledger().results().len(), result_count);
    // The session refreshed from is untouched.
    assert_eq!(s.session.head_hex(), head);
    assert_eq!(s.session.state_digest_hex(), state);
    assert!(s.session.synthesis(s.r_all).unwrap().is_some());

    // One refused change refuses the batch: the valid addition is not kept.
    let unknown = SourceId::from_hex(&"cd".repeat(32)).unwrap();
    let refusal = s
        .session
        .refresh(vec![
            SourceChange::Add(data("s5", 300..320)),
            SourceChange::Revise(SourceRevision::Withdraw { target: unknown, reason: "x".into() }),
        ])
        .unwrap_err();
    assert_eq!(refusal.code, "invalid_argument");
    assert_eq!(refusal.detail, "evidence_ledger.unknown_source");
    assert_eq!(s.session.state_digest_hex(), state);
    assert_eq!(s.session.ledger().sources().len(), 4);
}

// ---------------------------------------------------------------------------
// Artifact
// ---------------------------------------------------------------------------

#[test]
fn the_artifact_replays_and_a_consumer_refuses_another_history() {
    let s = scenario();
    let export = s.session.export("ledger-1").unwrap();
    assert_eq!(export.carried.len(), 4);
    assert!(export.not_carried.is_empty());
    let consumed = consume_evidence_ledger_artifact(&export.bytes, Some(&export.identity)).unwrap();
    assert_eq!(consumed.identity(), export.identity);
    assert_eq!(consumed.ledger().head(), s.session.ledger().head());
    assert_eq!(consumed.ledger().state_digest(), s.session.ledger().state_digest());
    // The consumer recomputed the numbers: the all-studies pool is a
    // fixed-effect pool of y = (0, 2, 6), v = (1, 2, 4): 10/7 and 4/7.
    let all = consumed.synthesis_of(s.r_all).unwrap();
    assert_eq!(all.method, "fixed_effect_inverse_variance");
    close(all.values[0], 10.0 / 7.0, "estimate");
    close(all.values[1], 4.0 / 7.0, "variance");
    close(all.values[2], 52.0 / 7.0, "Q");

    // A consumer that retained another identity refuses these bytes, and one
    // that holds none still refuses a damaged container.
    let other = s.session.refresh(vec![SourceChange::Add(data("extra", 900..910))]).unwrap();
    let other_export = other.session.export("ledger-2").unwrap();
    assert_ne!(other_export.identity, export.identity);
    let refusal =
        consume_evidence_ledger_artifact(&export.bytes, Some(&other_export.identity)).unwrap_err();
    assert_eq!(refusal.detail, "evidence_ledger.identity_mismatch");
    let mut damaged = export.bytes.clone();
    let at = damaged.len() / 2;
    damaged[at] ^= 0x55;
    assert!(consume_evidence_ledger_artifact(&damaged, None).is_err());
}

#[test]
fn a_resealed_edit_of_a_stored_value_is_refused_on_replay() {
    let s = scenario();
    let records = vec![SynthesisRecord {
        result: s.r_all,
        sources: vec![s.s1, s.s2, s.s3],
        method: SynthesisMethod::FixedEffectInverseVariance {
            estimates: vec![0.0, 2.0, 6.0],
            variances: vec![1.0, 2.0, 4.0],
        },
    }];
    let (_, mut meta) = EvidenceLedgerArtifact::seal(s.session.ledger(), &records).unwrap();
    // The untouched metadata replays.
    assert!(EvidenceLedgerArtifact::from_meta(&meta, None).is_ok());
    // A stored value, consistently resealed, does not.
    meta.syntheses[0].values[0] = 1.5_f64.to_bits();
    EvidenceLedgerArtifact::reseal_identity(&mut meta);
    let error = EvidenceLedgerArtifact::from_meta(&meta, None).unwrap_err();
    assert_eq!(error, EvidenceLedgerArtifactError::SynthesisMismatch("values"));
    let (code, detail, _) = error.refusal().unwrap();
    assert_eq!(code, "invalid_argument");
    assert_eq!(detail, "evidence_ledger.synthesis_mismatch");
}

/// A session holding only the dependence range of the hand-oracle fixture
/// y = (1, 3), se = (1, 2), rho in [0, 0.9].
fn range_session() -> (EvidenceLedgerSession, ResultId, SourceId, SourceId) {
    let mut session = StudyBuilder::evidence_ledger();
    let s3 = session.add_source(data("s3", 0..30)).unwrap();
    let vendor =
        session.add_source(SourceDeclaration::new("vendor", SourceKind::ExternalClaim)).unwrap();
    let outcome = session
        .dependence_range(
            DependenceRangeRequest {
                label: "range".into(),
                studies: vec![
                    StudyEstimate { source: s3, estimate: 1.0, standard_error: 1.0 },
                    StudyEstimate { source: vendor, estimate: 3.0, standard_error: 2.0 },
                ],
                rho_lo: 0.0,
                rho_hi: 0.9,
                replaces: None,
            },
            &ctx(),
        )
        .unwrap();
    (session, outcome.result, s3, vendor)
}

#[test]
fn a_range_outside_the_definite_interval_or_non_finite_is_refused_and_registers_nothing() {
    let (mut session, _, s3, vendor) = range_session();
    let before = session.ledger().results().len();
    let head = session.ledger().head();
    let request = |lo: f64, hi: f64, se: f64| DependenceRangeRequest {
        label: "bad".into(),
        studies: vec![
            StudyEstimate { source: s3, estimate: 1.0, standard_error: 1.0 },
            StudyEstimate { source: vendor, estimate: 3.0, standard_error: se },
        ],
        rho_lo: lo,
        rho_hi: hi,
        replaces: None,
    };
    // Inverted, perfectly correlated, below the definite interval for two
    // studies (-1/(k-1) = -1), non-finite bounds and a non-positive standard error.
    for (lo, hi, se) in [
        (0.5, 0.2, 2.0),
        (0.0, 1.0, 2.0),
        (-1.0, 0.2, 2.0),
        (f64::NAN, 0.2, 2.0),
        (0.0, f64::INFINITY, 2.0),
        (0.0, 0.5, 0.0),
    ] {
        let refusal = session.dependence_range(request(lo, hi, se), &ctx()).unwrap_err();
        assert_eq!(refusal.code, "invalid_argument", "({lo}, {hi}, {se}): {refusal}");
    }
    assert_eq!(session.ledger().results().len(), before);
    assert_eq!(session.ledger().head(), head);
}

#[test]
fn an_exported_dependence_range_is_replayed_as_an_assumption_range_by_the_consumer() {
    let (session, result, _, vendor) = range_session();
    let export = session.export("range-ledger").unwrap();
    assert_eq!(export.carried, vec![result]);
    assert!(export.not_carried.is_empty());
    let consumed = consume_evidence_ledger_artifact(&export.bytes, Some(&export.identity)).unwrap();
    // The consumer recomputed the envelope; wire order is estimate min, rho at
    // min, estimate max, rho at max, variance min, rho at min, variance max,
    // rho at max. Hand values: -1/7 at 0.9, 7/5 at 0, 19/35 at 0.9, 1 at 1/2.
    let env = consumed.synthesis_of(result).unwrap();
    assert_eq!(env.method, "dependence_envelope");
    let expected = [-1.0 / 7.0, 0.9, 1.4, 0.0, 19.0 / 35.0, 0.9, 1.0, 0.5];
    assert_eq!(env.values.len(), expected.len());
    for (value, want) in env.values.iter().zip(expected) {
        close(*value, want, "replayed envelope");
    }
    // The label and the closed interval route survive the replay, and the
    // unknown source stays unknown in the consumer's accounting.
    let record = consumed.ledger().result(result).unwrap();
    assert_eq!(record.claim_class.as_str(), "assumption_range");
    assert!(record.mass.unknown_sources.contains(&vendor));
    let refusal = consumed.ledger().request_interval(result).unwrap_err();
    assert_eq!(refusal.code, "cell_not_licensed");
    assert_eq!(refusal.detail, "evidence_ledger.interval_not_licensed");
    // The same history exports to the same identity, and the producer's range
    // is bound into it: another range is another artifact.
    assert_eq!(range_session().0.export("range-ledger").unwrap().identity, export.identity);
}

#[test]
fn a_resealed_edit_of_the_dependence_range_is_refused_on_replay() {
    let (session, result, s3, vendor) = range_session();
    let records = vec![SynthesisRecord {
        result,
        sources: vec![s3, vendor],
        method: SynthesisMethod::DependenceEnvelope {
            estimates: vec![1.0, 3.0],
            standard_errors: vec![1.0, 2.0],
            rho_lo: 0.0,
            rho_hi: 0.9,
        },
    }];
    let (_, mut meta) = EvidenceLedgerArtifact::seal(session.ledger(), &records).unwrap();
    assert!(EvidenceLedgerArtifact::from_meta(&meta, None).is_ok());
    // A range other than the one the ledger recorded, consistently resealed,
    // no longer binds to the ledger result.
    meta.syntheses[0].rho_hi = Some(0.95_f64.to_bits());
    EvidenceLedgerArtifact::reseal_identity(&mut meta);
    let error = EvidenceLedgerArtifact::from_meta(&meta, None).unwrap_err();
    assert_eq!(error, EvidenceLedgerArtifactError::SynthesisMismatch("assumption range"));
    let (code, detail, _) = error.refusal().unwrap();
    assert_eq!(code, "invalid_argument");
    assert_eq!(detail, "evidence_ledger.synthesis_mismatch");
}

// ---------------------------------------------------------------------------
// Cancellation
// ---------------------------------------------------------------------------

#[test]
fn cancellation_stops_before_a_synthesis_and_leaves_the_session_unchanged() {
    let mut session = StudyBuilder::evidence_ledger();
    let s1 = session.add_source(data("s1", 0..10)).unwrap();
    let s2 = session.add_source(data("s2", 10..20)).unwrap();
    let state = session.state_digest_hex();
    let context = ctx();
    context.cancellation.cancel();
    let refusal = session
        .pool_summaries(
            pooling("p", vec![summary(s1, 1.0, 1.0), summary(s2, 3.0, 4.0)]),
            PoolingDeclaration::Independent,
            &context,
        )
        .unwrap_err();
    assert_eq!(refusal.code, "cancelled_no_claim");
    assert_eq!(session.state_digest_hex(), state);
    assert!(session.ledger().results().is_empty());
    // The same request under a live context succeeds.
    assert!(
        session
            .pool_summaries(
                pooling("p", vec![summary(s1, 1.0, 1.0), summary(s2, 3.0, 4.0)]),
                PoolingDeclaration::Independent,
                &ctx(),
            )
            .is_ok()
    );
}
