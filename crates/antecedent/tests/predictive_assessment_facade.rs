//! Facade integration of predictive assessment: the held-out check from the
//! evidence ledger's row identities, scores against hand values, the closed
//! interval and causal routes, and an artifact a fresh consumer re-scores.
//!
//! Hand oracles:
//! * `N(0, 1)` at `0`: CRPS `(sqrt 2 - 1) / sqrt(pi) = 0.2336949772...`; log score
//!   `ln(sqrt(2 pi)) = 0.9189385332...`.
//! * `N(0, 1)` at `1`: CRPS `0.6024413576...`; log score `0.9189385332 + 1/2`.
//! * Over the two outcomes `0` and `1` the mean CRPS is `0.4180681674...`.

use antecedent::analysis::evidence_ledger::{
    Lineage, ObservationId, SourceDeclaration, SourceKind,
};
use antecedent::analysis::predictive_assessment::{
    DependenceDeclaration, FitDeclaration, ForecastLaw, Horizon, LawSource, Outcome, OverlapPolicy,
    OverlapStatus, PredictiveAssessmentConsumeLimits, PredictiveAssessmentRefusal,
    PreparedPredictiveAssessment, RealizedObservations, ScoreIntervalRequest, ScoreKind,
    ValidationDesign, consume_predictive_assessment_artifact, fit_declaration_from_ledger,
    observation_keys_from_ledger,
};
use antecedent_contract::evidence_ledger::{EvidenceLedger, SourceId};
use antecedent_core::ExecutionContext;
use antecedent_validate::predictive_assessment::{
    PredictiveAssessmentSpec, PredictiveTarget, PredictorLaws,
};

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(5)
}

fn close(actual: f64, expected: f64, what: &str) {
    assert!((actual - expected).abs() <= 1e-9, "{what}: expected {expected}, got {actual}");
}

fn rows(from: u32, to: u32) -> Vec<ObservationId> {
    (from..to).map(|i| ObservationId::from_key("cohort", &i.to_le_bytes())).collect()
}

#[allow(
    clippy::struct_field_names,
    reason = "the fields name the ledger and the two sources it holds"
)]
struct Ledger {
    ledger: EvidenceLedger,
    train: SourceId,
    test: SourceId,
}

fn ledger() -> Ledger {
    let mut ledger = EvidenceLedger::new();
    let train = ledger
        .add_source(
            SourceDeclaration::new("train", SourceKind::ObservationSet)
                .with_observations(rows(0, 50)),
        )
        .unwrap();
    let test = ledger
        .add_source(
            SourceDeclaration::new("test", SourceKind::ObservationSet)
                .with_observations(rows(100, 104)),
        )
        .unwrap();
    Ledger { ledger, train, test }
}

fn target() -> PredictiveTarget {
    PredictiveTarget::new("outcome", Horizon::Contemporaneous, "cohort", "units")
}

/// Two assessed observations with outcomes `0` and `1` under `N(0, 1)` laws, the cases
/// listed in the ledger's key order.
fn spec_over(
    keys: Vec<antecedent_validate::predictive_assessment::ObservationKey>,
    fit: FitDeclaration,
) -> PredictiveAssessmentSpec {
    let n = keys.len();
    let outcomes: Vec<Outcome> = (0..n).map(|i| Outcome::Real((i % 2) as f64)).collect();
    let mut spec = PredictiveAssessmentSpec::new(
        RealizedObservations { target: target(), ids: keys, outcomes },
        vec![PredictorLaws {
            name: "standard_normal".to_owned(),
            source: LawSource::Native { model: "standard_normal".to_owned() },
            target: target(),
            laws: vec![ForecastLaw::gaussian(0.0, 1.0); n],
            fit,
        }],
        ValidationDesign::HeldOutSplit,
        vec![ScoreKind::Crps, ScoreKind::Log],
    );
    spec.dependence = Some(DependenceDeclaration::Iid);
    spec
}

fn mean(
    report: &antecedent_validate::predictive_assessment::PredictiveAssessmentReport,
    kind: ScoreKind,
) -> f64 {
    report.predictors[0].scores.iter().find(|s| s.kind == kind).unwrap().mean
}

#[test]
fn a_model_fitted_on_other_rows_is_verified_held_out_from_the_ledger() {
    let mut l = ledger();
    let model = l
        .ledger
        .add_source(
            SourceDeclaration::new("model", SourceKind::FittedModel)
                .with_lineage(Lineage::DerivedFromUsedData { inputs: vec![l.train] }),
        )
        .unwrap();
    let fit = fit_declaration_from_ledger(&l.ledger, &[model]).unwrap();
    let keys = observation_keys_from_ledger(&l.ledger, l.test).unwrap();
    assert_eq!(keys.len(), 4);
    let mut spec = spec_over(keys, fit);
    spec.overlap_policy = OverlapPolicy::RequireHeldOut;
    let prepared = PreparedPredictiveAssessment::assess(&spec, &ctx()).unwrap();
    let report = prepared.report();
    assert_eq!(report.predictors[0].overlap, OverlapStatus::HeldOutVerified { fitted_on: 50 });
    // Cases alternate outcome 0 and 1: the mean CRPS is the average of the two hand values.
    close(
        mean(report, ScoreKind::Crps),
        0.5 * (0.233_694_977_255_109 + 0.602_441_357_627_616_3),
        "crps",
    );
    close(mean(report, ScoreKind::Log), 0.918_938_533_204_672_7 + 0.25, "log score");
    assert_eq!(report.inference_claim, "point_only");
}

#[test]
fn training_rows_among_the_assessed_ones_are_in_sample_and_can_be_refused() {
    let mut l = ledger();
    let model = l
        .ledger
        .add_source(
            SourceDeclaration::new("model", SourceKind::FittedModel)
                .with_lineage(Lineage::DerivedFromUsedData { inputs: vec![l.train, l.test] }),
        )
        .unwrap();
    let fit = fit_declaration_from_ledger(&l.ledger, &[model]).unwrap();
    let keys = observation_keys_from_ledger(&l.ledger, l.test).unwrap();

    let labelled =
        PreparedPredictiveAssessment::assess(&spec_over(keys.clone(), fit.clone()), &ctx())
            .unwrap();
    assert_eq!(
        labelled.report().predictors[0].overlap,
        OverlapStatus::InSample { shared: 4, assessed: 4 }
    );

    let mut strict = spec_over(keys, fit);
    strict.overlap_policy = OverlapPolicy::RequireHeldOut;
    let refusal = PreparedPredictiveAssessment::assess(&strict, &ctx()).unwrap_err();
    assert_eq!(refusal.code, "evidence_double_counted");
    assert_eq!(refusal.detail, "predictive_assessment.in_sample");
}

#[test]
fn an_undeclared_fit_set_is_unverified_never_held_out() {
    let mut l = ledger();
    // A model with no declared observations: the ledger cannot say what it rests on.
    let model = l
        .ledger
        .add_source(SourceDeclaration::new("provider_model", SourceKind::ExternalClaim))
        .unwrap();
    let fit = fit_declaration_from_ledger(&l.ledger, &[model]).unwrap();
    assert!(matches!(fit, FitDeclaration::Undeclared { .. }));
    let keys = observation_keys_from_ledger(&l.ledger, l.test).unwrap();
    let prepared =
        PreparedPredictiveAssessment::assess(&spec_over(keys.clone(), fit.clone()), &ctx())
            .unwrap();
    assert_eq!(prepared.report().predictors[0].overlap.label(), "unverified");

    let mut strict = spec_over(keys, fit);
    strict.overlap_policy = OverlapPolicy::RequireHeldOut;
    let refusal = PreparedPredictiveAssessment::assess(&strict, &ctx()).unwrap_err();
    assert_eq!(refusal.code, "evidence_overlap_unknown");

    // No fit source declared at all is the same.
    assert!(matches!(
        fit_declaration_from_ledger(&l.ledger, &[]).unwrap(),
        FitDeclaration::Undeclared { .. }
    ));
    // An unknown source id and an undeclared observation source are refusals.
    let stranger = SourceId::from_digest(antecedent_core::SemanticDigest::from_bytes([7; 32]));
    let error = fit_declaration_from_ledger(&l.ledger, &[stranger]).unwrap_err();
    assert_eq!(error.detail, "predictive_assessment.unknown_source");
    let error = observation_keys_from_ledger(&l.ledger, model).unwrap_err();
    assert_eq!(error.code, "evidence_overlap_unknown");
}

#[test]
fn the_interval_causal_and_probability_routes_are_closed() {
    let l = ledger();
    let keys = observation_keys_from_ledger(&l.ledger, l.test).unwrap();
    let prepared = PreparedPredictiveAssessment::assess(
        &spec_over(keys.clone(), FitDeclaration::NotFitted),
        &ctx(),
    )
    .unwrap();
    let interval = prepared.request_score_interval().unwrap_err();
    assert_eq!(interval.code, "cell_not_licensed");
    assert_eq!(interval.detail, "predictive_assessment.interval_not_licensed");
    let causal = prepared.request_causal_validity().unwrap_err();
    assert_eq!(causal.code, "route_not_supported");
    assert_eq!(causal.detail, "predictive_assessment.not_a_causal_claim");
    let probability = prepared.request_score_as_probability().unwrap_err();
    assert_eq!(probability.detail, "predictive_assessment.score_not_probability");

    // Asking for an interval in the specification is refused before anything is scored.
    let mut requested = spec_over(keys, FitDeclaration::NotFitted);
    requested.score_interval =
        Some(ScoreIntervalRequest { level: 0.95, method: "stationary bootstrap".into() });
    let refusal = PreparedPredictiveAssessment::assess(&requested, &ctx()).unwrap_err();
    assert_eq!(refusal.code, "cell_not_licensed");
    // The refusal is a registered reason with the standard prefix when displayed.
    assert!(refusal.to_string().contains("reason=cell_not_licensed"));
}

#[test]
fn a_fresh_consumer_rescores_the_exported_assessment() {
    let l = ledger();
    let keys = observation_keys_from_ledger(&l.ledger, l.test).unwrap();
    let prepared =
        PreparedPredictiveAssessment::assess(&spec_over(keys, FitDeclaration::NotFitted), &ctx())
            .unwrap();
    let bytes = prepared.export("predictive-assessment-facade").unwrap();
    let consumed = consume_predictive_assessment_artifact(
        &bytes,
        Some(prepared.identity()),
        PredictiveAssessmentConsumeLimits::default(),
        &ctx(),
    )
    .unwrap();
    assert_eq!(consumed.report(), prepared.report());

    // An assessment of the same laws on other observations is another identity.
    let other_keys: Vec<_> = (200..204_u32)
        .map(|i| {
            antecedent_validate::predictive_assessment::ObservationKey::from_key(
                "other",
                &i.to_le_bytes(),
            )
        })
        .collect();
    let other = PreparedPredictiveAssessment::assess(
        &spec_over(other_keys, FitDeclaration::NotFitted),
        &ctx(),
    )
    .unwrap();
    let other_bytes = other.export("other").unwrap();
    let refusal: PredictiveAssessmentRefusal = consume_predictive_assessment_artifact(
        &other_bytes,
        Some(prepared.identity()),
        PredictiveAssessmentConsumeLimits::default(),
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(refusal.code, "route_not_supported");
    assert_eq!(refusal.detail, "predictive_assessment.wrong_contract");
}
