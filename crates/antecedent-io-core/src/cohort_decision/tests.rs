//! Hand-calculated cohort means, support and population gates, refusals and artifact
//! tamper tests for the cohort decision source.
//!
//! Oracle. The retained model is `tau(z) = 2 + 0.5 z` (a DR-learner linear final stage
//! gives exactly this map on the fixture of the facade tests). The cohort
//! `z = (-1, 0.5, 1)` with weights `(0.2, 0.3, 0.5)` has row contrasts `(1.5, 2.25, 2.5)`
//! and weighted mean `0.2 * 1.5 + 0.3 * 2.25 + 0.5 * 2.5 = 2.225`. For the compact model
//! with covariance `[[0.04, -0.01], [-0.01, 0.09]]` the variance of the weighted mean is
//! `phibar' V phibar` with `phibar = (1, 0.45)`: `0.04 - 2 * 0.01 * 0.45 + 0.09 * 0.45^2
//! = 0.049225`.

use std::io::Cursor;

use antecedent_core::{ExecutionContext, SupportStatus};
use antecedent_learn::{EffectSupport, FeatureRange, LearnerProvenance, PortablePredictor};

use super::*;
use crate::compact_export::{ExportInput, ExportSpec, InputSupport, MaskCondition, MaskRegion};

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(5)
}

fn approx(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
}

fn fitted(support: Option<EffectSupport>) -> FittedEffect {
    FittedEffect {
        version: 1,
        features: vec![7],
        intercept: true,
        predictor: PortablePredictor {
            version: 1,
            columns: 2,
            provenance: LearnerProvenance {
                spec: "linear".into(),
                implementation: "test".into(),
                version: "1".into(),
            },
            model: PredictionMap::Linear { coefficients: vec![2.0, 0.5], logistic: false },
        },
        support,
    }
}

fn box_support() -> EffectSupport {
    EffectSupport { version: 1, rows: 160, ranges: vec![FeatureRange { lo: -1.0, hi: 1.0 }] }
}

fn parent(claim: &str) -> ParentIdentity {
    ParentIdentity {
        claim_id: claim.into(),
        artifact_digest: format!("artifact-of-{claim}"),
        diagnostics: Vec::new(),
        assumptions: Vec::new(),
    }
}

fn contrast() -> ContrastDeclaration {
    ContrastDeclaration {
        outcome: "y".into(),
        regime_id: ContrastDeclaration::regime_of("t", 1.0, 0.0),
        treatment: Some("t".into()),
        control: Some(0.0),
        active: Some(1.0),
    }
}

fn portable(support: Option<EffectSupport>, claim: &str) -> RetainedEffectModel {
    RetainedEffectModel::from_portable_parts(
        fitted(support),
        vec!["z".into()],
        "target".into(),
        contrast(),
        parent(claim),
    )
    .unwrap()
}

fn ids(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("r{i}")).collect()
}

fn cohort_with(population: &str, z: &[f64], w: &[f64]) -> CohortBinding {
    CohortBinding::new(
        "clinic-2026-q1",
        population,
        ids(z.len()),
        vec!["z".into()],
        vec![CohortColumn::Numeric(z.to_vec())],
        w.to_vec(),
    )
    .unwrap()
}

fn cohort() -> CohortBinding {
    cohort_with("target", &[-1.0, 0.5, 1.0], &[0.2, 0.3, 0.5])
}

fn build(
    model: &RetainedEffectModel,
    cohort: &CohortBinding,
) -> Result<CohortDecisionSource, CohortRefusal> {
    CohortDecisionSource::build(model, cohort, "mmHg", &CohortOptions::default(), &ctx())
}

fn quantity_wire(
    id: &str,
    role: &str,
    units: &str,
    functional: &str,
    regime: &str,
) -> ScientificQuantityWire {
    ScientificQuantityWire {
        version: 1,
        variable_id: id.into(),
        variable_name: id.into(),
        role: role.into(),
        units: units.into(),
        population_id: "target".into(),
        regime_id: regime.into(),
        horizon: 0,
        functional_id: functional.into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

fn compact_model_with(
    functional: &str,
    mask: Vec<MaskRegion>,
) -> Result<RetainedEffectModel, CohortRefusal> {
    let export = CompactExport::build(ExportSpec {
        response: quantity_wire(
            "y",
            "outcome",
            "mmHg",
            functional,
            &ContrastDeclaration::regime_of("t", 1.0, 0.0),
        ),
        inputs: vec![ExportInput {
            quantity: quantity_wire("z", "covariate", "1", "mean", "observed"),
            support: InputSupport::Range { lo: -1.0, hi: 1.0 },
        }],
        terms: vec![TermSpec::Intercept, TermSpec::Linear { quantity: "z".into() }],
        coefficients: vec![2.0, 0.5],
        covariance: vec![0.04, -0.01, -0.01, 0.09],
        mask,
    })
    .unwrap();
    RetainedEffectModel::from_compact_export(&export)
}

fn compact_model() -> RetainedEffectModel {
    compact_model_with(CATE_FUNCTIONAL_ID, vec![]).unwrap()
}

// ---------------------------------------------------------------------------
// Values
// ---------------------------------------------------------------------------

#[test]
fn weighted_cohort_mean_matches_the_hand_calculation() {
    let source = build(&portable(Some(box_support()), "claim-a"), &cohort()).unwrap();
    approx(source.mean_contrast(), 2.225);
    for (got, want) in source.row_contrasts().iter().zip([1.5, 2.25, 2.5]) {
        approx(*got, want);
    }
    assert!(source.is_native());
    assert_eq!(source.support_status(), SupportStatus::Supported);
    assert_eq!(source.support().rows_checked, 3);
    assert_eq!(source.support().rows_outside, 0);
    assert_eq!(source.support().fitted_rows, Some(160));
    assert_eq!(source.support().basis, SUPPORT_BASIS_FITTED_BOX);
    assert!(source.uncalibrated_standard_error().is_none());

    let quantity = source.quantity();
    assert_eq!(quantity.variable_id, "y");
    assert_eq!(quantity.units, "mmHg");
    assert_eq!(quantity.population_id, "target");
    assert_eq!(quantity.regime_id, "contrast:do(t=1)-do(t=0)");
    assert_eq!(quantity.functional_id, COHORT_FUNCTIONAL_ID);
    assert_eq!(quantity.horizon, 0);
    assert_eq!(quantity.conditioning.len(), 1);
    assert_eq!(quantity.conditioning[0].variable_id, COHORT_CONDITIONING_VARIABLE);
    assert_eq!(quantity.conditioning[0].value_id, cohort().digest());
    quantity.validate().unwrap();
}

#[test]
fn row_order_changes_the_cohort_identity_but_not_the_mean() {
    let model = portable(Some(box_support()), "claim-a");
    let forward = build(&model, &cohort()).unwrap();
    let reversed =
        build(&model, &cohort_with("target", &[1.0, 0.5, -1.0], &[0.5, 0.3, 0.2])).unwrap();
    approx(reversed.mean_contrast(), forward.mean_contrast());
    assert_ne!(reversed.identity(), forward.identity());
    assert_ne!(reversed.cohort().digest(), forward.cohort().digest());
}

#[test]
fn compact_variance_matches_the_hand_calculation_and_a_double_sum_oracle() {
    let source = build(&compact_model(), &cohort()).unwrap();
    approx(source.mean_contrast(), 2.225);
    let diagnostic = source.uncalibrated_standard_error().unwrap();
    approx(diagnostic.value, 0.049_225_f64.sqrt());
    assert_eq!(diagnostic.basis, STANDARD_ERROR_BASIS);
    assert_eq!(diagnostic.calibration, STANDARD_ERROR_CALIBRATION);
    // Independent oracle: Var = sum_i sum_j w_i w_j phi_i' V phi_j with phi = (1, z).
    let z = [-1.0, 0.5, 1.0];
    let w = [0.2, 0.3, 0.5];
    let v = [[0.04, -0.01], [-0.01, 0.09]];
    let mut variance: f64 = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            let (pi, pj) = ([1.0, z[i]], [1.0, z[j]]);
            for a in 0..2 {
                for (b, &p_jb) in pj.iter().enumerate() {
                    variance += w[i] * w[j] * pi[a] * v[a][b] * p_jb;
                }
            }
        }
    }
    approx(diagnostic.value, variance.sqrt());
    assert!(!source.is_native());
    assert!(source.provider_id().starts_with("declared_model:"));
    assert_eq!(source.support().basis, SUPPORT_BASIS_DECLARED);
}

#[test]
fn the_interval_route_is_closed_with_a_remedy() {
    let source = build(&compact_model(), &cohort()).unwrap();
    let interval = source.interval_refusal();
    assert_eq!(interval.code, "cell_not_licensed");
    assert_eq!(interval.detail, "cohort_decision.interval_requires_native_fit");
    assert!(interval.remedy.contains("measured native-fit route"));
    interval.to_refusal().validate().unwrap();
    // A diagnostic is present for the compact model and none for the portable one.
    assert!(source.uncalibrated_standard_error().is_some());
    let portable = build(&portable(Some(box_support()), "claim-a"), &cohort()).unwrap();
    assert!(portable.uncalibrated_standard_error().is_none());
}

#[test]
fn law_valued_requests_are_refused_with_a_remedy() {
    let source = build(&compact_model(), &cohort()).unwrap();
    for (request, code, detail) in [
        (
            LawRequest::Probability,
            "decision_contract_unsatisfied",
            "cohort_decision.law_not_supplied",
        ),
        (LawRequest::Quantile, "decision_contract_unsatisfied", "cohort_decision.law_not_supplied"),
        (
            LawRequest::NonlinearUtility,
            "decision_contract_unsatisfied",
            "cohort_decision.law_not_supplied",
        ),
        (LawRequest::JointLaw, "joint_law_required", "cohort_decision.joint_law_required"),
    ] {
        let refusal = source.law_refusal(request);
        assert_eq!((refusal.code, refusal.detail), (code, detail));
        assert!(!refusal.remedy.is_empty());
        refusal.to_refusal().validate().unwrap();
    }
}

// ---------------------------------------------------------------------------
// Negative gates
// ---------------------------------------------------------------------------

#[test]
fn a_model_without_a_support_contract_is_refused_not_assumed_supported() {
    let error = build(&portable(None, "claim-a"), &cohort()).unwrap_err();
    assert_eq!(error.code, "cell_not_licensed");
    assert_eq!(error.detail, "cohort_decision.support_contract_absent");
    assert!(!portable(None, "claim-a").has_support_contract());
}

#[test]
fn rows_outside_the_fitted_support_are_refused_and_named() {
    let outside = cohort_with("target", &[-1.0, 1.5, 2.0], &[0.2, 0.3, 0.5]);
    let error = build(&portable(Some(box_support()), "claim-a"), &outside).unwrap_err();
    assert_eq!(error.code, "cell_not_licensed");
    assert_eq!(error.detail, "cohort_decision.outside_support");
    assert!(error.subject.contains("row=r1"), "{}", error.subject);
    assert!(error.subject.contains("rows_outside=2"), "{}", error.subject);
    // The compact route refuses the same row through its declared range.
    let compact = build(&compact_model(), &outside).unwrap_err();
    assert_eq!(compact.detail, "cohort_decision.outside_support");
    assert!(compact.subject.contains("row=r1"), "{}", compact.subject);
}

#[test]
fn a_masked_region_of_a_compact_model_is_refused() {
    let masked = compact_model_with(
        CATE_FUNCTIONAL_ID,
        vec![MaskRegion {
            id: "top_decile".into(),
            conditions: vec![MaskCondition {
                quantity: "z".into(),
                within: InputSupport::Range { lo: 0.9, hi: 1.0 },
            }],
        }],
    )
    .unwrap();
    let error = build(&masked, &cohort()).unwrap_err();
    assert_eq!(error.detail, "cohort_decision.outside_support");
    assert!(error.subject.contains("top_decile"), "{}", error.subject);
}

#[test]
fn schema_unit_and_contrast_mismatches_are_typed_refusals() {
    let model = portable(Some(box_support()), "claim-a");
    // Missing and extra features.
    let renamed = CohortBinding::new(
        "s",
        "target",
        ids(1),
        vec!["age".into()],
        vec![CohortColumn::Numeric(vec![0.0])],
        vec![1.0],
    )
    .unwrap();
    let error = build(&model, &renamed).unwrap_err();
    assert_eq!(error.code, "schema_mismatch");
    assert_eq!(error.detail, "cohort_decision.feature_schema_mismatch");
    let extra = CohortBinding::new(
        "s",
        "target",
        ids(1),
        vec!["z".into(), "age".into()],
        vec![CohortColumn::Numeric(vec![0.0]), CohortColumn::Numeric(vec![40.0])],
        vec![1.0],
    )
    .unwrap();
    assert_eq!(
        build(&model, &extra).unwrap_err().detail,
        "cohort_decision.feature_schema_mismatch"
    );
    // A categorical column where the model reads a number.
    let categorical = CohortBinding::new(
        "s",
        "target",
        ids(1),
        vec!["z".into()],
        vec![CohortColumn::Categorical(vec!["a".into()])],
        vec![1.0],
    )
    .unwrap();
    assert_eq!(
        build(&model, &categorical).unwrap_err().detail,
        "cohort_decision.feature_type_mismatch"
    );
    // Units are required, and a compact response states its own.
    let blank =
        CohortDecisionSource::build(&model, &cohort(), " ", &CohortOptions::default(), &ctx());
    assert_eq!(blank.unwrap_err().detail, "cohort_decision.outcome_units_required");
    let wrong_units = CohortDecisionSource::build(
        &compact_model(),
        &cohort(),
        "g/dL",
        &CohortOptions::default(),
        &ctx(),
    );
    assert_eq!(wrong_units.unwrap_err().detail, "cohort_decision.units_mismatch");
    // The contrast the caller expects must be the model's.
    let options = CohortOptions {
        expected_contrast_regime: Some("contrast:do(t=2)-do(t=0)".into()),
        shift: None,
    };
    let error =
        CohortDecisionSource::build(&model, &cohort(), "mmHg", &options, &ctx()).unwrap_err();
    assert_eq!(error.code, "quantity_semantics_mismatch");
    assert_eq!(error.detail, "cohort_decision.contrast_mismatch");
    // A compact export of an absolute mean is not a contrast.
    let absolute = compact_model_with("mean", vec![]).unwrap_err();
    assert_eq!(absolute.detail, "cohort_decision.contrast_not_declared");
}

#[test]
fn invalid_cohorts_are_refused_at_binding() {
    let bind = |ids: Vec<String>, z: Vec<f64>, w: Vec<f64>| {
        CohortBinding::new("s", "target", ids, vec!["z".into()], vec![CohortColumn::Numeric(z)], w)
    };
    assert_eq!(
        bind(ids(2), vec![0.0, 1.0], vec![0.5, 0.6]).unwrap_err().detail,
        "cohort_decision.weights_not_normalized"
    );
    assert_eq!(
        bind(ids(2), vec![0.0, 1.0], vec![1.5, -0.5]).unwrap_err().detail,
        "cohort_decision.weights_invalid"
    );
    assert_eq!(
        bind(ids(2), vec![0.0, f64::NAN], vec![0.5, 0.5]).unwrap_err().detail,
        "cohort_decision.cohort_invalid"
    );
    assert_eq!(
        bind(vec!["a".into(), "a".into()], vec![0.0, 1.0], vec![0.5, 0.5]).unwrap_err().detail,
        "cohort_decision.cohort_invalid"
    );
    assert_eq!(bind(vec![], vec![], vec![]).unwrap_err().detail, "cohort_decision.cohort_empty");
    let normalized = CohortBinding::normalize_weights(&[2.0, 6.0]).unwrap();
    approx(normalized[0], 0.25);
    approx(normalized[1], 0.75);
    approx(CohortBinding::uniform_weights(4).unwrap()[3], 0.25);
}

// ---------------------------------------------------------------------------
// Population and declared shift
// ---------------------------------------------------------------------------

#[test]
fn a_cohort_from_another_population_needs_a_declared_shift_and_support_overlap() {
    let model = portable(Some(box_support()), "claim-a");
    let other = cohort_with("clinic_b", &[-1.0, 0.5, 1.0], &[0.2, 0.3, 0.5]);
    // Without a declaration the unmet premise is named.
    let refusal = build(&model, &other).unwrap_err();
    assert_eq!(refusal.code, "transport_missing_evidence");
    assert_eq!(refusal.detail, "cohort_decision.shift_declaration_required");
    // With the declaration the same prediction is issued under an assumption label.
    let declared = CovariateShiftDeclaration::new("target", "clinic_b", &["z"]);
    let options = CohortOptions { shift: Some(declared.clone()), expected_contrast_regime: None };
    let source = CohortDecisionSource::build(&model, &other, "mmHg", &options, &ctx()).unwrap();
    approx(source.mean_contrast(), 2.225);
    assert_eq!(source.support_status(), SupportStatus::Extrapolative);
    assert_eq!(source.quantity().population_id, "clinic_b");
    assert!(source.provider_id().contains("declared_covariate_shift"));
    assert!(source.assumptions().iter().any(|a| a.contains(COVARIATE_SHIFT_ASSUMPTION)));
    // The declaration is bound into the identity.
    let within = build(&model, &cohort()).unwrap();
    assert_ne!(source.identity(), within.identity());
    // A declaration that names other modifiers, or another population, is not certified.
    for bad in [
        CovariateShiftDeclaration::new("target", "clinic_b", &["z", "age"]),
        CovariateShiftDeclaration::new("target", "clinic_c", &["z"]),
        CovariateShiftDeclaration::new("elsewhere", "clinic_b", &["z"]),
    ] {
        let options = CohortOptions { shift: Some(bad), expected_contrast_regime: None };
        let error =
            CohortDecisionSource::build(&model, &other, "mmHg", &options, &ctx()).unwrap_err();
        assert_eq!(error.code, "transport_not_certified");
        assert_eq!(error.detail, "cohort_decision.shift_declaration_mismatch");
    }
    // A shift declared for a cohort that is the model's own population is refused.
    let options = CohortOptions { shift: Some(declared), expected_contrast_regime: None };
    let error =
        CohortDecisionSource::build(&model, &cohort(), "mmHg", &options, &ctx()).unwrap_err();
    assert_eq!(error.detail, "cohort_decision.shift_declaration_unneeded");
}

#[test]
fn a_shifted_cohort_outside_the_support_is_a_transport_support_failure() {
    let model = portable(Some(box_support()), "claim-a");
    let outside = cohort_with("clinic_b", &[0.0, 3.0], &[0.5, 0.5]);
    let options = CohortOptions {
        shift: Some(CovariateShiftDeclaration::new("target", "clinic_b", &["z"])),
        expected_contrast_regime: None,
    };
    let error =
        CohortDecisionSource::build(&model, &outside, "mmHg", &options, &ctx()).unwrap_err();
    assert_eq!(error.code, "transport_support_failure");
    assert_eq!(error.detail, "cohort_decision.shift_support_failure");
}

/// The shift gate compares four declared strings and the support box inclusively: a declaration
/// whose assumption label was changed is not certified, the fitted box edge is inside the
/// support and the next representable value outside it is a transport support failure, and the
/// refusal for an unshifted cohort names the plain out-of-support detail instead.
#[test]
fn the_shift_gate_is_exact_on_the_assumption_label_and_the_support_edge() {
    let model = portable(Some(box_support()), "claim-a");
    let on_edge = cohort_with("clinic_b", &[-1.0, 1.0], &[0.5, 0.5]);
    let declared = CovariateShiftDeclaration::new("target", "clinic_b", &["z"]);
    let options = CohortOptions { shift: Some(declared.clone()), expected_contrast_regime: None };
    // tau(-1) = 1.5 and tau(1) = 2.5: the edges of the fitted box are inside the support.
    let source = CohortDecisionSource::build(&model, &on_edge, "mmHg", &options, &ctx()).unwrap();
    approx(source.mean_contrast(), 2.0);
    assert_eq!(source.support_status(), SupportStatus::Extrapolative);

    let mut relabelled = declared;
    relabelled.assumption.push('x');
    let options = CohortOptions { shift: Some(relabelled), expected_contrast_regime: None };
    let error =
        CohortDecisionSource::build(&model, &on_edge, "mmHg", &options, &ctx()).unwrap_err();
    assert_eq!(error.detail, "cohort_decision.shift_declaration_mismatch");

    let just_outside =
        cohort_with("clinic_b", &[-1.0, f64::from_bits(1.0_f64.to_bits() + 1)], &[0.5, 0.5]);
    let options = CohortOptions {
        shift: Some(CovariateShiftDeclaration::new("target", "clinic_b", &["z"])),
        expected_contrast_regime: None,
    };
    let error =
        CohortDecisionSource::build(&model, &just_outside, "mmHg", &options, &ctx()).unwrap_err();
    assert_eq!(
        (error.code, error.detail),
        ("transport_support_failure", "cohort_decision.shift_support_failure")
    );
    let unshifted =
        cohort_with("target", &[-1.0, f64::from_bits(1.0_f64.to_bits() + 1)], &[0.5, 0.5]);
    let error = build(&model, &unshifted).unwrap_err();
    assert_eq!(error.detail, "cohort_decision.outside_support");
}

// ---------------------------------------------------------------------------
// Artifact
// ---------------------------------------------------------------------------

fn read_body(bytes: &[u8]) -> (ArtifactManifest, SourceBody) {
    let mut reader = ArtifactReader::open_seek(Cursor::new(bytes)).unwrap();
    let manifest = reader.manifest().clone();
    let section = reader.load_section(COHORT_DECISION_SECTION).unwrap();
    (manifest, from_cbor(section.as_bytes()).unwrap())
}

fn write_body(mut manifest: ArtifactManifest, body: &SourceBody) -> Vec<u8> {
    let payload = to_cbor(body).unwrap();
    manifest.sections = vec![section_descriptor_with_policy(
        COHORT_DECISION_SECTION,
        "application/cbor",
        &payload,
        CompressPolicy::Never,
    )];
    let encoded = EncodedArtifact {
        manifest,
        sections: vec![SectionBytes::new(COHORT_DECISION_SECTION, payload)],
    };
    let mut out = Vec::new();
    encoded.write_to(&mut out).unwrap();
    out
}

#[test]
fn the_artifact_round_trips_under_the_retained_identity() {
    for model in [portable(Some(box_support()), "claim-a"), compact_model()] {
        let source = build(&model, &cohort()).unwrap();
        let bytes = source.to_bytes("cohort-decision").unwrap();
        let expectation = CohortExpectation {
            identity: source.identity().to_owned(),
            parent_claim: Some(model.parent().claim_id.clone()),
            population_id: Some("target".into()),
            outcome_units: Some("mmHg".into()),
            contrast_regime: Some("contrast:do(t=1)-do(t=0)".into()),
            cohort_digest: Some(cohort().digest()),
        };
        let consumed = CohortDecisionSource::consume(&bytes, &expectation).unwrap();
        assert_eq!(consumed, source);
        assert_eq!(consumed.mean_contrast().to_bits(), source.mean_contrast().to_bits());
        assert_eq!(consumed.uncalibrated_standard_error(), source.uncalibrated_standard_error());
    }
}

#[test]
fn every_retained_expectation_is_a_typed_gate() {
    let source = build(&portable(Some(box_support()), "claim-a"), &cohort()).unwrap();
    let bytes = source.to_bytes("cohort-decision").unwrap();
    let identity = source.identity().to_owned();
    let with = |edit: &dyn Fn(&mut CohortExpectation)| {
        let mut expectation = CohortExpectation::for_identity(identity.clone());
        edit(&mut expectation);
        CohortDecisionSource::consume(&bytes, &expectation).unwrap_err().detail
    };
    assert_eq!(
        with(&|e| e.parent_claim = Some("claim-b".into())),
        "cohort_decision.wrong_parent_identity"
    );
    assert_eq!(
        with(&|e| e.population_id = Some("clinic_b".into())),
        "cohort_decision.population_mismatch"
    );
    assert_eq!(with(&|e| e.outcome_units = Some("g/dL".into())), "cohort_decision.units_mismatch");
    assert_eq!(
        with(&|e| e.contrast_regime = Some("contrast:do(t=2)-do(t=0)".into())),
        "cohort_decision.contrast_mismatch"
    );
    assert_eq!(with(&|e| e.cohort_digest = Some("0".repeat(64))), "cohort_decision.cohort_changed");
    assert_eq!(with(&|e| e.identity = "0".repeat(64)), "cohort_decision.identity_unexpected");
}

#[test]
fn a_different_parent_claim_is_a_different_source() {
    let a = build(&portable(Some(box_support()), "claim-a"), &cohort()).unwrap();
    let b = build(&portable(Some(box_support()), "claim-b"), &cohort()).unwrap();
    approx(a.mean_contrast(), b.mean_contrast());
    assert_ne!(a.identity(), b.identity());
    let bytes = b.to_bytes("cohort-decision").unwrap();
    let error =
        CohortDecisionSource::consume(&bytes, &CohortExpectation::for_identity(a.identity()))
            .unwrap_err();
    assert_eq!(error.detail, "cohort_decision.identity_unexpected");
}

#[test]
fn semantic_mutations_are_refused_even_when_the_container_is_resealed() {
    let model = portable(Some(box_support()), "claim-a");
    let source = build(&model, &cohort()).unwrap();
    let identity = source.identity().to_owned();
    let bytes = source.to_bytes("cohort-decision").unwrap();
    let consume = |mutated: &[u8]| {
        CohortDecisionSource::consume(mutated, &CohortExpectation::for_identity(identity.clone()))
            .unwrap_err()
            .detail
    };
    let (manifest, body) = read_body(&bytes);

    // A changed weight pair (still normalized), stored identity untouched.
    let mut changed = body.clone();
    changed.cohort = cohort_with("target", &[-1.0, 0.5, 1.0], &[0.3, 0.2, 0.5]);
    assert_eq!(consume(&write_body(manifest.clone(), &changed)), "cohort_decision.replay_mismatch");

    // A changed stored mean, identity untouched.
    let mut changed = body.clone();
    changed.mean_contrast += 1e-9;
    assert_eq!(consume(&write_body(manifest.clone(), &changed)), "cohort_decision.replay_mismatch");

    // A changed coefficient under the same stored model identity.
    let mut changed = body.clone();
    if let ModelWire::Portable { effect, .. } = &mut changed.model {
        effect.predictor.model =
            PredictionMap::Linear { coefficients: vec![2.0, 0.6], logistic: false };
    }
    assert_eq!(
        consume(&write_body(manifest.clone(), &changed)),
        "cohort_decision.model_identity_mismatch"
    );

    // A changed unit, identity untouched.
    let mut changed = body.clone();
    changed.outcome_units = "g/dL".into();
    assert_eq!(consume(&write_body(manifest.clone(), &changed)), "cohort_decision.replay_mismatch");

    // A fully resealed artifact of a different cohort carries a different identity.
    let other = build(&model, &cohort_with("target", &[-1.0, 0.5, 1.0], &[0.3, 0.2, 0.5])).unwrap();
    assert_ne!(other.identity(), identity);
    assert_eq!(
        consume(&other.to_bytes("cohort-decision").unwrap()),
        "cohort_decision.identity_unexpected"
    );

    // An unknown body version and a foreign container are refused before any decode.
    let mut changed = body;
    changed.version = 2;
    assert_eq!(consume(&write_body(manifest, &changed)), "cohort_decision.unsupported_version");
    assert!(consume(&bytes[..bytes.len() / 2]).starts_with("cohort_decision."));
}

#[test]
fn a_tampered_compact_payload_is_refused() {
    let source = build(&compact_model(), &cohort()).unwrap();
    let identity = source.identity().to_owned();
    let bytes = source.to_bytes("cohort-decision").unwrap();
    let (manifest, mut body) = read_body(&bytes);
    // Flip one stored byte of the embedded export container.
    if let ModelWire::Compact { export, .. } = &mut body.model {
        let last = export.len() - 1;
        export[last] ^= 0x01;
    }
    let error = CohortDecisionSource::consume(
        &write_body(manifest, &body),
        &CohortExpectation::for_identity(identity),
    )
    .unwrap_err();
    assert_eq!(error.detail, "cohort_decision.fitted_model_invalid");
}

#[test]
fn a_declared_shift_round_trips_and_is_bound_into_the_identity() {
    let model = portable(Some(box_support()), "claim-a");
    let other = cohort_with("clinic_b", &[-1.0, 0.5, 1.0], &[0.2, 0.3, 0.5]);
    let declared = CovariateShiftDeclaration::new("target", "clinic_b", &["z"]);
    let options = CohortOptions { shift: Some(declared.clone()), expected_contrast_regime: None };
    let source = CohortDecisionSource::build(&model, &other, "mmHg", &options, &ctx()).unwrap();
    let bytes = source.to_bytes("cohort-decision").unwrap();
    let consumed =
        CohortDecisionSource::consume(&bytes, &CohortExpectation::for_identity(source.identity()))
            .unwrap();
    assert_eq!(consumed.shift(), Some(&declared));
    assert_eq!(consumed.support_status(), SupportStatus::Extrapolative);
    // Dropping the declaration from the stored body no longer replays.
    let (manifest, mut body) = read_body(&bytes);
    body.shift = None;
    let error = CohortDecisionSource::consume(
        &write_body(manifest, &body),
        &CohortExpectation::for_identity(source.identity()),
    )
    .unwrap_err();
    assert_eq!(error.detail, "cohort_decision.shift_declaration_required");
}

#[test]
fn an_exported_source_carries_one_mean_and_no_law() {
    let source = build(&portable(Some(box_support()), "claim-a"), &cohort()).unwrap();
    let identity = source.identity().to_owned();
    let bytes = source.to_bytes("cohort-decision").unwrap();
    let (manifest, mut body) = read_body(&bytes);
    assert_eq!(body.quantity.functional_id, COHORT_FUNCTIONAL_ID);
    assert_eq!(body.scope, POINT_SCOPE);
    assert!(body.standard_error.is_none());
    // Relabelling the single mean as an outcome law is another quantity and does not replay.
    body.quantity.functional_id = "outcome".into();
    let error = CohortDecisionSource::consume(
        &write_body(manifest, &body),
        &CohortExpectation::for_identity(identity),
    )
    .unwrap_err();
    assert_eq!(error.detail, "cohort_decision.replay_mismatch");
}

#[test]
fn caller_intake_limits_refuse_what_the_defaults_admit() {
    let defaults = CohortIntakeLimits::default();
    let bind = |limits: &CohortIntakeLimits| {
        CohortBinding::new_within(
            "s",
            "target",
            ids(3),
            vec!["z".into()],
            vec![CohortColumn::Numeric(vec![-1.0, 0.5, 1.0])],
            vec![0.2, 0.3, 0.5],
            limits,
        )
    };
    assert!(bind(&defaults).is_ok());
    let small = CohortIntakeLimits { max_rows: 2, ..defaults };
    assert_eq!(bind(&small).unwrap_err().detail, "cohort_decision.cohort_too_large");

    let model = portable(Some(box_support()), "claim-a");
    assert_eq!(
        CohortDecisionSource::build_within(
            &model,
            &cohort(),
            "mmHg",
            &CohortOptions::default(),
            &ctx(),
            &small
        )
        .unwrap_err()
        .detail,
        "cohort_decision.cohort_too_large"
    );

    let source = build(&model, &cohort()).unwrap();
    let bytes = source.to_bytes("limits").unwrap();
    let expectation = CohortExpectation::for_identity(source.identity());
    CohortDecisionSource::consume_within(&bytes, &expectation, &defaults).unwrap();
    assert_eq!(
        CohortDecisionSource::consume_within(&bytes, &expectation, &small).unwrap_err().detail,
        "cohort_decision.cohort_too_large"
    );
    let tight = CohortIntakeLimits { max_bytes: bytes.len() - 1, ..defaults };
    assert_eq!(
        CohortDecisionSource::consume_within(&bytes, &expectation, &tight).unwrap_err().detail,
        "cohort_decision.oversized"
    );
    assert_eq!(
        source.to_bytes_within("limits", bytes.len() - 1).unwrap_err().detail,
        "cohort_decision.oversized"
    );
}
