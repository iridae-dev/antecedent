//! The measured cohort-mean interval envelope: both source artifacts replay under their
//! retained identities, a declared (caller-built) export is refused with its own detail,
//! and the reported scalar is bound to a currently attesting coverage record or refused
//! for want of one.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_estimate::cohort_linear_effect::{NativeCohortFit, fit_cohort_linear_effect};
use antecedent_io::measured_inference::MeasuredInference;
use antecedent_io::measured_inference::cohort_interval::{CohortIntervalArtifactWire, ROUTE};
use antecedent_io_core::cohort_decision::{
    CATE_FUNCTIONAL_ID, CohortBinding, CohortColumn, CohortDecisionSource, CohortOptions,
    ContrastDeclaration, RetainedEffectModel,
};
use antecedent_io_core::compact_export::{
    CompactExport, ExportInput, ExportSpec, InputSupport, TermSpec,
};
use antecedent_io_core::quantity_wire::ScientificQuantityWire;

fn wire(id: &str, role: &str, functional: &str, regime: &str) -> ScientificQuantityWire {
    ScientificQuantityWire {
        version: 1,
        variable_id: id.into(),
        variable_name: id.into(),
        role: role.into(),
        units: "unit".into(),
        population_id: "target".into(),
        regime_id: regime.into(),
        horizon: 0,
        functional_id: functional.into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

fn lcg(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
    ((*state >> 11) as f64 + 0.5) / 9_007_199_254_740_992.0
}

fn data(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut s = 29_u64;
    let (mut x, mut t, mut y) = (vec![], vec![], vec![]);
    for i in 0..n {
        let xi = 4.0 * lcg(&mut s) - 2.0;
        let ti = f64::from(u8::from(i % 2 == 0));
        x.push(xi);
        t.push(ti);
        y.push(2.0 + xi + ti * (1.0 + 0.5 * xi) + (lcg(&mut s) - 0.5));
    }
    (x, t, y)
}

fn response() -> ScientificQuantityWire {
    wire("y", "outcome", CATE_FUNCTIONAL_ID, &ContrastDeclaration::regime_of("t", 1.0, 0.0))
}

fn source_of(export: &CompactExport, x: &[f64]) -> CohortDecisionSource {
    let model = RetainedEffectModel::from_compact_export(export).unwrap();
    let cohort = CohortBinding::new(
        "cohort",
        "target",
        (0..x.len()).map(|i| format!("r{i}")).collect(),
        vec!["x".into()],
        vec![CohortColumn::Numeric(x.to_vec())],
        CohortBinding::uniform_weights(x.len()).unwrap(),
    )
    .unwrap();
    CohortDecisionSource::build(
        &model,
        &cohort,
        "unit",
        &CohortOptions::default(),
        &ExecutionContext::for_tests(3),
    )
    .unwrap()
}

fn native_fit(x: &[f64], t: &[f64], y: &[f64]) -> NativeCohortFit {
    NativeCohortFit::fit(y, t, &[x], &response(), &[wire("x", "covariate", "mean", "observed")])
        .unwrap()
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(11)
}

#[test]
fn a_native_fit_is_bound_to_its_record_or_refused_for_want_of_one() {
    let (x, t, y) = data(400);
    let native = native_fit(&x, &t, &y);
    let source = source_of(native.export(), &x[..100]);
    let bytes = CohortIntervalArtifactWire::encode(&native, &source, 0.95).unwrap();
    match CohortIntervalArtifactWire::measure(&bytes, &ctx()) {
        Ok(measured) => {
            let scalar = &measured.report().scalars[0];
            assert_eq!(measured.report().route, ROUTE);
            assert_eq!(scalar.calibration.status, "calibrated");
            assert!(scalar.lower <= scalar.point && scalar.point <= scalar.upper);
            let loaded =
                MeasuredInference::consume(measured.export(), measured.expectation(), &ctx())
                    .unwrap();
            assert_eq!(loaded.export(), measured.export());
        }
        Err(error) => {
            let text = format!("{error:?}");
            assert!(text.contains("cell_not_licensed"), "{text}");
            assert!(text.contains("record_not_attesting"), "{text}");
        }
    }
}

#[test]
fn a_declared_export_with_a_forged_marker_is_refused_with_its_own_detail() {
    let (x, t, y) = data(400);
    let native = native_fit(&x, &t, &y);
    // Declared numbers under the native export's marker, over the real rows.
    let mut forged_response = response();
    forged_response.conditioning = native.export().body().response.conditioning.clone();
    let forged = CompactExport::build(ExportSpec {
        response: forged_response,
        inputs: vec![ExportInput {
            quantity: wire("x", "covariate", "mean", "observed"),
            support: InputSupport::Range { lo: -2.0, hi: 2.0 },
        }],
        terms: vec![TermSpec::Intercept, TermSpec::Linear { quantity: "x".into() }],
        coefficients: vec![1.0, 0.5],
        covariance: vec![0.0004, 0.0, 0.0, 0.0004],
        mask: vec![],
    })
    .unwrap();
    let source = source_of(&forged, &x[..100]);
    let honest = CohortIntervalArtifactWire::encode(&native, &source, 0.95).unwrap();
    let mut wire: CohortIntervalArtifactWire = antecedent_io::from_cbor(&honest).unwrap();
    wire.export = forged.to_bytes("cohort_interval_model").unwrap();
    wire.export_identity = forged.identity().to_owned();
    let bytes = antecedent_io::to_cbor(&wire).unwrap();
    let error = CohortIntervalArtifactWire::measure(&bytes, &ctx()).unwrap_err();
    let text = format!("{error:?}");
    assert!(text.contains("cohort_decision.interval_requires_native_fit"), "{text}");
    let _ = fit_cohort_linear_effect(&y, &t, &[&x]).unwrap();
}

#[test]
fn a_tampered_source_or_a_changed_identity_is_refused() {
    let (x, t, y) = data(400);
    let native = native_fit(&x, &t, &y);
    let source = source_of(native.export(), &x[..100]);
    let bytes = CohortIntervalArtifactWire::encode(&native, &source, 0.95).unwrap();
    let mut wire: CohortIntervalArtifactWire = antecedent_io::from_cbor(&bytes).unwrap();
    // A resealed export identity that does not match the embedded export.
    let mut changed_identity = wire.clone();
    changed_identity.export_identity = "0".repeat(wire.export_identity.len());
    let bad = antecedent_io::to_cbor(&changed_identity).unwrap();
    assert!(CohortIntervalArtifactWire::measure(&bad, &ctx()).is_err());
    // A flipped byte inside the embedded decision source.
    let middle = wire.source.len() / 2;
    wire.source[middle] ^= 0x55;
    let tampered = antecedent_io::to_cbor(&wire).unwrap();
    assert!(CohortIntervalArtifactWire::measure(&tampered, &ctx()).is_err());
    // A non-canonical suffix is refused.
    let mut padded = bytes;
    padded.push(0);
    assert!(CohortIntervalArtifactWire::measure(&padded, &ctx()).is_err());
}
