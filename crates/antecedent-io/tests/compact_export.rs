//! B4 compact runtime export: hand-computed point/SE values, support and mask
//! refusals, tamper and reseal refusals, bounded decode and canonical identity.

use std::io::Cursor;

use antecedent_io::compact_export::{
    CALIBRATION_STATUS, COMPACT_EXPORT_SECTION, CompactExport, CompactExportBody, ExportInput,
    ExportLimits, ExportQuery, ExportRefusal, ExportSpec, InputSupport, MaskCondition, MaskRegion,
    POINT_SCOPE, SE_BASIS, TermSpec, UNCERTAINTY_SCOPE,
};
use antecedent_io::container::{
    ArtifactManifest, CONTAINER_VERSION, CompressPolicy, EncodedArtifact, MAGIC, SectionBytes,
    section_descriptor_with_policy,
};
use antecedent_io::convert::{from_cbor, to_cbor};
use antecedent_io::quantity_wire::ScientificQuantityWire;
use antecedent_io::reader::ArtifactReader;

fn quantity(id: &str, role: &str, units: &str) -> ScientificQuantityWire {
    ScientificQuantityWire {
        version: 1,
        variable_id: id.into(),
        variable_name: id.into(),
        role: role.into(),
        units: units.into(),
        population_id: "target".into(),
        regime_id: "observed".into(),
        horizon: 0,
        functional_id: "mean".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

fn dose_input() -> ExportInput {
    ExportInput {
        quantity: quantity("x", "treatment", "mg"),
        support: InputSupport::Range { lo: 0.0, hi: 10.0 },
    }
}

fn group_input() -> ExportInput {
    ExportInput {
        quantity: quantity("g", "covariate", "category"),
        // Deliberately unsorted: build canonicalises.
        support: InputSupport::Levels { levels: vec!["c".into(), "a".into(), "b".into()] },
    }
}

fn linear(q: &str) -> TermSpec {
    TermSpec::Linear { quantity: q.into() }
}

fn one_hot(q: &str, level: &str) -> TermSpec {
    TermSpec::OneHot { quantity: q.into(), level: level.into() }
}

/// y = 1 + 2 x - 0.5 x^2 on x in [0, 10].
fn quadratic_spec() -> ExportSpec {
    ExportSpec {
        response: quantity("y", "outcome", "mmHg"),
        inputs: vec![dose_input()],
        terms: vec![
            TermSpec::Intercept,
            linear("x"),
            TermSpec::Power { quantity: "x".into(), degree: 2 },
        ],
        coefficients: vec![1.0, 2.0, -0.5],
        covariance: vec![0.04, 0.0, -0.002, 0.0, 0.01, 0.0, -0.002, 0.0, 0.0004],
        mask: vec![],
    }
}

/// y = 1 + 2 x + 0.5 [g = b] - 1 [g = c], refusing x in [8, 10] with g = c.
fn one_hot_spec() -> ExportSpec {
    ExportSpec {
        response: quantity("y", "outcome", "mmHg"),
        inputs: vec![dose_input(), group_input()],
        terms: vec![TermSpec::Intercept, linear("x"), one_hot("g", "b"), one_hot("g", "c")],
        coefficients: vec![1.0, 2.0, 0.5, -1.0],
        covariance: vec![
            0.04, 0.0, 0.01, 0.0, //
            0.0, 0.01, 0.0, 0.0, //
            0.01, 0.0, 0.09, 0.0, //
            0.0, 0.0, 0.0, 0.16,
        ],
        mask: vec![MaskRegion {
            id: "high_dose_group_c".into(),
            conditions: vec![
                MaskCondition {
                    quantity: "x".into(),
                    within: InputSupport::Range { lo: 8.0, hi: 10.0 },
                },
                MaskCondition {
                    quantity: "g".into(),
                    within: InputSupport::Levels { levels: vec!["c".into()] },
                },
            ],
        }],
    }
}

fn approx(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
}

fn x_query(x: f64) -> ExportQuery {
    ExportQuery::new().with_number("x", x)
}

fn xg_query(x: f64, g: &str) -> ExportQuery {
    ExportQuery::new().with_number("x", x).with_level("g", g)
}

fn limits() -> ExportLimits {
    ExportLimits::default()
}

fn read_body(bytes: &[u8]) -> (ArtifactManifest, CompactExportBody) {
    let mut reader = ArtifactReader::open_seek(Cursor::new(bytes)).unwrap();
    let manifest = reader.manifest().clone();
    let section = reader.load_section(COMPACT_EXPORT_SECTION).unwrap();
    (manifest, from_cbor(section.as_bytes()).unwrap())
}

fn write_body(mut manifest: ArtifactManifest, body: &CompactExportBody) -> Vec<u8> {
    let payload = to_cbor(body).unwrap();
    manifest.sections = vec![section_descriptor_with_policy(
        COMPACT_EXPORT_SECTION,
        "application/cbor",
        &payload,
        CompressPolicy::Never,
    )];
    let encoded = EncodedArtifact {
        manifest,
        sections: vec![SectionBytes::new(COMPACT_EXPORT_SECTION, payload)],
    };
    let mut out = Vec::new();
    encoded.write_to(&mut out).unwrap();
    out
}

/// Mutate the stored body and reseal the container (fresh section checksum).
/// With `reseal_digests` the forger also recomputes the premises, data and
/// identity digests.
fn forge(bytes: &[u8], reseal_digests: bool, mutate: fn(&mut CompactExportBody)) -> Vec<u8> {
    let (manifest, mut body) = read_body(bytes);
    mutate(&mut body);
    if reseal_digests {
        body.seal();
    }
    write_body(manifest, &body)
}

fn refusal(result: Result<CompactExport, ExportRefusal>) -> ExportRefusal {
    result.unwrap_err()
}

/// A stale-digest forgery and a fully resealed forgery are both refused.
fn assert_tamper_refused(
    export: &CompactExport,
    mutate: fn(&mut CompactExportBody),
    stale_detail: &str,
) {
    let bytes = export.to_bytes("tamper-test").unwrap();
    let identity = export.identity().to_owned();

    let stale = forge(&bytes, false, mutate);
    assert_eq!(refusal(CompactExport::consume(&stale, &limits(), &identity)).detail, stale_detail);

    let resealed = forge(&bytes, true, mutate);
    let (_, forged_body) = read_body(&resealed);
    assert_ne!(forged_body.identity, identity, "the mutation must change the identity");
    let err = refusal(CompactExport::consume(&resealed, &limits(), &identity));
    assert_eq!(err.detail, "compact_export.identity_unexpected");
    assert_eq!(err.code, "invalid_argument");
}

#[test]
fn b4_compact_quadratic_point_and_se_match_hand_values() {
    let export = CompactExport::build(quadratic_spec()).unwrap();
    // x = 2: phi = (1, 2, 4); b'phi = 1 + 4 - 2 = 3.
    // phi'V phi = 0.04 + 0.01*4 + 0.0004*16 + 2*(-0.002)*4 = 0.0704 = 0.0064 * 11.
    let at_two = export.evaluate(&x_query(2.0)).unwrap();
    approx(at_two.point, 3.0);
    approx(at_two.model_based_se, 0.08 * 11f64.sqrt());
    // x = 0: phi = (1, 0, 0); point 1, se sqrt(0.04) = 0.2.
    let at_zero = export.evaluate(&x_query(0.0)).unwrap();
    approx(at_zero.point, 1.0);
    approx(at_zero.model_based_se, 0.2);
    // x = 10 (closed support bound): phi = (1, 10, 100); point 1 + 20 - 50 = -29;
    // phi'V phi = 0.04 + 1 + 4 - 0.4 = 4.64, sqrt = 2.154065922853802.
    let at_ten = export.evaluate(&x_query(10.0)).unwrap();
    approx(at_ten.point, -29.0);
    assert!((at_ten.model_based_se - 2.154_065_922_853_802).abs() < 1e-9);
}

#[test]
fn b4_compact_scope_states_point_and_model_based_se_only() {
    let export = CompactExport::build(quadratic_spec()).unwrap();
    let result = export.evaluate(&x_query(2.0)).unwrap();
    assert_eq!(result.se_basis, SE_BASIS);
    assert_eq!(result.se_basis, "model_based_sqrt_phi_v_phi");
    assert_eq!(result.calibration, CALIBRATION_STATUS);
    assert_eq!(result.calibration, "calibration unmeasured");
    assert_eq!(result.export_identity, export.identity());
    let scope = &export.body().scope;
    assert_eq!(scope.point, POINT_SCOPE);
    assert_eq!(scope.uncertainty, UNCERTAINTY_SCOPE);
    assert!(scope.point.starts_with("point prediction"));
    assert!(scope.uncertainty.contains("model-based standard error"));
    assert!(scope.uncertainty.contains("not an interval"));
    assert_eq!(scope.calibration, "calibration unmeasured");
}

#[test]
fn b4_compact_one_hot_terms_select_the_level_with_hand_values() {
    let export = CompactExport::build(one_hot_spec()).unwrap();
    // g = b: phi = (1, 1, 1, 0); point 1 + 2 + 0.5 = 3.5;
    // phi'V phi = 0.04 + 0.01 + 0.09 + 2*0.01 = 0.16, se 0.4.
    let b = export.evaluate(&xg_query(1.0, "b")).unwrap();
    approx(b.point, 3.5);
    approx(b.model_based_se, 0.4);
    // g = c: phi = (1, 1, 0, 1); point 1 + 2 - 1 = 2; variance 0.04 + 0.01 + 0.16 = 0.21.
    let c = export.evaluate(&xg_query(1.0, "c")).unwrap();
    approx(c.point, 2.0);
    approx(c.model_based_se, 0.21f64.sqrt());
    // g = a (reference level): phi = (1, 1, 0, 0); point 3; variance 0.05.
    let a = export.evaluate(&xg_query(1.0, "a")).unwrap();
    approx(a.point, 3.0);
    approx(a.model_based_se, 0.05f64.sqrt());
    // Levels were canonicalised even though supplied as c, a, b.
    assert_eq!(
        export.body().inputs[0].support,
        InputSupport::Levels { levels: vec!["a".into(), "b".into(), "c".into()] }
    );
}

#[test]
fn b4_compact_out_of_support_queries_refuse_with_exact_details() {
    let export = CompactExport::build(one_hot_spec()).unwrap();
    let expect = |query: ExportQuery, detail: &'static str, subject: &str| {
        let err = export.evaluate(&query).unwrap_err();
        assert_eq!(
            err,
            ExportRefusal { code: "cell_not_licensed", detail, subject: subject.into() }
        );
    };
    expect(xg_query(10.5, "a"), "compact_export.out_of_support", "x");
    expect(xg_query(-0.001, "a"), "compact_export.out_of_support", "x");
    expect(xg_query(1.0, "d"), "compact_export.out_of_support", "g");
    expect(ExportQuery::new().with_number("x", 1.0), "compact_export.missing_quantity", "g");
    expect(xg_query(1.0, "a").with_number("z", 3.0), "compact_export.unknown_quantity", "z");
    expect(xg_query(f64::NAN, "a"), "compact_export.non_finite_input", "x");
    expect(xg_query(f64::INFINITY, "a"), "compact_export.non_finite_input", "x");
    expect(
        ExportQuery::new().with_level("x", "a").with_level("g", "a"),
        "compact_export.query_type_mismatch",
        "x",
    );
    // The closed bounds themselves are inside the support.
    export.evaluate(&xg_query(0.0, "a")).unwrap();
    export.evaluate(&xg_query(10.0, "a")).unwrap();
}

#[test]
fn b4_compact_masked_region_refuses_even_inside_the_support() {
    let export = CompactExport::build(one_hot_spec()).unwrap();
    let masked = ExportRefusal {
        code: "cell_not_licensed",
        detail: "compact_export.masked_region",
        subject: "high_dose_group_c".into(),
    };
    assert_eq!(export.evaluate(&xg_query(9.0, "c")).unwrap_err(), masked);
    assert_eq!(export.evaluate(&xg_query(8.0, "c")).unwrap_err(), masked);
    assert_eq!(export.evaluate(&xg_query(10.0, "c")).unwrap_err(), masked);
    // Outside either condition the query is answered.
    export.evaluate(&xg_query(7.999, "c")).unwrap();
    export.evaluate(&xg_query(9.0, "b")).unwrap();
}

#[test]
fn b4_compact_verifier_round_trip_matches_the_producer() {
    let export = CompactExport::build(one_hot_spec()).unwrap();
    let bytes = export.to_bytes("round-trip").unwrap();
    let consumed = CompactExport::consume(&bytes, &limits(), export.identity()).unwrap();
    assert_eq!(consumed.body(), export.body());
    assert_eq!(consumed.identity(), export.identity());
    assert_eq!(consumed.identity().len(), 64);
    for (x, g) in [(1.0, "a"), (3.0, "b"), (7.5, "c")] {
        assert_eq!(
            consumed.evaluate(&xg_query(x, g)).unwrap(),
            export.evaluate(&xg_query(x, g)).unwrap()
        );
    }
    // The consumed export keeps refusing the same regions.
    assert_eq!(
        consumed.evaluate(&xg_query(9.0, "c")).unwrap_err().detail,
        "compact_export.masked_region"
    );
    assert_eq!(
        consumed.evaluate(&xg_query(11.0, "a")).unwrap_err().detail,
        "compact_export.out_of_support"
    );
}

#[test]
fn b4_compact_wrong_expected_identity_refused() {
    let export = CompactExport::build(quadratic_spec()).unwrap();
    let bytes = export.to_bytes("expected").unwrap();
    let err = refusal(CompactExport::consume(&bytes, &limits(), "deadbeef"));
    assert_eq!(err.detail, "compact_export.identity_unexpected");
    assert_eq!(err.subject, "deadbeef");
    // A different model has a different identity and is refused under the first.
    let mut other = quadratic_spec();
    other.coefficients[0] = 1.0 + 1e-9;
    let other = CompactExport::build(other).unwrap();
    assert_ne!(other.identity(), export.identity());
    let other_bytes = other.to_bytes("expected").unwrap();
    let err = refusal(CompactExport::consume(&other_bytes, &limits(), export.identity()));
    assert_eq!(err.detail, "compact_export.identity_unexpected");
}

#[test]
fn b4_compact_tampered_coefficient_refused_even_when_resealed() {
    let export = CompactExport::build(quadratic_spec()).unwrap();
    assert_tamper_refused(
        &export,
        |body| body.coefficients[1] += 0.5,
        "compact_export.data_digest_mismatch",
    );
}

#[test]
fn b4_compact_tampered_covariance_refused_even_when_resealed() {
    let export = CompactExport::build(quadratic_spec()).unwrap();
    // Shrinking a variance keeps the matrix valid, so only the identity catches it.
    assert_tamper_refused(
        &export,
        |body| body.covariance[0] = 0.0001,
        "compact_export.data_digest_mismatch",
    );
}

#[test]
fn b4_compact_tampered_support_refused_even_when_resealed() {
    let export = CompactExport::build(quadratic_spec()).unwrap();
    assert_tamper_refused(
        &export,
        |body| body.inputs[0].support = InputSupport::Range { lo: 0.0, hi: 100.0 },
        "compact_export.premises_digest_mismatch",
    );
}

#[test]
fn b4_compact_tampered_mask_refused_even_when_resealed() {
    let export = CompactExport::build(one_hot_spec()).unwrap();
    // Dropping the refusal region would silently license a refused query.
    assert_tamper_refused(
        &export,
        |body| body.mask.clear(),
        "compact_export.premises_digest_mismatch",
    );
    assert_tamper_refused(
        &export,
        |body| body.mask[0].conditions[1].within = InputSupport::Range { lo: 9.5, hi: 10.0 },
        "compact_export.premises_digest_mismatch",
    );
}

#[test]
fn b4_compact_tampered_units_scope_and_term_spec_refused_even_when_resealed() {
    let export = CompactExport::build(one_hot_spec()).unwrap();
    assert_tamper_refused(
        &export,
        |body| body.response.units = "g".into(),
        "compact_export.premises_digest_mismatch",
    );
    assert_tamper_refused(
        &export,
        |body| body.scope.calibration = "calibrated".into(),
        "compact_export.premises_digest_mismatch",
    );
    assert_tamper_refused(
        &export,
        |body| body.terms[2] = one_hot("g", "a"),
        "compact_export.premises_digest_mismatch",
    );
}

#[test]
fn b4_compact_tampered_identity_string_refused() {
    let export = CompactExport::build(quadratic_spec()).unwrap();
    let bytes = export.to_bytes("identity").unwrap();
    let forged = forge(&bytes, false, |body| body.identity = "0".repeat(64));
    // Even a consumer that pins the forged string is refused: the identity must
    // be the recomputed digest of the stored fields.
    for pinned in [export.identity().to_owned(), "0".repeat(64)] {
        let err = refusal(CompactExport::consume(&forged, &limits(), &pinned));
        assert_eq!(err.detail, "compact_export.identity_mismatch");
    }
}

#[test]
fn b4_compact_resealed_forgery_passes_only_under_the_forgers_own_identity() {
    let export = CompactExport::build(quadratic_spec()).unwrap();
    let bytes = export.to_bytes("pin").unwrap();
    let forged = forge(&bytes, true, |body| body.coefficients[0] = 9.0);
    let (_, forged_body) = read_body(&forged);
    // The producer's retained identity refuses it; only the forger's own would pass,
    // which is why the consumer must retain the identity independently.
    assert!(CompactExport::consume(&forged, &limits(), export.identity()).is_err());
    let accepted = CompactExport::consume(&forged, &limits(), &forged_body.identity).unwrap();
    approx(accepted.evaluate(&x_query(0.0)).unwrap().point, 9.0);
}

#[test]
fn b4_compact_verifier_validates_semantics_even_when_identity_is_pinned() {
    type Case = (fn(&mut CompactExportBody), &'static str);
    let export = CompactExport::build(quadratic_spec()).unwrap();
    let bytes = export.to_bytes("semantics").unwrap();
    let cases: [Case; 4] = [
        (|body| body.terms.swap(0, 1), "compact_export.terms_not_canonical"),
        (|body| body.covariance[1] = 0.5, "compact_export.covariance_asymmetric"),
        (|body| body.coefficients[2] = f64::INFINITY, "compact_export.non_finite_value"),
        (|body| body.covariance[4] = -0.01, "compact_export.covariance_not_psd"),
    ];
    for (mutate, detail) in cases {
        let forged = forge(&bytes, true, mutate);
        let (_, body) = read_body(&forged);
        let err = refusal(CompactExport::consume(&forged, &limits(), &body.identity));
        assert_eq!(err.detail, detail);
    }
}

#[test]
fn b4_compact_unknown_versions_refused() {
    let export = CompactExport::build(quadratic_spec()).unwrap();
    let bytes = export.to_bytes("version").unwrap();
    // Body version, resealed with recomputed digests and a fresh checksum.
    let forged = forge(&bytes, true, |body| body.version = 2);
    let (_, body) = read_body(&forged);
    let err = refusal(CompactExport::consume(&forged, &limits(), &body.identity));
    assert_eq!(err.detail, "compact_export.unsupported_version");
    assert_eq!(err.subject, "2");
    // Container version bytes follow the 8-byte magic.
    let mut container = bytes.clone();
    container[8..12].copy_from_slice(&(CONTAINER_VERSION + 1).to_le_bytes());
    let err = refusal(CompactExport::consume(&container, &limits(), export.identity()));
    assert_eq!(err.detail, "compact_export.unsupported_version");
}

#[test]
fn b4_compact_oversized_claim_refused_before_decode() {
    let export = CompactExport::build(quadratic_spec()).unwrap();
    let bytes = export.to_bytes("oversized").unwrap();
    let (mut manifest, body) = read_body(&bytes);
    let payload = to_cbor(&body).unwrap();
    let mut descriptor = section_descriptor_with_policy(
        COMPACT_EXPORT_SECTION,
        "application/cbor",
        &payload,
        CompressPolicy::Never,
    );
    // The artifact claims a 64 MiB payload while shipping a few hundred bytes.
    descriptor.uncompressed_size = 64 * 1024 * 1024;
    manifest.sections = vec![descriptor];
    let manifest_bytes = to_cbor(&manifest).unwrap();
    let mut raw = Vec::new();
    raw.extend_from_slice(MAGIC);
    raw.extend_from_slice(&CONTAINER_VERSION.to_le_bytes());
    raw.extend_from_slice(&u32::try_from(manifest_bytes.len()).unwrap().to_le_bytes());
    raw.extend_from_slice(&manifest_bytes);
    raw.extend_from_slice(&u32::try_from(payload.len()).unwrap().to_le_bytes());
    raw.extend_from_slice(&payload);
    let err = refusal(CompactExport::consume(&raw, &limits(), export.identity()));
    assert_eq!(err.detail, "compact_export.oversized");
    assert_eq!(err.subject, "declared_uncompressed_size");
}

#[test]
fn b4_compact_limits_bound_bytes_and_counts() {
    let export = CompactExport::build(one_hot_spec()).unwrap();
    let bytes = export.to_bytes("limits").unwrap();
    let id = export.identity();
    let tight_bytes = ExportLimits { max_bytes: 64, ..limits() };
    assert_eq!(
        refusal(CompactExport::consume(&bytes, &tight_bytes, id)).detail,
        "compact_export.oversized"
    );
    let tight_terms = ExportLimits { max_terms: 2, ..limits() };
    assert_eq!(
        refusal(CompactExport::consume(&bytes, &tight_terms, id)).detail,
        "compact_export.too_many_terms"
    );
    let tight_inputs = ExportLimits { max_inputs: 1, ..limits() };
    assert_eq!(
        refusal(CompactExport::consume(&bytes, &tight_inputs, id)).detail,
        "compact_export.too_many_inputs"
    );
    let tight_mask = ExportLimits { max_mask_regions: 0, ..limits() };
    assert_eq!(
        refusal(CompactExport::consume(&bytes, &tight_mask, id)).detail,
        "compact_export.too_many_mask_regions"
    );
    let tight_levels = ExportLimits { max_levels: 2, ..limits() };
    assert_eq!(
        refusal(CompactExport::consume(&bytes, &tight_levels, id)).detail,
        "compact_export.too_many_levels"
    );
    CompactExport::consume(&bytes, &limits(), id).unwrap();
}

#[test]
fn b4_compact_term_order_permutation_has_one_canonical_identity() {
    let canonical = CompactExport::build(quadratic_spec()).unwrap();
    // Given order (Power, Intercept, Linear) is canonical indices [2, 0, 1].
    let order = [2usize, 0, 1];
    let base = quadratic_spec();
    let n = base.terms.len();
    let mut permuted = base.clone();
    permuted.terms = order.iter().map(|&i| base.terms[i].clone()).collect();
    permuted.coefficients = order.iter().map(|&i| base.coefficients[i]).collect();
    permuted.covariance = Vec::new();
    for &i in &order {
        for &j in &order {
            permuted.covariance.push(base.covariance[i * n + j]);
        }
    }
    let shuffled = CompactExport::build(permuted).unwrap();
    assert_eq!(shuffled.identity(), canonical.identity());
    assert_eq!(shuffled.body(), canonical.body());
    assert_eq!(
        shuffled.evaluate(&x_query(2.0)).unwrap(),
        canonical.evaluate(&x_query(2.0)).unwrap()
    );

    // Reversing terms and inputs of the one-hot model changes nothing either.
    let base = one_hot_spec();
    let n = base.terms.len();
    let order: Vec<usize> = (0..n).rev().collect();
    let mut reversed = base.clone();
    reversed.terms = order.iter().map(|&i| base.terms[i].clone()).collect();
    reversed.coefficients = order.iter().map(|&i| base.coefficients[i]).collect();
    reversed.covariance = order
        .iter()
        .flat_map(|&i| order.iter().map(move |&j| (i, j)))
        .map(|(i, j)| base.covariance[i * n + j])
        .collect();
    reversed.inputs.reverse();
    let reference = CompactExport::build(base).unwrap();
    let reversed = CompactExport::build(reversed).unwrap();
    assert_eq!(reversed.identity(), reference.identity());

    // A genuinely different coefficient changes the identity.
    let mut changed = one_hot_spec();
    changed.coefficients[2] = 0.75;
    assert_ne!(CompactExport::build(changed).unwrap().identity(), reference.identity());
}

#[test]
fn b4_compact_build_refuses_malformed_specifications() {
    let detail = |spec: ExportSpec| CompactExport::build(spec).unwrap_err().detail;

    let mut spec = quadratic_spec();
    spec.covariance[1] = 0.5;
    assert_eq!(detail(spec), "compact_export.covariance_asymmetric");

    let mut spec = quadratic_spec();
    spec.covariance[0] = -0.04;
    assert_eq!(detail(spec), "compact_export.covariance_not_psd");

    // |V02| = 0.01 exceeds sqrt(0.04 * 0.0004) = 0.004.
    let mut spec = quadratic_spec();
    spec.covariance[2] = 0.01;
    spec.covariance[6] = 0.01;
    assert_eq!(detail(spec), "compact_export.covariance_not_psd");

    let mut spec = quadratic_spec();
    spec.coefficients[0] = f64::INFINITY;
    assert_eq!(detail(spec), "compact_export.non_finite_value");

    let mut spec = quadratic_spec();
    spec.covariance.pop();
    assert_eq!(detail(spec), "compact_export.dimension_mismatch");

    let mut spec = quadratic_spec();
    spec.coefficients.pop();
    assert_eq!(detail(spec), "compact_export.dimension_mismatch");

    let mut spec = quadratic_spec();
    spec.terms[2] = linear("x");
    assert_eq!(detail(spec), "compact_export.duplicate_term");

    let mut spec = quadratic_spec();
    spec.terms[1] = linear("z");
    assert_eq!(detail(spec), "compact_export.unknown_quantity");

    let mut spec = quadratic_spec();
    spec.terms[1] = one_hot("x", "a");
    assert_eq!(detail(spec), "compact_export.term_support_mismatch");

    let mut spec = one_hot_spec();
    spec.terms[2] = one_hot("g", "d");
    assert_eq!(detail(spec), "compact_export.unknown_level");

    let mut spec = quadratic_spec();
    spec.terms[2] = TermSpec::Power { quantity: "x".into(), degree: 1 };
    assert_eq!(detail(spec), "compact_export.invalid_degree");

    let mut spec = quadratic_spec();
    spec.terms[2] = TermSpec::Power { quantity: "x".into(), degree: 9 };
    assert_eq!(detail(spec), "compact_export.invalid_degree");

    let mut spec = quadratic_spec();
    spec.inputs[0].support = InputSupport::Range { lo: 5.0, hi: 1.0 };
    assert_eq!(detail(spec), "compact_export.invalid_support");

    let mut spec = one_hot_spec();
    spec.mask[0].conditions[0].quantity = "z".into();
    assert_eq!(detail(spec), "compact_export.unknown_quantity");

    let mut spec = one_hot_spec();
    spec.mask[0].conditions[1].within = InputSupport::Levels { levels: vec!["zz".into()] };
    assert_eq!(detail(spec), "compact_export.unknown_level");

    let mut spec = one_hot_spec();
    spec.mask[0].conditions[0].within = InputSupport::Levels { levels: vec!["a".into()] };
    assert_eq!(detail(spec), "compact_export.mask_support_mismatch");

    let mut spec = quadratic_spec();
    spec.response.units = String::new();
    assert_eq!(detail(spec), "compact_export.invalid_quantity");
}
