//! 2.3 B1 (X4): the dose-grid functional artifact through its recomputing consumer.
//!
//! The oracle is the closed-form quadratic response `m(d) = 1 + 0.5 d + 0.25 d^2`,
//! `m'(d) = 0.5 + 0.5 d` and `m(3) - m(1) = 3`; a Gaussian-kernel local quadratic
//! reproduces a quadratic exactly on noise-free data. An alternating deterministic
//! perturbation gives non-degenerate standard errors. No coverage is claimed anywhere:
//! every interval is `unmeasured`. A fresh consumer must reproduce the stored result from
//! the embedded table alone and refuse every change, including one resealed with
//! recomputed digests.
#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    reason = "exact deterministic design and bit-identical replay are the properties under test"
)]

use antecedent_io::IoError;
use antecedent_io::dose_grid_functional_artifact::{
    DOSE_GRID_ARTIFACT_VERSION, DOSE_GRID_BAND_CLOSED, DOSE_GRID_CALIBRATION,
    DOSE_GRID_INFERENCE_CLAIM, DoseClaimsWire, DoseFunctionalWire, DoseGridArtifactError,
    DoseGridArtifactWire, DoseGridConsumeLimits, DoseGridRequestWire, dose_support_table,
};

const N: usize = 201;
const GRID: [f64; 4] = [0.5, 1.0, 2.0, 3.5];

fn truth_level(d: f64) -> f64 {
    1.0 + 0.5 * d + 0.25 * d * d
}

fn truth_derivative(d: f64) -> f64 {
    0.5 + 0.5 * d
}

fn table(noise: f64) -> (Vec<f64>, Vec<f64>) {
    let dose: Vec<f64> = (0..N).map(|i| 4.0 * i as f64 / (N - 1) as f64).collect();
    let outcome = dose
        .iter()
        .enumerate()
        .map(|(i, &d)| truth_level(d) + if i % 2 == 0 { noise } else { -noise })
        .collect();
    (dose, outcome)
}

fn request(kind: &str, noise: f64) -> DoseGridRequestWire {
    let (dose, outcome) = table(noise);
    let contrast = kind == "contrast";
    DoseGridRequestWire {
        design: "randomized_dose".into(),
        functional: DoseFunctionalWire {
            kind: kind.into(),
            from: contrast.then_some(1.0),
            to: contrast.then_some(3.0),
        },
        grid: if contrast { Vec::new() } else { GRID.to_vec() },
        bandwidth: 0.4,
        bandwidth_range: (0.1, 1.0),
        minimum_local_ess: 10.0,
        claims: DoseClaimsWire {
            pointwise_level: kind != "derivative",
            derivative: kind == "derivative",
            simultaneous_band: false,
        },
        dose,
        outcome,
    }
}

fn seal(kind: &str, noise: f64) -> DoseGridArtifactWire {
    let sealed = DoseGridArtifactWire::seal(request(kind, noise));
    assert!(sealed.is_ok(), "row must seal: {:?}", sealed.as_ref().err());
    sealed.unwrap()
}

fn reseal(mut wire: DoseGridArtifactWire) -> Vec<u8> {
    wire.premises_digest = wire.expected_premises_digest().unwrap();
    wire.data_digest = wire.expected_data_digest().unwrap();
    wire.export().unwrap()
}

fn consume(bytes: &[u8]) -> Result<DoseGridArtifactWire, DoseGridArtifactError> {
    DoseGridArtifactWire::consume_typed(bytes, DoseGridConsumeLimits::default())
}

#[test]
fn b1_artifact_level_round_trips_and_matches_closed_form() {
    let sealed = seal("level", 0.0);
    let bytes = sealed.export().unwrap();
    let consumed = consume(&bytes).unwrap();
    assert_eq!(consumed, sealed);
    let expected = [1.3125, 1.75, 3.0, 5.8125];
    assert_eq!(consumed.result.levels.len(), 4);
    for (point, truth) in consumed.result.levels.iter().zip(expected) {
        assert!((point.value - truth).abs() < 1e-6, "{} vs {truth}", point.value);
        assert!((truth - truth_level(point.dose)).abs() < 1e-12);
        assert_eq!(point.support, "supported");
        assert_eq!(point.calibration, DOSE_GRID_CALIBRATION);
        assert!(!point.smoothing_bias_included);
    }
    assert!(consumed.result.derivatives.is_empty() && consumed.result.contrast.is_none());
    assert!(consumed.result.max_abs_influence_sum < 1e-9);
}

#[test]
fn b1_artifact_derivative_and_contrast_replay_to_closed_form() {
    let derivative = consume(&seal("derivative", 0.0).export().unwrap()).unwrap();
    let expected = [0.75, 1.0, 1.5, 2.25];
    for (point, truth) in derivative.result.derivatives.iter().zip(expected) {
        assert!((point.value - truth).abs() < 1e-5, "{} vs {truth}", point.value);
        assert!((truth - truth_derivative(point.dose)).abs() < 1e-12);
    }
    assert!(derivative.result.levels.is_empty());

    let contrast = consume(&seal("contrast", 0.1).export().unwrap()).unwrap();
    let item = contrast.result.contrast.as_ref().unwrap();
    assert!((item.estimate - 3.0).abs() < 0.05, "{}", item.estimate);
    assert!(item.standard_error.is_finite() && item.standard_error > 0.0);
    assert!(item.lower < item.estimate && item.estimate < item.upper);
    assert_eq!((item.from_support.as_str(), item.to_support.as_str()), ("supported", "supported"));
    assert!(contrast.result.levels.is_empty() && contrast.result.derivatives.is_empty());
    assert_eq!(contrast.result.support.len(), 2);
}

#[test]
fn b1_artifact_records_support_unmeasured_calibration_and_closed_band() {
    let wire = seal("level", 0.1);
    assert_eq!(wire.result.support.len(), GRID.len());
    for (row, dose) in wire.result.support.iter().zip(GRID) {
        assert_eq!(row.dose, dose);
        assert_eq!(row.label, "supported");
        assert!(row.local_ess >= wire.request.minimum_local_ess);
    }
    assert_eq!(wire.result.support_status, "supported");
    assert_eq!(wire.result.calibration, DOSE_GRID_CALIBRATION);
    assert_eq!(wire.result.simultaneous_band, DOSE_GRID_BAND_CLOSED);
    assert_eq!(wire.result.inference_claim, DOSE_GRID_INFERENCE_CLAIM);
    assert!(!wire.result.smoothing_bias_included);
    assert_eq!(wire.result.n_rows, N);
    assert!(wire.result.fits >= GRID.len());
    for point in &wire.result.levels {
        assert!(point.standard_error.is_finite() && point.standard_error > 0.0);
        assert_eq!(point.nominal_level, 0.95);
        assert!(point.lower < point.value && point.value < point.upper);
    }
}

#[test]
fn b1_resealed_table_change_is_refused() {
    let mut wire = seal("level", 0.1);
    wire.request.outcome[100] += 0.5;
    let error = consume(&reseal(wire)).unwrap_err();
    assert_eq!(error, DoseGridArtifactError::ResultMismatch);
    let (code, detail) = error.refusal();
    assert_eq!((code, detail.as_str()), ("invalid_argument", "dose_grid.result_replay_mismatch"));
}

#[test]
fn b1_resealed_premise_changes_are_refused() {
    let mut bandwidth = seal("level", 0.1);
    bandwidth.request.bandwidth = 0.5;
    assert_eq!(consume(&reseal(bandwidth)).unwrap_err(), DoseGridArtifactError::ResultMismatch);

    let mut grid = seal("level", 0.1);
    grid.request.grid[1] = 1.5;
    assert_eq!(consume(&reseal(grid)).unwrap_err(), DoseGridArtifactError::ResultMismatch);

    // Turning a level artifact into a derivative one changes the stored functional.
    let mut functional = seal("level", 0.1);
    functional.request.functional.kind = "derivative".into();
    functional.request.claims.derivative = true;
    assert_eq!(consume(&reseal(functional)).unwrap_err(), DoseGridArtifactError::ResultMismatch);
}

#[test]
fn b1_resealed_stored_value_changes_are_refused() {
    let mut value = seal("level", 0.1);
    value.result.levels[0].value += 1e-9;
    assert_eq!(consume(&reseal(value)).unwrap_err(), DoseGridArtifactError::ResultMismatch);

    let mut interval = seal("level", 0.1);
    interval.result.levels[2].upper += 0.25;
    assert_eq!(consume(&reseal(interval)).unwrap_err(), DoseGridArtifactError::ResultMismatch);

    let mut label = seal("level", 0.1);
    label.result.support[0].label = "weak_overlap".into();
    assert_eq!(consume(&reseal(label)).unwrap_err(), DoseGridArtifactError::ResultMismatch);
}

#[test]
fn b1_unresealed_changes_hit_the_digests() {
    let mut table_changed = seal("level", 0.1);
    table_changed.request.dose[3] += 0.001;
    assert_eq!(
        consume(&table_changed.export().unwrap()).unwrap_err(),
        DoseGridArtifactError::DataMismatch
    );
    let mut premise_changed = seal("level", 0.1);
    premise_changed.request.minimum_local_ess = 12.0;
    assert_eq!(
        consume(&premise_changed.export().unwrap()).unwrap_err(),
        DoseGridArtifactError::PremisesMismatch
    );
}

#[test]
fn b1_seal_refuses_closed_unsupported_and_uncertified_requests() {
    let detail = |request: DoseGridRequestWire| match DoseGridArtifactWire::seal(request) {
        Err(error) => error.refusal(),
        Ok(_) => panic!("the request must be refused"),
    };
    let mut band = request("level", 0.1);
    band.claims.simultaneous_band = true;
    let (code, why) = detail(band);
    assert_eq!((code, why.as_str()), ("route_not_supported", "dose_grid.simultaneous_band_closed"));

    let mut observational = request("level", 0.1);
    observational.design = "observational".into();
    let (code, why) = detail(observational);
    assert_eq!((code, why.as_str()), ("effect_not_identified", "dose_grid.graph_not_certified"));

    let mut outside = request("level", 0.1);
    outside.grid.push(10.0);
    let (code, why) = detail(outside);
    assert_eq!((code, why.as_str()), ("cell_not_licensed", "dose_grid.unsupported_dose"));

    let mut weak = request("level", 0.1);
    weak.minimum_local_ess = 1_000.0;
    let (code, why) = detail(weak);
    assert_eq!((code, why.as_str()), ("cell_not_licensed", "dose_grid.insufficient_local_weight"));

    let mut derivative_without_claim = request("derivative", 0.1);
    derivative_without_claim.claims.derivative = false;
    let (_, why) = detail(derivative_without_claim);
    assert_eq!(why, "dose_grid.derivative_without_claim");

    let mut bandwidth = request("level", 0.1);
    bandwidth.bandwidth = 2.0;
    let (_, why) = detail(bandwidth);
    assert_eq!(why, "dose_grid.bandwidth_outside_range");
}

#[test]
fn b1_unknown_version_band_marker_and_shape_are_refused() {
    let mut future = seal("level", 0.1);
    future.version = DOSE_GRID_ARTIFACT_VERSION + 1;
    let bytes = future.export().unwrap();
    assert!(matches!(
        DoseGridArtifactWire::decode(&bytes),
        Err(IoError::UnsupportedVersion { version }) if version == DOSE_GRID_ARTIFACT_VERSION + 1
    ));
    assert!(matches!(consume(&bytes), Err(DoseGridArtifactError::Undecodable(_))));

    let mut open = seal("level", 0.1);
    open.result.simultaneous_band = "open".into();
    assert!(matches!(consume(&reseal(open)), Err(DoseGridArtifactError::UnsupportedSemantics(_))));

    let mut interval = seal("level", 0.1);
    interval.result.calibration = "measured".into();
    assert!(matches!(
        consume(&reseal(interval)),
        Err(DoseGridArtifactError::UnsupportedSemantics(_))
    ));

    let mut misaligned = seal("level", 0.1);
    misaligned.request.outcome.pop();
    assert!(matches!(
        consume(&reseal(misaligned)),
        Err(DoseGridArtifactError::UnsupportedSemantics(_))
    ));

    let mut missing_feature = seal("level", 0.1);
    missing_feature.required_features.clear();
    assert!(matches!(
        consume(&reseal(missing_feature)),
        Err(DoseGridArtifactError::UnsupportedSemantics(_))
    ));
}

#[test]
fn b1_consumer_limits_bound_the_stored_table_and_grid() {
    let bytes = seal("level", 0.1).export().unwrap();
    let rows = DoseGridConsumeLimits { max_rows: 10, max_grid: 1_024 };
    assert_eq!(
        DoseGridArtifactWire::consume_typed(&bytes, rows).unwrap_err(),
        DoseGridArtifactError::LimitsExceeded("rows")
    );
    let grid = DoseGridConsumeLimits { max_rows: 100_000, max_grid: 2 };
    assert_eq!(
        DoseGridArtifactWire::consume_typed(&bytes, grid).unwrap_err(),
        DoseGridArtifactError::LimitsExceeded("grid doses")
    );
    let refused = DoseGridArtifactWire::consume_with_limits(&bytes, rows).unwrap_err();
    assert_eq!(refused.reason_code(), Some("invalid_argument"));
}

#[test]
fn b1_support_table_labels_every_dose_without_refusing() {
    let (dose, _) = table(0.0);
    let labels = dose_support_table(&dose, &[-1.0, 2.0, 10.0], 0.4, 10.0).unwrap();
    let names: Vec<&str> = labels.iter().map(|row| row.label.as_str()).collect();
    assert_eq!(names, ["outside_empirical_support", "supported", "outside_empirical_support"]);
    // Inside the observed range but below a high effective-sample-size requirement.
    let weak = dose_support_table(&dose, &[2.0], 0.4, 1_000.0).unwrap();
    assert_eq!(weak[0].label, "weak_overlap");
    assert!(weak[0].local_ess > 10.0 && weak[0].local_ess < 1_000.0);
    // Bad inputs are the row's typed invalid-request refusal.
    let (code, why) = dose_support_table(&dose, &[], 0.4, 10.0).unwrap_err().refusal();
    assert_eq!((code, why.as_str()), ("invalid_argument", "dose_grid.invalid_request"));
}
