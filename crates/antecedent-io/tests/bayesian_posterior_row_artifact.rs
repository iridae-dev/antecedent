//! The Bayesian randomized-dose grid posterior through its recomputing consumer.
//!
//! The oracle is independent of the implementation: the conjugate ridge posterior of the
//! polynomial dose curve (mean and covariance of the level, derivative and contrast rows) was
//! computed with numpy from the normal equations on the rescaled design. A fresh consumer must
//! reproduce the stored result from the embedded table alone and refuse every change,
//! including one resealed with recomputed digests. The row is not licensed: its result says so.
#![allow(
    clippy::float_cmp,
    reason = "bit-identical replay and exact stored markers are the properties under test"
)]

use antecedent_io::IoError;
use antecedent_io::bayesian_posterior_row_artifact::{
    DOSE_GRID_POSTERIOR_ESTIMATOR, DoseGridPosteriorWire, POSTERIOR_ROW_CALIBRATION,
    POSTERIOR_ROW_UNCERTAINTY, PosteriorRowArtifactError, PosteriorRowArtifactWire,
    PosteriorRowConsumeLimits, PosteriorRowRequestWire, ResidualPriorWire,
};
use antecedent_io::dose_grid_functional_artifact::DoseFunctionalWire;

const DOSE: [f64; 10] = [0.1, 0.5, 0.9, 1.4, 1.8, 2.2, 2.7, 3.1, 3.5, 3.9];
const OUTCOME: [f64; 10] = [1.2, 1.9, 2.1, 3.0, 3.2, 3.1, 3.0, 2.6, 2.2, 1.5];

fn close(left: f64, right: f64, tolerance: f64) {
    assert!((left - right).abs() <= tolerance, "{left} != {right}");
}

fn dose_grid(kind: &str, grid: &[f64]) -> PosteriorRowRequestWire {
    let contrast = kind == "contrast";
    PosteriorRowRequestWire::DoseGrid(DoseGridPosteriorWire {
        design: "randomized_dose".into(),
        functional: DoseFunctionalWire {
            kind: kind.into(),
            from: contrast.then_some(1.0),
            to: contrast.then_some(3.0),
        },
        grid: if contrast { Vec::new() } else { grid.to_vec() },
        support: (0.0, 4.0),
        coefficient_scales: vec![3.0, 2.0, 1.0],
        residual: ResidualPriorWire::KnownVariance(0.25),
        draws: 40_000,
        seed: 21,
        dose: DOSE.to_vec(),
        outcome: OUTCOME.to_vec(),
    })
}

fn seal(request: PosteriorRowRequestWire) -> PosteriorRowArtifactWire {
    let sealed = PosteriorRowArtifactWire::seal(request);
    assert!(sealed.is_ok(), "row must seal: {:?}", sealed.as_ref().err());
    sealed.unwrap()
}

fn consume(bytes: &[u8]) -> Result<PosteriorRowArtifactWire, PosteriorRowArtifactError> {
    PosteriorRowArtifactWire::consume_typed(bytes, PosteriorRowConsumeLimits::default())
}

fn reseal(mut wire: PosteriorRowArtifactWire) -> Vec<u8> {
    wire.premises_digest = wire.expected_premises_digest().unwrap();
    wire.data_digest = wire.expected_data_digest().unwrap();
    wire.export().unwrap()
}

fn refusal_of(request: PosteriorRowRequestWire) -> (&'static str, String, String) {
    match PosteriorRowArtifactWire::seal(request) {
        Err(PosteriorRowArtifactError::Refused { code, detail, message }) => {
            (code, detail, message)
        }
        other => panic!("expected a typed refusal, got {:?}", other.map(|w| w.result.estimator)),
    }
}

fn assert_unlicensed(wire: &PosteriorRowArtifactWire, estimator: &str) {
    assert_eq!(wire.result.estimator, estimator);
    assert_eq!(wire.result.uncertainty, POSTERIOR_ROW_UNCERTAINTY);
    assert_eq!(wire.result.calibration, POSTERIOR_ROW_CALIBRATION);
    assert!(!wire.result.licensed);
    let width = wire.result.coordinates.len();
    assert_eq!(
        (wire.result.mean.len(), wire.result.lower.len(), wire.result.upper.len()),
        (width, width, width)
    );
    for g in 0..width {
        assert!(
            wire.result.lower[g] < wire.result.mean[g]
                && wire.result.mean[g] < wire.result.upper[g]
        );
    }
}

#[test]
fn dose_grid_level_round_trips_and_keeps_the_joint_covariance() {
    let sealed = seal(dose_grid("level", &[0.5, 2.0, 3.5]));
    let consumed = consume(&sealed.export().unwrap()).unwrap();
    assert_eq!(consumed, sealed);
    assert_unlicensed(&consumed, DOSE_GRID_POSTERIOR_ESTIMATOR);
    let result = &consumed.result;
    assert_eq!(result.coordinates, ["level(0.5)", "level(2)", "level(3.5)"]);
    assert_eq!(result.vectors["points"], [0.5, 2.0, 3.5]);
    let exact = result.exact_mean.as_ref().unwrap();
    for (got, want) in exact.iter().zip([2.018_449_65, 2.714_270_59, 2.318_891_91]) {
        close(*got, want, 1e-7);
    }
    let covariance = result.exact_covariance.as_ref().unwrap();
    let reference = [
        0.064_536_70,
        0.016_412_15,
        -0.005_911_84,
        0.016_412_15,
        0.041_541_01,
        0.015_904_12,
        -0.005_911_84,
        0.015_904_12,
        0.063_660_80,
    ];
    for (got, want) in covariance.iter().zip(reference) {
        close(*got, want, 1e-7);
    }
    // The retained draw covariance reproduces the exact joint moments, off-diagonals
    // included; independent per-point draws would leave them near zero.
    let joint = result.covariance.as_ref().unwrap();
    for g in 0..3 {
        close(result.mean[g], exact[g], 0.01);
        for h in 0..3 {
            let scale = (covariance[g * 3 + g] * covariance[h * 3 + h]).sqrt();
            close(joint[g * 3 + h], covariance[g * 3 + h], 0.05 * scale);
        }
    }
    assert!(joint[1] > 0.01 && joint[5] > 0.01);
}

#[test]
fn dose_grid_derivative_and_contrast_replay_to_the_conjugate_reference() {
    let derivative =
        consume(&seal(dose_grid("derivative", &[0.5, 2.0, 3.5])).export().unwrap()).unwrap();
    assert_unlicensed(&derivative, DOSE_GRID_POSTERIOR_ESTIMATOR);
    let want = [0.827_613_84, 0.100_147_42, -0.627_319_00];
    for (got, want) in derivative.result.exact_mean.as_ref().unwrap().iter().zip(want) {
        close(*got, want, 1e-7);
    }
    let joint = derivative.result.covariance.as_ref().unwrap();
    close(joint[0], 0.083_616_85, 0.004);
    close(joint[2], -0.052_563_33, 0.004);

    let contrast = consume(&seal(dose_grid("contrast", &[])).export().unwrap()).unwrap();
    assert_eq!(contrast.result.coordinates, ["contrast(1->3)"]);
    close(contrast.result.exact_mean.as_ref().unwrap()[0], 0.200_294_84, 1e-7);
    close(contrast.result.exact_covariance.as_ref().unwrap()[0], 0.062_231_64, 1e-7);
    close(contrast.result.mean[0], 0.200_294_84, 0.01);
    // The joint contrast variance, not the larger one independent draws would give.
    close(contrast.result.covariance.as_ref().unwrap()[0], 0.062_231_64, 0.004);
}

#[test]
fn refusals_are_typed_and_name_the_next_attempt() {
    let PosteriorRowRequestWire::DoseGrid(mut observational) = dose_grid("level", &[0.5]);
    observational.design = "observational".into();
    let (code, detail, _) = refusal_of(PosteriorRowRequestWire::DoseGrid(observational));
    assert_eq!((code, detail.as_str()), ("effect_not_identified", "dose_grid.graph_not_certified"));

    let (_, detail, _) = refusal_of(dose_grid("level", &[0.01]));
    assert_eq!(detail, "dose_grid.unsupported_dose");
}

#[test]
fn forged_artifacts_are_refused_even_when_resealed() {
    let sealed = seal(dose_grid("level", &[0.5, 2.0, 3.5]));

    // A changed table entry without resealing breaks the data digest.
    let mut edited = sealed.clone();
    let PosteriorRowRequestWire::DoseGrid(w) = &mut edited.request;
    w.outcome[3] += 0.5;
    assert_eq!(consume(&edited.export().unwrap()), Err(PosteriorRowArtifactError::DataMismatch));
    // The same edit resealed with fresh digests is caught by the replay.
    assert_eq!(consume(&reseal(edited)), Err(PosteriorRowArtifactError::ResultMismatch));

    // A changed premise (prior scale) resealed.
    let mut prior = sealed.clone();
    let PosteriorRowRequestWire::DoseGrid(w) = &mut prior.request;
    w.coefficient_scales[2] = 0.5;
    assert_eq!(consume(&prior.export().unwrap()), Err(PosteriorRowArtifactError::PremisesMismatch));
    assert_eq!(consume(&reseal(prior)), Err(PosteriorRowArtifactError::ResultMismatch));

    // A forged stored value resealed.
    let mut forged = sealed.clone();
    forged.result.mean[0] += 0.001;
    assert_eq!(consume(&reseal(forged)), Err(PosteriorRowArtifactError::ResultMismatch));
    let mut narrowed = sealed.clone();
    narrowed.result.covariance.as_mut().unwrap()[1] = 0.0;
    assert_eq!(consume(&reseal(narrowed)), Err(PosteriorRowArtifactError::ResultMismatch));

    // A claimed licence, a foreign estimator id, a changed calibration or feature marker are
    // refused as unsupported semantics before any replay.
    let tampers: [fn(&mut PosteriorRowArtifactWire); 5] = [
        |w| w.result.licensed = true,
        |w| w.result.estimator = "bayesian.piecewise_survival".into(),
        |w| w.result.calibration = "measured".into(),
        |w| w.result.uncertainty = "interval".into(),
        |w| w.required_features = vec!["bayesian_posterior_row_v2".into()],
    ];
    for tamper in tampers {
        let mut wire = sealed.clone();
        tamper(&mut wire);
        let refused = consume(&reseal(wire)).unwrap_err();
        assert!(
            matches!(refused, PosteriorRowArtifactError::UnsupportedSemantics(_)),
            "{refused:?}"
        );
    }

    // An unknown version is refused before the payload is interpreted.
    let mut future = sealed.clone();
    future.version = 2;
    assert!(matches!(
        PosteriorRowArtifactWire::decode(&future.export().unwrap()),
        Err(IoError::UnsupportedVersion { version: 2 })
    ));
    assert!(matches!(
        consume(&future.export().unwrap()),
        Err(PosteriorRowArtifactError::Undecodable(_))
    ));
    assert!(matches!(consume(b"not cbor"), Err(PosteriorRowArtifactError::Undecodable(_))));
}

#[test]
fn consumer_limits_bound_the_replay() {
    let bytes = seal(dose_grid("level", &[0.5, 2.0, 3.5])).export().unwrap();
    let tight = PosteriorRowConsumeLimits { max_rows: 5, max_points: 1_024 };
    assert_eq!(
        PosteriorRowArtifactWire::consume_typed(&bytes, tight),
        Err(PosteriorRowArtifactError::LimitsExceeded("rows"))
    );
    let few = PosteriorRowConsumeLimits { max_rows: 100_000, max_points: 2 };
    assert!(matches!(
        PosteriorRowArtifactWire::consume_typed(&bytes, few),
        Err(PosteriorRowArtifactError::LimitsExceeded(_))
    ));
    let refused: IoError = PosteriorRowArtifactError::ResultMismatch.into();
    assert!(refused.to_string().contains("bayesian_row.result_replay_mismatch"));
}
