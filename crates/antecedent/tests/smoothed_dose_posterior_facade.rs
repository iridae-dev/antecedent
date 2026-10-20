//! The Bayesian smoothed dose-response transport posterior through the public facade.
//!
//! The data-generating process is the closed-form one of the smoothed-dose lifecycle
//! fixtures: `y = a^2 + z (1 + a / 2) + e` with unit noise, a randomized dose with known
//! density, and a baseline standardization certificate. For the observed target sample the
//! smoothed target is `psi_h(a) = a^2 + h^2 / 5 + mean_T(z) (1 + a / 2)`, derived from the
//! Epanechnikov moments `1/5` and `integral K_h(a - t) t dt = a`; it is computed here from
//! the rows, never from the implementation. The posterior must place that value inside its
//! credible interval at every grid dose under both sampling designs, keep the joint
//! covariance across the grid (checked against the exact Gaussian moments of the declared
//! conjugate model), export an artifact an independent loader recomputes, and refuse every
//! forged or resealed change. The row is unlicensed: its calibration is `unmeasured`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::needless_range_loop,
    reason = "integration test: exact markers are compared and fixture indices are small"
)]

#[path = "smoothed_dose_dgp/mod.rs"]
mod dgp;

use antecedent::analysis::bayesian_posterior_row::{
    POSTERIOR_ROW_CALIBRATION, POSTERIOR_ROW_UNCERTAINTY, PosteriorRowArtifactError,
    PosteriorRowConsumeLimits, ResidualPriorWire,
};
use antecedent::analysis::consume_smoothed_dose_posterior_artifact;
use antecedent::{PreparedSmoothedDose, StudyBuilder};
use antecedent_core::ExecutionContext;
use antecedent_estimate::SmoothedDoseOptions;
use antecedent_estimate::smoothed_dose::DoseBasis;
use antecedent_io::IoError;
use antecedent_io::smoothed_dose_posterior_artifact::{
    SMOOTHED_DOSE_POSTERIOR_ESTIMATOR, SmoothedDosePosteriorArtifactWire,
    SmoothedDosePosteriorModelWire,
};
use dgp::{Design, Scenario, diagram, draw, query};

const GRID: [f64; 3] = [1.0, 2.0, 3.0];
const H: f64 = 0.5;

fn model(draws: usize) -> SmoothedDosePosteriorModelWire {
    SmoothedDosePosteriorModelWire {
        basis: DoseBasis::default(),
        intercept_scale: 10.0,
        coefficient_scale: 10.0,
        residual: ResidualPriorWire::KnownVariance(1.0),
        min_local_rows: 50,
        draws,
        seed: 7,
    }
}

fn prepare(design: Design, seed: u64) -> (PreparedSmoothedDose, f64) {
    let input = draw(design, Scenario::Good, 4000, 3000, seed);
    let target: Vec<f64> = input.covariates[0]
        .iter()
        .zip(&input.source)
        .filter_map(|(z, in_trial)| (!*in_trial).then_some(*z))
        .collect();
    let target_mean = target.iter().sum::<f64>() / target.len() as f64;
    let (diagram, names) = diagram();
    let options = SmoothedDoseOptions { folds: 3, ..SmoothedDoseOptions::default() };
    let prepared = StudyBuilder::smoothed_dose_transport(
        diagram,
        query(&GRID, H),
        input,
        options,
        names,
        &ExecutionContext::for_tests(seed),
    )
    .unwrap();
    (prepared, target_mean)
}

fn smoothed_truth(a: f64, target_mean: f64) -> f64 {
    a * a + H * H / 5.0 + target_mean * (1.0 + a / 2.0)
}

fn reseal(mut wire: SmoothedDosePosteriorArtifactWire) -> Vec<u8> {
    wire.premises_digest = wire.expected_premises_digest().unwrap();
    wire.data_digest = wire.expected_data_digest().unwrap();
    wire.export().unwrap()
}

fn consume(bytes: &[u8]) -> Result<SmoothedDosePosteriorArtifactWire, PosteriorRowArtifactError> {
    SmoothedDosePosteriorArtifactWire::consume_typed(bytes, PosteriorRowConsumeLimits::default())
}

#[test]
fn posterior_covers_the_closed_form_smoothed_target_under_both_designs() {
    for design in [Design::NestedCohort, Design::IndependentSamples] {
        let (prepared, target_mean) = prepare(design, 11);
        let posterior = prepared.posterior(model(20_000)).unwrap();
        let result = posterior.result();
        assert_eq!(result.estimator, SMOOTHED_DOSE_POSTERIOR_ESTIMATOR);
        assert_eq!(result.uncertainty, POSTERIOR_ROW_UNCERTAINTY);
        assert_eq!(result.calibration, POSTERIOR_ROW_CALIBRATION);
        assert!(!result.licensed);
        assert_eq!(result.coordinates, ["psi_h(1)", "psi_h(2)", "psi_h(3)"]);
        assert_eq!(result.vectors["grid"], GRID);
        let exact = result.exact_mean.as_ref().unwrap();
        let covariance = result.exact_covariance.as_ref().unwrap();
        let joint = result.covariance.as_ref().unwrap();
        for g in 0..3 {
            let truth = smoothed_truth(GRID[g], target_mean);
            let sd = covariance[g * 3 + g].sqrt();
            assert!(
                (exact[g] - truth).abs() < 4.0 * sd + 0.02,
                "{design:?} a={}: {} vs {truth} (sd {sd})",
                GRID[g],
                exact[g]
            );
            // The population smoothed curve of the DGP has the same closed form at the
            // declared target mean 0.3.
            assert!((exact[g] - Scenario::Good.truth(GRID[g], H)).abs() < 0.5);
            assert!((result.mean[g] - exact[g]).abs() < 0.01);
            assert!(result.lower[g] < truth && truth < result.upper[g], "{design:?} a={}", GRID[g]);
            assert!((result.lower[g] - (exact[g] - 1.96 * sd)).abs() < 0.02);
            assert!((result.upper[g] - (exact[g] + 1.96 * sd)).abs() < 0.02);
            // The retained draws keep the cross-dose covariance of the exact posterior.
            for h in 0..3 {
                let scale = (covariance[g * 3 + g] * covariance[h * 3 + h]).sqrt();
                assert!(
                    (joint[g * 3 + h] - covariance[g * 3 + h]).abs() < 0.05 * scale,
                    "{design:?} ({g},{h}): {} vs {}",
                    joint[g * 3 + h],
                    covariance[g * 3 + h]
                );
            }
        }
        assert!((joint[1] - joint[3]).abs() < 1e-12);
        if design == Design::IndependentSamples {
            assert_eq!(result.extras["source_rows"], 4000.0);
            assert_eq!(result.extras["target_rows"], 3000.0);
        }
        assert_eq!(result.extras["bandwidth"], H);
    }
}

#[test]
fn artifact_round_trips_through_an_independent_loader_and_refuses_forgeries() {
    let (prepared, _) = prepare(Design::IndependentSamples, 5);
    let posterior = prepared.posterior(model(2_000)).unwrap();
    let bytes = posterior.export().unwrap();

    let loaded =
        consume_smoothed_dose_posterior_artifact(&bytes, PosteriorRowConsumeLimits::default())
            .unwrap();
    assert_eq!(loaded.result(), posterior.result());
    let direct = consume(&bytes).unwrap();
    assert_eq!(direct.result, *posterior.result());
    assert_eq!(direct.certificate.formula, "standardize");
    assert_eq!(direct.certificate.over, [0]);

    // A changed row without resealing breaks the data digest; resealed, the replay differs.
    let mut edited = direct.clone();
    edited.input.outcome[0] += 1.0;
    assert_eq!(
        consume(&edited.export().unwrap()).unwrap_err(),
        PosteriorRowArtifactError::DataMismatch
    );
    assert_eq!(consume(&reseal(edited)).unwrap_err(), PosteriorRowArtifactError::ResultMismatch);

    // Changed premises: the prior and the seed.
    let mut prior = direct.clone();
    prior.model.intercept_scale = 1.0;
    assert_eq!(
        consume(&prior.export().unwrap()).unwrap_err(),
        PosteriorRowArtifactError::PremisesMismatch
    );
    assert_eq!(consume(&reseal(prior)).unwrap_err(), PosteriorRowArtifactError::ResultMismatch);
    let mut seed = direct.clone();
    seed.model.seed += 1;
    assert_eq!(consume(&reseal(seed)).unwrap_err(), PosteriorRowArtifactError::ResultMismatch);

    // A forged stored value or joint covariance.
    let mut mean = direct.clone();
    mean.result.mean[1] += 1e-3;
    assert_eq!(consume(&reseal(mean)).unwrap_err(), PosteriorRowArtifactError::ResultMismatch);
    let mut joint = direct.clone();
    joint.result.covariance.as_mut().unwrap()[1] = 0.0;
    assert_eq!(consume(&reseal(joint)).unwrap_err(), PosteriorRowArtifactError::ResultMismatch);

    // A certificate that is not the one the graph and query derive.
    let mut certificate = direct.clone();
    certificate.certificate.formula = "direct".into();
    certificate.certificate.over.clear();
    let refused = consume(&reseal(certificate)).unwrap_err();
    assert!(
        matches!(&refused, PosteriorRowArtifactError::Refused { detail, .. }
            if detail == "smoothed_dose_posterior.certificate_mismatch"),
        "{refused:?}"
    );

    // A claimed licence is refused before any replay.
    let mut licensed = direct.clone();
    licensed.result.licensed = true;
    assert!(matches!(
        consume(&reseal(licensed)),
        Err(PosteriorRowArtifactError::UnsupportedSemantics(_))
    ));

    // The consumer's own bound applies to the stored rows.
    let tight = PosteriorRowConsumeLimits { max_rows: 100, max_points: 1_024 };
    assert!(matches!(
        SmoothedDosePosteriorArtifactWire::consume_typed(&bytes, tight),
        Err(PosteriorRowArtifactError::LimitsExceeded("rows"))
    ));
}

#[test]
fn invalid_models_and_thin_windows_refuse_with_their_reason() {
    let (prepared, _) = prepare(Design::IndependentSamples, 3);
    let refused = |model: SmoothedDosePosteriorModelWire| match prepared.posterior(model) {
        Err(IoError::Refused { code, message }) => (code, message),
        other => panic!("expected a coded refusal, got {:?}", other.map(|r| r.result().draws)),
    };
    let mut thin = model(400);
    thin.min_local_rows = 100_000;
    let (code, message) = refused(thin);
    assert_eq!(code, "invalid_argument");
    assert!(message.contains("local support failure"), "{message}");

    let mut few = model(400);
    few.draws = 10;
    assert!(refused(few).1.contains("between 200 and 100000"));

    let mut flat = model(400);
    flat.coefficient_scale = 0.0;
    assert!(refused(flat).1.contains("finite and positive"));
}
