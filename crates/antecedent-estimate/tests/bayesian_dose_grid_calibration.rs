//! Coverage designs for the pointwise 95% credible intervals of the Bayesian randomized dose
//! grid (`bayesian.dose_grid`), on the exactly quadratic randomized-dose design the frequentist
//! local-quadratic row is measured on: `D ~ Uniform[0, 4]`, independent `N(0, 1)` noise and the
//! mean `1 + 0.5 d + 0.25 d^2`. The declared polynomial has degree two, so the model is correct;
//! the prior is a ridge prior of scale 10 on every rescaled coefficient and an inverse-gamma
//! prior on the residual variance. The level at 2 (truth 3), the derivative at 2 (truth 1.5) and
//! the contrast from 1 to 3 (truth 3) are separate pointwise claims, each scored on the
//! equal-tailed interval of the posterior draws the route publishes.
//!
//! This route is standalone (no study query) and has no runtime calibration key, so the
//! construction of its record is declared here beside the draw count, mirroring the key of the
//! frequentist dose-grid record (`quadratic_dose_calibration.rs`). A request the route refuses
//! is a skip, which counts as a miss.
//!
//! The designs are `#[ignore]`d measurement designs; the closed-form references of the route are
//! `dose_grid_functional_bayesian_reference.rs`.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(feature = "calibration-internal")]
#![allow(clippy::cast_precision_loss, reason = "bounded simulation dimensions")]

#[path = "../../antecedent/tests/common/calibration.rs"]
mod calibration;
#[path = "common/candidate_calibration.rs"]
mod candidate;

use antecedent_estimate::dose_basis_regression::ResidualPrior;
use antecedent_estimate::dose_grid_functional::{DoseDesign, DoseFunctional};
use antecedent_estimate::dose_grid_posterior::{
    DoseGridPosteriorRequest, estimate_dose_grid_posterior,
};
use calibration::{
    Construction, CoverageTally, RecordKey, ScopeFacts, grid_n, grid_seed, map_replicates, n_sim,
};

const DRAWS: usize = 2_000;
const LEVEL: f64 = 0.95;
const EVALUATION_DOSE: f64 = 2.0;

fn construction(functional: &str) -> Construction {
    Construction {
        query: "DoseResponse".into(),
        graph_class: "Dag".into(),
        structure: "fixed".into(),
        modality: "tabular".into(),
        inference: "Bayesian".into(),
        estimator: "bayesian.dose_grid".into(),
        interval_method: "posterior_quantile".into(),
        se_kind: String::new(),
        dependence: "iid".into(),
        posterior: "polynomial_dose.degree=2.ridge_scale=10.residual=inverse_gamma".into(),
        functional: functional.into(),
        identification: "point".into(),
        reported_level: LEVEL,
        unit: None,
    }
}

fn measure(test: &'static str, functional_key: &str, functional: DoseFunctional, truth: f64) {
    let n = grid_n(500);
    let construction = construction(functional_key);
    let mut tally = CoverageTally::for_record(
        RecordKey {
            test,
            dgp: "randomized_uniform_quadratic_mean_gaussian_errors",
            interval: "posterior_quantile",
        },
        LEVEL,
    )
    .measures("capability", "estimate.dose_grid_posterior");
    let runs = map_replicates(n_sim(), |rep| {
        let mut rng = candidate::Generator::new(grid_seed(0x47ab_5000 + rep));
        let (mut dose, mut outcome) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for _ in 0..n {
            let d = 4. * rng.uniform();
            dose.push(d);
            outcome.push(1. + 0.5 * d + 0.25 * d * d + rng.normal());
        }
        estimate_dose_grid_posterior(&DoseGridPosteriorRequest {
            dose: &dose,
            outcome: &outcome,
            grid: &[EVALUATION_DOSE],
            support: (0.0, 4.0),
            coefficient_scales: &[10.0, 10.0, 10.0],
            residual: ResidualPrior::InverseGamma { shape: 1.0, scale: 1.0 },
            functional,
            design: DoseDesign::RandomizedDose,
            draws: DRAWS,
            seed: grid_seed(0x5eed_0000 + rep),
        })
        .ok()
        .and_then(|posterior| posterior.interval_95.first().copied())
    });
    for run in runs {
        let Some([lower, upper]) = run else {
            tally.skip();
            continue;
        };
        tally.bind_validation(
            &construction,
            ScopeFacts {
                row_count: n as u64,
                replicates_ok: None,
                posterior_draws: Some(u32::try_from(DRAWS).expect("draw count fits u32")),
                unidentified_mass: 0.0,
            },
            "not_applicable",
        );
        tally.record(Some((lower, upper)), truth);
    }
    tally.assert();
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_dose_grid_level_credible_interval() {
    measure(
        "bayesian_dose_grid_level_credible_interval",
        &format!("level:{:016x}.polynomial_degree:2", EVALUATION_DOSE.to_bits()),
        DoseFunctional::Level,
        3.0,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_dose_grid_derivative_credible_interval() {
    measure(
        "bayesian_dose_grid_derivative_credible_interval",
        &format!("derivative:{:016x}.polynomial_degree:2", EVALUATION_DOSE.to_bits()),
        DoseFunctional::Derivative,
        1.5,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_dose_grid_contrast_credible_interval() {
    measure(
        "bayesian_dose_grid_contrast_credible_interval",
        &format!(
            "contrast:{:016x}:{:016x}.polynomial_degree:2",
            1.0_f64.to_bits(),
            3.0_f64.to_bits()
        ),
        DoseFunctional::Contrast { from: 1.0, to: 3.0 },
        3.0,
    );
}
