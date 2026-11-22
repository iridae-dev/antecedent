//! Coverage of the transported estimate under kernel target weights.
//!
//! Source rows carry a covariate `x ~ U(0, 1)` and a noisy unit effect. The effect is `TAU` in
//! the neighbourhood `|x - 0.5| < NEIGHBOURHOOD` and `TAU + 3` outside it, so the kernel-weighted
//! mean at the neighbourhood centre is `TAU` for every bandwidth whose support lies inside the
//! neighbourhood. Each replicate declares a kernel over the rows' distances to the centre,
//! transports a tight source prior toward that weighted target, and scores the interval
//! `mean ± z·sd` read off the transported coefficient moments against `TAU`.
//!
//! The bandwidth grid spans the overlap levels the check supports (Kish ESS fraction from about
//! 0.1 to 0.45 of the rows); one cell per family and bandwidth is asserted. A bandwidth thin
//! enough to fail the overlap check is refused on every replicate instead of reporting an
//! interval, which the last test asserts.
//!
//! Ignored tests run via `scripts/measure_calibration.sh`.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

mod common;

use std::sync::Arc;

use antecedent_prob::{
    ExternalPriorSource, ExternalPriorWeight, GaussianCoefficientPrior, KernelFamily, PriorSet,
    PriorSpec, TransportKernel, TransportPolicy, WeightedTarget, apply_weighted_target_transport,
};
use common::calibration::{
    Construction, CoverageTally, RecordKey, ScopeFacts, gaussian, grid_n, grid_seed,
    map_replicates, n_sim, stream_seed, uniform,
};

const LEVEL: f64 = 0.95;
const INTERVAL: &str = "reweighted_mean_wald";
const DGP: &str = "planted_neighbourhood_effect_with_kernel_weighted_target_closed_form_mean";
const Z: f64 = 1.959_963_984_540_054;
const TAU: f64 = 1.5;
const OUTSIDE_SHIFT: f64 = 3.0;
const NOISE_SD: f64 = 1.0;
const CENTRE: f64 = 0.5;
const NEIGHBOURHOOD: f64 = 0.3;
const BASE_ROWS: usize = 1500;
const SEED_BASE: u64 = 0x7A5C_0E11;

fn tight_source() -> ExternalPriorSource {
    let mut prior = PriorSet::new();
    prior.push(PriorSpec::GaussianCoefficients(
        GaussianCoefficientPrior::shared(1, 0.0, 1e-9).unwrap(),
    ));
    ExternalPriorSource {
        id: Arc::from("source"),
        prior,
        weight: ExternalPriorWeight::power(0.5).unwrap(),
        ess: None,
    }
}

/// One replicate: the transported interval, or `None` when the overlap check refuses.
fn interval(family: KernelFamily, bandwidth: f64, n: usize, seed: u64) -> Option<(f64, f64)> {
    let mut x_stream = uniform(stream_seed(seed, 1), 0x11);
    let mut noise = gaussian(stream_seed(seed, 2));
    let mut distances = Vec::with_capacity(n);
    let mut effects = Vec::with_capacity(n);
    for _ in 0..n {
        let x = x_stream();
        let distance = (x - CENTRE).abs();
        let shift = if distance < NEIGHBOURHOOD { 0.0 } else { OUTSIDE_SHIFT };
        distances.push(distance);
        effects.push(TAU + shift + NOISE_SD * noise());
    }
    let kernel = TransportKernel::new(family, bandwidth).unwrap();
    let target = WeightedTarget::from_kernel(effects, &distances, kernel).ok()?;
    let (out, _) = apply_weighted_target_transport(
        &[tight_source()],
        TransportPolicy::InvariantConditionalOutcome,
        &target,
        None,
    )
    .ok()?;
    let coef = out[0].prior.gaussian_coefficients()?;
    let mean = coef.mean[0];
    let sd = coef.variance[0].sqrt();
    Some((mean - Z * sd, mean + Z * sd))
}

fn construction(family: KernelFamily) -> Construction {
    Construction {
        query: "WeightedTargetTransportedPrior".into(),
        graph_class: "Dag".into(),
        structure: "fixed".into(),
        modality: "tabular".into(),
        inference: "Frequentist".into(),
        estimator: format!("weighted_target_transport.{}_kernel_weighted_mean", family.id()),
        interval_method: INTERVAL.into(),
        se_kind: "weighted_mean_sampling_variance".into(),
        dependence: "independent_by_design".into(),
        posterior: String::new(),
        functional: "kernel_weighted_neighbourhood_mean_effect".into(),
        identification: "point".into(),
        reported_level: LEVEL,
    }
}

fn assert_coverage_over(
    test: &'static str,
    dgp: &'static str,
    family: KernelFamily,
    bandwidths: &[f64],
) {
    let n = grid_n(BASE_ROWS);
    for &bandwidth in bandwidths {
        let runs = map_replicates(n_sim(), |rep| {
            interval(family, bandwidth, n, grid_seed(SEED_BASE + rep))
        });
        let mut tally =
            CoverageTally::for_record(RecordKey { test, dgp, interval: INTERVAL }, LEVEL)
                .labelled(format!("bandwidth {bandwidth}"));
        for run in runs {
            let Some(band) = run else {
                tally.skip();
                continue;
            };
            tally.bind_validation(
                &construction(family),
                ScopeFacts {
                    row_count: n as u64,
                    replicates_ok: None,
                    posterior_draws: None,
                    unidentified_mass: 0.0,
                },
                "not_applicable",
            );
            tally.record(Some(band), TAU);
        }
        tally.assert();
    }
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn gaussian_kernel_transported_estimate_covers_across_bandwidths() {
    assert_coverage_over(
        "gaussian_kernel_transported_estimate_covers_across_bandwidths",
        DGP,
        KernelFamily::Gaussian,
        &[0.03, 0.06, 0.1],
    );
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn epanechnikov_kernel_transported_estimate_covers_across_bandwidths() {
    assert_coverage_over(
        "epanechnikov_kernel_transported_estimate_covers_across_bandwidths",
        DGP,
        KernelFamily::Epanechnikov,
        &[0.06, 0.12, 0.25],
    );
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn tricube_kernel_transported_estimate_covers_across_bandwidths() {
    assert_coverage_over(
        "tricube_kernel_transported_estimate_covers_across_bandwidths",
        DGP,
        KernelFamily::Tricube,
        &[0.06, 0.12, 0.25],
    );
}

#[test]
fn a_bandwidth_too_thin_for_the_rows_is_refused_on_every_replicate() {
    // ESS fraction of a Gaussian kernel is about 3.5 * bandwidth, far below the 0.05 floor.
    let n = BASE_ROWS / 2;
    for rep in 0..50 {
        assert!(interval(KernelFamily::Gaussian, 0.002, n, grid_seed(SEED_BASE + rep)).is_none());
    }
}
