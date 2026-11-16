//! Coverage of the trial-to-target IPW interval for a kernel-weighted target population.
//!
//! Trial rows have `z ~ N(0, 1)`, target rows `z ~ N(1, 1)`, treatment is randomized in the
//! trial and the effect is `tau(z) = 1 + z`. The target is the Gaussian kernel neighbourhood of
//! centre `c` and bandwidth `h` within the target population, whose effect has the closed form
//! `1 + (1 + c / h^2) / (1 + 1 / h^2)`. Each replicate transports the trial contrast with the
//! true selection and treatment probabilities and scores the delta-method interval
//! `ipw ± z·se` against that closed form.
//!
//! One test per overlap regime. The moderate regime keeps the Kish ESS fraction of the target
//! weights well above the floor; the thin regime sits in the tail of the target, where the
//! overlap gate refuses some replicates, which are skipped rather than scored.
//!
//! Ignored tests run via `scripts/measure_calibration.sh`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod common;

use std::sync::Arc;

use antecedent::inference::{KernelFamily, TransportKernel};
use antecedent::{WeightedTargetDeclaration, transport_trial_to_weighted_target};
use antecedent_core::{ExecutionContext, StreamDomain, VariableId};
use antecedent_data::TabularData;
use antecedent_identify::{
    PopulationFactor, TransportCertificate, TransportFormula, TransportIdentification,
};
use antecedent_kernels::standard_normal;
use common::calibration::{
    Construction, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts, grid_n, grid_seed,
    map_replicates, n_sim,
};

const INTERVAL: &str = "trial_ipw_delta_wald";
const DGP: &str = "planted_effect_modification_trial_to_kernel_target_closed_form";
const Z95: f64 = 1.959_963_984_540_054;
const BASE_ROWS: usize = 3000;
const SEED_BASE: u64 = 0x7B1A_C0DE;

fn identification() -> TransportIdentification {
    TransportIdentification::Transportable {
        formula: TransportFormula::Direct(PopulationFactor {
            regime: None,
            population: Arc::from("source"),
            variables: Arc::from([]),
            conditioned_on: Arc::from([]),
            interventions: Arc::from([]),
        }),
        certificate: TransportCertificate {
            rule: Arc::from("transport.direct"),
            selection_targets: Arc::from([]),
            premises: Arc::from([]),
        },
    }
}

/// One replicate: the delta-method interval of the weighted-target contrast, or `None` when
/// the overlap gate refuses the target.
fn interval(centre: f64, bandwidth: f64, n: usize, seed: u64) -> Option<(f64, f64)> {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Test, 0x42);
    let (mut z, mut y) = (vec![0.0; n], vec![0.0; n]);
    let (mut treatment, mut trial) = (vec![false; n], vec![false; n]);
    let mut selection = vec![0.0; n];
    for i in 0..n {
        let in_trial = i % 2 == 0;
        trial[i] = in_trial;
        let zi = standard_normal(&mut rng) + if in_trial { 0.0 } else { 1.0 };
        z[i] = zi;
        selection[i] = 1.0 / (1.0 + (zi - 0.5).exp());
        let a = rng.next_f64() < 0.5;
        treatment[i] = a;
        y[i] = (1.0 + zi) * f64::from(a) + zi + 0.3 * standard_normal(&mut rng);
    }
    let pairs = [("z", z)];
    let borrowed: Vec<(&str, &[f64])> = pairs.iter().map(|(n, v)| (*n, v.as_slice())).collect();
    let data = TabularData::from_f64_columns(borrowed).ok()?;
    let declaration = WeightedTargetDeclaration::kernel(
        TransportKernel::new(KernelFamily::Gaussian, bandwidth).ok()?,
        vec![VariableId::from_raw(0)],
        vec![centre],
        None,
    )
    .ok()?;
    let result = transport_trial_to_weighted_target(
        &declaration,
        &data,
        &identification(),
        &y,
        &treatment,
        &trial,
        &selection,
        &vec![0.5; n],
        None,
    )
    .ok()?;
    let half = Z95 * result.estimate.ipw_se;
    Some((result.estimate.effect.ipw - half, result.estimate.effect.ipw + half))
}

fn construction() -> Construction {
    Construction {
        query: "TransportQuery".into(),
        graph_class: "Admg".into(),
        structure: "fixed".into(),
        modality: "tabular".into(),
        inference: "Frequentist".into(),
        estimator: "trial_ipw.weighted_target".into(),
        interval_method: INTERVAL.into(),
        se_kind: "iid".into(),
        dependence: "iid".into(),
        posterior: String::new(),
        functional: "weighted_target_transported.mean".into(),
        identification: "point".into(),
        reported_level: REPORTED_LEVEL,
    }
}

fn assert_coverage_over(test: &'static str, centre: f64, bandwidths: &[f64]) {
    let n = grid_n(BASE_ROWS);
    for &bandwidth in bandwidths {
        let h2 = bandwidth * bandwidth;
        let truth = 1.0 + (1.0 + centre / h2) / (1.0 + 1.0 / h2);
        let runs = map_replicates(n_sim(), |rep| {
            interval(centre, bandwidth, n, grid_seed(SEED_BASE + rep))
        });
        let mut tally = CoverageTally::for_record(
            RecordKey { test, dgp: DGP, interval: INTERVAL },
            REPORTED_LEVEL,
        )
        .labelled(format!("centre {centre} bandwidth {bandwidth}"));
        for run in runs {
            let Some(band) = run else {
                tally.skip();
                continue;
            };
            tally.bind_validation(
                &construction(),
                ScopeFacts {
                    row_count: n as u64,
                    replicates_ok: None,
                    posterior_draws: None,
                    unidentified_mass: 0.0,
                },
                "not_applicable",
            );
            tally.record(Some(band), truth);
        }
        tally.assert();
    }
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn weighted_target_transport_interval_covers_across_bandwidths() {
    assert_coverage_over(
        "weighted_target_transport_interval_covers_across_bandwidths",
        1.0,
        &[0.6, 1.0, 1.6],
    );
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn weighted_target_transport_interval_covers_under_thin_overlap() {
    assert_coverage_over(
        "weighted_target_transport_interval_covers_under_thin_overlap",
        2.5,
        &[0.5, 0.8],
    );
}

#[test]
fn a_target_far_from_the_target_rows_is_refused_on_every_replicate() {
    let n = BASE_ROWS / 2;
    for rep in 0..20 {
        assert!(interval(8.0, 0.2, n, grid_seed(SEED_BASE + rep)).is_none());
    }
}
