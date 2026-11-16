//! Coverage of the effect interval for a kernel-weighted target population.
//!
//! The confounder is `z ~ N(0, 1)`, the planted effect is `tau(z) = 1 + z`, and the target is the
//! Gaussian kernel neighbourhood `w(z) = exp(-(z - c)^2 / (2 h^2))` of the centre `c`, declared
//! as per-row weights (a custom target distribution over `z`). The target population is the
//! source population reweighted by `w`, so its effect has the closed form
//! `E[w(z) tau(z)] / E[w(z)] = 1 + c / (1 + h^2)`: the product of `N(0, 1)` and the kernel is a
//! normal with mean `c / (1 + h^2)`. Each replicate estimates the effect with AIPW under those
//! weights and scores the analytic interval `ate ± z·se` against the closed form.
//!
//! One test per overlap regime. The moderate regime keeps the Kish ESS fraction of the weights
//! from about 0.3 to 0.8 of the rows across its bandwidths; the thin regime sits in the tail of
//! the source where the weighted-overlap gate refuses some replicates, which are skipped rather
//! than scored, so the cell reports coverage of the intervals that were published.
//!
//! Ignored tests run via `scripts/measure_calibration.sh`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod common;

use antecedent::{EstimatorId, RefuteSuite, Study};
use antecedent_core::{
    AverageEffectQuery, DistributionRef, ExecutionContext, PopulationRegistry, StreamDomain,
    TargetPopulation, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_kernels::standard_normal;
use common::calibration::{
    Construction, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts, grid_n, grid_seed,
    map_replicates, n_sim,
};

const INTERVAL: &str = "analytic_wald";
const DGP: &str = "planted_heterogeneous_effect_with_gaussian_kernel_target_closed_form";
const Z95: f64 = 1.959_963_984_540_054;
const BASE_ROWS: usize = 1500;
const SEED_BASE: u64 = 0x5E1D_7A4C;

/// One replicate: the analytic interval of the kernel-weighted effect, or `None` when the
/// weighted-overlap check refuses the target.
fn interval(centre: f64, bandwidth: f64, n: usize, seed: u64) -> Option<(f64, f64)> {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Test, 0x32);
    let mut z = vec![0.0; n];
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    let mut w = vec![0.0; n];
    for i in 0..n {
        let zi = standard_normal(&mut rng);
        z[i] = zi;
        let p = 1.0 / (1.0 + (-(0.9 * zi)).exp());
        t[i] = f64::from(rng.next_f64() < p);
        y[i] = (1.0 + zi) * t[i] + zi + 0.3 * standard_normal(&mut rng);
        w[i] = (-0.5 * ((zi - centre) / bandwidth).powi(2)).exp();
    }
    let pairs = [("t", t), ("y", y), ("z", z)];
    let borrowed: Vec<(&str, &[f64])> = pairs.iter().map(|(n, v)| (*n, v.as_slice())).collect();
    let data = TabularData::from_f64_columns(borrowed).ok()?;
    let mut graph = Dag::with_variables(3);
    for (from, to) in [(2, 0), (2, 1), (0, 1)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).ok()?;
    }
    let href = DistributionRef::from_raw(1);
    let mut registry = PopulationRegistry::new();
    registry.insert_distribution_with_dependence(href, w, [VariableId::from_raw(2)]);
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
        .with_target_population(TargetPopulation::CustomDistribution(href));
    let result = Study::tabular(data)
        .graph(graph)
        .query(query)
        .population_registry(registry)
        .estimator(EstimatorId::Aipw)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .ok()?
        .run(&ExecutionContext::for_tests(seed))
        .ok()?;
    let effect = result.estimate().as_effect()?;
    let half = Z95 * effect.se_analytic();
    Some((effect.ate - half, effect.ate + half))
}

fn construction() -> Construction {
    Construction {
        query: "AverageEffect".into(),
        graph_class: "Dag".into(),
        structure: "fixed".into(),
        modality: "tabular".into(),
        inference: "Frequentist".into(),
        estimator: "aipw".into(),
        interval_method: INTERVAL.into(),
        se_kind: "iid".into(),
        dependence: "iid".into(),
        posterior: String::new(),
        functional: "weighted_target.mean".into(),
        identification: "point".into(),
        reported_level: REPORTED_LEVEL,
    }
}

fn assert_coverage_over(test: &'static str, centre: f64, bandwidths: &[f64]) {
    let n = grid_n(BASE_ROWS);
    for &bandwidth in bandwidths {
        let truth = 1.0 + centre / (1.0 + bandwidth * bandwidth);
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
fn weighted_target_effect_interval_covers_across_bandwidths() {
    assert_coverage_over(
        "weighted_target_effect_interval_covers_across_bandwidths",
        0.5,
        &[0.5, 1.0, 2.0],
    );
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn weighted_target_effect_interval_covers_under_thin_overlap() {
    assert_coverage_over(
        "weighted_target_effect_interval_covers_under_thin_overlap",
        1.5,
        &[0.5, 0.8],
    );
}

#[test]
fn a_target_far_from_the_rows_is_refused_on_every_replicate() {
    // The kernel at centre 6 with bandwidth 0.3 leaves no arm with effective rows.
    let n = BASE_ROWS / 2;
    for rep in 0..20 {
        assert!(interval(6.0, 0.3, n, grid_seed(SEED_BASE + rep)).is_none());
    }
}
