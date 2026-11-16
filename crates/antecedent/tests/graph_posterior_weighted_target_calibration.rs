//! Coverage of the effect interval of a structure posterior for a kernel-weighted target.
//!
//! The confounder is `z ~ N(0, 1)`, a noise variable `w` is independent of everything, the
//! planted effect is `tau(z) = 1 + z`, and the target is the Gaussian kernel neighbourhood of
//! the centre `c`, whose effect has the closed form `1 + c / (1 + h^2)` (see
//! `weighted_target_effect_calibration.rs`). The posterior holds two structures that both
//! identify that effect, `z -> t, z -> y, t -> y` (mass 0.6) and the same with `w -> y`
//! (mass 0.4): the effect is estimated in both, mixed by posterior weight, and the published
//! interval is `ate ± z·se` of the mixture with `se = sqrt((sum w se)^2 + between-structure
//! variance)`. The structure-wise effects estimate the same planted effect, so the mixed
//! effect has the closed form as its truth.
//!
//! One test per overlap regime; a replicate the weighted-overlap check refuses is skipped, so
//! the cell reports coverage of the intervals that were published.
//!
//! Ignored tests run via `scripts/measure_calibration.sh`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod common;

use antecedent::inference::{KernelFamily, TransportKernel};
use antecedent::{
    AverageEffectQuery, BayesianConfig, EstimatorId, InferenceMode, RefuteSuite, Study,
    WeightedTargetDeclaration,
};
use antecedent_core::{ExecutionContext, StreamDomain, VariableId};
use antecedent_data::TabularData;
use antecedent_discovery::adjacency_mask::{AdjacencyMask, set_edge};
use antecedent_discovery::{GraphPosterior, SamplingSummary};
use antecedent_kernels::standard_normal;
use antecedent_prob::InferenceDiagnostics;
use common::calibration::{
    Construction, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts, grid_n, grid_seed,
    map_replicates, n_sim,
};
use common::calibration_bind::{bind_reported, constructions};

const INTERVAL: &str = "analytic_wald";
const POSTERIOR_INTERVAL: &str = "posterior_quantile";
const DGP: &str =
    "planted_heterogeneous_effect_two_identifying_structures_kernel_target_closed_form";
const Z95: f64 = 1.959_963_984_540_054;
const BASE_ROWS: usize = 1500;
const SEED_BASE: u64 = 0x6B2E_91D3;
/// Columns in schema order: `t`, `y`, `z`, `w`.
const T: usize = 0;
const Y: usize = 1;
const Z: usize = 2;
const W: usize = 3;
const MASSES: [f64; 2] = [0.6, 0.4];

fn mask_of(edges: &[(usize, usize)]) -> AdjacencyMask {
    let mut mask = set_edge(0_u64, 4, edges[0].0, edges[0].1, true);
    for &(from, to) in &edges[1..] {
        mask = set_edge(mask, 4, from, to, true);
    }
    mask
}

fn posterior() -> Option<GraphPosterior> {
    let base = [(Z, T), (Z, Y), (T, Y)];
    let with_noise = [(Z, T), (Z, Y), (T, Y), (W, Y)];
    let mut marginals = vec![0.0; 16];
    for (edges, mass) in [base.as_slice(), with_noise.as_slice()].into_iter().zip(MASSES) {
        for &(from, to) in edges {
            marginals[from * 4 + to] += mass;
        }
    }
    GraphPosterior::from_atoms(
        4,
        MASSES.to_vec(),
        vec![mask_of(&base), mask_of(&with_noise)],
        marginals.clone(),
        marginals,
        SamplingSummary {
            ess: 1.0 / MASSES.iter().map(|m| m * m).sum::<f64>(),
            diagnostics: InferenceDiagnostics::analytic("graph_posterior_weighted_target"),
            rejected_invalid: 0,
        },
    )
    .ok()
}

/// The planted table of one replicate.
fn replicate_table(n: usize, seed: u64) -> Option<TabularData> {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Test, 0x33);
    let mut columns = vec![vec![0.0; n]; 4];
    for i in 0..n {
        let zi = standard_normal(&mut rng);
        let wi = standard_normal(&mut rng);
        let p = 1.0 / (1.0 + (-(0.9 * zi)).exp());
        let ti = f64::from(rng.next_f64() < p);
        columns[T][i] = ti;
        columns[Z][i] = zi;
        columns[W][i] = wi;
        columns[Y][i] = (1.0 + zi) * ti + zi + 0.3 * standard_normal(&mut rng);
    }
    let names = ["t", "y", "z", "w"];
    let borrowed: Vec<(&str, &[f64])> =
        names.iter().zip(&columns).map(|(name, values)| (*name, values.as_slice())).collect();
    TabularData::from_f64_columns(borrowed).ok()
}

fn declaration(centre: f64, bandwidth: f64) -> Option<WeightedTargetDeclaration> {
    WeightedTargetDeclaration::kernel(
        TransportKernel::new(KernelFamily::Gaussian, bandwidth).ok()?,
        vec![VariableId::from_raw(2)],
        vec![centre],
        None,
    )
    .ok()
}

fn effect_query() -> AverageEffectQuery {
    AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
}

/// One replicate: the analytic interval of the structure-averaged kernel-weighted effect, or
/// `None` when the target is refused.
fn interval(centre: f64, bandwidth: f64, n: usize, seed: u64) -> Option<(f64, f64)> {
    let result = Study::tabular(replicate_table(n, seed)?)
        .graph_posterior(posterior()?)
        .query(effect_query())
        .weighted_target(declaration(centre, bandwidth)?)
        .estimator(EstimatorId::Aipw)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .ok()?
        .run(&ExecutionContext::for_tests(seed))
        .ok()?;
    let effect = result.estimate().as_effect()?;
    if !effect.ate.is_finite() || !effect.se_analytic().is_finite() {
        return None;
    }
    let half = Z95 * effect.se_analytic();
    Some((effect.ate - half, effect.ate + half))
}

/// One Bayesian replicate: the posterior's 95% interval of the structure-averaged effect and
/// the construction the runtime reports for it.
type Reported = Vec<(Construction, ScopeFacts, &'static str)>;

fn posterior_interval(
    centre: f64,
    bandwidth: f64,
    n: usize,
    seed: u64,
) -> Option<((f64, f64), Reported)> {
    let study = Study::tabular(replicate_table(n, seed)?)
        .graph_posterior(posterior()?)
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(1_000)))
        .query(effect_query())
        .weighted_target(declaration(centre, bandwidth)?)
        .refute(RefuteSuite::None)
        .build()
        .ok()?;
    let result = study.run(&ExecutionContext::for_tests(seed)).ok()?;
    let posterior = result.posterior.as_ref()?;
    let (lower, upper) = (posterior.summaries.q025[0], posterior.summaries.q975[0]);
    if !(lower.is_finite() && upper.is_finite() && lower <= upper) {
        return None;
    }
    let reported = constructions(&study.inspect().ok()?, &result);
    Some(((lower, upper), reported))
}

fn construction() -> Construction {
    Construction {
        query: "AverageEffect".into(),
        graph_class: "Dag".into(),
        structure: "graph_posterior".into(),
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
fn graph_posterior_weighted_target_interval_covers_across_bandwidths() {
    assert_coverage_over(
        "graph_posterior_weighted_target_interval_covers_across_bandwidths",
        0.5,
        &[0.5, 1.0, 2.0],
    );
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn graph_posterior_weighted_target_interval_covers_under_thin_overlap() {
    assert_coverage_over(
        "graph_posterior_weighted_target_interval_covers_under_thin_overlap",
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

fn assert_posterior_coverage_over(test: &'static str, centre: f64, bandwidths: &[f64]) {
    let n = grid_n(BASE_ROWS);
    for &bandwidth in bandwidths {
        let truth = 1.0 + centre / (1.0 + bandwidth * bandwidth);
        let runs = map_replicates(n_sim(), |rep| {
            posterior_interval(centre, bandwidth, n, grid_seed(SEED_BASE + rep))
        });
        let mut tally = CoverageTally::for_record(
            RecordKey { test, dgp: DGP, interval: POSTERIOR_INTERVAL },
            REPORTED_LEVEL,
        )
        .labelled(format!("centre {centre} bandwidth {bandwidth}"));
        for run in runs {
            let Some((band, reported)) = run else {
                tally.skip();
                continue;
            };
            bind_reported(&mut tally, &reported);
            tally.record(Some(band), truth);
        }
        tally.assert();
    }
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn graph_posterior_weighted_target_credible_interval_covers_across_bandwidths() {
    assert_posterior_coverage_over(
        "graph_posterior_weighted_target_credible_interval_covers_across_bandwidths",
        0.5,
        &[0.5, 1.0, 2.0],
    );
}

#[test]
fn a_target_far_from_the_rows_is_refused_under_bayesian_inference() {
    let n = BASE_ROWS / 2;
    for rep in 0..20 {
        assert!(posterior_interval(6.0, 0.3, n, grid_seed(SEED_BASE + rep)).is_none());
    }
}
