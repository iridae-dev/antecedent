//! Criterion smokes for the 0.5 causal-response and interference hot paths.
//!
//! Both carry soft-budget gates (asserted on every invocation including the
//! `--test` smoke) sized with ~10× headroom over the accepted local means, so
//! superlinear regressions — like the O(n²) pseudo-outcome loop fixed in
//! 0.5.2 — fail the gate instead of shipping.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(missing_docs, clippy::cast_precision_loss, clippy::many_single_char_names)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use antecedent_core::{
    AssignmentDesign, AssumptionSet, ContinuousDomain, ExposureLevel, ExposureMapping, GridSpec,
    IdentificationStatus, InterferenceFunctional, InterferenceQuery, ResponseFunctional,
    ResponseQuery, VariableId,
};
use antecedent_data::{NetworkData, NetworkEdge, TabularData};
use antecedent_estimate::{ContinuousResponseEstimator, estimate_interference};
use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn curve_data(n: usize) -> TabularData {
    let mut a = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut x = Vec::with_capacity(n);
    for i in 0..n {
        let z = -1.0 + 2.0 * i as f64 / (n - 1) as f64;
        let noise = ((i * 37 % 101) as f64 / 100.0 - 0.5) * 0.3;
        let treatment = 0.7 * z + noise;
        x.push(z);
        a.push(treatment);
        y.push(1.0 + 2.0 * treatment + 0.8 * z + 0.05 * (i as f64).sin());
    }
    TabularData::from_f64_columns([("a", a.as_slice()), ("y", y.as_slice()), ("x", x.as_slice())])
        .unwrap()
}

fn run_curve(data: &TabularData) {
    let a = VariableId::from_raw(0);
    let y = VariableId::from_raw(1);
    let x = VariableId::from_raw(2);
    let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: y,
        treatment: ContinuousDomain::new(
            a,
            GridSpec::Values(Arc::from([-0.4, -0.2, 0.0, 0.2, 0.4])),
        ),
    });
    let response = ContinuousResponseEstimator::new([x])
        .estimate_identified(
            data,
            &query,
            IdentificationStatus::NonparametricallyIdentified,
            AssumptionSet::new(),
        )
        .unwrap();
    black_box(response);
}

fn interference_fixture(n: usize) -> (NetworkData, Vec<bool>, InterferenceQuery) {
    let outcome: Vec<f64> = (0..n).map(|i| (i % 17) as f64 * 0.1).collect();
    let data = TabularData::from_f64_columns([("y", outcome.as_slice())]).unwrap();
    let mut edges = Vec::with_capacity(n * 4);
    for i in 0..n {
        for step in [1usize, 7, 31, 101] {
            let j = (i + step) % n;
            edges.push(NetworkEdge {
                from: u32::try_from(i).unwrap(),
                to: u32::try_from(j).unwrap(),
                weight: 1.0,
            });
        }
    }
    let network = NetworkData::try_new(data, edges).unwrap();
    let n_clusters = n / 10;
    let clusters: Arc<[u32]> = (0..n).map(|i| u32::try_from(i % n_clusters).unwrap()).collect();
    let assignment: Vec<bool> = (0..n).map(|i| (i % n_clusters) < n_clusters / 2).collect();
    let query = InterferenceQuery {
        assignment: AssignmentDesign::ClusterRandomization {
            clusters,
            treated_clusters: n_clusters / 2,
        },
        exposure: ExposureMapping::OwnTreatment,
        functional: InterferenceFunctional::ExposureContrast {
            outcome: VariableId::from_raw(0),
            from: ExposureLevel { own: 0.0, neighbors: 0.0 },
            to: ExposureLevel { own: 1.0, neighbors: 0.0 },
        },
        probability_draws: 2_000,
    };
    (network, assignment, query)
}

fn bench_response_interference(c: &mut Criterion) {
    let curve = curve_data(4_000);
    c.bench_function("kennedy_curve_n4k_grid5", |b| {
        b.iter(|| run_curve(&curve));
    });

    let (network, assignment, query) = interference_fixture(10_000);
    c.bench_function("interference_cluster_n10k_2kdraws", |b| {
        b.iter(|| {
            black_box(estimate_interference(&query, &network, &assignment, 7).unwrap());
        });
    });

    // Soft-budget gates. Accepted local means (Apple M1 Max, 0.5.2): curve
    // ~210 ms, interference ~167 ms. Budgets carry ~5× headroom; the pre-0.5.2
    // quadratic pseudo-outcome loop (~4 s at this size) and the per-draw
    // cluster scan both fail them.
    let t0 = Instant::now();
    run_curve(&curve);
    let elapsed = t0.elapsed();
    assert!(
        elapsed < Duration::from_secs(1),
        "kennedy_curve_n4k_grid5 exceeded soft budget: {elapsed:?}"
    );
    let t0 = Instant::now();
    black_box(estimate_interference(&query, &network, &assignment, 7).unwrap());
    let elapsed = t0.elapsed();
    assert!(
        elapsed < Duration::from_secs(1),
        "interference_cluster_n10k_2kdraws exceeded soft budget: {elapsed:?}"
    );
}

criterion_group!(benches, bench_response_interference);
criterion_main!(benches);
