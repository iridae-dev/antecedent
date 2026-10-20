//! CI-search throughput of the static PC and the contemporaneous PCMCI+ / LPCMCI phases.
//!
//! Prints wall time, CI-test count and a digest of the discovered links per thread count, so a
//! change to the search loops can be checked for both speed and unchanged output.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::Instant;

use antecedent_core::{
    CausalSchemaBuilder, ExecutionContext, Lag, MeasurementSpec, NonZeroThreadCount, Parallelism,
    RoleHint, SmallRoleSet, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TabularData, TimeIndex,
    TimeSeriesData, ValidityBitmap,
};
use antecedent_discovery::{
    DiscoveryConstraints, DiscoveryWorkspace, Lpcmci, Pc, PcmciPlus, TemporalConstraints,
};

struct Noise(u64);

impl Noise {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }

    fn gauss(&mut self) -> f64 {
        let (u, v) = (self.next().max(1e-12), self.next());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

fn schema(p: usize) -> antecedent_core::CausalSchema {
    let mut b = CausalSchemaBuilder::new();
    for i in 0..p {
        b.add_variable(
            format!("v{i}"),
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    b.build().unwrap()
}

fn storage(p: usize, columns: Vec<Vec<f64>>) -> (OwnedColumnarStorage, Vec<VariableId>) {
    let vars: Vec<_> = (0..p).map(|i| VariableId::from_raw(u32::try_from(i).unwrap())).collect();
    let n = columns[0].len();
    let cols = columns
        .into_iter()
        .enumerate()
        .map(|(i, v)| {
            OwnedColumn::Float64(
                Float64Column::new(vars[i], Arc::from(v), ValidityBitmap::all_valid(n)).unwrap(),
            )
        })
        .collect();
    (OwnedColumnarStorage::try_new(schema(p), cols, None, None).unwrap(), vars)
}

/// Random sparse linear-Gaussian DAG over `p` nodes in index order.
fn static_data(p: usize, n: usize, seed: u64, mean_parents: f64) -> (TabularData, Vec<VariableId>) {
    let mut rng = Noise(seed);
    let mut parents: Vec<Vec<(usize, f64)>> = vec![Vec::new(); p];
    for (j, slot) in parents.iter_mut().enumerate().skip(1) {
        for i in 0..j {
            if rng.next() < mean_parents / p as f64 {
                slot.push((i, 0.5 + 0.5 * rng.next()));
            }
        }
    }
    let mut cols = vec![vec![0.0; n]; p];
    for r in 0..n {
        for j in 0..p {
            let mut v = rng.gauss();
            for &(i, w) in &parents[j] {
                v += w * cols[i][r];
            }
            cols[j][r] = v;
        }
    }
    let (storage, vars) = storage(p, cols);
    (TabularData::new(storage), vars)
}

fn temporal_data(p: usize, n: usize, seed: u64) -> (TimeSeriesData, Vec<VariableId>) {
    let mut rng = Noise(seed);
    let mut cols = vec![vec![0.0; n]; p];
    for t in 1..n {
        for j in 0..p {
            let mut v = rng.gauss() + 0.3 * cols[j][t - 1];
            if j > 0 {
                v += 0.5 * cols[j - 1][t];
            }
            if j > 1 && j % 2 == 0 {
                v += 0.4 * cols[j - 2][t];
            }
            if j + 1 < p {
                v += 0.1 * cols[j + 1][t - 1];
            }
            cols[j][t] = v;
        }
    }
    let (storage, vars) = storage(p, cols);
    let data = TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap();
    (data, vars)
}

fn context(threads: u32) -> ExecutionContext {
    let mut ctx = ExecutionContext::for_tests(1);
    ctx.parallelism = Parallelism::bounded(NonZeroThreadCount::new(threads).unwrap());
    ctx = ctx.with_admission(antecedent_core::execution::AdmissionPool::fixed(threads, u64::MAX));
    ctx
}

fn links(links: &[antecedent_discovery::ScoredLink]) -> String {
    let mut rows: Vec<String> = links.iter().map(|l| format!("{l:?}")).collect();
    rows.sort();
    rows.join(";")
}

fn digest(parts: &[String]) -> u64 {
    let mut h = DefaultHasher::new();
    parts.hash(&mut h);
    h.finish()
}

/// Retain every repeat and report the median; never select the fastest run.
const RUNS: usize = 5;

fn measure<T>(name: &str, threads: u32, mut run: impl FnMut() -> T) -> (std::time::Duration, T) {
    let mut samples = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let t = Instant::now();
        let out = run();
        samples.push((t.elapsed(), out));
    }
    println!(
        "ci_search control={name} threads={threads} samples_ns={:?}",
        samples.iter().map(|(time, _)| time.as_nanos()).collect::<Vec<_>>()
    );
    samples.sort_by_key(|(time, _)| *time);
    samples.remove(RUNS / 2)
}

fn main() {
    let (tab, tab_vars) = static_data(30, 1500, 7, 8.0);
    let pc = Pc::new().with_fdr(false).with_constraints(DiscoveryConstraints {
        temporal: TemporalConstraints {
            max_lag: Lag::CONTEMPORANEOUS,
            min_lag: Lag::CONTEMPORANEOUS,
        },
        max_cond_size: 4,
        ..DiscoveryConstraints::default()
    });
    let (ts, ts_vars) = temporal_data(7, 1200, 11);
    let temporal = DiscoveryConstraints {
        temporal: TemporalConstraints { max_lag: Lag::from_raw(2), min_lag: Lag::CONTEMPORANEOUS },
        max_cond_size: 3,
        ..DiscoveryConstraints::default()
    };
    let plus = PcmciPlus::new().with_fdr(false).with_constraints(temporal.clone());
    let lpcmci = Lpcmci::new().with_fdr(false).with_constraints(temporal);

    for threads in [1u32, 4] {
        let ctx = context(threads);
        let (elapsed, r) = measure("pc", threads, || {
            pc.run(&tab, &tab_vars, &mut DiscoveryWorkspace::default(), &ctx).unwrap()
        });
        let d = digest(&[
            links(&r.evidence.links),
            format!("{:?}", r.evidence.graph.directed_edge_count()),
        ]);
        println!(
            "pc      threads={threads} {elapsed:>9.1?} ci_tests={} links={} digest={d:016x}",
            r.performance.ci_tests,
            r.evidence.links.len()
        );

        let (elapsed, r) = measure("pcmci+", threads, || {
            plus.run(&ts, &ts_vars, &mut DiscoveryWorkspace::default(), &ctx).unwrap()
        });
        let d = digest(&[
            links(&r.evidence.links),
            format!(
                "{} {}",
                r.evidence.graph.directed_edge_count(),
                r.review.pending_undirected.len()
            ),
        ]);
        println!(
            "pcmci+  threads={threads} {elapsed:>9.1?} ci_tests={} links={} digest={d:016x}",
            r.performance.ci_tests,
            r.evidence.links.len()
        );

        let (elapsed, r) = measure("lpcmci", threads, || {
            lpcmci.run(&ts, &ts_vars, &mut DiscoveryWorkspace::default(), &ctx).unwrap()
        });
        let d = digest(&[links(&r.evidence.links)]);
        println!(
            "lpcmci  threads={threads} {elapsed:>9.1?} ci_tests={} links={} digest={d:016x}",
            r.performance.ci_tests,
            r.evidence.links.len()
        );
    }
}
