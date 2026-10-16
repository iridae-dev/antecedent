//! Discovery results must not depend on the thread count: the CI search loops evaluate
//! upcoming conditioning sets speculatively and in parallel, and report exactly what the
//! one-test-at-a-time loop would.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    CausalSchemaBuilder, ExecutionContext, Lag, MeasurementSpec, NonZeroThreadCount, Parallelism,
    RoleHint, SmallRoleSet, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TabularData, TimeIndex,
    TimeSeriesData, ValidityBitmap,
};
use antecedent_discovery::{
    DiscoveryConstraints, DiscoveryWorkspace, Fci, Lpcmci, Pc, PcmciPlus, Rfci, ScoredLink,
    TemporalConstraints,
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
    ctx
}

fn links(links: &[ScoredLink]) -> Vec<String> {
    let mut rows: Vec<String> = links.iter().map(|l| format!("{l:?}")).collect();
    rows.sort();
    rows
}

fn sepsets(sepsets: &antecedent_discovery::PcSepsets) -> Vec<String> {
    let mut rows: Vec<String> = sepsets.iter().map(|(k, v)| format!("{k:?}={v:?}")).collect();
    rows.sort();
    rows
}

fn static_constraints() -> DiscoveryConstraints {
    DiscoveryConstraints {
        temporal: TemporalConstraints {
            max_lag: Lag::CONTEMPORANEOUS,
            min_lag: Lag::CONTEMPORANEOUS,
        },
        max_cond_size: 4,
        ..DiscoveryConstraints::default()
    }
}

#[test]
fn static_skeleton_searches_match_across_thread_counts() {
    let (data, vars) = static_data(14, 400, 5, 5.0);
    let run = |threads: u32| {
        let ctx = context(threads);
        let pc = Pc::new()
            .with_fdr(false)
            .with_constraints(static_constraints())
            .run(&data, &vars, &mut DiscoveryWorkspace::default(), &ctx)
            .unwrap();
        let fci = Fci::new()
            .with_constraints(static_constraints())
            .run(&data, &vars, &mut DiscoveryWorkspace::default(), &ctx)
            .unwrap();
        let rfci = Rfci::new()
            .with_constraints(static_constraints())
            .run(&data, &vars, &mut DiscoveryWorkspace::default(), &ctx)
            .unwrap();
        (
            (links(&pc.evidence.links), sepsets(&pc.sepsets), pc.performance.ci_tests),
            (links(&fci.evidence.links), sepsets(&fci.sepsets), fci.performance.ci_tests),
            (links(&rfci.evidence.links), sepsets(&rfci.sepsets), rfci.performance.ci_tests),
        )
    };
    let serial = run(1);
    assert!(serial.0.2 > 100, "fixture must exercise deep conditioning-set searches");
    assert_eq!(serial, run(4));
    assert_eq!(serial, run(3));
}

#[test]
fn temporal_searches_match_across_thread_counts() {
    let (data, vars) = temporal_data(6, 600, 3);
    let constraints = DiscoveryConstraints {
        temporal: TemporalConstraints { max_lag: Lag::from_raw(2), min_lag: Lag::CONTEMPORANEOUS },
        max_cond_size: 3,
        ..DiscoveryConstraints::default()
    };
    let run = |threads: u32| {
        let ctx = context(threads);
        let plus = PcmciPlus::new()
            .with_fdr(false)
            .with_constraints(constraints.clone())
            .run(&data, &vars, &mut DiscoveryWorkspace::default(), &ctx)
            .unwrap();
        let lpcmci = Lpcmci::new()
            .with_fdr(false)
            .with_constraints(constraints.clone())
            .run(&data, &vars, &mut DiscoveryWorkspace::default(), &ctx)
            .unwrap();
        (
            (links(&plus.evidence.links), sepsets(&plus.sepsets), plus.performance.ci_tests),
            (links(&lpcmci.evidence.links), sepsets(&lpcmci.sepsets), lpcmci.performance.ci_tests),
        )
    };
    let serial = run(1);
    assert!(serial.0.2 > 100 && serial.1.2 > 100);
    assert_eq!(serial, run(4));
}
