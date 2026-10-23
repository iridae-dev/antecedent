//! Static FCI search: the conditional-independence tests, Possible-D-Sep phase and orientation.
//!
//! The timed section is one complete `Fci::run` on a linear-Gaussian system with one omitted
//! common cause, so the search has tests to run and a latent confounder to leave unresolved.
//! The CI-test count is read from the result inside the section. Every repeat is retained and the
//! median is reported; the fastest run is never selected. Prints the sample times, the test count
//! and a digest of the scored links and the oriented graph, so a change to the search can be
//! checked for both speed and unchanged output.
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
    Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
};
use antecedent_discovery::{DiscoveryConstraints, DiscoveryWorkspace, Fci, TemporalConstraints};

/// Observed variables; the data generator draws one more and omits it.
const OBSERVED: usize = 16;
const ROWS: usize = 2000;
const RUNS: usize = 5;

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

/// Sparse linear-Gaussian DAG over `OBSERVED + 1` nodes in index order; node 0 is the omitted
/// common cause and is dropped from the returned table.
fn latent_confounded_data(seed: u64) -> (TabularData, Vec<VariableId>) {
    let total = OBSERVED + 1;
    let mut rng = Noise(seed);
    let mut parents: Vec<Vec<(usize, f64)>> = vec![Vec::new(); total];
    for (j, slot) in parents.iter_mut().enumerate().skip(1) {
        for i in 0..j {
            if rng.next() < 3.0 / total as f64 {
                slot.push((i, 0.5 + 0.5 * rng.next()));
            }
        }
    }
    // The omitted node drives at least two observed variables.
    for child in [1usize, 2] {
        if !parents[child].iter().any(|&(i, _)| i == 0) {
            parents[child].push((0, 0.8));
        }
    }
    let mut cols = vec![vec![0.0; ROWS]; total];
    for r in 0..ROWS {
        for j in 0..total {
            let mut v = rng.gauss();
            for &(i, w) in &parents[j] {
                v += w * cols[i][r];
            }
            cols[j][r] = v;
        }
    }
    let vars: Vec<_> =
        (0..OBSERVED).map(|i| VariableId::from_raw(u32::try_from(i).unwrap())).collect();
    let columns = cols
        .into_iter()
        .skip(1)
        .enumerate()
        .map(|(i, v)| {
            OwnedColumn::Float64(
                Float64Column::new(vars[i], Arc::from(v), ValidityBitmap::all_valid(ROWS)).unwrap(),
            )
        })
        .collect();
    let storage = OwnedColumnarStorage::try_new(schema(OBSERVED), columns, None, None).unwrap();
    (TabularData::new(storage), vars)
}

fn context(threads: u32) -> ExecutionContext {
    let mut ctx = ExecutionContext::for_tests(1);
    ctx.parallelism = Parallelism::bounded(NonZeroThreadCount::new(threads).unwrap());
    ctx
}

fn main() {
    let (data, vars) = latent_confounded_data(7);
    let fci = Fci::new().with_fdr(false).with_constraints(DiscoveryConstraints {
        temporal: TemporalConstraints {
            max_lag: Lag::CONTEMPORANEOUS,
            min_lag: Lag::CONTEMPORANEOUS,
        },
        max_cond_size: 3,
        ..DiscoveryConstraints::default()
    });

    for threads in [1u32, 4] {
        let ctx = context(threads);
        let mut samples = Vec::with_capacity(RUNS);
        for _ in 0..RUNS {
            let started = Instant::now();
            let result = fci.run(&data, &vars, &mut DiscoveryWorkspace::default(), &ctx).unwrap();
            samples.push((started.elapsed(), result));
        }
        println!(
            "fci_search control=fci threads={threads} samples_ns={:?}",
            samples.iter().map(|(time, _)| time.as_nanos()).collect::<Vec<_>>()
        );
        samples.sort_by_key(|(time, _)| *time);
        let (elapsed, r) = samples.remove(RUNS / 2);
        let mut rows: Vec<String> = r.evidence.links.iter().map(|l| format!("{l:?}")).collect();
        rows.sort();
        let mut hasher = DefaultHasher::new();
        (rows, format!("{:?}", r.evidence.graph)).hash(&mut hasher);
        println!(
            "fci     threads={threads} {elapsed:>9.1?} ci_tests={} links={} digest={:016x}",
            r.performance.ci_tests,
            r.evidence.links.len(),
            hasher.finish()
        );
    }
}
