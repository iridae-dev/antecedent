//! Replicate-loop throughput of the effect refuters.
//!
//! Prints wall time and the bits of each report's numbers per thread count, so a change to the
//! replicate loops can be checked for both speed and bit-identical output.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;
use std::time::{Duration, Instant};

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec,
    NonZeroThreadCount, Parallelism, RoleHint, SmallRoleSet, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
};
use antecedent_estimate::{EstimationWorkspace, LinearAdjustmentAte};
use antecedent_expr::ExprId;
use antecedent_identify::IdentifiedEstimand;
use antecedent_validate::{
    BootstrapRefute, DataSubsetRefuter, PlaceboMode, PlaceboTreatment, RandomCommonCause,
    RefutationProblem, RefutationReport, UnobservedCommonCause,
};

const ROWS: usize = 20_000;
const COVARIATES: usize = 5;
const REPLICATES: u32 = 120;
const RUNS: usize = 3;

fn data() -> TabularData {
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut uniform = move || {
        state =
            state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((state >> 11) as f64 / (1_u64 << 53) as f64) - 0.5
    };
    let mut columns = vec![vec![0.0; ROWS]; 2 + COVARIATES];
    for i in 0..ROWS {
        let mut z_sum = 0.0;
        for j in 0..COVARIATES {
            let z = uniform();
            columns[2 + j][i] = z;
            z_sum += z * (j as f64 + 1.0);
        }
        let t = z_sum + uniform();
        columns[0][i] = t;
        columns[1][i] = 0.5 * t + z_sum + uniform();
    }
    let mut builder = CausalSchemaBuilder::new();
    for j in 0..columns.len() {
        let (name, role) = match j {
            0 => ("t".to_owned(), RoleHint::TreatmentCandidate),
            1 => ("y".to_owned(), RoleHint::OutcomeCandidate),
            _ => (format!("z{}", j - 2), RoleHint::Context),
        };
        builder
            .add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(role),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
    }
    let cols = columns
        .into_iter()
        .enumerate()
        .map(|(j, values)| {
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(u32::try_from(j).unwrap()),
                    Arc::from(values),
                    ValidityBitmap::all_valid(ROWS),
                )
                .unwrap(),
            )
        })
        .collect();
    TabularData::new(
        OwnedColumnarStorage::try_new(builder.build().unwrap(), cols, None, None).unwrap(),
    )
}

fn best_of(mut run: impl FnMut() -> RefutationReport) -> (Duration, RefutationReport) {
    let mut best: Option<(Duration, RefutationReport)> = None;
    for _ in 0..RUNS {
        let t = Instant::now();
        let out = run();
        let elapsed = t.elapsed();
        if best.as_ref().is_none_or(|(d, _)| elapsed < *d) {
            best = Some((elapsed, out));
        }
    }
    best.unwrap()
}

fn main() {
    let data = data();
    let adjustment: Vec<VariableId> =
        (0..COVARIATES).map(|j| VariableId::from_raw(u32::try_from(j + 2).unwrap())).collect();
    let estimand = IdentifiedEstimand::backdoor(
        "backdoor.adjustment",
        Arc::from(adjustment),
        ExprId::from_raw(0),
    );
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    let est = LinearAdjustmentAte { bootstrap_replicates: 0, ..LinearAdjustmentAte::new() };
    let prep = est.prepare(&data, &estimand, &query).unwrap();
    let ctx = ExecutionContext::for_tests(5);
    let original =
        est.fit(&prep, &mut EstimationWorkspace::default(), &ctx, AssumptionSet::new()).unwrap();
    let problem = RefutationProblem::new(&data, &estimand, &query, &original, None, None);

    let rcc = RandomCommonCause { replicates: REPLICATES, ..RandomCommonCause::new() };
    let placebo = PlaceboTreatment { replicates: REPLICATES, ..PlaceboTreatment::new() };
    let permute = PlaceboTreatment { mode: PlaceboMode::Permute, ..placebo.clone() };
    let subset = DataSubsetRefuter { replicates: REPLICATES, ..DataSubsetRefuter::new() };
    let unobserved =
        UnobservedCommonCause { replicates: REPLICATES, ..UnobservedCommonCause::new() };
    let bootstrap = BootstrapRefute { replicates: REPLICATES, ..BootstrapRefute::new() };

    for threads in [1u32, 4] {
        let mut ctx = ExecutionContext::for_tests(5);
        ctx.parallelism = Parallelism::bounded(NonZeroThreadCount::new(threads).unwrap());
        ctx =
            ctx.with_admission(antecedent_core::execution::AdmissionPool::fixed(threads, u64::MAX));
        let report = |name: &str, run: &dyn Fn(&mut EstimationWorkspace) -> RefutationReport| {
            let (elapsed, r) = best_of(|| run(&mut EstimationWorkspace::default()));
            println!(
                "{name:<14} threads={threads} {elapsed:>9.1?} refuted={:016x} p={:016x} passed={}",
                r.refuted_ate.to_bits(),
                r.comparison.to_bits(),
                r.passed
            );
        };
        report("rcc", &|ws| rcc.refute(&problem, ws, &ctx).unwrap());
        report("placebo", &|ws| placebo.refute(&problem, ws, &ctx).unwrap());
        report("placebo-perm", &|ws| permute.refute(&problem, ws, &ctx).unwrap());
        report("data-subset", &|ws| subset.refute(&problem, ws, &ctx).unwrap());
        report("unobserved-cc", &|ws| unobserved.refute(&problem, ws, &ctx).unwrap());
        report("bootstrap", &|ws| bootstrap.refute(&problem, ws, &ctx).unwrap());
    }
}
