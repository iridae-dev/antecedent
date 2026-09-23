//! Coverage tallies are reproducible bit for bit.
//!
//! A coverage record stands until the code it measured changes, because the
//! measurement is deterministic: every replicate's data and every resample are
//! drawn from fixed seeds, so re-running unchanged code reproduces the same
//! tally. That property is what lets `scripts/gate_calibration_attestation.sh`
//! attest a record by the commit it was measured at instead of re-measuring it.
//! These cells run the calibration gate's own harness
//! (`crates/antecedent/tests/common/calibration.rs`) on a few small replicates
//! twice with the same seeds and require identical tallies and identical
//! per-replicate estimates, so hidden nondeterminism (unordered iteration, a
//! clock- or address-derived seed, an unseeded resample) fails ordinary CI in
//! seconds instead of surfacing as drift in a measurement hours long.
//!
//! One cell is analytic (the fit's arithmetic only); the other draws a
//! bootstrap from the execution context's seeded RNG. A different seed must
//! change the result, so identical tallies cannot come from a pipeline that
//! ignores its seeds.
//!
//! The sample-size grid keeps that property per grid point: this binary
//! re-runs itself at each `ANTECEDENT_CALIBRATION_GRID_POINT` and requires the
//! same point to reproduce its tally, the points to measure different data at
//! strictly growing sample sizes, and the unset variable to be exactly the
//! base point (so the base point's data is the data measured before the grid).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec,
    RoleHint, SmallRoleSet, ValueType, VariableId,
};
use antecedent_data::column::{Float64Column, ValidityBitmap};
use antecedent_data::{OwnedColumn, OwnedColumnarStorage, TabularData};
use antecedent_estimate::{
    EstimationWorkspace, LinearAdjustmentAte, PropensityEstimationWorkspace, PropensityWeighting,
};
use antecedent_expr::{ExprId, IdentifiedEstimand};

#[path = "../../antecedent/tests/common/calibration.rs"]
mod calibration;

use calibration::{
    BASE_GRID_POINT, CoverageTally, GRID_POINT_ENV, GRID_POINTS, SHORT_SERIES_MAX_BASE, SampleGrid,
    Z95, gaussian, grid_for, grid_n, grid_point, grid_seed, normal_interval, uniform,
};

const TRUE_ATE: f64 = 2.0;
const ROWS: usize = 120;
const REPLICATES: u64 = 6;
const BOOTSTRAP_REPLICATES: u32 = 12;

/// The tally plus every replicate's estimate and standard error, as bits.
#[derive(Debug, PartialEq, Eq)]
struct Fingerprint {
    tally: String,
    rate: u64,
    mean_length: u64,
    estimates: Vec<(u64, u64)>,
}

fn fingerprint(tally: &CoverageTally, estimates: Vec<(u64, u64)>) -> Fingerprint {
    Fingerprint {
        tally: format!("{tally:?}"),
        rate: tally.rate().to_bits(),
        mean_length: tally.mean_length().to_bits(),
        estimates,
    }
}

fn column(id: u32, values: Vec<f64>) -> OwnedColumn {
    let n = values.len();
    OwnedColumn::Float64(
        Float64Column::new(
            VariableId::from_raw(id),
            Arc::from(values),
            ValidityBitmap::all_valid(n),
        )
        .unwrap(),
    )
}

/// Binary treatment with a logistic propensity in `z`; `y = 2·t + 1.5·z + noise`.
fn confounded(seed: u64) -> TabularData {
    confounded_rows(ROWS, seed)
}

fn confounded_rows(rows: usize, seed: u64) -> TabularData {
    let mut normal = gaussian(seed);
    let mut unit = uniform(seed, 0x7EA7);
    let (mut t, mut y, mut z) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..rows {
        let zi = normal();
        let p = 1.0 / (1.0 + (-0.8 * zi).exp());
        let ti = if unit() < p { 1.0 } else { 0.0 };
        t.push(ti);
        y.push(TRUE_ATE * ti + 1.5 * zi + 0.5 * normal());
        z.push(zi);
    }
    let mut schema = CausalSchemaBuilder::new();
    for (name, role) in [
        ("t", RoleHint::TreatmentCandidate),
        ("y", RoleHint::OutcomeCandidate),
        ("z", RoleHint::Context),
    ] {
        schema
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
    let columns = vec![column(0, t), column(1, y), column(2, z)];
    TabularData::new(
        OwnedColumnarStorage::try_new(schema.build().unwrap(), columns, None, None).unwrap(),
    )
}

fn query() -> AverageEffectQuery {
    AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
}

fn backdoor_z() -> IdentifiedEstimand {
    IdentifiedEstimand::backdoor(
        "backdoor.adjustment",
        Arc::from([VariableId::from_raw(2)]),
        ExprId::from_raw(0),
    )
}

/// Analytic-SE linear adjustment over `REPLICATES` datasets seeded from `base`.
fn analytic_cell(base: u64) -> Fingerprint {
    analytic_cell_rows(ROWS, base)
}

fn analytic_cell_rows(rows: usize, base: u64) -> Fingerprint {
    let estimator =
        LinearAdjustmentAte { bootstrap_replicates: 0, ..LinearAdjustmentAte::default() };
    let ctx = ExecutionContext::for_tests(base);
    let mut tally = CoverageTally::new("determinism.linear_adjustment", 0.95);
    let mut estimates = Vec::new();
    for replicate in 0..REPLICATES {
        let data = confounded_rows(rows, base + replicate);
        let prepared = estimator.prepare(&data, &backdoor_z(), &query()).unwrap();
        let mut workspace = EstimationWorkspace::default();
        let effect = estimator.fit(&prepared, &mut workspace, &ctx, AssumptionSet::new()).unwrap();
        tally.record(normal_interval(effect.ate, Some(effect.se_analytic), Z95), TRUE_ATE);
        estimates.push((effect.ate.to_bits(), effect.se_analytic.to_bits()));
    }
    fingerprint(&tally, estimates)
}

/// Bootstrap-SE Hajek IPW: every replicate resamples from the context's seeded RNG.
fn bootstrap_cell(base: u64) -> Fingerprint {
    let estimator = PropensityWeighting {
        bootstrap_replicates: BOOTSTRAP_REPLICATES,
        ..PropensityWeighting::new()
    };
    let mut tally = CoverageTally::new("determinism.ipw_bootstrap", 0.95);
    let mut estimates = Vec::new();
    for replicate in 0..REPLICATES {
        let ctx = ExecutionContext::for_tests(base + replicate);
        let data = confounded(base + replicate);
        let prepared = estimator.prepare(&data, &backdoor_z(), &query()).unwrap();
        let mut workspace = PropensityEstimationWorkspace::default();
        let effect = estimator.fit(&prepared, &mut workspace, &ctx, AssumptionSet::new()).unwrap();
        let se = effect.se_bootstrap.expect("a 12-replicate bootstrap on 120 rows reports an SE");
        tally.record(normal_interval(effect.ate, Some(se), Z95), TRUE_ATE);
        estimates.push((effect.ate.to_bits(), se.to_bits()));
    }
    fingerprint(&tally, estimates)
}

#[test]
fn analytic_coverage_tally_is_reproducible_bit_for_bit() {
    let first = analytic_cell(4_100);
    assert_eq!(first, analytic_cell(4_100), "same seeds, different analytic tally");
    assert_ne!(first.estimates, analytic_cell(4_200).estimates, "the tally ignores its seeds");
}

#[test]
fn bootstrap_coverage_tally_is_reproducible_bit_for_bit() {
    let first = bootstrap_cell(5_100);
    assert_eq!(first, bootstrap_cell(5_100), "same seeds, different bootstrap tally");
    // Same data, another resampling stream: only the bootstrap seed differs.
    let estimator = PropensityWeighting {
        bootstrap_replicates: BOOTSTRAP_REPLICATES,
        ..PropensityWeighting::new()
    };
    let data = confounded(5_100);
    let prepared = estimator.prepare(&data, &backdoor_z(), &query()).unwrap();
    let se = |seed| {
        let mut workspace = PropensityEstimationWorkspace::default();
        estimator
            .fit(
                &prepared,
                &mut workspace,
                &ExecutionContext::for_tests(seed),
                AssumptionSet::new(),
            )
            .unwrap()
            .se_bootstrap
            .unwrap()
            .to_bits()
    };
    assert_eq!(first.estimates[0].1, se(5_100), "the cell's first SE is this resample's");
    assert_ne!(se(5_100), se(9_999), "the bootstrap ignores the context seed");
}

#[test]
fn sample_grids_grow_strictly_and_keep_the_base_point() {
    for grid in [SampleGrid::STANDARD, SampleGrid::SHORT_SERIES, SampleGrid::HEAVY] {
        for base in [40, 60, 80, 100, 160, 300, 400, 500, 600, 800, 1000, 1200, 2500] {
            let points: Vec<usize> = (0..GRID_POINTS).map(|k| grid.n_at(k, base)).collect();
            assert_eq!(
                points[BASE_GRID_POINT], base,
                "{}: the base point is the design",
                grid.name
            );
            assert!(points.windows(2).all(|w| w[0] < w[1]), "{}: {points:?}", grid.name);
        }
    }
    assert_eq!(SampleGrid::STANDARD.n_at(0, 500), 250);
    assert_eq!(SampleGrid::STANDARD.n_at(2, 500), 1000);
    assert_eq!(SampleGrid::SHORT_SERIES.n_at(0, 60), 45);
    assert_eq!(SampleGrid::HEAVY.n_at(2, 1000), 1500);
    assert_eq!(grid_for(60), SampleGrid::SHORT_SERIES, "a 60-step series is short");
    assert_eq!(grid_for(SHORT_SERIES_MAX_BASE + 1), SampleGrid::STANDARD);
}

/// Prints this process's grid-point fingerprint when run as the probe child.
#[test]
fn grid_point_fingerprint_probe() {
    if std::env::var_os("ANTECEDENT_GRID_DETERMINISM_PROBE").is_none() {
        return;
    }
    let rows = grid_n(ROWS);
    println!(
        "grid-probe point={} rows={rows} salt={} fingerprint={:?}",
        grid_point(),
        grid_seed(0),
        analytic_cell_rows(rows, 6_100)
    );
}

fn probe(point: Option<usize>) -> String {
    let mut command = std::process::Command::new(std::env::current_exe().unwrap());
    command
        .args(["--exact", "grid_point_fingerprint_probe", "--nocapture", "--test-threads", "1"])
        .env("ANTECEDENT_GRID_DETERMINISM_PROBE", "1")
        .env_remove(GRID_POINT_ENV);
    if let Some(point) = point {
        command.env(GRID_POINT_ENV, point.to_string());
    }
    let output = command.output().unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .find_map(|line| line.find("grid-probe ").map(|at| line[at..].to_string()))
        .expect("the probe prints its fingerprint")
}

#[test]
fn every_grid_point_is_deterministic_and_the_unset_point_is_the_base() {
    let points: Vec<String> = (0..GRID_POINTS).map(|k| probe(Some(k))).collect();
    for (k, line) in points.iter().enumerate() {
        assert_eq!(line, &probe(Some(k)), "grid point {k} is not reproducible");
    }
    assert_eq!(probe(None), points[BASE_GRID_POINT], "unset must be the base point");
    assert!(points[BASE_GRID_POINT].contains(" salt=0 "), "the base point is unsalted");
    let fingerprints: Vec<&str> =
        points.iter().map(|line| line.split(" fingerprint=").nth(1).unwrap()).collect();
    assert!(fingerprints[0] != fingerprints[1] && fingerprints[1] != fingerprints[2]);
    // The base point measures exactly the pre-grid data.
    assert!(
        points[BASE_GRID_POINT].ends_with(&format!("{:?}", analytic_cell_rows(ROWS, 6_100))),
        "the base point's tally is the one measured without the grid"
    );
}
