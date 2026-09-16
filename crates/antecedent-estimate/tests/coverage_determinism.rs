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
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss)]

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

use calibration::{CoverageTally, Z95, gaussian, normal_interval, uniform};

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
    let mut normal = gaussian(seed);
    let mut unit = uniform(seed, 0x7EA7);
    let (mut t, mut y, mut z) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..ROWS {
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
    let estimator =
        LinearAdjustmentAte { bootstrap_replicates: 0, ..LinearAdjustmentAte::default() };
    let ctx = ExecutionContext::for_tests(base);
    let mut tally = CoverageTally::new("determinism.linear_adjustment", 0.95);
    let mut estimates = Vec::new();
    for replicate in 0..REPLICATES {
        let data = confounded(base + replicate);
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
