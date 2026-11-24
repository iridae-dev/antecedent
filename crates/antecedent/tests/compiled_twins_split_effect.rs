//! The measured sample-split effect and parent-set effect are the same computation as their
//! compiled families.
//!
//! `SplitEffect::measure`, `sample_split_effect::measure_source`, `ParentSetEffect::measure` and
//! `sample_split_parent_set_effect::measure_source` replay a canonical source and bind its
//! interval to the coverage records. The compiled `SampleSplitEffect` and
//! `SampleSplitParentSetEffect` families bind the same source into a lowered program, pass the
//! support gate of their unit and run the same replay. Each test measures one source both ways
//! under the same `ExecutionContext::for_tests` seed and compares the exported envelope, every
//! point and interval bound bit for bit and the whole calibration slot, so the calibration
//! basis and the coverage record the interval is read under are the direct path's. The fixtures
//! are those of `sample_split_effect_facade.rs` and `sample_split_parent_set_effect_facade.rs`.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::float_cmp, reason = "the bits are compared exactly")]

use antecedent::Study;
use antecedent::analysis::sample_split_effect::{estimate_effect_on_split, measure_source};
use antecedent::analysis::sample_split_parent_set_effect::{
    ParentSetOptions, estimate_parent_set_effect, measure_source as measure_parent_set_source,
};
use antecedent::planner::PlanView;
use antecedent_core::{ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_io::measured_inference::MeasuredInference;

#[path = "common/sparse_scm.rs"]
#[allow(dead_code, reason = "the shared fixture module serves several test targets")]
mod sparse_scm;

use sparse_scm::{Scm, Topology};

const SEED: u64 = 4_242;
const SPLIT_SEED: u64 = 17;
const PARENT_SET_SPLIT_SEED: u64 = 19;
/// A row count whose halves lie inside the measured analysis range.
const MEASURED_ROWS: usize = 1_200;
const PARENT_SET_ROWS: usize = 4_000;
const WEIGHT_SEED: u64 = 0x5EED_0011;
const DATA_SEED: u64 = 0xDA7A_0022;
const XY: f64 = 1.2;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(SEED)
}

fn v(index: u32) -> VariableId {
    VariableId::from_raw(index)
}

/// `SplitMix64` stream, as in `sample_split_effect_facade.rs`.
struct Lcg(u64);

impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (f64::from(u32::try_from(z >> 40).unwrap()) + 0.5) / f64::from(1u32 << 24)
    }

    fn normal(&mut self) -> f64 {
        let (u, v) = (self.unit(), self.unit());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

/// `z1 -> x <- w`, `x -> y <- z3`, `w -> y`, as in `sample_split_effect_facade.rs`.
fn collider_table(seed: u64, rows: usize) -> TabularData {
    let mut rng = Lcg(seed);
    let (mut z1, mut w, mut x, mut y, mut z3) = (vec![], vec![], vec![], vec![], vec![]);
    for _ in 0..rows {
        let (a, b, c) = (rng.normal(), rng.normal(), rng.normal());
        let t = 0.8 * a - 0.6 * b + rng.normal();
        let o = XY * t + 1.4 * b + 0.8 * c + rng.normal();
        z1.push(a);
        w.push(b);
        x.push(t);
        y.push(o);
        z3.push(c);
    }
    TabularData::from_f64_columns([
        ("z1", z1.as_slice()),
        ("w", w.as_slice()),
        ("x", x.as_slice()),
        ("y", y.as_slice()),
        ("z3", z3.as_slice()),
    ])
    .unwrap()
}

fn assert_bound(view: &PlanView, route: &str, unit: &str) {
    assert_eq!(view.route().as_str(), route);
    let decision = view.unit_support().expect("the route is bound to a unit");
    assert_eq!(decision.unit(), unit);
    assert!(decision.supported(), "{unit} is supported");
}

fn assert_same_measured(direct: &MeasuredInference, compiled: &MeasuredInference) {
    assert_eq!(direct.export(), compiled.export(), "measured envelope");
    assert_eq!(direct.source_artifact(), compiled.source_artifact(), "source");
    assert_eq!(direct.expectation(), compiled.expectation(), "retained identity");
    let (direct, compiled) = (direct.report(), compiled.report());
    assert_eq!(direct.route, compiled.route);
    assert_eq!(direct.calibration, compiled.calibration);
    assert_eq!(direct.scalars.len(), compiled.scalars.len());
    for (d, c) in direct.scalars.iter().zip(&compiled.scalars) {
        assert_eq!(d.name, c.name);
        assert_eq!(d.point.to_bits(), c.point.to_bits(), "{}: point", d.name);
        assert_eq!(d.lower.to_bits(), c.lower.to_bits(), "{}: lower", d.name);
        assert_eq!(d.upper.to_bits(), c.upper.to_bits(), "{}: upper", d.name);
        assert_eq!(d.level.to_bits(), c.level.to_bits(), "{}: level", d.name);
        assert_eq!(d.calibration, c.calibration, "{}: calibration slot", d.name);
    }
}

#[test]
fn the_compiled_split_effect_measures_the_source_as_the_direct_path_does() {
    let data = collider_table(4, MEASURED_ROWS);
    let estimate = estimate_effect_on_split(&data, v(2), v(3), SPLIT_SEED, &ctx()).unwrap();
    let compiled = Study::compile_split_effect_source(estimate.source_artifact(), &ctx()).unwrap();
    assert_eq!(compiled.source_artifact(), estimate.source_artifact());
    assert_bound(&compiled.plan_view(), "SampleSplitEffect", "discovery.sample_split_effect");
    let via_compiled = compiled.measure(&ctx()).unwrap();
    assert_same_measured(&estimate.measure(&ctx()).unwrap(), &via_compiled);
    assert_same_measured(
        &measure_source(estimate.source_artifact(), &ctx()).unwrap(),
        &via_compiled,
    );
}

#[test]
fn a_malformed_split_effect_source_is_refused_alike() {
    let data = collider_table(4, MEASURED_ROWS);
    let estimate = estimate_effect_on_split(&data, v(2), v(3), SPLIT_SEED, &ctx()).unwrap();
    let mut padded = estimate.source_artifact().to_vec();
    padded.push(0);
    let direct = measure_source(&padded, &ctx()).unwrap_err();
    let compiled = Study::compile_split_effect_source(&padded, &ctx()).unwrap();
    assert_eq!(compiled.measure(&ctx()).unwrap_err().to_string(), direct.to_string());
}

#[test]
fn the_compiled_parent_set_effect_measures_the_source_as_the_direct_path_does() {
    let variables = 10;
    let scm = Scm::new(Topology::HubConfoundedMultiPath, variables, WEIGHT_SEED + variables as u64);
    let columns = scm.simulate(PARENT_SET_ROWS, DATA_SEED + variables as u64);
    let data = TabularData::from_f64_columns(
        scm.names.iter().map(String::as_str).zip(columns.iter().map(Vec::as_slice)),
    )
    .unwrap();
    let (x, y) = (u32::try_from(scm.treatment).unwrap(), u32::try_from(scm.outcome).unwrap());
    let estimate = estimate_parent_set_effect(
        &data,
        v(x),
        v(y),
        ParentSetOptions::linear(4),
        PARENT_SET_SPLIT_SEED,
        &ctx(),
    )
    .unwrap();
    let compiled =
        Study::compile_split_parent_set_effect_source(estimate.source_artifact(), &ctx()).unwrap();
    assert_bound(
        &compiled.plan_view(),
        "SampleSplitParentSetEffect",
        "discovery.sample_split_parent_set_effect",
    );
    let via_compiled = compiled.measure(&ctx()).unwrap();
    assert_same_measured(&estimate.measure(&ctx()).unwrap(), &via_compiled);
    assert_same_measured(
        &measure_parent_set_source(estimate.source_artifact(), &ctx()).unwrap(),
        &via_compiled,
    );
}
