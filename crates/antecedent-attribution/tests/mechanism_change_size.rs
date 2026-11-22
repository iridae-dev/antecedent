//! Size and power of mechanism-change detection under repeated sampling.
//!
//! Two populations are drawn from one chain `x -> y`, `x -> z`. Under the null neither
//! mechanism changes between them; the raw per-target p-value of a calibrated test then rejects
//! at level 0.05 about 5% of the time, and the Benjamini-Hochberg adjusted p-values of the
//! two-target family reject anywhere in the family at most 5% of the time (under the global null
//! the family-wise rate of the step-up rule is its level). Each rate is held to its nominal value
//! within `Z` binomial standard deviations of the `REPLICATIONS` replications it is read from.
//! A comparison the route refuses (parents that look shifted by chance) publishes no p-value and
//! is not a replication.
//!
//! Under an intercept shift of the outcome `y` the same detector must reject `y` and leave `z`,
//! whose mechanism did not change, at its size.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_attribution::{MechanismChangeMethod, detect_mechanism_changes};
use antecedent_core::{ExecutionContext, MechanismChangeQuery, PopulationSelector, VariableId};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_model::CompiledCausalModel;

const REPLICATIONS: u64 = 400;
const ROWS: usize = 120;
const LEVEL: f64 = 0.05;
/// Half-width of every acceptance band, in binomial standard deviations.
const Z: f64 = 4.0;

/// Deterministic standard-normal stream (xorshift64* with Box-Muller).
struct Stream(u64);

impl Stream {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let bits = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11;
        (bits as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

/// `ROWS` baseline rows then `ROWS` comparison rows of the chain; `shift` moves the intercept of
/// `y` in the comparison rows only.
fn replication(seed: u64, shift: f64) -> TabularData {
    let mut stream = Stream((0x9E37_79B9_7F4A_7C15 ^ seed.wrapping_mul(0xA24B_AED4_963E_E407)) | 1);
    for _ in 0..8 {
        stream.uniform();
    }
    let n = 2 * ROWS;
    let x: Vec<f64> = (0..n).map(|_| stream.normal()).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| 1.0 + 2.0 * x[i] + if i >= ROWS { shift } else { 0.0 } + stream.normal())
        .collect();
    let z: Vec<f64> = (0..n).map(|i| -1.0 + 0.5 * x[i] + stream.normal()).collect();
    TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())])
        .unwrap()
}

fn model() -> CompiledCausalModel {
    let mut graph = Dag::with_variables(3);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
    CompiledCausalModel::compile(graph).unwrap()
}

fn query() -> MechanismChangeQuery {
    MechanismChangeQuery::new(
        [VariableId::from_raw(1), VariableId::from_raw(2)],
        PopulationSelector::TimeRange { start: 0, end: ROWS },
        PopulationSelector::TimeRange { start: ROWS, end: 2 * ROWS },
        LEVEL,
        10,
    )
}

/// Raw and adjusted p-values of `(y, z)` of every replication the route answered.
struct Answered {
    raw: Vec<[f64; 2]>,
    adjusted: Vec<[f64; 2]>,
    refused: usize,
}

fn run(method: MechanismChangeMethod, shift: f64) -> Answered {
    let (model, query) = (model(), query());
    let mut out = Answered { raw: Vec::new(), adjusted: Vec::new(), refused: 0 };
    for seed in 0..REPLICATIONS {
        let data = replication(seed + 1, shift);
        let ctx = ExecutionContext::for_tests(seed + 1);
        match detect_mechanism_changes(&model, &data, &query, method, &ctx) {
            Ok(found) => {
                assert_eq!(found.len(), 2);
                out.raw.push([found[0].p_value, found[1].p_value]);
                out.adjusted.push([
                    found[0].adjusted_p_value.expect("the family correction is applied"),
                    found[1].adjusted_p_value.expect("the family correction is applied"),
                ]);
            }
            Err(_) => out.refused += 1,
        }
    }
    assert!(
        out.refused as u64 * 10 <= REPLICATIONS,
        "{} of {REPLICATIONS} comparisons refused",
        out.refused
    );
    out
}

fn rate(hits: usize, of: usize) -> f64 {
    hits as f64 / of as f64
}

fn band(level: f64, of: usize) -> f64 {
    Z * (level * (1.0 - level) / of as f64).sqrt()
}

#[test]
fn the_raw_p_values_reject_a_true_null_at_their_level() {
    for method in [MechanismChangeMethod::MeanDiff, MechanismChangeMethod::LikelihoodRatio] {
        let answered = run(method, 0.0);
        let n = answered.raw.len();
        for (target, name) in [(0, "y"), (1, "z")] {
            let size = rate(answered.raw.iter().filter(|p| p[target] < LEVEL).count(), n);
            assert!(
                (size - LEVEL).abs() <= band(LEVEL, n),
                "{method:?} target {name}: raw rejection rate {size} over {n} replications, level {LEVEL}"
            );
        }
    }
}

#[test]
fn the_adjusted_p_values_hold_the_family_wise_rate_at_most_at_their_level() {
    for method in [MechanismChangeMethod::MeanDiff, MechanismChangeMethod::LikelihoodRatio] {
        let answered = run(method, 0.0);
        let n = answered.adjusted.len();
        let family =
            rate(answered.adjusted.iter().filter(|p| p[0] < LEVEL || p[1] < LEVEL).count(), n);
        assert!(
            family <= LEVEL + band(LEVEL, n),
            "{method:?}: family-wise rejection rate {family} over {n} replications, level {LEVEL}"
        );
        // The correction never makes a raw p-value more extreme.
        for (raw, adjusted) in answered.raw.iter().zip(&answered.adjusted) {
            assert!(adjusted[0] >= raw[0] - 1e-12 && adjusted[1] >= raw[1] - 1e-12);
        }
    }
}

#[test]
fn a_shifted_mechanism_is_rejected_and_an_unchanged_one_keeps_its_size() {
    for method in [MechanismChangeMethod::MeanDiff, MechanismChangeMethod::LikelihoodRatio] {
        // A shift of one noise standard deviation over 120 rows per population is far beyond
        // the detection threshold of either test.
        let answered = run(method, 1.0);
        let n = answered.raw.len();
        let power = rate(answered.adjusted.iter().filter(|p| p[0] < LEVEL).count(), n);
        assert!(power >= 0.95, "{method:?}: the shifted mechanism was rejected in {power}");
        let unchanged = rate(answered.raw.iter().filter(|p| p[1] < LEVEL).count(), n);
        assert!(
            unchanged <= LEVEL + band(LEVEL, n),
            "{method:?}: the unchanged mechanism was rejected in {unchanged}"
        );
    }
}
