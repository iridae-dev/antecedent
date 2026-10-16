//! Discovery on a row split followed by an adjusted effect, through the facade.
//!
//! Oracles: the effect is the least-squares coefficient of the treatment on the analysis rows
//! given the treatment's parents, checked against a solver written here; the true coefficient
//! of the generating law lies inside the interval; the basis is keyed under the split
//! structure, never the known-graph key; the attestation is a lookup against current records
//! and is replayed bit for bit when it holds.

#![allow(clippy::float_cmp, reason = "the declared level and replayed values are compared exactly")]
use antecedent::analysis::sample_split_analysis::{SplitRule, discover_on_split};
use antecedent::analysis::sample_split_effect::{
    ALPHA, CI_TEST, DISCOVERY_FRACTION, ESTIMATOR, SPLIT_STRUCTURE, estimate_effect_on_split,
    measure_source,
};
use antecedent::analysis::selective_discovery::{
    CiTestSpec, StaticAlgorithm, StaticDiscoveryContract,
};
use antecedent_core::{ExecutionContext, VariableId};
use antecedent_data::{TableView, TabularData};
use antecedent_io::measured_inference::MeasuredInference;
use antecedent_io::measured_inference::split_effect::SplitEffectSource;

const SEED: u64 = 4_242;
const SPLIT_SEED: u64 = 17;
const ROWS: usize = 6_000;
/// A row count whose halves lie inside the measured analysis range.
const MEASURED_ROWS: usize = 1_200;
/// Columns `z1 w x y z3`; the true coefficient of `x` on `y` is `XY`.
const XY: f64 = 1.2;

/// `SplitMix64` stream: every draw is a finalized hash of a counter, so consecutive draws
/// carry none of the serial structure of a bare congruential generator.
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

/// `z1 -> x <- w`, `x -> y <- z3`, `w -> y`: every edge is oriented by a collider. The weights
/// keep every adjacent pair's partial correlation above 0.3 under every conditioning set
/// (a positive `w -> x` and `w -> y` of equal size nearly cancel `w ⟂ x | y`, and PC then
/// drops the edge), so the skeleton is recovered at 3000 discovery rows.
fn collider_table(seed: u64) -> TabularData {
    collider_table_of(seed, ROWS)
}

fn collider_table_of(seed: u64, rows: usize) -> TabularData {
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

/// A chain `a -> b -> c -> d -> e`: no collider, so no edge is oriented.
fn chain_table(seed: u64) -> TabularData {
    let mut rng = Lcg(seed);
    let mut cols: Vec<Vec<f64>> = vec![vec![]; 5];
    for _ in 0..ROWS {
        let mut previous = rng.normal();
        cols[0].push(previous);
        for col in cols.iter_mut().skip(1) {
            previous = 0.8 * previous + 0.6 * rng.normal();
            col.push(previous);
        }
    }
    TabularData::from_f64_columns([
        ("a", cols[0].as_slice()),
        ("b", cols[1].as_slice()),
        ("c", cols[2].as_slice()),
        ("d", cols[3].as_slice()),
        ("e", cols[4].as_slice()),
    ])
    .unwrap()
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(SEED)
}

fn v(index: u32) -> VariableId {
    VariableId::from_raw(index)
}

/// Solve the normal equations by Gauss-Jordan elimination, independently of the library.
fn least_squares(rows: &[Vec<f64>], y: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let k = rows[0].len();
    let mut a = vec![vec![0.0; k + 1]; k];
    for (row, target) in rows.iter().zip(y) {
        for i in 0..k {
            for j in 0..k {
                a[i][j] += row[i] * row[j];
            }
            a[i][k] += row[i] * target;
        }
    }
    let mut inverse_diag = vec![0.0; k];
    // Coefficients, then the diagonal of the inverse by solving for each unit vector.
    let solve = |rhs: &[f64]| -> Vec<f64> {
        let mut m: Vec<Vec<f64>> = (0..k)
            .map(|i| {
                let mut r = a[i][..k].to_vec();
                r.push(rhs[i]);
                r
            })
            .collect();
        for col in 0..k {
            let pivot =
                (col..k).max_by(|&p, &q| m[p][col].abs().total_cmp(&m[q][col].abs())).unwrap();
            m.swap(col, pivot);
            let d = m[col][col];
            for c in col..=k {
                m[col][c] /= d;
            }
            for r in 0..k {
                if r != col {
                    let f = m[r][col];
                    for c in col..=k {
                        let sub = f * m[col][c];
                        m[r][c] -= sub;
                    }
                }
            }
        }
        (0..k).map(|i| m[i][k]).collect()
    };
    let moments: Vec<f64> = (0..k).map(|i| a[i][k]).collect();
    let beta = solve(&moments);
    for (i, slot) in inverse_diag.iter_mut().enumerate() {
        let mut unit = vec![0.0; k];
        unit[i] = 1.0;
        *slot = solve(&unit)[i];
    }
    (beta, inverse_diag)
}

fn column(table: &TabularData, index: u32) -> Vec<f64> {
    table.float64_values(v(index)).unwrap()
}

#[test]
fn the_effect_is_the_adjusted_least_squares_coefficient_on_the_analysis_rows() {
    let data = collider_table(1);
    let estimate = estimate_effect_on_split(&data, v(2), v(3), SPLIT_SEED, &ctx()).unwrap();
    // The graph is fully oriented, so the treatment's parents are z1 and w.
    assert_eq!(estimate.adjustment_set(), &[0, 1]);
    assert_eq!(estimate.discovery_rows(), (ROWS / 2) as u64);
    assert_eq!(estimate.analysis_rows(), (ROWS / 2) as u64);
    assert_eq!(estimate.level(), 0.95);

    // The analysis rows the construction used: the split the facade's own discovery makes.
    let mut contract = StaticDiscoveryContract::new(
        StaticAlgorithm::Pc,
        (0..5).map(v).collect(),
        CiTestSpec::named(CI_TEST),
        SEED,
    );
    contract.fdr = None;
    contract.constraints.alpha = ALPHA;
    let split = discover_on_split(
        &contract,
        &data,
        SplitRule::SeededPermutation { seed: SPLIT_SEED, discovery_fraction: DISCOVERY_FRACTION },
        &ctx(),
    )
    .unwrap();
    let rows = split.analysis_rows();
    let (z1, w, x, y) = (column(rows, 0), column(rows, 1), column(rows, 2), column(rows, 3));
    let design: Vec<Vec<f64>> = (0..z1.len()).map(|i| vec![1.0, x[i], z1[i], w[i]]).collect();
    let (beta, inverse_diag) = least_squares(&design, &y);
    assert!((estimate.effect() - beta[1]).abs() < 1e-9, "{} vs {}", estimate.effect(), beta[1]);

    let n = z1.len();
    let rss: f64 = design
        .iter()
        .zip(&y)
        .map(|(row, target)| {
            let fitted: f64 = row.iter().zip(&beta).map(|(a, b)| a * b).sum();
            (target - fitted).powi(2)
        })
        .sum();
    let se = (rss / (n - 4) as f64 * inverse_diag[1]).sqrt();
    assert!((estimate.standard_error() - se).abs() < 1e-9);
    let (lo, hi) = estimate.interval();
    assert!(lo < estimate.effect() && estimate.effect() < hi);
    // About 1.96 standard errors either side: a Student quantile on about 3000 degrees of freedom.
    assert!(((hi - lo) / (2.0 * se) - 1.9608).abs() < 5e-3);
    assert!(lo <= XY && XY <= hi, "truth {XY} outside [{lo}, {hi}]");
}

#[test]
fn the_basis_is_keyed_under_the_split_structure_and_its_own_estimator() {
    let data = collider_table(2);
    let estimate = estimate_effect_on_split(&data, v(2), v(3), SPLIT_SEED, &ctx()).unwrap();
    let basis = estimate.keyed_basis();
    assert_eq!(&*basis.structure, SPLIT_STRUCTURE);
    assert_eq!(SPLIT_STRUCTURE, "discovered_sample_split");
    assert_eq!(&*basis.estimator, ESTIMATOR);
    assert_eq!(&*basis.dependence, "disjoint_rows");
    assert_eq!(&*basis.query, "AverageEffect");
    assert_eq!(&*basis.interval_method, "analytic_se");
    assert_eq!(&*basis.identification, "point");
    assert_eq!(basis.row_count, (ROWS / 2) as u64);
    assert!((basis.level - 0.95).abs() < 1e-12);
    // Never the keys a declared graph is measured under.
    assert_ne!(&*basis.structure, "fixed");
    assert_ne!(&*basis.structure, "graph_posterior");
    assert_ne!(&*basis.estimator, "linear.adjustment.ate");
}

#[test]
fn the_same_inputs_give_the_same_estimate_and_source() {
    let data = collider_table(3);
    let first = estimate_effect_on_split(&data, v(2), v(3), SPLIT_SEED, &ctx()).unwrap();
    let again = estimate_effect_on_split(&data, v(2), v(3), SPLIT_SEED, &ctx()).unwrap();
    assert_eq!(first, again);
    assert_eq!(first.source_artifact(), again.source_artifact());
    let other = estimate_effect_on_split(&data, v(2), v(3), SPLIT_SEED + 1, &ctx()).unwrap();
    assert_ne!(first.receipt_identity(), other.receipt_identity());
}

#[test]
fn attestation_is_a_lookup_against_current_records_and_replays_when_it_holds() {
    let data = collider_table_of(4, MEASURED_ROWS);
    let estimate = estimate_effect_on_split(&data, v(2), v(3), SPLIT_SEED, &ctx()).unwrap();
    assert_eq!(estimate.analysis_rows(), (MEASURED_ROWS / 2) as u64);
    match estimate.measure(&ctx()) {
        Ok(measured) => {
            // A current attesting record covers this construction and row count: the report
            // is calibrated, resolves to that record and a fresh consumer re-derives it.
            let report = measured.report();
            assert_eq!(report.calibration, "calibrated");
            assert_eq!(report.scalars.len(), 1);
            assert_eq!(report.scalars[0].name, "effect");
            assert_eq!(report.scalars[0].calibration.status, "calibrated");
            assert!((report.scalars[0].point - estimate.effect()).abs() < 1e-12);
            assert_eq!(report.scalars[0].lower.to_bits(), estimate.interval().0.to_bits());
            let again =
                MeasuredInference::consume(measured.export(), measured.expectation(), &ctx())
                    .unwrap();
            assert_eq!(again.export(), measured.export());
        }
        Err(error) => {
            // Until a current attesting record exists the interval is computed, not attested.
            let text = error.to_string();
            assert!(text.contains("cell_not_licensed"), "{text}");
            assert!(text.contains("measured_inference.record_not_attesting"), "{text}");
        }
    }
}

#[test]
fn an_interval_outside_the_measured_row_range_is_never_attested() {
    // 3000 analysis rows lie above the largest measured analysis count (1200), so no record can
    // cover the execution whatever has been measured: the interval is computed and
    // `measure` refuses it with cell_not_licensed rather than relabelling it.
    let data = collider_table(7);
    let estimate = estimate_effect_on_split(&data, v(2), v(3), SPLIT_SEED, &ctx()).unwrap();
    assert!(estimate.interval().0 < estimate.interval().1);
    let error = estimate.measure(&ctx()).unwrap_err();
    let text = error.to_string();
    assert!(text.contains("cell_not_licensed"), "{text}");
    assert!(text.contains("measured_inference.record_not_attesting"), "{text}");
}

#[test]
fn an_unresolved_structure_reports_no_interval() {
    let data = chain_table(5);
    let error = estimate_effect_on_split(&data, v(1), v(3), SPLIT_SEED, &ctx()).unwrap_err();
    let text = error.to_string();
    assert!(text.contains("cell_not_licensed"), "{text}");
    assert!(text.contains("split_effect.structure_unresolved"), "{text}");
}

#[test]
fn a_table_outside_the_measured_variable_count_is_refused() {
    let a: Vec<f64> = (0..40).map(|i| f64::from(i).sin()).collect();
    let b: Vec<f64> = (0..40).map(|i| f64::from(i).cos()).collect();
    let c: Vec<f64> = (0..40).map(|i| f64::from(i * i % 7)).collect();
    let data = TabularData::from_f64_columns([
        ("a", a.as_slice()),
        ("b", b.as_slice()),
        ("c", c.as_slice()),
    ])
    .unwrap();
    let error = estimate_effect_on_split(&data, v(0), v(1), SPLIT_SEED, &ctx()).unwrap_err();
    assert!(error.to_string().contains("split_effect.variable_count_not_measured"));
}

#[test]
fn a_source_that_states_another_format_or_roles_is_refused() {
    let data = collider_table(6);
    let estimate = estimate_effect_on_split(&data, v(2), v(3), SPLIT_SEED, &ctx()).unwrap();
    let source = SplitEffectSource::from_bytes(estimate.source_artifact()).unwrap();
    assert_eq!(source.to_bytes().unwrap(), estimate.source_artifact());

    let mut other_version = source.clone();
    other_version.version = 2;
    let error = measure_source(&other_version.to_bytes().unwrap(), &ctx()).unwrap_err();
    assert!(error.to_string().contains("split_effect.source_invalid"));

    let mut same_role = source;
    same_role.outcome = same_role.treatment;
    let error = measure_source(&same_role.to_bytes().unwrap(), &ctx()).unwrap_err();
    assert!(error.to_string().contains("split_effect.source_invalid"));

    // Bytes that decode but are not the canonical encoding are refused before any replay.
    let mut padded = estimate.source_artifact().to_vec();
    padded.push(0);
    assert!(measure_source(&padded, &ctx()).is_err());
}
