//! Discovery followed by analysis on a row split, through the facade.
//!
//! Oracles: the seeded permutation assigns `round(fraction * n)` rows to discovery, so the part
//! sizes are arithmetic; a `row` column carrying each row's index in the unsplit table witnesses
//! that the parts are disjoint and cover the table; and the discovery on the split must equal,
//! bit for bit, a direct run of the same contract on the discovery part.

use std::collections::BTreeSet;
use std::sync::Arc;

use antecedent::analysis::sample_split_analysis::{
    AnalysisEffectRequest, SplitAnalysisEffect, SplitRule, WHOLE_METHOD_CALIBRATION,
    discover_on_split, split_evidence_sources,
};
use antecedent::analysis::selective_discovery::discovery_data_source;
use antecedent_contract::evidence_ledger::{EvidenceLedger, Overlap};
use antecedent_core::{ExecutionContext, VariableId};
use antecedent_data::{OwnedColumnarStorage, TableView, TabularData, ValidityBitmap};
use antecedent_discovery::selective::{
    CiTestSpec, StaticAlgorithm, StaticDiscoveryContract, StaticSelectiveDiscovery, column_digest,
    digest_hex,
};
use antecedent_stats::WeightedPartialCorrelation;

const SEED: u64 = 8_675;
const ROWS: usize = 500;

struct Lcg(u64);

impl Lcg {
    fn normal(&mut self) -> f64 {
        let mut unit = || {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (f64::from(u32::try_from(self.0 >> 40).unwrap()) + 0.5) / f64::from(1u32 << 24)
        };
        let (u, v) = (unit(), unit());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

/// `x -> y -> z` plus a `row` column holding each row's index (not searched).
fn table(seed: u64) -> TabularData {
    let mut rng = Lcg(seed);
    let (mut x, mut y, mut z, mut row) = (vec![], vec![], vec![], vec![]);
    for i in 0..ROWS {
        let a = rng.normal();
        let b = 0.8 * a + 0.6 * rng.normal();
        let c = 0.8 * b + 0.6 * rng.normal();
        x.push(a);
        y.push(b);
        z.push(c);
        row.push(i as f64);
    }
    TabularData::from_f64_columns([
        ("x", x.as_slice()),
        ("y", y.as_slice()),
        ("z", z.as_slice()),
        ("row", row.as_slice()),
    ])
    .unwrap()
}

fn contract() -> StaticDiscoveryContract {
    let mut contract = StaticDiscoveryContract::new(
        StaticAlgorithm::Pc,
        (0..3).map(VariableId::from_raw).collect(),
        CiTestSpec::named("parcorr"),
        SEED,
    );
    contract.fdr = None;
    contract.constraints.alpha = 0.001;
    contract
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(SEED)
}

fn rule(seed: u64) -> SplitRule {
    SplitRule::SeededPermutation { seed, discovery_fraction: 0.4 }
}

fn row_ids(data: &TabularData) -> Vec<usize> {
    let ids = data.float64_values(VariableId::from_raw(3)).unwrap();
    ids.iter().map(|r| r.to_string().parse().unwrap()).collect()
}

#[test]
fn the_parts_are_disjoint_cover_the_table_and_discovery_sees_only_its_part() {
    let data = table(1);
    let split = discover_on_split(&contract(), &data, rule(5), &ctx()).unwrap();

    let receipt = split.receipt();
    assert_eq!(receipt.n_rows, ROWS);
    assert_eq!(receipt.n_discovery, 200); // round(0.4 * 500)
    assert_eq!(receipt.n_analysis, 300);
    split.verify().unwrap();

    let discovery: BTreeSet<usize> = row_ids(split.discovery_rows()).into_iter().collect();
    let analysis: BTreeSet<usize> = row_ids(split.analysis_rows()).into_iter().collect();
    assert_eq!((discovery.len(), analysis.len()), (200, 300));
    assert!(discovery.is_disjoint(&analysis));
    assert_eq!(
        discovery.union(&analysis).copied().collect::<Vec<_>>(),
        (0..ROWS).collect::<Vec<_>>()
    );

    // Discovery ran on the discovery part and nothing else: it matches a direct run on it.
    let mut direct = StaticSelectiveDiscovery::new();
    direct.run(&contract(), split.discovery_rows(), &ctx()).unwrap();
    let direct = direct.outcome().unwrap();
    assert_eq!(split.outcome().result_identity, direct.result_identity);
    assert_eq!(split.outcome().structure_identity, direct.structure_identity);
    assert_eq!(split.outcome().structure.unresolved_edges().len(), 2);
    assert!(split.report().work.ci_tests_attempted > 0);
}

#[test]
fn only_the_analysis_part_passes_the_check_and_the_binding_states_what_it_licenses() {
    let data = table(2);
    let split = discover_on_split(&contract(), &data, rule(6), &ctx()).unwrap();
    let binding = split.check_analysis_table(split.analysis_rows()).unwrap();
    assert!(binding.conditional_on_selected_graph());
    assert_eq!(binding.whole_method_calibration(), WHOLE_METHOD_CALIBRATION);
    assert_eq!(WHOLE_METHOD_CALIBRATION, "unmeasured");
    assert_eq!(binding.receipt_identity, split.receipt().identity);
    assert_eq!(binding.structure_identity, split.outcome().structure_identity);

    let detail =
        |table: &TabularData| split.check_analysis_table(table).unwrap_err().detail().unwrap();
    let wrong = "selective_discovery.analysis_rows_not_the_split_part";
    assert_eq!(detail(split.discovery_rows()), wrong);
    assert_eq!(detail(&data), wrong);
    // Same shape, other rows.
    let other = discover_on_split(&contract(), &table(3), rule(6), &ctx()).unwrap();
    assert_eq!(detail(other.analysis_rows()), wrong);
}

#[test]
fn a_changed_rule_or_data_is_a_new_receipt_and_a_new_binding() {
    let data = table(4);
    let base = discover_on_split(&contract(), &data, rule(7), &ctx()).unwrap();
    let again = discover_on_split(&contract(), &data, rule(7), &ctx()).unwrap();
    assert_eq!(base.receipt().identity, again.receipt().identity);
    let binding = |s: &antecedent::analysis::sample_split_analysis::SplitDiscovery| {
        s.check_analysis_table(s.analysis_rows()).unwrap().identity
    };
    assert_eq!(binding(&base), binding(&again));

    // Another seed at an unchanged shape: other rows, so another receipt and binding.
    let reseeded = discover_on_split(&contract(), &data, rule(8), &ctx()).unwrap();
    assert_ne!(base.receipt().identity, reseeded.receipt().identity);
    assert_ne!(binding(&base), binding(&reseeded));
    // Another fraction at the same seed changes the sizes.
    let wider = discover_on_split(
        &contract(),
        &data,
        SplitRule::SeededPermutation { seed: 7, discovery_fraction: 0.6 },
        &ctx(),
    )
    .unwrap();
    assert_eq!(wider.receipt().n_discovery, 300);
    assert_ne!(base.receipt().identity, wider.receipt().identity);
}

#[test]
fn an_explicit_rule_names_the_discovery_rows_and_malformed_rules_are_refused() {
    let data = table(5);
    let split = discover_on_split(
        &contract(),
        &data,
        SplitRule::Explicit { discovery_rows: (0..250).collect() },
        &ctx(),
    )
    .unwrap();
    assert_eq!(split.receipt().n_discovery, 250);
    assert_eq!(row_ids(split.analysis_rows()), (250..ROWS).collect::<Vec<_>>());

    let refused = |rule: SplitRule| {
        discover_on_split(&contract(), &data, rule, &ctx()).unwrap_err().detail().unwrap()
    };
    assert_eq!(
        refused(SplitRule::Explicit { discovery_rows: vec![3, 3] }),
        "selective_discovery.split_rows_repeated"
    );
    assert_eq!(
        refused(SplitRule::Explicit { discovery_rows: vec![ROWS] }),
        "selective_discovery.split_row_out_of_range"
    );
    assert_eq!(
        refused(SplitRule::Explicit { discovery_rows: (0..ROWS).collect() }),
        "selective_discovery.split_part_empty"
    );
    assert_eq!(
        refused(SplitRule::SeededPermutation { seed: 1, discovery_fraction: 1.0 }),
        "selective_discovery.split_fraction_invalid"
    );
}

/// `data` with the observation weights and analysis mask given (`mask[i]` keeps row `i`).
fn with_rows(data: &TabularData, weights: Option<Vec<f64>>, mask: Option<&[bool]>) -> TabularData {
    let storage = data.storage();
    let mask = mask.map(|keep| {
        let mut bytes = vec![0u8; keep.len().div_ceil(8)];
        for (i, kept) in keep.iter().enumerate() {
            if *kept {
                bytes[i / 8] |= 1 << (i % 8);
            }
        }
        ValidityBitmap::from_bytes(bytes, keep.len()).unwrap()
    });
    TabularData::new(
        OwnedColumnarStorage::try_new(
            storage.schema().clone(),
            storage.columns().to_vec(),
            mask,
            weights.map(Arc::from),
        )
        .unwrap(),
    )
}

#[test]
fn masked_rows_are_filtered_before_the_split_and_recorded_in_the_receipt() {
    let data = table(6);
    let mut keep = vec![true; ROWS];
    keep[0] = false;
    keep[17] = false;
    let masked = with_rows(&data, None, Some(&keep));
    let split = discover_on_split(&contract(), &masked, rule(1), &ctx()).unwrap();

    let receipt = split.receipt();
    assert_eq!(receipt.masked_out_rows, vec![0, 17]);
    assert!(receipt.incomplete_rows.is_empty());
    // The rule splits the 498 eligible rows: round(0.4 * 498) = 199.
    assert_eq!((receipt.n_discovery, receipt.n_analysis), (199, 299));
    split.verify().unwrap();

    // The original row index of each part row is the `row` column's value; the two masked-out
    // rows are in neither part.
    assert_eq!(row_ids(split.discovery_rows()), split.discovery_row_ids());
    assert_eq!(row_ids(split.analysis_rows()), split.analysis_row_ids());
    let held: BTreeSet<usize> =
        split.discovery_row_ids().iter().chain(split.analysis_row_ids()).copied().collect();
    assert_eq!(held.len(), ROWS - 2);
    assert!(!held.contains(&0) && !held.contains(&17));

    // The mask is preserved: each part is the all-included mask of rows that passed it, and the
    // check accepts the analysis part only with it.
    assert!(split.analysis_rows().storage().analysis_mask().is_some());
    split.check_analysis_table(split.analysis_rows()).unwrap();
    split.verify_against_source(&masked).unwrap();
    // The unmasked table is not the table that was split.
    assert_eq!(
        split.verify_against_source(&data).unwrap_err().detail(),
        Some("selective_discovery.split_mask_mismatch")
    );
}

#[test]
fn weights_must_be_read_by_the_declared_test() {
    let data = table(8);
    let weighted = with_rows(&data, Some(vec![1.0; ROWS]), None);
    let error = discover_on_split(&contract(), &weighted, rule(2), &ctx()).unwrap_err();
    assert_eq!(error.code(), Some("option_not_applicable"));
    assert_eq!(error.detail(), Some("selective_discovery.weights_not_read_by_test"));

    let mut weighted_test = contract();
    weighted_test.ci_test = CiTestSpec::named("weighted_parcorr");
    let error = discover_on_split(&weighted_test, &data, rule(2), &ctx()).unwrap_err();
    assert_eq!(error.code(), Some("option_not_applicable"));
    assert_eq!(error.detail(), Some("selective_discovery.weighted_test_without_weights"));
}

#[test]
fn weighted_discovery_reads_the_discovery_weights_and_the_analysis_weights_are_exposed() {
    let data = table(9);
    // Unequal weights with explicit zeros.
    let weights: Vec<f64> = (0..ROWS).map(|i| [0.0, 1.0, 2.5, 4.0][i % 4]).collect();
    let weighted = with_rows(&data, Some(weights.clone()), None);
    let mut weighted_contract = contract();
    weighted_contract.ci_test = CiTestSpec::named("weighted_parcorr");
    let split = discover_on_split(&weighted_contract, &weighted, rule(3), &ctx()).unwrap();
    split.verify().unwrap();

    // The analysis weights are the source weights of the analysis rows, unnormalized.
    let ids = split.analysis_row_ids();
    let gathered: Vec<f64> = ids.iter().map(|&r| weights[r]).collect();
    assert_eq!(split.analysis_weights(), Some(gathered.as_slice()));
    let (sum, sum_sq) = gathered.iter().fold((0.0, 0.0), |(s, q), w| (s + w, q + w * w));
    let kish = sum * sum / sum_sq;
    assert!((split.analysis_effective_sample_size().unwrap() - kish).abs() < 1e-9 * kish);
    assert!(kish < gathered.len() as f64, "unequal weights lose information");
    assert!(sum > 1.0, "the gathered weights are not normalized");

    // Discovery ran the weighted test on the discovery part's own weights: it equals a direct
    // run of the same declaration on that part.
    let part_weights = split.discovery_rows().storage().weights().unwrap().to_vec();
    let mut direct_contract = contract();
    direct_contract.ci_test = CiTestSpec::declared(
        format!("weighted_parcorr:weights={}", digest_hex(&column_digest(&part_weights))),
        Arc::new(WeightedPartialCorrelation::new(part_weights)),
    );
    let mut direct = StaticSelectiveDiscovery::new();
    direct.run(&direct_contract, split.discovery_rows(), &ctx()).unwrap();
    assert_eq!(split.outcome().result_identity, direct.outcome().unwrap().result_identity);

    // Re-weighted analysis rows are other rows.
    let other_weights = vec![1.0; split.analysis_rows().row_count()];
    let reweighted = with_rows(split.analysis_rows(), Some(other_weights), None);
    assert_eq!(
        split.check_analysis_table(&reweighted).unwrap_err().detail(),
        Some("selective_discovery.analysis_rows_not_the_split_part")
    );
    split.check_analysis_table(split.analysis_rows()).unwrap();
}

#[test]
fn reusing_the_rows_that_selected_is_refused_and_the_ledger_sees_the_overlap() {
    let data = table(7);
    let split = discover_on_split(&contract(), &data, rule(9), &ctx()).unwrap();
    let error = split.request_unsplit_interval().unwrap_err();
    assert_eq!(error.code(), Some("cell_not_licensed"));
    assert_eq!(error.detail(), Some("selective_discovery.downstream_interval_not_licensed"));

    let snapshot = b"split-snapshot-1";
    let (discovery, analysis) = split_evidence_sources(split.receipt(), snapshot).unwrap();
    let mut ledger = EvidenceLedger::new();
    let discovery = ledger.add_source(discovery).unwrap();
    let analysis = ledger.add_source(analysis).unwrap();
    let unsplit = ledger
        .add_source(discovery_data_source("the unsplit table", snapshot, 0..ROWS as u64))
        .unwrap();
    // The two parts of one split are independent of each other.
    assert_eq!(ledger.overlap(discovery, analysis).unwrap(), Overlap::Independent);
    // Each overlaps the unsplit table on exactly the rows it holds.
    assert_eq!(
        ledger.overlap(discovery, unsplit).unwrap(),
        Overlap::SharedObservations { shared: split.receipt().n_discovery as _ }
    );
    assert_eq!(
        ledger.overlap(analysis, unsplit).unwrap(),
        Overlap::SharedObservations { shared: split.receipt().n_analysis as _ }
    );
}

/// Weighted least squares of `y` on `[1, x]` over rows of positive weight, and the HC1 sandwich
/// standard error of the slope, from the normal equations.
fn weighted_slope(x: &[f64], y: &[f64], weights: &[f64]) -> (f64, f64) {
    let rows: Vec<usize> = (0..x.len()).filter(|&i| weights[i] > 0.0).collect();
    let n = rows.len() as f64;
    let scale = n / rows.iter().map(|&i| weights[i]).sum::<f64>();
    let w = |i: usize| weights[i] * scale;
    let (sw, sx, sxx, sy, sxy) = rows.iter().fold((0.0, 0.0, 0.0, 0.0, 0.0), |a, &i| {
        (
            a.0 + w(i),
            a.1 + w(i) * x[i],
            a.2 + w(i) * x[i] * x[i],
            a.3 + w(i) * y[i],
            a.4 + w(i) * x[i] * y[i],
        )
    });
    let det = sw * sxx - sx * sx;
    let slope = (sw * sxy - sx * sy) / det;
    let intercept = (sxx * sy - sx * sxy) / det;
    // Bread rows (2x2 inverse) and the meat sum of w^2 e^2 [1 x]'[1 x].
    let inv = [[sxx / det, -sx / det], [-sx / det, sw / det]];
    let mut meat = [[0.0_f64; 2]; 2];
    for &i in &rows {
        let e = y[i] - intercept - slope * x[i];
        let v = [1.0, x[i]];
        for a in 0..2 {
            for b in 0..2 {
                meat[a][b] += (w(i) * e).powi(2) * v[a] * v[b];
            }
        }
    }
    let mut variance = 0.0;
    for a in 0..2 {
        for b in 0..2 {
            variance += inv[1][a] * meat[a][b] * inv[b][1];
        }
    }
    (slope, (variance * n / (n - 2.0)).sqrt())
}

#[test]
fn the_analysis_part_is_estimated_under_its_own_weights_with_its_effective_size() {
    let data = table(9);
    let weights: Vec<f64> = (0..ROWS).map(|i| [0.0, 1.0, 2.5, 4.0][i % 4]).collect();
    let weighted = with_rows(&data, Some(weights.clone()), None);
    let mut weighted_contract = contract();
    weighted_contract.ci_test = CiTestSpec::named("weighted_parcorr");
    let split = discover_on_split(&weighted_contract, &weighted, rule(3), &ctx()).unwrap();

    let request = AnalysisEffectRequest {
        treatment: VariableId::from_raw(0),
        outcome: VariableId::from_raw(1),
        adjustment: vec![],
        levels: (0.0, 1.0),
        clusters: None,
    };
    let effect: SplitAnalysisEffect = split.estimate_analysis_effect(&request, &ctx()).unwrap();

    let rows = split.analysis_rows();
    let x = rows.float64_values(VariableId::from_raw(0)).unwrap();
    let y = rows.float64_values(VariableId::from_raw(1)).unwrap();
    let analysis_weights = split.analysis_weights().unwrap();
    let (slope, se) = weighted_slope(&x, &y, analysis_weights);
    assert!((effect.effect - slope).abs() < 1e-9, "{} vs {slope}", effect.effect);
    assert!((effect.standard_error - se).abs() < 1e-9, "{} vs {se}", effect.standard_error);

    // The weights' accounting: zero-weight rows are counted, the Kish size is the part's own.
    let record = effect.sampling_weights.expect("a weighted analysis part reports its record");
    let zeros = analysis_weights.iter().filter(|w| **w == 0.0).count() as u64;
    assert_eq!(record.zero_weight_rows, zeros);
    assert_eq!(record.rows_used + zeros, rows.row_count() as u64);
    let kish = split.analysis_effective_sample_size().unwrap();
    assert!((record.effective_sample_size - kish).abs() < 1e-9 * kish);
    assert_eq!(effect.analysis_row_ids, split.analysis_row_ids());
    assert_eq!(effect.binding, split.check_analysis_table(rows).unwrap());

    // Rescaling the source weights changes no estimate and no effective size.
    let scaled = with_rows(&data, Some(weights.iter().map(|w| w * 250.0).collect()), None);
    let scaled_split = discover_on_split(&weighted_contract, &scaled, rule(3), &ctx()).unwrap();
    let again = scaled_split.estimate_analysis_effect(&request, &ctx()).unwrap();
    assert!((again.effect - effect.effect).abs() < 1e-9);
    assert!((again.standard_error - effect.standard_error).abs() < 1e-9);
    let rescaled = again.sampling_weights.unwrap();
    assert!((rescaled.effective_sample_size - record.effective_sample_size).abs() < 1e-9 * kish);
}

#[test]
fn an_unweighted_analysis_part_reports_no_weight_record() {
    let data = table(4);
    let split = discover_on_split(&contract(), &data, rule(8), &ctx()).unwrap();
    let request = AnalysisEffectRequest {
        treatment: VariableId::from_raw(0),
        outcome: VariableId::from_raw(1),
        adjustment: vec![],
        levels: (0.0, 1.0),
        clusters: None,
    };
    let effect = split.estimate_analysis_effect(&request, &ctx()).unwrap();
    assert!(effect.sampling_weights.is_none());
    let rows = split.analysis_rows();
    let x = rows.float64_values(VariableId::from_raw(0)).unwrap();
    let y = rows.float64_values(VariableId::from_raw(1)).unwrap();
    let (slope, se) = weighted_slope(&x, &y, &vec![1.0; x.len()]);
    assert!((effect.effect - slope).abs() < 1e-9 && (effect.standard_error - se).abs() < 1e-9);
}
