//! Discovery followed by analysis on a row split, through the facade.
//!
//! Oracles: the seeded permutation assigns `round(fraction * n)` rows to discovery, so the part
//! sizes are arithmetic; a `row` column carrying each row's index in the unsplit table witnesses
//! that the parts are disjoint and cover the table; and the discovery on the split must equal,
//! bit for bit, a direct run of the same contract on the discovery part.

use std::collections::BTreeSet;

use antecedent::analysis::sample_split_analysis::{
    SplitRule, WHOLE_METHOD_CALIBRATION, discover_on_split, split_evidence_sources,
};
use antecedent::analysis::selective_discovery::discovery_data_source;
use antecedent_contract::evidence_ledger::{EvidenceLedger, Overlap};
use antecedent_core::{ExecutionContext, VariableId};
use antecedent_data::{TableView, TabularData, ValidityBitmap};
use antecedent_discovery::selective::{
    CiTestSpec, StaticAlgorithm, StaticDiscoveryContract, StaticSelectiveDiscovery,
};

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

#[test]
fn masked_rows_are_not_split() {
    let data = table(6);
    let mut bytes = vec![0xFF_u8; ROWS.div_ceil(8)];
    bytes[0] &= !1; // row 0 is outside the analysis mask
    let masked = data.with_analysis_mask(ValidityBitmap::from_bytes(bytes, ROWS).unwrap()).unwrap();
    let error = discover_on_split(&contract(), &masked, rule(1), &ctx()).unwrap_err();
    assert_eq!(error.code(), Some("route_not_supported"));
    assert_eq!(error.detail(), Some("selective_discovery.split_unsupported_rows"));
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
