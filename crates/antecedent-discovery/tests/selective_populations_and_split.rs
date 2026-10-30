//! Per-population discovery and the discovery/analysis row split, run over simulated data.
//!
//! Oracles are independent of the library:
//!
//! * **Known graphs.** Each population is generated from a hand-written linear-Gaussian
//!   structural model with strong coefficients, so its skeleton is known: the `north`
//!   population has `0 -> 2 <- 1, 2 -> 3`; the `south` population has the same graph with
//!   the `0 -> 2` arrow removed (column `0` is independent noise there). The edge set the
//!   comparison reports must differ in exactly the `0 - 2` edge.
//! * **Nothing pooled.** A population's result must equal a standalone run of the same
//!   contract on that population's rows, whatever the other populations hold.
//! * **Split.** Row sets are recomputed by hand from an explicit rule; the parts' values are
//!   read back from the original column; forged receipts are refused by name.

use std::sync::Arc;

use antecedent_core::{ExecutionContext, VariableId};
use antecedent_data::{OwnedColumnarStorage, TableView, TabularData, ValidityBitmap};
use antecedent_discovery::selective::populations::{PerPopulationDiscovery, PopulationRows};
use antecedent_discovery::selective::reuse::{SplitRule, split_rows};
use antecedent_discovery::selective::{
    CiTestSpec, StaticAlgorithm, StaticDiscoveryContract, StaticSelectiveDiscovery,
};

const SEED: u64 = 20_240_611;

struct Lcg(u64);

impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 =
            self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        let top = u32::try_from(self.0 >> 40).unwrap();
        (f64::from(top) + 0.5) / f64::from(1u32 << 24)
    }

    fn normal(&mut self) -> f64 {
        let (u, v) = (self.unit(), self.unit());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

/// `0 -> 2 <- 1`, `2 -> 3` with the `0 -> 2` coefficient `a_to_c`.
fn collider_table(n: usize, seed: u64, a_to_c: f64) -> TabularData {
    let mut rng = Lcg(seed);
    let (mut c0, mut c1, mut c2, mut c3) = (vec![], vec![], vec![], vec![]);
    for _ in 0..n {
        let a = rng.normal();
        let b = rng.normal();
        let c = a_to_c * a + 0.9 * b + 0.5 * rng.normal();
        let d = 0.9 * c + 0.5 * rng.normal();
        c0.push(a);
        c1.push(b);
        c2.push(c);
        c3.push(d);
    }
    TabularData::from_f64_columns([
        ("a", c0.as_slice()),
        ("b", c1.as_slice()),
        ("c", c2.as_slice()),
        ("d", c3.as_slice()),
    ])
    .unwrap()
}

fn vars() -> Vec<VariableId> {
    (0..4).map(VariableId::from_raw).collect()
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(SEED)
}

fn contract() -> StaticDiscoveryContract {
    let mut contract = StaticDiscoveryContract::new(
        StaticAlgorithm::Pc,
        vars(),
        CiTestSpec::named("parcorr"),
        SEED,
    );
    contract.constraints.alpha = 0.01;
    contract.fdr = None;
    contract
}

fn edge_pairs(
    report: &antecedent_discovery::selective::populations::PerPopulationReport,
) -> Vec<(u32, u32)> {
    report.edges.iter().map(|e| (e.a.variable, e.b.variable)).collect()
}

// -- per-population discovery ---------------------------------------------------------------

#[test]
fn populations_with_a_known_graph_difference_report_exactly_that_edge() {
    let north = collider_table(1500, 101, 0.9);
    let south = collider_table(1500, 202, 0.0);
    let mut driver = PerPopulationDiscovery::new();
    let rows = [
        PopulationRows { population: "south", data: &south },
        PopulationRows { population: "north", data: &north },
    ];
    let report = driver.run(&contract(), &rows, &ctx()).unwrap().clone();

    // Label order, not request order.
    let labels: Vec<&str> = report.populations.iter().map(|p| p.population.as_str()).collect();
    assert_eq!(labels, vec!["north", "south"]);

    // Union of the two skeletons: 0-2 (north only), 1-2 and 2-3 (both).
    assert_eq!(edge_pairs(&report), vec![(0, 2), (1, 2), (2, 3)]);
    // The recovered skeletons are the true ones: north has 0-2, 1-2, 2-3; south has 1-2, 2-3.
    let skeleton = |label: &str| -> Vec<(u32, u32)> {
        report
            .edges
            .iter()
            .filter(|e| e.marks.iter().any(|(l, m)| l == label && m.is_some()))
            .map(|e| (e.a.variable, e.b.variable))
            .collect()
    };
    assert_eq!(skeleton("north"), vec![(0, 2), (1, 2), (2, 3)]);
    assert_eq!(skeleton("south"), vec![(1, 2), (2, 3)]);
    // Presence differs on exactly the 0-2 edge; `north` has the v-structure 0 -> 2 <- 1 while
    // `south` is the chain 1 - 2 - 3 (one equivalence class, no orientation), so the marks of
    // 1-2 and 2-3 also differ: 0-2 by presence, the other two by orientation only.
    let differing = report.differing_edges();
    let pairs: Vec<(u32, u32)> = differing.iter().map(|e| (e.a.variable, e.b.variable)).collect();
    assert_eq!(pairs, vec![(0, 2), (1, 2), (2, 3)]);
    let absent: Vec<(u32, u32)> = differing
        .iter()
        .filter(|e| e.marks.iter().any(|(_, m)| m.is_none()))
        .map(|e| (e.a.variable, e.b.variable))
        .collect();
    assert_eq!(absent, vec![(0, 2)]);
    assert!(differing[0].marks[0].1.is_some());
    assert_eq!(differing[0].marks[1].1, None);
    assert_eq!(differing[0].marks[0].0, "north");
    assert_eq!(differing[0].marks[1].0, "south");

    // Every population retained its own statistics and its own data identity.
    assert!(report.populations.iter().all(|p| p.retained_statistics > 0));
    assert_ne!(
        report.populations[0].data_identity.digest,
        report.populations[1].data_identity.digest
    );
    assert_ne!(report.populations[0].result_identity, report.populations[1].result_identity);
    assert!(driver.adapter("north").is_some());
    assert!(driver.adapter("south").is_some());
    assert!(driver.adapter("west").is_none());
}

#[test]
fn each_population_equals_a_standalone_run_so_nothing_is_pooled() {
    let north = collider_table(1500, 101, 0.9);
    let south = collider_table(1500, 202, 0.0);
    let mut driver = PerPopulationDiscovery::new();
    let rows = [
        PopulationRows { population: "north", data: &north },
        PopulationRows { population: "south", data: &south },
    ];
    let report = driver.run(&contract(), &rows, &ctx()).unwrap().clone();

    for (label, data) in [("north", &north), ("south", &south)] {
        let mut standalone = contract();
        label.clone_into(&mut standalone.population);
        let mut adapter = StaticSelectiveDiscovery::new();
        adapter.run(&standalone, data, &ctx()).unwrap();
        let alone = adapter.outcome().unwrap();
        let joint = report.populations.iter().find(|p| p.population == label).unwrap();
        assert_eq!(joint.result_identity, alone.result_identity, "{label}");
        assert_eq!(joint.structure_identity, alone.structure_identity, "{label}");
        assert_eq!(joint.retained_statistics, adapter.retained_statistics(), "{label}");
    }
}

#[test]
fn a_repeated_request_is_pure_reuse_with_the_same_set_identity() {
    let north = collider_table(800, 11, 0.9);
    let south = collider_table(800, 12, 0.0);
    let mut driver = PerPopulationDiscovery::new();
    let rows = [
        PopulationRows { population: "north", data: &north },
        PopulationRows { population: "south", data: &south },
    ];
    let first = driver.run(&contract(), &rows, &ctx()).unwrap().identity;
    let second = driver.run(&contract(), &rows, &ctx()).unwrap().clone();
    assert_eq!(second.identity, first);
    for population in &second.populations {
        assert!(population.report.plan.is_pure_reuse(), "{}", population.population);
        assert!(population.report.work.is_pure_reuse(), "{}", population.population);
    }
}

#[test]
fn changed_rows_at_unchanged_shape_move_only_that_population() {
    let north = collider_table(800, 11, 0.9);
    let south = collider_table(800, 12, 0.0);
    let south_other = collider_table(800, 13, 0.0);
    let mut driver = PerPopulationDiscovery::new();
    let before = driver
        .run(
            &contract(),
            &[
                PopulationRows { population: "north", data: &north },
                PopulationRows { population: "south", data: &south },
            ],
            &ctx(),
        )
        .unwrap()
        .clone();
    let after = driver
        .run(
            &contract(),
            &[
                PopulationRows { population: "north", data: &north },
                PopulationRows { population: "south", data: &south_other },
            ],
            &ctx(),
        )
        .unwrap()
        .clone();
    assert_eq!(after.populations[0].result_identity, before.populations[0].result_identity);
    assert_eq!(
        after.populations[0].data_identity.digest,
        before.populations[0].data_identity.digest
    );
    assert_ne!(
        after.populations[1].data_identity.digest,
        before.populations[1].data_identity.digest
    );
    assert_ne!(after.identity, before.identity);
}

#[test]
fn failed_later_population_does_not_publish_earlier_candidate() {
    let north = collider_table(200, 41, 0.9);
    let south = collider_table(200, 42, 0.0);
    let north_changed = collider_table(200, 43, 0.9);
    // A declared CI test supplies an operational failure only for the later
    // population, after both plans have passed and the first candidate ran.
    struct RejectLarge;
    impl antecedent_stats::ConditionalIndependenceTest for RejectLarge {
        fn test_batch(
            &self,
            prepared: &antecedent_stats::PreparedCiTest,
            request: &antecedent_stats::CiBatchRequest<'_>,
            _workspace: &mut antecedent_stats::CiWorkspace,
            _ctx: &ExecutionContext,
        ) -> Result<antecedent_stats::CiBatchResult, antecedent_stats::StatsError> {
            prepared.ensure_compatible(request)?;
            if request.nrows()? > 200 {
                return Err(antecedent_stats::StatsError::Cancelled);
            }
            Ok(antecedent_stats::CiBatchResult {
                results: request
                    .queries
                    .iter()
                    .map(|_| antecedent_stats::CiResult {
                        statistic: 0.0,
                        p_value: 1.0,
                        df: 0.0,
                        ci: None,
                    })
                    .collect(),
            })
        }
    }
    let bad = collider_table(201, 44, 0.0);
    let mut request_contract = contract();
    request_contract.ci_test = CiTestSpec::Declared {
        identity: "reject-large-for-atomicity".into(),
        test: Arc::new(RejectLarge),
    };
    let mut driver = PerPopulationDiscovery::new();
    driver
        .run(
            &request_contract,
            &[
                PopulationRows { population: "north", data: &north },
                PopulationRows { population: "south", data: &south },
            ],
            &ctx(),
        )
        .unwrap();
    let before = driver.adapter("north").unwrap().export_state().unwrap();
    let report = driver.report().unwrap().identity;
    // Both plans must pass: otherwise the failing request never reaches the second execution.
    let mut south_contract = request_contract.clone();
    south_contract.population = "south".into();
    assert!(driver.adapter("south").unwrap().plan(&south_contract, &bad, &ctx()).is_ok());
    let mut north_contract = request_contract.clone();
    north_contract.population = "north".into();
    let mut north_probe = StaticSelectiveDiscovery::new();
    assert!(north_probe.run(&north_contract, &north_changed, &ctx()).is_ok());
    assert!(
        driver
            .run(
                &request_contract,
                &[
                    PopulationRows { population: "north", data: &north_changed },
                    PopulationRows { population: "south", data: &bad }
                ],
                &ctx()
            )
            .is_err()
    );
    let after = driver.adapter("north").unwrap().export_state().unwrap();
    assert_eq!(after.data_identity, before.data_identity);
    assert_eq!(after.receipts_identity, before.receipts_identity);
    assert_eq!(driver.report().unwrap().identity, report);
}

#[test]
fn malformed_and_pooled_requests_refuse_by_name_and_leave_the_report_untouched() {
    let north = collider_table(600, 21, 0.9);
    let south = collider_table(600, 22, 0.0);
    let mut driver = PerPopulationDiscovery::new();
    let good = [
        PopulationRows { population: "north", data: &north },
        PopulationRows { population: "south", data: &south },
    ];
    let committed = driver.run(&contract(), &good, &ctx()).unwrap().identity;

    // One population is the single-population route.
    let one = [PopulationRows { population: "north", data: &north }];
    let err = driver.run(&contract(), &one, &ctx()).unwrap_err();
    assert_eq!(err.detail(), Some("selective_discovery.populations_missing"));
    assert_eq!(err.code(), Some("invalid_argument"));

    // A blank label.
    let blank = [
        PopulationRows { population: "north", data: &north },
        PopulationRows { population: "  ", data: &south },
    ];
    let err = driver.run(&contract(), &blank, &ctx()).unwrap_err();
    assert_eq!(err.detail(), Some("selective_discovery.population_missing"));

    // A repeated label.
    let repeated = [
        PopulationRows { population: "north", data: &north },
        PopulationRows { population: "north", data: &south },
    ];
    let err = driver.run(&contract(), &repeated, &ctx()).unwrap_err();
    assert_eq!(err.detail(), Some("selective_discovery.population_repeated"));

    // The same rows under two labels are one dataset, not two populations.
    let relabelled = [
        PopulationRows { population: "north", data: &north },
        PopulationRows { population: "north_copy", data: &north },
    ];
    let err = driver.run(&contract(), &relabelled, &ctx()).unwrap_err();
    assert_eq!(err.detail(), Some("selective_discovery.pooled_population_data"));

    // A pooled search declaration is not this route.
    let mut pooled = contract();
    pooled.additional_populations.push("west".to_owned());
    let err = driver.run(&pooled, &good, &ctx()).unwrap_err();
    assert_eq!(err.detail(), Some("selective_discovery.pooled_population_search"));
    assert_eq!(err.code(), Some("route_not_supported"));

    // Every refusal left the last committed report in place.
    assert_eq!(driver.report().unwrap().identity, committed);
}

// -- sample split ----------------------------------------------------------------------------

fn column(data: &TabularData, index: u32) -> Vec<f64> {
    data.float64_values(VariableId::from_raw(index)).unwrap()
}

#[test]
fn an_explicit_split_gathers_exactly_the_named_rows_and_verifies() {
    let data = collider_table(10, 5, 0.9);
    let split =
        split_rows(&data, &vars(), SplitRule::Explicit { discovery_rows: vec![7, 0, 3, 9] })
            .unwrap();
    let receipt = &split.receipt;
    assert_eq!((receipt.n_rows, receipt.n_discovery, receipt.n_analysis), (10, 4, 6));
    receipt.verify().unwrap();

    for index in 0..4 {
        let whole = column(&data, index);
        let discovery_expected: Vec<f64> = [0, 3, 7, 9].iter().map(|&r| whole[r]).collect();
        let analysis_expected: Vec<f64> = [1, 2, 4, 5, 6, 8].iter().map(|&r| whole[r]).collect();
        assert_eq!(column(&split.discovery, index), discovery_expected, "column {index}");
        assert_eq!(column(&split.analysis, index), analysis_expected, "column {index}");
    }
    assert_ne!(receipt.discovery_data.digest, receipt.analysis_data.digest);
    assert_ne!(receipt.discovery_rows_identity, receipt.analysis_rows_identity);

    // Re-splitting the same rows by the same rule is bit-identical.
    let again = split_rows(&data, &vars(), receipt.rule.clone()).unwrap();
    assert_eq!(again.receipt, *receipt);
}

#[test]
fn a_seeded_split_has_the_declared_sizes_and_a_seed_changes_the_partition() {
    let data = collider_table(100, 6, 0.9);
    let a = split_rows(
        &data,
        &vars(),
        SplitRule::SeededPermutation { seed: 3, discovery_fraction: 0.3 },
    )
    .unwrap();
    assert_eq!((a.receipt.n_discovery, a.receipt.n_analysis), (30, 70));
    assert_eq!(a.discovery.row_count(), 30);
    assert_eq!(a.analysis.row_count(), 70);
    a.receipt.verify().unwrap();
    // The parts partition the rows: every original row appears in exactly one part,
    // and each part's values are the original rows named by its row ids.
    let mut ids: Vec<usize> =
        a.discovery_row_ids.iter().chain(a.analysis_row_ids.iter()).copied().collect();
    ids.sort_unstable();
    assert_eq!(ids, (0..100).collect::<Vec<_>>());
    for index in 0..4 {
        let whole = column(&data, index);
        let from_ids = |rows: &[usize]| rows.iter().map(|&r| whole[r]).collect::<Vec<f64>>();
        assert_eq!(column(&a.discovery, index), from_ids(&a.discovery_row_ids), "column {index}");
        assert_eq!(column(&a.analysis, index), from_ids(&a.analysis_row_ids), "column {index}");
    }

    let b = split_rows(
        &data,
        &vars(),
        SplitRule::SeededPermutation { seed: 4, discovery_fraction: 0.3 },
    )
    .unwrap();
    assert_ne!(a.receipt.discovery_rows_identity, b.receipt.discovery_rows_identity);
    assert_ne!(a.receipt.identity, b.receipt.identity);
}

#[test]
fn forged_receipts_are_refused_by_name() {
    let data = collider_table(40, 7, 0.9);
    let split = split_rows(
        &data,
        &vars(),
        SplitRule::SeededPermutation { seed: 9, discovery_fraction: 0.5 },
    )
    .unwrap();
    let mismatch = Some("selective_discovery.split_receipt_mismatch");

    let mut sizes = split.receipt.clone();
    sizes.n_discovery += 1;
    assert_eq!(sizes.verify().unwrap_err().detail(), mismatch);

    let mut rows = split.receipt.clone();
    rows.discovery_rows_identity[0] ^= 1;
    assert_eq!(rows.verify().unwrap_err().detail(), mismatch);

    let mut other_rule = split.receipt.clone();
    other_rule.rule = SplitRule::SeededPermutation { seed: 10, discovery_fraction: 0.5 };
    assert_eq!(other_rule.verify().unwrap_err().detail(), mismatch);

    let mut data_identity = split.receipt.clone();
    data_identity.analysis_data.digest[0] ^= 1;
    assert_eq!(data_identity.verify().unwrap_err().detail(), mismatch);

    let mut identity = split.receipt.clone();
    identity.identity[0] ^= 1;
    assert_eq!(identity.verify().unwrap_err().detail(), mismatch);

    // A receipt whose rule no longer assigns both parts is refused as the rule's own defect.
    let mut empty = split.receipt.clone();
    empty.rule = SplitRule::Explicit { discovery_rows: (0..40).collect() };
    assert_eq!(empty.verify().unwrap_err().detail(), Some("selective_discovery.split_part_empty"));
}

#[test]
fn analysis_table_check_accepts_only_the_recorded_analysis_part() {
    let data = collider_table(60, 8, 0.9);
    let split = split_rows(
        &data,
        &vars(),
        SplitRule::SeededPermutation { seed: 1, discovery_fraction: 0.5 },
    )
    .unwrap();
    split.receipt.check_analysis_table(&split.analysis, &vars()).unwrap();

    let not_part = Some("selective_discovery.analysis_rows_not_the_split_part");
    // The discovery rows, the unsplit table and a same-shape table with one value moved.
    assert_eq!(
        split.receipt.check_analysis_table(&split.discovery, &vars()).unwrap_err().detail(),
        not_part
    );
    assert_eq!(split.receipt.check_analysis_table(&data, &vars()).unwrap_err().detail(), not_part);
    let mut moved: Vec<Vec<f64>> = (0..4).map(|i| column(&split.analysis, i)).collect();
    moved[2][0] += 0.25;
    let altered = TabularData::from_f64_columns([
        ("a", moved[0].as_slice()),
        ("b", moved[1].as_slice()),
        ("c", moved[2].as_slice()),
        ("d", moved[3].as_slice()),
    ])
    .unwrap();
    assert_eq!(altered.row_count(), split.analysis.row_count());
    assert_eq!(
        split.receipt.check_analysis_table(&altered, &vars()).unwrap_err().detail(),
        not_part
    );

    // A forged receipt is refused before its table is compared.
    let mut forged = split.receipt.clone();
    forged.n_analysis += 1;
    assert_eq!(
        forged.check_analysis_table(&split.analysis, &vars()).unwrap_err().detail(),
        Some("selective_discovery.split_receipt_mismatch")
    );
}

#[test]
fn split_refuses_rules_that_cannot_assign_both_parts() {
    let data = collider_table(20, 9, 0.9);
    let detail = |rule: SplitRule| split_rows(&data, &vars(), rule).unwrap_err().detail();
    assert_eq!(
        detail(SplitRule::SeededPermutation { seed: 1, discovery_fraction: 0.0 }),
        Some("selective_discovery.split_fraction_invalid")
    );
    assert_eq!(
        detail(SplitRule::Explicit { discovery_rows: vec![2, 2] }),
        Some("selective_discovery.split_rows_repeated")
    );
    assert_eq!(
        detail(SplitRule::Explicit { discovery_rows: vec![20] }),
        Some("selective_discovery.split_row_out_of_range")
    );
    assert_eq!(
        detail(SplitRule::Explicit { discovery_rows: (0..20).collect() }),
        Some("selective_discovery.split_part_empty")
    );
}

// -- masks, incomplete rows and weights --------------------------------------------------------

/// `data` with the given observation weights and analysis mask (`mask[i]` keeps row `i`).
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

/// The table with `value` written over cell `(row, column)`.
fn with_cell(data: &TabularData, column_index: u32, row: usize, value: f64) -> TabularData {
    let mut columns: Vec<Vec<f64>> = (0..4).map(|i| column(data, i)).collect();
    columns[column_index as usize][row] = value;
    TabularData::from_f64_columns([
        ("a", columns[0].as_slice()),
        ("b", columns[1].as_slice()),
        ("c", columns[2].as_slice()),
        ("d", columns[3].as_slice()),
    ])
    .unwrap()
}

#[test]
fn masked_and_incomplete_rows_are_excluded_before_the_split_and_recorded() {
    let base = collider_table(40, 21, 0.9);
    // Row 10 and 11 miss a value in a searched variable; row 5 does too but is also masked out.
    let incomplete =
        with_cell(&with_cell(&with_cell(&base, 1, 10, f64::NAN), 2, 11, f64::NAN), 0, 5, f64::NAN);
    let mut keep = vec![true; 40];
    for row in [0, 5, 6] {
        keep[row] = false;
    }
    let data = with_rows(&incomplete, None, Some(&keep));
    let split = split_rows(
        &data,
        &vars(),
        SplitRule::SeededPermutation { seed: 4, discovery_fraction: 0.4 },
    )
    .unwrap();
    let receipt = &split.receipt;

    // The mask is applied first, so row 5 is a masked-out row, not an incomplete one.
    assert_eq!(receipt.masked_out_rows, vec![0, 5, 6]);
    assert_eq!(receipt.incomplete_rows, vec![10, 11]);
    // 40 - 5 = 35 eligible rows; round(0.4 * 35) = 14.
    assert_eq!((receipt.n_discovery, receipt.n_analysis), (14, 21));
    assert!(receipt.mask_identity.is_some());
    assert!(receipt.weights.is_none());

    // Original row identities: ascending, disjoint, covering exactly the eligible rows.
    let eligible: Vec<usize> = (0..40).filter(|r| ![0, 5, 6, 10, 11].contains(r)).collect();
    let mut union: Vec<usize> =
        split.discovery_row_ids.iter().chain(split.analysis_row_ids.iter()).copied().collect();
    union.sort_unstable();
    assert_eq!(union, eligible);
    assert!(split.discovery_row_ids.windows(2).all(|w| w[0] < w[1]));
    assert!(split.analysis_row_ids.windows(2).all(|w| w[0] < w[1]));
    assert_eq!(
        receipt.assigned_rows().unwrap(),
        (split.discovery_row_ids.clone(), split.analysis_row_ids.clone())
    );

    // Values come from the original rows, read back by hand.
    for index in 0..4 {
        let whole = column(&incomplete, index);
        let expected: Vec<f64> = split.analysis_row_ids.iter().map(|&r| whole[r]).collect();
        assert_eq!(column(&split.analysis, index), expected, "column {index}");
    }
    split.verify(&vars()).unwrap();
    receipt.verify_against_source(&data, &vars()).unwrap();
}

#[test]
fn an_explicit_rule_may_not_name_an_excluded_row() {
    let mut keep = vec![true; 20];
    keep[3] = false;
    let data = with_rows(&collider_table(20, 22, 0.9), None, Some(&keep));
    let refused =
        split_rows(&data, &vars(), SplitRule::Explicit { discovery_rows: vec![3, 4] }).unwrap_err();
    assert_eq!(refused.detail(), Some("selective_discovery.split_row_excluded"));
    let split =
        split_rows(&data, &vars(), SplitRule::Explicit { discovery_rows: vec![4, 5] }).unwrap();
    assert_eq!(split.discovery_row_ids, vec![4, 5]);
    assert_eq!(split.analysis_row_ids.len(), 17);
    assert!(!split.analysis_row_ids.contains(&3));
}

#[test]
fn weights_are_carried_by_gather_unnormalized_with_zero_and_unequal_weights() {
    // Six rows, weights 1, 2, 3, 4, 0, 2. Discovery takes rows 4 and 5, analysis rows 0..4.
    let weights = vec![1.0, 2.0, 3.0, 4.0, 0.0, 2.0];
    let data = with_rows(&collider_table(6, 23, 0.9), Some(weights.clone()), None);
    let split =
        split_rows(&data, &vars(), SplitRule::Explicit { discovery_rows: vec![4, 5] }).unwrap();

    // Bit-for-bit gather, no rescaling: neither part sums to one.
    assert_eq!(split.analysis_weights(), Some(&[1.0, 2.0, 3.0, 4.0][..]));
    assert_eq!(split.discovery_weights(), Some(&[0.0, 2.0][..]));

    // Kish effective sizes by hand: (1+2+3+4)^2 / (1+4+9+16) = 100/30; (0+2)^2 / (0+4) = 1.
    let recorded = split.receipt.weights.as_ref().unwrap();
    assert!((recorded.analysis_effective_sample_size - 100.0 / 30.0).abs() < 1e-12);
    assert!((recorded.discovery_effective_sample_size - 1.0).abs() < 1e-12);
    assert!((split.analysis_effective_sample_size().unwrap() - 100.0 / 30.0).abs() < 1e-12);
    assert_ne!(recorded.source, recorded.analysis);
    assert_ne!(recorded.analysis, recorded.discovery);
    split.verify(&vars()).unwrap();
    split.receipt.verify_against_source(&data, &vars()).unwrap();

    // The receipt of an unweighted split of the same values differs.
    let plain = split_rows(
        &collider_table(6, 23, 0.9),
        &vars(),
        SplitRule::Explicit { discovery_rows: vec![4, 5] },
    )
    .unwrap();
    assert_ne!(plain.receipt.identity, split.receipt.identity);
    assert!(plain.receipt.weights.is_none());
}

#[test]
fn a_part_without_positive_weight_is_refused() {
    let weights = vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0];
    let data = with_rows(&collider_table(6, 24, 0.9), Some(weights), None);
    let error =
        split_rows(&data, &vars(), SplitRule::Explicit { discovery_rows: vec![4, 5] }).unwrap_err();
    assert_eq!(error.detail(), Some("selective_discovery.split_part_weight_invalid"));
}

#[test]
fn replay_and_verify_detect_tampering_with_the_mask_weights_and_row_ids() {
    let n = 30;
    let base = collider_table(n, 25, 0.9);
    let mut keep = vec![true; n];
    keep[2] = false;
    let weights: Vec<f64> = (0..n).map(|i| 1.0 + (i % 5) as f64).collect();
    let data = with_rows(&base, Some(weights.clone()), Some(&keep));
    let rule = SplitRule::SeededPermutation { seed: 6, discovery_fraction: 0.5 };
    let split = split_rows(&data, &vars(), rule).unwrap();
    split.verify(&vars()).unwrap();
    split.receipt.verify_against_source(&data, &vars()).unwrap();

    // -- the source: a different mask, different weights, a changed value --
    let mut other_keep = keep.clone();
    other_keep[7] = false;
    let other_mask = with_rows(&base, Some(weights.clone()), Some(&other_keep));
    assert_eq!(
        split.receipt.verify_against_source(&other_mask, &vars()).unwrap_err().detail(),
        Some("selective_discovery.split_mask_mismatch")
    );
    let no_mask = with_rows(&base, Some(weights.clone()), None);
    assert_eq!(
        split.receipt.verify_against_source(&no_mask, &vars()).unwrap_err().detail(),
        Some("selective_discovery.split_mask_mismatch")
    );
    let mut other_weights = weights.clone();
    other_weights[4] += 0.5;
    let reweighted = with_rows(&base, Some(other_weights), Some(&keep));
    assert_eq!(
        split.receipt.verify_against_source(&reweighted, &vars()).unwrap_err().detail(),
        Some("selective_discovery.split_weights_mismatch")
    );
    let no_weights = with_rows(&base, None, Some(&keep));
    assert_eq!(
        split.receipt.verify_against_source(&no_weights, &vars()).unwrap_err().detail(),
        Some("selective_discovery.split_weights_mismatch")
    );
    let edited = with_rows(&with_cell(&base, 2, 9, 123.0), Some(weights.clone()), Some(&keep));
    assert_eq!(
        split.receipt.verify_against_source(&edited, &vars()).unwrap_err().detail(),
        Some("selective_discovery.split_source_mismatch")
    );

    // -- the parts: row ids and weights --
    let mismatch = Some("selective_discovery.split_rows_mismatch");
    let mut moved = split.clone();
    moved.analysis_row_ids[0] += 1;
    assert_eq!(moved.verify(&vars()).unwrap_err().detail(), mismatch);
    let mut swapped = split.clone();
    swapped.discovery_row_ids.swap(0, 1);
    assert_eq!(swapped.verify(&vars()).unwrap_err().detail(), mismatch);

    let mut tampered = split.clone();
    let rows = tampered.analysis.row_count();
    // The part keeps its all-included mask; only its weights are other than the recorded ones.
    tampered.analysis = with_rows(&split.analysis, Some(vec![1.0; rows]), Some(&vec![true; rows]));
    assert_eq!(
        tampered.verify(&vars()).unwrap_err().detail(),
        Some("selective_discovery.split_weights_mismatch")
    );

    // -- the receipt: exclusions, mask identity, weights --
    let receipt_mismatch = Some("selective_discovery.split_receipt_mismatch");
    let mut unexcluded = split.receipt.clone();
    unexcluded.masked_out_rows.clear();
    assert_eq!(unexcluded.verify().unwrap_err().detail(), receipt_mismatch);
    let mut forged_mask = split.receipt.clone();
    forged_mask.mask_identity = None;
    assert_eq!(forged_mask.verify().unwrap_err().detail(), receipt_mismatch);
    let mut forged_weights = split.receipt.clone();
    forged_weights.weights.as_mut().unwrap().analysis_effective_sample_size += 1.0;
    assert_eq!(forged_weights.verify().unwrap_err().detail(), receipt_mismatch);
    let mut dropped_weights = split.receipt.clone();
    dropped_weights.weights = None;
    assert_eq!(dropped_weights.verify().unwrap_err().detail(), receipt_mismatch);
    let mut source = split.receipt.clone();
    source.source_data.digest[0] ^= 1;
    assert_eq!(source.verify().unwrap_err().detail(), receipt_mismatch);

    // -- an analysis table with the right values but other weights is other rows --
    let same_values = with_rows(&split.analysis, Some(vec![2.0; rows]), None);
    assert_eq!(
        split.receipt.check_analysis_table(&same_values, &vars()).unwrap_err().detail(),
        Some("selective_discovery.analysis_rows_not_the_split_part")
    );
}

/// A weighted, masked, incomplete split with background knowledge, built from exact values
/// (no transcendental functions), pinned so that a change to the new identity encodings is a
/// deliberate act. The encodings of tables without these inputs are the pre-existing ones.
fn pinned_split() -> antecedent_discovery::selective::reuse::SplitRows {
    let a: Vec<f64> = (0..12).map(|i| f64::from(i) * 0.5).collect();
    let b: Vec<f64> = (0..12).map(|i| f64::from((i * 7) % 5)).collect();
    let mut c: Vec<f64> = (0..12).map(|i| f64::from(i * i % 7)).collect();
    c[3] = f64::NAN;
    let base = TabularData::from_f64_columns([
        ("a", a.as_slice()),
        ("b", b.as_slice()),
        ("c", c.as_slice()),
    ])
    .unwrap();
    let mut keep = vec![true; 12];
    keep[0] = false;
    let weights: Vec<f64> = (0..12).map(|i| f64::from(i % 3) + 0.5).collect();
    let data = with_rows(&base, Some(weights), Some(&keep));
    split_rows(
        &data,
        &[VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2)],
        SplitRule::Explicit { discovery_rows: vec![1, 2, 5, 8, 9] },
    )
    .unwrap()
}

#[test]
fn the_weighted_masked_split_and_its_knowledge_have_pinned_identities() {
    use antecedent_discovery::constraints::BackgroundKnowledge;
    use antecedent_discovery::result::LaggedLink;
    use antecedent_discovery::selective::digest_hex;

    let split = pinned_split();
    assert_eq!(split.receipt.masked_out_rows, vec![0]);
    assert_eq!(split.receipt.incomplete_rows, vec![3]);
    split
        .verify(&[VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2)])
        .unwrap();
    assert_eq!(
        digest_hex(&split.receipt.identity),
        "41c49979fad2542bc2c55210ffd340a5c6f954b183a60140d11fca9cb5265423"
    );
    let link = LaggedLink {
        source: VariableId::from_raw(0),
        source_lag: antecedent_core::Lag::CONTEMPORANEOUS,
        target: VariableId::from_raw(1),
        target_lag: antecedent_core::Lag::CONTEMPORANEOUS,
    };
    let knowledge = BackgroundKnowledge {
        required: Arc::from([link]),
        max_parents: Some(2),
        ..BackgroundKnowledge::default()
    };
    assert_eq!(
        digest_hex(&knowledge.identity()),
        "eb4920ff77eead29425d3cbad735f06f690bf219fd84ad8c3afd797e7338c152"
    );
}
