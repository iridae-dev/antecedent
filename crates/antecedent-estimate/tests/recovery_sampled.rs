//! 2.3A X10: sampled observation recovery against an enumerated binary
//! missingness SCM, its refusals and its replayable receipt. Interval calibration
//! is unmeasured here: nothing below asserts a coverage claim.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "test counts and indices are small and nonnegative"
)]

mod support {
    pub mod recovery_scm;
}

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use antecedent_core::{EvidenceCatalog, ExecutionContext, RegimeBinding};
use antecedent_estimate::{
    ObservationPattern, ObservationRow, SAMPLED_RECOVERY_CALIBRATION, SampledObservationInput,
    SampledRecoveryConfig, SampledRecoveryDetail, derive_sampled_recovery,
    estimate_sampled_recovery, refuse_component_variance_only, replay_sampled_recovery,
    sampled_recovery_route_frozen,
};
use antecedent_graph::{Admg, DenseNodeId, NodeRef};
use antecedent_identify::{
    RecoveredEffectQuery, RecoveryDerivation, RecoveryDetail, RecoveryLimits,
};
use support::recovery_scm::{MModel, Rng, SNAPSHOT, v};

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

// The SCM. Nodes: X0 = 0, X1 = 1, O = 2, R0 = 3, R1 = 4 (proxies 5, 6).
// O -> X0, O -> X1, X0 -> X1 (treatment X0, outcome X1, confounder O);
// R0 depends on O and R1 on X0. Every probability is a multiple of 0.1, so
// pattern probabilities are multiples of 1e-5.
const EDGES: [(u32, u32); 3] = [(0, 1), (2, 0), (2, 1)];
const P_O: f64 = 0.4;
const P_X0: [f64; 2] = [0.3, 0.7]; // by O
const P_X1: [f64; 4] = [0.2, 0.5, 0.6, 0.8]; // by 2 * X0 + O
const P_R0: [f64; 2] = [0.5, 0.8]; // by O
const P_R1: [f64; 2] = [0.6, 0.9]; // by X0
// Backdoor truth by hand: sum_o P(o) [P(X1 | X0 = 1, o) - P(X1 | X0 = 0, o)]
// = 0.6 (0.6 - 0.2) + 0.4 (0.8 - 0.5) = 0.36.
const TRUE_EFFECT: f64 = 0.36;

fn bern(p1: f64, level: u8) -> f64 {
    if level == 1 { p1 } else { 1.0 - p1 }
}

fn model() -> MModel {
    let mut model = MModel::new(2, 1, &EDGES, &[vec![2], vec![0]], 5).unwrap();
    let parents = |n: u32| -> Vec<u32> {
        model.graph.parents(DenseNodeId::from_raw(n)).iter().map(|p| p.raw()).collect()
    };
    assert_eq!(parents(0), vec![2]);
    assert_eq!(parents(1), vec![0, 2], "X1 mechanism is indexed by 2 * X0 + O");
    assert_eq!(parents(3), vec![2]);
    assert_eq!(parents(4), vec![0]);
    model.mechanisms.insert(2, vec![P_O]);
    model.mechanisms.insert(0, P_X0.to_vec());
    model.mechanisms.insert(1, P_X1.to_vec());
    model.mechanisms.insert(3, P_R0.to_vec());
    model.mechanisms.insert(4, P_R1.to_vec());
    model
}

/// Observed-pattern probabilities, enumerated by hand from the mechanisms.
/// Pattern bit `i` is the `i`-th partially observed variable (X0, X1); `fully`
/// bit 0 is O.
fn pattern_probabilities() -> BTreeMap<ObservationPattern, f64> {
    let mut out = BTreeMap::new();
    for o in 0..2u8 {
        for x0 in 0..2u8 {
            for x1 in 0..2u8 {
                for r0 in 0..2u8 {
                    for r1 in 0..2u8 {
                        let p = bern(P_O, o)
                            * bern(P_X0[usize::from(o)], x0)
                            * bern(P_X1[usize::from(2 * x0 + o)], x1)
                            * bern(P_R0[usize::from(o)], r0)
                            * bern(P_R1[usize::from(x0)], r1);
                        let pattern = ObservationPattern {
                            responses: r0 | (r1 << 1),
                            proxies: (r0 & x0) | ((r1 & x1) << 1),
                            fully: o,
                        };
                        *out.entry(pattern).or_insert(0.0) += p;
                    }
                }
            }
        }
    }
    out
}

/// Rows whose pattern counts are exactly proportional to the probabilities.
fn exact_rows(n: usize) -> Vec<ObservationRow> {
    let mut rows = Vec::new();
    let mut id = 0u64;
    for (pattern, p) in pattern_probabilities() {
        let exact = p * n as f64;
        let count = exact.round();
        assert!((exact - count).abs() < 1e-6, "{pattern:?}: {exact} is not an integer count");
        for _ in 0..(count as usize) {
            rows.push(ObservationRow { id, pattern });
            id += 1;
        }
    }
    rows
}

/// `n` rows drawn from the pattern distribution with a seeded generator.
fn seeded_rows(n: usize, seed: u64) -> Vec<ObservationRow> {
    let table: Vec<(ObservationPattern, f64)> = pattern_probabilities().into_iter().collect();
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|i| {
            let u = (rng.next() >> 11) as f64 / (1u64 << 53) as f64;
            let mut acc = 0.0;
            let mut chosen = table[table.len() - 1].0;
            for (pattern, p) in &table {
                acc += p;
                if u < acc {
                    chosen = *pattern;
                    break;
                }
            }
            ObservationRow { id: i as u64, pattern: chosen }
        })
        .collect()
}

fn input(rows: Vec<ObservationRow>) -> SampledObservationInput {
    SampledObservationInput { snapshot_id: SNAPSHOT.to_string(), rows }
}

fn effect_graph(model: &MModel, edges: &[(u32, u32)]) -> Admg {
    let mut graph = Admg::empty();
    for n in 0..model.k + model.m {
        graph.add_node(NodeRef::Static(v(n))).unwrap();
    }
    for (a, b) in edges {
        graph.insert_directed(DenseNodeId::from_raw(*a), DenseNodeId::from_raw(*b)).unwrap();
    }
    graph
}

fn effect_query(model: &MModel, edges: &[(u32, u32)], t: u32, y: u32) -> RecoveredEffectQuery {
    RecoveredEffectQuery {
        graph: effect_graph(model, edges),
        outcomes: Arc::from([v(y)]),
        treatments: Arc::from([v(t)]),
    }
}

fn derive_with(model: &MModel, catalog: &EvidenceCatalog) -> Box<RecoveryDerivation> {
    derive_sampled_recovery(
        &model.graph,
        &model.query(),
        catalog,
        &effect_query(model, &EDGES, 0, 1),
        RecoveryLimits::default(),
        &ctx(),
    )
    .unwrap()
}

fn derive(model: &MModel) -> Box<RecoveryDerivation> {
    derive_with(model, &model.catalog())
}

fn loose(replicates: usize, seed: u64) -> SampledRecoveryConfig {
    SampledRecoveryConfig { normalization_tolerance: 0.25, ..SampledRecoveryConfig::new(replicates, seed) }
}

#[test]
fn x10_sampled_recovery_truth_exact_proportions_recover_the_enumerated_law_and_effect() {
    let model = model();
    let derivation = derive(&model);
    let n = 100_000;
    let rows = exact_rows(n);
    assert_eq!(rows.len(), n);
    let result = estimate_sampled_recovery(
        &derivation,
        &input(rows),
        &SampledRecoveryConfig::new(20, 3),
        &ctx(),
    )
    .unwrap();
    // The recovered effect is the enumerated backdoor truth.
    assert!((result.effect - TRUE_EFFECT).abs() < 1e-9, "{}", result.effect);
    // The fixture discriminates: the confounded contrast P(X1 | X0 = 1) - P(X1 | X0 = 0)
    // on the complete data is far from the effect.
    let joint_x1 = |x0: usize| -> f64 {
        (0..2u8)
            .map(|o| {
                let px0 = if x0 == 1 { P_X0[usize::from(o)] } else { 1.0 - P_X0[usize::from(o)] };
                bern(P_O, o) * px0 * P_X1[2 * x0 + usize::from(o)]
            })
            .sum()
    };
    let margin = |x0: usize| -> f64 {
        (0..2u8)
            .map(|o| {
                let px0 = if x0 == 1 { P_X0[usize::from(o)] } else { 1.0 - P_X0[usize::from(o)] };
                bern(P_O, o) * px0
            })
            .sum()
    };
    let confounded = joint_x1(1) / margin(1) - joint_x1(0) / margin(0);
    assert!((confounded - TRUE_EFFECT).abs() > 0.05, "{confounded}");
    // The whole recovered law equals P(x0, x1, o) = P(o) P(x0 | o) P(x1 | x0, o),
    // axes (X0, X1, O), last fastest; it normalizes and is strictly positive.
    let law = result.recovered_law.law().probabilities();
    assert_eq!(law.len(), 8);
    for x0 in 0..2u8 {
        for x1 in 0..2u8 {
            for o in 0..2u8 {
                let truth = bern(P_O, o)
                    * bern(P_X0[usize::from(o)], x0)
                    * bern(P_X1[usize::from(2 * x0 + o)], x1);
                let index = usize::from((x0 * 2 + x1) * 2 + o);
                assert!((law[index] - truth).abs() < 1e-9, "cell {index}: {} vs {truth}", law[index]);
                assert!(law[index] > 0.0);
            }
        }
    }
    assert!((law.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    // Support diagnostics.
    let d = &result.diagnostics;
    assert!(d.normalization_defect < 1e-9, "{}", d.normalization_defect);
    assert_eq!(d.rows, n);
    assert_eq!(d.pattern_counts.iter().map(|(_, c)| *c).sum::<u64>(), n as u64);
    assert!(d.min_complete_case_count > 0);
    assert!(d.min_recovered_cell > 0.0 && d.min_recovered_cell_count > 0.0);
    // The interval is formed, ordered, and makes no coverage claim.
    assert!(result.interval.lower <= result.interval.upper);
    assert!((result.interval.level - 0.95).abs() < 1e-12);
    assert_eq!(result.interval.calibration, SAMPLED_RECOVERY_CALIBRATION);
    assert_eq!(SAMPLED_RECOVERY_CALIBRATION, "unmeasured");
    assert_eq!(result.failed_replicates, 0);
    assert_eq!(result.replicate_effects.len(), 20);
}

#[test]
fn x10_sampled_recovery_truth_seeded_interval_is_centered_reproducible_and_keeps_covariance() {
    let model = model();
    let derivation = derive(&model);
    let data = input(seeded_rows(4000, 17));
    let config = loose(200, 99);
    let first = estimate_sampled_recovery(&derivation, &data, &config, &ctx()).unwrap();
    let second = estimate_sampled_recovery(&derivation, &data, &config, &ctx()).unwrap();
    // Reproducible replicate ids, selection digests and receipt.
    assert_eq!(first.receipt, second.receipt);
    assert_eq!(first.receipt.receipt_digest, second.receipt.receipt_digest);
    assert!(first.receipt.replicates.iter().map(|r| r.id).eq(0..200u32));
    let digests: BTreeSet<&str> =
        first.receipt.replicates.iter().map(|r| r.selection_digest.as_str()).collect();
    assert_eq!(digests.len(), 200, "every replicate draws different rows");
    assert!(first.receipt.replicates.iter().all(|r| r.selection_digest.len() == 32));
    // The point is near the truth and the interval center near the point.
    assert!((first.effect - TRUE_EFFECT).abs() < 0.15, "{}", first.effect);
    let center = 0.5 * (first.interval.lower + first.interval.upper);
    assert!((center - first.effect).abs() < 0.03, "center {center} vs point {}", first.effect);
    assert!(first.interval.lower < first.interval.upper);
    assert!(first.effect_standard_error > 0.0);
    assert_eq!(first.replicate_effects.len() + first.failed_replicates, 200);
    // Covariance of the recovered margins: full symmetric matrix, nonnegative
    // diagonal and a nonzero off-diagonal (margins share rows).
    assert_eq!(first.recovered_cells, 8);
    assert_eq!(first.recovered_cell_covariance.len(), 64);
    let mut off_diagonal = 0.0f64;
    for i in 0..8 {
        assert!(first.cell_covariance(i, i).unwrap() >= 0.0);
        for j in 0..8 {
            let (a, b) = (first.cell_covariance(i, j).unwrap(), first.cell_covariance(j, i).unwrap());
            assert!((a - b).abs() < 1e-15);
            if i != j {
                off_diagonal = off_diagonal.max(a.abs());
            }
        }
    }
    assert!(off_diagonal > 0.0);
    assert!(first.cell_covariance(8, 0).is_none());
    // Diagnostics bind the counted rows.
    assert_eq!(first.diagnostics.rows, 4000);
    assert_eq!(first.diagnostics.pattern_counts.iter().map(|(_, c)| *c).sum::<u64>(), 4000);
    assert!(first.diagnostics.min_complete_case_count > 0);
    // Another seed draws other rows.
    let other = estimate_sampled_recovery(&derivation, &data, &loose(200, 100), &ctx()).unwrap();
    assert_ne!(other.receipt.replicates[0].selection_digest, first.receipt.replicates[0].selection_digest);
}

#[test]
fn x10_unrecoverable_pattern_zero_complete_case_cell_refuses_with_the_pattern() {
    let model = model();
    let derivation = derive(&model);
    // No row of the complete-case pattern X0 = 1, X1 = 0, O = 0.
    let missing = ObservationPattern { responses: 3, proxies: 1, fully: 0 };
    let mut rows = seeded_rows(4000, 17);
    let before = rows.len();
    rows.retain(|r| r.pattern != missing);
    assert!(rows.len() < before);
    let error = estimate_sampled_recovery(&derivation, &input(rows), &loose(30, 1), &ctx())
        .unwrap_err();
    assert_eq!(error.detail, SampledRecoveryDetail::UnrecoverablePattern);
    assert_eq!(error.detail.detail(), "sampled_recovery.unrecoverable_pattern");
    assert_eq!(error.reason_code(), "route_not_supported");
    assert_eq!(error.patterns, vec![missing]);
}

#[test]
fn x10_unrecoverable_pattern_zero_denominator_refuses_with_every_offending_pattern() {
    let model = model();
    let derivation = derive(&model);
    // No unit has R0 = 1 and X*0 = 1: the margin P(R0 = 1, X*0 = 1) that
    // normalizes p(R1 = 1 | X0 = 1) is zero, and with it all four complete-case
    // cells at X0 = 1.
    let mut rows = seeded_rows(4000, 17);
    rows.retain(|r| !(r.pattern.responses & 1 == 1 && r.pattern.proxies & 1 == 1));
    let error = estimate_sampled_recovery(&derivation, &input(rows), &loose(30, 1), &ctx())
        .unwrap_err();
    assert_eq!(error.detail, SampledRecoveryDetail::UnrecoverablePattern);
    assert_eq!(error.reason_code(), "route_not_supported");
    assert_eq!(
        error.patterns,
        vec![
            ObservationPattern { responses: 3, proxies: 1, fully: 0 },
            ObservationPattern { responses: 3, proxies: 1, fully: 1 },
            ObservationPattern { responses: 3, proxies: 3, fully: 0 },
            ObservationPattern { responses: 3, proxies: 3, fully: 1 },
        ]
    );
    assert!(error.message.contains("zero count"), "{error}");
}

#[test]
fn x10_unrecoverable_pattern_adjacent_and_nonrecoverable_mgraphs_refuse() {
    // Self-censoring: X0 -> R0 with a verified nonrecoverability witness.
    let censoring = MModel::new(1, 1, &[(1, 0)], &[vec![0]], 3).unwrap();
    let error = derive_sampled_recovery(
        &censoring.graph,
        &censoring.query(),
        &censoring.catalog(),
        &effect_query(&censoring, &[(1, 0)], 1, 0),
        RecoveryLimits::default(),
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(error.detail, SampledRecoveryDetail::UnrecoverablePattern);
    assert_eq!(error.reason_code(), "route_not_supported");
    assert_eq!(error.recovery, Some(RecoveryDetail::NonrecoverableWitness));
    assert!(error.patterns.is_empty());
    // Adjacent class: a colluder R0 -> R1 is outside the licensed m-graph.
    let mut colluder = MModel::new(2, 0, &[(0, 1)], &[vec![], vec![0]], 13).unwrap();
    colluder.graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(3)).unwrap();
    let error = derive_sampled_recovery(
        &colluder.graph,
        &colluder.query(),
        &colluder.catalog(),
        &effect_query(&colluder, &[(0, 1)], 0, 1),
        RecoveryLimits::default(),
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(error.detail, SampledRecoveryDetail::UnrecoverablePattern);
    assert_eq!(error.recovery, Some(RecoveryDetail::UnsupportedMechanism));
    assert_eq!(error.reason_code(), "route_not_supported");
}

#[test]
fn x10_unrecoverable_pattern_failed_replicates_above_the_fraction_refuse() {
    let model = model();
    let derivation = derive(&model);
    // One row of a complete-case pattern: about 37% of bootstrap replicates drop it.
    let rare = ObservationPattern { responses: 3, proxies: 1, fully: 0 };
    let mut rows = seeded_rows(2000, 5);
    let mut seen = false;
    rows.retain(|r| {
        if r.pattern != rare {
            return true;
        }
        let keep = !seen;
        seen = true;
        keep
    });
    assert!(seen);
    let config = SampledRecoveryConfig {
        normalization_tolerance: 0.5,
        ..SampledRecoveryConfig::new(100, 8)
    };
    let error = estimate_sampled_recovery(&derivation, &input(rows), &config, &ctx()).unwrap_err();
    assert_eq!(error.detail, SampledRecoveryDetail::TooManyFailedReplicates);
    assert_eq!(error.detail.detail(), "sampled_recovery.too_many_failed_replicates");
    assert_eq!(error.reason_code(), "route_not_supported");
    assert!(error.patterns.contains(&rare), "{:?}", error.patterns);
}

#[test]
fn x10_unrecoverable_pattern_bounds_and_malformed_inputs_refuse() {
    let model = model();
    let derivation = derive(&model);
    let rows = seeded_rows(500, 2);
    let run = |data: &SampledObservationInput, config: &SampledRecoveryConfig| {
        estimate_sampled_recovery(&derivation, data, config, &ctx()).unwrap_err()
    };
    let good = input(rows.clone());
    let error = run(&good, &SampledRecoveryConfig::new(2001, 1));
    assert_eq!(error.detail, SampledRecoveryDetail::BoundsExceeded);
    assert_eq!(error.reason_code(), "route_not_supported");
    let error = run(&good, &SampledRecoveryConfig::new(5, 1));
    assert_eq!(error.detail, SampledRecoveryDetail::InvalidInput);
    assert_eq!(error.reason_code(), "invalid_argument");
    let mut duplicated = rows.clone();
    duplicated[1].id = duplicated[0].id;
    assert_eq!(run(&input(duplicated), &loose(20, 1)).detail, SampledRecoveryDetail::InvalidInput);
    let other_snapshot = SampledObservationInput { snapshot_id: "snap-other".to_string(), rows: rows.clone() };
    assert_eq!(run(&other_snapshot, &loose(20, 1)).detail, SampledRecoveryDetail::InvalidInput);
    // A missing proxy that carries a value is outside the proxy model.
    let mut impossible = rows;
    impossible[0].pattern = ObservationPattern { responses: 0, proxies: 1, fully: 0 };
    assert_eq!(run(&input(impossible), &loose(20, 1)).detail, SampledRecoveryDetail::InvalidInput);
    assert_eq!(run(&input(Vec::new()), &loose(20, 1)).detail, SampledRecoveryDetail::InvalidInput);
}

#[test]
fn x10_sampled_recovery_replay_digests_recompute_and_change_with_pattern_snapshot_and_seed() {
    let model = model();
    let derivation = derive(&model);
    let rows = seeded_rows(1500, 21);
    let data = input(rows.clone());
    let config = loose(30, 4);
    let base = estimate_sampled_recovery(&derivation, &data, &config, &ctx()).unwrap();
    // The stored digest recomputes bit-identically, and a fresh replay reproduces it.
    assert!(base.receipt.verify_digest());
    assert_eq!(base.receipt.recompute_digest(), base.receipt.receipt_digest);
    let replayed = replay_sampled_recovery(&derivation, &data, &base.receipt, &ctx()).unwrap();
    assert_eq!(replayed.receipt, base.receipt);
    assert_eq!(replayed.effect.to_bits(), base.effect.to_bits());
    // A changed seed changes the selections and the digest, not the input digest.
    let reseeded =
        estimate_sampled_recovery(&derivation, &data, &loose(30, 5), &ctx()).unwrap();
    assert_eq!(reseeded.receipt.input_digest, base.receipt.input_digest);
    assert_ne!(reseeded.receipt.receipt_digest, base.receipt.receipt_digest);
    // A changed pattern changes the input digest and the receipt, and a replay of
    // the old receipt on the new rows refuses.
    let mut edited = rows.clone();
    edited[0].pattern = ObservationPattern { responses: 0, proxies: 0, fully: edited[0].pattern.fully ^ 1 };
    let edited = input(edited);
    let changed = estimate_sampled_recovery(&derivation, &edited, &config, &ctx()).unwrap();
    assert_ne!(changed.receipt.input_digest, base.receipt.input_digest);
    assert_ne!(changed.receipt.receipt_digest, base.receipt.receipt_digest);
    let error = replay_sampled_recovery(&derivation, &edited, &base.receipt, &ctx()).unwrap_err();
    assert_eq!(error.detail, SampledRecoveryDetail::ReceiptMismatch);
    assert_eq!(error.reason_code(), "invalid_argument");
    // A changed snapshot (the catalog binds another one) changes the digests; the
    // old snapshot offered to the new derivation is refused.
    let mut catalog = model.catalog();
    let binding = RegimeBinding {
        snapshot_identity: Arc::from("snap-other"),
        ..catalog.bindings[0].clone()
    };
    catalog.bindings = Arc::from([binding]);
    let other_derivation = derive_with(&model, &catalog);
    let moved = SampledObservationInput { snapshot_id: "snap-other".to_string(), rows: rows.clone() };
    let relocated =
        estimate_sampled_recovery(&other_derivation, &moved, &config, &ctx()).unwrap();
    assert_ne!(relocated.receipt.input_digest, base.receipt.input_digest);
    assert_ne!(relocated.receipt.receipt_digest, base.receipt.receipt_digest);
    assert_ne!(relocated.receipt.derivation_identity, base.receipt.derivation_identity);
    let error = estimate_sampled_recovery(&other_derivation, &data, &config, &ctx()).unwrap_err();
    assert_eq!(error.detail, SampledRecoveryDetail::InvalidInput);
    // A tampered receipt fails its own digest, and a re-digested forgery fails replay.
    let mut tampered = base.receipt.clone();
    tampered.point_effect += 1e-6;
    assert!(!tampered.verify_digest());
    let error = replay_sampled_recovery(&derivation, &data, &tampered, &ctx()).unwrap_err();
    assert_eq!(error.detail, SampledRecoveryDetail::ReceiptMismatch);
    tampered.receipt_digest = tampered.recompute_digest();
    assert!(tampered.verify_digest());
    let error = replay_sampled_recovery(&derivation, &data, &tampered, &ctx()).unwrap_err();
    assert_eq!(error.detail, SampledRecoveryDetail::ReceiptMismatch);
}

#[test]
fn x10_sampled_recovery_replay_component_variance_only_and_route_frozen_refuse() {
    let error = refuse_component_variance_only(&[0.01, 0.02, 0.03]);
    assert_eq!(error.detail, SampledRecoveryDetail::ComponentVarianceOnly);
    assert_eq!(error.detail.detail(), "sampled_recovery.component_variance_only");
    assert_eq!(error.reason_code(), "cell_not_licensed");
    assert!(error.message.contains('3'), "{error}");
    let frozen = sampled_recovery_route_frozen();
    assert_eq!(frozen.detail, SampledRecoveryDetail::RouteFrozen);
    assert_eq!(frozen.detail.detail(), "sampled_recovery.route_frozen");
    assert_eq!(frozen.reason_code(), "cell_not_licensed");
}
