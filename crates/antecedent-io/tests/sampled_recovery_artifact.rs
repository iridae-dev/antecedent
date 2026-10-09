//! 2.3A A6 (X10) `x10_sampled_recovery_replay`: the sampled observation-recovery artifact
//! (version 2: derivation identity, observation rows and pattern counts, recovered law,
//! point, percentile interval with calibration `unmeasured`, replicate receipt) replays
//! independently and refuses a changed pattern, snapshot, seed or interval receipt.
//!
//! The oracle is the enumerated binary missingness SCM of the engine's own tests (the
//! shared fixture module is included by path), never the code under test. Interval
//! calibration is unmeasured; nothing here asserts a coverage claim.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "test counts and indices are small and nonnegative"
)]

#[path = "../../antecedent-estimate/tests/support/recovery_scm.rs"]
mod recovery_scm;

use std::collections::BTreeMap;
use std::sync::Arc;

use antecedent_core::{EvidenceCatalog, ExecutionContext};
use antecedent_estimate::{
    ObservationPattern, ObservationRow, SampledObservationInput, SampledRecoveryConfig,
    SampledRecoveryResult, derive_sampled_recovery, estimate_sampled_recovery,
};
use antecedent_graph::{Admg, DenseNodeId, NodeRef};
use antecedent_identify::{RecoveredEffectQuery, RecoveryDerivation, RecoveryLimits};
use antecedent_io::IoError;
use antecedent_io::sampled_recovery_artifact::{
    SampledRecoveryArtifactInput, SampledRecoveryArtifactWire, SampledRecoveryConsumeLimits,
    SampledRecoveryExpectation,
};
use recovery_scm::{MModel, Rng, SNAPSHOT, v};

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

// The SCM. Nodes: X0 = 0, X1 = 1, O = 2, R0 = 3, R1 = 4 (proxies 5, 6).
// O -> X0, O -> X1, X0 -> X1 (treatment X0, outcome X1, confounder O);
// R0 depends on O and R1 on X0. Every probability is a multiple of 0.1.
const EDGES: [(u32, u32); 3] = [(0, 1), (2, 0), (2, 1)];
const P_O: f64 = 0.4;
const P_X0: [f64; 2] = [0.3, 0.7]; // by O
const P_X1: [f64; 4] = [0.2, 0.5, 0.6, 0.8]; // by 2 * X0 + O
const P_R0: [f64; 2] = [0.5, 0.8]; // by O
const P_R1: [f64; 2] = [0.6, 0.9]; // by X0
// Backdoor truth by hand: 0.6 (0.6 - 0.2) + 0.4 (0.8 - 0.5) = 0.36.
const TRUE_EFFECT: f64 = 0.36;

fn bern(p1: f64, level: u8) -> f64 {
    if level == 1 { p1 } else { 1.0 - p1 }
}

fn model() -> MModel {
    let mut model = MModel::new(2, 1, &EDGES, &[vec![2], vec![0]], 5).unwrap();
    model.mechanisms.insert(2, vec![P_O]);
    model.mechanisms.insert(0, P_X0.to_vec());
    model.mechanisms.insert(1, P_X1.to_vec());
    model.mechanisms.insert(3, P_R0.to_vec());
    model.mechanisms.insert(4, P_R1.to_vec());
    model
}

/// Observed-pattern probabilities, enumerated by hand from the mechanisms.
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

fn effect_query(model: &MModel) -> RecoveredEffectQuery {
    let mut graph = Admg::empty();
    for n in 0..model.k + model.m {
        graph.add_node(NodeRef::Static(v(n))).unwrap();
    }
    for (a, b) in EDGES {
        graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    RecoveredEffectQuery { graph, outcomes: Arc::from([v(1)]), treatments: Arc::from([v(0)]) }
}

fn loose(replicates: usize, seed: u64) -> SampledRecoveryConfig {
    SampledRecoveryConfig {
        normalization_tolerance: 0.25,
        ..SampledRecoveryConfig::new(replicates, seed)
    }
}

struct Fixture {
    model: MModel,
    catalog: EvidenceCatalog,
    effect: RecoveredEffectQuery,
    derivation: Box<RecoveryDerivation>,
    data: SampledObservationInput,
    result: SampledRecoveryResult,
}

fn fixture(rows: Vec<ObservationRow>, config: &SampledRecoveryConfig) -> Fixture {
    let model = model();
    let catalog = model.catalog();
    let effect = effect_query(&model);
    let derivation = derive_sampled_recovery(
        &model.graph,
        &model.query(),
        &catalog,
        &effect,
        RecoveryLimits::default(),
        &ctx(),
    )
    .unwrap();
    let data = SampledObservationInput { snapshot_id: SNAPSHOT.to_string(), rows };
    let result = estimate_sampled_recovery(&derivation, &data, config, &ctx()).unwrap();
    Fixture { model, catalog, effect, derivation, data, result }
}

fn wire_of(f: &Fixture) -> SampledRecoveryArtifactWire {
    SampledRecoveryArtifactWire::checked(&SampledRecoveryArtifactInput {
        graph: &f.model.graph,
        effect: &f.effect,
        derivation: &f.derivation,
        catalog: &f.catalog,
        input: &f.data,
        result: &f.result,
        variable_names: &[],
    })
    .expect("the fixture exports")
}

fn consume(
    bytes: &[u8],
) -> Result<antecedent_io::sampled_recovery_artifact::ConsumedSampledRecovery, IoError> {
    SampledRecoveryArtifactWire::consume_with_limits(
        bytes,
        SampledRecoveryConsumeLimits::default(),
        &ctx(),
    )
}

fn resealed(
    wire: &SampledRecoveryArtifactWire,
    mutate: impl FnOnce(&mut SampledRecoveryArtifactWire),
) -> Vec<u8> {
    let mut copy = wire.clone();
    mutate(&mut copy);
    copy.premises_digest = copy.expected_premises_digest().expect("premises digest");
    copy.data_digest = copy.expected_data_digest().expect("data digest");
    copy.export().expect("export")
}

fn refusal(bytes: &[u8]) -> String {
    consume(bytes).expect_err("the consumer must refuse").to_string()
}

#[test]
fn x10_sampled_recovery_replay_exact_proportions_replay_against_the_enumerated_oracle() {
    let f = fixture(exact_rows(100_000), &SampledRecoveryConfig::new(20, 3));
    let wire = wire_of(&f);
    assert_eq!(wire.version, 2);
    let bytes = wire.export().expect("export");
    let consumed = consume(&bytes).expect("a faithful artifact replays");
    // The rerun is the original value for value: point, interval, law, receipt.
    assert_eq!(consumed.result.effect.to_bits(), f.result.effect.to_bits());
    assert!((consumed.result.effect - TRUE_EFFECT).abs() < 1e-9, "{}", consumed.result.effect);
    assert_eq!(consumed.result.receipt, f.result.receipt);
    assert_eq!(consumed.result.interval, f.result.interval);
    assert_eq!(consumed.result.diagnostics, f.result.diagnostics);
    assert_eq!(consumed.result.replicate_effects, f.result.replicate_effects);
    assert_eq!(consumed.result.recovered_cell_covariance, f.result.recovered_cell_covariance);
    let law = consumed.result.recovered_law.law().probabilities();
    assert_eq!(law, f.result.recovered_law.law().probabilities());
    // The recovered law is the enumerated P(x0, x1, o) = P(o) P(x0 | o) P(x1 | x0, o).
    for x0 in 0..2u8 {
        for x1 in 0..2u8 {
            for o in 0..2u8 {
                let truth = bern(P_O, o)
                    * bern(P_X0[usize::from(o)], x0)
                    * bern(P_X1[usize::from(2 * x0 + o)], x1);
                let index = usize::from((x0 * 2 + x1) * 2 + o);
                assert!((law[index] - truth).abs() < 1e-9, "cell {index}");
            }
        }
    }
    // The stored summary: pattern counts, calibration and the unmeasured interval.
    assert_eq!(wire.calibration, "unmeasured");
    assert_eq!(wire.result.interval.calibration, "unmeasured");
    assert_eq!(wire.receipt.interval.calibration, "unmeasured");
    let counted: u64 = wire.result.diagnostics.pattern_counts.iter().map(|c| c.3).sum();
    assert_eq!(counted, 100_000);
    assert_eq!(wire.receipt.config.seed, 3);
    assert_eq!(wire.receipt.replicates.len(), 20);
    assert_ne!(wire.premises_digest, wire.data_digest);
}

#[test]
fn x10_sampled_recovery_replay_a_seeded_interval_replays_and_keeps_the_covariance() {
    let f = fixture(seeded_rows(4000, 17), &loose(40, 99));
    let wire = wire_of(&f);
    let consumed = consume(&wire.export().expect("export")).expect("replays");
    assert_eq!(consumed.result.receipt.receipt_digest, f.result.receipt.receipt_digest);
    assert_eq!(consumed.result.interval, f.result.interval);
    assert!(consumed.result.interval.lower <= consumed.result.effect);
    assert!(consumed.result.effect <= consumed.result.interval.upper);
    assert_eq!(
        consumed.result.recovered_cell_covariance, f.result.recovered_cell_covariance,
        "the covariance of overlapping margins is part of the replayed result"
    );
    assert!(consumed.result.cell_covariance(0, 0).is_some_and(|c| c > 0.0));
    assert_eq!(consumed.wire.receipt.replicates.len(), 40);
    wire.check_variable_names(&[]).expect("empty name mapping");
    assert!(wire.check_variable_names(&["x".to_string()]).is_err());
}

type Mutation = (&'static str, fn(&mut SampledRecoveryArtifactWire));

#[test]
fn x10_sampled_recovery_replay_refuses_a_changed_pattern_snapshot_seed_or_interval_receipt() {
    let f = fixture(seeded_rows(4000, 17), &loose(40, 99));
    let wire = wire_of(&f);
    let mutations: [Mutation; 7] = [
        ("changed pattern", |w| w.rows[0].3 ^= 1),
        ("changed row id", |w| w.rows[0].0 = 9_999_999),
        ("swapped row order", |w| w.rows.swap(0, 1)),
        ("seed", |w| w.receipt.config.seed += 1),
        ("interval receipt", |w| w.receipt.interval.upper += 0.01),
        ("replicate selection", |w| w.receipt.replicates[0].selection_digest = "00".into()),
        ("stored point", |w| w.result.effect += 1e-6),
    ];
    for (name, mutate) in mutations {
        let message = refusal(&resealed(&wire, mutate));
        assert!(message.contains("sampled_recovery.receipt_mismatch"), "{name}: {message}");
    }
    // A changed snapshot is not the snapshot the catalog binds.
    let message = refusal(&resealed(&wire, |w| w.snapshot_id = "snap-other".into()));
    assert!(message.contains("sampled_recovery.invalid_input"), "{message}");
    // A pattern the proxy model excludes (a proxy value with no response).
    let message = refusal(&resealed(&wire, |w| w.rows[0] = (0, 0, 1, 0)));
    assert!(message.contains("sampled_recovery.invalid_input"), "{message}");
    // A tampered derivation does not verify.
    assert!(
        consume(&resealed(&wire, |w| w.derivation.rule_version = "x10.recovery.v0".into()))
            .is_err()
    );
    // Without resealing, edited rows break the data identity and an edited seed breaks
    // the premises digest.
    let mut edited = wire.clone();
    edited.rows[5].3 ^= 1;
    assert!(refusal(&edited.export().expect("export")).contains("data identity digest"));
    let mut edited = wire.clone();
    edited.receipt.config.seed += 1;
    assert!(refusal(&edited.export().expect("export")).contains("premises digest"));
    // A tampered stored covariance does not replay.
    let mut edited = wire;
    edited.result.recovered_cell_covariance[0] += 1e-9;
    assert!(
        refusal(&edited.export().expect("export")).contains("sampled_recovery.receipt_mismatch")
    );
}

#[test]
fn x10_sampled_recovery_replay_enforces_an_expected_identity() {
    let f = fixture(seeded_rows(4000, 17), &loose(40, 99));
    let wire = wire_of(&f);
    let expected = SampledRecoveryExpectation {
        premises_digest: Some(wire.premises_digest.clone()),
        data_digest: Some(wire.data_digest.clone()),
    };
    let accepted = SampledRecoveryArtifactWire::consume_expecting(
        &wire.export().expect("export"),
        &expected,
        SampledRecoveryConsumeLimits::default(),
        &ctx(),
    );
    assert!(accepted.is_ok());
    // A different seed is a different premises identity (it replays on its own terms).
    let other = fixture(seeded_rows(4000, 17), &loose(40, 100));
    let message = SampledRecoveryArtifactWire::consume_expecting(
        &wire_of(&other).export().expect("export"),
        &expected,
        SampledRecoveryConsumeLimits::default(),
        &ctx(),
    )
    .expect_err("a different premises identity is refused")
    .to_string();
    assert!(message.contains("differs from the consumer's expectation"), "{message}");
}

#[test]
fn x10_sampled_recovery_replay_refuses_an_unknown_version_and_an_interval_claim() {
    let f = fixture(seeded_rows(4000, 17), &loose(40, 99));
    let wire = wire_of(&f);
    // Version 1 is the exact-law point artifact; version 4 is unknown.
    for version in [1, 4] {
        let mut other = wire.clone();
        other.version = version;
        assert!(matches!(
            SampledRecoveryArtifactWire::decode(&other.export().expect("export")),
            Err(IoError::UnsupportedVersion { version: found }) if found == version
        ));
    }
    let mut foreign = wire.clone();
    foreign.required_features = vec!["something_else".into()];
    assert!(SampledRecoveryArtifactWire::decode(&foreign.export().expect("export")).is_err());
    let mut calibrated = wire.clone();
    calibrated.result.interval.calibration = "calibrated".into();
    assert!(SampledRecoveryArtifactWire::decode(&calibrated.export().expect("export")).is_err());
    let mut claim = wire;
    claim.calibration = "calibrated".into();
    assert!(SampledRecoveryArtifactWire::decode(&claim.export().expect("export")).is_err());
}

#[test]
fn x10_sampled_recovery_replay_refuses_stored_sizes_above_the_consumer_limits() {
    let f = fixture(seeded_rows(4000, 17), &loose(40, 99));
    let bytes = wire_of(&f).export().expect("export");
    let tight = [
        SampledRecoveryConsumeLimits { max_rows: 10, ..SampledRecoveryConsumeLimits::default() },
        SampledRecoveryConsumeLimits {
            max_replicates: 5,
            ..SampledRecoveryConsumeLimits::default()
        },
    ];
    for limits in tight {
        let message = SampledRecoveryArtifactWire::consume_with_limits(&bytes, limits, &ctx())
            .expect_err("a stored size above the consumer limit is refused")
            .to_string();
        assert!(message.contains("consumer limit exceeded"), "{message}");
    }
}

#[test]
fn sampled_bca_v3_replays_and_binds_jackknife_method_and_consumer_work() {
    let f = fixture(seeded_rows(2000, 8123), &SampledRecoveryConfig::bca(7142));
    let wire = wire_of(&f);
    assert_eq!(wire.version, 3);
    let bca = wire.receipt.bca.as_ref().unwrap();
    assert_eq!(bca.convention, "midrank_exact_ties:type7:delete_one_row");
    let bytes = wire.export().unwrap();
    let consumed = consume(&bytes).unwrap();
    assert_eq!(consumed.wire.export().unwrap(), wire.export().unwrap());
    for corrupt in [
        resealed(&wire, |w| w.receipt.bca.as_mut().unwrap().acceleration += 0.01),
        resealed(&wire, |w| w.receipt.bca.as_mut().unwrap().jackknife[0].effect += 0.01),
        resealed(&wire, |w| w.receipt.bca.as_mut().unwrap().jackknife[0].multiplicity += 1),
        resealed(&wire, |w| w.receipt.replicates[0].failure = Some("forged_failure".into())),
    ] {
        assert!(consume(&corrupt).is_err());
    }
    let legacy_disguise = resealed(&wire, |w| w.version = 2);
    assert!(SampledRecoveryArtifactWire::decode(&legacy_disguise).is_err());
    let wrong_convention =
        resealed(&wire, |w| w.receipt.bca.as_mut().unwrap().convention = "strict_less".into());
    assert!(SampledRecoveryArtifactWire::decode(&wrong_convention).is_err());
    assert!(
        SampledRecoveryArtifactWire::consume_with_limits(
            &bytes,
            SampledRecoveryConsumeLimits { max_replicates: 500, ..Default::default() },
            &ctx()
        )
        .is_err()
    );
    let cancelled = ctx();
    cancelled.cancellation.cancel();
    assert!(
        SampledRecoveryArtifactWire::consume_with_limits(
            &bytes,
            SampledRecoveryConsumeLimits::default(),
            &cancelled
        )
        .is_err()
    );
}

#[test]
fn legacy_percentile_v2_remains_method_distinct_and_replayable() {
    let f = fixture(seeded_rows(2000, 8123), &SampledRecoveryConfig::new(20, 7142));
    let wire = wire_of(&f);
    assert_eq!(wire.version, 2);
    assert!(wire.receipt.config.interval_method.is_none());
    assert!(wire.receipt.bca.is_none());
    assert_eq!(
        consume(&wire.export().unwrap()).unwrap().wire.export().unwrap(),
        wire.export().unwrap()
    );
    // Legacy digests omitted failure strings; complete replay still checks them.
    let forged =
        resealed(&wire, |w| w.receipt.replicates[0].failure = Some("invented_failure".into()));
    assert!(consume(&forged).is_err());
    let bca_disguise = resealed(&wire, |w| {
        w.version = 3;
        w.required_features = vec!["sampled_observation_recovery_bca_v3".into()];
        w.receipt.config.interval_method = Some("bootstrap_bca".into());
    });
    assert!(SampledRecoveryArtifactWire::decode(&bca_disguise).is_err());
}
