//! 2.2B X8: the counterfactual-identification facade end to end: prepare once,
//! evaluate and refresh on laws (exact and counted), export and consume the
//! `counterfactual_id_admg_v1` artifact (replay plus an independent direct-sum
//! recomputation), typed and re-sealed artifact mutations, consumer limits, the
//! refusal boundary, and the Markovian regression against the 2.2A edge
//! contrast.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::type_complexity,
    clippy::too_many_lines,
    clippy::needless_range_loop,
    clippy::result_large_err,
    reason = "test fixtures: small exact indices, levels and enumerated probabilities; exact bit comparisons are the assertion"
)]

#[path = "../../antecedent-identify/tests/support/latent_scm.rs"]
mod latent_scm;

use antecedent::AcceptedGraph;
use antecedent::counterfactual_id::{
    CounterfactualIdConsumeLimits, CounterfactualIdEffect, CounterfactualIdOptions,
    PreparedCounterfactualId, consume_counterfactual_id_artifact,
    consume_counterfactual_id_artifact_for_data, prepare_counterfactual_id,
};
use antecedent::cross_world::{CrossWorldOptions, evaluate_cross_world_effect};
use antecedent_core::{
    CounterfactualEvent, CounterfactualEventQuery, CrossWorldQuery, EdgeRoute, ExecutionContext,
    ExogenousCoupling, MemoryBudget, RegimeId, SearchLimits, Value, VariableId, WorldId, WorldSpec,
};
use antecedent_data::TabularData;
use antecedent_expr::{DiscreteAxis, ExactDiscreteLaw, LawTolerance};
use antecedent_graph::{Admg, Cpdag, Dag, DenseNodeId, Pag, TemporalDag};
use antecedent_io::counterfactual_id_artifact::{
    COUNTERFACTUAL_ID_ESTIMAND, CounterfactualIdArtifactError, CounterfactualIdArtifactWire,
};
use antecedent_io::cross_world_artifact::CrossWorldArtifactWire;
use latent_scm::{LatentScm, Rng};

fn v(i: u32) -> VariableId {
    VariableId::from_raw(i)
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(3)
}

fn admg(n: u32, directed: &[(u32, u32)], bidirected: &[(u32, u32)]) -> Admg {
    let mut g = Admg::with_variables(n);
    for &(a, b) in directed {
        g.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    for &(a, b) in bidirected {
        g.insert_bidirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    g
}

fn names(n: usize) -> Vec<String> {
    ["x", "m", "y", "w", "z", "u"][..n].iter().map(|s| (*s).to_string()).collect()
}

fn levels(cards: &[usize]) -> Vec<Vec<f64>> {
    cards.iter().map(|&c| (0..c).map(|l| l as f64).collect()).collect()
}

fn axes(cards: &[usize]) -> Vec<DiscreteAxis> {
    cards
        .iter()
        .enumerate()
        .map(|(i, &c)| DiscreteAxis {
            variable: v(u32::try_from(i).unwrap()),
            values: (0..c).map(|l| Value::f64(l as f64)).collect::<Vec<_>>().into(),
        })
        .collect()
}

fn exact(cards: &[usize], probabilities: Vec<f64>, snapshot: &str) -> ExactDiscreteLaw {
    ExactDiscreteLaw::try_new(
        "target",
        RegimeId::from_raw(0),
        [],
        axes(cards),
        probabilities,
        snapshot,
        LawTolerance::default(),
    )
    .unwrap()
}

fn counted(cards: &[usize], counts: Vec<u64>) -> ExactDiscreteLaw {
    let total: u64 = counts.iter().sum();
    let probabilities: Vec<f64> = counts.iter().map(|&c| c as f64 / total as f64).collect();
    ExactDiscreteLaw::try_empirical(
        "target",
        RegimeId::from_raw(0),
        [],
        axes(cards),
        probabilities,
        "counts",
        LawTolerance::default(),
    )
    .unwrap()
    .with_empirical_counts(counts)
    .unwrap()
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-10 * (1.0 + a.abs().max(b.abs()))
}

const FRONT_DIRECTED: [(u32, u32); 2] = [(0, 1), (1, 2)];
const FRONT_BIDIRECTED: [(u32, u32); 1] = [(0, 2)];

fn frontdoor_scm(seed: u64, cards: &[usize]) -> LatentScm {
    LatentScm::random(&mut Rng::new(seed), cards, &[(0, 1), (1, 2)], &[(0, 2)], 3)
}

fn prepare_frontdoor(cards: &[usize]) -> PreparedCounterfactualId {
    prepare_counterfactual_id(
        admg(3, &FRONT_DIRECTED, &FRONT_BIDIRECTED),
        names(3),
        levels(cards),
        &CounterfactualEventQuery::effect_on_treated(v(0), 1.0, 0.0, v(2), 1.0).unwrap(),
        CounterfactualIdOptions::default(),
        &ctx(),
    )
    .unwrap()
}

fn frontdoor_effect() -> CounterfactualIdEffect {
    let cards = [2, 3, 2];
    let scm = frontdoor_scm(8, &cards);
    prepare_frontdoor(&cards).evaluate(&exact(&cards, scm.observational(), "s8"), &ctx()).unwrap()
}

#[test]
fn prepare_once_and_evaluate_every_law_against_the_enumerated_truth() {
    let cards = [2, 3, 2];
    let prepared = prepare_frontdoor(&cards);
    let derivation = prepared.derivation().clone();
    for seed in [1, 2, 3] {
        let scm = frontdoor_scm(seed, &cards);
        let effect = prepared.evaluate(&exact(&cards, scm.observational(), "s"), &ctx()).unwrap();
        // Refresh keeps the derivation: no re-identification.
        assert_eq!(effect.derivation(), &derivation);
        let truths = scm.ett_numerators(0, 1, 0, 2);
        let total: f64 = truths.iter().sum();
        for ((level, p), truth) in effect.outcome_distribution.iter().zip(&truths) {
            assert!(close(*p, truth / total), "level {level}: {p} vs {}", truth / total);
        }
        assert!(close(effect.probability, truths[1] / total));
        // The contrast: E[Y_x | x'] - E[Y | x'] from the enumerated model.
        let observed = scm.ett_numerators(0, 0, 0, 2);
        let obs_total: f64 = observed.iter().sum();
        let truth_effect = truths[1] / total - observed[1] / obs_total;
        assert!(
            close(effect.effect.unwrap(), truth_effect),
            "{:?} vs {truth_effect}",
            effect.effect
        );
        assert!(!effect.independently_verified);
    }
}

#[test]
fn a_counted_law_gives_the_empirical_plug_in_of_the_same_functional() {
    let cards = [2, 2, 2];
    let counts: Vec<u64> = vec![30, 12, 7, 21, 9, 14, 25, 18];
    let effect =
        prepare_frontdoor(&cards).evaluate(&counted(&cards, counts.clone()), &ctx()).unwrap();
    // sum_m P(m | x = 1) P(y = 1 | m, x' = 0) from the counts.
    let n = |x: usize, m: usize, y: usize| counts[x * 4 + m * 2 + y] as f64;
    let nx = |x: usize| {
        (0..2).flat_map(|m| (0..2).map(move |y| (m, y))).map(|(m, y)| n(x, m, y)).sum::<f64>()
    };
    let nxm = |x: usize, m: usize| n(x, m, 0) + n(x, m, 1);
    let formula: f64 = (0..2).map(|m| nxm(1, m) / nx(1) * n(0, m, 1) / nxm(0, m)).sum();
    assert!(close(effect.probability, formula), "{} vs {formula}", effect.probability);
    // A law of another origin is refused.
    let posterior = ExactDiscreteLaw::try_bayesian_posterior(
        "target",
        RegimeId::from_raw(0),
        [],
        axes(&cards),
        vec![0.125; 8],
        "draw",
        LawTolerance::default(),
    )
    .unwrap();
    let error = prepare_frontdoor(&cards).evaluate(&posterior, &ctx()).unwrap_err().to_string();
    assert!(error.contains("reason=invalid_argument: counterfactual_id.invalid_query"), "{error}");
}

#[test]
fn the_markovian_ett_equals_the_edge_contrast_restricted_to_the_untreated() {
    // M = X + e_M, Y = 2 X - M + e_Y with X, e_M, e_Y independent: an exact
    // enumeration table (multiplicities proportional to the probabilities).
    // One row per exogenous state (8 rows): few enough rows per level that the
    // 2.2A fitter treats the columns as continuous and fits the linear-Gaussian
    // mechanisms its cell licenses (5 rows per level would make them discrete).
    let (px, pm, py) = ([1u32, 1], [1u32, 1], [1u32, 1]);
    let (mut xs, mut ms, mut ys) = (Vec::new(), Vec::new(), Vec::new());
    let mut cells = std::collections::BTreeMap::new();
    for x in 0..2 {
        for em in 0..2 {
            for ey in 0..2 {
                let copies = px[x] * pm[em] * py[ey];
                let (xv, mv) = (x as f64, (x + em) as f64);
                let yv = 2.0 * xv - mv + ey as f64;
                for _ in 0..copies {
                    xs.push(xv);
                    ms.push(mv);
                    ys.push(yv);
                }
                *cells.entry((x, x + em, (yv + 1.0) as usize)).or_insert(0u64) += u64::from(copies);
            }
        }
    }
    // The 2.2A edge contrast with every edge intervened is the total effect per unit.
    let edges = [(0u32, 1u32), (0, 2), (1, 2)];
    let mut dag = Dag::with_variables(3);
    for &(a, b) in &edges {
        dag.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    let data =
        TabularData::from_f64_columns([("x", &xs[..]), ("m", &ms[..]), ("y", &ys[..])]).unwrap();
    let graph_edges: Vec<(VariableId, VariableId)> =
        edges.iter().map(|&(a, b)| (v(a), v(b))).collect();
    let total =
        CrossWorldQuery::path_specific(v(0), v(2), 0.0, 1.0, &graph_edges, &graph_edges).unwrap();
    let a5 = evaluate_cross_world_effect(
        dag.clone(),
        &data,
        &total,
        CrossWorldOptions::default(),
        &ctx(),
    )
    .unwrap();
    let untreated: Vec<usize> = (0..xs.len()).filter(|&i| xs[i] == 0.0).collect();
    let a5_effect =
        untreated.iter().map(|&i| a5.unit_effects[i]).sum::<f64>() / untreated.len() as f64;
    let a5_counterfactual =
        untreated.iter().map(|&i| ys[i] + a5.unit_effects[i]).sum::<f64>() / untreated.len() as f64;
    // B5 on the same law as a finite-discrete joint (Y levels -1..2).
    let y_levels = vec![-1.0, 0.0, 1.0, 2.0];
    let cards = [2usize, 3, 4];
    let mut counts = [0u64; 24];
    for (&(x, m, y), &c) in &cells {
        counts[x * 12 + m * 4 + y] = c;
    }
    let total_count: u64 = counts.iter().sum();
    let law = ExactDiscreteLaw::try_new(
        "target",
        RegimeId::from_raw(0),
        [],
        vec![
            DiscreteAxis { variable: v(0), values: vec![Value::f64(0.0), Value::f64(1.0)].into() },
            DiscreteAxis {
                variable: v(1),
                values: vec![Value::f64(0.0), Value::f64(1.0), Value::f64(2.0)].into(),
            },
            DiscreteAxis {
                variable: v(2),
                values: y_levels.iter().map(|&y| Value::f64(y)).collect::<Vec<_>>().into(),
            },
        ],
        counts.iter().map(|&c| c as f64 / total_count as f64).collect::<Vec<_>>(),
        "markov",
        LawTolerance::default(),
    )
    .unwrap();
    let prepared = prepare_counterfactual_id(
        dag,
        names(3),
        vec![vec![0.0, 1.0], vec![0.0, 1.0, 2.0], y_levels],
        &CounterfactualEventQuery::effect_on_treated(v(0), 1.0, 0.0, v(2), 0.0).unwrap(),
        CounterfactualIdOptions::default(),
        &ctx(),
    )
    .unwrap();
    let b5 = prepared.evaluate(&law, &ctx()).unwrap();
    assert_eq!(cards.len(), 3);
    assert!((b5.effect.unwrap() - a5_effect).abs() < 1e-9, "{:?} vs {a5_effect}", b5.effect);
    assert!(
        (b5.counterfactual_mean.unwrap() - a5_counterfactual).abs() < 1e-9,
        "{:?} vs {a5_counterfactual}",
        b5.counterfactual_mean
    );
    assert!((a5_effect - 1.0).abs() < 1e-9, "the structural total effect is 1");
}

// ---------------------------------------------------------------- artifact

#[test]
fn the_artifact_is_replayed_and_independently_recomputed() {
    let effect = frontdoor_effect();
    let bytes = effect.export_artifact().unwrap();
    let consumed = consume_counterfactual_id_artifact(
        &bytes,
        CounterfactualIdConsumeLimits::default(),
        &ctx(),
    )
    .unwrap();
    assert!(consumed.independently_verified);
    assert_eq!(consumed.probability.to_bits(), effect.probability.to_bits());
    assert_eq!(consumed.effect.map(f64::to_bits), effect.effect.map(f64::to_bits));
    assert_eq!(consumed.derivation(), effect.derivation());
    assert_eq!(consumed.data_digest, effect.data_digest);
    // Bound to the data it was computed on.
    consume_counterfactual_id_artifact_for_data(
        &bytes,
        &effect.data_digest,
        CounterfactualIdConsumeLimits::default(),
        &ctx(),
    )
    .unwrap();
    let other = frontdoor_scm(99, &[2, 3, 2]);
    let other_effect = prepare_frontdoor(&[2, 3, 2])
        .evaluate(&exact(&[2, 3, 2], other.observational(), "other"), &ctx())
        .unwrap();
    assert_eq!(
        consume_counterfactual_id_artifact_for_data(
            &bytes,
            &other_effect.data_digest,
            CounterfactualIdConsumeLimits::default(),
            &ctx(),
        )
        .unwrap_err(),
        CounterfactualIdArtifactError::DataIdentityMismatch
    );
    // A counted law round-trips too (origin and counts are carried).
    let counts = vec![30, 12, 7, 21, 9, 14, 25, 18];
    let counted_effect =
        prepare_frontdoor(&[2, 2, 2]).evaluate(&counted(&[2, 2, 2], counts), &ctx()).unwrap();
    let consumed = consume_counterfactual_id_artifact(
        &counted_effect.export_artifact().unwrap(),
        CounterfactualIdConsumeLimits::default(),
        &ctx(),
    )
    .unwrap();
    assert!(consumed.independently_verified);
    assert_eq!(consumed.probability.to_bits(), counted_effect.probability.to_bits());
}

fn wire_of(effect: &CounterfactualIdEffect) -> CounterfactualIdArtifactWire {
    CounterfactualIdArtifactWire::decode(&effect.export_artifact().unwrap()).unwrap()
}

fn consume_wire(
    wire: &CounterfactualIdArtifactWire,
) -> Result<CounterfactualIdEffect, CounterfactualIdArtifactError> {
    consume_counterfactual_id_artifact(
        &wire.export().unwrap(),
        CounterfactualIdConsumeLimits::default(),
        &ctx(),
    )
}

fn resealed(mut wire: CounterfactualIdArtifactWire) -> CounterfactualIdArtifactWire {
    wire.data_digest.clear();
    wire.premises_digest.clear();
    wire.sealed().unwrap()
}

#[test]
fn a_mutated_artifact_fails_consumption_with_a_typed_error() {
    let effect = frontdoor_effect();
    let base = wire_of(&effect);
    let bytes = effect.export_artifact().unwrap();
    // Corrupt bytes, another version or feature.
    assert!(matches!(
        consume_counterfactual_id_artifact(
            &bytes[..bytes.len() / 2],
            CounterfactualIdConsumeLimits::default(),
            &ctx()
        ),
        Err(CounterfactualIdArtifactError::Decode(_))
    ));
    let mut version = base.clone();
    version.version = 2;
    assert!(matches!(
        consume_wire(&version),
        Err(CounterfactualIdArtifactError::UnsupportedSemantics(_))
    ));
    let mut feature = base.clone();
    feature.required_features = vec!["cross_world_edge_contrast_v1".into()];
    assert!(matches!(
        consume_wire(&feature),
        Err(CounterfactualIdArtifactError::UnsupportedSemantics(_))
    ));
    // Unsealed edits: premises and data digests.
    let mut renamed = base.clone();
    renamed.variable_names[1] = "mediator".into();
    assert_eq!(
        consume_wire(&renamed).unwrap_err(),
        CounterfactualIdArtifactError::PremisesMismatch
    );
    let mut edge = base.clone();
    edge.directed.push((0, 2));
    assert_eq!(consume_wire(&edge).unwrap_err(), CounterfactualIdArtifactError::PremisesMismatch);
    let mut query = base.clone();
    query.query.event[0].2 = 0.0f64.to_bits();
    assert_eq!(consume_wire(&query).unwrap_err(), CounterfactualIdArtifactError::PremisesMismatch);
    let mut limits = base.clone();
    limits.search.operations_limit -= 1;
    assert_eq!(consume_wire(&limits).unwrap_err(), CounterfactualIdArtifactError::PremisesMismatch);
    let mut cell = base.clone();
    cell.law.probabilities.swap(0, 1);
    assert_eq!(
        consume_wire(&cell).unwrap_err(),
        CounterfactualIdArtifactError::DataIdentityMismatch
    );
    let mut data_digest = base.clone();
    data_digest.data_digest = "00".into();
    assert_eq!(
        consume_wire(&data_digest).unwrap_err(),
        CounterfactualIdArtifactError::DataIdentityMismatch
    );
    let mut premises_digest = base.clone();
    premises_digest.premises_digest = "00".into();
    assert_eq!(
        consume_wire(&premises_digest).unwrap_err(),
        CounterfactualIdArtifactError::PremisesMismatch
    );
    // Outputs are not premises: an edited derivation, graph, accounting or point
    // keeps valid digests and fails replay.
    let mut derivation = base.clone();
    derivation.derivation.push(' ');
    assert!(matches!(
        consume_wire(&derivation),
        Err(CounterfactualIdArtifactError::DerivationMismatch(_))
    ));
    let mut graph = base.clone();
    graph.counterfactual_graph = None;
    assert!(matches!(
        consume_wire(&graph),
        Err(CounterfactualIdArtifactError::DerivationMismatch(_))
    ));
    let mut accounting = base.clone();
    accounting.search.operations_consumed += 1;
    assert!(matches!(
        consume_wire(&accounting),
        Err(CounterfactualIdArtifactError::DerivationMismatch(_))
    ));
    let mut point = base.clone();
    point.point.probability = (f64::from_bits(point.point.probability) + 1e-12).to_bits();
    assert_eq!(consume_wire(&point).unwrap_err(), CounterfactualIdArtifactError::PointMismatch);
    let mut contrast = base.clone();
    contrast.point.effect = contrast.point.effect.map(|b| (f64::from_bits(b) * 2.0).to_bits());
    assert_eq!(consume_wire(&contrast).unwrap_err(), CounterfactualIdArtifactError::PointMismatch);
    // An unknown field on the wire is refused at decode.
    let mut value: serde_json::Value = serde_json::to_value(&base).unwrap();
    value["point"]["unexpected"] = 1.into();
    assert!(serde_json::from_value::<CounterfactualIdArtifactWire>(value).is_err());
    // Every error carries a registered (reason code, detail) pair.
    assert_eq!(
        CounterfactualIdArtifactError::PointMismatch.reason(),
        ("invalid_argument", "counterfactual_id.invalid_artifact")
    );
}

#[test]
fn resealed_semantic_mutations_fail_replay_for_the_right_reason() {
    let effect = frontdoor_effect();
    let base = wire_of(&effect);
    // A different law with valid digests: same derivation, different point.
    let mut law = base.clone();
    law.law.probabilities.swap(0, 1);
    assert_eq!(
        consume_wire(&resealed(law)).unwrap_err(),
        CounterfactualIdArtifactError::PointMismatch
    );
    // Another outcome level: the query text is in the derivation.
    let mut level = base.clone();
    level.query.event[0].2 = 0.0f64.to_bits();
    assert!(matches!(
        consume_wire(&resealed(level)),
        Err(CounterfactualIdArtifactError::DerivationMismatch(_))
    ));
    // An extra directed edge X -> Y: the front door no longer holds (bow arc).
    let mut edge = base.clone();
    edge.directed.push((0, 2));
    edge.directed.sort_unstable();
    let error = consume_wire(&resealed(edge)).unwrap_err();
    assert!(
        matches!(&error, CounterfactualIdArtifactError::DerivationMismatch(m) if m.contains("cross_world_not_identified")),
        "{error:?}"
    );
    // The latent confounder removed: identified differently.
    let mut unconfounded = base.clone();
    unconfounded.bidirected.clear();
    assert!(matches!(
        consume_wire(&resealed(unconfounded)),
        Err(CounterfactualIdArtifactError::DerivationMismatch(_))
    ));
    // A different but sufficient stored limit is another premise: the replay
    // runs under it and reproduces the same derivation and accounting.
    let mut limits = base.clone();
    limits.search.operations_limit -= 1;
    assert!(consume_wire(&resealed(limits)).unwrap().independently_verified);
    // A limit too small for the derivation: the replay stops, never a verdict.
    let mut tiny = base.clone();
    tiny.search.operations_limit = 3;
    let error = consume_wire(&resealed(tiny)).unwrap_err();
    assert!(
        matches!(&error, CounterfactualIdArtifactError::DerivationMismatch(m) if m.contains("transport_budget_cancel")),
        "{error:?}"
    );
    // Another estimand or contract text.
    let mut estimand = base.clone();
    estimand.estimand = format!("{COUNTERFACTUAL_ID_ESTIMAND} ");
    assert!(matches!(
        consume_wire(&resealed(estimand)),
        Err(CounterfactualIdArtifactError::UnsupportedSemantics("estimand"))
    ));
    let mut contract = base.clone();
    contract.contract = "other".into();
    assert!(matches!(
        consume_wire(&resealed(contract)),
        Err(CounterfactualIdArtifactError::UnsupportedSemantics("contract"))
    ));
    // A non-canonical query on the wire is malformed, not repaired.
    let mut duplicated = base.clone();
    let atom = duplicated.query.event[0];
    duplicated.query.event.push(atom);
    assert!(matches!(
        consume_wire(&resealed(duplicated)),
        Err(CounterfactualIdArtifactError::Malformed(_))
    ));
    // A label-only rename replays: names are premises, not semantics.
    let mut renamed = base.clone();
    renamed.variable_names = vec!["a".into(), "b".into(), "c".into()];
    let consumed = consume_wire(&resealed(renamed)).unwrap();
    assert_eq!(consumed.probability.to_bits(), effect.probability.to_bits());
    assert_eq!(consumed.names(), ["a", "b", "c"]);
}

#[test]
fn consumer_limits_refuse_before_any_work_and_bounds_hold_at_export() {
    let effect = frontdoor_effect();
    let base = wire_of(&effect);
    let bytes = effect.export_artifact().unwrap();
    let stored = base.search;
    // Consumer maxima below the stored limits: a limits refusal, never invalid.
    let below = |search: SearchLimits, memory: u64| CounterfactualIdConsumeLimits {
        search,
        search_memory_bytes: memory,
    };
    for limits in [
        below(
            SearchLimits { operations: stored.operations_limit - 1, depth: stored.depth_limit },
            stored.memory_limit_bytes,
        ),
        below(
            SearchLimits { operations: stored.operations_limit, depth: stored.depth_limit - 1 },
            stored.memory_limit_bytes,
        ),
        below(
            SearchLimits { operations: stored.operations_limit, depth: stored.depth_limit },
            stored.memory_limit_bytes - 1,
        ),
    ] {
        let error = consume_counterfactual_id_artifact(&bytes, limits, &ctx()).unwrap_err();
        assert!(matches!(error, CounterfactualIdArtifactError::LimitsExceeded(_)), "{error:?}");
        assert_eq!(error.reason(), ("route_not_supported", "counterfactual_id.bounds_exceeded"));
    }
    // The context's hard memory limit below the stored cap refuses up front too.
    let mut small = ctx();
    small.memory = MemoryBudget {
        soft_limit_bytes: None,
        hard_limit_bytes: Some(stored.memory_limit_bytes - 1),
    };
    assert!(matches!(
        consume_counterfactual_id_artifact(
            &bytes,
            CounterfactualIdConsumeLimits::default(),
            &small
        ),
        Err(CounterfactualIdArtifactError::LimitsExceeded(_))
    ));
    // Exactly the stored limits replay.
    let exact_limits = below(
        SearchLimits { operations: stored.operations_limit, depth: stored.depth_limit },
        stored.memory_limit_bytes,
    );
    assert!(
        consume_counterfactual_id_artifact(&bytes, exact_limits, &ctx())
            .unwrap()
            .independently_verified
    );
    // The format bound (6 variables, 4 levels) is enforced at export and at consumption.
    let mut wide = base.clone();
    wide.variable_names = (0..7).map(|i| format!("v{i}")).collect();
    assert_eq!(
        wide.export().unwrap_err(),
        CounterfactualIdArtifactError::LimitsExceeded("variables")
    );
    let mut many = base.clone();
    many.levels[0] = vec![0, 1, 2, 3, 4];
    assert_eq!(many.export().unwrap_err(), CounterfactualIdArtifactError::LimitsExceeded("levels"));
    // Encode past the producer check to confirm the consumer refuses it too.
    let mut value: serde_json::Value = serde_json::to_value(&base).unwrap();
    value["levels"][0] = serde_json::json!([0, 1, 2, 3, 4]);
    let forged: CounterfactualIdArtifactWire = serde_json::from_value(value).unwrap();
    let forged_bytes = antecedent_io::to_cbor(&forged).unwrap();
    assert_eq!(
        consume_counterfactual_id_artifact(
            &forged_bytes,
            CounterfactualIdConsumeLimits::default(),
            &ctx()
        )
        .unwrap_err(),
        CounterfactualIdArtifactError::LimitsExceeded("levels")
    );
}

#[test]
fn the_two_x8_formats_refuse_each_other() {
    let bytes = frontdoor_effect().export_artifact().unwrap();
    // The 2.2A reader refuses these bytes, whatever it names the failure.
    assert!(CrossWorldArtifactWire::decode(&bytes).is_err());
    // A 2.2A artifact is refused by this reader.
    // Continuous columns, the 2.2A cell's linear-Gaussian fit.
    let xs: Vec<f64> = (0..40).map(|i| (f64::from(i) * 0.37).sin()).collect();
    let ms: Vec<f64> =
        xs.iter().enumerate().map(|(i, x)| 0.8 * x + (i as f64 * 0.91).cos()).collect();
    let ys: Vec<f64> = xs.iter().zip(&ms).map(|(x, m)| 1.7 * x + 4.0 * m).collect();
    let data =
        TabularData::from_f64_columns([("x", &xs[..]), ("m", &ms[..]), ("y", &ys[..])]).unwrap();
    let mut dag = Dag::with_variables(3);
    for (a, b) in [(0, 1), (0, 2), (1, 2)] {
        dag.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    let query = CrossWorldQuery::natural_direct(v(0), v(1), v(2), 0.0, 1.0).unwrap();
    let a5 = evaluate_cross_world_effect(dag, &data, &query, CrossWorldOptions::default(), &ctx())
        .unwrap();
    let a5_bytes = a5.export_artifact().unwrap();
    assert!(
        consume_counterfactual_id_artifact(
            &a5_bytes,
            CounterfactualIdConsumeLimits::default(),
            &ctx()
        )
        .is_err()
    );
}

// ---------------------------------------------------------------- refusals

fn refusal_text(result: Result<PreparedCounterfactualId, antecedent::CausalError>) -> String {
    result.unwrap_err().to_string()
}

#[test]
fn structures_queries_and_inference_outside_the_cell_are_refused() {
    let query = CounterfactualEventQuery::effect_on_treated(v(0), 1.0, 0.0, v(2), 1.0).unwrap();
    let text = refusal_text(prepare_counterfactual_id(
        admg(3, &FRONT_DIRECTED, &FRONT_BIDIRECTED),
        names(3),
        levels(&[2, 2, 2]),
        &query,
        CounterfactualIdOptions { interval_requested: true, ..CounterfactualIdOptions::default() },
        &ctx(),
    ));
    assert!(
        text.contains("reason=estimator_inference_mismatch: counterfactual_id.interval_requested"),
        "{text}"
    );
    for text in [
        refusal_text(prepare_counterfactual_id(
            Cpdag::with_variables(3),
            names(3),
            levels(&[2, 2, 2]),
            &query,
            CounterfactualIdOptions::default(),
            &ctx(),
        )),
        refusal_text(prepare_counterfactual_id(
            Pag::with_variables(3),
            names(3),
            levels(&[2, 2, 2]),
            &query,
            CounterfactualIdOptions::default(),
            &ctx(),
        )),
        refusal_text(prepare_counterfactual_id(
            TemporalDag::empty(),
            names(3),
            levels(&[2, 2, 2]),
            &query,
            CounterfactualIdOptions::default(),
            &ctx(),
        )),
        refusal_text(prepare_counterfactual_id(
            AcceptedGraph::from(admg(3, &FRONT_DIRECTED, &FRONT_BIDIRECTED)),
            names(3),
            levels(&[2, 2, 2]),
            &query,
            CounterfactualIdOptions::default(),
            &ctx(),
        )),
    ] {
        assert!(
            text.contains("reason=cell_not_licensed: counterfactual_id.graph_outside_contract"),
            "{text}"
        );
    }
    // Not identified: the bow arc, with the conflicting pair in the message.
    let text = refusal_text(prepare_counterfactual_id(
        admg(3, &[(0, 2), (1, 2)], &[(0, 2)]),
        names(3),
        levels(&[2, 2, 2]),
        &query,
        CounterfactualIdOptions::default(),
        &ctx(),
    ));
    assert!(
        text.contains(
            "reason=cross_world_not_identified: counterfactual_id.conflicting_subscripts"
        ),
        "{text}"
    );
    // A path-specific (routed) query on an ADMG: deferred to 2.3.
    let at = |w: u8, var: u32, level: f64| {
        CounterfactualEvent::new(WorldId::new(w), v(var), level).unwrap()
    };
    let routed = CounterfactualEventQuery::new(
        vec![
            WorldSpec::new([], []).unwrap(),
            WorldSpec::new(
                [(v(0), 1.0)],
                [EdgeRoute { parent: v(1), child: v(2), source: WorldId::new(0) }],
            )
            .unwrap(),
        ],
        ExogenousCoupling::SharedLatentExogenous,
        [at(1, 2, 1.0)],
        [at(0, 0, 0.0)],
    )
    .unwrap();
    let text = refusal_text(prepare_counterfactual_id(
        admg(3, &FRONT_DIRECTED, &FRONT_BIDIRECTED),
        names(3),
        levels(&[2, 2, 2]),
        &routed,
        CounterfactualIdOptions::default(),
        &ctx(),
    ));
    assert!(
        text.contains("reason=route_not_supported: counterfactual_id.path_specific_deferred"),
        "{text}"
    );
    // Another shape: a joint of two potential outcomes.
    let joint = CounterfactualEventQuery::new(
        vec![WorldSpec::new([], []).unwrap(), WorldSpec::new([(v(0), 1.0)], []).unwrap()],
        ExogenousCoupling::SharedLatentExogenous,
        [at(1, 2, 1.0), at(0, 2, 0.0)],
        [],
    )
    .unwrap();
    let text = refusal_text(prepare_counterfactual_id(
        admg(3, &FRONT_DIRECTED, &FRONT_BIDIRECTED),
        names(3),
        levels(&[2, 2, 2]),
        &joint,
        CounterfactualIdOptions::default(),
        &ctx(),
    ));
    assert!(
        text.contains("reason=route_not_supported: counterfactual_id.query_outside_contract"),
        "{text}"
    );
    // Names that do not match the graph.
    let text = refusal_text(prepare_counterfactual_id(
        admg(3, &FRONT_DIRECTED, &FRONT_BIDIRECTED),
        names(2),
        levels(&[2, 2, 2]),
        &query,
        CounterfactualIdOptions::default(),
        &ctx(),
    ));
    assert!(text.contains("reason=invalid_argument: counterfactual_id.invalid_query"), "{text}");
    // Limits over the cap.
    let text = refusal_text(prepare_counterfactual_id(
        admg(3, &FRONT_DIRECTED, &FRONT_BIDIRECTED),
        names(3),
        levels(&[2, 2, 2]),
        &query,
        CounterfactualIdOptions {
            limits: SearchLimits { operations: 100_001, depth: 10 },
            ..CounterfactualIdOptions::default()
        },
        &ctx(),
    ));
    assert!(
        text.contains("reason=route_not_supported: counterfactual_id.bounds_exceeded"),
        "{text}"
    );
    // A budget stop.
    let text = refusal_text(prepare_counterfactual_id(
        admg(3, &FRONT_DIRECTED, &FRONT_BIDIRECTED),
        names(3),
        levels(&[2, 2, 2]),
        &query,
        CounterfactualIdOptions {
            limits: SearchLimits { operations: 2, depth: 10 },
            ..CounterfactualIdOptions::default()
        },
        &ctx(),
    ));
    assert!(text.contains("reason=transport_budget_cancel: counterfactual_id.budget"), "{text}");
}

#[test]
fn laws_without_support_or_positivity_are_refused() {
    let cards = [2, 2, 2];
    let prepared = prepare_frontdoor(&cards);
    // P(X = 0) = 0: the conditioning event is null.
    let mut probabilities = vec![0.0; 8];
    for (i, p) in probabilities.iter_mut().enumerate().skip(4) {
        *p = [0.1, 0.2, 0.3, 0.4][i - 4];
    }
    let text =
        prepared.evaluate(&exact(&cards, probabilities, "null"), &ctx()).unwrap_err().to_string();
    assert!(text.contains("reason=invalid_argument: counterfactual_id.invalid_query"), "{text}");
    // P(X = 0, M = 1) = 0 while P(M = 1 | X = 1) > 0: P(y | m = 1, x' = 0) is undefined.
    let probabilities = vec![0.2, 0.2, 0.0, 0.0, 0.1, 0.1, 0.2, 0.2];
    let text =
        prepared.evaluate(&exact(&cards, probabilities, "gap"), &ctx()).unwrap_err().to_string();
    assert!(
        text.contains("reason=invalid_argument: counterfactual_id.positivity_violation"),
        "{text}"
    );
    // A law over other variables or levels.
    let text = prepared
        .evaluate(&exact(&[2, 2], vec![0.25; 4], "narrow"), &ctx())
        .unwrap_err()
        .to_string();
    assert!(text.contains("counterfactual_id.invalid_query"), "{text}");
}

#[test]
fn prepare_decides_once_and_matches_the_identification_route() {
    use antecedent_identify::counterfactual_id::{
        COUNTERFACTUAL_ID_DEFAULT_LIMITS, COUNTERFACTUAL_ID_MEMORY_BYTES, CounterfactualIdProblem,
        decide_counterfactual_id,
    };
    let query = CounterfactualEventQuery::effect_on_treated(v(0), 1.0, 0.0, v(2), 1.0).unwrap();
    let prepared = prepare_counterfactual_id(
        admg(3, &FRONT_DIRECTED, &FRONT_BIDIRECTED),
        names(3),
        levels(&[2, 3, 2]),
        &query,
        CounterfactualIdOptions::default(),
        &ctx(),
    )
    .unwrap();
    let problem =
        CounterfactualIdProblem::new(levels(&[2, 3, 2]), &FRONT_DIRECTED, &FRONT_BIDIRECTED)
            .unwrap();
    let direct = decide_counterfactual_id(
        &problem,
        &query,
        COUNTERFACTUAL_ID_DEFAULT_LIMITS,
        COUNTERFACTUAL_ID_MEMORY_BYTES,
        &ctx(),
    )
    .unwrap();
    assert_eq!(prepared.derivation(), &direct);
    assert_eq!(prepared.query(), &query);
    assert_eq!(prepared.derivation().numerators.len(), 2, "one functional per level of Y");
    // A DAG is accepted as an ADMG without bidirected edges.
    let mut dag = Dag::with_variables(3);
    for (a, b) in [(0, 1), (1, 2)] {
        dag.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    let from_dag = prepare_counterfactual_id(
        dag,
        names(3),
        levels(&[2, 3, 2]),
        &query,
        CounterfactualIdOptions::default(),
        &ctx(),
    )
    .unwrap();
    assert!(from_dag.derivation().counterfactual_graph.is_some());
}
