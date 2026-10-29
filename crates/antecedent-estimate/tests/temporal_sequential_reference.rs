//! The unfolded linear g-formula engine against an exact rational reference.
//!
//! `oracle_data/temporal_sequential_reference.py` (frozen in the JSON beside it) fixes a small
//! discrete-time SCM (confounder Z, treatment A, mediator M, outcome Y; a three-period window
//! A(-2), A(-1), A(0) against the outcome Y(0)), fits every needed mechanism by exact rational
//! least squares and propagates the g-formula by exact enumeration of the unfolded nodes, for the
//! Sustained contrast and for Sequence levels under hard, shift, multiplicative and bounded
//! overlays. The product's point values are asserted against it.
//!
//! The Frequentist block bootstrap is checked piece by piece. The replicate row maps are
//! regenerated here from the library RNG stream (and compared with the frozen copy the reference
//! read), each replicate is refit exactly on its rows, and the published standard error is
//! the replicate standard deviation times the fixed-b polynomial times the kernel-bias factor the
//! estimate itself publishes (recomputed by the reference). The block length is the dependence-aware Politis-White rule,
//! recomputed from the exact contrast influence and the exact normal-equation scores, and the
//! Bartlett kernel-bias factor from a reference autoregression of the exact influence. The
//! influence is compared with the exact one. The Bayesian composition is compared with the exact
//! moments of the composed Student-t draws, given the tempering factor each mechanism's own note
//! publishes (frozen in the inputs file and compared with the live note).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent_core::{
    AssumptionSet, ExecutionContext, IdentificationStatus, Intervention, Lag, StreamDomain,
    TargetPopulation, TemporalEffectQuery, TemporalNodeKey, TemporalPolicy, Value, VariableId,
    derive_stream,
};
use std::sync::Arc;

use antecedent_core::{CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TemporalIndexer,
    TimeIndex, TimeSeriesData, ValidityBitmap, fill_circular_block_indexes,
};
use antecedent_estimate::serial_dependence::tempering_kappa_from_notes;
use antecedent_estimate::temporal_sequential::{
    SequenceTarget, SequentialContrastDesign, SequentialInference, SequentialMechanismOverlay,
    SequentialNodeOverlay, SequentialProblem, estimate_sequence_mechanisms,
    estimate_sustained_window, estimate_sustained_window_with_gaps,
};
use antecedent_estimate::{BayesianBackendKind, BayesianGComputationAte, EffectEstimate};
use antecedent_expr::IdentifiedEstimand;
use antecedent_graph::{DenseNodeId, TemporalDag, ensure_lagged};
use antecedent_identify::TemporalBackdoorIdentifier;
use serde_json::Value as Json;

const Z: u32 = 0;
const A: u32 = 1;
const M: u32 = 2;
const Y: u32 = 3;
/// Base of the per-replicate streams of the sequential row bootstrap.
const BLOCK_STREAM: u64 = 0x5120_0000;
const CTX_SEED: u64 = 5;
const REPLICATES: u32 = 12;
const DRAWS: usize = 200_000;
const PRIOR_SCALE: f64 = 4.0;

/// The rendered refusal of a result that must be an `Err`.
fn refusal<T, E: std::fmt::Display>(result: Result<T, E>) -> String {
    match result {
        Ok(_) => panic!("expected a refusal, got Ok"),
        Err(error) => error.to_string(),
    }
}

fn reference() -> Json {
    serde_json::from_str(include_str!("oracle_data/temporal_sequential_reference.json")).unwrap()
}

fn inputs() -> Json {
    serde_json::from_str(include_str!("oracle_data/temporal_sequential_reference_inputs.json"))
        .unwrap()
}

fn floats(v: &Json) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn series(block: &Json) -> TimeSeriesData {
    let (z, a, m, y) =
        (floats(&block["z"]), floats(&block["a"]), floats(&block["m"]), floats(&block["y"]));
    TimeSeriesData::from_f64_columns(
        [("z", z.as_slice()), ("a", a.as_slice()), ("m", m.as_slice()), ("y", y.as_slice())],
        1,
    )
    .unwrap()
}

/// The series with some cells missing and/or per-time sampling weights declared.
fn series_with(block: &Json, missing: &Json, weights: Option<Vec<f64>>) -> TimeSeriesData {
    let mut builder = CausalSchemaBuilder::new();
    for name in ["z", "a", "m", "y"] {
        builder
            .add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
    }
    let schema = builder.build().unwrap();
    let n = floats(&block["z"]).len();
    let columns = ["z", "a", "m", "y"]
        .iter()
        .enumerate()
        .map(|(id, key)| {
            let gone: Vec<u64> = missing
                .get(id.to_string())
                .map(|v| v.as_array().unwrap().iter().map(|t| t.as_u64().unwrap()).collect())
                .unwrap_or_default();
            let valid: Vec<bool> = (0..n).map(|t| !gone.contains(&(t as u64))).collect();
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(id as u32),
                    Arc::from(floats(&block[*key])),
                    ValidityBitmap::from_bools(&valid),
                )
                .unwrap(),
            )
        })
        .collect();
    let storage =
        OwnedColumnarStorage::try_new(schema, columns, None, weights.map(Arc::from)).unwrap();
    TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap()
}

fn graph() -> TemporalDag {
    let mut g = TemporalDag::empty();
    let id = VariableId::from_raw;
    let mut edge = |from: u32, from_lag: u32, to: u32| {
        let src = ensure_lagged(&mut g, id(from), Lag::from_raw(from_lag)).unwrap();
        let dst = ensure_lagged(&mut g, id(to), Lag::CONTEMPORANEOUS).unwrap();
        g.insert_directed(src, dst).unwrap();
    };
    edge(Z, 0, A);
    edge(Z, 0, Y);
    edge(Z, 1, Y);
    edge(A, 1, M);
    edge(A, 2, M);
    edge(M, 0, Y);
    edge(M, 1, Y);
    edge(A, 0, Y);
    g
}

/// A graph, its identification and the unfolded estimand every estimate in this file fits.
struct Identified {
    graph: TemporalDag,
    indexer: TemporalIndexer,
    estimand: IdentifiedEstimand,
    status: IdentificationStatus,
}

impl Identified {
    fn problem<'a>(&'a self, data: &'a TimeSeriesData) -> SequentialProblem<'a> {
        SequentialProblem {
            data,
            graph: &self.graph,
            indexer: &self.indexer,
            estimand: &self.estimand,
        }
    }
}

fn window_query(active: f64, control: f64) -> TemporalEffectQuery {
    window_query_over(active, control, (-2, 0))
}

fn window_query_over(active: f64, control: f64, window: (i32, i32)) -> TemporalEffectQuery {
    let mut query =
        TemporalEffectQuery::sustained(VariableId::from_raw(A), VariableId::from_raw(Y), 0, active)
            .with_policy(TemporalPolicy::sustained(window.0, window.1));
    query.control = Intervention::set(VariableId::from_raw(A), Value::f64(control));
    query
}

fn identify_contrast(query: &TemporalEffectQuery) -> Identified {
    let graph = graph();
    let result = TemporalBackdoorIdentifier::new().identify_temporal(&graph, query).unwrap();
    Identified {
        estimand: result.result.estimands.first().cloned().expect("identified estimand"),
        status: result.result.status,
        indexer: result.indexer,
        graph,
    }
}

fn identify_schedule(schedule: &[(VariableId, i32, Option<f64>)]) -> Identified {
    let graph = graph();
    let result = TemporalBackdoorIdentifier::new()
        .identify_temporal_schedule(
            &graph,
            VariableId::from_raw(Y),
            0,
            schedule,
            None,
            TargetPopulation::AllObserved,
        )
        .unwrap();
    Identified {
        estimand: result.result.estimands.first().cloned().expect("identified estimand"),
        status: result.result.status,
        indexer: result.indexer,
        graph,
    }
}

fn overlays_of(case: &Json) -> Vec<SequentialMechanismOverlay> {
    case["overlays"]
        .as_array()
        .unwrap()
        .iter()
        .map(|o| {
            let node = o["node"].as_array().unwrap();
            SequentialMechanismOverlay {
                node: SequentialNodeOverlay {
                    variable: VariableId::from_raw(node[0].as_u64().unwrap() as u32),
                    offset: node[1].as_i64().unwrap() as i32,
                    level: o.get("level").and_then(Json::as_f64),
                    shift: o.get("shift").and_then(Json::as_f64).unwrap_or(0.0),
                },
                multiplier: o.get("mult").and_then(Json::as_f64).unwrap_or(1.0),
                bounds: o.get("bounds").map(|b| {
                    let b = floats(b);
                    (b[0], b[1])
                }),
            }
        })
        .collect()
}

fn schedule_of(overlays: &[SequentialMechanismOverlay]) -> Vec<(VariableId, i32, Option<f64>)> {
    overlays.iter().map(|o| (o.node.variable, o.node.offset, o.node.level)).collect()
}

fn inference<'a>(
    identified: &Identified,
    replicates: u32,
    bayesian: Option<&'a BayesianGComputationAte>,
) -> SequentialInference<'a> {
    SequentialInference {
        status: identified.status,
        assumptions: AssumptionSet::new(),
        bootstrap_replicates: replicates,
        bayesian,
    }
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(CTX_SEED)
}

fn close(label: &str, got: f64, want: f64, tolerance: f64) {
    assert!(
        (got - want).abs() <= tolerance * (1.0 + want.abs()),
        "{label}: product {got} vs reference {want}"
    );
}

/// The product estimate of a reference case: a Sustained contrast or a Sequence level.
fn estimate_case(
    case: &Json,
    data: &TimeSeriesData,
    replicates: u32,
    bayesian: Option<&BayesianGComputationAte>,
) -> (EffectEstimate, Option<antecedent_estimate::CausalPosterior>) {
    if case["kind"] == "contrast" {
        let delta = case["delta"].as_f64().unwrap();
        let window = floats(&case["window"]);
        let query = window_query_over(delta, 0.0, (window[0] as i32, window[1] as i32));
        let identified = identify_contrast(&query);
        estimate_sustained_window(
            &identified.problem(data),
            &query,
            inference(&identified, replicates, bayesian),
            &ctx(),
        )
        .unwrap()
    } else {
        let overlays = overlays_of(case);
        let identified = identify_schedule(&schedule_of(&overlays));
        estimate_sequence_mechanisms(
            &identified.problem(data),
            &SequenceTarget {
                outcome: VariableId::from_raw(Y),
                outcome_offset: 0,
                overlays: &overlays,
            },
            inference(&identified, replicates, bayesian),
            &ctx(),
        )
        .unwrap()
    }
}

fn point_case<'a>(reference: &'a Json, name: &str) -> &'a Json {
    reference["point"].as_array().unwrap().iter().find(|c| c["name"] == name).unwrap()
}

#[test]
fn temporal_sequential_unfolded_structure_matches_the_hand_derived_graph() {
    let reference = reference();
    let identified = identify_contrast(&window_query(1.0, 0.0));
    let unfolded = identified.graph.unfold(identified.indexer.clone()).unwrap();
    let expected = reference["structure"].as_object().unwrap();
    assert_eq!(unfolded.dag.node_count(), expected.len());
    let label = |dense: usize| {
        let key = identified.indexer.key_of(dense as u32).unwrap();
        format!("{}@{}", key.variable.raw(), key.offset)
    };
    for node in 0..unfolded.dag.node_count() {
        let mut parents: Vec<String> = unfolded
            .dag
            .parents(DenseNodeId::from_raw(node as u32))
            .iter()
            .map(|p| label(p.as_usize()))
            .collect();
        parents.sort();
        let mut want: Vec<String> = expected[&label(node)]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p.as_str().unwrap().to_owned())
            .collect();
        want.sort();
        assert_eq!(parents, want, "parents of {}", label(node));
    }
    assert_eq!(
        identified
            .indexer
            .dense_id(TemporalNodeKey { variable: VariableId::from_raw(Y), offset: 0 })
            .is_ok(),
        true
    );
}

#[test]
fn temporal_sequential_point_values_equal_the_exact_g_formula() {
    let reference = reference();
    let data = series(&reference["data"]);
    for case in reference["point"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let (effect, posterior) = estimate_case(case, &data, 0, None);
        assert!(posterior.is_none());
        close(name, effect.ate, case["value"].as_f64().unwrap(), 1e-10);
        assert!(effect.se_bootstrap().is_none(), "{name}: no replicates, no bootstrap SE");
    }
    // The bounded cases are not slack: the clamp moves the level away from the unbounded shift.
    let level = |name: &str| point_case(&reference, name)["value"].as_f64().unwrap();
    assert!((level("level_bounded_upper") - level("level_bounded_slack")).abs() > 1e-3);
    assert!((level("level_bounded_lower") - level("level_bounded_slack")).abs() > 1e-3);
    assert!((level("level_multiplicative") - level("level_mixture")).abs() > 1e-3);
}

#[test]
fn temporal_sequential_fitted_mechanisms_match_the_dose_scaled_contrast() {
    // The contrast is linear in the active-minus-control difference: delta 3 is 3 times delta 1.
    let reference = reference();
    let unit = point_case(&reference, "contrast_unit")["value"].as_f64().unwrap();
    let dose = point_case(&reference, "contrast_dose")["value"].as_f64().unwrap();
    close("dose", dose, 3.0 * unit, 1e-12);
    let data = series(&reference["data"]);
    let query = window_query(2.5, -0.5);
    let identified = identify_contrast(&query);
    let (effect, _) = estimate_sustained_window(
        &identified.problem(&data),
        &query,
        inference(&identified, 0, None),
        &ctx(),
    )
    .unwrap();
    close("active 2.5 control -0.5", effect.ate, dose, 1e-10);
}

#[test]
fn temporal_sequential_prepared_design_refits_every_mechanism_on_frozen_rows() {
    let reference = reference();
    let data = series(&reference["data"]);
    let query = window_query(1.0, 0.0);
    let identified = identify_contrast(&query);
    let design = SequentialContrastDesign::prepare(
        &identified.problem(&data),
        &query,
        identified.status,
        &ctx(),
    )
    .unwrap();
    let unit = point_case(&reference, "contrast_unit")["value"].as_f64().unwrap();
    close("full", design.estimate().unwrap(), unit, 1e-10);
    let rows: Vec<usize> = reference["fixed_rows"]["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r.as_u64().unwrap() as usize)
        .collect();
    close(
        "frozen rows",
        design.estimate_on_rows(&rows).unwrap(),
        reference["fixed_rows"]["contrast"].as_f64().unwrap(),
        1e-9,
    );
    let aligned = design.aligned_rows();
    assert_eq!((aligned.first_time, aligned.rows), (3, 21));
    assert_eq!(design.structural_span(), 4);
    assert!(
        refusal(design.estimate_on_rows(&[21])).contains("aligned row index out of range"),
        "a row past the aligned rows is refused"
    );
    assert_eq!(
        design.block_length(),
        reference["bootstrap"]["long_contrast"]["block_length"].as_u64().unwrap() as usize
    );
}

/// Replicate `r`'s row map as the product draws it: the library stream, then circular blocks.
fn regenerated_rows(rows: usize, block: usize, replicates: u32) -> Vec<Vec<usize>> {
    let ctx = ctx();
    (0..replicates)
        .map(|r| {
            let mut rng = ctx
                .rng
                .stream_for(StreamDomain::TemporalBlock, derive_stream(BLOCK_STREAM, u64::from(r)));
            let mut map = vec![0usize; rows];
            fill_circular_block_indexes(rows, block.clamp(1, rows), &mut rng, &mut map).unwrap();
            map
        })
        .collect()
}

fn bootstrap_cases() -> [(&'static str, &'static str, &'static str); 4] {
    [
        ("short_contrast", "contrast_unit", "short"),
        ("long_contrast", "contrast_unit", "data"),
        ("long_level", "level_hard", "data"),
        ("persistent_contrast", "contrast_unit", "persistent"),
    ]
}

#[test]
fn temporal_sequential_block_bootstrap_interval_is_the_scaled_replicate_deviation() {
    let reference = reference();
    let inputs = inputs();
    for (tag, name, data_key) in bootstrap_cases() {
        let entry = &reference["bootstrap"][tag];
        let data = series(&reference[data_key]);
        let case = point_case(&reference, name);
        let (effect, _) = estimate_case(case, &data, REPLICATES, None);
        let block = effect.uncertainty.block.resampling.expect("block geometry");
        // The reference derives the block length from the exact influence and scores.
        assert_eq!(block.block_length, entry["block_length"].as_u64().unwrap() as usize, "{tag}");
        assert_eq!(block.rows, entry["rows"].as_u64().unwrap() as usize, "{tag}");
        close(tag, effect.ate, entry["value"].as_f64().unwrap(), 1e-10);
        // The replicate row maps are the library stream's, and equal the frozen copy.
        let rows = regenerated_rows(block.rows, block.block_length, REPLICATES);
        let frozen: Vec<Vec<usize>> = inputs["rows"][tag]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m.as_array().unwrap().iter().map(|r| r.as_u64().unwrap() as usize).collect())
            .collect();
        assert_eq!(rows, frozen, "{tag}: regenerated rows differ from the frozen copy");
        // A replicate whose resampled rows leave the normal equations singular is a soft failure.
        let ok = entry["replicates_ok"].as_u64().unwrap() as u32;
        assert_eq!(effect.uncertainty.bootstrap.replicates_ok, Some(ok), "{tag}");
        assert_eq!(effect.uncertainty.bootstrap.replicates_failed, Some(REPLICATES - ok), "{tag}");
        // se = raw replicate sd * fixed-b factor * the published kernel-bias factor; more than half
        // of the replicates failing publishes none.
        match entry["raw_sd"].as_f64() {
            Some(raw) => {
                let se = effect.se_bootstrap().expect("bootstrap SE");
                let fixed_b = entry["fixed_b"].as_f64().unwrap();
                close(tag, se, raw * fixed_b * block.kernel_bias, 1e-9);
            }
            None => {
                assert!(effect.se_bootstrap().is_none(), "{tag}: SE published past the failure cap")
            }
        }
        close(tag, block.kernel_bias, entry["kernel_bias"].as_f64().unwrap(), 1e-9);
        if data_key == "short" {
            assert_eq!(block.kernel_bias, 1.0, "fewer than eight rows carry no kernel bias");
        }
    }
}

#[test]
fn temporal_sequential_replicates_equal_exact_refits_on_the_regenerated_rows() {
    let reference = reference();
    let inputs = inputs();
    let query = window_query(1.0, 0.0);
    let identified = identify_contrast(&query);
    for (tag, data_key) in [
        ("short_contrast", "short"),
        ("long_contrast", "data"),
        ("persistent_contrast", "persistent"),
    ] {
        let data = series(&reference[data_key]);
        let design = SequentialContrastDesign::prepare(
            &identified.problem(&data),
            &query,
            identified.status,
            &ctx(),
        )
        .unwrap();
        let replicates: Vec<Option<f64>> = reference["bootstrap"][tag]["replicates"]
            .as_array()
            .unwrap()
            .iter()
            .map(Json::as_f64)
            .collect();
        let maps = inputs["rows"][tag].as_array().unwrap();
        assert_eq!(maps.len(), replicates.len());
        for (map, want) in maps.iter().zip(&replicates) {
            let rows: Vec<usize> =
                map.as_array().unwrap().iter().map(|r| r.as_u64().unwrap() as usize).collect();
            match want {
                Some(want) => close(tag, design.estimate_on_rows(&rows).unwrap(), *want, 1e-9),
                None => assert!(
                    design.estimate_on_rows(&rows).is_err(),
                    "{tag}: a singular resample must not be fit"
                ),
            }
        }
    }
}

#[test]
fn temporal_sequential_influence_equals_the_exact_score_of_the_propagated_value() {
    let reference = reference();
    for (tag, name, data_key) in bootstrap_cases() {
        let entry = &reference["bootstrap"][tag];
        let data = series(&reference[data_key]);
        let (effect, _) = estimate_case(point_case(&reference, name), &data, 0, None);
        let got = effect.scores.influence.as_ref().expect("influence").to_vec();
        let want = floats(&entry["influence"]);
        assert_eq!(got.len(), want.len(), "{tag}");
        let scale = want.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
        for (r, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!((g - w).abs() <= 1e-7 * scale, "{tag} row {r}: {g} vs {w}");
        }
        // Without replicates the block length is the structural rule, and no SE is published.
        let block = effect.uncertainty.block.resampling.expect("rule geometry");
        assert_eq!(block.block_length, entry["rule"].as_u64().unwrap() as usize, "{tag}");
    }
}

#[test]
fn temporal_sequential_dependence_block_length_is_the_politis_white_rule() {
    let reference = reference();
    let short = &reference["bootstrap"]["short_contrast"];
    // Seven rows: no Politis-White bandwidth, so the rule length (the unfolded span) stands.
    assert_eq!(short["pw"], 0);
    assert_eq!(short["block_length"], short["rule"]);
    assert_eq!(short["block_length"], 4);
    let long = &reference["bootstrap"]["long_contrast"];
    let (rule, testing) = (long["rule"].as_u64().unwrap(), long["testing"].as_u64().unwrap());
    // 21 rows: the rule is max(span 4, ceil cbrt 21 = 3) and the cap rows/3 = 7 binds the testing length.
    assert_eq!(rule, 4);
    assert_eq!(long["block_length"].as_u64().unwrap(), rule.max(testing.min(7)));
    // No ceiling is within 1e-6 of an integer boundary, so float order cannot move the length.
    for b in long["pw_continuous"].as_array().unwrap() {
        let b = b.as_f64().unwrap();
        assert!((b - b.round()).abs() > 1e-6 || b.round() == 0.0, "fragile ceiling {b}");
    }
}

fn kappas_by_rows(notes: &[std::sync::Arc<str>]) -> BTreeMap<u64, f64> {
    notes
        .iter()
        .filter(|note| note.starts_with("serial_dependence.long_run_tempering"))
        .filter_map(|note| {
            let field = |key: &str| {
                note.split_whitespace().find_map(|kv| kv.strip_prefix(key)).map(str::to_owned)
            };
            let rows = field("n=")?.parse::<u64>().ok()?;
            Some((rows, tempering_kappa_from_notes(std::slice::from_ref(note))?))
        })
        .collect()
}

fn bayes_estimator() -> BayesianGComputationAte {
    BayesianGComputationAte::conjugate()
        .with_n_draws(DRAWS)
        .with_seed(17)
        .with_prior_scale(PRIOR_SCALE)
}

fn bayes_run(
    reference: &Json,
    name: &str,
) -> (EffectEstimate, antecedent_estimate::CausalPosterior) {
    let data = series(&reference["data"]);
    let (effect, posterior) =
        estimate_case(point_case(reference, name), &data, 0, Some(&bayes_estimator()));
    (effect, posterior.expect("posterior"))
}

#[test]
fn temporal_sequential_bayesian_tempering_factors_are_the_ones_the_reference_assumed() {
    let reference = reference();
    let inputs = inputs();
    for (kind, name) in [("contrast", "contrast_unit"), ("level", "level_hard")] {
        let (_, posterior) = bayes_run(&reference, name);
        let live = kappas_by_rows(&posterior.diagnostics.notes);
        // Row counts of the A, Y and M mechanisms' unique observed rows.
        for (mechanism, rows) in [("A", 24_u64), ("Y", 23), ("M", 22)] {
            let frozen = inputs["kappa"][kind][mechanism].as_f64().unwrap();
            let got = *live.get(&rows).unwrap_or_else(|| panic!("{kind}: no note for {rows} rows"));
            assert!((got - frozen).abs() <= 1e-5, "{kind} {mechanism}: live {got} vs {frozen}");
            assert!(got >= 1.0, "tempering never widens the likelihood below 1");
        }
    }
}

#[test]
fn temporal_sequential_bayesian_contrast_has_the_exact_composed_moments() {
    let reference = reference();
    let bayes = &reference["bayes"]["contrast"];
    let (effect, posterior) = bayes_run(&reference, "contrast_unit");
    let column = posterior.effect_column().expect("effect column");
    let (mean, sd) = (posterior.summaries.mean[column], posterior.summaries.sd[column]);
    let (want_mean, want_sd) = (bayes["mean"].as_f64().unwrap(), bayes["sd"].as_f64().unwrap());
    let root = (DRAWS as f64).sqrt();
    assert!((mean - want_mean).abs() < 5.0 * want_sd / root, "mean {mean} vs {want_mean}");
    assert!((sd / want_sd - 1.0).abs() < 10.0 / root, "sd {sd} vs {want_sd}");
    assert_eq!(effect.ate, mean);
    assert_eq!(posterior.draws.n_draws, DRAWS);
}

#[test]
fn temporal_sequential_bayesian_contrast_on_a_gapped_series_uses_each_mechanisms_intact_rows() {
    let reference = reference();
    let bayes = &reference["bayes"]["gapped"];
    let data = series_with(&reference["data"], &reference["gaps"]["missing"], None);
    let query = window_query(1.0, 0.0);
    let identified = identify_contrast(&query);
    let estimator = bayes_estimator();
    let (_, posterior, _) = estimate_sustained_window_with_gaps(
        &identified.problem(&data),
        &query,
        inference(&identified, 0, Some(&estimator)),
        &ctx(),
        None,
    )
    .unwrap();
    let posterior = posterior.expect("posterior");
    let live = kappas_by_rows(&posterior.diagnostics.notes);
    for mechanism in ["A", "M", "Y"] {
        let rows = bayes["rows"][mechanism].as_u64().unwrap();
        let frozen = inputs()["kappa"]["gapped"][mechanism].as_f64().unwrap();
        let got = *live.get(&rows).unwrap_or_else(|| panic!("no note for {rows} rows"));
        assert!((got - frozen).abs() <= 1e-5, "gapped {mechanism}: live {got} vs {frozen}");
    }
    let column = posterior.effect_column().expect("effect column");
    let (mean, sd) = (posterior.summaries.mean[column], posterior.summaries.sd[column]);
    let (want_mean, want_sd) = (bayes["mean"].as_f64().unwrap(), bayes["sd"].as_f64().unwrap());
    let root = (DRAWS as f64).sqrt();
    assert!((mean - want_mean).abs() < 5.0 * want_sd / root, "mean {mean} vs {want_mean}");
    assert!((sd / want_sd - 1.0).abs() < 10.0 / root, "sd {sd} vs {want_sd}");
}

#[test]
fn temporal_sequential_bayesian_level_has_the_exact_mechanism_moments_and_a_root_mean_shift() {
    let reference = reference();
    let bayes = &reference["bayes"]["level"];
    let (_, posterior) = bayes_run(&reference, "level_hard");
    let column = posterior.effect_column().expect("effect column");
    let (mean, sd) = (posterior.summaries.mean[column], posterior.summaries.sd[column]);
    let want_sd = bayes["sd"].as_f64().unwrap();
    let root = (DRAWS as f64).sqrt();
    // The mechanism draws and the independent root-mean shift add in variance.
    assert!(
        (mean - bayes["mean"].as_f64().unwrap()).abs() < 5.0 * want_sd / root,
        "level mean {mean} vs {}",
        bayes["mean"]
    );
    assert!((sd / want_sd - 1.0).abs() < 10.0 / root, "level sd {sd} vs {want_sd}");
    assert!(bayes["root_mean_se"].as_f64().unwrap() > 0.0);
    assert!(want_sd > bayes["mechanism_sd"].as_f64().unwrap());
}

#[test]
fn temporal_sequential_declared_options_are_read() {
    let reference = reference();
    let data = series(&reference["data"]);
    let unit = point_case(&reference, "contrast_unit");
    // bootstrap_replicates: zero publishes no SE, a positive count publishes one.
    let (none, _) = estimate_case(unit, &data, 0, None);
    let (some, _) = estimate_case(unit, &data, 6, None);
    assert!(none.se_bootstrap().is_none() && some.se_bootstrap().is_some());
    assert_eq!(some.uncertainty.bootstrap.replicates_ok, Some(6));
    // bayesian: a Bayesian estimator turns the estimate into a posterior.
    let (_, posterior) = estimate_case(unit, &data, 0, Some(&bayes_estimator().with_n_draws(400)));
    assert!(posterior.is_some());
    // status: an unidentified design is refused.
    let query = window_query(1.0, 0.0);
    let identified = identify_contrast(&query);
    let refused = estimate_sustained_window(
        &identified.problem(&data),
        &query,
        SequentialInference {
            status: IdentificationStatus::NotIdentified,
            assumptions: AssumptionSet::new(),
            bootstrap_replicates: 0,
            bayesian: None,
        },
        &ctx(),
    );
    assert!(
        refusal(refused).contains("requires an identified unfolded multi-step contrast"),
        "an unidentified status must be refused"
    );
    // target population: only AllObserved is estimated.
    let restricted = window_query(1.0, 0.0).with_target_population(TargetPopulation::Treated);
    assert!(
        refusal(estimate_sustained_window(
            &identified.problem(&data),
            &restricted,
            inference(&identified, 0, None),
            &ctx(),
        ))
        .contains("AllObserved")
    );
    // policy: a single-step window is not a sequential contrast.
    let single = window_query(1.0, 0.0).with_policy(TemporalPolicy::sustained(0, 0));
    assert!(
        refusal(estimate_sustained_window(
            &identified.problem(&data),
            &single,
            inference(&identified, 0, None),
            &ctx(),
        ))
        .contains("identified unfolded multi-step contrast")
    );
    // backend: composed HMC is refused rather than approximated, and so is a non-Gaussian likelihood.
    let hmc = BayesianGComputationAte { backend: BayesianBackendKind::Hmc, ..bayes_estimator() };
    assert!(
        refusal(estimate_sustained_window(
            &identified.problem(&data),
            &query,
            inference(&identified, 0, Some(&hmc)),
            &ctx(),
        ))
        .contains("composed sustained HMC requires derived-contrast chain diagnostics")
    );
    // assumptions: the premises passed in are carried onto the estimate.
    let carried = antecedent_estimate::temporal_sequential::SequentialMechanismOverlay::mean_target_assumption();
    let mut assumptions = AssumptionSet::new();
    assumptions.push(carried.clone());
    let (effect, _) = estimate_sustained_window(
        &identified.problem(&data),
        &query,
        SequentialInference {
            status: identified.status,
            assumptions,
            bootstrap_replicates: 0,
            bayesian: None,
        },
        &ctx(),
    )
    .unwrap();
    assert!(effect.provenance.assumptions.entries.contains(&carried));
}

#[test]
fn temporal_sequential_effect_modifiers_enter_as_interactions_and_are_averaged() {
    let reference = reference();
    let data = series(&reference["data"]);
    let modifier = &reference["modifier"];
    let query = window_query(1.0, 0.0).with_effect_modifiers(vec![VariableId::from_raw(
        modifier["modifiers"][0].as_u64().unwrap() as u32,
    )]);
    let identified = identify_contrast(&query);
    let (effect, _) = estimate_sustained_window(
        &identified.problem(&data),
        &query,
        inference(&identified, 0, None),
        &ctx(),
    )
    .unwrap();
    close("modified contrast", effect.ate, modifier["value"].as_f64().unwrap(), 1e-9);
    // The modifier is read: the contrast moves off the main-effects one.
    assert!((effect.ate - modifier["unmodified"].as_f64().unwrap()).abs() > 1e-2);
}

#[test]
fn temporal_sequential_missing_time_points_drop_whole_windows_and_resample_the_original_axis() {
    let reference = reference();
    let gaps = &reference["gaps"];
    let data = series_with(&reference["data"], &gaps["missing"], None);
    let query = window_query(1.0, 0.0);
    let identified = identify_contrast(&query);
    // Without the gap switch a series with a missing value is refused.
    assert!(
        estimate_sustained_window(
            &identified.problem(&data),
            &query,
            inference(&identified, 0, None),
            &ctx(),
        )
        .is_err()
    );
    let (effect, _, selection) = estimate_sustained_window_with_gaps(
        &identified.problem(&data),
        &query,
        inference(&identified, REPLICATES, None),
        &ctx(),
        None,
    )
    .unwrap();
    close("gapped contrast", effect.ate, gaps["value"].as_f64().unwrap(), 1e-10);
    let selection = selection.expect("gap selection");
    let complete: Vec<bool> =
        gaps["complete"].as_array().unwrap().iter().map(|c| c.as_bool().unwrap()).collect();
    assert_eq!(selection.complete, complete);
    assert_eq!(selection.first_time, 3);
    // The resampled axis is the original windows; the block is the structural rule with no
    // kernel-bias widening, and the replicates are exact refits on the intact rows each block covers.
    let block = effect.uncertainty.block.resampling.expect("block geometry");
    assert_eq!(block.rows, gaps["windows"].as_u64().unwrap() as usize);
    assert_eq!(block.block_length, gaps["block_length"].as_u64().unwrap() as usize);
    assert_eq!(block.kernel_bias, 1.0);
    let positions = regenerated_rows(block.rows, block.block_length, REPLICATES);
    let frozen: Vec<Vec<usize>> = inputs()["rows"]["gapped"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m.as_array().unwrap().iter().map(|r| r.as_u64().unwrap() as usize).collect())
        .collect();
    assert_eq!(positions, frozen, "regenerated positions differ from the frozen copy");
    let ok = gaps["replicates_ok"].as_u64().unwrap() as u32;
    assert_eq!(effect.uncertainty.bootstrap.replicates_ok, Some(ok));
    let raw = gaps["raw_sd"].as_f64().expect("enough replicates");
    close(
        "gapped se",
        effect.se_bootstrap().expect("SE"),
        raw * gaps["fixed_b"].as_f64().unwrap(),
        1e-9,
    );
}

#[test]
fn temporal_sequential_sampling_weights_make_every_mechanism_a_weighted_fit() {
    let reference = reference();
    let weights = &reference["weights"];
    let data = series_with(&reference["data"], &Json::Null, Some(floats(&weights["weights"])));
    let query = window_query(1.0, 0.0);
    let identified = identify_contrast(&query);
    let (effect, _) = estimate_sustained_window(
        &identified.problem(&data),
        &query,
        inference(&identified, 0, None),
        &ctx(),
    )
    .unwrap();
    close("weighted contrast", effect.ate, weights["value"].as_f64().unwrap(), 1e-9);
    assert!((effect.ate - weights["unweighted"].as_f64().unwrap()).abs() > 1e-2);
    // A level and a composed Bayesian posterior have no weighted construction: refused, not ignored.
    let level = point_case(&reference, "level_hard");
    let overlays = overlays_of(level);
    let level_identified = identify_schedule(&schedule_of(&overlays));
    assert!(
        estimate_sequence_mechanisms(
            &level_identified.problem(&data),
            &SequenceTarget {
                outcome: VariableId::from_raw(Y),
                outcome_offset: 0,
                overlays: &overlays,
            },
            inference(&level_identified, 0, None),
            &ctx(),
        )
        .is_err()
    );
    assert!(
        estimate_sustained_window(
            &identified.problem(&data),
            &query,
            inference(&identified, 0, Some(&bayes_estimator().with_n_draws(200))),
            &ctx(),
        )
        .is_err()
    );
}

#[test]
fn temporal_sequential_overlay_validation_refuses_and_discloses() {
    let reference = reference();
    let data = series(&reference["data"]);
    let run = |overlays: &[SequentialMechanismOverlay], bayes: Option<&BayesianGComputationAte>| {
        let identified =
            identify_schedule(&schedule_of(&overlays_of(point_case(&reference, "level_hard"))));
        estimate_sequence_mechanisms(
            &identified.problem(&data),
            &SequenceTarget { outcome: VariableId::from_raw(Y), outcome_offset: 0, overlays },
            inference(&identified, 0, bayes),
            &ctx(),
        )
    };
    let base = overlays_of(point_case(&reference, "level_hard"));
    assert!(run(&base, None).is_ok());
    assert!(
        refusal(run(&[], None)).contains("requires at least one step"),
        "an empty sequence is refused"
    );
    let duplicated = [base[0], base[0]];
    assert!(run(&duplicated, None).is_err(), "one (variable, time) assigned twice is refused");
    for broken in [
        SequentialMechanismOverlay { multiplier: f64::NAN, ..base[0] },
        SequentialMechanismOverlay {
            node: SequentialNodeOverlay { shift: f64::INFINITY, ..base[0].node },
            ..base[0]
        },
        SequentialMechanismOverlay { bounds: Some((1.0, 0.0)), ..base[0] },
    ] {
        assert!(
            refusal(run(&[broken, base[1], base[2]], None))
                .contains("sequential overlays require finite levels and shifts"),
            "{broken:?}"
        );
    }
    let hmc = BayesianGComputationAte { backend: BayesianBackendKind::Hmc, ..bayes_estimator() };
    assert!(
        refusal(run(&base, Some(&hmc)))
            .contains("composed sequence HMC requires derived-contrast chain diagnostics"),
        "composed HMC is refused for a level too"
    );
    // A bounded shift discloses the population-mean-targeting restriction; an unbounded level does not.
    let bounded = overlays_of(point_case(&reference, "level_bounded_upper"));
    let restriction = SequentialMechanismOverlay::mean_target_assumption();
    let identified = identify_schedule(&schedule_of(&bounded));
    let with_bounds = estimate_sequence_mechanisms(
        &identified.problem(&data),
        &SequenceTarget { outcome: VariableId::from_raw(Y), outcome_offset: 0, overlays: &bounded },
        inference(&identified, 0, None),
        &ctx(),
    )
    .unwrap()
    .0;
    assert!(with_bounds.provenance.assumptions.entries.contains(&restriction));
    let plain = run(&base, None).unwrap().0;
    assert!(!plain.provenance.assumptions.entries.contains(&restriction));
}

/// Writes the inputs the reference reads: the replicate row maps and the tempering factors.
/// Run with `--ignored --nocapture` and save the printed JSON to the inputs file.
#[test]
#[ignore = "regenerates oracle_data/temporal_sequential_reference_inputs.json"]
fn temporal_sequential_dump_reference_inputs() {
    let reference = reference();
    let mut rows = serde_json::Map::new();
    for (tag, name, data_key) in bootstrap_cases() {
        let data = series(&reference[data_key]);
        let (effect, _) = estimate_case(point_case(&reference, name), &data, REPLICATES, None);
        let block = effect.uncertainty.block.resampling.unwrap();
        rows.insert(
            tag.into(),
            serde_json::to_value(regenerated_rows(block.rows, block.block_length, REPLICATES))
                .unwrap(),
        );
    }
    let gaps = &reference["gaps"];
    let gapped = series_with(&reference["data"], &gaps["missing"], None);
    let query = window_query(1.0, 0.0);
    let identified = identify_contrast(&query);
    let (effect, _, _) = estimate_sustained_window_with_gaps(
        &identified.problem(&gapped),
        &query,
        inference(&identified, REPLICATES, None),
        &ctx(),
        None,
    )
    .unwrap();
    let block = effect.uncertainty.block.resampling.unwrap();
    rows.insert(
        "gapped".into(),
        serde_json::to_value(regenerated_rows(block.rows, block.block_length, REPLICATES)).unwrap(),
    );
    let mut kappa = serde_json::Map::new();
    for (kind, name) in [("contrast", "contrast_unit"), ("level", "level_hard")] {
        let (_, posterior) = bayes_run(&reference, name);
        let live = kappas_by_rows(&posterior.diagnostics.notes);
        let mut by_mechanism = serde_json::Map::new();
        for (mechanism, count) in [("A", 24_u64), ("Y", 23), ("M", 22)] {
            by_mechanism.insert(mechanism.into(), live[&count].into());
        }
        kappa.insert(kind.into(), by_mechanism.into());
    }
    let estimator = bayes_estimator();
    let (_, posterior, _) = estimate_sustained_window_with_gaps(
        &identified.problem(&gapped),
        &query,
        inference(&identified, 0, Some(&estimator)),
        &ctx(),
        None,
    )
    .unwrap();
    let live = kappas_by_rows(&posterior.expect("posterior").diagnostics.notes);
    // Intact rows of the A, M and Y mechanisms after the three missing cells.
    let mut by_mechanism = serde_json::Map::new();
    for (mechanism, count) in [("A", 23_u64), ("M", 21), ("Y", 18)] {
        by_mechanism.insert(mechanism.into(), live[&count].into());
    }
    kappa.insert("gapped".into(), by_mechanism.into());
    println!("{}", serde_json::json!({ "rows": rows, "kappa": kappa }));
}
