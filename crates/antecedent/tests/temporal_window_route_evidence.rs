//! Known-truth evidence for the pulse, sustained, conditional and mediation analyses that read the
//! unrolled window of a stationary template (a `TemporalAdmg` supplied, accepted or drawn from a
//! graph posterior, and a `TemporalPag`).
//!
//! Every expected value is read from `conformance/estimate/temporal_window_truth`, where an
//! independent generator enumerates the noise cells of the unrolled structural model with exact
//! rational arithmetic: the effects of an intervention at a period of the window on an outcome
//! at a later period (no identification formula is involved), and a series whose consecutive
//! windows have exactly the law of the process. The window plug-in therefore reproduces the
//! enumerated effects to rounding error, and the Bayesian bootstrap to its declared band.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::too_many_lines,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "one loop covers every licensed coordinate; fixture states are small exact indices"
)]

use antecedent::analysis::RetainedOperation;
use antecedent::{
    AcceptedGraph, BayesianConfig, CellStatus, InferenceMode, RefuteSuite, Study, StudyResult,
};
use antecedent_core::{
    CausalQuery, ExecutionContext, IdentificationStatus, Lag, MediationContrast, MediationQuery,
    ResponseValue, TemporalEffectQuery, TemporalPolicy, VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_discovery::adjacency_mask::set_edge;
use antecedent_discovery::{
    AdjacencyMask, GraphPosterior, GraphPosteriorAtomKind, LagMask, SamplingSummary,
    TemporalAdmgAtom, with_lagged_edge,
};
use antecedent_graph::{TemporalAdmg, TemporalPag};
use antecedent_prob::InferenceDiagnostics;
use serde_json::Value;

const ALPHABET: &str = "0123456789abcdefghijklmnopqrstuv";

/// Posterior-mean tolerance of the Bayesian bootstrap around the enumerated truth.
const BAYESIAN_TOLERANCE: f64 = 3.0e-2;

fn pin() -> Value {
    serde_json::from_str(include_str!(
        "../../../conformance/estimate/temporal_window_truth/expected.json"
    ))
    .unwrap()
}

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn bayesian() -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(100))
}

fn inferences() -> [(InferenceMode, f64); 2] {
    [(InferenceMode::Frequentist, 1.0e-9), (bayesian(), BAYESIAN_TOLERANCE)]
}

fn states(text: &str) -> Vec<u32> {
    text.chars().map(|c| ALPHABET.find(c).unwrap() as u32).collect()
}

fn bit(states: &[u32], position: u32) -> Vec<f64> {
    states.iter().map(|state| f64::from((state >> position) & 1)).collect()
}

// ---------------------------------------------------------------------------
// The contrast process: columns x, y, w.
// ---------------------------------------------------------------------------

const X: u32 = 0;
const Y: u32 = 1;
const W: u32 = 2;

fn contrast_frame(states: &[u32]) -> TimeSeriesData {
    let (x, y, w) = (bit(states, 0), bit(states, 1), bit(states, 2));
    TimeSeriesData::from_f64_columns(
        [("x", x.as_slice()), ("y", y.as_slice()), ("w", w.as_slice())],
        1,
    )
    .unwrap()
}

/// The series whose windows are exactly the law of the process.
fn contrast_series() -> TimeSeriesData {
    contrast_frame(&states(pin()["contrast"]["series"].as_str().unwrap()))
}

/// The process followed by the process whose treatment has no effect: stationary in each half
/// with one support, the contrast an exact zero in the second.
fn broken_series() -> TimeSeriesData {
    let pin = pin();
    let mut joined = states(pin["contrast"]["series"].as_str().unwrap());
    joined.extend(states(pin["contrast"]["planted_series"].as_str().unwrap()));
    contrast_frame(&joined)
}

const KINDS: [&str; 4] = ["pulse", "sustained", "conditional_pulse", "conditional_sustained"];

fn contrast_truth(kind: &str) -> f64 {
    pin()["contrast"]["truth"][kind].as_f64().unwrap()
}

fn planted_truth(kind: &str) -> f64 {
    pin()["contrast"]["planted_truth"][kind].as_f64().unwrap()
}

/// The pulse at the first treated period read at the next period, the sustained policy on that
/// period and the one after it, and each conditioned on `w` at the first treated period.
fn contrast_query(kind: &str) -> TemporalEffectQuery {
    let base = TemporalEffectQuery::pulse(v(X), v(Y), 1.0).with_horizon_steps(2);
    match kind {
        "pulse" => base,
        "sustained" => base.with_policy(TemporalPolicy::sustained(0, 1)),
        "conditional_pulse" => base.with_effect_modifiers([v(W)]),
        "conditional_sustained" => {
            base.with_policy(TemporalPolicy::sustained(0, 1)).with_effect_modifiers([v(W)])
        }
        other => panic!("unknown contrast kind {other}"),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Shape {
    /// `x -> y`, `w -> y` within a period and `y(t-1) -> y(t)`.
    Generating,
    /// The generating template with a declared latent cause of `w(t-1)` and `y(t)`.
    DeclaredLatent,
    /// The generating template without `x -> y`: the treatment reaches nothing.
    NoEffect,
    /// The generating template with `x(t-1) -> y(t)` and `x(t-1) <-> y(t)`: a bow arc.
    Bow,
    /// The generating template with `x -> w`: the modifier is a consequence of the treatment.
    ModifierReached,
}

fn contrast_template(shape: Shape) -> TemporalAdmg {
    let mut graph = TemporalAdmg::empty();
    let y1 = graph.add_lagged(v(Y), Lag::from_raw(1)).unwrap();
    let x0 = graph.add_lagged(v(X), Lag::CONTEMPORANEOUS).unwrap();
    let w0 = graph.add_lagged(v(W), Lag::CONTEMPORANEOUS).unwrap();
    let y0 = graph.add_lagged(v(Y), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(y1, y0).unwrap();
    graph.insert_directed(w0, y0).unwrap();
    if shape != Shape::NoEffect {
        graph.insert_directed(x0, y0).unwrap();
    }
    match shape {
        Shape::DeclaredLatent => {
            let w1 = graph.add_lagged(v(W), Lag::from_raw(1)).unwrap();
            graph.insert_bidirected(w1, y0).unwrap();
        }
        Shape::Bow => {
            let x1 = graph.add_lagged(v(X), Lag::from_raw(1)).unwrap();
            graph.insert_directed(x1, y0).unwrap();
            graph.insert_bidirected(x1, y0).unwrap();
        }
        Shape::ModifierReached => graph.insert_directed(x0, w0).unwrap(),
        Shape::Generating | Shape::NoEffect => {}
    }
    graph
}

fn contrast_study(
    data: &TimeSeriesData,
    structure: &str,
    template: TemporalAdmg,
    query: TemporalEffectQuery,
    inference: InferenceMode,
    suite: RefuteSuite,
) -> Study {
    let base = Study::series(data.clone());
    let base = if structure == "accepted" {
        base.graph(AcceptedGraph::from(template))
    } else {
        base.graph(template)
    };
    base.temporal_query(query)
        .inference(inference)
        .refute(suite)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
}

fn discloses(result: &StudyResult, code: &str) -> bool {
    result.diagnostics.iter().any(|diagnostic| diagnostic.code.as_ref() == code)
}

fn refuter_ids(result: &StudyResult) -> Vec<String> {
    result.refutations.iter().map(|report| report.refuter.to_string()).collect()
}

fn report<'a>(result: &'a StudyResult, id: &str) -> &'a antecedent_validate::RefutationReport {
    result
        .refutations
        .iter()
        .find(|report| report.refuter.as_ref() == id)
        .unwrap_or_else(|| panic!("no report {id} in {:?}", refuter_ids(result)))
}

#[test]
fn the_fixture_enumerates_distinct_pulse_and_sustained_truths_and_a_bow_that_is_not_identified() {
    let pin = pin();
    let truth = &pin["contrast"]["truth"];
    assert!((truth["pulse"].as_f64().unwrap() - 0.125).abs() < 1e-15);
    assert!((truth["sustained"].as_f64().unwrap() - 0.375).abs() < 1e-15);
    // Two structural models over one observed law whose effects differ: the effect is not a
    // function of the observed law, so a window that holds a bow arc has no identified value.
    let bow = &pin["bow_witness"];
    assert_eq!(bow["observed_law_with_effect"], bow["observed_law_without_effect"]);
    assert_ne!(bow["effect_with"], bow["effect_without"]);
}

// ---------------------------------------------------------------------------
// A TemporalAdmg supplied or accepted: pulse, sustained and the conditional contrasts.
// ---------------------------------------------------------------------------

#[test]
fn temporal_admg_window_contrasts_match_the_enumerated_truth_on_both_inferences() {
    let _inferences = ("Frequentist", "Bayesian");
    let data = contrast_series();
    let ctx = ExecutionContext::for_tests(41_001);
    let mut statuses = Vec::new();
    for structure in ["explicit", "accepted"] {
        for kind in KINDS {
            let want = contrast_truth(kind);
            for (inference, tolerance) in inferences() {
                for shape in [Shape::Generating, Shape::DeclaredLatent] {
                    // A declared latent cause of the modifier is read through the conditional
                    // identification as well, but only the unconditional contrasts pin it.
                    if shape == Shape::DeclaredLatent && kind.starts_with("conditional") {
                        continue;
                    }
                    let built = contrast_study(
                        &data,
                        structure,
                        contrast_template(shape),
                        contrast_query(kind),
                        inference.clone(),
                        RefuteSuite::None,
                    );
                    statuses.push(built.support_status());
                    let result = built.run(&ctx).unwrap_or_else(|error| {
                        panic!("{structure} {kind} {inference:?}: {error}")
                    });
                    assert!(
                        (result.effect() - want).abs() < tolerance,
                        "{structure} {kind} {inference:?}: {} vs {want}",
                        result.effect()
                    );
                    assert!(discloses(&result, "estimate.temporal_admg.window"));
                    assert!(result.refutations.is_empty());
                }
            }
        }
    }
    // A treatment that reaches nothing contrasts to exactly zero, whatever the series.
    for kind in KINDS {
        let result = contrast_study(
            &data,
            "explicit",
            contrast_template(Shape::NoEffect),
            contrast_query(kind),
            InferenceMode::Frequentist,
            RefuteSuite::None,
        )
        .run(&ctx)
        .unwrap();
        assert!(result.effect().abs() < 1e-9, "{kind}: {}", result.effect());
    }
    assert!(statuses.iter().all(|status| *status == Some(CellStatus::Licensed)), "{statuses:?}");
}

#[test]
fn temporal_admg_window_contrasts_run_from_the_retained_plan_without_the_builder() {
    let _inferences = ("Frequentist", "Bayesian");
    let data = contrast_series();
    let ctx = ExecutionContext::for_tests(41_002);
    for structure in ["explicit", "accepted"] {
        for kind in KINDS {
            let want = contrast_truth(kind);
            for (inference, tolerance) in inferences() {
                let builder = contrast_study(
                    &data,
                    structure,
                    contrast_template(Shape::Generating),
                    contrast_query(kind),
                    inference.clone(),
                    RefuteSuite::None,
                );
                let prepared = builder.prepare(&ctx).unwrap();
                assert!(matches!(
                    prepared.retained_operation(),
                    RetainedOperation::TemporalAdmgEffect
                ));
                drop(builder);
                let result = prepared.estimate_series(&data, &ctx).unwrap();
                assert!(
                    (result.effect() - want).abs() < tolerance,
                    "{structure} {kind} {inference:?}: {} vs {want}",
                    result.effect()
                );
            }
        }
    }
}

#[test]
fn temporal_admg_window_contrasts_refuse_a_hedge_and_a_modifier_the_treatment_reaches() {
    let _inferences = ("Frequentist", "Bayesian");
    let data = contrast_series();
    let ctx = ExecutionContext::for_tests(41_003);
    for kind in ["pulse", "sustained", "conditional_pulse", "conditional_sustained"] {
        let built = contrast_study(
            &data,
            "explicit",
            contrast_template(Shape::Bow),
            contrast_query(kind),
            InferenceMode::Frequentist,
            RefuteSuite::None,
        );
        for refusal in
            [built.run(&ctx).unwrap_err().to_string(), built.prepare(&ctx).unwrap_err().to_string()]
        {
            assert!(refusal.contains("effect_not_identified"), "{kind}: {refusal}");
            assert!(refusal.contains("next attempts"), "{kind}: {refusal}");
        }
    }
    let refusal = contrast_study(
        &data,
        "explicit",
        contrast_template(Shape::ModifierReached),
        contrast_query("conditional_pulse"),
        InferenceMode::Frequentist,
        RefuteSuite::None,
    )
    .run(&ctx)
    .unwrap_err()
    .to_string();
    assert!(refusal.contains("descendant") && refusal.contains("next attempt"), "{refusal}");
}

#[test]
fn temporal_admg_window_validation_reports_the_known_refits_and_detects_a_planted_break() {
    let _inferences = ("Frequentist", "Bayesian");
    let clean = contrast_series();
    let ctx = ExecutionContext::for_tests(41_004);
    let mut statuses = Vec::new();
    for structure in ["explicit", "accepted"] {
        for kind in KINDS {
            // The accepted structure reads the same plan, and the Bayesian form refits the same
            // window table: one contrast pins each of those cells.
            if structure == "accepted" && kind != "pulse" {
                continue;
            }
            let want = contrast_truth(kind);
            for (inference, tolerance) in inferences() {
                if matches!(inference, InferenceMode::Bayesian(_)) && kind != "pulse" {
                    continue;
                }
                for (suite, expected) in [
                    (RefuteSuite::Cheap, vec!["window.stretch_80", "window.placebo_treatment"]),
                    (
                        RefuteSuite::Full,
                        vec![
                            "window.stretch_80",
                            "window.stretch_50",
                            "window.placebo_treatment",
                            "window.split_half",
                        ],
                    ),
                ] {
                    let built = contrast_study(
                        &clean,
                        structure,
                        contrast_template(Shape::Generating),
                        contrast_query(kind),
                        inference.clone(),
                        suite,
                    );
                    statuses.push(built.support_status());
                    let result = built.run(&ctx).unwrap_or_else(|error| {
                        panic!("{structure} {kind} {inference:?} {suite:?}: {error}")
                    });
                    assert_eq!(refuter_ids(&result), expected, "{kind} {suite:?}");
                    assert!((result.effect() - want).abs() < tolerance);
                    // A contiguous stretch of a stationary exact-law series refits to its
                    // own contrast, near the truth.
                    let stretch = report(&result, "window.stretch_80");
                    assert!((stretch.refuted_ate - want).abs() < 0.1, "{kind}: {stretch:?}");
                    // The rotated treatment keeps its law and loses its link to the outcome.
                    let placebo = report(&result, "window.placebo_treatment");
                    assert_eq!(placebo.replicates, 20);
                    assert!(placebo.refuted_ate.abs() < 0.1, "{kind}: {placebo:?}");
                    assert!((0.0..=1.0).contains(&placebo.comparison));
                    if suite == RefuteSuite::Full {
                        // Each half of a stationary series carries the process's contrast.
                        let split = report(&result, "window.split_half");
                        assert!(split.informative, "{kind}: {split:?}");
                        assert!((split.original_ate - want).abs() < 0.1, "{kind}: {split:?}");
                        assert!((split.refuted_ate - want).abs() < 0.1, "{kind}: {split:?}");
                    }
                }
            }
        }
    }
    // The planted violation: the effect of the treatment vanishes in the second half of the
    // series. Each half is the exact law of its own process (the same support, the same
    // stationary law), so the halves carry the contrast and an exact zero; the split-half
    // stationarity check detects the break.
    let broken = broken_series();
    for kind in KINDS {
        let result = contrast_study(
            &broken,
            "explicit",
            contrast_template(Shape::Generating),
            contrast_query(kind),
            InferenceMode::Frequentist,
            RefuteSuite::Full,
        )
        .run(&ctx)
        .unwrap();
        let split = report(&result, "window.split_half");
        assert!(split.informative && !split.passed, "{kind}: {split:?}");
        assert!(split.comparison < 0.01, "{kind}: {}", split.comparison);
        // The two junction windows are foreign rows of the first half, and a sustained policy
        // reads sparse conditioning cells, so the first half is only near its regime's contrast.
        assert!((split.original_ate - contrast_truth(kind)).abs() < 0.15, "{kind}: {split:?}");
        assert!((split.refuted_ate - planted_truth(kind)).abs() < 0.05, "{kind}: {split:?}");
    }
    assert!(statuses.iter().all(|status| *status == Some(CellStatus::Licensed)), "{statuses:?}");
}

// ---------------------------------------------------------------------------
// A graph posterior over lagged TemporalAdmg atoms: the window contrasts.
// ---------------------------------------------------------------------------

fn atom_posterior(weights: &[f64], atoms: &[TemporalAdmgAtom], n_vars: usize) -> GraphPosterior {
    GraphPosterior::from_atoms(
        n_vars,
        weights.to_vec(),
        atoms.iter().map(|atom| atom.contemporaneous.clone()),
        vec![0.0; n_vars * n_vars],
        vec![0.0; n_vars * n_vars],
        SamplingSummary {
            ess: 1.0,
            diagnostics: InferenceDiagnostics::analytic("temporal_window_truth"),
            rejected_invalid: 0,
        },
    )
    .unwrap()
    .with_atom_kind(GraphPosteriorAtomKind::TemporalAdmg)
    .with_lagged_marginals(1, vec![0.0; n_vars * n_vars])
    .unwrap()
    .with_lag_masks(atoms.iter().map(|atom| atom.lagged_directed.clone()).collect::<Vec<_>>())
    .unwrap()
    .with_lag_bidirected_masks(
        atoms.iter().map(|atom| atom.lagged_bidirected.clone()).collect::<Vec<_>>(),
    )
    .unwrap()
}

fn lagged(entries: &[(u32, usize, usize)], n_vars: usize) -> LagMask {
    entries.iter().fold(LagMask::EMPTY, |word, &(lag, from, to)| {
        with_lagged_edge(word, n_vars, lag, from, to, true).unwrap()
    })
}

fn contemporaneous(edges: &[(usize, usize)], n_vars: usize) -> AdjacencyMask {
    edges
        .iter()
        .fold(AdjacencyMask::EMPTY, |mask, &(from, to)| set_edge(mask, n_vars, from, to, true))
}

const XI: usize = 0;
const YI: usize = 1;
const WI: usize = 2;

/// The generating template.
fn generating_atom() -> TemporalAdmgAtom {
    TemporalAdmgAtom {
        contemporaneous: contemporaneous(&[(XI, YI), (WI, YI)], 3),
        lagged_directed: lagged(&[(1, YI, YI)], 3),
        lagged_bidirected: LagMask::EMPTY,
    }
}

/// The treatment reaches nothing: the contrast of this template is exactly zero.
fn no_effect_atom() -> TemporalAdmgAtom {
    TemporalAdmgAtom {
        contemporaneous: contemporaneous(&[(WI, YI)], 3),
        lagged_directed: lagged(&[(1, YI, YI)], 3),
        lagged_bidirected: LagMask::EMPTY,
    }
}

/// `x(t-1) -> y(t)` with `x(t-1) <-> y(t)`: a bow arc, so the window holds a hedge.
fn bow_atom() -> TemporalAdmgAtom {
    TemporalAdmgAtom {
        contemporaneous: contemporaneous(&[(XI, YI), (WI, YI)], 3),
        lagged_directed: lagged(&[(1, YI, YI), (1, XI, YI)], 3),
        lagged_bidirected: lagged(&[(1, XI, YI)], 3),
    }
}

fn scalar(value: Option<&ResponseValue>) -> Option<f64> {
    match value? {
        ResponseValue::Scalar(value) => Some(*value),
        _ => None,
    }
}

fn posterior_study(
    data: &TimeSeriesData,
    posterior: GraphPosterior,
    query: TemporalEffectQuery,
    inference: InferenceMode,
    suite: RefuteSuite,
) -> Study {
    Study::series(data.clone())
        .graph_posterior(posterior)
        .temporal_query(query)
        .inference(inference)
        .refute(suite)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
}

#[test]
fn temporal_admg_window_posterior_mixes_by_posterior_weight_and_accounts_for_unidentified_mass() {
    let _inferences = ("Frequentist", "Bayesian");
    let data = contrast_series();
    let ctx = ExecutionContext::for_tests(41_005);
    let weights = [0.5, 0.3, 0.2];
    let mut statuses = Vec::new();
    for kind in KINDS {
        let truth = contrast_truth(kind);
        // The treatment-free atom contributes an exact zero at its own weight; the bow atom's
        // mass is unidentified mass and is not redistributed.
        let expected = (0.5 * truth + 0.3 * 0.0) / 0.8;
        for (inference, tolerance) in inferences() {
            let gp =
                atom_posterior(&weights, &[generating_atom(), no_effect_atom(), bow_atom()], 3);
            assert!(gp.is_lagged_temporal_admg());
            let built = posterior_study(
                &data,
                gp,
                contrast_query(kind),
                inference.clone(),
                RefuteSuite::None,
            );
            statuses.push(built.support_status());
            let result =
                built.run(&ctx).unwrap_or_else(|error| panic!("{kind} {inference:?}: {error}"));
            assert_eq!(result.logical_plan.identifier.as_deref(), Some("general.id"));
            assert_eq!(result.logical_plan.estimator.as_deref(), Some("functional.effect"));
            let mixture = result.structural_response.as_ref().expect("the atom mixture");
            assert_eq!(mixture.atoms.len(), 3);
            assert!(
                (mixture.unidentified_mass - 0.2).abs() < 1e-12,
                "{kind}: {}",
                mixture.unidentified_mass
            );
            assert!(mixture.unevaluable_mass.abs() < 1e-12);
            let by_key =
                |key: u64| mixture.atoms.iter().find(|atom| atom.graph_key == key).unwrap();
            assert!((by_key(0).weight - 0.5).abs() < 1e-12);
            assert!((by_key(1).weight - 0.3).abs() < 1e-12);
            if matches!(inference, InferenceMode::Frequentist) {
                let atom = scalar(by_key(0).value.as_ref()).expect("the generating atom");
                assert!((atom - truth).abs() < 1e-9, "{kind}: {atom} vs {truth}");
                let none = scalar(by_key(1).value.as_ref()).expect("the treatment-free atom");
                assert!(none.abs() < 1e-9, "{kind}: {none}");
            }
            assert!(scalar(by_key(2).value.as_ref()).is_none(), "the bow atom has no value");
            assert_eq!(by_key(2).status, IdentificationStatus::NotIdentified);
            let effect = result.estimate().as_effect().unwrap().ate;
            assert!((effect - expected).abs() < tolerance, "{kind} {inference:?}: {effect}");
            assert_eq!(result.identification.status, IdentificationStatus::GraphDependent);
            let window = result
                .diagnostics
                .iter()
                .find(|d| d.code.as_ref() == "estimate.graph_posterior.temporal_admg_window")
                .expect("window disclosure");
            assert!(window.message.contains("3 periods"), "{}", window.message);
        }
    }
    assert!(statuses.iter().all(|status| *status == Some(CellStatus::Licensed)), "{statuses:?}");
}

#[test]
fn temporal_admg_window_posterior_runs_from_the_retained_plan_without_the_builder() {
    let _inferences = ("Frequentist", "Bayesian");
    let data = contrast_series();
    let ctx = ExecutionContext::for_tests(41_006);
    for kind in KINDS {
        let truth = contrast_truth(kind);
        let expected = 0.5 * truth / 0.8;
        for (inference, tolerance) in inferences() {
            let gp = atom_posterior(
                &[0.5, 0.3, 0.2],
                &[generating_atom(), no_effect_atom(), bow_atom()],
                3,
            );
            let builder = posterior_study(
                &data,
                gp,
                contrast_query(kind),
                inference.clone(),
                RefuteSuite::None,
            );
            let prepared = builder.prepare(&ctx).unwrap();
            assert_ne!(prepared.retained_operation().name(), "fresh_run");
            drop(builder);
            let result = prepared.estimate_retained(&ctx).unwrap();
            let effect = result.estimate().as_effect().unwrap().ate;
            assert!((effect - expected).abs() < tolerance, "{kind} {inference:?}: {effect}");
            let mixture = result.structural_response.as_ref().unwrap();
            assert!((mixture.unidentified_mass - 0.2).abs() < 1e-12);
        }
    }
}

#[test]
fn temporal_admg_window_posterior_validation_mixes_the_per_atom_window_suite() {
    let _inferences = ("Frequentist", "Bayesian");
    let clean = contrast_series();
    let ctx = ExecutionContext::for_tests(41_007);
    let mut statuses = Vec::new();
    for kind in KINDS {
        let truth = contrast_truth(kind);
        let expected_mixture = 0.6 * truth / 1.0;
        for (inference, tolerance) in inferences() {
            if matches!(inference, InferenceMode::Bayesian(_)) && kind != "pulse" {
                continue;
            }
            let none = posterior_study(
                &clean,
                atom_posterior(&[0.6, 0.4], &[generating_atom(), no_effect_atom()], 3),
                contrast_query(kind),
                inference.clone(),
                RefuteSuite::None,
            )
            .run(&ctx)
            .unwrap();
            assert!(
                (none.effect() - expected_mixture).abs() < tolerance,
                "{kind}: {}",
                none.effect()
            );
            for (suite, mut expected) in [
                (RefuteSuite::Cheap, vec!["window.stretch_80", "window.placebo_treatment"]),
                (
                    RefuteSuite::Full,
                    vec![
                        "window.stretch_80",
                        "window.stretch_50",
                        "window.placebo_treatment",
                        "window.split_half",
                    ],
                ),
            ] {
                let built = posterior_study(
                    &clean,
                    atom_posterior(&[0.6, 0.4], &[generating_atom(), no_effect_atom()], 3),
                    contrast_query(kind),
                    inference.clone(),
                    suite,
                );
                statuses.push(built.support_status());
                let result = built
                    .run(&ctx)
                    .unwrap_or_else(|error| panic!("{kind} {inference:?} {suite:?}: {error}"));
                let mut found = refuter_ids(&result);
                found.sort();
                expected.sort_unstable();
                // One mixed report per refuter, not one per atom.
                assert_eq!(found, expected, "{kind} {suite:?}");
                // Validation observes the mixture; it does not move it.
                assert!((result.effect() - none.effect()).abs() < 1e-12);
                let stretch = report(&result, "window.stretch_80");
                assert!((stretch.original_ate - result.effect()).abs() < tolerance, "{stretch:?}");
                assert!((stretch.refuted_ate - stretch.original_ate).abs() < 0.1, "{stretch:?}");
                let placebo = report(&result, "window.placebo_treatment");
                assert!(placebo.refuted_ate.abs() < 0.1, "{kind}: {placebo:?}");
            }
        }
    }
    // The planted break: the same posterior over the series whose treatment loses its effect
    // halfway is flagged by the mixed split-half check, which passes only when every atom passes.
    let broken = broken_series();
    for kind in KINDS {
        let result = posterior_study(
            &broken,
            atom_posterior(&[0.6, 0.4], &[generating_atom(), no_effect_atom()], 3),
            contrast_query(kind),
            InferenceMode::Frequentist,
            RefuteSuite::Full,
        )
        .run(&ctx)
        .unwrap();
        let split = report(&result, "window.split_half");
        assert!(!split.passed, "{kind}: {split:?}");
    }
    assert!(statuses.iter().all(|status| *status == Some(CellStatus::Licensed)), "{statuses:?}");
}

// ---------------------------------------------------------------------------
// The mediation process: columns c, x, w, m, y.
// ---------------------------------------------------------------------------

const C: u32 = 0;
const MX: u32 = 1;
const MW: u32 = 2;
const MM: u32 = 3;
const MY: u32 = 4;

fn mediation_series() -> TimeSeriesData {
    let states = states(pin()["mediation"]["series"].as_str().unwrap());
    let columns: Vec<Vec<f64>> = (0..5).map(|position| bit(&states, position)).collect();
    TimeSeriesData::from_f64_columns(
        [
            ("c", columns[0].as_slice()),
            ("x", columns[1].as_slice()),
            ("w", columns[2].as_slice()),
            ("m", columns[3].as_slice()),
            ("y", columns[4].as_slice()),
        ],
        1,
    )
    .unwrap()
}

const CONTRASTS: [(MediationContrast, &str); 3] = [
    (MediationContrast::Total, "total_effect"),
    (MediationContrast::NaturalDirect, "natural_direct_effect"),
    (MediationContrast::NaturalIndirect, "natural_indirect_effect"),
];

fn mediation_truth(key: &str) -> f64 {
    pin()["mediation"]["truth"][key].as_f64().unwrap()
}

fn mediation_query(contrast: MediationContrast) -> MediationQuery {
    MediationQuery::binary(v(MX), v(MY), [v(MM)], contrast)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum MediationShape {
    /// `c(t-1) -> x`, `x -> m`, `w -> m`, `m -> y`, `x -> y`.
    Generating,
    /// The generating template with a latent common cause of the mediator and the outcome.
    Recanting,
}

fn mediation_template(shape: MediationShape) -> TemporalAdmg {
    let mut graph = TemporalAdmg::empty();
    let c1 = graph.add_lagged(v(C), Lag::from_raw(1)).unwrap();
    let x0 = graph.add_lagged(v(MX), Lag::CONTEMPORANEOUS).unwrap();
    let w0 = graph.add_lagged(v(MW), Lag::CONTEMPORANEOUS).unwrap();
    let m0 = graph.add_lagged(v(MM), Lag::CONTEMPORANEOUS).unwrap();
    let y0 = graph.add_lagged(v(MY), Lag::CONTEMPORANEOUS).unwrap();
    for (from, to) in [(c1, x0), (x0, m0), (w0, m0), (m0, y0), (x0, y0)] {
        graph.insert_directed(from, to).unwrap();
    }
    if shape == MediationShape::Recanting {
        graph.insert_bidirected(m0, y0).unwrap();
    }
    graph
}

fn mediation_study(
    data: &TimeSeriesData,
    structure: &str,
    template: TemporalAdmg,
    contrast: MediationContrast,
    inference: InferenceMode,
    suite: RefuteSuite,
) -> Study {
    let base = Study::series(data.clone());
    let base = if structure == "accepted" {
        base.graph(AcceptedGraph::from(template))
    } else {
        base.graph(template)
    };
    base.query(CausalQuery::Mediation(mediation_query(contrast)))
        .inference(inference)
        .refute(suite)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
}

#[test]
fn temporal_admg_window_mediation_matches_the_enumerated_decomposition_on_both_inferences() {
    let _inferences = ("Frequentist", "Bayesian");
    let data = mediation_series();
    let ctx = ExecutionContext::for_tests(42_001);
    let mut statuses = Vec::new();
    for structure in ["explicit", "accepted"] {
        for (inference, tolerance) in inferences() {
            for (contrast, key) in CONTRASTS {
                let built = mediation_study(
                    &data,
                    structure,
                    mediation_template(MediationShape::Generating),
                    contrast,
                    inference.clone(),
                    RefuteSuite::None,
                );
                statuses.push(built.support_status());
                let result = built.run(&ctx).unwrap_or_else(|error| {
                    panic!("{structure} {contrast:?} {inference:?}: {error}")
                });
                let want = mediation_truth(key);
                assert!(
                    (result.effect() - want).abs() < tolerance,
                    "{structure} {contrast:?} {inference:?}: {} vs {want}",
                    result.effect()
                );
                let components = result.payload().mediation().expect("the decomposition");
                for (published, key) in [
                    (components.total, "total_effect"),
                    (components.direct, "natural_direct_effect"),
                    (components.mediated, "natural_indirect_effect"),
                ] {
                    let published = published.expect("every component is identified");
                    assert!(
                        (published - mediation_truth(key)).abs() < tolerance,
                        "{structure} {contrast:?} {inference:?} {key}: {published}"
                    );
                }
                assert!(discloses(&result, "estimate.temporal_admg.window"));
                assert!(result.refutations.is_empty());
            }
        }
    }
    // The pure natural indirect effect differs from the total minus the direct effect.
    let gap = mediation_truth("total_effect")
        - mediation_truth("natural_direct_effect")
        - mediation_truth("natural_indirect_effect");
    assert!(gap.abs() > 0.05, "{gap}");
    assert!(statuses.iter().all(|status| *status == Some(CellStatus::Licensed)), "{statuses:?}");
}

#[test]
fn temporal_admg_window_mediation_runs_from_the_retained_plan_without_the_builder() {
    let _inferences = ("Frequentist", "Bayesian");
    let data = mediation_series();
    let ctx = ExecutionContext::for_tests(42_002);
    for structure in ["explicit", "accepted"] {
        for (inference, tolerance) in inferences() {
            for (contrast, key) in CONTRASTS {
                let builder = mediation_study(
                    &data,
                    structure,
                    mediation_template(MediationShape::Generating),
                    contrast,
                    inference.clone(),
                    RefuteSuite::None,
                );
                let prepared = builder.prepare(&ctx).unwrap();
                assert_ne!(prepared.retained_operation().name(), "fresh_run");
                drop(builder);
                let result = prepared.estimate_retained(&ctx).unwrap();
                assert!(
                    (result.effect() - mediation_truth(key)).abs() < tolerance,
                    "{structure} {contrast:?} {inference:?}: {}",
                    result.effect()
                );
            }
        }
    }
}

#[test]
fn temporal_admg_window_mediation_refuses_a_recanting_district() {
    let _inferences = ("Frequentist", "Bayesian");
    let data = mediation_series();
    let ctx = ExecutionContext::for_tests(42_003);
    let refusal = mediation_study(
        &data,
        "explicit",
        mediation_template(MediationShape::Recanting),
        MediationContrast::NaturalIndirect,
        InferenceMode::Frequentist,
        RefuteSuite::None,
    )
    .run(&ctx)
    .unwrap_err()
    .to_string();
    assert!(refusal.contains("effect_not_identified"), "{refusal}");
}

const CHEAP_IDS: [&str; 8] = [
    "mediation.sequential_ignorability",
    "mediation.placebo_mediator_total",
    "mediation.placebo_mediator_direct",
    "mediation.placebo_mediator_indirect",
    "mediation.subset_total",
    "mediation.subset_direct",
    "mediation.subset_indirect",
    "mediation.joint_stability_subset",
];

const FULL_ONLY_IDS: [&str; 5] = [
    "mediation.half_sample_total",
    "mediation.half_sample_direct",
    "mediation.half_sample_indirect",
    "mediation.joint_stability_half_sample",
    "mediation.joint_bootstrap_range",
];

fn assert_mediation_suite(result: &StudyResult, suite: RefuteSuite) {
    let found = refuter_ids(result);
    let expected = if suite == RefuteSuite::Full {
        CHEAP_IDS.len() + FULL_ONLY_IDS.len()
    } else {
        CHEAP_IDS.len()
    };
    assert_eq!(found.len(), expected, "{found:?}");
    for id in CHEAP_IDS {
        assert!(found.iter().any(|f| f == id), "missing {id} in {found:?}");
    }
    for id in FULL_ONLY_IDS {
        assert_eq!(found.iter().any(|f| f == id), suite == RefuteSuite::Full, "{id}: {found:?}");
    }
}

#[test]
fn temporal_admg_window_mediation_validation_runs_the_mediation_suite_on_the_window_table() {
    let _inferences = ("Frequentist", "Bayesian");
    let data = mediation_series();
    let ctx = ExecutionContext::for_tests(42_004);
    let mut statuses = Vec::new();
    for structure in ["explicit", "accepted"] {
        for (inference, tolerance) in inferences() {
            for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
                let built = mediation_study(
                    &data,
                    structure,
                    mediation_template(MediationShape::Generating),
                    MediationContrast::NaturalIndirect,
                    inference.clone(),
                    suite,
                );
                statuses.push(built.support_status());
                let result = built
                    .run(&ctx)
                    .unwrap_or_else(|error| panic!("{structure} {inference:?} {suite:?}: {error}"));
                assert_mediation_suite(&result, suite);
                let want = mediation_truth("natural_indirect_effect");
                assert!((result.effect() - want).abs() < tolerance);
                // The mediator rotated by whole windows loses its link to the outcome: the
                // indirect effect the series carries is removed and the total is kept.
                let placebo = report(&result, "mediation.placebo_mediator_indirect");
                assert!((placebo.original_ate - want).abs() < tolerance, "{placebo:?}");
                assert!(placebo.refuted_ate.abs() < 0.05, "{placebo:?}");
                let total = report(&result, "mediation.placebo_mediator_total");
                assert!(
                    (total.refuted_ate - mediation_truth("total_effect")).abs() < 0.05,
                    "{total:?}"
                );
                // Every stability refit returns the enumerated component on the exact-law table.
                let subset = report(&result, "mediation.subset_indirect");
                assert!((subset.refuted_ate - want).abs() < 0.05, "{subset:?}");
            }
        }
    }
    assert!(statuses.iter().all(|status| *status == Some(CellStatus::Licensed)), "{statuses:?}");
}

// ---------------------------------------------------------------------------
// A graph posterior over lagged TemporalAdmg atoms: the mediation decomposition.
// ---------------------------------------------------------------------------

const CI: usize = 0;
const XM: usize = 1;
const WM: usize = 2;
const MM_I: usize = 3;
const YM: usize = 4;

/// `c(t-1) -> x`, `x -> m`, `w -> m`, `m -> y`, `x -> y`.
fn mediation_atom() -> TemporalAdmgAtom {
    TemporalAdmgAtom {
        contemporaneous: contemporaneous(&[(XM, MM_I), (WM, MM_I), (MM_I, YM), (XM, YM)], 5),
        lagged_directed: lagged(&[(1, CI, XM)], 5),
        lagged_bidirected: LagMask::EMPTY,
    }
}

/// The mediator does not reach the outcome: no path runs through it, so the pure natural
/// indirect effect has nothing to identify and the atom keeps its mass as unidentified mass.
fn inert_mediator_atom() -> TemporalAdmgAtom {
    TemporalAdmgAtom {
        contemporaneous: contemporaneous(&[(XM, MM_I), (WM, MM_I), (XM, YM)], 5),
        lagged_directed: lagged(&[(1, CI, XM)], 5),
        lagged_bidirected: LagMask::EMPTY,
    }
}

/// A latent cause `w(t-1)` of both the mediator and the outcome puts them in one district; no
/// node of that district has both a mediated and an unmediated edge out of it, so the pure
/// natural indirect effect is still identified, by the district formula.
fn recanting_atom() -> TemporalAdmgAtom {
    TemporalAdmgAtom {
        lagged_bidirected: lagged(&[(1, WM, MM_I), (1, WM, YM)], 5),
        ..mediation_atom()
    }
}

fn mediation_posterior() -> GraphPosterior {
    atom_posterior(
        &[0.5, 0.3, 0.2],
        &[mediation_atom(), inert_mediator_atom(), recanting_atom()],
        5,
    )
}

#[test]
fn temporal_admg_window_mediation_posterior_mixes_each_component_over_the_atoms_that_identify_it() {
    let _inferences = ("Frequentist", "Bayesian");
    let data = mediation_series();
    let ctx = ExecutionContext::for_tests(42_005);
    let mut statuses = Vec::new();
    for (inference, tolerance) in inferences() {
        // The indirect effect: the generating atom (weight 0.5) and the latent-cause atom (0.2)
        // identify it to the enumerated value; the inert-mediator atom's 0.3 is unidentified
        // mass, never redistributed, so the mixture is the mean over the identified 0.7.
        let indirect = mediation_truth("natural_indirect_effect");
        let built = Study::series(data.clone())
            .graph_posterior(mediation_posterior())
            .query(CausalQuery::Mediation(mediation_query(MediationContrast::NaturalIndirect)))
            .inference(inference.clone())
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap();
        statuses.push(built.support_status());
        let result = built.run(&ctx).unwrap_or_else(|error| panic!("{inference:?}: {error}"));
        let mixture = result.structural_response.as_ref().expect("the atom mixture");
        assert!((mixture.unidentified_mass - 0.3).abs() < 1e-12, "{}", mixture.unidentified_mass);
        assert!(mixture.unevaluable_mass.abs() < 1e-12);
        assert_eq!(result.identification.status, IdentificationStatus::GraphDependent);
        let by_key = |key: u64| mixture.atoms.iter().find(|atom| atom.graph_key == key).unwrap();
        assert_eq!(by_key(1).status, IdentificationStatus::NotIdentified);
        if matches!(inference, InferenceMode::Frequentist) {
            for key in [0, 2] {
                let atom = scalar(by_key(key).value.as_ref()).expect("an identified atom");
                assert!((atom - indirect).abs() < 1e-9, "atom {key}: {atom}");
            }
        }
        let expected = indirect;
        assert!(
            (result.effect() - expected).abs() < tolerance,
            "{inference:?}: {} vs {expected}",
            result.effect()
        );
        let components = result.payload().mediation().expect("the decomposition");
        // The total effect is identified by every atom, whatever the confounder of the mediator.
        assert!(
            (components.total.unwrap() - mediation_truth("total_effect")).abs() < tolerance,
            "{:?}",
            components.total
        );
        // The requested contrast is also mixed when it is the total.
        let total = Study::series(data.clone())
            .graph_posterior(mediation_posterior())
            .query(CausalQuery::Mediation(mediation_query(MediationContrast::Total)))
            .inference(inference.clone())
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .run(&ctx)
            .unwrap();
        assert!(
            (total.effect() - mediation_truth("total_effect")).abs() < tolerance,
            "{inference:?}: {}",
            total.effect()
        );
        let total_mixture = total.structural_response.as_ref().unwrap();
        assert!(total_mixture.unidentified_mass.abs() < 1e-12);
    }
    // The decomposition of a retained plan: the builder dropped, the frozen proofs re-estimate.
    let builder = Study::series(data.clone())
        .graph_posterior(mediation_posterior())
        .query(CausalQuery::Mediation(mediation_query(MediationContrast::NaturalIndirect)))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let prepared = builder.prepare(&ctx).unwrap();
    assert_ne!(prepared.retained_operation().name(), "fresh_run");
    drop(builder);
    let again = prepared.estimate_retained(&ctx).unwrap();
    let mixture = again.structural_response.as_ref().unwrap();
    assert!((mixture.unidentified_mass - 0.3).abs() < 1e-12);
    let expected = mediation_truth("natural_indirect_effect");
    assert!((again.effect() - expected).abs() < 1e-9, "{}", again.effect());
    assert!(statuses.iter().all(|status| *status == Some(CellStatus::Licensed)), "{statuses:?}");
}

#[test]
fn temporal_admg_window_mediation_posterior_validation_mixes_the_per_atom_reports() {
    let _inferences = ("Frequentist", "Bayesian");
    let data = mediation_series();
    let ctx = ExecutionContext::for_tests(42_006);
    let mut statuses = Vec::new();
    for (inference, tolerance) in inferences() {
        let none = Study::series(data.clone())
            .graph_posterior(mediation_posterior())
            .query(CausalQuery::Mediation(mediation_query(MediationContrast::NaturalIndirect)))
            .inference(inference.clone())
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .run(&ctx)
            .unwrap();
        assert!(none.refutations.is_empty());
        for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
            let built = Study::series(data.clone())
                .graph_posterior(mediation_posterior())
                .query(CausalQuery::Mediation(mediation_query(MediationContrast::NaturalIndirect)))
                .inference(inference.clone())
                .refute(suite)
                .bootstrap_replicates(0)
                .build()
                .unwrap();
            statuses.push(built.support_status());
            let result =
                built.run(&ctx).unwrap_or_else(|error| panic!("{inference:?} {suite:?}: {error}"));
            let found = refuter_ids(&result);
            for id in [
                "mediation.sequential_ignorability",
                "mediation.placebo_mediator_indirect",
                "mediation.joint_stability_subset",
            ] {
                assert!(found.iter().any(|f| f == id), "missing {id} in {found:?}");
            }
            assert_eq!(
                found.iter().any(|id| id == "mediation.joint_bootstrap_range"),
                suite == RefuteSuite::Full,
                "{found:?}"
            );
            // Validation observes the mixture, it does not move it, and the recanting atom still
            // keeps its mass as unidentified mass.
            assert!((result.effect() - none.effect()).abs() < 1e-12);
            let mixture = result.structural_response.as_ref().expect("the atom mixture");
            assert!((mixture.unidentified_mass - 0.3).abs() < 1e-12);
            let expected = mediation_truth("natural_indirect_effect");
            assert!((result.effect() - expected).abs() < tolerance);
        }
    }
    assert!(statuses.iter().all(|status| *status == Some(CellStatus::Licensed)), "{statuses:?}");
}

// ---------------------------------------------------------------------------
// A TemporalPag: the same process read through the ADMGs that share its maximal ancestral graph.
// ---------------------------------------------------------------------------

/// `c(t-1) -> x`, `x -> m`, `w -> m`, `m -> y`, `x -> y`: every edge into `m` and `y` and out of
/// `x` is visible because `c` points into `x` and `w` into `m`.
fn pag_template() -> TemporalPag {
    let mut graph = TemporalPag::empty();
    let c1 = graph.add_lagged(v(C), Lag::from_raw(1)).unwrap();
    let x0 = graph.add_lagged(v(MX), Lag::CONTEMPORANEOUS).unwrap();
    let w0 = graph.add_lagged(v(MW), Lag::CONTEMPORANEOUS).unwrap();
    let m0 = graph.add_lagged(v(MM), Lag::CONTEMPORANEOUS).unwrap();
    let y0 = graph.add_lagged(v(MY), Lag::CONTEMPORANEOUS).unwrap();
    for (from, to) in [(c1, x0), (x0, m0), (w0, m0), (m0, y0), (x0, y0)] {
        graph.insert_directed(from, to).unwrap();
    }
    graph
}

/// The same template with nothing pointing into `x`: every edge out of the treatment is invisible.
fn invisible_pag_template() -> TemporalPag {
    let mut graph = TemporalPag::empty();
    let x0 = graph.add_lagged(v(MX), Lag::CONTEMPORANEOUS).unwrap();
    let w0 = graph.add_lagged(v(MW), Lag::CONTEMPORANEOUS).unwrap();
    let m0 = graph.add_lagged(v(MM), Lag::CONTEMPORANEOUS).unwrap();
    let y0 = graph.add_lagged(v(MY), Lag::CONTEMPORANEOUS).unwrap();
    for (from, to) in [(x0, m0), (w0, m0), (m0, y0), (x0, y0)] {
        graph.insert_directed(from, to).unwrap();
    }
    graph
}

/// The enumerated truth lies in the identified set the result publishes (the range of the
/// functionals over the candidates), or equals the point when one functional identifies it.
fn assert_brackets(result: &StudyResult, truth: f64, tolerance: f64, label: &str) {
    match result.structural_response.as_ref().and_then(|mixture| mixture.identified_set.as_ref()) {
        Some(set) => {
            assert!(
                set.lower[0] - tolerance <= truth && truth <= set.upper[0] + tolerance,
                "{label}: [{}, {}] against {truth}",
                set.lower[0],
                set.upper[0]
            );
        }
        None => {
            assert!((result.effect() - truth).abs() < tolerance, "{label}: {}", result.effect())
        }
    }
}

fn pag_study(
    data: &TimeSeriesData,
    structure: &str,
    template: TemporalPag,
    contrast: MediationContrast,
    inference: InferenceMode,
    suite: RefuteSuite,
) -> Study {
    let base = Study::series(data.clone());
    let base = if structure == "accepted" {
        base.graph(AcceptedGraph::from(template))
    } else {
        base.graph(template)
    };
    base.query(CausalQuery::Mediation(mediation_query(contrast)))
        .inference(inference)
        .refute(suite)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
}

#[test]
fn temporal_pag_window_mediation_brackets_the_enumerated_decomposition_on_both_inferences() {
    let _inferences = ("Frequentist", "Bayesian");
    let data = mediation_series();
    let ctx = ExecutionContext::for_tests(43_001);
    let mut statuses = Vec::new();
    for structure in ["explicit", "accepted"] {
        for (inference, tolerance) in inferences() {
            for (contrast, key) in CONTRASTS {
                let built = pag_study(
                    &data,
                    structure,
                    pag_template(),
                    contrast,
                    inference.clone(),
                    RefuteSuite::None,
                );
                statuses.push(built.support_status());
                let result = built.run(&ctx).unwrap_or_else(|error| {
                    panic!("{structure} {contrast:?} {inference:?}: {error}")
                });
                assert_brackets(
                    &result,
                    mediation_truth(key),
                    tolerance,
                    &format!("{structure} {contrast:?} {inference:?}"),
                );
                assert!(discloses(&result, "estimate.temporal_pag.window"));
                assert!(result.refutations.is_empty());
            }
        }
    }
    assert!(statuses.iter().all(|status| *status == Some(CellStatus::Licensed)), "{statuses:?}");
}

#[test]
fn temporal_pag_window_mediation_runs_from_the_retained_plan_without_the_builder() {
    let _inferences = ("Frequentist", "Bayesian");
    let data = mediation_series();
    let ctx = ExecutionContext::for_tests(43_002);
    for structure in ["explicit", "accepted"] {
        for (inference, tolerance) in inferences() {
            for (contrast, key) in CONTRASTS {
                let builder = pag_study(
                    &data,
                    structure,
                    pag_template(),
                    contrast,
                    inference.clone(),
                    RefuteSuite::None,
                );
                let prepared = builder.prepare(&ctx).unwrap();
                assert_ne!(prepared.retained_operation().name(), "fresh_run");
                drop(builder);
                let result = prepared.estimate_retained(&ctx).unwrap();
                assert_brackets(
                    &result,
                    mediation_truth(key),
                    tolerance,
                    &format!("{structure} {contrast:?} {inference:?}"),
                );
            }
        }
    }
}

#[test]
fn temporal_pag_window_mediation_refuses_a_treatment_with_only_invisible_edges() {
    let _inferences = ("Frequentist", "Bayesian");
    let data = mediation_series();
    let ctx = ExecutionContext::for_tests(43_003);
    let error = pag_study(
        &data,
        "explicit",
        invisible_pag_template(),
        MediationContrast::NaturalIndirect,
        InferenceMode::Frequentist,
        RefuteSuite::None,
    )
    .run(&ctx)
    .unwrap_err();
    assert_eq!(error.reason_code(), Some("effect_not_identified"), "{error}");
}

#[test]
fn temporal_pag_window_mediation_posterior_accounts_for_the_atom_it_cannot_identify() {
    let _graph_class = "TemporalPag";
    let _inferences = ("Frequentist", "Bayesian");
    let data = mediation_series();
    let ctx = ExecutionContext::for_tests(43_004);
    let mut statuses = Vec::new();
    for (inference, tolerance) in inferences() {
        for (contrast, key) in CONTRASTS {
            let built = Study::series(data.clone())
                .graph_posterior(pag_posterior())
                .query(CausalQuery::Mediation(mediation_query(contrast)))
                .inference(inference.clone())
                .refute(RefuteSuite::None)
                .bootstrap_replicates(0)
                .build()
                .unwrap();
            statuses.push(built.support_status());
            let result = built
                .run(&ctx)
                .unwrap_or_else(|error| panic!("{contrast:?} {inference:?}: {error}"));
            let mixture = result.structural_response.as_ref().expect("the atom mixture");
            assert!(
                (mixture.unidentified_mass - 0.3).abs() < 1e-12,
                "{contrast:?} {inference:?}: {}",
                mixture.unidentified_mass
            );
            assert_eq!(result.identification.status, IdentificationStatus::GraphDependent);
            assert_brackets(
                &result,
                mediation_truth(key),
                tolerance,
                &format!("{contrast:?} {inference:?}"),
            );
        }
    }
    // The retained plan: the builder dropped, the frozen atom proofs re-estimate.
    let builder = Study::series(data.clone())
        .graph_posterior(pag_posterior())
        .query(CausalQuery::Mediation(mediation_query(MediationContrast::NaturalIndirect)))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let prepared = builder.prepare(&ctx).unwrap();
    assert_ne!(prepared.retained_operation().name(), "fresh_run");
    drop(builder);
    let again = prepared.estimate_retained(&ctx).unwrap();
    assert!((again.structural_response.as_ref().unwrap().unidentified_mass - 0.3).abs() < 1e-12);
    assert!(statuses.iter().all(|status| *status == Some(CellStatus::Licensed)), "{statuses:?}");
}

/// Two TemporalPag atoms: the template whose edges are visible (weight 0.7) and the template
/// with nothing into the treatment (weight 0.3, not identified).
fn pag_posterior() -> GraphPosterior {
    let n_vars = 5;
    let cell = n_vars * n_vars;
    let edges = [(XM, MM_I), (WM, MM_I), (MM_I, YM), (XM, YM)];
    let adjacency = contemporaneous(&edges, n_vars);
    GraphPosterior::from_atoms(
        n_vars,
        vec![0.7, 0.3],
        vec![adjacency.clone(), adjacency],
        vec![0.0; cell],
        vec![0.0; cell],
        SamplingSummary {
            ess: 1.0,
            diagnostics: InferenceDiagnostics::analytic("temporal_window_truth_pag"),
            rejected_invalid: 0,
        },
    )
    .unwrap()
    .with_atom_kind(GraphPosteriorAtomKind::Pag)
    .with_lagged_marginals(1, vec![0.0; cell])
    .unwrap()
    .with_lag_masks(vec![lagged(&[(1, CI, XM)], n_vars), LagMask::EMPTY])
    .unwrap()
}

/// The mediator column read `shift` windows further down, circularly: the mediator keeps its law
/// and loses its link to the outcome, so the indirect effect of the planted series is zero.
fn mediator_detached_series(shift: usize) -> TimeSeriesData {
    let states = states(pin()["mediation"]["series"].as_str().unwrap());
    let columns: Vec<Vec<f64>> = (0..5)
        .map(|position| {
            let mut column = bit(&states, position);
            if position == 3 {
                column.rotate_left(shift);
            }
            column
        })
        .collect();
    TimeSeriesData::from_f64_columns(
        [
            ("c", columns[0].as_slice()),
            ("x", columns[1].as_slice()),
            ("w", columns[2].as_slice()),
            ("m", columns[3].as_slice()),
            ("y", columns[4].as_slice()),
        ],
        1,
    )
    .unwrap()
}

#[test]
fn temporal_pag_window_mediation_validation_runs_the_mediation_suite_on_every_member() {
    let _inferences = ("Frequentist", "Bayesian");
    let _graph_class = "TemporalPag";
    let clean = mediation_series();
    let detached = mediator_detached_series(3_001);
    let ctx = ExecutionContext::for_tests(43_005);
    let mut statuses = Vec::new();
    for structure in ["explicit", "accepted", "graph_posterior"] {
        for (inference, tolerance) in inferences() {
            for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
                for (label, data) in [("clean", &clean), ("detached", &detached)] {
                    let built = if structure == "graph_posterior" {
                        Study::series(data.clone())
                            .graph_posterior(pag_posterior())
                            .query(CausalQuery::Mediation(mediation_query(
                                MediationContrast::NaturalIndirect,
                            )))
                            .inference(inference.clone())
                            .refute(suite)
                            .bootstrap_replicates(0)
                            .build()
                            .unwrap()
                    } else {
                        pag_study(
                            data,
                            structure,
                            pag_template(),
                            MediationContrast::NaturalIndirect,
                            inference.clone(),
                            suite,
                        )
                    };
                    statuses.push(built.support_status());
                    let result = built.run(&ctx).unwrap_or_else(|error| {
                        panic!("{structure} {label} {inference:?} {suite:?}: {error}")
                    });
                    let found = refuter_ids(&result);
                    let has = |prefix: &str| found.iter().any(|id| id.starts_with(prefix));
                    assert!(has("mediation.sequential_ignorability"), "{found:?}");
                    assert!(has("mediation.joint_stability_subset"), "{found:?}");
                    assert_eq!(has("mediation.joint_bootstrap_range"), suite == RefuteSuite::Full);
                    let placebos: Vec<_> = result
                        .refutations
                        .iter()
                        .filter(|r| r.refuter.starts_with("mediation.placebo_mediator_indirect"))
                        .collect();
                    assert!(!placebos.is_empty(), "{found:?}");
                    let truth = mediation_truth("natural_indirect_effect");
                    for placebo in &placebos {
                        // The rotated mediator carries no indirect effect in either series.
                        assert!(placebo.refuted_ate.abs() < 0.05, "{label}: {placebo:?}");
                    }
                    if label == "clean" {
                        let low = placebos.iter().map(|p| p.original_ate).fold(f64::MAX, f64::min);
                        let high = placebos.iter().map(|p| p.original_ate).fold(f64::MIN, f64::max);
                        assert!(
                            low - tolerance.max(0.05) <= truth
                                && truth <= high + tolerance.max(0.05),
                            "{structure}: [{low}, {high}] against {truth}"
                        );
                    } else {
                        // The planted series has its mediator detached: the effect is gone
                        // before the placebo runs.
                        for placebo in &placebos {
                            assert!(placebo.original_ate.abs() < 0.05, "{structure}: {placebo:?}");
                        }
                    }
                    if structure == "graph_posterior" {
                        let mixture = result.structural_response.as_ref().unwrap();
                        assert!((mixture.unidentified_mass - 0.3).abs() < 1e-12);
                    }
                }
            }
        }
    }
    assert!(statuses.iter().all(|status| *status == Some(CellStatus::Licensed)), "{statuses:?}");
}
