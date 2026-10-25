//! Natural mediation on the finite window of a temporal PAG, checked against the effect an
//! enumerated structural model produces.
//!
//! The reference solves the window model with the treatment read at one level along the
//! selected edges and at the control level along the others, sharing every exogenous state
//! between the worlds; it uses no identification code. The cases pin what the temporal route
//! adds to the static PAG route:
//!
//! * a template without lags unfolds to the window of one period, and its natural indirect,
//!   natural direct and total effects equal the effects of every model the window MAG
//!   represents;
//! * a cause that precedes the window is a latent common cause of its in-window effects: two
//!   ancestors of the treatment and of the outcome that share a cause before the window are
//!   joined by a bidirected edge, and the functional the envelope carries is the effect of the
//!   model with that confounder; the induced sub-graph (which drops it) yields the same effect
//!   here, because the confounded ancestors of the treatment and the outcome are observed in
//!   the window and block every path through the hidden cause;
//! * a treatment whose edges are all invisible leaves a candidate with a recanting district,
//!   and the route refuses instead of reporting the functional of one of the models.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    reason = "test fixtures: small exact indices, levels and enumerated probabilities"
)]

#[path = "support/path_specific_scm.rs"]
mod path_specific_scm;

use antecedent_core::{
    CausalQuery, Lag, MediationContrast, MediationQuery, TemporalEffectQuery, VariableId,
};
use antecedent_graph::TemporalPag;
use antecedent_identify::{
    GeneralizedAdjustmentIdentifier, TemporalClassEnvelope, TemporalPagMediationWindow,
    carries_identified_mass, temporal_pag_mediation_window,
};
use path_specific_scm::latent_scm::{LatentScm, ResponseType, Rng};
use path_specific_scm::{admg, evaluate, identify, path_specific_truth};

fn var(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn lagged(graph: &mut TemporalPag, variable: u32, lag: u32) -> antecedent_graph::DenseNodeId {
    graph.add_lagged(var(variable), Lag::from_raw(lag)).unwrap()
}

/// The window of the contrast on `template` at one outcome horizon, and the envelope of its
/// completions.
fn identified(
    template: &TemporalPag,
    treatment: u32,
    outcome: u32,
    mediators: &[u32],
    contrast: MediationContrast,
    variables: u32,
) -> (TemporalPagMediationWindow, TemporalClassEnvelope) {
    let query = MediationQuery::binary(
        var(treatment),
        var(outcome),
        mediators.iter().copied().map(var).collect::<Vec<_>>(),
        contrast,
    );
    let effect =
        TemporalEffectQuery::pulse(var(treatment), var(outcome), 1.0).with_horizon_steps(1);
    let window = temporal_pag_mediation_window(template, &effect, &query, variables).unwrap();
    let envelope = GeneralizedAdjustmentIdentifier::new()
        .identify_temporal_pag_path_specific_envelope(template, &window)
        .unwrap();
    (window, envelope)
}

const CONTRASTS: [(MediationContrast, &str); 3] = [
    (MediationContrast::NaturalIndirect, "indirect"),
    (MediationContrast::NaturalDirect, "direct"),
    (MediationContrast::Total, "total"),
];

/// Children of the treatment whose edge is read at the active level for each contrast, given
/// the mediator and the outcome (the treatment's two children).
fn selected_children(contrast: MediationContrast, mediator: usize, outcome: usize) -> Vec<usize> {
    match contrast {
        MediationContrast::NaturalIndirect => vec![mediator],
        MediationContrast::NaturalDirect => vec![outcome],
        _ => vec![mediator, outcome],
    }
}

/// `C -> T -> M -> Y` within one period, `S -> M`, and `T -> Y` when `direct`: the only
/// invisible edge is `C -> T`, and a hidden common cause of `C` and `T` does not reach the
/// effect of `T` on `Y`. The cause `S` of the mediator makes `M <-> Y` a different
/// m-separation model (it would open `S` to `Y` given `M`), so no model the MAG represents
/// confounds the mediator and the outcome; without it the direct edge would hide such a model,
/// whose district a selected and a complementary edge both enter.
fn chain_template(direct: bool) -> TemporalPag {
    let mut graph = TemporalPag::empty();
    let nodes: Vec<_> = (0..5).map(|i| lagged(&mut graph, i, 0)).collect();
    for (from, to) in [(0, 1), (1, 2), (2, 3), (4, 2)] {
        graph.insert_directed(nodes[from], nodes[to]).unwrap();
    }
    if direct {
        graph.insert_directed(nodes[1], nodes[3]).unwrap();
    }
    graph
}

#[test]
fn a_window_without_lags_carries_the_effect_of_every_model_its_mag_represents() {
    // The treatment-outcome path also runs straight, so the three contrasts differ.
    let template = chain_template(true);
    let directed = [(0, 1), (1, 2), (2, 3), (1, 3), (4, 2)];
    for (contrast, name) in CONTRASTS {
        let (window, found) = identified(&template, 1, 3, &[2], contrast, 5);
        // History 0 and one period: the coordinates are the variables.
        assert_eq!(window.query.treatment, var(1), "{name}");
        let envelope = &found.envelope;
        assert_eq!(envelope.unidentified_weight.0, 0.0, "{name}: {:?}", envelope.status);
        assert!(envelope.identified_weight.0 > 0.0, "{name}");
        assert!(
            found.indexers.iter().all(|indexer| indexer.history() == 0 && indexer.horizon() == 1),
            "{name}: every case carries the window's own layout"
        );
        for bidirected in [&[][..], &[(0, 1)][..]] {
            for seed in 0..6u64 {
                let scm = LatentScm::random(
                    &mut Rng::new(80 + seed),
                    &[2, 2, 2, 2, 2],
                    &directed,
                    bidirected,
                    3,
                );
                let truth = path_specific_truth(&scm, 1, 3, &selected_children(contrast, 2, 3));
                assert!(
                    envelope
                        .cases
                        .iter()
                        .any(|case| (evaluate(&case.result, &scm) - truth).abs() < 1e-9),
                    "{name}, seed {seed}, latent edges {bidirected:?}: no carried functional \
                     equals the effect {truth}"
                );
            }
        }
    }
}

/// `u(t-1) -> p(t)`, `u(t-2) -> q(t)`, `p(t-2) -> x(t)`, `q(t-1) -> y(t)`, `x -> m -> y` and
/// `x -> y` within a period, and `s -> m` (a cause of the mediator, which makes `m <-> y` a
/// different m-separation model, so no model the MAG represents confounds them). The window
/// holds two periods of history, so `u` two periods back is a cause before the window of `p`
/// two periods back and of `q` one period back, both ancestors of the treatment and of the
/// outcome.
fn lagged_confounder_template() -> TemporalPag {
    let mut graph = TemporalPag::empty();
    let u1 = lagged(&mut graph, 0, 1);
    let u2 = lagged(&mut graph, 0, 2);
    let p0 = lagged(&mut graph, 1, 0);
    let p2 = lagged(&mut graph, 1, 2);
    let q0 = lagged(&mut graph, 2, 0);
    let q1 = lagged(&mut graph, 2, 1);
    let x0 = lagged(&mut graph, 3, 0);
    let m0 = lagged(&mut graph, 4, 0);
    let y0 = lagged(&mut graph, 5, 0);
    let s0 = lagged(&mut graph, 6, 0);
    for (from, to) in
        [(u1, p0), (u2, q0), (p2, x0), (q1, y0), (x0, m0), (m0, y0), (x0, y0), (s0, m0)]
    {
        graph.insert_directed(from, to).unwrap();
    }
    graph
}

/// Window coordinates: `(offset + 2) * 7 + variable`.
const P_MINUS_2: usize = 1;
const Q_MINUS_1: usize = 9;
const X_NOW: usize = 17;
const M_NOW: usize = 18;
const Y_NOW: usize = 19;
const S_NOW: usize = 20;
const WINDOW_NODES: usize = 21;

/// The structural model of the window: the six ancestors of the effect are binary, every
/// other window node is a constant, and the hidden cause before the window is the latent of
/// the bidirected edge between `p(-2)` and `q(-1)`.
fn confounded_window_scm(seed: u64) -> LatentScm {
    let cards: Vec<usize> = (0..WINDOW_NODES)
        .map(|node| usize::from([P_MINUS_2, Q_MINUS_1, X_NOW, M_NOW, Y_NOW, S_NOW].contains(&node)) + 1)
        .collect();
    let directed = [
        (P_MINUS_2, X_NOW),
        (Q_MINUS_1, Y_NOW),
        (X_NOW, M_NOW),
        (M_NOW, Y_NOW),
        (X_NOW, Y_NOW),
        (S_NOW, M_NOW),
    ];
    let mut scm =
        LatentScm::random(&mut Rng::new(seed), &cards, &directed, &[(P_MINUS_2, Q_MINUS_1)], 3);
    for node in 0..WINDOW_NODES {
        if cards[node] == 1 {
            let size = scm.table_size(node);
            scm.types[node] = vec![ResponseType { probability: 1.0, table: vec![0; size] }];
        }
    }
    scm
}

#[test]
fn a_cause_before_the_window_is_a_confounder_of_its_in_window_effects() {
    let template = lagged_confounder_template();
    let mut widest_gap: f64 = 0.0;
    for (contrast, name) in CONTRASTS {
        let (window, found) = identified(&template, 3, 5, &[4], contrast, 7);
        assert_eq!(window.indexer.history(), 2, "{name}");
        assert_eq!(window.query.treatment.raw() as usize, X_NOW, "{name}");
        assert_eq!(window.query.outcome.raw() as usize, Y_NOW, "{name}");
        assert_eq!(window.query.mediators.as_ref(), &[var(M_NOW as u32)], "{name}");
        let envelope = &found.envelope;
        assert_eq!(
            envelope.unidentified_weight.0, 0.0,
            "{name}: {:?} {:?}",
            envelope.status, envelope.critical_graph_features
        );
        assert!(envelope.identified_weight.0 > 0.0, "{name}");
        // The sub-graph the window induces drops the cause before the window: no bidirected
        // edge joins p(-2) and q(-1). The ancestors it joins are observed, and the path
        // x <- p <-> q -> y is closed by conditioning on either.
        let directed = [
            (P_MINUS_2, X_NOW),
            (Q_MINUS_1, Y_NOW),
            (X_NOW, M_NOW),
            (M_NOW, Y_NOW),
            (X_NOW, Y_NOW),
            (S_NOW, M_NOW),
        ];
        let induced = identify(
            &admg(WINDOW_NODES, &directed, &[]),
            &CausalQuery::Mediation(window.query.clone()),
        );
        for seed in 0..8u64 {
            let scm = confounded_window_scm(100 + seed);
            let truth =
                path_specific_truth(&scm, X_NOW, Y_NOW, &selected_children(contrast, M_NOW, Y_NOW));
            assert!(
                envelope
                    .cases
                    .iter()
                    .filter(|case| carries_identified_mass(case.result.status))
                    .any(|case| (evaluate(&case.result, &scm) - truth).abs() < 1e-9),
                "{name}, seed {seed}: no carried functional equals the effect {truth} of the \
                 model whose hidden cause precedes the window"
            );
            if carries_identified_mass(induced.status) {
                widest_gap = widest_gap.max((evaluate(&induced, &scm) - truth).abs());
            }
        }
    }
    assert!(
        widest_gap < 1e-9,
        "an observed ancestor of the treatment closes the hidden cause's path: {widest_gap}"
    );
}

#[test]
fn a_treatment_with_only_invisible_edges_is_refused_with_the_candidate_that_recants() {
    // T -> M -> Y and T -> Y with nothing pointing into T: every edge is invisible, so the
    // MAG leaves a model with M <-> Y whose district recants.
    let mut template = TemporalPag::empty();
    let nodes: Vec<_> = (0..3).map(|i| lagged(&mut template, i, 0)).collect();
    for (from, to) in [(0, 1), (1, 2), (0, 2)] {
        template.insert_directed(nodes[from], nodes[to]).unwrap();
    }
    let (_, found) = identified(&template, 0, 2, &[1], MediationContrast::NaturalIndirect, 3);
    let envelope = &found.envelope;
    assert!(envelope.unidentified_weight.0 > 0.0, "{:?}", envelope.status);
    assert_eq!(envelope.identified_weight.0, 0.0);
    assert!(
        envelope.cases.iter().all(|case| !carries_identified_mass(case.result.status)),
        "a refused candidate must leave no point behind"
    );
    assert!(
        envelope.cases.iter().any(|case| case
            .result
            .diagnostics
            .iter()
            .any(|d| d.code.as_ref() == "identify.path_specific_pag.unidentified_candidate")),
        "the refusal must name the candidate"
    );
}

#[test]
fn a_mediation_without_a_mediator_period_in_the_window_is_a_typed_error() {
    let template = chain_template(false);
    let query = MediationQuery::binary(var(1), var(3), [var(1)], MediationContrast::Total);
    let effect = TemporalEffectQuery::pulse(var(1), var(3), 1.0).with_horizon_steps(1);
    assert!(temporal_pag_mediation_window(&template, &effect, &query, 5).is_err());
}
