//! Natural mediation over a graph posterior of temporal PAGs.
//!
//! Each atom is a stationary template read as a supplied `TemporalPag` is, so the expected
//! value of every atom is the explicit route on the same series and the same query; the
//! posterior moves only the weights the atoms' decompositions are mixed with.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    reason = "test fixtures: small exact indices and enumerated probabilities"
)]

use antecedent::{CellStatus, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, ExecutionContext, Lag, MediationContrast, MediationQuery, VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_discovery::{
    GraphPosterior, GraphPosteriorAtomKind, LagMask, SamplingSummary, set_edge, with_lagged_edge,
};
use antecedent_graph::TemporalPag;
use antecedent_prob::InferenceDiagnostics;

const C: u32 = 0;
const X: u32 = 1;
const W: u32 = 2;
const M: u32 = 3;
const Y: u32 = 4;
const N_VARS: usize = 5;

fn var(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

/// `c(t-1) -> x(t)`, `x -> m`, `w -> m`, `m -> y`, `x -> y` within a period.
fn lagged_template() -> TemporalPag {
    let mut graph = TemporalPag::empty();
    let c1 = graph.add_lagged(var(C), Lag::from_raw(1)).unwrap();
    let x0 = graph.add_lagged(var(X), Lag::CONTEMPORANEOUS).unwrap();
    let w0 = graph.add_lagged(var(W), Lag::CONTEMPORANEOUS).unwrap();
    let m0 = graph.add_lagged(var(M), Lag::CONTEMPORANEOUS).unwrap();
    let y0 = graph.add_lagged(var(Y), Lag::CONTEMPORANEOUS).unwrap();
    for (from, to) in [(c1, x0), (x0, m0), (w0, m0), (m0, y0), (x0, y0)] {
        graph.insert_directed(from, to).unwrap();
    }
    graph
}

/// The same template with `c(t)` instead of `c(t-1)`: no lag, a window of one period.
fn contemporaneous_template() -> TemporalPag {
    let mut graph = TemporalPag::empty();
    let c0 = graph.add_lagged(var(C), Lag::CONTEMPORANEOUS).unwrap();
    let x0 = graph.add_lagged(var(X), Lag::CONTEMPORANEOUS).unwrap();
    let w0 = graph.add_lagged(var(W), Lag::CONTEMPORANEOUS).unwrap();
    let m0 = graph.add_lagged(var(M), Lag::CONTEMPORANEOUS).unwrap();
    let y0 = graph.add_lagged(var(Y), Lag::CONTEMPORANEOUS).unwrap();
    for (from, to) in [(c0, x0), (x0, m0), (w0, m0), (m0, y0), (x0, y0)] {
        graph.insert_directed(from, to).unwrap();
    }
    graph
}

/// Deterministic uniform draws (splitmix64).
struct Draws(u64);

impl Draws {
    fn unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }

    fn flip(&mut self, p: f64) -> f64 {
        f64::from(self.unit() < p)
    }
}

/// A series whose treatment follows the previous period's `c` and whose other variables follow
/// the generating equations within the period.
fn lagged_series(len: usize, seed: u64) -> TimeSeriesData {
    let mut draws = Draws(seed);
    let (mut c, mut x, mut w, mut m, mut y) =
        (vec![0.0; len], vec![0.0; len], vec![0.0; len], vec![0.0; len], vec![0.0; len]);
    for t in 0..len {
        c[t] = draws.flip(0.45);
        w[t] = draws.flip(0.35);
        let previous = if t == 0 { 0.0 } else { c[t - 1] };
        x[t] = draws.flip(0.25 + 0.5 * previous);
        m[t] = draws.flip(0.15 + 0.4 * x[t] + 0.3 * w[t]);
        y[t] = draws.flip(0.1 + 0.2 * x[t] + 0.3 * m[t] + 0.2 * x[t] * m[t]);
    }
    TimeSeriesData::from_f64_columns(
        [
            ("c", c.as_slice()),
            ("x", x.as_slice()),
            ("w", w.as_slice()),
            ("m", m.as_slice()),
            ("y", y.as_slice()),
        ],
        1,
    )
    .unwrap()
}

fn query(contrast: MediationContrast) -> MediationQuery {
    MediationQuery::binary(var(X), var(Y), [var(M)], contrast)
}

fn contemporaneous(edges: &[(u32, u32)]) -> u64 {
    edges
        .iter()
        .fold(0_u64, |mask, &(from, to)| set_edge(mask, N_VARS, from as usize, to as usize, true))
}

/// `(contemporaneous mask, lag mask)` of the lagged template.
fn lagged_atom() -> (u64, LagMask) {
    (
        contemporaneous(&[(X, M), (W, M), (M, Y), (X, Y)]),
        with_lagged_edge(0, N_VARS, 1, C as usize, X as usize, true).unwrap(),
    )
}

/// `(contemporaneous mask, lag mask)` of the template with no lag.
fn contemporaneous_atom() -> (u64, LagMask) {
    (contemporaneous(&[(C, X), (X, M), (W, M), (M, Y), (X, Y)]), LagMask::EMPTY)
}

fn pag_posterior(weights: &[f64], atoms: &[(u64, LagMask)]) -> GraphPosterior {
    GraphPosterior::from_atoms(
        N_VARS,
        weights.to_vec(),
        atoms.iter().map(|atom| atom.0),
        vec![0.0; N_VARS * N_VARS],
        vec![0.0; N_VARS * N_VARS],
        SamplingSummary {
            ess: 1.0,
            diagnostics: InferenceDiagnostics::analytic("graph_posterior_temporal_pag"),
            rejected_invalid: 0,
        },
    )
    .unwrap()
    .with_atom_kind(GraphPosteriorAtomKind::Pag)
    .with_lagged_marginals(1, vec![0.0; N_VARS * N_VARS])
    .unwrap()
    .with_lag_masks(atoms.iter().map(|atom| atom.1.clone()).collect::<Vec<_>>())
    .unwrap()
    .with_mark_atoms(vec![0_u64; atoms.len()])
    .unwrap()
}

fn explicit(series: &TimeSeriesData, graph: TemporalPag, contrast: MediationContrast) -> f64 {
    Study::series(series.clone())
        .graph(graph)
        .query(CausalQuery::Mediation(query(contrast)))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(8_201))
        .unwrap()
        .effect()
}

fn mixed_study(
    series: &TimeSeriesData,
    graphs: GraphPosterior,
    contrast: MediationContrast,
) -> Study {
    Study::series(series.clone())
        .graph_posterior(graphs)
        .query(CausalQuery::Mediation(query(contrast)))
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
}

/// Each atom decomposes on the window of its own template, and the decomposition of each
/// contrast is the posterior-weighted mean of the explicit routes' own values.
#[test]
fn a_posterior_of_temporal_pags_mixes_each_atoms_window_decomposition() {
    let series = lagged_series(6_000, 31);
    let ctx = ExecutionContext::for_tests(8_202);
    for contrast in [
        MediationContrast::Total,
        MediationContrast::NaturalDirect,
        MediationContrast::NaturalIndirect,
    ] {
        let expected = 0.6 * explicit(&series, lagged_template(), contrast)
            + 0.4 * explicit(&series, contemporaneous_template(), contrast);
        let graphs = pag_posterior(&[0.6, 0.4], &[lagged_atom(), contemporaneous_atom()]);
        let built = mixed_study(&series, graphs, contrast);
        assert_eq!(built.support_status(), Some(CellStatus::Refused));
        let result = built.run(&ctx).unwrap_or_else(|e| panic!("{contrast:?}: {e}"));
        assert!((result.effect() - expected).abs() < 1e-9, "{contrast:?}: {}", result.effect());
        let mixture = result.structural_response.as_ref().expect("the atom mixture");
        assert_eq!(mixture.atoms.len(), 2);
        assert!((mixture.identified_mass - 1.0).abs() < 1e-9);
        assert!(mixture.unidentified_mass.abs() < 1e-12);
        let components = result.payload().mediation().expect("the decomposition");
        for (component, contrast) in [
            (components.total, MediationContrast::Total),
            (components.direct, MediationContrast::NaturalDirect),
            (components.mediated, MediationContrast::NaturalIndirect),
        ] {
            let each = 0.6 * explicit(&series, lagged_template(), contrast)
                + 0.4 * explicit(&series, contemporaneous_template(), contrast);
            assert!((component.unwrap() - each).abs() < 1e-9, "{contrast:?}");
        }
    }
}

/// Moving the posterior weight moves the mixture along the segment between the atoms' values.
#[test]
fn the_mixture_follows_the_posterior_weights() {
    let series = lagged_series(6_000, 31);
    let ctx = ExecutionContext::for_tests(8_203);
    let contrast = MediationContrast::NaturalIndirect;
    let (a, b) = (
        explicit(&series, lagged_template(), contrast),
        explicit(&series, contemporaneous_template(), contrast),
    );
    for weight in [0.1, 0.5, 0.9] {
        let graphs =
            pag_posterior(&[weight, 1.0 - weight], &[lagged_atom(), contemporaneous_atom()]);
        let result = mixed_study(&series, graphs, contrast).run(&ctx).unwrap();
        let expected = weight * a + (1.0 - weight) * b;
        assert!((result.effect() - expected).abs() < 1e-9, "{weight}: {}", result.effect());
    }
}

/// A posterior of one atom is that atom's explicit route.
#[test]
fn a_posterior_of_one_template_equals_the_supplied_temporal_pag() {
    let series = lagged_series(6_000, 31);
    let ctx = ExecutionContext::for_tests(8_204);
    let contrast = MediationContrast::NaturalDirect;
    let graphs = pag_posterior(&[1.0], &[lagged_atom()]);
    let result = mixed_study(&series, graphs, contrast).run(&ctx).unwrap();
    assert!((result.effect() - explicit(&series, lagged_template(), contrast)).abs() < 1e-9);
    let prepared = mixed_study(&series, pag_posterior(&[1.0], &[lagged_atom()]), contrast)
        .prepare(&ctx)
        .unwrap();
    let again = prepared.estimate_series(&series, &ctx).unwrap();
    assert!((again.effect() - result.effect()).abs() < 1e-9);
}

/// Several temporal-PAG atoms are re-run on shared resampled series, and the standard error is
/// the spread of the weighted mixture over the replicates. A standard error needs at least
/// `PERCENTILE_95_MIN_REPLICATES` (40) surviving replicates, so fewer are withheld.
#[test]
fn a_frequentist_mixture_of_templates_has_the_standard_error_of_its_shared_series() {
    let series = lagged_series(3_000, 31);
    let ctx = ExecutionContext::for_tests(8_205);
    let result = Study::series(series.clone())
        .graph_posterior(pag_posterior(&[0.6, 0.4], &[lagged_atom(), contemporaneous_atom()]))
        .query(CausalQuery::Mediation(query(MediationContrast::Total)))
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(40)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    let effect = result.estimate().as_effect().unwrap();
    let se = effect.se_bootstrap().expect("the spread of the shared replicates");
    assert!(se > 0.0 && se < 0.5, "{se}");
    assert!(effect.uncertainty.bootstrap.replicates_ok.unwrap() >= 40);
    assert!(
        result
            .diagnostics
            .iter()
            .any(|d| d.code.as_ref() == "estimate.graph_posterior.cross_world_uncertainty")
    );
}
