//! Known-truth check of `temporal.sequential.gcomp` (multi-step sustained) on the incomplete and
//! posterior structures it can estimate: a time-oriented `TemporalCpdag` and `TemporalPag`
//! (explicit and accepted), and single-identified-atom graph posteriors of every atom kind.
//!
//! The series is noise-free and exactly linear, `y_t = 1 + 2 t_{t-1} + 3 t_{t-2}` with the
//! binary treatment cycling `0, 0, 1` and the extra column `z` an alternating nuisance on which `y`
//! does not depend. Every valid adjustment set contains the two treatment lags, so least squares
//! recovers the coefficients exactly whatever else is adjusted for, and the sustained effect over
//! lags `-2..-1` is `2 + 3 = 5`. The two graphs hold only time-oriented edges, so they have one
//! completion and the identified envelope collapses to the point `5`; each posterior carries one
//! identified atom (plus, for the incomplete kinds, an atom that stays unidentified, which the
//! mixture reports as unidentified mass without moving the identified value).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "fixtures_common/mod.rs"]
mod common;

use antecedent::{AcceptedGraph, EstimatorId, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, ExecutionContext, TemporalEffectQuery, TemporalPolicy, VariableId,
};
use antecedent_data::{TableView, TimeSeriesData};
use antecedent_discovery::{GraphPosterior, GraphPosteriorAtomKind, SamplingSummary, set_edge};
use antecedent_graph::{TemporalCpdag, TemporalPag};
use antecedent_prob::InferenceDiagnostics;

use antecedent_core::Lag;

const TRUTH: f64 = 5.0;
const TOLERANCE: f64 = 1e-8;
const SUITES: [RefuteSuite; 3] = [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full];

fn var(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

/// Columns `t`, `y`, `z`.
fn series() -> TimeSeriesData {
    let n = 320usize;
    let t = (0..n).map(|i| if i % 3 == 2 { 1.0 } else { 0.0 }).collect::<Vec<_>>();
    let y = (0..n)
        .map(|i| {
            1.0 + 2.0 * i.checked_sub(1).map_or(0.0, |j| t[j])
                + 3.0 * i.checked_sub(2).map_or(0.0, |j| t[j])
        })
        .collect::<Vec<_>>();
    let z = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect::<Vec<_>>();
    TimeSeriesData::from_f64_columns(
        [("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())],
        1,
    )
    .unwrap()
}

/// Columns `t`, `y` only.
fn series_two() -> TimeSeriesData {
    let full = series();
    let t = full.float64_values(var(0)).unwrap().clone();
    let y = full.float64_values(var(1)).unwrap().clone();
    TimeSeriesData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice())], 1).unwrap()
}

/// Columns `t`, `y`, `z`, `m`, `v` for the five-variable chain PAG: `y_t = 1 + 2 t_{t-1}`, with
/// `z`, `m` and `v` nuisance cycles of coprime periods on which `y` does not depend, so every
/// completion's adjustment set recovers the coefficient exactly.
fn series_chain() -> TimeSeriesData {
    let n = 320usize;
    let t = (0..n).map(|i| if i % 3 == 2 { 1.0 } else { 0.0 }).collect::<Vec<_>>();
    let y = (0..n).map(|i| 1.0 + 2.0 * i.checked_sub(1).map_or(0.0, |j| t[j])).collect::<Vec<_>>();
    let z = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect::<Vec<_>>();
    let m = (0..n).map(|i| [1.0, 0.0, -1.0, 2.0, -2.0][i % 5]).collect::<Vec<_>>();
    let v = (0..n).map(|i| [0.0, 1.0, 2.0, -1.0, 0.0, 3.0, -2.0][i % 7]).collect::<Vec<_>>();
    TimeSeriesData::from_f64_columns(
        [
            ("t", t.as_slice()),
            ("y", y.as_slice()),
            ("z", z.as_slice()),
            ("m", m.as_slice()),
            ("v", v.as_slice()),
        ],
        1,
    )
    .unwrap()
}

fn query() -> CausalQuery {
    CausalQuery::TemporalEffect(
        TemporalEffectQuery::sustained(var(0), var(1), 0, 1.0)
            .with_policy(TemporalPolicy::sustained(-2, -1)),
    )
}

fn cpdag() -> TemporalCpdag {
    let mut g = TemporalCpdag::empty();
    let t1 = g.add_lagged(var(0), Lag::from_raw(1)).unwrap();
    let t2 = g.add_lagged(var(0), Lag::from_raw(2)).unwrap();
    let y0 = g.add_lagged(var(1), Lag::CONTEMPORANEOUS).unwrap();
    g.insert_directed(t1, y0).unwrap();
    g.insert_directed(t2, y0).unwrap();
    g
}

/// The seven-atom chain PAG `v o-o t o-o z o-o m`, `t -> y`, `m -> y` over `(t, y, z, m, v)`.
fn pag() -> TemporalPag {
    common::fixtures::chain_pag()
}

fn summary(label: &'static str) -> SamplingSummary {
    SamplingSummary {
        ess: 1.0,
        diagnostics: InferenceDiagnostics::analytic(label),
        rejected_invalid: 0,
    }
}

/// Two variables `(t, y)`: an identified atom `t@1 -> y`, `t@2 -> y` (mass 0.7) and the same
/// atom plus the autoregressive `t@1 -> t`, which stays unidentified (mass 0.3).
fn cpdag_posterior() -> GraphPosterior {
    // Lag `l` edge `from -> to` is bit `(l - 1) * 4 + from * 2 + to`.
    let identified = (1u64 << 1) | (1u64 << 5);
    let unidentified = identified | 1;
    let mut marginals = vec![0.0; 8];
    marginals[0] = 0.3;
    marginals[1] = 1.0;
    marginals[5] = 1.0;
    GraphPosterior::new(
        2,
        vec![0.7, 0.3],
        vec![0, 0],
        vec![0.0; 4],
        vec![0.0; 4],
        summary("sequential_cpdag_known_truth"),
    )
    .unwrap()
    .with_atom_kind(GraphPosteriorAtomKind::Cpdag)
    .with_lagged_marginals(2, marginals)
    .unwrap()
    .with_lag_masks(vec![identified, unidentified])
    .unwrap()
}

/// Three variables `(t, y, z)`: contemporaneous `z -> t`, `z -> y`; an identified atom
/// `t@1 -> y`, `t@2 -> y` (mass 0.7) and the same atom plus `t@1 -> t` (mass 0.3, unidentified).
fn pag_posterior() -> GraphPosterior {
    // Lag `l` edge `from -> to` is bit `(l - 1) * 9 + from * 3 + to`.
    let visible = set_edge(set_edge(0, 3, 2, 0, true), 3, 2, 1, true);
    let identified = (1u64 << 1) | (1u64 << 10);
    let unidentified = identified | 1;
    let mut marginals = vec![0.0; 18];
    marginals[0] = 0.3;
    marginals[1] = 1.0;
    marginals[10] = 1.0;
    GraphPosterior::new(
        3,
        vec![0.7, 0.3],
        vec![visible, visible],
        vec![0.0; 9],
        vec![0.0; 9],
        summary("sequential_pag_known_truth"),
    )
    .unwrap()
    .with_atom_kind(GraphPosteriorAtomKind::Pag)
    .with_lagged_marginals(2, marginals)
    .unwrap()
    .with_lag_masks(vec![identified, unidentified])
    .unwrap()
}

fn envelope_is_the_point(result: &antecedent::StudyResult, truth: f64, label: &str) {
    let envelope = result
        .structural_response
        .as_ref()
        .and_then(|mixture| mixture.identified_set.as_ref())
        .unwrap_or_else(|| panic!("{label}: the completion values define an envelope"));
    assert!((envelope.lower[0] - truth).abs() < TOLERANCE, "{label}: lower {}", envelope.lower[0]);
    assert!((envelope.upper[0] - truth).abs() < TOLERANCE, "{label}: upper {}", envelope.upper[0]);
}

#[test]
fn sequential_gcomp_recovers_the_sustained_effect_on_time_oriented_cpdag_and_pag() {
    let ctx = ExecutionContext::for_tests(977);
    for accepted in [false, true] {
        for class in ["TemporalCpdag", "TemporalPag"] {
            for suite in SUITES {
                let label = format!("{class} accepted={accepted} {suite:?}");
                let data = if class == "TemporalPag" { series_chain() } else { series() };
                let truth = if class == "TemporalPag" { 2.0 } else { TRUTH };
                let base = Study::series(data);
                let base = match (class, accepted) {
                    ("TemporalCpdag", false) => base.graph(cpdag()),
                    ("TemporalCpdag", true) => base.graph(AcceptedGraph::from(cpdag())),
                    (_, false) => base.graph(pag()),
                    (_, true) => base.graph(AcceptedGraph::from(pag())),
                };
                let result = base
                    .query(query())
                    .estimator(EstimatorId::TemporalSequentialGcomp)
                    .inference(InferenceMode::Frequentist)
                    .refute(suite)
                    .bootstrap_replicates(8)
                    .build()
                    .unwrap()
                    .run(&ctx)
                    .unwrap_or_else(|error| panic!("{label}: {error}"));
                envelope_is_the_point(&result, truth, &label);
            }
        }
    }
}

#[test]
fn sequential_gcomp_recovers_the_sustained_effect_over_single_identified_atom_posteriors() {
    let ctx = ExecutionContext::for_tests(4_100);
    let posteriors = [
        ("TemporalDag", common::fixtures::dbn_atom_a_only(), series(), 0.0),
        ("TemporalCpdag", cpdag_posterior(), series_two(), 0.3),
        ("TemporalPag", pag_posterior(), series(), 0.3),
    ];
    for (class, posterior, data, unidentified_mass) in posteriors {
        for suite in SUITES {
            let label = format!("{class} graph_posterior {suite:?}");
            let result = Study::series(data.clone())
                .graph_posterior(posterior.clone())
                .query(query())
                .estimator(EstimatorId::TemporalSequentialGcomp)
                .inference(InferenceMode::Frequentist)
                .refute(suite)
                .bootstrap_replicates(8)
                .build()
                .unwrap()
                .run(&ctx)
                .unwrap_or_else(|error| panic!("{label}: {error}"));
            let value = result.estimate().as_effect().unwrap().ate;
            assert!((value - TRUTH).abs() < TOLERANCE, "{label}: {value}");
            let mass = result
                .structural_response
                .as_ref()
                .map(|mixture| mixture.unidentified_mass)
                .or_else(|| result.posterior.as_ref().map(|p| p.unidentified_mass))
                .unwrap_or(0.0);
            assert!((mass - unidentified_mass).abs() < 1e-9, "{label}: unidentified mass {mass}");
        }
    }
}
