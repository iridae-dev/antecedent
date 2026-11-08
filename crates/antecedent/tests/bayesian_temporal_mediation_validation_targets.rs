//! Temporal mediation on the `conformance/bayesian/posterior_moments` finite sample:
//! the estimate matches the independently pinned posterior mean, and the
//! mediation validation targets behave as designed for every graph source,
//! validation suite and inference mode.
//!
//! The fixture's `mediation` family (`t`, `m`, `o`; graph `t@1 → m`, `t@1 → y`,
//! `m → y`) has posterior moments derived by `generate.py` from matrix equations
//! alone. The placebo mediator must refute to zero, the random common cause must
//! leave the estimate in place, the full suite adds the contiguous-window
//! report, a Bayesian full suite carries the prior sensitivity, and no suite
//! produces no reports.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{AcceptedGraph, BayesianConfig, InferenceMode, RefuteSuite, Study, StudyResult};
use antecedent_core::{
    CausalQuery, ExecutionContext, Lag, MediationContrast, MediationQuery, VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_graph::{ensure_lagged, TemporalDag};

/// Tolerance of the estimate against the pinned posterior mean.
const ATE_TOLERANCE: f64 = 0.02;
/// The placebo mediator carries no mediated effect.
const PLACEBO_TOLERANCE: f64 = 0.05;
/// The random common cause leaves the estimate unchanged.
const RCC_TOLERANCE: f64 = 0.02;

fn fixture() -> serde_json::Value {
    serde_json::from_str(
        &std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../conformance/bayesian/posterior_moments/expected.json"
        ))
        .unwrap(),
    )
    .unwrap()
}

fn column(fixture: &serde_json::Value, name: &str) -> Vec<f64> {
    fixture["data"][name].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect()
}

fn run(
    data: &TimeSeriesData,
    graph: &TemporalDag,
    accepted: bool,
    suite: RefuteSuite,
    bayesian: bool,
) -> StudyResult {
    let builder = Study::series(data.clone());
    let builder = if accepted {
        builder.graph(AcceptedGraph::temporal_dag(graph.clone()))
    } else {
        builder.graph(graph.clone())
    };
    let builder = builder.query(CausalQuery::Mediation(
        MediationQuery::binary(
            VariableId::from_raw(0),
            VariableId::from_raw(2),
            [VariableId::from_raw(1)],
            MediationContrast::Mediated,
        )
        .with_horizons(vec![1])
        .unwrap(),
    ));
    let builder = if bayesian {
        builder.inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().n_draws(8192).prior_scale(10.0),
        ))
    } else {
        builder
    };
    builder
        .refute(suite)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(1201))
        .unwrap()
}

#[test]
fn mediation_validation_targets_and_prior_grid_match_the_pinned_posterior() {
    let fixture = fixture();
    let data = TimeSeriesData::from_f64_columns(
        [
            ("t", column(&fixture, "t").as_slice()),
            ("m", &column(&fixture, "m")),
            ("y", &column(&fixture, "o")),
        ],
        1,
    )
    .unwrap();
    let mut graph = TemporalDag::empty();
    let t1 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let m0 = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let y0 = ensure_lagged(&mut graph, VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    for (from, to) in [(t1, m0), (t1, y0), (m0, y0)] {
        graph.insert_directed(from, to).unwrap();
    }
    let pinned = fixture["posterior"]["10.0"]["mediation"][0].as_f64().unwrap();

    for accepted in [false, true] {
        for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
            for bayesian in [false, true] {
                let label = format!("accepted={accepted} suite={suite:?} bayesian={bayesian}");
                let result = run(&data, &graph, accepted, suite, bayesian);
                let ate = result.estimate().as_effect().unwrap().ate;
                assert!(
                    (ate - pinned).abs() < ATE_TOLERANCE,
                    "{label}: ate {ate} vs pinned posterior mean {pinned}"
                );
                if suite == RefuteSuite::None {
                    assert!(result.refutations.is_empty(), "{label}: reports without a suite");
                    continue;
                }
                let report =
                    |id: &str| result.refutations.iter().find(|r| r.refuter.as_ref() == id);
                let placebo = report("mediation.placebo_mediator")
                    .unwrap_or_else(|| panic!("{label}: placebo mediator report missing"));
                assert!(
                    placebo.refuted_ate.abs() < PLACEBO_TOLERANCE,
                    "{label}: placebo {}",
                    placebo.refuted_ate
                );
                let rcc = report("mediation.random_common_cause")
                    .unwrap_or_else(|| panic!("{label}: random common cause report missing"));
                assert!(
                    (rcc.refuted_ate - ate).abs() < RCC_TOLERANCE,
                    "{label}: random common cause {} vs ate {ate}",
                    rcc.refuted_ate
                );
                if suite == RefuteSuite::Full {
                    assert!(
                        report("mediation.contiguous_window").is_some(),
                        "{label}: contiguous window report missing"
                    );
                    if bayesian {
                        let post = result.posterior.as_ref().expect("Bayesian posterior");
                        assert!(
                            post.prior_sensitivity.is_some(),
                            "{label}: prior sensitivity missing"
                        );
                    }
                } else {
                    assert!(
                        report("mediation.contiguous_window").is_none(),
                        "{label}: contiguous window outside the full suite"
                    );
                }
            }
        }
    }
}
