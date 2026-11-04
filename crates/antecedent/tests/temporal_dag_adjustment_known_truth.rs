//! Known-truth check of `temporal.linear.adjustment` (pulse and one-step sustained) and
//! `temporal.sequential.gcomp` (multi-step sustained) on a fixed `TemporalDag`, explicit and
//! accepted, under every validation suite.
//!
//! The pulse estimator adjusts for the treatment's confounders only: a second treatment lag that
//! is a cause of `y` but not a confounder is omitted, so its slope is the simple regression of
//! `y` on the first lag. That slope equals the coefficient only when the lags are uncorrelated,
//! which holds in no finite sample of a periodic series (hence a 6e-5 miss on the earlier
//! `0,1,0,-1` design). The pulse and one-step sustained cases therefore use `y` with a single
//! treatment lag (exact for any row set); the multi-step case keeps both lags, which
//! sequential g-computation adjusts for.
//!
//! The series is noise-free and exactly linear: `y_t = 1 + 2 t_{t-1} + 3 t_{t-2}` with the
//! binary treatment cycling `0, 0, 1`, so ordinary least squares is exact. The pulse at lag 1 has
//! effect `2`; the sustained policy over lags `-2..-1` has effect `2 + 3`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{AcceptedGraph, EstimatorId, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, ExecutionContext, Lag, TemporalEffectQuery, TemporalPolicy, VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_graph::{TemporalDag, ensure_lagged};

const TOLERANCE: f64 = 1e-8;

fn series(two_lag: bool) -> TimeSeriesData {
    let n = 320usize;
    let t = (0..n).map(|i| if i % 3 == 2 { 1.0 } else { 0.0 }).collect::<Vec<_>>();
    let second = if two_lag { 3.0 } else { 0.0 };
    let y = (0..n)
        .map(|i| {
            1.0 + 2.0 * i.checked_sub(1).map_or(0.0, |j| t[j])
                + second * i.checked_sub(2).map_or(0.0, |j| t[j])
        })
        .collect::<Vec<_>>();
    TimeSeriesData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice())], 1).unwrap()
}

fn graph(two_lag: bool) -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let t = VariableId::from_raw(0);
    let y = VariableId::from_raw(1);
    let t1 = ensure_lagged(&mut graph, t, Lag::from_raw(1)).unwrap();
    let t2 = two_lag.then(|| ensure_lagged(&mut graph, t, Lag::from_raw(2)).unwrap());
    let y0 = ensure_lagged(&mut graph, y, Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(t1, y0).unwrap();
    if let Some(t2) = t2 {
        graph.insert_directed(t2, y0).unwrap();
    }
    graph
}

#[derive(Clone, Copy, Debug)]
enum Kind {
    Pulse,
    OneStepSustained,
    MultiStepSustained,
}

fn query(kind: Kind) -> TemporalEffectQuery {
    let t = VariableId::from_raw(0);
    let y = VariableId::from_raw(1);
    match kind {
        Kind::Pulse => TemporalEffectQuery::pulse(t, y, 1.0).with_policy(TemporalPolicy::pulse(-1)),
        Kind::OneStepSustained => {
            TemporalEffectQuery::pulse(t, y, 1.0).with_policy(TemporalPolicy::sustained(-1, -1))
        }
        Kind::MultiStepSustained => TemporalEffectQuery::sustained(t, y, 0, 1.0)
            .with_policy(TemporalPolicy::sustained(-2, -1)),
    }
}

fn truth(kind: Kind) -> f64 {
    match kind {
        Kind::Pulse | Kind::OneStepSustained => 2.0,
        Kind::MultiStepSustained => 2.0 + 3.0,
    }
}

fn run(estimator: EstimatorId, kind: Kind, accepted: bool, suite: RefuteSuite) -> f64 {
    let two_lag = matches!(kind, Kind::MultiStepSustained);
    let base = Study::series(series(two_lag));
    let base = if accepted {
        base.graph(AcceptedGraph::temporal_dag(graph(two_lag)))
    } else {
        base.graph(graph(two_lag))
    };
    let result = base
        .query(CausalQuery::TemporalEffect(query(kind)))
        .estimator(estimator)
        .inference(InferenceMode::Frequentist)
        .refute(suite)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(415))
        .unwrap();
    result.estimate().as_effect().unwrap().ate
}

#[test]
fn linear_adjustment_returns_the_pulse_and_one_step_sustained_effect_on_a_fixed_dag() {
    for kind in [Kind::Pulse, Kind::OneStepSustained] {
        for accepted in [false, true] {
            for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
                let value = run(EstimatorId::TemporalLinearAdjustment, kind, accepted, suite);
                assert!(
                    (value - truth(kind)).abs() < TOLERANCE,
                    "{kind:?} accepted={accepted} {suite:?}: {value}"
                );
            }
        }
    }
}

#[test]
fn sequential_gcomp_returns_the_multi_step_sustained_effect_on_a_fixed_dag() {
    for accepted in [false, true] {
        for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
            let value = run(
                EstimatorId::TemporalSequentialGcomp,
                Kind::MultiStepSustained,
                accepted,
                suite,
            );
            assert!(
                (value - truth(Kind::MultiStepSustained)).abs() < TOLERANCE,
                "accepted={accepted} {suite:?}: {value}"
            );
        }
    }
}
