//! Boundaries of the temporal and static routes through the facade: degenerate temporal designs (a
//! constant treatment, a constant outcome and an exactly collinear adjustment column, for the pulse
//! and the single-step sustained window), a temporal graph given to a static effect estimator, and a
//! three-level treatment given to the binary average-effect query.
//!
//! The known answers: a constant treatment leaves the regression of the outcome on the lagged
//! treatment rank deficient and is refused with that detail rather than given a number; a constant
//! outcome has effect exactly zero (the regression slope of a constant is zero); and an adjustment
//! column that is an exact affine copy of another makes the adjustment design rank deficient and
//! is refused.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, ExecutionContext, Lag, TemporalEffectQuery, TemporalPolicy, VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_graph::{TemporalDag, ensure_lagged};

const N: usize = 160;

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn wiggle(i: usize, rate: f64) -> f64 {
    (i as f64 * rate).sin()
}

/// `t@-1 -> y@0`, with each listed confounder a parent of both.
fn graph(confounders: &[u32]) -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let t1 = ensure_lagged(&mut graph, v(0), Lag::from_raw(1)).unwrap();
    let y0 = ensure_lagged(&mut graph, v(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(t1, y0).unwrap();
    for &z in confounders {
        let zl = ensure_lagged(&mut graph, v(z), Lag::from_raw(1)).unwrap();
        graph.insert_directed(zl, t1).unwrap();
        graph.insert_directed(zl, y0).unwrap();
    }
    graph
}

fn run(
    columns: &[(&str, &[f64])],
    confounders: &[u32],
    policy: TemporalPolicy,
) -> Result<f64, String> {
    let data = TimeSeriesData::from_f64_columns(columns.iter().copied(), 1).unwrap();
    let query = TemporalEffectQuery::pulse(v(0), v(1), 1.0).with_policy(policy);
    let result = Study::series(data)
        .graph(graph(confounders))
        .query(CausalQuery::TemporalEffect(query))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .map_err(|e| e.to_string())?
        .run(&ExecutionContext::for_tests(5))
        .map_err(|e| e.to_string())?;
    Ok(result.estimate().as_effect().unwrap().ate)
}

enum Expect {
    Zero,
    Refused(&'static str),
}

#[test]
fn degenerate_treatment_outcome_and_adjustment_columns_follow_their_known_answers() {
    let varying: Vec<f64> = (0..N).map(|i| wiggle(i, 0.7)).collect();
    let confounder: Vec<f64> = (0..N).map(|i| wiggle(i, 0.31)).collect();
    let copy: Vec<f64> = confounder.iter().map(|z| 2.0 * z + 1.0).collect();
    let outcome: Vec<f64> =
        (0..N).map(|i| 2.0 + 1.5 * wiggle(i.saturating_sub(1), 0.7) + wiggle(i, 0.13)).collect();
    let constant = vec![1.0; N];
    let flat_outcome = vec![3.0; N];

    let policies: [(&str, fn() -> TemporalPolicy); 2] = [
        ("pulse", || TemporalPolicy::pulse(-1)),
        ("single-step sustained", || TemporalPolicy::sustained(-1, -1)),
    ];
    for (name, policy) in policies {
        // Table: label, columns, confounder ids, expected answer.
        let cases: Vec<(&str, Vec<(&str, &[f64])>, Vec<u32>, Expect)> = vec![
            (
                "constant treatment",
                vec![("t", &constant), ("y", &outcome)],
                vec![],
                Expect::Refused("rank deficient"),
            ),
            (
                "constant treatment and outcome",
                vec![("t", &constant), ("y", &flat_outcome)],
                vec![],
                Expect::Refused("rank deficient"),
            ),
            ("constant outcome", vec![("t", &varying), ("y", &flat_outcome)], vec![], Expect::Zero),
            (
                "adjustment column that is an affine copy of another",
                vec![("t", &varying), ("y", &outcome), ("z", &confounder), ("z2", &copy)],
                vec![2, 3],
                Expect::Refused("rank deficient"),
            ),
        ];
        for (label, columns, confounders, expect) in cases {
            let got = run(&columns, &confounders, policy());
            match expect {
                Expect::Zero => {
                    let ate = got.unwrap_or_else(|e| panic!("{name} / {label}: {e}"));
                    assert!(ate.abs() < 1e-12, "{name} / {label}: ate={ate}");
                }
                Expect::Refused(detail) => {
                    let error = got.expect_err(&format!("{name} / {label} must be refused"));
                    assert!(error.contains(detail), "{name} / {label}: {error}");
                }
            }
        }
    }
}

/// A static effect estimator reads rows of a table under a static graph. Given a temporal graph it
/// is refused with the typed `unsupported data/graph/query combination` rather than reading the
/// lagged nodes as static variables.
#[test]
fn static_effect_estimators_refuse_a_temporal_graph() {
    use antecedent::EstimatorId;
    use antecedent_core::AverageEffectQuery;
    use antecedent_data::TabularData;

    let t: Vec<f64> = (0..N).map(|i| wiggle(i, 0.7)).collect();
    let z: Vec<f64> = (0..N).map(|i| wiggle(i, 0.31)).collect();
    let y: Vec<f64> = (0..N).map(|i| 2.0 * t[i] + z[i] + wiggle(i, 0.13)).collect();
    let estimators = [
        None,
        Some(EstimatorId::LinearAdjustmentAte),
        Some(EstimatorId::GlmAdjustment),
        Some(EstimatorId::Aipw),
        Some(EstimatorId::PropensityWeighting),
        Some(EstimatorId::PropensityMatching),
        Some(EstimatorId::PropensityStratification),
        Some(EstimatorId::DistanceMatching),
    ];
    for estimator in estimators {
        let data = TabularData::from_f64_columns([
            ("t", t.as_slice()),
            ("y", y.as_slice()),
            ("z", z.as_slice()),
        ])
        .unwrap();
        let builder = Study::tabular(data)
            .graph(graph(&[2]))
            .query(AverageEffectQuery::binary_ate(v(0), v(1)));
        let builder = match estimator {
            Some(id) => builder.estimator(id),
            None => builder,
        };
        let outcome = builder
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .map_err(|e| e.to_string())
            .and_then(|study| {
                study.run(&ExecutionContext::for_tests(5)).map_err(|e| e.to_string())
            });
        let error = outcome.expect_err(&format!("{estimator:?} must refuse a temporal graph"));
        assert!(
            error.contains("unsupported data/graph/query combination"),
            "{estimator:?}: {error}"
        );
    }
}

/// The binary average-effect query reads a treatment at two declared levels. A treatment with a
/// further level is refused with `treatment_level_undeclared`, naming the observed levels and their
/// row counts, on every graph class and with or without refuters, rather than dropping the
/// third level's rows or collapsing it into a neighbouring level.
#[test]
fn a_three_level_treatment_is_refused_by_name_on_every_graph_class_with_or_without_refuters() {
    use antecedent_core::AverageEffectQuery;
    use antecedent_data::TabularData;
    use antecedent_graph::{Admg, Cpdag, Dag, DenseNodeId};

    let t: Vec<f64> = (0..N).map(|i| (i % 3) as f64).collect();
    let z: Vec<f64> = (0..N).map(|i| wiggle(i, 0.31)).collect();
    let y: Vec<f64> = (0..N).map(|i| 2.0 * t[i] + z[i] + wiggle(i, 0.13)).collect();
    let edges = [(2_u32, 0_u32), (2, 1), (0, 1)];
    let node = DenseNodeId::from_raw;
    for class in ["dag", "admg", "cpdag"] {
        for refute in [RefuteSuite::None, RefuteSuite::Cheap] {
            let data = TabularData::from_f64_columns([
                ("t", t.as_slice()),
                ("y", y.as_slice()),
                ("z", z.as_slice()),
            ])
            .unwrap();
            let builder = Study::tabular(data).query(AverageEffectQuery::binary_ate(v(0), v(1)));
            let builder = match class {
                "dag" => {
                    let mut graph = Dag::with_variables(3);
                    for (from, to) in edges {
                        graph.insert_directed(node(from), node(to)).unwrap();
                    }
                    builder.graph(graph)
                }
                "admg" => {
                    let mut graph = Admg::with_variables(3);
                    for (from, to) in edges {
                        graph.insert_directed(node(from), node(to)).unwrap();
                    }
                    builder.graph(graph)
                }
                _ => {
                    let mut graph = Cpdag::with_variables(3);
                    for (from, to) in edges {
                        graph.insert_directed(node(from), node(to)).unwrap();
                    }
                    builder.graph(graph)
                }
            };
            let error = builder
                .refute(refute)
                .bootstrap_replicates(0)
                .build()
                .map_err(|e| e.to_string())
                .and_then(|study| {
                    study.run(&ExecutionContext::for_tests(5)).map_err(|e| e.to_string())
                })
                .expect_err(&format!("{class}: a three-level treatment must be refused"));
            assert!(error.contains("treatment_level_undeclared"), "{class}: {error}");
            assert!(
                error.contains("0 (54 rows), 1 (53 rows), 2 (53 rows)"),
                "{class}: the refusal must name the observed levels and counts: {error}"
            );
        }
    }
}
