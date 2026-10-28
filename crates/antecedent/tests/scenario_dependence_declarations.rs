//! A declared sampling design is held by the routes that resample, or the route is refused.
//!
//! The structural model is `y = 3 a + 4 m + z` with `m = 2 a + noise` and `a` driven by the
//! confounder `z`, so the mean unit effect of setting `a` to 1 against 0 is 11. The counterfactual
//! route refits its mechanisms by a bootstrap over independent subjects: declared as independent
//! subject trajectories it runs and recovers 11, while a clustered panel or a dependent time
//! series needs a whole-cluster or block resample the route does not have and is refused with
//! `option_not_applicable` instead of being resampled by row. The average effect, which does have
//! a resample per design, refuses a row-by-row bootstrap under each declared dependence and
//! names the draw the design allows.
//!
//! With the bootstrap off the only uncertainty is the analytic error, and the study-level design
//! carries no cluster labels or lag of its own. An average effect through a configured regression
//! estimator reads the declared panel or series through its cluster or Newey-West error, which
//! is the closed-form sandwich of the regression scores; every route whose analytic error treats
//! rows as independent (the default estimator, the latent-variable graph, the conditional
//! effect, the average derivative and the intervention response) refuses a clustered panel or a
//! dependent time series instead of publishing that too-small error, naming the estimator
//! declaration that reads it.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{CausalError, RefuteSuite, Study, StudyResult};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ConditionalEffectQuery, CounterfactualQuery,
    DeclaredSamplingDesign, DerivativeWeighting, ExecutionContext, Intervention, ResponseFunctional,
    ResponseQuery, Value, VariableId,
};
use antecedent_data::TabularData;
use antecedent_estimate::{AnalyticSeKind, LinearAdjustmentAte};
use antecedent_graph::{Admg, Dag, DenseNodeId};

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

const N: usize = 300;

fn data() -> TabularData {
    let at = |i: usize| i as f64;
    let z: Vec<f64> = (0..N).map(|i| (at(i) * 0.37).sin()).collect();
    let a: Vec<f64> = (0..N).map(|i| z[i] + (at(i) * 0.71).cos()).collect();
    let m: Vec<f64> = (0..N).map(|i| 2.0 * a[i] + (at(i) * 1.13).cos()).collect();
    let y: Vec<f64> =
        (0..N).map(|i| 3.0 * a[i] + 4.0 * m[i] + z[i] + 0.1 * (at(i) * 0.31).sin()).collect();
    TabularData::from_f64_columns([
        ("a", a.as_slice()),
        ("m", m.as_slice()),
        ("y", y.as_slice()),
        ("z", z.as_slice()),
    ])
    .unwrap()
}

fn dag() -> Dag {
    let mut graph = Dag::with_variables(4);
    for (from, to) in [(3, 0), (3, 2), (0, 1), (1, 2), (0, 2)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    graph
}

fn counterfactual() -> CausalQuery {
    CausalQuery::Counterfactual(
        CounterfactualQuery::new(v(2), Arc::from([Intervention::set(v(0), Value::f64(1.0))]))
            .with_control_level(0.0),
    )
}

fn run(query: CausalQuery, design: DeclaredSamplingDesign) -> Result<StudyResult, CausalError> {
    Study::tabular(data())
        .graph(dag())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(20)
        .sampling_design(design)
        .query(query)
        .build()
        .and_then(|study| study.run(&ExecutionContext::for_tests(5)))
}

#[test]
fn the_counterfactual_runs_under_independent_subjects_and_refuses_clustered_and_serial_designs() {
    let effect = run(counterfactual(), DeclaredSamplingDesign::IndependentSubjectTrajectories)
        .unwrap()
        .counterfactual()
        .unwrap()
        .mean_ite;
    assert!((effect - 11.0).abs() < 0.3, "{effect}");
    for design in [DeclaredSamplingDesign::ClusteredPanel, DeclaredSamplingDesign::DependentTimeSeries]
    {
        let error = run(counterfactual(), design)
            .err()
            .unwrap_or_else(|| panic!("{design:?}: the counterfactual resampled by row"));
        assert_eq!(error.reason_code(), Some("option_not_applicable"), "{design:?}: {error}");
        let text = error.to_string();
        assert!(text.contains("resamples independent subjects"), "{design:?}: {text}");
        assert!(text.contains("whole-cluster or block resample"), "{design:?}: {text}");
    }
}

#[test]
fn the_average_effect_refuses_a_row_bootstrap_under_each_declared_dependence_and_names_the_draw() {
    let query = || CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(v(0), v(2)));
    for (design, name, allowed) in [
        (DeclaredSamplingDesign::ClusteredPanel, "clustered_panel", "whole_cluster"),
        (DeclaredSamplingDesign::DependentTimeSeries, "dependent_time_series", "contiguous_block"),
        (
            DeclaredSamplingDesign::IndependentSubjectTrajectories,
            "independent_subject_trajectories",
            "row_subject_iid",
        ),
    ] {
        let text = run(query(), design)
            .err()
            .unwrap_or_else(|| panic!("{name}: resampled rows as if independent"))
            .to_string();
        assert!(text.contains(&format!("{name} does not allow row_iid resampling")), "{name}: {text}");
        assert!(text.contains(&format!("{name} allows {allowed} resampling")), "{name}: {text}");
    }
}

fn admg() -> Admg {
    let mut graph = Admg::with_variables(4);
    for (from, to) in [(3, 0), (3, 2), (0, 1), (1, 2), (0, 2)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    graph
}

fn point_routes() -> Vec<(&'static str, CausalQuery)> {
    vec![
        ("average effect", CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(v(0), v(2)))),
        (
            "conditional effect",
            CausalQuery::ConditionalEffect(
                ConditionalEffectQuery::try_new(
                    AverageEffectQuery::binary_ate(v(0), v(2)).with_effect_modifiers(vec![v(3)]),
                )
                .unwrap(),
            ),
        ),
        (
            "average derivative",
            CausalQuery::Response(ResponseQuery::new(ResponseFunctional::AverageDerivative {
                outcome: v(2),
                treatment: v(0),
                weighting: DerivativeWeighting::Observed,
            })),
        ),
        (
            "intervention response",
            CausalQuery::Response(ResponseQuery::new(ResponseFunctional::InterventionResponse {
                outcome: v(2),
                interventions: Arc::from([Intervention::set(v(0), Value::f64(0.25))]),
            })),
        ),
    ]
}

fn point_run(
    query: &CausalQuery,
    design: DeclaredSamplingDesign,
    latent: bool,
    estimator: Option<LinearAdjustmentAte>,
) -> Result<StudyResult, CausalError> {
    let builder = Study::tabular(data());
    let builder = if latent { builder.graph(admg()) } else { builder.graph(dag()) };
    let builder = match estimator {
        Some(estimator) => builder.estimator(estimator),
        None => builder.refute(RefuteSuite::None).bootstrap_replicates(0),
    };
    builder
        .sampling_design(design)
        .query(query.clone())
        .build()
        .and_then(|study| study.run(&ExecutionContext::for_tests(5)))
}

#[test]
fn a_declared_panel_or_series_is_refused_where_the_analytic_error_reads_rows_as_independent() {
    for (design, phrase) in [
        (DeclaredSamplingDesign::ClusteredPanel, "clustered-panel"),
        (DeclaredSamplingDesign::DependentTimeSeries, "dependent-time-series"),
    ] {
        for latent in [false, true] {
            for (name, query) in point_routes() {
                let error = point_run(&query, design, latent, None).err().unwrap_or_else(|| {
                    panic!("{name} (latent {latent}) {design:?}: an error that ignores the design")
                });
                assert_eq!(error.reason_code(), Some("option_not_applicable"), "{name}: {error}");
                let text = error.to_string();
                assert!(text.contains(phrase), "{name}: {text}");
                assert!(text.contains("Next attempt"), "{name}: {text}");
            }
        }
    }
    // Independent subject trajectories are what the rows-as-independent error describes.
    for latent in [false, true] {
        for (name, query) in point_routes() {
            point_run(&query, DeclaredSamplingDesign::IndependentSubjectTrajectories, latent, None)
                .unwrap_or_else(|error| panic!("{name} (latent {latent}): {error}"));
        }
    }
}

#[test]
fn an_error_kind_that_does_not_read_the_declared_design_is_refused_for_a_configured_estimator() {
    let query = &point_routes()[0].1;
    let clustered = DeclaredSamplingDesign::ClusteredPanel;
    let serial = DeclaredSamplingDesign::DependentTimeSeries;
    let base = || LinearAdjustmentAte::new().with_bootstrap_replicates(0);
    // A panel needs a cluster kind and a series a Newey-West kind; the other pairing is refused.
    for (design, estimator) in [
        (clustered, base().with_se_kind(AnalyticSeKind::Hc1)),
        (clustered, base().with_se_kind(AnalyticSeKind::NeweyWest { lag: 2 })),
        (serial, base().with_se_kind(AnalyticSeKind::Hc1)),
        (serial, base().with_se_kind(AnalyticSeKind::Cluster).with_cluster_ids(vec![0; N])),
    ] {
        let error = point_run(query, design, false, Some(estimator))
            .err()
            .unwrap_or_else(|| panic!("{design:?}: an error kind that ignores the design"));
        assert_eq!(error.reason_code(), Some("option_not_applicable"), "{design:?}: {error}");
        assert!(error.to_string().contains("Next attempt"), "{design:?}: {error}");
    }
}

fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let k = b.len();
    for col in 0..k {
        let pivot = (col..k).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs())).unwrap();
        a.swap(col, pivot);
        b.swap(col, pivot);
        for row in col + 1..k {
            let f = a[row][col] / a[col][col];
            for c in col..k {
                a[row][c] -= f * a[col][c];
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = vec![0.0; k];
    for row in (0..k).rev() {
        let tail: f64 = (row + 1..k).map(|c| a[row][c] * x[c]).sum();
        x[row] = (b[row] - tail) / a[row][row];
    }
    x
}

/// The standard error of the coefficient of `a` in `y ~ 1 + a + z`, from the definition: scores
/// `x_i e_i`, summed within `groups` (CR1) or Bartlett-weighted over `lag` (no small-sample
/// factor for Newey-West).
fn sandwich(groups: Option<&[u32]>, lag: Option<usize>) -> f64 {
    let table = data();
    let column = |raw: u32| -> Vec<f64> {
        let antecedent_data::ColumnView::Float64(c) =
            antecedent_data::TableView::column(&table, v(raw)).unwrap()
        else {
            panic!("a continuous column");
        };
        c.values.as_slice().to_vec()
    };
    let (a, y, z) = (column(0), column(2), column(3));
    let x: Vec<[f64; 3]> = (0..N).map(|i| [1.0, a[i], z[i]]).collect();
    let mut xtx = vec![vec![0.0; 3]; 3];
    let mut xty = vec![0.0; 3];
    for i in 0..N {
        for p in 0..3 {
            xty[p] += x[i][p] * y[i];
            for q in 0..3 {
                xtx[p][q] += x[i][p] * x[i][q];
            }
        }
    }
    let beta = solve(xtx.clone(), xty);
    let scores: Vec<[f64; 3]> = (0..N)
        .map(|i| {
            let e = y[i] - (0..3).map(|k| x[i][k] * beta[k]).sum::<f64>();
            [x[i][0] * e, x[i][1] * e, x[i][2] * e]
        })
        .collect();
    let mut omega = [[0.0_f64; 3]; 3];
    if let Some(groups) = groups {
        let mut cells = std::collections::BTreeMap::<u32, [f64; 3]>::new();
        for (s, g) in scores.iter().zip(groups) {
            let cell = cells.entry(*g).or_insert([0.0; 3]);
            for k in 0..3 {
                cell[k] += s[k];
            }
        }
        for cell in cells.values() {
            for p in 0..3 {
                for q in 0..3 {
                    omega[p][q] += cell[p] * cell[q];
                }
            }
        }
        let g = cells.len() as f64;
        let factor = g / (g - 1.0) * (N as f64 - 1.0) / (N as f64 - 3.0);
        for row in &mut omega {
            for value in row {
                *value *= factor;
            }
        }
    } else if let Some(lag) = lag {
        for l in 0..=lag {
            let weight = if l == 0 { 1.0 } else { 1.0 - l as f64 / (lag as f64 + 1.0) };
            for i in l..N {
                for p in 0..3 {
                    for q in 0..3 {
                        let term = weight * scores[i][p] * scores[i - l][q];
                        omega[p][q] += term;
                        if l > 0 {
                            omega[q][p] += term;
                        }
                    }
                }
            }
        }
    }
    let inverse: Vec<Vec<f64>> = (0..3)
        .map(|c| {
            let mut e = vec![0.0; 3];
            e[c] = 1.0;
            solve(xtx.clone(), e)
        })
        .collect();
    let variance: f64 = (0..3)
        .map(|p| (0..3).map(|q| inverse[1][p] * omega[p][q] * inverse[q][1]).sum::<f64>())
        .sum();
    variance.sqrt()
}

#[test]
fn a_declared_panel_or_series_is_read_by_the_cluster_and_newey_west_errors_of_the_average_effect() {
    let query = &point_routes()[0].1;
    let clusters: Vec<u32> = (0..N).map(|i| (i / 10) as u32).collect();
    let se_of = |design: DeclaredSamplingDesign, estimator: LinearAdjustmentAte| {
        let result = point_run(query, design, false, Some(estimator)).unwrap();
        result.estimate().as_effect().unwrap().se_analytic()
    };
    let base = || LinearAdjustmentAte::new().with_bootstrap_replicates(0);
    let independent = se_of(
        DeclaredSamplingDesign::IndependentSubjectTrajectories,
        base().with_se_kind(AnalyticSeKind::Hc1),
    );
    let panel = se_of(
        DeclaredSamplingDesign::ClusteredPanel,
        base().with_se_kind(AnalyticSeKind::Cluster).with_cluster_ids(clusters.clone()),
    );
    let want = sandwich(Some(&clusters), None);
    assert!((panel - want).abs() < 1e-9 * (1.0 + want), "cluster {panel} vs {want}");
    assert!((panel - independent).abs() > 1e-3 * independent, "clustering must matter");
    for lag in [1_usize, 4] {
        let series = se_of(
            DeclaredSamplingDesign::DependentTimeSeries,
            base().with_se_kind(AnalyticSeKind::NeweyWest { lag }),
        );
        let want = sandwich(None, Some(lag));
        assert!((series - want).abs() < 1e-9 * (1.0 + want), "newey-west {lag}: {series} vs {want}");
    }
}
