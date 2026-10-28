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
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{CausalError, RefuteSuite, Study, StudyResult};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, CounterfactualQuery, DeclaredSamplingDesign,
    ExecutionContext, Intervention, Value, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};

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
