//! A mean-curve response reads or refuses a declared dependence, table weights and a bootstrap.
//!
//! The Kennedy curve publishes an influence-function interval over independent rows. A study that
//! declares a clustered panel or a dependent series, or asks for bootstrap replicates, must not
//! receive that interval as though the declaration had been read.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{CausalError, RefuteSuite, Study, StudyResult};
use antecedent_core::{
    CausalQuery, ContinuousDomain, DeclaredSamplingDesign, ExecutionContext, GridSpec,
    ResponseFunctional, ResponseQuery, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};

fn run(
    design: Option<DeclaredSamplingDesign>,
    replicates: u32,
    weights: bool,
) -> Result<StudyResult, CausalError> {
    let n = 80usize;
    let x: Vec<f64> = (0..n).map(|i| ((i * 37) % 80) as f64 / 40.0 - 1.0).collect();
    let a: Vec<f64> = (0..n).map(|i| 0.5 * x[i] + ((i * 53) % 80) as f64 / 160.0 - 0.25).collect();
    let y: Vec<f64> =
        (0..n).map(|i| 1.5 * a[i] + 0.8 * x[i] + ((i * 29) % 80) as f64 / 160.0).collect();
    let data = TabularData::from_f64_columns([
        ("a", a.as_slice()),
        ("y", y.as_slice()),
        ("x", x.as_slice()),
    ])
    .unwrap();
    let data = if weights {
        let w: Vec<f64> = (0..n).map(|i| 1.0 + (i % 3) as f64).collect();
        data.with_weights(Arc::from(w)).unwrap()
    } else {
        data
    };
    let mut graph = Dag::with_variables(3);
    for (from, to) in [(2u32, 0u32), (2, 1), (0, 1)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(Arc::from([-0.2, 0.0, 0.2])),
        ),
    });
    let builder = Study::tabular(data)
        .graph(graph)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(replicates)
        .query(CausalQuery::Response(query));
    let builder = match design {
        Some(design) => builder.sampling_design(design),
        None => builder,
    };
    builder.build().and_then(|study| study.run(&ExecutionContext::for_tests(5)))
}

fn assert_refused(label: &str, result: Result<StudyResult, CausalError>, phrase: &str) {
    let error = result
        .err()
        .unwrap_or_else(|| panic!("{label}: published an interval that ignores the declaration"));
    assert_eq!(error.reason_code(), Some("option_not_applicable"), "{label}: {error}");
    let text = error.to_string();
    assert!(text.contains(phrase), "{label}: {text}");
    assert!(text.contains("Next attempt"), "{label}: {text}");
}

#[test]
fn a_declared_dependence_is_refused_with_or_without_a_bootstrap_setting() {
    run(None, 0, false).unwrap();
    run(None, 20, false).unwrap();
    run(Some(DeclaredSamplingDesign::IndependentSubjectTrajectories), 20, false).unwrap();
    for design in
        [DeclaredSamplingDesign::ClusteredPanel, DeclaredSamplingDesign::DependentTimeSeries]
    {
        // Bootstrap off: the analytic error is the one that treats rows as independent.
        assert_refused(
            &format!("{design:?}"),
            run(Some(design), 0, false),
            "treats rows as independent",
        );
        // Bootstrap on: the curve does not bootstrap, so the setting reads nothing of the design.
        assert_refused(
            &format!("{design:?} with replicates"),
            run(Some(design), 20, false),
            "do not bootstrap",
        );
    }
    let error = run(None, 0, true).err().expect("observation weights were ignored");
    assert_eq!(error.reason_code(), Some("option_not_applicable"), "{error}");
    assert!(error.to_string().contains("sampling_weights.route_unsupported"), "{error}");
}
