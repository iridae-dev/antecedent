//! Same-resample uncertainty agreement for the production OLS fast path.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{RefuteSuite, Study};
use antecedent_core::{AverageEffectQuery, CausalQuery, ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};

#[test]
fn public_pairs_bootstrap_preserves_seeded_uncertainty_and_failure_accounting() {
    let n = 1024;
    let z: Vec<_> = (0..n).map(|i| (f64::from(i) * 0.17).sin()).collect();
    let t: Vec<_> = (0..n).map(|i| f64::from(i % 2)).collect();
    let y: Vec<_> = (0..n)
        .map(|i| {
            let r = usize::try_from(i).unwrap();
            1.0 + 2.0 * t[r] + z[r] + 0.4 * (f64::from(i) * 0.31).cos()
        })
        .collect();
    let data = TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("y", y.as_slice()),
        ("z", z.as_slice()),
    ])
    .unwrap();
    let mut graph = Dag::with_variables(3);
    for (from, to) in [(2, 0), (2, 1), (0, 1)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    let study = Study::tabular(data)
        .graph(graph)
        .query(CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        )))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(199)
        .build()
        .unwrap();
    let reference = study.clone().run(&ExecutionContext::for_tests(31)).unwrap();
    for threads in [1, 4] {
        let actual = study.clone().run(&ExecutionContext::production(31, threads)).unwrap();
        let a = actual.estimate.as_effect().unwrap();
        let b = reference.estimate.as_effect().unwrap();
        assert!((a.ate - b.ate).abs() < 1e-12);
        assert!((a.se_bootstrap().unwrap() - b.se_bootstrap().unwrap()).abs() < 1e-10);
        assert_eq!(actual.performance.bootstrap_replicates_ok, Some(199));
        assert_eq!(
            actual.performance.bootstrap_replicates_ok,
            reference.performance.bootstrap_replicates_ok
        );
        assert_eq!(
            a.uncertainty.bootstrap.replicates_failed,
            b.uncertainty.bootstrap.replicates_failed
        );
        assert!(!actual.performance.early_stopped);
        assert!(!actual.performance.cancelled);
    }
}
