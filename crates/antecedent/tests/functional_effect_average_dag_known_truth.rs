//! Known-truth evidence for the explicitly selected `functional.effect` plug-in on a supplied
//! or accepted DAG: `AverageEffect x Dag x {explicit, accepted} x {Frequentist, Bayesian} x
//! {none, cheap, full}`.
//!
//! The expected value is enumerated here from the cell counts and never read from the library.
//! The confounder `z` drives both the treatment and the outcome. Outcome risks by `(z, t)`
//! are `.1` and `.5` for `z = 0` and `.5` and `.9` for `z = 1`, and `P(z = 1) = 1/2`, so the
//! adjustment formula gives
//!
//! ```text
//! E[Y | do(t=1)] - E[Y | do(t=0)] = sum_z P(z) (E[Y | t=1, z] - E[Y | t=0, z]) = 0.4
//! ```
//!
//! while the unadjusted contrast is confounded at `0.6`. The finite-discrete plug-in on the
//! retained table is exact for the Frequentist twin (tolerance 1e-9). The Bayesian twin reports
//! the mean of Dirichlet posterior draws around the same value; with 1 024 draws on 800 rows
//! the Monte-Carlo standard error of that mean is about 1e-3, so the stated band 0.01 is
//! several standard errors wide while still failing a 0.05 bias or the confounded 0.6.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{
    AcceptedGraph, BayesianConfig, EstimatorId, IdentifierId, InferenceMode, RefuteSuite,
    StructureSource, Study,
};
use antecedent_contract::SlotAvailability;
use antecedent_core::{AverageEffectQuery, CausalQuery, ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};

/// `(z, t, y, count)` cells, doubled so the table has 800 rows.
const CELLS: [(f64, f64, f64, usize); 8] = [
    (0.0, 0.0, 0.0, 270),
    (0.0, 0.0, 1.0, 30),
    (0.0, 1.0, 0.0, 50),
    (0.0, 1.0, 1.0, 50),
    (1.0, 0.0, 0.0, 50),
    (1.0, 0.0, 1.0, 50),
    (1.0, 1.0, 0.0, 30),
    (1.0, 1.0, 1.0, 270),
];

fn vid(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn fixture() -> (TabularData, Dag, AverageEffectQuery) {
    let mut t = Vec::new();
    let mut y = Vec::new();
    let mut z = Vec::new();
    for (zv, tv, yv, count) in CELLS {
        for _ in 0..count {
            z.push(zv);
            t.push(tv);
            y.push(yv);
        }
    }
    let data = TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("y", y.as_slice()),
        ("z", z.as_slice()),
    ])
    .unwrap();
    let mut graph = Dag::with_variables(3);
    graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    (data, graph, AverageEffectQuery::binary_ate(vid(0), vid(1)))
}

/// The back-door adjustment `sum_z P(z) (E[Y | t=1, z] - E[Y | t=0, z])`, enumerated from the
/// cell counts.
fn adjustment_truth() -> f64 {
    let total: f64 = CELLS.iter().map(|c| c.3 as f64).sum();
    let mut effect = 0.0;
    for z_level in [0.0, 1.0] {
        let p_z = CELLS.iter().filter(|c| c.0 == z_level).map(|c| c.3 as f64).sum::<f64>() / total;
        let mean_y = |t_level: f64| {
            let n: f64 =
                CELLS.iter().filter(|c| c.0 == z_level && c.1 == t_level).map(|c| c.3 as f64).sum();
            let ones: f64 = CELLS
                .iter()
                .filter(|c| c.0 == z_level && c.1 == t_level && c.2 == 1.0)
                .map(|c| c.3 as f64)
                .sum();
            ones / n
        };
        effect += p_z * (mean_y(1.0) - mean_y(0.0));
    }
    effect
}

/// The unadjusted contrast `E[Y | t=1] - E[Y | t=0]`.
fn naive_contrast() -> f64 {
    let mean_y = |t_level: f64| {
        let n: f64 = CELLS.iter().filter(|c| c.1 == t_level).map(|c| c.3 as f64).sum();
        let ones: f64 =
            CELLS.iter().filter(|c| c.1 == t_level && c.2 == 1.0).map(|c| c.3 as f64).sum();
        ones / n
    };
    mean_y(1.0) - mean_y(0.0)
}

fn verify_coordinate(coordinate: &str, accepted: bool, bayesian: bool, validation: &str) {
    let truth = adjustment_truth();
    assert!((truth - 0.4).abs() < 1e-12, "enumerated truth {truth}");
    assert!((naive_contrast() - 0.6).abs() < 1e-12, "the design must be confounded");

    let ctx = ExecutionContext::for_tests(2_203);
    let (data, graph, query) = fixture();
    let suite = match validation {
        "none" => RefuteSuite::None,
        "cheap" => RefuteSuite::Cheap,
        "full" => RefuteSuite::Full,
        other => panic!("unexpected validation suite {other}"),
    };
    let base = Study::tabular(data.clone())
        .query(query.clone())
        .identifier(IdentifierId::GeneralId)
        .estimator(EstimatorId::FunctionalEffect)
        .inference(if bayesian {
            InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(1_024))
        } else {
            InferenceMode::Frequentist
        })
        .refute(suite)
        .bootstrap_replicates(0);
    let builder = if accepted { base.graph(AcceptedGraph::from(graph)) } else { base.graph(graph) };
    let study = builder.clone().build().unwrap_or_else(|error| panic!("{coordinate}: {error}"));
    let prepared = study.prepare(&ctx).unwrap_or_else(|error| panic!("{coordinate}: {error}"));
    drop(builder);
    drop(study);

    assert_eq!(prepared.query(), &CausalQuery::AverageEffect(query));
    assert_eq!(
        prepared.structure_source(),
        if accepted { StructureSource::Accepted } else { StructureSource::Explicit },
        "{coordinate} structure source"
    );
    assert_eq!(prepared.support_status(), Some(antecedent::CellStatus::Supported), "{coordinate}");
    let contract = prepared.contract().unwrap();
    assert_eq!(contract.query_kind.as_ref(), "AverageEffect", "{coordinate}");
    assert_eq!(contract.identifier.as_deref(), Some("general.id"), "{coordinate}");
    assert_eq!(contract.estimator.as_deref(), Some("functional.effect"), "{coordinate}");
    assert_eq!(contract.inference.as_ref(), if bayesian { "bayesian" } else { "frequentist" });
    match &contract.reasoning.support {
        SlotAvailability::Available(slot) => assert_eq!(
            slot.matrix_coordinate.as_deref(),
            Some(coordinate),
            "prepared handle must bind this exact coordinate"
        ),
        other => panic!("{coordinate} support unavailable: {other:?}"),
    }
    assert_eq!(prepared.plan().logical.record.estimator.as_deref(), Some("functional.effect"));
    assert_eq!(
        prepared.plan().logical.record.validation_suite.as_deref(),
        match validation {
            "none" => None,
            "cheap" => Some("overlap+evalue"),
            _ => Some("validation.full"),
        },
        "{coordinate} validation choice"
    );

    let result = prepared.estimate(&data, &ctx).unwrap_or_else(|e| panic!("{coordinate}: {e}"));
    let tolerance = if bayesian { 0.01 } else { 1e-9 };
    assert!(
        (result.effect() - truth).abs() < tolerance,
        "{coordinate}: estimate {} vs enumerated {truth}",
        result.effect()
    );
    if bayesian {
        assert!(result.posterior.is_some(), "{coordinate} posterior missing");
    } else {
        assert!(result.posterior.is_none(), "{coordinate} unexpected posterior");
    }
    assert_eq!(result.refutations.is_empty(), validation == "none", "{coordinate} refutations");
}

macro_rules! coordinate_test {
    ($name:ident, $coordinate:literal, $accepted:literal, $bayesian:literal, $validation:literal) => {
        #[test]
        fn $name() {
            verify_coordinate($coordinate, $accepted, $bayesian, $validation);
        }
    };
}

coordinate_test!(
    dag_explicit_frequentist_none,
    "AverageEffect:Dag:explicit:Frequentist:none",
    false,
    false,
    "none"
);
coordinate_test!(
    dag_explicit_frequentist_cheap,
    "AverageEffect:Dag:explicit:Frequentist:cheap",
    false,
    false,
    "cheap"
);
coordinate_test!(
    dag_explicit_frequentist_full,
    "AverageEffect:Dag:explicit:Frequentist:full",
    false,
    false,
    "full"
);
coordinate_test!(
    dag_explicit_bayesian_none,
    "AverageEffect:Dag:explicit:Bayesian:none",
    false,
    true,
    "none"
);
coordinate_test!(
    dag_explicit_bayesian_cheap,
    "AverageEffect:Dag:explicit:Bayesian:cheap",
    false,
    true,
    "cheap"
);
coordinate_test!(
    dag_explicit_bayesian_full,
    "AverageEffect:Dag:explicit:Bayesian:full",
    false,
    true,
    "full"
);
coordinate_test!(
    dag_accepted_frequentist_none,
    "AverageEffect:Dag:accepted:Frequentist:none",
    true,
    false,
    "none"
);
coordinate_test!(
    dag_accepted_frequentist_cheap,
    "AverageEffect:Dag:accepted:Frequentist:cheap",
    true,
    false,
    "cheap"
);
coordinate_test!(
    dag_accepted_frequentist_full,
    "AverageEffect:Dag:accepted:Frequentist:full",
    true,
    false,
    "full"
);
coordinate_test!(
    dag_accepted_bayesian_none,
    "AverageEffect:Dag:accepted:Bayesian:none",
    true,
    true,
    "none"
);
coordinate_test!(
    dag_accepted_bayesian_cheap,
    "AverageEffect:Dag:accepted:Bayesian:cheap",
    true,
    true,
    "cheap"
);
coordinate_test!(
    dag_accepted_bayesian_full,
    "AverageEffect:Dag:accepted:Bayesian:full",
    true,
    true,
    "full"
);
