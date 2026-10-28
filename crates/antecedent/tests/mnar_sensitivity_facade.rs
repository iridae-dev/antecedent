//! The missing-not-at-random sensitivity facade: the table reader, the adjusted estimators
//! and the route from a built study's average-effect query.
//!
//! The table is the stratified example of the validate suite: a binary covariate `w`, 20
//! treated and 20 control rows, missing outcomes `NaN`. Every adjusted estimator equals
//! `1.5 + 0.25 delta` under a treated-only shift.
#![allow(clippy::float_cmp, reason = "exact grid endpoints and identical computations")]

use antecedent::analysis::mnar_sensitivity::{
    mnar_sensitivity_for_study, mnar_sensitivity_for_table, ArmLoading, MnarEstimator,
    MnarMechanism, MnarModel, MnarSensitivitySpec,
};
use antecedent::{RefuteSuite, Study};
use antecedent_core::{AverageEffectQuery, VariableId};
use antecedent_data::{MissingDeclaration, MissingPolicy, TabularData};

fn table() -> TabularData {
    let (mut t, mut y, mut w) = (Vec::new(), Vec::new(), Vec::new());
    // (arm, cell, observed values, repeats, missing rows)
    let cells: [(f64, f64, [f64; 3], usize, usize); 4] = [
        (1.0, 0.0, [1.0, 2.0, 3.0], 3, 3),
        (1.0, 1.0, [4.0, 5.0, 6.0], 2, 2),
        (0.0, 0.0, [0.0, 1.0, 2.0], 2, 2),
        (0.0, 1.0, [2.0, 3.0, 4.0], 3, 3),
    ];
    for (arm, cell, values, repeats, missing) in cells {
        for _ in 0..repeats {
            for v in values {
                t.push(arm);
                y.push(v);
                w.push(cell);
            }
        }
        for _ in 0..missing {
            t.push(arm);
            y.push(f64::NAN);
            w.push(cell);
        }
    }
    TabularData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice()), ("w", w.as_slice())])
        .unwrap()
}

fn spec(estimator: MnarEstimator) -> MnarSensitivitySpec {
    let mut spec = MnarSensitivitySpec::new(
        MnarMechanism {
            model: MnarModel::DeltaPatternMixture,
            loading: ArmLoading::treated_only(),
        },
        -10.0,
        10.0,
    );
    spec.estimator = estimator;
    spec.bootstrap_replicates = 40;
    spec
}

#[test]
fn the_table_reader_runs_every_estimator_on_named_columns() {
    let data = table();
    for (estimator, covariates) in [
        (MnarEstimator::RegressionAdjusted, vec!["w"]),
        (MnarEstimator::InverseProbabilityWeighting, vec!["w"]),
        (MnarEstimator::AugmentedInverseProbabilityWeighting, vec!["w"]),
    ] {
        let result =
            mnar_sensitivity_for_table(&data, "t", "y", &covariates, &spec(estimator)).unwrap();
        for point in &result.grid {
            assert!((point.effect - (1.5 + 0.25 * point.parameter)).abs() < 1e-9);
        }
        assert!(!result.assumption_range.is_confidence_interval);
    }
    let unadjusted =
        mnar_sensitivity_for_table(&data, "t", "y", &[], &spec(MnarEstimator::DifferenceInMeans))
            .unwrap();
    // Observed arm means: treated (9*2 + 6*5)/15 = 3.2, control (6*1 + 9*3)/15 = 2.2.
    assert!((unadjusted.mar.effect - 1.0).abs() < 1e-12);
}

#[test]
fn the_table_reader_refuses_an_unknown_column() {
    let err = mnar_sensitivity_for_table(
        &table(),
        "t",
        "missing_column",
        &[],
        &spec(MnarEstimator::DifferenceInMeans),
    )
    .unwrap_err();
    assert_eq!(err.detail(), "mnar_sensitivity.invalid_data");
}

#[test]
fn a_study_requests_the_sensitivity_of_its_own_average_effect() {
    let data = table();
    let mut graph = antecedent_graph::Dag::empty();
    let ids: Vec<_> = (0..3)
        .map(|i| {
            graph.add_node(antecedent_graph::NodeRef::Static(VariableId::from_raw(i))).unwrap()
        })
        .collect();
    graph.insert_directed(ids[2], ids[0]).unwrap();
    graph.insert_directed(ids[2], ids[1]).unwrap();
    graph.insert_directed(ids[0], ids[1]).unwrap();
    let study = Study::tabular(data)
        .query(AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1)))
        .graph(graph)
        .missing(MissingDeclaration::new(MissingPolicy::CompleteCase))
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    let result = mnar_sensitivity_for_study(
        &study,
        &[VariableId::from_raw(2)],
        &spec(MnarEstimator::RegressionAdjusted),
    )
    .unwrap();
    assert!((result.mar.effect - 1.5).abs() < 1e-9);
    let sign = result.tipping.iter().find(|t| t.label == "sign").unwrap();
    assert!((sign.negative.unwrap().parameter + 6.0).abs() < 1e-7);
}
