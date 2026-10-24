// SPDX-License-Identifier: MIT OR Apache-2.0
//! Nuisance-model routes whose resident design does not fit stream their rows and equal the
//! resident estimates.
use crate::{EstimatorId, IdentifierId, RefuteSuite, Study};
use antecedent_core::{AverageEffectQuery, ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};

const ROWS: usize = 100_000;
/// Below the resident design of either route (`ROWS` rows of two or three float64 columns),
/// above the batch window of the streamed reduction.
const LIMIT: u64 = 1_500_000;

fn noise(i: usize, k: usize) -> f64 {
    (((i * 31 + k * 977) as f64 * 12.9898).sin() * 43758.5453).fract() - 0.5
}

fn dag(edges: &[(u32, u32)], variables: u32) -> Dag {
    let mut graph = Dag::with_variables(variables);
    for &(from, to) in edges {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    graph
}

fn bounded() -> ExecutionContext {
    let mut ctx = ExecutionContext::for_tests(1);
    ctx.set_memory_limit(Some(LIMIT));
    ctx
}

fn close(left: f64, right: f64, tolerance: f64) -> bool {
    (left - right).abs() <= tolerance * left.abs().max(right.abs()).max(1.0)
}

#[test]
fn a_two_sls_design_over_the_resident_budget_streams_and_equals_the_resident_fit() {
    // z -> t -> y, with an unobserved u confounding t and y.
    let z: Vec<f64> = (0..ROWS).map(|i| noise(i, 1) * 3.0).collect();
    let u: Vec<f64> = (0..ROWS).map(|i| noise(i, 2)).collect();
    let t: Vec<f64> = (0..ROWS).map(|i| 0.8 * z[i] + u[i]).collect();
    let y: Vec<f64> = (0..ROWS).map(|i| 2.0 * t[i] + 1.5 * u[i] + 0.2 * noise(i, 3)).collect();
    let data = TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("y", y.as_slice()),
        ("z", z.as_slice()),
    ])
    .unwrap();
    let study = Study::tabular(data)
        .graph(dag(&[(2, 0), (0, 1)], 3))
        .query(AverageEffectQuery::with_levels(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            0.0,
            1.0,
        ))
        .identifier(IdentifierId::Iv)
        .estimator(EstimatorId::Iv2Sls)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();

    let resident_ctx = ExecutionContext::for_tests(1);
    let resident_plan = study.compile(&resident_ctx).unwrap();
    assert!(!resident_plan.streams_design(), "without a limit the design is resident");
    let resident = study.run(&resident_ctx).unwrap();

    let ctx = bounded();
    let plan = study.compile(&ctx).unwrap();
    assert!(plan.streams_design(), "the resident design cannot fit, so the plan streams");
    assert!(plan.record.materializations.iter().any(|(name, how)| &**name == "design.matrix"
        && *how == antecedent_contract::BufferMaterialization::Chunked));
    let operators = plan.lowered().expect("compile attached a program").physical().operators();
    let scan = operators.last().expect("a scan operator");
    assert_eq!(scan.effect(), antecedent_core::PhysicalEffect::Scan);
    assert!(scan.estimates().retained().is_some_and(|bytes| bytes > 0), "the Gram stays resident");
    let streamed = study.run(&ctx).unwrap();

    let (got, want) =
        (streamed.estimate.as_effect().unwrap(), resident.estimate.as_effect().unwrap());
    assert!(close(got.ate, want.ate, 1e-6), "ate {} vs {}", got.ate, want.ate);
    let (got, want) = (
        got.diagnostics.first_stage.as_ref().expect("streamed first-stage diagnostics"),
        want.diagnostics.first_stage.as_ref().expect("resident first-stage diagnostics"),
    );
    assert!(
        close(got.f_statistic, want.f_statistic, 1e-6),
        "F {} vs {}",
        got.f_statistic,
        want.f_statistic
    );
    assert_eq!((got.df1, got.df2), (want.df1, want.df2));
    match (got.anderson_rubin, want.anderson_rubin) {
        (Some((lo, hi, _)), Some((want_lo, want_hi, _))) => {
            assert!(close(lo, want_lo, 1e-5) && close(hi, want_hi, 1e-5));
        }
        (None, None) => {}
        other => panic!("the Anderson-Rubin sets disagree: {other:?}"),
    }
}

#[test]
fn a_weighting_design_over_the_resident_budget_streams_and_equals_the_resident_fit() {
    // z1, z2 -> t -> y, with z1, z2 -> y.
    let z1: Vec<f64> = (0..ROWS).map(|i| noise(i, 1) * 3.0).collect();
    let z2: Vec<f64> = (0..ROWS).map(|i| noise(i, 2) * 2.0).collect();
    let t: Vec<f64> = (0..ROWS)
        .map(|i| {
            let eta = 0.5 * z1[i] - 0.3 * z2[i];
            f64::from(0.5 + noise(i, 3) < 1.0 / (1.0 + (-eta).exp()))
        })
        .collect();
    let y: Vec<f64> = (0..ROWS)
        .map(|i| 1.0 + 2.0 * t[i] + 0.5 * z1[i] - 0.3 * z2[i] + 0.4 * noise(i, 4))
        .collect();
    let data = TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("y", y.as_slice()),
        ("z1", z1.as_slice()),
        ("z2", z2.as_slice()),
    ])
    .unwrap();
    let study = Study::tabular(data)
        .graph(dag(&[(2, 0), (3, 0), (2, 1), (3, 1), (0, 1)], 4))
        .query(AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1)))
        .estimator(EstimatorId::PropensityWeighting)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();

    let resident_ctx = ExecutionContext::for_tests(1);
    assert!(!study.compile(&resident_ctx).unwrap().streams_design());
    let resident = study.run(&resident_ctx).unwrap();

    let ctx = bounded();
    let plan = study.compile(&ctx).unwrap();
    assert!(plan.streams_design(), "the resident design cannot fit, so the plan streams");
    let streamed = study.run(&ctx).unwrap();

    let (got, want) =
        (streamed.estimate.as_effect().unwrap(), resident.estimate.as_effect().unwrap());
    assert!(close(got.ate, want.ate, 1e-6), "ate {} vs {}", got.ate, want.ate);
    assert!(
        close(got.se_analytic(), want.se_analytic(), 1e-5),
        "se {} vs {}",
        got.se_analytic(),
        want.se_analytic()
    );
    let (got, want) = (
        got.diagnostics.overlap_report.as_ref().expect("streamed overlap report"),
        want.diagnostics.overlap_report.as_ref().expect("resident overlap report"),
    );
    assert!(close(got.propensity_min, want.propensity_min, 1e-6));
    assert!(close(got.propensity_max, want.propensity_max, 1e-6));
    assert!(close(got.ess.unwrap(), want.ess.unwrap(), 1e-6));
    assert_eq!(got.extreme_weight_count, want.extreme_weight_count);
}

#[test]
fn a_route_without_a_closed_form_reduction_is_still_refused_over_budget() {
    // A bootstrapped weighting estimator has no streamed reduction: the plan refuses the
    // limit instead of declaring a path it cannot execute.
    let z: Vec<f64> = (0..ROWS).map(|i| noise(i, 1) * 3.0).collect();
    let t: Vec<f64> = (0..ROWS).map(|i| f64::from(z[i] + noise(i, 3) > 0.0)).collect();
    let y: Vec<f64> = (0..ROWS).map(|i| 1.0 + 2.0 * t[i] + 0.5 * z[i] + noise(i, 4)).collect();
    let data = TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("y", y.as_slice()),
        ("z", z.as_slice()),
    ])
    .unwrap();
    let study = Study::tabular(data)
        .graph(dag(&[(2, 0), (2, 1), (0, 1)], 3))
        .query(AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1)))
        .estimator(EstimatorId::PropensityWeighting)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(50)
        .build()
        .unwrap();
    assert!(matches!(study.compile(&bounded()), Err(crate::CausalError::Resource { .. })));
}

#[test]
fn an_aipw_design_over_the_resident_budget_streams_and_equals_the_resident_cross_fit() {
    // z1, z2 -> t -> y, with z1, z2 -> y.
    let z1: Vec<f64> = (0..ROWS).map(|i| noise(i, 1) * 3.0).collect();
    let z2: Vec<f64> = (0..ROWS).map(|i| noise(i, 2) * 2.0).collect();
    let t: Vec<f64> = (0..ROWS)
        .map(|i| {
            let eta = 0.5 * z1[i] - 0.3 * z2[i];
            f64::from(0.5 + noise(i, 3) < 1.0 / (1.0 + (-eta).exp()))
        })
        .collect();
    let y: Vec<f64> = (0..ROWS)
        .map(|i| 1.0 + 2.0 * t[i] + 0.5 * z1[i] - 0.3 * z2[i] + 0.4 * noise(i, 4))
        .collect();
    let data = TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("y", y.as_slice()),
        ("z1", z1.as_slice()),
        ("z2", z2.as_slice()),
    ])
    .unwrap();
    let study = Study::tabular(data)
        .graph(dag(&[(2, 0), (3, 0), (2, 1), (3, 1), (0, 1)], 4))
        .query(AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1)))
        .estimator(EstimatorId::Aipw)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();

    // The same seed in both contexts is the same fold plan.
    let resident_ctx = ExecutionContext::for_tests(1);
    assert!(!study.compile(&resident_ctx).unwrap().streams_design());
    let resident = study.run(&resident_ctx).unwrap();

    // The design is four float64 columns: above this limit, below the batch window and fold ids.
    let mut ctx = ExecutionContext::for_tests(1);
    ctx.set_memory_limit(Some(2_000_000));
    let plan = study.compile(&ctx).unwrap();
    assert!(plan.streams_design(), "the resident design cannot fit, so the plan streams");
    let streamed = study.run(&ctx).unwrap();

    let (got, want) =
        (streamed.estimate.as_effect().unwrap(), resident.estimate.as_effect().unwrap());
    assert!(close(got.ate, want.ate, 1e-6), "ate {} vs {}", got.ate, want.ate);
    assert!(
        close(got.se_analytic(), want.se_analytic(), 1e-5),
        "se {} vs {}",
        got.se_analytic(),
        want.se_analytic()
    );
    let (got, want) = (
        got.diagnostics.overlap_report.as_ref().expect("streamed overlap report"),
        want.diagnostics.overlap_report.as_ref().expect("resident overlap report"),
    );
    assert!(close(got.propensity_min, want.propensity_min, 1e-6));
    assert!(close(got.propensity_max, want.propensity_max, 1e-6));
}
