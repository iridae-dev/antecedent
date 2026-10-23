//! Composition across the horizons of one temporal claim.
//!
//! Hand example. Two horizons whose joint replicates are `x_r` and `2 x_r + 1` (perfectly
//! coupled): the contrast `h2 - h1` is `x_r + 1`, with the spread of `x`, whereas independent
//! marginals of the same spreads would give `sqrt(var x + 4 var x)`. A response that retains no
//! joint realizations (horizons fit one by one) is refused rather than paired by draw index, and
//! the composed coordinate carries the horizon axis, so a record measured over one horizon set
//! is not another's.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::analysis::horizon_composition::{
    compose_horizons, compose_horizons_with, horizon_draws,
};
use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_contract::PropagationOperation;
use antecedent_core::{
    AssumptionSet, CausalQuery, CausalResponse, ContinuousDomain, DiagnosticScope, GridSpec,
    HorizonIdentification, IdentificationStatus, ResponseFunctional, ResponseIdentification,
    ResponseQuery, ResponseUncertainty, ResponseValue, SupportDiagnostic, SupportRegion,
    SupportReport, SupportStatus, VariableId,
};
use antecedent_core::{ExecutionContext, Lag, TemporalPolicy, TemporalResponseSpec};
use antecedent_data::TimeSeriesData;
use antecedent_estimate::{JOINT_HORIZON_LAYOUT, JOINT_HORIZON_REPLICATES};
use antecedent_graph::{TemporalDag, ensure_lagged};

fn horizons(list: &[u32]) -> Arc<[HorizonIdentification]> {
    list.iter()
        .map(|&horizon| HorizonIdentification {
            horizon,
            status: IdentificationStatus::NonparametricallyIdentified,
            method: Arc::from("temporal.backdoor.unfolded"),
            adjustment: Arc::from([]),
        })
        .collect()
}

/// A one-dose, two-horizon mean curve; `replicates` is replicate-major over the two cells.
fn response(replicates: Option<&[[f64; 2]]>) -> CausalResponse {
    let mut support = SupportReport {
        status: SupportStatus::Supported,
        query_region: SupportRegion { minima: Arc::from([0.0]), maxima: Arc::from([0.0]) },
        point_status: None,
        diagnostics: Vec::new(),
        warnings: Vec::new(),
    };
    if let Some(rows) = replicates {
        support.diagnostics.push(SupportDiagnostic {
            id: Arc::from(JOINT_HORIZON_LAYOUT),
            values: Arc::from([rows.len() as f64, 2.0, 0.0, 1.0]),
            detail: Arc::from("layout"),
            scope: DiagnosticScope::Global,
        });
        support.diagnostics.push(SupportDiagnostic {
            id: Arc::from(JOINT_HORIZON_REPLICATES),
            values: rows.iter().flatten().copied().collect::<Vec<_>>().into(),
            detail: Arc::from("replicates"),
            scope: DiagnosticScope::Global,
        });
    }
    CausalResponse {
        estimand: ResponseFunctional::MeanCurve {
            outcome: VariableId::from_raw(1),
            treatment: ContinuousDomain::new(
                VariableId::from_raw(0),
                GridSpec::Values(Arc::from([0.0])),
            ),
        },
        identification_status: IdentificationStatus::NonparametricallyIdentified,
        estimate: ResponseIdentification::PointIdentified(ResponseValue::Surface {
            grid: Arc::from([0.0]),
            dimension: 1,
            mean: Arc::from([1.0, 3.0]),
        }),
        uncertainty: ResponseUncertainty::None,
        support,
        assumptions: AssumptionSet::new(),
        provenance_id: Arc::from("temporal.response.test"),
        horizon_identification: Some(horizons(&[1, 2])),
        interaction_structurally_zero: false,
    }
}

fn coupled_rows() -> Vec<[f64; 2]> {
    (0..200)
        .map(|r| {
            let x = f64::from(r % 17) - 8.0 + 0.01 * f64::from(r);
            [x, 2.0 * x + 1.0]
        })
        .collect()
}

#[test]
fn coupled_horizons_compose_with_the_spread_of_the_realizations() {
    let rows = coupled_rows();
    let draws = horizon_draws(&response(Some(&rows)), 0).unwrap();
    assert_eq!(draws.horizons, vec![1, 2]);
    assert_eq!(draws.columns.len(), 2);
    assert_eq!(draws.center, vec![1.0, 3.0]);
    let contrast = compose_horizons(&draws, &[-1.0, 1.0], PropagationOperation::Contrast).unwrap();
    let x: Vec<f64> = rows.iter().map(|row| row[0]).collect();
    let mean = x.iter().sum::<f64>() / x.len() as f64;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (x.len() as f64 - 1.0);
    // h2 - h1 = x + 1 on every row: the spread is that of x, not the independent-marginal
    // combination sqrt(var(x) + var(2x + 1)).
    assert!((contrast.spread.variance - var).abs() < 1e-9 * var.max(1.0));
    let independent = (var + 4.0 * var).sqrt();
    assert!(contrast.spread.standard_deviation < 0.5 * independent);
    assert_eq!(contrast.label, "uncalibrated_spread");
    assert_eq!(contrast.coordinate.horizons, vec![1, 2]);
    assert!(contrast.coordinate.id().ends_with("|horizons=1,2"));
    assert!(contrast.plan.alignment_id.as_deref().is_some_and(|id| id.contains("#joint:200:")));
    // The same combination over a different horizon set is a different coordinate.
    let one = compose_horizons(
        &horizon_draws(&response(Some(&rows)), 0).unwrap(),
        &[0.5, 0.5],
        PropagationOperation::WeightedSum,
    )
    .unwrap();
    assert_ne!(one.coordinate.id(), contrast.coordinate.id());
}

#[test]
fn horizons_fit_one_by_one_are_refused_not_paired_by_index() {
    let refused = horizon_draws(&response(None), 0).expect_err("no joint realizations retained");
    refused.validate().expect("a well-formed refusal");
    assert!(refused.detail.contains("horizon_draws_not_joint"));
    assert!(refused.remedy.is_some());
    // A dose outside the grid and a wrong coefficient count are refused with their own detail.
    let rows = coupled_rows();
    let draws = horizon_draws(&response(Some(&rows)), 0).unwrap();
    assert!(horizon_draws(&response(Some(&rows)), 3).is_err());
    assert!(compose_horizons(&draws, &[1.0], PropagationOperation::Sum).is_err());
}

#[test]
fn a_nonlinear_functional_of_the_horizons_composes_the_same_joint_draws() {
    let rows = coupled_rows();
    let draws = horizon_draws(&response(Some(&rows)), 0).unwrap();
    let ctx = antecedent_core::ExecutionContext::for_tests(3);
    // The peak of the two horizons: row by row the larger of h1 = x and h2 = 2x + 1.
    let peak = compose_horizons_with(
        &draws,
        PropagationOperation::NonlinearFunctional,
        &|values| values[0].max(values[1]),
        &ctx,
    )
    .unwrap();
    for (row, composed) in rows.iter().zip(&peak.draws) {
        assert_eq!(*composed, row[0].max(row[1]));
    }
    assert_eq!(peak.coordinate.horizons, vec![1, 2]);
    assert!(peak.coordinate.id().starts_with("nonlinear_functional|aligned_joint_law|"));
    // A linear operation belongs to compose_horizons.
    assert!(
        compose_horizons_with(&draws, PropagationOperation::Sum, &|values| values[0], &ctx)
            .is_err()
    );
}

/// `x_t = 0.3 + 0.2 x_{t-1} + e`, `y_t = 1 + 2 x_{t-1} + 3 x_{t-2} + 0.5 e'`.
fn compiled_series() -> (TimeSeriesData, TemporalDag) {
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut normal = move || {
        let mut uniform = || {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            (state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64 + 1e-12
        };
        let (u, v) = (uniform(), uniform());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    };
    let n = 300;
    let mut x = vec![0.0; n];
    let mut y = vec![0.0; n];
    for t in 2..n {
        x[t] = 0.3 + 0.2 * x[t - 1] + 0.3 * normal();
        y[t] = 1.0 + 2.0 * x[t - 1] + 3.0 * x[t - 2] + 0.5 * normal();
    }
    let series =
        TimeSeriesData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())], 1).unwrap();
    let mut graph = TemporalDag::empty();
    let x1 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let x2 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(2)).unwrap();
    let y0 = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(x1, y0).unwrap();
    graph.insert_directed(x2, y0).unwrap();
    (series, graph)
}

fn compiled_study(inference: InferenceMode, retain: bool, replicates: u32) -> Study {
    let (series, graph) = compiled_series();
    let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(Arc::from([0.0, 1.0])),
        ),
    })
    .with_temporal(TemporalResponseSpec::new(vec![1, 2], TemporalPolicy::pulse(0), None).unwrap());
    Study::series(series)
        .graph(graph)
        .query(CausalQuery::Response(query))
        .inference(inference)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(replicates)
        .retain_joint_horizons(retain)
        .build()
        .unwrap()
}

#[test]
fn the_compiled_temporal_response_retains_joint_horizons_in_both_modes() {
    std::thread::Builder::new()
        .name("compiled-joint-horizons".into())
        .stack_size(32 * 1024 * 1024)
        .spawn(|| {
            let ctx = ExecutionContext::for_tests(8);
            // Frequentist: the joint circular-block bootstrap replicates.
            let kept = compiled_study(InferenceMode::Frequentist, true, 60).run(&ctx).unwrap();
            let draws = horizon_draws(kept.response().unwrap(), 1).unwrap();
            assert_eq!(draws.horizons, vec![1, 2]);
            let cumulative =
                compose_horizons(&draws, &[1.0, 1.0], PropagationOperation::Sum).unwrap();
            assert!(cumulative.spread.standard_deviation > 0.0);
            let plain = compiled_study(InferenceMode::Frequentist, false, 60).run(&ctx).unwrap();
            assert!(horizon_draws(plain.response().unwrap(), 1).is_err(), "off by default");
            // Bayesian: the block-weight posterior over the surface, in the lowered program.
            let bayesian = InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(60));
            let study = compiled_study(bayesian.clone(), true, 0);
            let plan = study.compile(&ctx).unwrap();
            let lowered = plan.lowered().expect("a lowered program");
            let posterior = lowered.statistical().posterior().expect("a posterior plan");
            assert_eq!(posterior.prior().name(), "dirichlet_block_weights(1)");
            let kept = study.run(&ctx).unwrap();
            let draws = horizon_draws(kept.response().unwrap(), 1).unwrap();
            let contrast =
                compose_horizons(&draws, &[-1.0, 1.0], PropagationOperation::Contrast).unwrap();
            assert!(contrast.spread.standard_deviation > 0.0);
            let conjugate = compiled_study(bayesian, false, 0).run(&ctx).unwrap();
            assert!(
                horizon_draws(conjugate.response().unwrap(), 1).is_err(),
                "the per-horizon conjugate fits are independent marginals"
            );
            // The option needs what produces the replicates.
            let (series, graph) = compiled_series();
            let refused = Study::series(series)
                .graph(graph)
                .query(CausalQuery::Response(
                    ResponseQuery::new(ResponseFunctional::MeanCurve {
                        outcome: VariableId::from_raw(1),
                        treatment: ContinuousDomain::new(
                            VariableId::from_raw(0),
                            GridSpec::Values(Arc::from([0.0])),
                        ),
                    })
                    .with_temporal(
                        TemporalResponseSpec::new(vec![1], TemporalPolicy::pulse(0), None).unwrap(),
                    ),
                ))
                .retain_joint_horizons(true)
                .bootstrap_replicates(0)
                .build();
            assert!(
                refused.is_err(),
                "a Frequentist response retains replicates only with a bootstrap"
            );
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn the_horizons_of_an_intervention_level_response_compose_too() {
    let rows = coupled_rows();
    let mut response = response(Some(&rows));
    response.estimand = ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(1),
        interventions: Arc::from([antecedent_core::Intervention::set(
            VariableId::from_raw(0),
            antecedent_core::Value::f64(1.0),
        )]),
    };
    let draws = horizon_draws(&response, 0).unwrap();
    let cumulative = compose_horizons(&draws, &[1.0, 1.0], PropagationOperation::Sum).unwrap();
    assert_eq!(cumulative.coordinate.horizons, vec![1, 2]);
    assert!(cumulative.spread.standard_deviation > 0.0);
}
