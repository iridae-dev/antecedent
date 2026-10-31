//! The Dirichlet row-weight mechanism refit of the Bayesian GCM routes against closed forms.
//!
//! Each posterior draw refits the selected mechanisms under one Dirichlet(1, ..., 1) weight vector
//! drawn from a documented stream of the execution context. This test regenerates those weight
//! vectors from the same streams and compares every published draw with the closed form of the
//! weighted least-squares fit:
//!
//! * counterfactual route, chain `t -> m -> y` with binary `t`: abduction and prediction are
//!   additive in the fitted mechanisms, so the unit contrast of every unit, and hence `mean_ite`,
//!   is `b_m * b_y`, the product of the weighted slope of `m` on `t` (the difference of the weighted
//!   group means) and the weighted slope of `y` on `m`, exactly, draw by draw. The frequentist
//!   point route equals the same product at unit weights;
//! * change-attribution route, an exactly linear two-period chain `x -> y` (`y = 1 + 2x` then
//!   `y = 6 + 2x`): every draw's intercept shift is exact whatever the weights, so the intercept
//!   mechanism's Shapley component is 5 in every draw; the covariate component is twice the
//!   difference of the two weighted covariate means of that draw's two weight vectors (up to the
//!   Monte Carlo error of the coalition sampler); and the components add up to the total change.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    AllocationMethod, CausalQuery, ChangeAttributionQuery, CounterfactualQuery, ExecutionContext,
    Intervention, PopulationSelector, ShapleyConfig, StreamDomain, Value, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_kernels::bayesian_bootstrap_weights;

const SEED: u64 = 812;
const DRAWS: usize = 16;

fn chain() -> (TabularData, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = 24;
    let t: Vec<f64> = (0..n).map(|i| if i < 9 { 0.0 } else { 1.0 }).collect();
    let m: Vec<f64> = (0..n).map(|i| 0.5 + 1.2 * t[i] + 0.3 * f64::from(i as u32).sin()).collect();
    let y: Vec<f64> =
        (0..n).map(|i| -1.0 + 0.8 * m[i] + 0.2 * (2.1 * f64::from(i as u32)).cos()).collect();
    let data = TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("m", m.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap();
    (data, t, m, y)
}

fn weighted_slope(x: &[f64], y: &[f64], w: &[f64]) -> f64 {
    let sw: f64 = w.iter().sum();
    let xb = x.iter().zip(w).map(|(a, w)| a * w).sum::<f64>() / sw;
    let yb = y.iter().zip(w).map(|(a, w)| a * w).sum::<f64>() / sw;
    let sxy: f64 = (0..x.len()).map(|i| w[i] * (x[i] - xb) * (y[i] - yb)).sum();
    let sxx: f64 = (0..x.len()).map(|i| w[i] * (x[i] - xb) * (x[i] - xb)).sum();
    sxy / sxx
}

fn chain_query() -> CounterfactualQuery {
    CounterfactualQuery::new(
        VariableId::from_raw(2),
        Arc::from([Intervention::set(VariableId::from_raw(0), Value::f64(1.0))]),
    )
    .with_control_level(0.0)
}

fn chain_dag() -> Dag {
    let mut dag = Dag::with_variables(3);
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    dag
}

#[test]
fn every_counterfactual_draw_is_the_product_of_the_weighted_slopes_of_its_weight_vector() {
    let (data, t, m, y) = chain();
    let n = t.len();
    let ctx = ExecutionContext::for_tests(SEED);
    let result = Study::tabular(data.clone())
        .graph(chain_dag())
        .query(CausalQuery::Counterfactual(chain_query()))
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(DRAWS)))
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    let posterior = result.posterior.as_ref().expect("posterior");
    let column = posterior.draws.column(posterior.effect_column().unwrap()).unwrap();
    assert_eq!(column.len(), DRAWS);
    let mut rng = ctx.rng.stream_for(StreamDomain::Attribution, 0x0CF0);
    for (draw, &got) in column.iter().enumerate() {
        let w = bayesian_bootstrap_weights(&mut rng, n, false);
        let want = weighted_slope(&t, &m, &w) * weighted_slope(&m, &y, &w);
        assert!((got - want).abs() <= 1e-9 * want.abs(), "draw {draw}: {got} vs {want}");
    }
    // The point route is the same product at unit weights.
    let point = Study::tabular(data)
        .graph(chain_dag())
        .query(CausalQuery::Counterfactual(chain_query()))
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    let ones = vec![1.0; n];
    let want = weighted_slope(&t, &m, &ones) * weighted_slope(&m, &y, &ones);
    let got = point.estimate().as_effect().unwrap().ate;
    assert!((got - want).abs() <= 1e-9 * want.abs(), "point {got} vs {want}");
}

fn two_period() -> (TabularData, Vec<f64>) {
    let n = 80usize;
    let x: Vec<f64> = (0..n).map(|i| ((i * 7) % 40) as f64 * 0.1).collect();
    let y: Vec<f64> =
        (0..n).map(|i| if i < 40 { 1.0 + 2.0 * x[i] } else { 6.0 + 2.0 * x[i] }).collect();
    (TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())]).unwrap(), x)
}

fn column_named(posterior: &antecedent::CausalPosterior, name: &str) -> Vec<f64> {
    let index = posterior
        .draws
        .schema
        .quantities
        .iter()
        .position(|q| match q {
            antecedent_prob::PosteriorQuantityKind::Effect { name: n }
            | antecedent_prob::PosteriorQuantityKind::Scalar { name: n } => &**n == name,
            _ => false,
        })
        .unwrap_or_else(|| panic!("no column {name}"));
    posterior.draws.column(index).unwrap().to_vec()
}

#[test]
fn change_attribution_draws_use_one_weight_vector_per_population_and_stay_coupled() {
    let (data, x) = two_period();
    let mut dag = Dag::with_variables(2);
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let ctx = ExecutionContext::for_tests(SEED);
    let query = ChangeAttributionQuery::new(
        VariableId::from_raw(1),
        PopulationSelector::TimeRange { start: 0, end: 40 },
        PopulationSelector::TimeRange { start: 40, end: 80 },
    )
    .with_allocation(AllocationMethod::Shapley { approximation: ShapleyConfig::exact() });
    let result = Study::tabular(data)
        .graph(dag)
        .query(CausalQuery::ChangeAttribution(query))
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(DRAWS)))
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    let posterior = result.posterior.as_ref().expect("posterior");
    let total = column_named(&posterior, "total_change");
    let covariate = column_named(&posterior, "component[0]");
    let intercept = column_named(&posterior, "component[1]");
    assert_eq!(total.len(), DRAWS);
    let mut rng = ctx.rng.stream_for(StreamDomain::Attribution, 0xC4A6_7A7E);
    let weighted_mean = |values: &[f64], w: &[f64]| {
        values.iter().zip(w).map(|(a, w)| a * w).sum::<f64>() / w.iter().sum::<f64>()
    };
    let mut worst = 0.0_f64;
    for draw in 0..DRAWS {
        let baseline = bayesian_bootstrap_weights(&mut rng, 40, false);
        let comparison = bayesian_bootstrap_weights(&mut rng, 40, false);
        let shift = weighted_mean(&x[40..], &comparison) - weighted_mean(&x[..40], &baseline);
        // The fits are exact in every draw, so the intercept mechanism carries the shift of 5.
        assert!((intercept[draw] - 5.0).abs() < 1e-6, "draw {draw}: intercept {}", intercept[draw]);
        // Efficiency of the Shapley split, draw by draw.
        assert!(
            (covariate[draw] + intercept[draw] - total[draw]).abs() < 1e-9,
            "draw {draw}: components do not add to the total"
        );
        // The covariate component is 2 * (difference of the weighted covariate means) up to the
        // Monte Carlo error of the sampled covariate marginals (2000 samples, sd about 1.15); the
        // largest observed deviation is 9e-3.
        worst = worst.max((covariate[draw] - 2.0 * shift).abs());
        assert!(
            (covariate[draw] - 2.0 * shift).abs() < 0.03,
            "draw {draw}: {} vs {}",
            covariate[draw],
            2.0 * shift
        );
    }
    eprintln!("largest covariate-component deviation {worst}");
}
