//! One policy, every route: a price-floor program gives the same realised values when the
//! model is sampled forward, when abduced noise is replayed through the counterfactual
//! engine, and when it is rolled out over time with a lagged read of earlier steps.
//!
//! Every route calls the same per-node precedence and program executor, so on identical
//! noise the values agree bit for bit; independent hand computations pin the values.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    CausalRng, ExecutionContext, Intervention, InterventionSequence, SequencedIntervention,
    TemporalPolicy, ValueExpr, VariableId,
};
use antecedent_counterfactual::{
    AbductionMissingPolicy, CounterfactualEngine, CounterfactualWorld, ExogenousPosterior,
    NoiseInferenceKind,
};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_model::{
    CompiledCausalModel, CompiledMechanismStore, InterventionOverlay, MechanismSlot,
    MechanismWorkspace, ModelView, NoiseReplayer, NoiseSource, TemporalNoise, TemporalPlan,
    ValueBatch, rollout_temporal, sample_interventional,
};

const PRICE: VariableId = VariableId::from_raw(0);
const DEMAND: VariableId = VariableId::from_raw(1);

/// `price -> demand`, `price = 11 + u_price`, `demand = 50 - 0.8 price + u_demand`.
fn price_model() -> CompiledCausalModel {
    let mut graph = Dag::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    CompiledCausalModel::compile(graph).unwrap().with_mechanisms(CompiledMechanismStore {
        slots: Arc::from([
            MechanismSlot::LinearGaussian { intercept: 11.0, coeffs: Arc::from([]), sigma: 2.0 },
            MechanismSlot::LinearGaussian {
                intercept: 50.0,
                coeffs: Arc::from([-0.8]),
                sigma: 1.0,
            },
        ]),
    })
}

fn floor(slope: f64) -> ValueExpr {
    ValueExpr::constant(10.0).max(ValueExpr::constant(slope).mul(ValueExpr::generated()))
}

/// Raise the price by a premium when the generated price is high, otherwise floor it.
fn conditional() -> ValueExpr {
    ValueExpr::generated().gt(ValueExpr::constant(12.0)).if_then_else(
        ValueExpr::constant(10.0).add(ValueExpr::constant(0.5).mul(ValueExpr::generated())),
        floor(0.85),
    )
}

fn hand_floor(generated: f64) -> f64 {
    10.0_f64.max(0.85 * generated)
}

fn hand_conditional(generated: f64) -> f64 {
    if generated > 12.0 { 10.0 + 0.5 * generated } else { hand_floor(generated) }
}

fn exogenous(noise: &[f64], n_units: usize) -> ExogenousPosterior {
    ExogenousPosterior {
        noise: Arc::from(noise),
        n_units,
        n_nodes: 2,
        kind: NoiseInferenceKind::Invertible,
        assumed_columns: Arc::from([false, false]),
    }
}

#[test]
#[allow(clippy::float_cmp, reason = "the routes share one executor, so equal noise is equal bits")]
fn a_policy_agrees_between_sampling_and_counterfactual_execution_on_shared_noise() {
    let model = price_model();
    let engine = CounterfactualEngine::new(price_model());
    let ctx = ExecutionContext::for_tests(41);
    let n = 3_000;
    let cases: [(&str, ValueExpr, fn(f64) -> f64); 2] = [
        ("price floor", floor(0.85), hand_floor),
        ("conditional branch", conditional(), hand_conditional),
    ];
    for (label, expr, hand) in cases {
        let policy = Intervention::transform(PRICE, expr);
        let overlay =
            InterventionOverlay::from_interventions(&model, std::slice::from_ref(&policy)).unwrap();
        // Intervention sampling: the noise it drew is retained alongside the realised values.
        let (forward, table) = NoiseReplayer::new()
            .sample(&ModelView::with_overlay(&model, overlay), n, &ctx)
            .unwrap();
        let factual = NoiseReplayer::new()
            .replay(&ModelView::observational(&model), NoiseSource::Table(&table), 0, &ctx)
            .unwrap();
        let (p0, d0) = (factual.column(0).unwrap(), factual.column(1).unwrap());

        // Counterfactual execution on that same noise is bit-identical to the sample.
        let world =
            CounterfactualWorld { unit_rows: None, interventions: Arc::from([policy.clone()]) };
        let mut ws = MechanismWorkspace::default();
        let same_noise = engine
            .predict(
                &exogenous(table.values(), n),
                &[world.clone()],
                &[PRICE, DEMAND],
                false,
                &mut ws,
                &ctx,
            )
            .unwrap();
        let (p1, d1) = (forward.column(0).unwrap(), forward.column(1).unwrap());
        for i in 0..n {
            assert_eq!(same_noise.get(0, DenseNodeId::from_raw(0), i), p1[i], "{label}: price {i}");
            assert_eq!(
                same_noise.get(0, DenseNodeId::from_raw(1), i),
                d1[i],
                "{label}: demand {i}"
            );
            // The realised price is the hand-applied policy on the unit's own generated price.
            assert_eq!(p1[i], hand(p0[i]), "{label}: hand price {i}");
        }

        // Abducting the noise from the factual rows recovers it up to rounding, and the
        // counterfactual then reproduces the sample.
        let data = TabularData::from_f64_columns([("price", p0), ("demand", d0)]).unwrap();
        let abduced = engine.abduct(&data, AbductionMissingPolicy::Error, &ctx).unwrap();
        let replay =
            engine.predict(&abduced, &[world], &[PRICE, DEMAND], false, &mut ws, &ctx).unwrap();
        for i in 0..n {
            for (node, forward_column) in [(0, p1), (1, d1)] {
                let got = replay.get(0, DenseNodeId::from_raw(node), i);
                assert!((got - forward_column[i]).abs() < 1e-9, "{label}: abduced node {node} {i}");
            }
        }
        assert!(p1.iter().any(|p| *p == 10.0), "{label}: the floor binds for some units");
    }
}

/// One floor, three routes, one set of noise: the sample, the counterfactual on that noise
/// and every step of a temporal rollout fed the same noise are the same bits.
#[test]
#[allow(clippy::float_cmp, reason = "the routes share one executor, so equal noise is equal bits")]
fn policy_semantics_agree_across_sampling_counterfactual_and_temporal_routes() {
    let model = price_model();
    let engine = CounterfactualEngine::new(price_model());
    let ctx = ExecutionContext::for_tests(55);
    let (n, steps) = (5_000, 12);
    let policy = Intervention::transform(PRICE, floor(0.85));
    let overlay =
        InterventionOverlay::from_interventions(&model, std::slice::from_ref(&policy)).unwrap();
    let (sampled, table) =
        NoiseReplayer::new().sample(&ModelView::with_overlay(&model, overlay), n, &ctx).unwrap();

    let mut ws = MechanismWorkspace::default();
    let exo = exogenous(table.values(), n);
    let world = CounterfactualWorld { unit_rows: None, interventions: Arc::from([policy.clone()]) };
    let counterfactual =
        engine.predict(&exo, &[world], &[PRICE, DEMAND], false, &mut ws, &ctx).unwrap();

    let schedule = InterventionSequence::new(vec![SequencedIntervention::new(
        policy,
        TemporalPolicy::sustained(0, 1_000),
    )]);
    let plan = TemporalPlan {
        sequence: &schedule,
        n_steps: steps,
        n_rows: n,
        window: 0,
        initial_history: &[],
    };
    let path = engine.predict_temporal(&vec![exo; steps], &plan, &mut ws, &ctx).unwrap();

    for raw in 0..2u32 {
        let node = raw as usize;
        let column = sampled.column(node).unwrap();
        for i in 0..n {
            let dense = DenseNodeId::from_raw(raw);
            assert_eq!(counterfactual.get(0, dense, i), column[i], "node {node} unit {i}");
            for t in 0..steps {
                assert_eq!(path.column(t, node).unwrap()[i], column[i], "step {t} node {node}");
            }
        }
    }
    let prices = sampled.column(0).unwrap();
    let held = prices.iter().filter(|p| **p == 10.0).count();
    assert!(held > 0 && held < n, "the floor binds for some units and not others");
}

#[test]
#[allow(clippy::float_cmp, reason = "an assignment ignores noise, so the recursion is exact")]
fn a_lagged_policy_matches_a_hand_recursion_on_every_temporal_route() {
    let model = price_model();
    let engine = CounterfactualEngine::new(price_model());
    let ctx = ExecutionContext::for_tests(7);
    let (n, steps) = (64, 12);
    let initial_price: Vec<f64> = (0..n).map(|i| 30.0 + (i % 9) as f64).collect();
    let mut start = initial_price.clone();
    start.extend(std::iter::repeat_n(0.0, n));
    let initial = [ValueBatch { n_rows: n, n_nodes: 2, values: Arc::from(start) }];

    // price_t := max(10, 0.85 * price_{t-1}) from the unit's last observed price.
    let assign = Intervention::assign(
        PRICE,
        ValueExpr::constant(10.0).max(ValueExpr::constant(0.85).mul(ValueExpr::lag(PRICE, 1))),
    );
    let schedule = InterventionSequence::new(vec![SequencedIntervention::new(
        assign,
        TemporalPolicy::sustained(0, 1_000),
    )]);
    let plan = TemporalPlan {
        sequence: &schedule,
        n_steps: steps,
        n_rows: n,
        window: 1,
        initial_history: &initial,
    };
    let mut ws = MechanismWorkspace::default();

    // Fresh noise.
    let mut rng = CausalRng::from_seed(5);
    let drawn =
        rollout_temporal(&model, &plan, TemporalNoise::Draw(&mut rng), &mut ws, &ctx).unwrap();
    // Retained noise, as a counterfactual replays it.
    let noise: Vec<Arc<[f64]>> = (0..steps)
        .map(|t| {
            let mut per_step = vec![0.0; 2 * n];
            for i in 0..n {
                per_step[i] = (((i * 7 + t * 3) % 11) as f64 - 5.0) * 0.9;
                per_step[n + i] = (((i * 5 + t) % 7) as f64 - 3.0) * 0.4;
            }
            Arc::from(per_step)
        })
        .collect();
    let exo: Vec<ExogenousPosterior> = noise.iter().map(|s| exogenous(s, n)).collect();
    let replayed = engine.predict_temporal(&exo, &plan, &mut ws, &ctx).unwrap();

    for i in 0..n {
        let mut price = initial_price[i];
        for t in 0..steps {
            price = 10.0_f64.max(0.85 * price);
            assert_eq!(drawn.column(t, 0).unwrap()[i], price, "drawn price, step {t} unit {i}");
            assert_eq!(replayed.column(t, 0).unwrap()[i], price, "replayed price, step {t}");
            // Demand follows its mechanism from the policy's price and the retained noise.
            let demand = 50.0 + -0.8 * price + noise[t][n + i];
            assert!((replayed.column(t, 1).unwrap()[i] - demand).abs() < 1e-12);
        }
    }
}

#[test]
#[allow(clippy::float_cmp, reason = "a transform reads the same generated value on every route")]
fn a_transform_with_a_lag_read_matches_the_hand_recursion_and_the_static_route_without_one() {
    let engine = CounterfactualEngine::new(price_model());
    let ctx = ExecutionContext::for_tests(8);
    let (n, steps) = (48, 8);
    let noise: Vec<Arc<[f64]>> = (0..steps)
        .map(|t| {
            let mut per_step = vec![0.0; 2 * n];
            for i in 0..n {
                per_step[i] = (((i * 3 + t * 5) % 13) as f64 - 6.0) * 0.7;
                per_step[n + i] = (((i + t * 2) % 5) as f64 - 2.0) * 0.3;
            }
            Arc::from(per_step)
        })
        .collect();
    let exo: Vec<ExogenousPosterior> = noise.iter().map(|s| exogenous(s, n)).collect();
    let mut prior = vec![25.0; n];
    prior.extend(std::iter::repeat_n(0.0, n));
    let initial = [ValueBatch { n_rows: n, n_nodes: 2, values: Arc::from(prior) }];
    let mut ws = MechanismWorkspace::default();
    let always = TemporalPolicy::sustained(0, 1_000);

    // price_t := max(10, 0.85 * generated_t + 0.1 * price_{t-1}).
    let lagged = Intervention::transform(
        PRICE,
        ValueExpr::constant(10.0).max(
            ValueExpr::constant(0.85)
                .mul(ValueExpr::generated())
                .add(ValueExpr::constant(0.1).mul(ValueExpr::lag(PRICE, 1))),
        ),
    );
    let schedule =
        InterventionSequence::new(vec![SequencedIntervention::new(lagged, always.clone())]);
    let plan = TemporalPlan {
        sequence: &schedule,
        n_steps: steps,
        n_rows: n,
        window: 1,
        initial_history: &initial,
    };
    let path = engine.predict_temporal(&exo, &plan, &mut ws, &ctx).unwrap();
    for i in 0..n {
        let mut previous = 25.0;
        for t in 0..steps {
            let generated = 11.0 + noise[t][i];
            let price = 10.0_f64.max(0.85 * generated + 0.1 * previous);
            assert!((path.column(t, 0).unwrap()[i] - price).abs() < 1e-12, "step {t} unit {i}");
            previous = path.column(t, 0).unwrap()[i];
        }
    }

    // Without a lag read the temporal route is the static counterfactual, step by step.
    let floor_policy = Intervention::transform(PRICE, floor(0.85));
    let schedule =
        InterventionSequence::new(vec![SequencedIntervention::new(floor_policy.clone(), always)]);
    let plan = TemporalPlan { sequence: &schedule, window: 0, initial_history: &[], ..plan };
    let path = engine.predict_temporal(&exo, &plan, &mut ws, &ctx).unwrap();
    let world = CounterfactualWorld { unit_rows: None, interventions: Arc::from([floor_policy]) };
    for (t, step_exo) in exo.iter().enumerate() {
        let static_world = engine
            .predict(step_exo, &[world.clone()], &[PRICE, DEMAND], false, &mut ws, &ctx)
            .unwrap();
        for i in 0..n {
            for node in 0..2u32 {
                assert_eq!(
                    path.column(t, node as usize).unwrap()[i],
                    static_world.get(0, DenseNodeId::from_raw(node), i),
                    "step {t} node {node} unit {i}"
                );
            }
        }
    }
}

#[test]
fn history_the_route_cannot_supply_is_refused() {
    let model = price_model();
    let ctx = ExecutionContext::for_tests(9);
    let mut ws = MechanismWorkspace::default();
    let always = TemporalPolicy::sustained(0, 1_000);
    let read_back = |lag: u32| {
        InterventionSequence::new(vec![SequencedIntervention::new(
            Intervention::assign(PRICE, ValueExpr::lag(PRICE, lag).add(ValueExpr::constant(1.0))),
            always.clone(),
        )])
    };
    let mut rng = CausalRng::from_seed(1);
    let mut run = |schedule: &InterventionSequence, window: u32, history: usize| {
        let initial: Vec<ValueBatch> = (0..history).map(|_| ValueBatch::zeros(4, 2)).collect();
        let plan = TemporalPlan {
            sequence: schedule,
            n_steps: 3,
            n_rows: 4,
            window,
            initial_history: &initial,
        };
        rollout_temporal(&model, &plan, TemporalNoise::Draw(&mut rng), &mut ws, &ctx)
            .unwrap_err()
            .to_string()
    };

    // A look-back of two steps with one observed step: unavailable at the first step.
    let early = run(&read_back(2), 2, 1);
    assert!(early.contains("policy_history_unavailable"), "{early}");
    assert!(early.contains("history_not_yet_observed"), "{early}");
    assert!(early.contains("step 0"), "{early}");
    // A look-back deeper than the declared window.
    let beyond = run(&read_back(3), 2, 4);
    assert!(beyond.contains("policy_history_unavailable"), "{beyond}");
    assert!(beyond.contains("history_beyond_window"), "{beyond}");
    // A window of zero is the static model, which has no history.
    let none = run(&read_back(1), 0, 0);
    assert!(none.contains("policy_history_unavailable"), "{none}");
    assert!(none.contains("history_unavailable"), "{none}");

    // A future value does not exist when the policy applies: a zero look-back is refused too.
    let future = InterventionSequence::new(vec![SequencedIntervention::new(
        Intervention::assign(PRICE, ValueExpr::lag(PRICE, 0)),
        always.clone(),
    )]);
    let zero = run(&future, 2, 4);
    assert!(zero.contains("policy_history_unavailable"), "{zero}");
    assert!(zero.contains("history_not_past"), "{zero}");

    // The static samplers never supply history either.
    let lagged = Intervention::assign(PRICE, ValueExpr::lag(PRICE, 1));
    let error =
        sample_interventional(&model, &[lagged], 8, &mut CausalRng::from_seed(2), &mut ws, &ctx)
            .unwrap_err()
            .to_string();
    assert!(error.contains("policy_history_unavailable"), "{error}");
}
