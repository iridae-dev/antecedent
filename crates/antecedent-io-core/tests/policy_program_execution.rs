//! A policy encoded into a portable artifact decodes to the same program: the same identity
//! and, executed, the same values bit for bit, in forward sampling and in a temporal rollout
//! that reads earlier steps.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    CausalRng, ExecutionContext, Intervention, InterventionSequence, SequencedIntervention,
    TemporalPolicy, ValueExpr, VariableId,
};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_io_core::policy_program_artifact::{
    decode_policy_program, encode_policy_program, policy_program_identity,
};
use antecedent_model::{
    CompiledCausalModel, CompiledMechanismStore, MechanismSlot, MechanismWorkspace, TemporalNoise,
    TemporalPlan, ValueBatch, rollout_temporal, sample_interventional,
};

const PRICE: VariableId = VariableId::from_raw(0);

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

#[test]
#[allow(clippy::float_cmp, reason = "a decoded program is the same program, so bits are equal")]
fn a_saved_policy_replays_with_the_same_identity_and_the_same_draws() {
    let model = price_model();
    let ctx = ExecutionContext::for_tests(3);
    let floor = Intervention::transform(
        PRICE,
        ValueExpr::constant(10.0).max(ValueExpr::constant(0.85).mul(ValueExpr::generated())),
    );
    let saved = vec![floor];
    let bytes = encode_policy_program(&saved).unwrap();
    let loaded = decode_policy_program(&bytes).unwrap();
    assert_eq!(loaded, saved);
    assert_eq!(policy_program_identity(&loaded).unwrap(), policy_program_identity(&saved).unwrap());

    let draw = |program: &[Intervention]| {
        let mut ws = MechanismWorkspace::default();
        sample_interventional(&model, program, 5_000, &mut CausalRng::from_seed(77), &mut ws, &ctx)
            .unwrap()
    };
    let (before, after) = (draw(&saved), draw(&loaded));
    assert_eq!(before.values, after.values);
    assert!(before.column(0).unwrap().iter().all(|price| *price >= 10.0));
}

#[test]
#[allow(clippy::float_cmp, reason = "a decoded program is the same program, so bits are equal")]
fn a_saved_lagged_policy_replays_over_time() {
    let model = price_model();
    let ctx = ExecutionContext::for_tests(4);
    let lagged = Intervention::assign(
        PRICE,
        ValueExpr::constant(10.0).max(ValueExpr::constant(0.85).mul(ValueExpr::lag(PRICE, 1))),
    );
    let schedule = |program: Vec<Intervention>| {
        InterventionSequence::new(
            program
                .into_iter()
                .map(|iv| SequencedIntervention::new(iv, TemporalPolicy::sustained(0, 100)))
                .collect::<Vec<_>>(),
        )
    };
    let saved = vec![lagged];
    let loaded = decode_policy_program(&encode_policy_program(&saved).unwrap()).unwrap();
    let mut start = vec![40.0; 16];
    start.extend(std::iter::repeat_n(0.0, 16));
    let initial = [ValueBatch { n_rows: 16, n_nodes: 2, values: Arc::from(start) }];
    let roll = |sequence: &InterventionSequence| {
        let plan = TemporalPlan {
            sequence,
            n_steps: 10,
            n_rows: 16,
            window: 1,
            initial_history: &initial,
        };
        let mut ws = MechanismWorkspace::default();
        let mut rng = CausalRng::from_seed(12);
        rollout_temporal(&model, &plan, TemporalNoise::Draw(&mut rng), &mut ws, &ctx).unwrap()
    };
    let (before, after) = (roll(&schedule(saved)), roll(&schedule(loaded)));
    for (a, b) in before.steps.iter().zip(&after.steps) {
        assert_eq!(a.values, b.values);
    }
    let mut price = 40.0_f64;
    for step in &before.steps {
        price = 10.0_f64.max(0.85 * price);
        assert!(step.column(0).unwrap().iter().all(|p| *p == price));
    }
}
