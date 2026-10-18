//! Intervention programs: a price floor `max(10, 0.85 * price)` as a typed expression.
//!
//! A transform replaces the value a node's mechanism generated; the node keeps its incoming
//! edges and its noise. The same program is sampled forward, replayed on each factual unit's
//! own abduced noise, licensed per route and valued against the unintervened model, saved as a
//! portable artifact and, as a batch callback, saved only as a verified record of its source: an
//! in-process function has no durable encoding.
//!
//! Run: `cargo run -p antecedent --example intervention_programs`
//! Source: `examples/rust/intervention_programs.rs`

use std::sync::Arc;

use antecedent::gcm::{counterfactual_policy, fit_gcm, policy_value, sample_do};
use antecedent_core::{
    BatchCallback, CallbackError, CallbackImplementation, CallbackOutput, CallbackSpec, CausalRng,
    ExecutionContext, Intervention, PinnedDependency, Portability, RecoveryRecord, ValueExpr,
    VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_io::policy_program_artifact::{
    decode_policy_program, decode_policy_program_with, encode_policy_program,
    policy_program_callback_records, policy_program_identity,
};
use antecedent_model::{RouteDecision, policy_support};

const PRICE: VariableId = VariableId::from_raw(0);
const DEMAND: VariableId = VariableId::from_raw(1);

/// The floor as a plain function of a batch of columns, bound to the model as a callback.
#[derive(Debug)]
struct FloorCallback {
    spec: CallbackSpec,
}

impl BatchCallback for FloorCallback {
    fn spec(&self) -> &CallbackSpec {
        &self.spec
    }

    fn call(&self, inputs: &[&[f64]], out: &mut [f64]) -> Result<(), CallbackError> {
        for (o, generated) in out.iter_mut().zip(inputs[0]) {
            *o = 10.0_f64.max(0.85 * generated);
        }
        Ok(())
    }
}

/// Digest of the source that defines the recoverable floor (a consumer holds that source).
const SOURCE_DIGEST: &str = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08";

fn floor_callback(implementation: CallbackImplementation) -> Arc<dyn BatchCallback> {
    Arc::new(FloorCallback {
        spec: CallbackSpec {
            name: "floor".into(),
            inputs: Arc::from(["generated".to_owned()]),
            output: CallbackOutput::Real,
            dependencies: Arc::from([PinnedDependency {
                name: "numpy".into(),
                version: "2.0.0".into(),
            }]),
            implementation,
        },
    })
}

fn recoverable(digest: &str) -> CallbackImplementation {
    CallbackImplementation::Recoverable {
        source_sha256: digest.to_owned(),
        entry_point: "floor".to_owned(),
    }
}

/// Runs the example end to end; `main` calls it and the example test suite runs it.
#[allow(clippy::too_many_lines, reason = "the example reads top to bottom as one walkthrough")]
pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    // Deterministic synthetic data: demand falls with price, prices straddle the floor.
    let rows = 600_u32;
    let price: Vec<f64> =
        (0..rows).map(|i| 11.0 + (f64::from((i * 37) % 101) - 50.0) * 0.04).collect();
    let demand: Vec<f64> = (0..rows)
        .zip(&price)
        .map(|(i, p)| 50.0 - 0.8 * p + (f64::from((i * 13) % 23) - 11.0) * 0.05)
        .collect();
    let data = TabularData::from_f64_columns([
        ("price", price.as_slice()),
        ("demand", demand.as_slice()),
    ])?;
    let mut graph = Dag::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1))?;
    let fitted = fit_gcm(graph, &data)?;
    let ctx = ExecutionContext::for_tests(7);

    // 1. Forward sampling. On the same noise the policy is the hand-applied floor.
    let floor = Intervention::transform(
        PRICE,
        ValueExpr::constant(10.0).max(ValueExpr::constant(0.85).mul(ValueExpr::generated())),
    );
    let observed = sample_do(&fitted.model, &[], 512, &mut CausalRng::from_seed(7), &ctx)?;
    let policy = sample_do(
        &fitted.model,
        std::slice::from_ref(&floor),
        512,
        &mut CausalRng::from_seed(7),
        &ctx,
    )?;
    let (before, after) = (observed.column(0)?, policy.column(0)?);
    for (generated, floored) in before.iter().zip(after) {
        assert!((floored - 10.0_f64.max(0.85 * generated)).abs() < 1e-12);
    }
    assert!(after.iter().all(|p| *p >= 10.0), "no draw falls below the floor");
    assert!(before.iter().any(|p| 0.85 * p < 10.0), "the floor binds for some draws");

    // 2. Counterfactual execution: each factual unit's own noise goes through the floor.
    let outcomes = [PRICE, DEMAND];
    let worlds = counterfactual_policy(
        fitted.model.clone(),
        &data,
        std::slice::from_ref(&floor),
        &outcomes,
        &ctx,
    )?;
    let dense_price = fitted.model.dense_of(PRICE).ok_or("price is in the model")?;
    let dense_demand = fitted.model.dense_of(DEMAND).ok_or("demand is in the model")?;
    for (unit, factual) in price.iter().enumerate() {
        let floored = worlds.get(0, dense_price, unit);
        assert!((floored - 10.0_f64.max(0.85 * factual)).abs() < 1e-9, "unit {unit}");
        if floored > *factual {
            // Demand falls with price, so a unit whose price was raised sells less.
            assert!(worlds.get(0, dense_demand, unit) < demand[unit], "unit {unit}");
        }
    }

    // 3. Each route is licensed separately. A transform is simulable and replayable but is a
    //    function of each unit's noise, not a level, so it has no observational identification.
    let support = policy_support(&fitted.model, std::slice::from_ref(&floor));
    assert_eq!(support.simulation, RouteDecision::Licensed);
    assert_eq!(support.counterfactual, RouteDecision::Licensed);
    match &support.identification {
        RouteDecision::Refused { code, .. } => {
            assert_eq!(code.as_str(), "policy_route_not_licensed")
        }
        other => panic!("a transform has no identification route, got {other:?}"),
    }

    // 4. A program that reads its own descendant is refused before any draw.
    let circular = Intervention::assign(PRICE, ValueExpr::var(DEMAND));
    match policy_support(&fitted.model, &[circular]).simulation {
        RouteDecision::Refused { code, .. } => {
            assert_eq!(code.as_str(), "policy_dependency_circular")
        }
        other => panic!("a circular read must be refused, got {other:?}"),
    }

    // 5. A closed-form program saves as versioned bytes under a content identity, and the
    //    decoded program is the same program.
    let saved = [floor.clone()];
    let bytes = encode_policy_program(&saved)?;
    let loaded = decode_policy_program(&bytes)?;
    assert_eq!(loaded, saved);
    assert_eq!(policy_program_identity(&loaded)?, policy_program_identity(&saved)?);
    let other = Intervention::transform(
        PRICE,
        ValueExpr::constant(10.0).max(ValueExpr::constant(0.9).mul(ValueExpr::generated())),
    );
    assert_ne!(policy_program_identity(&[other])?, policy_program_identity(&saved)?);

    // 6. The same floor as a batch callback: one call per batch, bit-identical to the
    //    expression, but an in-process function has no durable encoding.
    let by_callback = Intervention::transform(
        PRICE,
        ValueExpr::callback(
            floor_callback(CallbackImplementation::Runtime),
            [ValueExpr::generated()],
        ),
    );
    let drawn = sample_do(
        &fitted.model,
        std::slice::from_ref(&by_callback),
        512,
        &mut CausalRng::from_seed(7),
        &ctx,
    )?;
    assert!(drawn.column(0)?.iter().zip(after).all(|(a, b)| a.to_bits() == b.to_bits()));
    assert!(matches!(by_callback.portability(), Portability::RuntimeBound { .. }));
    let refused =
        encode_policy_program(&[by_callback]).expect_err("a runtime callback cannot be saved");
    assert!(refused.to_string().contains("policy_callback_not_portable"), "{refused}");

    // 7. Declared with the digest of self-contained source, a callback leaves the process only as
    //    a record, never as code. The consumer supplies the implementation; it is verified
    //    against the record before the program can run, and altered source never replays.
    let portable = Intervention::transform(
        PRICE,
        ValueExpr::callback(floor_callback(recoverable(SOURCE_DIGEST)), [ValueExpr::generated()]),
    );
    let artifact = encode_policy_program(std::slice::from_ref(&portable))?;
    let records = policy_program_callback_records(&artifact)?;
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].name, "floor");
    let supply = |digest: &str| {
        let digest = digest.to_owned();
        move |record: &RecoveryRecord| {
            (record.name == "floor").then(|| floor_callback(recoverable(&digest)))
        }
    };
    let restored = decode_policy_program_with(&artifact, &supply(SOURCE_DIGEST))?;
    let replayed = sample_do(&fitted.model, &restored, 512, &mut CausalRng::from_seed(7), &ctx)?;
    assert!(replayed.column(0)?.iter().zip(after).all(|(a, b)| a.to_bits() == b.to_bits()));
    let tampered = decode_policy_program_with(&artifact, &supply(&"0".repeat(64)))
        .expect_err("source that differs from the record must not replay");
    assert!(tampered.to_string().contains("policy_callback_unrecoverable"), "{tampered}");
    let unsupplied =
        decode_policy_program(&artifact).expect_err("a record needs an implementation");
    assert!(unsupplied.to_string().contains("policy_callback_unrecoverable"), "{unsupplied}");

    // 8. The value of the policy is the paired difference against the unintervened model on the
    //    same noise; only the simulation error is quantified.
    let value = policy_value(&fitted.model, std::slice::from_ref(&floor), DEMAND, 4_000, &ctx)?;
    // The floor lowers most prices, and demand falls with price.
    assert!(value.mean_difference > 5.0 * value.difference_simulation_stderr);
    assert!(value.support.simulation.is_licensed());
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    run()
}
