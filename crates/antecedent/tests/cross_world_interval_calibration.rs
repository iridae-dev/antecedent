//! Coverage of the whole-path cross-world interval under the declared model.
//!
//! The whole method is abduction of the exogenous posterior given each unit's
//! factual evidence (exact enumeration), prediction of the counterfactual
//! quantity in the named worlds under the shared exogenous state, and the
//! normal interval built on the unit values under a declared unit sampling.
//! The data are generated from the declared model, so the measurement is
//! conditional on model adequacy, exactly as the licence states.
//!
//! Model: `X ~ Bernoulli(0.4)`, `Y = f(X, U)` with response types
//! `U in {never, complier, defier, always}`; evidence is the observed `(X, Y)`
//! of each unit; the quantity is `P(Y_{do(X = 0)} = 0 | evidence)`.
//!
//! * `independent_sample`: units are an independent sample of the population;
//!   the truth is the population probability `P(U in {never, complier})`.
//! * `fixed_units`: the truth is the realized mean of the counterfactual
//!   outcome `1{U in {never, complier}}` of the simulated units.
//!
//! Designs probing near-deterministic mechanisms and rare posterior states are
//! not records here: they are properties of the declared model that would sit
//! on the same key as the gated designs.
//!
//! Ignored tests run via `scripts/gate_calibration.sh`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "bounded simulation sizes and small indices"
)]

mod common;

use antecedent::analysis::cross_world_execution::{
    CrossWorldExecution, ExogenousInference, ExogenousVariable, FactualUnit,
    FiniteExecutionRequest, FiniteQuantity, FiniteScm, FiniteVariable, FiniteWorld, NoiseCoupling,
    UnitSampling, WorldAtom,
};
use antecedent::analysis::cross_world_interval::{
    CROSS_WORLD_CONSTRUCTION, CROSS_WORLD_LEVEL, whole_path_interval, whole_path_interval_basis,
};
use antecedent_contract::RequestedReading;
use antecedent_core::ExecutionContext;
use common::calibration::{
    Construction, CoverageTally, RecordKey, ScopeFacts, grid_n, grid_seed, map_replicates, n_sim,
    uniform,
};

fn scm(q: [f64; 4]) -> FiniteScm {
    FiniteScm {
        population: "pop-a".to_owned(),
        history: "regime-1".to_owned(),
        exogenous: vec![
            ExogenousVariable { name: "u_x".to_owned(), pmf: vec![0.6, 0.4] },
            ExogenousVariable { name: "u_y".to_owned(), pmf: q.to_vec() },
        ],
        variables: vec![
            FiniteVariable {
                name: "x".to_owned(),
                levels: 2,
                parents: vec![],
                exogenous: 0,
                table: vec![0, 1],
            },
            FiniteVariable {
                name: "y".to_owned(),
                levels: 2,
                parents: vec![0],
                exogenous: 1,
                table: vec![0, 0, 0, 1, 1, 0, 1, 1],
            },
        ],
    }
}

fn world(name: &str, level: u8) -> FiniteWorld {
    FiniteWorld { name: name.to_owned(), interventions: vec![(0, level)], routes: vec![] }
}

/// The units' factual evidence and the realized counterfactual outcomes.
fn simulate(q: [f64; 4], n: usize, seed: u64) -> (Vec<FactualUnit>, f64) {
    let mut next = uniform(seed, 1);
    let (mut units, mut realized) = (Vec::with_capacity(n), 0.0);
    for i in 0..n {
        let x = u8::from(next() < 0.4);
        let draw = next();
        let mut cumulative = 0.0;
        let mut response = 3_u8;
        for (r, p) in q.iter().enumerate() {
            cumulative += p;
            if draw < cumulative {
                response = r as u8;
                break;
            }
        }
        let y = match response {
            0 => 0,
            1 => x,
            2 => 1 - x,
            _ => 1,
        };
        // Y under do(X = 0) is 0 for never and complier.
        realized += f64::from(u8::from(response < 2));
        units.push(FactualUnit { unit: format!("u{i}"), observed: vec![(0, x), (1, y)] });
    }
    (units, realized / n as f64)
}

fn request(units: Vec<FactualUnit>, q: [f64; 4]) -> FiniteExecutionRequest {
    let model = scm(q);
    FiniteExecutionRequest {
        evidence_history: model.history.clone(),
        scm: model,
        units,
        worlds: vec![world("x0", 0), world("x1", 1)],
        quantity: FiniteQuantity::Probability(WorldAtom { world: 0, variable: 1, level: 0 }),
        coupling: NoiseCoupling::SharedAbducedExogenous,
        exogenous: ExogenousInference::ExactEnumeration,
        target_population: None,
        reading: RequestedReading::Point,
    }
}

fn design(test: &'static str, dgp: &'static str, q: [f64; 4], sampling: UnitSampling) {
    let n = grid_n(60);
    let mut tally = CoverageTally::for_record(
        RecordKey { test, dgp, interval: CROSS_WORLD_CONSTRUCTION },
        CROSS_WORLD_LEVEL,
    );
    let results = map_replicates(n_sim(), |replicate| {
        let seed = grid_seed(0x95_0000 + replicate);
        let (units, realized_mean) = simulate(q, n, seed);
        let req = request(units, q);
        let ctx = ExecutionContext::for_tests(seed);
        let execution = CrossWorldExecution::seal(&req, &ctx).expect("seal");
        let account = execution.whole_path_uncertainty(sampling, &ctx).expect("account");
        let interval =
            whole_path_interval(account.uncertainty(), CROSS_WORLD_LEVEL).map(|(_, ends, _)| ends);
        let basis = whole_path_interval_basis(&req, sampling).expect("key");
        let truth = match sampling {
            UnitSampling::IndependentSample => q[0] + q[1],
            _ => realized_mean,
        };
        (interval, basis, truth)
    });
    for (interval, basis, truth) in results {
        tally.bind(
            &Construction {
                query: basis.query.to_string(),
                graph_class: basis.graph_class.to_string(),
                structure: basis.structure.to_string(),
                modality: basis.modality.to_string(),
                inference: basis.inference.to_string(),
                estimator: basis.estimator.to_string(),
                interval_method: basis.interval_method.to_string(),
                se_kind: basis.se_kind.to_string(),
                dependence: basis.dependence.to_string(),
                posterior: basis.posterior.to_string(),
                functional: basis.functional.to_string(),
                identification: basis.identification.to_string(),
                reported_level: basis.level,
            },
            ScopeFacts {
                row_count: basis.row_count,
                replicates_ok: basis.replicates_ok,
                posterior_draws: basis.posterior_draws,
                unidentified_mass: basis.unidentified_mass,
            },
        );
        tally.record(interval, truth);
    }
    tally.assert();
}

#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn cross_world_interval_independent_sample_balanced_types_l95() {
    design(
        "cross_world_interval_independent_sample_balanced_types_l95",
        "response_type_scm_x_0.4_types_0.2_0.5_0.1_0.2_independent_sample_exact_abduction",
        [0.2, 0.5, 0.1, 0.2],
        UnitSampling::IndependentSample,
    );
}

#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn cross_world_interval_independent_sample_rare_types_l95() {
    design(
        "cross_world_interval_independent_sample_rare_types_l95",
        "response_type_scm_x_0.4_types_0.05_0.45_0.05_0.45_independent_sample_exact_abduction",
        [0.05, 0.45, 0.05, 0.45],
        UnitSampling::IndependentSample,
    );
}

#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn cross_world_interval_fixed_units_balanced_types_l95() {
    design(
        "cross_world_interval_fixed_units_balanced_types_l95",
        "response_type_scm_x_0.4_types_0.2_0.5_0.1_0.2_fixed_units_realized_outcomes",
        [0.2, 0.5, 0.1, 0.2],
        UnitSampling::FixedUnits,
    );
}
