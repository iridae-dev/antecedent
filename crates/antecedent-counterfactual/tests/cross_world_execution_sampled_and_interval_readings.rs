//! The closed sampled and interval readings of the finite cross-world executor.
//!
//! The executor enumerates (or eliminates) the exogenous posterior exactly and returns a point
//! value labelled `point_only`. Every other reading of that row is closed: a sampled posterior
//! (no measured coverage record for the whole abduction-action-prediction path) and an
//! interval or posterior reading. This file sweeps the whole request surface rather than one
//! hand-picked request: every quantity kind crossed with every coupling declaration and every
//! requested reading, plus the sampled-posterior request under each of them, and asserts for
//! each cell either the exact reference value (a hand-enumerated response-type model) or a
//! typed refusal that did no abduction, prediction or sampling work and retained nothing.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::RequestedReading;
use antecedent_core::ExecutionContext;
use antecedent_counterfactual::selective::{
    CrossWorldExecutionError, ExogenousInference, ExogenousVariable, FactualUnit,
    FiniteExecutionRequest, FiniteQuantity, FiniteScm, FiniteScmExecutor, FiniteVariable,
    FiniteWorld, NoiseCoupling, WorldAtom,
};

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(17)
}

// X = U_X (P(1) = 0.375), Y = f(X, U_Y) over (never, complier, defier, always) with
// q = [0.125, 0.5, 0.125, 0.25]; all masses are dyadic so every value below is exact.
fn scm() -> FiniteScm {
    FiniteScm {
        population: "pop-a".to_owned(),
        history: "regime-1".to_owned(),
        exogenous: vec![
            ExogenousVariable { name: "u_x".to_owned(), pmf: vec![0.625, 0.375] },
            ExogenousVariable { name: "u_y".to_owned(), pmf: vec![0.125, 0.5, 0.125, 0.25] },
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

fn do_x(name: &str, level: u8) -> FiniteWorld {
    FiniteWorld { name: name.to_owned(), interventions: vec![(0, level)], routes: vec![] }
}

fn atom(world: usize, variable: usize, level: u8) -> WorldAtom {
    WorldAtom { world, variable, level }
}

/// Each quantity with its exact value for the evidence `X = 1, Y = 1` (consistent types:
/// complier 0.5 and always 0.25, so the posterior is 2/3 and 1/3).
///   Y_{x0} = 0: complier gives 0, always gives 1            -> 2/3
///   Y_{x1} = 1 and Y_{x0} = 0 jointly: complier only        -> 2/3
///   E[Y_{x1}] - E[Y_{x0}]: complier 1 - 0, always 1 - 1     -> 2/3
fn quantities() -> Vec<(&'static str, FiniteQuantity, f64)> {
    vec![
        ("probability", FiniteQuantity::Probability(atom(0, 1, 0)), 2.0 / 3.0),
        ("joint", FiniteQuantity::JointProbability(vec![atom(1, 1, 1), atom(0, 1, 0)]), 2.0 / 3.0),
        ("contrast", FiniteQuantity::MeanContrast { plus: (1, 1), minus: (0, 1) }, 2.0 / 3.0),
    ]
}

fn request(
    quantity: FiniteQuantity,
    coupling: NoiseCoupling,
    exogenous: ExogenousInference,
    reading: RequestedReading,
) -> FiniteExecutionRequest {
    FiniteExecutionRequest {
        evidence_history: "regime-1".to_owned(),
        scm: scm(),
        units: vec![FactualUnit { unit: "u".to_owned(), observed: vec![(0, 1), (1, 1)] }],
        worlds: vec![do_x("x0", 0), do_x("x1", 1)],
        quantity,
        coupling,
        exogenous,
        target_population: None,
        reading,
    }
}

fn detail(e: &CrossWorldExecutionError) -> (String, String) {
    let r = e.refusal().expect("a typed refusal");
    (r.code.to_owned(), r.detail.clone())
}

const COUPLINGS: [NoiseCoupling; 3] = [
    NoiseCoupling::SharedAbducedExogenous,
    NoiseCoupling::IndependentWorldNoise,
    NoiseCoupling::Undeclared,
];

#[test]
fn a_sampled_posterior_is_refused_before_any_work_for_every_request_shape() {
    for (name, quantity, _) in quantities() {
        for coupling in COUPLINGS {
            for reading in [
                RequestedReading::Point,
                RequestedReading::Spread,
                RequestedReading::Interval,
                RequestedReading::Posterior,
                RequestedReading::Envelope,
            ] {
                let mut exec = FiniteScmExecutor::new();
                let r = request(
                    quantity.clone(),
                    coupling,
                    ExogenousInference::PosteriorSampling,
                    reading,
                );
                let err = exec.execute(&r, &ctx()).expect_err(name);
                assert_eq!(
                    detail(&err),
                    (
                        "cell_not_licensed".to_owned(),
                        "cross_world_execution.sampled_abduction_not_licensed".to_owned()
                    ),
                    "{name} {coupling:?} {reading:?}"
                );
                let c = exec.counters();
                assert_eq!(
                    (c.abductions, c.shared_draws, c.world_predictions, c.mechanism_fits),
                    (0, 0, 0, 0)
                );
                assert_eq!(exec.retained_len(), 0);
            }
        }
    }
}

#[test]
fn interval_and_posterior_readings_are_closed_for_every_quantity_and_coupling() {
    for (name, quantity, _) in quantities() {
        for coupling in COUPLINGS {
            for (reading, expected) in [
                (RequestedReading::Interval, "joint_law_propagation.interval_route_closed"),
                (RequestedReading::Posterior, "joint_law_propagation.posterior_route_closed"),
            ] {
                let mut exec = FiniteScmExecutor::new();
                let r = request(
                    quantity.clone(),
                    coupling,
                    ExogenousInference::ExactEnumeration,
                    reading,
                );
                let err = exec.execute(&r, &ctx()).expect_err(name);
                let (code, d) = detail(&err);
                assert_eq!(code, "cell_not_licensed", "{name} {coupling:?} {reading:?}");
                assert_eq!(d, expected, "{name} {coupling:?} {reading:?}");
                assert_eq!(exec.retained_len(), 0, "a closed reading retains no abduction");
                assert_eq!(exec.counters().shared_draws, 0);
            }
        }
    }
}

#[test]
fn the_point_reading_of_the_exact_row_is_the_enumerated_value_with_no_draws() {
    for (name, quantity, expected) in quantities() {
        let mut exec = FiniteScmExecutor::new();
        let r = request(
            quantity,
            NoiseCoupling::SharedAbducedExogenous,
            ExogenousInference::ExactEnumeration,
            RequestedReading::Point,
        );
        let out = exec.execute(&r, &ctx()).unwrap();
        assert!((out.value - expected).abs() < 1e-14, "{name}: {} vs {expected}", out.value);
        assert_eq!(out.label, "point_only");
        assert_eq!(out.work.shared_draws, 0);
        assert_eq!(exec.counters().shared_draws, 0);
    }
}

/// `execute` has no declared parameter set to range over, so the envelope reading is a typed
/// refusal that did no work (the range is `execute_envelope`'s); the spread reading is the exact
/// noise variance of the same quantity, which sits on the point value.
#[test]
fn envelope_reading_is_refused_before_any_work_and_spread_is_the_exact_noise_variance() {
    for (name, quantity, expected) in quantities() {
        let mut exec = FiniteScmExecutor::new();
        let r = request(
            quantity.clone(),
            NoiseCoupling::SharedAbducedExogenous,
            ExogenousInference::ExactEnumeration,
            RequestedReading::Envelope,
        );
        let err = exec.execute(&r, &ctx()).expect_err(name);
        assert_eq!(
            detail(&err),
            (
                "cell_not_licensed".to_owned(),
                "cross_world_execution.envelope_needs_parameter_set".to_owned()
            ),
            "{name}"
        );
        let c = exec.counters();
        assert_eq!(
            (c.abductions, c.shared_draws, c.world_predictions, c.mechanism_fits),
            (0, 0, 0, 0)
        );
        assert_eq!(exec.retained_len(), 0);

        // Posterior {complier 2/3, always 1/3}: every quantity takes the two values {0, 1} (or
        // {1, 0} for the contrast) with masses 1/3 and 2/3, so the variance is 2/9.
        let mut exec = FiniteScmExecutor::new();
        let r = request(
            quantity,
            NoiseCoupling::SharedAbducedExogenous,
            ExogenousInference::ExactEnumeration,
            RequestedReading::Spread,
        );
        let out = exec.execute(&r, &ctx()).expect(name);
        assert!((out.value - expected).abs() < 1e-14);
        let spread = out.spread.expect("spread");
        assert!((spread.units[0].variance - 2.0 / 9.0).abs() < 1e-14, "{name}");
        assert!((spread.mean_variance - 2.0 / 9.0).abs() < 1e-14, "{name}");
    }
}
