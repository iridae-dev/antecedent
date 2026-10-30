// SPDX-License-Identifier: MIT OR Apache-2.0
//! Typed cyclic interventions: resolution, refusals, analytic fixtures, DAG
//! regression, and world results keyed by `ScientificWorldId`.

use std::sync::Arc;

use antecedent_core::intervention::{
    CYCLIC_CONTINUOUS_TIME_ATTEMPT, CYCLIC_DRIFT_INPUT_ATTEMPT, CYCLIC_PER_LEVEL_CLAMP_ATTEMPT,
    CYCLIC_SOFT_REPLACEMENT_ATTEMPT, CYCLIC_SUPPORTED_OPERATION_ATTEMPT, CyclicFamily,
};
use antecedent_core::refusal::NextAttempt;
use antecedent_core::{
    CanonicalEncode, CanonicalWriter, ExecutionContext, Intervention, InterventionSequence,
    MechanismOverride, ScientificWorldId, SequencedIntervention, StochasticPolicy, TemporalPolicy,
    Value, ValueExpr, VariableId, WorldAction,
};
use antecedent_model::continuous_cyclic::{OdePolicy, OdeSystem};
use antecedent_model::cyclic_world::{
    BIND_VARIABLE_ATTEMPT, CERTIFYING_EVIDENCE_ATTEMPT, CertificateDemand, CertificateGap,
    ContinuousTimeSpecification, CyclicAction, CyclicInterventionRefusal, CyclicVariables,
    CyclicWorldOutcome, CyclicWorldRequest, CyclicWorldResult, DECLARED_INITIAL_LAW_ATTEMPT,
    DISTINCT_WORLD_ATTEMPT, FINITE_FLOAT_LEVEL_ATTEMPT, LinearCyclicSpecification,
    MechanismReplacement, NONZERO_DIAGONAL_ATTEMPT, NonlinearCyclicSpecification,
    ONE_ACTION_PER_VARIABLE_ATTEMPT, QUERY_SPECIFIC_CERTIFICATE_ATTEMPT,
    SELECTION_OR_DYNAMIC_LAW_ATTEMPT, SOFT_PARAMETER_SHAPE_ATTEMPT, SUPPORTED_SOFT_FAMILY_ATTEMPT,
    WELL_FORMED_INTERVENTION_ATTEMPT, WorldFailureCause, evaluate_continuous_time_world,
    evaluate_linear_world, evaluate_linear_worlds, evaluate_nonlinear_world, resolve_cyclic_world,
};
use antecedent_model::linear_solution::{
    LinearStructuralMap, PhysicalStability, SolvabilityRefusal,
};
use antecedent_model::nonlinear_equilibrium::{
    Activation, Equation, EquilibriumModel, EquilibriumOutcome, SolvePolicy,
};
use antecedent_model::scc_compile::{CompileBudget, GaussianDisturbance};

/// The mean and covariance of one world.
type Moments = (Vec<f64>, Vec<f64>);

const EXACT: f64 = 1.0e-12;

fn var(index: u32) -> VariableId {
    // Ids that differ from model positions, so binding is exercised.
    VariableId::from_raw(10 + index)
}

fn world(raw: u32) -> ScientificWorldId {
    ScientificWorldId::from_raw(raw)
}

fn variables(n: u32) -> CyclicVariables {
    CyclicVariables::try_new((0..n).map(var).collect()).expect("variables")
}

fn identity(n: usize) -> Vec<f64> {
    let mut matrix = vec![0.0; n * n];
    for index in 0..n {
        matrix[index * n + index] = 1.0;
    }
    matrix
}

fn linear_spec(
    coefficients: Vec<f64>,
    intercept: Vec<f64>,
    noise_mean: Vec<f64>,
    noise_covariance: Vec<f64>,
) -> LinearCyclicSpecification {
    let n = u32::try_from(intercept.len()).expect("order");
    LinearCyclicSpecification::try_new(
        variables(n),
        LinearStructuralMap::try_new(coefficients, intercept).expect("map"),
        GaussianDisturbance::try_new(noise_mean, noise_covariance).expect("disturbance"),
        "feedback",
        "source-measurements",
        &["solve-note"],
    )
    .expect("specification")
}

/// `A = 0.4 B + U_A`, `B = 0.5 A + U_B`, independent centred unit disturbances.
fn feedback_pair() -> LinearCyclicSpecification {
    linear_spec(vec![0.0, 0.4, 0.5, 0.0], vec![0.0, 0.0], vec![0.0, 0.0], identity(2))
}

/// `A = B + C + U_A`, `B = A + U_B`, `C = A + U_C`.
fn overlapping_loops() -> LinearCyclicSpecification {
    linear_spec(
        vec![0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        vec![0.0; 3],
        vec![0.0; 3],
        identity(3),
    )
}

fn set(index: u32, level: f64) -> Intervention {
    Intervention::set(var(index), Value::f64(level))
}

fn shift(index: u32, delta: f64) -> Intervention {
    Intervention::shift(var(index), Value::f64(delta))
}

fn soft(index: u32, family: &str, parameters: Vec<f64>) -> Intervention {
    Intervention::soft(var(index), MechanismOverride::named(family, parameters))
}

fn evaluate(spec: &LinearCyclicSpecification, request: &CyclicWorldRequest) -> CyclicWorldResult {
    evaluate_linear_world(
        spec,
        request,
        CertificateDemand::ReportUncertified,
        CompileBudget::unlimited(),
    )
    .expect("world")
}

fn law(result: &CyclicWorldResult) -> (&[f64], &[f64]) {
    match result.outcome() {
        CyclicWorldOutcome::GaussianLaw { mean, covariance, .. } => (mean, covariance),
        other => panic!("expected a Gaussian law, got {other:?}"),
    }
}

fn close(left: f64, right: f64, tolerance: f64) -> bool {
    (left - right).abs() <= tolerance
}

#[test]
fn linear_world_refuses_indefinite_disturbances_in_both_routes() {
    let request = CyclicWorldRequest::observational(world(0));
    for edge in [0.4, 1.2] {
        for covariance in [vec![-1.0, 0.0, 0.0, 0.0], vec![1.0, 2.0, 2.0, 1.0]] {
            let spec =
                linear_spec(vec![0.0, edge, edge, 0.0], vec![0.0; 2], vec![0.0; 2], covariance);
            assert!(
                evaluate_linear_world(
                    &spec,
                    &request,
                    CertificateDemand::ReportUncertified,
                    CompileBudget::unlimited()
                )
                .is_err()
            );
        }
        let singular = linear_spec(
            vec![0.0, edge, edge, 0.0],
            vec![0.0; 2],
            vec![0.0; 2],
            vec![1.0, 1.0, 1.0, 1.0],
        );
        let result = evaluate_linear_world(
            &singular,
            &request,
            CertificateDemand::ReportUncertified,
            CompileBudget::unlimited(),
        )
        .unwrap();
        assert!(matches!(result.outcome(), CyclicWorldOutcome::GaussianLaw { .. }));
    }
}

#[test]
fn roadmap_feedback_pair_reproduces_three_different_quantities() {
    let spec = feedback_pair();
    let requests = [
        CyclicWorldRequest::observational(world(0)),
        CyclicWorldRequest::new(world(1), [set(0, 0.0)]),
        CyclicWorldRequest::new(world(2), [set(0, 1.0)]),
        CyclicWorldRequest::new(world(3), [shift(0, 1.0)]),
    ];
    let results = evaluate_linear_worlds(
        &spec,
        &requests,
        CertificateDemand::Required,
        CompileBudget::unlimited(),
    )
    .expect("worlds");
    assert_eq!(results.len(), 4);
    for (id, result) in results.iter() {
        assert_eq!(*id, result.world());
        assert!(result.certificate().is_certified(), "{id}");
        // A hard clamp cuts the loop, so the remaining system is solved in acyclic order; the
        // observational and shifted worlds keep the loop and need the strongly connected block.
        let clamped = *id == world(1) || *id == world(2);
        let expected = if clamped { "acyclic_order" } else { "strongly_connected_blocks" };
        assert_eq!(result.solved_by().as_str(), expected, "{id}");
        assert!(close(result.mass().solved(), 1.0, 0.0));
        assert_eq!(result.physical_stability(), PhysicalStability::Unclaimed);
    }

    // B-on-A association in the observational world.
    let (_, covariance) = law(results.get(world(0)).expect("observational"));
    let association = covariance[1] / covariance[0];
    assert!(close(association, 1.40625 / 1.8125, EXACT), "{association}");
    assert!(close(association, 0.776, 5.0e-4), "{association}");

    // Hard-clamp derivative of B under do(A = a).
    let low = results.get(world(1)).expect("do(A=0)").mean_of(var(1)).expect("mean");
    let high = results.get(world(2)).expect("do(A=1)").mean_of(var(1)).expect("mean");
    let derivative = high - low;
    assert!(close(derivative, 0.5, EXACT), "{derivative}");
    let (_, clamped) = law(results.get(world(2)).expect("do(A=1)"));
    assert!(close(clamped[0], 0.0, EXACT), "a clamped variable has no variance");
    assert!(close(clamped[3], 1.0, EXACT), "the free disturbance stays marginal");

    // B response to a unit mean shift of U_A.
    let shifted = results.get(world(3)).expect("shift");
    let response = shifted.mean_of(var(1)).expect("mean");
    assert!(close(response, 0.625, EXACT), "{response}");
    assert!(close(shifted.mean_of(var(0)).expect("mean"), 1.25, EXACT));
    assert!(matches!(shifted.actions(), [CyclicAction::AdditiveShock { node: 0, .. }]));

    assert!((association - derivative).abs() > 0.1);
    assert!((derivative - response).abs() > 0.1);
    assert!((association - response).abs() > 0.1);

    let summaries = results.summaries();
    assert_eq!(summaries.len(), 4);
    assert!(summaries.iter().all(|summary| summary.certified && summary.certificate_gap.is_none()));
    let shifted_summary = &summaries[3];
    assert_eq!(shifted_summary.world, world(3));
    let means = shifted_summary.means.as_ref().expect("means");
    assert_eq!(means[1].0, var(1));
    assert!(close(means[1].1, 0.625, EXACT));
}

#[test]
fn clamping_c_in_overlapping_loops_leaves_no_solution_and_is_retained_as_failed_mass() {
    let spec = overlapping_loops();
    let clamped = evaluate(&spec, &CyclicWorldRequest::new(world(4), [set(2, 1.0)]));
    let CyclicWorldOutcome::NoSolution { failure } = clamped.outcome() else {
        panic!("do(C) leaves a singular free block: {:?}", clamped.outcome());
    };
    assert_eq!(
        failure.cause,
        WorldFailureCause::Solvability(SolvabilityRefusal::Singular { members: vec![0, 1] })
    );
    assert_eq!(failure.next_attempt, SELECTION_OR_DYNAMIC_LAW_ATTEMPT);
    assert!(close(clamped.mass().failed(), 1.0, 0.0));
    assert!(close(clamped.mass().solved(), 0.0, 0.0));
    assert_eq!(clamped.certificate().gap(), Some(&CertificateGap::NoSolution));
    assert_eq!(clamped.solved_by().as_str(), "not_solved");
    let summary = clamped.summary();
    assert!(summary.means.is_none() && !summary.certified);
    assert_eq!(summary.next_attempt, Some(SELECTION_OR_DYNAMIC_LAW_ATTEMPT));

    // The full system is uniquely solvable even though ρ(|K|) < 1 is not shown.
    let open = evaluate(&spec, &CyclicWorldRequest::observational(world(5)));
    assert!(matches!(
        open.certificate().gap(),
        Some(CertificateGap::LinearFamilyBoundInsufficient { rho_abs_upper }) if *rho_abs_upper >= 1.0
    ));
    assert_eq!(open.certificate().next_attempt(), Some(QUERY_SPECIFIC_CERTIFICATE_ATTEMPT));
    assert_eq!(open.solved_by().as_str(), "direct_free_block_solve");
    // X_A = -(U_A + U_B + U_C), X_B = -(U_A + U_C), X_C = -(U_A + U_B).
    let expected = [3.0, 2.0, 2.0, 2.0, 2.0, 1.0, 2.0, 1.0, 2.0];
    let (mean, covariance) = law(&open);
    assert!(mean.iter().all(|value| close(*value, 0.0, EXACT)));
    for (value, truth) in covariance.iter().zip(expected) {
        assert!(close(*value, truth, EXACT), "{covariance:?}");
    }

    let withheld = evaluate_linear_world(
        &spec,
        &CyclicWorldRequest::observational(world(5)),
        CertificateDemand::Required,
        CompileBudget::unlimited(),
    )
    .expect("withheld world");
    assert_eq!(withheld.outcome(), &CyclicWorldOutcome::Withheld);
    assert!(close(withheld.mass().withheld(), 1.0, 0.0));
    assert!(withheld.summary().means.is_none());
}

#[test]
fn every_action_kind_resolves_to_its_own_identity() {
    let request = CyclicWorldRequest::new(
        world(0),
        [
            set(0, 1.0),
            shift(1, 0.5),
            soft(1, "additive_shift", vec![0.25]),
            soft(2, "constant", vec![2.0]),
            soft(3, "linear_mechanism", vec![0.1, 0.2, 0.3, 0.0, 0.0]),
        ],
    );
    let resolved =
        resolve_cyclic_world(CyclicFamily::LinearEquilibrium, &variables(4), &[], &request)
            .expect("resolved");
    let kinds: Vec<&str> = resolved.actions().iter().map(CyclicAction::kind).collect();
    assert_eq!(
        kinds,
        [
            "hard_clamp",
            "change_mechanism",
            "additive_shock",
            "replace_mechanism",
            "replace_mechanism"
        ]
    );
    assert!(matches!(
        &resolved.actions()[4],
        CyclicAction::ReplaceMechanism {
            node: 3,
            replacement: MechanismReplacement::Linear { weights, .. },
            ..
        } if weights == &[0.2, 0.3, 0.0, 0.0]
    ));
    let nonlinear =
        resolve_cyclic_world(CyclicFamily::NonlinearEquilibrium, &variables(4), &[], &request)
            .expect("the nonlinear family takes the same kinds");
    assert_eq!(nonlinear.actions(), resolved.actions());

    let laws: Vec<Arc<str>> = vec![Arc::from("a0"), Arc::from("b0")];
    let initial = CyclicWorldRequest::new(world(1), [set(1, 3.0)])
        .with_initial_state("a0", 2.0)
        .expect("initial state");
    let continuous =
        resolve_cyclic_world(CyclicFamily::ContinuousTime, &variables(2), &laws, &initial)
            .expect("continuous");
    assert!(matches!(
        continuous.actions(),
        [
            CyclicAction::InitialState { node: 0, level, .. },
            CyclicAction::HardClamp { node: 1, .. },
        ] if close(*level, 2.0, 0.0)
    ));

    // The same numerical answer from a shock and an intercept change stays two actions.
    let spec = feedback_pair();
    let shock = evaluate(&spec, &CyclicWorldRequest::new(world(2), [shift(0, 1.0)]));
    let change =
        evaluate(&spec, &CyclicWorldRequest::new(world(3), [soft(0, "additive_shift", vec![1.0])]));
    assert!(close(shock.mean_of(var(1)).unwrap(), change.mean_of(var(1)).unwrap(), EXACT));
    assert_ne!(shock.actions()[0].kind(), change.actions()[0].kind());

    // A constant replacement keeps the disturbance; a clamp does not.
    let replaced =
        evaluate(&spec, &CyclicWorldRequest::new(world(4), [soft(0, "constant", vec![2.0])]));
    let clamped = evaluate(&spec, &CyclicWorldRequest::new(world(5), [set(0, 2.0)]));
    let (replaced_mean, replaced_covariance) = law(&replaced);
    let (clamped_mean, clamped_covariance) = law(&clamped);
    assert!(close(replaced_mean[1], 1.0, EXACT) && close(clamped_mean[1], 1.0, EXACT));
    assert!(close(replaced_covariance[0], 1.0, EXACT));
    assert!(close(clamped_covariance[0], 0.0, EXACT));

    // B := U_B: A = 0.4 B + U_A, so Cov(A, B) = 0.4 and Var(A) = 1.16.
    let rewired = evaluate(
        &spec,
        &CyclicWorldRequest::new(world(6), [soft(1, "linear_mechanism", vec![0.0, 0.0, 0.0])]),
    );
    let (_, covariance) = law(&rewired);
    assert!(close(covariance[1], 0.4, EXACT));
    assert!(close(covariance[0], 1.16, EXACT));
    assert_eq!(rewired.solved_by().as_str(), "acyclic_order");
}

fn assert_names(refusal: &CyclicInterventionRefusal, expected: NextAttempt) {
    assert_eq!(refusal.next_attempt, expected, "{refusal}");
    assert!(!refusal.next_attempt.name().is_empty());
    assert!(!refusal.next_attempt.sentence().is_empty());
    assert_ne!(refusal.next_attempt, CYCLIC_SUPPORTED_OPERATION_ATTEMPT, "{refusal}");
    let text = refusal.to_string();
    assert!(text.contains(&format!("next attempt: {}", refusal.next_attempt.sentence())), "{text}");
}

#[test]
#[allow(
    clippy::too_many_lines,
    reason = "a linear script of one request per refusal, each checked for its next attempt"
)]
fn every_refusal_names_its_next_attempt() {
    let linear = CyclicFamily::LinearEquilibrium;
    let vars = variables(2);
    let resolve = |family, laws: &[Arc<str>], request: CyclicWorldRequest| {
        resolve_cyclic_world(family, &vars, laws, &request).expect_err("refused")
    };
    let laws: Vec<Arc<str>> = vec![Arc::from("a0"), Arc::from("b0")];
    let cases = [
        (
            resolve(
                linear,
                &[],
                CyclicWorldRequest::new(
                    world(0),
                    [Intervention::stochastic(var(0), StochasticPolicy::bernoulli(0.5))],
                ),
            ),
            CYCLIC_PER_LEVEL_CLAMP_ATTEMPT,
        ),
        (
            resolve(
                linear,
                &[],
                CyclicWorldRequest::new(
                    world(0),
                    [Intervention::transform(var(0), ValueExpr::generated())],
                ),
            ),
            CYCLIC_SOFT_REPLACEMENT_ATTEMPT,
        ),
        (
            resolve(
                linear,
                &[],
                CyclicWorldRequest::new(world(0), [set(0, 1.0)]).timed(TemporalPolicy::pulse(0)),
            ),
            CYCLIC_CONTINUOUS_TIME_ATTEMPT,
        ),
        (
            resolve(
                CyclicFamily::NonlinearEquilibrium,
                &[],
                CyclicWorldRequest::observational(world(0)).with_initial_state("a0", 1.0).unwrap(),
            ),
            CYCLIC_CONTINUOUS_TIME_ATTEMPT,
        ),
        (
            resolve(
                CyclicFamily::ContinuousTime,
                &laws,
                CyclicWorldRequest::new(world(0), [shift(0, 1.0)]),
            ),
            CYCLIC_DRIFT_INPUT_ATTEMPT,
        ),
        (
            resolve(linear, &[], CyclicWorldRequest::new(world(0), [set(7, 1.0)])),
            BIND_VARIABLE_ATTEMPT,
        ),
        (
            resolve(linear, &[], CyclicWorldRequest::new(world(0), [set(0, 1.0), shift(0, 1.0)])),
            ONE_ACTION_PER_VARIABLE_ATTEMPT,
        ),
        (
            resolve(
                linear,
                &[],
                CyclicWorldRequest::new(world(0), [Intervention::set(var(0), Value::Int64(1))]),
            ),
            FINITE_FLOAT_LEVEL_ATTEMPT,
        ),
        (
            resolve(linear, &[], CyclicWorldRequest::new(world(0), [set(0, f64::NAN)])),
            FINITE_FLOAT_LEVEL_ATTEMPT,
        ),
        (
            resolve(
                linear,
                &[],
                CyclicWorldRequest::new(world(0), [soft(0, "multiplicative", vec![2.0])]),
            ),
            SUPPORTED_SOFT_FAMILY_ATTEMPT,
        ),
        (
            resolve(
                linear,
                &[],
                CyclicWorldRequest::new(world(0), [soft(0, "constant", vec![1.0, 2.0])]),
            ),
            SOFT_PARAMETER_SHAPE_ATTEMPT,
        ),
        (
            resolve(
                linear,
                &[],
                CyclicWorldRequest::new(
                    world(0),
                    [soft(1, "linear_mechanism", vec![0.0, 0.3, 0.2])],
                ),
            ),
            NONZERO_DIAGONAL_ATTEMPT,
        ),
        (
            resolve(
                CyclicFamily::ContinuousTime,
                &laws,
                CyclicWorldRequest::observational(world(0)).with_initial_state("z0", 1.0).unwrap(),
            ),
            DECLARED_INITIAL_LAW_ATTEMPT,
        ),
        (
            resolve(
                linear,
                &[],
                CyclicWorldRequest::new(
                    world(0),
                    [Intervention::sequence(InterventionSequence::new(vec![
                        SequencedIntervention::new(set(0, 1.0), TemporalPolicy::pulse(0)),
                    ]))],
                )
                .timed(TemporalPolicy::pulse(1)),
            ),
            WELL_FORMED_INTERVENTION_ATTEMPT,
        ),
        (
            evaluate_linear_worlds(
                &feedback_pair(),
                &[
                    CyclicWorldRequest::observational(world(0)),
                    CyclicWorldRequest::observational(world(0)),
                ],
                CertificateDemand::Required,
                CompileBudget::unlimited(),
            )
            .expect_err("duplicate world"),
            DISTINCT_WORLD_ATTEMPT,
        ),
        (
            LinearCyclicSpecification::try_new(
                variables(2),
                LinearStructuralMap::try_new(vec![0.0, 0.4, 0.5, 0.0], vec![0.0; 2]).unwrap(),
                GaussianDisturbance::try_new(vec![0.0; 2], identity(2)).unwrap(),
                "feedback",
                "source-measurements",
                &["source-measurements"],
            )
            .expect_err("the source does not certify itself"),
            CERTIFYING_EVIDENCE_ATTEMPT,
        ),
        (
            CyclicVariables::try_new(vec![var(0), var(0)]).expect_err("bound twice"),
            BIND_VARIABLE_ATTEMPT,
        ),
        (
            CyclicWorldRequest::observational(world(0))
                .with_initial_state("", 1.0)
                .expect_err("unnamed law"),
            WELL_FORMED_INTERVENTION_ATTEMPT,
        ),
    ];
    for (refusal, expected) in &cases {
        assert_names(refusal, *expected);
    }
}

#[test]
fn acyclic_specification_matches_an_independent_linear_moment_computation() {
    // A → B (0.8), B → C (0.6), A → C (0.3), with correlated disturbances.
    let n = 3;
    let coefficients = vec![0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.3, 0.6, 0.0];
    let intercept = vec![0.5, -0.2, 0.1];
    let noise_mean = vec![0.1, 0.0, -0.3];
    let noise_covariance = vec![1.0, 0.2, 0.0, 0.2, 0.5, 0.1, 0.0, 0.1, 0.8];
    let spec = linear_spec(
        coefficients.clone(),
        intercept.clone(),
        noise_mean.clone(),
        noise_covariance.clone(),
    );

    // Independent reference: zero the clamped rows and use the nilpotent series
    // (I - K)^{-1} = I + K + K², without any solve.
    let reference = |clamped: &[(usize, f64)],
                     replaced: &[(usize, f64)],
                     shocked: &[(usize, f64)]|
     -> (Vec<f64>, Vec<f64>) {
        let mut k = coefficients.clone();
        let mut c = intercept.clone();
        let mut mu = noise_mean.clone();
        let mut sigma = noise_covariance.clone();
        for &(node, value) in replaced {
            k[node * n..(node + 1) * n].fill(0.0);
            c[node] = value;
        }
        for &(node, delta) in shocked {
            mu[node] += delta;
        }
        for &(node, level) in clamped {
            k[node * n..(node + 1) * n].fill(0.0);
            c[node] = level;
            mu[node] = 0.0;
            for other in 0..n {
                sigma[node * n + other] = 0.0;
                sigma[other * n + node] = 0.0;
            }
        }
        let product = |left: &[f64], right: &[f64]| {
            let mut out = vec![0.0; n * n];
            for row in 0..n {
                for col in 0..n {
                    out[row * n + col] = (0..n)
                        .map(|inner| left[row * n + inner] * right[inner * n + col])
                        .sum::<f64>();
                }
            }
            out
        };
        let squared = product(&k, &k);
        let mut series = identity(n);
        for index in 0..n * n {
            series[index] += k[index] + squared[index];
        }
        let shifted: Vec<f64> = (0..n).map(|node| c[node] + mu[node]).collect();
        let mean = (0..n)
            .map(|row| (0..n).map(|col| series[row * n + col] * shifted[col]).sum::<f64>())
            .collect();
        let mut transposed = vec![0.0; n * n];
        for row in 0..n {
            for col in 0..n {
                transposed[col * n + row] = series[row * n + col];
            }
        }
        let covariance = product(&product(&series, &sigma), &transposed);
        (mean, covariance)
    };

    // The reference mean and covariance of one world.
    let cases: Vec<(CyclicWorldRequest, Moments)> = vec![
        (CyclicWorldRequest::observational(world(0)), reference(&[], &[], &[])),
        (CyclicWorldRequest::new(world(1), [set(1, 2.0)]), reference(&[(1, 2.0)], &[], &[])),
        (CyclicWorldRequest::new(world(2), [shift(0, 0.5)]), reference(&[], &[], &[(0, 0.5)])),
        (
            CyclicWorldRequest::new(world(3), [soft(2, "constant", vec![1.0])]),
            reference(&[], &[(2, 1.0)], &[]),
        ),
        (
            CyclicWorldRequest::new(world(4), [set(0, -1.0), shift(2, 0.25)]),
            reference(&[(0, -1.0)], &[], &[(2, 0.25)]),
        ),
    ];
    for (request, (mean_truth, covariance_truth)) in &cases {
        let result = evaluate(&spec, request);
        assert!(result.certificate().is_certified(), "{:?}", result.certificate());
        assert_eq!(result.solved_by().as_str(), "acyclic_order");
        let (mean, covariance) = law(&result);
        for (value, truth) in mean.iter().zip(mean_truth) {
            assert!(
                close(*value, *truth, EXACT),
                "{} mean {mean:?} vs {mean_truth:?}",
                request.world()
            );
        }
        for (value, truth) in covariance.iter().zip(covariance_truth) {
            assert!(
                close(*value, *truth, EXACT),
                "{} covariance {covariance:?} vs {covariance_truth:?}",
                request.world()
            );
        }
    }
}

fn nonlinear_pair(weight_ab: f64, weight_ba: f64) -> NonlinearCyclicSpecification {
    let equation = |id: &str, weights: Vec<f64>| Equation {
        id: id.into(),
        intercept: 0.0,
        weights,
        activation: Activation::Identity,
        expression: None,
        parameters: vec![],
    };
    NonlinearCyclicSpecification::try_new(
        variables(2),
        EquilibriumModel {
            equations: vec![
                equation("a", vec![0.0, weight_ab]),
                equation("b", vec![weight_ba, 0.0]),
            ],
            noise_covariance: identity(2),
        },
    )
    .expect("nonlinear specification")
}

fn draw_values(result: &CyclicWorldResult) -> Vec<Vec<f64>> {
    let CyclicWorldOutcome::EquilibriumDraws { outcomes } = result.outcome() else {
        panic!("expected draws, got {:?}", result.outcome());
    };
    outcomes
        .iter()
        .map(|outcome| match outcome {
            EquilibriumOutcome::Solved(solution) => solution.values.clone(),
            EquilibriumOutcome::Failed(failure) => panic!("draw failed: {failure:?}"),
        })
        .collect()
}

#[test]
fn nonlinear_worlds_share_draws_and_keep_withheld_mass() {
    let spec = nonlinear_pair(0.4, 0.5);
    let draws = [1.0, 0.0, 0.3, -0.2];
    let ctx = ExecutionContext::for_tests(1);
    let run = |request: CyclicWorldRequest| {
        evaluate_nonlinear_world(&spec, &request, &draws, &[0.0, 0.0], SolvePolicy::default(), &ctx)
            .expect("nonlinear world")
    };
    let open = run(CyclicWorldRequest::observational(world(0)));
    let low = run(CyclicWorldRequest::new(world(1), [set(0, 0.0)]));
    let high = run(CyclicWorldRequest::new(world(2), [set(0, 1.0)]));
    let shocked = run(CyclicWorldRequest::new(world(3), [shift(0, 1.0)]));
    for result in [&open, &low, &high, &shocked] {
        assert!(result.certificate().is_certified(), "{:?}", result.certificate());
        assert_eq!(result.solved_by().as_str(), "nonlinear_blocks");
        assert!(close(result.mass().solved(), 1.0, 0.0));
    }
    let (open, low, high, shocked) =
        (draw_values(&open), draw_values(&low), draw_values(&high), draw_values(&shocked));
    for draw in 0..2 {
        assert!(close(high[draw][1] - low[draw][1], 0.5, 1.0e-9));
        assert!(close(shocked[draw][1] - open[draw][1], 0.625, 1.0e-9));
    }

    // |K| row sums of two: the contraction certificate is not earned, so every draw is withheld.
    let expanding = nonlinear_pair(2.0, 2.0);
    let withheld = evaluate_nonlinear_world(
        &expanding,
        &CyclicWorldRequest::observational(world(4)),
        &draws,
        &[0.0, 0.0],
        SolvePolicy::default(),
        &ctx,
    )
    .expect("withheld draws are a result");
    assert!(close(withheld.mass().withheld(), 1.0, 0.0));
    assert!(close(withheld.mass().failed(), 0.0, 0.0));
    assert!(matches!(
        withheld.certificate().gap(),
        Some(CertificateGap::NonlinearMassLost { failed_draws: 0, withheld_draws: 2 })
    ));
    assert!(withheld.summary().means.is_none(), "withheld mass is not renormalized away");
}

#[test]
fn continuous_time_worlds_take_initial_states_clamps_and_drift_replacements() {
    let equation = |id: &str, weights: Vec<f64>| Equation {
        id: id.into(),
        intercept: 0.0,
        weights,
        activation: Activation::Identity,
        expression: None,
        parameters: vec![],
    };
    // dA/dt = -A, dB/dt = 0.5 A - B.
    let spec = ContinuousTimeSpecification::try_new(
        variables(2),
        OdeSystem {
            equations: vec![equation("a", vec![0.0, 0.0]), equation("b", vec![0.5, 0.0])],
            decay: vec![1.0, 1.0],
        },
        vec![1.0, 0.0],
        vec![Arc::from("a0"), Arc::from("b0")],
        vec![0.0, 1.0],
    )
    .expect("continuous specification");
    let ctx = ExecutionContext::for_tests(1);
    let run = |request: CyclicWorldRequest| {
        evaluate_continuous_time_world(&spec, &request, OdePolicy::default(), &ctx)
            .expect("trajectory")
    };
    let e_inverse = (-1.0_f64).exp();

    let started =
        run(CyclicWorldRequest::observational(world(0)).with_initial_state("a0", 2.0).unwrap());
    assert!(close(started.mean_of(var(0)).unwrap(), 2.0 * e_inverse, 1.0e-6));
    assert_eq!(started.certificate().gap(), Some(&CertificateGap::NumericalIntegrationOnly));
    assert_eq!(started.solved_by().as_str(), "adaptive_integration");

    let held = run(CyclicWorldRequest::new(world(1), [set(1, 3.0)]));
    let CyclicWorldOutcome::Trajectory { trajectory } = held.outcome() else {
        panic!("expected a trajectory");
    };
    assert!(trajectory.states.iter().all(|state| close(state[1], 3.0, 0.0)));

    let replaced = run(CyclicWorldRequest::new(world(2), [soft(0, "constant", vec![2.0])]));
    assert!(close(replaced.mean_of(var(0)).unwrap(), 2.0 - e_inverse, 1.0e-6));
}

#[test]
fn results_keep_the_defining_intervention_encodings() {
    let interventions = vec![set(0, 1.5), soft(1, "additive_shift", vec![0.25])];
    let digest = |intervention: &Intervention| {
        let mut writer = CanonicalWriter::new("cyclic-world-intervention");
        intervention.encode(&mut writer);
        writer.finish()
    };
    let before: Vec<[u8; 32]> = interventions.iter().map(digest).collect();
    let action = WorldAction::at_decision(interventions.clone()).expect("action");
    let request = CyclicWorldRequest::from_action(world(9), &action);
    let result = evaluate(&feedback_pair(), &request);
    assert_eq!(result.world(), world(9));
    assert_eq!(result.request().world(), world(9));
    let after: Vec<[u8; 32]> = result.request().interventions().iter().map(digest).collect();
    assert_eq!(after, before);
    assert_eq!(result.request().interventions(), interventions.as_slice());
    assert_eq!(result.variables(), &[var(0), var(1)]);
}
