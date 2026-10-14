//! Whole-path uncertainty of the finite Markovian row against hand-computed truth.
//!
//! The response-type model: `X = U_X` with `P(U_X = 1) = 0.4`; `Y = f(X, U_Y)`
//! with `U_Y` over four response types (never, complier, defier, always) with
//! masses `q = [0.2, 0.5, 0.1, 0.2]`; `f(x, never) = 0`, `f(x, complier) = x`,
//! `f(x, defier) = 1 - x`, `f(x, always) = 1`. Worlds: `0` is `do(X = 0)`,
//! `1` is `do(X = 1)`. Every expected value is derived by hand in the comment
//! above it.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::RequestedReading;
use antecedent_core::ExecutionContext;
use antecedent_counterfactual::selective::whole_path_uncertainty::{
    SourceTreatment, UncertaintySource, UnitSampling, derive_whole_path_uncertainty,
};
use antecedent_counterfactual::selective::{
    CrossWorldExecutionError, ExogenousInference, ExogenousVariable, FactualUnit,
    FiniteExecutionRequest, FiniteExecutionResult, FiniteQuantity, FiniteScm, FiniteScmExecutor,
    FiniteVariable, FiniteWorld, NoiseCoupling, WorldAtom,
};

const TOL: f64 = 1e-12;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(11)
}

fn response_types(q: [f64; 4]) -> FiniteScm {
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

fn do_x(name: &str, level: u8) -> FiniteWorld {
    FiniteWorld { name: name.to_owned(), interventions: vec![(0, level)], routes: vec![] }
}

fn unit(id: &str, observed: &[(usize, u8)]) -> FactualUnit {
    FactualUnit { unit: id.to_owned(), observed: observed.to_vec() }
}

fn atom(world: usize, variable: usize, level: u8) -> WorldAtom {
    WorldAtom { world, variable, level }
}

fn request(units: Vec<FactualUnit>, quantity: FiniteQuantity) -> FiniteExecutionRequest {
    let scm = response_types([0.2, 0.5, 0.1, 0.2]);
    FiniteExecutionRequest {
        evidence_history: scm.history.clone(),
        scm,
        units,
        worlds: vec![do_x("x0", 0), do_x("x1", 1)],
        quantity,
        coupling: NoiseCoupling::SharedAbducedExogenous,
        exogenous: ExogenousInference::ExactEnumeration,
        target_population: None,
        reading: RequestedReading::Point,
    }
}

fn execute(request: &FiniteExecutionRequest) -> FiniteExecutionResult {
    FiniteScmExecutor::new().execute(request, &ctx()).expect("executes")
}

fn refusal(
    result: Result<impl std::fmt::Debug, CrossWorldExecutionError>,
) -> (String, String) {
    let error = result.expect_err("a typed refusal");
    let refusal = error.refusal().expect("refusal");
    (refusal.code.to_owned(), refusal.detail.clone())
}

// Three units, quantity P(Y_{x0} = 0 | evidence):
//   case (X = 1, Y = 1): consistent types complier (0.5) and always (0.2), so
//     q = 0.5 / 0.7 = 5/7 and v = q (1 - q) = 10/49;
//   (X = 0, Y = 0): consistent types never (0.2) and complier (0.5); both have
//     Y_{x0} = 0, so q = 1 and v = 0;
//   no evidence: q = 0.2 + 0.5 = 0.7 and v = 0.21.
// Mean = (5/7 + 1 + 7/10) / 3 = 169/210. Realized-outcome variance
// = (10/49 + 0 + 21/100) / 9 = 2029 / 44100. Deviations of the unit values from
// the mean are -19/210, 41/210, -22/210, so s^2 = 2526 / 88200 and the sampling
// variance of the mean is s^2 / 3 = 2526 / 264600.
#[test]
fn the_spread_is_the_enumerated_posterior_variance_and_the_unit_sampling_variance() {
    let req = request(
        vec![unit("case", &[(0, 1), (1, 1)]), unit("clean", &[(0, 0), (1, 0)]), unit("open", &[])],
        FiniteQuantity::Probability(atom(0, 1, 0)),
    );
    let result = execute(&req);
    let out =
        derive_whole_path_uncertainty(&req, &result, UnitSampling::IndependentSample, &ctx())
            .expect("derives");
    assert_eq!(out.label, "uncalibrated_spread");
    assert!((out.value - 169.0 / 210.0).abs() < TOL);
    assert_eq!(out.value.to_bits(), result.value.to_bits());
    let by_unit = |id: &str| out.units.iter().find(|u| u.unit == id).expect("unit");
    assert!((by_unit("case").value - 5.0 / 7.0).abs() < TOL);
    assert!((by_unit("case").predictive_variance - 10.0 / 49.0).abs() < TOL);
    assert!((by_unit("clean").value - 1.0).abs() < TOL);
    assert!(by_unit("clean").predictive_variance.abs() < TOL);
    assert!((by_unit("open").value - 0.7).abs() < TOL);
    assert!((by_unit("open").predictive_variance - 0.21).abs() < TOL);
    assert!((out.realized_mean_predictive_variance - 2029.0 / 44100.0).abs() < TOL);
    let sampling = out.unit_mean_sampling_variance.expect("two or more units");
    assert!((sampling - 2526.0 / 264_600.0).abs() < TOL);
    assert!((out.realized_mean_standard_error() - (2029.0_f64 / 44100.0).sqrt()).abs() < TOL);

    // Each source has exactly one account.
    for source in UncertaintySource::ALL {
        assert_eq!(out.sources.iter().filter(|a| a.source == source).count(), 1);
    }
    assert_eq!(
        out.source(UncertaintySource::Structural).unwrap().treatment,
        SourceTreatment::Premise
    );
    assert!(matches!(
        out.source(UncertaintySource::Posterior).unwrap().treatment,
        SourceTreatment::Propagated { variance } if (variance - 2029.0 / 44100.0).abs() < TOL
    ));
    assert!(matches!(
        out.source(UncertaintySource::Sampling).unwrap().treatment,
        SourceTreatment::Propagated { variance } if (variance - 2526.0 / 264_600.0).abs() < TOL
    ));
    assert!(matches!(
        out.source(UncertaintySource::Numerical).unwrap().treatment,
        SourceTreatment::Exact { observed_weight_defect } if observed_weight_defect < 1e-12
    ));
    assert_eq!(out.routes, vec!["linear_moments", "linear_moments"]);

    // Observed work: three abductions of eight joint exogenous states, one
    // summary, no draws and no fitted mechanism.
    assert_eq!(out.work.abductions, 3);
    assert_eq!(out.work.abduction_states_enumerated, 24);
    assert_eq!(out.work.summaries, 1);
    assert_eq!(out.work.shared_draws, 0);
    assert_eq!(out.work.mechanism_fits, 0);
}

// Fixed units: the mean is the estimand of these three units, so there is no
// sampling variance; only the posterior predictive variance is propagated.
#[test]
fn fixed_units_have_no_sampling_variance_and_one_unit_has_no_between_unit_variance() {
    let three = request(
        vec![unit("case", &[(0, 1), (1, 1)]), unit("clean", &[(0, 0), (1, 0)]), unit("open", &[])],
        FiniteQuantity::Probability(atom(0, 1, 0)),
    );
    let out = derive_whole_path_uncertainty(
        &three,
        &execute(&three),
        UnitSampling::FixedUnits,
        &ctx(),
    )
    .expect("derives");
    assert_eq!(out.unit_mean_sampling_variance, None);
    assert!(matches!(
        out.source(UncertaintySource::Sampling).unwrap().treatment,
        SourceTreatment::NotApplicable(_)
    ));
    assert!((out.realized_mean_predictive_variance - 2029.0 / 44100.0).abs() < TOL);
    assert_eq!(out.routes, vec!["linear_moments"]);

    // One unit under an independent-sample declaration: the between-unit
    // variance is not determined, and it is reported as unavailable, never as
    // zero.
    let one = request(
        vec![unit("case", &[(0, 1), (1, 1)])],
        FiniteQuantity::Probability(atom(0, 1, 0)),
    );
    let out = derive_whole_path_uncertainty(
        &one,
        &execute(&one),
        UnitSampling::IndependentSample,
        &ctx(),
    )
    .expect("derives");
    assert_eq!(out.unit_mean_sampling_variance, None);
    assert!(matches!(
        out.source(UncertaintySource::Sampling).unwrap().treatment,
        SourceTreatment::Unavailable(_)
    ));
    assert!((out.realized_mean_predictive_variance - 10.0 / 49.0).abs() < TOL);
}

// A cross-world joint event with no evidence: P(Y_{x1} = 1, Y_{x0} = 0) is the
// complier mass 0.5, an indicator, so v = 0.5 * 0.5 = 0.25. The marginal
// variances 0.7 * 0.3 = 0.21 of the two worlds do not give it.
// The contrast Y_{x1} - Y_{x0} is 0 (never), 1 (complier), -1 (defier), 0
// (always): q = 0.5 - 0.1 = 0.4, E[g^2] = 0.5 + 0.1 = 0.6, v = 0.6 - 0.16 = 0.44;
// the independent-noise value 0.21 + 0.21 = 0.42 is a different joint law.
#[test]
fn a_cross_world_spread_uses_the_shared_exogenous_joint_law() {
    let joint = request(
        vec![unit("open", &[])],
        FiniteQuantity::JointProbability(vec![atom(1, 1, 1), atom(0, 1, 0)]),
    );
    let out =
        derive_whole_path_uncertainty(&joint, &execute(&joint), UnitSampling::FixedUnits, &ctx())
            .expect("derives");
    assert!((out.units[0].value - 0.5).abs() < TOL);
    assert!((out.units[0].predictive_variance - 0.25).abs() < TOL);
    assert_eq!(out.routes, vec!["aligned_joint_law"]);

    let contrast = request(
        vec![unit("open", &[])],
        FiniteQuantity::MeanContrast { plus: (1, 1), minus: (0, 1) },
    );
    let out = derive_whole_path_uncertainty(
        &contrast,
        &execute(&contrast),
        UnitSampling::FixedUnits,
        &ctx(),
    )
    .expect("derives");
    assert!((out.units[0].value - 0.4).abs() < TOL);
    assert!((out.units[0].predictive_variance - 0.44).abs() < TOL);
    assert!((out.units[0].predictive_variance - 0.42).abs() > 1e-3);

    // The mean contrast itself executes under any declared coupling (it is
    // linear), but its spread is a property of the joint law: independent or
    // undeclared world noise is refused for the spread.
    let independent =
        FiniteExecutionRequest { coupling: NoiseCoupling::IndependentWorldNoise, ..contrast.clone() };
    let result = execute(&independent);
    assert!((result.value - 0.4).abs() < TOL);
    let (code, detail) = refusal(derive_whole_path_uncertainty(
        &independent,
        &result,
        UnitSampling::FixedUnits,
        &ctx(),
    ));
    assert_eq!(code, "joint_law_required");
    assert_eq!(detail, "joint_law_propagation.nonlinear_needs_aligned_joint_law");
    let undeclared = FiniteExecutionRequest { coupling: NoiseCoupling::Undeclared, ..contrast };
    let result = execute(&undeclared);
    let (code, detail) = refusal(derive_whole_path_uncertainty(
        &undeclared,
        &result,
        UnitSampling::FixedUnits,
        &ctx(),
    ));
    assert_eq!(code, "joint_law_required");
    assert_eq!(detail, "joint_law_propagation.index_pairing_refused");
}

#[test]
fn undeclared_unit_sampling_sampled_or_transported_paths_are_refused() {
    let req = request(vec![unit("open", &[])], FiniteQuantity::Probability(atom(0, 1, 0)));
    let result = execute(&req);
    let (code, detail) = refusal(derive_whole_path_uncertainty(
        &req,
        &result,
        UnitSampling::Undeclared,
        &ctx(),
    ));
    assert_eq!(code, "sampling_dependence_unknown");
    assert_eq!(detail, "cross_world_uncertainty.unit_sampling_undeclared");

    let sampled = FiniteExecutionRequest {
        exogenous: ExogenousInference::PosteriorSampling,
        ..req.clone()
    };
    let (code, detail) = refusal(derive_whole_path_uncertainty(
        &sampled,
        &result,
        UnitSampling::FixedUnits,
        &ctx(),
    ));
    assert_eq!(code, "cell_not_licensed");
    assert_eq!(detail, "cross_world_uncertainty.sampled_path_not_licensed");

    let transported = FiniteExecutionRequest {
        target_population: Some("pop-b".to_owned()),
        ..req.clone()
    };
    let (code, detail) = refusal(derive_whole_path_uncertainty(
        &transported,
        &result,
        UnitSampling::FixedUnits,
        &ctx(),
    ));
    assert_eq!(code, "cell_not_licensed");
    assert_eq!(detail, "cross_world_uncertainty.transported_path_closed");
}

// A result is never annotated for a different request: the mechanism table at
// an unchanged shape (response-type masses), the evidence at an unchanged
// shape, and an edited value are each refused.
#[test]
fn a_result_of_another_request_is_refused_even_at_an_unchanged_shape() {
    let req = request(
        vec![unit("case", &[(0, 1), (1, 1)])],
        FiniteQuantity::Probability(atom(0, 1, 0)),
    );
    let result = execute(&req);
    derive_whole_path_uncertainty(&req, &result, UnitSampling::FixedUnits, &ctx())
        .expect("the executed pair derives");

    let mut other_mechanism = req.clone();
    other_mechanism.scm = response_types([0.3, 0.4, 0.1, 0.2]);
    let (code, detail) = refusal(derive_whole_path_uncertainty(
        &other_mechanism,
        &result,
        UnitSampling::FixedUnits,
        &ctx(),
    ));
    assert_eq!(code, "invalid_argument");
    assert_eq!(detail, "cross_world_uncertainty.result_mismatch");

    // Same unit id, evidence (X = 0, Y = 0): another posterior.
    let mut other_evidence = req.clone();
    other_evidence.units = vec![unit("case", &[(0, 0), (1, 0)])];
    let (_, detail) = refusal(derive_whole_path_uncertainty(
        &other_evidence,
        &result,
        UnitSampling::FixedUnits,
        &ctx(),
    ));
    assert_eq!(detail, "cross_world_uncertainty.result_mismatch");

    let mut edited = result.clone();
    edited.units[0].value += 1e-9;
    let (_, detail) =
        refusal(derive_whole_path_uncertainty(&req, &edited, UnitSampling::FixedUnits, &ctx()));
    assert_eq!(detail, "cross_world_uncertainty.result_mismatch");
    let mut renamed = result.clone();
    renamed.units[0].unit = "other".to_owned();
    let (_, detail) =
        refusal(derive_whole_path_uncertainty(&req, &renamed, UnitSampling::FixedUnits, &ctx()));
    assert_eq!(detail, "cross_world_uncertainty.result_mismatch");
}

#[test]
fn the_identity_changes_with_every_input_and_replays_for_the_same_pair() {
    let req = request(
        vec![unit("case", &[(0, 1), (1, 1)]), unit("open", &[])],
        FiniteQuantity::Probability(atom(0, 1, 0)),
    );
    let result = execute(&req);
    let a = derive_whole_path_uncertainty(&req, &result, UnitSampling::IndependentSample, &ctx())
        .expect("a");
    let again =
        derive_whole_path_uncertainty(&req, &result, UnitSampling::IndependentSample, &ctx())
            .expect("again");
    assert_eq!(a, again);
    let fixed = derive_whole_path_uncertainty(&req, &result, UnitSampling::FixedUnits, &ctx())
        .expect("fixed");
    assert_ne!(a.identity, fixed.identity);
    // A different executed result (another mechanism at the same shape) has
    // another identity.
    let mut other = req.clone();
    other.scm = response_types([0.3, 0.4, 0.1, 0.2]);
    let other_result = execute(&other);
    let b = derive_whole_path_uncertainty(
        &other,
        &other_result,
        UnitSampling::IndependentSample,
        &ctx(),
    )
    .expect("b");
    assert_ne!(a.identity, b.identity);
}

#[test]
fn a_cancelled_context_produces_no_spread() {
    let req = request(vec![unit("open", &[])], FiniteQuantity::Probability(atom(0, 1, 0)));
    let result = execute(&req);
    let context = ExecutionContext::for_tests(5);
    context.cancellation.cancel();
    let error =
        derive_whole_path_uncertainty(&req, &result, UnitSampling::FixedUnits, &context)
            .expect_err("cancelled");
    assert_eq!(error, CrossWorldExecutionError::Cancelled);
}
