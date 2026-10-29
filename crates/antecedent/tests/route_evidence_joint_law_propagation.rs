//! Composition-aware uncertainty through the facade: one test per entry, each
//! against its own hand-derived value.
//!
//! Enumerated joint law over ten aligned rows, `(x, y)` with counts
//! `(0,0) x1, (0,2) x4, (1,0) x3, (1,2) x2`:
//! `E[X] = 0.5`, `E[Y] = 1.2`, `Var X = 0.25`, `Var Y = 0.96`,
//! `E[XY] = 0.4`, `Cov = -0.2`.

use antecedent::analysis::joint_law_propagation::{
    CorrelationRangeSummary, EnvelopeQueryWire, LinearDependence, MixtureComponent, MomentInput,
    PropagationDraws, ValueFunctional, alignment_identity, consume_joint_law_propagation_artifact,
    propagate, propagate_aligned_functional, propagate_aligned_linear, propagate_correlation_range,
    propagate_dependence_envelope, propagate_linear_moments, propagate_mixture,
};
use antecedent_contract::{
    InputAlignment, InputRole, PropagationDependence, PropagationInput, PropagationOperation,
    PropagationRequest, PropagationRoute, QuantityRole, RequestedReading, ScientificQuantity,
    check_propagation,
};
use antecedent_core::ExecutionContext;
use antecedent_io::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io::joint_law_propagation_artifact::{
    JointLawPropagationArtifactWire, PropagationRequestWire, float_bits,
};
use antecedent_io::quantity_wire::DistributionMeaningWire;

const TOL: f64 = 1e-12;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(5)
}

fn near(got: f64, want: f64) {
    assert!((got - want).abs() < TOL, "{got} vs {want}");
}

fn quantity(name: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: name.into(),
        variable_name: name.into(),
        role: QuantityRole::Outcome,
        units: "units".into(),
        population_id: "target".into(),
        regime_id: "do(a=1)".into(),
        horizon: 0,
        functional_id: "outcome".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

fn artifact(
    snapshot: &str,
    alignment: DrawAlignment,
    x: &[f64],
    y: &[f64],
) -> DistributionArtifact {
    let identity = DistributionIdentity::new(
        DistributionMeaningWire::InterventionalPredictive,
        &[quantity("x"), quantity("y")],
        alignment,
        DistributionProvenance {
            source_id: format!("study-{snapshot}"),
            provider_id: "engine".into(),
            rng_id: format!("splitmix-{snapshot}"),
            snapshot_id: snapshot.into(),
            causal_contract_id: "checked-contract".into(),
        },
    )
    .expect("identity");
    let mut draws = Vec::with_capacity(2 * x.len());
    for (a, b) in x.iter().zip(y) {
        draws.extend([*a, *b]);
    }
    DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity,
            axes: ["draw".into(), "quantity".into()],
            shape: [x.len(), 2],
            weights: None,
            supported: None,
            calibration: DistributionCalibration::Unmeasured,
            trust: DistributionTrust::NativeLicensed,
            legacy_posterior: None,
            legacy_bindings: None,
        },
        draws,
    )
    .expect("artifact")
}

fn enumerated_rows() -> (Vec<f64>, Vec<f64>) {
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for (xv, yv, count) in [(0.0, 0.0, 1), (0.0, 2.0, 4), (1.0, 0.0, 3), (1.0, 2.0, 2)] {
        for _ in 0..count {
            x.push(xv);
            y.push(yv);
        }
    }
    (x, y)
}

fn joint_columns(snapshot: &str) -> (DistributionArtifact, PropagationDraws, PropagationDraws) {
    let (x, y) = enumerated_rows();
    let joint = artifact(snapshot, DrawAlignment::Joint, &x, &y);
    let dx = PropagationDraws::from_artifact_column("x", &joint, 0).expect("x");
    let dy = PropagationDraws::from_artifact_column("y", &joint, 1).expect("y");
    (joint, dx, dy)
}

fn moment(id: &str, evidence: &str, mean: f64) -> MomentInput {
    MomentInput { id: id.into(), evidence_id: evidence.into(), role: InputRole::Estimate, mean }
}

fn aligned_input(id: &str, evidence: &str, alignment: &str) -> PropagationInput {
    PropagationInput {
        id: id.into(),
        evidence_id: evidence.into(),
        role: InputRole::Estimate,
        alignment: InputAlignment::Aligned { alignment_id: alignment.into() },
    }
}

#[test]
fn the_rule_plans_an_aligned_route_only_for_inputs_sharing_one_alignment() {
    let request = |left: PropagationInput, right: PropagationInput| PropagationRequest {
        operation: PropagationOperation::Product,
        inputs: vec![left, right],
        dependence: PropagationDependence::AlignedJointLaw,
        reading: RequestedReading::Point,
    };
    let plan = check_propagation(&request(
        aligned_input("x", "snapshot_j", "joint_7"),
        aligned_input("y", "snapshot_j", "joint_7"),
    ))
    .expect("one alignment licenses the aligned route");
    assert_eq!(plan.route, PropagationRoute::AlignedJointLaw);
    assert_eq!(plan.alignment_id.as_deref(), Some("joint_7"));
    assert_eq!(plan.label, "point_only");
    assert_eq!(plan.shared_evidence, vec!["snapshot_j".to_owned()]);

    // Two alignments are two joint laws: their rows are not paired.
    let mismatch = check_propagation(&request(
        aligned_input("x", "s1", "joint_1"),
        aligned_input("y", "s2", "joint_2"),
    ))
    .expect_err("different alignments");
    assert_eq!(mismatch.code, "joint_law_required");
    assert_eq!(mismatch.detail, "joint_law_propagation.alignment_mismatch");
    assert!(mismatch.remedy.is_some());

    // Independent marginals have no row pairing at all.
    let unpaired = |id: &str, evidence: &str| PropagationInput {
        id: id.into(),
        evidence_id: evidence.into(),
        role: InputRole::Estimate,
        alignment: InputAlignment::IndependentMarginals,
    };
    let refused = check_propagation(&request(unpaired("x", "s1"), unpaired("y", "s2")))
        .expect_err("independent marginals");
    assert_eq!(refused.detail, "joint_law_propagation.index_pairing_refused");
}

#[test]
fn an_alignment_identity_is_shared_by_the_columns_of_one_artifact_only() {
    let (joint, dx, dy) = joint_columns("snap-identity");
    let id = alignment_identity(&joint).expect("joint artifact has an identity");
    assert!(!id.is_empty());
    // The same artifact always gives the same identity.
    assert_eq!(alignment_identity(&joint).expect("again"), id);
    // Both columns carry it into a claim.
    let claim = propagate_aligned_linear(PropagationOperation::Sum, (1.0, 1.0), &dx, &dy, &ctx())
        .expect("sum");
    assert_eq!(claim.alignment_id(), Some(id.as_str()));
    // Another snapshot is another joint law and another identity.
    let (x, y) = enumerated_rows();
    let other = artifact("snap-identity-other", DrawAlignment::Joint, &x, &y);
    assert_ne!(alignment_identity(&other).expect("other"), id);
    // Independent marginals have none.
    let marginals = artifact("snap-marginals", DrawAlignment::IndependentMarginals, &x, &y);
    let refusal = alignment_identity(&marginals).expect_err("no identity");
    assert_eq!(refusal.code(), "joint_law_required");
    assert_eq!(refusal.details(), vec!["joint_law_propagation.index_pairing_refused"]);
}

#[test]
fn propagate_runs_a_declared_covariance_composition_to_its_closed_form() {
    // X + Y with means 1, 2, variances 1, 1 and covariance 0.25:
    // mean 3, variance 1 + 1 + 2 * 0.25 = 2.5.
    let request = PropagationRequest {
        operation: PropagationOperation::Sum,
        inputs: ["x", "y"]
            .into_iter()
            .zip(["s1", "s2"])
            .map(|(id, evidence)| PropagationInput {
                id: id.into(),
                evidence_id: evidence.into(),
                role: InputRole::Estimate,
                alignment: InputAlignment::MomentsOnly,
            })
            .collect(),
        dependence: PropagationDependence::Covariance { id: "pair_cov".into() },
        reading: RequestedReading::Point,
    };
    let claim = propagate(
        &request,
        PropagationRequestWire::LinearCombination {
            coefficients: float_bits(&[1.0, 1.0]),
            means: float_bits(&[1.0, 2.0]),
            covariance: float_bits(&[1.0, 0.25, 0.25, 1.0]),
        },
        &ctx(),
    )
    .expect("propagate");
    assert_eq!(claim.route(), "linear_moments");
    assert_eq!(claim.label(), "point_only");
    near(claim.value("mean").expect("mean"), 3.0);
    near(claim.value("variance").expect("variance"), 2.5);
    // Numbers that do not fit the declared route are refused, not reshaped.
    let wrong = propagate(
        &request,
        PropagationRequestWire::Mixture {
            weights: float_bits(&[0.5, 0.5]),
            means: float_bits(&[1.0, 2.0]),
            variances: float_bits(&[1.0, 1.0]),
        },
        &ctx(),
    );
    assert_eq!(wrong.expect_err("wrong shape").details(), vec!["joint_law_propagation.invalid_request"]);
}

#[test]
fn linear_moments_of_three_independent_inputs_add_their_scaled_variances() {
    // a = (1, 2, -1), mu = (1, 0, 2): mean 1 + 0 - 2 = -1;
    // diag(1, 4, 9): variance 1 + 4 * 4 + 9 = 26.
    let inputs = [moment("a", "s1", 1.0), moment("b", "s2", 0.0), moment("c", "s3", 2.0)];
    let covariance = [1.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 9.0];
    let claim = propagate_linear_moments(
        PropagationOperation::WeightedSum,
        &inputs,
        &[1.0, 2.0, -1.0],
        &covariance,
        &LinearDependence::DeclaredIndependent,
        &ctx(),
    )
    .expect("independent");
    near(claim.value("mean").expect("mean"), -1.0);
    near(claim.value("variance").expect("variance"), 26.0);
    assert_eq!(claim.route(), "linear_moments");

    // The same inputs resting on one snapshot cannot be declared independent.
    let shared = [moment("a", "s1", 1.0), moment("b", "s1", 0.0), moment("c", "s3", 2.0)];
    let refused = propagate_linear_moments(
        PropagationOperation::WeightedSum,
        &shared,
        &[1.0, 2.0, -1.0],
        &covariance,
        &LinearDependence::DeclaredIndependent,
        &ctx(),
    )
    .expect_err("shared evidence");
    assert_eq!(refused.code(), "evidence_double_counted");
}

#[test]
fn aligned_linear_draws_give_the_covariance_from_the_rows_themselves() {
    let (_joint, dx, dy) = joint_columns("snap-aligned-linear");
    // 2X + 3Y: mean 2 * 0.5 + 3 * 1.2 = 4.6;
    // variance 4 * 0.25 + 9 * 0.96 + 2 * 2 * 3 * (-0.2) = 7.24.
    let claim =
        propagate_aligned_linear(PropagationOperation::WeightedSum, (2.0, 3.0), &dx, &dy, &ctx())
            .expect("weighted sum");
    near(claim.value("mean").expect("mean"), 4.6);
    near(claim.value("variance").expect("variance"), 7.24);
    near(claim.value("covariance").expect("covariance"), -0.2);
    assert_eq!(claim.route(), "aligned_joint_law");
    // Columns of different artifacts are not paired.
    let (_other, _ox, oy) = joint_columns("snap-aligned-linear-other");
    let refused = propagate_aligned_linear(PropagationOperation::Sum, (1.0, 1.0), &dx, &oy, &ctx())
        .expect_err("different joint laws");
    assert_eq!(refused.code(), "joint_law_required");
    assert_eq!(refused.details(), vec!["joint_law_propagation.alignment_mismatch"]);
}

#[test]
fn aligned_functionals_are_expectations_under_the_joint_law() {
    let (_joint, dx, dy) = joint_columns("snap-aligned-functional");
    // E[XY] = 0.4 (see the module oracle); E[max(X, Y)]: max is 0, 2, 1, 2
    // on the four cells, so 0 * 0.1 + 2 * 0.4 + 1 * 0.3 + 2 * 0.2 = 1.5.
    let product = propagate_aligned_functional(
        PropagationOperation::Product,
        &dx,
        &dy,
        ValueFunctional::Product,
        &ctx(),
    )
    .expect("product");
    near(product.value("mean").expect("mean"), 0.4);
    let maximum = propagate_aligned_functional(
        PropagationOperation::NonlinearFunctional,
        &dx,
        &dy,
        ValueFunctional::Maximum,
        &ctx(),
    )
    .expect("maximum");
    near(maximum.value("mean").expect("mean"), 1.5);
    assert_eq!(maximum.n_rows(), Some(10));
    // A ratio over a zero denominator row is refused, not averaged.
    let refused = propagate_aligned_functional(
        PropagationOperation::NonlinearFunctional,
        &dx,
        &dy,
        ValueFunctional::Ratio,
        &ctx(),
    );
    assert_eq!(refused.expect_err("zero denominator").details(), vec!["joint_law_propagation.artifact_invalid"]);
}

#[test]
fn a_known_weight_mixture_splits_variance_into_within_and_between() {
    // Weights (1/2, 1/2), means (1, 3), variances (2, 2):
    // mean 2, within 2, between 1, variance 3.
    let component = |id: &str, evidence: &str, mean: f64| MixtureComponent {
        id: id.into(),
        evidence_id: evidence.into(),
        weight: 0.5,
        mean,
        variance: 2.0,
    };
    let claim = propagate_mixture(&[component("a", "s1", 1.0), component("b", "s2", 3.0)], &ctx())
        .expect("mixture");
    near(claim.value("mean").expect("mean"), 2.0);
    near(claim.value("within").expect("within"), 2.0);
    near(claim.value("variance").expect("variance"), 3.0);
    assert_eq!(claim.route(), "known_weight_mixture");
    // Weights that do not sum to one are refused.
    let bad = MixtureComponent { weight: 0.4, ..component("c", "s3", 0.0) };
    assert_eq!(
        propagate_mixture(&[bad.clone(), bad], &ctx()).expect_err("weights").details(),
        vec!["joint_law_propagation.invalid_request"]
    );
}

#[test]
fn a_dependence_envelope_ranges_over_every_coupling_of_two_marginals() {
    // X ~ Bernoulli(1/2) from four draws, Y ~ Bernoulli(1/4) from four draws:
    // c = P(1, 1) in [0, 1/4], Cov = c - 1/8, Var(X + Y) = 7/16 + 2 Cov.
    let x = PropagationDraws::independent_marginal("x", "s1", vec![1.0, 1.0, 0.0, 0.0]);
    let y = PropagationDraws::independent_marginal("y", "s2", vec![1.0, 0.0, 0.0, 0.0]);
    let product = propagate_dependence_envelope(
        PropagationOperation::Product,
        EnvelopeQueryWire::ProductMoment,
        &x,
        &y,
        &ctx(),
    )
    .expect("product envelope");
    near(product.value("product_moment_lower").expect("lower"), 0.0);
    near(product.value("product_moment_upper").expect("upper"), 0.25);
    assert_eq!(product.label(), "structural_envelope");
    assert!(product.is_envelope());
    let variance = propagate_dependence_envelope(
        PropagationOperation::Sum,
        EnvelopeQueryWire::LinearVariance { a: 1.0_f64.to_bits(), b: 1.0_f64.to_bits() },
        &x,
        &y,
        &ctx(),
    )
    .expect("variance envelope");
    near(variance.value("linear_variance_lower").expect("lower"), 0.1875);
    near(variance.value("linear_variance_upper").expect("upper"), 0.6875);
    // The envelope is never read as an interval.
    assert_eq!(product.interval().expect_err("closed").code(), "cell_not_licensed");
}

#[test]
fn a_correlation_range_gives_an_assumption_range_linear_in_the_correlation() {
    // x (0, 1), y (2, 3), 1 * X + 2 * Y, rho in [-0.2, 0.6]; s_x s_y = 3:
    // E[XY] in [-0.6, 1.8]; Var = 1 + 4 * 9 + 12 rho in [34.6, 44.2].
    let claim = propagate_correlation_range(
        PropagationOperation::WeightedSum,
        ("x", "s1"),
        ("y", "s2"),
        CorrelationRangeSummary { x_mean: 0.0, x_sd: 1.0, y_mean: 2.0, y_sd: 3.0, a: 1.0, b: 2.0 },
        -0.2,
        0.6,
        &ctx(),
    )
    .expect("range");
    assert_eq!(claim.label(), "assumption_range");
    near(claim.value("product_moment_lower").expect("lower"), -0.6);
    near(claim.value("product_moment_upper").expect("upper"), 1.8);
    near(claim.value("linear_variance_lower").expect("lower"), 34.6);
    near(claim.value("linear_variance_upper").expect("upper"), 44.2);
    // A single correlation is a covariance declaration, not a range.
    let refused = propagate_correlation_range(
        PropagationOperation::WeightedSum,
        ("x", "s1"),
        ("y", "s2"),
        CorrelationRangeSummary { x_mean: 0.0, x_sd: 1.0, y_mean: 2.0, y_sd: 3.0, a: 1.0, b: 2.0 },
        0.3,
        0.3,
        &ctx(),
    );
    assert_eq!(refused.expect_err("single correlation").details(), vec!["joint_law_propagation.invalid_request"]);
}

#[test]
fn an_exported_claim_is_the_sealed_artifact_of_its_own_identity_and_result() {
    let (_joint, dx, dy) = joint_columns("snap-export");
    let claim =
        propagate_aligned_linear(PropagationOperation::Contrast, (1.0, -1.0), &dx, &dy, &ctx())
            .expect("contrast");
    let bytes = claim.export().expect("export");
    assert!(!bytes.is_empty());
    // Export is a pure function of the claim.
    assert_eq!(claim.export().expect("again"), bytes);
    // The bytes decode to the sealed wire the claim carries: its identity and
    // its alignment, with Var(X - Y) = 0.25 + 0.96 + 0.4 = 1.61 stored.
    let wire = JointLawPropagationArtifactWire::decode(&bytes).expect("decode");
    assert_eq!(wire.identity, claim.identity());
    assert_eq!(wire.alignment_id.as_deref(), claim.alignment_id());
    assert_eq!(wire.route, "aligned_joint_law");
    let variance = wire.result.value("variance").expect("stored variance");
    near(variance, 1.61);
}

#[test]
fn a_fresh_consumer_replays_an_envelope_and_refuses_a_foreign_identity_or_truncation() {
    let x = PropagationDraws::independent_marginal("x", "s1", vec![0.0, 1.0]);
    let y = PropagationDraws::independent_marginal("y", "s2", vec![0.0, 1.0]);
    // Uniform {0,1} marginals: P(X + Y <= 0) = P(0,0) in [0, 1/2].
    let claim = propagate_dependence_envelope(
        PropagationOperation::Sum,
        EnvelopeQueryWire::SumCdf { threshold: 0.0_f64.to_bits() },
        &x,
        &y,
        &ctx(),
    )
    .expect("envelope");
    let bytes = claim.export().expect("export");
    let consumed = consume_joint_law_propagation_artifact(&bytes, Some(claim.identity()), &ctx())
        .expect("replays");
    assert!(consumed.is_envelope());
    assert_eq!(consumed.label, "structural_envelope");
    near(consumed.value("sum_cdf_lower").expect("lower"), 0.0);
    near(consumed.value("sum_cdf_upper").expect("upper"), 0.5);
    // A retained identity that is not this claim's is refused.
    let refusal = consume_joint_law_propagation_artifact(&bytes, Some("0000"), &ctx())
        .expect_err("foreign identity");
    assert_eq!(refusal.details(), vec!["joint_law_propagation.artifact_refused"]);
    // Truncated bytes are not an artifact.
    assert!(
        consume_joint_law_propagation_artifact(&bytes[..bytes.len() / 2], None, &ctx()).is_err()
    );
}
