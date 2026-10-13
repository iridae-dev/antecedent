//! Composition-aware uncertainty through the facade.
//!
//! Oracles: for a bivariate normal pair `E[XY] = rho s_x s_y + mu_x mu_y`
//! (checked inside a band set from the closed-form `Var(XY)` before the
//! draws are made); for an enumerated joint law the contrast variance is
//! `s_x^2 + s_y^2 - 2 cov`; for two uniform marginals on `m` atoms the
//! Makarov bounds of `P(X + Y <= 1)` are `0` and `1` up to `1/m`, and
//! `E[XY]` lies between the countermonotone `1/6` and comonotone `1/3`.

use antecedent::analysis::joint_law_propagation::{
    CorrelationRangeSummary, EnvelopeQueryWire, LinearDependence, MixtureComponent, MomentInput,
    PropagationDraws, ValueFunctional, alignment_identity, consume_joint_law_propagation_artifact,
    propagate, propagate_aligned_functional, propagate_aligned_linear, propagate_correlation_range,
    propagate_dependence_envelope, propagate_linear_moments, propagate_mixture,
};
use antecedent_contract::{
    InputAlignment, InputRole, PropagationDependence, PropagationInput, PropagationOperation,
    PropagationRequest, QuantityRole, RequestedReading, ScientificQuantity,
};
use antecedent_core::{CausalRng, ExecutionContext};
use antecedent_io::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io::joint_law_propagation_artifact::{PropagationRequestWire, float_bits};
use antecedent_io::quantity_wire::DistributionMeaningWire;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(5)
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

/// Draws of a correlated normal pair and the predeclared five-sigma band of
/// the mean of their product, from the closed-form variance of `XY`.
fn correlated_normals(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>, f64) {
    let (mu_x, mu_y, s_x, s_y, rho) = (1.0, 2.0, 1.5, 0.5, 0.6);
    let mut rng = CausalRng::from_seed(seed);
    let (mut x, mut y) = (Vec::with_capacity(n), Vec::with_capacity(n));
    while x.len() < n {
        let u1 = 1.0 - rng.next_f64();
        let u2 = rng.next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        let (z1, z2) = (r * theta.cos(), r * theta.sin());
        x.push(mu_x + s_x * z1);
        y.push(mu_y + s_y * (rho * z1 + (1.0_f64 - rho * rho).sqrt() * z2));
    }
    let variance = mu_x * mu_x * s_y * s_y
        + mu_y * mu_y * s_x * s_x
        + 2.0 * mu_x * mu_y * rho * s_x * s_y
        + s_x * s_x * s_y * s_y * (1.0 + rho * rho);
    let band = 5.0 * (variance / n as f64).sqrt();
    (x, y, band)
}

#[test]
fn an_aligned_nonlinear_functional_matches_the_joint_moment_truth() {
    let n = 100_000;
    let (x, y, band) = correlated_normals(n, 20_240_917);
    let truth = 0.6 * 1.5 * 0.5 + 1.0 * 2.0;
    let joint = artifact("snap-joint", DrawAlignment::Joint, &x, &y);
    let dx = PropagationDraws::from_artifact_column("x", &joint, 0).expect("x");
    let dy = PropagationDraws::from_artifact_column("y", &joint, 1).expect("y");
    let claim = propagate_aligned_functional(
        PropagationOperation::Product,
        &dx,
        &dy,
        ValueFunctional::Product,
        &ctx(),
    )
    .expect("aligned product");
    let mean = claim.value("mean").expect("mean");
    assert!((mean - truth).abs() < band, "mean {mean} truth {truth} band {band}");
    assert_eq!(claim.route(), "aligned_joint_law");
    assert_eq!(claim.label(), "point_only");
    assert_eq!(claim.n_rows(), Some(n as u64));
    assert_eq!(claim.alignment_id(), Some(alignment_identity(&joint).expect("id").as_str()));
}

#[test]
fn index_pairing_of_unrelated_marginal_artifacts_is_refused() {
    let n = 2_000;
    let (x, y, _) = correlated_normals(n, 7);
    let left = artifact("snap-left", DrawAlignment::Joint, &x, &y);
    let right = artifact("snap-right", DrawAlignment::Joint, &y, &x);
    let lx = PropagationDraws::from_artifact_column("left_x", &left, 0).expect("lx");
    let ry = PropagationDraws::from_artifact_column("right_y", &right, 1).expect("ry");
    // Two joint laws that are not the same joint law are not paired.
    let error = propagate_aligned_functional(
        PropagationOperation::Product,
        &lx,
        &ry,
        ValueFunctional::Product,
        &ctx(),
    )
    .expect_err("refused");
    assert_eq!(error.code(), "joint_law_required");
    assert_eq!(error.details(), vec!["joint_law_propagation.alignment_mismatch"]);
    // Independently sampled marginals are never paired by index.
    let marginals = artifact("snap-marginals", DrawAlignment::IndependentMarginals, &x, &y);
    let mx = PropagationDraws::from_artifact_column("mx", &marginals, 0).expect("mx");
    let my = PropagationDraws::from_artifact_column("my", &marginals, 1).expect("my");
    let error = propagate_aligned_functional(
        PropagationOperation::NonlinearFunctional,
        &mx,
        &my,
        ValueFunctional::Minimum,
        &ctx(),
    )
    .expect_err("refused");
    assert_eq!(error.code(), "joint_law_required");
    assert_eq!(error.details(), vec!["joint_law_propagation.index_pairing_refused"]);
    assert!(alignment_identity(&marginals).is_err());
}

#[test]
fn a_covariance_aware_contrast_agrees_between_moments_and_aligned_draws() {
    // Enumerated joint law, ten rows: (0,0) x1, (0,2) x4, (1,0) x3, (1,2) x2.
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for (xv, yv, count) in [(0.0, 0.0, 1), (0.0, 2.0, 4), (1.0, 0.0, 3), (1.0, 2.0, 2)] {
        for _ in 0..count {
            x.push(xv);
            y.push(yv);
        }
    }
    // E[X] = 0.5, E[Y] = 1.2, Var X = 0.25, Var Y = 0.96, E[XY] = 0.4,
    // Cov = 0.4 - 0.6 = -0.2, Var(X - Y) = 0.25 + 0.96 + 0.4 = 1.61.
    let joint = artifact("snap-enum", DrawAlignment::Joint, &x, &y);
    let dx = PropagationDraws::from_artifact_column("x", &joint, 0).expect("x");
    let dy = PropagationDraws::from_artifact_column("y", &joint, 1).expect("y");
    let from_draws =
        propagate_aligned_linear(PropagationOperation::Contrast, (1.0, -1.0), &dx, &dy, &ctx())
            .expect("aligned contrast");
    assert!((from_draws.value("mean").expect("mean") - (0.5 - 1.2)).abs() < 1e-12);
    assert!((from_draws.value("variance").expect("var") - 1.61).abs() < 1e-12);
    assert!((from_draws.value("covariance").expect("cov") + 0.2).abs() < 1e-12);

    let inputs = [
        MomentInput {
            id: "x".into(),
            evidence_id: "snap-enum".into(),
            role: InputRole::Estimate,
            mean: 0.5,
        },
        MomentInput {
            id: "y".into(),
            evidence_id: "snap-enum".into(),
            role: InputRole::Estimate,
            mean: 1.2,
        },
    ];
    let from_moments = propagate_linear_moments(
        PropagationOperation::Contrast,
        &inputs,
        &[1.0, -1.0],
        &[0.25, -0.2, -0.2, 0.96],
        &LinearDependence::Covariance { id: "joint-enumeration".into() },
        &ctx(),
    )
    .expect("moments");
    assert!((from_moments.value("variance").expect("var") - 1.61).abs() < 1e-12);
    // The same inputs declared independent (shared evidence) are refused, and
    // an undeclared covariance with cross terms is not silently dropped.
    let error = propagate_linear_moments(
        PropagationOperation::Contrast,
        &inputs,
        &[1.0, -1.0],
        &[0.25, 0.0, 0.0, 0.96],
        &LinearDependence::DeclaredIndependent,
        &ctx(),
    )
    .expect_err("shared evidence");
    assert_eq!(error.code(), "evidence_double_counted");
    assert_eq!(error.details(), vec!["joint_law_propagation.shared_evidence_declared_independent"]);
}

#[test]
fn known_weight_mixture_and_missing_weights() {
    let components = [
        MixtureComponent {
            id: "a".into(),
            evidence_id: "s1".into(),
            weight: 0.25,
            mean: 0.0,
            variance: 1.0,
        },
        MixtureComponent {
            id: "b".into(),
            evidence_id: "s2".into(),
            weight: 0.75,
            mean: 4.0,
            variance: 4.0,
        },
    ];
    let claim = propagate_mixture(&components, &ctx()).expect("mixture");
    assert!((claim.value("mean").expect("mean") - 3.0).abs() < 1e-15);
    assert!((claim.value("variance").expect("variance") - 6.25).abs() < 1e-13);
    // The same numbers without a declaration of the weights are refused.
    let undeclared = PropagationRequest {
        operation: PropagationOperation::Mixture,
        inputs: ["a", "b"]
            .into_iter()
            .map(|id| PropagationInput {
                id: id.into(),
                evidence_id: "s1".into(),
                role: InputRole::Estimate,
                alignment: InputAlignment::MomentsOnly,
            })
            .collect(),
        dependence: PropagationDependence::Unknown,
        reading: RequestedReading::Point,
    };
    let error = propagate(
        &undeclared,
        PropagationRequestWire::Mixture {
            weights: float_bits(&[0.25, 0.75]),
            means: float_bits(&[0.0, 4.0]),
            variances: float_bits(&[1.0, 4.0]),
        },
        &ctx(),
    )
    .expect_err("refused");
    assert_eq!(error.details(), vec!["joint_law_propagation.mixture_weights_unknown"]);
}

#[test]
fn the_dependence_envelope_is_labelled_and_never_a_point_or_an_interval() {
    let m = 2_000_usize;
    let atoms: Vec<f64> = (0..m).map(|j| (j as f64 + 0.5) / m as f64).collect();
    // Unrelated marginals: two different studies, so no joint law exists.
    let a = artifact("snap-a", DrawAlignment::IndependentMarginals, &atoms, &atoms);
    let b = artifact("snap-b", DrawAlignment::IndependentMarginals, &atoms, &atoms);
    let x = PropagationDraws::from_artifact_column("x", &a, 0).expect("x");
    let y = PropagationDraws::from_artifact_column("y", &b, 1).expect("y");
    let tolerance = 2.0 / m as f64;
    let cdf = propagate_dependence_envelope(
        PropagationOperation::Sum,
        EnvelopeQueryWire::SumCdf { threshold: 1.0_f64.to_bits() },
        &x,
        &y,
        &ctx(),
    )
    .expect("envelope");
    assert_eq!(cdf.label(), "structural_envelope");
    assert!(cdf.is_envelope());
    assert!(cdf.value("sum_cdf_lower").expect("lower").abs() < tolerance);
    assert!((cdf.value("sum_cdf_upper").expect("upper") - 1.0).abs() < tolerance);
    let product = propagate_dependence_envelope(
        PropagationOperation::Product,
        EnvelopeQueryWire::ProductMoment,
        &x,
        &y,
        &ctx(),
    )
    .expect("envelope");
    assert!((product.value("product_moment_upper").expect("upper") - 1.0 / 3.0).abs() < 1e-3);
    assert!((product.value("product_moment_lower").expect("lower") - 1.0 / 6.0).abs() < 1e-3);
    // The envelope states no point and no interval.
    for refusal in [cdf.interval(), cdf.posterior()] {
        let error = refusal.expect_err("closed");
        assert_eq!(error.code(), "cell_not_licensed");
    }
    assert_eq!(
        cdf.interval().expect_err("closed").details(),
        vec!["joint_law_propagation.interval_route_closed"]
    );
    assert_eq!(
        cdf.posterior().expect_err("closed").details(),
        vec!["joint_law_propagation.posterior_route_closed"]
    );
    // A value function without a verified sign structure is refused.
    let refused = propagate_dependence_envelope(
        PropagationOperation::NonlinearFunctional,
        EnvelopeQueryWire::Functional { functional: "ratio".into() },
        &PropagationDraws::independent_marginal("x", "s1", vec![-1.0, 1.0, 2.0]),
        &PropagationDraws::independent_marginal("y", "s2", vec![-1.0, 1.0, 2.0]),
        &ctx(),
    )
    .expect_err("refused");
    assert_eq!(refused.code(), "route_not_supported");
    assert_eq!(refused.details(), vec!["joint_law_propagation.structure_not_verified"]);
}

#[test]
fn a_declared_correlation_range_is_an_assumption_range() {
    let claim = propagate_correlation_range(
        PropagationOperation::Sum,
        ("x", "s1"),
        ("y", "s2"),
        CorrelationRangeSummary { x_mean: 1.0, x_sd: 2.0, y_mean: 3.0, y_sd: 0.5, a: 1.0, b: 1.0 },
        -0.5,
        0.5,
        &ctx(),
    )
    .expect("range");
    assert_eq!(claim.label(), "assumption_range");
    assert!((claim.value("linear_variance_lower").expect("lower") - 3.25).abs() < 1e-15);
    assert!((claim.value("linear_variance_upper").expect("upper") - 5.25).abs() < 1e-15);
}

#[test]
fn the_artifact_replays_in_a_fresh_consumer_and_every_mutation_is_refused() {
    let (x, y, _) = correlated_normals(1_000, 99);
    let joint = artifact("snap-replay", DrawAlignment::Joint, &x, &y);
    let dx = PropagationDraws::from_artifact_column("x", &joint, 0).expect("x");
    let dy = PropagationDraws::from_artifact_column("y", &joint, 1).expect("y");
    let claim = propagate_aligned_functional(
        PropagationOperation::Product,
        &dx,
        &dy,
        ValueFunctional::Product,
        &ctx(),
    )
    .expect("claim");
    let bytes = claim.export().expect("export");
    let consumed = consume_joint_law_propagation_artifact(&bytes, Some(claim.identity()), &ctx())
        .expect("replays");
    assert_eq!(consumed.value("mean"), claim.value("mean"));
    assert_eq!(consumed.identity, claim.identity());
    // A retained identity that differs is refused.
    let error =
        consume_joint_law_propagation_artifact(&bytes, Some("other"), &ctx()).expect_err("refused");
    assert_eq!(error.details(), vec!["joint_law_propagation.artifact_refused"]);
    // Every sampled single-byte change either fails to decode or fails to
    // verify; none replays to a different accepted claim.
    for position in (0..bytes.len()).step_by((bytes.len() / 64).max(1)) {
        let mut mutated = bytes.clone();
        mutated[position] ^= 0x55;
        if let Ok(other) = consume_joint_law_propagation_artifact(&mutated, None, &ctx()) {
            assert_eq!(other.identity, claim.identity());
            assert_eq!(other.result, consumed.result);
        }
    }
}

#[test]
fn a_transported_interval_needs_an_aligned_law_or_a_covariance() {
    let mut transported = PropagationInput {
        id: "transported".into(),
        evidence_id: "source-study".into(),
        role: InputRole::TransportedInterval,
        alignment: InputAlignment::MomentsOnly,
    };
    let other = PropagationInput {
        id: "other_claim".into(),
        evidence_id: "source-study".into(),
        role: InputRole::Estimate,
        alignment: InputAlignment::MomentsOnly,
    };
    let numbers = PropagationRequestWire::LinearCombination {
        coefficients: float_bits(&[1.0, 1.0]),
        means: float_bits(&[1.0, 2.0]),
        covariance: float_bits(&[1.0, 0.25, 0.25, 1.0]),
    };
    let unknown = PropagationRequest {
        operation: PropagationOperation::Sum,
        inputs: vec![transported.clone(), other.clone()],
        dependence: PropagationDependence::Unknown,
        reading: RequestedReading::Point,
    };
    let error = propagate(&unknown, numbers.clone(), &ctx()).expect_err("refused");
    assert_eq!(error.code(), "joint_law_required");
    assert_eq!(error.details(), vec!["joint_law_propagation.transported_dependence_required"]);
    transported.role = InputRole::SourceSampling;
    let with_covariance = PropagationRequest {
        dependence: PropagationDependence::Covariance { id: "shared-source-evidence".into() },
        inputs: vec![transported, other],
        ..unknown
    };
    let claim = propagate(&with_covariance, numbers, &ctx()).expect("covariance licenses it");
    assert!((claim.value("variance").expect("variance") - 2.5).abs() < 1e-15);
}
