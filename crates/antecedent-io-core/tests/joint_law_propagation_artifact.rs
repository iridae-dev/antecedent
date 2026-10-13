//! `joint_law_propagation_v1`: sealed composition of uncertain quantities.
//!
//! Hand-derived oracles. Contrast: `Var(X - Y) = 4 + 9 - 2 * 3 = 7`.
//! Mixture: weights `(0.25, 0.75)`, means `(0, 4)`, variances `(1, 4)` give
//! mean `3` and variance `0.25 + 3 + (0.25 * 9 + 0.75 * 1) = 6.25`. Finite
//! joint law `(0,0) 0.1, (0,2) 0.4, (1,0) 0.3, (1,2) 0.2`: `E[XY] = 0.4`,
//! `E[min(X, Y)] = 0.2`. Bernoulli marginals `p = 0.3`, `q = 0.6`:
//! `E[XY]` lies in `[0, 0.3]`, `P(X + Y <= 0)` in `[0.1, 0.4]`.

use antecedent_contract::{
    InputAlignment, InputRole, PropagationDependence, PropagationInput, PropagationOperation,
    PropagationRequest, RequestedReading,
};
use antecedent_core::ExecutionContext;
use antecedent_io_core::joint_law_propagation_artifact::{
    EnvelopeQueryWire, JointLawPropagationArtifactError, JointLawPropagationArtifactWire,
    MarginalWire, PropagationRequestWire, float_bits, joint_law_propagation_identity,
};

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(11)
}

fn summary(id: &str, evidence: &str) -> PropagationInput {
    PropagationInput {
        id: id.to_owned(),
        evidence_id: evidence.to_owned(),
        role: InputRole::Estimate,
        alignment: InputAlignment::MomentsOnly,
    }
}

fn aligned(id: &str, evidence: &str, alignment: &str) -> PropagationInput {
    PropagationInput {
        id: id.to_owned(),
        evidence_id: evidence.to_owned(),
        role: InputRole::Estimate,
        alignment: InputAlignment::Aligned { alignment_id: alignment.to_owned() },
    }
}

fn declared(
    operation: PropagationOperation,
    inputs: Vec<PropagationInput>,
    dependence: PropagationDependence,
    reading: RequestedReading,
) -> PropagationRequest {
    PropagationRequest { operation, inputs, dependence, reading }
}

fn bits(value: f64) -> u64 {
    value.to_bits()
}

fn value(wire: &JointLawPropagationArtifactWire, name: &str) -> f64 {
    let (_, b) = wire.result.iter().find(|(n, _)| n == name).expect("named value");
    f64::from_bits(*b)
}

fn details(error: &JointLawPropagationArtifactError) -> Vec<(String, String)> {
    error.to_refusals().into_iter().map(|r| (r.code.to_owned(), r.detail)).collect()
}

fn reseal(wire: &mut JointLawPropagationArtifactWire) {
    wire.identity = joint_law_propagation_identity(
        &wire.rule,
        &wire.inputs,
        &wire.request,
        &wire.route,
        &wire.label,
        wire.alignment_id.as_deref(),
        wire.covariance_id.as_deref(),
        &wire.shared_evidence,
    )
    .expect("identity");
}

fn contrast() -> JointLawPropagationArtifactWire {
    JointLawPropagationArtifactWire::seal(
        &declared(
            PropagationOperation::Contrast,
            vec![summary("effect_a", "snapshot_1"), summary("effect_b", "snapshot_1")],
            PropagationDependence::Covariance { id: "cov_ab".into() },
            RequestedReading::Point,
        ),
        PropagationRequestWire::LinearCombination {
            coefficients: float_bits(&[1.0, -1.0]),
            means: float_bits(&[5.0, 2.0]),
            covariance: float_bits(&[4.0, 3.0, 3.0, 9.0]),
        },
        &ctx(),
    )
    .expect("seal")
}

fn joint_rows() -> (Vec<f64>, Vec<f64>) {
    let x = [0.0, 0.0, 1.0, 1.0];
    let y = [0.0, 2.0, 0.0, 2.0];
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    for (i, count) in [1, 4, 3, 2].into_iter().enumerate() {
        for _ in 0..count {
            xs.push(x[i]);
            ys.push(y[i]);
        }
    }
    (xs, ys)
}

fn aligned_product(
    functional: &str,
    ids: (&str, &str),
    xs: &[f64],
    ys: &[f64],
) -> JointLawPropagationArtifactWire {
    JointLawPropagationArtifactWire::seal(
        &declared(
            PropagationOperation::Product,
            vec![aligned(ids.0, "snapshot_j", "joint_7"), aligned(ids.1, "snapshot_j", "joint_7")],
            PropagationDependence::AlignedJointLaw,
            RequestedReading::Point,
        ),
        PropagationRequestWire::AlignedFunctional {
            functional: functional.to_owned(),
            x: float_bits(xs),
            y: float_bits(ys),
        },
        &ctx(),
    )
    .expect("seal")
}

fn bernoulli(p: f64) -> MarginalWire {
    MarginalWire { values: float_bits(&[0.0, 1.0]), masses: float_bits(&[1.0 - p, p]) }
}

fn envelope(
    query: EnvelopeQueryWire,
    operation: PropagationOperation,
) -> JointLawPropagationArtifactWire {
    JointLawPropagationArtifactWire::seal(
        &declared(
            operation,
            vec![summary("x", "s1"), summary("y", "s2")],
            PropagationDependence::MarginalsOnly,
            RequestedReading::Envelope,
        ),
        PropagationRequestWire::FrechetEnvelope { query, x: bernoulli(0.3), y: bernoulli(0.6) },
        &ctx(),
    )
    .expect("seal")
}

#[test]
fn a_covariance_aware_contrast_replays_to_the_closed_form() {
    let wire = contrast();
    assert_eq!(wire.route, "linear_moments");
    assert_eq!(wire.label, "point_only");
    assert_eq!(wire.covariance_id.as_deref(), Some("cov_ab"));
    assert_eq!(wire.shared_evidence, vec!["snapshot_1".to_owned()]);
    assert!((value(&wire, "mean") - 3.0).abs() < 1e-15);
    assert!((value(&wire, "variance") - 7.0).abs() < 1e-12);
    let bytes = wire.export().expect("export");
    let decoded = JointLawPropagationArtifactWire::decode(&bytes).expect("decode");
    let consumed = decoded.consume(Some(&wire.identity), &ctx()).expect("consume");
    assert_eq!(consumed.identity, wire.identity);
    assert_eq!(consumed.value("variance"), Some(7.0));
    assert!(!consumed.is_envelope());
}

#[test]
fn an_aligned_nonlinear_functional_matches_the_finite_joint_law_enumeration() {
    let (xs, ys) = joint_rows();
    let product = aligned_product("product", ("x", "y"), &xs, &ys);
    assert_eq!(product.route, "aligned_joint_law");
    assert_eq!(product.alignment_id.as_deref(), Some("joint_7"));
    assert!((value(&product, "mean") - 0.4).abs() < 1e-15);
    assert_eq!(product.result.iter().find(|(n, _)| n == "n_rows").map(|(_, v)| *v), Some(10));
    let minimum = aligned_product("minimum", ("x", "y"), &xs, &ys);
    assert!((value(&minimum, "mean") - 0.2).abs() < 1e-15);
    let bytes = product.export().expect("export");
    let decoded = JointLawPropagationArtifactWire::decode(&bytes).expect("decode");
    decoded.consume(None, &ctx()).expect("replays");
}

#[test]
fn a_known_weight_mixture_replays_to_the_closed_form() {
    let wire = JointLawPropagationArtifactWire::seal(
        &declared(
            PropagationOperation::Mixture,
            vec![summary("a", "s1"), summary("b", "s1")],
            PropagationDependence::KnownWeights,
            RequestedReading::Point,
        ),
        PropagationRequestWire::Mixture {
            weights: float_bits(&[0.25, 0.75]),
            means: float_bits(&[0.0, 4.0]),
            variances: float_bits(&[1.0, 4.0]),
        },
        &ctx(),
    )
    .expect("seal");
    assert!((value(&wire, "mean") - 3.0).abs() < 1e-15);
    assert!((value(&wire, "variance") - 6.25).abs() < 1e-13);
    assert!((value(&wire, "within") - 3.25).abs() < 1e-15);
    wire.consume(None, &ctx()).expect("replays");
}

#[test]
fn the_dependence_envelope_endpoints_are_the_closed_form_and_the_label_is_structural() {
    let product = envelope(EnvelopeQueryWire::ProductMoment, PropagationOperation::Product);
    assert_eq!(product.route, "frechet_envelope");
    assert_eq!(product.label, "structural_envelope");
    assert!((value(&product, "product_moment_lower") - 0.0).abs() < 1e-12);
    assert!((value(&product, "product_moment_upper") - 0.3).abs() < 1e-12);
    let cdf =
        envelope(EnvelopeQueryWire::SumCdf { threshold: bits(0.0) }, PropagationOperation::Sum);
    assert!((value(&cdf, "sum_cdf_lower") - 0.1).abs() < 1e-12);
    assert!((value(&cdf, "sum_cdf_upper") - 0.4).abs() < 1e-12);
    let var = envelope(
        EnvelopeQueryWire::LinearVariance { a: bits(1.0), b: bits(-1.0) },
        PropagationOperation::Contrast,
    );
    assert!((value(&var, "linear_variance_lower") - 0.21).abs() < 1e-12);
    assert!((value(&var, "linear_variance_upper") - 0.81).abs() < 1e-12);
    let consumed = cdf.consume(None, &ctx()).expect("replays");
    assert!(consumed.is_envelope());
    assert_eq!(consumed.label, "structural_envelope");
}

#[test]
fn a_declared_correlation_range_is_an_assumption_range() {
    let wire = JointLawPropagationArtifactWire::seal(
        &declared(
            PropagationOperation::Product,
            vec![summary("x", "s1"), summary("y", "s2")],
            PropagationDependence::CorrelationRange { rho_lo: -0.5, rho_hi: 0.5 },
            RequestedReading::Envelope,
        ),
        PropagationRequestWire::CorrelationRange {
            x_mean: bits(1.0),
            x_sd: bits(2.0),
            y_mean: bits(3.0),
            y_sd: bits(0.5),
            a: bits(1.0),
            b: bits(-1.0),
        },
        &ctx(),
    )
    .expect("seal");
    assert_eq!(wire.label, "assumption_range");
    assert!((value(&wire, "product_moment_lower") - 2.5).abs() < 1e-15);
    assert!((value(&wire, "product_moment_upper") - 3.5).abs() < 1e-15);
    assert!((value(&wire, "linear_variance_lower") - 3.25).abs() < 1e-15);
    assert!((value(&wire, "linear_variance_upper") - 5.25).abs() < 1e-15);
    wire.consume(None, &ctx()).expect("replays");
}

#[test]
fn unrelated_marginals_are_never_paired_by_index() {
    let (xs, ys) = joint_rows();
    let numbers = PropagationRequestWire::AlignedFunctional {
        functional: "product".to_owned(),
        x: float_bits(&xs),
        y: float_bits(&ys),
    };
    let unpaired = declared(
        PropagationOperation::Product,
        vec![
            PropagationInput {
                id: "x".into(),
                evidence_id: "s1".into(),
                role: InputRole::Estimate,
                alignment: InputAlignment::IndependentMarginals,
            },
            PropagationInput {
                id: "y".into(),
                evidence_id: "s2".into(),
                role: InputRole::Estimate,
                alignment: InputAlignment::IndependentMarginals,
            },
        ],
        PropagationDependence::AlignedJointLaw,
        RequestedReading::Point,
    );
    let error = JointLawPropagationArtifactWire::seal(&unpaired, numbers.clone(), &ctx())
        .expect_err("refused");
    assert_eq!(
        details(&error),
        vec![(
            "joint_law_required".to_owned(),
            "joint_law_propagation.index_pairing_refused".to_owned()
        )]
    );
    let mismatched = declared(
        PropagationOperation::Product,
        vec![aligned("x", "s1", "joint_1"), aligned("y", "s2", "joint_2")],
        PropagationDependence::AlignedJointLaw,
        RequestedReading::Point,
    );
    let error =
        JointLawPropagationArtifactWire::seal(&mismatched, numbers, &ctx()).expect_err("refused");
    assert_eq!(details(&error)[0].1, "joint_law_propagation.alignment_mismatch");
}

#[test]
fn a_transported_interval_is_composed_only_under_a_covariance_or_an_aligned_law() {
    let mut transported = summary("transported", "source_study");
    transported.role = InputRole::TransportedInterval;
    let other = summary("other_claim", "other_study");
    let numbers = PropagationRequestWire::LinearCombination {
        coefficients: float_bits(&[1.0, 1.0]),
        means: float_bits(&[1.0, 2.0]),
        covariance: float_bits(&[1.0, 0.0, 0.0, 1.0]),
    };
    let independent = declared(
        PropagationOperation::Sum,
        vec![transported.clone(), other.clone()],
        PropagationDependence::DeclaredIndependent,
        RequestedReading::Point,
    );
    let error = JointLawPropagationArtifactWire::seal(&independent, numbers.clone(), &ctx())
        .expect_err("no");
    assert_eq!(
        details(&error),
        vec![(
            "joint_law_required".to_owned(),
            "joint_law_propagation.transported_dependence_required".to_owned()
        )]
    );
    let with_covariance = declared(
        PropagationOperation::Sum,
        vec![transported, other],
        PropagationDependence::Covariance { id: "transport_cov".into() },
        RequestedReading::Point,
    );
    let wire = JointLawPropagationArtifactWire::seal(&with_covariance, numbers, &ctx())
        .expect("a covariance licenses the composition");
    assert!((value(&wire, "variance") - 2.0).abs() < 1e-15);
}

#[test]
fn semantic_mutations_are_refused() {
    let wire = contrast();
    let id = wire.identity.clone();
    // An unresealed edit of a stored number.
    let mut edited = wire.clone();
    if let PropagationRequestWire::LinearCombination { covariance, .. } = &mut edited.request {
        covariance[1] = bits(0.0);
        covariance[2] = bits(0.0);
    }
    assert_eq!(
        edited.consume(None, &ctx()),
        Err(JointLawPropagationArtifactError::IdentityMismatch)
    );
    // A resealed edit of the numbers no longer reproduces the stored result.
    reseal(&mut edited);
    assert_eq!(edited.consume(None, &ctx()), Err(JointLawPropagationArtifactError::ResultMismatch));
    // An unresealed edit of the stored result.
    let mut result_edit = wire.clone();
    result_edit.result[1].1 = bits(6.0);
    assert_eq!(
        result_edit.consume(None, &ctx()),
        Err(JointLawPropagationArtifactError::ResultMismatch)
    );
    // A wrong retained identity.
    assert_eq!(
        wire.consume(Some("not-the-identity"), &ctx()),
        Err(JointLawPropagationArtifactError::IdentityMismatch)
    );
    // A resealed edit that drops the declared covariance to an undeclared
    // dependence is refused by the rule.
    let mut unknown = wire.clone();
    unknown.rule.dependence = "unknown".to_owned();
    unknown.rule.covariance_id = None;
    unknown.covariance_id = None;
    reseal(&mut unknown);
    let error = unknown.consume(None, &ctx()).expect_err("refused");
    assert_eq!(details(&error)[0].0, "sampling_dependence_unknown");
    // A resealed edit of the operation to one the numbers do not fit.
    let mut other_operation = wire.clone();
    other_operation.rule.operation = "mixture".to_owned();
    reseal(&mut other_operation);
    assert!(other_operation.consume(None, &ctx()).is_err());
    // A resealed upgrade of an envelope's label to a point.
    let envelope_wire = envelope(EnvelopeQueryWire::ProductMoment, PropagationOperation::Product);
    let mut upgraded = envelope_wire.clone();
    upgraded.label = "point_only".to_owned();
    reseal(&mut upgraded);
    assert_eq!(
        upgraded.consume(None, &ctx()),
        Err(JointLawPropagationArtifactError::IdentityMismatch)
    );
    // A resealed edit of the alignment identity.
    let (xs, ys) = joint_rows();
    let mut moved = aligned_product("product", ("x", "y"), &xs, &ys);
    moved.alignment_id = Some("joint_8".to_owned());
    reseal(&mut moved);
    assert_eq!(
        moved.consume(None, &ctx()),
        Err(JointLawPropagationArtifactError::IdentityMismatch)
    );
    // Another format version is refused before the payload is interpreted.
    let mut future = wire.clone();
    future.version = 2;
    let bytes = future.export().expect("export");
    assert_eq!(
        JointLawPropagationArtifactWire::decode(&bytes),
        Err(JointLawPropagationArtifactError::UnsupportedVersion { version: 2 })
    );
    // The original still replays against the retained identity.
    wire.consume(Some(&id), &ctx()).expect("original replays");
}

#[test]
fn values_are_invariant_to_input_order_and_relabelling() {
    let (xs, ys) = joint_rows();
    let forward = aligned_product("product", ("x", "y"), &xs, &ys);
    let swapped = aligned_product("product", ("renamed_y", "renamed_x"), &ys, &xs);
    assert_eq!(forward.result, swapped.result);
    let one =
        envelope(EnvelopeQueryWire::SumCdf { threshold: bits(1.0) }, PropagationOperation::Sum);
    let two = JointLawPropagationArtifactWire::seal(
        &declared(
            PropagationOperation::Sum,
            vec![summary("y", "s2"), summary("x", "s1")],
            PropagationDependence::MarginalsOnly,
            RequestedReading::Envelope,
        ),
        PropagationRequestWire::FrechetEnvelope {
            query: EnvelopeQueryWire::SumCdf { threshold: bits(1.0) },
            x: bernoulli(0.6),
            y: bernoulli(0.3),
        },
        &ctx(),
    )
    .expect("seal");
    assert_eq!(one.result, two.result);
}

#[test]
fn a_cancelled_context_produces_no_claim() {
    let context = ExecutionContext::for_tests(3);
    context.cancellation.cancel();
    let error = JointLawPropagationArtifactWire::seal(
        &declared(
            PropagationOperation::Sum,
            vec![summary("a", "s1"), summary("b", "s2")],
            PropagationDependence::DeclaredIndependent,
            RequestedReading::Point,
        ),
        PropagationRequestWire::LinearCombination {
            coefficients: float_bits(&[1.0, 1.0]),
            means: float_bits(&[0.0, 0.0]),
            covariance: float_bits(&[1.0, 0.0, 0.0, 1.0]),
        },
        &context,
    )
    .expect_err("cancelled");
    assert_eq!(error, JointLawPropagationArtifactError::Cancelled);
    assert_eq!(details(&error)[0].0, "cancelled_no_claim");
}
