//! The transport identity moves with every field that changes a result or a
//! verdict, each changed at unchanged shape: the marginal parameter law, the
//! produced-variable support, a parent's fitted domain, the conditioning set
//! and the numbers of a finite table. A natively refitted finite table is a
//! typed finding, not an unsupported-error.

mod transport_fixture;

use antecedent_model::compose::{
    CompositionError, Domain, MechanismForm, MechanismUncertainty, ParameterId, ParameterTerm,
    PosteriorDependence, TabularRow, UncertaintyMeaning,
};
use antecedent_model::mechanism_transport::{
    DETAIL_FAMILY_UNSUPPORTED, MechanismTransportRequest, transport_linear_gaussian,
};
use transport_fixture::{
    Selection, diagram, edges, outcome_invariance, source_x, source_y, target_context, target_x,
};

fn request() -> MechanismTransportRequest {
    MechanismTransportRequest {
        target: target_context(),
        diagram: diagram(Selection::TreatmentSelected),
        source_mechanisms: vec![source_x(), source_y()],
        target_mechanisms: vec![target_x()],
        edges: edges(),
        invariances: vec![outcome_invariance()],
        dependence: PosteriorDependence::Unknown,
    }
}

fn identity_of(request: &MechanismTransportRequest) -> [u8; 32] {
    transport_linear_gaussian(request).unwrap().identity()
}

fn uncertainty(variance: f64) -> MechanismUncertainty {
    MechanismUncertainty {
        meaning: UncertaintyMeaning::SamplingDistribution,
        parameters: vec![ParameterTerm::Intercept, ParameterTerm::Coefficient("x".into())],
        covariance: vec![variance, 0.0, 0.0, variance],
    }
}

#[test]
fn joint_law_meaning_moves_identity_and_parameter_order_is_canonical() {
    let x = ParameterId { mechanism: "x".into(), term: ParameterTerm::Intercept };
    let y = ParameterId { mechanism: "y".into(), term: ParameterTerm::Intercept };
    let slope = ParameterId { mechanism: "y".into(), term: ParameterTerm::Coefficient("x".into()) };
    let covariance = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let mut sampling = request();
    sampling.dependence = PosteriorDependence::Joint {
        meaning: UncertaintyMeaning::SamplingDistribution,
        parameters: vec![x.clone(), y.clone(), slope.clone()],
        covariance: covariance.clone(),
    };
    let mut posterior = sampling.clone();
    posterior.dependence = PosteriorDependence::Joint {
        meaning: UncertaintyMeaning::Posterior,
        parameters: vec![x.clone(), y.clone(), slope.clone()],
        covariance: covariance.clone(),
    };
    assert_ne!(identity_of(&sampling), identity_of(&posterior));
    let mut reordered = sampling.clone();
    reordered.dependence = PosteriorDependence::Joint {
        meaning: UncertaintyMeaning::SamplingDistribution,
        parameters: vec![slope, x, y],
        covariance,
    };
    assert_eq!(identity_of(&sampling), identity_of(&reordered));
}

#[test]
fn a_changed_parameter_law_moves_the_identity_at_unchanged_shape() {
    let mut a = request();
    a.source_mechanisms[1].uncertainty = Some(uncertainty(0.04));
    let mut b = request();
    b.source_mechanisms[1].uncertainty = Some(uncertainty(0.09));
    let base = identity_of(&request());
    assert_ne!(identity_of(&a), identity_of(&b));
    assert_ne!(identity_of(&a), base);

    let mut meaning = request();
    let mut u = uncertainty(0.04);
    u.meaning = UncertaintyMeaning::Posterior;
    meaning.source_mechanisms[1].uncertainty = Some(u);
    assert_ne!(identity_of(&a), identity_of(&meaning));
}

#[test]
fn a_changed_support_or_parent_domain_moves_the_identity() {
    let base = identity_of(&request());

    let mut output = request();
    output.source_mechanisms[1].output_support = Domain::Interval { lo: -100.0, hi: 100.0 };
    assert_ne!(identity_of(&output), base);

    // The producer's bounded support is contained in the consumer's fitted
    // domain either way, so both stay coherent; only the fit description moves.
    let mut bounded = request();
    bounded.target_mechanisms[0].output_support = Domain::Interval { lo: -10.0, hi: 10.0 };
    let mut parent = bounded.clone();
    parent.source_mechanisms[1].parents[0].domain = Domain::Interval { lo: -1e6, hi: 1e6 };
    assert_ne!(identity_of(&bounded), base);
    assert_ne!(identity_of(&parent), identity_of(&bounded));
}

#[test]
fn a_natively_refitted_finite_table_is_a_typed_finding_not_an_unsupported_error() {
    let mut req = request();
    req.target_mechanisms[0].form = MechanismForm::Tabular {
        levels: vec!["0".into(), "1".into()],
        rows: vec![TabularRow {
            parent_levels: vec![],
            probabilities: vec![("0".into(), 0.5), ("1".into(), 0.5)],
        }],
    };
    match transport_linear_gaussian(&req).unwrap_err() {
        CompositionError::Incoherent { findings } => {
            assert!(
                findings.iter().any(|f| f.detail == DETAIL_FAMILY_UNSUPPORTED && f.subject == "x"),
                "{findings:?}"
            );
        }
        other => panic!("expected an incoherent composition, got {other:?}"),
    }
}
