//! Export of a discretized composition and its fresh-consumer re-derivation.
//!
//! Oracle: `X = e_X`, `Y = 0.25 + X + e_Y`, two levels each. X cells split at 0
//! and carry `+-2 phi(0) = +-0.797884560...`; the consumer re-derives these from
//! the carried premises, so every claimed number is checked against the
//! re-derivation, never read back.

use antecedent_core::ExecutionContext;
use antecedent_model::compose::{
    ComposedModelSpec, Domain, MechanismContext, MechanismDescriptor, MechanismForm,
    MechanismOrigin, MechanismProvenance, ParentBinding, PosteriorDependence,
};
use antecedent_model::discretized_scm::{
    DiscretizationMethod, DiscretizationPlan, DiscretizedLinearGaussian,
};
use antecedent_model::discretized_scm_artifact::{
    DETAIL_CLAIM_MISMATCH, DETAIL_IDENTITY_MISMATCH, DETAIL_MALFORMED, DETAIL_REDERIVATION_REFUSED,
    DETAIL_VERSION_UNSUPPORTED, DISCRETIZED_SCM_ARTIFACT_VERSION, consume_discretized_scm,
    export_discretized_scm,
};

const TWO_PHI_ZERO: f64 = 0.797_884_560_802_865_4;

fn lg(var: &str, intercept: f64, parents: &[(&str, f64)]) -> MechanismDescriptor {
    MechanismDescriptor {
        variable: var.into(),
        form: MechanismForm::LinearGaussian {
            intercept,
            sigma: 1.0,
            coefficients: parents.iter().map(|(p, c)| ((*p).to_owned(), *c)).collect(),
        },
        parents: parents
            .iter()
            .map(|(p, _)| ParentBinding {
                variable: (*p).to_owned(),
                units: "u".into(),
                transform: "identity".into(),
                domain: Domain::Real,
            })
            .collect(),
        conditioning_set: parents.iter().map(|(p, _)| (*p).to_owned()).collect(),
        output_units: "u".into(),
        output_transform: "identity".into(),
        output_support: Domain::Real,
        context: MechanismContext { population: "pop".into(), regime: "observational".into() },
        provenance: MechanismProvenance {
            origin: MechanismOrigin::Fitted,
            artifact_id: format!("fit:{var}"),
            evidence_id: format!("data:{var}"),
        },
        latents: Vec::new(),
        uncertainty: None,
    }
}

fn spec() -> ComposedModelSpec {
    ComposedModelSpec {
        target: MechanismContext { population: "pop".into(), regime: "observational".into() },
        mechanisms: vec![lg("X", 0.0, &[]), lg("Y", 0.25, &[("X", 1.0)])],
        edges: vec![("X".into(), "Y".into())],
        invariances: Vec::new(),
        selection_diagrams: Vec::new(),
        dependence: PosteriorDependence::Unknown,
        admit_declared_mechanisms: false,
    }
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

fn discretized() -> DiscretizedLinearGaussian {
    let plan = DiscretizationPlan::uniform(
        DiscretizationMethod::EqualProbabilityQuantile,
        &["X".to_owned(), "Y".to_owned()],
        2,
    );
    DiscretizedLinearGaussian::from_spec(&spec(), &plan, &ctx()).unwrap()
}

fn replace(bytes: &[u8], from: f64, to: f64) -> Vec<u8> {
    let from = from.to_bits().to_le_bytes();
    let at = bytes.windows(8).position(|w| w == from).expect("pattern present");
    let mut out = bytes.to_vec();
    out[at..at + 8].copy_from_slice(&to.to_bits().to_le_bytes());
    out
}

#[test]
fn a_round_trip_re_derives_the_same_discretization() {
    let original = discretized();
    let bytes = export_discretized_scm(&original).unwrap();
    let consumed = consume_discretized_scm(&bytes, Some(original.identity()), &ctx()).unwrap();
    assert_eq!(consumed.identity(), original.identity());
    let x = consumed.level_values("X").unwrap();
    assert!((x[0] + TWO_PHI_ZERO).abs() < 1e-12 && (x[1] - TWO_PHI_ZERO).abs() < 1e-12, "{x:?}");
    assert_eq!(consumed.cell_edges("X").unwrap(), &[0.0]);
    assert_eq!(
        consumed.mechanism_mean_error("Y").unwrap().to_bits(),
        original.mechanism_mean_error("Y").unwrap().to_bits()
    );
    // Deterministic export.
    assert_eq!(bytes, export_discretized_scm(&consumed).unwrap());
}

#[test]
fn malformed_bytes_and_other_versions_are_refused_typed() {
    let bytes = export_discretized_scm(&discretized()).unwrap();

    let truncated = &bytes[..bytes.len() - 3];
    assert_eq!(consume_discretized_scm(truncated, None, &ctx()).unwrap_err().detail, DETAIL_MALFORMED);

    let mut trailing = bytes.clone();
    trailing.push(0);
    assert_eq!(consume_discretized_scm(&trailing, None, &ctx()).unwrap_err().detail, DETAIL_MALFORMED);

    // The version word follows the length-prefixed tag.
    let tag_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let at = 8 + tag_len;
    let mut other = bytes.clone();
    other[at..at + 4].copy_from_slice(&(DISCRETIZED_SCM_ARTIFACT_VERSION + 1).to_le_bytes());
    let refusal = consume_discretized_scm(&other, None, &ctx()).unwrap_err();
    assert_eq!(refusal.detail, DETAIL_VERSION_UNSUPPORTED);

    assert_eq!(consume_discretized_scm(&[], None, &ctx()).unwrap_err().detail, DETAIL_MALFORMED);
}

#[test]
fn an_edited_premise_is_re_derived_and_does_not_match_the_claimed_identity() {
    let bytes = export_discretized_scm(&discretized()).unwrap();
    // Y's intercept 0.25 -> 0.5: a different composition, claims left as they were.
    let edited = replace(&bytes, 0.25, 0.5);
    let refusal = consume_discretized_scm(&edited, None, &ctx()).unwrap_err();
    assert_eq!(refusal.detail, DETAIL_IDENTITY_MISMATCH);
}

#[test]
fn an_edited_claimed_number_is_refused_against_the_re_derivation() {
    let original = discretized();
    let bytes = export_discretized_scm(&original).unwrap();
    // The last real of the claims is Y's mean-error diagnostic.
    let claimed = original.mechanism_mean_error("Y").unwrap();
    let mut edited = bytes.clone();
    let n = edited.len();
    edited[n - 8..].copy_from_slice(&(claimed * 0.5).to_bits().to_le_bytes());
    let refusal = consume_discretized_scm(&edited, None, &ctx()).unwrap_err();
    assert_eq!(refusal.detail, DETAIL_CLAIM_MISMATCH);
    assert_eq!(refusal.subject, "Y");
}

#[test]
fn a_self_consistent_artifact_for_another_composition_fails_the_expected_identity() {
    let original = discretized();
    let mut other_spec = spec();
    other_spec.mechanisms[1] = lg("Y", 0.75, &[("X", 1.0)]);
    let plan = DiscretizationPlan::uniform(
        DiscretizationMethod::EqualProbabilityQuantile,
        &["X".to_owned(), "Y".to_owned()],
        2,
    );
    let other = DiscretizedLinearGaussian::from_spec(&other_spec, &plan, &ctx()).unwrap();
    let bytes = export_discretized_scm(&other).unwrap();
    // Consistent on its own...
    assert!(consume_discretized_scm(&bytes, None, &ctx()).is_ok());
    // ... but not the composition the consumer expects.
    let refusal =
        consume_discretized_scm(&bytes, Some(original.identity()), &ctx()).unwrap_err();
    assert_eq!(refusal.detail, DETAIL_IDENTITY_MISMATCH);
}

#[test]
fn premises_that_no_longer_discretize_are_a_typed_rederivation_refusal() {
    let bytes = export_discretized_scm(&discretized()).unwrap();
    // The first 1.0 in the premises is X's residual scale; a negative scale is not
    // a composition. Whether the checker or the identity comparison names it, it is
    // refused and never consumed.
    let edited = replace(&bytes, 1.0, -1.0);
    let refusal = consume_discretized_scm(&edited, None, &ctx()).unwrap_err();
    assert!(
        refusal.detail == DETAIL_REDERIVATION_REFUSED || refusal.detail == DETAIL_IDENTITY_MISMATCH,
        "{refusal}"
    );
}
