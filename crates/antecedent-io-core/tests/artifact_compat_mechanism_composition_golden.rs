//! Golden fixture for the mechanism composition artifact (`mechanism_composition_v1`).
//!
//! The golden lives at `tests/fixtures/mechanism_composition_v1.bin`. The loader decodes
//! it, consumes it as a fresh consumer (recompiling the premises and replaying the query
//! rather than trusting the stored answer) and requires the premises digest a producer
//! derives today. A missing golden fails with the command that writes it.
//!
//! Generate or refresh it with
//! `cargo test -p antecedent-io-core --test artifact_compat_mechanism_composition_golden -- --ignored write_`
//!
//! Oracle: X = 1 + 2 `e_X` ; M = 0.5 + 2 X + `e_M` ; Y = -1 + 0.5 X + 3 M + 0.5 `e_Y`, so
//! do(X=2) gives E[Y] = 13.5 and Var[Y] = 9.25.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::path::PathBuf;

use antecedent_core::ExecutionContext;
use antecedent_io_core::{
    CompositionQueryWire, CompositionResultWire, MECHANISM_COMPOSITION_ARTIFACT_VERSION,
    MechanismCompositionArtifactError, MechanismCompositionArtifactWire,
};
use antecedent_model::compose::{
    ComposedModelSpec, Domain, MechanismContext, MechanismDescriptor, MechanismForm,
    MechanismOrigin, MechanismProvenance, ParentBinding, PosteriorDependence,
};
use antecedent_test_support::read_fixture_bytes;

const GENERATOR: &str = "cargo test -p antecedent-io-core --test artifact_compat_mechanism_composition_golden -- --ignored write_";
const GOLDEN: &str = "mechanism_composition_v1.bin";

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(11)
}

fn lg(var: &str, intercept: f64, sigma: f64, parents: &[(&str, f64)]) -> MechanismDescriptor {
    MechanismDescriptor {
        variable: var.into(),
        form: MechanismForm::LinearGaussian {
            intercept,
            sigma,
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

fn chain_spec() -> ComposedModelSpec {
    let mechanisms = vec![
        lg("X", 1.0, 2.0, &[]),
        lg("M", 0.5, 1.0, &[("X", 2.0)]),
        lg("Y", -1.0, 0.5, &[("X", 0.5), ("M", 3.0)]),
    ];
    let mut edges = Vec::new();
    for m in &mechanisms {
        for p in &m.parents {
            edges.push((p.variable.clone(), m.variable.clone()));
        }
    }
    ComposedModelSpec {
        target: MechanismContext { population: "pop".into(), regime: "observational".into() },
        mechanisms,
        edges,
        invariances: Vec::new(),
        selection_diagrams: Vec::new(),
        dependence: PosteriorDependence::Unknown,
        admit_declared_mechanisms: false,
    }
}

fn law_query() -> CompositionQueryWire {
    CompositionQueryWire::GaussianInterventionLaw {
        interventions: vec![("X".into(), 2.0)],
        outcome: "Y".into(),
    }
}

fn sealed() -> MechanismCompositionArtifactWire {
    MechanismCompositionArtifactWire::seal(&chain_spec(), law_query(), &ctx()).expect("seals")
}

#[test]
#[ignore = "writes tests/fixtures; run with `-- --ignored write_`"]
fn write_mechanism_composition_golden() {
    let dir = fixture_dir();
    std::fs::create_dir_all(&dir).expect("create the fixture directory");
    std::fs::write(dir.join(GOLDEN), sealed().export().expect("export")).expect("write the golden");
}

#[test]
fn the_mechanism_composition_golden_is_replayed_and_gated() {
    let bytes =
        read_fixture_bytes(&fixture_dir(), GOLDEN, &format!("generate it with `{GENERATOR}`"));
    let fresh = sealed();
    let decoded = MechanismCompositionArtifactWire::decode(&bytes).expect("decodes");
    assert_eq!(decoded.version, MECHANISM_COMPOSITION_ARTIFACT_VERSION);
    let consumed = decoded
        .consume(Some(&fresh.premises_digest), &ctx())
        .expect("a fresh consumer replays the stored composition");
    assert_eq!(consumed.premises_digest, fresh.premises_digest);
    let CompositionResultWire::GaussianLaw { mean_bits, variance_bits } = &consumed.result else {
        panic!("wrong result kind");
    };
    assert!((f64::from_bits(*mean_bits) - 13.5).abs() < 1e-12);
    assert!((f64::from_bits(*variance_bits) - 9.25).abs() < 1e-12);

    // A newer version is refused at the byte boundary, before the payload is read.
    let mut newer = decoded.clone();
    newer.version = MECHANISM_COMPOSITION_ARTIFACT_VERSION + 1;
    let newer_bytes = antecedent_io_core::to_cbor(&newer).expect("encode");
    assert_eq!(
        MechanismCompositionArtifactWire::decode(&newer_bytes).unwrap_err(),
        MechanismCompositionArtifactError::UnsupportedVersion { version: 2 }
    );
    // An unknown required feature is refused even under a consistent identity.
    let mut feature = decoded;
    feature.required_features = vec!["future_feature".into()];
    let feature_bytes = antecedent_io_core::to_cbor(&feature).expect("encode");
    assert_eq!(
        MechanismCompositionArtifactWire::decode(&feature_bytes).unwrap_err(),
        MechanismCompositionArtifactError::UnsupportedSemantics("required features")
    );
}
