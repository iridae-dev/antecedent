//! Golden fixture for the observation query artifact (`observation_query_v1`).
//!
//! The golden lives at `tests/fixtures/observation_query_v1.bin`. The loader decodes it,
//! consumes it as a fresh consumer (recompiling the premises and replaying the query
//! rather than trusting the stored answer) and requires the identity a producer derives
//! today. A missing golden fails with the command that writes it.
//!
//! Generate or refresh it with
//! `cargo test -p antecedent-io-core --test artifact_compat_observation_query_golden -- --ignored write_`
//!
//! Oracle: Z ~ Bernoulli(0.5); P(S=1 | Z) = 0.2, 0.8; P(Y=1 | S, Z) = 0.1, 0.5, 0.3, 0.9
//! keyed (S=0,Z=0), (0,1), (1,0), (1,1); then P(Y=1 | S=1) = 0.78 and
//! P(Y=1 | do(S=1)) = 0.6.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::path::PathBuf;

use antecedent_contract::InferenceBasis;
use antecedent_core::ExecutionContext;
use antecedent_io_core::{
    OBSERVATION_QUERY_ARTIFACT_VERSION, ObservationQueryArtifactError,
    ObservationQueryArtifactWire, ObservationQueryWire, ObservationResultWire,
};
use antecedent_model::compose::{
    ComposedModelSpec, Domain, MechanismContext, MechanismDescriptor, MechanismForm,
    MechanismOrigin, MechanismProvenance, ParentBinding, PosteriorDependence, TabularRow,
};
use antecedent_test_support::read_fixture_bytes;

const GENERATOR: &str = "cargo test -p antecedent-io-core --test artifact_compat_observation_query_golden -- --ignored write_";
const GOLDEN: &str = "observation_query_v1.bin";

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(5)
}

fn table(
    var: &str,
    parents: &[&str],
    rows: &[(&[&str], [f64; 2])],
    origin: MechanismOrigin,
) -> MechanismDescriptor {
    let levels: Vec<String> = vec!["0".into(), "1".into()];
    MechanismDescriptor {
        variable: var.into(),
        form: MechanismForm::Tabular {
            levels: levels.clone(),
            rows: rows
                .iter()
                .map(|(config, p)| TabularRow {
                    parent_levels: parents
                        .iter()
                        .zip(config.iter())
                        .map(|(parent, l)| ((*parent).to_owned(), (*l).to_owned()))
                        .collect(),
                    probabilities: vec![("0".to_owned(), p[0]), ("1".to_owned(), p[1])],
                })
                .collect(),
        },
        parents: parents
            .iter()
            .map(|p| ParentBinding {
                variable: (*p).to_owned(),
                units: "category".into(),
                transform: "identity".into(),
                domain: Domain::Levels(levels.clone()),
            })
            .collect(),
        conditioning_set: parents.iter().map(|p| (*p).to_owned()).collect(),
        output_units: "category".into(),
        output_transform: "identity".into(),
        output_support: Domain::Levels(levels),
        context: MechanismContext { population: "pop".into(), regime: "observational".into() },
        provenance: MechanismProvenance {
            origin,
            artifact_id: format!("fit:{var}"),
            evidence_id: format!("data:{var}"),
        },
        latents: Vec::new(),
        uncertainty: None,
    }
}

fn confounded() -> ComposedModelSpec {
    let origin = MechanismOrigin::Declared;
    let mechanisms = vec![
        table("Z", &[], &[(&[], [0.5, 0.5])], origin),
        table("S", &["Z"], &[(&["0"], [0.8, 0.2]), (&["1"], [0.2, 0.8])], origin),
        table(
            "Y",
            &["S", "Z"],
            &[
                (&["0", "0"], [0.9, 0.1]),
                (&["0", "1"], [0.5, 0.5]),
                (&["1", "0"], [0.7, 0.3]),
                (&["1", "1"], [0.1, 0.9]),
            ],
            origin,
        ),
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
        admit_declared_mechanisms: true,
    }
}

fn paired_query() -> ObservationQueryWire {
    ObservationQueryWire::ConditioningVersusIntervention {
        outcome: "Y".into(),
        assignment: vec![("S".to_owned(), "1".to_owned())],
    }
}

fn sealed() -> ObservationQueryArtifactWire {
    ObservationQueryArtifactWire::seal(
        Some(&confounded()),
        paired_query(),
        InferenceBasis::DeclaredModel,
        &ctx(),
    )
    .expect("seals")
}

#[test]
#[ignore = "writes tests/fixtures; run with `-- --ignored write_`"]
fn write_observation_query_golden() {
    let dir = fixture_dir();
    std::fs::create_dir_all(&dir).expect("create the fixture directory");
    std::fs::write(dir.join(GOLDEN), sealed().export().expect("export")).expect("write the golden");
}

#[test]
fn the_observation_query_golden_is_replayed_and_gated() {
    let bytes =
        read_fixture_bytes(&fixture_dir(), GOLDEN, &format!("generate it with `{GENERATOR}`"));
    let fresh = sealed();
    let decoded = ObservationQueryArtifactWire::decode(&bytes).expect("decodes");
    assert_eq!(decoded.version, OBSERVATION_QUERY_ARTIFACT_VERSION);
    let consumed = decoded
        .consume(Some(&fresh.identity), &ctx())
        .expect("a fresh consumer replays the stored query");
    assert_eq!(consumed.operation, "conditioning_and_intervention");
    let ObservationResultWire::Paired { conditioning, intervention, .. } = &consumed.result else {
        panic!("wrong result kind");
    };
    let level = |levels: &[(String, u64)]| {
        f64::from_bits(levels.iter().find(|(l, _)| l == "1").map(|(_, b)| *b).expect("level"))
    };
    assert!((level(conditioning) - 0.78).abs() <= 1e-12);
    assert!((level(intervention) - 0.6).abs() <= 1e-12);

    // A newer version is refused at the byte boundary, before the payload is read.
    let mut newer = decoded.clone();
    newer.version = OBSERVATION_QUERY_ARTIFACT_VERSION + 1;
    let newer_bytes = antecedent_io_core::to_cbor(&newer).expect("encode");
    assert_eq!(
        ObservationQueryArtifactWire::decode(&newer_bytes).unwrap_err(),
        ObservationQueryArtifactError::UnsupportedVersion { version: 2 }
    );
    let mut feature = decoded;
    feature.required_features = vec!["future_feature".into()];
    let feature_bytes = antecedent_io_core::to_cbor(&feature).expect("encode");
    assert_eq!(
        ObservationQueryArtifactWire::decode(&feature_bytes).unwrap_err(),
        ObservationQueryArtifactError::UnsupportedSemantics("required features")
    );
}
