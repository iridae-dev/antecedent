//! `cross_world_execution_v1`: sealed finite Markovian cross-world execution.
//!
//! Oracles (hand computed): response types of `Y` given `X` with masses
//! `q = [0.2, 0.5, 0.1, 0.2]` (never, complier, defier, always) and
//! `P(U_X = 1) = 0.4`. With evidence `X = 1, Y = 1`, `P(Y_{x0} = 0 | e) = 5/7`
//! and `P(e) = 0.28`; with no evidence `P(Y_{x1} = 1, Y_{x0} = 0) = 0.5`.
//!
//! Run the resealed-mutation tests with `--features test-util`.

use antecedent_contract::RequestedReading;
use antecedent_core::ExecutionContext;
use antecedent_counterfactual::selective::{
    ExogenousInference, ExogenousVariable, FactualUnit, FiniteExecutionRequest, FiniteQuantity,
    FiniteScm, FiniteVariable, FiniteWorld, NoiseCoupling, WorldAtom,
};
use antecedent_io_core::cross_world_execution_artifact::{
    CrossWorldExecutionArtifact, CrossWorldExecutionArtifactError,
};

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(5)
}

fn scm(q: [f64; 4]) -> FiniteScm {
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

fn request(
    q: [f64; 4],
    observed: &[(usize, u8)],
    quantity: FiniteQuantity,
) -> FiniteExecutionRequest {
    let scm = scm(q);
    FiniteExecutionRequest {
        evidence_history: scm.history.clone(),
        scm,
        units: vec![FactualUnit { unit: "case".to_owned(), observed: observed.to_vec() }],
        worlds: vec![do_x("x0", 0), do_x("x1", 1)],
        quantity,
        coupling: NoiseCoupling::SharedAbducedExogenous,
        exogenous: ExogenousInference::ExactEnumeration,
        target_population: None,
        reading: RequestedReading::Point,
    }
}

fn necessity() -> FiniteExecutionRequest {
    request(
        [0.2, 0.5, 0.1, 0.2],
        &[(0, 1), (1, 1)],
        FiniteQuantity::Probability(WorldAtom { world: 0, variable: 1, level: 0 }),
    )
}

fn joint() -> FiniteExecutionRequest {
    request(
        [0.2, 0.5, 0.1, 0.2],
        &[],
        FiniteQuantity::JointProbability(vec![
            WorldAtom { world: 1, variable: 1, level: 1 },
            WorldAtom { world: 0, variable: 1, level: 0 },
        ]),
    )
}

#[test]
fn sealing_and_consuming_replays_the_enumerated_truth() {
    let sealed = CrossWorldExecutionArtifact::seal(&necessity(), &ctx()).unwrap();
    assert!((sealed.result().value - 5.0 / 7.0).abs() < 1e-12);
    assert!((sealed.result().units[0].evidence_probability - 0.28).abs() < 1e-12);
    let bytes = sealed.to_bytes("necessity").unwrap();
    let consumed =
        CrossWorldExecutionArtifact::from_bytes(&bytes, Some(sealed.identity()), &ctx()).unwrap();
    assert_eq!(consumed.result().value.to_bits(), sealed.result().value.to_bits());
    assert_eq!(consumed.identity(), sealed.identity());
    assert_eq!(consumed.meta().shared_draws, sealed.meta().shared_draws);
    assert_eq!(consumed.meta().inference_claim, "point_only");
    // Byte-for-byte reproducible export.
    assert_eq!(consumed.to_bytes("necessity").unwrap(), bytes);
    // The posterior states are the shared draws: complier (u_y = 1) and
    // always (u_y = 3) with X = 1 (u_x = 1): joint state = u_x * 4 + u_y.
    assert_eq!(sealed.meta().shared_draws[0].states, vec![5, 7]);
}

#[test]
fn the_consumer_resumes_an_action_only_change_from_the_verified_abduction() {
    let sealed = CrossWorldExecutionArtifact::seal(&joint(), &ctx()).unwrap();
    let bytes = sealed.to_bytes("joint").unwrap();
    let mut consumed = CrossWorldExecutionArtifact::from_bytes(&bytes, None, &ctx()).unwrap();
    let mut next = joint();
    next.worlds.push(do_x("x1_again", 1));
    next.quantity = FiniteQuantity::Probability(WorldAtom { world: 2, variable: 1, level: 1 });
    let resumed = consumed.executor_mut().execute(&next, &ctx()).unwrap();
    assert_eq!(resumed.work.abductions, 0);
    assert_eq!(resumed.work.abductions_reused, 1);
    // P(Y_{x1} = 1) = complier + always = 0.7.
    assert!((resumed.value - 0.7).abs() < 1e-12);
}

#[test]
fn a_different_declared_identity_is_refused_even_for_a_valid_artifact() {
    let a = CrossWorldExecutionArtifact::seal(&joint(), &ctx()).unwrap();
    let mut other = joint();
    other.scm.exogenous[1].pmf = vec![0.2, 0.4, 0.2, 0.2];
    let b = CrossWorldExecutionArtifact::seal(&other, &ctx()).unwrap();
    assert!((b.result().value - 0.4).abs() < 1e-12);
    let bytes = a.to_bytes("a").unwrap();
    let error = CrossWorldExecutionArtifact::from_bytes(&bytes, Some(b.identity()), &ctx())
        .expect_err("identity retained by the consumer differs");
    assert!(matches!(
        error,
        CrossWorldExecutionArtifactError::IdentityMismatch { field: "mechanism_digest" }
    ));
    assert_eq!(error.refusal().unwrap().detail, "cross_world_execution.artifact_changed");
}

#[test]
fn unsupported_requests_cannot_be_sealed() {
    let mut transported = joint();
    transported.target_population = Some("pop-b".to_owned());
    let error = CrossWorldExecutionArtifact::seal(&transported, &ctx()).expect_err("closed");
    let refusal = error.refusal().unwrap();
    assert_eq!(refusal.code, "cell_not_licensed");
    assert!(refusal.missing.iter().any(|m| m == "transport_license_missing"));

    let mut independent = joint();
    independent.coupling = NoiseCoupling::IndependentWorldNoise;
    let error = CrossWorldExecutionArtifact::seal(&independent, &ctx()).expect_err("no joint law");
    assert_eq!(error.refusal().unwrap().code, "joint_law_required");

    let mut interval = joint();
    interval.reading = RequestedReading::Interval;
    let error = CrossWorldExecutionArtifact::seal(&interval, &ctx()).expect_err("closed");
    assert_eq!(error.refusal().unwrap().code, "cell_not_licensed");
}

#[test]
fn corrupted_bytes_and_cancellation_are_refused() {
    let sealed = CrossWorldExecutionArtifact::seal(&necessity(), &ctx()).unwrap();
    let mut bytes = sealed.to_bytes("necessity").unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0x5a;
    assert!(CrossWorldExecutionArtifact::from_bytes(&bytes, None, &ctx()).is_err());
    assert!(
        CrossWorldExecutionArtifact::from_bytes(&bytes[..bytes.len() / 2], None, &ctx()).is_err()
    );
    let cancelled = ExecutionContext::for_tests(1);
    cancelled.cancellation.cancel();
    let good = sealed.to_bytes("necessity").unwrap();
    let error = CrossWorldExecutionArtifact::from_bytes(&good, None, &cancelled).expect_err("stop");
    assert_eq!(error, CrossWorldExecutionArtifactError::Cancelled);
}

#[cfg(feature = "test-util")]
mod resealed {
    use super::*;
    use antecedent_io_core::cross_world_execution_artifact::{decode_parts, encode_parts};

    fn reseal(
        bytes: &[u8],
        edit: impl FnOnce(
            &mut antecedent_io_core::cross_world_execution_artifact::CrossWorldExecutionMeta,
            &mut Vec<f64>,
        ),
    ) -> Vec<u8> {
        let (mut meta, mut values) = decode_parts(bytes).unwrap();
        edit(&mut meta, &mut values);
        encode_parts(&meta, &values, "edited").unwrap()
    }

    fn refused(bytes: &[u8]) -> CrossWorldExecutionArtifactError {
        CrossWorldExecutionArtifact::from_bytes(bytes, None, &ctx()).expect_err("must refuse")
    }

    #[test]
    fn resealed_semantic_mutations_are_refused() {
        let good = CrossWorldExecutionArtifact::seal(&necessity(), &ctx())
            .unwrap()
            .to_bytes("necessity")
            .unwrap();
        assert!(CrossWorldExecutionArtifact::from_bytes(&good, None, &ctx()).is_ok());

        // A stored number edited: the value does not replay.
        let edited = reseal(&good, |_, values| values[0] = 0.9);
        assert!(matches!(refused(&edited), CrossWorldExecutionArtifactError::ResultMismatch(_)));

        // A mechanism parameter edited: the stored identity no longer matches.
        let mechanism =
            reseal(&good, |meta, _| meta.request.scm.exogenous[1].pmf = vec![0.25, 0.45, 0.1, 0.2]);
        assert!(matches!(
            refused(&mechanism),
            CrossWorldExecutionArtifactError::IdentityMismatch { field: "mechanism_digest" }
        ));

        // The factual conditioning edited.
        let evidence = reseal(&good, |meta, _| meta.request.units[0].observed = vec![(0, 1)]);
        assert!(matches!(
            refused(&evidence),
            CrossWorldExecutionArtifactError::IdentityMismatch { .. }
                | CrossWorldExecutionArtifactError::ResultMismatch(_)
        ));

        // A world definition edited.
        let world = reseal(&good, |meta, _| meta.request.worlds[0].interventions = vec![(0, 1)]);
        assert!(matches!(
            refused(&world),
            CrossWorldExecutionArtifactError::IdentityMismatch { .. }
                | CrossWorldExecutionArtifactError::ResultMismatch(_)
        ));

        // The shared exogenous states edited.
        let draws = reseal(&good, |meta, _| meta.shared_draws[0].states = vec![5, 6]);
        assert!(matches!(refused(&draws), CrossWorldExecutionArtifactError::ResultMismatch(_)));

        // The declared coupling upgraded from independent to shared on a request
        // that was sealed as shared is a no-op; downgrade it to independent and
        // the joint-law rule refuses the replay.
        let joint_bytes =
            CrossWorldExecutionArtifact::seal(&joint(), &ctx()).unwrap().to_bytes("joint").unwrap();
        let coupling = reseal(&joint_bytes, |meta, _| {
            meta.request.coupling = "independent_world_noise".to_owned();
        });
        let error = refused(&coupling);
        assert_eq!(error.refusal().unwrap().code, "joint_law_required");

        // A claim upgrade is unsupported semantics.
        let claim = reseal(&good, |meta, _| meta.inference_claim = "interval".to_owned());
        assert!(matches!(
            refused(&claim),
            CrossWorldExecutionArtifactError::UnsupportedSemantics(_)
        ));

        // Another major version is refused before the payload is read.
        let version = reseal(&good, |meta, _| meta.version = 2);
        assert!(matches!(
            refused(&version),
            CrossWorldExecutionArtifactError::UnsupportedVersion { version: 2 }
        ));
    }
}
