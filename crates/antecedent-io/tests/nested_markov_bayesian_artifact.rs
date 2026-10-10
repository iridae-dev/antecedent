//! Independent posterior replay verifies declarations and candidate outputs.
use antecedent_core::ExecutionContext;
use antecedent_estimate::nested_markov_binary::{
    AdmgDeclaration, FitOptions, NestedMarkovInput, Regime, RegimeCounts,
};
use antecedent_io::nested_markov_bayesian_artifact::{Artifact, Expectation, Limits};
use antecedent_learn::nested_markov_bayesian::{Options, Prior};
fn input() -> NestedMarkovInput {
    NestedMarkovInput {
        graph: AdmgDeclaration::selected(),
        regimes: vec![RegimeCounts {
            regime: Regime::Observational,
            levels: vec![2; 4],
            cells: vec![1.0; 16],
        }],
    }
}
#[test]
fn posterior_receipt_replays_and_every_changed_output_refuses() {
    let ctx = ExecutionContext::for_tests(0);
    let artifact = Artifact::build(
        &input(),
        &FitOptions::default(),
        Prior::default(),
        Options { seed: 817, ..Options::default() },
        &ctx,
    )
    .unwrap();
    let expected = Expectation {
        premises_digest: Some(artifact.premises_digest.clone()),
        point: antecedent_io::nested_markov_artifact::NestedMarkovExpectation {
            data_digest: Some(artifact.point.data_digest.clone()),
            premises_digest: Some(artifact.point.premises_digest.clone()),
        },
    };
    let replay =
        Artifact::consume(&artifact.export().unwrap(), &expected, Limits::default(), &ctx).unwrap();
    assert_eq!(replay.posterior, artifact.posterior);
    for mutation in 0..13 {
        let mut changed = artifact.clone();
        match mutation {
            0 => changed.posterior.samples[0][7] += 0.001,
            1 => changed.posterior.credible[13][0] += 0.001,
            2 => changed.posterior.covariance[3] += 0.001,
            3 => changed.posterior.diagnostics[2].tail_ess += 1.0,
            4 => changed.posterior.mean[11] += 0.001,
            5 => changed.options.seed += 1,
            6 => changed.prior.alpha[4] = 2.0,
            7 => changed.point.regimes[0].cells[0] = 2.0,
            8 => changed.calibration = "calibrated".into(),
            9 => changed.inference = "validated_interval".into(),
            10 => changed.point.graph.bidirected.clear(),
            11 => changed.coordinates.swap(0, 1),
            _ => changed.rng = "unknown".into(),
        }
        assert!(
            Artifact::consume(&changed.export().unwrap(), &expected, Limits::default(), &ctx)
                .is_err(),
            "changed receipt {mutation}"
        );
    }
    assert!(
        Artifact::consume(
            &artifact.export().unwrap(),
            &expected,
            Limits { draws: 256, ..Limits::default() },
            &ctx
        )
        .is_err()
    );
    let mut unsupported = artifact.clone();
    unsupported.options.credible_mass = 0.90;
    unsupported.premises_digest = unsupported.expected_premises_digest().unwrap();
    assert!(
        Artifact::consume(
            &unsupported.export().unwrap(),
            &Expectation::default(),
            Limits::default(),
            &ctx
        )
        .unwrap_err()
        .to_string()
        .contains("fixed 95%")
    );
    let mut altered = artifact.clone();
    altered.options.seed += 1;
    altered.premises_digest = altered.expected_premises_digest().unwrap();
    assert!(
        Artifact::consume(
            &altered.export().unwrap(),
            &Expectation::default(),
            Limits::default(),
            &ctx
        )
        .unwrap_err()
        .to_string()
        .contains("receipt does not replay")
    );
}
#[test]
fn fractional_sampling_cancel_and_proposal_bounds_remain_refusals() {
    let ctx = ExecutionContext::for_tests(0);
    let mut declaration = input();
    declaration.regimes[0].cells[0] = 1.5;
    let err = Artifact::build(
        &declaration,
        &FitOptions::default(),
        Prior::default(),
        Options::default(),
        &ctx,
    )
    .unwrap_err();
    assert!(err.to_string().contains("integer multinomial"));
    let err = Artifact::build(
        &input(),
        &FitOptions::default(),
        Prior::default(),
        Options { max_proposals: 1, ..Options::default() },
        &ctx,
    )
    .unwrap_err();
    assert!(err.to_string().contains("proposal limit"));
}
