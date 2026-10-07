//! 2.3A A3 (X4) `x4_nested_fit_replay`: the binary nested-Markov pilot receipt (graph id,
//! parameterization, regime counts, fit receipt, diagnostics and three separate status
//! fields) replays independently.
//!
//! The oracle is the enumerated latent-variable SCM of the engine's own tests, summed out
//! by hand; it calls nothing in the code under test. Calibration is unmeasured, the
//! public interval route stays closed, and nothing here asserts coverage.
#![allow(
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    reason = "binary levels and tiny fixed enumerations"
)]

use antecedent_core::ExecutionContext;
use antecedent_estimate::nested_markov_binary::{
    AdmgDeclaration, FitOptions, NestedMarkovInput, PilotReport, Regime, RegimeCounts,
};
use antecedent_io::IoError;
use antecedent_io::nested_markov_artifact::{
    NESTED_MARKOV_GRAPH_ID, NestedMarkovArtifactWire, NestedMarkovConsumeLimits,
    NestedMarkovExpectation,
};

fn b(p: f64, level: usize) -> f64 {
    if level == 1 { p } else { 1.0 - p }
}

fn p2(x1: usize, u: usize) -> f64 {
    0.2 + 0.3 * x1 as f64 + 0.4 * u as f64
}
fn p3(x2: usize) -> f64 {
    0.25 + 0.5 * x2 as f64
}
fn p4(x3: usize, u: usize, x1: usize, eps: f64) -> f64 {
    0.15 + 0.35 * x3 as f64 + 0.3 * u as f64 + eps * x1 as f64
}

const PU: f64 = 0.3;

/// Observed law by explicit summation over the latent variable.
fn scm_law(px1: f64, eps: f64) -> [f64; 16] {
    let mut law = [0.0; 16];
    for x1 in 0..2 {
        for x2 in 0..2 {
            for x3 in 0..2 {
                for x4 in 0..2 {
                    let mut mass = 0.0;
                    for u in 0..2 {
                        mass += b(px1, x1)
                            * b(PU, u)
                            * b(p2(x1, u), x2)
                            * b(p3(x2), x3)
                            * b(p4(x3, u, x1, eps), x4);
                    }
                    law[x1 * 8 + x2 * 4 + x3 * 2 + x4] = mass;
                }
            }
        }
    }
    law
}

/// `E[X4 | do(X2 = x2)]` in the SCM (the X1 -> X4 effect is zero).
fn truth_mean(x2: usize) -> f64 {
    let mut m = 0.0;
    for x3 in 0..2 {
        for u in 0..2 {
            m += b(p3(x2), x3) * b(PU, u) * p4(x3, u, 0, 0.0);
        }
    }
    m
}

fn input_for(law: &[f64; 16], n: f64) -> NestedMarkovInput {
    NestedMarkovInput {
        graph: AdmgDeclaration::selected(),
        regimes: vec![RegimeCounts {
            regime: Regime::Observational,
            levels: vec![2, 2, 2, 2],
            cells: law.iter().map(|p| p * n).collect(),
        }],
    }
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

fn build(law: &[f64; 16]) -> (NestedMarkovArtifactWire, PilotReport) {
    NestedMarkovArtifactWire::build(&input_for(law, 1.0e6), &FitOptions::default(), &ctx())
        .expect("the pilot fits and exports")
}

fn consume(bytes: &[u8]) -> Result<(NestedMarkovArtifactWire, PilotReport), IoError> {
    NestedMarkovArtifactWire::consume_with_limits(
        bytes,
        NestedMarkovConsumeLimits::default(),
        &ctx(),
    )
}

fn resealed(
    wire: &NestedMarkovArtifactWire,
    mutate: impl FnOnce(&mut NestedMarkovArtifactWire),
) -> Vec<u8> {
    let mut copy = wire.clone();
    mutate(&mut copy);
    copy.premises_digest = copy.expected_premises_digest().expect("premises digest");
    copy.data_digest = copy.expected_data_digest().expect("data digest");
    copy.export().expect("export")
}

fn refusal(bytes: &[u8]) -> IoError {
    consume(bytes).expect_err("the consumer must refuse")
}

#[test]
fn x4_nested_fit_replay_exact_counts_replay_value_for_value_against_the_scm() {
    let law = scm_law(0.4, 0.0);
    let (wire, report) = build(&law);
    assert_eq!(wire.graph_id, NESTED_MARKOV_GRAPH_ID);
    let bytes = wire.export().expect("export");
    let (loaded, replayed) = consume(&bytes).expect("a faithful artifact replays");
    assert_eq!(replayed, report);
    // The receipt matches the enumerated SCM: cells, parameters, contrast.
    for (cell, truth) in replayed.fit.cells.iter().zip(law.iter()) {
        assert!((cell - truth).abs() < 1e-9, "{cell} vs {truth}");
    }
    let truth = truth_mean(1) - truth_mean(0);
    assert!((loaded.receipt.contrast.model_contrast - truth).abs() < 1e-9);
    assert!((loaded.receipt.contrast.plugin_contrast - truth).abs() < 1e-9);
    assert!(loaded.receipt.contrast.difference.abs() < 1e-9);
    assert!((loaded.receipt.parameters.a - 0.6).abs() < 1e-9);
    assert!(loaded.receipt.diagnostics.empirical_residuals.max_abs < 1e-9);
    assert_eq!(loaded.receipt.diagnostics.method, "saturated_feasible");
    assert!(loaded.receipt.diagnostics.normalization.is_some_and(|n| n.normalized && n.positive));
    // Three separate status fields; calibration unmeasured; no interval anywhere.
    assert_eq!(loaded.receipt.status.identification, "nonparametrically_identified");
    assert_eq!(loaded.receipt.status.likelihood_fit, "converged_constraints_satisfied");
    assert_eq!(loaded.receipt.status.inference, "interval_withheld_calibration_unmeasured");
    assert_eq!(loaded.calibration, "unmeasured");
    assert_ne!(loaded.premises_digest, loaded.data_digest);
}

#[test]
fn x4_nested_fit_replay_a_constraint_violating_fit_replays_with_its_residuals() {
    // A direct X1 -> X4 effect leaves the selected ADMG: the Verma constraint fails and
    // the coordinate-ascent projection is reported, not silently fit.
    let (wire, report) = build(&scm_law(0.4, 0.15));
    let (loaded, replayed) = consume(&wire.export().expect("export")).expect("replays");
    assert_eq!(replayed, report);
    assert_eq!(loaded.receipt.diagnostics.constraint_status, "violated_by_data");
    assert_eq!(loaded.receipt.diagnostics.method, "coordinate_ascent");
    assert!(loaded.receipt.diagnostics.empirical_residuals.max_abs > 1e-3);
    assert!(loaded.receipt.diagnostics.converged && loaded.receipt.diagnostics.iterations >= 1);
    assert_eq!(loaded.receipt.status.likelihood_fit, "converged_constraints_violated_by_data");
    assert!(loaded.receipt.contrast.difference.abs() > 1e-9);
    // A tampered residual or status refuses even though no digest covers the receipt.
    let mut edited = wire.clone();
    edited.receipt.diagnostics.empirical_residuals.verma[0] = 0.0;
    assert!(refusal(&edited.export().expect("export")).to_string().contains("does not replay"));
    let mut edited = wire;
    edited.receipt.status.likelihood_fit = "converged_constraints_satisfied".into();
    assert!(refusal(&edited.export().expect("export")).to_string().contains("does not replay"));
}

#[test]
fn x4_nested_fit_replay_refuses_resealed_semantic_mutations() {
    let (wire, _) = build(&scm_law(0.4, 0.0));
    // A changed count replays to a different receipt.
    let message = refusal(&resealed(&wire, |w| w.regimes[0].cells[0] += 50_000.0)).to_string();
    assert!(message.contains("does not replay"), "{message}");
    // An adjacent graph (the bidirected edge removed) is outside the one selected pilot.
    let error = refusal(&resealed(&wire, |w| w.graph.bidirected.clear()));
    assert_eq!(error.reason_code(), Some("route_not_supported"));
    assert!(error.to_string().contains("nested_markov.outside_binary_pilot"));
    // So is a different directed edge.
    let error = refusal(&resealed(&wire, |w| w.graph.directed.push((0, 3))));
    assert_eq!(error.reason_code(), Some("route_not_supported"));
    // An experimental regime is not the observational regime the functional cites.
    let error = refusal(&resealed(&wire, |w| {
        w.regimes[0].regime = "interventional".into();
        w.regimes[0].fixed = vec![1];
    }));
    assert_eq!(error.reason_code(), Some("route_not_supported"));
    assert!(error.to_string().contains("nested_markov.outside_binary_pilot"));
    // A non-binary domain is outside the pilot.
    let error = refusal(&resealed(&wire, |w| w.regimes[0].levels = vec![2, 3, 2, 2]));
    assert_eq!(error.reason_code(), Some("route_not_supported"));
    // A different graph id is not the selected pilot's.
    let message = refusal(&resealed(&wire, |w| w.graph_id = "binary_admg.other.v1".into()));
    assert!(message.to_string().contains("graph or parameterization id"), "{message}");
    let message = refusal(&resealed(&wire, |w| w.parameterization = "dag_factorization".into()));
    assert!(message.to_string().contains("graph or parameterization id"), "{message}");
    // Without resealing, edited counts or graph break the digests.
    let mut edited = wire.clone();
    edited.regimes[0].cells[3] += 1.0;
    assert!(refusal(&edited.export().expect("export")).to_string().contains("data identity"));
    let mut edited = wire.clone();
    edited.graph.bidirected.clear();
    assert!(refusal(&edited.export().expect("export")).to_string().contains("premises digest"));
    // A tampered fit receipt (parameter, contrast) does not replay.
    let mut edited = wire.clone();
    edited.receipt.contrast.model_contrast += 1e-6;
    assert!(refusal(&edited.export().expect("export")).to_string().contains("does not replay"));
    let mut edited = wire;
    edited.receipt.parameters.g[0][0] += 1e-9;
    assert!(refusal(&edited.export().expect("export")).to_string().contains("does not replay"));
}

#[test]
fn x4_nested_fit_replay_enforces_an_expected_identity() {
    let (wire, _) = build(&scm_law(0.4, 0.0));
    let expected = NestedMarkovExpectation {
        premises_digest: Some(wire.premises_digest.clone()),
        data_digest: Some(wire.data_digest.clone()),
    };
    let ok = NestedMarkovArtifactWire::consume_expecting(
        &wire.export().expect("export"),
        &expected,
        NestedMarkovConsumeLimits::default(),
        &ctx(),
    );
    assert!(ok.is_ok());
    // Different counts that replay (refit of a consistent receipt) are still another
    // data identity.
    let (other, _) = build(&scm_law(0.35, 0.0));
    let message = NestedMarkovArtifactWire::consume_expecting(
        &other.export().expect("export"),
        &expected,
        NestedMarkovConsumeLimits::default(),
        &ctx(),
    )
    .expect_err("a different data identity is refused")
    .to_string();
    assert!(message.contains("differs from the consumer's expectation"), "{message}");
}

#[test]
fn x4_nested_fit_replay_refuses_an_unknown_version_foreign_feature_and_claim() {
    let (wire, _) = build(&scm_law(0.4, 0.0));
    let mut future = wire.clone();
    future.version = 9;
    assert!(matches!(
        NestedMarkovArtifactWire::decode(&future.export().expect("export")),
        Err(IoError::UnsupportedVersion { version: 9 })
    ));
    let mut foreign = wire.clone();
    foreign.required_features = vec!["something_else".into()];
    assert!(NestedMarkovArtifactWire::decode(&foreign.export().expect("export")).is_err());
    let mut calibrated = wire.clone();
    calibrated.calibration = "calibrated".into();
    assert!(NestedMarkovArtifactWire::decode(&calibrated.export().expect("export")).is_err());
    let mut interval = wire;
    interval.receipt.status.inference = "interval_published".into();
    assert!(NestedMarkovArtifactWire::decode(&interval.export().expect("export")).is_err());
}

#[test]
fn x4_nested_fit_replay_refuses_stored_sizes_above_the_consumer_limits() {
    let (wire, _) = build(&scm_law(0.4, 0.0));
    let bytes = wire.export().expect("export");
    let tight = [
        NestedMarkovConsumeLimits { max_iterations: 10, ..NestedMarkovConsumeLimits::default() },
        NestedMarkovConsumeLimits { max_regimes: 0, ..NestedMarkovConsumeLimits::default() },
        NestedMarkovConsumeLimits { max_variables: 3, ..NestedMarkovConsumeLimits::default() },
        NestedMarkovConsumeLimits { max_cells: 8, ..NestedMarkovConsumeLimits::default() },
    ];
    for limits in tight {
        let message = NestedMarkovArtifactWire::consume_with_limits(&bytes, limits, &ctx())
            .expect_err("a stored size above the consumer limit is refused")
            .to_string();
        assert!(message.contains("consumer limit exceeded"), "{message}");
    }
}

#[test]
fn x4_nested_fit_replay_producer_refuses_outside_the_pilot_without_a_nonidentification_claim() {
    let mut adjacent = input_for(&scm_law(0.4, 0.0), 1.0e6);
    adjacent.graph.bidirected.clear();
    let error = NestedMarkovArtifactWire::build(&adjacent, &FitOptions::default(), &ctx())
        .expect_err("an adjacent ADMG is refused");
    assert_eq!(error.reason_code(), Some("route_not_supported"));
    let message = error.to_string();
    assert!(message.contains("nested_markov.outside_binary_pilot"), "{message}");
    assert!(!message.contains("nonidentif") && !message.contains("not_identified"), "{message}");
}
