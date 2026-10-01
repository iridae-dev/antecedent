//! 2.2B X3 lifecycle: the prepared z stage evaluates joint mechanism
//! deviations, exports a version 3 artifact, and an independent consumer
//! replays it; mutations fail for the right reason and the interval route is
//! closed.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::too_many_lines,
    clippy::float_cmp,
    clippy::type_complexity,
    reason = "mutation tables of boxed closures; bit-exact replay is the property under test"
)]

use std::sync::Arc;

use antecedent::{
    JointSensitivityArtifactWire, JointSensitivityConsumeLimits, PreparedZTransport, StudyBuilder,
    ZTransportSensitivityArtifactWire,
};
use antecedent_core::{
    DependenceGroup, DistributionAvailability, Environment, EvidenceCatalog, EvidenceKind,
    EvidenceRegime, ExecutionContext, InterventionAssignment, MemoryBudget, ProgressSink,
    RegimeBinding, RegimeId, RegimeKind, SamplingDesign, Value, VariableCoordinate, VariableDomain,
    VariableId,
};
use antecedent_expr::{
    Assignment, DiscreteAxis, ExactDiscreteLaw, ExactEvaluationLimits, ExactTransportData,
    LawTolerance,
};
use antecedent_graph::{Admg, DenseNodeId, SelectionDiagram};
use antecedent_identify::{
    SidLimits, ZTransportQuery, ZTransportResult, bind_z_transport_catalog, identify_z_transport,
};
use antecedent_io::IoError;
use antecedent_validate::{
    JointDeviationSpec, JointFactor, JointFactorBound, z_transport_joint_mechanism_sensitivity,
};

const W: VariableId = VariableId::from_raw(0);
const Z: VariableId = VariableId::from_raw(1);
const X: VariableId = VariableId::from_raw(2);
const Y: VariableId = VariableId::from_raw(3);

fn diagram() -> SelectionDiagram {
    let mut graph = Admg::with_variables(4);
    for (from, to) in [(0, 1), (1, 2), (2, 3), (0, 3)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    for (a, b) in [(0, 3), (1, 3), (1, 2)] {
        graph.insert_bidirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from([])).unwrap()
}

fn query() -> ZTransportQuery {
    ZTransportQuery {
        outcomes: Arc::from([Y]),
        treatments: Arc::from([X]),
        controllable: Arc::from([Z]),
        experiment_assignment: Arc::from([InterventionAssignment {
            variable: Z,
            value: Value::Bool(false),
        }]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    }
}

fn catalog() -> EvidenceCatalog {
    let regime = EvidenceRegime::try_new(
        RegimeId::from_raw(0),
        RegimeKind::Experimental,
        EvidenceKind::Available,
        [Z],
        [InterventionAssignment { variable: Z, value: Value::Bool(false) }],
        [W, X, Y],
        "source",
        DistributionAvailability::Joint,
    )
    .unwrap();
    EvidenceCatalog::try_new(
        [Environment::try_new(
            "source",
            [W, Z, X, Y]
                .into_iter()
                .map(|variable| VariableCoordinate {
                    variable,
                    domain: VariableDomain::Binary,
                    unit: None,
                })
                .collect::<Vec<_>>(),
            [],
        )
        .unwrap()],
        [regime],
        [RegimeBinding {
            dataset_identity: None,
            regime: RegimeId::from_raw(0),
            snapshot_identity: Arc::from("snapshot-0"),
            schema_names: Arc::from([]),
            sampling: SamplingDesign::Independent,
            weights: None,
            dependence: DependenceGroup::IndependentStudies,
        }],
        None,
    )
    .unwrap()
}

/// `P(W) P(X | W) P(Y | W, X)` under `do(Z = 0)`, W-major, last axis fastest.
fn probabilities(shift: f64) -> Vec<f64> {
    let mut out = Vec::with_capacity(8);
    for w in [false, true] {
        for x in [false, true] {
            for y in [false, true] {
                let p_w = if w { 0.35 } else { 0.65 };
                let p_x = if x { 0.4 } else { 0.6 };
                let p_y1 = match (w, x) {
                    (false, false) => 0.2,
                    (false, true) => 0.5 + shift,
                    (true, false) => 0.3,
                    (true, true) => 0.8,
                };
                out.push(p_w * p_x * if y { p_y1 } else { 1.0 - p_y1 });
            }
        }
    }
    out
}

fn data(shift: f64) -> ExactTransportData {
    let binary = |variable| DiscreteAxis {
        variable,
        values: Arc::from([Value::Bool(false), Value::Bool(true)]),
    };
    let law = ExactDiscreteLaw::try_new(
        "source",
        RegimeId::from_raw(0),
        [antecedent_expr::InterventionAssignment::concrete(Z, Value::Bool(false))],
        [binary(W), binary(X), binary(Y)],
        probabilities(shift),
        "snapshot-0",
        LawTolerance::default(),
    )
    .unwrap();
    ExactTransportData::try_new([law], 64).unwrap()
}

struct Builder {
    diagram: SelectionDiagram,
    functional: antecedent_identify::BoundZTransportFunctional,
    data: ExactTransportData,
}

fn builder() -> Builder {
    let ZTransportResult::Identified(proof) = identify_z_transport(
        &diagram(),
        &query(),
        SidLimits::default(),
        &ExecutionContext::for_tests(0),
    )
    .unwrap() else {
        panic!("the registered surrogate formula identifies");
    };
    let functional = bind_z_transport_catalog(&diagram(), &query(), &proof, &catalog()).unwrap();
    Builder { diagram: diagram(), functional, data: data(0.0) }
}

fn prepare(builder: &Builder) -> PreparedZTransport {
    StudyBuilder::z_transport(
        builder.diagram.clone(),
        builder.functional.clone(),
        builder.data.clone(),
        Assignment::from_pairs([(X, Value::Bool(true))]),
        ExactEvaluationLimits::default(),
        &ExecutionContext::for_tests(1),
    )
    .unwrap()
}

fn spec() -> JointDeviationSpec {
    let mut spec = JointDeviationSpec::new(vec![
        JointFactorBound { factor: JointFactor::SharedParentMarginal, max_fraction: 0.3 },
        JointFactorBound { factor: JointFactor::OutcomeKernel, max_fraction: 0.2 },
    ]);
    spec.decision_threshold = Some(0.5);
    spec.frontier_points = 9;
    spec
}

fn baseline_bytes(prepared: &PreparedZTransport) -> Vec<u8> {
    let ctx = ExecutionContext::for_tests(1);
    prepared.estimate(&ctx).unwrap().export(prepared).unwrap()
}

fn artifact() -> JointSensitivityArtifactWire {
    let built = builder();
    let prepared = prepare(&built);
    drop(built);
    JointSensitivityArtifactWire::checked(
        baseline_bytes(&prepared),
        &spec(),
        &ExecutionContext::for_tests(1),
    )
    .unwrap()
}

fn refused(error: &IoError) -> (&'static str, String) {
    match error {
        IoError::Refused { code, message } => {
            (code, message.split(':').next().unwrap_or_default().to_owned())
        }
        other => panic!("expected a reason-coded refusal, got {other:?}"),
    }
}

fn consume(wire: &JointSensitivityArtifactWire) -> Result<JointSensitivityArtifactWire, IoError> {
    JointSensitivityArtifactWire::consume(&wire.export().unwrap(), &ExecutionContext::for_tests(1))
}

fn reseal(wire: &mut JointSensitivityArtifactWire) {
    wire.premises_digest = wire.computed_premises_digest().unwrap();
    wire.data_digest = wire.computed_data_digest().unwrap();
}

#[test]
fn prepared_z_stage_evaluates_joint_sensitivity_without_reidentifying() {
    let built = builder();
    let direct = z_transport_joint_mechanism_sensitivity(
        &built.diagram,
        &built.functional,
        &built.data,
        &spec(),
        &ExecutionContext::for_tests(1),
    )
    .unwrap();
    let prepared = prepare(&built);
    drop(built);
    let ctx = ExecutionContext::for_tests(1);
    // The retained plan executes; the joint analysis reads the retained program's laws.
    let point = prepared.estimate(&ctx).unwrap();
    assert!(!point.distribution().probabilities.is_empty());
    let result = prepared.joint_mechanism_sensitivity(&spec(), &ctx).unwrap();
    assert_eq!(result, direct, "the prepared stage adds nothing and re-identifies nothing");
    assert!(result.range.minimum < result.baseline && result.baseline < result.range.maximum);
    assert_eq!(result.frontier.len(), 9);
    assert_eq!(result.inference_claim, "assumption_range");
    // Refreshing the laws moves the range; the checked functional is kept.
    let refreshed = prepared.refresh(data(0.1), &ctx).unwrap();
    let moved = refreshed.joint_mechanism_sensitivity(&spec(), &ctx).unwrap();
    assert!(moved.baseline > result.baseline);
    assert!(moved.range.maximum > result.range.maximum);
    assert_eq!(refreshed.functional().root(), prepared.functional().root());
    // Invalid requests refuse with their reason code through the stage.
    let mut coupled = spec();
    coupled.total_budget = Some(0.4);
    let error = prepared.joint_mechanism_sensitivity(&coupled, &ctx).unwrap_err();
    assert_eq!(
        refused(&error),
        ("route_not_supported", "joint_sensitivity.budget_coupling".into())
    );
}

#[test]
fn the_exported_joint_artifact_is_replayed_by_an_independent_consumer() {
    let wire = artifact();
    assert_eq!(wire.version, 3);
    let bytes = wire.export().unwrap();
    let consumed = JointSensitivityArtifactWire::consume_with_limits(
        &bytes,
        JointSensitivityConsumeLimits::default(),
        &ExecutionContext::for_tests(9),
    )
    .unwrap();
    assert_eq!(consumed, wire);
    // The replayed numbers are the in-memory analysis, bit for bit.
    let built = builder();
    let direct = z_transport_joint_mechanism_sensitivity(
        &built.diagram,
        &built.functional,
        &built.data,
        &spec(),
        &ExecutionContext::for_tests(1),
    )
    .unwrap();
    assert_eq!(consumed.body.outcome.range[0].to_bits(), direct.range.minimum.to_bits());
    assert_eq!(consumed.body.outcome.range[1].to_bits(), direct.range.maximum.to_bits());
    assert_eq!(consumed.body.outcome.frontier.len(), 9);
    assert_eq!(consumed.body.perturbation.factors[0].factor, "outcome_kernel");
    assert_eq!(consumed.body.sampling.status, "withheld");
    assert_eq!(consumed.body.sampling.interval, None);
    assert_eq!(consumed.body.receipt.operations_consumed, direct.receipt.operations_consumed);
    // A budget-truncated (operations) analysis exports and replays identically.
    let prepared = prepare(&built);
    let mut tight = spec();
    tight.limits.operations = 6;
    let truncated = JointSensitivityArtifactWire::checked(
        baseline_bytes(&prepared),
        &tight,
        &ExecutionContext::for_tests(1),
    )
    .unwrap();
    assert_eq!(truncated.body.receipt.stop.as_deref(), Some("search.operations"));
    assert_eq!(consume(&truncated).unwrap(), truncated);
    // A producer whose own hard memory limit is below the declared cap runs
    // under that smaller effective cap; the consumer replays under the stored
    // effective cap (not the declared one) and reproduces the receipt.
    let mut ctx = ExecutionContext::for_tests(1);
    ctx.memory = MemoryBudget { soft_limit_bytes: None, hard_limit_bytes: Some(32 << 20) };
    let memory_bound =
        JointSensitivityArtifactWire::checked(baseline_bytes(&prepared), &spec(), &ctx).unwrap();
    assert_eq!(memory_bound.body.receipt.memory_limit_bytes, 32 << 20);
    assert_eq!(memory_bound.body.perturbation.memory_bytes, 64 << 20);
    assert_eq!(consume(&memory_bound).unwrap(), memory_bound);
}

#[test]
fn a_mutated_joint_artifact_fails_consumption_for_the_right_reason() {
    let wire = artifact();
    let premises = ("transport_not_certified", "joint_sensitivity.premises_mismatch".to_owned());
    let data_identity =
        ("transport_not_certified", "joint_sensitivity.data_identity_mismatch".to_owned());
    let replay = ("transport_not_certified", "joint_sensitivity.replay_mismatch".to_owned());
    let cases: Vec<(&str, Box<dyn Fn(&mut JointSensitivityArtifactWire)>, _)> = vec![
        (
            "fraction",
            Box::new(|w| w.body.perturbation.factors[0].max_fraction = 0.25),
            premises.clone(),
        ),
        (
            "threshold",
            Box::new(|w| w.body.perturbation.decision_threshold = Some(0.6)),
            premises.clone(),
        ),
        ("limits", Box::new(|w| w.body.perturbation.operations = 50_000), premises.clone()),
        ("method", Box::new(|w| w.body.receipt.method.push('!')), premises.clone()),
        ("claim", Box::new(|w| w.body.inference_claim = "calibrated".into()), premises.clone()),
        (
            "sampling",
            Box::new(|w| w.body.sampling.method = "percentile_bootstrap".into()),
            premises.clone(),
        ),
        (
            "snapshots",
            Box::new(|w| w.body.provenance.provider_snapshots[0].push('x')),
            data_identity.clone(),
        ),
        ("regime", Box::new(|w| w.body.provenance.source_regime = 7), data_identity.clone()),
        (
            "baseline bytes",
            Box::new(|w| {
                let last = w.baseline_artifact.len() - 1;
                w.baseline_artifact[last] ^= 1;
            }),
            data_identity.clone(),
        ),
        ("range", Box::new(|w| w.body.outcome.range[1] += 1e-9), replay.clone()),
        (
            "bracket",
            Box::new(|w| {
                let bracket = w.body.outcome.frontier[2].bracket.as_mut().unwrap();
                bracket[0] = f64::from_bits(bracket[0].to_bits() + 1);
            }),
            replay.clone(),
        ),
        ("receipt", Box::new(|w| w.body.receipt.operations_consumed += 1), replay.clone()),
    ];
    for (name, mutate, expected) in &cases {
        let mut mutated = wire.clone();
        mutate(&mut mutated);
        let error = consume(&mutated).unwrap_err();
        assert_eq!(&refused(&error), expected, "unsealed {name}");
    }
    // Another format version is refused by version, not by content.
    let mut other = wire.clone();
    other.version = 2;
    assert!(matches!(consume(&other), Err(IoError::UnsupportedVersion { version: 2 })));
}

#[test]
fn resealed_semantic_mutations_fail_replay_for_the_right_reason() {
    let wire = artifact();
    let replay = ("transport_not_certified", "joint_sensitivity.replay_mismatch".to_owned());
    let cases: Vec<(&str, Box<dyn Fn(&mut JointSensitivityArtifactWire)>, (&str, String))> = vec![
        (
            "kernel fraction",
            Box::new(|w| w.body.perturbation.factors[0].max_fraction = 0.25),
            replay.clone(),
        ),
        (
            "parent fraction",
            Box::new(|w| w.body.perturbation.factors[1].max_fraction = 0.35),
            replay.clone(),
        ),
        (
            "threshold",
            Box::new(|w| w.body.perturbation.decision_threshold = Some(0.55)),
            replay.clone(),
        ),
        (
            "factor set",
            Box::new(|w| {
                w.body.perturbation.factors.pop();
            }),
            replay.clone(),
        ),
        ("tolerance", Box::new(|w| w.body.perturbation.tolerance = 1e-6), replay.clone()),
        ("frontier grid", Box::new(|w| w.body.perturbation.frontier_points = 5), replay.clone()),
        ("operations", Box::new(|w| w.body.perturbation.operations = 7), replay.clone()),
        (
            "method",
            Box::new(|w| w.body.receipt.method = "general optimizer".into()),
            replay.clone(),
        ),
        ("interpretation", Box::new(|w| w.body.interpretation = "a range".into()), replay.clone()),
        (
            "bracket",
            Box::new(|w| {
                let bracket = w.body.outcome.frontier[2].bracket.as_mut().unwrap();
                bracket[1] = f64::from_bits(bracket[1].to_bits() - 1);
            }),
            replay.clone(),
        ),
        (
            "axis analytic",
            Box::new(|w| w.body.outcome.axis[0].analytic = Some(0.123)),
            replay.clone(),
        ),
        (
            "receipt stop",
            Box::new(|w| w.body.receipt.stop = Some("search.operations".into())),
            replay.clone(),
        ),
        (
            "explored",
            Box::new(|w| {
                w.body.receipt.explored.pop();
            }),
            replay.clone(),
        ),
        (
            "sampling method",
            Box::new(|w| w.body.sampling.method = "percentile_bootstrap".into()),
            replay.clone(),
        ),
        (
            "coverage target",
            Box::new(|w| w.body.sampling.coverage_target = "two_sided".into()),
            replay.clone(),
        ),
        (
            "snapshots",
            Box::new(|w| w.body.provenance.provider_snapshots[0] = "snapshot-9".into()),
            replay.clone(),
        ),
        (
            "baseline premises",
            Box::new(|w| w.body.provenance.baseline_premises_digest.push('0')),
            replay.clone(),
        ),
        ("query binding", Box::new(|w| w.body.provenance.query_binding.push('0')), replay.clone()),
        (
            "interval",
            Box::new(|w| w.body.sampling.interval = Some([0.1, 0.9])),
            ("cell_not_licensed", "joint_sensitivity.interval_withheld".to_owned()),
        ),
        (
            "claim",
            Box::new(|w| w.body.inference_claim = "confidence_interval".into()),
            ("estimator_inference_mismatch", "joint_sensitivity.union_labelled_ci".to_owned()),
        ),
        (
            "factor name",
            Box::new(|w| w.body.perturbation.factors[0].factor = "latent_mechanism".into()),
            ("invalid_argument", "joint_sensitivity.unknown_factor".to_owned()),
        ),
        (
            "out-of-scope factor",
            Box::new(|w| w.body.perturbation.factors[0].factor = "fixed_graph_parent".into()),
            ("route_not_supported", "joint_sensitivity.fixed_graph_parent".to_owned()),
        ),
    ];
    for (name, mutate, expected) in &cases {
        let mut mutated = wire.clone();
        mutate(&mut mutated);
        reseal(&mut mutated);
        let error = consume(&mutated).unwrap_err();
        assert_eq!(&refused(&error), expected, "re-sealed {name}");
    }
    // A re-sealed baseline that is itself a different valid point artifact
    // (refreshed laws) replays to different numbers.
    let built = builder();
    let prepared = prepare(&built).refresh(data(0.1), &ExecutionContext::for_tests(1)).unwrap();
    let mut swapped = wire.clone();
    swapped.baseline_artifact = baseline_bytes(&prepared);
    swapped.body.provenance.baseline_artifact_digest = {
        let fresh = JointSensitivityArtifactWire::checked(
            swapped.baseline_artifact.clone(),
            &spec(),
            &ExecutionContext::for_tests(1),
        )
        .unwrap();
        fresh.body.provenance.baseline_artifact_digest
    };
    reseal(&mut swapped);
    assert_eq!(refused(&consume(&swapped).unwrap_err()), replay);
}

#[test]
fn stored_limits_above_the_consumer_maxima_refuse_before_any_work() {
    let wire = artifact();
    let bytes = wire.export().unwrap();
    let limited = ("route_not_supported", "joint_sensitivity.consumer_limits".to_owned());
    let consumer =
        |max_operations: usize, max_depth: usize, max_memory_bytes: u64, hard: Option<u64>| {
            let mut ctx = ExecutionContext::for_tests(1);
            ctx.memory = MemoryBudget { soft_limit_bytes: None, hard_limit_bytes: hard };
            JointSensitivityArtifactWire::consume_with_limits(
                &bytes,
                JointSensitivityConsumeLimits {
                    max_operations,
                    max_depth,
                    max_memory_bytes,
                    ..JointSensitivityConsumeLimits::default()
                },
                &ctx,
            )
        };
    let declared = &wire.body.perturbation;
    let effective = wire.body.receipt.memory_limit_bytes;
    // Exactly the stored limits replay.
    consumer(declared.operations, declared.depth, declared.memory_bytes, Some(effective)).unwrap();
    for (name, result) in [
        ("operations", consumer(declared.operations - 1, 64, declared.memory_bytes, None)),
        ("depth", consumer(100_000, declared.depth - 1, declared.memory_bytes, None)),
        ("memory cap", consumer(100_000, 64, declared.memory_bytes - 1, None)),
        ("hard memory", consumer(100_000, 64, declared.memory_bytes, Some(effective - 1))),
    ] {
        assert_eq!(refused(&result.unwrap_err()), limited, "{name}");
    }
    // Before any work: a corrupted baseline behind excessive limits is refused
    // for the limits, never for the baseline.
    let mut corrupt = wire.clone();
    corrupt.baseline_artifact.truncate(8);
    let corrupt_bytes = corrupt.export().unwrap();
    let error = JointSensitivityArtifactWire::consume_with_limits(
        &corrupt_bytes,
        JointSensitivityConsumeLimits {
            max_operations: 10,
            ..JointSensitivityConsumeLimits::default()
        },
        &ExecutionContext::for_tests(1),
    )
    .unwrap_err();
    assert_eq!(refused(&error), limited);
}

#[test]
fn a_resealed_declared_memory_cap_is_bound_by_the_premises_digest_not_by_replay() {
    let wire = artifact();
    let effective = wire.body.receipt.memory_limit_bytes;
    assert_eq!(wire.body.perturbation.memory_bytes, 64 << 20);
    assert_eq!(effective, 64 << 20, "a producer context without a hard limit");
    let limited = ("route_not_supported", "joint_sensitivity.consumer_limits".to_owned());
    let consume_under = |mutated: &JointSensitivityArtifactWire, max_memory_bytes: u64| {
        JointSensitivityArtifactWire::consume_with_limits(
            &mutated.export().unwrap(),
            JointSensitivityConsumeLimits { max_memory_bytes, ..Default::default() },
            &ExecutionContext::for_tests(1),
        )
    };
    let declared = |cap: u64, sealed: bool| {
        let mut mutated = wire.clone();
        mutated.body.perturbation.memory_bytes = cap;
        if sealed {
            reseal(&mut mutated);
        }
        mutated
    };
    // Not re-sealed: the declared cap is a premise.
    assert_eq!(
        refused(&consume_under(&declared(128 << 20, false), 128 << 20).unwrap_err()),
        ("transport_not_certified", "joint_sensitivity.premises_mismatch".to_owned())
    );
    // Re-sealed and at or above the effective cap: accepted. This is the
    // documented gap (what replay does not protect against): the numbers depend
    // only on the effective cap, and a producer with a context hard limit writes
    // effective < declared, so replay cannot re-derive the declared cap.
    let raised = declared(128 << 20, true);
    assert_eq!(consume_under(&raised, 128 << 20).unwrap(), raised);
    // Re-sealed below the effective cap: refused before any work.
    assert_eq!(
        refused(&consume_under(&declared(effective - 1, true), 128 << 20).unwrap_err()),
        limited
    );
    // Re-sealed above this consumer's maximum, or above the producer's 512 MiB
    // ceiling (which no producer can declare) under any consumer maximum:
    // refused before any work.
    assert_eq!(refused(&consume_under(&raised, 64 << 20).unwrap_err()), limited);
    assert_eq!(
        refused(&consume_under(&declared((512 << 20) + 1, true), u64::MAX).unwrap_err()),
        limited
    );
    assert_eq!(
        consume_under(&declared(512 << 20, true), u64::MAX).unwrap().body.perturbation.memory_bytes,
        512 << 20
    );
}

#[test]
fn unknown_wire_fields_are_refused() {
    use ciborium::value::Value as Cbor;
    let wire = artifact();
    let bytes = wire.export().unwrap();
    let decoded: Cbor = ciborium::de::from_reader(bytes.as_slice()).unwrap();
    let with_field = |mut value: Cbor, path: &[&str]| {
        let mut node = &mut value;
        for key in path {
            let Cbor::Map(entries) = node else { panic!("map") };
            node = &mut entries.iter_mut().find(|(k, _)| k.as_text() == Some(key)).unwrap().1;
        }
        let Cbor::Map(entries) = node else { panic!("map") };
        entries.push((Cbor::Text("extra".into()), Cbor::Bool(true)));
        let mut out = Vec::new();
        ciborium::ser::into_writer(&value, &mut out).unwrap();
        out
    };
    for path in [&[][..], &["body"][..], &["body", "sampling"][..], &["body", "receipt"][..]] {
        let tampered = with_field(decoded.clone(), path);
        let error =
            JointSensitivityArtifactWire::consume(&tampered, &ExecutionContext::for_tests(1))
                .unwrap_err();
        assert!(matches!(error, IoError::Cbor(_)), "unknown field under {path:?}: {error:?}");
    }
}

struct CancelAfterFirstLine(antecedent_core::CancellationToken);

impl ProgressSink for CancelAfterFirstLine {
    fn report(&self, _fraction: f64, stage: &str) {
        if stage == "joint sensitivity frontier" {
            self.0.cancel();
        }
    }
}

#[test]
fn the_producer_binds_premises_and_data_and_refuses_a_cancelled_frontier() {
    let built = builder();
    let prepared = prepare(&built);
    drop(built);
    let baseline = baseline_bytes(&prepared);
    let wire = JointSensitivityArtifactWire::checked(
        baseline.clone(),
        &spec(),
        &ExecutionContext::for_tests(1),
    )
    .unwrap();
    assert_eq!(wire.premises_digest, wire.computed_premises_digest().unwrap());
    assert_eq!(wire.data_digest, wire.computed_data_digest().unwrap());
    // The premises digest moves with the perturbation, not with the data; the
    // data digest moves with the baseline, not with the perturbation.
    let mut other = spec();
    other.factors[0].max_fraction = 0.31;
    let moved = JointSensitivityArtifactWire::checked(
        baseline.clone(),
        &other,
        &ExecutionContext::for_tests(1),
    )
    .unwrap();
    assert_ne!(moved.premises_digest, wire.premises_digest);
    assert_eq!(moved.data_digest, wire.data_digest);
    let refreshed = prepared.refresh(data(0.1), &ExecutionContext::for_tests(1)).unwrap();
    let rebased = JointSensitivityArtifactWire::checked(
        baseline_bytes(&refreshed),
        &spec(),
        &ExecutionContext::for_tests(1),
    )
    .unwrap();
    assert_ne!(rebased.data_digest, wire.data_digest);
    // Producer-side bounds are the evaluator's hard caps.
    let mut over = spec();
    over.frontier_points = 34;
    let error = JointSensitivityArtifactWire::checked(
        baseline.clone(),
        &over,
        &ExecutionContext::for_tests(1),
    )
    .unwrap_err();
    assert_eq!(
        refused(&error),
        ("route_not_supported", "joint_sensitivity.bounds_exceeded".into())
    );
    // A cancelled frontier cannot replay and is not exported.
    let mut ctx = ExecutionContext::for_tests(1);
    ctx.progress = Some(Arc::new(CancelAfterFirstLine(ctx.cancellation.clone())));
    let error = JointSensitivityArtifactWire::checked(baseline, &spec(), &ctx).unwrap_err();
    assert_eq!(refused(&error), ("transport_budget_cancel", "joint_sensitivity.budget".into()));
}

#[test]
fn the_unmeasured_interval_route_refuses_with_cell_not_licensed() {
    let built = builder();
    let prepared = prepare(&built);
    drop(built);
    let ctx = ExecutionContext::for_tests(1);
    let error = prepared.joint_mechanism_sensitivity_interval(&spec(), &ctx).unwrap_err();
    assert_eq!(
        refused(&error),
        ("cell_not_licensed", "joint_sensitivity.interval_withheld".into())
    );
    // Every evaluation reports the interval withheld, never attaches one.
    let result = prepared.joint_mechanism_sensitivity(&spec(), &ctx).unwrap();
    assert_eq!(result.uncertainty.reason_code, "cell_not_licensed");
    assert_eq!(result.uncertainty.detail, "joint_sensitivity.interval_withheld");
    // A consumer refuses an artifact that carries an interval.
    let mut wire = artifact();
    wire.body.sampling.interval = Some([0.0, 1.0]);
    reseal(&mut wire);
    let error = consume(&wire).unwrap_err();
    assert_eq!(
        refused(&error),
        ("cell_not_licensed", "joint_sensitivity.interval_withheld".into())
    );
}

#[test]
fn the_v2_sensitivity_artifact_is_unchanged_and_rejects_v3() {
    let built = builder();
    let prepared = prepare(&built);
    let ctx = ExecutionContext::for_tests(1);
    let baseline = baseline_bytes(&prepared);
    let v2 = ZTransportSensitivityArtifactWire::checked(baseline.clone(), 0.2, Some(0.45), &ctx)
        .unwrap();
    let v2_bytes = v2.export().unwrap();
    assert_eq!(ZTransportSensitivityArtifactWire::consume(&v2_bytes, &ctx).unwrap(), v2);
    let v3_bytes =
        JointSensitivityArtifactWire::checked(baseline, &spec(), &ctx).unwrap().export().unwrap();
    assert!(matches!(
        ZTransportSensitivityArtifactWire::consume(&v3_bytes, &ctx),
        Err(IoError::UnsupportedVersion { version: 3 })
    ));
    assert!(matches!(
        JointSensitivityArtifactWire::consume(&v2_bytes, &ctx),
        Err(IoError::UnsupportedVersion { version: 2 })
    ));
    // The v2 one-factor range equals the joint route with only the kernel factor.
    let mut kernel_only = JointDeviationSpec::new(vec![JointFactorBound {
        factor: JointFactor::OutcomeKernel,
        max_fraction: 0.2,
    }]);
    kernel_only.decision_threshold = Some(0.45);
    let joint = prepared.joint_mechanism_sensitivity(&kernel_only, &ctx).unwrap();
    assert_eq!(joint.range.minimum.to_bits(), v2.assumption_range[0].to_bits());
    assert_eq!(joint.range.maximum.to_bits(), v2.assumption_range[1].to_bits());
    assert!(v2.tipping_fraction.is_some());
    assert_eq!(
        joint.axis_tipping[0].analytic.map(f64::to_bits),
        v2.tipping_fraction.map(f64::to_bits)
    );
}

/// Cross-format: a 2.1 z-transport point artifact (version 2) is refused by the
/// 2.2 mz-transport consumer (version 1) by version, before its payload is read.
#[test]
fn a_z_transport_artifact_is_refused_by_the_mz_consumer() {
    let built = builder();
    let bytes = baseline_bytes(&prepare(&built));
    let ctx = ExecutionContext::for_tests(1);
    assert!(matches!(
        antecedent::consume_mz_transport_artifact(
            &bytes,
            antecedent_io::mz_transport_artifact::MzTransportConsumeLimits::default(),
            &ctx
        ),
        Err(IoError::UnsupportedVersion { version: 2 })
    ));
}
