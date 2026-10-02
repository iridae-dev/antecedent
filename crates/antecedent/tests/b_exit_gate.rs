//! 2.2 B exit gate: six end-to-end user stories, one test each (B1-B6).
//!
//! Every story runs start to finish in ONE test through the public facade: it
//! builds its evidence from the package's known-truth fixture, prepares once,
//! estimates against the enumerated (or closed-form) truth, exports the
//! artifact, drops every producer object, and then consumes the artifact bytes
//! alone in a fresh scope. The consumer's answers must equal the producer's bit
//! for bit. Each story then mutates its artifact in the four categories the
//! release closure names (theorem premises, source identity, data identity and
//! result body), unsealed and re-sealed, and asserts its named refusals.
//!
//! B2's interval and B3's sampling uncertainty are not measured here: their
//! routes stay closed (`cell_not_licensed`), and `scripts/gate_b_exit.sh`
//! reports the calibration state separately. No story attaches an interval.
//!
//! Registered in `scripts/b_exit_report.py` (`B_PACKAGES`).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::too_many_lines,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::type_complexity,
    reason = "end-to-end stories: one test per package, exact replays compare bits"
)]

#[path = "../../antecedent-estimate/tests/common/admg_conditional_scm.rs"]
mod admg_conditional_scm;
#[path = "../../antecedent-estimate/tests/common/mod.rs"]
mod common;
#[path = "../../antecedent-identify/tests/support/latent_scm.rs"]
mod latent_scm;
#[path = "../../antecedent-estimate/tests/support/recovery_scm.rs"]
mod recovery_scm;
#[path = "smoothed_dose_dgp/mod.rs"]
mod smoothed_dose_dgp;

use std::sync::Arc;

use antecedent::counterfactual_id::{
    CounterfactualIdConsumeLimits, CounterfactualIdOptions, consume_counterfactual_id_artifact,
    consume_counterfactual_id_artifact_for_data, prepare_counterfactual_id,
};
use antecedent::design::{
    StudyArrivalDecision, StudyCandidate, StudyCost, StudyPlanArtifactWire, StudyPlanConsumeLimits,
    StudyPlanLimits, StudyPlanResult, StudyPlanRoute, plan_studies,
};
use antecedent::{
    JointSensitivityArtifactWire, JointSensitivityConsumeLimits, StudyBuilder,
    consume_admg_conditional_transport_artifact, consume_observation_recovery_artifact,
    consume_smoothed_dose_artifact,
};
use antecedent_core::{
    CounterfactualEventQuery, DependenceGroup, DistributionAvailability, Environment,
    EvidenceCatalog, EvidenceKind, EvidenceRegime, ExecutionContext, InterventionAssignment,
    RegimeBinding, RegimeId, RegimeKind, SamplingDesign, SearchLimits, Value, VariableCoordinate,
    VariableDomain, VariableId,
};
use antecedent_expr::{
    Assignment, DiscreteAxis, ExactDiscreteLaw, ExactEvaluationLimits, ExactTransportData,
    InterventionAssignment as ExprInterventionAssignment, LawTolerance,
};
use antecedent_graph::{Admg, DenseNodeId, NodeRef, SelectionDiagram};
use antecedent_identify::{
    ADMG_CONDITIONAL_DEFAULT_LIMITS, ConditionalTransportDecision, MZ_TRANSPORT_DEFAULT_LIMITS,
    MzTransportDecision, RecoveredEffectQuery, RecoveryDetail, RecoveryLimits, SidLimits,
    ZTransportQuery, ZTransportResult, bind_mz_transport_catalog, bind_z_transport_catalog,
    decide_admg_conditional_transport, decide_mz_transport, identify_z_transport,
};
use antecedent_io::IoError;
use antecedent_io::admg_conditional_transport_artifact::{
    AdmgConditionalArtifactError, AdmgConditionalArtifactWire, AdmgConditionalConsumeLimits,
};
use antecedent_io::counterfactual_id_artifact::{
    CounterfactualIdArtifactError, CounterfactualIdArtifactWire,
};
use antecedent_io::recovery_artifact::{
    RecoveryArtifactError, RecoveryArtifactWire, RecoveryConsumeLimits,
};
use antecedent_io::smoothed_dose_artifact::{
    SmoothedDoseArtifactError, SmoothedDoseArtifactWire, SmoothedDoseConsumeLimits,
};
use antecedent_validate::{JointDeviationSpec, JointFactor, JointFactorBound};

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn coded(error: &IoError) -> (&'static str, String) {
    match error {
        IoError::Refused { code, message } => {
            (code, message.split(':').next().unwrap_or_default().to_owned())
        }
        other => panic!("expected a reason-coded refusal, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// B1 (X2): ADMG conditional transport.
// ---------------------------------------------------------------------------

/// `X(0) -> Y(1) -> W(2)`, `X <-> Y`, the target's `W` mechanism selected:
/// `P*(y | do(x), w)` is the reduced joint normalized at `w`.
fn b1_model(params: usize, selected: usize) -> admg_conditional_scm::Scm {
    admg_conditional_scm::Scm {
        n: 3,
        directed: vec![(0, 1), (1, 2)],
        bidirected: vec![(0, 1)],
        selected: vec![selected],
        params,
        target_zero: Vec::new(),
    }
}

const B1_LEVELS: [usize; 4] = [0b000, 0b001, 0b100, 0b101];

fn b1_refusal(bytes: &[u8]) -> AdmgConditionalArtifactError {
    match AdmgConditionalArtifactWire::consume_with_limits(
        bytes,
        AdmgConditionalConsumeLimits::default(),
        &ExecutionContext::for_tests(4),
    ) {
        Err(IoError::AdmgConditional(error)) => error,
        Err(other) => panic!("typed artifact refusal expected: {other:?}"),
        Ok(_) => panic!("a mutated artifact must not be consumed"),
    }
}

fn b1_reseal(wire: &mut AdmgConditionalArtifactWire) -> Vec<u8> {
    wire.premises_digest = wire.expected_premises_digest().unwrap();
    wire.data_digest = wire.expected_data_digest().unwrap();
    wire.export().unwrap()
}

#[test]
fn story_b1_admg_conditional_transport_against_enumerated_truth_with_typed_refusals() {
    use admg_conditional_scm::{query, request};
    let names = ["X".to_owned(), "Y".to_owned(), "W".to_owned()];
    let ctx = ExecutionContext::for_tests(5);

    // Prepare: decide once on the latent-confounded selection ADMG, then bind.
    let (bytes, produced) = {
        let model = b1_model(1, 2);
        let (catalog, data) = model.catalog_and_laws();
        let ConditionalTransportDecision::Identified(bound) = decide_admg_conditional_transport(
            &model.diagram(),
            &query(&[1], &[0], &[2]),
            &catalog,
            ADMG_CONDITIONAL_DEFAULT_LIMITS,
            &ctx,
        )
        .unwrap() else {
            panic!("the movable-free conditional query identifies");
        };
        let prepared = StudyBuilder::admg_conditional_transport(
            model.diagram(),
            *bound,
            ADMG_CONDITIONAL_DEFAULT_LIMITS,
            data,
            B1_LEVELS.iter().map(|level| request(&[0], &[2], *level)).collect(),
            ExactEvaluationLimits::default(),
            &ctx,
        )
        .unwrap();
        // Execute: every (X, W) level equals the enumerated target truth.
        let result = prepared.estimate(&ctx).unwrap();
        for (distribution, level) in result.distributions().iter().zip(B1_LEVELS) {
            let truth = model.truth(&[1], &[0], &[2], level).unwrap();
            for (p, t) in distribution.probabilities.iter().zip(&truth) {
                assert!((p - t).abs() < 1e-12, "level {level}: {p} vs {t}");
            }
        }
        // Conditioning is not a no-op on this fixture.
        let at = |i: usize| result.distributions()[i].probabilities[1];
        assert!((at(0) - at(2)).abs() > 1e-3);
        // Counted laws are refused: the route publishes exact points only.
        let mut laws = model.catalog_and_laws().1.laws().to_vec();
        let first = &laws[0];
        let cells = first.probabilities().len();
        laws[0] = ExactDiscreteLaw::try_new(
            first.population(),
            first.regime(),
            first.interventions().to_vec(),
            first.axes().to_vec(),
            vec![1.0 / cells as f64; cells],
            first.snapshot_identity(),
            LawTolerance::default(),
        )
        .unwrap()
        .with_empirical_counts(vec![1; cells])
        .unwrap();
        let counted = ExactTransportData::try_new(laws, 1_000_000).unwrap();
        assert_eq!(
            coded(&prepared.refresh(counted, &ctx).unwrap_err()),
            ("cell_not_licensed", "admg_transport.interval_withheld".into())
        );
        let produced: Vec<Vec<u64>> =
            result.distributions().iter().map(|d| bits(&d.probabilities)).collect();
        (result.export_named(&prepared, &names).unwrap(), produced)
    };

    // Independent consumption from the bytes alone.
    let consumed = consume_admg_conditional_transport_artifact(
        &bytes,
        AdmgConditionalConsumeLimits::default(),
        &ExecutionContext::for_tests(9),
    )
    .unwrap();
    let replayed: Vec<Vec<u64>> =
        consumed.distributions().iter().map(|d| bits(&d.probabilities)).collect();
    assert_eq!(replayed, produced);
    let truth = b1_model(1, 2);
    for (distribution, level) in consumed.distributions().iter().zip(B1_LEVELS) {
        let t = truth.truth(&[1], &[0], &[2], level).unwrap();
        assert!(distribution.probabilities.iter().zip(&t).all(|(p, t)| (p - t).abs() < 1e-12));
    }

    // Mutations: unsealed (digests) and re-sealed (replay), four categories.
    let original = AdmgConditionalArtifactWire::decode(&bytes).unwrap();
    let mut w = original.clone();
    w.query.conditioned_on = vec![0];
    assert_eq!(b1_refusal(&w.export().unwrap()), AdmgConditionalArtifactError::PremisesMismatch);
    let mut w = original.clone();
    w.proof.moves = vec![2];
    w.proof.remaining.clear();
    assert_eq!(
        b1_refusal(&b1_reseal(&mut w)).refusal(),
        ("transport_not_certified", "admg_transport.invalid_derivation"),
        "re-sealed premises"
    );
    let mut w = original.clone();
    w.laws[0].population = "elsewhere".into();
    assert!(b1_refusal(&w.export().unwrap()).refusal().0 == "transport_not_certified");
    let mut w = original.clone();
    w.bindings[0].0 = "elsewhere".into();
    assert_eq!(
        b1_refusal(&b1_reseal(&mut w)),
        AdmgConditionalArtifactError::ReplayMismatch("the leaf bindings"),
        "re-sealed source identity"
    );
    let mut w = original.clone();
    w.laws[0].snapshot = "renamed".into();
    assert_eq!(
        b1_refusal(&w.export().unwrap()),
        AdmgConditionalArtifactError::DataIdentityMismatch
    );
    let mut w = original.clone();
    let target = w.laws.iter().position(|law| law.population == "target").unwrap();
    w.laws[target].probabilities.reverse();
    assert_eq!(
        b1_refusal(&b1_reseal(&mut w)),
        AdmgConditionalArtifactError::ReplayMismatch("a point"),
        "re-sealed data identity"
    );
    let mut w = original.clone();
    w.results[0].probabilities[0] += 1e-9;
    assert_eq!(
        b1_refusal(&w.export().unwrap()),
        AdmgConditionalArtifactError::ReplayMismatch("a point")
    );
    let mut w = original.clone();
    w.results[0].probabilities[0] += 1e-9;
    assert_eq!(
        b1_refusal(&b1_reseal(&mut w)),
        AdmgConditionalArtifactError::ReplayMismatch("a point"),
        "re-sealed result body"
    );
    let mut w = original.clone();
    w.uncertainty = "percentile_bootstrap".into();
    assert_eq!(b1_refusal(&b1_reseal(&mut w)), AdmgConditionalArtifactError::IntervalWithheld);

    // A non-transportable diagram (selection on Y): the decision never
    // identifies (not certified, or proven non-transportable when an exact
    // witness is verified), and a functional decided on the transportable
    // diagram does not certify on it: `transport_not_certified`.
    let blocked = b1_model(1, 1);
    let (catalog, _) = blocked.catalog_and_laws();
    let decision = decide_admg_conditional_transport(
        &blocked.diagram(),
        &query(&[1], &[0], &[2]),
        &catalog,
        ADMG_CONDITIONAL_DEFAULT_LIMITS,
        &ctx,
    )
    .unwrap();
    assert!(!matches!(decision, ConditionalTransportDecision::Identified(_)));
    assert!(
        matches!(
            decision.reason_code(),
            Some("transport_not_certified" | "transport_proven_non_transportable")
        ),
        "{:?}",
        decision.reason_code()
    );
    let model = b1_model(1, 2);
    let (catalog, data) = model.catalog_and_laws();
    let ConditionalTransportDecision::Identified(bound) = decide_admg_conditional_transport(
        &model.diagram(),
        &query(&[1], &[0], &[2]),
        &catalog,
        ADMG_CONDITIONAL_DEFAULT_LIMITS,
        &ctx,
    )
    .unwrap() else {
        panic!("identified on the transportable diagram");
    };
    let error = StudyBuilder::admg_conditional_transport(
        blocked.diagram(),
        *bound,
        ADMG_CONDITIONAL_DEFAULT_LIMITS,
        data,
        vec![request(&[0], &[2], 0)],
        ExactEvaluationLimits::default(),
        &ctx,
    )
    .unwrap_err();
    assert_eq!(
        coded(&error),
        ("transport_not_certified", "admg_transport.invalid_derivation".into())
    );
}

// ---------------------------------------------------------------------------
// B2 (X4): smoothed dose-response transport grid.
// ---------------------------------------------------------------------------

fn b2_refused(bytes: &[u8]) -> SmoothedDoseArtifactError {
    match consume_smoothed_dose_artifact(
        bytes,
        SmoothedDoseConsumeLimits::default(),
        &ExecutionContext::for_tests(0),
    ) {
        Err(IoError::SmoothedDose(kind)) => kind,
        Err(other) => panic!("typed smoothed-dose refusal expected: {other:?}"),
        Ok(_) => panic!("a mutated artifact must not be consumed"),
    }
}

fn b2_mutate(
    wire: &SmoothedDoseArtifactWire,
    edit: impl Fn(&mut SmoothedDoseArtifactWire),
    reseal: bool,
) -> Vec<u8> {
    let mut wire = wire.clone();
    edit(&mut wire);
    if reseal {
        wire.premises_digest = wire.expected_premises_digest().unwrap();
        wire.data_digest = wire.expected_data_digest().unwrap();
        wire.evidence_digest = wire.expected_evidence_digest().unwrap();
    }
    wire.export().unwrap()
}

#[test]
fn story_b2_smoothed_dose_grid_against_closed_form_truth_with_closed_interval() {
    use smoothed_dose_dgp::{Design, Scenario, diagram, draw, query};
    const GRID: [f64; 3] = [1.0, 2.0, 3.0];
    const H: f64 = 0.5;
    let options = antecedent_estimate::SmoothedDoseOptions {
        folds: 3,
        ..antecedent_estimate::SmoothedDoseOptions::default()
    };
    let ctx = ExecutionContext::for_tests(21);

    let (bytes, produced) = {
        let (graph, names) = diagram();
        let prepared = StudyBuilder::smoothed_dose_transport(
            graph,
            query(&GRID, H),
            draw(Design::IndependentSamples, Scenario::Good, 4000, 3000, 21),
            options.clone(),
            names,
            &ctx,
        )
        .unwrap();
        let plan = prepared.estimate(&ctx).unwrap();
        // psi_h at every grid dose, against the closed form.
        for point in &plan.estimate().grid {
            let truth = Scenario::Good.truth(point.dose, H);
            assert!((point.estimate - truth).abs() < 0.2, "a={}: {}", point.dose, point.estimate);
        }
        assert_eq!(plan.estimate().uncertainty.status, "point_only");
        // The interval route is closed until it is measured.
        assert_eq!(
            coded(&prepared.interval(&ctx).unwrap_err()),
            ("cell_not_licensed", "dose_response.interval_withheld".into())
        );
        let produced = plan.estimate().clone();
        (plan.export().unwrap(), produced)
    };

    let consumed = consume_smoothed_dose_artifact(
        &bytes,
        SmoothedDoseConsumeLimits::default(),
        &ExecutionContext::for_tests(0),
    )
    .unwrap();
    assert_eq!(consumed.estimate(), &produced);
    assert_eq!(consumed.export().unwrap(), bytes, "consumption re-exports the same bytes");

    let original = SmoothedDoseArtifactWire::decode(&bytes).unwrap();
    // Premises.
    assert_eq!(
        b2_refused(&b2_mutate(&original, |w| w.query.bandwidth = 0.45, false)),
        SmoothedDoseArtifactError::PremisesMismatch
    );
    assert_eq!(
        b2_refused(&b2_mutate(&original, |w| w.query.bandwidth = 0.45, true)),
        SmoothedDoseArtifactError::PointMismatch
    );
    // Source identity: the source population key is a premise.
    assert_eq!(
        b2_refused(&b2_mutate(&original, |w| w.query.source_population = "other".into(), false)),
        SmoothedDoseArtifactError::PremisesMismatch
    );
    // Re-sealed, the relabelled population is a different premise set: the rows
    // carry no population name, so the replay itself cannot contradict the label
    // (as for any label-only premise); the consumed identity names the forgery
    // and a consumer naming the populations it expects refuses it.
    let relabelled = b2_mutate(&original, |w| w.query.source_population = "other".into(), true);
    let forged = consume_smoothed_dose_artifact(
        &relabelled,
        SmoothedDoseConsumeLimits::default(),
        &ExecutionContext::for_tests(0),
    )
    .unwrap();
    assert_ne!(forged.identity(), consumed.identity(), "re-sealed source identity");
    let (source, target) =
        (original.query.source_population.clone(), original.query.target_population.clone());
    original.check_population_names(&source, &target).unwrap();
    assert_eq!(
        SmoothedDoseArtifactWire::decode(&relabelled)
            .unwrap()
            .check_population_names(&source, &target),
        Err(SmoothedDoseArtifactError::NamesMismatch),
        "a consumer naming its populations refuses the relabel"
    );
    // Data identity.
    let row = (0..original.input.source.len()).find(|i| original.input.source[*i]).unwrap();
    assert_eq!(
        b2_refused(&b2_mutate(&original, |w| w.input.outcome[row] += 1.0, false)),
        SmoothedDoseArtifactError::DataIdentityMismatch
    );
    assert!(matches!(
        b2_refused(&b2_mutate(&original, |w| w.input.outcome[row] += 1.0, true)),
        SmoothedDoseArtifactError::PointMismatch
    ));
    // Result body.
    assert_eq!(
        b2_refused(&b2_mutate(&original, |w| w.result.grid[0].estimate += 1e-9, false)),
        SmoothedDoseArtifactError::EvidenceMismatch
    );
    assert_eq!(
        b2_refused(&b2_mutate(&original, |w| w.result.grid[0].estimate += 1e-9, true)),
        SmoothedDoseArtifactError::PointMismatch
    );
}

// ---------------------------------------------------------------------------
// B3 (X3): joint mechanism deviations on a prepared z-transport baseline.
// ---------------------------------------------------------------------------

const B3_W: VariableId = VariableId::from_raw(0);
const B3_Z: VariableId = VariableId::from_raw(1);
const B3_X: VariableId = VariableId::from_raw(2);
const B3_Y: VariableId = VariableId::from_raw(3);

fn b3_prepared() -> antecedent::PreparedZTransport {
    let mut graph = Admg::with_variables(4);
    for (from, to) in [(0, 1), (1, 2), (2, 3), (0, 3)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    for (a, b) in [(0, 3), (1, 3), (1, 2)] {
        graph.insert_bidirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    let diagram = SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from([])).unwrap();
    let query = ZTransportQuery {
        outcomes: Arc::from([B3_Y]),
        treatments: Arc::from([B3_X]),
        controllable: Arc::from([B3_Z]),
        experiment_assignment: Arc::from([InterventionAssignment {
            variable: B3_Z,
            value: Value::Bool(false),
        }]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    };
    let regime = EvidenceRegime::try_new(
        RegimeId::from_raw(0),
        RegimeKind::Experimental,
        EvidenceKind::Available,
        [B3_Z],
        [InterventionAssignment { variable: B3_Z, value: Value::Bool(false) }],
        [B3_W, B3_X, B3_Y],
        "source",
        DistributionAvailability::Joint,
    )
    .unwrap();
    let catalog = EvidenceCatalog::try_new(
        [Environment::try_new(
            "source",
            [B3_W, B3_Z, B3_X, B3_Y]
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
    .unwrap();
    let ctx = ExecutionContext::for_tests(0);
    let ZTransportResult::Identified(proof) =
        identify_z_transport(&diagram, &query, SidLimits::default(), &ctx).unwrap()
    else {
        panic!("the surrogate formula identifies");
    };
    let functional = bind_z_transport_catalog(&diagram, &query, &proof, &catalog).unwrap();
    // P(W) P(X | W) P(Y | W, X) under do(Z = 0), W-major, last axis fastest.
    let mut probabilities = Vec::with_capacity(8);
    for w in [false, true] {
        for x in [false, true] {
            for y in [false, true] {
                let p_w = if w { 0.35 } else { 0.65 };
                let p_x = if x { 0.4 } else { 0.6 };
                let p_y1 = match (w, x) {
                    (false, false) => 0.2,
                    (false, true) => 0.5,
                    (true, false) => 0.3,
                    (true, true) => 0.8,
                };
                probabilities.push(p_w * p_x * if y { p_y1 } else { 1.0 - p_y1 });
            }
        }
    }
    let binary = |variable| DiscreteAxis {
        variable,
        values: Arc::from([Value::Bool(false), Value::Bool(true)]),
    };
    let law = ExactDiscreteLaw::try_new(
        "source",
        RegimeId::from_raw(0),
        [ExprInterventionAssignment::concrete(B3_Z, Value::Bool(false))],
        [binary(B3_W), binary(B3_X), binary(B3_Y)],
        probabilities,
        "snapshot-0",
        LawTolerance::default(),
    )
    .unwrap();
    StudyBuilder::z_transport(
        diagram,
        functional,
        ExactTransportData::try_new([law], 64).unwrap(),
        Assignment::from_pairs([(B3_X, Value::Bool(true))]),
        ExactEvaluationLimits::default(),
        &ctx,
    )
    .unwrap()
}

fn b3_spec() -> JointDeviationSpec {
    let mut spec = JointDeviationSpec::new(vec![
        JointFactorBound { factor: JointFactor::SharedParentMarginal, max_fraction: 0.3 },
        JointFactorBound { factor: JointFactor::OutcomeKernel, max_fraction: 0.2 },
    ]);
    spec.decision_threshold = Some(0.5);
    spec.frontier_points = 9;
    spec
}

fn b3_consume(
    wire: &JointSensitivityArtifactWire,
) -> Result<JointSensitivityArtifactWire, IoError> {
    JointSensitivityArtifactWire::consume(&wire.export().unwrap(), &ExecutionContext::for_tests(1))
}

fn b3_reseal(mut wire: JointSensitivityArtifactWire) -> JointSensitivityArtifactWire {
    wire.premises_digest = wire.computed_premises_digest().unwrap();
    wire.data_digest = wire.computed_data_digest().unwrap();
    wire
}

#[test]
fn story_b3_joint_two_factor_deviation_with_tipping_frontier_is_an_assumption_range() {
    let ctx = ExecutionContext::for_tests(1);
    let (bytes, range) = {
        let prepared = b3_prepared();
        let baseline = prepared.estimate(&ctx).unwrap();
        let result = prepared.joint_mechanism_sensitivity(&b3_spec(), &ctx).unwrap();
        // The z baseline equals the closed form of the declared law:
        // P(Y = 1 | do(X = 1)) = 0.65 * 0.5 + 0.35 * 0.8 and the contrast
        // against do(X = 0) (0.65 * 0.2 + 0.35 * 0.3) is the analysed baseline.
        let p = baseline.distribution();
        let y1 = p.atoms.iter().position(|a| a[0] == Value::Bool(true)).unwrap();
        assert!((p.probabilities[y1] - 0.605).abs() < 1e-12, "{p:?}");
        assert!((result.baseline - (0.605 - 0.235)).abs() < 1e-12, "{}", result.baseline);
        // The joint range strictly contains the baseline.
        assert!(result.range.minimum < result.baseline && result.baseline < result.range.maximum);
        assert_eq!(result.frontier.len(), 9);
        assert_eq!(result.inference_claim, "assumption_range");
        let wire = JointSensitivityArtifactWire::checked(
            baseline.export(&prepared).unwrap(),
            &b3_spec(),
            &ctx,
        )
        .unwrap();
        (wire.export().unwrap(), [result.range.minimum, result.range.maximum])
    };

    let consumed = JointSensitivityArtifactWire::consume_with_limits(
        &bytes,
        JointSensitivityConsumeLimits::default(),
        &ExecutionContext::for_tests(9),
    )
    .unwrap();
    assert_eq!(bits(&consumed.body.outcome.range), bits(&range));
    assert_eq!(consumed.body.outcome.frontier.len(), 9);
    // The union is never a confidence interval: the sampling interval is withheld.
    assert_eq!(consumed.body.inference_claim, "assumption_range");
    assert_eq!(consumed.body.sampling.status, "withheld");
    assert_eq!(consumed.body.sampling.interval, None);

    let original = consumed;
    let refused = |wire: &JointSensitivityArtifactWire| coded(&b3_consume(wire).unwrap_err());
    let premises = ("transport_not_certified", "joint_sensitivity.premises_mismatch".to_owned());
    let data = ("transport_not_certified", "joint_sensitivity.data_identity_mismatch".to_owned());
    let replay = ("transport_not_certified", "joint_sensitivity.replay_mismatch".to_owned());
    // Premises: the threshold and the factor set.
    let mut w = original.clone();
    w.body.perturbation.decision_threshold = Some(0.6);
    assert_eq!(refused(&w), premises);
    assert_eq!(refused(&b3_reseal(w)), replay, "re-sealed threshold");
    let mut w = original.clone();
    w.body.perturbation.factors.pop();
    assert_eq!(refused(&b3_reseal(w)), replay, "re-sealed factor set");
    // Source identity: the source regime.
    let mut w = original.clone();
    w.body.provenance.source_regime = 7;
    assert_eq!(refused(&w), data);
    assert_eq!(refused(&b3_reseal(w)), replay, "re-sealed source regime");
    // Data identity: the provider snapshot.
    let mut w = original.clone();
    w.body.provenance.provider_snapshots[0].push('x');
    assert_eq!(refused(&w), data);
    let mut w = original.clone();
    w.body.provenance.provider_snapshots[0] = "snapshot-9".into();
    assert_eq!(refused(&b3_reseal(w)), replay, "re-sealed snapshot");
    // Result body.
    let mut w = original.clone();
    w.body.outcome.range[1] += 1e-9;
    assert_eq!(refused(&w), replay);
    let mut w = original.clone();
    w.body.outcome.range[1] += 1e-9;
    assert_eq!(refused(&b3_reseal(w)), replay, "re-sealed range");
    // The union relabelled as a CI or given an interval is refused by name.
    let mut w = original.clone();
    w.body.inference_claim = "confidence_interval".into();
    assert_eq!(
        refused(&b3_reseal(w)),
        ("estimator_inference_mismatch", "joint_sensitivity.union_labelled_ci".to_owned())
    );
    let mut w = original.clone();
    w.body.sampling.interval = Some([0.1, 0.9]);
    assert_eq!(
        refused(&b3_reseal(w)),
        ("cell_not_licensed", "joint_sensitivity.interval_withheld".to_owned())
    );
}

// ---------------------------------------------------------------------------
// B4 (X6): planning additions for a failed X1 decision.
// ---------------------------------------------------------------------------

fn b4_study(
    id: &str,
    population: &str,
    on: &[usize],
    levels: Option<Vec<Vec<(usize, bool)>>>,
    measured: &[usize],
    cost: u64,
) -> StudyCandidate {
    use common::z_scm::vid;
    StudyCandidate {
        id: Arc::from(id),
        population: Arc::from(population),
        interventions: on.iter().map(|v| vid(*v)).collect(),
        levels: levels.map(|levels| {
            levels
                .into_iter()
                .map(|level| {
                    level
                        .into_iter()
                        .map(|(v, l)| InterventionAssignment {
                            variable: vid(v),
                            value: Value::Bool(l),
                        })
                        .collect()
                })
                .collect()
        }),
        measured: measured.iter().map(|v| vid(*v)).collect(),
        recruitment: Arc::from("independent recruitment, one arm per level"),
        cost: StudyCost { units: cost, sample_budget: 500 },
        requires: Arc::from([]),
        conflicts: Arc::from([]),
        feasibility_constraints: Arc::from([Arc::from("ethics approval in hand")]),
    }
}

fn b4_base() -> (EvidenceCatalog, ExactTransportData) {
    let (catalog, data) = common::mz_fixture::evidence();
    let base = EvidenceCatalog::try_new(
        Arc::clone(&catalog.environments),
        catalog.regimes.iter().filter(|r| r.id.raw() == 0).cloned().collect::<Vec<_>>(),
        catalog.bindings.iter().filter(|b| b.regime.raw() == 0).cloned().collect::<Vec<_>>(),
        None,
    )
    .unwrap();
    let data = ExactTransportData::try_new(
        data.laws().iter().filter(|l| l.regime().raw() == 0).cloned().collect::<Vec<_>>(),
        4096,
    )
    .unwrap();
    (base, data)
}

/// The base plus every proposed regime delivered as available evidence under
/// `snapshot`, each law enumerated from its population's structural model.
fn b4_arrive(
    base: &EvidenceCatalog,
    base_data: &ExactTransportData,
    regimes: &[EvidenceRegime],
    snapshot: &str,
) -> (EvidenceCatalog, ExactTransportData) {
    use common::mz_fixture::{source_a_scm, source_b_scm, target_scm};
    use common::z_scm::vid;
    let mut all = base.regimes.to_vec();
    let mut bindings = base.bindings.to_vec();
    let mut laws = base_data.laws().to_vec();
    for proposed in regimes {
        let mut regime = proposed.clone();
        regime.evidence_kind = EvidenceKind::Available;
        bindings.push(RegimeBinding {
            dataset_identity: None,
            regime: regime.id,
            snapshot_identity: Arc::from(snapshot),
            schema_names: Arc::from([]),
            sampling: SamplingDesign::Independent,
            weights: None,
            dependence: DependenceGroup::IndependentStudies,
        });
        let measured = regime.measured.iter().map(|v| v.as_usize()).collect::<Vec<_>>();
        let axes = measured
            .iter()
            .map(|i| DiscreteAxis {
                variable: vid(*i),
                values: Arc::from([Value::Bool(false), Value::Bool(true)]),
            })
            .collect::<Vec<_>>();
        let level: Vec<(usize, u8)> = regime
            .intervention_values
            .iter()
            .map(|a| (a.variable.as_usize(), u8::from(a.value == Value::Bool(true))))
            .collect();
        let model = match regime.population.as_ref() {
            "a" => source_a_scm(),
            "b" => source_b_scm(),
            _ => target_scm(),
        };
        laws.push(
            ExactDiscreteLaw::try_new(
                regime.population.as_ref(),
                regime.id,
                level
                    .iter()
                    .map(|(v, l)| {
                        ExprInterventionAssignment::concrete(vid(*v), Value::Bool(*l == 1))
                    })
                    .collect::<Vec<_>>(),
                axes,
                model.law(&level, &measured),
                snapshot,
                LawTolerance::default(),
            )
            .unwrap(),
        );
        all.push(regime);
    }
    (
        EvidenceCatalog::try_new(Arc::clone(&base.environments), all, bindings, None).unwrap(),
        ExactTransportData::try_new(laws, 4096).unwrap(),
    )
}

#[test]
fn story_b4_a_failed_x1_decision_is_planned_replayed_and_flipped_by_arrival() {
    use common::mz_fixture::{X, Y, Z1, Z2, graph, query, sources, target_scm};
    use common::z_scm::{risk_of, vid};
    let ctx = ExecutionContext::for_tests(3);
    let (base, base_data) = b4_base();
    let route = StudyPlanRoute::Mz(query(sources()));

    // The X1 decision on the target law alone fails for missing evidence.
    let failed =
        decide_mz_transport(&graph(), &query(sources()), &base, MZ_TRANSPORT_DEFAULT_LIMITS, &ctx)
            .unwrap();
    assert_eq!(failed.reason_code(), Some("transport_missing_evidence"));

    let candidates = vec![
        b4_study(
            "a_do_z2",
            "a",
            &[Z2],
            Some(vec![vec![(Z2, false)], vec![(Z2, true)]]),
            &[Z1, X, Y],
            3,
        ),
        b4_study("b_do_z1", "b", &[Z1], Some(vec![vec![(Z1, false)]]), &[X, Z2, Y], 2),
        b4_study("b_observe", "b", &[], None, &[Z1, X, Z2, Y], 1),
    ];
    let bytes = {
        let plan =
            plan_studies(&graph(), &route, &base, &candidates, StudyPlanLimits::default(), &ctx)
                .unwrap();
        assert_eq!(plan.plan().failure.detail, "mz_transport.missing_joint_regime");
        let top = plan.proposal(0).expect("the complementary pair is sufficient");
        assert_eq!(top.proposal().candidates, [Arc::from("a_do_z2"), Arc::from("b_do_z1")]);
        assert!(plan.plan().minimal);
        // Every unsuccessful subset is kept in the plan, not filtered out.
        let statuses: Vec<&str> = plan.plan().subsets.iter().map(|s| s.outcome.status()).collect();
        assert_eq!(statuses.iter().filter(|s| **s == "sufficient").count(), 1);
        assert!(statuses.contains(&"insufficient") && statuses.contains(&"dominated"));
        serde_json::to_vec(&plan.to_artifact().unwrap()).unwrap()
    };

    // Independent replay of the exported plan.
    let wire: StudyPlanArtifactWire = serde_json::from_slice(&bytes).unwrap();
    let consume = |wire: &StudyPlanArtifactWire| {
        wire.consume_with_limits(StudyPlanConsumeLimits::default(), &ExecutionContext::for_tests(9))
    };
    let consumed: StudyPlanResult = consume(&wire).unwrap();
    assert_eq!(serde_json::to_vec(&consumed.to_artifact().unwrap()).unwrap(), bytes);

    // Arrival of the proposed studies flips the decision; the point equals truth.
    let top = consumed.proposal(0).unwrap();
    let (actual, data) = b4_arrive(&base, &base_data, &top.delta().proposed_regimes, "arrival-1");
    let arrival = top.receive(&actual, "arrival-1", &ctx).unwrap();
    let StudyArrivalDecision::Mz(decision) = arrival.decision else { panic!("mz route") };
    let MzTransportDecision::Identified { derivation, .. } = *decision else {
        panic!("the arriving studies identify");
    };
    let bound = bind_mz_transport_catalog(&graph(), &derivation, &actual).unwrap();
    for x in [false, true] {
        let point = risk_of(
            &antecedent_estimate::evaluate_exact_mz_transport(
                &bound,
                data.clone(),
                Assignment::from_pairs([(vid(X), Value::Bool(x))]),
                ExactEvaluationLimits::default(),
                &ctx,
            )
            .unwrap(),
        );
        let truth = target_scm().risk(&[(X, u8::from(x))], Y);
        assert!((point - truth).abs() < 1e-12, "do(X={x}): {point} vs {truth}");
    }

    // A catalog over the bounds is refused as bounds_exceeded, never "sufficient".
    let many: Vec<_> =
        (0..17).map(|i| b4_study(&format!("obs{i:02}"), "b", &[], None, &[X], 1)).collect();
    let error =
        plan_studies(&graph(), &route, &base, &many, StudyPlanLimits::default(), &ctx).unwrap_err();
    assert_eq!((error.code, error.detail), ("route_not_supported", "study_plan.bounds_exceeded"));
    // A budget that stops the plan before the pair is a receipt, never a verdict.
    let short = StudyPlanLimits {
        search: SearchLimits { operations: 1, depth: 24 },
        ..StudyPlanLimits::default()
    };
    let error = plan_studies(&graph(), &route, &base, &candidates, short, &ctx).unwrap_err();
    assert_eq!((error.code, error.detail), ("transport_budget_cancel", "study_plan.budget"));

    // Mutations: unsealed and re-sealed, four categories.
    let invalid = ("transport_not_certified", "study_plan.invalid_artifact");
    let refused = |w: &StudyPlanArtifactWire| {
        let error = consume(w).unwrap_err();
        (error.code, error.detail)
    };
    let edits: [(&str, Box<dyn Fn(&mut StudyPlanArtifactWire)>); 4] = [
        ("premises", Box::new(|w| w.premises.candidates[0].cost_units = 1)),
        (
            "source identity",
            Box::new(|w| {
                w.data.catalog.bindings[0].snapshot_identity = "hypothetical:0".into();
            }),
        ),
        ("data identity", Box::new(|w| w.data.catalog.bindings.clear())),
        ("result body", Box::new(|w| w.plan.proposals[0].cost_units = 4)),
    ];
    for (label, edit) in &edits {
        let mut w = wire.clone();
        edit(&mut w);
        assert_eq!(refused(&w), invalid, "unsealed {label}");
        let sealed = w.sealed().unwrap();
        assert_ne!(sealed.plan_digest, wire.plan_digest);
        assert_eq!(refused(&sealed), invalid, "re-sealed {label}");
    }
}

// ---------------------------------------------------------------------------
// B5 (X8): bounded ADMG counterfactual identification (ETT).
// ---------------------------------------------------------------------------

fn b5_admg(n: u32, directed: &[(u32, u32)], bidirected: &[(u32, u32)]) -> Admg {
    let mut g = Admg::with_variables(n);
    for &(a, b) in directed {
        g.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    for &(a, b) in bidirected {
        g.insert_bidirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    g
}

fn b5_law(cards: &[usize], probabilities: Vec<f64>, snapshot: &str) -> ExactDiscreteLaw {
    let axes = cards
        .iter()
        .enumerate()
        .map(|(i, &c)| DiscreteAxis {
            variable: VariableId::from_raw(u32::try_from(i).unwrap()),
            values: (0..c).map(|l| Value::f64(l as f64)).collect::<Vec<_>>().into(),
        })
        .collect::<Vec<_>>();
    ExactDiscreteLaw::try_new(
        "target",
        RegimeId::from_raw(0),
        [],
        axes,
        probabilities,
        snapshot,
        LawTolerance::default(),
    )
    .unwrap()
}

#[test]
fn story_b5_confounded_admg_ett_matches_the_latent_model_and_refuses_outside_the_cell() {
    use latent_scm::{LatentScm, Rng};
    let cards = [2usize, 3, 2];
    let names: Vec<String> = ["x", "m", "y"].iter().map(|s| (*s).to_string()).collect();
    let levels: Vec<Vec<f64>> = cards.iter().map(|&c| (0..c).map(|l| l as f64).collect()).collect();
    let ett = CounterfactualEventQuery::effect_on_treated(
        VariableId::from_raw(0),
        1.0,
        0.0,
        VariableId::from_raw(2),
        1.0,
    )
    .unwrap();
    let ctx = ExecutionContext::for_tests(3);
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-10 * (1.0 + a.abs().max(b.abs()));

    // Front door with a latent X <-> Y: the ETT is identified, never by backdoor.
    let scm = LatentScm::random(&mut Rng::new(8), &cards, &[(0, 1), (1, 2)], &[(0, 2)], 3);
    let (bytes, probability) = {
        let prepared = prepare_counterfactual_id(
            b5_admg(3, &[(0, 1), (1, 2)], &[(0, 2)]),
            names.clone(),
            levels.clone(),
            &ett,
            CounterfactualIdOptions::default(),
            &ctx,
        )
        .unwrap();
        let effect = prepared.evaluate(&b5_law(&cards, scm.observational(), "s8"), &ctx).unwrap();
        let truths = scm.ett_numerators(0, 1, 0, 2);
        let total: f64 = truths.iter().sum();
        assert!(close(effect.probability, truths[1] / total), "{}", effect.probability);
        (effect.export_artifact().unwrap(), effect.probability)
    };

    let consumed =
        consume_counterfactual_id_artifact(&bytes, CounterfactualIdConsumeLimits::default(), &ctx)
            .unwrap();
    assert!(consumed.independently_verified);
    assert_eq!(consumed.probability.to_bits(), probability.to_bits());

    // Mutations: unsealed (digests) and re-sealed (replay), four categories.
    let base = CounterfactualIdArtifactWire::decode(&bytes).unwrap();
    let consume_wire = |w: &CounterfactualIdArtifactWire| {
        consume_counterfactual_id_artifact(
            &w.export().unwrap(),
            CounterfactualIdConsumeLimits::default(),
            &ctx,
        )
    };
    let resealed = |mut w: CounterfactualIdArtifactWire| {
        w.data_digest.clear();
        w.premises_digest.clear();
        w.sealed().unwrap()
    };
    let mut w = base.clone();
    w.bidirected.clear();
    assert_eq!(consume_wire(&w).unwrap_err(), CounterfactualIdArtifactError::PremisesMismatch);
    assert!(matches!(
        consume_wire(&resealed(w)),
        Err(CounterfactualIdArtifactError::DerivationMismatch(_))
    ));
    // Source identity: the law's provider snapshot. Re-sealed, the artifact is
    // bound to another snapshot, so the data-bound consumer refuses it.
    let mut w = base.clone();
    w.law.snapshot = "elsewhere".into();
    assert_eq!(consume_wire(&w).unwrap_err(), CounterfactualIdArtifactError::DataIdentityMismatch);
    let rebound = resealed(w);
    assert_eq!(
        consume_counterfactual_id_artifact_for_data(
            &rebound.export().unwrap(),
            &base.data_digest,
            CounterfactualIdConsumeLimits::default(),
            &ctx,
        )
        .unwrap_err(),
        CounterfactualIdArtifactError::DataIdentityMismatch
    );
    let mut w = base.clone();
    w.law.probabilities.swap(0, 1);
    assert_eq!(consume_wire(&w).unwrap_err(), CounterfactualIdArtifactError::DataIdentityMismatch);
    assert_eq!(
        consume_wire(&resealed(w)).unwrap_err(),
        CounterfactualIdArtifactError::PointMismatch
    );
    let mut w = base.clone();
    w.point.probability = (f64::from_bits(w.point.probability) + 1e-12).to_bits();
    assert_eq!(consume_wire(&w).unwrap_err(), CounterfactualIdArtifactError::PointMismatch);
    assert_eq!(
        consume_wire(&resealed(w)).unwrap_err(),
        CounterfactualIdArtifactError::PointMismatch
    );

    // A non-identified event (the bow arc X -> Y, X <-> Y) refuses, never a number.
    let text = prepare_counterfactual_id(
        b5_admg(3, &[(0, 2), (1, 2)], &[(0, 2)]),
        names.clone(),
        vec![vec![0.0, 1.0]; 3],
        &ett,
        CounterfactualIdOptions::default(),
        &ctx,
    )
    .unwrap_err()
    .to_string();
    assert!(
        text.contains("reason=route_not_supported: counterfactual_id.conflicting_subscripts"),
        "{text}"
    );
    // Uncertainty is refused: the cell is point-only.
    let text = prepare_counterfactual_id(
        b5_admg(3, &[(0, 1), (1, 2)], &[(0, 2)]),
        names,
        levels,
        &ett,
        CounterfactualIdOptions { interval_requested: true, ..CounterfactualIdOptions::default() },
        &ctx,
    )
    .unwrap_err()
    .to_string();
    assert!(
        text.contains("reason=estimator_inference_mismatch: counterfactual_id.interval_requested"),
        "{text}"
    );
}

// ---------------------------------------------------------------------------
// B6 (X10): exact binary causal observation recovery.
// ---------------------------------------------------------------------------

const B6_EDGES: [(u32, u32); 3] = [(2, 0), (2, 1), (0, 1)];

fn b6_effect(model: &recovery_scm::MModel) -> RecoveredEffectQuery {
    use recovery_scm::v;
    let mut graph = Admg::empty();
    for n in 0..model.k + model.m {
        graph.add_node(NodeRef::Static(v(n))).unwrap();
    }
    for (a, b) in B6_EDGES {
        graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    RecoveredEffectQuery { graph, outcomes: Arc::from([v(1)]), treatments: Arc::from([v(0)]) }
}

fn b6_prepare(
    model: &recovery_scm::MModel,
    graph: Admg,
    law: ExactDiscreteLaw,
) -> Result<antecedent::PreparedObservationRecovery, IoError> {
    use recovery_scm::v;
    StudyBuilder::observation_recovery(
        graph,
        &model.query(),
        model.catalog(),
        Some(b6_effect(model)),
        RecoveryLimits::default(),
        law,
        [0.0, 1.0].iter().map(|x| Assignment::from_pairs([(v(0), Value::f64(*x))])).collect(),
        ExactEvaluationLimits::default(),
        &ExecutionContext::for_tests(1),
    )
}

#[test]
fn story_b6_binary_m_graph_recovers_the_joint_law_and_feeds_downstream_identification() {
    use recovery_scm::MModel;
    let ctx = ExecutionContext::for_tests(1);
    // Treatment X0, outcome X1, confounder O; R0 <- O, R1 <- X0.
    let model = MModel::new(2, 1, &B6_EDGES, &[vec![2], vec![0]], 17).unwrap();
    let names: Vec<String> = (0..model.nodes()).map(|n| format!("n{n}")).collect();

    let (bytes, recovered, effects) = {
        let prepared = b6_prepare(&model, model.graph.clone(), model.observed_law()).unwrap();
        let result = prepared.estimate(&ctx).unwrap();
        // The recovered joint law equals the complete-data truth...
        let law = result.recovered().law().probabilities();
        assert!(law.iter().zip(model.truth()).all(|(a, b)| (a - b).abs() < 1e-12));
        // ...and feeds the downstream backdoor effect equal to the interventional truth.
        for (point, level) in result.effects().iter().zip([0usize, 1]) {
            let one = point.atoms.iter().position(|a| a[0].as_f64() == Some(1.0)).unwrap();
            assert!((point.probabilities[one] - model.interventional(1, 0, level)).abs() < 1e-12);
        }
        // A counted law is a sampled provider: refused as not licensed.
        let law = model.observed_law();
        let counts: Vec<u64> = law.probabilities().iter().map(|_| 1).collect();
        let uniform = ExactDiscreteLaw::try_new(
            law.population(),
            law.regime(),
            Vec::new(),
            law.axes().to_vec(),
            vec![1.0 / counts.len() as f64; counts.len()],
            law.snapshot_identity(),
            LawTolerance::default(),
        )
        .unwrap()
        .with_empirical_counts(counts)
        .unwrap();
        let error = prepared.refresh(uniform, &ctx).unwrap_err();
        assert_eq!(error.reason_code(), Some("cell_not_licensed"));
        assert!(error.to_string().contains("recovery.empirical_not_licensed"), "{error}");
        (
            result.export_named(&prepared, &names).unwrap(),
            bits(result.recovered().law().probabilities()),
            result.effects().iter().map(|e| bits(&e.probabilities)).collect::<Vec<_>>(),
        )
    };

    let consumed =
        consume_observation_recovery_artifact(&bytes, RecoveryConsumeLimits::default(), &ctx)
            .unwrap();
    assert_eq!(bits(consumed.recovered().law().probabilities()), recovered);
    assert_eq!(
        consumed.effects().iter().map(|e| bits(&e.probabilities)).collect::<Vec<_>>(),
        effects
    );

    // Mutations: unsealed and re-sealed, four categories.
    let original = RecoveryArtifactWire::decode(&bytes).unwrap();
    let consume = |w: &RecoveryArtifactWire| {
        RecoveryArtifactWire::consume_typed(
            &w.export().unwrap(),
            RecoveryConsumeLimits::default(),
            &ctx,
        )
        .map(|_| ())
    };
    let reseal = |mut w: RecoveryArtifactWire| {
        w.premises_digest = w.expected_premises_digest().unwrap();
        w.data_digest = w.expected_data_digest().unwrap();
        w
    };
    let proof = |error: RecoveryArtifactError| matches!(error, RecoveryArtifactError::ProofMismatch(e) if e.detail == RecoveryDetail::InvalidDerivation);
    let mut w = original.clone();
    w.graph.directed.push((2, 4)); // O -> R1: another m-graph.
    assert_eq!(consume(&w).unwrap_err(), RecoveryArtifactError::PremisesMismatch);
    assert!(proof(consume(&reseal(w)).unwrap_err()), "re-sealed premises");
    let mut w = original.clone();
    let mut catalog = w.catalog.to_catalog().unwrap();
    Arc::make_mut(&mut catalog.bindings)[0].snapshot_identity = Arc::from("snap-renamed");
    w.catalog = antecedent_io::transport_catalog_wire::EvidenceCatalogWire::from_catalog(&catalog);
    w.observed_law.snapshot = "snap-renamed".into();
    assert_eq!(consume(&w).unwrap_err(), RecoveryArtifactError::DataIdentityMismatch);
    assert!(proof(consume(&reseal(w)).unwrap_err()), "re-sealed source identity");
    let other = MModel::new(2, 1, &B6_EDGES, &[vec![2], vec![0]], 99).unwrap().observed_law();
    let mut w = original.clone();
    w.observed_law.probabilities = other.probabilities().to_vec();
    assert_eq!(consume(&w).unwrap_err(), RecoveryArtifactError::DataIdentityMismatch);
    assert_eq!(consume(&reseal(w)).unwrap_err(), RecoveryArtifactError::RecoveredMismatch);
    let mut w = original.clone();
    w.recovered.probabilities[0] = f64::from_bits(w.recovered.probabilities[0].to_bits() + 1);
    assert_eq!(consume(&w).unwrap_err(), RecoveryArtifactError::RecoveredMismatch);
    assert_eq!(consume(&reseal(w)).unwrap_err(), RecoveryArtifactError::RecoveredMismatch);

    // A mechanism outside the class (unmeasured confounding) is refused, not "unrecoverable".
    let mut confounded = model.graph.clone();
    confounded.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let error = b6_prepare(&model, confounded, model.observed_law()).unwrap_err();
    assert_eq!(error.reason_code(), Some("route_not_supported"));
    assert!(error.to_string().contains("recovery.unsupported_mechanism"), "{error}");
}
