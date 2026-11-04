//! The source-backed sensitivity surface equals the hand-derived assumption range.
//!
//! The registered formula is `psi = sum_w P_W(w) [m(w,1) - m(w,0)]` with `m(w,x) = P(y=1|w,x)`,
//! and the box-contaminated extrema are `U = (1 - e_W) sum_w P_W(w) D+(w) + e_W max_w D+(w)`,
//! `D+(w) = (1 - e_Y) Delta(w) + e_Y (y_max - y_min)`, and `L` symmetric with the minimum.
//! Here `P_W = (0.65, 0.35)`, `m(0,0) = 0.2`, `m(1,0) = 0.3`, `m(1,1) = 0.8`, `m(0,1) = 0.5 + shift`,
//! `e_W = 0.3`, `e_Y = 0.2` and a binary outcome (range 1).
//!
//! shift 0:   `Delta` = (0.3, 0.5), `psi` = 0.195 + 0.175 = 0.37; `D+` = (0.44, 0.60),
//!            U = 0.7 * 0.496 + 0.3 * 0.60 = 0.5272; D- = (0.04, 0.20),
//!            L = 0.7 * 0.096 + 0.3 * 0.04 = 0.0792.
//! shift 0.1: Delta = (0.4, 0.5), psi = 0.26 + 0.175 = 0.435; D+ = (0.52, 0.60),
//!            U = 0.7 * 0.548 + 0.3 * 0.60 = 0.5636; D- = (0.12, 0.20),
//!            L = 0.7 * 0.148 + 0.3 * 0.12 = 0.1396.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{JointSensitivityArtifactWire, PreparedZTransport, StudyBuilder};
use antecedent_core::{
    DependenceGroup, DistributionAvailability, Environment, EvidenceCatalog, EvidenceKind,
    EvidenceRegime, ExecutionContext, InterventionAssignment, RegimeBinding, RegimeId, RegimeKind,
    SamplingDesign, Value, VariableCoordinate, VariableDomain, VariableId,
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
use antecedent_validate::{JointDeviationSpec, JointFactor, JointFactorBound};

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

fn builder(shift: f64) -> Builder {
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
    Builder { diagram: diagram(), functional, data: data(shift) }
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

fn artifact(shift: f64) -> JointSensitivityArtifactWire {
    let built = builder(shift);
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

fn surface_of(shift: f64) -> antecedent_io::sensitivity_artifact::SensitivityArtifact {
    use antecedent_contract::{QuantityRole, ScientificQuantity};
    use antecedent_io::sensitivity_artifact::{ActionUtility, SensitivityArtifact, UtilityTerm};
    let prepared = prepare(&builder(shift));
    let context = ExecutionContext::for_tests(91);
    let result = prepared.joint_mechanism_sensitivity(&spec(), &context).unwrap();
    let effect = ScientificQuantity {
        variable_id: "effect".into(),
        variable_name: "Y".into(),
        role: QuantityRole::Outcome,
        units: "outcome_units".into(),
        population_id: "target".into(),
        regime_id: "do(x=1)".into(),
        horizon: 0,
        functional_id: "sensitivity_surface".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    };
    SensitivityArtifact::from_joint_result(
        &result,
        &effect,
        2,
        vec![
            ActionUtility { id: "adopt".into(), utility: UtilityTerm::quantity("effect") },
            ActionUtility { id: "wait".into(), utility: UtilityTerm::Const(0.) },
        ],
        "checked-contract",
    )
    .unwrap()
}

fn near(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

#[test]
fn the_replayed_surface_is_the_hand_derived_range_and_moves_with_the_law() {
    use antecedent::analysis::sensitivity_source::SourceBackedSensitivity;
    let context = ExecutionContext::for_tests(91);
    for (shift, psi, lower, upper) in [(0.0, 0.37, 0.0792, 0.5272), (0.1, 0.435, 0.1396, 0.5636)] {
        let surface = surface_of(shift);
        let source = artifact(shift).export().unwrap();
        let checked = SourceBackedSensitivity::checked(&source, &surface, &context).unwrap();
        let loaded =
            SourceBackedSensitivity::consume(&checked.export().unwrap(), &context).unwrap();
        let quantity = &loaded.surface().quantities()[0];
        assert_eq!(loaded.surface().grid(), &[0.0, 1.0]);
        assert!(near(quantity.lower[0], psi) && near(quantity.upper[0], psi), "{shift}");
        assert!(near(quantity.lower[1], lower), "{shift}: {}", quantity.lower[1]);
        assert!(near(quantity.upper[1], upper), "{shift}: {}", quantity.upper[1]);
    }
}

#[test]
fn a_surface_from_a_perturbed_law_is_not_replayed_by_the_unperturbed_source() {
    use antecedent::analysis::sensitivity_source::SourceBackedSensitivity;
    let context = ExecutionContext::for_tests(91);
    let source = artifact(0.0).export().unwrap();
    // The shifted law gives psi 0.435, not the source's 0.37: the replay recomputes 0.37.
    let error = SourceBackedSensitivity::checked(&source, &surface_of(0.1), &context).unwrap_err();
    assert_eq!(
        refused(&error),
        (
            "invalid_argument",
            antecedent_core::reason_details::sensitivity_source::SURFACE_MISMATCH.into()
        )
    );
}
