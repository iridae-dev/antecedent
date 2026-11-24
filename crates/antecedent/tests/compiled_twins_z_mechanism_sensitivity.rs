//! The compiled outcome-kernel contamination range of a checked z-transport formula against the
//! direct kernel call, the joint route with only the kernel factor, and the closed form.
//!
//! `compile_z_transport_mechanism_sensitivity` calls `z_transport_mechanism_sensitivity` on the
//! same diagram, functional and data, so the twin asserts equality of the whole result and the
//! bits of the baseline, the range and the tipping fraction. The joint route with only the outcome
//! kernel factor is bit-equal to the one-factor route (the record's existing fixture); here the
//! compiled one-factor program is compared with it too. The closed form: with the outcome kernel of
//! every `(w, x)` stratum contaminated up to `eps`, the contrast
//! `psi = sum_w P(w) [m(w, 1) - m(w, 0)]` of a 0/1 outcome ranges over
//! `[(1 - eps) psi - eps, (1 - eps) psi + eps]`, and the upper end reaches a threshold `t` at
//! `eps = (t - psi) / (1 - psi)`.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::float_cmp, reason = "the bits are compared exactly")]

use std::sync::Arc;

use antecedent::analysis::z_transport_sensitivity_program::{
    TerminalError, compile_z_transport_mechanism_sensitivity,
};
use antecedent_core::{
    DependenceGroup, DistributionAvailability, Environment, EvidenceCatalog, EvidenceKind,
    EvidenceRegime, ExecutionContext, InterventionAssignment, RegimeBinding, RegimeId, RegimeKind,
    SamplingDesign, SemanticDigest, Value, VariableCoordinate, VariableDomain, VariableId,
};
use antecedent_expr::{DiscreteAxis, ExactDiscreteLaw, ExactTransportData, LawTolerance};
use antecedent_graph::{Admg, DenseNodeId, SelectionDiagram};
use antecedent_identify::{
    BoundZTransportFunctional, SidLimits, ZTransportQuery, ZTransportResult,
    bind_z_transport_catalog, identify_z_transport,
};
use antecedent_validate::{
    JointDeviationSpec, JointFactor, JointFactorBound, z_transport_joint_mechanism_sensitivity,
    z_transport_mechanism_sensitivity,
};

const W: VariableId = VariableId::from_raw(0);
const Z: VariableId = VariableId::from_raw(1);
const X: VariableId = VariableId::from_raw(2);
const Y: VariableId = VariableId::from_raw(3);

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

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

/// `P(Y = 1 | w, x)` of the source law, `shift` added at `(w, x) = (0, 1)`.
fn outcome_mean(w: bool, x: bool, shift: f64) -> f64 {
    match (w, x) {
        (false, false) => 0.2,
        (false, true) => 0.5 + shift,
        (true, false) => 0.3,
        (true, true) => 0.8,
    }
}

const P_W1: f64 = 0.35;

/// `P(W) P(X | W) P(Y | W, X)` under `do(Z = 0)`, W-major, last axis fastest.
fn data(shift: f64) -> ExactTransportData {
    let mut probabilities = Vec::with_capacity(8);
    for w in [false, true] {
        for x in [false, true] {
            for y in [false, true] {
                let p_w = if w { P_W1 } else { 1.0 - P_W1 };
                let p_x = if x { 0.4 } else { 0.6 };
                let p_y1 = outcome_mean(w, x, shift);
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
        [antecedent_expr::InterventionAssignment::concrete(Z, Value::Bool(false))],
        [binary(W), binary(X), binary(Y)],
        probabilities,
        "snapshot-0",
        LawTolerance::default(),
    )
    .unwrap();
    ExactTransportData::try_new([law], 64).unwrap()
}

fn functional() -> BoundZTransportFunctional {
    let ZTransportResult::Identified(proof) =
        identify_z_transport(&diagram(), &query(), SidLimits::default(), &ctx()).unwrap()
    else {
        panic!("the registered surrogate formula identifies");
    };
    bind_z_transport_catalog(&diagram(), &query(), &proof, &catalog()).unwrap()
}

#[test]
fn the_compiled_range_equals_the_direct_kernel_call() {
    for (fraction, threshold) in [(0.2, None), (0.2, Some(0.45)), (0.05, Some(0.3)), (0.5, None)] {
        let program = compile_z_transport_mechanism_sensitivity(
            diagram(),
            functional(),
            data(0.0),
            fraction,
            threshold,
            &ctx(),
        )
        .unwrap();
        assert_eq!(program.lowered().route(), "ZTransportMechanismSensitivity");
        let compiled = program.execute(&ctx()).unwrap();
        let direct = z_transport_mechanism_sensitivity(
            &diagram(),
            &functional(),
            &data(0.0),
            fraction,
            threshold,
            &ctx(),
        )
        .unwrap();
        assert_eq!(compiled, direct);
        assert_eq!(compiled.response.baseline.to_bits(), direct.response.baseline.to_bits());
        assert_eq!(compiled.response.minimum.to_bits(), direct.response.minimum.to_bits());
        assert_eq!(compiled.response.maximum.to_bits(), direct.response.maximum.to_bits());
        assert_eq!(
            compiled.response.tipping_fraction.map(f64::to_bits),
            direct.response.tipping_fraction.map(f64::to_bits)
        );
    }
    // A kernel refusal is the same refusal on both paths.
    let program = compile_z_transport_mechanism_sensitivity(
        diagram(),
        functional(),
        data(0.0),
        1.5,
        None,
        &ctx(),
    )
    .unwrap();
    let Err(TerminalError::Kernel(compiled)) = program.execute(&ctx()) else {
        panic!("a fraction above one is the kernel's refusal");
    };
    let direct =
        z_transport_mechanism_sensitivity(&diagram(), &functional(), &data(0.0), 1.5, None, &ctx())
            .unwrap_err();
    assert_eq!(compiled, direct);
    // Inside another program's lowered dispatch the compiler refuses.
    let scoped = ctx().enter_lowered_dispatch(SemanticDigest::from_bytes([9_u8; 32]));
    let refusal = compile_z_transport_mechanism_sensitivity(
        diagram(),
        functional(),
        data(0.0),
        0.2,
        None,
        &scoped,
    )
    .unwrap_err();
    assert!(refusal.to_string().contains("public_builder_reentry"));
}

#[test]
fn the_compiled_one_factor_range_is_the_joint_route_with_only_the_kernel_factor() {
    let program = compile_z_transport_mechanism_sensitivity(
        diagram(),
        functional(),
        data(0.0),
        0.2,
        Some(0.45),
        &ctx(),
    )
    .unwrap();
    let compiled = program.execute(&ctx()).unwrap();
    let mut kernel_only = JointDeviationSpec::new(vec![JointFactorBound {
        factor: JointFactor::OutcomeKernel,
        max_fraction: 0.2,
    }]);
    kernel_only.decision_threshold = Some(0.45);
    let joint = z_transport_joint_mechanism_sensitivity(
        &diagram(),
        &functional(),
        &data(0.0),
        &kernel_only,
        &ctx(),
    )
    .unwrap();
    assert_eq!(joint.baseline.to_bits(), compiled.response.baseline.to_bits());
    assert_eq!(joint.range.minimum.to_bits(), compiled.response.minimum.to_bits());
    assert_eq!(joint.range.maximum.to_bits(), compiled.response.maximum.to_bits());
    assert_eq!(
        joint.axis_tipping[0].analytic.map(f64::to_bits),
        compiled.response.tipping_fraction.map(f64::to_bits)
    );
    // The closed form of the contaminated contrast of a 0/1 outcome.
    let psi = (1.0 - P_W1) * (outcome_mean(false, true, 0.0) - outcome_mean(false, false, 0.0))
        + P_W1 * (outcome_mean(true, true, 0.0) - outcome_mean(true, false, 0.0));
    let eps = 0.2;
    assert!((compiled.response.baseline - psi).abs() < 1e-12);
    assert!((compiled.response.maximum - ((1.0 - eps) * psi + eps)).abs() < 1e-12);
    assert!((compiled.response.minimum - ((1.0 - eps) * psi - eps)).abs() < 1e-12);
    let tipping = compiled.response.tipping_fraction.unwrap();
    assert!((tipping - (0.45 - psi) / (1.0 - psi)).abs() < 1e-6, "{tipping}");
}

#[test]
fn the_z_mechanism_program_identity_is_its_declared_inputs() {
    let identity = |shift: f64, fraction: f64, threshold: Option<f64>| {
        compile_z_transport_mechanism_sensitivity(
            diagram(),
            functional(),
            data(shift),
            fraction,
            threshold,
            &ctx(),
        )
        .unwrap()
        .identity()
    };
    let base = identity(0.0, 0.2, Some(0.45));
    assert_eq!(base, identity(0.0, 0.2, Some(0.45)));
    assert_ne!(base, identity(0.0, 0.25, Some(0.45)));
    assert_ne!(base, identity(0.0, 0.2, None));
    assert_ne!(base, identity(0.0, 0.2, Some(0.4)));
    // Another source law is another program.
    assert_ne!(base, identity(0.05, 0.2, Some(0.45)));
}
