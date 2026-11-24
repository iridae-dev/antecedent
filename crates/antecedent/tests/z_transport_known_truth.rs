//! Known truth for the prepared single-source z-transport route, enumerated from a structural
//! causal model.
//!
//! The model is on the registered surrogate graph W -> Z -> X -> Y, W -> Y with the confounding
//! arcs Z <-> Y and Z <-> X (and W <-> Y allowed). Three independent latent bits drive it:
//! `U1 ~ Bern(0.3)`, `U2 ~ Bern(0.4)`, `U3 ~ Bern(0.35)`, with
//!
//! ```text
//! W = U1,  Z = W xor U2 xor U3,  X = Z xor U3,  Y = (X and U2) or (W and not X).
//! ```
//!
//! No selection node is declared, so the source and target share the model. The target law
//! `P*(Y = 1 | do(X = x))` is read off the equations by setting X and summing over the latents:
//! `Y = U2` when `x = 1` (0.4) and `Y = W = U1` when `x = 0` (0.3). The only evidence is the
//! source joint of (W, X, Y) under `do(Z = 0)`, also enumerated from the equations (there
//! `X = U3`). The observational target conditional `P(Y = 1 | X = x)` is confounded through U2
//! and W (0.28 / 0.46 at x = 1, 0.12 / 0.54 at x = 0), so a route that read it instead of the
//! identified formula would miss the truth. The test computes every number from the equations
//! and never calls the identifier to compute a probability; the identifier only supplies the
//! formula the prepared route evaluates.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{StudyBuilder, ZTransportResult};
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
    BoundZTransportFunctional, SidLimits, ZTransportQuery, ZTransportResult as Identified,
    bind_z_transport_catalog, identify_z_transport,
};

const W: VariableId = VariableId::from_raw(0);
const Z: VariableId = VariableId::from_raw(1);
const X: VariableId = VariableId::from_raw(2);
const Y: VariableId = VariableId::from_raw(3);

const P_U1: f64 = 0.3;
const P_U2: f64 = 0.4;
const P_U3: f64 = 0.35;

/// The structural equations. `do_z` and `do_x` replace the equation of Z or X.
fn world(u: [bool; 3], do_z: Option<bool>, do_x: Option<bool>) -> [bool; 4] {
    let [u1, u2, u3] = u;
    let w = u1;
    let z = do_z.unwrap_or(w ^ u2 ^ u3);
    let x = do_x.unwrap_or(z ^ u3);
    let y = (x && u2) || (w && !x);
    [w, z, x, y]
}

/// Every latent configuration with its probability.
fn latents() -> Vec<([bool; 3], f64)> {
    let bit = |value: bool, p: f64| if value { p } else { 1.0 - p };
    let mut out = Vec::with_capacity(8);
    for u1 in [false, true] {
        for u2 in [false, true] {
            for u3 in [false, true] {
                out.push(([u1, u2, u3], bit(u1, P_U1) * bit(u2, P_U2) * bit(u3, P_U3)));
            }
        }
    }
    out
}

/// `P*(Y = 1 | do(X = x))` from the equations.
fn truth(x: bool) -> f64 {
    latents().iter().filter(|(u, _)| world(*u, None, Some(x))[3]).map(|(_, p)| p).sum()
}

/// The observational `P(Y = 1 | X = x)` of the unintervened model.
fn observational(x: bool) -> f64 {
    let (mut joint, mut margin) = (0.0, 0.0);
    for (u, p) in latents() {
        let v = world(u, None, None);
        if v[2] == x {
            margin += p;
            if v[3] {
                joint += p;
            }
        }
    }
    joint / margin
}

/// The source joint of (W, X, Y) under `do(Z = 0)`, W slowest and Y fastest.
fn source_joint() -> Vec<f64> {
    let mut joint = vec![0.0; 8];
    for (u, p) in latents() {
        let [w, _, x, y] = world(u, Some(false), None);
        joint[usize::from(w) * 4 + usize::from(x) * 2 + usize::from(y)] += p;
    }
    joint
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

fn functional() -> BoundZTransportFunctional {
    let query = ZTransportQuery {
        outcomes: Arc::from([Y]),
        treatments: Arc::from([X]),
        controllable: Arc::from([Z]),
        experiment_assignment: Arc::from([InterventionAssignment {
            variable: Z,
            value: Value::Bool(false),
        }]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    };
    let Identified::Identified(proof) = identify_z_transport(
        &diagram(),
        &query,
        SidLimits::default(),
        &ExecutionContext::for_tests(0),
    )
    .unwrap() else {
        panic!("the surrogate graph identifies P*(Y | do(X)) from the do(Z) experiment");
    };
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
    let coordinates = [W, Z, X, Y].map(|variable| VariableCoordinate {
        variable,
        domain: VariableDomain::Binary,
        unit: None,
    });
    let catalog = EvidenceCatalog::try_new(
        [Environment::try_new("source", coordinates, Arc::<[VariableId]>::from([])).unwrap()],
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
    bind_z_transport_catalog(&diagram(), &query, &proof, &catalog).unwrap()
}

fn law(probabilities: Vec<f64>, counts: Option<Vec<u64>>) -> ExactTransportData {
    let binary = |variable| DiscreteAxis {
        variable,
        values: Arc::from([Value::Bool(false), Value::Bool(true)]),
    };
    let mut law = ExactDiscreteLaw::try_new(
        "source",
        RegimeId::from_raw(0),
        [antecedent_expr::InterventionAssignment::concrete(Z, Value::Bool(false))],
        [binary(W), binary(X), binary(Y)],
        probabilities,
        "snapshot-0",
        LawTolerance::default(),
    )
    .unwrap();
    if let Some(counts) = counts {
        law = law.with_empirical_counts(counts).unwrap();
    }
    ExactTransportData::try_new([law], 64).unwrap()
}

fn y_risk(result: &ZTransportResult) -> f64 {
    let distribution = result.distribution();
    assert_eq!(distribution.outcomes.as_ref(), [Y]);
    let total: f64 = distribution.probabilities.iter().sum();
    assert!((total - 1.0).abs() < 1e-12, "the target law sums to {total}");
    distribution
        .atoms
        .iter()
        .zip(distribution.probabilities.iter())
        .filter(|(atom, _)| atom[0] == Value::Bool(true))
        .map(|(_, probability)| probability)
        .sum()
}

#[test]
fn the_prepared_route_recovers_the_enumerated_target_law_from_the_source_experiment() {
    // The truth from the equations, and its hand-derived value.
    assert!((truth(true) - P_U2).abs() < 1e-12, "{}", truth(true));
    assert!((truth(false) - P_U1).abs() < 1e-12, "{}", truth(false));
    // The confounded observational conditional is not the truth.
    assert!((observational(true) - 0.28 / 0.46).abs() < 1e-12);
    assert!((observational(false) - 0.12 / 0.54).abs() < 1e-12);
    for x in [false, true] {
        assert!((observational(x) - truth(x)).abs() > 0.05, "the fixture must separate them");
    }
    let joint = source_joint();
    assert!((joint.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    let ctx = ExecutionContext::for_tests(3);
    for x in [false, true] {
        let prepared = StudyBuilder::z_transport(
            diagram(),
            functional(),
            law(joint.clone(), None),
            Assignment::from_pairs([(X, Value::Bool(x))]),
            ExactEvaluationLimits::default(),
            &ctx,
        )
        .unwrap();
        let result = prepared.estimate(&ctx).unwrap();
        let risk = y_risk(&result);
        assert!(
            (risk - truth(x)).abs() < 1e-12,
            "P*(Y=1 | do(X={x})) = {risk}, truth {}",
            truth(x)
        );
        assert_eq!(result.interval_type(), "no_interval_reported");
        assert!(result.mean_intervals().is_empty());
    }
}

#[test]
fn an_empirical_table_at_the_law_returns_the_enumerated_point_with_a_nominal_interval() {
    // 10 000 units whose cell counts equal the source law exactly (every cell is a multiple of
    // 1 / 10 000), so the plug-in point is the enumerated truth.
    let joint = source_joint();
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a probability in [0, 1] scaled by 10 000 and rounded to a count"
    )]
    let counts: Vec<u64> = joint.iter().map(|p| (p * 10_000.0).round() as u64).collect();
    assert_eq!(counts.iter().sum::<u64>(), 10_000);
    for (count, p) in counts.iter().zip(&joint) {
        assert!((*count as f64 / 10_000.0 - p).abs() < 1e-12, "a cell is not on the grid");
    }
    let frequencies: Vec<f64> = counts.iter().map(|count| *count as f64 / 10_000.0).collect();
    let ctx = ExecutionContext::for_tests(5);
    for x in [false, true] {
        let prepared = StudyBuilder::z_transport_empirical(
            diagram(),
            functional(),
            law(frequencies.clone(), Some(counts.clone())),
            Assignment::from_pairs([(X, Value::Bool(x))]),
            ExactEvaluationLimits::default(),
            &ctx,
        )
        .unwrap();
        let result = prepared.estimate(&ctx).unwrap();
        let risk = y_risk(&result);
        assert!((risk - truth(x)).abs() < 1e-12, "plug-in P*(Y=1 | do(X={x})) = {risk}");
        assert_eq!(result.interval_method(), Some(antecedent_estimate::PERCENTILE_BOOTSTRAP));
        assert_eq!(
            result.interval_reason(),
            antecedent_estimate::Z_TRANSPORT_INTERVAL_NOT_MEASURED
        );
        let (outcome, lower, upper) = result.mean_intervals()[0];
        assert_eq!(outcome, Y);
        assert!(lower <= risk && risk <= upper, "[{lower}, {upper}] around {risk}");
        assert!(upper - lower < 0.1, "10 000 units give a narrow interval");
    }
}
