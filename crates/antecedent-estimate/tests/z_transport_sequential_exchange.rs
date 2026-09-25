//! Exact evidence for a recursive z-transport derivation with two source exchanges.
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    DependenceGroup, DistributionAvailability, Environment, EvidenceCatalog, EvidenceKind,
    EvidenceRegime, ExecutionContext, InterventionAssignment, RegimeBinding, RegimeId, RegimeKind,
    SamplingDesign, Value, VariableCoordinate, VariableDomain, VariableId,
};
use antecedent_estimate::evaluate_exact_z_transport;
use antecedent_expr::{
    Assignment, DiscreteAxis, ExactDiscreteLaw, ExactEvaluationLimits, ExactTransportData,
    InterventionAssignment as ExprInterventionAssignment, LawTolerance,
};
use antecedent_graph::{Admg, DenseNodeId, SelectionDiagram};
use antecedent_identify::{
    SidLimits, ZTransportDecision, ZTransportQuery, bind_z_transport_catalog,
    decide_z_transport_with_catalog,
};

const N: usize = 5;
const X: VariableId = VariableId::from_raw(0);
const Z: VariableId = VariableId::from_raw(1);
const V: VariableId = VariableId::from_raw(2);
const Y: VariableId = VariableId::from_raw(3);
const R: VariableId = VariableId::from_raw(4);

#[derive(Clone, Copy)]
struct RegimeSpec {
    kind: RegimeKind,
    assignments: [(VariableId, bool); 2],
    assignment_count: usize,
    population: &'static str,
}

fn diagram() -> SelectionDiagram {
    // X -> Z -> Y, X <-> Z, X <-> Y, Z <-> V. R is an isolated measured
    // coordinate. The graph makes a multi-step recursive source exchange
    // necessary even though no selection node is present.
    let mut graph = Admg::with_variables(u32::try_from(N).unwrap());
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    graph.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(3)).unwrap();
    graph.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    graph.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(3)).unwrap();
    graph.insert_bidirected(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from([])).unwrap()
}

fn source_experiments() -> Vec<RegimeSpec> {
    let mut regimes = Vec::new();
    // Full family over controllable set {X, Z}: every nonempty subset and
    // every binary assignment, including the joint do(X,Z) laws.
    for x in [false, true] {
        regimes.push(RegimeSpec {
            kind: RegimeKind::Experimental,
            assignments: [(X, x), (Z, false)],
            assignment_count: 1,
            population: "source",
        });
    }
    for z in [false, true] {
        regimes.push(RegimeSpec {
            kind: RegimeKind::Experimental,
            assignments: [(Z, z), (X, false)],
            assignment_count: 1,
            population: "source",
        });
    }
    for x in [false, true] {
        for z in [false, true] {
            regimes.push(RegimeSpec {
                kind: RegimeKind::Experimental,
                assignments: [(X, x), (Z, z)],
                assignment_count: 2,
                population: "source",
            });
        }
    }
    regimes
}

fn enumerate_law(spec: RegimeSpec) -> Vec<f64> {
    let observed = [X, Z, V, Y, R];
    let measured = observed
        .iter()
        .copied()
        .filter(|variable| assigned(spec, *variable).is_none())
        .collect::<Vec<_>>();
    let mut probabilities = vec![0.0; 1 << measured.len()];
    for uxz in [false, true] {
        for uxy in [false, true] {
            for uzv in [false, true] {
                for ur in [false, true] {
                    let x = assigned(spec, X).unwrap_or(uxz ^ uxy);
                    let z = assigned(spec, Z).unwrap_or(x ^ uxz);
                    let v = z ^ uzv;
                    let y = z ^ uxy;
                    let values = [(X, x), (Z, z), (V, v), (Y, y), (R, ur)];
                    let index = measured.iter().fold(0_usize, |index, variable| {
                        let value =
                            values.iter().find(|(candidate, _)| candidate == variable).unwrap().1;
                        (index << 1) | usize::from(value)
                    });
                    probabilities[index] += 1.0 / 16.0;
                }
            }
        }
    }
    probabilities
}

fn assigned(spec: RegimeSpec, variable: VariableId) -> Option<bool> {
    spec.assignments[..spec.assignment_count]
        .iter()
        .find(|(candidate, _)| *candidate == variable)
        .map(|(_, value)| *value)
}

fn independently_enumerated_target_y_risk(do_x: bool) -> f64 {
    let mut y_ones = 0_u32;
    let mut states = 0_u32;
    for uxz in [false, true] {
        for uxy in [false, true] {
            for _uzv in [false, true] {
                for _ur in [false, true] {
                    let z = do_x ^ uxz;
                    let y = z ^ uxy;
                    y_ones += u32::from(y);
                    states += 1;
                }
            }
        }
    }
    f64::from(y_ones) / f64::from(states)
}

#[expect(
    clippy::too_many_lines,
    reason = "the fixture setup keeps its catalog, binding, and exact laws visibly aligned"
)]
fn setup() -> (SelectionDiagram, ZTransportQuery, EvidenceCatalog, ExactTransportData) {
    let diagram = diagram();
    let query = ZTransportQuery {
        outcomes: Arc::from([Y]),
        treatments: Arc::from([X]),
        controllable: Arc::from([X, Z]),
        // This selected assignment is used by the formula. The declared
        // controllable family remains larger and is fully catalogued below.
        experiment_assignment: Arc::from([
            InterventionAssignment { variable: X, value: Value::Bool(false) },
            InterventionAssignment { variable: Z, value: Value::Bool(false) },
        ]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    };

    let coordinates = [X, Z, V, Y, R].map(|variable| VariableCoordinate {
        variable,
        domain: VariableDomain::Binary,
        unit: None,
    });
    let source = Environment::try_new("source", coordinates.clone(), []).unwrap();
    let target = Environment::try_new("target", coordinates.clone(), []).unwrap();
    let measured: Arc<[VariableId]> = Arc::from([X, Z, V, Y, R]);
    let mut regimes = Vec::new();
    let mut bindings = Vec::new();
    let mut laws = Vec::new();
    let mut add_regime = |kind, population, assignments: &[(VariableId, bool)]| {
        let id = RegimeId::from_raw(u32::try_from(regimes.len()).unwrap());
        let intervention_ids =
            assignments.iter().map(|(variable, _)| *variable).collect::<Vec<_>>();
        let intervention_values = assignments
            .iter()
            .map(|(variable, value)| InterventionAssignment {
                variable: *variable,
                value: Value::Bool(*value),
            })
            .collect::<Vec<_>>();
        regimes.push(
            EvidenceRegime::try_new(
                id,
                kind,
                EvidenceKind::Available,
                intervention_ids,
                intervention_values,
                Arc::clone(&measured),
                population,
                DistributionAvailability::Joint,
            )
            .unwrap(),
        );
        bindings.push(RegimeBinding {
            dataset_identity: None,
            regime: id,
            snapshot_identity: Arc::from(format!("{population}-regime-{}", id.raw())),
            schema_names: Arc::from([]),
            sampling: SamplingDesign::Independent,
            weights: None,
            dependence: DependenceGroup::IndependentStudies,
        });
        let mut spec = RegimeSpec {
            kind,
            assignments: [(X, false), (X, false)],
            assignment_count: assignments.len(),
            population,
        };
        for (index, assignment) in assignments.iter().enumerate() {
            spec.assignments[index] = *assignment;
        }
        let axes = [X, Z, V, Y, R]
            .into_iter()
            .filter(|variable| assigned(spec, *variable).is_none())
            .map(|variable| DiscreteAxis {
                variable,
                values: Arc::from([Value::Bool(false), Value::Bool(true)]),
            })
            .collect::<Vec<_>>();
        let probabilities = enumerate_law(spec);
        let expression_assignments = assignments
            .iter()
            .map(|(variable, value)| {
                ExprInterventionAssignment::concrete(*variable, Value::Bool(*value))
            })
            .collect::<Vec<_>>();
        laws.push(
            ExactDiscreteLaw::try_new(
                population,
                id,
                expression_assignments,
                axes,
                probabilities,
                format!("{population}-regime-{}", id.raw()),
                LawTolerance::default(),
            )
            .unwrap(),
        );
    };
    // Source and target observational laws make the population evidence
    // explicit in the catalog, even though this formula only uses experiments.
    add_regime(RegimeKind::Observational, "source", &[]);
    add_regime(RegimeKind::Observational, "target", &[]);
    for spec in source_experiments() {
        let assignments = spec.assignments[..spec.assignment_count].to_vec();
        add_regime(spec.kind, spec.population, &assignments);
    }
    let catalog = EvidenceCatalog::try_new([source, target], regimes, bindings, None).unwrap();
    let exact = ExactTransportData::try_new(laws, 128).unwrap();
    (diagram, query, catalog, exact)
}

#[test]
fn recursive_formula_uses_two_source_exchange_factors_and_matches_exact_scm_truth() {
    let (diagram, query, catalog, exact) = setup();
    let context = ExecutionContext::for_tests(5521);
    let ZTransportDecision::Identified(derivation) =
        decide_z_transport_with_catalog(&diagram, &query, &catalog, SidLimits::default(), &context)
            .unwrap()
    else {
        panic!("the checked two-exchange query should be identified");
    };
    let rules = &derivation.to_record().rules;
    assert_eq!(
        rules.iter().filter(|rule| rule.contains("line10.source_exchange")).count(),
        2,
        "each exchange factor must remain explicit in the recursive proof: {rules:?}"
    );
    assert!(rules.iter().any(|rule| rule.contains("[(0, Some(0.0))]")), "{rules:?}");
    assert!(
        rules.iter().any(|rule| rule.contains("[(0, Some(0.0)), (1, Some(0.0))]")),
        "{rules:?}"
    );
    let bound = bind_z_transport_catalog(&diagram, &query, &derivation, &catalog).unwrap();
    let result = evaluate_exact_z_transport(
        &bound,
        exact,
        Assignment::from_pairs([(X, Value::Bool(false))]),
        ExactEvaluationLimits::default(),
        &context,
    )
    .unwrap();
    let y_risk = result
        .atoms
        .iter()
        .zip(result.probabilities.iter())
        .filter(|(atom, _)| atom[0] == Value::Bool(true))
        .map(|(_, probability)| probability)
        .sum::<f64>();
    // Enumerate the structural equations under do(X=0), separately from the
    // provider's law construction, to obtain the known target truth.
    let truth = independently_enumerated_target_y_risk(false);
    assert!((truth - 0.5).abs() < 1e-12, "independent SCM truth={truth}");
    assert!((y_risk - truth).abs() < 1e-12, "P(Y=1 | do(X=0))={y_risk}");
    assert_eq!(result.outcomes.as_ref(), &[Y]);
}
