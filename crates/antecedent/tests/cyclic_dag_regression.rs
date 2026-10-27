//! The cyclic paths leave acyclic models on the DAG algorithm.
//!
//! A recovered law whose free support is lower triangular is evaluated by the DAG
//! schedule (`RecoveredSolvedBy::Acyclic`, an `AcyclicScm` plan of mechanism blocks)
//! and agrees with an independent forward substitution written here to 1e-12. A
//! specified equilibrium model with DAG structure compiles to mechanism blocks with no
//! simultaneous-equation block. On acyclic support the recovered evidence ports
//! equal those of a cyclic law with the same origin; only the solve route differs.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::cyclic_evidence::{CyclicEvidenceSchema, CyclicEvidenceSelector as S};
use antecedent::analysis::{CyclicExecutionRequest, Study};
use antecedent_core::ExecutionContext;
use antecedent_io::cyclic_family_artifact::{
    CertificateDemandWire, CyclicComponentWire, CyclicFamilyAnswer, CyclicFamilyRequest,
    CyclicInterventionWire, CyclicWorldOutcomeAnswer, CyclicWorldWire, LinearCyclicWire,
    RecoveredCoefficientWire, RecoveredDrawAnswer, RecoveredDrawOriginWire, RecoveredDrawWire,
    RecoveredInterventionWire, RecoveredSolvedBy,
};
use antecedent_model::compile::{CompiledSystemPlan, ExecutionBlock, SystemSemantics};
use antecedent_model::nonlinear_equilibrium::{
    Activation, Clamp, Equation, EquilibriumModel, SolvePolicy,
};

const TOLERANCE: f64 = 1e-12;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!((actual - expected).abs() <= tolerance, "{actual} vs {expected} (±{tolerance})");
}

fn mechanisms_only(plan: &CompiledSystemPlan) -> bool {
    plan.blocks.iter().all(|block| matches!(block, ExecutionBlock::Mechanism { .. }))
}

/// Gaussian law of `x = K x + c + u`, `u ~ N(mu, sigma)`, for a `K` that is strictly
/// lower triangular in node order, under an optional hard clamp. Independent of the
/// library: forward substitution of the mean and of `x` as a linear map of `u`.
fn dag_law(
    k: &[Vec<f64>],
    c: &[f64],
    mu: &[f64],
    sigma: &[f64],
    clamp: Option<(usize, f64)>,
) -> (Vec<f64>, Vec<f64>) {
    let n = c.len();
    let mut mean = vec![0.0; n];
    let mut map = vec![vec![0.0; n]; n];
    for i in 0..n {
        match clamp {
            Some((node, value)) if node == i => mean[i] = value,
            _ => {
                mean[i] = c[i] + mu[i];
                map[i][i] = 1.0;
                for j in 0..i {
                    mean[i] += k[i][j] * mean[j];
                    for m in 0..n {
                        map[i][m] += k[i][j] * map[j][m];
                    }
                }
            }
        }
    }
    let mut covariance = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut total = 0.0;
            for a in 0..n {
                for b in 0..n {
                    total += map[i][a] * sigma[a * n + b] * map[j][b];
                }
            }
            covariance[i * n + j] = total;
        }
    }
    (mean, covariance)
}

/// Lower-triangular coefficients `K[into][from]` of one draw, as wire entries.
/// Every off-diagonal coefficient is named; a structural zero is a stated zero, not an omission.
fn wire_coefficients(k: &[Vec<f64>]) -> Vec<RecoveredCoefficientWire> {
    let mut entries = Vec::new();
    for (into, row) in k.iter().enumerate() {
        for (from, value) in row.iter().enumerate() {
            if into != from {
                entries.push(RecoveredCoefficientWire {
                    into: u32::try_from(into).unwrap(),
                    from: u32::try_from(from).unwrap(),
                    value: *value,
                });
            }
        }
    }
    entries
}

const INTERCEPT: [f64; 3] = [0.2, -0.1, 0.4];
const NOISE_MEAN: [f64; 3] = [1.0, 0.0, 0.5];
const NOISE_COVARIANCE: [f64; 9] = [1.0, 0.2, 0.0, 0.2, 1.5, 0.1, 0.0, 0.1, 0.8];

fn triangular_draws() -> Vec<Vec<Vec<f64>>> {
    vec![
        vec![vec![0.0, 0.0, 0.0], vec![0.5, 0.0, 0.0], vec![0.3, -0.7, 0.0]],
        vec![vec![0.0, 0.0, 0.0], vec![0.25, 0.0, 0.0], vec![-0.4, 0.6, 0.0]],
    ]
}

#[test]
fn a_recovered_law_on_acyclic_support_is_the_dag_law() {
    let draws = triangular_draws();
    let clamps = [None, Some((0_usize, 2.0)), Some((1_usize, -1.0))];
    let request = CyclicFamilyRequest::RecoveredLinear {
        nodes: vec![0, 1, 2],
        draws: draws
            .iter()
            .map(|k| RecoveredDrawWire {
                coefficients: wire_coefficients(k),
                intercept: INTERCEPT.to_vec(),
                noise_mean: NOISE_MEAN.to_vec(),
                noise_covariance: NOISE_COVARIANCE.to_vec(),
                unresolved: Vec::new(),
                weight: 0.5,
                marked_inadmissible: false,
            })
            .collect(),
        worlds: clamps
            .iter()
            .map(|clamp| match clamp {
                None => RecoveredInterventionWire { clamp_nodes: vec![], clamp_values: vec![] },
                Some((node, value)) => RecoveredInterventionWire {
                    clamp_nodes: vec![*node],
                    clamp_values: vec![*value],
                },
            })
            .collect(),
        outcome: 2,
        source_id: "clamp-experiments".into(),
        certifying_evidence: vec!["held-out-regime".into()],
        uncertainty: RecoveredDrawOriginWire::Supplied,
    };
    let prepared = Study::prepare_recovered_linear(&request, &ctx()).unwrap();
    let inspection = prepared.inspect();
    assert_eq!(inspection.execution_plans.len(), clamps.len());
    for plan in &inspection.execution_plans {
        assert_eq!(plan.semantics, SystemSemantics::AcyclicScm);
        assert!(mechanisms_only(plan), "{plan:?}");
    }
    let report = prepared.estimate(&ctx()).unwrap();
    let CyclicFamilyAnswer::RecoveredLinear(answer) = &report.answer else {
        panic!("recovered linear answer");
    };
    for (world, clamp) in answer.worlds.iter().zip(clamps) {
        assert_eq!(world.draws.len(), draws.len());
        close(world.withheld_mass, 0.0, 0.0);
        for (evaluated, k) in world.draws.iter().zip(&draws) {
            let (expected_mean, expected_covariance) =
                dag_law(k, &INTERCEPT, &NOISE_MEAN, &NOISE_COVARIANCE, clamp);
            match &evaluated.outcome {
                RecoveredDrawAnswer::Equilibrium {
                    mean,
                    covariance,
                    solved_by: RecoveredSolvedBy::Acyclic,
                    ..
                } => {
                    for (actual, expected) in mean.iter().zip(&expected_mean) {
                        close(*actual, *expected, TOLERANCE);
                    }
                    for (actual, expected) in covariance.iter().zip(&expected_covariance) {
                        close(*actual, *expected, TOLERANCE);
                    }
                }
                other => panic!("an acyclic draw was {other:?}"),
            }
        }
    }
}

fn dag_equation(id: &str, intercept: f64, weights: Vec<f64>) -> Equation {
    Equation {
        id: id.into(),
        intercept,
        weights,
        activation: Activation::Identity,
        expression: None,
        parameters: vec![],
    }
}

#[test]
fn a_specified_model_with_dag_structure_compiles_to_mechanism_blocks() {
    let draws = triangular_draws();
    let k = &draws[0];
    let model = EquilibriumModel {
        equations: (0..3)
            .map(|i| dag_equation(["a", "b", "c"][i], INTERCEPT[i], k[i].clone()))
            .collect(),
        noise_covariance: NOISE_COVARIANCE.to_vec(),
    };
    let noise = vec![NOISE_MEAN.to_vec(), vec![0.3, -0.2, 0.1]];
    let request = CyclicExecutionRequest {
        model,
        fit: None,
        worlds: vec![vec![], vec![Clamp { variable: 0, value: 2.0 }]],
        noise: noise.clone(),
        initial: vec![0.0; 3],
        policy: SolvePolicy::default(),
        factual: None,
    };
    let prepared = Study::prepare_cyclic(request, &ctx()).unwrap();
    let inspection = prepared.inspect().unwrap();
    for plan in &inspection.execution_plans {
        // The declared semantics stay equilibrium; the schedule is the DAG's.
        assert_eq!(plan.semantics, SystemSemantics::EquilibriumScm);
        assert!(mechanisms_only(plan), "{plan:?}");
    }
    for blocks in &inspection.solve_blocks {
        assert!(blocks.iter().all(|block| block.len() == 1), "{blocks:?}");
    }
    let report = prepared.estimate(&ctx()).unwrap();
    for (world, clamp) in report.worlds.iter().zip([None, Some((0_usize, 2.0))]) {
        for (row, u) in world.iter().zip(&noise) {
            // A fixed disturbance row is a point law: zero covariance, mean `c + u`.
            let (expected, _) = dag_law(k, &INTERCEPT, u, &[0.0; 9], clamp);
            for (actual, expected) in row.values.iter().zip(&expected) {
                close(*actual, *expected, 1e-9);
            }
        }
    }

    // The same structure as a specified linear family takes the DAG schedule too.
    let linear = CyclicFamilyRequest::LinearWorlds {
        variables: vec![0, 1, 2],
        model: LinearCyclicWire {
            coefficients: k.iter().flatten().copied().collect(),
            intercept: INTERCEPT.to_vec(),
            noise_mean: NOISE_MEAN.to_vec(),
            noise_covariance: NOISE_COVARIANCE.to_vec(),
        },
        component: CyclicComponentWire {
            component_id: "chain".into(),
            source_id: "specification".into(),
            certifying_evidence: vec!["independent-bench-test".into()],
        },
        worlds: vec![
            CyclicWorldWire { world: 0, interventions: vec![], initial_states: vec![] },
            CyclicWorldWire {
                world: 1,
                interventions: vec![CyclicInterventionWire::Set { variable: 0, value: 2.0 }],
                initial_states: vec![],
            },
        ],
        certificate: CertificateDemandWire::Required,
    };
    let prepared = Study::prepare_cyclic_family(&linear, &ctx()).unwrap();
    for plan in &prepared.inspect().execution_plans {
        assert_eq!(plan.semantics, SystemSemantics::AcyclicScm);
        assert!(mechanisms_only(plan), "{plan:?}");
    }
    let report = prepared.estimate(&ctx()).unwrap();
    let CyclicFamilyAnswer::Worlds(worlds) = &report.answer else { panic!("typed worlds") };
    for (answer, clamp) in worlds.iter().zip([None, Some((0_usize, 2.0))]) {
        assert_eq!(answer.solved_by.method, "acyclic_order");
        let CyclicWorldOutcomeAnswer::GaussianLaw { mean, covariance, .. } = &answer.outcome else {
            panic!("world {} was {:?}", answer.world, answer.outcome);
        };
        let (expected_mean, expected_covariance) =
            dag_law(k, &INTERCEPT, &NOISE_MEAN, &NOISE_COVARIANCE, clamp);
        for (actual, expected) in mean.iter().zip(&expected_mean) {
            close(*actual, *expected, TOLERANCE);
        }
        for (actual, expected) in covariance.iter().zip(&expected_covariance) {
            close(*actual, *expected, TOLERANCE);
        }
    }
}

/// A four-node law with `A -> B` and `C -> D`; `feedback` adds `D -> C`, a cycle
/// that does not reach the outcome `B`. Two draws under one declared origin.
fn four_node_law(feedback: bool) -> CyclicFamilyRequest {
    let draw = |scale: f64| {
        let mut coefficients = vec![
            RecoveredCoefficientWire { into: 1, from: 0, value: 0.5 * scale },
            RecoveredCoefficientWire { into: 3, from: 2, value: 0.3 },
        ];
        let feedback_value = if feedback { 0.6 } else { 0.0 };
        coefficients.push(RecoveredCoefficientWire { into: 2, from: 3, value: feedback_value });
        // Every other off-diagonal pair is a stated structural zero.
        for into in 0..4_u32 {
            for from in 0..4_u32 {
                let named = coefficients.iter().any(|c| c.into == into && c.from == from);
                if into != from && !named {
                    coefficients.push(RecoveredCoefficientWire { into, from, value: 0.0 });
                }
            }
        }
        RecoveredDrawWire {
            coefficients,
            intercept: vec![0.0; 4],
            noise_mean: vec![1.0, 0.0, 0.0, 0.0],
            noise_covariance: (0..16).map(|i| if i % 5 == 0 { 1.0 } else { 0.0 }).collect(),
            unresolved: Vec::new(),
            weight: 0.5,
            marked_inadmissible: false,
        }
    };
    CyclicFamilyRequest::RecoveredLinear {
        nodes: vec![0, 1, 2, 3],
        draws: vec![draw(1.0), draw(0.8)],
        worlds: vec![
            RecoveredInterventionWire { clamp_nodes: vec![], clamp_values: vec![] },
            RecoveredInterventionWire { clamp_nodes: vec![0], clamp_values: vec![2.0] },
        ],
        outcome: 1,
        source_id: "clamp-experiments".into(),
        certifying_evidence: vec!["held-out-regime".into()],
        uncertainty: RecoveredDrawOriginWire::AsymptoticNormal { draws: 2, seed: 0 },
    }
}

fn schema() -> CyclicEvidenceSchema {
    CyclicEvidenceSchema {
        id: "four-node".into(),
        population: "bench".into(),
        variables: ["a", "b", "c", "d"].map(|name| (name.to_owned(), "volt".to_owned())).to_vec(),
    }
}

fn selectors() -> Vec<S> {
    let mut selectors = Vec::new();
    for world in 0..2 {
        selectors.push(S::Response { world, noise_row: 0, variable: 1 });
        selectors.push(S::RecoveredDraw { world, draw: 0, variable: 1 });
    }
    selectors
}

#[test]
fn recovered_evidence_on_acyclic_support_equals_the_cyclic_case_except_the_solve_route() {
    let acyclic = Study::prepare_recovered_linear(&four_node_law(false), &ctx()).unwrap();
    let cyclic = Study::prepare_recovered_linear(&four_node_law(true), &ctx()).unwrap();
    for plan in &acyclic.inspect().execution_plans {
        assert_eq!(plan.semantics, SystemSemantics::AcyclicScm);
        assert!(mechanisms_only(plan), "{plan:?}");
    }
    for plan in &cyclic.inspect().execution_plans {
        assert_eq!(plan.semantics, SystemSemantics::EquilibriumScm);
        let joint = |block: &ExecutionBlock| match block {
            ExecutionBlock::SimultaneousEquations { variables, .. } => {
                variables.len() == 2 && variables.contains(&2) && variables.contains(&3)
            }
            _ => false,
        };
        assert!(plan.blocks.iter().any(joint), "{plan:?}");
    }
    let from_dag = acyclic.cyclic_evidence(schema(), selectors(), &ctx()).unwrap();
    let from_cycle = cyclic.cyclic_evidence(schema(), selectors(), &ctx()).unwrap();
    let (dag, cycle) = (&from_dag.report().columns, &from_cycle.report().columns);
    assert_eq!(dag.len(), cycle.len());
    for (left, right) in dag.iter().zip(cycle) {
        assert_eq!(left.port, right.port);
        assert_eq!(left.uncertainty_axes, right.uncertainty_axes);
        // The regime binds the whole request, so the two requests have different regime ids.
        assert_ne!(left.quantity.regime_id, right.quantity.regime_id);
        let (mut left_quantity, mut right_quantity) =
            (left.quantity.clone(), right.quantity.clone());
        left_quantity.regime_id.clear();
        right_quantity.regime_id.clear();
        assert_eq!(left_quantity, right_quantity);
        assert_eq!(left.world, right.world);
        assert_eq!(left.time, right.time);
        assert_eq!(left.mass_bounds, right.mass_bounds);
        assert_eq!(left.numerical_enclosures, right.numerical_enclosures);
        assert_eq!(left.samples.len(), right.samples.len());
        for (a, b) in left.samples.iter().zip(&right.samples) {
            close(a.expect("a certified draw"), b.expect("a certified draw"), TOLERANCE);
        }
    }
    // The declared origin sets the response port; one draw is an interventional law.
    assert_eq!(dag[0].port, "sampling_law");
    assert_eq!(dag[1].port, "interventional_law");
    for index in [1, 3] {
        let (left, right) = (&dag[index].diagnostics, &cycle[index].diagnostics);
        close(left["variance"].as_f64().unwrap(), right["variance"].as_f64().unwrap(), TOLERANCE);
        assert_eq!(left["draw"]["outcome"]["solved_by"]["algorithm"], "acyclic");
        assert_eq!(right["draw"]["outcome"]["solved_by"]["algorithm"], "cyclic");
    }
}

/// A Gaussian observational functional over `outcomes`, identified from the chain
/// `0 -> 1 -> 2` by the graph procedure, evaluated at one full assignment.
fn gaussian_functional(outcomes: Vec<usize>) -> CyclicFamilyRequest {
    use antecedent_io::cyclic_graph::{CyclicGraphReport, CyclicGraphRequest, CyclicGraphSpec};
    let source = CyclicGraphRequest::Identify {
        graph: CyclicGraphSpec { nodes: 3, directed: vec![(0, 1), (1, 2)], bidirected: vec![] },
        outcomes,
        treatments: vec![0],
        continuous: true,
        max_steps: 1000,
        io_scm: true,
        scc_product_density: true,
    };
    let identified =
        CyclicFamilyRequest::Graph { request: source.clone() }.execute(&ctx()).unwrap();
    let CyclicFamilyAnswer::Graph(CyclicGraphReport::Identify { program: Some(program), .. }) =
        identified.answer
    else {
        panic!("chain identification")
    };
    CyclicFamilyRequest::GaussianFunctional {
        source,
        source_identity: identified.identity,
        program,
        law: antecedent_io::cyclic_gaussian_functional::GaussianCyclicLaw::Specified {
            mean: vec![0.; 3],
            covariance: vec![2., 1., 1.5, 1., 1.5, 1.55, 1.5, 1.55, 3.01],
        },
        assignments: vec![vec![(0, 1.), (1, 0.), (2, 0.)]],
    }
}

fn functional_schema() -> CyclicEvidenceSchema {
    CyclicEvidenceSchema {
        id: "chain".into(),
        population: "bench".into(),
        variables: vec![
            ("x".into(), "volt".into()),
            ("m".into(), "amp".into()),
            ("y".into(), "ohm".into()),
        ],
    }
}

#[test]
fn a_multi_outcome_functional_projects_a_joint_event_coordinate_and_round_trips() {
    use antecedent::analysis::cyclic_evidence::CyclicEvidence;
    use antecedent_contract::{JointEventKind, JointEventQuantity, ScientificQuantity};
    let selectors = vec![S::Functional { assignment: 0 }];
    let joint = Study::prepare_cyclic_family(&gaussian_functional(vec![1, 2]), &ctx())
        .unwrap()
        .cyclic_evidence(functional_schema(), selectors.clone(), &ctx())
        .unwrap();
    let wire = &joint.columns()[0].quantity;
    assert_eq!(wire.functional_id, "joint_density");
    assert_eq!(wire.units, "1/((amp)*(ohm))");
    let quantity = ScientificQuantity::try_from(wire.clone()).unwrap();
    let schema = functional_schema();
    let event = JointEventQuantity::from_coordinate(&quantity, &schema.id, &schema.variables)
        .expect("a joint event coordinate");
    assert_eq!(event.kind, JointEventKind::Density);
    assert_eq!(event.variables.iter().map(|v| v.name.as_str()).collect::<Vec<_>>(), vec!["m", "y"]);
    // The coordinate survives the artifact encoding.
    let bytes = joint.export(&ctx()).unwrap();
    let loaded = CyclicEvidence::consume(&bytes, joint.identity(), &ctx()).unwrap();
    assert_eq!(loaded.columns()[0].quantity, *wire);

    // A single outcome keeps its scalar coordinate.
    let single = Study::prepare_cyclic_family(&gaussian_functional(vec![2]), &ctx())
        .unwrap()
        .cyclic_evidence(functional_schema(), selectors, &ctx())
        .unwrap();
    let scalar = &single.columns()[0].quantity;
    assert_eq!(scalar.variable_id, "chain:y");
    assert_eq!(scalar.functional_id, "identified_density");
    assert_eq!(scalar.units, "1/(ohm)");
    let scalar = ScientificQuantity::try_from(scalar.clone()).unwrap();
    assert!(JointEventQuantity::from_coordinate(&scalar, &schema.id, &schema.variables).is_none());
}
