//! Scenario boundaries for the interference, synthetic-control and mediation routes: a temporal
//! graph, missing outcomes, a third exposure level and declared dependence given to the
//! interference contrast; declared dependence given to the single-treated-unit synthetic control;
//! and treatment levels the mediation data never visited.
//!
//! Each route is first run where it works against an answer written out by hand, then given the
//! declaration it cannot read, and is refused with a typed reason and a next attempt.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{CausalError, InterferenceSpec, RefuteSuite, Study, SyntheticControlQuery};
use antecedent_core::{
    AssignmentDesign, CausalQuery, DeclaredSamplingDesign, ExecutionContext, ExposureLevel,
    ExposureMapping, InterferenceFunctional, InterferenceQuery, Intervention, MediationContrast,
    MediationQuery, Value, VariableId,
};
use antecedent_data::{
    ImputationSpec, MissingDeclaration, MissingPolicy, NetworkData, NetworkEdge, TabularData,
};
use antecedent_graph::{Admg, Cpdag, Dag, DenseNodeId, TemporalDag};

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(3)
}

// ---------------------------------------------------------------------------------------------
// Interference: the exact two-unit Bernoulli design. The four assignments are enumerated by
// hand in conformance/response/randomized_interference: HT -6, Hajek -3, variance bound 204.
// ---------------------------------------------------------------------------------------------

fn contrast(from_own: f64, to_own: f64) -> InterferenceQuery {
    InterferenceQuery::new(
        AssignmentDesign::Bernoulli { probabilities: Arc::from([0.5]) },
        ExposureMapping::NeighborCount,
        InterferenceFunctional::ExposureContrast {
            outcome: v(0),
            from: ExposureLevel { own: from_own, neighbors: 1.0 },
            to: ExposureLevel { own: to_own, neighbors: 0.0 },
        },
    )
}

struct Declared {
    sampling: Option<DeclaredSamplingDesign>,
    missing: Option<MissingDeclaration>,
}

const NONE: Declared = Declared { sampling: None, missing: None };

fn interference(
    outcomes: [f64; 2],
    query: InterferenceQuery,
    declared: Declared,
) -> Result<antecedent::StudyResult, CausalError> {
    let units = TabularData::from_f64_columns([("y", outcomes.as_slice())]).unwrap();
    let network = NetworkData::try_new(
        units.clone(),
        [NetworkEdge::new(0, 1, 1.0), NetworkEdge::new(1, 0, 1.0)],
    )
    .unwrap();
    let mut builder = Study::tabular(units)
        .graph(Dag::with_variables(1))
        .query(CausalQuery::Interference(query))
        .interference(InterferenceSpec { network, assignment: Arc::from([true, false]) })
        .refute(RefuteSuite::None);
    if let Some(design) = declared.sampling {
        builder = builder.sampling_design(design);
    }
    if let Some(missing) = declared.missing {
        builder = builder.missing(missing);
    }
    builder.build()?.run(&ctx())
}

fn assert_exact_enumeration(result: &antecedent::StudyResult) {
    let estimated = result.interference().expect("interference estimate");
    assert!((estimated.contrast.horvitz_thompson + 6.0).abs() < 1e-12);
    assert!((estimated.contrast.hajek + 3.0).abs() < 1e-12);
    assert!((estimated.contrast.conservative_variance - 204.0).abs() < 1e-12);
}

#[test]
fn a_temporal_graph_is_refused_by_the_interference_contrast_which_runs_on_a_static_graph() {
    assert_exact_enumeration(&interference([1.0, 4.0], contrast(0.0, 1.0), NONE).unwrap());
    let units = TabularData::from_f64_columns([("y", [1.0, 4.0].as_slice())]).unwrap();
    let network = NetworkData::try_new(
        units.clone(),
        [NetworkEdge::new(0, 1, 1.0), NetworkEdge::new(1, 0, 1.0)],
    )
    .unwrap();
    let error = Study::tabular(units)
        .graph(TemporalDag::empty())
        .query(CausalQuery::Interference(contrast(0.0, 1.0)))
        .interference(InterferenceSpec { network, assignment: Arc::from([true, false]) })
        .refute(RefuteSuite::None)
        .build()
        .err()
        .expect("a temporal graph was read as a static structure");
    let text = error.to_string();
    assert!(text.contains("typed against static graphs"), "{text}");
    assert!(text.contains("do not denote on a temporal graph class"), "{text}");
}

#[test]
fn a_missing_outcome_or_declared_missing_policy_is_refused_by_the_interference_contrast() {
    // An outcome cell that is not observed is not dropped or imputed: the contrast is refused.
    let text = interference([1.0, f64::NAN], contrast(0.0, 1.0), NONE)
        .err()
        .expect("a missing outcome was averaged over")
        .to_string();
    assert!(text.contains("non-finite outcome"), "{text}");
    // And no missing-data policy is executed: each is refused at build with the policy named.
    for (policy, name) in [
        (MissingPolicy::CompleteCase, "complete_case"),
        (MissingPolicy::MultipleImputation(ImputationSpec::new()), "multiple_imputation"),
    ] {
        let error = interference(
            [1.0, f64::NAN],
            contrast(0.0, 1.0),
            Declared { sampling: None, missing: Some(MissingDeclaration::new(policy)) },
        )
        .err()
        .expect("a missing-data policy was ignored");
        assert_eq!(error.reason_code(), Some("missing_policy_not_applicable"), "{error}");
        let text = error.to_string();
        assert!(text.contains(name), "{text}");
        assert!(text.contains("reads its rows in its own way"), "{text}");
    }
}

#[test]
fn an_exposure_level_beyond_the_binary_assignment_is_refused_not_estimated() {
    assert_exact_enumeration(&interference([1.0, 4.0], contrast(0.0, 1.0), NONE).unwrap());
    // The assignment is one binary indicator per unit, so a third own-treatment level is never
    // realised by any unit and its exposure probability is zero.
    let text = interference([1.0, 4.0], contrast(0.0, 2.0), NONE)
        .err()
        .expect("an unrealisable exposure level was estimated")
        .to_string();
    assert!(text.contains("exposure positivity violation"), "{text}");
}

#[test]
fn a_declared_serial_or_unclustered_design_is_refused_by_the_interference_contrast() {
    for design in [None, Some(DeclaredSamplingDesign::IndependentSubjectTrajectories)] {
        let result = interference(
            [1.0, 4.0],
            contrast(0.0, 1.0),
            Declared { sampling: design, missing: None },
        )
        .unwrap();
        assert_exact_enumeration(&result);
    }
    for design in
        [DeclaredSamplingDesign::DependentTimeSeries, DeclaredSamplingDesign::ClusteredPanel]
    {
        let error = interference(
            [1.0, 4.0],
            contrast(0.0, 1.0),
            Declared { sampling: Some(design), missing: None },
        )
        .err()
        .unwrap_or_else(|| panic!("{design:?}: the declared design was silently ignored"));
        assert_eq!(error.reason_code(), Some("option_not_applicable"), "{design:?}: {error}");
        let text = error.to_string();
        assert!(text.contains("would be silently ignored"), "{design:?}: {text}");
        assert!(text.contains("Next attempt"), "{design:?}: {text}");
    }
}

#[test]
fn a_clustered_panel_is_read_through_a_clustered_assignment_design() {
    // Eight two-unit clusters, four treated; the cluster labels come from the assignment design.
    let clusters: Vec<u32> = (0..8_u32).flat_map(|c| [c; 2]).collect();
    let assignment: Vec<bool> = (0..8_u32).flat_map(|c| [c < 4; 2]).collect();
    let edges: Vec<NetworkEdge> = (0..8_u32)
        .flat_map(|c| {
            let first = u64::from(c) * 2;
            [NetworkEdge::new(first, first + 1, 1.0), NetworkEdge::new(first + 1, first, 1.0)]
        })
        .collect();
    let outcomes: Vec<f64> = (0..16)
        .map(|unit| 0.5 * f64::from((unit / 2 * 3) % 8) + if unit / 2 < 4 { 2.0 } else { 0.0 })
        .collect();
    let units = TabularData::from_f64_columns([("y", outcomes.as_slice())]).unwrap();
    let query = InterferenceQuery::new(
        AssignmentDesign::ClusterRandomization {
            clusters: Arc::from(clusters),
            treated_clusters: 4,
        },
        ExposureMapping::NeighborFraction,
        InterferenceFunctional::ExposureContrast {
            outcome: v(0),
            from: ExposureLevel { own: 0.0, neighbors: 0.0 },
            to: ExposureLevel { own: 1.0, neighbors: 1.0 },
        },
    );
    let run = |design: Option<DeclaredSamplingDesign>| {
        let network = NetworkData::try_new(units.clone(), edges.clone()).unwrap();
        let mut builder = Study::tabular(units.clone())
            .graph(Dag::with_variables(1))
            .query(CausalQuery::Interference(query.clone()))
            .interference(InterferenceSpec { network, assignment: Arc::from(assignment.clone()) })
            .refute(RefuteSuite::None);
        if let Some(design) = design {
            builder = builder.sampling_design(design);
        }
        builder.build().and_then(|study| study.run(&ctx()))
    };
    let undeclared = run(None).unwrap();
    let declared = run(Some(DeclaredSamplingDesign::ClusteredPanel)).unwrap();
    assert_eq!(declared.effect().to_bits(), undeclared.effect().to_bits());
    let error = run(Some(DeclaredSamplingDesign::DependentTimeSeries)).err().unwrap();
    assert_eq!(error.reason_code(), Some("option_not_applicable"), "{error}");
}

// ---------------------------------------------------------------------------------------------
// Synthetic control: donors a, b, c and a treated unit equal to 0.2 a + 0.3 b + 0.5 c plus an
// effect of 5 in the last two periods, so the effect at the intervention period is 5.
// ---------------------------------------------------------------------------------------------

fn synthetic() -> (TabularData, SyntheticControlQuery) {
    let donors = [
        ("a", [0.0, 0.0, 1.0, 0.0, 1.0, 0.0]),
        ("b", [0.0, 1.0, 0.0, 1.0, 2.0, 2.0]),
        ("c", [1.0, 0.0, 0.0, 1.0, 0.0, 2.0]),
    ];
    let (mut y, mut units, mut periods) = (vec![], vec![], vec![]);
    for (unit, trajectory) in donors {
        for (index, value) in trajectory.into_iter().enumerate() {
            units.push(Arc::<str>::from(unit));
            periods.push(index as i64 + 1);
            y.push(value);
        }
    }
    for index in 0..6 {
        units.push(Arc::<str>::from("treated"));
        periods.push(index as i64 + 1);
        let counterfactual =
            0.2 * donors[0].1[index] + 0.3 * donors[1].1[index] + 0.5 * donors[2].1[index];
        y.push(counterfactual + if index >= 4 { 5.0 } else { 0.0 });
    }
    let data = TabularData::from_f64_columns([("outcome", y.as_slice())]).unwrap();
    let query = SyntheticControlQuery::new(v(0), units, periods, Arc::<str>::from("treated"), 5);
    (data, query)
}

#[test]
fn a_single_treated_unit_synthetic_control_honours_a_clustered_design_and_refuses_a_serial_one_it_cannot_test()
 {
    let (data, query) = synthetic();
    let run = |design: Option<DeclaredSamplingDesign>| {
        let mut builder = Study::tabular(data.clone()).query(query.clone());
        if let Some(design) = design {
            builder = builder.sampling_design(design);
        }
        builder.build().and_then(|study| study.run(&ctx()))
    };
    for design in [None, Some(DeclaredSamplingDesign::IndependentSubjectTrajectories)] {
        let effect = run(design).unwrap().synthetic_control().unwrap().effect;
        assert!((effect - 5.0).abs() < 1e-5, "{design:?}: {effect}");
    }
    // A clustered panel reports the placebo variance of the donors, read over the whole units:
    // the effect at the intervention period is the same 5, and the interval is the placebo one.
    let clustered = run(Some(DeclaredSamplingDesign::ClusteredPanel)).unwrap();
    let overall = clustered.quasi_rollout().unwrap().overall().unwrap().clone();
    assert!((overall.estimate - 5.0).abs() < 1e-5, "{}", overall.estimate);
    assert!(overall.standard_error.is_some());
    // Four pre-periods cannot reach the 5% conformal level (1 / (4 + 1) = 0.2), so a serial design
    // is refused with the next attempt instead of reporting an interval of another level.
    let error = run(Some(DeclaredSamplingDesign::DependentTimeSeries))
        .err()
        .expect("a serial design on four pre-periods was silently ignored");
    assert_eq!(error.reason_code(), Some("option_not_applicable"), "{error}");
    let text = error.to_string();
    assert!(text.contains("pre-periods"), "{text}");
    assert!(text.to_lowercase().contains("next attempt"), "{text}");
}

// ---------------------------------------------------------------------------------------------
// Mediation on the counted contingency table of mediation_mixed_graph: the treatment only ever
// takes the levels 0 and 1, so a contrast to any other level is a regime the data never visited.
// ---------------------------------------------------------------------------------------------

/// `(t, m, y, count)` of the table.
const CELLS: [(f64, f64, f64, usize); 8] = [
    (0.0, 0.0, 0.0, 30),
    (0.0, 0.0, 1.0, 10),
    (0.0, 1.0, 0.0, 5),
    (0.0, 1.0, 1.0, 15),
    (1.0, 0.0, 0.0, 12),
    (1.0, 0.0, 1.0, 8),
    (1.0, 1.0, 0.0, 20),
    (1.0, 1.0, 1.0, 25),
];

fn table() -> TabularData {
    let (mut t, mut m, mut y) = (Vec::new(), Vec::new(), Vec::new());
    for (tv, mv, yv, count) in CELLS {
        for _ in 0..count {
            t.push(tv);
            m.push(mv);
            y.push(yv);
        }
    }
    TabularData::from_f64_columns([("t", t.as_slice()), ("m", m.as_slice()), ("y", y.as_slice())])
        .unwrap()
}

fn edges() -> [(u32, u32); 3] {
    [(0, 1), (0, 2), (1, 2)]
}

fn dag() -> Dag {
    let mut dag = Dag::with_variables(3);
    for (a, b) in edges() {
        dag.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    dag
}

fn admg() -> Admg {
    let mut admg = Admg::with_variables(3);
    for (a, b) in edges() {
        admg.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    admg
}

fn mediation(
    graph: impl antecedent::IntoGraphInput,
    contrast: MediationContrast,
    levels: (f64, f64),
) -> Result<f64, CausalError> {
    let mut query = MediationQuery::binary(v(0), v(2), [v(1)], contrast);
    query.control = Intervention::set(v(0), Value::f64(levels.0));
    query.active = Intervention::set(v(0), Value::f64(levels.1));
    Study::tabular(table())
        .graph(graph)
        .query(CausalQuery::Mediation(query))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()?
        .run(&ctx())
        .map(|result| result.effect())
}

#[test]
fn a_mediation_level_the_treatment_never_took_is_refused_not_extrapolated() {
    // The counted total effect: 33/65 - 25/60.
    let counted = 33.0 / 65.0 - 25.0 / 60.0;
    assert!(
        (mediation(dag(), MediationContrast::Total, (0.0, 1.0)).unwrap() - counted).abs() < 1e-12
    );
    assert!(
        (mediation(admg(), MediationContrast::Total, (0.0, 1.0)).unwrap() - counted).abs() < 1e-12
    );
    assert!(
        (mediation(Cpdag::from_dag(&dag()), MediationContrast::Total, (0.0, 1.0)).unwrap()
            - counted)
            .abs()
            < 1e-12
    );
    for levels in [(0.0, 2.0), (-1.0, 1.0), (0.0, 50.0)] {
        for contrast in [MediationContrast::Total, MediationContrast::NaturalDirect] {
            let outcomes = [
                ("dag", mediation(dag(), contrast, levels)),
                ("admg", mediation(admg(), contrast, levels)),
                ("cpdag", mediation(Cpdag::from_dag(&dag()), contrast, levels)),
            ];
            for (class, outcome) in outcomes {
                let error = outcome.err().unwrap_or_else(|| {
                    panic!("{class} {contrast:?} {levels:?}: extrapolated an unvisited level")
                });
                let text = error.to_string();
                if error.reason_code() == Some("arm_not_populated") {
                    assert!(
                        text.contains(
                            "is not one of the two levels the binary treatment took (0 and 1)"
                        ),
                        "{text}"
                    );
                    assert!(text.contains("Next attempt"), "{text}");
                } else {
                    // The counted tables of a latent-free graph have no cell for the level.
                    assert!(text.contains("missing probability table entry"), "{class}: {text}");
                }
                if class == "dag" {
                    assert_eq!(error.reason_code(), Some("arm_not_populated"), "{text}");
                }
            }
        }
    }
}
