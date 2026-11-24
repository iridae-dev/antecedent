//! The compiled transport families reproduce the direct calls they replace, and recover the
//! known truths of the existing fixtures.
//!
//! * `Study::compile_transport_scenarios`: the report equals the one
//!   `PreparedScenarioSet::evaluate` returns on the same prepared set, before and after a
//!   refresh; the identified scenarios read the standardized and direct transport formulas
//!   worked by hand from the laws.
//! * `Study::compile_measurement_transport`: the receipt and re-bound model are the ones
//!   `transport_measurement_model` issues, and the corrected point and standard error equal the
//!   direct correction and the inverted misclassification algebra at the carried rates.
//! * `Study::compile_pooled_differential_transport`: the correction, interval and draws equal
//!   `transport_pooled_differential_prevalence` under the same seed, bit for bit.
//! * `Study::compile_transported_cohort`: the transported cohort equals the direct route over
//!   `transport_cohort_effect`, and its mean is the hand-calculated `2.225`.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::float_cmp, reason = "the bits of both paths are compared exactly")]
#![allow(
    deprecated,
    reason = "the twins call the direct facade entries the compiled routes replace"
)]

use std::sync::Arc;

use antecedent::analysis::compiled_measurement_transport::{
    MeasurementTransportRequest, PooledDifferentialTransportRequest,
};
use antecedent::analysis::compiled_transported_cohort::TransportedCohortRequest;
use antecedent::analysis::transported_measurement::{
    ErrorParameter, InvarianceDeclaration, PooledDifferentialRequest, PopulationErrorStructure,
    PopulationValidation, transport_measurement_estimate, transport_measurement_model,
    transport_pooled_differential_prevalence,
};
use antecedent::{MeasurementAnalysisBinding, MeasurementInput, Study, StudyBuilder};
use antecedent_core::measurement_model::{
    ArmBinding, BinaryArmRates, ErrorStructure, MeasurementBinding, MeasurementKind,
    MeasurementModel, MeasurementParameters, MeasurementRoute, MeasurementVariable,
    ParameterEvidence, RepresentationVersion,
};
use antecedent_core::{
    DistributionAvailability, EvidenceCatalog, EvidenceKind, EvidenceRegime, ExecutionContext,
    RegimeId, RegimeKind, SearchLimits, Value, VariableDomain, VariableId,
};
use antecedent_expr::{
    Assignment, DiscreteAxis, ExactDiscreteLaw, ExactEvaluationLimits, ExactTransportData,
    InterventionAssignment, LawTolerance,
};
use antecedent_graph::{Admg, DenseNodeId, SelectionDiagram};
use antecedent_identify::ClassicalTransportQuery;
use antecedent_identify::sid::scenarios::{
    ScenarioCoordinate, TransportScenario, TransportScenarioSet,
};

// ---- scenario sets (fixture of tests/transport_scenarios.rs) ---------------------------------

const Z: u32 = 0;
const X: u32 = 1;
const Y: u32 = 2;
/// Source `do(x = 1)` over `(z, y)`: `P_s(z = 1) = 0.6`, `P_s(y = 1 | z) = 0.2, 0.8`.
const SOURCE: [f64; 4] = [0.32, 0.08, 0.12, 0.48];
/// Target observational `(z, x, y)` with `P*(z = 1) = 0.25`.
const TARGET: [f64; 8] = [0.3, 0.15, 0.2, 0.1, 0.05, 0.05, 0.05, 0.1];
const STANDARDIZED: f64 = 0.75 * 0.2 + 0.25 * 0.8;
const DIRECT: f64 = 0.08 + 0.48;

fn v(i: u32) -> VariableId {
    VariableId::from_raw(i)
}

fn scenario(name: &str, selections: &[u32]) -> TransportScenario {
    let mut graph = Admg::with_variables(3);
    for (a, b) in [(Z, X), (Z, Y), (X, Y)] {
        graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    graph.insert_bidirected(DenseNodeId::from_raw(X), DenseNodeId::from_raw(Y)).unwrap();
    let selections = selections.iter().copied().map(v).collect::<Vec<_>>();
    TransportScenario {
        name: Arc::from(name),
        diagram: SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from(selections)).unwrap(),
        weight: None,
        coordinates: [(Z, "z"), (X, "x"), (Y, "y")]
            .into_iter()
            .map(|(id, name)| ScenarioCoordinate {
                variable: v(id),
                name: Arc::from(name),
                domain: VariableDomain::Binary,
                unit: None,
            })
            .collect(),
    }
}

fn regime(id: u32, population: &str, interventions: &[u32], measured: &[u32]) -> EvidenceRegime {
    EvidenceRegime::try_new(
        RegimeId::from_raw(id),
        if interventions.is_empty() { RegimeKind::Observational } else { RegimeKind::Experimental },
        EvidenceKind::Available,
        interventions.iter().copied().map(v).collect::<Vec<_>>(),
        [],
        measured.iter().copied().map(v).collect::<Vec<_>>(),
        population,
        DistributionAvailability::Joint,
    )
    .unwrap()
}

fn axis(variable: u32) -> DiscreteAxis {
    DiscreteAxis { variable: v(variable), values: Arc::from([Value::f64(0.0), Value::f64(1.0)]) }
}

fn laws(source: [f64; 4]) -> ExactTransportData {
    let law = |population, regime, interventions: Vec<InterventionAssignment>, axes, p: &[f64]| {
        ExactDiscreteLaw::try_new(
            population,
            RegimeId::from_raw(regime),
            interventions,
            axes,
            p.to_vec(),
            population,
            LawTolerance::default(),
        )
        .unwrap()
    };
    ExactTransportData::try_new(
        vec![
            law(
                "source",
                1,
                vec![InterventionAssignment::concrete(v(X), Value::f64(1.0))],
                vec![axis(Z), axis(Y)],
                &source,
            ),
            law("target", 0, Vec::new(), vec![axis(Z), axis(X), axis(Y)], &TARGET),
        ],
        1000,
    )
    .unwrap()
}

fn prepared_scenarios(ctx: &ExecutionContext) -> antecedent::PreparedTransportScenarios {
    let set = TransportScenarioSet::try_new(vec![
        scenario("standardize", &[Z]),
        scenario("direct", &[]),
        scenario("outcome_shift", &[Y]),
    ])
    .unwrap();
    let catalog = EvidenceCatalog::try_new(
        [],
        vec![regime(1, "source", &[X], &[Z, Y]), regime(0, "target", &[], &[Z, X, Y])],
        [],
        None,
    )
    .unwrap();
    StudyBuilder::transport_scenarios(
        &set,
        ClassicalTransportQuery {
            outcomes: Arc::from([v(Y)]),
            treatments: Arc::from([v(X)]),
            source: Arc::from("source"),
            target: Arc::from("target"),
        },
        antecedent::EvidenceSearch {
            catalog,
            budget: SearchLimits { operations: 100_000, depth: 256 },
        },
        laws(SOURCE),
        Assignment::from_pairs([(v(X), Value::f64(1.0))]),
        ExactEvaluationLimits::default(),
        ctx,
    )
    .unwrap()
}

fn mean_of(
    report: &antecedent_estimate::transport_scenarios::ScenarioSetReport,
    name: &str,
) -> f64 {
    let scenario = report.scenarios.iter().find(|s| &*s.name == name).unwrap();
    scenario.distribution.as_ref().unwrap().mean(v(Y)).unwrap()
}

#[test]
fn compiled_scenarios_report_what_the_prepared_set_evaluates() {
    let ctx = ExecutionContext::for_tests(1);
    let prepared = prepared_scenarios(&ctx);
    let compiled = Study::compile_transport_scenarios(prepared.clone()).unwrap();
    assert_eq!(compiled.plan_view().route().as_str(), "TransportScenarioSet");
    let run = compiled.execute(&ctx).unwrap();
    let direct = prepared.prepared().evaluate(&ctx).unwrap();
    assert_eq!(format!("{:?}", run.report), format!("{direct:?}"));
    assert!((mean_of(&run.report, "standardize") - STANDARDIZED).abs() < 1e-12);
    assert!((mean_of(&run.report, "direct") - DIRECT).abs() < 1e-12);
    assert_eq!(run.claim, "per_scenario_points");

    let moved = [0.1, 0.3, 0.3, 0.3];
    let refreshed = compiled.refresh(laws(moved), &ctx).unwrap();
    let direct = prepared.prepared().refresh(laws(moved), &ctx).unwrap().evaluate(&ctx).unwrap();
    let run = refreshed.execute(&ctx).unwrap();
    assert_eq!(format!("{:?}", run.report), format!("{direct:?}"));
    assert!((mean_of(&run.report, "direct") - 0.6).abs() < 1e-12);
    assert_ne!(refreshed.lowered().identity(), compiled.lowered().identity(), "new source");
}

// ---- measurement transport (fixture of tests/transported_measurement_facade.rs) --------------

const BOTH: [ErrorParameter; 2] = [ErrorParameter::Sensitivity, ErrorParameter::Specificity];

fn scalar() -> RepresentationVersion {
    RepresentationVersion { coordinate_system_id: "scalar".into(), version: 1 }
}

fn model(population: &str, se: f64, sp: f64) -> MeasurementModel {
    MeasurementModel {
        model_id: format!("screen-{population}"),
        route: MeasurementRoute::BinaryMisclassification,
        indicator: MeasurementVariable {
            id: "schema:flag".into(),
            kind: MeasurementKind::NoisyMisclassified,
            units: "binary".into(),
        },
        construct: MeasurementVariable {
            id: "construct:disease".into(),
            kind: MeasurementKind::LatentScientific,
            units: "binary".into(),
        },
        binding: MeasurementBinding {
            population_id: population.into(),
            regime_id: "observational".into(),
            arm: ArmBinding::AllArms,
        },
        structure: ErrorStructure::NonDifferential,
        parameters: MeasurementParameters::Binary {
            rates: vec![BinaryArmRates {
                arm_id: None,
                sensitivity: se,
                specificity: sp,
                sensitivity_variance: 1e-4,
                specificity_variance: 4e-4,
            }],
            min_youden: 0.1,
        },
        representation: scalar(),
        evidence: ParameterEvidence::ValidationSample {
            sample_id: format!("val-{population}"),
            units: 500,
        },
    }
}

fn binding(population: &str) -> MeasurementAnalysisBinding {
    MeasurementAnalysisBinding {
        construct_id: "construct:disease".into(),
        population_id: population.into(),
        regime_id: "observational".into(),
        arm_id: None,
        representation: scalar(),
        representation_mapping: None,
        assumed_structure: ErrorStructure::NonDifferential,
    }
}

fn premise() -> PopulationErrorStructure {
    PopulationErrorStructure::Invariant(InvarianceDeclaration::new(
        "clinic_a", "clinic_b", "analyst", &BOTH,
    ))
}

#[test]
fn compiled_measurement_transport_carries_the_kernel_receipt_and_correction() {
    let ctx = ExecutionContext::for_tests(3);
    let models = vec![model("clinic_a", 0.9, 0.8)];
    let input = MeasurementInput::BinaryPrevalence { events: 520, total: 1000 };
    let compiled = Study::compile_measurement_transport(
        MeasurementTransportRequest::new(
            models.clone(),
            Some(premise()),
            binding("clinic_b"),
            input.clone(),
        ),
        &ctx,
    )
    .unwrap();
    assert_eq!(compiled.plan_view().route().as_str(), "MeasurementModelTransport");
    let kernel = transport_measurement_model(
        &models,
        Some(&premise()),
        &antecedent_estimate::measurement::DataBinding {
            population_id: "clinic_b".into(),
            regime_id: "observational".into(),
            arm_id: None,
            representation: scalar(),
        },
    )
    .unwrap();
    assert_eq!(compiled.transported(), &kernel, "the receipt and re-bound model");

    let run = compiled.execute(&ctx).unwrap();
    let direct = transport_measurement_estimate(
        &models,
        Some(&premise()),
        binding("clinic_b"),
        input,
        None,
        &ctx,
    )
    .unwrap();
    assert_eq!(run.point().to_bits(), direct.point().to_bits());
    assert_eq!(run.standard_error().map(f64::to_bits), direct.standard_error().map(f64::to_bits));
    assert_eq!(run.identity_hex(), direct.identity_hex());
    // Rogan-Gladen at the carried rates: (0.52 + 0.8 - 1) / (0.9 + 0.8 - 1).
    assert!((run.point() - 0.32 / 0.7).abs() < 1e-12, "{}", run.point());
}

#[test]
fn compiled_measurement_transport_refuses_without_the_premise() {
    let ctx = ExecutionContext::for_tests(3);
    let refusal = Study::compile_measurement_transport(
        MeasurementTransportRequest::new(
            vec![model("clinic_a", 0.9, 0.8)],
            None,
            binding("clinic_b"),
            MeasurementInput::BinaryPrevalence { events: 520, total: 1000 },
        ),
        &ctx,
    )
    .expect_err("no premise");
    assert_eq!(refusal.detail, "measurement_model.invariance_premise_required");
}

fn validation(population: &str, counts: [u64; 4]) -> PopulationValidation {
    PopulationValidation {
        population: population.into(),
        true_positive: counts[0],
        false_negative: counts[1],
        true_negative: counts[2],
        false_positive: counts[3],
    }
}

#[test]
fn compiled_pooled_differential_transport_equals_the_kernel_under_the_same_seed() {
    let sources =
        vec![validation("north", [180, 20, 170, 30]), validation("south", [164, 36, 172, 28])];
    let target = validation("east", [172, 28, 171, 29]);
    for seed in [3_u64, 4] {
        let ctx = ExecutionContext::for_tests(seed);
        let compiled = Study::compile_pooled_differential_transport(
            PooledDifferentialTransportRequest::new(sources.clone(), "east", 400, 1000, 4000)
                .with_target_validation(target.clone()),
        )
        .unwrap();
        assert_eq!(compiled.plan_view().route().as_str(), "PooledDifferentialTransport");
        let run = compiled.execute(&ctx).unwrap();
        let direct = transport_pooled_differential_prevalence(
            PooledDifferentialRequest {
                sources: &sources,
                target_validation: Some(&target),
                target_population: "east",
                observed_positive: 400,
                observed_total: 1000,
                draws: 4000,
            },
            &ctx,
        )
        .unwrap();
        assert_eq!(run.correction.prevalence.to_bits(), direct.prevalence.to_bits());
        assert_eq!(run.correction.interval.0.to_bits(), direct.interval.0.to_bits());
        assert_eq!(run.correction.interval.1.to_bits(), direct.interval.1.to_bits());
        assert_eq!(run.correction, direct, "rates, draws and counts");
        assert_eq!(run.correction.calibration, "not_measured");
    }
}

// ---- cohort transport (fixture of tests/transported_cohort_facade.rs) ------------------------

mod cohort {
    use antecedent::analysis::cohort_decision::{CohortBinding, CohortColumn};
    use antecedent::analysis::transported_cohort::{
        SelectionShiftDeclaration, transport_cohort_effect_from_result,
    };
    use antecedent::{RefuteSuite, Study};
    use antecedent_core::{AverageEffectQuery, ExecutionContext, VariableId};
    use antecedent_data::TabularData;
    use antecedent_estimate::DrLearner;
    use antecedent_graph::{Dag, DenseNodeId};
    use antecedent_learn::{LearnerSpec, LinearSpec};

    use super::TransportedCohortRequest;

    fn ctx() -> ExecutionContext {
        ExecutionContext::for_tests(97)
    }

    /// Noiseless `Y(0) = 1 + z`, `Y(1) = 3 + 1.5 z`, so `tau(z) = 2 + 0.5 z` exactly.
    fn fit() -> Vec<u8> {
        let (mut t, mut z, mut y) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..40 {
            for treatment in [0.0, 1.0] {
                for confounder in [-1.0, 1.0] {
                    t.push(treatment);
                    z.push(confounder);
                    y.push(1.0 + confounder + treatment * (2.0 + 0.5 * confounder));
                }
            }
        }
        let data = TabularData::from_f64_columns([
            ("t", t.as_slice()),
            ("y", y.as_slice()),
            ("z", z.as_slice()),
        ])
        .unwrap();
        let mut graph = Dag::with_variables(3);
        for (from, to) in [(2, 0), (2, 1), (0, 1)] {
            graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
        }
        let study = Study::tabular(data.clone())
            .graph(graph)
            .query(AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1)))
            .estimator(
                DrLearner::new()
                    .with_outcome(LearnerSpec::Linear(LinearSpec::default()))
                    .with_folds(4),
            )
            .refute(RefuteSuite::None)
            .build()
            .unwrap();
        let prepared = study.prepare(&ctx()).unwrap();
        let result = prepared.estimate(&data, &ctx()).unwrap();
        prepared.encode_contracted_result_with_source(&result, &data, "dr-fit", &ctx()).unwrap()
    }

    fn cohort() -> CohortBinding {
        CohortBinding::new(
            "clinic-b-2026-q1",
            "clinic_b",
            (0..3).map(|i| format!("patient-{i}")).collect(),
            vec!["z".into()],
            vec![CohortColumn::Numeric(vec![-1.0, 0.5, 1.0])],
            vec![0.2, 0.3, 0.5],
        )
        .unwrap()
    }

    fn declaration() -> SelectionShiftDeclaration {
        SelectionShiftDeclaration::new("clinic_a", "clinic_b", "S", &["z"], &["z"])
    }

    #[test]
    fn compiled_cohort_transport_equals_the_direct_route_and_the_hand_mean() {
        let bytes = fit();
        let compiled = Study::compile_transported_cohort(TransportedCohortRequest::new(
            bytes.clone(),
            "clinic_a",
            cohort(),
            Some(declaration()),
        ))
        .unwrap();
        assert_eq!(compiled.plan_view().route().as_str(), "TransportedCohortEffect");
        let run = compiled.execute(&ctx()).unwrap();
        let direct = transport_cohort_effect_from_result(
            &bytes,
            None,
            "clinic_a",
            &cohort(),
            Some(&declaration()),
            &ctx(),
        )
        .unwrap();
        assert_eq!(run, direct);
        assert_eq!(run.mean_contrast().to_bits(), direct.mean_contrast().to_bits());
        // 0.2 * 1.5 + 0.3 * 2.25 + 0.5 * 2.5.
        assert!((run.mean_contrast() - 2.225).abs() < 1e-8, "{}", run.mean_contrast());
    }

    #[test]
    fn compiled_cohort_transport_refuses_without_a_declaration() {
        let compiled = Study::compile_transported_cohort(TransportedCohortRequest::new(
            fit(),
            "clinic_a",
            cohort(),
            None,
        ))
        .unwrap();
        let refusal = compiled.execute(&ctx()).expect_err("no declaration");
        assert_eq!(refusal.detail, "transported_effect.shift_declaration_required");
    }
}
