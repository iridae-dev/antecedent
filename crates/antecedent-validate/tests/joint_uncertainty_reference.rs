//! Conservative endpoint percentile bootstrap of the joint mechanism range against exhaustive
//! vertex enumeration.
//!
//! The recorded bootstrap draws (the count table the sampler produces for each replicate at the
//! recorded seed, or none for a draw it drops) are input to `oracle_data/joint_uncertainty_reference.py`
//! (frozen in the JSON beside it). For every draw it forms the law, declares a draw with an empty
//! `(w, x)` row failed, and takes the minimum and maximum of the surrogate response over every
//! vertex of the product polytope of independently contaminated kernel rows and parent marginal
//! in exact rational arithmetic, with no closed form. The reference interval is
//! `[q_{(1-level)/2}(minima), q_{(1+level)/2}(maxima)]` with type-7 quantiles over the successful
//! draws, and the baseline interval is the same percentile interval of the zero-deviation
//! responses. The test asserts that the product's sampler reproduces the recorded draws and that
//! the published interval, the baseline interval and the replicate bookkeeping equal the
//! reference for a balanced 200-row table, a 16-row table whose draws sometimes lose a row, a zero
//! box, and an outcome offset of 1e6. It runs only with the `calibration-internal` feature that
//! compiles the closed bootstrap. The sampler's resampling stream and the closed-form range are
//! reviewed in their own records.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    DependenceGroup, DistributionAvailability, Environment, EvidenceCatalog, EvidenceKind,
    EvidenceRegime, ExecutionContext, InterventionAssignment, RegimeBinding, RegimeId, RegimeKind,
    SamplingDesign, Value, VariableCoordinate, VariableDomain, VariableId,
};
use antecedent_estimate::z_transport_bootstrap_law_draws;
use antecedent_expr::{DiscreteAxis, ExactDiscreteLaw, ExactTransportData, LawTolerance};
use antecedent_graph::{Admg, DenseNodeId, SelectionDiagram};
use antecedent_identify::{
    BoundZTransportFunctional, SidLimits, ZTransportQuery, ZTransportResult,
    bind_z_transport_catalog, identify_z_transport,
};
use antecedent_validate::{
    JointDeviationSpec, JointFactor, JointFactorBound,
    joint_sensitivity_bootstrap_interval_internal,
};
use serde_json::Value as Json;

const W: VariableId = VariableId::from_raw(0);
const Z: VariableId = VariableId::from_raw(1);
const X: VariableId = VariableId::from_raw(2);
const Y: VariableId = VariableId::from_raw(3);
const SEED: u64 = 17;

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
                    domain: VariableDomain::Unspecified,
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

fn functional() -> BoundZTransportFunctional {
    let ZTransportResult::Identified(proof) = identify_z_transport(
        &diagram(),
        &query(),
        SidLimits::default(),
        &ExecutionContext::for_tests(0),
    )
    .unwrap() else {
        panic!("the registered surrogate formula identifies");
    };
    bind_z_transport_catalog(&diagram(), &query(), &proof, &catalog()).unwrap()
}

fn table(counts: &[u64], y_values: &[f64]) -> ExactTransportData {
    let levels: Arc<[Value]> =
        (0..2).map(|i| Value::Int64(i64::from(i))).collect::<Vec<_>>().into();
    let total = counts.iter().sum::<u64>() as f64;
    let law = ExactDiscreteLaw::try_empirical(
        "source",
        RegimeId::from_raw(0),
        [antecedent_expr::InterventionAssignment::concrete(Z, Value::Bool(false))],
        [
            DiscreteAxis { variable: W, values: levels },
            DiscreteAxis {
                variable: X,
                values: Arc::from([Value::Bool(false), Value::Bool(true)]),
            },
            DiscreteAxis {
                variable: Y,
                values: y_values.iter().map(|y| Value::Float64(*y)).collect::<Vec<_>>().into(),
            },
        ],
        counts.iter().map(|c| *c as f64 / total).collect::<Vec<_>>(),
        "snapshot-0",
        LawTolerance::default(),
    )
    .unwrap()
    .with_empirical_counts(counts.to_vec())
    .unwrap();
    ExactTransportData::try_new([law], 1 << 20).unwrap()
}

fn spec(kernel: f64, parent: f64) -> JointDeviationSpec {
    JointDeviationSpec::new(vec![
        JointFactorBound { factor: JointFactor::OutcomeKernel, max_fraction: kernel },
        JointFactorBound { factor: JointFactor::SharedParentMarginal, max_fraction: parent },
    ])
}

fn u64s(v: &Json) -> Vec<u64> {
    v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect()
}

fn f64s(v: &Json) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn recorded_draws(case: &Json) -> Vec<Option<Vec<u64>>> {
    case["draws"].as_array().unwrap().iter().map(|d| (!d.is_null()).then(|| u64s(d))).collect()
}

fn product_draws(case: &Json) -> Vec<Option<Vec<u64>>> {
    let data = table(&u64s(&case["counts"]), &f64s(&case["y_values"]));
    z_transport_bootstrap_law_draws(
        &functional(),
        &data,
        case["replicates"].as_u64().unwrap() as u32,
        &ExecutionContext::for_tests(SEED),
    )
    .unwrap()
    .unwrap()
    .into_iter()
    .map(|draw| draw.map(|d| d.laws()[0].empirical_counts().unwrap().to_vec()))
    .collect()
}

#[test]
fn the_endpoint_interval_equals_the_vertex_enumeration_reference() {
    let reference: Json =
        serde_json::from_str(include_str!("oracle_data/joint_uncertainty_reference.json")).unwrap();
    for case in reference["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        assert_eq!(product_draws(case), recorded_draws(case), "{name}: recorded draws");
        let data = table(&u64s(&case["counts"]), &f64s(&case["y_values"]));
        let got = joint_sensitivity_bootstrap_interval_internal(
            &diagram(),
            &functional(),
            &data,
            &spec(
                case["kernel_fraction"].as_f64().unwrap(),
                case["parent_fraction"].as_f64().unwrap(),
            ),
            case["replicates"].as_u64().unwrap() as u32,
            case["level"].as_f64().unwrap(),
            &ExecutionContext::for_tests(SEED),
        )
        .unwrap()
        .unwrap();
        assert_eq!(u64::from(got.replicates_ok), case["replicates_ok"].as_u64().unwrap(), "{name}");
        assert_eq!(u64::from(got.replicates_failed), case["replicates_failed"].as_u64().unwrap());
        // outcome values up to 1e6 with 53-bit arithmetic leave about 1e-9 of absolute slack
        let slack = 1e-9 * f64s(&case["y_values"]).iter().fold(1.0_f64, |a, b| a.max(b.abs()));
        let near = |g: f64, w: f64, what: &str| {
            assert!(
                (g - w).abs() <= 1e-9 + slack * 1e-3 + 1e-9 * w.abs(),
                "{name} {what}: {g} vs {w}"
            );
        };
        near(got.lower, case["lower"].as_f64().unwrap(), "lower");
        near(got.upper, case["upper"].as_f64().unwrap(), "upper");
        let zero = f64s(&case["pointwise_at_zero"]);
        near(got.pointwise_at_zero.0, zero[0], "baseline lower");
        near(got.pointwise_at_zero.1, zero[1], "baseline upper");
        assert!(
            got.lower <= got.pointwise_at_zero.0 + 1e-12
                && got.pointwise_at_zero.1 <= got.upper + 1e-12
        );
    }
}

/// Writes the recorded draws the reference script needs (the first stage of regeneration).
#[test]
#[ignore = "regeneration helper"]
fn emit_draws() {
    let designs: [(&str, [u64; 8], [f64; 2], f64, f64, u32, f64); 4] = [
        ("balanced 200-row table", [30, 20, 15, 35, 25, 25, 10, 40], [0.0, 1.0], 0.2, 0.3, 99, 0.9),
        (
            "16-row table with failed draws",
            [4, 2, 1, 3, 3, 1, 2, 0],
            [0.0, 1.0],
            0.1,
            0.25,
            60,
            0.9,
        ),
        ("zero box", [30, 20, 15, 35, 25, 25, 10, 40], [0.0, 1.0], 0.0, 0.0, 40, 0.9),
        (
            "outcome offset 1e6",
            [30, 20, 15, 35, 25, 25, 10, 40],
            [1e6, 1e6 + 2.0],
            0.15,
            0.2,
            50,
            0.95,
        ),
    ];
    let mut cases = Vec::new();
    for (name, counts, y_values, kernel, parent, replicates, level) in designs {
        let mut case = serde_json::json!({
            "name": name, "counts": counts, "y_values": y_values, "w_levels": 2,
            "kernel_fraction": kernel, "parent_fraction": parent,
            "replicates": replicates, "level": level,
        });
        case["draws"] = serde_json::json!(product_draws(&case));
        cases.push(case);
    }
    std::fs::write(
        std::env::var("JOINT_UNCERTAINTY_DRAWS").unwrap(),
        serde_json::to_string(&serde_json::json!({ "cases": cases })).unwrap(),
    )
    .unwrap();
}
