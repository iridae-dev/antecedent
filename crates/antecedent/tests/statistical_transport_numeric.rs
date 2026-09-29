//! Designated numeric evidence for T6 joint outer bootstrap.
use std::collections::BTreeMap;
use std::sync::Arc;

use antecedent::DenseNodeId;
use antecedent_core::{
    DistributionAvailability, Environment, EvidenceCatalog, EvidenceKind, EvidenceRegime,
    ExecutionContext, RegimeId, RegimeKind, StreamDomain, TargetSampling, Value,
    VariableCoordinate, VariableDomain, VariableId,
};
use antecedent_estimate::{
    EmpiricalTableOptions, RegimeSample, StatisticalTransportInput, evaluate_statistical_transport,
    percentile_interval,
};
use antecedent_expr::Assignment;
use antecedent_graph::{Admg, SelectionDiagram};
use antecedent_identify::{ClassicalTransportQuery, SidLimits};

fn v(i: u32) -> VariableId {
    VariableId::from_raw(i)
}

fn binary_env(identity: &str, vars: &[u32]) -> Environment {
    Environment::try_new(
        identity,
        vars.iter()
            .map(|i| VariableCoordinate {
                variable: v(*i),
                domain: VariableDomain::Binary,
                unit: None,
            })
            .collect::<Vec<_>>(),
        [],
    )
    .unwrap()
}

fn counts_sample(
    population: &str,
    regime: RegimeId,
    snapshot: &str,
    counts: [usize; 4],
) -> RegimeSample {
    let mut x = Vec::new();
    let mut y = Vec::new();
    for (xv, yv, n) in
        [(0.0, 0.0, counts[0]), (0.0, 1.0, counts[1]), (1.0, 0.0, counts[2]), (1.0, 1.0, counts[3])]
    {
        x.extend(std::iter::repeat_n(Some(xv), n));
        y.extend(std::iter::repeat_n(Some(yv), n));
    }
    RegimeSample {
        population: Arc::from(population),
        regime,
        snapshot_identity: Arc::from(snapshot),
        interventions: Arc::from([]),
        columns: BTreeMap::from([(v(0), x), (v(1), y)]),
    }
}

#[test]
fn mixed_supplied_law_is_bit_identical_across_replicates() {
    let mut graph = Admg::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let diagram = SelectionDiagram::try_new(graph, [v(1)]).unwrap();
    let query = ClassicalTransportQuery {
        outcomes: Arc::from([v(1)]),
        treatments: Arc::from([v(0)]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    };
    let ctx = ExecutionContext::for_tests(3);
    let antecedent_identify::ClassicalTransportResult::Identified(proof) =
        antecedent_identify::identify_classical_transport(
            &diagram,
            &query,
            SidLimits::default(),
            &ctx,
        )
        .unwrap()
    else {
        panic!("identified");
    };
    let supplied = antecedent_expr::ExactDiscreteLaw::try_new(
        "target",
        RegimeId::from_raw(0),
        [],
        [
            antecedent_expr::DiscreteAxis {
                variable: v(0),
                values: Arc::from([Value::Int64(0), Value::Int64(1)]),
            },
            antecedent_expr::DiscreteAxis {
                variable: v(1),
                values: Arc::from([Value::Int64(0), Value::Int64(1)]),
            },
        ],
        [0.25, 0.25, 0.25, 0.25],
        "fixed",
        antecedent_expr::LawTolerance::default(),
    )
    .unwrap();
    let catalog = EvidenceCatalog::try_new(
        [binary_env("target", &[0, 1])],
        [EvidenceRegime::try_new(
            RegimeId::from_raw(0),
            RegimeKind::Observational,
            EvidenceKind::Available,
            [],
            [],
            [v(0), v(1)],
            "target",
            DistributionAvailability::Joint,
        )
        .unwrap()],
        [],
        Some(TargetSampling::SuppliedPopulationLaw),
    )
    .unwrap();
    let functional = proof.bind_catalog(&catalog).unwrap();
    let input = StatisticalTransportInput { supplied: vec![supplied.clone()], samples: Vec::new() };
    let first = evaluate_statistical_transport(
        &functional,
        &input,
        Assignment::from_pairs([(v(0), Value::Int64(1))]),
        antecedent_expr::ExactEvaluationLimits::default(),
        &EmpiricalTableOptions { bootstrap_replicates: 12, ..EmpiricalTableOptions::default() },
        &ctx,
    )
    .unwrap();
    let second = evaluate_statistical_transport(
        &functional,
        &input,
        Assignment::from_pairs([(v(0), Value::Int64(1))]),
        antecedent_expr::ExactEvaluationLimits::default(),
        &EmpiricalTableOptions { bootstrap_replicates: 12, ..EmpiricalTableOptions::default() },
        &ctx,
    )
    .unwrap();
    assert_eq!(
        first.distribution.probabilities.as_ref(),
        second.distribution.probabilities.as_ref()
    );
    assert_eq!(supplied.probabilities(), &[0.25, 0.25, 0.25, 0.25]);
    assert_eq!(
        first.uncertainty_reason.as_deref(),
        Some("exact_supplied_law_no_sampling_uncertainty")
    );
}

#[test]
fn holding_target_fixed_narrows_the_joint_bootstrap() {
    let target = counts_sample("target", RegimeId::from_raw(0), "target", [40, 10, 20, 30]);
    let source = counts_sample("source", RegimeId::from_raw(1), "source", [30, 20, 25, 25]);
    let axes = [
        antecedent_expr::DiscreteAxis {
            variable: v(0),
            values: Arc::from([Value::Int64(0), Value::Int64(1)]),
        },
        antecedent_expr::DiscreteAxis {
            variable: v(1),
            values: Arc::from([Value::Int64(0), Value::Int64(1)]),
        },
    ];
    let options = EmpiricalTableOptions::default();
    let target_point =
        antecedent_estimate::fit_empirical_joint(&target, &axes, &options, None).unwrap();
    let mut rng = ExecutionContext::for_tests(11).rng.stream_for(StreamDomain::Test, 3);
    let mut idx = Vec::new();
    let mut honest = Vec::new();
    let mut broken = Vec::new();
    for _ in 0..80 {
        antecedent_data::fill_resample_indexes(
            antecedent_data::ResamplingPlan::IidBootstrap,
            source.n(),
            &mut rng,
            &mut idx,
        )
        .unwrap();
        let source_rep =
            antecedent_estimate::fit_empirical_joint(&source, &axes, &options, Some(&idx)).unwrap();
        let mut target_idx = Vec::new();
        antecedent_data::fill_resample_indexes(
            antecedent_data::ResamplingPlan::IidBootstrap,
            target.n(),
            &mut rng,
            &mut target_idx,
        )
        .unwrap();
        let target_rep =
            antecedent_estimate::fit_empirical_joint(&target, &axes, &options, Some(&target_idx))
                .unwrap();
        honest.push(
            conditional_mean(source_rep.probabilities())
                + conditional_mean(target_rep.probabilities()),
        );
        broken.push(
            conditional_mean(source_rep.probabilities())
                + conditional_mean(target_point.probabilities()),
        );
    }
    let honest_width = interval_width(&honest);
    let broken_width = interval_width(&broken);
    assert!(
        honest_width > broken_width + 1e-9,
        "holding the target sample fixed must understate joint width: honest={honest_width} broken={broken_width}"
    );
}

#[test]
fn independent_leaf_resamples_overstate_precision_against_one_joint() {
    let sample = counts_sample("target", RegimeId::from_raw(0), "s", [25, 5, 10, 20]);
    let joint = antecedent_estimate::fit_empirical_joint(
        &sample,
        &[
            antecedent_expr::DiscreteAxis {
                variable: v(0),
                values: Arc::from([Value::Int64(0), Value::Int64(1)]),
            },
            antecedent_expr::DiscreteAxis {
                variable: v(1),
                values: Arc::from([Value::Int64(0), Value::Int64(1)]),
            },
        ],
        &EmpiricalTableOptions::default(),
        None,
    )
    .unwrap();
    let mut rng =
        antecedent_core::ExecutionContext::for_tests(5).rng.stream_for(StreamDomain::Test, 1);
    let mut idx = Vec::new();
    let mut joint_means = Vec::new();
    let mut split_means = Vec::new();
    for replicate in 0..80u32 {
        antecedent_data::fill_resample_indexes(
            antecedent_data::ResamplingPlan::IidBootstrap,
            sample.n(),
            &mut rng,
            &mut idx,
        )
        .unwrap();
        let fitted = antecedent_estimate::fit_empirical_joint(
            &sample,
            joint.axes(),
            &EmpiricalTableOptions::default(),
            Some(&idx),
        )
        .unwrap();
        joint_means.push(marginal_product(fitted.probabilities()));
        let mut x_idx = Vec::new();
        let mut y_idx = Vec::new();
        let mut rng_x = antecedent_core::ExecutionContext::for_tests(5)
            .rng
            .stream_for(StreamDomain::Test, 100u64 ^ (u64::from(replicate)));
        let mut rng_y = antecedent_core::ExecutionContext::for_tests(5)
            .rng
            .stream_for(StreamDomain::Test, 200u64 ^ (u64::from(replicate)));
        antecedent_data::fill_resample_indexes(
            antecedent_data::ResamplingPlan::IidBootstrap,
            sample.n(),
            &mut rng_x,
            &mut x_idx,
        )
        .unwrap();
        antecedent_data::fill_resample_indexes(
            antecedent_data::ResamplingPlan::IidBootstrap,
            sample.n(),
            &mut rng_y,
            &mut y_idx,
        )
        .unwrap();
        let px = antecedent_estimate::fit_empirical_joint(
            &sample,
            joint.axes(),
            &EmpiricalTableOptions::default(),
            Some(&x_idx),
        )
        .unwrap();
        let py = antecedent_estimate::fit_empirical_joint(
            &sample,
            joint.axes(),
            &EmpiricalTableOptions::default(),
            Some(&y_idx),
        )
        .unwrap();
        split_means.push(px_py_product(px.probabilities(), py.probabilities()));
    }
    let joint_width = interval_width(&joint_means);
    let split_width = interval_width(&split_means);
    assert!(
        split_width + 1e-9 < joint_width,
        "independent leaves must look tighter: split={split_width} joint={joint_width}"
    );
}

#[test]
fn trial_ipw_still_refuses_recursive_factorization() {
    use antecedent_estimate::trial_to_target_effect;
    use antecedent_identify::{
        PopulationFactor, TransportCertificate, TransportFormula, TransportIdentification,
    };
    let err = trial_to_target_effect(
        &TransportIdentification::Transportable {
            formula: TransportFormula::RecursiveFactorization {
                sum_out: Arc::from([]),
                factors: Arc::from([PopulationFactor {
                    regime: None,
                    population: Arc::from("source"),
                    variables: Arc::from([]),
                    conditioned_on: Arc::from([]),
                    interventions: Arc::from([]),
                }]),
            },
            certificate: TransportCertificate {
                rule: Arc::from("transport.sid.recursive"),
                selection_targets: Arc::from([]),
                premises: Arc::from([]),
            },
        },
        &[1.0, 0.0],
        &[true, false],
        &[true, false],
        &[0.5, 0.5],
        &[0.5, 0.5],
        None,
    )
    .unwrap_err();
    assert!(
        err.to_string().contains("recursive")
            || err.to_string().contains("Recursive")
            || err.to_string().contains("certificate")
    );
}

fn interval_width(values: &[f64]) -> f64 {
    let (lo, hi) = percentile_interval(values, 0.95).unwrap();
    hi - lo
}

fn conditional_mean(p: &[f64]) -> f64 {
    let den = p[2] + p[3];
    p[3] / den
}

fn marginal_product(p: &[f64]) -> f64 {
    let treatment_mass = p[2] + p[3];
    let outcome_mass = p[1] + p[3];
    treatment_mass * outcome_mass
}

fn px_py_product(px: &[f64], py: &[f64]) -> f64 {
    (px[2] + px[3]) * (py[1] + py[3])
}
