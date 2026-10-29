// SPDX-License-Identifier: MIT OR Apache-2.0
//! The misclassification-corrected temporal design: what it reads and what it still refuses.
//!
//! The corrected unit-period design reads one recorded binary outcome per unit and period, a known
//! assignment law and the validation counts of the recording error. A categorical three-arm
//! design recovers the corrected arm means and contrasts, and its interval is the whole-unit
//! cluster sandwich plus the delta-method term for the validation counts, which absorbs serial
//! dependence within a unit. The design reads three further declarations, each against an exact
//! rational reference (`oracle_data/temporal_design_measurement_reference.py`, frozen beside it):
//! analysis weights on the table (weighted arm means, the weighted unit-cluster sandwich), a
//! missing-data policy (`complete_case`, `available_case`, `balanced_panel`, with exact pair and
//! unit bookkeeping), and a declared dependent time series (a Newey-West variance over the
//! periods). A declaration the design cannot read stays refused with a typed reason.
//!
//! The oracle below is written from the definitions: the Horvitz-Thompson arm mean
//! `mu_a = (1/N) sum 1[A=a] y / p_a`, the Rogan-Gladen correction `(p + sp - 1) / (se + sp - 1)`,
//! the unit-cluster sandwich with the `G / (G - 1)` factor, and the binomial variance of the
//! sensitivity and specificity from the validation counts.

use std::sync::Arc;

use antecedent::{BayesianConfig, InferenceMode};
use antecedent::{CausalError, Study, StudyBuilder, StudyResult};
use antecedent_core::{
    CausalQuery, DeclaredSamplingDesign, ExecutionContext, OutcomeMisclassification,
    TemporalAssignment, TemporalDesignFunctional, TemporalDesignQuery, VariableId,
};
use antecedent_data::{ImputationSpec, MissingDeclaration, MissingPolicy, TabularData};

const UNITS: usize = 60;
const PERIODS: usize = 6;
const SENSITIVITY: f64 = 0.9;
const SPECIFICITY: f64 = 0.85;
const VALIDATION_POSITIVES: f64 = 1000.0;
const VALIDATION_NEGATIVES: f64 = 1000.0;

fn counts() -> OutcomeMisclassification {
    OutcomeMisclassification {
        true_positive: 900,
        false_negative: 100,
        true_negative: 850,
        false_positive: 150,
    }
}

fn labels(values: &[String]) -> Arc<[Arc<str>]> {
    values.iter().map(|value| Arc::<str>::from(value.as_str())).collect()
}

/// Three equally likely arms in a fixed rotation. The recorded outcome persists within a unit
/// (serial dependence) and rises with the arm.
struct Panel {
    units: Vec<String>,
    unit: Vec<usize>,
    periods: Vec<i64>,
    arm: Vec<usize>,
    recorded: Vec<f64>,
}

fn panel() -> Panel {
    let mut result =
        Panel { units: vec![], unit: vec![], periods: vec![], arm: vec![], recorded: vec![] };
    for unit in 0..UNITS {
        for period in 0..PERIODS {
            let arm = (unit + period) % 3;
            let positive = (unit * 7) % 10 < 3 + arm;
            result.units.push(format!("u{unit}"));
            result.unit.push(unit);
            result.periods.push(i64::try_from(period).unwrap());
            result.arm.push(arm);
            result.recorded.push(f64::from(u8::from(positive)));
        }
    }
    result
}

fn query(panel: &Panel, categorical: bool) -> TemporalDesignQuery {
    TemporalDesignQuery {
        outcome: VariableId::from_raw(0),
        units: labels(&panel.units),
        periods: panel.periods.as_slice().into(),
        lag: 0,
        assignment: TemporalAssignment::MultiArm {
            assignment: panel.arm.as_slice().into(),
            probabilities: vec![vec![1.0 / 3.0; 3]; panel.arm.len()].into(),
            arms: labels(&["a0".into(), "a1".into(), "a2".into()]),
            categorical,
            ordered: false,
        },
        functional: TemporalDesignFunctional::Mean,
        min_probability: 0.1,
        outcome_misclassification: Some(counts()),
    }
}

fn table(recorded: &[f64]) -> TabularData {
    TabularData::from_f64_columns([("y", recorded)]).unwrap()
}

fn builder(data: TabularData, query: TemporalDesignQuery) -> StudyBuilder {
    Study::tabular(data).query(CausalQuery::TemporalDesign(query))
}

fn run(builder: StudyBuilder) -> Result<StudyResult, CausalError> {
    builder.build().and_then(|study| study.run(&ExecutionContext::for_tests(3)))
}

/// Corrected estimate and standard error of `sum_a weight_a mu_a` (`mean` when the weights sum to
/// one, a contrast when they sum to zero).
fn oracle(panel: &Panel, weights: [f64; 3], mean: bool, clustered: bool) -> (f64, f64) {
    let n = panel.arm.len() as f64;
    let mut mu = [0.0_f64; 3];
    for (&arm, &y) in panel.arm.iter().zip(&panel.recorded) {
        mu[arm] += 3.0 * y / n;
    }
    let observed: f64 = weights.iter().zip(mu).map(|(w, m)| w * m).sum();
    let denominator = SENSITIVITY + SPECIFICITY - 1.0;
    let corrected =
        if mean { (observed + SPECIFICITY - 1.0) / denominator } else { observed / denominator };
    let influence = |row: usize| -> f64 {
        (0..3)
            .map(|a| {
                weights[a]
                    * (if panel.arm[row] == a { 3.0 * panel.recorded[row] } else { 0.0 } - mu[a])
            })
            .sum()
    };
    let sampling = if clustered {
        let mut per_unit = vec![0.0_f64; UNITS];
        for row in 0..panel.arm.len() {
            per_unit[panel.unit[row]] += influence(row);
        }
        let g = UNITS as f64;
        per_unit.iter().map(|s| s * s).sum::<f64>() * g / (g - 1.0)
    } else {
        let rows = n;
        (0..panel.arm.len()).map(|row| influence(row).powi(2)).sum::<f64>() * rows / (rows - 1.0)
    } / (n * n);
    let (d_sp, d_se) = if mean {
        (
            (SENSITIVITY - observed) / denominator.powi(2),
            -(observed + SPECIFICITY - 1.0) / denominator.powi(2),
        )
    } else {
        (-observed / denominator.powi(2), -observed / denominator.powi(2))
    };
    let validation = d_se.powi(2) * SENSITIVITY * (1.0 - SENSITIVITY) / VALIDATION_POSITIVES
        + d_sp.powi(2) * SPECIFICITY * (1.0 - SPECIFICITY) / VALIDATION_NEGATIVES;
    (corrected, (sampling / denominator.powi(2) + validation).sqrt())
}

fn coordinate(result: &StudyResult, name: &str) -> (f64, f64) {
    let section = result.temporal_design().expect("a temporal design section");
    let index = section.coordinates.iter().position(|c| &**c == name).unwrap();
    (section.estimate[index], section.standard_error[index])
}

const COORDINATES: [(&str, [f64; 3], bool); 5] = [
    ("mean:a0", [1.0, 0.0, 0.0], true),
    ("mean:a1", [0.0, 1.0, 0.0], true),
    ("mean:a2", [0.0, 0.0, 1.0], true),
    ("contrast:a1-a0", [-1.0, 1.0, 0.0], false),
    ("contrast:a2-a0", [-1.0, 0.0, 1.0], false),
];

#[test]
fn unavailable_b_a_categorical_misclassified_design_equals_the_hand_corrected_arm_means() {
    let p = panel();
    let result = run(builder(table(&p.recorded), query(&p, true))).unwrap();
    assert_eq!(&*result.temporal_design().unwrap().design, "categorical");
    for (name, weights, mean) in COORDINATES {
        let (estimate, se) = coordinate(&result, name);
        let (expected, expected_se) = oracle(&p, weights, mean, true);
        assert!((estimate - expected).abs() < 1e-12, "{name}: {estimate} vs {expected}");
        assert!((se - expected_se).abs() < 1e-9, "{name}: se {se} vs {expected_se}");
    }
}

#[test]
fn unavailable_b_the_interval_is_by_whole_unit_and_a_clustered_declaration_reads_the_same_unit() {
    let p = panel();
    let plain = run(builder(table(&p.recorded), query(&p, false))).unwrap();
    // The unit-persistent outcome makes the whole-unit error differ from a row-independent one,
    // so matching the cluster oracle shows the serial dependence within a unit is read.
    let (_, cluster_se) = oracle(&p, [-1.0, 1.0, 0.0], false, true);
    let (_, row_se) = oracle(&p, [-1.0, 1.0, 0.0], false, false);
    assert!((cluster_se / row_se - 1.0).abs() > 0.05, "{cluster_se} vs {row_se}");
    let (_, plain_se) = coordinate(&plain, "contrast:a1-a0");
    assert!((plain_se - cluster_se).abs() < 1e-9, "{plain_se} vs {cluster_se}");
    // Independent subject trajectories and a clustered panel both take the unit as the cluster:
    // that is the rule they declare, so the number is the unit-cluster one.
    for design in [
        DeclaredSamplingDesign::ClusteredPanel,
        DeclaredSamplingDesign::IndependentSubjectTrajectories,
    ] {
        let declared =
            run(builder(table(&p.recorded), query(&p, false)).sampling_design(design)).unwrap();
        for (name, _, _) in COORDINATES {
            assert_eq!(coordinate(&declared, name), coordinate(&plain, name), "{design:?} {name}");
        }
    }
    // A dependent series is a different rule: the variance is over the periods, so the declared
    // design changes the interval and the label says which rule ran.
    let serial = run(builder(table(&p.recorded), query(&p, false))
        .sampling_design(DeclaredSamplingDesign::DependentTimeSeries))
    .unwrap();
    let section = serial.temporal_design().unwrap();
    assert!(
        section.uncertainty.starts_with("newey_west_ratio_normal_over_periods"),
        "{}",
        section.uncertainty
    );
    let (_, serial_se) = coordinate(&serial, "contrast:a1-a0");
    assert!((serial_se / plain_se - 1.0).abs() > 0.01, "{serial_se} vs {plain_se}");
    // Estimates are the same ratio; only the variance rule moves.
    for (name, _, _) in COORDINATES {
        assert_eq!(coordinate(&serial, name).0, coordinate(&plain, name).0, "{name}");
    }
}

fn assert_refusal(error: &CausalError, reason: &str, detail: &str) {
    let text = error.to_string();
    assert!(text.starts_with(&format!("reason={reason}")), "{text}");
    assert!(text.contains(detail), "{text}");
}

fn reference() -> serde_json::Value {
    serde_json::from_str(include_str!("oracle_data/temporal_design_measurement_reference.json"))
        .unwrap()
}

fn column_f64(values: &serde_json::Value) -> Vec<f64> {
    values.as_array().unwrap().iter().map(|v| v.as_f64().unwrap_or(f64::NAN)).collect()
}

/// A reference case as a study: the panel rows, the optional weights, policy and dependence.
fn reference_study(case: &serde_json::Value) -> Result<StudyResult, CausalError> {
    let arms: Vec<usize> = case["arm"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| usize::try_from(v.as_u64().unwrap()).unwrap())
        .collect();
    let units: Vec<String> = case["unit"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| format!("u{}", v.as_u64().unwrap()))
        .collect();
    let periods: Vec<i64> =
        case["period"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect();
    let recorded = column_f64(&case["y"]);
    let query = TemporalDesignQuery {
        outcome: VariableId::from_raw(0),
        units: labels(&units),
        periods: periods.as_slice().into(),
        lag: u32::try_from(case["lag"].as_u64().unwrap()).unwrap(),
        assignment: TemporalAssignment::MultiArm {
            assignment: arms.as_slice().into(),
            probabilities: vec![vec![1.0 / 3.0; 3]; arms.len()].into(),
            arms: labels(&["a0".into(), "a1".into(), "a2".into()]),
            categorical: false,
            ordered: false,
        },
        functional: TemporalDesignFunctional::Mean,
        min_probability: 0.1,
        outcome_misclassification: Some(counts()),
    };
    let mut data = table(&recorded);
    if !case["weights"].is_null() {
        data = data.with_weights(Arc::from(column_f64(&case["weights"]))).unwrap();
    }
    let mut study = builder(data, query);
    if let Some(policy) = case["policy"].as_str() {
        let policy = match policy {
            "complete_case" => MissingPolicy::CompleteCase,
            "available_case" => MissingPolicy::AvailableCase,
            _ => MissingPolicy::BalancedPanel,
        };
        study = study.missing(MissingDeclaration::new(policy));
    }
    if case["serial"].as_bool().unwrap() {
        study = study.sampling_design(DeclaredSamplingDesign::DependentTimeSeries);
    }
    run(study)
}

#[test]
fn unavailable_b_observation_weights_are_read_by_the_weighted_arm_means_and_cluster_sandwich() {
    let reference = reference();
    let mut weighted = 0;
    for case in reference["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let result = reference_study(case).unwrap_or_else(|error| panic!("{name}: {error}"));
        for (coordinate_name, expected) in case["coordinates"].as_object().unwrap() {
            let (estimate, se) = coordinate(&result, coordinate_name);
            let want = (expected[0].as_f64().unwrap(), expected[1].as_f64().unwrap());
            assert!(
                (estimate - want.0).abs() <= 1e-9 * want.0.abs().max(1.0),
                "{name} {coordinate_name}: {estimate} vs {}",
                want.0
            );
            assert!(
                (se - want.1).abs() <= 1e-8 * want.1.abs().max(1e-3),
                "{name} {coordinate_name}: se {se} vs {}",
                want.1
            );
        }
        if !case["weights"].is_null() {
            weighted += 1;
            let note = result
                .diagnostics
                .iter()
                .find(|d| d.code.as_ref() == "estimate.temporal_design.sampling_weights")
                .unwrap_or_else(|| panic!("{name}: no sampling-weight record"));
            let used = case["pairs_used"].as_u64().unwrap();
            let zero = case["zero_weight"].as_u64().unwrap();
            assert!(
                note.message
                    .starts_with(&format!("{used} decisions used, {zero} zero-weight decisions")),
                "{name}: {}",
                note.message
            );
            let kish = case["kish"].as_f64().unwrap();
            assert!(
                note.message.contains(&format!("{kish:.2}")),
                "{name}: {} vs {kish}",
                note.message
            );
        }
    }
    assert!(weighted >= 4);
}

#[test]
fn unavailable_b_a_missing_data_policy_drops_exactly_the_pairs_it_names_and_accounts_for_them() {
    let reference = reference();
    let mut policies = 0;
    for case in reference["cases"].as_array().unwrap() {
        let Some(policy) = case["policy"].as_str() else { continue };
        policies += 1;
        let name = case["name"].as_str().unwrap();
        let result = reference_study(case).unwrap();
        let report = result.missing_data().as_ref().unwrap_or_else(|| panic!("{name}: no report"));
        let accounting = &report.accounting;
        assert_eq!(accounting.policy.name(), policy, "{name}");
        assert!(report.declared, "{name}");
        assert_eq!(accounting.rows_total as u64, case["pairs_formed"].as_u64().unwrap(), "{name}");
        assert_eq!(accounting.rows_kept as u64, case["pairs_kept"].as_u64().unwrap(), "{name}");
        let dropped: Vec<u64> =
            case["dropped"].as_array().unwrap().iter().map(|d| d[0].as_u64().unwrap()).collect();
        assert_eq!(
            accounting.dropped_rows.as_ref(),
            dropped.as_slice(),
            "{name}: dropped decisions"
        );
        for reason in ["missing_value", "unit_incomplete"] {
            let want =
                case["dropped"].as_array().unwrap().iter().filter(|d| d[1] == reason).count();
            let got = accounting
                .reasons
                .iter()
                .find(|(r, _)| r.name() == reason)
                .map_or(0, |(_, count)| *count);
            assert_eq!(got, want, "{name}: {reason}");
        }
        if policy == "balanced_panel" {
            let units = accounting.units.expect("a balanced panel reports its units");
            assert_eq!(units.total as u64, case["units"].as_u64().unwrap(), "{name}");
            assert_eq!(units.dropped as u64, case["units_incomplete"].as_u64().unwrap(), "{name}");
        } else {
            assert!(accounting.units.is_none(), "{name}");
        }
        assert!(report.non_default, "{name}");
    }
    assert!(policies >= 4);
    // The policies differ: at lag one a deleted row also removes the decision made on it.
    let by_name = |wanted: &str| {
        reference["cases"].as_array().unwrap().iter().find(|c| c["name"] == wanted).unwrap().clone()
    };
    assert!(
        by_name("complete_case_lag_one")["pairs_kept"].as_u64()
            < by_name("available_case_lag_one")["pairs_kept"].as_u64()
    );
}

#[test]
fn unavailable_b_serial_dependence_is_a_newey_west_variance_over_the_periods() {
    let reference = reference();
    let mut serial = 0;
    for case in reference["cases"].as_array().unwrap() {
        if !case["serial"].as_bool().unwrap() {
            continue;
        }
        serial += 1;
        let result = reference_study(case).unwrap();
        let section = result.temporal_design().unwrap();
        assert_eq!(
            &*section.uncertainty,
            "newey_west_ratio_normal_over_periods_with_validation_delta"
        );
        // One cluster per period: the periods are the units of the long-run variance.
        let periods: std::collections::HashSet<i64> =
            case["period"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect();
        let lag = case["lag"].as_i64().unwrap();
        let expected = periods.iter().filter(|p| periods.contains(&(**p + lag))).count();
        assert_eq!(section.clusters as usize, expected, "{}", case["name"]);
    }
    assert_eq!(serial, 3);
}

#[test]
fn unavailable_b_weights_and_policies_the_design_cannot_read_stay_refused() {
    let p = panel();
    // A missing outcome with no policy names the first missing cell and the policies to declare.
    let mut recorded = p.recorded.clone();
    recorded[10] = f64::NAN;
    let error = run(builder(table(&recorded), query(&p, false)))
        .err()
        .expect("a missing recorded outcome with no policy is refused, not dropped");
    assert_refusal(&error, "missing_policy_required", "missing_data.policy_required");
    assert!(error.to_string().contains("complete_case"), "{error}");
    // Inverse-probability weighting needs an observation model and multiple imputation a pooled
    // estimator; neither is built for this design.
    for policy in [
        MissingPolicy::InverseProbabilityOfObservation,
        MissingPolicy::MultipleImputation(ImputationSpec::new()),
    ] {
        let error = run(builder(table(&recorded), query(&p, false))
            .missing(MissingDeclaration::new(policy.clone())))
        .err()
        .unwrap_or_else(|| panic!("{policy:?}: accepted"));
        assert_refusal(
            &error,
            "missing_policy_not_applicable",
            "missing_data.policy_not_applicable",
        );
    }
    // A table whose weights leave no decision, or a negative one, is invalid rather than ignored.
    let rows = p.arm.len();
    let zero = table(&p.recorded).with_weights(Arc::from(vec![0.0; rows]));
    assert!(zero.is_err(), "an all-zero weight vector is refused by the table");
    // A dependent series has no posterior: the Bayesian bootstrap reweights independent units.
    let error = run(builder(table(&p.recorded), query(&p, false))
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(50)))
        .sampling_design(DeclaredSamplingDesign::DependentTimeSeries))
    .err()
    .expect("a dependent series has no posterior");
    assert_refusal(&error, "option_not_applicable", "has no posterior");
}

#[test]
fn unavailable_b_the_dependent_series_and_policies_are_refused_by_designs_that_cannot_read_them() {
    let p = panel();
    let mut quantile = query(&p, false);
    quantile.outcome_misclassification = None;
    let TemporalAssignment::MultiArm { .. } = quantile.assignment else { unreachable!() };
    // A quantile functional is a two-arm whole-procedure bootstrap: it reads neither a policy nor
    // a dependent series, and says so.
    let two_arm = TemporalDesignQuery {
        assignment: TemporalAssignment::TwoArm {
            assignment: p.arm.iter().map(|a| *a == 1).collect::<Vec<_>>().into(),
            probability: vec![0.5; p.arm.len()].into(),
        },
        functional: TemporalDesignFunctional::Quantile { level: 0.5 },
        outcome_misclassification: None,
        ..quantile
    };
    let error = run(builder(table(&p.recorded), two_arm.clone())
        .missing(MissingDeclaration::new(MissingPolicy::CompleteCase)))
    .err()
    .expect("the quantile design reads no missing-data policy");
    assert_refusal(&error, "missing_policy_not_applicable", "missing_data.policy_not_applicable");
    let error = run(builder(table(&p.recorded), two_arm)
        .sampling_design(DeclaredSamplingDesign::DependentTimeSeries))
    .err()
    .expect("the quantile design reads no dependent series");
    assert_refusal(&error, "option_not_applicable", "dependent time series");
}
