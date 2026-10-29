//! The smoothed dose-response transport score under declared sampling weights, clusters and serial
//! lags against a 50-digit reference.
//!
//! `oracle_data/smoothed_dose_declared_reference.py` (frozen in the JSON beside it) fits, for a
//! stated fold assignment, the weighted logistic source-membership model and the weighted
//! least-squares outcome curve on the rows of the other folds, integrates the fitted curve against
//! the Epanechnikov kernel over each target row's window by adaptive quadrature cut at the basis
//! knots, and forms the weighted plug-in mean, the weighted odds-weighted augmentation, their sum,
//! the Kish size of the weighted kernel weights and the influence-function standard error of the
//! row scores under independence, one-way clusters and a Newey-West lag. Serial folds are the
//! contiguous blocks `floor(i * folds / n)`; cluster folds are the product's recorded assignment,
//! whose definition (every cluster whole in one fold, both roles in every fold) is checked here.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{ExecutionContext, SmoothedDoseTransportQuery, SmoothingKernel, VariableId};
use antecedent_estimate::smoothed_dose::{
    smoothed_dose_bootstrap_unit_rows, smoothed_dose_bootstrap_units, smoothed_dose_fold_assignment,
};
use antecedent_estimate::{
    DoseBasis, SmoothedDoseInput, SmoothedDoseOptions, TrialSampling, estimate_smoothed_dose,
    validate_smoothed_dose,
};
use antecedent_graph::{Admg, DenseNodeId, SelectionDiagram};
use antecedent_identify::{TransportIdentification, TransportIdentifier};
use serde_json::Value;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn reference() -> Vec<Value> {
    serde_json::from_str::<Value>(include_str!("oracle_data/smoothed_dose_declared_reference.json"))
        .unwrap()["cases"]
        .as_array()
        .cloned()
        .unwrap()
}

fn input(case: &Value) -> SmoothedDoseInput {
    SmoothedDoseInput {
        features: vec![0],
        covariates: vec![floats(&case["covariate"])],
        outcome: floats(&case["outcome"]),
        dose: floats(&case["dose"]),
        dose_density: floats(&case["density"]),
        source: case["source"].as_array().unwrap().iter().map(|v| v.as_bool().unwrap()).collect(),
        sampling: TrialSampling::NestedCohort,
        sampling_weights: Some(floats(&case["weights"])),
        cluster: case["groups"]
            .is_array()
            .then(|| floats(&case["groups"]).into_iter().map(|g| g as u32).collect()),
        serial_lag: case["lag"].as_u64().map(|l| u32::try_from(l).unwrap()),
    }
}

fn close(actual: f64, expected: f64, tol: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= tol * expected.abs().max(1.0),
        "{what}: {actual} vs {expected}"
    );
}

fn setup(case: &Value) -> (TransportIdentification, SmoothedDoseTransportQuery, SmoothedDoseOptions) {
    let mut graph = Admg::with_variables(3);
    for (from, to) in [(0, 2), (1, 2)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    let diagram = SelectionDiagram::try_new(graph, vec![VariableId::from_raw(0)]).unwrap();
    let grid = floats(&case["grid"]);
    let query = SmoothedDoseTransportQuery {
        outcome: VariableId::from_raw(2),
        dose: VariableId::from_raw(1),
        source_population: Arc::from("trial"),
        target_population: Arc::from("target"),
        grid: Arc::from(grid.as_slice()),
        bandwidth: case["h"].as_f64().unwrap(),
        kernel: SmoothingKernel::Epanechnikov,
        dose_support: (0.0, 4.0),
        density_provenance: Arc::from("known"),
    };
    let id = TransportIdentifier::new().identify(&diagram, &query.transport_query()).unwrap();
    let options = SmoothedDoseOptions {
        folds: 3,
        basis: DoseBasis {
            degree: u8::try_from(case["degree"].as_u64().unwrap()).unwrap(),
            knots: floats(&case["knots"]),
            interactions: case["interactions"].as_bool().unwrap(),
        },
        min_membership_probability: 0.02,
        min_local_ess: 2.0,
        min_distinct_doses: 2,
        ..SmoothedDoseOptions::default()
    };
    (id, query, options)
}

#[test]
fn weighted_clustered_and_serial_scores_and_standard_errors_match_the_multiprecision_reference() {
    for case in reference() {
        let name = case["name"].as_str().unwrap();
        let rows = input(&case);
        let (id, query, options) = setup(&case);
        let ctx = ExecutionContext::for_tests(7);
        let estimate = estimate_smoothed_dose(&id, &query, &rows, &options, &ctx).expect(name);
        // The fold assignment is the one the reference was computed under.
        let folds: Vec<u64> = estimate.folds.assignment.iter().map(|f| u64::from(*f)).collect();
        let want: Vec<u64> =
            case["folds"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
        assert_eq!(folds, want, "{name}: fold assignment");
        for (point, expected) in estimate.grid.iter().zip(case["points"].as_array().unwrap()) {
            let at = format!("{name} dose {}", point.dose);
            close(point.plug_in, expected["plug_in"].as_f64().unwrap(), 1e-8, &format!("{at} plug_in"));
            close(
                point.augmentation,
                expected["augmentation"].as_f64().unwrap(),
                1e-7,
                &format!("{at} augmentation"),
            );
            close(point.estimate, expected["estimate"].as_f64().unwrap(), 1e-8, &format!("{at} estimate"));
            close(
                point.influence_se_diagnostic,
                expected["se"].as_f64().unwrap(),
                1e-7,
                &format!("{at} influence standard error"),
            );
            close(
                point.support.effective_sample_size,
                expected["kernel_ess"].as_f64().unwrap(),
                1e-9,
                &format!("{at} weighted kernel ess"),
            );
            // The row scores average to the estimate, with and without weights.
            let mean = point.row_scores.iter().sum::<f64>() / point.row_scores.len() as f64;
            close(mean, point.estimate, 1e-12, &format!("{at} row score mean"));
        }
        close(
            estimate.overlap.max_odds_weight,
            case["max_odds"].as_f64().unwrap(),
            1e-6,
            &format!("{name} largest odds weight"),
        );
    }
}

#[test]
fn clusters_stay_whole_in_one_fold_and_serial_folds_are_contiguous_blocks() {
    for case in reference() {
        let name = case["name"].as_str().unwrap();
        let rows = input(&case);
        let folds = smoothed_dose_fold_assignment(&rows, 3);
        assert_eq!(folds.len(), rows.source.len(), "{name}");
        if let Some(groups) = &rows.cluster {
            for (a, ga) in groups.iter().enumerate() {
                for (b, gb) in groups.iter().enumerate() {
                    if ga == gb {
                        assert_eq!(folds[a], folds[b], "{name}: cluster {ga} split across folds");
                    }
                }
            }
        }
        if rows.serial_lag.is_some() {
            let n = rows.source.len();
            for (i, fold) in folds.iter().enumerate() {
                assert_eq!(usize::from(*fold), i * 3 / n, "{name}: block of row {i}");
            }
        }
        for fold in 0..3u16 {
            for role in [true, false] {
                assert!(
                    (0..folds.len()).any(|i| folds[i] == fold && rows.source[i] == role),
                    "{name}: fold {fold} lacks role {role}"
                );
            }
        }
    }
}

#[test]
fn equal_weights_at_any_scale_are_the_unweighted_score_and_plain_rows_are_unchanged() {
    let case = reference().into_iter().find(|c| c["name"] == "equal_weights_scale_invariant").unwrap();
    let weighted = input(&case);
    let (id, query, options) = setup(&case);
    let ctx = ExecutionContext::for_tests(7);
    let mut plain = weighted.clone();
    plain.sampling_weights = None;
    let a = estimate_smoothed_dose(&id, &query, &plain, &options, &ctx).unwrap();
    let b = estimate_smoothed_dose(&id, &query, &weighted, &options, &ctx).unwrap();
    for (p, q) in a.grid.iter().zip(&b.grid) {
        close(q.estimate, p.estimate, 1e-9, "estimate");
        close(q.influence_se_diagnostic, p.influence_se_diagnostic, 1e-9, "influence standard error");
        close(q.support.effective_sample_size, p.support.effective_sample_size, 1e-12, "ess");
    }
    // Declared rows change the identity of the premises only through the rows: the plain input
    // serializes without the new fields, so existing artifacts and digests are untouched.
    let text = serde_json::to_string(&plain).unwrap();
    assert!(!text.contains("sampling_weights") && !text.contains("cluster"), "{text}");
}

#[test]
fn zero_weight_rows_are_deleted_and_invalid_declarations_are_refused_by_name() {
    let case = reference().into_iter().find(|c| c["name"] == "weights_only_extreme_weight").unwrap();
    let (id, query, options) = setup(&case);
    let rows = input(&case);
    let n = rows.source.len();
    // A zero weight is deleted by the helper (and counted); the validator refuses it as given.
    let mut zeroed = rows.clone();
    let mut w = zeroed.sampling_weights.clone().unwrap();
    w[3] = 0.0;
    w[60] = 0.0;
    zeroed.sampling_weights = Some(w);
    let error = validate_smoothed_dose(&id, &query, &zeroed, &options).unwrap_err().to_string();
    assert!(error.contains("sampling_weights.invalid") && error.contains("without_zero_weight_rows"), "{error}");
    let (kept, deleted) = zeroed.without_zero_weight_rows().unwrap();
    assert_eq!((deleted, kept.source.len()), (2, n - 2));
    validate_smoothed_dose(&id, &query, &kept, &options).unwrap();
    // Wrong lengths, non-finite weights, clusters with a lag, a lag outside the rows.
    let refuse = |edit: &dyn Fn(&mut SmoothedDoseInput)| {
        let mut bad = rows.clone();
        edit(&mut bad);
        validate_smoothed_dose(&id, &query, &bad, &options).unwrap_err().to_string()
    };
    assert!(refuse(&|r| r.sampling_weights = Some(vec![1.0; n - 1])).contains("sampling_weights.invalid"));
    assert!(refuse(&|r| r.sampling_weights.as_mut().unwrap()[4] = f64::NAN).contains("sampling_weights.invalid"));
    assert!(refuse(&|r| r.sampling_weights.as_mut().unwrap()[4] = -1.0).contains("sampling_weights.invalid"));
    assert!(refuse(&|r| r.cluster = Some(vec![0; n - 1])).contains("dose_response.invalid_declared_rows"));
    assert!(
        refuse(&|r| {
            r.cluster = Some((0..n as u32).map(|i| i % 6).collect());
            r.serial_lag = Some(2);
        })
        .contains("not offered together")
    );
    for lag in [0u32, u32::try_from(n).unwrap(), u32::try_from(n).unwrap() + 4] {
        assert!(
            refuse(&|r| r.serial_lag = Some(lag)).contains("dose_response.invalid_declared_rows"),
            "lag {lag}"
        );
    }
    // Fewer clusters than folds, or clusters that leave a fold without a role, cannot be cross-fitted.
    assert!(
        refuse(&|r| r.cluster = Some((0..n as u32).map(|i| i % 2).collect()))
            .contains("dose_response.declared_folds_unbalanced")
    );
    assert!(
        refuse(&|r| r.cluster = Some((0..n as u32).map(|i| u32::from(!r.source[i as usize])).collect()))
            .contains("dose_response.declared_folds_unbalanced")
    );
}

#[test]
fn the_bootstrap_resamples_clusters_and_blocks_and_rows_otherwise() {
    let case = reference().into_iter().find(|c| c["name"] == "clusters_weighted").unwrap();
    let ctx = ExecutionContext::for_tests(5);
    let clustered = input(&case);
    let units = smoothed_dose_bootstrap_units(&clustered);
    assert_eq!(units.len(), 1, "a nested cohort is one resampling group");
    assert_eq!(units[0].len(), 12, "twelve clusters");
    for replicate in 0..20 {
        let rows = smoothed_dose_bootstrap_unit_rows(&units, replicate, &ctx).unwrap();
        // Whole clusters enter: every drawn cluster contributes all its rows.
        let groups = clustered.cluster.as_ref().unwrap();
        let mut seen = std::collections::BTreeMap::new();
        for &row in &rows {
            *seen.entry(groups[row]).or_insert(0usize) += 1;
        }
        for (group, count) in seen {
            let size = groups.iter().filter(|g| **g == group).count();
            assert_eq!(count % size, 0, "cluster {group}: {count} rows is not a whole number of clusters");
        }
    }
    let serial = reference().into_iter().find(|c| c["name"] == "serial_lag_2_blocks").unwrap();
    let blocks = smoothed_dose_bootstrap_units(&input(&serial));
    assert!(blocks[0].iter().all(|b| b.len() <= 3) && blocks[0][0] == vec![0, 1, 2]);
    // Without a declaration every row is its own unit: the draw is the historical one, row for row.
    let mut plain = clustered.clone();
    plain.cluster = None;
    plain.sampling_weights = None;
    let plain_units = smoothed_dose_bootstrap_units(&plain);
    assert!(plain_units[0].iter().all(|u| u.len() == 1));
    let groups = antecedent_estimate::smoothed_dose::smoothed_dose_bootstrap_groups(&plain);
    for replicate in 0..5 {
        assert_eq!(
            smoothed_dose_bootstrap_unit_rows(&plain_units, replicate, &ctx).unwrap(),
            antecedent_estimate::learned_trial::bootstrap_rows(&groups, replicate, &ctx).unwrap()
        );
    }
}
