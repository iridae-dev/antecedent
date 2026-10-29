//! The trial-to-target IPW contrast under observation weights, whole clusters and one dependent
//! series, against exact rational references.
//!
//! `oracle_data/transport_design_reference.py` (frozen in the JSON beside it) forms the point
//! estimate and the linearized variance of the weighted ratio in exact fractions, and the exact
//! distribution function of the weighted Bayesian-bootstrap draw by the partial-fraction form of
//! a weighted sum of independent exponentials (cross-checked by simulation in the script). Every
//! input is dyadic, so the references are exact for the binary64 values the product reads.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_estimate::transport::{
    trial_to_target_effect, trial_to_target_ipw_hac_se, trial_to_target_ipw_se,
};
use antecedent_estimate::transport_design::{RowDependence, TrialToTargetRows};
use antecedent_estimate::util::DrawSpec;
use antecedent_identify::{
    PopulationFactor, TransportCertificate, TransportFormula, TransportIdentification,
};
use serde_json::Value;

fn identification() -> TransportIdentification {
    TransportIdentification::Transportable {
        formula: TransportFormula::Direct(PopulationFactor {
            regime: None,
            population: Arc::from("source"),
            variables: Arc::from([]),
            conditioned_on: Arc::from([]),
            interventions: Arc::from([]),
        }),
        certificate: TransportCertificate {
            rule: Arc::from("transport.sid.direct"),
            selection_targets: Arc::from([]),
            premises: Arc::from([]),
        },
    }
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn flags(v: &Value) -> Vec<bool> {
    v.as_array().unwrap().iter().map(|x| x.as_bool().unwrap()).collect()
}

fn text(v: &Value) -> f64 {
    v.as_str().unwrap().parse().unwrap()
}

/// Owned columns of one reference case.
struct Owned {
    outcome: Vec<f64>,
    treatment: Vec<bool>,
    trial: Vec<bool>,
    selection: Vec<f64>,
    propensity: Vec<f64>,
    weights: Option<Vec<f64>>,
}

impl Owned {
    fn read(rows: &Value) -> Self {
        Self {
            outcome: floats(&rows["outcome"]),
            treatment: flags(&rows["treatment"]),
            trial: flags(&rows["trial"]),
            selection: floats(&rows["selection"]),
            propensity: floats(&rows["propensity"]),
            weights: (!rows["weights"].is_null()).then(|| floats(&rows["weights"])),
        }
    }

    fn view(&self) -> TrialToTargetRows<'_> {
        TrialToTargetRows {
            outcome: &self.outcome,
            treatment: &self.treatment,
            trial: &self.trial,
            selection_probability: &self.selection,
            treatment_probability: &self.propensity,
            weights: self.weights.as_deref(),
        }
    }

    /// The same table with the rows of weight zero removed.
    fn without_zero_weight_rows(&self) -> Self {
        let keep: Vec<usize> = (0..self.outcome.len())
            .filter(|&i| self.weights.as_ref().is_none_or(|w| w[i] > 0.0))
            .collect();
        let pick = |v: &Vec<f64>| keep.iter().map(|&i| v[i]).collect::<Vec<f64>>();
        Self {
            outcome: pick(&self.outcome),
            treatment: keep.iter().map(|&i| self.treatment[i]).collect(),
            trial: keep.iter().map(|&i| self.trial[i]).collect(),
            selection: pick(&self.selection),
            propensity: pick(&self.propensity),
            weights: self.weights.as_ref().map(|w| keep.iter().map(|&i| w[i]).collect()),
        }
    }
}

fn close(got: f64, want: f64, what: &str) {
    assert!((got - want).abs() <= 1e-12 * want.abs().max(1.0), "{what}: {got} vs {want}");
}

fn cases(key: &str) -> Vec<Value> {
    serde_json::from_str::<Value>(include_str!("oracle_data/transport_design_reference.json"))
        .unwrap()[key]
        .as_array()
        .cloned()
        .unwrap()
}

fn dependence_of(case: &Value, labels: &mut Vec<u32>) -> Option<usize> {
    match case["dependence"]["kind"].as_str().unwrap() {
        "clusters" => {
            *labels = case["dependence"]["labels"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| u32::try_from(v.as_u64().unwrap()).unwrap())
                .collect();
            None
        }
        "series" => Some(usize::try_from(case["dependence"]["lag"].as_u64().unwrap()).unwrap()),
        _ => None,
    }
}

#[test]
fn weighted_clustered_and_serial_point_and_standard_error_match_the_exact_references() {
    for case in &cases("estimate") {
        let name = case["name"].as_str().unwrap();
        let owned = Owned::read(&case["rows"]);
        let rows = owned.view();
        let mut labels = Vec::new();
        let lag = dependence_of(case, &mut labels);
        let dependence = match case["dependence"]["kind"].as_str().unwrap() {
            "clusters" => RowDependence::Clusters(&labels),
            "series" => RowDependence::SeriesBlocks(lag.unwrap()),
            _ => RowDependence::Independent,
        };
        let ipw = rows.ipw(&identification()).unwrap();
        close(ipw, text(&case["ipw"]), &format!("{name} ipw"));
        let se = rows.ipw_se(ipw, dependence).unwrap();
        close(se, text(&case["se"]), &format!("{name} se"));
        // Squaring the error recovers the exact variance, not only its root.
        close(se * se, text(&case["variance"]), &format!("{name} variance"));
    }
}

#[test]
fn unit_weights_reduce_to_the_unweighted_estimators() {
    for case in cases("estimate").iter().filter(|c| c["rows"]["weights"].is_null()) {
        let name = case["name"].as_str().unwrap();
        let owned = Owned::read(&case["rows"]);
        let rows = owned.view();
        let old = trial_to_target_effect(
            &identification(),
            &owned.outcome,
            &owned.treatment,
            &owned.trial,
            &owned.selection,
            &owned.propensity,
            None,
        )
        .unwrap()
        .ipw;
        close(rows.ipw(&identification()).unwrap(), old, &format!("{name} ipw"));
        let independent = trial_to_target_ipw_se(
            &owned.outcome,
            &owned.treatment,
            &owned.trial,
            &owned.selection,
            &owned.propensity,
            old,
        )
        .unwrap();
        close(
            rows.ipw_se(old, RowDependence::Independent).unwrap(),
            independent,
            &format!("{name} independent"),
        );
        for lag in [0, 2, 3] {
            let series = trial_to_target_ipw_hac_se(
                &owned.outcome,
                &owned.treatment,
                &owned.trial,
                &owned.selection,
                &owned.propensity,
                (old, lag),
            )
            .unwrap();
            close(
                rows.ipw_se(old, RowDependence::SeriesBlocks(lag)).unwrap(),
                series,
                &format!("{name} lag {lag}"),
            );
        }
    }
}

#[test]
fn rows_of_weight_zero_are_deleted_and_never_read() {
    let case = cases("estimate")
        .into_iter()
        .find(|c| c["name"] == "independent_zero_weight_rows_are_deleted")
        .unwrap();
    let owned = Owned::read(&case["rows"]);
    // The deleted rows carry a selection probability of 0 and a treatment probability of 1,
    // which would be refused if either row were read.
    assert!(owned.selection.contains(&0.0) && owned.propensity.contains(&1.0));
    let reduced = owned.without_zero_weight_rows();
    let (full, small) = (owned.view(), reduced.view());
    let ipw = full.ipw(&identification()).unwrap();
    close(ipw, small.ipw(&identification()).unwrap(), "ipw");
    close(
        full.ipw_se(ipw, RowDependence::Independent).unwrap(),
        small.ipw_se(ipw, RowDependence::Independent).unwrap(),
        "se",
    );
    let sampling = DrawSpec { draws: 64, seed: 11 };
    let a = full.bayesian_bootstrap(&identification(), RowDependence::Independent, sampling);
    let b = small.bayesian_bootstrap(&identification(), RowDependence::Independent, sampling);
    assert_eq!(a.unwrap(), b.unwrap());
}

#[test]
fn weighted_clustered_and_blocked_bayesian_bootstrap_follows_the_exact_law() {
    let draws = 200_000;
    for case in &cases("bayes") {
        let name = case["name"].as_str().unwrap();
        let owned = Owned::read(&case["rows"]);
        let rows = owned.view();
        let labels: Vec<u32> = case["labels"]
            .as_array()
            .map(|a| a.iter().map(|v| u32::try_from(v.as_u64().unwrap()).unwrap()).collect())
            .unwrap_or_default();
        let dependence = match case["grouping"].as_str().unwrap() {
            "clusters" => RowDependence::Clusters(&labels),
            "blocks" => RowDependence::SeriesBlocks(
                usize::try_from(case["block"].as_u64().unwrap()).unwrap(),
            ),
            _ => RowDependence::Independent,
        };
        let out = rows
            .bayesian_bootstrap(&identification(), dependence, DrawSpec { draws, seed: 20261029 })
            .unwrap();
        let n = draws as f64;
        assert!(out.iter().all(|v| v.is_finite()));
        // Every draw is the trial's weighted mean score for positive weights, so it lies in the
        // convex hull of the group means.
        assert!(
            out.iter().all(|v| *v >= text(&case["min"]) - 1e-9 && *v <= text(&case["max"]) + 1e-9),
            "{name}: draws escape the hull"
        );
        for point in case["cdf"].as_array().unwrap() {
            let (x, p) = (text(&point["x"]), text(&point["p"]));
            let empirical = out.iter().filter(|v| **v <= x).count() as f64 / n;
            let tol = 5.0 * (p * (1.0 - p) / n).sqrt();
            assert!((empirical - p).abs() < tol, "{name} F({x}) = {empirical} vs {p}");
        }
    }
}

#[test]
fn unusable_weights_and_clusters_are_refused() {
    let case = cases("estimate").into_iter().find(|c| c["name"] == "clusters_unbalanced_weighted");
    let case = case.unwrap();
    let base = Owned::read(&case["rows"]);
    let n = base.outcome.len();
    let one = vec![5_u32; n];
    let ipw = 1.0;
    let mut labels = Vec::new();
    dependence_of(&case, &mut labels);

    // A single cluster, or a trial that lives in a single cluster, carries no between-cluster
    // variation.
    let rows = base.view();
    assert!(rows.ipw_se(ipw, RowDependence::Clusters(&one)).is_err());
    let trial_in_one: Vec<u32> =
        (0..n).map(|i| if base.trial[i] { 0 } else { 1 + u32::try_from(i % 3).unwrap() }).collect();
    assert!(rows.ipw_se(ipw, RowDependence::Clusters(&trial_in_one)).is_err());
    assert!(
        rows.bayesian_bootstrap(
            &identification(),
            RowDependence::Clusters(&trial_in_one),
            DrawSpec { draws: 10, seed: 1 }
        )
        .is_err()
    );
    // Labels that do not cover the rows.
    assert!(rows.ipw_se(ipw, RowDependence::Clusters(&labels[..n - 1])).is_err());

    // Negative, non-finite and misaligned weights, and weights that leave no target row.
    for bad in [-1.0, f64::NAN, f64::INFINITY] {
        let mut owned = Owned::read(&case["rows"]);
        owned.weights.as_mut().unwrap()[3] = bad;
        assert!(owned.view().ipw(&identification()).is_err(), "{bad}");
    }
    let mut short = Owned::read(&case["rows"]);
    short.weights.as_mut().unwrap().pop();
    assert!(short.view().ipw(&identification()).is_err());
    let mut no_target = Owned::read(&case["rows"]);
    for i in 0..n {
        if !no_target.trial[i] {
            no_target.weights.as_mut().unwrap()[i] = 0.0;
        }
    }
    assert!(no_target.view().ipw(&identification()).is_err());
    let mut no_trial = Owned::read(&case["rows"]);
    for i in 0..n {
        if no_trial.trial[i] {
            no_trial.weights.as_mut().unwrap()[i] = 0.0;
        }
    }
    assert!(no_trial.view().ipw(&identification()).is_err());
}
