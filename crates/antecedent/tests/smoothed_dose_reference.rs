//! The smoothed dose-response transport score against a 50-digit reference.
//!
//! `oracle_data/smoothed_dose_reference.py` (frozen in the JSON beside it) fits, for a recorded
//! fold assignment, the logistic source-membership model on the rows of the other folds and the
//! least-squares outcome curve on their source rows, in multiprecision arithmetic; integrates the
//! fitted curve against the Epanechnikov kernel over each target row's window by adaptive
//! quadrature cut at the basis knots; and forms the plug-in mean, the odds-weighted kernel
//! residual augmentation, their sum and the plug-in difference at half the bandwidth. The product
//! cross-fits the same nuisances through its learner interface and integrates with composite
//! Gauss-Legendre rules, so this checks the fold plumbing (each row predicted by the model that
//! excludes its fold), the knot-split quadrature, the score assembly and the odds weights, under
//! hinge knots inside the window, odds weights above ten, and an outcome offset of 1e6.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{ExecutionContext, SmoothedDoseTransportQuery, SmoothingKernel, VariableId};
use antecedent_estimate::{
    DoseBasis, SmoothedDoseInput, SmoothedDoseOptions, TrialSampling, estimate_smoothed_dose,
};
use antecedent_graph::{Admg, DenseNodeId, SelectionDiagram};
use antecedent_identify::TransportIdentifier;
use serde_json::Value;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
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
    }
}

fn close(actual: f64, expected: f64, tol: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= tol * expected.abs().max(1.0),
        "{what}: {actual} vs {expected}"
    );
}

#[test]
fn cross_fitted_smoothed_score_matches_the_multiprecision_reference() {
    let reference: Value =
        serde_json::from_str(include_str!("oracle_data/smoothed_dose_reference.json")).unwrap();
    let mut graph = Admg::with_variables(3);
    for (from, to) in [(0, 2), (1, 2)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    let diagram = SelectionDiagram::try_new(graph, vec![VariableId::from_raw(0)]).unwrap();
    for case in reference["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let rows = input(case);
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
            min_membership_probability: case["min_membership"].as_f64().unwrap(),
            min_local_ess: 2.0,
            min_distinct_doses: 2,
            ..SmoothedDoseOptions::default()
        };
        let ctx = ExecutionContext::for_tests(7);
        let estimate = estimate_smoothed_dose(&id, &query, &rows, &options, &ctx).unwrap();
        // The recorded fold ids are the product's stratified assignment: pin them, and check the
        // definition they satisfy (every fold holds both roles in balanced numbers).
        let folds = &estimate.folds.assignment;
        let pinned: Vec<u16> = case["folds"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| u16::try_from(v.as_u64().unwrap()).unwrap())
            .collect();
        assert_eq!(*folds, pinned, "{name}: fold assignment");
        for fold in 0..3u16 {
            for role in [true, false] {
                let count = (0..folds.len())
                    .filter(|&i| folds[i] == fold && rows.source[i] == role)
                    .count();
                let total = rows.source.iter().filter(|s| **s == role).count();
                assert!(count == total / 3 || count == total.div_ceil(3), "{name}: balance");
            }
        }
        for (point, want) in estimate.grid.iter().zip(case["points"].as_array().unwrap()) {
            let at = format!("{name} dose {}", point.dose);
            assert!(point.quadrature.exact, "{at}: polynomial curve integrates exactly");
            close(point.plug_in, want["plug_in"].as_f64().unwrap(), 1e-8, &format!("{at} plug_in"));
            close(
                point.augmentation,
                want["augmentation"].as_f64().unwrap(),
                1e-6,
                &format!("{at} augmentation"),
            );
            close(
                point.estimate,
                want["estimate"].as_f64().unwrap(),
                1e-8,
                &format!("{at} estimate"),
            );
            close(
                point.smoothing_bias.half_bandwidth_difference,
                want["half_bandwidth_difference"].as_f64().unwrap(),
                1e-6,
                &format!("{at} half-bandwidth difference"),
            );
        }
        close(
            estimate.overlap.max_odds_weight,
            case["max_odds"].as_f64().unwrap(),
            1e-6,
            &format!("{name} largest odds weight"),
        );
        close(
            estimate.overlap.probability_min,
            case["membership_min"].as_f64().unwrap(),
            1e-7,
            &format!("{name} membership minimum"),
        );
    }
}
