//! The long-run-tempered Bayesian temporal response against a posterior written from its definition.
//!
//! `oracle_data/estimate_response_temporal_bayesian_reference.py` (frozen in the JSON beside it)
//! builds each horizon's lag-aligned design from a series, forms the tempering factor kappa of the
//! horizon's dose-grid levels from the residual autoregression's restricted likelihood (dense
//! covariance matrices, a Lyapunov-equation autocovariance, a multi-start optimizer; no routine
//! of the product), the Student-t posterior of every level under rows weighted `1 / kappa`, and,
//! for several horizons, the cross-horizon covariance of the joint Gaussian law from the
//! long-run covariance of the stacked normal-equation scores at the dependence block length.
//! The product's per-horizon kappa (its published diagnostic) and its dependence block length
//! are compared with those values; its Monte Carlo summaries are compared with the exact
//! posterior to derived bounds: the mean to five Monte Carlo standard errors, the retained-draw
//! standard deviation to `8 / sqrt(draws)` relative, an interval end to 3 percent of the
//! posterior standard deviation, and the correlation of two horizons to `0.015`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    Assumption, AssumptionSet, CausalSchemaBuilder, ContinuousDomain, ExecutionContext, GridSpec,
    IdentificationStatus, Intervention, Lag, MeasurementSpec, ResponseFunctional,
    ResponseIdentification, ResponseQuery, ResponseUncertainty, ResponseValue, RoleHint,
    SmallRoleSet, TemporalEffectQuery, TemporalPolicy, TemporalResponseSpec, Value, ValueType,
    VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
    TimeSeriesData, ValidityBitmap,
};
use antecedent_estimate::{
    BayesianGComputationAte, TEMPORAL_BAYESIAN_TEMPERING_DIAGNOSTIC, TemporalIdentification,
    TemporalResponseEstimator,
};
use antecedent_graph::{TemporalDag, ensure_lagged};
use antecedent_identify::TemporalBackdoorIdentifier;
use serde_json::Value as Json;

const X: u32 = 0;
const Y: u32 = 1;
const Z: u32 = 2;
const DRAWS: usize = 200_000;

fn floats(v: &Json) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn series(case: &Json) -> TimeSeriesData {
    let mut b = CausalSchemaBuilder::new();
    for (name, hint) in [
        ("x", RoleHint::TreatmentCandidate),
        ("y", RoleHint::OutcomeCandidate),
        ("z", RoleHint::Context),
    ] {
        b.add_variable(
            name,
            ValueType::Continuous,
            SmallRoleSet::from_hint(hint),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let schema = b.build().unwrap();
    let n = floats(&case["y"]).len();
    let column = |id: u32, key: &str| {
        OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(id),
                Arc::from(floats(&case[key])),
                ValidityBitmap::all_valid(n),
            )
            .unwrap(),
        )
    };
    let storage = OwnedColumnarStorage::try_new(
        schema,
        vec![column(X, "x"), column(Y, "y"), column(Z, "z")],
        None,
        None,
    )
    .unwrap();
    TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap()
}

/// z_s -> x_s, z_{s-1} -> x_s, x_{s-1} -> y_s, x_{s-2} -> y_s, z_{s-1} -> y_s, z_{s-2} -> y_s.
fn graph() -> TemporalDag {
    let mut g = TemporalDag::empty();
    let (x, y, z) = (VariableId::from_raw(X), VariableId::from_raw(Y), VariableId::from_raw(Z));
    let x1 = ensure_lagged(&mut g, x, Lag::from_raw(1)).unwrap();
    let x2 = ensure_lagged(&mut g, x, Lag::from_raw(2)).unwrap();
    let y0 = ensure_lagged(&mut g, y, Lag::CONTEMPORANEOUS).unwrap();
    let z1 = ensure_lagged(&mut g, z, Lag::from_raw(1)).unwrap();
    let z2 = ensure_lagged(&mut g, z, Lag::from_raw(2)).unwrap();
    for (from, to) in [(x1, y0), (x2, y0), (z1, x1), (z2, x1), (z1, y0), (z2, y0)] {
        g.insert_directed(from, to).unwrap();
    }
    g
}

fn identify(
    g: &TemporalDag,
    horizon: u32,
) -> (antecedent_expr::IdentifiedEstimand, antecedent_data::TemporalIndexer) {
    let query = TemporalEffectQuery::pulse(VariableId::from_raw(X), VariableId::from_raw(Y), 1.0)
        .with_horizon_steps(horizon)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_max_history_lag(Some(6));
    let id = TemporalBackdoorIdentifier::new().identify_temporal(g, &query).unwrap();
    (id.result.estimands.first().cloned().expect("identified estimand"), id.indexer)
}

fn horizons_of(query: &Json) -> Vec<u32> {
    query["horizons"].as_array().unwrap().iter().map(|h| h.as_u64().unwrap() as u32).collect()
}

fn mean_curve(grid: &[f64]) -> ResponseFunctional {
    ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(Y),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(X),
            GridSpec::Values(Arc::from(grid.to_vec())),
        ),
    }
}

fn set_level(level: f64) -> ResponseFunctional {
    ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(Y),
        interventions: Arc::from(vec![Intervention::set(
            VariableId::from_raw(X),
            Value::f64(level),
        )]),
    }
}

fn estimate(
    case: &Json,
    query: &Json,
    functional: ResponseFunctional,
    history_lag: u32,
) -> antecedent_core::CausalResponse {
    let g = graph();
    let horizons = horizons_of(query);
    let data = series(case);
    let ids: Vec<_> = horizons.iter().map(|&h| identify(&g, h)).collect();
    let pairs: Vec<_> = ids.iter().map(|(e, i)| (e, i)).collect();
    let spec =
        TemporalResponseSpec::new(horizons, TemporalPolicy::pulse(-1), Some(history_lag)).unwrap();
    let rq = ResponseQuery::new(functional).with_temporal(spec);
    let bayes = BayesianGComputationAte::conjugate()
        .with_n_draws(DRAWS)
        .with_seed(29)
        .with_prior_scale(case["prior_scale"].as_f64().unwrap());
    TemporalResponseEstimator::new()
        .estimate_bayesian(
            &data,
            TemporalIdentification {
                horizons: &pairs,
                status: IdentificationStatus::NonparametricallyIdentified,
                assumptions: AssumptionSet::new(),
            },
            &rq,
            &bayes,
            &ExecutionContext::for_tests(3),
        )
        .unwrap()
}

fn kappas(response: &antecedent_core::CausalResponse) -> Vec<f64> {
    response
        .support
        .diagnostics
        .iter()
        .find(|d| d.id.as_ref() == TEMPORAL_BAYESIAN_TEMPERING_DIAGNOSTIC)
        .expect("tempering diagnostic")
        .values
        .to_vec()
}

fn published_block(response: &antecedent_core::CausalResponse) -> Option<usize> {
    response.assumptions.entries.iter().find_map(|record| {
        let Assumption::ParametricRestriction(restriction) = &record.assumption else {
            return None;
        };
        let text = restriction.description.as_ref();
        let rest = text.split("dependence block length ").nth(1)?;
        rest.split_whitespace().next()?.trim_end_matches(',').parse().ok()
    })
}

fn surface_band(response: &antecedent_core::CausalResponse) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) =
        &response.estimate
    else {
        panic!("expected a point-identified surface");
    };
    let ResponseUncertainty::PointwiseBand { lower, upper, .. } = &response.uncertainty else {
        panic!("expected a pointwise band");
    };
    (mean.to_vec(), lower.to_vec(), upper.to_vec())
}

fn assert_kappas(name: &str, response: &antecedent_core::CausalResponse, query: &Json) {
    let got = kappas(response);
    let horizons = horizons_of(query);
    assert_eq!(got.len(), horizons.len(), "{name}: one tempering factor per requested horizon");
    for (h, kappa) in horizons.iter().zip(&got) {
        let want = query["kappa"][h.to_string()].as_f64().unwrap();
        assert!(
            (kappa / want - 1.0).abs() < 2e-4,
            "{name} horizon {h}: kappa {kappa} vs the definition {want}"
        );
    }
}

/// The cell summaries of a dose-major, horizon-minor surface against the exact posterior.
fn assert_cells(name: &str, response: &antecedent_core::CausalResponse, query: &Json, t_law: bool) {
    let (mean, lower, upper) = surface_band(response);
    let horizons = horizons_of(query);
    let doses = query["grid"].as_array().unwrap().len();
    assert_eq!(mean.len(), doses * horizons.len(), "{name}: dose-major, horizon-minor layout");
    let root = (DRAWS as f64).sqrt();
    for d in 0..doses {
        for (hi, h) in horizons.iter().enumerate() {
            let cell = &query["levels"][h.to_string()][d];
            let (m, sd) = (cell["mean"].as_f64().unwrap(), cell["sd"].as_f64().unwrap());
            let i = d * horizons.len() + hi;
            assert!(
                (mean[i] - m).abs() < 5.0 * sd / root,
                "{name} cell {i}: mean {} vs {m}",
                mean[i]
            );
            let (lo, hi_end) = if t_law {
                (cell["t_lower"].as_f64().unwrap(), cell["t_upper"].as_f64().unwrap())
            } else {
                (m - 1.959_963_984_540_054 * sd, m + 1.959_963_984_540_054 * sd)
            };
            assert!(
                (lower[i] - lo).abs() < 0.03 * sd,
                "{name} cell {i}: lower {} vs {lo}",
                lower[i]
            );
            assert!(
                (upper[i] - hi_end).abs() < 0.03 * sd,
                "{name} cell {i}: upper {} vs {hi_end}",
                upper[i]
            );
        }
    }
}

fn load() -> Vec<Json> {
    serde_json::from_str::<Json>(include_str!(
        "oracle_data/estimate_response_temporal_bayesian_reference.json"
    ))
    .unwrap()
    .as_array()
    .unwrap()
    .clone()
}

#[test]
fn estimate_response_temporal_bayesian_reference_tempering_factor_and_curve_match_the_definition() {
    let main = &load()[0];
    for (key, history) in [("curve_h12", 6), ("curve_h1", 6)] {
        let query = &main["queries"][key];
        let grid = floats(&query["grid"]);
        let response = estimate(main, query, mean_curve(&grid), history);
        // Every horizon's factor is above one on this persistent design, so the tempering is real.
        let got = kappas(&response);
        assert!(got.iter().all(|k| *k > 1.5), "{key}: {got:?}");
        assert_kappas(key, &response, query);
        if key == "curve_h12" {
            // Horizon 2 is the case where the score long-run-variance bound is the smaller term.
            assert_eq!(query["bounded"]["2"], true);
            assert_eq!(query["bounded"]["1"], false);
        }
        // A single horizon is its own Student-t posterior; several horizons are tied into one
        // Gaussian trajectory law with the same mean and covariance.
        assert_cells(key, &response, query, horizons_of(query).len() == 1);
    }
}

#[test]
fn estimate_response_temporal_bayesian_reference_joint_trajectory_dependence_matches_the_definition()
 {
    let main = &load()[0];
    let query = &main["queries"]["level_h12"];
    let response = estimate(main, query, set_level(1.0), 6);
    assert_kappas("level_h12", &response, query);
    assert_cells("level_h12", &response, query, false);
    let ResponseUncertainty::PointwiseBand { draws, .. } = &response.uncertainty else {
        panic!("a pointwise band");
    };
    let draws = draws.as_ref().expect("a joint trajectory retains its draws");
    let (a, b) = (draws.column(0).unwrap(), draws.column(1).unwrap());
    assert_eq!(a.len(), DRAWS);
    let moments = |v: &[f64]| {
        let m = v.iter().sum::<f64>() / v.len() as f64;
        (m, v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64)
    };
    let ((ma, va), (mb, vb)) = (moments(a), moments(b));
    let cov = a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum::<f64>() / (a.len() - 1) as f64;
    let reference = query["level_cross_covariance"]["1,2"][0][0].as_f64().unwrap();
    let sd = |h: &str| query["levels"][h][0]["sd"].as_f64().unwrap();
    let want = reference / (sd("1") * sd("2"));
    assert!(want.abs() > 0.2 && want.abs() < 0.95, "a non-trivial reference correlation {want}");
    let got = cov / (va * vb).sqrt();
    assert!((got - want).abs() < 0.015, "cross-horizon correlation {got} vs the definition {want}");
    // Each horizon keeps its own posterior standard deviation.
    for (h, v) in [("1", va), ("2", vb)] {
        assert!((v.sqrt() / sd(h) - 1.0).abs() < 8.0 / (DRAWS as f64).sqrt(), "sd of horizon {h}");
    }
}

#[test]
fn estimate_response_temporal_bayesian_reference_block_length_reads_the_declared_history_lag() {
    let main = &load()[0];
    for (key, history) in [("curve_h12", 6), ("curve_h12_short_history", 0)] {
        let query = &main["queries"][key];
        let grid = floats(&query["grid"]);
        let response = estimate(main, query, mean_curve(&grid), history);
        let want = query["block"].as_u64().unwrap() as usize;
        assert_eq!(published_block(&response), Some(want), "{key}: dependence block length");
        assert_kappas(key, &response, query);
    }
    // A longer declared history must change the block the dependence is estimated at.
    assert_ne!(
        main["queries"]["curve_h12"]["block"], main["queries"]["curve_h12_short_history"]["block"],
        "the cases must separate the two histories"
    );
}

#[test]
fn estimate_response_temporal_bayesian_reference_too_short_a_horizon_keeps_the_untempered_posterior()
 {
    let tiny = &load()[1];
    let query = &tiny["queries"]["curve_h1"];
    let grid = floats(&query["grid"]);
    let response = estimate(tiny, query, mean_curve(&grid), 6);
    assert_eq!(kappas(&response), vec![1.0]);
    assert!(
        response
            .support
            .warnings
            .iter()
            .any(|w| w.code.as_ref() == "response.temporal_bayesian.tempering_inestimable"),
        "the untempered fit must be disclosed"
    );
    assert_cells("seven_rows", &response, query, true);
}
