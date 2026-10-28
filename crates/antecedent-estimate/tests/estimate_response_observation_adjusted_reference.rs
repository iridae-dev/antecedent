//! Selected-outcome correction composed into the Kennedy curve against an exact reference.
//!
//! `oracle_data/estimate_response_observation_adjusted_reference.py` (frozen in the JSON beside it)
//! fits the selection probability by logistic maximum likelihood at 40 digits, clamps it into
//! [0.01, 0.99], and forms the inverse-probability pseudo-value `R y / p` (fitted on every row) or the
//! cross-fitted augmented pseudo-value `m + R (y - m) / p` (row `i` in fold `i mod K`, both nuisances
//! fitted on the other rows, the outcome regression on their selected rows), then passes those
//! pseudo-values as the outcome of the exact Kennedy pipeline and reports its local-quadratic level.
//! Cases: ordinary IPW and augmented selection, a selection model whose fitted probability falls under
//! the floor for 46 of 160 rows, an outcome offset of 1e5 with a treatment offset of 1e3, and two
//! folds. The composition publishes no interval, names its method, and refuses a conditioning set that
//! omits the treatment or an adjuster and simultaneous bands.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, ContinuousDomain, GridSpec, IdentificationStatus, ObservationAssumption,
    ObservationSpec, ResponseFunctional, ResponseIdentification, ResponseQuery,
    ResponseUncertainty, ResponseValue, VariableId,
};
use antecedent_data::TabularData;
use antecedent_estimate::response::ContinuousResponseEstimator;
use antecedent_estimate::{
    ObservationEstimatorOptions, ObservationMechanismEstimator, SelectedOutcomeCorrection,
};
use serde_json::Value;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| {
            x.as_str().map_or_else(
                || x.as_f64().unwrap(),
                |s| if s == "nan" { f64::NAN } else { s.parse().unwrap() },
            )
        })
        .collect()
}

fn query(grid: &[f64], conditioning: &[u32]) -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(Arc::from(grid.to_vec())),
        ),
    })
    .with_observation(
        ObservationSpec::Selected {
            latent: VariableId::from_raw(1),
            observed: VariableId::from_raw(1),
            indicator: VariableId::from_raw(2),
        },
        [ObservationAssumption::OutcomeIndependentGiven(
            conditioning.iter().map(|i| VariableId::from_raw(*i)).collect(),
        )],
    )
}

#[test]
fn adjusted_outcomes_and_the_composed_curve_match_the_exact_reference() {
    let frozen: Value = serde_json::from_str(include_str!(
        "oracle_data/estimate_response_observation_adjusted_reference.json"
    ))
    .unwrap();
    for case in frozen["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let (a, x, y, r) =
            (floats(&case["a"]), floats(&case["x"]), floats(&case["y"]), floats(&case["r"]));
        let data = TabularData::from_f64_columns([
            ("a", a.as_slice()),
            ("y", y.as_slice()),
            ("r", r.as_slice()),
            ("x", x.as_slice()),
        ])
        .unwrap();
        let grid = floats(&case["grid"]);
        let q = query(&grid, &[0, 3]);
        let mode = if case["mode"] == "ipw" {
            SelectedOutcomeCorrection::Ipw
        } else {
            SelectedOutcomeCorrection::Aipw
        };
        let observation = ObservationMechanismEstimator::new(ObservationEstimatorOptions {
            selected_correction: mode,
            crossfit_folds: usize::try_from(case["observation_folds"].as_u64().unwrap()).unwrap(),
            ..ObservationEstimatorOptions::default()
        });
        // The pseudo-values themselves, row by row.
        let adjusted = observation.adjusted_outcome(&data, &q, None).unwrap();
        let spread = y.iter().filter(|v| v.is_finite()).fold(0.0_f64, |m, v| m.max(v.abs()));
        for (i, (got, want)) in adjusted.values.iter().zip(floats(&case["adjusted"])).enumerate() {
            // The logistic fit stops at a score norm near 1e-8 (see the Gaussian review).
            assert!(
                (got - want).abs() <= 2e-6 * (1.0 + want.abs()) + 1e-9 * spread,
                "{name} row {i}: adjusted {got} vs {want}"
            );
        }
        // The composed curve.
        let mut curve_estimator = ContinuousResponseEstimator::new([VariableId::from_raw(3)]);
        curve_estimator.options.folds = usize::try_from(case["folds"].as_u64().unwrap()).unwrap();
        curve_estimator.options.nuisance_basis =
            usize::try_from(case["basis"].as_u64().unwrap()).unwrap();
        curve_estimator.options.nuisance_lambda = case["lambda"].as_f64().unwrap();
        curve_estimator.options.bandwidth = Some(case["bandwidth"].as_f64().unwrap());
        let response = observation
            .estimate_mean_curve(
                &curve_estimator,
                &data,
                &q,
                None,
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
            )
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) =
            &response.estimate
        else {
            panic!("{name}: a surface");
        };
        let tolerance = 1e-4 * (1.0 + spread * 1e-5) + 2e-9 * spread;
        for (g, (got, want)) in mean.iter().zip(floats(&case["level"])).enumerate() {
            assert!((got - want).abs() <= tolerance, "{name} grid {g}: level {got} vs {want}");
        }
        assert!(matches!(response.uncertainty, ResponseUncertainty::None), "{name}");
        assert_eq!(response.provenance_id.as_ref(), "estimate.response.observation_adjusted");
        let method = match mode {
            SelectedOutcomeCorrection::Ipw => "observation.selected.logistic_ipw.v1",
            SelectedOutcomeCorrection::Aipw => "observation.selected.crossfit_logistic_aipw.v1",
        };
        assert!(
            response
                .support
                .warnings
                .iter()
                .any(|w| w.code.as_ref() == "response.observation_adjustment_method"
                    && w.message.as_ref() == method),
            "{name}: the method is named"
        );
        assert!(
            response
                .support
                .warnings
                .iter()
                .any(|w| w.code.as_ref() == "response.observation_joint_uncertainty_unavailable")
        );
    }
}

#[test]
fn an_incomplete_conditioning_set_and_a_requested_band_are_refused() {
    let frozen: Value = serde_json::from_str(include_str!(
        "oracle_data/estimate_response_observation_adjusted_reference.json"
    ))
    .unwrap();
    let case = &frozen["cases"][0];
    let (a, x, y, r) =
        (floats(&case["a"]), floats(&case["x"]), floats(&case["y"]), floats(&case["r"]));
    let data = TabularData::from_f64_columns([
        ("a", a.as_slice()),
        ("y", y.as_slice()),
        ("r", r.as_slice()),
        ("x", x.as_slice()),
    ])
    .unwrap();
    let observation = ObservationMechanismEstimator::default();
    let estimator = ContinuousResponseEstimator::new([VariableId::from_raw(3)]);
    let run = |estimator: &ContinuousResponseEstimator, conditioning: &[u32]| {
        observation.estimate_mean_curve(
            estimator,
            &data,
            &query(&[0.0, 0.1], conditioning),
            None,
            IdentificationStatus::NonparametricallyIdentified,
            AssumptionSet::new(),
        )
    };
    // The treatment and the adjuster must both be conditioned on.
    for conditioning in [&[0_u32][..], &[3][..]] {
        let error = run(&estimator, conditioning).err().expect("incomplete conditioning");
        assert!(error.to_string().contains("OutcomeIndependentGiven"), "{error}");
    }
    let mut banded = estimator.clone();
    banded.options.simultaneous_replicates = Some(200);
    banded.options.bandwidth = Some(0.5);
    let error = run(&banded, &[0, 3]).err().expect("a band was requested");
    assert!(error.to_string().contains("simultaneous bands are unavailable"), "{error}");
}
