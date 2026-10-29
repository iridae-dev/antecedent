//! Fixed-grid multiplier band against the definition evaluated by an executing reference.
//!
//! `oracle_data/estimate_response_kennedy_dr_simultaneous_reference.py` (frozen in the JSON beside
//! it) draws the sign vectors from an independent SplitMix64 implementation, forms the maximum of
//! the standardized signed influence sums over the grid, takes the `ceil(level (B + 1) - 1e-9)`-th
//! smallest maximum, and is infinite when that rank exceeds the replicates, on five dyadic influence
//! matrices (one point, a few, nearly collinear columns, a rank that does not exist, a rank equal
//! to the replicate count). The band construction is private to the estimator, so the same
//! computation is written in this test, shown to equal the frozen reference on those matrices and to
//! consume the same random stream as the product (`CausalRng`), and then used to predict the band
//! the estimator publishes from its own exported influences: critical value, floor at the normal
//! quantile, and the floor at the largest pointwise Student interval. Refusals of a band without a
//! bandwidth or with fewer than 100 replicates are executed.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, CausalRng, ContinuousDomain, GridSpec, IdentificationStatus, ResponseFunctional,
    ResponseIdentification, ResponseQuery, ResponseUncertainty, ResponseValue, VariableId,
};
use antecedent_data::TabularData;
use antecedent_estimate::response::ContinuousResponseEstimator;
use antecedent_stats::{normal_ppf, student_t_ppf};
use serde_json::Value;

const GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Standard errors and the critical value of the multiplier band, from the definition.
fn reference_band(psi: &[Vec<f64>], level: f64, replicates: usize, seed: u64) -> (Vec<f64>, f64) {
    let n = psi[0].len();
    let se: Vec<f64> =
        psi.iter().map(|row| row.iter().map(|v| v * v).sum::<f64>().sqrt()).collect();
    let mut state = seed ^ GAMMA;
    let mut maxima = Vec::with_capacity(replicates);
    for _ in 0..replicates {
        let signs: Vec<f64> = (0..n)
            .map(|_| {
                state = state.wrapping_add(GAMMA);
                if mix64(state) & 1 == 0 { -1.0 } else { 1.0 }
            })
            .collect();
        let maximum = psi
            .iter()
            .zip(&se)
            .map(|(row, se)| row.iter().zip(&signs).map(|(a, b)| a * b).sum::<f64>().abs() / se)
            .fold(0.0_f64, f64::max);
        maxima.push(maximum);
    }
    maxima.sort_by(f64::total_cmp);
    let rank = ((level * (replicates as f64 + 1.0) - 1e-9).ceil().max(1.0)) as usize;
    let critical = if rank > replicates { f64::INFINITY } else { maxima[rank - 1] };
    (se, critical)
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_str().map_or_else(|| x.as_f64().unwrap(), |s| s.parse().unwrap()))
        .collect()
}

#[test]
fn the_in_test_band_equals_the_frozen_reference_and_the_product_stream() {
    let frozen: Value = serde_json::from_str(include_str!(
        "oracle_data/estimate_response_kennedy_dr_simultaneous_reference.json"
    ))
    .unwrap();
    // The product's stream is the SplitMix64 the reference implements.
    let stream = &frozen["stream"];
    let mut rng = CausalRng::from_seed(stream["seed"].as_u64().unwrap());
    for want in stream["first_outputs"].as_array().unwrap() {
        assert_eq!(rng.next_u64().to_string(), want.as_str().unwrap());
    }
    for case in frozen["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let psi: Vec<Vec<f64>> = case["psi"].as_array().unwrap().iter().map(floats).collect();
        let (se, critical) = reference_band(
            &psi,
            case["level"].as_f64().unwrap(),
            usize::try_from(case["replicates"].as_u64().unwrap()).unwrap(),
            case["seed"].as_u64().unwrap(),
        );
        for (got, want) in se.iter().zip(floats(&case["se"])) {
            assert!((got - want).abs() <= 1e-12 * want, "{name}: se");
        }
        match case["critical"].as_f64() {
            Some(want) => assert!((critical - want).abs() <= 1e-12 * want, "{name}: {critical}"),
            None => assert!(critical.is_infinite(), "{name}"),
        }
    }
}

fn estimator(
    replicates: Option<u32>,
    level: f64,
    seed: u64,
    bandwidth: Option<f64>,
) -> ContinuousResponseEstimator {
    let mut estimator = ContinuousResponseEstimator::new([VariableId::from_raw(2)]);
    estimator.options.folds = 3;
    estimator.options.nuisance_basis = 5;
    estimator.options.nuisance_lambda = 1.0;
    estimator.options.bandwidth = bandwidth;
    estimator.options.simultaneous_replicates = replicates;
    estimator.options.confidence_level = level;
    estimator.options.multiplier_seed = seed;
    estimator.options.export_row_diagnostics = true;
    estimator
}

#[test]
fn the_published_band_is_the_reference_band_of_its_own_influences() {
    let source: Value = serde_json::from_str(include_str!(
        "oracle_data/estimate_response_kennedy_dr_reference.json"
    ))
    .unwrap();
    let case = &source["cases"][0];
    let (a, x, y) = (floats(&case["a"]), floats(&case["x"]), floats(&case["y"]));
    let n = a.len();
    let data = TabularData::from_f64_columns([
        ("a", a.as_slice()),
        ("y", y.as_slice()),
        ("x", x.as_slice()),
    ])
    .unwrap();
    let grid = [-0.5, 0.0, 0.5];
    let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(Arc::from(grid)),
        ),
    });
    let run = |replicates: Option<u32>, level: f64, seed: u64, bandwidth: Option<f64>| {
        estimator(replicates, level, seed, bandwidth).estimate_identified(
            &data,
            &query,
            IdentificationStatus::NonparametricallyIdentified,
            AssumptionSet::new(),
        )
    };
    let diagnostic = |response: &antecedent_core::CausalResponse, id: &str| -> Vec<f64> {
        response
            .support
            .diagnostics
            .iter()
            .find(|d| d.id.as_ref() == id)
            .unwrap_or_else(|| panic!("missing diagnostic {id}"))
            .values
            .to_vec()
    };
    for (level, replicates, seed) in [(0.95, 400_u32, 99_u64), (0.9, 250, 7), (0.995, 100, 5)] {
        let response = run(Some(replicates), level, seed, Some(0.75)).unwrap();
        let flat = diagnostic(&response, "response.row_influence");
        let psi: Vec<Vec<f64>> = flat.chunks(n).map(<[f64]>::to_vec).collect();
        let (se, critical) = reference_band(&psi, level, replicates as usize, seed);
        let ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) =
            &response.estimate
        else {
            panic!("a mean curve is a surface");
        };
        let ResponseUncertainty::SimultaneousBand { lower, upper, replicates: got, .. } =
            &response.uncertainty
        else {
            panic!("a requested band is simultaneous");
        };
        assert_eq!(*got, replicates);
        // Pointwise floor: the widest Student critical value over the grid.
        let df = diagnostic(&response, "response.local_variance_degrees_of_freedom");
        let pointwise =
            df.iter().map(|d| student_t_ppf(0.5 + level / 2.0, *d)).fold(0.0_f64, f64::max);
        let simultaneous = critical.max(normal_ppf(0.5 + level / 2.0));
        for g in 0..grid.len() {
            let want_low = (mean[g] - simultaneous * se[g]).min(mean[g] - pointwise * se[g]);
            let want_up = (mean[g] + simultaneous * se[g]).max(mean[g] + pointwise * se[g]);
            if critical.is_infinite() {
                assert!(lower[g] == f64::NEG_INFINITY && upper[g] == f64::INFINITY);
            } else {
                assert!((lower[g] - want_low).abs() <= 1e-9 * (1.0 + want_low.abs()), "lower {g}");
                assert!((upper[g] - want_up).abs() <= 1e-9 * (1.0 + want_up.abs()), "upper {g}");
            }
        }
        // The same seed reproduces the band exactly; another seed moves it.
        let again = run(Some(replicates), level, seed, Some(0.75)).unwrap();
        assert_eq!(again.uncertainty, response.uncertainty);
    }
    let one = run(Some(400), 0.95, 1, Some(0.75)).unwrap();
    let two = run(Some(400), 0.95, 2, Some(0.75)).unwrap();
    assert_ne!(one.uncertainty, two.uncertainty);

    // A band needs a caller bandwidth and at least 100 replicates.
    assert!(format!("{:?}", run(Some(400), 0.95, 1, None).err()).contains("invalid continuous-response options"), "invalid continuous-response options");
    assert!(format!("{:?}", run(Some(99), 0.95, 1, Some(0.75)).err()).contains("invalid continuous-response options"), "invalid continuous-response options");
}
