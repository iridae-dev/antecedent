//! One cross-fit fold of the ridge- and lasso-penalised logistic propensity against an
//! independent reference with recorded inner folds.
//!
//! `tests/oracle_data/penalized_propensity_reference.py` (frozen in the JSON beside it) standardises
//! the training columns by their (weighted) mean and population standard deviation, zeroing a
//! column without spread; fits `-loglik + (lambda / 2)|b|^2` (ridge, Newton) or
//! `-loglik + lambda|b|_1` (lasso, proximal Newton accepted only at the KKT conditions of the
//! exact objective) on the inner training folds for every penalty of the grid; ranks the grid by
//! the weighted inner out-of-fold log loss with a larger penalty winning an exact tie; refits the
//! winner on all training rows and predicts the validation rows. The test asserts that the
//! product's seeded inner plan equals the recorded one, and that the selected penalty, the lasso
//! support, the inner loss and every validation probability equal the reference, for a default
//! ridge grid, a lasso with a constant column and a near-collinear pair, a weighted ridge, and a
//! ridge at a covariate offset of 1000 at scale 50.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use serde_json::Value;

use super::*;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn colmajor(rows: &Value) -> (Vec<f64>, usize, usize) {
    let rows: Vec<Vec<f64>> = rows.as_array().unwrap().iter().map(floats).collect();
    let (n, p) = (rows.len(), rows[0].len());
    let mut out = vec![0.0; n * p];
    for (i, row) in rows.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            out[j * n + i] = *v;
        }
    }
    (out, n, p)
}

fn inner_plan(case: &Value, t: &[f64], units: &[antecedent_data::GlobalId]) -> Vec<u32> {
    let arms: Vec<u32> =
        t.iter().map(|&v| u32::from(TreatmentArm::is_treated_indicator(v))).collect();
    derive_fold_ids(
        arms.len(),
        case["inner_k"].as_u64().unwrap() as usize,
        FoldSource::Stratified {
            strata: &arms,
            units,
            seed: mix64(
                case["seed"].as_u64().unwrap()
                    ^ INNER_PLAN_TAG.wrapping_add(case["fold"].as_u64().unwrap()),
            ),
        },
        &SHARED_FOLD_ID_ERRORS,
    )
    .unwrap()
}

fn run(case: &Value) -> (Vec<f64>, RidgeFoldSelection, Vec<antecedent_data::GlobalId>, Vec<f64>) {
    let (train, n_train, ncols) = colmajor(&case["train_rows"]);
    let (valid, n_valid, _) = colmajor(&case["valid_rows"]);
    let t = floats(&case["t_train"]);
    let units: Vec<antecedent_data::GlobalId> =
        (0..n_train as u64).map(antecedent_data::GlobalId::new).collect();
    let weights = (!case["weights"].is_null()).then(|| floats(&case["weights"]));
    let tuning =
        RidgeTuning::new(&floats(&case["lambdas"]), case["inner_k"].as_u64().unwrap() as usize)
            .unwrap();
    let penalty = if case["kind"] == "ridge" { FoldPenalty::Ridge } else { FoldPenalty::Lasso };
    let input = RidgeFoldInput {
        design_train: &train,
        n_train,
        design_valid: &valid,
        n_valid,
        ncols,
        t_train: &t,
        units: &units,
        fold: case["fold"].as_u64().unwrap() as usize,
        seed: case["seed"].as_u64().unwrap(),
        weights: weights.as_deref(),
    };
    let ctx = ExecutionContext::for_tests(5);
    let (predictions, selection) = fit_penalized_fold(penalty, &tuning, &input, &ctx).unwrap();
    (predictions, selection, units, t)
}

#[test]
fn selected_penalty_support_and_predictions_match_the_independent_reference() {
    let reference: Value = serde_json::from_str(include_str!(
        "../../tests/oracle_data/penalized_propensity_reference.json"
    ))
    .unwrap();
    for case in reference["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let want = &case["expected"];
        let (predictions, selection, units, t) = run(case);
        let recorded: Vec<u32> = case["inner_folds"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(
            inner_plan(case, &t, &units),
            recorded,
            "{name}: inner plan is the recorded one"
        );
        assert_eq!(selection.lambda, want["lambda"].as_f64().unwrap(), "{name}: selected penalty");
        let losses = floats(&want["inner_losses"]);
        let best = losses.iter().copied().fold(f64::INFINITY, f64::min);
        assert!((selection.inner_logloss - best).abs() <= 1e-6 * best, "{name}: inner loss");
        if case["kind"] == "lasso" {
            let support: Vec<usize> = want["support"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            assert_eq!(selection.support.as_deref(), Some(support.as_slice()), "{name}: support");
        } else {
            assert!(selection.support.is_none(), "{name}");
        }
        for (r, (g, w)) in predictions.iter().zip(floats(&want["valid_probability"])).enumerate() {
            assert!((g - w).abs() <= 1e-5 * w.max(1e-3), "{name} probability {r}: {g} vs {w}");
        }
    }
}

/// Writes the inner folds the reference script needs (the second stage of regeneration).
#[test]
#[ignore = "regeneration helper"]
fn emit_inner_folds() {
    let data: Value = serde_json::from_str(
        &std::fs::read_to_string(std::env::var("PENALIZED_REFERENCE_DATA").unwrap()).unwrap(),
    )
    .unwrap();
    let mut out = serde_json::Map::new();
    for case in data["cases"].as_array().unwrap() {
        let t = floats(&case["t_train"]);
        let units: Vec<antecedent_data::GlobalId> =
            (0..t.len() as u64).map(antecedent_data::GlobalId::new).collect();
        out.insert(
            case["name"].as_str().unwrap().to_owned(),
            serde_json::json!(inner_plan(case, &t, &units)),
        );
    }
    std::fs::write(
        std::env::var("PENALIZED_REFERENCE_OUT").unwrap(),
        serde_json::to_string(&Value::Object(out)).unwrap(),
    )
    .unwrap();
}
