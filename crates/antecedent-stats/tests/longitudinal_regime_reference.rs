//! Longitudinal regime values (inverse-probability, g-formula, sequential doubly robust)
//! against exact rational references.
//!
//! `oracle_data/longitudinal_regime_reference.py` (frozen in the JSON beside it) forms the
//! matched-trajectory Horvitz-Thompson mean with its standard error and Kish size, the
//! g-formula mean of supplied period predictions, and the telescoped sequential doubly robust
//! score `Q_0 + sum G_{t-1}(Q_t - Q_{t-1}) + G_{T-1}(Y - Q_{T-1})`, in `Fraction` arithmetic.
//! The recursion the product runs backwards is not the formula the reference evaluates. Cases
//! cover static, alternating and history-dependent regimes, probabilities at 0.02 with
//! cumulative weights near 2.5e4, and an outcome offset of 1e8.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_stats::longitudinal_regime::{
    evaluate_g_formula_value, evaluate_regime_value,
    evaluate_sequential_dr_value_with_subject_scores, SequentialDrInputs, SubjectHistories,
};
use serde_json::Value;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn bools(v: &Value) -> Vec<bool> {
    v.as_array().unwrap().iter().map(|x| x.as_bool().unwrap()).collect()
}

fn close(got: f64, want: f64, rel: f64, what: &str) {
    assert!((got - want).abs() <= rel * want.abs().max(1e-300), "{what}: got {got} want {want}");
}

#[test]
fn regime_values_match_the_exact_rational_references() {
    let file: Value =
        serde_json::from_str(include_str!("oracle_data/longitudinal_regime_reference.json"))
            .unwrap();
    for case in file["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let n = case["n"].as_u64().unwrap() as usize;
        let periods = case["periods"].as_u64().unwrap() as usize;
        let floor = case["floor"].as_f64().unwrap();
        let (actions, treatment) = (bools(&case["actions"]), bools(&case["treatment"]));
        let (p, c) = (floats(&case["p"]), floats(&case["c"]));
        let (history, observed) =
            (bools(&case["observed_history"]), bools(&case["outcome_observed"]));
        let (outcomes, q) = (floats(&case["outcomes"]), floats(&case["q"]));
        let histories = SubjectHistories {
            treatment: &treatment,
            treatment_probability: &p,
            outcome_observed: &observed,
            censoring_probability: &c,
            periods,
            minimum_probability: floor,
        };

        let ipw = evaluate_regime_value(&outcomes, &histories, &actions)
            .unwrap_or_else(|e| panic!("{name} ipw: {e}"));
        let want = &case["ipw"];
        close(ipw.value, want["value"].as_f64().unwrap(), 1e-11, &format!("{name} ipw value"));
        close(
            ipw.score_standard_error.unwrap(),
            want["se"].as_f64().unwrap(),
            1e-9,
            &format!("{name} ipw se"),
        );
        close(
            ipw.effective_sample_size,
            want["ess"].as_f64().unwrap(),
            1e-11,
            &format!("{name} ipw kish size"),
        );
        close(
            ipw.maximum_weight,
            want["max_weight"].as_f64().unwrap(),
            1e-11,
            &format!("{name} max weight"),
        );
        close(
            ipw.matched_observed_fraction,
            want["matched_fraction"].as_f64().unwrap(),
            1e-14,
            name,
        );
        close(ipw.minimum_action_probability, want["min_pi"].as_f64().unwrap(), 1e-14, name);
        close(ipw.minimum_censoring_probability, want["min_c"].as_f64().unwrap(), 1e-14, name);

        let g = evaluate_g_formula_value(&q, &actions, &p, &c, n, periods, floor).unwrap();
        close(
            g.value,
            case["gformula"]["value"].as_f64().unwrap(),
            1e-12,
            &format!("{name} g-formula"),
        );

        let (dr, scores) = evaluate_sequential_dr_value_with_subject_scores(
            &outcomes,
            &histories,
            &SequentialDrInputs {
                actions: &actions,
                observation_history: &history,
                q_predictions: &q,
            },
        )
        .unwrap_or_else(|e| panic!("{name} dr: {e}"));
        let want = &case["sequential"];
        close(dr.value, want["value"].as_f64().unwrap(), 1e-10, &format!("{name} dr value"));
        close(
            dr.maximum_weight,
            want["max_weight"].as_f64().unwrap(),
            1e-11,
            &format!("{name} dr max weight"),
        );
        close(
            dr.matched_observed_fraction,
            want["observed_fraction"].as_f64().unwrap(),
            1e-14,
            name,
        );
        for (i, (got, want)) in scores.iter().zip(floats(&want["scores"])).enumerate() {
            let scale = want.abs().max(1.0);
            assert!((got - want).abs() <= 1e-9 * scale, "{name} dr score {i}: {got} vs {want}");
        }
    }
}
