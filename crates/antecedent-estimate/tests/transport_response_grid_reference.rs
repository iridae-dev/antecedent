//! The transported augmented response grid against an exact rational reference.
//!
//! `oracle_data/transport_response_grid_reference.py` (frozen in the JSON beside it) evaluates
//! `theta_g = (1/n_target) [ sum_target mu_g + sum_source ((1-p)/p) w_g (Y - m) ]` in exact
//! rational arithmetic, with the signed effective sample size of the correction weights and the
//! selection overlap summary. Cases cover signed local-linear weights, selection odds of 1e6 and
//! 1e-6, an outcome offset of 1e8, one source row, one target row, ties, a grid point whose
//! correction weights all vanish, and target rows whose outcomes and weights must not enter.
//! The refusal tests execute the declared input contract.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_estimate::{ResponseGridInputs, transport_augmented_response_grid};
use antecedent_identify::{
    PopulationFactor, TransportCertificate, TransportFormula, TransportIdentification,
};
use serde_json::Value;

fn direct() -> TransportIdentification {
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

fn cases() -> Vec<Value> {
    serde_json::from_str::<Value>(include_str!(
        "oracle_data/transport_response_grid_reference.json"
    ))
    .unwrap()
    .as_array()
    .unwrap()
    .clone()
}

struct Table {
    trial: Vec<bool>,
    p: Vec<f64>,
    y: Vec<f64>,
    m: Vec<f64>,
    grid: Vec<f64>,
    mu: Vec<f64>,
    w: Vec<f64>,
}

fn table(case: &Value) -> Table {
    Table {
        trial: case["trial"].as_array().unwrap().iter().map(|b| b.as_bool().unwrap()).collect(),
        p: floats(&case["p"]),
        y: floats(&case["y"]),
        m: floats(&case["m"]),
        grid: floats(&case["grid"]),
        mu: floats(&case["mu"]),
        w: floats(&case["w"]),
    }
}

fn run(t: &Table) -> Result<antecedent_estimate::TransportResponseGridEstimate, String> {
    transport_augmented_response_grid(
        &direct(),
        &t.y,
        &t.trial,
        &t.p,
        &ResponseGridInputs {
            grid: &t.grid,
            observed_regression: &t.m,
            target_regression: &t.mu,
            response_weight: &t.w,
        },
    )
    .map_err(|e| e.to_string())
}

#[test]
fn the_grid_mean_and_overlap_match_the_exact_rational_reference() {
    for case in cases() {
        let name = case["name"].as_str().unwrap();
        let t = table(&case);
        let got = run(&t).unwrap();
        let (mean, ess, scale) =
            (floats(&case["mean"]), floats(&case["ess"]), floats(&case["scale"]));
        assert_eq!(got.grid.to_vec(), t.grid, "{name} grid order");
        for g in 0..t.grid.len() {
            // the sum is formed in binary64: a few rounding errors of the sum of absolute terms
            let tol = 64.0 * f64::EPSILON * scale[g] * t.trial.len() as f64;
            assert!(
                (got.mean[g] - mean[g]).abs() <= tol,
                "{name} mean[{g}] {} vs {}",
                got.mean[g],
                mean[g]
            );
            assert!(
                (got.source_effective_sample_size[g] - ess[g]).abs() <= 1e-12 * ess[g].max(1.0),
                "{name} ess[{g}] {} vs {}",
                got.source_effective_sample_size[g],
                ess[g]
            );
        }
        let sel = &case["sel"];
        let overlap = &got.selection_overlap;
        assert_eq!(overlap.probability_min, sel["pmin"].as_f64().unwrap(), "{name} pmin");
        assert_eq!(overlap.probability_max, sel["pmax"].as_f64().unwrap(), "{name} pmax");
        let want = sel["ess"].as_f64().unwrap();
        assert!((overlap.effective_sample_size - want).abs() <= 1e-10 * want, "{name} kish");
        assert_eq!(
            overlap.extreme_weight_count as u64,
            sel["extreme"].as_u64().unwrap(),
            "{name} extreme"
        );
    }
}

#[test]
fn the_declared_input_contract_is_enforced() {
    let base = cases().into_iter().find(|c| c["name"] == "balanced").unwrap();
    let ok = table(&base);
    assert!(run(&ok).is_ok());

    for bad in [0.0, 1.0, f64::NAN, -0.1] {
        let mut t = table(&base);
        t.p[1] = bad; // a target row: probabilities are validated on every row
        assert!(run(&t).unwrap_err().contains("strictly inside"), "p = {bad}");
    }
    let mut t = table(&base);
    t.y[1] = f64::INFINITY; // a target row's outcome must still be finite
    assert!(run(&t).is_err());
    let mut t = table(&base);
    let last = t.w.len() - 1;
    t.w[last] = f64::NAN; // so must a target row's response weight
    assert!(run(&t).is_err());
    let mut t = table(&base);
    t.mu[0] = f64::NAN;
    assert!(run(&t).is_err());
    let mut t = table(&base);
    t.grid[0] = f64::NAN;
    assert!(run(&t).is_err());
    let mut t = table(&base);
    t.w.pop();
    assert!(run(&t).unwrap_err().contains("length mismatch"));
    let mut t = table(&base);
    t.trial = vec![true; t.trial.len()];
    assert!(run(&t).unwrap_err().contains("source-trial and target"));
    let mut t = table(&base);
    t.trial = vec![false; t.trial.len()];
    assert!(run(&t).unwrap_err().contains("source-trial and target"));
}

#[test]
fn a_recursive_factorization_certificate_is_refused_not_evaluated() {
    let t = table(&cases()[0]);
    let id = TransportIdentification::Transportable {
        formula: TransportFormula::RecursiveFactorization {
            sum_out: Arc::from([]),
            factors: Arc::from([]),
        },
        certificate: TransportCertificate {
            rule: Arc::from("transport.sid.singleton_c_components"),
            selection_targets: Arc::from([]),
            premises: Arc::from([]),
        },
    };
    let err = transport_augmented_response_grid(
        &id,
        &t.y,
        &t.trial,
        &t.p,
        &ResponseGridInputs {
            grid: &t.grid,
            observed_regression: &t.m,
            target_regression: &t.mu,
            response_weight: &t.w,
        },
    )
    .unwrap_err();
    assert!(err.to_string().contains("recursive factorization"));
}
