//! Posterior-mass mixture of per-atom refutation reports against an exact-rational evaluation of
//! its definition (`oracle_data/dbn_posterior_reference.py`).
//!
//! Compiled into the library's test build (`#[path]` in `lib.rs`) because a report is built only
//! inside the crate. Covers `RefutationReport::mixture_weighted` and
//! `QueryRefutationPlan::mix_weighted`: ordinary masses, a failing atom of every mass down to
//! 1e-300 (the mixture must fail), zero and negative masses, mismatched refuter ids, weights
//! of 1e300 (no overflow), weights of 1e-300 (no underflow), cancelling signs and the grouping
//! of several refuters over several atoms.

use serde_json::Value;

use crate::QueryRefutationPlan;
use crate::common::RefutationReport;

fn reference() -> Value {
    serde_json::from_str(include_str!("../tests/oracle_data/dbn_posterior_reference.json"))
        .expect("frozen reference")
}

fn report(v: &Value) -> RefutationReport {
    RefutationReport::builder(v["refuter"].as_str().unwrap())
        .original_ate(v["original"].as_f64().unwrap())
        .refuted_ate(v["refuted"].as_f64().unwrap())
        .comparison(v["comparison"].as_f64().unwrap())
        .informative(v["informative"].as_bool().unwrap())
        .passed(v["passed"].as_bool().unwrap())
        .failure_condition(v["failure"].as_str().map(std::sync::Arc::from))
        .replicates(v["replicates"].as_u64().unwrap() as u32)
        .build()
}

fn close(got: f64, want: f64, what: &str) {
    assert!((got - want).abs() <= 1e-13 * want.abs().max(1.0), "{what}: {got} vs {want}");
}

fn assert_mixed(got: &RefutationReport, want: &Value, label: &str) {
    assert_eq!(got.refuter.as_ref(), want["refuter"].as_str().unwrap(), "{label}");
    close(got.original_ate, want["original"].as_f64().unwrap(), &format!("{label} original"));
    close(got.refuted_ate, want["refuted"].as_f64().unwrap(), &format!("{label} refuted"));
    close(got.comparison, want["comparison"].as_f64().unwrap(), &format!("{label} comparison"));
    assert_eq!(got.informative, want["informative"].as_bool().unwrap(), "{label} informative");
    assert_eq!(got.passed, want["passed"].as_bool().unwrap(), "{label} passed");
    assert_eq!(got.replicates as u64, want["replicates"].as_u64().unwrap(), "{label} replicates");
    assert_eq!(
        got.failure_condition.as_deref(),
        want["failure"].as_str(),
        "{label} failure condition"
    );
}

#[test]
fn mixture_weighted_equals_the_exact_reference() {
    for case in reference()["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let owned: Vec<(f64, RefutationReport)> = case["items"]
            .as_array()
            .unwrap()
            .iter()
            .map(|item| (item["weight"].as_f64().unwrap(), report(&item["report"])))
            .collect();
        let borrowed: Vec<(f64, &RefutationReport)> = owned.iter().map(|(w, r)| (*w, r)).collect();
        let got = RefutationReport::mixture_weighted(&borrowed);
        match (&got, case["expected"].is_null()) {
            (None, true) => {}
            (Some(mixed), false) => assert_mixed(mixed, &case["expected"], name),
            _ => panic!("{name}: presence of a mixture disagrees with the reference"),
        }
    }
}

#[test]
fn mix_weighted_groups_refuters_over_atoms() {
    for plan in reference()["plans"].as_array().unwrap() {
        let atoms: Vec<(f64, Vec<RefutationReport>)> = plan["atoms"]
            .as_array()
            .unwrap()
            .iter()
            .map(|atom| {
                (
                    atom["weight"].as_f64().unwrap(),
                    atom["reports"].as_array().unwrap().iter().map(report).collect(),
                )
            })
            .collect();
        let mixed = QueryRefutationPlan::mix_weighted(atoms);
        let want = plan["expected"].as_array().unwrap();
        assert_eq!(mixed.len(), want.len(), "{}", plan["name"]);
        for (got, want) in mixed.iter().zip(want) {
            assert_mixed(got, want, plan["name"].as_str().unwrap());
        }
    }
}
