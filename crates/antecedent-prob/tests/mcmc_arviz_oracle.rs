//! Numerical comparison with pinned `ArviZ` rank diagnostics on shared chains.

use antecedent_prob::mcmc_summary;

#[test]
fn rank_rhat_bulk_and_tail_ess_match_pinned_arviz_chains() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/validate/bayesian_checks/diagnostics_oracle.json"
    ))
    .unwrap();
    assert_eq!(fixture["oracle"]["version"], "0.20.0");
    for case in fixture["scenarios"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let samples: Vec<f64> = case["samples_chain_major"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap())
            .collect();
        let n_chains = usize::try_from(case["chains"].as_u64().unwrap())
            .expect("fixture chain count fits usize");
        let n_draws = usize::try_from(case["draws"].as_u64().unwrap())
            .expect("fixture draw count fits usize");
        let observed = mcmc_summary(&samples, n_chains, n_draws, 1);
        let expected = &case["arviz"];
        let rhat = expected["rhat_rank"].as_f64().unwrap();
        let bulk = expected["ess_bulk"].as_f64().unwrap();
        let tail = expected["ess_tail"].as_f64().unwrap();
        eprintln!("{name}: rust={observed:?} arviz=({rhat:.5}, {bulk:.2}, {tail:.2})");
        assert!((observed.max_rhat - rhat).abs() <= 1e-6, "{name} R-hat");
        assert!((observed.min_bulk_ess - bulk).abs() <= 0.05 * bulk.max(1.0), "{name} bulk ESS");
        assert!((observed.min_tail_ess - tail).abs() <= 0.05 * tail.max(1.0), "{name} tail ESS");
    }
}
