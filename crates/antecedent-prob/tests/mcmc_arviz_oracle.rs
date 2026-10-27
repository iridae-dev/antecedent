//! Pinned `ArviZ` comparisons and independent high-precision primary-equation diagnostics.

use antecedent_prob::{McmcSummary, mcmc_summary};

fn summary(case: &serde_json::Value) -> McmcSummary {
    let samples: Vec<f64> = case["samples_chain_major"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap())
        .collect();
    let chains = usize::try_from(case["chains"].as_u64().unwrap()).unwrap();
    let draws = usize::try_from(case["draws"].as_u64().unwrap()).unwrap();
    assert_eq!(samples.len(), chains * draws);
    mcmc_summary(&samples, chains, draws, 1)
}

#[test]
fn rank_rhat_and_compatible_ess_cases_match_pinned_arviz_chains() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/validate/bayesian_checks/diagnostics_oracle.json"
    ))
    .unwrap();
    assert_eq!(fixture["oracle"]["version"], "0.20.0");
    assert_eq!(fixture["scenarios"].as_array().unwrap().len(), 3);
    for case in fixture["scenarios"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        assert!(matches!(name, "iid" | "shifted_chain" | "autocorrelated"));
        let observed = summary(case);
        let expected = &case["arviz"];
        let rhat = expected["rhat_rank"].as_f64().unwrap();
        let bulk = expected["ess_bulk"].as_f64().unwrap();
        let tail = expected["ess_tail"].as_f64().unwrap();
        eprintln!("{name}: rust={observed:?} arviz=({rhat:.5}, {bulk:.2}, {tail:.2})");
        assert!((observed.max_rhat - rhat).abs() <= 1e-6, "{name} R-hat");
        if name != "shifted_chain" {
            assert!(
                (observed.min_bulk_ess - bulk).abs() <= 0.05 * bulk.max(1.0),
                "{name} bulk ESS"
            );
            assert!(
                (observed.min_tail_ess - tail).abs() <= 0.05 * tail.max(1.0),
                "{name} tail ESS"
            );
        }
    }
}

#[test]
fn full_pair_diagnostics_match_the_high_precision_primary_equations() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/validate/bayesian_checks/independent_ips.json"
    ))
    .unwrap();
    assert_eq!(fixture["oracle"]["kind"], "independent_reimplementation");
    assert_eq!(fixture["tolerance"]["rhat_atol"], 1e-6);
    assert_eq!(fixture["tolerance"]["ess_atol"], 1e-6);
    let cases = fixture["scenarios"].as_array().unwrap();
    let mut names: Vec<_> = cases.iter().map(|case| case["name"].as_str().unwrap()).collect();
    names.sort_unstable();
    assert_eq!(
        names,
        ["autocorrelated", "iid", "odd_draws_drop_last", "shifted_chain", "tied_upper_tail"]
    );
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let observed = summary(case);
        for (actual, field) in [
            (observed.max_rhat, "rhat_rank"),
            (observed.min_bulk_ess, "ess_bulk"),
            (observed.min_tail_ess, "ess_tail"),
        ] {
            let expected = case["expected"][field].as_f64().unwrap();
            assert!((actual - expected).abs() <= 1e-6, "{name} {field}: {actual} vs {expected}");
        }
        // An explicit counterexample to the old blanket 5% ESS parity claim:
        // both references retain their own target and independently computed values.
        if name == "shifted_chain" {
            let raw = case["arviz_raw"]["ess_bulk"].as_f64().unwrap();
            assert!((observed.min_bulk_ess - raw).abs() > 0.05 * raw);
        }
        if name == "odd_draws_drop_last" {
            let raw = case["arviz_raw"]["rhat_rank"].as_f64().unwrap();
            assert!((observed.max_rhat - raw).abs() > 1e-6);
        }
    }
}
