//! Facade constructions measured by the estimator-level coverage suite
//! (`crates/antecedent-estimate/src/calibration_coverage.rs`).
//!
//! Each case names the `AverageEffect` construction the facade reports when a
//! study selects that estimator configuration on a `Dag`: the runtime
//! calibration key of such an execution. The estimator-level tests bind their
//! records to these constructions, and
//! `crates/antecedent/tests/calibration_binding.rs` runs the facade with each
//! configuration and checks that it reports exactly this construction, so an
//! estimator-level record can only describe what the facade reports.
//!
//! Plain data (no `antecedent` types): `antecedent-estimate` includes this
//! file beside `calibration.rs`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(dead_code)]

use super::calibration::Construction;

/// One estimator-level coverage test and the facade construction it measures.
#[derive(Debug, Clone, Copy)]
pub struct EstimatorLevelCase {
    /// `#[test]` fn in `calibration_coverage.rs`.
    pub test: &'static str,
    /// Resolved plan estimator (`EstimatorId::as_str()`).
    pub estimator: &'static str,
    /// Interval method the test scores.
    pub interval: &'static str,
    /// Analytic SE kind the estimator records; empty when none.
    pub se_kind: &'static str,
    /// Functional label (target population and outcome functional).
    pub functional: &'static str,
}

/// Every estimator-level test that measures a construction the facade reports.
pub const CASES: &[EstimatorLevelCase] = &[
    case(
        "linear_adjustment_analytic_ci_coverage",
        "linear.adjustment.ate",
        "analytic_se",
        "homoskedastic",
        "all_observed.mean",
    ),
    case(
        "linear_adjustment_hc1_ci_coverage",
        "linear.adjustment.ate",
        "analytic_se",
        "hc1",
        "all_observed.mean",
    ),
    case(
        "ipw_hajek_bootstrap_ci_coverage",
        "propensity.weighting",
        "bootstrap_se",
        "",
        "all_observed.mean",
    ),
    case(
        "ipw_hajek_analytic_ci_coverage",
        "propensity.weighting",
        "analytic_se",
        "",
        "all_observed.mean",
    ),
    case(
        "ipw_hajek_analytic_conformance_scm_ci_coverage",
        "propensity.weighting",
        "analytic_se",
        "",
        "all_observed.mean",
    ),
    case("aipw_analytic_ci_coverage", "aipw", "analytic_se", "homoskedastic", "all_observed.mean"),
    case("aipw_att_hc1_ci_coverage", "aipw", "analytic_se", "hc1", "treated.mean"),
    case("aipw_atc_hc1_boundary_within_band", "aipw", "analytic_se", "hc1", "untreated.mean"),
    case("aipw_ate_hc1_ci_coverage", "aipw", "analytic_se", "hc1", "all_observed.mean"),
    case("aipw_att_cluster_ci_coverage", "aipw", "analytic_se", "cluster", "treated.mean"),
    case(
        "matching_homoskedastic_ci_coverage",
        "propensity.matching",
        "analytic_se",
        "homoskedastic",
        "treated.mean",
    ),
    case("wald_iv_analytic_ci_coverage", "iv.wald", "anderson_rubin", "", "all_observed.mean"),
    // A non-homoskedastic Wald SE is never licensed (Anderson-Rubin requires the
    // homoskedastic assumption, and the estimator never publishes a finite
    // `se_analytic`), so the facade withholds the interval entirely: no product,
    // `calibration_coverage.rs::wald_coverage_on` only `report()`s this cell,
    // never `assert()`s or `emit()`s a coverage record for it.
    case("wald_iv_hc1_ci_coverage", "iv.wald", "none", "", "all_observed.mean"),
    case("iv_2sls_analytic_ci_coverage", "iv.2sls", "anderson_rubin", "", "all_observed.mean"),
    // Non-homoskedastic (HC1): Anderson-Rubin requires the homoskedastic
    // assumption and 2SLS never publishes a finite `se_analytic`, so the
    // facade withholds the interval entirely, same as the Wald IV HC1 case.
    case("iv_2sls_hc1_heteroskedastic_ci_coverage", "iv.2sls", "none", "", "all_observed.mean"),
    case(
        "frontdoor_stacked_hc0_ci_coverage",
        "frontdoor.linear_two_stage",
        "analytic_se",
        "hc0",
        "all_observed.mean",
    ),
    case(
        "frontdoor_stacked_hc1_ci_coverage",
        "frontdoor.linear_two_stage",
        "analytic_se",
        "hc1",
        "all_observed.mean",
    ),
    case(
        "frontdoor_functional_saturated_ci_coverage",
        "frontdoor.functional",
        "analytic_se",
        "",
        "all_observed.mean",
    ),
    case(
        "frontdoor_functional_arm_linear_ci_coverage",
        "frontdoor.functional",
        "analytic_se",
        "",
        "all_observed.mean",
    ),
    case(
        "rd_sharp_analytic_ci_coverage",
        "rd.sharp",
        "analytic_se",
        "homoskedastic",
        "local_at_cutoff.mean",
    ),
    case(
        "rd_sharp_hc1_heteroskedastic_ci_coverage",
        "rd.sharp",
        "analytic_se",
        "hc1",
        "local_at_cutoff.mean",
    ),
    case(
        "wald_iv_weak_first_stage_adversarial_ci_coverage",
        "iv.wald",
        "anderson_rubin",
        "",
        "all_observed.mean",
    ),
    case(
        "ipw_hajek_weak_overlap_adversarial_ci_coverage",
        "propensity.weighting",
        "analytic_se",
        "",
        "all_observed.mean",
    ),
    case(
        "rd_sharp_hc1_curved_adversarial_ci_coverage",
        "rd.sharp",
        "analytic_se",
        "hc1",
        "local_at_cutoff.mean",
    ),
    case(
        "matching_heteroskedastic_adversarial_ci_coverage",
        "propensity.matching",
        "analytic_se",
        "homoskedastic",
        "treated.mean",
    ),
    case(
        "matching_heterogeneous_att_ci_coverage",
        "propensity.matching",
        "analytic_se",
        "homoskedastic",
        "treated.mean",
    ),
    case(
        "matching_heterogeneous_atc_ci_coverage",
        "propensity.matching",
        "analytic_se",
        "homoskedastic",
        "untreated.mean",
    ),
    case(
        "matching_heterogeneous_ate_ci_coverage",
        "propensity.matching",
        "analytic_se",
        "homoskedastic",
        "all_observed.mean",
    ),
    case(
        "frontdoor_stacked_hc1_curved_mediator_ci_coverage",
        "frontdoor.linear_two_stage",
        "analytic_se",
        "hc1",
        "all_observed.mean",
    ),
];

const fn case(
    test: &'static str,
    estimator: &'static str,
    interval: &'static str,
    se_kind: &'static str,
    functional: &'static str,
) -> EstimatorLevelCase {
    EstimatorLevelCase { test, estimator, interval, se_kind, functional }
}

/// The case for `test`.
///
/// # Panics
///
/// When `test` is not listed.
#[must_use]
pub fn case_for(test: &str) -> EstimatorLevelCase {
    *CASES
        .iter()
        .find(|case| case.test == test)
        .unwrap_or_else(|| panic!("{test} is not an estimator-level case"))
}

impl EstimatorLevelCase {
    /// The facade construction: a Frequentist `AverageEffect` on an explicit
    /// `Dag`, point identified, iid rows, reported at 0.95 — or at 0.0 when no
    /// interval product is licensed (`interval == "none"`), matching
    /// `IntervalBinding::none`.
    #[must_use]
    pub fn construction(&self) -> Construction {
        Construction {
            query: "AverageEffect".into(),
            graph_class: "Dag".into(),
            structure: "fixed".into(),
            modality: "tabular".into(),
            inference: "Frequentist".into(),
            estimator: self.estimator.into(),
            interval_method: self.interval.into(),
            se_kind: self.se_kind.into(),
            dependence: "iid".into(),
            posterior: String::new(),
            functional: self.functional.into(),
            identification: "point".into(),
            reported_level: if self.interval == "none" { 0.0 } else { 0.95 },
        }
    }
}
