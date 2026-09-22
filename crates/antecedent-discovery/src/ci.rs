//! Conditional-independence surface owned by discovery.
//!
//! Numeric kernels remain in `antecedent-stats`; this module re-exports the DESIGN
//! trait contract so discovery algorithms depend on a discovery-owned CI API.
//! Concrete test constructors live in `antecedent-stats` (use
//! [`ci_from_name`] or import the type from that crate).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_stats::ensure_alpha_resolvable;

use crate::error::DiscoveryError;

/// Refuse a CI test / `alpha` / multiplicity combination whose decisions would not mean what
/// they say.
///
/// - A permutation test cannot report a p-value below `1/(replicates+1)`; with `alpha` at or
///   below that floor every test "fails to reject" whatever the data show, which reads as
///   evidence of independence (default 49 replicates: nothing rejects at `alpha <= 0.02`).
/// - A test whose `p_value` is a posterior probability (the Bayesian tests) is not a
///   frequentist p-value, so Benjamini–Hochberg / FDR adjustment of it is meaningless.
///
/// # Errors
///
/// [`DiscoveryError::Stats`] when `alpha` is unresolvable;
/// [`DiscoveryError::Unsupported`] when FDR is requested for a non-frequentist p-value.
pub fn ensure_ci_decisions_meaningful(
    ci: &dyn ConditionalIndependence,
    significance: SignificanceMethod,
    alpha: f64,
    fdr_requested: bool,
) -> Result<(), DiscoveryError> {
    ensure_alpha_resolvable(ci.min_attainable_p(significance), alpha)?;
    if fdr_requested && !ci.p_value_is_frequentist() {
        return Err(DiscoveryError::unsupported(
            "FDR adjustment requires frequentist p-values; this CI test reports a posterior \
             probability of independence in p_value",
        ));
    }
    Ok(())
}

/// A test whose per-row weights are aligned to the end of a series may only run on a frame that
/// is exactly that series minus its leading lag rows, with no interior rows removed.
///
/// Trailing alignment is only correct then: a weight vector of another length, a stacked
/// multi-environment frame, or a frame with masked / missing rows (which discovery compacts
/// before testing) would silently pair weights with the wrong observations.
///
/// # Errors
///
/// [`DiscoveryError::Unsupported`] when the weights do not fit the frame.
pub fn ensure_ci_fits_frame(
    ci: &dyn ConditionalIndependence,
    frame: &antecedent_data::LaggedFrame,
) -> Result<(), DiscoveryError> {
    let Some(weights) = ci.series_aligned_weights_len() else {
        return Ok(());
    };
    let series_rows = frame.n_effective() + frame.max_lag() as usize;
    if weights != series_rows || !frame.is_fully_valid() {
        return Err(DiscoveryError::unsupported(
            "observation weights must have one entry per series row and the lagged frame must be \
             the complete series (no masked, missing or pooled rows) to pair them with \
             observations",
        ));
    }
    Ok(())
}

pub use antecedent_stats::{
    CiBatchRequest, CiBatchResult, CiPreparationPlan, CiQuery, CiResult, CiWorkspace,
    ConditionalIndependence, ConditionalIndependenceTest, ConfidenceMethod, PartialCorrelation,
    PreparedCiTest, SignificanceMethod, ci_from_name,
};

#[cfg(test)]
mod tests {
    use antecedent_core::VariableId;
    use antecedent_data::{LaggedFrame, TimeSeriesData};
    use antecedent_stats::{
        BayesFactorCi, PartialCorrelation, PosteriorDependenceCi, WeightedPartialCorrelation,
    };

    use super::*;

    #[test]
    fn series_aligned_weights_must_cover_exactly_the_series() {
        let x: Vec<f64> = (0..20).map(|i| f64::from(i).sin()).collect();
        let y: Vec<f64> = (0..20).map(|i| f64::from(i).cos()).collect();
        let data = TimeSeriesData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())], 1)
            .unwrap();
        let variables = [VariableId::from_raw(0), VariableId::from_raw(1)];
        let frame = LaggedFrame::from_series(
            &data,
            &variables,
            2,
            &antecedent_core::KernelPolicy::default_policy(),
        )
        .unwrap();
        // One weight per series row: the frame is that series minus its two leading rows.
        assert!(
            ensure_ci_fits_frame(
                &WeightedPartialCorrelation::aligned_to_series_end(vec![1.0; 20]),
                &frame
            )
            .is_ok()
        );
        // Too long or too short: the trailing window would pair weights with other rows.
        for len in [19, 21, 40] {
            let ci = WeightedPartialCorrelation::aligned_to_series_end(vec![1.0; len]);
            assert!(ensure_ci_fits_frame(&ci, &frame).is_err(), "{len}");
        }
        // Tests without series-aligned weights are unaffected.
        assert!(ensure_ci_fits_frame(&PartialCorrelation, &frame).is_ok());
    }

    #[test]
    fn posterior_probability_tests_are_refused_under_fdr_but_not_without_it() {
        let significance = SignificanceMethod::Analytic;
        for ci in
            [&BayesFactorCi::new() as &dyn ConditionalIndependence, &PosteriorDependenceCi::new()]
        {
            assert!(ensure_ci_decisions_meaningful(ci, significance, 0.05, true).is_err());
            assert!(ensure_ci_decisions_meaningful(ci, significance, 0.05, false).is_ok());
        }
        assert!(
            ensure_ci_decisions_meaningful(&PartialCorrelation, significance, 0.05, true).is_ok()
        );
    }
}
