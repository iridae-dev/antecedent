//! The one admission of the observation weights a series carries.
//!
//! A series' observation weights change the estimand (the weighted population) and the
//! variance, and the contract labels the functional `+sampling_weights` whenever the series
//! carries them. Every temporal route therefore either reads the weights faithfully or refuses
//! them: a route that fit the unweighted series would publish the unweighted population's number
//! under a weighted label. The routes call [`admit_series_weights`] at their executor, so the
//! decision, the refusal's code and the way forward live here once.
//!
//! The weighted constructions are the frequentist temporal linear adjustment of a single window
//! and the frequentist sustained contrast of a supplied `TemporalDag` (weighted least squares on
//! the lag windows, each weighted by the observation it ends at), without effect modifiers or
//! refuters, and the trial/panel designs that read table weights elsewhere. Every other temporal
//! route (mediation, response and derivative, class and graph-posterior mixtures, Bayesian
//! fits, conditional effects, cross-world contrasts, association) has no weighted construction
//! and refuses.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_data::TimeSeriesData;

use crate::CausalError;

/// What a temporal route does with the observation weights its series carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SeriesWeightReading {
    /// The route fits the weighted windows (weighted least squares), so the weights are honoured
    /// and the result may carry the `+sampling_weights` label.
    Honoured,
    /// The route has no weighted construction for the request.
    Unread {
        /// Why the route does not read the weights.
        reason: &'static str,
        /// The next attempt the refusal names.
        remedy: &'static str,
    },
}

impl SeriesWeightReading {
    /// A route with no weighted construction, whose remedy is to analyse with a route that
    /// has one or to drop the weights.
    pub(super) const fn unread(reason: &'static str) -> Self {
        Self::Unread { reason, remedy: WEIGHTED_ROUTE_REMEDY }
    }
}

/// Why a route with no weighted construction at all does not read the weights.
pub(super) const NO_WEIGHTED_CONSTRUCTION: &str = "it has no sampling-weighted construction";

/// The next attempt of a refusal whose route has no weighted construction: the routes that do
/// read a series' weights, or dropping them.
const WEIGHTED_ROUTE_REMEDY: &str = "analyse with the frequentist temporal linear adjustment or \
     sustained contrast on a supplied TemporalDag, with no effect modifiers and no refuters \
     (refute(None)), which fit the weighted population, or drop the weights from the series";

/// Admit the observation weights of a series to `route`.
///
/// `weighted` is whether the series carries weights; `reading` is what the route does with
/// them. A series without weights is always admitted, and so is a route that honours them.
///
/// # Errors
///
/// `option_not_applicable` (detail `sampling_weights.route_unsupported`) when the series is
/// weighted and the route would fit it unweighted, naming the route, why it has no weighted
/// construction and the next attempt.
pub(super) fn admit_series_weights(
    weighted: bool,
    route: &str,
    reading: SeriesWeightReading,
) -> Result<(), CausalError> {
    let SeriesWeightReading::Unread { reason, remedy } = reading else {
        return Ok(());
    };
    if !weighted {
        return Ok(());
    }
    Err(CausalError::Estimate(antecedent_estimate::sampling_weights::unsupported_route(&format!(
        "the series carries observation weights, which {route} does not read: {reason}, so the \
         estimate would describe the unweighted series under a sampling-weighted label. Next \
         attempt: {remedy}"
    ))))
}

/// [`admit_series_weights`] for a route with no weighted construction at all: refuse when
/// `data` carries weights.
///
/// # Errors
///
/// As [`admit_series_weights`].
pub(super) fn refuse_unread_series_weights(
    data: &TimeSeriesData,
    route: &str,
) -> Result<(), CausalError> {
    admit_series_weights(
        data.storage().weights().is_some(),
        route,
        SeriesWeightReading::unread(NO_WEIGHTED_CONSTRUCTION),
    )
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    fn series(weights: Option<&[f64]>) -> TimeSeriesData {
        let t = [0.0, 1.0, 2.0, 3.0];
        let y = [1.0, 2.0, 1.5, 3.0];
        let data = TimeSeriesData::from_f64_columns([("t", &t[..]), ("y", &y[..])], 1).unwrap();
        match weights {
            Some(weights) => data.with_weights(Arc::from(weights.to_vec())).unwrap(),
            None => data,
        }
    }

    #[test]
    fn an_unweighted_series_is_admitted_to_every_route() {
        let data = series(None);
        assert!(refuse_unread_series_weights(&data, "temporal mediation").is_ok());
        assert!(
            admit_series_weights(false, "a route", SeriesWeightReading::unread("it reads nothing"))
                .is_ok()
        );
    }

    #[test]
    fn a_weighted_series_is_admitted_only_to_a_route_that_honours_the_weights() {
        let data = series(Some(&[1.0, 2.0, 1.0, 2.0]));
        assert!(
            admit_series_weights(true, "the window fit", SeriesWeightReading::Honoured).is_ok()
        );
        let error = refuse_unread_series_weights(&data, "temporal mediation")
            .expect_err("a route without a weighted construction refuses")
            .to_string();
        assert!(
            error.contains(antecedent_core::reason_details::sampling_weights::ROUTE_UNSUPPORTED),
            "{error}"
        );
        assert!(error.contains("temporal mediation"), "{error}");
        assert!(error.contains("Next attempt:"), "{error}");
        assert!(error.contains("refute(None)"), "{error}");
    }

    #[test]
    fn the_refusal_names_the_reason_the_route_gave() {
        let error = admit_series_weights(
            true,
            "the conditional effect",
            SeriesWeightReading::unread("its interaction design centres at unweighted means"),
        )
        .expect_err("refused")
        .to_string();
        assert!(error.contains("centres at unweighted means"), "{error}");
        assert!(
            error.contains(antecedent_core::reason_details::sampling_weights::ROUTE_UNSUPPORTED),
            "{error}"
        );
    }
}
