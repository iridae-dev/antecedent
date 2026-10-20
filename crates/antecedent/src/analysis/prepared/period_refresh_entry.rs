//! Prepared-lifecycle entry of the linear Gaussian refresh route.
//!
//! A prepared series study keeps its schema and sampling regularity. An autoregressive
//! mechanism of one of its columns is fitted by least squares and by the conjugate
//! normal-inverse-gamma posterior on the same rows, and each later observation period of the
//! same schema is applied through [`PeriodRefresh`]: it extends the summary in place when the
//! mechanism was declared invariant across periods, restarts it otherwise, and is refused when
//! it overlaps or reuses what was already used. Nothing here re-identifies or recompiles the
//! study.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{PreparedModality, PreparedStudy};
use crate::error::CausalError;
use crate::period_refresh::{PeriodRefresh, PeriodRefreshOutcome, PeriodRefreshSpec};
use antecedent_data::{TableView, TimeSeriesData};
use antecedent_estimate::temporal_refresh::{ObservationCadence, TemporalWindowIdentity};
use antecedent_state::ConjugateOlsPrior;

impl PreparedStudy {
    /// The values of `column` of `data`, after the prepared handle's own compatibility checks.
    fn period_values(&self, data: &TimeSeriesData, column: &str) -> Result<Vec<f64>, CausalError> {
        if self.modality != PreparedModality::Series {
            return Err(CausalError::Compile {
                message: "a period refresh needs a prepared series study; re-prepare with series \
                          data"
                    .into(),
            });
        }
        if data.schema() != &self.schema {
            return Err(CausalError::Compile {
                message: "a period refresh requires the same schema (variable names, types, and \
                          order) as prepare-time data"
                    .into(),
            });
        }
        if self.time_regularity.as_ref() != Some(&data.time_index().regularity) {
            return Err(CausalError::Compile {
                message: "a period refresh requires the same time-index regularity as \
                          prepare-time data; re-prepare after a time-index change"
                    .into(),
            });
        }
        let id = data.schema().id_of(column)?;
        Ok(data.float64_values(id)?)
    }

    /// Fit the first period of the autoregressive mechanism of `column` under the stated
    /// `prior`: one value per time index of `window.period`.
    ///
    /// # Errors
    /// A handle that is not a prepared series study, a schema or regularity change, an unknown
    /// column, or the refusals of [`PeriodRefresh::start`].
    pub fn start_period_refresh(
        &self,
        data: &TimeSeriesData,
        column: &str,
        spec: &PeriodRefreshSpec<'_>,
        window: &TemporalWindowIdentity,
        prior: &ConjugateOlsPrior,
    ) -> Result<PeriodRefresh, CausalError> {
        let values = self.period_values(data, column)?;
        Ok(PeriodRefresh::start(spec, window, prior, &values)?)
    }

    /// Apply a later observation period to a mechanism kept by
    /// [`Self::start_period_refresh`]. The handle is unchanged; a refused period leaves
    /// `refresh` unchanged.
    ///
    /// # Errors
    /// The refusals of [`Self::start_period_refresh`] and [`PeriodRefresh::refresh`]: an
    /// overlapping period or a reused snapshot (the next attempt is a later period of a new
    /// snapshot), or a refused window change.
    pub fn apply_period_refresh(
        &self,
        refresh: &mut PeriodRefresh,
        data: &TimeSeriesData,
        column: &str,
        window: &TemporalWindowIdentity,
        cadence: ObservationCadence,
        initial_law: &str,
    ) -> Result<PeriodRefreshOutcome, CausalError> {
        let values = self.period_values(data, column)?;
        Ok(refresh.refresh(window, cadence, initial_law, &values)?)
    }
}
