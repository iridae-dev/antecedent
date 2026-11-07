use std::sync::Arc;

use super::{HistoryConstruction, HistoryExecutionError};

/// The one multi-variable lag-aligned scan of a series.
///
/// A scan plans the windows `[t - max_lag, t]` over any set of lagged columns and gathers them
/// into reusable storage in one of the three window disciplines the temporal routes need:
/// every window must be intact ([`Self::complete`]), incomplete windows are kept in place and
/// flagged ([`Self::with_gaps`]), or incomplete windows are removed ([`Self::windows`] with
/// `gaps`). Every route engine reads its windows through this operator, so history
/// construction has one implementation; [`HistoryConstruction::scan_columns`] binds it to a
/// declared frame.
#[derive(Debug)]
pub struct HistoryScan {
    plan: antecedent_data::LaggedSamplePlan,
    workspace: antecedent_data::LaggedSampleWorkspace,
}

impl HistoryScan {
    /// Plan the windows of `columns` over `data`, `max_lag` lags deep.
    ///
    /// # Errors
    ///
    /// The refusals of [`antecedent_data::LaggedSamplePlan::new`].
    pub fn plan(
        data: &antecedent_data::TimeSeriesData,
        max_lag: u32,
        columns: impl Into<Arc<[antecedent_data::LaggedColumn]>>,
    ) -> Result<Self, antecedent_data::DataError> {
        Ok(Self {
            plan: data.plan_lagged_sample(max_lag, columns)?,
            workspace: antecedent_data::LaggedSampleWorkspace::default(),
        })
    }

    /// The planned windows.
    #[must_use]
    pub const fn lag_plan(&self) -> &antecedent_data::LaggedSamplePlan {
        &self.plan
    }

    /// Gather the windows; every window must be intact.
    ///
    /// # Errors
    ///
    /// Missing or non-float64 columns, a length mismatch, or missing values.
    pub fn complete(
        &mut self,
        data: &antecedent_data::TimeSeriesData,
        policy: &antecedent_core::KernelPolicy,
    ) -> Result<antecedent_data::LaggedPreparedSample<'_>, antecedent_data::DataError> {
        self.plan.prepare(data, &mut self.workspace, policy)
    }

    /// Gather the windows keeping the incomplete ones in place; `complete[i]` flags window
    /// `i`.
    ///
    /// # Errors
    ///
    /// Missing or non-float64 columns, a length mismatch, or a row-hiding analysis mask.
    pub fn with_gaps(
        &mut self,
        data: &antecedent_data::TimeSeriesData,
        policy: &antecedent_core::KernelPolicy,
    ) -> Result<antecedent_data::GappedSample<'_>, antecedent_data::DataError> {
        self.plan.prepare_with_gaps(data, &mut self.workspace, policy)
    }

    /// Gather the windows, removing the incomplete ones when `gaps` is set. The flags are
    /// `Some` exactly when `gaps` is set, one per planned window.
    ///
    /// # Errors
    ///
    /// As [`Self::complete`] and [`Self::with_gaps`].
    pub fn windows(
        &mut self,
        data: &antecedent_data::TimeSeriesData,
        policy: &antecedent_core::KernelPolicy,
        gaps: bool,
    ) -> Result<
        (antecedent_data::LaggedPreparedSample<'_>, Option<Vec<bool>>),
        antecedent_data::DataError,
    > {
        if gaps {
            let (sample, complete) =
                self.plan.prepare_complete_windows(data, &mut self.workspace, policy)?;
            Ok((sample, Some(complete)))
        } else {
            Ok((self.plan.prepare(data, &mut self.workspace, policy)?, None))
        }
    }
}

impl HistoryConstruction {
    /// Plan the multi-variable scan of `columns` on this frame.
    ///
    /// # Errors
    ///
    /// [`HistoryExecutionError::HistoryShorterThanColumns`] when the declared history cap is
    /// shorter than the deepest column (the history is not extended), or
    /// [`HistoryExecutionError::Scan`] when the plan is refused.
    pub fn scan_columns(
        &self,
        data: &antecedent_data::TimeSeriesData,
        columns: impl Into<Arc<[antecedent_data::LaggedColumn]>>,
    ) -> Result<HistoryScan, HistoryExecutionError> {
        let columns: Arc<[antecedent_data::LaggedColumn]> = columns.into();
        let needed = columns.iter().map(|c| c.lag.raw()).max().unwrap_or(0);
        if let Some(cap) = self.history.resolved_cap() {
            if needed > cap {
                return Err(HistoryExecutionError::HistoryShorterThanColumns { cap, needed });
            }
        }
        HistoryScan::plan(data, needed, columns)
            .map_err(|error| HistoryExecutionError::Scan { message: error.to_string() })
    }
}
