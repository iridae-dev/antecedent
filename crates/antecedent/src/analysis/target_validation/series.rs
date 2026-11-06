//! Series and multi-environment perturbation refits.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, CausalError, CausalRng, ExecutionContext, JoinFreeRows, Modal, OwnedColumn, PreparedRefit,
    TableView, TargetFamily, TargetPerturbation, TargetRefitError, TimeSeriesData, VariableId,
    block_weights, data_failed, failed, join_free_mask, join_free_rows, max_graph_lag,
    operational_refit_error, series_with, shifted, subsample_bounds, times, truncated_series,
    unbiased_index, unsupported, window_weights,
};

impl PreparedRefit<'_> {
    /// One series perturbed for `perturbation`; `None` when the perturbation is not a column
    /// or weight change of the series (common cause, handled by the compiler).
    fn perturb_series(
        &self,
        series: &TimeSeriesData,
        perturbation: TargetPerturbation,
        rng: &mut CausalRng,
        exposure: Option<VariableId>,
        outcome_control: Option<VariableId>,
    ) -> Result<TimeSeriesData, TargetRefitError> {
        let storage = series.storage();
        let n = series.row_count();
        let span = max_graph_lag(self.study()) + 1;
        // A design-respecting resample: replicate 0 is the series itself.
        let resample =
            |replicate: u32, rng: &mut CausalRng| -> Result<Option<Arc<[f64]>>, TargetRefitError> {
                if replicate == 0 { Ok(None) } else { block_weights(n, span, rng).map(Some) }
            };
        match perturbation {
            TargetPerturbation::PlaceboTreatment { .. } => {
                let values = Self::permuted(Self::column_of(storage, self.treatment)?, rng, true);
                series_with(series, &[(self.treatment, values)], None, None)
            }
            TargetPerturbation::PlaceboOutcome { .. } => {
                let values = Self::permuted(Self::column_of(storage, self.outcome)?, rng, true);
                series_with(series, &[(self.outcome, values)], None, None)
            }
            TargetPerturbation::Subset { fraction, .. } => {
                if n < 2 {
                    return Err(unsupported("a window needs at least two periods", "more data"));
                }
                // A circular window of design rows: the row ending at each period keeps its own
                // lag window, and every period is in a window with the same probability
                // (`width / n`), so the window estimates centre on the full-series estimate
                // instead of over-weighting the middle of the series.
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "the subset fraction lies in (0, 1), and the saturating cast is clamped into [1, n]"
                )]
                let width = ((fraction * n as f64).round() as usize).clamp(1, n);
                let start = unbiased_index(rng, n);
                let weights: Arc<[f64]> =
                    (0..n).map(|t| if (t + n - start) % n < width { 1.0 } else { 0.0 }).collect();
                series_with(series, &[], None, Some(weights))
            }
            TargetPerturbation::Bootstrap { .. } => {
                series_with(series, &[], None, Some(block_weights(n, span, rng)?))
            }
            TargetPerturbation::NegativeControlExposure { replicate } => {
                let control = exposure.ok_or_else(|| unsupported("no exposure", "declare one"))?;
                let values: Arc<[f64]> = Arc::from(Self::column_of(storage, control)?);
                let weights = resample(replicate, rng)?;
                series_with(series, &[(self.treatment, values)], None, weights)
            }
            TargetPerturbation::NegativeControlOutcome { replicate } => {
                let weights = resample(replicate, rng)?;
                if let Some(control) = outcome_control {
                    let values: Arc<[f64]> = Arc::from(Self::column_of(storage, control)?);
                    series_with(series, &[(self.outcome, values)], None, weights)
                } else {
                    // Lagged placebo: the outcome moved back past the longest lag, which no
                    // treatment the target reads can have caused.
                    let back = isize::try_from(span).unwrap_or(isize::MAX);
                    let (values, _) = shifted(Self::column_of(storage, self.outcome)?, -back);
                    let rows = window_weights(n, 2 * span, n);
                    series_with(
                        series,
                        &[(self.outcome, values)],
                        None,
                        Some(times(&rows, weights)),
                    )
                }
            }
            TargetPerturbation::PreTrend { replicate, lead } => {
                // Placebo timing: the treatment `lead` periods later in the role of the
                // treatment. Its lags in the design are the actual treatment path, so the
                // lead's coefficient is the effect of a future treatment on the present
                // outcome given the past: zero without anticipation or a pre-trend.
                let lead_rows = lead as usize;
                let lead = isize::try_from(lead).unwrap_or(isize::MAX);
                let (values, _) = shifted(Self::column_of(storage, self.treatment)?, lead);
                let weights = resample(replicate, rng)?;
                let rows = window_weights(n, 0, n.saturating_sub(lead_rows));
                series_with(series, &[(self.treatment, values)], None, Some(times(&rows, weights)))
            }
            TargetPerturbation::Anticipation { replicate } => {
                // Pre-period event study: the outcome from before every treatment period the
                // target reads in the role of the outcome.
                let back = isize::try_from(span).unwrap_or(isize::MAX);
                let (values, _) = shifted(Self::column_of(storage, self.outcome)?, -back);
                let weights = resample(replicate, rng)?;
                let rows = window_weights(n, 2 * span, n);
                series_with(series, &[(self.outcome, values)], None, Some(times(&rows, weights)))
            }
            TargetPerturbation::RandomCommonCause { .. } => Err(unsupported(
                "the common cause is constructed by the compiler",
                "random_common_cause",
            )),
        }
    }

    /// The pre-trend, anticipation and lagged-placebo series as an explicit window: the shifted
    /// column cut to the design rows that have a value, with the lag history kept in front of
    /// them, and a gathered circular-block resample of that window for replicates after the
    /// first. This is what the weighted construction means for a route that reads no row
    /// weights; `None` for any other perturbation.
    #[allow(
        clippy::type_complexity,
        reason = "the returned series retains its provenance and row selection"
    )]
    fn materialised_window(
        &self,
        series: &TimeSeriesData,
        perturbation: TargetPerturbation,
        rng: &mut CausalRng,
        ctx: &ExecutionContext,
    ) -> Result<
        Option<(
            TimeSeriesData,
            Option<crate::analysis::execute::TemporalRowProvenance>,
            Option<JoinFreeRows>,
        )>,
        TargetRefitError,
    > {
        let mut budget =
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
        let storage = series.storage();
        let n = series.row_count();
        let span = max_graph_lag(self.study()) + 1;
        let back = isize::try_from(span).unwrap_or(isize::MAX);
        let (replacement, from, to, replicate) = match perturbation {
            TargetPerturbation::PreTrend { replicate, lead } => {
                let lead_rows = lead as usize;
                let lead = isize::try_from(lead).unwrap_or(isize::MAX);
                let (values, _) = shifted(Self::column_of(storage, self.treatment)?, lead);
                ((self.treatment, values), 0, n.saturating_sub(lead_rows), replicate)
            }
            TargetPerturbation::Anticipation { replicate } => {
                let (values, _) = shifted(Self::column_of(storage, self.outcome)?, -back);
                ((self.outcome, values), 2 * span, n, replicate)
            }
            TargetPerturbation::NegativeControlOutcome { replicate }
                if self.options.negative_control_outcome.is_none() =>
            {
                let (values, _) = shifted(Self::column_of(storage, self.outcome)?, -back);
                ((self.outcome, values), 2 * span, n, replicate)
            }
            _ => return Ok(None),
        };
        let base = series_with(series, &[replacement], None, None)?;
        let window_start = from.saturating_sub(span);
        let Some(window) = truncated_series(&base, window_start, to)? else {
            return Ok(None);
        };
        if replicate == 0 {
            let provenance = if self.target.family() == TargetFamily::Attribution {
                Some(
                    crate::analysis::execute::TemporalRowProvenance::admit_contiguous_range(
                        window_start,
                        window.row_count(),
                        series.storage().weights().is_some(),
                        &mut budget,
                        ctx,
                    )
                    .map_err(|error| failed(&error))?,
                )
            } else {
                None
            };
            return Ok(Some((window, provenance, None)));
        }
        let rows = window.row_count();
        if self.target.family() == TargetFamily::Attribution {
            // An attribution fits row-wise lag-aligned tables and applies the analysis mask, so
            // the circular block resample is gathered with its join-crossing rows hidden.
            let length = antecedent_data::circular_block_length(span, rows);
            let mut map = vec![0_usize; rows];
            antecedent_data::fill_circular_block_indexes(rows, length, rng, &mut map)
                .map_err(data_failed)?;
            let (gathered, guard) =
                self.gathered_rows(&window, &map, Some(length), &mut budget, ctx)?;
            budget
                .charge_work(u64::try_from(map.len()).map_err(|_| {
                    failed(&CausalError::Resource {
                        message: "refit coordinate work overflow".into(),
                    })
                })?)
                .map_err(|e| failed(&CausalError::Resource { message: e.to_string() }))?;
            for tick in &mut map {
                if ctx.cancellation.is_cancelled() {
                    return Err(failed(&CausalError::Cancelled { stage: "temporal attribution" }));
                }
                *tick = tick.checked_add(window_start).ok_or_else(|| {
                    failed(&CausalError::Resource {
                        message: "refit source coordinate overflow".into(),
                    })
                })?;
            }
            let provenance = crate::analysis::execute::TemporalRowProvenance::admit_fixed_blocks(
                &map,
                Some(length),
                series.storage().weights().is_some(),
                &mut budget,
                ctx,
            )
            .map_err(|error| failed(&error))?;
            return Ok(Some((gathered, Some(provenance), Some(guard))));
        }
        // Any other route that reads neither row weights nor an analysis mask cannot skip the
        // design rows whose lag window crosses a join of a block resample, and a join pairs
        // values observed apart. Its replicate is a contiguous sub-window instead, which has no
        // join: a subsample of a dependent series (its spread is wider than the full series',
        // so the check is conservative where the window is short).
        let (start, width) = subsample_bounds(rows, span, rng);
        Ok(truncated_series(&window, start, start + width)?.map(|window| (window, None, None)))
    }

    /// The gathered series of the rows `map` (positions of `series`), with a design row whose lag
    /// window crosses a join of the map weighted zero and the sampling weight carried over.
    fn gathered_rows(
        &self,
        series: &TimeSeriesData,
        map: &[usize],
        block_length: Option<usize>,
        budget: &mut antecedent_core::TemporalBudget,
        ctx: &ExecutionContext,
    ) -> Result<(TimeSeriesData, JoinFreeRows), TargetRefitError> {
        let n = map.len();
        let span = max_graph_lag(self.study()) + 1;
        let time_index = match truncated_series(series, 0, n)? {
            Some(head) => head.time_index().clone(),
            None => series.time_index().clone(),
        };
        let storage = series.storage();
        let mut columns = Vec::with_capacity(storage.columns().len());
        for column in storage.columns() {
            let OwnedColumn::Float64(column) = column else {
                return Err(unsupported(
                    "a gathered window gathers float64 columns only",
                    "encode every column as float64",
                ));
            };
            let values: Arc<[f64]> = map.iter().map(|&row| column.values[row]).collect();
            let validity = column.validity.gather_rows(map).map_err(data_failed)?;
            columns.push(OwnedColumn::Float64(
                antecedent_data::Float64Column::new(column.id, values, validity)
                    .map_err(data_failed)?,
            ));
        }
        let block_boundaries =
            (self.target.family() == TargetFamily::Attribution).then_some(block_length).flatten();
        let weight_free = join_free_rows(map, span, block_boundaries, budget, ctx)?;
        let mask = if self.target.family() == TargetFamily::Attribution {
            Some(join_free_mask(storage, map, span, block_length, budget, ctx)?)
        } else {
            storage
                .analysis_mask()
                .map(|mask| mask.gather_rows(map))
                .transpose()
                .map_err(data_failed)?
        };
        budget
            .charge_work(u64::try_from(n).map_err(|_| {
                failed(&CausalError::Resource { message: "refit weight work overflow".into() })
            })?)
            .map_err(|e| failed(&CausalError::Resource { message: e.to_string() }))?;
        let mut values = Vec::new();
        values.try_reserve_exact(n).map_err(|_| {
            failed(&CausalError::Resource { message: "refit weight allocation refused".into() })
        })?;
        for (row, &source) in map.iter().enumerate() {
            if ctx.cancellation.is_cancelled() {
                return Err(failed(&CausalError::Cancelled { stage: "temporal attribution" }));
            }
            let sampling = storage.weights().map_or(1.0, |weights| weights[source]);
            values.push(if weight_free[row] { sampling } else { 0.0 });
        }
        let weights: Arc<[f64]> = values.into();
        let gathered = antecedent_data::OwnedColumnarStorage::try_new(
            storage.schema().clone(),
            columns,
            mask,
            Some(weights),
        )
        .and_then(|gathered| TimeSeriesData::try_new(gathered, time_index))
        .map_err(data_failed)?;
        Ok((gathered, weight_free))
    }

    /// The bootstrap of a series drawn as a gathered circular-block series, with a zero row
    /// weight on every design row whose lag window crosses a block join (such a row pairs values
    /// observed apart and would attenuate a lagged effect). A route that reads row weights
    /// recovers the replicate the multiplicities define. An attribution reads none, so those
    /// rows are hidden by the analysis mask, which it applies by deleting them before fitting
    /// and scoring; any other route that reads neither keeps the attenuation, of order
    /// `(span - 1) / block length`.
    fn gathered_bootstrap(
        &self,
        series: &TimeSeriesData,
        rng: &mut CausalRng,
        ctx: &ExecutionContext,
    ) -> Result<Vec<f64>, TargetRefitError> {
        let mut budget =
            antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
        let n = series.row_count();
        let span = max_graph_lag(self.study()) + 1;
        let length = antecedent_data::circular_block_length(span, n);
        let mut map = vec![0_usize; n];
        antecedent_data::fill_circular_block_indexes(n, length, rng, &mut map)
            .map_err(data_failed)?;
        let storage = series.storage();
        let mut columns = Vec::with_capacity(storage.columns().len());
        for column in storage.columns() {
            let OwnedColumn::Float64(column) = column else {
                return Err(unsupported(
                    "the gathered bootstrap gathers float64 columns only",
                    "encode every column as float64",
                ));
            };
            let values: Arc<[f64]> = map.iter().map(|&row| column.values[row]).collect();
            // Missing values and the analysis mask travel with the rows they belong to.
            let validity = column.validity.gather_rows(&map).map_err(data_failed)?;
            columns.push(OwnedColumn::Float64(
                antecedent_data::Float64Column::new(column.id, values, validity)
                    .map_err(data_failed)?,
            ));
        }
        let block_boundaries =
            (self.target.family() == TargetFamily::Attribution).then_some(length);
        let weight_free = join_free_rows(&map, span, block_boundaries, &mut budget, ctx)?;
        let mask = if self.target.family() == TargetFamily::Attribution {
            Some(join_free_mask(storage, &map, span, Some(length), &mut budget, ctx)?)
        } else {
            storage
                .analysis_mask()
                .map(|mask| mask.gather_rows(&map))
                .transpose()
                .map_err(data_failed)?
        };
        // A design row whose lag window crosses a block join pairs values observed apart, which
        // attenuates a lagged effect by about the share of such rows. Their weight is zero, so
        // a route that reads row weights recovers the replicate the multiplicities define; a
        // route that reads none keeps the attenuation, of order `(span - 1) / length`. A
        // sampling weight of the original sample is carried to the row it was gathered from, so
        // the replicate is the block resample of the weighted sample.
        budget
            .charge_work(u64::try_from(n).map_err(|_| {
                failed(&CausalError::Resource { message: "refit weight work overflow".into() })
            })?)
            .map_err(|e| failed(&CausalError::Resource { message: e.to_string() }))?;
        let mut values = Vec::new();
        values.try_reserve_exact(n).map_err(|_| {
            failed(&CausalError::Resource { message: "refit weight allocation refused".into() })
        })?;
        for (row, &source) in map.iter().enumerate() {
            if ctx.cancellation.is_cancelled() {
                return Err(failed(&CausalError::Cancelled { stage: "temporal attribution" }));
            }
            let sampling = storage.weights().map_or(1.0, |weights| weights[source]);
            values.push(if weight_free[row] { sampling } else { 0.0 });
        }
        let weights: Arc<[f64]> = values.into();
        let gathered = antecedent_data::OwnedColumnarStorage::try_new(
            storage.schema().clone(),
            columns,
            mask,
            Some(weights),
        )
        .and_then(|gathered| TimeSeriesData::try_new(gathered, series.time_index().clone()))
        .map_err(data_failed)?;
        if self.target.family() == TargetFamily::Attribution {
            self.reestimate_gathered_series(
                &gathered,
                &map,
                Some(length),
                storage.weights().is_some(),
                &mut budget,
                ctx,
            )
        } else {
            self.reestimate(&Modal::Series(gathered), ctx)
        }
    }

    #[allow(
        clippy::too_many_lines,
        reason = "the target-specific branches preserve a single checked operation"
    )]
    pub(super) fn series_refit(
        &self,
        series: &TimeSeriesData,
        perturbation: TargetPerturbation,
        rng: &mut CausalRng,
        ctx: &ExecutionContext,
    ) -> Result<Vec<f64>, TargetRefitError> {
        // An attribution fits its mechanisms on the rows it is given and reads no row weights,
        // so its block resample must be drawn as a gathered series, not as multiplicities.
        if (self.options.gathered_resamples || self.target.family() == TargetFamily::Attribution)
            && !matches!(perturbation, TargetPerturbation::RandomCommonCause { .. })
        {
            if matches!(perturbation, TargetPerturbation::Bootstrap { .. }) {
                return self.gathered_bootstrap(series, rng, ctx);
            }
            // Negative controls are substituted before drawing the same
            // circular block map. Gathering preserves the sampled multiplicities
            // and their source/block provenance without treating generated
            // bookkeeping weights as user-declared weights.
            let replacement = match perturbation {
                TargetPerturbation::NegativeControlExposure { replicate } if replicate > 0 => {
                    let control = self.control_exposure()?;
                    Some((self.treatment, Arc::from(Self::column_of(series.storage(), control)?)))
                }
                TargetPerturbation::NegativeControlOutcome { replicate } if replicate > 0 => self
                    .options
                    .negative_control_outcome
                    .map(|control| {
                        Self::column_of(series.storage(), control)
                            .map(|values| (self.outcome, Arc::from(values)))
                    })
                    .transpose()?,
                _ => None,
            };
            if let Some(replacement) = replacement {
                let substituted = series_with(series, &[replacement], None, None)?;
                return self.gathered_bootstrap(&substituted, rng, ctx);
            }
            if let Some((window, provenance, _guard)) =
                self.materialised_window(series, perturbation, rng, ctx)?
            {
                return self.reestimate_materialised_series(&window, provenance.as_ref(), ctx);
            }
        }
        if matches!(perturbation, TargetPerturbation::RandomCommonCause { .. }) {
            return self.random_common_cause(rng, ctx);
        }
        // The window of an attribution subset is drawn circularly: a window that must lie inside
        // the series includes the first and last periods far less often than the middle ones,
        // so the replicate mean of a statistic that depends on which periods are present
        // (a planted outlier at the edge) is not the original even for a clean series. A
        // circular window includes every period with the same probability.
        if let (TargetPerturbation::Subset { fraction, .. }, TargetFamily::Attribution) =
            (perturbation, self.target.family())
        {
            let n = series.row_count();
            if n < 2 {
                return Err(unsupported("a window needs at least two periods", "more data"));
            }
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "the subset fraction lies in (0, 1), and the saturating cast is clamped into [1, n]"
            )]
            let width = ((fraction * n as f64).round() as usize).clamp(1, n);
            let start = unbiased_index(rng, n);
            let map: Vec<usize> = (0..width).map(|i| (start + i) % n).collect();
            let mut budget =
                antecedent_core::TemporalBudget::new(ctx.temporal_materialization_limits());
            let (gathered, _guard) = self.gathered_rows(series, &map, None, &mut budget, ctx)?;
            return self.reestimate_gathered_series(
                &gathered,
                &map,
                None,
                series.storage().weights().is_some(),
                &mut budget,
                ctx,
            );
        }
        let exposure = match perturbation {
            TargetPerturbation::NegativeControlExposure { .. } => Some(self.control_exposure()?),
            _ => None,
        };
        let weighted = match perturbation {
            TargetPerturbation::Bootstrap { .. }
            | TargetPerturbation::PreTrend { .. }
            | TargetPerturbation::Anticipation { .. } => true,
            TargetPerturbation::NegativeControlOutcome { .. } => {
                self.options.negative_control_outcome.is_none()
                    || !matches!(
                        perturbation,
                        TargetPerturbation::NegativeControlOutcome { replicate: 0 }
                    )
            }
            TargetPerturbation::NegativeControlExposure { replicate } => replicate > 0,
            _ => false,
        };
        let perturbed = self.perturb_series(
            series,
            perturbation,
            rng,
            exposure,
            self.options.negative_control_outcome,
        )?;
        let refit = self.reestimate(&Modal::Series(perturbed), ctx);
        if refit.as_ref().err().is_some_and(operational_refit_error) {
            return refit;
        }
        // A route that reads no row weights (a conditional effect) cannot take the circular
        // window as zero weights; the same subset is then cut as a contiguous window, which keeps
        // every lag the procedure reads inside it.
        if refit.is_err() {
            if let TargetPerturbation::Subset { fraction, .. } = perturbation {
                let n = series.row_count();
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "the subset fraction lies in (0, 1), and the saturating cast is clamped into [1, n]"
                )]
                let width = ((fraction * n as f64).round() as usize).clamp(1, n);
                let start = unbiased_index(rng, n - width + 1);
                if let Some(window) = truncated_series(series, start, start + width)? {
                    match self.reestimate(&Modal::Series(window), ctx) {
                        Ok(values) => return Ok(values),
                        Err(error) if operational_refit_error(&error) => return Err(error),
                        Err(_) => {}
                    }
                }
            }
        }
        // Routes that do not consume row weights need a gathered circular-block series.
        // Attribution refits retain the block boundaries so complete windows do not cross joins.
        if refit.is_err() {
            if let Some((window, provenance, _guard)) =
                self.materialised_window(series, perturbation, rng, ctx)?
            {
                match self.reestimate_materialised_series(&window, provenance.as_ref(), ctx) {
                    Ok(values) => return Ok(values),
                    Err(error) if operational_refit_error(&error) => return Err(error),
                    Err(_) => {}
                }
            }
        }
        if matches!(perturbation, TargetPerturbation::Bootstrap { .. }) && refit.is_err() {
            match self.gathered_bootstrap(series, rng, ctx) {
                Ok(values) => return Ok(values),
                Err(error) if operational_refit_error(&error) => return Err(error),
                Err(_) => {}
            }
        }
        match refit {
            // The circular-block resample is carried as row multiplicities; a route without a
            // weighted construction refuses them, and the refuter is recorded with its reason.
            Err(TargetRefitError::Failed(error)) if weighted => Err(unsupported(
                format!("the route has no weighted construction for the block resample: {error}"),
                "the frequentist single-window temporal linear adjustment, which reads row \
                 weights, or subset stability",
            )),
            other => other,
        }
    }

    /// Independent environments: every environment is perturbed on its own by the series rules
    /// (a circular block resample carried as row multiplicities on top of the environment's own
    /// sampling weights, a circular shift of a placebo column, a window of the environment), and
    /// the study is re-run on the perturbed environments together.
    pub(super) fn multi_env_refit(
        &self,
        data: &antecedent_data::MultiEnvironmentData,
        perturbation: TargetPerturbation,
        rng: &mut CausalRng,
        ctx: &ExecutionContext,
    ) -> Result<Vec<f64>, TargetRefitError> {
        if matches!(perturbation, TargetPerturbation::RandomCommonCause { .. }) {
            return self.random_common_cause(rng, ctx);
        }
        let exposure = match perturbation {
            TargetPerturbation::NegativeControlExposure { .. } => Some(self.control_exposure()?),
            _ => None,
        };
        let mut perturbed = Vec::with_capacity(data.env_count());
        for environment in data.environments() {
            perturbed.push(self.perturb_series(
                environment,
                perturbation,
                rng,
                exposure,
                self.options.negative_control_outcome,
            )?);
        }
        let rebuilt =
            antecedent_data::MultiEnvironmentData::try_new(perturbed).map_err(data_failed)?;
        self.reestimate(&Modal::MultiEnv(rebuilt), ctx).map_err(|error| match error {
            TargetRefitError::Failed(error) => unsupported(
                format!("the route has no construction for the perturbed environments: {error}"),
                "a route that reads row weights and environment windows, or subset stability",
            ),
            other @ TargetRefitError::Unsupported { .. } => other,
        })
    }
}
