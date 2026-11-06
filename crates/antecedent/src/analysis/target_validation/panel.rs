//! Panel perturbation and refitting.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, CausalRng, ExecutionContext, Modal, PanelData, PanelUnit, PreparedRefit, TableView,
    TargetPerturbation, TargetRefitError, TimeSeriesData, VariableId, data_failed, max_graph_lag,
    series_with, shifted, shuffle, truncated_series, unbiased_index, unsupported,
};

impl PreparedRefit<'_> {
    /// Periods `[start, end)` of a unit series, complete; refused when a column is not float64.
    fn cut(
        series: &TimeSeriesData,
        start: usize,
        end: usize,
    ) -> Result<TimeSeriesData, TargetRefitError> {
        truncated_series(series, start, end)?.ok_or_else(|| {
            unsupported(
                "a shifted panel series is cut to its complete periods, which needs float64 \
                 columns",
                "encode every column as float64",
            )
        })
    }

    /// Shift a panel column and retain the complete window of each unit.
    pub(super) fn shifted_panel_units(
        units: &[PanelUnit],
        id: VariableId,
        offset: isize,
    ) -> Result<Vec<PanelUnit>, TargetRefitError> {
        units
            .iter()
            .map(|unit| {
                let (values, _) = shifted(Self::column_of(unit.series.storage(), id)?, offset);
                let rows = unit.series.row_count();
                let trim = offset.unsigned_abs();
                let (start, end) =
                    if offset < 0 { (trim, rows) } else { (0, rows.saturating_sub(trim)) };
                Ok(PanelUnit {
                    unit_id: unit.unit_id,
                    series: Self::cut(
                        &series_with(&unit.series, &[(id, values)], None, None)?,
                        start,
                        end,
                    )?,
                })
            })
            .collect()
    }

    #[allow(
        clippy::too_many_lines,
        reason = "one exhaustive match over the perturbations of a panel"
    )]
    pub(super) fn panel_refit(
        &self,
        panel: &PanelData,
        perturbation: TargetPerturbation,
        rng: &mut CausalRng,
        ctx: &ExecutionContext,
    ) -> Result<Vec<f64>, TargetRefitError> {
        let units = panel.units();
        let rebuild = |units: Vec<PanelUnit>| PanelData::try_new(units).map_err(data_failed);
        let resample_units = |rng: &mut CausalRng, units: &[PanelUnit]| -> Vec<PanelUnit> {
            (0..units.len())
                .map(|slot| {
                    let source = &units[unbiased_index(rng, units.len())];
                    PanelUnit {
                        unit_id: u32::try_from(slot).unwrap_or(u32::MAX),
                        series: source.series.clone(),
                    }
                })
                .collect()
        };
        let rebuild_replicate = |rng: &mut CausalRng, units: Vec<PanelUnit>, replicate: u32| {
            if replicate == 0 { rebuild(units) } else { rebuild(resample_units(rng, &units)) }
        };
        let swap_all =
            |id: VariableId, from: VariableId| -> Result<Vec<PanelUnit>, TargetRefitError> {
                units
                    .iter()
                    .map(|unit| {
                        let values: Arc<[f64]> =
                            Arc::from(Self::column_of(unit.series.storage(), from)?);
                        Ok(PanelUnit {
                            unit_id: unit.unit_id,
                            series: series_with(&unit.series, &[(id, values)], None, None)?,
                        })
                    })
                    .collect()
            };
        let panel = match perturbation {
            TargetPerturbation::RandomCommonCause { .. } => {
                return self.random_common_cause(rng, ctx);
            }
            TargetPerturbation::PlaceboTreatment { .. }
            | TargetPerturbation::PlaceboOutcome { .. } => {
                let id = if matches!(perturbation, TargetPerturbation::PlaceboTreatment { .. }) {
                    self.treatment
                } else {
                    self.outcome
                };
                let equal =
                    units.iter().all(|u| u.series.row_count() == units[0].series.row_count());
                let mut out = Vec::with_capacity(units.len());
                if equal && units.len() > 1 {
                    // Whole paths permuted across units keep each path's own dependence.
                    let mut order: Vec<usize> = (0..units.len()).collect();
                    shuffle(rng, &mut order);
                    for (unit, &source) in units.iter().zip(&order) {
                        let values: Arc<[f64]> =
                            Arc::from(Self::column_of(units[source].series.storage(), id)?);
                        out.push(PanelUnit {
                            unit_id: unit.unit_id,
                            series: series_with(&unit.series, &[(id, values)], None, None)?,
                        });
                    }
                } else {
                    for unit in units {
                        let values =
                            Self::permuted(Self::column_of(unit.series.storage(), id)?, rng, true);
                        out.push(PanelUnit {
                            unit_id: unit.unit_id,
                            series: series_with(&unit.series, &[(id, values)], None, None)?,
                        });
                    }
                }
                rebuild(out)?
            }
            TargetPerturbation::Subset { fraction, .. } => {
                let mut order: Vec<usize> = (0..units.len()).collect();
                shuffle(rng, &mut order);
                #[allow(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "the subset fraction lies in (0, 1), and the saturating cast is clamped into [1, units]"
                )]
                let kept = ((fraction * units.len() as f64).round() as usize).clamp(1, units.len());
                let mut chosen: Vec<usize> = order[..kept].to_vec();
                chosen.sort_unstable();
                rebuild(chosen.into_iter().map(|i| units[i].clone()).collect())?
            }
            TargetPerturbation::Bootstrap { .. } => rebuild(resample_units(rng, units))?,
            TargetPerturbation::NegativeControlExposure { replicate } => {
                let swapped = swap_all(self.treatment, self.control_exposure()?)?;
                rebuild_replicate(rng, swapped, replicate)?
            }
            TargetPerturbation::NegativeControlOutcome { replicate } => {
                let swapped = if let Some(control) = self.options.negative_control_outcome {
                    swap_all(self.outcome, control)?
                } else {
                    let span =
                        isize::try_from(max_graph_lag(self.study()) + 1).unwrap_or(isize::MAX);
                    Self::shifted_panel_units(units, self.outcome, -span)?
                };
                rebuild_replicate(rng, swapped, replicate)?
            }
            TargetPerturbation::PreTrend { replicate, lead } => {
                let lead = isize::try_from(lead).unwrap_or(isize::MAX);
                let shifted_units = Self::shifted_panel_units(units, self.treatment, lead)?;
                rebuild_replicate(rng, shifted_units, replicate)?
            }
            TargetPerturbation::Anticipation { replicate } => {
                let back = isize::try_from(max_graph_lag(self.study()) + 1).unwrap_or(isize::MAX);
                let shifted_units = Self::shifted_panel_units(units, self.outcome, -back)?;
                rebuild_replicate(rng, shifted_units, replicate)?
            }
        };
        self.reestimate(&Modal::Panel(panel), ctx)
    }
}
