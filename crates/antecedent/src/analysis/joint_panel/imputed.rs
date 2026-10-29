//! Multiple imputation of missing outcomes and confounders in a joint-panel source.
//!
//! The follow-up rows of every unit are stacked into one table whose columns are the
//! assignment channels with their one- and two-period lags (always observed), the season
//! levels, and the outcome and confounder with their missing cells. The proper imputer of
//! `antecedent-estimate` (chained equations, parameter draws that resample whole units, one
//! neighbouring step of serial context) draws `m` completed tables. Each completed panel is
//! prepared and fitted exactly as an observed one, and the `m` response vectors are pooled
//! with Rubin's rules (the within-imputation covariance from the fit's own uncertainty draws),
//! or stacked when the fits are Bayesian.
//!
//! Missing assignments and season levels are not imputed: an assignment is the intervention
//! the response is about, and a season level is a design label.
//!
//! Serial context is one neighbouring step of the stacked table, so the first and last
//! follow-up row of a unit see a row of the adjacent unit as their neighbour. That affects
//! two rows per unit and is the only departure of the imputation model from the panel's own
//! within-unit serial law.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
use super::{JointPanelError, JointPanelRequest, Study};
use antecedent_core::ExecutionContext;
use antecedent_data::{ImputationSpec, PanelData, PanelUnit, SamplingRegularity, TableView};
use antecedent_estimate::joint_panel::PanelInference;
use antecedent_estimate::multiple_imputation::{
    self, ImputationColumn, ImputationUnits, Imputer, MultipleImputationFailure,
    PooledVectorEstimate, VectorImputationFit,
};
use std::sync::Arc;

/// A joint-panel source whose missing outcomes and confounders are multiply imputed.
#[derive(Clone, Debug)]
pub struct PreparedJointPanelImputation {
    source: PanelData,
    request: JointPanelRequest,
    imputer: Imputer,
    /// Per unit: its source rows and the follow-up rows stacked into the imputation table.
    layout: Arc<[(usize, usize)]>,
}

/// The pooled response of every declared program, with the evidence of how it was pooled.
#[derive(Clone, Debug)]
pub struct JointPanelImputedResult {
    /// Program identity of every world, in request order.
    pub programs: Vec<String>,
    /// Periods of every program; the pooled vector is the concatenation of the programs'
    /// mean responses in this order.
    pub horizons: Vec<usize>,
    /// Rubin's rules over frequentist fits, or the stacked posterior of Bayesian fits.
    pub pooled: PooledVectorEstimate,
    /// The point response of every imputation, in imputation order.
    pub fits: Vec<Vec<f64>>,
    /// Cells imputed per imputation-model variable.
    pub imputed_cells: Vec<(Arc<str>, usize)>,
}

impl Study {
    /// Prepare a multiple-imputation analysis of a panel with missing outcomes or confounders.
    ///
    /// The request's inference must carry uncertainty draws (a frequentist draw count of at
    /// least two, or a Bayesian sampler): they are the within-imputation uncertainty that
    /// Rubin's rules combine with the between-imputation spread.
    ///
    /// # Errors
    /// A missing assignment or season cell, an analysis mask, a time-varying declaration, a
    /// request without uncertainty draws, or an imputation model that cannot identify the
    /// missing variable.
    pub fn prepare_joint_panel_imputed(
        data: PanelData,
        request: JointPanelRequest,
        spec: ImputationSpec,
        _ctx: &ExecutionContext,
    ) -> Result<PreparedJointPanelImputation, JointPanelError> {
        let columns = &request.columns;
        let prefix = request.lag_order;
        match &request.options.inference {
            PanelInference::Frequentist { draws }
            | PanelInference::FrequentistBootstrap { draws }
                if *draws >= 2 => {}
            PanelInference::Bayesian { .. } => {}
            _ => {
                return Err(JointPanelError::compile(
                    "pooling imputations needs each fit's uncertainty draws: declare at least two frequentist draws or a Bayesian sampler",
                ));
            }
        }
        if request.temporal.is_some() {
            return Err(JointPanelError::Unsupported(
                "multiple imputation is not carried through a time-varying mechanism's selection procedure".into(),
            ));
        }
        if data.units().iter().any(|unit| unit.series.storage().analysis_mask().is_some()) {
            return Err(JointPanelError::Unsupported(
                "an analysis mask hides rows from the imputation model and the fit alike; declare the missing-data policy instead of imputing a masked panel".into(),
            ));
        }
        let mut names: Vec<Arc<str>> = Vec::new();
        let mut table: Vec<Vec<f64>> = Vec::new();
        let mut layout = Vec::with_capacity(data.units().len());
        let mut groups: Vec<u32> = Vec::new();
        let mut push = |name: String, values: Vec<f64>| {
            names.push(Arc::from(name));
            table.push(values);
        };
        let channel_count = columns.channels.len();
        let lags = if prefix == 2 { 2 } else { 0 };
        let mut channel_columns: Vec<Vec<f64>> = vec![Vec::new(); channel_count * (1 + lags)];
        let mut season_columns: Vec<Vec<f64>> = vec![Vec::new(); request.seasons.saturating_sub(1)];
        let (mut outcome, mut confounder) = (Vec::new(), Vec::new());
        for (position, unit) in data.units().iter().enumerate() {
            let rows = unit.series.row_count();
            if rows < prefix + 3 {
                return Err(JointPanelError::compile(
                    "panel needs observed prehistory and at least three follow-up rows",
                ));
            }
            for (k, &channel) in columns.channels.iter().enumerate() {
                let values = unit.series.float64_cow(channel).map_err(JointPanelError::compile)?;
                for lag in 0..=lags {
                    channel_columns[k * (1 + lags) + lag]
                        .extend((prefix..rows).map(|row| values[row - lag]));
                }
                if values[..].iter().any(|value| !value.is_finite()) {
                    return Err(JointPanelError::compile(
                        "a missing assignment is the intervention itself and is not imputed",
                    ));
                }
            }
            let season =
                unit.series.float64_cow(columns.season).map_err(JointPanelError::compile)?;
            if season[prefix..].iter().any(|value| !value.is_finite()) {
                return Err(JointPanelError::compile(
                    "a missing season level is a design label and is not imputed",
                ));
            }
            for (level, column) in season_columns.iter_mut().enumerate() {
                column.extend(
                    season[prefix..]
                        .iter()
                        .map(|value| f64::from(u8::from(*value == (level + 1) as f64))),
                );
            }
            let y = unit.series.float64_cow(columns.outcome).map_err(JointPanelError::compile)?;
            let z =
                unit.series.float64_cow(columns.confounder).map_err(JointPanelError::compile)?;
            outcome.extend_from_slice(&y[prefix..]);
            confounder.extend_from_slice(&z[prefix..]);
            let follow = rows - prefix;
            layout.push((rows, follow));
            groups.extend(std::iter::repeat_n(u32::try_from(position).unwrap_or(u32::MAX), follow));
        }
        for (k, &channel) in columns.channels.iter().enumerate() {
            for lag in 0..=lags {
                push(
                    format!("assignment_{}_lag{lag}", channel.raw()),
                    std::mem::take(&mut channel_columns[k * (1 + lags) + lag]),
                );
            }
        }
        for (level, column) in season_columns.into_iter().enumerate() {
            if column.iter().any(|v| *v != column[0]) {
                push(format!("season_{}", level + 1), column);
            }
        }
        push("outcome".to_owned(), outcome);
        push("confounder".to_owned(), confounder);
        let imputation_columns: Vec<ImputationColumn> = names
            .into_iter()
            .zip(table)
            .map(|(name, values)| ImputationColumn::new(name, values))
            .collect();
        let imputer =
            Imputer::new(imputation_columns, spec, ImputationUnits::Groups(groups.into()))
                .and_then(|imputer| imputer.with_dynamics(1))
                .map_err(|error| JointPanelError::compile(error.to_string()))?;
        Ok(PreparedJointPanelImputation { source: data, request, imputer, layout: layout.into() })
    }
}

impl PreparedJointPanelImputation {
    /// Impute, fit every completed panel and pool the response vectors.
    ///
    /// # Errors
    /// An imputation or fit refusal on any completed table, or fits that cannot be pooled.
    pub fn estimate(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<JointPanelImputedResult, JointPanelError> {
        let columns = &self.request.columns;
        let prefix = self.request.lag_order;
        let run = multiple_imputation::run_vector(&self.imputer, ctx, |table, ctx| {
            let outcome = table
                .column("outcome")
                .ok_or_else(|| JointPanelError::compile("no completed outcome"))?;
            let confounder = table
                .column("confounder")
                .ok_or_else(|| JointPanelError::compile("no completed confounder"))?;
            let mut offset = 0;
            let mut units = Vec::with_capacity(self.layout.len());
            for (unit, &(rows, follow)) in self.source.units().iter().zip(self.layout.iter()) {
                let mut y = unit
                    .series
                    .float64_cow(columns.outcome)
                    .map_err(JointPanelError::compile)?
                    .into_owned();
                let mut z = unit
                    .series
                    .float64_cow(columns.confounder)
                    .map_err(JointPanelError::compile)?
                    .into_owned();
                y[prefix..rows].copy_from_slice(&outcome[offset..offset + follow]);
                z[prefix..rows].copy_from_slice(&confounder[offset..offset + follow]);
                offset += follow;
                units.push(PanelUnit {
                    unit_id: unit.unit_id,
                    series: rebuilt(unit, columns, &y, &z)?,
                });
            }
            let completed =
                PanelData::try_new(Arc::from(units)).map_err(JointPanelError::compile)?;
            let prepared = Study::prepare_joint_panel(completed, self.request.clone(), ctx)?;
            let result = prepared.estimate(ctx)?;
            let point: Vec<f64> =
                result.point.iter().flat_map(|response| response.means.iter().copied()).collect();
            let p = point.len();
            let draws: Vec<&Vec<_>> = result.draws.iter().flatten().collect();
            if matches!(self.request.options.inference, PanelInference::Bayesian { .. }) {
                let mut per_coordinate = vec![Vec::with_capacity(draws.len()); p];
                for draw in &draws {
                    let flat: Vec<f64> =
                        draw.iter().flat_map(|response| response.means.iter().copied()).collect();
                    for (slot, value) in per_coordinate.iter_mut().zip(flat) {
                        slot.push(value);
                    }
                }
                return Ok(VectorImputationFit::bayesian(per_coordinate));
            }
            if draws.len() < 2 {
                return Err(JointPanelError::compile(
                    "a fit retained fewer than two uncertainty draws",
                ));
            }
            let flat: Vec<Vec<f64>> = draws
                .iter()
                .map(|draw| {
                    draw.iter().flat_map(|response| response.means.iter().copied()).collect()
                })
                .collect();
            let n = flat.len() as f64;
            let mean: Vec<f64> =
                (0..p).map(|j| flat.iter().map(|row| row[j]).sum::<f64>() / n).collect();
            let mut covariance = vec![0.0; p * p];
            for row in &flat {
                for i in 0..p {
                    for j in 0..p {
                        covariance[i * p + j] +=
                            (row[i] - mean[i]) * (row[j] - mean[j]) / (n - 1.0);
                    }
                }
            }
            Ok(VectorImputationFit::frequentist(point, covariance, None))
        })
        .map_err(|failure| match failure {
            MultipleImputationFailure::Imputation(error) => {
                JointPanelError::compile(error.to_string())
            }
            MultipleImputationFailure::Fit { error, .. } => error,
        })?;
        Ok(JointPanelImputedResult {
            programs: self.request.worlds.iter().map(|world| world.program.id.clone()).collect(),
            horizons: self.request.worlds.iter().map(super::JointPanelWorld::horizon).collect(),
            pooled: run.pooled,
            fits: run.fits.iter().map(|fit| fit.point.clone()).collect(),
            imputed_cells: self
                .imputer
                .variables()
                .into_iter()
                .filter(|variable| variable.missing > 0)
                .map(|variable| (variable.name, variable.missing))
                .collect(),
        })
    }
}

/// One unit's series with its outcome and confounder replaced, every other column kept.
fn rebuilt(
    unit: &PanelUnit,
    columns: &super::JointPanelColumns,
    y: &[f64],
    z: &[f64],
) -> Result<antecedent_data::TimeSeriesData, JointPanelError> {
    let interval = match unit.series.time_index().regularity {
        SamplingRegularity::Regular { interval_ns } if interval_ns > 0 => interval_ns,
        _ => {
            return Err(JointPanelError::compile(
                "panel requires explicit positive regular cadence",
            ));
        }
    };
    let schema = unit.series.schema();
    let mut values: Vec<(String, Vec<f64>)> = Vec::new();
    for variable in schema.variables() {
        let column = if variable.id == columns.outcome {
            y.to_vec()
        } else if variable.id == columns.confounder {
            z.to_vec()
        } else {
            unit.series.float64_cow(variable.id).map_err(JointPanelError::compile)?.into_owned()
        };
        values.push((variable.name.to_string(), column));
    }
    let mut series = antecedent_data::TimeSeriesData::from_f64_columns(
        values.iter().map(|(name, column)| (name.as_str(), column.as_slice())),
        interval,
    )
    .map_err(JointPanelError::compile)?;
    if let Some(weights) = unit.series.storage().weights() {
        series = series.with_weights(Arc::from(weights)).map_err(JointPanelError::compile)?;
    }
    Ok(series)
}
