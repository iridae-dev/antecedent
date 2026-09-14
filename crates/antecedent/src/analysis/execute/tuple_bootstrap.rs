// Observation / Sequence tuple-level circular-block bootstrap.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

/// `values[replicate][target]` joint replicate surfaces (`None` where a target's
/// refit failed), the number of attempted replicates, and each target's full-sample
/// surface.
pub type TupleReplicates = (Vec<Vec<Option<Vec<f64>>>>, u32, Vec<Vec<f64>>);

/// What a [`TupleObservationTarget`] refits on every replicate.
pub enum TupleSurface<'a> {
    /// Curve / single Set-Shift: one identified estimand per horizon of the query.
    Curve {
        /// One identified estimand per horizon of the query.
        identifications: &'a [(&'a IdentifiedEstimand, &'a TemporalIndexer)],
    },
    /// Sequence overlays on the unfolded sequential engine, one level per horizon.
    Sequence {
        graph: &'a TemporalDag,
        overlays: &'a [antecedent_estimate::SequentialMechanismOverlay],
        outcome: VariableId,
        /// `(estimand, indexer, outcome offset, identification status)` per horizon.
        horizons: Vec<(&'a IdentifiedEstimand, &'a TemporalIndexer, i32, IdentificationStatus)>,
    },
}

/// One surface refit by [`tuple_block_observation_replicates`].
pub struct TupleObservationTarget<'a> {
    /// Observation-bearing response query (its own horizons).
    pub(super) query: &'a ResponseQuery,
    /// Contemporaneous causal adjustment variables for the containment check.
    pub(super) adjustment: &'a [VariableId],
    /// Downstream design columns the selected-AIPW outcome nuisance conditions on
    /// (offsets relative to the outcome time).
    pub(super) outcome_regressors: Vec<antecedent_core::TemporalNodeKey>,
    /// Surface refit on the resampled tuples.
    pub(super) surface: TupleSurface<'a>,
}

/// A [`TupleObservationTarget`] fitted once on the full observation-adjusted series.
pub enum PreparedTupleTarget {
    Curve(antecedent_estimate::PreparedTemporalSurface),
    Sequence {
        levels: Vec<antecedent_estimate::PreparedSequenceLevel>,
        outcome: VariableId,
        /// Distinct lags at which the outcome enters any horizon's tuple.
        outcome_lags: Vec<u32>,
    },
}

impl PreparedTupleTarget {
    fn point(&self) -> Vec<f64> {
        match self {
            Self::Curve(surface) => surface.point(),
            Self::Sequence { levels, .. } => {
                levels.iter().map(antecedent_estimate::PreparedSequenceLevel::point).collect()
            }
        }
    }

    fn parameters(&self) -> usize {
        match self {
            Self::Curve(surface) => surface.max_parameters(),
            Self::Sequence { levels, .. } => levels
                .iter()
                .map(antecedent_estimate::PreparedSequenceLevel::max_parameters)
                .max()
                .unwrap_or(0),
        }
    }

    /// Earliest anchor whose tuple and every lagged observation row exist.
    fn first_anchor(&self, observation_lag: usize) -> usize {
        match self {
            Self::Curve(surface) => surface.first_common_anchor().max(observation_lag),
            Self::Sequence { levels, outcome_lags, .. } => levels
                .iter()
                .map(antecedent_estimate::PreparedSequenceLevel::first_anchor)
                .max()
                .unwrap_or(0)
                .max(outcome_lags.iter().max().map_or(0, |&lag| lag as usize) + observation_lag),
        }
    }
}

/// Shared tuple-level outer bootstrap: every target is refit on the same resampled
/// anchors in each replicate, so per-target draws are jointly distributed. Replicate
/// deviations from each target's full-sample surface carry the response family's
/// fixed-b dispersion factor ([`antecedent_estimate::block_dispersion_inflation`]).
pub fn tuple_block_observation_replicates(
    source: &TimeSeriesData,
    targets: &[TupleObservationTarget<'_>],
    structural_span: usize,
    options: antecedent_estimate::ObservationEstimatorOptions,
    replicates: u32,
    stream: u64,
    ctx: &ExecutionContext,
) -> Result<TupleReplicates, CausalError> {
    let observation = ObservationMechanismEstimator::new(options);
    let mut prepared = Vec::with_capacity(targets.len());
    let mut first_anchor = 0usize;
    for target in targets {
        // A complete-data target (Sequence only) refits on the source tuples with no
        // observation nuisance; an observation-bearing one on the adjusted series.
        let complete = target.query.observation == ObservationSpec::Complete;
        let adjusted_owned;
        let fit_data = if complete {
            source
        } else {
            adjusted_owned = observation
                .adjust_temporal_series(
                    source,
                    target.query,
                    target.adjustment,
                    &target.outcome_regressors,
                )
                .map_err(CausalError::from)?
                .0;
            &adjusted_owned
        };
        // Rows back from the outcome time that one observation row reads: the declared
        // conditioning set at the policy offset and every outcome-model regressor.
        let observation_lag = if complete {
            0
        } else {
            observation
                .observation_row_lag(target.query, &target.outcome_regressors)
                .map_err(CausalError::from)?
        };
        let fitted = match &target.surface {
            TupleSurface::Curve { .. } if complete => {
                return Err(CausalError::Unsupported {
                    message: "complete-data curves use the surface estimator's joint bootstrap",
                });
            }
            TupleSurface::Curve { identifications } => {
                let mut working = target.query.clone();
                working.observation = ObservationSpec::Complete;
                working.observation_assumptions = Arc::from([]);
                PreparedTupleTarget::Curve(
                    TemporalResponseEstimator::new()
                        .prepare_surface(fit_data, identifications, &working, ctx)
                        .map_err(CausalError::from)?,
                )
            }
            TupleSurface::Sequence { graph, overlays, outcome, horizons } => {
                let levels = horizons
                    .iter()
                    .map(|&(estimand, indexer, outcome_offset, status)| {
                        antecedent_estimate::prepare_sequence_level(
                            fit_data,
                            graph,
                            indexer,
                            estimand,
                            *outcome,
                            outcome_offset,
                            overlays,
                            status,
                            ctx,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(CausalError::from)?;
                // Complete data: nothing to refit, the stored outcome columns stand.
                let mut outcome_lags: Vec<u32> = if complete {
                    Vec::new()
                } else {
                    levels.iter().flat_map(|level| level.lags_of(*outcome)).collect()
                };
                outcome_lags.sort_unstable();
                outcome_lags.dedup();
                PreparedTupleTarget::Sequence { levels, outcome: *outcome, outcome_lags }
            }
        };
        first_anchor = first_anchor.max(fitted.first_anchor(observation_lag));
        prepared.push(fitted);
    }
    let m = source.row_count().saturating_sub(first_anchor);
    let block = antecedent_estimate::temporal_block_length(structural_span, m);
    let parameters = prepared.iter().map(PreparedTupleTarget::parameters).max().unwrap_or(0);
    let inflation = antecedent_estimate::block_dispersion_inflation(m, block, parameters);
    let points: Vec<Vec<f64>> = prepared.iter().map(PreparedTupleTarget::point).collect();
    let mut out = Vec::new();
    let mut attempted = 0u32;
    let mut positions = Vec::with_capacity(m);
    let mut anchors = Vec::with_capacity(m);
    let mut shifted = Vec::with_capacity(m);
    let mut by_lag: Vec<Vec<f64>> = Vec::new();
    for replicate in 0..replicates {
        if ctx.cancellation.is_cancelled() || m < 3 {
            break;
        }
        attempted += 1;
        let mut rng = ctx.rng.stream(stream + u64::from(replicate));
        antecedent_estimate::circular_block_positions_into(m, block, &mut rng, &mut positions);
        anchors.clear();
        anchors.extend(positions.iter().map(|&position| first_anchor + position));
        let values = targets
            .iter()
            .zip(&prepared)
            .zip(&points)
            .map(|((target, fitted), point)| {
                let values = match fitted {
                    PreparedTupleTarget::Curve(surface) => {
                        let outcomes = observation
                            .adjust_temporal_anchors(
                                source,
                                target.query,
                                target.adjustment,
                                &target.outcome_regressors,
                                &anchors,
                            )
                            .ok()?;
                        surface.replicate(&anchors, &outcomes).ok()??
                    }
                    PreparedTupleTarget::Sequence { levels, outcome, outcome_lags } => {
                        // One observation refit per lag at which the outcome enters the
                        // tuple, on the same blocks shifted to that lag.
                        by_lag.resize_with(outcome_lags.len(), Vec::new);
                        for (slot, &lag) in by_lag.iter_mut().zip(outcome_lags.iter()) {
                            shifted.clear();
                            shifted.extend(anchors.iter().map(|&anchor| anchor - lag as usize));
                            *slot = observation
                                .adjust_temporal_anchors(
                                    source,
                                    target.query,
                                    target.adjustment,
                                    &target.outcome_regressors,
                                    &shifted,
                                )
                                .ok()?;
                        }
                        let replacements: Vec<antecedent_estimate::SequenceColumnReplacement<'_>> =
                            outcome_lags
                                .iter()
                                .zip(&by_lag)
                                .map(|(&lag, values)| {
                                    antecedent_estimate::SequenceColumnReplacement {
                                        variable: *outcome,
                                        lag,
                                        values,
                                    }
                                })
                                .collect();
                        levels
                            .iter()
                            .map(|level| level.replicate(&anchors, &replacements).ok()?)
                            .collect::<Option<Vec<f64>>>()?
                    }
                };
                if !values.iter().all(|value| value.is_finite()) {
                    return None;
                }
                let mut draw = [values];
                antecedent_estimate::inflate_replicates(point, &mut draw, inflation);
                let [values] = draw;
                Some(values)
            })
            .collect();
        out.push(values);
    }
    Ok((out, attempted, points))
}
