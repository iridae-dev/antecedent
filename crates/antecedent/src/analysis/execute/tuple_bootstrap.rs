// Observation / Sequence tuple-level circular-block bootstrap.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

/// Joint replicate surfaces of every target of one tuple-level bootstrap.
pub struct TupleReplicates {
    /// `values[replicate][target]` (`None` where a target's refit failed).
    pub values: Vec<Vec<Option<Vec<f64>>>>,
    /// Replicates attempted.
    pub attempted: u32,
    /// Each target's full-sample surface.
    pub points: Vec<Vec<f64>>,
    /// Block length of the resample and how it was chosen.
    pub block: antecedent_estimate::ResponseBlockLength,
    /// Fixed-b × HC1 dispersion factor applied to every replicate deviation.
    pub inflation: f64,
    /// Each target's per-cell kernel-bias factors and effective rows, applied to that
    /// target's replicate deviations after `inflation`.
    pub dispersion: Vec<antecedent_estimate::CellDispersion>,
}

impl TupleReplicates {
    /// The dispersion readings of every target joined into one band's disclosure: the
    /// largest kernel-bias factor and the fewest effective rows over all targets, as a
    /// one-cell summary (a mixed class band has no single per-cell layout).
    #[must_use]
    pub fn summary_dispersion(&self) -> antecedent_estimate::CellDispersion {
        antecedent_estimate::CellDispersion {
            kernel_factors: vec![
                self.dispersion
                    .iter()
                    .map(antecedent_estimate::CellDispersion::max_factor)
                    .fold(1.0, f64::max),
            ],
            effective_rows: vec![
                self.dispersion
                    .iter()
                    .map(antecedent_estimate::CellDispersion::min_effective_rows)
                    .filter(|rows| rows.is_finite())
                    .fold(f64::NAN, f64::min),
            ],
        }
    }
}

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
        /// Whether the outcome-time column is refit from the observation correction
        /// (false for a complete-data target, whose stored outcome stands).
        observation_adjusted: bool,
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

    /// Estimating-equation scores of the target's full-sample level.
    fn scores(&self) -> Vec<Vec<f64>> {
        match self {
            Self::Curve(surface) => surface.estimating_scores(),
            Self::Sequence { levels, .. } => levels
                .iter()
                .flat_map(antecedent_estimate::PreparedSequenceLevel::estimating_scores)
                .collect(),
        }
    }

    /// Per-cell kernel-bias factors and effective rows at block length `block`: the
    /// delta-method influence of every curve cell; for a Sequence level, whose
    /// g-computed level composes several mechanisms, the largest factor and fewest rows
    /// over that level's estimating scores.
    fn dispersion(&self, block: usize) -> antecedent_estimate::CellDispersion {
        match self {
            Self::Curve(surface) => surface.cell_dispersion(block),
            Self::Sequence { levels, .. } => {
                let mut out = antecedent_estimate::CellDispersion::default();
                for level in levels {
                    let scores = level.estimating_scores();
                    let refs: Vec<&[f64]> = scores.iter().map(Vec::as_slice).collect();
                    out.extend(antecedent_estimate::CellDispersion::from_scores(&refs, block));
                }
                out
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
            Self::Sequence { levels, .. } => levels
                .iter()
                .map(antecedent_estimate::PreparedSequenceLevel::first_anchor)
                .max()
                .unwrap_or(0)
                .max(observation_lag),
        }
    }
}

/// Shared tuple-level outer bootstrap: every target is refit on the same resampled
/// anchors in each replicate, so per-target draws are jointly distributed. Blocks follow
/// the response family's rule ([`antecedent_estimate::ResponseBlockLength`], lengthened
/// by every target's full-sample estimating scores), and replicate deviations from
/// each target's full-sample surface carry its fixed-b dispersion factor
/// ([`antecedent_estimate::block_dispersion_inflation`]) and each cell's kernel-bias
/// factor ([`antecedent_estimate::CellDispersion`]).
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
                // The observation correction replaces only the outcome-time column. A
                // lagged outcome regressor would be a pseudo-outcome (errors in
                // variables); the full-sample correction refuses that design too.
                if !complete
                    && levels
                        .iter()
                        .any(|level| level.lags_of(*outcome).iter().any(|&lag| lag != 0))
                {
                    return Err(CausalError::Unsupported {
                        message: antecedent_estimate::LAGGED_OUTCOME_REGRESSOR_REFUSAL,
                    });
                }
                PreparedTupleTarget::Sequence {
                    levels,
                    outcome: *outcome,
                    observation_adjusted: !complete,
                }
            }
        };
        first_anchor = first_anchor.max(fitted.first_anchor(observation_lag));
        prepared.push(fitted);
    }
    let m = source.row_count().saturating_sub(first_anchor);
    let scores: Vec<Vec<f64>> = prepared.iter().flat_map(PreparedTupleTarget::scores).collect();
    let score_refs: Vec<&[f64]> = scores.iter().map(Vec::as_slice).collect();
    let block_length =
        antecedent_estimate::ResponseBlockLength::new(structural_span, m, &score_refs);
    let block = block_length.length;
    let parameters = prepared.iter().map(PreparedTupleTarget::parameters).max().unwrap_or(0);
    let inflation = antecedent_estimate::block_dispersion_inflation(m, block, parameters);
    let dispersion: Vec<antecedent_estimate::CellDispersion> =
        prepared.iter().map(|fitted| fitted.dispersion(block)).collect();
    let points: Vec<Vec<f64>> = prepared.iter().map(PreparedTupleTarget::point).collect();
    let mut out = Vec::new();
    let mut attempted = 0u32;
    let mut positions = Vec::with_capacity(m);
    let mut anchors = Vec::with_capacity(m);
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
            .zip(points.iter().zip(&dispersion))
            .map(|((target, fitted), (point, cell_dispersion))| {
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
                    PreparedTupleTarget::Sequence { levels, outcome, observation_adjusted } => {
                        // The observation nuisance is refit on the replicate's outcome-time
                        // rows; the pseudo-outcomes replace the lag-0 outcome column.
                        let outcomes = if *observation_adjusted {
                            observation
                                .adjust_temporal_anchors(
                                    source,
                                    target.query,
                                    target.adjustment,
                                    &target.outcome_regressors,
                                    &anchors,
                                )
                                .ok()?
                        } else {
                            Vec::new()
                        };
                        let replacement = antecedent_estimate::SequenceColumnReplacement {
                            variable: *outcome,
                            lag: 0,
                            values: &outcomes,
                        };
                        let replacements = if *observation_adjusted {
                            std::slice::from_ref(&replacement)
                        } else {
                            &[]
                        };
                        levels
                            .iter()
                            .map(|level| level.replicate(&anchors, replacements).ok()?)
                            .collect::<Option<Vec<f64>>>()?
                    }
                };
                if !values.iter().all(|value| value.is_finite()) {
                    return None;
                }
                let mut draw = [values];
                antecedent_estimate::inflate_replicates(point, &mut draw, inflation);
                cell_dispersion.inflate(point, &mut draw);
                let [values] = draw;
                Some(values)
            })
            .collect();
        out.push(values);
    }
    Ok(TupleReplicates {
        values: out,
        attempted,
        points,
        block: block_length,
        inflation,
        dispersion,
    })
}
