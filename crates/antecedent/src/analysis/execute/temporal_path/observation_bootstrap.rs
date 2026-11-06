//! Observation-adjusted temporal response bootstraps.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, AtomBandSource, CausalError, CausalResponse, ClassBandLayout, ClassBootstrapTargets,
    ClassObservationAtom, ClassObservationKind, Diagnostic, DiagnosticKind, DiagnosticSeverity,
    ExecutionContext, IdentifiedEstimand, ObservationBootstrap, ObservationBootstrapBand,
    ResponseIdentification, ResponseQuery, ResponseUncertainty, ResponseValue, SequenceMechanisms,
    TemporalIndexer, TimeSeriesData, TupleBandSource, TupleObservationTarget, TupleReplicates,
    TupleSurface, VariableId, contemporaneous_adjustment_variables,
    tuple_block_observation_replicates,
};

/// How a class-observation response reports its band: from the outer circular-block
/// bootstrap when a single completion identifies, otherwise withheld.
pub(super) fn class_observation_band_note(class_band_applied: bool) -> Diagnostic {
    Diagnostic::new(
        "estimate.temporal_class.observation_no_complete_band",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        if class_band_applied {
            "observation is adjusted per completion; complete-data bands are not reused; the \
             class band is the outer circular-block observation bootstrap of the single \
             identified completion"
        } else {
            "observation is adjusted per completion; complete-data bands are not reused and \
             joint observation/curve bands stay unavailable at the class level"
        },
    )
}

/// Pointwise band `center ± z·SD` of the (already fixed-b scaled) joint replicates.
///
/// Same success floor as the inner estimator's own joint bootstrap
/// (`temporal_response::bootstrap_surface`): at least two surviving replicates and at
/// most half of the attempted ones failed, or a cancelled run (even at the success
/// floor) cannot justify a reported band. Cancellation is not adaptive early-stop. This
/// is deliberately looser than [`super::super::PERCENTILE_95_BAND_MIN_SUCCESSES`], which gates
/// only the *simultaneous* band (see [`antecedent_estimate::TEMPORAL_RESPONSE_FEW_REPLICATES`]):
/// a pointwise band below that floor still publishes, flagged as resting on a noisy SD.
pub(super) fn summarize_observation_bootstrap(
    draws: &[Vec<f64>],
    center: &[f64],
    attempted: u32,
    cancelled: bool,
) -> ObservationBootstrapBand {
    let completed = u32::try_from(draws.len()).unwrap_or(u32::MAX);
    let mut band = ObservationBootstrapBand {
        lower: Vec::new(),
        upper: Vec::new(),
        draws: Vec::new(),
        completed,
        attempted,
        cancelled,
        block: None,
    };
    let enough_successes = completed >= 2 && completed >= attempted.saturating_sub(completed);
    if cancelled
        || !enough_successes
        || draws.iter().any(|draw| draw.len() != center.len())
        || center.iter().any(|value| !value.is_finite())
    {
        return band;
    }
    let z = crate::result::reported_se_interval_z();
    // One scalar per retained replicate draw, which the bootstrap already admitted.
    let mut values = Vec::with_capacity(draws.len());
    for (cell, &mid) in center.iter().enumerate() {
        values.clear();
        values.extend(draws.iter().map(|draw| draw[cell]));
        let sd = antecedent_stats::sample_std(&values);
        band.lower.push(mid - z * sd);
        band.upper.push(mid + z * sd);
    }
    band.draws = draws.to_vec();
    band
}

/// Full-sample point surface of a response, in the mean layout.
fn response_point_values(response: &CausalResponse) -> Option<Vec<f64>> {
    match &response.estimate {
        ResponseIdentification::PointIdentified(ResponseValue::Scalar(value)) => Some(vec![*value]),
        ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) => {
            Some(mean.to_vec())
        }
        // A single identified completion: the class set collapses to one surface.
        ResponseIdentification::PartiallyIdentified(ResponseValue::Envelope(envelope))
            if envelope.lower == envelope.upper =>
        {
            Some(envelope.lower.to_vec())
        }
        _ => None,
    }
}

pub(super) fn class_atom_adjustment(
    estimand: &IdentifiedEstimand,
    indexer: &TemporalIndexer,
) -> Vec<VariableId> {
    estimand
        .adjustment_set
        .iter()
        .filter_map(|&dense| indexer.key_of(dense.raw()).ok().map(|key| key.variable))
        .collect()
}

/// The joint circular-block tuple replicates the class atoms' observation
/// bands are summarized from.
fn class_tuple_replicates(
    source: &TimeSeriesData,
    atoms: &[ClassObservationAtom],
    options: antecedent_estimate::ObservationEstimatorOptions,
    replicates: u32,
    ctx: &ExecutionContext,
) -> Result<TupleReplicates, CausalError> {
    let structural_span = atoms
        .iter()
        .map(|atom| atom.indexer.history() as usize + atom.indexer.horizon() as usize)
        .max()
        .unwrap_or(1);
    // Every atom (curve or Sequence) resamples the same lag-aligned outcome-time tuples.
    let identifications: Vec<[(&IdentifiedEstimand, &TemporalIndexer); 1]> =
        atoms.iter().map(|atom| [(&atom.estimand, &atom.indexer)]).collect();
    let targets: Vec<TupleObservationTarget<'_>> = atoms
        .iter()
        .zip(&identifications)
        .map(|(atom, identification)| -> Result<TupleObservationTarget<'_>, CausalError> {
            Ok(TupleObservationTarget {
                query: &atom.observation_query,
                adjustment: &atom.adjustment,
                outcome_regressors: match &atom.kind {
                    ClassObservationKind::Curve => {
                        antecedent_estimate::temporal_curve_outcome_regressors(
                            &atom.observation_query,
                            identification,
                        )
                        .map_err(CausalError::from)?
                    }
                    ClassObservationKind::Sequence { dag, outcome, .. } => {
                        antecedent_estimate::temporal_sequence_outcome_regressors(dag, *outcome)
                    }
                },
                surface: match &atom.kind {
                    ClassObservationKind::Curve => {
                        TupleSurface::Curve { identifications: identification }
                    }
                    ClassObservationKind::Sequence {
                        dag,
                        overlays,
                        outcome,
                        outcome_offset,
                        status,
                    } => TupleSurface::Sequence {
                        graph: dag,
                        overlays,
                        outcome: *outcome,
                        horizons: vec![(&atom.estimand, &atom.indexer, *outcome_offset, *status)],
                    },
                },
            })
        })
        .collect::<Result<_, _>>()?;
    tuple_block_observation_replicates(
        source,
        &targets,
        structural_span,
        options,
        replicates,
        0x0C1A_5500,
        ctx,
    )
}

/// Withholds the class-level observation band on a multi-completion
/// identified set, recording the counts and why the atoms keep their own bands.
fn withhold_class_observation_band(
    response: &mut CausalResponse,
    diagnostics: &mut Vec<Diagnostic>,
    (replicates, attempted): (u32, u32),
    cancelled: bool,
) {
    response.support.warnings.retain(|warning| {
        warning.code.as_ref() != antecedent_estimate::TEMPORAL_RESPONSE_NO_BAND_METHOD
    });
    diagnostics.push(Diagnostic::new(
        "estimate.temporal_class.no_observation_class_band_method",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        "class-level observation band is withheld on a multi-completion identified set; \
         each completion atom retains its outer circular-block band; complete-data bands \
         are not reused",
    ));
    response.support.diagnostics.push(antecedent_core::SupportDiagnostic {
        id: Arc::from("response.observation_block_bootstrap"),
        values: Arc::from([
            f64::from(replicates),
            0.0,
            f64::from(u32::from(cancelled)),
            f64::from(attempted),
        ]),
        detail: Arc::from(
            "requested, completed class-band, cancellation, and attempted counts; the \
             class band is withheld; atoms retain per-completion circular-block bands",
        ),
        scope: antecedent_core::DiagnosticScope::Global,
    });
}

/// Writes each completion atom's observation circular-block band onto its
/// structural atom, building a point-only response for atoms without one.
fn apply_atom_observation_bands(
    atoms: &[ClassObservationAtom],
    structural_atoms: &mut [crate::result::StructuralResponseAtom],
    response: &CausalResponse,
    (tuple, atom_draws, (attempted, cancelled)): AtomBandSource<'_>,
    replicates: u32,
) {
    let points = &tuple.points;
    for (atom_i, atom) in atoms.iter().enumerate() {
        let band = summarize_observation_bootstrap(
            &atom_draws[atom_i],
            &points[atom_i],
            attempted,
            cancelled,
        )
        .with_block(tuple, atom_i);
        if let Some(atom_response) =
            structural_atoms.get_mut(atom.atom_index).and_then(|item| item.response.as_mut())
        {
            apply_observation_bootstrap(atom_response, &band, replicates);
        } else if let Some(structural) = structural_atoms.get_mut(atom.atom_index) {
            if let Some(value) = structural.value.clone() {
                let mut stub = CausalResponse {
                    estimand: response.estimand.clone(),
                    identification_status: structural.status,
                    estimate: ResponseIdentification::PointIdentified(value),
                    uncertainty: ResponseUncertainty::None,
                    support: response.support.clone(),
                    assumptions: atom.assumptions.clone(),
                    provenance_id: Arc::from("estimate.temporal_response.class_atom"),
                    horizon_identification: None,
                    interaction_structurally_zero: false,
                };
                apply_observation_bootstrap(&mut stub, &band, replicates);
                structural.response = Some(stub);
            }
        }
    }
}

pub(super) fn apply_class_observation_bootstrap(
    source: &TimeSeriesData,
    atoms: &[ClassObservationAtom],
    targets: ClassBootstrapTargets<'_>,
    layout: ClassBandLayout,
    bootstrap: ObservationBootstrap,
    ctx: &ExecutionContext,
) -> Result<bool, CausalError> {
    let ClassBootstrapTargets { structural_atoms, response, diagnostics } = targets;
    let ClassBandLayout { cells_per_horizon, n_horizons, apply_class_band } = layout;
    let ObservationBootstrap { options, replicates } = bootstrap;
    if replicates == 0 || atoms.is_empty() {
        return Ok(false);
    }
    let tuple = class_tuple_replicates(source, atoms, options, replicates, ctx)?;
    let mut atom_draws: Vec<Vec<Vec<f64>>> = vec![Vec::new(); atoms.len()];
    let mut class_draws: Vec<Vec<f64>> = Vec::new();
    let class_len = cells_per_horizon.saturating_mul(n_horizons);
    // Frozen-weight mix of per-atom surfaces into the class layout.
    let mix = |per_atom: &[Option<&Vec<f64>>]| -> Option<Vec<f64>> {
        let mut mixed = vec![0.0; class_len];
        let mut totals = vec![0.0; n_horizons];
        for (atom, values) in atoms.iter().zip(per_atom) {
            let values = (*values)?;
            if atom.horizon_index >= n_horizons || values.len() != cells_per_horizon {
                return None;
            }
            totals[atom.horizon_index] += atom.weight;
            for (cell, value) in values.iter().enumerate() {
                mixed[cell * n_horizons + atom.horizon_index] += atom.weight * value;
            }
        }
        if !totals.iter().all(|&total| total > 0.0) {
            return None;
        }
        for (index, value) in mixed.iter_mut().enumerate() {
            *value /= totals[index % n_horizons];
        }
        Some(mixed)
    };

    let (attempted, points) = (tuple.attempted, &tuple.points);
    for replicate_atom_values in &tuple.values {
        for (atom_i, values) in replicate_atom_values.iter().enumerate() {
            if let Some(values) = values {
                if atom_draws[atom_i].first().is_none_or(|first| first.len() == values.len()) {
                    atom_draws[atom_i].push(values.clone());
                }
            }
        }
        if apply_class_band && class_len > 0 {
            let per_atom: Vec<Option<&Vec<f64>>> =
                replicate_atom_values.iter().map(Option::as_ref).collect();
            class_draws.extend(mix(&per_atom));
        }
    }

    let cancelled = ctx.cancellation.is_cancelled();
    apply_atom_observation_bands(
        atoms,
        structural_atoms,
        response,
        (&tuple, &atom_draws, (attempted, cancelled)),
        replicates,
    );

    if apply_class_band {
        let center = mix(&points.iter().map(Some).collect::<Vec<_>>()).unwrap_or_default();
        let band = summarize_observation_bootstrap(&class_draws, &center, attempted, cancelled)
            .with_block_summary(&tuple);
        apply_observation_bootstrap(response, &band, replicates);
        Ok(!band.lower.is_empty())
    } else {
        withhold_class_observation_band(response, diagnostics, (replicates, attempted), cancelled);
        Ok(false)
    }
}

pub(super) fn apply_observation_bootstrap(
    response: &mut CausalResponse,
    bootstrap: &ObservationBootstrapBand,
    requested: u32,
) {
    apply_tuple_bootstrap_band(response, bootstrap, requested, TupleBandSource::Observation);
}

pub(super) fn apply_tuple_bootstrap_band(
    response: &mut CausalResponse,
    bootstrap: &ObservationBootstrapBand,
    requested: u32,
    source: TupleBandSource,
) {
    // A tuple bootstrap ran, so the estimator's zero-replicate withholding note (the
    // inner fit runs without replicates on these paths) no longer describes the band.
    response.support.warnings.retain(|warning| {
        warning.code.as_ref() != antecedent_estimate::TEMPORAL_RESPONSE_NO_BAND_METHOD
    });
    if bootstrap.lower.is_empty() {
        antecedent_estimate::publish_simultaneous_band(
            &mut response.support,
            Err(antecedent_estimate::EstimationError::unsupported(
                "the tuple bootstrap produced too few successful replicates",
            )),
            "",
        );
        response.support.warnings.push(Diagnostic::new(
            match source {
                TupleBandSource::Observation => "response.observation_bootstrap_insufficient",
                TupleBandSource::CompleteSequence => "response.sequence_bootstrap_insufficient",
            },
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            format!(
                "bootstrap band withheld: {} of {} attempted replicates succeeded; at least two \
                 successes and at most half failed attempts are required",
                bootstrap.completed, bootstrap.attempted
            ),
        ));
    } else {
        response.uncertainty = ResponseUncertainty::PointwiseBand {
            level: 0.95,
            lower: Arc::from(bootstrap.lower.clone()),
            upper: Arc::from(bootstrap.upper.clone()),
            interpretation: antecedent_core::IntervalInterpretation::Confidence,
            draws: None,
        };
        response.support.warnings.retain(|warning| {
            !matches!(
                warning.code.as_ref(),
                "response.observation_joint_uncertainty_unavailable"
                    | "response.temporal.sequence_uncertainty_unavailable"
            )
        });
        if let Some((block, inflation, dispersion)) = &bootstrap.block {
            antecedent_estimate::disclose_response_block_bootstrap(
                &mut response.support,
                &mut response.assumptions,
                *block,
                *inflation,
                dispersion,
                bootstrap.draws.len(),
                source.refit(),
            );
        }
        let simultaneous = response_point_values(response)
            .ok_or_else(|| {
                antecedent_estimate::EstimationError::unsupported(
                    "the response has no single point surface",
                )
            })
            .and_then(|center| {
                antecedent_estimate::max_deviation_band(&center, &bootstrap.draws, 0.95)
            });
        antecedent_estimate::publish_simultaneous_band(
            &mut response.support,
            simultaneous,
            source.construction(),
        );
    }
    response.support.diagnostics.push(antecedent_core::SupportDiagnostic {
        id: Arc::from(source.counts_id()),
        values: Arc::from([
            f64::from(requested),
            f64::from(bootstrap.completed),
            f64::from(bootstrap.cancelled),
            f64::from(bootstrap.attempted),
        ]),
        detail: Arc::from(source.counts_detail()),
        scope: antecedent_core::DiagnosticScope::Global,
    });
}

/// Outer circular-block bootstrap of an observation-adjusted curve / Set-Shift surface.
///
/// Resampling is tuple-level: each replicate draws circular blocks of outcome-time
/// anchors, refits the observation nuisance on those anchors' lag-aligned observation
/// rows, and refits every horizon on the same anchors' lag-aligned design rows with
/// the replicate pseudo-outcome. Reordering the raw series instead (and rebuilding lags
/// on the reordered rows) pairs values across block junctions and biases the replicate
/// slope toward zero.
pub(super) fn bootstrap_observation_adjusted_temporal_response(
    source: &TimeSeriesData,
    query: &ResponseQuery,
    aligned: &[&crate::analysis::prepared::CachedTemporalHorizonIdentification],
    bootstrap: ObservationBootstrap,
    ctx: &ExecutionContext,
) -> Result<Option<ObservationBootstrapBand>, CausalError> {
    let ObservationBootstrap { options, replicates } = bootstrap;
    if replicates == 0 {
        return Ok(None);
    }
    let adjustment = contemporaneous_adjustment_variables(aligned);
    let identifications =
        aligned.iter().map(|entry| (&entry.estimand, &entry.indexer)).collect::<Vec<_>>();
    let target = TupleObservationTarget {
        query,
        adjustment: &adjustment,
        outcome_regressors: antecedent_estimate::temporal_curve_outcome_regressors(
            query,
            &identifications,
        )
        .map_err(CausalError::from)?,
        surface: TupleSurface::Curve { identifications: &identifications },
    };
    single_target_observation_band(source, target, aligned, options, replicates, 0x0B5E_0000, ctx)
        .map(Some)
}

/// Joint circular-block tuple bootstrap of a Sequence overlay surface.
///
/// Each replicate resamples blocks of lag-aligned outcome-time tuples and refits every
/// unfolded sequential mechanism of every horizon on them through
/// [`antecedent_estimate::PreparedSequenceLevel`], recomputing root-node means. For an
/// observation-adjusted query, every replicate also refits the observation nuisance on
/// the resampled tuples and replaces the outcome-time column with the replicate
/// pseudo-outcomes (a design that reads the outcome at a nonzero lag is refused); a
/// complete-data query keeps its recorded outcomes.
pub(super) fn bootstrap_sequence_response(
    source: &TimeSeriesData,
    query: &ResponseQuery,
    mechanisms: SequenceMechanisms<'_>,
    outcome: VariableId,
    aligned: &[&crate::analysis::prepared::CachedTemporalHorizonIdentification],
    bootstrap: ObservationBootstrap,
    ctx: &ExecutionContext,
) -> Result<ObservationBootstrapBand, CausalError> {
    let SequenceMechanisms { graph, overlays } = mechanisms;
    let ObservationBootstrap { options, replicates } = bootstrap;
    let adjustment = contemporaneous_adjustment_variables(aligned);
    let horizons = aligned
        .iter()
        .map(|entry| {
            (
                &entry.estimand,
                &entry.indexer,
                i32::try_from(entry.horizon.saturating_sub(1)).unwrap_or(i32::MAX),
                entry.identification.status,
            )
        })
        .collect();
    let target = TupleObservationTarget {
        query,
        adjustment: &adjustment,
        outcome_regressors: antecedent_estimate::temporal_sequence_outcome_regressors(
            graph, outcome,
        ),
        surface: TupleSurface::Sequence { graph, overlays, outcome, horizons },
    };
    single_target_observation_band(source, target, aligned, options, replicates, 0x0B5E_5E00, ctx)
}

fn single_target_observation_band(
    source: &TimeSeriesData,
    target: TupleObservationTarget<'_>,
    aligned: &[&crate::analysis::prepared::CachedTemporalHorizonIdentification],
    options: antecedent_estimate::ObservationEstimatorOptions,
    replicates: u32,
    stream: u64,
    ctx: &ExecutionContext,
) -> Result<ObservationBootstrapBand, CausalError> {
    let structural_span = aligned
        .iter()
        .map(|entry| entry.indexer.history() as usize + entry.indexer.horizon() as usize)
        .max()
        .unwrap_or(1);
    let mut tuple = tuple_block_observation_replicates(
        source,
        &[target],
        structural_span,
        options,
        replicates,
        stream,
        ctx,
    )?;
    let draws: Vec<Vec<f64>> = std::mem::take(&mut tuple.values)
        .into_iter()
        .filter_map(|mut values| values.pop().flatten())
        .collect();
    let center = tuple.points.pop().unwrap_or_default();
    Ok(summarize_observation_bootstrap(
        &draws,
        &center,
        tuple.attempted,
        ctx.cancellation.is_cancelled(),
    )
    .with_block(&tuple, 0))
}
