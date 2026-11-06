//! Aggregation of response values, uncertainty, and influence.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, CausalError, EffectEstimate, IdentificationStatus, ObservationSpec, ResponseFunctional,
    ResponseIdentification, ResponseQuery, ResponseUncertainty, ResponseValue,
    scalar_identified_set,
};

pub(in crate::analysis::execute) fn response_identified_value(
    response: &antecedent_core::CausalResponse,
) -> Option<ResponseValue> {
    match &response.estimate {
        ResponseIdentification::PointIdentified(value)
        | ResponseIdentification::PartiallyIdentified(value) => Some(value.clone()),
        ResponseIdentification::GraphDependent(_) | ResponseIdentification::Unidentified { .. } => {
            None
        }
    }
}

pub(in crate::analysis::execute) fn response_envelope_from_weighted(
    weighted: &[(u64, f64, antecedent_core::CausalResponse)],
) -> Option<antecedent_core::ResponseEnvelope> {
    let values = weighted
        .iter()
        .filter_map(|(_, _, response)| response_identified_value(response))
        .collect::<Vec<_>>();
    let first = values.first()?;
    match first {
        ResponseValue::Scalar(value) => {
            let mut lower = *value;
            let mut upper = *value;
            for candidate in values.iter().skip(1) {
                let ResponseValue::Scalar(candidate) = candidate else {
                    return None;
                };
                lower = lower.min(*candidate);
                upper = upper.max(*candidate);
            }
            Some(scalar_identified_set(lower, upper))
        }
        ResponseValue::Surface { grid, dimension, mean } => {
            let mut lower = mean.to_vec();
            let mut upper = mean.to_vec();
            for candidate in values.iter().skip(1) {
                let ResponseValue::Surface {
                    grid: candidate_grid,
                    dimension: candidate_dimension,
                    mean: candidate_mean,
                } = candidate
                else {
                    return None;
                };
                if candidate_grid.as_ref() != grid.as_ref()
                    || candidate_dimension != dimension
                    || candidate_mean.len() != mean.len()
                {
                    return None;
                }
                for (index, value) in candidate_mean.iter().enumerate() {
                    lower[index] = lower[index].min(*value);
                    upper[index] = upper[index].max(*value);
                }
            }
            Some(antecedent_core::ResponseEnvelope {
                grid: Arc::clone(grid),
                dimension: *dimension,
                lower: Arc::from(lower),
                upper: Arc::from(upper),
            })
        }
        ResponseValue::Vector(_) | ResponseValue::Jacobian { .. } => {
            // A derivative vector or Jacobian: one coordinate per entry, no grid.
            let first_entries = coordinate_entries(first)?;
            let mut lower = first_entries.to_vec();
            let mut upper = first_entries.to_vec();
            for candidate in values.iter().skip(1) {
                let entries = coordinate_entries(candidate)?;
                if entries.len() != lower.len() || !same_coordinate_shape(first, candidate) {
                    return None;
                }
                for (index, value) in entries.iter().enumerate() {
                    lower[index] = lower[index].min(*value);
                    upper[index] = upper[index].max(*value);
                }
            }
            Some(antecedent_core::ResponseEnvelope {
                grid: Arc::from([]),
                dimension: 0,
                lower: Arc::from(lower),
                upper: Arc::from(upper),
            })
        }
        ResponseValue::Envelope(_) => None,
    }
}

/// The coordinates of a vector or Jacobian payload, `None` for any other shape.
fn coordinate_entries(value: &ResponseValue) -> Option<&[f64]> {
    match value {
        ResponseValue::Vector(entries) => Some(entries),
        ResponseValue::Jacobian { values, .. } => Some(values),
        _ => None,
    }
}

/// Whether two coordinate payloads are the same kind with the same matrix shape.
fn same_coordinate_shape(first: &ResponseValue, other: &ResponseValue) -> bool {
    match (first, other) {
        (ResponseValue::Vector(a), ResponseValue::Vector(b)) => a.len() == b.len(),
        (
            ResponseValue::Jacobian { outcomes: ao, treatments: at, .. },
            ResponseValue::Jacobian { outcomes: bo, treatments: bt, .. },
        ) => ao == bo && at == bt,
        _ => false,
    }
}

fn identified_set_is_singleton(envelope: &antecedent_core::ResponseEnvelope) -> bool {
    envelope.lower.len() == envelope.upper.len()
        && envelope
            .lower
            .iter()
            .zip(envelope.upper.iter())
            .all(|(lo, hi)| lo.to_bits() == hi.to_bits())
}

fn singleton_response_value(
    items: &[(f64, &ResponseValue)],
    envelope: &antecedent_core::ResponseEnvelope,
) -> Result<ResponseValue, CausalError> {
    let Some((_, first)) = items.first() else {
        return Err(CausalError::Compile {
            message: "class-aware response mix requires at least one identified case".into(),
        });
    };
    match first {
        ResponseValue::Scalar(_) => Ok(ResponseValue::Scalar(envelope.lower[0])),
        ResponseValue::Surface { .. } => Ok(ResponseValue::Surface {
            grid: Arc::clone(&envelope.grid),
            dimension: envelope.dimension,
            mean: Arc::clone(&envelope.lower),
        }),
        ResponseValue::Vector(_) => Ok(ResponseValue::Vector(Arc::clone(&envelope.lower))),
        ResponseValue::Jacobian { outcomes, treatments, .. } => Ok(ResponseValue::Jacobian {
            outcomes: *outcomes,
            treatments: *treatments,
            values: Arc::clone(&envelope.lower),
        }),
        ResponseValue::Envelope(_) => Err(CausalError::Compile {
            message: "class-aware response only publishes scalar, surface, vector and Jacobian \
                      identified sets"
                .into(),
        }),
    }
}

pub(in crate::analysis::execute) fn mix_class_responses(
    weighted: &[(u64, f64, antecedent_core::CausalResponse)],
    envelope_status: IdentificationStatus,
) -> Result<antecedent_core::CausalResponse, CausalError> {
    if weighted
        .iter()
        .all(|(_, _, r)| matches!(r.estimate, ResponseIdentification::Unidentified { .. }))
    {
        return Ok(weighted[0].2.clone());
    }
    let first = &weighted[0].2;
    let items: Vec<(f64, &ResponseValue)> = weighted
        .iter()
        .map(|(_, w, r)| {
            let value = match &r.estimate {
                ResponseIdentification::PointIdentified(v)
                | ResponseIdentification::PartiallyIdentified(v) => v,
                ResponseIdentification::GraphDependent(_)
                | ResponseIdentification::Unidentified { .. } => {
                    return Err(CausalError::Compile {
                        message: "class-aware response cannot mix unidentified case payloads"
                            .into(),
                    });
                }
            };
            Ok((*w, value))
        })
        .collect::<Result<_, _>>()?;
    let mixed_uncertainty = mix_response_uncertainty(
        &weighted.iter().map(|(_, w, r)| (*w, &r.uncertainty)).collect::<Vec<_>>(),
    );
    let mixed_support =
        mix_support_reports(&weighted.iter().map(|(_, _, r)| &r.support).collect::<Vec<_>>());
    let mut assumptions = first.assumptions.clone();
    for (_, _, response) in weighted.iter().skip(1) {
        assumptions.extend_unique(&response.assumptions.entries);
    }
    let (identification_status, estimate) =
        if matches!(envelope_status, IdentificationStatus::GraphDependent) {
            let atoms = weighted
                .iter()
                .filter_map(|(key, _, response)| {
                    response_identified_value(response).map(|value| (*key, value))
                })
                .collect();
            (IdentificationStatus::GraphDependent, ResponseIdentification::GraphDependent(atoms))
        } else {
            let envelope =
                response_envelope_from_weighted(weighted).ok_or_else(|| CausalError::Compile {
                    message:
                        "class-aware response could not construct a common identified envelope"
                            .into(),
                })?;
            if identified_set_is_singleton(&envelope)
                && matches!(
                    envelope_status,
                    IdentificationStatus::NonparametricallyIdentified
                        | IdentificationStatus::IdentifiedUnderParametricRestrictions
                        | IdentificationStatus::IdentifiedUnderPriorRestrictions
                )
            {
                let value = singleton_response_value(&items, &envelope)?;
                (envelope_status, ResponseIdentification::PointIdentified(value))
            } else {
                (
                    IdentificationStatus::PartiallyIdentified,
                    ResponseIdentification::PartiallyIdentified(ResponseValue::Envelope(envelope)),
                )
            }
        };
    Ok(antecedent_core::CausalResponse {
        estimand: first.estimand.clone(),
        identification_status,
        estimate,
        uncertainty: mixed_uncertainty,
        support: mixed_support,
        assumptions,
        provenance_id: Arc::clone(&first.provenance_id),
        horizon_identification: None,
        interaction_structurally_zero: first.interaction_structurally_zero,
    })
}

pub(in crate::analysis::execute) fn mix_response_values(
    items: &[(f64, &ResponseValue)],
) -> Result<ResponseValue, CausalError> {
    let Some((_, first)) = items.first() else {
        return Err(CausalError::Compile {
            message: "class-aware response mix requires at least one identified case".into(),
        });
    };
    match first {
        ResponseValue::Scalar(_) => {
            let mut acc = 0.0;
            let mut wsum = 0.0;
            for (w, value) in items {
                let ResponseValue::Scalar(x) = value else {
                    return Err(CausalError::Compile {
                        message: "class-aware response cases returned mixed payload shapes".into(),
                    });
                };
                acc += *w * *x;
                wsum += *w;
            }
            Ok(ResponseValue::Scalar(acc / wsum))
        }
        ResponseValue::Surface { grid, dimension, mean } => {
            let n = mean.len();
            let mut acc = vec![0.0; n];
            let mut wsum = 0.0;
            for (w, value) in items {
                let ResponseValue::Surface { grid: g, dimension: d, mean: m } = value else {
                    return Err(CausalError::Compile {
                        message: "class-aware response cases returned mixed payload shapes".into(),
                    });
                };
                if g.as_ref() != grid.as_ref() || *d != *dimension || m.len() != n {
                    return Err(CausalError::Compile {
                        message: "class-aware response cases returned incompatible grids".into(),
                    });
                }
                for (i, point) in m.iter().enumerate() {
                    acc[i] += *w * *point;
                }
                wsum += *w;
            }
            for point in &mut acc {
                *point /= wsum;
            }
            Ok(ResponseValue::Surface {
                grid: Arc::clone(grid),
                dimension: *dimension,
                mean: Arc::from(acc),
            })
        }
        ResponseValue::Vector(_) | ResponseValue::Jacobian { .. } => {
            let mut acc = vec![0.0; coordinate_entries(first).map_or(0, <[f64]>::len)];
            let mut wsum = 0.0;
            for (w, value) in items {
                let entries = coordinate_entries(value).filter(|entries| {
                    entries.len() == acc.len() && same_coordinate_shape(first, value)
                });
                let Some(entries) = entries else {
                    return Err(CausalError::Compile {
                        message: "class-aware response cases returned mixed payload shapes".into(),
                    });
                };
                for (slot, entry) in acc.iter_mut().zip(entries) {
                    *slot += *w * *entry;
                }
                wsum += *w;
            }
            for entry in &mut acc {
                *entry /= wsum;
            }
            Ok(match first {
                ResponseValue::Jacobian { outcomes, treatments, .. } => ResponseValue::Jacobian {
                    outcomes: *outcomes,
                    treatments: *treatments,
                    values: Arc::from(acc),
                },
                _ => ResponseValue::Vector(Arc::from(acc)),
            })
        }
        ResponseValue::Envelope(_) => Err(CausalError::Compile {
            message: "class-aware response only mixes scalar, surface, vector and Jacobian \
                      payloads"
                .into(),
        }),
    }
}

pub(super) fn mix_response_uncertainty(
    items: &[(f64, &ResponseUncertainty)],
) -> ResponseUncertainty {
    // A single atom retains its valid uncertainty. Averaging confidence limits
    // does not produce confidence limits for the weighted mean (the fits share
    // observations), nor quantiles of a distribution over graphs.
    if let [(weight, uncertainty)] = items {
        if weight.is_finite() && *weight > 0.0 {
            return (*uncertainty).clone();
        }
    }
    // Markov-equivalent completions of a fully identified class produce the same
    // adjustment, hence the same estimate and the same interval (for a Bayesian
    // response, the same retained draws). Their weighted posterior is that one
    // distribution, so the class interval is the shared interval — collapsing
    // agreeing completions invents nothing, unlike averaging distinct components.
    if let Some((first_weight, first)) = items.first() {
        if first_weight.is_finite()
            && *first_weight > 0.0
            && !matches!(**first, ResponseUncertainty::None)
            && items.iter().all(|(weight, u)| weight.is_finite() && *weight > 0.0 && *u == *first)
        {
            return (*first).clone();
        }
    }
    ResponseUncertainty::None
}

pub(in crate::analysis::execute) fn mix_support_reports(
    reports: &[&antecedent_core::SupportReport],
) -> antecedent_core::SupportReport {
    let first = reports[0];
    let mut status = first.status;
    let mut warnings = first.warnings.clone();
    for report in reports.iter().skip(1) {
        if support_rank(report.status) > support_rank(status) {
            status = report.status;
        }
        for warning in &report.warnings {
            if !warnings
                .iter()
                .any(|seen| seen.code == warning.code && seen.message == warning.message)
            {
                warnings.push(warning.clone());
            }
        }
    }
    // Per-cell status is the most conservative over the mixed reports whenever
    // they share one cell layout; a cell outside any report's support is outside
    // the mixture's.
    let point_status = match first.point_status.as_deref() {
        Some(cells)
            if reports
                .iter()
                .all(|r| r.point_status.as_deref().is_some_and(|p| p.len() == cells.len())) =>
        {
            Some(
                (0..cells.len())
                    .map(|cell| {
                        reports
                            .iter()
                            .filter_map(|r| r.point_status.as_deref().map(|p| p[cell]))
                            .max_by_key(|status| support_rank(*status))
                            .unwrap_or(cells[cell])
                    })
                    .collect::<Vec<_>>()
                    .into(),
            )
        }
        _ => first.point_status.clone(),
    };
    let mut mixed = antecedent_core::SupportReport {
        status,
        query_region: first.query_region.clone(),
        diagnostics: first.diagnostics.clone(),
        warnings,
        point_status,
    };
    // A simultaneous band belongs to one atom's surface; it stays on that atom and
    // never describes the mixed class report.
    antecedent_estimate::clear_simultaneous_band(&mut mixed);
    mixed
}

const fn support_rank(status: antecedent_core::SupportStatus) -> u8 {
    status.severity()
}

/// E-value / overlap on cell.aipw use the requested cell minus the all-zero control cell.
pub(super) fn cell_contrast_for_refute(
    estimate: &EffectEstimate,
    requested_arm: u32,
) -> EffectEstimate {
    let Some(table) = estimate.scores.table.as_ref() else {
        return estimate.clone();
    };
    let Ok((contrast, _)) = antecedent_estimate::cell_minus_control_contrast(table, requested_arm)
    else {
        return estimate.clone();
    };
    let mut out = estimate.clone();
    out.ate = contrast.value;
    out.uncertainty.analytic.se = contrast.se;
    out
}

pub(in crate::analysis::execute) fn attach_response_influence(
    estimate: &mut EffectEstimate,
    scores: Option<&antecedent_estimate::ResponseInfluence>,
) -> Result<(), CausalError> {
    let Some(scores) = scores else {
        return Ok(());
    };
    if scores.columns.is_empty() || scores.columns.iter().any(|c| c.len() < 2) {
        return Ok(());
    }
    let cols: Vec<&[f64]> = scores.columns.iter().map(Vec::as_slice).collect();
    match antecedent_estimate::joint_influence_covariance(&cols, None) {
        Ok(cov) => {
            // Column 0's SE describes the reported scalar only. A surface response publishes no
            // scalar (`ate` is NaN), so a standard error here would sit beside no estimate and
            // describe one unnamed grid cell; the joint covariance below carries every cell.
            if estimate.se_analytic().is_nan() && estimate.ate.is_finite() {
                estimate.uncertainty.analytic.se = cov.se(0);
            }
            estimate.joint.covariance = Some(cov);
        }
        Err(err) if cols.len() > 1 => {
            return Err(CausalError::from(err));
        }
        Err(_) => {}
    }
    estimate.scores.influence = scores.columns.first().map(|c| Arc::from(c.as_slice()));
    Ok(())
}

/// SE of the frozen-weight mixture of scalar atom responses from the joint
/// influence of the atoms on the shared data rows. Each atom's influence is
/// placed on its own complete-case rows ([`mix_response_influences`]), so atoms
/// that drop different rows are aligned by row index, not by position.
pub(in crate::analysis::execute) fn mixed_response_scalar_se(
    atoms: &[(f64, antecedent_estimate::ResponseInfluence)],
    full_n: usize,
) -> Option<f64> {
    let mixed = mix_response_influences(atoms, full_n)?;
    let column = mixed.columns.first()?;
    antecedent_estimate::joint_influence_covariance(&[column.as_slice()], None)
        .ok()
        .map(|cov| cov.se(0))
        .filter(|se| se.is_finite())
}

/// The refusals every graph-posterior response shares (fresh run and prepared
/// handle): a static response on complete observations, and one intervention
/// coordinate for [`ResponseFunctional::InterventionResponse`].
pub(crate) fn graph_posterior_response_supported(query: &ResponseQuery) -> Result<(), CausalError> {
    if query.temporal.is_some() || query.observation != ObservationSpec::Complete {
        return Err(CausalError::Unsupported {
            message: "static graph-posterior response requires complete observations",
        });
    }
    if matches!(
        &query.functional,
        ResponseFunctional::InterventionResponse { interventions, .. }
            if interventions.len() != 1
    ) {
        return Err(CausalError::Unsupported {
            message: "graph-posterior InterventionResponse currently requires one intervention coordinate",
        });
    }
    Ok(())
}

/// The coordinates of a scalar, vector or Jacobian payload.
fn derivative_coordinates(value: &ResponseValue) -> Option<Vec<f64>> {
    match value {
        ResponseValue::Scalar(x) => Some(vec![*x]),
        ResponseValue::Vector(values) | ResponseValue::Jacobian { values, .. } => {
            Some(values.to_vec())
        }
        _ => None,
    }
}

/// Every atom's influence columns on the full row set, centred over the atom's own rows and
/// scaled by `full_n / rows` so a missing row contributes zero (the alignment
/// [`mix_response_influences`] mixes with). `aligned[atom][column]`.
fn aligned_atom_columns(
    atoms: &[(f64, antecedent_estimate::ResponseInfluence)],
    full_n: usize,
) -> Option<Vec<Vec<Vec<f64>>>> {
    if atoms.is_empty() || full_n < 2 {
        return None;
    }
    let n_cols = atoms[0].1.columns.len();
    let mut aligned = Vec::with_capacity(atoms.len());
    for (_, scores) in atoms {
        let n = scores.row_index.len();
        if n < 2 || scores.columns.len() != n_cols || scores.columns.iter().any(|c| c.len() != n) {
            return None;
        }
        let mut columns = Vec::with_capacity(n_cols);
        for col in &scores.columns {
            let center = col.iter().sum::<f64>() / n as f64;
            let mut full = vec![0.0; full_n];
            for (&row, &v) in scores.row_index.iter().zip(col) {
                *full.get_mut(usize::try_from(row.get()).ok()?)? =
                    (v - center) * full_n as f64 / n as f64;
            }
            columns.push(full);
        }
        aligned.push(columns);
    }
    Some(aligned)
}

/// The sampling band of a derivative whose completions disagree: per coordinate, the
/// Imbens-Manski interval of the completions' identified set with standard deviations from
/// their joint row influence. One coordinate publishes a scalar interval (and the interval
/// itself for the structural mixture); several publish a pointwise band.
pub(super) fn derivative_identified_set_band(
    weighted: &[(u64, f64, antecedent_core::CausalResponse)],
    atom_scores: &[(f64, antecedent_estimate::ResponseInfluence)],
    full_n: usize,
    level: f64,
) -> Option<(ResponseUncertainty, Option<antecedent_estimate::IdentifiedSetInterval>)> {
    if weighted.len() < 2 || atom_scores.len() != weighted.len() {
        return None;
    }
    let coordinates: Vec<Vec<f64>> = weighted
        .iter()
        .map(|(_, _, response)| {
            response_identified_value(response).as_ref().and_then(derivative_coordinates)
        })
        .collect::<Option<_>>()?;
    let dim = coordinates[0].len();
    if coordinates.iter().any(|c| c.len() != dim) {
        return None;
    }
    let aligned = aligned_atom_columns(atom_scores, full_n)?;
    if aligned.iter().any(|columns| columns.len() != dim) {
        return None;
    }
    let mut lower = Vec::with_capacity(dim);
    let mut upper = Vec::with_capacity(dim);
    let mut intervals = Vec::with_capacity(dim);
    for j in 0..dim {
        let points: Vec<f64> = coordinates.iter().map(|c| c[j]).collect();
        let columns: Vec<&[f64]> = aligned.iter().map(|atom| atom[j].as_slice()).collect();
        let interval = antecedent_estimate::imbens_manski_influence(&points, &columns, level)?;
        lower.push(interval.lower);
        upper.push(interval.upper);
        intervals.push(interval);
    }
    let interpretation = antecedent_core::IntervalInterpretation::Confidence;
    if dim == 1 {
        let interval = intervals.remove(0);
        let standard_error = interval.lower_se.max(interval.upper_se);
        Some((
            ResponseUncertainty::Scalar {
                standard_error,
                level,
                lower: interval.lower,
                upper: interval.upper,
                interpretation,
                draws: None,
            },
            Some(interval),
        ))
    } else {
        Some((
            ResponseUncertainty::PointwiseBand {
                level,
                lower: lower.into(),
                upper: upper.into(),
                interpretation,
                draws: None,
            },
            None,
        ))
    }
}

pub(in crate::analysis::execute) fn mix_response_influences(
    atoms: &[(f64, antecedent_estimate::ResponseInfluence)],
    full_n: usize,
) -> Option<antecedent_estimate::ResponseInfluence> {
    if atoms.is_empty() {
        return None;
    }
    let n_cols = atoms[0].1.columns.len();
    if full_n < 2 || n_cols == 0 {
        return None;
    }
    let mut aligned = Vec::with_capacity(atoms.len());
    for (_, scores) in atoms {
        let n = scores.row_index.len();
        if n < 2 || scores.columns.len() != n_cols || scores.columns.iter().any(|c| c.len() != n) {
            return None;
        }
        let mut seen = vec![false; full_n];
        for &row in scores.row_index.iter() {
            let slot = seen.get_mut(usize::try_from(row.get()).expect("row id"))?;
            if *slot {
                return None;
            }
            *slot = true;
        }
        let mut columns = Vec::new();
        for col in &scores.columns {
            let center = col.iter().sum::<f64>() / n as f64;
            let mut full = vec![0.0; full_n];
            for (&row, &v) in scores.row_index.iter().zip(col) {
                full[usize::try_from(row.get()).expect("row id")] =
                    (v - center) * full_n as f64 / n as f64;
            }
            columns.push(full);
        }
        aligned.push(columns);
    }
    let weights: Vec<f64> = atoms.iter().map(|(w, _)| *w).collect();
    let mut columns = Vec::with_capacity(n_cols);
    for j in 0..n_cols {
        let refs: Vec<&[f64]> = aligned.iter().map(|cols| cols[j].as_slice()).collect();
        columns.push(antecedent_estimate::frozen_weight_mixture_scores(&refs, &weights).ok()?);
    }
    Some(antecedent_estimate::ResponseInfluence {
        columns,
        row_index: (0..full_n)
            .map(|i| u64::try_from(i).map(antecedent_data::GlobalId::new))
            .collect::<Result<Vec<_>, _>>()
            .ok()?
            .into(),
    })
}
