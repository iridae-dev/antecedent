use super::{
    Arc, CompiledDesign, DesignColumnMap, DesignColumnRole, EstimationError, TimeSeriesData,
    VariableId,
};

/// Root-weight every mechanism design by the sampling weight of the observation its window
/// ends at (window `r` ends at series time `max_lag + r`), so each mechanism's least-squares
/// solve is the weighted fit, the contrast influence is the weighted score and a row-block
/// resample carries each window's weight with it. The weights are normalized to mean one over
/// the windows of positive weight; a window of weight zero stays in sequence and contributes
/// no score.
///
/// Returns `None` for an unweighted series.
///
/// # Errors
///
/// `route_unsupported` when `unweighted_only` (a series with gaps, an effect modifier or a
/// level, none of which has a weighted construction); `invalid` for weights that do not cover
/// the windows or leave none positive.
pub(super) fn weight_sequential_windows(
    data: &TimeSeriesData,
    designs: &mut [Option<CompiledDesign>],
    max_lag: usize,
    windows: usize,
    unweighted_only: bool,
) -> Result<Option<crate::sampling_weights::SamplingWeightRecord>, EstimationError> {
    let Some(weights) = data.storage().weights() else {
        return Ok(None);
    };
    if unweighted_only {
        return Err(crate::sampling_weights::unsupported_route(
            "sampling weights are read by the sustained contrast on consecutive lag windows; a \
             series with gaps, an effect modifier or a level has no weighted construction",
        ));
    }
    let Some(window_weights) = weights.get(max_lag..max_lag + windows) else {
        return Err(crate::sampling_weights::invalid(
            "the series' sampling weights do not cover its lag-aligned windows",
        ));
    };
    if let Some((window, value)) =
        window_weights.iter().copied().enumerate().find(|(_, w)| !w.is_finite() || *w < 0.0)
    {
        return Err(crate::sampling_weights::invalid(&format!(
            "window {window} has sampling weight {value}; weights must be finite and non-negative"
        )));
    }
    let positive: Vec<f64> = window_weights.iter().copied().filter(|w| *w > 0.0).collect();
    if positive.is_empty() {
        return Err(crate::sampling_weights::invalid(
            "no lag window has a positive sampling weight",
        ));
    }
    let retained = crate::sampling_weights::RetainedSamplingWeights::from_positive(
        &positive,
        (window_weights.len() - positive.len()) as u64,
    );
    let scale = positive.len() as f64 / retained.record.weight_sum;
    let normalized: Vec<f64> = window_weights.iter().map(|w| w * scale).collect();
    for slot in designs.iter_mut() {
        if let Some(design) = slot.take() {
            *slot = Some(crate::sampling_weights::root_weight_design(design, &normalized));
        }
    }
    Ok(Some(retained.record))
}

/// Append modifier main effects and intervened-parent interactions.
///
/// Parent slopes stay in columns `1..=parents`. Each intervened parent then
/// gains one `parent × modifier` column per modifier, after any modifier that
/// is not already a parent.
pub(super) fn extend_mechanism_design(
    design: CompiledDesign,
    parent_values: &[&[f64]],
    intervened: &[bool],
    modifier_values: &[&[f64]],
    modifier_ids: &[VariableId],
    modifier_is_parent: &[bool],
) -> Result<(CompiledDesign, Vec<Vec<usize>>), EstimationError> {
    let mut slots = vec![Vec::new(); parent_values.len()];
    if modifier_values.is_empty() || !intervened.iter().any(|flag| *flag) {
        return Ok((design, slots));
    }
    let mut extras = Vec::new();
    let mut roles = Vec::new();
    for (values, (&id, &already)) in
        modifier_values.iter().zip(modifier_ids.iter().zip(modifier_is_parent))
    {
        if !already {
            extras.push((*values).to_vec());
            roles.push(id);
        }
    }
    let base = design.ncols;
    let n_main = extras.len();
    let mut products = 0usize;
    for (modifier, values) in modifier_values.iter().enumerate() {
        for (parent, (column, &is_intervened)) in parent_values.iter().zip(intervened).enumerate() {
            if !is_intervened {
                continue;
            }
            extras.push(column.iter().zip(*values).map(|(left, right)| left * right).collect());
            roles.push(modifier_ids[modifier]);
            slots[parent].push(base + n_main + products);
            products += 1;
        }
    }
    Ok((append_design_columns(design, &extras, &roles)?, slots))
}

fn append_design_columns(
    mut design: CompiledDesign,
    extras: &[Vec<f64>],
    roles: &[VariableId],
) -> Result<CompiledDesign, EstimationError> {
    if extras.is_empty() {
        return Ok(design);
    }
    let mut matrix = design.matrix.to_vec();
    let mut column_roles = Vec::with_capacity(design.ncols);
    for index in 0..design.ncols {
        column_roles.push(
            design
                .columns
                .role(index)
                .ok_or_else(|| EstimationError::stats_msg("design column role is missing"))?,
        );
    }
    for (extra, &role) in extras.iter().zip(roles) {
        matrix.extend_from_slice(extra);
        column_roles.push(DesignColumnRole::Covariate(role));
    }
    design.ncols = column_roles.len();
    design.matrix = Arc::from(matrix);
    design.columns = DesignColumnMap::from_roles(column_roles);
    Ok(design)
}
